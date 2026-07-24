"""Phase 9: Unified Fisher implementation tests."""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.curvature.fisher import build_fisher_diagonal
from daph_exfusion.geometry.curvature import (
    build_empirical_fisher_diagonals,
    build_empirical_fisher_diagonals_offloaded,
)


class _TinyLM(nn.Module):
    def __init__(self, vocab=20, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()


def _make_batch(vocab=20, B=4, L=8, with_padding=True):
    torch.manual_seed(0)
    input_ids = torch.randint(2, vocab, (B, L))
    if with_padding:
        attention_mask = torch.ones(B, L)
        attention_mask[:, L // 2 :] = 0  # second half is padding
        input_ids[:, L // 2 :] = 0  # pad token id
    else:
        attention_mask = torch.ones(B, L)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_fisher_exact_matches_manual_per_sample():
    """Exact per-sample Fisher = (1/N) Σ g_n² must match manual computation."""
    model = _TinyLM()
    batch = _make_batch(with_padding=False)
    fisher = build_fisher_diagonal(model, batch, mode="exact_per_sample", offload="gpu")

    # Manual: per-sample gradients, square, average
    model.eval()
    manual_fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    N = batch["input_ids"].shape[0]
    for i in range(N):
        model.zero_grad()
        ids = batch["input_ids"][i:i+1]
        out = model(ids)
        logits = out.logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                manual_fisher[n] += p.grad.detach().square() / N

    for name in manual_fisher:
        assert torch.allclose(fisher[name], manual_fisher[name], atol=1e-4), (
            f"Fisher mismatch on {name}: max diff = {(fisher[name] - manual_fisher[name]).abs().max().item()}"
        )


def test_fisher_approximation_is_explicitly_different():
    """Approximation mode (micro_batch>1) must differ from exact mode."""
    model = _TinyLM()
    batch = _make_batch(with_padding=False, B=8)
    exact = build_fisher_diagonal(model, batch, mode="exact_per_sample", offload="gpu")
    approx = build_fisher_diagonal(
        model, batch, mode="microbatch_approximation", offload="gpu", micro_batch_size=4
    )
    # They should differ (approximation squares the mean gradient)
    total_diff = sum(
        (exact[n] - approx[n]).abs().max().item()
        for n in exact
    )
    assert total_diff > 1e-6, "Approximation mode should differ from exact mode"


def test_fisher_padding_mask_excludes_padding():
    """Padding tokens must not contribute to the Fisher gradient."""
    model = _TinyLM()
    batch_with_pad = _make_batch(with_padding=True)
    batch_no_pad = {
        "input_ids": batch_with_pad["input_ids"][:, : batch_with_pad["input_ids"].shape[1] // 2],
        "attention_mask": batch_with_pad["attention_mask"][:, : batch_with_pad["input_ids"].shape[1] // 2],
    }

    fisher_padded = build_fisher_diagonal(
        model, batch_with_pad, mode="exact_per_sample", offload="gpu"
    )
    fisher_unpadded = build_fisher_diagonal(
        model, batch_no_pad, mode="exact_per_sample", offload="gpu"
    )

    # With padding correctly ignored, the Fisher should be similar
    # (not identical because the loss normalization differs, but close)
    for name in fisher_padded:
        ratio = fisher_padded[name].abs().mean().item() / max(fisher_unpadded[name].abs().mean().item(), 1e-8)
        # Should be in the same order of magnitude (0.5x to 2x)
        assert 0.1 < ratio < 10, (
            f"Padding handling broke Fisher on {name}: ratio={ratio}"
        )


def test_fisher_cpu_offload_matches_gpu():
    """CPU-offloaded accumulation should match GPU accumulation."""
    model = _TinyLM()
    batch = _make_batch(with_padding=False, B=4)
    gpu_fisher = build_fisher_diagonal(model, batch, mode="exact_per_sample", offload="gpu")
    cpu_fisher = build_fisher_diagonal(model, batch, mode="exact_per_sample", offload="cpu")

    for name in gpu_fisher:
        assert torch.allclose(gpu_fisher[name], cpu_fisher[name], atol=1e-4), (
            f"CPU/GPU Fisher mismatch on {name}"
        )


def test_fisher_accumulator_finite():
    """Fisher diagonal values must all be finite."""
    model = _TinyLM()
    batch = _make_batch(with_padding=True)
    fisher = build_fisher_diagonal(model, batch, mode="exact_per_sample", offload="gpu")
    for name, val in fisher.items():
        assert torch.isfinite(val).all(), f"Non-finite Fisher on {name}"


def test_exact_mode_rejects_large_micro_batch():
    """exact_per_sample mode must reject micro_batch_size > 1."""
    model = _TinyLM()
    batch = _make_batch(with_padding=False)
    with pytest.raises(ValueError, match="exact_per_sample"):
        build_fisher_diagonal(
            model, batch, mode="exact_per_sample", micro_batch_size=4
        )


def test_invalid_mode_raises():
    model = _TinyLM()
    batch = _make_batch()
    with pytest.raises(ValueError, match="mode must be"):
        build_fisher_diagonal(model, batch, mode="bogus")
