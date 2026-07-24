"""Phase 1 (P0) acceptance tests: sparse Mamba recurrent semantics.

These tests verify the repaired SSM routing path:
  - no cross-batch state leakage
  - matches a masked dense reference
  - preserves temporal gap semantics
  - batch permutation independence
  - inactive tokens do not inject input

A failure here is a P0 blocker for release v2.4.0-correctness.
"""
import pytest
import torch

from daph_hybrid_exfusion_v2_3 import (
    DAPHConfig,
    DAPHHybridDecoderLayer,
    MemoryBankExFusionMamba,
    SelectiveSSM,
    PointwiseSparseDispatch,
    SequencePathExecutor,
)


def _make_layer(hidden=8, state=4, num_experts=2, num_paths=4):
    cfg = DAPHConfig(
        hidden_size=hidden,
        intermediate_size=16,
        num_attention_heads=4,
        state_size=state,
        num_experts=num_experts,
        num_paths=num_paths,
        routing_mode="hard",
        ssm_bypass_decay=0.0,
    )
    torch.manual_seed(0)
    return DAPHHybridDecoderLayer(cfg), cfg


def test_sparse_mamba_no_cross_batch_leakage():
    """Changing sample b=0 must not change output for b=1."""
    layer, _ = _make_layer()
    layer.eval()

    B, L, H = 2, 6, 8
    torch.manual_seed(123)
    x = torch.randn(B, L, H)
    mamba_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        out_before, _ = layer(
            x, attention_mask=None, use_cache=False, mamba_state=None
        )

        x_perturbed = x.clone()
        x_perturbed[0] += torch.randn(L, H) * 10.0
        out_after, _ = layer(
            x_perturbed, attention_mask=None, use_cache=False, mamba_state=None
        )

    # Sample 1 must be byte-identical (within fp tolerance)
    delta = (out_before[1] - out_after[1]).abs().max().item()
    assert delta == pytest.approx(0.0, abs=1e-5), (
        f"Cross-batch leakage detected: max_abs(out_before[1]-out_after[1])={delta}"
    )


def test_sparse_mamba_matches_masked_dense_reference():
    """Hard-routed partial-mask run must equal a full-sequence run with the
    same mask applied inside the SSM (no sparse compaction shortcut)."""
    ssm = SelectiveSSM(hidden_size=8, state_size=4)
    ssm.eval()
    torch.manual_seed(7)
    B, L, H = 2, 5, 8
    x = torch.randn(B, L, H)
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        # Repaired semantics: full sequence + mask into SSM
        out_full, _ = ssm(x, state=None, mask=mask, bypass_decay=0.0)

        # Reference: zero out inactive inputs AND mask dt to zero
        x_masked = x * mask.unsqueeze(-1)
        out_ref, _ = ssm(x_masked, state=None, mask=mask, bypass_decay=0.0)

    # The SSM output at active positions must match (inactive positions may
    # differ in output but state preservation is what matters; we test
    # active-position equivalence here)
    active = mask.bool().unsqueeze(-1).expand_as(out_full)
    diff = (out_full - out_ref).abs()
    diff_active = diff[active].max().item()
    assert diff_active == pytest.approx(0.0, abs=1e-4), (
        f"Active-position mismatch: {diff_active}"
    )


def test_sparse_mamba_preserves_temporal_gap_semantics():
    """With DECAY semantics (bypass_decay>0), an inactive gap must decay the
    state, not compress it away.

    With bypass_decay>0, the full-sequence run decays state across the gap.
    The illegal compressed run (gap tokens removed) skips the decay entirely.
    The two must therefore DIFFER at the first active token after the gap.

    Note: with freeze semantics (bypass_decay=0.0) compression is
    mathematically equivalent to preservation, so this test uses decay to
    detect the compression bug.
    """
    ssm = SelectiveSSM(hidden_size=8, state_size=4)
    ssm.eval()
    torch.manual_seed(11)
    B, L, H = 1, 6, 8
    x = torch.randn(B, L, H)
    mask_with_gap = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.float32)
    decay = 0.05  # nonzero decay so the gap is not free

    with torch.no_grad():
        # Repaired: full sequence + mask, state decays across the gap
        out_full, _ = ssm(x, state=None, mask=mask_with_gap, bypass_decay=decay)
        # Illegal compressed reference: gather only active tokens, no gap
        b_idx, t_idx = torch.where(mask_with_gap.bool())
        x_compressed = x[b_idx, t_idx].unsqueeze(0)  # [1, 4, H]
        out_compressed, _ = ssm(
            x_compressed, state=None, mask=None, bypass_decay=decay
        )

    # First active token AFTER the gap: full position 4 vs compressed position 2
    first_after_gap_full = out_full[0, 4]
    first_after_gap_compressed = out_compressed[0, 2]
    diff = (first_after_gap_full - first_after_gap_compressed).abs().max().item()
    assert diff > 1e-4, (
        f"Temporal gap was compressed: full-run first-post-gap output matches "
        f"compressed-run output (diff={diff}); decay across gap was skipped."
    )


def test_mamba_batch_permutation_independence():
    """Permuting the batch dimension and then unpermuting the output must
    reproduce the original output exactly."""
    ssm = SelectiveSSM(hidden_size=8, state_size=4)
    ssm.eval()
    torch.manual_seed(99)
    B, L, H = 3, 5, 8
    x = torch.randn(B, L, H)
    perm = torch.tensor([2, 0, 1])
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        out_orig, _ = ssm(x, state=None, mask=None, bypass_decay=0.0)
        out_perm, _ = ssm(x[perm], state=None, mask=None, bypass_decay=0.0)
        out_restored = out_perm[inv_perm]

    assert torch.allclose(out_orig, out_restored, atol=1e-5), (
        "SSM output is not invariant under batch permutation"
    )


def test_mamba_inactive_tokens_do_not_inject_input():
    """With bypass_decay=0.0, changing the input at inactive positions must
    not change the output at any active position later in the sequence."""
    ssm = SelectiveSSM(hidden_size=8, state_size=4)
    ssm.eval()
    torch.manual_seed(42)
    B, L, H = 1, 6, 8
    x = torch.randn(B, L, H)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.float32)

    with torch.no_grad():
        out_a, _ = ssm(x, state=None, mask=mask, bypass_decay=0.0)
        x_perturbed = x.clone()
        x_perturbed[0, 2:4] += torch.randn(2, H) * 100.0  # perturb inactive
        out_b, _ = ssm(x_perturbed, state=None, mask=mask, bypass_decay=0.0)

    active = mask.bool().unsqueeze(-1).expand_as(out_a)
    diff = (out_a - out_b).abs()
    diff_active = diff[active].max().item()
    assert diff_active == pytest.approx(0.0, abs=1e-4), (
        f"Inactive tokens injected input into active positions: {diff_active}"
    )


def test_pointwise_dispatch_is_not_sequence_executor():
    """Structural contract: PointwiseSparseDispatch and SequencePathExecutor
    are distinct classes and the sequence executor rejects pointwise ops."""
    assert PointwiseSparseDispatch is not SequencePathExecutor
    # SequencePathExecutor must reject pointwise operators
    with pytest.raises(ValueError):
        SequencePathExecutor.validate_operator("ffn")
    # and accept sequence operators
    SequencePathExecutor.validate_operator("mamba")
    SequencePathExecutor.validate_operator("attention")
