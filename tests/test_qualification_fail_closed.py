"""Phase 4: Expert qualification fail-closed + finiteness guard tests."""
import math
import os

import pytest
import torch
import torch.nn as nn

from experiments.qualification import (
    ExpertQualification,
    ExpertQualificationPipeline,
    InvalidExperiment,
    QualificationError,
)


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, **kwargs):
        max_length = kwargs.get("max_length", 64)
        if isinstance(texts, str):
            texts = [texts]
        all_ids = []
        all_masks = []
        for text in texts:
            chars = list(text[:max_length])
            ids = [ord(c) % 100 for c in chars]
            mask = [1] * len(ids)
            pad = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad
            mask = mask + [0] * pad
            all_ids.append(ids)
            all_masks.append(mask)
        return {
            "input_ids": torch.tensor(all_ids),
            "attention_mask": torch.tensor(all_masks),
        }


class _StubModel(nn.Module):
    """Tiny model whose NLL is controlled by a target_loss parameter.

    Produces logits that yield cross-entropy loss ~= target_loss by
    setting the correct-next-token logit to a bias value and all others
    to 0. With bias b, loss = ln(1 + (vocab-1)*exp(-b)).
    """

    def __init__(self, target_loss: float = 2.0, vocab=100, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.target_loss = target_loss
        self.vocab = vocab
        self.config = type("Cfg", (), {"vocab_size": vocab})()
        # Solve bias from target_loss: loss = ln(1 + (V-1)*exp(-bias))
        # => exp(loss) = 1 + (V-1)*exp(-bias)
        # => exp(-bias) = (exp(loss)-1)/(V-1)
        # => bias = -ln((exp(loss)-1)/(V-1))
        if target_loss >= math.log(vocab):
            self._bias = 0.0  # uniform
        else:
            self._bias = -math.log((math.exp(target_loss) - 1) / (vocab - 1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, L = input_ids.shape
        logits = torch.zeros(B, L, self.vocab, device=input_ids.device)
        # Position i predicts token at position i+1
        if L > 1:
            next_ids = input_ids[:, 1:].clamp(min=0)  # [B, L-1]
            # Set logit for correct next token at position i
            row_idx = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(B, L - 1)
            col_idx = torch.arange(L - 1, device=input_ids.device).unsqueeze(0).expand(B, L - 1)
            logits[row_idx, col_idx, next_ids] = self._bias
        return type("Out", (), {"logits": logits})()


def _make_pipeline(base_loss=4.0, threshold=0.05):
    base = _StubModel(target_loss=base_loss)
    return ExpertQualificationPipeline(base, _StubTokenizer(), device="cpu",
                                       min_expert_improvement=threshold), base


def test_qualification_error_is_invalid_experiment_subclass():
    assert issubclass(QualificationError, InvalidExperiment)


def test_qualified_expert_passes():
    pipe, base = _make_pipeline(base_loss=4.0)  # high base NLL
    expert = _StubModel(target_loss=2.0)  # lower NLL -> improvement
    q = pipe.qualify_expert("test-expert", "main", expert, "math", ["1+1=2", "2+2=4"])
    assert q.passed, f"Expected pass; got rel_imp={q.relative_improvement}, reason={q.rejection_reason}"
    assert q.relative_improvement > 0


def test_unqualified_expert_fails():
    pipe, base = _make_pipeline(base_loss=2.0)
    expert = _StubModel(target_loss=4.0)  # worse NLL
    q = pipe.qualify_expert("bad-expert", "main", expert, "math", ["1+1=2"])
    assert not q.passed
    assert "Relative improvement" in (q.rejection_reason or "")


def test_non_finite_nll_disqualifies():
    pipe, base = _make_pipeline()
    # Expert with inf parameters -> inf NLL
    expert = _StubModel()
    with torch.no_grad():
        expert.embed.weight.fill_(float("inf"))
    q = pipe.qualify_expert("nan-expert", "main", expert, "math", ["1+1=2"])
    assert not q.passed
    assert "Non-finite" in (q.rejection_reason or "")


def test_validate_preflight_raises_qualification_error():
    pipe, _ = _make_pipeline()
    q = ExpertQualification(
        expert_name="bad", expert_revision="main", domain="math",
        base_nll=5.0, expert_nll=6.0, relative_improvement=-0.2,
        architecture_compatible=True, tokenizer_compatible=True,
        state_dict_compatible=True, passed=False,
        rejection_reason="worse than base",
    )
    with pytest.raises(InvalidExperiment):
        pipe.validate_preflight([q])


def test_topology_mismatch_rejected():
    pipe, base = _make_pipeline()
    # Different architecture
    expert = nn.Linear(8, 8)  # not an embedding model
    q = pipe.qualify_expert("mismatch", "main", expert, "math", ["1+1=2"])
    assert not q.passed
    assert not q.architecture_compatible or not q.state_dict_compatible
