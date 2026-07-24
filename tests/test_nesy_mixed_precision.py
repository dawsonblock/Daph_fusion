"""Phase 2: Mixed-precision NeSy router hardening tests.

Verifies that symbolic priors (BIAS_FORCE=1e5) do not overflow FP16/BF16
into inf/NaN after the clamp_symbolic_priors repair.
"""
import pytest
import torch

from daph_nesy_v1_0 import (
    BIAS_FORCE,
    BIAS_FORBID,
    NeSyMacroRouter,
    clamp_symbolic_priors,
)


def _make_router(num_paths=5, device="cpu", dtype=None):
    router = NeSyMacroRouter(32, num_paths=num_paths, granularity="token").to(device)
    if dtype is not None:
        router = router.to(dtype)
    router.eval()
    return router


def test_clamp_symbolic_priors_rejects_non_float_logits():
    int_logits = torch.zeros(2, 3, dtype=torch.int32)
    priors = torch.zeros(2, 3)
    with pytest.raises(TypeError):
        clamp_symbolic_priors(priors, int_logits)


def test_clamp_symbolic_priors_rejects_non_finite_logits():
    logits = torch.tensor([[1.0, float("inf"), 2.0]], dtype=torch.float32)
    priors = torch.zeros(1, 3)
    with pytest.raises(FloatingPointError):
        clamp_symbolic_priors(priors, logits)


def test_clamp_symbolic_priors_fp16_safe():
    """BIAS_FORCE=1e5 must be clamped below FP16 max (~65504)."""
    logits = torch.zeros(1, 3, dtype=torch.float16)
    priors = torch.full((1, 3), BIAS_FORCE, dtype=torch.float32)
    clamped = clamp_symbolic_priors(priors, logits)
    assert clamped.dtype == torch.float16
    assert torch.isfinite(clamped).all()
    # FP16 max/4 ~ 16384; requested 1e4 -> safe_limit = 1e4
    assert clamped.abs().max().item() <= 1.0e4


def test_clamp_symbolic_priors_bf16_safe():
    logits = torch.zeros(1, 3, dtype=torch.bfloat16)
    priors = torch.full((1, 3), BIAS_FORCE, dtype=torch.float32)
    clamped = clamp_symbolic_priors(priors, logits)
    assert clamped.dtype == torch.bfloat16
    assert torch.isfinite(clamped).all()


def test_nesy_router_fp16_all_finite():
    """Full router forward with FP16 + BIAS_FORCE priors must stay finite."""
    router = _make_router(dtype=torch.float16)
    B, L, H, P = 2, 8, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float16)
    priors = torch.full((B, L, P), BIAS_FORCE, dtype=torch.float32)
    priors[:, :, 2] = BIAS_FORBID  # forbid path 2
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    assert logits.dtype == torch.float16
    assert torch.isfinite(logits).all(), "FP16 router logits contain inf/NaN"


def test_nesy_router_bf16_all_finite():
    router = _make_router(dtype=torch.bfloat16)
    B, L, H, P = 2, 8, 32, 5
    x = torch.randn(B, L, H, dtype=torch.bfloat16)
    priors = torch.full((B, L, P), BIAS_FORCE, dtype=torch.float32)
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all(), "BF16 router logits contain inf/NaN"


def test_nesy_router_fp32_all_finite():
    router = _make_router()
    B, L, H, P = 2, 8, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float32)
    priors = torch.full((B, L, P), BIAS_FORCE, dtype=torch.float32)
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    assert torch.isfinite(logits).all()


def test_nesy_router_hard_prior_no_nan():
    """Even with both FORCE and FORBID, no NaN should appear."""
    router = _make_router(dtype=torch.float16)
    B, L, H, P = 1, 4, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float16)
    priors = torch.zeros(B, L, P, dtype=torch.float32)
    priors[:, :, 1] = BIAS_FORCE
    priors[:, :, 3] = BIAS_FORBID
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    assert torch.isfinite(logits).all()


def test_symbolic_prior_forces_expected_path_fp16():
    """A +1e4 forced path should dominate the softmax in FP16."""
    router = _make_router(dtype=torch.float16)
    B, L, H, P = 1, 1, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float16)
    priors = torch.zeros(B, L, P, dtype=torch.float32)
    priors[:, :, 2] = BIAS_FORCE  # force path 2
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    probs = torch.softmax(logits.float(), dim=-1)
    chosen = probs.argmax(dim=-1)
    assert chosen.item() == 2, f"Forced path 2 not selected; got {chosen.item()}"


def test_symbolic_prior_forbids_expected_path_fp16():
    """A -1e4 forbidden path should never be selected in FP16."""
    router = _make_router(dtype=torch.float16)
    B, L, H, P = 1, 1, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float16)
    priors = torch.zeros(B, L, P, dtype=torch.float32)
    priors[:, :, 0] = BIAS_FORBID  # forbid path 0
    with torch.no_grad():
        logits = router(x, symbolic_priors=priors)
    probs = torch.softmax(logits.float(), dim=-1)
    chosen = probs.argmax(dim=-1)
    assert chosen.item() != 0, f"Forbidden path 0 was selected; got {chosen.item()}"


def test_assert_numerics_raises_on_non_finite():
    router = _make_router()
    router.assert_numerics = True
    router.train()
    B, L, H, P = 1, 1, 32, 5
    x = torch.randn(B, L, H, dtype=torch.float32)
    # Patch super().forward to return inf logits
    import daph_nesy_v1_0 as mod

    orig = mod.PredictiveDifficultyMacroRouter.forward
    mod.PredictiveDifficultyMacroRouter.forward = lambda self, h, d: torch.full(
        (1, 1, 5), float("inf")
    )
    try:
        with pytest.raises(FloatingPointError):
            router(x, symbolic_priors=None)
    finally:
        mod.PredictiveDifficultyMacroRouter.forward = orig
