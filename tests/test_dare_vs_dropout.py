"""Phase 8: DARE vs delta-dropout distinction + expectation test."""
import pytest
import torch

from daph_hybrid_exfusion_v2_3 import (
    apply_dare_preprocessing,
    apply_delta_dropout,
)


def test_dare_preserves_expected_delta():
    """E[tilde_Delta] ≈ Delta for DARE (with rescaling by 1/(1-p))."""
    torch.manual_seed(42)
    delta = {"w": torch.randn(1000) * 5.0 + 3.0}
    task_vectors = [delta]
    p = 0.3
    num_trials = 500

    accumulated = torch.zeros(1000)
    for _ in range(num_trials):
        processed, _ = apply_dare_preprocessing(
            task_vectors, dare_base_p=p, rescale_deltas=True
        )
        accumulated += processed[0]["w"]

    empirical_mean = accumulated / num_trials
    # E[tilde_Delta] should be close to Delta
    relative_error = (empirical_mean - delta["w"]).abs().mean().item() / delta["w"].abs().mean().item()
    assert relative_error < 0.05, (
        f"DARE does not preserve expected delta: relative_error={relative_error}"
    )


def test_delta_dropout_does_not_preserve_expected_delta():
    """E[tilde_Delta] = (1-p) * Delta for plain dropout (no rescaling)."""
    torch.manual_seed(42)
    delta = {"w": torch.randn(1000) * 5.0 + 3.0}
    task_vectors = [delta]
    p = 0.3
    num_trials = 500

    accumulated = torch.zeros(1000)
    for _ in range(num_trials):
        processed, _ = apply_delta_dropout(task_vectors, drop_probability=p)
        accumulated += processed[0]["w"]

    empirical_mean = accumulated / num_trials
    # E[tilde_Delta] should be approximately (1-p) * Delta
    expected = (1.0 - p) * delta["w"]
    relative_error = (empirical_mean - expected).abs().mean().item() / expected.abs().mean().item()
    assert relative_error < 0.05, (
        f"Delta dropout expectation wrong: relative_error={relative_error}"
    )

    # And it should NOT equal Delta (that's DARE's job)
    not_equal = (empirical_mean - delta["w"]).abs().mean().item() / delta["w"].abs().mean().item()
    assert not_equal > 0.1, (
        f"Delta dropout unexpectedly preserved expected delta (not_equal={not_equal})"
    )


def test_dare_default_is_rescaled():
    """The default for apply_dare_preprocessing must be rescale=True (DARE)."""
    import inspect

    sig = inspect.signature(apply_dare_preprocessing)
    assert sig.parameters["rescale_deltas"].default is True, (
        "DARE default must be rescale_deltas=True; False is delta dropout"
    )


def test_delta_dropout_default_is_not_rescaled():
    """apply_delta_dropout must NOT rescale."""
    torch.manual_seed(0)
    delta = {"w": torch.ones(100)}
    processed, _ = apply_delta_dropout([delta], drop_probability=0.5)
    # Without rescaling, kept values are still 1.0 (not 2.0)
    kept = processed[0]["w"][processed[0]["w"] != 0.0]
    assert torch.allclose(kept, torch.ones_like(kept))
