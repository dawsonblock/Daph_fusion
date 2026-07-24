"""Tests for v3 RegMean merge.

RegMean tests (required by plan):
    Synthetic linear system with known W₁, W₂, C₁, C₂.
    Assert implementation matches: W* = (W₁C₁ + W₂C₂)(C₁ + C₂ + ρI)⁻¹
    Also: identical experts → identical result, singular covariance → ridge stabilizes,
    block mode approximates full mode.
"""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.merge.types import MergeConfig, MergeMethod, RegMeanMode
from daph_exfusion.merge.regmean import (
    merge_regmean, regmean_solve, is_regmean_eligible,
)


# =============================================================================
# RegMean solver tests
# =============================================================================


class TestRegMeanSolve:
    def test_analytical_solution_full(self):
        """Verify W* = (W₁C₁ + W₂C₂)(C₁ + C₂ + ρI)⁻¹ for full mode."""
        torch.manual_seed(42)
        out_dim, in_dim = 4, 3

        W1 = torch.randn(out_dim, in_dim)
        W2 = torch.randn(out_dim, in_dim)
        C1 = torch.randn(in_dim, in_dim)
        C1 = C1 @ C1.t()  # Make SPD
        C2 = torch.randn(in_dim, in_dim)
        C2 = C2 @ C2.t()  # Make SPD
        rho = 0.01

        # Analytical solution
        numerator = W1 @ C1 + W2 @ C2
        denominator = C1 + C2 + rho * torch.eye(in_dim)
        W_star_analytical = numerator @ torch.linalg.inv(denominator)

        # RegMean solve
        W_star = regmean_solve(
            [W1, W2], [C1, C2], ridge=rho, mode=RegMeanMode.FULL
        )

        assert torch.allclose(W_star, W_star_analytical, atol=1e-4)

    def test_identical_experts_identical_result(self):
        """Identical experts with identical covariance → result equals the expert."""
        W = torch.randn(3, 4)
        C = torch.eye(4)

        W_star = regmean_solve([W, W], [C, C], ridge=1e-6, mode=RegMeanMode.FULL)
        assert torch.allclose(W_star, W, atol=1e-3)

    def test_singular_covariance_ridge_stabilizes(self):
        """Singular covariance should not crash — ridge stabilizes."""
        W1 = torch.randn(2, 3)
        W2 = torch.randn(2, 3)
        # Singular covariance (rank 1)
        C1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        C2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        # Should not crash
        W_star = regmean_solve([W1, W2], [C1, C2], ridge=0.1, mode=RegMeanMode.FULL)
        assert W_star.shape == (2, 3)
        assert torch.isfinite(W_star).all()

    def test_diagonal_mode_matches_full_for_diagonal_cov(self):
        """Diagonal mode should match full mode when covariance is diagonal."""
        torch.manual_seed(42)
        out_dim, in_dim = 3, 4

        W1 = torch.randn(out_dim, in_dim)
        W2 = torch.randn(out_dim, in_dim)
        c1 = torch.rand(in_dim) + 0.1
        c2 = torch.rand(in_dim) + 0.1
        C1 = torch.diag(c1)
        C2 = torch.diag(c2)
        rho = 0.01

        W_full = regmean_solve([W1, W2], [C1, C2], ridge=rho, mode=RegMeanMode.FULL)
        W_diag = regmean_solve([W1, W2], [c1, c2], ridge=rho, mode=RegMeanMode.DIAGONAL)

        assert torch.allclose(W_full, W_diag, atol=1e-4)

    def test_block_mode_approximates_full(self):
        """Block mode with large block size should approximate full mode."""
        torch.manual_seed(42)
        out_dim, in_dim = 4, 4

        W1 = torch.randn(out_dim, in_dim)
        W2 = torch.randn(out_dim, in_dim)
        C1 = torch.eye(in_dim)
        C2 = torch.eye(in_dim)
        rho = 0.01

        W_full = regmean_solve([W1, W2], [C1, C2], ridge=rho, mode=RegMeanMode.FULL)
        W_block = regmean_solve(
            [W1, W2], [C1, C2], ridge=rho, mode=RegMeanMode.BLOCK, block_size=in_dim
        )

        assert torch.allclose(W_full, W_block, atol=1e-4)


# =============================================================================
# RegMean eligibility tests
# =============================================================================


class TestRegMeanEligibility:
    def test_linear_weight_eligible(self):
        param = torch.randn(10, 5)
        assert is_regmean_eligible("model.layers.0.attn.q_proj.weight", param)

    def test_bias_not_eligible(self):
        param = torch.randn(10)
        assert not is_regmean_eligible("model.layers.0.attn.q_proj.bias", param)

    def test_norm_not_eligible(self):
        param = torch.randn(8)
        assert not is_regmean_eligible("model.layers.0.input_layernorm.weight", param)

    def test_embedding_not_eligible(self):
        param = torch.randn(100, 8)
        assert not is_regmean_eligible("model.embed_tokens.weight", param)

    def test_a_log_not_eligible(self):
        param = torch.randn(8)
        assert not is_regmean_eligible("model.layers.0.ssm.a_log", param)

    def test_1d_param_not_eligible(self):
        param = torch.randn(10)
        assert not is_regmean_eligible("some.param", param)


# =============================================================================
# RegMean merge integration tests
# =============================================================================


class TestRegMeanMerge:
    def test_regmean_merge_with_covariance(self):
        """RegMean merge with activation covariance should produce valid result."""
        base = nn.Linear(8, 4)
        expert1 = nn.Linear(8, 4)
        expert2 = nn.Linear(8, 4)
        with torch.no_grad():
            expert1.weight.copy_(base.weight + 0.5)
            expert2.weight.copy_(base.weight - 0.3)

        # Simple diagonal covariance
        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }

        config = MergeConfig(method=MergeMethod.REGMEAN, regmean_mode=RegMeanMode.DIAGONAL)
        result = merge_regmean(base, [expert1, expert2], config, activation_bank)

        assert result.method == "regmean"
        assert "REGMEAN" in result.trace.operators
        assert result.trace.activation_covariance_used
        assert not result.trace.fisher_used

    def test_regmean_falls_back_for_non_eligible(self):
        """Non-RegMean-eligible params should fall back to Task Arithmetic."""
        base = nn.Linear(8, 4)
        expert1 = nn.Linear(8, 4)
        expert2 = nn.Linear(8, 4)
        with torch.no_grad():
            expert1.weight.copy_(base.weight + 0.5)
            expert1.bias.copy_(base.bias + 0.1)
            expert2.weight.copy_(base.weight - 0.3)
            expert2.bias.copy_(base.bias - 0.05)

        activation_bank = {
            "expert_0": {"weight": torch.rand(8) + 0.1},
            "expert_1": {"weight": torch.rand(8) + 0.1},
        }

        config = MergeConfig(method=MergeMethod.REGMEAN, regmean_mode=RegMeanMode.DIAGONAL)
        result = merge_regmean(base, [expert1, expert2], config, activation_bank)

        # Bias should be merged via TA (fallback)
        assert "TASK_ARITHMETIC_FALLBACK" in result.trace.operators
