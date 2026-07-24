"""Tests for v3 geometry modules: spectral, interactions, profiler, activations."""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.geometry.spectral import (
    compute_spectral_diagnostics, spectral_gate_passes, SpectralDiagnostics,
)
from daph_exfusion.geometry.interactions import (
    compute_fisher_interaction_matrix, compute_curvature_cosine,
    compute_euclidean_cosine, compute_interaction_matrix, InteractionMatrix,
)
from daph_exfusion.geometry.profiler import (
    profile_all_groups, profile_group, GroupGeometryProfile,
    compute_sign_conflict_rate, compute_principal_angles,
)
from daph_exfusion.geometry.activations import ActivationBank, ActivationStats


# =============================================================================
# Spectral diagnostics tests
# =============================================================================


class TestSpectralDiagnostics:
    def test_low_rank_matrix(self):
        """A rank-4 matrix should have rank_90 <= 4."""
        # Create a rank-4 matrix
        U = torch.randn(64, 4)
        V = torch.randn(4, 128)
        delta = U @ V  # rank 4
        diag = compute_spectral_diagnostics(delta)
        assert diag.rank_90 <= 4
        assert diag.rank_95 <= 4

    def test_full_rank_matrix(self):
        """A random matrix should have high rank."""
        torch.manual_seed(42)
        delta = torch.randn(32, 32)
        diag = compute_spectral_diagnostics(delta)
        assert diag.rank_90 > 10  # should be high

    def test_zero_matrix(self):
        """Zero matrix should have all-zero ranks."""
        delta = torch.zeros(10, 10)
        diag = compute_spectral_diagnostics(delta)
        assert diag.rank_50 == 0
        assert diag.rank_90 == 0
        assert diag.total_energy == 0.0

    def test_spectral_gate_passes_for_low_rank(self):
        """Low-rank matrix should pass the spectral gate."""
        U = torch.randn(64, 2)
        V = torch.randn(2, 128)
        delta = U @ V
        assert spectral_gate_passes(delta, threshold=0.1)

    def test_spectral_gate_fails_for_full_rank(self):
        """Full-rank matrix should fail the spectral gate."""
        torch.manual_seed(42)
        delta = torch.randn(32, 32)
        assert not spectral_gate_passes(delta, threshold=0.1)

    def test_effective_rank(self):
        """Effective rank should be between 1 and min(m, n)."""
        torch.manual_seed(42)
        delta = torch.randn(20, 15)
        diag = compute_spectral_diagnostics(delta)
        assert 1 <= diag.effective_rank <= 15


# =============================================================================
# Fisher interaction matrix tests
# =============================================================================


class TestFisherInteraction:
    def test_interaction_matrix_symmetric(self):
        """G_ij = Δᵢᵀ F₀ Δⱼ should be symmetric."""
        torch.manual_seed(42)
        tv1 = {"w": torch.randn(10)}
        tv2 = {"w": torch.randn(10)}
        base_fisher = {"w": torch.rand(10) + 0.1}

        G = compute_fisher_interaction_matrix([tv1, tv2], base_fisher)
        assert G.shape == (2, 2)
        assert torch.allclose(G, G.t(), atol=1e-5)

    def test_curvature_cosine_diagonal_is_one(self):
        """C^F_ii = 1 (cosine with self)."""
        torch.manual_seed(42)
        tv1 = {"w": torch.randn(10)}
        tv2 = {"w": torch.randn(10)}
        base_fisher = {"w": torch.rand(10) + 0.1}

        C = compute_curvature_cosine([tv1, tv2], base_fisher)
        assert torch.allclose(C.diagonal(), torch.ones(2), atol=1e-4)

    def test_curvature_cosine_in_range(self):
        """C^F values should be in [-1, 1]."""
        torch.manual_seed(42)
        tv1 = {"w": torch.randn(10)}
        tv2 = {"w": torch.randn(10)}
        base_fisher = {"w": torch.rand(10) + 0.1}

        C = compute_curvature_cosine([tv1, tv2], base_fisher)
        assert (C >= -1.0 - 1e-5).all()
        assert (C <= 1.0 + 1e-5).all()

    def test_euclidean_cosine_self_is_one(self):
        """Euclidean cosine of a vector with itself is 1."""
        torch.manual_seed(42)
        tv1 = {"w": torch.randn(10)}
        C = compute_euclidean_cosine([tv1])
        assert torch.allclose(C, torch.ones(1), atol=1e-4)

    def test_interaction_matrix_complete(self):
        """compute_interaction_matrix returns all components."""
        torch.manual_seed(42)
        tv1 = {"w": torch.randn(10)}
        tv2 = {"w": torch.randn(10)}
        base_fisher = {"w": torch.rand(10) + 0.1}

        interaction = compute_interaction_matrix([tv1, tv2], base_fisher)
        assert interaction.n_experts == 2
        assert interaction.G.shape == (2, 2)
        assert interaction.C_fisher.shape == (2, 2)
        assert interaction.C_euclidean.shape == (2, 2)

    def test_interaction_type_classification(self):
        """Interaction types should be classifiable."""
        torch.manual_seed(42)
        # Aligned experts
        tv1 = {"w": torch.randn(10)}
        tv2 = {"w": tv1["w"] * 0.9}  # nearly aligned
        base_fisher = {"w": torch.ones(10)}

        interaction = compute_interaction_matrix([tv1, tv2], base_fisher)
        itype = interaction.interaction_type(0, 1)
        assert itype in ("aligned_aligned", "moderate_interaction")


# =============================================================================
# Geometry profiler tests
# =============================================================================


class TestGeometryProfiler:
    def test_profile_all_groups(self):
        """Profile should classify parameters into groups."""
        torch.manual_seed(42)
        tv1 = {
            "model.embed_tokens.weight": torch.randn(10, 8),
            "model.layers.0.attn.q_proj.weight": torch.randn(8, 8),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
            "model.layers.0.input_layernorm.weight": torch.randn(8),
        }
        tv2 = {
            "model.embed_tokens.weight": torch.randn(10, 8),
            "model.layers.0.attn.q_proj.weight": torch.randn(8, 8),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
            "model.layers.0.input_layernorm.weight": torch.randn(8),
        }

        profiles = profile_all_groups([tv1, tv2], num_layers=1)
        assert len(profiles) > 0
        # Should have at least embeddings, attention, ffn, normalization
        group_names = set(profiles.keys())
        assert "embeddings" in group_names or "normalization" in group_names

    def test_sign_conflict_rate(self):
        """Sign conflict rate should be 0 for aligned, >0 for conflicting."""
        d1 = torch.tensor([1.0, 2.0, -3.0, 4.0])
        d2 = torch.tensor([1.0, -2.0, -3.0, 4.0])
        rate = compute_sign_conflict_rate([d1, d2])
        assert 0 < rate < 1  # some conflict

    def test_sign_conflict_rate_aligned(self):
        """Aligned deltas should have 0 conflict."""
        d1 = torch.tensor([1.0, 2.0, 3.0])
        d2 = torch.tensor([2.0, 3.0, 4.0])
        rate = compute_sign_conflict_rate([d1, d2])
        assert rate == 0.0

    def test_profile_to_dict(self):
        """Profile should be serializable to dict."""
        torch.manual_seed(42)
        tv1 = {"model.layers.0.attn.q_proj.weight": torch.randn(8, 8)}
        tv2 = {"model.layers.0.attn.q_proj.weight": torch.randn(8, 8)}
        profiles = profile_all_groups([tv1, tv2], num_layers=1)
        for name, profile in profiles.items():
            d = profile.to_dict()
            assert "group_name" in d
            assert "n_experts" in d


# =============================================================================
# ActivationBank tests
# =============================================================================


class TestActivationBank:
    def test_activation_bank_collect(self):
        """ActivationBank should collect covariance from linear layers."""
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        data = [
            {"input_ids": torch.randn(1, 8), "attention_mask": torch.ones(1, 1)},
            {"input_ids": torch.randn(1, 8), "attention_mask": torch.ones(1, 1)},
        ]

        bank = ActivationBank(mode="diagonal", max_samples=2)
        # The model doesn't have forward(input_ids=...) so we need a custom forward
        def forward_fn(m, x, attention_mask=None):
            return m(x)

        stats = bank.collect(model, "test_expert", data, forward_fn=forward_fn)
        # Should have collected something for the linear layers
        assert len(stats) > 0

    def test_activation_bank_to_dict(self):
        """ActivationBank should convert to dict format for merge_regmean."""
        bank = ActivationBank(mode="diagonal")
        # Manually add some data
        bank._bank = {
            "expert_0": {
                "layer.weight": ActivationStats(
                    sample_count=10, token_count=10, input_dim=8,
                    covariance=torch.rand(8),
                    rank_approximation=8, mode="diagonal",
                )
            }
        }
        d = bank.to_dict()
        assert "expert_0" in d
        assert "layer.weight" in d["expert_0"]
        assert d["expert_0"]["layer.weight"].shape == (8,)
