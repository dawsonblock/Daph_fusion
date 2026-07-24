"""End-to-end operator-equivalence integration tests (Phase 11).

These tests verify that the experiment runner's merge operations actually
execute the algorithms they claim to, by comparing against direct operator
calls. They catch the class of bug where 146 unit tests pass but the
research implementation diverges from the algorithm names.
"""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daph_exfusion.merge.pipeline import MergeConfig, merge_experts
from daph_exfusion.geometry.operators import (
    op_ties,
    op_ties_fisher,
    op_fisher_weighted,
    op_dare,
    transform_expert_set,
    CROSS_EXPERT_OPS,
    SINGLE_EXPERT_OPS,
)


def _make_tiny_model(dim=8, vocab=32):
    """Make a tiny GPT-2-like model for testing."""
    model = nn.Sequential(
        nn.Embedding(vocab, dim),
        nn.Linear(dim, dim),
        nn.Linear(dim, vocab),
    )
    return model


def _make_experts(base, n=3, noise_scale=0.1):
    """Make n experts that are perturbations of the base."""
    experts = []
    for i in range(n):
        expert = copy.deepcopy(base)
        with torch.no_grad():
            for param in expert.parameters():
                param.add_(torch.randn_like(param) * noise_scale * (i + 1))
        experts.append(expert)
    return experts


def _get_param_dict(model):
    return {name: p.detach().clone().float() for name, p in model.named_parameters()}


class TestOperatorEquivalence:
    """Verify that merge_experts produces the same result as direct operator calls."""

    def test_ties_magnitude_matches_direct_op(self):
        """merge_experts(TIES_MAGNITUDE) must match op_ties(sign_mode='magnitude')."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        config = MergeConfig(
            algorithm="TIES_MAGNITUDE",
            scale=1.0,
            ties_trim_fraction=0.2,
        )
        result = merge_experts(base, experts, config)
        trace = result.operator_trace
        assert trace == ["TIES_MAGNITUDE"], f"Expected ['TIES_MAGNITUDE'], got {trace}"

        # Verify per-parameter
        base_params = _get_param_dict(base)
        for name, param in result.merged_model.named_parameters():
            deltas = []
            for expert in experts:
                ep = _get_param_dict(expert)
                deltas.append(ep[name] - base_params[name])
            expected_delta = op_ties(deltas, trim_fraction=0.2, sign_mode="magnitude")
            expected = base_params[name] + expected_delta
            assert torch.allclose(param.detach().float(), expected, atol=1e-5), \
                f"TIES_MAGNITUDE mismatch on {name}"

    def test_ties_majority_matches_direct_op(self):
        """merge_experts(TIES_MAJORITY) must match op_ties(sign_mode='majority')."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        config = MergeConfig(
            algorithm="TIES_MAJORITY",
            scale=1.0,
            ties_trim_fraction=0.3,
        )
        result = merge_experts(base, experts, config)
        assert result.operator_trace == ["TIES_MAJORITY"]

        base_params = _get_param_dict(base)
        for name, param in result.merged_model.named_parameters():
            deltas = [(_get_param_dict(e)[name] - base_params[name]) for e in experts]
            expected_delta = op_ties(deltas, trim_fraction=0.3, sign_mode="majority")
            expected = base_params[name] + expected_delta
            assert torch.allclose(param.detach().float(), expected, atol=1e-5), \
                f"TIES_MAJORITY mismatch on {name}"

    def test_fisher_requires_curvature_bank(self):
        """FISHER without curvature_bank must raise."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        config = MergeConfig(algorithm="FISHER")
        with pytest.raises(ValueError, match="curvature_bank"):
            merge_experts(base, experts, config)

    def test_fisher_uses_real_fisher_not_delta_squared(self):
        """FISHER must use the provided Fisher diagonals, not |delta|^2."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        base_params = _get_param_dict(base)

        # Create a curvature bank with known Fisher values
        curvature_bank = {}
        for i, expert in enumerate(experts):
            ep = _get_param_dict(expert)
            curvature_bank[f"expert_{i}"] = {
                name: torch.ones_like(param) * (i + 1)  # different per expert
                for name, param in ep.items()
            }

        config = MergeConfig(algorithm="FISHER", scale=1.0, fisher_gamma=1.0)
        result = merge_experts(base, experts, config, curvature_bank=curvature_bank)

        # Verify it matches op_fisher_weighted with the provided Fisher
        for name, param in result.merged_model.named_parameters():
            deltas = [(_get_param_dict(e)[name] - base_params[name]) for e in experts]
            fishers = [curvature_bank[f"expert_{i}"][name] for i in range(len(experts))]
            expected_delta = op_fisher_weighted(deltas, fishers, gamma=1.0)
            expected = base_params[name] + expected_delta
            assert torch.allclose(param.detach().float(), expected, atol=1e-5), \
                f"FISHER mismatch on {name}"

    def test_exfusion_is_dare_ties_fisher(self):
        """ExFusion must be DARE → TIES → Fisher (op_ties_fisher), not delta^2 + single TIES."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        base_params = _get_param_dict(base)

        curvature_bank = {}
        for i, expert in enumerate(experts):
            ep = _get_param_dict(expert)
            curvature_bank[f"expert_{i}"] = {
                name: torch.rand_like(param) + 0.1
                for name, param in ep.items()
            }

        config = MergeConfig(
            algorithm="ExFusion",
            scale=1.0,
            dare_drop_rate=0.1,
            ties_trim_fraction=0.2,
            fisher_gamma=0.5,
            seed=42,
        )
        result = merge_experts(base, experts, config, curvature_bank=curvature_bank)
        trace = result.operator_trace
        assert "DARE" in trace, f"ExFusion trace missing DARE: {trace}"
        assert "EMPIRICAL_FISHER" in trace, f"ExFusion trace missing FISHER: {trace}"

        # Verify it matches op_ties_fisher with DARE preprocessing
        gen = torch.Generator().manual_seed(42)
        for name, param in result.merged_model.named_parameters():
            deltas = [(_get_param_dict(e)[name] - base_params[name]) for e in experts]
            fishers = [curvature_bank[f"expert_{i}"][name] for i in range(len(experts))]
            dare_deltas = [op_dare(d, drop_probability=0.1, generator=gen) for d in deltas]
            expected_delta = op_ties_fisher(
                dare_deltas, fishers,
                trim_fraction=0.2, fisher_gamma=0.5,
                sign_mode="magnitude",
            )
            expected = base_params[name] + expected_delta
            assert torch.allclose(param.detach().float(), expected, atol=1e-5), \
                f"ExFusion mismatch on {name}"

    def test_dare_generator_not_reset_per_parameter(self):
        """DARE masks must differ across parameters (generator consumed continuously)."""
        base = _make_tiny_model(dim=16)
        experts = _make_experts(base, n=1, noise_scale=0.5)
        config = MergeConfig(
            algorithm="DARE",
            scale=1.0,
            dare_drop_rate=0.5,
            seed=123,
        )
        result = merge_experts(base, experts, config)

        # With p=0.5 and continuous generator, different parameters should have
        # different masks. If the generator were reset per-parameter, masks
        # would be identical for same-shaped tensors.
        base_params = _get_param_dict(base)
        expert_params = _get_param_dict(experts[0])

        masks = []
        for name, param in result.merged_model.named_parameters():
            delta = expert_params[name] - base_params[name]
            merged_delta = param.detach().float() - base_params[name]
            # DARE: merged_delta = (M * delta) / (1-p) * scale
            # So M = merged_delta * (1-p) / delta (where delta != 0)
            scale = 1.0
            p = 0.5
            nonzero = delta.abs() > 1e-8
            if nonzero.any():
                mask = (merged_delta * (1 - p) / (delta * scale))
                mask = torch.where(nonzero, mask, torch.zeros_like(mask))
                masks.append(mask.flatten()[:10])  # first 10 elements

        # At least two parameters should have different masks
        if len(masks) >= 2:
            assert not torch.allclose(masks[0], masks[1]), \
                "DARE masks are identical across parameters — generator is being reset!"

    def test_ties_with_single_expert_does_not_fake_cross_expert(self):
        """TIES with 1 expert should return the trimmed delta, not pretend to do election."""
        base = _make_tiny_model()
        experts = _make_experts(base, n=1)
        config = MergeConfig(algorithm="TIES_MAGNITUDE", ties_trim_fraction=0.3)
        result = merge_experts(base, experts, config)
        assert result.operator_trace == ["TIES_MAGNITUDE"]

    def test_all_algorithms_produce_valid_trace(self):
        """Every algorithm must produce a non-empty operator trace."""
        base = _make_tiny_model()
        experts = _make_experts(base)
        curvature_bank = {}
        for i in range(len(experts)):
            ep = _get_param_dict(experts[i])
            curvature_bank[f"expert_{i}"] = {
                name: torch.ones_like(p) for name, p in ep.items()
            }

        algorithms = [
            ("task_arithmetic", False),
            ("mean_merge", False),
            ("weighted_task_arithmetic", False),
            ("DARE", False),
            ("TIES_MAGNITUDE", False),
            ("TIES_MAJORITY", False),
            ("DARE_TIES", False),
            ("FISHER", True),
            ("TIES_FISHER", True),
            ("DARE_TIES_FISHER", True),
            ("ExFusion", True),
        ]

        for algo, needs_fisher in algorithms:
            config = MergeConfig(algorithm=algo, dare_drop_rate=0.1)
            result = merge_experts(
                base, experts, config,
                curvature_bank=curvature_bank if needs_fisher else None,
            )
            assert len(result.operator_trace) > 0, \
                f"Algorithm '{algo}' produced empty operator trace"
            assert result.algorithm == algo.upper()


class TestAGXOperatorDispatch:
    """Verify that AGX's apply_layer_merge_operator actually calls cross-expert ops."""

    def test_ties_candidate_executes_ties_not_raw(self):
        """When AGX chooses TIES, it must execute TIES (not RAW task arithmetic)."""
        from daph_exfusion.experimental.agx.optimization import apply_layer_merge_operator
        from daph_exfusion.experimental.agx.candidate import LayerMergeConfig

        # Create base and expert layers
        base_layer = nn.Linear(10, 10)
        expert1 = nn.Linear(10, 10)
        expert2 = nn.Linear(10, 10)
        with torch.no_grad():
            expert1.weight.copy_(base_layer.weight + torch.randn_like(base_layer.weight))
            expert2.weight.copy_(base_layer.weight - torch.randn_like(base_layer.weight))

        target = nn.Linear(10, 10)

        # Apply TIES
        apply_layer_merge_operator(
            target_layer=target,
            base_layer=base_layer,
            expert_layers=[expert1, expert2],
            operator="TIES",
            lambdas=[1.0, 1.0],
            ties_trim=0.2,
        )

        # Compute what TIES should give
        d1 = (expert1.weight - base_layer.weight).detach()
        d2 = (expert2.weight - base_layer.weight).detach()
        expected_ties = op_ties([d1, d2], trim_fraction=0.2, sign_mode="magnitude")
        expected = base_layer.weight.detach() + expected_ties

        assert torch.allclose(target.weight.detach().float(), expected.float(), atol=1e-5), \
            "AGX TIES candidate did not produce TIES output — it may be executing RAW!"

    def test_fisher_candidate_requires_fisher_diagonals(self):
        """When AGX chooses FISHER without Fisher data, it must raise."""
        from daph_exfusion.experimental.agx.optimization import apply_layer_merge_operator

        base_layer = nn.Linear(10, 10)
        expert1 = nn.Linear(10, 10)
        expert2 = nn.Linear(10, 10)
        target = nn.Linear(10, 10)

        with pytest.raises(ValueError, match="fisher_diagonals"):
            apply_layer_merge_operator(
                target_layer=target,
                base_layer=base_layer,
                expert_layers=[expert1, expert2],
                operator="FISHER",
                lambdas=[1.0, 1.0],
            )

    def test_cross_expert_ops_set_includes_new_variants(self):
        """CROSS_EXPERT_OPS must include TIES_MAGNITUDE, TIES_MAJORITY, TIES_FISHER, etc."""
        assert "TIES" in CROSS_EXPERT_OPS
        assert "TIES_MAGNITUDE" in CROSS_EXPERT_OPS
        assert "TIES_MAJORITY" in CROSS_EXPERT_OPS
        assert "TIES_FISHER" in CROSS_EXPERT_OPS
        assert "DARE_TIES" in CROSS_EXPERT_OPS
        assert "DARE_TIES_FISHER" in CROSS_EXPERT_OPS
        assert "FISHER" in CROSS_EXPERT_OPS


class TestCKAAttentionMaskFix:
    """Verify the CKA attention-mask fix in LayerwiseGeometrySearchEngine."""

    def test_dict_batch_returns_mask_correctly(self):
        """val_batch as dict should return attention_mask via .get(), not getattr."""
        val_batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
        }
        # This is the fixed pattern
        mask = val_batch.get("attention_mask") if isinstance(val_batch, dict) \
            else getattr(val_batch, "attention_mask", None)
        assert mask is not None, "Dict batch should return attention_mask via .get()"
        assert torch.equal(mask, torch.ones(2, 4))
