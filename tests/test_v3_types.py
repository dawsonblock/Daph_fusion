"""Unit tests for v3 canonical merge types.

Tests the mathematical contracts of the v3 type system:
    - MergeConfig/MergeMethod dispatch
    - ExpertSpec immutability
    - OperatorTrace provenance
    - Task vector extraction
    - Parameter family classification
    - SSM stability validation
"""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.merge.types import (
    ExpertSpec,
    MergeConfig,
    MergeMethod,
    MergeResult,
    OperatorTrace,
    CoefficientGranularity,
    CoefficientParameterization,
    FisherStabilization,
    RegMeanMode,
    extract_task_vectors,
    validate_parameter_names,
    classify_parameter_family,
    classify_parameter_family_fine,
    FINE_FAMILIES,
    get_layer_index,
    count_layers,
    validate_ssm_stability,
)


# =============================================================================
# ExpertSpec
# =============================================================================


class TestExpertSpec:
    def test_immutable(self):
        spec = ExpertSpec(
            name="math",
            checkpoint_path="/path/to/math",
            checkpoint_hash="abc123",
            target_domain="math",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "code"

    def test_to_dict(self):
        spec = ExpertSpec("math", "/path", "abc", "math")
        d = spec.to_dict()
        assert d["name"] == "math"
        assert d["checkpoint_hash"] == "abc"


# =============================================================================
# MergeConfig
# =============================================================================


class TestMergeConfig:
    def test_default_method(self):
        config = MergeConfig()
        assert config.method == MergeMethod.TASK_ARITHMETIC

    def test_method_from_string(self):
        config = MergeConfig(method="fisher_dense")
        assert config.method == MergeMethod.FISHER_DENSE

    def test_method_from_enum(self):
        config = MergeConfig(method=MergeMethod.REGMEAN)
        assert config.method == MergeMethod.REGMEAN

    def test_config_hash_deterministic(self):
        c1 = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=0.5)
        c2 = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=0.5)
        assert c1.config_hash() == c2.config_hash()

    def test_config_hash_differs_for_different_configs(self):
        c1 = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=0.5)
        c2 = MergeConfig(method=MergeMethod.TASK_ARITHMETIC, task_scale=1.0)
        assert c1.config_hash() != c2.config_hash()

    def test_is_stochastic_dare(self):
        config = MergeConfig(method=MergeMethod.DARE)
        assert config.is_stochastic

    def test_is_not_stochastic_ta(self):
        config = MergeConfig(method=MergeMethod.TASK_ARITHMETIC)
        assert not config.is_stochastic

    def test_stabilization_from_string(self):
        config = MergeConfig(fisher_stabilization="floor")
        assert config.fisher_stabilization == FisherStabilization.FLOOR

    def test_regmean_mode_from_string(self):
        config = MergeConfig(regmean_mode="diagonal")
        assert config.regmean_mode == RegMeanMode.DIAGONAL

    def test_coefficient_granularity_from_string(self):
        config = MergeConfig(coefficient_granularity="family")
        assert config.coefficient_granularity == CoefficientGranularity.FAMILY

    def test_to_dict_serializable(self):
        import json
        config = MergeConfig(method=MergeMethod.FISHER_DENSE, fisher_gamma=0.5)
        d = config.to_dict()
        # Should be JSON serializable
        json.dumps(d)


# =============================================================================
# MergeMethod enum
# =============================================================================


class TestMergeMethod:
    def test_dense_methods_contains_ta(self):
        assert MergeMethod.TASK_ARITHMETIC in MergeMethod.dense_methods()

    def test_dense_methods_contains_fisher(self):
        assert MergeMethod.FISHER_DENSE in MergeMethod.dense_methods()
        assert MergeMethod.FISHER_BASE_ANCHORED in MergeMethod.dense_methods()

    def test_dense_methods_contains_regmean(self):
        assert MergeMethod.REGMEAN in MergeMethod.dense_methods()

    def test_legacy_methods_contains_dare(self):
        assert MergeMethod.DARE in MergeMethod.legacy_methods()

    def test_legacy_methods_contains_ties(self):
        assert MergeMethod.TIES_MAGNITUDE in MergeMethod.legacy_methods()

    def test_dense_and_legacy_disjoint(self):
        assert MergeMethod.dense_methods().isdisjoint(MergeMethod.legacy_methods())


# =============================================================================
# OperatorTrace
# =============================================================================


class TestOperatorTrace:
    def test_assert_operator_present(self):
        trace = OperatorTrace(method="regmean", operators=["REGMEAN"])
        trace.assert_operator("REGMEAN")

    def test_assert_operator_absent(self):
        trace = OperatorTrace(method="regmean", operators=["REGMEAN"])
        trace.assert_no_operator("DARE")

    def test_assert_operator_fails_when_absent(self):
        trace = OperatorTrace(method="task_arithmetic", operators=["TASK_ARITHMETIC"])
        with pytest.raises(AssertionError):
            trace.assert_operator("FISHER")

    def test_assert_no_operator_fails_when_present(self):
        trace = OperatorTrace(method="fisher_dense", operators=["EMPIRICAL_FISHER"])
        with pytest.raises(AssertionError):
            trace.assert_no_operator("EMPIRICAL_FISHER")

    def test_to_dict(self):
        trace = OperatorTrace(
            method="fisher_dense",
            operators=["EMPIRICAL_FISHER", "DENSE_PRECISION_MERGE"],
            fisher_used=True,
            fisher_estimator="exact_per_sample",
        )
        d = trace.to_dict()
        assert d["fisher_used"] is True
        assert d["fisher_estimator"] == "exact_per_sample"


# =============================================================================
# Task vector extraction
# =============================================================================


class TestTaskVectorExtraction:
    def test_extract_task_vectors(self):
        base = nn.Linear(10, 5)
        expert = nn.Linear(10, 5)
        # Make expert different from base
        with torch.no_grad():
            expert.weight.copy_(base.weight + 1.0)
            expert.bias.copy_(base.bias + 0.5)

        tvs = extract_task_vectors([expert], base)
        assert len(tvs) == 1
        assert "weight" in tvs[0]
        assert "bias" in tvs[0]
        assert torch.allclose(tvs[0]["weight"], torch.ones_like(base.weight))
        assert torch.allclose(tvs[0]["bias"], torch.ones_like(base.bias) * 0.5)

    def test_extract_fp32(self):
        base = nn.Linear(10, 5)
        expert = nn.Linear(10, 5)
        with torch.no_grad():
            expert.weight.copy_(base.weight + 1.0)

        tvs = extract_task_vectors([expert], base)
        assert tvs[0]["weight"].dtype == torch.float32

    def test_validate_parameter_names_match(self):
        base = nn.Linear(10, 5)
        expert = nn.Linear(10, 5)
        validate_parameter_names([expert], base)  # should not raise

    def test_validate_parameter_names_mismatch(self):
        base = nn.Linear(10, 5)
        expert = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        # expert has different parameter names (0.weight, 0.bias, 1.weight, 1.bias)
        with pytest.raises(ValueError):
            validate_parameter_names([expert], base)


# =============================================================================
# Parameter family classification
# =============================================================================


class TestParameterFamily:
    def test_embeddings(self):
        assert classify_parameter_family("model.embed_tokens.weight") == "embeddings"

    def test_lm_head(self):
        assert classify_parameter_family("lm_head.weight") == "lm_head"

    def test_normalization(self):
        assert classify_parameter_family("model.layers.0.input_layernorm.weight") == "normalization"

    def test_attention_early(self):
        assert classify_parameter_family("model.layers.0.attn.q_proj.weight", 0, 12) == "early_attention"

    def test_attention_middle(self):
        assert classify_parameter_family("model.layers.5.attn.q_proj.weight", 5, 12) == "middle_attention"

    def test_attention_late(self):
        assert classify_parameter_family("model.layers.10.attn.q_proj.weight", 10, 12) == "late_attention"

    def test_ffn(self):
        assert "ffn" in classify_parameter_family("model.layers.5.mlp.gate_proj.weight", 5, 12)

    def test_ssm_recurrence_a_log(self):
        assert classify_parameter_family("model.layers.0.ssm.a_log") == "ssm_recurrence"

    def test_ssm_recurrence_dt(self):
        assert classify_parameter_family("model.layers.0.ssm.dt_proj.weight") == "ssm_recurrence"

    def test_ssm_projections(self):
        assert classify_parameter_family("model.layers.0.ssm.in_proj.weight") == "ssm_projections"


class TestParameterFamilyFine:
    def test_fine_families_constant(self):
        assert FINE_FAMILIES == (
            "attention",
            "ssm",
            "ffn",
            "norm",
            "embedding",
            "lm_head",
            "router",
            "other",
        )

    def test_embedding(self):
        assert classify_parameter_family_fine("model.embed_tokens.weight") == "embedding"

    def test_lm_head(self):
        assert classify_parameter_family_fine("lm_head.weight") == "lm_head"

    def test_norm(self):
        assert classify_parameter_family_fine("model.layers.0.input_layernorm.weight") == "norm"

    def test_attention(self):
        assert classify_parameter_family_fine("model.layers.0.attn.q_proj.weight") == "attention"

    def test_ffn_gate_proj(self):
        assert classify_parameter_family_fine("model.layers.0.mlp.gate_proj.weight") == "ffn"

    def test_ffn_down_proj(self):
        assert classify_parameter_family_fine("model.layers.0.mlp.down_proj.weight") == "ffn"

    def test_ssm_in_proj(self):
        assert classify_parameter_family_fine("model.layers.0.ssm.in_proj.weight") == "ssm"

    def test_ssm_a_log(self):
        assert classify_parameter_family_fine("model.layers.0.ssm.a_log") == "ssm"

    def test_router(self):
        assert classify_parameter_family_fine("model.layers.0.router.weight") == "router"

    def test_other(self):
        assert classify_parameter_family_fine("some.unknown.param") == "other"


# =============================================================================
# SSM stability validation
# =============================================================================


class TestSSMStability:
    def test_valid_ssm(self):
        model = nn.Module()
        model.ssm = nn.Module()
        model.ssm.a_log = nn.Parameter(torch.randn(4))
        model.ssm.dt = nn.Parameter(torch.ones(4) * 0.1)
        result = validate_ssm_stability(model)
        assert result["valid"]

    def test_invalid_a_log(self):
        model = nn.Module()
        model.ssm = nn.Module()
        # a_log = 100 → A = -exp(100) = -inf
        model.ssm.a_log = nn.Parameter(torch.tensor([100.0]))
        result = validate_ssm_stability(model)
        assert not result["valid"]

    def test_negative_dt(self):
        model = nn.Module()
        model.ssm = nn.Module()
        model.ssm.dt = nn.Parameter(torch.tensor([-0.1, 0.1]))
        result = validate_ssm_stability(model)
        assert not result["valid"]


# =============================================================================
# Layer index extraction
# =============================================================================


class TestLayerIndex:
    def test_get_layer_index_hf_style(self):
        assert get_layer_index("model.layers.5.attn.q_proj.weight") == 5

    def test_get_layer_index_gpt2_style(self):
        assert get_layer_index("transformer.h.3.attn.c_attn.weight") == 3

    def test_get_layer_index_no_layer(self):
        assert get_layer_index("model.embed_tokens.weight") == -1

    def test_count_layers(self):
        model = nn.ModuleDict()
        layers = nn.ModuleList()
        for i in range(5):
            layer = nn.Module()
            layer.weight = nn.Parameter(torch.zeros(1))
            layers.append(layer)
        model["layers"] = layers
        assert count_layers(model) == 5
