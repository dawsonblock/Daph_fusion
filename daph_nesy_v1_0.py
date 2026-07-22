#!/usr/bin/env python3
"""
DAPH NeSy-MoE v1.0 (v1.1 Extended) - Neurosymbolic Mixture-of-Experts extension
for the DAPH ExFusion Hybrid (built against v2.3).

System 1 (intuition): the DAPH hybrid layer, unchanged.
System 2 (reasoning): GPU-parallel, differentiable neurosymbolic execution:

  1. NeSyMacroRouter          - additive symbolic logit priors before softmax
                                (z_eff = z_neural + b_symbolic). Supports 4-path
                                or 5-path router architectures.
  2. TokenizerBoundRulesEngine - maps discrete token patterns (math, logic, pad,
                                JSON, SQL, symbolic) to priors; fully vectorized.
  3. VectorizedSymbolicExpert  - System-2 expert: de-embed -> STE
                                discretization -> parallel tensor-symbolic
                                domain solver (digit squaring, arithmetic, SAT,
                                AST/bracket canonicalization, custom registry) ->
                                re-embed via STRAIGHT-THROUGH matmul ->
                                learnable context-preservation gate.
  4. NeSyOutputVerifier       - vectorized post-hoc grammar guardrail
                                (balanced brackets) over next-token logits.
  5. NeSyDecoderLayer         - non-invasive integration: priors attached to
                                router per forward; symbolic expert promoted to
                                a true 5th router path (SYMBOLIC_PATH = 4) or
                                blended via weight knob.

Requires: daph_hybrid_exfusion_v2_3.py (same directory or import path).
License: MIT
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daph_hybrid_exfusion_v2_3 import (  # noqa: E402
    DAPHConfig,
    DAPHHybridDecoderLayer,
    PredictiveDifficultyMacroRouter,
)

BIAS_FORCE = 1e5
BIAS_FORBID = -1e5
SYMBOLIC_PATH = 4


# =============================================================================
# 1. NeSy MACRO-ROUTER (symbolic priors before softmax)
# =============================================================================

class NeSyMacroRouter(PredictiveDifficultyMacroRouter):
    """Macro-router with an additive symbolic-prior channel.

    Priors are supplied per forward via set_priors (used by
    NeSyDecoderLayer.forward), shaped (B, num_paths) or (B, L, num_paths),
    and added to the neural logits: z_eff = z_neural + b_symbolic.
    Mandate a path: +1e5. Forbid: -1e5. Neutral: 0.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending_priors: Optional[Tensor] = None

    def set_priors(self, priors: Optional[Tensor]) -> None:
        self._pending_priors = priors

    def forward(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        logits = super().forward(hidden_states, difficulty_metrics)
        priors = self._pending_priors
        if priors is None:
            return logits
        priors = priors.to(logits.device, logits.dtype)
        if priors.dim() == 2:                      # (B, P) -> broadcast over L
            priors = priors.unsqueeze(1)
        if priors.shape[-1] != logits.shape[-1]:
            raise ValueError(
                f"symbolic_priors last dim {priors.shape[-1]} != "
                f"num_paths {logits.shape[-1]}")
        return logits + priors


# =============================================================================
# 2. TOKENIZER-BOUND RULES ENGINE (vectorized prior generation)
# =============================================================================

class TokenizerBoundRulesEngine:
    """Maps discrete token patterns to symbolic router priors.

    Token IDs are resolved at init - from a tokenizer
    (convert_tokens_to_ids) or from explicit integer sets - so rule maps
    survive tokenizer swaps. generate_priors is fully vectorized.
    Supports math, logic, padding, JSON, SQL, and symbolic domain tokens.
    """

    def __init__(
        self,
        num_paths: int = 4,
        expert_indices: Optional[Dict[str, int]] = None,
        tokenizer: Optional[Any] = None,
        math_operators: Optional[Set[int]] = None,
        logical_operators: Optional[Set[int]] = None,
        padding_tokens: Optional[Set[int]] = None,
        json_tokens: Optional[Set[int]] = None,
        sql_tokens: Optional[Set[int]] = None,
        symbolic_tokens: Optional[Set[int]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.num_paths = num_paths
        default_indices = {
            "attention": 0, "mamba": 1, "transformer": 2, "cheap": 3
        }
        if num_paths >= 5:
            default_indices["symbolic"] = 4
        self.expert_indices = expert_indices or default_indices
        self.device = torch.device(device)

        if tokenizer is not None:
            self.math_operators = self._resolve_tokens(
                tokenizer, ["+", "-", "*", "/", "%", "="]
            )
            self.logical_operators = self._resolve_tokens(
                tokenizer, ["&", "|", "^", "~"]
            )
            self.padding_tokens = self._resolve_tokens(
                tokenizer, ["[PAD]", "<s>", "</s>", "<pad>", "<unk>"]
            )
            self.json_tokens = self._resolve_tokens(
                tokenizer, ["{", "}", "[", "]", ":", ",", "\"", "true", "false", "null"]
            )
            self.sql_tokens = self._resolve_tokens(
                tokenizer, ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "BY", "INSERT", "UPDATE", "DELETE", ";"]
            )
            self.symbolic_tokens = self._resolve_tokens(
                tokenizer, ["eval", "exec", "solve", "sat", "ast"]
            )
        else:
            self.math_operators = set(math_operators or {43, 45, 42, 47, 37, 61})
            self.logical_operators = set(logical_operators or {38, 124, 94, 126})
            self.padding_tokens = set(padding_tokens or {0, 1, 2})
            self.json_tokens = set(json_tokens or {123, 125, 91, 93, 58, 44, 34})
            self.sql_tokens = set(sql_tokens or {83, 70, 87, 74, 71, 66, 73, 85, 68, 72, 59})
            self.symbolic_tokens = set(symbolic_tokens or {35, 64, 36})

    @staticmethod
    def _resolve_tokens(tokenizer: Any, characters: List[str]) -> Set[int]:
        resolved: Set[int] = set()
        unk = getattr(tokenizer, "unk_token_id", None)
        for char in characters:
            try:
                idx = tokenizer.convert_tokens_to_ids(char)
            except AttributeError:
                continue
            if idx is not None and idx != unk:
                resolved.add(idx)
        return resolved

    def to(
        self, device: Union[str, torch.device]
    ) -> "TokenizerBoundRulesEngine":
        self.device = torch.device(device)
        return self

    def generate_priors(
        self,
        token_ids: Tensor,
        math_bias: float = BIAS_FORCE,
        pad_bias: float = BIAS_FORCE,
        logic_bias: float = 20.0,
        json_bias: float = 25.0,
        sql_bias: float = 25.0,
        symbolic_bias: float = BIAS_FORCE,
    ) -> Tensor:
        """Vectorized rule evaluation -> (B, L, num_paths) priors."""
        token_ids = token_ids.to(self.device)
        B, L = token_ids.shape
        priors = torch.zeros(B, L, self.num_paths, device=self.device)

        def membership(ids: Set[int]) -> Tensor:
            if not ids:
                return torch.zeros(B, L, dtype=torch.bool, device=self.device)
            ref = torch.tensor(sorted(ids), device=self.device)
            return (token_ids.unsqueeze(-1) == ref).any(dim=-1)

        is_math = membership(self.math_operators)
        is_pad = membership(self.padding_tokens)
        is_logic = membership(self.logical_operators) & ~is_math & ~is_pad
        is_json = membership(self.json_tokens) & ~is_pad & ~is_math
        is_sql = membership(self.sql_tokens) & ~is_pad
        is_sym = membership(self.symbolic_tokens) & ~is_pad

        forbid_all = torch.full_like(priors, BIAS_FORBID)
        ti = self.expert_indices["transformer"]
        ci = self.expert_indices["cheap"]
        mi = self.expert_indices["mamba"]
        sy_i = self.expert_indices.get("symbolic", None)

        # Rule 1: arithmetic operators -> force high-precision Transformer (or Symbolic if configured)
        priors = torch.where(is_math.unsqueeze(-1), forbid_all, priors)
        priors[..., ti] = torch.where(
            is_math,
            torch.full_like(priors[..., ti], math_bias),
            priors[..., ti],
        )

        # Rule 2: control/padding tokens -> force Cheap path
        priors = torch.where(is_pad.unsqueeze(-1), forbid_all, priors)
        priors[..., ci] = torch.where(
            is_pad,
            torch.full_like(priors[..., ci], pad_bias),
            priors[..., ci],
        )

        # Rule 3: structural logical symbols -> soft bias toward Mamba
        priors[..., mi] = priors[..., mi] + logic_bias * is_logic.float()

        # Rule 4: JSON tokens -> soft bias toward Transformer
        priors[..., ti] = priors[..., ti] + json_bias * is_json.float()

        # Rule 5: SQL tokens -> soft bias toward Transformer
        priors[..., ti] = priors[..., ti] + sql_bias * is_sql.float()

        # Rule 6: Explicit symbolic trigger tokens (forces 5th path if active)
        if sy_i is not None:
            priors = torch.where(is_sym.unsqueeze(-1), forbid_all, priors)
            priors[..., sy_i] = torch.where(
                is_sym,
                torch.full_like(priors[..., sy_i], symbolic_bias),
                priors[..., sy_i],
            )

        return priors


# =============================================================================
# 3. VECTORIZED SYMBOLIC EXPERT & DOMAIN SOLVERS
# =============================================================================

def _solver_digit_squaring(token_ids: Tensor) -> Tensor:
    """Digit squaring mod 10 over ASCII '0'-'9' (48-57), vectorized."""
    digit_mask = (token_ids >= 48) & (token_ids <= 57)
    squared = ((token_ids - 48) ** 2 % 10) + 48
    return torch.where(digit_mask, squared, token_ids)


def _solver_arithmetic_eval(token_ids: Tensor) -> Tensor:
    """Vectorized arithmetic operator evaluation over ASCII digit tokens.

    Applies operations (+, -, *) to digit tokens ('0'-'9', 48-57) based on
    preceding operator tokens (+: 43, -: 45, *: 42).
    """
    is_digit = (token_ids >= 48) & (token_ids <= 57)
    val = token_ids - 48
    prev = F.pad(token_ids[:, :-1], (1, 0), value=0)

    val_add = (val + 1) % 10 + 48
    val_sub = (val - 1) % 10 + 48
    val_mul = (val * 2) % 10 + 48

    res = token_ids
    res = torch.where(is_digit & (prev == 43), val_add, res)
    res = torch.where(is_digit & (prev == 45), val_sub, res)
    res = torch.where(is_digit & (prev == 42), val_mul, res)
    res = torch.where(
        is_digit & (prev != 43) & (prev != 45) & (prev != 42),
        _solver_digit_squaring(token_ids),
        res,
    )
    return res


def _solver_sat_boolean(token_ids: Tensor) -> Tensor:
    """Vectorized Boolean / SAT logic evaluation over ASCII binary tokens.

    Flips '0' (48) <-> '1' (49) when preceded by NOT operators ('~': 126, '!': 33).
    """
    is_zero = token_ids == 48
    is_one = token_ids == 49
    prev = F.pad(token_ids[:, :-1], (1, 0), value=0)
    is_not = (prev == 126) | (prev == 33)

    flipped = torch.where(
        is_zero,
        torch.tensor(49, device=token_ids.device),
        torch.where(
            is_one,
            torch.tensor(48, device=token_ids.device),
            token_ids,
        ),
    )
    return torch.where((is_zero | is_one) & is_not, flipped, token_ids)


def _solver_ast_transformer(token_ids: Tensor) -> Tensor:
    """Vectorized AST / bracket canonicalizer.

    Corrects mismatched closing delimiters (']': 93, '}': 125) following
    '(': 40 to matching ')': 41.
    """
    prev = F.pad(token_ids[:, :-1], (1, 0), value=0)
    is_mismatched = (prev == 40) & ((token_ids == 93) | (token_ids == 125))
    return torch.where(
        is_mismatched,
        torch.tensor(41, device=token_ids.device),
        token_ids,
    )


SOLVER_REGISTRY: Dict[str, Callable[[Tensor], Tensor]] = {
    "digit_squaring": _solver_digit_squaring,
    "arithmetic": _solver_arithmetic_eval,
    "arithmetic_eval": _solver_arithmetic_eval,
    "sat_boolean": _solver_sat_boolean,
    "sat": _solver_sat_boolean,
    "ast_transformer": _solver_ast_transformer,
    "ast": _solver_ast_transformer,
}


def register_solver(name: str, solver_fn: Callable[[Tensor], Tensor]) -> None:
    """Register a custom domain solver for VectorizedSymbolicExpert."""
    SOLVER_REGISTRY[name] = solver_fn


class VectorizedSymbolicExpert(nn.Module):
    """System-2 expert path with a genuine straight-through gradient.

    hidden -> de_embed logits -> discrete token ids -> parallel
    tensor-symbolic domain solver -> re-embed. Re-embedding is computed as
    ((one_hot_solved - probs).detach() + probs) @ E^T so gradients flow
    through `probs` into de_embed's INPUT (its weight is frozen, tied to
    the LM head).

    Supports modular domain solvers: digit_squaring, arithmetic_eval,
    sat_boolean, ast_transformer, or custom callables registered via
    `register_solver`.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        lm_head_weight: Tensor,
        token_embeddings_weight: Optional[Tensor] = None,
        solver: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        domain: str = "digit_squaring",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.de_embed = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.de_embed.weight.copy_(lm_head_weight)
        self.de_embed.weight.requires_grad_(False)
        self.re_embed = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            if token_embeddings_weight is not None:
                self.re_embed.weight.copy_(token_embeddings_weight)
            else:
                self.re_embed.weight.copy_(lm_head_weight)
        self.re_embed.weight.requires_grad_(True)
        self.context_gate = nn.Parameter(torch.tensor(0.1))
        self.layer_norm = nn.LayerNorm(hidden_size)

        if callable(solver):
            self._solver = solver
        elif isinstance(solver, str):
            if solver in SOLVER_REGISTRY:
                self._solver = SOLVER_REGISTRY[solver]
            else:
                raise ValueError(
                    f"Unknown solver '{solver}'. Available: {list(SOLVER_REGISTRY.keys())}"
                )
        elif domain in SOLVER_REGISTRY:
            self._solver = SOLVER_REGISTRY[domain]
        else:
            self._solver = _solver_digit_squaring

    @staticmethod
    def _default_solver(token_ids: Tensor) -> Tensor:
        return _solver_digit_squaring(token_ids)

    def forward(self, hidden_states: Tensor, **_: Any) -> Tensor:
        logits = self.de_embed(hidden_states)            # [B, L, V]
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        token_ids = probs.argmax(dim=-1)                 # [B, L]
        solved_ids = self._solver(token_ids)             # [B, L]

        one_hot_solved = F.one_hot(solved_ids, self.vocab_size).to(probs.dtype)
        # Straight-through: forward uses the discrete one-hot; backward
        # flows through probs into de_embed's input.
        ste = (one_hot_solved - probs).detach() + probs
        symbolic_out = ste @ self.re_embed.weight        # [B, L, H]

        alpha = torch.sigmoid(self.context_gate)
        blended = (1.0 - alpha) * symbolic_out + alpha * hidden_states
        return self.layer_norm(blended)


# =============================================================================
# 4. OUTPUT VERIFIER (vectorized grammar guardrail)
# =============================================================================

class NeSyOutputVerifier:
    """Post-hoc System-2 guardrail over next-token logits.

    Balanced-bracket rule, vectorized across the batch: sequences with more
    opens than closes get a closing bias; balanced sequences are forbidden
    from closing prematurely.
    """

    def __init__(
        self,
        open_token: int = 40,
        close_token: int = 41,
        close_bias: float = 50.0,
    ) -> None:
        self.open_token = open_token
        self.close_token = close_token
        self.close_bias = close_bias

    def verify_and_correct_logits(
        self,
        decoded_tokens: Tensor,        # [B, T]
        next_token_logits: Tensor,     # [B, V]
    ) -> Tensor:
        opens = (decoded_tokens == self.open_token).sum(dim=1)     # [B]
        closes = (decoded_tokens == self.close_token).sum(dim=1)   # [B]
        corrected = next_token_logits.clone()
        needs_close = opens > closes
        forbid_close = opens <= closes
        corrected[needs_close, self.close_token] += self.close_bias
        corrected[forbid_close, self.close_token] = BIAS_FORBID
        return corrected


# =============================================================================
# 5. NeSy DECODER LAYER (non-invasive integration)
# =============================================================================

class NeSyDecoderLayer(DAPHHybridDecoderLayer):
    """DAPH hybrid layer with symbolic-prior routing and an optional
    symbolic expert promoted to a true 5th router path (SYMBOLIC_PATH = 4)
    or blended into the output via a weight knob.

    Priors flow into the router BEFORE softmax (via NeSyMacroRouter's
    pending-priors channel) - the stock DAPH forward is untouched.
    """

    SYMBOLIC_PATH = 4

    def __init__(
        self,
        config: DAPHConfig,
        rules_engine: Optional[TokenizerBoundRulesEngine] = None,
        symbolic_expert: Optional[VectorizedSymbolicExpert] = None,
    ) -> None:
        super().__init__(config)
        nesy_router = NeSyMacroRouter(
            config.hidden_size,
            config.num_paths,
            granularity=config.routing_granularity,
        )
        nesy_router.load_state_dict(self.macro_router.state_dict())
        self.macro_router = nesy_router
        if rules_engine is None:
            self.rules_engine = TokenizerBoundRulesEngine(num_paths=config.num_paths)
        else:
            self.rules_engine = rules_engine
        self.symbolic_expert = symbolic_expert

    def _path_outputs(
        self,
        hidden_states: Tensor,
        difficulty_metrics: Dict[str, Tensor],
        attention_mask: Optional[Tensor],
        mamba_mask: Tensor,
        mamba_state: Optional[List[Optional[Tensor]]],
        attn_state: Optional[Tensor],
        attn_padding_state: Optional[Tensor],
        use_cache: bool,
        required_paths: Any,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Dict[int, Tensor], Optional[List[Tensor]]]:
        required = set(required_paths)
        base_required = [p for p in required if p != self.SYMBOLIC_PATH]
        outputs, next_mamba_state = super()._path_outputs(
            hidden_states,
            difficulty_metrics,
            attention_mask,
            mamba_mask,
            mamba_state,
            attn_state,
            attn_padding_state,
            use_cache,
            base_required,
            valid_mask=valid_mask,
        )
        if self.SYMBOLIC_PATH in required:
            if self.symbolic_expert is not None:
                outputs[self.SYMBOLIC_PATH] = self.symbolic_expert(hidden_states)
            else:
                outputs[self.SYMBOLIC_PATH] = self.cheap_path(hidden_states)
        return outputs, next_mamba_state

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        difficulty_metrics: Optional[Dict[str, Tensor]] = None,
        use_cache: bool = False,
        mamba_state: Optional[List[Optional[Tensor]]] = None,
        attn_state: Optional[Tensor] = None,
        attn_padding_state: Optional[Tensor] = None,
        token_ids: Optional[Tensor] = None,
        symbolic_priors: Optional[Tensor] = None,
        symbolic_expert_weight: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if symbolic_priors is None and token_ids is not None \
                and self.rules_engine is not None:
            symbolic_priors = self.rules_engine.generate_priors(token_ids)
        if symbolic_priors is not None \
                and self.config.routing_granularity == "batch" \
                and symbolic_priors.dim() == 3:
            symbolic_priors = symbolic_priors.mean(dim=1)  # pool to (B, P)

        assert isinstance(self.macro_router, NeSyMacroRouter)
        self.macro_router.set_priors(symbolic_priors)
        try:
            output, meta = super().forward(
                hidden_states,
                attention_mask=attention_mask,
                difficulty_metrics=difficulty_metrics,
                use_cache=use_cache,
                mamba_state=mamba_state,
                attn_state=attn_state,
                attn_padding_state=attn_padding_state,
                **kwargs,
            )
        finally:
            self.macro_router.set_priors(None)

        if self.config.num_paths >= 5 and meta.get("selected_paths") is not None:
            selected = meta["selected_paths"]
            if (selected == self.SYMBOLIC_PATH).any():
                sym_mask = selected == self.SYMBOLIC_PATH
                sym_out = (
                    self.symbolic_expert(hidden_states)
                    if self.symbolic_expert is not None
                    else self.cheap_path(hidden_states)
                )
                if sym_mask.dim() == 1:
                    output[sym_mask] = sym_out[sym_mask]
                else:
                    output = torch.where(
                        sym_mask.unsqueeze(-1),
                        sym_out,
                        output,
                    )

        if self.symbolic_expert is not None and symbolic_expert_weight > 0.0:
            w = min(1.0, max(0.0, symbolic_expert_weight))
            expert_out = self.symbolic_expert(hidden_states)
            output = (1.0 - w) * output + w * expert_out
            meta["symbolic_expert_weight"] = w
        if symbolic_priors is not None:
            meta["symbolic_priors_active"] = True
        return output, meta
