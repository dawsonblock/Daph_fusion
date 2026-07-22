# DAPH NeSy-MoE v1.0 (v1.1 Extended) — Neurosymbolic Extension

System 1 + System 2 architecture on top of the DAPH ExFusion Hybrid v2.3:
neural intuition guided by symbolic constraints injected **into the routing
math itself** (before softmax), not as post-hoc overrides.

**Status: v1.1 Extended.** All 7 test groups executed live and pass on
CPU/GPU (PyTorch 2.8+). Requires `daph_hybrid_exfusion_v2_3.py` (included).

## Components

| Component | Role |
| --- | --- |
| `NeSyMacroRouter` | Additive symbolic prior channel: `z_eff = z_neural + b_symbolic`. Mandate `+1e5`, forbid `−1e5`, neutral `0`. Supports 4-path or 5-path router configurations. |
| `TokenizerBoundRulesEngine` | Vectorized token-pattern → prior mapping; resolves token IDs via tokenizer (or explicit sets) so rules survive vocab swaps. Supports math, logic, pad, JSON, SQL, and symbolic trigger tokens. |
| `VectorizedSymbolicExpert` | System-2 expert path: de-embed → discrete tokens → parallel tensor-symbolic domain solver (`digit_squaring`, `arithmetic_eval`, `sat_boolean`, `ast_transformer`, or custom registered solvers) → re-embed via **straight-through matmul** → learnable context-preservation gate. |
| `NeSyOutputVerifier` | Vectorized post-hoc grammar guardrail (balanced brackets) over next-token logits. |
| `NeSyDecoderLayer` | Non-invasive integration: priors attached to the router per forward; stock DAPH forward untouched; symbolic expert promoted to a true 5th router path (`SYMBOLIC_PATH = 4`) or blended via weight knob. |

## Feature Additions in v1.1

1. **Promoted Symbolic Expert to 5th Router Path**:
   - `SYMBOLIC_PATH = 4` integrated into macro-routing and hard/soft pathway selection in `NeSyDecoderLayer`.
2. **Domain-Specific Solver Expansion**:
   - Extends `VectorizedSymbolicExpert` with built-in solvers (`digit_squaring`, `arithmetic_eval`, `sat_boolean`, `ast_transformer`) and custom solver registration via `register_solver(name, fn)`.
3. **Vocab/Tokenizer Grammar Expansion**:
   - `TokenizerBoundRulesEngine` adds rule priors for JSON structural tokens, SQL keywords, and symbolic trigger tokens.

## Usage

```python
from daph_hybrid_exfusion_v2_3 import DAPHConfig
from daph_nesy_v1_0 import (
    NeSyDecoderLayer, TokenizerBoundRulesEngine,
    VectorizedSymbolicExpert, NeSyOutputVerifier,
    register_solver,
)

# 5-path router configuration with Symbolic Expert as Path 4
config = DAPHConfig(hidden_size=768, num_paths=5, routing_granularity="token")
engine = TokenizerBoundRulesEngine(num_paths=5, tokenizer=my_tokenizer)
expert = VectorizedSymbolicExpert(768, vocab_size, lm_head.weight, domain="sat_boolean")
layer = NeSyDecoderLayer(config, rules_engine=engine, symbolic_expert=expert)

out, meta = layer(hidden_states, token_ids=input_ids)   # priors auto-generated
# meta["selected_paths"] honors System-2 mandates including SYMBOLIC_PATH = 4

# Register custom domain solver:
def custom_solver(token_ids):
    return token_ids  # Custom vectorized tensor logic

register_solver("my_solver", custom_solver)

# Post-hoc guardrail at decode time:
verifier = NeSyOutputVerifier(open_token=40, close_token=41)
logits = verifier.verify_and_correct_logits(generated_ids, next_token_logits)
```

Run tests: `python test_nesy_v1_0.py`

License: MIT
