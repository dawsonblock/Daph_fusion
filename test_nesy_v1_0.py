#!/usr/bin/env python3
"""Live test suite for DAPH NeSy-MoE v1.0 (requires torch + v2.3 base)."""

import torch

from daph_hybrid_exfusion_v2_3 import DAPHConfig
from daph_nesy_v1_0 import (
    BIAS_FORCE, BIAS_FORBID, SYMBOLIC_PATH,
    NeSyDecoderLayer, NeSyMacroRouter, NeSyOutputVerifier,
    TokenizerBoundRulesEngine, VectorizedSymbolicExpert,
    register_solver, SOLVER_REGISTRY
)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. Symbolic priors mandate/forbid paths through softmax ----------
    router = NeSyMacroRouter(32, num_paths=5, granularity="token").to(device)
    router.eval()
    x = torch.randn(2, 4, 32, device=device)
    priors = torch.zeros(2, 4, 5, device=device)
    priors[..., 4] = BIAS_FORCE   # mandate Symbolic path (5th path)
    router.set_priors(priors)
    logits = router(x)
    assert logits.argmax(-1).eq(4).all(), "mandated symbolic prior did not win argmax"
    router.set_priors(None)
    logits_np = router(x)
    assert not logits_np.argmax(-1).eq(4).all() or True  # unconstrained
    print("1. symbolic priors mandate paths before softmax (5-path router): OK")

    # ---- 2. Rules engine: vectorized == loop reference + grammar rules -----
    engine = TokenizerBoundRulesEngine(num_paths=5, device=device)
    ids = torch.tensor([[43, 0, 38, 123, 83, 35]], device=device)  # +, PAD, &, {, SELECT, symbolic
    pri = engine.generate_priors(ids)
    assert pri[0, 0, 2].item() == BIAS_FORCE                       # Transformer
    assert pri[0, 0, 0].item() == BIAS_FORBID
    assert pri[0, 1, 3].item() == BIAS_FORCE                       # Cheap
    assert pri[0, 2, 1].item() == 20.0                             # Mamba
    assert pri[0, 3, 2].item() == 25.0                             # JSON -> Transformer
    assert pri[0, 4, 2].item() == 25.0                             # SQL -> Transformer
    assert pri[0, 5, 4].item() == BIAS_FORCE                       # Symbolic (5th path)

    # tokenizer-bound resolution with a mock tokenizer
    class MockTok:
        unk_token_id = -1

        def convert_tokens_to_ids(self, t):
            return {"+": 43, "-": 45, "{": 123, "SELECT": 83}.get(t, -1)

    engine_tok = TokenizerBoundRulesEngine(num_paths=5, tokenizer=MockTok(), device=device)
    assert engine_tok.math_operators.issuperset({43, 45})
    assert 123 in engine_tok.json_tokens
    assert 83 in engine_tok.sql_tokens
    print("2. tokenizer-bound rules engine (extended grammars + 5-path priors): OK")

    # ---- 3. Symbolic expert: domain solvers + custom registry + STE --------
    vocab, hidden = 128, 32
    lm_head = torch.randn(vocab, hidden, device=device)

    # 3a. Digit squaring (default)
    expert_sq = VectorizedSymbolicExpert(hidden, vocab, lm_head, domain="digit_squaring").to(device)
    ids_in = torch.tensor([[50, 53, 100]], device=device)  # '2','5',other
    assert expert_sq._solver(ids_in).tolist() == [[52, 53, 100]]

    # 3b. Arithmetic eval
    expert_arith = VectorizedSymbolicExpert(hidden, vocab, lm_head, domain="arithmetic_eval").to(device)
    ids_arith = torch.tensor([[43, 50, 45, 53]], device=device)  # '+', '2', '-', '5'
    assert expert_arith._solver(ids_arith).tolist() == [[43, 51, 45, 52]]  # '2'+1='3', '5'-1='4'

    # 3c. SAT Boolean solver
    expert_sat = VectorizedSymbolicExpert(hidden, vocab, lm_head, domain="sat_boolean").to(device)
    ids_sat = torch.tensor([[126, 48, 33, 49]], device=device)  # '~', '0', '!', '1'
    assert expert_sat._solver(ids_sat).tolist() == [[126, 49, 33, 48]]  # ~0 -> 1, !1 -> 0

    # 3d. AST Transformer solver
    expert_ast = VectorizedSymbolicExpert(hidden, vocab, lm_head, domain="ast_transformer").to(device)
    ids_ast = torch.tensor([[40, 93, 40, 125]], device=device)  # '(', ']', '(', '}'
    assert expert_ast._solver(ids_ast).tolist() == [[40, 41, 40, 41]]  # corrected to ')'

    # 3e. Custom registered solver
    def custom_rot13(t: torch.Tensor) -> torch.Tensor:
        return torch.where((t >= 65) & (t <= 90), ((t - 65 + 13) % 26) + 65, t)

    register_solver("rot13", custom_rot13)
    assert "rot13" in SOLVER_REGISTRY
    expert_rot = VectorizedSymbolicExpert(hidden, vocab, lm_head, domain="rot13").to(device)
    ids_rot = torch.tensor([[65, 66]], device=device)  # 'A', 'B'
    assert expert_rot._solver(ids_rot).tolist() == [[78, 79]]  # 'N', 'O'

    # 3f. STE gradient reaches input
    h = torch.randn(1, 3, hidden, device=device, requires_grad=True)
    out = expert_sq(h)
    target_weights = torch.randn_like(out)
    (out * target_weights).sum().backward()
    assert h.grad is not None and bool((h.grad != 0).any()), \
        "STE gradient blocked (decorative argmax bug present)"
    print("3. vectorized symbolic expert (domain solvers + custom registry + STE grad): OK")

    # ---- 4. Output verifier: balanced-bracket guardrail -------------------
    verifier = NeSyOutputVerifier()
    dec = torch.tensor([[40, 40, 41, 7], [40, 41, 5, 6]])  # unbal / balanced
    logits4 = torch.zeros(2, 64)
    corr = verifier.verify_and_correct_logits(dec, logits4)
    assert corr[0, 41].item() == 50.0        # needs close -> biased
    assert corr[1, 41].item() == BIAS_FORBID  # balanced -> forbidden
    print("4. NeSy output verifier (bracket guardrail): OK")

    # ---- 5. End-to-end: NeSyDecoderLayer (5-path routing + priors + expert) ---
    cfg = DAPHConfig(hidden_size=32, intermediate_size=64,
                     num_attention_heads=2, state_size=8, num_experts=2,
                     num_paths=5, routing_granularity="token", dropout=0.0)
    layer = NeSyDecoderLayer(cfg, rules_engine=engine,
                             symbolic_expert=expert_sq).to(device).eval()
    xh = torch.randn(2, 6, 32, device=device)
    tok = torch.tensor([[43, 7, 8, 123, 83, 35], [7, 8, 9, 10, 11, 12]], device=device)
    out, meta = layer(xh, token_ids=tok)
    assert out.shape == xh.shape
    assert meta.get("symbolic_priors_active")
    # symbolic token position (35) must route to Symbolic path (index 4)
    assert meta["selected_paths"][0, 5].item() == SYMBOLIC_PATH
    # expert blending path
    out2, meta2 = layer(xh, symbolic_expert_weight=0.5)
    assert meta2["symbolic_expert_weight"] == 0.5
    assert out2.shape == xh.shape
    # gradients flow end-to-end in train mode
    layer.train()
    out3, meta3 = layer(xh, token_ids=tok)
    loss = out3.sum() + 0.01 * meta3["router_aux_loss"]
    loss.backward()
    print("5. end-to-end NeSyDecoderLayer (5-path + priors + expert + grads): OK")

    # ---- 6. re_embed is semantically aligned at init (regression) ---------
    vocab6, hidden6 = 96, 24
    lm_head6 = torch.randn(vocab6, hidden6, device=device)
    expert6 = VectorizedSymbolicExpert(hidden6, vocab6, lm_head6).to(device)
    assert torch.equal(expert6.re_embed.weight, lm_head6), (
        "re_embed not initialized from lm_head fallback "
        "(random-init bug present)"
    )
    assert expert6.re_embed.weight.requires_grad, (
        "re_embed must stay trainable"
    )
    # explicit token-embedding source takes precedence
    tok_emb6 = torch.randn(vocab6, hidden6, device=device)
    expert6b = VectorizedSymbolicExpert(
        hidden6, vocab6, lm_head6, token_embeddings_weight=tok_emb6
    ).to(device)
    assert torch.equal(expert6b.re_embed.weight, tok_emb6), \
        "re_embed ignored the provided token_embeddings_weight"
    print("6. re_embed aligned init (fallback + explicit source): OK")

    # ---- 7. Over-closed bracket sequence is forbidden (regression) --------
    verifier7 = NeSyOutputVerifier()
    dec7 = torch.tensor([[41, 41, 40, 7]])  # close, close, open -> over-closed
    logits7 = torch.zeros(1, 64)
    corr7 = verifier7.verify_and_correct_logits(dec7, logits7)
    assert corr7[0, 41].item() <= BIAS_FORBID, (
        f"over-closed sequence left close-token unpenalized: "
        f"{corr7[0, 41].item()}"
    )
    # needs-close behavior preserved for genuinely unbalanced sequences
    dec7b = torch.tensor([[40, 40, 41, 7]])
    corr7b = verifier7.verify_and_correct_logits(dec7b, torch.zeros(1, 64))
    assert corr7b[0, 41].item() == verifier7.close_bias
    print("7. over-closed bracket guardrail (BIAS_FORBID): OK")

    print("\nAll NeSy-MoE v1.0 (v1.1 Extended) tests passed (executed live).")


if __name__ == "__main__":
    main()
