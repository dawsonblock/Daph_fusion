#!/usr/bin/env python3
"""
DAPH NeSy-MoE Pretrained Model & Neurosymbolic Execution Runner.

This script:
1. Loads 'distilbert/distilgpt2' from Hugging Face.
2. Runs standard neural text generation on sample prompts.
3. Integrates the HF tokenizer with DAPH NeSy-MoE (v1.1 Extended).
4. Demonstrates 5-path predictive macro-routing, Tokenizer Rules Engine,
   Vectorized Symbolic Expert (System 2 domain solvers), and NeSy Decoder Layer execution.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daph_hybrid_exfusion_v2_3 import DAPHConfig
from daph_nesy_v1_0 import (
    TokenizerBoundRulesEngine,
    VectorizedSymbolicExpert,
    NeSyDecoderLayer,
    NeSyOutputVerifier,
)

def run_hf_generation(model_id="distilbert/distilgpt2"):
    print("=" * 70)
    print(f"1. LOADING HUGGING FACE PRETRAINED MODEL: '{model_id}'")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    prompts = [
        "The fundamental law of physics states that",
        "To solve the equation 2 * x + 5 = 15, we first",
    ]

    print("\n--- Running Neural Generation with Pretrained Model ---")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated Output:\n{generated_text}")

    return tokenizer, model

def run_nesy_moe_integration(tokenizer):
    print("\n" + "=" * 70)
    print("2. INTEGRATING WITH DAPH NeSy-MoE v1.1 DUAL SYSTEM ENGINE")
    print("=" * 70)

    # Initialize Tokenizer-Bound Rules Engine with Hugging Face Tokenizer
    rules_engine = TokenizerBoundRulesEngine(
        num_paths=5,
        tokenizer=tokenizer,
    )
    print("-> TokenizerBoundRulesEngine bound to HF Tokenizer successfully.")

    # Configure DAPH Hybrid NeSy System (5-path macro routing)
    daph_config = DAPHConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=8,
        num_paths=5,
        state_size=16,
    )

    vocab_size = len(tokenizer)
    lm_head_weight = torch.randn(vocab_size, daph_config.hidden_size)

    # Initialize Symbolic Expert
    expert = VectorizedSymbolicExpert(
        hidden_size=daph_config.hidden_size,
        vocab_size=vocab_size,
        lm_head_weight=lm_head_weight,
        domain="arithmetic_eval",
    )
    print("-> VectorizedSymbolicExpert (Domain Solvers: Arithmetic, SAT, AST) initialized.")

    # Initialize NeSy Decoder Layer (System 1 Mamba/Attention + System 2 Symbolic)
    nesy_layer = NeSyDecoderLayer(
        config=daph_config,
        rules_engine=rules_engine,
        symbolic_expert=expert,
    )
    print("-> NeSyDecoderLayer (5-Path Macro Router) initialized.")

    # Test sample prompts requiring reasoning / math
    test_inputs = [
        "solve: 45 * 12 =",
        "sat: (A or B) and (not A)",
        "eval: 3 ** 2 + 4 ** 2",
    ]

    print("\n--- Running Dual-System (System 1 + System 2) Forward Pass ---")
    for text in test_inputs:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"] # (1, L)
        batch_size, seq_len = input_ids.shape

        # Create continuous embeddings for input
        dummy_embeddings = torch.randn(batch_size, seq_len, daph_config.hidden_size)

        # Run forward pass through NeSyDecoderLayer
        out, meta = nesy_layer(
            dummy_embeddings,
            token_ids=input_ids,
        )

        print(f"\nInput: '{text}' (Tokens: {input_ids.tolist()[0]})")
        print(f" -> Output Representation Shape: {out.shape}")
        if "aux_loss" in meta and meta["aux_loss"] is not None:
            print(f" -> Aux Loss (Router Balance): {meta['aux_loss'].item():.6f}")

        # Run direct symbolic expert execution
        expert_out = expert(dummy_embeddings)
        print(f" -> Symbolic Expert Forward Output Shape: {expert_out.shape}")

def main():
    tokenizer, model = run_hf_generation("distilbert/distilgpt2")
    run_nesy_moe_integration(tokenizer)
    print("\n" + "=" * 70)
    print("RUN COMPLETE: Pretrained model & NeSy-MoE engine executed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
