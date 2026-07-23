#!/usr/bin/env python3
"""
DAPH ExFusion Model Merging Pipeline - Real Hugging Face Model Merging Runner.

This script:
1. Downloads 3 distinct fine-tuned expert models + 1 base model from Hugging Face Hub:
   - Base Model: 'distilbert/distilgpt2'
   - Expert 1:   'postbot/distilgpt2-emailgen' (Email & Professional Writing)
   - Expert 2:   'FredZhang7/distilgpt2-stable-diffusion' (Prompt & Image Generation)
   - Expert 3:   'misterkilgore/distilgpt2-psy-ita' (Psychology & Dialogue)
2. Clones the base model to create an empty target container (`merged_model`).
3. Executes the DAPH ExFusion Multi-Stage Model Merging Pipeline:
   - Task Vector Extraction (Δ = W_expert - W_base)
   - DARE Preprocessing (stochastic parameter dropping)
   - TIES v2 Pure Sign-Majority Consensus Voting
   - Empirical Fisher Diagonal Weighting (using empirical calibration batch)
   - TIES + Fisher Parameter Delta Blending
   - Delta Application using provisional scale λ=0.35 (selected from initial qualitative sweep)
4. Generates text across all individual experts, the base model, and the unified merged model.
"""

import copy
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daph_hybrid_exfusion_v2_3 import merge_expert_family

def generate_sample(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("=" * 80)
    print("1. DOWNLOADING & LOADING HUGGING FACE MODELS FROM HUB")
    print("=" * 80)

    base_id = "distilbert/distilgpt2"
    expert_ids = [
        "postbot/distilgpt2-emailgen",
        "FredZhang7/distilgpt2-stable-diffusion",
        "misterkilgore/distilgpt2-psy-ita",
    ]

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[+] Loading Base Model: '{base_id}'...")
    base_model = AutoModelForCausalLM.from_pretrained(base_id)
    base_model.eval()

    experts = []
    for i, exp_id in enumerate(expert_ids, 1):
        print(f"[+] Loading Expert {i}: '{exp_id}'...")
        exp_model = AutoModelForCausalLM.from_pretrained(exp_id)
        exp_model.eval()
        experts.append(exp_model)

    print("\n[✓] All 3 Hugging Face Expert Models and Base Model Loaded Successfully.")

    print("\n" + "=" * 80)
    print("2. EXECUTING DAPH EXFUSION MODEL MERGING PIPELINE")
    print("   (DARE Preprocessing -> TIES v2 Sign Election -> Fisher Diagonal Weighting)")
    print("=" * 80)

    # Prepare target model container (clone of base model)
    merged_model = copy.deepcopy(base_model)
    merged_model.eval()

    # Define memory bank weights for 3 experts
    memory_bank_weights = torch.tensor([1.0, 1.0, 1.0])

    # Build empirical calibration batch for Empirical Fisher Diagonal Weighting
    calib_samples = [
        "Dear Team, I am writing to share an update regarding the project.",
        "A highly detailed digital painting of a futuristic city, 8k, trending on artstation.",
        "In psychological terms, emotional resilience is defined as the capacity to adapt.",
    ]
    calibration_batch = tokenizer(calib_samples, return_tensors="pt", padding=True)

    print("\n[+] Merging expert family deltas into target model container...")
    merged_deltas = merge_expert_family(
        experts=experts,
        base_model=base_model,
        memory_bank_weights=memory_bank_weights,
        calibration_batch=calibration_batch,
        apply_to=merged_model,
        scale=0.35,
        policies={
            "dare_base_p": 0.25,
            "ties_trim_ratio": 0.25,
            "ties_fisher_blend": 0.50,
        },
    )

    print(f"[✓] ExFusion Merging Complete. Applied merged deltas across {len(merged_deltas)} parameter tensors.")

    print("\n" + "=" * 80)
    print("3. GENERATION COMPARISON ACROSS EXPERTS VS. UNIFIED MERGED MODEL")
    print("=" * 80)

    test_prompts = [
        ("Email / Business Writing", "Dear Team,\nI am writing to share an update regarding"),
        ("Art / Stable Diffusion Prompt", "A highly detailed digital painting of a futuristic city with"),
        ("Psychology & Dialogue", "In psychological terms, emotional resilience is defined as"),
    ]

    for category, prompt in test_prompts:
        print(f"\n{'-'*70}")
        print(f"Category: {category}")
        print(f"Prompt: '{prompt}'")
        print(f"{'-'*70}")

        print("\n -> Base Model (distilgpt2):")
        print("    " + generate_sample(base_model, tokenizer, prompt).replace("\n", "\n    "))

        print("\n -> Expert 1 (Email Gen):")
        print("    " + generate_sample(experts[0], tokenizer, prompt).replace("\n", "\n    "))

        print("\n -> Expert 2 (Stable Diffusion Art):")
        print("    " + generate_sample(experts[1], tokenizer, prompt).replace("\n", "\n    "))

        print("\n -> Expert 3 (Psychology & Dialogue):")
        print("    " + generate_sample(experts[2], tokenizer, prompt).replace("\n", "\n    "))

        print("\n -> UNIFIED MERGED MODEL (DAPH ExFusion Merged):")
        print("    " + generate_sample(merged_model, tokenizer, prompt).replace("\n", "\n    "))

    print("\n" + "=" * 80)
    print("SUCCESS: 3 Hugging Face Models Downloaded & Merged via DAPH ExFusion Pipeline!")
    print("=" * 80)

if __name__ == "__main__":
    main()
