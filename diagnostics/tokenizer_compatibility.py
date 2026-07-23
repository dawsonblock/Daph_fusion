"""
Tokenizer & Vocabulary Compatibility Diagnostics (Phase 2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def audit_tokenizer_compatibility(
    model: nn.Module,
    tokenizer: Any,
) -> Dict[str, Any]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    model_vocab = getattr(model.config, "vocab_size", None)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    in_shape = (
        tuple(input_embeddings.weight.shape) if input_embeddings is not None else None
    )
    out_shape = (
        tuple(output_embeddings.weight.shape) if output_embeddings is not None else None
    )

    compatible = (
        vocab_size == model_vocab
        and (in_shape is None or in_shape[0] == vocab_size)
        and (out_shape is None or out_shape[0] == vocab_size)
    )

    return {
        "tokenizer_vocab_size": vocab_size,
        "model_vocab_size": model_vocab,
        "input_embeddings_shape": in_shape,
        "output_embeddings_shape": out_shape,
        "compatible": compatible,
    }
