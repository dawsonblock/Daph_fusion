"""ActivationBank: activation covariance collection (Phase 14).

For mergeable linear layer l, expert i:
    C_{i,l} = E[x_{i,l} x_{i,l}ᵀ]

Collect pre-linear input activation using forward hooks.

Use hooks:
    def pre_hook(module, args):
        x = args[0].detach()

Apply the attention mask before flattening token positions.
Do not count padding.

Store: sample count, token count, mean, covariance, rank approximation,
condition number estimate.

The K-FAC A factor can serve this role directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ActivationStats:
    """Statistics for an activation covariance collection."""
    sample_count: int
    token_count: int
    input_dim: int
    mean: Optional[Tensor] = None       # [input_dim]
    covariance: Optional[Tensor] = None  # [input_dim, input_dim] or [input_dim] (diag)
    rank_approximation: int = 0
    condition_number_estimate: float = 0.0
    mode: str = "diagonal"               # "full", "diagonal", "low_rank"


class ActivationBank:
    """Collects and stores activation covariance for RegMean and K-FAC.

    For each linear layer, collects pre-input activation covariance
    C = E[xxᵀ] using forward hooks. Padding tokens are excluded.
    """

    def __init__(
        self,
        mode: str = "diagonal",
        max_samples: Optional[int] = None,
        low_rank: int = 64,
    ):
        """
        Args:
            mode: "full", "diagonal", or "low_rank" covariance collection.
            max_samples: Maximum number of samples to process.
            low_rank: Rank for low-rank approximation.
        """
        self.mode = mode
        self.max_samples = max_samples
        self.low_rank = low_rank
        self._hooks: List[Any] = []
        self._current_mask: Optional[Tensor] = None
        self._bank: Dict[str, Dict[str, ActivationStats]] = {}

    def _make_pre_hook(self, expert_name: str, module_name: str, input_dim: int):
        """Create a forward pre-hook for a linear layer."""
        # Accumulators (use lists for mutable closure capture)
        sample_count = [0]
        token_count = [0]
        sum_x = [torch.zeros(input_dim, dtype=torch.float32)]
        sum_xx = [torch.zeros(input_dim, dtype=torch.float32)] if self.mode == "diagonal" else [None]
        sum_xx_full = [torch.zeros(input_dim, input_dim, dtype=torch.float32)] if self.mode == "full" else [None]

        def hook(module, args):
            x = args[0].detach().float()  # [batch, seq, dim] or [batch, dim]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [batch, 1, dim]

            # Apply attention mask if available
            if self._current_mask is not None:
                mask = self._current_mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)  # [batch, seq, 1]
                x_masked = x * mask
                n_tokens = mask.sum().item()
            else:
                x_masked = x
                n_tokens = x.shape[0] * x.shape[1]

            # Flatten batch and sequence dimensions
            x_flat = x_masked.reshape(-1, x.shape[-1])  # [batch*seq, dim]

            if self.max_samples is not None and sample_count[0] >= self.max_samples:
                return

            sample_count[0] += x.shape[0]
            token_count[0] += int(n_tokens)

            # Accumulate (in-place to work with closure)
            sum_x[0].add_(x_flat.sum(dim=0))

            if self.mode == "diagonal":
                sum_xx[0].add_((x_flat ** 2).sum(dim=0))
            elif self.mode == "full":
                sum_xx_full[0].add_(x_flat.t() @ x_flat)

        return hook, (sample_count, token_count, sum_x, sum_xx, sum_xx_full)

    def collect(
        self,
        model: nn.Module,
        expert_name: str,
        dataset: Any,
        forward_fn: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, ActivationStats]:
        """Collect activation covariance for a model's linear layers.

        Args:
            model: The model to collect activations from.
            expert_name: Name to store activations under.
            dataset: Calibration data (iterable of batches).
            forward_fn: Optional custom forward function.
            device: Device for computation.
            layer_names: Optional list of parameter names to collect for.
                        If None, auto-detects linear layers.

        Returns:
            Dict mapping param_name -> ActivationStats.
        """
        model.to(device)
        model.eval()

        # Find linear layers and register hooks
        hooks_data: Dict[str, Tuple[Any, tuple]] = {}
        target_modules: Dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear,)):
                # Determine input dimension
                input_dim = module.in_features

                # Skip if not in target list
                param_name = f"{name}.weight"
                if layer_names is not None and param_name not in layer_names:
                    continue

                hook, accumulators = self._make_pre_hook(expert_name, name, input_dim)
                h = module.register_forward_pre_hook(hook)
                self._hooks.append(h)
                hooks_data[param_name] = accumulators
                target_modules[param_name] = module

        if not hooks_data:
            return {}

        # Run forward passes
        for batch in dataset:
            if self.max_samples is not None:
                total_samples = sum(a[0][0] for a in hooks_data.values())
                if total_samples >= self.max_samples:
                    break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
            else:
                input_ids = batch
                attention_mask = None

            self._current_mask = attention_mask.to(device) if attention_mask is not None else None

            with torch.no_grad():
                if forward_fn is not None:
                    forward_fn(model, input_ids.to(device), attention_mask=self._current_mask)
                else:
                    model(input_ids.to(device), attention_mask=self._current_mask)

        # Finalize statistics
        stats: Dict[str, ActivationStats] = {}
        for param_name, (sample_count, token_count, sum_x, sum_xx, sum_xx_full) in hooks_data.items():
            n = token_count[0]
            if n == 0:
                continue

            mean = sum_x[0] / n

            if self.mode == "diagonal":
                cov = sum_xx[0] / n - mean ** 2
                cov = cov.clamp(min=0)  # Ensure non-negative variance
                rank_approx = int((cov > 1e-10).sum().item())
                cond = (cov.max() / cov[cov > 0].min()).item() if (cov > 0).any() else 0.0
                stats[param_name] = ActivationStats(
                    sample_count=sample_count[0],
                    token_count=n,
                    input_dim=sum_x[0].shape[0],
                    mean=mean,
                    covariance=cov,
                    rank_approximation=rank_approx,
                    condition_number_estimate=cond,
                    mode="diagonal",
                )
            elif self.mode == "full":
                cov = sum_xx_full[0] / n - torch.outer(mean, mean)
                # Ensure PSD
                eigvals = torch.linalg.eigvalsh(cov)
                cov = cov - eigvals.min().clamp(max=0) * torch.eye(cov.shape[0])
                rank_approx = int((eigvals > 1e-10 * eigvals.max()).sum().item())
                cond = (eigvals.max() / eigvals[eigvals > 0].min()).item() if (eigvals > 0).any() else 0.0
                stats[param_name] = ActivationStats(
                    sample_count=sample_count[0],
                    token_count=n,
                    input_dim=sum_x[0].shape[0],
                    mean=mean,
                    covariance=cov,
                    rank_approximation=rank_approx,
                    condition_number_estimate=cond,
                    mode="full",
                )

        # Store in bank
        self._bank[expert_name] = stats

        # Clean up hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._current_mask = None

        return stats

    def get_covariance(self, expert_name: str, param_name: str) -> Optional[Tensor]:
        """Get the activation covariance for a specific expert and parameter."""
        if expert_name in self._bank and param_name in self._bank[expert_name]:
            return self._bank[expert_name][param_name].covariance
        return None

    def to_dict(self) -> Dict[str, Dict[str, Tensor]]:
        """Convert to a dict mapping expert_name -> {param_name -> covariance}.

        This is the format expected by merge_regmean.
        """
        result: Dict[str, Dict[str, Tensor]] = {}
        for expert_name, stats_dict in self._bank.items():
            result[expert_name] = {}
            for param_name, stats in stats_dict.items():
                if stats.covariance is not None:
                    result[expert_name][param_name] = stats.covariance
        return result

    @property
    def expert_names(self) -> List[str]:
        return list(self._bank.keys())
