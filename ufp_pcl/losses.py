"""Prediction loss and the proxy consistency loss (Eq. 1 of the paper)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE that skips NaN targets rather than poisoning the whole batch."""
    m = torch.isfinite(target)
    if not bool(m.any()):
        return pred.sum() * 0.0
    return ((pred[m] - target[m]) ** 2).mean()


class ProxyConsistencyLoss(nn.Module):
    r"""L^{proxy-consistency} = (\hat z - z)^\top \Lambda (\hat z - z).

    `weights` is the diagonal of \Lambda, one entry per proxy variable, letting a
    task-aligned proxy (e.g. a reanalysis of the pollutant itself) be weighted above
    merely physically-related auxiliaries (boundary-layer height, wind, radiation).
    The paper's ablation finds a task-aligned proxy is what buys extrapolation; the
    auxiliaries help mainly in-sample.
    """

    def __init__(self, weights: Optional[Sequence[float]] = None, n_proxy: int = 1):
        super().__init__()
        w = torch.ones(n_proxy) if not weights else torch.as_tensor(list(weights),
                                                                    dtype=torch.float32)
        if w.numel() != n_proxy:
            raise ValueError(f"got {w.numel()} proxy weights for {n_proxy} proxy variables")
        self.register_buffer("weights", w)

    def forward(self, z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m = torch.isfinite(z)
        if not bool(m.any()):
            return z_hat.sum() * 0.0
        se = (z_hat - torch.nan_to_num(z)) ** 2
        w = self.weights.to(se.dtype).unsqueeze(0)
        num = (se * w * m).sum()
        den = (w.expand_as(se) * m).sum().clamp_min(1.0)
        return num / den

    @torch.no_grad()
    def per_variable(self, z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Unweighted MSE per proxy variable -- useful for diagnosing which proxy the
        location encoder can actually reconstruct."""
        m = torch.isfinite(z)
        se = (z_hat - torch.nan_to_num(z)) ** 2 * m
        return se.sum(0) / m.sum(0).clamp_min(1)
