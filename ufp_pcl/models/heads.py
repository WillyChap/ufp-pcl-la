"""Small MLP heads: one for the primary task, one for the proxy targets."""
from __future__ import annotations

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
