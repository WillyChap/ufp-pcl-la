"""Observation encoders: the green branch's input side.

Two backbones, matching the two shapes Earth-observation predictors arrive in:

  * `MLPObsEncoder`          -- one feature vector per observation (seq_len == 1).
  * `BiLSTMAttentionEncoder` -- a (seq_len, n_feat) window with a bidirectional LSTM and
    Luong attention, the backbone Wang et al. use for daily PM2.5 and the right choice
    when antecedent meteorology matters (it does for UFP: nucleation depends on the
    preceding days' temperature, radiation and boundary-layer history).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPObsEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int = 256, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x):
        if x.dim() == 3:            # (B, T, F) -> flatten a trivially short window
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class LuongAttention(nn.Module):
    """General (bilinear) Luong attention pooling over the sequence axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, seq: torch.Tensor, query: torch.Tensor):
        scores = torch.bmm(self.W(seq), query.unsqueeze(-1)).squeeze(-1)   # (B, T)
        w = torch.softmax(scores, dim=1)
        return torch.bmm(w.unsqueeze(1), seq).squeeze(1), w


class BiLSTMAttentionEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int = 256, lstm_hidden: int = 128,
                 lstm_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn = LuongAttention(2 * lstm_hidden)
        self.proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, dim), nn.LayerNorm(dim), nn.SiLU()
        )
        self.out_dim = dim
        self.last_attention = None

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq, _ = self.lstm(x)
        pooled, w = self.attn(seq, seq[:, -1, :])     # query = final (most recent) step
        self.last_attention = w.detach()
        return self.proj(pooled)


def build_obs_encoder(cfg, in_dim: int, seq_len: int = 1):
    kind = cfg.kind
    if kind == "none" or in_dim == 0:
        return None
    if kind == "auto":
        kind = "bilstm" if seq_len > 1 else "mlp"
    if kind == "mlp":
        return MLPObsEncoder(in_dim * (seq_len if seq_len > 1 else 1), cfg.dim,
                             cfg.hidden, cfg.n_layers, cfg.dropout)
    if kind == "bilstm":
        return BiLSTMAttentionEncoder(in_dim, cfg.dim, cfg.lstm_hidden,
                                      cfg.lstm_layers, cfg.dropout)
    raise ValueError(f"unknown observation encoder '{kind}'")
