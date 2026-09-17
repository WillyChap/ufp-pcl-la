"""The fused model of Figure 1.

    e_obs = ObsEncoder(x)
    e_loc = LocationTimeEncoder(lon, lat, doy, year, hour)
    y_hat = PredHead(concat(e_obs, e_loc))        <- prediction loss flows through both
    z_hat = ProxyHead(e_loc)                      <- proxy loss flows through e_loc ONLY

The asymmetry is deliberate and is the whole mechanism: because the proxy branch never
touches the observation encoder, proxy points can be drawn anywhere in the domain
without needing observation predictors or labels there, and the resulting gradient acts
purely as a spatial regulariser on the location encoder.
"""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

from .heads import MLPHead
from .location_encoder import LocationTimeEncoder
from .obs_encoder import build_obs_encoder


class ProxyGroundedFusion(nn.Module):
    def __init__(
        self,
        n_obs_features: int,
        n_proxy: int,
        cfg,
        seq_len: int = 1,
        year0: float = 2020.0,
        seed: int = 0,
    ):
        super().__init__()
        self.variant = cfg.variant
        self.use_loc = cfg.variant != "obs_only"
        self.use_proxy_head = cfg.variant in ("pcl", "proxy_pretrain", "climo_anomaly")

        self.obs_encoder = build_obs_encoder(cfg.obs, n_obs_features, seq_len)
        obs_dim = self.obs_encoder.out_dim if self.obs_encoder is not None else 0

        self.loc_encoder: Optional[LocationTimeEncoder] = None
        loc_dim = 0
        if self.use_loc:
            self.loc_encoder = LocationTimeEncoder(cfg.location, cfg.time, year0=year0, seed=seed)
            loc_dim = self.loc_encoder.out_dim

        if obs_dim + loc_dim == 0:
            raise ValueError("model has neither an observation encoder nor a location encoder")

        self.pred_head = MLPHead(obs_dim + loc_dim, 1, cfg.pred_head.hidden,
                                 cfg.pred_head.n_layers, cfg.pred_head.dropout)
        self.proxy_head: Optional[MLPHead] = None
        if self.use_proxy_head and n_proxy > 0 and self.loc_encoder is not None:
            self.proxy_head = MLPHead(loc_dim, n_proxy, cfg.proxy_head.hidden,
                                      cfg.proxy_head.n_layers, cfg.proxy_head.dropout)

        self.obs_dim, self.loc_dim = obs_dim, loc_dim

    # ------------------------------------------------------------------ forward passes
    def embed(self, obs: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        if self.obs_encoder is not None:
            out["e_obs"] = self.obs_encoder(obs)
        if self.loc_encoder is not None:
            out["e_loc"] = self.loc_encoder(coords)
        parts = [v for k, v in (("e_obs", out.get("e_obs")), ("e_loc", out.get("e_loc")))
                 if v is not None]
        out["fused"] = torch.cat(parts, dim=-1)
        return out

    def forward(self, obs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.pred_head(self.embed(obs, coords)["fused"])

    def forward_proxy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.proxy_head is None or self.loc_encoder is None:
            raise RuntimeError("this variant has no proxy head")
        return self.proxy_head(self.loc_encoder(coords))

    @torch.no_grad()
    def location_embedding(self, coords: torch.Tensor) -> torch.Tensor:
        """e_loc for latent-space analysis (no gradient, eval mode assumed)."""
        if self.loc_encoder is None:
            raise RuntimeError("this variant has no location encoder")
        return self.loc_encoder(coords)

    # ------------------------------------------------------------------ parameter sets
    def location_parameters(self) -> Iterator[nn.Parameter]:
        if self.loc_encoder is not None:
            yield from self.loc_encoder.parameters()
        if self.proxy_head is not None:
            yield from self.proxy_head.parameters()

    def freeze_location(self, freeze: bool = True) -> None:
        if self.loc_encoder is not None:
            for p in self.loc_encoder.parameters():
                p.requires_grad_(not freeze)


def build_model(cfg, n_obs_features: int, n_proxy: int, seq_len: int = 1,
                year0: float = 2020.0, seed: int = 0) -> ProxyGroundedFusion:
    return ProxyGroundedFusion(n_obs_features, n_proxy, cfg.model, seq_len=seq_len,
                               year0=year0, seed=seed)
