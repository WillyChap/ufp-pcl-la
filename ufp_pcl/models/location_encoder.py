"""Trainable location-time encoder (GeoCLIP-style).

Design follows Sec. 4.3 of the PCL paper: coordinates are pushed through an equal-area
map projection, expanded with *multi-scale* random Fourier features, and passed through
one small MLP per scale whose outputs are summed.  The multi-scale part matters: a
single bandwidth either blurs neighbourhood-scale structure (too broad) or turns the
encoder into a lookup table over training sites (too fine).  Summing across scales lets
the proxy-consistency loss decide which scales carry signal.

For UFP specifically the fine scales are doing real work -- particle number falls off
sharply within a few hundred metres of a roadway -- so `length_scale_km` should be set
to the size of the air basin, not the continent.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

EARTH_RADIUS_KM = 6371.0


class EqualEarth(nn.Module):
    """Equal Earth projection (Savric et al. 2018), lon/lat in degrees -> km.

    An equal-area projection keeps the RFF bandwidths interpretable as physical
    distances, which is what makes `length_scale_km` mean something.
    """

    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796

    def forward(self, lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
        lon = torch.deg2rad(lon_deg)
        lat = torch.deg2rad(lat_deg.clamp(-89.999, 89.999))
        theta = torch.asin((math.sqrt(3.0) / 2.0) * torch.sin(lat))
        t2 = theta * theta
        denom = 3.0 * (9.0 * self.A4 * t2**4 + 7.0 * self.A3 * t2**3
                       + 3.0 * self.A2 * t2 + self.A1)
        x = 2.0 * math.sqrt(3.0) * lon * torch.cos(theta) / denom
        y = (self.A4 * theta**9 + self.A3 * theta**7 + self.A2 * theta**3 + self.A1 * theta)
        return torch.stack([x, y], dim=-1) * EARTH_RADIUS_KM


class MultiScaleRFF(nn.Module):
    """Random Fourier features at `n_scales` logarithmically spaced bandwidths.

    Frequencies are fixed (registered as buffers, not parameters): the paper's encoder
    learns the MLP on top of a *fixed* random basis, which is what keeps the location
    encoder from collapsing onto training coordinates early in training.
    """

    def __init__(self, in_dim: int, n_scales: int, sigma_min: float, sigma_max: float,
                 n_frequencies: int, seed: int = 0):
        super().__init__()
        self.n_scales = n_scales
        self.n_frequencies = n_frequencies
        g = torch.Generator().manual_seed(seed)
        ratio = 1.0 if n_scales == 1 else (sigma_max / sigma_min) ** (1.0 / (n_scales - 1))
        for s in range(n_scales):
            sigma = sigma_min * (ratio ** s)
            b = torch.randn(in_dim, n_frequencies, generator=g) * sigma
            self.register_buffer(f"B{s}", b)
        self.out_dim = 2 * n_frequencies

    def forward(self, v: torch.Tensor):
        """-> list of `n_scales` tensors, each (..., 2 * n_frequencies)."""
        feats = []
        for s in range(self.n_scales):
            proj = 2.0 * math.pi * (v @ getattr(self, f"B{s}"))
            feats.append(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))
        return feats


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(max(1, n_layers)):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LocationEncoder(nn.Module):
    """(lon, lat) in degrees -> R^dim."""

    def __init__(self, cfg, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.proj = EqualEarth()
        self.length_scale_km = float(cfg.length_scale_km)
        self.rff = MultiScaleRFF(2, cfg.n_scales, cfg.sigma_min, cfg.sigma_max,
                                 cfg.n_frequencies, seed=seed)
        self.branches = nn.ModuleList([
            _mlp(self.rff.out_dim, cfg.hidden, cfg.dim, cfg.n_layers, cfg.dropout)
            for _ in range(cfg.n_scales)
        ])
        self.out_dim = cfg.dim

    def forward(self, lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        xy = self.proj(lon, lat) / self.length_scale_km
        return sum(br(f) for br, f in zip(self.branches, self.rff(xy)))


class TimeEncoder(nn.Module):
    """Day-of-year (and optionally hour-of-day / year) -> R^dim.

    Day-of-year enters as *cyclic* harmonics so that 31 Dec and 1 Jan are adjacent in
    the embedding -- the paper's ablation (Table 3) shows the location->location-time
    upgrade is where most of the in-sample gain comes from.
    """

    def __init__(self, cfg, year0: float = 2020.0, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.year0 = year0
        self.n_harm = cfg.n_harmonics
        self.use_hour = cfg.use_hour
        self.use_year = cfg.use_year

        self.n_hour_harm = getattr(cfg, "n_hour_harmonics", 3)
        self.use_dow = getattr(cfg, "use_dayofweek", False)

        in_dim = 2 * self.n_harm
        self.rff = MultiScaleRFF(1, 2, 1.0, 16.0, cfg.n_frequencies, seed=seed + 7)
        in_dim += 2 * self.rff.out_dim
        if self.use_hour:
            in_dim += 2 * self.n_hour_harm
        if self.use_dow:
            in_dim += 3                    # weekly sin/cos + a weekend indicator
        if self.use_year:
            in_dim += 1
        self.net = _mlp(in_dim, cfg.hidden, cfg.dim, 2, 0.0)
        self.out_dim = cfg.dim

    def forward(self, doy, year, hour, dow=None) -> torch.Tensor:
        phase = 2.0 * math.pi * (doy / 365.25)
        ks = torch.arange(1, self.n_harm + 1, device=doy.device, dtype=doy.dtype)
        ang = phase[:, None] * ks[None, :]
        feats = [torch.sin(ang), torch.cos(ang)]
        feats += self.rff((doy / 365.25).unsqueeze(-1))
        if self.use_hour:
            hk = torch.arange(1, self.n_hour_harm + 1, device=hour.device, dtype=hour.dtype)
            hang = (2.0 * math.pi * hour / 24.0)[:, None] * hk[None, :]
            feats += [torch.sin(hang), torch.cos(hang)]
        if self.use_dow:
            if dow is None:
                dow = torch.zeros_like(doy)
            w = 2.0 * math.pi * (dow / 7.0)
            weekend = ((dow >= 5).to(doy.dtype))
            feats.append(torch.stack([torch.sin(w), torch.cos(w), weekend], dim=-1))
        if self.use_year:
            feats.append(((year - self.year0) / 10.0).unsqueeze(-1))
        return self.net(torch.cat(feats, dim=-1))


class LocationTimeEncoder(nn.Module):
    """The blue branch of Figure 1: coords -> e_loc.

    Input is the coordinate tensor produced by the data layer,
    columns = (lon, lat, day-of-year, year, hour).
    """

    def __init__(self, loc_cfg, time_cfg, year0: float = 2020.0, seed: int = 0):
        super().__init__()
        source = getattr(loc_cfg, "source", "trainable")
        self.source = source

        self.loc: Optional[LocationEncoder] = None
        self.pretrained: Optional[PrecomputedLocationEncoder] = None
        if source in ("trainable", "hybrid"):
            self.loc = LocationEncoder(loc_cfg, seed=seed)
        if source in ("precomputed", "hybrid"):
            from ..data.netcdf import load_embedding_cube

            if not loc_cfg.pretrained_path:
                raise ValueError(f"location.source='{source}' needs location.pretrained_path")
            lat, lon, cube = load_embedding_cube(
                loc_cfg.pretrained_path,
                variables=loc_cfg.pretrained_vars or None,
                prefix=loc_cfg.pretrained_prefix,
            )
            self.pretrained = PrecomputedLocationEncoder(
                lat, lon, cube, out_dim=loc_cfg.dim,
                adapter_hidden=loc_cfg.pretrained_adapter_hidden,
            )

        self.time: Optional[TimeEncoder] = (
            TimeEncoder(time_cfg, year0=year0, seed=seed) if time_cfg.enabled else None
        )
        fuse_in = ((self.loc.out_dim if self.loc else 0)
                   + (self.pretrained.out_dim if self.pretrained else 0)
                   + (self.time.out_dim if self.time else 0))
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, loc_cfg.dim), nn.LayerNorm(loc_cfg.dim), nn.SiLU(),
            nn.Linear(loc_cfg.dim, loc_cfg.dim),
        )
        self.out_dim = loc_cfg.dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """`coords` columns follow ufp_pcl.data.netcdf.COORD_COLS:
        (lon, lat, doy, year, hour, dow, time_days)."""
        lon, lat = coords[:, 0], coords[:, 1]
        parts = []
        if self.loc is not None:
            parts.append(self.loc(lon, lat))
        if self.pretrained is not None:
            parts.append(self.pretrained(lon, lat))
        e = torch.cat(parts, dim=-1)
        if self.time is not None:
            doy, year, hour = coords[:, 2], coords[:, 3], coords[:, 4]
            dow = coords[:, 5] if coords.shape[1] > 5 else None
            e = torch.cat([e, self.time(doy, year, hour, dow)], dim=-1)
        return self.fuse(e)


# --------------------------------------------------------------------------------------
class PrecomputedLocationEncoder(nn.Module):
    """A frozen, externally pretrained embedding field, sampled bilinearly at any point.

    This is how a pretrained Earth/location embedding enters the model without vendoring
    anyone's inference code: rasterise the embedding onto a lat/lon grid once (any of
    SatCLIP, GeoCLIP, Climplicit or an embedding-field product such as AlphaEarth can be
    written this way), store it as a NetCDF, and point the config at it.  The cube is a
    buffer, not a parameter, so it never receives gradient; a small trainable adapter
    projects it into the fusion dimension.

    A caution specific to this project: globally pretrained *location* encoders vary over
    hundreds of kilometres.  Inside a single air basin they supply almost no contrast,
    which is exactly where UFP varies most.  Their value here is as a baseline and,
    where the product is a high-resolution embedding *field*, as observation-encoder
    input rather than as the geographic prior.
    """

    def __init__(self, lat: "np.ndarray", lon: "np.ndarray", cube: "np.ndarray",
                 out_dim: int = 256, adapter_hidden: int = 256, freeze: bool = True,
                 adapter: bool = True):
        super().__init__()
        import numpy as np

        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)
        cube = np.asarray(cube, dtype=np.float32)          # (C, ny, nx)
        if lat[0] > lat[-1]:
            lat, cube = lat[::-1].copy(), cube[:, ::-1, :].copy()
        if lon[0] > lon[-1]:
            lon, cube = lon[::-1].copy(), cube[:, :, ::-1].copy()

        # persistent=False: the cube is a frozen copy of the file on disk, reconstructed
        # on load from `pretrained_path`.  Serialising it into every checkpoint makes a
        # 30 m AlphaEarth run write ~1 GB each time validation improves -- tens of GB
        # across a cross-validation sweep, and slow enough to dominate epoch time.
        self.register_buffer("cube", torch.from_numpy(np.ascontiguousarray(cube))[None],
                             persistent=False)
        self.register_buffer("lat_min", torch.tensor(float(lat[0])))
        self.register_buffer("lat_max", torch.tensor(float(lat[-1])))
        self.register_buffer("lon_min", torch.tensor(float(lon[0])))
        self.register_buffer("lon_max", torch.tensor(float(lon[-1])))
        self.n_bands = int(cube.shape[0])

        if adapter:
            self.adapter = _mlp(self.n_bands, adapter_hidden, out_dim, 1, 0.0)
            if freeze:
                pass          # the cube is frozen either way; the adapter stays trainable
        else:
            self.adapter = nn.Identity()
            out_dim = self.n_bands
        self.out_dim = out_dim

    def forward(self, lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        gx = 2.0 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1.0
        gy = 2.0 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(self.cube, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return self.adapter(sampled[0, :, 0, :].transpose(0, 1))
