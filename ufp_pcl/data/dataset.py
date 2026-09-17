"""Turning a NetCDF file into (labelled points, proxy sampler) for PCL training.

Two data streams come out of here, and keeping them separate is the whole trick of the
method:

  1. `LabeledPointDataset` -- the sparse UFP observations with their predictor stack.
     These drive the prediction loss and are limited to wherever monitors exist.
  2. `ProxySampler`        -- proxy field values at points drawn uniformly at random
     over the whole space-time domain.  These drive the proxy-consistency loss and are
     unconstrained by label availability, which is what lets the location encoder be
     supervised "everywhere".
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..climo import ClimoModel
from ..features import compute_derived
from .netcdf import (
    COORD_COLS,
    FieldStack,
    NetCDFSource,
    calendar_features,
    days_to_datetime64,
    solar_elevation_deg,
)
from .scaling import Standardizer, apply_transform
from .splits import make_split, split_report


# --------------------------------------------------------------------------------------
def sample_stacks(
    stacks: List[FieldStack],
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: Optional[np.ndarray],
    interp: str = "bilinear",
    valid_ranges: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Sample several grids (possibly at different resolutions) at the same points."""
    cols, names = [], []
    for st in stacks:
        cols.append(st.sample(lon, lat, None if st.is_static else time_days, interp=interp))
        names.extend(st.var_names)
    if not cols:
        return np.zeros((len(lon), 0), np.float32), names
    out = np.concatenate(cols, axis=1)
    return apply_valid_ranges(out, names, valid_ranges), names


def apply_valid_ranges(
    arr: np.ndarray, names: Sequence[str],
    valid_ranges: Optional[Dict[str, Sequence[float]]],
) -> np.ndarray:
    """NaN sampled values outside their configured physical bounds.

    Screening after sampling rather than in the file keeps the source data untouched and
    means a threshold can be revised without regenerating anything.  The trailing axis
    must be the variable axis, which holds for both (n, n_var) and (n, seq, n_var).
    """
    if not valid_ranges:
        return arr
    for j, n in enumerate(names):
        rng = valid_ranges.get(n)
        if rng is None:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        col = arr[..., j]
        arr[..., j] = np.where(np.isfinite(col) & (col >= lo) & (col <= hi), col, np.nan)
    return arr


def climo_features(sources, wanted: Sequence[str], lon, lat, tdays, interp="bilinear",
                   valid_ranges=None):
    """Sample the climatology's static predictors, independently of the obs encoder.

    Deliberately a separate sampling path rather than a slice of the observation stack.
    A static, site-distinguishing field handed to the observation encoder is a
    seven-monitor fingerprint sitting in the one branch no regulariser reaches -- the
    failure mode this whole decomposition exists to remove.  Keeping the climatology's
    inputs out of `obs_vars` means they inform the spatial mean and nothing else.

    A `log:` prefix fits on log10(x+1); traffic intensity spans orders of magnitude and
    is meaningless raw.
    """
    if not wanted:
        return np.zeros((len(lon), 0)), []
    plain, logged = [], set()
    for w in wanted:
        n = w[4:] if w.startswith("log:") else w
        if w.startswith("log:"):
            logged.add(n)
        plain.append(n)
    stacks = build_stacks(sources, plain)
    X, names = sample_stacks(stacks, lon, lat, tdays, interp, valid_ranges=valid_ranges)
    out, cols = [], []
    for n in plain:
        v = X[:, names.index(n)].astype(np.float64)
        out.append(np.log10(np.clip(v, 0, None) + 1.0) if n in logged else v)
        cols.append(f"log:{n}" if n in logged else n)
    return np.column_stack(out), cols


def resolve_proxy_weights(spec, names):
    """Proxy weights in the order the sampler actually returns variables.

    Accepts a {name: weight} mapping (safe) or a positional list (legacy).  The mapping
    form exists because proxy variables are materialised grouped by grid and in file
    order, which is rarely the order they were written in the config -- a positional list
    therefore applies the weights to the wrong fields, silently, and the run looks fine.
    """
    if not spec:
        return None
    if isinstance(spec, dict):
        missing = [n for n in spec if n not in names]
        if missing:
            raise KeyError(f"proxy_weights names not among proxies {list(names)}: {missing}")
        return [float(spec.get(n, 1.0)) for n in names]
    if len(spec) != len(names):
        raise ValueError(f"{len(spec)} proxy weights for {len(names)} proxies {list(names)}")
    return [float(v) for v in spec]


# --------------------------------------------------------------------------------------
class ProxySampler:
    """Uniform-at-random space-time sampling of the proxy fields.

    `rho` in the paper is the ratio |B_proxy| / |B_sup|; we simply draw `n` points per
    call and the trainer sets n = rho * batch_size.  Sampling is rejection-based against
    the fields' valid mask so that ocean / out-of-domain cells never enter the loss.
    """

    def __init__(
        self,
        stacks: List[FieldStack],
        transforms: Dict[str, str],
        interp: str = "bilinear",
        seed: int = 0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        time_days: Optional[np.ndarray] = None,
        valid_ranges: Optional[Dict[str, Sequence[float]]] = None,
        require_all: bool = False,
    ):
        if not stacks:
            raise ValueError("no proxy variables configured -- PCL needs at least one")
        self.stacks = stacks
        self.interp = interp
        self.rng = np.random.default_rng(seed)
        self.valid_ranges = valid_ranges or {}
        self.require_all = bool(require_all)
        self._accept = 1.0            # running acceptance rate of the rejection sampler
        self.var_names = [n for st in stacks for n in st.var_names]
        self.transforms = [transforms.get(n, "none") for n in self.var_names]

        if bounds is None:
            b = np.array([st.bounds for st in stacks], dtype=float)
            bounds = (b[:, 0].max(), b[:, 1].min(), b[:, 2].max(), b[:, 3].min())
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = bounds

        if time_days is None:
            tv = [st.time_days for st in stacks if st.time_days is not None]
            time_days = tv[0] if tv else None
        self.time_days = time_days
        self.scaler: Optional[Standardizer] = None

    # ---------------------------------------------------------------------------------
    def _raw(self, lon, lat, t) -> np.ndarray:
        z, _ = sample_stacks(self.stacks, lon, lat, t, interp=self.interp,
                             valid_ranges=self.valid_ranges)
        for j, kind in enumerate(self.transforms):
            if kind != "none":
                z[:, j] = apply_transform(z[:, j], kind)
        return z

    def fit_scaler(self, n: int = 20000) -> Standardizer:
        """Fit the proxy standardiser on a UAR draw, so proxy targets are O(1)."""
        lon, lat, t, z = self.draw_raw(n)
        self.scaler = Standardizer().fit(z)
        return self.scaler

    def draw_raw(self, n: int, max_tries: int = 12):
        """Rejection-sample `n` valid space-time points and their proxy values.

        The oversampling factor adapts to the observed acceptance rate.  A fixed 2x is
        fine for a proxy that covers most of the domain, but a cloud-screened, daylight-
        only field like GOES AOD is valid in ~13% of space-time cells, and a fixed factor
        then returns short batches -- quietly lowering the effective rho.
        """
        lons, lats, ts, zs = [], [], [], []
        got = 0
        for _ in range(max_tries):
            over = 1.0 / max(self._accept, 0.02)
            m = int(min(max(64, (n - got) * over * 1.3), 20 * n + 64))
            lo = self.rng.uniform(self.lon_min, self.lon_max, m)
            la = self.rng.uniform(self.lat_min, self.lat_max, m)
            if self.time_days is not None and len(self.time_days) > 0:
                tt = self.time_days[self.rng.integers(0, len(self.time_days), m)]
            else:
                tt = np.zeros(m)
            zz = self._raw(lo, la, tt)
            # Accept a point if ANY proxy is retrievable there.  The consistency loss is
            # NaN-masked per element, so a partially-observed point still contributes the
            # variables it has; requiring all of them would shrink a multi-proxy draw to
            # the intersection of their coverage -- with cloud-screened satellite fields
            # that is far smaller than any one of them, and biased toward clear-sky.
            fin = np.isfinite(zz)
            ok = fin.all(axis=1) if self.require_all else fin.any(axis=1)
            # exponential moving average of the acceptance rate, seeded optimistically
            self._accept = 0.7 * self._accept + 0.3 * float(max(ok.mean(), 1e-3))
            if ok.any():
                lons.append(lo[ok]); lats.append(la[ok]); ts.append(tt[ok]); zs.append(zz[ok])
                got += int(ok.sum())
            if got >= n:
                break
        if got == 0:
            raise RuntimeError("proxy rejection sampling found no valid points in the domain")
        lon = np.concatenate(lons)[:n]; lat = np.concatenate(lats)[:n]
        t = np.concatenate(ts)[:n]; z = np.concatenate(zs)[:n]
        return lon, lat, t, z

    def draw(self, n: int):
        """-> (coords (n,3) float64 lon/lat/time_days, z (n,m) standardised float32)."""
        lon, lat, t, z = self.draw_raw(n)
        if self.scaler is not None:
            z = self.scaler.transform(z)
        return np.stack([lon, lat, t], 1), z.astype(np.float32)

    def at(self, lon, lat, t):
        """Proxy values at supplied coordinates (used by `proxy_at_label_sites`)."""
        z = self._raw(np.asarray(lon), np.asarray(lat), np.asarray(t))
        if self.scaler is not None:
            z = self.scaler.transform(z)
        return z.astype(np.float32)


# --------------------------------------------------------------------------------------
class LabeledPointDataset(Dataset):
    """Pre-materialised labelled samples: predictors, coordinates and target.

    `climo` carries the per-sample climatological offset that was subtracted from the
    target, in transform space.  It travels with the samples so that metrics can be
    reported on the reconstructed field rather than on the residual the model fits.
    """

    def __init__(self, obs: np.ndarray, coords: np.ndarray, y: np.ndarray,
                 climo: Optional[np.ndarray] = None):
        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.coords = torch.as_tensor(coords, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.climo = (np.zeros(len(self.y), np.float64) if climo is None
                      else np.asarray(climo, np.float64).ravel())

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.coords[i], self.y[i]


# --------------------------------------------------------------------------------------
@dataclass
class DataBundle:
    """Everything the trainer and the latent-analysis tools need."""

    train: LabeledPointDataset
    val: LabeledPointDataset
    test: LabeledPointDataset
    proxy: ProxySampler
    obs_names: List[str]
    proxy_names: List[str]
    obs_scaler: Standardizer
    y_scaler: Standardizer
    y_transform: str
    coord_meta: Dict[str, float]
    masks: Dict[str, np.ndarray]
    table: Dict[str, np.ndarray]
    seq_len: int = 1
    climo: Optional["ClimoModel"] = None
    report: str = ""
    extra_sets: Dict[str, "LabeledPointDataset"] = field(default_factory=dict)

    @property
    def n_obs_features(self) -> int:
        return self.train.obs.shape[-1]

    @property
    def n_proxy(self) -> int:
        return len(self.proxy_names)


# --------------------------------------------------------------------------------------
def open_sources(cfg: Config) -> List[NetCDFSource]:
    """The primary file first, then any auxiliary files, each keeping its own grids."""
    srcs = [NetCDFSource(cfg.data.path, cfg.data.coords, cfg.data.target)]
    for extra in cfg.data.aux_paths:
        srcs.append(NetCDFSource(extra, cfg.data.coords, cfg.data.target))
    return srcs


def build_stacks(sources: List[NetCDFSource], var_names) -> List[FieldStack]:
    """Materialise the requested variables from whichever source holds each one."""
    remaining = list(var_names)
    stacks: List[FieldStack] = []
    for src in sources:
        available = {n for names in src.gridded_vars().values() for n in names}
        take = [n for n in remaining if n in available]
        if take:
            stacks.extend(src.build_stack(take))
            remaining = [n for n in remaining if n not in available]
    if remaining:
        raise KeyError(f"variables not found on any lat/lon grid: {sorted(remaining)}")
    return stacks


def resolve_variable_roles(cfg: Config, src):
    """Decide which variables feed the observation encoder and which are proxy targets.

    Shared with the mapping code so a prediction surface is built from exactly the same
    variables, in the same order, as the model was trained on.
    """
    dc = cfg.data
    sources = src if isinstance(src, list) else [src]
    proxy_vars = list(dc.proxy_vars)
    if dc.obs_vars == "auto":
        gridded = [n for s in sources for names in s.gridded_vars().values() for n in names]
        obs_vars = [n for n in gridded
                    if n not in proxy_vars and n != dc.target.var
                    and not n.startswith("truth_")]
    else:
        obs_vars = list(dc.obs_vars)
    if cfg.model.variant == "proxy_stacked":
        # baseline: hand the proxy fields to the observation encoder as extra bands
        obs_vars = obs_vars + [v for v in proxy_vars if v not in obs_vars]
    return obs_vars, proxy_vars


def build_datasets(cfg: Config, verbose: bool = True) -> DataBundle:
    dc = cfg.data
    sources = open_sources(cfg)
    src = sources[0]

    table = src.target_table()
    lon, lat, tdays, y = table["lon"], table["lat"], table["time_days"], table["y"]

    obs_vars, proxy_vars = resolve_variable_roles(cfg, sources)

    obs_stacks = build_stacks(sources, obs_vars) if obs_vars else []
    proxy_stacks = build_stacks(sources, proxy_vars) if proxy_vars else []

    # --------------------------------------------------------- predictors at the labels
    seq = max(1, int(dc.seq_len))
    if seq == 1:
        obs, obs_names = sample_stacks(obs_stacks, lon, lat, tdays, dc.interp,
                                       valid_ranges=dc.valid_ranges)
    else:
        # a backward-looking window ending on the observation day
        frames = []
        for k in range(seq - 1, -1, -1):
            f, obs_names = sample_stacks(obs_stacks, lon, lat, tdays - k, dc.interp,
                                         valid_ranges=dc.valid_ranges)
            frames.append(f)
        obs = np.stack(frames, axis=1)                      # (n, seq, n_feat)

    if dc.derived:
        obs, obs_names = compute_derived(obs, obs_names, lon, lat, tdays, dc.derived)

    keep = np.isfinite(y)
    if int(dc.time_stride) > 1:
        # thin by rank within each site, so every site is thinned the same way
        thin = np.zeros(len(y), bool)
        sid_all = table["site_id"]          # site_id is only subset later, below
        for s_id in np.unique(sid_all):
            idx = np.flatnonzero(sid_all == s_id)
            idx = idx[np.argsort(tdays[idx])]
            thin[idx[::int(dc.time_stride)]] = True
        keep &= thin
    if dc.time_range:
        # Harmonised collections often pad short records by repeating the final field.
        # Those rows look like data and would land in the temporal test set, so trim the
        # labelled samples to the window every source genuinely covers.
        lo, hi = (np.datetime64(dc.time_range[0]), np.datetime64(dc.time_range[1]))
        tt = days_to_datetime64(tdays)
        keep &= (tt >= lo) & (tt <= hi)
    if dc.daylight_only:
        # TEMPO only retrieves in daylight; training on night hours feeds the observation
        # encoder fill values and lets it learn "night = missing" instead of chemistry.
        keep &= solar_elevation_deg(tdays, lon, lat) >= dc.solar_elevation_min
    if dc.drop_nan_obs and obs.shape[-1] > 0:
        axes = tuple(range(1, obs.ndim))
        keep &= np.isfinite(obs).all(axis=axes)
    dropped = int((~keep).sum())
    lon, lat, tdays, y = lon[keep], lat[keep], tdays[keep], y[keep]
    obs = obs[keep]
    site_id = table["site_id"][keep]
    table = {"lon": lon, "lat": lat, "time_days": tdays, "y": y, "site_id": site_id}

    # ------------------------------------------------------------------------- splits
    masks = make_split(
        lon, lat, site_id, kind=cfg.split.kind, time_days=tdays,
        holdout_site=cfg.split.holdout_site, test_frac=cfg.split.test_frac,
        val_frac=cfg.split.val_frac, seed=cfg.split.seed, delta=cfg.split.delta,
        offset_index=cfg.split.offset_index, swap=cfg.split.swap,
        block_days=cfg.split.block_days,
    )

    # ------------------------------------------------------------- scaling (train only)
    obs_scaler = Standardizer()
    if obs.shape[-1] > 0:
        obs_scaler.fit(obs[masks["train"]].reshape(-1, obs.shape[-1]))
        obs = obs_scaler.transform(obs)
    else:
        obs_scaler.mean = np.zeros(0); obs_scaler.std = np.ones(0)

    y_t = apply_transform(y, dc.target.transform)

    # ------------------------------------------------- explicit spatial climatology
    # Fitted on TRAINING sites only.  Under leave-one-site-out the held-out monitor must
    # contribute nothing to its own climatology, or the protocol measures nothing.
    climo_model, climo_off = None, np.zeros(len(y_t))
    if cfg.model.variant == "climo_anomaly":
        cf, cn = climo_features(sources, dc.climo_vars, lon, lat, tdays,
                                dc.interp, dc.valid_ranges)
        if cf.shape[1] == 0:
            raise ValueError(
                "variant 'climo_anomaly' needs data.climo_vars, e.g. "
                "['log:truck5_intensity', 'road_dist_km'] -- these are sampled separately "
                "and must NOT also appear in data.obs_vars"
            )
        tr = masks["train"]
        CF, CN, CY, CS = cf[tr], cn, y_t[tr], site_id[tr]
        if dc.climo_extra:
            # sampled through the same path, so the predictors are identical in
            # construction; only the target comes from elsewhere
            ex = xr.open_dataset(dc.climo_extra)
            elat = np.asarray(ex["latitude"].values, float)
            elon = np.asarray(ex["longitude"].values, float)
            ey = np.log10(np.clip(np.asarray(ex[dc.climo_extra_var].values, float), 1, None))
            ex.close()
            # drop any extra site that duplicates a training monitor
            keep_e = np.array([
                np.min(np.hypot((lat[tr] - a) * 110.57,
                                (lon[tr] - o) * 92.3)) > 0.5
                for a, o in zip(elat, elon)])
            elat, elon, ey = elat[keep_e], elon[keep_e], ey[keep_e]
            ecf, _ = climo_features(sources, dc.climo_vars, elon, elat,
                                    np.full(len(elat), float(np.median(tdays))),
                                    dc.interp, dc.valid_ranges)
            esid = np.arange(len(elat)) + (site_id.max() + 1)
            CF = np.vstack([CF, ecf]); CY = np.concatenate([CY, ey])
            CS = np.concatenate([CS, esid])
            if verbose:
                print(f"  climatology: +{len(elat)} extra sites from "
                      f"{os.path.basename(dc.climo_extra)} "
                      f"({10**ey.min():.0f}-{10**ey.max():.0f} cm-3)")
        climo_model = ClimoModel.fit(
            CF, CN, CY, CS,
            signs=dc.climo_signs, max_terms=dc.climo_max_terms,
            clip=dc.climo_clip, verbose=verbose,
        )
        climo_off = climo_model.predict(cf, cn)
        y_t = y_t - climo_off              # the network now fits the anomaly only

    y_scaler = Standardizer().fit(y_t[masks["train"]].reshape(-1, 1))
    y_s = y_scaler.transform(y_t.reshape(-1, 1)).ravel()

    # -------------------------------------------------------------------- coordinates
    cal = calendar_features(tdays)
    # Layout is ufp_pcl.data.netcdf.COORD_COLS: the encoder reads columns 0..5, and the
    # trailing raw time axis lets the proxy sampler be queried at labelled coordinates.
    coords = np.stack(
        [lon, lat, cal["doy"], cal["year"], cal["hour"], cal["dow"], tdays], axis=1
    ).astype(np.float32)
    assert coords.shape[1] == len(COORD_COLS)
    year0 = float(np.median(cal["year"][masks["train"]]))

    # -------------------------------------------------------------------------- proxy
    proxy = ProxySampler(
        proxy_stacks, dc.proxy_transform, interp=dc.interp, seed=cfg.train.seed,
        valid_ranges=dc.valid_ranges, require_all=dc.proxy_require_all,
        bounds=((float(dc.domain[0]), float(dc.domain[1]),
                 float(dc.domain[2]), float(dc.domain[3])) if dc.domain else None),
    ) if proxy_stacks else None
    if proxy is not None:
        proxy.fit_scaler()

    # The analysis domain is the region the model is meant to *predict over*, which is
    # the extent of the gridded fields -- not the bounding box of seven monitors.  Fall
    # back to the label extent only when there are no gridded fields at all.
    domain = None
    for stacks in (proxy_stacks, obs_stacks):
        if stacks:
            b = np.array([st.bounds for st in stacks], dtype=float)
            domain = (b[:, 0].max(), b[:, 1].min(), b[:, 2].max(), b[:, 3].min())
            break
    if domain is None:
        domain = (lon.min(), lon.max(), lat.min(), lat.max())
    if dc.domain:
        d = [float(v) for v in dc.domain]
        domain = (max(domain[0], d[0]), min(domain[1], d[1]),
                  max(domain[2], d[2]), min(domain[3], d[3]))
    coord_meta = {"year0": year0,
                  "lon_min": float(domain[0]), "lon_max": float(domain[1]),
                  "lat_min": float(domain[2]), "lat_max": float(domain[3]),
                  "label_lon_min": float(lon.min()), "label_lon_max": float(lon.max()),
                  "label_lat_min": float(lat.min()), "label_lat_max": float(lat.max())}

    def subset(m):
        return LabeledPointDataset(obs[m], coords[m], y_s[m], climo_off[m])

    # additional evaluation cells (e.g. the three cells of the "combined" protocol)
    extra_sets = {k: subset(v) for k, v in masks.items()
                  if k not in ("train", "val", "test") and v.any()}

    lines = [
        f"target        : {dc.target.var}  ({dc.target.transform}) n={len(y)}"
        f"  sites={len(np.unique(site_id))}  dropped={dropped}",
        f"obs features  : {len(obs_names)} vars on {len(obs_stacks)} grid(s)"
        + (f", seq_len={seq}" if seq > 1 else ""),
        f"proxy targets : {proxy.var_names if proxy else '[]'}",
        (f"climatology    : {' + '.join(climo_model.names)}  LOO R2={climo_model.loo_r2:+.3f}"
         f"  ({climo_model.n_sites} train sites)" if climo_model is not None
         and climo_model.names else None),
        f"split ({cfg.split.kind})" + (f" delta={cfg.split.delta}"
                                       if cfg.split.kind == "checkerboard" else ""),
        split_report(masks, site_id),
    ]
    report = "\n".join(l for l in lines if l)
    if verbose:
        print(report)
    for s_ in sources:
        s_.close()

    return DataBundle(
        train=subset(masks["train"]), val=subset(masks["val"]), test=subset(masks["test"]),
        proxy=proxy, obs_names=obs_names,
        proxy_names=(proxy.var_names if proxy else []),
        obs_scaler=obs_scaler, y_scaler=y_scaler, y_transform=dc.target.transform,
        coord_meta=coord_meta, masks=masks, table=table, seq_len=seq, report=report,
        climo=climo_model, extra_sets=extra_sets,
    )
