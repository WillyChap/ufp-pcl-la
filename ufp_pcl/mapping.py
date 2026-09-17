"""Predicting the UFP surface everywhere, and scoring how well the gaps are filled.

Training only ever sees seven points. The product is a map. This module closes that gap:
it evaluates the trained model on every pixel of the domain at chosen times, and — on the
synthetic testbed, where the true surface is known at every pixel — scores the fill
directly rather than only at the monitors.

The diagnostic that matters most is `error_vs_distance`: prediction error as a function
of distance to the nearest monitor. A model that is accurate at its training sites and
degrades steeply a few kilometres away has not produced a usable exposure surface, and
that failure is invisible in any metric computed at the monitors.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .data.dataset import build_stacks, open_sources, resolve_variable_roles, sample_stacks
from .data.netcdf import NetCDFSource, calendar_features, solar_elevation_deg, to_days
from .data.scaling import invert_transform
from .features import compute_derived
from .latent import haversine_km


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------------------
def pick_times(cfg: Config, n: int = 3, hours_local: Sequence[int] = (7, 12, 16),
               utc_offset: int = 8) -> np.ndarray:
    """Pick times from the file's own axis: the same day at several local hours.

    Defaults straddle the two UFP regimes -- the 07:00 traffic peak and the midday
    photochemical peak -- so one figure shows whether the model reproduces the switch.
    """
    import xarray as xr

    ds = xr.open_dataset(cfg.data.path, decode_timedelta=False)
    t = to_days(ds[cfg.data.coords.time].values)
    ds.close()
    hours = np.round((t % 1.0) * 24).astype(int)
    mid_day = np.floor(t[len(t) // 2])
    out = []
    for hl in hours_local[:n]:
        target_utc = (hl + utc_offset) % 24
        cand = t[(np.floor(t) == mid_day) & (hours == target_utc)]
        if cand.size:
            out.append(float(cand[0]))
    if not out:                                     # fall back to evenly spaced times
        out = np.quantile(t, np.linspace(0.2, 0.8, n)).tolist()
    return np.array(out, dtype=float)


@torch.no_grad()
def predict_surface(
    cfg: Config, bundle, model, device, times: np.ndarray,
    step: Optional[float] = None, batch_size: int = 16384,
) -> Tuple[np.ndarray, Dict]:
    """Predict the target on a regular grid at each requested time.

    Returns the surface in *native* units (particles cm-3) with shape (n_times, ny, nx),
    plus a meta dict carrying the axes.
    """
    from .latent import make_coord_grid

    sources = open_sources(cfg)
    obs_vars, _ = resolve_variable_roles(cfg, sources)
    obs_stacks = build_stacks(sources, obs_vars) if obs_vars else []
    climo_model = getattr(bundle, "climo", None)
    if step is None:                                 # default: the native grid spacing
        step = float(np.median(np.diff(obs_stacks[0].lon))) if obs_stacks else 0.02

    cm = bundle.coord_meta
    coords, meta = make_coord_grid(cm["lon_min"], cm["lon_max"], cm["lat_min"],
                                   cm["lat_max"], step=step, time_days=times)
    lon, lat, tdays = coords[:, 0].astype(float), coords[:, 1].astype(float), coords[:, 6].astype(float)

    seq = max(1, bundle.seq_len)
    if seq == 1:
        obs, names = sample_stacks(obs_stacks, lon, lat, tdays, cfg.data.interp,
                                   valid_ranges=cfg.data.valid_ranges)
    else:
        frames = []
        for k in range(seq - 1, -1, -1):
            f, names = sample_stacks(obs_stacks, lon, lat, tdays - k / 24.0, cfg.data.interp,
                                     valid_ranges=cfg.data.valid_ranges)
            frames.append(f)
        obs = np.stack(frames, axis=1)
    if cfg.data.derived:
        obs, names = compute_derived(obs, names, lon, lat, tdays, cfg.data.derived)

    # The climo_anomaly network predicts a departure, so the surface is only UFP once the
    # climatology is evaluated on the same grid and added back.
    climo_grid = None
    if climo_model is not None:
        from .data.dataset import climo_features
        cf, cn = climo_features(sources, cfg.data.climo_vars, lon, lat, tdays,
                                cfg.data.interp, cfg.data.valid_ranges)
        climo_grid = climo_model.predict(cf, cn)
    for s_ in sources:
        s_.close()

    if list(names) != list(bundle.obs_names):
        raise RuntimeError("feature order on the grid differs from training:\n"
                           f"  grid:  {names}\n  train: {bundle.obs_names}")

    valid = np.isfinite(obs).all(axis=tuple(range(1, obs.ndim)))
    obs = np.nan_to_num(bundle.obs_scaler.transform(obs))

    model.eval()
    preds = []
    for i in range(0, len(coords), batch_size):
        o = torch.as_tensor(obs[i:i + batch_size], device=device)
        c = torch.as_tensor(coords[i:i + batch_size], device=device)
        preds.append(model(o, c).float().cpu().numpy().ravel())
    pred_s = np.concatenate(preds)
    pred_t = bundle.y_scaler.inverse(pred_s.reshape(-1, 1)).ravel()
    if climo_grid is not None:
        pred_t = pred_t + climo_grid
    rep = float(getattr(cfg.data, "representativeness", 1.0) or 1.0)
    if rep != 1.0:
        # the network runs high relative to the basin; correct before reporting
        pred_t = pred_t - np.log10(rep) if bundle.y_transform == "log10" else pred_t
    native = invert_transform(pred_t, bundle.y_transform)
    if rep != 1.0 and bundle.y_transform != "log10":
        native = native / rep
    native = np.where(valid, native, np.nan)

    T, ny, nx = meta["shape"]
    meta["valid"] = valid.reshape(T, ny, nx)
    return native.reshape(T, ny, nx), meta


def truth_surface(cfg: Config, meta: Dict, var: str = "truth_ufp_field") -> Optional[np.ndarray]:
    """The true surface at the same grid/times, when the file carries one."""
    import xarray as xr

    ds = xr.open_dataset(cfg.data.path, decode_timedelta=False)
    if var not in ds:
        ds.close()
        return None
    from .data.netcdf import FieldStack

    latd = next(d for d in ds[var].dims if d.startswith("lat"))
    lond = next(d for d in ds[var].dims if d.startswith("lon"))
    timd = next((d for d in ds[var].dims if d.startswith("time")), None)
    latv = np.asarray(ds[latd].values, float)
    lonv = np.asarray(ds[lond].values, float)
    order = ([timd] if timd else []) + [latd, lond]
    arr = ds[var].transpose(*order).values.astype(np.float32)
    tdays = to_days(ds[timd].values) if timd else None
    ds.close()

    st = FieldStack(name="truth", lat=latv, lon=lonv, time_days=tdays,
                    var_names=[var], data=arr[:, None] if timd else arr[None])
    T, ny, nx = meta["shape"]
    LON, LAT = np.meshgrid(meta["lons"], meta["lats"])
    out = np.empty((T, ny, nx), dtype=np.float32)
    for i, t in enumerate(meta["times"]):
        out[i] = st.sample(LON.ravel(), LAT.ravel(),
                           np.full(LON.size, t)).reshape(ny, nx)
    return out


# --------------------------------------------------------------------------------------
def error_vs_distance(
    pred: np.ndarray, truth: np.ndarray, meta: Dict,
    site_lon: np.ndarray, site_lat: np.ndarray, n_bins: int = 12,
) -> Dict[str, np.ndarray]:
    """Log10 prediction error binned by distance to the nearest monitor."""
    LON, LAT = np.meshgrid(meta["lons"], meta["lats"])
    d = np.min(np.stack([haversine_km(LON, LAT, lo, la)
                         for lo, la in zip(site_lon, site_lat)]), axis=0)
    err = np.log10(np.clip(pred, 1, None)) - np.log10(np.clip(truth, 1, None))
    dd = np.tile(d[None], (pred.shape[0], 1, 1)).ravel()
    ee = err.ravel()
    m = np.isfinite(dd) & np.isfinite(ee)
    dd, ee = dd[m], ee[m]

    edges = np.linspace(0, np.percentile(dd, 98), n_bins + 1)
    idx = np.clip(np.digitize(dd, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rmse = np.array([np.sqrt((ee[idx == b] ** 2).mean()) if (idx == b).any() else np.nan
                     for b in range(n_bins)])
    bias = np.array([ee[idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(n_bins)])
    n = np.array([int((idx == b).sum()) for b in range(n_bins)])
    return {"distance_km": centres, "rmse_log10": rmse, "bias_log10": bias, "count": n,
            "distance_field": d}


def surface_scores(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Skill of the whole predicted surface, in log10 space and in native units."""
    p = np.log10(np.clip(pred, 1, None)).ravel()
    t = np.log10(np.clip(truth, 1, None)).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    if p.size < 10:
        return {}
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "surface_r2_log10": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "surface_rmse_log10": float(np.sqrt(((p - t) ** 2).mean())),
        "surface_bias_log10": float((p - t).mean()),
        "surface_spatial_r": float(np.corrcoef(p, t)[0, 1]),
        "n_pixels": int(p.size),
    }


# ------------------------------------------------------------------------------ figures
def _norm(vmin, vmax):
    from matplotlib.colors import LogNorm
    return LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, 10.0))


def plot_fill(pred, truth, meta, site_lon, site_lat, out_path, title="",
              utc_offset: int = 8):
    """Rows = times; columns = truth, prediction, and the log10 ratio between them."""
    plt = _mpl()
    T = pred.shape[0]
    ncol = 3 if truth is not None else 1
    fig, axes = plt.subplots(T, ncol, figsize=(5.0 * ncol, 3.5 * T),
                             squeeze=False, constrained_layout=True)
    ext = [meta["lons"][0], meta["lons"][-1], meta["lats"][0], meta["lats"][-1]]
    finite = np.isfinite(pred) & (np.isfinite(truth) if truth is not None else True)
    lo = np.nanpercentile(np.where(finite, pred, np.nan), 2)
    hi = np.nanpercentile(np.where(finite, pred, np.nan), 98)
    if truth is not None:
        lo = min(lo, np.nanpercentile(truth, 2))
        hi = max(hi, np.nanpercentile(truth, 98))
    norm = _norm(lo, hi)

    for i in range(T):
        hutc = int(round((meta["times"][i] % 1.0) * 24)) % 24
        hloc = (hutc - utc_offset) % 24
        stamp = f"{hloc:02d}:00 local"
        if truth is not None:
            im = axes[i][0].imshow(truth[i], origin="lower", extent=ext, aspect="auto",
                                   cmap="magma", norm=norm)
            axes[i][0].set_title(f"true UFP — {stamp}", fontsize=10)
            fig.colorbar(im, ax=axes[i][0], shrink=0.85, label="particles cm$^{-3}$")
        j = 1 if truth is not None else 0
        im = axes[i][j].imshow(pred[i], origin="lower", extent=ext, aspect="auto",
                               cmap="magma", norm=norm)
        axes[i][j].set_title(f"predicted — {stamp}", fontsize=10)
        fig.colorbar(im, ax=axes[i][j], shrink=0.85, label="particles cm$^{-3}$")
        if truth is not None:
            ratio = np.log10(np.clip(pred[i], 1, None)) - np.log10(np.clip(truth[i], 1, None))
            im = axes[i][2].imshow(ratio, origin="lower", extent=ext, aspect="auto",
                                   cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            axes[i][2].set_title(f"log$_{{10}}$(pred / true) — {stamp}", fontsize=10)
            fig.colorbar(im, ax=axes[i][2], shrink=0.85)
        for ax in axes[i]:
            ax.scatter(site_lon, site_lat, s=26, c="cyan", edgecolors="k",
                       linewidths=0.7, zorder=4)
    fig.suptitle(title + "\ncyan = the seven monitoring sites the model was trained on",
                 fontsize=12)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_variant_maps(surfaces: Dict[str, np.ndarray], truth, meta, site_lon, site_lat,
                      out_path, t_index: int = 0, utc_offset: int = 8):
    """One time step, truth beside every variant's predicted surface."""
    plt = _mpl()
    panels = ([("true UFP", truth[t_index])] if truth is not None else []) + \
             [(k, v[t_index]) for k, v in surfaces.items()]
    n = len(panels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False, constrained_layout=True)
    ext = [meta["lons"][0], meta["lons"][-1], meta["lats"][0], meta["lats"][-1]]
    allv = np.concatenate([np.asarray(v).ravel() for _, v in panels])
    norm = _norm(np.nanpercentile(allv, 2), np.nanpercentile(allv, 98))

    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off"); continue
        name, f = panels[k]
        im = ax.imshow(f, origin="lower", extent=ext, aspect="auto", cmap="magma", norm=norm)
        ax.scatter(site_lon, site_lat, s=24, c="cyan", edgecolors="k", linewidths=0.7, zorder=4)
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85, label="particles cm$^{-3}$")
    hutc = int(round((meta["times"][t_index] % 1.0) * 24)) % 24
    fig.suptitle(f"Filling in the space between monitors — {(hutc - utc_offset) % 24:02d}:00 local",
                 fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_error_vs_distance(curves: Dict[str, Dict], out_path: str):
    """The key panel: does skill survive away from the monitors?"""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for name, c in curves.items():
        style = dict(lw=2.2) if name == "pcl" else dict(lw=1.4, alpha=0.85)
        axes[0].plot(c["distance_km"], c["rmse_log10"], "o-", ms=3.5, label=name, **style)
        axes[1].plot(c["distance_km"], c["bias_log10"], "o-", ms=3.5, label=name, **style)
    axes[0].set_ylabel("RMSE of log$_{10}$ UFP"); axes[0].set_title("Error vs distance from the nearest monitor")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("bias of log$_{10}$ UFP"); axes[1].set_title("Bias vs distance from the nearest monitor")
    for ax in axes:
        ax.set_xlabel("distance to nearest monitoring site (km)")
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
