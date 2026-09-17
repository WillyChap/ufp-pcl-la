"""End-to-end latent-space and attribution report for a trained run.

This is the part that turns a trained model into claims.  It produces, for one run:

  * PCA of the location-time embedding sampled on a regular grid across the basin and
    across the diurnal/seasonal cycle, mapped in space and profiled in time;
  * a space/time/interaction variance partition of that embedding;
  * smoothness diagnostics (gradient magnitude, Moran's I) that test whether the proxy
    term actually regularised the prior or the encoder just memorised the seven sites;
  * the correlation length of the prior, i.e. how far the model transports information;
  * linear probes measuring what the prior encodes about each covariate and the proxy;
  * a k-means zonation of the basin from the embedding alone;
  * an ablation replacing the prior with its mean, to size its predictive contribution;
  * integrated-gradient and permutation attributions, collapsed into a primary vs
    secondary pathway index and profiled by hour of day and weekday/weekend.

Where the file carries synthetic ground truth (`tpn_primary` / `tpn_secondary`), the
report also scores the recovered secondary fraction against the truth -- the check that
the interpretation machinery works before it is pointed at real data.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from . import attribution as attrib
from . import latent as lat
from .config import Config, load_config
from .data.dataset import build_datasets
from .evaluate import evaluate_split


# --------------------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _sample_times(bundle, n_hours: int = 6, n_days: int = 4) -> np.ndarray:
    """A small space-filling set of times: several hours of day across several dates."""
    t = np.asarray(bundle.table["time_days"], dtype=float)
    lo, hi = np.quantile(t, 0.02), np.quantile(t, 0.98)
    days = np.linspace(np.floor(lo), np.floor(hi), n_days)
    hours = np.unique(np.round(((t % 1.0) * 24)).astype(int))
    hours = hours[np.linspace(0, len(hours) - 1, min(n_hours, len(hours))).astype(int)]
    return np.array([d + h / 24.0 for d in days for h in hours], dtype=float)


def load_run(run_dir: str, verbose: bool = False):
    """Rebuild (config, bundle, model, device) from a run directory."""
    from .train import Trainer

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    bundle = build_datasets(cfg, verbose=verbose)
    trainer = Trainer(cfg, bundle)
    trainer.load(os.path.join(run_dir, "best.pt"))
    trainer.model.eval()
    return cfg, bundle, trainer


# --------------------------------------------------------------------------------------
def latent_report(trainer, bundle, out_dir: str, grid_step: float = 0.02) -> Dict:
    model, device = trainer.model, trainer.device
    if model.loc_encoder is None:
        return {"note": "variant has no location encoder; latent analysis skipped"}
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    cm = bundle.coord_meta
    times = _sample_times(bundle)

    coords, meta = lat.make_coord_grid(
        cm["lon_min"], cm["lon_max"], cm["lat_min"], cm["lat_max"],
        step=grid_step, time_days=times,
    )
    E = lat.embed_coords(model, coords, device)
    T, ny, nx = meta["shape"]

    res: Dict = {}
    res["grid"] = {"n_points": int(len(E)), "shape": [T, ny, nx],
                   "step_deg": grid_step, "embedding_dim": int(E.shape[1])}

    # ---- structure of the latent space -----------------------------------------------
    p = lat.pca(E, k=8)
    fields = p.as_fields(meta)                                # (T, ny, nx, k)
    res["pca_explained_variance"] = p.explained.tolist()
    res["effective_rank"] = lat.effective_rank(E)
    res["variance_partition"] = lat.variance_partition(E, meta["shape"])

    pc1_mean = fields[..., 0].mean(axis=0)
    res["spatial_roughness_pc1"] = lat.spatial_roughness(pc1_mean)
    res["morans_i_pc1"] = lat.morans_i(pc1_mean)
    res["spatial_roughness_pc2"] = lat.spatial_roughness(fields[..., 1].mean(axis=0))

    # ---- how far does the prior carry information ------------------------------------
    dd = lat.distance_decay(E, coords[:, 0], coords[:, 1], n_pairs=150_000)
    res["correlation_length_km"] = dd["correlation_length_km"]

    # ---- what does the prior encode --------------------------------------------------
    e_label = lat.embed_coords(model, bundle.train.coords.numpy(), device)
    X = (bundle.train.obs.numpy() if bundle.train.obs.ndim == 2
         else bundle.train.obs.numpy().mean(1))
    res["probe_covariates"] = lat.linear_probe(e_label, X, names=bundle.obs_names)
    # A probe on a time-invariant field is not evidence of anything: land use and
    # elevation are pure functions of (lon, lat), so any location encoder reconstructs
    # them exactly.  Flag them so the interesting probes -- the time-varying ones -- are
    # not buried under a row of 1.000s.
    res["probe_is_static"] = _static_flags(X, bundle.train.coords.numpy(),
                                           bundle.obs_names)
    res["probe_target"] = lat.linear_probe(
        e_label, bundle.train.y.numpy().ravel(), names=[bundle.target_name
                                                        if hasattr(bundle, "target_name")
                                                        else "target"],
    )
    if bundle.proxy is not None:
        zc, zz = bundle.proxy.draw(min(20000, len(E)))
        from .train import proxy_coords_to_tensor
        pcoords = proxy_coords_to_tensor(zc, device).cpu().numpy()
        e_proxy = lat.embed_coords(model, pcoords, device)
        res["probe_proxy"] = lat.linear_probe(e_proxy, zz, names=bundle.proxy_names)

    # ---- zonation ---------------------------------------------------------------------
    pix_mean = E.reshape(T, ny * nx, -1).mean(axis=0)
    labels = lat.cluster_regions(pix_mean, k=6)
    res["n_zones"] = int(len(np.unique(labels)))

    # ---- how much predictive work is the prior doing ---------------------------------
    res["prior_ablation"] = lat.prior_ablation(
        model, bundle.test, device, bundle.y_scaler, bundle.y_transform
    )

    # ------------------------------------------------------------------------- figures
    ext = [meta["lons"][0], meta["lons"][-1], meta["lats"][0], meta["lats"][-1]]
    slon = bundle.table["lon"]; slat = bundle.table["lat"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i < 3:
            f, ttl = fields[..., i].mean(axis=0), f"PC{i+1} (time mean)"
            sub = f"{100*p.explained[i]:.1f}% of embedding variance"
        else:
            j = i - 3
            f = fields[j * max(1, T // 3) % T, :, :, 0]
            ttl, sub = "PC1 snapshot", f"t = {meta['times'][j * max(1, T//3) % T]:.2f} d"
        im = ax.imshow(f, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        ax.scatter(slon, slat, s=14, c="white", edgecolors="k", linewidths=0.6, zorder=3)
        ax.set_title(f"{ttl}\n{sub}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Location-time embedding: leading principal components\n"
                 "white dots = monitoring sites", fontsize=12)
    fig.savefig(os.path.join(out_dir, "latent_pca_maps.png"), dpi=130)
    plt.close(fig)

    # temporal profile (Figure 5 of the paper)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    hours = (meta["times"] % 1.0) * 24
    order = np.argsort(meta["times"])
    for i in range(3):
        s = fields[..., i].reshape(T, -1)
        ax.plot(meta["times"][order] - meta["times"].min(), s.mean(1)[order],
                marker="o", ms=3, label=f"PC{i+1} ({100*p.explained[i]:.0f}%)")
        ax.fill_between(meta["times"][order] - meta["times"].min(),
                        s.min(1)[order], s.max(1)[order], alpha=0.12)
    ax.set_xlabel("days from start of sampled window")
    ax.set_ylabel("PC score (spatial mean; band = spatial min-max)")
    ax.set_title("Temporal structure of the location-time embedding")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "latent_temporal.png"), dpi=130)
    plt.close(fig)

    # distance decay + zonation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(dd["distance_km"], dd["similarity"], "o-", ms=3)
    axes[0].set_xscale("log")
    axes[0].fill_between(dd["distance_km"], dd["similarity"] - dd["sd"],
                         dd["similarity"] + dd["sd"], alpha=0.15)
    if np.isfinite(dd["correlation_length_km"]):
        axes[0].axvline(dd["correlation_length_km"], ls="--", c="crimson",
                        label=f"correlation length {dd['correlation_length_km']:.0f} km")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("separation (km)"); axes[0].set_ylabel("embedding cosine similarity")
    axes[0].set_title("How far the geographic prior carries")

    im = axes[1].imshow(labels.reshape(ny, nx), origin="lower", extent=ext,
                        aspect="auto", cmap="tab10", interpolation="nearest")
    axes[1].scatter(slon, slat, s=16, c="white", edgecolors="k", linewidths=0.6)
    axes[1].set_title("Latent zonation (k-means on embeddings)")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.savefig(os.path.join(out_dir, "latent_structure.png"), dpi=130)
    plt.close(fig)

    res["figures"] = ["latent_pca_maps.png", "latent_temporal.png", "latent_structure.png"]
    return res


# --------------------------------------------------------------------------------------
def attribution_report(trainer, bundle, out_dir: str, n_steps: int = 32,
                       utc_offset: int = 8) -> Dict:
    """`utc_offset` is hours behind UTC at the study site (8 for Pacific standard time);
    it only affects axis labelling, never the model."""
    model, device = trainer.model, trainer.device
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)

    A = attrib.integrated_gradients(model, bundle.test, device, n_steps=n_steps,
                                    names=bundle.obs_names)
    paths = attrib.pathway_scores(A)
    res: Dict = {
        "mean_abs_attribution": A.mean_abs(),
        "signed_mean_attribution": A.signed_mean(),
        "feature_groups": {k: [bundle.obs_names[i] for i in v]
                           for k, v in attrib.classify_features(bundle.obs_names).items()},
        "secondary_fraction_mean": float(np.nanmean(paths["secondary_fraction"])),
        "weekend_contrast_secondary_fraction":
            attrib.weekend_contrast(paths["secondary_fraction"], A.coords),
        "weekend_contrast_primary_attribution":
            attrib.weekend_contrast(paths["primary"], A.coords),
    }
    try:
        res["permutation_importance"] = attrib.permutation_importance(
            model, bundle.test, device, bundle.y_scaler, n_repeats=3,
            names=bundle.obs_names)
    except Exception as exc:                                   # pragma: no cover
        res["permutation_importance"] = {"error": str(exc)}

    diurnal_sec = attrib.conditional_profile(paths["secondary_fraction"], A.coords, "hour")
    diurnal_pri = attrib.conditional_profile(paths["primary"], A.coords, "hour")
    diurnal_sec_abs = attrib.conditional_profile(paths["secondary"], A.coords, "hour")
    res["utc_offset_hours"] = utc_offset
    res["diurnal_secondary_fraction"] = {
        "hour_utc": diurnal_sec["labels"].tolist(),
        "hour_local": ((diurnal_sec["labels"] - utc_offset) % 24).tolist(),
        "mean": np.round(diurnal_sec["mean"], 4).tolist(),
        "count": diurnal_sec["count"].tolist(),
    }

    # ---------------------------------------------------------------------- figures
    top = list(res["mean_abs_attribution"].items())[:15][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].barh([k for k, _ in top], [v for _, v in top], color="steelblue")
    axes[0].set_xlabel("mean |integrated gradient|  (standardised log10 UFP)")
    axes[0].set_title("Feature attribution (test set)")

    h = diurnal_pri["labels"]
    axes[1].plot(h, diurnal_pri["mean"], "o-", label="primary tracers |attr|", color="#b2182b")
    axes[1].plot(h, diurnal_sec_abs["mean"], "o-", label="secondary tracers |attr|", color="#2166ac")
    ax2 = axes[1].twinx()
    ax2.plot(h, diurnal_sec["mean"], "s--", ms=4, color="grey", label="secondary fraction")
    ax2.set_ylabel("secondary fraction")
    axes[1].set_xticks(h[::2])
    axes[1].set_xticklabels([f"{u:02d}\n({(u - utc_offset) % 24:02d})" for u in h[::2]],
                            fontsize=8)
    axes[1].set_xlabel("hour  UTC  (local)")
    axes[1].set_ylabel("attribution magnitude")
    axes[1].set_title("Pathway attribution through the day")
    lines = axes[1].get_lines() + ax2.get_lines()
    axes[1].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    fig.savefig(os.path.join(out_dir, "attribution.png"), dpi=130)
    plt.close(fig)
    res["figures"] = ["attribution.png"]

    # ------------------------------------------- validation against synthetic truth
    truth = _synthetic_truth(bundle)
    if truth is not None:
        idx = _match_rows(bundle.test, truth)
        if idx is not None:
            tv = truth["secondary_fraction"][idx]
            sf = paths["secondary_fraction"]
            m = np.isfinite(tv) & np.isfinite(sf)
            if m.sum() > 20:
                res["synthetic_validation"] = {
                    "n": int(m.sum()),
                    "pearson_r_secondary_fraction": float(np.corrcoef(tv[m], sf[m])[0, 1]),
                    "spearman_r_secondary_fraction": float(_spearman(tv[m], sf[m])),
                    "true_mean_secondary_fraction": float(tv[m].mean()),
                    "model_mean_secondary_fraction": float(sf[m].mean()),
                    "note": ("attribution-derived index vs the known generative split; "
                             "magnitudes are not expected to match, the ranking is the test"),
                }
    return res


def _static_flags(X: np.ndarray, coords: np.ndarray, names: List[str],
                  tol: float = 1e-3) -> Dict[str, bool]:
    """True where a covariate barely varies in time at a fixed location."""
    key = np.round(coords[:, 0], 3) * 1000 + np.round(coords[:, 1], 3)
    _, inv = np.unique(key, return_inverse=True)
    out = {}
    for j, n in enumerate(names):
        col = X[:, j]
        within = np.array([col[inv == g].std() for g in np.unique(inv)])
        total = col.std()
        out[n] = bool(total <= 0 or (within.mean() / total) < tol)
    return out


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _synthetic_truth(bundle) -> Optional[Dict[str, np.ndarray]]:
    """Read tpn_primary / tpn_secondary from the source file, if present."""
    import xarray as xr

    path = bundle.__dict__.get("_source_path")
    if path is None:
        return None
    try:
        ds = xr.open_dataset(path, decode_timedelta=False)
    except Exception:
        return None
    if "tpn_primary" not in ds or "tpn_secondary" not in ds:
        ds.close()
        return None
    pri = ds["tpn_primary"].values.ravel()
    sec = ds["tpn_secondary"].values.ravel()
    lon = np.repeat(ds["site_lon"].values, ds.sizes["time"])
    lat = np.repeat(ds["site_lat"].values, ds.sizes["time"])
    t = (ds["time"].values.astype("datetime64[ns]").astype("int64") / 86_400e9)
    t = np.tile(t, ds.sizes["site"])
    ds.close()
    return {"lon": lon, "lat": lat, "time_days": t,
            "secondary_fraction": sec / np.clip(pri + sec, 1e-9, None)}


def _match_rows(dataset, truth) -> Optional[np.ndarray]:
    """Map dataset rows onto the truth table by (lon, lat, hour) key.

    Coordinates are carried as float32, whose precision at ~19,600 days since epoch is
    about three minutes -- fine for the model, too coarse for an exact float key.  So the
    time axis is matched at whole-hour resolution, which is the data's own resolution.
    """
    def key(lon, lat, t):
        return np.stack([np.round(np.asarray(lon, float), 3),
                         np.round(np.asarray(lat, float), 3),
                         np.round(np.asarray(t, float) * 24.0)], 1)

    key_t = key(truth["lon"], truth["lat"], truth["time_days"])
    c = dataset.coords.numpy()
    key_d = key(c[:, 0], c[:, 1], c[:, 6])
    lut = {tuple(r): i for i, r in enumerate(key_t)}
    idx = np.array([lut.get(tuple(r), -1) for r in key_d])
    if (idx < 0).all():
        return None
    return np.where(idx < 0, 0, idx)


# --------------------------------------------------------------------------------------
def full_report(trainer, bundle, out_dir: str, compare: Optional[tuple] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    report: Dict = {"run_dir": out_dir, "variant": trainer.cfg.model.variant}
    print("\n" + "=" * 78)
    print("latent-space analysis")
    print("=" * 78)
    report["latent"] = latent_report(trainer, bundle, out_dir)
    _print_latent(report["latent"])

    print("\n" + "=" * 78)
    print("feature attribution and pathway apportionment")
    print("=" * 78)
    report["attribution"] = attribution_report(trainer, bundle, out_dir)
    _print_attribution(report["attribution"])

    if compare is not None:
        other_trainer, other_bundle, label = compare
        report["comparison"] = compare_runs(trainer, bundle, other_trainer, label)
        _print_comparison(report["comparison"])

    path = os.path.join(out_dir, "analysis.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(report), fh, indent=2)
    print(f"\nanalysis written to {path}")
    print(f"figures: {', '.join(report['latent'].get('figures', []) + report['attribution'].get('figures', []))}")
    return report


def compare_runs(trainer_a, bundle_a, trainer_b, label_b: str = "other") -> Dict:
    """Contrast two trained location encoders on the same coordinates."""
    if trainer_a.model.loc_encoder is None or trainer_b.model.loc_encoder is None:
        return {"note": "one of the runs has no location encoder"}
    cm = bundle_a.coord_meta
    times = _sample_times(bundle_a)
    coords, meta = lat.make_coord_grid(cm["lon_min"], cm["lon_max"], cm["lat_min"],
                                       cm["lat_max"], step=0.02, time_days=times)
    Ea = lat.embed_coords(trainer_a.model, coords, trainer_a.device)
    Eb = lat.embed_coords(trainer_b.model, coords, trainer_b.device)
    T, ny, nx = meta["shape"]
    pa, pb = lat.pca(Ea, 3), lat.pca(Eb, 3)
    fa = pa.as_fields(meta)[..., 0].mean(0)
    fb = pb.as_fields(meta)[..., 0].mean(0)
    return {
        "label_b": label_b,
        "cka": lat.cka(Ea, Eb),
        "variance_partition_a": lat.variance_partition(Ea, meta["shape"]),
        "variance_partition_b": lat.variance_partition(Eb, meta["shape"]),
        "spatial_roughness_pc1_a": lat.spatial_roughness(fa),
        "spatial_roughness_pc1_b": lat.spatial_roughness(fb),
        "morans_i_pc1_a": lat.morans_i(fa),
        "morans_i_pc1_b": lat.morans_i(fb),
        "effective_rank_a": lat.effective_rank(Ea),
        "effective_rank_b": lat.effective_rank(Eb),
        "correlation_length_km_a": lat.distance_decay(Ea, coords[:, 0], coords[:, 1])["correlation_length_km"],
        "correlation_length_km_b": lat.distance_decay(Eb, coords[:, 0], coords[:, 1])["correlation_length_km"],
    }


def analyze_run(run_dir: str, compare_dir: Optional[str] = None,
                out_dir: Optional[str] = None) -> Dict:
    cfg, bundle, trainer = load_run(run_dir, verbose=True)
    bundle.__dict__["_source_path"] = cfg.data.path
    compare = None
    if compare_dir:
        _, bundle_b, trainer_b = load_run(compare_dir, verbose=False)
        compare = (trainer_b, bundle_b, os.path.basename(compare_dir.rstrip("/")))
    return full_report(trainer, bundle, out_dir or run_dir, compare=compare)


# --------------------------------------------------------------------------------- print
def _fmt(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def _print_latent(r: Dict) -> None:
    if "note" in r:
        print(r["note"]); return
    vp = r["variance_partition"]
    print(f"embedding dim {r['grid']['embedding_dim']}, "
          f"effective rank {_fmt(r['effective_rank'], 1)} "
          f"(how many directions are actually used)")
    print(f"variance partition   space {vp['space']:.1%}   time {vp['time']:.1%}   "
          f"interaction {vp['interaction']:.1%}")
    print(f"PC1 explains {r['pca_explained_variance'][0]:.1%};  "
          f"roughness {_fmt(r['spatial_roughness_pc1'])}  Moran's I {_fmt(r['morans_i_pc1'])}")
    print(f"prior correlation length {_fmt(r['correlation_length_km'], 0)} km")
    ab = r.get("prior_ablation", {})
    if ab:
        print(f"replacing the prior with its mean: test R2 -> {_fmt(ab.get('ablated_r2'))}")
    pp = r.get("probe_proxy", {})
    if pp:
        print("proxy reconstruction from e_loc alone (ridge R2): "
              + ", ".join(f"{k}={_fmt(v)}" for k, v in pp.items()))
    pc = r.get("probe_covariates", {})
    static = r.get("probe_is_static", {})
    if pc:
        dyn = {k: v for k, v in pc.items() if not static.get(k, False)}
        top = sorted(dyn.items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -9))[:8]
        print("what the prior encodes about time-varying covariates (ridge R2, top 8):")
        for k, v in top:
            print(f"    {k:<24}{_fmt(v)}")
        n_static = sum(1 for v in static.values() if v)
        if n_static:
            print(f"    ({n_static} time-invariant fields omitted -- a location encoder "
                  f"reconstructs those by construction)")


def _print_attribution(r: Dict) -> None:
    top = list(r["mean_abs_attribution"].items())[:10]
    print("top features by mean |integrated gradient|:")
    for k, v in top:
        print(f"    {k:<24}{v:.4f}")
    print(f"mean secondary fraction (attribution index): {_fmt(r['secondary_fraction_mean'])}")
    wc = r["weekend_contrast_primary_attribution"]
    print(f"primary-tracer attribution, weekday {_fmt(wc['weekday_mean'])} vs weekend "
          f"{_fmt(wc['weekend_mean'])}  (diff {_fmt(wc['difference'])}, t={_fmt(wc['t'], 1)})")
    sv = r.get("synthetic_validation")
    if sv:
        print(f"synthetic-truth check: Pearson r={_fmt(sv['pearson_r_secondary_fraction'])}, "
              f"Spearman r={_fmt(sv['spearman_r_secondary_fraction'])} "
              f"against the known secondary fraction (n={sv['n']})")


def _print_comparison(c: Dict) -> None:
    if "note" in c:
        print(c["note"]); return
    b = c["label_b"]
    print("\n" + "=" * 78)
    print(f"latent comparison: this run vs {b}")
    print("=" * 78)
    print(f"{'metric':<28}{'this':>14}{b:>18}")
    rows = [
        ("variance share: space", c["variance_partition_a"]["space"], c["variance_partition_b"]["space"]),
        ("variance share: time", c["variance_partition_a"]["time"], c["variance_partition_b"]["time"]),
        ("PC1 spatial roughness", c["spatial_roughness_pc1_a"], c["spatial_roughness_pc1_b"]),
        ("PC1 Moran's I", c["morans_i_pc1_a"], c["morans_i_pc1_b"]),
        ("effective rank", c["effective_rank_a"], c["effective_rank_b"]),
        ("correlation length (km)", c["correlation_length_km_a"], c["correlation_length_km_b"]),
    ]
    for name, a, bb in rows:
        print(f"{name:<28}{_fmt(a):>14}{_fmt(bb):>18}")
    print(f"{'linear CKA between them':<28}{_fmt(c['cka']):>14}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
