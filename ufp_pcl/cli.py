"""Command line entry points.

    python -m ufp_pcl inspect data/ufp.nc [--emit-config configs/auto.yaml]
    python -m ufp_pcl train   configs/ufp_la_synth.yaml [--set train.rho=8 ...]
    python -m ufp_pcl compare configs/ufp_la_synth.yaml [--variants pcl,fusion,obs_only]
    python -m ufp_pcl sweep   configs/ufp_la_synth.yaml --param train.rho --values 0,1,4,16
    python -m ufp_pcl analyze outputs/pcl [--compare outputs/fusion]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import yaml


def _parse_set(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got '{p}'")
        k, v = p.split("=", 1)
        try:
            val = yaml.safe_load(v)
        except Exception:
            val = v
        # YAML 1.1 only recognises scientific notation written as 1.0e-4 -- a bare
        # "1e-4" parses as a *string*, which then reaches a float field and fails deep
        # inside training with an unrelated TypeError.  Learning rates are written the
        # bare way by everyone, so coerce it here.
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        out[k] = val
    return out


# --------------------------------------------------------------------------------------
def cmd_inspect(args) -> int:
    from .data.netcdf import describe, guess_role
    import xarray as xr

    print(describe(args.path))
    if args.emit_config:
        ds = xr.open_dataset(args.path, decode_timedelta=False)
        roles = {n: guess_role(n, da.attrs) for n, da in ds.data_vars.items()}
        target = next((n for n, r in roles.items() if r == "target?"), None)
        proxies = [n for n, r in roles.items() if r == "proxy?"]
        obs = [n for n, r in roles.items() if r == "obs?" and n != target]
        lat = next((c for c in ds.coords if "lat" in c.lower()), "lat")
        lon = next((c for c in ds.coords if "lon" in c.lower()), "lon")
        blob = {
            "data": {
                "path": args.path,
                "coords": {"lon": lon, "lat": lat, "time": "time"},
                "target": {"var": target or "CHANGE_ME", "mode": "points",
                           "transform": "log10"},
                "obs_vars": obs,
                "proxy_vars": proxies,
                "proxy_transform": {p: "log10" for p in proxies},
            },
            "split": {"kind": "temporal", "test_frac": 0.2},
            "model": {"variant": "pcl"},
            "train": {"out_dir": "outputs/run"},
        }
        os.makedirs(os.path.dirname(args.emit_config) or ".", exist_ok=True)
        with open(args.emit_config, "w") as fh:
            yaml.safe_dump(blob, fh, sort_keys=False)
        print(f"\nstarter config written to {args.emit_config} -- review the roles above "
              f"before training (the guesses are name-based only)")
        ds.close()
    return 0


def _run_one(cfg, verbose: bool = True):
    from .data.dataset import build_datasets
    from .train import Trainer

    bundle = build_datasets(cfg, verbose=verbose)
    trainer = Trainer(cfg, bundle)
    metrics = trainer.fit()
    return trainer, bundle, metrics


def cmd_train(args) -> int:
    from .config import load_config

    cfg = load_config(args.config, _parse_set(args.set))
    trainer, bundle, metrics = _run_one(cfg)
    print(f"\nartefacts in {cfg.train.out_dir}/  (best.pt, metrics.json, history.json, config.yaml)")
    if args.analyze:
        from .analysis import full_report
        full_report(trainer, bundle, cfg.train.out_dir)
    return 0


def cmd_compare(args) -> int:
    from .config import load_config

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    base_over = _parse_set(args.set)
    rows = []
    for v in variants:
        print("\n" + "=" * 78)
        print(f"variant: {v}")
        print("=" * 78)
        over = dict(base_over)
        over["model.variant"] = v
        cfg = load_config(args.config, over)
        cfg.train.out_dir = os.path.join(args.out or "outputs/compare", v)
        _, _, m = _run_one(cfg)
        rows.append({"variant": v, **m})

    keys = ["test_r2", "test_rmse", "test_mae", "test_native_rmse", "sec_per_epoch"]
    print("\n" + "=" * 78)
    print(f"{'variant':<16}" + "".join(f"{k:>18}" for k in keys))
    print("-" * 78)
    for r in rows:
        print(f"{r['variant']:<16}" + "".join(
            f"{r.get(k, float('nan')):>18.4g}" for k in keys))
    out = os.path.join(args.out or "outputs/compare", "comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


def cmd_sweep(args) -> int:
    from .config import load_config

    values = [yaml.safe_load(v) for v in args.values.split(",")]
    rows = []
    for v in values:
        print("\n" + "=" * 78)
        print(f"{args.param} = {v}")
        print("=" * 78)
        over = _parse_set(args.set)
        over[args.param] = v
        # rho = 0 means "no proxy supervision at all", i.e. the plain fusion baseline
        if args.param == "train.rho" and v == 0:
            over["model.variant"] = "fusion"
        cfg = load_config(args.config, over)
        tag = f"{args.param.split('.')[-1]}_{v}"
        cfg.train.out_dir = os.path.join(args.out or "outputs/sweep", tag)
        _, _, m = _run_one(cfg)
        rows.append({args.param: v, **m})

    print("\n" + "=" * 78)
    print(f"{args.param:<16}{'test_r2':>12}{'test_rmse':>12}{'test_mae':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{str(r[args.param]):<16}{r.get('test_r2', np.nan):>12.4f}"
              f"{r.get('test_rmse', np.nan):>12.4f}{r.get('test_mae', np.nan):>12.4f}")
    out = os.path.join(args.out or "outputs/sweep", "sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


def cmd_loso_cv(args) -> int:
    """Full leave-one-site-out cross-validation, which is the evaluation a seven-site
    network can actually support: every site takes a turn as the unmonitored location,
    and the spread across folds is the error bar."""
    from .config import load_config
    from .data.dataset import build_datasets
    import xarray as xr

    base = load_config(args.config, _parse_set(args.set))
    probe = build_datasets(base, verbose=False)
    sites = np.unique(probe.table["site_id"]).tolist()
    names = {}
    try:
        ds = xr.open_dataset(base.data.path, decode_timedelta=False)
        if "site_name" in ds:
            names = {i: str(v) for i, v in enumerate(np.asarray(ds["site_name"].values))}
        ds.close()
    except Exception:
        pass

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = []
    for v in variants:
        for site in sites:
            label = names.get(site, f"site{site}")
            print("\n" + "=" * 78)
            print(f"variant {v}  |  holding out {label}")
            print("=" * 78)
            over = _parse_set(args.set)
            over.update({"model.variant": v, "split.kind": "loso", "split.holdout_site": int(site)})
            cfg = load_config(args.config, over)
            cfg.train.out_dir = os.path.join(args.out or "outputs/loso_cv", v, f"site{site}")
            _, _, m = _run_one(cfg, verbose=False)
            rows.append({"variant": v, "site": int(site), "site_name": label, **m})

    print("\n" + "=" * 78)
    print("leave-one-site-out cross-validation  (mean +- SE across folds)")
    print("=" * 78)
    print(f"{'variant':<16}{'test R2':>20}{'test RMSE':>20}{'native RMSE':>20}")
    print("-" * 78)
    summary = []
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        line = {"variant": v, "n_folds": len(sub)}
        cells = []
        for k in ("test_r2", "test_rmse", "test_native_rmse"):
            vals = np.array([r.get(k, np.nan) for r in sub], dtype=float)
            vals = vals[np.isfinite(vals)]
            mu = vals.mean() if vals.size else np.nan
            se = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            line[k] = float(mu); line[k + "_se"] = float(se)
            cells.append(f"{mu:.4g} +- {se:.2g}")
        summary.append(line)
        print(f"{v:<16}" + "".join(f"{c:>20}" for c in cells))

    out = os.path.join(args.out or "outputs/loso_cv", "loso_cv.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"folds": rows, "summary": summary}, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


def cmd_map(args) -> int:
    """Predict the surface everywhere and, where truth exists, score the fill."""
    from . import mapping as mp
    from .analysis import load_run
    import xarray as xr

    run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
    out_dir = args.out or os.path.join(os.path.dirname(run_dirs[0].rstrip("/")) or ".", "maps")
    os.makedirs(out_dir, exist_ok=True)

    cfg0, bundle0, trainer0 = load_run(run_dirs[0], verbose=False)
    times = mp.pick_times(cfg0, n=args.n_times)
    site_lon = np.unique(np.round(bundle0.table["lon"], 4))
    sl = {}
    for lo, la in zip(bundle0.table["lon"], bundle0.table["lat"]):
        sl[(round(float(lo), 4), round(float(la), 4))] = True
    site_lon = np.array([k[0] for k in sl]); site_lat = np.array([k[1] for k in sl])

    surfaces, curves, scores = {}, {}, {}
    meta = None
    truth = None
    for rd in run_dirs:
        cfg, bundle, trainer = load_run(rd, verbose=False)
        name = os.path.basename(rd.rstrip("/"))
        pred, meta = mp.predict_surface(cfg, bundle, trainer.model, trainer.device,
                                        times, step=args.step)
        surfaces[name] = pred
        if truth is None:
            truth = mp.truth_surface(cfg, meta)
        if truth is not None:
            scores[name] = mp.surface_scores(pred, truth)
            curves[name] = mp.error_vs_distance(pred, truth, meta, site_lon, site_lat)
        print(f"  {name:<18} surface predicted  "
              + (f"R2(log10)={scores[name].get('surface_r2_log10', float('nan')):.3f}  "
                 f"RMSE={scores[name].get('surface_rmse_log10', float('nan')):.3f}"
                 if truth is not None else ""))

    figs = []
    main = run_dirs[0].rstrip("/").split("/")[-1]
    figs.append(mp.plot_fill(surfaces[main], truth, meta, site_lon, site_lat,
                             os.path.join(out_dir, f"fill_{main}.png"),
                             title=f"UFP surface predicted by '{main}'"))
    if len(surfaces) > 1:
        figs.append(mp.plot_variant_maps(surfaces, truth, meta, site_lon, site_lat,
                                         os.path.join(out_dir, "fill_variants.png"),
                                         t_index=min(1, len(times) - 1)))
    if curves:
        figs.append(mp.plot_error_vs_distance(
            curves, os.path.join(out_dir, "error_vs_distance.png")))

    if scores:
        print("\n" + "=" * 78)
        print("whole-surface skill against the known true field")
        print("=" * 78)
        print(f"{'variant':<18}{'R2 (log10)':>14}{'RMSE (log10)':>16}{'bias':>12}{'spatial r':>12}")
        print("-" * 78)
        for k, v in scores.items():
            print(f"{k:<18}{v.get('surface_r2_log10', np.nan):>14.4f}"
                  f"{v.get('surface_rmse_log10', np.nan):>16.4f}"
                  f"{v.get('surface_bias_log10', np.nan):>12.4f}"
                  f"{v.get('surface_spatial_r', np.nan):>12.4f}")
        with open(os.path.join(out_dir, "surface_scores.json"), "w") as fh:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in scores.items()},
                      fh, indent=2)
    print("\nfigures:")
    for f in figs:
        print(f"  {f}")
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze_run

    analyze_run(args.run_dir, compare_dir=args.compare, out_dir=args.out)
    return 0


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(prog="ufp_pcl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="print an inventory of a NetCDF file")
    p.add_argument("path")
    p.add_argument("--emit-config", default=None, help="write a starter YAML config")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("train", help="train one model")
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[], metavar="key=value")
    p.add_argument("--analyze", action="store_true", help="run the latent report afterwards")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="train several fusion variants and tabulate")
    p.add_argument("config")
    p.add_argument("--variants", default="obs_only,proxy_stacked,fusion,pcl")
    p.add_argument("--set", action="append", default=[], metavar="key=value")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="sweep one config value")
    p.add_argument("config")
    p.add_argument("--param", required=True, help="dotted key, e.g. train.rho")
    p.add_argument("--values", required=True, help="comma separated")
    p.add_argument("--set", action="append", default=[], metavar="key=value")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("loso-cv", help="leave-one-site-out CV over every site")
    p.add_argument("config")
    p.add_argument("--variants", default="obs_only,fusion,pcl")
    p.add_argument("--set", action="append", default=[], metavar="key=value")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_loso_cv)

    p = sub.add_parser("map", help="predict the surface everywhere and score the fill")
    p.add_argument("runs", help="comma-separated run directories; the first is the focus")
    p.add_argument("--n-times", type=int, default=3)
    p.add_argument("--step", type=float, default=None, help="grid spacing in degrees")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("analyze", help="latent-space and attribution report for a run")
    p.add_argument("run_dir")
    p.add_argument("--compare", default=None, help="a second run to contrast against")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
