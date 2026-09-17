#!/usr/bin/env python3
"""Render every trained model's UFP surface on one shared colour scale.

Maps are the deliverable, and they are the only diagnostic that separates these models:
monitor R2 moves by 0.001 across configurations whose surfaces differ enormously.  A
shared scale matters -- per-panel normalisation makes a saturated blob and a structured
field look equally plausible, which is how the first baseline map flattered itself.

Overlays the OSM motorway network so the freeway bands can be checked against geometry
rather than judged by eye, and marks the monitors with their median UFP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# runnable from anywhere: the repo root is this file's parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def surface(run: str, cfg_path: str, hour_local: int, step: float):
    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.mapping import predict_surface, pick_times
    cfg = load_config(cfg_path)
    b = build_datasets(cfg, verbose=False)
    dev = resolve_device("auto")
    sd = torch.load(f"{run}/best.pt", map_location=dev, weights_only=False)
    m = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
    m.load_state_dict(sd["model"], strict=False)
    m.eval()
    S, meta = predict_surface(cfg, b, m, dev, pick_times(cfg, 1, hours_local=(hour_local,)),
                              step=step)
    r2 = json.load(open(f"{run}/metrics.json")).get("test_r2", float("nan"))
    del b, m
    return S[0], meta["lons"], meta["lats"], r2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*",
                    default=["base:outputs/real/m_base:configs/map_base.yaml",
                             "static:outputs/real/m_static:configs/map_static.yaml",
                             "all:outputs/real/m_all:configs/map_all.yaml",
                             "climo_best2:outputs/real/m_climo_best2:configs/map_climo_best2.yaml"])
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--step", type=float, default=0.008)
    ap.add_argument("--roads", default="/tmp/traffic_cache.json")
    ap.add_argument("--out", default="outputs/real/maps/all_surfaces.png")
    args = ap.parse_args()

    u = xr.open_dataset("HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    sl, so = u.latitude.values, u.longitude.values
    med = np.array([np.nanmedian(u.UFP.values[i]) * 1000 for i in range(7)])
    u.close()

    panels = []
    for spec in args.arms:
        name, run, cfg = spec.split(":")
        if not os.path.exists(f"{run}/best.pt"):
            print(f"  skip {name}: no model at {run}")
            continue
        print(f"  rendering {name} ...", flush=True)
        Z, lo, la, r2 = surface(run, cfg, args.hour, args.step)
        panels.append((name, lo, la, Z, r2))
    if not panels:
        raise SystemExit("no models to render")

    allv = np.concatenate([p[3][np.isfinite(p[3])] for p in panels])
    vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
    print(f"  shared colour scale: {vmin:,.0f} .. {vmax:,.0f} cm-3")

    ways = []
    if os.path.exists(args.roads):
        ways = json.load(open(args.roads)).get("ways", [])

    n = len(panels)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0 * ncol, 5.6 * nrow), squeeze=False)
    for ax, (name, lo, la, Z, r2) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(lo, la, Z, cmap="inferno", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        for w in ways:
            g = w.get("geometry") or []
            if len(g) > 1:
                ax.plot([p["lon"] for p in g], [p["lat"] for p in g],
                        color="cyan", lw=0.3, alpha=0.45, zorder=4)
        ax.scatter(so, sl, s=90, facecolors="none", edgecolors="lime", lw=1.8, zorder=6)
        for i in range(7):
            ax.annotate(f"{med[i]/1000:.0f}k", (so[i], sl[i]), fontsize=7, color="lime",
                        xytext=(4, 3), textcoords="offset points", weight="bold", zorder=7)
        ax.set_title(f"{name}   (monitor R$^2$ {r2:.3f})", fontsize=11)
        ax.set_xlim(lo.min(), lo.max()); ax.set_ylim(la.min(), la.max())
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, label="particles cm$^{-3}$")
    fig.suptitle(f"UFP surfaces, {args.hour}:00 local -- shared colour scale, "
                 f"cyan = OSM freeways, green = monitors", fontsize=13)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
