#!/usr/bin/env python3
"""Render the UFP surface with its support marked, and the network bias removed.

Two corrections, both measured rather than assumed.

**Level.**  The seven AB-617 monitors have a daylight geometric mean of 14,602 cm-3
against 10,299 at six independent background sites -- they are sited at populated and
near-road locations, so they run 1.42x high, and a model fitted on them inherits that
everywhere.  Independent validation measures 1.36-1.52x over-prediction across four
structurally different models, which matches.  `data.representativeness` divides it out.

**Structure.**  Between *background* locations the model cannot discriminate: rank
correlation against six unseen sites is negative for every predictor and every
architecture tried, and a bias-corrected constant beats every model there (RMSE 1.19x
against 1.48-1.51x).  Drawing a smoothly varying background field therefore asserts
something the data does not support, and it does so in the same visual language as the
near-road corridor, which IS supported.

So the surface is split.  Cells whose road density falls inside the range the monitors
actually span are drawn normally; cells outside it -- where the only evidence is
extrapolation from a single near-road station -- are hatched.  The map then says what it
knows and marks what it is guessing.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/real/m_final")
    ap.add_argument("--config", default="configs/map_climo_mates.yaml")
    ap.add_argument("--representativeness", type=float, default=1.42)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--step", type=float, default=0.008)
    ap.add_argument("--out", default="outputs/real/maps/honest_surface.png")
    args = ap.parse_args()

    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.mapping import predict_surface, pick_times

    cfg = load_config(args.config)
    cfg.data.representativeness = args.representativeness
    b = build_datasets(cfg, verbose=False)
    dev = resolve_device("auto")
    sd = torch.load(f"{args.run}/best.pt", map_location=dev, weights_only=False)
    mod = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
    mod.load_state_dict(sd["model"], strict=False)
    S, meta = predict_surface(cfg, b, mod, dev,
                              pick_times(cfg, 1, hours_local=(args.hour,)), step=args.step)
    Z = S[0]; lo, la = meta["lons"], meta["lats"]
    print(f"surface {Z.shape}, corrected by /{args.representativeness:.2f}")

    # support = road density within the range the monitors actually sample
    u = xr.open_dataset("HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    slat, slon = u.latitude.values, u.longitude.values
    med = np.array([np.nanmedian(u.UFP.values[i]) * 1000 for i in range(7)]); u.close()
    r = xr.open_dataset("data/real/roads_la.nc")
    RA, RO = r.lat.values, r.lon.values
    grid = np.array([[float(r.road_w_0p3km.values[int(np.abs(RA - a).argmin()),
                                                  int(np.abs(RO - o).argmin())])
                      for o in lo] for a in la])
    site = np.array([float(r.road_w_0p3km.values[int(np.abs(RA - a).argmin()),
                                                 int(np.abs(RO - o).argmin())])
                     for a, o in zip(slat, slon)]); r.close()
    lo_s, hi_s = site.min(), site.max()
    supported = (grid >= lo_s) & (grid <= hi_s)
    print(f"monitors span road density {lo_s:.2f}-{hi_s:.2f}; "
          f"{supported.mean():.1%} of the domain lies inside that range")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4))
    v = Z[np.isfinite(Z)]
    vmin, vmax = np.percentile(v, 2), np.percentile(v, 98)
    for ax, masked in zip(axes, (False, True)):
        im = ax.pcolormesh(lo, la, Z, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
        if masked:
            # grey wash over cells the monitors do not constrain -- readable at a glance,
            # unlike hatching, which at this grid spacing obscures the field itself
            grey = np.where(supported, np.nan, 1.0)
            ax.pcolormesh(lo, la, grey, cmap="Greys", vmin=0, vmax=1.6,
                          alpha=0.72, shading="auto")
            ax.set_title(f"within the range the monitors sample "
                         f"({100*supported.mean():.0f}% of the domain)", fontsize=11)
        else:
            ax.set_title(f"predicted surface, {args.hour}:00 local", fontsize=11)
        ax.scatter(slon, slat, s=120, facecolors="none", edgecolors="lime", lw=2.2, zorder=6)
        for i in range(7):
            ax.annotate(f"{med[i]/1000:.0f}k", (slon[i], slat[i]), fontsize=8, color="lime",
                        xytext=(5, 4), textcoords="offset points", weight="bold", zorder=7)
        ax.set_xlim(lo.min(), lo.max()); ax.set_ylim(la.min(), la.max())
    fig.colorbar(im, ax=axes, fraction=0.02, label="particles cm$^{-3}$")
    fig.suptitle(f"Hourly UFP, network representativeness bias removed (/{args.representativeness:.2f}).  "
                 f"Grey = unconstrained by the monitoring network.", fontsize=12)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
