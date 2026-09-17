#!/usr/bin/env python3
"""Validate a trained model at held-out monitoring sites, not on a rendered surface.

Predicts at the site coordinates directly.  Rendering a whole surface for many times and
then sampling 13 pixels out of it costs gigabytes and will be killed; the model only ever
needed 13 columns.

Two comparisons, and the distinction matters:

  training sites   the model was fitted here, so bias should be ~0.  A bias here is a
                   calibration bug, not extrapolation error.
  MATES sites      never seen.  Bias is extrapolation; the RANK correlation is the part
                   no level offset can rescue.

Averages many daylight hours spread across the record.  An earlier version of this check
used three hours on a single day and reported a 2.5x over-prediction that was mostly that
day's weather -- the giveaway was a 1.63x bias at training sites where the model's own
test MBE is zero.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# runnable from anywhere: the repo root is this file's parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import xarray as xr

warnings.filterwarnings("ignore")


def predict_at(cfg, bundle, model, dev, lat, lon, times, batch=8192):
    from ufp_pcl.data.dataset import build_stacks, open_sources, resolve_variable_roles, sample_stacks, climo_features
    from ufp_pcl.features import compute_derived
    from ufp_pcl.data.netcdf import calendar_features, COORD_COLS
    srcs = open_sources(cfg)
    obs_vars, _ = resolve_variable_roles(cfg, srcs)
    stacks = build_stacks(srcs, obs_vars)
    n, T = len(lat), len(times)
    LA = np.repeat(np.asarray(lat, float), T)
    LO = np.repeat(np.asarray(lon, float), T)
    TD = np.tile(np.asarray(times, float), n)
    obs, names = sample_stacks(stacks, LO, LA, TD, cfg.data.interp,
                               valid_ranges=cfg.data.valid_ranges)
    if cfg.data.derived:
        obs, names = compute_derived(obs, names, LO, LA, TD, cfg.data.derived)
    climo = None
    if getattr(bundle, "climo", None) is not None:
        cf, cn = climo_features(srcs, cfg.data.climo_vars, LO, LA, TD,
                                cfg.data.interp, cfg.data.valid_ranges)
        climo = bundle.climo.predict(cf, cn)
    for s in srcs:
        s.close()
    valid = np.isfinite(obs).all(1)
    obs = np.nan_to_num(bundle.obs_scaler.transform(obs))
    cal = calendar_features(TD)
    coords = np.stack([LO, LA, cal["doy"], cal["year"], cal["hour"], cal["dow"], TD],
                      1).astype(np.float32)
    assert coords.shape[1] == len(COORD_COLS)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            o = torch.as_tensor(obs[i:i + batch], dtype=torch.float32, device=dev)
            c = torch.as_tensor(coords[i:i + batch], device=dev)
            out.append(model(o, c).float().cpu().numpy().ravel())
    p = bundle.y_scaler.inverse(np.concatenate(out).reshape(-1, 1)).ravel()
    if climo is not None:
        p = p + climo
    p = np.where(valid, p, np.nan).reshape(n, T)              # transform space (log10)
    return np.array([np.nanmean(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan
                     for r in p])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", required=True, help="name:run_dir:config")
    ap.add_argument("--mates", default="data/real/mates_ufp_daylight.nc")
    ap.add_argument("--ufp", default="HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    ap.add_argument("--n-times", type=int, default=300)
    args = ap.parse_args()

    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.data.netcdf import solar_elevation_deg, to_days

    u = xr.open_dataset(args.ufp)
    T = to_days(u.time.values)
    el = solar_elevation_deg(T, np.full(len(T), -118.2), np.full(len(T), 34.0))
    rng = np.random.default_rng(0)
    times = np.sort(rng.choice(T[el >= 5], min(args.n_times, int((el >= 5).sum())),
                               replace=False))
    tlat, tlon, tobs = [], [], []
    for i in range(u.sizes["site"]):
        v = u.UFP.values[i] * 1000
        e = solar_elevation_deg(T, np.full(len(T), float(u.longitude[i])),
                                np.full(len(T), float(u.latitude[i])))
        m = np.isfinite(v) & (v > 0) & (e >= 5)
        tlat.append(float(u.latitude[i])); tlon.append(float(u.longitude[i]))
        tobs.append(np.mean(np.log10(v[m])))
    u.close()
    mm = xr.open_dataset(args.mates)
    mlat = mm.latitude.values; mlon = mm.longitude.values
    mobs = np.log10(mm.ufp_geomean.values); mnm = [str(s) for s in mm.site.values]
    mm.close()
    print(f"{len(times)} daylight hours | {len(tlat)} training sites | {len(mlat)} MATES sites\n")
    dev = resolve_device("auto")
    print(f"  {'model':14s}{'train bias':>12s}{'MATES bias':>12s}{'MATES rank r':>14s}")
    for spec in args.runs:
        name, run, cfgp = spec.split(":")
        cfg = load_config(cfgp)
        b = build_datasets(cfg, verbose=False)
        sd = torch.load(f"{run}/best.pt", map_location=dev, weights_only=False)
        mod = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
        mod.load_state_dict(sd["model"], strict=False)
        pt = predict_at(cfg, b, mod, dev, tlat, tlon, times)
        pm = predict_at(cfg, b, mod, dev, mlat, mlon, times)
        ok = np.isfinite(pm) & np.isfinite(mobs)
        r = np.corrcoef(pm[ok], mobs[ok])[0, 1] if ok.sum() > 2 else np.nan
        print(f"  {name:14s}{10**np.nanmean(pt - tobs):11.2f}x{10**np.nanmean(pm[ok]-mobs[ok]):11.2f}x"
              f"{r:+14.3f}")
        if name.startswith("climo+"):
            for i in np.where(ok)[0]:
                print(f"      {mnm[i]:28s} obs {10**mobs[i]:7.0f}  pred {10**pm[i]:7.0f}  "
                      f"{10**(pm[i]-mobs[i]):5.2f}x")
        del b, mod


if __name__ == "__main__":
    main()
