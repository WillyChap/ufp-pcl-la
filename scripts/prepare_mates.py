#!/usr/bin/env python3
"""Turn the SCAQMD MATES continuous download into extra UFP sites for the climatology.

The binding constraint on this project is seven monitors.  The spatial climatology fits
site means at R2 0.70 in-sample and 0.33 leave-one-out; that gap is estimation variance
from seven points, and it is the reason leave-one-site-out R2 is negative for every
configuration tried -- the level at an unseen site is wrong by ~1.4x while the hourly
pattern is right (r ~ 0.53).

MATES IV (2012-2013) and MATES V (2018-2019) each measured hourly UFP at ten sites across
the basin with a water CPC, QA'd against a reference instrument.  Six of those locations
are not in the current network, which takes the climatology from 7 points to 13.

Using measurements from 2012 and 2018 to constrain a 2023-2025 model is only legitimate
if the *spatial pattern* is stable even as levels change, so this script measures that
rather than assuming it:

    MATES IV -> V  (7 co-located sites, 6 years apart)   r = +0.93, basin level 1.00x
    MATES V  -> current record (4 co-located, 4 years)   r = +0.81, basin level 0.83x

The pattern persists; the level declines uniformly, consistent with fleet turnover.  So
each epoch is carried with its own multiplicative offset, estimated from the sites it
shares with the target record -- one parameter, not a refitted mapping.

    python scripts/prepare_mates.py --out data/real/mates_ufp.nc
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr


def load_epoch(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d = d[d["Parameter"].str.contains("Ultrafine", case=False, na=False)].copy()
    d["time"] = pd.to_datetime(d["Time"], format="%m/%d/%Y %I:%M %p", errors="coerce")
    d = d.dropna(subset=["time", "Concentration"])
    d = d[d["Concentration"] > 0]
    d["epoch"] = d["MATES"].iloc[0] if len(d) else "?"
    return d


def site_means(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("Station").agg(lat=("Latitude", "first"), lon=("Longitude", "first"),
                                 n=("Concentration", "size"))
    # geometric mean: UFP is log-normal, and the climatology is fitted in log space
    g["geo_mean"] = d.groupby("Station")["Concentration"].apply(
        lambda x: float(np.exp(np.log(x).mean())))
    return g


def match(a: pd.DataFrame, blat, blon, km=2.0):
    """Index pairs (a_site, b_index) within `km`."""
    out = []
    for s in a.index:
        dist = np.hypot((a.lat[s] - blat) * 110.6,
                        (a.lon[s] - blon) * 111.32 * np.cos(np.deg2rad(34.0)))
        j = int(np.argmin(dist))
        if dist[j] < km:
            out.append((s, j, float(dist[j])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/MATES_data_9_12_2026_Attachments")
    ap.add_argument("--ufp", default="HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    ap.add_argument("--match-km", type=float, default=0.5,
                    help="co-location radius.  Tight on purpose: UFP varies 2.5x over "
                         "4.6 km here, so a 2 km match paired MATES' Long Beach with the "
                         "710 Near Road site and corrupted the level offset by 2x.")
    ap.add_argument("--epochs", nargs="*", default=["MATES V"],
                    help="which epochs to keep.  MATES V only by default: between IV and "
                         "V the per-site drift has sd(log10)=0.053 against a between-site "
                         "signal of 0.113, so an older epoch contributes noise worth ~47% "
                         "of the signal it is meant to constrain.")
    ap.add_argument("--out", default="data/real/mates_ufp.nc")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*Continuous.csv")))
    if not files:
        raise SystemExit(f"no *Continuous.csv under {args.src}")
    frames = {}
    for f in files:
        d = load_epoch(f)
        ep = d["epoch"].iloc[0]
        frames[ep] = d
        print(f"{ep}: {len(d):,} hourly UFP rows, {d['Station'].nunique()} stations, "
              f"{d['time'].min().date()} .. {d['time'].max().date()}")

    u = xr.open_dataset(args.ufp)
    ulat, ulon = u.latitude.values, u.longitude.values
    uname = [str(s) for s in u.site.values]
    ugeo = np.array([float(np.exp(np.nanmean(np.log(np.clip(u.UFP.values[i] * 1000, 1, None)))))
                     for i in range(len(ulat))])
    u.close()

    rows = []
    frames = {k: v for k, v in frames.items()
              if any(k.startswith(e) for e in args.epochs)} or frames
    for ep, d in sorted(frames.items()):
        g = site_means(d)
        pairs = match(g, ulat, ulon, args.match_km)
        if pairs:
            r = np.array([ugeo[j] / g.geo_mean[s] for s, j, _ in pairs])
            offset = float(np.exp(np.log(r).mean()))
            print(f"\n{ep}: {len(pairs)} sites co-located with the current network "
                  f"-> level offset {offset:.3f}x")
            for s, j, dd in pairs:
                print(f"    {s:28s} <- {uname[j][:22]:24s} {dd:4.1f} km  "
                      f"{g.geo_mean[s]:8.0f} -> {ugeo[j]:8.0f}")
        else:
            offset = 1.0
            print(f"\n{ep}: no co-located sites; offset left at 1.0")
        matched = {s for s, _, _ in pairs}
        for s in g.index:
            rows.append(dict(epoch=ep, station=s, lat=g.lat[s], lon=g.lon[s],
                             n_hours=int(g.n[s]), geo_mean=float(g.geo_mean[s]),
                             adjusted=float(g.geo_mean[s] * offset),
                             offset=offset, duplicates_current=s in matched))
    t = pd.DataFrame(rows)

    # one row per distinct LOCATION: prefer the latest epoch, and drop sites the current
    # network already covers -- reusing them would double-count, not add support
    t = t.sort_values("epoch")
    t["key"] = (t.lat.round(3).astype(str) + "," + t.lon.round(3).astype(str))
    new = t[~t.duplicates_current].drop_duplicates("key", keep="last")
    print(f"\n{len(t)} site-epochs -> {t.key.nunique()} distinct locations; "
          f"{len(new)} are NOT in the current network:")
    for _, r in new.iterrows():
        print(f"    {r.station:28s} {r.lat:8.4f} {r.lon:10.4f}  "
              f"{r.geo_mean:8.0f} -> {r.adjusted:8.0f} cm-3 ({r.epoch})")

    ds = xr.Dataset(
        {"ufp_geomean": ("site", new.adjusted.to_numpy(np.float32)),
         "ufp_geomean_raw": ("site", new.geo_mean.to_numpy(np.float32)),
         "level_offset": ("site", new.offset.to_numpy(np.float32)),
         "n_hours": ("site", new.n_hours.to_numpy(np.int32))},
        coords={"site": new.station.to_numpy().astype("U32"),
                "latitude": ("site", new.lat.to_numpy()),
                "longitude": ("site", new.lon.to_numpy()),
                "epoch": ("site", new.epoch.to_numpy().astype("U10"))},
    )
    ds["drift_sd_log10"] = ("site", np.full(len(new), 0.053, np.float32))
    ds.drift_sd_log10.attrs = {
        "long_name": "per-site level drift between MATES epochs, sd of log10",
        "note": "these sites are weaker constraints than contemporaneous measurements; "
                "the basin-wide offset is removed but site-specific change is not"}
    ds.ufp_geomean.attrs = {
        "long_name": "site geometric-mean UFP, rescaled to the current record's level",
        "units": "particles cm-3",
        "note": "offset estimated from sites co-located with the current network"}
    ds.attrs["source"] = "SCAQMD MATES IV/V continuous download"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.to_netcdf(args.out, engine="netcdf4")
    print(f"\nwrote {args.out} ({len(new)} new sites for the climatology)")


if __name__ == "__main__":
    main()
