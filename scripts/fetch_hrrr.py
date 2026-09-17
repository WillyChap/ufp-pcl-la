#!/usr/bin/env python3
"""Pull HRRR 3 km hourly analyses for the study domain into a ufp_pcl-readable NetCDF.

Why bother, when ERA5 is already in the model: across the seven SCAQMD monitors, ERA5
(0.25 deg) takes **three** distinct values and MERRA-2 takes **two**.  Thirty-four of the
model's thirty-eight predictors are effectively constant over the network while UFP varies
3.5x across it.  HRRR is 3 km, so the same monitors land in their own cells, and unlike
ERA5-Land -- which is a different land-surface model masquerading as a resolution change,
and measurably hurt -- this is the same kind of product at an eight-fold finer grid.

It also targets the right half of the error.  95.5% of the model's test error is *within*
site, hour to hour, which a static land-surface embedding cannot touch; boundary-layer
height and ventilation are what drive it, and those are exactly what a 3 km analysis
resolves and a 25 km one smears.

Source is the MesoWest/University of Utah Zarr mirror (s3://hrrrzarr, anonymous), not the
GRIB archive: the analysis fields there are chunked 150x150, so this domain costs two
chunk reads per variable per hour instead of pulling a ~150 MB CONUS GRIB and discarding
99.9% of it.  Two years is ~136 GB the GRIB way and a few GB this way.

HRRR is on a Lambert conformal grid, so the subset is resampled once onto a regular
lat/lon grid -- nearest neighbour, since at 3 km native against a ~3 km target there is
nothing to gain from interpolating and something to lose from smoothing gradients.

    python scripts/fetch_hrrr.py --start 2023-08-02 --end 2025-06-18 \
        --out data/real/hrrr_la.nc

Missing hours (the archive has occasional gaps) are written as NaN rather than skipped,
so the time axis stays regular and `data.valid_ranges` / `drop_nan_obs` handle them.
"""
from __future__ import annotations

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

BUCKET = "hrrrzarr"
# level -> (HRRR name, output name).  Chosen to mirror the ERA5 block already in the
# config so the two can be swapped one-for-one and the comparison isolates resolution.
VARS = [
    ("surface",           "HPBL",  "hrrr_blh"),   # boundary layer height  <- the key field
    ("surface",           "DSWRF", "hrrr_ssrd"),
    ("surface",           "PRES",  "hrrr_sp"),
    ("surface",           "GUST",  "hrrr_gust"),
    ("2m_above_ground",   "TMP",   "hrrr_t2m"),
    ("2m_above_ground",   "DPT",   "hrrr_d2m"),
    ("2m_above_ground",   "RH",    "hrrr_rh"),
    ("10m_above_ground",  "UGRD",  "hrrr_u10"),
    ("10m_above_ground",  "VGRD",  "hrrr_v10"),
    ("entire_atmosphere", "TCDC",  "hrrr_tcc"),
]


def load_grid(fs):
    with fs.open(f"{BUCKET}/grid/HRRR_latlon.h5") as f:
        raw = f.read()
    import h5py
    h = h5py.File(io.BytesIO(raw), "r")
    return np.asarray(h["latitude"][:], float), np.asarray(h["longitude"][:], float)


def index_box(lat, lon, bbox, pad=0.08):
    lo, la0, hi, la1 = bbox
    m = (lat >= la0 - pad) & (lat <= la1 + pad) & (lon >= lo - pad) & (lon <= hi + pad)
    if not m.any():
        raise SystemExit("bounding box does not intersect the HRRR grid")
    iy, ix = np.where(m)
    return slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1)


def read_one(fs, stamp, level, var, sy, sx):
    """One (hour, variable) slice, or None when the archive has no such analysis."""
    import zarr
    import s3fs as _s3
    d, hh = stamp.strftime("%Y%m%d"), stamp.strftime("%H")
    p = (f"{BUCKET}/sfc/{d}/{d}_{hh}z_anl.zarr/{level}/{var}/{level}/{var}")
    try:
        z = zarr.open_array(_s3.S3Map(p, s3=fs), mode="r", zarr_format=2)
        return np.asarray(z[sy, sx], dtype=np.float32)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-08-02")
    ap.add_argument("--end", default="2025-06-18")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-118.60, 33.65, -117.00, 34.30],
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--res", type=float, default=0.03, help="output grid spacing, degrees")
    ap.add_argument("--out", default="data/real/hrrr_la.nc")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--vars", nargs="*", default=None,
                    help="subset of output names to fetch; default all")
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many hours")
    args = ap.parse_args()

    import s3fs
    from scipy.spatial import cKDTree
    fs = s3fs.S3FileSystem(anon=True)

    lat2d, lon2d = load_grid(fs)
    sy, sx = index_box(lat2d, lon2d, args.bbox)
    sub_lat, sub_lon = lat2d[sy, sx], lon2d[sy, sx]
    print(f"HRRR subset rows {sy.start}..{sy.stop-1} cols {sx.start}..{sx.stop-1} "
          f"-> {sub_lat.shape[0]} x {sub_lat.shape[1]} cells at ~3 km")

    lo, la0, hi, la1 = args.bbox
    tlat = np.arange(la0, la1 + 1e-9, args.res)
    tlon = np.arange(lo, hi + 1e-9, args.res)
    LA, LO = np.meshgrid(tlat, tlon, indexing="ij")
    kx = 111.32 * np.cos(np.deg2rad(float(tlat.mean())))
    tree = cKDTree(np.column_stack([sub_lat.ravel() * 110.57, sub_lon.ravel() * kx]))
    dist, idx = tree.query(np.column_stack([LA.ravel() * 110.57, LO.ravel() * kx]), k=1)
    far = dist > 4.0
    print(f"output grid {len(tlat)} x {len(tlon)} at {args.res} deg; "
          f"nn distance median {np.median(dist):.2f} km, max {dist.max():.2f} km")

    use = [t for t in VARS if (args.vars is None or t[2] in args.vars)]
    if not use:
        raise SystemExit(f"no variables matched {args.vars}")
    times = pd.date_range(args.start, args.end, freq="h", tz=None)
    if args.limit:
        times = times[: args.limit]
    print(f"{len(times)} hourly analyses, {len(use)} variables\n")

    ny, nx = len(tlat), len(tlon)
    out = {name: np.full((len(times), ny, nx), np.nan, np.float32) for _, _, name in use}
    jobs = [(ti, t, lev, v, name)
            for ti, t in enumerate(times) for lev, v, name in use]

    done = [0]
    def work(job):
        ti, t, lev, v, name = job
        a = read_one(fs, t, lev, v, sy, sx)
        if a is not None:
            g = a.ravel()[idx].astype(np.float32)
            g[far] = np.nan
            out[name][ti] = g.reshape(ny, nx)
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"  {done[0]}/{len(jobs)} reads", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))

    got = {n: float(np.isfinite(out[n]).mean()) for _, _, n in use}
    print("\ncoverage after fetch:")
    for _, _, n in use:
        print(f"  {n:12s} {got[n]:6.1%}")

    ds = xr.Dataset({n: (("time", "latitude", "longitude"), out[n]) for _, _, n in use},
                    coords={"time": times.values, "latitude": tlat, "longitude": tlon})
    ds.attrs["source"] = "NOAA HRRR 3 km analyses via s3://hrrrzarr (MesoWest / U. Utah)"
    for lev, v, n in use:
        ds[n].attrs = {"hrrr_level": lev, "hrrr_name": v}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    enc = {n: {"zlib": True, "complevel": 4, "dtype": "float32"} for _, _, n in use}
    ds.to_netcdf(args.out, engine="netcdf4", encoding=enc)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
