"""Put GOES-ABI AOD on a regular lat/lon grid so the toolbox can read it.

GOES_ABI_MASTER.nc stores AOD on the ABI fixed grid: the `x`/`y` axes are satellite
scan angles in radians, and geolocation lives in 2-D `lat`/`lon` arrays.  Every other
file in the collection -- and every grid reader in ufp_pcl.data.netcdf -- assumes 1-D
lat/lon axes, so the field has to be resampled once, offline.

The target grid is TEMPO's (102x102 at 0.02 deg).  That is deliberate: the proxy and
the main satellite predictor then share a grid and a domain, and TEMPO's resolution is
close enough to ABI's ~2.4 km that nearest-neighbour resampling neither invents nor
destroys structure.  Bilinear would be wrong here -- AOD is ~84% missing after cloud
screening, and interpolating across those gaps would smear retrievals into cloud.

Also NaNs the trailing rows that the harmonisation padded by repeating the last valid
field, which would otherwise present a frozen AOD map as if it were 10 days of data.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

SRC_SHAPE = (86, 105)


def build_index(glat, glon, tlat, tlon, cutoff_km):
    """Nearest ABI pixel for each target cell, in a local equal-ish-area km frame."""
    lat0 = float(tlat.mean())
    kx = 111.32 * np.cos(np.deg2rad(lat0))
    ky = 110.57
    src = np.column_stack([glat.ravel() * ky, glon.ravel() * kx])
    la, lo = np.meshgrid(tlat, tlon, indexing="ij")
    tgt = np.column_stack([la.ravel() * ky, lo.ravel() * kx])
    dist, ind = cKDTree(src).query(tgt, k=1)
    return ind, dist <= cutoff_km, dist


def last_valid_row(times):
    """Index one past the last genuinely-observed row (the pad repeats the final stamp).

    The harmonisation padded short records by repeating their final timestamp, so the
    tail is a run of identical stamps; exactly one of them is real.
    """
    t = times.astype("datetime64[m]").astype(np.int64)
    return len(t) - int(np.sum(t == t[-1])) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="HARMONIZED_MASTER_FILES/GOES_ABI_MASTER.nc")
    ap.add_argument("--grid", default="HARMONIZED_MASTER_FILES/TEMPO_MASTER.nc",
                    help="file whose 1-D latitude/longitude define the target grid")
    ap.add_argument("--out", default="data/real/goes_aod.nc")
    ap.add_argument("--cutoff-km", type=float, default=2.5,
                    help="drop target cells with no ABI pixel this close")
    ap.add_argument("--max-dqf", type=int, default=1,
                    help="keep retrievals with DQF <= this (0 high, 1 medium)")
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()

    grid = xr.open_dataset(args.grid, decode_timedelta=False)
    tlat = np.asarray(grid.latitude.values, float)
    tlon = np.asarray(grid.longitude.values, float)
    grid.close()

    ds = xr.open_dataset(args.src, decode_timedelta=False)
    glat = ds.lat.values.reshape(SRC_SHAPE).astype(float)
    glon = ds.lon.values.reshape(SRC_SHAPE).astype(float)
    ind, ok, dist = build_index(glat, glon, tlat, tlon, args.cutoff_km)
    print(f"target {len(tlat)}x{len(tlon)}  nn distance km: "
          f"median {np.median(dist):.2f}  max {dist.max():.2f}  "
          f"cells covered {ok.mean():.1%}")

    times = ds.t.values
    nt = len(times)
    n_real = last_valid_row(times)
    if n_real < nt:
        print(f"trailing pad: rows {n_real}..{nt - 1} repeat {times[n_real - 1]} -> NaN")

    ny, nx = len(tlat), len(tlon)
    out = np.full((nt, ny, nx), np.nan, np.float32)
    for s in range(0, n_real, args.chunk):
        e = min(s + args.chunk, n_real)
        a = ds.AOD.isel(t=slice(s, e)).values.reshape(e - s, -1)
        q = ds.DQF.isel(t=slice(s, e)).values.reshape(e - s, -1)
        a = np.where(np.isfinite(q) & (q <= args.max_dqf), a, np.nan)
        g = a[:, ind]
        g[:, ~ok] = np.nan
        out[s:e] = g.reshape(e - s, ny, nx).astype(np.float32)
        print(f"  {e}/{n_real}", end="\r", flush=True)
    ds.close()

    valid = np.isfinite(out[:n_real])
    print(f"\nregridded: {valid.mean():.2%} of space-time cells valid; "
          f"{(valid.reshape(n_real, -1).any(1)).mean():.1%} of hours carry data")

    res = xr.Dataset(
        {"goes_aod": (("time", "latitude", "longitude"), out)},
        coords={"time": times, "latitude": tlat, "longitude": tlon},
    )
    res.goes_aod.attrs = {
        "long_name": "GOES-ABI aerosol optical depth at 550 nm",
        "units": "1",
        "note": (f"nearest-neighbour from the ABI fixed grid, cutoff {args.cutoff_km} km, "
                 f"DQF <= {args.max_dqf}; padded rows blanked"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    enc = {"goes_aod": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    res.to_netcdf(args.out, engine="netcdf4", encoding=enc)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
