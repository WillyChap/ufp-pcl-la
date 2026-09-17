#!/usr/bin/env python3
"""Road-class-weighted network density, from the full arterial network.

The first version of the traffic fields used only `motorway|trunk` -- 12,620 ways, 7% of
the basin's road network -- which left Central LA reading 22 five-axle trucks on its
nearest road.  That is true of the nearest *freeway* and useless as a description of a
dense urban site: UFP at an urban-background monitor is driven by the street network
around it, not by a freeway 4 km away.

Emission weights are per-class vehicle-activity proxies, not counts: a motorway lane
carries far more traffic than a tertiary street, and heavy-duty diesel -- the dominant
ultrafine source -- is concentrated on motorways and trunk routes.  Where OSM tags
`lanes`, the weight is scaled by them.

Produces, for several buffer radii, the total class-weighted road length near each cell.
Unlike nearest-postmile AADT this is defined everywhere, including off the state highway
network where Caltrans has no postmiles at all.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import xarray as xr

# relative vehicle activity by class; motorway = 1.0
WEIGHT = {"motorway": 1.00, "trunk": 0.70, "primary": 0.35,
          "secondary": 0.18, "tertiary": 0.08}


def segments(ways):
    """-> midpoint lon/lat and weighted length (km) for every road segment."""
    mx, my, w = [], [], []
    for way in ways:
        g = way.get("geometry") or []
        if len(g) < 2:
            continue
        t = way.get("tags", {})
        base = WEIGHT.get(t.get("highway"), 0.0)
        if base <= 0:
            continue
        try:
            lanes = float(str(t.get("lanes", "")).split(";")[0])
        except ValueError:
            lanes = np.nan
        scale = base * (np.clip(lanes, 1, 8) / 2.0 if np.isfinite(lanes) else 1.0)
        for a, b in zip(g[:-1], g[1:]):
            mx.append(0.5 * (a["lon"] + b["lon"]))
            my.append(0.5 * (a["lat"] + b["lat"]))
            w.append(scale * np.hypot((b["lat"] - a["lat"]) * 110.57,
                                      (b["lon"] - a["lon"]) * 92.3))
    return np.array(mx), np.array(my), np.array(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm", default="/tmp/osm_all_roads.json")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-118.60, 33.65, -117.00, 34.30])
    ap.add_argument("--res", type=float, default=0.002)
    ap.add_argument("--radii", type=float, nargs="*", default=[0.3, 1.0, 3.0])
    ap.add_argument("--out", default="data/real/roads_la.nc")
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    ways = json.load(open(args.osm))["elements"]
    mx, my, w = segments(ways)
    print(f"{len(ways)} ways -> {len(mx)} weighted segments, total weighted length "
          f"{w.sum():,.0f} km-equivalents")

    lo, la0, hi, la1 = args.bbox
    lat = np.arange(la0, la1 + 1e-9, args.res)
    lon = np.arange(lo, hi + 1e-9, args.res)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    kx = 111.32 * np.cos(np.deg2rad(float(lat.mean())))
    tree = cKDTree(np.column_stack([my * 110.57, mx * kx]))
    q = np.column_stack([LA.ravel() * 110.57, LO.ravel() * kx])
    print(f"grid {len(lat)} x {len(lon)}  ({len(q):,} cells)")

    out = {}
    for R in args.radii:
        acc = np.zeros(len(q), np.float32)
        # blocked so the neighbour lists never all exist at once
        for s in range(0, len(q), 20000):
            e = min(s + 20000, len(q))
            for i, nb in enumerate(tree.query_ball_point(q[s:e], R)):
                if nb:
                    acc[s + i] = float(w[nb].sum())
        name = f"road_w_{str(R).replace('.', 'p')}km"
        out[name] = acc.reshape(len(lat), len(lon))
        print(f"  {name}: median {np.median(acc):.2f}  p99 {np.percentile(acc, 99):.2f}")

    ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in out.items()},
                    coords={"lat": lat, "lon": lon})
    ds.attrs["source"] = "OpenStreetMap via Overpass; class- and lane-weighted road length"
    ds.attrs["weights"] = json.dumps(WEIGHT)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.to_netcdf(args.out, engine="netcdf4",
                 encoding={k: {"zlib": True, "complevel": 4} for k in out})
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
