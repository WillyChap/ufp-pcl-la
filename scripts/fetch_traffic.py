#!/usr/bin/env python3
"""Build near-road traffic predictors for the study domain: geometry + volume + trucks.

The gap this fills is the sharpest one in the dataset.  710 Near Road runs at a median
31,440 cm-3 against 9,070 at Compton -- 2.5x, over 4.6 km -- and the model currently has
no variable that knows a freeway is there, let alone how many heavy-duty trucks are on it.
Every gridded predictor in the stack is 0.25 deg or coarser and takes three or fewer
distinct values across the seven monitors.

Two free, programmatic sources (PeMS has the hourly counts but Caltrans disallows
automated access to the Clearinghouse, so hourly traffic stays a manual download; these
are the parts that can be fetched):

  OpenStreetMap (Overpass)   motorway and trunk geometry -> distance to road, road density
  Caltrans ArcGIS REST       Traffic_AADT and Truck_Volumes_AADT postmile points

Truck volume is the one to watch.  Ultrafine emission per vehicle is far higher for
heavy-duty diesel than for light-duty petrol, so `TRK_5_AXLE` -- five-axle tractors, the
port drayage fleet -- is a better physical predictor of primary UFP than total AADT, and
it is what most distinguishes the 710 corridor from an ordinary busy freeway.

Outputs a static (no time axis) NetCDF on a fine grid; the toolbox samples it bilinearly
like any other field.

    python scripts/fetch_traffic.py --out data/real/traffic_la.nc
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import requests
import xarray as xr

UA = {"User-Agent": "ufp-pcl-research/0.1 (academic air-quality modelling)"}
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
CT = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway"


def fetch_osm(bbox):
    lo, la0, hi, la1 = bbox
    q = (f'[out:json][timeout:300];'
         f'(way["highway"~"^(motorway|trunk)$"]({la0},{lo},{la1},{hi}););out geom;')
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=360)
            if r.ok:
                return r.json().get("elements", [])
        except Exception:
            continue
    raise SystemExit("Overpass unreachable; try again or use a local .osm extract")


def segments(ways):
    """Way geometries -> (n,4) array of segment endpoints, lon0 lat0 lon1 lat1."""
    seg = []
    for w in ways:
        g = w.get("geometry") or []
        for a, b in zip(g[:-1], g[1:]):
            seg.append((a["lon"], a["lat"], b["lon"], b["lat"]))
    return np.asarray(seg, float)


def fetch_points(service, fields, bbox):
    """Paginate an ArcGIS FeatureServer layer over the bounding box."""
    lo, la0, hi, la1 = bbox
    out, offset = [], 0
    while True:
        p = {"where": "1=1", "geometry": f"{lo},{la0},{hi},{la1}",
             "geometryType": "esriGeometryEnvelope", "inSR": "4326",
             "spatialRel": "esriSpatialRelIntersects", "outFields": ",".join(fields),
             "returnGeometry": "true", "outSR": "4326", "f": "json",
             "resultOffset": offset, "resultRecordCount": 1000}
        j = requests.get(f"{CT}/{service}/FeatureServer/0/query", params=p,
                         headers=UA, timeout=180).json()
        f = j.get("features", [])
        out += f
        if len(f) < 1000:
            return out
        offset += len(f)


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return np.nan


def point_arrays(feats, field):
    lon, lat, val = [], [], []
    for f in feats:
        g = f.get("geometry") or {}
        if "x" not in g:
            continue
        v = num(f["attributes"].get(field))
        if not np.isfinite(v) or v <= 0:
            continue
        lon.append(g["x"]); lat.append(g["y"]); val.append(v)
    return np.array(lon), np.array(lat), np.array(val)


def seg_distance_km(glon, glat, seg, kx, block=4000):
    """Distance from each grid point to the nearest road segment, in km.

    Done as a KD-tree over segment midpoints followed by exact point-to-segment distance
    for a handful of candidates.  The naive all-pairs form is a (grid x segments) array --
    260k x 39k here, which is 80 GB and will take a laptop down; chunking only the segment
    axis still leaves 8 GB per block, which is how the first version of this went wrong.
    """
    from scipy.spatial import cKDTree
    ax, ay = seg[:, 0] * kx, seg[:, 1] * 110.57
    bx, by = seg[:, 2] * kx, seg[:, 3] * 110.57
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    half = np.hypot(bx - ax, by - ay) * 0.5
    tree = cKDTree(np.column_stack([my, mx]))
    px, py = glon * kx, glat * 110.57
    out = np.full(px.shape, np.inf, np.float32)
    K = min(24, len(seg))
    for s0 in range(0, len(px), block):
        e0 = min(s0 + block, len(px))
        q = np.column_stack([py[s0:e0], px[s0:e0]])
        _, nb = tree.query(q, k=K)                      # (block, K) candidate segments
        A = np.stack([ax[nb], ay[nb]], -1)
        V = np.stack([bx[nb] - ax[nb], by[nb] - ay[nb]], -1)
        L2 = (V * V).sum(-1)
        L2[L2 == 0] = 1e-12
        P = np.stack([px[s0:e0], py[s0:e0]], -1)[:, None, :]
        t = np.clip(((P - A) * V).sum(-1) / L2, 0.0, 1.0)
        D = P - (A + t[..., None] * V)
        out[s0:e0] = np.sqrt((D * D).sum(-1)).min(1).astype(np.float32)
    return out


def idw(glon, glat, plon, plat, pval, kx, radius_km, eps_km=0.1):
    """Sum of value / (distance + eps) within `radius_km` -- the standard LUR form."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([plat * 110.57, plon * kx]))
    q = np.column_stack([glat * 110.57, glon * kx])
    out = np.zeros(len(q), np.float32)
    for i, nb in enumerate(tree.query_ball_point(q, radius_km)):
        if nb:
            d = np.hypot(q[i, 0] - plat[nb] * 110.57, q[i, 1] - plon[nb] * kx)
            out[i] = float(np.sum(pval[nb] / (d + eps_km)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=[-118.60, 33.65, -117.00, 34.30])
    ap.add_argument("--res", type=float, default=0.002, help="degrees (~200 m)")
    ap.add_argument("--out", default="data/real/traffic_la.nc")
    ap.add_argument("--cache", default="/tmp/traffic_cache.json")
    args = ap.parse_args()

    lo, la0, hi, la1 = args.bbox
    if os.path.exists(args.cache):
        C = json.load(open(args.cache))
        ways, aadt, truck = C["ways"], C["aadt"], C["truck"]
        print(f"cache: {len(ways)} OSM ways, {len(aadt)} AADT pts, {len(truck)} truck pts")
    else:
        ways = fetch_osm(args.bbox)
        print(f"OSM: {len(ways)} motorway/trunk ways")
        aadt = fetch_points("Traffic_AADT", ["AHEAD_AADT", "BACK_AADT", "RTE"], args.bbox)
        print(f"Caltrans AADT: {len(aadt)} postmile points")
        truck = fetch_points("Truck_Volumes_AADT",
                             ["VEHICLE_AADT_TOTAL", "TOT_TRK_AADT", "TRK_5_AXLE", "RTE"],
                             args.bbox)
        print(f"Caltrans trucks: {len(truck)} postmile points")
        json.dump({"ways": ways, "aadt": aadt, "truck": truck}, open(args.cache, "w"))

    seg = segments(ways)
    print(f"road segments: {len(seg)}")

    lat = np.arange(la0, la1 + 1e-9, args.res)
    lon = np.arange(lo, hi + 1e-9, args.res)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    kx = 111.32 * np.cos(np.deg2rad(float(lat.mean())))
    glat, glon = LA.ravel(), LO.ravel()
    print(f"grid {len(lat)} x {len(lon)} at {args.res} deg (~{args.res*111:.0f} m)")

    fields = {}
    d = seg_distance_km(glon, glat, seg, kx)
    fields["road_dist_km"] = d
    fields["road_prox"] = (1.0 / (d + 0.1)).astype(np.float32)   # sharper near-road form

    # Volume comes from the NEAREST postmile, not from a radius-limited sum.  Caltrans
    # postmiles are ~1 km apart along routes and absent off them, so a 1 km IDW returns
    # exactly zero at most places -- including all seven monitors, which makes the field
    # useless.  Pairing sparse volume with dense OSM geometry is also the standard
    # land-use-regression form: intensity = (traffic on the nearest road) / (distance to
    # the nearest road), which is what actually drives near-road enhancement.
    from scipy.spatial import cKDTree
    for feats, field, tag in [
        (aadt,  "AHEAD_AADT",   "aadt"),
        (truck, "TOT_TRK_AADT", "truck"),
        (truck, "TRK_5_AXLE",   "truck5"),
    ]:
        plon, plat, pval = point_arrays(feats, field)
        if not len(pval):
            print(f"  {tag}: no usable points"); continue
        print(f"  {tag:7s} {len(pval):5d} points, median {np.median(pval):,.0f}")
        tree = cKDTree(np.column_stack([plat * 110.57, plon * kx]))
        dd, ii = tree.query(np.column_stack([glat * 110.57, glon * kx]), k=1)
        nearest = pval[ii].astype(np.float32)
        fields[f"{tag}_nearest"] = nearest
        # intensity: volume on the nearest road, attenuated by distance to any road
        fields[f"{tag}_intensity"] = (nearest / (d + 0.1)).astype(np.float32)
        # a genuinely local sum, at a radius the postmile spacing can actually support
        fields[f"{tag}_idw_5km"] = idw(glon, glat, plon, plat, pval, kx, 5.0)

    ds = xr.Dataset({k: (("lat", "lon"), v.reshape(len(lat), len(lon)).astype(np.float32))
                     for k, v in fields.items()},
                    coords={"lat": lat, "lon": lon})
    ds.road_dist_km.attrs = {"long_name": "distance to nearest motorway/trunk", "units": "km"}
    ds.attrs["source"] = "OpenStreetMap (Overpass) + Caltrans ArcGIS Traffic/Truck AADT"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    enc = {k: {"zlib": True, "complevel": 4, "dtype": "float32"} for k in ds.data_vars}
    ds.to_netcdf(args.out, engine="netcdf4", encoding=enc)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB): {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
