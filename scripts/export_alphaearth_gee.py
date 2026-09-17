#!/usr/bin/env python3
"""Export AlphaEarth / Satellite Embedding tiles for the UFP study domain.

Collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`: 64 bands named A00..A63, 10 m,
annual composites from 2017, each pixel a unit-length 64-vector summarising a year of
Sentinel-1/2, Landsat 8/9, GEDI, GLO-30, ERA5-Land, PALSAR-2 and GRACE observations.

Why this dataset for this problem.  Of the 38 predictors currently in the model, 34 take
three or fewer distinct values across the seven monitors -- ERA5 and GEOS-CF are 0.25 deg,
MERRA-2 coarser still.  The sharpest signal in the network, Compton (12,000 cm-3) against
710 Near Road (30,000 cm-3), is a 2.5x contrast over 4.6 km, and nothing in the stack
except TEMPO resolves it.  That contrast is a land-surface fact -- a freight corridor, a
rail yard, warehouse density -- which is exactly what an imagery embedding encodes and
what the covariate stack has no representation of at all.

Two exports per year:

  mean  the embedding averaged from 10 m to `--scale`, renormalised downstream
  sd    the within-cell standard deviation, which `prepare_alphaearth.py` turns into
        `emb_heterogeneity`.  For near-road work this is often the more useful band: a
        cell holding both a freeway and a residential block is heterogeneous in a way
        neither its mean embedding nor a 2 km retrieval records.

Usage:

    pip install earthengine-api
    earthengine authenticate
    python scripts/export_alphaearth_gee.py --project YOUR_GCP_PROJECT

Exports land in Google Drive (folder `--folder`).  Download them, then:

    python scripts/prepare_alphaearth.py ae_mean_2023_100m.tif \
        --sd ae_sd_2023_100m.tif --n-components 16 --out data/real/alphaearth_la.nc

Sizes: the domain is ~2.02 x 2.02 deg.  At 100 m that is ~4.3M cells x 64 bands, about
1.1 GB per export as float32.  `--scale 200` cuts that ~4x and still puts ~23 cells
between Compton and 710 Near Road, so it is a reasonable first pass if Drive is tight.
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def domain_from(path: str, pad: float):
    """Bounds of the analysis grid, padded so edge cells still interpolate."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    lat = np.asarray(ds.latitude.values, float)
    lon = np.asarray(ds.longitude.values, float)
    ds.close()
    return [float(lon.min()) - pad, float(lat.min()) - pad,
            float(lon.max()) + pad, float(lat.max()) + pad]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Cloud project for Earth Engine")
    ap.add_argument("--grid", default="HARMONIZED_MASTER_FILES/TEMPO_MASTER.nc",
                    help="file whose lat/lon define the domain to cover")
    ap.add_argument("--years", type=int, nargs="*", default=[2023, 2024])
    ap.add_argument("--scale", type=int, default=100, help="export resolution in metres")
    ap.add_argument("--pad", type=float, default=0.05, help="degrees of margin")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="explicit domain, overriding --grid.  At 30 m the full TEMPO "
                         "footprint is ~13 GB of embedding, most of it desert and ocean "
                         "the model has no business predicting over; restricting to the "
                         "urbanised basin buys the resolution that actually separates "
                         "near-road sites from their neighbours.")
    ap.add_argument("--folder", default="alphaearth_ufp", help="Google Drive folder")
    ap.add_argument("--no-sd", action="store_true", help="skip the dispersion export")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    args = ap.parse_args()

    box = list(args.bbox) if args.bbox else domain_from(args.grid, args.pad)
    span_km = ((box[3] - box[1]) * 110.6, (box[2] - box[0]) * 111.3 * np.cos(np.deg2rad(34)))
    cells = (span_km[0] * 1000 / args.scale) * (span_km[1] * 1000 / args.scale)
    print(f"domain   lon {box[0]:.3f}..{box[2]:.3f}  lat {box[1]:.3f}..{box[3]:.3f}"
          f"  ({span_km[0]:.0f} x {span_km[1]:.0f} km)")
    print(f"scale    {args.scale} m  ->  ~{cells/1e6:.1f}M cells, "
          f"~{cells*64*4/1e9:.2f} GB per 64-band export")
    print(f"years    {args.years}   drive folder: {args.folder}")
    if args.dry_run:
        print("\n--dry-run: nothing submitted")
        return

    import ee
    ee.Initialize(project=args.project)
    aoi = ee.Geometry.Rectangle(box)

    for year in args.years:
        col = (ee.ImageCollection(COLLECTION)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .filterBounds(aoi))
        n = col.size().getInfo()
        if n == 0:
            print(f"  {year}: no imagery in the collection for this year -- skipped")
            continue
        # mosaic() drops the projection, and reduceResolution refuses to run without one,
        # so carry the native 10 m projection across from a source tile explicitly.
        proj = col.first().select(0).projection()
        img = col.mosaic().setDefaultProjection(proj)
        # reduceResolution aggregates the native 10 m pixels inside each output cell;
        # without it, reprojection would point-sample and throw away the sub-cell detail
        # that is the entire reason for using a 10 m product.
        for kind, red in (("mean", ee.Reducer.mean()), ("sd", ee.Reducer.stdDev())):
            if kind == "sd" and args.no_sd:
                continue
            out = (img.reduceResolution(reducer=red, maxPixels=1024)
                      .reproject(crs="EPSG:4326", scale=args.scale)
                      .toFloat())
            name = f"ae_{kind}_{year}_{args.scale}m"
            ee.batch.Export.image.toDrive(
                image=out, description=name, fileNamePrefix=name,
                folder=args.folder, region=aoi, scale=args.scale,
                crs="EPSG:4326", maxPixels=int(1e10), fileFormat="GeoTIFF",
            ).start()
            print(f"  submitted {name}")
    print("\ntrack with:  earthengine task list      (or the Code Editor Tasks tab)")


if __name__ == "__main__":
    main()
