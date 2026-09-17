#!/usr/bin/env python3
"""Prepare AlphaEarth / Satellite Embedding fields for the ufp_pcl pipeline.

AlphaEarth Foundations (Brown et al. 2025, arXiv:2507.22291 -- reference [1] in the PCL
paper) publishes a global 64-dimensional embedding field at 10 m, as annual composites,
distributed through Earth Engine as the Satellite Embedding dataset.  Verify the exact
collection id and year coverage against the current Earth Engine catalogue before
exporting; at time of writing it is `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 2017 onward.

Two facts drive everything this script does.

**It is enormous at native resolution.**  The LA basin domain used here is about
151 x 82 km.  At 10 m that is ~124 million pixels; times 64 bands times 4 bytes is
roughly 32 GB.  Unusable as a model input.  At 100 m with 16 principal components it is
about 80 MB, which is comfortable, and 100 m still resolves a freeway corridor and a rail
yard -- the structures that matter for UFP and that a 2 km TEMPO pixel cannot see.

**It is annual.**  There is no diurnal or synoptic variation in it at all.  It is a
static-per-year surface descriptor, so it enters the model in the same slot as land use,
and it cannot help with the time dimension.  Do not expect it to explain the morning
traffic peak; expect it to explain *where* the peak is large.

Aggregation detail: the embeddings are unit-length vectors, so cell aggregation averages
the vectors and renormalises rather than averaging each band independently.  The script
also emits a per-cell dispersion band, `emb_heterogeneity`, measuring how much the 10 m
embeddings vary inside each coarse cell.  That band is often the useful one for near-road
work: a cell containing both a freeway and a residential block is heterogeneous in a way
neither its mean embedding nor a 2 km chemistry retrieval records.

Export from Earth Engine roughly like this (adjust to your domain and year):

    var aoi = ee.Geometry.Rectangle([-118.75, 33.60, -117.11, 34.34]);
    var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterDate('2023-01-01', '2024-01-01')
                .filterBounds(aoi).first();
    var mean = emb.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
                  .reproject({crs: 'EPSG:4326', scale: 100});
    var sd   = emb.reduceResolution({reducer: ee.Reducer.stdDev(), maxPixels: 1024})
                  .reproject({crs: 'EPSG:4326', scale: 100});
    Export.image.toDrive({image: mean.toFloat(), description: 'ae_mean_100m',
                          region: aoi, scale: 100, crs: 'EPSG:4326', maxPixels: 1e10});
    Export.image.toDrive({image: sd.toFloat(), description: 'ae_sd_100m',
                          region: aoi, scale: 100, crs: 'EPSG:4326', maxPixels: 1e10});

Then:

    python scripts/prepare_alphaearth.py ae_mean_100m.tif \
        --sd ae_sd_100m.tif --n-components 16 --out data/alphaearth_la.nc

The output NetCDF works in either role:

  * observation-encoder input -- add the band names to `data.obs_vars` (recommended:
    the bands describe surface context, which is what the covariate stack is missing);
  * geographic prior -- `model.location.source: precomputed`, `pretrained_path` pointing
    here.  Unlike a global location encoder, this one has the resolution to be a
    sensible prior inside one basin, so the comparison is worth running both ways.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import xarray as xr


def _open_tiled(pattern: str):
    """Merge the numbered tiles Earth Engine emits when an export exceeds its file limit.

    A large Export.image.toDrive is written as `name-0000000000-0000004096.tif` pieces on
    a regular row/column lattice rather than one file.  They share a CRS and pixel grid,
    so they can be stitched by their coordinates; `combine_by_coords` does it without
    assuming an ordering.
    """
    import glob
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no files match {pattern}")
    if len(hits) == 1:
        return _open_raster(hits[0])
    print(f"merging {len(hits)} export tiles")
    import rioxarray  # noqa: F401
    das = []
    for h in hits:
        d = xr.open_dataarray(h, engine="rasterio")
        print(f"  {os.path.basename(h)}: {d.shape}")
        das.append(d)
    m = xr.combine_by_coords([d.to_dataset(name="v") for d in das])["v"]
    lat = np.asarray(m["y"].values, float)
    lon = np.asarray(m["x"].values, float)
    cube = np.asarray(m.values, np.float32)
    for d in das:
        d.close()
    print(f"  merged -> {cube.shape[0]} bands, {cube.shape[1]} x {cube.shape[2]} pixels")
    return lat, lon, cube


def _open_raster(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (lat, lon, cube (C, ny, nx)) from a GeoTIFF or NetCDF."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and not os.path.exists(path):
        raise SystemExit(f"no such file: {path}")
    if ext in (".tif", ".tiff"):
        try:
            import rioxarray  # noqa: F401
        except ImportError as exc:                       # pragma: no cover
            raise SystemExit(
                "reading GeoTIFF needs rioxarray:  pip install rioxarray\n"
                "or convert the export to NetCDF first"
            ) from exc
        da = xr.open_dataarray(path, engine="rasterio")
        lat = np.asarray(da["y"].values, dtype=float)
        lon = np.asarray(da["x"].values, dtype=float)
        cube = np.asarray(da.values, dtype=np.float32)   # (band, y, x)
        da.close()
        return lat, lon, cube

    ds = xr.open_dataset(path, decode_timedelta=False)
    latd = next(d for d in ds.dims if str(d).lower().startswith(("lat", "y")))
    lond = next(d for d in ds.dims if str(d).lower().startswith(("lon", "x")))
    lat = np.asarray(ds[latd].values, dtype=float)
    lon = np.asarray(ds[lond].values, dtype=float)
    names = [n for n in ds.data_vars if latd in ds[n].dims and lond in ds[n].dims]
    arrs = []
    for n in sorted(names):
        da = ds[n]
        band = [d for d in da.dims if d not in (latd, lond)]
        a = da.transpose(*(band + [latd, lond])).values
        arrs.append(a.reshape(-1, len(lat), len(lon)))
    cube = np.concatenate(arrs, 0).astype(np.float32)
    ds.close()
    return lat, lon, cube


def fill_nodata(cube: np.ndarray) -> np.ndarray:
    """Replace no-data pixels with their nearest valid neighbour.

    AlphaEarth has no embedding over open water, so a coastal domain arrives with a few
    percent NaN.  That is fatal downstream rather than merely untidy: the precomputed
    location encoder samples this cube with grid_sample, and one NaN anywhere in the
    interpolation stencil turns the embedding, the loss and every gradient into NaN --
    training silently produces nan loss from the first epoch.

    Nearest-neighbour fill rather than zero: after PCA zero is the domain mean, which
    would drag genuinely coastal cells toward an inland-average character exactly where
    the near-shore sites sit.
    """
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(cube).all(0)
    if not bad.any():
        return cube
    _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
    out = cube.copy()
    out[:, bad] = cube[:, iy[bad], ix[bad]]
    out[~np.isfinite(out)] = 0.0
    print(f"filled {bad.mean():.2%} no-data pixels from nearest valid neighbour")
    return out


def _renormalise(cube: np.ndarray) -> np.ndarray:
    """AlphaEarth embeddings are unit vectors; keep them that way after any averaging."""
    norm = np.linalg.norm(cube, axis=0, keepdims=True)
    return cube / np.where(norm > 1e-8, norm, 1.0)


def reduce_dims(cube: np.ndarray, k: int, seed: int = 0):
    """PCA over the band axis, fitted on a subsample of valid pixels."""
    from sklearn.decomposition import PCA

    C, ny, nx = cube.shape
    flat = cube.reshape(C, -1).T                          # (pixels, C)
    ok = np.isfinite(flat).all(1)
    if ok.sum() < k * 10:
        raise SystemExit(f"only {ok.sum()} valid pixels; cannot fit {k} components")
    rng = np.random.default_rng(seed)
    idx = np.where(ok)[0]
    sub = flat[rng.choice(idx, size=min(200_000, idx.size), replace=False)]
    pca = PCA(n_components=min(k, C)).fit(sub)
    out = np.full((pca.n_components_, flat.shape[0]), np.nan, dtype=np.float32)
    out[:, ok] = pca.transform(flat[ok]).T.astype(np.float32)
    return out.reshape(-1, ny, nx), pca.explained_variance_ratio_


def coarsen(lat: np.ndarray, lon: np.ndarray, cube: np.ndarray, factor: int):
    """Block-average by an integer factor, renormalising the embedding vectors."""
    if factor <= 1:
        return lat, lon, cube, None
    C, ny, nx = cube.shape
    ny2, nx2 = (ny // factor) * factor, (nx // factor) * factor
    c = cube[:, :ny2, :nx2].reshape(C, ny2 // factor, factor, nx2 // factor, factor)
    mean = np.nanmean(c, axis=(2, 4))
    # dispersion of the fine embeddings inside each coarse cell, before renormalising
    het = np.sqrt(np.nansum(np.nanvar(c, axis=(2, 4)), axis=0)).astype(np.float32)
    lat2 = lat[:ny2].reshape(-1, factor).mean(1)
    lon2 = lon[:nx2].reshape(-1, factor).mean(1)
    return lat2, lon2, _renormalise(mean.astype(np.float32)), het


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="AlphaEarth export: GeoTIFF or NetCDF")
    ap.add_argument("--sd", default=None,
                    help="optional companion export of the within-cell standard deviation")
    ap.add_argument("--out", default="data/alphaearth.nc")
    ap.add_argument("--n-components", type=int, default=16,
                    help="PCA components to keep; 0 keeps all 64 bands")
    ap.add_argument("--coarsen", type=int, default=1,
                    help="integer block-average factor applied before PCA")
    ap.add_argument("--prefix", default="emb_")
    ap.add_argument("--year", type=int, default=None,
                    help="stamped into the attributes; AlphaEarth is an annual product")
    a = ap.parse_args()

    lat, lon, cube = (_open_tiled(a.path) if any(c in a.path for c in '*?[')
                      else _open_raster(a.path))
    print(f"read {a.path}: {cube.shape[0]} bands, {cube.shape[1]} x {cube.shape[2]} pixels "
          f"({cube.nbytes / 1e6:.0f} MB)")

    lat, lon, cube, het = coarsen(lat, lon, cube, a.coarsen)
    if a.coarsen > 1:
        print(f"coarsened by {a.coarsen}x -> {cube.shape[1]} x {cube.shape[2]}")

    if a.sd:
        _, _, sd_cube = (_open_tiled(a.sd) if any(c in a.sd for c in "*?[")
                         else _open_raster(a.sd))
        _, _, sd_cube, _ = coarsen(np.arange(sd_cube.shape[1]),
                                   np.arange(sd_cube.shape[2]), sd_cube, a.coarsen)
        het = np.sqrt(np.nansum(sd_cube ** 2, axis=0)).astype(np.float32)
        print("within-cell heterogeneity taken from the companion stdDev export")

    explained = None
    if a.n_components and a.n_components < cube.shape[0]:
        cube, explained = reduce_dims(cube, a.n_components)
        print(f"PCA -> {cube.shape[0]} components, "
              f"{100 * explained.sum():.1f}% of embedding variance retained")

    cube = fill_nodata(cube)
    if lat[0] > lat[-1]:
        lat, cube = lat[::-1].copy(), cube[:, ::-1, :].copy()
        het = het[::-1].copy() if het is not None else None

    names = [f"{a.prefix}{i:03d}" for i in range(cube.shape[0])]
    data = {n: (("lat", "lon"), cube[i]) for i, n in enumerate(names)}
    if het is not None:
        data["emb_heterogeneity"] = (
            ("lat", "lon"), het,
            {"long_name": "dispersion of the native-resolution embeddings within each cell",
             "comment": "high where a cell mixes distinct surface types, e.g. a freeway "
                        "corridor beside residential blocks"},
        )
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon},
                    attrs={
                        "title": "AlphaEarth / Satellite Embedding field prepared for ufp_pcl",
                        "source": a.path,
                        "year": a.year if a.year is not None else "unspecified",
                        "note": ("annual product: constant within a year, so it enters the "
                                 "model as a surface-context covariate, not a time-varying one"),
                        "pca_explained_variance": (list(map(float, explained))
                                                   if explained is not None else "none"),
                    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ds.to_netcdf(a.out, engine="netcdf4",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print(f"  bands: {names[0]} .. {names[-1]}"
          + (" + emb_heterogeneity" if het is not None else ""))
    print("\nuse it either way:")
    print(f"  observation input   data.obs_vars: [..., {names[0]}, ..., {names[-1]}]")
    print(f"  geographic prior    model.location.source: precomputed")
    print(f"                      model.location.pretrained_path: {a.out}")
    print(f"                      model.location.pretrained_prefix: {a.prefix}")


if __name__ == "__main__":
    main()
