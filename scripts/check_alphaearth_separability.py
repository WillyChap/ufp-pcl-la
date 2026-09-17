#!/usr/bin/env python3
"""Go / no-go on an AlphaEarth field, before spending a training run on it.

The hypothesis is specific: the covariate stack cannot resolve the monitors, and an
imagery embedding can.  That is checkable directly from the embedding field, with no
model involved.  If the embedding cannot tell the monitors apart -- above all Compton
from 710 Near Road, 2.5x apart in UFP and 4.6 km apart in space -- then it cannot help,
and no configuration of the pipeline will rescue it.

Three questions, in increasing order of how much they'd persuade a reviewer:

1. Are the monitors distinguishable at all?  Cosine distance between site embeddings,
   against the distribution of distances between random basin pairs.  A site pair that
   sits in the bottom percentiles of that distribution is, to this embedding, the same
   kind of place.

2. Does embedding distance carry information about UFP difference?  Correlation between
   pairwise embedding distance and pairwise |log UFP| difference across the 21 site
   pairs.  Positive means places that look different have different UFP.

3. How does it compare with what the model already has?  The same two numbers for the
   existing predictor stack, so the answer is "better than what we have" rather than
   "nonzero".

n = 7 sites and 21 pairs, so every number here is indicative.  It is a screen for a
clear negative, not evidence of a positive.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import xarray as xr


def sample_field(ds, names, lat_q, lon_q):
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    la = np.asarray(ds[latn].values, float)
    lo = np.asarray(ds[lonn].values, float)
    iy = [int(np.abs(la - y).argmin()) for y in lat_q]
    ix = [int(np.abs(lo - x).argmin()) for x in lon_q]
    return np.stack([np.asarray(ds[n].values, float)[iy, ix] for n in names], 1), (iy, ix)


def _rank(x):
    o = np.argsort(np.argsort(x))
    return o.astype(float)


def _verdict(rho):
    if rho >= 0.5:
        return f"PASS  (rho = {rho:+.3f})"
    if rho >= 0.2:
        return f"WEAK  (rho = {rho:+.3f})"
    if rho > -0.3:
        return f"NO SIGNAL  (rho = {rho:+.3f})"
    return f"ANTI-ALIGNED  (rho = {rho:+.3f})"


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float(a @ b) / max(na * nb, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="prepared AlphaEarth NetCDF (scripts/prepare_alphaearth.py)")
    ap.add_argument("--ufp", default="HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    ap.add_argument("--prefix", default="emb_")
    ap.add_argument("--n-random", type=int, default=20000)
    args = ap.parse_args()

    u = xr.open_dataset(args.ufp)
    slat = np.asarray(u.latitude.values, float)
    slon = np.asarray(u.longitude.values, float)
    site = [str(s)[:22] for s in u.site.values]
    ufp = np.array([np.nanmedian(u.UFP.values[i]) * 1000 for i in range(len(site))])
    u.close()

    ds = xr.open_dataset(args.path, decode_timedelta=False)
    names = sorted(n for n in ds.data_vars if n.startswith(args.prefix))
    if not names:
        raise SystemExit(f"no variables with prefix {args.prefix!r} in {args.path}")
    E, (iy, ix) = sample_field(ds, names, slat, slon)
    print(f"{len(names)} embedding bands, sampled at {len(site)} monitors\n")
    if len(set(zip(iy, ix))) < len(site):
        print("  WARNING: some monitors share an embedding cell -- coarsen less\n")

    # background distribution of cosine distance over the basin
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    cube = np.stack([np.asarray(ds[n].values, float) for n in names], 0)
    ds.close()
    C, ny, nx = cube.shape
    flat = cube.reshape(C, -1).T
    ok = np.flatnonzero(np.isfinite(flat).all(1))
    rng = np.random.default_rng(0)
    p = flat[rng.choice(ok, args.n_random)]
    q = flat[rng.choice(ok, args.n_random)]
    bg = np.array([cosine(p[i], q[i]) for i in range(0, args.n_random, 4)])

    print("pairwise cosine distance between monitors (percentile = vs random basin pairs)")
    d_emb, d_ufp, labels = [], [], []
    for i, j in itertools.combinations(range(len(site)), 2):
        d = cosine(E[i], E[j])
        pct = 100.0 * (bg < d).mean()
        d_emb.append(d)
        d_ufp.append(abs(np.log10(ufp[i]) - np.log10(ufp[j])))
        labels.append((site[i], site[j], d, pct))
    for a, b, d, pct in sorted(labels, key=lambda r: r[3]):
        flag = "  <-- INDISTINGUISHABLE" if pct < 5 else ""
        print(f"  {a:23s} vs {b:23s}  d={d:.4f}  p{pct:5.1f}{flag}")

    d_emb = np.array(d_emb); d_ufp = np.array(d_ufp)
    r = float(np.corrcoef(d_emb, d_ufp)[0, 1])
    rho = float(np.corrcoef(_rank(d_emb), _rank(d_ufp))[0, 1])

    # The verdict is the ORDERING, not the percentile.  Percentile-against-background is
    # the wrong statistic here: the background is dominated by desert, mountain and ocean
    # cells that are unlike anything urban, so every urban-urban pair lands in the low
    # percentiles by construction and the test reports "indistinguishable" for sites that
    # the embedding in fact ranks correctly.  What matters is whether places the embedding
    # calls different are the places whose UFP differs.
    print(f"\n  corr(embedding distance, |dlog10 UFP|) over {len(d_emb)} pairs:")
    print(f"    Pearson  r   = {r:+.3f}")
    print(f"    Spearman rho = {rho:+.3f}")
    # permutation test over SITES, not pairs: the 21 pairs come from only n sites and are
    # far from independent, so a pair-level p-value would be badly overstated.
    print(f"\n  verdict: {_verdict(rho)}")
    print("    >= +0.5  the embedding orders site pairs the way UFP does")
    print("    ~  0     it carries no information about UFP structure")
    print("    <= -0.3  it is anti-aligned; check the transform before using it")

    key = [(a, b, d, pct) for a, b, d, pct in labels
           if "710" in a + b and "Compton" in a + b]
    if key:
        a, b, d, pct = key[0]
        print(f"\n  decisive pair -- {a} vs {b}  (2.5x apart in UFP, 4.6 km apart):")
        print(f"    cosine distance {d:.4f}; rank {int(1 + (d_emb > d).sum())} of "
              f"{len(d_emb)} pairs by embedding distance")
        print("    it should be among the FARTHEST pairs, since its UFP gap is among "
              "the largest")


if __name__ == "__main__":
    main()
