#!/usr/bin/env python3
"""Identify what the leading components of the learned location-time embedding represent.

The embedding's variance is 31% spatial and 68% temporal, so a single PCA of the whole
field mixes two different things. This separates them:

  spatial modes   embedding on a grid, averaged over time -> maps, correlated against
                  physical surface fields (road density, terrain, coastal distance,
                  climatological mixing depth)
  temporal modes  embedding at the monitor locations across the record -> time series,
                  correlated against solar elevation, mixing depth, temperature, season

Reporting eigenvalues without identifying the eigenvectors says nothing about what the
model learned; the correlations below are what make the components interpretable.
"""
from __future__ import annotations

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/real/m_final")
    ap.add_argument("--config", default="configs/map_climo_mates.yaml")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--n-times", type=int, default=60)
    ap.add_argument("--out", default="outputs/real/maps")
    a = ap.parse_args()
    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.latent import make_coord_grid, embed_coords
    from ufp_pcl.data.netcdf import solar_elevation_deg, to_days, days_to_datetime64

    cfg = load_config(a.config); b = build_datasets(cfg, verbose=False)
    dev = resolve_device("auto")
    sd = torch.load(f"{a.run}/best.pt", map_location=dev, weights_only=False)
    m = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
    m.load_state_dict(sd["model"], strict=False); m.eval()
    cm = b.coord_meta

    # ---------------- spatial modes: time-averaged embedding on a grid ----------------
    T = to_days(xr.open_dataset(cfg.data.path).time.values)
    el = solar_elevation_deg(T, np.full(len(T), -118.2), np.full(len(T), 34.0))
    rng = np.random.default_rng(0)
    times = np.sort(rng.choice(T[el >= 5], a.n_times, replace=False))
    coords, meta = make_coord_grid(cm["lon_min"], cm["lon_max"], cm["lat_min"],
                                   cm["lat_max"], step=a.step, time_days=times)
    E = embed_coords(m, coords, dev)                       # (T*ny*nx, d)
    Tn, ny, nx = meta["shape"]
    Eg = E.reshape(Tn, ny * nx, -1).mean(0)                # time-mean per grid cell
    Eg = Eg - Eg.mean(0)
    U, S, Vt = np.linalg.svd(Eg, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    SP = U[:, :4] * S[:4]                                   # spatial scores
    lons, lats = meta["lons"], meta["lats"]
    LA, LO = np.meshgrid(lats, lons, indexing="ij")

    # physical fields on the same grid, for identification
    def grid_of(path, var_, ln="lat", lo_="lon"):
        x = xr.open_dataset(path); A, O = x[ln].values, x[lo_].values
        g = np.array([[float(x[var_].values[int(np.abs(A - p).argmin()),
                                            int(np.abs(O - q).argmin())]) for q in lons]
                      for p in lats]); x.close(); return g.ravel()
    phys = {
        "log road density 300 m": np.log10(grid_of("data/real/roads_la.nc", "road_w_0p3km") + 1),
        "log truck intensity": np.log10(grid_of("data/real/traffic_la.nc", "truck5_intensity") + 1),
        "land-surface embedding PC1": grid_of("data/real/alphaearth_la_30m.nc", "emb_000"),
        "longitude (inland distance)": LO.ravel(),
        "latitude": LA.ravel(),
    }
    h = xr.open_dataset("data/real/hrrr_la.nc")
    blh = np.nanmean(h.hrrr_blh.values[:3000], 0); HA, HO = h.latitude.values, h.longitude.values
    phys["mean mixing depth"] = np.array([[float(blh[int(np.abs(HA - p).argmin()),
                                                     int(np.abs(HO - q).argmin())])
                                           for q in lons] for p in lats]).ravel(); h.close()

    print("SPATIAL modes of the location-time embedding (time-averaged)\n")
    print(f"  {'':6s}{'var':>8s}   correlation with physical fields")
    for k in range(4):
        cors = {n: float(np.corrcoef(SP[:, k], v)[0, 1]) for n, v in phys.items()}
        top = sorted(cors.items(), key=lambda x: -abs(x[1]))[:3]
        print(f"  PC{k+1}  {var[k]*100:6.1f}%   " +
              ",  ".join(f"{n} {c:+.2f}" for n, c in top))

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for k in range(4):
        ax = axes[0, k]
        Zk = SP[:, k].reshape(ny, nx)
        lim = np.percentile(np.abs(Zk), 98)
        im = ax.pcolormesh(lons, lats, Zk, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        cors = {n: float(np.corrcoef(SP[:, k], v)[0, 1]) for n, v in phys.items()}
        n1, c1 = max(cors.items(), key=lambda x: abs(x[1]))
        ax.set_title(f"spatial PC{k+1} — {var[k]*100:.1f}%\nbest match: {n1} (r={c1:+.2f})",
                     fontsize=9.5)
        ax.scatter(b.table["lon"][::400], b.table["lat"][::400], s=1, c="k", alpha=.25)
        plt.colorbar(im, ax=ax, fraction=0.04)

    # ---------------- temporal modes: embedding at the monitors through time ----------
    u = xr.open_dataset(cfg.data.path)
    slat, slon = u.latitude.values, u.longitude.values; u.close()
    tt = np.sort(rng.choice(T[el >= 5], 1500, replace=False))
    cl, ca = np.repeat(slon, len(tt)), np.repeat(slat, len(tt))
    ct = np.tile(tt, len(slon))
    from ufp_pcl.data.netcdf import calendar_features, COORD_COLS
    cal = calendar_features(ct)
    C = np.stack([cl, ca, cal["doy"], cal["year"], cal["hour"], cal["dow"], ct], 1).astype(np.float32)
    Et = embed_coords(m, C, dev).reshape(len(slon), len(tt), -1).mean(0)   # site-mean
    Et = Et - Et.mean(0)
    U2, S2, _ = np.linalg.svd(Et, full_matrices=False)
    var2 = (S2 ** 2) / (S2 ** 2).sum()
    TP = U2[:, :4] * S2[:4]
    ts = pd.to_datetime(days_to_datetime64(tt))
    hr = (ts.hour - 8) % 24
    solar = solar_elevation_deg(ct[:len(tt)], np.full(len(tt), -118.2), np.full(len(tt), 34.0))
    hh = xr.open_dataset("data/real/hrrr_la.nc")
    hb = hh.hrrr_blh.interp(time=days_to_datetime64(tt), latitude=34.0,
                            longitude=-118.2, method="nearest").values; hh.close()
    tphys = {"solar elevation": solar, "mixing depth": hb,
             "hour of day": hr.to_numpy().astype(float),
             "day of year": ts.dayofyear.to_numpy().astype(float)}
    print("\nTEMPORAL modes (embedding at the monitors through time)\n")
    print(f"  {'':6s}{'var':>8s}   correlation with physical drivers")
    for k in range(4):
        cors = {n: float(np.corrcoef(TP[:, k], v)[0, 1]) for n, v in tphys.items()
                if np.isfinite(v).all()}
        top = sorted(cors.items(), key=lambda x: -abs(x[1]))[:3]
        print(f"  PC{k+1}  {var2[k]*100:6.1f}%   " + ",  ".join(f"{n} {c:+.2f}" for n, c in top))
        ax = axes[1, k]
        o = np.argsort(hr)
        d = pd.DataFrame({"h": hr, "v": TP[:, k]}).groupby("h").mean()
        ax.plot(d.index, d.v, "-o", ms=3, color="C3")
        cors2 = {n: float(np.corrcoef(TP[:, k], v)[0, 1]) for n, v in tphys.items()}
        n1, c1 = max(cors2.items(), key=lambda x: abs(x[1]))
        ax.set_title(f"temporal PC{k+1} — {var2[k]*100:.1f}%\nbest match: {n1} (r={c1:+.2f})",
                     fontsize=9.5)
        ax.set_xlabel("hour, local"); ax.grid(alpha=.3)
    fig.suptitle("What the learned geographic prior represents: spatial modes (top) and "
                 "temporal modes (bottom)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{a.out}/eigen_modes.png", dpi=110)
    print(f"\nwrote {a.out}/eigen_modes.png")


if __name__ == "__main__":
    main()
