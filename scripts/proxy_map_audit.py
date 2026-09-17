"""Does the proxy term help the MAP and the physical modes, not just the 13 points?

Point metrics can improve for a degenerate reason: a model that collapses toward a flat
field is the best point predictor when it cannot order sites, but it is useless as a map.
This audits, per run:

  MAP     spatial dynamic range of the rendered surface, and whether its structure lines up
          with road density and coastal distance.
  LATENT  effective rank of the location embedding, and whether the leading spatial and
          temporal modes still identify with physical drivers (coastal-inland, solar).
"""
import argparse, os, sys, warnings, numpy as np, xarray as xr, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name:run_dir:config")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--n-times", type=int, default=40)
    a = ap.parse_args()

    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.latent import make_coord_grid, embed_coords, effective_rank
    from ufp_pcl.data.netcdf import solar_elevation_deg, to_days, calendar_features
    from ufp_pcl.mapping import predict_surface

    dev = resolve_device("auto")
    rows = []
    for spec in a.runs:
        name, run, cfgp = spec.split(":")
        cfg = load_config(cfgp)
        b = build_datasets(cfg, verbose=False)
        sd = torch.load(f"{run}/best.pt", map_location=dev, weights_only=False)
        m = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
        m.load_state_dict(sd["model"], strict=False); m.eval()
        cm = b.coord_meta

        T = to_days(xr.open_dataset(cfg.data.path).time.values)
        el = solar_elevation_deg(T, np.full(len(T), -118.2), np.full(len(T), 34.0))
        rng = np.random.default_rng(0)
        times = np.sort(rng.choice(T[el >= 5], a.n_times, replace=False))

        # ---------------- the rendered surface ----------------
        S, meta = predict_surface(cfg, b, m, dev, times, step=a.step)
        Z = np.nanmean(S, axis=0)                       # time-mean log10 field
        lons, lats = meta["lons"], meta["lats"]
        fin = np.isfinite(Z)
        sd_sp = float(np.nanstd(Z))
        rng98 = float(np.nanpercentile(Z, 98) - np.nanpercentile(Z, 2))

        # physical fields on the same grid
        r = xr.open_dataset("data/real/roads_la.nc")
        RA, RO = r.lat.values, r.lon.values
        rd = np.array([[float(r.road_w_0p3km.values[int(np.abs(RA-p).argmin()),
                                                    int(np.abs(RO-q).argmin())])
                        for q in lons] for p in lats]); r.close()
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        coast = LO - (-118.5)                            # crude inland distance proxy
        ok = fin & np.isfinite(rd)
        r_road = float(np.corrcoef(Z[ok], np.log1p(rd[ok]))[0, 1])
        r_coast = float(np.corrcoef(Z[fin], coast[fin])[0, 1])

        # ---------------- latent structure ----------------
        coords, gmeta = make_coord_grid(cm["lon_min"], cm["lon_max"], cm["lat_min"],
                                        cm["lat_max"], step=a.step, time_days=times)
        E = embed_coords(m, coords, dev)
        er = float(effective_rank(E))
        Tn, ny, nx = gmeta["shape"]
        Eg = E.reshape(Tn, ny*nx, -1).mean(0); Eg = Eg - Eg.mean(0)
        U, Sv, _ = np.linalg.svd(Eg, full_matrices=False)
        v1 = float((Sv**2/(Sv**2).sum())[0])
        sp1 = U[:, 0]*Sv[0]
        rc1 = float(np.corrcoef(sp1, coast.ravel())[0, 1])

        # temporal PC vs solar elevation, at the monitors
        u = xr.open_dataset(cfg.data.path); slat, slon = u.latitude.values, u.longitude.values; u.close()
        tt = np.sort(rng.choice(T[el >= 5], 900, replace=False))
        cl, ca = np.repeat(slon, len(tt)), np.repeat(slat, len(tt))
        ct = np.tile(tt, len(slon)); cal = calendar_features(ct)
        C = np.stack([cl, ca, cal["doy"], cal["year"], cal["hour"], cal["dow"], ct], 1).astype(np.float32)
        Et = embed_coords(m, C, dev).reshape(len(slon), len(tt), -1).mean(0)
        Et = Et - Et.mean(0)
        U2, S2, _ = np.linalg.svd(Et, full_matrices=False)
        solar = solar_elevation_deg(tt, np.full(len(tt), -118.2), np.full(len(tt), 34.0))
        rsol = max(abs(float(np.corrcoef(U2[:, k]*S2[k], solar)[0, 1])) for k in range(3))

        rows.append((name, sd_sp, rng98, r_road, r_coast, er, v1*100, rc1, rsol))
        print(f"  done {name}", flush=True)

    print("\nMAP STRUCTURE AND PHYSICAL INTERPRETABILITY")
    print(f"  {'run':<18}{'map sd':>8}{'p2-98':>8}{'r(road)':>9}{'r(coast)':>10}"
          f"{'eff rank':>10}{'PC1 %':>8}{'PC1~coast':>11}{'|r| solar':>10}")
    print("  " + "-"*92)
    for n, s, g, rr, rc, er, v1, rc1, rs in rows:
        print(f"  {n:<18}{s:>8.3f}{g:>8.3f}{rr:>+9.3f}{rc:>+10.3f}{er:>10.2f}"
              f"{v1:>8.1f}{rc1:>+11.3f}{rs:>10.3f}")


if __name__ == "__main__":
    main()
