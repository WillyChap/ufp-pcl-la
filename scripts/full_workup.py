#!/usr/bin/env python3
"""Full interpretability workup for a trained UFP model.

Beyond the standard latent/attribution report: does the model reproduce the *diurnal*
cycle site by site, and which chemistry does it actually use, split by the pathways the
science question cares about (primary combustion vs secondary photochemical)?
"""
from __future__ import annotations

import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, torch, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Exact names, not substrings.  A substring rule put `wind_dir_cos` in the primary
# bucket because it contains "co", which silently inflated the traffic pathway by an
# attribution of 0.138 -- one of the ten largest in the model.
PRIMARY = {"no2","co","bcpi","bcpo","bcsmass","noy","benz","tolu","vcd_no2",
           "truck5_intensity","truck5_nearest","aadt_intensity",
           "road_dist_km","road_w_0p3km","road_w_1p0km","road_w_3p0km"}
SECONDARY = {"hcho","o3","so2","so4smass","ocsmass","pm25soa_rh35_gc","pm25su_rh35_gcc",
             "pm25oc_rh35_gcc","isop","pan","solar_elev_deg","fnr_hcho_no2","vcd_hcho",
             "ssrd","hrrr_ssrd"}
METEO = {"blh","u10","v10","t2m","d2m","sp","tp","lcc","hrrr_blh","hrrr_u10","hrrr_v10",
         "hrrr_t2m","hrrr_rh","hrrr_tcc","hrrr_sp","wind_speed","wind_dir_sin",
         "wind_dir_cos","dewpoint_depression","temp_c","blh_dilution"}


def bucket(n):
    k = n.lower()
    if k in PRIMARY: return "primary"
    if k in SECONDARY: return "secondary"
    if k in METEO: return "meteo"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/real/m_final")
    ap.add_argument("--config", default="configs/map_climo_mates.yaml")
    ap.add_argument("--out", default="outputs/real/maps")
    a = ap.parse_args()
    from ufp_pcl.config import load_config
    from ufp_pcl.data.dataset import build_datasets
    from ufp_pcl.models.fusion import build_model
    from ufp_pcl.train import resolve_device
    from ufp_pcl.evaluate import predict
    from ufp_pcl.attribution import integrated_gradients
    from ufp_pcl.data.netcdf import days_to_datetime64

    cfg = load_config(a.config); b = build_datasets(cfg, verbose=False)
    dev = resolve_device("auto")
    sd = torch.load(f"{a.run}/best.pt", map_location=dev, weights_only=False)
    m = build_model(cfg, b.n_obs_features, b.n_proxy, b.seq_len).to(dev)
    m.load_state_dict(sd["model"], strict=False); m.eval()

    # ---------- diurnal, per site, predicted vs observed ----------
    te = b.masks["test"]; ds = b.test
    p = predict(m, ds, dev)
    pt = b.y_scaler.inverse(p.reshape(-1,1)).ravel() + ds.climo
    ot = b.y_scaler.inverse(ds.y.numpy().reshape(-1,1)).ravel() + ds.climo
    t = pd.to_datetime(days_to_datetime64(b.table["time_days"][te]))
    hr = (t.hour - 8) % 24
    sid = b.table["site_id"][te]
    u = xr.open_dataset("HARMONIZED_MASTER_FILES/UFP_MASTER.nc")
    nm = [str(s).split("_")[0] for s in u.site.values]; u.close()
    sites = sorted(set(sid))
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    print("Diurnal reproduction, per site (test set, local hour)\n")
    print(f"  {'site':22s}{'r':>7s}{'amp obs':>10s}{'amp pred':>10s}{'peak obs':>10s}{'peak pred':>11s}")
    rows=[]
    for ax, s in zip(axes.ravel(), sites):
        msk = sid == s
        d = pd.DataFrame({"h":hr[msk], "o":ot[msk], "p":pt[msk]}).groupby("h").mean()
        ax.plot(d.index, 10**d.o, "k-o", ms=3, label="observed")
        ax.plot(d.index, 10**d.p, "r-s", ms=3, label="predicted")
        ax.set_title(nm[s], fontsize=10); ax.grid(alpha=.3)
        r = np.corrcoef(d.o, d.p)[0,1]
        rows.append(dict(site=nm[s], r=float(r), amp_obs=float(10**(d.o.max()-d.o.min())),
                         amp_pred=float(10**(d.p.max()-d.p.min())),
                         peak_obs=int(d.o.idxmax()), peak_pred=int(d.p.idxmax())))
        print(f"  {nm[s]:22s}{r:+7.3f}{10**(d.o.max()-d.o.min()):10.2f}x{10**(d.p.max()-d.p.min()):9.2f}x"
              f"{int(d.o.idxmax()):9d}h{int(d.p.idxmax()):10d}h")
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(sites):]: ax.axis("off")
    fig.suptitle("Diurnal cycle: observed vs predicted, by site (test set)", fontsize=13)
    fig.supxlabel("hour, local"); fig.supylabel("particles cm$^{-3}$")
    fig.tight_layout(); fig.savefig(f"{a.out}/diurnal_by_site.png", dpi=110)
    print(f"\n  mean diurnal r = {np.mean([x['r'] for x in rows]):+.3f}")
    print(f"  wrote {a.out}/diurnal_by_site.png")

    # ---------- attribution by hour, split by pathway ----------
    att = integrated_gradients(m, ds, dev, n_steps=32, max_samples=4000,
                               names=b.obs_names)
    A = np.abs(att.values)
    names = list(att.names)
    bk = np.array([bucket(n) for n in names])
    # the attribution keeps its own coordinate rows; take the local hour from those
    H = ((att.coords[:, 4].astype(int)) - 8) % 24
    print("\nAttribution by pathway and hour (share of |integrated gradient|)\n")
    print(f"  {'hour':>6s}{'primary':>10s}{'secondary':>11s}{'meteo':>9s}{'sec frac':>10s}")
    prof={}
    for h in range(5, 20):
        mm = H == h
        if mm.sum() < 20: continue
        tot = A[mm].sum()
        pr = A[mm][:, bk=="primary"].sum()/tot
        se = A[mm][:, bk=="secondary"].sum()/tot
        me = A[mm][:, bk=="meteo"].sum()/tot
        prof[h]=(pr,se,me)
        print(f"  {h:5d}h{pr:10.3f}{se:11.3f}{me:9.3f}{se/(pr+se+1e-9):10.3f}")
    if prof:
        hh=sorted(prof); fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(hh,[prof[h][0] for h in hh],"-o",label="primary (traffic/combustion)")
        ax.plot(hh,[prof[h][1] for h in hh],"-s",label="secondary (photochemical)")
        ax.plot(hh,[prof[h][2] for h in hh],"-^",label="meteorology")
        ax.set_xlabel("hour, local"); ax.set_ylabel("share of |attribution|")
        ax.set_title("Which pathway the model uses, by hour of day"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{a.out}/pathway_by_hour.png", dpi=110)
        print(f"  wrote {a.out}/pathway_by_hour.png")
    # ---------- top chemical predictors ----------
    mean_abs = A.mean(0)
    order = np.argsort(-mean_abs)
    print("\nTop 15 predictors by mean |IG|, with pathway:")
    for i in order[:15]:
        print(f"    {names[i]:24s}{mean_abs[i]:8.4f}   {bk[i]}")
    json.dump({"diurnal":rows,"pathway_by_hour":{str(k):list(map(float,v)) for k,v in prof.items()},
               "top_features":[{"name":names[i],"mean_abs_ig":float(mean_abs[i]),"bucket":str(bk[i])}
                               for i in order[:25]]},
              open(f"{a.run}/workup.json","w"), indent=1)
    print(f"\nwrote {a.run}/workup.json")


if __name__ == "__main__":
    main()
