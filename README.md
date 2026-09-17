# Hourly ultrafine particles over the Los Angeles basin

Model, evaluation and reproduction code for hourly ultrafine particle number (total
particle count, diameter < 100 nm) across the South Coast Air Basin, trained on seven
SCAQMD AB-617 monitors, August 2023 – June 2025.

**[REPORT.md](REPORT.md)** is the write-up: what the model is, why each choice was made,
how it was evaluated, and what it does and does not support. Read that first.

## The model in one line

```
log10 N(place, time) = climatology(place) + anomaly(chemistry, meteorology, time)
```

A sign-constrained ridge fit on site means handles the spatial mean; a proxy-consistency
network handles everything that varies in time. The split follows the error: 4.5% of it is
between-site, 95.5% is hour-to-hour, and those two problems have very different amounts of
evidence behind them — seven numbers against ~29,000 hourly samples.

## Headline results

| | |
|---|---|
| Temporal hold-out R² (log₁₀ / native) | 0.699 / 0.630 |
| Diurnal reproduction, mean r across sites | +0.980 |
| Secondary fraction, solar noon → late afternoon | 0.585 → 0.372 |
| Leave-one-site-out R² | −0.195 ± 0.057 |
| Pattern correlation at held-out sites | 0.50 – 0.54 |
| Over-prediction at independent sites | 1.37× |

The model predicts *when* concentrations are high very well and *where* poorly. Relative
spatio-temporal structure transfers between stations; absolute concentration does not.
Section 4.3 of the report gives the independent validation that establishes this, and
Section 7 lists the limitations it implies.

## Install

```bash
conda create -n ufp python=3.11 && conda activate ufp
pip install -r requirements.txt
```

Fetching the land-surface embedding additionally needs `earthengine-api` and `rioxarray`,
and a Google Earth Engine account.

## Getting the data

The target and the satellite/reanalysis inputs are not redistributed here. Derived
predictors are rebuilt with:

```bash
python scripts/regrid_goes.py            # GOES AOD onto a regular grid (the proxy)
python scripts/fetch_hrrr.py             # 3 km hourly meteorology, from s3://hrrrzarr
python scripts/fetch_traffic.py          # OpenStreetMap + Caltrans traffic counts
python scripts/road_density.py           # class-weighted road density
python scripts/export_alphaearth_gee.py --project YOUR_GCP_PROJECT --scale 30 \
    --bbox -118.60 33.65 -117.00 34.30 --years 2023
python scripts/prepare_alphaearth.py 'ae_mean_2023_30m-*.tif' \
    --sd 'ae_sd_2023_30m-*.tif' --n-components 16 --out data/alphaearth_la_30m.nc
python scripts/prepare_mates.py          # historical sites for the climatology
```

Export resolution matters more than it looks. The land-surface embedding must be exported
at **30 m**: at 200 m the signal that distinguishes a near-road site from its neighbours
is averaged away entirely (r = +0.76 at 30 m against −0.05 at 200 m). Run
`scripts/check_alphaearth_separability.py` on any export before training on it.

## Training and evaluation

```bash
python -m ufp_pcl train configs/final.yaml --set train.out_dir=outputs/m_final
python -m ufp_pcl loso-cv configs/final_loso.yaml --variants pcl --out outputs/loso
python -m ufp_pcl analyze outputs/m_final
```

Interpretation and validation:

```bash
python scripts/full_workup.py       # diurnal reproduction, pathway apportionment by hour
python scripts/eigen_workup.py      # spatial and temporal modes, identified physically
python scripts/validate_sites.py --runs final:outputs/m_final:configs/final.yaml
python scripts/honest_surface.py --representativeness 1.42
python scripts/compare_surfaces.py  # model surfaces on a shared colour scale
```

`outputs/m_final/` holds the trained weights and the analysis output from the run the
report describes, so the figures reproduce without retraining.

## Two things to know before using the output

**Apply the representativeness correction to any predicted surface.** The seven monitors
are sited at populated and near-road locations and run **1.42×** above an independent
background sample, so a model fitted on them predicts that level everywhere.
`scripts/honest_surface.py` divides it out; anything else you write must do the same. It
is deliberately *not* applied to station metrics, where the model should reproduce the
stations it was fitted to.

**The background field is not validated.** Between background locations the model does not
discriminate — rank correlation against six independent sites is negative for every
predictor tested, and a bias-corrected constant beats every model there. Near-road
enhancement is the one spatial signal with support, and it rests on a single station.
72% of the domain has road density outside the range the network spans;
`honest_surface.py` marks it.

See `DATA_WISHLIST.md` for the data gaps behind both cautions, ranked by what closing them
would buy, with sources and two proposal concepts that would close most of the list.

## Layout

```
configs/final.yaml         the model
configs/final_loso.yaml    spatial cross-validation variant
ufp_pcl/                   package: data, model, training, evaluation, interpretation
scripts/                   data preparation, validation, figures
outputs/m_final/           trained weights, metrics, analysis
figures/                   report figures
REPORT.md                  the write-up
DATA_WISHLIST.md           what this model is missing, ranked — plus proposal ideas
```

Paths in the configs are relative to the directory you run from, and expect two data
roots alongside the ones above (neither is redistributed here; both are gitignored):

```
HARMONIZED_MASTER_FILES/   UFP_MASTER.nc and the satellite/reanalysis masters
                           (TEMPO, ERA5, MERRA, GEOS)
data/                      everything the scripts above build: goes_aod.nc,
                           hrrr_la.nc, traffic_la.nc, roads_la.nc,
                           alphaearth_la_30m.nc, mates_ufp.nc
```
