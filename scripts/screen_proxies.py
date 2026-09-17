"""Screening test: does a candidate proxy carry state information about UFP that a
climatology does not already supply?

Both UFP and the candidate have per-site (hour-of-day x month) climatology removed.
Whatever correlation survives is information a climatology term cannot give the model.
"""
import numpy as np, xarray as xr, warnings
warnings.filterwarnings('ignore')

ufp = xr.open_dataset('HARMONIZED_MASTER_FILES/UFP_MASTER.nc')
t   = ufp.time.values
lat = ufp.latitude.values; lon = ufp.longitude.values
y   = np.log10(np.clip(ufp.UFP.values, 1, None))          # (site, time)
hod = ufp.time.dt.hour.values; mon = ufp.time.dt.month.values

def deseason(a, hod, mon):
    """Remove per-series (hour-of-day x month) climatology. a: (site, time)."""
    out = np.full_like(a, np.nan, dtype=float)
    key = hod * 100 + mon
    for s in range(a.shape[0]):
        v = a[s]
        for k in np.unique(key):
            m = (key == k) & np.isfinite(v)
            if m.sum() >= 5:
                out[s, m] = v[m] - v[m].mean()
    return out

ya = deseason(y, hod, mon)
print(f"UFP: {np.isfinite(y).sum()} obs; after climatology removal, "
      f"{np.nanstd(ya)/np.nanstd(y):.2f} of the sd remains\n")

def _dedupe(ds):
    _, i = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=np.sort(i)) if len(i) != ds.sizes['time'] else ds

def sample_grid(ds, var, transform=None):
    ds = _dedupe(ds)
    """Nearest-gridcell sample of a gridded field at the 7 UFP sites, aligned in time."""
    d = ds[var]
    v = d.sel(latitude=xr.DataArray(lat, dims='site'),
              longitude=xr.DataArray(lon, dims='site'), method='nearest')
    v = v.reindex(time=t)
    return v.transpose('site', 'time').values

def sample_points(ds, k=1):
    """Nearest AQS station(s) to each UFP site, aligned in time."""
    alat = ds.latitude.values; alon = ds.longitude.values
    ds = _dedupe(ds)
    a = ds.aqs_value.reindex(time=t).values                # (asite, time)
    out = np.full((len(lat), len(t)), np.nan)
    for s in range(len(lat)):
        dd = np.hypot((alat-lat[s])*111, (alon-lon[s])*92)
        idx = np.argsort(dd)[:k]
        out[s] = np.nanmean(a[idx], axis=0)
    return out

tempo = xr.open_dataset('HARMONIZED_MASTER_FILES/TEMPO_MASTER.nc')
goes  = xr.open_dataset('data/real/goes_aod.nc')
hrrr  = xr.open_dataset('data/real/hrrr_la.nc')

no2 = sample_grid(tempo,'VCD_NO2'); hcho = sample_grid(tempo,'VCD_HCHO')
blh = sample_grid(hrrr,'hrrr_blh')
cands = {
  'TEMPO NO2 column          (used)' : no2,
  'TEMPO HCHO column      (UNUSED!)' : hcho,
  'TEMPO HCHO/NO2 ratio   (UNUSED!)' : hcho/np.where(no2>0,no2,np.nan),
  'TEMPO NO2 / mixing depth  (new)' : no2/np.where(blh>50,blh,np.nan),
  'GOES AOD                  (used)' : sample_grid(goes,'aod' if 'aod' in goes else list(goes.data_vars)[0]),
  'AQS surface NO2, nearest  (new)' : sample_points(xr.open_dataset('data/real/aqs_42602.nc'),1),
  'AQS surface NO2, mean of 3(new)' : sample_points(xr.open_dataset('data/real/aqs_42602.nc'),3),
  'AQS PM2.5, nearest        (new)' : sample_points(xr.open_dataset('data/real/aqs_pm25.nc'),1),
  '-- reference: HRRR mixing depth' : blh,
}

print(f"{'candidate':<34}{'r':>8}{'r^2':>7}{'n':>8}   per-site r range")
print("-"*80)
rows=[]
for name, z in cands.items():
    za = deseason(np.asarray(z,float), hod, mon)
    m = np.isfinite(ya) & np.isfinite(za)
    if m.sum() < 200: print(f"{name:<34}{'--- too few overlapping obs ---':>30}"); continue
    r = np.corrcoef(ya[m], za[m])[0,1]
    per=[]
    for s in range(ya.shape[0]):
        ms = np.isfinite(ya[s]) & np.isfinite(za[s])
        if ms.sum() > 100: per.append(np.corrcoef(ya[s][ms], za[s][ms])[0,1])
    rng = f"{min(per):+.2f} .. {max(per):+.2f}" if per else "n/a"
    print(f"{name:<34}{r:>+8.3f}{r*r:>7.3f}{m.sum():>8}   {rng}")
    rows.append((name,r,m.sum()))
