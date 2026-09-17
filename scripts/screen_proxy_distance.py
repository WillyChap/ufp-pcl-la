import numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
ufp=xr.open_dataset('HARMONIZED_MASTER_FILES/UFP_MASTER.nc')
t=ufp.time.values; lat=ufp.latitude.values; lon=ufp.longitude.values
y=np.log10(np.clip(ufp.UFP.values,1,None))
hod=ufp.time.dt.hour.values; mon=ufp.time.dt.month.values
names=[str(s) for s in ufp.site.values]
def des(a):
    out=np.full_like(a,np.nan,dtype=float); key=hod*100+mon
    for s in range(a.shape[0]):
        v=a[s]
        for k in np.unique(key):
            m=(key==k)&np.isfinite(v)
            if m.sum()>=5: out[s,m]=v[m]-v[m].mean()
    return out
def dedupe(ds):
    _,i=np.unique(ds.time.values,return_index=True)
    return ds.isel(time=np.sort(i)) if len(i)!=ds.sizes['time'] else ds
aq=dedupe(xr.open_dataset('data/real/aqs_42602.nc'))
alat=aq.latitude.values; alon=aq.longitude.values; A=aq.aqs_value.reindex(time=t).values
hr=dedupe(xr.open_dataset('data/real/hrrr_la.nc'))
blh=hr['hrrr_blh'].sel(latitude=xr.DataArray(lat,dims='site'),
        longitude=xr.DataArray(lon,dims='site'),method='nearest').reindex(time=t).transpose('site','time').values

ya=des(y); ba=des(blh)
print(f"{'UFP site':<14}{'dist to nearest':>16}{'r(NO2)':>9}{'r|BLH':>8}{'n':>8}")
print("-"*58)
z=np.full((len(lat),len(t)),np.nan); dists=[]
for s in range(len(lat)):
    d=np.hypot((alat-lat[s])*111,(alon-lon[s])*92); j=int(np.argmin(d)); dists.append(d[j])
    z[s]=A[j]
za=des(z)
for s in range(len(lat)):
    m=np.isfinite(ya[s])&np.isfinite(za[s])&np.isfinite(ba[s])
    if m.sum()<100: continue
    Y,Z,B=ya[s][m],za[s][m],ba[s][m]
    r=np.corrcoef(Y,Z)[0,1]
    # partial correlation of UFP with NO2 controlling for mixing depth
    ry=Y-np.polyval(np.polyfit(B,Y,1),B); rz=Z-np.polyval(np.polyfit(B,Z,1),B)
    rp=np.corrcoef(ry,rz)[0,1]
    print(f"{names[s][:13]:<14}{dists[s]:>13.1f} km{r:>9.3f}{rp:>8.3f}{m.sum():>8}")
m=np.isfinite(ya)&np.isfinite(za)&np.isfinite(ba)
Y,Z,B=ya[m],za[m],ba[m]
ry=Y-np.polyval(np.polyfit(B,Y,1),B); rz=Z-np.polyval(np.polyfit(B,Z,1),B)
print("-"*58)
print(f"{'POOLED':<14}{np.mean(dists):>13.1f} km{np.corrcoef(Y,Z)[0,1]:>9.3f}"
      f"{np.corrcoef(ry,rz)[0,1]:>8.3f}{m.sum():>8}")
print(f"\n  r|BLH = partial correlation controlling for HRRR mixing depth,")
print(f"  i.e. skill beyond the meteorology the model already uses.")
print(f"\n  AQS NO2 network: {len(alat)} sites vs {len(lat)} UFP monitors")
# how far can it reach?
print("\n  Does it transfer with distance? r using the k-th nearest AQS site:")
for k in [1,2,3,5,8]:
    zz=np.full((len(lat),len(t)),np.nan); dd=[]
    for s in range(len(lat)):
        d=np.hypot((alat-lat[s])*111,(alon-lon[s])*92); j=int(np.argsort(d)[k-1]); dd.append(d[j]); zz[s]=A[j]
    zza=des(zz); mm=np.isfinite(ya)&np.isfinite(zza)
    print(f"    k={k}  mean dist {np.mean(dd):5.1f} km   r = {np.corrcoef(ya[mm],zza[mm])[0,1]:+.3f}")
