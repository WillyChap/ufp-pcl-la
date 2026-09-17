"""Reading heterogeneous NetCDF stacks.

The pipeline assumes one `.nc` file that mixes

  * a sparse in-situ target (UFP), either as a point table or as a mostly-NaN grid,
  * gridded predictor variables that may live on *several different* lat/lon grids
    (e.g. 1 km land-use, 4 km AOD, 12 km reanalysis), and
  * gridded proxy variables with (near-)complete coverage.

Variables are grouped by the spatial grid they live on; each group becomes a
`FieldStack` that can be sampled at arbitrary (lon, lat, time) with bilinear or
nearest-neighbour interpolation.  That is what lets predictors of different
resolutions be fused without pre-regridding anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

LAT_PAT = re.compile(r"^(lat|latitude|y|nav_lat|lat_[\w]+|[\w]+_lat)$", re.I)
LON_PAT = re.compile(r"^(lon|long|longitude|x|nav_lon|lon_[\w]+|[\w]+_lon)$", re.I)
TIME_PAT = re.compile(r"^(time|date|datetime|t|day|valid_time)$", re.I)


def _is_lat(name: str) -> bool:
    return bool(LAT_PAT.match(name))


def _is_lon(name: str) -> bool:
    return bool(LON_PAT.match(name))


def _is_time(name: str) -> bool:
    return bool(TIME_PAT.match(name))


def to_days(values: np.ndarray) -> np.ndarray:
    """Convert a time coordinate to float days since 1970-01-01 (monotone, uniform)."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype("int64") / 86_400e9
    if arr.dtype == object:  # cftime objects
        import cftime

        ref = cftime.DatetimeGregorian(1970, 1, 1)
        return np.array([(v - ref).total_seconds() / 86400.0 for v in arr], dtype=float)
    return arr.astype(float)


def days_to_datetime64(days: np.ndarray) -> np.ndarray:
    return (np.asarray(days) * 86_400e9).astype("int64").astype("datetime64[ns]")


#: Column order of the coordinate tensor handed to the location-time encoder.
#: Columns 0..5 are consumed by the encoder; `time_days` is carried along so the proxy
#: sampler and the analysis tools can recover the absolute time of any sample.
COORD_COLS = ["lon", "lat", "doy", "year", "hour", "dow", "time_days"]


def calendar_features(days: np.ndarray) -> Dict[str, np.ndarray]:
    """Day-of-year (1..366), year, hour-of-day and day-of-week (0=Mon) from float days.

    Day-of-week is not decoration: the weekday/weekend contrast in the morning UFP peak
    is the cleanest observational separator between traffic-primary and photochemical-
    secondary particle formation, so the encoder needs access to it.
    """
    d = np.asarray(days, dtype=float)
    dt = days_to_datetime64(d)
    years = dt.astype("datetime64[Y]")
    doy = (dt.astype("datetime64[D]") - years.astype("datetime64[D]")).astype(int) + 1
    year = years.astype(int) + 1970
    hour = (d % 1.0) * 24.0
    dow = (np.floor(d).astype(np.int64) + 3) % 7        # 1970-01-01 was a Thursday
    return {"doy": doy.astype(np.float32), "year": year.astype(np.float32),
            "hour": hour.astype(np.float32), "dow": dow.astype(np.float32)}


def solar_elevation_deg(days: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Solar elevation angle, used for daylight masking and as a derived predictor.

    Standard low-precision NOAA formulation; accurate to a few tenths of a degree, which
    is far beyond what an hourly air-quality model needs.
    """
    d = np.asarray(days, dtype=float)
    cal = calendar_features(d)
    doy, hour_utc = cal["doy"].astype(float), cal["hour"].astype(float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    tst = (hour_utc * 60.0 + eqtime + 4.0 * np.asarray(lon, float)) % 1440.0
    ha = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(np.asarray(lat, float))
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cosz, -1.0, 1.0)))


# --------------------------------------------------------------------------------------
@dataclass
class FieldStack:
    """A set of variables sharing one lat/lon grid (and optionally one time axis)."""

    name: str
    lat: np.ndarray                  # (ny,)  ascending
    lon: np.ndarray                  # (nx,)  ascending
    time_days: Optional[np.ndarray]  # (nt,)  ascending float days, or None for static
    var_names: List[str]
    data: np.ndarray                 # (nt, nv, ny, nx) or (nv, ny, nx) when static

    # ---------------------------------------------------------------- geometry helpers
    @property
    def is_static(self) -> bool:
        return self.time_days is None

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(lon_min, lon_max, lat_min, lat_max)."""
        return float(self.lon[0]), float(self.lon[-1]), float(self.lat[0]), float(self.lat[-1])

    def _axis_weights(self, axis: np.ndarray, q: np.ndarray):
        i1 = np.clip(np.searchsorted(axis, q, side="left"), 1, len(axis) - 1)
        i0 = i1 - 1
        span = axis[i1] - axis[i0]
        w = np.where(span > 0, (q - axis[i0]) / np.where(span > 0, span, 1.0), 0.0)
        return i0, i1, np.clip(w, 0.0, 1.0)

    def _time_index(self, time_days: Optional[np.ndarray], n: int) -> np.ndarray:
        if self.is_static:
            return np.zeros(n, dtype=int)
        if time_days is None:
            raise ValueError(f"FieldStack '{self.name}' is time-varying; time is required")
        idx = np.searchsorted(self.time_days, time_days, side="left")
        idx = np.clip(idx, 0, len(self.time_days) - 1)
        left = np.clip(idx - 1, 0, len(self.time_days) - 1)
        pick_left = np.abs(self.time_days[left] - time_days) <= np.abs(
            self.time_days[idx] - time_days
        )
        return np.where(pick_left, left, idx)

    # ------------------------------------------------------------------------- sampling
    def sample(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        time_days: Optional[np.ndarray] = None,
        interp: str = "bilinear",
    ) -> np.ndarray:
        """Sample every variable in the stack at n scattered points -> (n, n_vars)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        n = lon.size
        ti = self._time_index(None if time_days is None else np.asarray(time_days, float), n)
        cube = self.data if self.is_static else self.data[ti]  # (n, nv, ny, nx) gathered

        if interp == "nearest":
            iy = np.clip(np.abs(self.lat[None, :] - lat[:, None]).argmin(1), 0, len(self.lat) - 1)
            ix = np.clip(np.abs(self.lon[None, :] - lon[:, None]).argmin(1), 0, len(self.lon) - 1)
            if self.is_static:
                return cube[:, iy, ix].T.astype(np.float32)
            return cube[np.arange(n), :, iy, ix].astype(np.float32)

        y0, y1, wy = self._axis_weights(self.lat, lat)
        x0, x1, wx = self._axis_weights(self.lon, lon)
        if self.is_static:
            g = lambda a, b: cube[:, a, b].T                      # -> (n, nv)
        else:
            ar = np.arange(n)
            g = lambda a, b: cube[ar, :, a, b]                    # -> (n, nv)
        c00, c01, c10, c11 = g(y0, x0), g(y0, x1), g(y1, x0), g(y1, x1)
        wy, wx = wy[:, None], wx[:, None]
        out = (
            c00 * (1 - wy) * (1 - wx)
            + c01 * (1 - wy) * wx
            + c10 * wy * (1 - wx)
            + c11 * wy * wx
        )
        return out.astype(np.float32)

    def valid_mask_2d(self) -> np.ndarray:
        """(ny, nx) True where every variable in the stack is finite.

        For time-varying stacks a cell counts as valid if it is finite at *any* time
        step, so that proxy sampling is not thrown off by a single missing scene.
        """
        if self.is_static:                                   # (nv, ny, nx)
            return np.isfinite(self.data).all(axis=0)
        return np.isfinite(self.data).all(axis=1).any(axis=0)  # (nt, nv, ny, nx)


# --------------------------------------------------------------------------------------
def guess_role(name: str, attrs: dict) -> str:
    """Heuristic classification used only to pre-fill a starter config."""
    n = name.lower()
    if n.startswith("truth_") or "ground truth" in str(attrs.get("comment", "")).lower():
        return "truth (exclude)"
    if any(k in n for k in ("ufp", "pnc", "particle_number", "cpc", "n_tot")):
        return "target?"
    if any(k in n for k in ("aod", "aerosol_optical", "reanalysis", "cmaq", "_proxy",
                            "pm25", "pm2_5")):
        return "proxy?"
    return "obs?"


def describe(path: str) -> str:
    """Human-readable inventory of an .nc file: dims, grids, coverage, roles."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    lines = [f"file: {path}", f"dims: {dict(ds.sizes)}", ""]
    lines.append(f"{'variable':<28}{'dims':<34}{'%finite':>9}  {'range':<26}role")
    lines.append("-" * 118)
    for name, da in ds.data_vars.items():
        vals = da.values
        finite = np.isfinite(vals) if np.issubdtype(vals.dtype, np.number) else np.ones(1, bool)
        pct = 100.0 * finite.mean()
        if np.issubdtype(vals.dtype, np.number) and finite.any():
            rng = f"[{np.nanmin(vals):.3g}, {np.nanmax(vals):.3g}]"
        else:
            rng = "-"
        lines.append(
            f"{name:<28}{str(tuple(da.dims)):<34}{pct:>8.1f}%  {rng:<26}"
            f"{guess_role(name, da.attrs)}"
        )
    lines.append("")
    lines.append("coordinates:")
    for name, da in ds.coords.items():
        v = da.values
        if np.issubdtype(v.dtype, np.number) and v.ndim == 1 and v.size > 1:
            step = float(np.median(np.diff(v)))
            lines.append(f"  {name:<24} n={v.size:<7} [{v.min():.4g}, {v.max():.4g}] step~{step:.4g}")
        else:
            lines.append(f"  {name:<24} n={np.asarray(v).size:<7} {str(v.dtype)}")
    ds.close()
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
class NetCDFSource:
    """Opens the file once and exposes grouped `FieldStack`s plus the point target."""

    def __init__(self, path: str, coords, target, engine: Optional[str] = None):
        self.path = path
        self.coords = coords
        self.target = target
        self.ds = xr.open_dataset(path, engine=engine, decode_timedelta=False)
        self._stacks: Dict[Tuple[str, str, Optional[str]], FieldStack] = {}

    def close(self):
        self.ds.close()

    # ----------------------------------------------------------------- grid discovery
    def _spatial_dims(self, da) -> Optional[Tuple[str, str, Optional[str]]]:
        lat = lon = tim = None
        for d in da.dims:
            if _is_lat(d):
                lat = d
            elif _is_lon(d):
                lon = d
            elif _is_time(d):
                tim = d
        if lat is None or lon is None:
            return None
        return (lat, lon, tim)

    def gridded_vars(self) -> Dict[Tuple[str, str, Optional[str]], List[str]]:
        groups: Dict[Tuple[str, str, Optional[str]], List[str]] = {}
        for name, da in self.ds.data_vars.items():
            key = self._spatial_dims(da)
            if key is None:
                continue
            groups.setdefault(key, []).append(name)
        return groups

    def point_vars(self) -> List[str]:
        """Variables that share the target's sample dimension (already point-wise)."""
        tgt = self.ds[self.target.var]
        if self.target.mode != "points":
            return []
        sample_dims = tuple(d for d in tgt.dims if not (_is_lat(d) or _is_lon(d)))
        return [
            n for n, da in self.ds.data_vars.items()
            if n != self.target.var and tuple(da.dims) == sample_dims
        ]

    def build_stack(self, var_names: Sequence[str]) -> List[FieldStack]:
        """Group the requested variables by grid and materialise them as FieldStacks."""
        wanted = set(var_names)
        stacks: List[FieldStack] = []
        for key, names in self.gridded_vars().items():
            sel = [n for n in names if n in wanted]
            if not sel:
                continue
            latd, lond, timd = key
            lat = np.asarray(self.ds[latd].values, dtype=float)
            lon = np.asarray(self.ds[lond].values, dtype=float)
            flip_lat, flip_lon = lat[0] > lat[-1], lon[0] > lon[-1]
            if flip_lat:
                lat = lat[::-1]
            if flip_lon:
                lon = lon[::-1]
            tdays = to_days(self.ds[timd].values) if timd else None

            arrs = []
            for n in sel:
                da = self.ds[n]
                # Chemical transport output usually keeps a length-1 vertical dimension
                # even when only the surface level was extracted (GEOS-CF writes lev=72).
                # Drop any such degenerate extra axis so the field is a plain 2-D grid.
                extra = [d for d in da.dims if d not in (latd, lond, timd)]
                if extra:
                    squeeze = [d for d in extra if da.sizes[d] == 1]
                    if len(squeeze) != len(extra):
                        kept = [d for d in extra if d not in squeeze]
                        raise ValueError(
                            f"variable {n!r} has non-degenerate extra dimension(s) "
                            f"{kept}; select a single level before ingesting"
                        )
                    da = da.squeeze(squeeze, drop=True)
                order = ([timd] if timd else []) + [latd, lond]
                a = da.transpose(*order).values.astype(np.float32)
                if flip_lat:
                    a = a[..., ::-1, :]
                if flip_lon:
                    a = a[..., ::-1]
                arrs.append(a)
            if timd:
                cube = np.stack(arrs, axis=1)      # (nt, nv, ny, nx)
            else:
                cube = np.stack(arrs, axis=0)      # (nv, ny, nx)
            stacks.append(
                FieldStack(
                    name=f"{latd}x{lond}" + (f"@{timd}" if timd else ""),
                    lat=lat, lon=lon, time_days=tdays,
                    var_names=list(sel), data=np.ascontiguousarray(cube),
                )
            )
        missing = wanted - {n for s in stacks for n in s.var_names}
        if missing:
            raise KeyError(f"variables not found on any lat/lon grid: {sorted(missing)}")
        return stacks

    # --------------------------------------------------------------------- the target
    def target_table(self) -> Dict[str, np.ndarray]:
        """Flatten the target into columns: lon, lat, time_days, y, site_id."""
        tvar = self.target.var
        da = self.ds[tvar]
        cn = self.coords

        if self.target.mode == "points":
            lon = np.asarray(self.ds[cn.lon].values, dtype=float)
            lat = np.asarray(self.ds[cn.lat].values, dtype=float)
            tvals = to_days(self.ds[cn.time].values) if (cn.time and cn.time in self.ds) \
                else None
            y = np.asarray(da.values, dtype=float)

            if y.ndim == 1:
                # flat observation table: one row per measurement
                lon, lat, y = lon.ravel(), lat.ravel(), y.ravel()
                if tvals is None:
                    t = np.zeros_like(y)
                elif tvals.size == y.size:
                    t = tvals.ravel()
                else:
                    raise ValueError(
                        f"time has {tvals.size} entries but the target has {y.size}; "
                        "for a (site, time) layout give the target both dimensions"
                    )
            elif y.ndim == 2:
                # (site, time) or (time, site) panel
                site_dim = self.ds[cn.lon].dims[0]
                if da.dims[0] == site_dim:
                    n_site, n_time = y.shape
                    lon = np.repeat(lon.ravel(), n_time)
                    lat = np.repeat(lat.ravel(), n_time)
                    t = np.tile(tvals, n_site) if tvals is not None else np.zeros(y.size)
                else:
                    n_time, n_site = y.shape
                    lon = np.tile(lon.ravel(), n_time)
                    lat = np.tile(lat.ravel(), n_time)
                    t = np.repeat(tvals, n_site) if tvals is not None else np.zeros(y.size)
                y = y.ravel()
            else:
                raise ValueError(
                    f"point target '{tvar}' has {y.ndim} dims; expected 1 (flat table) "
                    "or 2 (site x time panel)"
                )
        else:  # sparse grid: keep the finite cells
            key = self._spatial_dims(da)
            if key is None:
                raise ValueError(f"target '{tvar}' is not on a lat/lon grid")
            latd, lond, timd = key
            latv = np.asarray(self.ds[latd].values, float)
            lonv = np.asarray(self.ds[lond].values, float)
            order = ([timd] if timd else []) + [latd, lond]
            arr = da.transpose(*order).values.astype(float)
            if timd:
                tv = to_days(self.ds[timd].values)
                T, Y, X = arr.shape
                ti, yi, xi = np.nonzero(np.isfinite(arr))
                y, lon, lat, t = arr[ti, yi, xi], lonv[xi], latv[yi], tv[ti]
            else:
                yi, xi = np.nonzero(np.isfinite(arr))
                y, lon, lat = arr[yi, xi], lonv[xi], latv[yi]
                t = np.zeros_like(y)

        keep = np.isfinite(y) & np.isfinite(lon) & np.isfinite(lat)
        lon, lat, t, y = lon[keep], lat[keep], t[keep], y[keep]

        if self.target.site_var and self.target.site_var in self.ds:
            sv = np.asarray(self.ds[self.target.site_var].values).ravel()
            site = sv if sv.size == keep.size else np.repeat(sv, keep.size // max(sv.size, 1))
            site = site[: keep.size][keep]
            _, site_id = np.unique(site, return_inverse=True)
        else:  # derive sites by rounding coordinates to ~10 m
            keys = np.stack([np.round(lon, 4), np.round(lat, 4)], 1)
            _, site_id = np.unique(keys, axis=0, return_inverse=True)

        return {"lon": lon.astype(np.float64), "lat": lat.astype(np.float64),
                "time_days": t.astype(np.float64),
                "y": y.astype(np.float64) * float(self.target.scale),
                "site_id": site_id.astype(np.int64)}


def load_embedding_cube(path: str, variables=None, prefix: Optional[str] = None,
                        lat_name: Optional[str] = None, lon_name: Optional[str] = None):
    """Read a rasterised embedding field -> (lat, lon, cube of shape (C, ny, nx)).

    Accepts either one variable with a band dimension, or many single-band variables
    selected by an explicit list or a name prefix (e.g. ``prefix="emb_"``).
    """
    ds = xr.open_dataset(path, decode_timedelta=False)
    latd = lat_name or next(d for d in ds.dims if _is_lat(d))
    lond = lon_name or next(d for d in ds.dims if _is_lon(d))
    lat = np.asarray(ds[latd].values, dtype=float)
    lon = np.asarray(ds[lond].values, dtype=float)

    if variables:
        names = list(variables)
    elif prefix:
        names = sorted(n for n in ds.data_vars if str(n).startswith(prefix))
    else:
        names = [n for n, da in ds.data_vars.items()
                 if latd in da.dims and lond in da.dims]
    if not names:
        ds.close()
        raise KeyError(f"no embedding variables found in {path}")

    arrs = []
    for n in names:
        da = ds[n]
        band = [d for d in da.dims if d not in (latd, lond)]
        if band:
            a = da.transpose(*band, latd, lond).values
            arrs.append(a.reshape(-1, len(lat), len(lon)))
        else:
            arrs.append(da.transpose(latd, lond).values[None])
    cube = np.concatenate(arrs, axis=0).astype(np.float32)
    ds.close()
    return lat, lon, cube
