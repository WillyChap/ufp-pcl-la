"""Physically-motivated derived predictors.

The observation encoder is given raw retrievals and reanalysis fields, but a handful of
ratios and geometric quantities carry mechanism that a network would otherwise have to
rediscover from limited data.  Each entry below exists because it separates a *process*,
not because it adds a column:

  fnr            HCHO / NO2, the formaldehyde-to-nitrogen-dioxide ratio.  A standard
                 satellite indicator of the ozone-production regime; low values mean
                 VOC-limited / high-NOx (fresh traffic plume, primary UFP), high values
                 mean NOx-limited (aged, photochemically active, secondary formation).
  solar_elev     Solar elevation angle.  Drives photochemistry directly and separates
                 the TEMPO observing window from the rest of the day.
  blh_dilution   1 / boundary-layer height.  Surface number concentration scales roughly
                 with the inverse of the mixing depth at fixed emissions, so this is the
                 meteorological term that explains the morning peak without any change
                 in emissions at all.
  wind_speed,
  wind_dir_sin,
  wind_dir_cos   Speed and a continuous encoding of direction from u/v components.
                 Direction matters in the LA basin: onshore flow ventilates, offshore
                 flow recirculates aged air back over the basin.
  rh_proxy       Dewpoint depression (t2m - d2m), a cheap stand-in for relative humidity,
                 which modulates condensational growth of freshly nucleated particles.
  temp_c         2 m temperature in Celsius, for readability of attribution plots.

Configure with `data.derived: [fnr, solar_elev, blh_dilution, ...]`.  Anything whose
inputs are missing from the file is skipped with a warning rather than failing the run.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data.netcdf import solar_elevation_deg


def _find(names: Sequence[str], *candidates: str) -> Optional[int]:
    low = [n.lower() for n in names]
    for c in candidates:
        for i, n in enumerate(low):
            if n == c:
                return i
    for c in candidates:                      # then substring, longest name last
        for i, n in enumerate(low):
            if c in n:
                return i
    return None


def _safe_ratio(a: np.ndarray, b: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return a / np.where(np.abs(b) < floor, np.nan, b)


def compute_derived(
    obs: np.ndarray,
    names: List[str],
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: np.ndarray,
    wanted: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Append derived columns to `obs` (n, F) or (n, T, F) -> (n, ..., F + k)."""
    if not wanted:
        return obs, names

    seq = obs.ndim == 3
    flat = obs.reshape(-1, obs.shape[-1]) if seq else obs
    reps = obs.shape[1] if seq else 1
    lon_r = np.repeat(lon, reps) if seq else lon
    lat_r = np.repeat(lat, reps) if seq else lat
    t_r = np.repeat(time_days, reps) if seq else time_days

    def col(*cands):
        i = _find(names, *cands)
        return None if i is None else flat[:, i].astype(np.float64)

    new_cols: List[np.ndarray] = []
    new_names: List[str] = []

    def add(name, arr):
        if arr is None:
            warnings.warn(f"derived feature '{name}' skipped: inputs not in the file")
            return
        new_cols.append(np.asarray(arr, dtype=np.float32))
        new_names.append(name)

    for w in wanted:
        if w == "fnr":
            hcho, no2 = col("hcho", "formaldehyde", "vertical_column_hcho"), col("no2", "nitrogen_dioxide")
            add("fnr_hcho_no2", None if hcho is None or no2 is None else _safe_ratio(hcho, no2))
        elif w == "solar_elev":
            add("solar_elev_deg", solar_elevation_deg(t_r, lon_r, lat_r))
        elif w == "blh_dilution":
            blh = col("blh", "boundary_layer_height", "pblh", "pbl")
            add("blh_dilution", None if blh is None else 1000.0 / np.clip(blh, 10.0, None))
        elif w in ("wind_speed", "wind_dir_sin", "wind_dir_cos"):
            u, v = col("u10", "u_wind", "eastward_wind"), col("v10", "v_wind", "northward_wind")
            if u is None or v is None:
                add(w, None)
                continue
            if w == "wind_speed":
                add("wind_speed", np.hypot(u, v))
            else:
                ang = np.arctan2(v, u)
                add(w, np.sin(ang) if w.endswith("sin") else np.cos(ang))
        elif w == "rh_proxy":
            t2, d2 = col("t2m", "temperature_2m"), col("d2m", "dewpoint")
            add("dewpoint_depression", None if t2 is None or d2 is None else t2 - d2)
        elif w == "temp_c":
            t2 = col("t2m", "temperature_2m")
            add("temp_c", None if t2 is None else np.where(t2 > 150.0, t2 - 273.15, t2))
        else:
            raise ValueError(f"unknown derived feature '{w}'")

    if not new_cols:
        return obs, names
    extra = np.stack(new_cols, axis=1)
    out = np.concatenate([flat, extra], axis=1)
    if seq:
        out = out.reshape(obs.shape[0], obs.shape[1], -1)
    return out.astype(np.float32), list(names) + new_names
