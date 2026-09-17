"""Feature and target standardisation, persisted alongside the model."""
from __future__ import annotations

from typing import Optional

import numpy as np


class Standardizer:
    """Per-column z-scoring that ignores NaNs and never divides by zero."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        x = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1)
        with np.errstate(all="ignore"):
            self.mean = np.nanmean(x, axis=0)
            self.std = np.nanstd(x, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {"mean": np.asarray(self.mean), "std": np.asarray(self.std)}

    @classmethod
    def from_state(cls, blob: dict) -> "Standardizer":
        return cls(np.asarray(blob["mean"]), np.asarray(blob["std"]))


def apply_transform(x: np.ndarray, kind: str) -> np.ndarray:
    """Variance-stabilising transforms.  UFP counts are ~log-normal, so `log10` is the
    sensible default for the target and for concentration-like proxies."""
    if kind in (None, "none", ""):
        return x
    if kind == "log10":
        return np.log10(np.clip(x, 1e-6, None))
    if kind == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if kind == "sqrt":
        return np.sqrt(np.clip(x, 0.0, None))
    raise ValueError(f"unknown transform '{kind}'")


def invert_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind in (None, "none", ""):
        return x
    if kind == "log10":
        return np.power(10.0, x)
    if kind == "log1p":
        return np.expm1(x)
    if kind == "sqrt":
        return np.square(x)
    raise ValueError(f"unknown transform '{kind}'")
