"""Regression metrics, reported in both the modelling space and the native units.

For UFP the modelling space is log10(particles cm^-3).  R^2 computed there answers
"how much of the log-variability do we explain", which is the fair comparison between
models; RMSE in native units answers "how wrong are we in particles per cm^3", which is
what an exposure assessment cares about.  Reporting only one of the two hides a lot, so
both are always returned.  MBE is included because a proxy-consistency term can inherit
a systematic bias from the proxy field, and that shows up here first.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .data.scaling import Standardizer, invert_transform


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged.  Local so scipy stays an optional dependency."""
    if a.size < 2:
        return float("nan")
    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        # average tied ranks
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {}
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        f"{prefix}r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        f"{prefix}rmse": float(np.sqrt((err ** 2).mean())),
        f"{prefix}mae": float(np.abs(err).mean()),
        f"{prefix}mbe": float(err.mean()),
        f"{prefix}pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan"),
        # Spearman as well as Pearson: UFP is heavy-tailed even after a log transform, so
        # a handful of extreme hours can carry the Pearson value.  Rank correlation says
        # whether the *ordering* of hours is right independently of how the magnitudes are
        # spread, and it is invariant to the affine rescaling and the climatology offset
        # that sit between the network output and a reported concentration.
        f"{prefix}spearman": _spearman(y_true, y_pred),
        f"{prefix}n": int(y_true.size),
    }


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(dataset), batch_size):
        obs = dataset.obs[i:i + batch_size].to(device)
        coords = dataset.coords[i:i + batch_size].to(device)
        outs.append(model(obs, coords).float().cpu().numpy())
    return np.concatenate(outs, axis=0).ravel() if outs else np.zeros(0)


def evaluate_split(
    model,
    dataset,
    device,
    y_scaler: Standardizer,
    y_transform: str,
    prefix: str = "",
) -> Dict[str, float]:
    """Metrics in standardised space, in transform space, and in native units.

    For the climo_anomaly variant the network predicts a departure from an explicit
    climatology, so the offset is added back here: every reported number then refers to
    the reconstructed field, and is directly comparable with the other variants.
    """
    pred_s = predict(model, dataset, device)
    true_s = dataset.y.numpy().ravel()

    pred_t = y_scaler.inverse(pred_s.reshape(-1, 1)).ravel()
    true_t = y_scaler.inverse(true_s.reshape(-1, 1)).ravel()

    # NOTE: data.representativeness is deliberately NOT applied here.  These metrics are
    # scored at the monitors, where the model should reproduce the monitors; the
    # correction describes the gap between that network and the basin, so it belongs on
    # the predicted surface only (see mapping.predict_surface).
    climo = getattr(dataset, "climo", None)
    if climo is not None and np.any(climo):
        pred_t = pred_t + climo
        true_t = true_t + climo

    out = regression_metrics(true_t, pred_t, prefix=f"{prefix}")           # transform space
    if y_transform not in (None, "none", ""):
        pred_n = invert_transform(pred_t, y_transform)
        true_n = invert_transform(true_t, y_transform)
        out.update(regression_metrics(true_n, pred_n, prefix=f"{prefix}native_"))
    return out


def climo_only_metrics(dataset, y_scaler: Standardizer, y_transform: str,
                      prefix: str = "") -> Dict[str, float]:
    """Score the climatology alone -- the floor the anomaly network has to beat.

    The prediction is the climatology plus the mean training anomaly, i.e. what you would
    get from the spatial fit and nothing else.  Reported next to the full model so a run
    answers its own most obvious question: is the network contributing anything beyond the
    two-term ridge, or is it just reproducing it with 2.6M parameters?
    """
    # NOTE: data.representativeness is deliberately NOT applied here.  These metrics are
    # scored at the monitors, where the model should reproduce the monitors; the
    # correction describes the gap between that network and the basin, so it belongs on
    # the predicted surface only (see mapping.predict_surface).
    climo = getattr(dataset, "climo", None)
    if climo is None or not np.any(climo):
        return {}
    true_t = y_scaler.inverse(dataset.y.numpy().reshape(-1, 1)).ravel() + climo
    pred_t = y_scaler.inverse(np.zeros((len(climo), 1))).ravel() + climo
    out = regression_metrics(true_t, pred_t, prefix=f"{prefix}climo_only_")
    if y_transform not in (None, "none", ""):
        out.update(regression_metrics(invert_transform(true_t, y_transform),
                                      invert_transform(pred_t, y_transform),
                                      prefix=f"{prefix}climo_only_native_"))
    return out


def format_metrics(metrics: Dict[str, float], keys=("r2", "rmse", "mae", "mbe")) -> str:
    parts = []
    for k in keys:
        for full in (k, f"native_{k}"):
            if full in metrics:
                parts.append(f"{full}={metrics[full]:.4g}")
    return "  ".join(parts)
