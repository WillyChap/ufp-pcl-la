"""Feature attribution, and the primary-vs-secondary apportionment built on top of it.

Objective 2 of the project is process-level: decide, for a given place and hour, whether
UFP is dominated by *primary* emission (fresh combustion, co-emitted with NOx, peaking
with the weekday morning traffic rush under a shallow boundary layer) or by *secondary*
new-particle formation (photochemical, peaking in the afternoon, tracked by HCHO/O3 and
solar elevation, favoured in low-NOx conditions).

The model cannot be asked that question directly, but it can be asked which inputs its
prediction depends on, sample by sample.  This module provides:

  integrated_gradients   Per-sample, per-feature attribution with a completeness
                         guarantee (attributions sum to the prediction minus the
                         baseline prediction), so shares are meaningful.
  permutation_importance Model-agnostic sanity check on the same ranking; it measures
                         predictive reliance rather than local gradient, and the two
                         disagreeing is itself informative.
  pathway_scores         Collapses attributions onto a primary tracer set and a
                         secondary tracer set, giving a per-sample secondary fraction.
  conditional_profile    Averages any of the above over hour-of-day, weekday/weekend or
                         season -- which is where the diurnal double peak should appear
                         as a crossover from primary to secondary dominance.

A caution that belongs in the results, not just the code: attribution tells you what the
model used, which equals what drives UFP only to the extent that the model is right and
its inputs are not collinear.  NO2 and rush hour are collinear by construction, so a high
NO2 attribution at 07:00 is consistent with, but not proof of, primary dominance.  The
weekday/weekend contrast is the stronger evidence, because it moves the emissions without
moving the meteorology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

#: Tracers of directly emitted (primary) ultrafine particles.
PRIMARY_TRACERS = (
    "no2", "nox", "no_", "co", "black_carbon", "bc", "ec_", "traffic", "road",
    "diesel", "port", "truck",
)
#: Tracers of photochemical (secondary) new particle formation.
SECONDARY_TRACERS = (
    "hcho", "formaldehyde", "o3", "ozone", "so2", "sulf", "voc", "isop",
    "solar", "ssrd", "radiation", "fnr", "oc_", "soa",
)
#: Learned surface-context bands (AlphaEarth / Satellite Embedding and friends).  They
#: describe *where* you are, not which chemical pathway is active, so they get their own
#: bucket and are deliberately kept out of the primary/secondary index -- an embedding
#: band carrying a freeway signature would otherwise be scored as a primary tracer on the
#: strength of its name alone.
CONTEXT_PREFIXES = ("emb_", "alphaearth", "satellite_embedding", "ae_")

#: Meteorological modulators -- these change concentration without changing emission.
METEO_TERMS = (
    "blh", "pbl", "dilution", "wind", "u10", "v10", "t2m", "temp", "d2m", "dewpoint",
    "rh", "sp", "pressure", "precip",
)


def classify_features(names: Sequence[str]) -> Dict[str, List[int]]:
    """Bucket feature indices into primary / secondary / meteo / context / other."""
    out = {"primary": [], "secondary": [], "meteo": [], "context": [], "other": []}
    for i, n in enumerate(names):
        low = n.lower()
        if low.startswith(CONTEXT_PREFIXES) or "heterogeneity" in low:
            out["context"].append(i)
        elif any(k in low for k in SECONDARY_TRACERS):
            out["secondary"].append(i)
        elif any(k in low for k in PRIMARY_TRACERS):
            out["primary"].append(i)
        elif any(k in low for k in METEO_TERMS):
            out["meteo"].append(i)
        else:
            out["other"].append(i)
    return out


# --------------------------------------------------------------------------------------
@dataclass
class Attribution:
    values: np.ndarray          # (n, n_features) in standardised target units
    names: List[str]
    coords: np.ndarray          # (n, 7) COORD_COLS
    baseline_pred: float
    prediction: np.ndarray      # (n,)

    def mean_abs(self, group_prefixes: Sequence[str] = CONTEXT_PREFIXES) -> Dict[str, float]:
        """Mean |attribution| per feature, most important first.

        Bands sharing one of `group_prefixes` are collapsed into a single summed row.
        Without this, sixteen or sixty-four embedding components each carrying a modest
        attribution bury every named chemistry and meteorology term in the ranking,
        which makes the table useless for the question it is meant to answer.
        """
        m = np.nanmean(np.abs(self.values), axis=0)
        out: Dict[str, float] = {}
        grouped: Dict[str, List[float]] = {}
        for name, v in zip(self.names, m.tolist()):
            pref = next((p for p in group_prefixes if name.lower().startswith(p)), None)
            if pref is None:
                out[name] = v
            else:
                grouped.setdefault(pref, []).append(v)
        for pref, vals in grouped.items():
            out[f"{pref}* ({len(vals)} bands)"] = float(np.sum(vals))
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def signed_mean(self) -> Dict[str, float]:
        return dict(zip(self.names, np.nanmean(self.values, axis=0).tolist()))


def integrated_gradients(
    model,
    dataset,
    device,
    n_steps: int = 32,
    baseline: Optional[np.ndarray] = None,
    max_samples: int = 20000,
    batch_size: int = 512,
    names: Optional[List[str]] = None,
    seed: int = 0,
) -> Attribution:
    """Integrated gradients of the prediction w.r.t. the observation inputs.

    The baseline is the training mean (zero in standardised space), so an attribution
    reads as "how much of this prediction's departure from the average day is explained
    by this feature".  Location inputs are held at their true values: we are attributing
    the *observation* branch, with the geographic prior treated as context.
    """
    model.eval()
    n = len(dataset)
    idx = np.arange(n)
    if n > max_samples:
        idx = np.random.default_rng(seed).choice(n, max_samples, replace=False)
        idx.sort()

    obs_all = dataset.obs[idx]
    coords_all = dataset.coords[idx]
    feat_dim = obs_all.shape[-1]
    base = torch.zeros_like(obs_all[:1]) if baseline is None else torch.as_tensor(
        np.broadcast_to(np.asarray(baseline, np.float32), obs_all.shape[1:])[None]
    )

    attrs, preds = [], []
    for i in range(0, len(idx), batch_size):
        x = obs_all[i:i + batch_size].to(device)
        c = coords_all[i:i + batch_size].to(device)
        b = base.to(device).expand_as(x)
        delta = x - b
        total = torch.zeros_like(x)
        for k in range(n_steps):
            alpha = (k + 0.5) / n_steps
            xi = (b + alpha * delta).detach().requires_grad_(True)
            out = model(xi, c).sum()
            g, = torch.autograd.grad(out, xi)
            total += g
        attrs.append((delta * total / n_steps).detach().cpu().numpy())
        with torch.no_grad():
            preds.append(model(x, c).float().cpu().numpy().ravel())

    A = np.concatenate(attrs, 0)
    if A.ndim == 3:                    # (n, T, F): sum the sequence axis per feature
        A = A.sum(axis=1)
    with torch.no_grad():
        b_pred = float(model(base.to(device).expand(1, *obs_all.shape[1:]),
                             coords_all[:1].to(device)).item())
    names = names or [f"f{i}" for i in range(feat_dim)]
    return Attribution(A, list(names), coords_all.numpy(), b_pred, np.concatenate(preds))


def permutation_importance(
    model, dataset, device, y_scaler, n_repeats: int = 5, batch_size: int = 4096,
    names: Optional[List[str]] = None, seed: int = 0,
) -> Dict[str, float]:
    """Increase in RMSE when one feature is shuffled across samples.

    Reported in standardised target units.  Unlike integrated gradients this is a global,
    model-agnostic measure and is insensitive to gradient saturation.
    """
    from .evaluate import predict

    rng = np.random.default_rng(seed)
    base = predict(model, dataset, device, batch_size)
    truth = dataset.y.numpy().ravel()
    base_rmse = float(np.sqrt(((base - truth) ** 2).mean()))

    obs = dataset.obs.clone()
    feat_dim = obs.shape[-1]
    names = names or [f"f{i}" for i in range(feat_dim)]
    out = {}
    for f in range(feat_dim):
        drops = []
        for _ in range(n_repeats):
            saved = obs[..., f].clone()
            perm = torch.as_tensor(rng.permutation(len(obs)))
            dataset.obs[..., f] = saved[perm]
            p = predict(model, dataset, device, batch_size)
            drops.append(float(np.sqrt(((p - truth) ** 2).mean())) - base_rmse)
            dataset.obs[..., f] = saved
        out[names[f]] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------------------
def pathway_scores(attr: Attribution) -> Dict[str, np.ndarray]:
    """Per-sample attribution mass on each pathway, plus a secondary fraction.

    `secondary_fraction` in [0, 1] is |secondary| / (|primary| + |secondary|), computed
    from absolute attributions so that a strongly negative primary term still counts as
    primary influence.  It is a *relative* index, not an apportionment of concentration:
    read it as "which pathway is this prediction leaning on", and check it against the
    weekday/weekend and diurnal contrasts before making a physical claim.
    """
    groups = classify_features(attr.names)
    A = np.abs(attr.values)
    out = {}
    for g, cols in groups.items():
        out[g] = A[:, cols].sum(axis=1) if cols else np.zeros(len(A))
    # context bands are excluded from the index on purpose -- see CONTEXT_PREFIXES
    denom = out["primary"] + out["secondary"]
    out["secondary_fraction"] = np.where(denom > 0, out["secondary"] / denom, np.nan)
    out["total"] = A.sum(axis=1)
    return out


def conditional_profile(
    values: np.ndarray, coords: np.ndarray, by: str = "hour",
) -> Dict[str, np.ndarray]:
    """Mean +- SE of a per-sample quantity, grouped by a temporal regime.

    `by` is one of 'hour' (diurnal cycle), 'dow' (day of week), 'weekend'
    (weekday vs weekend), 'month' or 'season'.
    """
    v = np.asarray(values, dtype=float).ravel()
    hour, dow, doy = coords[:, 4], coords[:, 5], coords[:, 2]
    if by == "hour":
        key, labels = np.round(hour).astype(int) % 24, np.arange(24)
    elif by == "dow":
        key, labels = dow.astype(int), np.arange(7)
    elif by == "weekend":
        key, labels = (dow >= 5).astype(int), np.array([0, 1])
    elif by == "month":
        key = np.clip((doy / 30.5).astype(int), 0, 11)
        labels = np.arange(12)
    elif by == "season":
        key = ((np.clip((doy / 30.5).astype(int), 0, 11) + 1) % 12) // 3
        labels = np.arange(4)
    else:
        raise ValueError(f"unknown grouping '{by}'")

    mean = np.full(len(labels), np.nan)
    se = np.full(len(labels), np.nan)
    cnt = np.zeros(len(labels), dtype=int)
    for i, lab in enumerate(labels):
        m = (key == lab) & np.isfinite(v)
        cnt[i] = int(m.sum())
        if cnt[i]:
            mean[i] = v[m].mean()
            se[i] = v[m].std(ddof=1) / np.sqrt(cnt[i]) if cnt[i] > 1 else 0.0
    return {"labels": labels, "mean": mean, "se": se, "count": cnt}


def weekend_contrast(values: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
    """Weekday-minus-weekend difference with a Welch t statistic.

    The cleanest natural experiment available: traffic emissions drop on weekends while
    meteorology and photochemistry do not, so a quantity that is genuinely traffic-driven
    must show a weekday excess here.
    """
    v = np.asarray(values, float).ravel()
    wknd = coords[:, 5] >= 5
    a, b = v[~wknd & np.isfinite(v)], v[wknd & np.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return {"weekday_mean": float("nan"), "weekend_mean": float("nan"),
                "difference": float("nan"), "t": float("nan")}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return {"weekday_mean": float(a.mean()), "weekend_mean": float(b.mean()),
            "difference": float(a.mean() - b.mean()),
            "t": float((a.mean() - b.mean()) / np.sqrt(va + vb + 1e-30)),
            "n_weekday": int(len(a)), "n_weekend": int(len(b))}
