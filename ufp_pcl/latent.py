"""Latent-space analysis of the learned location-time embedding.

The point of the proxy-consistency loss is that it *shapes* e_loc.  So the interesting
scientific claims are not only "RMSE went down" but statements about what the geographic
prior learned.  This module supplies the measurements that back such claims:

  variance_partition   How much of the embedding's variance is spatial vs seasonal vs
                       their interaction.  The paper's qualitative finding -- PCL makes
                       time dominant, label-only training makes space dominant and spiky
                       -- becomes a number here.
  spatial_roughness    Mean gradient magnitude of an embedding field.  Directly tests
  morans_i             "PCL embeddings are spatially smoother, without the star-shaped
                       artefacts radiating from monitor locations".
  distance_decay       Embedding similarity as a function of separation, i.e. the
                       correlation length of the learned prior.  Answers "over what
                       distance does this model consider two places interchangeable?",
                       which for UFP is the difference between a near-road gradient and
                       an urban background.
  linear_probe         Ridge R^2 of predicting a held-out field from e_loc alone.  This
                       is the load-bearing test for claims of the form "the location
                       encoder absorbed the traffic/NO2 signal".
  cka                  Representational similarity between two trained models, e.g. with
                       and without PCL, or across seeds.
  cluster_regions      k-means over embeddings -> a data-driven zonation of the domain
                       ("latent air-sheds") that can be compared to known basins.
  prior_ablation       Drop in test R^2 when e_loc is replaced by its domain mean --
                       how much predictive work the geographic prior is actually doing.

Every routine takes plain arrays so results can be compared across model variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data.netcdf import calendar_features

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- extraction
def make_coord_grid(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    step: float = 0.05, time_days: Optional[Sequence[float]] = None,
):
    """Regular lon/lat grid crossed with a set of times -> coordinate tensor + shape."""
    lons = np.arange(lon_min, lon_max + 1e-9, step)
    lats = np.arange(lat_min, lat_max + 1e-9, step)
    LON, LAT = np.meshgrid(lons, lats)                      # (ny, nx)
    times = np.asarray(time_days if time_days is not None else [0.0], dtype=float)
    n_pix = LON.size

    lon_f = np.tile(LON.ravel(), len(times))
    lat_f = np.tile(LAT.ravel(), len(times))
    t_f = np.repeat(times, n_pix)
    cal = calendar_features(t_f)
    coords = np.stack(
        [lon_f, lat_f, cal["doy"], cal["year"], cal["hour"], cal["dow"], t_f], 1
    ).astype(np.float32)
    return coords, {"lons": lons, "lats": lats, "times": times,
                    "shape": (len(times), len(lats), len(lons))}


@torch.no_grad()
def embed_coords(model, coords: np.ndarray, device, batch_size: int = 16384) -> np.ndarray:
    """e_loc for a big pile of coordinates -> (n, D)."""
    model.eval()
    out = []
    for i in range(0, len(coords), batch_size):
        c = torch.as_tensor(coords[i:i + batch_size], device=device)
        out.append(model.location_embedding(c).float().cpu().numpy())
    return np.concatenate(out, 0)


# --------------------------------------------------------------------------------- PCA
@dataclass
class PCAResult:
    components: np.ndarray        # (k, D)
    scores: np.ndarray            # (n, k)
    explained: np.ndarray         # (k,) fraction of total variance
    mean: np.ndarray              # (D,)

    def as_fields(self, meta) -> np.ndarray:
        """Reshape scores back to (n_times, ny, nx, k) for mapping."""
        T, ny, nx = meta["shape"]
        return self.scores.reshape(T, ny, nx, -1)


def pca(E: np.ndarray, k: int = 8) -> PCAResult:
    from sklearn.decomposition import PCA as _PCA

    k = int(min(k, E.shape[1], max(E.shape[0] - 1, 1)))
    p = _PCA(n_components=k)
    scores = p.fit_transform(E)
    return PCAResult(p.components_, scores, p.explained_variance_ratio_, p.mean_)


# ------------------------------------------------------------------ variance partition
def variance_partition(E: np.ndarray, shape: Tuple[int, int, int]) -> Dict[str, float]:
    """Two-way decomposition of embedding variance into space, time and interaction.

    E must be the flattened (time, pixel, D) grid produced by `make_coord_grid`.
    Returns fractions summing to 1.  A geographic prior that has genuinely absorbed
    seasonality shows a large `time` share; one that has memorised monitor locations
    shows a large `space` share with a small, noisy `interaction`.
    """
    T, ny, nx = shape
    X = E.reshape(T, ny * nx, E.shape[-1])
    grand = X.mean(axis=(0, 1), keepdims=True)
    space_mean = X.mean(axis=0, keepdims=True)       # (1, P, D)
    time_mean = X.mean(axis=1, keepdims=True)        # (T, 1, D)

    total = float(((X - grand) ** 2).sum())
    if total <= 0:
        return {"space": float("nan"), "time": float("nan"), "interaction": float("nan")}
    v_space = float(((space_mean - grand) ** 2).sum()) * T
    v_time = float(((time_mean - grand) ** 2).sum()) * (ny * nx)
    v_int = max(total - v_space - v_time, 0.0)
    return {"space": v_space / total, "time": v_time / total, "interaction": v_int / total,
            "total_variance": total / X.size}


# ---------------------------------------------------------------------- spatial texture
def spatial_roughness(field: np.ndarray) -> float:
    """Mean gradient magnitude of a 2-D map, normalised by its own std.

    Scale-free, so it can be compared between a PCL model and a label-only model whose
    embeddings have different magnitudes.  Higher = spikier.
    """
    f = np.asarray(field, dtype=float)
    gy, gx = np.gradient(f)
    s = np.nanstd(f)
    return float(np.nanmean(np.hypot(gy, gx)) / (s if s > 0 else 1.0))


def morans_i(field: np.ndarray) -> float:
    """Moran's I with rook adjacency on a regular grid (spatial autocorrelation).

    Near 1 = smooth, spatially coherent surface; near 0 = spatial noise.  Reported
    alongside `spatial_roughness` because the two can disagree when a field is smooth
    but has a strong trend.
    """
    f = np.asarray(field, dtype=float)
    m = np.isfinite(f)
    if m.sum() < 8:
        return float("nan")
    z = np.where(m, f - np.nanmean(f), 0.0)
    num, w = 0.0, 0.0
    for a, b in ((z[:-1, :], z[1:, :]), (z[:, :-1], z[:, 1:])):
        ma = np.isfinite(a) & np.isfinite(b)
        num += 2.0 * float((a * b)[ma].sum())
        w += 2.0 * float(ma.sum())
    den = float((z[m] ** 2).sum())
    if den <= 0 or w <= 0:
        return float("nan")
    return (m.sum() / w) * (num / den)


# ------------------------------------------------------------------------ distance decay
def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_decay(
    E: np.ndarray, lon: np.ndarray, lat: np.ndarray,
    n_pairs: int = 200_000, n_bins: int = 30, seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Cosine similarity of embeddings vs great-circle separation, binned.

    The distance at which similarity falls to 1/e of its near-zero value is a direct,
    interpretable "correlation length" of the learned geographic prior.
    """
    rng = np.random.default_rng(seed)
    n = len(E)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    A = E[i] / (np.linalg.norm(E[i], axis=1, keepdims=True) + 1e-9)
    B = E[j] / (np.linalg.norm(E[j], axis=1, keepdims=True) + 1e-9)
    sim = (A * B).sum(1)
    d = haversine_km(lon[i], lat[i], lon[j], lat[j])

    # Log-spaced bins: the interesting decay happens over the first few km, which linear
    # bins across a 150 km domain cannot resolve at all -- every model then reports the
    # same number, namely the width of bin zero.
    d_max = float(np.percentile(d, 99))
    d_min = max(float(np.percentile(d[d > 0], 0.5)), d_max / 1e4)
    edges = np.geomspace(d_min, d_max, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    mean = np.array([sim[idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(n_bins)])
    sd = np.array([sim[idx == b].std() if (idx == b).any() else np.nan
                   for b in range(n_bins)])

    # Correlation length: separation at which similarity has fallen 1 - 1/e of the way
    # from its shortest-range value to its asymptote.  Interpolated between bins in log
    # distance so the answer is not quantised to the bin grid.
    ok = np.isfinite(mean)
    corr_len = float("nan")
    if ok.sum() > 3:
        c, m = centres[ok], mean[ok]
        s0, s_inf = m[0], float(np.mean(m[-3:]))
        thresh = s_inf + (s0 - s_inf) / np.e
        below = np.where(m <= thresh)[0] if s0 > s_inf else np.where(m >= thresh)[0]
        if below.size and below[0] > 0:
            i = below[0]
            m0, m1 = m[i - 1], m[i]
            w = 0.0 if m1 == m0 else (thresh - m0) / (m1 - m0)
            corr_len = float(np.exp(np.log(c[i - 1]) + np.clip(w, 0, 1)
                                    * (np.log(c[i]) - np.log(c[i - 1]))))
        elif below.size:
            corr_len = float(c[0])
    return {"distance_km": centres, "similarity": mean, "sd": sd,
            "correlation_length_km": corr_len}


# ------------------------------------------------------------------------- linear probes
def linear_probe(
    E: np.ndarray, targets: np.ndarray, names: Optional[List[str]] = None,
    alpha: float = 1.0, n_splits: int = 5, seed: int = 0,
) -> Dict[str, float]:
    """Cross-validated ridge R^2 of predicting each column of `targets` from e_loc.

    This is the evidence for statements like "the PCL location encoder linearly encodes
    82% of the variance in the NO2 field but only 31% of boundary-layer height" -- i.e.
    which parts of the proxy stack actually made it into the geographic prior.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    Y = np.asarray(targets, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    names = names or [f"target_{i}" for i in range(Y.shape[1])]
    X = StandardScaler().fit_transform(np.asarray(E, dtype=float))

    out: Dict[str, float] = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c, name in enumerate(names):
        y = Y[:, c]
        ok = np.isfinite(y)
        if ok.sum() < n_splits * 5:
            out[name] = float("nan")
            continue
        Xi, yi = X[ok], y[ok]
        preds = np.zeros_like(yi)
        for tr, te in kf.split(Xi):
            model = Ridge(alpha=alpha).fit(Xi[tr], yi[tr])
            preds[te] = model.predict(Xi[te])
        ss_res = float(((preds - yi) ** 2).sum())
        ss_tot = float(((yi - yi.mean()) ** 2).sum())
        out[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


# ------------------------------------------------------------- representation similarity
def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two embeddings of the same points.

    1.0 = the two models learned the same representation up to rotation/scale.  Use it
    to show that PCL moves the latent space somewhere genuinely different from
    label-only training, rather than just rescaling it.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xty / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


def effective_rank(E: np.ndarray) -> float:
    """exp(entropy of the normalised eigenvalue spectrum): how many directions the
    embedding actually uses.  Collapse to a handful of directions is a warning that the
    proxy weight `lam` is too high."""
    X = np.asarray(E, float)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    p = p / p.sum() if p.sum() > 0 else p
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------- zonation
def cluster_regions(E: np.ndarray, k: int = 6, seed: int = 0) -> np.ndarray:
    """k-means over embeddings -> integer label per point ("latent air-sheds")."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E)


# ------------------------------------------------------------------------- prior ablation
@torch.no_grad()
def prior_ablation(model, dataset, device, y_scaler, y_transform,
                   batch_size: int = 4096) -> Dict[str, float]:
    """Test metrics when e_loc is replaced by its dataset mean.

    The gap against the intact model quantifies how much of the skill comes from the
    geographic prior rather than from the observation predictors -- the honest version
    of "location matters".
    """
    from .evaluate import regression_metrics
    from .data.scaling import invert_transform

    model.eval()
    if model.loc_encoder is None:
        return {}
    embs = []
    for i in range(0, len(dataset), batch_size):
        embs.append(model.loc_encoder(dataset.coords[i:i + batch_size].to(device)))
    mean_loc = torch.cat(embs, 0).mean(0, keepdim=True)

    preds = []
    for i in range(0, len(dataset), batch_size):
        obs = dataset.obs[i:i + batch_size].to(device)
        n = obs.shape[0]
        parts = []
        if model.obs_encoder is not None:
            parts.append(model.obs_encoder(obs))
        parts.append(mean_loc.expand(n, -1))
        preds.append(model.pred_head(torch.cat(parts, -1)).float().cpu().numpy())
    pred_s = np.concatenate(preds).ravel()
    pred_t = y_scaler.inverse(pred_s.reshape(-1, 1)).ravel()
    true_t = y_scaler.inverse(dataset.y.numpy().reshape(-1, 1)).ravel()
    out = regression_metrics(true_t, pred_t, prefix="ablated_")
    if y_transform not in (None, "none", ""):
        out.update(regression_metrics(invert_transform(true_t, y_transform),
                                      invert_transform(pred_t, y_transform),
                                      prefix="ablated_native_"))
    return out
