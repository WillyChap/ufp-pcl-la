"""Train / validation / test splits.

Several protocols, each answering a different question.  For a 7-site hourly network in
one air basin the relevant ones are `temporal`, `loso` and `combined`; the checkerboard
protocol needs a network dense enough to tile.

  * `uar`          -- 50/50 uniform-at-random split over *monitoring sites*.  Test sites
                      may sit next to training sites, so this measures spatial
                      *interpolation*: "can the model fill in between my monitors?"
  * `checkerboard` -- the systematic protocol of Rolf et al.: tile the domain into
                      squares of side `delta` degrees and train on alternating squares.
                      This measures spatial *extrapolation*: "does this model transfer to
                      a neighbourhood where I have never measured?"  Sweeping `delta`
                      traces how far the geographic prior actually carries.

Sample-level random splitting leaks a site's own history into its test rows and will
flatter any location encoder, so it is offered only for debugging.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def checkerboard_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    delta: float,
    offset_index: int = 0,
    swap: bool = False,
) -> np.ndarray:
    """True where a point falls in a *training* square of the checkerboard.

    `offset_index` in 0..3 shifts the grid origin by (0,0), (d/2,0), (0,d/2), (d/2,d/2),
    which together with `swap` gives the 8 partitions used for error bars in the paper.
    """
    ox = (delta / 2.0) if offset_index in (1, 3) else 0.0
    oy = (delta / 2.0) if offset_index in (2, 3) else 0.0
    ix = np.floor((np.asarray(lon) + ox) / delta).astype(np.int64)
    iy = np.floor((np.asarray(lat) + oy) / delta).astype(np.int64)
    train = ((ix + iy) % 2 == 0)
    return ~train if swap else train


def checkerboard_partitions(delta: float) -> List[Tuple[int, bool]]:
    """The 8 (offset_index, swap) partitions used to report mean +- SE."""
    return [(o, s) for o in range(4) for s in (False, True)]


def temporal_mask(time_days: np.ndarray, test_frac: float) -> np.ndarray:
    """True for the trailing `test_frac` of the record (test), False earlier (train).

    Splitting on time rather than at random is mandatory for an hourly series: adjacent
    hours are almost the same sample, so a random split reports skill the model does not
    have out of sample.
    """
    t = np.asarray(time_days, dtype=float)
    cutoff = np.quantile(t, 1.0 - test_frac)
    return t > cutoff


def blocked_time_mask(
    time_days: np.ndarray, test_frac: float, val_frac: float,
    block_days: float = 7.0, seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Assign whole contiguous blocks of time to train / val / test.

    A single contiguous cut is the honest test of forecasting forward, but it confounds
    the question with season: over a 22-month record a trailing 20% test set is one
    season, the validation window before it is another, and early stopping then tunes to
    whichever season it happened to land on.  UFP here swings from a December median of
    11,790 to an August one of 18,850, so that confound is larger than most effects worth
    measuring.

    Blocks are long enough (a week by default) that adjacent-hour leakage across the
    boundary is negligible -- the objection that rules out a random split -- while every
    split still sees every season.  Use this to compare configurations; use `temporal`
    to quote a forecasting number.
    """
    t = np.asarray(time_days, float)
    block = np.floor((t - t.min()) / float(block_days)).astype(np.int64)
    uniq = np.unique(block)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = int(round(test_frac * len(uniq)))
    n_val = int(round(val_frac * len(uniq)))
    role = np.empty(len(uniq), dtype="<U5")
    role[perm[:n_test]] = "test"
    role[perm[n_test:n_test + n_val]] = "val"
    role[perm[n_test + n_val:]] = "train"
    lookup = dict(zip(uniq.tolist(), role.tolist()))
    tag = np.array([lookup[b] for b in block])
    return {"train": tag == "train", "val": tag == "val", "test": tag == "test"}


def make_split(
    lon: np.ndarray,
    lat: np.ndarray,
    site_id: np.ndarray,
    kind: str = "temporal",
    time_days: Optional[np.ndarray] = None,
    holdout_site: Optional[int] = None,
    test_frac: float = 0.5,
    val_frac: float = 0.15,
    seed: int = 0,
    delta: float = 0.25,
    offset_index: int = 0,
    swap: bool = False,
    block_days: float = 7.0,
) -> Dict[str, np.ndarray]:
    """Return boolean masks {'train','val','test'} over the sample table.

    Validation is always carved out by *site* from the training portion, so the early
    stopping signal is never contaminated by a test-adjacent monitor.
    """
    rng = np.random.default_rng(seed)
    n = len(lon)

    if kind == "blocked_time":
        if time_days is None:
            raise ValueError("split kind 'blocked_time' needs the time axis")
        return blocked_time_mask(time_days, test_frac, val_frac,
                                 block_days=block_days, seed=seed)

    if kind in ("temporal", "loso", "combined"):
        if kind in ("temporal", "combined") and time_days is None:
            raise ValueError(f"split kind '{kind}' needs the time axis")

        site_test = np.zeros(n, bool)
        if kind in ("loso", "combined"):
            sites, counts = np.unique(site_id, return_counts=True)
            held = sites[counts.argmax()] if holdout_site is None else holdout_site
            if held not in set(sites.tolist()):
                raise ValueError(f"holdout_site {held} not present; sites are {sites.tolist()}")
            site_test = site_id == held

        time_test = np.zeros(n, bool)
        if kind in ("temporal", "combined"):
            time_test = temporal_mask(time_days, test_frac)

        test = site_test | time_test if kind == "combined" else (
            site_test if kind == "loso" else time_test
        )
        # For the combined protocol the test set contains three qualitatively different
        # cells; reporting them separately is what distinguishes "cannot extrapolate in
        # space" from "cannot extrapolate in time".
        extra = {}
        if kind == "combined":
            extra = {
                "test_new_site": site_test & ~time_test,
                "test_new_time": time_test & ~site_test,
                "test_new_both": site_test & time_test,
            }
        # Validation always comes from the *end of the training period*, so early
        # stopping is judged on the same kind of shift the test set represents.
        pool = np.where(~test)[0]
        if len(pool) == 0:
            raise ValueError("split left no training samples")
        if time_days is not None and val_frac > 0:
            tt = np.asarray(time_days, float)[pool]
            cut = np.quantile(tt, 1.0 - val_frac)
            val_idx = pool[tt > cut]
        else:
            val_idx = rng.permutation(pool)[: int(round(val_frac * len(pool)))]
        val = np.zeros(n, bool)
        val[val_idx] = True
        out = {"train": ~test & ~val, "val": val, "test": test}
        out.update({k: v & ~val for k, v in extra.items()})
        return out

    if kind == "random":
        perm = rng.permutation(n)
        n_test = int(round(test_frac * n))
        test = np.zeros(n, bool)
        test[perm[:n_test]] = True
        pool = np.where(~test)[0]
        n_val = int(round(val_frac * len(pool)))
        val = np.zeros(n, bool)
        val[rng.permutation(pool)[:n_val]] = True
        return {"train": ~test & ~val, "val": val, "test": test}

    if kind == "uar":
        sites = np.unique(site_id)
        perm = rng.permutation(sites)
        n_test = int(round(test_frac * len(sites)))
        test_sites = set(perm[:n_test].tolist())
        test = np.isin(site_id, list(test_sites))
    elif kind == "checkerboard":
        # assign whole sites, using each site's mean position, so a site never straddles
        sites, inv = np.unique(site_id, return_inverse=True)
        slon = np.bincount(inv, weights=lon) / np.bincount(inv)
        slat = np.bincount(inv, weights=lat) / np.bincount(inv)
        site_train = checkerboard_mask(slon, slat, delta, offset_index, swap)
        test = ~site_train[inv]
    else:
        raise ValueError(f"unknown split kind '{kind}'")

    train_sites = np.unique(site_id[~test])
    n_val = max(1, int(round(val_frac * len(train_sites)))) if len(train_sites) > 1 else 0
    val_sites = set(rng.permutation(train_sites)[:n_val].tolist())
    val = np.isin(site_id, list(val_sites)) & ~test
    return {"train": ~test & ~val, "val": val, "test": test}


def split_report(masks: Dict[str, np.ndarray], site_id: np.ndarray) -> str:
    rows = []
    for k in ("train", "val", "test"):
        m = masks[k]
        rows.append(f"  {k:<6} {int(m.sum()):>8} samples  {len(np.unique(site_id[m])):>5} sites")
    return "\n".join(rows)
