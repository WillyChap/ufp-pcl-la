"""Explicit spatial climatology, so the location branch stops memorising monitors.

The decomposition is

    log10 UFP(place, time) = climatology(place) + anomaly(state, time)

and it is motivated by a measurement rather than a preference: of this model's test
error, **4.5% is between-site and 95.5% is within-site**.  Those are two different
problems and they have wildly different amounts of data behind them.  The between-site
part is seven numbers; the within-site part is ~29,000 hourly samples.

Training one network on both lets the 1.8M-parameter location branch chase the
between-site part with hourly supervision, and what it learns is which of seven
coordinates it is looking at -- validation turns over by epoch 2 and leave-one-site-out
R2 is negative.  Fitting the climatology separately, on seven site means, with one or two
ridge-regularised predictors, reaches leave-one-site-out R2 ~ +0.33 on the same quantity.
Four orders of magnitude fewer parameters, better out-of-sample skill.

Three things this module is careful about.

**No leakage.**  The fit uses only sites in the training mask.  Under leave-one-site-out
the held-out monitor contributes nothing to its own climatology, which is the whole point
of the protocol and easy to break by fitting on all sites first.

**Honest model selection.**  The predictor subset and ridge alpha are chosen by
leave-one-site-out *within the training sites*, so selection never sees the test site.
With six or seven points this is the difference between a defensible number and a fitted
one.

**Extrapolation.**  710 Near Road sits 0.28 decades in traffic intensity beyond every
other monitor, so predicting it means extrapolating past all training data -- LOO R2 is
+0.33 with it and +0.51 without.  Predictions are optionally clipped to the training
range plus a margin, because an unclipped linear extrapolation in log space is how a
near-road hotspot becomes a physically impossible number.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Ridge on standardised columns; returns (weights, intercept) for raw-space input."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return w / sd, float(y.mean() - (mu / sd) @ w)


def _loo_r2(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Leave-one-out R2 of a ridge fit; -inf when there is nothing to hold out."""
    n = len(y)
    if n < 4:
        return -np.inf
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        w, b = _ridge(X[m], y[m], alpha)
        pred[i] = X[i] @ w + b
    ss = float(((y - pred) ** 2).sum())
    st = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / st if st > 0 else -np.inf


@dataclass
class ClimoModel:
    """A tiny, leakage-free spatial climatology fitted on per-site means."""

    names: List[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    alpha: float = 1.0
    loo_r2: float = float("nan")
    n_sites: int = 0
    lo: float = -np.inf
    hi: float = np.inf
    clip: bool = True

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def fit(
        feats: np.ndarray,
        names: Sequence[str],
        y: np.ndarray,
        site_id: np.ndarray,
        signs: Optional[Dict[str, float]] = None,
        max_terms: int = 2,
        alphas: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0),
        clip: bool = True,
        clip_margin: float = 0.15,
        verbose: bool = True,
    ) -> "ClimoModel":
        """Fit on site means of `y`.  `feats` are static predictors at each sample."""
        sites = np.unique(site_id)
        if len(sites) < 4:
            m = ClimoModel(n_sites=len(sites), intercept=float(np.mean(y)) if len(y) else 0.0)
            if verbose:
                print(f"  climatology: only {len(sites)} training sites -- using the mean")
            return m

        # one clean row per site: the mean predictor and the mean target
        X = np.stack([feats[site_id == s].mean(0) for s in sites])
        Y = np.array([y[site_id == s].mean() for s in sites])
        keep = np.isfinite(X).all(0) & (X.std(0) > 1e-12)
        X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
        if not names:
            return ClimoModel(n_sites=len(sites), intercept=float(Y.mean()))

        # Physical sign constraints.  Leave-one-out selection optimises fit, not sense:
        # given a site set with little near-road contrast it will happily choose
        # "further from a road -> more ultrafine particles", which then paints mountains
        # and open ocean as the dirtiest places in the domain.  A climatology is
        # extrapolated far beyond its fitting sites, so a physically backwards coefficient
        # is not a small error -- it is wrong everywhere there is no monitor.
        want = {}
        for n in names:
            for key, sg in (signs or {}).items():
                if key in n:
                    want[n] = float(sg)
        best, rejected = None, 0
        for k in range(1, min(max_terms, X.shape[1], len(sites) - 2) + 1):
            for cols in itertools.combinations(range(X.shape[1]), k):
                Xi = X[:, list(cols)]
                for a in alphas:
                    w, _ = _ridge(Xi, Y, a)
                    bad = any(np.sign(w[i]) * want.get(names[c], np.sign(w[i])) < 0
                              for i, c in enumerate(cols))
                    if bad:
                        rejected += 1
                        continue
                    r2 = _loo_r2(Xi, Y, a)
                    if best is None or r2 > best[0]:
                        best = (r2, list(cols), a)
        if best is None:
            raise SystemExit("every candidate climatology violated its sign constraints")
        if rejected and verbose:
            print(f"  climatology: rejected {rejected} candidates on physical sign")
        r2, cols, alpha = best
        w, b = _ridge(X[:, cols], Y, alpha)
        m = ClimoModel(names=[names[c] for c in cols], weights=w, intercept=b,
                       alpha=alpha, loo_r2=r2, n_sites=len(sites), clip=clip)
        pred_sites = X[:, cols] @ w + b
        span = float(Y.max() - Y.min())
        m.lo, m.hi = float(Y.min() - clip_margin * span), float(Y.max() + clip_margin * span)
        if verbose:
            print(f"  climatology: {' + '.join(m.names)}  alpha={alpha:g}  "
                  f"LOO R2={r2:+.3f} over {len(sites)} training sites")
            print(f"    site means {Y.min():.3f}..{Y.max():.3f}; "
                  f"in-sample residual sd {np.std(Y - pred_sites):.3f}")
        return m

    # ---------------------------------------------------------------- prediction
    def predict(self, feats: np.ndarray, names: Sequence[str]) -> np.ndarray:
        if not len(self.names):
            return np.full(len(feats), self.intercept, np.float64)
        idx = [list(names).index(n) for n in self.names]
        out = feats[:, idx] @ self.weights + self.intercept
        if self.clip:
            out = np.clip(out, self.lo, self.hi)
        return np.asarray(out, np.float64)

    # ------------------------------------------------------------------- persist
    def state_dict(self) -> Dict:
        return {"names": list(self.names), "weights": np.asarray(self.weights).tolist(),
                "intercept": self.intercept, "alpha": self.alpha, "loo_r2": self.loo_r2,
                "n_sites": self.n_sites, "lo": self.lo, "hi": self.hi, "clip": self.clip}

    @staticmethod
    def from_state(d: Dict) -> "ClimoModel":
        return ClimoModel(names=list(d.get("names", [])),
                          weights=np.asarray(d.get("weights", []), float),
                          intercept=float(d.get("intercept", 0.0)),
                          alpha=float(d.get("alpha", 1.0)),
                          loo_r2=float(d.get("loo_r2", float("nan"))),
                          n_sites=int(d.get("n_sites", 0)),
                          lo=float(d.get("lo", -np.inf)), hi=float(d.get("hi", np.inf)),
                          clip=bool(d.get("clip", True)))
