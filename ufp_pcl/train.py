"""Training loop for proxy-consistent fusion.

Each optimisation step consumes two independently drawn minibatches:

    B      labelled UFP observations  -> prediction loss, back-props through everything
    rho*B  uniform-at-random proxy points -> proxy-consistency loss, back-props through
                                             the location encoder and proxy head only

`rho` is the single most important knob after `lam`: the paper's scaling study shows
performance climbing steeply up to rho ~ 8-16 and then flattening, with slight
degradation at very large rho where the imperfect proxy starts to dominate.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import DataBundle
from .data.netcdf import calendar_features
from .evaluate import climo_only_metrics, evaluate_split, format_metrics
from .losses import ProxyConsistencyLoss, masked_mse
from .data.dataset import resolve_proxy_weights
from .models.fusion import build_model


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def proxy_coords_to_tensor(coords_lltd: np.ndarray, device) -> torch.Tensor:
    """(n,3) lon/lat/time_days -> the (n,6) coordinate layout used by the label batches."""
    cal = calendar_features(coords_lltd[:, 2])
    arr = np.stack(
        [coords_lltd[:, 0], coords_lltd[:, 1], cal["doy"], cal["year"], cal["hour"],
         cal["dow"], coords_lltd[:, 2]],
        axis=1,
    ).astype(np.float32)
    return torch.as_tensor(arr, device=device)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Config, bundle: DataBundle):
        self.cfg = cfg
        self.bundle = bundle
        set_seed(cfg.train.seed)
        self.device = resolve_device(cfg.train.device)

        self.model = build_model(
            cfg, bundle.n_obs_features, bundle.n_proxy,
            seq_len=bundle.seq_len, year0=bundle.coord_meta["year0"], seed=cfg.train.seed,
        ).to(self.device)

        self.use_pcl = (
            cfg.model.variant in ("pcl", "proxy_pretrain", "climo_anomaly")
            and self.model.proxy_head is not None
            and bundle.proxy is not None
        )
        weights = resolve_proxy_weights(cfg.data.proxy_weights, bundle.proxy_names)
        self.pcl = ProxyConsistencyLoss(weights, n_proxy=max(bundle.n_proxy, 1)).to(self.device)

        self.opt = torch.optim.AdamW(
            self._param_groups(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=cfg.train.lr_factor, patience=cfg.train.lr_patience
        )
        self.history: List[Dict[str, float]] = []
        self.out_dir = cfg.train.out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------------------------------------------------------------------------------
    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle, drop_last=False,
            num_workers=self.cfg.train.num_workers,
        )

    def _proxy_batch(self, n: int, label_coords: Optional[torch.Tensor] = None):
        coords_lltd, z = self.bundle.proxy.draw(n)
        pc = proxy_coords_to_tensor(coords_lltd, self.device)
        zt = torch.as_tensor(z, device=self.device)
        if label_coords is not None:
            # "sites + random" variant of Fig. 3: additionally anchor the PCL at the
            # monitor coordinates and times present in this labelled batch.  The paper
            # finds random-only sampling is as good or better, so this is off by default.
            lc = label_coords.detach().cpu().numpy()
            z2 = self.bundle.proxy.at(lc[:, 0], lc[:, 1], lc[:, 6])
            ok = np.isfinite(z2).all(axis=1)
            if ok.any():
                keep = torch.as_tensor(ok, device=label_coords.device)
                pc = torch.cat([pc, label_coords[keep].to(self.device)], dim=0)
                zt = torch.cat([zt, torch.as_tensor(z2[ok], device=self.device)], dim=0)
        return pc, zt

    # ---------------------------------------------------------------------------- steps
    def _param_groups(self):
        """Parameter groups, so the location encoder can be decayed separately.

        `obs_only` delays the validation turnover from epoch 2 to epoch 32, which places
        the overfitting squarely in the location branch -- but deleting that branch costs
        0.11 test R2, so it has to be constrained rather than removed.  A single global
        weight decay cannot do that: raising it far enough to discipline 2.1M location
        parameters also crushes the observation encoder, which is not misbehaving.
        """
        wd = float(getattr(self.cfg.train, "location_weight_decay", 0.0) or 0.0)
        if wd <= 0 or not getattr(self.model, "use_loc", False):
            return self.model.parameters()
        loc, rest = [], []
        for n, p in self.model.named_parameters():
            (loc if n.startswith("loc_encoder") else rest).append(p)
        print(f"  location weight decay {wd:g} on {sum(p.numel() for p in loc):,} params; "
              f"{self.cfg.train.weight_decay:g} on the other {sum(p.numel() for p in rest):,}")
        return [{"params": rest, "weight_decay": self.cfg.train.weight_decay},
                {"params": loc, "weight_decay": wd}]

    def _jitter(self, coords: torch.Tensor) -> torch.Tensor:
        """Perturb lon/lat by a Gaussian in km.  Training batches only."""
        km = float(self.cfg.train.coord_jitter_km)
        if km <= 0:
            return coords
        out = coords.clone()
        lat = out[:, 1]
        dlat = km / 110.57
        dlon = km / (111.32 * torch.cos(torch.deg2rad(lat)).clamp(min=0.2))
        out[:, 0] = out[:, 0] + torch.randn_like(out[:, 0]) * dlon
        out[:, 1] = lat + torch.randn_like(lat) * dlat
        return out

    def _train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        tot, tot_pred, tot_pcl, nb = 0.0, 0.0, 0.0, 0
        n_proxy_pts = int(self.cfg.train.rho * self.cfg.train.batch_size)
        for obs, coords, y in loader:
            obs, coords, y = obs.to(self.device), coords.to(self.device), y.to(self.device)
            coords = self._jitter(coords)
            loss_pred = masked_mse(self.model(obs, coords), y)
            loss = loss_pred
            l_pcl = torch.zeros((), device=self.device)
            if self.use_pcl and n_proxy_pts > 0:
                pc, z = self._proxy_batch(
                    n_proxy_pts, coords if self.cfg.train.proxy_at_label_sites else None
                )
                l_pcl = self.pcl(self.model.forward_proxy(pc), z)
                loss = loss + self.cfg.train.lam * l_pcl

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()

            tot += float(loss); tot_pred += float(loss_pred); tot_pcl += float(l_pcl); nb += 1
        nb = max(nb, 1)
        return {"loss": tot / nb, "loss_pred": tot_pred / nb, "loss_pcl": tot_pcl / nb}

    @torch.no_grad()
    def _val_loss(self, loader) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for obs, coords, y in loader:
            obs, coords, y = obs.to(self.device), coords.to(self.device), y.to(self.device)
            tot += float(masked_mse(self.model(obs, coords), y)) * len(y)
            n += len(y)
        return tot / max(n, 1)

    # -------------------------------------------------------------------- proxy pretrain
    def pretrain_location_encoder(self, epochs: int) -> None:
        """Two-stage baseline: fit the location encoder on the proxy alone, then freeze."""
        if not self.use_pcl:
            return
        params = list(self.model.location_parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.train.lr,
                                weight_decay=self.cfg.train.weight_decay)
        n = int(self.cfg.train.rho * self.cfg.train.batch_size)
        steps = max(1, len(self.bundle.train) // self.cfg.train.batch_size)
        self.model.train()
        for ep in range(epochs):
            tot = 0.0
            for _ in range(steps):
                pc, z = self._proxy_batch(n)
                loss = self.pcl(self.model.forward_proxy(pc), z)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.train.grad_clip)
                opt.step()
                tot += float(loss)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [pretrain] epoch {ep+1:3d}/{epochs}  proxy_mse={tot/steps:.4f}")
        self.model.freeze_location(True)
        self.use_pcl = False   # stage 2 trains the fusion model with the encoder frozen
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=self.cfg.train.lr_factor,
            patience=self.cfg.train.lr_patience,
        )

    # -------------------------------------------------------------------------- fitting
    def fit(self) -> Dict[str, float]:
        tr, va = self._loader(self.bundle.train, True), self._loader(self.bundle.val, False)
        if self.cfg.model.variant == "proxy_pretrain":
            print(f"stage 1: pretraining location encoder on proxy "
                  f"({self.cfg.train.proxy_pretrain_epochs} epochs)")
            self.pretrain_location_encoder(self.cfg.train.proxy_pretrain_epochs)

        best, best_ep, bad = float("inf"), -1, 0
        ckpt = os.path.join(self.out_dir, "best.pt")
        t0 = time.time()
        for ep in range(self.cfg.train.epochs):
            te = time.time()
            stats = self._train_epoch(tr)
            vl = self._val_loss(va) if len(self.bundle.val) else stats["loss_pred"]
            self.sched.step(vl)
            rec = {"epoch": ep + 1, **stats, "val_loss": vl,
                   "lr": self.opt.param_groups[0]["lr"], "sec": time.time() - te}
            self.history.append(rec)
            if vl < best - 1e-6:
                best, best_ep, bad = vl, ep + 1, 0
                self.save(ckpt)
            else:
                bad += 1
            if (ep + 1) % self.cfg.train.log_every == 0:
                print(f"epoch {ep+1:3d}/{self.cfg.train.epochs}  "
                      f"loss={stats['loss']:.4f}  pred={stats['loss_pred']:.4f}  "
                      f"pcl={stats['loss_pcl']:.4f}  val={vl:.4f}  "
                      f"lr={rec['lr']:.2e}  {rec['sec']:.1f}s"
                      + ("  *" if bad == 0 else ""))
            if bad >= self.cfg.train.patience:
                print(f"early stop at epoch {ep+1} (best {best:.4f} @ {best_ep})")
                break

        self.load(ckpt)
        metrics = self.evaluate()
        metrics.update({"best_val_loss": best, "best_epoch": best_ep,
                        "train_seconds": time.time() - t0,
                        "sec_per_epoch": float(np.mean([h["sec"] for h in self.history]))})
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        with open(os.path.join(self.out_dir, "history.json"), "w") as fh:
            json.dump(self.history, fh, indent=2)
        self.cfg.save(os.path.join(self.out_dir, "config.yaml"))
        return metrics

    def evaluate(self) -> Dict[str, float]:
        out = {}
        cm = getattr(self.bundle, "climo", None)
        if cm is not None and cm.names:
            # so a run is self-describing: which climatology was fitted, on how many
            # sites, and how well it held up when one of them was withheld
            out["climo_loo_r2"] = float(cm.loo_r2)
            out["climo_n_sites"] = int(cm.n_sites)
            out["climo_terms"] = ", ".join(cm.names)
        splits = [("train", self.bundle.train), ("val", self.bundle.val),
                  ("test", self.bundle.test)]
        splits += list(self.bundle.extra_sets.items())
        for name, ds in splits:
            if len(ds) == 0:
                continue
            out.update(evaluate_split(self.model, ds, self.device, self.bundle.y_scaler,
                                      self.bundle.y_transform, prefix=f"{name}_"))
            out.update(climo_only_metrics(ds, self.bundle.y_scaler,
                                          self.bundle.y_transform, prefix=f"{name}_"))
        print("\n" + "-" * 72)
        for name in ["train", "val", "test"] + list(self.bundle.extra_sets):
            sub = {k[len(name) + 1:]: v for k, v in out.items() if k.startswith(name + "_")}
            if sub:
                print(f"{name:<6} {format_metrics(sub)}")
        print("-" * 72)
        return out

    # ------------------------------------------------------------------- serialisation
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "obs_scaler": self.bundle.obs_scaler.state_dict(),
                "y_scaler": self.bundle.y_scaler.state_dict(),
                "proxy_scaler": (self.bundle.proxy.scaler.state_dict()
                                 if self.bundle.proxy and self.bundle.proxy.scaler else None),
                "obs_names": self.bundle.obs_names,
                "proxy_names": self.bundle.proxy_names,
                "coord_meta": self.bundle.coord_meta,
                "seq_len": self.bundle.seq_len,
                # without this the run cannot reconstruct UFP later: the network only
                # ever predicts the departure from this climatology
                "climo": (self.bundle.climo.state_dict()
                          if getattr(self.bundle, "climo", None) is not None else None),
            },
            path,
        )

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False: non-persistent buffers (the precomputed embedding cube) are
        # rebuilt by the constructor from the configured path, not restored here.
        self.model.load_state_dict(blob["model"], strict=False)
