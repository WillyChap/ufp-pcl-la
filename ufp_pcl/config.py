"""Configuration objects for the UFP proxy-consistency pipeline.

Everything the pipeline needs is declared here so that a run is reproducible from a
single YAML file.  `ufp-pcl inspect <file.nc> --emit-config` writes a starter YAML
filled in from the actual contents of your NetCDF file.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@dataclass
class CoordSpec:
    """Names of the coordinate variables inside the NetCDF file."""

    lon: str = "lon"
    lat: str = "lat"
    time: Optional[str] = "time"


@dataclass
class TargetSpec:
    """The sparse in-situ target, e.g. ultrafine particle number concentration."""

    var: str = "ufp"
    # "points": target has a sample/site dimension with lon/lat as auxiliary coords.
    # "grid":   target lives on the (time, lat, lon) grid and is mostly NaN.
    mode: str = "points"
    # log1p/log10 are the right default for particle *number* concentrations, which are
    # close to log-normal (Kumar et al. 2014).  Metrics are reported in both spaces.
    transform: str = "log10"
    # optional: name of a variable identifying the monitoring site, used for site-level
    # (rather than sample-level) train/test splits.  None -> sites are derived by
    # rounding coordinates.
    site_var: Optional[str] = None
    units: str = "particles cm-3"
    # Multiplied onto the raw target before `transform`.  Instruments are often archived
    # in convenience units (SCAQMD UFP is stored as counts/cm3 / 1000); converting here
    # keeps native-unit metrics in the units the exposure product is actually reported in,
    # without maintaining a rescaled copy of the file.
    scale: float = 1.0


@dataclass
class DataConfig:
    path: str = "data/ufp.nc"
    # Additional NetCDF files contributing gridded variables on their own lat/lon grids.
    # Real inputs rarely arrive pre-merged, and some of them -- a 100 m embedding field
    # beside 2 km chemistry -- should not be merged, because forcing them onto one grid
    # throws away the resolution that made them worth having.
    aux_paths: List[str] = field(default_factory=list)
    coords: CoordSpec = field(default_factory=CoordSpec)
    target: TargetSpec = field(default_factory=TargetSpec)

    # Predictor variables fed to the observation encoder.  "auto" = every gridded
    # variable that is neither the target nor a proxy.
    obs_vars: Any = "auto"
    # Continuous proxy fields used for the proxy-consistency loss.  These must have
    # (near-)complete coverage of the domain: that is the whole point.
    proxy_vars: List[str] = field(default_factory=list)
    # Per-proxy weights (the diagonal of Lambda).  Empty -> all ones.
    #
    # Prefer a mapping {name: weight}.  A bare list is positional, and the position is
    # NOT the order written in `proxy_vars`: variables are materialised grouped by grid,
    # in file order, so a list silently permutes the weights onto the wrong fields.
    proxy_weights: Any = field(default_factory=list)
    proxy_transform: Dict[str, str] = field(default_factory=dict)  # var -> none|log10|log1p
    # Require every proxy to be retrievable at a sampled point, rather than at least one.
    # Only meaningful with several proxies; True reproduces intersection sampling.
    proxy_require_all: bool = False

    # Bilinear sampling of gridded fields at arbitrary lon/lat.  "nearest" is faster and
    # avoids smearing categorical fields.
    interp: str = "bilinear"
    # Sequence length for the observation encoder.  1 -> plain MLP over a single day.
    # >1 -> a (seq_len, n_feat) window ending on the observation day, which switches the
    # default observation encoder to the BiLSTM+attention backbone of Wang et al.
    seq_len: int = 1
    # Drop labelled samples whose predictor stack contains NaN after interpolation.
    drop_nan_obs: bool = True
    # Physically-motivated derived predictors appended to the observation stack.
    # See ufp_pcl/features.py for the catalogue; "fnr" (HCHO/NO2) is the ozone-production
    # regime indicator that flags when secondary formation pathways can be active.
    derived: List[str] = field(default_factory=list)
    # Restrict to daylight hours.  TEMPO only observes in daylight, so training on night
    # hours means training the observation encoder on fill values.
    daylight_only: bool = False
    solar_elevation_min: float = 0.0

    # Inclusive [lo, hi] physical bounds per variable; sampled values outside become NaN.
    # Retrieval outliers are common in satellite L3 (TEMPO VCDs reach 3e17 molec/cm2,
    # ~5x anything physical) and a single one survives standardisation as a huge z-score.
    # Applied at sample time so no file has to be rewritten.
    valid_ranges: Dict[str, List[float]] = field(default_factory=dict)
    # Static predictors for the explicit climatology of the `climo_anomaly` variant.
    # Prefix an entry with "log:" to fit on log10(x+1) -- traffic intensity spans orders
    # of magnitude and is meaningless untransformed.  Keep this list SHORT: it is fitted
    # on one row per training site, so 6-7 points, and every extra term is a chance to
    # fit noise.  Empty -> fall back to every static predictor, selected by LOO.
    climo_vars: List[str] = field(default_factory=list)
    # Multiplicative correction for the monitoring network's unrepresentativeness.
    #
    # The seven AB-617 monitors are sited for regulatory reasons at populated and
    # near-road locations, so their geometric mean (14,602 cm-3 in daylight) sits 1.42x
    # above an independent background sample (MATES, 10,299).  A model fitted on them
    # reproduces that level everywhere, and independent validation measures exactly the
    # predicted over-prediction: 1.36-1.52x across four structurally different models.
    #
    # This divides predictions by the factor.  It is estimated from out-of-network
    # observations, not tuned -- set it to 1.0 to disable, and re-estimate it if the
    # network changes.
    representativeness: float = 1.0

    # Extra site-level observations for the climatology only: a NetCDF with `latitude`,
    # `longitude` and a per-site level variable.  Independent validation showed the
    # 7-site climatology over-predicts by ~2.6x at unmonitored background locations and
    # even orders them backwards -- it was fitted on a network whose only high-road-density
    # point is a near-road station, so it extrapolates that relationship everywhere.
    # Historical campaign sites cover the background end the network never samples.
    climo_extra: str = ""
    climo_extra_var: str = "ufp_geomean"

    # Expected sign of each climatology predictor, matched as a substring of the name.
    # More road / more traffic -> more UFP; further from a road -> less.
    climo_signs: Dict[str, float] = field(default_factory=lambda: {
        "road_w_": 1.0, "truck5": 1.0, "aadt": 1.0, "road_dist_km": -1.0})

    # Maximum number of terms the climatology may select.
    climo_max_terms: int = 2
    # Clip climatology predictions to the training site range plus a margin.
    climo_clip: bool = True

    # Analysis domain [lon_min, lon_max, lat_min, lat_max].  Empty -> the extent of the
    # gridded fields.  Set it when a predictor covers less than the rest: a precomputed
    # embedding field is sampled with border padding, so outside its own extent the model
    # silently reuses edge values -- a map over ocean and desert built from the embedding
    # of the nearest coastline.  It also stops the proxy sampler drawing training signal
    # from places the model has no business predicting.
    domain: List[float] = field(default_factory=list)

    # Keep every Nth hour of the labelled record.  1 -> keep everything.
    #
    # A diagnostic for whether the sample count is real.  Adjacent hours at one monitor
    # are near-duplicates, so 18,000 samples may be worth only a few hundred independent
    # observations -- and if that is the binding constraint, training on a third of the
    # data should barely move validation loss.  Applied before the split, so train, val
    # and test are thinned alike and remain comparable.
    time_stride: int = 1

    # Inclusive ISO date bounds on labelled samples, e.g. ["2023-08-02", "2025-06-18"].
    # Collections harmonised to a common length often pad the short records by repeating
    # their last field; that reads as real data, so trim to the shortest honest record.
    time_range: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """How train / val / test are carved out.

    With only ~7 SCAQMD sites, the checkerboard protocol of the PCL paper has no room to
    operate; `loso` is the spatial-generalisation test that fits this network size, and
    `temporal` is the leakage-safe protocol for the hourly time series.  Use `combined`
    to hold out one site *and* the final fraction of time, which is the honest test of
    "new place, new period".
    """

    # "temporal"     -> first (1-test_frac) of the record trains, the tail tests
    # "loso"         -> leave-one-site-out: `holdout_site` is the test set
    # "combined"     -> loso for space AND temporal for time, simultaneously
    # "uar"          -> uniform-at-random over sites (geographic in-sample)
    # "checkerboard" -> systematic spatial extrapolation (needs many sites)
    # "random"       -> sample-level random split (leaks; debugging only)
    kind: str = "temporal"
    test_frac: float = 0.2
    val_frac: float = 0.15  # carved out of the training portion
    seed: int = 0
    # loso parameters
    holdout_site: Optional[int] = None   # site index; None -> the site with most samples
    # checkerboard parameters
    # Block length in days for kind="blocked_time".
    block_days: float = 7.0
    delta: float = 0.25          # square side length in degrees
    offset_index: int = 0        # 0..3 -> (none, +d/2 lon, +d/2 lat, both)
    swap: bool = False           # swap which squares are train vs test


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
@dataclass
class LocationEncoderConfig:
    """GeoCLIP-style location encoder: Equal Earth projection -> multi-scale random
    Fourier features -> one MLP per scale -> summed embedding."""

    dim: int = 256
    n_scales: int = 4
    sigma_min: float = 2.0 ** 0     # smallest RFF bandwidth (broad / continental)
    sigma_max: float = 2.0 ** 8     # largest RFF bandwidth  (fine / neighbourhood)
    n_frequencies: int = 64         # per scale; feature dim per scale = 2 * n_frequencies
    hidden: int = 512
    n_layers: int = 2
    dropout: float = 0.0
    # Metric scale, in km, that the projected coordinates are divided by before the RFF.
    # Smaller -> the encoder can resolve finer spatial structure.  The LA basin is ~100 km
    # across and TEMPO L3 pixels are ~2 km, so 25 km puts the coarsest RFF scale at basin
    # size and the finest well below a pixel.  Raise it for regional/continental domains.
    length_scale_km: float = 25.0

    # Where the geographic prior comes from:
    #   "trainable"   -- the multi-scale RFF encoder above, learned from scratch (default)
    #   "precomputed" -- a frozen externally pretrained embedding field, read from
    #                    `pretrained_path` and sampled bilinearly; only a small adapter
    #                    is trained.  This is the "frozen pretrained LE fusion" baseline.
    #   "hybrid"      -- both, concatenated: a warm start from a pretrained field plus a
    #                    trainable component that the proxy loss can still shape.
    source: str = "trainable"
    pretrained_path: Optional[str] = None
    pretrained_vars: List[str] = field(default_factory=list)
    pretrained_prefix: Optional[str] = None
    pretrained_adapter_hidden: int = 256


@dataclass
class TimeEncoderConfig:
    """Temporal half of the location-time encoder.

    UFP is a diurnal-cycle problem before it is anything else: a traffic-driven morning
    number peak and a photochemical afternoon peak sit on top of a seasonal cycle, and
    the morning peak collapses on weekends.  So hour-of-day and day-of-week are on by
    default -- without them the encoder cannot represent the very contrast (weekday
    morning vs weekend morning) that separates primary from secondary formation.
    """

    enabled: bool = True
    dim: int = 64
    # cyclic day-of-year harmonics + random Fourier features, per Sec. 4.3 of the paper
    n_harmonics: int = 6
    n_frequencies: int = 32
    hidden: int = 128
    use_year: bool = True          # a small MLP branch on (year - year0)
    use_hour: bool = True          # hour-of-day harmonics (diurnal cycle)
    n_hour_harmonics: int = 3      # up to 3 captures the twin-peak weekday shape
    use_dayofweek: bool = True     # weekday/weekend contrast = the traffic signal


@dataclass
class ObsEncoderConfig:
    kind: str = "auto"             # auto|mlp|bilstm|none
    dim: int = 256
    hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    lstm_hidden: int = 128
    lstm_layers: int = 3


@dataclass
class HeadConfig:
    hidden: int = 256
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class ModelConfig:
    location: LocationEncoderConfig = field(default_factory=LocationEncoderConfig)
    time: TimeEncoderConfig = field(default_factory=TimeEncoderConfig)
    obs: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    pred_head: HeadConfig = field(default_factory=HeadConfig)
    proxy_head: HeadConfig = field(default_factory=lambda: HeadConfig(hidden=256, n_layers=2))
    # Fusion variants to compare (Table 1 of the paper):
    #   pcl            -> trained location encoder + proxy consistency loss   (ours)
    #   fusion         -> trained location encoder, no PCL
    #   proxy_pretrain -> pretrain the location encoder on the proxy, freeze, then fuse
    #   obs_only       -> no location encoder at all
    #   proxy_stacked  -> proxy fields appended as extra observation-encoder inputs
    #   climo_anomaly  -> explicit spatial climatology + PCL-regularised anomaly model.
    #                     The network fits the departure from a tiny ridge climatology
    #                     rather than the field itself, which matches how the error is
    #                     actually distributed (4.5% between-site, 95.5% within-site) and
    #                     removes most of the location encoder's incentive to memorise
    #                     monitor coordinates.  See ufp_pcl/climo.py.
    variant: str = "climo_anomaly" if False else "pcl"


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    # Weight decay applied to the LOCATION encoder only, overriding `weight_decay`.
    # <= 0 means "use the global value".
    #
    # Removing the location branch entirely (variant obs_only) delays the validation
    # turnover from epoch 2 to epoch 32 -- it is demonstrably the part that overfits --
    # but costs 0.11 test R2, so deleting it is not the answer.  Dropout, coordinate
    # jitter, coarser basis functions and a 4x smaller branch all failed to constrain it;
    # global weight decay was the only lever that helped at all.  This applies that lever
    # where the problem is, at a strength the global setting cannot reach without also
    # crushing the observation encoder.
    location_weight_decay: float = 0.0

    # Gaussian jitter, in km, added to training coordinates each batch.
    #
    # With seven monitors and a location encoder whose finest basis function has a ~100 m
    # wavelength, the encoder can place an independent value at each site: it memorises
    # seven points instead of learning a field, which is why validation loss turns over by
    # epoch 2 while training loss keeps falling.  Jittering the coordinates makes that
    # impossible -- a monitor is no longer a fixed point but a small neighbourhood, so the
    # encoder must be smooth on the jitter scale to fit it at all.
    #
    # Choose it relative to the monitor spacing, not the grid: the closest pair here
    # (Compton and 710 Near Road) is 4.6 km apart, so 1-2 km blurs the point identity
    # while leaving genuine between-site contrast intact.  Applies to training batches
    # only -- validation, test and prediction use exact coordinates.
    coord_jitter_km: float = 0.0

    # loss weight on the proxy-consistency term (lambda in Eq. 1)
    lam: float = 0.2
    # proxy sampling ratio rho: rho * batch_size proxy points per optimisation step,
    # drawn uniformly at random over space-time.  Paper finds gains saturate at rho~8-16.
    rho: int = 16
    # If True, also evaluate the PCL at the labelled monitor coordinates
    # ("EPA sites + random" in Fig. 3).  Paper finds random-only is sufficient.
    proxy_at_label_sites: bool = False
    proxy_pretrain_epochs: int = 50     # only used by variant="proxy_pretrain"
    patience: int = 15                  # early stopping on validation loss
    lr_patience: int = 6
    lr_factor: float = 0.5
    num_workers: int = 0
    device: str = "auto"                # auto|cpu|mps|cuda
    seed: int = 0
    out_dir: str = "outputs/run"
    log_every: int = 1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _build(cls, blob):
    """Recursively instantiate nested dataclasses from plain dicts."""
    if not dataclasses.is_dataclass(cls) or blob is None:
        return blob
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (blob or {}).items():
        if key not in fields:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = fields[key].type
        # resolve string annotations from `from __future__ import annotations`
        if isinstance(ftype, str):
            ftype = {
                "CoordSpec": CoordSpec, "TargetSpec": TargetSpec, "DataConfig": DataConfig,
                "SplitConfig": SplitConfig, "ModelConfig": ModelConfig,
                "TrainConfig": TrainConfig, "LocationEncoderConfig": LocationEncoderConfig,
                "TimeEncoderConfig": TimeEncoderConfig, "ObsEncoderConfig": ObsEncoderConfig,
                "HeadConfig": HeadConfig,
            }.get(ftype, None)
        kwargs[key] = _build(ftype, value) if dataclasses.is_dataclass(ftype) else value
    return cls(**kwargs)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config.  `overrides` uses dotted keys, e.g. {"train.rho": 8}."""
    with open(path) as fh:
        blob = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = blob
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return _build(Config, blob)
