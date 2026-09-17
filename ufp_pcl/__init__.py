"""ufp_pcl -- Proxy-Consistency-Loss fusion of Earth-observation and location encoders,
targeted at sparse ultrafine-particle (UFP) observations.

Implements the architecture of Wang et al., "A Proxy Consistency Loss for Grounded
Fusion of Earth Observation and Location Encoders" (arXiv:2604.18881):

    e_obs = ObsEncoder(x)                      # gridded / time-series predictors
    e_loc = LocTimeEncoder(lon, lat, t)        # trainable GeoCLIP-style location-time encoder
    y_hat = PredHead([e_obs ; e_loc])          # primary task (UFP)
    z_hat = ProxyHead(e_loc)                   # proxy task, location encoder ONLY

    L = L_pred(y_hat, y) + lambda * (z_hat - z)^T Lambda (z_hat - z)

The proxy branch touches only the location encoder, so proxy points can be sampled
uniformly at random over the whole space-time domain (ratio rho per labelled batch),
giving "everywhere" supervision without extra labels.
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
