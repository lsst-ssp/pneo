"""pneo: the probability that a newly detected moving object is a near-Earth object.

The public API:

* :func:`load_model` -- the population model P(NEO) is computed from;
* :func:`classify` -- P(NEO) for one or many tracklets (numpy in, numpy out);
* :func:`read_observations` -- ADES PSV or MPC 80-column astrometry to tracklets;
* :func:`posterior_map`, :func:`plot_posterior_map` -- where the probability
  lies in distance and range-rate, for one tracklet;
* :func:`fetch` -- download the model and the observatory codes.

Units are degrees and degrees per day; ``ra_rate`` is the on-sky rate,
d(alpha)/dt * cos(delta).
"""

from ._api import classify, fetch, plot_posterior_map, posterior_map, read_observations
from .models import PopulationComponent, PopulationModel, load_model

__all__ = [
    "load_model",
    "classify",
    "read_observations",
    "posterior_map",
    "plot_posterior_map",
    "fetch",
    "PopulationModel",
    "PopulationComponent",
]

try:  # single source of truth: installed metadata (setuptools-scm derived)
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("pneo")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0+uninstalled"
