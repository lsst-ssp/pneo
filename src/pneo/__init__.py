"""pneo — Bayesian NEO tracklet classification by systematic ranging."""

from .attributable import Attributable, observer_state
from .classifier import ClassificationResult, classify, classify_many
from .models.digest2 import Digest2Component, PopulationComponent
from .ranging import RangingGrid, admissible_rhodot_interval, build_grid

__all__ = [
    "Attributable",
    "observer_state",
    "ClassificationResult",
    "classify",
    "classify_many",
    "Digest2Component",
    "PopulationComponent",
    "RangingGrid",
    "admissible_rhodot_interval",
    "build_grid",
]

try:  # single source of truth: installed metadata (setuptools-scm derived)
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("pneo")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0+uninstalled"
