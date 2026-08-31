"""WP5 — Marginal likelihoods, posterior, and diagnostics.

Frozen contract (WP0). See DESIGN.md §2.3–2.5, IMPLEMENTATION.md WP5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributable import Attributable
from .population import PopulationDensity
from .ranging import RangingGrid


@dataclass
class ClassificationResult:
    """Output of a single-tracklet classification.

    Attributes
    ----------
    p_neo : float
        P(NEO | A, V) — the calibrated posterior (DESIGN §2.4).
    log_lr : float
        ln(Lambda_NEO / Lambda_other), the prior-free sufficient statistic.
    lambda_neo, lambda_other : float
        The class marginal likelihoods (differential number densities in
        observable space).
    grid : RangingGrid
        The nodes used.
    node_density_neo, node_density_other : np.ndarray
        Per-node integrand values n_C * |J| (same length as grid.rho) — the
        (rho, rho_dot) posterior maps for diagnostics/collapse tests.
    """

    p_neo: float
    log_lr: float
    lambda_neo: float
    lambda_other: float
    grid: RangingGrid
    node_density_neo: np.ndarray
    node_density_other: np.ndarray

    def plot_posterior_map(self, ax=None, component: str = "both"):
        """Render the (rho, rho_dot) posterior density map(s). Returns the
        matplotlib Axes. Import matplotlib lazily (module must not require a
        display)."""
        raise NotImplementedError("WP5")


def classify(
    att: Attributable,
    neo: PopulationDensity,
    other: PopulationDensity,
    R_obs: np.ndarray | None = None,
    V_obs: np.ndarray | None = None,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    n_rho: int = 200,
    n_rhodot: int = 100,
    G: float | None = None,
) -> ClassificationResult:
    """Classify one tracklet (DESIGN §2.3–2.5).

    Pipeline: observer state (computed from att.epoch if R_obs/V_obs not
    given, geocentric) -> ranging grid over the admissible region -> per-node
    elements, H(V, rho), Jacobian dets (with the parameterization of each
    density) -> Lambda_C as weighted sums -> posterior. Hyperbolic/NaN nodes
    contribute zero. A tracklet with an empty admissible region or zero total
    likelihood returns p_neo = nan with a warning, not an exception.
    """
    raise NotImplementedError("WP5")


def classify_many(
    atts: list[Attributable],
    neo: PopulationDensity,
    other: PopulationDensity,
    **kwargs,
) -> list[ClassificationResult]:
    """Batch API. v0: a loop over classify(); vectorization is a v2 concern."""
    raise NotImplementedError("WP5")
