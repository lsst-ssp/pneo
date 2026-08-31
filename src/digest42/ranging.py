"""WP3 — Admissible region and (rho, rho_dot) integration grids.

Frozen contract (WP0). See DESIGN.md §2.5, IMPLEMENTATION.md WP3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributable import Attributable


def admissible_rhodot_interval(
    att: Attributable,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound-orbit interval of rho_dot at each rho (DESIGN §2.5).

    Solves 0.5 |v|^2 - mu/|r| < 0 for rho_dot at fixed rho; quadratic in
    rho_dot, so the admissible set is a single interval (possibly empty).

    Parameters
    ----------
    att : Attributable
    R_obs, V_obs : (3,) observer heliocentric state, equ frame [AU, AU/day].
    rho : (...,) topocentric ranges [AU].

    Returns
    -------
    (lo, hi) : each shape (...,) [AU/day]; lo > hi (or NaN) marks an empty
    interval at that rho.
    """
    raise NotImplementedError("WP3")


@dataclass
class RangingGrid:
    """Quadrature nodes over the admissible region.

    Attributes
    ----------
    rho : (N,) node ranges [AU].
    rho_dot : (N,) node range-rates [AU/day].
    weight : (N,) cell measures d(rho) * d(rho_dot) [AU^2/day] such that
        an integral over the admissible region is sum(f(node) * weight).
    n_rho, n_rhodot : the grid resolution it was built with.
    """

    rho: np.ndarray
    rho_dot: np.ndarray
    weight: np.ndarray
    n_rho: int
    n_rhodot: int


def build_grid(
    att: Attributable,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    n_rho: int = 200,
    n_rhodot: int = 100,
) -> RangingGrid:
    """Log-spaced rho x linear rho_dot grid over the admissible region.

    For each of the n_rho log-spaced ranges, place n_rhodot nodes across the
    (nonempty) admissible rho_dot interval; weights are the local cell
    measures (midpoint rule). Empty-interval ranges contribute no nodes; a
    fully empty region returns a zero-length grid.

    WP3 acceptance: boundaries match a brute-force scan of the energy
    condition; integrals converge under refinement.
    """
    raise NotImplementedError("WP3")
