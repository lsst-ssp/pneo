"""WP3 — Admissible region and (rho, rho_dot) integration grids.

Frozen contract (WP0). See docs/design/DESIGN.md §2.5, docs/design/IMPLEMENTATION.md WP3.

Derivation of the bound-orbit condition (DESIGN §2.1 + §2.5)
------------------------------------------------------------
At fixed topocentric range ``rho`` the heliocentric state implied by the
attributable is

    r = R_obs + rho * u_hat
    v = V_obs + rho * (ra_rate * du/dra + dec_rate * du/ddec) + rho_dot * u_hat
      = W(rho) + rho_dot * u_hat,

so only ``v`` depends on ``rho_dot``, and it does so affinely.  Since
``|u_hat| = 1``,

    |v|^2 = rho_dot^2 + 2 (u_hat . W) rho_dot + |W|^2.

The bound-orbit (negative two-body energy) condition
``0.5 |v|^2 - mu / |r| < 0`` is therefore the quadratic inequality

    rho_dot^2 + 2 (u_hat . W) rho_dot + |W|^2 - 2 mu / |r| < 0,

whose solution set is the open interval between the two roots

    rho_dot = -(u_hat . W) +/- sqrt(D),
    D = (u_hat . W)^2 - |W|^2 + 2 mu / |r|
      = 2 mu / |r| - |W_perp|^2,

with ``W_perp`` the component of ``W`` orthogonal to ``u_hat`` (the transverse
heliocentric velocity, which ``rho_dot`` cannot change).  The last form makes
the geometry explicit: the region is nonempty exactly where the local escape
speed ``sqrt(2 mu / r)`` exceeds the transverse speed already fixed by the
attributable, and its half-width is that "leftover" speed budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributable import Attributable
from .constants import MU_SUN
from .transforms import tangent_basis


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
    rho = np.asarray(rho, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)
    if R_obs.shape[-1] != 3 or V_obs.shape[-1] != 3:
        raise ValueError("R_obs and V_obs must be (3,) vectors")

    u_hat, du_dra, du_ddec = tangent_basis(att.ra, att.dec)

    # Motion-induced (transverse) part of the heliocentric velocity:
    # W = V_obs + rho * (ra_rate * du/dra + dec_rate * du/ddec).
    mu_vec = att.ra_rate * du_dra + att.dec_rate * du_ddec  # (3,)
    W = V_obs + rho[..., None] * mu_vec  # (..., 3)

    # Heliocentric position and distance.
    r_vec = R_obs + rho[..., None] * u_hat  # (..., 3)
    r = np.sqrt(np.sum(r_vec * r_vec, axis=-1))  # (...,)

    b = np.sum(W * u_hat, axis=-1)  # u_hat . W
    w2 = np.sum(W * W, axis=-1)  # |W|^2

    # Discriminant / 4 of rho_dot^2 + 2 b rho_dot + (w2 - 2 mu / r) = 0.
    # Equivalently 2 mu / r - |W_perp|^2 with |W_perp|^2 = w2 - b^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b * b - w2 + 2.0 * MU_SUN / r

    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    lo = np.where(ok, -b - sq, np.nan)
    hi = np.where(ok, -b + sq, np.nan)
    return lo, hi


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

    Notes
    -----
    The quadrature measure is ``d(rho) d(rho_dot)``, NOT ``d(log rho)
    d(rho_dot)``: the log spacing is only a node placement choice, so the
    rho cell width is the width of the cell in LINEAR rho.  Cell edges are
    the geometric midpoints between adjacent nodes, clamped to
    ``[rho_min, rho_max]`` at the ends, so the rho cells tile the interval
    exactly (``sum of d(rho) cells == rho_max - rho_min``).
    """
    if not (rho_min > 0.0):
        raise ValueError("rho_min must be > 0 (log-spaced rho grid)")
    if not (rho_max > rho_min):
        raise ValueError("rho_max must be > rho_min")
    n_rho = int(n_rho)
    n_rhodot = int(n_rhodot)
    if n_rho < 1 or n_rhodot < 1:
        raise ValueError("n_rho and n_rhodot must be >= 1")

    # --- rho nodes and LINEAR cell widths -----------------------------------
    if n_rho == 1:
        rho_nodes = np.array([np.sqrt(rho_min * rho_max)])
        d_rho = np.array([rho_max - rho_min])
    else:
        rho_nodes = np.geomspace(rho_min, rho_max, n_rho)
        edges = np.empty(n_rho + 1, dtype=float)
        edges[0] = rho_min
        edges[-1] = rho_max
        edges[1:-1] = np.sqrt(rho_nodes[:-1] * rho_nodes[1:])
        d_rho = np.diff(edges)

    # --- per-rho admissible rho_dot interval --------------------------------
    lo, hi = admissible_rhodot_interval(att, R_obs, V_obs, rho_nodes)
    width = hi - lo
    valid = np.isfinite(width) & (width > 0.0)

    if not np.any(valid):
        empty = np.zeros(0, dtype=float)
        return RangingGrid(
            rho=empty, rho_dot=empty.copy(), weight=empty.copy(),
            n_rho=n_rho, n_rhodot=n_rhodot,
        )

    rho_v = rho_nodes[valid]
    lo_v = lo[valid]
    d_rhodot = width[valid] / n_rhodot  # (M,) linear rho_dot cell width
    d_rho_v = d_rho[valid]

    # Midpoint nodes across [lo, hi].
    offsets = (np.arange(n_rhodot, dtype=float) + 0.5)  # (n_rhodot,)
    rhodot_nodes = lo_v[:, None] + d_rhodot[:, None] * offsets[None, :]  # (M, K)

    rho_out = np.repeat(rho_v, n_rhodot)
    weight_out = np.repeat(d_rho_v * d_rhodot, n_rhodot)

    return RangingGrid(
        rho=rho_out,
        rho_dot=rhodot_nodes.ravel(),
        weight=weight_out,
        n_rho=n_rho,
        n_rhodot=n_rhodot,
    )
