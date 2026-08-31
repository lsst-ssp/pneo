"""WP2 — Geometry, photometry, element transforms, and Jacobians.

Frozen contract (WP0). Implementations must preserve every public signature,
unit, and frame documented here. See DESIGN.md §2.1–2.3, IMPLEMENTATION.md WP2.

Batching convention: all functions accept either scalars/(3,) vectors or
batched arrays with a LEADING batch dimension (shape (N,) / (N, 3) / (N, 6));
outputs follow. Frames: "equ" = equatorial ICRS/J2000, "ecl" = ecliptic J2000.
The equatorial->ecliptic rotation is applied exactly once in the pipeline, in
`state_equ_to_ecl` (IMPLEMENTATION §1).

Element order everywhere: (a, e, i, Omega, omega, M) — ecliptic J2000, rad, AU.
"""

from __future__ import annotations

import numpy as np

from .constants import G_SLOPE_DEFAULT

# ----------------------------------------------------------------------------
# Sky geometry (equatorial frame)
# ----------------------------------------------------------------------------


def unit_vector(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Topocentric line-of-sight unit vector u_hat(ra, dec), equ frame. -> (..., 3)"""
    ra, dec = np.asarray(ra), np.asarray(dec)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def tangent_basis(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u_hat, du/dra, du/ddec), each (..., 3), equ frame.

    du/dra is the RAW partial (no cos(dec) factor), consistent with
    Attributable.ra_rate being d(ra)/dt. Implemented in WP0 and shared by
    WP2/WP3; do not reimplement.
    """
    ra, dec = np.asarray(ra), np.asarray(dec)
    sa, ca = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    u = np.stack([cd * ca, cd * sa, sd], axis=-1)
    du_dra = np.stack([-cd * sa, cd * ca, np.zeros_like(sd)], axis=-1)
    du_ddec = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
    return u, du_dra, du_ddec


def topocentric_to_heliocentric(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_rate: np.ndarray,
    dec_rate: np.ndarray,
    rho: np.ndarray,
    rho_dot: np.ndarray,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(A, rho, rho_dot) -> heliocentric state (r, v), equ frame (DESIGN §2.1).

    r = R_obs + rho * u_hat
    v = V_obs + rho_dot * u_hat + rho * (ra_rate * du/dra + dec_rate * du/ddec)

    R_obs, V_obs: observer heliocentric state, equ frame [AU, AU/day].
    Returns (r, v): (..., 3) each, heliocentric equ frame [AU, AU/day].
    """
    raise NotImplementedError("WP2")


def state_equ_to_ecl(x: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) vectors from equatorial ICRS/J2000 to ecliptic J2000."""
    raise NotImplementedError("WP2")


def state_ecl_to_equ(x: np.ndarray) -> np.ndarray:
    """Inverse of state_equ_to_ecl."""
    raise NotImplementedError("WP2")


# ----------------------------------------------------------------------------
# Two-body elements (ecliptic frame; bound orbits)
# ----------------------------------------------------------------------------


def elements_from_state(r_ecl: np.ndarray, v_ecl: np.ndarray) -> np.ndarray:
    """Heliocentric ecl state -> osculating elements (..., 6): (a, e, i, Omega, omega, M).

    Defined for BOUND orbits (a > 0, e < 1); hyperbolic/parabolic inputs return
    NaN rows (callers mask). Angles in [0, 2pi); i in [0, pi].
    """
    raise NotImplementedError("WP2")


def state_from_elements(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of elements_from_state: elements (..., 6) -> (r_ecl, v_ecl).

    WP2 acceptance: round-trip with elements_from_state to 1e-10 relative.
    """
    raise NotImplementedError("WP2")


def qei_from_elements(elements: np.ndarray) -> np.ndarray:
    """Elements -> (..., 3) array (q, e, i) with q = a(1-e) [AU, -, rad]."""
    raise NotImplementedError("WP2")


# ----------------------------------------------------------------------------
# Photometry (DESIGN §2.2)
# ----------------------------------------------------------------------------


def phase_angle(r_helio: np.ndarray, R_obs: np.ndarray) -> np.ndarray:
    """Sun-object-observer phase angle [rad] from heliocentric object position
    and heliocentric observer position (same frame for both). -> (...,)"""
    raise NotImplementedError("WP2")


def hg_phase_integral(alpha_ph: np.ndarray, G: float = G_SLOPE_DEFAULT) -> np.ndarray:
    """IAU H,G phase function Phi(alpha; G) in [0, 1] (Bowell et al. 1989)."""
    raise NotImplementedError("WP2")


def absolute_magnitude(
    v_mag: np.ndarray,
    rho: np.ndarray,
    r_helio: np.ndarray,
    alpha_ph: np.ndarray,
    G: float = G_SLOPE_DEFAULT,
) -> np.ndarray:
    """H = V - 5 log10(rho * r) + 2.5 log10 Phi(alpha; G). r_helio = |r| [AU]."""
    raise NotImplementedError("WP2")


def apparent_magnitude(
    H: np.ndarray,
    rho: np.ndarray,
    r_helio: np.ndarray,
    alpha_ph: np.ndarray,
    G: float = G_SLOPE_DEFAULT,
) -> np.ndarray:
    """Inverse of absolute_magnitude."""
    raise NotImplementedError("WP2")


# ----------------------------------------------------------------------------
# Jacobians (DESIGN §2.3)
# ----------------------------------------------------------------------------


def jacobian_state_wrt_observables(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_rate: np.ndarray,
    dec_rate: np.ndarray,
    rho: np.ndarray,
    rho_dot: np.ndarray,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
) -> np.ndarray:
    """Analytic d(r, v)/d(ra, dec, ra_rate, dec_rate, rho, rho_dot), equ frame.

    -> (..., 6, 6); row order (r, v), column order as in the signature.
    """
    raise NotImplementedError("WP2")


def jacobian_elements_wrt_state(r_ecl: np.ndarray, v_ecl: np.ndarray) -> np.ndarray:
    """d(elements)/d(r_ecl, v_ecl) -> (..., 6, 6). Analytic or high-accuracy
    numerical differentiation; WP2 acceptance requires central-FD agreement to
    1e-6 relative away from degeneracies (e -> 0, i -> 0, e -> 1)."""
    raise NotImplementedError("WP2")


def ranging_jacobian_det(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_rate: np.ndarray,
    dec_rate: np.ndarray,
    rho: np.ndarray,
    rho_dot: np.ndarray,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
    parameterization: str = "aei",
) -> np.ndarray:
    """|det d(elements)/d(A, rho, rho_dot)| composed per DESIGN §2.3 -> (...,).

    Includes the frame rotation (orthogonal; det 1) and, for
    parameterization="qei", the (a,e)->(q,e) factor |d(q,e,i,...)/d(a,e,i,...)|
    = (1 - e) so densities expressed in (q, e, i) are consumed consistently.
    """
    raise NotImplementedError("WP2")


# ----------------------------------------------------------------------------
# Forward generator (inverse problem; used by WP6 test-set generation)
# ----------------------------------------------------------------------------


def observables_from_elements(
    elements: np.ndarray,
    H: np.ndarray,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
    G: float = G_SLOPE_DEFAULT,
) -> dict[str, np.ndarray]:
    """Elements AT THE OBSERVATION EPOCH + H + observer state -> observables.

    No light-time correction in v0 (generator and classifier share the
    convention, so self-consistency tests are exact).

    Returns dict with keys: "ra", "dec", "ra_rate", "dec_rate", "v_mag",
    "rho", "rho_dot" (the last two are the ground truth for collapse tests).
    WP2 acceptance: composing with topocentric_to_heliocentric +
    elements_from_state recovers the input elements (identity on (rho,
    rho_dot)).
    """
    raise NotImplementedError("WP2")
