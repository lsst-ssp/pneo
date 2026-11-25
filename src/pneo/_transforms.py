"""WP2 — Geometry, photometry, element transforms, and Jacobians.

Frozen contract (WP0). Implementations must preserve every public signature,
unit, and frame documented here. See docs/design/DESIGN.md §2.1–2.3, docs/design/IMPLEMENTATION.md WP2.

Batching convention: all functions accept either scalars/(3,) vectors or
batched arrays with a LEADING batch dimension (shape (N,) / (N, 3) / (N, 6));
outputs follow. Frames: "equ" = equatorial ICRS/J2000, "ecl" = ecliptic J2000.
The equatorial->ecliptic rotation is applied exactly once in the pipeline, in
`state_equ_to_ecl` (IMPLEMENTATION §1).

Element order everywhere: (a, e, i, Omega, omega, M) — ecliptic J2000, rad, AU.
"""

from __future__ import annotations

import numpy as np

from ._constants import G_SLOPE_DEFAULT, MU_SUN, OBLIQUITY_J2000

__all__ = [
    "unit_vector",
    "tangent_basis",
    "topocentric_to_heliocentric",
    "state_equ_to_ecl",
    "state_ecl_to_equ",
    "elements_from_state",
    "state_from_elements",
    "qei_from_elements",
    "phase_angle",
    "hg_phase_integral",
    "absolute_magnitude",
    "apparent_magnitude",
    "jacobian_state_wrt_observables",
    "jacobian_state_wrt_elements",
    "jacobian_elements_wrt_state",
    "ranging_jacobian_det",
    "observables_from_elements",
]

# ----------------------------------------------------------------------------
# Internal tolerances (degeneracy handling).  These are *convention* switches,
# not accuracy knobs: they only decide which of two conventionally-equivalent
# angle sets is returned when the true one is undefined.  They are deliberately
# at the numerical-noise level so that a genuinely (if barely) eccentric or
# inclined orbit is still described by its own node/periapsis direction.
# ----------------------------------------------------------------------------
_ECC_DEGENERATE = 1e-14  # |e| below this -> periapsis direction undefined
_INC_DEGENERATE = 1e-14  # |sin i| below this -> node direction undefined

# Conditioning ceiling for inverting d(state)/d(elements) in
# `jacobian_elements_wrt_state`.  Above it the inverse is returned as NaN
# rather than as a plausible-looking but wrong matrix.
#
# Justification (measured, WP2 review): the relative error of |det J^-1|
# against the exact Delaunay value 2/(mu^1.5 sqrt(a) e sin i) tracks
# eps * cond_inf almost exactly --
#     cond_inf   1e6    1e8    1e10   1e12   1e13   1e16
#     rel. err   2e-13  2e-11  2e-9   5e-7   1e-6   4e-3
# so 1e12 is the largest conditioning at which the result still meets the WP2
# accuracy bar of 1e-6.  It also sits >3 decades above the worst conditioning
# seen anywhere in the WP2 sweeps (max 6.0e8 over 20 000 orbits with
# a in [0.5, 50], e in [0, 0.95], isotropic i; 1.3e7 for TNOs), so no
# physically interesting orbit is masked: reaching it needs e or sin(i) below
# ~1e-9.
_COND_MAX = 1e12

# Two-term IAU H,G phase function coefficients (Bowell et al. 1989, Asteroids II,
# "Application of photometric models to asteroids", eqs. A4-A5).
_HG_A = (3.33, 1.87)
_HG_B = (0.63, 1.22)


# ----------------------------------------------------------------------------
# Sky geometry (equatorial frame)
# ----------------------------------------------------------------------------


def unit_vector(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Topocentric line-of-sight unit vector u_hat(ra, dec), equ frame. -> (..., 3)

    `ra` and `dec` are broadcast against each other first, so mixed
    scalar/array calls (e.g. a whole declination row at fixed ra) follow the
    module's batching convention instead of raising from np.stack.
    """
    ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def tangent_basis(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u_hat, du/dra, du/ddec), each (..., 3), equ frame.

    du/dra is the RAW partial (no cos(dec) factor), consistent with
    Attributable.ra_rate being d(ra)/dt. Implemented in WP0 and shared by
    WP2/WP3; do not reimplement.

    `ra` and `dec` are broadcast against each other first: without it a mixed scalar/array call such as
    tangent_basis(ra=(5,), dec=scalar) raised from np.stack while the
    transposed call succeeded, and the asymmetry leaked into
    `topocentric_to_heliocentric` / `ranging_jacobian_det` while
    `jacobian_state_wrt_observables` (which pre-broadcasts) disagreed — the two
    halves of one composition must accept the same inputs.
    """
    ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    sa, ca = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    u = np.stack([cd * ca, cd * sa, sd], axis=-1)
    du_dra = np.stack([-cd * sa, cd * ca, np.zeros_like(sd)], axis=-1)
    du_ddec = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
    return u, du_dra, du_ddec


def _tangent_basis_second(ra, dec):
    """Second derivatives of u_hat: (d2u/dra2, d2u/(dra ddec), d2u/ddec2).

    With u = (cos(dec) cos(ra), cos(dec) sin(ra), sin(dec)):

        d2u/dra^2      = (-cos(dec) cos(ra), -cos(dec) sin(ra), 0)
        d2u/dra ddec   = ( sin(dec) sin(ra), -sin(dec) cos(ra), 0)
        d2u/ddec^2     = (-cos(dec) cos(ra), -cos(dec) sin(ra), -sin(dec)) = -u

    Each returned array has shape (..., 3), equ frame.  `ra` and `dec` are
    broadcast against each other first, as in `tangent_basis`.
    """
    ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    sa, ca = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    z = np.zeros_like(sd * ca)
    d2u_dra2 = np.stack([-cd * ca, -cd * sa, z], axis=-1)
    d2u_drad = np.stack([sd * sa, -sd * ca, z], axis=-1)
    d2u_ddec2 = np.stack([-cd * ca, -cd * sa, -sd], axis=-1)
    return d2u_dra2, d2u_drad, d2u_ddec2


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
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    ra_rate = np.asarray(ra_rate, dtype=float)
    dec_rate = np.asarray(dec_rate, dtype=float)
    rho = np.asarray(rho, dtype=float)
    rho_dot = np.asarray(rho_dot, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)

    u, u_a, u_d = tangent_basis(ra, dec)

    r = R_obs + rho[..., None] * u
    v = (
        V_obs
        + rho_dot[..., None] * u
        + rho[..., None] * (ra_rate[..., None] * u_a + dec_rate[..., None] * u_d)
    )
    return r, v


# ----------------------------------------------------------------------------
# Equatorial <-> ecliptic rotation
# ----------------------------------------------------------------------------

_CE = np.cos(OBLIQUITY_J2000)
_SE = np.sin(OBLIQUITY_J2000)

# Rotation of the *coordinate axes* by +eps about the common x-axis.
# Equivalently: R_equ_to_ecl = Rx(+eps) acting on components, where
#     Rx(eps) = [[1, 0, 0], [0, cos eps, sin eps], [0, -sin eps, cos eps]].
# Sanity: the north ECLIPTIC pole has equatorial components (0, -sin eps,
# cos eps); applying the matrix gives (0, 0, 1) exactly, as required.
_R_EQU_TO_ECL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, _CE, _SE],
        [0.0, -_SE, _CE],
    ]
)
_R_ECL_TO_EQU = _R_EQU_TO_ECL.T  # orthogonal, so the transpose is the inverse


def _rotate(x, R):
    x = np.asarray(x, dtype=float)
    if x.shape[-1] == 3:
        return x @ R.T
    if x.shape[-1] == 6:
        out = np.empty_like(x)
        out[..., 0:3] = x[..., 0:3] @ R.T
        out[..., 3:6] = x[..., 3:6] @ R.T
        return out
    raise ValueError("last axis must have length 3 (vector) or 6 (state)")


def state_equ_to_ecl(x: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) vectors from equatorial ICRS/J2000 to ecliptic J2000.

    Sign convention (documented once, applied everywhere): the ecliptic frame is
    obtained from the equatorial frame by rotating the axes about the common
    x-axis (the equinox direction) by +OBLIQUITY_J2000, i.e. the components
    transform as

        x_ecl = x_equ
        y_ecl =  cos(eps) * y_equ + sin(eps) * z_equ
        z_ecl = -sin(eps) * y_equ + cos(eps) * z_equ

    with eps = OBLIQUITY_J2000 (constants.py; IAU 1976 mean obliquity,
    84381.448 arcsec).  Check: the north ecliptic pole, whose equatorial
    components are (0, -sin eps, cos eps), maps to (0, 0, 1).

    As a convenience the function also accepts (..., 6) state vectors
    (position stacked with velocity) and rotates both 3-blocks; the contracted
    (..., 3) behaviour is unchanged.
    """
    return _rotate(x, _R_EQU_TO_ECL)


def state_ecl_to_equ(x: np.ndarray) -> np.ndarray:
    """Inverse of state_equ_to_ecl.

    The rotation is orthogonal, so this is the transpose of the matrix in
    `state_equ_to_ecl` (a rotation of the axes by -OBLIQUITY_J2000 about x).
    Accepts (..., 3) vectors or (..., 6) states.
    """
    return _rotate(x, _R_ECL_TO_EQU)


# ----------------------------------------------------------------------------
# Two-body elements (ecliptic frame; bound orbits)
# ----------------------------------------------------------------------------


def _norm(x):
    return np.sqrt(np.sum(x * x, axis=-1))


def _wrap2pi(x):
    """Wrap into [0, 2pi).  np.mod can round up to exactly 2pi for tiny
    negative inputs, so the upper end is folded back to 0."""
    y = np.mod(x, 2.0 * np.pi)
    return np.where(y >= 2.0 * np.pi, 0.0, y)


def elements_from_state(r_ecl: np.ndarray, v_ecl: np.ndarray) -> np.ndarray:
    """Heliocentric ecl state -> osculating elements (..., 6): (a, e, i, Omega, omega, M).

    Defined for BOUND orbits (a > 0, e < 1); hyperbolic/parabolic inputs return
    NaN rows (callers mask). Angles in [0, 2pi); i in [0, pi].

    Units: r [AU], v [AU/day], mu = MU_SUN [AU^3/day^2].

    Conventions and degenerate loci
    -------------------------------
    * ``Omega`` is measured in the ecliptic from the +x axis to the ascending
      node; ``omega`` is measured **in the orbital plane, from the ascending
      node to periapsis, in the direction of motion**; ``M`` is the mean anomaly
      **measured from periapsis**.
    * Near-equatorial orbits (sin i -> 0): the line of nodes is undefined.  For
      |sin i| < 1e-14 the convention adopted here is ``Omega = 0`` with the
      reference in-plane direction taken to be the +x axis, so that ``omega``
      becomes the longitude of periapsis measured from +x (in the direction of
      motion; for a retrograde equatorial orbit this runs clockwise as seen from
      +z, the standard convention).
    * Near-circular orbits (e -> 0): periapsis is undefined.  For e < 1e-14 the
      convention is ``omega = 0``, so ``M`` degenerates into the argument of
      latitude (angle from the node/reference direction to the object).
    * Both degenerate: ``Omega = omega = 0`` and ``M`` is the true longitude.
    * Only the *sum* of the affected angles is well conditioned near these loci;
      the individual angles (and hence ``jacobian_elements_wrt_state``) blow up
      like 1/e and 1/sin(i) respectively.  Callers that need derivatives must
      stay away from them (see `jacobian_elements_wrt_state`).
    """
    r_vec = np.asarray(r_ecl, dtype=float)
    v_vec = np.asarray(v_ecl, dtype=float)
    r_vec, v_vec = np.broadcast_arrays(r_vec, v_vec)
    mu = MU_SUN

    r = _norm(r_vec)
    v2 = np.sum(v_vec * v_vec, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_a = 2.0 / r - v2 / mu
        a = 1.0 / inv_a

    h_vec = np.cross(r_vec, v_vec)
    h = _norm(h_vec)

    with np.errstate(divide="ignore", invalid="ignore"):
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r[..., None]
    e = _norm(e_vec)

    with np.errstate(divide="ignore", invalid="ignore"):
        h_hat = h_vec / h[..., None]

    # Inclination from the (clipped) z-component of h_hat.
    i = np.arccos(np.clip(h_hat[..., 2], -1.0, 1.0))

    # Node vector n = z_hat x h  =  (-h_y, h_x, 0); |n| = h sin(i).
    n_vec = np.stack([-h_vec[..., 1], h_vec[..., 0], np.zeros_like(h)], axis=-1)
    n = _norm(n_vec)

    equatorial = n <= _INC_DEGENERATE * np.maximum(h, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        node_hat = np.where(
            equatorial[..., None],
            np.broadcast_to(np.array([1.0, 0.0, 0.0]), n_vec.shape),
            n_vec / np.where(n[..., None] == 0.0, 1.0, n[..., None]),
        )
    Omega = np.where(equatorial, 0.0, np.arctan2(h_vec[..., 0], -h_vec[..., 1]))

    # Periapsis direction; falls back to the node/reference direction when
    # the orbit is (numerically) circular.
    circular = e <= _ECC_DEGENERATE
    with np.errstate(divide="ignore", invalid="ignore"):
        peri_hat = np.where(
            circular[..., None],
            node_hat,
            e_vec / np.where(e[..., None] == 0.0, 1.0, e[..., None]),
        )

    # omega: angle node_hat -> peri_hat about h_hat (positive in the direction
    # of motion).  h_hat x node_hat is the in-plane direction 90 deg ahead of
    # the node.
    ahead_node = np.cross(h_hat, node_hat)
    omega = np.where(
        circular,
        0.0,
        np.arctan2(
            np.sum(ahead_node * peri_hat, axis=-1),
            np.sum(node_hat * peri_hat, axis=-1),
        ),
    )

    # True anomaly: angle peri_hat -> r about h_hat.
    ahead_peri = np.cross(h_hat, peri_hat)
    nu = np.arctan2(
        np.sum(ahead_peri * r_vec, axis=-1),
        np.sum(peri_hat * r_vec, axis=-1),
    )

    # Eccentric -> mean anomaly.  The half-angle form is accurate for all e < 1.
    with np.errstate(invalid="ignore"):
        E = 2.0 * np.arctan2(
            np.sqrt(np.clip(1.0 - e, 0.0, None)) * np.sin(0.5 * nu),
            np.sqrt(np.clip(1.0 + e, 0.0, None)) * np.cos(0.5 * nu),
        )
        M = E - e * np.sin(E)

    out = np.stack(
        [a, e, i, _wrap2pi(Omega), _wrap2pi(omega), _wrap2pi(M)], axis=-1
    )

    with np.errstate(invalid="ignore"):
        bound = np.isfinite(a) & (a > 0.0) & (e < 1.0) & np.isfinite(e) & (h > 0.0)
    out = np.where(bound[..., None], out, np.nan)
    return out


def _kepler_E(M, e, maxiter: int = 100, tol: float = 1e-15):
    """Solve M = E - e sin(E) for E, vectorized, bisection-safeguarded Newton.

    Returns E in the branch containing M reduced to [-pi, pi).  Guaranteed to
    converge for 0 <= e < 1: f(E) = E - e sin E - M is strictly increasing and
    the root is bracketed by [M - e, M + e].
    """
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    M, e = np.broadcast_arrays(M, e)
    bad = ~(np.isfinite(M) & np.isfinite(e)) | (e < 0.0) | (e >= 1.0)
    Ms = np.where(bad, 0.0, M)
    es = np.where(bad, 0.0, e)

    Ms = np.mod(Ms + np.pi, 2.0 * np.pi) - np.pi
    lo = Ms - es
    hi = Ms + es
    E = np.clip(Ms + es * np.sin(Ms), lo, hi)

    for _ in range(maxiter):
        f = E - es * np.sin(E) - Ms
        lo = np.where(f < 0.0, E, lo)
        hi = np.where(f > 0.0, E, hi)
        fp = 1.0 - es * np.cos(E)
        E_new = E - f / fp
        outside = ~((E_new > lo) & (E_new < hi))
        E_new = np.where(outside, 0.5 * (lo + hi), E_new)
        step = np.abs(E_new - E)
        E = E_new
        if np.all(step <= tol):
            break

    return np.where(bad, np.nan, E)


def _pqw_basis(i, Omega, omega):
    """Perifocal basis vectors (P, W, h_hat) in the ecliptic frame.

    P points to periapsis, W is 90 deg ahead of P in the direction of motion,
    and h_hat = P x W is the orbit normal.  Columns of
    Rz(Omega) Rx(i) Rz(omega).
    """
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(omega), np.sin(omega)

    P = np.stack(
        [cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1
    )
    W = np.stack(
        [-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1
    )
    h_hat = np.stack([sO * si, -cO * si, ci], axis=-1)
    return P, W, h_hat


def _elements_split(elements):
    el = np.asarray(elements, dtype=float)
    if el.shape[-1] != 6:
        raise ValueError("elements must have shape (..., 6)")
    return el


def state_from_elements(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of elements_from_state: elements (..., 6) -> (r_ecl, v_ecl).

    Elements are (a, e, i, Omega, omega, M), ecliptic J2000, AU/rad.
    Bound orbits only: rows with a <= 0, e < 0 or e >= 1 (or any non-finite
    entry) return NaN position and velocity.

    WP2 acceptance: round-trip with elements_from_state to 1e-10 relative.
    """
    el = _elements_split(elements)
    a, e, i, Om, om, M = (el[..., k] for k in range(6))
    mu = MU_SUN

    with np.errstate(invalid="ignore"):
        good = (
            np.all(np.isfinite(el), axis=-1)
            & (a > 0.0)
            & (e >= 0.0)
            & (e < 1.0)
        )

    a_s = np.where(good, a, 1.0)
    e_s = np.where(good, e, 0.0)

    E = _kepler_E(np.where(good, M, 0.0), e_s)
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(np.clip(1.0 - e_s * e_s, 0.0, None))

    X = a_s * (cE - e_s)
    Y = a_s * b * sE
    r = a_s * (1.0 - e_s * cE)
    s = np.sqrt(mu * a_s)
    Xd = -s * sE / r
    Yd = s * b * cE / r

    P, W, _ = _pqw_basis(np.where(good, i, 0.0), np.where(good, Om, 0.0), np.where(good, om, 0.0))

    r_vec = X[..., None] * P + Y[..., None] * W
    v_vec = Xd[..., None] * P + Yd[..., None] * W

    r_vec = np.where(good[..., None], r_vec, np.nan)
    v_vec = np.where(good[..., None], v_vec, np.nan)
    return r_vec, v_vec


def qei_from_elements(elements: np.ndarray) -> np.ndarray:
    """Elements -> (..., 3) array (q, e, i) with q = a(1-e) [AU, -, rad]."""
    el = _elements_split(elements)
    a, e, i = el[..., 0], el[..., 1], el[..., 2]
    return np.stack([a * (1.0 - e), e, i], axis=-1)


# ----------------------------------------------------------------------------
# Photometry (DESIGN §2.2)
# ----------------------------------------------------------------------------


def phase_angle(r_helio: np.ndarray, R_obs: np.ndarray) -> np.ndarray:
    """Sun-object-observer phase angle [rad] from heliocentric object position
    and heliocentric observer position (same frame for both). -> (...,)

    alpha = angle at the object between the direction to the Sun (-r) and the
    direction to the observer (R_obs - r).  Computed with atan2(|cross|, dot)
    for accuracy at both small and large angles.  Result in [0, pi].
    """
    r_vec = np.asarray(r_helio, dtype=float)
    R_vec = np.asarray(R_obs, dtype=float)
    r_vec, R_vec = np.broadcast_arrays(r_vec, R_vec)

    to_sun = -r_vec
    to_obs = R_vec - r_vec
    cross = np.cross(to_sun, to_obs)
    with np.errstate(invalid="ignore"):
        return np.arctan2(_norm(cross), np.sum(to_sun * to_obs, axis=-1))


def hg_phase_integral(alpha_ph: np.ndarray, G: float = G_SLOPE_DEFAULT) -> np.ndarray:
    """IAU H,G phase function Phi(alpha; G) in [0, 1] (Bowell et al. 1989).

    Two-term form:

        Phi_i(alpha) = exp(-A_i * tan(alpha/2)**B_i),
        (A1, B1) = (3.33, 0.63),  (A2, B2) = (1.87, 1.22)
        Phi(alpha; G) = (1 - G) * Phi_1 + G * Phi_2

    Phi(0) = 1 exactly; Phi -> 0 as alpha -> pi.  `alpha_ph` [rad] is clipped
    to [0, pi] (values outside arise only from round-off in the caller).
    Vectorized over alpha; G is a scalar.

    Caveat (v0 scope).  This is the two-term exponential form ONLY.  The full
    Bowell et al. prescription additionally blends in a spline/linear branch at
    large phase; without it this form UNDERPREDICTS Phi for alpha greater than
    roughly 120 deg (e.g. Phi = 4.1e-4 at 150 deg with G = 0.15, i.e. ~8.4 mag
    of darkening).  Consequence for the pipeline: at such phase angles — only
    reachable at very small rho, since the phase angle is bounded by the
    observer-Sun-object geometry — the implied H from
    `absolute_magnitude` comes out too BRIGHT (too small).  That is the
    conservative direction for NEO triage and the affected corner of the
    admissible region is negligible, so v0 accepts it deliberately.
    """
    alpha = np.asarray(alpha_ph, dtype=float)
    a = np.clip(alpha, 0.0, np.pi)
    t = np.tan(0.5 * a)  # 0 at alpha=0, +inf at alpha=pi
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi1 = np.exp(-_HG_A[0] * np.power(t, _HG_B[0]))
        phi2 = np.exp(-_HG_A[1] * np.power(t, _HG_B[1]))
    # tan(pi/2) overflows to +inf -> power -> inf -> exp(-inf) = 0.  Guard the
    # exact-pi case explicitly so no NaN leaks through.
    phi1 = np.where(np.isfinite(phi1), phi1, 0.0)
    phi2 = np.where(np.isfinite(phi2), phi2, 0.0)
    out = (1.0 - G) * phi1 + G * phi2
    return np.where(np.isnan(alpha), np.nan, out)


def _check_r_helio(v_mag, rho, alpha_ph, r_helio):
    """Reject a heliocentric POSITION VECTOR passed as the distance `r_helio`.

    `r_helio` is |r| [AU], a magnitude.  Passing the (..., 3) position vector
    instead used to broadcast silently and return a plausible-looking 3-array
    of magnitudes (WP2 review finding).  The guard: `r_helio` may not introduce
    an axis that none of the other (scalar-valued) arguments has, which catches
    both the (3,) single-object case and the (N, 3) batched case.
    """
    others = max(np.ndim(v_mag), np.ndim(rho), np.ndim(alpha_ph))
    if np.ndim(r_helio) > others:
        raise ValueError(
            "r_helio must be the heliocentric DISTANCE |r| [AU], not the position "
            f"vector: got ndim {np.ndim(r_helio)} (shape {np.shape(r_helio)}) against "
            f"ndim {others} for (v_mag/H, rho, alpha_ph). Pass "
            "np.linalg.norm(r_vec, axis=-1)."
        )


def absolute_magnitude(
    v_mag: np.ndarray,
    rho: np.ndarray,
    r_helio: np.ndarray,
    alpha_ph: np.ndarray,
    G: float = G_SLOPE_DEFAULT,
) -> np.ndarray:
    """H = V - 5 log10(rho * r) + 2.5 log10 Phi(alpha; G). r_helio = |r| [AU].

    `rho` is the topocentric distance [AU], `r_helio` the heliocentric distance
    [AU] (a SCALAR magnitude, not a vector), `alpha_ph` the phase angle [rad].
    Exact inverse of `apparent_magnitude` with the same arguments.

    Raises ValueError if `r_helio` adds a dimension the other arguments lack —
    i.e. if a position vector was passed where a distance was expected.
    """
    _check_r_helio(v_mag, rho, alpha_ph, r_helio)
    v_mag = np.asarray(v_mag, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r_helio = np.asarray(r_helio, dtype=float)
    phi = hg_phase_integral(alpha_ph, G)
    with np.errstate(divide="ignore", invalid="ignore"):
        return v_mag - 5.0 * np.log10(rho * r_helio) + 2.5 * np.log10(phi)


def apparent_magnitude(
    H: np.ndarray,
    rho: np.ndarray,
    r_helio: np.ndarray,
    alpha_ph: np.ndarray,
    G: float = G_SLOPE_DEFAULT,
) -> np.ndarray:
    """Inverse of absolute_magnitude.

    V = H + 5 log10(rho * r) - 2.5 log10 Phi(alpha; G).

    Raises ValueError if `r_helio` adds a dimension the other arguments lack —
    i.e. if a position vector was passed where a distance was expected.
    """
    _check_r_helio(H, rho, alpha_ph, r_helio)
    H = np.asarray(H, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r_helio = np.asarray(r_helio, dtype=float)
    phi = hg_phase_integral(alpha_ph, G)
    with np.errstate(divide="ignore", invalid="ignore"):
        return H + 5.0 * np.log10(rho * r_helio) - 2.5 * np.log10(phi)


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

    Derivation (DESIGN §2.1).  With u = u_hat(ra, dec), u_a = du/dra,
    u_d = du/ddec, and the second derivatives u_aa, u_ad, u_dd from
    `_tangent_basis_second`:

        r = R_obs + rho u
        v = V_obs + rho_dot u + rho (ra_rate u_a + dec_rate u_d)

        dr/dra      = rho u_a
        dr/ddec     = rho u_d
        dr/dra_rate = 0
        dr/ddec_rate= 0
        dr/drho     = u
        dr/drho_dot = 0

        dv/dra      = rho_dot u_a + rho (ra_rate u_aa + dec_rate u_ad)
        dv/ddec     = rho_dot u_d + rho (ra_rate u_ad + dec_rate u_dd)
        dv/dra_rate = rho u_a
        dv/ddec_rate= rho u_d
        dv/drho     = ra_rate u_a + dec_rate u_d
        dv/drho_dot = u

    (u_ad = d(u_a)/ddec = d(u_d)/dra, mixed partials commute.)  R_obs and V_obs
    are additive constants of the map and therefore do not appear; they are
    accepted for signature symmetry and to participate in broadcasting.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    ra_rate = np.asarray(ra_rate, dtype=float)
    dec_rate = np.asarray(dec_rate, dtype=float)
    rho = np.asarray(rho, dtype=float)
    rho_dot = np.asarray(rho_dot, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)

    batch = np.broadcast_shapes(
        ra.shape,
        dec.shape,
        ra_rate.shape,
        dec_rate.shape,
        rho.shape,
        rho_dot.shape,
        R_obs.shape[:-1],
        V_obs.shape[:-1],
    )
    ra = np.broadcast_to(ra, batch)
    dec = np.broadcast_to(dec, batch)
    ra_rate = np.broadcast_to(ra_rate, batch)
    dec_rate = np.broadcast_to(dec_rate, batch)
    rho = np.broadcast_to(rho, batch)
    rho_dot = np.broadcast_to(rho_dot, batch)

    u, u_a, u_d = tangent_basis(ra, dec)
    u_aa, u_ad, u_dd = _tangent_basis_second(ra, dec)

    zero = np.zeros(batch + (3,))
    rho_c = rho[..., None]
    rd_c = rho_dot[..., None]
    ar_c = ra_rate[..., None]
    dr_c = dec_rate[..., None]

    dr = [
        rho_c * u_a,  # d/dra
        rho_c * u_d,  # d/ddec
        zero,  # d/dra_rate
        zero,  # d/ddec_rate
        u,  # d/drho
        zero,  # d/drho_dot
    ]
    dv = [
        rd_c * u_a + rho_c * (ar_c * u_aa + dr_c * u_ad),  # d/dra
        rd_c * u_d + rho_c * (ar_c * u_ad + dr_c * u_dd),  # d/ddec
        rho_c * u_a,  # d/dra_rate
        rho_c * u_d,  # d/ddec_rate
        ar_c * u_a + dr_c * u_d,  # d/drho
        u,  # d/drho_dot
    ]

    J = np.empty(batch + (6, 6))
    for k in range(6):
        J[..., 0:3, k] = dr[k]
        J[..., 3:6, k] = dv[k]
    return J


def jacobian_state_wrt_elements(elements: np.ndarray) -> np.ndarray:
    """Analytic d(r_ecl, v_ecl)/d(a, e, i, Omega, omega, M) -> (..., 6, 6).

    Helper for `jacobian_elements_wrt_state` (which returns its inverse) and
    useful on its own for the forward generator's error propagation.

    Derivation.  Write the state in the perifocal basis (P, W) of
    `_pqw_basis`:

        r_vec = X P + Y W,      X = a (cos E - e),   Y = a b sin E
        v_vec = Xd P + Yd W,    Xd = -s sin E / r,   Yd = s b cos E / r

    with b = sqrt(1 - e^2), r = a (1 - e cos E), s = sqrt(mu a), and E(e, M)
    given by Kepler's equation, whence

        dE/dM = a / r,      dE/de = a sin E / r,     dE/da = 0
        dr/dM = a^2 e sin E / r
        dr/de = a (-cos E + a e sin^2 E / r)
        dr/da = r / a
        ds/da = s / (2a).

    (P, W) depend only on the three angles, and the rotation generators give

        dP/dOmega = z_hat x P,   dW/dOmega = z_hat x W
        dP/di     = n_hat x P,   dW/di     = n_hat x W,  n_hat = (cos Om, sin Om, 0)
        dP/domega = W,           dW/domega = -P

    so d(r_vec)/dOmega = z_hat x r_vec and d(r_vec)/domega = X W - Y P
    = h_hat x r_vec (the form used in the code), and likewise for v_vec.

    Rows are (r, v); columns are (a, e, i, Omega, omega, M).  Non-bound or
    non-finite element rows give NaN.
    """
    el = _elements_split(elements)
    a, e, i, Om, om, M = (el[..., k] for k in range(6))
    mu = MU_SUN

    with np.errstate(invalid="ignore"):
        good = np.all(np.isfinite(el), axis=-1) & (a > 0.0) & (e >= 0.0) & (e < 1.0)

    a_s = np.where(good, a, 1.0)
    e_s = np.where(good, e, 0.0)
    i_s = np.where(good, i, 0.0)
    Om_s = np.where(good, Om, 0.0)
    om_s = np.where(good, om, 0.0)

    E = _kepler_E(np.where(good, M, 0.0), e_s)
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(np.clip(1.0 - e_s * e_s, 1e-300, None))
    r = a_s * (1.0 - e_s * cE)
    s = np.sqrt(mu * a_s)

    dE_de = a_s * sE / r
    dE_dM = a_s / r

    dr_da = r / a_s
    dr_de = a_s * (-cE + a_s * e_s * sE * sE / r)
    dr_dM = a_s * a_s * e_s * sE / r

    X = a_s * (cE - e_s)
    Y = a_s * b * sE
    Xd = -s * sE / r
    Yd = s * b * cE / r

    dX_da = cE - e_s
    dX_de = -a_s * (1.0 + a_s * sE * sE / r)
    dX_dM = -a_s * a_s * sE / r

    dY_da = b * sE
    dY_de = a_s * (-e_s * sE / b + b * cE * dE_de)
    dY_dM = a_s * b * cE * dE_dM

    dXd_da = s * sE / (2.0 * a_s * r)
    dXd_de = -(s * cE * dE_de / r - s * sE * dr_de / (r * r))
    dXd_dM = -(s * cE * dE_dM / r - s * sE * dr_dM / (r * r))

    dYd_da = -s * b * cE / (2.0 * a_s * r)
    dYd_de = s * (
        (-e_s / b) * cE / r - b * sE * dE_de / r - b * cE * dr_de / (r * r)
    )
    dYd_dM = s * (-b * sE * dE_dM / r - b * cE * dr_dM / (r * r))

    P, W, h_hat = _pqw_basis(i_s, Om_s, om_s)
    r_vec = X[..., None] * P + Y[..., None] * W
    v_vec = Xd[..., None] * P + Yd[..., None] * W

    z_hat = np.broadcast_to(np.array([0.0, 0.0, 1.0]), P.shape)
    n_hat = np.stack(
        [np.cos(Om_s), np.sin(Om_s), np.zeros_like(Om_s)], axis=-1
    )

    dP_di = np.cross(n_hat, P)
    dW_di = np.cross(n_hat, W)

    batch = el.shape[:-1]
    J = np.empty(batch + (6, 6))

    # columns 0 (a) and 1 (e): scalar coefficients times the fixed (P, W) basis
    for col, (dXp, dYp, dXdp, dYdp) in (
        (0, (dX_da, dY_da, dXd_da, dYd_da)),
        (1, (dX_de, dY_de, dXd_de, dYd_de)),
    ):
        J[..., 0:3, col] = dXp[..., None] * P + dYp[..., None] * W
        J[..., 3:6, col] = dXdp[..., None] * P + dYdp[..., None] * W

    # column 2: inclination
    J[..., 0:3, 2] = X[..., None] * dP_di + Y[..., None] * dW_di
    J[..., 3:6, 2] = Xd[..., None] * dP_di + Yd[..., None] * dW_di

    # column 3: Omega  -> rotation about z_hat
    J[..., 0:3, 3] = np.cross(z_hat, r_vec)
    J[..., 3:6, 3] = np.cross(z_hat, v_vec)

    # column 4: omega  -> rotation about h_hat (in-plane by construction)
    J[..., 0:3, 4] = np.cross(h_hat, r_vec)
    J[..., 3:6, 4] = np.cross(h_hat, v_vec)

    # column 5: M
    J[..., 0:3, 5] = dX_dM[..., None] * P + dY_dM[..., None] * W
    J[..., 3:6, 5] = dXd_dM[..., None] * P + dYd_dM[..., None] * W

    J = np.where(good[..., None, None], J, np.nan)
    return J


def jacobian_elements_wrt_state(r_ecl: np.ndarray, v_ecl: np.ndarray) -> np.ndarray:
    """d(elements)/d(r_ecl, v_ecl) -> (..., 6, 6). Analytic or high-accuracy
    numerical differentiation; WP2 acceptance requires central-FD agreement to
    1e-6 relative away from degeneracies (e -> 0, i -> 0, e -> 1).

    Implementation: fully ANALYTIC.  The closed form is obtained by
    differentiating the *forward* map `state_from_elements` (see
    `jacobian_state_wrt_elements`, whose derivation is documented there) and
    inverting the resulting 6x6:

        d(E)/d(r, v) = [ d(r, v)/d(E) ]^{-1}

    which is exact because the two maps are local inverses of one another.  No
    finite differencing is used anywhere.

    Rows are (a, e, i, Omega, omega, M); columns are (rx, ry, rz, vx, vy, vz),
    all in the ecliptic J2000 frame, AU / day / rad.

    Degenerate loci (the forward Jacobian is genuinely singular there, so the
    inverse does not exist and the returned entries blow up like 1/e or
    1/sin(i)):
      * e -> 0   : omega and M are individually undefined (only omega + M is);
                   columns 4 and 5 of the forward Jacobian coincide.
      * sin i -> 0 : Omega and omega are individually undefined; columns 3 and 4
                   of the forward Jacobian coincide.
      * e -> 1 / a -> inf : the bound-orbit parameterization degenerates.

    NaN is returned (never an exception, never a plausible-looking wrong
    matrix) when the state is unbound or non-finite, when the forward Jacobian
    is exactly singular, OR when it is merely NUMERICALLY singular -- judged by
    the infinity-norm condition estimate cond_inf(J) = ||J||_inf ||J^-1||_inf
    exceeding `_COND_MAX` = 1e12, the point beyond which the inverse no longer
    meets the WP2 1e-6 accuracy bar (see the constant's derivation).  The
    estimate costs two reductions on the already-computed inverse, so the guard
    is essentially free.
    """
    elements = elements_from_state(r_ecl, v_ecl)
    Jse = jacobian_state_wrt_elements(elements)

    finite = np.all(np.isfinite(Jse), axis=(-2, -1))
    safe = np.where(finite[..., None, None], Jse, np.eye(6))

    with np.errstate(divide="ignore", invalid="ignore"):
        sign, logdet = np.linalg.slogdet(safe)
    nonsingular = (sign != 0) & np.isfinite(logdet)
    safe = np.where(nonsingular[..., None, None], safe, np.eye(6))

    Jes = np.linalg.inv(safe)

    # Infinity-norm condition estimate (LAPACK-style): cheap, and within a
    # factor of ~n of the true 2-norm condition number -- ample for a guard
    # whose threshold sits three decades clear of any real orbit.
    with np.errstate(invalid="ignore", over="ignore"):
        norm_fwd = np.max(np.sum(np.abs(safe), axis=-1), axis=-1)
        norm_inv = np.max(np.sum(np.abs(Jes), axis=-1), axis=-1)
        well_conditioned = (norm_fwd * norm_inv) <= _COND_MAX

    ok = finite & nonsingular & well_conditioned
    return np.where(ok[..., None, None], Jes, np.nan)


def _rotation_block6():
    """Block-diagonal d(state_ecl)/d(state_equ): diag(R, R), |det| = 1."""
    B = np.zeros((6, 6))
    B[0:3, 0:3] = _R_EQU_TO_ECL
    B[3:6, 3:6] = _R_EQU_TO_ECL
    return B


_ROT6_EQU_TO_ECL = _rotation_block6()


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

    Chain (all 6x6):

        d(E)/d(A, rho, rho_dot)
          = d(E)/d(state_ecl) . d(state_ecl)/d(state_equ) . d(state_equ)/d(A, rho, rho_dot)

    The middle factor is block-diag(R, R) with R the equatorial->ecliptic
    rotation; it is orthogonal so |det| = 1, but it is applied explicitly here
    rather than dropped, so the composed matrix is the true Jacobian.

    For parameterization="qei" the reparameterization (a, e) -> (q = a(1-e), e)
    has Jacobian matrix with rows dq = ((1-e), -a, 0, 0, 0, 0) and de =
    (0, 1, 0, 0, 0, 0), the rest identity; its determinant is (1 - e), which is
    non-negative for bound orbits, so |det| picks up a factor (1 - e).

    Column order of the observable block is
    (ra, dec, ra_rate, dec_rate, rho, rho_dot).  Returns NaN where the implied
    orbit is unbound or the element map is singular.
    """
    if parameterization not in ("aei", "qei"):
        raise ValueError(f"unknown parameterization {parameterization!r}")

    r_equ, v_equ = topocentric_to_heliocentric(
        ra, dec, ra_rate, dec_rate, rho, rho_dot, R_obs, V_obs
    )
    r_ecl = state_equ_to_ecl(r_equ)
    v_ecl = state_equ_to_ecl(v_equ)

    J_obs = jacobian_state_wrt_observables(
        ra, dec, ra_rate, dec_rate, rho, rho_dot, R_obs, V_obs
    )
    J_el = jacobian_elements_wrt_state(r_ecl, v_ecl)

    composed = J_el @ _ROT6_EQU_TO_ECL @ J_obs
    with np.errstate(invalid="ignore"):
        det = np.abs(np.linalg.det(composed))

    if parameterization == "qei":
        elements = elements_from_state(r_ecl, v_ecl)
        det = det * (1.0 - elements[..., 1])

    return det


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

    Parameters
    ----------
    elements : (..., 6)
        (a, e, i, Omega, omega, M), heliocentric ecliptic J2000, AU/rad.
    H : (...)
        Absolute magnitude, Johnson V.
    R_obs, V_obs : (..., 3)
        Observer heliocentric state in the EQUATORIAL (ICRS/J2000) frame
        [AU, AU/day] — the same frame/units as `topocentric_to_heliocentric`.
    G : float
        Slope parameter of the IAU H,G phase function.

    Notes
    -----
    Rates are obtained by projecting the relative velocity onto the tangent
    basis.  With rho_vec = r - R_obs, rho = |rho_vec|, u = rho_vec / rho and
    w = v - V_obs:

        rho_dot   = u . w
        ra_rate   = (w . du/dra)  / (rho cos^2(dec))
        dec_rate  = (w . du/ddec) /  rho

    The cos^2(dec) is the norm-squared of du/dra (the tangent basis is
    orthogonal but NOT orthonormal); u . du/dra = u . du/ddec = 0 exactly, so
    subtracting the radial part of w first is unnecessary.  `ra_rate` is the
    RAW d(ra)/dt (no cos(dec) factor), matching `Attributable.ra_rate`.

    Singular where cos(dec) -> 0 (the celestial poles) and where rho -> 0
    (object at the observer).  Rows whose elements are unbound/non-finite come
    back NaN.  `ra` is wrapped to [0, 2pi); `dec` is in [-pi/2, pi/2].
    """
    el = _elements_split(elements)
    H = np.asarray(H, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)

    r_ecl, v_ecl = state_from_elements(el)
    r_equ = state_ecl_to_equ(r_ecl)
    v_equ = state_ecl_to_equ(v_ecl)

    r_equ, R_obs_b = np.broadcast_arrays(r_equ, R_obs)
    v_equ, V_obs_b = np.broadcast_arrays(v_equ, V_obs)

    rho_vec = r_equ - R_obs_b
    rho = _norm(rho_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = rho_vec / rho[..., None]

    dec = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    ra = _wrap2pi(np.arctan2(u[..., 1], u[..., 0]))

    # Recompute the tangent basis from (ra, dec) so the generator uses exactly
    # the same basis the inverse map does.
    _, u_a, u_d = tangent_basis(ra, dec)

    w = v_equ - V_obs_b
    rho_dot = np.sum(u * w, axis=-1)
    cd2 = np.cos(dec) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ra_rate = np.sum(w * u_a, axis=-1) / (rho * cd2)
        dec_rate = np.sum(w * u_d, axis=-1) / rho

    r_helio = _norm(r_equ)
    alpha_ph = phase_angle(r_equ, R_obs_b)
    v_mag = apparent_magnitude(H, rho, r_helio, alpha_ph, G)

    return {
        "ra": ra,
        "dec": dec,
        "ra_rate": ra_rate,
        "dec_rate": dec_rate,
        "v_mag": v_mag,
        "rho": rho,
        "rho_dot": rho_dot,
    }
