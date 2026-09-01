"""WP2 acceptance tests — transforms, photometry, and Jacobians.

All randomized sweeps use a FIXED seed (``SEED``) so failures are reproducible.
Sweeps deliberately span the NEO-like (a ~ 0.7-3 AU), MBA-like (2-4 AU) and
TNO-like (30-50 AU) regimes.

Degenerate loci excluded from the finite-difference comparisons, and why
--------------------------------------------------------------------------
The element map ``(r, v) -> (a, e, i, Omega, omega, M)`` is a genuine
diffeomorphism only away from a set of measure zero.  On those loci the exact
Jacobian is unbounded, so "analytic agrees with FD to 1e-6 relative" is not a
meaningful statement — both are dominated by round-off.  The masks used below:

* ``e < 0.02``            — near-circular.  Periapsis direction is undefined at
  e = 0; ``omega`` and ``M`` are individually meaningless and only ``omega + M``
  is well determined.  ``d(omega, M)/d(r, v)`` diverges like 1/e.
* ``sin(i) < 0.02``       — near-equatorial (also covers i -> pi).  The line of
  nodes is undefined; ``Omega`` and ``omega`` are individually meaningless and
  only ``Omega + omega`` is well determined.  Divergence like 1/sin(i).
* ``e > 0.95`` / unbound  — near-parabolic.  ``a = -mu/2E`` diverges as the
  energy crosses zero; excluded by construction in the sweeps (and the contract
  returns NaN for e >= 1 or a <= 0).
* ``|dec| > 85 deg``      — celestial poles.  ``ra`` and hence ``ra_rate`` are
  singular there (the tangent vector ``du/dra`` has norm cos(dec) -> 0), so the
  observable parameterization itself degenerates.
* ``rho < 0.05 AU``       — object essentially at the observer.  ``d r/d ra`` =
  rho * du/dra -> 0, so ``d(state)/d(observables)`` becomes singular like rho^4.
* ``cond(J) > 1e10``      — belt-and-braces guard for any residual
  near-degeneracy the explicit masks above do not catch (e.g. an orbit whose
  node happens to be nearly aligned with the periapsis and *both* e and sin i
  are only marginally above the cuts).  Reported, not silently dropped.

Everything else in the sweeps is required to meet the 1e-6 relative bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from digest42 import transforms as T
from digest42.constants import G_SLOPE_DEFAULT, MU_SUN, OBLIQUITY_J2000

SEED = 20260831

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _element_diff(el_a, el_b):
    """el_a - el_b with the three cyclic angles wrapped into (-pi, pi]."""
    d = np.asarray(el_a, dtype=float) - np.asarray(el_b, dtype=float)
    d = d.copy()
    d[..., 3:] = _wrap_pi(d[..., 3:])
    return d


def _fake_observer(n, rng):
    """A stand-in heliocentric observer state, EQUATORIAL frame.

    WP1's `observer_state` is not available to WP2's tests, so the observer is
    placed on a circular 1 AU ecliptic orbit at a random phase and rotated into
    the equatorial frame.  Only the *values* matter here: every function under
    test treats (R_obs, V_obs) as an arbitrary additive constant.
    """
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    R_ecl = np.stack([np.cos(th), np.sin(th), np.zeros(n)], axis=-1)
    v_circ = math.sqrt(MU_SUN / 1.0)
    V_ecl = v_circ * np.stack([-np.sin(th), np.cos(th), np.zeros(n)], axis=-1)
    return T.state_ecl_to_equ(R_ecl), T.state_ecl_to_equ(V_ecl)


def _sample_elements(n, rng, regime="mixed"):
    """Randomized bound elements covering NEO / MBA / TNO regimes."""
    if regime == "mixed":
        which = rng.integers(0, 3, n)
        a = np.where(
            which == 0,
            10 ** rng.uniform(np.log10(0.7), np.log10(3.0), n),  # NEO-like
            np.where(
                which == 1,
                rng.uniform(2.0, 5.2, n),  # MBA / Trojan-like
                rng.uniform(30.0, 50.0, n),  # TNO-like
            ),
        )
        e = np.where(which == 0, rng.uniform(0.05, 0.8, n), rng.uniform(0.02, 0.4, n))
    elif regime == "wide":
        a = 10 ** rng.uniform(np.log10(0.5), np.log10(50.0), n)
        e = rng.uniform(0.0, 0.95, n)
    else:  # pragma: no cover
        raise ValueError(regime)
    i = np.arccos(rng.uniform(-1.0, 1.0, n))  # isotropic: all inclinations
    Om = rng.uniform(0.0, 2.0 * np.pi, n)
    om = rng.uniform(0.0, 2.0 * np.pi, n)
    M = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([a, e, i, Om, om, M], axis=-1)


def _fd_jacobian(f, x, h, wrap_rows=()):
    """Fifth-order-accurate central difference Jacobian d f / d x.

    Uses the 5-point stencil (f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h)) / (12 h),
    whose truncation error is O(h^4).  Differences are taken relative to f(x)
    before combining (the stencil weights sum to zero, so this is exact) which
    lets rows listed in `wrap_rows` be unwrapped across the 0/2pi branch cut.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    n = x.size
    J = np.empty((f0.size, n))
    for j in range(n):

        def ev(step, j=j):
            xx = x.copy()
            xx[j] += step * h[j]
            d = np.asarray(f(xx), dtype=float) - f0
            for k in wrap_rows:
                d[k] = _wrap_pi(d[k])
            return d

        J[:, j] = (ev(-2.0) - 8.0 * ev(-1.0) + 8.0 * ev(1.0) - ev(2.0)) / (12.0 * h[j])
    return J


def _rowwise_relerr(J_analytic, J_fd):
    """max_ij |dJ_ij| / max_k |J_analytic_ik|.

    Each row of a Jacobian is one output's gradient; its natural scale is the
    row's largest entry, which is also what sets the finite-difference
    round-off floor.  Entry-wise relative error would be meaningless for the
    (many) structurally tiny entries.
    """
    scale = np.max(np.abs(J_analytic), axis=-1, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    return float(np.max(np.abs(J_analytic - J_fd) / scale))


# --------------------------------------------------------------------------
# 5. Frame rotation
# --------------------------------------------------------------------------


def test_rotation_orthogonality_and_inverse():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(500, 3))

    y = T.state_equ_to_ecl(x)
    # Orthogonal: norms and pairwise dot products preserved.
    assert np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=0, atol=1e-15)
    assert np.allclose(np.sum(y[:-1] * y[1:], axis=-1), np.sum(x[:-1] * x[1:], axis=-1), atol=1e-14)

    # Exact inverse, both directions.
    assert np.allclose(T.state_ecl_to_equ(y), x, rtol=0, atol=1e-15)
    assert np.allclose(T.state_equ_to_ecl(T.state_ecl_to_equ(x)), x, rtol=0, atol=1e-15)

    # Matrix form is a proper rotation (det = +1), not a reflection.
    basis = np.eye(3)
    Rm = T.state_equ_to_ecl(basis).T  # columns = images of e_x, e_y, e_z
    assert np.linalg.det(Rm) == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-15)

    # The x axis (the equinox) is the rotation axis and is left untouched.
    assert np.allclose(T.state_equ_to_ecl(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0], atol=1e-16)

    # (..., 6) convenience form rotates both 3-blocks identically.
    st = rng.normal(size=(7, 6))
    st_ecl = T.state_equ_to_ecl(st)
    assert np.allclose(st_ecl[..., 0:3], T.state_equ_to_ecl(st[..., 0:3]), atol=0)
    assert np.allclose(st_ecl[..., 3:6], T.state_equ_to_ecl(st[..., 3:6]), atol=0)


def test_rotation_hardcoded_reference():
    """Independent hand-computed rotation of two vectors.

    Sign convention: components transform as
        x_ecl = x_equ
        y_ecl =  cos(eps) y_equ + sin(eps) z_equ
        z_ecl = -sin(eps) y_equ + cos(eps) z_equ
    with eps = 84381.448 arcsec (IAU 1976 mean obliquity, constants.py), for
    which the standard published values are
        sin(eps) = 0.3977771559319137,  cos(eps) = 0.9174820620691818.
    These are the classic J2000 obliquity rotation-matrix entries (e.g. the
    Astronomical Almanac / IAU 1976 precession constants), used here as an
    external reference rather than re-derived from constants.py.
    """
    SIN_EPS = 0.3977771559319137
    COS_EPS = 0.9174820620691818
    assert math.sin(OBLIQUITY_J2000) == pytest.approx(SIN_EPS, abs=5e-16)
    assert math.cos(OBLIQUITY_J2000) == pytest.approx(COS_EPS, abs=5e-16)

    # North celestial pole (0, 0, 1) -> (0, sin eps, cos eps).
    ncp = T.state_equ_to_ecl(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(ncp, [0.0, SIN_EPS, COS_EPS], rtol=0, atol=1e-16)

    # North ecliptic pole, equatorial components (0, -sin eps, cos eps) -> (0, 0, 1).
    nep = T.state_equ_to_ecl(np.array([0.0, -SIN_EPS, COS_EPS]))
    assert np.allclose(nep, [0.0, 0.0, 1.0], rtol=0, atol=1e-16)

    # A generic vector, arithmetic done by hand from the two constants above:
    #   y = 2 * 0.9174820620691818 + 3 * 0.3977771559319137 = 3.028295591934105
    #   z = -2 * 0.3977771559319137 + 3 * 0.9174820620691818 = 1.956891874343718
    got = T.state_equ_to_ecl(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(got, [1.0, 3.028295591934105, 1.956891874343718], rtol=0, atol=1e-15)


def test_rotation_matches_astropy():
    """Cross-check against astropy (TESTS ONLY — never used in the implementation).

    astropy's BarycentricMeanEcliptic uses the IAU 2006 obliquity
    (84381.406 arcsec) and additionally applies the ICRS frame bias, while
    digest42 uses the IAU 1976 value (84381.448 arcsec) pinned in constants.py.
    The two differ by 0.042 arcsec = 2.0e-7 rad, so for a vector of length
    |v| ~ 3.7 AU the expected disagreement is ~7.6e-7 AU.  The tolerance below
    is set just above that; it is a convention check, not a precision check.
    """
    astropy_units = pytest.importorskip("astropy.units")
    coords = pytest.importorskip("astropy.coordinates")

    v = np.array([1.0, 2.0, 3.0])
    icrs = coords.ICRS(
        coords.CartesianRepresentation(
            v[0] * astropy_units.au, v[1] * astropy_units.au, v[2] * astropy_units.au
        )
    )
    ecl = icrs.transform_to(coords.BarycentricMeanEcliptic(equinox="J2000"))
    ref = ecl.cartesian.xyz.to_value(astropy_units.au)

    got = T.state_equ_to_ecl(v)
    assert np.allclose(got, ref, rtol=0, atol=5e-6)
    # And the disagreement really is at the obliquity-difference level, not a
    # sign error (which would give O(1) differences).
    assert np.max(np.abs(got - ref)) < 2e-6


# --------------------------------------------------------------------------
# Sky geometry
# --------------------------------------------------------------------------


def test_topocentric_to_heliocentric_matches_definition():
    rng = np.random.default_rng(SEED + 1)
    n = 200
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    ra_rate = rng.normal(0, 0.01, n)
    dec_rate = rng.normal(0, 0.01, n)
    rho = 10 ** rng.uniform(-1.5, 1.5, n)
    rho_dot = rng.normal(0, 0.01, n)
    R, V = _fake_observer(n, rng)

    r, v = T.topocentric_to_heliocentric(ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V)
    u, u_a, u_d = T.tangent_basis(ra, dec)

    assert np.allclose(r, R + rho[:, None] * u, rtol=0, atol=1e-15)
    expect_v = (
        V
        + rho_dot[:, None] * u
        + rho[:, None] * (ra_rate[:, None] * u_a + dec_rate[:, None] * u_d)
    )
    assert np.allclose(v, expect_v, rtol=0, atol=1e-15)

    # Topocentric range and range-rate are recovered by construction (to the
    # cancellation floor eps * |R| / rho of forming r - R in floating point).
    rho_vec = r - R
    assert np.allclose(np.linalg.norm(rho_vec, axis=-1), rho, rtol=1e-13, atol=0)
    assert np.allclose(np.sum((v - V) * u, axis=-1), rho_dot, rtol=0, atol=1e-16)

    # Scalar / unbatched call shapes.
    r1, v1 = T.topocentric_to_heliocentric(
        ra[0], dec[0], ra_rate[0], dec_rate[0], rho[0], rho_dot[0], R[0], V[0]
    )
    assert r1.shape == (3,) and v1.shape == (3,)
    assert np.allclose(r1, r[0]) and np.allclose(v1, v[0])


def test_mixed_scalar_array_broadcasting():
    """WP2 review regression: ra/dec must broadcast against each other.

    Before the fix, `tangent_basis(ra=(5,), dec=scalar)` raised from np.stack
    while the transposed call worked; the asymmetry propagated into
    `topocentric_to_heliocentric` and `ranging_jacobian_det`, and disagreed
    with `jacobian_state_wrt_observables`, which pre-broadcasts.  Both halves
    of the same composition must accept the same inputs.
    """
    ra_arr = np.linspace(0.1, 1.1, 5)
    dec_arr = np.linspace(-0.4, 0.4, 5)

    # Both mixed orders work and agree with the fully-broadcast call.
    for ra, dec in ((ra_arr, 0.3), (0.3, dec_arr)):
        ra_b, dec_b = np.broadcast_arrays(np.asarray(ra, float), np.asarray(dec, float))
        assert T.unit_vector(ra, dec).shape == (5, 3)
        assert np.allclose(T.unit_vector(ra, dec), T.unit_vector(ra_b, dec_b))
        got = T.tangent_basis(ra, dec)
        ref = T.tangent_basis(ra_b, dec_b)
        assert all(g.shape == (5, 3) for g in got)
        assert all(np.allclose(g, r) for g, r in zip(got, ref))

    # The whole composition accepts the mixed call, and matches the explicit one.
    rng = np.random.default_rng(SEED + 21)
    R, V = _fake_observer(5, rng)
    kwargs = dict(ra_rate=0.004, dec_rate=-0.002, rho=1.7, rho_dot=0.006)
    r1, v1 = T.topocentric_to_heliocentric(ra_arr, 0.3, R_obs=R, V_obs=V, **kwargs)
    r2, v2 = T.topocentric_to_heliocentric(
        ra_arr, np.full(5, 0.3), R_obs=R, V_obs=V, **kwargs
    )
    assert r1.shape == (5, 3) and np.allclose(r1, r2) and np.allclose(v1, v2)

    J1 = T.jacobian_state_wrt_observables(ra_arr, 0.3, R_obs=R, V_obs=V, **kwargs)
    J2 = T.jacobian_state_wrt_observables(ra_arr, np.full(5, 0.3), R_obs=R, V_obs=V, **kwargs)
    assert J1.shape == (5, 6, 6) and np.allclose(J1, J2)

    d1 = T.ranging_jacobian_det(ra_arr, 0.3, R_obs=R, V_obs=V, **kwargs)
    d2 = T.ranging_jacobian_det(ra_arr, np.full(5, 0.3), R_obs=R, V_obs=V, **kwargs)
    assert d1.shape == (5,)
    assert np.allclose(d1, d2, equal_nan=True)


def test_public_api_is_exported():
    """WP2 review note: every public function appears in __all__."""
    import inspect

    public = {
        name
        for name, obj in vars(T).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == T.__name__
    }
    assert public == set(T.__all__), public.symmetric_difference(T.__all__)
    assert "jacobian_state_wrt_elements" in T.__all__


def test_tangent_basis_geometry():
    """Sanity properties of the (non-orthonormal) tangent basis used everywhere."""
    rng = np.random.default_rng(SEED + 2)
    ra = rng.uniform(0, 2 * np.pi, 300)
    dec = np.arcsin(rng.uniform(-0.99, 0.99, 300))
    u, u_a, u_d = T.tangent_basis(ra, dec)
    assert np.allclose(np.linalg.norm(u, axis=-1), 1.0, atol=1e-15)
    assert np.allclose(np.sum(u * u_a, axis=-1), 0.0, atol=1e-15)
    assert np.allclose(np.sum(u * u_d, axis=-1), 0.0, atol=1e-15)
    assert np.allclose(np.sum(u_a * u_d, axis=-1), 0.0, atol=1e-15)
    # |du/dra| = cos(dec) (the non-orthonormality the rate projection must respect)
    assert np.allclose(np.linalg.norm(u_a, axis=-1), np.cos(dec), atol=1e-15)
    assert np.allclose(np.linalg.norm(u_d, axis=-1), 1.0, atol=1e-15)


# --------------------------------------------------------------------------
# 1. Element <-> state round-trips
# --------------------------------------------------------------------------


def test_elements_state_roundtrip():
    """Acceptance 1: elements -> state -> elements to 1e-10 relative.

    Sweep: a in [0.5, 50] AU (log-uniform), e in [0, 0.95], isotropic
    inclinations (all i), uniform angles.  Near-degenerate draws (e < 1e-3 or
    sin i < 1e-3) are checked in the state metric instead, because the *angles*
    themselves carry no information there (see module docstring); they are
    covered explicitly by `test_degenerate_element_conventions`.
    """
    rng = np.random.default_rng(SEED + 3)
    n = 20000
    el = _sample_elements(n, rng, regime="wide")

    r, v = T.state_from_elements(el)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))

    el2 = T.elements_from_state(r, v)
    assert np.all(np.isfinite(el2))

    good = (el[:, 1] > 1e-3) & (np.sin(el[:, 2]) > 1e-3)
    assert good.sum() > 0.95 * n  # the excluded set is tiny

    d = _element_diff(el2[good], el[good])
    a_relerr = np.max(np.abs(d[:, 0]) / el[good, 0])
    e_err = np.max(np.abs(d[:, 1]))
    ang_err = np.max(np.abs(d[:, 2:]))
    assert a_relerr < 1e-10, a_relerr
    assert e_err < 1e-10, e_err
    assert ang_err < 1e-10, ang_err

    # ... and the round-trip through the state is accurate for the FULL sweep,
    # degenerate draws included.
    r2, v2 = T.state_from_elements(el2)
    rn = np.linalg.norm(r, axis=-1)[:, None]
    vn = np.linalg.norm(v, axis=-1)[:, None]
    assert np.max(np.abs(r2 - r) / rn) < 1e-12
    assert np.max(np.abs(v2 - v) / vn) < 1e-12


def test_state_elements_state_roundtrip():
    """The other composition order: state -> elements -> state."""
    rng = np.random.default_rng(SEED + 4)
    n = 5000
    el = _sample_elements(n, rng, regime="wide")
    r, v = T.state_from_elements(el)

    el2 = T.elements_from_state(r, v)
    r2, v2 = T.state_from_elements(el2)
    assert np.max(np.abs(r2 - r) / np.linalg.norm(r, axis=-1)[:, None]) < 1e-12
    assert np.max(np.abs(v2 - v) / np.linalg.norm(v, axis=-1)[:, None]) < 1e-12


def test_elements_physical_invariants():
    """Elements reproduce the independently computed energy, |h| and q."""
    rng = np.random.default_rng(SEED + 5)
    n = 3000
    el = _sample_elements(n, rng, regime="wide")
    r, v = T.state_from_elements(el)
    a, e, i = el[:, 0], el[:, 1], el[:, 2]

    rn = np.linalg.norm(r, axis=-1)
    energy = 0.5 * np.sum(v * v, axis=-1) - MU_SUN / rn
    assert np.allclose(energy, -MU_SUN / (2.0 * a), rtol=1e-12)

    h = np.linalg.norm(np.cross(r, v), axis=-1)
    assert np.allclose(h, np.sqrt(MU_SUN * a * (1.0 - e**2)), rtol=1e-12)

    hz = np.cross(r, v)[:, 2]
    assert np.allclose(hz / h, np.cos(i), atol=1e-12)

    # Periapsis / apoapsis bracket the instantaneous radius.
    assert np.all(rn >= a * (1 - e) - 1e-12)
    assert np.all(rn <= a * (1 + e) + 1e-12)

    qei = T.qei_from_elements(el)
    assert np.allclose(qei[:, 0], a * (1 - e), rtol=0, atol=1e-15)
    assert np.allclose(qei[:, 1], e) and np.allclose(qei[:, 2], i)


def test_kepler_equation_solved_exactly():
    """M = E - e sin E is satisfied to machine precision, including e -> 1."""
    rng = np.random.default_rng(SEED + 6)
    e = np.concatenate([rng.uniform(0.0, 0.999, 5000), [0.0, 1e-12, 0.5, 0.99, 0.999]])
    M = np.concatenate([rng.uniform(-6 * np.pi, 6 * np.pi, 5000), [0.0, 0.0, np.pi, 1e-9, 2 * np.pi]])
    E = T._kepler_E(M, e)
    resid = _wrap_pi(E - e * np.sin(E) - M)
    assert np.max(np.abs(resid)) < 1e-14, np.max(np.abs(resid))


def test_degenerate_element_conventions():
    """Documented conventions on the degenerate loci (DESIGN §2.1 / docstring)."""
    # Circular, inclined: omega == 0, M is the argument of latitude.
    el = np.array([2.5, 0.0, 0.7, 1.1, 0.0, 0.9])
    r, v = T.state_from_elements(el)
    back = T.elements_from_state(r, v)
    assert back[1] == pytest.approx(0.0, abs=1e-15)
    assert back[4] == pytest.approx(0.0, abs=1e-12)  # omega convention
    assert back[3] == pytest.approx(1.1, abs=1e-12)  # Omega still well defined
    assert back[5] == pytest.approx(0.9, abs=1e-12)  # M carries the phase

    # Equatorial prograde, eccentric: Omega == 0, omega absorbs the node.
    el = np.array([1.7, 0.3, 0.0, 1.2, 0.4, 2.0])
    r, v = T.state_from_elements(el)
    back = T.elements_from_state(r, v)
    assert back[2] == pytest.approx(0.0, abs=1e-15)
    assert back[3] == pytest.approx(0.0, abs=1e-12)  # Omega convention
    assert back[4] == pytest.approx((1.2 + 0.4) % (2 * np.pi), abs=1e-12)
    assert back[5] == pytest.approx(2.0, abs=1e-12)

    # Equatorial RETROGRADE: i = pi, Omega = 0, omega measured in the direction
    # of motion (clockwise seen from +z), i.e. Omega - omega_in for the input.
    el = np.array([1.7, 0.3, np.pi, 1.2, 0.4, 2.0])
    r, v = T.state_from_elements(el)
    back = T.elements_from_state(r, v)
    assert back[2] == pytest.approx(np.pi, abs=1e-8)
    assert back[3] == pytest.approx(0.0, abs=1e-12)
    assert back[4] == pytest.approx((0.4 - 1.2) % (2 * np.pi), abs=1e-8)

    # Circular AND equatorial: Omega = omega = 0, M is the true longitude.
    el = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.3])
    r, v = T.state_from_elements(el)
    back = T.elements_from_state(r, v)
    assert back[3] == pytest.approx(0.0, abs=1e-14)
    assert back[4] == pytest.approx(0.0, abs=1e-14)
    assert back[5] == pytest.approx(1.3, abs=1e-12)
    assert np.arctan2(r[1], r[0]) % (2 * np.pi) == pytest.approx(1.3, abs=1e-12)


def test_elements_against_published_worked_example():
    """Independent check of the element conventions against a textbook case.

    Reference: H. D. Curtis, *Orbital Mechanics for Engineering Students*
    (3rd ed.), Example 4.3 — a geocentric state with mu = 398600 km^3/s^2:

        r = (-6045, -3490,  2500) km
        v = (-3.457,  6.618, 2.533) km/s

    published answers: h = 58310 km^2/s, a = 8788 km, e = 0.1712,
    i = 153.2 deg, Omega = 255.3 deg, omega = 20.07 deg, nu = 28.45 deg.

    The two-body problem is scale invariant, so the case is transplanted into
    digest42 units by keeping km as the length unit and rescaling TIME by
    Ts = sqrt(MU_SUN / mu_earth) so that the gravitational parameter becomes
    MU_SUN.  This is a genuinely external check: it fixes the sign conventions
    of Omega and omega (note i > 90 deg — a RETROGRADE orbit, the case where a
    sign slip would show up most loudly) without reusing state_from_elements.
    """
    mu_earth = 398600.0  # km^3 / s^2
    r = np.array([-6045.0, -3490.0, 2500.0])  # km
    v = np.array([-3.457, 6.618, 2.533])  # km / s
    Ts = math.sqrt(MU_SUN / mu_earth)  # new time unit, in seconds

    el = T.elements_from_state(r, v * Ts)
    a, e, i, Om, om, M = el

    assert a == pytest.approx(8788.0, rel=5e-5)
    assert e == pytest.approx(0.1712, rel=1e-3)
    assert math.degrees(i) == pytest.approx(153.2, abs=0.05)
    assert math.degrees(Om) == pytest.approx(255.3, abs=0.05)
    assert math.degrees(om) == pytest.approx(20.07, abs=0.01)

    E = T._kepler_E(M, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0), math.sqrt(1.0 - e) * math.cos(E / 2.0)
    )
    assert math.degrees(nu) % 360.0 == pytest.approx(28.45, abs=0.01)

    h = np.linalg.norm(np.cross(r, v))  # km^2/s, unscaled
    assert h == pytest.approx(58310.0, rel=1e-4)

    # And the state is reproduced by the inverse map (in the scaled units).
    r_back, v_back = T.state_from_elements(el)
    assert np.allclose(r_back, r, rtol=1e-12)
    assert np.allclose(v_back, v * Ts, rtol=1e-12)


def test_circular_orbit_hand_computed():
    """A hand-computable case: 1 AU circular ecliptic orbit.

    v_circ = sqrt(mu/a) = K_GAUSS AU/day at a = 1 AU, and the period is
    2 pi / K_GAUSS = 365.2568983... days (the Gaussian year) — the number the
    Gaussian gravitational constant was defined to produce.
    """
    k = math.sqrt(MU_SUN)
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, k, 0.0])
    el = T.elements_from_state(r, v)
    assert el[0] == pytest.approx(1.0, abs=1e-15)  # a
    assert el[1] == pytest.approx(0.0, abs=1e-16)  # e
    assert el[2] == pytest.approx(0.0, abs=1e-16)  # i
    assert el[3] == pytest.approx(0.0, abs=1e-16)  # Omega (degenerate convention)
    assert el[4] == pytest.approx(0.0, abs=1e-16)  # omega (degenerate convention)
    assert el[5] == pytest.approx(0.0, abs=1e-16)  # M = true longitude = 0

    period = 2.0 * math.pi / k
    assert period == pytest.approx(365.2568983, abs=1e-6)

    # A quarter period later the mean anomaly is pi/2 and the object is at +y.
    el_q = el.copy()
    el_q[5] = math.pi / 2
    r_q, v_q = T.state_from_elements(el_q)
    assert np.allclose(r_q, [0.0, 1.0, 0.0], atol=1e-15)
    assert np.allclose(v_q, [-k, 0.0, 0.0], atol=1e-16)


def test_unbound_states_return_nan():
    """Hyperbolic / parabolic inputs give NaN rows; bound ones do not."""
    r = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v_circ = math.sqrt(MU_SUN)
    v = np.array(
        [
            [0.0, v_circ, 0.0],  # circular: bound
            [0.0, math.sqrt(2.0) * v_circ, 0.0],  # exactly parabolic
            [0.0, 2.0 * v_circ, 0.0],  # hyperbolic
        ]
    )
    el = T.elements_from_state(r, v)
    assert np.all(np.isfinite(el[0]))
    assert np.all(np.isnan(el[1]))
    assert np.all(np.isnan(el[2]))

    # The inverse direction refuses unbound / invalid elements too.
    bad = np.array([[-2.0, 0.3, 0.1, 0.0, 0.0, 0.0], [2.0, 1.4, 0.1, 0.0, 0.0, 0.0]])
    rb, vb = T.state_from_elements(bad)
    assert np.all(np.isnan(rb)) and np.all(np.isnan(vb))

    # ... and so do the Jacobians / the determinant.
    assert np.all(np.isnan(T.jacobian_elements_wrt_state(r[2], v[2])))
    assert np.all(np.isnan(T.jacobian_state_wrt_elements(bad)))


# --------------------------------------------------------------------------
# 4. Photometry
# --------------------------------------------------------------------------


def test_hg_phase_integral_reference_values():
    """Acceptance 4: IAU H,G phase function spot checks.

    Source of the functional form and coefficients: Bowell, Hapke, Domingue,
    Lumme, Peltoniemi & Harris (1989), "Application of photometric models to
    asteroids", in *Asteroids II* (Univ. Arizona Press), eqs. (A4)-(A5):

        Phi_i(alpha) = exp(-A_i tan^{B_i}(alpha/2)),
        (A1, B1) = (3.33, 0.63),  (A2, B2) = (1.87, 1.22)
        Phi(alpha; G) = (1 - G) Phi_1(alpha) + G Phi_2(alpha).

    Reference values used below, recomputed by hand from those published
    coefficients (they isolate A_i and B_i independently):

      * Phi(0; G) = 1 exactly for every G  (definition of H).
      * G = 0, alpha = 90 deg: tan(45 deg) = 1, so Phi = Phi_1 = exp(-3.33)
        = 0.0357931050676553  -> isolates A1.
      * G = 1, alpha = 90 deg: Phi = Phi_2 = exp(-1.87) = 0.1541236618151314
        -> isolates A2.
      * G = 0.15, alpha = 20 deg: Phi = 0.39824440766, i.e. a phase correction
        of 0.99963 mag.  The "~1 mag of phase darkening at 20 deg for
        G = 0.15" is the standard, widely quoted property of the adopted IAU
        system and pins B1/B2 as well as the (1-G)/G mixing.
      * G = 0.15, alpha = 90 deg: Phi = 0.05354268856 (3.17825 mag).
    """
    assert T.hg_phase_integral(0.0, 0.15) == pytest.approx(1.0, abs=1e-15)
    for G in (0.0, 0.15, 0.25, 0.5, 1.0):
        assert T.hg_phase_integral(0.0, G) == pytest.approx(1.0, abs=1e-15)

    half_pi = math.radians(90.0)
    assert T.hg_phase_integral(half_pi, 0.0) == pytest.approx(0.0357931050676553, rel=1e-12)
    assert T.hg_phase_integral(half_pi, 1.0) == pytest.approx(0.1541236618151314, rel=1e-12)
    assert T.hg_phase_integral(half_pi, 0.15) == pytest.approx(0.05354268856, rel=1e-9)

    twenty = math.radians(20.0)
    assert T.hg_phase_integral(twenty, 0.15) == pytest.approx(0.39824440766, rel=1e-9)
    dmag20 = -2.5 * math.log10(T.hg_phase_integral(twenty, 0.15))
    assert dmag20 == pytest.approx(1.0, abs=1e-3)  # the ~1 mag @ 20 deg property

    # Monotone decreasing in alpha, bounded in [0, 1], and -> 0 at alpha = pi.
    alpha = np.linspace(0.0, np.pi, 2001)
    phi = T.hg_phase_integral(alpha, G_SLOPE_DEFAULT)
    assert np.all((phi >= 0.0) & (phi <= 1.0))
    assert np.all(np.diff(phi) < 0.0)
    assert T.hg_phase_integral(np.pi, 0.15) == pytest.approx(0.0, abs=1e-12)

    # Linear in G at fixed alpha, by construction.
    a = np.array([0.1, 0.5, 1.0, 2.0])
    p0 = T.hg_phase_integral(a, 0.0)
    p1 = T.hg_phase_integral(a, 1.0)
    assert np.allclose(T.hg_phase_integral(a, 0.3), 0.7 * p0 + 0.3 * p1, rtol=1e-14)


def test_magnitude_inverses():
    """Acceptance 4: absolute_magnitude and apparent_magnitude are inverses."""
    rng = np.random.default_rng(SEED + 7)
    n = 5000
    V = rng.uniform(14.0, 26.0, n)
    rho = 10 ** rng.uniform(-2.0, 1.7, n)
    r = 10 ** rng.uniform(np.log10(0.3), np.log10(60.0), n)
    alpha = rng.uniform(0.0, math.radians(160.0), n)

    H = T.absolute_magnitude(V, rho, r, alpha)
    V2 = T.apparent_magnitude(H, rho, r, alpha)
    assert np.max(np.abs(V2 - V)) < 1e-12

    H2 = T.absolute_magnitude(V2, rho, r, alpha)
    assert np.max(np.abs(H2 - H)) < 1e-12

    # dH/dV = 1 exactly (DESIGN §2.2: unit Jacobian factor in the magnitude
    # dimension).
    assert np.allclose(T.absolute_magnitude(V + 1.0, rho, r, alpha) - H, 1.0, atol=1e-12)

    # Explicit formula check on one case.
    H0 = T.absolute_magnitude(20.0, 1.5, 2.4, math.radians(20.0), 0.15)
    expect = 20.0 - 5 * math.log10(1.5 * 2.4) + 2.5 * math.log10(0.39824440766)
    assert H0 == pytest.approx(expect, abs=1e-9)

    # Nonzero phase always LOWERS the implied H (DESIGN §2.2).
    H_opp = T.absolute_magnitude(20.0, 1.5, 2.4, 0.0)
    assert H_opp > H0


def test_phase_angle():
    # Object at opposition (Sun - Earth - object aligned, object beyond).
    R = np.array([1.0, 0.0, 0.0])
    r = np.array([3.0, 0.0, 0.0])
    assert T.phase_angle(r, R) == pytest.approx(0.0, abs=1e-12)

    # Object between the Sun and the observer, on the line: phase = pi.
    r = np.array([0.5, 0.0, 0.0])
    assert T.phase_angle(r, R) == pytest.approx(np.pi, abs=1e-12)

    # Right angle at the object: Sun at origin, observer such that
    # (-r) . (R - r) = 0.  Take r = (1, 0, 0), R = (1, 1, 0).
    r = np.array([1.0, 0.0, 0.0])
    assert T.phase_angle(r, np.array([1.0, 1.0, 0.0])) == pytest.approx(np.pi / 2, abs=1e-12)

    # Frame independence: rotating both vectors leaves the angle unchanged.
    rng = np.random.default_rng(SEED + 8)
    rr = rng.normal(size=(200, 3)) * 3.0
    RR = rng.normal(size=(200, 3))
    a1 = T.phase_angle(rr, RR)
    a2 = T.phase_angle(T.state_equ_to_ecl(rr), T.state_equ_to_ecl(RR))
    assert np.max(np.abs(a1 - a2)) < 1e-13
    assert np.all((a1 >= 0.0) & (a1 <= np.pi))

    # Law of cosines cross-check: cos(alpha) = (r^2 + rho^2 - R^2) / (2 r rho)
    rn = np.linalg.norm(rr, axis=-1)
    Rn = np.linalg.norm(RR, axis=-1)
    rhon = np.linalg.norm(RR - rr, axis=-1)
    cosa = (rn**2 + rhon**2 - Rn**2) / (2.0 * rn * rhon)
    assert np.max(np.abs(np.cos(a1) - cosa)) < 1e-12


# --------------------------------------------------------------------------
# 2. Jacobians vs finite differences
# --------------------------------------------------------------------------


def test_jacobian_state_wrt_observables_vs_fd():
    """Acceptance 2 (a): analytic d(r,v)/d(A, rho, rho_dot) vs 5-point FD."""
    rng = np.random.default_rng(SEED + 9)
    n = 300
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-0.99, 0.99, n))
    ra_rate = rng.normal(0.0, 0.02, n)
    dec_rate = rng.normal(0.0, 0.02, n)
    rho = 10 ** rng.uniform(-1.0, 1.7, n)
    rho_dot = rng.normal(0.0, 0.02, n)
    R, V = _fake_observer(n, rng)

    J = T.jacobian_state_wrt_observables(ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V)
    assert J.shape == (n, 6, 6)

    worst = 0.0
    for k in range(n):
        x0 = np.array([ra[k], dec[k], ra_rate[k], dec_rate[k], rho[k], rho_dot[k]])

        def f(x, k=k):
            r, v = T.topocentric_to_heliocentric(x[0], x[1], x[2], x[3], x[4], x[5], R[k], V[k])
            return np.concatenate([r, v])

        h = 1e-5 * np.array([1.0, 1.0, 1e-2, 1e-2, max(rho[k], 1e-2), 1e-2])
        Jfd = _fd_jacobian(f, x0, h)
        worst = max(worst, _rowwise_relerr(J[k], Jfd))
    print(f"\n  max rowwise rel err d(state)/d(observables): {worst:.3e}")
    assert worst < 1e-6, worst

    # Structural zeros required by the contract (r does not depend on the
    # rates or on rho_dot).
    assert np.all(J[..., 0:3, 2] == 0.0)
    assert np.all(J[..., 0:3, 3] == 0.0)
    assert np.all(J[..., 0:3, 5] == 0.0)

    # Unbatched call.
    J1 = T.jacobian_state_wrt_observables(
        ra[0], dec[0], ra_rate[0], dec_rate[0], rho[0], rho_dot[0], R[0], V[0]
    )
    assert J1.shape == (6, 6)
    assert np.allclose(J1, J[0], atol=0)


def test_jacobian_state_wrt_elements_vs_fd():
    """The analytic forward Jacobian d(r,v)/d(elements) that WP2 inverts."""
    rng = np.random.default_rng(SEED + 10)
    n = 300
    el = _sample_elements(n, rng, regime="mixed")
    J = T.jacobian_state_wrt_elements(el)
    assert J.shape == (n, 6, 6)

    worst = 0.0
    for k in range(n):

        def f(x):
            r, v = T.state_from_elements(x)
            return np.concatenate([r, v])

        a = el[k, 0]
        h = 1e-5 * np.array([a, 1.0, 1.0, 1.0, 1.0, 1.0])
        Jfd = _fd_jacobian(f, el[k], h)
        worst = max(worst, _rowwise_relerr(J[k], Jfd))
    print(f"\n  max rowwise rel err d(state)/d(elements): {worst:.3e}")
    assert worst < 1e-6, worst


def test_jacobian_elements_wrt_state_vs_fd():
    """Acceptance 2 (b): analytic d(elements)/d(r, v) vs 5-point FD.

    Degeneracies excluded: e < 0.02, sin(i) < 0.02 (see module docstring).
    """
    rng = np.random.default_rng(SEED + 11)
    n = 400
    el = _sample_elements(n, rng, regime="mixed")
    r, v = T.state_from_elements(el)

    J = T.jacobian_elements_wrt_state(r, v)
    assert J.shape == (n, 6, 6)

    keep = (el[:, 1] > 0.02) & (np.sin(el[:, 2]) > 0.02)
    assert keep.sum() > 0.8 * n

    worst = 0.0
    n_skipped_cond = 0
    for k in np.flatnonzero(keep):
        if np.linalg.cond(J[k]) > 1e10:
            n_skipped_cond += 1
            continue
        x0 = np.concatenate([r[k], v[k]])

        def f(x):
            return T.elements_from_state(x[:3], x[3:])

        scale = np.concatenate(
            [
                np.full(3, max(np.max(np.abs(r[k])), 1e-3)),
                np.full(3, max(np.max(np.abs(v[k])), 1e-8)),
            ]
        )
        h = 1e-5 * scale
        Jfd = _fd_jacobian(f, x0, h, wrap_rows=(3, 4, 5))
        worst = max(worst, _rowwise_relerr(J[k], Jfd))
    print(
        f"\n  max rowwise rel err d(elements)/d(state): {worst:.3e}"
        f"  (cond-skipped {n_skipped_cond}/{keep.sum()})"
    )
    assert worst < 1e-6, worst


@pytest.mark.parametrize(
    "a, e, i_deg, label",
    [
        (0.9, 0.90, 30.0, "NEO, high e"),
        (1.2, 0.95, 5.0, "NEO, e at the sweep ceiling"),
        (2.7, 0.10, 10.0, "MBA"),
        (5.2, 0.05, 20.0, "Trojan-like"),
        (45.0, 0.30, 25.0, "TNO"),
        (45.0, 0.02, 25.0, "TNO, e exactly at the mask edge"),
        (2.5, 0.20, 1.2, "sin(i) just above the mask edge"),
        (30.0, 0.05, 175.0, "retrograde"),
        (0.6, 0.50, 90.0, "polar"),
    ],
)
def test_jacobians_at_regime_boundaries(a, e, i_deg, label):
    """Both element Jacobians at the edges of the documented valid domain.

    Random angles at each fixed (a, e, i); this pins the FD agreement right up
    against the excluded loci (e = 0.02, sin i = 0.02) and at the eccentricity
    and semimajor-axis extremes of the WP2 sweeps, where the analytic form is
    most likely to lose accuracy.
    """
    rng = np.random.default_rng(SEED + 100 + int(a * 100 + e * 1000 + i_deg))
    i = math.radians(i_deg)
    worst_fwd = 0.0
    worst_inv = 0.0
    for _ in range(30):
        el = np.array(
            [a, e, i, rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)]
        )
        r, v = T.state_from_elements(el)

        Jf = T.jacobian_state_wrt_elements(el)
        Jf_fd = _fd_jacobian(
            lambda x: np.concatenate(T.state_from_elements(x)),
            el,
            1e-5 * np.array([a, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        worst_fwd = max(worst_fwd, _rowwise_relerr(Jf, Jf_fd))

        Ji = T.jacobian_elements_wrt_state(r, v)
        x0 = np.concatenate([r, v])
        hs = 1e-5 * np.concatenate(
            [
                np.full(3, max(np.max(np.abs(r)), 1e-3)),
                np.full(3, max(np.max(np.abs(v)), 1e-8)),
            ]
        )
        Ji_fd = _fd_jacobian(
            lambda x: T.elements_from_state(x[:3], x[3:]), x0, hs, wrap_rows=(3, 4, 5)
        )
        worst_inv = max(worst_inv, _rowwise_relerr(Ji, Ji_fd))

    print(f"\n  [{label}] fwd {worst_fwd:.2e}  inv {worst_inv:.2e}")
    assert worst_fwd < 1e-6, (label, worst_fwd)
    assert worst_inv < 1e-6, (label, worst_inv)


def test_jacobian_elements_wrt_state_nan_when_ill_conditioned():
    """WP2 review regression: no plausible-looking garbage near-degeneracy.

    Previously only EXACT singularity was caught, so an e = 1e-16 state
    returned a finite inverse whose |det| was off by a factor 0.74.  The
    condition guard (`_COND_MAX`) must NaN out the ill-conditioned cases while
    leaving every physically interesting orbit alone.
    """
    i = 0.3
    a = 2.7
    for e in (1e-13, 1e-16, 0.0):
        el = np.array([a, e, i, 1.0, 2.0, 3.0])
        r, v = T.state_from_elements(el)
        assert np.all(np.isnan(T.jacobian_elements_wrt_state(r, v))), e

    for sin_i in (1e-13, 1e-16, 0.0):
        el = np.array([a, 0.2, math.asin(sin_i), 1.0, 2.0, 3.0])
        r, v = T.state_from_elements(el)
        assert np.all(np.isnan(T.jacobian_elements_wrt_state(r, v))), sin_i

    # Comfortably non-degenerate cases stay finite AND accurate: e = 1e-6 is
    # already six decades below anything in a real population.
    for e in (1e-6, 1e-4, 0.02, 0.5):
        el = np.array([a, e, i, 1.0, 2.0, 3.0])
        r, v = T.state_from_elements(el)
        J = T.jacobian_elements_wrt_state(r, v)
        assert np.all(np.isfinite(J)), e
        exact = 2.0 / (MU_SUN**1.5 * math.sqrt(a) * e * math.sin(i))
        assert abs(abs(np.linalg.det(J)) / exact - 1.0) < 1e-6, e

    # No orbit in the WP2 sweeps is masked by the guard.
    rng = np.random.default_rng(SEED + 22)
    el = _sample_elements(4000, rng, regime="mixed")
    r, v = T.state_from_elements(el)
    J = T.jacobian_elements_wrt_state(r, v)
    assert np.all(np.isfinite(J))

    # Batched mixed good/degenerate input: per-row masking, no exception.
    el_mix = np.stack([np.array([a, 0.3, i, 1.0, 2.0, 3.0]), np.array([a, 0.0, i, 1.0, 2.0, 3.0])])
    r, v = T.state_from_elements(el_mix)
    Jm = T.jacobian_elements_wrt_state(r, v)
    assert np.all(np.isfinite(Jm[0])) and np.all(np.isnan(Jm[1]))


def test_magnitude_rejects_position_vector():
    """WP2 review regression: r_helio is a DISTANCE, not a position vector."""
    r_vec = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="DISTANCE"):
        T.absolute_magnitude(20.0, 1.5, r_vec, 0.1)
    with pytest.raises(ValueError, match="DISTANCE"):
        T.apparent_magnitude(15.0, 1.5, r_vec, 0.1)

    # Batched form too.
    with pytest.raises(ValueError, match="DISTANCE"):
        T.absolute_magnitude(np.full(4, 20.0), np.full(4, 1.5), np.zeros((4, 3)), np.zeros(4))

    # The correct call still works, and matches the intended value.
    r_scalar = float(np.linalg.norm(r_vec))
    H = T.absolute_magnitude(20.0, 1.5, r_scalar, 0.1)
    assert np.ndim(H) == 0 and np.isfinite(H)

    # Legitimate broadcasts are unaffected: scalar V with batched geometry.
    n = 5
    out = T.absolute_magnitude(20.0, np.full(n, 1.5), np.full(n, 2.4), np.full(n, 0.1))
    assert out.shape == (n,)


def test_jacobian_elements_state_is_true_inverse():
    """The two analytic Jacobians are exact matrix inverses of one another."""
    rng = np.random.default_rng(SEED + 12)
    n = 500
    el = _sample_elements(n, rng, regime="mixed")
    keep = (el[:, 1] > 0.02) & (np.sin(el[:, 2]) > 0.02)
    el = el[keep]
    r, v = T.state_from_elements(el)

    A = T.jacobian_state_wrt_elements(el)
    B = T.jacobian_elements_wrt_state(r, v)
    prod = B @ A
    ident = np.broadcast_to(np.eye(6), prod.shape)
    # Scale-free check: the residual is measured relative to cond(A).
    resid = np.max(np.abs(prod - ident), axis=(-2, -1))
    cond = np.linalg.cond(A)
    assert np.all(resid < 1e-9 * np.maximum(cond, 1.0)), np.max(resid / cond)
    assert np.max(resid) < 1e-6, np.max(resid)


def _exact_ranging_det(el, obs):
    """Closed-form |det d(a,e,i,Omega,omega,M)/d(A, rho, rho_dot)|.

    Independent of every forward map in `transforms`, derived via Delaunay
    variables (L, G, H, l, g, h), which are canonically conjugate to (r, v) so
    that |det d(Delaunay)/d(r, v)| = 1 exactly.  With
    L = sqrt(mu a), G = L sqrt(1-e^2), H = G cos i and (l, g, h) = (M, omega,
    Omega), the triangular block gives

        |det d(L,G,H)/d(a,e,i)| = mu^{3/2} sqrt(a) e sin(i) / 2

    hence |det d(E)/d(r, v)| = 2 / (mu^{3/2} sqrt(a) e sin i).  The observable
    block contributes |det d(r, v)/d(A, rho, rho_dot)| = rho^4 cos^2(dec)
    (rho^2 from the two position rows that carry rho * du/dra, rho * du/ddec,
    rho^2 again from the matching velocity rows, and cos^2(dec) = |du/dra|^2
    from the non-orthonormal tangent basis), and the frame rotation is
    orthogonal.  So

        |det| = rho^4 cos^2(dec) * 2 / (mu^{3/2} sqrt(a) e sin i).

    This pins the CONVENTIONS, the COMPOSITION and the rho^4 cos^2(dec) scaling
    at once, without reusing any of the module's own derivatives.
    """
    a, e, i = el[..., 0], el[..., 1], el[..., 2]
    return (
        obs["rho"] ** 4
        * np.cos(obs["dec"]) ** 2
        * 2.0
        / (MU_SUN**1.5 * np.sqrt(a) * e * np.sin(i))
    )


def test_ranging_jacobian_det_exact_closed_form():
    """The determinant against an exact, FD-free closed form (see helper).

    A large sweep is affordable because no finite differencing is involved.
    Excluded: sin(i) < 0.02, |dec| > 85 deg, rho < 0.05 AU (module docstring);
    e is bounded away from 0 by the sampler.
    """
    rng = np.random.default_rng(SEED + 20)
    n = 8000
    el = _sample_elements(n, rng, regime="mixed")
    R, V = _fake_observer(n, rng)
    obs = T.observables_from_elements(el, np.full(n, 15.0), R, V)

    keep = (
        (np.sin(el[:, 2]) > 0.02)
        & (np.abs(obs["dec"]) < math.radians(85.0))
        & (obs["rho"] > 0.05)
    )
    assert keep.sum() > 0.6 * n

    det = T.ranging_jacobian_det(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )
    exact = _exact_ranging_det(el, obs)
    rel = np.abs(det[keep] / exact[keep] - 1.0)
    print(f"\n  |det| vs exact Delaunay form: n={keep.sum()} median {np.median(rel):.2e} max {np.max(rel):.2e}")
    assert np.allclose(det[keep], exact[keep], rtol=1e-9)

    # The (a, e) -> (q, e) reparameterization is the same closed form times (1-e).
    det_q = T.ranging_jacobian_det(
        obs["ra"],
        obs["dec"],
        obs["ra_rate"],
        obs["dec_rate"],
        obs["rho"],
        obs["rho_dot"],
        R,
        V,
        parameterization="qei",
    )
    assert np.allclose(det_q[keep], exact[keep] * (1.0 - el[keep, 1]), rtol=1e-9)


def test_ranging_jacobian_det_vs_fd():
    """Acceptance 2 (c): |det d(E)/d(A, rho, rho_dot)| for both parameterizations.

    Geometries are generated *forward* from randomized NEO/MBA/TNO elements, so
    every test point corresponds to a real bound orbit.  Excluded loci:
    e < 0.02, sin(i) < 0.02, |dec| > 85 deg, rho < 0.05 AU (module docstring).
    """
    rng = np.random.default_rng(SEED + 13)
    n = 400
    el = _sample_elements(n, rng, regime="mixed")
    H = rng.uniform(5.0, 25.0, n)
    R, V = _fake_observer(n, rng)
    obs = T.observables_from_elements(el, H, R, V)

    keep = (
        (el[:, 1] > 0.02)
        & (np.sin(el[:, 2]) > 0.02)
        & (np.abs(obs["dec"]) < math.radians(85.0))
        & (obs["rho"] > 0.05)
    )
    assert keep.sum() > 0.6 * n

    det_aei = T.ranging_jacobian_det(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )
    det_qei = T.ranging_jacobian_det(
        obs["ra"],
        obs["dec"],
        obs["ra_rate"],
        obs["dec_rate"],
        obs["rho"],
        obs["rho_dot"],
        R,
        V,
        parameterization="qei",
    )
    assert det_aei.shape == (n,)
    assert np.all(det_aei[keep] > 0.0)
    # The reparameterization factor is exactly (1 - e).
    assert np.allclose(det_qei[keep], det_aei[keep] * (1.0 - el[keep, 1]), rtol=1e-12)
    # ... and the closed form below must hold here too.
    exact = _exact_ranging_det(el, obs)
    assert np.allclose(det_aei[keep], exact[keep], rtol=1e-9)

    worst_aei = 0.0
    worst_qei = 0.0
    n_used = 0
    for k in np.flatnonzero(keep):
        x0 = np.array(
            [
                obs["ra"][k],
                obs["dec"][k],
                obs["ra_rate"][k],
                obs["dec_rate"][k],
                obs["rho"][k],
                obs["rho_dot"][k],
            ]
        )

        def f_aei(x, k=k):
            r, v = T.topocentric_to_heliocentric(x[0], x[1], x[2], x[3], x[4], x[5], R[k], V[k])
            return T.elements_from_state(T.state_equ_to_ecl(r), T.state_equ_to_ecl(v))

        def f_qei(x, k=k):
            E = f_aei(x, k)
            return np.concatenate([[E[0] * (1.0 - E[1])], E[1:]])

        h = 1e-5 * np.array(
            [
                1.0,
                1.0,
                max(abs(obs["ra_rate"][k]), 1e-3),
                max(abs(obs["dec_rate"][k]), 1e-3),
                obs["rho"][k],
                max(abs(obs["rho_dot"][k]), 1e-4),
            ]
        )
        J_aei = _fd_jacobian(f_aei, x0, h, wrap_rows=(3, 4, 5))
        if np.linalg.cond(J_aei) > 1e10:
            continue
        n_used += 1
        fd_aei = abs(np.linalg.det(J_aei))
        fd_qei = abs(np.linalg.det(_fd_jacobian(f_qei, x0, h, wrap_rows=(3, 4, 5))))
        worst_aei = max(worst_aei, abs(det_aei[k] / fd_aei - 1.0))
        worst_qei = max(worst_qei, abs(det_qei[k] / fd_qei - 1.0))

    print(
        f"\n  max rel err |det| aei: {worst_aei:.3e}, qei: {worst_qei:.3e}"
        f"  (n = {n_used}/{keep.sum()})"
    )
    assert n_used > 0.5 * keep.sum()
    assert worst_aei < 1e-6, worst_aei
    assert worst_qei < 1e-6, worst_qei


def test_ranging_jacobian_det_composition_and_scaling():
    """The determinant really is the product of its documented factors."""
    rng = np.random.default_rng(SEED + 14)
    n = 50
    el = _sample_elements(n, rng, regime="mixed")
    H = rng.uniform(5.0, 25.0, n)
    R, V = _fake_observer(n, rng)
    obs = T.observables_from_elements(el, H, R, V)

    r_equ, v_equ = T.topocentric_to_heliocentric(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )
    J_obs = T.jacobian_state_wrt_observables(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )
    J_el = T.jacobian_elements_wrt_state(T.state_equ_to_ecl(r_equ), T.state_equ_to_ecl(v_equ))

    expect = np.abs(np.linalg.det(J_el)) * np.abs(np.linalg.det(J_obs))
    got = T.ranging_jacobian_det(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )
    assert np.allclose(got, expect, rtol=1e-9)

    # The frame-rotation block is orthogonal: |det| = 1 exactly.
    assert abs(abs(np.linalg.det(T._ROT6_EQU_TO_ECL)) - 1.0) < 1e-15

    with pytest.raises(ValueError):
        T.ranging_jacobian_det(
            obs["ra"][0],
            obs["dec"][0],
            obs["ra_rate"][0],
            obs["dec_rate"][0],
            obs["rho"][0],
            obs["rho_dot"][0],
            R[0],
            V[0],
            parameterization="cometary",
        )


def test_ranging_jacobian_det_unbound_is_nan():
    """An unbound (rho, rho_dot) hypothesis yields NaN, not garbage."""
    rng = np.random.default_rng(SEED + 15)
    R, V = _fake_observer(1, rng)
    det = T.ranging_jacobian_det(
        0.5, 0.2, 0.0, 0.0, 1.0, 5.0, R[0], V[0]  # rho_dot = 5 AU/day -> hyperbolic
    )
    assert np.isnan(det)


# --------------------------------------------------------------------------
# 3. Forward generator round-trip
# --------------------------------------------------------------------------


def test_observables_from_elements_roundtrip():
    """Acceptance 3: forward generator o inverse map = identity.

    observables_from_elements -> topocentric_to_heliocentric ->
    state_equ_to_ecl -> elements_from_state must return the input elements, and
    (rho, rho_dot) must be recovered exactly (they are outputs of the generator
    and inputs of the inverse map, so this is a true identity check on the
    geometry, not just a consistency check).
    """
    rng = np.random.default_rng(SEED + 16)
    n = 20000
    el = _sample_elements(n, rng, regime="wide")
    H = rng.uniform(3.0, 26.0, n)
    R, V = _fake_observer(n, rng)

    obs = T.observables_from_elements(el, H, R, V)
    for key in ("ra", "dec", "ra_rate", "dec_rate", "v_mag", "rho", "rho_dot"):
        assert key in obs
        assert obs[key].shape == (n,)
    assert np.all((obs["ra"] >= 0.0) & (obs["ra"] < 2.0 * np.pi))
    assert np.all(np.abs(obs["dec"]) <= np.pi / 2)

    r_equ, v_equ = T.topocentric_to_heliocentric(
        obs["ra"], obs["dec"], obs["ra_rate"], obs["dec_rate"], obs["rho"], obs["rho_dot"], R, V
    )

    # (rho, rho_dot) identity, to machine precision.
    rho_back = np.linalg.norm(r_equ - R, axis=-1)
    u = (r_equ - R) / rho_back[:, None]
    rho_dot_back = np.sum((v_equ - V) * u, axis=-1)
    assert np.max(np.abs(rho_back / obs["rho"] - 1.0)) < 1e-13
    assert np.max(np.abs(rho_dot_back - obs["rho_dot"])) < 1e-13

    # The heliocentric state itself is recovered (frame-independent statement).
    r_true, v_true = T.state_from_elements(el)
    r_true_equ = T.state_ecl_to_equ(r_true)
    v_true_equ = T.state_ecl_to_equ(v_true)
    assert np.max(np.abs(r_equ - r_true_equ)) < 1e-12
    assert np.max(np.abs(v_equ - v_true_equ)) < 1e-14

    el_back = T.elements_from_state(T.state_equ_to_ecl(r_equ), T.state_equ_to_ecl(v_equ))
    good = (el[:, 1] > 1e-3) & (np.sin(el[:, 2]) > 1e-3) & (np.cos(obs["dec"]) > 1e-3)
    assert good.sum() > 0.9 * n
    d = _element_diff(el_back[good], el[good])
    assert np.max(np.abs(d[:, 0]) / el[good, 0]) < 1e-10
    assert np.max(np.abs(d[:, 1])) < 1e-10
    assert np.max(np.abs(d[:, 2:])) < 1e-10


def test_observables_photometry_consistency():
    """v_mag from the generator inverts back to the input H."""
    rng = np.random.default_rng(SEED + 17)
    n = 3000
    el = _sample_elements(n, rng, regime="mixed")
    H = rng.uniform(3.0, 26.0, n)
    R, V = _fake_observer(n, rng)
    obs = T.observables_from_elements(el, H, R, V)

    r_ecl, _ = T.state_from_elements(el)
    r_equ = T.state_ecl_to_equ(r_ecl)
    alpha = T.phase_angle(r_equ, R)
    H_back = T.absolute_magnitude(
        obs["v_mag"], obs["rho"], np.linalg.norm(r_equ, axis=-1), alpha
    )
    assert np.max(np.abs(H_back - H)) < 1e-11

    # A non-default G must propagate.
    obs2 = T.observables_from_elements(el, H, R, V, G=0.5)
    changed = alpha > math.radians(0.5)
    assert np.all(obs2["v_mag"][changed] != obs["v_mag"][changed])


def test_observables_rate_projection_uses_correct_metric():
    """Rates come back from the tangential velocity with the cos^2(dec) metric."""
    rng = np.random.default_rng(SEED + 18)
    n = 500
    el = _sample_elements(n, rng, regime="mixed")
    R, V = _fake_observer(n, rng)
    obs = T.observables_from_elements(el, np.full(n, 15.0), R, V)

    r_ecl, v_ecl = T.state_from_elements(el)
    r_equ, v_equ = T.state_ecl_to_equ(r_ecl), T.state_ecl_to_equ(v_ecl)
    rho_vec = r_equ - R
    rho = np.linalg.norm(rho_vec, axis=-1)
    w = v_equ - V
    u, u_a, u_d = T.tangent_basis(obs["ra"], obs["dec"])

    # Reconstruct the full relative velocity from the three observables.
    w_rebuilt = (
        obs["rho_dot"][:, None] * u
        + rho[:, None] * (obs["ra_rate"][:, None] * u_a + obs["dec_rate"][:, None] * u_d)
    )
    assert np.max(np.abs(w_rebuilt - w)) < 1e-14

    # Naive (wrong) normalization would use cos(dec), not cos^2(dec); confirm
    # the difference is real for off-equator geometries, i.e. the test bites.
    off = np.abs(obs["dec"]) > math.radians(20.0)
    naive = np.sum(w * u_a, axis=-1) / (rho * np.cos(obs["dec"]))
    assert np.all(np.abs(naive[off] - obs["ra_rate"][off]) > 1e-12)


def test_observables_batching_and_scalar_shapes():
    """Leading-batch-dimension convention holds and scalars work."""
    rng = np.random.default_rng(SEED + 19)
    el = _sample_elements(4, rng, regime="mixed")
    H = rng.uniform(10.0, 20.0, 4)
    R, V = _fake_observer(4, rng)

    batched = T.observables_from_elements(el, H, R, V)
    for k in range(4):
        single = T.observables_from_elements(el[k], H[k], R[k], V[k])
        for key, val in single.items():
            assert np.ndim(val) == 0
            assert val == pytest.approx(batched[key][k], rel=1e-14, abs=1e-14)

    # Elements / state functions keep the (...,) -> (..., 6) shape contract.
    assert T.elements_from_state(np.zeros(3) + [1.0, 0.0, 0.0], np.zeros(3)).shape == (6,)
    r, v = T.state_from_elements(el[0])
    assert r.shape == (3,) and v.shape == (3,)
    assert T.qei_from_elements(el).shape == (4, 3)
    assert T.qei_from_elements(el[0]).shape == (3,)

    # Multi-dimensional batch (N, M, ...) works too.
    el2 = el.reshape(2, 2, 6)
    assert T.state_from_elements(el2)[0].shape == (2, 2, 3)
    assert T.jacobian_state_wrt_elements(el2).shape == (2, 2, 6, 6)
