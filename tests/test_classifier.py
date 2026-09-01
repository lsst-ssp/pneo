"""WP5 unit tests — the classifier (marginal likelihoods, posterior, maps).

Acceptance criteria (IMPLEMENTATION.md WP5):
  1. hand-constructed obvious cases with the real digest2 model score as
     expected (fast opposition tracklet -> P(NEO) ~ 1; slow one -> ~ 0),
  2. results are invariant under grid refinement to a stated tolerance,
  3. the *machinery* (weights, masking, Jacobian pairing, photometry) is
     verified against a brute-force integration with analytically known toy
     densities, so the test does not depend on the digest2 file at all,
  4. edge cases (empty admissible region, everything truncated away, NaN
     nodes, overflow, bad input) degrade per contract rather than silently,
  5. classify_many matches per-tracklet classify, and the posterior map
     renders headless.

Design notes
------------
* The synthetic-density tests re-derive Lambda_C with an explicit per-node
  Python loop and, separately, with an independent quadrature rule on a
  *linear* rho grid, so a bookkeeping bug in the vectorized classifier cannot
  hide behind the same bug in the test.
* Geometries are deliberately NOT all at opposition.  At exact opposition the
  phase angle is identically zero at every node, Phi(0; G) = 1 for any G, and
  the whole H,G branch of the photometry is unexercised — a classifier that
  ignored the phase angle entirely would pass.  The off-opposition cases below
  (elongation 120 deg and 90 deg) close that hole.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
import numpy as np
import pytest
from astropy.time import Time

matplotlib.use("Agg")  # headless: must be set before pyplot is imported

from digest42.attributable import Attributable, observer_state
from digest42.classifier import (
    N_RHODOT_DEFAULT,
    N_RHO_DEFAULT,
    ClassificationResult,
    classify,
    classify_many,
)
from digest42.constants import G_SLOPE_DEFAULT, OBLIQUITY_J2000
from digest42.population import Digest2Histogram, PopulationDensity
from digest42.ranging import RangingGrid, admissible_rhodot_interval, build_grid
from digest42.transforms import (
    absolute_magnitude,
    elements_from_state,
    phase_angle,
    qei_from_elements,
    ranging_jacobian_det,
    state_equ_to_ecl,
    tangent_basis,
    topocentric_to_heliocentric,
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "digest2.model.csv",
)

#: One fixed epoch for every hand-constructed case (results must not depend on
#: when the test is run).
EPOCH = Time("2026-03-15T06:00:00", scale="utc")

#: North ecliptic pole in EQUATORIAL components (see transforms.state_equ_to_ecl).
_ECL_POLE = np.array([0.0, -np.sin(OBLIQUITY_J2000), np.cos(OBLIQUITY_J2000)])


# ---------------------------------------------------------------------------
# Helpers: hand-constructed tracklets on the ecliptic
# ---------------------------------------------------------------------------


def _observer():
    """Geocentric heliocentric state at EPOCH, equatorial frame."""
    return observer_state(EPOCH, obscode="500")


def ecliptic_tracklet(elongation_deg: float, rate_deg_per_day: float,
                      v_mag: float, R_obs=None):
    """A tracklet on the ecliptic at a given solar elongation, moving retrograde.

    ``R_obs`` is the observer's heliocentric position, so ``+R_obs/|R_obs|`` is
    the anti-solar direction as seen from the observer (elongation 180 deg).
    The line of sight is rotated away from it, within the ecliptic plane, by
    ``180 - elongation`` degrees.

    At elongation 180 (opposition) the object sits exactly on the Sun-observer
    line, so the phase angle is exactly 0 at every node of the ranging grid —
    convenient for the truncation test, but it makes the H,G phase function a
    no-op, hence the off-opposition geometries used elsewhere.

    The apparent motion is placed along the ecliptic in the direction of
    *decreasing* ecliptic longitude — the retrograde loop that any prograde
    heliocentric orbit shows near opposition — with the given total on-sky
    rate.  The tangent basis is orthogonal but not orthonormal
    (|du/dra| = cos dec), hence the cos^2(dec) in ra_rate.
    """
    if R_obs is None:
        R_obs, _ = _observer()
    e1 = np.asarray(R_obs, dtype=float)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(_ECL_POLE, e1)
    e2 = e2 / np.linalg.norm(e2)

    delta = np.radians(180.0 - float(elongation_deg))
    u = np.cos(delta) * e1 + np.sin(delta) * e2
    u = u / np.linalg.norm(u)

    ra = float(np.arctan2(u[1], u[0]) % (2.0 * np.pi))
    dec = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))

    # +(pole x u) points along increasing ecliptic longitude; retrograde is -.
    s = -np.cross(_ECL_POLE, u)
    s = s / np.linalg.norm(s)

    _, du_dra, du_ddec = tangent_basis(ra, dec)
    rate = np.radians(rate_deg_per_day)
    return Attributable(
        ra=ra,
        dec=dec,
        ra_rate=float(rate * np.dot(s, du_dra) / np.cos(dec) ** 2),
        dec_rate=float(rate * np.dot(s, du_ddec)),
        epoch=EPOCH,
        v_mag=float(v_mag),
    )


def opposition_tracklet(rate_deg_per_day: float, v_mag: float, R_obs=None):
    """Shorthand for ``ecliptic_tracklet(180, ...)``."""
    return ecliptic_tracklet(180.0, rate_deg_per_day, v_mag, R_obs=R_obs)


@pytest.fixture(scope="module")
def d2pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(MODEL_PATH)


@pytest.fixture(scope="module")
def obs():
    return _observer()


# ---------------------------------------------------------------------------
# Toy densities (machinery tests; independent of the digest2 file)
# ---------------------------------------------------------------------------


class BoxDensity(PopulationDensity):
    """Uniform density inside an (x, e, i, H) box, zero outside.

    ``x`` is q or a depending on ``parameterization``.  The amplitude is the
    full 7-D density (uniform angles already folded in), matching the
    PopulationDensity contract, so no extra (2 pi)^-3 appears anywhere.
    """

    def __init__(self, label, parameterization, x, e, i, H, amplitude):
        self.label = label
        self.parameterization = parameterization
        self._x = tuple(map(float, x))
        self._e = tuple(map(float, e))
        self._i = tuple(map(float, i))
        self._H = tuple(map(float, H))
        self.amplitude = float(amplitude)
        self.seen = []  # (elements3, H) of every call, for input-hygiene tests

    def density(self, elements3, H):
        elements3 = np.asarray(elements3, dtype=float)
        H = np.asarray(H, dtype=float)
        self.seen.append((elements3, H))
        x, e, i = elements3[..., 0], elements3[..., 1], elements3[..., 2]
        x, e, i, h = np.broadcast_arrays(x, e, i, H)
        inside = (
            (x >= self._x[0]) & (x < self._x[1])
            & (e >= self._e[0]) & (e < self._e[1])
            & (i >= self._i[0]) & (i < self._i[1])
            & (h >= self._H[0]) & (h < self._H[1])
        )
        return np.where(inside, self.amplitude, 0.0)

    def sample(self, n, rng):  # pragma: no cover - not needed by WP5
        raise NotImplementedError

    @property
    def support(self):
        key = "q" if self.parameterization == "qei" else "a"
        return {key: self._x, "e": self._e, "i": self._i, "H": self._H}


class ConstantDensity(PopulationDensity):
    """Returns the same value for every input, finite or not.

    Used to prove the classifier's masking, not the density's: if a masked
    node were allowed through, this density would happily contribute.
    """

    def __init__(self, label="NEO", parameterization="qei", value=1.0):
        self.label = label
        self.parameterization = parameterization
        self.value = float(value)
        self.seen = []

    def density(self, elements3, H):
        elements3 = np.asarray(elements3, dtype=float)
        H = np.asarray(H, dtype=float)
        self.seen.append((elements3, H))
        return np.full(np.broadcast(elements3[..., 0], H).shape, self.value)

    def sample(self, n, rng):  # pragma: no cover
        raise NotImplementedError

    @property
    def support(self):
        return {"q": (0.0, np.inf), "e": (0.0, 1.0), "i": (0.0, np.pi),
                "H": (-np.inf, np.inf)}


def _toy_pair(parameterization="qei", amp_neo=1.0, amp_other=3.0):
    """Two toy components partitioning at x = 1.3 AU (cf. DESIGN §2.4/D-2).

    The inclination window spans the full [0, pi) so that the "other" box is
    populated off opposition too (retrograde-on-sky motion far from opposition
    only maps onto retrograde heliocentric orbits at large q).
    """
    common = dict(e=(0.0, 0.9), i=(0.0, np.pi), H=(10.0, 25.5))
    neo = BoxDensity("NEO", parameterization, x=(0.0, 1.3),
                     amplitude=amp_neo, **common)
    other = BoxDensity("other", parameterization, x=(1.3, 5.0),
                       amplitude=amp_other, **common)
    return neo, other


def _brute_force_lambdas(att, densities, grid, R_obs, V_obs, G=G_SLOPE_DEFAULT):
    """Lambda_C on the SAME nodes, computed one node at a time.

    Deliberately unvectorized and written independently of classifier.py: it
    checks the bookkeeping (which weight goes with which node, which Jacobian
    parameterization goes with which density, how the phase angle enters H,
    how NaN nodes are masked).
    """
    totals = [0.0] * len(densities)
    for k in range(grid.rho.size):
        rho = float(grid.rho[k])
        rho_dot = float(grid.rho_dot[k])
        w = float(grid.weight[k])

        r_equ, v_equ = topocentric_to_heliocentric(
            att.ra, att.dec, att.ra_rate, att.dec_rate,
            np.array(rho), np.array(rho_dot), R_obs, V_obs,
        )
        el = elements_from_state(state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ))
        if not np.all(np.isfinite(el)):
            continue  # unbound node -> contributes exactly zero
        r_helio = float(np.linalg.norm(r_equ))
        alpha = float(phase_angle(r_equ, R_obs))
        H = float(absolute_magnitude(att.v_mag, rho, r_helio, alpha, G=G))
        if not np.isfinite(H):
            continue

        for j, dens in enumerate(densities):
            param = dens.parameterization
            det = float(
                ranging_jacobian_det(
                    att.ra, att.dec, att.ra_rate, att.dec_rate,
                    np.array(rho), np.array(rho_dot), R_obs, V_obs,
                    parameterization=param,
                )
            )
            if not np.isfinite(det):
                continue
            if param == "qei":
                el3 = np.asarray(qei_from_elements(el))
            else:
                el3 = np.asarray(el)[..., 0:3]
            n_c = float(dens.density(el3.reshape(1, 3), np.array([H]))[0])
            totals[j] += n_c * det * w
    return totals


def _independent_quadrature(att, densities, R_obs, V_obs, rho_min, rho_max,
                            n_rho, n_rhodot, G=G_SLOPE_DEFAULT,
                            chunk=200_000):
    """Lambda_C by a *different* quadrature: LINEAR-rho midpoint rule.

    classify() integrates on a log-spaced rho grid; this uses uniformly spaced
    rho with midpoint nodes and the same per-rho admissible rho_dot interval,
    so the two agree only if both are integrating the same measure.  Evaluated
    in chunks so the reference can be run at a resolution well beyond the
    classifier's without exhausting memory on the 6x6 Jacobian stacks.
    """
    edges = np.linspace(rho_min, rho_max, n_rho + 1)
    rho_nodes = 0.5 * (edges[:-1] + edges[1:])
    d_rho = np.diff(edges)

    lo, hi = admissible_rhodot_interval(att, R_obs, V_obs, rho_nodes)
    width = hi - lo
    valid = np.isfinite(width) & (width > 0.0)
    rho_v, lo_v, d_rho_v = rho_nodes[valid], lo[valid], d_rho[valid]
    d_rhodot = width[valid] / n_rhodot

    off = np.arange(n_rhodot, dtype=float) + 0.5
    rho_all = np.repeat(rho_v, n_rhodot)
    rho_dot_all = (lo_v[:, None] + d_rhodot[:, None] * off[None, :]).ravel()
    weight_all = np.repeat(d_rho_v * d_rhodot, n_rhodot)

    totals = [0.0] * len(densities)
    for start in range(0, rho_all.size, chunk):
        sl = slice(start, start + chunk)
        rho, rho_dot, weight = rho_all[sl], rho_dot_all[sl], weight_all[sl]

        r_equ, v_equ = topocentric_to_heliocentric(
            att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rho_dot,
            R_obs, V_obs,
        )
        el = elements_from_state(
            state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ)
        )
        good = np.all(np.isfinite(el), axis=-1)
        r_helio = np.linalg.norm(r_equ, axis=-1)
        H = absolute_magnitude(
            att.v_mag, rho, r_helio, phase_angle(r_equ, R_obs), G=G
        )

        for j, dens in enumerate(densities):
            param = dens.parameterization
            det = ranging_jacobian_det(
                att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rho_dot,
                R_obs, V_obs, parameterization=param,
            )
            el_s = np.where(good[:, None], el, 0.0)
            el3 = qei_from_elements(el_s) if param == "qei" else el_s[..., 0:3]
            n_c = dens.density(el3, np.where(good, H, 0.0))
            integ = np.where(
                good & np.isfinite(det), n_c * np.nan_to_num(det), 0.0
            )
            totals[j] += float(np.sum(integ * weight))
    return totals


def _node_phase_angles(att, grid, R_obs, V_obs):
    r_equ, _ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate,
        grid.rho, grid.rho_dot, R_obs, V_obs,
    )
    return phase_angle(r_equ, R_obs)


# ---------------------------------------------------------------------------
# 0. Grid defaults
# ---------------------------------------------------------------------------


def test_default_grid_is_rho_heavy():
    """The defaults are deliberately lopsided; see N_RHO_DEFAULT's rationale.

    H(V, rho) turns every H-bin (and q-bin) edge of a binned population model
    into a rho-aligned discontinuity of the integrand, so all the quadrature
    error lives in the rho direction; the integrand is smooth across rho_dot.
    """
    assert (N_RHO_DEFAULT, N_RHODOT_DEFAULT) == (1200, 50)
    sig = classify.__defaults__
    assert N_RHO_DEFAULT in sig and N_RHODOT_DEFAULT in sig


# ---------------------------------------------------------------------------
# 1. Hand-constructed obvious cases (real digest2 model)
# ---------------------------------------------------------------------------


def test_fast_opposition_tracklet_scores_neo(d2pair, obs):
    """~1.5 deg/day at opposition: a rate no main-belt object can show."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(1.5, 21.0, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V)

    assert isinstance(res, ClassificationResult)
    assert res.p_neo > 0.9, res.p_neo
    assert res.log_lr > 2.0
    assert res.lambda_neo > 0.0 and res.lambda_other > 0.0
    # Internal consistency of the reported scalars.
    assert res.p_neo == pytest.approx(
        res.lambda_neo / (res.lambda_neo + res.lambda_other), rel=1e-12
    )
    assert res.log_lr == pytest.approx(
        np.log(res.lambda_neo) - np.log(res.lambda_other), rel=1e-12
    )
    # Per-node records are the integrand, NOT weighted.
    assert res.node_density_neo.shape == res.grid.rho.shape
    assert res.lambda_neo == pytest.approx(
        float(np.sum(res.node_density_neo * res.grid.weight)), rel=1e-12
    )
    # Bookkeeping counters.
    assert res.n_nodes == res.grid.rho.size
    assert res.node_H.shape == res.node_q.shape == res.grid.rho.shape


@pytest.mark.parametrize("rate_deg", [1.0, 1.5, 2.0])
def test_fast_rates_all_score_high(d2pair, obs, rate_deg):
    neo, other = d2pair
    R, V = obs
    res = classify(opposition_tracklet(rate_deg, 21.0, R_obs=R), neo, other,
                   R_obs=R, V_obs=V)
    assert res.p_neo > 0.9, (rate_deg, res.p_neo)


def test_slow_opposition_tracklet_scores_mba(d2pair, obs):
    """~0.2 deg/day at V = 20: the textbook main-belt tracklet."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V)

    assert res.p_neo < 0.01, res.p_neo
    assert res.log_lr < -4.0
    assert res.lambda_other > res.lambda_neo


def test_rate_monotonicity_between_the_two_regimes(d2pair, obs):
    """P(NEO) rises with the on-sky rate at fixed V (a sanity slice)."""
    neo, other = d2pair
    R, V = obs
    ps = [
        classify(opposition_tracklet(rate, 21.0, R_obs=R), neo, other,
                 R_obs=R, V_obs=V, n_rho=600, n_rhodot=50).p_neo
        for rate in (0.2, 0.5, 1.5)
    ]
    assert ps[0] < ps[1] < ps[2], ps


# ---------------------------------------------------------------------------
# 2. Grid-refinement invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate_deg, v_mag", [(1.5, 21.0), (0.2, 20.0)])
def test_grid_refinement_invariance(d2pair, obs, rate_deg, v_mag):
    """p_neo must move by < 0.01 between the defaults and a 2x refinement."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(rate_deg, v_mag, R_obs=R)
    coarse = classify(att, neo, other, R_obs=R, V_obs=V,
                      n_rho=N_RHO_DEFAULT, n_rhodot=N_RHODOT_DEFAULT)
    fine = classify(att, neo, other, R_obs=R, V_obs=V,
                    n_rho=2 * N_RHO_DEFAULT, n_rhodot=2 * N_RHODOT_DEFAULT)
    assert abs(fine.p_neo - coarse.p_neo) < 0.01, (coarse.p_neo, fine.p_neo)
    assert fine.lambda_other == pytest.approx(coarse.lambda_other, rel=0.10)


def test_refinement_in_rho_dominates(d2pair, obs):
    """Refining rho moves p_neo more than refining rho_dot (defaults' premise)."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(1.0, 20.0, R_obs=R)
    base = classify(att, neo, other, R_obs=R, V_obs=V, n_rho=150, n_rhodot=50)
    more_rho = classify(att, neo, other, R_obs=R, V_obs=V,
                        n_rho=1200, n_rhodot=50)
    more_rhodot = classify(att, neo, other, R_obs=R, V_obs=V,
                           n_rho=150, n_rhodot=400)
    d_rho = abs(more_rho.p_neo - base.p_neo)
    d_rhodot = abs(more_rhodot.p_neo - base.p_neo)
    assert d_rho > d_rhodot, (d_rho, d_rhodot)


# ---------------------------------------------------------------------------
# 3. Synthetic-density machinery tests (no digest2 file involved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elongation", [180.0, 120.0, 90.0])
def test_toy_density_matches_brute_force_same_grid(obs, elongation):
    """Exact bookkeeping check: same nodes, explicit per-node loop.

    Run at three elongations.  The two off-opposition ones have a nonzero,
    node-varying phase angle, so the H,G branch of the photometry is part of
    what is being compared: a classifier that dropped `phase_angle` (or used a
    different G) would disagree with the loop here even though it agrees at
    opposition.
    """
    R, V = obs
    att = ecliptic_tracklet(elongation, 0.4, 20.0, R_obs=R)
    neo, other = _toy_pair()

    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    res = classify(att, neo, other, R_obs=R, V_obs=V, **kw)
    grid = build_grid(att, R, V, **kw)

    assert np.allclose(grid.rho, res.grid.rho)

    alpha = _node_phase_angles(att, grid, R, V)
    if elongation == 180.0:
        assert np.allclose(alpha, 0.0, atol=1e-12)  # documents the blind spot
    else:
        assert np.degrees(np.nanmin(alpha)) > 5.0
        assert np.degrees(np.nanmax(alpha)) > 30.0

    ref_neo, ref_other = _brute_force_lambdas(att, (neo, other), grid, R, V)

    # Guard against a degenerate comparison: both boxes must actually be hit.
    assert ref_neo > 0.0 and ref_other > 0.0
    assert 0.01 < ref_neo / (ref_neo + ref_other) < 0.99

    assert res.lambda_neo == pytest.approx(ref_neo, rel=1e-12)
    assert res.lambda_other == pytest.approx(ref_other, rel=1e-12)
    assert res.p_neo == pytest.approx(ref_neo / (ref_neo + ref_other), rel=1e-12)


def test_toy_density_matches_independent_quadrature(obs):
    """p_neo from an independent (linear-rho) quadrature agrees to < 0.01."""
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    neo, other = _toy_pair()

    res = classify(att, neo, other, R_obs=R, V_obs=V,
                   rho_min=0.05, rho_max=6.0, n_rho=1200, n_rhodot=50)
    # The reference is run well past the classifier's resolution: off
    # opposition the "other" box is a thin sliver of the admissible region
    # (~4% of nodes), so an under-resolved reference is the noisier of the two.
    ref_neo, ref_other = _independent_quadrature(
        att, (neo, other), R, V, rho_min=0.05, rho_max=6.0,
        n_rho=4000, n_rhodot=400,
    )
    p_ref = ref_neo / (ref_neo + ref_other)
    assert 0.01 < p_ref < 0.99, p_ref  # a non-degenerate comparison
    assert res.p_neo == pytest.approx(p_ref, abs=0.01), (res.p_neo, p_ref)
    # The absolute likelihoods must also agree, not just their ratio.
    assert res.lambda_neo == pytest.approx(ref_neo, rel=0.05)
    assert res.lambda_other == pytest.approx(ref_other, rel=0.05)


def test_toy_amplitude_scaling_is_exact(obs):
    """Doubling a component's absolute calibration must scale its Lambda."""
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=80, n_rhodot=40)

    a_neo, a_other = _toy_pair(amp_neo=1.0, amp_other=1.0)
    b_neo, b_other = _toy_pair(amp_neo=2.0, amp_other=1.0)
    ra = classify(att, a_neo, a_other, R_obs=R, V_obs=V, **kw)
    rb = classify(att, b_neo, b_other, R_obs=R, V_obs=V, **kw)

    assert ra.lambda_neo > 0.0 and ra.lambda_other > 0.0
    assert rb.lambda_neo == pytest.approx(2.0 * ra.lambda_neo, rel=1e-12)
    assert rb.lambda_other == pytest.approx(ra.lambda_other, rel=1e-12)
    assert rb.log_lr == pytest.approx(ra.log_lr + np.log(2.0), rel=1e-12)


def test_aei_parameterization_uses_the_aei_jacobian(obs):
    """A stub density tagged 'aei' must get |J|_aei, not |J|_qei.

    The two differ by exactly the (a, e) -> (q, e) factor (1 - e), so pairing
    the wrong Jacobian with the density is detectable at the Lambda level.
    """
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=80, n_rhodot=40)

    wide = dict(e=(0.0, 1.0), i=(0.0, np.pi), H=(-50.0, 50.0), amplitude=1.0)
    dens_q = BoxDensity("NEO", "qei", x=(0.0, 1e6), **wide)
    dens_a = BoxDensity("other", "aei", x=(0.0, 1e6), **wide)

    res = classify(att, dens_q, dens_a, R_obs=R, V_obs=V, **kw)

    grid = build_grid(att, R, V, **kw)
    det_q = ranging_jacobian_det(
        att.ra, att.dec, att.ra_rate, att.dec_rate, grid.rho, grid.rho_dot,
        R, V, parameterization="qei",
    )
    det_a = ranging_jacobian_det(
        att.ra, att.dec, att.ra_rate, att.dec_rate, grid.rho, grid.rho_dot,
        R, V, parameterization="aei",
    )
    exp_q = float(np.nansum(np.nan_to_num(det_q) * grid.weight))
    exp_a = float(np.nansum(np.nan_to_num(det_a) * grid.weight))

    assert res.lambda_neo == pytest.approx(exp_q, rel=1e-12)
    assert res.lambda_other == pytest.approx(exp_a, rel=1e-12)
    assert res.lambda_neo < res.lambda_other  # the (1 - e) factor


def test_jacobian_computed_once_per_parameterization(monkeypatch, obs):
    """Efficiency contract: one ranging_jacobian_det call when both are qei."""
    import digest42.classifier as clf

    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    calls = []
    real = clf.ranging_jacobian_det

    def counting(*args, **kwargs):
        calls.append(kwargs.get("parameterization", "aei"))
        return real(*args, **kwargs)

    monkeypatch.setattr(clf, "ranging_jacobian_det", counting)

    neo, other = _toy_pair("qei")
    classify(att, neo, other, R_obs=R, V_obs=V,
             rho_min=0.05, rho_max=6.0, n_rho=40, n_rhodot=20)
    assert calls == ["qei"], calls

    calls.clear()
    mixed_a, mixed_b = _toy_pair("qei")[0], _toy_pair("aei")[1]
    classify(att, mixed_a, mixed_b, R_obs=R, V_obs=V,
             rho_min=0.05, rho_max=6.0, n_rho=40, n_rhodot=20)
    assert sorted(calls) == ["aei", "qei"], calls


# ---------------------------------------------------------------------------
# 3b. Photometry: the phase function must actually be used
# ---------------------------------------------------------------------------


def test_G_changes_the_answer_off_opposition(d2pair, obs):
    """G = 0.0 vs G = 0.5 must give materially different likelihoods.

    ...but only where the phase angle is nonzero: at exact opposition
    Phi(0; G) = 1 for every G, so the same comparison there is bit-for-bit
    identical.  Both halves are asserted, because the second is exactly the
    blind spot an opposition-only test suite has.
    """
    neo, other = d2pair
    R, V = obs
    kw = dict(R_obs=R, V_obs=V, n_rho=400, n_rhodot=50)

    off = ecliptic_tracklet(90.0, 0.4, 20.0, R_obs=R)
    g0 = classify(off, neo, other, G=0.0, **kw)
    g5 = classify(off, neo, other, G=0.5, **kw)
    assert abs(g0.p_neo - g5.p_neo) > 0.01, (g0.p_neo, g5.p_neo)
    assert max(g0.lambda_neo, g5.lambda_neo) > 1.2 * min(
        g0.lambda_neo, g5.lambda_neo
    )

    opp = opposition_tracklet(0.4, 20.0, R_obs=R)
    o0 = classify(opp, neo, other, G=0.0, **kw)
    o5 = classify(opp, neo, other, G=0.5, **kw)
    assert o0.lambda_neo == o5.lambda_neo
    assert o0.lambda_other == o5.lambda_other


def test_G_none_is_the_default_slope_and_not_an_arbitrary_one(d2pair, obs):
    """G=None means G_SLOPE_DEFAULT — checked where G demonstrably matters."""
    neo, other = d2pair
    R, V = obs
    att = ecliptic_tracklet(90.0, 0.4, 20.0, R_obs=R)
    kw = dict(R_obs=R, V_obs=V, n_rho=400, n_rhodot=50)

    auto = classify(att, neo, other, G=None, **kw)
    same = classify(att, neo, other, G=G_SLOPE_DEFAULT, **kw)
    other_g = classify(att, neo, other, G=0.5, **kw)

    assert auto.lambda_neo == same.lambda_neo
    assert auto.lambda_other == same.lambda_other
    # Non-tautological: this geometry can tell the slopes apart.
    assert auto.lambda_neo != other_g.lambda_neo


def test_phase_angle_enters_H(obs):
    """H at a node equals V - 5log10(rho r) + 2.5log10 Phi, phase included."""
    R, V = obs
    att = ecliptic_tracklet(90.0, 0.4, 20.0, R_obs=R)
    neo, other = _toy_pair()
    res = classify(att, neo, other, R_obs=R, V_obs=V,
                   rho_min=0.05, rho_max=6.0, n_rho=40, n_rhodot=20)

    r_equ, _ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate,
        res.grid.rho, res.grid.rho_dot, R, V,
    )
    alpha = phase_angle(r_equ, R)
    assert np.degrees(alpha).min() > 5.0
    expect = absolute_magnitude(
        att.v_mag, res.grid.rho, np.linalg.norm(r_equ, axis=-1), alpha,
        G=G_SLOPE_DEFAULT,
    )
    assert np.allclose(res.node_H, expect, equal_nan=True)
    # And the no-phase version would be a materially different H.
    no_phase = absolute_magnitude(
        att.v_mag, res.grid.rho, np.linalg.norm(r_equ, axis=-1),
        np.zeros_like(alpha), G=G_SLOPE_DEFAULT,
    )
    assert np.abs(no_phase - expect).max() > 0.2


# ---------------------------------------------------------------------------
# 3c. Masking counters and per-node diagnostics
# ---------------------------------------------------------------------------


def test_densities_never_see_non_finite_inputs(monkeypatch, obs):
    """Layer 1 of the mask: NaN nodes are replaced before density() is called.

    A ConstantDensity would return a positive value for a NaN node; the
    classifier must both keep NaN out of the density's arguments AND zero the
    node's contribution.
    """
    import digest42.classifier as clf

    R, V = obs
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=8.0, n_rho=60, n_rhodot=30)
    n_extra = 7

    real_build = clf.build_grid

    def augmented(att_, R_, V_, **kwargs):
        g = real_build(att_, R_, V_, **kwargs)
        rho_extra = np.geomspace(0.1, 5.0, n_extra)
        _, hi = admissible_rhodot_interval(att_, R_, V_, rho_extra)
        rho_dot_extra = np.nan_to_num(hi, nan=0.0) + 5.0  # way past escape
        return RangingGrid(
            rho=np.concatenate([g.rho, rho_extra]),
            rho_dot=np.concatenate([g.rho_dot, rho_dot_extra]),
            weight=np.concatenate([g.weight, np.full(n_extra, 1.0)]),
            n_rho=g.n_rho, n_rhodot=g.n_rhodot,
        )

    monkeypatch.setattr(clf, "build_grid", augmented)
    dens_a = ConstantDensity("NEO", "qei", 1.0)
    dens_b = ConstantDensity("other", "qei", 1.0)
    res = classify(att, dens_a, dens_b, R_obs=R, V_obs=V, **kw)

    # The injected nodes really are unbound...
    r_equ, v_equ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate,
        res.grid.rho[-n_extra:], res.grid.rho_dot[-n_extra:], R, V,
    )
    el = elements_from_state(state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ))
    assert np.all(np.isnan(el))

    # ... they are counted ...
    assert res.n_masked_nonfinite == n_extra
    assert res.n_nodes == res.grid.rho.size

    # ... the densities never saw a NaN ...
    assert dens_a.seen and dens_b.seen
    for el3, h in dens_a.seen + dens_b.seen:
        assert np.all(np.isfinite(el3))
        assert np.all(np.isfinite(h))

    # ... and they contributed exactly nothing despite a nonzero density.
    assert np.all(res.node_density_neo[-n_extra:] == 0.0)
    assert np.all(res.node_density_other[-n_extra:] == 0.0)
    assert np.all(np.isnan(res.node_q[-n_extra:]))


def test_non_finite_jacobian_is_masked_and_counted(monkeypatch, obs):
    """Layer 2 of the mask: finite elements, non-finite |J| -> zero + counted.

    WP2's `ranging_jacobian_det` returns NaN where the element map is
    numerically singular. Such nodes have a perfectly good density, so without
    the guard they would poison the sum with NaN; with it they silently vanish,
    which is why `n_masked_det` exists.
    """
    import digest42.classifier as clf

    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    neo, other = _toy_pair()

    clean = classify(att, neo, other, R_obs=R, V_obs=V, **kw)

    real = clf.ranging_jacobian_det
    victims = np.array([3, 17, 250, 900])

    def poisoned(*args, **kwargs):
        det = np.array(real(*args, **kwargs), dtype=float)
        det[victims] = np.nan
        return det

    monkeypatch.setattr(clf, "ranging_jacobian_det", poisoned)
    res = classify(att, neo, other, R_obs=R, V_obs=V, **kw)

    assert res.n_masked_det == victims.size
    assert res.n_masked_nonfinite == clean.n_masked_nonfinite
    assert np.isfinite(res.lambda_neo) and np.isfinite(res.lambda_other)
    assert np.all(res.node_density_neo[victims] == 0.0)
    # The lost nodes are exactly the difference.
    lost = float(
        np.sum(clean.node_density_neo[victims] * clean.grid.weight[victims])
    )
    assert res.lambda_neo == pytest.approx(clean.lambda_neo - lost, rel=1e-10)


def test_real_grids_have_no_nonfinite_nodes(d2pair, obs):
    """Documented property of WP3: the grid itself never places a NaN node."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V,
                   n_rho=300, n_rhodot=50)
    assert res.n_masked_nonfinite == 0
    # The det guard may still trip near the parabolic boundary; it must be rare.
    assert res.n_masked_det <= 0.02 * res.n_nodes


def test_node_H_and_node_q_and_the_section_2_6_helpers(d2pair, obs):
    """node_H / node_q and the two diagnostic helpers (DESIGN §2.6)."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(0.5, 22.5, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V,
                   n_rho=300, n_rhodot=50)

    good = np.isfinite(res.node_q)
    assert good.any()
    # q is a(1-e) of the implied orbit and is positive for bound orbits.
    assert np.all(res.node_q[good] > 0.0)
    assert np.all(np.isfinite(res.node_H[good]))

    # measure_fraction_H_above is a weight fraction in [0, 1], monotone in h.
    f_low = res.measure_fraction_H_above(-1e3)
    f_mid = res.measure_fraction_H_above(25.5)
    f_high = res.measure_fraction_H_above(1e3)
    assert f_low == pytest.approx(1.0)
    assert f_high == 0.0
    assert 0.0 <= f_mid <= 1.0
    assert f_low >= f_mid >= f_high

    # Direct re-derivation of the middle one.
    w = res.grid.weight
    assert f_mid == pytest.approx(
        float(w[res.node_H > 25.5].sum() / w.sum()), rel=1e-12
    )

    # posterior_mass_where: the whole domain is 1, the empty selection is 0,
    # and the truncation corner is a fraction of it.
    all_nodes = np.ones(res.n_nodes, dtype=bool)
    assert res.posterior_mass_where(all_nodes) == pytest.approx(1.0)
    assert res.posterior_mass_where(np.zeros(res.n_nodes, dtype=bool)) == 0.0

    corner = res.posterior_mass_where(
        lambda r: np.nan_to_num(r.node_q, nan=-1.0) > 1.3
    )
    assert 0.0 <= corner <= 1.0
    # Unnormalized 'other' mass over q > 1.3 is all of Lambda_other: the
    # digest2 'other' component is exactly the q > 1.3 population (D-2).
    other_mass = res.posterior_mass_where(
        lambda r: np.nan_to_num(r.node_q, nan=-1.0) > 1.3,
        component="other", normalize=False,
    )
    assert other_mass == pytest.approx(res.lambda_other, rel=1e-10)


def test_posterior_mass_where_validates(d2pair, obs):
    neo, other = d2pair
    R, V = obs
    res = classify(opposition_tracklet(1.0, 21.0, R_obs=R), neo, other,
                   R_obs=R, V_obs=V, n_rho=40, n_rhodot=20)
    with pytest.raises(ValueError, match="component"):
        res.posterior_mass_where(np.ones(res.n_nodes, bool), component="all")
    with pytest.raises(ValueError, match="shape"):
        res.posterior_mass_where(np.ones(3, bool))


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_empty_admissible_region_returns_nan_and_warns(d2pair):
    """A tiny rho window plus an absurd rate leaves no bound orbits at all."""
    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 20.0, R_obs=R)
    att.ra_rate *= 500.0
    att.dec_rate *= 500.0

    with pytest.warns(RuntimeWarning, match="empty admissible region"):
        res = classify(att, neo, other, R_obs=R, V_obs=V,
                       rho_min=0.01, rho_max=0.02)

    assert np.isnan(res.p_neo)
    assert np.isnan(res.log_lr)
    assert res.lambda_neo == 0.0 and res.lambda_other == 0.0
    assert res.grid.rho.size == 0
    assert res.n_nodes == 0
    assert res.node_density_neo.size == 0
    assert res.node_density_other.size == 0
    assert res.node_H.size == 0 and res.node_q.size == 0
    assert res.measure_fraction_H_above(20.0) == 0.0
    assert res.posterior_mass_where(np.zeros(0, dtype=bool)) == 0.0


def test_truncated_H_gives_zero_likelihoods_and_warns(d2pair):
    """V so faint that H > 25.5 everywhere: both components truncate to 0.

    At the anti-solar point the phase angle is exactly zero at every node
    (the object sits on the Sun-observer line), so
    H = V - 5 log10(rho * r) exactly, and with rho <= 10 AU the largest
    rho * r on the grid is ~110 -> H >= V - 10.2 = 29.8 > 25.5 (DESIGN §2.6).
    """
    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(0.3, 40.0, R_obs=R)

    with pytest.warns(RuntimeWarning, match="both marginal likelihoods are zero"):
        res = classify(att, neo, other, R_obs=R, V_obs=V,
                       rho_min=0.01, rho_max=10.0, n_rho=100, n_rhodot=50)

    assert res.grid.rho.size > 0  # the region is NOT empty; the density is 0
    assert res.lambda_neo == 0.0 and res.lambda_other == 0.0
    assert np.isnan(res.p_neo) and np.isnan(res.log_lr)
    assert np.all(res.node_density_neo == 0.0)
    # ... and the §2.6 diagnostic says exactly why.
    assert res.measure_fraction_H_above(25.5) == pytest.approx(1.0)


def test_one_sided_likelihood_gives_clean_zero_and_one(obs):
    """Lambda_other = 0 -> p_neo = 1 exactly, log_lr = +inf (no NaN, no warn)."""
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    neo, _ = _toy_pair()
    dead = BoxDensity("other", "qei", x=(1.3, 5.0), e=(0.0, 0.9),
                      i=(0.0, np.pi), H=(1e3, 1e4), amplitude=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning may be raised here
        res = classify(att, neo, dead, R_obs=R, V_obs=V,
                       rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    assert res.p_neo == 1.0
    assert res.log_lr == np.inf
    assert res.lambda_other == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res2 = classify(att, dead, neo, R_obs=R, V_obs=V,
                        rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    assert res2.p_neo == 0.0
    assert res2.log_lr == -np.inf


def test_overflowing_likelihoods_are_not_reported_as_zero(obs):
    """Both Lambdas overflow -> p = 0.5 with an accurate warning, not 'zero'."""
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    huge = 1e308
    neo = ConstantDensity("NEO", "qei", huge)
    other = ConstantDensity("other", "qei", huge)

    with pytest.warns(RuntimeWarning, match="overflowed") as rec:
        res = classify(att, neo, other, R_obs=R, V_obs=V,
                       rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    assert np.isinf(res.lambda_neo) and np.isinf(res.lambda_other)
    assert res.p_neo == 0.5
    assert res.log_lr == 0.0
    assert not any("are zero" in str(w.message) for w in rec)


def test_non_finite_v_mag_is_rejected_early(d2pair, obs):
    """A missing V must not masquerade as a support-truncation result."""
    neo, other = d2pair
    R, V = obs
    for bad in (np.nan, np.inf):
        att = opposition_tracklet(1.0, 21.0, R_obs=R)
        att.v_mag = bad
        with pytest.raises(ValueError, match="v_mag"):
            classify(att, neo, other, R_obs=R, V_obs=V,
                     n_rho=40, n_rhodot=20)


def test_bad_parameterization_raises(obs):
    R, V = obs
    att = ecliptic_tracklet(120.0, 0.4, 20.0, R_obs=R)
    neo, other = _toy_pair()
    other.parameterization = "cometary"
    with pytest.raises(ValueError, match="parameterization"):
        classify(att, neo, other, R_obs=R, V_obs=V,
                 rho_min=0.05, rho_max=6.0, n_rho=20, n_rhodot=10)


def test_observer_state_defaults_to_geocenter(d2pair):
    """Omitting R_obs/V_obs reproduces the geocentric observer_state call."""
    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.5, 21.0, R_obs=R)
    kw = dict(n_rho=60, n_rhodot=30)
    auto = classify(att, neo, other, **kw)
    explicit = classify(att, neo, other, R_obs=R, V_obs=V, **kw)
    assert auto.p_neo == pytest.approx(explicit.p_neo, rel=1e-12)


def test_obscode_passthrough(d2pair):
    """obscode/lon/lat/height are forwarded to observer_state."""
    neo, other = d2pair
    R_topo, V_topo = observer_state(EPOCH, obscode="X05")
    R_geo, _ = _observer()
    assert not np.allclose(R_topo, R_geo)  # the sites really do differ

    att = opposition_tracklet(1.0, 21.0, R_obs=R_geo)
    kw = dict(n_rho=60, n_rhodot=30)
    via_code = classify(att, neo, other, obscode="X05", **kw)
    explicit = classify(att, neo, other, R_obs=R_topo, V_obs=V_topo, **kw)
    assert via_code.lambda_neo == explicit.lambda_neo
    assert via_code.lambda_other == explicit.lambda_other

    lon, lat, hgt = -70.749417, -30.244633, 2647.0
    R_ll, V_ll = observer_state(EPOCH, lon_deg=lon, lat_deg=lat, height_m=hgt)
    via_ll = classify(att, neo, other, lon_deg=lon, lat_deg=lat,
                      height_m=hgt, **kw)
    ref_ll = classify(att, neo, other, R_obs=R_ll, V_obs=V_ll, **kw)
    assert via_ll.lambda_neo == ref_ll.lambda_neo


# ---------------------------------------------------------------------------
# 5. Batch API and diagnostics plot
# ---------------------------------------------------------------------------


def test_classify_many_matches_individual_calls(d2pair):
    neo, other = d2pair
    R, V = _observer()
    atts = [
        opposition_tracklet(0.2, 20.0, R_obs=R),
        ecliptic_tracklet(120.0, 0.6, 21.0, R_obs=R),
        opposition_tracklet(1.5, 21.0, R_obs=R),
    ]
    kw = dict(R_obs=R, V_obs=V, n_rho=100, n_rhodot=50)
    batch = classify_many(atts, neo, other, **kw)
    assert len(batch) == len(atts)
    for att, res in zip(atts, batch):
        one = classify(att, neo, other, **kw)
        assert res.p_neo == one.p_neo
        assert res.log_lr == one.log_lr
        assert res.lambda_neo == one.lambda_neo
        assert np.array_equal(res.node_density_neo, one.node_density_neo)


def test_classify_many_degenerate_results_are_nan_not_warning_counts(d2pair):
    """Warnings are deduplicated; nan p_neo is the reliable per-tracklet flag."""
    neo, other = d2pair
    R, V = _observer()
    atts = []
    for _ in range(5):
        a = opposition_tracklet(1.0, 20.0, R_obs=R)
        a.ra_rate *= 500.0
        a.dec_rate *= 500.0
        atts.append(a)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("default")
        batch = classify_many(atts, neo, other, R_obs=R, V_obs=V,
                              rho_min=0.01, rho_max=0.02)
    assert len(batch) == 5
    assert all(np.isnan(r.p_neo) for r in batch)
    assert all(r.n_nodes == 0 for r in batch)


def test_classify_many_empty_list(d2pair):
    neo, other = d2pair
    assert classify_many([], neo, other) == []


@pytest.mark.parametrize("component", ["neo", "other", "both"])
def test_plot_posterior_map_headless(d2pair, component):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V, n_rho=80, n_rhodot=40)

    ax = res.plot_posterior_map(component=component)
    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert ax.collections  # something was actually drawn
    assert ax.get_title()
    plt.close(ax.figure)


def test_plot_posterior_map_respects_a_supplied_axes(d2pair):
    """A caller-supplied Axes keeps its own title/scale; no colorbar by default."""
    import matplotlib.pyplot as plt

    neo, other = d2pair
    R, V = _observer()
    res = classify(opposition_tracklet(1.0, 21.0, R_obs=R), neo, other,
                   R_obs=R, V_obs=V, n_rho=80, n_rhodot=40)

    fig, ax = plt.subplots()
    ax.set_title("mine")
    n_axes_before = len(fig.axes)
    out = res.plot_posterior_map(ax=ax, component="both")
    assert out is ax
    assert ax.get_title() == "mine"          # not clobbered
    assert ax.get_xscale() == "linear"       # not clobbered
    assert len(fig.axes) == n_axes_before    # no colorbar axes added
    assert ax.get_xlabel()                   # empty labels are filled in

    # ... but the caller can ask for one.
    res.plot_posterior_map(ax=ax, component="both", colorbar=True)
    assert len(fig.axes) == n_axes_before + 1
    plt.close(fig)


def test_plot_posterior_map_colorbar_flag_on_owned_figure(d2pair):
    import matplotlib.pyplot as plt

    neo, other = d2pair
    R, V = _observer()
    res = classify(opposition_tracklet(1.0, 21.0, R_obs=R), neo, other,
                   R_obs=R, V_obs=V, n_rho=80, n_rhodot=40)

    ax = res.plot_posterior_map()  # owned -> colorbar by default
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)

    ax = res.plot_posterior_map(colorbar=False)
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_plot_posterior_map_bad_component(d2pair):
    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    res = classify(att, neo, other, R_obs=R, V_obs=V, n_rho=40, n_rhodot=20)
    with pytest.raises(ValueError, match="component"):
        res.plot_posterior_map(component="NEOs")


def test_plot_posterior_map_empty_region(d2pair):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 20.0, R_obs=R)
    att.ra_rate *= 500.0
    att.dec_rate *= 500.0
    with pytest.warns(RuntimeWarning):
        res = classify(att, neo, other, R_obs=R, V_obs=V,
                       rho_min=0.01, rho_max=0.02)
    ax = res.plot_posterior_map()
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_posterior_map_concentrates_near_the_truth(d2pair):
    """Collapse-test smoke check (DESIGN §4.1): the map's peak is where the
    orbit that generated the tracklet actually is.

    A synthetic main-belt object is placed at a known (rho, rho_dot) with the
    WP2 forward generator; the combined posterior map must peak within a factor
    of a few of the true range.  The generator's geometry is arbitrary (not
    opposition), so the phase angle is nonzero here too.
    """
    from digest42.transforms import observables_from_elements

    neo, other = d2pair
    R, V = _observer()

    elements = np.array([2.6, 0.15, np.radians(8.0), 1.1, 2.2, 0.7])
    obsv = observables_from_elements(elements, 17.0, R, V)
    att = Attributable(
        ra=float(obsv["ra"]), dec=float(obsv["dec"]),
        ra_rate=float(obsv["ra_rate"]), dec_rate=float(obsv["dec_rate"]),
        epoch=EPOCH, v_mag=float(obsv["v_mag"]),
    )
    res = classify(att, neo, other, R_obs=R, V_obs=V, n_rho=600, n_rhodot=50)

    mass = (res.node_density_neo + res.node_density_other) * res.grid.weight
    assert mass.sum() > 0
    peak_rho = float(res.grid.rho[int(np.argmax(mass))])
    true_rho = float(obsv["rho"])
    assert 0.5 < peak_rho / true_rho < 2.0, (peak_rho, true_rho)
    # ... and the object is (correctly) not a NEO.
    assert res.p_neo < 0.1, res.p_neo
