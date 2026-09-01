"""WP5 unit tests — the classifier (marginal likelihoods, posterior, maps).

Acceptance criteria (IMPLEMENTATION.md WP5):
  1. hand-constructed obvious cases with the real digest2 model score as
     expected (fast opposition tracklet -> P(NEO) ~ 1; slow one -> ~ 0),
  2. results are invariant under grid refinement to a stated tolerance,
  3. the *machinery* (weights, masking, Jacobian pairing) is verified against a
     brute-force integration with analytically known toy densities, so the test
     does not depend on the digest2 file at all,
  4. edge cases (empty admissible region, everything truncated away, NaN
     nodes) degrade to nan + a warning rather than an exception,
  5. classify_many matches per-tracklet classify, and the posterior map renders
     headless.

Design note: the synthetic-density tests deliberately re-derive Lambda_C with
an explicit per-node Python loop and, separately, with an independent
quadrature rule on a *linear* rho grid, so a bookkeeping bug in the vectorized
classifier cannot hide behind the same bug in the test.
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
from digest42.classifier import ClassificationResult, classify, classify_many
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


# ---------------------------------------------------------------------------
# Helpers: hand-constructed opposition tracklets
# ---------------------------------------------------------------------------


def _observer():
    """Geocentric heliocentric state at EPOCH, equatorial frame."""
    return observer_state(EPOCH, obscode="500")


def opposition_tracklet(rate_deg_per_day: float, v_mag: float, R_obs=None):
    """A tracklet at the anti-solar point moving retrograde along the ecliptic.

    At opposition the line of sight is the anti-Sun direction, which for a
    heliocentric observer position ``R_obs`` is simply ``+R_obs / |R_obs|``:
    the object sits on the Sun-Earth line beyond the Earth, so the phase angle
    is exactly zero at every node of the ranging grid (a useful property for
    the truncation test below).

    The apparent motion is placed along the ecliptic in the direction of
    *decreasing* ecliptic longitude — the retrograde loop that any prograde
    heliocentric orbit shows at opposition — with the given total on-sky rate.
    The tangent basis is orthogonal but not orthonormal (|du/dra| = cos dec),
    hence the cos^2(dec) in ra_rate.
    """
    if R_obs is None:
        R_obs, _ = _observer()
    u = np.asarray(R_obs, dtype=float)
    u = u / np.linalg.norm(u)
    ra = float(np.arctan2(u[1], u[0]) % (2.0 * np.pi))
    dec = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))

    # North ecliptic pole in equatorial components (see transforms.state_equ_to_ecl).
    pole = np.array([0.0, -np.sin(OBLIQUITY_J2000), np.cos(OBLIQUITY_J2000)])
    # +(pole x u) points along increasing ecliptic longitude; retrograde is -.
    s = -np.cross(pole, u)
    s = s / np.linalg.norm(s)

    _, du_dra, du_ddec = tangent_basis(ra, dec)
    rate = np.radians(rate_deg_per_day)
    ra_rate = float(rate * np.dot(s, du_dra) / np.cos(dec) ** 2)
    dec_rate = float(rate * np.dot(s, du_ddec))

    return Attributable(
        ra=ra, dec=dec, ra_rate=ra_rate, dec_rate=dec_rate,
        epoch=EPOCH, v_mag=float(v_mag),
    )


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

    def density(self, elements3, H):
        elements3 = np.asarray(elements3, dtype=float)
        x, e, i = elements3[..., 0], elements3[..., 1], elements3[..., 2]
        x, e, i, h = np.broadcast_arrays(x, e, i, np.asarray(H, dtype=float))
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


def _toy_pair(parameterization="qei", amp_neo=1.0, amp_other=3.0):
    """Two toy components partitioning at x = 1.3 AU (cf. DESIGN §2.4/D-2)."""
    common = dict(e=(0.0, 0.9), i=(0.0, 0.5 * np.pi), H=(10.0, 25.5))
    neo = BoxDensity("NEO", parameterization, x=(0.0, 1.3),
                     amplitude=amp_neo, **common)
    other = BoxDensity("other", parameterization, x=(1.3, 5.0),
                       amplitude=amp_other, **common)
    return neo, other


def _brute_force_lambdas(att, densities, grid, R_obs, V_obs, G=G_SLOPE_DEFAULT):
    """Lambda_C on the SAME nodes, computed one node at a time.

    Deliberately unvectorized and written independently of classifier.py: it
    checks the bookkeeping (which weight goes with which node, which Jacobian
    parameterization goes with which density, how NaN nodes are masked).
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
                            n_rho, n_rhodot, G=G_SLOPE_DEFAULT):
    """Lambda_C by a *different* quadrature: LINEAR-rho midpoint rule.

    classify() integrates on a log-spaced rho grid; this uses uniformly spaced
    rho with midpoint nodes and the same per-rho admissible rho_dot interval,
    so the two agree only if both are integrating the same measure.
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
    rho = np.repeat(rho_v, n_rhodot)
    rho_dot = (lo_v[:, None] + d_rhodot[:, None] * off[None, :]).ravel()
    weight = np.repeat(d_rho_v * d_rhodot, n_rhodot)

    r_equ, v_equ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rho_dot, R_obs, V_obs
    )
    el = elements_from_state(state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ))
    good = np.all(np.isfinite(el), axis=-1)
    r_helio = np.linalg.norm(r_equ, axis=-1)
    H = absolute_magnitude(att.v_mag, rho, r_helio, phase_angle(r_equ, R_obs), G=G)

    out = []
    for dens in densities:
        param = dens.parameterization
        det = ranging_jacobian_det(
            att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rho_dot,
            R_obs, V_obs, parameterization=param,
        )
        el_s = np.where(good[:, None], el, 0.0)
        el3 = qei_from_elements(el_s) if param == "qei" else el_s[..., 0:3]
        n_c = dens.density(el3, np.where(good, H, 0.0))
        integ = np.where(good & np.isfinite(det), n_c * np.nan_to_num(det), 0.0)
        out.append(float(np.sum(integ * weight)))
    return out


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
                 R_obs=R, V_obs=V).p_neo
        for rate in (0.2, 0.5, 1.5)
    ]
    assert ps[0] < ps[1] < ps[2], ps


# ---------------------------------------------------------------------------
# 2. Grid-refinement invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate_deg, v_mag", [(1.5, 21.0), (0.2, 20.0)]
)
def test_grid_refinement_invariance(d2pair, obs, rate_deg, v_mag):
    """p_neo must move by < 0.01 between (200, 100) and (400, 200)."""
    neo, other = d2pair
    R, V = obs
    att = opposition_tracklet(rate_deg, v_mag, R_obs=R)
    coarse = classify(att, neo, other, R_obs=R, V_obs=V,
                      n_rho=200, n_rhodot=100)
    fine = classify(att, neo, other, R_obs=R, V_obs=V,
                    n_rho=400, n_rhodot=200)
    assert abs(fine.p_neo - coarse.p_neo) < 0.01, (coarse.p_neo, fine.p_neo)
    # The likelihoods themselves converge too (looser: they are absolute
    # numbers dominated by a piecewise-constant, i.e. discontinuous, integrand).
    assert fine.lambda_other == pytest.approx(coarse.lambda_other, rel=0.25)


# ---------------------------------------------------------------------------
# 3. Synthetic-density machinery tests (no digest2 file involved)
# ---------------------------------------------------------------------------


def test_toy_density_matches_brute_force_same_grid(obs):
    """Exact bookkeeping check: same nodes, explicit per-node loop."""
    R, V = obs
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    neo, other = _toy_pair()

    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=60, n_rhodot=30)
    res = classify(att, neo, other, R_obs=R, V_obs=V, **kw)
    grid = build_grid(att, R, V, **kw)

    assert np.allclose(grid.rho, res.grid.rho)
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
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    neo, other = _toy_pair()

    res = classify(att, neo, other, R_obs=R, V_obs=V,
                   rho_min=0.05, rho_max=6.0, n_rho=400, n_rhodot=200)
    ref_neo, ref_other = _independent_quadrature(
        att, (neo, other), R, V, rho_min=0.05, rho_max=6.0,
        n_rho=1500, n_rhodot=300,
    )
    p_ref = ref_neo / (ref_neo + ref_other)
    assert 0.01 < p_ref < 0.99, p_ref  # a non-degenerate comparison
    assert res.p_neo == pytest.approx(p_ref, abs=0.01), (res.p_neo, p_ref)
    # The absolute likelihoods must also agree, not just their ratio.
    assert res.lambda_neo == pytest.approx(ref_neo, rel=0.05)
    assert res.lambda_other == pytest.approx(ref_other, rel=0.05)


def test_toy_amplitude_scaling_is_exact():
    """Doubling a component's absolute calibration must scale its Lambda."""
    R, V = _observer()
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=80, n_rhodot=40)

    a_neo, a_other = _toy_pair(amp_neo=1.0, amp_other=1.0)
    b_neo, b_other = _toy_pair(amp_neo=2.0, amp_other=1.0)
    ra = classify(att, a_neo, a_other, R_obs=R, V_obs=V, **kw)
    rb = classify(att, b_neo, b_other, R_obs=R, V_obs=V, **kw)

    assert ra.lambda_neo > 0.0 and ra.lambda_other > 0.0
    assert rb.lambda_neo == pytest.approx(2.0 * ra.lambda_neo, rel=1e-12)
    assert rb.lambda_other == pytest.approx(ra.lambda_other, rel=1e-12)
    assert rb.log_lr == pytest.approx(ra.log_lr + np.log(2.0), rel=1e-12)


def test_aei_parameterization_uses_the_aei_jacobian():
    """A stub density tagged 'aei' must get |J|_aei, not |J|_qei.

    The two differ by exactly the (a, e) -> (q, e) factor (1 - e), so pairing
    the wrong Jacobian with the density is detectable at the Lambda level.
    """
    R, V = _observer()
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=6.0, n_rho=80, n_rhodot=40)

    # One component in each parameterization, both "everything is 1" boxes over
    # a range wide enough that the box edges never bind: then
    #   Lambda_qei / Lambda_aei = <1 - e> weighted by |J|_aei, which is < 1.
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
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
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
    assert res.node_density_neo.size == 0
    assert res.node_density_other.size == 0


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

    with pytest.warns(RuntimeWarning, match="both marginal likelihoods"):
        res = classify(att, neo, other, R_obs=R, V_obs=V,
                       rho_min=0.01, rho_max=10.0, n_rho=100, n_rhodot=50)

    assert res.grid.rho.size > 0  # the region is NOT empty; the density is 0
    assert res.lambda_neo == 0.0 and res.lambda_other == 0.0
    assert np.isnan(res.p_neo) and np.isnan(res.log_lr)
    assert np.all(res.node_density_neo == 0.0)


def test_one_sided_likelihood_gives_clean_zero_and_one():
    """Lambda_other = 0 -> p_neo = 1 exactly, log_lr = +inf (no NaN, no warn)."""
    R, V = _observer()
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
    # "other" support is empty by construction (H window nobody can reach).
    neo, _ = _toy_pair()
    dead = BoxDensity("other", "qei", x=(1.3, 5.0), e=(0.0, 0.9),
                      i=(0.0, 0.5 * np.pi), H=(1e3, 1e4), amplitude=1.0)

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


def test_unbound_nodes_are_masked_to_exactly_zero(monkeypatch, d2pair):
    """NaN-element nodes contribute nothing and do not poison the sums.

    build_grid only ever places nodes strictly inside the bound-orbit interval,
    so to exercise the mask we inject extra nodes at rho_dot far outside that
    interval (hyperbolic by construction) and check that (a) the Lambdas are
    bit-for-bit those of the un-augmented grid and (b) the injected nodes'
    per-node records are exactly 0.
    """
    import digest42.classifier as clf

    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    kw = dict(rho_min=0.05, rho_max=8.0, n_rho=60, n_rhodot=30)

    baseline = classify(att, neo, other, R_obs=R, V_obs=V, **kw)

    real_build = clf.build_grid
    n_extra = 7

    def augmented(att_, R_, V_, **kwargs):
        g = real_build(att_, R_, V_, **kwargs)
        rho_extra = np.geomspace(0.1, 5.0, n_extra)
        _, hi = admissible_rhodot_interval(att_, R_, V_, rho_extra)
        rho_dot_extra = np.nan_to_num(hi, nan=0.0) + 5.0  # way past escape
        return RangingGrid(
            rho=np.concatenate([g.rho, rho_extra]),
            rho_dot=np.concatenate([g.rho_dot, rho_dot_extra]),
            weight=np.concatenate([g.weight, np.full(n_extra, 1.0)]),
            n_rho=g.n_rho,
            n_rhodot=g.n_rhodot,
        )

    monkeypatch.setattr(clf, "build_grid", augmented)
    res = classify(att, neo, other, R_obs=R, V_obs=V, **kw)

    # The injected nodes really are unbound.
    r_equ, v_equ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate,
        res.grid.rho[-n_extra:], res.grid.rho_dot[-n_extra:], R, V,
    )
    el = elements_from_state(state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ))
    assert np.all(np.isnan(el))

    assert np.all(res.node_density_neo[-n_extra:] == 0.0)
    assert np.all(res.node_density_other[-n_extra:] == 0.0)
    # (equal up to the summation-order round-off that appending nodes causes)
    assert res.lambda_neo == pytest.approx(baseline.lambda_neo, rel=1e-14)
    assert res.lambda_other == pytest.approx(baseline.lambda_other, rel=1e-14)
    assert res.p_neo == pytest.approx(baseline.p_neo, rel=1e-14)


def test_all_real_grid_nodes_are_bound(d2pair):
    """Documented property of WP3: the grid itself never places a NaN node."""
    R, V = _observer()
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    grid = build_grid(att, R, V, rho_min=0.01, rho_max=100.0,
                      n_rho=200, n_rhodot=100)
    r_equ, v_equ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate, grid.rho, grid.rho_dot, R, V
    )
    el = elements_from_state(state_equ_to_ecl(r_equ), state_equ_to_ecl(v_equ))
    assert np.all(np.isfinite(el))


def test_bad_parameterization_raises():
    R, V = _observer()
    att = opposition_tracklet(0.2, 20.0, R_obs=R)
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


def test_G_none_equals_default_slope(d2pair):
    neo, other = d2pair
    R, V = _observer()
    att = opposition_tracklet(1.0, 21.0, R_obs=R)
    kw = dict(R_obs=R, V_obs=V, n_rho=60, n_rhodot=30)
    a = classify(att, neo, other, G=None, **kw)
    b = classify(att, neo, other, G=G_SLOPE_DEFAULT, **kw)
    assert a.lambda_neo == b.lambda_neo
    assert a.lambda_other == b.lambda_other


# ---------------------------------------------------------------------------
# 5. Batch API and diagnostics plot
# ---------------------------------------------------------------------------


def test_classify_many_matches_individual_calls(d2pair):
    neo, other = d2pair
    R, V = _observer()
    atts = [
        opposition_tracklet(0.2, 20.0, R_obs=R),
        opposition_tracklet(0.6, 21.0, R_obs=R),
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
    plt.close(ax.figure)

    fig, ax2 = plt.subplots()
    out = res.plot_posterior_map(ax=ax2, component=component)
    assert out is ax2
    plt.close(fig)


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
    of a few of the true range.
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
    res = classify(att, neo, other, R_obs=R, V_obs=V, n_rho=300, n_rhodot=150)

    mass = (res.node_density_neo + res.node_density_other) * res.grid.weight
    assert mass.sum() > 0
    peak_rho = float(res.grid.rho[int(np.argmax(mass))])
    true_rho = float(obsv["rho"])
    assert 0.5 < peak_rho / true_rho < 2.0, (peak_rho, true_rho)
    # ... and the object is (correctly) not a NEO.
    assert res.p_neo < 0.1, res.p_neo
