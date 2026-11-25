"""Cross-validation of the Rust core kernel against the Python reference.

The two backends compute the same integral by *different* routes — the Python
path builds and inverts the 6x6 ``d(elements)/d(state)`` chain per node, the
Rust path evaluates the composed determinant in the exact Delaunay closed form
and needs no Kepler solver — so agreement here is a two-sided check, not a
tautology.

Layout of the file:

1. kernel-level parity: bin lookup (bit-exact, including edges, the D-14 clamp,
   NaN/inf), rho node placement (bit-exact), admissible intervals, elements,
   H(V, rho), and the closed-form |J| against ``transforms.ranging_jacobian_det``;
2. end-to-end parity of :func:`pneo.classify` on ~200 diverse tracklets;
3. the sigma-point (K > 1) path against an explicit Python-side weighted mean;
4. per-node diagnostics arrays;
5. a representative subset of the existing WP5 tests re-run under both backends
   through the ``backend`` fixture;
6. regression tests for each finding of the adversarial review — hostile grid
   sizes, the bin-shape overflow, the bounds-checked table read, the
   content-aware model cache, ``PNEO_BACKEND=auto`` semantics, the
   perihelion coverage of the conditioning prefilter, and the one **expected**
   divergence (the reference's Jacobian at ``|dec| = 90`` deg exactly, recorded
   as a strict xfail because the Rust value is the correct one there).

Tolerances achieved on this suite are far inside the acceptance bar
(|dp_neo| <= 1e-6, Lambda relative <= 1e-5); see the constants below.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
from astropy.time import Time

from pneo import transforms as T
from pneo.attributable import Attributable
from pneo.backend import core_available, resolve_backend, rho_axis
from pneo.classifier import classify, classify_many
from pneo.constants import G_SLOPE_DEFAULT, MU_SUN
from pneo.population import (
    E_EDGES,
    H_EDGES,
    I_EDGES,
    Q_EDGES,
    Digest2Histogram,
)
from pneo.ranging import admissible_rhodot_interval, build_grid

pytestmark = pytest.mark.skipif(
    not core_available(),
    reason="pneo._core is not built (run rust/build.sh)",
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
from model_location import MODEL_PATH  # noqa: E402
GATE_GEN = os.path.join(REPO, "validation", "output", "gate_v0", "gen.npz")

# Acceptance bars from the approved design (issue #1).
P_TOL = 1e-6
LAM_RTOL = 1e-5
#: The tolerance the two Jacobian routes actually meet away from the guard.
JAC_RTOL = 1e-9

# A cheap grid for the many-tracklet sweeps; the production (1200, 50) is used
# for a handful of spot checks only, to keep this file under a few minutes.
FAST = dict(n_rho=180, n_rhodot=12)
PROD = dict(n_rho=1200, n_rhodot=50)


@pytest.fixture(autouse=True)
def _isolate_backend_env(monkeypatch):
    """Ignore any inherited ``PNEO_BACKEND``.

    This file drives both backends explicitly through the ``backend=``
    argument; a global override would silently turn every parity comparison
    into a self-comparison (and block the tests that need the kernel).  The
    two tests that care about the environment variable set it themselves, and
    the ``backend`` fixture below sets it after this one has cleared it.
    """
    monkeypatch.delenv("PNEO_BACKEND", raising=False)


@pytest.fixture(scope="module")
def d2pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(MODEL_PATH)


@pytest.fixture(scope="module")
def clamped_pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(MODEL_PATH, h_top_bin_infinite=True)


@pytest.fixture(scope="module")
def core():
    from pneo import _core

    return _core


# ---------------------------------------------------------------------------
# Tracklet zoo
# ---------------------------------------------------------------------------


def _gate_tracklets(n: int, seed: int = 4242):
    """`n` tracklets drawn from the stored gate_v0 catalogue, with observers."""
    if not os.path.exists(GATE_GEN):  # pragma: no cover
        pytest.skip("gate_v0 catalogue not present")
    g = np.load(GATE_GEN)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(g["ra"]), size=min(n, len(g["ra"])), replace=False)
    out = []
    for k in idx:
        att = Attributable(
            ra=float(g["ra"][k]),
            dec=float(g["dec"][k]),
            ra_rate=float(g["ra_rate"][k]),
            dec_rate=float(g["dec_rate"][k]),
            epoch=Time(float(g["epoch_jd_tdb"][k]), format="jd", scale="tdb"),
            v_mag=float(g["v_mag"][k]),
        )
        out.append((att, g["R_obs"][k].copy(), g["V_obs"][k].copy()))
    return out


def _synthetic_corners():
    """Hand-built corner cases: off-opposition, faint, fast, empty, truncated."""
    epoch = Time(2460000.5, format="jd", scale="tdb")
    R = np.array([0.4, -0.85, -0.37])
    V = np.array([0.0155, 0.0062, 0.0027])
    cases = []
    for name, dec, rate, v_mag in [
        ("opposition_slow", 0.0, 0.002, 20.0),
        ("opposition_fast", 0.0, 0.035, 21.5),
        ("high_dec_fast", 1.15, 0.030, 22.0),
        ("south_slow", -0.9, 0.0015, 19.0),
        ("very_faint", 0.25, 0.010, 25.4),
        ("beyond_H_support", 0.25, 0.010, 30.0),   # fully truncated in H
        ("bright", 0.25, 0.010, 12.0),
        ("near_ecliptic_pole", 1.4, 0.006, 21.0),
        # High declination: the reference's composed determinant degrades
        # smoothly from ~89.99 deg (see the xfail below), so the end-to-end
        # comparison is taken just inside that, at 89.9 deg.
        ("high_dec_89p9", np.radians(89.9), 0.004, 21.0),
        ("huge_rate_empty", 0.0, 2.0, 20.0),       # empty admissible region
        ("retrograde_like", 0.3, 0.05, 21.0),
    ]:
        ra = 1.9 + 0.13 * len(cases)
        cases.append(
            (
                name,
                Attributable(
                    ra=ra, dec=dec, ra_rate=rate / max(np.cos(dec), 1e-3),
                    dec_rate=0.3 * rate, epoch=epoch, v_mag=v_mag,
                ),
                R.copy(),
                V.copy(),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# 1. Kernel-level parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edges_name", ["q", "e", "i", "h"]
)
def test_bin_index_is_bit_exact(core, edges_name):
    """`_core.bin_index` reproduces `population._bin_index` exactly."""
    from pneo.population import _bin_index

    edges = {"q": Q_EDGES, "e": E_EDGES, "i": I_EDGES, "h": H_EDGES}[edges_name]
    rng = np.random.default_rng(7)
    lo, hi = float(edges[0]), float(edges[-1])
    span = hi - lo
    probes = np.concatenate(
        [
            np.asarray(edges, dtype=float),                      # exact edges
            np.nextafter(edges, -np.inf),                        # just below
            np.nextafter(edges, +np.inf),                        # just above
            rng.uniform(lo - 0.1 * span, hi + 0.1 * span, 4000),
            np.array([
                -0.0, 0.0, -1e-300, 1e-300, -1e30, 1e30,
                np.inf, -np.inf, np.nan,
            ]),
        ]
    )
    idx_py, ok_py = _bin_index(np.asarray(edges, dtype=float), probes)
    want = np.where(ok_py, idx_py, -1).astype(np.int64)
    got = core.bin_index(np.ascontiguousarray(edges, dtype=float), probes)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("clamp", [False, True])
def test_density_lookup_is_bit_exact(core, clamp):
    """The Rust density equals `Digest2Histogram.density` bit for bit.

    Probes hit every bin edge from both sides plus the D-14 clamp branch
    (H >= 25.5 -> top bin when clamped, 0 otherwise), +/-inf and NaN.
    """
    rng = np.random.default_rng(11)
    counts = rng.random((29, 8, 11, 18)) * 1e4
    counts[counts < 3e3] = 0.0  # keep genuinely empty bins in the mix
    neo = Digest2Histogram(counts, "NEO", h_top_bin_infinite=clamp)
    other = Digest2Histogram(
        counts[::-1].copy(), "other", h_top_bin_infinite=clamp
    )
    from pneo.backend import core_model_for

    cm = core_model_for(neo, other)

    def probes(edges, n):
        e = np.asarray(edges, dtype=float)
        return np.concatenate(
            [
                e,
                np.nextafter(e, -np.inf),
                np.nextafter(e, +np.inf),
                rng.uniform(e[0] - 0.5, e[-1] + 0.5, n),
                np.array([np.inf, -np.inf, np.nan, -0.0]),
            ]
        )

    q = probes(Q_EDGES, 800)
    e = probes(E_EDGES, 800)
    i = probes(I_EDGES, 800)
    h = probes(H_EDGES, 800)
    n = min(len(q), len(e), len(i), len(h))
    q, e, i, h = q[:n], e[:n], i[:n], h[:n]
    # shuffle so combinations are not edge-aligned across all four axes at once
    for arr in (e, i, h):
        rng.shuffle(arr)

    el3 = np.ascontiguousarray(np.stack([q, e, i], axis=-1))
    want_neo = neo.density(el3, h)
    want_other = other.density(el3, h)

    got_neo, got_other = cm.density_at(el3, np.ascontiguousarray(h))
    assert np.array_equal(got_neo, want_neo)
    assert np.array_equal(got_other, want_other)
    # Not vacuous: the probe set must hit both in-support and out-of-support.
    assert (want_neo > 0.0).any() and (want_neo == 0.0).any()
    if clamp:
        above = h >= H_EDGES[-1]
        assert above.any()
        assert (want_neo[above] > 0.0).any()


def test_rho_axis_matches_build_grid(d2pair):
    """`backend.rho_axis` is bit-identical to the rho half of `build_grid`.

    The kernel no longer generates the rho axis: `rho_axis` computes it with the
    same `np.geomspace` call `ranging.build_grid` makes, and the array is handed
    across the boundary.  This test is what keeps that duplicated ten lines
    honest, and it checks against `build_grid`'s **actual output** rather than
    against a restatement of its formula.

    (It replaces a test that asserted the *Rust* node placement was bit-equal to
    `np.geomspace`.  That contract is not portable: `geomspace` is
    `10 ** linspace(...)`, `pow` is not IEEE-754 correctly rounded, and numpy
    dispatches its `power` loop on runtime CPU features -- so the doubles depend
    on the machine and the libm, not just on the numpy version.  CI on a
    different CPU caught it.)
    """
    from pneo.backend import rho_axis

    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=11)[0]
    for rho_min, rho_max, n_rho, n_rhodot in [
        (0.01, 100.0, 1200, 50),
        (0.01, 100.0, 180, 12),
        (0.01, 100.0, 1, 3),
        (0.05, 5.0, 2, 7),
        (1e-3, 1e3, 137, 5),
        (0.5, 0.500001, 9, 4),
        (0.01, 100.0, 2, 1),
    ]:
        nodes, d_rho = rho_axis(rho_min, rho_max, n_rho)
        assert nodes.shape == d_rho.shape == (n_rho,)
        # The cells tile [rho_min, rho_max] exactly -- build_grid's documented
        # invariant, and the one property that must hold on every platform.
        assert d_rho.sum() == rho_max - rho_min
        assert np.all(nodes > 0.0) and np.all(np.isfinite(nodes))
        if n_rho > 1:
            assert np.all(np.diff(nodes) > 0.0)
            assert nodes[0] == rho_min and nodes[-1] == rho_max

        # ... and it agrees with build_grid node for node, bit for bit.
        grid = build_grid(att, R, V, rho_min, rho_max, n_rho, n_rhodot)
        lo, hi = admissible_rhodot_interval(att, R, V, nodes)
        width = hi - lo
        valid = np.isfinite(width) & (width > 0.0)
        assert np.array_equal(grid.rho, np.repeat(nodes[valid], n_rhodot))
        assert np.array_equal(
            grid.weight,
            np.repeat(d_rho[valid] * (width[valid] / n_rhodot), n_rhodot),
        )

    # The cache hands back the same read-only arrays, so no caller can corrupt
    # the axis another tracklet will use.
    a1, w1 = rho_axis(0.01, 100.0, 180)
    a2, w2 = rho_axis(0.01, 100.0, 180)
    assert a1 is a2 and w1 is w2
    assert not a1.flags.writeable and not w1.flags.writeable
    with pytest.raises(ValueError):
        a1[0] = 1.0
    # Same validation, same messages, as build_grid.
    for bad in [(0.0, 100.0, 10), (-1.0, 100.0, 10), (1.0, 0.5, 10),
                (0.01, 100.0, 0), (0.01, 100.0, -3)]:
        with pytest.raises(ValueError):
            rho_axis(*bad)


def test_parity_survives_a_differently_rounding_geomspace(d2pair, monkeypatch):
    """The regression test for the CI failure this design fixes.

    Simulates a numpy whose `geomspace` lands on different doubles, by swapping
    in an equally legitimate formulation of `10 ** y` (`exp(y ln 10)`, measured
    to differ from `np.power(10, y)` by up to 7 ulp on 881 of 1200 nodes).  That
    is a faithful stand-in for what CI hit: `pow` is not IEEE-754 correctly
    rounded, and numpy dispatches its `power` loop on runtime CPU features, so a
    different machine legitimately produces a different axis.

    Under the old design (kernel regenerates the axis) this diverges.  Under the
    current one both sides consume the same array, so the grid stays bit-exact
    and the parity bars hold -- on any numpy, any CPU, any libm.
    """
    from pneo import ranging
    from pneo.backend import rho_axis

    real_geomspace = np.geomspace

    def hostile_geomspace(lo, hi, num, **kw):
        y = np.linspace(np.log10(lo), np.log10(hi), num)
        r = np.exp(y * np.log(10.0))
        if num > 0:
            r[0] = lo
            if num > 1:
                r[-1] = hi
        return r

    # Sanity: the stand-in really does move the nodes.
    a = real_geomspace(0.01, 100.0, 1200)
    b = hostile_geomspace(0.01, 100.0, 1200)
    n_moved = int((a != b).sum())
    max_ulp = int(np.abs(a.view(np.int64) - b.view(np.int64))[a != b].max())
    assert n_moved > 500 and max_ulp >= 2, (n_moved, max_ulp)

    neo, other = d2pair
    monkeypatch.setattr(np, "geomspace", hostile_geomspace)
    monkeypatch.setattr(ranging.np, "geomspace", hostile_geomspace)
    rho_axis.cache_clear()
    try:
        for cfg in (FAST, PROD):
            for att, R, V in _gate_tracklets(3, seed=606):
                want = build_grid(att, R, V, 0.01, 100.0, **cfg)
                rr = classify(att, neo, other, R, V, backend="rust",
                              diagnostics=True, **cfg)
                rp = classify(att, neo, other, R, V, backend="python", **cfg)
                # The moved axis is still shared, so the grid is still exact ...
                assert np.array_equal(rr.grid.rho, want.rho)
                assert np.array_equal(rr.grid.rho_dot, want.rho_dot)
                assert np.array_equal(rr.grid.weight, want.weight)
                # ... and so is the whole parity contract.
                _compare(rp, rr, f"hostile-geomspace {cfg}")
    finally:
        rho_axis.cache_clear()
    assert np.geomspace is hostile_geomspace  # monkeypatch still in force
    # And the axis genuinely moved for the run above, i.e. this was not vacuous.
    nodes, _ = rho_axis(0.01, 100.0, 1200)
    rho_axis.cache_clear()
    assert not np.array_equal(nodes, real_geomspace(0.01, 100.0, 1200))


@pytest.mark.parametrize("cfg", [FAST, PROD], ids=["fast", "production"])
def test_grid_is_reproduced_bit_exactly(d2pair, cfg):
    """The kernel's grid == `build_grid`'s, node for node.

    The rho axis is exact by construction (both sides consume the array
    `backend.rho_axis` produced).  What this still tests is everything the
    kernel derives itself: the admissible-interval quadratic including its
    NaN/empty convention, the geometric cell edges, and the midpoint rho_dot
    placement -- all of which use only the IEEE-754-exact operations
    (`+ - * /`, `sqrt`), so they are bit-exact portably.

    Parametrized over BOTH resolutions on purpose.  The predecessor of this test
    ran at n_rho = 180 only, and the geomspace portability bug that CI caught
    showed up at n_rho = 1200 -- 180 passing was luck (fewer exponents sampled,
    so a smaller chance of hitting one where numpy's `pow` and the kernel's
    disagreed), not safety.
    """
    neo, other = d2pair
    n = 25 if cfg is FAST else 4
    for att, R, V in _gate_tracklets(n, seed=5):
        want = build_grid(att, R, V, 0.01, 100.0, **cfg)
        got = classify(
            att, neo, other, R, V, backend="rust", diagnostics=True, **cfg
        )
        assert np.array_equal(got.grid.rho, want.rho)
        assert np.array_equal(got.grid.rho_dot, want.rho_dot)
        assert np.array_equal(got.grid.weight, want.weight)


def test_admissible_interval_convention_on_empty_rows(d2pair, core):
    """Rows where the quadratic has no solution contribute no nodes."""
    neo, other = d2pair
    epoch = Time(2460000.5, format="jd", scale="tdb")
    R = np.array([0.4, -0.85, -0.37])
    V = np.array([0.0155, 0.0062, 0.0027])
    att = Attributable(ra=1.0, dec=0.2, ra_rate=0.6, dec_rate=0.2,
                       epoch=epoch, v_mag=20.0)
    from pneo.backend import rho_axis

    rho_nodes, _ = rho_axis(0.01, 100.0, FAST["n_rho"])
    lo, hi = admissible_rhodot_interval(att, R, V, rho_nodes)
    width = hi - lo
    n_valid = int(np.count_nonzero(np.isfinite(width) & (width > 0.0)))
    assert n_valid < FAST["n_rho"], "test geometry must truncate some rows"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = classify(att, neo, other, R, V, backend="rust", **FAST)
    assert got.n_nodes == n_valid * FAST["n_rhodot"]


def test_node_H_and_node_q_match_the_reference(d2pair):
    """Per-node H(V, rho) and q = a(1-e) agree with the numpy pipeline."""
    neo, other = d2pair
    for att, R, V in _gate_tracklets(12, seed=6):
        rp = classify(att, neo, other, R, V, backend="python", **FAST)
        rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        assert np.array_equal(np.isnan(rp.node_H), np.isnan(rr.node_H))
        assert np.array_equal(np.isnan(rp.node_q), np.isnan(rr.node_q))
        mh = np.isfinite(rp.node_H)
        np.testing.assert_allclose(
            rr.node_H[mh], rp.node_H[mh], rtol=1e-14, atol=0.0
        )
        mq = np.isfinite(rp.node_q)
        np.testing.assert_allclose(
            rr.node_q[mq], rp.node_q[mq], rtol=1e-12, atol=0.0
        )


def test_closed_form_jacobian_matches_the_6x6(core):
    """|J| closed form vs `ranging_jacobian_det` on randomized geometries.

    Nodes near the WP2 conditioning guard are excluded (that is the one place
    the two routes are *designed* to differ); everywhere else they agree to
    `JAC_RTOL`.
    """
    rng = np.random.default_rng(19)
    n_bad = 0
    n_ok = 0
    for _ in range(40):
        ra = rng.uniform(0.0, 2.0 * np.pi)
        dec = np.arcsin(rng.uniform(-1.0, 1.0))
        ra_rate = rng.normal(0.0, 0.01)
        dec_rate = rng.normal(0.0, 0.01)
        R = rng.normal(0.0, 1.0, 3)
        R = R / np.linalg.norm(R) * rng.uniform(0.9, 1.05)
        V = rng.normal(0.0, 0.008, 3)
        rho = np.geomspace(0.02, 60.0, 400)
        lo, hi = admissible_rhodot_interval(
            Attributable(ra, dec, ra_rate, dec_rate,
                         Time(2460000.5, format="jd", scale="tdb"), 20.0),
            R, V, rho,
        )
        w = hi - lo
        good_rows = np.isfinite(w) & (w > 0.0)
        if not np.any(good_rows):
            continue
        rho = rho[good_rows]
        rho_dot = lo[good_rows] + 0.5 * w[good_rows]

        r_equ, v_equ = T.topocentric_to_heliocentric(
            ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V
        )
        el = T.elements_from_state(
            T.state_equ_to_ecl(r_equ), T.state_equ_to_ecl(v_equ)
        )
        for param in ("aei", "qei"):
            det = T.ranging_jacobian_det(
                ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V,
                parameterization=param,
            )
            for k in range(len(rho)):
                a, e, inc = el[k, 0], el[k, 1], el[k, 2]
                if not np.all(np.isfinite(el[k])):
                    assert not np.isfinite(det[k])
                    continue
                closed = core.jacobian_closed_form(
                    float(rho[k]), float(dec), float(a), float(e), float(inc),
                    param,
                )
                if not np.isfinite(det[k]):
                    # WP2's conditioning guard fired; the closed form is
                    # finite there and the two are *meant* to differ.
                    n_bad += 1
                    continue
                n_ok += 1
                assert np.isclose(closed, det[k], rtol=JAC_RTOL, atol=0.0), (
                    param, a, e, inc, closed, det[k]
                )
    assert n_ok > 5000, f"only {n_ok} usable comparisons"
    # The guard region must be a small corner, not the bulk.
    assert n_bad < 0.02 * (n_ok + n_bad)


# ---------------------------------------------------------------------------
# 2. End-to-end parity
# ---------------------------------------------------------------------------


def _d42_warnings(records):
    """(category, message) for the warnings pneo itself raises."""
    return [
        (w.category, str(w.message))
        for w in records
        if str(w.message).startswith("pneo.")
    ]


def _compare(rp, rr, label, p_tol=P_TOL, lam_rtol=LAM_RTOL):
    """Assert two ClassificationResults agree within the acceptance bars."""
    assert rp.n_nodes == rr.n_nodes, label
    assert rp.n_masked_nonfinite == rr.n_masked_nonfinite, label
    # WP2's conditioning guard is reproduced exactly, not approximated.
    assert rp.n_masked_det == rr.n_masked_det, label
    if np.isnan(rp.p_neo):
        assert np.isnan(rr.p_neo), f"{label}: {rp.p_neo} vs {rr.p_neo}"
    else:
        assert abs(rp.p_neo - rr.p_neo) <= p_tol, (
            f"{label}: dp = {abs(rp.p_neo - rr.p_neo):.3e}"
        )
    for name in ("lambda_neo", "lambda_other"):
        x = getattr(rp, name)
        y = getattr(rr, name)
        if x == 0.0 or not np.isfinite(x):
            assert (y == x) or (np.isnan(x) and np.isnan(y)), (
                f"{label}.{name}: {x} vs {y}"
            )
        else:
            assert abs(y - x) / abs(x) <= lam_rtol, (
                f"{label}.{name}: rel = {abs(y - x) / abs(x):.3e}"
            )
    # `log_lr` NaN-aware, and to the same relative bar as Lambda where it is
    # an ordinary number.  A NaN log_lr is now *meaningful* (D-19: a
    # policy-assigned score has no likelihood ratio at all), so the two
    # backends disagreeing about it is a real defect, not a formality.
    if np.isnan(rp.log_lr):
        assert np.isnan(rr.log_lr), (
            f"{label}.log_lr: reference nan vs {rr.log_lr}"
        )
    elif np.isinf(rp.log_lr):
        assert rr.log_lr == rp.log_lr, label
    else:
        assert np.isfinite(rr.log_lr), (
            f"{label}.log_lr: {rp.log_lr} vs {rr.log_lr}"
        )
        scale = max(abs(rp.log_lr), 1.0)
        assert abs(rr.log_lr - rp.log_lr) / scale <= lam_rtol, (
            f"{label}.log_lr: {rp.log_lr} vs {rr.log_lr}"
        )
    # D-19: the status is part of the contract, and it is produced by the
    # shared post-processing — so a mismatch means one backend fed it
    # different mechanism evidence, which is exactly the failure the
    # `n_h_at_or_above_support` counter exists to prevent.
    assert rp.status == rr.status, (
        f"{label}.status: {rp.status!r} vs {rr.status!r}"
    )
    assert rp.policy_assigned == rr.policy_assigned, label


def test_cond_guard_matches_the_reference_condition_estimate(core):
    """`_core.cond_inf_elements` == the estimate inside WP2's guard.

    The Rust kernel reproduces `transforms.jacobian_elements_wrt_state`'s
    conditioning test rather than approximating it, so the two estimates must
    agree.  They come from different inverses (Gauss-Jordan vs LAPACK), and an
    inverse loses ~eps*cond relative accuracy, so the value is only compared
    where it still means something (cond <= 1e13, four digits or better).
    Above that only the *decision* is compared — which is all the kernel uses
    it for, and the decision is what `n_masked_det` parity rests on.
    """
    rng = np.random.default_rng(2024)
    n = 4000
    a = 10.0 ** rng.uniform(-0.7, 3.6, n)
    e = np.where(
        rng.random(n) < 0.5,
        rng.uniform(0.0, 0.999, n),
        1.0 - 10.0 ** rng.uniform(-11, -0.5, n),
    )
    inc = np.minimum(10.0 ** rng.uniform(-11, np.log10(np.pi), n), np.pi)
    Om = rng.uniform(0, 2 * np.pi, n)
    om = rng.uniform(0, 2 * np.pi, n)
    M = rng.uniform(0, 2 * np.pi, n)
    el = np.stack([a, e, inc, Om, om, M], axis=-1)

    Jse = T.jacobian_state_wrt_elements(el)
    finite = np.all(np.isfinite(Jse), axis=(-2, -1))
    safe = np.where(finite[..., None, None], Jse, np.eye(6))
    with np.errstate(all="ignore"):
        sign, logdet = np.linalg.slogdet(safe)
    nonsing = (sign != 0) & np.isfinite(logdet)
    safe = np.where(nonsing[..., None, None], safe, np.eye(6))
    Jes = np.linalg.inv(safe)
    with np.errstate(all="ignore"):
        want = (
            np.max(np.sum(np.abs(safe), axis=-1), axis=-1)
            * np.max(np.sum(np.abs(Jes), axis=-1), axis=-1)
        )
    ok_py = finite & nonsing

    n_checked = 0
    n_decide = 0
    E = T._kepler_E(M, e)
    for k in range(n):
        got = core.cond_inf_elements(
            float(a[k]), float(e[k]), float(inc[k]),
            float(Om[k]), float(om[k]), float(E[k]),
        )
        if not ok_py[k]:
            assert got is None or not (got <= core.COND_MAX)
            continue
        assert got is not None
        if want[k] <= 1e13:
            assert np.isclose(got, want[k], rtol=1e-6, atol=0.0), (
                k, a[k], e[k], inc[k], got, want[k]
            )
            n_checked += 1
        # The decision must agree except within a hair of the threshold.
        ratio = want[k] / core.COND_MAX
        if not (0.99 < ratio < 1.01):
            assert (got > core.COND_MAX) == (want[k] > core.COND_MAX), (
                k, got, want[k]
            )
            n_decide += 1
    assert n_checked > 1000, n_checked
    assert n_decide > 2000, n_decide


def test_prefilter_never_misses_a_masked_node_on_real_geometries(d2pair, core):
    """Every node the Python guard masks is one the prefilter routes to the
    exact estimate — checked on real ranging grids, not synthetic elements."""
    neo, other = d2pair
    n_masked = 0
    for att, R, V in _gate_tracklets(8, seed=8080):
        grid = build_grid(att, R, V, 0.01, 100.0, **FAST)
        if grid.rho.size == 0:
            continue
        r_equ, v_equ = T.topocentric_to_heliocentric(
            att.ra, att.dec, att.ra_rate, att.dec_rate,
            grid.rho, grid.rho_dot, R, V,
        )
        r_ecl = T.state_equ_to_ecl(r_equ)
        el = T.elements_from_state(r_ecl, T.state_equ_to_ecl(v_equ))
        det = T.ranging_jacobian_det(
            att.ra, att.dec, att.ra_rate, att.dec_rate,
            grid.rho, grid.rho_dot, R, V, parameterization="qei",
        )
        good = np.all(np.isfinite(el), axis=-1)
        rn = np.linalg.norm(r_ecl, axis=-1)
        masked = good & ~np.isfinite(det)
        for k in np.where(masked)[0]:
            a, e, inc = el[k, 0], el[k, 1], el[k, 2]
            assert core.needs_exact_guard(
                float(a), float(e), float(np.sin(inc)), float(rn[k])
            ), (a, e, inc, rn[k])
        n_masked += int(masked.sum())
        # ... and the prefilter must stay a filter, not a pass-through.
        fires = np.array([
            core.needs_exact_guard(float(x), float(y), float(np.sin(z)),
                                   float(rr))
            for x, y, z, rr in zip(el[good, 0], el[good, 1], el[good, 2],
                                   rn[good])
        ])
        assert fires.mean() < 0.30, fires.mean()
    assert n_masked > 0, "no node exercised the guard"


def test_end_to_end_parity_on_the_gate_catalogue(d2pair):
    """~200 real gate_v0 tracklets, both backends, on the same grid."""
    neo, other = d2pair
    cases = _gate_tracklets(200, seed=1234)
    worst_p = 0.0
    worst_lam = 0.0
    for k, (att, R, V) in enumerate(cases):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **FAST)
            rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        _compare(rp, rr, f"gate[{k}]")
        if np.isfinite(rp.p_neo):
            worst_p = max(worst_p, abs(rp.p_neo - rr.p_neo))
        for name in ("lambda_neo", "lambda_other"):
            x, y = getattr(rp, name), getattr(rr, name)
            if x > 0.0 and np.isfinite(x):
                worst_lam = max(worst_lam, abs(y - x) / x)
    assert worst_p <= P_TOL
    assert worst_lam <= LAM_RTOL
    print(f"\n  gate parity: max |dp| = {worst_p:.3e}, "
          f"max Lambda rel = {worst_lam:.3e}")


def test_end_to_end_parity_on_corner_geometries(d2pair):
    """Off-opposition, faint, fast, empty-region and fully-truncated corners."""
    neo, other = d2pair
    for name, att, R, V in _synthetic_corners():
        with warnings.catch_warnings(record=True) as wp:
            warnings.simplefilter("always")
            rp = classify(att, neo, other, R, V, backend="python", **FAST)
        with warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter("always")
            rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        _compare(rp, rr, name)
        # Identical warnings -- restricted to pneo's own, which are the
        # contract.  numpy warnings are NOT part of it: the reference path
        # evaluates a 6x6 determinant per node and leaks e.g. "divide by zero
        # encountered in det" at extreme rho, which the closed form has no
        # occasion to emit.  Comparing those would be comparing implementation
        # detail, not behaviour.
        assert _d42_warnings(wp) == _d42_warnings(wr), (
            name, _d42_warnings(wp), _d42_warnings(wr)
        )


def test_end_to_end_parity_in_clamped_mode(clamped_pair):
    """The D-14 digest2-emulation pair (top H bin [24, inf)) also matches."""
    neo, other = clamped_pair
    assert neo.h_top_bin_infinite and other.h_top_bin_infinite
    for k, (att, R, V) in enumerate(_gate_tracklets(40, seed=99)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **FAST)
            rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        _compare(rp, rr, f"clamped[{k}]")
    # And the clamp actually changes something, so this is not vacuous.
    plain = Digest2Histogram.neo_other_pair(MODEL_PATH)
    att, R, V = _gate_tracklets(1, seed=99)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = classify(att, *plain, R, V, backend="rust", **FAST)
        b = classify(att, neo, other, R, V, backend="rust", **FAST)
    assert a.lambda_neo != b.lambda_neo or a.lambda_other != b.lambda_other


def test_end_to_end_parity_at_production_resolution(d2pair):
    """A few tracklets at the real (1200, 50) grid."""
    neo, other = d2pair
    for k, (att, R, V) in enumerate(_gate_tracklets(6, seed=31)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **PROD)
            rr = classify(att, neo, other, R, V, backend="rust", **PROD)
        _compare(rp, rr, f"prod[{k}]")


def test_batch_matches_single_calls(d2pair):
    """`classify_many(backend="rust")` == per-tracklet Rust calls."""
    neo, other = d2pair
    cases = _gate_tracklets(24, seed=77)
    atts = [c[0] for c in cases]
    R = np.stack([c[1] for c in cases])
    V = np.stack([c[2] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = classify_many(
            atts, neo, other, R_obs=R, V_obs=V, backend="rust",
            n_threads=4, **FAST
        )
        one = [
            classify(a, neo, other, r, v, backend="rust", **FAST)
            for a, r, v in cases
        ]
    assert len(batch) == len(one)
    for k, (b, o) in enumerate(zip(batch, one)):
        assert b.lambda_neo == o.lambda_neo, k
        assert b.lambda_other == o.lambda_other, k
        assert b.n_nodes == o.n_nodes
        assert b.n_masked_nonfinite == o.n_masked_nonfinite
        assert b.n_masked_det == o.n_masked_det
        assert np.array_equal(b.grid.rho, o.grid.rho)


def test_batch_is_thread_count_invariant(d2pair):
    neo, other = d2pair
    cases = _gate_tracklets(16, seed=78)
    atts = [c[0] for c in cases]
    R = np.stack([c[1] for c in cases])
    V = np.stack([c[2] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runs = [
            classify_many(atts, neo, other, R_obs=R, V_obs=V, backend="rust",
                          diagnostics=False, n_threads=n, **FAST)
            for n in (1, 3, 16)
        ]
    for a, b in zip(runs[0], runs[1]):
        assert a.lambda_neo == b.lambda_neo
    for a, b in zip(runs[0], runs[2]):
        assert a.lambda_neo == b.lambda_neo
        assert a.lambda_other == b.lambda_other


# ---------------------------------------------------------------------------
# 3. Sigma points (K > 1)
# ---------------------------------------------------------------------------


def test_sigma_points_are_a_weighted_average(d2pair, core):
    """K = 3 with weights == the Python-side weighted mean of 3 classify calls."""
    from pneo.backend import core_model_for

    neo, other = d2pair
    cm = core_model_for(neo, other)
    att, R, V = _gate_tracklets(1, seed=1010)[0]
    d = 3e-6
    variants = np.array(
        [
            [att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag],
            [att.ra + d, att.dec - d, att.ra_rate * 1.02,
             att.dec_rate * 0.98, att.v_mag + 0.05],
            [att.ra - d, att.dec + 2 * d, att.ra_rate * 0.97,
             att.dec_rate * 1.03, att.v_mag - 0.07],
        ]
    )
    weights = np.array([0.5, 0.3, 0.2])

    out = core.classify_core(
        cm, variants, weights,
        np.ascontiguousarray(R), np.ascontiguousarray(V),
        *rho_axis(0.01, 100.0, FAST["n_rho"]), FAST["n_rhodot"],
        G_SLOPE_DEFAULT, False,
    )

    singles = []
    for row in variants:
        a = Attributable(
            ra=row[0], dec=row[1], ra_rate=row[2], dec_rate=row[3],
            epoch=att.epoch, v_mag=row[4],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            singles.append(
                classify(a, neo, other, R, V, backend="rust",
                         diagnostics=False, **FAST)
            )
    want_neo = float(sum(w * s.lambda_neo for w, s in zip(weights, singles)))
    want_other = float(
        sum(w * s.lambda_other for w, s in zip(weights, singles))
    )
    # The ensemble accumulator is exactly sum_k w_k Lambda_k, so this identity
    # holds to rounding, not to a fitted tolerance.
    assert np.isclose(out["lambda_neo"], want_neo, rtol=2e-16, atol=0.0), (
        out["lambda_neo"], want_neo
    )
    assert np.isclose(out["lambda_other"], want_other, rtol=2e-16, atol=0.0)
    assert out["n_nodes"] == sum(s.n_nodes for s in singles)
    assert out["n_masked_nonfinite"] == sum(
        s.n_masked_nonfinite for s in singles
    )

    # The same ensemble also matches a weighted average of the *Python*
    # backend's Lambdas, which is the independent check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        py = [
            classify(
                Attributable(ra=row[0], dec=row[1], ra_rate=row[2],
                             dec_rate=row[3], epoch=att.epoch, v_mag=row[4]),
                neo, other, R, V, backend="python", **FAST,
            )
            for row in variants
        ]
    py_neo = float(sum(w * s.lambda_neo for w, s in zip(weights, py)))
    py_other = float(sum(w * s.lambda_other for w, s in zip(weights, py)))
    assert abs(out["lambda_neo"] - py_neo) / py_neo <= LAM_RTOL
    assert abs(out["lambda_other"] - py_other) / py_other <= LAM_RTOL


def test_k1_with_unit_weight_is_the_plain_path(d2pair, core):
    from pneo.backend import core_model_for

    neo, other = d2pair
    cm = core_model_for(neo, other)
    att, R, V = _gate_tracklets(1, seed=1011)[0]
    v = np.array([[att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag]])
    out = core.classify_core(
        cm, v, np.ones(1), np.ascontiguousarray(R), np.ascontiguousarray(V),
        *rho_axis(0.01, 100.0, FAST["n_rho"]), FAST["n_rhodot"],
        G_SLOPE_DEFAULT, False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        one = classify(att, neo, other, R, V, backend="rust",
                       diagnostics=False, **FAST)
    assert out["lambda_neo"] == one.lambda_neo
    assert out["lambda_other"] == one.lambda_other


def test_sigma_points_reject_diagnostics(d2pair, core):
    from pneo.backend import core_model_for

    neo, other = d2pair
    cm = core_model_for(neo, other)
    att, R, V = _gate_tracklets(1, seed=1012)[0]
    v = np.tile(
        [att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag], (2, 1)
    )
    with pytest.raises(ValueError, match="single attributable variant"):
        core.classify_core(
            cm, v, np.ones(2), np.ascontiguousarray(R),
            np.ascontiguousarray(V), *rho_axis(0.01, 100.0, 10), 4,
            G_SLOPE_DEFAULT, True,
        )


def test_python_backend_refuses_sigma_points(d2pair):
    """No backend may silently drop an attached sigma-point ensemble.

    The Python reference cannot marginalize over the attributable [D-6], so
    `classify` refuses rather than scoring the central value and returning a
    number that differs from the Rust answer by O(1).
    """
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=2001)[0]
    att.sigma_points = np.array(
        [
            [att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag],
            [att.ra + 2e-4, att.dec - 1e-4, att.ra_rate * 1.3,
             att.dec_rate * 0.7, att.v_mag + 1.5],
        ]
    )
    att.sigma_weights = np.array([0.5, 0.5])
    try:
        with pytest.raises(NotImplementedError, match="sigma_points"):
            classify(att, neo, other, R, V, backend="python", **FAST)
        # ... and the Rust path honours them, so `auto` is never wrong-but-quiet.
        got = classify(att, neo, other, R, V, backend="rust",
                       diagnostics=False, **FAST)
        assert np.isfinite(got.lambda_neo)
        # A malformed ensemble is rejected by the shared detector.
        att.sigma_points = np.zeros((3, 4))
        with pytest.raises(ValueError, match=r"\(K, 5\)"):
            classify(att, neo, other, R, V, backend="rust", **FAST)
    finally:
        del att.sigma_points
        del att.sigma_weights


def test_sigma_point_hook_through_classify(d2pair):
    """`att.sigma_points`/`sigma_weights` drive the ensemble via classify()."""
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=1013)[0]
    rows = np.array(
        [
            [att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag],
            [att.ra + 1e-6, att.dec, att.ra_rate, att.dec_rate, att.v_mag],
        ]
    )
    w = np.array([0.6, 0.4])
    att.sigma_points = rows
    att.sigma_weights = w
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = classify(att, neo, other, R, V, backend="rust",
                           diagnostics=False, **FAST)
            parts = []
            for row in rows:
                a = Attributable(
                    ra=row[0], dec=row[1], ra_rate=row[2], dec_rate=row[3],
                    epoch=att.epoch, v_mag=row[4],
                )
                parts.append(
                    classify(a, neo, other, R, V, backend="rust",
                             diagnostics=False, **FAST)
                )
    finally:
        del att.sigma_points
        del att.sigma_weights
    want = float(sum(wi * p.lambda_neo for wi, p in zip(w, parts)))
    assert np.isclose(got.lambda_neo, want, rtol=2e-16, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Diagnostics arrays
# ---------------------------------------------------------------------------


def test_diagnostics_arrays_match_node_for_node(d2pair):
    """Per-node integrands agree away from the masked nodes."""
    neo, other = d2pair
    checked = 0
    for k, (att, R, V) in enumerate(_gate_tracklets(10, seed=222)):
        rp = classify(att, neo, other, R, V, backend="python", **FAST)
        rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        assert rp.node_density_neo.shape == rr.node_density_neo.shape
        # Nodes both backends consider usable: finite q/H on both sides and
        # a nonzero integrand somewhere (a zero can mean "masked" or "outside
        # the model support" and the two are indistinguishable by contract).
        live = (
            np.isfinite(rp.node_q) & np.isfinite(rr.node_q)
            & (rp.node_density_neo + rp.node_density_other > 0.0)
            & (rr.node_density_neo + rr.node_density_other > 0.0)
        )
        assert live.sum() > 100, k
        for name in ("node_density_neo", "node_density_other"):
            x = getattr(rp, name)[live]
            y = getattr(rr, name)[live]
            nz = x > 0.0
            np.testing.assert_allclose(y[nz], x[nz], rtol=1e-9, atol=0.0)
        checked += int(live.sum())
    assert checked > 5000


def test_diagnostics_off_returns_empty_arrays_in_both_backends(d2pair):
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=333)[0]
    for backend in ("python", "rust"):
        r = classify(att, neo, other, R, V, backend=backend,
                     diagnostics=False, **FAST)
        assert r.grid.rho.size == 0
        assert r.node_density_neo.size == 0
        assert r.node_density_other.size == 0
        assert r.node_H.size == 0
        assert r.node_q.size == 0
        assert r.n_nodes > 0            # the counters are still real
        assert r.grid.n_rho == FAST["n_rho"]
        assert r.diagnostics is False
        full = classify(att, neo, other, R, V, backend=backend, **FAST)
        assert full.diagnostics is True
        assert r.lambda_neo == full.lambda_neo
        assert r.lambda_other == full.lambda_other
        # The section 2.6 helpers refuse rather than answering 0.0 from
        # arrays that were never materialized (review finding 11).
        with pytest.raises(ValueError, match="diagnostics=False"):
            r.measure_fraction_H_above(25.5)
        with pytest.raises(ValueError, match="diagnostics=False"):
            r.posterior_mass_where(lambda x: x.node_q > 1.3)
        # ... and still work when the arrays are there.
        assert 0.0 <= full.measure_fraction_H_above(25.5) <= 1.0
        assert 0.0 <= full.posterior_mass_where(lambda x: x.node_q > 1.3) <= 1.0


def test_plot_method_works_on_the_rust_result(d2pair):
    """The ClassificationResult contract, including plotting, is unchanged."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=444)[0]
    r = classify(att, neo, other, R, V, backend="rust", **FAST)
    ax = r.plot_posterior_map()
    assert ax is not None
    plt.close(ax.figure)
    assert 0.0 <= r.measure_fraction_H_above(25.5) <= 1.0
    frac = r.posterior_mass_where(lambda x: x.node_q > 1.3)
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# 5. Backend plumbing
# ---------------------------------------------------------------------------


def test_env_var_overrides_the_argument(monkeypatch, d2pair):
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=555)[0]
    monkeypatch.setenv("PNEO_BACKEND", "python")
    assert resolve_backend("rust") == "python"
    a = classify(att, neo, other, R, V, backend="rust", **FAST)
    monkeypatch.delenv("PNEO_BACKEND")
    b = classify(att, neo, other, R, V, backend="python", **FAST)
    assert a.lambda_neo == b.lambda_neo


def test_bad_backend_name_raises(d2pair):
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=556)[0]
    with pytest.raises(ValueError, match="backend must be one of"):
        classify(att, neo, other, R, V, backend="fortran", **FAST)


def test_unsupported_model_falls_back_under_auto_and_raises_under_rust(d2pair):
    """A custom PopulationDensity is a fallback under auto, an error under rust."""
    from pneo.backend import ModelNotSupported
    from pneo.population import PopulationDensity

    class Flat(PopulationDensity):
        parameterization = "qei"

        def density(self, elements3, H):
            return np.ones(np.shape(H), dtype=float)

        def sample(self, n, rng):  # pragma: no cover - unused
            raise NotImplementedError

        def support(self):  # pragma: no cover - unused
            return {}

    att, R, V = _gate_tracklets(1, seed=557)[0]
    a = Flat()
    b = Flat()
    got = classify(att, a, b, R, V, backend="auto", **FAST)
    assert np.isfinite(got.p_neo)
    with pytest.raises(ModelNotSupported):
        classify(att, a, b, R, V, backend="rust", **FAST)


def test_core_model_is_cached(d2pair):
    from pneo.backend import core_model_for

    neo, other = d2pair
    assert core_model_for(neo, other) is core_model_for(neo, other)


def test_core_model_rejects_mismatched_edges(d2pair):
    from pneo.backend import ModelNotSupported, core_model_for

    neo, other = d2pair

    class Shifted:
        parameterization = "qei"
        h_top_bin_infinite = False
        label = "shifted"

        def __init__(self, src):
            self._density_table = src._density_table
            self.q_edges = np.asarray(src.q_edges) + 0.5
            self.e_edges = src.e_edges
            self.i_edges = src.i_edges
            self.h_edges = src.h_edges

    with pytest.raises(ModelNotSupported):
        core_model_for(neo, Shifted(other))


# ---------------------------------------------------------------------------
# 7. Findings from the adversarial review
# ---------------------------------------------------------------------------


def test_hostile_grid_sizes_raise_cleanly(d2pair, core):
    """Grid sizes from Python must never reach an abort, a hang, or a wrap.

    Review findings 2 and 4: `n_rho = 1e14` used to abort the process in the
    allocator (exit 134); `n_rho = n_rhodot = 2**32` wrapped the node product to
    zero and then span a ~4e9-iteration loop with the GIL released, i.e. an
    uninterruptible hang; a negative count raised OverflowError instead of the
    ValueError the Python reference gives.
    """
    from pneo.backend import core_model_for

    neo, other = d2pair
    cm = core_model_for(neo, other)
    att, R, V = _gate_tracklets(1, seed=3001)[0]
    v = np.array([[att.ra, att.dec, att.ra_rate, att.dec_rate, att.v_mag]])

    def run(n_rho, n_rhodot, nodes=None, d_rho=None):
        if nodes is None:
            nodes = np.geomspace(0.01, 100.0, max(n_rho, 1))
            d_rho = np.full(max(n_rho, 1), 0.1)
        return core.classify_core(
            cm, v, np.ones(1), np.ascontiguousarray(R),
            np.ascontiguousarray(V), np.ascontiguousarray(nodes),
            np.ascontiguousarray(d_rho), n_rhodot, G_SLOPE_DEFAULT, False,
        )

    for n_rhodot in (0, -1, 10 ** 14, 2 ** 32, 2 ** 40):
        with pytest.raises(ValueError):
            run(1200, n_rhodot)
    # an empty axis, and mismatched axis lengths
    with pytest.raises(ValueError):
        run(0, 50, nodes=np.zeros(0), d_rho=np.zeros(0))
    with pytest.raises(ValueError):
        run(0, 50, nodes=np.geomspace(0.01, 1.0, 10), d_rho=np.full(3, 0.1))
    # non-finite / non-positive nodes
    for bad in (np.nan, np.inf, 0.0, -1.0):
        nodes = np.geomspace(0.01, 100.0, 20).copy()
        nodes[7] = bad
        with pytest.raises(ValueError):
            run(20, 5, nodes=nodes, d_rho=np.full(20, 0.1))
    with pytest.raises(ValueError):
        run(20, 5, nodes=np.geomspace(0.01, 100.0, 20),
            d_rho=np.full(20, np.nan))
    # the product cap still bites through the paired axis
    with pytest.raises(ValueError):
        run(10 ** 6, 10 ** 4, nodes=np.geomspace(0.01, 100.0, 10 ** 6),
            d_rho=np.full(10 ** 6, 0.1))
    # The same sizes go through classify() with the same exception type as the
    # Python reference gives -- including counts too large for an i64, which
    # would otherwise surface as PyO3's OverflowError.
    for backend in ("python", "rust"):
        with pytest.raises(ValueError):
            classify(att, neo, other, R, V, backend=backend, n_rho=-5,
                     n_rhodot=50)
        with pytest.raises(ValueError):
            classify(att, neo, other, R, V, backend=backend, n_rho=10,
                     n_rhodot=4, rho_min=0.0)
    with pytest.raises(ValueError):
        classify(att, neo, other, R, V, backend="rust", n_rho=2 ** 70,
                 n_rhodot=1)
    with pytest.raises(ValueError):
        classify_many([att], neo, other, R_obs=R, V_obs=V, backend="rust",
                      n_rho=2 ** 70, n_rhodot=1)
    # ... and a sane grid still works.
    assert run(180, 12)["n_nodes"] > 0


def test_core_model_rejects_overflowing_and_invalid_bin_structures(core):
    """Review finding 1: the cell count must be checked, not wrapped.

    Four 65 537-long edge arrays give dims = [65536] * 4, whose product is
    exactly 0 modulo 2**64.  An unchecked product let an empty table pass the
    length check and then be indexed ~2**64 elements out of bounds — a
    reproducible SIGSEGV from pure Python through the public `CoreModel`.
    """
    from pneo.backend import ModelNotSupported, core_model_for

    big = 65536
    edges = [np.arange(big + 1, dtype=float) for _ in range(4)]
    with pytest.raises(ValueError, match="overflow"):
        core.CoreModel(np.zeros(0), np.zeros(0), (big,) * 4, *edges)
    assert issubclass(core.UnsupportedModel, ValueError)

    # Other malformed structures are rejected as UnsupportedModel too, so that
    # backend="auto" can fall back (review finding 7).
    good = [
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3),
    ]
    tab = np.ones(2 ** 4, dtype=float)
    for bad_table in (
        np.full(16, np.nan),        # NaN cell
        -np.ones(16),               # negative cell
        np.ones(15),                # wrong length
        np.full(16, np.inf),        # infinite cell
    ):
        with pytest.raises(core.UnsupportedModel):
            core.CoreModel(bad_table, tab, (2, 2, 2, 2), *good)
    bad_edges = list(good)
    bad_edges[0] = np.array([0.0, 1.0, 0.5])          # non-monotone
    with pytest.raises(core.UnsupportedModel):
        core.CoreModel(tab, tab, (2, 2, 2, 2), *bad_edges)

    # And such a model makes `auto` fall back instead of hard-failing.
    class Broken:
        parameterization = "qei"
        h_top_bin_infinite = False
        label = "broken"

        def __init__(self):
            self._density_table = np.ones((2, 2, 2, 2))
            self.q_edges = np.array([0.0, 1.0, 0.5])   # non-monotone
            self.e_edges = np.linspace(0.0, 1.0, 3)
            self.i_edges = np.linspace(0.0, 1.0, 3)
            self.h_edges = np.linspace(0.0, 1.0, 3)

        def density(self, elements3, H):  # pragma: no cover - unused
            return np.zeros(np.shape(H))

    with pytest.raises(ModelNotSupported):
        core_model_for(Broken(), Broken())


def test_density_lookup_is_bounds_checked(core):
    """The table read is bounds-checked, so a bad shape cannot read wild memory.

    Constructed with a *consistent* but tiny shape, every probe — including
    values far outside every edge array — must return 0.0 and not crash.
    """
    edges = [np.linspace(0.0, 1.0, 3) for _ in range(4)]
    tab = np.arange(16, dtype=float)
    cm = core.CoreModel(tab, tab, (2, 2, 2, 2), *edges)
    probes = np.array([
        -1e300, -1.0, -0.0, 0.0, 0.25, 0.5, 0.999, 1.0, 1e300,
        np.inf, -np.inf, np.nan,
    ])
    el3 = np.ascontiguousarray(
        np.stack(np.broadcast_arrays(probes, probes, probes), axis=-1)
    )
    a, b = cm.density_at(el3, np.ascontiguousarray(probes))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(a >= 0.0)


def test_model_cache_tracks_in_place_mutation(d2pair):
    """Review finding 3: the cache must not serve a stale Rust copy.

    Flipping `h_top_bin_infinite` after the first call does not touch the
    density table, so an id()-keyed cache kept scoring the old clamp mode —
    a measured 63% divergence in Lambda, with no warning.  Editing a cell of
    the table in place is the other half of the same hole.
    """
    neo, other = d2pair
    att, R, V = _gate_tracklets(1, seed=3002)[0]
    base = classify(att, neo, other, R, V, backend="rust", **FAST)

    neo.h_top_bin_infinite = True
    other.h_top_bin_infinite = True
    try:
        clamped = classify(att, neo, other, R, V, backend="rust", **FAST)
        ref = classify(att, neo, other, R, V, backend="python", **FAST)
        assert clamped.lambda_neo != base.lambda_neo, "stale cached model"
        assert abs(clamped.lambda_neo - ref.lambda_neo) / ref.lambda_neo <= LAM_RTOL
    finally:
        neo.h_top_bin_infinite = False
        other.h_top_bin_infinite = False
    restored = classify(att, neo, other, R, V, backend="rust", **FAST)
    assert restored.lambda_neo == base.lambda_neo

    # An in-place edit of a *sampled, non-empty* cell must also be picked up.
    nonzero = np.argwhere(neo._density_table > 0.0)
    changed = False
    for idx in nonzero[::37]:
        key = tuple(int(x) for x in idx)
        old = neo._density_table[key]
        neo._density_table[key] = old * 3.0
        try:
            got = classify(att, neo, other, R, V, backend="rust", **FAST)
            ref = classify(att, neo, other, R, V, backend="python", **FAST)
        finally:
            neo._density_table[key] = old
        if got.lambda_neo != base.lambda_neo:
            assert abs(got.lambda_neo - ref.lambda_neo) / ref.lambda_neo <= LAM_RTOL
            changed = True
            break
    assert changed, "no probed cell was sampled by this tracklet"
    assert classify(att, neo, other, R, V, backend="rust",
                    **FAST).lambda_neo == base.lambda_neo


def test_env_auto_does_not_override_an_explicit_argument(monkeypatch, d2pair):
    """Review finding 8: env `auto` means "no preference", not "override"."""
    from pneo.backend import requested_backend

    for value in ("auto", "  AUTO  ", "Auto"):
        monkeypatch.setenv("PNEO_BACKEND", value)
        assert requested_backend("python") == "python"
        assert requested_backend("rust") == "rust"
        assert requested_backend(None) == "auto"
        assert resolve_backend("python") == "python"
    # The two concrete names still override.
    monkeypatch.setenv("PNEO_BACKEND", "python")
    assert requested_backend("rust") == "python"
    monkeypatch.setenv("PNEO_BACKEND", "rust")
    assert requested_backend("python") == "rust"
    monkeypatch.setenv("PNEO_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="PNEO_BACKEND"):
        requested_backend("python")


def test_prefilter_bound_holds_at_perihelion(core):
    """Review finding 5: the prefilter must cover perihelion passage.

    The bound is an upper bound on the *reachable* manifold only (see
    `rust/src/guard.rs::cond_upper_bound`), so this checks it exactly there:
    element sets whose implied r = a(1 - e cos E) lies inside the sampled range.
    The old `a^{3/2}/(1 - e)` term self-violated 64 times once E = 0 was
    included.
    """
    rng = np.random.default_rng(5150)
    n_masked = 0
    n_checked = 0
    for _ in range(30000):
        a = 10.0 ** rng.uniform(-1.3, 4.5)
        e = (1.0 - 10.0 ** rng.uniform(-13, -0.3)
             if rng.random() < 0.5 else rng.uniform(0.0, 0.999))
        inc = min(10.0 ** rng.uniform(-13, np.log10(np.pi)), np.pi)
        E = rng.choice([0.0, np.pi, rng.uniform(-np.pi, np.pi)])
        r = a * (1.0 - e * np.cos(E))
        if not (0.01 <= r <= 101.0):
            continue          # not reachable; the bound makes no claim there
        n_checked += 1
        cond = core.cond_inf_elements(
            float(a), float(e), float(inc), 0.7, 2.1, float(E)
        )
        masked = cond is None or not (cond <= core.COND_MAX)
        if masked:
            n_masked += 1
            assert core.needs_exact_guard(
                float(a), float(e), float(np.sin(inc)), float(r)
            ), (a, e, inc, E, r, cond)
    assert n_checked > 3000, n_checked
    assert n_masked > 100, n_masked


def test_prefilter_spares_the_bulk_of_real_nodes(d2pair, core):
    """The prefilter must stay a filter: most real nodes take the fast path."""
    neo, other = d2pair
    fired = 0
    total = 0
    for att, R, V in _gate_tracklets(6, seed=5151):
        grid = build_grid(att, R, V, 0.01, 100.0, **FAST)
        if grid.rho.size == 0:
            continue
        r_equ, v_equ = T.topocentric_to_heliocentric(
            att.ra, att.dec, att.ra_rate, att.dec_rate,
            grid.rho, grid.rho_dot, R, V,
        )
        r_ecl = T.state_equ_to_ecl(r_equ)
        el = T.elements_from_state(r_ecl, T.state_equ_to_ecl(v_equ))
        m = np.all(np.isfinite(el), axis=-1)
        rn = np.linalg.norm(r_ecl, axis=-1)[m]
        for a, e, inc, r in zip(el[m, 0], el[m, 1], el[m, 2], rn):
            total += 1
            fired += core.needs_exact_guard(
                float(a), float(e), float(np.sin(inc)), float(r)
            )
    assert total > 5000
    assert fired / total < 0.30, f"prefilter fires on {100 * fired / total:.1f}%"


@pytest.mark.xfail(
    reason="EXPECTED DIVERGENCE, reference-side limitation. At |dec| = 90 deg "
           "exactly the tangent basis degenerates (|du/dra| = cos dec = 0) and "
           "the Python 6x6 composed determinant loses up to 219x, with smooth "
           "degradation from 89.99 deg. The Rust closed form "
           "rho^4 cos^2(dec) * 2/(mu^1.5 sqrt(a) e sin i) is the correct value "
           "there (adversarial review, 2026-09). Recorded as an xfail so the "
           "day the reference is fixed this turns into an XPASS and we notice.",
    strict=True,
)
def test_reference_jacobian_degrades_at_the_pole(core):
    """|J| closed form vs the 6x6 reference at |dec| -> 90 deg exactly."""
    R = np.array([0.4, -0.85, -0.37])
    V = np.array([0.0155, 0.0062, 0.0027])
    dec = np.pi / 2                       # exactly the pole
    ra, ra_rate, dec_rate = 1.3, 0.0, 0.004
    rho = np.array([1.5, 3.0, 8.0])
    rho_dot = np.array([0.0, -0.002, 0.001])
    el = T.elements_from_state(
        *[T.state_equ_to_ecl(x) for x in T.topocentric_to_heliocentric(
            ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V)]
    )
    det = T.ranging_jacobian_det(
        ra, dec, ra_rate, dec_rate, rho, rho_dot, R, V, parameterization="qei"
    )
    checked = 0
    for k in range(len(rho)):
        if not np.all(np.isfinite(el[k])) or not np.isfinite(det[k]):
            continue
        closed = core.jacobian_closed_form(
            float(rho[k]), float(dec), float(el[k, 0]), float(el[k, 1]),
            float(el[k, 2]), "qei",
        )
        checked += 1
        assert np.isclose(closed, det[k], rtol=JAC_RTOL, atol=0.0)
    assert checked > 0, "no usable node at the pole"


def test_closed_form_jacobian_near_the_pole_is_self_consistent(core):
    """Approaching the pole, the closed form stays smooth and finite.

    Companion to the xfail above: the Rust value is the one that behaves, so
    check it against its own analytic limit rather than against the reference.
    """
    for dec_deg in (89.0, 89.9, 89.99, 89.99999, 90.0):
        dec = np.radians(dec_deg)
        got = core.jacobian_closed_form(2.5, dec, 3.0, 0.2, 0.4, "qei")
        want = (2.5 ** 4 * np.cos(dec) ** 2 * 2.0
                / (MU_SUN ** 1.5 * np.sqrt(3.0) * 0.2 * np.sin(0.4))
                * (1.0 - 0.2))
        assert np.isfinite(got)
        assert np.isclose(got, want, rtol=1e-13, atol=0.0), dec_deg
    # cos(pi/2) is 6.1e-17 in binary64, not 0, so the value is tiny rather
    # than exactly zero -- and it is the exact analytic value for that float.
    pole = core.jacobian_closed_form(2.5, np.pi / 2, 3.0, 0.2, 0.4, "qei")
    assert 0.0 <= pole < 1e-20


def test_batched_diagnostics_round_trip(d2pair):
    """Draining the result vector must not corrupt or reorder the arrays."""
    neo, other = d2pair
    cases = _gate_tracklets(6, seed=5152)
    atts = [c[0] for c in cases]
    Rb = np.stack([c[1] for c in cases])
    Vb = np.stack([c[2] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = classify_many(atts, neo, other, R_obs=Rb, V_obs=Vb,
                              backend="rust", diagnostics=True, n_threads=3,
                              **FAST)
        singles = [
            classify(a, neo, other, r, v, backend="rust", diagnostics=True,
                     **FAST)
            for a, r, v in cases
        ]
    for k, (b, o) in enumerate(zip(batch, singles)):
        assert np.array_equal(b.grid.rho, o.grid.rho), k
        assert np.array_equal(b.grid.rho_dot, o.grid.rho_dot), k
        assert np.array_equal(b.node_density_neo, o.node_density_neo), k
        assert np.array_equal(b.node_density_other, o.node_density_other), k
        assert np.array_equal(b.node_H, o.node_H, equal_nan=True), k
        assert b.diagnostics and o.diagnostics


# ---------------------------------------------------------------------------
# 6. A representative subset of the existing WP5 suite, on both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["python", "rust"])
def backend(request, monkeypatch):
    """Pin the process default backend for the duration of one test.

    Existing tests do not take a ``backend`` argument, so the switch is made
    through ``PNEO_BACKEND`` — which is exactly the knob a user would use
    to re-run a whole suite on the kernel.
    """
    if request.param == "rust" and not core_available():  # pragma: no cover
        pytest.skip("rust backend unavailable")
    monkeypatch.setenv("PNEO_BACKEND", request.param)
    return request.param


#: Existing tests re-run on both backends.  Chosen because they (a) use the
#: digest2 histogram pair, so the Rust path is actually exercised, and (b) do
#: not monkeypatch the Python pipeline's internals (those are, by construction,
#: reference-backend tests).  Names are resolved lazily so this file does not
#: fail to import if the WP5 suite is refactored.
_SHARED_TESTS = [
    "test_fast_opposition_tracklet_scores_neo",
    "test_slow_opposition_tracklet_scores_mba",
    "test_rate_monotonicity_between_the_two_regimes",
    "test_G_changes_the_answer_off_opposition",
    "test_G_none_is_the_default_slope_and_not_an_arbitrary_one",
    "test_real_grids_have_no_nonfinite_nodes",
    "test_node_H_and_node_q_and_the_section_2_6_helpers",
    "test_posterior_mass_where_validates",
    "test_empty_admissible_region_returns_nan_and_warns",
    "test_truncated_H_gives_zero_likelihoods_and_warns",
    "test_classify_many_matches_individual_calls",
    "test_classify_many_degenerate_results_are_nan_not_warning_counts",
    "test_classify_many_empty_list",
    "test_non_finite_v_mag_is_rejected_early",
    "test_observer_state_defaults_to_geocenter",
]


@pytest.mark.parametrize("name", _SHARED_TESTS)
def test_existing_wp5_test_on_both_backends(backend, name, d2pair):
    """Re-run one existing WP5 test with the process backend pinned.

    The test's own fixture names are read off its signature, so this keeps
    working if the WP5 suite gains or drops a fixture argument.
    """
    import importlib
    import inspect

    mod = importlib.import_module("test_classifier")
    fn = getattr(mod, name, None)
    if fn is None:  # pragma: no cover - suite refactored
        pytest.skip(f"{name} no longer exists in test_classifier")
    values = {"d2pair": d2pair, "obs": mod._observer()}
    params = list(inspect.signature(fn).parameters)
    if any(p not in values for p in params):  # pragma: no cover
        pytest.skip(f"{name} needs fixtures {params} this shim cannot supply")
    assert resolve_backend("auto") == backend
    fn(*[values[p] for p in params])


# ---------------------------------------------------------------------------
# 7. Smooth-in-H reconstruction (h_interpolation="pchip")
#
# Here the two backends are NOT independent implementations: Python
# precomputes the monotone-cubic coefficients once and the kernel copies them,
# so both evaluate the identical Horner form `a + s (b + s c)` on identical
# doubles.  That is a deliberate design choice — it makes the density lookup
# *bit-identical* rather than merely close, which in turn keeps the
# end-to-end Lambda agreement at the same level as the histogram mode instead
# of adding a coefficient-derivation seam.  The tests below therefore assert
# bit-equality of the lookup, and check that the glue refuses (rather than
# silently downgrades) any component whose coefficients it cannot pass on.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pchip_pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(MODEL_PATH, h_interpolation="pchip")


def test_pchip_density_lookup_is_bit_exact(core, pchip_pair):
    """Rust == Python bit for bit on the smooth branch, edges included.

    Probes hit every bin edge from both sides (so `s = H - h_edges[ih]` is
    exercised at 0 and at one ulp below the interval width), plus +/-inf, NaN
    and the out-of-support tails.
    """
    from pneo.backend import core_model_for

    neo, other = pchip_pair
    cm = core_model_for(neo, other)
    rng = np.random.default_rng(20260902)

    def probes(edges, n):
        e = np.asarray(edges, dtype=float)
        return np.concatenate(
            [
                e,
                np.nextafter(e, -np.inf),
                np.nextafter(e, +np.inf),
                rng.uniform(e[0] - 0.5, e[-1] + 0.5, n),
                np.array([np.inf, -np.inf, np.nan, -0.0]),
            ]
        )

    q = probes(Q_EDGES, 4000)
    e = probes(E_EDGES, 4000)
    i = probes(I_EDGES, 4000)
    h = probes(H_EDGES, 4000)
    n = min(len(q), len(e), len(i), len(h))
    q, e, i, h = q[:n], e[:n], i[:n], h[:n]
    for arr in (e, i, h):
        rng.shuffle(arr)

    # A random (q, e, i, H) point almost never lands in a populated cell of
    # the real model (the NEO component lives at q < 1.3 alone), so add draws
    # taken from the model itself: those are guaranteed in-support and are
    # what the classifier actually evaluates.
    for comp in (neo, other):
        el6, hh = comp.sample(2000, rng)
        bound = el6[:, 1] < 1.0
        q = np.concatenate([q, (el6[:, 0] * (1.0 - el6[:, 1]))[bound]])
        e = np.concatenate([e, el6[bound, 1]])
        i = np.concatenate([i, el6[bound, 2]])
        h = np.concatenate([h, hh[bound]])

    el3 = np.ascontiguousarray(np.stack([q, e, i], axis=-1))
    want_neo = neo.density(el3, h)
    want_other = other.density(el3, h)
    got_neo, got_other = cm.density_at(el3, np.ascontiguousarray(h))
    assert np.array_equal(got_neo, want_neo)
    assert np.array_equal(got_other, want_other)
    # Not vacuous, and genuinely the smooth branch: in-support values must be
    # present and must differ from the piecewise-constant model.
    assert (want_neo > 0.0).any() and (want_neo == 0.0).any()
    hist_neo, hist_other = Digest2Histogram.neo_other_pair(MODEL_PATH)
    assert not np.array_equal(want_neo, hist_neo.density(el3, h))
    assert not np.array_equal(want_other, hist_other.density(el3, h))


def test_pchip_end_to_end_parity_on_the_gate_catalogue(pchip_pair):
    """~120 real gate_v0 tracklets scored by both backends in pchip mode."""
    neo, other = pchip_pair
    worst_p = 0.0
    worst_lam = 0.0
    for k, (att, R, V) in enumerate(_gate_tracklets(120, seed=606)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **FAST)
            rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        _compare(rp, rr, f"pchip[{k}]")
        if np.isfinite(rp.p_neo):
            worst_p = max(worst_p, abs(rp.p_neo - rr.p_neo))
        for name in ("lambda_neo", "lambda_other"):
            x, y = getattr(rp, name), getattr(rr, name)
            if x > 0.0 and np.isfinite(x):
                worst_lam = max(worst_lam, abs(y - x) / x)
    assert worst_p <= P_TOL
    assert worst_lam <= LAM_RTOL
    print(f"\n  pchip parity: max |dp| = {worst_p:.3e}, "
          f"max Lambda rel = {worst_lam:.3e}")


def test_pchip_end_to_end_parity_at_production_resolution(pchip_pair):
    neo, other = pchip_pair
    for k, (att, R, V) in enumerate(_gate_tracklets(4, seed=77)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **PROD)
            rr = classify(att, neo, other, R, V, backend="rust", **PROD)
        _compare(rp, rr, f"pchip-prod[{k}]")


def test_pchip_changes_the_answer(d2pair, pchip_pair):
    """Not a vacuous parity test: the mode moves Lambda on real tracklets."""
    moved = 0
    for att, R, V in _gate_tracklets(20, seed=808):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = classify(att, *d2pair, R, V, backend="rust", **FAST)
            b = classify(att, *pchip_pair, R, V, backend="rust", **FAST)
        assert a.n_nodes == b.n_nodes          # geometry is untouched
        assert a.n_masked_det == b.n_masked_det
        if a.lambda_neo != b.lambda_neo or a.lambda_other != b.lambda_other:
            moved += 1
    assert moved >= 18, moved


def test_pchip_batch_matches_single_calls(pchip_pair):
    neo, other = pchip_pair
    cases = _gate_tracklets(24, seed=505)
    atts = [c[0] for c in cases]
    R = np.stack([c[1] for c in cases])
    V = np.stack([c[2] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        many = classify_many(atts, neo, other, R_obs=R, V_obs=V,
                             backend="rust", diagnostics=False, **FAST)
        for k, (att, r, v) in enumerate(cases):
            one = classify(att, neo, other, r, v, backend="rust",
                           diagnostics=False, **FAST)
            assert many[k].lambda_neo == one.lambda_neo, k
            assert many[k].lambda_other == one.lambda_other, k


def test_model_cache_distinguishes_the_h_interpolation_mode(d2pair, pchip_pair):
    """The fingerprint must see the mode: the density TABLE is identical.

    Switching `h_interpolation` leaves `_density_table` untouched (the smooth
    mode carries its coefficients separately), so a fingerprint that ignored
    the mode would hand back the piecewise-constant handle and silently score
    a different model — the same failure the D-14 clamp flag once had.
    """
    from pneo.backend import _fingerprint, core_model_for

    hist_neo, hist_other = d2pair
    sm_neo, sm_other = pchip_pair
    assert np.array_equal(hist_neo._density_table, sm_neo._density_table)
    assert _fingerprint(hist_neo) != _fingerprint(sm_neo)
    assert _fingerprint(hist_other) != _fingerprint(sm_other)
    a = core_model_for(hist_neo, hist_other)
    b = core_model_for(sm_neo, sm_other)
    assert a is not b
    assert "h_interpolation=(\"none\", \"none\")" in repr(a)
    assert "h_interpolation=(\"pchip\", \"pchip\")" in repr(b)
    # and the cached handles stay distinct on a second lookup
    assert core_model_for(hist_neo, hist_other) is a
    assert core_model_for(sm_neo, sm_other) is b


def test_declared_h_interpolation_without_coefficients_is_refused(pchip_pair):
    """No silent downgrade to the flat table.

    A component that claims a smooth mode but cannot produce coefficients is
    `ModelNotSupported`: `auto` then runs the Python reference (which would
    also fail loudly on such an object) and an explicit `backend="rust"`
    raises.  It must never be scored as if it were piecewise constant.
    """
    from pneo.backend import ModelNotSupported, core_model_for

    class Liar:
        def __init__(self, src):
            for name in ("q_edges", "e_edges", "i_edges", "h_edges",
                         "parameterization", "label", "variant", "classes",
                         "_density_table", "counts", "bin_volume"):
                setattr(self, name, getattr(src, name))
            self.h_top_bin_infinite = False
            self.h_interpolation = "pchip"
            self.h_coeffs = None

    # The partner is in the SAME mode, so `check_component_pair` passes and the
    # coefficient validation is what this test exercises.
    neo, other = pchip_pair
    with pytest.raises(ModelNotSupported, match="h_coeffs"):
        core_model_for(Liar(neo), other)

    class WrongShape(Liar):
        def __init__(self, src):
            super().__init__(src)
            self.h_coeffs = np.zeros((29, 8, 11, 18, 4))

    with pytest.raises(ModelNotSupported, match="h_coeffs"):
        core_model_for(WrongShape(neo), other)

    class WrongCells(Liar):
        def __init__(self, src):
            super().__init__(src)
            self.h_coeffs = np.zeros((29, 8, 11, 17, 3))

    with pytest.raises(ModelNotSupported, match="h_coeffs"):
        core_model_for(WrongCells(neo), other)

    # An h_interpolation this build does not implement must be refused even
    # when coefficients ARE supplied: the glue cannot know what polynomial
    # form they are in, and guessing would score an arbitrary model.
    class FutureMode(Liar):
        def __init__(self, src):
            super().__init__(src)
            self.h_interpolation = "akima"
            self.h_coeffs = np.zeros((29, 8, 11, 18, 3))

    # BOTH sides declare the unknown mode, so `check_component_pair` is happy
    # and the unknown-mode refusal is what this asserts.
    with pytest.raises(ModelNotSupported, match="does not know"):
        core_model_for(FutureMode(neo), FutureMode(other))


def test_core_model_rejects_bad_coefficient_arrays(core, d2pair):
    """The kernel's own validation of the coefficient block.

    Reachable from pure Python, so it is checked at the boundary and not only
    in the glue: a wrong length, a non-finite entry, or the D-14 clamp
    combination must all raise `UnsupportedModel` rather than be indexed
    into.
    """
    neo, other = d2pair
    tab = np.ascontiguousarray(neo._density_table.reshape(-1))
    dims = tuple(int(s) for s in neo._density_table.shape)
    n = tab.size
    edges = (Q_EDGES, E_EDGES, I_EDGES, H_EDGES)
    good = np.zeros(3 * n)

    def build(**kw):
        return core.CoreModel(tab, tab, dims, *edges, **kw)

    # sanity: the good case builds
    build(neo_h_coeffs=good, other_h_coeffs=good)
    with pytest.raises(core.UnsupportedModel, match="h_coeffs"):
        build(neo_h_coeffs=np.zeros(3 * n - 1))
    with pytest.raises(core.UnsupportedModel, match="h_coeffs"):
        build(neo_h_coeffs=np.zeros(3 * n + 3))
    bad = good.copy()
    bad[17] = np.nan
    with pytest.raises(core.UnsupportedModel, match="finite"):
        build(neo_h_coeffs=bad)
    bad = good.copy()
    bad[3] = np.inf
    with pytest.raises(core.UnsupportedModel, match="finite"):
        build(other_h_coeffs=bad)
    # D-14 clamp + smooth reconstruction: refused on both sides, matching
    # Digest2Histogram's own ValueError.
    with pytest.raises(core.UnsupportedModel, match="h_top_bin_infinite"):
        build(neo_h_coeffs=good, neo_h_top_bin_infinite=True)
    with pytest.raises(core.UnsupportedModel, match="h_top_bin_infinite"):
        build(other_h_coeffs=good, other_h_top_bin_infinite=True)
    with pytest.raises(ValueError, match="h_top_bin_infinite"):
        Digest2Histogram.neo_other_pair(MODEL_PATH, h_interpolation="pchip",
                                        h_top_bin_infinite=True)


def test_mixed_mode_component_pairs_are_refused(core, d2pair, pchip_pair,
                                                clamped_pair):
    """A one-sided density mode is a caller error, not a supported model.

    This test previously pinned the *opposite* behaviour — it asserted that a
    (pchip NEO, histogram other) pair built fine and evaluated each side in its
    own mode, on the reasoning that the kernel keeps the two `Histogram`s
    independent and "policy lives in Python".  The adversarial review showed
    there was no such policy anywhere: `classify` ran the mixed pair happily on
    both backends and moved p(NEO) by up to 6.4e-3, which is the size of the
    entire honest mode change.  So the policy now lives in
    `backend.check_component_pair`, and both mismatch axes are covered.

    `ValueError`, deliberately, not `ModelNotSupported`: the latter means "the
    reference can do this and the kernel cannot" and makes `auto` fall back —
    which here would score the artifact instead of refusing it.
    """
    from pneo.backend import (ModelNotSupported, check_component_pair,
                                  core_model_for)

    hist_neo, hist_other = d2pair
    sm_neo, sm_other = pchip_pair
    cl_neo, cl_other = clamped_pair

    cases = [
        ("h_interpolation", sm_neo, hist_other),
        ("h_interpolation", hist_neo, sm_other),
        ("h_top_bin_infinite", cl_neo, hist_other),
        ("h_top_bin_infinite", hist_neo, cl_other),
    ]
    for attr, a, b in cases:
        with pytest.raises(ValueError, match=attr) as ei:
            check_component_pair(a, b)
        # not the recoverable kind: `auto` must not fall back past this
        assert not isinstance(ei.value, ModelNotSupported)
        with pytest.raises(ValueError, match=attr):
            core_model_for(a, b)
        att, R, V = _gate_tracklets(1, seed=4242)[0]
        with pytest.raises(ValueError, match=attr):
            classify(att, a, b, R, V, backend="rust", **FAST)
        with pytest.raises(ValueError, match=attr):
            classify_many([att], a, b, R_obs=R.reshape(1, 3),
                          V_obs=V.reshape(1, 3), backend="rust", **FAST)

    # Matched pairs of every flavour still build.
    for a, b in (d2pair, pchip_pair, clamped_pair):
        check_component_pair(a, b)
        assert core_model_for(a, b) is not None


def test_mixed_mode_guard_does_not_fire_on_matched_or_unaware_components(
        d2pair):
    """The guard is about the paired flags only, and defaults to "matched".

    A component that has never heard of either knob (a custom
    `PopulationDensity`) must pair with anything, since `getattr` defaults
    stand in for both flags; and differing `label`/`variant`/`parameterization`
    are NOT the guard's business.
    """
    from pneo.backend import check_component_pair

    class Minimal:
        parameterization = "aei"
        label = "custom"

    neo, other = d2pair
    check_component_pair(Minimal(), Minimal())
    check_component_pair(Minimal(), other)      # other is h_interpolation="none"
    check_component_pair(neo, Minimal())
    # a differing parameterization is legitimate and must not be flagged here
    check_component_pair(neo, Minimal())


def test_mixed_mode_artifact_is_real_and_the_guard_is_what_stops_it(
        d2pair, pchip_pair):
    """Measure the artifact the guard prevents, so it cannot be dismissed.

    Bypasses `check_component_pair` by driving the kernel handle directly, and
    compares the mixed pair's p(NEO) against BOTH honest modes on real
    tracklets.  If this ever comes out negligible the guard could be argued
    down; recording the number is what makes that argument checkable.
    """
    from pneo.backend import (_build_core_model, _edges_of, _table_of,
                                  core_model_for)
    from pneo import _core

    hist_neo, hist_other = d2pair
    sm_neo, sm_other = pchip_pair
    tabs = [_table_of(sm_neo), _table_of(hist_other)]
    mixed = _build_core_model(_core, tabs, _edges_of(sm_neo),
                              ["qei", "qei"], sm_neo, hist_other)
    assert 'h_interpolation=("pchip", "none")' in repr(mixed)

    # The artifact CANNOT be seen node by node: the two components are exactly
    # complementary at q = 1.3 AU, so at any single (q, e, i, H) point one of
    # them is identically zero and the node-level P(NEO) is 0 or 1 in every
    # mode.  The mismatch only shows up once the admissible region is
    # integrated over, mixing nodes from both sides of q = 1.3 — so this has to
    # be measured through the kernel's own integral.
    cases = _gate_tracklets(40, seed=4242)
    worst_artifact = 0.0
    worst_honest = 0.0
    for att, R, V in cases:
        p = {}
        for name, handle in (("mixed", mixed),
                             ("hist", core_model_for(hist_neo, hist_other)),
                             ("smooth", core_model_for(sm_neo, sm_other))):
            out = _core.classify_core(
                handle,
                np.array([[att.ra, att.dec, att.ra_rate, att.dec_rate,
                           att.v_mag]]),
                np.ones(1),
                np.ascontiguousarray(R, dtype=float),
                np.ascontiguousarray(V, dtype=float),
                *rho_axis(0.01, 100.0, FAST["n_rho"]), FAST["n_rhodot"],
                G_SLOPE_DEFAULT, False,
            )
            ln, lo = float(out["lambda_neo"]), float(out["lambda_other"])
            p[name] = ln / (ln + lo) if (ln + lo) > 0.0 else float("nan")
        if not all(np.isfinite(v) for v in p.values()):
            continue
        worst_artifact = max(worst_artifact, abs(p["mixed"] - p["hist"]))
        worst_honest = max(worst_honest, abs(p["smooth"] - p["hist"]))
    # Not vacuous: the honest mode change moves something.
    assert worst_honest > 1e-5
    # And the mismatch is a displacement of the same ORDER as the effect under
    # study — a few times 1e-3, which is why it has to be refused rather than
    # documented.  The measured ratio is grid- and sample-dependent (this runs
    # 40 tracklets at the fast grid; the review measured 6.4e-3 on a larger
    # set), so the assertion is on the magnitude, not on an ordering.
    assert worst_artifact > 1e-3
    print(f"\n  mixed-pair artifact: max |dp| = {worst_artifact:.3e}; the "
          f"honest mode change on the same 40 tracklets: {worst_honest:.3e} "
          f"(ratio {worst_artifact / worst_honest:.2f})")



# ---------------------------------------------------------------------------
# D-19 — the finite H extension across the backend boundary
#
# The kernel needed NO change for this mode: the H axis length and the H edge
# array are data-driven (`rust/src/model.rs` header), and `pneo.backend`
# hands over the already-extended table / edges / coefficients.  These tests
# are what turns "no change needed" from an assertion into a measurement — the
# lookup is asserted bit-equal, the end-to-end Lambdas are asserted at the
# pre-existing D-15 level, and the mode is asserted to actually move the answer
# so none of it is vacuous.
# ---------------------------------------------------------------------------

H_EXT = 30.0


@pytest.fixture(scope="module")
def ext_pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(MODEL_PATH, h_extend_to=H_EXT)


@pytest.fixture(scope="module")
def ext_pchip_pair():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return Digest2Histogram.neo_other_pair(
        MODEL_PATH, h_extend_to=H_EXT, h_interpolation="pchip"
    )


def _ext_probes(pair, rng, n=4000):
    """(elements3, H) probes spanning the EXTENDED H axis and beyond it."""
    neo = pair[0]
    h_edges = np.asarray(neo.h_edges, dtype=float)

    def probes(edges, m):
        e = np.asarray(edges, dtype=float)
        return np.concatenate([
            e,
            np.nextafter(e, -np.inf),
            np.nextafter(e, +np.inf),
            rng.uniform(e[0] - 0.5, e[-1] + 0.5, m),
            np.array([np.inf, -np.inf, np.nan, -0.0]),
        ])

    q = probes(Q_EDGES, n)
    e = probes(E_EDGES, n)
    i = probes(I_EDGES, n)
    h = probes(h_edges, n)
    k = min(len(q), len(e), len(i), len(h))
    q, e, i, h = q[:k], e[:k], i[:k], h[:k]
    for arr in (e, i, h):
        rng.shuffle(arr)
    # Random points almost never land in a populated cell, so add draws from
    # the components themselves — which now include H above 25.5.
    for comp in pair:
        el6, hh = comp.sample(3000, rng)
        bound = el6[:, 1] < 1.0
        q = np.concatenate([q, (el6[:, 0] * (1.0 - el6[:, 1]))[bound]])
        e = np.concatenate([e, el6[bound, 1]])
        i = np.concatenate([i, el6[bound, 2]])
        h = np.concatenate([h, hh[bound]])
    return np.ascontiguousarray(np.stack([q, e, i], axis=-1)), \
        np.ascontiguousarray(h)


@pytest.mark.parametrize("which", ["hist", "pchip"])
def test_extended_density_lookup_is_bit_exact(core, ext_pair, ext_pchip_pair,
                                              which):
    """Rust == Python bit for bit on the extended H axis, both H modes.

    Probes hit every extended bin edge from both sides (so the new edge at 30
    is exercised at, one ulp below and one ulp above), plus +/-inf, NaN, and
    both out-of-support tails.
    """
    from pneo.backend import core_model_for

    pair = ext_pair if which == "hist" else ext_pchip_pair
    neo, other = pair
    cm = core_model_for(neo, other)
    assert "dims=[29, 8, 11, 19]" in repr(cm)   # the extended axis crossed over
    rng = np.random.default_rng(190190)
    el3, h = _ext_probes(pair, rng)
    want_neo, want_other = neo.density(el3, h), other.density(el3, h)
    got_neo, got_other = cm.density_at(el3, h)
    assert np.array_equal(got_neo, want_neo)
    assert np.array_equal(got_other, want_other)
    # Not vacuous, and genuinely exercising the extension: in-support values
    # above H = 25.5 must be present on both sides of the boundary.
    above = h >= 25.5
    assert (want_neo[above] > 0.0).any()
    assert (want_neo > 0.0).any() and (want_neo == 0.0).any()
    # ... and D-4 still bites at the NEW boundary, in both backends.
    beyond = h >= H_EXT
    assert beyond.any()
    assert np.all(got_neo[beyond] == 0.0) and np.all(got_other[beyond] == 0.0)


@pytest.mark.parametrize("which", ["hist", "pchip"])
def test_extended_end_to_end_parity_on_the_gate_catalogue(
        ext_pair, ext_pchip_pair, which):
    """~120 real gate_v0 tracklets scored by both backends in extended mode."""
    neo, other = ext_pair if which == "hist" else ext_pchip_pair
    worst_p = 0.0
    worst_lam = 0.0
    for k, (att, R, V) in enumerate(_gate_tracklets(120, seed=909)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **FAST)
            rr = classify(att, neo, other, R, V, backend="rust", **FAST)
        _compare(rp, rr, f"ext-{which}[{k}]")
        assert rp.status == rr.status, f"ext-{which}[{k}] status"
        if np.isfinite(rp.p_neo):
            worst_p = max(worst_p, abs(rp.p_neo - rr.p_neo))
        for name in ("lambda_neo", "lambda_other"):
            x, y = getattr(rp, name), getattr(rr, name)
            if x > 0.0 and np.isfinite(x):
                worst_lam = max(worst_lam, abs(y - x) / x)
    assert worst_p <= P_TOL
    assert worst_lam <= LAM_RTOL
    print(f"\n  extended({which}) parity: max |dp| = {worst_p:.3e}, "
          f"max Lambda rel = {worst_lam:.3e}")


@pytest.mark.parametrize("which", ["hist", "pchip"])
def test_extended_end_to_end_parity_at_production_resolution(
        ext_pair, ext_pchip_pair, which):
    neo, other = ext_pair if which == "hist" else ext_pchip_pair
    for k, (att, R, V) in enumerate(_gate_tracklets(4, seed=717)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = classify(att, neo, other, R, V, backend="python", **PROD)
            rr = classify(att, neo, other, R, V, backend="rust", **PROD)
        _compare(rp, rr, f"ext-{which}-prod[{k}]")


def test_extended_batch_matches_single_calls(ext_pair):
    neo, other = ext_pair
    cases = _gate_tracklets(24, seed=606)
    atts = [c[0] for c in cases]
    R = np.stack([c[1] for c in cases])
    V = np.stack([c[2] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = classify_many(atts, neo, other, R_obs=R, V_obs=V,
                              backend="rust", **FAST)
        for k, (att, r, v) in enumerate(cases):
            one = classify(att, neo, other, r, v, backend="rust", **FAST)
            assert batch[k].lambda_neo == one.lambda_neo, k
            assert batch[k].lambda_other == one.lambda_other, k
            assert batch[k].p_neo == one.p_neo or (
                np.isnan(batch[k].p_neo) and np.isnan(one.p_neo)
            ), k
            assert batch[k].status == one.status, k


def test_extended_mode_changes_the_answer(d2pair, ext_pair):
    """Not a vacuous parity test: the extension moves Lambda on real tracklets,
    and it moves the NEO side much harder than the other side (55.8% vs 1.4%
    of top-bin mass, DESIGN §2.6)."""
    moved = 0
    ratios = []
    for att, R, V in _gate_tracklets(30, seed=818):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = classify(att, *d2pair, R, V, backend="rust", **FAST)
            b = classify(att, *ext_pair, R, V, backend="rust", **FAST)
        # the geometry is untouched: only the density changed
        assert a.n_nodes == b.n_nodes
        assert a.n_masked_det == b.n_masked_det
        assert a.n_masked_nonfinite == b.n_masked_nonfinite
        if a.lambda_neo != b.lambda_neo or a.lambda_other != b.lambda_other:
            moved += 1
        # Lambda can only GROW: the extension adds non-negative density and
        # touches nothing below 25.5 (histogram mode).
        assert b.lambda_neo >= a.lambda_neo
        assert b.lambda_other >= a.lambda_other
        if a.lambda_neo > 0.0 and a.lambda_other > 0.0:
            ratios.append((b.lambda_neo / a.lambda_neo,
                           b.lambda_other / a.lambda_other))
    # Measured: 18 of 30 gate_v0 tracklets move at all.  The other 12 have
    # negligible admissible-region measure above H = 25.5, which is exactly the
    # DESIGN §2.6 diagnostic saying the truncation never bound for them — the
    # extension is then a no-op on their score BY CONSTRUCTION, not by
    # accident, since nothing below 25.5 changes in histogram mode.  The bar is
    # set below the measurement so the test asserts "the mode does something on
    # most real tracklets" without pinning a sample-specific count.
    assert moved >= 15, moved
    assert ratios
    gn = float(np.median([r[0] for r in ratios]))
    go = float(np.median([r[1] for r in ratios]))
    print(f"\n  extended vs truncated on 30 gate tracklets: {moved} moved; "
          f"median Lambda_NEO x{gn:.4f}, Lambda_other x{go:.4f}")


def test_extended_pair_mismatch_is_refused_at_every_entry_point(
        core, d2pair, ext_pair):
    """The D-19 flag joins the paired-mode guard, on BOTH backends.

    A one-sided extension is the most violent of the three mismatches:
    extending only the NEO side adds ~1.7x its own total mass to the numerator
    of P(NEO) and nothing to the denominator.  All four entry points are
    covered, including the two `backend="python"` ones — `classify` and
    `classify_many` call `check_component_pair` themselves for exactly this
    reason, so the guard cannot fire on one backend and not the other.
    """
    from pneo.backend import (ModelNotSupported, check_component_pair,
                                  core_model_for)

    hist_neo, hist_other = d2pair
    ext_neo, ext_other = ext_pair
    # Also a mismatch of REACH, not just on/off: 30 vs 40 have the same table
    # shape AND (because the synthetic slab is a copy of the top slab) the same
    # density table, so only h_extend_to itself distinguishes them.
    far_neo, far_other = Digest2Histogram.neo_other_pair(MODEL_PATH,
                                                         h_extend_to=40.0)
    cases = [
        (ext_neo, hist_other),
        (hist_neo, ext_other),
        (ext_neo, far_other),
        (far_neo, ext_other),
    ]
    att, R, V = _gate_tracklets(1, seed=4242)[0]
    for a, b in cases:
        with pytest.raises(ValueError, match="h_extend_to") as ei:
            check_component_pair(a, b)
        assert not isinstance(ei.value, ModelNotSupported)
        with pytest.raises(ValueError, match="h_extend_to"):
            core_model_for(a, b)
        for be in ("rust", "python"):
            with pytest.raises(ValueError, match="h_extend_to"):
                classify(att, a, b, R, V, backend=be, **FAST)
            with pytest.raises(ValueError, match="h_extend_to"):
                classify_many([att], a, b, R_obs=R.reshape(1, 3),
                              V_obs=V.reshape(1, 3), backend=be, **FAST)
    # Matched extended pairs still build and score on both backends.
    check_component_pair(*ext_pair)
    assert core_model_for(*ext_pair) is not None
    check_component_pair(far_neo, far_other)
    assert core_model_for(far_neo, far_other) is not None


def test_two_reaches_do_not_share_a_cached_core_model(core, ext_pair):
    """The fingerprint hazard D-19 introduces, closed and measured.

    h_extend_to = 30 and = 40 produce density tables that are **bit-identical
    and identically shaped** — the synthetic slab is a copy of the top slab —
    and differ only in `h_edges[-1]`, which the fingerprint does not hash.
    Without `h_extend_to` in the key the second pair would silently reuse the
    first pair's CoreModel and score every H in [30, 40) as out of support.
    """
    from pneo.backend import _fingerprint, core_model_for

    neo30, other30 = ext_pair
    neo40, other40 = Digest2Histogram.neo_other_pair(MODEL_PATH,
                                                     h_extend_to=40.0)
    # The premise: the tables really are indistinguishable.
    assert np.array_equal(neo30._density_table, neo40._density_table)
    assert neo30._density_table.shape == neo40._density_table.shape
    # ... and the fingerprints are not.
    assert _fingerprint(neo30) != _fingerprint(neo40)
    cm30 = core_model_for(neo30, other30)
    cm40 = core_model_for(neo40, other40)
    assert cm30 is not cm40
    el3 = np.ascontiguousarray(np.array([[0.5, 0.15, 0.05]]))
    h = np.ascontiguousarray(np.array([32.0]))
    assert cm30.density_at(el3, h)[0][0] == 0.0
    assert cm40.density_at(el3, h)[0][0] > 0.0


def test_extended_and_clamped_are_mutually_exclusive_in_the_kernel_too(core):
    """The Python constructor refuses the combination; the kernel cannot even
    be handed it, because the clamp would then extend a constant to infinity
    from a boundary that is itself synthetic.  Nothing in the glue can build
    such a pair — asserted here so a future change to `_build_core_model`
    cannot open the door quietly."""
    with pytest.raises(ValueError, match="h_top_bin_infinite"):
        Digest2Histogram.neo_other_pair(MODEL_PATH, h_extend_to=H_EXT,
                                        h_top_bin_infinite=True)
    # And a hand-built CoreModel with the clamp on an extended axis is at
    # least self-consistent about which mode it is in: the clamp maps H above
    # the LAST edge into the last bin, which is the synthetic one.  This is
    # unreachable from Python and is documented as such; the assertion records
    # what the kernel would do rather than pretending it cannot happen.
    hist = Digest2Histogram.neo_other_pair(MODEL_PATH, h_extend_to=H_EXT)[0]
    cm = core.CoreModel(
        np.ascontiguousarray(hist._density_table.reshape(-1)),
        np.ascontiguousarray(hist._density_table.reshape(-1)),
        tuple(int(s) for s in hist.shape),
        np.asarray(hist.q_edges, dtype=float),
        np.asarray(hist.e_edges, dtype=float),
        np.asarray(hist.i_edges, dtype=float),
        np.asarray(hist.h_edges, dtype=float),
        neo_h_top_bin_infinite=True,
        other_h_top_bin_infinite=True,
    )
    el3 = np.ascontiguousarray(np.array([[0.5, 0.15, 0.05]]))
    at_30 = cm.density_at(el3, np.ascontiguousarray(np.array([H_EXT])))[0][0]
    inside = hist.density(el3, np.array([26.0]))[0]
    assert at_30 == inside
