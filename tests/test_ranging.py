"""WP3 tests — admissible region and (rho, rho_dot) ranging grids.

Everything here is checked against *independent* brute-force machinery: the
energy condition is re-implemented from its definition (no reuse of the
quadratic algebra in ``pneo.ranging``), and the reference integrals are
computed on uniform (rho, rho_dot) grids so that the log-spacing measure in
``build_grid`` is genuinely under test.

Geometries are hand-built (R_obs / V_obs literals) so these tests do not
depend on WP1's ``observer_state`` or on WP2's transforms.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from pneo.attributable import Attributable
from pneo.constants import MU_SUN
from pneo.ranging import RangingGrid, admissible_rhodot_interval, build_grid

EPOCH = Time(60000.0, format="mjd", scale="tdb")

# Observer: 1 AU, on the +x axis, moving in +y at roughly circular speed.
R_OBS = np.array([1.0, 0.0, 0.0])
V_OBS = np.array([0.0, 0.0172, 0.0])

V_ESC_1AU = np.sqrt(2.0 * MU_SUN / 1.0)  # ~0.02433 AU/day


# ---------------------------------------------------------------------------
# Independent re-implementation of the geometry / energy condition
# ---------------------------------------------------------------------------


def _basis(ra: float, dec: float):
    """u_hat and its RAW ra/dec partials, written out by hand (equ frame)."""
    ca, sa = np.cos(ra), np.sin(ra)
    cd, sd = np.cos(dec), np.sin(dec)
    u = np.array([cd * ca, cd * sa, sd])
    du_dra = np.array([-cd * sa, cd * ca, 0.0])
    du_ddec = np.array([-sd * ca, -sd * sa, cd])
    return u, du_dra, du_ddec


def brute_energy(att, R_obs, V_obs, rho, rho_dot):
    """0.5|v|^2 - mu/|r| from the DESIGN §2.1 definitions. Broadcasts."""
    u, du_dra, du_ddec = _basis(att.ra, att.dec)
    rho = np.asarray(rho, dtype=float)
    rho_dot = np.asarray(rho_dot, dtype=float)
    rho_b, rhod_b = np.broadcast_arrays(rho, rho_dot)
    r_vec = R_obs + rho_b[..., None] * u
    v_vec = (
        V_obs
        + rhod_b[..., None] * u
        + rho_b[..., None] * (att.ra_rate * du_dra + att.dec_rate * du_ddec)
    )
    r = np.sqrt((r_vec * r_vec).sum(axis=-1))
    v2 = (v_vec * v_vec).sum(axis=-1)
    return 0.5 * v2 - MU_SUN / r


def brute_interval(att, R_obs, V_obs, rho, rd_span=0.06, n=400001):
    """Scan rho_dot densely; return (lo, hi) of the sign-negative set and step."""
    grid = np.linspace(-rd_span, rd_span, n)
    step = grid[1] - grid[0]
    e = brute_energy(att, R_obs, V_obs, float(rho), grid)
    inside = e < 0.0
    if not inside.any():
        return np.nan, np.nan, step
    idx = np.flatnonzero(inside)
    return grid[idx[0]], grid[idx[-1]], step


# ---------------------------------------------------------------------------
# Reference integrators on UNIFORM rho grids (independent of the log measure)
# ---------------------------------------------------------------------------


def ref_integral_uniform(f, att, R_obs, V_obs, rho_min, rho_max, n_rho, n_rhodot):
    """Nested midpoint rule: uniform rho, exact per-rho [lo, hi] limits."""
    edges = np.linspace(rho_min, rho_max, n_rho + 1)
    rho = 0.5 * (edges[:-1] + edges[1:])
    d_rho = edges[1] - edges[0]
    lo, hi = admissible_rhodot_interval(att, R_obs, V_obs, rho)
    w = hi - lo
    ok = np.isfinite(w) & (w > 0.0)
    if not ok.any():
        return 0.0
    rho, lo, w = rho[ok], lo[ok], w[ok]
    d_rd = w / n_rhodot
    nodes = lo[:, None] + d_rd[:, None] * (np.arange(n_rhodot) + 0.5)[None, :]
    vals = f(np.repeat(rho, n_rhodot), nodes.ravel())
    weights = np.repeat(d_rho * d_rd, n_rhodot)
    return float(np.sum(vals * weights))


def ref_integral_rectangles(f, att, R_obs, V_obs, rho_min, rho_max, rd_max,
                            n_rho=2000, n_rhodot=4000):
    """Fully brute force: uniform 2-D rectangle rule with an energy indicator."""
    rho = rho_min + (rho_max - rho_min) * (np.arange(n_rho) + 0.5) / n_rho
    d_rho = (rho_max - rho_min) / n_rho
    rd = -rd_max + 2.0 * rd_max * (np.arange(n_rhodot) + 0.5) / n_rhodot
    d_rd = 2.0 * rd_max / n_rhodot
    total = 0.0
    for chunk in np.array_split(np.arange(n_rho), 20):
        rr = rho[chunk][:, None]
        e = brute_energy(att, R_obs, V_obs, rr, rd[None, :])
        vals = np.where(e < 0.0, f(np.broadcast_to(rr, e.shape), np.broadcast_to(rd[None, :], e.shape)), 0.0)
        total += float(vals.sum())
    return total * d_rho * d_rd


# ---------------------------------------------------------------------------
# Test geometries
# ---------------------------------------------------------------------------


def att_opposition():
    """Anti-solar direction seen from R_obs = (1,0,0): u_hat = +x, no motion."""
    return Attributable(ra=0.0, dec=0.0, ra_rate=0.0, dec_rate=0.0,
                        epoch=EPOCH, v_mag=21.0)


def att_generic():
    """Off-axis line of sight with typical main-belt-ish sky motion."""
    return Attributable(ra=0.35, dec=0.20, ra_rate=-3.0e-3, dec_rate=1.0e-3,
                        epoch=EPOCH, v_mag=22.0)


def att_fast():
    """A fast mover, well away from opposition."""
    return Attributable(ra=2.10, dec=-0.55, ra_rate=1.2e-2, dec_rate=-4.0e-3,
                        epoch=EPOCH, v_mag=20.0)


ALL_GEOMETRIES = [
    ("opposition", att_opposition(), R_OBS, V_OBS),
    ("generic", att_generic(), R_OBS, V_OBS),
    ("fast", att_fast(), R_OBS, V_OBS),
    ("tilted-observer", att_generic(),
     np.array([0.7, -0.6, 0.05]), np.array([0.010, 0.0125, -4.0e-4])),
]


# ===========================================================================
# 1. Interval boundaries vs. a brute-force scan of the energy condition
# ===========================================================================


@pytest.mark.parametrize("name,att,R,V", ALL_GEOMETRIES, ids=[g[0] for g in ALL_GEOMETRIES])
def test_interval_matches_brute_force_scan(name, att, R, V):
    rhos = [0.02, 0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 5.0]
    checked_nonempty = 0
    for rho in rhos:
        lo, hi = admissible_rhodot_interval(att, R, V, np.array(rho))
        b_lo, b_hi, step = brute_interval(att, R, V, rho)
        if np.isnan(b_lo):
            # brute force found nothing: analytic must agree (empty, or an
            # interval so thin it fell between scan samples).
            assert np.isnan(lo) or (hi - lo) <= 2.0 * step
            continue
        assert np.isfinite(lo) and np.isfinite(hi), f"{name} rho={rho}"
        # The scan brackets the true roots from the inside, to one step.
        assert abs(lo - b_lo) <= step, f"{name} rho={rho}: lo {lo} vs {b_lo}"
        assert abs(hi - b_hi) <= step, f"{name} rho={rho}: hi {hi} vs {b_hi}"
        checked_nonempty += 1
    assert checked_nonempty >= 3


@pytest.mark.parametrize("name,att,R,V", ALL_GEOMETRIES, ids=[g[0] for g in ALL_GEOMETRIES])
def test_endpoints_are_zero_energy_and_interior_is_bound(name, att, R, V):
    rho = np.geomspace(0.01, 3.0, 40)
    lo, hi = admissible_rhodot_interval(att, R, V, rho)
    ok = np.isfinite(lo)
    assert ok.any()
    lo_v, hi_v = lo[ok], hi[ok]
    rho_v = rho[ok]
    # Roots have exactly zero energy...
    assert np.allclose(brute_energy(att, R, V, rho_v, lo_v), 0.0, atol=1e-18)
    assert np.allclose(brute_energy(att, R, V, rho_v, hi_v), 0.0, atol=1e-18)
    # ...the midpoint is bound...
    mid = 0.5 * (lo_v + hi_v)
    assert np.all(brute_energy(att, R, V, rho_v, mid) <= 0.0)
    # ...and just outside is unbound.
    pad = 0.05 * (hi_v - lo_v) + 1e-6
    assert np.all(brute_energy(att, R, V, rho_v, lo_v - pad) > 0.0)
    assert np.all(brute_energy(att, R, V, rho_v, hi_v + pad) > 0.0)


def test_empty_where_brute_force_finds_nothing():
    """Far beyond the region the analytic interval must be NaN, not garbage."""
    att, R, V = att_opposition(), R_OBS, V_OBS
    for rho in (1.5, 3.0, 10.0, 50.0):
        lo, hi = admissible_rhodot_interval(att, R, V, np.array(rho))
        assert np.isnan(lo) and np.isnan(hi)
        b_lo, _, _ = brute_interval(att, R, V, rho, rd_span=0.5, n=200001)
        assert np.isnan(b_lo)


def test_interval_vectorization_and_shapes():
    att, R, V = att_generic(), R_OBS, V_OBS
    rho = np.geomspace(0.01, 2.0, 17)
    lo, hi = admissible_rhodot_interval(att, R, V, rho)
    assert lo.shape == rho.shape and hi.shape == rho.shape
    for i, r in enumerate(rho):
        li, hi_i = admissible_rhodot_interval(att, R, V, np.array(r))
        assert np.array_equal(np.array([li, hi_i]), np.array([lo[i], hi[i]]), equal_nan=True)
    # 2-D input shape is preserved too.
    rho2 = rho[:12].reshape(3, 4)
    lo2, hi2 = admissible_rhodot_interval(att, R, V, rho2)
    assert lo2.shape == (3, 4) and hi2.shape == (3, 4)


def test_interval_half_width_is_leftover_speed_budget():
    """(hi - lo)/2 == sqrt(2 mu / r - |W_perp|^2), the derivation's second form."""
    att, R, V = att_generic(), R_OBS, V_OBS
    rho = np.geomspace(0.02, 1.5, 25)
    lo, hi = admissible_rhodot_interval(att, R, V, rho)
    u, du_dra, du_ddec = _basis(att.ra, att.dec)
    W = V + rho[:, None] * (att.ra_rate * du_dra + att.dec_rate * du_ddec)
    r = np.linalg.norm(R + rho[:, None] * u, axis=-1)
    w_par = W @ u
    w_perp2 = (W * W).sum(axis=-1) - w_par**2
    expect = 2.0 * MU_SUN / r - w_perp2
    ok = expect >= 0
    assert np.allclose(0.5 * (hi[ok] - lo[ok]), np.sqrt(expect[ok]), rtol=1e-12)
    assert np.all(np.isnan(lo[~ok]))
    # Interval centre is -(u . W).
    assert np.allclose(0.5 * (hi[ok] + lo[ok]), -w_par[ok], rtol=1e-12)


# ===========================================================================
# 2. Integral convergence under grid refinement
# ===========================================================================


def smooth_f(rho, rho_dot):
    """A smooth, O(1), non-separable test integrand on the region."""
    return np.exp(-1.5 * rho) * (1.0 + 0.5 * np.cos(60.0 * rho_dot + 2.0 * rho))


@pytest.mark.parametrize(
    "name,att,R,V,rho_min,rho_max",
    [
        ("opposition", att_opposition(), R_OBS, V_OBS, 0.05, 0.80),
        ("generic", att_generic(), R_OBS, V_OBS, 0.05, 0.80),
        ("fast", att_fast(), R_OBS, V_OBS, 0.05, 0.60),
    ],
)
def test_integral_converges_under_refinement(name, att, R, V, rho_min, rho_max):
    reference = ref_integral_uniform(smooth_f, att, R, V, rho_min, rho_max, 20000, 2000)

    errs = []
    for n_rho, n_rhodot in [(20, 10), (40, 20), (80, 40), (160, 80), (320, 160)]:
        g = build_grid(att, R, V, rho_min, rho_max, n_rho, n_rhodot)
        val = float(np.sum(smooth_f(g.rho, g.rho_dot) * g.weight))
        errs.append(abs(val - reference) / abs(reference))

    # Monotone-ish decrease, and at least ~2nd order overall.
    assert errs[-1] < errs[0] / 50.0, f"{name}: no convergence, errs={errs}"
    assert errs[-1] < 1e-4, f"{name}: errs={errs}"
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert fine < coarse, f"{name}: non-monotone errs={errs}"


@pytest.mark.parametrize(
    "name,att,R,V,rho_min,rho_max,rd_max",
    [
        ("opposition", att_opposition(), R_OBS, V_OBS, 0.05, 0.80, 0.04),
        ("generic", att_generic(), R_OBS, V_OBS, 0.05, 0.80, 0.05),
    ],
)
def test_integral_matches_brute_force_rectangles(name, att, R, V, rho_min, rho_max, rd_max):
    brute = ref_integral_rectangles(smooth_f, att, R, V, rho_min, rho_max, rd_max)
    g = build_grid(att, R, V, rho_min, rho_max, n_rho=400, n_rhodot=200)
    val = float(np.sum(smooth_f(g.rho, g.rho_dot) * g.weight))
    rel = abs(val - brute) / abs(brute)
    assert rel < 1e-3, f"{name}: grid={val!r} brute={brute!r} rel={rel:.3e}"


# ===========================================================================
# 3. Weight / measure correctness  (the log-measure bug)
# ===========================================================================


@pytest.mark.parametrize(
    "name,att,R,V,rho_min,rho_max,rd_max",
    [
        ("opposition", att_opposition(), R_OBS, V_OBS, 0.05, 0.80, 0.04),
        ("generic", att_generic(), R_OBS, V_OBS, 0.05, 0.80, 0.05),
        ("fast", att_fast(), R_OBS, V_OBS, 0.05, 0.60, 0.06),
    ],
)
def test_unit_integrand_gives_region_area(name, att, R, V, rho_min, rho_max, rd_max):
    """f = 1 integrates to the (rho, rho_dot) AREA in AU^2/day."""
    one = lambda rho, rd: np.ones_like(np.asarray(rho, dtype=float))

    area_semi = ref_integral_uniform(one, att, R, V, rho_min, rho_max, 20000, 1)
    area_brute = ref_integral_rectangles(one, att, R, V, rho_min, rho_max, rd_max)

    # The two references agree with each other first.
    assert abs(area_semi - area_brute) / area_semi < 1e-3

    g = build_grid(att, R, V, rho_min, rho_max, n_rho=400, n_rhodot=200)
    area_grid = float(np.sum(g.weight))
    assert abs(area_grid - area_semi) / area_semi < 1e-5
    assert abs(area_grid - area_brute) / area_brute < 1e-3

    # Sanity on units/scale: area ~ (rho span) x (typical rho_dot width).
    assert 0.0 < area_grid < (rho_max - rho_min) * 2.0 * rd_max


def test_rho_cell_widths_are_linear_not_logarithmic():
    """The rho measure must be d(rho), not d(log rho).

    With the region nonempty across the whole rho range, the per-rho cell
    widths must tile [rho_min, rho_max] in LINEAR rho.  A d(log rho) measure
    would instead tile log(rho_max/rho_min) -- a factor ~2.7 off here, and
    with the wrong rho-dependence.
    """
    att, R, V = att_opposition(), R_OBS, V_OBS
    rho_min, rho_max, n_rho, n_rhodot = 0.05, 0.80, 64, 8

    g = build_grid(att, R, V, rho_min, rho_max, n_rho, n_rhodot)
    rho_u, inverse = np.unique(g.rho, return_inverse=True)
    assert rho_u.size == n_rho, "region should be nonempty at every rho here"

    lo, hi = admissible_rhodot_interval(att, R, V, rho_u)
    w_per_rho = np.bincount(inverse, weights=g.weight)
    d_rho = w_per_rho / (hi - lo)  # recovered linear cell widths

    assert np.isclose(d_rho.sum(), rho_max - rho_min, rtol=1e-12)
    # ...and definitely not the log widths.
    d_log = np.log(rho_max / rho_min) / (n_rho - 1)
    assert not np.isclose(d_rho.sum(), np.log(rho_max / rho_min), rtol=1e-3)
    # Interior cells scale linearly with rho: d_rho_i ~ rho_i * d_log.
    interior = slice(1, -1)
    assert np.allclose(d_rho[interior], rho_u[interior] * 2.0 * np.sinh(d_log / 2.0),
                       rtol=1e-12)
    # Which means the widths are NOT all equal (the log-measure signature).
    assert d_rho.max() / d_rho.min() > 5.0


def test_area_is_measure_invariant_across_resolutions():
    """The area must not drift with n_rho / n_rhodot (a measure bug would)."""
    att, R, V = att_generic(), R_OBS, V_OBS
    areas = [
        float(np.sum(build_grid(att, R, V, 0.05, 0.8, n, k).weight))
        for n, k in [(25, 5), (50, 25), (200, 100), (800, 40)]
    ]
    assert np.allclose(areas, areas[0], rtol=2e-4), areas


# ===========================================================================
# 4. Physical sanity
# ===========================================================================


def test_opposition_region_is_wide_and_escape_velocity_scaled():
    att, R, V = att_opposition(), R_OBS, V_OBS

    # Nonempty over a wide range of rho.
    rho = np.geomspace(1e-3, 1.0, 200)
    lo, hi = admissible_rhodot_interval(att, R, V, rho)
    ok = np.isfinite(lo)
    assert ok.all(), "opposition region should span 1e-3 .. 1 AU here"
    assert np.log10(rho[ok].max() / rho[ok].min()) > 2.5

    # At small rho the object sits next to the observer at ~1 AU, so the
    # largest bound |rho_dot| is set by the local escape speed.
    lo0, hi0 = admissible_rhodot_interval(att, R, V, np.array(1e-4))
    v_max = max(abs(float(lo0)), abs(float(hi0)))
    assert 0.3 * V_ESC_1AU < v_max < 1.05 * V_ESC_1AU
    # Order of magnitude: ~0.024 AU/day.
    assert 1e-3 < v_max < 1e-1

    # A momentarily co-moving observer recovers the escape speed exactly.
    att_still = att_opposition()
    lo_s, hi_s = admissible_rhodot_interval(att_still, R, np.zeros(3), np.array(0.0))
    assert np.isclose(float(hi_s), V_ESC_1AU, rtol=1e-12)
    assert np.isclose(float(lo_s), -V_ESC_1AU, rtol=1e-12)


def test_region_shrinks_with_range_and_terminates():
    """Width must fall monotonically with rho at opposition and vanish at r=2."""
    att, R, V = att_opposition(), R_OBS, V_OBS
    rho = np.linspace(0.01, 0.99, 300)
    lo, hi = admissible_rhodot_interval(att, R, V, rho)
    w = hi - lo
    assert np.all(np.diff(w) < 0.0)
    # |W_perp| = |V_obs| here, so the region ends where 2 mu / r = |V_obs|^2.
    rho_edge = 2.0 * MU_SUN / np.dot(V, V) - 1.0
    lo_in, _ = admissible_rhodot_interval(att, R, V, np.array(rho_edge * 0.999))
    lo_out, _ = admissible_rhodot_interval(att, R, V, np.array(rho_edge * 1.001))
    assert np.isfinite(lo_in) and np.isnan(lo_out)


def test_fast_mover_region_is_smaller_than_slow_mover():
    """Larger sky motion eats the speed budget, shrinking the region."""
    slow = Attributable(ra=0.0, dec=0.0, ra_rate=1.0e-3, dec_rate=0.0,
                        epoch=EPOCH, v_mag=21.0)
    fast = Attributable(ra=0.0, dec=0.0, ra_rate=2.0e-2, dec_rate=0.0,
                        epoch=EPOCH, v_mag=21.0)
    a_slow = float(np.sum(build_grid(slow, R_OBS, V_OBS, 0.05, 0.8, 400, 50).weight))
    a_fast = float(np.sum(build_grid(fast, R_OBS, V_OBS, 0.05, 0.8, 400, 50).weight))
    assert 0.0 <= a_fast < a_slow


# ===========================================================================
# build_grid structure, degenerate cases, validation
# ===========================================================================


def test_grid_structure_and_nodes_are_inside_the_region():
    att, R, V = att_generic(), R_OBS, V_OBS
    n_rho, n_rhodot = 60, 25
    g = build_grid(att, R, V, 0.02, 5.0, n_rho, n_rhodot)

    assert isinstance(g, RangingGrid)
    assert g.n_rho == n_rho and g.n_rhodot == n_rhodot
    assert g.rho.shape == g.rho_dot.shape == g.weight.shape
    assert g.rho.size % n_rhodot == 0
    assert 0 < g.rho.size <= n_rho * n_rhodot
    assert np.all(np.isfinite(g.rho)) and np.all(np.isfinite(g.rho_dot))
    assert np.all(np.isfinite(g.weight)) and np.all(g.weight > 0.0)
    assert np.all(g.rho >= 0.02) and np.all(g.rho <= 5.0)

    # Every node is strictly bound.
    assert np.all(brute_energy(att, R, V, g.rho, g.rho_dot) < 0.0)

    # Nodes are midpoints of equal sub-cells of [lo, hi].
    lo, hi = admissible_rhodot_interval(att, R, V, g.rho)
    frac = (g.rho_dot - lo) / (hi - lo)
    expect = (np.arange(n_rhodot) + 0.5) / n_rhodot
    assert np.allclose(np.sort(np.unique(np.round(frac, 12))), expect, atol=1e-9)


def test_grid_rho_nodes_are_log_spaced():
    att, R, V = att_opposition(), R_OBS, V_OBS
    g = build_grid(att, R, V, 0.05, 0.8, 50, 3)
    rho_u = np.unique(g.rho)
    assert rho_u.size == 50
    dlog = np.diff(np.log(rho_u))
    assert np.allclose(dlog, dlog[0], rtol=1e-10)
    assert np.isclose(rho_u[0], 0.05) and np.isclose(rho_u[-1], 0.8)


def test_fully_empty_region_returns_zero_length_grid():
    att, R, V = att_opposition(), R_OBS, V_OBS
    g = build_grid(att, R, V, rho_min=5.0, rho_max=100.0, n_rho=50, n_rhodot=10)
    assert g.rho.size == 0 and g.rho_dot.size == 0 and g.weight.size == 0
    assert g.n_rho == 50 and g.n_rhodot == 10
    assert float(np.sum(g.weight)) == 0.0


def test_partially_empty_region_drops_only_empty_ranges():
    att, R, V = att_opposition(), R_OBS, V_OBS
    n_rho, n_rhodot = 100, 7
    g = build_grid(att, R, V, 0.05, 20.0, n_rho, n_rhodot)
    assert 0 < g.rho.size < n_rho * n_rhodot
    rho_u = np.unique(g.rho)
    lo, hi = admissible_rhodot_interval(att, R, V, np.geomspace(0.05, 20.0, n_rho))
    assert rho_u.size == int(np.sum(np.isfinite(lo) & (hi - lo > 0)))
    assert rho_u.max() < 1.01  # region ends at rho = 1 AU for this geometry


def test_single_cell_grid():
    att, R, V = att_opposition(), R_OBS, V_OBS
    g = build_grid(att, R, V, 0.1, 0.4, n_rho=1, n_rhodot=1)
    assert g.rho.size == 1
    assert np.isclose(g.rho[0], np.sqrt(0.1 * 0.4))
    lo, hi = admissible_rhodot_interval(att, R, V, g.rho)
    assert np.isclose(g.rho_dot[0], 0.5 * (lo[0] + hi[0]))
    assert np.isclose(g.weight[0], (0.4 - 0.1) * (hi[0] - lo[0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rho_min=0.0, rho_max=1.0),
        dict(rho_min=-1.0, rho_max=1.0),
        dict(rho_min=1.0, rho_max=1.0),
        dict(rho_min=2.0, rho_max=1.0),
        dict(rho_min=0.1, rho_max=1.0, n_rho=0),
        dict(rho_min=0.1, rho_max=1.0, n_rhodot=0),
    ],
)
def test_build_grid_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        build_grid(att_opposition(), R_OBS, V_OBS, **kwargs)


def test_interval_rejects_bad_observer_state():
    with pytest.raises(ValueError):
        admissible_rhodot_interval(att_opposition(), np.zeros(2), V_OBS, np.array(0.5))
