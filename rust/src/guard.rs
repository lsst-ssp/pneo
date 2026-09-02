//! Exact port of WP2's conditioning guard.
//!
//! `transforms.jacobian_elements_wrt_state` refuses to report a Jacobian when
//! the forward matrix `d(state)/d(elements)` is too ill-conditioned to invert
//! accurately, using the infinity-norm condition estimate
//! `cond_inf = ||J||_inf ||J^-1||_inf` against `_COND_MAX = 1e12`.  Those nodes
//! come back NaN and `classifier.classify` zeroes them and counts them in
//! `n_masked_det`.
//!
//! The Rust kernel gets its Jacobian from the Delaunay closed form and has no
//! such failure mode, so for **behavioural parity** it reproduces the guard
//! itself: [`cond_inf_6x6`] builds the same analytic 6x6 and estimates the same
//! condition number.  Doing this at every node would reinstate exactly the cost
//! the closed form removes (31 ms/tracklet, measured), so it runs only where it
//! can possibly matter — see [`needs_exact_guard`], which fires on ~16.5% of
//! real ranging nodes.  **That prefilter is sound on the reachable manifold
//! only**, which is a deliberate and documented premise; see
//! [`cond_upper_bound`] for why no cheap global bound exists.
//!
//! Measured result (`validation/backend_parity.py` over the whole gate_v0
//! catalogue): `n_masked_det` agrees with the Python reference **exactly** —
//! 397 900 masked nodes on both sides, zero disagreements out of 8.0e8 nodes.
//!
//! Two things make this cheap and exact at once:
//!
//! * no Kepler solver is needed.  Python's path goes elements -> `M` ->
//!   `_kepler_E` -> `E`; here `E` comes directly from the state via the true
//!   anomaly, which is the same value up to the `M` round trip (~1e-15).
//! * only the *decision* matters, never the matrix, so a plain Gauss-Jordan
//!   inverse suffices; it agrees with LAPACK's on the condition estimate to far
//!   better than the factor-of-40 gaps between adjacent grid nodes near the
//!   escape boundary.

// `!(x <= y)` rather than `x > y` throughout: the negated form is the one
// that treats NaN as "out of range", matching numpy's masking semantics.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::consts::{COND_MAX, MU_SUN};

/// `transforms._ECC_DEGENERATE`.
const ECC_DEGENERATE: f64 = 1e-14;
/// `transforms._INC_DEGENERATE`.
const INC_DEGENERATE: f64 = 1e-14;

/// Fire the exact estimate when [`cond_upper_bound`] exceeds this.
///
/// Equal to `_COND_MAX` itself: the bound is an upper bound on `cond_inf`, so
/// anything that could reach the ceiling is checked exactly.
const PREFILTER_THRESHOLD: f64 = COND_MAX;

// Coefficients of [`cond_upper_bound`], each the measured worst case rounded
// generously up.  See that function for the (load-bearing) reachability premise
// and for how these were measured.
const C_PERIHELION: f64 = 1e10;
const C_CIRCULAR: f64 = 1e8;
const C_EQUATORIAL: f64 = 1e8;

/// Cheap upper bound on `cond_inf(d(state)/d(elements))`, evaluated at every node.
///
/// **This is an upper bound on the REACHABLE manifold only, and that premise is
/// load-bearing.**  No cheap closed form bounds `cond_inf` over all of
/// `(a, e, i, Omega, omega, M)`: the degeneracy mechanisms compound, and a
/// dense sweep of that space reaches `cond_inf = 5.8e62` while any product- or
/// max-form tight enough to be a useful filter is violated by 20+ decades.
/// What makes a cheap bound possible is that the kernel never evaluates
/// abstract elements — it evaluates the state implied by
/// `(A, rho, rho_dot)` with `rho <= rho_max`, so the heliocentric distance `r`
/// is known at the node and is bounded by `rho_max + |R_obs|`.  The bound is
/// therefore written in terms of `r`, and validated by sampling the reachable
/// manifold itself (see the tests and the measurement below).
///
/// The controlling quantity is
///
/// ```text
///     w = a / (r * sqrt(1 - e^2))
/// ```
///
/// which is what actually appears in the entries of `d(state)/d(elements)`
/// (`dE/dM = a/r`, `dY/dM = a^2 b/r`, `dYd/de ~ s/(b r)`).  It is `O(1)` for an
/// ordinary orbit, grows like `(1 - e)^{-3/2}` at perihelion passage of an
/// eccentric orbit, and like `a/r` for an orbit far larger than the sampled
/// range — so it subsumes the `a -> inf` mechanism on the reachable manifold,
/// where `r <= rho_max + |R_obs|` forces large `a` to mean large `a/r`.
///
/// The two terms `w` cannot see are the degeneracies at which `w` stays `O(1)`
/// while the element frame itself becomes undefined:
///
/// * `e -> 0` — periapsis undefined, `cond ~ 1/e`;
/// * `sin i -> 0` — line of nodes undefined, `cond ~ 1/sin i`.
///
/// Degenerate inputs (`e = 0`, `sin i = 0`, `e = 1`, `r = 0`) give `+inf` and
/// therefore always take the exact path.
///
/// # Measurement
///
/// | term | measured worst case | coefficient | margin |
/// |---|---|---|---|
/// | `w^{5/2}` | `cond <= 8.41e7 w^{5/2}` | `1e10` | 119x |
/// | `1/e` | `cond <= 2.8e5 / e` | `1e8` | 360x |
/// | `1/sin i` | `cond <= 2.5e5 / sin i` | `1e8` | 400x |
///
/// The `w` row comes from **58 201 040 nodes sampled off the reachable
/// manifold** — observer on a ~1 AU orbit, solar elongation down to 5 deg,
/// on-sky rates to ~100 deg/day, `rho` over the full `[0.01, 100]` AU range and
/// `rho_dot` across the admissible interval — with `cond_inf` evaluated
/// **exactly** at every one.  That sample does reach perihelion passage
/// (`r/q` down to 1.000, `1 - e` down to 1.3e-8) and `cond_inf` up to 5.06e25,
/// and the bound has zero false negatives on it.  The `1/e` and `1/sin i` rows
/// come from single-mechanism synthetic sweeps holding one variable at
/// `1e-14 .. 1e-3`.
///
/// History (kept deliberately): the first version of this bound used a
/// `a^{3/2}/(1 - e)` perihelion term.  That was wrong twice over — the true
/// perihelion scaling is far steeper than `1/(1 - e)`, and the `a^{3/2}` factor
/// cancels the blow-up exactly where the conditioning is worst — and it needed
/// `C = 1.9e9` rather than the `1e6` it shipped with.  It happened to have zero
/// false negatives on the reachable manifold anyway, but only by accident.
#[inline(always)]
pub fn cond_upper_bound(a: f64, e: f64, sin_i: f64, r: f64) -> f64 {
    // Up front, because `f64::max` *discards* a NaN operand rather than
    // propagating it: a single NaN term would otherwise fall through to the
    // others and could under-report the bound.  Anything non-finite takes the
    // exact path.  (Unreachable from the kernel, which only calls this for
    // nodes that already passed the bound-orbit test, but the guard must not
    // depend on that.)
    if !(a.is_finite() && e.is_finite() && sin_i.is_finite() && r.is_finite()) {
        return f64::INFINITY;
    }
    let b = (1.0 - e * e).sqrt();
    let w = a / (r * b);
    if !w.is_finite() {
        return f64::INFINITY;
    }
    let t_peri = C_PERIHELION * w * w * w.sqrt();
    let t_circ = C_CIRCULAR / e;
    let t_equ = C_EQUATORIAL / sin_i;
    t_peri.max(t_circ).max(t_equ)
}

/// Whether a node needs the exact 6x6 condition estimate.
///
/// Fires on ~16.5% of real gate_v0 ranging nodes (and ~18% of the adversarial
/// reachable sample).  Non-finite or non-positive scores fire too.
#[inline(always)]
pub fn needs_exact_guard(a: f64, e: f64, sin_i: f64, r: f64) -> bool {
    !(cond_upper_bound(a, e, sin_i, r) <= PREFILTER_THRESHOLD)
}

/// Angles needed by the forward Jacobian, taken straight from the state.
///
/// Port of the `Omega`/`omega`/`nu`/`E` half of
/// `transforms.elements_from_state`, including its degenerate-locus
/// conventions (equatorial -> `Omega = 0` with the `+x` reference direction;
/// circular -> `omega = 0`).
#[allow(clippy::too_many_arguments)]
fn angles_from_state(
    r_ecl: [f64; 3],
    h_vec: [f64; 3],
    h: f64,
    e_vec: [f64; 3],
    e: f64,
) -> (f64, f64, f64) {
    let h_hat = [h_vec[0] / h, h_vec[1] / h, h_vec[2] / h];
    let n_vec = [-h_vec[1], h_vec[0], 0.0];
    let n = (n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1]).sqrt();

    let equatorial = n <= INC_DEGENERATE * h.max(f64::MIN_POSITIVE);
    let node_hat = if equatorial {
        [1.0, 0.0, 0.0]
    } else {
        [n_vec[0] / n, n_vec[1] / n, 0.0]
    };
    let omega_big = if equatorial {
        0.0
    } else {
        h_vec[0].atan2(-h_vec[1])
    };

    let circular = e <= ECC_DEGENERATE;
    let peri_hat = if circular {
        node_hat
    } else {
        [e_vec[0] / e, e_vec[1] / e, e_vec[2] / e]
    };

    let ahead_node = cross(h_hat, node_hat);
    let omega_small = if circular {
        0.0
    } else {
        dot(ahead_node, peri_hat).atan2(dot(node_hat, peri_hat))
    };

    let ahead_peri = cross(h_hat, peri_hat);
    let nu = dot(ahead_peri, r_ecl).atan2(dot(peri_hat, r_ecl));
    let ecc_anom = 2.0
        * ((1.0 - e).max(0.0).sqrt() * (0.5 * nu).sin())
            .atan2((1.0 + e).max(0.0).sqrt() * (0.5 * nu).cos());

    (wrap2pi(omega_big), wrap2pi(omega_small), ecc_anom)
}

#[inline(always)]
fn wrap2pi(x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let y = x.rem_euclid(two_pi);
    if y >= two_pi {
        0.0
    } else {
        y
    }
}

#[inline(always)]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

#[inline(always)]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `transforms._pqw_basis`: the perifocal basis `(P, W, h_hat)`.
fn pqw_basis(i: f64, om_big: f64, om_small: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (c_o, s_o) = (om_big.cos(), om_big.sin());
    let (ci, si) = (i.cos(), i.sin());
    let (cw, sw) = (om_small.cos(), om_small.sin());
    (
        [c_o * cw - s_o * sw * ci, s_o * cw + c_o * sw * ci, sw * si],
        [
            -c_o * sw - s_o * cw * ci,
            -s_o * sw + c_o * cw * ci,
            cw * si,
        ],
        [s_o * si, -c_o * si, ci],
    )
}

/// `transforms.jacobian_state_wrt_elements` for one node.
///
/// Rows are `(r, v)`, columns `(a, e, i, Omega, omega, M)`, ecliptic J2000.
/// `ecc_anom` is `E` (taken from the state rather than solved from `M`).
pub fn jacobian_state_wrt_elements(
    a: f64,
    e: f64,
    i: f64,
    om_big: f64,
    om_small: f64,
    ecc_anom: f64,
) -> [[f64; 6]; 6] {
    let mu = MU_SUN;
    let (c_e, s_e) = (ecc_anom.cos(), ecc_anom.sin());
    let b = (1.0 - e * e).max(1e-300).sqrt();
    let r = a * (1.0 - e * c_e);
    let s = (mu * a).sqrt();

    let de_de = a * s_e / r;
    let de_dm = a / r;

    let dr_de = a * (-c_e + a * e * s_e * s_e / r);
    let dr_dm = a * a * e * s_e / r;

    let x = a * (c_e - e);
    let y = a * b * s_e;
    let xd = -s * s_e / r;
    let yd = s * b * c_e / r;

    let dx_da = c_e - e;
    let dx_de = -a * (1.0 + a * s_e * s_e / r);
    let dx_dm = -a * a * s_e / r;

    let dy_da = b * s_e;
    let dy_de = a * (-e * s_e / b + b * c_e * de_de);
    let dy_dm = a * b * c_e * de_dm;

    let dxd_da = s * s_e / (2.0 * a * r);
    let dxd_de = -(s * c_e * de_de / r - s * s_e * dr_de / (r * r));
    let dxd_dm = -(s * c_e * de_dm / r - s * s_e * dr_dm / (r * r));

    let dyd_da = -s * b * c_e / (2.0 * a * r);
    let dyd_de =
        s * ((-e / b) * c_e / r - b * s_e * de_de / r - b * c_e * dr_de / (r * r));
    let dyd_dm = s * (-b * s_e * de_dm / r - b * c_e * dr_dm / (r * r));

    let (p, w, h_hat) = pqw_basis(i, om_big, om_small);
    let r_vec = [
        x * p[0] + y * w[0],
        x * p[1] + y * w[1],
        x * p[2] + y * w[2],
    ];
    let v_vec = [
        xd * p[0] + yd * w[0],
        xd * p[1] + yd * w[1],
        xd * p[2] + yd * w[2],
    ];

    let z_hat = [0.0, 0.0, 1.0];
    let n_hat = [om_big.cos(), om_big.sin(), 0.0];
    let dp_di = cross(n_hat, p);
    let dw_di = cross(n_hat, w);

    let mut j = [[0.0f64; 6]; 6];
    for (col, (dxp, dyp, dxdp, dydp)) in [
        (0usize, (dx_da, dy_da, dxd_da, dyd_da)),
        (1usize, (dx_de, dy_de, dxd_de, dyd_de)),
        (5usize, (dx_dm, dy_dm, dxd_dm, dyd_dm)),
    ] {
        for k in 0..3 {
            j[k][col] = dxp * p[k] + dyp * w[k];
            j[k + 3][col] = dxdp * p[k] + dydp * w[k];
        }
    }
    for k in 0..3 {
        j[k][2] = x * dp_di[k] + y * dw_di[k];
        j[k + 3][2] = xd * dp_di[k] + yd * dw_di[k];
    }
    let dr_dbig = cross(z_hat, r_vec);
    let dv_dbig = cross(z_hat, v_vec);
    let dr_dsmall = cross(h_hat, r_vec);
    let dv_dsmall = cross(h_hat, v_vec);
    for k in 0..3 {
        j[k][3] = dr_dbig[k];
        j[k + 3][3] = dv_dbig[k];
        j[k][4] = dr_dsmall[k];
        j[k + 3][4] = dv_dsmall[k];
    }
    j
}

/// Infinity-norm condition estimate of a 6x6, or `None` if it is singular or
/// non-finite (both of which the Python path also turns into a mask).
pub fn cond_inf_6x6(j: [[f64; 6]; 6]) -> Option<f64> {
    for row in &j {
        for v in row {
            if !v.is_finite() {
                return None;
            }
        }
    }
    // Gauss-Jordan with partial pivoting; `inv` starts as the identity and ends
    // as J^-1.  The infinity norms are max row sums, so the row permutation the
    // pivoting applies does not affect either of them.
    let mut m = j;
    let mut inv = [[0.0f64; 6]; 6];
    for k in 0..6 {
        inv[k][k] = 1.0;
    }
    for col in 0..6 {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for row in (col + 1)..6 {
            let v = m[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if !(best > 0.0) {
            return None; // exactly singular
        }
        if piv != col {
            m.swap(piv, col);
            inv.swap(piv, col);
        }
        let d = m[col][col];
        for c in 0..6 {
            m[col][c] /= d;
            inv[col][c] /= d;
        }
        for row in 0..6 {
            if row == col {
                continue;
            }
            let f = m[row][col];
            if f == 0.0 {
                continue;
            }
            for c in 0..6 {
                m[row][c] -= f * m[col][c];
                inv[row][c] -= f * inv[col][c];
            }
        }
    }
    let norm = |x: &[[f64; 6]; 6]| {
        x.iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let fwd = norm(&j);
    let back = norm(&inv);
    let c = fwd * back;
    if c.is_nan() {
        None
    } else {
        Some(c)
    }
}

/// The exact WP2 decision for one node: `true` when the Jacobian must be
/// masked (`n_masked_det`).
///
/// `r_ecl`, `h_vec`, `e_vec` and the scalars come from the kernel's own
/// element evaluation, so nothing is recomputed.
#[allow(clippy::too_many_arguments)]
pub fn det_is_masked(
    r_ecl: [f64; 3],
    h_vec: [f64; 3],
    h: f64,
    e_vec: [f64; 3],
    a: f64,
    e: f64,
    inc: f64,
) -> bool {
    let (om_big, om_small, ecc_anom) = angles_from_state(r_ecl, h_vec, h, e_vec, e);
    let j = jacobian_state_wrt_elements(a, e, inc, om_big, om_small, ecc_anom);
    match cond_inf_6x6(j) {
        None => true,
        Some(c) => !(c <= COND_MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
        let mut c = [[0.0; 6]; 6];
        for i in 0..6 {
            for k in 0..6 {
                for j in 0..6 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn inverse(j: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
        // Same elimination as cond_inf_6x6, exposed for the round-trip test.
        let mut m = j;
        let mut inv = [[0.0f64; 6]; 6];
        for k in 0..6 {
            inv[k][k] = 1.0;
        }
        for col in 0..6 {
            let mut piv = col;
            let mut best = m[col][col].abs();
            for row in (col + 1)..6 {
                if m[row][col].abs() > best {
                    best = m[row][col].abs();
                    piv = row;
                }
            }
            m.swap(piv, col);
            inv.swap(piv, col);
            let d = m[col][col];
            for c in 0..6 {
                m[col][c] /= d;
                inv[col][c] /= d;
            }
            for row in 0..6 {
                if row == col {
                    continue;
                }
                let f = m[row][col];
                for c in 0..6 {
                    m[row][c] -= f * m[col][c];
                    inv[row][c] -= f * inv[col][c];
                }
            }
        }
        inv
    }

    #[test]
    fn inverse_round_trips() {
        let j = jacobian_state_wrt_elements(2.7, 0.15, 0.17, 1.1, 2.2, 0.6);
        let p = matmul(j, inverse(j));
        for i in 0..6 {
            for k in 0..6 {
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((p[i][k] - want).abs() < 1e-10, "{i},{k} = {}", p[i][k]);
            }
        }
    }

    #[test]
    fn well_conditioned_orbit_is_not_masked() {
        let c = cond_inf_6x6(jacobian_state_wrt_elements(2.7, 0.15, 0.17, 1.1, 2.2, 0.6));
        let c = c.expect("nonsingular");
        assert!(c > 1.0 && c < 1e9, "cond = {c}");
    }

    #[test]
    fn degenerate_orbits_are_masked() {
        // e -> 0 and sin i -> 0 both blow the condition number past 1e12.
        let c = cond_inf_6x6(jacobian_state_wrt_elements(
            2.7, 1e-12, 0.17, 1.1, 2.2, 0.6,
        ))
        .unwrap();
        assert!(c > COND_MAX, "cond = {c}");
        let c = cond_inf_6x6(jacobian_state_wrt_elements(
            2.7, 0.3, 1e-12, 1.1, 2.2, 0.6,
        ))
        .unwrap();
        assert!(c > COND_MAX, "cond = {c}");
    }

    #[test]
    fn prefilter_never_misses_a_masked_node_on_the_reachable_manifold() {
        // The invariant the prefilter must satisfy.  It is checked on the
        // REACHABLE manifold — the states the kernel can actually evaluate,
        // built from (A, rho, rho_dot) exactly as classify_one does — because
        // no cheap bound holds over abstract (a, e, i, Omega, omega, M); see
        // `cond_upper_bound`.  The sweep spans solar elongations from 5 deg
        // (so perihelion-side nodes with r << a really do occur), on-sky rates
        // over four decades, and the full rho range.
        let mut n_masked = 0usize;
        let mut n_total = 0usize;
        let mut n_fire = 0usize;
        for i_obs in 0..7 {
            let th = (i_obs as f64) * 0.897;
            let r_obs = [th.cos(), th.sin(), 0.0];
            let v_obs = [-th.sin() * 0.0172, th.cos() * 0.0172, 0.0];
            for i_el in 0..8 {
                // elongation 5 .. 180 deg, measured from the anti-solar point
                let elong = 5.0 + (i_el as f64) * 25.0;
                let delta = (180.0 - elong).to_radians();
                let e1 = [-r_obs[0], -r_obs[1], -r_obs[2]];
                let perp = [-e1[1], e1[0], 0.0];
                let pn = (perp[0] * perp[0] + perp[1] * perp[1]).sqrt();
                let u = [
                    delta.cos() * -e1[0] + delta.sin() * perp[0] / pn,
                    delta.cos() * -e1[1] + delta.sin() * perp[1] / pn,
                    delta.cos() * -e1[2],
                ];
                let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                let ra = u[1].atan2(u[0]);
                let dec = (u[2] / un).asin();
                for i_rate in 0..5 {
                    let rate = 10f64.powf(-4.0 + (i_rate as f64) * 1.05);
                    for i_ph in 0..3 {
                        let ph = (i_ph as f64) * 1.047;
                        let cfg = GridCfgLike {
                            ra,
                            dec,
                            ra_rate: rate * ph.cos() / dec.cos().max(1e-6).powi(2),
                            dec_rate: rate * ph.sin(),
                        };
                        let (t, m, f) = sweep_nodes(&cfg, r_obs, v_obs);
                        n_total += t;
                        n_masked += m;
                        n_fire += f;
                    }
                }
            }
        }
        assert!(n_total > 200_000, "only {n_total} nodes swept");
        assert!(n_masked > 50, "only {n_masked} masked nodes reached");
        let frac = (n_fire as f64) / (n_total as f64);
        assert!(frac < 0.35, "prefilter fires on {:.1}% of nodes", frac * 100.0);
    }

    struct GridCfgLike {
        ra: f64,
        dec: f64,
        ra_rate: f64,
        dec_rate: f64,
    }

    /// Walk the ranging nodes of one geometry, asserting the prefilter invariant
    /// at each.  Returns (nodes, masked, prefilter firings).
    fn sweep_nodes(
        c: &GridCfgLike,
        r_obs: [f64; 3],
        v_obs: [f64; 3],
    ) -> (usize, usize, usize) {
        use crate::consts::MU_SUN;
        let (n_rho, n_rhodot) = (120usize, 12usize);
        let (u, ua, ud) = crate::kernel::tangent_basis(c.ra, c.dec);
        let mvec = [
            c.ra_rate * ua[0] + c.dec_rate * ud[0],
            c.ra_rate * ua[1] + c.dec_rate * ud[1],
            c.ra_rate * ua[2] + c.dec_rate * ud[2],
        ];
        let (lo_l, hi_l) = (0.01f64.log10(), 100.0f64.log10());
        let (mut n, mut nm, mut nf) = (0usize, 0usize, 0usize);
        for k in 0..n_rho {
            let rho =
                10f64.powf(lo_l + (hi_l - lo_l) * (k as f64) / ((n_rho - 1) as f64));
            let wv = [
                v_obs[0] + rho * mvec[0],
                v_obs[1] + rho * mvec[1],
                v_obs[2] + rho * mvec[2],
            ];
            let rv = [
                r_obs[0] + rho * u[0],
                r_obs[1] + rho * u[1],
                r_obs[2] + rho * u[2],
            ];
            let rn = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
            let b = wv[0] * u[0] + wv[1] * u[1] + wv[2] * u[2];
            let w2 = wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2];
            let disc = b * b - w2 + 2.0 * MU_SUN / rn;
            if !(disc >= 0.0) {
                continue;
            }
            let sq = disc.sqrt();
            let (rlo, rhi) = (-b - sq, -b + sq);
            let width = rhi - rlo;
            if !(width.is_finite() && width > 0.0) {
                continue;
            }
            for j in 0..n_rhodot {
                let rd = rlo + width * ((j as f64) + 0.5) / (n_rhodot as f64);
                let vv = [
                    (v_obs[0] + rd * u[0]) + rho * mvec[0],
                    (v_obs[1] + rd * u[1]) + rho * mvec[1],
                    (v_obs[2] + rd * u[2]) + rho * mvec[2],
                ];
                let (ce, se) = crate::consts::obliquity_cos_sin();
                let re = [rv[0], ce * rv[1] + se * rv[2], -se * rv[1] + ce * rv[2]];
                let ve = [vv[0], ce * vv[1] + se * vv[2], -se * vv[1] + ce * vv[2]];
                let el = match osculating(re, ve) {
                    None => continue,
                    Some(x) => x,
                };
                let (a, e, inc, hv, hh, ev) = el;
                n += 1;
                let rr = (re[0] * re[0] + re[1] * re[1] + re[2] * re[2]).sqrt();
                let fires = needs_exact_guard(a, e, inc.sin(), rr);
                if fires {
                    nf += 1;
                }
                if det_is_masked(re, hv, hh, ev, a, e, inc) {
                    nm += 1;
                    assert!(
                        fires,
                        "prefilter MISSED a masked node: \
                         a={a} e={e} i={inc} r={rr} bound={}",
                        cond_upper_bound(a, e, inc.sin(), rr)
                    );
                }
            }
        }
        (n, nm, nf)
    }

    /// (a, e, i, h_vec, |h|, e_vec) from a state, as the kernel computes them.
    #[allow(clippy::type_complexity)]
    fn osculating(
        r: [f64; 3],
        v: [f64; 3],
    ) -> Option<(f64, f64, f64, [f64; 3], f64, [f64; 3])> {
        let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let a = 1.0 / (2.0 / rn - v2 / MU_SUN);
        let hv = cross(r, v);
        let h = (hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2]).sqrt();
        let cvh = cross(v, hv);
        let ev = [
            cvh[0] / MU_SUN - r[0] / rn,
            cvh[1] / MU_SUN - r[1] / rn,
            cvh[2] / MU_SUN - r[2] / rn,
        ];
        let e = (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]).sqrt();
        if !(a.is_finite() && a > 0.0 && e < 1.0 && e.is_finite() && h > 0.0) {
            return None;
        }
        let inc = (hv[2] / h).clamp(-1.0, 1.0).acos();
        Some((a, e, inc, hv, h, ev))
    }

    #[test]
    fn prefilter_covers_perihelion_and_aphelion_explicitly() {
        // The reviewer's finding: the previous sweep omitted M = 0, and the old
        // bound self-violated 64 times once perihelion was included.  Sweep
        // (a, e, i) at BOTH E = 0 (perihelion) and E = pi (aphelion), keeping
        // only combinations whose implied r = a(1 - e cos E) is reachable,
        // i.e. lies within [rho_min, rho_max + |R_obs|] ~ [0.01, 101] AU.
        // That restriction is the documented premise of `cond_upper_bound`.
        let mut n_checked = 0usize;
        let mut n_masked = 0usize;
        let mut a = 0.05f64;
        while a < 3.0e4 {
            for e_exp in -13..0 {
                for &e in &[10f64.powi(e_exp), 1.0 - 10f64.powi(e_exp)] {
                    if !(0.0..1.0).contains(&e) {
                        continue;
                    }
                    for i_exp in -13..1 {
                        let inc = 10f64.powi(i_exp).min(1.4);
                        for &ecc_anom in &[0.0, std::f64::consts::PI] {
                            let r = a * (1.0 - e * ecc_anom.cos());
                            if !(0.01..=101.0).contains(&r) {
                                continue; // not reachable
                            }
                            for &(ob, os) in
                                &[(0.7, 2.1), (4.0, 5.5), (1.0, 0.2)]
                            {
                                n_checked += 1;
                                let c = cond_inf_6x6(jacobian_state_wrt_elements(
                                    a, e, inc, ob, os, ecc_anom,
                                ));
                                let masked = match c {
                                    None => true,
                                    Some(v) => !(v <= COND_MAX),
                                };
                                if masked {
                                    n_masked += 1;
                                    assert!(
                                        needs_exact_guard(a, e, inc.sin(), r),
                                        "missed a={a} e={e} i={inc} \
                                         E={ecc_anom} r={r} (cond={c:?}) \
                                         bound={}",
                                        cond_upper_bound(a, e, inc.sin(), r)
                                    );
                                }
                            }
                        }
                    }
                }
            }
            a *= 1.6;
        }
        assert!(n_checked > 5_000, "only {n_checked} reachable combinations");
        assert!(n_masked > 500, "only {n_masked} masked of {n_checked}");
    }

    #[test]
    fn prefilter_spares_the_physical_bulk() {
        // r is the node's heliocentric distance; ordinary orbits observed at
        // ordinary distances stay on the fast path.
        for &(a, e, si, r) in &[
            (2.7, 0.15, 0.17, 2.5),
            (39.0, 0.25, 0.30, 30.0),
            (0.9, 0.6, 0.5, 1.0),
            (3.2, 0.09, 0.4, 3.0),
        ] {
            assert!(!needs_exact_guard(a, e, si, r), "a={a} e={e} si={si} r={r}");
        }
        for &(a, e, si, r) in &[
            (730.0, 0.998, 0.99, 3.0),   // near-parabolic, far from the Sun
            (2.0, 1.0 - 1e-8, 0.4, 0.1), // sungrazer at perihelion
            (3.0, 1e-9, 0.5, 3.0),       // circular degeneracy
            (3.0, 0.5, 1e-9, 3.0),       // equatorial degeneracy
        ] {
            assert!(needs_exact_guard(a, e, si, r), "a={a} e={e} si={si} r={r}");
        }
        // Degenerate inputs must fire, never fall through a discarded NaN.
        for &(a, e, si, r) in &[
            (2.0, 1.0, 0.4, 1.0),        // e = 1 exactly -> b = 0
            (2.0, 0.3, 0.4, 0.0),        // r = 0
            (f64::NAN, 0.3, 0.4, 1.0),
            (2.0, f64::NAN, 0.4, 1.0),
            (2.0, 0.3, f64::NAN, 1.0),
            (2.0, 0.3, 0.4, f64::NAN),
            (f64::INFINITY, 0.3, 0.4, 1.0),
        ] {
            assert!(
                needs_exact_guard(a, e, si, r),
                "degenerate input did not fire: a={a} e={e} si={si} r={r}"
            );
        }
    }
}
