//! The fused per-node classify kernel.
//!
//! One pass per quadrature node, no intermediate arrays:
//!
//! ```text
//! admissible rho_dot interval -> node placement -> tangent basis
//!   -> topocentric->heliocentric -> equ->ecl rotation -> (a, e, i)
//!   -> phase angle -> H(V, rho) -> density lookups -> |J| -> accumulate
//! ```
//!
//! Every arithmetic expression is written in the same association order as the
//! numpy expression it replaces (see the `parity:` comments), so the two
//! backends differ only by libm rounding.
//!
//! **Jacobian.**  Instead of building and inverting the 6x6
//! `d(elements)/d(state)` per node, the composed determinant is evaluated in
//! closed form (DESIGN §2.3 + the Delaunay identity):
//!
//! ```text
//! |det d(a,e,i,Om,om,M)/d(A, rho, rho_dot)|
//!     = |det d(state_equ)/d(A, rho, rho_dot)| * |det d(E)/d(state)|
//!     = rho^4 cos^2(dec)  *  2 / (mu^{3/2} sqrt(a) e sin i)
//! ```
//!
//! times `(1 - e)` when the component is parameterized in `(q, e, i)`.  The
//! first factor is exact (the observable block is block-triangular with two
//! copies of `[rho u_a, rho u_d, u]`, whose determinant is `rho^2 cos(dec)`);
//! the second is the standard Delaunay result
//! `d(L,G,H)/d(a,e,i) = 1/2 mu^{3/2} sqrt(a) e sin i` with `d(L,G,H,l,g,h)/d(r,v) = 1`.
//! No Kepler solver is needed anywhere: `(a, e, i)` come straight from the state.

// `!(x >= y)` / `!(x <= y)` rather than the positive form throughout: the
// negated comparison treats NaN as "out of range", which is exactly numpy's
// masking semantics and is what the reference path relies on.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::consts::*;
use crate::guard;
use crate::model::Model;

/// Neumaier (compensated Kahan-Babuska) accumulator.
///
/// Chosen over pairwise summation because it is *order-independent in accuracy*
/// (error bound independent of n) while still being a strictly sequential,
/// deterministic left-to-right traversal — two runs on the same inputs give
/// bit-identical results, and the node order is the Python node order
/// (`rho` major, `rho_dot` minor).  Non-finite running sums stop the
/// compensation so that an overflow to +/-inf propagates the way
/// `np.sum` does (the classifier's "overflow" posterior status depends on it).
#[derive(Clone, Copy, Debug, Default)]
pub struct Neumaier {
    s: f64,
    c: f64,
}

impl Neumaier {
    #[inline(always)]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if t.is_finite() {
            if self.s.abs() >= x.abs() {
                self.c += (self.s - t) + x;
            } else {
                self.c += (x - t) + self.s;
            }
        } else {
            self.c = 0.0;
        }
        self.s = t;
    }

    #[inline(always)]
    pub fn total(&self) -> f64 {
        if self.s.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
}

// ---------------------------------------------------------------------------
// Small vector helpers.  Association order matches numpy's 3-element reduction
// (`pairwise_sum` falls through to a plain sequential loop below n = 8).
// ---------------------------------------------------------------------------

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

#[inline(always)]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `transforms.tangent_basis`: `(u_hat, du/dra, du/ddec)`, equatorial frame.
/// `du/dra` is the RAW partial (no `cos(dec)` factor).
#[inline]
pub fn tangent_basis(ra: f64, dec: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (sa, ca) = (ra.sin(), ra.cos());
    let (sd, cd) = (dec.sin(), dec.cos());
    (
        [cd * ca, cd * sa, sd],
        [-cd * sa, cd * ca, 0.0],
        [-sd * ca, -sd * sa, cd],
    )
}

/// `transforms.state_equ_to_ecl` for a single 3-vector.
#[inline(always)]
fn equ_to_ecl(x: [f64; 3], ce: f64, se: f64) -> [f64; 3] {
    // parity: x @ R.T with R = [[1,0,0],[0,ce,se],[0,-se,ce]]; the zero columns
    // contribute exact zeros so each output has a single addition.
    [x[0], ce * x[1] + se * x[2], -se * x[1] + ce * x[2]]
}

/// `transforms.phase_angle` for a single node.
#[inline(always)]
fn phase_angle(r: [f64; 3], r_obs: [f64; 3]) -> f64 {
    let to_sun = [-r[0], -r[1], -r[2]];
    let to_obs = [r_obs[0] - r[0], r_obs[1] - r[1], r_obs[2] - r[2]];
    let c = cross3(to_sun, to_obs);
    norm3(c).atan2(dot3(to_sun, to_obs))
}

/// `transforms.hg_phase_integral` for a single node.
#[inline(always)]
fn hg_phase_integral(alpha: f64, g: f64) -> f64 {
    if alpha.is_nan() {
        return f64::NAN;
    }
    let a = alpha.clamp(0.0, std::f64::consts::PI);
    let t = (0.5 * a).tan();
    let mut p1 = (-HG_A1 * t.powf(HG_B1)).exp();
    let mut p2 = (-HG_A2 * t.powf(HG_B2)).exp();
    if !p1.is_finite() {
        p1 = 0.0;
    }
    if !p2.is_finite() {
        p2 = 0.0;
    }
    (1.0 - g) * p1 + g * p2
}

/// Grid configuration, mirroring `ranging.build_grid`'s arguments.
#[derive(Clone, Copy, Debug)]
pub struct GridCfg {
    pub rho_min: f64,
    pub rho_max: f64,
    pub n_rho: usize,
    pub n_rhodot: usize,
}

/// Largest grid the kernel will accept, per axis and in total.
///
/// These are not accuracy limits — the production grid is (1200, 50), six
/// orders of magnitude below the cap.  They exist because the node count
/// reaches Rust as a Python integer, and without them
/// `n_rho = n_rhodot = 2^32` wraps the product to zero (a ~4e9-iteration loop
/// with the GIL released, i.e. an uninterruptible hang) while
/// `n_rho = 1e14` aborts the process in the allocator instead of raising.
pub const MAX_NODES_PER_AXIS: i64 = 100_000_000;
pub const MAX_NODES_TOTAL: i64 = 1_000_000_000;

impl GridCfg {
    /// Build a validated config from the raw (possibly hostile) Python values.
    ///
    /// `n_rho`/`n_rhodot` arrive signed so that a negative count raises
    /// `ValueError` with the same message the Python reference gives, rather
    /// than `OverflowError` from the unsigned conversion.
    pub fn new(
        rho_min: f64,
        rho_max: f64,
        n_rho: i64,
        n_rhodot: i64,
    ) -> Result<GridCfg, String> {
        if !(rho_min > 0.0) {
            return Err("rho_min must be > 0 (log-spaced rho grid)".into());
        }
        if !(rho_max > rho_min) {
            return Err("rho_max must be > rho_min".into());
        }
        if n_rho < 1 || n_rhodot < 1 {
            return Err("n_rho and n_rhodot must be >= 1".into());
        }
        if n_rho > MAX_NODES_PER_AXIS || n_rhodot > MAX_NODES_PER_AXIS {
            return Err(format!(
                "n_rho and n_rhodot must each be <= {MAX_NODES_PER_AXIS} \
                 (got {n_rho} and {n_rhodot})"
            ));
        }
        match n_rho.checked_mul(n_rhodot) {
            Some(p) if p <= MAX_NODES_TOTAL => {}
            _ => {
                return Err(format!(
                    "n_rho * n_rhodot must be <= {MAX_NODES_TOTAL} nodes \
                     (got {n_rho} * {n_rhodot})"
                ))
            }
        }
        Ok(GridCfg {
            rho_min,
            rho_max,
            n_rho: n_rho as usize,
            n_rhodot: n_rhodot as usize,
        })
    }

    /// `(rho_nodes, d_rho)` exactly as `ranging.build_grid` computes them:
    /// `np.geomspace` node placement with cell edges at the *geometric*
    /// midpoints, clamped to `[rho_min, rho_max]` at the ends, and widths taken
    /// in LINEAR rho so the cells tile `[rho_min, rho_max]` exactly.
    pub fn rho_nodes(&self) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = self.n_rho;
        if n == 1 {
            return Ok((
                vec![(self.rho_min * self.rho_max).sqrt()],
                vec![self.rho_max - self.rho_min],
            ));
        }
        // Fallible: MAX_NODES_PER_AXIS still allows an 800 MB request, and an
        // allocation failure must raise, not abort.
        let mut nodes: Vec<f64> = Vec::new();
        nodes
            .try_reserve_exact(n)
            .map_err(|_| format!("cannot allocate {n} rho nodes"))?;
        let mut edges: Vec<f64> = Vec::new();
        edges
            .try_reserve_exact(n + 1)
            .map_err(|_| format!("cannot allocate {n} rho cell edges"))?;
        // parity: np.geomspace == 10 ** linspace(log10 lo, log10 hi, n), with
        // the two endpoints overwritten by the exact bounds.
        let lo = self.rho_min.log10();
        let hi = self.rho_max.log10();
        let step = (hi - lo) / ((n - 1) as f64);
        for i in 0..n {
            nodes.push(10f64.powf((i as f64) * step + lo));
        }
        nodes[0] = self.rho_min;
        nodes[n - 1] = self.rho_max;

        edges.push(self.rho_min);
        for k in 1..n {
            edges.push((nodes[k - 1] * nodes[k]).sqrt());
        }
        edges.push(self.rho_max);
        let mut d_rho: Vec<f64> = Vec::new();
        d_rho
            .try_reserve_exact(n)
            .map_err(|_| format!("cannot allocate {n} rho cell widths"))?;
        for k in 0..n {
            d_rho.push(edges[k + 1] - edges[k]);
        }
        Ok((nodes, d_rho))
    }
}

/// One attributable variant (a sigma point in v1; the attributable itself in v0).
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub ra: f64,
    pub dec: f64,
    pub ra_rate: f64,
    pub dec_rate: f64,
    pub v_mag: f64,
}

/// Scalar outputs + counters.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub lambda_neo: f64,
    pub lambda_other: f64,
    pub n_nodes: usize,
    pub n_masked_nonfinite: usize,
    pub n_masked_det: usize,
    /// D-19 mechanism counter: of the `n_nodes - n_masked_nonfinite` nodes with
    /// finite elements AND finite implied H (the ones the density was actually
    /// evaluated at), how many have `H >= h_support_end`, where
    /// `h_support_end` is the pair's H support end —
    /// `max` over the two components of (`+inf` if that component clamps,
    /// else its last H edge).
    ///
    /// This exists so that `classifier._posterior` can tell the *mechanism* of
    /// a 0/0 outcome apart from its symptom, in `diagnostics=False` mode where
    /// no per-node array is available. Zero density over a non-empty region
    /// has several causes — the H support running out (the D-19 triage case),
    /// implied H below zero, or simply a region the model has no objects in —
    /// and only the first licenses the policy. Counted over `good` nodes only,
    /// because a node with non-finite H has no position relative to the
    /// support at all.
    pub n_h_at_or_above_support: usize,
    /// Total quadrature measure (sum of cell weights) over those same
    /// finite-H nodes, and the part of it at/above the support end.  The D-19
    /// trigger uses the RATIO of these two rather than the node counts: the
    /// measure is what the DESIGN §2.6(b) diagnostic reports and is
    /// independent of node placement, whereas a node-count fraction moves
    /// with the grid.
    pub w_h_finite: f64,
    pub w_h_at_or_above_support: f64,
    pub diag: Option<Diagnostics>,
}

/// Per-node arrays, allocated only when `diagnostics` is on.
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub rho: Vec<f64>,
    pub rho_dot: Vec<f64>,
    pub weight: Vec<f64>,
    pub integrand_neo: Vec<f64>,
    pub integrand_other: Vec<f64>,
    pub node_h: Vec<f64>,
    pub node_q: Vec<f64>,
}

impl Diagnostics {
    /// Reserve room for `n` nodes, or fail cleanly.
    ///
    /// Seven f64 arrays: at the cap that is 56 GB, so `try_reserve` is the
    /// difference between a `MemoryError` and SIGABRT.
    fn with_capacity(n: usize) -> Result<Self, String> {
        let mut d = Diagnostics::default();
        for (v, what) in [
            (&mut d.rho, "rho"),
            (&mut d.rho_dot, "rho_dot"),
            (&mut d.weight, "weight"),
            (&mut d.integrand_neo, "integrand_neo"),
            (&mut d.integrand_other, "integrand_other"),
            (&mut d.node_h, "node_H"),
            (&mut d.node_q, "node_q"),
        ] {
            v.try_reserve_exact(n)
                .map_err(|_| format!("cannot allocate {n} nodes for {what}"))?;
        }
        Ok(d)
    }
}

/// Run the fused kernel for one tracklet.
///
/// `variants` / `weights` carry the sigma-point ensemble (v0 passes `K = 1`
/// with weight `1.0`).  Each variant gets its own admissible region, its own
/// grid and its own pass; `Lambda_C = sum_k w_k Lambda_C^(k)`.
pub fn classify_one(
    model: &Model,
    variants: &[Variant],
    weights: &[f64],
    r_obs: [f64; 3],
    v_obs: [f64; 3],
    cfg: &GridCfg,
    g: f64,
    diagnostics: bool,
) -> Result<Outcome, String> {
    let (rho_nodes, d_rho) = cfg.rho_nodes()?;
    let (ce, se) = obliquity_cos_sin();
    let mu32 = mu_32();

    let mut lam_neo = Neumaier::default();
    let mut lam_other = Neumaier::default();
    let mut out = Outcome::default();
    // D-19: the pair's H support end.  `max` of the two components, because
    // "outside BOTH supports" needs the further of the two boundaries; a
    // clamped component has no upper boundary at all, so the pair's is +inf
    // and the policy can never fire on it (correctly: the clamp never runs
    // out of model above 25.5).
    let h_support_end = model
        .neo
        .h_support_end()
        .max(model.other.h_support_end());
    let mut diag = if diagnostics {
        Some(Diagnostics::with_capacity(cfg.n_rho * cfg.n_rhodot)?)
    } else {
        None
    };

    for (vi, var) in variants.iter().enumerate() {
        let wv = weights[vi];
        let (u, u_a, u_d) = tangent_basis(var.ra, var.dec);
        // parity: ranging.build_grid's `mu_vec`, reused as the motion term of
        // `topocentric_to_heliocentric` (identical expression, so bit-identical).
        let mvec = [
            var.ra_rate * u_a[0] + var.dec_rate * u_d[0],
            var.ra_rate * u_a[1] + var.dec_rate * u_d[1],
            var.ra_rate * u_a[2] + var.dec_rate * u_d[2],
        ];
        let cos_dec = var.dec.cos();
        let cos2_dec = cos_dec * cos_dec;

        let mut vlam_neo = Neumaier::default();
        let mut vlam_other = Neumaier::default();

        for (k, &rho) in rho_nodes.iter().enumerate() {
            // --- admissible rho_dot interval (ranging.admissible_rhodot_interval)
            let w_vec = [
                v_obs[0] + rho * mvec[0],
                v_obs[1] + rho * mvec[1],
                v_obs[2] + rho * mvec[2],
            ];
            let r_equ = [
                r_obs[0] + rho * u[0],
                r_obs[1] + rho * u[1],
                r_obs[2] + rho * u[2],
            ];
            let r_norm = norm3(r_equ);
            let b = dot3(w_vec, u);
            let w2 = dot3(w_vec, w_vec);
            // parity: disc = b*b - w2 + 2.0*MU_SUN/r  (left to right).
            let disc = b * b - w2 + 2.0 * MU_SUN / r_norm;
            if !(disc >= 0.0) {
                continue;
            }
            let sq = disc.sqrt();
            let lo = -b - sq;
            let hi = -b + sq;
            let width = hi - lo;
            if !(width.is_finite() && width > 0.0) {
                continue;
            }
            let d_rhodot = width / (cfg.n_rhodot as f64);
            let weight = d_rho[k] * d_rhodot;

            let rho2 = rho * rho;
            let rho4 = rho2 * rho2;

            // --- quantities that depend on rho ALONE, hoisted out of the
            // rho_dot loop.  The heliocentric POSITION (and hence |r|, the
            // phase angle and H(V, rho)) is independent of rho_dot, so all of
            // the photometry's transcendentals are evaluated once per rho
            // column instead of once per node.  This is an exact hoist: the
            // expressions are bit-for-bit the ones the per-node form would
            // have computed.
            let r_ecl = equ_to_ecl(r_equ, ce, se);
            let rn = norm3(r_ecl);
            // parity: classifier.classify takes r_helio and the phase angle in
            // the EQUATORIAL frame; `r_norm` above is the same expression.
            let r_helio = r_norm;
            let alpha = phase_angle(r_equ, r_obs);
            let phi = hg_phase_integral(alpha, g);
            let h_abs = var.v_mag - 5.0 * (rho * r_helio).log10() + 2.5 * phi.log10();
            let h_finite = h_abs.is_finite();
            let node_h = if h_finite { h_abs } else { f64::NAN };
            let r_over_rn = [r_ecl[0] / rn, r_ecl[1] / rn, r_ecl[2] / rn];

            for j in 0..cfg.n_rhodot {
                let rho_dot = lo + d_rhodot * ((j as f64) + 0.5);
                out.n_nodes += 1;

                // --- topocentric -> heliocentric (equ) --------------------
                // parity: (V_obs + rho_dot*u) + rho*(ra_rate*u_a + dec_rate*u_d)
                let v_equ = [
                    (v_obs[0] + rho_dot * u[0]) + rho * mvec[0],
                    (v_obs[1] + rho_dot * u[1]) + rho * mvec[1],
                    (v_obs[2] + rho_dot * u[2]) + rho * mvec[2],
                ];

                // --- equ -> ecl, then osculating (a, e, i) ----------------
                let v_ecl = equ_to_ecl(v_equ, ce, se);

                let v2 = dot3(v_ecl, v_ecl);
                let inv_a = 2.0 / rn - v2 / MU_SUN;
                let a = 1.0 / inv_a;
                let h_vec = cross3(r_ecl, v_ecl);
                let h = norm3(h_vec);
                let cvh = cross3(v_ecl, h_vec);
                let e_vec = [
                    cvh[0] / MU_SUN - r_over_rn[0],
                    cvh[1] / MU_SUN - r_over_rn[1],
                    cvh[2] / MU_SUN - r_over_rn[2],
                ];
                let e = norm3(e_vec);
                let inc = (h_vec[2] / h).clamp(-1.0, 1.0).acos();
                // parity: transforms.elements_from_state's `bound` mask; every
                // other element is finite whenever this holds, so
                // `all(isfinite(elements))` reduces to exactly this.
                let bound = a.is_finite() && a > 0.0 && e < 1.0 && e.is_finite() && h > 0.0;

                let good = bound && h_finite;
                if !good {
                    out.n_masked_nonfinite += 1;
                    if let Some(d) = diag.as_mut() {
                        d.rho.push(rho);
                        d.rho_dot.push(rho_dot);
                        d.weight.push(weight);
                        d.integrand_neo.push(0.0);
                        d.integrand_other.push(0.0);
                        d.node_h.push(node_h);
                        d.node_q.push(f64::NAN);
                    }
                    continue;
                }

                // D-19 mechanism counter.  `h_abs` is finite here (`good`), so
                // this is a plain comparison: NaN never reaches it, and
                // `h_support_end = +inf` (a clamped pair) makes it always
                // false, which is the intended answer.
                out.w_h_finite += weight;
                if h_abs >= h_support_end {
                    out.n_h_at_or_above_support += 1;
                    out.w_h_at_or_above_support += weight;
                }

                // --- Jacobian (closed form) + conditioning guard ----------
                let sin_i = inc.sin();
                let j_aei = rho4 * cos2_dec * (2.0 / (mu32 * a.sqrt() * e * sin_i));
                // WP2's conditioning guard, reproduced EXACTLY (not
                // approximated) but evaluated only where it can possibly fire
                // — ~16.5% of real ranging nodes; see guard.rs, including why
                // the prefilter needs the node's heliocentric distance `rn`.
                let det_bad = !j_aei.is_finite()
                    || (guard::needs_exact_guard(a, e, sin_i, rn)
                        && guard::det_is_masked(
                            r_ecl, h_vec, h, e_vec, a, e, inc,
                        ));
                if det_bad {
                    out.n_masked_det += 1;
                }
                let j_aei = if det_bad { 0.0 } else { j_aei };
                let q = a * (1.0 - e);
                let j_qei = j_aei * (1.0 - e);

                // --- densities -------------------------------------------
                let n_neo = model.neo.density(
                    if model.neo.qei { q } else { a },
                    e,
                    inc,
                    h_abs,
                );
                let n_other = model.other.density(
                    if model.other.qei { q } else { a },
                    e,
                    inc,
                    h_abs,
                );
                let int_neo = n_neo * if model.neo.qei { j_qei } else { j_aei };
                let int_other = n_other * if model.other.qei { j_qei } else { j_aei };

                vlam_neo.add(int_neo * weight);
                vlam_other.add(int_other * weight);

                if let Some(d) = diag.as_mut() {
                    d.rho.push(rho);
                    d.rho_dot.push(rho_dot);
                    d.weight.push(weight);
                    d.integrand_neo.push(int_neo);
                    d.integrand_other.push(int_other);
                    d.node_h.push(h_abs);
                    d.node_q.push(q);
                }
            }
        }
        lam_neo.add(wv * vlam_neo.total());
        lam_other.add(wv * vlam_other.total());
    }

    out.lambda_neo = lam_neo.total();
    out.lambda_other = lam_other.total();
    out.diag = diag;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_config_rejects_hostile_sizes() {
        assert!(GridCfg::new(0.01, 100.0, -5, 50).is_err());
        assert!(GridCfg::new(0.01, 100.0, 0, 50).is_err());
        assert!(GridCfg::new(0.01, 100.0, 1200, 0).is_err());
        assert!(GridCfg::new(0.0, 100.0, 1200, 50).is_err());
        assert!(GridCfg::new(0.01, 0.001, 1200, 50).is_err());
        // per-axis cap
        assert!(GridCfg::new(0.01, 100.0, 100_000_000_000_000, 1).is_err());
        // product cap, including the case that used to wrap to zero
        assert!(GridCfg::new(0.01, 100.0, 4_294_967_296, 4_294_967_296).is_err());
        assert!(GridCfg::new(0.01, 100.0, 1_000_000, 10_000).is_err());
        // and the production grid still passes
        assert!(GridCfg::new(0.01, 100.0, 1200, 50).is_ok());
    }

    #[test]
    fn geomspace_endpoints_and_tiling() {
        let cfg = GridCfg::new(0.01, 100.0, 1200, 50).unwrap();
        let (nodes, d_rho) = cfg.rho_nodes().unwrap();
        assert_eq!(nodes.len(), 1200);
        assert_eq!(nodes[0], 0.01);
        assert_eq!(nodes[1199], 100.0);
        let total: f64 = d_rho.iter().sum();
        assert!((total - (100.0 - 0.01)).abs() < 1e-9);
        assert!(nodes.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn single_rho_node_special_case() {
        let cfg = GridCfg::new(0.01, 100.0, 1, 3).unwrap();
        let (nodes, d_rho) = cfg.rho_nodes().unwrap();
        assert_eq!(nodes, vec![(0.01f64 * 100.0).sqrt()]);
        assert_eq!(d_rho, vec![100.0 - 0.01]);
    }

    #[test]
    fn neumaier_beats_naive() {
        let mut acc = Neumaier::default();
        acc.add(1.0);
        for _ in 0..10_000_000 {
            acc.add(1e-9);
        }
        assert!((acc.total() - (1.0 + 1e-2)).abs() < 1e-15);
    }

    #[test]
    fn neumaier_propagates_infinity() {
        let mut acc = Neumaier::default();
        acc.add(1.0);
        acc.add(f64::INFINITY);
        acc.add(1.0);
        assert!(acc.total().is_infinite());
    }

    #[test]
    fn tangent_basis_is_orthogonal() {
        let (u, ua, ud) = tangent_basis(1.234, -0.5);
        assert!((norm3(u) - 1.0).abs() < 1e-15);
        assert!((norm3(ud) - 1.0).abs() < 1e-15);
        assert!((norm3(ua) - (-0.5f64).cos()).abs() < 1e-15);
        assert!(dot3(u, ua).abs() < 1e-15);
        assert!(dot3(u, ud).abs() < 1e-15);
        assert!(dot3(ua, ud).abs() < 1e-15);
    }

    #[test]
    fn obliquity_maps_ecliptic_pole() {
        let (ce, se) = obliquity_cos_sin();
        let p = equ_to_ecl([0.0, -se, ce], ce, se);
        assert!(p[0].abs() < 1e-16 && p[1].abs() < 1e-16);
        assert!((p[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn phase_function_endpoints() {
        assert!((hg_phase_integral(0.0, 0.15) - 1.0).abs() < 1e-15);
        assert_eq!(hg_phase_integral(std::f64::consts::PI, 0.15), 0.0);
        assert!(hg_phase_integral(f64::NAN, 0.15).is_nan());
    }

}
