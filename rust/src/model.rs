//! The population-model handle: flattened binned densities owned by Rust.
//!
//! Port of `digest42.population._bin_index`, `Digest2Histogram._h_bin` and
//! `Digest2Histogram.density`.  The lookup semantics must be *bit-identical* to
//! the Python ones:
//!
//! * half-open `[lo, hi)` bins, matching digest2's `x >= part[b]` rule;
//! * anything below the first edge or at/above the last edge is out of support
//!   and evaluates to exactly `0.0` (D-4 hard truncation) — this also covers
//!   negative q/e/i, since every edge array starts at `0.0`;
//! * NaN is out of support;
//! * with `h_top_bin_infinite` (D-14 digest2-emulation clamp) `H >= h_edges[-1]`
//!   maps to the top H bin.  `+inf` clamps (it compares `>=`), `NaN` does not.
//!   `H < 0` is deliberately left out of support in both modes.
//!
//! The smooth-in-H mode (`Digest2Histogram(h_interpolation="pchip")`) is carried
//! as an optional per-interval coefficient array instead of the flat density
//! table.  The coefficients are **not** recomputed here: Python precomputes them
//! once (`Digest2Histogram.h_coeffs`, already divided by
//! `dq de di (2 pi)^3`) and hands them over, and both sides evaluate the same
//! Horner form `a + s (b + s c)` with `s = H - h_edges[ih]`.  Sharing the
//! coefficients and fixing the grouping is what makes the two backends
//! *bit-identical* rather than merely close: neither IEEE-754 double arithmetic
//! nor numpy's ufunc-per-operation evaluation leaves any freedom in
//! `a + s * (b + s * c)`, and LLVM does not fuse it into FMAs without
//! fast-math (which this crate does not enable).

/// One binned population component (a flattened `Digest2Histogram`).
pub struct Histogram {
    /// (n_q, n_e, n_i, n_h) — the C-order shape of `table`.
    pub dims: [usize; 4],
    /// Bin edges: q [AU], e [-], i [rad], H [mag]; `edges[k].len() == dims[k]+1`.
    pub edges: [Vec<f64>; 4],
    /// `Digest2Histogram._density_table` flattened C-order.
    pub table: Vec<f64>,
    /// `Digest2Histogram.h_coeffs` flattened C-order (`3` per (cell, H bin)),
    /// or `None` for the piecewise-constant mode.  When present it *replaces*
    /// `table` in `density` — `table` is then carried for provenance only.
    pub h_coeffs: Option<Vec<f64>>,
    /// D-14 clamp mode.
    pub h_top_bin_infinite: bool,
    /// `true` for parameterization "qei" (adds the `(1 - e)` Jacobian factor),
    /// `false` for "aei".
    pub qei: bool,
    /// Provenance only.
    pub label: String,
}

/// Half-open `[lo, hi)` bin lookup, matching
/// `np.searchsorted(edges, x, side="right") - 1` plus the in-range test.
#[inline(always)]
pub fn bin_index(edges: &[f64], x: f64) -> Option<usize> {
    if x.is_nan() {
        // numpy sorts NaN to the end, so searchsorted returns len(edges) and the
        // index falls out of range.
        return None;
    }
    // Number of edges <= x  ==  np.searchsorted(edges, x, side="right").
    let c = edges.partition_point(|&e| e <= x);
    if c == 0 || c == edges.len() {
        None
    } else {
        Some(c - 1)
    }
}

impl Histogram {
    /// `Digest2Histogram._h_bin`, returning `None` when out of support.
    #[inline(always)]
    pub fn h_bin(&self, h: f64) -> Option<usize> {
        if self.h_top_bin_infinite {
            let top = *self.edges[3].last().unwrap();
            // NaN >= top is false, +inf >= top is true — exactly numpy's rule.
            if h >= top {
                return Some(self.dims[3] - 1);
            }
        }
        bin_index(&self.edges[3], h)
    }

    /// Number density; exactly `0.0` outside the support.
    ///
    /// Piecewise constant in q/e/i always; in H either the flat table lookup or
    /// one Horner evaluation of the precomputed monotone-cubic derivative.  The
    /// four bin lookups are shared, so the two modes differ by exactly one
    /// branch and three loads — the smooth mode costs one extra multiply-add
    /// pair per node (measured wall-clock overhead in
    /// `validation/output/h_interpolation/REPORT.md`).
    #[inline(always)]
    pub fn density(&self, q: f64, e: f64, inc: f64, h: f64) -> f64 {
        let iq = match bin_index(&self.edges[0], q) {
            Some(v) => v,
            None => return 0.0,
        };
        let ie = match bin_index(&self.edges[1], e) {
            Some(v) => v,
            None => return 0.0,
        };
        let ii = match bin_index(&self.edges[2], inc) {
            Some(v) => v,
            None => return 0.0,
        };
        let ih = match self.h_bin(h) {
            Some(v) => v,
            None => return 0.0,
        };
        let [_nq, ne, ni, nh] = self.dims;
        // The index arithmetic cannot overflow: `CoreModel::new` verifies with
        // `checked_mul` that nq*ne*ni*nh fits in a usize, and every partial
        // product here is bounded by that total.
        //
        // The read is nonetheless bounds-checked on purpose.  Every index IS
        // below its dimension by construction of `bin_index`, but `dims`
        // arrives from Python and an unchecked read turns any future slip in
        // the shape validation into an out-of-bounds read reachable from pure
        // Python.  Measured cost: below the benchmark's noise floor (the
        // unchecked build came out 1% *slower*), because the four bin lookups
        // dominate and the branch is perfectly predicted.
        let flat = ((iq * ne + ie) * ni + ii) * nh + ih;
        match &self.h_coeffs {
            None => self.table.get(flat).copied().unwrap_or(0.0),
            Some(co) => {
                // `flat * 3` cannot overflow: `CoreModel::new` verifies with
                // `checked_mul` that 3 * nq*ne*ni*nh fits in a usize whenever
                // coefficients are supplied, and `flat < nq*ne*ni*nh`.  The
                // slice read is bounds-checked for the same reason as above.
                match co.get(flat * 3..flat * 3 + 3) {
                    None => 0.0,
                    Some(c) => {
                        // `ih` came from `bin_index`/`h_bin`, so
                        // `edges[3][ih] <= h` and s >= 0; the bins are
                        // half-open, so s stays below the interval width.
                        // Exactly the Python expression, in exactly this
                        // grouping — see the module header on bit-identity.
                        let s = h - self.edges[3][ih];
                        c[0] + s * (c[1] + s * c[2])
                    }
                }
            }
        }
    }
}

/// The NEO/other component pair, built once per process.
pub struct Model {
    pub neo: Histogram,
    pub other: Histogram,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges() -> Vec<f64> {
        vec![0.0, 1.0, 2.0, 3.0]
    }

    #[test]
    fn bin_edges_are_half_open() {
        let e = edges();
        assert_eq!(bin_index(&e, -1e-300), None);
        assert_eq!(bin_index(&e, -0.0), Some(0)); // -0.0 <= 0.0 -> in bin 0
        assert_eq!(bin_index(&e, 0.0), Some(0));
        assert_eq!(bin_index(&e, 0.999), Some(0));
        assert_eq!(bin_index(&e, 1.0), Some(1));
        assert_eq!(bin_index(&e, 2.999), Some(2));
        assert_eq!(bin_index(&e, 3.0), None); // upper edge excluded
        assert_eq!(bin_index(&e, 3.1), None);
        assert_eq!(bin_index(&e, f64::INFINITY), None);
        assert_eq!(bin_index(&e, f64::NEG_INFINITY), None);
        assert_eq!(bin_index(&e, f64::NAN), None);
    }

    #[test]
    fn h_clamp_mode() {
        let mut hist = Histogram {
            dims: [1, 1, 1, 3],
            edges: [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], edges()],
            table: vec![10.0, 20.0, 30.0],
            h_coeffs: None,
            h_top_bin_infinite: false,
            qei: true,
            label: "t".into(),
        };
        assert_eq!(hist.h_bin(3.0), None);
        assert_eq!(hist.h_bin(f64::INFINITY), None);
        hist.h_top_bin_infinite = true;
        assert_eq!(hist.h_bin(3.0), Some(2));
        assert_eq!(hist.h_bin(1e9), Some(2));
        assert_eq!(hist.h_bin(f64::INFINITY), Some(2));
        assert_eq!(hist.h_bin(f64::NAN), None);
        assert_eq!(hist.h_bin(-0.5), None); // H < 0 stays out of support
        assert_eq!(hist.density(0.5, 0.5, 0.5, 1e9), 30.0);
        assert_eq!(hist.density(-0.5, 0.5, 0.5, 1.5), 0.0);
    }

    #[test]
    fn density_read_is_bounds_checked() {
        // A table shorter than `dims` claims must yield 0.0, not a wild read.
        // `dims` comes from Python; CoreModel::new rejects this, but the lookup
        // must not depend on that being the only line of defence.
        let hist = Histogram {
            dims: [4, 4, 4, 4],
            edges: [
                vec![0.0, 1.0, 2.0, 3.0, 4.0],
                vec![0.0, 1.0, 2.0, 3.0, 4.0],
                vec![0.0, 1.0, 2.0, 3.0, 4.0],
                vec![0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            table: vec![1.0, 2.0],
            h_coeffs: None,
            h_top_bin_infinite: false,
            qei: true,
            label: "short".into(),
        };
        assert_eq!(hist.density(0.5, 0.5, 0.5, 0.5), 1.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 1.5), 2.0);
        assert_eq!(hist.density(3.5, 3.5, 3.5, 3.5), 0.0);
        assert_eq!(hist.density(2.5, 1.5, 3.5, 0.5), 0.0);
    }

    /// The smooth-in-H branch: one Horner per lookup, D-4 truncation intact,
    /// and a short coefficient array must degrade to 0.0 rather than read wild.
    #[test]
    fn pchip_branch_evaluates_the_horner_form() {
        // Three H bins [0,1), [1,2), [2,3); coefficients chosen by hand.
        let co = vec![
            1.0, 2.0, 3.0, // bin 0: 1 + s(2 + 3s)
            4.0, 0.0, 0.0, // bin 1: constant 4
            0.0, 6.0, -6.0, // bin 2: s(6 - 6s)
        ];
        let hist = Histogram {
            dims: [1, 1, 1, 3],
            edges: [
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0, 2.0, 3.0],
            ],
            table: vec![10.0, 20.0, 30.0],
            h_coeffs: Some(co),
            h_top_bin_infinite: false,
            qei: true,
            label: "pchip".into(),
        };
        // s = h - edges[3][ih], so the SECOND bin evaluates at s = h - 1.
        assert_eq!(hist.density(0.5, 0.5, 0.5, 0.0), 1.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 0.5), 1.0 + 0.5 * (2.0 + 3.0 * 0.5));
        assert_eq!(hist.density(0.5, 0.5, 0.5, 1.0), 4.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 1.75), 4.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 2.5), 0.5 * (6.0 - 6.0 * 0.5));
        // D-4: outside the support, still exactly zero, table or no table.
        assert_eq!(hist.density(0.5, 0.5, 0.5, 3.0), 0.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, -0.5), 0.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, f64::NAN), 0.0);
        assert_eq!(hist.density(1.5, 0.5, 0.5, 0.5), 0.0);
    }

    #[test]
    fn pchip_coeff_read_is_bounds_checked() {
        let hist = Histogram {
            dims: [1, 1, 1, 3],
            edges: [
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0, 2.0, 3.0],
            ],
            table: vec![10.0, 20.0, 30.0],
            // Only the first bin's triple is present; CoreModel::new rejects
            // this, but the lookup must not depend on that.
            h_coeffs: Some(vec![1.0, 0.0, 0.0]),
            h_top_bin_infinite: false,
            qei: true,
            label: "short-pchip".into(),
        };
        assert_eq!(hist.density(0.5, 0.5, 0.5, 0.5), 1.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 1.5), 0.0);
        assert_eq!(hist.density(0.5, 0.5, 0.5, 2.5), 0.0);
    }
}
