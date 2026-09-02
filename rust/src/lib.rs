//! `digest42._core` — the fused Rust classify kernel.
//!
//! The Python↔Rust boundary is crossed **once per tracklet** (or once per
//! batch), never per function: [`CoreModel`] is built once per process and
//! [`classify_core`] / [`classify_batch`] run the whole
//! grid → elements → photometry → density → Jacobian → accumulate pipeline
//! inside Rust.  The Python implementation in `digest42.classifier` remains the
//! reference backend; see `tests/test_backend_parity.py`.
//!
//! This is a public extension module, so everything crossing the boundary is
//! treated as hostile: the bin-shape cell count is a `checked_mul` fold, the
//! table read is bounds-checked, node counts arrive signed and are bounded per
//! axis and in product, and every large allocation uses `try_reserve`.  A
//! wrapped product or an unchecked read here is a segfault reachable from pure
//! Python, not merely a wrong answer.

mod consts;
mod guard;
mod kernel;
mod model;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use kernel::{classify_one, GridCfg, Outcome, Variant};
use model::{Histogram, Model};

pyo3::create_exception!(
    _core,
    UnsupportedModel,
    PyValueError,
    "The population components cannot be handed to the Rust kernel.\n\n\
     Subclasses ValueError, so existing handlers keep working; \
     `digest42.backend` maps it onto `ModelNotSupported` so that \
     `backend=\"auto\"` falls back to the Python reference instead of failing \
     where the reference would succeed."
);

/// A model-adaptation failure: recoverable by falling back to Python.
fn model_error(msg: String) -> PyErr {
    UnsupportedModel::new_err(msg)
}

#[allow(clippy::too_many_arguments)]
fn hist_from_py(
    table: PyReadonlyArray1<f64>,
    h_coeffs: Option<PyReadonlyArray1<f64>>,
    dims: (usize, usize, usize, usize),
    edges: &[Vec<f64>; 4],
    parameterization: &str,
    h_top_bin_infinite: bool,
    label: &str,
) -> PyResult<Histogram> {
    let dims = [dims.0, dims.1, dims.2, dims.3];
    // Checked: an unchecked product wraps (four 65 537-long edge arrays give
    // dims = [65536; 4], whose product is 0 mod 2^64), which would let an empty
    // table pass the length check and then be indexed far out of bounds.
    let n = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            model_error(format!(
                "bin shape {dims:?} overflows: the number of cells does not fit \
                 in a usize"
            ))
        })?;
    if n == 0 {
        return Err(model_error(format!("bin shape {dims:?} has an empty axis")));
    }
    let t = table.as_slice()?;
    if t.len() != n {
        return Err(model_error(format!(
            "density table for {label:?} has {} entries, expected {n} for shape {dims:?}",
            t.len()
        )));
    }
    for (k, e) in edges.iter().enumerate() {
        if e.len() != dims[k] + 1 {
            return Err(model_error(format!(
                "edge array {k} has length {} but dimension {k} is {}",
                e.len(),
                dims[k]
            )));
        }
    }
    let qei = match parameterization {
        "qei" => true,
        "aei" => false,
        other => {
            return Err(model_error(format!(
                "unknown density parameterization {other:?} (expected 'qei' or 'aei')"
            )))
        }
    };
    if !t.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return Err(model_error(format!(
            "density table for {label:?} must be finite and non-negative"
        )));
    }
    let coeffs = match h_coeffs {
        None => None,
        Some(arr) => {
            // The D-14 clamp and the smooth reconstruction are mutually
            // exclusive on the Python side too (`Digest2Histogram.__init__`
            // raises); repeated here so the kernel cannot be driven into a
            // state the reference cannot express.
            if h_top_bin_infinite {
                return Err(model_error(format!(
                    "component {label:?} supplies smooth-in-H coefficients and \
                     h_top_bin_infinite=True; the D-14 clamp extends a constant \
                     beyond the support and has no smooth analogue"
                )));
            }
            // 3 coefficients per (cell, H bin).  Checked: on a hostile `dims`
            // an unchecked 3*n could wrap and let a short array pass the length
            // test, after which `density` would index past the end.
            let n3 = n.checked_mul(3).ok_or_else(|| {
                model_error(format!(
                    "bin shape {dims:?} overflows: 3 coefficients per cell do \
                     not fit in a usize"
                ))
            })?;
            let c = arr.as_slice()?;
            if c.len() != n3 {
                return Err(model_error(format!(
                    "h_coeffs for {label:?} has {} entries, expected {n3} \
                     (3 per cell for shape {dims:?})",
                    c.len()
                )));
            }
            // Coefficients may legitimately be negative (b and c of a rising
            // then flattening segment), so only finiteness is checkable here.
            // Non-negativity is a property of the *polynomial on its own
            // interval*, guaranteed by PCHIP's monotone slopes upstream and
            // asserted by the Python unit tests, not re-derivable from the
            // triples in isolation.
            if !c.iter().all(|v| v.is_finite()) {
                return Err(model_error(format!(
                    "h_coeffs for {label:?} must be finite"
                )));
            }
            Some(c.to_vec())
        }
    };
    Ok(Histogram {
        dims,
        edges: edges.clone(),
        table: t.to_vec(),
        h_coeffs: coeffs,
        h_top_bin_infinite,
        qei,
        label: label.to_string(),
    })
}

/// Rust-owned copy of a `Digest2Histogram` NEO/other pair.
///
/// Built once per process from the flattened density tables and the four shared
/// bin-edge arrays.  Rust owns a *copy* of everything, so the numpy arrays
/// handed in may be freed or mutated afterwards without affecting the model.
///
/// That independence is also a hazard, and it is `digest42.backend`'s job to
/// manage: because the copy is frozen at construction, mutating the Python
/// component afterwards (flipping `h_top_bin_infinite`, editing a cell) leaves
/// this handle stale.  The glue therefore keys its handle cache on a content
/// fingerprint, not on object identity.
#[pyclass(module = "digest42._core", frozen)]
pub struct CoreModel {
    inner: Model,
}

#[pymethods]
impl CoreModel {
    /// Build the handle.
    ///
    /// Parameters mirror `Digest2Histogram`: `*_table` are the components'
    /// `_density_table` arrays flattened in C order, `*_edges` the shared
    /// `(q, e, i[rad], H)` bin edges, `*_parameterization` is `"qei"`/`"aei"`
    /// and `*_h_top_bin_infinite` selects the D-14 digest2-emulation clamp.
    ///
    /// The bin shape is taken from `dims` and the axis lengths from the edge
    /// arrays, so the D-19 finite H extension
    /// (`Digest2Histogram(h_extend_to=...)`) arrives here as nothing more than
    /// `dims.3 == 19` with a 20-long `h_edges` — Python builds the extended
    /// arrays before the handoff and this constructor needs no knowledge of the
    /// mode.  See `model.rs`'s header.
    ///
    /// `*_h_coeffs` are the optional smooth-in-H Horner coefficients
    /// (`Digest2Histogram.h_coeffs` flattened C-order, 3 per (cell, H bin)).
    /// When a component supplies them they *replace* its table inside
    /// `density`; the table is still required and still validated, so a handle
    /// always carries the piecewise-constant model that was reconstructed.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        neo_table, other_table, dims, q_edges, e_edges, i_edges, h_edges,
        neo_parameterization="qei", other_parameterization="qei",
        neo_h_top_bin_infinite=false, other_h_top_bin_infinite=false,
        neo_label="NEO", other_label="other",
        neo_h_coeffs=None, other_h_coeffs=None,
    ))]
    fn new(
        neo_table: PyReadonlyArray1<f64>,
        other_table: PyReadonlyArray1<f64>,
        dims: (usize, usize, usize, usize),
        q_edges: PyReadonlyArray1<f64>,
        e_edges: PyReadonlyArray1<f64>,
        i_edges: PyReadonlyArray1<f64>,
        h_edges: PyReadonlyArray1<f64>,
        neo_parameterization: &str,
        other_parameterization: &str,
        neo_h_top_bin_infinite: bool,
        other_h_top_bin_infinite: bool,
        neo_label: &str,
        other_label: &str,
        neo_h_coeffs: Option<PyReadonlyArray1<f64>>,
        other_h_coeffs: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let edges = [
            q_edges.as_slice()?.to_vec(),
            e_edges.as_slice()?.to_vec(),
            i_edges.as_slice()?.to_vec(),
            h_edges.as_slice()?.to_vec(),
        ];
        for (k, e) in edges.iter().enumerate() {
            if e.len() < 2 || !e.windows(2).all(|w| w[1] > w[0]) || !e.iter().all(|v| v.is_finite())
            {
                return Err(model_error(format!(
                    "edge array {k} must be finite and strictly increasing"
                )));
            }
        }
        let neo = hist_from_py(
            neo_table,
            neo_h_coeffs,
            dims,
            &edges,
            neo_parameterization,
            neo_h_top_bin_infinite,
            neo_label,
        )?;
        let other = hist_from_py(
            other_table,
            other_h_coeffs,
            dims,
            &edges,
            other_parameterization,
            other_h_top_bin_infinite,
            other_label,
        )?;
        Ok(CoreModel {
            inner: Model { neo, other },
        })
    }

    /// Evaluate both components' densities at explicit `(q|a, e, i, H)` tuples.
    ///
    /// Exposed for the bit-exactness tests against
    /// `Digest2Histogram.density`; the kernel itself calls the same lookup
    /// inline.  The first column of `elements3` is interpreted per component
    /// (`q` for a "qei" component, `a` for "aei"), exactly as in the kernel.
    fn density_at<'py>(
        &self,
        py: Python<'py>,
        elements3: PyReadonlyArray2<f64>,
        h: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let el = elements3.as_array();
        if el.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "elements3 must have trailing dimension 3",
            ));
        }
        let hh = h.as_slice()?;
        let n = el.shape()[0];
        if hh.len() != n {
            return Err(PyValueError::new_err(format!(
                "H has length {} but elements3 has {n} rows",
                hh.len()
            )));
        }
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for k in 0..n {
            let (x, e, i) = (el[[k, 0]], el[[k, 1]], el[[k, 2]]);
            a.push(self.inner.neo.density(x, e, i, hh[k]));
            b.push(self.inner.other.density(x, e, i, hh[k]));
        }
        Ok((a.to_pyarray(py), b.to_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        let n = &self.inner.neo;
        let mode = |h: &Histogram| if h.h_coeffs.is_some() { "pchip" } else { "none" };
        format!(
            "CoreModel(neo={:?}, other={:?}, dims={:?}, h_top_bin_infinite=({}, {}), \
             h_interpolation=({:?}, {:?}))",
            n.label,
            self.inner.other.label,
            n.dims,
            n.h_top_bin_infinite,
            self.inner.other.h_top_bin_infinite,
            mode(n),
            mode(&self.inner.other),
        )
    }
}

struct Inputs {
    variants: Vec<Variant>,
    weights: Vec<f64>,
    r_obs: [f64; 3],
    v_obs: [f64; 3],
}

fn vec3(a: &[f64], what: &str) -> PyResult<[f64; 3]> {
    if a.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must be a (3,) vector, got length {}",
            a.len()
        )));
    }
    Ok([a[0], a[1], a[2]])
}

fn check_variants(k: usize, n_cols: usize, weights: &[f64]) -> PyResult<()> {
    if n_cols != 5 {
        return Err(PyValueError::new_err(format!(
            "variants must have 5 columns (ra, dec, ra_rate, dec_rate, v_mag), got {n_cols}"
        )));
    }
    if k == 0 {
        return Err(PyValueError::new_err("variants must have at least one row"));
    }
    if weights.len() != k {
        return Err(PyValueError::new_err(format!(
            "weights has length {} but there are {k} variants",
            weights.len()
        )));
    }
    if !weights.iter().all(|w| w.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }
    Ok(())
}

/// Move one `Outcome` into a Python dict.
///
/// Takes the outcome **by value** and hands each node array to numpy with
/// `into_pyarray`, which adopts the `Vec`'s buffer instead of copying it.  With
/// a batch this is what keeps peak memory at one copy: the driver drains its
/// `Vec<Outcome>` as it goes, so a converted tracklet's ~4.6 MB of node arrays
/// is released (into numpy) before the next one is touched.
fn outcome_to_dict<'py>(py: Python<'py>, o: Outcome) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("lambda_neo", o.lambda_neo)?;
    d.set_item("lambda_other", o.lambda_other)?;
    d.set_item("n_nodes", o.n_nodes)?;
    d.set_item("n_masked_nonfinite", o.n_masked_nonfinite)?;
    d.set_item("n_masked_det", o.n_masked_det)?;
    // D-19 mechanism evidence; see `kernel::Outcome::n_h_at_or_above_support`.
    // Always emitted, including in `diagnostics=False` mode — that is the
    // whole point of it being a counter rather than a per-node array.
    d.set_item("n_h_at_or_above_support", o.n_h_at_or_above_support)?;
    d.set_item("w_h_finite", o.w_h_finite)?;
    d.set_item("w_h_at_or_above_support", o.w_h_at_or_above_support)?;
    match o.diag {
        None => {
            for k in [
                "rho",
                "rho_dot",
                "weight",
                "integrand_neo",
                "integrand_other",
                "node_H",
                "node_q",
            ] {
                d.set_item(k, py.None())?;
            }
        }
        Some(g) => {
            d.set_item("rho", g.rho.into_pyarray(py))?;
            d.set_item("rho_dot", g.rho_dot.into_pyarray(py))?;
            d.set_item("weight", g.weight.into_pyarray(py))?;
            d.set_item("integrand_neo", g.integrand_neo.into_pyarray(py))?;
            d.set_item("integrand_other", g.integrand_other.into_pyarray(py))?;
            d.set_item("node_H", g.node_h.into_pyarray(py))?;
            d.set_item("node_q", g.node_q.into_pyarray(py))?;
        }
    }
    Ok(d)
}

/// Classify one tracklet.
///
/// Returns a dict with `lambda_neo`, `lambda_other`, `n_nodes`,
/// `n_masked_nonfinite`, `n_masked_det`, `n_h_at_or_above_support`,
/// `w_h_finite`, `w_h_at_or_above_support`, and — when `diagnostics` is true —
/// the per-node arrays `rho`, `rho_dot`, `weight`, `integrand_neo`,
/// `integrand_other`, `node_H`, `node_q` (all `None` otherwise, with no
/// per-node allocation performed at all).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    model, variants, weights, R_obs, V_obs, rho_nodes, d_rho, n_rhodot,
    G, diagnostics=false,
))]
#[allow(non_snake_case)]
fn classify_core<'py>(
    py: Python<'py>,
    model: &CoreModel,
    variants: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    R_obs: PyReadonlyArray1<f64>,
    V_obs: PyReadonlyArray1<f64>,
    rho_nodes: PyReadonlyArray1<f64>,
    d_rho: PyReadonlyArray1<f64>,
    n_rhodot: i64,
    G: f64,
    diagnostics: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let rho_nodes = rho_nodes.as_slice()?;
    let d_rho = d_rho.as_slice()?;
    let cfg = GridCfg::new(rho_nodes, d_rho, n_rhodot)
        .map_err(PyValueError::new_err)?;
    // Owned copies so the node loop can run with the GIL released.  ~9.6 kB at
    // n_rho = 1200; measured cost below the benchmark's noise floor.
    let rho_nodes = rho_nodes.to_vec();
    let d_rho = d_rho.to_vec();

    let v = variants.as_array();
    let (k, ncol) = (v.shape()[0], v.shape()[1]);
    let w = weights.as_slice()?;
    check_variants(k, ncol, w)?;
    if diagnostics && k > 1 {
        return Err(PyValueError::new_err(
            "diagnostics=True is only defined for a single attributable variant \
             (K = 1): each sigma point has its own admissible region and grid, so \
             the per-node arrays would not correspond to one RangingGrid.",
        ));
    }
    let inputs = Inputs {
        variants: (0..k)
            .map(|r| Variant {
                ra: v[[r, 0]],
                dec: v[[r, 1]],
                ra_rate: v[[r, 2]],
                dec_rate: v[[r, 3]],
                v_mag: v[[r, 4]],
            })
            .collect(),
        weights: w.to_vec(),
        r_obs: vec3(R_obs.as_slice()?, "R_obs")?,
        v_obs: vec3(V_obs.as_slice()?, "V_obs")?,
    };

    let out = py
        .detach(|| {
            classify_one(
                &model.inner,
                &inputs.variants,
                &inputs.weights,
                inputs.r_obs,
                inputs.v_obs,
                &cfg,
                &rho_nodes,
                &d_rho,
                G,
                diagnostics,
            )
        })
        .map_err(PyMemoryError::new_err)?;
    outcome_to_dict(py, out)
}

/// Classify a batch of tracklets, in parallel across tracklets with rayon.
///
/// `variants` is `(T, K, 5)`, `weights` `(T, K)`, `R_obs`/`V_obs` `(T, 3)`.
/// The GIL is released for the whole batch.  `n_threads = None` uses rayon's
/// global pool (one thread per core); `1` runs serially in the calling thread.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    model, variants, weights, R_obs, V_obs, rho_nodes, d_rho, n_rhodot,
    G, diagnostics=false, n_threads=None,
))]
#[allow(non_snake_case)]
fn classify_batch<'py>(
    py: Python<'py>,
    model: &CoreModel,
    variants: PyReadonlyArray3<f64>,
    weights: PyReadonlyArray2<f64>,
    R_obs: PyReadonlyArray2<f64>,
    V_obs: PyReadonlyArray2<f64>,
    rho_nodes: PyReadonlyArray1<f64>,
    d_rho: PyReadonlyArray1<f64>,
    n_rhodot: i64,
    G: f64,
    diagnostics: bool,
    n_threads: Option<usize>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let rho_nodes = rho_nodes.as_slice()?;
    let d_rho = d_rho.as_slice()?;
    let cfg = GridCfg::new(rho_nodes, d_rho, n_rhodot)
        .map_err(PyValueError::new_err)?;
    // One copy for the whole batch, so the rayon workers can run GIL-free.
    let rho_nodes = rho_nodes.to_vec();
    let d_rho = d_rho.to_vec();

    let v = variants.as_array();
    let wts = weights.as_array();
    let r = R_obs.as_array();
    let vo = V_obs.as_array();
    let (t, k, ncol) = (v.shape()[0], v.shape()[1], v.shape()[2]);
    if wts.shape() != [t, k] {
        return Err(PyValueError::new_err(format!(
            "weights must have shape ({t}, {k}), got {:?}",
            wts.shape()
        )));
    }
    if r.shape() != [t, 3] || vo.shape() != [t, 3] {
        return Err(PyValueError::new_err(format!(
            "R_obs and V_obs must have shape ({t}, 3), got {:?} and {:?}",
            r.shape(),
            vo.shape()
        )));
    }
    if diagnostics && k > 1 {
        return Err(PyValueError::new_err(
            "diagnostics=True is only defined for a single attributable variant (K = 1)",
        ));
    }

    let mut inputs: Vec<Inputs> = Vec::with_capacity(t);
    for i in 0..t {
        let w: Vec<f64> = (0..k).map(|j| wts[[i, j]]).collect();
        check_variants(k, ncol, &w)?;
        inputs.push(Inputs {
            variants: (0..k)
                .map(|j| Variant {
                    ra: v[[i, j, 0]],
                    dec: v[[i, j, 1]],
                    ra_rate: v[[i, j, 2]],
                    dec_rate: v[[i, j, 3]],
                    v_mag: v[[i, j, 4]],
                })
                .collect(),
            weights: w,
            r_obs: [r[[i, 0]], r[[i, 1]], r[[i, 2]]],
            v_obs: [vo[[i, 0]], vo[[i, 1]], vo[[i, 2]]],
        });
    }

    let outs: Result<Vec<Outcome>, String> = py.detach(|| {
        let run = || -> Result<Vec<Outcome>, String> {
            inputs
                .par_iter()
                .map(|inp| {
                    classify_one(
                        &model.inner,
                        &inp.variants,
                        &inp.weights,
                        inp.r_obs,
                        inp.v_obs,
                        &cfg,
                        &rho_nodes,
                        &d_rho,
                        G,
                        diagnostics,
                    )
                })
                .collect()
        };
        match n_threads {
            None => run(),
            Some(n) => rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| e.to_string())
                .and_then(|pool| pool.install(run)),
        }
    });
    let outs = outs.map_err(PyMemoryError::new_err)?;
    // Drain rather than borrow: each Outcome's node arrays are moved into numpy
    // and freed from the Rust side immediately, so the batch never holds two
    // copies of the diagnostics at once.
    let mut dicts = Vec::with_capacity(outs.len());
    for o in outs.into_iter() {
        dicts.push(outcome_to_dict(py, o)?);
    }
    Ok(dicts)
}

/// Direct access to the closed-form Jacobian factor, for the parity tests.
///
/// Returns `|det d(a,e,i,Om,om,M)/d(A, rho, rho_dot)|` (or the `(q,e,i)` form
/// when `parameterization="qei"`) given the node geometry and the osculating
/// `(a, e, i)`.
#[pyfunction]
#[pyo3(signature = (rho, dec, a, e, inc, parameterization="aei"))]
fn jacobian_closed_form(
    rho: f64,
    dec: f64,
    a: f64,
    e: f64,
    inc: f64,
    parameterization: &str,
) -> PyResult<f64> {
    let cd = dec.cos();
    let rho2 = rho * rho;
    let j = rho2 * rho2 * (cd * cd) * (2.0 / (consts::mu_32() * a.sqrt() * e * inc.sin()));
    match parameterization {
        "aei" => Ok(j),
        "qei" => Ok(j * (1.0 - e)),
        other => Err(PyValueError::new_err(format!(
            "unknown parameterization {other:?}"
        ))),
    }
}

/// Cheap upper bound on `cond_inf(d(state)/d(elements))` (the prefilter).
#[pyfunction]
fn cond_upper_bound(a: f64, e: f64, sin_i: f64, r: f64) -> f64 {
    guard::cond_upper_bound(a, e, sin_i, r)
}

/// Whether a node takes the exact 6x6 conditioning path.
#[pyfunction]
fn needs_exact_guard(a: f64, e: f64, sin_i: f64, r: f64) -> bool {
    guard::needs_exact_guard(a, e, sin_i, r)
}

/// WP2's exact infinity-norm condition estimate of `d(state)/d(elements)`,
/// for the parity tests against `transforms.jacobian_elements_wrt_state`.
/// Returns `None` where the matrix is singular or non-finite.
#[pyfunction]
#[pyo3(signature = (a, e, inc, Omega, omega, ecc_anom))]
#[allow(non_snake_case)]
fn cond_inf_elements(
    a: f64,
    e: f64,
    inc: f64,
    Omega: f64,
    omega: f64,
    ecc_anom: f64,
) -> Option<f64> {
    guard::cond_inf_6x6(guard::jacobian_state_wrt_elements(
        a, e, inc, Omega, omega, ecc_anom,
    ))
}

/// Bin lookup exposed for bit-exactness tests against `population._bin_index`.
#[pyfunction]
fn bin_index<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let e = edges.as_slice()?;
    let out: Vec<i64> = x
        .as_slice()?
        .iter()
        .map(|&v| model::bin_index(e, v).map(|i| i as i64).unwrap_or(-1))
        .collect();
    Ok(out.to_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The crate carries no version (publish = false); the package version is
    // the git tag via setuptools-scm.  Report nothing rather than "0.0.0".
    m.add("__version__", "(digest42 package version; see digest42.__version__)")?;
    m.add("COND_MAX", consts::COND_MAX)?;
    m.add("UnsupportedModel", m.py().get_type::<UnsupportedModel>())?;
    m.add_class::<CoreModel>()?;
    m.add_function(wrap_pyfunction!(classify_core, m)?)?;
    m.add_function(wrap_pyfunction!(classify_batch, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_closed_form, m)?)?;
    m.add_function(wrap_pyfunction!(cond_upper_bound, m)?)?;
    m.add_function(wrap_pyfunction!(needs_exact_guard, m)?)?;
    m.add_function(wrap_pyfunction!(cond_inf_elements, m)?)?;
    m.add_function(wrap_pyfunction!(bin_index, m)?)?;
    Ok(())
}
