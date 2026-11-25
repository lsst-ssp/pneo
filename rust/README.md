# pneo Rust core kernel (`pneo._core`)

The fused classify kernel behind `backend="rust"`. It implements the WP5
marginal-likelihood integral (DESIGN §2.3–2.5) as a single pass over the
quadrature nodes, crossing the Python↔Rust boundary **once per tracklet or
once per batch** — never per function. The numpy pipeline in
`pneo.classifier` stays the reference implementation; the two are
cross-validated in `tests/test_backend_parity.py` and
`validation/backend_parity.py`.

## Build

```sh
rust/build.sh            # release build, installs src/pneo/_core.abi3.so
rust/build.sh --debug    # debug build (much slower kernel)
cargo test --release --manifest-path rust/Cargo.toml     # rust unit tests
```

**Layout choice.** The crate is a plain cargo/pyo3 `cdylib` and `build.sh`
copies `target/release/libpneo_core.so` to
`src/pneo/_core.abi3.so`. The repository's Python package is installed
editable from `src/`, so that path *is* the import location and a rebuild is
picked up with no reinstall — this is the `maturin develop` equivalent for this
repo. A real `maturin develop` would have to own the whole `pneo` package
and would fight the existing editable install, which is why the build is a
`cargo build` plus a copy instead. The install is an atomic rename rather than
an overwrite, so rebuilding while a long run has the old library mapped cannot
corrupt it. The extension is built against the stable ABI (`abi3-py311`), so
one `.so` works for CPython ≥ 3.11 and no rebuild is needed across interpreter
minor versions.

maturin is deliberately not in the loop. Driven from this crate alone it names
the wheel's module `pneo_core` rather than `pneo._core` (it warns about
exactly that), and making it emit `pneo/_core` would have the wheel claim
the whole `pneo` package and collide with the editable install. Producing a
single `pip`-installable wheel that carries both the Python package and the
extension is a packaging change at the repo root (`pyproject.toml` moving to
the maturin build backend with `python-source = "src"`), which this work
deliberately does not touch.

Dependencies are exactly the three the design allows: `pyo3`, `numpy`
(rust-numpy) and `rayon`.

## What is in here

| file | contents |
| --- | --- |
| `src/consts.rs` | copies of `pneo.constants` / `transforms` constants (`K_GAUSS`, obliquity, H,G coefficients, `_COND_MAX`) |
| `src/model.rs` | `Histogram`: the flattened `(q, e, i, H)` density table and the bin lookup, bit-identical to `population._bin_index` / `Digest2Histogram.density` including the D-14 clamp |
| `src/kernel.rs` | the fused node loop: geometry → elements → photometry → density → Jacobian → accumulation; ρ̇ node generation; grid validation |
| `src/guard.rs` | exact port of WP2's conditioning guard, plus the reachable-manifold prefilter that decides when to run it |
| `src/lib.rs` | PyO3 surface: `CoreModel`, `classify_core`, `classify_batch`, and the test hooks |

### The Jacobian

Instead of building and inverting `d(elements)/d(state)` per node, the composed
determinant is evaluated in closed form:

```
|det d(a,e,i,Ω,ω,M)/d(A, ρ, ρ̇)| = ρ⁴ cos²δ · 2 / (μ^{3/2} √a · e · sin i)
```

times `(1 − e)` for a `"qei"` component. The first factor is exact: the
observable block is block-triangular in the column order
`(α, δ, ρ | α̇, δ̇, ρ̇)` with two copies of `[ρ u_α, ρ u_δ, u]`, whose
determinant is `ρ² cos δ`. The second is the Delaunay identity
(`d(L,G,H,l,g,h)/d(r,v) = 1` with `L = √(μa)`, `G = L√(1−e²)`, `H = G cos i`).
This removes the 6×6 build/invert *and* every Kepler solve: `(a, e, i)` come
straight from the state.

### The conditioning guard

`transforms.jacobian_elements_wrt_state` refuses to report a Jacobian when its
own 6×6 inversion is too ill-conditioned (`cond_inf > 1e12`); `classify` zeroes
those nodes and counts them in `n_masked_det`. The closed form has no such
failure mode, so `guard.rs` reproduces the *decision* exactly — it builds the
same analytic forward 6×6 and estimates the same infinity-norm condition
number. No Kepler solver is needed there either: `E` comes from the state via
the true anomaly rather than from `M`.

Running that at every node costs 31 ms/tracklet (measured), so it runs behind a
prefilter. **The prefilter is an upper bound on the reachable manifold only,
and that premise is load-bearing.** There is no cheap global bound: the
degeneracy mechanisms compound, a dense sweep of abstract
`(a, e, i, Ω, ω, M)` reaches `cond_inf = 5.8e62`, and every product- or max-form
tight enough to be a useful filter is violated there by 20+ decades (measured;
the best global product form fires on 95 % of the sweep). What makes a cheap
bound possible is that the kernel never evaluates abstract elements — it
evaluates the state implied by `(A, ρ, ρ̇)` with `ρ ≤ ρ_max`, so the
heliocentric distance `r` is *known at the node*. The bound is therefore
written in terms of `r`:

```
cond_upper_bound = max( 1e10·w^{5/2},  1e8/e,  1e8/sin i ),   w = a/(r·√(1−e²))
```

`w` is the quantity that actually appears in the entries of
`d(state)/d(elements)` (`dE/dM = a/r`, `dY/dM = a²b/r`, `dYd/de ~ s/(br)`). It
is O(1) for an ordinary orbit, grows like `(1−e)^{−3/2}` at perihelion passage
of an eccentric orbit, and like `a/r` for an orbit far larger than the sampled
range — so it subsumes the `a → ∞` mechanism, because `r ≤ ρ_max + |R_obs|`
forces large `a` to mean large `a/r`. The other two terms cover the
degeneracies at which `w` stays O(1) while the element frame itself becomes
undefined (`e → 0`, `sin i → 0`).

| term | measured worst case | coefficient | margin |
| --- | --- | --- | --- |
| `w^{5/2}` | `cond ≤ 8.41e7·w^{5/2}` | `1e10` | **119×** |
| `1/e` | `cond ≤ 2.8e5/e` | `1e8` | 360× |
| `1/sin i` | `cond ≤ 2.5e5/sin i` | `1e8` | 400× |

The `w` row is measured on **58 201 040 nodes sampled off the reachable
manifold** — observer on a ~1 AU orbit, solar elongation down to 5°, on-sky
rates to ~100°/day, `ρ` over the full range, `ρ̇` across the admissible
interval — with `cond_inf` evaluated *exactly* at every one. That sample does
reach perihelion passage (`r/q` down to 1.000, `1−e` down to 1.3e-8) and
`cond_inf` up to 5.06e25, with **zero false negatives**. Two in-crate tests
enforce the invariant: one walks the reachable manifold node by node, the other
sweeps `(a, e, i)` at `E ∈ {0, π}` restricted to reachable `r`.

The prefilter fires on ~16.5 % of real gate_v0 nodes, costing 8.2 → 12.0
ms/tracklet.

> **Superseded first attempt, recorded deliberately.** The original bound used
> an `a^{3/2}/(1−e)` perihelion term. It was wrong twice over — the true
> perihelion scaling is far steeper than `1/(1−e)`, and the `a^{3/2}` factor
> cancels the blow-up exactly where the conditioning is worst — and it needed
> `C = 1.9e9` where it shipped with `1e6`. It had zero false negatives on the
> reachable manifold anyway, but only by accident, and its in-crate test passed
> only because the sweep omitted `M = 0`; with perihelion included it
> self-violated 64 times. Found by adversarial review.

Result: `n_masked_det` matches the Python reference **node for node** over the
whole gate_v0 catalogue — 397 900 masked nodes on both sides, zero
disagreements out of 8.0 × 10⁸ nodes.

### Robustness of the Python-facing surface

`_core` is a public extension module, so every value crossing the boundary is
treated as hostile:

* the cell count is a `checked_mul` fold, and the table read is
  bounds-checked. Four 65 537-long edge arrays give `dims = [65536; 4]`, whose
  product is exactly 0 mod 2⁶⁴ — an unchecked product let an empty table pass
  validation and then be indexed ~2⁶⁴ elements out of bounds (a reproducible
  SIGSEGV from pure Python). The bounds check costs nothing measurable
  (12.05 vs 12.18 ms/tracklet — the unchecked build is *slower*, i.e. the
  difference is noise).
* `n_rho`/`n_rhodot` arrive **signed** and are bounded per axis (1e8) and in
  product (1e9 nodes). Previously `n_rho = 1e14` aborted in the allocator
  (exit 134), `n_rho = n_rhodot = 2³²` wrapped the product to zero and then
  span a ~4e9-iteration loop with the GIL released (an uninterruptible hang),
  and a negative count raised `OverflowError` instead of the reference's
  `ValueError`.
* every large allocation uses `try_reserve`, so exhaustion raises
  `MemoryError` rather than aborting.
* model-adaptation failures raise `UnsupportedModel` (a `ValueError` subclass),
  which `pneo.backend` maps onto `ModelNotSupported` so that
  `backend="auto"` falls back to the reference instead of failing where the
  reference succeeds. An explicit `backend="rust"` still raises.

### Numerics

* `f64` throughout.
* Λ accumulators use **Neumaier (compensated Kahan–Babuska) summation** in the
  Python node order (ρ major, ρ̇ minor). Chosen over pairwise summation because
  its error bound is independent of the node count while the traversal stays
  strictly sequential, so repeated runs are bit-identical and the batch result
  does not depend on the worker count. A non-finite running sum stops the
  compensation so that an overflow to `+inf` propagates the way `np.sum` does
  (the classifier's `"overflow"` posterior status depends on it).
* Every arithmetic expression is written in the same association order as the
  numpy expression it replaces (marked `parity:` in the source), so the two
  backends differ only by libm rounding. The ranging grid comes out
  **bit-identical** to `ranging.build_grid`'s, node for node.
* **The ρ axis is passed in, not regenerated.** `np.geomspace` is
  `10 ** linspace(...)`, and `pow` is *not* an IEEE-754 correctly-rounded
  operation: numpy dispatches its `power` loop on runtime CPU features
  (SVML/AVX-512 kernels where available, scalar libm otherwise), so the doubles
  depend on the machine, the numpy build and the libm — not just the numpy
  version. Measured on one host: equally legitimate formulations of the same
  quantity (`exp(y ln 10)`, `exp2(y log2 10)`) differ from `np.power(10, y)` by
  up to **7 ulp on 881 of 1200 nodes**. So `pneo.backend.rho_axis` computes
  the 1200-element axis with the same `np.geomspace` call `build_grid` makes and
  hands the two small arrays (18.8 kB) over; ρ-grid parity is then exact **by
  construction** on every platform. Cost: 0.24 µs/call cached (106 µs
  uncached), i.e. inside the benchmark's noise. Everything the kernel still
  derives itself — the admissible-interval quadratic, the geometric cell edges,
  the ρ̇ midpoints — uses only `+ - * /` and `sqrt`, which IEEE-754 *does* pin
  down, so those stay bit-exact portably.

  > The first version of this port reimplemented `np.geomspace` in Rust and
  > asserted bit-equality. That passed on the development host (where numpy's
  > `power`, glibc's `pow` and Rust's `powf` are all the same libm call) and
  > failed on CI's different CPU — the first CI run the parity suite ever got
  > to. The full-grid test at `n_rho = 180` passed there by luck, not safety:
  > the exposure is probabilistic in the number of exponents sampled. Both
  > resolutions are now covered, plus a regression test that swaps in a
  > deliberately differently-rounding `geomspace` and re-checks the whole parity
  > contract.
* Quantities that depend on ρ alone — the heliocentric position, |r|, the phase
  angle and `H(V, ρ)` — are hoisted out of the ρ̇ loop. That is an exact hoist
  and it removes seven of the ten transcendental calls per node (a 2× speedup).

## Python surface

```python
from pneo.classifier import classify, classify_many

res = classify(att, neo, other, R, V, backend="rust")            # or "auto"/"python"
res = classify(att, neo, other, R, V, backend="rust", diagnostics=False)
out = classify_many(atts, neo, other, backend="rust", n_threads=32)
```

`PNEO_BACKEND=python|rust` overrides the argument process-wide (a
preference, not an assertion: where the kernel cannot be used, the reference
still runs). `PNEO_BACKEND=auto` means "no preference" and overrides
nothing. See `pneo/backend.py` for the model-adaptation rules and the
limitations.

`diagnostics=False` returns only the scalars and the mask counters, with no
per-node allocation in Rust; `diagnostics=True` (the default, and the
historical behaviour) fills the full `ClassificationResult` contract, including
a `RangingGrid` identical to `build_grid`'s.

Sigma points are implemented **only** here, and `classify()` raises
`NotImplementedError` on the Python path when `att.sigma_points` is present, so
neither backend can silently score just the central attributable. The kernel
takes a `(K, 5)` matrix of
`(α, δ, α̇, δ̇, V)` variants plus `(K,)` weights and accumulates
`Λ_C = Σ_k w_k Λ_C^{(k)}`, each variant getting its own admissible region and
grid. v0 calls it with `K = 1`; attach `att.sigma_points` /
`att.sigma_weights` to use more. `diagnostics=True` requires `K = 1` (the
per-node arrays would not correspond to one `RangingGrid`).

### Known reference-side limitation

At `|dec| = 90°` exactly the tangent basis degenerates (`|du/dra| = cos δ = 0`)
and the Python 6×6 composed determinant loses up to 219×, degrading smoothly
from 89.99°. **The Rust closed form is the correct value there** — it is
`ρ⁴cos²δ · 2/(μ^{3/2}√a·e·sin i)` analytically, with no cancellation to lose.
Recorded as a `strict=True` xfail in `tests/test_backend_parity.py`
(`test_reference_jacobian_degrades_at_the_pole`) so that fixing the reference
turns it into an XPASS. Found by adversarial review.

## Measured numbers

Grid (1200, 50), the digest2 model, variant `All`, one core of a 96-core
node:

| path | s/tracklet | speedup |
| --- | --- | --- |
| python, diagnostics on | 482 ms | — |
| python, diagnostics off | 468 ms | — |
| rust, diagnostics on | 12.02 ms | 40× |
| rust, diagnostics off | 11.22 ms | 43× |

(8.2 ms before the guard prefilter was corrected to an honest bound; 31 ms with
no prefilter at all.)

Batch throughput via `classify_many(backend="rust")`, 400 tracklets,
diagnostics off:

| workers | ms/tracklet | tracklet/s |
| --- | --- | --- |
| 1 | 12.50 | 80 |
| 8 | 1.73 | 580 |
| 32 | 0.65 | 1 539 |
| 96 | 0.40 | 2 501 |

The whole 20 000-tracklet gate_v0 catalogue scores in 4.9 s of wall time
(0.245 ms/tracklet) against 1 170 ms/tracklet single-core for the reference.

Golden A/B against the stored `gate_v0` scores, all 20 000 tracklets
(`validation/output/backend_parity/summary.txt`):

| quantity | result | tolerance |
| --- | --- | --- |
| max \|Δp_neo\| | 7.7e-13 | 1e-6 |
| max Λ_NEO relative error | 1.7e-9 (median 2.2e-16) | 1e-5 |
| max Λ_other relative error | 1.4e-10 (median 8.9e-16) | 1e-5 |
| `n_nodes` | identical | exact |
| `n_masked_nonfinite` | identical (0 everywhere) | exact |
| `n_masked_det` | identical (397 900 / 8.0e8 nodes) | — |
| `status` | identical | exact |
| `log_lr` (NaN-aware) | identical NaN/±inf pattern | 1e-5 rel |

These measurements were made with `rust/bench.py` and `validation/backend_parity.py`,
which are kept in the archived repository lsst-ssp/pneo-full.

### Result fields the kernel returns

`classify_core` / `classify_batch` hand back a dict of `lambda_neo`,
`lambda_other`, `n_nodes`, `n_masked_nonfinite`, `n_masked_det`,
`n_h_at_or_above_support`, plus the seven per-node arrays when
`diagnostics=True` (`None` otherwise).

`n_h_at_or_above_support` is the **D-19 mechanism counter**: of the
`n_nodes - n_masked_nonfinite` nodes with finite elements and finite implied H,
how many lie at or above the pair's H support end (`+inf` for a D-14 clamped
pair, `h_extend_to` under D-19, else 25.5). It exists so that
`pneo.classifier._posterior` can tell *why* a 0/0 outcome happened without
the per-node arrays, i.e. in `diagnostics=False` mode. Zero density over a
non-empty region has several causes and only the faint-end one licenses the
D-19 triage score; see `_fully_truncated_in_H`.

`pneo.backend` turns the dict into a `ClassificationResult`, which since
D-19 also carries `status` (`"ok"`, `"fully_truncated_flagged"`,
`"empty_region"`, `"both_zero"`, `"overflow"`, `"nan"`) and the derived
`policy_assigned` flag. Both are produced by the shared post-processing in
`pneo.classifier`, so they are identical between the backends by
construction, and `validation/backend_parity.py` compares them.
