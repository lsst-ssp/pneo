# digest42 Decision Log

*Append-only log of design decisions: what was decided, why, and what was considered
instead. Referenced from [DESIGN.md](DESIGN.md) as [D-n]. All entries 2026-08-31 unless
noted.*

---

## D-1: Two-class problem — NEO vs. everything else

**Decision.** The classifier is a two-component mixture: NEO vs. a single "other"
component aggregating all non-NEO populations (MBA, Hungaria, Mars-crosser, Trojan, Hilda,
TNO, Centaur, …). No subclass identification.

**Rationale.** The operational goal is identifying NEO candidates that require rapid
follow-up; which non-NEO class an object belongs to is irrelevant to that decision.

**Alternatives.** Full multi-class mixture (per-population posteriors) — deferred; the
architecture (per-population density modules) supports it without redesign.

## D-2: NEO defined by population membership

**Decision.** P(NEO) is the posterior weight of the NEO population component, not the
posterior mass with q < 1.3 AU evaluated at each ranging node.

**Rationale.** Simpler and matches the component structure. Requires the non-NEO component
to be cleanly complementary at q = 1.3 AU — automatic in v0 (digest2's classes partition on
that boundary); a checklist item on every component swap.

**Alternatives.** (a) q < 1.3 cut per node — robust to model boundary mismatches, lets a
nominally-MBA density contribute NEO probability in its q < 1.3 tail; (b) computing both as
diagnostics. Revisit if component complementarity ever becomes messy.

## D-3: No prior-odds knob; absolute calibration throughout

**Decision.** Population components are absolute number densities; the posterior has no
tunable prior weight. NEOMOD3 is natively calibrated; S3M-derived components carry the
Wagg et al. (2025) MPCORB-anchored rescaling (f = 0.80); in v0 both components come from
one file and share one calibration by construction.

**Rationale.** With both components absolutely calibrated the prior is determined, not
free. A scalar prior over normalized PDFs is mathematically equivalent, but its value is
convention-coupled to the models' supports and invites misinterpretation; and for a
two-class problem sweeping a prior is redundant with sweeping the decision threshold (the
ranking/ROC is invariant). The log-likelihood ratio ln(Λ_NEO/Λ_other) is reported alongside
P(NEO) as the prior-free sufficient statistic.

**Alternatives.** Normalized PDFs + explicit prior odds w (initially chosen, then dropped);
w as a calibration/sensitivity knob.

## D-4: Hard truncation outside model support

**Decision.** Density is zero outside each model's declared element and H ranges; no
extrapolation of size distributions.

**Rationale.** Conservative and honest about model validity. Analysis showed the belt
proper (r ≥ 1.66 AU) can never imply H beyond typical supports at Rubin depth
(H_max(r) ≈ V_lim − 5 log₁₀(r(r−1)), ≈ 23.8 at the belt's inner edge for V_lim = 24), so
truncation never binds there. The exposed corner — non-NEO orbits with q ∈ (1.3, ~1.5) AU
at V ≳ 23.5, where implied H can exceed the non-NEO support and inflate P(NEO) — is
handled by a mandatory validation flag (DESIGN §2.6, §4) rather than by extrapolation. In
v0 the truncation is symmetric (shared H ≤ 25.5 support).

**Alternatives.** Power-law SFD extrapolation beyond support; common H_max imposed across
components (the fallback if the validation flag shows the corner matters in practice).

## D-5: Photometry — standardized V in, HG phase function with G = 0.15

**Decision.** The tracklet magnitude arrives already standardized to Johnson V; per-class
colors and taxonomy–class coupling are not modeled. H(V, ρ) uses the IAU H,G phase
function with G = 0.15 for all classes, exposed as a parameter.

**Rationale.** Keeps the v0 photometric model to one assumption; G variations and
band→V color coupling are second-order for the triage use case.

**Alternatives.** Ingesting native Rubin bands with per-class color distributions;
per-class phase functions (matters most at large phase angles typical of close NEOs).
Both possible later within the density-module interface.

## D-6: Delta-function attributable in v0; API takes a distribution

**Decision.** v0 treats (α, δ, α̇, δ̇) and V as exact. The public API accepts the
attributable as a distribution (point value + optional covariance) so an outer
marginalization (sigma-point or MC) can be added without interface changes.

**Rationale.** Validate the core method first; rate uncertainties on short Rubin arcs are
significant and will be marginalized in v1.

**Alternatives.** Full 4×4 covariance marginalization from the start; rates-only
uncertainty.

## D-7: Single-tracklet scope

**Decision.** Each tracklet is scored independently; no multi-tracklet/multi-night
evidence accumulation.

**Rationale.** Matches the triage use case (score at first detection); linking is a
separate, downstream problem.

**Alternatives.** Exposing Λ_C(ρ, ρ̇) for later joint scoring — not precluded (the
per-node likelihood maps exist as diagnostics), just not a design requirement.

## D-8: Bound orbits only; ISOs out of scope

**Decision.** The admissible region excludes heliocentric-hyperbolic solutions; no ISO
component.

**Rationale.** Both v0 population components are bound; the bound-orbit constraint prunes
most of the (ρ, ρ̇) plane for free. Adding an ISO class later changes the ranging domain
(hyperbolic solutions admitted) and requires an ISO population model — recorded in
POPULATION_MODELS.md §9.3.

## D-9: Self-contained Python implementation

**Decision.** Plain Python + numpy, vectorized; the attributable→state→elements
transforms and their Jacobians are written in-repo. No pyoorb/adam_core dependency.

**Rationale.** The transforms are simple two-body operations; owning them gives us the
Jacobians (analytic or autodiff) and keeps the dependency surface minimal.

**Alternatives.** pyoorb/adam_core for transformations; JAX-first for GPU + autodiff
Jacobians (kept open as a v2 performance option).

## D-10: v0 bootstrap on digest2's (q, e, i, H) population model — for both components

**Decision.** v0 uses the MPC-maintained digest2 population histogram (45,936 bins over
q, e, i, H; 15 orbit classes; uniform angles) as the density for *both* the NEO component
(digest2's NEO classes) and the other component (all remaining classes). "All" vs. "Unk"
(undiscovered) variants selectable by flag; piecewise-constant evaluation first, with
interpolation only as an explicit option.

**Rationale.** (a) Isolates the methodological contribution: digest42 and digest2 then
share inputs exactly, so any performance difference is attributable to the math (Jacobian,
marginalization, calibrated posterior) — the clean experiment for the paper. (b) Avoids a
calibration seam: digest2's content is unrescaled 2011 S3M while NEOMOD3 is independently
calibrated; with no prior knob (D-3), mixing them in v0 would bake a known ~25%
inconsistency into first results. Within pure digest2, calibration and q = 1.3
complementarity (D-2) hold by construction. (c) Defers the S3M density-estimation problem,
which is orthogonal to getting the method working. (d) The histogram is directly
sampleable, making the self-consistency test sets of DESIGN §4.1 cheap to generate — the
same file serves as prior and as test-population generator.

**Alternatives.** (a) NEOMOD3 + digest2-rest hybrid in v0 — rejected for the seam above;
(b) density-estimating S3M directly in v0 — rejected as the rabbit hole this decision
avoids. Modern components arrive as module swaps in v1/v2 with deliberate
cross-normalization (DESIGN §3, §5).

## D-11: Include the change-of-measure Jacobian

**Decision.** The marginal likelihood integrand includes
|det ∂E/∂(A, ρ, ρ̇)| (plus any element-reparameterization factor), computed via the
state-vector chain rule.

**Rationale.** This is the core methodological difference from digest2: it converts
orbit-space density into observation-space density correctly (volume growth with distance
included), making the posterior a proper probability rather than a heuristic score.

## D-12: Bootstrap-calibrated gate statistics (2026-09-01)

**Decision.** The §4.1 gate's reliability criterion uses 5 equal-*expected-NEO-count*
bins with parametric-bootstrap thresholds (B = 2000; labels re-simulated from the
observed p vector), a gated global count statistic Z_tot (exact Poisson-binomial tail
in mixture mode), and a fitted global odds-scale ĉ with CI — replacing the original
"reliability diagram within binomial error bars" wording. Enriched-mode variance is
w_NEO·w_other·Σp(1−p). The gate ran at N = 2×10⁴ (power 0.88 vs p^1.3 at exact 1%
Type-I) plus an N = 10⁵ calibration run (ĉ to ±15%).

**Rationale.** The WP6 adversarial review measured the naive criterion at a 43%
false-FAIL rate and ~zero power at mixture base rates (~65 NEOs per 10⁴ tracklets put
9 of 10 equal-count bins below one expected NEO). Bootstrap thresholds make the Type-I
error exact by construction regardless of binning or base rate; the review's power
analysis drove both the binning and the doubled N.

**Alternatives.** 10 equal-count bins + |z| < 2 (the original: mis-sized both ways);
min-p multiplicity correction across arms (measured: over-conservative, cost a third
of the power — the plain union of two 1% arms achieves 0.98% family-wise).

## D-13: NEO enrichment is a low-p diagnostic, not a calibration booster

**Decision.** The harness's --neo-fraction mode defaults to f = 0.1, is marked
non-gate-eligible, and is documented as amplifying sensitivity only at p ≲ 0.02.

**Rationale.** More NEOs ≠ more calibration information: enrichment pushes mid-p
tracklets to near-deterministic labels, destroying their calibration content (measured
by both the reviewer and implementer; they disagreed on the size at mid-p — 3.4× worse
vs neutral — but agreed on the conclusion, and both measurements are recorded in the
module docstring). Gate-mode sampling is the calibration test; enrichment probes the
low-p tail.

## D-14: digest2-emulation H-clamp knob (2026-09-02)

**Decision.** `Digest2Histogram(h_top_bin_infinite=True)` reinstates digest2's clamp of
the top H bin to [24, ∞) — H ≥ 25.5 evaluates to the top bin's density instead of D-4's
zero. Emulation is one-sided (digest2's negative-H clamp is not emulated; unreachable at
ρ ≥ 0.01 AU, V ≥ 14). Clamped components are non-normalizable: `sample()` raises, and
the mode is gate-ineligible — comparison use only, both components clamped together.

**Rationale.** Turns the WP7 same-inputs confound into a measurement. Results (gate_v0,
20k): with the premise repaired everywhere, full-set ΔAUC = +0.0126 [+0.0048, +0.0217];
the broken-subset WP7 gap decomposes as +0.0109 method − 0.0021 boundary handling (D-4's
truncation was mildly *handicapping* digest42 there — WP7's number was conservative);
and the clamp explains ~0.01% of digest2's 6.1× calibration overshoot (a correctly
marginalized posterior absorbs the prior's faint-end change: predicted NEOs 187.6
clamped vs 187.5 truncated vs 185 observed). See
validation/DIGEST2_COMPARISON_CLAMPED.md.

**Alternatives.** Leaving D-4 truncation as the only mode (keeps the confound
unmeasured); two-sided emulation including negative-H (rejected: changes nothing
measurable while doubling the deviation from D-4).

## D-15: Rust core kernel behind a backend switch (2026-09-02)

**Decision.** The classify hot path is implemented twice: the original Python/numpy
implementation remains the *reference backend* (unchanged, authoritative for semantics
except at |dec| = 90° exactly — see DESIGN §2.3), and a Rust crate (rust/, PyO3, module
`digest42._core`) provides the production backend — one fused per-node kernel, Jacobian
via the exact Delaunay closed form (no per-node 6×6 or Kepler solve; the Python cond
guard is ported exactly behind a reachability-premised prefilter with a measured 119×
margin), process-lifetime CoreModel with content-fingerprinted caching, sigma-point-ready
(K variants per call; non-Rust backends raise rather than silently ignore),
diagnostics-off zero-allocation mode, rayon batching with bit-reproducible results
independent of thread count. Selection: `backend="auto"|"python"|"rust"` (auto = Rust if
importable and the density pair is core-compatible, else silent fallback);
DIGEST42_BACKEND env overrides only with `python`/`rust`.

**Rationale.** Performance goal (DESIGN §1, issue #1): measured 12.0 ms/tracklet
single-core at (1200,50) vs 482 ms Python (40×), 0.40 ms/tracklet batched at 96 threads;
the 20k gate catalogue scores in ~5 s. Acceptance evidence: golden A/B on gate_v0 —
max |Δp| = 7.7e-13, Λ rel ≤ 1.7e-9, masking decisions identical on 8.0e8 nodes; WP6 gate
re-run on the Rust backend passes with statistics identical to the run of record;
adversarial review found the arithmetic core clean and four shell defects, all fixed
(overflow-unsound unsafe, grid-size crash paths, stale model cache, sigma-point
asymmetry).

**Alternatives.** Numba spike (skipped — Rust chosen directly for the production-service
framing); JAX/GPU (still open for v1 covariance-marginalization scale); maturin wheel
packaging (deferred: needs pyproject build-backend migration; current build is
rust/build.sh + atomic .so install into the editable tree).

## D-16: PyPI packaging — setuptools sdist, standalone-maturin wheels (2026-09-02)

**Decision.** The sdist is built by setuptools (the PEP 517 backend stays
setuptools.build_meta) and installs as pure Python everywhere — no Rust toolchain
required, yielding the numpy reference backend that `backend="auto"` degrades to.
Release wheels are built by maturin driven as a CLI over the `[tool.maturin]` table
(never as the build backend), carrying `digest42._core` as abi3-py311 — one wheel per
platform (linux x86_64, macOS arm64, Windows x64 initially; the rest via sdist).
Publishing: `.github/workflows/release.yml` on GitHub releases, PyPI Trusted Publishing
(OIDC, environment `pypi`, no tokens); `workflow_dispatch` is a build-only rehearsal.
No TestPyPI path (maintainer decision). The `digest42-fetch-model` console script gives
wheel users the model download; a LICENSE file (BSD-3-Clause) now ships in both
artifacts.

**Rationale.** Resolves D-15's deferred packaging question without disturbing anything
proven: zero ci.yml changes (the no-extension degradation test still tests what sdist
users get), dev flow unchanged (`pip install -e .` + `rust/build.sh`), and uncovered
platforms install-and-work instead of failing at build time. Validated locally: wheel
contents audited, clean-venv install + A/B smoke outside the repo tree, abi3 claim
tested (py3.13 on the cp311 wheel), sdist round-trip through maturin, twine strict.

**Alternatives.** maturin as the PEP 517 backend (rejected: forces cargo into every
source install and hard-fails uncovered platforms); pure-Python-only package (rejected:
shipping the kernel is the point); TestPyPI staging (removed at maintainer request —
workflow_dispatch rehearsal covers the build side).
