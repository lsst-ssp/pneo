# pneo Decision Log

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

**Rationale.** (a) Isolates the methodological contribution: pneo and digest2 then
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
truncation was mildly *handicapping* pneo there — WP7's number was conservative);
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
`pneo._core`) provides the production backend — one fused per-node kernel, Jacobian
via the exact Delaunay closed form (no per-node 6×6 or Kepler solve; the Python cond
guard is ported exactly behind a reachability-premised prefilter with a measured 119×
margin), process-lifetime CoreModel with content-fingerprinted caching, sigma-point-ready
(K variants per call; non-Rust backends raise rather than silently ignore),
diagnostics-off zero-allocation mode, rayon batching with bit-reproducible results
independent of thread count. Selection: `backend="auto"|"python"|"rust"` (auto = Rust if
importable and the density pair is core-compatible, else silent fallback);
PNEO_BACKEND env overrides only with `python`/`rust`.

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
(never as the build backend), carrying `pneo._core` as abi3-py311 — one wheel per
platform (linux x86_64, macOS arm64, Windows x64 initially; the rest via sdist).
Publishing: `.github/workflows/release.yml` on GitHub releases, PyPI Trusted Publishing
(OIDC, environment `pypi`, no tokens); `workflow_dispatch` is a build-only rehearsal.
No TestPyPI path (maintainer decision). The `pneo-fetch-model` console script gives
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

## D-17: Mass-conserving smooth-in-H density mode; grid unchanged (2026-09-02)

**Decision.** `Digest2Histogram(h_interpolation="pchip")` reconstructs each (q,e,i)
cell's H density as the derivative of a monotone (Fritsch–Carlson) piecewise cubic
through the cumulative counts — exactly mass-conserving per bin (2e-15), non-negative,
continuous in H, zero outside support, sampleable (inverse-CDF), gate-eligible (PASSED
at N=2e4: ĉ = 0.958 [0.670, 1.370]), bit-identical between backends, mutually exclusive
with the D-14 clamp. Default remains "none" (the D-10 same-inputs experiment). The
production grid stays (1200, 50).

**Rationale — including the honest negative.** Adopted as the more physical prior (the
staircase is a binning artifact, not sky structure — maintainer's diagnosis). The
performance hypothesis behind it did NOT survive measurement: at (1200,50) the H
staircase is a minority error term whose cost largely cancels in the posterior ratio
(mid-p p90 |Δp| improves 2%; the HPD estimator gap did not shrink: 0.000447 vs
0.000358; DESIGN §2.5's 3.5e-3 figure is a mid-p statistic). Neither grid axis
dominates any more at (1200,50). Mode change on the gate catalogue: two-sided score
shifts (Spearman 0.991), ΔAUC +0.0003, calibration absorbed. Cost: +21–28% per-node
(cache footprint, not flops); 2.2 MB coefficients.

**Alternatives.** H-bin-edge-aware ρ node placement (superseded — targeted the same
minority term with more machinery); multidimensional reconstruction in (e, i) with
physical boundary behavior ∝ e, ∝ sin i (the 1/(e sin i) fix; deferred).

## D-18: Operational findings from the real-data NEOCP pilot (2026-09-02)

**Decision.** For NEOCP-like operation: (a) use the **All** variant (measured on 499
real labeled NEOCP tracklets: ΔAUC(All−Unk) = +0.0104 [+0.0039, +0.0187]; the model
file's "known" catalogue is frozen at 2011, so Unk inflates P(NEO) for belt-like
tracklets); (b) use the **D-14 clamped pair** — 2.2% of real tracklets (100% of them
true NEOs; fast, faint, q < 1 AU) have their ENTIRE admissible region above H = 25.5
and score 0/0 under D-4 truncation; the clamp recovers all of them at median |Δp| 2e-6
elsewhere. The real fix is a v1 model whose H support reaches what surveys discover.

**Context.** Pilot headline: calibration survives genuine prior mismatch (pred/obs 1.01
vs digest2's 1.38; ECE 2.2e-2 vs 2.3e-1) on a digest2-censored sample — censoring is a
selection on observables, which cancels per DESIGN §2.4, so calibration is testable
where ranking is not. Only 5/499 real tracklets are "clean" per §2.6(b) (vs 22%
synthetic). The pilot also found the auc_bootstrap stratified-resampling bug (negative
indices addressed the positive block when n_pos ≥ n_neg); fixed, regression-tested, and
all published WP7 CIs regenerated — point estimates unchanged, CIs widened (clamped
full-set ΔAUC now [+0.0015, +0.0253], P = 0.010; enriched full-set +0.0140
[+0.0066, +0.0222] is the strongest ranking evidence).

## D-17/D-18 erratum and addendum (2026-09-02, post-adversarial-review)

The pchip adversarial review reproduced every implementation guarantee independently
(coefficients match scipy's PchipInterpolator to 5.7e-14; mass conservation, exact
non-negativity through density(), sampler, boundaries, backend density parity, stamp
machinery all confirmed) but found the D-17 evidence record needed corrections:

1. **Per-node cost**: the "+21–28%" figure was load contamination (bench taken at
   loadavg 88–162 on 96 cores). Re-measured on the quiet machine with the identical
   protocol: **≲2% at (1200, 50)** (+1.0% median over 25 interleaved reps). The cost
   objection to pchip mode is effectively void.
2. **Quadrature-win statement**: "improves 2%" carried no interval. The improvement's
   bootstrap 95% CI on an independent catalogue is **[0%, +44%]** — the honest claim is
   "no *detectable* quadrature win at (1200, 50)", and the win may be larger than
   recorded. The grid recommendation (unchanged (1200, 50)) stands.
3. **Mid-p reconciliation**: the report's 3.88e-3/1.62e-2 were unreproducible literals;
   the reproducible gate-catalogue mid-p p90 is ≈7×10⁻³ (see DESIGN §2.5 erratum). The
   *direction* of the reinterpretation (DESIGN's 3.5e-3 was never an unconditional
   statistic) was independently confirmed.
4. **Backend parity wording**: "bit-identical between backends" holds for the density
   evaluation; end-to-end Λ agrees at the pre-existing D-15 level (≤2×10⁻¹⁴ here),
   which is not a pchip regression.
5. **Operational constraint (new)**: pchip is mutually exclusive with the D-14 clamp,
   and it inherits D-4's 0/0 failure on fully-H-truncated tracklets exactly — so
   **pchip cannot be the NEOCP operational configuration that D-18 recommends** (the
   clamped histogram pair remains it). Until a v1 model's H support reaches survey
   depth, pchip is a modelling/gate mode.
6. **Guards added post-review**: mixed-mode component pairs (h_interpolation or
   h_top_bin_infinite disagreeing between NEO and other) are rejected — the measured
   mixed-pair artifact (|Δp| up to 6.4×10⁻³) is as large as the honest mode change;
   and the WP6 harness hard-fails on a generator/scorer mode mismatch and prints the
   mode in its summary.
7. **Provenance correction**: due to a supervisor `git add -A` slip, the pchip
   *implementation* physically landed in commit 3db9ef3 ("PyPI packaging"), while
   5ed64e4 carries the study driver and report. History is pushed and stands; auditors
   should look in 3db9ef3 for the code diff.

## D-19: Finite H extension (h_extend_to) and the fully-truncated triage policy (2026-09-02)

**Decision (1) — finite extension.** `Digest2Histogram(h_extend_to=30.0)` appends a
synthetic H bin [25.5, h_extend_to) per (q,e,i) cell carrying the flat continuation of
the top bin's per-unit-H density (digest2's implicit assumption, made finite), hard zero
above. Unlike the D-14 clamp it is a normalizable population: `sample()` works, so
extended mode is **gate-eligible** (first extended-mode gate: OVERALL PASS,
ĉ = 1.134 [0.803, 1.601]), and it composes with D-17 pchip (one more cumulative knot;
mass-conserving; note the extra knot changes the [24, 25.5) pchip curve pointwise by up
to 250% while preserving its mass exactly). Scoring is clamp-equivalent below the new
wall (extended−clamped median exactly 0, max 3×10⁻⁵ on 20k; median 0, max 8.6×10⁻⁶ on
the 499 real tracklets). **This supersedes the D-14 clamp as the operational
configuration** (D-18(b) is superseded accordingly); the clamp survives only as the
digest2-emulation mode for same-inputs comparisons, and remains mutually exclusive with
both h_extend_to and pchip.

**Why 30 and not the reachable ~34** (review-affirmed): 30 is the *validated* reach —
on the 11 real wall tracklets the residual truncation (4/11 keep 7–10% of measure above
30, max implied H 30.9) is invisible in the output (Λ_other = 0 there regardless, so
p_neo is unchanged); extending to 34 would inflate the NEO total ×6.7, making ~85% of
NEO mass pure extrapolation of one bin. The flat-continuation law is scale-free — 30 is
a judgment about how far to extrapolate, not a measurement — and the machinery is
reach-agnostic if that judgment changes. Counts: NEO total ×2.675 (277,469 → 742,145),
other ×1.042, mixture P(NEO) 1.91% → 4.75%.

**Decision (2) — triage policy for fully-truncated tracklets** (maintainer decision).
When the admissible region is non-empty, both Λ are exactly zero, at least one node has
finite implied H, and **the finite-H measure fraction at/above the support end is
≥ 0.95** (the mechanism-gated condition; not merely "both Λ = 0", which the review
showed also occurs for bright/slow objects inside the support and must keep returning
nan with status "both_zero"), the classifier assigns **p_neo = 1.0 as an explicit triage policy**: status
"fully_truncated_flagged", policy_assigned=True, log_lr = nan, warning naming it a
policy, implemented once in shared post-processing so backends cannot diverge.
Empirical basis: 11/11 fully-truncated real NEOCP tracklets were true NEOs (fast + faint
⇒ close ⇒ small). The 0.95 threshold sits in a measured total-separation gap — the 11 wall
tracklets have fractions ≥ 0.9856 (5 of them below 1.0 only via slivers of nodes at
H 25.4–25.5 in empty cells, which a strict all-nodes rule would wrongly exclude), while
every constructed false-positive class sits at exactly 0 — and a unit test pins it
inside (0, 0.9856]. This is an operational assignment, not a posterior — accordingly,
**policy rows are excluded from the gate's reliability inputs** (measured: ~34 correct
policy rows would make one of five bins vacuous) and do not count as likelihood-defined
scores; the harness reports undefined / policy-assigned / likelihood-defined as three
distinct counts. With h_extend_to = 30 the policy is a rare residual (0 occurrences on
all synthetic catalogues; it exists for real tracklets beyond the extended wall).

**Supersessions.** D-18(b) (clamped pair for NEOCP-like operation) → use the extended
pair. D-17/D-18 erratum item 5 ("pchip inherits D-4's 0/0 failure"; "pchip cannot be
the NEOCP operational configuration") → superseded: extension+pchip composes and is
operationally eligible; there is no 0/0 nan any more for genuinely fully-truncated
tracklets (policy) and the extension covers the measured H range.

## D-20: The pneo CLI (2026-09-02)

**Decision.** The wheel ships a `pneo` console entry (stdlib argparse only — no CLI
dependencies; maintainer constraint) with `score` (observation files in, one row per
tracklet out), `fetch-model`, and `fetch-obscodes`. Scoring defaults are the D-18/D-19
**operational** configuration (All variant, h_extend_to = 30, h_interpolation none,
(1200, 50) grid, backend auto, triage policy active and visible in the status column) —
not the library defaults. `--workers` defaults to all cores (maintainer decision):
one batched rayon call on the Rust backend, a multiprocessing pool with per-worker
model loading on the reference. Failure degrades per tracklet on every path; stdout
carries only data; exit codes 0 (≥1 scored) / 1 (nothing scored) / 2 (configuration).
Ingestion (`pneo.ingest`, also public API) is the NEOCP pilot's machinery promoted
into the package, equivalence-tested against the pilot's 499 stored real tracklets.

**Deliberate divergences from digest2's ingestion, recorded rather than hidden:**
1. *Blank band on MPC 80-column records* → treated as V (offset 0.0), where digest2
   applies its −0.8 unknown-band fallback — a 0.8 mag difference. pneo's choice is
   the better default (V is the declared standardization target, D-5; −0.8 is digest2's
   crudest path); the divergence is stated in the docstrings and --help, and counted on
   stderr per run. The band-offset *table* itself is digest2's verbatim (verified
   entry-by-entry), plus the two-character ADES aliases digest2 lacks.
2. *note2 filter*: digest2's 80-col parser accepts only note2 ∈ {C, S, B}; pneo
   accepts everything except space/roving/radar record types (skipped and counted).
3. *Tracklet selection*: earliest same-night same-station set (deterministic from the
   file alone), where the pilot used the externally-known discovery station.
4. *Packed MPC designations are not unpacked* (grouping-only risk; documented).

**Rationale.** digest2's audience is command-line users; before the CLI, the wheel
offered them nothing. The end-to-end CLI reproduces the pilot's real-data scores
(median |Δp| ~ 0, max 8.6×10⁻⁶ vs the clamped reference — D-19's published
extended-vs-clamped agreement) and scores the 499-tracklet pilot sample in ~6.5 s.
Known operational hazard, mitigated: stations outside the built-in 14-code table fall
back to the geocentre (up to ~0.13 in p_neo for survey stations) — mitigations are the
local obscodes table (`--obscodes` / `$PNEO_OBSCODES` / `./data/ObsCodesF.html`,
fetched once via `pneo fetch-obscodes`), per-code warnings, `obscode_source` in the
JSON output, and a run-level warning when >10% of tracklets used the fallback.

## D-21: Project renamed digest42 → pneo (2026-09-08)

**Decision.** The project is renamed from digest42 to pneo, across the package, CLI,
environment variable, Rust crate, build/release configuration, all documentation
(including retroactively updating D-1 through D-20's self-references), and the paper
manuscript. Third-party references (the MPC's digest2 tool; rlseaman/digest3) are
unaffected. **Rationale.** Maintainer decision. No design, method, or validated result
changed — only the project's own name throughout its history of documentation.
