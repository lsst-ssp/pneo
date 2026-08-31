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
