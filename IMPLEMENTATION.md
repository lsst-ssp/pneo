# digest42 Implementation Plan

*Execution plan for building v0 as specified in [DESIGN.md](DESIGN.md). Work is performed
by a team of implementation agents (Opus-class), supervised and integrated by a
coordinating agent; math-critical contributions receive independent adversarial review.
Status: awaiting approval to begin.*

## 1. Ground rules and conventions

Pinned up front so parallel agents cannot diverge:

- **Language/deps.** Python ≥ 3.11. Runtime dependencies: `numpy`, `astropy` (time scales
  and planetary ephemerides *only* — no coordinate-frame shortcuts for the science math),
  `matplotlib` (diagnostics only). Dev: `pytest`. Explicitly excluded: pyoorb, adam_core,
  spiceypy (per D-9).
- **Units.** AU, day, radian. Gaussian gravitational constant k = 0.01720209895,
  μ_⊙ = k². All internal state vectors heliocentric.
- **Frames.** Attributables are ICRS/equatorial (α, δ, α̇, δ̇). Orbital elements are
  heliocentric ecliptic J2000 (MPC/digest2 convention). The equatorial↔ecliptic rotation
  lives in `transforms.py` and is applied exactly once, at a documented point.
- **Time.** Epochs are TDB internally; constructors accept astropy `Time` and convert.
- **Photometry.** H and V in Johnson V; IAU H,G phase function, G = 0.15 default (D-5).
- **Style.** numpy-vectorized public functions (leading array dimension = batch);
  dataclasses for value types; every public function's docstring states units and frame.
- **Repo layout.**

```
pyproject.toml
src/digest42/
  __init__.py
  constants.py      units, k, μ, obliquity; single source of truth
  attributable.py   WP1
  transforms.py     WP2
  ranging.py        WP3
  population.py     WP4
  classifier.py     WP5
tests/              unit tests per module + integration tests
validation/         WP6, WP7 harnesses and their outputs (plots not committed)
data/               fetched digest2 model file (gitignored) + fetch script
```

## 2. Work packages

Interface contracts (signatures + docstrings + dataclass definitions) are frozen by the
supervisor in WP0 **before** parallel work starts; agents implement against them. Each WP
must ship with its unit tests passing; acceptance criteria below are the merge gate.

### WP0 — Skeleton and interface contracts *(supervisor)*
Package scaffolding, `constants.py`, frozen stubs for every public class/function in
WP1–WP5 (types, units, frames in docstrings), pytest wiring, digest2-model fetch script
(license permitting; see risks). **Acceptance:** `pip install -e . && pytest` runs (stubs
xfail); contracts reviewed against DESIGN §2–3 before Phase B launch.

### WP1 — Attributable and observer state (`attributable.py`)
`Attributable` dataclass (α, δ, α̇, δ̇, epoch, V; optional covariance field, unused in v0
per D-6); observer heliocentric position/velocity from astropy ephemerides + observatory
geolocation (MPC observatory code or explicit `EarthLocation`); geocentric fallback.
**Acceptance:** observer state matches JPL Horizons reference vectors (hardcoded test
fixtures) to ≤ 10⁻⁶ AU and ≤ 10⁻⁸ AU/day; round-trip epoch handling across time scales.

### WP2 — Transforms and Jacobians (`transforms.py`) **[math-critical → independent review]**
- $\hat{u}(\alpha,\delta)$ and tangent-basis derivatives; (A, ρ, ρ̇) → heliocentric
  (r, v) (DESIGN §2.1), including the equatorial→ecliptic rotation.
- State → osculating elements (bound orbits), plus (q, e, i) extraction for the digest2
  parameterization.
- **Forward generator** (needed by WP6): elements + epoch + observer → (A, V) — i.e. the
  inverse map, including light-of-sight solution and H → V via §2.2.
- H(V, ρ): phase angle, H,G phase function, and its inverse.
- Jacobians: analytic ∂(r,v)/∂(A,ρ,ρ̇); ∂E/∂(r,v) analytic or via numerical
  differentiation; composition per DESIGN §2.3, including the (a,e) → (q,e)
  reparameterization factor.

**Acceptance:** element↔state round-trips to 10⁻¹⁰ relative; every Jacobian verified
against central finite differences over a randomized sweep of geometries (relative error
≤ 10⁻⁶ away from degeneracies, degeneracy loci documented); forward generator ∘ ranging
transform is the identity on (ρ, ρ̇); spot checks against an independent implementation
(e.g. astropy two-body or hand-computed cases) for a handful of orbits.

### WP3 — Admissible region and grids (`ranging.py`)
Per-ρ admissible ρ̇ interval from the heliocentric energy constraint (DESIGN §2.5);
optional geocentric-unbound cut; log-ρ × linear-ρ̇ grid with cell measures; degenerate
(empty-region) handling. **Acceptance:** admissible-region boundaries match an independent
brute-force scan of the energy condition; total region area converges under grid
refinement; qualitative agreement with published admissible-region figures (Milani et al.
2004) for reproduced geometries.

### WP4 — Population densities (`population.py`) **[semantics-critical → independent review]**
`PopulationDensity` abstract interface: `density(elements, H)`, element parameterization
tag, support description, class label (per DESIGN §3). `Digest2Histogram`: parse the
MPC `digest2.model.csv`; bin-volume normalization to objects per unit (q, e, i, H) volume
(uniform angles); class grouping into NEO vs. other; All/Unk variants; piecewise-constant
evaluation (interpolation stub, off by default per D-10); **`sample(n)`** drawing (E, H)
from the histogram (uniform within bins, uniform angles) — the §4.1 test-population
generator. **Acceptance:** integrated counts per class reproduce the file's totals;
NEO/other partition is exactly complementary at q = 1.3 AU; `sample()` output histograms
converge to the model under binning; densities are zero outside declared support.

### WP5 — Classifier (`classifier.py`) *(depends on WP1–4)*
Orchestration per DESIGN §2.3–2.5: grid over the admissible region → transforms →
density evaluations → Λ_NEO, Λ_other, P(NEO), log-likelihood ratio; per-node records for
the (ρ, ρ̇) posterior diagnostic maps; batch API over many tracklets. **Acceptance:**
integration test on hand-constructed cases (an obvious NEO geometry scores high, an
obvious MBA geometry low); results invariant under grid refinement to stated tolerance;
posterior maps render.

### WP6 — Self-consistency validation harness (`validation/self_consistency.py`) **[gate → independent review]**
Implements DESIGN §4.1: sample objects via WP4 `sample()`, generate synthetic tracklets
via the WP2 forward generator over randomized epochs/geometries (with a V < 24 cut),
score with WP5, produce reliability diagrams, collapse tests, and the targeted slices
(opposition rate scans, admissible-region edges, truncation corner §2.6).
**Acceptance — this is the v0 gate:** reliability diagram consistent with the diagonal
within binomial error bars at N ≥ 10⁴ tracklets; collapse tests pass for known-orbit
objects. *The harness itself is independently reviewed before the gate result is
trusted — a buggy test can pass buggy code.*

### WP7 — digest2 head-to-head (`validation/digest2_comparison.py`)
Implements DESIGN §4.2: emit the WP6 synthetic tracklets as two-detection observation
pairs in a format digest2 accepts (ADES or MPC 80-col), run the pip-installed MPC digest2
on them, compare ROC/ranking against digest42 on identical inputs. **Acceptance:** digest2
runs end-to-end on generated input; comparison plots + a short written summary of the
ranking gap.

## 3. Execution schedule

| Phase | Work | Executor | Parallel? |
|---|---|---|---|
| A | WP0 skeleton + frozen contracts | supervisor | — |
| B | WP1, WP2, WP3, WP4 | 4 implementation agents (Opus) | yes — disjoint file ownership |
| B-rev | independent review of WP2, WP4 | 2 reviewer agents (Opus), adversarial | yes |
| C | WP5 | 1 implementation agent (Opus) | after B integrated |
| C-rev | review of WP5 | reviewer agent | — |
| D | WP6 harness | 1 implementation agent (Opus) | after C |
| D-rev | independent review of WP6 harness, then run the gate | reviewer agent + supervisor | — |
| E | WP7 + docs polish | 1 agent + supervisor | after D gate passes |

## 4. Supervision and review protocol

- **Contracts first.** No Phase B agent starts before WP0 contracts are committed; agents
  implement to the frozen signatures and may not alter another WP's files. Interface
  change requests go through the supervisor (contract commit + rebroadcast).
- **Agent briefs.** Each agent receives: DESIGN.md, its WP section verbatim, the frozen
  contracts, and the conventions block (§1). Deliverable: implementation + unit tests +
  a summary of assumptions and known limitations.
- **Integration.** The supervisor merges each WP, runs the full test suite, and
  spot-reads every diff. Nothing merges with failing or skipped acceptance tests.
- **Independent review** for math/semantics-critical WPs (WP2, WP4, WP6; WP5 lighter):
  a separate Opus agent with no stake in the implementation, briefed adversarially
  ("find the error in the Jacobian/measure/normalization; verify claims independently,
  including re-deriving one Jacobian column and re-checking bin-volume normalization by
  hand"). Findings triaged by the supervisor; fixes go back to the implementing agent or
  are applied directly; the reviewer confirms resolution.
- **Gate discipline.** v0 is done only when the WP6 reliability diagram is diagonal
  within error bars (DESIGN §4.1) — an objective pass/fail that does not depend on anyone's
  judgment of the code.
- **Commit discipline.** One commit per WP merge (plus review-fix commits), pushed to
  main after the suite passes; decision-relevant deviations recorded in DECISIONS.md.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| digest2 model CSV format/license surprises | WP0 fetch script verifies format hash + license before Phase B; fallback: rebuild via the MPC's documented population pipeline |
| Jacobian conditioning near degeneracies (e = 0, i = 0, circular/parabolic boundaries) | FD sweeps in WP2 explicitly probe near-degenerate geometries; document + mask loci rather than patch silently |
| Ephemeris/frame convention drift between WP1 and WP2 | Conventions pinned in §1 + `constants.py`; one integration test computes a full pipeline result for a Horizons-checked geometry |
| WP6 gate fails for a subtle measure reason | The (ρ, ρ̇) collapse maps and per-node records are built for exactly this debugging; §4.1 slices localize which regime is miscalibrated |
| digest2 input-format friction in WP7 | WP7 is last and non-blocking for the v0 gate; ADES preferred, 80-col fallback |
| Agent scope creep (features not in DESIGN) | Briefs quote acceptance criteria as the definition of done; supervisor rejects out-of-scope diffs |

## 6. Definition of done (v0)

1. `pip install -e . && pytest` green, including all WP acceptance tests.
2. WP6 self-consistency gate passed (diagonal reliability, N ≥ 10⁴).
3. WP7 comparison produced with a written summary.
4. All math-critical WPs carry a completed independent review with resolved findings.
5. README updated with quickstart; DECISIONS.md updated with any deviations.
