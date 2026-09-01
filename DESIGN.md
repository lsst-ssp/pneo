# digest42 Design

*Design document for the digest42 Bayesian tracklet classifier. Status: design phase
(2026-08). Decisions referenced as [D-n] are recorded in [DECISIONS.md](DECISIONS.md);
population-model inputs are surveyed in [POPULATION_MODELS.md](POPULATION_MODELS.md).*

## 1. Goals and scope

**Goal.** For a single asteroid tracklet — an attributable $A = (\alpha, \delta,
\dot\alpha, \dot\delta)$ at epoch $t$ plus an apparent magnitude $V$ — compute a
**calibrated posterior probability** that the object belongs to the NEO population, for use
in rapid follow-up triage of Rubin/LSST discoveries. This is a two-class problem: NEO vs.
everything else (MBA, Trojan, TNO, …); we do not care which non-NEO subclass an object
belongs to [D-1].

**Why not digest2.** digest2 (Keys et al. 2019) answers the same question with a heuristic
score. digest42 differs in method: a proper Bayesian formulation with (a) the exact
change-of-measure Jacobian between element space and observable space, (b) marginalization
over the admissible region, (c) absolutely calibrated population components and therefore
no tunable prior, and (d) an output that is a probability, not a score. To make the
methodological comparison clean, **v0 uses digest2's own population model as input** [D-10]:
same inputs, different math — any difference in performance is attributable to the method.

**Scope for v0:**
- Single tracklet only; no multi-tracklet evidence accumulation [D-7].
- Bound heliocentric orbits only; ISOs are out of scope [D-8].
- The attributable is treated as exact (delta function); the API accepts a distribution so
  covariance marginalization can be added without redesign [D-6].
- $V$ arrives already standardized to Johnson V from the native survey bands; color/taxonomy
  coupling is not modeled [D-5].
- Methodology first, then Rubin-scale application; v0 optimizes for correctness and
  diagnostics, not throughput.

## 2. Formalism

### 2.1 Geometry: from observables to orbits

Let $\mathbf{R}(t)$, $\dot{\mathbf{R}}(t)$ be the observer's heliocentric position and
velocity at the tracklet epoch (from a planetary ephemeris + observatory geolocation).
Define the topocentric unit vector and its tangent-plane derivatives:

$$
\hat{\mathbf{u}}(\alpha, \delta), \qquad
\hat{\mathbf{u}}_\alpha = \frac{\partial \hat{\mathbf{u}}}{\partial \alpha}, \qquad
\hat{\mathbf{u}}_\delta = \frac{\partial \hat{\mathbf{u}}}{\partial \delta}.
$$

Given hypothesized topocentric range $\rho$ and range-rate $\dot\rho$, the heliocentric
state of the object is

$$
\mathbf{r} = \mathbf{R} + \rho\, \hat{\mathbf{u}}, \qquad
\mathbf{v} = \dot{\mathbf{R}} + \dot\rho\, \hat{\mathbf{u}}
           + \rho\, ( \dot\alpha\, \hat{\mathbf{u}}_\alpha + \dot\delta\, \hat{\mathbf{u}}_\delta ),
$$

and the map to osculating elements $E = (a, e, i, \Omega, \omega, M)$ at epoch $t$ is the
standard two-body state-to-elements conversion. The composite map

$$
\Phi:\; (A, \rho, \dot\rho) \;\mapsto\; E
$$

is a smooth bijection between two 6-dimensional spaces (away from degenerate loci), which
is what makes the change of measure below well-defined.

### 2.2 Photometry: from apparent to absolute magnitude

At each node, heliocentric distance $r = |\mathbf{r}|$ and phase angle $\alpha_{\rm ph}$
(Sun–object–observer) are known. With the IAU H,G phase function $\Phi_{HG}$ and default
$G = 0.15$ for all classes [D-5]:

$$
H(V, \rho) = V - 5 \log_{10}(\rho\, r) + 2.5 \log_{10} \Phi_{HG}(\alpha_{\rm ph}; G).
$$

Since $\Phi_{HG} \le 1$, nonzero phase always lowers the implied $H$; the maximum
implied $H$ at fixed $V$ occurs at opposition. Note $\partial H / \partial V = 1$, so the
magnitude dimension contributes a unit Jacobian factor.

### 2.3 The marginal likelihood

Each population class $C \in \{\mathrm{NEO}, \mathrm{other}\}$ is described by an
**intrinsic, absolutely calibrated number density** $n_C(E, H)$: objects per unit
element-space volume per unit magnitude. The number of class-$C$ objects presenting an
attributable in $dA$ around $A$ with hidden variables in $d\rho\, d\dot\rho$ is

$$
dN_C = n_C\big(E, H\big)\, dE\, dH
     = n_C\big(\Phi(A,\rho,\dot\rho),\, H(V,\rho)\big)\,
       \left| \det \frac{\partial E}{\partial (A, \rho, \dot\rho)} \right|
       dA\, d\rho\, d\dot\rho\, dV .
$$

Marginalizing the hidden variables gives the class marginal likelihood (a differential
number density in observable space):

$$
\Lambda_C(A, V) = \iint_{\mathcal{AR}}
  n_C\big(\Phi(A,\rho,\dot\rho),\, H(V,\rho)\big)\,
  \left| \det \frac{\partial E}{\partial (A, \rho, \dot\rho)} \right|
  \, d\rho\, d\dot\rho ,
$$

integrated over the admissible region $\mathcal{AR}$ (§2.5). The Jacobian is the piece
heuristic approaches omit: it is the change of measure that converts "density of orbits"
into "density of observations," and it encodes, among other things, the growth of the
volume element with distance. Numerically it factors through the state vector,

$$
\frac{\partial E}{\partial (A, \rho, \dot\rho)}
 = \frac{\partial E}{\partial (\mathbf{r}, \mathbf{v})}
   \cdot
   \frac{\partial (\mathbf{r}, \mathbf{v})}{\partial (A, \rho, \dot\rho)},
$$

where the second factor is analytic from §2.1 and the first is the standard
elements-from-state partial matrix (or obtained by automatic/numerical differentiation).
If the density is parameterized in different element coordinates (e.g. $(q, e, i)$ rather
than $(a, e, i)$ — the digest2 case), the corresponding reparameterization Jacobian is
included so the measure stays consistent.

### 2.4 The posterior

$$
P(\mathrm{NEO} \mid A, V)
 = \frac{\Lambda_{\rm NEO}}{\Lambda_{\rm NEO} + \Lambda_{\rm other}} .
$$

**No prior knob** [D-3]: because the $n_C$ are absolute number densities, the relative
abundance of the classes is already contained in the components. NEOMOD3 is natively
calibrated; S3M-derived components carry the Wagg et al. MPCORB-anchored rescaling
(0.8×); in v0 both components come from the same digest2/S3M file and share one
calibration by construction.

**Why survey selection cancels.** The probability that this survey detects a tracklet
with observables $(A, V)$ is a single function $\varepsilon(A, V)$ — the same observation,
whatever the object is. It multiplies both $\Lambda_{\rm NEO}$ and $\Lambda_{\rm other}$
and cancels in the ratio. This is why the inputs must be *intrinsic* (debiased)
populations: using the discovered catalog would import discovery selection effects that do
**not** cancel.

**NEO definition** [D-2]: P(NEO) is the posterior weight of the NEO *population component*
(membership definition), not the posterior mass with $q < 1.3$ AU evaluated node-by-node.
This requires the two components to be cleanly complementary at $q = 1.3$ AU — automatic
in v0 (digest2's classes partition on exactly that boundary), and a checklist item whenever
a component is swapped.

### 2.5 Admissible region and numerical integration

The integration domain in $(\rho, \dot\rho)$ is bounded by (Milani et al. 2004):
- $\rho > \rho_{\min} > 0$ (excluding the immediate observer vicinity; configurable),
  $\rho < \rho_{\max}$ set by the outermost population support;
- the bound-orbit constraint: heliocentric energy
  $\tfrac{1}{2}|\mathbf{v}|^2 - \mu/r < 0$, which for each $\rho$ restricts $\dot\rho$ to
  an interval (possibly empty) — this prunes most of the plane;
- optionally, exclusion of geocentrically bound solutions (satellites).

v0 integrates on a grid: $\rho$ log-spaced (the interesting structure spans
$10^{-2}$–$10^{1.5}$ AU), $\dot\rho$ linear within the per-$\rho$ admissible interval,
likelihood accumulated as $\sum_k w_k\, n_C(E_k, H_k)\, |J_k|$ with $w_k$ the cell
measures. Monte Carlo sampling of $\mathcal{AR}$ is a drop-in alternative when covariance
marginalization is added.

**Integrable-singularity caveat (WP2 review).** The exact measure carries
$|J| \propto 1/(e \sin i)$ — for a smooth population, $n(e) \to 0$ as $e \to 0$ cancels
it, but a piecewise-constant density (the digest2 histogram) is finite across
$e \in [0, 0.1)$ and $i \in [0°, 2°)$, so the integrand has genuine integrable
singularities along the $e = 0$ and $i = 0$ curves in the $(\rho, \dot\rho)$ plane. The
integral converges, but grid quadrature across those curves converges slowly; expect mild
grid dependence of $\Lambda$ for near-ecliptic, near-circular hypotheses. The
grid-refinement acceptance tests (WP5, §4.1) are the guard; the fix is quadrature
refinement, never a change to the (correct) Jacobian.

**Where the quadrature error actually lives (WP5 review).** With the digest2 histogram,
the dominant grid error is not the $e$/$\sin i$ singularity but the **ρ-axis staircase**:
$H(\rho)$ depends on $\rho$ alone, so every H-bin and q-bin edge is a ρ-aligned
discontinuity of the integrand, and $\dot\rho$ refinement buys nothing (measured: at
fixed cost, $800{\times}100$ beats $400{\times}200$ by ~2× and $200{\times}400$ equals
$200{\times}100$). Sensible v0 grids are ρ-heavy — $n_\rho \approx 1200$,
$n_{\dot\rho} \approx 50$ (p90 $|\Delta p| \approx 0.0035$, max $\approx 0.01$ vs a
$1200{\times}600$ reference). The targeted future fix is node placement on the H-bin
edge curves, not global refinement.

### 2.6 Support truncation and its known artifact

Outside a model's declared support (element ranges, $H$ range) the density is zero — hard
truncation, no extrapolation [D-4]. Consequence, worked out during design: with
$H_{\max}(r) \approx V_{\lim} - 5\log_{10}(r(r-1))$ at opposition and $V_{\lim} = 24$,

| $r$ (AU) | 1.3 | 1.45 | 1.55 | 1.67 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|
| $H_{\max}$ | 26.1 | 25.0 | 24.4 | 23.8 | 23.2 | 22.5 |

the main belt proper ($r \ge 1.66$ AU) can never imply $H$ beyond typical model supports —
truncation never binds there. The exposed corner is non-NEO orbits with
$q \in (1.3, \sim1.5)$ AU (Mars-crosser zone) observed near perihelion/opposition at
$V \gtrsim 23.5$, where implied $H$ can exceed the non-NEO support and P(NEO) inflates by
model boundary rather than physics. In v0 both components share the digest2 support
($H \le 25.5$) — but the WP4 review established that a shared boundary is **not**
symmetric truncation in effect: 55.8% of the NEO component's total mass sits in the
single top H bin $[24, 25.5)$ versus 1.4% of the other component's, so the faint-end
mixture ratio is ~40× more sensitive per unit mass to the H boundary on the NEO side.
Validation must therefore flag, per tracklet, (a) posterior mass in
$\{q > 1.3,\, H > H_{\max}^{\rm model}\}$ and (b) the admissible-region measure with
implied $H > 25.5$ (nodes both components zero out purely by boundary) (§4).

### 2.7 Measurement uncertainty (deferred, designed-for)

The formalism above treats $A$ and $V$ as exact. The generalization replaces
$\Lambda_C(A, V)$ with $\int \Lambda_C(A', V')\, p(A', V' \mid \text{data})\, dA'\, dV'$ —
an outer marginalization over the astrometric/photometric error distribution (sigma-point
or MC). The public API therefore takes the attributable as a distribution from day one;
v0 implements the delta-function case [D-6].

## 3. Architecture

Self-contained Python (numpy-vectorized); the two-body transforms and Jacobians are
written in-repo rather than imported from pyoorb/adam_core [D-9].

```
digest42/
  attributable.py   Attributable + epoch + (future) covariance; observer state from ephemeris
  ranging.py        admissible region construction; (ρ, ρ̇) grids / samples
  transforms.py     (A, ρ, ρ̇) → state → elements; H(V, ρ); Jacobians
  population.py     PopulationDensity interface + implementations
  classifier.py     marginal likelihoods, posterior, diagnostics
```

**The density interface is the load-bearing abstraction** [D-10]: a `PopulationDensity`
exposes `density(elements, H) → objects per unit (element, H) volume`, its element
parameterization (so the classifier applies the right reparameterization Jacobian), its
support, and its class label (NEO / other). Everything else is generic. Planned
implementations:

1. **`Digest2Histogram`** (v0): the MPC-maintained digest2 $(q, e, i, H)$ binned model
   (45,936 bins, 15 orbit classes, uniform angles), grouped into NEO vs. other. Both the
   "All" (total) and "Unk" (undiscovered = total − known) variants, selectable by flag —
   "All" for validation against complete simulated populations, "Unk" as the arguably
   correct prior for unattributed real tracklets. Piecewise-constant evaluation first;
   interpolation only as an explicit option (interpolating near empty bins invents
   density).
2. **`NEOMod3Grid`** (v1): NEOMOD3 $(a, e, i, H)$ with analytic uniform angle factors.
3. **Parametric components** (v2): Trojans, Hildas (Vokrouhlický et al.) evaluated
   directly; OSSOS TNO components; the SMB belt model when released.

Swapping a component is a deliberate act that must re-check: absolute normalization
consistency across components, complementarity at $q = 1.3$ AU, and support boundaries
(the §2.6 artifact moves when supports differ).

## 4. Validation plan

A hierarchy: correctness first, on test sets whose generating distribution matches the
prior exactly; robustness later, on realistic sets that deliberately do not. The two must
not be conflated — on a test population that differs from the prior, even a perfect
implementation shows miscalibration, and method error becomes indistinguishable from prior
mismatch.

### 4.1 Self-consistency (v0 gate, self-generated test sets)

A Bayesian classifier is *exactly* calibrated when the test population is drawn from the
same distribution as its prior. We therefore generate our own test sets by sampling
$(E, H)$ from the digest2 histogram itself (directly sampleable: binned density, uniform
angles), placing objects at random epochs and observer geometries, and computing the
implied $(A, V)$. On these sets:

- **Reliability diagrams must be diagonal.** Among tracklets scored
  $P(\mathrm{NEO}) \approx p$, the true NEO fraction must be $\approx p$; any deviation is
  a bug — in the Jacobian, the measure, the admissible-region bounds, or the integration —
  because prior mismatch is zero by construction.
- **Collapse tests.** The $(\rho, \dot\rho)$ posterior maps must cover and concentrate
  around each object's true $(\rho, \dot\rho)$; the diagnostic maps are a first-class
  output.
- **Targeted slices**, also self-generated: controlled rate scans at opposition (the
  NEO/MBA separation should match intuition), admissible-region edges, and the truncation
  corner of §2.6 (report posterior mass at $\{q > 1.3,\, H > H_{\max}^{\rm model}\}$;
  expected rare — the faintest ~1.5 mag of the tracklet population — verify empirically).

Selection applied on the observables — a $V < 24$ cut, a toy footprint — does not break
$(A, V)$-conditional calibration, so these sets can be made survey-like without a survey
simulator.

**Unbound-draw filtering.** The digest2 model's top eccentricity bin straddles $e = 1$,
so `sample()` yields formally unbound objects — measured at 1.93% of All/NEO draws
(2.24% Unk/NEO) versus ~2×10⁻⁶ of "other" (WP4 review). The generator must discard
unbound draws and report the discarded fraction per component. Calibration is preserved:
the classifier's admissible region restricts the likelihood integral to bound orbits
identically, so both the generating distribution and the effective prior are the
bound-restricted model. Rejection must be **joint** (an unbound/geometry/V-cut rejection
discards the class draw too): per-class redraw would reproduce the unconditioned mixture
instead of the bound-restricted, selection-conditioned one the posterior describes —
verified to produce a ~2.5× calibration distortion in a toy study (WP6 review).

**What the gate does and does not validate (WP6 review).** Two structural limits, to be
stated rather than discovered:

1. *Jacobian normalization cancels.* With both v0 components in the same
   parameterization, $P(\mathrm{NEO})$ is exactly invariant under $|J| \to c\,|J|$ — the
   gate validates the Jacobian's **local ρ-shape** (the $(\rho,\dot\rho)$ posterior is a
   near-delta: median 99.99% of mass within a factor 2 of the true ρ) and the collapse
   statistics, not the measure's absolute scale. The (1−e) qei/aei pairing is pinned by
   unit tests and the exact Delaunay closed form instead (WP2).
2. *Reliability power scales with the accepted NEO count.* At mixture proportions,
   $N = 10^4$ yields ~65 NEOs: enough to constrain a global odds error only to roughly
   ±100% above $p \approx 0.05$ (a 20% odds error produces an expected significance of
   0.8σ; 3σ requires $N \approx 1.4\times10^5$). The reliability criterion must be sized
   accordingly: bootstrap-calibrated thresholds (exact Type-I regardless of binning or
   base rate), bins with equal *expected NEO count*, the corrected enriched-mode variance
   $w_{\rm NEO} w_{\rm other} \sum p(1-p)$, and the single global statistic
   $Z_{\rm tot} = \sum w(y-p)/\sqrt{w_{\rm NEO} w_{\rm other} \sum p(1-p)}$ with a fitted
   odds-scale $\hat c$ — the most powerful summary at these base rates. NEO enrichment
   amplifies power only at $p \lesssim 0.02$ (use $f \approx 0.1$, never 0.5, and only as
   a low-p diagnostic). A claim of calibration above $p \approx 0.05$ requires the
   $N \sim 10^5$ run.

### 4.2 Method comparison on shared inputs (v0)

digest2 vs. digest42 on the same self-generated sets: both codes use the same population
file, and the test population matches that shared prior, so the ranking (ROC) gap isolates
the value of the Jacobian + marginalization + calibrated posterior. This is the clean
experiment for the methods paper.

One caveat to the "same inputs" premise (WP4 review): digest2's lookup clamps $H \ge 24$
into a $[24, \infty)$ top bin, while digest42 truncates at 25.5 per D-4 — so for ranging
nodes with implied $H > 25.5$ the two codes consume *different* effective models. The
comparison must report the fraction of tracklets with non-negligible admissible-region
measure at $H > 25.5$ (the §2.6 diagnostic) and either exclude them or discuss them
separately.

### 4.3 Prior-misspecification stress test (v1+)

In production the true population matches nobody's prior. Scoring tracklets generated from
*different, more modern* populations — Sorcha with the Kurlander et al. (2025) inputs
and/or DP0.3, plus real NEOCP-style labeled candidates — deliberately measures how
calibration and ranking degrade under realistic prior mismatch, and provides the
operational metrics (completeness–contamination at triage thresholds, score distributions)
on realistic Rubin observables including cadence, trailing losses, and rate errors.
Explicitly a robustness test, not a correctness test: miscalibration here reflects prior
mismatch as much as method quality.

## 5. Roadmap

- **v0 — method bootstrap.** Ranging engine + Jacobian + `Digest2Histogram` for both
  components; delta-function attributable; HG $G=0.15$; All/Unk flag; $(\rho, \dot\rho)$
  posterior diagnostics; validation §4.1–4.2.
- **v1 — modern components.** `NEOMod3Grid` NEO side + 0.8×S3M-derived other side, with
  deliberate cross-normalization; attributable-covariance marginalization; empirical
  MPCORB-above-completeness belt density as an alternative module; stress-test validation
  §4.3.
- **v2 — full mixture & scale.** Parametric Trojans/Hildas/TNO components (with angular
  structure where it matters); SMB belt model when released; Centaur re-fit against the
  PS1 selection function; comet-scope decision; vectorization/throughput work for
  alert-scale operation.
