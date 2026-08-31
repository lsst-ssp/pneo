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
($H \le 25.5$), making the truncation symmetric. Validation must flag tracklets with
posterior mass in $\{q > 1.3,\, H > H_{\max}^{\rm model}\}$ (§4).

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

1. **Head-to-head vs digest2 on identical inputs.** Same population file, same tracklets;
   compare ranking (ROC) and calibration. This isolates the value of the Jacobian +
   marginalization. Test sets: simulated Rubin tracklets (Sorcha with the Kurlander et al.
   2025 input populations and/or DP0.3), plus real NEOCP-style labeled candidates.
2. **Calibration curves.** Among tracklets scored $P(\mathrm{NEO}) \approx p$, the true NEO
   fraction must be $\approx p$ (reliability diagram). This is the claim digest2 cannot
   make and the headline plot.
3. **Completeness–contamination at triage thresholds** as the operational metric for
   follow-up targeting.
4. **Truncation-corner flag** (§2.6): report the fraction of tracklets with significant
   posterior mass at $\{q > 1.3,\, H > H_{\max}^{\rm model}\}$; expected rare (faintest
   ~1.5 mag of the tracklet population), verify empirically.
5. **Collapse tests.** For objects with known orbits, the $(\rho, \dot\rho)$ posterior maps
   must concentrate around the true (ρ, ρ̇); the diagnostic maps are a first-class output.

## 5. Roadmap

- **v0 — method bootstrap.** Ranging engine + Jacobian + `Digest2Histogram` for both
  components; delta-function attributable; HG $G=0.15$; All/Unk flag; $(\rho, \dot\rho)$
  posterior diagnostics; validation items 1, 2, 5.
- **v1 — modern components.** `NEOMod3Grid` NEO side + 0.8×S3M-derived other side, with
  deliberate cross-normalization; attributable-covariance marginalization; empirical
  MPCORB-above-completeness belt density as an alternative module.
- **v2 — full mixture & scale.** Parametric Trojans/Hildas/TNO components (with angular
  structure where it matters); SMB belt model when released; Centaur re-fit against the
  PS1 selection function; comet-scope decision; vectorization/throughput work for
  alert-scale operation.
