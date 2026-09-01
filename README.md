# digest42

A principled Bayesian classifier for asteroid tracklets: given a single tracklet — sky
position, rates of motion, and apparent magnitude — compute a calibrated posterior
probability that the object is a near-Earth object (NEO) rather than a member of any other
solar-system population. Built for rapid follow-up triage at Rubin/LSST scale; named as the
answer to digest2.

**Method in one paragraph.** A tracklet determines an attributable
A = (α, δ, α̇, δ̇) and an apparent magnitude V, leaving exactly two unknowns: topocentric
range ρ and range-rate ρ̇. For each hypothesized (ρ, ρ̇) the full heliocentric orbit — and,
through the distance, the absolute magnitude H — is determined. digest42 marginalizes over
the admissible region in (ρ, ρ̇), evaluating absolutely calibrated population densities
n_C(elements, H) for an NEO component and an everything-else component, with the proper
change-of-measure Jacobian between element space and observable space. The posterior
P(NEO) is the NEO component's share of the marginal likelihood; because both components are
absolutely calibrated, there is no tunable prior. This replaces digest2's heuristic score
with a calibrated probability, using — in v0 — digest2's own population model as input, so
the two codes differ in method alone.

**Status (2026-09):** v0 implemented and validated — the DESIGN §4.1 self-consistency
gate **passes** (N = 2×10⁴: reliability, HPD uniformity, collapse coverage all within
criteria; see [validation/GATE_RESULTS.md](validation/GATE_RESULTS.md)). Remaining for
v0: the digest2 head-to-head comparison (WP7) and docs polish.

```python
from astropy.time import Time
from digest42 import Attributable, Digest2Histogram, classify

neo, other = Digest2Histogram.neo_other_pair("data/digest2.model.csv")  # scripts/fetch_digest2_model.py
att = Attributable(ra=2.9, dec=0.1, ra_rate=-0.02, dec_rate=0.005,     # rad, rad/day
                   epoch=Time("2026-03-15T06:00:00"), v_mag=21.0)
res = classify(att, neo, other)
print(res.p_neo, res.log_lr)
res.plot_posterior_map()   # (rho, rho_dot) diagnostic
```

## Documentation

| Document | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | Goals and scope, full mathematical formalism, architecture, validation plan, roadmap |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | v0 execution plan: work packages, agent team schedule, review protocol, acceptance gates |
| [DECISIONS.md](DECISIONS.md) | Dated log of design decisions with rationale and alternatives |
| [POPULATION_MODELS.md](POPULATION_MODELS.md) | Survey of available solar-system population models (through 2026-08) and open actions |
