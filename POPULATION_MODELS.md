# Solar System Population Models — Research Summary

*Survey of available small-body population models for the digest42 Bayesian tracklet
classifier (NEO vs. everything else). Literature and data products through 2026-08-31.
Compiled from web/literature research on 2026-08-31.*

## Context: what we need

The classifier computes, for a single tracklet (attributable A = (α, δ, α̇, δ̇) + standardized
V mag), the posterior probability that the object belongs to the NEO population, by ranging
over topocentric distance and range-rate (ρ, ρ̇) and evaluating population densities at the
implied orbital elements E and absolute magnitude H(V, ρ). A population model is useful to us
if it provides — or lets us construct — a density n_C(E, H), ideally:

1. **Evaluable** (analytic/parametric or gridded), rather than a synthetic catalog we must
   density-estimate; catalog form is acceptable since we build a density-estimation layer anyway.
2. **Debiased/intrinsic** (not the raw discovered catalog) — required for correctness, since
   the survey's detection efficiency at fixed (A, V) cancels between mixture components only
   if the inputs are intrinsic populations.
3. **Absolutely calibrated** (objects per unit element-space volume per unit H) — with both
   components absolutely calibrated, no free prior-odds parameter is needed (decision already
   taken: no `w` knob; NEOMOD3 is natively calibrated, S3M-like inputs get the Wagg et al.
   MPCORB-anchored rescaling).
4. **H coverage to Rubin depth** — V_lim ≈ 24 (single visit); the reachable H at heliocentric
   distance r is H_max(r) ≈ V_lim − 5 log₁₀(r(r−1)), i.e. ≈26 at r = 1.3 AU, ≈23.8 at the belt's
   inner edge, ≈9 at 30 AU.

## Bottom line

- **There is no published, evaluable, integrated all-population successor to S3M** (as of
  2026-08). The per-population frontier has moved to evaluable models (NEOMOD3 for NEOs,
  OSSOS/CFEPS parametric TNO components, Vokrouhlický et al. parametric Trojans and Hildas),
  but nobody has stitched them into one evaluable all-sky model.
- **The main belt — the population that dominates NEO confusion — still has no published
  replacement for S3M.** State-of-the-art usage is 0.8 × S3M (Wagg et al. rescaling).
- **An S3M successor for the belt is in progress**: the "Synthetic Main Belt" (SMB) by
  DeMartini, Dorsey & Granvik (Helsinki), presented at ACM 2026 in Poznań. Conference-stage;
  no paper or data release yet. Catalog form. See §10.
- The only existing evaluable "everything" density is **digest2's (q, e, i, H) histogram**
  (MPC-maintained) — a form precedent, but its content is unrescaled 2011 S3M.
- **No published KDE/normalizing-flow/learned-density representation of small-body populations
  exists** (through 2026-08). Our density layer over modern per-population inputs would be novel.

---

## 1. NEO component (settled): NEOMOD3

- **Reference:** Nesvorný et al. 2024 ([arXiv:2404.18805](https://arxiv.org/abs/2404.18805));
  lineage: NEOMOD (Nesvorný et al. 2023, Icarus 411, 115922), NEOMOD2, NEOMOD3.
- **Form:** calibrated model grid in (a, e, i, H) + simulator code. Angles (Ω, ω, M) treated
  as uniform. Absolutely calibrated. Extends to H ≈ 28 (NEOMOD3 is the small-NEO extension).
- **Availability:** SwRI download area,
  [www2.boulder.swri.edu/~davidn/NEOMOD_Simulator/](https://www2.boulder.swri.edu/~davidn/NEOMOD_Simulator/)
  (hosts NEOMOD/NEOMOD2/NEOMOD3 + `diamod`/`magmod` variants; server TLS certificate broken).
- **Design decision already taken:** NEO mixture component = NEOMOD3 on (a, e, i, H) with
  analytic uniform angle factors. "NEO" in the posterior = population-membership (posterior
  weight of the NEOMOD3 component), so the non-NEO component must be cleanly complementary
  at q = 1.3 AU (verify per input catalog — see Open Actions).
- Kurlander et al. 2025 note S3M's NEO component is obsolete vs. NEOMOD3 (e.g., no H > 25
  objects in S3M).

## 2. Integrated all-population models

### 2.1 S3M — the 2011 baseline
- **Reference:** Grav, Jedicke, Denneau et al. 2011, PASP 123, 423
  ([doi:10.1086/659833](https://iopscience.iop.org/article/10.1086/659833)).
- **Form:** synthetic catalog (6D orbits + H_V), ~14M objects; full solar system. Built by
  cloning/extrapolating the circa-2011 known population (~500k objects; 3× fewer than known
  today).
- **Known deficiencies:** calibrated to a bright/small sample for a shallower survey than
  LSST; **overpredicts counts by ~25% at m_V ≈ 17–20.5** (Wagg et al.); MBA component's H
  truncation near H ≈ 24.5–25 (verify exact value per file); no dedicated debiasing of
  Hungarias/Phocaeas/Mars-crossers (represented only via 2011 known-catalog cloning);
  NEO component superseded by NEOMOD3. Exact q boundary between S3M's NEO and MBA components
  not re-verified — check before excising q < 1.3.

### 2.2 Wagg et al. rescaling + hybrid catalog (state-of-the-art S3M usage)
- **Reference:** Wagg, Jurić, Yoachim et al. 2025, AJ 169, 315
  ([arXiv:2408.12517](https://arxiv.org/abs/2408.12517)).
- S3M vs MPCORB at m_V = 17–20.5 → **adopt f = 0.80 uniform retention ("80%-scale S3M")**.
  Hybrid catalog: k-d-tree match of each MPCORB object to nearest S3M object in
  (position, velocity, H) within H bins; matched synthetic objects replaced by real ones.
- **Form:** synthetic catalog + open-source pipeline
  ([github.com/dirac-institute/hybrid_sso_catalogue](https://github.com/dirac-institute/hybrid_sso_catalogue)).
  No pre-built data release found — regenerate from current MPCORB + S3M (unverified whether
  a frozen product exists elsewhere).
- The recalibration is a **single global scale factor**, not a per-population re-fit.

### 2.3 Kurlander et al. 2025 LSST-yield input ensemble (most current curated set)
- **Reference:** Kurlander, Wagg et al. 2025, AJ 170, 99
  ([arXiv:2506.02487](https://arxiv.org/abs/2506.02487)); part of the PSJ/AJ
  [Rubin LSST Solar System Predictions focus issue](https://iopscience.iop.org/collections/PSJ-250528-01).
- Per-population Sorcha inputs: **NEOs = NEOMOD3 derivative** (6.1M objects, d ≥ 10 m);
  **MBAs = 0.8 × S3M** (11.1M objects, H_r < 26.4); **Trojans** = Vokrouhlický et al. 2024
  model extrapolated H_V 15 → 19 (L4 faint-end slope, L4/L5 separate); **TNOs = CFEPS-L7**
  orbits + updated SFDs/color fractions, 9 dynamical subpopulations (to H_r < 16.38).
  Colors by reference spectral type; HG phase with G = 0.15.
- **Availability:** all input orbit + physical-parameter catalogs + Sorcha configs at CANFAR,
  [DOI 25.0062](https://www.canfar.net/citation/landing?doi=25.0062) (landing page login-gated;
  contents/format unverified programmatically — but we have data rights and coauthorship).
- Companion papers: Centaurs / Neptune Trojans (Murtagh et al.,
  [arXiv:2506.02779](https://arxiv.org/pdf/2506.02779)); imminent impactors (Chow, Cheng);
  ISOs (Dorsey et al.).

### 2.4 Rubin DP0.3 simulation inputs
- MPCORB (2023-05-01) + 91% of S3M merged via Wagg's hybridcat, + 12,148 ISOs + 2×5,000 LPCs
  ([DP0.3 simulation description](https://dp0-3.lsst.io/data-products-dp0-3/data-simulation-dp0-3.html)).
- Queryable as TAP tables (`dp03_catalogs_1yr/10yr`) on the Rubin Science Platform (data
  rights required). Form: catalog, frozen at 2023 MPCORB.

### 2.5 NEO Surveyor Reference Small Body Population Model (RSBPM)
- **References:** Mainzer et al. 2023, PSJ 4, 224 ([arXiv:2310.12918](https://arxiv.org/pdf/2310.12918), §5.1);
  Lilly et al. LPSC 2021 abstract [2584](https://www.hou.usra.edu/meetings/lpsc2021/pdf/2584.pdf);
  known-object completeness model: Grav et al. 2023, PSJ 4, 228
  ([arXiv:2310.20149](https://arxiv.org/abs/2310.20149)).
- **Form:** synthetic catalog from semi-evaluable ingredients — analytic triple-power-law SFDs
  (NEA slopes −2.75/−1.64/−3.2, breaks at 1.5 km and 70 m; MBA power laws fit to NEOWISE);
  **orbital elements from Gaussian KDEs over the observationally complete known population**
  (NEOs H < 20; MBAs H < 17.75/17.25/16.75 inner/middle/outer); NEOWISE albedo/beaming KDEs.
  MBA H-distribution is S3M's — not independent.
- **Not public** (internal mission yardstick; no data release found). KDE-of-knowns assumes
  undiscovered orbital structure follows the discovered bright population — a bias for our
  use case.
- **No "S3M v2" from Pan-STARRS exists**; 2025–26 papers still cite and rescale 2011 S3M.

### 2.6 digest2 population model — the only evaluable "everything" density
- **References:** Keys, Vereš, Payne et al. 2019, PASP 131, 064501
  ([arXiv:1904.09188](https://arxiv.org/pdf/1904.09188)); ADES update: Vereš, Payne, Cloete
  et al. 2023, PASP ([arXiv:2309.16407](https://arxiv.org/pdf/2309.16407)).
- **Form:** binned 4D histogram over **(q, e, i, H)** — 29 q-bins (0–100 AU), 8 e-bins (0–1.1),
  11 i-bins (0–180°), 18 H-bins (H < 6 to 25.5); 45,936 non-uniform bins; 15 orbit classes
  (NEO, Hungaria, Phocaea, MC, MBA zones, Hilda, Trojan, … — exact class list unverified).
  Variants: "All" (total, S3M-derived) and "Unk" (undiscovered = total − known, re-anchored
  to the 2023 known catalog). Angles implicitly uniform.
- **Availability:** actively MPC-maintained. `digest2.model.csv` + pure-Python reader
  (`digest2.population`); build pipeline documented in the
  [MPC tutorial](https://docs.minorplanetcenter.net/tutorials/notebooks/mpc_tutorial_digest2_population_model/).
  Repo migrated (per MPEC 2026-E23, March 2026) to
  [github.com/Smithsonian/mpc-public](https://github.com/Smithsonian/mpc-public/tree/main/digest2)
  (pip-installable).
- **Relevance:** the closest existing artifact to what we're building — a Bayesian tracklet
  prior over (q, e, i, H) with an "undiscovered" variant. But: content is unrescaled 2011 S3M
  (the ~25% excess presumably propagates), coarse bins, no correlations beyond 4D, heuristic
  (no Jacobian/measure transformation). It is a repackaging of S3M, not a re-derivation.

## 3. Main asteroid belt (non-NEO inner populations)

### 3.1 Current state
**No published NEOMOD-analog for the main belt exists (2026-08).** Verified by checking
Nesvorný's and Vokrouhlický's complete arXiv records through Aug 2026, the SwRI model server
(probes for Trojan/Hilda/main-belt simulator directories 404), and the Hilda paper's
future-work text (discusses Trojan/Hilda collisional modeling, not the belt). The NEOMOD
team's trajectory is NEOs → Trojans (2024) → Hildas (2025); a belt model is plausible but
**not published or announced**. Nesvorný's 2026 belt papers are a family census
(arXiv:2602.20382) and family analyses, not debiased population models.

State-of-the-art usage remains **0.8 × S3M** (Kurlander et al. 2025: 11.1M MBAs to H_r < 26.4).

### 3.2 SMB — the in-progress successor (ACM 2026) → see §10 for full details

### 3.3 Empirical route (viable DIY alternative)
- **Hendler & Malhotra 2020**, PSJ 1, 75 ([arXiv:2010.07822](https://arxiv.org/abs/2010.07822)):
  observational completeness limit H_lim(a) at high resolution in a, belt through Trojans;
  the known sample above it is complete/unbiased (2× larger sample than a single conservative cut).
- Combine MPCORB (~1.4M known MBAs vs the ~0.5M underlying S3M) above H_lim(a) with
  Nesvorný et al. 2024 proper elements for 1.25M MBAs
  ([arXiv:2407.18221](https://arxiv.org/abs/2407.18221)); build KDE/grid in (a, e, i, H),
  angles uniform; extrapolate faint bins with per-zone SFD slopes.
- Recent SFD constraints for the faint extrapolation: Vávra & Brož 2026, A&A 707, A367
  ([arXiv:2602.19656](https://arxiv.org/abs/2602.19656)) — observed SFDs of >1M asteroids by
  region and C/S taxonomy + calibrated collisional model; JWST decameter MBA counts
  (Burdanov et al. 2025, Nature, [arXiv:2502.01744](https://arxiv.org/abs/2502.01744)) —
  shallow-to-steep SFD break near ~100 m.
- Note the belt is complete to H ≈ 17–18 (inner belt); for the classifier the belt's reachable
  H at Rubin depth is only ≈23.8 anyway (V_lim = 24, r ≥ 1.8 AU), so the extrapolation span
  is modest.

### 3.4 Legacy models (reference only; superseded)
- **Jedicke & Metcalfe 1998**, Icarus 131, 245
  ([astro-ph/9801023](https://arxiv.org/abs/astro-ph/9801023)): classic debiased 4D (a, e, i, H)
  MBA distribution from Spacewatch — conceptually exactly the right form, 28 years old.
- **Statistical Asteroid Model (SAM)**: Tedesco, Cellino, Zappalà 2005, AJ 129, 2869 —
  1.88M MBAs D > 1 km with orbits, H, albedos; public at [PDS SBN](https://sbn.psi.edu/pds/SAM-I/).
- **Ivezić et al. 2001**, AJ 122, 2749 (SDSS): analytic broken-power-law SFD; a calibration
  input to S3M.

## 4. Hungarias, Phocaeas, Mars-crossers

**No dedicated debiased population model exists in the 2023–2026 literature.** They appear
only as: (a) NEO *source regions* in Granvik et al. 2018
([Icarus 312, 181](https://www.sciencedirect.com/science/article/pii/S0019103517307017)) and
NEOMOD — which model the resulting NEOs, not the source populations; (b) implicit
subpopulations of S3M / digest2 classes; (c) dynamical studies (e.g. A&A 692, 2024, on
Mars-crosser origins — dynamics, not a density). Practical options for 1.3 < q < 1.66:
0.8 × S3M, the digest2 grid, or an empirical KDE of the knowns (deeply complete there —
bright, close populations).

**Classifier-critical note:** the Mars-crosser zone q ∈ (1.3, ~1.5) AU is where the
hard-H-truncation issue lives. At V = 24, ranging nodes at r ≈ 1.3 AU imply H ≈ 26 — beyond
S3M's MBA truncation — so non-NEO orbits there get zeroed by model boundary, not physics,
inflating P(NEO) exactly in the borderline triage regime. (For the belt proper, r ≥ 1.8 AU
implies H ≤ 23.8 at V = 24, so truncation never binds — verified reasoning, 2026-08-31
discussion.) Validation must flag posterior mass in the {q > 1.3, H > H_max(catalog)} corner.

## 5. Jupiter Trojans — evaluable, state of the art

- **Vokrouhlický, Nesvorný & Bottke 2024**, AJ 167, 138
  ([arXiv:2401.15537](https://arxiv.org/abs/2401.15537)).
- **Form: fully parametric/analytic** — separable generalized-exponential distributions in
  proper inclination, proper eccentricity, and **libration amplitude** (resonant angle handled
  explicitly); H distribution as a 6-segment cubic spline. Background L4/L5 modeled separately
  (families excised); N4 ≈ 2040, N5 ≈ 1410 for H ≤ 13.8; N4/N5 = 1.45 ± 0.05 (H < 15).
- Debiased against Catalina Sky Survey G96 (decade). Valid 7 < H < 15 (tested to 15.5);
  Rubin reaches fainter — Kurlander et al. 2025 extrapolate to H_V = 19 with the L4 faint slope.
- **Availability:** parametric model fully specified in the paper; proper-elements catalog at
  [sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/](https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/).
  No public simulator.

## 6. Hildas — evaluable, state of the art

- **Vokrouhlický, Nesvorný, Brož, Bottke, Deienno, Fuls, Shelly 2025**, AJ 169, 242
  ([arXiv:2503.04403](https://arxiv.org/abs/2503.04403)).
- Same formalism as Trojans (CSS G96 debiasing): background + collisional-family components
  as products of 1D distribution functions in resonant/proper elements; spline H distribution
  over 7.5 < H < 18, valid to H ≤ 16; absolute numbers (families ≈ 50% of H ≤ 16 Hildas).
- **Availability:** parameters in the paper; proper-elements + family catalog at
  [Zenodo 10.5281/zenodo.14959239](https://doi.org/10.5281/zenodo.14959239). No public simulator.

## 7. Trans-Neptunian objects

### 7.1 CFEPS-L7 (incumbent baseline)
- Petit et al. 2011, AJ 142, 131 ([arXiv:1108.4836](https://arxiv.org/pdf/1108.4836));
  resonant: Gladman et al. 2012, AJ 144, 23.
- **Form: hybrid** — parametric generator (GiMeObj: a, e, i, q, H per component, explicit
  libration amplitude/phase for 11 Neptune MMRs) + released synthetic realization
  (~51k classical, 13.5k resonant, 1.6k scattering). Debiased against characterized CFEPS.
- **Availability:** [cfeps.net](http://www.cfeps.net) and
  [github.com/OSSOS/SurveySimulator](https://github.com/OSSOS/SurveySimulator)
  (parametric + lookup modes). 2011-era statistics; SFDs superseded (below).

### 7.2 OSSOS parametric sub-population models (published state of the art, piecewise)
Methodology: Lawler et al. 2018, OSSOS X ([arXiv:1802.00460](https://arxiv.org/abs/1802.00460)).
Absolutely calibrated parametric models per dynamical class:
- **Scattering disk:** Shankman et al. 2016 ([arXiv:1511.02896](https://arxiv.org/abs/1511.02896));
  Lawler et al. 2019, OSSOS XV ([arXiv:1905.09286](https://arxiv.org/pdf/1905.09286)) —
  orbital model + divot/knee H_r distributions.
- **Distant resonances (beyond 2:1):** Crompvoets et al. 2022, PSJ 3, 113
  ([arXiv:2204.09139](https://arxiv.org/abs/2204.09139)) — per-resonance parametric models
  incl. libration parameters; ≥10⁵ objects H_r < 8.66 (factor ~4 uncertainty).
- **Detached:** Beaudoin et al. 2023, PSJ 4, 145
  ([arXiv:2306.12847](https://arxiv.org/abs/2306.12847)) — parametric q distribution;
  ~36–48k detached TNOs D ≳ 100 km.
- **SFDs at Rubin depth:** cold classicals = exponentially tapered power law
  (Kavelaars et al. 2021, ApJL 920, L28, [arXiv:2107.06120](https://arxiv.org/abs/2107.06120));
  hot population (Petit et al. 2023, ApJL, [arXiv:2309.01530](https://arxiv.org/abs/2309.01530));
  DEEP V (Napier et al. 2024, PSJ, [arXiv:2309.09478](https://arxiv.org/pdf/2309.09478))
  measures cold-classical H to r ~ 26.2, consistent with the taper. **Do not use S3M's single
  power laws for TNOs at V ~ 24.**

### 7.3 "OSSOS++" unified Kuiper belt model — NOT YET PUBLISHED
- Petit, Gladman, Kavelaars, Volk et al.; [DPS 55 abstract (2023)](https://baas.aas.org/pub/2023n8i209p04/release/1);
  CFHT Users' Meeting 2025 abstract. All components in barycentric free-inclination space.
- **No refereed paper found through 2026-08** (verified by repeated searches). Would be the
  ideal single TNO component — watch for it / contact J.-M. Petit.

### 7.4 DES
- Bernardinelli et al. 2022 ApJS (~817 TNOs + selection function; Y4:
  [arXiv:1909.01478](https://arxiv.org/abs/1909.01478)); H/color analysis
  [2025, AJ 169, 305](https://iopscience.iop.org/article/10.3847/1538-3881/adc459).
- **Form: characterized catalog + selection function, not an intrinsic model** — useful for
  validating the TNO component off-ecliptic, not as the mixture density.

## 8. Centaurs

- **Nesvorný et al. 2019 (OSSOS XIX)**, AJ 158
  ([IOP](https://iopscience.iop.org/article/10.3847/1538-3881/ab3651)): N17-based steady-state
  residence-time model rescaled to 15 OSSOS detections — **21,000 ± 8,000 Centaurs H_r < 13.7**
  (a < 30 AU, q > 7.5 AU). Form: synthetic + parametric SFD (knee at H_r = 7.7; slopes 0.9/0.4).
- **Kurlander et al. 2024/2025 (PS1)**, AJ ([arXiv:2412.01687](https://arxiv.org/abs/2412.01687);
  note: mjuric coauthor): 21,400 (+3,400/−2,800) H_r < 13.7 — consistent with OSSOS — but
  **rejects the joint N17/N19 orbit+H distribution at p = 0.0036** (overpredicts a < 18 AU
  Centaurs — the fastest-moving, most NEO-confusable outer objects). Selection function +
  notebooks: [Zenodo 10.5281/zenodo.14201491](https://doi.org/10.5281/zenodo.14201491) —
  usable to re-fit a Centaur density against PS1.
- LSST Centaur yield paper ([arXiv:2506.02779](https://arxiv.org/pdf/2506.02779)) provides a
  ready-made Rubin-depth synthetic Centaur catalog (N-body output to H_r = 20, rescaled to
  N(H_r < 13.7) = 21,400).

## 9. Comets and ISOs

### 9.1 Jupiter-family / ecliptic comets
- **Nesvorný et al. 2017 ("N17")**, ApJ 845, 27 ([arXiv:1706.07447](https://arxiv.org/abs/1706.07447)):
  end-to-end 4.5-Gyr simulation; steady-state (a, e, i) of ecliptic comets/JFCs with fading
  (N_p ≈ 500 perihelion passages at q < 2.5 AU); covers Halley-types. Absolutely calibrated to
  active JFC counts + OSSOS Centaurs. Form: synthetic (residence-time catalog); H via assumed
  SFD, not fitted. **No public archive found** (likely available from D. Nesvorný on request —
  unverified). NEOMOD3 already contains the JFC-sourced NEO component at q < 1.3, so a JFC
  module only needs q ≳ 1.3.

### 9.2 Long-period comets
- **Vokrouhlický, Nesvorný & Dones 2019**, AJ 157, 181
  ([arXiv:1904.00728](https://arxiv.org/abs/1904.00728)): Oort-cloud formation + tides +
  stellar passages; Whipple fading κ = 0.6; analytic fits to cumulative q distribution;
  absolutely calibrated to observed LPC flux. No public data product found (unverified).
- **Boe et al. 2019**, Icarus 333, 252 ([arXiv:1905.13458](https://arxiv.org/abs/1905.13458)):
  debiased LPC SFD from PS1 (α_big = 0.72, α_small = 0.07; 0.46 ± 0.15 × 10⁹ LPCs D > 1 km,
  q < 10 AU) + activity/brightening model.
- Older evaluable alternative: Silsbee & Tremaine 2016, AJ 152, 103 (nearly isotropic comets
  for LSST; attribution from memory — verify).
- **Structural caveat:** comet apparent magnitude is activity-dominated; a static-H density
  (our formalism) is wrong for active comets. Decide whether to (a) exclude active comets as
  out-of-scope, (b) add an activity-aware magnitude model, or (c) accept the bias and document.

### 9.3 ISOs
- **Ōtautahi–Oxford model** (Hopkins et al. 2025; LSST visibility: Dorsey et al. 2025, PSJ 6,
  214, [arXiv:2502.16741](https://arxiv.org/abs/2502.16741)): genuinely probabilistic/evaluable
  (Gaia kinematics + disk chemistry → joint velocity/composition distribution). Code
  availability unverified. **Currently out of scope** — the classifier assumes bound orbits
  (admissible-region pruning); recorded here in case an ISO class is added later (would change
  the ranging domain).

## 10. ACM 2026 (Poznań, July 6–10): the S3M successor talk

**"A New Synthetic Model of the Asteroid Belt" — Joseph V. DeMartini, Rosemary Dorsey,
Mikael Granvik** (Univ. Helsinki / Luleå), oral O 34, Mon July 6, 15:00, Asteroid Science
session. Sources:
[program PDF](https://acm2026.eu/conf-data/ACM2026/files/Program/ACM2026_program_new_final3.pdf),
[Book of Abstracts](https://share.ppnt.poznan.pl/index.php/s/QNCyAPcAnpdMLDK) (pp. 70, 121).

From the abstract — the **"Synthetic Main Belt" (SMB)** model:
- Motivation: S3M outdated (built ~15 yr ago from ~500k objects, 3× fewer than known);
  overpredicts bright 10 ≲ H ≲ 16 objects (cites Wagg et al. 2025); LSST-testable model needed.
- Method: compile/extend H-frequency distributions with extrapolation accounting for
  **collisional family membership, dynamically distinct belt regions, and updated
  observational + dynamical completeness limits** (Hendler & Malhotra 2020); orbital elements
  drawn from physically motivated distributions per dynamical subgroup.
- Claims: better match to observed distribution at H > 10 than S3M; reproduces predicted HFD
  flattening beyond the completeness limit (Moskovitz et al. 2008); more accurate orbital
  element distributions in the complete regime.
- **Form: synthetic catalog** (same form as S3M). **Status: conference-stage; no arXiv paper
  or data release found as of 2026-08-31.**
- Granvik's Tuesday plenary ("Models of the NEO and MBA Populations: from Sky Surveys to
  Geophysics", O 85) cites SMB as "this meeting" — flagship project of the Granvik group,
  so likely to be built consistent with NEOMOD-family products (Granvik is a NEO-model author).

## 11. ML / learned-density representations

**Nothing published through 2026-08** as a normalizing-flow or neural-density representation
of small-body populations. Closest items: RSBPM's internal Gaussian KDEs (§2.5); NEOMOD's
grid+simulator form; and ML *classifiers* consuming population-model outputs — direct
precedents/competitors for digest42:
- Vereš, Cloete, Payne & Loeb 2025, A&A 698, A242 ([arXiv:2505.11910](https://arxiv.org/abs/2505.11910)):
  GBM/RF on 30 digest2 parameter categories for NEOCP candidates 2019–2024.
  *(author order corrected 2026-09-02 against ADS, 2025A&A...698A.242V)*
- Cloete, Vereš & Loeb 2024, A&A 691, A338 ([arXiv:2412.02112](https://arxiv.org/pdf/2412.02112)): tracklet-level
  ISO classification for LSST, trained on DP0.3; digest2 scores dominate feature importance.

## 12. Recommended composition for digest42

| Mixture piece | Launch choice | Upgrade path |
|---|---|---|
| NEO | NEOMOD3 grid (a, e, i, H), uniform angles | — |
| Main belt + MC/Hungaria/Phocaea | 0.8 × S3M (or Kurlander CANFAR MBA set), q > 1.3 excised | **SMB (DeMartini et al.) when released**; or empirical MPCORB + H_lim(a) density |
| Jupiter Trojans | Vokrouhlický et al. 2024 parametric, evaluated directly | extrapolate H per Kurlander et al. 2025 |
| Hildas | Vokrouhlický et al. 2025 parametric, evaluated directly | — |
| TNOs | CFEPS-L7 / Kurlander 9-subpopulation set with tapered/knee SFDs | OSSOS++ when published |
| Centaurs | N19 catalog rescaled (21,400 at H_r < 13.7) | re-fit vs PS1 selection function (Zenodo) given the a < 18 AU tension |
| Comets | decide scope (see §9.1–9.2 caveat) | — |
| ISOs | out of scope (bound orbits only) | separate class + hyperbolic ranging domain |

Since digest42 builds its own density-estimation layer, catalog-form inputs are not a blocker:
each population is a pluggable density module (parametric ones evaluated directly, catalog
ones density-estimated). Inner populations dominate NEO confusion → the belt module matters
most; the SMB timeline question is worth resolving early.

## 13. Open actions

1. **Email DeMartini/Granvik** re SMB timeline, data access, and whether it will ship with a
   q < 1.3 exclusion consistent with NEOMOD3 (membership-based NEO definition requires clean
   complementarity at q = 1.3).
2. Verify S3M file specifics before use: exact NEO/MBA q boundary, MBA H_max, Mars-crosser
   coverage in q ∈ (1.3, 1.66).
3. Enumerate the CANFAR DOI 25.0062 holdings (login-gated; we have access).
4. Watch for: OSSOS++ paper (Petit et al.), SMB paper/release, NEO Surveyor RSBPM release paper.
5. Validation: flag tracklets with posterior mass in {q > 1.3, H > H_max(non-NEO model)} —
   the H-truncation artifact corner (Mars-crosser zone, V ≳ 23.5, small-ρ nodes).
6. Decide comet scope (active-comet magnitudes break the static-H formalism).

## Explicitly unverified items

- OSSOS++ publication status (no refereed paper found through 2026-08).
- Public availability of N17 JFC and Vokrouhlický 2019 LPC model outputs.
- Silsbee & Tremaine 2016 attribution.
- CANFAR DOI 25.0062 contents (login-gated).
- Existence of a frozen Wagg hybrid-catalog data release (code is public; product may be
  regenerate-only).
- digest2's exact 15-class list; S3M's exact q boundary and per-component H truncations.
- RSBPM status beyond the 2021 LPSC abstract and Mainzer et al. 2023 description.
