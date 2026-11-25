# pneo

[![CI](https://github.com/lsst-ssp/pneo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lsst-ssp/pneo/actions/workflows/ci.yml)

A principled Bayesian classifier for asteroid tracklets: given a single tracklet — sky
position, rates of motion, and apparent magnitude — compute a calibrated posterior
probability that the object is a near-Earth object (NEO) rather than a member of any other
solar-system population. Built for rapid follow-up triage at Rubin/LSST scale. The name is the
quantity it computes, P(NEO); the project was previously called ds42.

**Method in one paragraph.** A tracklet determines an attributable
A = (α, δ, α̇, δ̇) and an apparent magnitude V, leaving exactly two unknowns: topocentric
range ρ and range-rate ρ̇. For each hypothesized (ρ, ρ̇) the full heliocentric orbit — and,
through the distance, the absolute magnitude H — is determined. pneo marginalizes over
the admissible region in (ρ, ρ̇), evaluating absolutely calibrated population densities
n_C(elements, H) for an NEO component and an everything-else component, with the proper
change-of-measure Jacobian between element space and observable space. The posterior
P(NEO) is the NEO component's share of the marginal likelihood; because both components are
absolutely calibrated, there is no tunable prior. This replaces digest2's heuristic score
with a calibrated probability, using — in v0 — digest2's own population model as input, so
the two codes differ in method alone.

**Status (2026-09): v0 complete.** The DESIGN §4.1 self-consistency gate **passes** at
N = 2×10⁴ and at N = 10⁵ (global odds scale ĉ = 1.03 [0.88, 1.20]; see
the paper), and the digest2 head-to-head
on identical inputs shows better ranking (premise-repaired full set:
ΔAUC +0.013 [+0.002, +0.025], P ≤ 0.01; independently corroborated at +0.014
[+0.007, +0.022] on the enriched set) and an order-of-magnitude calibration advantage —
at matched 88% completeness, 59.1% → 6.3% contamination
(the paper). On **real NEOCP
tracklets** (499 labeled discoveries) the calibration survives genuine prior mismatch:
predicted/observed NEO count 1.01 vs digest2's 1.38
(the paper).

```python
from astropy.time import Time
from pneo import Attributable, Digest2Histogram, classify
from pneo.paths import default_model_path

neo, other = Digest2Histogram.neo_other_pair(default_model_path())  # `pneo fetch-model` first
att = Attributable(ra=2.9, dec=0.1, ra_rate=-0.02, dec_rate=0.005,     # rad, rad/day
                   epoch=Time("2026-03-15T06:00:00"), v_mag=21.0)
res = classify(att, neo, other)
print(res.p_neo, res.log_lr)
res.plot_posterior_map()   # (rho, rho_dot) diagnostic; needs `pip install 'pneo[plot]'`
```

## Installation

```bash
pip install pneo            # wheels: linux x86_64, macOS arm64, Windows x64 (CPython >= 3.11)
pip install 'pneo[plot]'    # adds matplotlib, for the diagnostic plots
pneo fetch-model            # the MPC digest2 population model (~4 MB) -> user cache
pneo fetch-obscodes         # the MPC observatory-code table -> user cache
```

Both downloads go to a per-user cache directory: `~/.cache/pneo` on Linux and macOS
(or `$XDG_CACHE_HOME/pneo`), `%LOCALAPPDATA%\pneo\Cache` on Windows. Set
`$PNEO_CACHE_DIR` to use a different directory. `pneo score` finds the model there
from any working directory; `--model PATH` or `$PNEO_MODEL` override it.

Wheels ship the Rust-accelerated kernel (`backend="auto"` uses it automatically). On
platforms without a wheel, `pip install pneo` builds from the sdist as a pure-Python
package — same results via the numpy reference implementation, ~40× slower; build the
extension later with `rust/build.sh` (needs cargo). From a source checkout:
`pip install -e ".[dev]" && python scripts/fetch_digest2_model.py`. See
[RELEASING.md](RELEASING.md) for the release process.

## Command line

`pneo score` is the drop-in place to start if you already run digest2 on an
observation file. It ingests **ADES PSV** or **MPC 80-column** astrometry
(auto-detected from content), fits one attributable per object, and writes one row
per tracklet:

```console
$ pneo score neocp.psv
object_id  n_obs  arc_h   epoch                     obscode  V      p_neo     log_lr   status
T000001    4      0.7500  2026-03-15T06:00:00.000Z  X05      19.77  1         inf      ok
T000002    3      0.0667  2026-03-15T06:00:00.000Z  X05      22.62  1         nan      fully_truncated_flagged
T000003    5      1.3333  2026-03-15T06:00:00.000Z  F51      20.25  0.929963  2.58612  ok
T000004    1                                        X05                                single_detection
```

Columns are tab-separated (`--format csv|json`; JSON emits every field, one strict
RFC-8259 object per line) and a missing value is empty. `p_neo` is a **calibrated
posterior**, not a digest2 score — it can be thresholded against a contamination
budget. `status` says how each row was arrived at: `ok` is a likelihood-based
posterior, `fully_truncated_flagged` is the D-19 triage policy (`p_neo = 1.0`
*assigned*, `log_lr = nan`, because the whole admissible region is fainter than the
population model's H support), and an ingestion failure names its own cause. Any
calibration statistic must use `status == ok` only.

Photometry uses digest2's own band→V offset table, entry for entry, with **one
deliberate divergence** [D-20]: an MPC 80-column record carrying a magnitude but a
*blank* band is converted as V, where digest2's 80-column reader applies its −0.8
unknown-band fallback — so pneo reads 0.80 mag brighter there. A blank band in
pre-ADES astrometry is an unfiltered magnitude, and −0.8 is digest2's crudest
fallback, applied only because a different parser ran; pneo's ADES and
80-column paths agree with each other instead. Affected records are counted and
reported on stderr, so the divergence is never silent.

Scoring defaults are the D-18/D-19 operational configuration (`All` variant,
`h_extend_to = 30`, the 1200×50 grid, `backend=auto`); `--variant`,
`--h-extend-to`, `--h-interpolation`, `--n-rho`/`--n-rhodot`,
`--rho-min`/`--rho-max`, `--backend` and `-G` override each. `--workers` defaults
to all cores and maps onto whichever backend resolves: one batched
`classify_many(n_threads=N)` call on the Rust kernel, a multiprocessing pool over
tracklets on the numpy reference. Ingestion and scoring degrade **per tracklet** —
one bad object is one failure row plus a line on stderr, never a lost run (a
batched kernel call that fails as a unit is re-scored one tracklet at a time) — and
stdout carries nothing but data, so `pneo score obs.psv > scores.tsv` is clean.
`pneo fetch-model` is the same downloader as the `pneo-fetch-model` script.

**Observatory codes.** Anything outside the built-in 14-code table needs the MPC
observatory-code page; `pneo fetch-obscodes` downloads it to
the user cache, where `pneo score` looks by default (`--obscodes PATH`
or `$PNEO_OBSCODES` to point elsewhere). Scoring itself never touches the
network. An unknown code falls back to the **geocentre** with a warning, and that
is not cosmetic: scoring the pilot's 499 real tracklets with and without the table,
106 fell back and their P(NEO) moved by up to **0.169** (F52: 0.258 → 0.427; median
8.5×10⁻⁴), while every correctly-placed row was bit-identical. The run summary
shouts if more than 10% of a run fell back.

**Measured on the WP8 pilot's 499 real MPC NEOCP discovery tracklets** (with
`--obscodes`): 6.1 s wall on the default grid (Rust kernel, 8 threads), all 499
scored, and `p_neo` reproduces the pilot's own D-14-clamped scores with **median
|Δ| = 2×10⁻¹⁹ — i.e. exactly zero, 245 of 499 bit-identical — p75 4×10⁻¹⁰ and a
maximum of 8.6×10⁻⁶**, which is D-19's published extended-vs-clamped agreement,
reproduced end to end through the CLI.

`pneo score --help` is self-contained; the ingestion machinery is also a public
API (`pneo.ingest`).

## Documentation

| Document | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | Goals and scope, full mathematical formalism, architecture, validation plan, roadmap |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | v0 execution plan: work packages, agent team schedule, review protocol, acceptance gates |
| [DECISIONS.md](DECISIONS.md) | Dated log of design decisions with rationale and alternatives |
| [POPULATION_MODELS.md](POPULATION_MODELS.md) | Survey of available solar-system population models (through 2026-08) and open actions |
