# `pneo` — a Bayesian update to digest2

[![CI](https://github.com/lsst-ssp/pneo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lsst-ssp/pneo/actions/workflows/ci.yml)

`pneo` computes the probability that a newly detected moving object is a near-Earth object
(NEO), using only what a single night's tracklet measures: its sky position, angular
motion, and apparent magnitude. Its classifications are provably optimal for the adopted
population model, and its probabilities are calibrated.

Applied to present-day NEOCP candidates, its selection is 1.14x purer than digest2's. It
was shown to be 1.4–1.7x as pure on early Rubin data, and is expected to be ~2x as pure
at full LSST depth.

## Installation

```bash
pip install pneo            # CPython >= 3.11
pneo fetch-model            # once: download the MPC population model (~4 MB)
pneo fetch-obscodes         # once: download the MPC observatory-code table
```

Both are stored in a per-user cache; see
[docs/cli.md](https://github.com/lsst-ssp/pneo/blob/main/docs/cli.md#downloaded-files)
to change where.

Add the `plot` extra (`pip install 'pneo[plot]'`) for the diagnostic plots, which need
matplotlib.

Wheels for Linux (x86_64), macOS (Apple silicon) and Windows (x64) include a compiled
Rust kernel. On other platforms `pip` installs a pure-Python version that gives the same
results about 40 times more slowly.

## Quick start: command line

`pneo score` reads astrometry in ADES PSV or MPC 80-column format (detected
automatically), fits one tracklet per object, and writes one row per tracklet:

```console
$ pneo score neocp.psv
object_id  n_obs  arc_h   epoch                     obscode  V      p_neo     log_lr   status
T000001    4      0.7500  2026-03-15T06:00:00.000Z  X05      19.77  1         inf      ok
T000002    3      0.0667  2026-03-15T06:00:00.000Z  X05      22.62  1         nan      fully_truncated_flagged
T000003    5      1.3333  2026-03-15T06:00:00.000Z  F51      20.25  0.929963  2.58612  ok
T000004    1                                        X05                                single_detection
```

`p_neo` is the probability that the object is an NEO. `status` says how each row was
obtained: `ok` is a computed probability; `fully_truncated_flagged` marks a very faint,
nearby object beyond the population model's size range, assigned P(NEO) = 1 by rule; any
other value names why the object could not be scored. Use only `status == ok` rows when
checking calibration. Output is tab-separated by default (`--format csv` or `json` for
other formats). `pneo score --help` lists every option, and [docs/cli.md](https://github.com/lsst-ssp/pneo/blob/main/docs/cli.md)
describes them in more detail.

## Quick start: Python

```python
from astropy.time import Time
from pneo import Attributable, Digest2Histogram, classify
from pneo.paths import default_model_path

# The population model, as `pneo score` uses it (run `pneo fetch-model` first).
neo, other = Digest2Histogram.neo_other_pair(default_model_path(), h_extend_to=30.0)

att = Attributable(ra=2.9, dec=0.1, ra_rate=-0.02, dec_rate=0.005,  # rad, rad/day
                   epoch=Time("2026-03-15T06:00:00"), v_mag=21.0)
res = classify(att, neo, other)
print(res.p_neo)
res.plot_posterior_map()   # where the probability lies in distance and range-rate
```

`classify_many` scores a list of tracklets in one call, in parallel when the Rust kernel
is available.

## How it works

A tracklet measures an object's position and angular motion on the sky, but not its
distance or its velocity along the line of sight. `pneo` considers every distance and
radial velocity that gives a bound orbit around the Sun. Each such hypothesis implies a
full orbit and, from the apparent magnitude, the object's absolute magnitude (its size).
`pneo` weighs each hypothesis by how many objects with that orbit and size a population
model predicts, accounting for how a grid of distances and velocities maps onto orbits.
P(NEO) is the fraction of this weight that falls on NEO orbits (perihelion below
1.3 AU).

The population model is the one digest2 uses, maintained by the Minor Planet Center, so
differences from digest2 come from the method alone. It is a separate component and can
be replaced with a newer model without changing the classifier.

## Performance

The Rust kernel scores one tracklet in about 12 ms on a single CPU core, or about
1.5×10⁷ tracklets per hour on a 96-thread server.

## Development

```bash
git clone https://github.com/lsst-ssp/pneo && cd pneo
pip install -e ".[dev]"
python scripts/fetch_digest2_model.py   # the population model, into ./data/
rust/build.sh                           # optional: build the Rust kernel (needs cargo)
pytest
```

An editable install is pure Python until `rust/build.sh` compiles the kernel into the
source tree; rerun it after changing the Rust code. The `docs/` directory holds the
detailed [command-line notes](https://github.com/lsst-ssp/pneo/blob/main/docs/cli.md), the [release process](https://github.com/lsst-ssp/pneo/blob/main/docs/RELEASING.md),
and the [design records](https://github.com/lsst-ssp/pneo/blob/main/docs/design/README.md) from `pneo`'s development.

## Citing `pneo`

The method and the tests behind the numbers above are described in Singh, Jurić &
Ivezić (in preparation). If you use `pneo`, please cite that paper; see
[CITATION.cff](https://github.com/lsst-ssp/pneo/blob/main/CITATION.cff).

## License

BSD 3-Clause; see [LICENSE](https://github.com/lsst-ssp/pneo/blob/main/LICENSE).
