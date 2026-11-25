# The `pneo` command line

`pneo` has three subcommands:

| Command | What it does |
|---|---|
| `pneo score FILE ...` | Classify the tracklets in one or more observation files |
| `pneo fetch-model` | Download the MPC digest2 population model (~4 MB) |
| `pneo fetch-obscodes` | Download the MPC list of observatory codes |

`pneo score --help` is self-contained and lists every option. This page explains the
behavior behind them.

## Downloaded files

`pneo fetch-model` and `pneo fetch-obscodes` save their downloads in a per-user cache
directory, where `pneo score` finds them from any working directory:

| Platform | Directory |
|---|---|
| Linux, macOS | `~/.cache/pneo` (or `$XDG_CACHE_HOME/pneo` if set) |
| Windows | `%LOCALAPPDATA%\pneo\Cache` |

Set `$PNEO_CACHE_DIR` to use another directory. `pneo fetch-model --help` prints the
path in use.

## Input

`pneo score` reads **ADES PSV** or **MPC 80-column** astrometry, detected from the file
content (`--obs-format` forces one). Detections are grouped by object across all the
files given, so an object split over two files becomes one tracklet.

For each object, `pneo` uses the earliest same-night, same-station set of at least two
detections. A gap longer than `--night-gap-hours` (default 3) splits a night, and a
night longer than `--max-arc-hours` (default 2) is cut to its first two hours, because
the fit assumes linear motion over the tracklet. `pneo` fits the position and angular rates
at the first detection and converts the photometry to a single V magnitude.

**Photometry.** Magnitudes are converted to Johnson V with digest2's own table of band
offsets. There is one deliberate difference: an MPC 80-column record with a magnitude but
a blank band is treated as V, where digest2's 80-column reader applies its −0.8 mag
unknown-band correction, so `pneo` reads such a record 0.8 mag fainter than digest2 does.
A blank band in older astrometry usually means an unfiltered magnitude, and this keeps
`pneo`'s ADES and 80-column readers consistent with each other. The number of affected
records is reported on stderr.

## Output

One row per object, in the order the objects first appear in the input. The default
format is tab-separated with nine columns:

| Column | Meaning |
|---|---|
| `object_id` | Object identifier from the input |
| `n_obs` | Number of detections in the tracklet used |
| `arc_h` | Time spanned by the tracklet, in hours |
| `epoch` | Epoch of the fitted position and rates (UTC) |
| `obscode` | Observatory code |
| `V` | Apparent V magnitude used |
| `p_neo` | Probability that the object is an NEO |
| `log_lr` | Natural log of the NEO-to-other likelihood ratio |
| `status` | How the row was obtained (below) |

A field with no value is empty. `--format csv` writes the same columns comma-separated;
`--format json` writes one JSON object per line with every field, including the fitted
position and rates, both class likelihoods, the fit residual and where the observatory
location came from. `-o PATH` writes to a file instead of stdout.

**Status values.**

- `ok`: a probability computed from the population model.
- `fully_truncated_flagged`: every distance consistent with the tracklet implies an
  object fainter (smaller) than the population model covers, which happens only for very
  close, very small objects. `pneo` assigns `p_neo = 1` by rule and leaves `log_lr` empty.
  In simulations this rule fired only for NEOs.
- Anything else names why the object could not be scored, for example
  `single_detection`.

Calibration checks (comparing summed `p_neo` with observed NEO counts) should use only
`status == ok` rows.

**Failures are per object.** An object that cannot be read or classified produces one
row with a failure status and one line on stderr; the rest of the run continues. If a
batched call to the Rust kernel fails, its tracklets are re-scored one at a time.
Warnings, per-row notes and the run summary all go to stderr, so stdout carries only
data and `pneo score obs.psv > scores.tsv` is safe. `-q` suppresses the summary line.

## The population model

`pneo` looks for the model in this order: `--model PATH`, then `$PNEO_MODEL`, then the
[per-user cache](#downloaded-files) (where `pneo fetch-model` writes it).

The defaults are the configuration used in the paper:

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `All` | `All` is the full modeled population; `Unk` removes objects already in the MPC catalog. The paper scores NEO Confirmation Page candidates with `Unk` and samples of known objects with `All`. |
| `--h-extend-to` | `30` | The model ends at absolute magnitude H = 25.5; `pneo` extends its faintest bin to H = 30 so that very small, nearby NEOs can be scored. `none` disables the extension. |
| `--h-interpolation` | `none` | How the faint end is reconstructed: `none` uses digest2's piecewise-constant bins; `pchip` uses a smooth, count-conserving interpolation. |

## Numerical options

| Option | Default | Meaning |
|---|---|---|
| `--n-rho`, `--n-rhodot` | 1200, 50 | Grid points in distance and in range-rate. Almost all the quadrature error is in distance, hence the uneven grid. |
| `--rho-min`, `--rho-max` | 0.01, 100 AU | Range of distances considered |
| `-G` | 0.15 | Slope parameter of the H,G phase function |
| `--backend` | `auto` | `rust` for the compiled kernel, `python` for the numpy reference; `auto` uses the kernel when it is installed. `$PNEO_BACKEND` overrides the flag. |
| `--workers` | all cores | With the Rust kernel, the number of threads in one batched call; with the numpy reference, the number of worker processes. `1` runs in-process. |

## Observatory codes

`pneo` has a built-in table of 14 survey sites. Any other station needs the MPC list of
observatory codes: `pneo fetch-obscodes` downloads it to the [user cache](#downloaded-files), where
`pneo score` looks by default (`--obscodes PATH` or `$PNEO_OBSCODES` point elsewhere).
Scoring never touches the network.

An unknown code falls back to the center of the Earth, with a warning, and this matters:
it moves the observer by up to one Earth radius. In a test on 499 real NEO Confirmation
Page discovery tracklets, 106 used stations outside the built-in table; without the code
list their P(NEO) changed by up to 0.17 (median 8.5×10⁻⁴), while every correctly placed
tracklet was unchanged. The run summary warns when more than 10% of a run fell back.

## Speed

Scoring those 499 NEO Confirmation Page tracklets takes about 6 seconds on the default
grid with the Rust kernel on 8 threads. More generally, the kernel scores one tracklet in
about 12 ms on one core.
