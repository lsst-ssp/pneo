# The `pneo` Python API

Everything is available from the top-level package:

```python
import pneo
```

| Function | What it does |
|---|---|
| `pneo.classify(...)` | P(NEO) for one or many tracklets |
| `pneo.read_observations(paths)` | Read ADES PSV or MPC 80-column astrometry into tracklets |
| `pneo.load_model(...)` | Load a population model |
| `pneo.posterior_map(...)` | Where the probability lies in distance and range-rate, for one tracklet |
| `pneo.plot_posterior_map(m)` | Plot a `posterior_map` result (needs `pneo[plot]`) |
| `pneo.fetch()` | Download the population model and the MPC observatory codes |

Inputs and outputs are numpy arrays (or scalars). **Units are degrees and degrees per
day**; `ra_rate` is the **on-sky** rate in right ascension, d(α)/dt·cos δ. Epochs are MJD
(UTC) or `astropy.time.Time` objects, and magnitudes are Johnson V.

## `classify`

```python
pneo.classify(ra, dec, ra_rate, dec_rate, epoch, mag, *,
              model=None, obscode=None, observer=None, obscodes=None,
              details=False, backend="auto", workers=None,
              n_rho=1200, n_rhodot=50, rho_min=0.01, rho_max=100.0, G=None)
```

Returns P(NEO) with the broadcast shape of the inputs, or a float for scalar inputs.

- **`model`**: a `PopulationModel` from `load_model()`. By default the default model is
  loaded from the cache, once per process.
- **`obscode`**: an MPC observatory code, or an array of codes, one per tracklet. Codes are
  looked up in the built-in table of survey sites, then in the MPC list that `pneo fetch`
  downloads (or the file given as `obscodes`). The default is the center of the Earth;
  an unknown code also falls back to it, with a warning.
- **`observer`**: instead of `obscode`, the observer's heliocentric position and velocity
  in the equatorial (ICRS) frame, `(position [AU], velocity [AU/day])`, each of shape
  `(3,)` or `(n, 3)`. Use this for an observer without a code, such as a spacecraft.
- **`details=True`**: return a numpy structured array with fields `p_neo`, `log_lr` (the
  natural log of the NEO-to-other likelihood ratio), `lambda_neo`, `lambda_other` (the two
  class likelihoods) and `status`.
- **`backend`**: `"rust"` for the compiled kernel, `"python"` for the numpy reference;
  `"auto"` uses the kernel when it is installed. `$PNEO_BACKEND` overrides it.
- **`workers`**: threads for the compiled kernel (default: all cores).
- **`n_rho`, `n_rhodot`, `rho_min`, `rho_max`, `G`**: the integration grid, the range of
  distances [AU] and the H,G phase slope (default 0.15). The defaults are the paper's
  configuration.

**Status values.** `ok` is a computed probability. `beyond_model` means every orbit
consistent with the tracklet implies an object smaller than the model covers (a very
close, very small object); P(NEO) is set to 1 by rule. `no_support` means neither class
has objects consistent with the tracklet, and `empty_region` that no bound orbit fits;
P(NEO) is NaN for both. Use only `ok` rows when checking calibration.

## `read_observations`

```python
tracklets, failures = pneo.read_observations(paths, *, obscodes=None, format="auto",
                                             night_gap_hours=3.0, max_arc_hours=2.0)
```

Reads one or more ADES PSV or MPC 80-column files and fits one tracklet per object (the
rules are described in [cli.md](cli.md#input)). Returns two dicts of numpy arrays:

- **`tracklets`**: `ra`, `dec`, `ra_rate`, `dec_rate`, `epoch` (MJD, UTC), `mag` and
  `obscode`, which are `classify`'s inputs, plus `object_id`, `n_obs`, `arc_hours` and
  `resid_rms_arcsec`. `classify` accepts the extra keys, so `pneo.classify(**tracklets)`
  scores them all.
- **`failures`**: `object_id`, `status` and `message` for each object that could not
  become a tracklet.

## `load_model`

```python
model = pneo.load_model(path=None, *, source="digest2", variant="All",
                        h_extend_to=30.0, h_interpolation="none")
```

Returns a `PopulationModel`. The defaults are the configuration used in the paper and by
`pneo classify`.

- **`path`**: the model file; by default the copy `pneo fetch` downloaded.
- **`source`**: which model; only `"digest2"` (the MPC's digest2 population model) exists
  today.
- **`variant`**: `"All"` is the full modeled population; `"Unk"` removes objects already
  in the MPC catalog.
- **`h_extend_to`**: the digest2 model ends at absolute magnitude H = 25.5; its faintest
  bin is extended to this magnitude so that very small, nearby objects can be scored.
  `None` disables the extension.
- **`h_interpolation`**: `"none"` keeps the model's piecewise-constant bins; `"pchip"`
  reconstructs the faint end with a smooth, count-conserving interpolation.

A `PopulationModel` has two components, `model.neo` and `model.other`, each a
`pneo.PopulationComponent`: an absolutely calibrated number density over orbital elements
and absolute magnitude. Because both are absolute, their ratio carries the relative
abundance of the two classes, and no separate prior is needed. A new model (for example
NEOMOD3, or one inferred from Rubin discoveries) is added by implementing
`PopulationComponent` in a module next to `pneo/models/digest2.py`.

## `posterior_map` and `plot_posterior_map`

```python
m = pneo.posterior_map(ra, dec, ra_rate, dec_rate, epoch, mag, *, model=None,
                       obscode=None, observer=None, obscodes=None, backend="auto", ...)
ax = pneo.plot_posterior_map(m, ax=None, component="both", colorbar=None)
```

For a single tracklet, `posterior_map` returns a dict of numpy arrays with one entry per
grid point: `rho` (distance [AU]), `rho_dot` (range-rate [AU/day]), `weight` (the cell
area), `density_neo` and `density_other` (the integrands), `H` (the implied absolute
magnitude) and `q` (the perihelion distance [AU]); plus `p_neo` and `status`. The NEO
share of `(density_neo + density_other) * weight` is P(NEO).

`plot_posterior_map` draws it; `component` selects `"neo"`, `"other"` or `"both"` (the
combined posterior).

## `fetch`

```python
pneo.fetch(model=True, obscodes=True)
```

Downloads the population model and the MPC list of observatory codes into the per-user
cache (see [cli.md](cli.md#downloaded-files)); the same as `pneo fetch`.

## Paths

`pneo.paths.cache_dir()`, `pneo.paths.default_model_path()` and
`pneo.paths.default_obscodes_path()` give the locations used for downloaded files.
