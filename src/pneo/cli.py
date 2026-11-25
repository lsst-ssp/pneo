"""The ``pneo`` command-line tool.

``pneo classify OBSFILE...`` is the counterpart of running ``digest2`` on an
observation file: it ingests ADES PSV or MPC 80-column astrometry
(:mod:`pneo._ingest`), classifies one tracklet per object, and writes one
row per tracklet to stdout.

Everything about the pipeline is spelled out in ``--help``; the design
decisions worth stating here are the ones a reviewer should be able to check:

**Scoring defaults are the D-18/D-19 operational configuration**, not the
library defaults: the ``All`` model variant, ``h_extend_to = 30`` (the finite
faint-end extension that supersedes the D-14 clamp),
``h_interpolation = "none"``, the ``(1200, 50)`` grid, ``backend="auto"``, and
the D-19 triage policy — which is implemented in the classifier's shared
post-processing and is therefore always active.  Its effect is **visible**, not
hidden: a policy-assigned row carries ``status = beyond_model``, and
its ``log_lr`` is ``nan`` because no likelihood ratio exists.  Every default is
overridable by a flag.

**Failure is per tracklet, never per batch.**  Real files break in every way
the NEOCP pilot's census enumerates (``validation/NEOCP_PILOT.md``).  An object
that cannot be ingested, or a tracklet that cannot be classified, produces a
row whose ``status`` says which and a one-line explanation on stderr; the run
continues and still exits 0.  Exit 1 means **no row carries a P(NEO)** at all
(:data:`SCORED_STATUSES`), and exit 2 is reserved for configuration errors —
no population model, an unreadable ``--obscodes``, a bad flag value — which are
caught before any scoring is done.

**stdout carries data, stderr carries everything else.**  Warnings, per-row
failure notes and the final summary all go to stderr, so
``pneo classify obs.psv > scores.tsv`` yields a clean parseable file.

``--workers`` defaults to ``os.cpu_count()`` and means different machinery on
each backend; see :func:`_score_tracklets` and the ``--help`` text.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

from . import __version__
from ._classifier import N_RHO_DEFAULT, N_RHODOT_DEFAULT
from ._constants import G_SLOPE_DEFAULT
from ._ingest import (
    DEFAULT_MAX_ARC_HOURS,
    DEFAULT_NIGHT_GAP_HOURS,
    FORMATS,
    OBSCODE_ENV_VAR,
    ObscodeTable,
    Tracklet,
    fetch_obscodes,
    ingest,
)
from .paths import default_model_path, default_obscodes_path
from .models.digest2 import D2_VARIANTS, H_INTERPOLATIONS

__all__ = ["main", "resolve_model", "OUTPUT_COLUMNS", "MODEL_ENV_VAR"]

#: Environment variable naming the digest2 population model file.
MODEL_ENV_VAR = "PNEO_MODEL"

#: The default (TSV/CSV) output columns, in order.  Names are
#: machine-friendly — no spaces, no units in the header — because these files
#: get fed to ``awk``, ``cut`` and ``pandas``; the units are in ``--help``.
OUTPUT_COLUMNS = (
    "object_id",
    "n_obs",
    "arc_hours",
    "epoch",
    "obscode",
    "mag_v",
    "p_neo",
    "log_lr",
    "status",
)

#: How many tracklets go into one batched Rust call.  Bounds the peak memory of
#: the batched path (which materializes every result before returning) without
#: costing anything measurable in per-call overhead.
RUST_CHUNK = 256

#: Per-status advice printed to stderr, once per status, when a row is not
#: ``ok``.  Terse on purpose: the classifier's own warning text is a paragraph,
#: which is right for a library and wrong for a terminal.
_STATUS_NOTES = {
    "beyond_model": (
        "p_neo = 1.0 is assigned by rule, not computed: every orbit consistent "
        "with the tracklet implies an object fainter (smaller) than the "
        "population model covers, so both likelihoods are zero and log_lr is "
        "empty. Such tracklets are very close and very small; on real NEOCP "
        "data all 11 were NEOs. Leave these rows out of calibration checks."
    ),
    "no_support": (
        "neither population component has objects anywhere consistent with "
        "the tracklet, so p_neo is undefined. Usually a very bright and/or "
        "very slow tracklet."
    ),
    "empty_region": (
        "the admissible region is empty: no bound orbit is consistent with "
        "this attributable, so p_neo is undefined (nan). Check the rates and "
        "the observatory code."
    ),
    "overflow": (
        "both marginal likelihoods overflowed to infinity; p_neo = 0.5 is the "
        "symmetric limit, not a measurement."
    ),
    "nan": "a population component returned NaN; this is a bug, please report it.",
}

#: Statuses a :class:`~pneo._classifier.ClassificationResult` can carry, i.e.
#: the rows for which the classifier ran to completion.  Everything else in the
#: status column is an ingestion failure or ``"error"``.
CLASSIFIER_STATUSES = frozenset(
    {"ok", "beyond_model", "no_support", "empty_region", "overflow",
     "nan"}
)

#: Above this fraction of scored tracklets resolved geocentrically, the run gets
#: a prominent stderr warning naming ``pneo fetch``.  A geocentric
#: stand-in for a real site moves the observer by up to one Earth radius, which
#: on a survey station is worth up to 0.17 in p_neo: measured by scoring the WP8
#: pilot's 499 real tracklets twice, once with the MPC table and once without.
#: 106 fell back; their |dp| ran to a maximum of 0.169 (F52, p 0.258 -> 0.427),
#: median 8.5e-4, while every correctly-placed row was bit-identical.  Large
#: enough to change a triage decision, so it must not be findable only by
#: reading the per-row provenance.
GEOCENTRIC_FALLBACK_ALERT = 0.10

#: How many per-record warnings / per-row failure notes go to stderr before the
#: rest are counted instead of printed.  A real file can carry thousands of
#: single-detection objects (an ITF dump is nothing but), and a terminal full of
#: identical lines hides the summary that matters.  Every row is still IN the
#: output; only the stderr commentary is capped.
MAX_STDERR_NOTES = 50

#: Statuses whose row carries an actual P(NEO) number.  The exit code turns on
#: this: a run in which no row got one produced nothing usable and exits 1,
#: whether that is because ingestion failed everywhere or because every
#: admissible region came back empty.
SCORED_STATUSES = frozenset({"ok", "beyond_model"})


class ConfigError(Exception):
    """A usage/configuration problem: reported cleanly, exit code 2."""


# --------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------


def resolve_model(flag: str | None = None) -> str:
    """Locate the digest2 population model.

    Resolution order is ``--model PATH``, then ``$PNEO_MODEL``, then the
    per-user cache (:func:`pneo.paths.default_model_path`, where
    ``pneo fetch`` writes it).  A path that was named explicitly (by flag or
    environment) and does not exist is an error naming *that* path — silently
    falling through to a different model would make the run unreproducible.

    Raises
    ------
    ConfigError
        With a message naming ``pneo fetch`` when nothing was found.
    """
    env = os.environ.get(MODEL_ENV_VAR, "").strip()
    for path, how in ((flag, "--model"), (env or None, f"${MODEL_ENV_VAR}")):
        if not path:
            continue
        if os.path.exists(path):
            return str(path)
        raise ConfigError(
            f"the population model given by {how} does not exist: {path}"
        )
    cached = str(default_model_path())
    if os.path.exists(cached):
        return cached
    raise ConfigError(
        "no digest2 population model found. Looked at: --model (not given), "
        f"${MODEL_ENV_VAR} (not set), {cached} (does not exist).\n"
        f"  Run `pneo fetch` to download it (~4 MB) into {cached}, "
        "or pass --model PATH."
    )


def _load_pair(model: str, variant: str, h_extend_to, h_interpolation: str):
    """Build the (NEO, other) component pair, with clean errors for bad knobs."""
    from .models.digest2 import Digest2Component

    try:
        return Digest2Component.neo_other_pair(
            model,
            variant,
            h_extend_to=h_extend_to,
            h_interpolation=h_interpolation,
        )
    except (ValueError, OSError) as exc:
        raise ConfigError(f"cannot build the population model: {exc}") from None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

#: Per-worker state, populated by :func:`_init_worker`.  A module-level dict
#: (rather than a closure) because ``multiprocessing`` has to be able to reach
#: it from a top-level, picklable function.
_W: dict = {}


def _init_worker(model, variant, h_extend_to, h_interpolation, cfg) -> None:
    """Parse the 45,936-bin model **once per worker**, not once per tracklet.

    This is the harness pattern (``validation/neocp_pilot.py:_init_worker``):
    the model file takes ~1.5 s to parse, which would otherwise dominate every
    task.  It is also the only mechanism that works under the ``spawn`` start
    method, where a child inherits nothing.

    **It must not be able to raise.**  ``multiprocessing.Pool`` respawns a
    worker whose initializer failed, forever — a bad ``--h-extend-to`` used to
    hang the whole command.  :func:`_score_tracklets` therefore builds the same
    pair in the parent first, so every way this call can fail has already been
    turned into a ``ConfigError`` by the time a pool exists.  (That is also why
    the ``--workers 1`` path no longer calls this function: it reuses the pair
    the parent already built.)
    """
    from .models.digest2 import Digest2Component

    neo, other = Digest2Component.neo_other_pair(
        model, variant, h_extend_to=h_extend_to,
        h_interpolation=h_interpolation,
    )
    _W["neo"], _W["other"], _W["cfg"] = neo, other, cfg


def _score_one(task: tuple) -> tuple:
    """Score one tracklet.  Module level and picklable (multiprocessing).

    Returns ``(idx, p_neo, log_lr, lambda_neo, lambda_other, n_nodes, status,
    message, notes)``.  Never raises: an exception becomes
    ``status = "error"`` with the exception text, because one pathological
    tracklet must not take down a pool — or a run.
    """
    from astropy.time import Time

    from ._attributable import Attributable
    from ._classifier import classify

    idx, ra, dec, ra_rate, dec_rate, jd, v_mag, obs_kw = task
    cfg = _W["cfg"]
    att = Attributable(
        ra=float(ra), dec=float(dec), ra_rate=float(ra_rate),
        dec_rate=float(dec_rate),
        epoch=Time(float(jd), format="jd", scale="utc"),
        v_mag=float(v_mag),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = classify(
                att, _W["neo"], _W["other"],
                rho_min=cfg["rho_min"], rho_max=cfg["rho_max"],
                n_rho=cfg["n_rho"], n_rhodot=cfg["n_rhodot"], G=cfg["G"],
                backend=cfg["backend"], diagnostics=False, **obs_kw,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately total
            return (
                int(idx), float("nan"), float("nan"), float("nan"),
                float("nan"), 0, "error", f"{type(exc).__name__}: {exc}",
                _foreign_warnings(caught),
            )
    return (
        int(idx), float(res.p_neo), float(res.log_lr), float(res.lambda_neo),
        float(res.lambda_other), int(res.n_nodes), str(res.status), "",
        _foreign_warnings(caught),
    )


def _foreign_warnings(caught) -> list[str]:
    """Warnings worth showing the user, i.e. not the classifier's own status text.

    Status warnings are dropped here and regenerated per row from
    ``res.status`` (:data:`_STATUS_NOTES`): they are paragraphs meant for a
    library caller, they are emitted identically by both backends, and the
    batched Rust path cannot attribute them to a tracklet anyway.  Anything
    *else* (astropy, numpy, a custom component) is a real surprise and is
    forwarded.
    """
    return [
        f"{w.category.__name__}: {w.message}"
        for w in caught
        if not str(w.message).startswith("pneo.classify:")
    ]


def _task(idx: int, tr: Tracklet) -> tuple:
    return (
        idx, tr.ra, tr.dec, tr.ra_rate, tr.dec_rate, tr.epoch_jd_utc,
        tr.v_mag, tr.observer_kwargs(),
    )


def _score_tracklets(
    tracklets: list[Tracklet], *, model: str, variant: str, h_extend_to,
    h_interpolation: str, cfg: dict, which: str, workers: int,
) -> tuple[list[tuple], list[str], str]:
    """Score every tracklet; returns ``(rows, notes, backend_actually_used)``.

    The third element is not always ``which``: an unsupported model makes the
    rust request fall back to the numpy reference, and the run summary has to
    report the backend that produced the numbers, not the one that was asked
    for.

    ``--workers N`` maps onto each backend's own parallelism:

    * **rust** — one batched ``classify_many(..., n_threads=N)`` call per chunk
      of :data:`RUST_CHUNK` tracklets.  The kernel releases the GIL and rayon
      parallelizes *inside* the call, so processes would only duplicate the
      model in memory.  Observer states are resolved here, in Python (astropy),
      one call per distinct (epoch, site) — ``classify_many`` takes a single
      ``obscode``/``lon``/``lat`` for the whole batch, so per-station batching
      is only possible by passing the ``(T, 3)`` states directly.  A batch that
      raises is **re-scored one tracklet at a time** (``classify_many``
      validates the whole batch before scoring any of it, so one non-finite
      v_mag would otherwise fail 256 tracklets); that is what makes "degrade
      per tracklet, never per batch" true on this path too.
    * **python** — a ``multiprocessing`` pool of N processes over tracklets,
      whose initializer loads the density pair once per worker.  The reference
      backend is pure numpy under the GIL, so processes are the only way to use
      more than one core.  ``N <= 1`` runs in-process with no pool, scoring
      through the same :func:`_score_one` against the same pair, so the two
      settings are required to agree exactly (``tests/test_cli.py``).

    Either way the pair is built **once, in this process, first** — see the
    comment below; that is what makes a bad model or a bad knob a clean
    ``ConfigError`` rather than a traceback or a hang.
    """
    notes: list[str] = []

    # Build the component pair HERE, in the parent, on every path.  Two reasons,
    # both learned the hard way:
    #   * it is the only place a bad knob (`--h-extend-to 10`) or an unreadable
    #     model can be turned into a clean ConfigError.  Deferring it to the
    #     pool's initializer produced a raw traceback at --workers 1 and, at
    #     --workers N, an INFINITE HANG: multiprocessing.Pool respawns a worker
    #     whose initializer raised, forever.
    #   * the rust path needs the pair anyway, and the workers <= 1 python path
    #     can reuse it instead of parsing the 45,936-bin file a second time.
    #
    # Note this happens BEFORE the empty-input shortcut below: `pneo classify
    # --h-extend-to 10` on a file with no usable tracklets is a configuration
    # error (exit 2) whether or not there was anything to score.  Reporting
    # "nothing scored" (exit 1) for a run whose configuration was never valid
    # would send the user looking at their data instead of their command line.
    pair = _load_pair(model, variant, h_extend_to, h_interpolation)
    if not tracklets:
        return [], [], which

    if which == "rust":
        from ._backend import ModelNotSupported, core_model_for

        neo, other = pair
        try:
            # Memoized, so paying for it here costs the batched call nothing.
            # Asking now (rather than letting classify_many raise mid-batch)
            # keeps the fallback a whole-run decision with one clear note.
            core_model_for(neo, other)
        except ModelNotSupported as exc:
            notes.append(
                "the compiled kernel cannot represent this population model "
                f"({exc}); scoring with the numpy reference backend instead. "
                "--workers now means processes, not threads."
            )
        else:
            rows, more = _score_rust(
                tracklets, pair=(neo, other), cfg=cfg, workers=workers
            )
            return rows, notes + more, "rust"
        cfg = dict(cfg, backend="python")
        which = "python"
    rows, more = _score_python(
        tracklets, model=model, variant=variant, h_extend_to=h_extend_to,
        h_interpolation=h_interpolation, cfg=cfg, workers=workers, pair=pair,
    )
    return rows, notes + more, which


def _score_python(
    tracklets, *, model, variant, h_extend_to, h_interpolation, cfg, workers,
    pair=None,
) -> tuple[list[tuple], list[str]]:
    tasks = [_task(i, tr) for i, tr in enumerate(tracklets)]
    if workers <= 1:
        if pair is None:  # pragma: no cover - the CLI always supplies it
            pair = _load_pair(model, variant, h_extend_to, h_interpolation)
        _W["neo"], _W["other"], _W["cfg"] = pair[0], pair[1], cfg
        rows = [_score_one(t) for t in tasks]
    else:
        import multiprocessing as mp

        # `fork` where it exists, `spawn` otherwise (macOS/Windows).
        #
        # Why fork and not forkserver, given that Python 3.12+ warns about
        # forking a possibly multi-threaded process:
        #   * it is what the WP8 harness uses, at 90 workers on this same
        #     workload, so it is the configuration with real mileage on it;
        #   * forkserver and spawn have to reconstruct `__main__` in the child,
        #     which fails outright for some legitimate ways of invoking a CLI
        #     (`python -c`, an embedded interpreter) -- a hard failure traded
        #     for an advisory one;
        #   * this process has NOT used the Rust kernel when it gets here: the
        #     rust branch returns before this, and the ModelNotSupported
        #     fallback fails inside `core_model_for` before rayon's pool is
        #     ever built. So the threads the warning is really about are absent.
        # The payload is deliberately fork-agnostic anyway -- a module-level
        # function plus picklable scalars -- so `spawn` is a correct fallback
        # and not a degraded one.
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        with ctx.Pool(
            processes=min(int(workers), len(tasks)),
            initializer=_init_worker,
            initargs=(model, variant, h_extend_to, h_interpolation, cfg),
        ) as pool:
            rows = pool.map(_score_one, tasks, chunksize=1)
    rows.sort(key=lambda r: r[0])
    notes: list[str] = []
    for r in rows:
        notes.extend(r[8])
    return rows, notes


def _score_rust(
    tracklets, *, pair, cfg, workers
) -> tuple[list[tuple], list[str]]:
    from astropy.time import Time

    from ._attributable import observer_state
    from ._classifier import classify_many

    neo, other = pair

    # One astropy call per distinct (epoch, site).  Identical epochs are the
    # common case in a survey's nightly file, and the ephemeris evaluation is
    # the whole Python-side cost of the batched path.
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    atts, R, V = [], [], []
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for tr in tracklets:
            kw = tr.observer_kwargs()
            key = (
                round(float(tr.epoch_jd_utc), 12),
                kw.get("lon_deg"), kw.get("lat_deg"), kw.get("height_m"),
            )
            hit = cache.get(key)
            if hit is None:
                hit = observer_state(
                    Time(float(tr.epoch_jd_utc), format="jd", scale="utc"), **kw
                )
                cache[key] = hit
            atts.append(tr.attributable())
            R.append(hit[0])
            V.append(hit[1])
    notes.extend(_foreign_warnings(caught))

    rows: list[tuple] = []
    for start in range(0, len(atts), RUST_CHUNK):
        stop = min(start + RUST_CHUNK, len(atts))
        ok, results, msg, caught = _rust_call(
            neo, other, atts[start:stop], R[start:stop], V[start:stop],
            cfg, workers,
        )
        notes.extend(_foreign_warnings(caught))
        if ok:
            rows.extend(
                _row_from_result(start + k, res)
                for k, res in enumerate(results)
            )
            continue

        # The batch raised.  "Degrade per tracklet, never per batch" is the
        # contract, so re-score this chunk ONE AT A TIME: whichever tracklet is
        # pathological (a non-finite v_mag is the reproducible case, since
        # classify_many validates the whole batch before scoring any of it)
        # gets its own error row and its 255 innocent neighbours get real
        # scores.  n_threads is kept at `workers` -- rayon simply has one item
        # to spread.
        notes.append(
            f"a batch of {stop - start} tracklets failed as a unit ({msg}); "
            "re-scoring it one tracklet at a time so the failure lands on the "
            "tracklet that caused it"
        )
        for k in range(start, stop):
            ok1, res1, msg1, caught1 = _rust_call(
                neo, other, atts[k:k + 1], R[k:k + 1], V[k:k + 1], cfg, workers
            )
            notes.extend(_foreign_warnings(caught1))
            if ok1 and res1:
                rows.append(_row_from_result(k, res1[0]))
            else:
                rows.append(
                    (k, float("nan"), float("nan"), float("nan"),
                     float("nan"), 0, "error", msg1, [])
                )
    return rows, notes


def _rust_call(neo, other, atts, R, V, cfg, workers):
    """One batched kernel call.  Returns ``(ok, results, message, warnings)``.

    Never raises: the caller decides whether a failure means "retry this batch
    one at a time" or "this single tracklet is the bad one".
    """
    from ._classifier import classify_many

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            results = classify_many(
                list(atts), neo, other,
                R_obs=np.asarray(R, dtype=float),
                V_obs=np.asarray(V, dtype=float),
                rho_min=cfg["rho_min"], rho_max=cfg["rho_max"],
                n_rho=cfg["n_rho"], n_rhodot=cfg["n_rhodot"], G=cfg["G"],
                backend="rust", diagnostics=False,
                n_threads=int(workers) if workers > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            return False, [], f"{type(exc).__name__}: {exc}", caught
    return True, results, "", caught


def _row_from_result(idx: int, res) -> tuple:
    """A scored-row tuple from a :class:`~pneo._classifier.ClassificationResult`."""
    return (
        int(idx), float(res.p_neo), float(res.log_lr), float(res.lambda_neo),
        float(res.lambda_other), int(res.n_nodes), str(res.status), "", [],
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _num(x, fmt: str) -> str:
    """Format a float for a text column; empty string for a missing value."""
    if x is None:
        return ""
    x = float(x)
    if np.isnan(x):
        return "nan"
    return format(x, fmt)


def _text(value) -> str:
    """A text field with the delimiters a TSV cannot carry removed.

    Only the tab/CR/LF characters, and only for the text columns.  An
    ``object_id`` or ``obscode`` is whatever the input file had between two
    delimiters, so a stray tab inside an ADES cell would otherwise shift every
    column to its right.  CSV quotes and JSON escapes handle this themselves;
    TSV has no escape, so the character has to go.
    """
    out = str("" if value is None else value)
    for bad in ("\t", "\r", "\n"):
        out = out.replace(bad, " ")
    return out


def _row_cells(row: dict) -> list[str]:
    """One output record as the nine :data:`OUTPUT_COLUMNS` strings."""
    return [
        _text(row["object_id"]),
        str(row["n_obs"]),
        "" if row["arc_hours"] is None else _num(row["arc_hours"], ".4f"),
        _text(row["epoch"]),
        _text(row["obscode"]),
        "" if row["mag_v"] is None else _num(row["mag_v"], ".2f"),
        "" if row["p_neo"] is None else _num(row["p_neo"], ".6g"),
        "" if row["log_lr"] is None else _num(row["log_lr"], ".6g"),
        _text(row["status"]),
    ]


def _write(rows: list[dict], fmt: str, stream) -> None:
    if fmt == "json":
        for row in rows:
            # allow_nan=False: the output must be RFC 8259 JSON, which jq, Go
            # and JSON.parse accept and Python's `Infinity`/`NaN` extension is
            # not.  `_json_ready` has already turned every non-finite float
            # into null, so this only ever asserts that.
            stream.write(
                json.dumps(_json_ready({k: v for k, v in row.items()
                                        if k not in _INTERNAL_FIELDS}),
                           default=_json_default,
                           allow_nan=False) + "\n"
            )
        return
    if fmt == "csv":
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow(_row_cells(row))
        return
    stream.write("\t".join(OUTPUT_COLUMNS) + "\n")
    for row in rows:
        stream.write("\t".join(_row_cells(row)) + "\n")


def _json_ready(row: dict) -> dict:
    """``row`` with every non-finite float replaced by ``None`` (JSON null).

    The text formats print ``inf``/``nan`` verbatim, because a human and awk
    both read them; JSON cannot carry them and stay standard.  Nothing is lost:
    ``status`` says *why* a value is missing — ``beyond_model``
    means the nan ``log_lr`` is the D-19 policy's "no likelihood ratio exists",
    ``ok`` with a null ``log_lr`` means a one-sided posterior (+/-inf), and an
    ingestion-failure status means the tracklet never existed.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.ndarray, tuple)):
        return list(obj)
    return str(obj)


#: Every key an output record carries, in emission order.  Both
#: :func:`_tracklet_row` and :func:`_failure_row` produce exactly this key set,
#: so `--format json` has one schema for every line.
_ROW_FIELDS = (
    "seq", "object_id", "n_obs", "arc_hours", "epoch", "epoch_mjd", "obscode",
    "mag_v", "p_neo", "log_lr", "status", "message", "ra", "dec", "ra_rate",
    "dec_rate", "resid_rms_arcsec", "bands", "obscode_source", "lon_deg",
    "lat_deg", "height_m", "truncated_arc", "n_detections_seen", "source",
    "lambda_neo", "lambda_other",
)

#: Fields used internally (input order) and left out of the JSON output.
_INTERNAL_FIELDS = ("seq",)


def _tracklet_row(tr: Tracklet, scored: tuple | None) -> dict:
    """Assemble the full output record for one scored tracklet.

    ``scored is None`` cannot happen through :func:`_score_tracklets` (which
    returns a row per task) but is handled anyway: the row then carries
    ``status = "error"`` rather than a silently blank score.
    """
    row = dict.fromkeys(_ROW_FIELDS)
    row.update(
        seq=tr.seq,
        object_id=tr.object_id,
        n_obs=tr.n_obs,
        arc_hours=tr.arc_hours,
        epoch=tr.epoch_isot + "Z",
        epoch_mjd=tr.epoch_jd_utc - 2400000.5,
        obscode=tr.obscode,
        mag_v=tr.v_mag,
        status="error",
        message="no result was returned for this tracklet",
        # provenance / diagnostics: JSON only, not in the text columns
        # degrees and deg/day; ra_rate is the on-sky rate (x cos dec)
        ra=float(np.degrees(tr.ra)),
        dec=float(np.degrees(tr.dec)),
        ra_rate=float(np.degrees(tr.ra_rate) * np.cos(tr.dec)),
        dec_rate=float(np.degrees(tr.dec_rate)),
        resid_rms_arcsec=tr.resid_rms_arcsec,
        bands=list(tr.bands),
        obscode_source=tr.obscode_source,
        lon_deg=tr.lon_deg,
        lat_deg=tr.lat_deg,
        height_m=tr.height_m,
        truncated_arc=tr.truncated_arc,
        n_detections_seen=tr.n_detections_seen,
        source=tr.source,
    )
    if scored is not None:
        (_i, p, llr, lam_neo, lam_other, n_nodes, status, message, _n) = scored
        row.update(
            p_neo=None if status == "error" else p,
            log_lr=None if status == "error" else llr,
            status=status,
            message=message,
            lambda_neo=None if status == "error" else lam_neo,
            lambda_other=None if status == "error" else lam_other,
        )
    return row


def _failure_row(fail) -> dict:
    """The output record for an object that never became a tracklet.

    Carries the **same keys** as :func:`_tracklet_row`, with ``None`` for
    everything a tracklet would have supplied: one JSON schema for every line
    of the output is worth more to a consumer than a shorter record.
    """
    row = dict.fromkeys(_ROW_FIELDS)
    row.update(
        seq=fail.seq,
        object_id=fail.object_id,
        n_obs=fail.n_obs,
        epoch="",
        obscode=fail.obscode,
        status=fail.status,
        message=fail.message,
        source=fail.source,
        bands=[],
    )
    return row


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _h_extend_to(value: str):
    """``--h-extend-to``: a magnitude, or ``none`` for the D-4 support."""
    text = str(value).strip().lower()
    if text in ("none", "off"):
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a magnitude (e.g. 30) or 'none', got {value!r}"
        ) from None


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") \
            from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


_EPILOG = """\
example
  $ pneo fetch                          # once: the population model and the
                                        # MPC observatory codes
  $ pneo classify examples/example.psv
  object_id  n_obs  arc_hours  epoch                     obscode  mag_v  p_neo     log_lr   status
  T000001    4      0.7500     2026-03-15T06:00:00.000Z  X05      19.77  1         inf      ok
  T000002    3      0.0667     2026-03-15T06:00:00.000Z  X05      22.62  1         nan      beyond_model
  T000003    5      1.3333     2026-03-15T06:00:00.000Z  F51      20.25  0.929963  2.58612  ok
  T000004    1                                           X05                                single_detection

  Columns are TAB-separated (aligned above for reading) and a field with no
  value is empty. One row per object, in the order the objects first appear.

columns
  object_id  trkSub, else provID, else permID (ADES); columns 1-12 (MPC 80).
  n_obs      detections in the tracklet used (on a failed row: seen in total).
  arc_hours  time spanned by the tracklet.
  epoch      UTC time of the first detection, where position and rates are fit.
  obscode    MPC observatory code.
  mag_v      median Johnson V over the tracklet, converted with digest2's band
             offsets. One difference from digest2: an MPC 80-column record
             with a blank band is read as V (digest2 applies -0.8 mag), so
             pneo reads it 0.8 mag fainter; such records are counted on stderr.
  p_neo      probability that the object is an NEO (printed with %.6g).
  log_lr     natural log of the NEO-to-other likelihood ratio; +/-inf when only
             one class has objects consistent with the tracklet.
  status     how the row was obtained:
    ok             a computed probability (including exactly 0 or 1).
    beyond_model   every consistent orbit is fainter (smaller) than the model
                   covers -- a very close, very small object. p_neo = 1 by
                   rule; log_lr is empty. Leave out of calibration checks.
    no_support     neither class has objects consistent with the tracklet;
                   p_neo is undefined.
    empty_region   no bound orbit fits the tracklet; p_neo is undefined.
    overflow, nan  numerical problems; see stderr.
    single_detection, no_magnitude, fit_failed, no_detections
                   the object could not become a tracklet; see stderr.
    error          classification raised; see stderr. The run continues.

json output
  --format json writes one JSON object per line with every field: the fitted
  position and rates (ra, dec [deg]; ra_rate = d(ra)/dt * cos(dec), dec_rate
  [deg/day]), epoch_mjd, both likelihoods, the fit residual RMS, the bands
  seen, the observer location and how it was found (obscode_source: builtin,
  file, geocenter or fallback), and whether the arc was truncated. Infinite
  and undefined numbers are written as null; status says why.

exit status
  0  at least one row has a P(NEO) (status ok or beyond_model)
  1  no row does
  2  configuration error (no population model, unreadable --obscodes, bad flag)
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pneo",
        description=(
            "The probability that each tracklet in an ADES PSV or MPC\n"
            "80-column observation file is a near-Earth object."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"pneo {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cls = sub.add_parser(
        "classify",
        help="classify the tracklets in one or more observation files",
        description=(
            "Read observation files, fit one tracklet per object, and write\n"
            "one row per object to stdout. The defaults are the configuration\n"
            "used in the paper.\n"
            "\n"
            "An object that cannot be read or classified becomes one row with a\n"
            "failure status and a line on stderr; the run continues. Warnings\n"
            "and the summary go to stderr, so stdout carries only data."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cls.add_argument(
        "obsfile", nargs="+", metavar="OBSFILE",
        help="observation file(s), ADES PSV or MPC 80-column (detected from "
             "the content). Detections are grouped by object across files.",
    )
    cls.add_argument(
        "--format", choices=("tsv", "csv", "json"), default="tsv",
        help="output format: tsv (default) or csv with the columns %s; json "
             "with every field, one object per line." % ", ".join(OUTPUT_COLUMNS),
    )
    cls.add_argument(
        "-o", "--output", metavar="PATH", default=None,
        help="write the rows to PATH instead of stdout.",
    )
    cls.add_argument(
        "-q", "--quiet", action="store_true",
        help="do not print the summary line on stderr.",
    )
    cls.add_argument(
        "--model", metavar="PATH", default=None,
        help=f"the population model file. Default: ${MODEL_ENV_VAR} if set, "
             f"else {default_model_path()} (see `pneo fetch`).",
    )
    cls.add_argument(
        "--variant", choices=D2_VARIANTS, default="All",
        help="All (default): the full modeled population; Unk: without the "
             "objects already in the MPC catalog.",
    )
    cls.add_argument(
        "--obscodes", metavar="PATH", default=None,
        help="a local copy of the MPC list of observatory codes, for stations "
             f"outside the built-in table. Default: ${OBSCODE_ENV_VAR} if set, "
             f"else {default_obscodes_path()} (see `pneo fetch`). Never "
             "downloaded during a run; an unknown code falls back to the "
             "center of the Earth with a warning.",
    )
    cls.add_argument(
        "--workers", type=_positive_int, default=os.cpu_count() or 1,
        metavar="N",
        help="parallelism (default: all %d cores): threads for the compiled "
             "kernel, processes for the numpy reference. 1 runs in-process."
             % (os.cpu_count() or 1),
    )

    g = cls.add_argument_group("advanced options")
    g.add_argument(
        "--backend", choices=("auto", "python", "rust"), default="auto",
        help="auto (default) uses the compiled kernel when it is installed; "
             "PNEO_BACKEND overrides this flag.",
    )
    g.add_argument(
        "--h-extend-to", type=_h_extend_to, default=30.0, metavar="MAG",
        help="extend the model's faintest bin (H 24-25.5) to this absolute "
             "magnitude (default 30), so very small nearby objects can be "
             "scored; 'none' disables it.",
    )
    g.add_argument(
        "--h-interpolation", choices=H_INTERPOLATIONS, default="none",
        help="faint-end reconstruction: none (default, the model's "
             "piecewise-constant bins) or pchip (smooth, count-conserving).",
    )
    g.add_argument(
        "--n-rho", type=_positive_int, default=N_RHO_DEFAULT, metavar="N",
        help=f"grid points in distance (default {N_RHO_DEFAULT}).",
    )
    g.add_argument(
        "--n-rhodot", type=_positive_int, default=N_RHODOT_DEFAULT,
        metavar="N",
        help=f"grid points in range-rate (default {N_RHODOT_DEFAULT}).",
    )
    g.add_argument(
        "--rho-min", type=float, default=0.01, metavar="AU",
        help="smallest distance considered (default 0.01 AU).",
    )
    g.add_argument(
        "--rho-max", type=float, default=100.0, metavar="AU",
        help="largest distance considered (default 100 AU).",
    )
    g.add_argument(
        "-G", type=float, default=G_SLOPE_DEFAULT, dest="g_slope",
        metavar="G",
        help=f"slope parameter of the H,G phase function (default "
             f"{G_SLOPE_DEFAULT}).",
    )
    g.add_argument(
        "--obs-format", choices=FORMATS, default="auto",
        help="force the input format instead of detecting it.",
    )
    g.add_argument(
        "--max-arc-hours", type=float, default=DEFAULT_MAX_ARC_HOURS,
        metavar="H",
        help=f"cut a night longer than H hours to its first H hours (default "
             f"{DEFAULT_MAX_ARC_HOURS:g}); the fit assumes linear motion.",
    )
    g.add_argument(
        "--night-gap-hours", type=float, default=DEFAULT_NIGHT_GAP_HOURS,
        metavar="H",
        help=f"a gap longer than H hours splits a night (default "
             f"{DEFAULT_NIGHT_GAP_HOURS:g}).",
    )

    fetch = sub.add_parser(
        "fetch",
        help="download the population model and the MPC observatory codes",
        description=(
            "Download the MPC digest2 population model (~4 MB) and the MPC\n"
            "list of observatory codes into the per-user cache, where\n"
            "`pneo classify` finds them. Set $PNEO_CACHE_DIR to use another\n"
            "directory. Each download is checked before it is written."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    which = fetch.add_mutually_exclusive_group()
    which.add_argument("--model-only", action="store_true",
                       help="download only the population model.")
    which.add_argument("--obscodes-only", action="store_true",
                       help="download only the observatory codes.")
    return parser


def _cmd_fetch(args) -> int:
    from ._fetch_model import fetch as fetch_model

    try:
        if not args.obscodes_only:
            fetch_model()
        if not args.model_only:
            path, n_codes = fetch_obscodes()
            print(f"wrote {path} ({n_codes} observatory codes with parallax "
                  "constants)", file=sys.stderr)
    except (RuntimeError, OSError) as exc:  # a clean CLI error, no traceback
        print(f"pneo: error: {exc}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
# The classify command
# --------------------------------------------------------------------------


def _cmd_classify(args, out_stream=None, err=None) -> int:
    err = err if err is not None else sys.stderr

    def note(msg: str) -> None:
        print(msg, file=err)

    model = resolve_model(args.model)
    which = _resolved_backend(args.backend)
    if not args.rho_min > 0.0:
        raise ConfigError(
            f"--rho-min must be > 0 (the rho axis is log-spaced), got "
            f"{args.rho_min!r}"
        )
    if not args.rho_max > args.rho_min:
        raise ConfigError(
            f"--rho-max must exceed --rho-min, got {args.rho_max!r} <= "
            f"{args.rho_min!r}"
        )
    if not args.max_arc_hours > 0.0:
        raise ConfigError(
            f"--max-arc-hours must be > 0, got {args.max_arc_hours!r}; a "
            "non-positive limit cannot admit even a two-detection tracklet"
        )
    if not args.night_gap_hours > 0.0:
        raise ConfigError(
            f"--night-gap-hours must be > 0, got {args.night_gap_hours!r}; a "
            "non-positive gap would split every detection into its own night"
        )
    # Grid floors.  Below these the quadrature is not coarse, it is meaningless:
    # the rho_dot direction needs at least an interval to have a midpoint, and a
    # handful of rho columns cannot straddle a single H bin edge, so the answer
    # is an artifact of node placement rather than an integral.
    if args.n_rho < 8 or args.n_rhodot < 2:
        raise ConfigError(
            f"the ranging grid is too coarse to integrate: --n-rho "
            f"{args.n_rho} (minimum 8), --n-rhodot {args.n_rhodot} (minimum 2). "
            f"The validated default is {N_RHO_DEFAULT} x {N_RHODOT_DEFAULT}."
        )
    if args.n_rho < N_RHO_DEFAULT or args.n_rhodot < N_RHODOT_DEFAULT:
        note(
            f"warning: grid {args.n_rho}x{args.n_rhodot} is coarser than the "
            f"validated default {N_RHO_DEFAULT}x{N_RHODOT_DEFAULT}; p_neo is "
            "under-resolved. Measured on a 64-geometry sweep, (200, 100) gives "
            "p90 |dp| 0.018 and max 0.071 against the reference, versus p90 "
            "0.0035 at the default. Do not compare these numbers with "
            "default-grid ones."
        )

    if args.output:
        # Opened (and truncated) BEFORE the run, not after it: discovering an
        # unwritable destination once the scoring is done throws the results
        # away, and truncating now means a run that dies later leaves an empty
        # file rather than a stale one -- exactly what `> file` would do.
        # The handle is deliberately NOT held open across the run: a file
        # object kept through an early `raise` would leak, and a failure at
        # write time is already covered by the same error path below.
        try:
            open(args.output, "w", encoding="utf-8").close()
        except OSError as exc:
            raise ConfigError(
                f"cannot write --output {args.output}: {exc}"
            ) from None

    # -- ingest --------------------------------------------------------------
    table = ObscodeTable(args.obscodes)
    if args.obscodes and table.load_error:
        # Checked before any work: an explicitly named obscode table that
        # cannot be read would otherwise silently degrade every station in the
        # file to the geocentre.
        raise ConfigError(f"--obscodes {table.load_error}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracklets, failures, issues = ingest(
            args.obsfile,
            fmt=args.obs_format,
            obscodes=table,
            night_gap_hours=args.night_gap_hours,
            max_arc_hours=args.max_arc_hours,
        )
    # Deduplicated and capped like every other stderr stream here: the same
    # advisory can arrive once per input file (the D-20 blank-band note names
    # its file, so 500 files would otherwise mean 500 lines).
    seen_ingest: set[str] = set()
    for w in caught:
        text = str(w.message)
        if text in seen_ingest:
            continue
        seen_ingest.add(text)
        if len(seen_ingest) > MAX_STDERR_NOTES:
            note("warning: ... and more ingestion warnings, not listed")
            break
        note(f"warning: {text}")
    for k, issue in enumerate(issues):
        if k >= MAX_STDERR_NOTES:
            note(f"warning: ... and {len(issues) - k} more unparseable "
                 "record(s), not listed")
            break
        where = f"{issue.source}:{issue.line}" if issue.line else issue.source
        note(f"warning: {where}: {issue.reason}")

    # -- score ---------------------------------------------------------------
    cfg = {
        "rho_min": float(args.rho_min),
        "rho_max": float(args.rho_max),
        "n_rho": int(args.n_rho),
        "n_rhodot": int(args.n_rhodot),
        "G": float(args.g_slope),
        # The worker calls classify() with the RESOLVED backend, not "auto":
        # a pool worker must not be free to disagree with the parent about
        # which kernel it is using.
        "backend": which,
    }
    scored_rows, notes, used_backend = _score_tracklets(
        tracklets, model=model, variant=args.variant,
        h_extend_to=args.h_extend_to, h_interpolation=args.h_interpolation,
        cfg=cfg, which=which, workers=int(args.workers),
    )
    seen_notes: set[str] = set()
    for msg in notes:
        # The same astropy/numpy warning can come back from every worker; the
        # user needs to see it once, not once per tracklet.
        if msg in seen_notes:
            continue
        seen_notes.add(msg)
        note(f"warning: {msg}")

    by_index = {r[0]: r for r in scored_rows}
    rows = [
        _tracklet_row(tr, by_index.get(i)) for i, tr in enumerate(tracklets)
    ]
    rows.extend(_failure_row(f) for f in failures)
    # Input order, across both lists: `group_tracklets` splits an object into
    # one of two lists but stamps each with its first-appearance index, so the
    # output row order is the order the objects appear in the input files.
    rows.sort(key=lambda r: r["seq"])

    # -- report --------------------------------------------------------------
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_policy = sum(1 for r in rows if r["status"] == "beyond_model")
    n_scored = sum(1 for r in rows if r["status"] in SCORED_STATUSES)
    n_undefined = sum(
        1 for r in rows
        if r["status"] in CLASSIFIER_STATUSES
        and r["status"] not in SCORED_STATUSES
    )
    n_failed = len(rows) - n_scored - n_undefined
    explained: set[str] = set()
    listed = 0
    unlisted = 0
    for row in rows:
        status = row["status"]
        if status == "ok":
            continue
        if listed >= MAX_STDERR_NOTES:
            unlisted += 1
            continue
        listed += 1
        head = f"{row['object_id']}: {status}"
        if row.get("message"):
            head += f": {row['message']}"
        note(head)
        # The per-status explanation is worth a paragraph, but only once.
        if status in _STATUS_NOTES and status not in explained:
            explained.add(status)
            note(f"  note: {status}: {_STATUS_NOTES[status]}")
    if unlisted:
        note(f"... and {unlisted} more non-ok row(s), not listed; every one of "
             "them is in the output with its own status")

    # A geocentric stand-in for a real station is a silent accuracy loss, not an
    # error, so it has to be loud in aggregate or it will not be noticed.
    n_fallback = sum(1 for r in rows if r.get("obscode_source") == "fallback")
    n_placed = sum(1 for r in rows if r.get("obscode_source") is not None)
    if n_placed and n_fallback / n_placed > GEOCENTRIC_FALLBACK_ALERT:
        codes = sorted(
            {r["obscode"] for r in rows if r.get("obscode_source") == "fallback"}
        )
        shown = ", ".join(codes[:12]) + (", ..." if len(codes) > 12 else "")
        note(
            f"WARNING: {n_fallback} of {n_placed} scored tracklets "
            f"({100.0 * n_fallback / n_placed:.0f}%) were placed at the "
            f"GEOCENTRE because their observatory code is unknown here: "
            f"{shown}. That displaces the observer by up to one Earth radius "
            "and is worth up to 0.17 in p_neo at a survey station -- enough to "
            "change a triage decision. Fix it with:\n"
            "    pneo fetch                 # downloads it to the user cache\n"
            "  then re-run (or pass --obscodes PATH)."
        )

    stream = out_stream if out_stream is not None else sys.stdout
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8", newline="") as fh:
                _write(rows, args.format, fh)
        except OSError as exc:  # the disk filled up during the run
            raise ConfigError(f"cannot write --output {args.output}: {exc}") \
                from None
    else:
        _write(rows, args.format, stream)

    if not args.quiet:
        obsc = table.loaded_from
        note(
            "pneo: {ok} ok, {policy} beyond the model, {undef} undefined, "
            "{failed} not classified, of {n} object(s) | model={model} "
            "variant={variant} h_extend_to={hx} h_interpolation={hi} "
            "grid={nr}x{nd} backend={be} workers={w} obscodes={obsc}".format(
                ok=n_ok, policy=n_policy, undef=n_undefined, failed=n_failed,
                n=len(rows), model=model, variant=args.variant,
                hx="none" if args.h_extend_to is None else f"{args.h_extend_to:g}",
                hi=args.h_interpolation, nr=args.n_rho, nd=args.n_rhodot,
                be=used_backend, w=args.workers,
                obsc=obsc if obsc else "built-in table only",
            )
        )
    return 0 if n_scored > 0 else 1


def _resolved_backend(flag: str) -> str:
    from ._backend import resolve_backend

    try:
        return resolve_backend(flag)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``pneo`` console entry point.  Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # stderr, not stdout: `pneo > out.tsv` with a forgotten subcommand
        # must leave out.tsv empty, not full of help text that a downstream
        # parser will then choke on.  Exit 2 for the same reason argparse uses
        # it -- the command line was wrong.
        parser.print_help(sys.stderr)
        return 2

    if args.command == "fetch":
        return _cmd_fetch(args)

    try:
        return _cmd_classify(args)
    except ConfigError as exc:
        print(f"pneo: error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # `pneo classify ... | head`
        # Python would otherwise print "Exception ignored" at shutdown when it
        # flushes a closed stdout.  Redirect the fd and exit quietly, which is
        # what a well-behaved filter does.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        print("pneo: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
