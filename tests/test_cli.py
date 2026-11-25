"""End-to-end tests for the ``pneo`` command-line tool (:mod:`pneo.cli`).

The input is a synthetic ADES PSV file built with
``transforms.observables_from_elements`` through the *inverse* ranging map, so
every tracklet's geometry is chosen rather than found:

* ``T000001`` a close NEO (rho = 0.1 AU, q = 1.09) -- an ordinary ``ok`` row;
* ``T000002`` rho = 0.01 AU, 30 deg/day, H = 32: its **entire** admissible
  region lies fainter than the model's H support even with ``h_extend_to=30``,
  so it exercises the **D-19 triage policy** (``fully_truncated_flagged``);
* ``T000003`` a slow, distant tracklet at a second station (F51), which also
  exercises the per-station observer state and the ``r`` -> V band correction;
* ``T000004`` one single detection -- an **ingestion failure** row.

Most tests call ``cli.main([...])`` in process (fast, and capsys separates the
streams the way the contract requires); one goes through
``python -m pneo.cli`` to prove the installed entry point and the exit
codes are real.  Everything runs on the coarse ``200 x 20`` grid except the
one test that pins the documented default-grid output, so the whole file stays
well inside a minute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import astropy.units as u  # noqa: E402
from astropy.time import Time  # noqa: E402

from pneo import __version__, cli  # noqa: E402
from pneo import transforms as T  # noqa: E402
from pneo.attributable import observer_state  # noqa: E402
from pneo.backend import core_available  # noqa: E402
from pneo.ingest import BAND_TO_V  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "data", "digest2.model.csv")
EPOCH = "2026-03-15T06:00:00"
#: Coarse grid for everything whose assertion is about plumbing, not accuracy.
FAST = ["--n-rho", "200", "--n-rhodot", "20"]


def _require_model():
    if not os.path.exists(MODEL_PATH):  # pragma: no cover
        pytest.skip(f"digest2 model file not found: {MODEL_PATH}")
    return MODEL_PATH


# --------------------------------------------------------------------------
# Synthetic input
# --------------------------------------------------------------------------


def _attributable_from_ranging(
    ra, dec, mu_ra_degday, mu_dec_degday, rho, rho_dot, H, obscode="X05"
):
    """Choose (A, rho, rho_dot, H); get back the exact observables they imply.

    ``topocentric_to_heliocentric`` -> ``elements_from_state`` ->
    ``observables_from_elements`` is the identity on (rho, rho_dot) by WP2's
    acceptance test, so this pins the geometry a tracklet will be scored at --
    including whether its implied H is above the population model's support.
    """
    t = Time(EPOCH, scale="utc")
    R, V = observer_state(t, obscode=obscode)
    r, v = T.topocentric_to_heliocentric(
        np.array([ra]), np.array([dec]),
        np.array([np.radians(mu_ra_degday)]),
        np.array([np.radians(mu_dec_degday)]),
        np.array([rho]), np.array([rho_dot]), R[None, :], V[None, :],
    )
    el = T.elements_from_state(T.state_equ_to_ecl(r), T.state_equ_to_ecl(v))
    assert np.isfinite(el).all(), "geometry implies an unbound orbit"
    obs = T.observables_from_elements(el, np.array([H]), R[None, :], V[None, :])
    out = {k: float(x[0]) for k, x in obs.items()}
    out["q_au"] = float(el[0, 0] * (1.0 - el[0, 1]))
    return out


_PSV_HEADER = (
    "permID |provID |trkSub  |mode |stn  |obsTime                     |"
    "ra          |dec         |mag   |band |astCat"
)


def _write_psv(path, specs):
    """``specs`` = [(trkSub, observables, n_det, cadence_min, band, stn), ...].

    The magnitude written is the *reported* one, i.e. V minus the band's
    offset, so that ingestion's band -> V conversion lands back on the
    generator's V and the round trip is exact to the field's two decimals.
    """
    lines = ["# synthetic pneo CLI test tracklets", _PSV_HEADER]
    t0 = Time(EPOCH, scale="utc")
    for trk, o, n_det, dt_min, band, stn in specs:
        reported = o["v_mag"] - BAND_TO_V[band]
        for j in range(n_det):
            dt = j * dt_min / (60.0 * 24.0)
            lines.append(
                "       |       |%-8s|CCD  |%-4s |%-27s |%12.8f |%+12.8f "
                "|%6.2f |%-4s |Gaia3"
                % (trk, stn, (t0 + dt * u.day).isot + "Z",
                   np.degrees(o["ra"] + o["ra_rate"] * dt) % 360.0,
                   np.degrees(o["dec"] + o["dec_rate"] * dt), reported, band)
            )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def _mpc80_line(desig, t, ra_deg, dec_deg, mag, band, obscode):
    d = t.datetime
    day = (
        d.day
        + (d.hour + (d.minute + (d.second + d.microsecond * 1e-6) / 60.0) / 60.0)
        / 24.0
    )
    rh = ra_deg / 15.0
    h = int(rh)
    m = int((rh - h) * 60.0)
    s = (rh - h - m / 60.0) * 3600.0
    sign = "-" if dec_deg < 0 else "+"
    ad = abs(dec_deg)
    dd = int(ad)
    dm = int((ad - dd) * 60.0)
    ds = (ad - dd - dm / 60.0) * 3600.0
    line = (
        f"{desig:<12s}  C{d.year:4d} {d.month:02d} {day:09.6f}"
        f"{h:02d} {m:02d} {s:06.3f}{sign}{dd:02d} {dm:02d} {ds:05.2f}"
        f"{'':9s}{mag:5.2f}{band}{'':6s}{obscode:>3s}"
    )
    assert len(line) == 80, (len(line), line)
    return line


@pytest.fixture(scope="module")
def geometry():
    return {
        "neo": _attributable_from_ranging(2.90, 0.10, 2.0, 0.7, 0.10, 0.0, 24.0),
        # Fast + faint + close: the whole admissible region is fainter than
        # H = 30, which is the D-19 triage policy's trigger.
        "wall": _attributable_from_ranging(2.90, 0.10, 30.0, 20.0, 0.01, 0.0, 32.0),
        "slow": _attributable_from_ranging(1.20, -0.30, 0.15, 0.05, 2.0, 0.0, 16.0),
    }


@pytest.fixture(scope="module")
def obsfile(tmp_path_factory, geometry):
    """The four-object PSV every test below scores."""
    path = tmp_path_factory.mktemp("cli") / "demo.psv"
    return _write_psv(path, [
        ("T000001", geometry["neo"], 4, 15.0, "V", "X05"),
        ("T000002", geometry["wall"], 3, 2.0, "V", "X05"),
        ("T000003", geometry["slow"], 5, 20.0, "r", "F51"),
        ("T000004", geometry["neo"], 1, 0.0, "V", "X05"),
    ])


def _score(capsys, obsfile, *extra, expect=0):
    """Run ``pneo score`` in process; return (rows, stdout, stderr)."""
    code = cli.main(["score", obsfile, "--model", _require_model(), *extra])
    out, err = capsys.readouterr()
    assert code == expect, f"exit {code} != {expect}\nstderr:\n{err}"
    return out, err


def _tsv(out):
    lines = out.strip("\n").split("\n")
    header = lines[0].split("\t")
    assert header == list(cli.OUTPUT_COLUMNS)
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


# --------------------------------------------------------------------------
# The documented output
# --------------------------------------------------------------------------


def test_score_tsv_end_to_end(capsys, obsfile):
    """The default invocation, on the default grid, with the documented shape."""
    out, err = _score(capsys, obsfile, "--workers", "4")
    rows = _tsv(out)
    assert [r["object_id"] for r in rows] == [
        "T000001", "T000002", "T000003", "T000004"
    ], "one row per object, in input order"

    neo, wall, slow, lone = rows
    assert neo["status"] == "ok"
    assert 0.0 <= float(neo["p_neo"]) <= 1.0
    assert neo["obscode"] == "X05" and neo["n_obs"] == "4"
    assert neo["epoch"] == "2026-03-15T06:00:00.000Z"
    assert float(neo["arc_h"]) == pytest.approx(0.75, abs=1e-3)

    # The D-19 triage policy, visible in the status column.
    assert wall["status"] == "fully_truncated_flagged"
    assert float(wall["p_neo"]) == 1.0
    assert wall["log_lr"] == "nan", "no likelihood ratio exists for a policy row"

    assert slow["status"] == "ok" and slow["obscode"] == "F51"
    # r -> V: the file carries the reported r magnitude, the row carries V.
    assert float(slow["V"]) == pytest.approx(20.25, abs=0.01)

    # The ingestion failure: a row, with empty numbers and a naming status.
    assert lone["status"] == "single_detection"
    assert (lone["p_neo"], lone["log_lr"], lone["epoch"], lone["V"]) == (
        "", "", "", ""
    )
    assert lone["n_obs"] == "1", "detections seen for the object"


def test_stdout_is_data_and_stderr_is_everything_else(capsys, obsfile):
    out, err = _score(capsys, obsfile, "--workers", "2", *FAST)
    assert "warning" not in out.lower()
    assert "note" not in out.lower()
    assert "TRIAGE POLICY" in err, "the policy must be explained once, on stderr"
    assert "single_detection" in err
    assert "pneo:" in err, "the summary line"
    assert err.count("TRIAGE POLICY") == 1, "explained once per status, not per row"


def test_quiet_suppresses_only_the_summary(capsys, obsfile):
    _out, err = _score(capsys, obsfile, "--workers", "2", "-q", *FAST)
    assert "pneo: " not in err
    assert "single_detection" in err, "per-row notes still reported"


def test_summary_line_states_the_operational_configuration(capsys, obsfile):
    _out, err = _score(capsys, obsfile, "--workers", "3", *FAST)
    for token in ("variant=All", "h_extend_to=30", "h_interpolation=none",
                  "grid=200x20", "workers=3", "backend="):
        assert token in err, token


# --------------------------------------------------------------------------
# Formats
# --------------------------------------------------------------------------


def test_csv_format(capsys, obsfile):
    import csv as _csv

    out, _err = _score(capsys, obsfile, "--workers", "2", "--format", "csv",
                        *FAST)
    rows = list(_csv.reader(out.strip("\n").split("\n")))
    assert rows[0] == list(cli.OUTPUT_COLUMNS)
    assert len(rows) == 5
    assert rows[2][-1] == "fully_truncated_flagged"


def test_json_format_is_strict_and_single_schema(capsys, obsfile):
    out, _err = _score(capsys, obsfile, "--workers", "2", "--format", "json",
                        *FAST)
    lines = out.strip("\n").split("\n")
    assert len(lines) == 4

    def reject(_value):  # json.loads accepts Infinity/NaN unless told not to
        raise AssertionError("non-finite literal in --format json output")

    rows = [json.loads(line, parse_constant=reject) for line in lines]
    for row in rows:
        assert tuple(row) == cli._ROW_FIELDS, "one schema for every line"

    neo, wall, slow, lone = rows
    assert wall["status"] == "fully_truncated_flagged"
    assert wall["policy_assigned"] is True
    assert wall["p_neo"] == 1.0
    assert wall["log_lr"] is None, "nan is emitted as JSON null"
    assert wall["lambda_neo"] == 0.0 and wall["lambda_other"] == 0.0
    assert neo["policy_assigned"] is False
    assert neo["n_nodes"] > 0
    assert neo["obscode_source"] == "builtin"
    assert neo["bands"] == ["V"] and slow["bands"] == ["r"]
    assert neo["ra"] == pytest.approx(2.90, abs=1e-6), "radians, at the epoch"
    assert lone["p_neo"] is None and "single_detection" in lone["status"]
    assert "no two within" in lone["message"]


def test_output_flag_writes_a_file(capsys, obsfile, tmp_path):
    dest = tmp_path / "scores.tsv"
    out, _err = _score(capsys, obsfile, "--workers", "2", "-o", str(dest),
                        *FAST)
    assert out == "", "nothing on stdout when -o is given"
    rows = _tsv(dest.read_text())
    assert len(rows) == 4


def test_tsv_cells_cannot_carry_a_delimiter(capsys, tmp_path, geometry):
    """A tab inside an ADES cell must not shift every column to its right."""
    path = tmp_path / "tabbed.psv"
    body = _write_psv(tmp_path / "_tmp.psv", [
        ("T1", geometry["neo"], 3, 15.0, "V", "X05"),
    ])
    path.write_text(open(body).read().replace("T1      ", "A\tB     "))
    out, _err = _score(capsys, str(path), "--workers", "1", *FAST)
    (row,) = _tsv(out)
    assert row["object_id"] == "A B"
    assert row["status"] == "ok", "the row is intact, not shifted"


def test_output_columns_are_a_subset_of_the_json_schema():
    assert set(cli.OUTPUT_COLUMNS) <= set(cli._ROW_FIELDS)
    assert set(cli.SCORED_STATUSES) <= set(cli.CLASSIFIER_STATUSES)
    assert set(cli._STATUS_NOTES) <= set(cli.CLASSIFIER_STATUSES)


def test_every_status_the_tool_can_print_is_documented_in_help():
    """No drift between the status vocabulary and what --help explains."""
    from pneo.ingest import IngestFailure

    documented = cli._EPILOG
    for status in cli.CLASSIFIER_STATUSES:
        assert status in documented, status
    # The ingestion-failure vocabulary, from IngestFailure's own docstring.
    for status in ("single_detection", "no_magnitude", "fit_failed",
                   "no_detections"):
        assert status in documented, status
        assert status in IngestFailure.__doc__, status
    assert "error" in documented


# --------------------------------------------------------------------------
# --workers, and the two backends
# --------------------------------------------------------------------------


def test_workers_one_matches_many_on_the_python_backend(capsys, obsfile,
                                                        monkeypatch):
    """--workers 1 (in process) and --workers N (a pool) must be identical.

    Both go through the same ``_score_one``; this pins that they also see the
    same model and the same configuration, which is the thing a pool
    initializer can get wrong.
    """
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    one, _ = _score(capsys, obsfile, "--backend", "python", "--workers", "1",
                    "-q", *FAST)
    many, _ = _score(capsys, obsfile, "--backend", "python", "--workers", "4",
                     "-q", *FAST)
    assert one == many


@pytest.mark.skipif(not core_available(), reason="pneo._core not built")
def test_the_two_backends_agree(capsys, obsfile, monkeypatch):
    """The batched rust path and the pooled python path must agree on p_neo.

    They reach the classifier by completely different routes -- one batched
    call with observer states resolved here, versus one process per tracklet
    resolving its own -- so this is the CLI's own backend-parity check on top
    of ``tests/test_backend_parity.py``.
    """
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    rust, _ = _score(capsys, obsfile, "--backend", "rust", "--workers", "4",
                     "-q", "--format", "json", *FAST)
    py, _ = _score(capsys, obsfile, "--backend", "python", "--workers", "4",
                   "-q", "--format", "json", *FAST)
    for a, b in zip(rust.strip().split("\n"), py.strip().split("\n")):
        ra, rb = json.loads(a), json.loads(b)
        assert ra["object_id"] == rb["object_id"]
        assert ra["status"] == rb["status"]
        if ra["p_neo"] is None:
            assert rb["p_neo"] is None
        else:
            assert ra["p_neo"] == pytest.approx(rb["p_neo"], abs=1e-6)


def test_workers_must_be_positive(capsys, obsfile):
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", obsfile, "--workers", "0"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# Exit-code semantics
# --------------------------------------------------------------------------


def test_exit_1_when_nothing_could_be_scored(capsys, tmp_path, geometry):
    """Every object fails to ingest -> rows are still written, exit 1."""
    path = _write_psv(tmp_path / "lonely.psv", [
        ("A", geometry["neo"], 1, 0.0, "V", "X05"),
        ("B", geometry["slow"], 1, 0.0, "V", "X05"),
    ])
    out, err = _score(capsys, path, "--workers", "1", *FAST, expect=1)
    rows = _tsv(out)
    assert [r["status"] for r in rows] == ["single_detection"] * 2
    assert "single_detection" in err


def test_exit_2_when_no_model_can_be_found(tmp_path, monkeypatch, capsys,
                                           obsfile):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(cli.MODEL_ENV_VAR, raising=False)
    assert cli.main(["score", obsfile, "--workers", "1"]) == 2
    _out, err = capsys.readouterr()
    assert "fetch-model" in err, "the error must name the fix"
    assert "pneo: error:" in err


def test_exit_2_when_an_explicit_model_is_missing(tmp_path, capsys, obsfile):
    assert cli.main(
        ["score", obsfile, "--model", str(tmp_path / "nope.csv")]
    ) == 2
    _out, err = capsys.readouterr()
    assert "--model" in err and "nope.csv" in err


def test_exit_2_when_an_explicit_obscode_table_is_missing(tmp_path, capsys,
                                                          obsfile):
    """A named obscode file that cannot be read must not degrade silently."""
    assert cli.main(
        ["score", obsfile, "--model", _require_model(),
         "--obscodes", str(tmp_path / "nope.html")]
    ) == 2
    _out, err = capsys.readouterr()
    assert "--obscodes" in err and "no such file" in err


# --------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------


def test_resolve_model_order(tmp_path, monkeypatch):
    flag = tmp_path / "flag.csv"
    env = tmp_path / "env.csv"
    data = tmp_path / "data" / "digest2.model.csv"
    data.parent.mkdir()
    for p in (flag, env, data):
        p.write_text("x")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv(cli.MODEL_ENV_VAR, str(env))
    assert cli.resolve_model(str(flag)) == str(flag), "the flag wins"
    assert cli.resolve_model(None) == str(env), "then the environment"
    monkeypatch.delenv(cli.MODEL_ENV_VAR)
    assert cli.resolve_model(None) == cli.DEFAULT_MODEL_PATH, "then ./data"
    data.unlink()
    with pytest.raises(cli.ConfigError, match="fetch-model"):
        cli.resolve_model(None)


def test_resolve_model_refuses_to_fall_through_from_a_named_path(monkeypatch,
                                                                tmp_path):
    """A named-but-missing model is an error, not a fallback: reproducibility."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "digest2.model.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(cli.MODEL_ENV_VAR, str(tmp_path / "gone.csv"))
    with pytest.raises(cli.ConfigError, match="PNEO_MODEL"):
        cli.resolve_model(None)


def test_model_env_var_is_used_by_score(capsys, obsfile, monkeypatch):
    monkeypatch.setenv(cli.MODEL_ENV_VAR, _require_model())
    code = cli.main(["score", obsfile, "--workers", "2", *FAST])
    out, err = capsys.readouterr()
    assert code == 0
    assert f"model={_require_model()}" in err
    assert len(_tsv(out)) == 4


# --------------------------------------------------------------------------
# Scoring flags
# --------------------------------------------------------------------------


def test_h_extend_to_none_changes_the_faint_end(capsys, obsfile):
    """`--h-extend-to none` must actually drop the D-19 synthetic H bin.

    Lambda_NEO is the observable that moves: the extension adds NEO mass in
    [25.5, 30), so any tracklet whose admissible region reaches past 25.5
    integrates more of it.
    """
    ext, _ = _score(capsys, obsfile, "--workers", "2", "-q", "--format",
                    "json", *FAST)
    trunc, err = _score(capsys, obsfile, "--workers", "2", "--h-extend-to",
                        "none", "--format", "json", *FAST)
    assert "h_extend_to=none" in err
    lam_ext = [json.loads(x)["lambda_neo"] for x in ext.strip().split("\n")]
    lam_tr = [json.loads(x)["lambda_neo"] for x in trunc.strip().split("\n")]
    assert lam_ext[0] is not None and lam_tr[0] is not None
    assert lam_ext[0] > lam_tr[0], (lam_ext, lam_tr)


def test_h_extend_to_rejects_a_bad_value(obsfile):
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", obsfile, "--h-extend-to", "wat"])
    assert exc.value.code == 2


@pytest.mark.parametrize("backend", ["python", "auto"])
@pytest.mark.parametrize("workers", ["1", "4"])
def test_h_extend_to_below_the_support_end_is_a_clean_error(
    capsys, obsfile, monkeypatch, backend, workers
):
    """A knob the population model rejects must not become a traceback -- or a hang.

    Regression test for two real bugs found by running the suite under
    ``PNEO_BACKEND=python``: the pair used to be built inside the pool's
    initializer, so at ``--workers 1`` the ValueError escaped as a traceback and
    at ``--workers N`` the command **hung forever**, because
    ``multiprocessing.Pool`` respawns a worker whose initializer raised. The
    pair is now built in the parent, before any pool exists, on every path --
    hence the (backend x workers) matrix here.
    """
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    assert cli.main(
        ["score", obsfile, "--model", _require_model(), "--h-extend-to", "10",
         "--backend", backend, "--workers", workers]
    ) == 2
    out, err = capsys.readouterr()
    assert "pneo: error:" in err and "Traceback" not in err
    assert "h_extend_to" in err
    assert out == "", "nothing is written when the configuration is rejected"


def test_bad_knobs_are_rejected_even_when_nothing_ingested(capsys, tmp_path,
                                                           geometry):
    """A configuration error is exit 2 whether or not there was data.

    Reporting "nothing scored" (exit 1) for a run whose command line was never
    valid sends the user to look at their data instead of their flags.
    """
    empty = _write_psv(tmp_path / "lonely.psv", [
        ("A", geometry["neo"], 1, 0.0, "V", "X05"),   # ingests to nothing
    ])
    assert cli.main(["score", empty, "--model", _require_model(),
                     "--workers", "1", *FAST]) == 1, "valid config, no data"
    capsys.readouterr()
    assert cli.main(["score", empty, "--model", _require_model(),
                     "--h-extend-to", "10", "--workers", "1", *FAST]) == 2
    _out, err = capsys.readouterr()
    assert "h_extend_to" in err


@pytest.mark.parametrize(
    "flag,value,fragment",
    [
        ("--rho-min", "0", "must be > 0"),
        ("--rho-max", "0.001", "must exceed --rho-min"),
        ("--max-arc-hours", "0", "--max-arc-hours must be > 0"),
        ("--night-gap-hours", "-1", "--night-gap-hours must be > 0"),
        ("--n-rho", "4", "too coarse"),
        ("--n-rhodot", "1", "too coarse"),
    ],
)
def test_out_of_range_knobs_exit_2_with_a_named_reason(capsys, obsfile, flag,
                                                       value, fragment):
    assert cli.main(["score", obsfile, "--model", _require_model(),
                     "--workers", "1", flag, value]) == 2
    out, err = capsys.readouterr()
    assert fragment in err, err
    assert out == "" and "Traceback" not in err


def test_a_grid_below_the_default_warns_that_it_is_under_resolved(capsys,
                                                                  obsfile):
    _out, err = _score(capsys, obsfile, "--workers", "1", *FAST)
    assert "coarser than the validated default" in err
    assert "under-resolved" in err


def test_the_default_grid_does_not_warn(capsys, obsfile):
    _out, err = _score(capsys, obsfile, "--workers", "2")
    assert "under-resolved" not in err


def test_output_is_opened_before_scoring(capsys, obsfile, tmp_path,
                                         monkeypatch):
    """An unwritable -o must be caught BEFORE the work, not after it."""
    called = {"scored": False}
    real = cli._score_tracklets

    def spy(*a, **kw):
        called["scored"] = True
        return real(*a, **kw)

    monkeypatch.setattr(cli, "_score_tracklets", spy)
    assert cli.main(["score", obsfile, "--model", _require_model(),
                     "-o", str(tmp_path / "nodir" / "x.tsv"),
                     "--workers", "1", *FAST]) == 2
    _out, err = capsys.readouterr()
    assert "cannot write --output" in err
    assert called["scored"] is False, "scoring must not have run"


def test_output_is_truncated_up_front_like_a_shell_redirect(capsys, obsfile,
                                                            tmp_path):
    """A run that dies must leave -o empty, not holding a stale previous run."""
    dest = tmp_path / "scores.tsv"
    dest.write_text("STALE RESULTS FROM AN EARLIER RUN\n")
    assert cli.main(["score", obsfile, "--model", _require_model(),
                     "-o", str(dest), "--h-extend-to", "10",
                     "--workers", "1", *FAST]) == 2
    capsys.readouterr()
    assert dest.read_text() == "", "stale output must not survive a failed run"


def test_summary_reports_the_backend_actually_used(capsys, obsfile,
                                                   monkeypatch):
    """After a ModelNotSupported fallback the summary must say 'python'."""
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    if not core_available():
        pytest.skip("pneo._core not built")

    from pneo.backend import ModelNotSupported

    def refuse(neo, other):
        raise ModelNotSupported("pretend the kernel cannot represent this")

    monkeypatch.setattr(cli, "core_model_for", refuse, raising=False)
    import pneo.backend as B

    monkeypatch.setattr(B, "core_model_for", refuse)
    _out, err = _score(capsys, obsfile, "--backend", "rust", "--workers", "2",
                       *FAST)
    assert "backend=python" in err, err
    assert "cannot represent this population model" in err


def test_an_unreadable_model_file_is_a_clean_error(capsys, obsfile, tmp_path):
    """A file that exists but is not a digest2 model must not hang or trace back."""
    junk = tmp_path / "junk.csv"
    junk.write_text("this,is,not,a,population,model\n1,2,3,4,5,6\n")
    assert cli.main(
        ["score", obsfile, "--model", str(junk), "--workers", "4",
         "--backend", "python", *FAST]
    ) == 2
    _out, err = capsys.readouterr()
    assert "pneo: error:" in err and "Traceback" not in err


def test_variant_flag_changes_the_score(capsys, obsfile):
    all_v, _ = _score(capsys, obsfile, "--workers", "2", "-q", "--format",
                      "json", *FAST)
    unk, err = _score(capsys, obsfile, "--workers", "2", "--variant", "Unk",
                      "--format", "json", *FAST)
    assert "variant=Unk" in err
    a = json.loads(all_v.strip().split("\n")[2])["p_neo"]
    b = json.loads(unk.strip().split("\n")[2])["p_neo"]
    assert a != b, "All and Unk are different priors"


def test_grid_and_G_flags_are_honoured(capsys, obsfile):
    coarse, _ = _score(capsys, obsfile, "--workers", "2", "-q", "--format",
                       "json", "--n-rho", "120", "--n-rhodot", "12")
    other_g, _ = _score(capsys, obsfile, "--workers", "2", "-q", "--format",
                        "json", "--n-rho", "120", "--n-rhodot", "12",
                        "-G", "0.05")
    a = json.loads(coarse.strip().split("\n")[2])["p_neo"]
    b = json.loads(other_g.strip().split("\n")[2])["p_neo"]
    assert a != b, "G enters through H(V, rho) and must move the posterior"


def test_rho_bounds_can_empty_the_admissible_region(capsys, obsfile):
    """A domain no bound orbit fits gives `empty_region` rows and exit 1."""
    out, err = _score(capsys, obsfile, "--workers", "1", "--rho-min", "50",
                      "--rho-max", "100", *FAST, expect=1)
    statuses = {r["status"] for r in _tsv(out)}
    assert "empty_region" in statuses
    assert "admissible region is empty" in err


# --------------------------------------------------------------------------
# Ingestion flags reached through the CLI
# --------------------------------------------------------------------------


def test_unknown_obscode_falls_back_to_the_geocentre_with_a_warning(
    capsys, tmp_path, geometry, monkeypatch
):
    monkeypatch.delenv("PNEO_OBSCODES", raising=False)
    monkeypatch.chdir(tmp_path)      # no ./data/ObsCodesF.html to find
    path = _write_psv(tmp_path / "zzz.psv", [
        ("T1", geometry["neo"], 3, 15.0, "V", "ZZZ"),
    ])
    out, err = _score(capsys, path, "--workers", "1", "--format", "json",
                      *FAST)
    row = json.loads(out.strip())
    assert row["obscode"] == "ZZZ"
    assert row["obscode_source"] == "fallback"
    assert row["status"] == "ok", "a fallback is a degradation, not a failure"
    assert "GEOCENTRICALLY" in err and "ZZZ" in err
    assert "obscodes=built-in table only" in err


def test_mpc80_input_is_detected_and_scored(capsys, tmp_path, geometry):
    o = geometry["neo"]
    t0 = Time(EPOCH, scale="utc")
    band = "r"
    reported = o["v_mag"] - BAND_TO_V[band]
    lines = []
    for j in range(3):
        dt = j * 20.0 / (60.0 * 24.0)
        lines.append(_mpc80_line(
            "K26A00A", t0 + dt * u.day,
            np.degrees(o["ra"] + o["ra_rate"] * dt) % 360.0,
            np.degrees(o["dec"] + o["dec_rate"] * dt), reported, band, "703",
        ))
    path = tmp_path / "a.obs"
    path.write_text("\n".join(lines) + "\n")
    out, _err = _score(capsys, str(path), "--workers", "1", *FAST)
    (row,) = _tsv(out)
    assert row["object_id"] == "K26A00A"
    assert row["obscode"] == "703"
    assert row["status"] == "ok"
    assert float(row["V"]) == pytest.approx(o["v_mag"], abs=0.02)


def test_two_files_of_different_formats_in_one_run(capsys, tmp_path, geometry,
                                                   obsfile):
    o = geometry["neo"]
    t0 = Time(EPOCH, scale="utc")
    lines = [
        _mpc80_line("K26A00A", t0 + j * 20.0 / 1440.0 * u.day,
                    np.degrees(o["ra"] + o["ra_rate"] * j * 20.0 / 1440.0) % 360.0,
                    np.degrees(o["dec"] + o["dec_rate"] * j * 20.0 / 1440.0),
                    o["v_mag"], "V", "703")
        for j in range(3)
    ]
    obs = tmp_path / "a.obs"
    obs.write_text("\n".join(lines) + "\n")
    code = cli.main(["score", str(obs), obsfile, "--model", _require_model(),
                     "--workers", "2", "-q", *FAST])
    out, err = capsys.readouterr()
    assert code == 0, err
    ids = [r["object_id"] for r in _tsv(out)]
    assert ids[0] == "K26A00A" and "T000002" in ids


def test_obs_format_can_be_forced(capsys, obsfile):
    out, _err = _score(capsys, obsfile, "--workers", "1", "--obs-format",
                       "ades_psv", *FAST)
    assert len(_tsv(out)) == 4
    # Forcing the wrong format is reported, not crashed on.
    code = cli.main(["score", obsfile, "--model", _require_model(),
                     "--obs-format", "mpc80", "--workers", "1", *FAST])
    out, err = capsys.readouterr()
    assert code == 1, "nothing parses as 80-column, so nothing is scored"
    assert out.strip("\n").count("\n") == 0, "header only, no rows"
    assert "bad record" in err and "demo.psv:" in err


def test_max_arc_hours_truncates(capsys, tmp_path, geometry):
    path = _write_psv(tmp_path / "long.psv", [
        ("T1", geometry["slow"], 6, 40.0, "V", "X05"),   # 3.33 h of arc
    ])
    out, _err = _score(capsys, path, "--workers", "1", "--max-arc-hours", "1",
                       "--format", "json", *FAST)
    row = json.loads(out.strip())
    assert row["truncated_arc"] is True
    assert row["arc_h"] <= 1.0
    assert row["n_detections_seen"] == 6 and row["n_obs"] < 6


def test_unreadable_input_is_reported_not_fatal(capsys, tmp_path, obsfile):
    code = cli.main(["score", str(tmp_path / "nope.psv"), obsfile,
                     "--model", _require_model(), "--workers", "2", *FAST])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert "nope.psv" in err
    assert len(_tsv(out)) == 4, "the readable file was still scored"


# --------------------------------------------------------------------------
# Entry point, --version, fetch-model
# --------------------------------------------------------------------------


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out, _err = capsys.readouterr()
    assert __version__ in out


def test_no_command_prints_help_on_stderr(capsys):
    """`pneo > out.tsv` with no subcommand must leave out.tsv EMPTY."""
    assert cli.main([]) == 2
    out, err = capsys.readouterr()
    assert out == "", "help must not go to stdout; stdout is data"
    for token in ("score", "fetch-model", "fetch-obscodes"):
        assert token in err, token


def test_help_is_self_sufficient_for_a_digest2_user(capsys):
    with pytest.raises(SystemExit):
        cli.main(["score", "--help"])
    out, _err = capsys.readouterr()
    assert "$ pneo score" in out, "one worked example"
    assert "fully_truncated_flagged" in out, "the D-19 status is documented"
    assert "exit status" in out
    assert "NOT a digest2 score" in out, "the score is not a digest2 score"
    for flag in ("--variant", "--h-extend-to", "--h-interpolation", "--n-rho",
                 "--n-rhodot", "--rho-min", "--rho-max", "--backend", "-G",
                 "--workers", "--model", "--format"):
        assert flag in out, flag
    assert "n_threads" in out, "the --workers mapping must be documented"
    assert "multiprocessing" in out
    # The truncation rule's edge case must be stated, not left to be discovered.
    assert "first TWO detections" in out
    # The one deliberate photometric divergence from digest2 (D-20).
    assert "BLANK band" in out and "-0.8" in out
    # And the obscode story, since a fallback silently costs accuracy.
    assert "fetch-obscodes" in out and "GEOCENTRE" in out


# --------------------------------------------------------------------------
# Rust chunk failure degrades per tracklet (review MAJOR 4)
# --------------------------------------------------------------------------


def _bare_tracklet(i, v_mag):
    from pneo.ingest import Tracklet

    return Tracklet(
        object_id=f"T{i}", obscode="X05", n_obs=3, arc_hours=0.5,
        epoch_jd_utc=2461114.75, ra=2.9, dec=0.1, ra_rate=-0.02,
        dec_rate=0.005, v_mag=v_mag, resid_rms_arcsec=0.1,
        lon_deg=289.25058, lat_deg=-30.2446, height_m=2647.0,
        obscode_source="builtin", seq=i,
    )


@pytest.mark.skipif(not core_available(), reason="pneo._core not built")
@pytest.mark.parametrize("which", ["rust", "python"])
def test_one_bad_tracklet_costs_one_row_not_the_batch(which, monkeypatch):
    """`classify_many` validates a whole batch before scoring any of it.

    A single non-finite v_mag therefore raises for the entire batched call, and
    before the fix that turned every tracklet in the chunk (up to
    ``RUST_CHUNK`` = 256 of them) into an error row. The rust path now re-scores
    a failed chunk one tracklet at a time, so the failure lands where it
    belongs. The python path has always been per tracklet; it is parametrized
    here so the two backends are held to the same contract by the same test.
    """
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    tracklets = [
        _bare_tracklet(0, 21.0), _bare_tracklet(1, float("nan")),
        _bare_tracklet(2, 20.0),
    ]
    cfg = dict(rho_min=0.01, rho_max=100.0, n_rho=200, n_rhodot=20, G=0.15,
               backend=which)
    rows, notes, used = cli._score_tracklets(
        tracklets, model=_require_model(), variant="All", h_extend_to=30.0,
        h_interpolation="none", cfg=cfg, which=which, workers=4,
    )
    assert used == which
    by_idx = {r[0]: r for r in rows}
    assert len(by_idx) == 3
    assert by_idx[1][6] == "error", "the bad tracklet fails"
    assert "v_mag must be finite" in by_idx[1][7]
    for i in (0, 2):
        assert by_idx[i][6] == "ok", "its neighbours must still be scored"
        assert 0.0 <= by_idx[i][1] <= 1.0
    assert sum(1 for r in rows if r[6] == "error") == 1
    if which == "rust":
        assert any("one tracklet at a time" in n for n in notes)


@pytest.mark.skipif(not core_available(), reason="pneo._core not built")
def test_rust_retry_scores_a_full_chunk_around_one_bad_tracklet(monkeypatch):
    """The retry must not be a special case that only works for tiny batches."""
    monkeypatch.delenv("PNEO_BACKEND", raising=False)
    n = 12
    tracklets = [_bare_tracklet(i, 20.0 + 0.1 * i) for i in range(n)]
    tracklets[7] = _bare_tracklet(7, float("nan"))
    cfg = dict(rho_min=0.01, rho_max=100.0, n_rho=120, n_rhodot=12, G=0.15,
               backend="rust")
    rows, _notes, _used = cli._score_tracklets(
        tracklets, model=_require_model(), variant="All", h_extend_to=30.0,
        h_interpolation="none", cfg=cfg, which="rust", workers=4,
    )
    statuses = {r[0]: r[6] for r in rows}
    assert len(statuses) == n
    assert statuses[7] == "error"
    assert sum(1 for v in statuses.values() if v == "error") == 1, statuses


# --------------------------------------------------------------------------
# Observatory codes (review MAJOR 3)
# --------------------------------------------------------------------------


def test_more_than_ten_percent_geocentric_shouts_and_names_fetch_obscodes(
    capsys, tmp_path, geometry, monkeypatch
):
    monkeypatch.delenv("PNEO_OBSCODES", raising=False)
    monkeypatch.chdir(tmp_path)
    path = _write_psv(tmp_path / "zzz.psv", [
        ("A", geometry["neo"], 3, 15.0, "V", "ZZZ"),     # unknown -> fallback
        ("B", geometry["slow"], 3, 15.0, "V", "X05"),    # built-in
    ])
    _out, err = _score(capsys, str(path), "--workers", "1", *FAST)
    assert "WARNING:" in err
    assert "1 of 2 scored tracklets (50%)" in err
    assert "GEOCENTRE" in err and "ZZZ" in err
    assert "pneo fetch-obscodes" in err


def test_all_stations_known_means_no_fallback_alert(capsys, obsfile):
    _out, err = _score(capsys, obsfile, "--workers", "1", *FAST)
    assert "were placed at the GEOCENTRE" not in err


def test_fetch_obscodes_subcommand(monkeypatch, tmp_path, capsys):
    """Wiring, success reporting and clean failure -- without touching the network."""
    from pneo import ingest as ing

    seen = {}

    def fake(dest, timeout=120.0):
        seen["dest"] = dest
        return str(dest), 2692

    monkeypatch.setattr(cli, "fetch_obscodes", fake)
    assert cli.main(["fetch-obscodes"]) == 0
    _out, err = capsys.readouterr()
    assert seen["dest"] == ing.DEFAULT_OBSCODES_DEST
    assert "2692 observatory codes" in err

    assert cli.main(["fetch-obscodes", "--out", str(tmp_path / "x.html")]) == 0
    assert seen["dest"] == str(tmp_path / "x.html")

    def boom(dest, timeout=120.0):
        raise RuntimeError("no route to host")

    monkeypatch.setattr(cli, "fetch_obscodes", boom)
    assert cli.main(["fetch-obscodes"]) == 1
    _out, err = capsys.readouterr()
    assert "pneo: error: no route to host" in err
    assert "Traceback" not in err


def test_fetch_obscodes_writes_nothing_when_the_payload_is_not_the_table(
    monkeypatch, tmp_path
):
    """A captive portal must not be able to replace a good local table."""
    from pneo import ingest as ing

    class _Resp:
        def read(self):
            return b"<html><body>Sign in to the hotel wifi</body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ing.urllib.request if hasattr(ing, "urllib") else
                        __import__("urllib.request", fromlist=["request"]),
                        "urlopen", lambda *a, **k: _Resp())
    dest = tmp_path / "obs.html"
    dest.write_text("PRECIOUS EXISTING TABLE")
    with pytest.raises(RuntimeError, match="no observatory codes"):
        ing.fetch_obscodes(str(dest))
    assert dest.read_text() == "PRECIOUS EXISTING TABLE", "not clobbered"
    assert not (tmp_path / "obs.html.part").exists(), "temp file cleaned up"


def test_fetch_model_subcommand_wraps_the_existing_script(monkeypatch):
    """`pneo fetch-model` must be the same code as `pneo-fetch-model`."""
    from pneo import fetch_model

    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(fetch_model, "main", fake_main)
    assert cli.main(["fetch-model"]) == 7
    assert seen["argv"] == []
    assert cli.main(["fetch-model", "--out", "/tmp/x.csv"]) == 7
    assert seen["argv"] == ["--out", "/tmp/x.csv"]


def test_module_entry_point_in_a_subprocess(obsfile, tmp_path):
    """`python -m pneo.cli` -- the real process, the real exit code."""
    _require_model()
    proc = subprocess.run(
        [sys.executable, "-m", "pneo.cli", "score", obsfile,
         "--model", MODEL_PATH, "--workers", "2", "-q", *FAST],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    rows = _tsv(proc.stdout)
    assert [r["object_id"] for r in rows] == [
        "T000001", "T000002", "T000003", "T000004"
    ]
    assert rows[1]["status"] == "fully_truncated_flagged"
    assert proc.stdout.count("\n") == 5, "header + four rows, nothing else"


def test_console_script_is_declared_in_pyproject():
    """The `pneo` entry point must be wired, or none of this is reachable."""
    text = open(os.path.join(REPO_ROOT, "pyproject.toml")).read()
    assert 'pneo = "pneo.cli:main"' in text
    assert 'pneo-fetch-model = "pneo.fetch_model:main"' in text, (
        "the pre-existing script must keep working"
    )
