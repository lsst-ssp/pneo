"""Unit tests for :mod:`pneo.ingest` — observation files -> tracklets.

Entirely offline and fast.  Three groups carry most of the weight:

* **the band -> V table** is checked against a hardcoded copy of digest2
  2.8.2's own offsets (not against the module's own dict, which would be
  circular), including the two-character ADES survey codes real MPC output
  uses and the ``Vj`` case a mechanical "take the second character" rule gets
  wrong by 1.2 mag;
* **the attributable fit** is round-tripped against
  ``transforms.observables_from_elements`` on a synthetic Keplerian tracklet,
  and the residual is shown to be *curvature* (position error ~ arc^2, rate
  error ~ arc) rather than a bug — the pilot's pattern;
* **equivalence with the WP8 pilot**, which built and validated this ingestion
  logic on 500 real MPC NEOCP tracklets: the package must reproduce the
  pilot's stored ``tracklets.json`` from the pilot's own PSV file.  That test
  skips when the (gitignored) pilot output is absent, so CI passes without it.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from astropy.time import Time  # noqa: E402

from pneo import ingest as G  # noqa: E402
from pneo import transforms as T  # noqa: E402
from pneo.attributable import (  # noqa: E402
    MPC_OBSCODES,
    observatory_location,
    observer_state,
)
from pneo.constants import K_GAUSS  # noqa: E402

ARCSEC = 206264.80624709636


# --------------------------------------------------------------------------
# Photometry: band -> V (D-5, digest2's own table)
# --------------------------------------------------------------------------

#: digest2 2.8.2's ``observation.py:_BAND_CORRECTIONS``, transcribed here
#: independently so this test cannot pass by agreeing with itself.
_DIGEST2_BAND_CORRECTIONS = {
    "V": 0.0, "B": -0.8, "U": -1.3, "g": -0.28, "r": 0.23, "R": 0.4,
    "C": 0.4, "W": 0.4, "i": 0.39, "z": 0.37, "I": 0.8, "J": 1.2,
    "w": -0.16, "y": 0.36, "L": 0.2, "H": 1.4, "K": 1.7, "Y": 0.7,
    "G": 0.24, "v": 0.0, "c": -0.05, "o": 0.33, "u": 2.5,
}


def test_band_table_matches_digest2_verbatim():
    assert G.BAND_TO_V == _DIGEST2_BAND_CORRECTIONS
    assert G.BAND_TO_V_DEFAULT == -0.8


@pytest.mark.parametrize(
    "band,mag,expect",
    [
        ("V", 20.0, 20.0),
        ("r", 20.0, 20.23),
        ("w", 20.0, 19.84),
        ("u", 20.0, 22.5),
        ("", 20.0, 20.0),        # missing band is V, as digest2's parser does
        (None, 20.0, 20.0),
        ("Q", 20.0, 19.2),       # unknown band -> digest2's -0.8 fallback
        ("UNK", 20.0, 19.2),     # deliberately NOT aliased
    ],
)
def test_band_to_v_values(band, mag, expect):
    assert G.band_to_v(band, mag) == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize(
    "code,expect_band,expect_offset",
    [
        ("Pw", "w", -0.16),   # Pan-STARRS broad w
        ("Ao", "o", 0.33),    # ATLAS orange
        ("Ac", "c", -0.05),   # ATLAS cyan
        ("Lr", "r", 0.23),    # Rubin r
        ("Lz", "z", 0.37),
        ("Vj", "V", 0.0),     # Johnson V -- NOT J band
        ("Ic", "I", 0.8),     # Cousins I -- NOT C band
    ],
)
def test_two_character_band_codes_resolve(code, expect_band, expect_offset):
    assert G.normalize_band(code) == expect_band
    assert G.band_to_v(code, 20.0) == pytest.approx(20.0 + expect_offset, abs=1e-12)


def test_johnson_codes_are_not_a_second_character_rule():
    """`Vj` must be V (+0.0), not J (+1.2): the order is not uniform.

    This is the one entry where a plausible implementation ("take the second
    character of a two-character code") is wrong, and wrong by more than a
    magnitude.  Pinned so nobody replaces the table with the rule.
    """
    assert G.normalize_band("Vj") == "V"
    assert G.band_to_v("Vj", 20.0) == pytest.approx(20.0)
    assert G.band_to_v("Vj", 20.0) != pytest.approx(20.0 + 1.2)


def test_every_alias_target_is_a_band_digest2_knows():
    for code, target in G.MPC_BAND_ALIASES.items():
        assert target in G.BAND_TO_V, f"{code} -> {target} is not a digest2 band"
        assert len(code) == 2, code


def test_band_to_v_leaves_nonpositive_or_missing_magnitude_alone():
    """digest2 applies the offset only when mag > 0; 0 is its "no photometry"."""
    assert G.band_to_v("r", 0.0) == 0.0
    assert G.band_to_v("r", -3.0) == -3.0
    assert np.isnan(G.band_to_v("r", float("nan")))


# --------------------------------------------------------------------------
# Observatory codes
# --------------------------------------------------------------------------

_OBSCODES_FIXTURE = """<html><body><pre>
Code  Long.    cos       sin     Name
500   0.0000   0.000000 +0.000000 Geocentric
309   289.59569 0.909943 -0.414336 Cerro Paranal
G96   249.21128 0.845107 +0.533611 <a href="x">Mt. Lemmon Survey</a>
002   0.62     0.622    +0.781    Rayleigh
258                              Gaia
247                              Roving Observer
</pre></body></html>
"""


def test_parse_obscodes_fixture(tmp_path):
    path = tmp_path / "ObsCodesF.html"
    path.write_text(_OBSCODES_FIXTURE)
    table = G.parse_obscodes(str(path))
    assert set(table) == {"500", "309", "G96", "002"}, "space/roving must drop"
    assert table["G96"]["lon_deg"] == pytest.approx(249.21128)
    assert table["G96"]["rho_cos_phi"] == pytest.approx(0.845107)
    assert table["G96"]["rho_sin_phi"] == pytest.approx(0.533611)
    assert "Mt. Lemmon" in table["G96"]["name"], "HTML links must be stripped"
    assert table["002"]["rho_sin_phi"] == pytest.approx(0.781)


def test_obscode_geodetic_reproduces_the_builtin_table():
    """Parallax constants -> WGS84 -> ITRS must land back on the same site.

    This is the correctness guarantee for scoring at an arbitrary obscode: the
    CLI resolves *every* station to lon/lat/height and passes those to
    ``observer_state``, so that route has to agree with the package's own
    obscode lookup wherever both exist.
    """
    import astropy.units as u

    worst = 0.0
    for code, (lon, rho_cos, rho_sin) in MPC_OBSCODES.items():
        if code == "500":
            continue
        lon_d, lat_d, height = G.obscode_geodetic(
            {"lon_deg": lon, "rho_cos_phi": rho_cos, "rho_sin_phi": rho_sin}
        )
        a = observatory_location(obscode=code)
        b = observatory_location(lon_deg=lon_d, lat_deg=lat_d, height_m=height)
        worst = max(worst, float(np.linalg.norm(
            a.get_itrs().cartesian.xyz.to_value(u.m)
            - b.get_itrs().cartesian.xyz.to_value(u.m)
        )))
    assert worst < 1e-3, f"geodetic round trip off by {worst} m"


def test_obscode_table_prefers_the_builtin_entry_over_a_file(tmp_path):
    """A local table must not be able to move a site pneo already knows."""
    bad = tmp_path / "obs.html"
    bad.write_text(
        "<pre>\nCode Long. cos sin Name\n"
        "G96   0.00000 0.100000 +0.100000 Somewhere Wrong\n</pre>\n"
    )
    table = G.ObscodeTable(str(bad))
    lon, lat, height, source = table.resolve("G96")
    assert source == "builtin"
    expect = G.obscode_geodetic(
        {"lon_deg": MPC_OBSCODES["G96"][0],
         "rho_cos_phi": MPC_OBSCODES["G96"][1],
         "rho_sin_phi": MPC_OBSCODES["G96"][2]}
    )
    assert (lon, lat, height) == pytest.approx(expect)


def test_obscode_table_uses_the_file_for_an_unknown_code(tmp_path):
    path = tmp_path / "obs.html"
    path.write_text(_OBSCODES_FIXTURE)
    table = G.ObscodeTable(str(path))
    lon, lat, height, source = table.resolve("002")
    assert source == "file"
    assert table.loaded_from == os.path.abspath(str(path))
    assert table.load_error is None
    assert lon == pytest.approx(0.62)


def test_obscode_table_geocentre_and_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)              # no local table to find
    monkeypatch.delenv(G.OBSCODE_ENV_VAR, raising=False)
    table = G.ObscodeTable()
    assert table.resolve("500") == (0.0, 0.0, 0.0, "geocenter")
    with pytest.warns(UserWarning, match="ZZZ"):
        lon, lat, height, source = table.resolve("ZZZ")
    assert (lon, lat, height, source) == (0.0, 0.0, 0.0, "fallback")
    # Warned once per code, not once per tracklet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert table.resolve("ZZZ")[3] == "fallback"


def test_obscode_table_warns_when_a_detection_has_no_station(tmp_path,
                                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(G.OBSCODE_ENV_VAR, raising=False)
    table = G.ObscodeTable()
    with pytest.warns(UserWarning, match="no observatory code"):
        assert table.resolve("") == (0.0, 0.0, 0.0, "fallback")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert table.resolve("")[3] == "fallback", "warned once, not per row"


def test_obscode_table_keeps_looking_past_an_empty_file(tmp_path, monkeypatch):
    """An empty $PNEO_OBSCODES file must not mask a good copy in the cache."""
    empty = tmp_path / "empty.html"
    empty.write_text("<pre>\n</pre>\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ObsCodesF.html").write_text(_OBSCODES_FIXTURE)
    monkeypatch.setenv(G.OBSCODE_ENV_VAR, str(empty))
    monkeypatch.setenv("PNEO_CACHE_DIR", str(cache))
    table = G.ObscodeTable()
    assert table.resolve("002")[3] == "file"
    assert table.loaded_from == str(cache / "ObsCodesF.html")


def test_obscode_fallback_warning_names_the_table_it_searched(tmp_path):
    path = tmp_path / "obs.html"
    path.write_text(_OBSCODES_FIXTURE)
    table = G.ObscodeTable(str(path))
    with pytest.warns(UserWarning, match=str(path)):
        table.resolve("ZZZ")


def test_obscode_table_reports_a_missing_explicit_file(tmp_path):
    table = G.ObscodeTable(str(tmp_path / "nope.html"))
    assert "no such file" in (table.load_error or "")


def test_obscode_env_var_is_honoured(tmp_path, monkeypatch):
    path = tmp_path / "obs.html"
    path.write_text(_OBSCODES_FIXTURE)
    monkeypatch.setenv(G.OBSCODE_ENV_VAR, str(path))
    assert G.ObscodeTable().resolve("002")[3] == "file"


# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------

_PSV = (
    "# a comment\n"
    "permID |provID |trkSub  |mode |stn  |obsTime                     |"
    "ra          |dec         |mag   |band |astCat\n"
    "       |       |T1      |CCD  |X05  |2026-03-15T06:00:00.000Z    |"
    "166.15776059 | +5.72957795 | 20.00 |V    |Gaia3\n"
    "       |       |T1      |CCD  |X05  |2026-03-15T06:30:00.000Z    |"
    "166.20000000 | +5.75000000 | 20.10 |V    |Gaia3\n"
)

_MPC80 = (
    "K26A00A       C2026 03 15.25000011 04 37.863+05 43 46.48         "
    "19.54r      703\n"
    "K26A00A       C2026 03 15.26388911 04 44.529+05 44 21.48         "
    "19.54r      703\n"
)


def test_detect_format(tmp_path):
    psv = tmp_path / "a.txt"
    psv.write_text(_PSV)
    obs = tmp_path / "b.psv"          # deliberately misleading suffix
    obs.write_text(_MPC80)
    assert G.detect_format(str(psv)) == "ades_psv"
    assert G.detect_format(str(obs)) == "mpc80", "content wins over the suffix"


def test_read_lines_strips_a_utf8_bom(tmp_path):
    """A BOM must not land on the first character of the ADES header.

    Files exported by Windows tooling routinely carry one; without utf-8-sig
    the header column reads ``\ufeffpermID``, the required-column check fails,
    and a perfectly good file is rejected outright.
    """
    path = tmp_path / "bom.psv"
    path.write_bytes(b"\xef\xbb\xbf" + _PSV.encode("utf-8"))
    assert G.detect_format(str(path)) == "ades_psv"
    dets, issues, _ = G.parse_observations(str(path))
    assert issues == []
    assert len(dets) == 2 and dets[0].object_id == "T1"


def test_read_lines_strips_a_bom_on_the_80_column_path(tmp_path):
    path = tmp_path / "bom.obs"
    path.write_bytes(b"\xef\xbb\xbf" + _MPC80.encode("utf-8"))
    assert G.detect_format(str(path)) == "mpc80"
    dets, issues, _ = G.parse_observations(str(path))
    assert issues == [], issues
    assert len(dets) == 2, "a BOM must not shift the fixed columns"
    assert dets[0].object_id == "K26A00A"


def test_detect_format_does_not_spend_its_budget_on_comments(tmp_path):
    """A long ``#`` preamble must not push the header out of the window."""
    path = tmp_path / "preamble.psv"
    path.write_text("\n".join(["# provenance line"] * 500) + "\n" + _PSV)
    assert G.detect_format(str(path)) == "ades_psv"
    dets, _issues, _ = G.parse_observations(str(path))
    assert len(dets) == 2


def test_detect_format_still_gives_up_on_200_data_lines(tmp_path):
    path = tmp_path / "junk.dat"
    path.write_text("\n".join(["nonsense line"] * 400) + "\n")
    with pytest.raises(ValueError, match="cannot tell whether"):
        G.detect_format(str(path))


def test_detect_format_refuses_garbage_with_a_useful_message(tmp_path):
    path = tmp_path / "x.dat"
    path.write_text("this is not astrometry\nnor is this\n")
    with pytest.raises(ValueError, match="cannot tell whether"):
        G.detect_format(str(path))
    with pytest.raises(ValueError, match="not astrometry"):
        G.detect_format(str(path))    # quotes the offending line


# --------------------------------------------------------------------------
# ADES PSV parsing
# --------------------------------------------------------------------------


def test_parse_ades_psv_basics(tmp_path):
    path = tmp_path / "a.psv"
    path.write_text(_PSV)
    dets, issues, fmt = G.parse_observations(str(path))
    assert fmt == "ades_psv" and issues == []
    assert [d.object_id for d in dets] == ["T1", "T1"]
    assert dets[0].obscode == "X05"
    assert dets[0].ra_deg == pytest.approx(166.15776059)
    assert dets[0].dec_deg == pytest.approx(5.72957795)
    assert dets[0].v_mag == pytest.approx(20.00)
    assert dets[0].jd_utc == pytest.approx(
        Time("2026-03-15T06:00:00.000", scale="utc").jd, abs=1e-9
    )
    assert dets[0].line == 3, "line numbers are 1-based into the file"


def test_parse_ades_psv_accepts_any_column_order_and_subset(tmp_path):
    path = tmp_path / "a.psv"
    path.write_text(
        "dec |ra |obsTime |stn |trkSub\n"
        " +5.0 | 166.0 | 2026-03-15T06:00:00Z | X05 | T9\n"
    )
    dets, issues, _ = G.parse_observations(str(path))
    assert issues == []
    assert len(dets) == 1 and dets[0].object_id == "T9"
    assert dets[0].dec_deg == pytest.approx(5.0)
    assert np.isnan(dets[0].mag), "no mag column means no photometry"


def test_parse_ades_psv_object_id_precedence(tmp_path):
    path = tmp_path / "a.psv"
    path.write_text(
        "permID |provID |trkSub |stn |obsTime |ra |dec\n"
        "12345  |2026 AA |TRK1  |X05 |2026-03-15T06:00:00Z |1.0 |2.0\n"
        "12345  |2026 AA |       |X05 |2026-03-15T06:01:00Z |1.0 |2.0\n"
        "12345  |        |       |X05 |2026-03-15T06:02:00Z |1.0 |2.0\n"
    )
    dets, _, _ = G.parse_observations(str(path))
    assert [d.object_id for d in dets] == ["TRK1", "2026 AA", "12345"]


def test_parse_ades_psv_reports_bad_records_and_keeps_going(tmp_path):
    path = tmp_path / "a.psv"
    path.write_text(
        "trkSub |stn |obsTime |ra |dec |mag |band\n"
        "T1 |X05 |2026-03-15T06:00:00Z |166.0 |5.0 |20.0 |V\n"
        "T1 |X05 |not-a-time           |166.0 |5.0 |20.0 |V\n"
        "T1 |X05 |2026-03-15T06:10:00Z |wat   |5.0 |20.0 |V\n"
        "T1 |X05 |2026-03-15T06:20:00Z |166.0 |99.0 |20.0 |V\n"
        "T1 |X05 |2026-03-15T06:30:00Z |166.0 |5.0 |oops |V\n"
        "   |X05 |2026-03-15T06:40:00Z |166.0 |5.0 |20.0 |V\n"
        "T1 |X05 |2026-03-15T06:50:00Z |166.1 |5.1 |20.2 |V\n"
    )
    dets, issues, _ = G.parse_observations(str(path))
    reasons = " | ".join(i.reason for i in issues)
    assert "obsTime" in reasons and "ra/dec" in reasons
    assert "out of range" in reasons and "no object id" in reasons
    assert "magnitude" in reasons or "mag" in reasons
    # The unparseable-magnitude row survives (as no photometry); the four
    # structurally broken ones do not.
    assert len(dets) == 3, [d.line for d in dets]
    assert np.isnan([d for d in dets if d.line == 6][0].mag)


def test_parse_ades_psv_reports_short_and_long_rows(tmp_path):
    """Padding or truncating a row silently is how a shifted file scores clean."""
    path = tmp_path / "a.psv"
    path.write_text(
        "trkSub |stn |obsTime |ra |dec |mag |band\n"
        "T1 |X05 |2026-03-15T06:00:00Z |166.0 |5.0 |20.0 |V\n"
        "T1 |X05 |2026-03-15T06:10:00Z |166.1 |5.1\n"
        "T1 |X05 |2026-03-15T06:20:00Z |166.2 |5.2 |20.0 |V |junk\n"
    )
    dets, issues, _ = G.parse_observations(str(path))
    assert len(dets) == 3, "all three rows are still usable"
    assert [i.line for i in issues] == [3, 4]
    assert "5 fields, header has 7" in issues[0].reason
    assert "8 fields, header has 7" in issues[1].reason
    assert np.isnan(dets[1].mag), "the short row's missing mag is empty, not junk"


def test_parse_ades_psv_rejects_a_header_without_required_columns(tmp_path):
    path = tmp_path / "a.psv"
    path.write_text("trkSub |stn |mag\nT1 |X05 |20.0\n")
    with pytest.raises(ValueError, match="missing required column"):
        G.parse_ades_psv(str(path))


# --------------------------------------------------------------------------
# MPC 80-column parsing
# --------------------------------------------------------------------------


def test_parse_mpc80_fields(tmp_path):
    path = tmp_path / "a.obs"
    path.write_text(_MPC80)
    dets, issues, fmt = G.parse_observations(str(path))
    assert fmt == "mpc80" and issues == []
    d = dets[0]
    assert d.object_id == "K26A00A", "columns 1-12 verbatim, not unpacked"
    assert d.obscode == "703"
    assert d.band == "r"
    assert d.mag == pytest.approx(19.54)
    assert d.v_mag == pytest.approx(19.54 + 0.23)
    assert d.ra_deg == pytest.approx(15.0 * (11 + 4 / 60 + 37.863 / 3600))
    assert d.dec_deg == pytest.approx(5 + 43 / 60 + 46.48 / 3600)
    assert d.jd_utc == pytest.approx(
        Time("2026-03-15T00:00:00", scale="utc").jd + 0.250000, abs=1e-9
    )


def test_parse_mpc80_negative_declination_and_blank_magnitude(tmp_path):
    line = (
        "K26A00A       C2026 03 15.25000011 04 37.863-05 43 46.48         "
        "            703"
    )
    assert len(line) == 80
    path = tmp_path / "a.obs"
    path.write_text(line + "\n" + line + "\n")
    dets, issues, _ = G.parse_observations(str(path))
    assert issues == []
    assert dets[0].dec_deg == pytest.approx(-(5 + 43 / 60 + 46.48 / 3600))
    assert np.isnan(dets[0].mag) and np.isnan(dets[0].v_mag)


_MPC80_GOOD = (
    "K26A00A       C2026 03 15.25000011 04 37.863+05 43 46.48         "
    "19.54r      703"
)


def test_blank_band_80_column_is_read_as_V_and_counted(tmp_path):
    """D-20: the one deliberate photometric divergence from digest2.

    digest2's ADES reader substitutes V for a missing band; its 80-column
    reader hands the raw blank to the offset table, where it misses and takes
    the -0.8 unknown-band fallback.  pneo applies the ADES rule on both
    paths, so it reads 0.80 mag FAINTER (19.54, not 18.74) on a blank-band
    80-column record.
    That is a choice, and this test pins both halves of it: the value, and the
    fact that it is announced rather than silent.
    """
    blank = _MPC80_GOOD[:70] + " " + _MPC80_GOOD[71:]
    assert len(blank) == 80
    path = tmp_path / "a.obs"
    path.write_text(_MPC80_GOOD + "\n" + blank + "\n")

    with pytest.warns(UserWarning, match="BLANK band") as caught:
        dets, issues = G.parse_mpc80(str(path))
    assert issues == []
    assert dets[0].v_mag == pytest.approx(19.54 + 0.23), "named band 'r'"
    assert dets[1].v_mag == pytest.approx(19.54), "blank band -> V, offset 0"
    # NOT digest2's 80-column answer, and deliberately so.
    assert dets[1].v_mag != pytest.approx(19.54 - 0.8)
    text = str(caught[0].message)
    assert "1 MPC 80-column record" in text
    assert "-0.8" in text and "D-20" in text


def test_blank_band_with_no_magnitude_is_not_counted(tmp_path):
    """Nothing diverges when there is no photometry to convert."""
    nomag = _MPC80_GOOD[:65] + " " * 6 + _MPC80_GOOD[71:]
    assert len(nomag) == 80
    path = tmp_path / "a.obs"
    path.write_text(nomag + "\n" + nomag + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # no warning may be raised
        dets, _issues = G.parse_mpc80(str(path))
    assert all(np.isnan(d.v_mag) for d in dets)


def test_ades_blank_band_matches_digest2_exactly(tmp_path):
    """On the ADES path there is NO divergence: digest2 also substitutes V."""
    path = tmp_path / "a.psv"
    path.write_text(
        "trkSub |stn |obsTime |ra |dec |mag |band\n"
        "T1 |X05 |2026-03-15T06:00:00Z |166.0 |5.0 |20.0 |\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dets, _issues, _ = G.parse_observations(str(path))
    assert dets[0].v_mag == pytest.approx(20.0)


def test_parse_mpc80_impossible_date_costs_the_record_not_the_file(tmp_path):
    """An out-of-range month must be a ParseIssue, not an uncaught ValueError."""
    good = (
        "K26A00A       C2026 03 15.25000011 04 37.863+05 43 46.48         "
        "19.54r      703"
    )
    bad = good[:20] + "13" + good[22:]
    path = tmp_path / "a.obs"
    path.write_text("\n".join([good, bad, good]) + "\n")
    dets, issues = G.parse_mpc80(str(path))
    assert len(dets) == 2
    assert [i.line for i in issues] == [2]


@pytest.mark.parametrize("note2", list("SsVvRr"))
def test_parse_mpc80_skips_every_non_optical_record_type(tmp_path, note2):
    """Space-based, roving-observer and radar records hold no usable RA/Dec.

    Each of these would *parse* as astrometry rather than fail, so silently
    accepting one invents a detection at a position the object was never at.
    """
    good = (
        "K26A00A       C2026 03 15.25000011 04 37.863+05 43 46.48         "
        "19.54r      703"
    )
    path = tmp_path / "a.obs"
    path.write_text(good + "\n" + good[:14] + note2 + good[15:] + "\n")
    dets, issues = G.parse_mpc80(str(path))
    assert len(dets) == 1
    assert "not ground-based optical" in issues[0].reason


def test_parse_mpc80_skips_space_based_and_short_records(tmp_path):
    good = (
        "K26A00A       C2026 03 15.25000011 04 37.863+05 43 46.48         "
        "19.54r      703"
    )
    space = good[:14] + "S" + good[15:]
    path = tmp_path / "a.obs"
    path.write_text(good + "\n" + space + "\n" + good[:60] + "\n")
    dets, issues, _ = G.parse_observations(str(path))
    assert len(dets) == 1
    reasons = " | ".join(i.reason for i in issues)
    assert "not ground-based optical" in reasons
    assert "expected 80" in reasons


# --------------------------------------------------------------------------
# The attributable fit
# --------------------------------------------------------------------------


def test_fit_attributable_recovers_an_exact_line():
    t = np.array([0.0, 0.01, 0.02, 0.03]) + 2461000.0
    ra = 10.0 + 1.5 * (t - t[0])
    dec = -20.0 - 0.5 * (t - t[0])
    fit = G.fit_attributable(t, ra, dec)
    assert fit["epoch_jd_utc"] == t[0]
    assert fit["ra"] == pytest.approx(np.radians(10.0))
    assert fit["dec"] == pytest.approx(np.radians(-20.0))
    assert fit["ra_rate"] == pytest.approx(np.radians(1.5))
    assert fit["dec_rate"] == pytest.approx(np.radians(-0.5))
    assert fit["resid_rms_arcsec"] == pytest.approx(0.0, abs=1e-6)
    assert fit["arc_hours"] == pytest.approx(0.03 * 24.0)
    assert fit["n_det"] == 4


def test_fit_attributable_unwraps_ra_across_zero():
    t = np.array([0.0, 0.01, 0.02]) + 2461000.0
    ra = np.array([359.9, 0.0, 0.1])
    fit = G.fit_attributable(t, ra, np.zeros(3))
    assert fit["ra_rate"] == pytest.approx(np.radians(10.0), rel=1e-9)
    assert fit["ra"] == pytest.approx(np.radians(359.9), rel=1e-9)


def test_fit_attributable_is_order_insensitive():
    t = np.array([0.02, 0.0, 0.01]) + 2461000.0
    ra = 10.0 + 1.5 * (t - 2461000.0)
    a = G.fit_attributable(t, ra, np.zeros(3))
    order = np.argsort(t)
    b = G.fit_attributable(t[order], ra[order], np.zeros(3))
    assert a == b


def test_fit_attributable_rejects_a_single_detection():
    with pytest.raises(ValueError, match="at least 2"):
        G.fit_attributable(np.array([1.0]), np.array([1.0]), np.array([1.0]))


def test_fit_attributable_zero_length_arc_gives_nan_rates():
    fit = G.fit_attributable(
        np.array([2461000.0, 2461000.0]), np.array([10.0, 10.0]),
        np.array([1.0, 1.0]),
    )
    assert np.isnan(fit["ra_rate"]) and np.isnan(fit["dec_rate"])
    assert fit["arc_hours"] == 0.0


def test_fit_attributable_residual_is_zero_for_two_detections():
    """Documented behaviour: ~0, not NaN -- "nothing to check", not "perfect"."""
    t = np.array([0.0, 0.02]) + 2461000.0
    fit = G.fit_attributable(t, np.array([10.0, 10.03]), np.array([1.0, 0.99]))
    assert fit["resid_rms_arcsec"] == pytest.approx(0.0, abs=1e-9)
    assert not np.isnan(fit["resid_rms_arcsec"])
    assert "nothing to check" in G.fit_attributable.__doc__


def test_fit_attributable_residual_flags_a_bad_detection():
    t = np.array([0.0, 0.01, 0.02, 0.03]) + 2461000.0
    ra = 10.0 + 1.5 * (t - t[0])
    dec = np.zeros(4)
    clean = G.fit_attributable(t, ra, dec)
    ra_bad = ra.copy()
    ra_bad[2] += 2.0 / 3600.0          # a 2 arcsec blunder
    dirty = G.fit_attributable(t, ra_bad, dec)
    assert clean["resid_rms_arcsec"] < 1e-6
    assert dirty["resid_rms_arcsec"] > 0.4


def _kepler_tracklet(arc_minutes: float, n: int = 5, obscode: str = "X05",
                     a: float = 1.6, e: float = 0.32, inc: float = 0.15,
                     H: float = 22.0):
    """A two-body arc sampled n times, plus the truth at the FIRST epoch.

    Same construction as the pilot's own round-trip test: the generator is
    ``observables_from_elements``, so the only thing the fit can be wrong about
    is the fit.
    """
    jd0 = Time("2026-03-15T06:00:00", scale="utc").jd
    dt = np.linspace(0.0, arc_minutes / (60.0 * 24.0), n)
    jd = jd0 + dt
    Om, om, M0 = 0.7, 1.1, 0.3
    mean_motion = K_GAUSS / a**1.5
    el = np.column_stack([
        np.full(n, a), np.full(n, e), np.full(n, inc),
        np.full(n, Om), np.full(n, om), M0 + mean_motion * dt,
    ])
    epochs = Time(jd, format="jd", scale="utc")
    R, V = observer_state(epochs, obscode=obscode)
    obs = T.observables_from_elements(el, np.full(n, H), R, V)
    R0, V0 = observer_state(epochs[0], obscode=obscode)
    truth = T.observables_from_elements(
        el[:1], np.array([H]), R0[None, :], V0[None, :]
    )
    return jd, obs, {k: float(v[0]) for k, v in truth.items()}


def test_fit_roundtrip_against_observables_from_elements():
    """Keplerian tracklet -> fit -> compare with the generator's truth.

    Tolerances are set by *curvature*, not numerics: a straight-line fit of a
    curved arc cannot do better.  Over a 30-minute arc -- the scale of a real
    discovery tracklet -- the fitted position at the reference epoch is within
    20 mas, d(alpha)/dt within 0.2% and d(delta)/dt within 0.5%; the next test
    shows those numbers ARE the curvature by checking how they scale.
    """
    jd, obs, truth = _kepler_tracklet(30.0)
    fit = G.fit_attributable(jd, np.degrees(obs["ra"]), np.degrees(obs["dec"]))
    dra = (fit["ra"] - truth["ra"]) * np.cos(truth["dec"]) * ARCSEC
    ddec = (fit["dec"] - truth["dec"]) * ARCSEC
    assert abs(dra) < 0.02, f"RA off by {dra} arcsec"
    assert abs(ddec) < 0.02, f"Dec off by {ddec} arcsec"
    assert fit["ra_rate"] == pytest.approx(truth["ra_rate"], rel=2e-3)
    assert fit["dec_rate"] == pytest.approx(truth["dec_rate"], rel=5e-3)
    # A pure two-body arc has no scatter about its own smooth curve, so the
    # residual RMS measures only the curvature the straight line cannot follow.
    assert fit["resid_rms_arcsec"] < 0.02


def test_fit_roundtrip_error_is_curvature_not_a_bug():
    """Halving the arc must quarter the position error and halve the rate error.

    That scaling is the signature of a correct linear estimator applied to a
    quadratic truth; any indexing, epoch or unit error would break the powers.
    """
    pos_err, rate_err = [], []
    for arc in (60.0, 30.0, 15.0):
        jd, obs, truth = _kepler_tracklet(arc)
        fit = G.fit_attributable(
            jd, np.degrees(obs["ra"]), np.degrees(obs["dec"])
        )
        pos_err.append(
            abs(fit["ra"] - truth["ra"]) * np.cos(truth["dec"]) * ARCSEC
        )
        rate_err.append(abs(fit["ra_rate"] / truth["ra_rate"] - 1.0))
    for k in range(2):
        assert pos_err[k] / pos_err[k + 1] == pytest.approx(4.0, rel=0.25)
        assert rate_err[k] / rate_err[k + 1] == pytest.approx(2.0, rel=0.15)


# --------------------------------------------------------------------------
# Night clustering and grouping
# --------------------------------------------------------------------------


def test_cluster_nights_splits_on_the_gap_only():
    t = np.array([0.0, 0.01, 0.02, 1.0, 1.01, 5.0])
    groups = G.cluster_nights(t, gap_hours=3.0)
    assert [sorted(g.tolist()) for g in groups] == [[0, 1, 2], [3, 4], [5]]


def test_cluster_nights_sorts_unsorted_input():
    t = np.array([1.0, 0.0, 0.01])
    groups = G.cluster_nights(t, gap_hours=3.0)
    assert [g.tolist() for g in groups] == [[1, 2], [0]]


def _det(obj, jd, obscode="X05", ra=166.0, dec=5.0, mag=20.0, band="V"):
    return G.Detection(
        object_id=obj, obscode=obscode, jd_utc=jd, ra_deg=ra, dec_deg=dec,
        mag=mag, band=band, v_mag=G.band_to_v(band, mag), source="x", line=0,
    )


def test_group_takes_the_earliest_same_night_same_station_set():
    """Two nights and two stations: the earliest 2+ detection set wins."""
    t0 = 2461000.0
    dets = (
        # night 2 at X05 (later)
        [_det("A", t0 + 1.0 + i * 0.005, "X05", ra=166.0 + i * 0.01)
         for i in range(3)]
        # night 1 at F51 (earlier) -- this is the tracklet
        + [_det("A", t0 + i * 0.005, "F51", ra=100.0 + i * 0.01)
           for i in range(2)]
    )
    table = G.ObscodeTable(search=False)
    tracklets, failures = G.group_tracklets(dets, obscodes=table)
    assert failures == []
    (tr,) = tracklets
    assert tr.obscode == "F51"
    assert tr.n_obs == 2
    assert tr.epoch_jd_utc == pytest.approx(t0)


def test_group_never_mixes_two_stations_into_one_tracklet():
    t0 = 2461000.0
    dets = [
        _det("A", t0, "X05"),
        _det("A", t0 + 0.001, "F51"),
        _det("A", t0 + 0.002, "F51"),
    ]
    tracklets, failures = G.group_tracklets(
        dets, obscodes=G.ObscodeTable(search=False)
    )
    assert failures == []
    (tr,) = tracklets
    assert tr.obscode == "F51" and tr.n_obs == 2


def test_group_truncates_an_overlong_night():
    t0 = 2461000.0
    dets = [_det("A", t0 + i * 0.05, ra=166.0 + i * 0.01) for i in range(5)]
    tracklets, _ = G.group_tracklets(
        dets, obscodes=G.ObscodeTable(search=False), max_arc_hours=2.0,
    )
    (tr,) = tracklets
    assert tr.truncated_arc is True
    assert tr.n_obs == 2, "0.05 d = 1.2 h steps, so only two fit in 2 h"
    assert tr.arc_hours <= 2.0
    assert tr.n_detections_seen == 5


def test_group_keeps_two_detections_even_when_the_arc_limit_forbids_it():
    """max_arc_hours must never produce an unfittable one-detection tracklet."""
    t0 = 2461000.0
    dets = [_det("A", t0, ra=166.0), _det("A", t0 + 0.05, ra=166.01)]
    tracklets, failures = G.group_tracklets(
        dets, obscodes=G.ObscodeTable(search=False), max_arc_hours=0.1,
    )
    assert failures == []
    (tr,) = tracklets
    assert tr.n_obs == 2 and tr.truncated_arc is True


def test_group_failures_are_per_object_and_named():
    t0 = 2461000.0
    dets = (
        [_det("LONE", t0)]
        + [_det("NOMAG", t0 + i * 0.001, mag=float("nan")) for i in range(2)]
        + [_det("SAME", t0) for _ in range(2)]
        + [_det("GOOD", t0 + i * 0.001, ra=166.0 + i * 0.01) for i in range(2)]
    )
    tracklets, failures = G.group_tracklets(
        dets, obscodes=G.ObscodeTable(search=False)
    )
    assert [t.object_id for t in tracklets] == ["GOOD"]
    got = {f.object_id: f.status for f in failures}
    assert got == {
        "LONE": "single_detection",
        "NOMAG": "no_magnitude",
        "SAME": "fit_failed",
    }
    assert all(f.message for f in failures)
    # `seq` restores input order across the two lists.
    order = sorted(list(tracklets) + list(failures), key=lambda x: x.seq)
    assert [x.object_id for x in order] == ["LONE", "NOMAG", "SAME", "GOOD"]


def test_group_takes_the_median_magnitude_and_records_the_bands():
    t0 = 2461000.0
    dets = [
        _det("A", t0, ra=166.00, mag=20.0, band="V"),
        _det("A", t0 + 0.001, ra=166.01, mag=19.77, band="r"),  # -> 20.0
        _det("A", t0 + 0.002, ra=166.02, mag=25.0, band="V"),   # outlier
    ]
    tracklets, _ = G.group_tracklets(
        dets, obscodes=G.ObscodeTable(search=False)
    )
    (tr,) = tracklets
    assert tr.v_mag == pytest.approx(20.0, abs=1e-9), "median, not mean"
    assert tr.bands == ("V", "r")


def test_ingest_groups_one_object_across_two_files(tmp_path):
    a = tmp_path / "a.psv"
    b = tmp_path / "b.psv"
    head = "trkSub |stn |obsTime |ra |dec |mag |band\n"
    a.write_text(head + "T1 |X05 |2026-03-15T06:00:00Z |166.00 |5.0 |20.0 |V\n")
    b.write_text(head + "T1 |X05 |2026-03-15T06:30:00Z |166.02 |5.1 |20.0 |V\n")
    tracklets, failures, issues = G.ingest(
        [str(a), str(b)], obscodes=G.ObscodeTable(search=False)
    )
    assert issues == [] and failures == []
    (tr,) = tracklets
    assert tr.n_obs == 2


def test_ingest_reports_an_unreadable_file_and_keeps_going(tmp_path):
    good = tmp_path / "a.psv"
    good.write_text(_PSV)
    tracklets, _failures, issues = G.ingest(
        [str(tmp_path / "missing.psv"), str(good)],
        obscodes=G.ObscodeTable(search=False),
    )
    assert len(tracklets) == 1
    assert any("missing.psv" in i.source for i in issues)


def test_tracklet_observer_kwargs_distinguishes_the_geocentre():
    """A fallback tracklet must be scored AT the geocentre, not at (0, 0, 0).

    ``observer_state(lon_deg=0, lat_deg=0)`` is Greenwich on the equator — one
    Earth radius (4.3e-5 AU) from the geocentre, which is exactly the scale
    that matters for the close fast objects this tool triages.
    """
    common = dict(
        object_id="A", obscode="ZZZ", n_obs=2, arc_hours=1.0,
        epoch_jd_utc=2461000.0, ra=1.0, dec=0.1, ra_rate=0.01,
        dec_rate=0.001, v_mag=20.0, resid_rms_arcsec=0.1,
        lon_deg=0.0, lat_deg=0.0, height_m=0.0,
    )
    for source in ("geocenter", "fallback"):
        tr = G.Tracklet(obscode_source=source, **common)
        assert tr.is_geocentric and tr.observer_kwargs() == {}
    tr = G.Tracklet(**{**common, "obscode_source": "builtin",
                       "lon_deg": 289.25, "lat_deg": -30.24,
                       "height_m": 2647.0})
    assert not tr.is_geocentric
    assert tr.observer_kwargs() == {
        "lon_deg": 289.25, "lat_deg": -30.24, "height_m": 2647.0
    }


def test_tracklet_attributable_and_epoch_string():
    tr = G.Tracklet(
        object_id="A", obscode="X05", n_obs=2, arc_hours=1.0,
        epoch_jd_utc=float(Time("2026-03-15T06:00:00", scale="utc").jd),
        ra=1.0, dec=0.1, ra_rate=0.01, dec_rate=0.001, v_mag=20.0,
        resid_rms_arcsec=0.1, lon_deg=1.0, lat_deg=2.0, height_m=3.0,
        obscode_source="builtin",
    )
    assert tr.epoch_isot.startswith("2026-03-15T06:00:00")
    att = tr.attributable()
    assert att.ra == 1.0 and att.v_mag == 20.0
    assert att.epoch.scale == "utc"


