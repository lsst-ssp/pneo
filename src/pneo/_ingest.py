"""Observation-file ingestion: detections -> tracklets -> attributables.

This module turns the two observation formats a digest2 user already has on
disk — **ADES PSV** and **MPC 80-column** — into the
:class:`~pneo._attributable.Attributable` values
:func:`pneo.classify` consumes.  It is the ingestion half of the
``pneo`` command-line tool (:mod:`pneo.cli`) and is a supported public
API in its own right.

Provenance
----------
Every rule below is **promoted verbatim from** ``validation/neocp_pilot.py``
(WP8, DESIGN §4.3), where it was built and validated against 500 real MPC
NEOCP discovery tracklets: the band -> V table (cross-checked against digest2
2.8.2's own ``_BAND_CORRECTIONS``), the two-character ADES band aliases
(verified against the MPC by requesting ``ADES_DF`` and ``OBS80`` for the same
objects), the night clustering, the least-squares attributable fit, and the
MPC-parallax-constants -> WGS84 obscode resolution.  The pilot is a repo-only
validation harness, so the code is **copied, not imported**: an installed
``pneo`` wheel carries no ``validation/`` package.
``tests/test_ingest.py`` pins the copy against the pilot's own stored output
(``validation/output/neocp_pilot/tracklets.json``) where that file is present.

What ingestion does, in order
-----------------------------
1. **Parse** (:func:`parse_observations`).  Format is auto-detected from the
   file's content (:func:`detect_format`) and can be forced.  Bad records are
   skipped individually and reported; a malformed line never costs the file.
2. **Group** (:func:`group_tracklets`).  Detections are grouped by object id,
   then by station, then split into nights on any gap longer than
   ``night_gap_hours``; the **earliest** such (station, night) set carrying at
   least two detections becomes the tracklet, truncated to the first
   ``max_arc_hours`` of arc.  This is the pilot's rule minus its NEOCP-specific
   "discovery station / posting window" selection, which has no counterpart in
   a user's own file.
3. **Fit** (:func:`fit_attributable`).  Ordinary least squares straight lines
   in alpha(t) and delta(t), evaluated **at the first detection's epoch** — the
   WP6/WP7 convention, and also what digest2 anchors on (``obsPair[0]``).  RA is
   unwrapped about the first detection, so a tracklet straddling alpha = 0 fits
   correctly.
4. **Photometry** (:func:`band_to_v`).  Per-detection band -> Johnson V using
   digest2's own offsets, then the **median** over the tracklet.

Two deliberate divergences from digest2, both on the MPC 80-column path, both
stated here so nobody has to discover them from a diff:

* **A blank band is read as V, not as "unknown"** [D-20].  digest2's own
  80-column reader gives a blank band its -0.8 unknown-band fallback (its ADES
  reader substitutes V), so pneo's V is 0.80 mag fainter (numerically larger) on such a record.
  See :func:`normalize_band` for the argument; :func:`parse_mpc80` warns once
  per file whenever it happens, and the CLI repeats the count on stderr.
* **A wider set of observation types is accepted.**  digest2 keeps only
  note-2 ``C``/``S``/``B``; pneo keeps everything *except* the record types
  whose columns do not hold optical RA/Dec at all (see
  :data:`_MPC80_SKIP_NOTE2`) — photographic, transit, encoder, video and
  occultation records are therefore scored here and skipped there.  Their
  astrometry is real, if less precise; the attributable fit's
  ``resid_rms_arcsec`` is the honest per-tracklet quality flag for them.
5. **Observatory** (:class:`ObscodeTable`).  The built-in table
   (:data:`pneo._attributable.MPC_OBSCODES`) first, then a *local* copy of
   the MPC "List Of Observatory Codes" page if one is configured or found, then
   a geocentric fallback with a warning.  **Nothing here touches the network**:
   an obscode lookup must not be able to stall or fail a scoring run.

Degradation is always per tracklet.  Ingestion returns
``(tracklets, failures)`` and never raises for bad data; only an unreadable
file or an unrecognisable format raises.
"""

from __future__ import annotations

import html as _html
import os
import re
import warnings
from dataclasses import dataclass

import numpy as np

from .paths import default_obscodes_path
from ._attributable import (
    GEOCENTER_OBSCODES,
    MPC_EARTH_RADIUS_M,
    MPC_OBSCODES,
    Attributable,
)

__all__ = [
    "BAND_TO_V",
    "BAND_TO_V_DEFAULT",
    "MPC_BAND_ALIASES",
    "DEFAULT_NIGHT_GAP_HOURS",
    "DEFAULT_MAX_ARC_HOURS",
    "FORMATS",
    "OBSCODE_ENV_VAR",
    "URL_OBSCODES",
    "Detection",
    "Tracklet",
    "IngestFailure",
    "ParseIssue",
    "ObscodeTable",
    "band_to_v",
    "normalize_band",
    "cluster_nights",
    "detect_format",
    "fetch_obscodes",
    "fit_attributable",
    "group_tracklets",
    "ingest",
    "ingest_file",
    "obscode_geodetic",
    "parse_ades_psv",
    "parse_mpc80",
    "parse_obscodes",
    "parse_observations",
]


# --------------------------------------------------------------------------
# Photometry: band -> V, digest2's own table
# --------------------------------------------------------------------------

#: Band -> Johnson V offsets.  **Verbatim** from digest2 2.8.2's own table,
#: ``digest2/observation.py:_BAND_CORRECTIONS`` ("same as common.c
#: updateMagnitude"), so pneo and digest2 standardise photometry
#: identically **for every band that is actually named** — see
#: :func:`normalize_band` for the one deliberate divergence, a *blank* band in
#: an MPC 80-column record [D-20].  digest2 applies the offset only when the
#: reported magnitude is
#: > 0 and uses :data:`BAND_TO_V_DEFAULT` for any band not listed;
#: :func:`band_to_v` reproduces both rules.
BAND_TO_V: dict[str, float] = {
    "V": 0.0,
    "B": -0.8,
    "U": -1.3,
    "g": -0.28,
    "r": 0.23,
    "R": 0.4,
    "C": 0.4,
    "W": 0.4,
    "i": 0.39,
    "z": 0.37,
    "I": 0.8,
    "J": 1.2,
    "w": -0.16,
    "y": 0.36,
    "L": 0.2,
    "H": 1.4,
    "K": 1.7,
    "Y": 0.7,
    "G": 0.24,
    "v": 0.0,
    "c": -0.05,
    "o": 0.33,
    "u": 2.5,
}

#: digest2's fallback offset for an unrecognised band (``.get(band, -0.8)``).
BAND_TO_V_DEFAULT = -0.8

#: **Real-data finding (WP8).** The MPC observations API's ADES dataframe — and
#: some ADES PSV files derived from it — report the photometric band as a
#: *two-character* code for modern surveys (`Pw`, `Ao`, `Lr`, `Vj`), where
#: MPC 80-column and hand-written ADES carry a single character.  digest2's
#: table is keyed on the single character, so a naive lookup sends 2-char codes
#: to the -0.8 "unknown band" fallback: a 0.6-1.0 mag photometric error, and on
#: the NEOCP pilot's sample it would have hit 593 detections silently.
#:
#: The mapping was verified against the MPC itself (``ADES_DF`` vs ``OBS80`` for
#: the same objects: `Pw`->`w` 105/105 on 2017 UQ124; `Ac`->`c` 38/38 and
#: `Ao`->`o` 32/32 on 2006 UW157; `Lr`->`r`, `Lz`->`z` 6/6).  The order is
#: **not** uniform: the survey codes are (system, band) with an uppercase
#: system prefix, while the Johnson/Cousins codes are (band, system) with a
#: lowercase system suffix — `Vj` is Johnson V, and mechanically taking the
#: second character would read it as J band (+1.2 mag).  Hence an explicit
#: table rather than a slicing rule.
#:
#: ``UNK`` is deliberately absent: it is a *present* band value that digest2
#: does not know, so it must keep taking the -0.8 fallback, exactly as digest2
#: would on the same input.
MPC_BAND_ALIASES: dict[str, str] = {
    # Pan-STARRS (F51, F52, and the O18/W* mirrors): grizy + the broad w
    "Pg": "g", "Pr": "r", "Pi": "i", "Pz": "z", "Py": "y", "Pw": "w",
    # ATLAS (T05, T08, M22, W68): cyan and orange
    "Ac": "c", "Ao": "o",
    # Rubin / LSST (X05, Simonyi Survey Telescope): ugrizy
    "Lu": "u", "Lg": "g", "Lr": "r", "Li": "i", "Lz": "z", "Ly": "y",
    # SDSS-like ugriz reported with a 'G' system prefix
    "Gu": "u", "Gg": "g", "Gr": "r", "Gi": "i", "Gz": "z",
    # Johnson (j) / Cousins (c) systems: band first, system second
    "Uj": "U", "Bj": "B", "Vj": "V", "Rj": "R", "Ij": "I",
    "Rc": "R", "Ic": "I",
}


def normalize_band(band: str | None) -> str:
    """MPC/ADES band code -> the single character digest2's table is keyed on.

    Missing/empty becomes ``"V"``.  Two-character survey codes are resolved
    through :data:`MPC_BAND_ALIASES`; anything else is passed through
    unchanged, so a genuinely unknown band still takes digest2's -0.8 fallback
    rather than being silently invented.

    .. warning::
       **Blank band: a deliberate divergence from digest2 on the 80-column
       path** [D-20].  digest2 treats a missing band differently in its two
       parsers: its ADES reader substitutes ``"V"``
       (``row.get("band", "V") or "V"``, offset 0.0), while its MPC 80-column
       reader hands the raw blank straight to the offset table, where it misses
       and takes the **-0.8 "unknown band" fallback**.  pneo applies the
       ADES rule on **both** paths, so on a blank-band 80-column record
       pneo's V is **0.80 mag fainter** (numerically larger) than digest2's.

       This is a choice, not an oversight.  A blank band in pre-ADES astrometry
       overwhelmingly means an unfiltered or V-ish visual/CCD magnitude; -0.8 is
       digest2's crudest fallback, applied there only as an artifact of which
       parser ran.  Treating the two formats differently would also make
       pneo's answer depend on the file format rather than the observation.
       :func:`parse_mpc80` counts these records and warns once per file, so the
       divergence is never silent.
    """
    if band is None:
        return "V"
    key = str(band).strip()
    if key == "":
        return "V"
    return MPC_BAND_ALIASES.get(key, key)


def band_to_v(band: str | None, mag: float) -> float:
    """Convert a reported magnitude in ``band`` to Johnson V, digest2's way.

    Reproduces ``digest2/observation.py:_update_magnitude`` exactly: the offset
    from :data:`BAND_TO_V` (default :data:`BAND_TO_V_DEFAULT` for an unknown
    band) is added only when ``mag > 0``; a missing/zero magnitude passes
    through untouched, which is digest2's "no photometry" sentinel.  The band
    goes through :func:`normalize_band` first — see :data:`MPC_BAND_ALIASES`
    for why that step is not optional on real MPC output, and
    :func:`normalize_band`'s warning for the one place pneo deliberately
    does **not** reproduce digest2 (a blank band in an 80-column record).

    So: the offset *table* is digest2's, entry for entry, and every non-blank
    band agrees exactly.  Photometric identity is claimed for those, and only
    for those.
    """
    m = float(mag)
    if not np.isfinite(m) or m <= 0.0:
        return m
    return m + BAND_TO_V.get(normalize_band(band), BAND_TO_V_DEFAULT)


# --------------------------------------------------------------------------
# Observatory codes: MPC parallax constants -> WGS84 geodetic
# --------------------------------------------------------------------------

#: One MPC obscode row that yielded parallax constants.
_OBSCODE_RE = re.compile(
    r"^(\S{3})\s+(-?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)\s+"
    r"([-+]?\d+(?:\.\d*)?)\s*(.*)$"
)

#: Environment variable naming a local MPC obscode table.
OBSCODE_ENV_VAR = "PNEO_OBSCODES"



def parse_obscodes(path: str) -> dict[str, dict]:
    """Parse the MPC "List Of Observatory Codes" page.

    Returns ``{code: {"lon_deg", "rho_cos_phi", "rho_sin_phi", "name"}}`` for
    every code that publishes parallax constants.  Space-based and roving codes
    (Gaia, TESS, "Roving Observer", ...) leave those columns blank and are
    **omitted**: they define no fixed ITRS site, and pneo's observer model
    has no spacecraft ephemeris.
    """
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        import gzip

        with gzip.open(path, "rb") as fh:
            raw = fh.read()
    else:
        with open(path, "rb") as fh:
            raw = fh.read()
    text = raw.decode("utf-8", errors="replace")

    lo = text.find("<pre>")
    hi = text.find("</pre>", lo + 1)
    body = text[lo + 5 : hi] if lo >= 0 and hi > lo else text
    body = _html.unescape(re.sub(r"<[^>]+>", "", body))

    out: dict[str, dict] = {}
    for line in body.split("\n"):
        if not line.strip() or line.lstrip().startswith("Code"):
            continue
        m = _OBSCODE_RE.match(line)
        if not m:
            continue
        code, lon, rcos, rsin, name = m.groups()
        out[code] = {
            "lon_deg": float(lon),
            "rho_cos_phi": float(rcos),
            "rho_sin_phi": float(rsin),
            "name": name.strip(),
        }
    return out


#: The MPC's "List Of Observatory Codes" page, the source
#: :func:`fetch_obscodes` downloads.  It is a live page that gains sites and
#: revises parallax constants, so there is deliberately **no checksum pin**:
#: a mismatch would be the normal case, not a warning worth printing.
URL_OBSCODES = "https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html"

def fetch_obscodes(dest: str | None = None, timeout: float = 120.0
                   ) -> tuple[str, int]:
    """Download the MPC observatory-code table.  Returns ``(path, n_codes)``.

    **This is the only function in this module that touches the network, and
    nothing in the ingestion path calls it.**  It exists so that fetching the
    table is an explicit, dated act by the user (``pneo fetch``)
    rather than something a scoring run can be surprised by; see
    :class:`ObscodeTable` for why an obscode lookup must never be able to stall
    or fail a run.

    The payload is parsed with :func:`parse_obscodes` **before** anything is
    written, so a captive-portal HTML page or a truncated download cannot
    replace a good local table with a bad one.  The write is then atomic
    (``.part`` + ``os.replace``).

    Raises
    ------
    RuntimeError
        On any network error, or if what came back yields no observatory codes.
    """
    import urllib.error
    import urllib.request

    from . import __version__

    agent = f"pneo/{__version__} (+https://github.com/lsst-ssp/pneo)"
    req = urllib.request.Request(URL_OBSCODES, headers={"User-Agent": agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"GET {URL_OBSCODES} failed: {exc}") from None

    if dest is None:
        dest = default_obscodes_path()
    dest = os.fspath(dest)
    parent = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent, exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    try:
        table = parse_obscodes(tmp)
    except Exception as exc:
        os.unlink(tmp)
        raise RuntimeError(
            f"{URL_OBSCODES} returned {len(payload)} bytes that could not be "
            f"parsed as the MPC observatory-code page: {exc}"
        ) from None
    if not table:
        os.unlink(tmp)
        raise RuntimeError(
            f"{URL_OBSCODES} returned {len(payload)} bytes containing no "
            "observatory codes with parallax constants (a captive portal or an "
            "error page?); nothing was written."
        )
    os.replace(tmp, dest)
    return dest, len(table)


def obscode_geodetic(row: dict) -> tuple[float, float, float]:
    """MPC parallax constants -> WGS84 ``(lon_deg, lat_deg, height_m)``.

    The MPC publishes each site as geocentric cylindrical coordinates in units
    of the equatorial Earth radius: ``x = rho cos(phi') cos(lon)``,
    ``y = rho cos(phi') sin(lon)``, ``z = rho sin(phi')``.  Those define an
    ITRS position with no ellipsoid involved, so we build the position and let
    astropy invert it to geodetic — which is what
    :func:`pneo._attributable.observatory_location` consumes for any code
    outside its built-in table.  The round trip is exact to sub-millimetre;
    ``tests/test_ingest.py`` checks it against every entry of that built-in
    table.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    lam = np.radians(float(row["lon_deg"]))
    rc, rs = float(row["rho_cos_phi"]), float(row["rho_sin_phi"])
    xyz = np.array([rc * np.cos(lam), rc * np.sin(lam), rs]) * MPC_EARTH_RADIUS_M
    loc = EarthLocation.from_geocentric(*xyz, unit=u.m)
    geo = loc.to_geodetic(ellipsoid="WGS84")
    return (
        float(geo.lon.to_value(u.deg)),
        float(geo.lat.to_value(u.deg)),
        float(geo.height.to_value(u.m)),
    )


class ObscodeTable:
    """Resolve MPC observatory codes to WGS84 lon/lat/height, offline.

    Resolution order for a code, and the ``source`` string each step returns:

    ``"geocenter"``
        the code denotes the geocentre (:data:`
        pneo._attributable.GEOCENTER_OBSCODES`, i.e. ``"500"``) — no
        topocentric offset.
    ``"builtin"``
        :data:`pneo._attributable.MPC_OBSCODES`, the package's own small
        table of parallax constants, converted by :func:`obscode_geodetic`.
        Preferred over any file so that a stale or truncated local copy cannot
        move a site pneo already knows.
    ``"file"``
        a local copy of the MPC "List Of Observatory Codes" page: the ``path``
        argument, else the first of ``$PNEO_OBSCODES`` and the per-user cache
        copy (:func:`pneo.paths.default_obscodes_path`, written by
        ``pneo fetch``) that exists *and* yields codes.  The MPC
        publishes the table as an HTML page; the parser strips tags, so a
        plain-text save of it parses too.  Loaded lazily, once.  ``path``
        suppresses the search entirely, so an explicit table that cannot be
        read is reported through :attr:`load_error` instead of being silently
        replaced by the cached copy.
    ``"fallback"``
        nothing knew the code (or the detection carried no station at all): the
        **geocentre** is used and a ``UserWarning`` is issued once per code — so
        a 900-detection night warns once, not 900 times.  A geocentric
        approximation of
        a ground site displaces the observer by up to one Earth radius
        (4.3e-5 AU), which matters most for exactly the fast, close objects
        this code is used to triage — hence the warning, and hence
        ``Tracklet.obscode_source`` recording it in the output.

    **The network is never touched.**  Downloading an obscode table in the
    middle of a scoring run would make results depend on connectivity; use
    ``pneo classify --obscodes PATH`` (or the environment variable) to point
    at a copy fetched deliberately, out of band.
    """

    def __init__(self, path: str | None = None, *, search: bool = True) -> None:
        self._explicit = path
        self._search = bool(search)
        self._table: dict[str, dict] | None = None
        self._loaded_from: str | None = None
        self._load_error: str | None = None
        self._warned: set[str] = set()
        self._cache: dict[str, tuple[float, float, float, str]] = {}

    # -- the file half -----------------------------------------------------

    def _candidate_paths(self) -> list[str]:
        if self._explicit:
            return [self._explicit]
        out = []
        env = os.environ.get(OBSCODE_ENV_VAR, "").strip()
        if env:
            out.append(env)
        if self._search:
            out.append(str(default_obscodes_path()))
        return out

    def _ensure_loaded(self) -> dict[str, dict]:
        if self._table is not None:
            return self._table
        self._table = {}
        for cand in self._candidate_paths():
            if not cand:
                continue
            if not os.path.exists(cand):
                if cand == self._explicit:
                    self._load_error = f"{cand}: no such file"
                continue
            try:
                table = parse_obscodes(cand)
            except Exception as exc:  # a corrupt local copy must not be fatal
                self._load_error = f"{cand}: {exc}"
                continue
            if table:
                self._table = table
                self._loaded_from = os.path.abspath(cand)
                break
            # An existing file that yields no codes at all is not "the table":
            # keep looking, so an empty file named by $PNEO_OBSCODES does not
            # mask a good copy in the cache.
        if (
            self._explicit
            and self._loaded_from is None
            and self._load_error is None
        ):
            self._load_error = (
                f"{self._explicit}: no observatory codes with parallax "
                "constants found (is this the MPC 'List Of Observatory Codes' "
                "page?)"
            )
        return self._table

    @property
    def loaded_from(self) -> str | None:
        """Absolute path of the local table in use, or ``None`` if none is."""
        self._ensure_loaded()
        return self._loaded_from

    @property
    def load_error(self) -> str | None:
        """Why the configured local table could not be used, if it could not."""
        self._ensure_loaded()
        return self._load_error

    # -- resolution --------------------------------------------------------

    def resolve(self, obscode: str | None) -> tuple[float, float, float, str]:
        """``(lon_deg, lat_deg, height_m, source)`` for one observatory code.

        Never raises and never blocks; see the class docstring for the order and
        the ``source`` vocabulary.
        """
        code = "" if obscode is None else str(obscode).strip()
        hit = self._cache.get(code)
        if hit is not None:
            return hit

        key = code.upper()
        if key == "" or key in GEOCENTER_OBSCODES:
            out = (0.0, 0.0, 0.0, "geocenter" if key else "fallback")
            if not key:
                self._warn(
                    "pneo._ingest: a tracklet carries no observatory code; "
                    "scoring it at the geocentre. Check the 'stn' column "
                    "(ADES) or columns 78-80 (MPC 80-column).",
                    "",
                )
        elif key in MPC_OBSCODES:
            lon, rho_cos_phi, rho_sin_phi = MPC_OBSCODES[key]
            lon_d, lat_d, h = obscode_geodetic(
                {
                    "lon_deg": lon,
                    "rho_cos_phi": rho_cos_phi,
                    "rho_sin_phi": rho_sin_phi,
                }
            )
            out = (lon_d, lat_d, h, "builtin")
        else:
            row = self._ensure_loaded().get(code) or self._ensure_loaded().get(key)
            if row is None:
                where = (
                    f"nor in {self._loaded_from}" if self._loaded_from
                    else "and no local obscode table is configured"
                )
                self._warn_fallback(
                    code,
                    f"is not in pneo's built-in table {where}; scoring it "
                    "GEOCENTRICALLY, which displaces the observer by up to one "
                    "Earth radius (4.3e-5 AU)",
                )
                out = (0.0, 0.0, 0.0, "fallback")
            else:
                lon_d, lat_d, h = obscode_geodetic(row)
                out = (lon_d, lat_d, h, "file")

        self._cache[code] = out
        return out

    def _warn_fallback(self, code: str, why: str) -> None:
        self._warn(
            f"pneo._ingest: observatory code {code!r} {why}. Pass "
            f"--obscodes PATH (or set {OBSCODE_ENV_VAR}) to a local copy of "
            "the MPC 'List Of Observatory Codes' page.",
            code,
        )

    def _warn(self, message: str, key: str) -> None:
        """Warn once per ``key``, so a 900-detection night warns once."""
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


# --------------------------------------------------------------------------
# The values ingestion produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One parsed astrometric detection."""

    object_id: str
    obscode: str
    jd_utc: float
    ra_deg: float
    dec_deg: float
    #: The magnitude as reported (NaN when the record carries none).
    mag: float = float("nan")
    #: The band as reported ("" when the record carries none).
    band: str = ""
    #: :func:`band_to_v` of ``(band, mag)``; NaN when there is no photometry.
    v_mag: float = float("nan")
    source: str = ""
    line: int = 0


@dataclass
class Tracklet:
    """One tracklet, fitted and ready to classify."""

    object_id: str
    obscode: str
    n_obs: int
    arc_hours: float
    epoch_jd_utc: float
    #: Attributable, radians and radians/day, at ``epoch_jd_utc``.
    ra: float
    dec: float
    ra_rate: float
    dec_rate: float
    #: Median band-corrected Johnson V over the tracklet [D-5].
    v_mag: float
    #: RMS of the straight-line fit residuals [arcsec] — the per-tracklet
    #: quality flag.  **~0 for a two-detection tracklet** (a line through two
    #: points has no residual), so a 0 there means "nothing to check", not
    #: "a perfect fit"; see :func:`fit_attributable`.
    resid_rms_arcsec: float
    lon_deg: float
    lat_deg: float
    height_m: float
    #: How the observatory was resolved; see :class:`ObscodeTable`.
    obscode_source: str = "geocenter"
    #: Bands as reported, deduplicated and sorted (provenance).
    bands: tuple[str, ...] = ()
    #: True when the night was longer than ``max_arc_hours`` and was cut.
    truncated_arc: bool = False
    #: Detections seen for this object across the whole input, before grouping.
    n_detections_seen: int = 0
    source: str = ""
    #: First-appearance index of this object in the input.  Carried so a caller
    #: that reports tracklets and failures together (the CLI does) can restore
    #: input order across the two lists :func:`group_tracklets` returns.
    seq: int = 0

    @property
    def is_geocentric(self) -> bool:
        """True when no topocentric offset applies (geocentre, or a fallback).

        Keyed on :attr:`obscode_source`, **not** on the coordinates: a
        geocentric tracklet carries ``lon_deg = lat_deg = height_m = 0``, and
        feeding those to :func:`pneo._attributable.observer_state` as a
        *geodetic site* would place the observer on the equator at Greenwich —
        one Earth radius (4.3e-5 AU) from where it belongs.
        """
        return self.obscode_source in ("geocenter", "fallback")

    def observer_kwargs(self) -> dict:
        """Observer arguments for :func:`pneo.classify` / ``observer_state``.

        ``{}`` (i.e. the geocentre) for a geocentric tracklet, otherwise the
        resolved geodetic site.  Use this rather than reading ``lon_deg`` &c.
        directly; see :attr:`is_geocentric`.
        """
        if self.is_geocentric:
            return {}
        return {
            "lon_deg": float(self.lon_deg),
            "lat_deg": float(self.lat_deg),
            "height_m": float(self.height_m),
        }

    @property
    def epoch_isot(self) -> str:
        """``epoch_jd_utc`` as an ISO-8601 UTC string (millisecond precision)."""
        from astropy.time import Time

        return str(
            Time(float(self.epoch_jd_utc), format="jd", scale="utc").isot
        )[:23]

    def attributable(self) -> Attributable:
        """This tracklet as an :class:`~pneo._attributable.Attributable`."""
        from astropy.time import Time

        return Attributable(
            ra=float(self.ra),
            dec=float(self.dec),
            ra_rate=float(self.ra_rate),
            dec_rate=float(self.dec_rate),
            epoch=Time(float(self.epoch_jd_utc), format="jd", scale="utc"),
            v_mag=float(self.v_mag),
        )


@dataclass(frozen=True)
class IngestFailure:
    """One object that could not become a tracklet.

    ``status`` is a short machine-readable token — the value the ``pneo
    score`` status column carries for this row — and ``message`` is the
    human-readable reason.  The vocabulary:

    ``"single_detection"``
        no same-night, same-station set with at least two detections, so no
        rates can be fitted.
    ``"no_magnitude"``
        the chosen detection set carries no usable magnitude; V is a required
        observable in v0 (DESIGN §2.2), so the tracklet cannot be scored.
    ``"fit_failed"``
        the least-squares fit produced non-finite rates (a zero-length arc:
        every detection at the same instant).
    ``"no_detections"``
        the object survived parsing with no usable detection at all.
    """

    object_id: str
    status: str
    message: str
    source: str = ""
    obscode: str = ""
    n_obs: int = 0
    #: First-appearance index of this object in the input; see ``Tracklet.seq``.
    seq: int = 0


@dataclass(frozen=True)
class ParseIssue:
    """One record that could not be parsed.  Reported, never fatal."""

    source: str
    line: int
    reason: str
    text: str = ""


# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------

#: Accepted values of the ``fmt`` argument / ``--obs-format`` flag.
FORMATS: tuple[str, ...] = ("auto", "ades_psv", "mpc80")

#: ADES column names that identify a PSV header line.
_ADES_MARKERS = frozenset({"obstime", "ra", "dec", "stn", "trksub", "provid"})

#: "YYYY MM DD.dddddd" in columns 15:32 of an MPC 80-column optical record.
_MPC80_DATE_RE = re.compile(r"^\s*\d{4} \d{2} \d{2}\.\d+\s*$")


def _read_lines(path: str) -> list[str]:
    """Read a file as lines, tolerating a UTF-8 BOM.

    ``utf-8-sig`` strips a leading byte-order mark if one is present and is
    otherwise identical to ``utf-8``.  Files exported from Windows tooling
    routinely carry one, and without this the BOM lands on the first character
    of the ADES header (``\ufeffpermID``), so the header no longer names
    ``permID`` and the whole file is rejected.
    """
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return fh.read().splitlines()


def detect_format(path: str, *, lines: list[str] | None = None) -> str:
    """Guess the observation format of ``path`` from its content.

    Returns ``"ades_psv"`` or ``"mpc80"``.  ADES is recognised by a
    pipe-separated header line naming ADES columns; MPC 80-column by a record
    whose columns 15:32 hold an MPC date.  The suffix is deliberately **not**
    consulted: real files arrive named ``.txt``, ``.obs``, ``.psv`` and
    ``.ades`` interchangeably.

    Raises
    ------
    ValueError
        If neither signature is found in the first 200 non-blank, **non-comment**
        lines — with the offending line quoted, so the user can see what was
        rejected.  Comments do not count against that budget: an ADES file may
        carry a long ``#``-prefixed provenance preamble (the MPC's own exports
        do), and counting it would let the header fall outside the window and
        make a perfectly good file undetectable.
    """
    rows = _read_lines(path) if lines is None else lines
    first_data = ""
    seen = 0
    for raw in rows:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#") or line.lstrip().startswith("!"):
            continue  # ADES/PSV comment or a header pragma; not counted
        seen += 1
        if seen > 200:
            break
        if "|" in line:
            names = {c.strip().lower() for c in line.split("|")}
            if names & _ADES_MARKERS:
                return "ades_psv"
        if len(line) >= 32 and _MPC80_DATE_RE.match(line[15:32]):
            return "mpc80"
        if not first_data:
            first_data = line
    raise ValueError(
        f"{path}: cannot tell whether this is ADES PSV or MPC 80-column. "
        "Expected either a pipe-separated ADES header line naming obsTime/ra/"
        "dec, or 80-column records with 'YYYY MM DD.dddddd' in columns 16-32. "
        f"First data line was: {first_data[:90]!r}. "
        "Force it with --obs-format ades_psv|mpc80."
    )


# --------------------------------------------------------------------------
# Time parsing
# --------------------------------------------------------------------------


def _isot_to_jd_utc(values: list[str]) -> np.ndarray:
    """ADES ``obsTime`` strings -> JD (UTC); NaN where a value is unparseable.

    Vectorised through astropy in one call (which dominates ingestion cost on
    large files) with a per-row retry, so one malformed timestamp costs its own
    record and nothing else.  Note the retry is all-or-nothing: a single bad
    timestamp in a very large file drops the whole file onto the slow path
    (~0.5 ms per record).
    """
    from astropy.time import Time

    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    cleaned = [str(v).strip().rstrip("Zz") for v in values]
    try:
        out[:] = np.atleast_1d(
            Time(cleaned, format="isot", scale="utc").jd
        ).astype(float)
        return out
    except Exception:
        pass
    for i, s in enumerate(cleaned):
        try:
            out[i] = float(Time(s, format="isot", scale="utc").jd)
        except Exception:
            out[i] = np.nan
    return out


_MONTH_START_JD: dict[tuple[int, int], float] = {}


def _mpc_date_to_jd_utc(year: int, month: int, day: float) -> float:
    """MPC ``YYYY MM DD.dddddd`` -> JD (UTC).

    The JD of the month's first day at 0h UTC is computed once per (year,
    month) — real files hold a handful of distinct months and astropy's parser
    is the expensive part.

    Raises ValueError (from astropy) for an impossible year or month; callers
    must treat that as a per-record failure.
    """
    from astropy.time import Time

    key = (int(year), int(month))
    base = _MONTH_START_JD.get(key)
    if base is None:
        base = float(
            Time(f"{key[0]:04d}-{key[1]:02d}-01T00:00:00", format="isot",
                 scale="utc").jd
        )
        _MONTH_START_JD[key] = base
    return base + (float(day) - 1.0)


# --------------------------------------------------------------------------
# ADES PSV
# --------------------------------------------------------------------------


def _first_nonempty(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = row.get(k, "")
        if v:
            return v
    return ""


def parse_ades_psv(
    path: str, *, lines: list[str] | None = None
) -> tuple[list[Detection], list[ParseIssue]]:
    """Parse an ADES PSV (pipe-separated) observation file.

    The header is the first non-comment line containing ``|``; columns are
    matched by name (case-insensitively), so any column order or subset works
    and unknown columns are ignored.  Required: ``obsTime``, ``ra``, ``dec``.
    The object id is the first non-empty of ``trkSub``, ``provID``, ``permID``
    — ``trkSub`` first because that is what digest2 groups on and what a
    survey's own submission carries.

    Returns ``(detections, issues)``.  A record that cannot be parsed becomes a
    :class:`ParseIssue` and is skipped.
    """
    rows = _read_lines(path) if lines is None else lines
    issues: list[ParseIssue] = []
    header: list[str] | None = None
    records: list[tuple[int, dict[str, str]]] = []

    for lineno, raw in enumerate(rows, 1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("!"):
            continue
        if "|" not in line:
            issues.append(
                ParseIssue(path, lineno, "not a pipe-separated record",
                           line[:90])
            )
            continue
        cells = [c.strip() for c in line.split("|")]
        if header is None:
            header = [c.lower() for c in cells]
            if not ({"obstime", "ra", "dec"} <= set(header)):
                missing = sorted({"obstime", "ra", "dec"} - set(header))
                raise ValueError(
                    f"{path}: ADES PSV header is missing required column(s) "
                    f"{missing}; found {header}."
                )
            continue
        # A row that does not have exactly as many fields as the header is a
        # malformed row, and silently padding or truncating it is how a
        # column-shifted file scores as if nothing were wrong.  Both cases are
        # still USABLE (the named columns that are present parse fine), so the
        # row is kept and the discrepancy reported rather than dropped.
        if len(cells) < len(header):
            issues.append(
                ParseIssue(
                    path, lineno,
                    f"row has {len(cells)} fields, header has {len(header)}; "
                    "missing trailing fields treated as empty",
                    line[:90],
                )
            )
            cells = cells + [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            issues.append(
                ParseIssue(
                    path, lineno,
                    f"row has {len(cells)} fields, header has {len(header)}; "
                    "the extra trailing field(s) are ignored",
                    line[:90],
                )
            )
        records.append((lineno, dict(zip(header, cells))))

    if header is None:
        issues.append(ParseIssue(path, 0, "no ADES PSV header line found", ""))
        return [], issues

    times = _isot_to_jd_utc([r.get("obstime", "") for _, r in records])
    dets: list[Detection] = []
    for (lineno, row), jd in zip(records, times):
        if not np.isfinite(jd):
            issues.append(
                ParseIssue(path, lineno, "unparseable obsTime "
                           f"{row.get('obstime', '')!r}")
            )
            continue
        try:
            ra_deg = float(row["ra"])
            dec_deg = float(row["dec"])
        except (KeyError, TypeError, ValueError):
            issues.append(
                ParseIssue(path, lineno, "unparseable ra/dec "
                           f"{row.get('ra', '')!r}/{row.get('dec', '')!r}")
            )
            continue
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            issues.append(ParseIssue(path, lineno, "non-finite ra/dec"))
            continue
        if not -90.0 <= dec_deg <= 90.0:
            issues.append(
                ParseIssue(path, lineno, f"declination {dec_deg} out of range")
            )
            continue
        obj = _first_nonempty(row, ("trksub", "provid", "permid"))
        if not obj:
            issues.append(
                ParseIssue(path, lineno,
                           "no object id (trkSub/provID/permID all empty)")
            )
            continue
        band = row.get("band", "")
        raw_mag = row.get("mag", "")
        try:
            mag = float(raw_mag) if raw_mag not in ("", None) else float("nan")
        except (TypeError, ValueError):
            mag = float("nan")
            issues.append(
                ParseIssue(path, lineno, f"unparseable mag {raw_mag!r} "
                           "(treated as no photometry)")
            )
        dets.append(
            Detection(
                object_id=obj,
                obscode=row.get("stn", ""),
                jd_utc=float(jd),
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                mag=mag,
                band=band,
                v_mag=band_to_v(band, mag),
                source=path,
                line=lineno,
            )
        )
    return dets, issues


# --------------------------------------------------------------------------
# MPC 80-column
# --------------------------------------------------------------------------

#: Note-2 (column 15) codes whose records this parser skips, with the reason:
#:
#: * ``S``/``s`` — the two halves of a **space-based** observation.  pneo's
#:   observer model has no spacecraft ephemeris (this is also why the NEOCP
#:   pilot excluded C51/258), and the ``s`` line is not astrometry at all: it
#:   carries the satellite's geocentric position in the RA/Dec columns.
#: * ``V``/``v`` — a **roving observer** (obscode 247): no fixed ITRS site, and
#:   the ``v`` line likewise holds coordinates rather than astrometry.
#: * ``R``/``r`` — **radar**: the columns hold delay/Doppler, not RA/Dec.
#:
#: Parsing any of these as optical astrometry would not fail, it would silently
#: invent a detection, so each one is skipped and counted as a ParseIssue.
_MPC80_SKIP_NOTE2 = frozenset("SsVvRr")


def parse_mpc80(
    path: str, *, lines: list[str] | None = None
) -> tuple[list[Detection], list[ParseIssue]]:
    """Parse an MPC 80-column optical astrometry file.

    Fixed columns, per the MPC's format description: designation 1-12, note 2
    (observation type) 15, date 16-32, RA 33-44, Dec 45-56, magnitude 66-70,
    band 71, observatory code 78-80 (all 1-based, inclusive).

    The **object id is columns 1-12 verbatim** (whitespace-stripped): packed
    designations are *not* unpacked.  Unpacking is unnecessary for the only
    thing the id is used for — grouping an object's detections together and
    labelling its output row — and getting it wrong would silently merge or
    split tracklets.  The id you get out is the id that was in the file.

    **Blank-band records are converted as V and counted** [D-20], and a
    ``UserWarning`` naming the count is raised once for this file if there were
    any: digest2's 80-column reader would give them its -0.8 unknown-band
    fallback instead, so this is the one place the two codes' photometry
    differs, and it must never be silent.  See :func:`normalize_band`.

    Returns ``(detections, issues)``.
    """
    rows = _read_lines(path) if lines is None else lines
    issues: list[ParseIssue] = []
    dets: list[Detection] = []
    n_blank_band = 0

    for lineno, raw in enumerate(rows, 1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if len(line) < 80:
            issues.append(
                ParseIssue(path, lineno,
                           f"record is {len(line)} characters, expected 80",
                           line[:90])
            )
            continue
        note2 = line[14]
        if note2 in _MPC80_SKIP_NOTE2:
            issues.append(
                ParseIssue(path, lineno,
                           f"observation type {note2!r} is not ground-based "
                           "optical astrometry; skipped")
            )
            continue
        obj = line[0:12].strip()
        if not obj:
            issues.append(ParseIssue(path, lineno, "blank designation field"))
            continue
        try:
            year = int(line[15:19])
            month = int(line[20:22])
            day = float(line[23:32])
            # Inside the try: a nonsense month or day must cost this record and
            # not the file (astropy raises a bare ValueError on "2026-13-01").
            jd = _mpc_date_to_jd_utc(year, month, day)
            ra_tok = line[32:44].split()
            dec_field = line[44:56]
            sign = -1.0 if dec_field[0].strip() == "-" else 1.0
            dec_tok = dec_field[1:].split()
            if len(ra_tok) != 3 or len(dec_tok) != 3:
                raise ValueError("RA/Dec are not three sexagesimal fields")
            ra_deg = (
                float(ra_tok[0]) + float(ra_tok[1]) / 60.0
                + float(ra_tok[2]) / 3600.0
            ) * 15.0
            dec_deg = sign * (
                float(dec_tok[0]) + float(dec_tok[1]) / 60.0
                + float(dec_tok[2]) / 3600.0
            )
        except (ValueError, IndexError) as exc:
            issues.append(ParseIssue(path, lineno, f"bad record: {exc}",
                                     line[:90]))
            continue
        if not (0.0 <= ra_deg <= 360.0 and -90.0 <= dec_deg <= 90.0):
            issues.append(
                ParseIssue(path, lineno,
                           f"coordinates out of range: {ra_deg}, {dec_deg}")
            )
            continue
        mag_field = line[65:70].strip()
        band = line[70:71].strip()
        if mag_field and not band:
            # Counted only when there IS a magnitude to convert: a record with
            # neither carries no photometry at all and nothing diverges.
            n_blank_band += 1
        try:
            mag = float(mag_field) if mag_field else float("nan")
        except ValueError:
            mag = float("nan")
            issues.append(
                ParseIssue(path, lineno, f"unparseable magnitude "
                           f"{mag_field!r} (treated as no photometry)")
            )
        dets.append(
            Detection(
                object_id=obj,
                obscode=line[77:80].strip(),
                jd_utc=jd,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                mag=mag,
                band=band,
                v_mag=band_to_v(band, mag),
                source=path,
                line=lineno,
            )
        )
    if n_blank_band:
        warnings.warn(
            f"pneo._ingest: {path}: {n_blank_band} MPC 80-column record(s) "
            "carry a magnitude with a BLANK band; pneo converts these as V "
            "(offset 0.0), whereas digest2's own 80-column reader applies its "
            "-0.8 unknown-band fallback, so pneo's V is 0.80 mag fainter "
            "on those records. This is a deliberate choice [D-20], not a bug: "
            "a blank band in pre-ADES astrometry is an unfiltered/visual "
            "magnitude, and -0.8 is digest2's crudest fallback applied there "
            "only because a different parser ran.",
            UserWarning,
            stacklevel=2,
        )
    return dets, issues


# --------------------------------------------------------------------------
# Tracklet assembly
# --------------------------------------------------------------------------

#: Any gap longer than this splits a night [h].  Real discovery tracklets are
#: minutes to an hour or two long and consecutive nights are ~a day apart, so a
#: few hours is unambiguous (WP8).
DEFAULT_NIGHT_GAP_HOURS = 3.0

#: A night longer than this is truncated to its first ``max_arc_hours`` [h].
#: The attributable is a first-order motion model; over a long arc the
#: straight-line fit stops describing the motion at the reference epoch
#: (measured on synthetic Keplerian arcs: the position error grows as the arc
#: squared).
DEFAULT_MAX_ARC_HOURS = 2.0


def cluster_nights(
    t_days: np.ndarray, gap_hours: float = DEFAULT_NIGHT_GAP_HOURS
) -> list[np.ndarray]:
    """Split epochs into groups separated by more than ``gap_hours``.

    Returns index arrays into the input, in time order.  Input need not be
    sorted.
    """
    t = np.asarray(t_days, dtype=float)
    if t.size == 0:
        return []
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    breaks = np.nonzero(np.diff(ts) > float(gap_hours) / 24.0)[0] + 1
    return [order[sl] for sl in np.split(np.arange(t.size), breaks)]


def fit_attributable(
    t_days: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> dict:
    """Least-squares straight-line fit of alpha(t), delta(t); epoch = ``t[0]``.

    Ordinary (unweighted) least squares in each coordinate independently.  ADES
    per-observation uncertainties are frequently absent in real files, and
    weighting a 3-point fit by sporadic sigmas would make the estimator depend
    on which observers happened to report them.

    RA is unwrapped about the first detection before fitting, so a tracklet
    straddling alpha = 0 fits correctly.  The reported epoch is the **first**
    detection's, matching the WP6/WP7 convention and digest2's ``obsPair[0]``
    anchor.

    Returns radians and rad/day, plus the fit residual RMS in arcsec (the
    per-tracklet quality flag) and the arc length in hours.  A zero-length arc
    (all epochs equal) yields NaN rates.

    ``resid_rms_arcsec`` is **~0 for a two-detection tracklet**, not NaN: a
    straight line through two points has no residual, and the divisor is
    floored at 1 (``max(n - 2, 1)``) rather than being a true degrees-of-freedom
    count.  So a 0 there means "nothing to check", not "a perfect fit"; only
    with three or more detections does it measure anything.

    Raises
    ------
    ValueError
        With fewer than two detections: there is nothing to fit.
    """
    t = np.asarray(t_days, dtype=float)
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if t.size < 2:
        raise ValueError("need at least 2 detections to fit an attributable")
    order = np.argsort(t, kind="mergesort")
    t, ra, dec = t[order], ra[order], dec[order]

    t0 = t[0]
    dt = t - t0
    ra0 = ra[0]
    ra_un = ra0 + ((ra - ra0 + 180.0) % 360.0 - 180.0)

    design = np.column_stack([np.ones_like(dt), dt])
    if not np.any(dt != 0.0):
        return {
            "epoch_jd_utc": float(t0),
            "ra": float(np.radians(ra0 % 360.0)),
            "dec": float(np.radians(dec[0])),
            "ra_rate": float("nan"),
            "dec_rate": float("nan"),
            "resid_rms_arcsec": float("nan"),
            "arc_hours": 0.0,
            "n_det": int(t.size),
        }

    (a0, a1), *_ = np.linalg.lstsq(design, ra_un, rcond=None)
    (d0, d1), *_ = np.linalg.lstsq(design, dec, rcond=None)

    res_ra = (ra_un - (a0 + a1 * dt)) * np.cos(np.radians(d0))
    res_dec = dec - (d0 + d1 * dt)
    n_free = max(t.size - 2, 1)
    rms = np.sqrt((np.sum(res_ra**2) + np.sum(res_dec**2)) / (2 * n_free)) * 3600.0

    return {
        "epoch_jd_utc": float(t0),
        "ra": float(np.radians(a0 % 360.0)),
        "dec": float(np.radians(d0)),
        "ra_rate": float(np.radians(a1)),
        "dec_rate": float(np.radians(d1)),
        "resid_rms_arcsec": float(rms),
        "arc_hours": float((t[-1] - t[0]) * 24.0),
        "n_det": int(t.size),
    }


def group_tracklets(
    detections: list[Detection],
    *,
    obscodes: ObscodeTable | None = None,
    night_gap_hours: float = DEFAULT_NIGHT_GAP_HOURS,
    max_arc_hours: float = DEFAULT_MAX_ARC_HOURS,
) -> tuple[list[Tracklet], list[IngestFailure]]:
    """Detections -> fitted tracklets, one per object (WP8's rule).

    Detections are grouped by ``object_id`` (output order is first appearance),
    then by ``obscode``, then split into nights on any gap longer than
    ``night_gap_hours``.  Among the (station, night) sets with at least two
    detections, the one whose **first** detection is earliest becomes the
    tracklet; if it spans more than ``max_arc_hours`` it is cut back to the
    detections inside that span (never below two).

    Returns ``(tracklets, failures)``.  Every object appears in exactly one of
    the two lists.
    """
    table = obscodes if obscodes is not None else ObscodeTable()
    by_object: dict[str, list[Detection]] = {}
    for det in detections:
        by_object.setdefault(det.object_id, []).append(det)

    tracklets: list[Tracklet] = []
    failures: list[IngestFailure] = []

    for seq, (obj, dets) in enumerate(by_object.items()):
        source = dets[0].source
        n_seen = len(dets)
        usable = [
            d for d in dets
            if np.isfinite(d.jd_utc) and np.isfinite(d.ra_deg)
            and np.isfinite(d.dec_deg)
        ]
        if not usable:
            failures.append(
                IngestFailure(obj, "no_detections",
                              "no detection with a finite time and position",
                              source, "", n_seen, seq)
            )
            continue

        # -- earliest same-night, same-station set with >= 2 detections -----
        by_station: dict[str, list[Detection]] = {}
        for d in usable:
            by_station.setdefault(d.obscode, []).append(d)
        best: list[Detection] | None = None
        for station, sd in by_station.items():
            jd = np.array([d.jd_utc for d in sd])
            for idx in cluster_nights(jd, night_gap_hours):
                if idx.size < 2:
                    continue
                group = [sd[i] for i in idx]
                group.sort(key=lambda d: d.jd_utc)
                if best is None or group[0].jd_utc < best[0].jd_utc:
                    best = group
        if best is None:
            stations = sorted({d.obscode for d in usable})
            failures.append(
                IngestFailure(
                    obj, "single_detection",
                    f"{len(usable)} usable detection(s) at station(s) "
                    f"{stations}, but no two within {night_gap_hours:g} h of "
                    "each other at the same station: rates cannot be fitted",
                    source, usable[0].obscode, n_seen, seq,
                )
            )
            continue

        # -- truncate an over-long night -----------------------------------
        jd_sel = np.array([d.jd_utc for d in best])
        truncated = False
        if jd_sel[-1] - jd_sel[0] > float(max_arc_hours) / 24.0:
            keep = jd_sel - jd_sel[0] <= float(max_arc_hours) / 24.0
            if keep.sum() < 2:
                keep = np.zeros(jd_sel.size, bool)
                keep[:2] = True
            best = [d for d, k in zip(best, keep) if k]
            jd_sel = jd_sel[keep]
            truncated = True

        # -- photometry: median V over the tracklet -------------------------
        v = np.array([d.v_mag for d in best], dtype=float)
        finite_v = v[np.isfinite(v) & (v > 0.0)]
        if finite_v.size == 0:
            failures.append(
                IngestFailure(
                    obj, "no_magnitude",
                    f"none of the {len(best)} detections carries a usable "
                    "magnitude; V is a required observable (DESIGN 2.2)",
                    source, best[0].obscode, n_seen, seq,
                )
            )
            continue
        v_mag = float(np.median(finite_v))

        # -- the fit -------------------------------------------------------
        try:
            fit = fit_attributable(
                jd_sel,
                np.array([d.ra_deg for d in best]),
                np.array([d.dec_deg for d in best]),
            )
        except Exception as exc:  # pragma: no cover - guarded by len >= 2
            failures.append(
                IngestFailure(obj, "fit_failed", f"attributable fit failed: "
                              f"{exc}", source, best[0].obscode, n_seen, seq)
            )
            continue
        if not (np.isfinite(fit["ra_rate"]) and np.isfinite(fit["dec_rate"])):
            failures.append(
                IngestFailure(
                    obj, "fit_failed",
                    f"{len(best)} detections span a zero-length arc "
                    "(identical epochs); rates are undefined",
                    source, best[0].obscode, n_seen, seq,
                )
            )
            continue

        obscode = best[0].obscode
        lon, lat, height, src = table.resolve(obscode)
        tracklets.append(
            Tracklet(
                object_id=obj,
                obscode=obscode,
                n_obs=int(fit["n_det"]),
                arc_hours=float(fit["arc_hours"]),
                epoch_jd_utc=float(fit["epoch_jd_utc"]),
                ra=float(fit["ra"]),
                dec=float(fit["dec"]),
                ra_rate=float(fit["ra_rate"]),
                dec_rate=float(fit["dec_rate"]),
                v_mag=v_mag,
                resid_rms_arcsec=float(fit["resid_rms_arcsec"]),
                lon_deg=lon,
                lat_deg=lat,
                height_m=height,
                obscode_source=src,
                bands=tuple(sorted({d.band for d in best if d.band})),
                truncated_arc=truncated,
                n_detections_seen=n_seen,
                source=source,
                seq=seq,
            )
        )

    return tracklets, failures


# --------------------------------------------------------------------------
# The one-call entry points
# --------------------------------------------------------------------------


def parse_observations(
    path: str, fmt: str = "auto"
) -> tuple[list[Detection], list[ParseIssue], str]:
    """Parse one observation file.  Returns ``(detections, issues, format)``.

    ``fmt`` is one of :data:`FORMATS`; ``"auto"`` calls :func:`detect_format`.
    """
    fmt = str(fmt).strip().lower()
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    lines = _read_lines(path)
    if fmt == "auto":
        fmt = detect_format(path, lines=lines)
    if fmt == "ades_psv":
        dets, issues = parse_ades_psv(path, lines=lines)
    else:
        dets, issues = parse_mpc80(path, lines=lines)
    return dets, issues, fmt


def ingest_file(
    path: str,
    *,
    fmt: str = "auto",
    obscodes: ObscodeTable | None = None,
    night_gap_hours: float = DEFAULT_NIGHT_GAP_HOURS,
    max_arc_hours: float = DEFAULT_MAX_ARC_HOURS,
) -> tuple[list[Tracklet], list[IngestFailure], list[ParseIssue], str]:
    """One file -> ``(tracklets, failures, issues, format)``."""
    dets, issues, fmt = parse_observations(path, fmt)
    tracklets, failures = group_tracklets(
        dets,
        obscodes=obscodes,
        night_gap_hours=night_gap_hours,
        max_arc_hours=max_arc_hours,
    )
    return tracklets, failures, issues, fmt


def ingest(
    paths,
    *,
    fmt: str = "auto",
    obscodes: ObscodeTable | None = None,
    night_gap_hours: float = DEFAULT_NIGHT_GAP_HOURS,
    max_arc_hours: float = DEFAULT_MAX_ARC_HOURS,
) -> tuple[list[Tracklet], list[IngestFailure], list[ParseIssue]]:
    """Ingest several observation files.

    Each file is detected and parsed independently (so a PSV and an 80-column
    file can be scored in one command), but **detections are grouped across
    all of them**: the same object observed in two files becomes one tracklet,
    which is what a user splitting a night across files expects.

    A file that cannot be read or whose format cannot be determined is reported
    as a :class:`ParseIssue` on line 0 and skipped; the rest of the run
    proceeds.  Returns ``(tracklets, failures, issues)``.
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    all_dets: list[Detection] = []
    issues: list[ParseIssue] = []
    for path in paths:
        path = os.fspath(path)
        try:
            dets, iss, _fmt = parse_observations(path, fmt)
        except (OSError, ValueError) as exc:
            issues.append(ParseIssue(path, 0, str(exc)))
            continue
        all_dets.extend(dets)
        issues.extend(iss)
    tracklets, failures = group_tracklets(
        all_dets,
        obscodes=obscodes,
        night_gap_hours=night_gap_hours,
        max_arc_hours=max_arc_hours,
    )
    return tracklets, failures, issues
