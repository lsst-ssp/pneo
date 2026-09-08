"""WP1 — Attributable value type and observer heliocentric state.

Frozen contract (WP0). Implementations must preserve every public signature,
unit, and frame documented here. See DESIGN.md §2.1, IMPLEMENTATION.md §1/WP1.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    EarthLocation,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from astropy.time import Time


@dataclass
class Attributable:
    """A single-tracklet attributable + photometry (DESIGN §2.1).

    Attributes
    ----------
    ra : float
        Topocentric right ascension alpha, ICRS/J2000 [rad].
    dec : float
        Topocentric declination delta, ICRS/J2000 [rad].
    ra_rate : float
        d(alpha)/dt — the RAW right-ascension rate, NOT multiplied by cos(dec)
        [rad/day].
    dec_rate : float
        d(delta)/dt [rad/day].
    epoch : astropy.time.Time
        Epoch of the attributable. Stored as given; all internal computation
        uses TDB.
    v_mag : float
        Apparent magnitude, standardized Johnson V [D-5].
    covariance : np.ndarray | None
        Optional 4x4 covariance of (ra, dec, ra_rate, dec_rate) in the units
        above. Accepted and carried, but UNUSED in v0 [D-6].
    """

    ra: float
    dec: float
    ra_rate: float
    dec_rate: float
    epoch: Time
    v_mag: float
    covariance: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Planetary ephemeris selection
# ---------------------------------------------------------------------------

#: Ephemerides tried, in order, by :func:`ephemeris_in_use`.
#:
#: ``"jpl"`` is astropy's JPL DE kernel (de440s by default). Selecting it
#: requires the optional ``jplephem`` package AND a one-time ~32 MB download of
#: ``de440s.bsp`` into the astropy cache (``~/.astropy/cache/download``); after
#: that first fetch it works offline. Neither ``jplephem`` nor the kernel is a
#: declared dependency of ds42 (IMPLEMENTATION §1 lists numpy/astropy/
#: matplotlib only), so in a stock install this falls through to ``"builtin"``.
#:
#: ``"builtin"`` is astropy's ERFA ``epv00`` analytic series (a fit to JPL
#: DE405). It needs no downloads and no extra packages. Measured against JPL
#: Horizons/DE441 it reproduces the heliocentric Earth state to ~9e-9 AU and
#: ~2e-10 AU/day over 2020-2026 -- two orders of magnitude inside the WP1
#: acceptance tolerances -- so it is the ephemeris ds42 v0 ships with.
EPHEMERIS_PREFERENCE: tuple[str, ...] = ("jpl", "builtin")

_ephemeris_cache: str | None = None


def ephemeris_in_use() -> str:
    """Return the name of the planetary ephemeris ds42 will use.

    The first entry of :data:`EPHEMERIS_PREFERENCE` that astropy can actually
    load, resolved once per process and then cached. Never raises: it falls back
    to ``"builtin"``, which is always available offline.
    """
    global _ephemeris_cache
    if _ephemeris_cache is None:
        for name in EPHEMERIS_PREFERENCE:
            try:
                # ScienceState.set() validates eagerly (for "jpl" this loads or
                # downloads the SPK kernel), and used as a context manager it
                # restores the previous global setting on exit.
                with solar_system_ephemeris.set(name):
                    pass
            except Exception:  # missing package, no network, bad kernel, ...
                continue
            _ephemeris_cache = name
            break
        else:  # pragma: no cover - "builtin" cannot fail
            _ephemeris_cache = "builtin"
    return _ephemeris_cache


# ---------------------------------------------------------------------------
# Observatory geolocation
# ---------------------------------------------------------------------------

#: Equatorial Earth radius [m] used by the MPC to normalise its parallax
#: constants (IAU/WGS84 equatorial radius, 6378.137 km).
MPC_EARTH_RADIUS_M: float = 6378137.0

#: Minimal built-in MPC observatory-code table.
#:
#: Values are (east longitude [deg], rho*cos(phi'), rho*sin(phi')), i.e. the MPC
#: "parallax constants": the site's geocentric cylindrical coordinates expressed
#: in units of :data:`MPC_EARTH_RADIUS_M`. They define an ITRS position directly,
#: with no reference ellipsoid involved:
#:
#:     x = rho*cos(phi') * cos(lon),  y = rho*cos(phi') * sin(lon),
#:     z = rho*sin(phi'),             all times MPC_EARTH_RADIUS_M.
#:
#: Source: Minor Planet Center, "List Of Observatory Codes",
#: https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
#: (retrieved 2026-08-31). A full obscode table is out of scope for v0
#: (IMPLEMENTATION §WP1); pass lon/lat/height explicitly for any other site.
MPC_OBSCODES: dict[str, tuple[float, float, float]] = {
    # code:  (lon_east_deg, rho_cos_phi, rho_sin_phi)   # site
    "500": (0.00000, 0.000000, 0.000000),  # Geocentric
    "309": (289.59569, 0.909943, -0.414336),  # Cerro Paranal
    "568": (204.52780, 0.941710, 0.337250),  # Maunakea
    "691": (248.39966, 0.849466, 0.526479),  # Steward Obs., Kitt Peak-Spacewatch
    "703": (249.26736, 0.845311, 0.533211),  # Catalina Sky Survey
    "704": (253.34093, 0.831869, 0.553542),  # Lincoln Laboratory ETS, New Mexico
    "807": (289.19410, 0.865600, -0.499800),  # Cerro Tololo Observatory
    "950": (342.11760, 0.877640, 0.478470),  # La Palma
    "F51": (203.74409, 0.936241, 0.351543),  # Pan-STARRS 1, Haleakala
    "G96": (249.21128, 0.845107, 0.533611),  # Mt. Lemmon Survey
    "I11": (289.26345, 0.865020, -0.500901),  # Gemini South, Cerro Pachon
    "T05": (203.74299, 0.936236, 0.351547),  # ATLAS-HKO, Haleakala
    "W84": (289.19358, 0.865572, -0.499793),  # Cerro Tololo-DECam
    "X05": (289.25058, 0.864981, -0.500958),  # Simonyi Survey Telescope, Rubin Obs.
}

#: MPC codes that denote the geocenter (no topocentric offset).
GEOCENTER_OBSCODES: frozenset[str] = frozenset({"500"})


def observatory_location(
    obscode: str | None = None,
    lon_deg: float | None = None,
    lat_deg: float | None = None,
    height_m: float = 0.0,
) -> EarthLocation | None:
    """Resolve observer arguments to an `~astropy.coordinates.EarthLocation`.

    Argument precedence matches :func:`observer_state`: ``obscode`` wins if
    given, otherwise geodetic (WGS84) ``lon_deg``/``lat_deg``/``height_m``,
    otherwise the geocenter.

    Returns
    -------
    EarthLocation or None
        ``None`` for the geocenter (obscode "500", or no arguments at all),
        signalling "no topocentric offset".
    """
    if obscode is not None:
        if lon_deg is not None or lat_deg is not None:
            raise ValueError(
                "give either obscode or lon_deg/lat_deg/height_m, not both"
            )
        code = str(obscode).strip().upper()
        if code in GEOCENTER_OBSCODES:
            return None
        try:
            lon, rho_cos_phi, rho_sin_phi = MPC_OBSCODES[code]
        except KeyError:
            raise KeyError(
                f"MPC observatory code {obscode!r} is not in the ds42 v0 "
                f"built-in table (known: {sorted(MPC_OBSCODES)}). Pass "
                f"lon_deg/lat_deg/height_m explicitly instead."
            ) from None
        lam = np.radians(lon)
        xyz = (
            np.array(
                [rho_cos_phi * np.cos(lam), rho_cos_phi * np.sin(lam), rho_sin_phi]
            )
            * MPC_EARTH_RADIUS_M
        )
        return EarthLocation.from_geocentric(*xyz, unit=u.m)

    if lon_deg is None and lat_deg is None:
        return None  # geocenter
    if lon_deg is None or lat_deg is None:
        raise ValueError("lon_deg and lat_deg must be given together")
    return EarthLocation.from_geodetic(
        lon=lon_deg * u.deg,
        lat=lat_deg * u.deg,
        height=height_m * u.m,
        ellipsoid="WGS84",
    )


def observer_state(
    epoch: Time,
    obscode: str | None = None,
    lon_deg: float | None = None,
    lat_deg: float | None = None,
    height_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric state of the observer in the EQUATORIAL (ICRS/J2000) frame.

    Parameters
    ----------
    epoch : astropy.time.Time
        Epoch (any scale; converted to TDB internally).
    obscode : str, optional
        MPC observatory code (e.g. "X05"). If given, overrides lon/lat/height.
        v0 must support at least: geocenter ("500") and code lookup from a
        small built-in table containing the codes used in tests; a full MPC
        obscode table is NOT required in v0.
    lon_deg, lat_deg, height_m : float, optional
        Geodetic (WGS84) observatory location, used when obscode is None.
        If neither is given, the GEOCENTER is used.

    Returns
    -------
    (R, V) : tuple of np.ndarray, each shape (3,)
        Heliocentric position [AU] and velocity [AU/day] of the observer,
        equatorial ICRS/J2000 axes. Uses astropy planetary ephemerides
        (builtin or JPL as available) + Earth rotation for the topocentric
        offset.

    Accuracy target (WP1 acceptance): match JPL Horizons to <= 1e-6 AU in
    position and <= 1e-8 AU/day in velocity for the test epochs.

    Notes
    -----
    Batch: ``epoch`` may be an array-valued `~astropy.time.Time`, in which case
    the returned arrays have shape (N, 3) (leading batch dimension, per
    IMPLEMENTATION §1). A scalar epoch returns shape (3,).

    The state is purely geometric: no light-time, aberration, or relativistic
    corrections are applied (matching Horizons ``VEC_CORR='NONE'``).
    """
    location = observatory_location(
        obscode=obscode, lon_deg=lon_deg, lat_deg=lat_deg, height_m=height_m
    )
    return _observer_state_from_location(epoch, location)


def _observer_state_from_location(
    epoch: Time, location: EarthLocation | None
) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric equatorial (ICRS) state for an `EarthLocation` (or geocenter)."""
    scalar = epoch.isscalar
    t = epoch.tdb

    ephemeris = ephemeris_in_use()
    with solar_system_ephemeris.set(ephemeris):
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", t)
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", t)

    # Geocenter relative to the Sun's centre, ICRS/equatorial axes.
    r = (earth_pos - sun_pos).get_xyz().to_value(u.AU)
    v = (earth_vel - sun_vel).get_xyz().to_value(u.AU / u.day)

    if location is not None:
        # Topocentric offset from Earth rotation.  ``get_gcrs_posvel`` returns the
        # site's geocentric position and velocity in GCRS, whose axes are aligned
        # with ICRS, so the offset can be added directly.  Its velocity is the
        # analytic Earth-rotation cross product omega x r (astropy rotates the
        # CIP-aligned rotation vector into GCRS); this contributes ~2.3e-4 AU/day,
        # four orders of magnitude above the 1e-8 AU/day tolerance.
        dpos, dvel = location.get_gcrs_posvel(t)
        r = r + dpos.get_xyz().to_value(u.AU)
        v = v + dvel.get_xyz().to_value(u.AU / u.day)

    # (3,) for a scalar epoch; (N, 3) for a batch.
    if scalar:
        return np.ascontiguousarray(r), np.ascontiguousarray(v)
    return np.ascontiguousarray(r.T), np.ascontiguousarray(v.T)
