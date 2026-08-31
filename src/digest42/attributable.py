"""WP1 — Attributable value type and observer heliocentric state.

Frozen contract (WP0). Implementations must preserve every public signature,
unit, and frame documented here. See DESIGN.md §2.1, IMPLEMENTATION.md §1/WP1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    """
    raise NotImplementedError("WP1")
