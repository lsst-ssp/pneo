"""WP1 unit + acceptance tests: Attributable and observer heliocentric state.

Acceptance (IMPLEMENTATION §WP1): the observer state must match JPL Horizons
reference vectors to <= 1e-6 AU and <= 1e-8 AU/day, and epoch handling must be
time-scale agnostic.

--------------------------------------------------------------------------
PROVENANCE OF THE REFERENCE VECTORS
--------------------------------------------------------------------------
All reference vectors below were fetched from the JPL Horizons API on
2026-08-31 (Horizons API v1.2; target/center ephemeris source: DE441) and are
hard-coded here so the test suite needs no network at run time.

Common query parameters (URL-encoded onto
https://ssd.jpl.nasa.gov/api/horizons.api):

    format=text  OBJ_DATA='NO'  MAKE_EPHEM='YES'  EPHEM_TYPE='VECTORS'
    REF_PLANE='FRAME'   <-- ICRF/equatorial axes, NOT the default ecliptic
    VEC_TABLE='2'       <-- position + velocity
    OUT_UNITS='AU-D'    <-- AU and AU/day
    VEC_CORR='NONE'     <-- geometric state, no light-time/aberration
    TLIST='<JD TDB>'

Geocenter rows (Earth centre relative to the Sun's centre) use

    COMMAND='399'   CENTER='500@10'

and are stored exactly as returned.

Site rows are obtained the other way round -- the Sun as seen from a
user-defined site on Earth -- and NEGATED to give the site's heliocentric
state:

    COMMAND='10'  CENTER='coord@399'
    COORD_TYPE='CYLINDRICAL'  SITE_COORD='<lon_deg>,<Dxy_km>,<Dz_km>'

where (Dxy, Dz) = (rho*cos(phi'), rho*sin(phi')) * 6378.137 km are the MPC
parallax constants for the observatory code, expressed in km. Feeding Horizons
cylindrical (rather than geodetic) coordinates makes the comparison independent
of any reference-ellipsoid convention: both sides use exactly the MPC
constants. Horizons echoed back "Center pole/equ: ITRF93".

Two rows instead use COORD_TYPE='GEODETIC' with
SITE_COORD='<lon_deg>,<lat_deg>,<alt_km>' to exercise the explicit WGS84
lon/lat/height entry point (Horizons reports "Center radii: 6378.137, 6378.137,
6356.752 km", i.e. WGS84 to the metre).

Example full URL (the X05 / 2025-01-01 row):

https://ssd.jpl.nasa.gov/api/horizons.api?format=text&OBJ_DATA='NO'&MAKE_EPHEM='YES'
&EPHEM_TYPE='VECTORS'&REF_PLANE='FRAME'&VEC_TABLE='2'&OUT_UNITS='AU-D'
&VEC_CORR='NONE'&COMMAND='10'&CENTER='coord@399'&COORD_TYPE='CYLINDRICAL'
&SITE_COORD='289.25058,5516.967320,-3195.178755'&TLIST='2460676.5'
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from ds42.attributable import (
    MPC_OBSCODES,
    Attributable,
    ephemeris_in_use,
    observatory_location,
    observer_state,
)

# WP1 acceptance tolerances.
POS_TOL_AU = 1e-6
VEL_TOL_AU_PER_DAY = 1e-8

# Epochs, JD in the TDB scale.
JD1 = 2460676.5  # 2025-Jan-01 00:00:00.0000 TDB
JD2 = 2459016.0  # 2020-Jun-15 12:00:00.0000 TDB
JD3 = 2461109.75  # 2026-Mar-10 06:00:00.0000 TDB

# --------------------------------------------------------------------------
# Hard-coded Horizons reference states: (R [AU], V [AU/day]), heliocentric,
# ICRS/equatorial axes, geometric.
# --------------------------------------------------------------------------

HORIZONS_GEOCENTER = {
    JD1: (
        (-1.786834409731047e-01, 8.872096932094631e-01, 3.845967881324398e-01),
        (-1.720473858166942e-02, -2.930011586551128e-03, -1.270309542557479e-03),
    ),
    JD2: (
        (-9.626351059222485e-02, -9.278571381109199e-01, -4.022208231370140e-01),
        (1.684904389849250e-02, -1.560215085162827e-03, -6.766266865044715e-04),
    ),
    JD3: (
        (-9.758193593141539e-01, 1.690730187510102e-01, 7.329926958932281e-02),
        (-3.477711142719202e-03, -1.556774997255541e-02, -6.748291236569949e-03),
    ),
}

# (obscode, JD) -> (R, V)
HORIZONS_SITE = {
    ("X05", JD1): (
        (-1.786514083772820e-01, 8.872278753431714e-01, 3.845753512257709e-01),
        (-1.731929738581693e-02, -2.727866276478572e-03, -1.270038179475775e-03),
    ),
    ("X05", JD2): (
        (-9.622760885154263e-02, -9.278488856815507e-01, -4.022422516361297e-01),
        (1.679705128270542e-02, -1.333756885826855e-03, -6.765237914920704e-04),
    ),
    ("X05", JD3): (
        (-9.758560512305392e-01, 1.690688064135023e-01, 7.327800512051809e-02),
        (-3.451176682238507e-03, -1.579857990222727e-02, -6.748350802345227e-03),
    ),
    ("F51", JD1): (
        (-1.786610635346251e-01, 8.871766145843784e-01, 3.846117230545250e-01),
        (-1.699632779816186e-02, -2.789254081120614e-03, -1.270820742841763e-03),
    ),
    ("703", JD2): (
        (-9.623136947279268e-02, -9.278735296196544e-01, -4.021981523531256e-01),
        (1.695231572492402e-02, -1.357993005374764e-03, -6.768272583132461e-04),
    ),
    ("W84", JD3): (
        (-9.758560803104435e-01, 1.690688400129475e-01, 7.327805486391310e-02),
        (-3.451388359814407e-03, -1.579876391773574e-02, -6.748350254426486e-03),
    ),
}

# Explicit WGS84 geodetic sites: (lon_deg_east, lat_deg, height_m, JD) -> (R, V)
HORIZONS_GEODETIC = {
    # Cerro Tololo, ~coincident with obscode W84 (used as a cross-check that the
    # obscode and lon/lat/height entry points agree).
    (289.19350, -30.16960, 2215.0, JD3): (
        (-9.758560803848475e-01, 1.690688400563420e-01, 7.327805481910826e-02),
        (-3.451388633225912e-03, -1.579876438578692e-02, -6.748350253710499e-03),
    ),
    # An arbitrary northern-hemisphere site, well away from any table entry.
    (7.00000, 46.50000, 1500.0, JD1): (
        (-1.786921063468338e-01, 8.872377722361310e-01, 3.846275893020096e-01),
        (-1.738164012069246e-02, -2.985078117989556e-03, -1.269877849344378e-03),
    ),
}


def _tdb(jd: float) -> Time:
    return Time(jd, format="jd", scale="tdb")


def _check(got, ref, label):
    r, v = got
    ref_r, ref_v = np.asarray(ref[0]), np.asarray(ref[1])
    dr = np.abs(r - ref_r).max()
    dv = np.abs(v - ref_v).max()
    assert dr <= POS_TOL_AU, f"{label}: |dR|={dr:.3e} AU > {POS_TOL_AU:.0e}"
    assert dv <= VEL_TOL_AU_PER_DAY, (
        f"{label}: |dV|={dv:.3e} AU/day > {VEL_TOL_AU_PER_DAY:.0e}"
    )
    return dr, dv


# --------------------------------------------------------------------------
# Acceptance: agreement with JPL Horizons
# --------------------------------------------------------------------------


@pytest.mark.parametrize("jd", sorted(HORIZONS_GEOCENTER))
def test_geocenter_matches_horizons(jd):
    _check(observer_state(_tdb(jd)), HORIZONS_GEOCENTER[jd], f"geocenter @ {jd}")


@pytest.mark.parametrize(("code", "jd"), sorted(HORIZONS_SITE))
def test_topocentric_matches_horizons(code, jd):
    _check(observer_state(_tdb(jd), obscode=code), HORIZONS_SITE[(code, jd)],
           f"{code} @ {jd}")


@pytest.mark.parametrize("key", sorted(HORIZONS_GEODETIC))
def test_geodetic_site_matches_horizons(key):
    lon, lat, height, jd = key
    got = observer_state(_tdb(jd), lon_deg=lon, lat_deg=lat, height_m=height)
    _check(got, HORIZONS_GEODETIC[key], f"geodetic {lon},{lat} @ {jd}")


def test_obscode_500_is_geocenter():
    """MPC code "500" must reproduce the geocentric reference exactly."""
    r_code, v_code = observer_state(_tdb(JD1), obscode="500")
    r_def, v_def = observer_state(_tdb(JD1))
    np.testing.assert_array_equal(r_code, r_def)
    np.testing.assert_array_equal(v_code, v_def)
    _check((r_code, v_code), HORIZONS_GEOCENTER[JD1], "obscode 500")


def test_obscode_and_geodetic_entry_points_agree():
    """W84's parallax constants and its WGS84 lon/lat/height agree closely."""
    t = _tdb(JD3)
    r_code, v_code = observer_state(t, obscode="W84")
    r_geo, v_geo = observer_state(t, lon_deg=289.19350, lat_deg=-30.16960,
                                  height_m=2215.0)
    # Same physical site described two ways; the residual is the ~10 m mismatch
    # between W84's MPC constants and the geodetic coordinates chosen here, far
    # below the 1e-6 AU (150 km) acceptance tolerance.
    sep_m = np.linalg.norm(r_code - r_geo) * 1.495978707e11
    assert sep_m < 100.0, f"site definitions differ by {sep_m:.1f} m"
    assert np.abs(v_code - v_geo).max() < 1e-9


# --------------------------------------------------------------------------
# Epoch / time-scale handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ["utc", "tt", "tai", "tdb"])
@pytest.mark.parametrize("obscode", [None, "X05"])
def test_epoch_scale_roundtrip(scale, obscode):
    """The same physical instant expressed in any scale gives the same state."""
    t_tdb = _tdb(JD1)
    t_other = getattr(t_tdb, scale)
    assert t_other.scale == scale
    r_a, v_a = observer_state(t_tdb, obscode=obscode)
    r_b, v_b = observer_state(t_other, obscode=obscode)
    # Only float round-off in the scale conversion should separate these.
    assert np.abs(r_a - r_b).max() < 1e-13
    assert np.abs(v_a - v_b).max() < 1e-14


def test_utc_is_not_treated_as_tdb():
    """Sanity: the TDB-TT-UTC offset is real, so mislabelling would show up."""
    t_tdb = _tdb(JD1)
    # A Time with the *same numerical JD* but scale="utc" is a different instant
    # (~69 s later in TDB) and must give a measurably different state.
    t_wrong = Time(JD1, format="jd", scale="utc")
    r_a, _ = observer_state(t_tdb, obscode="X05")
    r_b, _ = observer_state(t_wrong, obscode="X05")
    sep = np.linalg.norm(r_a - r_b)
    assert sep > 1e-9, "time-scale handling appears to be a no-op"


# --------------------------------------------------------------------------
# Topocentric offset: magnitude and velocity consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["X05", "W84", "703", "F51"])
def test_topocentric_offset_magnitude(code):
    """Offset from the geocenter must be ~1 Earth radius (~4.26e-5 AU)."""
    t = _tdb(JD1)
    r_topo, v_topo = observer_state(t, obscode=code)
    r_geo, v_geo = observer_state(t)
    d_pos = np.linalg.norm(r_topo - r_geo)
    d_vel = np.linalg.norm(v_topo - v_geo)

    lon, rho_cos, rho_sin = MPC_OBSCODES[code]
    rho = np.hypot(rho_cos, rho_sin)  # site radius in Earth radii
    expected = rho * 6378137.0 / 1.495978707e11
    assert d_pos == pytest.approx(expected, rel=1e-6)
    assert 4.0e-5 < d_pos < 4.4e-5

    # Rotation speed: omega * rho*cos(phi') * R_earth, in AU/day.
    omega_per_day = 7.292115855306589e-5 * 86400.0
    expected_v = omega_per_day * rho_cos * 6378137.0 / 1.495978707e11
    assert d_vel == pytest.approx(expected_v, rel=1e-4)
    # ...and it is four orders of magnitude above the WP1 velocity tolerance,
    # i.e. it cannot be neglected.
    assert d_vel > 1e4 * VEL_TOL_AU_PER_DAY


@pytest.mark.parametrize("code", ["X05", "703"])
def test_topocentric_velocity_matches_finite_difference(code):
    """The analytic omega x r velocity equals d/dt of the offset over +/- 1 s."""
    t = _tdb(JD2)
    dt = TimeDelta(1.0, format="sec")
    dt_day = dt.to_value("day")

    def offset(time):
        r_topo, v_topo = observer_state(time, obscode=code)
        r_geo, v_geo = observer_state(time)
        return r_topo - r_geo, v_topo - v_geo

    _, v_analytic = offset(t)
    r_plus, _ = offset(t + dt)
    r_minus, _ = offset(t - dt)
    v_fd = (r_plus - r_minus) / (2.0 * dt_day)

    assert np.abs(v_analytic - v_fd).max() < 1e-9
    # And the check is meaningful at the required precision.
    assert np.linalg.norm(v_analytic) > 1e-4


def test_geocenter_velocity_finite_difference():
    """Geocentric heliocentric velocity is consistent with d/dt of position."""
    t = _tdb(JD1)
    h = TimeDelta(60.0, format="sec")
    r_p, _ = observer_state(t + h)
    r_m, _ = observer_state(t - h)
    _, v = observer_state(t)
    h_day = h.to_value("day")
    assert np.abs((r_p - r_m) / (2.0 * h_day) - v).max() < 1e-9


# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------


def test_shapes_and_dtypes():
    r, v = observer_state(_tdb(JD1), obscode="X05")
    assert r.shape == (3,) and v.shape == (3,)
    assert r.dtype == np.float64 and v.dtype == np.float64


def test_batch_epochs():
    """An array-valued Time yields (N, 3) states matching the scalar calls."""
    jds = [JD1, JD2, JD3]
    t = Time(jds, format="jd", scale="tdb")
    r, v = observer_state(t, obscode="X05")
    assert r.shape == (3, 3) and v.shape == (3, 3)
    for i, jd in enumerate(jds):
        r_i, v_i = observer_state(_tdb(jd), obscode="X05")
        np.testing.assert_allclose(r[i], r_i, rtol=0, atol=1e-15)
        np.testing.assert_allclose(v[i], v_i, rtol=0, atol=1e-16)


def test_obscode_is_case_insensitive_and_stripped():
    t = _tdb(JD1)
    ref = observer_state(t, obscode="X05")
    for spelling in ("x05", " X05 ", "x05 "):
        got = observer_state(t, obscode=spelling)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])


def test_unknown_obscode_raises():
    with pytest.raises(KeyError, match="not in the ds42 v0"):
        observer_state(_tdb(JD1), obscode="ZZZ")


def test_conflicting_location_arguments_raise():
    with pytest.raises(ValueError, match="not both"):
        observer_state(_tdb(JD1), obscode="X05", lon_deg=0.0, lat_deg=0.0)


def test_lon_without_lat_raises():
    with pytest.raises(ValueError, match="together"):
        observer_state(_tdb(JD1), lon_deg=10.0)


def test_observatory_location_geocenter_is_none():
    assert observatory_location() is None
    assert observatory_location(obscode="500") is None
    assert observatory_location(obscode="X05") is not None


def test_observatory_location_matches_mpc_constants():
    """The built-in table round-trips through EarthLocation geocentric xyz."""
    for code, (lon, rho_cos, rho_sin) in MPC_OBSCODES.items():
        loc = observatory_location(obscode=code)
        if loc is None:
            assert code == "500"
            continue
        xyz = np.array([loc.x.to_value("m"), loc.y.to_value("m"),
                        loc.z.to_value("m")]) / 6378137.0
        assert np.hypot(xyz[0], xyz[1]) == pytest.approx(rho_cos, rel=1e-12,
                                                         abs=1e-12)
        assert xyz[2] == pytest.approx(rho_sin, rel=1e-12, abs=1e-12)
        assert np.degrees(np.arctan2(xyz[1], xyz[0])) % 360.0 == pytest.approx(
            lon % 360.0, abs=1e-9
        )


def test_ephemeris_in_use_is_known_and_stable():
    name = ephemeris_in_use()
    assert name in ("jpl", "builtin")
    assert ephemeris_in_use() is name  # cached, no repeated download attempts


# --------------------------------------------------------------------------
# Attributable dataclass
# --------------------------------------------------------------------------


def test_attributable_fields():
    t = _tdb(JD1)
    a = Attributable(
        ra=1.234, dec=-0.567, ra_rate=1e-3, dec_rate=-2e-4, epoch=t, v_mag=21.5
    )
    assert a.ra == 1.234
    assert a.dec == -0.567
    assert a.ra_rate == 1e-3
    assert a.dec_rate == -2e-4
    assert a.epoch is t
    assert a.v_mag == 21.5
    assert a.covariance is None


def test_attributable_carries_covariance():
    cov = np.diag([1e-12, 1e-12, 1e-14, 1e-14])
    a = Attributable(0.0, 0.0, 0.0, 0.0, _tdb(JD1), 20.0, covariance=cov)
    np.testing.assert_array_equal(a.covariance, cov)
    assert a.covariance.shape == (4, 4)


def test_attributable_epoch_is_stored_as_given():
    """Epoch is stored verbatim (contract); conversion to TDB is internal."""
    t_utc = Time("2025-01-01T00:00:00", scale="utc")
    a = Attributable(0.1, 0.2, 0.0, 0.0, t_utc, 20.0)
    assert a.epoch.scale == "utc"
    r, v = observer_state(a.epoch, obscode="X05")
    r2, v2 = observer_state(a.epoch.tdb, obscode="X05")
    assert np.abs(r - r2).max() < 1e-13
    assert np.abs(v - v2).max() < 1e-14
