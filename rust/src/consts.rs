//! Physical constants and convention switches, mirrored from
//! `src/pneo/constants.py` and `src/pneo/transforms.py`.
//!
//! Every value here is a *copy* of the Python single source of truth, not a
//! re-derivation: the two backends must agree bit-for-bit on conventions.

/// Gaussian gravitational constant [AU^(3/2)/day] (constants.py K_GAUSS).
pub const K_GAUSS: f64 = 0.01720209895;

/// mu_sun = K_GAUSS^2 [AU^3/day^2] (constants.py MU_SUN).
pub const MU_SUN: f64 = K_GAUSS * K_GAUSS;

/// Mean obliquity of the ecliptic at J2000.0 [rad]: 84381.448 arcsec.
/// Written exactly as constants.py does it (`84381.448 / 3600.0 * pi / 180.0`)
/// so the rounding sequence is identical.
pub const OBLIQUITY_J2000: f64 =
    84381.448 / 3600.0 * std::f64::consts::PI / 180.0;

/// Two-term IAU H,G phase-function coefficients (transforms.py `_HG_A`/`_HG_B`).
pub const HG_A1: f64 = 3.33;
pub const HG_A2: f64 = 1.87;
pub const HG_B1: f64 = 0.63;
pub const HG_B2: f64 = 1.22;

/// Conditioning ceiling of `transforms._COND_MAX`.
pub const COND_MAX: f64 = 1e12;

/// mu^(3/2), the Delaunay-Jacobian prefactor.
#[inline(always)]
pub fn mu_32() -> f64 {
    MU_SUN * MU_SUN.sqrt()
}

/// cos/sin of the obliquity (transforms.py `_CE`/`_SE`).
#[inline(always)]
pub fn obliquity_cos_sin() -> (f64, f64) {
    (OBLIQUITY_J2000.cos(), OBLIQUITY_J2000.sin())
}
