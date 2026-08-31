"""Physical constants and unit conventions. Single source of truth (IMPLEMENTATION §1).

Conventions used throughout digest42:
  - Length: AU. Time: day (TDB). Angles: radian.
  - State vectors are heliocentric. Frame is stated per function: "equ" = equatorial
    ICRS/J2000; "ecl" = ecliptic J2000 (mean obliquity below).
  - Orbital elements (a, e, i, Omega, omega, M) are heliocentric ecliptic J2000,
    angles in radians, a in AU, at the stated epoch.
  - Magnitudes: Johnson V; absolute magnitude H in V.
"""

# Gaussian gravitational constant [AU^(3/2) / day]; mu_sun = K_GAUSS**2 [AU^3/day^2]
K_GAUSS: float = 0.01720209895
MU_SUN: float = K_GAUSS * K_GAUSS

# Mean obliquity of the ecliptic at J2000.0 (IAU 1976/2000): 84381.448 arcsec [rad]
import math as _math

OBLIQUITY_J2000: float = 84381.448 / 3600.0 * _math.pi / 180.0

# Speed of light [AU/day] (for future light-time treatment; unused in v0 kinematics)
C_LIGHT: float = 173.144632674240

# Default slope parameter of the IAU H,G phase function [D-5]
G_SLOPE_DEFAULT: float = 0.15

# NEO boundary: perihelion distance [AU]
Q_NEO: float = 1.3
