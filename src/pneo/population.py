"""WP4 — Population density interface and the digest2 histogram implementation.

Frozen contract (WP0). See docs/design/DESIGN.md §3, docs/design/IMPLEMENTATION.md WP4, D-10.

Semantics of the digest2 population model
-----------------------------------------
The MPC-maintained ``digest2.model.csv`` is a 4-D histogram over
(q, e, i, H) built from the Pan-STARRS Synthetic Solar System Model (S3M;
Grav et al. 2011) plus a known-object catalogue (astorb).  Everything below
was verified against the primary sources rather than inferred from the file:

* ``digest2/digest2/d2model.c`` — the partition arrays and the binning rule
  (``qeiToBin`` / ``hToBin``), and the orbit-class membership tests.
* ``digest2/population/make_population/muk.c`` and its modern Python
  reimplementation ``digest2/src/digest2/population.py`` (``_volume_scale``)
  — how the stored cell values are produced from raw object counts.
* ``digest2/population/make_population/s3mbin.c`` — how S3M orbits are binned.

(all in https://github.com/Smithsonian/mpc-public, and Keys et al. 2019,
PASP 131, 064501.)

**Bin edges are UPPER edges; the lower edge of the first bin is 0 in every
dimension.**  digest2 assigns a bin with ``for (b = 0; x >= part[b]; ++b)``,
i.e. bin ``b`` covers ``[part[b-1], part[b])`` with ``part[-1] == 0``.  So the
CSV's ``Q``/``e``/``i`` columns and its ``H*`` column labels name the *upper*
edge of the bin, the first q bin is [0, 0.4) AU, the first e bin is [0, 0.1),
the first i bin is [0, 2) deg and the first H bin is [0, 6) mag.  The zero
lower edges are not a guess: ``muk.c`` uses exactly ``q0 = e0 = i0 = h0 = 0``
when it computes bin volumes (see below), so they are the file's own
convention.

**The stored cell values are NOT object counts.**  ``muk.c`` divides every
cell by the square root of a bin "volume" before writing the CSV::

    dq   = q1 - q0
    dae  = dq * (e1 - e0) / ((1 - e0) + (1 - e1))     # (1-e) clipped to 0 for e >= 1
    daei = dae * (i1 - i0)                            # i in DEGREES
    isqv = 1 / sqrt(daei * (h1 - h0))
    cell = count * isqv

(the comment in ``muk.c`` explains why: digest2 accumulates a 2-D slice
through the 4-D histogram, so it wants a per-slice count.)  We therefore
undo the scaling to recover absolute object counts, which is what an
*absolutely calibrated* density needs (DESIGN §2.4, D-3).  The recovered
values are integers to < 2e-9 absolute in every one of the 45 936 x 32 cells,
which is the decisive confirmation of the inversion (it stands on its own —
nothing else is needed to establish it); ``read_digest2_model`` checks this
at load time.  With the scaling undone the "All"/"SS" table sums to
14 542 147 objects — the model file's own total — whereas the raw cell values
sum to a meaningless 6.3e7.

That total is *not* claimed to equal S3M's raw object count: ``s3mbin.c``
silently drops orbits outside the model box (q <= 0, e > 1.1, i >= 180, and
anything binning out of range), and ``muk.c`` then raises individual cells to
the known-catalogue count wherever ``known > all`` (4 375 bins holding
640 904 objects have All > 0 with Unk == 0, the signature of that fix-up).
The file's total is therefore the total of the *model as distributed*, which
is exactly what an absolutely calibrated prior should be normalized to.

**Class overlap.**  digest2's 15 orbit classes do *not* partition the solar
system: ``Int`` ("MPC interesting") overlaps everything, ``N18`` and ``N22``
are H-limited subsets of ``NEO``, and the belt classes neither tile nor
cover (their union over-counts ``SS - NEO`` by 1.2% while leaving parts of
element space in no class at all).  ``SS`` is the aggregate of *all* objects.
The only clean two-component split is therefore

    NEO   = the ``NEO`` class   (digest2's definition: q < 1.3 AU exactly)
    other = ``SS`` - ``NEO``    (cell-by-cell)

which is exact by construction (residual identically zero) and, because
q = 1.3 AU is a bin edge (``qpart[7]``), is exactly complementary there:
``NEO`` has zero counts in every q bin above 1.3 AU and ``SS - NEO`` has zero
counts in every q bin below it, as verified numerically in the unit tests.
This satisfies D-2 / DESIGN §2.4.

**Deviations from digest2's own lookup (both required by D-4).**

1. digest2's ``hToBin`` clamps any H >= 24 into the last bin, i.e. it treats
   the top H bin as [24, inf).  We treat it as [24, 25.5) and return exactly
   0 above 25.5 (hard truncation, no extrapolation).
2. digest2's ``qeiToBin`` loop also puts *negative* q, e or i into bin 0
   (``x >= part[0]`` is false).  We return 0 there: the declared support
   starts at 0 in every dimension and negative q/e/i/H are unphysical.

**The digest2-emulation knob (WP7 addendum).**  Deviation 1 is the only place
where the two codes consume *different* effective models on identical inputs,
which breaks the DESIGN §4.2 "same inputs" premise for any tracklet with
appreciable admissible-region measure at implied H > 25.5 (64.5% of the
gate_v0 catalogue at the > 1e-2 threshold).  To turn that confound into a
measurement, :class:`Digest2Histogram` takes ``h_top_bin_infinite=True``,
which reinstates digest2's clamp for the **upper** half of deviation 1 only:
H >= 25.5 returns the top bin's density (per unit H, the value unchanged)
instead of 0.  Everything else — q/e/i truncation, and H < 0 — is untouched.

The *lower* half of digest2's clamp is deliberately NOT emulated: digest2's
``for (ih = 0; h >= hpart[ih] && ih < HX - 1; ++ih)`` also leaves negative H
in bin 0, so digest2 treats the first H bin as (-inf, 6).  Implied H < 0 means
an object brighter than absolute magnitude 0 — at the geometries pneo
integrates over (rho >= 0.01 AU, V >= 14) that never occurs, so emulating it
would change nothing measurable while making the comparison mode a
two-sided deviation from D-4 instead of the one-sided one under study.
``h_top_bin_infinite=True`` therefore still returns 0 for H < 0.

Clamped components are a **comparison/emulation mode only, never a gate
mode** — see the class docstring for why (the clamped density is not
normalizable, so DESIGN §4.1 has no generative counterpart for it).

**Smooth-in-H reconstruction (``h_interpolation="pchip"``).**  The default
evaluation is piecewise constant in all four variables, which puts an
H-bin-edge discontinuity of the integrand on every ρ-aligned curve
``H(ρ) = V - 5 log10(ρ r(ρ)) + 2.5 log10 Φ(α(ρ))``.  DESIGN §2.5 measures
the consequence: the ρ-axis staircase, not the ``1/(e sin i)``
integrable singularity, dominates the quadrature error, forcing ρ-heavy
grids (n_ρ ≈ 1200) for p90 |Δp| ≈ 3.5e-3.  ``h_interpolation="pchip"``
removes the H staircase (it does **nothing** about the ``1/(e sin i)``
caveat, which lives on the e and i bin edges and is a property of the
*measure*, not of the H reconstruction).

The construction is deliberately a reconstruction of the *cumulative*
distribution, not of the density:

1. For each (q, e, i) cell, form the cumulative object count over the 19
   H edges, ``C(0) = 0`` … ``C(25.5) = cell total`` — 18 bin counts, 19
   knots, and the knot values are exact (integer counts, cumsum in double).
2. Fit a **monotone** piecewise cubic through those 19 knots
   (PCHIP: Fritsch–Carlson harmonic-mean interior slopes with the standard
   one-sided three-point end slopes and the 3Δ limiter).  The data are
   non-decreasing because counts are non-negative, so the interpolant is
   non-decreasing.
3. The density is the *derivative* of that cubic, divided by the cell's
   ``dq · de · di[rad]`` volume and by ``(2 π)^3`` — exactly the divisors the
   piecewise-constant mode uses, minus the ``dH`` that the derivative
   supplies.

Three properties follow, and they are what makes this safe rather than a
heuristic smoothing:

* **Mass is conserved per H bin, exactly.**  ``∫_{H_k}^{H_{k+1}} C'(H) dH =
  C(H_{k+1}) - C(H_k) = counts[k]``.  Interpolating the CDF and
  differentiating is the only construction with this property; interpolating
  the *density* values does not have it.
* **No mass leaks between bins.**  Monotonicity is the mechanism: because
  ``C`` is non-decreasing and pinned to the exact cumulative count at every
  knot, the mass in ``[H_k, H_{k+1})`` is ``counts[k]`` whatever the
  neighbouring counts are.  A cell with counts in one H bin only keeps all of
  its mass in that bin (the neighbouring intervals get ``Δ = 0``, both end
  slopes 0, and a cubic that is *constant* — density identically zero).  This
  is the WP4-review "interpolation invents density" objection, answered
  structurally: nothing can be invented outside a bin without violating an
  interpolation condition.
* **Non-negativity.**  ``C`` monotone ⇒ ``C' ≥ 0``.  Concretely, PCHIP's
  slopes satisfy ``0 ≤ m_k ≤ 3 min(Δ_{k-1}, Δ_k)``, which is inside the
  Fritsch–Carlson monotonicity region for every interval.  Scope, precisely:
  the density is non-negative **as evaluated through** :meth:`density`, which
  is what the classifier and the sampler use, and measured exactly ``0.0`` as
  its minimum over a dense sweep of every cell.  Evaluating the stored
  coefficient triple directly at ``s = dH`` — the right-hand knot, which
  :meth:`density` never reaches, because the H bins are half-open and ``s``
  stays strictly below ``dH`` — can return ~-1e-10 *relative* to the cell
  scale, from cancellation in the Horner form at a double root of the
  derivative.  That value is unreachable through the public API and is not
  clamped, so a genuine sign error could not hide behind a clamp.

D-4 truncation is untouched: the density is exactly ``0.0`` outside
``[0, 25.5)`` and outside the q/e/i support, and the q/e/i lookup is
bit-identical to the histogram mode.  The reconstructed density *is*
generally discontinuous at the two support endpoints H = 0 and H = 25.5
(the interpolant's one-sided slope there is not zero), which is the same
kind of edge the histogram mode has and is not a defect of the smoothing;
across every *interior* H edge the density is continuous (PCHIP is C¹, so
its derivative is C⁰).

``h_interpolation`` is mutually exclusive with ``h_top_bin_infinite``: the
D-14 clamp is a histogram-emulation device (it extends a *constant* past the
support), and a smooth density that is suddenly constant beyond 25.5 is
neither a reconstruction of the model nor an emulation of digest2.
Constructing both together raises ``ValueError``.

**Finite H extension (``h_extend_to``) [D-19].**  The D-14 clamp buys the
NEOCP operating point (validation/NEOCP_PILOT.md: 11 of 499 real discovery
tracklets — 100% of them true NEOs — have their *entire* admissible region at
implied H > 25.5 and score 0/0 under D-4) at the price of a non-normalizable
component: it is gate-ineligible, cannot be sampled, and cannot compose with
``h_interpolation``.  ``h_extend_to`` buys the same operating point without
paying that price.

Construction, in full.  With ``h_extend_to = X > 25.5`` every (q, e, i) cell
gains **one synthetic 19th H bin** ``[25.5, X)``, whose object count is a flat
continuation of the top bin's per-unit-H density::

    counts[..., 18] = counts[..., 17] / 1.5 * (X - 25.5)

(1.5 mag is the width of the model's top bin [24, 25.5)).  The density is
exactly zero at and above ``X`` — this is still D-4 hard truncation, just at a
declared boundary further out; nothing is extrapolated *without* a declared
support, and the extension has finite mass, so the component stays a
probability distribution.  Consequences, each of which is a test:

* **Exact continuity at H = 25.5.**  In histogram mode the synthetic bin's
  entry of the density table is set to the top bin's entry *bit-for-bit*, so
  ``density(nextafter(25.5, 0)) == density(25.5)`` exactly, cell by cell.
  (Computing it instead as ``counts[..., 18] / (bin_volume * (2 pi)^3)`` is the
  same number to <= 1 ulp — 3.5e-16 relative at worst over the shipped model,
  bit-equal in 296 of the 546 non-empty NEO cells — but "to 1 ulp" is not
  "by construction", so the table entry is copied rather than recomputed.  The
  1-ulp residual moves to the count/density relation of that one bin, where it
  is harmless and is asserted rather than hidden.)
* **Everything below 25.5 is bit-identical to the unextended model** in
  histogram mode: the first 18 volume slabs are *the same array* (the extended
  bin-volume array is ``concatenate((BIN_VOLUME, dq de di dH_ext))``), and the
  first 18 count slabs are the input counts, so the quotient is bit-for-bit
  what the unextended component computes.  In **pchip** mode the identity holds
  only for ``H < 24``: the extra cumulative knot at ``X`` turns the knot at
  25.5 from an *end* knot into an *interior* one, which changes that knot's
  slope and hence the cubic on ``[24, 25.5)``.  The mass in ``[24, 25.5)`` is
  still exactly ``counts[..., 17]`` — PCHIP interpolates the cumulative count,
  so per-bin mass conservation is structural and cannot be disturbed by adding
  a knot — but the *shape* inside that one bin does change.  This is stated
  because it is the one place the "nothing below 25.5 moves" claim needs a
  qualifier.

  **How big is that change?**  Measured over every populated cell at 61 H
  samples across ``[24, 25.5)``, as ``|d_extended - d_plain|`` divided by that
  bin's *mean* density (``count / (1.5 · dq de di (2π)³)`` — the scale-free
  denominator, and the one the histogram mode would report):
  median **5.4%**, p90 **15.9%**, max **93.5%** for the NEO component;
  8.6% / 25.5% / 93.5% for "other".  Divided instead by the *pointwise*
  unextended density the same statistic reads 5.6% / 15.7% / **2222%**, and the
  tail is an artifact of the denominator, not of the reconstruction: it occurs
  only where the unextended cubic's derivative is near one of its own zeros, so
  a small absolute change is a large relative one.  Mass is exactly preserved
  either way.  Both metrics are reported because neither alone is honest.
* **``h_interpolation="pchip"`` composes.**  The extension is simply one more
  cumulative knot: 20 edges, 19 knots + the ``C(0) = 0`` pin, 19 intervals.
  What is *guaranteed* is what PCHIP always guarantees, and the guarantee does
  not include flatness — the defined behaviour inside ``[25.5, X)`` is a
  mass-conserving monotone cubic through the extra knot, whatever shape that
  takes: the mass in the synthetic bin is exactly ``counts[..., 18]``, the
  density is non-negative, and it is continuous across H = 25.5 (PCHIP is C¹,
  so its derivative is C⁰ across every *interior* knot — and 25.5 is now
  interior).

  What actually *happens*, as a consequence of this particular extension count
  rather than of the interpolator, is that the cubic degenerates to a straight
  line and the reconstructed density in the extension is flat after all,
  equal to the histogram mode's constant to floating-point noise.  The reason
  is worth writing down because it is a one-line proof: the extension
  interval's secant is ``[counts[17] * (w/1.5)] / w = counts[17] / 1.5``, which
  is exactly the top bin's secant, so PCHIP's harmonic mean at the 25.5 knot
  and its three-point end slope at the ``X`` knot both return that same value,
  the two Hermite slopes of the interval are equal to its secant, and the
  quadratic and cubic Hermite coefficients vanish.  Practical consequence:
  extended+pchip and extended+histogram differ only on ``[24, 25.5)``, not
  above it.  This is measured, not assumed
  (``tests/test_population.py::test_extension_conserves_mass_including_the_synthetic_bin``),
  and it is stated as a derived observation so that a future extension law
  with a non-flat continuation does not silently break a promise this module
  never made.
* **Gate-eligible.**  :meth:`sample` needs no special case: the synthetic bin is
  drawn like any other bin, with probability proportional to its count, and the
  within-bin draw is the same uniform (histogram) or inverse-cubic (pchip) as
  everywhere else.  This is the substantive difference from the D-14 clamp,
  which has no generative counterpart at all.
* **:attr:`total_count` grows, and the growth is real, normalizable mass.**
  The NEO component's top H bin holds 55.8% of its mass (DESIGN §2.6), so at
  the canonical ``h_extend_to = 30`` (factor ``4.5/1.5 = 3``) the NEO
  component's total goes from 277 469 to 742 145 objects — **×2.675** — while
  the "other" component (1.4% in its top bin) grows only ×1.042.  The mixture
  fraction therefore moves a great deal; that is not an artifact to be
  suppressed but the model statement being made, namely that the faint end
  does not stop existing at H = 25.5.  It is honest mass: unlike the clamp's,
  it integrates to a finite number, which is exactly why the extended pair can
  be handed to the DESIGN §4.1 gate.

``h_extend_to`` is mutually exclusive with ``h_top_bin_infinite`` (the two are
alternative answers to the same problem, and stacking them would mean
extending a constant to infinity from a boundary that was itself synthetic).
The clamp survives **only** as the digest2-emulation mode for WP7-style
same-inputs comparisons.
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "PopulationDensity",
    "Digest2Histogram",
    "read_digest2_model",
    "QPART",
    "EPART",
    "IPART_DEG",
    "HPART",
    "Q_EDGES",
    "E_EDGES",
    "I_EDGES_DEG",
    "I_EDGES",
    "H_EDGES",
    "D2_CLASSES",
    "D2_VARIANTS",
    "H_INTERPOLATIONS",
]


class PopulationDensity(ABC):
    """An absolutely calibrated population density n_C(elements, H).

    The load-bearing abstraction of pneo (DESIGN §3): the classifier is
    generic over implementations of this interface.
    """

    #: Element parameterization the density is expressed in: "qei" or "aei".
    #: The classifier applies the matching reparameterization Jacobian.
    parameterization: str

    #: Class label: "NEO" or "other".
    label: str

    @abstractmethod
    def density(self, elements3: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Number density at (elements3, H).

        Parameters
        ----------
        elements3 : (..., 3) array in this density's parameterization —
            (q, e, i) or (a, e, i) [AU, -, rad]. Angular elements
            (Omega, omega, M) are uniform by convention in v0; the returned
            density is per unit (elements3, H) volume with the uniform-angle
            factor 1/(2 pi)^3 ALREADY INCLUDED, i.e. it is the full 7D density
            n(elements3, Omega, omega, M, H) which happens not to depend on
            the angles. Units: objects per unit
            q [AU] x e [-] x i [rad] x Omega, omega, M [rad^3] x H [mag],
            i.e. objects / (AU * rad^4 * mag).
            (WP0 contract amendment, authorized by the supervisor after the
            WP4 review: the original wording read "AU * rad * mag", which
            omitted the three angular dimensions carried by the included
            1/(2 pi)^3 factor.)
        H : (...,) absolute magnitudes.

        Returns
        -------
        (...,) densities; exactly 0.0 outside the declared support [D-4].
        """

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw n objects from the density.

        Returns
        -------
        (elements6, H) : elements6 is (n, 6) full (a, e, i, Omega, omega, M)
        with the three angles drawn uniformly in [0, 2 pi) (i converted from
        the parameterization as needed); H is (n,). This is the §4.1
        test-population generator.
        """

    @property
    @abstractmethod
    def support(self) -> dict[str, tuple[float, float]]:
        """Declared support: {"q"|"a": (lo, hi), "e": ..., "i": ..., "H": (lo, hi)}."""


# ---------------------------------------------------------------------------
# digest2 model geometry (verbatim from d2model.c; these are UPPER bin edges)
# ---------------------------------------------------------------------------

#: Perihelion-distance partition [AU]; upper edges, QX = 29 bins.
QPART = np.array(
    [0.4, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
     1.67, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5,
     4.0, 4.5, 5.0, 5.5, 10.0, 20.0, 30.0, 40.0, 100.0]
)
#: Eccentricity partition; upper edges, EX = 8 bins.
EPART = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1])
#: Inclination partition [DEGREES]; upper edges, IX = 11 bins.
IPART_DEG = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0, 180.0])
#: Absolute-magnitude partition [mag]; upper edges, HX = 18 bins.
HPART = np.array(
    [6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
     18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.5]
)

# Full edge arrays: the lower edge of the first bin is 0 in every dimension
# (muk.c initialises q0 = e0 = i0 = h0 = 0 when computing bin volumes).
Q_EDGES = np.concatenate(([0.0], QPART))          # (30,) [AU]
E_EDGES = np.concatenate(([0.0], EPART))          # (9,)  [-]
I_EDGES_DEG = np.concatenate(([0.0], IPART_DEG))  # (12,) [deg]
I_EDGES = np.radians(I_EDGES_DEG)                 # (12,) [rad]
H_EDGES = np.concatenate(([0.0], HPART))          # (19,) [mag]

MODEL_SHAPE = (len(QPART), len(EPART), len(IPART_DEG), len(HPART))  # (29, 8, 11, 18)

#: digest2's 15 orbit classes, in file order (classAbbr[] in d2model.c).
#: NOTE: these OVERLAP; they are not a partition.  "SS" (the whole model) is
#: written first in the file and is not one of them.
D2_CLASSES = ("Int", "NEO", "N22", "N18", "MC", "Hun", "Pho", "MB1",
              "Pal", "Han", "MB2", "MB3", "Hil", "JTr", "JFC")

#: Model variants: "All" = total population, "Unk" = total - known [D-10].
D2_VARIANTS = ("All", "Unk")

_ANGLE_VOLUME = (2.0 * np.pi) ** 3
_TWO_PI = 2.0 * np.pi


def _muk_pseudo_volume() -> np.ndarray:
    """The (29, 8, 11, 18) pseudo-volume ``muk.c`` divides the counts' sqrt by.

    This is *not* a physical volume — the eccentricity factor is
    ``de / ((1-e0) + (1-e1))`` and the inclination extent is in degrees.  It
    exists only to invert the file's scaling exactly; see the module
    docstring.  Reproduces ``muk.c`` / ``population.py::_volume_scale``
    operation for operation.
    """
    q0 = np.concatenate(([0.0], QPART[:-1]))
    dq = QPART - q0

    e0 = np.concatenate(([0.0], EPART[:-1]))
    d0 = np.where(e0 < 1.0, 1.0 - e0, 0.0)   # "d1" of the previous iteration
    d1 = np.where(EPART < 1.0, 1.0 - EPART, 0.0)
    fe = (EPART - e0) / (d0 + d1)

    i0 = np.concatenate(([0.0], IPART_DEG[:-1]))
    di = IPART_DEG - i0                       # degrees, as in muk.c

    h0 = np.concatenate(([0.0], HPART[:-1]))
    dh = HPART - h0

    return (dq[:, None, None, None] * fe[None, :, None, None]
            * di[None, None, :, None] * dh[None, None, None, :])


_MUK_PSEUDO_VOLUME = _muk_pseudo_volume()

#: True element-space bin volume dq * de * di[rad] * dH (no angle factor).
BIN_VOLUME = (
    np.diff(Q_EDGES)[:, None, None, None]
    * np.diff(E_EDGES)[None, :, None, None]
    * np.diff(I_EDGES)[None, None, :, None]
    * np.diff(H_EDGES)[None, None, None, :]
)

#: (29, 8, 11) cell volume dq * de * di[rad] — BIN_VOLUME with the dH factor
#: taken out.  The smooth-in-H mode divides by this instead, because there the
#: dH is supplied by differentiating the cumulative count (module docstring).
QEI_VOLUME = (
    np.diff(Q_EDGES)[:, None, None]
    * np.diff(E_EDGES)[None, :, None]
    * np.diff(I_EDGES)[None, None, :]
)

for _arr in (QPART, EPART, IPART_DEG, HPART, Q_EDGES, E_EDGES, I_EDGES_DEG,
             I_EDGES, H_EDGES, _MUK_PSEUDO_VOLUME, BIN_VOLUME, QEI_VOLUME):
    _arr.flags.writeable = False   # single source of truth; do not mutate
del _arr

_EXPECTED_HEADER = ["Model", "Class", "Q", "e", "i"] + [
    "H" + f"{h:g}" for h in HPART
]


def _edge_lookup(part: np.ndarray) -> dict[float, int]:
    return {round(float(v), 9): k for k, v in enumerate(part)}


_Q_LOOKUP = _edge_lookup(QPART)
_E_LOOKUP = _edge_lookup(EPART)
_I_LOOKUP = _edge_lookup(IPART_DEG)

_CACHE: dict[tuple, dict[tuple[str, str], np.ndarray]] = {}

#: Every table name the model file carries.
D2_TABLES = ("SS",) + D2_CLASSES


def _check_variant(variant: str) -> str:
    if variant not in D2_VARIANTS:
        raise ValueError(
            f"unknown model variant {variant!r}; valid variants are "
            f"{', '.join(D2_VARIANTS)}"
        )
    return variant


def _get_table(tables, variant: str, name: str, path) -> np.ndarray:
    """Fetch one (variant, class) table with a message a human can act on."""
    _check_variant(variant)
    if name not in D2_TABLES:
        raise ValueError(
            f"unknown digest2 class {name!r}; valid classes are "
            f"{', '.join(D2_TABLES)}"
        )
    try:
        return tables[(variant, name)]
    except KeyError:
        raise ValueError(
            f"{path}: model file has no table for class {name!r}, "
            f"variant {variant!r}"
        ) from None


def read_digest2_model(
    path: str | os.PathLike,
    *,
    integer_counts: bool = True,
    use_cache: bool = True,
) -> dict[tuple[str, str], np.ndarray]:
    """Parse ``digest2.model.csv`` into absolute object counts.

    Parameters
    ----------
    path : path to the MPC ``digest2.model.csv``.
    integer_counts : if True (default) verify that every recovered count is an
        integer to within 1e-6 and round it.  The stored cells are
        ``count / sqrt(V_muk)`` (module docstring); recovering exact integers
        is the load-time proof that the inversion is right.  Set False to
        keep the raw floating-point inversion (e.g. for a hypothetical future
        model file carrying fractional weights).
    use_cache : reuse a previously parsed result for the same file
        (keyed on path, size and mtime).

    Returns
    -------
    dict mapping ``(variant, class)`` -> (29, 8, 11, 18) float array of
    **object counts** in the (q, e, i, H) bins, with ``variant`` in
    ``D2_VARIANTS`` and ``class`` in ``("SS",) + D2_CLASSES``.

    The returned dict is a fresh shallow copy on every call (so rebinding
    entries cannot poison the cache) and the arrays it holds are read-only.
    """
    path = os.fspath(path)
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, bool(integer_counts))
    if use_cache and key in _CACHE:
        return dict(_CACHE[key])

    nq, ne, ni, nh = MODEL_SHAPE
    tables: dict[tuple[str, str], np.ndarray] = {}
    filled: dict[tuple[str, str], np.ndarray] = {}

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty file") from None
        if [h.strip() for h in header] != _EXPECTED_HEADER:
            raise ValueError(
                f"{path}: unexpected header.\n  got      {header}\n"
                f"  expected {_EXPECTED_HEADER}"
            )

        for lineno, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != 5 + nh:
                raise ValueError(
                    f"{path}:{lineno}: expected {5 + nh} fields, got {len(row)}"
                )
            variant, cls = row[0].strip(), row[1].strip()
            try:
                iq = _Q_LOOKUP[round(float(row[2]), 9)]
                ie = _E_LOOKUP[round(float(row[3]), 9)]
                ii = _I_LOOKUP[round(float(row[4]), 9)]
            except (KeyError, ValueError):
                raise ValueError(
                    f"{path}:{lineno}: (Q, e, i) = "
                    f"({row[2]!r}, {row[3]!r}, {row[4]!r}) is not a model bin"
                ) from None

            k = (variant, cls)
            arr = tables.get(k)
            if arr is None:
                arr = tables[k] = np.zeros(MODEL_SHAPE)
                filled[k] = np.zeros((nq, ne, ni), dtype=bool)
            if filled[k][iq, ie, ii]:
                raise ValueError(f"{path}:{lineno}: duplicate row for {k} bin")
            filled[k][iq, ie, ii] = True

            for ih, field in enumerate(row[5:]):
                field = field.strip()
                arr[iq, ie, ii, ih] = float(field) if field else 0.0

    if not tables:
        raise ValueError(f"{path}: no data rows")
    for k, mask in filled.items():
        if not mask.all():
            raise ValueError(
                f"{path}: {k} is missing {int((~mask).sum())} of "
                f"{mask.size} (q, e, i) rows"
            )

    # Undo muk.c's 1/sqrt(pseudo-volume) scaling to get absolute counts.
    scale = np.sqrt(_MUK_PSEUDO_VOLUME)
    for k, arr in tables.items():
        counts = arr * scale
        if integer_counts:
            rounded = np.round(counts)
            dev = np.abs(counts - rounded).max() if counts.size else 0.0
            if dev > 1e-6:
                raise ValueError(
                    f"{path}: un-scaled cells of {k} are not integer object "
                    f"counts (max deviation {dev:.3g}); the file's volume "
                    "scaling convention may have changed — see the "
                    "population.py module docstring."
                )
            counts = rounded
        if counts.min() < 0.0:
            raise ValueError(f"{path}: negative counts in {k}")
        counts.flags.writeable = False   # cached; callers must not mutate
        tables[k] = counts

    if use_cache:
        _CACHE[key] = tables
        return dict(tables)
    return tables


#: Valid values of ``Digest2Histogram(h_interpolation=...)``.
#:
#: ``"none"``   piecewise-constant in H (the D-10 default; digest2's own rule).
#: ``"pchip"``  monotone-cubic reconstruction of the cumulative count in H,
#:              evaluated as its derivative — mass-conserving per bin,
#:              non-negative, C⁰ across interior H edges (module docstring).
H_INTERPOLATIONS = ("none", "pchip")


# ---------------------------------------------------------------------------
# Monotone piecewise-cubic (PCHIP) machinery
#
# Implemented here rather than imported: scipy is not a pneo dependency
# [D-9].  This is the standard Fritsch–Carlson / Fritsch–Butland construction
# (Fritsch & Carlson 1980, SIAM J. Numer. Anal. 17, 238; Fritsch & Butland
# 1984, SIAM J. Sci. Stat. Comput. 5, 300), which is also what
# ``scipy.interpolate.PchipInterpolator`` implements — the unit tests check
# agreement against a hand-transcribed reference on a small case, and, when
# scipy happens to be importable, against scipy itself.
# ---------------------------------------------------------------------------


def _pchip_end_slope(h0: float, h1: float, d0, d1):
    """One-sided three-point end slope with the monotonicity limiter.

    ``h0``/``h1`` are the two intervals adjacent to the end (``h0`` the
    outermost), ``d0``/``d1`` the corresponding secant slopes.  The raw
    estimate is the quadratic-through-three-points slope; it is then set to 0
    if it disagrees in sign with the adjacent secant, and clipped to ``3 d0``
    if the secants bracket an extremum and the estimate exceeds the
    Fritsch–Carlson bound.  Both guards are what keeps the interpolant
    monotone all the way to the boundary.
    """
    d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    wrong_sign = np.sign(d) != np.sign(d0)
    over = (np.sign(d0) != np.sign(d1)) & (np.abs(d) > 3.0 * np.abs(d0))
    out = np.where(wrong_sign, 0.0, d)
    return np.where(~wrong_sign & over, 3.0 * d0, out)


def _pchip_slopes(x: np.ndarray, y: np.ndarray):
    """Monotone Hermite slopes at the knots of ``y`` (last axis) over ``x``.

    Returns ``(m, delta)`` where ``m`` has the shape of ``y`` and ``delta`` the
    secant slopes (one shorter).  Interior slopes are the weighted harmonic
    mean of the two adjacent secants, forced to 0 wherever the secants change
    sign or either vanishes; that bound (``m_k <= 3 min(|delta_{k-1}|,
    |delta_k|)``) is exactly the Fritsch–Carlson sufficient condition for the
    Hermite cubic to be monotone on every interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be 1-D with at least 3 knots")
    if y.shape[-1] != x.size:
        raise ValueError(
            f"y's last axis ({y.shape[-1]}) must match len(x) ({x.size})"
        )
    h = np.diff(x)
    delta = np.diff(y, axis=-1) / h

    m = np.zeros(y.shape, dtype=float)
    d0, d1 = delta[..., :-1], delta[..., 1:]
    h0, h1 = h[:-1], h[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    # "The two secants agree in sign and neither vanishes", tested on the signs
    # rather than on ``d0 * d1 > 0``.  The product form is what scipy uses and
    # is equivalent on normal inputs, but it *underflows*: two same-sign
    # secants of ~1e-200 multiply to 0, which would silently force the slope to
    # zero and flatten a legitimately rising segment.  It also overflows for
    # secants above ~1e154.  Neither occurs in the digest2 model (integer
    # counts over 18 bins), so this changes nothing there; it is here so the
    # helper is correct for any monotone data it is handed.
    same = (np.sign(d0) == np.sign(d1)) & (d0 != 0.0)
    # Evaluate the harmonic mean only where it is defined; the dummy 1.0
    # elsewhere keeps the expression finite and is discarded by the ``where``.
    # errstate rather than a mask: with extreme (but finite) secants the
    # reciprocals can overflow to inf in entries ``same`` already excludes, and
    # a RuntimeWarning from a value that is about to be thrown away is noise.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        whmean = (
            w1 / np.where(same, d0, 1.0) + w2 / np.where(same, d1, 1.0)
        ) / (w1 + w2)
        m[..., 1:-1] = np.where(same, 1.0 / whmean, 0.0)
    m[..., 0] = _pchip_end_slope(h[0], h[1], delta[..., 0], delta[..., 1])
    m[..., -1] = _pchip_end_slope(h[-1], h[-2], delta[..., -1], delta[..., -2])
    return m, delta


def _pchip_derivative_coeffs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-interval coefficients of the PCHIP *derivative*, Horner-ready.

    Returns ``(..., n_intervals, 3)`` = ``(a, b, c)`` such that on interval
    ``k`` the interpolant's derivative at ``H`` is

        ``a + s (b + s c)``,   ``s = H - x[k]``

    i.e. the quadratic ``C'(s)`` of the cubic Hermite segment.  The Horner
    grouping is fixed here on purpose: the Rust kernel evaluates the *same*
    expression on the *same* coefficients, which is what makes the two
    backends bit-identical (DESIGN §3, D-15).
    """
    m, delta = _pchip_slopes(x, y)
    h = np.diff(np.asarray(x, dtype=float))
    m0, m1 = m[..., :-1], m[..., 1:]
    # Cubic Hermite in the un-normalized local variable s = H - x[k]:
    #   C(s) = y_k + m0 s + c2 s^2 + c3 s^3
    c2 = (3.0 * delta - 2.0 * m0 - m1) / h
    c3 = (m0 + m1 - 2.0 * delta) / (h * h)
    return np.stack((m0, 2.0 * c2, 3.0 * c3), axis=-1)


#: Width of the model's top H bin, [24, 25.5) — the per-unit-H reference the
#: finite extension continues.  Named rather than spelled 1.5 inline so that a
#: future model file with different H partitions cannot leave a stale literal.
_TOP_H_BIN_WIDTH = float(H_EDGES[-1] - H_EDGES[-2])


#: Largest accepted ``h_extend_to``.  Not a physical limit — a sanity ceiling.
#:
#: The synthetic bin's count scales as ``(h_extend_to - 25.5) / 1.5``, so a
#: careless value (a mistyped exponent, a value in the wrong unit) silently
#: multiplies the NEO component's mass by that factor and, far enough out,
#: overflows to ``inf`` and makes the component unusable in a way whose error
#: message would point nowhere near the cause.  100 mag is ~70x the model's own
#: H range and far beyond any survey's reach (implied H at rho_min = 0.01 AU
#: and V = 24 is ~34), so nothing legitimate is excluded; the finiteness of the
#: extended counts is *also* checked directly in :func:`_extend_h_axis`, so the
#: ceiling is a better error message rather than the guarantee itself.
H_EXTEND_TO_MAX = 100.0


def _validated_h_extend_to(h_extend_to) -> float | None:
    """Normalize and validate the ``h_extend_to`` knob; ``None`` passes through.

    Must be a real number, finite, strictly greater than the model's H support
    end (25.5), and at most :data:`H_EXTEND_TO_MAX`: a non-finite value would
    be the D-14 clamp by another name (and non-normalizable), a value at or
    below 25.5 would either be a no-op or silently *shrink* the support, and an
    absurdly large one inflates the component's mass by the width ratio (see
    :data:`H_EXTEND_TO_MAX`).
    """
    if h_extend_to is None:
        return None
    # A str would otherwise sail through `float()` — `h_extend_to="30"` and,
    # worse, `h_extend_to="none"` (a plausible copy/paste from
    # `h_interpolation`) which raises a bare ValueError from float() naming
    # neither the parameter nor what it wanted.  Booleans are ints in Python
    # and `h_extend_to=True` is certainly a mistake, so they go too.
    if isinstance(h_extend_to, bool) or not isinstance(
        h_extend_to, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"h_extend_to must be a real number or None, got "
            f"{type(h_extend_to).__name__} {h_extend_to!r}. It is the upper "
            "edge of the synthetic H bin [25.5, h_extend_to) in magnitudes "
            "(canonical value 30.0); pass None to keep the D-4 support."
        )
    x = float(h_extend_to)
    top = float(H_EDGES[-1])
    if not np.isfinite(x):
        raise ValueError(
            f"h_extend_to must be finite, got {h_extend_to!r}. An infinite "
            "extension is the D-14 clamp (h_top_bin_infinite=True), which is "
            "non-normalizable and gate-ineligible; the point of h_extend_to is "
            "a finite, samplable support."
        )
    if not x > top:
        raise ValueError(
            f"h_extend_to must be greater than {top:g} (the model's H support "
            f"end), got {x!r}. h_extend_to appends a synthetic H bin "
            f"[{top:g}, h_extend_to); a value at or below {top:g} would be "
            "empty or would shrink the support."
        )
    if x > H_EXTEND_TO_MAX:
        raise ValueError(
            f"h_extend_to = {x!r} exceeds the sanity ceiling "
            f"{H_EXTEND_TO_MAX:g} mag. The synthetic bin's count scales as "
            f"(h_extend_to - {top:g}) / 1.5, so this would multiply the NEO "
            f"component's faint-end mass by {(x - top) / 1.5:.3g}x; at "
            "V = 24 and rho_min = 0.01 AU the implied H only reaches ~34, so "
            "a value this large is almost certainly a typo or a unit error. "
            "See population.H_EXTEND_TO_MAX if you really mean it."
        )
    return x


def _extend_h_axis(
    counts: np.ndarray, h_extend_to: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append the synthetic 19th H bin ``[25.5, h_extend_to)`` [D-19].

    Returns ``(counts_ext, h_edges_ext, bin_volume_ext)`` with H axis length
    ``n_h + 1``.  The synthetic bin's count is the top bin's count scaled by
    the width ratio, i.e. a flat continuation of the top bin's per-unit-H
    object density (module docstring).

    Both returned axis arrays are built by *concatenation onto the unextended
    arrays*, which is what makes everything below 25.5 bit-identical to the
    unextended model rather than merely equal: the first ``n_h`` volume slabs
    are copied, not recomputed.
    """
    width = float(h_extend_to) - float(H_EDGES[-1])
    counts_ext = np.concatenate(
        (counts, counts[..., -1:] * (width / _TOP_H_BIN_WIDTH)), axis=-1
    )
    # The ceiling on h_extend_to makes this unreachable for the shipped model,
    # but the input counts are caller-supplied: a component built from cells
    # near the float64 ceiling would overflow the width scaling and produce an
    # `inf` density that the Rust glue would reject much later with a message
    # about "finite and non-negative" pointing at the table rather than at the
    # extension.  Checked here, where the cause is nameable.
    if not np.all(np.isfinite(counts_ext[..., -1])):
        bad = int(np.count_nonzero(~np.isfinite(counts_ext[..., -1])))
        raise ValueError(
            f"h_extend_to = {h_extend_to!r} overflows the synthetic bin's "
            f"counts in {bad} cell(s): the top bin's count times "
            f"{width / _TOP_H_BIN_WIDTH:g} is not finite. Either the input "
            "counts are already near the float64 limit or h_extend_to is far "
            "too large."
        )
    h_edges_ext = np.concatenate((H_EDGES, [float(h_extend_to)]))
    bin_volume_ext = np.concatenate(
        (BIN_VOLUME, (QEI_VOLUME * width)[..., None]), axis=-1
    )
    h_edges_ext.flags.writeable = False
    bin_volume_ext.flags.writeable = False
    return counts_ext, h_edges_ext, bin_volume_ext


def _bin_index(edges: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open [lo, hi) bin lookup matching digest2's ``x >= part[b]`` rule.

    Returns (index clipped into range, in-support mask).  NaN maps to
    out-of-support.
    """
    nbins = len(edges) - 1
    idx = np.asarray(np.searchsorted(edges, x, side="right")) - 1
    ok = (idx >= 0) & (idx < nbins)
    return np.clip(idx, 0, nbins - 1), ok


class Digest2Histogram(PopulationDensity):
    """The MPC digest2 (q, e, i, H) binned population model (D-10).

    Density over the digest2 bin structure; uniform angles; absolute
    calibration inherited from the file.  Piecewise constant in q, e and i
    always; in H either piecewise constant (``h_interpolation="none"``, the
    default and digest2's own rule) or a mass-conserving monotone-cubic
    reconstruction (``h_interpolation="pchip"``).

    Attributes
    ----------
    counts : (29, 8, 11, 18) array of absolute object counts per bin.
    label : "NEO" or "other".
    variant : "All" or "Unk".
    classes : human-readable description of the class aggregation used.
    h_top_bin_infinite : whether the top H bin is treated as [24, inf)
        (digest2-emulation mode) rather than [24, 25.5) (D-4 default).
    h_interpolation : ``"none"`` or ``"pchip"`` — the H reconstruction.
    h_extend_to : ``None``, or the upper edge of a synthetic 19th H bin
        [25.5, h_extend_to) appended to every cell [D-19].
    shape : the effective bin shape — ``MODEL_SHAPE`` normally,
        ``(29, 8, 11, 19)`` in extended mode.  ``counts``, ``bin_volume`` and
        ``h_edges`` all carry the extension.

    Finite H extension (``h_extend_to``) [D-19]
    -------------------------------------------
    ``h_extend_to=30.0`` (the canonical value) appends one synthetic H bin
    ``[25.5, 30.0)`` per (q, e, i) cell, holding
    ``counts[..., 17] / 1.5 * (30.0 - 25.5)`` objects — a flat continuation of
    the top bin's per-unit-H density, with hard zero above ``h_extend_to``.
    Unlike the D-14 clamp this is *finite* mass, so the component remains
    normalizable and samplable, which makes the extended pair **gate-eligible**
    and composable with ``h_interpolation="pchip"``.  The module docstring
    derives the guarantees (exact continuity at 25.5, sub-25.5 bit-identity and
    its one pchip-mode qualifier, mass conservation of the synthetic bin under
    pchip, and the size of the resulting ``total_count`` growth: x2.675 on the
    NEO side, x1.042 on the "other" side).  Mutually exclusive with
    ``h_top_bin_infinite``.

    Smooth-in-H mode (``h_interpolation="pchip"``)
    ---------------------------------------------
    Per (q, e, i) cell the cumulative object count over the 19 H edges is
    interpolated by a monotone piecewise cubic (PCHIP) and the density is that
    cubic's derivative, divided by ``dq · de · di[rad]`` and ``(2 π)^3``.  The
    module docstring derives the three properties this buys — exact per-H-bin
    mass conservation, no mass leaking between bins (monotonicity, which is
    the answer to the WP4-review "interpolation invents density" objection),
    and non-negativity — and states the two things it does *not* change: D-4
    truncation outside ``[0, 25.5)``, and the ``1/(e sin i)`` integrable
    singularity of DESIGN §2.5, which lives on the e/i edges.

    The reconstruction is continuous across every interior H edge and
    generally *discontinuous* at H = 0 and H = 25.5, where the support ends.
    :meth:`sample` draws H by inverting the same cubic, so the generator and
    the density agree; the q/e/i lookup, :attr:`support`, :attr:`total_count`
    and :meth:`bin_indices` are all unaffected.

    digest2-emulation mode (``h_top_bin_infinite=True``)
    ----------------------------------------------------
    Reinstates ``d2model.c:hToBin``'s clamp of H >= 24 into the final bin, so
    that H >= 25.5 evaluates to the top bin's density (per unit H, unchanged)
    instead of 0.  This exists to make the DESIGN §4.2 head-to-head a
    controlled experiment: with the clamp on, digest2 and pneo consume the
    *same* effective model on every tracklet, so the residual gap is method
    alone.  See the module docstring for exactly which half of digest2's clamp
    is emulated (the H >= 25.5 half) and which is not (H < 0).

    .. warning::
       **A clamped component is gate-ineligible.**  The clamped density is
       constant in H above 25.5 and therefore has infinite integral: it is not
       a normalizable prior, it has no finite total count, and no population
       can be drawn from it.  The DESIGN §4.1 self-consistency logic — score a
       catalogue drawn from the same density that the classifier assumes — has
       no generative counterpart here, so :meth:`sample` raises and clamped
       components must never be fed to the WP6 gate harness.  Use them only
       for method comparison against digest2.
    """

    parameterization = "qei"

    def __init__(
        self,
        counts: np.ndarray,
        label: str,
        *,
        variant: str = "All",
        classes: str = "SS",
        interpolate: bool = False,
        h_top_bin_infinite: bool = False,
        h_interpolation: str = "none",
        h_extend_to: float | None = None,
    ):
        """Build a histogram density from absolute per-bin object counts.

        Parameters
        ----------
        counts : (29, 8, 11, 18) object counts on the digest2 (q, e, i, H)
            bins — *counts*, not the raw CSV cells (see module docstring);
            use :func:`read_digest2_model` or :meth:`neo_other_pair` to get
            them.
        label : "NEO" or "other" (the PopulationDensity class label).
        variant : "All" or "Unk", carried for provenance.
        classes : provenance string describing the class aggregation.
        interpolate : must be False.  Full (q, e, i, H) interpolation is not
            implemented; the H-only reconstruction is ``h_interpolation``.
        h_top_bin_infinite : digest2-emulation mode.  False (default) is D-4:
            the top H bin is [24, 25.5) and the density is 0 above it.  True
            reinstates digest2's ``hToBin`` clamp, making it [24, inf).  The
            resulting density is non-normalizable and the component is
            **gate-ineligible** (:meth:`sample` raises); see the class
            docstring.  Mutually exclusive with ``h_interpolation``.
        h_interpolation : ``"none"`` (default) or ``"pchip"``.  ``"pchip"``
            replaces the piecewise-constant H dependence by the derivative of
            a monotone cubic through the cell's cumulative counts — same mass
            in every H bin, continuous across interior H edges, still exactly
            0 outside [0, 25.5).  See the class and module docstrings.
        h_extend_to : ``None`` (default) or a finite float > 25.5 [D-19].  When
            set, one synthetic H bin ``[25.5, h_extend_to)`` is appended to
            every (q, e, i) cell with a count that continues the top bin's
            per-unit-H density flatly; the density is exactly 0 at and above
            ``h_extend_to``.  The extension has finite mass, so the component
            stays normalizable: :meth:`sample` works, the mode is
            gate-eligible, and it composes with ``h_interpolation``.  Mutually
            exclusive with ``h_top_bin_infinite``.

        Raises
        ------
        ValueError
            for an unknown ``h_interpolation``; for
            ``h_interpolation != "none"`` together with
            ``h_top_bin_infinite=True`` (the D-14 clamp extends a *constant*
            beyond the support, which a smooth reconstruction has no analogue
            for; the combination would be neither a reconstruction of the
            model nor an emulation of digest2); for ``h_extend_to`` together
            with ``h_top_bin_infinite=True`` (two alternative answers to the
            same problem — the clamp survives only as the digest2-emulation
            mode); and for an ``h_extend_to`` that is non-finite or not
            greater than 25.5.
        """
        if interpolate:
            raise NotImplementedError(
                "Digest2Histogram(interpolate=True) — full (q, e, i, H) "
                "interpolation is not implemented: interpolating in q/e/i "
                "near empty cells invents density with nothing to pin it "
                "down. The implemented reconstruction is H-only and "
                "mass-conserving: pass h_interpolation='pchip'."
            )
        # ``None`` means "no interpolation", not the string "None": without
        # this the error below reads `unknown h_interpolation 'None'`, which
        # sends the reader looking for a typo in a value they never typed.
        if h_interpolation is None:
            h_interpolation = "none"
        h_interpolation = str(h_interpolation)
        if h_interpolation not in H_INTERPOLATIONS:
            raise ValueError(
                f"unknown h_interpolation {h_interpolation!r}; valid options "
                f"are {', '.join(repr(v) for v in H_INTERPOLATIONS)}"
            )
        if h_interpolation != "none" and h_top_bin_infinite:
            raise ValueError(
                "h_interpolation="
                f"{h_interpolation!r} cannot be combined with "
                "h_top_bin_infinite=True: the D-14 clamp is a "
                "histogram-emulation device (it extends the top bin's "
                "*constant* density to H = inf, which is how digest2's "
                "hToBin behaves), while a smooth-in-H reconstruction has no "
                "constant to extend — the result would be neither a "
                "reconstruction of the model nor an emulation of digest2, "
                "and it would be non-normalizable on top of that. Pick one."
            )
        # Validated before the clamp cross-check so that a bad *value* is
        # reported as a bad value rather than as a mode conflict.
        h_extend_to = _validated_h_extend_to(h_extend_to)
        if h_extend_to is not None and h_top_bin_infinite:
            raise ValueError(
                f"h_extend_to={h_extend_to!r} cannot be combined with "
                "h_top_bin_infinite=True: they are two alternative answers to "
                "the same problem (the D-4 H-support wall of DESIGN §2.6). The "
                "D-14 clamp extends the top bin's constant density to H = inf, "
                "which is non-normalizable and therefore gate-ineligible and "
                "unsamplable; h_extend_to appends a *finite* synthetic bin "
                "[25.5, h_extend_to) and keeps the component a probability "
                "distribution. Stacking them would mean clamping to infinity "
                "from a boundary that is itself synthetic. The clamp survives "
                "only as the digest2-emulation mode for the DESIGN §4.2 "
                "same-inputs comparison (D-14); for operational scoring use "
                "h_extend_to (D-19). Pick one."
            )
        counts = np.array(counts, dtype=float)   # copy: the object owns it
        if counts.shape != MODEL_SHAPE:
            raise ValueError(
                f"counts must have shape {MODEL_SHAPE}, got {counts.shape}"
            )
        if not np.all(np.isfinite(counts)):
            raise ValueError("counts must be finite")
        if counts.min() < 0.0:
            raise ValueError("counts must be non-negative")

        self.label = str(label)
        self.variant = str(variant)
        self.classes = str(classes)
        self.h_top_bin_infinite = bool(h_top_bin_infinite)
        self.h_interpolation = h_interpolation
        self.h_extend_to = h_extend_to

        self.q_edges = Q_EDGES
        self.e_edges = E_EDGES
        self.i_edges = I_EDGES          # radians
        self.i_edges_deg = I_EDGES_DEG
        self.qei_volume = QEI_VOLUME    # dq * de * di[rad]

        if h_extend_to is None:
            self.h_edges = H_EDGES
            self.bin_volume = BIN_VOLUME    # dq * de * di[rad] * dH
        else:
            counts, self.h_edges, self.bin_volume = _extend_h_axis(
                counts, h_extend_to
            )
        #: Effective bin shape: MODEL_SHAPE, or one more H bin when extended.
        self.shape = tuple(int(s) for s in counts.shape)
        self._counts = counts

        # Number density: counts per unit (q, e, i[rad], H) volume, further
        # divided by (2 pi)^3 so that the returned value is the full 7-D
        # density including the uniform angles (interface contract).
        self._density_table = counts / (self.bin_volume * _ANGLE_VOLUME)
        if h_extend_to is not None:
            # Exact continuity at H = 25.5 *by construction*: the synthetic
            # bin's density is the top bin's density bit-for-bit, not a value
            # that merely rounds to it.  (Recomputing it from the synthetic
            # count and volume agrees to <= 1 ulp — 3.5e-16 relative at worst
            # over the shipped model — but 296 of the 546 non-empty NEO cells
            # are the *only* ones where it comes out bit-equal, and a
            # continuity claim that holds for 54% of cells is not a claim.
            # The 1-ulp residual is thereby moved onto the synthetic bin's
            # count/density relation, where it is asserted, not hidden; see
            # tests/test_population.py.)
            self._density_table[..., -1] = self._density_table[..., -2]
        self._total_count = float(counts.sum())
        self._cdf: np.ndarray | None = None
        #: 1 / (dq de di (2 pi)^3), the divisor the smooth-in-H mode uses.
        self._qei_inv_volume = 1.0 / (QEI_VOLUME * _ANGLE_VOLUME)
        self._h_coeffs: np.ndarray | None = None
        if h_interpolation == "pchip":
            self._h_coeffs = self._build_h_pchip(counts, self.h_edges)

    @staticmethod
    def _build_h_pchip(counts: np.ndarray, h_edges: np.ndarray) -> np.ndarray:
        """(29, 8, 11, n_h, 3) Horner coefficients of the smooth H density.

        ``h_edges`` is the component's own H edge array — ``H_EDGES``, or the
        D-19 extended one, in which case the synthetic bin is simply one more
        cumulative knot and nothing else changes.  Per-bin mass conservation is
        structural (the interpolant is pinned to the exact cumulative count at
        every knot), so it covers the synthetic bin automatically; what the
        extra knot *does* change is the slope at the 25.5 knot, which is an end
        knot without the extension and an interior one with it — hence the
        cubic on [24, 25.5) differs between the two (its mass does not).

        The knot values are the *exact* cumulative counts: the shipped model's
        cells are integers below 2^53, so ``cumsum`` in float64 is exact and
        ``C(H_{k+1}) - C(H_k)`` recovers ``counts[k]`` bit-for-bit.  (For a
        hypothetical fractional-count model the recovery would be exact to
        ~1e-16 of the *cell total* instead, which is the only place the mass
        guarantee is weaker than exact.)

        The returned coefficients are already divided by
        ``dq de di (2 pi)^3``, so :meth:`density` is one Horner evaluation and
        nothing else — the same expression the Rust kernel runs on the same
        numbers, which is what makes the backends bit-identical.
        """
        cum = np.zeros(counts.shape[:3] + (counts.shape[3] + 1,), dtype=float)
        cum[..., 1:] = np.cumsum(counts, axis=-1)
        coeffs = _pchip_derivative_coeffs(h_edges, cum)
        inv_vol = 1.0 / (QEI_VOLUME * _ANGLE_VOLUME)
        coeffs = np.ascontiguousarray(coeffs * inv_vol[..., None, None])
        coeffs.flags.writeable = False
        return coeffs

    # -- factories ---------------------------------------------------------

    @classmethod
    def neo_other_pair(
        cls, path: str, variant: str = "All", *,
        h_top_bin_infinite: bool = False,
        h_interpolation: str = "none",
        h_extend_to: float | None = None,
    ) -> tuple["Digest2Histogram", "Digest2Histogram"]:
        """Load the model file and return (neo, other) components.

        neo = digest2's NEO orbit class(es); other = all remaining classes,
        aggregated. The two must be exactly complementary at q = 1.3 AU
        (WP4 acceptance). variant: "All" (total population) or "Unk"
        (undiscovered = total - known) [D-10].

        ``h_top_bin_infinite`` is passed to both components: True builds the
        digest2-emulation pair whose top H bin is [24, inf) — a comparison
        mode only, never a gate mode (see the class docstring).  Both sides
        must be clamped together; a mixed pair would compare a truncated NEO
        density against a clamped "other" density and the resulting posterior
        would be an artifact of the mismatch.

        ``h_interpolation`` is likewise passed to **both** components, for the
        same reason: P(NEO) is a ratio of the two components' marginal
        likelihoods, so reconstructing the faint end of one and not the other
        would put a modelling artifact straight into the posterior.  The two
        modes are mutually exclusive (see the constructor).

        ``h_extend_to`` [D-19] is passed to both components too, and for a
        third instance of the same reason: extending only the NEO side would
        add the entire faint-end continuation to the numerator of P(NEO) and
        nothing to the denominator.  :func:`pneo.backend.check_component_pair`
        enforces the pairing at every classify entry point.

        Implementation note: digest2's named classes overlap and do not cover
        the model (module docstring), so "all remaining classes" is realised
        as ``SS - NEO`` cell by cell.  ``SS`` is digest2's aggregate over the
        whole model and ``NEO`` is exactly its q < 1.3 AU part, so the
        subtraction is exact (residual identically zero) and the two
        components are exactly complementary at the q = 1.3 AU bin edge.
        """
        _check_variant(variant)
        tables = read_digest2_model(path)
        ss = _get_table(tables, variant, "SS", path)
        neo = _get_table(tables, variant, "NEO", path)

        other = ss - neo
        worst = float(other.min())
        if worst < -1e-6:
            raise ValueError(
                f"{path}: SS - NEO is negative (min {worst:g}) — the model "
                "file violates the assumption that NEO is a subset of SS"
            )
        other = np.clip(other, 0.0, None)

        return (
            cls(neo, "NEO", variant=variant, classes="NEO",
                h_top_bin_infinite=h_top_bin_infinite,
                h_interpolation=h_interpolation,
                h_extend_to=h_extend_to),
            cls(other, "other", variant=variant, classes="SS - NEO",
                h_top_bin_infinite=h_top_bin_infinite,
                h_interpolation=h_interpolation,
                h_extend_to=h_extend_to),
        )

    @classmethod
    def from_classes(
        cls,
        path: str,
        classes,
        label: str,
        variant: str = "All",
        *,
        subtract=(),
        h_top_bin_infinite: bool = False,
        h_interpolation: str = "none",
        h_extend_to: float | None = None,
    ) -> "Digest2Histogram":
        """Aggregate named digest2 classes into one component (diagnostics).

        ``classes`` and ``subtract`` are iterables of names from
        ``("SS",) + D2_CLASSES``.  Because the named classes overlap, summing
        several of them double-counts; this is a diagnostic entry point, not
        the way the NEO/other pair is built (use :meth:`neo_other_pair`).

        ``h_top_bin_infinite`` selects the digest2-emulation top H bin
        [24, inf) (comparison mode only; see the class docstring);
        ``h_interpolation`` selects the smooth-in-H reconstruction;
        ``h_extend_to`` appends the D-19 finite synthetic H bin.  The clamp is
        mutually exclusive with each of the other two.
        """
        _check_variant(variant)
        tables = read_digest2_model(path)
        classes = tuple(classes)
        subtract = tuple(subtract)
        if not classes:
            raise ValueError("classes must be non-empty")
        counts = np.zeros(MODEL_SHAPE)
        for name in classes:
            counts += _get_table(tables, variant, name, path)
        for name in subtract:
            counts -= _get_table(tables, variant, name, path)
        counts = np.clip(counts, 0.0, None)
        desc = " + ".join(classes)
        if subtract:
            desc += " - " + " - ".join(subtract)
        return cls(counts, label, variant=variant, classes=desc,
                   h_top_bin_infinite=h_top_bin_infinite,
                   h_interpolation=h_interpolation,
                   h_extend_to=h_extend_to)

    # -- interface ---------------------------------------------------------

    def _h_bin(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(H bin index, in-support mask) honouring ``h_top_bin_infinite``.

        Single source of the H lookup rule for :meth:`density` and
        :meth:`bin_indices`, so the two cannot disagree about which H values
        the component actually evaluates.
        """
        ih, ok_h = _bin_index(self.h_edges, h)
        if self.h_top_bin_infinite:
            # digest2's hToBin clamp, upper half only: H >= 25.5 -> top bin.
            # ``h >= 25.5`` is False for nan (so nan stays out of support) and
            # True for +inf (which is what digest2's loop does with it).  H < 0
            # is deliberately left out of support; see the module docstring.
            above = h >= self.h_edges[-1]
            ih = np.where(above, len(self.h_edges) - 2, ih)
            ok_h = ok_h | above
        return ih, ok_h

    def density(self, elements3: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Number density; see PopulationDensity.density.

        ``elements3`` is (..., 3) = (q [AU], e [-], i [rad]); ``H`` is (...,)
        [mag].  Units of the result: objects / (AU * rad^4 * mag), i.e. per
        unit q x e x i x Omega x omega x M x H with the 1/(2 pi)^3 uniform-
        angle factor included.  Exactly 0.0 outside the support [D-4].

        Piecewise constant in q, e, i always.  In H it is piecewise constant
        (``h_interpolation="none"``) or the derivative of the cell's monotone
        cubic cumulative-count interpolant (``"pchip"``).  The q/e/i lookup —
        which bins, which points are in support, and the exact bit pattern of
        the answer where the answer is 0 — is identical in the two modes; only
        the value inside the support differs.

        With ``h_top_bin_infinite=True`` the sole change is that H >= 25.5
        evaluates to the [24, 25.5) bin's density instead of 0 (digest2's
        ``hToBin`` clamp; ``H = +inf`` clamps too, ``H = nan`` does not).
        q, e, i truncation and H < 0 are bit-identical between the two modes,
        as is every H in [0, 25.5): the clamp only ever *adds* the H >= 25.5
        branch, it never touches the in-range lookup.

        With ``h_extend_to`` set [D-19] the support runs to ``h_extend_to``
        instead of 25.5 and the density there is the top bin's per-unit-H value
        continued flatly (histogram mode) or the monotone cubic through the
        extra cumulative knot (pchip mode).  At and above ``h_extend_to`` the
        result is exactly 0.0 — still D-4, at a boundary further out.  In
        histogram mode every H below 25.5 evaluates bit-identically to the
        unextended component, and H = 25.5 evaluates bit-identically to
        H = nextafter(25.5, 0); in pchip mode the bit-identity holds below
        H = 24 (see the module docstring for why [24, 25.5) moves and why its
        mass does not).
        """
        elements3 = np.asarray(elements3, dtype=float)
        if elements3.ndim == 0 or elements3.shape[-1] != 3:
            raise ValueError(
                f"elements3 must have trailing dimension 3, got shape "
                f"{elements3.shape}"
            )
        q, ecc, inc = (elements3[..., 0], elements3[..., 1], elements3[..., 2])
        q, ecc, inc, h = np.broadcast_arrays(
            q, ecc, inc, np.asarray(H, dtype=float)
        )

        iq, ok_q = _bin_index(self.q_edges, q)
        ie, ok_e = _bin_index(self.e_edges, ecc)
        ii, ok_i = _bin_index(self.i_edges, inc)
        ih, ok_h = self._h_bin(h)
        ok = ok_q & ok_e & ok_i & ok_h

        out = np.zeros(q.shape, dtype=float)
        if ok.any():
            if self._h_coeffs is None:
                out[ok] = self._density_table[iq[ok], ie[ok], ii[ok], ih[ok]]
            else:
                ihx = ih[ok]
                co = self._h_coeffs[iq[ok], ie[ok], ii[ok], ihx]
                # s in [0, dH): the H bin is half-open, so s never reaches the
                # interval width and the derivative is evaluated strictly
                # inside the segment it belongs to.
                s = h[ok] - self.h_edges[ihx]
                out[ok] = co[..., 0] + s * (co[..., 1] + s * co[..., 2])
        return out

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw n objects; see PopulationDensity.sample.

        Bins are drawn with probability proportional to their object count;
        (q, e, i, H) are uniform within the chosen bin (which is exactly the
        piecewise-constant density) and Omega, omega, M are uniform in
        [0, 2 pi).  The returned semi-major axis is a = q / (1 - e).

        With ``h_interpolation="pchip"`` everything above is unchanged except
        the within-bin H draw, which is the inverse CDF of the cell's monotone
        cubic on that bin (:meth:`_sample_h_pchip`) instead of a uniform.  The
        bin selection is untouched, because the reconstruction conserves each
        bin's mass exactly — that is what lets the two parts of the sampler be
        combined this way at all.  Elements handling (uniform angles, the
        a = q/(1-e) conversion, the unbound-draw caveat below) is identical.

        UNBOUND DRAWS — NOT NEGLIGIBLE FOR THE NEO COMPONENT.  The model's
        top eccentricity bin is [0.9, 1.1), which straddles e = 1, and half
        of a uniform-in-bin draw from it lands at e >= 1.  That bin holds
        10 628 of the 277 469 All/NEO objects and 10 501 of the 235 675
        Unk/NEO objects, so the expected unbound fraction is

            All/NEO   1.92 %      All/other   0.0006 %
            Unk/NEO   2.23 %      Unk/other   0.00004 %

        (measured on 2e6 draws: 1.92 % / 2.23 % / 0.0004 % / 0.0000 %).
        For those draws a = q / (1 - e) <= 0 (and a = -inf at e exactly 1).

        **Callers generating test populations (WP6) must filter the unbound
        draws (keep e < 1) and report the discarded fraction.**  Doing so does
        not break the §4.1 self-consistency argument: the classifier's
        admissible region admits only bound orbits (D-8), so the likelihood
        side truncates the population at e < 1 identically.  Prior and test
        population therefore still match, and calibration is preserved — but
        the discarded fraction must be reported, because it is ~2 % of the
        NEO component and only ~1e-5 of the "other" component, i.e. it
        changes the components' relative normalization by ~2 % if it is
        silently dropped from one side only.

        With ``h_extend_to`` set [D-19] there is **no special case at all**:
        the synthetic bin ``[25.5, h_extend_to)`` participates in the bin CDF
        like every other bin (its selection probability is its object count
        over the total, which now includes the extension mass) and the
        within-bin draw is the same uniform — or the same inverse-cubic under
        ``h_interpolation="pchip"`` — as everywhere else.  That the extension
        is samplable at all is the substantive difference from the D-14 clamp
        and is what makes extended components **gate-eligible**.  The expected
        fraction of draws landing above H = 25.5 is
        ``extension mass / total_count``, which for the shipped All/NEO
        component at ``h_extend_to=30`` is 3 x 0.5582 / 2.6747 = 62.6%.

        Raises NotImplementedError when ``h_top_bin_infinite=True``: a clamped
        component is not a probability distribution, so there is nothing to
        draw from (see the class docstring).
        """
        if self.h_top_bin_infinite:
            raise NotImplementedError(
                "cannot sample a Digest2Histogram built with "
                "h_top_bin_infinite=True: with the top H bin treated as "
                "[24, inf) the density is constant in H above 25.5, so its "
                "integral diverges and it is not a normalizable probability "
                "distribution — there is no population it describes. The "
                "clamp is a digest2-emulation mode for method comparison "
                "(DESIGN §4.2) and is gate-ineligible: the §4.1 "
                "self-consistency test has no generative counterpart for it. "
                "Build the component with the default "
                "h_top_bin_infinite=False to sample."
            )
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        if self._total_count <= 0.0:
            raise ValueError("cannot sample from an empty population")
        if n == 0:
            return np.zeros((0, 6)), np.zeros(0)

        if self._cdf is None:
            cdf = np.cumsum(self._counts.ravel())
            cdf /= cdf[-1]
            self._cdf = cdf
        cdf = self._cdf
        # side="right" -> first bin whose cumulative probability exceeds u,
        # which correctly skips zero-count bins.
        k = np.searchsorted(cdf, rng.random(n), side="right")
        np.clip(k, 0, cdf.size - 1, out=k)
        iq, ie, ii, ih = np.unravel_index(k, self.shape)

        def _uniform_in(edges, idx):
            """Uniform in [edges[idx], edges[idx+1]), strictly half-open.

            ``lo + u * (hi - lo)`` with u < 1 can still round up to exactly
            ``hi`` in floating point, which would place the draw in the next
            bin (or outside the support, for the top bin).  Clamp to the
            largest double below ``hi``.
            """
            lo = edges[idx]
            hi = edges[idx + 1]
            return np.minimum(lo + rng.random(n) * (hi - lo),
                              np.nextafter(hi, lo))

        q = _uniform_in(self.q_edges, iq)
        ecc = _uniform_in(self.e_edges, ie)
        inc = _uniform_in(self.i_edges, ii)
        if self._h_coeffs is None:
            h = _uniform_in(self.h_edges, ih)
        else:
            h = self._sample_h_pchip(iq, ie, ii, ih, rng)

        with np.errstate(divide="ignore", invalid="ignore"):
            a = q / (1.0 - ecc)

        elements6 = np.empty((n, 6), dtype=float)
        elements6[:, 0] = a
        elements6[:, 1] = ecc
        elements6[:, 2] = inc
        elements6[:, 3:6] = rng.random((n, 3)) * _TWO_PI
        return elements6, h

    def _sample_h_pchip(self, iq, ie, ii, ih, rng) -> np.ndarray:
        """Inverse-CDF draw of H inside the already-chosen (cell, H bin).

        The bin was drawn with probability proportional to its object count,
        exactly as in histogram mode; what changes is the *within-bin*
        distribution, which here is the reconstructed density rather than a
        uniform.  So: solve ``C(s) - C(0) = u [C(dH) - C(0)]`` for
        ``s in [0, dH)``, where ``C`` is the cell's monotone cubic — i.e.
        invert the exact antiderivative of what :meth:`density` evaluates, so
        the generator and the density cannot disagree.

        ``C`` is built from the *stored* Horner coefficients ``(a, b, c)`` of
        the derivative, as ``a s + (b/2) s^2 + (c/3) s^3``; the normalization
        uses that same cubic at ``s = dH`` rather than the bin count, so
        ``u < 1`` always has a root strictly inside the interval regardless of
        the last-bit difference between ``(c/3)`` and the ``c3`` it came from.
        (The bin *count* is what sets the bin's selection probability, so the
        joint density is the intended one to within that ~1e-16 relative.)

        Newton, safeguarded by bisection: the cubic is non-decreasing, so
        ``[0, dH]`` brackets the root at every step (``C(0) - target <= 0`` and
        ``C(dH) - target >= 0``), and each iteration replaces one bracket end
        with the new point, so the bracket never grows.  A Newton step that
        leaves the bracket, or lands where the derivative is zero (a plateau
        of the CDF — a genuinely flat piece of the reconstructed density), is
        replaced by the bisection midpoint; every fifth iteration bisects
        unconditionally.

        What that guarantees, precisely: 60 iterations of which 12 are forced
        bisections bound the bracket by ``dH * 2^-12`` unconditionally — a
        pathological Newton sequence that crawled along one bracket end every
        non-forced step would stop there.  The real convergence is Newton's,
        and it is measured rather than assumed: over 2x10^5 random
        (cell, bin, u) triples spanning every populated bin of the shipped
        model, ``|C(s) - target| / C(dH)`` comes out at 6e-16, i.e. the
        quantile is exact to double precision.  See
        ``tests/test_population.py::test_pchip_sample_inversion_is_converged``,
        which re-derives that residual and asserts it below 1e-13, and
        ``test_pchip_sample_reproduces_the_analytic_density``, which would
        catch a systematically mis-inverted CDF by chi-square.
        """
        co = self._h_coeffs[iq, ie, ii, ih]
        a, b, c = co[..., 0], co[..., 1], co[..., 2]
        lo_edge = self.h_edges[ih]
        width = (self.h_edges[ih + 1] - lo_edge).astype(float)

        # Antiderivative of the stored derivative polynomial, relative to s=0.
        b2 = 0.5 * b
        c3 = c / 3.0

        def cdf(s):
            return s * (a + s * (b2 + s * c3))

        total = cdf(width)
        live = total > 0.0
        # A zero-mass bin cannot be selected (the bin CDF search skips
        # zero-count bins), so `live` is all-True on the shipped model; the
        # guard keeps a hand-built degenerate component from looping on a
        # constant CDF.
        safe_total = np.where(live, total, 1.0)
        target = rng.random(ih.shape) * total

        lo = np.zeros(ih.shape, dtype=float)
        hi = width.copy()
        # Linear-in-the-bin first guess (exact when the cell is locally flat).
        s = width * (target / safe_total)
        for it in range(60):
            f = cdf(s) - target
            lo = np.where(f <= 0.0, s, lo)
            hi = np.where(f > 0.0, s, hi)
            mid = 0.5 * (lo + hi)
            deriv = a + s * (b + s * c)
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = s - np.where(deriv > 0.0, f / deriv, np.nan)
            take = (
                np.isfinite(cand) & (cand > lo) & (cand < hi) & (it % 5 != 4)
            )
            s = np.where(take, cand, mid)
        s = np.where(live, s, 0.0)

        # Half-open [lo_edge, hi_edge): never return the upper edge, which
        # would fall in the next bin (or out of support, for the top bin).
        return np.minimum(
            lo_edge + s, np.nextafter(lo_edge + width, lo_edge)
        )

    @property
    def support(self) -> dict[str, tuple[float, float]]:
        """Declared support (D-4): q [AU], e [-], i [rad], H [mag].

        In digest2-emulation mode (``h_top_bin_infinite=True``) the H upper
        bound is ``inf``, because that is where the density is actually
        non-zero.  Reporting 25.5 there would mislabel every downstream
        diagnostic that asks the component where its support ends — notably
        the DESIGN §2.6(b) "measure above H_max^model" flag, which would
        otherwise flag as truncation-exposed exactly the nodes the clamped
        component evaluates normally.

        In extended mode [D-19] the H upper bound is ``h_extend_to``, for
        exactly the same reason — and there it is a *finite* honest boundary:
        the density really is zero above it (hard truncation, D-4, just further
        out), so §2.6(b)'s flag keeps measuring what it is meant to measure,
        now against the extended support.  ``self.h_edges[-1]`` already is
        ``h_extend_to``, so no branch is needed.
        """
        h_hi = (
            float("inf") if self.h_top_bin_infinite else float(self.h_edges[-1])
        )
        return {
            "q": (float(self.q_edges[0]), float(self.q_edges[-1])),
            "e": (float(self.e_edges[0]), float(self.e_edges[-1])),
            "i": (float(self.i_edges[0]), float(self.i_edges[-1])),
            "H": (float(self.h_edges[0]), h_hi),
        }

    @property
    def total_count(self) -> float:
        """Total object count over the component's binned H range.

        Equals the integral of the density over its support and the uniform
        angles in the default (D-4) mode; WP4 acceptance: reproduces the
        file's per-class totals.  Unchanged by ``h_interpolation``, and still
        the integral there — that is the point of reconstructing the
        cumulative count rather than the density (module docstring).

        .. note::
           With ``h_extend_to`` set [D-19] this **includes the synthetic bin's
           mass**, and the increase is large: the NEO component keeps 55.8% of
           its mass in the top H bin (DESIGN §2.6), so at the canonical
           ``h_extend_to = 30`` its total goes 277 469 -> 742 145 (x2.675)
           while the "other" component goes x1.042.  That is deliberate and it
           is *real, normalizable* mass — the value is still exactly the
           integral of the density over the declared support, so the mixture
           fraction built from it is the extended model's own mixture fraction
           (contrast the D-14 clamp's warning below, where no such statement
           can be made).  Any comparison of extended and unextended totals is
           a comparison of two different population models, which is the
           point.

        .. warning::
           In digest2-emulation mode (``h_top_bin_infinite=True``) this is
           **not** the integral of the density.  The clamped density is
           constant in H above 25.5, so its integral diverges; the value
           returned here is still the finite [0, 25.5) bin sum, i.e. the count
           of the underlying histogram, unchanged by the clamp.  It is
           reported for provenance (and so that the two modes' calibration can
           be compared against the same number); it must not be read as this
           component's total population, and the mixture fraction built from
           it is not the clamped model's mixture fraction.
        """
        return self._total_count

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of the per-bin object counts, shape :attr:`shape`.

        ``(29, 8, 11, 18)`` normally; ``(29, 8, 11, 19)`` with ``h_extend_to``
        set, the last H slab being the synthetic bin's counts [D-19].  It is
        the counts of the model *as evaluated*, which is what every consumer
        (the sampler, the Rust glue's table fallback, the diagnostics) needs.
        """
        view = self._counts.view()
        view.flags.writeable = False
        return view

    @property
    def h_coeffs(self) -> np.ndarray | None:
        """(29, 8, 11, 18, 3) smooth-in-H Horner coefficients, or None.

        ``None`` in ``h_interpolation="none"`` mode.  Otherwise the density at
        ``H`` in bin ``ih`` of cell ``(iq, ie, ii)`` is exactly

            ``a + s * (b + s * c)``,  ``s = H - h_edges[ih]``

        with ``(a, b, c) = h_coeffs[iq, ie, ii, ih]``, already divided by
        ``dq de di (2 pi)^3``.  Exposed so the Rust backend can copy the same
        numbers and evaluate the same expression — the two backends are
        bit-identical because they share the coefficients, not because they
        each re-derive them (see ``pneo.backend``).  Read-only.
        """
        return self._h_coeffs

    def bin_indices(self, elements3: np.ndarray, H: np.ndarray):
        """(iq, ie, ii, ih, in_support) for the given (q, e, i[rad]) and H.

        Exposed for validation/diagnostics (WP6) so callers can histogram
        samples on the model's own bins without duplicating the lookup rule.

        .. warning::
           The four index arrays are **clamped into range** for every input,
           including out-of-support and NaN ones, so they always look like
           plausible bin numbers.  Callers MUST mask on the returned
           ``in_support`` flag before using them; histogramming without the
           mask silently piles out-of-support points onto the edge bins.
           :meth:`density` applies the mask for you and returns 0.0 there.

        ``in_support`` follows ``h_top_bin_infinite``: in digest2-emulation
        mode H >= 25.5 is *in* support and carries ``ih = 17``, matching what
        :meth:`density` evaluates there.  It likewise follows ``h_extend_to``
        [D-19]: with the extension on, H in [25.5, h_extend_to) is in support
        and carries ``ih = 18`` (the synthetic bin), and H >= h_extend_to is
        out — again matching :meth:`density` exactly, since both go through
        :meth:`_h_bin` on the component's own :attr:`h_edges`.
        """
        elements3 = np.asarray(elements3, dtype=float)
        if elements3.ndim == 0 or elements3.shape[-1] != 3:
            raise ValueError(
                f"elements3 must have trailing dimension 3, got shape "
                f"{elements3.shape}"
            )
        q, ecc, inc = (elements3[..., 0], elements3[..., 1], elements3[..., 2])
        q, ecc, inc, h = np.broadcast_arrays(
            q, ecc, inc, np.asarray(H, dtype=float)
        )
        iq, ok_q = _bin_index(self.q_edges, q)
        ie, ok_e = _bin_index(self.e_edges, ecc)
        ii, ok_i = _bin_index(self.i_edges, inc)
        ih, ok_h = self._h_bin(h)
        return iq, ie, ii, ih, ok_q & ok_e & ok_i & ok_h

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        clamp = (
            ", h_top_bin_infinite=True" if self.h_top_bin_infinite else ""
        )
        interp = (
            f", h_interpolation={self.h_interpolation!r}"
            if self.h_interpolation != "none" else ""
        )
        extend = (
            f", h_extend_to={self.h_extend_to!r}"
            if self.h_extend_to is not None else ""
        )
        return (
            f"Digest2Histogram(label={self.label!r}, variant={self.variant!r}, "
            f"classes={self.classes!r}, total_count={self.total_count:.0f}"
            f"{clamp}{interp}{extend})"
        )
