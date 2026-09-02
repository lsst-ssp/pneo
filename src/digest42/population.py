"""WP4 — Population density interface and the digest2 histogram implementation.

Frozen contract (WP0). See DESIGN.md §3, IMPLEMENTATION.md WP4, D-10.

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
an object brighter than absolute magnitude 0 — at the geometries digest42
integrates over (rho >= 0.01 AU, V >= 14) that never occurs, so emulating it
would change nothing measurable while making the comparison mode a
two-sided deviation from D-4 instead of the one-sided one under study.
``h_top_bin_infinite=True`` therefore still returns 0 for H < 0.

Clamped components are a **comparison/emulation mode only, never a gate
mode** — see the class docstring for why (the clamped density is not
normalizable, so DESIGN §4.1 has no generative counterpart for it).
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
]


class PopulationDensity(ABC):
    """An absolutely calibrated population density n_C(elements, H).

    The load-bearing abstraction of digest42 (DESIGN §3): the classifier is
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

for _arr in (QPART, EPART, IPART_DEG, HPART, Q_EDGES, E_EDGES, I_EDGES_DEG,
             I_EDGES, H_EDGES, _MUK_PSEUDO_VOLUME, BIN_VOLUME):
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

    Piecewise-constant density over the digest2 bin structure; uniform angles;
    absolute calibration inherited from the file. Interpolation is OFF and not
    implemented in v0 (raise NotImplementedError if requested).

    Attributes
    ----------
    counts : (29, 8, 11, 18) array of absolute object counts per bin.
    label : "NEO" or "other".
    variant : "All" or "Unk".
    classes : human-readable description of the class aggregation used.
    h_top_bin_infinite : whether the top H bin is treated as [24, inf)
        (digest2-emulation mode) rather than [24, 25.5) (D-4 default).

    digest2-emulation mode (``h_top_bin_infinite=True``)
    ----------------------------------------------------
    Reinstates ``d2model.c:hToBin``'s clamp of H >= 24 into the final bin, so
    that H >= 25.5 evaluates to the top bin's density (per unit H, unchanged)
    instead of 0.  This exists to make the DESIGN §4.2 head-to-head a
    controlled experiment: with the clamp on, digest2 and digest42 consume the
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
        interpolate : must be False; interpolation is not implemented [D-10].
        h_top_bin_infinite : digest2-emulation mode.  False (default) is D-4:
            the top H bin is [24, 25.5) and the density is 0 above it.  True
            reinstates digest2's ``hToBin`` clamp, making it [24, inf).  The
            resulting density is non-normalizable and the component is
            **gate-ineligible** (:meth:`sample` raises); see the class
            docstring.
        """
        if interpolate:
            raise NotImplementedError(
                "Digest2Histogram interpolation is not implemented in v0 "
                "(D-10): interpolating near empty bins invents density. "
                "Evaluation is piecewise-constant."
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

        self._counts = counts
        self.label = str(label)
        self.variant = str(variant)
        self.classes = str(classes)
        self.h_top_bin_infinite = bool(h_top_bin_infinite)

        self.q_edges = Q_EDGES
        self.e_edges = E_EDGES
        self.i_edges = I_EDGES          # radians
        self.i_edges_deg = I_EDGES_DEG
        self.h_edges = H_EDGES
        self.bin_volume = BIN_VOLUME    # dq * de * di[rad] * dH

        # Number density: counts per unit (q, e, i[rad], H) volume, further
        # divided by (2 pi)^3 so that the returned value is the full 7-D
        # density including the uniform angles (interface contract).
        self._density_table = counts / (BIN_VOLUME * _ANGLE_VOLUME)
        self._total_count = float(counts.sum())
        self._cdf: np.ndarray | None = None

    # -- factories ---------------------------------------------------------

    @classmethod
    def neo_other_pair(
        cls, path: str, variant: str = "All", *,
        h_top_bin_infinite: bool = False,
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
                h_top_bin_infinite=h_top_bin_infinite),
            cls(other, "other", variant=variant, classes="SS - NEO",
                h_top_bin_infinite=h_top_bin_infinite),
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
    ) -> "Digest2Histogram":
        """Aggregate named digest2 classes into one component (diagnostics).

        ``classes`` and ``subtract`` are iterables of names from
        ``("SS",) + D2_CLASSES``.  Because the named classes overlap, summing
        several of them double-counts; this is a diagnostic entry point, not
        the way the NEO/other pair is built (use :meth:`neo_other_pair`).

        ``h_top_bin_infinite`` selects the digest2-emulation top H bin
        [24, inf) (comparison mode only; see the class docstring).
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
                   h_top_bin_infinite=h_top_bin_infinite)

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
        """Piecewise-constant number density; see PopulationDensity.density.

        ``elements3`` is (..., 3) = (q [AU], e [-], i [rad]); ``H`` is (...,)
        [mag].  Units of the result: objects / (AU * rad^4 * mag), i.e. per
        unit q x e x i x Omega x omega x M x H with the 1/(2 pi)^3 uniform-
        angle factor included.  Exactly 0.0 outside the support [D-4].

        With ``h_top_bin_infinite=True`` the sole change is that H >= 25.5
        evaluates to the [24, 25.5) bin's density instead of 0 (digest2's
        ``hToBin`` clamp; ``H = +inf`` clamps too, ``H = nan`` does not).
        q, e, i truncation and H < 0 are bit-identical between the two modes,
        as is every H in [0, 25.5): the clamp only ever *adds* the H >= 25.5
        branch, it never touches the in-range lookup.
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
            out[ok] = self._density_table[iq[ok], ie[ok], ii[ok], ih[ok]]
        return out

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw n objects; see PopulationDensity.sample.

        Bins are drawn with probability proportional to their object count;
        (q, e, i, H) are uniform within the chosen bin (which is exactly the
        piecewise-constant density) and Omega, omega, M are uniform in
        [0, 2 pi).  The returned semi-major axis is a = q / (1 - e).

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
        iq, ie, ii, ih = np.unravel_index(k, MODEL_SHAPE)

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
        h = _uniform_in(self.h_edges, ih)

        with np.errstate(divide="ignore", invalid="ignore"):
            a = q / (1.0 - ecc)

        elements6 = np.empty((n, 6), dtype=float)
        elements6[:, 0] = a
        elements6[:, 1] = ecc
        elements6[:, 2] = inc
        elements6[:, 3:6] = rng.random((n, 3)) * _TWO_PI
        return elements6, h

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
        """Total object count over the binned [0, 25.5) H range.

        Equals the integral of the density over its support and the uniform
        angles in the default (D-4) mode; WP4 acceptance: reproduces the
        file's per-class totals.

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
        """Read-only view of the (29, 8, 11, 18) per-bin object counts."""
        view = self._counts.view()
        view.flags.writeable = False
        return view

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
        :meth:`density` evaluates there.
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
        return (
            f"Digest2Histogram(label={self.label!r}, variant={self.variant!r}, "
            f"classes={self.classes!r}, total_count={self.total_count:.0f}"
            f"{clamp})"
        )
