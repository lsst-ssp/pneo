"""WP4 — Population density interface and the digest2 histogram implementation.

Frozen contract (WP0). See DESIGN.md §3, IMPLEMENTATION.md WP4, D-10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
            the angles. Units: objects / (AU * rad * mag).
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


class Digest2Histogram(PopulationDensity):
    """The MPC digest2 (q, e, i, H) binned population model (D-10).

    Piecewise-constant density over the digest2 bin structure; uniform angles;
    absolute calibration inherited from the file. Interpolation is OFF and not
    implemented in v0 (raise NotImplementedError if requested).
    """

    parameterization = "qei"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("WP4")

    @classmethod
    def neo_other_pair(
        cls, path: str, variant: str = "All"
    ) -> tuple["Digest2Histogram", "Digest2Histogram"]:
        """Load the model file and return (neo, other) components.

        neo = digest2's NEO orbit class(es); other = all remaining classes,
        aggregated. The two must be exactly complementary at q = 1.3 AU
        (WP4 acceptance). variant: "All" (total population) or "Unk"
        (undiscovered = total - known) [D-10].
        """
        raise NotImplementedError("WP4")

    def density(self, elements3: np.ndarray, H: np.ndarray) -> np.ndarray:
        raise NotImplementedError("WP4")

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("WP4")

    @property
    def support(self) -> dict[str, tuple[float, float]]:
        raise NotImplementedError("WP4")

    @property
    def total_count(self) -> float:
        """Total number of objects in this component (integral of the density
        over its support and the uniform angles). WP4 acceptance: reproduces
        the file's per-class totals."""
        raise NotImplementedError("WP4")
