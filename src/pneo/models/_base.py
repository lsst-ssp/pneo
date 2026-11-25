"""The interface every population-model component implements.

A population model (:class:`pneo.models.PopulationModel`) pairs two
components, one for NEOs and one for everything else.  Each component is an
absolutely calibrated number density over orbital elements and absolute
magnitude; the classifier is generic over this interface, so a new model
(NEOMOD3, one inferred from Rubin discoveries, ...) only has to provide it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["PopulationComponent"]


class PopulationComponent(ABC):
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
            (An earlier version of this docstring read "AU * rad * mag",
            omitting the three angular dimensions carried by the included
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
