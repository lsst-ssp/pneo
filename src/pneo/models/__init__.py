"""Population models: the prior knowledge P(NEO) is computed from.

A :class:`PopulationModel` pairs two absolutely calibrated components, one for
NEOs and one for every other small body, each implementing
:class:`PopulationComponent`.  :func:`load_model` builds the model pneo ships
with; other models can be added as sibling modules of :mod:`pneo.models.digest2`
and selected with ``load_model(source=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._base import PopulationComponent

__all__ = ["PopulationModel", "PopulationComponent", "load_model", "SOURCES"]

#: Model sources :func:`load_model` knows how to build.
SOURCES = ("digest2",)


@dataclass(frozen=True)
class PopulationModel:
    """A population model: an NEO component and a non-NEO component.

    Attributes
    ----------
    neo, other : PopulationComponent
        The two components.  Their densities are absolute (objects per unit
        orbital-element and magnitude volume), so their ratio carries the
        relative abundance of the two classes and no separate prior is needed.
    source : str
        Which model this is, e.g. ``"digest2"``.
    description : str
        A short human-readable summary (file, variant and options).
    """

    neo: PopulationComponent
    other: PopulationComponent
    source: str
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PopulationModel({self.description or self.source})"


def load_model(
    path=None,
    *,
    source: str = "digest2",
    variant: str = "All",
    h_extend_to: float | None = 30.0,
    h_interpolation: str = "none",
) -> PopulationModel:
    """Load a population model.

    The defaults give the configuration used in the paper and by
    ``pneo classify``.

    Parameters
    ----------
    path : str or path-like, optional
        The model file.  Defaults to the per-user cache, where
        ``pneo fetch`` downloads it (see :mod:`pneo.paths`).
    source : str
        Which model to build; only ``"digest2"`` (the MPC's digest2
        population model) is available today.
    variant : {"All", "Unk"}
        ``"All"`` is the full modeled population; ``"Unk"`` removes objects
        already in the MPC catalog.
    h_extend_to : float or None
        The digest2 model ends at absolute magnitude H = 25.5; its faintest bin
        is extended to this magnitude so that very small, nearby objects can be
        scored.  ``None`` disables the extension.
    h_interpolation : {"none", "pchip"}
        How the faint end is reconstructed: ``"none"`` keeps the model's
        piecewise-constant bins; ``"pchip"`` uses a smooth, count-conserving
        interpolation.
    """
    if source != "digest2":
        raise ValueError(f"unknown model source {source!r}; known: {SOURCES}")
    from ..paths import default_model_path
    from .digest2 import Digest2Component

    if path is None:
        path = default_model_path()
    path = str(path)
    neo, other = Digest2Component.neo_other_pair(
        path, variant, h_extend_to=h_extend_to, h_interpolation=h_interpolation
    )
    desc = f"digest2 {variant}, {path}"
    if h_extend_to is not None:
        desc += f", H extended to {h_extend_to:g}"
    if h_interpolation != "none":
        desc += f", {h_interpolation} faint end"
    return PopulationModel(neo=neo, other=other, source="digest2", description=desc)
