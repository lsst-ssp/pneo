"""WP5 — Marginal likelihoods, posterior, and diagnostics.

Frozen contract (WP0). See docs/design/DESIGN.md §2.3–2.5, docs/design/IMPLEMENTATION.md WP5.

What this module does
---------------------
`classify` is the orchestration layer: it turns an :class:`Attributable`
(plus its apparent magnitude) into the calibrated posterior P(NEO | A, V) by
evaluating, for each population component C,

    Lambda_C(A, V) = int_AR n_C(Phi(A, rho, rho_dot), H(V, rho))
                            |det d(E)/d(A, rho, rho_dot)| d(rho) d(rho_dot)

on the quadrature nodes produced by :func:`pneo._ranging.build_grid`
(DESIGN §2.3, §2.5), and then forming

    P(NEO) = Lambda_NEO / (Lambda_NEO + Lambda_other)          (DESIGN §2.4)

with no prior knob: both n_C are absolute number densities, so the class
abundances are already contained in the components (D-3).

Per-node bookkeeping
--------------------
The pipeline is fully vectorized over grid nodes:

    (rho, rho_dot)  -> topocentric_to_heliocentric  (equatorial state)
                    -> state_equ_to_ecl             (ecliptic state)
                    -> elements_from_state          (a, e, i, Om, om, M)
                    -> per-density elements3        (q, e, i) or (a, e, i)
    rho, |r|, phase -> absolute_magnitude           H(V, rho)
    integrand_C     = n_C(elements3, H) * |J|_(param of C)
    Lambda_C        = sum(integrand_C * grid.weight)

`ClassificationResult.node_density_*` stores the *integrand* n_C |J| at each
node (not multiplied by the cell weight), which is the (rho, rho_dot)
posterior density map used for the DESIGN §4.1 collapse tests.

Masking.  Two independent things can make a node uncomputable, and both are
zeroed rather than propagated:

1. the implied orbit is unbound or degenerate, so `elements_from_state`
   returns a NaN row (or H is non-finite).  Such nodes are excluded *before*
   the density is called, and the density is handed finite dummy values there,
   so `PopulationComponent` implementations never see NaN.  Counted in
   ``ClassificationResult.n_masked_nonfinite``.
2. the elements are finite but `ranging_jacobian_det` returns NaN — the
   conditioning guard in WP2 trips near the parabolic boundary.  These nodes
   have a well-defined density but no usable measure, so they too contribute
   zero.  They are counted *separately*, in
   ``ClassificationResult.n_masked_det``, because a zero in
   ``node_density_*`` is otherwise indistinguishable from genuinely zero
   density.  With the v0 digest2 components this path is empty; it is
   observable so that future components can be checked.

Efficiency: the composed Jacobian determinant depends only on the density's
*parameterization*, not on the density, so it is computed once per distinct
parameterization in use (one call in the v0 case, where both components are
`Digest2Component` with parameterization "qei").
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ._attributable import Attributable, observer_state
from ._constants import G_SLOPE_DEFAULT
from .models.digest2 import PopulationComponent
from ._ranging import RangingGrid, build_grid
from ._transforms import (
    absolute_magnitude,
    elements_from_state,
    phase_angle,
    qei_from_elements,
    ranging_jacobian_det,
    state_equ_to_ecl,
    topocentric_to_heliocentric,
)

__all__ = ["ClassificationResult", "classify", "classify_many"]

#: Default ranging-grid resolution.  The quadrature error is essentially ALL
#: in rho: H(V, rho) makes every H-bin edge of a binned population model a
#: rho-aligned discontinuity of the integrand, and the q bins add more, so the
#: rho direction needs many columns per bin while the rho_dot direction (across
#: which the integrand is smooth) needs very few.  Measured on a 64-geometry
#: sweep: (200, 100) -> p90 |dp| 0.018 / max 0.071; (200, 400) no better;
#: (1200, 50) -> p90 0.0035 / max 0.010 at ~3x the (200, 100) cost.
N_RHO_DEFAULT: int = 1200
N_RHODOT_DEFAULT: int = 50


def _require_matplotlib():
    """Import matplotlib for the diagnostic plots, or explain how to get it.

    matplotlib is an optional dependency (the ``plot`` extra): scoring never
    needs it, so it is imported only when a plotting method is called.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pneo's diagnostic plots need matplotlib, which is an optional "
            "dependency; install it with `pip install 'pneo[plot]'`"
        ) from exc


def _null_formatter():
    """matplotlib's NullFormatter, imported lazily (diagnostics only)."""
    _require_matplotlib()
    from matplotlib.ticker import NullFormatter

    return NullFormatter()


@dataclass
class ClassificationResult:
    """Output of a single-tracklet classification.

    Attributes
    ----------
    p_neo : float
        P(NEO | A, V) — the calibrated posterior (DESIGN §2.4).
    log_lr : float
        ln(Lambda_NEO / Lambda_other), the prior-free sufficient statistic.
    lambda_neo, lambda_other : float
        The class marginal likelihoods (differential number densities in
        observable space).
    grid : RangingGrid
        The nodes used.
    node_density_neo, node_density_other : np.ndarray
        Per-node integrand values n_C * |J| (same length as grid.rho) — the
        (rho, rho_dot) posterior maps for diagnostics/collapse tests.
    node_H : np.ndarray
        Per-node implied absolute magnitude H(V, rho) [mag]; NaN where the
        photometry is undefined.  For the DESIGN §2.6 truncation diagnostics.
    node_q : np.ndarray
        Per-node implied perihelion distance q = a(1 - e) [AU]; NaN at nodes
        whose implied orbit is unbound.  Also §2.6.
    n_nodes : int
        Total number of quadrature nodes (== grid.rho.size).
    n_masked_nonfinite : int
        Nodes dropped because the implied elements or H were non-finite
        (unbound/degenerate orbits).
    n_masked_det : int
        Nodes with finite elements and H whose Jacobian determinant came back
        non-finite (WP2's conditioning guard, near the parabolic boundary) and
        were therefore zeroed.  Reported separately because such a node is
        indistinguishable, in ``node_density_*``, from one where the density
        is genuinely zero.
    diagnostics : bool
        Whether the per-node arrays above were materialized.  ``False`` (from
        ``classify(diagnostics=False)``) means they are empty *by request*,
        which is a different thing from an empty admissible region — so
        :meth:`measure_fraction_H_above` and :meth:`posterior_mass_where`
        raise instead of returning a meaningless ``0.0``.
    status : str
        **How ``p_neo`` was arrived at**, so that a downstream analysis never
        has to guess.  One of:

        ``"ok"``
            A likelihood-based posterior.  This includes the one-sided cases
            (``p_neo`` exactly 1.0 or 0.0, ``log_lr`` +/-inf), which *are*
            posteriors: one component genuinely has zero support there.
        ``"fully_truncated_flagged"``
            ``p_neo = 1.0`` assigned by the **D-19 triage policy**, not
            computed: the admissible region is non-empty, both Lambdas are
            exactly zero, **and the cause is the H support running out** — at
            least :data:`H_TRUNCATION_MEASURE_MIN` of the region's measure over
            its evaluable nodes lies at or above both models' H support end.
            No posterior exists, so ``log_lr`` is ``nan``.  See
            :func:`_posterior` for the empirical basis (11/11 true NEOs on real
            NEOCP astrometry), :func:`_fully_truncated_in_H` for why the
            trigger is gated on that mechanism rather than on the zero itself,
            and :attr:`policy_assigned` for the boolean.
        ``"both_zero"``
            ``p_neo = nan``: both Lambdas exactly zero over a non-empty region
            whose mechanism is **not** the faint-end wall — the region lies
            inside the H support in cells the model holds no objects in, or
            below the support (implied H < 0), or has no evaluable node at all.
            Genuinely uninformative, and NOT eligible for the triage score.
        ``"empty_region"``
            ``p_neo = nan``: the admissible region is empty (``n_nodes == 0``),
            genuinely uninformative geometry.  Unchanged by D-19.
        ``"overflow"``
            ``p_neo = 0.5``: both Lambdas overflowed to +inf.
        ``"nan"``
            ``p_neo = nan``: a component returned NaN (a bug in that
            component).

        Defaults to ``"ok"`` so that hand-constructed results (tests, custom
        drivers) keep working.
    """

    p_neo: float
    log_lr: float
    lambda_neo: float
    lambda_other: float
    grid: RangingGrid
    node_density_neo: np.ndarray
    node_density_other: np.ndarray
    node_H: np.ndarray = field(default_factory=lambda: np.zeros(0))
    node_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_nodes: int = 0
    n_masked_nonfinite: int = 0
    n_masked_det: int = 0
    #: Whether the per-node arrays were kept (``classify(diagnostics=...)``).
    #: When False the node arrays are empty *by request*, which is a different
    #: thing from an empty admissible region, and the §2.6 helpers below refuse
    #: to answer rather than return a meaningless 0.0.
    diagnostics: bool = True
    #: How ``p_neo`` was arrived at; see the class docstring for the vocabulary.
    #: The one value that must never be conflated with the others is
    #: ``"fully_truncated_flagged"`` — a D-19 policy assignment rather than a
    #: posterior.
    status: str = "ok"

    @property
    def policy_assigned(self) -> bool:
        """True iff ``p_neo`` is a D-19 triage-policy assignment, not a posterior.

        Equivalent to ``status == "fully_truncated_flagged"``.  Exposed as a
        named boolean because this is the question every analysis has to ask
        before pooling scores: a policy-assigned 1.0 carries no calibration
        information (both likelihoods were exactly zero), so it belongs in its
        own accounting line rather than in a reliability bin.  See
        :func:`_posterior` for the full statement of the policy.
        """
        return self.status == "fully_truncated_flagged"

    def _require_diagnostics(self, what: str) -> None:
        if not self.diagnostics:
            raise ValueError(
                f"{what} needs the per-node arrays, but this tracklet was "
                "classified with diagnostics=False, so they were never "
                "materialized. Re-run classify(..., diagnostics=True)."
            )

    # -- DESIGN §2.6 diagnostics ------------------------------------------

    def measure_fraction_H_above(self, h: float) -> float:
        """Fraction of the admissible region's *measure* with implied H > h.

        Sum of the cell weights of nodes whose implied absolute magnitude
        exceeds ``h``, divided by the total weight — i.e. the area fraction of
        the (rho, rho_dot) admissible region that lies beyond a model's H
        support (DESIGN §2.6).  This is a property of the *geometry*, not of
        the posterior: it says how much of the integration domain a truncation
        removes.  Nodes with NaN H count as not exceeding ``h``.  Returns 0.0
        for an empty region.

        ``h`` is **not** defaulted and is not 25.5 by assumption: pass the
        support end of the components you actually scored with —
        ``pneo._classifier.h_support_end(neo, other)``, which is 25.5 for
        the D-4 digest2 pair, ``h_extend_to`` under D-19, and ``inf`` under the
        D-14 clamp (where this diagnostic is vacuous, since nothing is
        truncated above).

        .. warning::
           **This method cannot be used to detect a D-19 policy-assigned
           tracklet.**  It answers a question about the *region*, and it counts
           NaN-H nodes as "not above ``h``", so a tracklet whose nodes were all
           masked returns 0.0 — indistinguishable from a tracklet comfortably
           inside the support.  The reliable detector is the stored triple
           ``(n_nodes > 0, lambda_neo == 0.0, lambda_other == 0.0)``, or
           :attr:`status` / :attr:`policy_assigned` when the result object
           itself is at hand.

        Raises ValueError if the tracklet was classified with
        ``diagnostics=False``: the node arrays were never materialized, and 0.0
        would be indistinguishable from a real answer.
        """
        self._require_diagnostics("measure_fraction_H_above")
        w = np.asarray(self.grid.weight, dtype=float)
        total = float(w.sum())
        if not (total > 0.0):
            return 0.0
        with np.errstate(invalid="ignore"):
            sel = np.asarray(self.node_H, dtype=float) > float(h)
        return float(w[sel].sum() / total)

    def posterior_mass_where(self, mask, component: str = "both",
                             normalize: bool = True) -> float:
        """Posterior mass of the selected nodes.

        Parameters
        ----------
        mask : array of bool, or callable
            A boolean array over the nodes, or a callable taking this result
            and returning one.  Example (the §2.6 truncation corner) — note
            that the H threshold must be the support end of the components
            actually scored, ``h_support_end(neo, other)``, not the literal
            25.5, which is only right for the unextended D-4 pair::

                h_max = pneo._classifier.h_support_end(neo, other)
                res.posterior_mass_where(
                    lambda r: (r.node_q > 1.3) & (r.node_H > h_max)
                )
        component : {"neo", "other", "both"}
            Which integrand to weight by.
        normalize : bool
            If True (default) return the fraction of that component's
            Lambda that the selected nodes carry; if False, the absolute
            (unnormalized) contribution.

        Returns
        -------
        float — 0.0 when the region is empty or the component's Lambda is 0
        (and ``normalize`` is True).

        Raises
        ------
        ValueError
            If the tracklet was classified with ``diagnostics=False`` (the node
            arrays were never materialized), or if ``mask`` has the wrong shape.
        """
        comp = str(component).lower()
        if comp not in ("neo", "other", "both"):
            raise ValueError(
                f"component must be 'neo', 'other' or 'both', got {component!r}"
            )
        self._require_diagnostics("posterior_mass_where")
        node = self._node_map(comp)
        w = np.asarray(self.grid.weight, dtype=float)
        if node.size == 0:
            return 0.0
        sel = mask(self) if callable(mask) else np.asarray(mask)
        sel = np.asarray(sel, dtype=bool)
        if sel.shape != node.shape:
            raise ValueError(
                f"mask has shape {sel.shape}, expected {node.shape}"
            )
        mass = node * w
        mass = np.where(np.isfinite(mass), mass, 0.0)
        picked = float(mass[sel].sum())
        if not normalize:
            return picked
        total = float(mass.sum())
        return picked / total if total > 0.0 else 0.0

    def _node_map(self, comp: str) -> np.ndarray:
        if comp == "neo":
            return np.asarray(self.node_density_neo, dtype=float)
        if comp == "other":
            return np.asarray(self.node_density_other, dtype=float)
        return (
            np.asarray(self.node_density_neo, dtype=float)
            + np.asarray(self.node_density_other, dtype=float)
        )

    # -- diagnostics plot --------------------------------------------------

    def plot_posterior_map(self, ax=None, component: str = "both",
                           colorbar: bool | None = None):
        """Render the (rho, rho_dot) posterior density map(s). Returns the
        matplotlib Axes. Import matplotlib lazily (module must not require a
        display).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure/axes is created if omitted.
        component : {"neo", "other", "both"}
            Which map to render.  "both" shows the *combined* posterior mass
            n_NEO + n_other, which is the map the DESIGN §4.1 collapse test
            reads (it must concentrate around the object's true
            (rho, rho_dot)); "neo"/"other" show the individual components.
        colorbar : bool, optional
            Whether to attach a colorbar.  Default: True when this call
            created the figure, False when ``ax`` was supplied (the caller
            owns that figure's layout).

        Notes
        -----
        The quantity plotted is the posterior *mass per node*,
        ``n_C * |J| * weight`` — i.e. the integrand times the cell measure, so
        that node colours are comparable across the non-uniform grid and sum to
        Lambda_C.  The full admissible region is drawn in grey underneath, so
        that nodes with zero density stay visible.

        Ownership of the figure.  When ``ax`` is omitted this creates a new
        matplotlib figure and **does not close it** — the caller owns it and
        should call ``plt.close(ax.figure)`` when done.  Plotting a whole
        `classify_many` batch without closing leaks one figure per tracklet
        (and trips matplotlib's ``figure.max_open_warning``); pass your own
        ``ax`` or close as you go.  On a caller-supplied Axes this routine
        leaves the title and the x-scale alone — set ``ax.set_xscale("log")``
        yourself, since the rho grid is log-spaced — and only fills in axis
        labels that are still empty.

        Fully headless-safe: matplotlib is imported lazily and only the pyplot
        figure factory is touched, so a non-interactive backend (e.g. Agg) is
        sufficient.
        """
        comp = str(component).lower()
        if comp not in ("neo", "other", "both"):
            raise ValueError(
                f"component must be 'neo', 'other' or 'both', got {component!r}"
            )

        _require_matplotlib()
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        owned = ax is None
        if owned:
            _fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if colorbar is None:
            colorbar = owned

        if not ax.get_xlabel():
            ax.set_xlabel(r"$\rho$  [AU]")
        if not ax.get_ylabel():
            ax.set_ylabel(r"$\dot\rho$  [AU/day]")

        rho = np.asarray(self.grid.rho, dtype=float)
        rho_dot = np.asarray(self.grid.rho_dot, dtype=float)
        weight = np.asarray(self.grid.weight, dtype=float)

        if rho.size == 0:
            if owned:
                ax.set_xscale("log")
                ax.set_title(f"posterior map ({comp}) — no nodes")
            ax.text(
                0.5, 0.5, "empty admissible region",
                transform=ax.transAxes, ha="center", va="center",
            )
            return ax

        if comp == "neo":
            what = r"$n_{\rm NEO}\,|J|\,w$"
        elif comp == "other":
            what = r"$n_{\rm other}\,|J|\,w$"
        else:
            what = r"$(n_{\rm NEO}+n_{\rm other})\,|J|\,w$"
        node = self._node_map(comp)

        mass = node * weight
        mass = np.where(np.isfinite(mass), mass, 0.0)

        # The whole admissible region in grey first (the domain of integration),
        # then the nodes carrying posterior mass on top.  Zero-mass nodes must
        # stay visible: matplotlib's scatter drops points whose colour value is
        # non-finite, so a single scatter would silently rescale the axes to the
        # support of the density.
        ax.scatter(
            rho, rho_dot, s=2.0, marker="s", c="0.85", linewidths=0.0,
            zorder=1,
        )

        positive = mass > 0.0
        sc = None
        if np.any(positive):
            vmax = float(mass[positive].max())
            vmin = float(mass[positive].min())
            # Six decades of dynamic range is plenty for a diagnostic map.
            vmin = max(vmin, vmax * 1e-6)
            norm = LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else Normalize()
            sc = ax.scatter(
                rho[positive], rho_dot[positive],
                c=np.maximum(mass[positive], vmin),
                s=4.0, marker="s", norm=norm, cmap="viridis",
                linewidths=0.0, zorder=2,
            )

        if owned:
            ax.set_xscale("log")
            # Log minor ticks get labelled when the span is under a decade or
            # two, which overlaps badly; keep only the decade labels.
            ax.xaxis.set_minor_formatter(_null_formatter())
            lo, hi = float(rho.min()), float(rho.max())
            ax.set_xlim(lo / 1.2, hi * 1.2)
            ax.set_title(
                f"(rho, rho_dot) posterior map — {comp}   "
                f"P(NEO) = {self.p_neo:.4g}"
            )

        fig = getattr(ax, "figure", None)
        if colorbar and fig is not None and sc is not None:
            fig.colorbar(sc, ax=ax, label=what)
        return ax


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _elements3_for(parameterization: str, elements: np.ndarray) -> np.ndarray:
    """Slice/convert full elements (..., 6) into the density's (..., 3)."""
    if parameterization == "qei":
        return qei_from_elements(elements)
    if parameterization == "aei":
        return np.ascontiguousarray(elements[..., 0:3])
    raise ValueError(
        f"unknown density parameterization {parameterization!r} "
        "(expected 'qei' or 'aei')"
    )


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Replace non-finite entries by 0.0 (masking, per the contract)."""
    return np.where(np.isfinite(x), x, 0.0)


def h_support_end(neo, other) -> float:
    """The H value at and above which BOTH components are exactly zero.

    ``max`` over the pair of each component's ``support["H"][1]``, because
    "outside both supports" needs the *further* of the two boundaries.  For the
    shipped :class:`~pneo.models.digest2.Digest2Component` pair this is 25.5,
    or ``h_extend_to`` under D-19, or ``+inf`` under the D-14 clamp (which
    never runs out of model above 25.5).

    Returns ``nan`` for a component that does not expose a usable ``H``
    support.  That is deliberate and it fails *safe*: the D-19 mechanism test
    below compares ``H >= h_support_end``, which is False for every H when the
    bound is NaN, so an unknown support means the triage policy simply never
    fires and the tracklet keeps the honest ``nan``.
    """
    ends = []
    for d in (neo, other):
        try:
            hi = float(dict(d.support)["H"][1])
        except (AttributeError, TypeError, KeyError, IndexError, ValueError):
            return float("nan")
        ends.append(hi)
    if not ends or any(np.isnan(x) for x in ends):
        return float("nan")
    return float(max(ends))


#: D-19 trigger threshold: the fraction of the admissible region's *measure*
#: that must lie at or above the H support end before the triage policy fires.
#:
#: Not a knob — a declared constant, and the measurement that fixes it is a
#: total separation rather than a tuning:
#:
#:   * the 11 real NEOCP wall tracklets (validation/NEOCP_PILOT.md, the
#:     empirical basis of D-19) have measure fractions
#:     **0.985628 … 1.000000** (minimum: 2021 UY);
#:   * both adversarial-review false-positive classes — a hole in the model
#:     inside the support, and implied H below zero — have measure fraction
#:     **exactly 0.000000**.
#:
#: Any value in (0, 0.9856) separates them perfectly; 0.95 sits inside that
#: gap with a wide margin on both sides.
#:
#: **Why not exactly 1.0**, which is what "the whole region is above the
#: support" would literally mean: 5 of the 11 real cases have a thin sliver of
#: nodes just below 25.5 (minimum implied H 25.40–25.50, i.e. still inside the
#: top bin [24, 25.5)) which happen to fall in *empty* (q, e, i) cells.
#: Requiring 1.0 drops those 5 back to nan and destroys 5/11 of the very
#: measurement the policy rests on.  NEOCP_PILOT.md already characterises these
#: tracklets by exactly this statistic and in exactly this way — "the minimum
#: measure fraction above the support among them is 0.9856" — so a measure
#: criterion below that number is the population's pre-existing definition, not
#: a new one invented here.
#:
#: **Why measure and not a node count**: the measure is grid-independent and is
#: the quantity DESIGN §2.6(b) reports; a node-count fraction moves with node
#: placement (the same 11 give 0.9663 … 1.0 by node count).
H_TRUNCATION_MEASURE_MIN: float = 0.95


def _fully_truncated_in_H(
    n_h_finite: int, w_h_finite: float, w_h_at_or_above_support: float
) -> bool:
    """The D-19 trigger: is a 0/0 outcome caused by the H support running out?

    ``True`` iff there is at least one node with finite implied H **and at
    least :data:`H_TRUNCATION_MEASURE_MIN` of the admissible region's measure
    over those nodes lies at or above the pair's H support end**
    (:func:`h_support_end`).  That is the *mechanism* the triage policy is
    licensed for, and nothing else is.  See :data:`H_TRUNCATION_MEASURE_MIN`
    for why the threshold is 0.95 rather than 1.0, and for the measurement that
    separates the two cases by the whole interval (0, 0.9856).

    Why the mechanism and not the symptom.  "Both Lambdas are exactly zero over
    a non-empty admissible region" has at least three causes, and only the
    first is the DESIGN §2.6 wall:

    1. **the H support runs out** — every node implies an object fainter than
       the model describes.  Fast + faint => close => small => interesting;
       this is the case the NEOCP pilot measured at 11/11 true NEOs, and the
       only one D-19 covers;
    2. **implied H below zero** — a bright, slow, distant hypothesis
       (measured: V = 14, rho in [50, 100] AU, 0.01 deg/day gives implied H in
       [-6.0, -5.3]).  Also support truncation, at the *other* end, and the
       empirical argument inverts: absurdly *large* and bright is the opposite
       of a small close NEO.  Must not fire;
    3. **a hole in the model** — every node's implied H is comfortably inside
       [0, 25.5) but the model happens to hold no objects in those (q, e, i, H)
       cells (measured: V = 6, 0.001 deg/day at the default grid gives 29 900
       nodes with implied H in [4.6, 16.0] and zero density throughout, because
       nothing that bright exists in the digest2 model at those geometries).
       Nothing has been truncated; the model is simply saying "no". Must not
       fire — and firing here would have the classifier assert
       *support truncation* in its warning text about a tracklet where none
       occurred.

    Both false-positive classes were constructed by the adversarial review;
    both are regression-tested (``tests/test_classifier.py``).

    ``n_h_finite == 0`` (equivalently a zero total measure) — no node has a
    finite implied H at all — returns ``False`` **deliberately**.  There is then no evidence about where the
    region sits relative to the support, because the region has no evaluable
    point in it: every node was masked as unbound or non-finite (D-8, WP2's
    guard). That is a geometry/photometry failure, not a faint-end failure, and
    it gets the honest ``nan`` with status ``"both_zero"``.

    Cheap by construction: all three arguments are scalars both backends
    already accumulate while walking the grid, so the test costs nothing and
    works identically in ``diagnostics=False`` mode.

    **Why an exact-zero Lambda is a safe trigger for *this* model.**  The
    branch upstream tests ``lam_neo + lam_other <= 0.0`` exactly, with no
    tolerance, which is only sound because a *non-zero* contribution from the
    digest2 histogram cannot underflow to zero: the smallest non-zero density
    the shipped pair can return is 3.2e-2 (NEO) / 3.6e-5 (other) in the
    interface's units — some 300 orders of magnitude above the double
    denormal floor — and the Jacobian and cell weights are ordinary-magnitude
    numbers. So "exactly 0.0" really does mean "no node had any density",
    never "the mass rounded away".  A v1 population component with genuinely
    tiny per-cell counts (or one normalized to unit total) could break that
    reasoning, at which point the trigger would need a relative floor rather
    than an exact zero.  Recorded here because the assumption is invisible at
    the call site.
    """
    if int(n_h_finite) <= 0 or not (float(w_h_finite) > 0.0):
        return False
    frac = float(w_h_at_or_above_support) / float(w_h_finite)
    return frac >= H_TRUNCATION_MEASURE_MIN


def _posterior(
    lam_neo: float,
    lam_other: float,
    n_nodes: int,
    n_h_finite: int,
    w_h_finite: float,
    w_h_at_or_above_support: float,
) -> tuple[float, float, str]:
    """(p_neo, log_lr, status) from the two marginal likelihoods.

    **This is the single shared post-processing step of both backends.**  The
    Python reference (:func:`classify`) and the Rust glue
    (:func:`pneo._backend._result_from_core`) both call this function and
    then :func:`_warn_status`; nothing about the posterior, the status
    vocabulary or the warnings is implemented per backend, so the two cannot
    drift apart on any of it by construction.

    ``n_nodes`` is the number of quadrature nodes in the admissible region;
    ``n_h_finite`` how many of them have finite implied H (and finite
    elements), i.e. how many the density was actually evaluated at; and
    ``w_h_finite`` the total quadrature measure over those nodes; and
    ``w_h_at_or_above_support`` the part of that measure lying at or above the
    pair's H support end (:func:`h_support_end`).  The last three are the
    **mechanism evidence** for the D-19 triage policy — see
    :func:`_fully_truncated_in_H` for why the policy must be gated on the
    mechanism and not on the symptom, and why the criterion is measure-weighted
    rather than a node count.  All are taken as arguments rather than inferred,
    so that the distinction lives in exactly one place and works identically in
    ``diagnostics=False`` mode on both backends.

    ``status`` is one of:

    ``"ok"``          a usable posterior (includes the one-sided cases, where
                      p_neo is exactly 1.0 or 0.0 and log_lr is +/-inf);
    ``"fully_truncated_flagged"``
                      ``n_nodes > 0``, both Lambdas exactly zero, **and** the
                      cause is the H support running out: at least one node has
                      a finite implied H and at least
                      :data:`H_TRUNCATION_MEASURE_MIN` of the region's measure
                      over those nodes lies at or above the support end
                      (DESIGN §2.6, D-4).  Under the D-19
                      triage policy this returns ``p_neo = 1.0``,
                      ``log_lr = nan``.  **That 1.0 is a policy assignment, not
                      a posterior** — see below;
    ``"both_zero"``   both Lambdas are exactly zero and the D-19 mechanism test
                      does **not** hold — either the region is empty
                      (``n_nodes == 0``; both callers intercept that earlier
                      and report ``"empty_region"`` instead), or no node has a
                      finite implied H, or the region is not predominantly at
                      or above the support end — it lies inside the support (or
                      below it) and the model simply holds no objects there.
                      All of those are genuinely uninformative and return
                      ``nan``. This status **is** reachable from
                      :func:`classify`;
    ``"overflow"``    both Lambdas overflowed to +inf; the ratio is
                      indeterminate and p_neo = 0.5 is returned as the
                      symmetric limit;
    ``"nan"``         a Lambda came back NaN (should not happen: the integrand
                      is sanitized node by node).

    The D-19 triage policy, stated precisely
    ----------------------------------------
    A tracklet in the ``"fully_truncated_flagged"`` state carries **no
    likelihood information at all**:
    both components integrate to exactly zero, so the ratio that defines
    P(NEO) (DESIGN §2.4) does not exist and no amount of arithmetic can
    manufacture it.  What does exist is an empirical fact about which tracklets
    land in this state.  The NEOCP pilot measured it on real MPC discovery
    astrometry (validation/NEOCP_PILOT.md): **11 of 499 tracklets are fully
    truncated and 11 of those 11 are true NEOs** — median sky rate 6.8 deg/day,
    median V 20.7, median true q 0.94 AU.  The mechanism is not statistical
    luck: "the entire admissible region implies H > 25.5" means fast *and*
    faint, which means close, which (at that apparent brightness) means small,
    which in the v0 model's own terms means a near-Earth object.

    The maintainer's decision is therefore to score these 1.0 as a **triage
    policy** rather than refuse them.  Two consequences are deliberate:

    * ``log_lr`` stays ``nan``.  It is the prior-free sufficient statistic
      (D-3), and there is no statistic here — reporting ``+inf`` would claim an
      infinitely strong *likelihood* ratio, which is exactly the false claim
      this policy must not make.
    * the status is its own value, distinct from ``"ok"``, and is carried on
      :class:`ClassificationResult` (``status`` /
      :attr:`~ClassificationResult.policy_assigned`).  Any analysis that mixes
      calibrated posteriors with policy assignments — a reliability diagram, an
      expected-count check, an ECE — can and must separate them; a 1.0 that
      came from this branch is not evidence about the model's calibration.

    The genuinely uninformative cases are all the others, and they keep their
    ``nan``: an empty admissible region (``n_nodes == 0``, status
    ``"empty_region"``), a region with no evaluable node
    (``n_h_finite == 0``), and a region that sits inside — or below — the H
    support where the model simply holds nothing
    (status ``"both_zero"``).  In none of those has anything been observed
    about faintness or speed, so the empirical NEOCP argument does not apply.
    """
    if np.isnan(lam_neo) or np.isnan(lam_other):
        return float("nan"), float("nan"), "nan"

    inf_neo = np.isinf(lam_neo)
    inf_other = np.isinf(lam_other)
    if inf_neo and inf_other:
        return 0.5, 0.0, "overflow"
    if inf_neo:
        return 1.0, float("inf"), "ok"
    if inf_other:
        return 0.0, float("-inf"), "ok"

    total = lam_neo + lam_other
    if total <= 0.0:
        # D-19 triage policy, gated on the MECHANISM: a non-empty admissible
        # region every evaluable node of which lies at or above the pair's H
        # support end.  Empirically 11/11 true NEOs.  Any other route to 0/0 —
        # empty region, no evaluable node, H below the support, or a hole in
        # the model inside the support — falls through to the honest nan.
        if int(n_nodes) > 0 and _fully_truncated_in_H(
            n_h_finite, w_h_finite, w_h_at_or_above_support
        ):
            return 1.0, float("nan"), "fully_truncated_flagged"
        return float("nan"), float("nan"), "both_zero"
    if lam_other <= 0.0:
        return 1.0, float("inf"), "ok"
    if lam_neo <= 0.0:
        return 0.0, float("-inf"), "ok"
    return float(lam_neo / total), float(np.log(lam_neo) - np.log(lam_other)), "ok"


def _warn_status(status: str, lam_neo: float, lam_other: float,
                 stacklevel: int = 3) -> None:
    """Emit the RuntimeWarning that goes with a `_posterior` status.

    Factored out of `classify` so that the Rust backend raises *exactly* the
    same warnings (same text, same category) as the reference path.
    """
    if status == "fully_truncated_flagged":
        warnings.warn(
            "pneo.classify: both marginal likelihoods are zero "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}) over a "
            "NON-EMPTY admissible region whose measure lies almost entirely "
            f"(>= {100.0 * H_TRUNCATION_MEASURE_MIN:g}%) AT OR ABOVE both "
            "population models' H support end "
            "(support truncation at the faint end, DESIGN §2.6, D-4). "
            "P(NEO) = 1.0 is returned as a "
            "TRIAGE POLICY ASSIGNMENT, NOT a posterior — the likelihoods carry "
            "no information whatsoever, and log_lr is nan for that reason. The "
            "policy is empirical (D-19): on real NEOCP discovery astrometry "
            "11 of 11 fully truncated tracklets were true NEOs, because fast + "
            "faint implies close implies small. This result is flagged: "
            "ClassificationResult.status == 'fully_truncated_flagged' "
            "(ClassificationResult.policy_assigned is True). Analyses that "
            "measure calibration MUST separate these from likelihood-based "
            "scores.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    elif status == "both_zero":
        # Reachable, and its text must NOT claim support truncation: this is
        # the branch for every 0/0 whose mechanism is something else (a hole in
        # the model inside the support, implied H below zero, or no evaluable
        # node at all).  See `_fully_truncated_in_H`.
        warnings.warn(
            "pneo.classify: both marginal likelihoods are zero "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}); both "
            "population components evaluate to zero everywhere in the "
            "admissible region. P(NEO) is undefined. This is NOT the D-19 "
            "faint-end case and no triage score is assigned: less than "
            f"{100.0 * H_TRUNCATION_MEASURE_MIN:g}% of the region's measure "
            "lies at or above the models' H support end, so the cause "
            "is not the DESIGN §2.6 H wall but one of — no node with a finite "
            "implied H (all unbound/degenerate), an implied H below the "
            "support (a bright, slow, distant hypothesis), or implied H inside "
            "the support in (q, e, i, H) cells the model holds no objects in. "
            "Inspect ClassificationResult.node_H and node_q "
            "(diagnostics=True) to see which.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    elif status == "overflow":
        warnings.warn(
            "pneo.classify: both marginal likelihoods overflowed to "
            "infinity; their ratio is indeterminate and P(NEO) = 0.5 is "
            "returned as the symmetric limit. Rescale the population "
            "components.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    elif status == "nan":  # pragma: no cover - integrand is sanitized per node
        warnings.warn(
            "pneo.classify: a marginal likelihood is NaN "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}); "
            "this indicates a bug in a PopulationComponent implementation. "
            "P(NEO) is undefined.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def classify(
    att: Attributable,
    neo: PopulationComponent,
    other: PopulationComponent,
    R_obs: np.ndarray | None = None,
    V_obs: np.ndarray | None = None,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    n_rho: int = N_RHO_DEFAULT,
    n_rhodot: int = N_RHODOT_DEFAULT,
    G: float | None = None,
    obscode: str | None = None,
    lon_deg: float | None = None,
    lat_deg: float | None = None,
    height_m: float = 0.0,
    backend: str = "auto",
    diagnostics: bool = True,
) -> ClassificationResult:
    """Classify one tracklet (DESIGN §2.3–2.5).

    Pipeline: observer state (computed from att.epoch if R_obs/V_obs not
    given, geocentric) -> ranging grid over the admissible region -> per-node
    elements, H(V, rho), Jacobian dets (with the parameterization of each
    density) -> Lambda_C as weighted sums -> posterior. Hyperbolic/NaN nodes
    contribute zero. Nothing here raises for a degenerate tracklet; the two
    degenerate outcomes are distinguished by
    ``ClassificationResult.status``, and each carries a warning:

    * **empty admissible region** (``n_nodes == 0``, ``status ==
      "empty_region"``) -> ``p_neo = nan``.  The geometry admits no bound
      orbit; genuinely uninformative.
    * **fully truncated** (``n_nodes > 0`` but both Lambdas exactly zero,
      ``status == "fully_truncated_flagged"``) -> ``p_neo = 1.0``,
      ``log_lr = nan``, by the **D-19 triage policy**.  The region is a real
      set of orbits, all of them outside both models' H support (DESIGN §2.6),
      so no posterior exists; the 1.0 is a policy assignment backed by the
      NEOCP pilot's 11/11 true-NEO measurement, and
      ``ClassificationResult.policy_assigned`` marks it as such.  See
      :func:`_posterior`.

    Parameters
    ----------
    att : Attributable
        The tracklet: (ra, dec, ra_rate, dec_rate) [rad, rad/day], epoch, and
        the apparent magnitude ``v_mag`` (Johnson V).  A non-finite ``v_mag``
        is rejected with a ValueError rather than silently producing an empty
        likelihood.
    neo, other : PopulationComponent
        The two absolutely calibrated components.  Their
        ``parameterization`` attribute selects the reparameterization Jacobian
        applied to each; the determinant is computed once per distinct
        parameterization.
    R_obs, V_obs : (3,) arrays, optional
        Observer heliocentric state, EQUATORIAL frame [AU, AU/day].  When
        omitted it is computed from ``att.epoch`` via
        :func:`pneo._attributable.observer_state`, using the
        ``obscode``/``lon_deg``/``lat_deg``/``height_m`` arguments below (the
        geocenter if none are given).
    rho_min, rho_max : float
        Range bounds of the integration domain [AU] (DESIGN §2.5).
    n_rho, n_rhodot : int
        Grid resolution (log-spaced in rho, linear in rho_dot within the
        per-rho admissible interval).  The defaults are deliberately lopsided
        (1200 x 50): H(V, rho) turns every H-bin and q-bin edge of a binned
        population model into a rho-aligned discontinuity, so essentially all
        of the quadrature error lives in the rho direction, while the
        integrand is smooth across rho_dot (see N_RHO_DEFAULT).
    G : float, optional
        Slope parameter of the IAU H,G phase function; ``None`` (the default)
        uses ``constants.G_SLOPE_DEFAULT`` = 0.15 [D-5].
    obscode, lon_deg, lat_deg, height_m : optional
        Observatory, forwarded verbatim to
        :func:`pneo._attributable.observer_state` when the observer state
        is not supplied directly.  Ignored when both ``R_obs`` and ``V_obs``
        are given.
    backend : {"auto", "python", "rust"}
        Which implementation of the node loop to use.  ``"python"`` is the
        reference pipeline in this module; ``"rust"`` is the fused kernel in
        the ``pneo._core`` extension (see :mod:`pneo._backend`), which
        evaluates the composed Jacobian in closed form instead of building and
        inverting the 6x6 chain.  ``"auto"`` (the default) picks ``"rust"``
        when the extension is importable and falls back to ``"python"``
        otherwise; ``PNEO_BACKEND`` overrides the argument.  Both backends
        return the same ``ClassificationResult`` contract.
    diagnostics : bool
        Whether to populate the per-node diagnostics (``grid``,
        ``node_density_*``, ``node_H``, ``node_q``).  True by default, which is
        the historical behaviour.  With ``False`` those arrays come back empty
        in *both* backends (and the Rust kernel then performs no per-node
        allocation at all); only the scalars and the mask counters are filled.
        The result records the choice in ``ClassificationResult.diagnostics``,
        and the §2.6 node helpers raise rather than answer from empty arrays.

    Raises
    ------
    NotImplementedError
        If ``att`` carries ``sigma_points`` (more than one attributable
        variant) and the resolved backend is ``"python"``.  Marginalizing over
        the attributable distribution is a v1 capability that only the fused
        kernel implements [D-6]; refusing here is deliberate, so that the same
        call cannot silently score just the central attributable on one backend
        and the full ensemble on the other.

    Notes
    -----
    Backend differences.  The two paths are held to |dp_neo| <= 1e-6 and 1e-5
    relative on each Lambda by ``tests/test_backend_parity.py``, and both mask
    counters agree exactly: the Rust kernel reproduces WP2's conditioning guard
    itself rather than approximating it (``rust/src/guard.rs``).  The rho
    quadrature axis is computed once in Python by
    :func:`pneo._backend.rho_axis` -- the same ``np.geomspace`` call
    ``build_grid`` makes -- and handed to the kernel, so the two backends
    integrate over a bit-identical grid on every numpy build and CPU.  Measured on
    the stored gate_v0 catalogue (``validation/backend_parity.py``): max
    |dp_neo| ~ 2e-16, max Lambda relative error ~4e-13, identical
    ``n_masked_nonfinite`` and ``n_masked_det`` node counts.
    """
    from ._backend import (
        ModelNotSupported,
        check_component_pair,
        resolve_backend,
        sigma_point_count,
    )

    # The two components must agree on every paired mode flag (h_interpolation
    # [D-10/D-17], h_top_bin_infinite [D-14]): a mixed pair puts a modelling
    # artifact of the same order as the honest mode change straight into the
    # posterior (pchip review, F1). Guarded here so the PYTHON reference path
    # refuses hand-built mixed pairs just like the Rust path does.
    check_component_pair(neo, other)

    _which = resolve_backend(backend)

    # Sigma points are a v1 capability that only the fused kernel implements.
    # The detection is single-sourced (`backend.sigma_point_count`) and the
    # refusal is unconditional, so no backend can silently score the central
    # attributable and drop the ensemble -- which would make the same call
    # return materially different answers depending on whether the extension
    # happens to be built.
    _k_sigma = sigma_point_count(att)
    if _k_sigma > 1 and _which != "rust":
        raise NotImplementedError(
            f"att.sigma_points carries {_k_sigma} attributable variants, but "
            "the Python reference backend cannot marginalize over the "
            "attributable distribution in v0 (DESIGN §2.1, D-6). Use "
            "backend='rust', or drop att.sigma_points to score the central "
            "attributable."
        )

    if G is None:
        G = G_SLOPE_DEFAULT
    G = float(G)

    v_mag = float(att.v_mag)
    if not np.isfinite(v_mag):
        raise ValueError(
            f"att.v_mag must be finite, got {att.v_mag!r}; a missing apparent "
            "magnitude cannot be marginalized in v0 (DESIGN §2.2)."
        )

    if R_obs is None or V_obs is None:
        R_auto, V_auto = observer_state(
            att.epoch, obscode=obscode, lon_deg=lon_deg, lat_deg=lat_deg,
            height_m=height_m,
        )
        if R_obs is None:
            R_obs = R_auto
        if V_obs is None:
            V_obs = V_auto
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)
    if R_obs.shape != (3,) or V_obs.shape != (3,):
        raise ValueError("R_obs and V_obs must be (3,) equatorial vectors")

    if _which == "rust":
        from ._backend import classify_rust, strict_rust

        try:
            return classify_rust(
                att, neo, other, R_obs, V_obs, rho_min, rho_max, n_rho,
                n_rhodot, G, diagnostics=diagnostics, stacklevel=3,
            )
        except ModelNotSupported:
            # An explicit backend="rust" makes an unsupported model an error;
            # otherwise the Python reference is a correct answer, so use it.
            if strict_rust(backend):
                raise

    grid = build_grid(
        att,
        R_obs,
        V_obs,
        rho_min=rho_min,
        rho_max=rho_max,
        n_rho=n_rho,
        n_rhodot=n_rhodot,
    )

    n_nodes = int(grid.rho.size)
    if n_nodes == 0:
        warnings.warn(
            "pneo.classify: empty admissible region "
            f"(rho in [{rho_min:g}, {rho_max:g}] AU); P(NEO) is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
        empty = np.zeros(0, dtype=float)
        return ClassificationResult(
            p_neo=float("nan"),
            log_lr=float("nan"),
            lambda_neo=0.0,
            lambda_other=0.0,
            grid=grid,
            node_density_neo=empty,
            node_density_other=empty.copy(),
            node_H=empty.copy(),
            node_q=empty.copy(),
            n_nodes=0,
            n_masked_nonfinite=0,
            n_masked_det=0,
            diagnostics=diagnostics,
            status="empty_region",
        )

    rho = grid.rho
    rho_dot = grid.rho_dot

    # --- geometry: (A, rho, rho_dot) -> heliocentric state -> elements -------
    r_equ, v_equ = topocentric_to_heliocentric(
        att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rho_dot, R_obs, V_obs
    )
    r_ecl = state_equ_to_ecl(r_equ)
    v_ecl = state_equ_to_ecl(v_equ)
    elements = elements_from_state(r_ecl, v_ecl)  # (N, 6); NaN rows if unbound

    # --- photometry: H(V, rho) (DESIGN §2.2) --------------------------------
    # r_helio is the SCALAR heliocentric distance; the phase angle is computed
    # in the equatorial frame (it is rotation-invariant, and the ecliptic
    # rotation has not been applied to R_obs).
    r_helio = np.sqrt(np.sum(r_equ * r_equ, axis=-1))
    alpha_ph = phase_angle(r_equ, R_obs)
    H = absolute_magnitude(v_mag, rho, r_helio, alpha_ph, G=G)

    # Nodes whose implied orbit is unbound/degenerate: masked to exactly zero.
    good = np.all(np.isfinite(elements), axis=-1) & np.isfinite(H)
    n_masked_nonfinite = int(np.count_nonzero(~good))

    # Per-node diagnostics for the DESIGN §2.6 truncation checks.
    node_H = np.where(np.isfinite(H), H, np.nan)
    node_q = np.where(good, elements[..., 0] * (1.0 - elements[..., 1]), np.nan)

    # --- Jacobian determinants, one per distinct parameterization -----------
    jac_cache: dict[str, np.ndarray] = {}
    det_bad = np.zeros(n_nodes, dtype=bool)

    def _jac(param: str) -> np.ndarray:
        det = jac_cache.get(param)
        if det is None:
            det = ranging_jacobian_det(
                att.ra,
                att.dec,
                att.ra_rate,
                att.dec_rate,
                rho,
                rho_dot,
                R_obs,
                V_obs,
                parameterization=param,
            )
            jac_cache[param] = det
            # Finite elements but no usable measure: WP2's conditioning guard.
            det_bad[...] = det_bad | (good & ~np.isfinite(det))
        return det

    def _integrand(density: PopulationComponent) -> np.ndarray:
        param = getattr(density, "parameterization", None)
        if param not in ("qei", "aei"):
            raise ValueError(
                f"{type(density).__name__}.parameterization must be 'qei' or "
                f"'aei', got {param!r}"
            )
        det = _sanitize(_jac(param))
        # Feed the density finite dummy values on the masked nodes so that
        # implementations need not defend against NaN; the result is zeroed.
        el3 = _elements3_for(param, np.where(good[:, None], elements, 0.0))
        h = np.where(good, H, 0.0)
        with np.errstate(invalid="ignore"):
            n_c = np.asarray(density.density(el3, h), dtype=float)
        if n_c.shape != rho.shape:
            raise ValueError(
                f"{type(density).__name__}.density returned shape {n_c.shape}, "
                f"expected {rho.shape}"
            )
        # over="ignore": a badly normalized component can overflow here; that
        # is reported through the "overflow" posterior status, not as a numpy
        # warning from inside the hot loop.
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(good, _sanitize(n_c) * det, 0.0)

    node_neo = _integrand(neo)
    node_other = _integrand(other)

    lam_neo = float(np.sum(node_neo * grid.weight))
    lam_other = float(np.sum(node_other * grid.weight))

    # D-19 mechanism evidence, computed from the same `good` mask and the same
    # H array the integrand used.  `h_sup` is NaN for a component with no
    # declared H support, and `H >= nan` is False for every node, so an unknown
    # support means the policy never fires (fails safe).
    h_sup = h_support_end(neo, other)
    n_h_finite = int(np.count_nonzero(good))
    with np.errstate(invalid="ignore"):
        above = good & (H >= h_sup)
    w_h_finite = float(np.sum(grid.weight[good]))
    w_h_above = float(np.sum(grid.weight[above]))

    p_neo, log_lr, status = _posterior(
        lam_neo, lam_other, n_nodes, n_h_finite, w_h_finite, w_h_above
    )
    _warn_status(status, lam_neo, lam_other, stacklevel=2)

    if not diagnostics:
        # Same contract as the Rust backend's diagnostics=False mode: only the
        # scalars and the counters are meaningful.  (The reference path has
        # already paid for the arrays; the flag exists for API symmetry.)
        empty = np.zeros(0, dtype=float)
        grid = RangingGrid(
            rho=empty.copy(), rho_dot=empty.copy(), weight=empty.copy(),
            n_rho=grid.n_rho, n_rhodot=grid.n_rhodot,
        )
        node_neo = empty.copy()
        node_other = empty.copy()
        node_H = empty.copy()
        node_q = empty.copy()

    return ClassificationResult(
        p_neo=p_neo,
        log_lr=log_lr,
        lambda_neo=lam_neo,
        lambda_other=lam_other,
        grid=grid,
        node_density_neo=node_neo,
        node_density_other=node_other,
        node_H=node_H,
        node_q=node_q,
        n_nodes=n_nodes,
        n_masked_nonfinite=n_masked_nonfinite,
        n_masked_det=int(np.count_nonzero(det_bad)),
        diagnostics=diagnostics,
        status=status,
    )


def classify_many(
    atts: list[Attributable],
    neo: PopulationComponent,
    other: PopulationComponent,
    **kwargs,
) -> list[ClassificationResult]:
    """Batch API.

    ``kwargs`` are forwarded verbatim to :func:`classify`, so a shared
    ``R_obs``/``V_obs`` (same epoch for every tracklet) or a common grid
    resolution can be passed once.

    With the Python backend this is a loop over :func:`classify`; with the Rust
    backend (``backend="rust"``, or ``"auto"`` where the extension is present)
    the whole batch is handed to ``pneo._core`` in **one** call, the GIL is
    released, and rayon parallelizes across tracklets.  ``n_threads`` (Rust only)
    caps the worker count; the default uses rayon's global pool, i.e. one thread
    per core.  Observer states are still resolved in Python (astropy), one call
    per distinct epoch.

    Memory.  The batched path materializes every result before returning, so
    with ``diagnostics=True`` (the default) it holds the whole batch's per-node
    arrays at once: up to 3.4 MB per tracklet at the default (1200, 50) grid,
    2.2 MB measured on gate_v0 (admissible-region truncation removes some rho
    columns).  Peak is **one** copy, not two — the kernel hands each node array
    to numpy by adopting the Rust buffer, and the driver drains its result
    vector as it converts, so a tracklet's arrays leave the Rust side before
    the next one is touched (measured peak/retained = 1.02).  Even so, pass
    ``diagnostics=False`` or chunk the batch when scoring more than a few
    hundred tracklets at a time.

    Degenerate tracklets: **read ``res.status``, and never count warnings.**
    Python's default ``"default"`` filter deduplicates by (message, category,
    module, lineno), so a batch of five tracklets with an empty admissible
    region raises the warning once, not five times.

    The two degenerate outcomes no longer share a signal, and this is a change
    from earlier versions of this API:

    * ``res.status == "empty_region"`` (equivalently ``res.n_nodes == 0``) ->
      ``p_neo = nan``.  **``np.isnan(res.p_neo)`` still finds exactly the
      unscorable tracklets**, and after D-19 this is essentially the only way
      to get one.
    * ``res.status == "fully_truncated_flagged"`` (equivalently
      ``res.policy_assigned``) -> ``p_neo = 1.0``, ``log_lr = nan``.  This used
      to be a ``nan`` score too; under the D-19 triage policy it is a
      **defined** score, so a filter written as ``np.isnan(p)`` will no longer
      see it, while a filter written as ``np.isnan(log_lr)`` will see both
      degenerate classes.  Choose deliberately: for "which tracklets could not
      be scored" use ``isnan(p_neo)``; for "which scores are likelihood-based"
      use ``status == "ok"``, i.e. exclude ``policy_assigned``.

    Any calibration statistic (reliability diagram, expected NEO count, ECE,
    Brier) must exclude the policy-assigned tracklets or report them on their
    own line: both their likelihoods were exactly zero, so they carry no
    information about whether the posterior is calibrated.  Ranking statistics
    (AUC) may legitimately include them — the policy is a claim about their
    rank, and it is the claim being tested.
    """
    from ._backend import (
        ModelNotSupported,
        check_component_pair,
        resolve_backend,
        strict_rust,
    )

    check_component_pair(neo, other)  # see classify(); guards the python path too

    n_threads = kwargs.pop("n_threads", None)
    if resolve_backend(kwargs.get("backend", "auto")) == "rust" and atts:
        from ._backend import classify_many_rust

        try:
            return _classify_many_rust(
                atts, neo, other, n_threads=n_threads,
                _impl=classify_many_rust, **kwargs
            )
        except ModelNotSupported:
            if strict_rust(kwargs.get("backend")):
                raise
    kwargs.pop("backend", None)
    return [classify(att, neo, other, backend="python", **kwargs) for att in atts]


def _classify_many_rust(
    atts: list[Attributable],
    neo: PopulationComponent,
    other: PopulationComponent,
    *,
    _impl,
    n_threads: int | None = None,
    R_obs: np.ndarray | None = None,
    V_obs: np.ndarray | None = None,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    n_rho: int = N_RHO_DEFAULT,
    n_rhodot: int = N_RHODOT_DEFAULT,
    G: float | None = None,
    obscode: str | None = None,
    lon_deg: float | None = None,
    lat_deg: float | None = None,
    height_m: float = 0.0,
    backend: str = "auto",
    diagnostics: bool = True,
) -> list[ClassificationResult]:
    """Resolve observer states, then run the whole batch in one Rust call."""
    if G is None:
        G = G_SLOPE_DEFAULT
    G = float(G)

    for att in atts:
        if not np.isfinite(float(att.v_mag)):
            raise ValueError(
                f"att.v_mag must be finite, got {att.v_mag!r}; a missing "
                "apparent magnitude cannot be marginalized in v0 (DESIGN §2.2)."
            )

    n = len(atts)
    if R_obs is None or V_obs is None:
        R_auto = np.empty((n, 3), dtype=float)
        V_auto = np.empty((n, 3), dtype=float)
        # One astropy call per distinct epoch; identical epochs are common in a
        # batch and the ephemeris evaluation dominates the Python-side cost.
        cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for k, att in enumerate(atts):
            key = float(np.asarray(att.epoch.tdb.jd).item())
            hit = cache.get(key)
            if hit is None:
                hit = observer_state(
                    att.epoch, obscode=obscode, lon_deg=lon_deg,
                    lat_deg=lat_deg, height_m=height_m,
                )
                cache[key] = hit
            R_auto[k] = hit[0]
            V_auto[k] = hit[1]
        if R_obs is None:
            R_obs = R_auto
        if V_obs is None:
            V_obs = V_auto

    R_obs = np.broadcast_to(np.asarray(R_obs, dtype=float), (n, 3))
    V_obs = np.broadcast_to(np.asarray(V_obs, dtype=float), (n, 3))

    return _impl(
        atts, neo, other, R_obs, V_obs, rho_min, rho_max, n_rho, n_rhodot, G,
        diagnostics=diagnostics, n_threads=n_threads, stacklevel=4,
    )
