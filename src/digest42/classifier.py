"""WP5 — Marginal likelihoods, posterior, and diagnostics.

Frozen contract (WP0). See DESIGN.md §2.3–2.5, IMPLEMENTATION.md WP5.

What this module does
---------------------
`classify` is the orchestration layer: it turns an :class:`Attributable`
(plus its apparent magnitude) into the calibrated posterior P(NEO | A, V) by
evaluating, for each population component C,

    Lambda_C(A, V) = int_AR n_C(Phi(A, rho, rho_dot), H(V, rho))
                            |det d(E)/d(A, rho, rho_dot)| d(rho) d(rho_dot)

on the quadrature nodes produced by :func:`digest42.ranging.build_grid`
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

Nodes whose implied orbit is unbound/degenerate come back as NaN elements (and
NaN Jacobian) and contribute **exactly zero** — they are masked, not
propagated.  `ClassificationResult.node_density_*` stores the *integrand*
n_C |J| at each node (not multiplied by the cell weight), which is the
(rho, rho_dot) posterior density map used for the DESIGN §4.1 collapse tests.

Efficiency: the composed Jacobian determinant depends only on the density's
*parameterization*, not on the density, so it is computed once per distinct
parameterization in use (one call in the v0 case, where both components are
`Digest2Histogram` with parameterization "qei").
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .attributable import Attributable, observer_state
from .constants import G_SLOPE_DEFAULT
from .population import PopulationDensity
from .ranging import RangingGrid, build_grid
from .transforms import (
    absolute_magnitude,
    elements_from_state,
    phase_angle,
    qei_from_elements,
    ranging_jacobian_det,
    state_equ_to_ecl,
    topocentric_to_heliocentric,
)

__all__ = ["ClassificationResult", "classify", "classify_many"]


def _null_formatter():
    """matplotlib's NullFormatter, imported lazily (diagnostics only)."""
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
    """

    p_neo: float
    log_lr: float
    lambda_neo: float
    lambda_other: float
    grid: RangingGrid
    node_density_neo: np.ndarray
    node_density_other: np.ndarray

    def plot_posterior_map(self, ax=None, component: str = "both"):
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

        Notes
        -----
        The quantity plotted is the posterior *mass per node*,
        ``n_C * |J| * weight`` — i.e. the integrand times the cell measure, so
        that node colours are comparable across the non-uniform grid and sum to
        Lambda_C.  ``rho`` is drawn on a logarithmic axis (the grid is
        log-spaced in rho) and the colour scale is logarithmic where the values
        allow it.  Fully headless-safe: matplotlib is imported lazily and only
        the pyplot figure factory is touched, so a non-interactive backend
        (e.g. Agg) is sufficient.
        """
        comp = str(component).lower()
        if comp not in ("neo", "other", "both"):
            raise ValueError(
                f"component must be 'neo', 'other' or 'both', got {component!r}"
            )

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        if ax is None:
            _fig, ax = plt.subplots(figsize=(6.4, 4.8))

        ax.set_xlabel(r"$\rho$  [AU]")
        ax.set_ylabel(r"$\dot\rho$  [AU/day]")

        rho = np.asarray(self.grid.rho, dtype=float)
        rho_dot = np.asarray(self.grid.rho_dot, dtype=float)
        weight = np.asarray(self.grid.weight, dtype=float)

        if rho.size == 0:
            ax.set_xscale("log")
            ax.text(
                0.5, 0.5, "empty admissible region",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_title(f"posterior map ({comp}) — no nodes")
            return ax

        if comp == "neo":
            node = np.asarray(self.node_density_neo, dtype=float)
            what = r"$n_{\rm NEO}\,|J|\,w$"
        elif comp == "other":
            node = np.asarray(self.node_density_other, dtype=float)
            what = r"$n_{\rm other}\,|J|\,w$"
        else:
            node = (
                np.asarray(self.node_density_neo, dtype=float)
                + np.asarray(self.node_density_other, dtype=float)
            )
            what = r"$(n_{\rm NEO}+n_{\rm other})\,|J|\,w$"

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

        ax.set_xscale("log")
        # Log minor ticks get labelled when the span is under a decade or two,
        # which overlaps badly; keep only the decade labels.
        ax.xaxis.set_minor_formatter(_null_formatter())
        lo, hi = float(rho.min()), float(rho.max())
        ax.set_xlim(lo / 1.2, hi * 1.2)
        ax.set_title(
            f"(rho, rho_dot) posterior map — {comp}   "
            f"P(NEO) = {self.p_neo:.4g}"
        )
        fig = getattr(ax, "figure", None)
        if fig is not None and sc is not None:
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


def _posterior(lam_neo: float, lam_other: float) -> tuple[float, float]:
    """(p_neo, log_lr) from the two marginal likelihoods.

    Both zero (or non-finite) -> (nan, nan): the caller warns.  Exactly one
    zero -> a clean 1.0/0.0 posterior with an infinite log-likelihood ratio.
    """
    total = lam_neo + lam_other
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan")
    if lam_other <= 0.0:
        return 1.0, float("inf")
    if lam_neo <= 0.0:
        return 0.0, float("-inf")
    return float(lam_neo / total), float(np.log(lam_neo) - np.log(lam_other))


def classify(
    att: Attributable,
    neo: PopulationDensity,
    other: PopulationDensity,
    R_obs: np.ndarray | None = None,
    V_obs: np.ndarray | None = None,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    n_rho: int = 200,
    n_rhodot: int = 100,
    G: float | None = None,
) -> ClassificationResult:
    """Classify one tracklet (DESIGN §2.3–2.5).

    Pipeline: observer state (computed from att.epoch if R_obs/V_obs not
    given, geocentric) -> ranging grid over the admissible region -> per-node
    elements, H(V, rho), Jacobian dets (with the parameterization of each
    density) -> Lambda_C as weighted sums -> posterior. Hyperbolic/NaN nodes
    contribute zero. A tracklet with an empty admissible region or zero total
    likelihood returns p_neo = nan with a warning, not an exception.

    Parameters
    ----------
    att : Attributable
        The tracklet: (ra, dec, ra_rate, dec_rate) [rad, rad/day], epoch, and
        the apparent magnitude ``v_mag`` (Johnson V).
    neo, other : PopulationDensity
        The two absolutely calibrated components.  Their
        ``parameterization`` attribute selects the reparameterization Jacobian
        applied to each; the determinant is computed once per distinct
        parameterization.
    R_obs, V_obs : (3,) arrays, optional
        Observer heliocentric state, EQUATORIAL frame [AU, AU/day].  Computed
        from ``att.epoch`` at the geocenter when omitted.
    rho_min, rho_max : float
        Range bounds of the integration domain [AU] (DESIGN §2.5).
    n_rho, n_rhodot : int
        Grid resolution (log-spaced in rho, linear in rho_dot within the
        per-rho admissible interval).
    G : float, optional
        Slope parameter of the IAU H,G phase function; ``None`` (the default)
        uses ``constants.G_SLOPE_DEFAULT`` = 0.15 [D-5].
    """
    if G is None:
        G = G_SLOPE_DEFAULT
    G = float(G)

    if R_obs is None or V_obs is None:
        R_auto, V_auto = observer_state(att.epoch)
        if R_obs is None:
            R_obs = R_auto
        if V_obs is None:
            V_obs = V_auto
    R_obs = np.asarray(R_obs, dtype=float)
    V_obs = np.asarray(V_obs, dtype=float)
    if R_obs.shape != (3,) or V_obs.shape != (3,):
        raise ValueError("R_obs and V_obs must be (3,) equatorial vectors")

    grid = build_grid(
        att,
        R_obs,
        V_obs,
        rho_min=rho_min,
        rho_max=rho_max,
        n_rho=n_rho,
        n_rhodot=n_rhodot,
    )

    n_nodes = grid.rho.size
    if n_nodes == 0:
        warnings.warn(
            "digest42.classify: empty admissible region "
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
    # in the equatorial frame (frame-independent, and the rotation has not been
    # applied to R_obs).
    r_helio = np.sqrt(np.sum(r_equ * r_equ, axis=-1))
    alpha_ph = phase_angle(r_equ, R_obs)
    H = absolute_magnitude(att.v_mag, rho, r_helio, alpha_ph, G=G)

    # --- Jacobian determinants, one per distinct parameterization -----------
    jac_cache: dict[str, np.ndarray] = {}

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
        return det

    # Nodes whose implied orbit is unbound/degenerate: masked to exactly zero.
    good = np.all(np.isfinite(elements), axis=-1) & np.isfinite(H)

    def _integrand(density: PopulationDensity) -> np.ndarray:
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
        return np.where(good, _sanitize(n_c) * det, 0.0)

    node_neo = _integrand(neo)
    node_other = _integrand(other)

    lam_neo = float(np.sum(node_neo * grid.weight))
    lam_other = float(np.sum(node_other * grid.weight))

    p_neo, log_lr = _posterior(lam_neo, lam_other)
    if not np.isfinite(p_neo):
        warnings.warn(
            "digest42.classify: both marginal likelihoods are zero "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}); the "
            "tracklet has no support in either population model (support "
            "truncation, DESIGN §2.6). P(NEO) is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ClassificationResult(
        p_neo=p_neo,
        log_lr=log_lr,
        lambda_neo=lam_neo,
        lambda_other=lam_other,
        grid=grid,
        node_density_neo=node_neo,
        node_density_other=node_other,
    )


def classify_many(
    atts: list[Attributable],
    neo: PopulationDensity,
    other: PopulationDensity,
    **kwargs,
) -> list[ClassificationResult]:
    """Batch API. v0: a loop over classify(); vectorization is a v2 concern.

    ``kwargs`` are forwarded verbatim to :func:`classify`, so a shared
    ``R_obs``/``V_obs`` (same epoch for every tracklet) or a common grid
    resolution can be passed once.  Warnings raised for individual tracklets
    (empty admissible region, zero total likelihood) propagate as usual; the
    corresponding results carry ``p_neo = nan``.
    """
    return [classify(att, neo, other, **kwargs) for att in atts]
