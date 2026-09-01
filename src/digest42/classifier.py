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

`ClassificationResult.node_density_*` stores the *integrand* n_C |J| at each
node (not multiplied by the cell weight), which is the (rho, rho_dot)
posterior density map used for the DESIGN §4.1 collapse tests.

Masking.  Two independent things can make a node uncomputable, and both are
zeroed rather than propagated:

1. the implied orbit is unbound or degenerate, so `elements_from_state`
   returns a NaN row (or H is non-finite).  Such nodes are excluded *before*
   the density is called, and the density is handed finite dummy values there,
   so `PopulationDensity` implementations never see NaN.  Counted in
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
`Digest2Histogram` with parameterization "qei").
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

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

#: Default ranging-grid resolution.  The quadrature error is essentially ALL
#: in rho: H(V, rho) makes every H-bin edge of a binned population model a
#: rho-aligned discontinuity of the integrand, and the q bins add more, so the
#: rho direction needs many columns per bin while the rho_dot direction (across
#: which the integrand is smooth) needs very few.  Measured on a 64-geometry
#: sweep: (200, 100) -> p90 |dp| 0.018 / max 0.071; (200, 400) no better;
#: (1200, 50) -> p90 0.0035 / max 0.010 at ~3x the (200, 100) cost.
N_RHO_DEFAULT: int = 1200
N_RHODOT_DEFAULT: int = 50


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

    # -- DESIGN §2.6 diagnostics ------------------------------------------

    def measure_fraction_H_above(self, h: float) -> float:
        """Fraction of the admissible region's *measure* with implied H > h.

        Sum of the cell weights of nodes whose implied absolute magnitude
        exceeds ``h``, divided by the total weight — i.e. the area fraction of
        the (rho, rho_dot) admissible region that lies beyond a model's H
        support (DESIGN §2.6; ``h = 25.5`` for the digest2 components).  This
        is a property of the *geometry*, not of the posterior: it says how much
        of the integration domain a truncation removes.  Nodes with NaN H count
        as not exceeding ``h``.  Returns 0.0 for an empty region.
        """
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
            and returning one.  Example (the §2.6 truncation corner):
            ``res.posterior_mass_where(lambda r: (r.node_q > 1.3) & (r.node_H > 25.5))``
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
        """
        comp = str(component).lower()
        if comp not in ("neo", "other", "both"):
            raise ValueError(
                f"component must be 'neo', 'other' or 'both', got {component!r}"
            )
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


def _posterior(lam_neo: float, lam_other: float) -> tuple[float, float, str]:
    """(p_neo, log_lr, status) from the two marginal likelihoods.

    ``status`` is one of:

    ``"ok"``          a usable posterior (includes the one-sided cases, where
                      p_neo is exactly 1.0 or 0.0 and log_lr is +/-inf);
    ``"both_zero"``   both Lambdas are exactly zero — no support in either
                      model (support truncation, DESIGN §2.6);
    ``"overflow"``    both Lambdas overflowed to +inf; the ratio is
                      indeterminate and p_neo = 0.5 is returned as the
                      symmetric limit;
    ``"nan"``         a Lambda came back NaN (should not happen: the integrand
                      is sanitized node by node).
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
        return float("nan"), float("nan"), "both_zero"
    if lam_other <= 0.0:
        return 1.0, float("inf"), "ok"
    if lam_neo <= 0.0:
        return 0.0, float("-inf"), "ok"
    return float(lam_neo / total), float(np.log(lam_neo) - np.log(lam_other)), "ok"


def classify(
    att: Attributable,
    neo: PopulationDensity,
    other: PopulationDensity,
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
        the apparent magnitude ``v_mag`` (Johnson V).  A non-finite ``v_mag``
        is rejected with a ValueError rather than silently producing an empty
        likelihood.
    neo, other : PopulationDensity
        The two absolutely calibrated components.  Their
        ``parameterization`` attribute selects the reparameterization Jacobian
        applied to each; the determinant is computed once per distinct
        parameterization.
    R_obs, V_obs : (3,) arrays, optional
        Observer heliocentric state, EQUATORIAL frame [AU, AU/day].  When
        omitted it is computed from ``att.epoch`` via
        :func:`digest42.attributable.observer_state`, using the
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
        :func:`digest42.attributable.observer_state` when the observer state
        is not supplied directly.  Ignored when both ``R_obs`` and ``V_obs``
        are given.
    """
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
            node_H=empty.copy(),
            node_q=empty.copy(),
            n_nodes=0,
            n_masked_nonfinite=0,
            n_masked_det=0,
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
        # over="ignore": a badly normalized component can overflow here; that
        # is reported through the "overflow" posterior status, not as a numpy
        # warning from inside the hot loop.
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(good, _sanitize(n_c) * det, 0.0)

    node_neo = _integrand(neo)
    node_other = _integrand(other)

    lam_neo = float(np.sum(node_neo * grid.weight))
    lam_other = float(np.sum(node_other * grid.weight))

    p_neo, log_lr, status = _posterior(lam_neo, lam_other)
    if status == "both_zero":
        warnings.warn(
            "digest42.classify: both marginal likelihoods are zero "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}); the "
            "tracklet has no support in either population model (support "
            "truncation, DESIGN §2.6). P(NEO) is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif status == "overflow":
        warnings.warn(
            "digest42.classify: both marginal likelihoods overflowed to "
            "infinity; their ratio is indeterminate and P(NEO) = 0.5 is "
            "returned as the symmetric limit. Rescale the population "
            "components.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif status == "nan":  # pragma: no cover - integrand is sanitized per node
        warnings.warn(
            "digest42.classify: a marginal likelihood is NaN "
            f"(Lambda_NEO = {lam_neo:g}, Lambda_other = {lam_other:g}); "
            "this indicates a bug in a PopulationDensity implementation. "
            "P(NEO) is undefined.",
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
        node_H=node_H,
        node_q=node_q,
        n_nodes=n_nodes,
        n_masked_nonfinite=n_masked_nonfinite,
        n_masked_det=int(np.count_nonzero(det_bad)),
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
    resolution can be passed once.

    Degenerate tracklets (empty admissible region, zero total likelihood) come
    back with ``p_neo = nan``; **that is the reliable per-tracklet signal**.
    Do not count warnings: Python's default ``"default"`` filter deduplicates
    by (message, category, module, lineno), so a batch of five tracklets with
    an empty admissible region raises the warning once, not five times. Filter
    the results on ``np.isnan(res.p_neo)`` instead (and on ``res.n_nodes == 0``
    to tell an empty region from a fully truncated one).
    """
    return [classify(att, neo, other, **kwargs) for att in atts]
