"""Backend selection and the Rust-kernel glue.

digest42 ships two implementations of the WP5 marginal-likelihood integral:

``"python"``
    :func:`digest42.classifier.classify`'s own vectorized numpy pipeline.  It is
    the **reference**: every convention (bin edges, NaN/unbound masking, the
    D-14 clamp, the quadrature measure) is defined by what it does, and the
    6x6 Jacobian chain in :mod:`digest42.transforms` is built and inverted
    numerically there.

``"rust"``
    The fused kernel in the ``digest42._core`` extension module (see
    ``rust/``).  It crosses the Python/Rust boundary once per tracklet (or once
    per batch), generates the ranging grid internally, and evaluates the
    composed Jacobian determinant in the exact Delaunay closed form

        |det d(E)/d(A, rho, rho_dot)| = rho^4 cos^2(dec) * 2/(mu^{3/2} sqrt(a) e sin i)

    (times ``1 - e`` for a ``"qei"`` component), which removes both the per-node
    6x6 inversion and any Kepler solver.

The two backends are cross-validated from opposite directions in
``tests/test_backend_parity.py``.

Selection
---------
``backend="auto"`` (the default everywhere) uses ``"rust"`` when the extension
imports and ``"python"`` otherwise.  The environment variable
``DIGEST42_BACKEND=python`` or ``=rust`` overrides the argument, so a whole run
can be pinned to one backend without touching call sites::

    DIGEST42_BACKEND=python pytest tests/

``DIGEST42_BACKEND=auto`` means "no preference" and does **not** override
anything: a global ``auto`` must not quietly promote an explicit
``backend="python"`` call site back onto the kernel.  Only the two concrete
names override.  Case and surrounding whitespace are ignored; any other value
is an error.

Even when it does override, ``DIGEST42_BACKEND`` is a *preference*, not an
assertion: where the kernel cannot be used (see below) the Python reference
still runs.  Only an explicit ``backend="rust"`` argument turns that situation
into an error.

Limitations of the Rust backend (it falls back to Python unless
``backend="rust"`` was passed explicitly):

* both components must be :class:`digest42.population.Digest2Histogram`-shaped,
  i.e. expose a 4-D ``(q, e, i, H)`` density table on the same bin edges.  A
  component in the smooth-in-H mode must additionally expose the matching
  ``h_coeffs``; the kernel evaluates the very same Horner form on the very same
  precomputed coefficients, which makes the two backends bit-identical there
  rather than merely close.  A declared ``h_interpolation`` the glue cannot
  hand over raises :class:`ModelNotSupported` — it never falls back to the
  piecewise-constant table, which would silently score a different model;
* the bin structure must be representable: strictly increasing finite edges, a
  finite non-negative table, and a cell count that fits in a machine word.
  Anything else raises :class:`ModelNotSupported` (the kernel's own
  ``UnsupportedModel``, translated here), so ``auto`` falls back to the
  reference rather than failing where the reference would succeed;
* ``diagnostics=True`` is only defined for a single attributable variant.

Independently of the backend choice, the two components must agree on every
*paired* density-mode flag — ``h_interpolation`` (D-10) and
``h_top_bin_infinite`` (D-14).  Both change the effective model without touching
the geometry, and P(NEO) is a ratio of the two components' marginal likelihoods,
so a one-sided setting yields an artifact of the mismatch rather than the model
with that knob applied.  :func:`check_component_pair` enforces this and is
called from :func:`core_model_for` and both Rust entry points; it raises
``ValueError`` (a caller error) rather than :class:`ModelNotSupported`, because
falling back to the reference would score the artifact instead of refusing it.
See that function's docstring for the one place the guard does not reach.

Sigma points (``att.sigma_points`` / ``att.sigma_weights``, a ``(K, 5)`` matrix
of ``(ra, dec, ra_rate, dec_rate, v_mag)`` variants and ``(K,)`` weights) are
implemented **only** by the kernel, which accumulates
``Lambda_C = sum_k w_k Lambda_C^(k)`` with a separate admissible region and grid
per variant.  :func:`digest42.classifier.classify` refuses them on the Python
path instead of silently scoring the central attributable; see
:func:`sigma_point_count`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

__all__ = [
    "BACKENDS",
    "ModelNotSupported",
    "sigma_point_count",
    "core_available",
    "requested_backend",
    "resolve_backend",
    "strict_rust",
    "check_component_pair",
    "core_model_for",
    "classify_rust",
    "classify_many_rust",
]

BACKENDS = ("auto", "python", "rust")

_ANGLE_VOLUME = (2.0 * np.pi) ** 3

# ---------------------------------------------------------------------------
# Extension discovery
# ---------------------------------------------------------------------------

_core: Any | None = None
_core_error: str | None = None


def _load_core():
    """Import ``digest42._core`` once; remember the failure if it is absent."""
    global _core, _core_error
    if _core is not None or _core_error is not None:
        return _core
    try:
        from . import _core as mod  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - depends on the build
        _core_error = f"{type(exc).__name__}: {exc}"
        return None
    _core = mod
    return _core


def core_available() -> bool:
    """Whether the compiled ``digest42._core`` kernel can be imported."""
    return _load_core() is not None


def strict_rust(backend: str | None = None) -> bool:
    """Whether an unsupported model must raise rather than fall back.

    Only an explicit ``backend="rust"`` *argument* is strict.  A
    ``DIGEST42_BACKEND=rust`` environment variable is a global preference — it
    is meant to flip a whole test run or script onto the kernel, and must not
    turn a perfectly valid custom :class:`~digest42.population.PopulationDensity`
    into an error.
    """
    return backend is not None and str(backend).strip().lower() == "rust"


def requested_backend(backend: str | None = None) -> str:
    """The effective *request* — the argument, or ``DIGEST42_BACKEND`` if set.

    The argument is validated even when the environment overrides it: a typo in
    a call site must not be hidden by whatever the environment happens to say.
    """
    if backend is not None:
        arg = str(backend).strip().lower()
        if arg not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, got {backend!r}"
            )
    else:
        arg = "auto"
    env = os.environ.get("DIGEST42_BACKEND", "").strip().lower()
    if env:
        if env not in BACKENDS:
            raise ValueError(
                f"DIGEST42_BACKEND={env!r} is not one of {BACKENDS}"
            )
        # "auto" in the environment means "no preference", not "override the
        # call site with auto": a global DIGEST42_BACKEND=auto must not quietly
        # promote an explicit backend="python" back to the kernel.  Only the two
        # concrete names override.  Whitespace and case are already normalized
        # above, so "  Auto  " behaves the same way.
        if env != "auto":
            return env
    return arg


def resolve_backend(backend: str | None = None) -> str:
    """Map ``"auto"``/``None``/``"python"``/``"rust"`` onto a concrete backend.

    ``DIGEST42_BACKEND`` wins over the argument when it names a concrete
    backend (``python``/``rust``); ``auto``, empty and unset all mean "no
    preference" and leave the argument alone.  ``"rust"`` requested explicitly
    but unavailable is an error, not a silent downgrade.
    """
    backend = requested_backend(backend)
    if backend == "auto":
        return "rust" if core_available() else "python"
    if backend == "rust" and not core_available():
        raise RuntimeError(
            "backend='rust' requested but the digest42._core extension is not "
            f"importable ({_core_error}). Build it with rust/build.sh."
        )
    return backend


# ---------------------------------------------------------------------------
# Model adaptation
# ---------------------------------------------------------------------------


class ModelNotSupported(TypeError):
    """The population components cannot be handed to the Rust kernel.

    Raised both for shapes this glue rejects (non-binned components, mismatched
    bin edges) and for ones the kernel itself rejects (``_core.UnsupportedModel``
    — non-monotone edges, a negative/NaN cell, an overflowing cell count), which
    :func:`core_model_for` translates.  Either way it means "the reference can
    do this and the kernel cannot", so ``backend="auto"`` falls back rather than
    failing; only an explicit ``backend="rust"`` propagates it.
    """


def _table_of(density) -> np.ndarray:
    """The 4-D density table of a histogram component (objects per unit volume).

    Uses ``Digest2Histogram._density_table`` when present; otherwise rebuilds it
    from the public ``counts``/``bin_volume`` pair, which is the same quantity
    by construction.
    """
    tab = getattr(density, "_density_table", None)
    if tab is None:
        counts = getattr(density, "counts", None)
        vol = getattr(density, "bin_volume", None)
        if counts is None or vol is None:
            raise ModelNotSupported(
                f"{type(density).__name__} exposes neither _density_table nor "
                "counts/bin_volume; the Rust backend only supports binned "
                "(q, e, i, H) components"
            )
        tab = np.asarray(counts, dtype=float) / (
            np.asarray(vol, dtype=float) * _ANGLE_VOLUME
        )
    tab = np.ascontiguousarray(np.asarray(tab, dtype=float))
    if tab.ndim != 4:
        raise ModelNotSupported(
            f"{type(density).__name__} density table has ndim {tab.ndim}, "
            "expected 4 (q, e, i, H)"
        )
    return tab


def _edges_of(density) -> list[np.ndarray]:
    try:
        return [
            np.ascontiguousarray(np.asarray(getattr(density, name), dtype=float))
            for name in ("q_edges", "e_edges", "i_edges", "h_edges")
        ]
    except AttributeError as exc:
        raise ModelNotSupported(
            f"{type(density).__name__} does not expose q/e/i/h_edges"
        ) from exc


#: Per-component density-mode flags that BOTH components must agree on.
#:
#: Each entry is ``(attribute, default, what it selects)``.  Both are knobs that
#: change the effective population model without touching the geometry, and
#: P(NEO) is a ratio of the two components' marginal likelihoods — so applying
#: one to a single side does not produce "the model with that knob", it produces
#: an artifact of the mismatch.  Measured on gate_v0: a
#: (pchip NEO, histogram other) pair moves p(NEO) by up to 6.4e-3, i.e. as much
#: as the entire honest mode change, in a direction that is pure modelling
#: error.  The class docstrings of both knobs already said "both components
#: together"; this is the check that enforces it.
_PAIRED_MODE_FLAGS = (
    ("h_interpolation", "none", "the H reconstruction (D-10)"),
    ("h_top_bin_infinite", False, "the digest2-emulation H clamp (D-14)"),
)


def check_component_pair(neo, other) -> None:
    """Raise ``ValueError`` if the two components disagree on a density mode.

    Single source of truth for the pairing rule, so the kernel, the model cache
    and the entry points cannot drift apart on it.  Only the *paired* flags are
    checked here; genuinely per-component properties (``parameterization``,
    ``label``, the bin structure) are validated where they are consumed.

    Called from :func:`core_model_for` and from both Rust entry points, so
    every ``backend="rust"`` call and every ``backend="auto"`` call on a build
    with the extension present is guarded.

    .. warning::
       ``backend="python"`` on a hand-built mismatched pair is **not** guarded
       by this function, because the reference path in
       :mod:`digest42.classifier` never funnels the component pair through this
       module — it calls ``neo.density(...)`` and ``other.density(...)``
       directly.  Closing that hole needs one call to this function in
       ``classifier.classify`` / ``classify_many``; it is recorded here rather
       than worked around, because a guard that fires on one backend and not
       the other would be worse than one whose scope is written down.
       :meth:`~digest42.population.Digest2Histogram.neo_other_pair` and
       :meth:`~digest42.population.Digest2Histogram.from_classes` cannot
       produce a mismatched pair, so this is reachable only by pairing two
       separately constructed components by hand.
    """
    for attr, default, what in _PAIRED_MODE_FLAGS:
        a = getattr(neo, attr, default)
        b = getattr(other, attr, default)
        if a == b:
            continue
        raise ValueError(
            f"the two population components disagree on {attr!r}: "
            f"{getattr(neo, 'label', 'neo')!r} has {a!r} and "
            f"{getattr(other, 'label', 'other')!r} has {b!r}. "
            f"{what} must be applied to BOTH components or "
            "neither: P(NEO) is the ratio of the two components' marginal "
            "likelihoods, so a one-sided change compares two different "
            "effective models and the resulting posterior is an artifact of "
            "the mismatch, not a measurement of the knob (measured on "
            "gate_v0: up to 6.4e-3 in p(NEO), the size of the whole honest "
            "mode change). Build the pair with "
            "Digest2Histogram.neo_other_pair(..., "
            f"{attr}=...), which cannot produce a mismatch."
        )


def _h_coeffs_of(density) -> np.ndarray | None:
    """The smooth-in-H Horner coefficients of a component, or None.

    ``h_interpolation="none"`` (or a component that does not know about the
    knob at all) gives ``None`` and the kernel uses the flat density table.
    Anything else *must* hand over usable coefficients: an unknown mode, or a
    ``"pchip"`` component whose coefficients are missing or the wrong shape, is
    :class:`ModelNotSupported` — never a silent downgrade to the
    piecewise-constant table, which would be a different model scored under the
    same name.
    """
    from .population import H_INTERPOLATIONS

    mode = str(getattr(density, "h_interpolation", "none"))
    if mode == "none":
        return None
    if mode not in H_INTERPOLATIONS:
        raise ModelNotSupported(
            f"{type(density).__name__} declares h_interpolation={mode!r}, "
            f"which this build does not know (valid: "
            f"{', '.join(H_INTERPOLATIONS)}); refusing to guess how to "
            "evaluate it"
        )
    co = getattr(density, "h_coeffs", None)
    if co is None:
        raise ModelNotSupported(
            f"{type(density).__name__} declares h_interpolation={mode!r} but "
            "exposes no h_coeffs; the Rust backend cannot reconstruct the "
            "smooth H dependence on its own"
        )
    co = np.ascontiguousarray(np.asarray(co, dtype=float))
    if co.ndim != 5 or co.shape[-1] != 3:
        raise ModelNotSupported(
            f"{type(density).__name__}.h_coeffs has shape {co.shape}, expected "
            "(n_q, n_e, n_i, n_h, 3)"
        )
    return co


def _fingerprint(density) -> tuple:
    """A cache key that changes when the component's *content* changes.

    Keying on ``id()`` alone is wrong: a component can be mutated in place
    after the Rust copy was taken.  Flipping ``h_top_bin_infinite`` is the
    dangerous case — it does not touch the density table at all, so nothing
    downstream notices, and the kernel keeps scoring the stale clamp mode
    (measured: 63% divergence in Lambda, silently).

    The fingerprint therefore carries every field the Rust side copies —
    parameterization, the D-14 clamp flag, the ``h_interpolation`` mode and a
    content check on its coefficients, the provenance strings and the bin
    shape — plus a cheap content check on the density table: its buffer
    address, its exact sum, and a hash of a strided sample.

    ``h_interpolation`` has to be in here for exactly the same reason as the
    D-14 flag: switching the mode changes what the kernel must evaluate but
    leaves the density *table* untouched, so a fingerprint that ignored it
    would hand back a stale piecewise-constant handle for a smooth model.

    Residual assumption, deliberately not defended against, and the same on
    both halves: an in-place edit that preserves the buffer address, the shape,
    the exact sum *and* every sampled element would go unnoticed — for the
    density table and for the coefficient block alike.  Population components
    are expected to be immutable once used (``Digest2Histogram`` marks
    ``h_coeffs`` read-only and hands out a read-only view of ``counts``); this
    guards against the plausible accident, not against an adversary.  The one
    asymmetry left is unavoidable: the table is sampled every ~90th value and
    the coefficients every 33rd, because the arrays are different sizes and
    both budgets are fixed sample counts rather than fixed strides.

    A component that has no ``_density_table`` (so the table is rebuilt from
    ``counts``/``bin_volume`` on every call) gets a fresh buffer address each
    time and therefore never hits the cache.  That is correct, just not fast;
    the shipped :class:`~digest42.population.Digest2Histogram` owns its table.
    """
    tab = np.asarray(getattr(density, "_density_table", None)
                     if getattr(density, "_density_table", None) is not None
                     else _table_of(density), dtype=float)
    flat = np.ascontiguousarray(tab).reshape(-1)
    step = max(1, flat.size // 512)
    try:
        addr = int(tab.__array_interface__["data"][0])
    except (AttributeError, KeyError, TypeError):  # pragma: no cover
        addr = -1
    co = getattr(density, "h_coeffs", None)
    if co is None:
        co_key: tuple = ()
    else:
        co = np.asarray(co, dtype=float)
        cf = np.ascontiguousarray(co).reshape(-1)
        # Denser stride than the table's, and the buffer address as well, so
        # the coefficient block is checked to the SAME standard as the table
        # (the review found this half weaker than the other).  4096 samples of
        # 137,808 coefficients is every 33rd value.
        cstep = max(1, cf.size // 4096)
        try:
            co_addr = int(co.__array_interface__["data"][0])
        except (AttributeError, KeyError, TypeError):  # pragma: no cover
            co_addr = -1
        co_key = (
            cf.size,
            tuple(int(x) for x in co.shape),
            co_addr,
            float(cf.sum()),
            hash(cf[::cstep].tobytes()),
        )
    return (
        type(density).__module__,
        type(density).__qualname__,
        str(getattr(density, "parameterization", None)),
        bool(getattr(density, "h_top_bin_infinite", False)),
        str(getattr(density, "h_interpolation", "none")),
        co_key,
        str(getattr(density, "label", "")),
        str(getattr(density, "variant", "")),
        str(getattr(density, "classes", "")),
        tuple(int(x) for x in tab.shape),
        addr,
        float(flat.sum()),
        hash(flat[::step].tobytes()),
    )


# fingerprint pair -> (neo, other, CoreModel).  Strong references to the
# components are kept so their density buffers cannot be freed and their
# addresses recycled by an unrelated array while the entry lives; the cache is
# FIFO-bounded because a process normally uses one or two model pairs.
_MODEL_CACHE: dict[tuple, tuple[Any, Any, Any]] = {}
_MODEL_CACHE_MAX = 8


def core_model_for(neo, other):
    """Build (and memoize) the Rust ``CoreModel`` for a component pair.

    Raises :class:`ModelNotSupported` if the pair is not a binned
    ``(q, e, i, H)`` model on shared bin edges.
    """
    core = _load_core()
    if core is None:  # pragma: no cover - guarded by resolve_backend
        raise RuntimeError("digest42._core is not available")

    # Before the cache, so a mismatched pair can never be memoized: this is a
    # ValueError (the caller made a modelling error) and NOT ModelNotSupported,
    # because falling back to the Python reference would score the artifact
    # instead of refusing it.
    check_component_pair(neo, other)

    key = (_fingerprint(neo), _fingerprint(other))
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit[2]

    tabs = [_table_of(neo), _table_of(other)]
    if tabs[0].shape != tabs[1].shape:
        raise ModelNotSupported(
            f"components have different bin shapes {tabs[0].shape} and "
            f"{tabs[1].shape}; the Rust backend needs one shared bin structure"
        )
    e_neo, e_other = _edges_of(neo), _edges_of(other)
    for k, (a, b) in enumerate(zip(e_neo, e_other)):
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ModelNotSupported(
                f"components disagree on bin-edge array {k}; the Rust backend "
                "needs one shared bin structure"
            )
    params = []
    for d in (neo, other):
        p = getattr(d, "parameterization", None)
        if p not in ("qei", "aei"):
            raise ModelNotSupported(
                f"{type(d).__name__}.parameterization must be 'qei' or 'aei', "
                f"got {p!r}"
            )
        params.append(p)

    unsupported = getattr(core, "UnsupportedModel", ())
    try:
        cm = _build_core_model(core, tabs, e_neo, params, neo, other)
    except unsupported as exc:
        # A model the kernel cannot represent (non-monotone edges, a negative
        # or NaN cell, an overflowing shape).  Recoverable: the Python
        # reference handles all of these, so let `auto` fall back.
        raise ModelNotSupported(str(exc)) from exc
    if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _MODEL_CACHE[key] = (neo, other, cm)
    return cm


def _build_core_model(core, tabs, e_neo, params, neo, other):
    coeffs = []
    for d, tab in ((neo, tabs[0]), (other, tabs[1])):
        co = _h_coeffs_of(d)
        if co is not None and co.shape[:4] != tab.shape:
            raise ModelNotSupported(
                f"{type(d).__name__}.h_coeffs has cell shape {co.shape[:4]} "
                f"but its density table has shape {tab.shape}"
            )
        coeffs.append(None if co is None else np.ascontiguousarray(co.reshape(-1)))
    return core.CoreModel(
        np.ascontiguousarray(tabs[0].reshape(-1)),
        np.ascontiguousarray(tabs[1].reshape(-1)),
        tuple(int(s) for s in tabs[0].shape),
        e_neo[0],
        e_neo[1],
        e_neo[2],
        e_neo[3],
        neo_parameterization=params[0],
        other_parameterization=params[1],
        neo_h_top_bin_infinite=bool(getattr(neo, "h_top_bin_infinite", False)),
        other_h_top_bin_infinite=bool(
            getattr(other, "h_top_bin_infinite", False)
        ),
        neo_label=str(getattr(neo, "label", "NEO")),
        other_label=str(getattr(other, "label", "other")),
        neo_h_coeffs=coeffs[0],
        other_h_coeffs=coeffs[1],
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

_EMPTY = np.zeros(0, dtype=float)

#: Mirror of `kernel::MAX_NODES_PER_AXIS` / `MAX_NODES_TOTAL`.  Checked here as
#: well so that a count too large for an i64 raises the kernel's ValueError
#: rather than PyO3's OverflowError from the integer conversion.
_MAX_PER_AXIS = 100_000_000
_MAX_TOTAL = 1_000_000_000


def _grid_counts(n_rho, n_rhodot) -> tuple[int, int]:
    """Validate and narrow the node counts before crossing into Rust."""
    n_rho = int(n_rho)
    n_rhodot = int(n_rhodot)
    if n_rho < 1 or n_rhodot < 1:
        raise ValueError("n_rho and n_rhodot must be >= 1")
    if n_rho > _MAX_PER_AXIS or n_rhodot > _MAX_PER_AXIS:
        raise ValueError(
            f"n_rho and n_rhodot must each be <= {_MAX_PER_AXIS} "
            f"(got {n_rho} and {n_rhodot})"
        )
    if n_rho * n_rhodot > _MAX_TOTAL:
        raise ValueError(
            f"n_rho * n_rhodot must be <= {_MAX_TOTAL} nodes "
            f"(got {n_rho} * {n_rhodot})"
        )
    return n_rho, n_rhodot


def sigma_point_count(att) -> int:
    """Number of attributable variants attached to ``att`` (1 if none).

    Single source of truth for "does this tracklet carry sigma points", used by
    both backends.  The Python reference cannot marginalize over them [D-6], so
    :func:`digest42.classifier.classify` raises rather than let one backend
    honour them while the other silently scores the central value.
    """
    pts = getattr(att, "sigma_points", None)
    if pts is None:
        return 1
    n = np.shape(pts)
    if len(n) != 2 or n[1] != 5:
        raise ValueError(
            "att.sigma_points must be (K, 5) = (ra, dec, ra_rate, dec_rate, "
            f"v_mag), got shape {n}"
        )
    return int(n[0])


def _variants_of(att) -> tuple[np.ndarray, np.ndarray]:
    """(K, 5) sigma-point matrix and (K,) weights for one attributable.

    v0 has no covariance marginalization [D-6], so this is always K = 1 unless
    the caller attached a ``sigma_points``/``sigma_weights`` pair (the v1 hook;
    the kernel signature already takes the ensemble).
    """
    pts = getattr(att, "sigma_points", None)
    if pts is None:
        v = np.array(
            [
                [
                    float(att.ra),
                    float(att.dec),
                    float(att.ra_rate),
                    float(att.dec_rate),
                    float(att.v_mag),
                ]
            ],
            dtype=float,
        )
        return v, np.ones(1, dtype=float)
    v = np.ascontiguousarray(np.asarray(pts, dtype=float))
    if v.ndim != 2 or v.shape[1] != 5:
        raise ValueError(
            "att.sigma_points must be (K, 5) = (ra, dec, ra_rate, dec_rate, "
            f"v_mag), got shape {v.shape}"
        )
    w = getattr(att, "sigma_weights", None)
    w = (
        np.ones(v.shape[0], dtype=float)
        if w is None
        else np.ascontiguousarray(np.asarray(w, dtype=float))
    )
    if w.shape != (v.shape[0],):
        raise ValueError(
            f"att.sigma_weights must have shape ({v.shape[0]},), got {w.shape}"
        )
    return v, w


def _result_from_core(
    out: dict,
    n_rho: int,
    n_rhodot: int,
    rho_min: float,
    rho_max: float,
    diagnostics: bool,
    stacklevel: int,
):
    """Assemble a :class:`ClassificationResult` from the kernel's output dict.

    Warning text, posterior conventions and the empty-region contract are taken
    from :mod:`digest42.classifier` itself, so the two backends cannot drift
    apart on them.
    """
    from .classifier import ClassificationResult, _posterior, _warn_status
    from .ranging import RangingGrid

    n_nodes = int(out["n_nodes"])
    if n_nodes == 0:
        warnings.warn(
            "digest42.classify: empty admissible region "
            f"(rho in [{rho_min:g}, {rho_max:g}] AU); P(NEO) is undefined.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
        grid = RangingGrid(
            rho=_EMPTY.copy(),
            rho_dot=_EMPTY.copy(),
            weight=_EMPTY.copy(),
            n_rho=int(n_rho),
            n_rhodot=int(n_rhodot),
        )
        return ClassificationResult(
            p_neo=float("nan"),
            log_lr=float("nan"),
            lambda_neo=0.0,
            lambda_other=0.0,
            grid=grid,
            node_density_neo=_EMPTY.copy(),
            node_density_other=_EMPTY.copy(),
            node_H=_EMPTY.copy(),
            node_q=_EMPTY.copy(),
            n_nodes=0,
            n_masked_nonfinite=0,
            n_masked_det=0,
            diagnostics=diagnostics,
        )

    lam_neo = float(out["lambda_neo"])
    lam_other = float(out["lambda_other"])
    p_neo, log_lr, status = _posterior(lam_neo, lam_other)
    _warn_status(status, lam_neo, lam_other, stacklevel=stacklevel)

    if diagnostics:
        grid = RangingGrid(
            rho=out["rho"],
            rho_dot=out["rho_dot"],
            weight=out["weight"],
            n_rho=int(n_rho),
            n_rhodot=int(n_rhodot),
        )
        node_neo = out["integrand_neo"]
        node_other = out["integrand_other"]
        node_H = out["node_H"]
        node_q = out["node_q"]
    else:
        grid = RangingGrid(
            rho=_EMPTY.copy(),
            rho_dot=_EMPTY.copy(),
            weight=_EMPTY.copy(),
            n_rho=int(n_rho),
            n_rhodot=int(n_rhodot),
        )
        node_neo = _EMPTY.copy()
        node_other = _EMPTY.copy()
        node_H = _EMPTY.copy()
        node_q = _EMPTY.copy()

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
        n_masked_nonfinite=int(out["n_masked_nonfinite"]),
        n_masked_det=int(out["n_masked_det"]),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Entry points used by digest42.classifier
# ---------------------------------------------------------------------------


def classify_rust(
    att,
    neo,
    other,
    R_obs: np.ndarray,
    V_obs: np.ndarray,
    rho_min: float,
    rho_max: float,
    n_rho: int,
    n_rhodot: int,
    G: float,
    diagnostics: bool = True,
    stacklevel: int = 3,
):
    """Single-tracklet Rust path. ``R_obs``/``V_obs`` must already be resolved."""
    check_component_pair(neo, other)
    core = _load_core()
    n_rho, n_rhodot = _grid_counts(n_rho, n_rhodot)
    cm = core_model_for(neo, other)
    variants, weights = _variants_of(att)
    out = core.classify_core(
        cm,
        variants,
        weights,
        np.ascontiguousarray(R_obs, dtype=float),
        np.ascontiguousarray(V_obs, dtype=float),
        float(rho_min),
        float(rho_max),
        int(n_rho),
        int(n_rhodot),
        float(G),
        bool(diagnostics),
    )
    return _result_from_core(
        out, n_rho, n_rhodot, rho_min, rho_max, diagnostics, stacklevel
    )


def classify_many_rust(
    atts,
    neo,
    other,
    R_obs,
    V_obs,
    rho_min: float,
    rho_max: float,
    n_rho: int,
    n_rhodot: int,
    G: float,
    diagnostics: bool = True,
    n_threads: int | None = None,
    stacklevel: int = 3,
):
    """Batched Rust path: one boundary crossing, rayon across tracklets.

    ``R_obs``/``V_obs`` are ``(T, 3)`` arrays, already resolved per tracklet.
    The GIL is released for the whole batch.
    """
    check_component_pair(neo, other)
    core = _load_core()
    n_rho, n_rhodot = _grid_counts(n_rho, n_rhodot)
    cm = core_model_for(neo, other)
    T = len(atts)
    if T == 0:
        return []
    per = [_variants_of(a) for a in atts]
    K = per[0][0].shape[0]
    if any(v.shape[0] != K for v, _ in per):
        raise ValueError(
            "all tracklets in a batch must carry the same number of sigma "
            "points (K); got " + repr(sorted({v.shape[0] for v, _ in per}))
        )
    variants = np.ascontiguousarray(
        np.stack([v for v, _ in per], axis=0), dtype=float
    )
    weights = np.ascontiguousarray(
        np.stack([w for _, w in per], axis=0), dtype=float
    )
    outs = core.classify_batch(
        cm,
        variants,
        weights,
        np.ascontiguousarray(R_obs, dtype=float).reshape(T, 3),
        np.ascontiguousarray(V_obs, dtype=float).reshape(T, 3),
        float(rho_min),
        float(rho_max),
        int(n_rho),
        int(n_rhodot),
        float(G),
        bool(diagnostics),
        None if n_threads is None else int(n_threads),
    )
    return [
        _result_from_core(
            o, n_rho, n_rhodot, rho_min, rho_max, diagnostics, stacklevel
        )
        for o in outs
    ]
