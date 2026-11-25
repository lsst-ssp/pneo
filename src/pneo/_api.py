"""The public Python API: plain functions, numpy in and numpy out.

Units throughout are degrees and degrees per day.  ``ra_rate`` is the on-sky
rate in right ascension, d(alpha)/dt * cos(delta).  Epochs are MJD (UTC) or
astropy ``Time`` objects; magnitudes are Johnson V.
"""

from __future__ import annotations

import functools
import os
import warnings

import numpy as np

__all__ = [
    "classify",
    "posterior_map",
    "plot_posterior_map",
    "read_observations",
    "fetch",
    "TRACKLET_METADATA",
]

#: Keys :func:`read_observations` returns alongside the classifier inputs.
#: :func:`classify` accepts and ignores them, so ``classify(**tracklets)`` works.
TRACKLET_METADATA = ("object_id", "n_obs", "arc_hours", "resid_rms_arcsec")

#: Fields of the structured array ``classify(..., details=True)`` returns.
DETAIL_DTYPE = np.dtype([
    ("p_neo", "f8"),
    ("log_lr", "f8"),
    ("lambda_neo", "f8"),
    ("lambda_other", "f8"),
    ("status", "U16"),
])


@functools.lru_cache(maxsize=4)
def _cached_default_model(path: str, mtime: float):
    from .models import load_model

    return load_model(path)


def _default_model():
    """The default model, loaded once per process (reloaded if the file changes)."""
    from .paths import default_model_path

    path = str(default_model_path())
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no population model at {path}; run `pneo fetch` to download it, "
            "or pass model=pneo.load_model(PATH)"
        )
    return _cached_default_model(path, os.path.getmtime(path))


def _epochs(epoch, n):
    from astropy.time import Time

    if isinstance(epoch, Time):
        t = epoch.utc
    else:
        t = Time(np.asarray(epoch, dtype=float), format="mjd", scale="utc")
    if t.isscalar:
        t = Time(np.full(n, t.mjd), format="mjd", scale="utc")
    elif t.shape != (n,):
        t = Time(np.broadcast_to(t.mjd, (n,)).copy(), format="mjd", scale="utc")
    return t


def _observer_states(t, n, obscode, observer, obscodes):
    """Heliocentric equatorial observer states, shape (n, 3) each."""
    from ._attributable import observer_state
    from ._ingest import ObscodeTable

    if observer is not None:
        if obscode is not None:
            raise ValueError("pass either obscode= or observer=, not both")
        R, V = observer
        R = np.broadcast_to(np.asarray(R, dtype=float), (n, 3)).copy()
        V = np.broadcast_to(np.asarray(V, dtype=float), (n, 3)).copy()
        return R, V

    codes = np.broadcast_to(
        np.asarray("500" if obscode is None else obscode, dtype=object), (n,)
    )
    table = ObscodeTable(None if obscodes is None else str(obscodes))
    R = np.empty((n, 3))
    V = np.empty((n, 3))
    for code in set(codes.tolist()):
        idx = np.flatnonzero(codes == code)
        lon, lat, height, source = table.resolve(code)
        site = {} if source in ("geocenter", "fallback") else {
            "lon_deg": lon, "lat_deg": lat, "height_m": height}
        # One ephemeris evaluation per distinct epoch: batches often share
        # epochs, and the ephemeris dominates the Python-side cost.
        sub = t[idx]
        _, first, inverse = np.unique(np.stack([sub.jd1, sub.jd2], axis=-1),
                                      axis=0, return_index=True,
                                      return_inverse=True)
        r, v = observer_state(sub[first], **site)
        R[idx] = np.reshape(r, (-1, 3))[inverse.ravel()]
        V[idx] = np.reshape(v, (-1, 3))[inverse.ravel()]
    return R, V


def _prepare(ra, dec, ra_rate, dec_rate, epoch, mag, obscode, observer, obscodes):
    """Broadcast the inputs and build the internal attributables."""
    from ._attributable import Attributable

    arrs = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                 for x in (ra, dec, ra_rate, dec_rate, mag)))
    shape = arrs[0].shape
    ra, dec, ra_rate, dec_rate, mag = (a.ravel() for a in arrs)
    n = ra.size
    t = _epochs(epoch, n)
    R, V = _observer_states(t, n, obscode, observer, obscodes)

    ra_r, dec_r = np.radians(ra), np.radians(dec)
    # Internally the attributable carries the raw d(alpha)/dt in rad/day.
    ra_rate_raw = np.radians(ra_rate) / np.cos(dec_r)
    dec_rate_r = np.radians(dec_rate)
    # One scalar Time object per distinct epoch, shared by the tracklets that
    # have it (indexing an astropy Time is slow enough to matter in big batches).
    _, first, inverse = np.unique(np.stack([t.jd1, t.jd2], axis=-1), axis=0,
                                  return_index=True, return_inverse=True)
    epochs = [t[i] for i in first]
    inverse = inverse.ravel()
    atts = [
        Attributable(ra=float(ra_r[k]), dec=float(dec_r[k]),
                     ra_rate=float(ra_rate_raw[k]), dec_rate=float(dec_rate_r[k]),
                     epoch=epochs[inverse[k]], v_mag=float(mag[k]))
        for k in range(n)
    ]
    return atts, R, V, shape


def _run(atts, R, V, model, backend, workers, grid, diagnostics):
    """Classify the attributables; returns the internal result objects."""
    from ._backend import ModelNotSupported, resolve_backend
    from ._classifier import classify as _classify_one
    from ._classifier import classify_many as _classify_many

    if resolve_backend(backend) == "rust":
        try:
            return _classify_many(
                atts, model.neo, model.other, R_obs=R, V_obs=V, backend="rust",
                diagnostics=diagnostics, n_threads=workers, **grid,
            )
        except ModelNotSupported:
            if backend == "rust" or os.environ.get("PNEO_BACKEND") == "rust":
                raise
    return [
        _classify_one(att, model.neo, model.other, R_obs=R[k], V_obs=V[k],
                      backend="python", diagnostics=diagnostics, **grid)
        for k, att in enumerate(atts)
    ]


def classify(
    ra, dec, ra_rate, dec_rate, epoch, mag, *,
    model=None,
    obscode=None,
    observer=None,
    obscodes=None,
    details: bool = False,
    backend: str = "auto",
    workers: int | None = None,
    n_rho: int = 1200,
    n_rhodot: int = 50,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    G: float | None = None,
    **metadata,
):
    """Probability that each tracklet is a near-Earth object.

    Parameters
    ----------
    ra, dec : array_like
        Position at ``epoch`` [deg].
    ra_rate, dec_rate : array_like
        Angular rates [deg/day]; ``ra_rate`` is the on-sky rate,
        d(alpha)/dt * cos(delta).
    epoch : array_like or astropy.time.Time
        Epoch of the position and rates; numbers are MJD (UTC).
    mag : array_like
        Apparent magnitude, Johnson V.
    model : PopulationModel, optional
        The population model; defaults to :func:`pneo.load_model`'s default,
        loaded once per process.
    obscode : str or array of str, optional
        MPC observatory code(s); the built-in table is used first, then the
        MPC list downloaded by ``pneo fetch`` (or ``obscodes``).  Defaults to
        the geocenter.
    observer : (position, velocity), optional
        Heliocentric observer state in the equatorial (ICRS) frame, [AU] and
        [AU/day], shape (3,) or (n, 3).  Use instead of ``obscode`` for an
        observer without one, such as a spacecraft.
    obscodes : str or path-like, optional
        A local copy of the MPC list of observatory codes.
    details : bool
        Return a structured array with fields ``p_neo``, ``log_lr``,
        ``lambda_neo``, ``lambda_other`` and ``status`` instead of P(NEO) alone.
    backend : {"auto", "rust", "python"}
        ``"auto"`` uses the compiled kernel when it is installed.
    workers : int, optional
        Threads for the compiled kernel (default: all cores).
    n_rho, n_rhodot, rho_min, rho_max, G
        Integration grid, distance range [AU] and H,G phase slope (default
        0.15); the defaults are the paper's configuration.
    **metadata
        The extra keys :func:`read_observations` returns (``object_id``,
        ``n_obs``, ``arc_hours``, ``resid_rms_arcsec``) are accepted and
        ignored, so ``classify(**tracklets)`` works.

    Returns
    -------
    numpy.ndarray or float
        P(NEO), with the broadcast shape of the inputs (a float for scalar
        inputs).  ``status`` (with ``details=True``) is ``"ok"`` for a
        computed probability, ``"beyond_model"`` when every orbit consistent
        with the tracklet is fainter than the model covers (P(NEO) = 1 by
        rule), ``"no_support"`` when neither component has objects there
        (P(NEO) undefined) and ``"empty_region"`` when no bound orbit fits.
    """
    unknown = set(metadata) - set(TRACKLET_METADATA)
    if unknown:
        raise TypeError(f"classify() got unexpected keyword argument(s): "
                        f"{', '.join(sorted(unknown))}")
    if model is None:
        model = _default_model()
    atts, R, V, shape = _prepare(ra, dec, ra_rate, dec_rate, epoch, mag,
                                 obscode, observer, obscodes)
    grid = dict(n_rho=n_rho, n_rhodot=n_rhodot, rho_min=rho_min,
                rho_max=rho_max, G=G)
    results = _run(atts, R, V, model, backend, workers, grid, diagnostics=False)

    if details:
        out = np.empty(len(results), dtype=DETAIL_DTYPE)
        for k, r in enumerate(results):
            out[k] = (r.p_neo, r.log_lr, r.lambda_neo, r.lambda_other, r.status)
        return out.reshape(shape) if shape else out[0]
    p = np.array([r.p_neo for r in results], dtype=float)
    return p.reshape(shape) if shape else float(p[0])


def posterior_map(
    ra, dec, ra_rate, dec_rate, epoch, mag, *,
    model=None, obscode=None, observer=None, obscodes=None,
    backend: str = "auto",
    n_rho: int = 1200, n_rhodot: int = 50,
    rho_min: float = 0.01, rho_max: float = 100.0, G: float | None = None,
) -> dict:
    """Where the probability lies for one tracklet, over distance and range-rate.

    Takes the same inputs as :func:`classify`, for a single tracklet, and
    returns a dict of numpy arrays, one entry per grid point: ``rho`` [AU],
    ``rho_dot`` [AU/day], ``weight`` (the cell area), ``density_neo`` and
    ``density_other`` (the integrands; their sum times ``weight`` is the
    posterior), ``H`` (implied absolute magnitude) and ``q`` (perihelion
    distance [AU]); plus the scalars ``p_neo`` and ``status``.
    """
    if np.size(ra) != 1:
        raise ValueError("posterior_map() takes a single tracklet")
    if model is None:
        model = _default_model()
    atts, R, V, _ = _prepare(ra, dec, ra_rate, dec_rate, epoch, mag,
                             obscode, observer, obscodes)
    grid = dict(n_rho=n_rho, n_rhodot=n_rhodot, rho_min=rho_min,
                rho_max=rho_max, G=G)
    (r,) = _run(atts, R, V, model, backend, None, grid, diagnostics=True)
    return {
        "rho": r.grid.rho,
        "rho_dot": r.grid.rho_dot,
        "weight": r.grid.weight,
        "density_neo": r.node_density_neo,
        "density_other": r.node_density_other,
        "H": r.node_H,
        "q": r.node_q,
        "p_neo": float(r.p_neo),
        "status": r.status,
        "n_rho": int(r.grid.n_rho),
        "n_rhodot": int(r.grid.n_rhodot),
    }


def plot_posterior_map(m: dict, ax=None, component: str = "both",
                       colorbar: bool | None = None):
    """Plot a :func:`posterior_map` result; returns the matplotlib Axes.

    ``component`` is ``"neo"``, ``"other"`` or ``"both"`` (the combined
    posterior).  Needs matplotlib (``pip install 'pneo[plot]'``).
    """
    from ._classifier import ClassificationResult
    from ._ranging import RangingGrid

    res = ClassificationResult(
        p_neo=m["p_neo"], log_lr=float("nan"), lambda_neo=float("nan"),
        lambda_other=float("nan"),
        grid=RangingGrid(rho=m["rho"], rho_dot=m["rho_dot"], weight=m["weight"],
                         n_rho=m["n_rho"], n_rhodot=m["n_rhodot"]),
        node_density_neo=m["density_neo"], node_density_other=m["density_other"],
        node_H=m["H"], node_q=m["q"], n_nodes=len(m["rho"]),
        diagnostics=True, status=m["status"],
    )
    return res.plot_posterior_map(ax=ax, component=component, colorbar=colorbar)


def read_observations(paths, *, obscodes=None, format: str = "auto",
                      night_gap_hours: float = 3.0, max_arc_hours: float = 2.0):
    """Read astrometry and fit one tracklet per object.

    Parameters
    ----------
    paths : str, path-like, or a list of them
        ADES PSV or MPC 80-column files; the format is detected from the
        content unless ``format`` is ``"ades_psv"`` or ``"mpc80"``.
        Detections are grouped by object across all the files.
    obscodes : str or path-like, optional
        A local copy of the MPC list of observatory codes (default: the one
        ``pneo fetch`` downloaded).
    night_gap_hours, max_arc_hours
        A gap longer than ``night_gap_hours`` splits a night; a night longer
        than ``max_arc_hours`` is cut to that length.

    Returns
    -------
    tracklets : dict of numpy arrays
        One entry per tracklet: ``ra``, ``dec``, ``ra_rate``, ``dec_rate``,
        ``epoch`` (MJD, UTC), ``mag`` and ``obscode`` (the inputs of
        :func:`classify`), plus ``object_id``, ``n_obs``, ``arc_hours`` and
        ``resid_rms_arcsec``.  ``pneo.classify(**tracklets)`` scores them.
    failures : dict of numpy arrays
        One entry per object that could not become a tracklet:
        ``object_id``, ``status`` and ``message``.
    """
    from ._ingest import ObscodeTable, ingest

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    table = ObscodeTable(None if obscodes is None else str(obscodes))
    trks, fails, issues = ingest(
        [str(p) for p in paths], fmt=format, obscodes=table,
        night_gap_hours=night_gap_hours, max_arc_hours=max_arc_hours,
    )
    for issue in issues:
        warnings.warn(f"{issue.source}:{issue.line}: {issue.reason}", stacklevel=2)

    def col(values, dtype):
        return np.array(values, dtype=dtype)

    dec = col([t.dec for t in trks], float)
    tracklets = {
        "object_id": col([t.object_id for t in trks], str),
        "ra": np.degrees(col([t.ra for t in trks], float)),
        "dec": np.degrees(dec),
        "ra_rate": np.degrees(col([t.ra_rate for t in trks], float)) * np.cos(dec),
        "dec_rate": np.degrees(col([t.dec_rate for t in trks], float)),
        "epoch": col([t.epoch_jd_utc for t in trks], float) - 2400000.5,
        "mag": col([t.v_mag for t in trks], float),
        "obscode": col([t.obscode for t in trks], str),
        "n_obs": col([t.n_obs for t in trks], int),
        "arc_hours": col([t.arc_hours for t in trks], float),
        "resid_rms_arcsec": col([t.resid_rms_arcsec for t in trks], float),
    }
    failures = {
        "object_id": col([f.object_id for f in fails], str),
        "status": col([f.status for f in fails], str),
        "message": col([f.message for f in fails], str),
    }
    return tracklets, failures


def fetch(model: bool = True, obscodes: bool = True) -> None:
    """Download the population model and the MPC observatory codes.

    Both go to the per-user cache (see :mod:`pneo.paths`), where
    :func:`load_model`, :func:`classify` and ``pneo classify`` find them.
    """
    from ._fetch_model import fetch as fetch_model
    from ._ingest import fetch_obscodes

    if model:
        fetch_model()
    if obscodes:
        fetch_obscodes()
