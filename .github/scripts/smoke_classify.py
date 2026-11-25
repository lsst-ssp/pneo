#!/usr/bin/env python3
"""Smoke-test an *installed* pneo distribution, from outside the repo.

    python .github/scripts/smoke_classify.py rust     # a wheel: kernel present
    python .github/scripts/smoke_classify.py python   # the sdist: no kernel

Used by .github/workflows/release.yml from a clean virtualenv with a working
directory outside the checkout, so nothing it asserts can be satisfied by the
source tree.  Deliberately self-contained: it builds a toy population pair in
memory and passes the observer state explicitly, so it needs neither
the digest2 population model nor an ephemeris download.

``rust`` mode (the wheels) asserts, in order of what it would catch:

1. ``pneo._core`` imports from the installed package, ``core_available()``
   is true and ``backend="auto"`` resolves to ``"rust"`` -- i.e. the wheel
   really carries the extension and the module landed at the name
   ``pneo._core``, which is the whole point of the mixed maturin layout (a
   wheel naming it ``pneo_core`` would import fine as a top-level module
   and still fail here);
2. the extension and the Python package come from the same release
   (``_core.__version__ == pneo.__version__``);
3. the shipped kernel and the shipped numpy reference produce the same
   posterior on the same toy tracklet at a coarse grid.  A wheel built against
   the wrong crate, or with a mismatched model layout, does not survive this.

``python`` mode (the sdist) asserts the other half of the contract: no
extension was built or shipped, ``backend="auto"`` degrades to the reference
rather than failing, an explicit ``backend="rust"`` raises, and the reference
still classifies the same toy tracklet.
"""

from __future__ import annotations

import os
import sys

# The rust-mode A/B below has to be a real A/B: `PNEO_BACKEND` names a
# *concrete* backend and overrides the `backend=` argument (see
# pneo.backend), so a `PNEO_BACKEND=rust` inherited from the job
# environment would route the "python" call through the kernel as well and the
# comparison would be vacuous.  Drop it here -- the workflow sets it for the
# pytest subset, which is where it belongs.  tests/test_backend_parity.py does
# the same thing for the same reason.
_forced = os.environ.pop("PNEO_BACKEND", None)

import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

import pneo  # noqa: E402
from pneo import Attributable, classify  # noqa: E402
from pneo.backend import core_available, resolve_backend  # noqa: E402
from pneo.models.digest2 import Digest2Component  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "rust"
if mode not in ("rust", "python"):
    sys.exit(f"usage: {sys.argv[0]} [rust|python]")

print(f"mode         {mode}")
print(f"python       {sys.version.split()[0]}")
print(f"numpy        {np.__version__}")
print(f"pneo     {pneo.__version__} from {pneo.__file__}")
print(f"PNEO_BACKEND in the environment: {_forced!r} (ignored here)")

# --- the extension: present and named right, or absent and degraded ---------
if mode == "rust":
    import pneo._core as core

    print(f"pneo._core {core.__version__} from {core.__file__}")
    assert core_available(), "pneo._core did not import from the installed wheel"
    assert resolve_backend("auto") == "rust", resolve_backend("auto")
    assert resolve_backend("rust") == "rust"
    assert core.__version__ == pneo.__version__, (
        f"extension version {core.__version__} != package version "
        f"{pneo.__version__}"
    )
else:
    assert not core_available(), (
        "a pneo._core extension is importable from an install that must "
        "not have one"
    )
    assert resolve_backend("auto") == "python", resolve_backend("auto")
    try:
        resolve_backend("rust")
    except RuntimeError as exc:
        print(f'backend="rust" correctly refused: {exc}')
    else:
        raise AssertionError('backend="rust" did not raise without the extension')

# --- a toy classification --------------------------------------------------
# A toy (q, e, i, H) pair on the digest2 bin structure: the NEO component is
# concentrated in the low-q bins, "other" in the rest, so both marginal
# likelihoods are non-zero for an ordinary tracklet.
rng = np.random.default_rng(42)
shape = (29, 8, 11, 18)
neo_counts = np.zeros(shape)
neo_counts[:6] = rng.random((6,) + shape[1:]) * 1e3
other_counts = np.zeros(shape)
other_counts[4:] = rng.random((shape[0] - 4,) + shape[1:]) * 1e5
neo = Digest2Component(neo_counts, "NEO")
other = Digest2Component(other_counts, "other")

att = Attributable(
    ra=1.9,
    dec=0.15,
    ra_rate=0.006,
    dec_rate=0.002,
    epoch=Time(2460000.5, format="jd", scale="tdb"),
    v_mag=20.5,
)
R_obs = np.array([0.4, -0.85, -0.37])
V_obs = np.array([0.0155, 0.0062, 0.0027])
common = dict(R_obs=R_obs, V_obs=V_obs, n_rho=120, n_rhodot=24)


def report(label, res):
    print(
        f"{label:7s} p_neo={res.p_neo!r} lambda_neo={res.lambda_neo!r} "
        f"lambda_other={res.lambda_other!r} n_nodes={res.n_nodes} "
        f"n_masked_det={res.n_masked_det}"
    )


res_py = classify(att, neo, other, backend="python", **common)
report("python", res_py)
assert res_py.n_nodes > 0, "empty admissible region: the toy tracklet is wrong"
assert np.isfinite(res_py.p_neo) and 0.0 <= res_py.p_neo <= 1.0
assert res_py.lambda_neo > 0.0 and res_py.lambda_other > 0.0

if mode == "rust":
    res_rust = classify(att, neo, other, backend="rust", **common)
    report("rust", res_rust)
    assert res_rust.n_nodes == res_py.n_nodes
    assert res_rust.n_masked_det == res_py.n_masked_det
    assert np.isfinite(res_rust.p_neo) and 0.0 <= res_rust.p_neo <= 1.0
    for name in ("lambda_neo", "lambda_other"):
        a, b = getattr(res_rust, name), getattr(res_py, name)
        assert a > 0.0, f"{name} is not positive on the rust backend: {a}"
        assert abs(a - b) <= 1e-8 * abs(b), f"{name}: rust {a!r} vs python {b!r}"
    assert abs(res_rust.p_neo - res_py.p_neo) <= 1e-9, (res_rust.p_neo, res_py.p_neo)
    print("OK: the installed wheel's kernel agrees with its numpy reference")
else:
    print("OK: the installed sdist runs the numpy reference and says so")
