#!/usr/bin/env python3
"""Benchmark the two ds42 classify backends.

Reports, at the production grid resolution (n_rho, n_rhodot) = (1200, 50):

* single-core seconds/tracklet for ``backend="python"`` and ``backend="rust"``,
  with and without per-node diagnostics;
* Rust batch throughput (tracklets/s) through :func:`classify_many` at a few
  rayon worker counts.

Tracklets come from the stored gate_v0 catalogue
(``validation/output/gate_v0/gen.npz``) together with its observer states, so no
astropy/ephemeris time is included in the timings — this measures the kernel.

Usage:
    python3 rust/bench.py [--n 50] [--n-rho 1200] [--n-rhodot 50]
                          [--workers 1,8,32] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from astropy.time import Time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from ds42.attributable import Attributable  # noqa: E402
from ds42.backend import core_available  # noqa: E402
from ds42.classifier import classify, classify_many  # noqa: E402
from ds42.population import Digest2Histogram  # noqa: E402

GEN = os.path.join(REPO, "validation", "output", "gate_v0", "gen.npz")
MODEL = os.path.join(REPO, "data", "digest2.model.csv")


def load(n: int):
    g = np.load(GEN)
    idx = np.arange(min(n, len(g["ra"])))
    atts = [
        Attributable(
            ra=g["ra"][k],
            dec=g["dec"][k],
            ra_rate=g["ra_rate"][k],
            dec_rate=g["dec_rate"][k],
            epoch=Time(g["epoch_jd_tdb"][k], format="jd", scale="tdb"),
            v_mag=g["v_mag"][k],
        )
        for k in idx
    ]
    return atts, g["R_obs"][idx], g["V_obs"][idx]


def time_single(atts, R, V, neo, other, backend, diagnostics, cfg):
    t0 = time.perf_counter()
    for att, r, v in zip(atts, R, V):
        classify(
            att, neo, other, r, v, backend=backend,
            diagnostics=diagnostics, **cfg
        )
    return (time.perf_counter() - t0) / len(atts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-rho", type=int, default=1200)
    ap.add_argument("--n-rhodot", type=int, default=50)
    ap.add_argument("--workers", default="1,8,32")
    ap.add_argument("--batch", type=int, default=None,
                    help="tracklets for the batch runs (default: 8x --n)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = dict(n_rho=args.n_rho, n_rhodot=args.n_rhodot)
    neo, other = Digest2Histogram.neo_other_pair(MODEL, "All")
    atts, R, V = load(args.n)
    print(f"grid = ({args.n_rho}, {args.n_rhodot}); {len(atts)} tracklets; "
          f"rust available = {core_available()}")

    out: dict = {"n_rho": args.n_rho, "n_rhodot": args.n_rhodot,
                 "n_tracklets": len(atts)}

    warnings.simplefilter("ignore", RuntimeWarning)

    # Warm-up (model adaptation, first-touch pages).
    classify(atts[0], neo, other, R[0], V[0], backend="rust", **cfg)

    print("\n-- single core, s/tracklet --")
    for backend in (["python", "rust"] if core_available() else ["python"]):
        for diag in (True, False):
            t = time_single(atts, R, V, neo, other, backend, diag, cfg)
            key = f"single_{backend}_diag{int(diag)}"
            out[key] = t
            print(f"  {backend:6s} diagnostics={str(diag):5s}  {t*1e3:9.2f} ms"
                  f"   ({1.0/t:8.1f} tracklet/s)")
    if "single_python_diag1" in out and "single_rust_diag1" in out:
        out["speedup_diag"] = out["single_python_diag1"] / out["single_rust_diag1"]
        out["speedup_nodiag"] = out["single_python_diag1"] / out["single_rust_diag0"]
        print(f"  speedup (diagnostics on)  x{out['speedup_diag']:.1f}")
        print(f"  speedup (diagnostics off) x{out['speedup_nodiag']:.1f}")

    if core_available():
        nb = args.batch or 8 * args.n
        batts, bR, bV = load(nb)
        print(f"\n-- rust batch (classify_many), {len(batts)} tracklets, "
              "diagnostics off --")
        out["batch_n"] = len(batts)
        for w in [int(x) for x in args.workers.split(",")]:
            t0 = time.perf_counter()
            classify_many(
                batts, neo, other, R_obs=bR, V_obs=bV, backend="rust",
                diagnostics=False, n_threads=w, **cfg
            )
            dt = time.perf_counter() - t0
            out[f"batch_{w}w_s_per_tracklet"] = dt / len(batts)
            out[f"batch_{w}w_throughput"] = len(batts) / dt
            print(f"  {w:3d} worker(s): {dt:7.3f} s total, "
                  f"{dt/len(batts)*1e3:8.3f} ms/tracklet, "
                  f"{len(batts)/dt:9.1f} tracklet/s")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
