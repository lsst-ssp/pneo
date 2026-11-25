#!/usr/bin/env python3
"""Assert that a built pneo wheel is exactly what we mean to publish.

Used by .github/workflows/release.yml right after each ``maturin build``, so a
packaging regression fails the wheel job instead of reaching PyPI.  Runs on the
stock interpreter: standard library only, no pneo import (the wheel is not
installed at that point, and on the linux leg it is not even built for the
runner's Python).

    python .github/scripts/check_wheel.py <dir-or-wheel> [--plat-tag FRAGMENT]
                                          [--version X.Y.Z]

What is checked
    * exactly one wheel in the directory;
    * the wheel is tagged ``cp311-abi3`` -- the abi3 promise, i.e. one wheel
      per platform covers CPython >= 3.11.  A cp3XX-cp3XX wheel here would
      mean the crate lost its ``abi3-py311`` feature and every future
      interpreter release would need a new wheel;
    * ``--plat-tag`` (e.g. ``manylinux2014_x86_64``) appears in the filename,
      so a silently-degraded platform tag (a plain ``linux_x86_64`` wheel PyPI
      would reject, or a macOS deployment target drift) is caught here;
    * the compiled extension is present as ``pneo/_core.<abi3 suffix>``;
    * every other member is a ``pneo/*.py`` module or a ``dist-info``
      entry.  This is the check that matters: it fails on the two things this
      layout can get wrong -- a wheel that claims the whole ``pneo``
      package but forgets the Python modules, and a wheel that ships a
      developer's leftovers (``__pycache__``, a stale ``.so``,
      the population model, ``tests/``, ``validation/``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

# The population model is fetched at use time and is not ours to redistribute;
# these are the paths whose presence would mean the build picked up the repo
# working tree instead of the package.
FORBIDDEN_SUFFIXES = (".csv", ".pyc", ".pyo", ".npz", ".ipynb")
FORBIDDEN_PREFIXES = ("data/", "tests/", "validation/", "paper/", "rust/", "scripts/")


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="wheel file, or a directory holding exactly one")
    ap.add_argument("--plat-tag", default=None, help="fragment required in the wheel tag")
    ap.add_argument("--version", default=None, help="expected distribution version")
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    if p.is_dir():
        wheels = sorted(p.glob("*.whl"))
        if len(wheels) != 1:
            fail(f"expected exactly one wheel in {p}, found {[w.name for w in wheels]}")
        whl = wheels[0]
    else:
        whl = p
    if not whl.is_file():
        fail(f"{whl} does not exist")

    print(f"wheel: {whl.name} ({whl.stat().st_size / 1e6:.2f} MB)")

    name = whl.name
    if not name.startswith("pneo-"):
        fail(f"{name}: not a pneo wheel")
    if args.version and not name.startswith(f"pneo-{args.version}-"):
        fail(f"{name}: expected version {args.version}")
    if "-cp311-abi3-" not in name:
        fail(
            f"{name}: not a cp311-abi3 wheel -- the pyo3 'abi3-py311' feature is "
            "what lets one wheel serve every CPython >= 3.11"
        )
    if args.plat_tag and args.plat_tag not in name:
        fail(f"{name}: platform tag does not contain {args.plat_tag!r}")

    with zipfile.ZipFile(whl) as z:
        members = z.namelist()
    for m in members:
        print(f"    {m}")

    ext = [
        m
        for m in members
        if m.startswith("pneo/_core.")
        and m.endswith((".so", ".pyd", ".dylib"))
    ]
    if len(ext) != 1:
        fail(f"expected exactly one pneo/_core extension module, found {ext}")

    py = [m for m in members if m.startswith("pneo/") and m.endswith(".py")]
    if "pneo/__init__.py" not in py:
        fail("pneo/__init__.py is missing: the wheel carries no Python package")
    for required in ("_backend.py", "_classifier.py", "_api.py", "models/digest2.py"):
        if f"pneo/{required}" not in py:
            fail(f"pneo/{required} is missing from the wheel")

    stray = []
    for m in members:
        if m in ext or m in py:
            continue
        if m.endswith("/"):  # directory entry
            continue
        if m.startswith("pneo-") and ".dist-info/" in m:
            continue
        stray.append(m)
    if stray:
        fail(f"unexpected members in the wheel: {stray}")

    for m in members:
        if m.startswith(FORBIDDEN_PREFIXES) or m.endswith(FORBIDDEN_SUFFIXES):
            fail(f"{m}: must not be in the wheel")
        if "__pycache__" in m:
            fail(f"{m}: build picked up bytecode from the working tree")

    print(
        f"OK: {len(py)} Python modules + {ext[0]} "
        f"({len(members)} members total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
