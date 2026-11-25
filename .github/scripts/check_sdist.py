#!/usr/bin/env python3
"""Assert that the built sdist is a complete *source* release and nothing more.

Used by .github/workflows/release.yml.  Standard library only.

    python .github/scripts/check_sdist.py <dir-or-tarball> [--version X.Y.Z]

The sdist is the platform fallback: it installs with setuptools alone, compiles
nothing, and gives the numpy reference implementation (``backend="auto"`` then
resolves to ``"python"``).  Two properties therefore have to hold and are
checked here rather than trusted to MANIFEST.in:

* it must carry the *whole* source, the Rust crate included, so that the sdist
  is a real source release and a downstream packager can build the extension
  from it (``maturin build`` via the ``[tool.maturin]`` table, or
  ``rust/build.sh``);
* it must not carry the population model (~4 MB of MPC data fetched at use
  time and not ours to redistribute), any compiled extension, or the
  development trees (``validation/``, ``paper/``, ``.github/``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile

REQUIRED = (
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "PKG-INFO",
    "src/pneo/__init__.py",
    "src/pneo/_backend.py",
    "src/pneo/_classifier.py",
    "src/pneo/_api.py",
    "src/pneo/models/digest2.py",
    "rust/Cargo.toml",
    "rust/Cargo.lock",
    "rust/build.sh",
    "rust/build.rs",
    "rust/src/lib.rs",
    "rust/src/kernel.rs",
    "tests/test_imports.py",
)

FORBIDDEN_PREFIXES = (
    "data/",
    "validation/",
    "paper/",
    ".github/",
    "rust/target/",
)
FORBIDDEN_SUFFIXES = (".so", ".pyd", ".dylib", ".pyc", ".csv", ".npz", ".model")


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="sdist tarball, or a directory holding exactly one")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    if p.is_dir():
        tars = sorted(p.glob("*.tar.gz"))
        if len(tars) != 1:
            fail(f"expected exactly one sdist in {p}, found {[t.name for t in tars]}")
        sdist = tars[0]
    else:
        sdist = p
    if not sdist.is_file():
        fail(f"{sdist} does not exist")

    if args.version and sdist.name != f"pneo-{args.version}.tar.gz":
        fail(f"{sdist.name}: expected pneo-{args.version}.tar.gz")

    print(f"sdist: {sdist.name} ({sdist.stat().st_size / 1e6:.2f} MB)")

    with tarfile.open(sdist) as tf:
        names = tf.getnames()

    roots = {n.split("/", 1)[0] for n in names}
    if len(roots) != 1:
        fail(f"sdist has more than one top-level directory: {sorted(roots)}")
    root = roots.pop()

    rel = sorted(n[len(root) + 1 :] for n in names if n != root)
    for r in rel:
        print(f"    {r}")

    missing = [r for r in REQUIRED if r not in rel]
    if missing:
        fail(f"missing from the sdist: {missing}")

    bad = [
        r
        for r in rel
        if r.startswith(FORBIDDEN_PREFIXES)
        or r.endswith(FORBIDDEN_SUFFIXES)
        or "__pycache__" in r
    ]
    if bad:
        fail(f"must not be in the sdist: {bad}")

    print(f"OK: {len(rel)} entries, crate included, no model data and no build products")
    return 0


if __name__ == "__main__":
    sys.exit(main())
