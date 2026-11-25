#!/usr/bin/env python3
"""Assert that the sdist and every wheel describe the same distribution.

    python .github/scripts/check_metadata.py <dir with the sdist and the wheels>

pneo builds its sdist with setuptools and its wheels with maturin (see
pyproject.toml).  Both read the same ``[project]`` table, so they *cannot*
disagree -- this check is what makes that a fact rather than a belief.  It is
the guard against the one real hazard of the two-tool split: a future
`[project]` key that one tool honours and the other ignores, shipped as an sdist
and a wheel that promise different dependencies under the same version.

Standard library only, and no pneo import.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import pathlib
import re
import sys
import tarfile
import zipfile

# Compared field by field.  Everything the two tools emit is compared except:
#   Description-Content-Type -- maturin says "text/markdown; charset=UTF-8;
#     variant=GFM" where setuptools says "text/markdown".  Same README, same
#     renderer on PyPI; a cosmetic difference in how the tools spell it.
#   Dynamic -- only meaningful in an sdist, where it tells a builder which
#     fields may change when the wheel is built.  setuptools marks
#     "license-file" Dynamic there even though the License-File value is
#     fixed; maturin's wheels carry no Dynamic.  The fields themselves,
#     License-File included, are still compared.
IGNORED = {"description-content-type", "dynamic"}


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def normalize(key: str, values: list[str]) -> list[str]:
    """Canonical form of a repeated metadata field.

    Requires-Dist environment markers differ only in quoting and whitespace
    between the two tools (``pytest>=7 ; extra == 'dev'`` vs
    ``pytest>=7; extra == "dev"``), which is not a difference in what is
    required.
    """
    out = []
    for v in values:
        v = v.strip()
        if key == "requires-dist":
            v = re.sub(r"\s+", "", v).replace('"', "'")
        out.append(v)
    return sorted(out)


def fields(text: str) -> dict[str, list[str]]:
    msg = email.message_from_string(text, policy=email.policy.compat32)
    keys = {k.lower() for k in msg.keys()} - IGNORED
    return {k: normalize(k, msg.get_all(k) or []) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory holding one sdist and one or more wheels")
    args = ap.parse_args()

    d = pathlib.Path(args.path)
    tars = sorted(d.glob("*.tar.gz"))
    whls = sorted(d.glob("*.whl"))
    if len(tars) != 1:
        fail(f"expected exactly one sdist in {d}, found {[t.name for t in tars]}")
    if not whls:
        fail(f"no wheels in {d}")

    with tarfile.open(tars[0]) as tf:
        pkginfo = [n for n in tf.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(pkginfo) != 1:
            fail(f"{tars[0].name}: expected one top-level PKG-INFO, got {pkginfo}")
        member = tf.extractfile(pkginfo[0])
        assert member is not None
        ref = fields(member.read().decode("utf-8"))
    print(f"reference: {tars[0].name} ({len(ref)} metadata fields)")

    ok = True
    for whl in whls:
        with zipfile.ZipFile(whl) as z:
            meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
            if len(meta) != 1:
                fail(f"{whl.name}: expected one dist-info/METADATA, got {meta}")
            got = fields(z.read(meta[0]).decode("utf-8"))
        diff = []
        for key in sorted(set(ref) | set(got)):
            a, b = ref.get(key, []), got.get(key, [])
            if a != b:
                diff.append(f"    {key}: sdist={a} wheel={b}")
        if diff:
            ok = False
            print(f"::error::{whl.name} disagrees with {tars[0].name}:")
            for line in diff:
                print(line)
        else:
            print(f"OK: {whl.name} matches")

    if not ok:
        return 1
    print(f"OK: {len(whls)} wheel(s) and the sdist describe the same distribution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
