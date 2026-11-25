"""Download the MPC digest2 population model.

:func:`fetch` (used by ``pneo fetch`` and :func:`pneo.fetch`) downloads
``digest2.model.csv`` from the MPC's public repository (Git LFS media endpoint), verifies its format and
checksum, and writes it to the per-user cache directory (see
:mod:`pneo.paths`), where ``pneo classify`` finds it.

The mpc-public repository carries no LICENSE file (checked 2026-08-31); the
model is downloaded at use time and never redistributed with pneo.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://media.githubusercontent.com/media/Smithsonian/mpc-public/"
    "main/digest2/population/digest2.model.csv"
)
#: sha256 of the model as of 2026-08-31. The MPC refreshes the model
#: occasionally; a mismatch is a warning (the format may still be fine), not an
#: error.
KNOWN_SHA256 = "2b9d9d2a6d8ab16d3a2dbb835b30d3aa172431f27c3cb7c159675d7686668c45"
EXPECTED_HEADER = "Model,Class,Q,e,i,H6,H8,H10"

from .paths import default_model_path


def fetch(dest: Path | None = None, timeout: float = 120.0) -> Path:
    """Download, verify, and write the model file. Returns the path written.

    ``dest`` defaults to :func:`pneo.paths.default_model_path`.
    """
    if dest is None:
        dest = default_model_path()
    print(f"fetching {URL}")
    data = urllib.request.urlopen(URL, timeout=timeout).read()
    sha = hashlib.sha256(data).hexdigest()
    head = data[:200].decode("ascii", "replace")
    if not head.startswith(EXPECTED_HEADER):
        raise RuntimeError(f"unexpected format; header starts: {head[:80]!r}")
    if sha != KNOWN_SHA256:
        print(
            f"WARNING: sha256 {sha} != pinned {KNOWN_SHA256} "
            "(model refreshed upstream?)",
            file=sys.stderr,
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data)} bytes, sha256 {sha[:12]}...)")
    return dest
