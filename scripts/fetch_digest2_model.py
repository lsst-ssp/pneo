#!/usr/bin/env python3
"""Fetch the MPC digest2 population model into data/ (WP0; see data/README.md).

The file lives in github.com/Smithsonian/mpc-public under Git LFS, so the raw
URL serves a pointer; the media endpoint below serves the content.

Note: the mpc-public repository carries no LICENSE file (checked 2026-08-31).
We download at use time and do not redistribute the file in this repo.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://media.githubusercontent.com/media/Smithsonian/mpc-public/"
    "main/digest2/population/digest2.model.csv"
)
# sha256 of the model as of 2026-08-31; the MPC refreshes the model
# occasionally, so a mismatch is a warning (format may still be fine), not an
# error.
KNOWN_SHA256 = "2b9d9d2a6d8ab16d3a2dbb835b30d3aa172431f27c3cb7c159675d7686668c45"
EXPECTED_HEADER = "Model,Class,Q,e,i,H6,H8,H10"

DEST = Path(__file__).resolve().parent.parent / "data" / "digest2.model.csv"


def main() -> int:
    print(f"fetching {URL}")
    data = urllib.request.urlopen(URL, timeout=120).read()
    sha = hashlib.sha256(data).hexdigest()
    text_head = data[:200].decode("ascii", "replace")
    if not text_head.startswith(EXPECTED_HEADER):
        print(f"ERROR: unexpected format; header starts: {text_head[:80]!r}")
        return 1
    if sha != KNOWN_SHA256:
        print(f"WARNING: sha256 {sha} != pinned {KNOWN_SHA256} (model refreshed upstream?)")
    DEST.parent.mkdir(exist_ok=True)
    DEST.write_bytes(data)
    print(f"wrote {DEST} ({len(data)} bytes, sha256 {sha[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
