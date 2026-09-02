#!/usr/bin/env python3
"""Fetch the MPC digest2 population model into data/ (repo convenience wrapper).

The implementation lives in digest42.fetch_model (installed as the
``digest42-fetch-model`` console script); this wrapper keeps the documented
repo-relative default working from a source checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digest42.fetch_model import DEFAULT_DEST, main  # noqa: E402

if __name__ == "__main__":
    dest = Path(__file__).resolve().parent.parent / DEFAULT_DEST
    sys.exit(main(["--out", str(dest)]))
