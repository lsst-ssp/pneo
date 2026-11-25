"""Where pneo keeps the data files it downloads.

pneo fetches two inputs at the user's request and never ships them:
the MPC digest2 population model (``pneo fetch``) and the MPC
observatory-code table (``pneo fetch``).  Both are written to, and
looked up in, a per-user cache directory, so an installed pneo finds them
no matter what the current working directory is.

The directory is, in order of precedence:

1. ``$PNEO_CACHE_DIR``, if set;
2. ``%LOCALAPPDATA%\\pneo\\Cache`` on Windows;
3. ``$XDG_CACHE_HOME/pneo``, if ``XDG_CACHE_HOME`` is set;
4. ``~/.cache/pneo`` otherwise (Linux and macOS).

Everything is resolved when called, not at import time, so changing the
environment (or ``HOME``) takes effect immediately.  Nothing here creates the
directory; the fetchers do that when they write.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "CACHE_ENV_VAR",
    "MODEL_FILENAME",
    "OBSCODES_FILENAME",
    "cache_dir",
    "default_model_path",
    "default_obscodes_path",
]

#: Environment variable overriding the cache directory.
CACHE_ENV_VAR = "PNEO_CACHE_DIR"

#: File name of the digest2 population model inside the cache.
MODEL_FILENAME = "digest2.model.csv"

#: File name of the MPC observatory-code table inside the cache.
OBSCODES_FILENAME = "ObsCodesF.html"


def cache_dir() -> Path:
    """The per-user directory pneo downloads its data files into."""
    env = os.environ.get(CACHE_ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if base:
            return Path(base) / "pneo" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "pneo"
    return Path.home() / ".cache" / "pneo"


def default_model_path() -> Path:
    """Where ``pneo fetch`` writes the model and ``pneo classify`` finds it."""
    return cache_dir() / MODEL_FILENAME


def default_obscodes_path() -> Path:
    """Where ``pneo fetch`` writes the observatory-code table."""
    return cache_dir() / OBSCODES_FILENAME
