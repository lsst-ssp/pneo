"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path_factory, monkeypatch):
    """Point pneo's per-user cache at an empty directory for every test.

    A developer's real cache (``~/.cache/pneo``) may hold a population model or
    an observatory-code table; without this, lookups that fall through to the
    cache would behave differently on different machines.  Tests that need a
    populated cache set ``PNEO_CACHE_DIR`` themselves.
    """
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path_factory.mktemp("pneo-cache")))
