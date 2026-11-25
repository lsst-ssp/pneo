"""Tests for pneo.paths: where downloaded data files live."""

import sys
from pathlib import Path

import pytest

from pneo import paths


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.cache_dir() == tmp_path / "c"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX layout")
def test_xdg_then_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PNEO_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.cache_dir() == tmp_path / "xdg" / "pneo"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert paths.cache_dir() == tmp_path / "home" / ".cache" / "pneo"


def test_file_paths_live_in_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path))
    assert paths.default_model_path() == tmp_path / "digest2.model.csv"
    assert paths.default_obscodes_path() == tmp_path / "ObsCodesF.html"


def test_resolved_at_call_time_not_import_time(monkeypatch, tmp_path):
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path / "a"))
    first = paths.default_model_path()
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path / "b"))
    assert paths.default_model_path() != first


def test_fetch_model_defaults_to_the_cache(monkeypatch, tmp_path):
    """`pneo-fetch-model` with no --out writes into the cache (network stubbed)."""
    from pneo import _fetch_model as fetch_model

    payload = (fetch_model.EXPECTED_HEADER + "\n").encode()

    class _Resp:
        def read(self):
            return payload

    monkeypatch.setattr(fetch_model.urllib.request, "urlopen",
                        lambda url, timeout: _Resp())
    monkeypatch.setenv("PNEO_CACHE_DIR", str(tmp_path / "cache"))
    out = fetch_model.fetch()
    assert Path(out) == tmp_path / "cache" / "digest2.model.csv"
    assert out.read_bytes() == payload
