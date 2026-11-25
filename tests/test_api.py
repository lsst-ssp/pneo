"""Tests for the public Python API (pneo.classify and friends)."""

import numpy as np
import pytest
from astropy.time import Time

import pneo
from model_location import MODEL_PATH, needs_model
from test_cli import _attributable_from_ranging, _write_psv

pytestmark = needs_model

FAST = dict(n_rho=200, n_rhodot=20)


@pytest.fixture(scope="module")
def model():
    return pneo.load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def neo_tracklet():
    """An NEO-like tracklet in public units: deg, deg/day (on-sky), MJD, V."""
    o = _attributable_from_ranging(2.90, 0.10, 2.0, 0.7, 0.10, 0.0, 24.0)
    return dict(ra=np.degrees(o["ra"]), dec=np.degrees(o["dec"]),
                ra_rate=np.degrees(o["ra_rate"]) * np.cos(o["dec"]),
                dec_rate=np.degrees(o["dec_rate"]),
                epoch=Time("2026-03-15T06:00:00", scale="utc").mjd,
                mag=o["v_mag"], obscode="X05")


def test_load_model_defaults_to_the_paper_configuration(model):
    assert isinstance(model, pneo.PopulationModel)
    assert model.source == "digest2"
    assert "All" in model.description and "30" in model.description
    assert isinstance(model.neo, pneo.PopulationComponent)


def test_load_model_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="unknown model source"):
        pneo.load_model(MODEL_PATH, source="neomod3")


def test_scalar_in_float_out(model, neo_tracklet):
    p = pneo.classify(**neo_tracklet, model=model, **FAST)
    assert isinstance(p, float)
    assert p > 0.9


def test_arrays_broadcast_and_keep_their_shape(model, neo_tracklet):
    t = dict(neo_tracklet)
    t["mag"] = np.array([[20.0, 21.0], [22.0, 23.0]])
    p = pneo.classify(**t, model=model, **FAST)
    assert p.shape == (2, 2)
    assert np.all((p >= 0) & (p <= 1))


def test_details_is_a_structured_array(model, neo_tracklet):
    r = pneo.classify(**neo_tracklet, model=model, details=True, **FAST)
    assert r.dtype.names == ("p_neo", "log_lr", "lambda_neo", "lambda_other", "status")
    assert r["status"] == "ok"
    assert r["p_neo"] == pytest.approx(r["lambda_neo"] / (r["lambda_neo"] + r["lambda_other"]))


def test_backends_agree(model, neo_tracklet):
    from pneo._backend import core_available

    if not core_available():
        pytest.skip("pneo._core not built")
    a = pneo.classify(**neo_tracklet, model=model, backend="python", **FAST)
    b = pneo.classify(**neo_tracklet, model=model, backend="rust", **FAST)
    assert a == pytest.approx(b, abs=1e-6)


def test_observer_state_is_equivalent_to_its_obscode(model, neo_tracklet):
    from pneo._attributable import observer_state
    from pneo._ingest import ObscodeTable

    lon, lat, h, _ = ObscodeTable().resolve("X05")
    R, V = observer_state(Time(neo_tracklet["epoch"], format="mjd", scale="utc"),
                          lon_deg=lon, lat_deg=lat, height_m=h)
    a = pneo.classify(**neo_tracklet, model=model, **FAST)
    t = {k: v for k, v in neo_tracklet.items() if k != "obscode"}
    b = pneo.classify(**t, model=model, observer=(R, V), **FAST)
    assert a == pytest.approx(b, abs=1e-12)
    with pytest.raises(ValueError, match="not both"):
        pneo.classify(**neo_tracklet, model=model, observer=(R, V))


def test_unknown_keywords_are_rejected(model, neo_tracklet):
    with pytest.raises(TypeError, match="ra_rat"):
        pneo.classify(**neo_tracklet, model=model, ra_rat=1.0)


def test_read_observations_feeds_classify(model, tmp_path):
    geometry = _attributable_from_ranging(2.90, 0.10, 2.0, 0.7, 0.10, 0.0, 24.0)
    path = _write_psv(tmp_path / "obs.psv", [
        ("T1", geometry, 4, 15.0, "V", "X05"),
        ("T2", geometry, 1, 0.0, "V", "X05"),
    ])
    trk, fail = pneo.read_observations(path)
    assert list(trk["object_id"]) == ["T1"]
    assert list(fail["object_id"]) == ["T2"]
    assert set(trk) >= {"ra", "dec", "ra_rate", "dec_rate", "epoch", "mag", "obscode"}
    p = pneo.classify(**trk, model=model, **FAST)
    assert p.shape == (1,)
    assert p[0] > 0.9


def test_posterior_map_and_plot(model, neo_tracklet):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    m = pneo.posterior_map(**neo_tracklet, model=model, **FAST)
    for key in ("rho", "rho_dot", "weight", "density_neo", "density_other", "H", "q"):
        assert m[key].shape == m["rho"].shape, key
    mass = (m["density_neo"] + m["density_other"]) * m["weight"]
    neo_share = (m["density_neo"] * m["weight"]).sum() / mass.sum()
    assert neo_share == pytest.approx(m["p_neo"], rel=1e-9)
    ax = pneo.plot_posterior_map(m)
    assert ax is not None
