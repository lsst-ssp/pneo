"""WP0 smoke test: the package imports and the contracts exist."""

import pneo


def test_imports():
    assert pneo.__version__
    for name in ("load_model", "classify", "read_observations", "posterior_map",
                 "plot_posterior_map", "fetch", "PopulationModel",
                 "PopulationComponent"):
        assert getattr(pneo, name) is not None, name


def test_constants():
    from pneo import _constants as constants

    assert abs(constants.MU_SUN - constants.K_GAUSS**2) < 1e-18
    assert constants.Q_NEO == 1.3
