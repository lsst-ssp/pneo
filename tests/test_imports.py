"""WP0 smoke test: the package imports and the contracts exist."""

import digest42


def test_imports():
    assert digest42.__version__
    assert digest42.Attributable is not None
    assert digest42.PopulationDensity is not None


def test_constants():
    from digest42 import constants

    assert abs(constants.MU_SUN - constants.K_GAUSS**2) < 1e-18
    assert constants.Q_NEO == 1.3
