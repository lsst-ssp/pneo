"""WP0 smoke test: the package imports and the contracts exist."""

import ds42


def test_imports():
    assert ds42.__version__
    assert ds42.Attributable is not None
    assert ds42.PopulationDensity is not None


def test_constants():
    from ds42 import constants

    assert abs(constants.MU_SUN - constants.K_GAUSS**2) < 1e-18
    assert constants.Q_NEO == 1.3
