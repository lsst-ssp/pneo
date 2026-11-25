"""WP0 smoke test: the package imports and the contracts exist."""

import pneo


def test_imports():
    assert pneo.__version__
    assert pneo.Attributable is not None
    assert pneo.PopulationDensity is not None


def test_constants():
    from pneo import constants

    assert abs(constants.MU_SUN - constants.K_GAUSS**2) < 1e-18
    assert constants.Q_NEO == 1.3
