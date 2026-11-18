"""Public SI/deck conventions and version identity of the distribution."""
import pytest

from fftpf import units


def test_public_si_conversions():
    assert units.si_to_deck_landau(1e9) == pytest.approx(1.0)
    assert units.si_to_deck_elastic(1e9) == pytest.approx(1.0)
    assert units.si_to_deck_gradient(7.04e-11) == pytest.approx(0.0704)
    assert units.si_to_deck_field(1e5) == pytest.approx(1e-4)
    assert units.deck_to_si_field(1e-4) == pytest.approx(1e5)
    assert units.relative_to_deck_permittivity(10) == pytest.approx(0.088541878128)
    assert units.deck_to_si_permittivity(0.088541878128) == pytest.approx(
        10 * units.EPSILON_0_SI, abs=0, rel=1e-12
    )
    assert units.kT_langevin(300, 1, 1, 2) == pytest.approx(0.0020709735)
