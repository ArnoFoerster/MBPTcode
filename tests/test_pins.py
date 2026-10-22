"""The pin comparison of tests/pins.py: a few ulp under the recording BLAS,
the bar of the pin's quantity class under another, and every pin file names
its BLAS."""
import json
import pathlib

import numpy as np
import pytest

from src.Base.constants import (HARTREE_TO_EV, PIN_CROSS_BLAS_BAR,
                                PIN_CROSS_BLAS_CONTINUED_BAR,
                                PIN_CROSS_BLAS_GRADIENT_BAR,
                                PIN_CROSS_BLAS_ROOT_BAR,
                                PIN_CROSS_BLAS_VECTOR_BAR,
                                PIN_DAVIDSON_ROOT_FACTOR, PIN_SAME_ULPS)
from tests.pins import (PIN_CROSS_BLAS_BARS, align_signs, assert_pin,
                        assert_within_class, blas_identity, class_bar,
                        davidson_root_bar, matches, recorded_blas, same_bar)

REPO = pathlib.Path(__file__).resolve().parent.parent
PIN_FILES = (REPO / 'tests' / 'one_fit_pins.json',
             REPO / 'tests' / 'baseline_forward_routes.json')
#: A total energy of the size the pins hold, Ha.
E_PIN = -76.02784961778849


def another_blas():
    """This process's identity with numpy's BLAS renamed: a pretended
    different machine."""
    return dict(blas_identity(), numpy='pretended other BLAS')


@pytest.mark.parametrize('quantity', sorted(PIN_CROSS_BLAS_BARS))
def test_the_same_blas_holds_to_a_few_ulp(quantity):
    """Two runs on one machine differ by an ulp
    (1.1e-16 Ha); PIN_SAME_ULPS ulp of max(|pin|, 1) pass, twice that
    fails."""
    here = blas_identity()
    unit = '1' if quantity == 'vector' else 'Ha'
    ulp = np.spacing(abs(E_PIN))
    assert matches(E_PIN, E_PIN, here, quantity=quantity)
    assert matches(np.nextafter(E_PIN, 0.0), E_PIN, here, quantity=quantity)
    assert matches(E_PIN + PIN_SAME_ULPS * ulp / 2, E_PIN, here, unit,
                   quantity=quantity)
    assert not matches(E_PIN + 2 * PIN_SAME_ULPS * ulp, E_PIN, here, unit,
                       quantity=quantity)
    grad = np.array([[1e-3, -2e-4, 0.0]])
    eps = np.finfo(float).eps
    assert matches(grad, grad.tolist(), here, quantity=quantity)
    # below one unit the floor is one unit's ulps: 1.1e-16 holds
    assert matches(grad + [[0.0, 0.0, 1.1e-16]], grad, here, unit,
                   quantity=quantity)
    assert not matches(grad + [[0.0, 0.0, 2 * PIN_SAME_ULPS * eps]], grad, here,
                       unit, quantity=quantity)
    assert same_bar(grad, grad) == PIN_SAME_ULPS * eps
    assert not matches(np.nan, np.nan, here, quantity=quantity)


@pytest.mark.parametrize('quantity', sorted(PIN_CROSS_BLAS_BARS))
@pytest.mark.parametrize('unit, scale', [('Ha', 1.0), ('eV', HARTREE_TO_EV)])
def test_another_blas_holds_to_the_bar_of_its_class(quantity, unit, scale):
    other = another_blas()
    bar = PIN_CROSS_BLAS_BARS[quantity] * scale
    pin = E_PIN * scale
    assert matches(pin + 0.75 * bar, pin, other, unit, quantity=quantity)
    assert matches(pin - 0.75 * bar, pin, other, unit, quantity=quantity)
    assert not matches(pin + 1.5 * bar, pin, other, unit, quantity=quantity)
    assert not matches([pin, pin + 1.5 * bar], [pin, pin], other, unit,
                       quantity=quantity)


def test_the_classes_are_the_ruled_bars():
    assert PIN_CROSS_BLAS_BARS == {'energy': PIN_CROSS_BLAS_BAR,
                                   'gradient': PIN_CROSS_BLAS_GRADIENT_BAR,
                                   'continued': PIN_CROSS_BLAS_CONTINUED_BAR,
                                   'vector': PIN_CROSS_BLAS_VECTOR_BAR,
                                   'root': PIN_CROSS_BLAS_ROOT_BAR}
    # a gradient's bar admits a cluster's measured 2.2e-9 Ha/Bohr miss, which the
    # energy's would refuse
    other = another_blas()
    grad = np.array([[1e-3, -2e-4, 0.0]])
    moved = grad + [[2.2e-9, 0.0, 0.0]]
    assert matches(moved, grad, other, quantity='gradient')
    assert not matches(moved, grad, other, quantity='energy')


def test_a_pin_without_a_class_is_an_energy():
    other = another_blas()
    assert matches(E_PIN + 1.5e-12, E_PIN, other)
    assert not matches(E_PIN + 3e-12, E_PIN, other)
    assert PIN_CROSS_BLAS_BAR == 2e-12


def test_a_shape_mismatch_never_matches():
    assert not matches([1.0, 2.0], [1.0], blas_identity(), quantity='energy')
    assert not matches([1.0, 2.0], [1.0], another_blas(), quantity='energy')
    with pytest.raises(AssertionError, match=r'shape \(4, 3\)'):
        assert_pin(np.zeros((4, 3)), np.zeros((3, 3)), another_blas(),
                   quantity='gradient', label='formaldehyde')


def test_a_miss_names_its_size():
    with pytest.raises(AssertionError, match=r'max \|got - pin\| 3\.000e-08'):
        assert_pin(np.zeros((4, 3)) + 3e-8, np.zeros((4, 3)), another_blas(),
                   quantity='gradient', label='formaldehyde')


@pytest.mark.parametrize('path', PIN_FILES, ids=lambda p: p.name)
def test_every_pin_file_names_its_blas(path):
    blas = recorded_blas(path)
    assert set(blas) == set(blas_identity())
    assert all(isinstance(v, str) and v for v in blas.values())


def test_a_pin_file_without_a_blas_is_refused(tmp_path):
    path = tmp_path / 'pins.json'
    path.write_text(json.dumps({'water': {'e': E_PIN}}))
    with pytest.raises(KeyError, match='records no BLAS'):
        recorded_blas(path)


def test_a_vector_is_held_after_its_signs_are_aligned():
    other = another_blas()
    x = np.array([[0.6, -0.8], [0.8, 0.6]])
    flipped = x * [-1.0, 1.0]
    assert np.array_equal(align_signs(flipped, x), x)
    for blas in (blas_identity(), other):
        assert matches(flipped, x, blas, '1', quantity='vector')
    moved = flipped + 5e-11
    assert matches(moved, x, other, '1', quantity='vector')
    assert not matches(flipped + 2e-10, x, other, '1', quantity='vector')
    assert not matches(flipped, x, other, '1', quantity='energy')


def test_a_class_bar_holds_under_any_blas():
    assert class_bar('root') == PIN_CROSS_BLAS_ROOT_BAR
    bar = PIN_CROSS_BLAS_ROOT_BAR
    assert_within_class([1.0 + 0.9 * bar], [1.0], 'root')
    with pytest.raises(AssertionError):
        assert_within_class([1.0 + 1.1 * bar], [1.0], 'root')


def test_the_rank_split_bar_follows_the_davidson_tolerance():
    assert davidson_root_bar(1e-8) == PIN_DAVIDSON_ROOT_FACTOR * 1e-8
    assert davidson_root_bar(1e-5) == 1e3 * davidson_root_bar(1e-8)
