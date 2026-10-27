"""Equality of one quantity from two independent computations.

Two SCF solves, two processes or two call shapes of the same contraction
agree to a few ulp on the BLAS the pin files record (`same_bar` of
tests/pins.py: two runs of one code on one machine are not bitwise);
another BLAS rounds each of them differently in its last
bits. Under the recording BLAS the two numbers are compared to
PIN_SAME_ULPS ulp; under another one they are compared within the cross-BLAS
bar of their quantity class. Use this only where the claim is that two
computations give the SAME quantity, never to loosen a physics bar, and never
for a value against a stored copy of itself (that stays `==`).
"""
import pathlib

import numpy as np

from tests.pins import (PIN_CROSS_BLAS_BARS, align_signs, assert_pin,
                        blas_identity, matches, recorded_blas, same_bar)

PIN_FILE = pathlib.Path(__file__).resolve().parent / 'one_fit_pins.json'


def same(a, b, unit='Ha', quantity='energy'):
    """Whether `a` and `b` are one quantity: a few ulp on the recording BLAS,
    within the class bar elsewhere."""
    return matches(a, b, recorded_blas(PIN_FILE), unit, quantity=quantity)


def assert_same(a, b, unit='Ha', quantity='energy', label=None):
    """Raise unless `same`, naming `label` and the miss."""
    assert_pin(a, b, recorded_blas(PIN_FILE), unit, quantity=quantity,
               label=label)


def same_solution(a, b):
    """Whether two eigen-solutions (tuples of arrays) are one: a few ulp on
    the recording BLAS (each 2-D array after fixing each column's sign, unit
    1); elsewhere each 1-D array (roots, energies) within the
    energy class bar and each 2-D array (vectors, matrices) within the
    continued class bar after fixing each column's sign, which an eigenvector
    does not carry (an eigenvector moves by the energy error over the gap to
    its neighbour)."""
    if recorded_blas(PIN_FILE) == blas_identity():
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            x, y = np.asarray(x, float), np.asarray(y, float)
            if x.shape != y.shape:
                return False
            unit = '1' if x.ndim == 2 else 'Ha'
            x = align_signs(x, y)
            if not np.all(np.abs(x - y) <= same_bar(x, y, unit)):
                return False
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape:
            return False
        if x.ndim == 2:
            sign = np.sign(np.sum(x * y, axis=0))
            sign[sign == 0] = 1.0
            bar = PIN_CROSS_BLAS_BARS['continued']
            if not np.all(np.abs(x - y * sign) <= bar):
                return False
        elif not np.all(np.abs(x - y) <= PIN_CROSS_BLAS_BARS['energy']):
            return False
    return True


def assert_same_within(a, b, bar, label=None):
    """A few ulp (`same_bar`, unit 1) on the recording BLAS, within `bar` (the
    caller's derived bar for a quantity with no class of its own) elsewhere."""
    if recorded_blas(PIN_FILE) == blas_identity():
        a, b = np.asarray(a, float), np.asarray(b, float)
        assert a.shape == b.shape, (label, a.shape, b.shape)
        gap = np.abs(a - b).max(initial=0.0)
        assert gap <= same_bar(a, b, '1'), (label, gap, same_bar(a, b, '1'))
    else:
        gap = np.abs(np.asarray(a, float) - np.asarray(b, float)).max()
        assert gap <= bar, (label, gap, bar)


def bar_with_drift(tight, unit='Ha', quantity='energy'):
    """`tight` on the recording BLAS; elsewhere `tight` plus the class bar:
    a tolerance the algorithm itself owes (a Newton root's) and the drift the
    other BLAS adds to the compared quantity are independent errors."""
    from tests.pins import pin_bar
    return tight + pin_bar(recorded_blas(PIN_FILE), unit, quantity=quantity)
