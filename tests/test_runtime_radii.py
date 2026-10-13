"""A missing ISDF grid row is re-optimized loudly and in the element's own box.

The shipped radii table is the validated answer; a (element, basis, aux,
counts) it does not hold is re-optimized at run time, with a warning, in the
element's own search box -- the 5 Bohr box of a single descent cuts lithium's
valence density off (its multi-start box is 16 Bohr):

  1. A tabulated row is returned as it is, with no warning and no search.
  2. A missing row warns, and the search runs in `ELEMENT_R_MAX` -- 16 Bohr
     for Li, 5 Bohr for H and the second row.

The optimizer is replaced by a recorder here, so the test checks the decision
and the box without paying for a search.

Run as a script (`python tests/test_runtime_radii.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

import src.Base.separable_ri as sri
from src.Base.constants import ISDF_DEFAULT_COUNTS

UNTABULATED = {'A1': 3, 'A2': 2, 'A3': 1, 'A4': 1}


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, element, basis, auxbasis, counts=None, n_start=1,
                 r_max=None, **kwargs):
        self.calls.append({'element': element, 'r_max': r_max,
                           'n_start': n_start})
        return {name: np.ones(n) for name, n in (counts or {}).items()}, 0.01


def _with_recorder(fn):
    original = sri.optimize_atomic_radii
    recorder = _Recorder()
    sri.optimize_atomic_radii = recorder
    try:
        return fn(), recorder
    finally:
        sri.optimize_atomic_radii = original


def test_tabulated_row_is_returned_silently():
    if sri.shipped_radii_lookup('H', 'cc-pvdz', 'cc-pvdz-ri', ISDF_DEFAULT_COUNTS) is None:
        pytest.skip('no tabulated H/cc-pvdz row at the default counts')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        hit, recorder = _with_recorder(lambda: sri.runtime_atomic_radii(
            'H', 'cc-pvdz', 'cc-pvdz-ri', ISDF_DEFAULT_COUNTS))
    assert recorder.calls == [], 'a tabulated row must not trigger a search'
    assert hit[0], 'the row comes back'


@pytest.mark.parametrize('element,box', [('Li', 16.0), ('O', 5.0)])
def test_missing_row_warns_and_uses_the_element_box(element, box):
    with pytest.warns(RuntimeWarning, match='no tabulated ISDF grid'):
        _, recorder = _with_recorder(lambda: sri.runtime_atomic_radii(
            element, 'cc-pvdz', 'cc-pvdz-ri', UNTABULATED))
    assert len(recorder.calls) == 1
    assert recorder.calls[0]['r_max'] == box, recorder.calls[0]


def test_lookup_is_case_insensitive():
    """pyscf's basis names are case-insensitive, so the table's must be too:
    'cc-pVDZ' must not miss every row and fall through to a run-time search."""
    lower = sri.shipped_radii_lookup('H', 'cc-pvdz', 'cc-pvdz-ri', ISDF_DEFAULT_COUNTS)
    if lower is None:
        pytest.skip('no tabulated H/cc-pvdz row at the default counts')
    mixed = sri.shipped_radii_lookup('H', 'cc-pVDZ', 'cc-pVDZ-RI', ISDF_DEFAULT_COUNTS)
    assert mixed is not None
    assert all(np.array_equal(lower[0][k], mixed[0][k]) for k in lower[0])


def test_per_element_basis_resolves_each_element():
    basis = {'Li': 'cc-pvdz', 'H': 'cc-pVDZ'}
    aux = sri.default_auxbasis(basis)
    assert aux == {'Li': 'cc-pvdz-ri', 'H': 'cc-pvdz-ri'}
    assert sri.element_basis_name(basis, 'H') == 'cc-pvdz'
    assert sri.single_basis_name(basis) == 'cc-pvdz'
    with pytest.raises(ValueError, match='mixes'):
        sri.single_basis_name({'Li': 'cc-pvdz', 'H': 'aug-cc-pvdz'})
    with pytest.raises(TypeError, match='not a named set'):
        sri.element_basis_name({'H': [[0, [1.0, 1.0]]]}, 'H')


def test_per_element_basis_runs_end_to_end():
    """A dict basis on LiH reaches the space-time route through the table rows
    of each element, and returns a finite quasiparticle energy."""
    from pyscf import gto, scf
    from src.SingleReference.GW.qp_energy import calc_qp_energy
    basis = {'Li': 'cc-pvdz', 'H': 'cc-pvdz'}
    for el in ('Li', 'H'):
        if sri.shipped_radii_lookup(el, basis, sri.default_auxbasis(basis),
                                    ISDF_DEFAULT_COUNTS) is None:
            pytest.skip(f'no tabulated {el}/cc-pvdz row at the default counts')
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=sri.default_auxbasis(basis)).run()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)   # no run-time search
        ip = calc_qp_energy(mf, selfenergy='GW', df=True, state='homo',
                            mode='space-time')
    assert np.isfinite(ip)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
