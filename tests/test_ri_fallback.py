"""The Mg auxiliary fallback is explicit: nothing substitutes unless it is asked for by name."""
import os
import sys

import pytest
from pyscf import df, gto
from pyscf.gto import basis as pyscf_basis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base import separable_ri                                    # noqa: E402
from src.Base.basis import ri_fallback                                # noqa: E402
from src.Base.constants import ISDF_AUX_FALLBACK, ISDF_MAX_AUX_L      # noqa: E402

BASES = ('aug-cc-pvdz', 'aug-cc-pvtz')
COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}


@pytest.fixture(scope='module')
def names(tmp_path_factory):
    """Register the composite sets once; the alias outlives each test, so the cache must too."""
    os.environ['MBPT_RI_FALLBACK_CACHE'] = str(tmp_path_factory.mktemp('ri_fallback'))
    return {b: ri_fallback.fallback_auxbasis(b) for b in BASES}


def _genuine(basis, els='H B C N O F Ne Al Si P S Cl'.split()):
    return {el: pyscf_basis.load(f'{basis}-ri', el) for el in els}


@pytest.mark.parametrize('basis', BASES)
def test_default_name_still_refuses_mg(basis, names):
    """No alias may make the genuine augmented name resolve for Mg."""
    with pytest.raises(pyscf_basis.BasisNotFoundError):
        pyscf_basis.load(f'{basis}-ri', 'Mg')
    with pytest.raises(ValueError, match='fallback'):
        ri_fallback.auxbasis_for(basis, ['Mg', 'H'])


@pytest.mark.parametrize('basis', BASES)
def test_other_elements_are_the_genuine_set(basis, names):
    """The composite carries each other element's own aug-cc-pVXZ-ri, bit for bit."""
    before = _genuine(basis)
    for el, shells in before.items():
        assert pyscf_basis.load(names[basis], el) == shells
        assert pyscf_basis.load(f'{basis}-ri', el) == shells
    assert pyscf_basis.load(names[basis], 'Mg') == pyscf_basis.load(
        ISDF_AUX_FALLBACK[('Mg', basis)], 'Mg')


@pytest.mark.parametrize('basis', BASES)
def test_fallback_is_within_the_grid_angular_momentum(basis, names):
    assert max(sh[0] for sh in pyscf_basis.load(names[basis], 'Mg')) <= ISDF_MAX_AUX_L


def test_explicit_opt_in_only(names):
    assert ri_fallback.auxbasis_for('aug-cc-pvdz', ['O', 'H']) == 'aug-cc-pvdz-ri'
    assert ri_fallback.auxbasis_for('cc-pvdz', ['Mg', 'H']) == 'cc-pvdz-ri'
    assert ri_fallback.auxbasis_for('aug-cc-pvdz', ['Mg', 'O'],
                                    fallback=True) == names['aug-cc-pvdz']
    with pytest.raises(ValueError):                 # Li has no fallback rule
        ri_fallback.auxbasis_for('aug-cc-pvdz', ['Li', 'H'], fallback=True)
    with pytest.raises(ValueError):
        ri_fallback.fallback_auxbasis('cc-pvdz')


def test_composite_builds_a_molecule(names):
    mol = gto.M(atom='Mg 0 0 0; H 0 0 1.7; H 0 0 -1.7', basis='aug-cc-pvdz',
                verbose=0)
    aux = df.addons.make_auxmol(mol, auxbasis=names['aug-cc-pvdz'])
    mg = df.addons.make_auxmol(gto.M(atom='Mg 0 0 0', basis='aug-cc-pvdz', verbose=0),
                               auxbasis='cc-pvqz-ri')
    h = df.addons.make_auxmol(gto.M(atom='H 0 0 0', spin=1, basis='aug-cc-pvdz',
                                    verbose=0), auxbasis='aug-cc-pvdz-ri')
    assert aux.nao == mg.nao + 2 * h.nao


def test_radii_rows_are_keyed_on_the_substitution(names, monkeypatch):
    """Mg reads its cc-pvqz-ri row, H its genuine aug row; the plain name finds no Mg row."""
    basis, aux = 'aug-cc-pvdz', names['aug-cc-pvdz']
    row = {'radii': {'A1': [1.0] * 8}, 'fit_error': 1e-3, 'points': 148}
    rows = {separable_ri._shipped_key('Mg', basis, 'cc-pvqz-ri', COUNTS): row,
            separable_ri._shipped_key('H', basis, 'aug-cc-pvdz-ri', COUNTS): row}
    monkeypatch.setattr(separable_ri, 'shipped_radii', lambda: rows)
    assert ri_fallback.element_auxbasis('Mg', basis, aux) == 'cc-pvqz-ri'
    assert ri_fallback.element_auxbasis('H', basis, aux) == 'aug-cc-pvdz-ri'
    assert ri_fallback.element_auxbasis('Mg', basis, 'cc-pvqz-ri') == 'cc-pvqz-ri'
    assert separable_ri.shipped_radii_lookup('Mg', basis, aux, COUNTS) is not None
    assert separable_ri.shipped_radii_lookup('H', basis, aux, COUNTS) is not None
    assert separable_ri.shipped_radii_lookup('Mg', basis, f'{basis}-ri', COUNTS) is None


def test_missing_mg_row_raises_naming_the_row_key(names, monkeypatch):
    monkeypatch.setattr(separable_ri, 'shipped_radii', lambda: {})
    with pytest.raises(KeyError, match='Mg\\|aug-cc-pvdz\\|cc-pvqz-ri\\|<counts>'):
        separable_ri.atomic_grid('Mg', 'aug-cc-pvdz', names['aug-cc-pvdz'], COUNTS)
