"""One question decides how a mean field builds its exchange
(`isdf_jk.exchange_realization`): 'exact', 'df' or 'isdf'.

  * the three answers on water/cc-pVDZ: four-centre RHF, a density fit, an
    ISDF-K fit;
  * no module outside `src/Base/factorization` asks it again by
    `isinstance(..., ISDFJK)`, so a new realization of the exchange is added
    in one place; the scan sees a planted one.
"""
import ast
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import gto, scf

from src.Base.factorization.isdf_jk import exchange_realization, interpolated_exchange
from src.gradients.isdf_mean_field import isdf_jk_with_force
from tests.isdf_test_grid import g1

SRC = pathlib.Path(__file__).resolve().parent.parent / 'src'
#: where the realizations themselves live
OWNER = SRC / 'Base' / 'factorization'
#: modules that still ask by isinstance: a list that only shrinks
DEBT = {'properties/optimize.py'}


def isinstance_isdfjk(tree):
    """Line numbers of `isinstance(x, ISDFJK)` in a parsed module."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'isinstance'
            and len(n.args) == 2 and getattr(n.args[1], 'id', None) == 'ISDFJK']


def test_the_three_realizations():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                basis='cc-pvdz', verbose=0)
    exact = scf.RHF(mol)
    fitted = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    isdf = isdf_jk_with_force(scf.RHF(mol), auxbasis='cc-pvdz-ri',
                              counts=g1(scf.RHF(mol)))
    assert [exchange_realization(m) for m in (exact, fitted, isdf)] == [
        'exact', 'df', 'isdf']
    assert [interpolated_exchange(m) for m in (exact, fitted, isdf)] == [
        False, False, True]


@pytest.mark.parametrize('path', sorted(SRC.rglob('*.py')),
                         ids=lambda p: str(p.relative_to(SRC)))
def test_no_module_asks_by_isinstance(path):
    if OWNER in path.parents or str(path.relative_to(SRC)) in DEBT:
        return
    found = isinstance_isdfjk(ast.parse(path.read_text()))
    assert not found, (f'{path.relative_to(SRC)}:{found} asks isinstance(..., ISDFJK): '
                       'use isdf_jk.exchange_realization')


def test_the_debt_is_still_owed():
    for name in DEBT:
        assert isinstance_isdfjk(ast.parse((SRC / name).read_text())), name


def test_the_scan_sees_a_planted_question():
    assert isinstance_isdfjk(ast.parse('if isinstance(mf.with_df, ISDFJK): pass')) == [1]
