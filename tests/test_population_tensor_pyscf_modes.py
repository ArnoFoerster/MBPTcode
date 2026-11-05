"""The fragment population callback answers both pyscf call shapes.

`src.fragments.localization.population_tensor` returns the `atomic_pops`
callback pyscf's `PipekMezey` reads. pyscf up to 2.13 calls it as
`atomic_pops(mol, C)` (or with a method) and takes the full tensor
(nfrag, n, n) everywhere; pyscf 2.14's cost function calls
`atomic_pops(mol, C, mode='pop')` and takes the diagonal (nfrag, n). The
check calls it both ways, on two H2 fragments in both population schemes,
and asks that the 2.14 cost, (pop**2).sum(), is bitwise the one older pyscf
forms from the full tensor (einsum('xii->xi')), so the localization does not
move with the pyscf version. And the localization pyscf runs on that
callback is stationary for the fragment functional (pyscf 2.14 follows its
own populations unless `pop_method` is None).

Run: pytest tests/test_population_tensor_pyscf_modes.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pyscf import gto, scf  # noqa: E402

from src.Base.numerics.constants import FRAGMENT_PM_CONV_TOL_GRAD  # noqa: E402
from src.fragments.localization import (_localize, _stationarity,  # noqa: E402
                                        population_tensor)


@pytest.fixture(scope='module')
def dimer():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 3.0 0; H 0 3.0 0.74',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


@pytest.mark.parametrize('scheme', ['lowdin', 'iao'])
def test_pop_mode_is_the_diagonal_of_the_full_tensor(dimer, scheme):
    mol, mf = dimer
    occ = mf.mo_coeff[:, mf.mo_occ > 0]
    pops = population_tensor(mol, [[0, 1], [2, 3]], scheme=scheme, orbocc=occ)
    full = pops(mol, occ)
    assert full.shape == (2, occ.shape[1], occ.shape[1])
    assert np.array_equal(pops(mol, occ, 'meta_lowdin'), full)
    pop = pops(mol, occ, mode='pop')                    # pyscf 2.14
    assert np.array_equal(pop, np.einsum('kii->ki', full))
    assert pops(mol, occ, None, kpt=None, proj_data=None, mode='pop',
                verbose=None).shape == (2, occ.shape[1])
    assert (pop ** 2).sum() == (np.einsum('xii->xi', full) ** 2).sum()


def test_the_localization_is_stationary_for_the_fragment_functional():
    """pyscf 2.14 forms the Pipek-Mezey gradient and Hessian from its own
    `pop_method` populations unless that is None; with it left at
    'meta_lowdin' the walk followed atomic populations while the cost read
    the fragment ones, and stopped where the fragment functional is not
    stationary. The between-fragment residual, from the fragment populations
    alone (`_stationarity`), is held to the polish's own tolerance."""
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587; '
                     'O 0 0 3.0; H 0 0.757 3.587; H 0 -0.757 3.587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    occ = mf.mo_coeff[:, mf.mo_occ > 0]
    pops = population_tensor(mol, [[0, 1, 2], [3, 4, 5]])
    u, _ = _localize(mol, occ, pops)
    proj = pops(mol, occ @ u)
    labels = np.einsum('kii->ik', proj).argmax(axis=1)
    between, _ = _stationarity(proj, labels)
    assert between < FRAGMENT_PM_CONV_TOL_GRAD, between
