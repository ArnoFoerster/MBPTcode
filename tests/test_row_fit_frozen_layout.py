"""The row-distributed fit keeps the reference geometry's pair layout at a
displaced geometry, as the replicated fit does and as its adjoint assumes.

The test-set pairs are screened once, at the reference (`FrozenFactorization.
layout`), and the force differentiates that frozen set (`row_fit_branches`).
A forward that screened again at each geometry would be a different estimator
wherever a pair crosses `pair_tol`, a step in the surface the force cannot see.
Formaldehyde/cc-pVDZ drops one pair of the reference screen when a hydrogen
moves 0.1 Bohr along y.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base import separable_ri  # noqa: E402
from src.gradients.factor_chain import FrozenFactorization  # noqa: E402
from src.gradients.rpa_ground_state import RPAGroundStateChain  # noqa: E402

ATOM = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.58; H 0 -0.94 -0.58'


def _rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def _displaced(mol, atom, axis, h):
    coords = mol.atom_coords().copy()
    coords[atom, axis] += h
    m = mol.copy()
    m.set_geom_(coords, unit='Bohr')
    m.build(False, False)
    return m


def test_the_row_fit_keeps_the_reference_pairs_at_a_displaced_geometry():
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    fac = FrozenFactorization(mol, sliced=True, fit='rows')
    chain = RPAGroundStateChain(mol, _rhf, factorization=fac)
    m = _displaced(mol, 2, 1, 0.1)
    here = separable_ri.test_set_layout(m, fac.coords(m))
    # the case is only a test if the geometry's own screen differs
    assert len(here[0]) != len(fac.layout[0])
    mf = chain.mean_field(m)[1]
    x_mo, d, _, auxmol, crd, _ = chain.factors_at(m, mf)
    frozen = separable_ri.fit_M_streaming(m, auxmol, crd, fit='rows',
                                          block=fac.fit_block, layout=fac.layout)
    rescreened = separable_ri.fit_M_streaming(m, auxmol, crd, fit='rows',
                                              block=fac.fit_block, layout=here)
    d_frozen = frozen.metric_root_rows(auxmol, chain.environment_at(m))
    d_here = rescreened.metric_root_rows(auxmol, chain.environment_at(m))
    assert np.abs(d_frozen - d_here).max() > 0.0
    assert np.array_equal(d, d_frozen)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
