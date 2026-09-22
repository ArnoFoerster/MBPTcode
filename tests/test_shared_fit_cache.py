"""The fit cache is split at the gauge, and the gauge stays per chain.

`FactorChain.factors` is not a pure function of the factorization: its last step
dresses the auxiliary gauge with the chain's own environment. The cache is
therefore keyed before the gauge -- `shareable_factors` returns (X_ao, Mfit, V),
which is 96.6% of the per-geometry cost, and each chain applies its own
`aux_metric_sqrt`.

The cost and retention of that cache are gated in test_frozen_factorization.py,
which owns `FrozenFactorization`. Here: that the split moved no bits, and that
two environments on one factorization keep separate gauges.
"""
import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

# `test_set_D` is aliased on import: pytest COLLECTS any module-level
# callable whose name starts with `test_`, so importing it plainly adds a
# phantom test that errors on missing fixtures.
from src.Base.separable_ri import aux_metric_sqrt, fit_M_stable
from src.Base.separable_ri import test_set_D as build_test_set_D
from src.Base.solvent_screening import SolventScreening
from src.gradients.factor_chain import FrozenFactorization
from src.gradients.rpa_ground_state import RPAGroundStateChain

BASIS = 'cc-pvdz'


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis=BASIS, verbose=0)


def _fit_the_old_way(chain, mol, auxmol, crd, layout):
    """The fit assembled in one piece, to check the split against."""
    mu_i, nu_i, wc_l = layout
    naux = auxmol.nao_nr()
    Dt = build_test_set_D(mol, auxmol, crd, layout)
    V = auxmol.intor('int2c2e', aosym='s1')
    e3 = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                                aosym='s1').reshape(mol.nao, mol.nao, naux)
    F = np.hstack([np.linalg.solve(V, e3[mu_i, nu_i, :].T) * wc_l[None, :],
                   np.eye(naux)])
    M = fit_M_stable(Dt, F)
    return (mol.eval_gto('GTOval_sph', crd),
            M.T @ aux_metric_sqrt(auxmol, chain.environment_at(mol), V=V))


def test_the_split_is_bitwise(mol):
    """Splitting the fit at the gauge must not move a single bit."""
    ch = RPAGroundStateChain(mol, rhf)
    fac = ch.factorization
    auxmol, crd = fac.auxmol(mol), fac.coords(mol)
    ox, od = _fit_the_old_way(ch, mol, auxmol, crd, fac.layout)
    nx, nd = ch.factors(mol, auxmol, crd)
    assert np.array_equal(ox, nx)
    assert np.array_equal(od, nd)


def test_two_environments_do_not_share_a_gauge(mol):
    """THE GATE THIS SPLIT EXISTS FOR. Two chains on ONE factorization with
    DIFFERENT environments must get different D, and each must equal what it
    would get standing alone. If this ever fails, a solvated calculation is
    silently using someone else's screening."""
    shared = FrozenFactorization(mol, basis=BASIS, auxbasis=BASIS + '-ri')
    auxmol, crd = shared.auxmol(mol), shared.coords(mol)
    gas = RPAGroundStateChain(mol, rhf, factorization=shared)
    sol = RPAGroundStateChain(mol, rhf, factorization=shared,
                              environment=SolventScreening(mol, solvent='toluene'))
    _, d_gas = gas.factors(mol, auxmol, crd)
    _, d_sol = sol.factors(mol, auxmol, crd)
    assert not np.allclose(d_gas, d_sol), 'the environment did not reach D'

    alone = RPAGroundStateChain(mol, rhf,
                                environment=SolventScreening(mol, solvent='toluene'))
    _, d_alone = alone.factors(mol, alone.factorization.auxmol(mol),
                               alone.factorization.coords(mol))
    assert np.array_equal(d_sol, d_alone), 'sharing changed the solvated gauge'


