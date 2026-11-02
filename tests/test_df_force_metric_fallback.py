"""The DF-MP2 force on an auxiliary set with a near-linear dependency.

The energy's fit takes pyscf's eigen-replacement where the Coulomb metric is
not positive definite; the force has to take the same one, or it raises (a
bare Cholesky) or differentiates another fit. Water in 6-31G, correlation fit
a small even-tempered set carrying one function twice (exponents 1e-9 apart),
JK fit def2-universal-jkfit: the force is finite and is the five-point
central difference of E_HF + E_MP2 on every coordinate. The tiled force
(`df_correlation_force_tiled`, the route of every system above the dense
route's size) reads the same pruned fit and gives the dense force.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.factorization.distributed_df import metric_factor
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as dcft
from src.gradients.df_factors import mp2_factors

ATOM = [['O', (0.0, 0.1, 0.0)], ['H', (0.1, -0.76, 0.58)], ['H', (-0.05, 0.77, 0.56)]]
STEP = 1e-3
_SET = [[0, [28.0, 1.0]], [0, [7.0, 1.0]], [0, [1.75, 1.0]], [0, [1.75 + 1e-9, 1.0]],
        [0, [0.44, 1.0]], [1, [4.0, 1.0]], [1, [1.0, 1.0]], [2, [1.2, 1.0]]]
AUX = {'O': _SET, 'H': _SET}


def _mol(coords):
    atoms = [[sym, tuple(c)] for (sym, _), c in zip(ATOM, coords)]
    return gto.M(atom=atoms, basis='6-31g', unit='Bohr', verbose=0)


def _mean_field(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    return mf.run()


def _force_and_energy(coords):
    mol = _mol(coords)
    mf = _mean_field(mol)
    force, (e2, *_) = dcf.mp2_force(mf, mol, 0, AUX)
    return np.asarray(force), mf.e_tot + e2


def test_the_planted_set_is_not_positive_definite():
    mol = _mol(np.array([a[1] for a in ATOM]))
    auxmol = gto.mole.Mole()
    from pyscf import df
    auxmol = df.addons.make_auxmol(mol, AUX)
    assert metric_factor(mol, auxmol)[1] == 'eig'


def test_the_force_on_a_dependent_auxiliary_set_is_finite_and_the_energys_slope():
    coords = np.array([a[1] for a in ATOM]) * 1.8897261246
    force, _ = _force_and_energy(coords)
    assert np.isfinite(force).all()
    fd = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for k in range(3):
            e = []
            for h in (-2 * STEP, -STEP, STEP, 2 * STEP):
                c = coords.copy()
                c[i, k] += h
                e.append(_force_and_energy(c)[1])
            fd[i, k] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * STEP)
    assert np.abs(force - fd).max() < 1e-6, np.abs(force - fd).max()


def test_the_tiled_force_on_a_dependent_auxiliary_set_is_the_dense_force():
    mol = _mol(np.array([a[1] for a in ATOM]) * 1.8897261246)
    mf = _mean_field(mol)
    dense, (e_dense, *_) = dcf.mp2_force(mf, mol, 0, AUX)
    tiled, (e_tiled, *_) = dcft.mp2_force_tiled(mf, mol, mp2_factors, 0, AUX)
    assert abs(e_tiled - e_dense) < 1e-10, e_tiled - e_dense
    assert np.abs(np.asarray(tiled) - np.asarray(dense)).max() < 1e-9, \
        np.abs(np.asarray(tiled) - np.asarray(dense)).max()
