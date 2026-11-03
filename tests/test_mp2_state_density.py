"""The MP2 state model's call (gradients.state_models.MP2State): its energy and
orbital-relaxed density from the force's own engine (mp2_density on the dense
reference, mp2_density_tiled on the tiled route), no derivative integrals.

  1. water 6-31G (bent, non-symmetric), frozen 1, cc-pvdz-ri / -jkfit, on both
     routes: E_MP2 is pyscf's DF-MP2 energy to 1e-10 Ha, and Tr(D mu_z) is the
     five-point finite-field derivative of the call's energy to 1e-8.
  2. water aug-cc-pVDZ with the overlap eigenvalues below 3e-2 dropped (41 AOs,
     37 MOs, as 2CzPN aug-cc-pVDZ drops 13 of 990), tiled route: the call runs
     (pyscf's DF-MP2 relaxed density refuses such a mean field) and Tr(D mu_z)
     is the finite-field derivative to 1e-7.

Run: python -m pytest -q tests/test_mp2_state_density.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.mp import dfmp2_native
from pyscf.scf import hf as scf_hf

from src.Base.numerics.numerics import Numerics
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import ATOMS, AUX, NF, five_point, make_scf_loop, molecule
from tests.molecules import WATER

#: Drops the four overlap eigenvalues of water/aug-cc-pVDZ below it.
LINDEP = 3e-2
FIELD_STEP = 1e-3


def finite_field(model, mol):
    """(Tr(D mu_z) of the call at zero field, d E / d lambda by five points)."""
    mu = -mol.intor('int1e_r')[2]
    r = model(np.zeros((mol.nao, mol.nao)), 0.0)
    dE = five_point(lambda lam: model(lam * mu, 0.0).energy, FIELD_STEP)
    return float(np.sum(r.dm_eff * mu)), dE


@pytest.mark.parametrize('route', ['dense', 'tiled'])
def test_energy_and_density(route):
    mol = molecule(ATOMS)
    model = MP2State(HFState(mol, make_scf_loop), NF, AUX, force_route=route,
                     numerics=Numerics())
    r = model(np.zeros((mol.nao, mol.nao)), 0.0)
    mp = dfmp2_native.DFRMP2(model.mf, frozen=NF, auxbasis=AUX)
    mp.verbose = 0
    e_ref = mp.kernel()
    assert abs(r.info['e_mp2'] - e_ref) < 1e-10
    tr, dE = finite_field(model, mol)
    assert abs(tr - dE) < 1e-8, (tr, dE)


def lindep_scf(mol):
    """The loop's DF-RHF with the overlap eigenvalues below LINDEP dropped:
    pyscf >= 2.13 drops them inside hf.eig (module settings), earlier pyscf
    through remove_linear_dep_."""
    mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-12, 1e-9, 200, 0
    if not hasattr(scf_hf, 'overlap_zero_eigenvalue_threshold'):
        return scf.addons.remove_linear_dep_(mf, threshold=LINDEP, lindep=LINDEP)
    kernel = mf.kernel

    def dropped(*a, **k):
        saved = (scf_hf.remove_overlap_zero_eigenvalue,
                 scf_hf.overlap_zero_eigenvalue_threshold)
        scf_hf.remove_overlap_zero_eigenvalue = True
        scf_hf.overlap_zero_eigenvalue_threshold = LINDEP
        try:
            return kernel(*a, **k)
        finally:
            (scf_hf.remove_overlap_zero_eigenvalue,
             scf_hf.overlap_zero_eigenvalue_threshold) = saved
    mf.kernel = dropped
    return mf


def test_dropped_combinations():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER,
                basis='aug-cc-pvdz', verbose=0)
    model = MP2State(HFState(mol, lindep_scf), 1, 'aug-cc-pvdz-ri', force_route='tiled',
                     numerics=Numerics())
    tr, dE = finite_field(model, mol)
    assert model.mf.mo_coeff.shape == (41, 37)
    assert abs(tr - dE) < 1e-7, (tr, dE)
