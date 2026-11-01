"""The mean-field part of the correlated forces -- DF-MP2 and ADC(2), dense
and tiled, and the state models' -- is the tiled assembly every chain shares
(`isdf_mean_field.mean_field_skeleton_force`): Coulomb and exchange by
auxiliary tiles over the ranks, not pyscf's whole gradient on every rank.

Water cc-pVDZ DF-RHF (cc-pvdz-jkfit; correlation fit cc-pvdz-ri, frozen core):
  1. with pyscf's gradient kernel replaced by one that raises, the dense MP2
     force, the tiled MP2 force and the tiled ADC(2) singlet force still run;
  2. `HFState.force` in an external one-electron field (a uniform electric
     field in the core Hamiltonian): pyscf's DF gradient of the same SCF, the
     field's own derivative left out, within 1e-12 Ha/Bohr; Hartree-Fock and
     PBE0 (density pruning off, the grid pyscf's response regenerates);
  3. the mean-field addend of the tiled MP2 force at 1, 2 and 3 simulated
     ranks: the same bits on every rank and at every rank count;
  4. a consistency check: the GW-BSE state model's reference (S0) on DF-RKS
     PBE0, water def2-SVP (def2-universal-jkfit, the default pruned grid), its
     force taken with pyscf's gradient kernel refused, against the five-point
     difference of its energy on every coordinate (step 1e-3 Bohr) within 1e-8
     Ha/Bohr (2.1e-10 here; the difference's own floor is about 1e-10). pyscf's
     own force of this reference is within 6e-10 of the same difference, so
     the tiled force is the same derivative, not a more accurate one.

Tests 1 to 4 fail where a force still runs pyscf's whole gradient.

Run: python -m pytest tests/test_correlated_mean_field_force.py
"""
import numpy as np
import pytest
from pyscf import dft
from pyscf.grad import rhf as rhf_grad

from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients import df_factors
from src.gradients.adc import ee_gradient as eg
from src.gradients.gwbse.state_model import GWBSEState
from src.gradients.state_models import HFState
from tests.adc_common import (ATOMS, AUX, FD_STEP_DENSITY, JKFIT, NF, SCF_GRAD_TILED,
                              SCF_TOL_TIGHT, df_rhf, molecule, five_point)

BASIS = 'cc-pvdz'
# one occupied orbital per tile, a few AO functions per block, 20 auxiliary
# functions per chunk, so the correlation part runs over many tiles too
TILED = dict(tile_gb=1e-9, block_gb=1e-4, width=20)
FIELD = (0.003, -0.002, 0.004)


def water():
    mol = molecule(ATOMS, basis=BASIS)
    mf = df_rhf(mol, SCF_TOL_TIGHT, SCF_GRAD_TILED)
    mf.kernel()
    return mol, mf


def refuse_pyscf_kernel(monkeypatch):
    def refused(self, *args, **kwargs):
        raise AssertionError("pyscf's whole mean-field gradient was called")
    monkeypatch.setattr(rhf_grad.GradientsBase, 'kernel', refused)


def test_correlated_forces_never_run_pyscfs_whole_gradient(monkeypatch):
    mol, mf = water()
    refuse_pyscf_kernel(monkeypatch)
    with distributed(None):
        dense = dcf.mp2_force(mf, mol, NF, AUX)[0]
    tiled = tcf.mp2_force_tiled(mf, mol, df_factors.mp2_factors, NF, AUX,
                                **TILED)[0]
    singlet = eg.adc2_force_tiled(mf, mol, 'singlet', frozen=NF, auxbasis=AUX,
                                  aux_chunk=20, **TILED)[0]
    assert np.abs(dense - tiled).max() < 1e-10
    assert np.isfinite(singlet).all()


def field_operator(mol):
    return np.einsum('x,xij->ij', np.asarray(FIELD), mol.intor('int1e_r'))


def ks_scf(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=JKFIT)
    mf.grids.level, mf.small_rho_cutoff = 4, 0
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-9, 0
    return mf


@pytest.mark.parametrize('kind', ['hf', 'pbe0'])
def test_state_model_force_in_a_field_is_pyscfs(kind, monkeypatch):
    mol = molecule(ATOMS, basis=BASIS)
    make = (lambda m: df_rhf(m, SCF_TOL_TIGHT, SCF_GRAD_TILED)) if kind == 'hf' \
        else ks_scf
    state = HFState(mol, make)
    h = field_operator(mol)
    with monkeypatch.context() as patched:
        refuse_pyscf_kernel(patched)
        F, _ = state.force(h, 0.0)
    # pyscf's gradient of the SCF the force was taken on
    g = state.mf.nuc_grad_method()
    g.verbose = 0
    if kind == 'pbe0':
        g.grid_response = True
    apart = float(np.abs(F - g.kernel()).max())
    print(f'\n{kind} in a field: max |F - pyscf| {apart:.1e} Ha/Bohr')
    assert apart <= 1e-12


def test_mean_field_addend_is_the_same_bits_at_every_rank_count(monkeypatch):
    mol, mf = water()
    seen = {}
    own = tcf.mean_field_skeleton_force

    def recorded(m):
        out = own(m)
        comm = current_comm()
        seen.setdefault(1 if comm is None else comm.Get_size(), []).append(out.copy())
        return out
    monkeypatch.setattr(tcf, 'mean_field_skeleton_force', recorded)

    def run(comm):
        return tcf.mp2_force_tiled(mf, mol, df_factors.mp2_factors, NF, AUX,
                                   comm=comm, **TILED)[0]
    run(None)
    for size in (2, 3):
        run_simulated(run, size)
    ref = seen[1][0]
    assert sorted(seen) == [1, 2, 3] and [len(seen[s]) for s in (2, 3)] == [2, 3]
    for size in (2, 3):
        for got in seen[size]:
            np.testing.assert_array_equal(got, ref)


def pbe0_svp(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-13, 1e-9, 0
    return mf


def test_gwbse_reference_force_is_its_energys_difference(monkeypatch):
    mol = molecule(ATOMS, basis='def2-svp')
    with monkeypatch.context() as patched:
        refuse_pyscf_kernel(patched)
        F, _ = GWBSEState(mol, pbe0_svp, numerics=Numerics()).force(0.0, 0.0)
    worst = 0.0
    for a in range(mol.natm):
        for x in range(3):
            d = np.zeros((mol.natm, 3))
            d[a, x] = 1.0

            def energy(t):
                m = molecule(ATOMS, t * d, basis='def2-svp')
                return GWBSEState(m, pbe0_svp, numerics=Numerics())(0.0, 0.0).energy
            worst = max(worst, abs(F[a, x] - five_point(energy, FD_STEP_DENSITY)))
    print(f'\nGW-BSE reference, DF-PBE0 def2-SVP: max |F - FD| {worst:.1e} Ha/Bohr')
    assert worst <= 1e-8
