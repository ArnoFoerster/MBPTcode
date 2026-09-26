"""qsGW in a continuum, and on an unrestricted reference.

In a continuum the quasiparticle-self-consistent loop keeps what every other
route keeps: the ground state's PCM potential in the one-body Hamiltonian,
Sigma~ screened with the bare interaction, and the solute's response to its own
added charge as a static operator from Delta W = W_dressed - W_bare
(`reaction_field.continuum_operator`) whose diagonal is Duchemin et al.'s Eq.
(18). On an unrestricted reference it runs per spin around one W.

  1. The continuum operator's diagonal IS the Eq. (18) shift, on the same
     factors and the same W; it is Hermitian.
  2. The ground state's reaction field stays in the Fock matrix: at flow = 0
     (Sigma~ = 0) on a mean field converged in PCM, the loop's fixed point is
     that PCM Hartree-Fock spectrum itself.
  3. qsGW and qsGW0 converge in water, and the continuum moves the HOMO the
     way the one-shot route's Eq. (18) does.
  4. A closed shell carried as UHF reproduces the restricted loop, in the gas
     phase and in the continuum; a doublet runs through `calc_qp_energy` on
     both channels.

Run as a script (`python tests/test_qsgw_continuum.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, solvent

from src.Base.constants import HARTREE_TO_EV
from src.Base.solvent_screening import (attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qsGW import qsgw_eigenvalues
from src.SingleReference.GW.reaction_field import (
    continuum_operator, dressed_factors, environment_quasiparticle_shift)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
AUX = 'cc-pvdz-jkfit'


def _rhf(basis='6-31g'):
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    return scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-12)


def _as_unrestricted(rhf):
    """The restricted solution in both spin channels of an unrestricted object."""
    uhf = scf.UHF(rhf.mol).density_fit(auxbasis=AUX)
    uhf.mo_coeff = np.array([rhf.mo_coeff, rhf.mo_coeff])
    uhf.mo_energy = np.array([rhf.mo_energy, rhf.mo_energy])
    uhf.mo_occ = np.array([rhf.mo_occ / 2, rhf.mo_occ / 2])
    uhf.e_tot, uhf.converged = rhf.e_tot, True
    return uhf


def test_the_operator_diagonal_is_eq18():
    mf = _rhf()
    nocc = mf.mol.nelectron // 2
    attach_solvent_screening(mf, solvent='water')
    (coeff,), t = dressed_factors(mf)
    w = LinearResponseSolver(mf.mo_energy, coeff_df=coeff,
                             spin_mode='restricted').static_screening_aux(nocc)
    op = continuum_operator([coeff], t, w, (nocc,))[0]
    assert np.abs(op - op.T).max() < 1e-14
    shift = environment_quasiparticle_shift(mf)
    assert np.abs(np.diag(op) - shift).max() < 1e-12


def test_the_ground_state_reaction_field_stays_in_the_fock_matrix():
    """flow = 0 zeroes Sigma~, and with no screening attached there is no
    continuum operator either: what is left is h + J + K + v_PCM[D], whose
    fixed point is the PCM Hartree-Fock the mean field converged to. Without
    the PCM term it would be the gas-phase Hartree-Fock instead, tenths of an
    eV away on water. The loop builds J and K exactly, so the reference is
    converged without density fitting too (a fitted one sits 1e-4 Ha away)."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    pcm = solvent.PCM(scf.RHF(mol))
    pcm.with_solvent.eps = 78.355
    pcm.conv_tol = 1e-12
    pcm.kernel()
    eps, _, info = qsgw_eigenvalues(pcm, flow=0.0, tol=1e-9, dm_tol=1e-9)
    assert info['converged']
    assert np.abs(eps - pcm.mo_energy).max() < 1e-7, \
        np.abs(eps - pcm.mo_energy).max()


@pytest.mark.parametrize('screening', ['updated', 'fixed'])
def test_the_loop_converges_in_water(screening):
    mf = _rhf()
    nocc = mf.mol.nelectron // 2
    gas, _, _ = qsgw_eigenvalues(mf, screening=screening)
    attach_solvent_screening(mf, solvent='water')
    try:
        eps, _, info = qsgw_eigenvalues(mf, screening=screening)
        eq18 = environment_quasiparticle_shift(mf)
    finally:
        detach_solvent_screening(mf)
    assert info['converged']
    assert info['sigma_solvent'] is not None
    moved = (eps[nocc - 1] - gas[nocc - 1]) * HARTREE_TO_EV
    one_shot = eq18[nocc - 1] * HARTREE_TO_EV
    # the self-consistent response relaxes the orbitals in the reaction
    # field, so the shift is Eq. (18)'s sign and size, not its value
    assert moved > 0 and abs(moved / one_shot - 1.0) < 0.35, (moved, one_shot)


@pytest.mark.parametrize('continuum', [False, True])
def test_closed_shell_carried_unrestricted(continuum):
    rhf = _rhf()
    uhf = _as_unrestricted(rhf)
    if continuum:
        for mf in (rhf, uhf):
            attach_solvent_screening(mf, solvent='water')
    try:
        ref, _, _ = qsgw_eigenvalues(rhf, tol=1e-8, dm_tol=1e-8)
        got, mo, info = qsgw_eigenvalues(uhf, tol=1e-8, dm_tol=1e-8)
    finally:
        for mf in (rhf, uhf):
            detach_solvent_screening(mf)
    assert info['converged'] and got.shape == (2, ref.size)
    assert np.abs(got - ref[None]).max() < 1e-7, np.abs(got - ref[None]).max()


def test_a_doublet_through_the_front_door():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    attach_solvent_screening(uhf, solvent='water')
    for channel in ('alpha', 'beta'):
        e = calc_qp_energy(uhf, state='homo', spin_channel=channel,
                           self_consistency='qsGW')
        assert np.isfinite(e), channel


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
