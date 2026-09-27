"""The continuum's Eq. (18) shift for an unrestricted reference.

An unrestricted reference takes Duchemin et al.'s Eq. (18) like the restricted
routes, not the static COHSEX operator (a different solvated functional, 0.39
eV of quasiparticle gap apart on water in water): ONE screened interaction from
the spin-summed chi0, contracted against each spin orbital's own density with
its own occupancy sign.

  1. A closed shell run unrestricted gives the restricted shift in both spin
     rows, and the same solvent shift of its HOMO through calc_qp_energy.
  2. A doublet gives distinct alpha and beta shifts, and a finite solvated
     quasiparticle energy on both channels.
  3. evGW and evGW0 run on it in the continuum, on both spin channels; evGW0
     forms the shift on the mean field's spectrum alone and evGW re-forms it
     on its iterates.
  4. The reference is diagnosed: <S^2> and internal stability are reported.
  5. A doublet's compact hole in a sphere is Born's: the optical shift of the
     fluorine atom's 2p is (1 - 1/eps_inf)/2a.

Run as a script (`python tests/test_unrestricted_reaction_field.py`) or under
pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.solvent.pcm import modified_Bondi

from src.Base.environment import attach_environment
from src.Base.pyscf_interface import unrestricted_reference_diagnostics
from src.Base.solvent_screening import (SolventScreening,
                                        attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.GW import qp_energy
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
OH = 'O 0 0 0; H 0 0 0.97'
AUX = 'cc-pvdz-jkfit'


def _pair(atom, spin=0):
    mol = gto.M(atom=atom, basis='6-31g', spin=spin, verbose=0)
    rhf = scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11) if spin == 0 else None
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    return mol, rhf, uhf


def test_closed_shell_unrestricted_equals_restricted():
    """Compared as SOLVENT SHIFTS: the RHF and UHF self-consistent fields of one
    closed shell differ by ~1e-7 Ha in their orbital energies (2e-6 eV of gas-
    phase HOMO), which is SCF noise, not the continuum; the shift agrees to
    ~2e-7 eV, where the static COHSEX operator would sit a tenth of an eV
    away."""
    mol, rhf, uhf = _pair(WATER)

    def homo(mf):
        return calc_qp_energy(mf, selfenergy='GW', df=True, state='homo')

    gas = {'r': homo(rhf), 'u': homo(uhf)}
    for mf in (rhf, uhf):
        attach_solvent_screening(mf, solvent='water')
    try:
        s_r = environment_quasiparticle_shift(rhf)
        s_u = environment_quasiparticle_shift(uhf)
        assert s_u.shape == (2, s_r.size)
        assert np.abs(s_u - s_r[None, :]).max() < 1e-8, \
            np.abs(s_u - s_r[None, :]).max()
        shift_r = homo(rhf) - gas['r']
        shift_u = homo(uhf) - gas['u']
        assert abs(shift_r - shift_u) < 1e-6, (shift_r, shift_u)
    finally:
        for mf in (rhf, uhf):
            detach_solvent_screening(mf)


def test_doublet_has_distinct_spin_shifts():
    mol, _, uhf = _pair(OH, spin=1)
    attach_solvent_screening(uhf, solvent='water')
    try:
        s = environment_quasiparticle_shift(uhf)
        na, nb = uhf.nelec
        assert s.shape[0] == 2
        assert np.abs(s[0] - s[1]).max() > 1e-4, 'alpha and beta see different densities'
        # the occupancy sign per spin: an occupied level rises, a virtual falls
        assert s[0, na - 1] > 0 and s[0, na] < 0
        assert s[1, nb - 1] > 0 and s[1, nb] < 0
        for channel in ('alpha', 'beta'):
            e = calc_qp_energy(uhf, selfenergy='GW', df=True, state='homo',
                               spin_channel=channel)
            assert np.isfinite(e)
    finally:
        detach_solvent_screening(uhf)


def test_self_consistent_loops_take_the_spin_shift():
    """evGW and evGW0 on an unrestricted reference in a continuum: the loop
    re-evaluates the per-spin Eq. (18) shift (evGW0 keeps the mean field's)
    and converges to a finite level on both channels."""
    _, _, uhf = _pair(OH, spin=1)
    attach_solvent_screening(uhf, solvent='water')
    try:
        for consistency in ('evGW', 'evGW0'):
            for channel in ('alpha', 'beta'):
                e = calc_qp_energy(uhf, selfenergy='GW', df=True, state='homo',
                                   spin_channel=channel,
                                   self_consistency=consistency)
                assert np.isfinite(e), (consistency, channel, e)
    finally:
        detach_solvent_screening(uhf)


def test_evgw0_keeps_the_mean_field_shift():
    """Every spectrum the shift is formed on, recorded: evGW0's loop never
    leaves the mean field's, evGW's does."""
    _, _, uhf = _pair(OH, spin=1)
    attach_solvent_screening(uhf, solvent='water')
    real = qp_energy.environment_quasiparticle_shift
    seen = []

    def spy(mf, *args, **kwargs):
        seen.append(np.array(mf.mo_energy, float))
        return real(mf, *args, **kwargs)

    qp_energy.environment_quasiparticle_shift = spy
    try:
        mean_field = np.array(uhf.mo_energy, float)
        for consistency, moves in (('evGW0', False), ('evGW', True)):
            seen.clear()
            calc_qp_energy(uhf, selfenergy='GW', df=True, state='homo',
                           self_consistency=consistency)
            off = [np.abs(e - mean_field).max() for e in seen]
            assert seen and (max(off) > 1e-3) == moves, (consistency, off)
    finally:
        qp_energy.environment_quasiparticle_shift = real
        detach_solvent_screening(uhf)


def test_born_sphere_limit_of_a_doublet():
    """Measured on F/cc-pVDZ: 1.0024 of Born's, Lebedev-converged."""
    mol = gto.M(atom='F 0 0 0', basis='cc-pvdz', spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    eps_inf = 1.776
    attach_environment(uhf, SolventScreening(mol, eps=eps_inf,
                                             eps_static=78.355))
    radius = 1.2 * modified_Bondi[9]
    shift = environment_quasiparticle_shift(uhf)[0, uhf.nelec[0] - 1]
    assert abs(shift / (0.5 * (1 - 1 / eps_inf) / radius) - 1) < 1e-2, shift


def test_reference_is_diagnosed():
    _, _, uhf = _pair(OH, spin=1)
    diag = unrestricted_reference_diagnostics(uhf, 'test')
    assert abs(diag['s2_exact'] - 0.75) < 1e-12
    assert diag['s2'] >= diag['s2_exact'] - 1e-8
    assert isinstance(diag['stable'], bool)
    assert unrestricted_reference_diagnostics(scf.RHF(gto.M(atom=WATER, verbose=0)),
                                              'test') is None


TESTS = [test_closed_shell_unrestricted_equals_restricted,
         test_doublet_has_distinct_spin_shifts,
         test_self_consistent_loops_take_the_spin_shift,
         test_evgw0_keeps_the_mean_field_shift,
         test_born_sphere_limit_of_a_doublet,
         test_reference_is_diagnosed]


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f'[OK  ] {test.__name__}')
        except (Exception, pytest.fail.Exception) as exc:   # noqa: BLE001
            failed += 1
            print(f'[FAIL] {test.__name__} -- {type(exc).__name__}: {exc}')
    print('\nALL PASSED' if not failed else '\nFAILURES DETECTED')
    sys.exit(0 if not failed else 1)
