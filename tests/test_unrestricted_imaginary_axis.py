"""An unrestricted reference on the imaginary-axis GW routes.

The imaginary-frequency and space-time routes, and the contour continuations of
the latter (cd, laplace, sop), take a UHF reference: they build ONE screened
interaction from chi0_alpha + chi0_beta, each spin in its own orbitals, and the
self-energy, static exchange and continuum Eq. (18) shift of the requested spin
channel.

  1. The per-spin static exchange: a closed-shell hybrid carried as UKS gives
     the restricted <Sigma_x - v_xc> in both channels, and a Hartree-Fock
     doublet gives zero.
  2. A closed shell carried as UHF -- the RHF solution itself in both spin
     channels, so nothing but the unrestricted code path differs -- gives the
     restricted quasiparticle energies on every route and continuation, in the
     gas phase and in a continuum.
  3. A doublet agrees with the unrestricted Casida route, which sums the same
     GW@RPA self-energy over its exact RPA poles, within the continuation's
     own error, in the gas phase and in a continuum.
  4. evGW runs on the unrestricted imaginary-axis routes, and reproduces the
     restricted fixed point on the closed shell carried as UHF.
  5. The blocked space-time path, which is built for one spin, is refused by
     name.
  6. The space-time route over 2 and 3 simulated ranks, on the doublet in both
     channels, with and without the omega = 0 row a BSE and a continuum read:
     every rank's quasiparticle energies and W(0) are the serial ones, bit for
     bit -- each reduced sum is a single tile here, so the ranks add exact
     zeros -- and each rank inverts exactly its round-robin frequencies.

Run as a script (`python tests/test_unrestricted_imaginary_axis.py`) or under
pytest.
"""
import copy
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.solvent_screening import (attach_solvent_screening,
                                        detach_solvent_screening)
from src.Base.utils.mpi_grid import partition, run_simulated
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
OH = 'O 0 0 0; H 0 0 0.97'
BASIS = 'cc-pvdz'
AUX = 'cc-pvdz-ri'

ROUTES = [dict(mode='imagfrequency'), dict(mode='space-time'),
          dict(mode='space-time', continuation='cd'),
          dict(mode='space-time', continuation='laplace'),
          dict(mode='space-time', continuation='sop')]
#: the same arithmetic on the same orbitals, reassociated: ~1e-10 eV measured
SAME_ORBITALS_TOL_EV = 1e-7
#: UHF Casida vs the imaginary-axis routes on OH/cc-pVDZ, measured at most
#: 1.5 meV (space-time Pade, which carries the ISDF fit error too), 0.6 meV
#: (cd, laplace) and 0.4 meV (imaginary frequency), gas phase and water alike
DOUBLET_TOL_EV = 3e-3
RANK_SIZES = [2, 3]


def _restricted(atom, mean_field=scf.RHF, **kw):
    mol = gto.M(atom=atom, basis=BASIS, verbose=0)
    return mean_field(mol, **kw).density_fit(auxbasis=AUX).run(conv_tol=1e-11)


def _as_unrestricted(rhf, mean_field=scf.UHF, **kw):
    """The restricted solution in both spin channels of an unrestricted object."""
    uhf = mean_field(rhf.mol, **kw).density_fit(auxbasis=AUX)
    uhf.mo_coeff = np.array([rhf.mo_coeff, rhf.mo_coeff])
    uhf.mo_energy = np.array([rhf.mo_energy, rhf.mo_energy])
    uhf.mo_occ = np.array([rhf.mo_occ / 2, rhf.mo_occ / 2])
    uhf.e_tot, uhf.converged = rhf.e_tot, True
    return uhf


def _energies(mf, states, **kw):
    out = calc_qp_energy(mf, selfenergy='GW', df=True, state=list(states), **kw)
    if isinstance(out, dict):
        out = [out[p]['GW'] for p in states]
    return np.asarray(out, float)


def test_static_exchange_per_spin():
    """Hartree-Fock pins the exchange bookkeeping to round-off; the hybrid
    adds pyscf's spin-polarized functional evaluation, whose v_xc differs from
    the unpolarized one by 4e-11 Ha per AO element on the same density, ~1e-10
    on the diagonal -- a property of the quadrature, not of this code."""
    for mean_fields, kw, tol in (((scf.RHF, scf.UHF), {}, 1e-12),
                                 ((dft.RKS, dft.UKS), dict(xc='pbe0'), 1e-9)):
        restricted = _restricted(WATER, mean_fields[0], **kw)
        unrestricted = _as_unrestricted(restricted, mean_fields[1], **kw)
        states = list(range(restricted.mol.nao))
        for exchange in ('mf', 'df-direct'):
            ref = static_exchange_diagonal(restricted, restricted.mol, states,
                                           exchange=exchange)
            for spin in (0, 1):
                got = static_exchange_diagonal(unrestricted, unrestricted.mol,
                                               states, exchange=exchange,
                                               spin=spin)
                assert np.abs(got - ref).max() < tol, (kw, exchange, spin)
    uks = unrestricted
    doublet = scf.UHF(gto.M(atom=OH, basis='6-31g', spin=1, verbose=0))
    doublet.run(conv_tol=1e-11)
    for spin in (0, 1):
        assert np.abs(static_exchange_diagonal(doublet, doublet.mol, [0, 3, 5],
                                               spin=spin)).max() < 1e-10
    with pytest.raises(ValueError, match='one static exchange per spin'):
        static_exchange_diagonal(uks, uks.mol, states)


@pytest.mark.parametrize('continuum', [False, True])
def test_closed_shell_carried_unrestricted(continuum):
    rhf = _restricted(WATER)
    uhf = _as_unrestricted(rhf)
    if continuum:
        for mf in (rhf, uhf):
            attach_solvent_screening(mf, solvent='water')
    try:
        states = [3, 4, 5]
        for kw in ROUTES:
            ref = _energies(rhf, states, **kw)
            for channel in ('alpha', 'beta'):
                got = _energies(uhf, states, spin_channel=channel, **kw)
                assert np.abs(got - ref).max() < SAME_ORBITALS_TOL_EV, \
                    (kw, channel, got - ref)
    finally:
        for mf in (rhf, uhf):
            detach_solvent_screening(mf)


@pytest.mark.parametrize('continuum', [False, True])
def test_doublet_agrees_with_casida(continuum):
    mol = gto.M(atom=OH, basis=BASIS, spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    if continuum:
        attach_solvent_screening(uhf, solvent='water')
    try:
        na, nb = uhf.nelec
        windows = {'alpha': [na - 2, na - 1, na], 'beta': [nb - 1, nb, nb + 1]}
        for channel, states in windows.items():
            ref = _energies(uhf, states, mode='casida', spin_channel=channel)
            for kw in ROUTES[:4]:
                got = _energies(uhf, states, spin_channel=channel, **kw)
                assert np.abs(got - ref).max() < DOUBLET_TOL_EV, \
                    (continuum, channel, kw, got - ref)
    finally:
        detach_solvent_screening(uhf)


def test_evgw_on_the_unrestricted_routes():
    rhf = _restricted(WATER)
    uhf = _as_unrestricted(rhf)
    for mode in ('imagfrequency', 'space-time'):
        ref = calc_qp_energy(rhf, selfenergy='GW', df=True, state='homo',
                             mode=mode, self_consistency='evGW')
        for channel in ('alpha', 'beta'):
            got = calc_qp_energy(uhf, selfenergy='GW', df=True, state='homo',
                                 mode=mode, self_consistency='evGW',
                                 spin_channel=channel)
            assert abs(got - ref) < 1e-6, (mode, channel, got, ref)


def test_blocked_path_is_refused():
    mol = gto.M(atom=OH, basis=BASIS, spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    with pytest.raises(NotImplementedError, match='built for one spin'):
        solve_qp_energy_space_time(uhf, mol, uhf.nelec, [2], freq_block=4)


def _space_time(uhf, factors, channel, kind, comm=None):
    """(energies, W(0) or an empty array, timings) of the space-time route on
    this rank's own copy of the mean field and factors, which a distributed
    solve overwrites in place with rank 0's: a three-state window without
    extras, or the whole diagonal with W(0) as a BSE takes it."""
    mf = copy.copy(uhf)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = (
        np.array(a, float) for a in (uhf.mo_energy, uhf.mo_coeff, uhf.mo_occ))
    timings = {}
    kw = dict(factors=tuple(np.array(a) for a in factors),
              spin_channel=channel, timings=timings,
              distribute=comm is not None, comm=comm)
    if kind == 'window':
        n = uhf.nelec[channel == 'beta']
        qp = solve_qp_energy_space_time(mf, mf.mol, mf.nelec,
                                        np.array([n - 2, n - 1, n]), **kw)
        return qp, np.zeros(0), timings
    extras = {}
    qp, _ = solve_qp_diagonal_space_time(mf, mf.mol, mf.nelec, extras=extras,
                                         **kw)
    return qp, extras['w_static'], timings


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


@pytest.mark.parametrize('size', RANK_SIZES)
@pytest.mark.parametrize('continuum', [False, True])
def test_space_time_over_ranks(continuum, size):
    mol = gto.M(atom=OH, basis=BASIS, spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    if continuum:
        attach_solvent_screening(uhf, solvent='water')
    try:
        factors = separable_factors(uhf, mol, auxbasis=AUX)
        for channel in ('alpha', 'beta'):
            for kind in ('window', 'diagonal'):
                serial = _space_time(uhf, factors, channel, kind)
                outs = run_simulated(
                    lambda c: _space_time(uhf, factors, channel, kind, c), size)
                nrows = serial[2]['dyson_frequencies']
                passenger = continuum or kind == 'diagonal'
                static = nrows - 1 if passenger else None
                for r, (qp, ws, t) in enumerate(outs):
                    where = (continuum, size, channel, kind, r)
                    assert _bitwise(qp, serial[0]), where
                    assert _bitwise(ws, serial[1]), where
                    mine = [k for k in partition(nrows, r, size)
                            if k != static]
                    assert t['dyson_frequencies'] == len(mine), where
    finally:
        detach_solvent_screening(uhf)


TESTS = [test_static_exchange_per_spin,
         lambda: test_closed_shell_carried_unrestricted(False),
         lambda: test_closed_shell_carried_unrestricted(True),
         lambda: test_doublet_agrees_with_casida(False),
         lambda: test_doublet_agrees_with_casida(True),
         test_evgw_on_the_unrestricted_routes,
         test_blocked_path_is_refused] + [
    (lambda c=c, s=s: test_space_time_over_ranks(c, s))
    for c in (False, True) for s in RANK_SIZES]
NAMES = ['static_exchange_per_spin', 'closed_shell_carried_unrestricted[gas]',
         'closed_shell_carried_unrestricted[water]',
         'doublet_agrees_with_casida[gas]', 'doublet_agrees_with_casida[water]',
         'evgw_on_the_unrestricted_routes', 'blocked_path_is_refused'] + [
    f'space_time_over_ranks[{"water" if c else "gas"}-{s}]'
    for c in (False, True) for s in RANK_SIZES]


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    failed = 0
    for name, test in zip(NAMES, TESTS):
        try:
            test()
            print(f'[OK  ] {name}')
        except (Exception, pytest.fail.Exception) as exc:   # noqa: BLE001
            failed += 1
            print(f'[FAIL] {name} -- {type(exc).__name__}: {exc}')
    print('\nALL PASSED' if not failed else '\nFAILURES DETECTED')
    sys.exit(0 if not failed else 1)
