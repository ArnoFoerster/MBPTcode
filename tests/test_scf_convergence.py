"""How a mean field was converged: the per-cycle record of `SCFHistory`,
and the warm start of a walk's next SCF from its last converged density.

A warm start changes where the loop starts and nothing it converges to: the
mean field it reaches is the cold start's to the tolerances both ran to
(conv_tol 1e-12 Ha, conv_tol_grad 1e-9: energies within 1e-10 Ha, orbital
energies within |g| / gap ~ 1e-8 Ha, gated at 1e-7), in fewer cycles.

The record is read off pyscf's own `callback`, which sees every cycle's
locals and is called after the cycle's arithmetic, so attaching it moves no
number: the mean field converged with it is the one converged without it,
bit for bit (two runs of one serial SCF on one machine repeat their bits).
Under ranks every rank runs pyscf's loop and every rank's history holds the
same cycles; the converged arrays are rank 0's (`distributed_mean_field`).
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import SCF_HISTORY_GRAD_MARKS  # noqa: E402
from src.Base.distributed_df import distributed_mean_field  # noqa: E402
from src.Base.isdf_jk import isdf_jk  # noqa: E402
from src.Base.scf_convergence import (SCFHistory, WarmStart, newton_finished,  # noqa: E402
                                      scf_record, warm_started)
from src.Base.separable_ri import resolve_isdf_grid  # noqa: E402
from src.Base.utils.mpi_grid import run_simulated  # noqa: E402

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4615'


def water():
    """Water, slightly asymmetric, cc-pVDZ."""
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def built(mol, conv_tol=1e-12, conv_tol_grad=1e-9):
    """A density-fitted PBE0 mean field, built and not run."""
    mf = dft.RKS(mol, xc='pbe0').density_fit()
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = conv_tol, conv_tol_grad, 100
    return mf


def test_history_moves_no_number():
    """With and without the callback: the same energy and orbitals bit for
    bit, a history of exactly pyscf's cycles, ending below conv_tol_grad."""
    plain = built(water())
    plain.kernel()
    watched = built(water())
    history = SCFHistory()
    history.attach(watched)
    watched.kernel()
    assert watched.e_tot == plain.e_tot
    assert np.array_equal(watched.mo_coeff, plain.mo_coeff)
    (run,) = history.summary()
    assert run['cycles'] == watched.cycles == plain.cycles
    assert run['converged'] is True
    assert run['final_grad_norm'] < watched.conv_tol_grad
    cycles = [c[0] for c in run['history']]
    assert cycles == list(range(1, watched.cycles + 1))
    firsts = [run['first_cycle_below'][f'{m:.0e}'] for m in
              SCF_HISTORY_GRAD_MARKS if m >= watched.conv_tol_grad]
    assert all(f is not None for f in firsts) and firsts == sorted(firsts)
    rec = scf_record(watched)
    assert rec['pyscf_cycles'] == watched.cycles and rec['converged']
    assert rec['distributed_timings'] is None


def test_history_keeps_a_callback_it_finds():
    """A callback already on the mean field is still called every cycle."""
    mf = built(water())
    seen = []
    mf.callback = lambda envs: seen.append(envs['cycle'])
    history = SCFHistory()
    history.attach(mf)
    mf.kernel()
    assert len(seen) == mf.cycles == history.summary()[0]['cycles']


def displaced(shift=0.05):
    """Water with its oxygen moved by `shift` Angstrom, an optimizer step."""
    mol = water()
    coords = mol.atom_coords(unit='Angstrom')
    coords[0, 1] += shift
    return gto.M(atom=[(mol.atom_symbol(i), coords[i]) for i in
                       range(mol.natm)], basis='cc-pvdz', verbose=0)


def test_warm_start_is_the_same_mean_field_in_fewer_cycles():
    """From the previous geometry's density: the SCF at the displaced
    geometry lands on the cold start's mean field within the convergence it
    was run to (1e-10 Ha, orbital energies 1e-7 Ha at conv_tol_grad 1e-9)
    and takes fewer cycles; a holder that has seen no SCF leaves the guess
    to pyscf, bit for bit."""
    holder = WarmStart()
    first = warm_started(built(water()), holder)
    first.kernel()
    reference = built(water())
    reference.kernel()
    # nothing held yet: pyscf's own guess, and so pyscf's own run
    assert first.e_tot == reference.e_tot and holder.declined == 1
    warm = warm_started(built(displaced()), holder)
    warm.kernel()
    cold = built(displaced())
    cold.kernel()
    assert holder.used == 1
    assert abs(warm.e_tot - cold.e_tot) < 1e-10
    assert np.abs(warm.mo_energy - cold.mo_energy).max() < 1e-7
    assert warm.cycles < cold.cycles


def test_warm_start_declines_a_moved_frame():
    """A geometry further than SCF_WARM_START_MAX_SHIFT from the held one
    (here the molecule translated by 1 Bohr) starts from pyscf's guess."""
    holder = WarmStart()
    warm_started(built(water()), holder).kernel()
    mol = water()
    far = gto.M(atom=[(mol.atom_symbol(i), mol.atom_coord(i) + 1.0)
                      for i in range(mol.natm)], basis='cc-pvdz', unit='Bohr',
                verbose=0)
    # the first SCF found the holder empty, the second the frame moved
    assert holder.guess(far) is None and holder.declined == 2


@pytest.mark.parametrize('size', [2, 3])
def test_every_rank_records_the_same_cycles(size):
    """Distributed over simulated ranks: each rank's history holds pyscf's
    cycles, the same count and the same |g| on every rank, and the record
    carries the distributed SCF's stage clock."""
    def one_rank(comm):
        mf = built(water())
        history = SCFHistory()
        history.attach(mf)
        distributed_mean_field(mf)
        (run,) = history.summary()
        return (run['cycles'], mf.cycles, [c[2] for c in run['history']],
                scf_record(mf)['distributed_timings']['scf_cycles'])

    out = run_simulated(one_rank, size)
    cycles, pyscf_cycles, grads, timed = out[0]
    assert cycles == pyscf_cycles == timed
    for rank in out[1:]:
        assert rank[0] == cycles and rank[2] == grads


def tight(mf):
    """`mf` converged by DIIS to |g| < 1e-11 with a non-binding energy test."""
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-11, 200
    return mf


def isdf_water(xc):
    """Water on the ISDF-K mean field at its G1 grid."""
    mol = water()
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['H', 'O'],
                                        auxbasis='cc-pvdz-ri')
    mf = (scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc))
    return isdf_jk(mf, auxbasis='cc-pvdz-ri', counts=counts, n_start=n_start)


def density_fitted(xc):
    """Water on a density-fitted mean field."""
    mol = water()
    return (scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)).density_fit()


def same_mean_field(a, b):
    """Two SCFs each converged to |g| < 1e-11 are one mean field to that:
    energies within 1e-10 Ha (second order in |g|, plus rounding), densities
    within 1e-8 and orbital energies within 1e-8 Ha (first order, |g| over
    the gap)."""
    assert abs(a.e_tot - b.e_tot) < 1e-10
    assert np.abs(a.make_rdm1() - b.make_rdm1()).max() < 1e-8
    assert np.abs(a.mo_energy - b.mo_energy).max() < 1e-8


def max_fia(mf):
    """max |F_ia| of `mf`'s own Fock matrix in its orbitals."""
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    return float(np.abs((c.T @ mf.get_fock() @ c)[:nocc, nocc:]).max())


@pytest.mark.parametrize('route', ['df', 'isdf'])
@pytest.mark.parametrize('xc', ['hf', 'pbe0', 'lrc-wpbeh'])
def test_newton_finish_is_the_diis_mean_field(route, xc):
    """DIIS to 1e-7 and Newton to 1e-11 land on DIIS-to-1e-11's mean field,
    with canonical orbitals (F_ij and F_ab diagonal) and max |F_ia| at the
    gradient's level, in fewer DIIS cycles."""
    build = isdf_water if route == 'isdf' else density_fitted
    diis = tight(build(xc))
    SCFHistory().attach(diis)
    diis.kernel()
    newton = newton_finished(tight(build(xc)))
    history = SCFHistory()
    history.attach(newton)
    newton.kernel()
    assert newton.converged and newton._newton_finish['converged']
    runs = history.summary()
    assert [r['phase'] for r in runs] == ['diis', 'newton']
    assert runs[-1]['final_grad_norm'] < 1e-11
    assert runs[0]['cycles'] < diis.cycles
    same_mean_field(newton, diis)
    assert max_fia(newton) < 1e-11
    c = newton.mo_coeff
    fmo = c.T @ newton.get_fock() @ c
    nocc = int(np.count_nonzero(newton.mo_occ > 0))
    for block in (slice(0, nocc), slice(nocc, None)):
        sub = fmo[block, block]
        assert np.abs(sub - np.diag(np.diag(sub))).max() < 1e-10


@pytest.mark.parametrize('route', ['df', 'isdf'])
@pytest.mark.parametrize('size', [1, 2, 3])
def test_newton_finish_over_ranks(route, size):
    """The finish on the distributed handles (DistributedDF,
    DistributedISDFJK) at 1, 2 and 3 simulated ranks: every rank ends on the
    same steps and Hessian products, and the mean field is the serial
    finish's to the convergence both ran to."""
    build = isdf_water if route == 'isdf' else density_fitted

    def one_rank(comm):
        mf = newton_finished(tight(build('lrc-wpbeh')))
        distributed_mean_field(mf)
        return mf

    serial = newton_finished(tight(build('lrc-wpbeh')))
    serial.kernel()
    out = run_simulated(one_rank, size)
    for mf in out:
        assert mf.converged
        assert mf._newton_finish['steps'] == out[0]._newton_finish['steps']
        assert (mf._newton_finish['hessian_products']
                == out[0]._newton_finish['hessian_products'])
        assert mf.e_tot == out[0].e_tot
        same_mean_field(mf, serial)


@pytest.mark.parametrize('size', [2, 3])
def test_warm_start_over_ranks(size):
    """Each rank warm-starts from its own holder: every rank takes the guess,
    the ranks agree on the cycles, and the warm mean field is the serial
    cold one within the SCF's convergence (1e-10 Ha)."""
    def one_rank(comm):
        holder = WarmStart()
        distributed_mean_field(warm_started(built(water()), holder))
        mf = warm_started(built(displaced()), holder)
        distributed_mean_field(mf)
        return holder.used, mf.cycles, mf.e_tot

    cold = built(displaced())
    cold.kernel()
    out = run_simulated(one_rank, size)
    assert all(rank[0] == 1 and rank[1] == out[0][1] for rank in out)
    assert out[0][1] < cold.cycles
    assert abs(out[0][2] - cold.e_tot) < 1e-10


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
