"""The quasiparticle states of one self-energy are solved a block per rank.

`t_qp` (everything `solve_qp_energy_space_time` spends after Sigma_c is on
the imaginary axis) is two costs. <Sigma_x - v_xc> is one exchange build for
the whole window, state-independent, and stays replicated. The Pade fit of
Sigma_pp(i.omega) and the root search on w = eps_p + <Sigma_x - v_xc>_pp +
Re Sigma_c(w) are per state and share nothing; on a quasiparticle set (the
whole BSE diagonal) that loop is the larger of the two.

The split is over the states, with no reduction: rank r takes states r,
r + nranks, ... and the roots are all-gathered back into state order. Sigma
arrives all-reduced and identical on every rank and the static term is
replicated from rank 0, so each root is the same scalar iteration on the same
numbers wherever it runs, and the gates here are bitwise per state. Ranks
beyond the state count own an empty block and only serve the gather.

The BSE's block action is gated here too, because the same row split carries
its Hartree term (`isdf_block_action`) on the exchange terms' single
reduction. A sum over rows split between ranks is re-associated, so that gate
is 1e-13 relative, not bitwise.

What each gate catches:

  perturbation                           gates that then fail
  a rank solves the state next to its    all nine state gates (the window,
  own (`states[(i + 1) % n]` under a     bitwise and blocked, the
  partition; serial untouched)           oversubscribed one and the whole
                                         diagonal), and the BSE through its
                                         own refusal of an unstable reference
  the allgather made a no-op (the        the same nine, with the unowned
  collective still called, its result    states arriving as None
  discarded, so the ranks stay in step)
  the Hartree term reduced unsplit       test_block_action_row_split, and the
  (the full row range on every rank,     BSE roots. The three triplet cases
  then all-reduced, so the term is       still pass: the slot carries the
  counted nranks times)                  kappa = 2 term and nothing else

The adaptive explicit set over 2 and 3 simulated ranks selects the serial
partition, with its explicit roots bitwise across ranks.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, lib, scf

from src.Base.declaration import QPStates
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_block_action,
                                                         isdf_bse_factors,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.gradients.excited_state import ExcitedStateChain

SIZES = [2, 3]
#: More ranks than states: the surplus ranks own an empty block.
OVERSUBSCRIBED = 7
#: The row split re-associates a sum over grid rows, so the block action and
#: what a Davidson converges from it agree to the last bits, not in them.
ROW_SPLIT_REL = 1e-13
#: A Davidson stops at conv_tol=1e-5, and a last-bit change in its action moves
#: the converged root by more than the action itself moved.
ROOT_REL = 1e-11
#: The explicit roots of an adaptive set over ranks against serial, Ha: the
#: tau sweep's reduction re-associates, the roots' Newton does not.
ADAPTIVE_ROOT_TOL = 1e-10
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors,
                window=np.arange(nocc - 2, nocc + 3))     # 5 states


def relative(a, b):
    """max|a - b| over the larger scale, 0 when both vanish."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.abs(a).max(), np.abs(b).max())
    return 0.0 if scale == 0 else float(np.abs(a - b).max()) / scale


@pytest.mark.parametrize('size', SIZES)
def test_window_states_bitwise(water, size):
    """A window split over ranks is the serial window, state by state."""
    w = water
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], w['window'],
                                     factors=w['factors'])

    def one_rank(comm):
        return solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'],
                                          w['window'], factors=w['factors'],
                                          distribute=True, comm=comm)

    for qp in run_simulated(one_rank, size):
        assert qp.shape == ref.shape
        for i, p in enumerate(w['window']):
            assert qp[i] == ref[i], f'state {p} moved off the serial root'


@pytest.mark.parametrize('size', SIZES)
def test_blocked_window_states_bitwise(water, size):
    """The low-memory branch (`_qp_blocked`) carries the same state split."""
    w = water
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], w['window'],
                                     factors=w['factors'], freq_block=3)

    def one_rank(comm):
        return solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'],
                                          w['window'], factors=w['factors'],
                                          freq_block=3, distribute=True,
                                          comm=comm)

    for qp in run_simulated(one_rank, size):
        assert np.array_equal(qp, ref)


def test_more_ranks_than_states(water):
    """Seven ranks, four states: the surplus ranks return the whole array too.

    `partition` hands them an empty block, which is a legitimate configuration
    rather than an error -- they compute nothing and serve the gather.
    """
    w = water
    states = np.arange(w['nocc'] - 2, w['nocc'] + 2)          # 4 states
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], states,
                                     factors=w['factors'])

    def one_rank(comm):
        return solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], states,
                                          factors=w['factors'],
                                          distribute=True, comm=comm)

    out = run_simulated(one_rank, OVERSUBSCRIBED)
    for qp in out:
        assert qp.shape == (len(states),)
        assert np.array_equal(qp, ref)


@pytest.mark.parametrize('size', SIZES)
def test_qp_diagonal_bitwise(water, size):
    """The whole diagonal, which is what the BSE asks for and where the
    per-state loop is the dominant half of `t_qp`."""
    w = water
    ref, _ = solve_qp_diagonal_space_time(w['mf'], w['mol'], w['nocc'],
                                          factors=w['factors'])

    def one_rank(comm):
        eps, _ = solve_qp_diagonal_space_time(w['mf'], w['mol'], w['nocc'],
                                              factors=w['factors'],
                                              distribute=True, comm=comm)
        return eps

    for eps in run_simulated(one_rank, size):
        assert np.array_equal(eps, ref)


@pytest.mark.parametrize('size', SIZES)
def test_bse_diagonal_and_roots(water, size):
    """The BSE built on that diagonal: the quasiparticle energies bitwise, the
    roots and min eig(A-B) to the last bits of the row-split action."""
    w = water
    om0, _, _, info0 = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=3,
                                      probe=True, progress=False,
                                      factors=w['factors'])

    def one_rank(comm):
        om, _, _, info = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=3,
                                        probe=True, progress=False,
                                        factors=w['factors'], distribute=True,
                                        comm=comm)
        return om, info

    for om, info in run_simulated(one_rank, size):
        assert np.array_equal(info['eps'], info0['eps'])      # the GW diagonal
        assert relative(om, om0) < ROOT_REL
        assert relative([info['min_eig_amb']], [info0['min_eig_amb']]) < ROOT_REL
        assert info['nranks'] == size


@pytest.mark.parametrize('size', SIZES + [OVERSUBSCRIBED])
@pytest.mark.parametrize('mode,spin', [('BSE', 'singlet'), ('BSE', 'triplet'),
                                       ('TDHF', 'singlet'), ('RPA', 'singlet')])
def test_block_action_row_split(water, size, mode, spin):
    """A/B from the row-split action, against the serial one.

    Both exchange terms and the Hartree term are sums over the grid rows,
    and all three ride one reduction of the batch. RPA is the case with no
    exchange at all, where that reduction exists for the Hartree term alone;
    the triplet is the case with no Hartree term, where it does not exist for
    it. The ranks must also agree with EACH OTHER bitwise -- they reduce, so
    they cannot do otherwise, and a rank that did would be following a
    different iteration.
    """
    w = water
    eps = np.asarray(w['mf'].mo_energy, float)
    lbse = mode != 'RPA'
    W_aux = (isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                              factors=w['factors'])[2] if mode == 'BSE'
             else None)
    lr = LinearResponseSolver(eps, spin_mode='restricted')
    z = np.random.default_rng(1).normal(
        size=(3, w['nocc'], len(eps) - w['nocc']))
    action, _ = isdf_block_action(lr, w['nocc'], lbse, W_aux, w['factors'],
                                  spin=spin)
    A0, B0 = action(z)

    def one_rank(comm):
        act, _ = isdf_block_action(lr, w['nocc'], lbse, W_aux, w['factors'],
                                   spin=spin, comm=comm)
        return act(z)

    out = run_simulated(one_rank, size)
    for A, B in out:
        assert relative(A, A0) < ROW_SPLIT_REL
        assert relative(B, B0) < ROW_SPLIT_REL
        assert np.array_equal(A, out[0][0]) and np.array_equal(B, out[0][1])


def adaptive_scf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def adaptive_partition():
    """(explicit, tier_of, explicit roots) of an adaptive set selected on
    this rank's own Mole and mean field."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = adaptive_scf(mol)
    eps = np.asarray(mf.mo_energy, float)
    window = resolve_qp_states(QPStates('adaptive'), eps, mol.nelectron // 2,
                               degeneracy_tol=1e-4).explicit
    chain = ExcitedStateChain(
        mol, adaptive_scf, qp_window=list(window), residue_route='sop',
        scissor='calibrate', outside='scissor', solver='dense', mf=mf,
        qp_select=QPStates('adaptive', targets=(('singlet', 0),
                                                ('triplet', 0))))
    chain.energy(mol, mf)
    part = chain.qp_partition
    return part.explicit, part.tier_of, np.array(
        [chain.qp_seeds[p] for p in part.explicit])


@pytest.mark.parametrize('size', SIZES)
def test_adaptive_partition_over_ranks(size):
    """The continuation over ranks re-associates its sums, and the partition
    it selects does not move: the same explicit set and tier map on every
    rank and at every rank count, the explicit roots bitwise across ranks."""
    threads = lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(1)
        try:
            explicit0, tiers0, roots0 = adaptive_partition()
            out = run_simulated(lambda comm: adaptive_partition(), size)
        finally:
            lib.num_threads(threads)
    assert tiers0, 'the gate needs holes'
    for explicit, tiers, roots in out:
        assert explicit == explicit0 and tiers == tiers0
        assert np.abs(roots - roots0).max() < ADAPTIVE_ROOT_TOL
        assert np.array_equal(roots, out[0][2])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
