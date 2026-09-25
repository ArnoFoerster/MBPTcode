"""The quasiparticle states of one self-energy are solved a block per rank.

`t_qp` -- everything `solve_qp_energy_space_time` spends after Sigma_c is on
the imaginary axis -- is two costs. <Sigma_x - v_xc> is ONE exchange build for
the whole window, state-independent, and it stays replicated. The Pade fit of
Sigma_pp(i.omega) and the root search on w = eps_p + <Sigma_x - v_xc>_pp +
Re Sigma_c(w) are per state and share nothing at all, and on a quasiparticle
SET -- the whole BSE diagonal, which `solve_bse_isdf` asks for -- that loop is
the larger of the two: 24 states of water/cc-pVDZ are 0.59 s of a 0.59 s
`t_qp`, ~25 ms each, and the cost per state barely follows the system size
(150 evaluations of a 16-node continued fraction, whatever the molecule).

The split is therefore over the STATES, with no reduction: rank r takes states
r, r + nranks, ... and the roots are all-gathered back into state order. Sigma
arrives all-reduced and identical on every rank and the static term is
replicated from rank 0, so each root is the same scalar iteration on the same
numbers wherever it runs, and the gates here are BITWISE -- per state, not on a
norm. Ranks beyond the state count own an empty block and only serve the
gather.

The BSE's block action is gated here too, because the same row split now
carries its HARTREE term (`isdf_block_action`): its last contraction is a sum
over the grid rows, it was the one piece of the action a rank count did not
divide (7.7% of a serial action at the anthracene/cc-pVTZ shapes, 19% of a
four-rank one), and it rides the exchange terms' single reduction. That one is
NOT bitwise against serial and cannot be: a sum over rows split between ranks
is re-associated. It is gated at 1e-13 relative, and measures 1.3e-16.

EVERY GATE HERE WAS SHOWN TO FAIL, on a backup of the file, restored and
`cmp`-verified afterwards:

  perturbation                           gates that then fail
  a rank solves the state next to its    all nine state gates -- the window
  own (`states[(i + 1) % n]` under a     (bitwise and blocked), the
  partition; serial untouched)           oversubscribed one and the whole
                                         diagonal -- 7.0e-2 Ha out on the
                                         first state, and the BSE through its
                                         OWN refusal: min eig(A-B) comes back
                                         -21.3 Ha and the driver reports an
                                         unstable reference, the defect
                                         wearing the mask of physics
  the allgather made a no-op (the        the same nine, with the unowned
  collective still called, its result    states arriving as None -- an object
  discarded, so the ranks stay in step)  array where a float one is compared
  the Hartree term reduced UNSPLIT       test_block_action_row_split, 1.3e-2
  (the full row range on every rank,     to 7.5e-2 relative, and the BSE roots
  then all-reduced, so the term is       4.7e-2 to 7.9e-2. The three TRIPLET
  counted nranks times)                  cases still pass, which is what says
                                         the slot carries the kappa = 2 term
                                         and nothing else
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_block_action,
                                                         isdf_bse_factors,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver

SIZES = [2, 3]
#: More ranks than states: the surplus ranks own an empty block.
OVERSUBSCRIBED = 7
#: The row split re-associates a sum over grid rows, so the block action and
#: what a Davidson converges from it agree to the last bits, not in them.
ROW_SPLIT_REL = 1e-13
#: A Davidson stops at conv_tol=1e-5, and a last-bit change in its action moves
#: the converged root by more than the action itself moved.
ROOT_REL = 1e-11


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

    Both exchange terms and now the Hartree term are sums over the grid rows,
    and all three ride ONE reduction of the batch. RPA is the case with no
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


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
