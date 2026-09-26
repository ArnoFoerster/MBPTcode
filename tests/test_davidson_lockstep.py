"""The BSE Davidson runs on every rank alike; MPI lives inside the block action.

`isdf_block_action` locksteps its trial vectors at entry (`mpi_grid.lockstep`:
rank 0's batch written into every rank's, in place) and gathers or all-reduces
every term it returns, so its output is the same on every rank by
construction. `_run_davidson` and the (A-B) probe therefore run replicated,
with no protocol between the ranks, and the converged roots and vectors are
lockstepped once at the end.

Gated here, on water/cc-pVDZ BSE@HF at 2, 3 and 8 simulated ranks:

  * every rank returns rank 0's roots, vectors and probe value bitwise, and ran
    the same iteration: the same cycle and probe matvec counts and the same
    lockstep volume, all of it proven equal by the checked locksteps' digests
    and none of it broadcast. Against the SERIAL roots they sit within
    ROOT_TOL, not on them: the row split re-associates the sums it reduces;
  * within what the Davidson resolves of the one-rank roots
    (`roots_resolution`, 1.1e-11 of |roots| here; measured 1.0e-13, 9.1e-14
    and 8.2e-14 at 2, 3 and 8 ranks) after the one-rank iteration, trial
    vector for trial vector; and the grid reduce-scatter
    (`reduce_scatter_rows`) handing every rank the rows one off its block
    breaks the action so far that the Davidson refuses the solve;
  * a trial vector perturbed by one ulp on the last rank changes nothing
    anywhere, bitwise, and `lockstep_stats` under `audit=True` counts exactly
    that one repair on exactly that rank -- the block action handed the
    perturbed batch directly, and a whole solve with the perturbation planted
    in the driver's own buffer in one iteration;
  * the probe's Lanczos, which replaces ARPACK over ranks (eigsh holds one
    process-wide lock across its whole iteration, matvecs included, so rank
    threads cannot all be inside it), meets dense eigenvalues through thick
    restarts, a degenerate lowest one included, and ARPACK's value and sign
    certificate on the BSE;
  * kernels called with no comm inside a `distributed` region follow it, and
    the DF route, whose action is not divided, runs whole on every rank:
    `solve_bse_df` in a region at 2 and 3 ranks returns the SERIAL roots and
    vectors bitwise on every rank (its action locksteps its inputs and
    re-associates nothing), and a one-ulp perturbation of rank 1's batch is
    repaired at the action's entry.

Removing the entry lockstep of `isdf_block_action` fails both ISDF
perturbation gates (the action's output then differs on the perturbed rank,
and the planted ulp moves the returned roots), and removing that of
`df_block_action` fails the DF one on its audited repair, which is what makes
them gates.
"""
import copy
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import distributed, lockstep_stats, run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (
    _lanczos_lowest, isdf_bse_factors, isdf_block_action, lowest_amb_eigenvalue,
    solve_bse_df, solve_bse_isdf, static_screening_matrix)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.test_distributed_fit_mpi import relative, roots_resolution

SIZES = [2, 3, 8]
NROOTS = 5
#: Ha. The row split sums the ranks' partials in rank order instead of inside
#: one GEMM; measured 9.7e-14 at 2 ranks, 9.5e-14 at 3, 7.6e-14 at 8.
ROOT_TOL = 1e-11
#: Ha. The replicated probe's Lanczos against the serial ARPACK value, 6.2e-14
#: at every size: ARPACK sits 6.6e-14 from the dense eigenvalue, the Lanczos
#: 3.4e-15.
PROBE_TOL = 1e-12
#: Relative to the largest eigenvalue: the Lanczos against dense eigh on model
#: spectra, which it meets at 7.6e-15 of max|lambda| = 30.
DENSE_REL = 1e-13
#: The action application the solve-level perturbation is planted in: the
#: second iteration's X block, past the guess.
PLANT_AT = 3
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(mol=mol, mf=mf, nocc=nocc, factors=factors, W_aux=W_aux,
                eps=np.asarray(mf.mo_energy, float))


def rank_copy(w):
    """This rank's own mean field and factors: a distributed solve replicates
    rank 0's over them in place, and simulated ranks share one process."""
    mf = copy.copy(w['mf'])
    mf.mo_energy = np.asarray(w['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(w['mf'].mo_coeff, float).copy()
    return mf, tuple(np.array(a, copy=True) for a in w['factors'])


def solve_df(w, **kw):
    """BSE@HF by the DF route on this rank's own mean field, `qp=False`, with
    no comm: inside a `distributed` region it follows the region."""
    return solve_bse_df(rank_copy(w)[0], w['mol'], w['nocc'], nroots=NROOTS,
                        qp=False, progress=False, **kw)


def solve(w, comm=None, **kw):
    """BSE@HF on this rank's own copies, `qp=False`: the quasiparticle stage
    is other kernels' business, and the Davidson is this file's."""
    mf, factors = rank_copy(w)
    kw.setdefault('distribute', comm is not None)
    return solve_bse_isdf(mf, w['mol'], w['nocc'], nroots=NROOTS, qp=False,
                          progress=False, factors=factors, comm=comm, **kw)


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def one_ulp_up(z):
    """z with its largest-magnitude element moved one ulp, in place; the ulp."""
    i = np.unravel_index(np.abs(z).argmax(), z.shape)
    before = z[i]
    z[i] = np.nextafter(before, np.inf)
    return abs(z[i] - before)


@pytest.fixture(scope='module')
def serial(water):
    return solve(water)


@pytest.fixture(scope='module')
def ranks(water):
    """Every rank's (omega, X, Y, info) at each simulated size."""
    return {size: run_simulated(lambda comm: solve(water, comm), size)
            for size in SIZES}


@pytest.mark.parametrize('size', SIZES)
def test_every_rank_returns_rank_zeros_solve(serial, ranks, size):
    """Rank 0's roots, vectors and probe value on every rank, bitwise, from
    the same iteration; the roots at the row split's distance from serial."""
    om_s, _, _, info_s = serial
    out = ranks[size]
    om0, X0, Y0, info0 = out[0]
    assert np.abs(om0 - om_s).max() <= ROOT_TOL
    assert abs(info0['min_eig_amb'] - info_s['min_eig_amb']) <= PROBE_TOL
    # The trial-vector and result locksteps are checked: thread-ranks repeat
    # each other's bits, so the digests agree and the whole volume is skipped.
    assert info0['timings']['davidson_lockstep_skipped_mb'] > 0.0
    assert info0['timings']['davidson_lockstep_mb'] == 0.0
    for om, X, Y, info in out:
        assert bitwise(om, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        assert info['min_eig_amb'] == info0['min_eig_amb']
        assert info['nranks'] == size
        for key in ('davidson_vind_calls', 'davidson_block_actions',
                    'probe_matvecs'):
            assert info['stats'][key] == info0['stats'][key], key
        for key in ('davidson_lockstep_mb', 'davidson_lockstep_skipped_mb'):
            assert info['timings'][key] == info0['timings'][key], key


@pytest.mark.parametrize('size', SIZES)
def test_the_roots_are_one_ranks_to_the_davidsons_resolution(water, serial,
                                                             ranks, size):
    """The re-associated grid reduce-scatter and output reduction leave the
    roots within what the Davidson resolves of the one-rank solve, after the
    one-rank iteration."""
    om_s, _, _, info_s = serial
    om0, _, _, info0 = ranks[size][0]
    bar = roots_resolution(info_s['eps'], water['nocc'], om_s)
    assert relative(om0, om_s) <= bar
    for key in ('davidson_vind_calls', 'davidson_block_actions'):
        assert info0['stats'][key] == info_s['stats'][key], key
    assert (info0['timings']['davidson_vectors_applied']
            == info_s['timings']['davidson_vectors_applied'])


def test_a_wrong_row_block_breaks_the_solve(water, monkeypatch):
    """Every rank handed the grid reduction's rows one off its block: the
    action is no longer the symmetric Casida one and the Davidson refuses."""
    real = davidson.reduce_scatter_rows

    def one_row_off(a, comm, out=None):
        return real(np.roll(a, 1, axis=0), comm, out=out)

    monkeypatch.setattr(davidson, 'reduce_scatter_rows', one_row_off)
    with pytest.raises(RuntimeError, match='Davidson failed'):
        run_simulated(lambda comm: solve(water, comm, probe=False), 3)


@pytest.mark.parametrize('size', SIZES)
def test_the_action_repairs_one_ulp_on_one_rank(water, size):
    """The last rank hands the action a batch one ulp off rank 0's: every
    rank's A and B are the unperturbed ones, bitwise, the perturbed rank's own
    buffer holds rank 0's numbers afterwards, and the audit counts that one
    repair there and nowhere else."""
    w = water
    lr = LinearResponseSolver(w['eps'], spin_mode='restricted')
    no = w['nocc']
    z = np.random.default_rng(7).normal(size=(3, no, len(w['eps']) - no))
    target = size - 1

    def one_rank(comm, perturb):
        act, _ = isdf_block_action(lr, no, True, w['W_aux'], w['factors'],
                                   comm=comm)
        mine, ulp = z.copy(), 0.0
        if perturb and comm.Get_rank() == target:
            ulp = one_ulp_up(mine)
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            A, B = act(mine)
            stats = lockstep_stats()
        return A, B, mine, ulp, stats

    clean = run_simulated(one_rank, size, False)
    moved = run_simulated(one_rank, size, True)
    A0, B0 = clean[0][0], clean[0][1]
    assert moved[target][3] > 0.0                  # the perturbation was made
    for r, (A, B, mine, ulp, stats) in enumerate(moved):
        assert bitwise(A, A0) and bitwise(B, B0)
        assert bitwise(mine, z)                    # repaired in place
        assert stats['audited_calls'] == 1
        assert stats['mismatched_calls'] == (1 if r == target else 0)
        assert stats['max_abs_diff'] == ulp


@pytest.mark.parametrize('size', SIZES)
def test_one_ulp_planted_in_one_iteration_moves_no_bit(water, ranks,
                                                       monkeypatch, size):
    """The last rank's DAVIDSON buffer is moved one ulp before the action of
    one iteration sees it: the lockstep at the action's entry repairs it, so
    every rank returns the unperturbed solve bitwise, and the audit counts
    one mismatch, on that rank."""
    target = size - 1
    fired = []
    real = davidson.isdf_block_action

    def planting(*args, **kwargs):
        act, diag = real(*args, **kwargs)
        comm = kwargs.get('comm')
        calls = [0]

        def action(z):
            calls[0] += 1
            if (comm is not None and comm.Get_rank() == target
                    and calls[0] == PLANT_AT):
                fired.append(one_ulp_up(z))        # the driver's own buffer
            return act(z)
        return action, diag

    def one_rank(comm):
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = solve(water, comm, probe=False)
            return out, lockstep_stats()

    monkeypatch.setattr(davidson, 'isdf_block_action', planting)
    moved = run_simulated(one_rank, size)
    assert fired and fired[0] > 0.0, 'the perturbation was never planted'
    om0, X0, Y0, _ = ranks[size][0]
    for r, ((om, X, Y, _), stats) in enumerate(moved):
        assert bitwise(om, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        assert stats['mismatched_calls'] == (1 if r == target else 0)
        assert stats['max_abs_diff'] == (fired[0] if r == target else 0.0)


@pytest.mark.parametrize('n,degenerate,k', [(150, False, 1), (150, False, 3),
                                            (150, True, 1), (400, False, 1),
                                            (400, False, 3)])
def test_the_lanczos_meets_dense_eigenvalues(n, degenerate, k):
    """Model spectra wider than one Lanczos basis, so the thick restart runs;
    a doubly degenerate lowest eigenvalue among them, asked for alone: one
    start vector spans one direction of a degenerate eigenspace, so a second
    copy of it is not a Krylov method's to find, ARPACK's included."""
    rng = np.random.default_rng(n + k)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.sort(rng.uniform(0.1, 30.0, n))
    if degenerate:
        lam[1] = lam[0]
    a = (q * lam) @ q.T
    theta, x = _lanczos_lowest(lambda v: a @ v, np.ones(n), k,
                               np.sqrt(np.finfo(float).eps))
    assert theta.shape == (k,) and x.shape == (k, n)
    assert np.abs(theta - lam[:k]).max() <= DENSE_REL * lam.max()
    resid = np.linalg.norm(x @ a - theta[:, None] * x, axis=1)
    assert np.all(resid <= np.sqrt(np.finfo(float).eps) * np.abs(theta))


@pytest.mark.parametrize('size', [2, 3])
def test_the_replicated_probe_proves_arpacks_sign(water, size):
    """The sign-only probe over ranks: the same certificate on every rank,
    and the value ARPACK's serial probe proves positive."""
    w = water
    lr = LinearResponseSolver(w['eps'], spin_mode='restricted')
    ref = lowest_amb_eigenvalue(lr, w['nocc'], 'BSE', w['W_aux'], w['factors'])

    def one_rank(comm):
        stats = {}
        value = lowest_amb_eigenvalue(lr, w['nocc'], 'BSE', w['W_aux'],
                                      w['factors'], sign_only=True,
                                      stats=stats, comm=comm)
        return value, stats

    out = run_simulated(one_rank, size)
    for value, stats in out:
        assert bitwise(value, out[0][0])
        assert stats['probe_sign_proven']
        assert stats['probe_matvecs'] == out[0][1]['probe_matvecs']
        assert value[0] - stats['probe_residual'] > 0.0 and ref[0] > 0.0


def test_the_kernels_follow_the_distributed_region(water, ranks):
    """No comm and no `distribute` inside a region: the solve, and the static
    W underneath it, run over the region's communicator exactly as with the
    communicator passed."""
    w, size = water, 2
    X, D = w['factors'][:2]
    explicit = run_simulated(
        lambda comm: static_screening_matrix(X, D, w['eps'], w['nocc'],
                                             comm=comm), size)
    implied = run_simulated(
        lambda comm: static_screening_matrix(X, D, w['eps'], w['nocc']), size)
    for a, b in zip(explicit, implied):
        assert bitwise(a, b)
    om0, X0, Y0, _ = ranks[size][0]
    for om, Xr, Yr, info in run_simulated(
            lambda comm: solve(water, distribute=None), size):
        assert info['nranks'] == size
        assert bitwise(om, om0) and bitwise(Xr, X0) and bitwise(Yr, Y0)


@pytest.fixture(scope='module')
def df_serial(water):
    return solve_df(water)


@pytest.mark.parametrize('size', [2, 3])
def test_the_df_route_runs_whole_on_every_rank(water, df_serial, size):
    """`solve_bse_df` inside a region: every rank returns the serial roots and
    vectors bitwise, from the serial number of cycles, and one probe value."""
    om_s, X_s, Y_s, info_s = df_serial
    out = run_simulated(lambda comm: solve_df(water), size)
    amb0 = out[0][3]['min_eig_amb']
    assert abs(amb0 - info_s['min_eig_amb']) <= PROBE_TOL
    for om, X, Y, info in out:
        assert bitwise(om, om_s) and bitwise(X, X_s) and bitwise(Y, Y_s)
        assert (info['stats']['davidson_vind_calls']
                == info_s['stats']['davidson_vind_calls'])
        assert info['min_eig_amb'] == amb0
        assert info['timings']['davidson_lockstep_skipped_mb'] > 0.0
        assert info['timings']['davidson_lockstep_mb'] == 0.0


@pytest.mark.parametrize('size', [2, 3])
def test_the_df_action_repairs_one_ulp_on_rank_one(water, df_serial,
                                                   monkeypatch, size):
    """Rank 1's Davidson buffer moved one ulp before the DF action of one
    iteration sees it: every rank still returns the serial solve bitwise, and
    the audit counts that one repair, of one ulp, on rank 1 alone.

    The ulp is the detector here. The DF ranks share no partials, so without
    the entry lockstep rank 0 never sees rank 1's drift and the final
    lockstep hands rank 1 rank 0's roots anyway; what gives it away is that
    rank 1 then iterated on its own, and the final lockstep repaired 1.99
    where the entry one repairs 1.1e-16."""
    fired = []
    real = davidson.df_block_action

    def planting(*args, **kwargs):
        act, diag = real(*args, **kwargs)
        comm = kwargs.get('comm')
        calls = [0]

        def action(z):
            calls[0] += 1
            if (comm is not None and comm.Get_rank() == 1
                    and calls[0] == PLANT_AT):
                fired.append(one_ulp_up(z))        # the driver's own buffer
            return act(z)
        return action, diag

    def one_rank(comm):
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = solve_df(water, probe=False)
            return out, lockstep_stats()

    monkeypatch.setattr(davidson, 'df_block_action', planting)
    moved = run_simulated(one_rank, size)
    assert fired and fired[0] > 0.0, 'the perturbation was never planted'
    om_s, X_s, Y_s, _ = df_serial
    for r, ((om, X, Y, _), stats) in enumerate(moved):
        assert bitwise(om, om_s) and bitwise(X, X_s) and bitwise(Y, Y_s)
        assert stats['mismatched_calls'] == (1 if r == 1 else 0)
        assert stats['max_abs_diff'] == (fired[0] if r == 1 else 0.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
