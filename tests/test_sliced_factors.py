"""The separable factors held as grid-point slices: each rank its rows, a stage
that reads an array whole gathers it once, and every output is the replicated
factors' bit for bit.

`separable_factors(sliced=True)` inside a region returns `SlicedFactors`: the
rank's `contiguous_block` of the grid rows of X_mo, D and X_ao, cut from the
whole, lockstepped fit. Gated here over 2, 3 and 8 simulated ranks
(`run_simulated`, threads of this process with the real collectives), on
water/cc-pVDZ Hartree-Fock, against the unsliced factors built in the same
region:

  * serially the flag is inert: the tuple comes back, bitwise the unsliced one;
  * the slices ARE the rows of the replicated fit, bitwise, and tile the grid;
  * THE MEMORY ASSERTION: every array a rank's factors hold is its slice --
    X_mo, D and X_ao of exactly contiguous_block(M, rank, nranks) rows, the
    grid points whole -- read off the object, not off the formula, and over
    the ranks the slices add up to one copy of each array;
  * the BSE block action on slices (BSE, TDHF, RPA; Davidson and the (A-B)
    probe): roots, vectors and probe value bitwise the unsliced run's on every
    rank, every rank's equal to rank 0's, and ONE gather of D and one of X_o
    per action build whatever the iteration count (none for RPA, which reads
    rows alone);
  * the GW window on slices, in-core and frequency-blocked: quasiparticle
    energies bitwise, every rank rank 0's, and one gather each of D, X_o,
    X_v and X_ao per solve;
  * the whole ISDF BSE on slices (GW diagonal, W carried from the GW axis,
    probe, Davidson) and the static W rebuilt on its own: bitwise, with the
    gathers counted per stage;
  * what cannot be served from slices is refused on every rank before any
    collective: unpacking them as a tuple, a serial consumer, a reaction
    field;
  * `mpi_grid.allgather_rows`, the row-counted gather the slices travel
    through, returns what `allgather_blocks` does, bitwise.

Collectives are counted by wrapping `sliced_factors.allgather_rows`, per rank.
"""
import os
import sys
import threading
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base import sliced_factors
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (allgather_blocks, allgather_rows,
                                     contiguous_block, distributed,
                                     run_simulated)
from src.SingleReference.GW.space_time import (_sliced_solve_factors,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         isdf_block_action,
                                                         lowest_amb_eigenvalue,
                                                         solve_bse_isdf,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

SIZES = [2, 3, 8]
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
NROOTS = 3


@pytest.fixture(scope='module')
def water():
    """(mol, mf, nocc, static W) on water/cc-pVDZ, W from the serial factors."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis=AUXBASIS)
    W = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return mol, mf, nocc, W


@pytest.fixture
def collectives(monkeypatch):
    """{rank: number of `allgather_rows` calls the sliced factors made}."""
    calls, lock = defaultdict(int), threading.Lock()
    real = sliced_factors.allgather_rows

    def counted(a, comm):
        with lock:
            calls[comm.Get_rank()] += 1
        return real(a, comm)

    monkeypatch.setattr(sliced_factors, 'allgather_rows', counted)
    return calls


def both_fits(mol, mf):
    """(unsliced, sliced) factors from the same region, on this rank."""
    whole = separable_factors(mf, mol, auxbasis=AUXBASIS)
    return whole, separable_factors(mf, mol, auxbasis=AUXBASIS, sliced=True)


def bitwise(a, b):
    """Every array of `a` holds the bits of the same array of `b`."""
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_serial_call_returns_the_tuple(water):
    mol, mf, _, _ = water
    with distributed(None):
        plain = separable_factors(mf, mol, auxbasis=AUXBASIS)
        flagged = separable_factors(mf, mol, auxbasis=AUXBASIS, sliced=True)
    assert isinstance(flagged, tuple)
    assert bitwise(plain, flagged)


@pytest.mark.parametrize('size', SIZES)
def test_slices_are_the_rows_of_the_replicated_fit(water, size):
    mol, mf, _, _ = water

    def rank(comm):
        whole, part = both_fits(mol, mf)
        r0, r1 = part.rows
        return (part.rows, bitwise([a[r0:r1] for a in whole[:3]],
                                   [part.X_mo, part.D, part.X_ao]),
                np.array_equal(part.coords, whole[3]), part.npts)

    out = run_simulated(rank, size)
    npts = out[0][3]
    assert [o[0] for o in out] == [contiguous_block(npts, r, size)
                                   for r in range(size)]
    assert all(o[1] for o in out), 'a slice is not the rows of the fit'
    assert all(o[2] for o in out), 'the grid points are not whole'


@pytest.mark.parametrize('size', SIZES)
def test_each_rank_holds_its_slice(water, size):
    """THE MEMORY ASSERTION: what a rank's factors hold is its rows."""
    mol, mf, _, _ = water

    def rank(comm):
        whole, part = both_fits(mol, mf)
        return part.held_bytes(), {k: int(a.nbytes) for k, a in
                                   zip(('X_mo', 'D', 'X_ao', 'coords'), whole)}

    out = run_simulated(rank, size)
    whole = out[0][1]
    npts = whole['coords'] // (3 * 8)
    nmo, naux, nao = (whole[k] // (npts * 8) for k in ('X_mo', 'D', 'X_ao'))
    for r, (held, _) in enumerate(out):
        r0, r1 = contiguous_block(npts, r, size)
        rows = r1 - r0
        assert rows <= -(-npts // size)
        assert held == {'X_mo': rows * nmo * 8, 'D': rows * naux * 8,
                        'X_ao': rows * nao * 8, 'coords': npts * 3 * 8}, (
            f'rank {r} of {size} holds {held}, its slice is {rows} rows')
    for name in ('X_mo', 'D', 'X_ao'):
        assert sum(held[name] for held, _ in out) == whole[name]


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('mode', ['BSE', 'TDHF', 'RPA'])
def test_block_action_on_slices(water, collectives, size, mode):
    mol, mf, nocc, W = water
    W_aux = W if mode == 'BSE' else None

    def rank(comm):
        whole, part = both_fits(mol, mf)
        out = {}
        for tag, fac in (('whole', whole), ('sliced', part)):
            lr = LinearResponseSolver(np.asarray(mf.mo_energy),
                                      spin_mode='restricted')
            n0, stats = collectives[comm.Get_rank()], {}
            om, X, Y = solve_casida_davidson(lr, nocc, nroots=NROOTS,
                                             polarizability=mode, W_aux=W_aux,
                                             isdf_factors=fac, stats=stats)
            n1 = collectives[comm.Get_rank()]
            amb = lowest_amb_eigenvalue(lr, nocc, polarizability=mode,
                                        W_aux=W_aux, isdf_factors=fac)
            out[tag] = (om, X, Y, np.atleast_1d(amb[0]))
            out[tag + '_gathers'] = (n1 - n0, collectives[comm.Get_rank()] - n1,
                                     stats.get('davidson_vind_calls'))
        out['by_name'] = dict(part.gathers)
        return out

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        assert bitwise(out['sliced'], out['whole']), f'rank {r}: not bitwise'
        assert bitwise(out['sliced'], res[0]['sliced']), f"rank {r} != rank 0"
        per_build = 0 if mode == 'RPA' else 2          # D and X_o, once each
        davidson, probe, vind = out['sliced_gathers']
        assert vind > 1, 'the Davidson applied the action only once'
        assert (davidson, probe) == (per_build, per_build), (
            f'{davidson} + {probe} gathers for {vind} action calls')
        assert out['by_name'] == ({} if mode == 'RPA' else {'D': 2, 'X_o': 2})


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('path', ['in-core', 'freq_block'])
def test_gw_window_on_slices(water, collectives, size, path):
    mol, mf, nocc, _ = water
    window = np.array([nocc - 1, nocc])
    kwargs = {} if path == 'in-core' else {'freq_block': 3}

    def rank(comm):
        whole, part = both_fits(mol, mf)
        qp_whole = solve_qp_energy_space_time(mf, mol, nocc, window,
                                              factors=whole, **kwargs)
        n0 = collectives[comm.Get_rank()]
        qp = solve_qp_energy_space_time(mf, mol, nocc, window, factors=part,
                                        **kwargs)
        return qp, qp_whole, collectives[comm.Get_rank()] - n0, dict(part.gathers)

    res = run_simulated(rank, size)
    for r, (qp, qp_whole, n, by_name) in enumerate(res):
        assert np.array_equal(qp, qp_whole), f'rank {r}: {qp - qp_whole}'
        assert np.array_equal(qp, res[0][0]), f'rank {r} != rank 0'
        assert n == 4 and by_name == {'D': 1, 'X_o': 1, 'X_v': 1, 'X_ao': 1}


@pytest.mark.parametrize('size', SIZES)
def test_bse_on_slices(water, size):
    mol, mf, nocc, _ = water

    def rank(comm):
        whole, part = both_fits(mol, mf)
        out = {}
        for tag, fac in (('whole', whole), ('sliced', part)):
            om, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=NROOTS,
                                            factors=fac, progress=False)
            out[tag] = (om, X, Y, info['eps'], info['W_aux'],
                        np.atleast_1d(info['min_eig_amb']))
        out['by_name'] = dict(part.gathers)
        return out

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        assert bitwise(out['sliced'], out['whole']), f'rank {r}: not bitwise'
        assert bitwise(out['sliced'], res[0]['sliced']), f'rank {r} != rank 0'
        # the GW diagonal once each; the probe and the Davidson D and X_o each
        assert out['by_name'] == {'D': 3, 'X_o': 3, 'X_v': 1, 'X_ao': 1}


@pytest.mark.parametrize('size', [3])
def test_static_w_on_slices(water, size):
    mol, mf, nocc, _ = water

    def rank(comm):
        whole, part = both_fits(mol, mf)
        W_whole = isdf_bse_factors(mf, mol, nocc, factors=whole)[2]
        got = isdf_bse_factors(mf, mol, nocc, factors=part)
        return got[0] is part, got[1], got[2], W_whole, dict(part.gathers)

    for r, (same, D, W, W_whole, by_name) in enumerate(run_simulated(rank, size)):
        assert same and D is None
        assert np.array_equal(W, W_whole), f'rank {r}'
        assert by_name == {'D': 1, 'X_o': 1, 'X_v': 1}


def test_what_slices_cannot_serve_is_refused(water):
    mol, mf, nocc, W = water

    def rank(comm):
        part = both_fits(mol, mf)[1]
        refused = []
        for attempt in (lambda: tuple(part), lambda: part[1]):
            with pytest.raises(TypeError):
                attempt()
            refused.append(True)
        with distributed(None):
            with pytest.raises(ValueError, match='serial reference'):
                solve_qp_energy_space_time(mf, mol, nocc, nocc - 1,
                                           factors=part)
            lr = LinearResponseSolver(np.asarray(mf.mo_energy),
                                      spin_mode='restricted')
            with pytest.raises(ValueError, match='serial reference'):
                isdf_block_action(lr, nocc, True, W, part)
        refused.append(True)
        with pytest.raises(ValueError, match='reaction field'):
            _sliced_solve_factors(part, comm, np.eye(part.naux))
        refused.append(True)
        return refused

    assert all(all(r) for r in run_simulated(rank, 2))


@pytest.mark.parametrize('size', SIZES)
def test_allgather_rows_is_allgather_blocks(size):
    rng = np.random.default_rng(size)
    shapes = [(37, 5), (37,), (37, 2, 3), (37, 0), (size - 1, 4)]
    wholes = [rng.normal(size=s) for s in shapes]

    def rank(comm):
        out = []
        for whole in wholes:
            r0, r1 = contiguous_block(whole.shape[0], comm.Get_rank(), size)
            a, b = np.zeros_like(whole), np.zeros_like(whole)
            a[r0:r1], b[r0:r1] = whole[r0:r1], whole[r0:r1]
            out.append((allgather_rows(a, comm), allgather_blocks(b, comm)))
        return out

    for per_rank in run_simulated(rank, size):
        for whole, (a, b) in zip(wholes, per_rank):
            assert np.array_equal(a, whole) and np.array_equal(a, b)


def test_sliced_factors_refuse_one_rank(water):
    mol, mf, _, _ = water
    with distributed(None):
        whole = separable_factors(mf, mol, auxbasis=AUXBASIS)
    with pytest.raises(ValueError, match='more than one rank'):
        SlicedFactors(*whole, comm=None)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
