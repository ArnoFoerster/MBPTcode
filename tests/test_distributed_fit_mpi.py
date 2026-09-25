"""The row-distributed ISDF fit under real ranks.

    python tests/test_distributed_fit_mpi.py                 # one rank
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_distributed_fit_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator, so `run_simulated(main, n)` runs the same checks over
thread-ranks where MPI cannot start; tests/test_distributed_fit.py gates the
same properties in pytest over simulated ranks.

THE RUN IS ONE DISTRIBUTED REGION, as in tests/test_mpi_routes.py: the row
fit, `separable_factors(fit='rows')`, runs inside `with distributed(comm):`
and writes rank 0's orbitals and points into every rank's inputs; the serial
references are computed on every rank afterwards inside `distributed(None)`,
on those inputs. Each check says whether the fit or the consumer took the
region's communicator ([context]) or was handed it ([explicit]), and every
rank's verdict is gathered: a check that fails on rank 2 fails the run.

Checks, water/cc-pVDZ Hartree-Fock, tiles of `TILE` points (7 tiles):
  * rows only [context]: the factors a rank holds are its `contiguous_block`
    rows, and every grid-indexed array the fit held (`fit_held`) is its own
    tiles' or rows' size, below the whole
  * bitwise across the ranks' views [context]: every rank's rows of X_mo, D
    and X_ao are the same rows of a one-rank run of the row fit on the same
    points, at `TILE` and at `FIT_CHOLESKY_BLOCK`
  * the anchored gate [context]: D over all ranks' rows against the serial
    replicated fit, within `FIT_REASSOCIATION_K` times the replicated fit's
    own response to one reordering of its three-centre sum (blocks cut per
    shell, summed in reverse)
  * the quasiparticle window [explicit] and the whole ISDF BSE [context] on
    the row `SlicedFactors`: bitwise the same distributed solves on the
    one-rank fit's whole arrays, rank 0's on every rank, and within the
    anchored bar of the same solves on the replicated fit's `SlicedFactors`
  * the shell block [context]: the rank that evaluated water's one block
    held its kept pairs' integrals and one call over them, the others none
  * the metric root on rank 0 [context]: D = M^T V^1/2 with the root formed
    on rank 0 alone (`RowFit.metric_root_rows`) is every rank's rows of the
    one-rank run bitwise, the root held by rank 0 only and a slab of it
    elsewhere, and the whole-metric D to 1e-12
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np
from pyscf import df, gto, scf

from src.Base import separable_ri
from src.Base.constants import (FIT_CHOLESKY_BLOCK, FIT_REASSOCIATION_K,
                                HARTREE_TO_EV)
from src.Base.separable_ri import (DEFAULT_PAIR_TOL, aux_metric_sqrt,
                                   fit_M_streaming)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import distributed, grid_comm, partition
from src.SingleReference.GW.space_time import (_row_factors,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         solve_bse_isdf)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
#: Water's 444 points in 7 tiles: 2 and 3 ranks each own several, and hand
#: panels and solved blocks to each other.
TILE = 64
NROOTS = 3


class Gate:
    """This rank's verdicts, gathered from every rank at the end."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()
        self.size = 1 if comm is None else comm.Get_size()
        self.verdicts = []

    def say(self, text):
        """One line from rank 0."""
        if self.rank == 0:
            print(text, flush=True)

    def section(self, title):
        self.say(f'\n-- {title}, {self.size} rank(s)')

    def info(self, text):
        """A number printed beside the checks; it gates nothing."""
        self.say(f'  [info] {text}')

    def check(self, ok, label, detail=''):
        """Record one verdict of this rank; rank 0 prints its own."""
        ok = bool(ok)
        self.verdicts.append((ok, label, detail))
        self.say(f"  [{'ok' if ok else 'FAIL'}] {label}"
                 + (f'  ({detail})' if detail else ''))
        return ok

    def everyone(self, obj):
        """Every rank's `obj`, rank-ordered."""
        return self.comm.allgather(obj) if self.comm is not None else [obj]

    def finish(self):
        """0 when every check passed on every rank, 1 otherwise."""
        failures = [(r, label, detail)
                    for r, own in enumerate(self.everyone(self.verdicts))
                    for ok, label, detail in own if not ok]
        if self.rank == 0:
            for r, label, detail in failures:
                if r != 0:
                    print(f'  [FAIL on rank {r}] {label}'
                          + (f'  ({detail})' if detail else ''))
            print('\n' + ('All checks passed on every rank.' if not failures
                          else 'FAILURES above.'), flush=True)
        return 0 if not failures else 1


def serial(fn, *args, **kwargs):
    """fn(*args, **kwargs), no kernel inside picking up the region's comm."""
    with distributed(None):
        return fn(*args, **kwargs)


def rows_of(factors):
    """((r0, r1), X_mo, D, X_ao) of sliced or whole factors."""
    if isinstance(factors, SlicedFactors):
        return factors.rows, factors.X_mo, factors.D, factors.X_ao
    return (0, len(factors[3])), factors[0], factors[1], factors[2]


def bitwise(a, b):
    """Every array of `a` holds the bits of the same array of `b`."""
    return len(a) == len(b) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        for x, y in zip(a, b))


def relative(a, b):
    """||a - b|| / ||b||."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def within_bar(dist, anchor):
    """The anchored standard: K anchors, and bitwise where the anchor is."""
    return dist <= FIT_REASSOCIATION_K * anchor


def one_rank_rows(mf, mol, coords, block):
    """The row fit on one rank, on the region's own points: the whole tuple
    every rank's rows are held to."""
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    return serial(_row_factors, mf, mol, auxmol, coords, 4.0,
                  DEFAULT_PAIR_TOL, None, None, block, time.time())


def reassociated_replicated_fit(gate, mf, mol):
    """The serial replicated fit with its three-centre blocks cut per shell
    and summed in reverse: the anchor of every gate below.

    The blocks are the module's `ao_blocks`, patched: thread-ranks share the
    module, so every rank patches, meets the others, fits, and meets them
    again before it restores -- no rank's fit reads another's patch state.
    """
    real = separable_ri.ao_blocks
    per_shell_reversed = [(s, s + 1) for s in reversed(range(mol.nbas))]
    gate.everyone(None)                       # every rank's own fits are done
    separable_ri.ao_blocks = lambda *args, **kwargs: per_shell_reversed
    gate.everyone(None)                       # ...and every rank is patched
    try:
        return serial(separable_factors, mf, mol, auxbasis=AUXBASIS)
    finally:
        gate.everyone(None)                   # every patched fit has run
        separable_ri.ao_blocks = real


def solves(mf, mol, nocc, factors, comm=None):
    """(quasiparticle window, BSE roots) on these factors in this region."""
    qp = solve_qp_energy_space_time(mf, mol, nocc, np.array([nocc - 1, nocc]),
                                    factors=factors, comm=comm)
    om = solve_bse_isdf(mf, mol, nocc, nroots=NROOTS, factors=factors,
                        progress=False)[0]
    return qp, om


def rows_only(gate, part, mol, nao, naux):
    """What a rank's row factors and its fit held: its rows and tiles."""
    gate.section('rows only')
    if not isinstance(part, SlicedFactors):
        gate.info('one rank: the row fit returns the whole tuple')
        return
    npts = part.npts
    r0, r1 = part.rows
    held = part.held_bytes()
    want = {'X_mo': (r1 - r0) * part.nmo * 8, 'D': (r1 - r0) * naux * 8,
            'X_ao': (r1 - r0) * nao * 8, 'coords': npts * 3 * 8}
    gate.check(held == want, "row factors [context] hold the rank's rows "
               'alone', f'rows {part.rows} of {npts}')
    fit = part.fit_held
    tiles = [(t * TILE, min((t + 1) * TILE, npts))
             for t in partition(-(-npts // TILE), gate.rank, gate.size)]
    own = sum(b - a for a, b in tiles)
    gram = sum((b - a) * b for a, b in tiles) * 8
    sized = (fit['FD_rows'] == own * naux * 8
             and fit['X_rows'] == own * nao * 8
             and fit['aux_rows'] == own * naux * 8
             and fit['D_tiles'] == own * naux * 8
             and fit['D_rows'] == (r1 - r0) * naux * 8
             and gram <= fit['S_rows'] < npts * npts * 8
             and fit['panel'] < npts * TILE * 8)
    gate.check(sized, "the fit [context] held the rank's tiles and rows of "
               'every grid-indexed array, never one whole',
               f"S {fit['S_rows']} of {npts * npts * 8} B, F D^T "
               f"{fit['FD_rows']} of {npts * naux * 8} B")
    # water is one shell block, every pair kept, evaluated by rank 0 in one
    # call over every nu shell
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())
    want = 2 * nao * n2 * naux * 8 if gate.rank == 0 else 0
    gate.check(fit['shell_block'] == want, 'the shell block [context]: its '
               'kept pairs\' integrals and one call no larger, on the rank '
               'that evaluated it alone', f"{fit['shell_block']} B")


def metric_root(gate, mol, part, naux):
    """D on the root formed by rank 0 alone, streamed in slabs."""
    gate.section('the metric root on rank 0')
    coords = part.coords if isinstance(part, SlicedFactors) else part[3]
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    fit = fit_M_streaming(mol, auxmol, coords, fit='rows', block=TILE)
    D = fit.metric_root_rows(auxmol)                       # context
    whole = serial(lambda: fit_M_streaming(mol, auxmol, coords, fit='rows',
                                           block=TILE).metric_root_rows(
                                               auxmol))
    reference = serial(lambda: fit_M_streaming(
        mol, auxmol, coords, fit='rows', block=TILE).metric_rows(
            aux_metric_sqrt(auxmol)))
    r0, r1 = fit.rows
    gate.check(D.tobytes() == whole[r0:r1].tobytes(), 'D on the rank-0 root '
               '[context]: this rank\'s rows == the one-rank run\'s, bitwise',
               f'rows ({r0}, {r1})')
    held = fit.held
    root = 2 * naux * naux * 8 if gate.rank == 0 else 0
    gate.check(held['metric_root'] == root
               and held['metric_slab'] == min(TILE, naux) * naux * 8,
               'the root on rank 0 alone, a slab of it elsewhere',
               f"root {held['metric_root']} B, slab {held['metric_slab']} B")
    dist = relative(whole, reference)
    gate.check(dist < 1e-12, 'D on the root == D on the whole metric to '
               'rounding', f'{dist:.2e}')


def views_are_bitwise(gate, mf, mol, part, block):
    """Every rank's rows are the one-rank run's rows, bit for bit."""
    coords = part.coords if isinstance(part, SlicedFactors) else part[3]
    whole = one_rank_rows(mf, mol, coords, block)
    (r0, r1), X_mo, D, X_ao = rows_of(part)
    ok = bitwise((X_mo, D, X_ao),
                 (whole[0][r0:r1], whole[1][r0:r1], whole[2][r0:r1]))
    gate.check(ok, f'row fit [context] at tiles of {block}: this rank\'s rows '
               'of X_mo, D and X_ao == the one-rank run\'s, bitwise',
               f'rows ({r0}, {r1})')
    return whole


def anchored(gate, mf, mol, nocc, part, whole):
    """The row fit and its consumers against the replicated fit's, anchored
    on the replicated fit's own reassociation response."""
    gate.section('the anchored gate against the replicated fit')
    old = serial(separable_factors, mf, mol, auxbasis=AUXBASIS)
    moved = reassociated_replicated_fit(gate, mf, mol)
    (r0, r1), _, D, _ = rows_of(part)
    sums = np.sum(gate.everyone(np.array([
        np.sum((D - old[1][r0:r1]) ** 2), np.sum(old[1][r0:r1] ** 2),
        np.sum((moved[1][r0:r1] - old[1][r0:r1]) ** 2)])), axis=0)
    dist, anchor = np.sqrt(sums[0] / sums[1]), np.sqrt(sums[2] / sums[1])
    gate.check(within_bar(dist, anchor), 'D of the row fit [context], all '
               'ranks\' rows, within the anchored bar of the replicated fit',
               f'{dist:.2e} against a reassociation of {anchor:.2e} '
               f'(bar {FIT_REASSOCIATION_K} x)')
    bar = [relative(a, b) for a, b in zip(serial(solves, mf, mol, nocc, moved),
                                          serial(solves, mf, mol, nocc, old))]
    W_old = serial(isdf_bse_factors, mf, mol, nocc, factors=old)[2]
    W_moved = serial(isdf_bse_factors, mf, mol, nocc, factors=moved)[2]
    W_rows = isdf_bse_factors(mf, mol, nocc, factors=part)[2]
    gate.check(within_bar(relative(W_rows, W_old), relative(W_moved, W_old)),
               'W(0) [context] on the row factors within the anchored bar',
               f'{relative(W_rows, W_old):.2e} against '
               f'{relative(W_moved, W_old):.2e}')

    gate.section('the window and the BSE on the row factors')
    on_rows = solves(mf, mol, nocc, part, comm=gate.comm)
    on_whole = solves(mf, mol, nocc, whole, comm=gate.comm)
    distinct = len(set(gate.everyone(b''.join(np.ascontiguousarray(a).tobytes()
                                              for a in on_rows))))
    gate.check(bitwise(on_rows, on_whole) and distinct == 1,
               f'window [explicit] and BSE [context] on the row factors over '
               f'{gate.size} rank(s) == on the one-rank fit\'s whole arrays, '
               'bitwise, on every rank', f'{distinct} distinct')
    sliced_old = (separable_factors(mf, mol, auxbasis=AUXBASIS, sliced=True)
                  if gate.size > 1 else old)
    on_old = solves(mf, mol, nocc, sliced_old, comm=gate.comm)
    for name, got, ref, anchor in zip(('window', 'BSE roots'), on_rows, on_old,
                                      bar):
        dist = relative(got, ref)
        gate.check(within_bar(dist, anchor), f'{name} on the row factors '
                   'within the anchored bar of the replicated fit\'s sliced '
                   'factors', f'{dist:.2e} against {anchor:.2e}, max |dE| '
                   f'{np.abs(got - ref).max() * HARTREE_TO_EV:.2e} eV')


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    with distributed(comm):
        mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
        serial(mf.kernel)
        nocc = mol.nelectron // 2
        nao = mol.nao_nr()
        naux = df.addons.make_auxmol(mol, auxbasis=AUXBASIS).nao_nr()
        part = separable_factors(mf, mol, auxbasis=AUXBASIS, fit='rows',
                                 fit_block=TILE)                 # context
        rows_only(gate, part, mol, nao, naux)
        gate.section('bitwise across the ranks\' views')
        whole = views_are_bitwise(gate, mf, mol, part, TILE)
        production = separable_factors(mf, mol, auxbasis=AUXBASIS, fit='rows')
        views_are_bitwise(gate, mf, mol, production, FIT_CHOLESKY_BLOCK)
        anchored(gate, mf, mol, nocc, part, whole)
        metric_root(gate, mol, part, naux)
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
