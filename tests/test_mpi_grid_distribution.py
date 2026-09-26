"""Distributed tau loops must reproduce the serial answer exactly, and every
collective cut into windows at an artificial count limit the bits of its one
call (tests/test_proj_rows.py gates the same on simulated ranks under pytest).
The reduce-scatter hands each rank its rows of the all-reduce within the
spread of the summed partials over the orders a reduction adds them in, whole
and in windows: bitwise over simulated ranks, which add both in rank order,
at that bar over MPI ranks, whose Reduce_scatter and Allreduce may add in
different orders (tests/test_sliced_factors.py gates it under pytest).

    python tests/test_mpi_grid_distribution.py           # partition algebra
    mpirun -n 3 python tests/test_mpi_grid_distribution.py   # the real check
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import numpy as np
warnings.simplefilter('ignore')
from pyscf import gto, scf
import src.Base.utils.mpi_grid as mpi_grid
from src.Base.utils.mpi_grid import (contiguous_block, grid_comm, partition,
                                     reduce_scatter_rows, reduce_sum,
                                     run_simulated)
from src.SingleReference.GW.space_time import (solve_qp_energy_space_time,
                                               separable_factors)
from src.Base.constants import HARTREE_TO_EV
from tests.test_proj_rows import LIMIT, collectives
from tests.test_sliced_factors import (rank_partials, reassociation_bar,
                                       scatter_shapes)

comm, rank, size = grid_comm()


def check(ok, label, detail=''):
    if rank == 0:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'  ({detail})' if detail else ''))
    return bool(ok)


all_ok = True
if rank == 0:
    print(f'\n-- partition covers every index exactly once')
for n in (14, 18, 40):
    for s in (1, 2, 3, 5, 8, 64):
        idx = np.sort(np.concatenate([partition(n, r, s) for r in range(s)]))
        all_ok &= check(np.array_equal(idx, np.arange(n)), f'n={n}, {s} ranks')

if rank == 0:
    print(f'\n-- every collective cut at MPI_COUNT_MAX = {LIMIT} gives the '
          'bits of its one call')


def windowed(run):
    """(one call each, in windows): what `run` returns at the two limits."""
    saved = mpi_grid.MPI_COUNT_MAX
    whole = run()
    mpi_grid.MPI_COUNT_MAX = LIMIT
    try:
        cut = run()
    finally:
        mpi_grid.MPI_COUNT_MAX = saved
    return whole, cut


if size > 1:
    whole, cut = windowed(lambda: collectives(comm))
    moved = [k for k in whole if whole[k].tobytes() != cut[k].tobytes()]
    moved = [m for own in comm.allgather(moved) for m in own]
    all_ok &= check(not moved, f'over {size} MPI ranks', f'moved: {moved}')
else:
    for s in (2, 3, 8):
        whole, cut = windowed(lambda: run_simulated(collectives, s))
        moved = [k for a, b in zip(whole, cut) for k in a
                 if a[k].tobytes() != b[k].tobytes()]
        all_ok &= check(not moved, f'over {s} simulated ranks', f'moved: {moved}')

if rank == 0:
    print(f'\n-- reduce_scatter_rows: each rank\'s rows of the all-reduce, '
          f'whole and cut at MPI_COUNT_MAX = {LIMIT}')


def scattered_rows(c):
    """Per shape: this rank's reduce-scatter rows, its rows of the all-reduce."""
    s, r = c.Get_size(), c.Get_rank()
    out = []
    for shape in scatter_shapes(s):
        mine = rank_partials(s, shape)[r]
        r0, r1 = contiguous_block(shape[0], r, s)
        out.append((reduce_scatter_rows(mine, c),
                    reduce_sum(mine.copy(), c)[r0:r1]))
    return out


def scatter_faults(s, whole, cut, exact):
    """Per rank and shape: the rows against the all-reduce's, and the windowed
    rows against the one call's, past the summation-order bar (at all when
    `exact`)."""
    faults = []
    for r, (w_rank, c_rank) in enumerate(zip(whole, cut)):
        for shape, (rows, want), (cut_rows, _) in zip(scatter_shapes(s),
                                                      w_rank, c_rank):
            bar = reassociation_bar(rank_partials(s, shape))
            for tag, got, ref in (('rows', rows, want),
                                  ('windowed', cut_rows, rows)):
                miss = (np.abs(got - ref).max(initial=0.0)
                        if got.shape == ref.shape else np.inf)
                if miss > bar or (exact and miss):
                    faults.append(f'rank {r} {shape} {tag}: {miss:.2e} '
                                  f'against {bar:.2e}')
    return faults


if size > 1:
    whole, cut = windowed(lambda: scattered_rows(comm))
    both = comm.allgather((whole, cut))
    faults = scatter_faults(size, [w for w, _ in both], [c for _, c in both],
                            exact=False)
    all_ok &= check(not faults, f'over {size} MPI ranks, within the bar',
                    '; '.join(faults))
else:
    for s in (2, 3, 8):
        whole, cut = windowed(lambda: run_simulated(scattered_rows, s))
        faults = scatter_faults(s, whole, cut, exact=True)
        all_ok &= check(not faults, f'over {s} simulated ranks, bitwise',
                        '; '.join(faults))

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri'); mf.kernel()
nocc = mol.nelectron // 2
F = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
serial = solve_qp_energy_space_time(mf, mol, nocc, nocc-1, factors=F) * HARTREE_TO_EV
if rank == 0:
    print(f'\n-- QP energy, {size} rank(s)')
if size > 1:
    dist = solve_qp_energy_space_time(mf, mol, nocc, nocc-1, factors=F,
                                      distribute=True) * HARTREE_TO_EV
    d = abs(dist - serial)
    all_ok &= check(d < 1e-9, f'distributed over {size} ranks == serial',
                    f'{serial:.9f} vs {dist:.9f}, |d| = {d:.2e} eV')
else:
    check(True, 'serial reference', f'{serial:.9f} eV -- rerun under mpirun to compare')

if rank == 0:
    print('\n' + ('All checks passed.' if all_ok else 'FAILURES above.'))
sys.exit(0 if all_ok else 1)
