"""The distributed ISDF-K SCF under real ranks.

    python tests/test_distributed_isdf_scf_mpi.py                 # one rank
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_distributed_isdf_scf_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator, so `run_simulated(main, n)` runs the same checks over
thread-ranks; tests/test_distributed_isdf_scf.py gates the same properties
in pytest over simulated ranks.

THE RUN IS ONE DISTRIBUTED REGION: the handle (`distributed_isdf_jk`) and the
SCF (`distributed_mean_field`) take the region's communicator ([context]),
or are handed it ([explicit]); the serial references run on every rank
inside `distributed(None)`. Every rank's verdict is gathered: a check that
fails on rank 2 fails the run.

Checks, water/cc-pVDZ at 148 points per atom, tiles of `TILE` points (7):
  * the SCF [context], PBE0 and LRC-wPBEh: the energy within CONV_TOL of the
    serial ISDFJK SCF, and one energy and one set of orbitals on every rank
  * J, K and K_lr of one density [explicit] against the one-rank handle
    within COMPOSED_GRAD_K of the measured reassociation response, the same
    bits on every rank. K is a sum over row tiles, so its response is the
    largest move of the one-rank K when ITS OWN tile addends are
    re-associated as ranks re-associate them (`reassociation_response`:
    added last to first, and `REGROUPINGS` random orders cut into as many
    runs as ranks), or when the ranks' partials are added in reverse; J's is
    the ranks' partials reversed, floored at one ulp of its largest element
  * the interaction's tiles [explicit]: every rank's tiles of both
    operators' Z the one-rank tiles bitwise, and together the whole grid
  * the handle built inside the SCF [context] (`build=False`):
    every rank's tiles the one-rank tiles bitwise (at 8 ranks rank 7 owns
    none of the 7), the same operators built on every rank and their tiles
    each on one rank and together the grid's, and the BLAS threads its
    record says the fit, the interaction and K ran on the count this rank's
    process had outside the SCF -- on a node of BLAS_WRAP_MIN_THREADS or
    more, where the SCF holds BLAS at one thread, that is the handle's
    stages taking the pool back (`blas_full_pool`)
  * rows only [context]: what every rank holds after the SCF
    (`memory_faults`) is its own tiles' rows, nothing grid-indexed whole
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np
from pyscf import dft, gto, lib

from src.Base import distributed_isdf_jk as dist_isdf
from src.Base.constants import COMPOSED_GRAD_K
from src.Base.distributed_df import (distributed_mean_field,
                                     release_distributed)
from src.Base.distributed_isdf_jk import (ISDF_BLAS_KEYS, distributed_isdf_jk,
                                          distributed_isdf_storage)
from src.Base.isdf_jk import isdf_jk
from src.Base.utils.mpi_grid import (distributed, grid_comm,
                                     lockstep_mean_field)
from src.Base.utils.threads import blas_threads
from tests.test_distributed_fit_mpi import Gate, serial

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
#: Water's 444 points in 7 tiles, the last of 60: 2 and 3 ranks own several
#: each, 8 leave rank 7 none.
TILE = 64
#: The SCF's own thresholds: the energy gate is CONV_TOL itself.
CONV_TOL = 1e-10
CONV_TOL_GRAD = 1e-6
#: LRC-wPBEh's range-separation parameter in pyscf.
OMEGA = 0.2
ULP = np.finfo(float).eps
#: Random regroupings of the one-rank K's tile addends per rank count: on
#: water the largest move stops growing by 32 (K 3.6e-15, K_lr 1.1e-16),
#: where the seven fixed orders reach 1.8e-15 and 5.6e-17.
REGROUPINGS = 32


def fresh(xc):
    """An unrun ISDF mean field on water/cc-pVDZ, the default grid."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = isdf_jk(dft.RKS(mol, xc=xc), auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad = CONV_TOL, CONV_TOL_GRAD
    return mf


def digest(a):
    """The bytes of an array, hashed."""
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def tagged_density(mf):
    """`mf`'s density tagged with its orbitals, a copy."""
    return lib.tag_array(np.array(mf.make_rdm1()), mo_coeff=mf.mo_coeff.copy(),
                         mo_occ=mf.mo_occ.copy())


def memory_faults(storage, nocc, slab=None):
    """What a rank holds that the design says it must not, from its
    `distributed_isdf_storage`: an empty list is a pass."""
    faults = []
    nk, nao, naux, tile = (storage[k] for k in ('M', 'nao', 'naux', 'tile'))
    now, peak, whole = (storage['held_now'], storage['held_peak'],
                        storage['whole'])
    rows = storage['rows_here']
    if now['X_tiles'] != rows * nao * 8:
        faults.append(f"X tiles {now['X_tiles']} B, not {rows} rows")
    if now['MT_tiles'] != rows * naux * 8:
        faults.append(f"M^T tiles {now['MT_tiles']} B, not {rows} rows")
    for key, nbytes in now.items():
        if key.startswith('interaction_') and nbytes not in (
                rows * nk * 8, rows * naux * 8):
            faults.append(f'{key} {nbytes} B is not {rows} rows of Z or G')
    if max(peak.get('Z_rows', 0), peak.get('G_tiles', 0)) >= whole['Z'] > 0:
        faults.append('Z whole')
    if peak.get('A_block', 0) > tile * tile * 8:
        faults.append(f"a Hadamard block of {peak['A_block']} B")
    if peak.get('T_rows', 0) > rows * nao * 8:
        faults.append(f"T rows {peak['T_rows']} B")
    if peak.get('stream_tile', 0) > tile * max(nocc + nao + naux, naux) * 8:
        faults.append(f"a streamed tile of {peak['stream_tile']} B")
    if slab is not None and peak.get('metric_slab', 0) > slab * naux * 8:
        faults.append(f"a metric slab of {peak['metric_slab']} B")
    # the fit's grid-indexed arrays against their whole sizes
    fit_whole = {'S_rows': nk * nk, 'FD_rows': nk * naux, 'X_rows': nk * nao,
                 'B_rows': nk * nao, 'aux_rows': nk * naux, 'X_ext': nk * nao}
    for name, n in fit_whole.items():
        if storage['fit_held'].get(name, 0) >= n * 8:
            faults.append(f'the fit held {name} whole')
    return faults


def tile_addends(handle, dm, omega=None):
    """The one-rank handle's K addends X_i^T T_i of `dm`, one per row tile in
    its order: `exchange_partial` with `mine` that one tile, which adds the
    addend to zeros, so the addends summed in order are its K bitwise."""
    stack = np.asarray(dm).reshape(1, *dm.shape)
    factors = dist_isdf._occupied_factors(dm)
    mine, out = handle.mine, []
    try:
        for t in mine:
            handle.mine = [t]
            out.append(handle.exchange_partial(stack, factors, omega)[0])
    finally:
        handle.mine = mine
    return out


def regrouped(addends, groups):
    """The addends summed run by run, and the runs' sums added in turn."""
    total = np.zeros_like(addends[0])
    for group in groups:
        run = np.zeros_like(addends[0])
        for i in group:
            run = run + addends[i]
        total = total + run
    return total


def reassociation_response(addends, whole, sizes, seed=0):
    """The largest element of |K' - K| over re-associations K' of the
    one-rank K `whole` from its tile addends: added last to first, and for
    each rank count n of `sizes`, `REGROUPINGS` random tile orders cut at
    random into n runs, each run summed and the runs added -- how a K over n
    ranks re-associates the same addends. Refuses addends that do not sum to
    `whole` bitwise in tile order."""
    n = len(addends)
    if not np.array_equal(regrouped(addends, [range(n)]), whole):
        raise ValueError('the tile addends do not sum to the one-rank K')
    rng = np.random.default_rng(seed)
    groupings = [[[i] for i in reversed(range(n))]]
    for size in sizes:
        for _ in range(REGROUPINGS):
            cuts = np.sort(rng.choice(np.arange(1, n), min(size, n) - 1,
                                      replace=False))
            groupings.append(np.split(rng.permutation(n), cuts))
    return max(float(np.abs(regrouped(addends, g) - whole).max())
               for g in groupings)


def one_rank(xc, ref):
    """The one-rank handle (the same tiles, no collective): J, K, K_lr of
    `ref`'s density, its interaction tiles, and K's and K_lr's tile
    addends."""
    handle = dist_isdf.DistributedISDFJK(fresh(xc).with_df, comm=None,
                                         tile=TILE).build()
    vj, vk = handle.get_jk(tagged_density(ref))
    klr = handle.get_jk(tagged_density(ref), with_j=False, omega=OMEGA)[1]
    tiles = {(key, t): rows for key, kernel in handle.omega_kernels.items()
             for t, rows in kernel.items()}
    addends = tuple(tile_addends(handle, tagged_density(ref), omega)
                    for omega in (None, OMEGA))
    return (vj, vk, klr), tiles, addends


def ratio(diff, anchor):
    """diff / anchor, 0 where both are zero."""
    return diff / anchor if anchor else (float('inf') if diff else 0.0)


def scf_check(gate, xc, nocc):
    """The SCF over the region's ranks against the serial ISDFJK SCF."""
    gate.section(f'the SCF [context], {xc}')
    ref = fresh(xc)
    serial(ref.kernel)
    mf = fresh(xc)
    distributed_isdf_jk(mf, tile=TILE)                       # context
    distributed_mean_field(mf)                                # context
    storage = distributed_isdf_storage(mf)
    release_distributed(mf)
    dE = abs(mf.e_tot - ref.e_tot)
    gate.check(dE <= CONV_TOL, f'energy within CONV_TOL = {CONV_TOL:g} of '
               'the serial ISDF-K SCF', f'dE {dE:.2e} Ha')
    seen = gate.everyone((mf.e_tot, digest(mf.mo_coeff)))
    gate.check(len({e for e, _ in seen}) == 1, 'one energy on every rank')
    gate.check(len({d for _, d in seen}) == 1,
               'one set of orbitals on every rank')
    return storage


def jk_check(gate, xc):
    """J, K and K_lr [explicit] and the interaction tiles against one rank."""
    gate.section(f'J, K and K_lr of one density [explicit], {xc}')
    ref = fresh(xc)
    serial(ref.kernel)
    # one density on every rank, which the handle would lock at its entry
    lockstep_mean_field(ref, gate.comm)
    one, one_tiles, addends = serial(one_rank, xc, ref)
    handle = (distributed_isdf_jk(fresh(xc), gate.comm, tile=TILE)
              if gate.size > 1 else
              dist_isdf.DistributedISDFJK(fresh(xc).with_df, comm=None,
                                          tile=TILE).build())
    dm = tagged_density(ref)
    stack = np.asarray(dm).reshape(1, *dm.shape)
    factors = dist_isdf._occupied_factors(dm)
    partials = (handle._coulomb_engine(None, 1e-13)(stack)[0],
                handle.exchange_partial(stack, factors)[0],
                handle.exchange_partial(stack, factors, OMEGA)[0])
    vj, vk = handle.get_jk(tagged_density(ref))
    got = (vj, vk, handle.get_jk(tagged_density(ref), with_j=False,
                                 omega=OMEGA)[1])
    tiles = {(key, t): rows for key, kernel in handle.omega_kernels.items()
             for t, rows in kernel.items()}
    everyone = gate.everyone(partials)
    for n, name in enumerate(('J', 'K', 'K_lr')):
        parts = [p[n] for p in everyone]
        forward, backward = parts[0].copy(), parts[-1].copy()
        for p in parts[1:]:
            forward = forward + p
        for p in parts[-2::-1]:
            backward = backward + p
        anchor = float(np.abs(forward - backward).max())
        if n == 0:
            # J is no sum over tiles: the ranks' partials reversed, one ulp
            anchor = max(anchor, ULP * np.abs(one[n]).max())
        else:
            anchor = max(anchor, reassociation_response(
                addends[n - 1], one[n], [gate.size]))
        diff = float(np.abs(got[n] - one[n]).max())
        gate.check(diff <= COMPOSED_GRAD_K * anchor, f'{name} within '
                   f'COMPOSED_GRAD_K = {COMPOSED_GRAD_K} of the reassociation '
                   'response', f'{diff:.2e} = {ratio(diff, anchor):.2f} x '
                   f'{anchor:.2e}')
        gate.check(len(set(gate.everyone(digest(got[n])))) == 1,
                   f'{name}: the same bits on every rank')
    gate.check(all(np.array_equal(rows, one_tiles[key])
                   for key, rows in tiles.items()),
               "this rank's tiles of both operators' Z bitwise the one-rank "
               'tiles')
    covered = set().union(*[set(t) for t in gate.everyone(sorted(tiles))])
    gate.check(covered == set(one_tiles), "the ranks' tiles are the grid's")
    handle.release()
    built_in_scf_check(gate, xc, one_tiles)


def built_in_scf_check(gate, xc, one_tiles):
    """The handle built inside the SCF [context]: its tiles against the
    one-rank tiles -- a rank may own none, rank 7 of 8 on water's 7 tiles --
    the ranks' tiles of the operators every rank built each on one rank and
    together the grid, and its record of the BLAS threads against this
    process's count outside the SCF."""
    gate.section(f'the handle built inside the SCF [context], {xc}')
    mf = fresh(xc)
    if distributed_isdf_jk(mf, tile=TILE, build=False) is None:
        gate.info('one rank: the serial ISDFJK SCF, no handle to build')
        return
    outside = blas_threads() or 0
    distributed_mean_field(mf)
    kernels = mf._distributed[0].omega_kernels
    operators = sorted(kernels)
    tiles = {(key, t): rows for key, kernel in kernels.items()
             for t, rows in kernel.items()}
    record = {key: mf._distributed_timings.get(key) for key in ISDF_BLAS_KEYS}
    release_distributed(mf)
    gate.check(all(np.array_equal(rows, one_tiles.get(key))
                   for key, rows in tiles.items()),
               "this rank's tiles of the SCF's operators bitwise the "
               'one-rank tiles', f'{len(tiles)} here')
    everyone = gate.everyone((operators, sorted(tiles)))
    held = [key for _, keys in everyone for key in keys]
    grid = {key for key in one_tiles if key[0] in operators}
    gate.check(operators and all(ops == operators for ops, _ in everyone)
               and len(held) == len(set(held)) and set(held) == grid,
               "the SCF's operators on every rank, their tiles each on one "
               "rank and together the one-rank tiles",
               f'{operators}: {len(held)} tiles of {len(grid)}')
    gate.check(all(count == outside for count in record.values()),
               f'the fit, the interaction and K on the {outside} BLAS '
               'threads outside the SCF', str(record))


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    with distributed(comm):
        nocc = gto.M(atom=WATER, basis=BASIS).nelectron // 2
        for xc in ('pbe0', 'lrc-wpbeh'):
            storage = scf_check(gate, xc, nocc)
            if storage is not None:
                gate.section(f'rows only [context], {xc}')
                faults = memory_faults(storage, nocc)
                gate.check(not faults, "every array this rank holds is its "
                           "own tiles' rows", '; '.join(faults))
                rows = gate.everyone(storage['rows_here'])
                gate.check(sum(rows) == storage['M'], 'the rows tile the '
                           'grid', f'{rows} of {storage["M"]}')
                gate.info(f"held after the SCF: {storage['held_now']}")
            jk_check(gate, xc)
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
