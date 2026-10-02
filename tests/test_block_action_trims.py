"""The ISDF block action's element-wise passes on threads, and its owner-order
write without a memset, move no bit.

`isdf_block_action` runs three numpy passes that no BLAS call covers: the
Hadamard product of each row tile of Zt with P, the write of X_o^T (Zt * P)
into owner order, and the batch's d z and reduced-slab updates. Each now runs
on `row_map`'s threads, cut over rows, and a row block's first tile is
copied into the buffer instead of added to a zeroed one, 0 + x = x but for
the sign of an exact zero, which reaches no sum; later tiles add in as
before. Every element is the one operation it was, so the action is gated
BITWISE:

  * against the baseline tree's action (BASELINE_COMMIT, unpacked and run
    in its own process on this process's inputs), Az and Bz of two batches,
    three vectors and then one, the second reusing the buffers the first left
    behind -- serially and at 2, 3, 5 and 8 simulated ranks, on water,
    ethylene and benzene cc-pVDZ Hartree-Fock, with the default tile (one
    tile per rank, the single write) and with TILE_ROWS-row tiles (several
    per rank, the write then the adds), for the screened singlet; the
    triplet, TDHF and RPA kernels at 1, 3 and 8 ranks on water;
  * at CUT_RANKS ranks on water's fit cut to CUT_POINTS points, where four
    ranks own no grid row, write no tile and send a zero partial;
  * every rank returns rank 0's output bitwise;
  * at 1, 2 and 3 threads alike (`openmp_threads` patched), serially and at
    three ranks;
  * and the gate is shown to fail: a first tile added into the buffer the
    last vector left, instead of written, moves the bits, with one tile per
    rank and with several.

The per-piece timers (`davidson_action_<piece>`) that ride the same loop are
gated in tests/test_davidson_timings.py; the roots at the default tile stay
the baseline's by tests/test_probe_after_davidson.py.
"""
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         isdf_block_action)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')
BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; '
           'H 1.240 2.147 0; H -1.240 2.147 0; H -2.479 0 0; '
           'H -1.240 -2.147 0; H 1.240 -2.147 0')
MOLECULES = {'water': WATER, 'ethylene': ETHYLENE, 'benzene': BENZENE}
SIZES = [1, 2, 3, 5, 8]
#: Grid rows per tile of the several-tile runs: odd, so the last tile of
#: most row blocks is short.
TILE_ROWS = 7
TILES = [None, TILE_ROWS]
#: (lBSE, screened, spin) of the kernels besides the screened singlet, and
#: the rank counts they run at on water.
OTHER_KERNELS = {'triplet': (True, True, 'triplet'),
                 'tdhf': (True, False, 'singlet'),
                 'rpa': (False, False, 'singlet')}
OTHER_SIZES = [1, 3, 8]
THREADS = [1, 2, 3]
#: Grid points water's fit is cut to, and the ranks that then leave four of
#: them without a row.
CUT_POINTS, CUT_RANKS = 12, 16
#: Trial vectors per batch, applied in this order on one action.
BATCHES = (3, 1)
REPO = Path(__file__).resolve().parents[1]
#: A commit before the threaded passes and the first-tile write.
BASELINE_COMMIT = '8b5d2820f5ecb5920440048cbe307382c9040aa4'
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: The same actions on the unpacked baseline, on the inputs this process
#: saved, in their own process.
BASELINE_ACTIONS = '''
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.LinearResponse.davidson import isdf_block_action
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

warnings.simplefilter('ignore')
data = np.load({inputs!r})
out = {{}}
for key, name, lBSE, screened, spin, size, tile in {cases!r}:
    nocc = int(data[name + '_nocc'])
    factors = (data[name + '_X_mo'], data[name + '_D'])
    W_aux = data[name + '_W_aux'] if screened else None
    batches = [data[f'{{name}}_z{{b}}'] for b in range({nbatches})]
    npts = factors[0].shape[0]
    kw = {{}} if tile is None else {{'tile_memory_gb': tile * npts * 8 / 1e9}}

    def one_rank(comm):
        lr = LinearResponseSolver(data[name + '_eps'].copy(),
                                  spin_mode='restricted')
        act, _ = isdf_block_action(lr, nocc, lBSE, W_aux,
                                   tuple(a.copy() for a in factors),
                                   spin=spin, comm=comm, **kw)
        return [act(z.copy()) for z in batches]

    ranks = [one_rank(None)] if size == 1 else run_simulated(one_rank, size)
    for b, (A, B) in enumerate(ranks[0]):
        out[f'{{key}}_b{{b}}_A'], out[f'{{key}}_b{{b}}_B'] = A, B
np.savez({out!r}, **out)
'''


def _system(geometry):
    """Orbital energies, fit and static W of one molecule, cc-pVDZ HF, with
    one random batch per entry of BATCHES."""
    mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    eps = np.asarray(mf.mo_energy, float)
    rng = np.random.default_rng(nocc)
    batches = [rng.normal(size=(n, nocc, len(eps) - nocc)) for n in BATCHES]
    return dict(eps=eps, nocc=nocc, X_mo=np.asarray(factors[0]),
                D=np.asarray(factors[1]), W_aux=W_aux, batches=batches)


@pytest.fixture(scope='module')
def systems():
    warnings.simplefilter('ignore')
    out = {name: _system(geometry) for name, geometry in MOLECULES.items()}
    cut = dict(out['water'])
    for key in ('X_mo', 'D'):
        cut[key] = np.ascontiguousarray(cut[key][:CUT_POINTS])
    out['water_cut'] = cut
    return out


def _cases():
    """(key, molecule, lBSE, screened, spin, size, tile) of every action the
    parent is asked for."""
    out = [(f'{name}_bse_{size}_{tile}', name, True, True, 'singlet', size,
            tile) for name in MOLECULES for size in SIZES for tile in TILES]
    out += [(f'water_{kernel}_{size}_{tile}', 'water', *spec, size, tile)
            for kernel, spec in OTHER_KERNELS.items()
            for size in OTHER_SIZES for tile in TILES]
    out.append((f'water_cut_bse_{CUT_RANKS}_None', 'water_cut', True, True,
                'singlet', CUT_RANKS, None))
    return out


def actions(s, lBSE=True, screened=True, spin='singlet', size=1, tile=None):
    """Every rank's [(Az, Bz)] over the batches, on one action per rank."""
    kw = ({} if tile is None else
          {'tile_memory_gb': tile * s['X_mo'].shape[0] * 8 / 1e9})

    def one_rank(comm):
        lr = LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')
        act, _ = isdf_block_action(lr, s['nocc'], lBSE,
                                   s['W_aux'] if screened else None,
                                   (s['X_mo'].copy(), s['D'].copy()),
                                   spin=spin, comm=comm, **kw)
        return [act(z.copy()) for z in s['batches']]

    return [one_rank(None)] if size == 1 else run_simulated(one_rank, size)


@pytest.fixture(scope='module')
def parent(systems, tmp_path_factory):
    """The parent tree's Az and Bz of every case, keyed `<case>_b<batch>_A|B`."""
    out = tmp_path_factory.mktemp('block_action_trims_parent')
    tar = out.parent / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
    inputs, npz = out / 'inputs.npz', out / 'actions.npz'
    saved = {}
    for name, s in systems.items():
        for key in ('eps', 'nocc', 'X_mo', 'D', 'W_aux'):
            saved[f'{name}_{key}'] = s[key]
        for b, z in enumerate(s['batches']):
            saved[f'{name}_z{b}'] = z
    np.savez(inputs, **saved)
    script = out / 'parent.py'
    script.write_text(BASELINE_ACTIONS.format(
        archive=str(out), inputs=str(inputs), cases=_cases(),
        nbatches=len(BATCHES), out=str(npz)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(out),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(npz))


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def assert_parents(out, parent, key):
    """Rank 0's batches bitwise the parent's, and every rank's rank 0's."""
    for b, (A, B) in enumerate(out[0]):
        assert bitwise(A, parent[f'{key}_b{b}_A']), (key, b, 'A')
        assert bitwise(B, parent[f'{key}_b{b}_B']), (key, b, 'B')
    for rank_out in out[1:]:
        for (A, B), (A0, B0) in zip(rank_out, out[0]):
            assert bitwise(A, A0) and bitwise(B, B0), key


@pytest.mark.parametrize('tile', TILES)
@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_screened_action_is_bitwise_the_parents(systems, parent, name,
                                                    size, tile):
    assert_parents(actions(systems[name], size=size, tile=tile), parent,
                   f'{name}_bse_{size}_{tile}')


@pytest.mark.parametrize('tile', TILES)
@pytest.mark.parametrize('size', OTHER_SIZES)
@pytest.mark.parametrize('kernel', list(OTHER_KERNELS))
def test_every_kernel_is_bitwise_the_parents(systems, parent, kernel, size,
                                             tile):
    lBSE, screened, spin = OTHER_KERNELS[kernel]
    out = actions(systems['water'], lBSE, screened, spin, size, tile)
    assert_parents(out, parent, f'water_{kernel}_{size}_{tile}')


@pytest.mark.parametrize('tile', TILES)
@pytest.mark.parametrize('size', [1, 3])
@pytest.mark.parametrize('threads', THREADS)
def test_every_thread_count_gives_the_parents_bits(systems, parent,
                                                   monkeypatch, threads,
                                                   size, tile):
    """The passes cut over 1, 2 and 3 threads: benzene's 21 occupied rows
    and its tiles split unevenly over three, and the bits do not move."""
    monkeypatch.setattr(davidson, 'openmp_threads', lambda: threads)
    assert_parents(actions(systems['benzene'], size=size, tile=tile), parent,
                   f'benzene_bse_{size}_{tile}')


def test_ranks_that_own_no_grid_row(systems, parent):
    """Water's fit cut to CUT_POINTS grid points over CUT_RANKS ranks: the
    ranks without a row write no tile and send a zero partial, as before."""
    assert_parents(actions(systems['water_cut'], size=CUT_RANKS), parent,
                   f'water_cut_bse_{CUT_RANKS}_None')


@pytest.mark.parametrize('tile', TILES)
def test_a_first_tile_added_to_a_stale_buffer_moves_the_bits(systems, parent,
                                                             monkeypatch,
                                                             tile):
    """Every tile added in, the first too, onto what the last vector left in
    the owner-order buffer: the gate sees the missing write."""
    real = davidson._owner_write

    def stale(T_owned, owners, Tb, first, threads):
        return real(T_owned, owners, Tb, False, threads)

    monkeypatch.setattr(davidson, '_owner_write', stale)
    out = actions(systems['water'], size=2, tile=tile)
    key = f'water_bse_2_{tile}'
    assert any(not bitwise(A, parent[f'{key}_b{b}_A'])
               for b, (A, _) in enumerate(out[0]))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
