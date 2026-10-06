"""Each rank holds one calculation's force and energy under three stand-ins
for arithmetic that a single machine may not show.

The pyscf race: `pyscf.lib.ddot`, behind `lib.dot` and `lib.einsum`, calls
NPdgemm (pyscf/lib/np_helper/npdot.c), which, where k/m > 3 and k/n > 3,
splits the K index over the OpenMP threads and adds the partial products into
C under `omp critical` in thread-arrival order, so at 16 threads no pyscf
result repeats bit for bit. `racing_dgemm` replaces `numpy_helper._dgemm`, the
ctypes caller of NPdgemm, by what NPdgemm computes at `OMP_THREADS` threads,
the partials added in a fresh random order per call.

The shape-sensitive BLAS: on OpenBLAS a GEMM row depends on the call's shape
(tests/test_frequency_rows_serial_shaped.py). Every `a @ b`,
`np.dot/matmul/tensordot/inner/vdot`, contracting `np.einsum` and `x.dot(y)`
under src/ is rewritten at import (`ShapeRewriter`) to scale its result by
1 + k 2^-52, k = 1 + crc32(op, operand shapes) % `SHAPE_SKEW_CLASSES`. The
rewrite applies only to modules imported after it, so that part runs in a
subprocess that drops src and imports it rewritten.

The free memory: pyscf blocks a density-fitted gradient's auxiliary index by
the process's free memory; each simulated rank reports its own
(`RankMemoryLib`).

Gated on water/cc-pVDZ Hartree-Fock at 148 points per atom, at 2, 3 and 8
simulated ranks, for the whole layout, the sliced one (`sliced=True`) and the
row fit (`fit='rows'`), each layout of a rank handed one reference and one
displaced mean field:
  (a) under the race and the free memory: the composed state-pair force at a
      displaced geometry, its energy and the dRPA force of each layout, the
      optimizer's mean-field force (`mean_field_force`), the dense dRPA
      force and energy, and the Hartree-Fock mirror energy of a PBE0
      reference (`reference_energy`) are rank 0's bits on every rank, while
      pyscf's own gradient of the locked mean field differs between ranks;
  (b) on the shape-sensitive BLAS and the free memory (subprocess): the same
      three numbers of each layout rank 0's on every rank, the sliced layout
      bitwise the whole one, and the row fit's forces and energy within
      FIT_REALIZATION_FORCE_TOL and FIT_REALIZATION_ENERGY_TOL of the
      whole-fit sliced layout's, two realizations of the one fit;
  (c) on all three stand-ins at once (subprocess): every number of (b) rank
      0's on every rank.

The layouts are not compared with each other under the race: each runs its
own pyscf K builds and mean-field force, which a racing pyscf does not repeat.
On the stand-in, which repeats its bits, the row fit sits 7.6e-9 Ha/Bohr from
the whole fit in the composed force (0.076 of FIT_REALIZATION_FORCE_TOL),
7.6e-11 in the dRPA force and 9.1e-13 Ha in the energy (0.009 of
FIT_REALIZATION_ENERGY_TOL).
"""
import ast
import builtins
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import threading
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.df.grad import rhf as pyscf_df_grad
from pyscf.lib import numpy_helper

from src.Base.constants import (FIT_REALIZATION_ENERGY_TOL,
                                FIT_REALIZATION_FORCE_TOL)
from src.Base.utils.mpi_grid import (current_comm, lockstep_mean_field,
                                     run_simulated)
from src.gradients.dense_surfaces import DenseRPASurface
from src.gradients.rpa_bse_surface import RPABSESurface
from src.properties.optimize import mean_field_force
from src.SingleReference.LinearResponse.rpa_energy import reference_energy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [2, 3, 8]
#: The rank counts (c) runs at: its gate is (a)'s and (b)'s together.
ALL_SIZES = [3, 8]
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: One hydrogen 0.03 A along y, where a surface fits and differentiates anew.
WATER_DISPLACED = 'O 0 0 0.1173; H 0 0.7872 -0.4692; H 0 -0.7572 -0.4692'
#: A typical compute node's pyscf OpenMP thread count (`lib.num_threads()`).
OMP_THREADS = 16
#: The row fit's tile edge: water's 444 points in 7 tiles, so every rank count
#: owns a different set of them.
ROW_FIT_BLOCK = 64
#: The Davidson residual of every layout here, below the fit's response
#: (tests/test_chain_row_fit.py).
BSE_CONV_TOL = 1e-9
LAYOUTS = {'whole': {}, 'sliced': dict(sliced=True),
           'rows': dict(sliced=True, fit='rows', fit_block=ROW_FIT_BLOCK)}
#: How often each rank takes the PBE0 mirror energy: the race flips it by one
#: ulp only rarely, so one call per rank would almost never show a difference.
MIRROR_REPEATS = 40
#: What `layout_numbers` returns, in order.
NUMBERS = ('composed force', 'energy', 'dRPA force')
#: How many distinct last-bit scalings the shape-sensitive BLAS draws from.
SHAPE_SKEW_CLASSES = 61
ULP = np.finfo(float).eps
NP_OPS = frozenset({'matmul', 'dot', 'tensordot', 'inner', 'vdot', 'einsum'})
#: GEMMs the shape-sensitive BLAS scaled, over every thread.
SHAPE_STATE = {'perturbed': 0}
_SHAPE_LOCK = threading.Lock()


class RankMemoryLib:
    """pyscf.lib as `pyscf.df.grad.rhf` reads it, save `current_memory`,
    which reports a resident size that depends on the rank, so each rank
    blocks the density-fitted gradient's auxiliary index differently."""

    def __init__(self, lib, comm_of):
        self._lib, self._comm_of = lib, comm_of

    def __getattr__(self, name):
        return getattr(self._lib, name)

    def current_memory(self):
        comm = self._comm_of()
        rank = 0 if comm is None else comm.Get_rank()
        return (self._lib.param.MAX_MEMORY - (0.6 + 0.2 * rank), 0)


class ShapeRewriter(ast.NodeTransformer):
    """Every GEMM-shaped call of a module routed through the stand-in."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not isinstance(node.op, ast.MatMult):
            return node
        return ast.copy_location(ast.Call(
            func=ast.Name('__shape_mm__', ast.Load()),
            args=[node.left, node.right], keywords=[]), node)

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if not isinstance(f, ast.Attribute):
            return node
        if (isinstance(f.value, ast.Name) and f.value.id in ('np', 'numpy')
                and f.attr in NP_OPS):
            return ast.copy_location(ast.Call(
                func=ast.Name('__shape_np__', ast.Load()),
                args=[ast.Constant(f.attr)] + node.args,
                keywords=node.keywords), node)
        if f.attr == 'dot' and not (isinstance(f.value, ast.Name) and
                                    f.value.id in ('np', 'numpy', 'lib')):
            return ast.copy_location(ast.Call(
                func=ast.Name('__shape_mdot__', ast.Load()),
                args=[f.value] + node.args, keywords=node.keywords), node)
        return node


class ShapeLoader(importlib.machinery.SourceFileLoader):
    """A source loader that compiles the rewritten tree and caches nothing."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        tree = ShapeRewriter().visit(ast.parse(self.get_data(path), path))
        return compile(ast.fix_missing_locations(tree), path, 'exec',
                       dont_inherit=True)


class ShapeFinder(importlib.abc.MetaPathFinder):
    """Hands every module of the src package to `ShapeLoader`."""

    def find_spec(self, fullname, path, target=None):
        if fullname != 'src' and not fullname.startswith('src.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(
                spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = ShapeLoader(spec.loader.name, spec.loader.path)
        return spec


def chain_scf(m):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    out = scf.RHF(m).density_fit(auxbasis='cc-pvdz-ri')
    out.conv_tol, out.conv_tol_grad, out.max_cycle = 1e-14, 1e-11, 200
    out.kernel()
    return out


def pbe0_scf(m):
    """A PBE0 mean field, whose E_0 needs the Hartree-Fock mirror."""
    out = dft.RKS(m, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
    out.kernel()
    return out


def molecule(atom):
    """`atom` in cc-pVDZ, a Mole of its own."""
    return gto.M(atom=atom, basis='cc-pvdz', verbose=0)


def racing_dgemm(orig, nthreads=OMP_THREADS):
    """NPdgemm under OpenMP: the K split, the partials added in arrival order."""
    local = threading.local()

    def _dgemm(ta, tb, m, n, k, a, b, c, alpha=1, beta=0, oa=0, ob=0, oc=0):
        if oa or ob or oc or not (m and n and k // n > 3 and k // m > 3):
            return orig(ta, tb, m, n, k, a, b, c, alpha, beta, oa, ob, oc)
        rng = getattr(local, 'rng', None)
        if rng is None:
            rng = local.rng = np.random.default_rng(
                int.from_bytes(os.urandom(8), 'little'))
        A = a if ta == 'N' else a.T
        B = b if tb == 'N' else b.T
        if beta == 0:
            c[...] = 0.0
        else:
            c *= beta
        q, r = divmod(k, nthreads)
        bounds = [(t * q + min(t, r), (t + 1) * q + min(t + 1, r))
                  for t in range(nthreads)]
        parts = [alpha * (A[:, k0:k1] @ B[k0:k1]) for k0, k1 in bounds
                 if k1 > k0]
        for i in rng.permutation(len(parts)):
            c += parts[i]
        return c
    return _dgemm


def shape_factor(op, shapes):
    """1 + k ulp, k fixed by the operation and its operands' shapes."""
    k = 1 + zlib.crc32(repr((op, shapes)).encode()) % SHAPE_SKEW_CLASSES
    return 1.0 + k * ULP


def shapes_of(args):
    """The operands' shapes, or their type names, as a GEMM's signature."""
    return tuple(tuple(np.shape(a)) if isinstance(
        a, (np.ndarray, np.generic, list, tuple, float, int))
        else type(a).__name__ for a in args)


def scaled(res, factor, out=None):
    """`res` (or `out`, written in place) times `factor` where it is real."""
    with _SHAPE_LOCK:
        SHAPE_STATE['perturbed'] += 1
    if out is not None:
        out *= factor
        return out
    if isinstance(res, np.ndarray):
        return res * factor if res.dtype.kind in 'fc' else res
    if isinstance(res, (float, complex, np.floating, np.complexfloating)):
        return res * factor
    return res


def contracts(subscripts):
    """Whether an einsum sums over an index -- a GEMM, not a copy."""
    if not isinstance(subscripts, str):
        return True
    s = subscripts.replace(' ', '')
    if '->' in s:
        ins, out = s.split('->')
    else:
        ins = s
        letters = [c for c in s.replace(',', '') if c.isalpha()]
        out = ''.join(c for c in set(letters) if letters.count(c) == 1)
    ins = ins.split(',')
    return len(ins) > 1 and any(c.isalpha() and c not in out
                                for c in ''.join(ins))


def shape_mm(a, b):
    """a @ b on the shape-sensitive BLAS."""
    res = a @ b
    if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        return res
    return scaled(res, shape_factor('matmul', shapes_of((a, b))))


def shape_np(op, *args, **kw):
    """np.<op>(...) on the shape-sensitive BLAS."""
    res = getattr(np, op)(*args, **kw)
    if op == 'einsum':
        if isinstance(args[0], str) and not contracts(args[0]):
            return res
        sig = shapes_of(args[1:]) + (args[0],) if isinstance(args[0], str) \
            else shapes_of(args)
    elif op == 'tensordot':
        sig = shapes_of(args[:2]) + (repr(args[2:]) + repr(kw.get('axes')),)
    else:
        sig = shapes_of(args[:2])
    return scaled(res, shape_factor(op, sig), out=kw.get('out'))


def shape_mdot(obj, *args, **kw):
    """x.dot(y) on the shape-sensitive BLAS where x is an array."""
    if not isinstance(obj, np.ndarray):
        return obj.dot(*args, **kw)
    return shape_np('dot', obj, *args, **kw)


def import_src_on_the_shape_sensitive_blas():
    """Drop every imported src module and import src rewritten from now on."""
    builtins.__shape_mm__ = shape_mm
    builtins.__shape_np__ = shape_np
    builtins.__shape_mdot__ = shape_mdot
    sys.meta_path.insert(0, ShapeFinder())
    for name in [n for n in sys.modules if n == 'src' or n.startswith('src.')]:
        del sys.modules[name]


def layout_numbers(surface_class, mol, mf, here, mf_here, **kw):
    """(composed force, energy, dRPA force) of one layout at `here` on the
    given mean fields."""
    surface = surface_class(mol, chain_scf, spin='singlet', mf=mf,
                            solver='davidson', bse_conv_tol=BSE_CONV_TOL, **kw)
    g, e, _ = surface.total_gradient(here, mf_here)
    return (np.asarray(g), np.atleast_1d(e),
            np.asarray(surface.ground.total_gradient(here, mf_here)[0]))


def layouts_on_one_mean_field(surface_class, layouts):
    """Each layout's numbers on this rank, all on one pair of mean fields."""
    mol, here = molecule(WATER), molecule(WATER_DISPLACED)
    mf, mf_here = chain_scf(mol), chain_scf(here)
    out = {name: layout_numbers(surface_class, mol, mf, here, mf_here, **kw)
           for name, kw in layouts.items()}
    return out, (mol, mf, here, mf_here)


def digest(arrays):
    """The exact bytes of `arrays`, one string."""
    return b''.join(np.ascontiguousarray(a).tobytes() for a in arrays)


def distinct(values):
    """How many different byte strings the ranks hold."""
    return len({digest(v if isinstance(v, (tuple, list)) else (v,))
                for v in values})


def not_rank_0s(res, keys):
    """'key number: n distinct' wherever the ranks hold more than rank 0's."""
    faults = []
    for key in keys:
        for i, name in enumerate(NUMBERS if key in LAYOUTS else (key,)):
            n = distinct([r[key][i] if key in LAYOUTS else r[key]
                          for r in res])
            if n != 1:
                faults.append(f'{key} {name}: {n} distinct' if key in LAYOUTS
                              else f'{key}: {n} distinct')
    return faults


def race_rank(comm):
    """(a) on this rank: every number the race could move, and pyscf's own
    gradient of the locked displaced mean field."""
    out, (mol, _, here, mf_here) = layouts_on_one_mean_field(RPABSESurface,
                                                             LAYOUTS)
    # the chains locked mf_here to rank 0's, so what differs below is the
    # arithmetic of the call alone
    out['pyscf mean-field gradient'] = np.asarray(mf_here.Gradients().kernel())
    out['mean_field_force'] = mean_field_force(mf_here)
    ks = lockstep_mean_field(pbe0_scf(here))
    out['reference_energy'] = np.array([reference_energy(ks, here)
                                        for _ in range(MIRROR_REPEATS)])
    # density-fitted, so its mean-field force is the gradient the stand-ins
    # reach; the dense route itself runs pyscf's integral-direct C kernels
    g, e, _ = DenseRPASurface(mol, chain_scf).total_gradient(here)
    out['dense dRPA force'] = np.asarray(g)
    out['dense dRPA energy'] = np.atleast_1d(e)
    return out


@pytest.fixture
def pyscf_race(monkeypatch):
    monkeypatch.setattr(numpy_helper, '_dgemm',
                        racing_dgemm(numpy_helper._dgemm))
    monkeypatch.setattr(pyscf_df_grad, 'lib',
                        RankMemoryLib(pyscf_df_grad.lib, current_comm))


@pytest.mark.parametrize('size', SIZES)
def test_every_rank_holds_rank_0s_numbers_under_the_pyscf_race(pyscf_race,
                                                               size):
    res = run_simulated(race_rank, size)
    live = distinct([r['pyscf mean-field gradient'] for r in res])
    assert live > 1, 'the race emulation moved no mean-field gradient'
    faults = not_rank_0s(res, (*LAYOUTS, 'mean_field_force',
                               'reference_energy', 'dense dRPA force',
                               'dense dRPA energy'))
    assert not faults, f'{size} ranks: {faults}'


def run_in_subprocess(mode, size):
    """`shape_main(mode, size)` in a fresh interpreter, its verdict."""
    env = dict(os.environ, MPI4PY_RC_INITIALIZE='0',
               PYTHONPATH=os.pathsep.join(
                   [REPO] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
    out = subprocess.run([sys.executable, os.path.abspath(__file__), mode,
                          str(size)], env=env, cwd=REPO, capture_output=True,
                         text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith('RESULT '))
    return json.loads(line[len('RESULT '):])


@pytest.mark.parametrize('size', SIZES)
def test_layouts_on_the_shape_sensitive_blas(size):
    got = run_in_subprocess('shape', size)
    print(f'{size} ranks: {got}')
    assert got['perturbed'] > 0, 'the shape-sensitive BLAS scaled nothing'
    assert not got['not rank 0s'], got['not rank 0s']
    assert got['sliced is whole'] == [True] * len(NUMBERS), got
    for name, dist in got['row fit'].items():
        bar = (FIT_REALIZATION_ENERGY_TOL if name == 'energy'
               else FIT_REALIZATION_FORCE_TOL)
        print(f'[info] {size} ranks, row-fit {name}: |d| {dist:.2e} = '
              f'{dist / bar:.4f} of {bar:.0e}')
        assert dist <= bar, (size, name, dist, bar)


@pytest.mark.parametrize('size', ALL_SIZES)
def test_every_rank_holds_rank_0s_numbers_on_all_three(size):
    got = run_in_subprocess('all', size)
    assert got['perturbed'] > 0 and got['raced'] > 1, got
    assert not got['not rank 0s'], got['not rank 0s']


def shape_main(mode, size):
    """(b) or (c) here: src imported afresh on the shape-sensitive BLAS, the
    free memory rank-dependent and, for 'all', pyscf racing."""
    import_src_on_the_shape_sensitive_blas()
    fresh_grid = importlib.import_module('src.Base.utils.mpi_grid')
    surface_class = importlib.import_module(
        'src.gradients.rpa_bse_surface').RPABSESurface
    pyscf_df_grad.lib = RankMemoryLib(pyscf_df_grad.lib,
                                      fresh_grid.current_comm)
    if mode == 'all':
        numpy_helper._dgemm = racing_dgemm(numpy_helper._dgemm)

    def rank(comm):
        out, fields = layouts_on_one_mean_field(surface_class, LAYOUTS)
        out['raw'] = np.asarray(fields[3].Gradients().kernel())
        return out

    res = fresh_grid.run_simulated(rank, size)
    verdict = {'perturbed': SHAPE_STATE['perturbed'],
               'raced': distinct([r['raw'] for r in res]),
               'not rank 0s': not_rank_0s(res, LAYOUTS)}
    if mode == 'all':
        return verdict
    verdict['sliced is whole'] = [
        all(np.array_equal(r['sliced'][i], r['whole'][i]) for r in res)
        for i in range(len(NUMBERS))]
    sliced, rows = res[0]['sliced'], res[0]['rows']
    verdict['row fit'] = {
        name: float(np.abs(np.asarray(rows[i]) - np.asarray(sliced[i])).max())
        for i, name in enumerate(NUMBERS)}
    return verdict


if __name__ == '__main__':
    if len(sys.argv) > 1:              # the child of `run_in_subprocess`
        print('RESULT ' + json.dumps(shape_main(sys.argv[1],
                                                int(sys.argv[2]))), flush=True)
    else:
        sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
