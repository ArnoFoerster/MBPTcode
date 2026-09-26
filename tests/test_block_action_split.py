"""Nothing in the ISDF block action is computed at full grid length per rank.

The row split of Zt divides the three M^2 n_occ passes, and for a while that
was all it divided: z X_v^T, the diagonal p of P, p D and the tail contraction
X_o^T (Zt * P) X_v ran WHOLE on every rank, an Amdahl floor the docstring of
`isdf_block_action` put at 15% of a four-rank action and the cluster measured
as 1.75x on two ranks where the split alone predicts 2x.

Each of the four now follows the rows, and each takes the collective its shape
asks for:

  z X_v^T   an output partition -- a rank builds the grid rows it owns and the
            ranks ALL-GATHER them (`mpi_grid.allgather_blocks`), because the
            exchange reads the whole grid index. Gathered, not summed: a row
            is computed once, by its owner.
  p, p D    a reduction -- p is per grid point, so a rank forms its own rows
            and contracts them with its own rows of D; naux doubles cross.
  the tail  a reduction -- X_o^T (Zt * P) is partial over the rows in its
            first index and whole in its second, so one reduce of (n_occ, M)
            makes it the same everywhere and each rank then contracts only its
            own columns with X_v.

Gated here, on water/cc-pVDZ and ethylene/cc-pVDZ Hartree-Fock (888 grid points
against water's 444, so the row blocks are not a handful of rows):

  * the SERIAL action is bitwise what it was, against the roots and vectors of
    the code before any rank split existed (`BASELINE_COMMIT`), extracted with
    `git archive` and run in its own process. Serially the whole split
    collapses -- the row block is the whole grid -- and every expression must
    be the one it replaces, character for character in the arithmetic if not
    in the source;
  * at 2 and 3 simulated ranks the roots sit within 1e-11 Ha of the serial
    ones, which is the standard the row split has always been held to (it
    re-associates the sums it reduces, so it is not bitwise), and every rank
    returns the same bits as every other and ran as many cycles: each runs the
    same iteration on the lockstepped action output;
  * every rank reports its own block-action count and time, and
    `davidson_block_action_by_rank` carries all of them, which is the number
    the cluster reads the scaling off;
  * the per-rank HEAD work, measured off the shapes of the GEMM the action
    actually runs, falls with the rank count and sums across the ranks to the
    serial head -- no grid point's head is paid twice;
  * one rank's own slice, perturbed by 1 + 1e-6 before it travels, MOVES the
    roots -- separately for the gathered z X_v^T, for the reduced tail and for
    the kernel rows this rank builds. A split whose pieces did not reach the
    answer would swallow the perturbation, and the gate fails if any does.

The SETUP takes the same treatment as the loop. This rank's rows of the
screened kernel are D_mine (W D^T) serially, where the inner product is naux^2
M multiply-adds and an (naux, M) array on every rank; under a rank count they
are (D_mine W) D^T, the same matrix from nmine naux^2 + nmine naux M and an
(nmine, naux) intermediate, both of which the rank count divides. The two
associations differ in their last bits, so the serial one stands as it was --
the bitwise gate above is what holds it there -- and the reassociated rows are
gated against it at the level double precision allows, 1e-13 relative, where
water and ethylene measure 1.5e-16 and 1.1e-15. The TDHF branch, D_mine D^T,
has no inner product to reassociate and is bitwise the same either way.
"""
import copy
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import (contiguous_block, run_simulated,
                                     simulated_world)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         isdf_block_action,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

SIZES = [2, 3]
NROOTS = 3
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
MOLECULES = {
    'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'ethylene': ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321'),
}
#: Ha. The row split sums the ranks' partials in rank order instead of inside
#: one GEMM, so a distributed root differs from the serial one in its last
#: bits. Every distributed BSE probe in this repository has landed inside this.
ROOT_TOL = 1e-11
#: The perturbation one rank's slice carries, and the smallest root shift that
#: counts as the answer having noticed it. 1e-6 on a slice is six orders above
#: ROOT_TOL; anything that reaches the reduction cannot hide inside it.
SLICE_SCALE = 1.0 + 1e-6
MOVED = 1e-9
#: Relative to max|Zt|. The distributed association of this rank's kernel rows
#: against the serial one: two products of the same three factors in the other
#: order, so what separates them is double precision over an naux-long sum.
#: Measured 1.5e-16 at water and 1.1e-15 at ethylene, at 2 and 3 ranks.
REASSOCIATION_TOL = 1e-13

REPO = Path(__file__).resolve().parents[1]
#: The commit before the block action was split over ranks at all, so the
#: bitwise gate compares this tree against code that could not have moved a
#: bit for the reason being tested here. Pinned rather than `HEAD`, which
#: once this change lands would compare the tree with itself.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
#: A shared machine: the archived probe is capped rather than left to size
#: itself against the whole node, as every subprocess gate in this repo is.
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: Built fresh in its own process against the extracted tree, on the same two
#: molecules the fixture below builds.
SERIAL_PROBE = '''
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

warnings.simplefilter('ignore')
out = {{}}
for name, atom in {molecules!r}.items():
    mol = gto.M(atom=atom, basis={basis!r}, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis={auxbasis!r})
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis={auxbasis!r})
    omega, X, Y, _ = solve_bse_isdf(mf, mol, nocc, nroots={nroots}, probe=False,
                                    progress=False, factors=factors)
    out[name + '_omega'] = np.asarray(omega)
    out[name + '_X'] = np.asarray(X)
    out[name + '_Y'] = np.asarray(Y)
np.savez({out!r}, **out)
'''


class _MatmulLog(np.ndarray):
    """A trial vector that records the shapes of every matmul it enters.

    The head of the block action is one GEMM of the vector against X_v, so the
    grid length THAT GEMM runs over is the grid length this rank's head paid
    for -- M serially, its own row block under a comm. Reading it off the
    operands is a measurement; a count derived from the partition alone would
    only restate `contiguous_block`.
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        plain = tuple(_plain(x) for x in inputs)
        if ufunc is np.matmul and method == '__call__':
            self.shapes.append(tuple(np.shape(x) for x in plain))
        if kwargs.get('out') is not None:
            kwargs['out'] = tuple(_plain(x) for x in kwargs['out'])
        return getattr(ufunc, method)(*plain, **kwargs)

    def __array_finalize__(self, obj):
        self.shapes = getattr(obj, 'shapes', None)


def _plain(x):
    """The same memory as a base-class array, so an `out=` still writes here."""
    return x.view(np.ndarray) if isinstance(x, _MatmulLog) else x


def logging_vectors(z):
    """`z` as a `_MatmulLog` with a fresh, per-rank shape log."""
    v = np.asarray(z, float).view(_MatmulLog)
    v.shapes = []
    return v


def head_grid_length(shapes, no, nv):
    """The grid length one rank's z X_v^T ran over, off the recorded shapes.

    Serially the head is z @ X_v^T, (n_occ, n_vir) x (n_vir, M); under a comm
    it is X_v[rows] @ z^T, (nmine, n_vir) x (n_vir, n_occ). Either way the
    free dimension that is not n_occ is the grid length.
    """
    lengths = {b[1] for a, b in shapes if a == (no, nv) and b[0] == nv}
    lengths |= {a[0] for a, b in shapes if b == (nv, no) and a[1] == nv}
    assert len(lengths) == 1, shapes
    return lengths.pop()


def head_flops(no, nv, naux, length):
    """Multiply-adds one trial vector's head and tail run over `length` grid
    points: z X_v^T, p, p D, and the tail X_o^T (Zt * P) X_v."""
    return length * (no * nv + no + naux + no * nv)


def build(name):
    mol = gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis=AUXBASIS)
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(name=name, mol=mol, mf=mf, nocc=nocc, factors=factors,
                W_aux=W_aux, npts=factors[0].shape[0],
                naux=factors[1].shape[1], nvir=mol.nao_nr() - nocc)


@pytest.fixture(scope='module')
def cases():
    warnings.simplefilter('ignore')
    return {name: build(name) for name in MOLECULES}


@pytest.fixture(scope='module')
def serial_roots(cases):
    """The three lowest BSE@G0W0 roots of each molecule, this tree, one rank."""
    out = {}
    for name, c in cases.items():
        omega, X, Y, _ = solve_bse_isdf(c['mf'], c['mol'], c['nocc'],
                                        nroots=NROOTS, probe=False,
                                        progress=False, factors=c['factors'])
        out[name] = (omega, X, Y)
    return out


@pytest.fixture(scope='session')
def archive(tmp_path_factory):
    """The block action before the split, unpacked into a temporary directory."""
    out = tmp_path_factory.mktemp('block_action_baseline')
    tar = out.parent / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
    assert (out / 'src' / 'SingleReference' / 'LinearResponse'
            / 'davidson.py').is_file()
    return out


@pytest.fixture(scope='session')
def archived_roots(archive, tmp_path_factory):
    """(omega, X, Y) per molecule from the pre-split code, its own process."""
    tmp = tmp_path_factory.mktemp('block_action_reference')
    script = tmp / 'probe.py'
    npz = tmp / 'roots.npz'
    script.write_text(SERIAL_PROBE.format(
        archive=str(archive), molecules=MOLECULES, basis=BASIS,
        auxbasis=AUXBASIS, nroots=NROOTS, out=str(npz)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(archive),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(npz))


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def rank_copy(c):
    """This rank's own mean field and factors: a distributed solve replicates
    rank 0's over them in place, and simulated ranks share one process."""
    mf = copy.copy(c['mf'])
    mf.mo_energy = np.asarray(c['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(c['mf'].mo_coeff, float).copy()
    return mf, tuple(np.array(a, copy=True) for a in c['factors'])


@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_serial_roots_are_bitwise_the_archived_ones(name, serial_roots,
                                                    archived_roots):
    """One rank owns every grid row, so the split must reduce to the code it
    replaces -- not to within a tolerance, to the bit."""
    omega, X, Y = serial_roots[name]
    assert bitwise(archived_roots[name + '_omega'], omega)
    assert bitwise(archived_roots[name + '_X'], X)
    assert bitwise(archived_roots[name + '_Y'], Y)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_distributed_roots_and_per_rank_counters(cases, serial_roots, name,
                                                 size):
    """The roots off the split action, and the counters the cluster reads.

    Not bitwise against serial and cannot be: the reduced terms sum in rank
    order rather than inside one GEMM. Bitwise ACROSS the ranks, which they
    must be -- every one of them runs the same iteration on the same
    lockstepped action output, as its identical cycle count shows, and the
    result is lockstepped from rank 0 at the end.
    """
    c = cases[name]
    omega0 = serial_roots[name][0]

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, X, Y, info = solve_bse_isdf(mf, c['mol'], c['nocc'], nroots=NROOTS,
                                        probe=False, progress=False,
                                        factors=factors, distribute=True,
                                        comm=comm)
        return om, X, Y, info

    out = run_simulated(one_rank, size)
    om0, X0, Y0, info0 = out[0]
    assert np.abs(om0 - omega0).max() <= ROOT_TOL
    by_rank = info0['timings']['davidson_block_action_by_rank']
    assert len(by_rank) == size
    cycles = info0['stats']['davidson_vind_calls']
    assert cycles > 0
    for r, (om, X, Y, info) in enumerate(out):
        assert np.array_equal(om, om0)                  # rank 0's roots
        assert bitwise(X, X0)                           # and rank 0's vectors
        assert bitwise(Y, Y0)
        t = info['timings']
        assert t['davidson_block_action'] > 0.0
        assert t['davidson_block_action_by_rank'] == by_rank
        assert by_rank[r] == t['davidson_block_action']
        # Every rank iterates, and takes the steps rank 0 takes.
        assert info['stats']['davidson_vind_calls'] == cycles


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_head_work_falls_with_the_rank_count(cases, name, size):
    """The head's own GEMM shapes, serially and per rank.

    Every grid point's head is paid exactly once across the ranks -- the
    lengths sum to M -- and no rank pays more than its block, so the head
    falls with the rank count instead of standing as the Amdahl floor it was.
    """
    c = cases[name]
    no, nv, naux, npts = c['nocc'], c['nvir'], c['naux'], c['npts']
    eps = np.asarray(c['mf'].mo_energy, float)
    lr = LinearResponseSolver(eps, spin_mode='restricted')
    z = np.random.default_rng(3).normal(size=(2, no, nv))

    def run(comm=None):
        act, _ = isdf_block_action(lr, c['nocc'], True, c['W_aux'],
                                   c['factors'], comm=comm)
        v = logging_vectors(z)
        act(v)
        return head_grid_length(v.shapes, no, nv)

    serial = run()
    assert serial == npts                     # one rank owns the whole grid
    lengths = run_simulated(run, size)
    assert sum(lengths) == npts               # nothing computed twice
    mine = [head_flops(no, nv, naux, L) for L in lengths]
    whole = head_flops(no, nv, naux, serial)
    assert max(mine) <= whole * ((npts + size - 1) // size) / npts
    assert max(mine) < whole


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('piece', ['gathered_zXv', 'reduced_tail'])
def test_a_perturbed_rank_slice_moves_the_roots(cases, monkeypatch, size,
                                                piece):
    """One rank's own slice, scaled before it travels, must reach the answer.

    The last rank is the one perturbed, so an implementation that quietly let
    rank 0's copy stand for everyone's would be caught. `qp=False` puts the
    BSE on the mean field: the quasiparticle stage runs its own reductions and
    none of them is what this gate is about.

    The patch is asserted to have FIRED. A perturbation applied to a
    collective that the action no longer makes would leave the roots exactly
    where they were, and the gate would then be passing on nothing.
    """
    c = cases['water']
    target = size - 1
    fired = []

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, _, _, _ = solve_bse_isdf(mf, c['mol'], c['nocc'], nroots=NROOTS,
                                     probe=False, progress=False, qp=False,
                                     factors=factors, distribute=True,
                                     comm=comm)
        return om

    clean = run_simulated(one_rank, size)[0]

    real_gather, real_reduce = davidson.allgather_blocks, davidson.reduce_sum
    tail_shape = (c['nocc'], c['npts'])

    def gather(a, comm):
        if (piece == 'gathered_zXv' and comm is not None
                and comm.Get_rank() == target):
            start, stop = contiguous_block(a.shape[0], target, comm.Get_size())
            a[start:stop] *= SLICE_SCALE
            fired.append(1)
        return real_gather(a, comm)

    def reduce(a, comm):
        if (piece == 'reduced_tail' and comm is not None
                and comm.Get_rank() == target and a.shape == tail_shape):
            a *= SLICE_SCALE
            fired.append(1)
        return real_reduce(a, comm)

    monkeypatch.setattr(davidson, 'allgather_blocks', gather)
    monkeypatch.setattr(davidson, 'reduce_sum', reduce)
    moved = run_simulated(one_rank, size)[0]
    assert fired, f'the {piece} perturbation never ran'
    assert np.abs(moved - clean).max() > MOVED


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_reassociated_kernel_rows_match_the_serial_association(cases, name,
                                                               size):
    """This rank's rows of the screened kernel, built in the order a rank count
    makes cheap, against the order one rank builds them in.

    The distributed rows must DIFFER in their bits -- the whole point is that
    the inner product is a different one -- and agree to what double precision
    over an naux-long sum allows. The TDHF branch has no inner product to
    reassociate and must come back bitwise the same either way.
    """
    c = cases[name]
    D, W_aux, npts = c['factors'][1], c['W_aux'], c['npts']
    comms = simulated_world(size)
    for r in range(size):
        r0, r1 = contiguous_block(npts, r, size)
        D_mine = D[r0:r1]
        serial = D_mine @ (W_aux @ D.T)
        assert bitwise(davidson._screened_rows(D_mine, D, W_aux), serial)
        rows = davidson._screened_rows(D_mine, D, W_aux, comms[r])
        assert not bitwise(rows, serial)        # the other order, and it fired
        assert (np.abs(rows - serial).max()
                <= REASSOCIATION_TOL * np.abs(serial).max())
        assert bitwise(davidson._screened_rows(D_mine, D, None, comms[r]),
                       D_mine @ D.T)


@pytest.mark.parametrize('size', SIZES)
def test_a_perturbed_kernel_row_block_moves_the_roots(cases, monkeypatch,
                                                      size):
    """One rank's D_mine W, scaled before it meets D^T, must reach the answer.

    The reassociation is only worth taking if the intermediate it forms is the
    one the roots are built from; a setup whose rows were quietly rebuilt or
    overwritten elsewhere would swallow this. The last rank is the one
    perturbed, and the patch is asserted to have fired.
    """
    c = cases['water']
    target = size - 1
    fired = []

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, _, _, _ = solve_bse_isdf(mf, c['mol'], c['nocc'], nroots=NROOTS,
                                     probe=False, progress=False, qp=False,
                                     factors=factors, distribute=True,
                                     comm=comm)
        return om

    clean = run_simulated(one_rank, size)[0]
    real_rows = davidson._screened_rows

    def rows(D_mine, D, W_aux, comm=None):
        if (W_aux is not None and comm is not None
                and comm.Get_rank() == target):
            fired.append(1)
            return (SLICE_SCALE * (D_mine @ W_aux)) @ D.T
        return real_rows(D_mine, D, W_aux, comm)

    monkeypatch.setattr(davidson, '_screened_rows', rows)
    moved = run_simulated(one_rank, size)[0]
    assert fired, 'the kernel-row perturbation never ran'
    assert np.abs(moved - clean).max() > MOVED


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
