"""Nothing in the ISDF block action is computed at full grid length per rank.

Besides the row split of Zt (the three M^2 n_occ passes), each of the four
smaller terms follows the rows, with the collective its shape asks for:

  z X_v^T   an output partition: a rank builds the grid rows it owns and the
            ranks all-gather them (`mpi_grid.allgather_blocks`), because the
            exchange reads the whole grid index; each row is computed once,
            by its owner.
  p, p D    a reduction: p is per grid point, so a rank forms its own rows
            and contracts them with its own rows of D; naux doubles cross.
  the tail  a reduction: X_o^T (Zt * P) is partial over the rows in its
            first index and whole in its second, so one reduce-scatter of
            (n_occ, M), laid out in owner order, hands each rank only its own
            columns, which it contracts with X_v.

The setup's screened-kernel rows follow the same rule: serially they are
D_mine (W D^T), naux^2 M multiply-adds and an (naux, M) array on every rank;
under a comm they are (D_mine W) D^T, nmine naux^2 + nmine naux M with an
(nmine, naux) intermediate, both divided by the rank count. The two
associations differ in their last bits. The TDHF branch, D_mine D^T, has no
inner product to reassociate.

Gated on water/cc-pVDZ and ethylene/cc-pVDZ Hartree-Fock (ethylene's 888 grid
points make the row blocks more than a handful of rows):

  * at 2 and 3 simulated ranks the roots lie within the Davidson tolerance's
    bar of the serial ones (`davidson_root_bar`; the split reassociates the
    sums it reduces), and every rank returns the same bits after the same
    number of cycles;
  * every rank reports its own block-action count and time, collected in
    `davidson_block_action_by_rank`;
  * the per-rank head work, read off the shapes of the GEMM the action runs,
    falls with the rank count and sums over the ranks to the serial head;
  * one rank's slice, scaled by 1 + 1e-6 before it travels, moves the roots,
    separately for the gathered z X_v^T, the reduced tail and the kernel rows
    this rank builds;
  * the reassociated kernel rows agree with the serial association to 1e-13
    relative, and the TDHF rows are the same either way to a few ulp.

There is no comparison with an older tree: the serial-against-distributed
gates hold the split itself, and a tolerance against a tree from before the
Davidson and the GW layer moved on purpose would measure those changes.
"""
import copy
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics import Numerics
from src.Base.utils.mpi_grid import (contiguous_block, run_simulated,
                                     simulated_world)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import block_actions
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.block_actions import isdf_block_action
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.trial_space import ActionNotSymmetric
from tests.isdf_test_grid import g1
from tests.pins import assert_same, bitwise, davidson_root_bar

SIZES = [2, 3]
NROOTS = 3
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
MOLECULES = {
    'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'ethylene': ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321'),
}
#: Ha. The row split sums the ranks' partials in rank order instead of inside
#: one GEMM, so a distributed Davidson takes another path to roots certified
#: only to its residual tolerance: the bar follows the tolerance the solves
#: ask for (`davidson_root_bar`).
CONV_TOL = Numerics().bse.conv_tol
ROOT_TOL = davidson_root_bar(CONV_TOL)
#: The perturbation one rank's slice carries, and the smallest root shift
#: that counts as the answer having noticed it.
SLICE_SCALE = 1.0 + 1e-6
MOVED = 1e-9
#: Relative to max|Zt|. The distributed association of this rank's kernel rows
#: against the serial one: the same three factors multiplied in the other
#: order, separated by double-precision rounding over an naux-long sum
#: (1.5e-16 at water, 1.1e-15 at ethylene).
REASSOCIATION_TOL = 1e-13

class _MatmulLog(np.ndarray):
    """A trial vector that records the shapes of every matmul it enters.

    The head of the block action is one GEMM of the vector against X_v, so
    the grid length that GEMM runs over is the length this rank's head paid
    for: M serially, its own row block under a comm. Reading it off the
    operands measures it rather than restating `contiguous_block`.
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
    factors = separable_factors(mf, mol, auxbasis=AUXBASIS, counts=g1(mol))
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
    """The three lowest BSE@G0W0 roots of each molecule, this tree, one rank.

    Divided by the bare d, asked for; the default is the screened diagonal."""
    out = {}
    for name, c in cases.items():
        omega, X, Y, _ = solve_bse_isdf(c['mf'], c['mol'], c['nocc'],
                                        numerics=Numerics(bse={'nroots': NROOTS, 'preconditioner': 'bare'}), probe=False,
                                        progress=False, factors=c['factors'])
        out[name] = (omega, X, Y)
    return out


def rank_copy(c):
    """This rank's own mean field and factors: a distributed solve replicates
    rank 0's over them in place, and simulated ranks share one process."""
    mf = copy.copy(c['mf'])
    mf.mo_energy = np.asarray(c['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(c['mf'].mo_coeff, float).copy()
    return mf, tuple(np.array(a, copy=True) for a in c['factors'])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_distributed_roots_and_per_rank_counters(cases, serial_roots, name,
                                                 size):
    """The roots of the split action within `davidson_root_bar` of the serial ones (the
    reduced terms sum in rank order, so not bitwise), bitwise across the
    ranks after the same cycle count, and the per-rank block-action counters.

    Both solves use the bare diagonal preconditioner to isolate the row
    split's reassociation: the default screened diagonal is itself a
    reduction over grid rows whose last bits move with the rank count, and
    is gated in test_davidson_preconditioner and test_probe_after_davidson.
    """
    c = cases[name]
    omega0 = serial_roots[name][0]

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, X, Y, info = solve_bse_isdf(mf, c['mol'], c['nocc'], numerics=Numerics(bse={'nroots': NROOTS, 'preconditioner': 'bare'}),
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
    """The head's own GEMM shapes, serially and per rank: each grid point's
    head is paid once across the ranks (the lengths sum to M) and no rank pays
    more than its block, so the head falls with the rank count."""
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
    """One rank's own slice, scaled before it travels, moves the roots.

    The last rank is perturbed, so rank 0's copy standing in for everyone's
    would be caught. `qp=False` puts the BSE on the mean field, keeping the
    quasiparticle stage's own reductions out of the gate. The patch is
    asserted to have run, so the gate cannot pass on a collective the action
    does not make.

    The two pieces differ in what the answer is. Scaling the gathered slice
    of one rank makes the action non-symmetric, and the Davidson refuses such
    an action (`ActionNotSymmetric`) instead of returning moved roots, so the
    gathered piece is asserted to be refused. Scaling the reduced tail keeps
    the projected action symmetric and must move the roots.
    """
    c = cases['water']
    target = size - 1
    fired = []

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, _, _, _ = solve_bse_isdf(mf, c['mol'], c['nocc'], numerics=Numerics(bse={'nroots': NROOTS}),
                                     probe=False, progress=False, qp=False,
                                     factors=factors, distribute=True,
                                     comm=comm)
        return om

    clean = run_simulated(one_rank, size)[0]

    real_gather = block_actions.allgather_blocks
    real_reduce = block_actions.reduce_scatter_rows
    tail_shape = (c['npts'], c['nocc'])                 # in owner order

    def gather(a, comm):
        if (piece == 'gathered_zXv' and comm is not None
                and comm.Get_rank() == target):
            start, stop = contiguous_block(a.shape[0], target, comm.Get_size())
            a[start:stop] *= SLICE_SCALE
            fired.append(1)
        return real_gather(a, comm)

    def reduce(a, comm, out=None):
        if (piece == 'reduced_tail' and comm is not None
                and comm.Get_rank() == target and a.shape == tail_shape):
            a *= SLICE_SCALE
            fired.append(1)
        return real_reduce(a, comm, out=out)

    monkeypatch.setattr(block_actions, 'allgather_blocks', gather)
    monkeypatch.setattr(block_actions, 'reduce_scatter_rows', reduce)
    if piece == 'gathered_zXv':
        with pytest.raises(ActionNotSymmetric):
            run_simulated(one_rank, size)
        assert fired, f'the {piece} perturbation never ran'
        return
    moved = run_simulated(one_rank, size)[0]
    assert fired, f'the {piece} perturbation never ran'
    assert np.abs(moved - clean).max() > MOVED


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_reassociated_kernel_rows_match_the_serial_association(cases, name,
                                                               size):
    """This rank's rows of the screened kernel in the distributed association
    against the serial one: different bits (the inner product differs),
    agreeing to REASSOCIATION_TOL; the TDHF rows are the same to a few ulp
    (the test forms the serial rows again)."""
    c = cases[name]
    D, W_aux, npts = c['factors'][1], c['W_aux'], c['npts']
    comms = simulated_world(size)
    for r in range(size):
        r0, r1 = contiguous_block(npts, r, size)
        D_mine = D[r0:r1]
        serial = D_mine @ (W_aux @ D.T)
        assert_same(block_actions._screened_rows(D_mine, D, W_aux), serial,
                    quantity='root', label=(size, r, 'serial association'))
        rows = block_actions._screened_rows(D_mine, D, W_aux, comms[r])
        assert not bitwise(rows, serial)        # the other order, and it fired
        assert (np.abs(rows - serial).max()
                <= REASSOCIATION_TOL * np.abs(serial).max())
        assert_same(block_actions._screened_rows(D_mine, D, None, comms[r]),
                    D_mine @ D.T, quantity='root', label=(size, r, 'TDHF'))


@pytest.mark.parametrize('size', SIZES)
def test_a_perturbed_kernel_row_block_moves_the_roots(cases, monkeypatch,
                                                      size):
    """One rank's D_mine W, scaled before it meets D^T, moves the roots, so
    the roots are built from that intermediate. The last rank is perturbed,
    and the patch is asserted to have run."""
    c = cases['water']
    target = size - 1
    fired = []

    def one_rank(comm):
        mf, factors = rank_copy(c)
        om, _, _, _ = solve_bse_isdf(mf, c['mol'], c['nocc'], numerics=Numerics(bse={'nroots': NROOTS}),
                                     probe=False, progress=False, qp=False,
                                     factors=factors, distribute=True,
                                     comm=comm)
        return om

    clean = run_simulated(one_rank, size)[0]
    real_rows = block_actions._screened_rows

    def rows(D_mine, D, W_aux, comm=None):
        if (W_aux is not None and comm is not None
                and comm.Get_rank() == target):
            fired.append(1)
            return (SLICE_SCALE * (D_mine @ W_aux)) @ D.T
        return real_rows(D_mine, D, W_aux, comm)

    monkeypatch.setattr(block_actions, '_screened_rows', rows)
    moved = run_simulated(one_rank, size)[0]
    assert fired, 'the kernel-row perturbation never ran'
    assert np.abs(moved - clean).max() > MOVED


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
