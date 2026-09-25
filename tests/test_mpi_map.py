"""`mpi_grid.mpi_map` spreads whole calculations over ranks and every rank gets
the full, ordered result -- checked under simulated ranks, where the reduction
runs through threads of this process (MPI cannot start in the sandboxed test
runner; tests/test_mpi_grid_distribution.py covers the wire under mpirun).

Gated:
  * order and completeness: the result equals the builtin map for 2, 3 and 7
    ranks, including more ranks than items and an empty item list;
  * every rank evaluates only its stripe, and every item exactly once overall;
  * a `FiniteDifferenceGradient` on a cheap analytic surface returns the SAME
    BITS with `map_fn=partial(mpi_map, comm=...)` as serially: the displaced
    energies are the same numbers and the difference is taken in the same
    arithmetic, whichever rank produced them;
  * `numerical_hessian` accepts the same map and agrees bitwise too.
"""
import functools
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base.utils.mpi_grid import mpi_map, run_simulated
from src.properties.surface import FiniteDifferenceGradient

SIZES = [2, 3, 7]


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('n_items', [0, 1, 5, 12])
def test_map_is_ordered_and_complete(size, n_items):
    items = [f'item{i}' for i in range(n_items)]
    seen = [[] for _ in range(size)]
    lock = threading.Lock()

    def one_rank(comm):
        def fn(x):
            with lock:
                seen[comm.Get_rank()].append(x)
            return (x.upper(), len(x))
        return mpi_map(fn, items, comm=comm)

    for out in run_simulated(one_rank, size):
        assert out == [(x.upper(), len(x)) for x in items]
    evaluated = sorted(sum(seen, []))
    assert evaluated == sorted(items)                  # each item exactly once
    for r, s in enumerate(seen):                       # and on its own stripe
        assert s == [items[i] for i in range(r, n_items, size)]


def test_map_without_a_comm_is_the_builtin():
    assert mpi_map(lambda x: x * 2, range(5)) == [0, 2, 4, 6, 8]


class _Spring:
    """An energy-only surface: harmonic springs between every atom pair."""

    def __init__(self, mol):
        self.mol0 = mol

    def total_energy(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        c = np.asarray(mol.atom_coords())
        e = 0.0
        for i in range(mol.natm):
            for j in range(i):
                e += 0.5 * (np.linalg.norm(c[i] - c[j]) - 1.7) ** 2
        return e + 0.01 * np.sum(c ** 3)

    def refreeze(self, mol):
        return _Spring(mol)

    def label(self):
        return 'springs'


def _mol():
    return gto.M(atom='H 0 0 0; H 0 0 1.4; H 1.2 0.3 0.7', basis='sto-3g', spin=1,
                 unit='Bohr', verbose=0)


@pytest.mark.parametrize('size', [2, 3])
def test_finite_difference_gradient_is_bitwise_under_the_map(size):
    mol = _mol()
    serial = FiniteDifferenceGradient(_Spring(mol)).total_gradient(mol)

    def one_rank(comm):
        fd = FiniteDifferenceGradient(_Spring(mol),
                                      map_fn=functools.partial(mpi_map, comm=comm))
        return fd.total_gradient(mol)

    for grad, e0, info in run_simulated(one_rank, size):
        assert np.array_equal(grad, serial[0])
        assert e0 == serial[1] and info == serial[2]
    assert np.abs(serial[0]).max() > 0


@pytest.mark.parametrize('size', [2, 3])
def test_numerical_hessian_accepts_the_map(size):
    # the Hessian module is newer than the rest of this test's imports and may
    # not be in every checkout this test runs against
    H = pytest.importorskip('src.properties.hessian')
    numerical_hessian = H.numerical_hessian
    mol = _mol()

    class Springs:
        """scf_factory stand-in: `mean_field_force` is not used; the Hessian
        differences whatever `gradient_at` returns, so patch the force."""

    spring = _Spring(mol)

    def fake_gradient_at(m, scf_factory, ia, x, sign, step):
        crd = np.asarray(m.atom_coords())
        d = np.zeros_like(crd)
        d[ia, x] = sign * step
        moved = m.copy()
        moved.set_geom_(crd + d, unit='Bohr')
        moved.build(False, False)
        return FiniteDifferenceGradient(spring, h=1e-4).total_gradient(moved)[0]

    original = H.gradient_at
    H.gradient_at = fake_gradient_at
    try:
        serial = numerical_hessian(mol, None)

        def one_rank(comm):
            return numerical_hessian(mol, None,
                                     map_fn=functools.partial(mpi_map, comm=comm))

        for hess in run_simulated(one_rank, size):
            assert np.array_equal(hess, serial)
    finally:
        H.gradient_at = original


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
