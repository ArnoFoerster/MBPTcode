"""The slab validity check is actually invoked by the integral builders.

check_low_dim_support was written, correct, and never called. A validity check
nothing invokes provides no protection -- and this is the one whose violation
is SILENT and looks like poor k-convergence: the damped kernel's real-space
support grows with the k-grid while the vacuum does not, so a slab that is fine
on a coarse mesh stops being fine on a finer one with no visible symptom.

make_coulG_damped tags its kernel with the (r0, beta) it was built from, so
a builder handed only a coulG_fn can still run the check (assert_damping_fits).

Checks:
  1. kmesh_from_kpts inverts make_kpts, and refuses a non-regular k-set.
  2. The check is a no-op for a kernel that carries no damping envelope.
  3. Refining the k-mesh at FIXED vacuum eventually trips it -- the failure
     mode the docstring warns about.
  4. Every builder that takes a coulG_fn actually runs it: each raises on a
     slab whose vacuum is too small for its mesh, and each still builds with
     check_support=False.
  5. A PERIODIC direction carrying a single k-point is not flagged. Only a
     genuinely non-periodic one is. Same cell and same k-mesh, dimension=3 vs
     dimension=2: bulk sampled coarsely along a3 is coarse, not a slab, and
     its overlapping images are the physics. Wiring the check in is what made
     this distinction load-bearing -- the gate existed but never applied.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_rpa_damping import (
    nyquist_params, make_coulG_damped, damping_support, kmesh_from_kpts,
    assert_damping_fits)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_rpa import ri_rpa_ecorr
from src.SingleReference.Periodic.pbc_solvent_screening import (
    SlabDielectricEnvironment, build_dfintegrals_screened)
from src.Base.constants import BOHR_TO_ANGSTROM



def check(ok, label, detail=''):
    """Report a condition AND enforce it.

    pytest DISCARDS whatever a test function returns, so a suite built out of
    `return ok` passes whether ok is True or False -- verified on pytest 9.1.1.
    """
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    assert ok, f"{label}{(' -- ' + detail) if detail else ''}"
    return ok


def make_cell(lz, mesh):
    cell = gto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, lz])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    return cell


def test_kmesh_inverse():
    cell = make_cell(14.0, [13, 13, 42])
    ok = True
    for km in ([1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [2, 3, 1]):
        got = kmesh_from_kpts(cell, cell.make_kpts(km))
        ok &= check(got == tuple(km), f'kmesh_from_kpts inverts make_kpts({km})', str(got))
    try:
        kmesh_from_kpts(cell, cell.make_kpts([2, 2, 1])[:3])
    except ValueError:
        ok &= check(True, 'a non-regular k-set is refused')
    else:
        ok &= check(False, 'a non-regular k-set is refused')
    return ok


def test_undamped_is_noop():
    cell = make_cell(6.0, [13, 13, 20])          # deliberately tiny vacuum
    plain = lambda c, q, Gv: np.zeros(len(Gv))
    return check(assert_damping_fits(cell, cell.make_kpts([4, 4, 1]), plain) is None,
                 'no-op for a kernel with no damping envelope')


def test_refining_the_mesh_trips_it():
    """Fixed vacuum, finer k-mesh: r0 grows, and at some point the kernel wraps."""
    cell = make_cell(14.0, [13, 13, 42])
    verdicts = []
    for km in ([2, 2, 1], [4, 4, 1], [6, 6, 1]):
        r0, beta, _ = nyquist_params(cell, km)
        support = damping_support(r0, beta) * BOHR_TO_ANGSTROM
        try:
            assert_damping_fits(cell, cell.make_kpts(km), make_coulG_damped(r0, beta))
            verdicts.append((km, support, 'ok'))
        except ValueError:
            verdicts.append((km, support, 'raised'))
    for km, support, verdict in verdicts:
        print(f'       kmesh={str(km):9s} support {support:5.2f} A vs half-cell '
              f'{0.5 * 14.0:.2f} A -> {verdict}')
    return check([v[2] for v in verdicts] == ['ok', 'ok', 'raised'],
                 'a fixed vacuum passes on coarse meshes and trips on a fine one')


def test_periodic_axis_is_not_flagged():
    """Identical geometry and k-mesh; only cell.dimension differs."""
    verdicts = {}
    for dimension in (3, 2):
        cell = gto.Cell()
        cell.atom = 'H 0 0 0; H 1.1 1.1 1.1'
        cell.a = np.eye(3) * 3.2
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.dimension = dimension
        cell.mesh = [15, 15, 15]
        cell.verbose = 0
        cell.build()
        kmesh = [2, 2, 1]
        r0, beta, _ = nyquist_params(cell, kmesh)
        try:
            assert_damping_fits(cell, cell.make_kpts(kmesh), make_coulG_damped(r0, beta))
            verdicts[dimension] = 'ok'
        except ValueError:
            verdicts[dimension] = 'raised'
    ok = check(verdicts[3] == 'ok',
               'a periodic a3 at one k-point is NOT flagged (3D bulk, coarse mesh)')
    return ok & check(verdicts[2] == 'raised',
                      'the same cell as a slab IS flagged (a3 is now vacuum)')


def test_every_builder_runs_it():
    """The check must live in the builders, not only in a helper nobody calls."""
    cell = make_cell(8.0, [13, 13, 25])
    kmesh = [4, 4, 1]
    mf = scf.KRHF(cell, cell.make_kpts(kmesh), exxdiv=None).density_fit()
    mf.kernel()
    r0, beta, _ = nyquist_params(cell, kmesh)
    coulG = make_coulG_damped(r0, beta)
    env = SlabDielectricEnvironment(cell, solvent='water', z_center=0.0, z_half_width=2.5)

    builders = {
        'build_dfintegrals_coulG':
            lambda **kw: build_dfintegrals_coulG(mf, coulG_fn=coulG, **kw),
        'ri_rpa_ecorr':
            lambda **kw: ri_rpa_ecorr(mf, coulG_fn=coulG, nw=8, **kw),
        'build_dfintegrals_screened':
            lambda **kw: build_dfintegrals_screened(mf, env, coulG_fn=coulG, **kw),
    }
    ok = True
    for name, build in builders.items():
        try:
            build()
        except ValueError as err:
            ok &= check('wraps around the cell' in str(err), f'{name} runs the check')
        else:
            ok &= check(False, f'{name} runs the check', 'no error raised')
        try:
            build(check_support=False)
            ok &= check(True, f'{name} honours check_support=False')
        except ValueError:
            ok &= check(False, f'{name} honours check_support=False')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. kmesh_from_kpts')
    all_ok &= test_kmesh_inverse()
    print('\n-- 2. no-op without damping')
    all_ok &= test_undamped_is_noop()
    print('\n-- 3. refining the mesh trips a fixed vacuum')
    all_ok &= test_refining_the_mesh_trips_it()
    print('\n-- 5. periodic vs non-periodic single-k axis')
    all_ok &= test_periodic_axis_is_not_flagged()
    print('\n-- 4. every coulG_fn builder runs the check')
    all_ok &= test_every_builder_runs_it()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
