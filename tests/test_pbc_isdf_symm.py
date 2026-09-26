"""Symmetry-adapted ISDF.

The interpolating vectors are fitted only for irreducible transfers; a general
q reaches them through a rotated collocation X^k(S) = phi^k(S r_mu).

The load-bearing test here is `test_mesh_invariance_is_required`, because the
natural implementation -- use the space group -- is wrong in a way that does
not raise, does not look wrong, and costs four orders of magnitude.
"""
import itertools
import os
import sys

import numpy as np
import pytest
from pyscf.pbc import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.Periodic import pbc_isdf as isdf

ALPHA = 20      # symm-vs-plain: 1.9e-2 at alpha=8, 2.2e-7 at alpha=20


def _diamond():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.space_group_symmetry = True
    cell.build()
    return cell


def _quads(cell, kpts):
    ks = cell.get_scaled_kpts(kpts)
    out = []
    for a, b, c in itertools.product(range(len(kpts)), repeat=3):
        d = ks - (ks[a] - ks[b] + ks[c])
        m = np.where(np.linalg.norm(np.round(d) - d, axis=1) < 1e-8)[0]
        if len(m) == 1:
            out.append((a, b, c, int(m[0])))
    return out


@pytest.fixture(scope='module')
def symfix():
    cell = _diamond()
    kpts = cell.make_kpts([2, 2, 1])
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    mf.kernel()
    assert mf.converged
    mo = [np.asarray(c) for c in mf.mo_coeff]
    nmo = mo[0].shape[1]
    return cell, kpts, mo, nmo, np.asarray(mf.mo_energy), _quads(cell, kpts)


def test_cartesian_rotations_are_orthogonal_and_map_atoms(symfix):
    cell = symfix[0]
    R = isdf.cartesian_rotations(cell)
    assert abs(np.einsum('sij,skj->sik', R, R) - np.eye(3)).max() < 1e-10
    coords = cell.atom_coords()
    A = cell.lattice_vectors()
    frac = coords @ np.linalg.inv(A)
    for Rs in R:
        g = (coords @ Rs.T) @ np.linalg.inv(A)
        d = g[:, None, :] - frac[None, :, :]
        assert (np.abs(np.round(d) - d) < 1e-8).all(-1).any(-1).all()


def test_mesh_invariance_is_required(symfix):
    """The op set must be the subgroup leaving the k-MESH invariant, not the
    space group -- and the two genuinely differ.

    A 2x2x1 mesh on an FCC lattice does not have the lattice's cubic symmetry:
    only 4 of the 24 operations map it onto itself, while a 2x2x2 mesh keeps
    all 24. Using all 24 on the 2x2x1 mesh does NOT raise -- individual momenta
    still land on mesh points -- it just stops converging, because the Gram
    matrix C^q = sum_k A^{k-q} conj(A^k) is a sum over the whole mesh and under
    S becomes a sum over the ROTATED mesh. Measured: the ERI plateaued at
    1.7e-2 relative where the plain route reached 3.8e-7.
    """
    cell, kpts, *_ = symfix
    R = isdf.cartesian_rotations(cell)
    keep_221 = isdf.mesh_invariant_operations(cell, kpts, R)
    assert len(keep_221) < len(R), "2x2x1 must lose operations"
    assert len(keep_221) == 4 and len(R) == 24

    kpts222 = cell.make_kpts([2, 2, 2])
    keep_222 = isdf.mesh_invariant_operations(cell, kpts222, R)
    assert len(keep_222) == 24, "2x2x2 should keep the full group"


def test_symmetry_reduces_the_transfer_count(symfix):
    cell, kpts, mo, nmo, _, _ = symfix
    _, _, info = isdf.build_isdf_kpts_symm(cell, mo, kpts, 4 * nmo)
    assert info['nq_solved'] < info['nq_total']
    assert info['nop_mesh_invariant'] <= info['nop_group']


def test_symmetry_adapted_eri_matches_the_plain_route(symfix):
    """Same factorization, fewer solves. They are equally valid members of one
    approximation family rather than identical objects at finite rank, so they
    converge TO each other as the rank grows: 1.9e-2 at alpha=8, 2.1e-3 at 12,
    3.0e-5 at 16, 2.2e-7 at 20.
    """
    cell, kpts, mo, nmo, _, quads = symfix
    X, V, i1 = isdf.build_isdf_kpts(cell, mo, kpts, ALPHA * nmo)
    Xs, Vs, i2 = isdf.build_isdf_kpts_symm(cell, mo, kpts, ALPHA * nmo)
    worst = 0.0
    for key in quads:
        a = isdf.thc_eri_kpts(X, V, i1['kminus'], *key)
        b = isdf.thc_eri_kpts_symm(Xs, Vs, i2, *key)
        d = abs(a - b).max() / abs(a).max()
        worst = d if not (d <= worst) else worst
    assert worst < 1e-5, f"symmetry-adapted vs plain: {worst:.3e}"


def test_degenerate_block_guard(symfix):
    """An orbital window that cuts a degenerate shell breaks eq 9
    silently, and degeneracies live at exactly the high-symmetry k-points the
    IBZ reduction leans on."""
    _, _, _, nmo, mo_energy, _ = symfix
    isdf.check_degenerate_blocks_intact(mo_energy)          # full set: fine
    e = np.array([[0.0, 1.0, 2.0, 2.0, 3.0]])
    isdf.check_degenerate_blocks_intact(e, nmo_kept=2)       # shell boundary
    with pytest.raises(ValueError, match='degenerate shell'):
        isdf.check_degenerate_blocks_intact(e, nmo_kept=3)   # splits the pair

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
