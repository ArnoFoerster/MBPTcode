"""The five properties the ISDF frame sign convention has to have.

A frame axis SIGN carries no physics: every Lebedev sub-shell is an orbit of the
octahedral group, so negating an axis maps the shell onto itself and permutes
grid rows without moving a point. What the sign does control is ROW ORDER, and
anything indexed by a fixed grid row -- a gradient, a frozen adjoint, a
regression pin on the factors -- sees a row permutation as a discontinuity.

A sign read off the environment as an odd moment sum_j w_j (dhat_j . e_k)^3,
with the first moment as fallback, vanishes identically for an axis with no
neighbour projection, i.e. the out-of-plane axis of ANY planar environment, so
every atom of water would fall through to eigh's arbitrary sign. That is two
defects at once: not reproducible across LAPACK builds, and discontinuous
exactly at the reflection-symmetric geometries; `atomic_frames` avoids both.

No convention can be continuous everywhere -- equivariance at a symmetric
geometry would force an axis to equal its own negative -- so the test of the
convention is not continuity everywhere but: total, deterministic, continuous
THROUGH planarity, covariant, and leaving both the energy and the grid itself
untouched. The last is exact rather than approximate: the two conventions
produce the SAME point set in a different row order, which keeps an SCF
checkpoint written under one rule consistent with a fit built under the other.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df as pyscf_df
from pyscf import dft, gto
from scipy.spatial.transform import Rotation

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.factorization.separable_ri import (FRAME_DECAY, _FRAME_SIGN_REFS,
                                                 atomic_frames, atomic_points,
                                                 build_separable_ri,
                                                 molecular_points_covariant,
                                                 optimize_atomic_radii)
from src.SingleReference.GW.space_time import solve_qp_energy_space_time

COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}


def _water(theta=0.0):
    """Water with one H rotated out of the molecular plane by `theta` radians.

    theta = 0 is planar, which is where the environment-moment sign rule is
    undetermined;
    theta of either sign moves through it.
    """
    y, z = 0.7572, -0.4692
    return gto.M(atom=[('O', (0.0, 0.0, 0.1173)), ('H', (0.0, y, z)),
                       ('H', (np.sin(theta) * abs(y), -np.cos(theta) * y, z))],
                 basis='cc-pvdz', unit='Angstrom', verbose=0)


def _old_sign_rule(mol, axes_by_atom, decay=FRAME_DECAY):
    """The environment-moment sign rule, for comparison only.

    Takes the axes as LINES (which the convention does not change) and applies
    the moment sign choice, so the two conventions can be swept side by side.
    """
    coords = np.asarray(mol.atom_coords())
    out = np.array(axes_by_atom, dtype=float, copy=True)
    for i in range(len(coords)):
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        keep = r > 1e-8
        if not keep.any():
            continue
        dj, w = d[keep] / r[keep, None], np.exp(-r[keep] / decay)
        axes = out[i]
        for k in range(3):
            proj = dj @ axes[k]
            odd = float(np.sum(w * proj**3))
            if abs(odd) > 1e-12:
                if odd < 0:
                    axes[k] = -axes[k]
            else:                                # both moments vanish: chance
                if float(np.sum(w * proj)) < 0:
                    axes[k] = -axes[k]
        if np.linalg.det(axes) < 0:
            axes[2] = -axes[2]
    return out


def _lex(a, nd=9):
    """Rows sorted lexicographically, so two point clouds compare as SETS."""
    a = np.round(a, nd)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def _factors(mol, mf, points, auxmol, V_half):
    """(X_mo, D) for a given point set -- the tail of `separable_factors`."""
    X, _, M = build_separable_ri(mol, points, auxmol=auxmol)
    return X @ mf.mo_coeff, M.T @ V_half


if __name__ == '__main__':
    all_ok = True

    # 1. The rule is TOTAL: no axis is left to chance. An axis perpendicular to
    #    every reference would have no decisive overlap, which is why there are
    #    three linearly independent ones rather than one.
    print('1. sign rule is total and gives proper orthonormal frames:')
    ok1 = True
    worst_overlap, worst_orth, worst_det = np.inf, 0.0, 0.0
    for theta in np.linspace(-0.35, 0.35, 15):
        mol = _water(theta)
        frames, _ = atomic_frames(mol)
        for ia in range(mol.natm):
            ov = np.abs(_FRAME_SIGN_REFS @ frames[ia].T).max(axis=0)
            worst_overlap = min(worst_overlap, ov.min())
            worst_orth = max(worst_orth, np.abs(
                frames[ia] @ frames[ia].T - np.eye(3)).max())
            worst_det = max(worst_det, abs(np.linalg.det(frames[ia]) - 1.0))
    ok1 = worst_overlap > 1e-6 and worst_orth < 1e-12 and worst_det < 1e-12
    all_ok &= ok1
    print(f'   smallest decisive overlap over 15 geometries x 3 atoms x 3 axes'
          f' = {worst_overlap:.3f}')
    print(f'   max |F F^T - I| = {worst_orth:.2e}   max |det F - 1| = '
          f'{worst_det:.2e}   {"OK" if ok1 else "FAIL"}')

    # 2. Continuity THROUGH planarity, which is the defect. Sweep one H through
    #    the molecular plane and watch the frame; the moment rule is swept
    #    alongside on the same axes, so the comparison isolates the sign.
    print('\n2. frame continuity through the planar geometry (theta = 0):')
    thetas = np.linspace(-0.02, 0.02, 21)
    new_frames, old_frames = [], []
    for theta in thetas:
        mol = _water(theta)
        f, _ = atomic_frames(mol)
        new_frames.append(f)
        old_frames.append(_old_sign_rule(mol, f))
    new_jump = max(np.abs(new_frames[i + 1] - new_frames[i]).max()
                   for i in range(len(thetas) - 1))
    old_jump = max(np.abs(old_frames[i + 1] - old_frames[i]).max()
                   for i in range(len(thetas) - 1))
    # A step of 0.002 rad may rotate an axis by ~1e-3; a sign flip moves it by 2.
    ok2 = new_jump < 0.05 and old_jump > 1.0
    all_ok &= ok2
    print(f'   largest frame change between adjacent theta (step '
          f'{thetas[1] - thetas[0]:.4f} rad):')
    print(f'     reference rule (generic refs) {new_jump:.2e}   continuous')
    print(f'     retired rule (odd moment)     {old_jump:.2e}   '
          f'{"= a sign flip" if old_jump > 1.0 else ""}')
    print(f'   {"OK" if ok2 else "FAIL"}')

    # 3. The grid must still rotate with the molecule. Signs cannot break this:
    #    they permute rows, and the axes as LINES are covariant.
    print('\n3. grid point set is covariant under a global rotation:')
    rad = {el: optimize_atomic_radii(el, 'cc-pvdz', 'cc-pvdz-ri',
                                     counts=COUNTS)[0] for el in ('O', 'H')}
    zero = {el: False for el in rad}
    base_mol = _water(0.0)
    base_pts = molecular_points_covariant(base_mol, rad, origin_by_element=zero)
    ok3, worst_rot = True, 0.0
    for R in Rotation.random(6, random_state=0).as_matrix():
        atom = [(base_mol.atom_pure_symbol(ia), tuple(R @ base_mol.atom_coord(ia)))
                for ia in range(base_mol.natm)]
        rot_mol = gto.M(atom=atom, basis='cc-pvdz', unit='Bohr', verbose=0)
        rot_pts = molecular_points_covariant(rot_mol, rad, origin_by_element=zero)
        d = np.abs(_lex(rot_pts) - _lex(base_pts @ R.T)).max()
        worst_rot = max(worst_rot, d)
    ok3 = worst_rot < 1e-8
    all_ok &= ok3
    print(f'   max |grid(R.mol) - R.grid(mol)| as a set, over 6 rotations = '
          f'{worst_rot:.2e}   {"OK" if ok3 else "FAIL"}')

    # 4. The energy is gauge-invariant. Every alternative convention differs
    #    from this one by a proper sign pattern, so if the QP energy survives
    #    all four the choice cannot move a published number -- which is what
    #    makes changing the convention safe for a queued run.
    print('\n4. QP energy is invariant under every proper sign pattern:')
    mol = _water(0.0)
    mf = dft.RKS(mol, xc='PBE')
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    V = auxmol.intor('int2c2e', aosym='s1')
    wv, vv = np.linalg.eigh(V)
    keep = wv > 1e-12 * wv.max()
    V_half = (vv[:, keep] * np.sqrt(wv[keep])) @ vv[:, keep].T

    frames, _ = atomic_frames(mol)
    energies = {}
    # det = +1 keeps it a rotation, so an even number of axes may flip.
    for flip in ((), (0, 1), (0, 2), (1, 2)):
        F = frames.copy()
        for k in flip:
            F[:, k] = -F[:, k]
        pts = np.vstack([
            atomic_points(rad[mol.atom_pure_symbol(ia)], centre=(0.0, 0.0, 0.0),
                          origin=False) @ F[ia] + mol.atom_coord(ia)
            for ia in range(mol.natm)])
        X_mo, D = _factors(mol, mf, pts, auxmol, V_half)
        e = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1,
                                       factors=(X_mo, D)) * HARTREE_TO_EV
        energies[flip or 'none'] = e
    ref = energies['none']
    spread = max(abs(v - ref) for v in energies.values()) * 1e3
    ok4 = spread < 1e-3
    all_ok &= ok4
    for k, v in energies.items():
        print(f'   flip {str(k):8s} HOMO {v:.9f} eV   d={abs(v - ref) * 1e3:.6f} meV')
    print(f'   spread {spread:.6f} meV   {"OK" if ok4 else "FAIL"}')

    # 5. The strongest form of "gauge": the two conventions do not merely give
    #    the same energy, they give the SAME GRID, bit for bit, in a different
    #    row order. Only the axes as lines place a point, and those are
    #    sign-independent (including through the degenerate-subspace branch,
    #    where e_a depends on `fixed` only via the sign-invariant
    #    (ref.fixed)*fixed). So an SCF checkpoint converged under one
    #    convention stays consistent with a fit built under the other.
    print('\n5. retired and reference conventions give the SAME grid as a set:')
    ok5, worst_same = True, 0.0
    for theta in (0.0, 0.05, -0.05):
        mol = _water(theta)
        new_f, _ = atomic_frames(mol)
        old_f = _old_sign_rule(mol, new_f)
        flipped = np.abs(new_f - old_f).max()
        pts = {}
        for tag, F in (('new', new_f), ('old', old_f)):
            pts[tag] = np.vstack([
                atomic_points(rad[mol.atom_pure_symbol(ia)],
                              centre=(0.0, 0.0, 0.0), origin=False) @ F[ia]
                + mol.atom_coord(ia) for ia in range(mol.natm)])
        as_set = np.abs(_lex(pts['new']) - _lex(pts['old'])).max()
        as_rows = np.abs(pts['new'] - pts['old']).max()
        worst_same = max(worst_same, as_set)
        ok5 &= as_set == 0.0
        print(f'   theta={theta:+.2f}  frames differ by {flipped:.2f}  ->  '
              f'grid as SET {as_set:.2e}   as ORDERED rows {as_rows:.2e}')
    all_ok &= ok5
    print(f'   {"identical point sets, permuted rows only" if ok5 else "GRID MOVED"}'
          f'   {"OK" if ok5 else "FAIL"}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
