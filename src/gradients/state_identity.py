"""The same state, or the same orbital, at another geometry.

A walk, a finite difference and a root-following chain all ask which root or
orbital at a displaced geometry continues one at a reference geometry. The
answer is one rule here: the orbitals are matched by `body_overlap`
(Loewdin atomic orbitals that ride with their atoms and turn with the
molecule; `conventions`), and an excited state by the transported RPA metric
of its Casida vectors over the overlap of the two reference determinants
(`state_overlap`, `relative_overlap`), read with `ROOT_FOLLOW_WEIGHT_MIN` and
`ROOT_FOLLOW_MARGIN_MIN`. A plain dot product of two geometries' Casida
vectors has no orbital transport: a sign the SCF gives one orbital at the
displaced geometry, or a rotation of a near-degenerate pair, lowers or
reorders it (CH2O/cc-pVDZ, 0.01 Bohr: one MO sign flip swaps singlet roots 2
and 4 at a cosine of 0.73, where the transported overlap keeps 0.999).

`mo_overlap`, the spatial cross overlap, stays for what it is: the
<phi_p(R)|phi_q(R+h)> a finite-difference derivative coupling differentiates
(`src.properties.nonadiabatic`).

This module sits below the property layer: the chain follows its state here,
and `src.properties` reads the same primitives.
"""
import warnings

import numpy as np
from pyscf import gto, lo, scf as pyscf_scf
from pyscf.gto.mole import ao_rotation_matrix

from src.Base.numerics.constants import (CHARACTER_WINDOW, EXCITED_SPIN_FACTOR,
                                         GROUND_SPIN_FACTOR, ROOT_FOLLOW_MARGIN_MIN,
                                         ROOT_FOLLOW_WEIGHT_MIN)
from src.Base.geometry.body_frame import centred, kabsch_rotation
from src.SingleReference.LinearResponse.linear_response import check_normalization


def mo_overlap(mol_a, mo_a, mol_b, mo_b):
    """T_pq = <phi_p^A | phi_q^B>, the MO overlap between two geometries.

    The atomic-orbital cross overlap is exact for any separation; the basis
    functions of the two geometries sit on different centres and `intor_cross`
    integrates them against each other.

    This is PHYSICS, the overlap <Psi_I(R)|Psi_J(R+h)> a derivative coupling
    is the finite difference of. Whether an orbital or a state at one
    geometry is THE SAME as one at another is decided by
    `body_overlap` (Loewdin orbitals turned with the
    molecule), for the frozen conventions and the followed state alike: the
    spatial overlap loses a core orbital over a 0.2 Bohr step.
    """
    s_ao = gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)
    return np.asarray(mo_a).T @ s_ao @ np.asarray(mo_b)


def state_overlap(t, nocc, x_a, y_a, x_b, y_b):
    """(1 + n_A, 1 + n_B) overlaps <Psi_I(A)|Psi_J(B)>, ground state at 0.

    The transported RPA metric of the module docstring. Row and column 0 are
    the reference determinant, whose ground-to-excited block carries the
    sqrt(2) of the singlet spin adaptation -- the same factor
    `oscillator_strengths` and `spin_orbit.ground_state_element` carry for
    <X|X> - <Y|Y> = 1.

    Exact to O(h^2) in the displacement: the dropped pieces are the alpha-beta
    spin cross term of the determinant overlap and the T^vo (T^oo)^-1 T^ov
    screening of T^vv, both quadratic in the occupied-virtual block of T, which
    is itself O(h). That is the order of the central difference taken of it.
    """
    t = np.asarray(t, float)
    nvir = t.shape[0] - nocc
    x_a, y_a = check_normalization(x_a, y_a)
    x_b, y_b = check_normalization(x_b, y_b)
    n_a, n_b = x_a.shape[1], x_b.shape[1]
    xa = x_a.reshape(nocc, nvir, n_a)
    ya = y_a.reshape(nocc, nvir, n_a)
    xb = x_b.reshape(nocc, nvir, n_b)
    yb = y_b.reshape(nocc, nvir, n_b)
    t_oo, t_ov = t[:nocc, :nocc], t[:nocc, nocc:]
    t_vo, t_vv = t[nocc:, :nocc], t[nocc:, nocc:]
    # (T^oo)^-1 runs ket-occupied by bra-occupied; the closed-shell reference
    # overlap is det(T^oo) once per spin string.
    inv_oo = np.linalg.inv(t_oo)
    det2 = np.linalg.det(t_oo) ** 2

    out = np.zeros((n_a + 1, n_b + 1))
    out[0, 0] = det2
    out[0, 1:] = GROUND_SPIN_FACTOR * det2 * np.einsum(
        'jbJ,jb->J', xb - yb, inv_oo @ t_ov, optimize=True)
    out[1:, 0] = GROUND_SPIN_FACTOR * det2 * np.einsum(
        'iaI,ai->I', xa - ya, t_vo @ inv_oo, optimize=True)
    for u, v in ((xa, xb), (ya, yb)):
        moved = np.einsum('iaI,ji,ab->jbI', u, inv_oo, t_vv, optimize=True)
        block = det2 * np.einsum('jbI,jbJ->IJ', moved, v, optimize=True)
        out[1:, 1:] += EXCITED_SPIN_FACTOR * (block if u is xa else -block)
    return out


def relative_overlap(s):
    """`state_overlap` over S_00 = det(T^oo)^2, the overlap of the two
    reference determinants: the overlap root following reads.

    D multiplies every element and belongs to no excitation: the core
    electrons ride on the nuclei, so D ~ exp(-h^2 <P^2>/2) falls with any
    displacement. A rigid 0.1 Bohr translation leaves formaldehyde/6-31G at
    D = 0.71; one 0.19 Bohr rms step of DMABN/cc-pVDZ at D = 0.019, with the
    relative overlap 0.996. Over D a state overlaps itself to 1 - O(h^2).
    """
    s = np.asarray(s, float)
    return s / s[0, 0]


def align_roots(s, nstates):
    """(order, sign, weight) matching displaced roots to the first `nstates`.

    The displaced solve returns its roots in its own energy order and with an
    arbitrary overall sign, and both are fixed against the reference by the
    overlap matrix the coupling is built from: each reference root takes the
    unused displaced root of largest |S|, then the sign that makes S positive.

    `weight` is the smallest |S| any assignment had to accept. A value far
    below one is a root the reference cannot identify -- a near-degenerate pair
    or a genuine crossing inside the displacement -- and the coupling built
    from it is a number about two states that are not the two asked for.
    """
    s = np.asarray(s, float)
    block = np.abs(s[1:, 1:])
    order, sign = np.zeros(nstates, int), np.ones(nstates)
    # Largest element first, so an unambiguous root is never left with whatever
    # column the scan order hands an ambiguous one.
    rows, columns, weight = set(range(nstates)), set(range(block.shape[1])), 1.0
    while rows:
        i, j = max(((i, j) for i in rows for j in columns),
                   key=lambda p: block[p])
        order[i], sign[i] = j, (1.0 if s[1 + i, 1 + j] >= 0.0 else -1.0)
        weight = min(weight, block[i, j])
        rows.discard(i)
        columns.discard(j)
    return order, sign, float(weight)


def follow_state(s, state):
    """(index, weight, margin) of the displaced root that IS `state`.

    `align_roots` assigns a whole manifold; this follows ONE state, which is
    what a geometry optimization needs. The displaced root of largest |S| with
    the reference state is the same state, whatever energy order it came back
    in.

    weight: |S| with the chosen root. Near 1 the identification is certain;
            far below it the reference state has no counterpart in the
            displaced manifold -- it left the solved window, or `nroots` is
            too small to contain it.
    margin: |S| of the best minus the runner-up. SMALL MARGIN IS THE DANGEROUS
            CASE, and it is not the same as small weight: two roots that have
            mixed share the reference character between them, so both overlaps
            are moderate and neither is the state. Following the larger one
            then picks a state by a coin toss the size of the margin.
    """
    row = np.abs(np.asarray(s, float)[1 + state, 1:])
    if row.size == 0:
        raise ValueError('no displaced roots to follow into')
    order = np.argsort(row)[::-1]
    best = int(order[0])
    runner = float(row[order[1]]) if row.size > 1 else 0.0
    return best, float(row[best]), float(row[best] - runner)


def aligned_overlap(s, nstates):
    """`s` with its displaced columns permuted and signed onto the reference.

    Linear in the displaced vectors, so signing the columns of the overlap is
    the same as signing the vectors and rebuilding it.
    """
    order, sign, weight = align_roots(s, nstates)
    out = np.zeros((nstates + 1, nstates + 1))
    out[:, 0] = s[:nstates + 1, 0]
    out[:, 1:] = s[:nstates + 1, 1 + order] * sign[None, :]
    return out, order, weight


def localized_window(mol, orbitals, pop_method='meta_lowdin'):
    """Pipek-Mezey localized orbitals spanning the columns of `orbitals`.

    pop_method is pyscf's: 'meta_lowdin' (its default, and the one whose
    populations `fragment_populations` counts), 'mulliken', or 'iao' -- the
    last spans the occupied space only and is meaningless for a virtual
    window.
    """
    loc = lo.PM(mol, np.asarray(orbitals, float))
    loc.pop_method = pop_method
    loc.verbose = 0
    return loc.kernel()


def orbital_channel(mf, spin, mo_coeff):
    """(orbital coefficients, occupied count) of the channel asked for."""
    unrestricted = isinstance(mf, pyscf_scf.uhf.UHF)
    if unrestricted and spin not in (0, 1):
        raise ValueError(f'an unrestricted reference needs spin=0 (alpha) or '
                         f'1 (beta), got {spin!r}')
    if not unrestricted and spin is not None:
        raise ValueError(f'spin={spin!r} on a restricted reference')
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff[spin] if unrestricted else mf.mo_coeff
    nocc = int(mf.nelec[spin]) if unrestricted else mf.mol.nelectron // 2
    return np.asarray(mo_coeff, float), nocc


def frontier_window(orbital, nocc, nmo, window):
    """[start, stop) of the canonical orbitals on the orbital's side of the gap,
    from the frontier outwards, widened to hold the orbital itself."""
    if orbital < nocc:
        return max(0, min(nocc - window, orbital)), nocc
    return nocc, min(nmo, max(nocc + window, orbital + 1))


def orbital_fingerprint(mf, orbital, window=CHARACTER_WINDOW, spin=None,
                        mo_coeff=None, pop_method='meta_lowdin'):
    """The orbital at this geometry, as `track_orbital` follows it to another.

    Its weight on each Pipek-Mezey localized orbital of the window, together
    with the localized orbitals and the molecule they live on. A canonical
    orbital near a near-degeneracy is an arbitrary rotation within it; the
    localized set is not, which is what makes it the reference to follow.
    """
    coeff, nocc = orbital_channel(mf, spin, mo_coeff)
    start, stop = frontier_window(int(orbital), nocc, coeff.shape[1], int(window))
    local = localized_window(mf.mol, coeff[:, start:stop], pop_method)
    weights = (local.T @ mf.get_ovlp() @ coeff[:, int(orbital)]) ** 2
    return dict(mol=mf.mol.copy(), localized=local, weights=weights,
                orbital=int(orbital), occupied=int(orbital) < nocc,
                window=(start, stop), spin=spin)


def track_orbital(fingerprint, mf, spin=None, mo_coeff=None, candidates=None,
                  out=None):
    """(the orbital at `mf` that continues the fingerprinted one, similarities).

    Each candidate is projected on the REFERENCE geometry's localized orbitals
    through `body_overlap`, the one orbital identity (module docstring), and
    scored by the Bhattacharyya coefficient sum_k sqrt(v_k w_k) of its weights
    v against the reference orbital's w: one for the same localized character
    entirely inside the reference window, less as the character moves or
    leaks out. Signs and rotations within one localized orbital's share do
    not enter.

    candidates: the orbitals to choose among, by default the reference window
    on the same side of the gap. Read with the bars a followed state is read
    with: a best score below ROOT_FOLLOW_WEIGHT_MIN is an orbital not among
    the candidates, and two best closer than ROOT_FOLLOW_MARGIN_MIN are a
    guess -- a displaced geometry that picks the other one puts a step in the
    surface. Both are warned about and kept (refuse
    inputs, warn numerics); `out`, a dict, receives 'lost' (the candidate and
    its score) and 'ambiguous' (the two candidates and their margin) for the
    caller's record.

    Returns (orbital index, {candidate: similarity}).
    """
    coeff, nocc = orbital_channel(mf, spin, mo_coeff)
    if candidates is None:
        start, stop = fingerprint['window']
        if fingerprint['occupied']:
            candidates = range(max(0, nocc - (stop - start)), nocc)
        else:
            candidates = range(nocc, min(coeff.shape[1],
                                         nocc + (stop - start)))
    # the one orbital identity: the reference's localized orbitals carried
    # by their atoms and turned with the molecule (`body_overlap`)
    project = body_overlap(fingerprint['mol'], fingerprint['localized'],
                           mf.mol, np.eye(mf.mol.nao))
    w = fingerprint['weights']
    score = {int(c): float(np.sqrt((project @ coeff[:, int(c)]) ** 2 * w).sum())
             for c in candidates}
    ranked = sorted(score, key=score.get, reverse=True)
    best = score[ranked[0]]
    if best < ROOT_FOLLOW_WEIGHT_MIN:
        if out is not None:
            out['lost'] = {'candidate': ranked[0], 'similarity': best}
        warnings.warn(
            f'orbital tracking lost the orbital: the best candidate, '
            f'{ranked[0]}, matches the reference character to {best:.3f}, '
            f'below {ROOT_FOLLOW_WEIGHT_MIN}', RuntimeWarning, stacklevel=2)
    if len(ranked) > 1 and best - score[ranked[1]] < ROOT_FOLLOW_MARGIN_MIN:
        if out is not None:
            out['ambiguous'] = {
                'candidates': [ranked[0], ranked[1]],
                'margin': best - score[ranked[1]]}
        warnings.warn(
            f'orbital tracking is ambiguous: candidates {ranked[0]} and '
            f'{ranked[1]} match the reference character to '
            f'{best:.3f} and {score[ranked[1]]:.3f}, closer than '
            f'{ROOT_FOLLOW_MARGIN_MIN}', RuntimeWarning, stacklevel=2)
    return ranked[0], score

def transported_overlap(key, here, nocc, x_ref, y_ref, x, y):
    """`relative_overlap` of the states (x_ref, y_ref) at `key` with (x, y) at
    `here`, each (mol, mo_coeff), the orbitals matched by `body_overlap`: THE
    rule for "the same state at another geometry". Y may be None
    (Tamm-Dancoff)."""
    y_ref = np.zeros_like(x_ref) if y_ref is None else y_ref
    y = np.zeros_like(x) if y is None else y
    t = body_overlap(key[0], key[1], here[0], here[1])
    return relative_overlap(state_overlap(t, nocc, x_ref, y_ref, x, y))


#: What "the same orbital at two geometries" means wherever it is asked --
#: the conventions' keys and the followed state alike -- as a record names it.
ORBITAL_IDENTITY = 'body_overlap'


def body_overlap(key_mol, key_mo, mol, mo):
    """<key_p|q> in Loewdin-orthogonalized atomic orbitals S^(1/2) C, which
    move with their atoms and, the key orbitals first turned with the molecule
    (Kabsch superposition, `body_frame`), turn with it: an orbital that
    follows its atoms keeps its identity however far they move. The spatial
    overlap does not: over a 0.2 Bohr step a carbon 1s keeps 0.6 of itself
    there."""
    q = kabsch_rotation(key_mol.atom_coords(), mol.atom_coords())
    turned = key_mol.copy()
    turned.set_geom_(centred(key_mol.atom_coords()) @ q
                     + np.asarray(mol.atom_coords()).mean(axis=0), unit='Bohr')
    turned.build(False, False)
    key = loewdin_root(turned) @ (ao_rotation_matrix(key_mol, q) @ key_mo)
    return key.T @ (loewdin_root(mol) @ np.asarray(mo, float))


def loewdin_root(mol):
    """S^(1/2) of the atomic-orbital overlap."""
    w, v = np.linalg.eigh(mol.intor_symmetric('int1e_ovlp'))
    return (v * np.sqrt(w)) @ v.T
