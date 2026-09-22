"""Labelling BSE roots: the whole spectrum at one geometry, how much of each
root is charge transfer, and which point-group irrep it belongs to.

None of the three is a surface property -- they need the eigenvectors, not
just one energy -- so `roots` and `ct_character` take a BSE chain rather than
a `PotentialEnergySurface`. They are what turns a list of numbers into
assigned states: a solvatochromic shift is only interpretable per state, and
telling a charge-transfer root from a local one is what makes the shift a
prediction rather than a coincidence.

The irrep label is the direct product Gamma_i (x) Gamma_a over the transition
amplitude. Every abelian point group -- Cs, C2v, C2h, D2h, ... -- turns that
product into an XOR of pyscf's irrep ids, and the result need not be an
orbital irrep at all: a totally symmetric-forbidden product such as
A2 = B2 (x) B1 can label a root even when the basis carries no A2 orbital.
Selecting a root by this label rather than by energy order or oscillator
strength matters because a solver reports the lowest root OF A GIVEN
SYMMETRY, not the lowest root outright, and a symmetry-forbidden transition
has zero oscillator strength regardless of where its energy ranks.
"""
import numpy as np
from pyscf import symm

from src.Base.constants import PURITY_FLOOR


def roots(chain, mol=None, mf=None):
    """(Omega, X, Y, nocc) -- every BSE root at one geometry, not just `state`.

    A chain's `excitation` returns one root. This is the same forward pass with
    the whole spectrum kept, and it works unchanged on the gas-phase and the
    solvated chain.
    """
    mol, mf = chain.mean_field(mol, mf)
    om, pieces = chain._forward(mol, mf)
    xn, yn = pieces[10], pieces[11]
    return om, xn, yn, chain.nocc


def ct_character(chain, mf, xn, yn, fragment_atoms):
    """Per-root charge-transfer weight: hole on the fragment, electron off it.

    Loewdin-populates every MO on `fragment_atoms` (p_A(p) in [0, 1]) and
    contracts the Casida amplitudes with the product of the hole's and the
    electron's fragment weights:

        CT_n = sum_ia (|X_ia|^2 + |Y_ia|^2) [ p_A(i) (1 - p_A(a))
                                            + (1 - p_A(i)) p_A(a) ] / norm

    so 0 is a purely intrafragment excitation and 1 a complete interfragment
    transfer. Crude by design -- it needs no relaxed density and no dipole --
    and it is only used to LABEL roots, never to compute one.
    """
    mol = mf.mol
    s = mf.get_ovlp()
    w, u = np.linalg.eigh(s)
    s_half = (u * np.sqrt(w)) @ u.T
    c_lowdin = s_half @ mf.mo_coeff                       # (nao, nmo)
    ao_of_atom = np.zeros(mol.nao, bool)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        if ia in fragment_atoms:
            ao_of_atom[p0:p1] = True
    p_a = (c_lowdin[ao_of_atom] ** 2).sum(axis=0)         # (nmo,)

    nocc = chain.nocc
    nvirt = len(p_a) - nocc
    ph = p_a[:nocc][:, None]
    pe = p_a[nocc:][None, :]
    weight = (ph * (1.0 - pe) + (1.0 - ph) * pe).reshape(-1)
    amp = (np.asarray(xn) ** 2 + np.asarray(yn) ** 2)     # (n_ov, nroots)
    assert amp.shape[0] == nocc * nvirt
    return (weight @ amp) / amp.sum(axis=0)


def orbital_irreps(mol, mo_coeff):
    """(irrep ids, irrep names) per molecular orbital."""
    ids = np.asarray(symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                         mo_coeff))
    names = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                     mo_coeff))
    return ids, names


def root_irreps(mol, mo_coeff, nocc, X, Y):
    """Per root: (irrep name, purity, dominant (i, a), its weight).

    Purity is the share of sum (X+Y)^2 carried by the winning product channel.
    X + Y rather than X alone: it is the transition density's combination, and
    it is what the oscillator strength and the symmetry both read.
    """
    ids, names = orbital_irreps(mol, mo_coeff)
    nvirt = len(ids) - nocc
    # abelian groups only, where the direct product is an XOR of the ids
    channel = ids[:nocc, None] ^ ids[None, nocc:]
    labels = np.unique(channel)
    X, Y = np.asarray(X), np.asarray(Y)
    out = []
    for n in range(X.shape[1]):
        amp = (X[:, n] + Y[:, n]).reshape(nocc, nvirt)
        w = amp ** 2
        weights = np.array([w[channel == k].sum() for k in labels])
        total = weights.sum()
        k = labels[weights.argmax()]
        i, a = np.unravel_index(np.abs(amp).argmax(), amp.shape)
        out.append(dict(
            irrep=symm.irrep_id2name(mol.groupname, int(k)),
            purity=float(weights.max() / total) if total > 0 else 0.0,
            occ=int(i), virt=int(nocc + a),
            occ_irrep=names[i], virt_irrep=names[nocc + a],
            weight=float(abs(amp[i, a]))))
    return out


def select(mol, mo_coeff, nocc, omega, X, Y, target_irrep):
    """(index of the LOWEST root with `target_irrep`, the per-root labels).

    Lowest-of-symmetry, because that is the convention QUEST reports. Returns
    index None when the symmetry does not appear among the roots computed,
    which means too few were asked for -- a caller that silently fell back to
    root 0 would report a different state's energy under the right label.
    """
    labels = root_irreps(mol, mo_coeff, nocc, X, Y)
    for n in np.argsort(np.asarray(omega)):
        if labels[int(n)]['irrep'] == target_irrep:
            return int(n), labels
    return None, labels


def impure(labels):
    """Roots whose symmetry label is not trustworthy, as (index, purity)."""
    return [(n, r['purity']) for n, r in enumerate(labels)
            if r['purity'] < PURITY_FLOOR]
