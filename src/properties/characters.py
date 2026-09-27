"""Labelling states: the BSE roots at one geometry, how much of each is charge
transfer and which point-group irrep it belongs to, and where the orbital a
quasiparticle removes or adds an electron lives.

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

QUASIPARTICLE ORBITALS are labelled by fragment ("Li 2s-like" against
"carbonate pi*" for the electron an attachment adds). A canonical orbital near
a near-degeneracy is an arbitrary rotation within it, and its Loewdin
population then reports a mixture that belongs to neither state, so the label
is read through Pipek-Mezey localized orbitals (Pipek and Mezey, J. Chem.
Phys. 90, 4916 (1989)) of a window around the frontier: each localized orbital
belongs to one fragment by its population, and the quasiparticle orbital's
weight on a fragment is its weight on that fragment's localized orbitals. The
Loewdin population of the canonical orbital stays beside it as the cross-check.
The same localized set is the reference a geometry scan follows the orbital
by (`orbital_fingerprint`, `track_orbital`), and `quasiparticle_order` says
whether the orbital is the lowest-energy process at all.
"""
import warnings

import numpy as np
from pyscf import gto, lo, scf as pyscf_scf, symm

from src.Base.constants import (CHARACTER_WINDOW, LOCALIZED_ASSIGNMENT_FLOOR,
                                PURITY_FLOOR, QP_ORDER_SEARCH, TRACKING_MARGIN)
from src.SingleReference.GW.qp_energy import calc_qp_energy


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
    p_a = fragment_populations(mf.mol, mf.mo_coeff, [fragment_atoms],
                               method='lowdin', ovlp=mf.get_ovlp())[:, 0]

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


def fragment_populations(mol, orbitals, fragments, method='meta_lowdin',
                         ovlp=None):
    """(norb, nfrag) population of each column of `orbitals` on each fragment.

    Counted in an orthogonal AO basis, so a normalized orbital's populations
    over a partition of the atoms sum to one: 'lowdin' is S^1/2 C, and
    'meta_lowdin' is pyscf's `lo.orth_ao(mol, 'meta_lowdin')`, the basis
    pyscf's Pipek-Mezey functional counts its own atomic populations in.

    fragments: a sequence of atom-index lists.
    """
    s = mol.intor_symmetric('int1e_ovlp') if ovlp is None else ovlp
    orbitals = np.asarray(orbitals, float)
    if method == 'lowdin':
        w, u = np.linalg.eigh(s)
        coeff = (u * np.sqrt(w)) @ u.T @ orbitals
    elif method == 'meta_lowdin':
        coeff = lo.orth_ao(mol, 'meta_lowdin', s=s).T @ s @ orbitals
    else:
        raise ValueError(f"method must be 'lowdin' or 'meta_lowdin', got "
                         f"{method!r}")
    aoslices = mol.aoslice_by_atom()
    out = np.zeros((orbitals.shape[1], len(fragments)))
    for f, atoms in enumerate(fragments):
        idx = np.concatenate([np.arange(aoslices[a][2], aoslices[a][3])
                              for a in atoms])
        out[:, f] = (coeff[idx] ** 2).sum(axis=0)
    return out


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


def _named_partition(mol, fragments):
    """(names, atom lists) of `fragments`, checked to partition the atoms."""
    names, atoms = list(fragments), [list(v) for v in fragments.values()]
    flat = sorted(a for group in atoms for a in group)
    if flat != list(range(mol.natm)):
        raise ValueError(
            f'fragments must partition the {mol.natm} atoms, each exactly '
            f'once, for the weights to sum to one; got {dict(fragments)}')
    return names, atoms


def _channel(mf, spin, mo_coeff):
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


def _window(orbital, nocc, nmo, window):
    """[start, stop) of the canonical orbitals on the orbital's side of the gap,
    from the frontier outwards, widened to hold the orbital itself."""
    if orbital < nocc:
        return max(0, min(nocc - window, orbital)), nocc
    return nocc, min(nmo, max(nocc + window, orbital + 1))


def quasiparticle_character(mf, orbital, fragments, window=CHARACTER_WINDOW,
                            spin=None, mo_coeff=None, pop_method='meta_lowdin'):
    """Where the orbital a quasiparticle removes or adds an electron lives.

    Localizes `window` canonical orbitals on the orbital's own side of the gap
    (Pipek-Mezey, `localized_window`), assigns each localized orbital to the
    fragment carrying the largest share of its population, and returns the
    orbital's weight on each fragment's localized orbitals. The localized set
    spans the window, so the weights sum to one exactly. Occupied and virtual
    orbitals are never localized together: that would mix the hole's and the
    particle's character into one label.

    fragments: {name: atom indices}, a partition of the atoms.
    spin: the channel of an unrestricted reference.
    mo_coeff: the orbitals, in place of the mean field's -- a qsGW-rotated set.

    Returns a dict: `weights` {fragment: weight} from the localized orbitals,
    `lowdin` {fragment: Loewdin population of the canonical orbital} as the
    cross-check, `dominant` the fragment of largest weight, `localized` one
    record per localized orbital (its fragment, its population there, the
    orbital's weight on it), and `unassigned` the localized orbitals whose
    largest fragment population is below LOCALIZED_ASSIGNMENT_FLOOR -- a label
    that leans on one of those is not clean.
    """
    names, atoms = _named_partition(mf.mol, fragments)
    coeff, nocc = _channel(mf, spin, mo_coeff)
    start, stop = _window(int(orbital), nocc, coeff.shape[1], int(window))
    s = mf.get_ovlp()
    local = localized_window(mf.mol, coeff[:, start:stop], pop_method)
    pops = fragment_populations(mf.mol, local, atoms, 'meta_lowdin', s)
    owner = pops.argmax(axis=1)
    amp = (local.T @ s @ coeff[:, int(orbital)]) ** 2
    weights = {n: float(amp[owner == f].sum()) for f, n in enumerate(names)}
    lowdin = fragment_populations(mf.mol, coeff[:, [int(orbital)]], atoms,
                                  'lowdin', s)[0]
    localized = [dict(fragment=names[int(owner[k])],
                      population=float(pops[k, owner[k]]),
                      weight=float(amp[k])) for k in range(len(amp))]
    return dict(orbital=int(orbital), spin=spin, window=(start, stop),
                weights=weights,
                lowdin={n: float(lowdin[f]) for f, n in enumerate(names)},
                dominant=max(weights, key=weights.get), localized=localized,
                unassigned=[k for k, r in enumerate(localized)
                            if r['population'] < LOCALIZED_ASSIGNMENT_FLOOR])


def orbital_fingerprint(mf, orbital, window=CHARACTER_WINDOW, spin=None,
                        mo_coeff=None, pop_method='meta_lowdin'):
    """The orbital at this geometry, as `track_orbital` follows it to another.

    Its weight on each Pipek-Mezey localized orbital of the window, together
    with the localized orbitals and the molecule they live on. A canonical
    orbital near a near-degeneracy is an arbitrary rotation within it; the
    localized set is not, which is what makes it the reference to follow.
    """
    coeff, nocc = _channel(mf, spin, mo_coeff)
    start, stop = _window(int(orbital), nocc, coeff.shape[1], int(window))
    local = localized_window(mf.mol, coeff[:, start:stop], pop_method)
    weights = (local.T @ mf.get_ovlp() @ coeff[:, int(orbital)]) ** 2
    return dict(mol=mf.mol.copy(), localized=local, weights=weights,
                orbital=int(orbital), occupied=int(orbital) < nocc,
                window=(start, stop), spin=spin)


def track_orbital(fingerprint, mf, spin=None, mo_coeff=None, candidates=None):
    """(the orbital at `mf` that continues the fingerprinted one, similarities).

    Each candidate is projected on the REFERENCE geometry's localized orbitals
    through the cross-geometry overlap <chi_mu(R0)|chi_nu(R)>, and scored by
    the Bhattacharyya coefficient sum_k sqrt(v_k w_k) of its weights v against
    the reference orbital's w: one for the same localized character entirely
    inside the reference window, less as the character moves or leaks out.
    Signs and rotations within one localized orbital's share do not enter.

    candidates: the orbitals to choose among, by default the reference window
    on the same side of the gap. Warns when the best two are closer than
    TRACKING_MARGIN: the track is then a guess, and a displaced geometry that
    picks the other one puts a step in the surface.

    Returns (orbital index, {candidate: similarity}).
    """
    coeff, nocc = _channel(mf, spin, mo_coeff)
    if candidates is None:
        start, stop = fingerprint['window']
        if fingerprint['occupied']:
            candidates = range(max(0, nocc - (stop - start)), nocc)
        else:
            candidates = range(nocc, min(coeff.shape[1],
                                         nocc + (stop - start)))
    s01 = gto.intor_cross('int1e_ovlp', fingerprint['mol'], mf.mol)
    project = fingerprint['localized'].T @ s01
    w = fingerprint['weights']
    score = {int(c): float(np.sqrt((project @ coeff[:, int(c)]) ** 2 * w).sum())
             for c in candidates}
    ranked = sorted(score, key=score.get, reverse=True)
    if len(ranked) > 1 and score[ranked[0]] - score[ranked[1]] < TRACKING_MARGIN:
        warnings.warn(
            f'orbital tracking is ambiguous: candidates {ranked[0]} and '
            f'{ranked[1]} match the reference character to '
            f'{score[ranked[0]]:.3f} and {score[ranked[1]]:.3f}, closer than '
            f'{TRACKING_MARGIN}', RuntimeWarning, stacklevel=2)
    return ranked[0], score


def quasiparticle_order(mf, orbital, window=QP_ORDER_SEARCH,
                        spin_channel='alpha', **route):
    """Whether `orbital` is the lowest-energy removal or attachment, in eV.

    Solves the quasiparticle energies of the `window` orbitals nearest the
    frontier on the orbital's side of the gap (and the orbital itself) through
    `calc_qp_energy(**route)` -- by default the space-time route -- and warns
    when another orbital is the easier electron to remove (a higher occupied
    quasiparticle energy) or to add (a lower virtual one). G0W0 reorders
    states relative to the mean field, so a surface that follows a canonical
    index can be following the second-lowest process.

    Returns {'orbital', 'energies' {orbital: eV}, 'lowest_process',
    'reordered'}.
    """
    unrestricted = isinstance(mf, pyscf_scf.uhf.UHF)
    spin = (0 if spin_channel == 'alpha' else 1) if unrestricted else None
    coeff, nocc = _channel(mf, spin, None)
    start, stop = _window(int(orbital), nocc, coeff.shape[1], int(window))
    states = sorted(set(range(start, stop)) | {int(orbital)})
    route = dict({'mode': 'space-time'}, **route)
    if unrestricted:
        route['spin_channel'] = spin_channel
    out = calc_qp_energy(mf, selfenergy='GW', df=True, state=states, **route)
    if isinstance(out, dict):
        out = [out[p]['GW'] for p in states]
    energies = dict(zip(states, (float(e) for e in out)))
    pick = max if int(orbital) < nocc else min
    lowest = pick(energies, key=energies.get)
    if lowest != int(orbital):
        kind = 'removal' if int(orbital) < nocc else 'attachment'
        warnings.warn(
            f'orbital {orbital} is not the lowest-energy {kind}: orbital '
            f'{lowest} is, at {energies[lowest]:.3f} eV against '
            f'{energies[int(orbital)]:.3f} eV. A surface following orbital '
            f'{orbital} follows the higher-energy process.', RuntimeWarning,
            stacklevel=2)
    return dict(orbital=int(orbital), energies=energies,
                lowest_process=lowest, reordered=lowest != int(orbital))


def surface_order(surface, mol=None, window=QP_ORDER_SEARCH, **route):
    """`quasiparticle_order` for a charged-excitation surface at `mol`.

    The surface's own mean field -- through its environment, so a solvated
    surface is audited in its continuum -- and the orbital its declared
    `ChargedExcitation` names. Run it at the reference geometry before an
    optimization or a scan follows that orbital.
    """
    _, mf = surface.mean_field(mol)
    return quasiparticle_order(mf, surface.physics_excitation.orbital, window,
                               **route)
