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
from pyscf import lo, scf as pyscf_scf, symm

from src.Base.numerics.constants import (CHARACTER_WINDOW, LOCALIZED_ASSIGNMENT_FLOOR,
                                         PURITY_FLOOR, QP_ORDER_SEARCH)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.gradients.state_identity import (frontier_window, localized_window,
                                          orbital_channel)
from src.gradients.surface import require


def roots(chain, mol=None, mf=None):
    """(Omega, X, Y, nocc) -- every BSE root at one geometry, not just `state`.

    A chain's `excitation` returns one root. This is the same forward pass with
    the whole spectrum kept, and it works unchanged on the gas-phase and the
    solvated chain.
    """
    return require(chain, 'excited_states', 'characters.roots')(mol, mf)


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


def _named_partition(mol, fragments):
    """(names, atom lists) of `fragments`, checked to partition the atoms."""
    names, atoms = list(fragments), [list(v) for v in fragments.values()]
    flat = sorted(a for group in atoms for a in group)
    if flat != list(range(mol.natm)):
        raise ValueError(
            f'fragments must partition the {mol.natm} atoms, each exactly '
            f'once, for the weights to sum to one; got {dict(fragments)}')
    return names, atoms


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
    coeff, nocc = orbital_channel(mf, spin, mo_coeff)
    start, stop = frontier_window(int(orbital), nocc, coeff.shape[1], int(window))
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


def quasiparticle_order(mf, orbital, window=QP_ORDER_SEARCH,
                        spin_channel='alpha', **route):
    """Whether `orbital` is the lowest-energy removal or attachment, in eV.

    Solves the quasiparticle energies of the `window` orbitals nearest the
    frontier on the orbital's side of the gap (and the orbital itself) through
    `calc_qp_energy(**route)` -- by default the space-time route, whose ISDF
    grid `route` must name (`counts=` or `grid_accuracy=`) -- and warns
    when another orbital is the easier electron to remove (a higher occupied
    quasiparticle energy) or to add (a lower virtual one). G0W0 reorders
    states relative to the mean field, so a surface that follows a canonical
    index can be following the second-lowest process.

    Returns {'orbital', 'energies' {orbital: eV}, 'lowest_process',
    'reordered', 'reference_diagnostics'} -- the last the unrestricted
    reference's spin contamination and stability flags (`calc_qp_energy(out=)`),
    None for a restricted one.
    """
    unrestricted = isinstance(mf, pyscf_scf.uhf.UHF)
    spin = (0 if spin_channel == 'alpha' else 1) if unrestricted else None
    coeff, nocc = orbital_channel(mf, spin, None)
    start, stop = frontier_window(int(orbital), nocc, coeff.shape[1], int(window))
    states = sorted(set(range(start, stop)) | {int(orbital)})
    route = dict({'mode': 'space-time'}, **route)
    if unrestricted:
        route['spin_channel'] = spin_channel
    seen = {}
    out = calc_qp_energy(mf, selfenergy='GW', df=True, state=states,
                         out=seen, **route)
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
                lowest_process=lowest, reordered=lowest != int(orbital),
                reference_diagnostics=seen.get('reference_diagnostics'))


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
