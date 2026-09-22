"""Conformer search over the soft torsions of a twisted emitter.

A donor-acceptor TADF or ISC molecule has one or a few torsions whose barrier is
of the order of k_B T, so the structure an optimizer returns is whichever
minimum its input geometry happened to fall into. Everything the property layer
defines about that structure -- normal modes, Huang-Rhys factors, an adiabatic
Delta-E_ST, a Marcus lambda -- is then a property of that one minimum and not of
the molecule, and the harmonic reference is weakest exactly where the torsion is
softest. This module enumerates the torsional minima, relaxes each one, merges
the duplicates and reports Boltzmann populations, so a property can be averaged
over the ensemble instead of evaluated at an accident.

Connectivity is the only chemistry used: covalent radii for the graph, a
distance ratio standing in for the bond order, ring membership by cutting the
edge. A wrong guess costs extra starts that deduplication removes, never a wrong
energy.

SURFACE-AGNOSTIC BY CONSTRUCTION. Nothing here calls an SCF, a Casida solver or
a gradient. Every relaxation goes through `relax` and every energy through
`PotentialEnergySurface.total_energy`, so the same search runs on
`MeanFieldSurface` and on a BSE@GW surface unchanged. On an excited surface the
minima are the EXCITED state's, which is the twisted-ICT case: the emitting
conformer need not be the one the ground state relaxes to.

ONE FROZEN SURFACE FOR THE WHOLE SEARCH. Every start is relaxed on the surface
as it was handed in, frozen conventions and all, because energies from two
differently frozen surfaces are not comparable and the whole point here is to
order minima by energy. `surface.refreeze` at a converged conformer is the
caller's step afterwards, and its energy shift is the error bar on that
ordering.
"""
import importlib.util
from dataclasses import dataclass, field

import numpy as np
from pyscf.data import radii

from src.Base.constants import (BOHR_TO_ANGSTROM, BOLTZMANN_HARTREE_PER_KELVIN,
                                CONFORMER_BOND_SCALE, CONFORMER_ENERGY_TOL,
                                CONFORMER_MAX_AUTOMORPHISMS,
                                CONFORMER_MAX_STARTS, CONFORMER_RMSD_TOL,
                                CONFORMER_SCREEN_CONV,
                                CONFORMER_SINGLE_BOND_RATIO,
                                CONFORMER_TEMPERATURE, CONFORMER_TORSION_GRID,
                                HARTREE_TO_EV)
from src.properties.optimize import relax

# Weisfeiler-Lehman refinement rounds behind the symmetry-aware RMSD. Three
# rounds separate atoms that differ within three bonds, which is what tells a
# CH2 from a CH3 and one ring substitution pattern from another; more rounds
# only ever split classes further, so a missed automorphism costs a duplicate
# conformer and never a wrong merge.
COLOUR_REFINEMENT_ROUNDS = 3


@dataclass
class Torsion:
    """A rotatable bond j-k and the dihedral i-j-k-l that measures it.

    `moving` is the smaller of the two fragments the j-k bond separates, which
    is the one rigidly rotated; `sense` is +1 when that fragment carries atom i
    and -1 when it carries atom l, so that rotating it by `sense * delta` about
    j->k advances the dihedral by `delta`.
    """

    i: int
    j: int
    k: int
    l: int
    moving: tuple
    sense: int


@dataclass
class Conformer:
    """One relaxed minimum, its population, and where it came from.

    `energy` is the surface's TOTAL energy in Hartree -- an excitation energy
    alone would order the minima by the wrong quantity, since the ground state
    does not cancel between two different geometries. `torsions` are the
    dihedrals of the relaxed structure in degrees, one per rotatable bond.
    `start` is the grid offset tuple that produced it and `starts` every start
    index that relaxed into it.
    """

    mol: object
    energy: float
    weight: float
    torsions: np.ndarray
    start: tuple
    starts: list = field(default_factory=list)
    converged: bool = True
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------

def molecular_graph(mol, scale=CONFORMER_BOND_SCALE):
    """Covalent adjacency as a list of neighbour sets.

    A bond is drawn when the distance is within `scale` of the sum of the two
    covalent radii, taken from pyscf's table (`pyscf.data.radii.COVALENT`, in
    Bohr, indexed by nuclear charge).
    """
    z = np.asarray(mol.atom_charges())
    if z.max() >= len(radii.COVALENT):
        raise ValueError(f'no covalent radius tabulated for Z = {z.max()}')
    r = np.array([radii.COVALENT[int(zi)] for zi in z])
    crd = np.asarray(mol.atom_coords())
    d = np.linalg.norm(crd[:, None, :] - crd[None, :, :], axis=-1)
    bonded = d < scale * (r[:, None] + r[None, :])
    np.fill_diagonal(bonded, False)
    return [set(np.flatnonzero(row).tolist()) for row in bonded]


def bond_in_ring(adj, a, b):
    """True when a and b are still connected once the a-b edge is cut.

    Ring bonds are the ones a rigid rotation cannot act on at all: the two
    fragments the bond would separate are the same fragment, and turning the
    dihedral would have to break another bond. Puckering a ring is a different
    motion and is not what this module searches.
    """
    seen, stack = {a}, [a]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if (cur, nxt) in ((a, b), (b, a)):
                continue
            if nxt == b:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def heavy_atoms(mol):
    """Indices of the atoms with Z > 1.

    Hydrogens are left out of every geometric comparison here: their positions
    add the rotation of each methyl and hydroxyl rotor to an RMSD that is meant
    to distinguish heavy-atom skeletons.
    """
    return np.flatnonzero(np.asarray(mol.atom_charges()) > 1)


def split_bond(adj, j, k):
    """(atoms on j's side, atoms on k's side) once the j-k bond is cut."""
    side_j, stack = {j}, [j]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if (cur, nxt) in ((j, k), (k, j)) or nxt in side_j:
                continue
            side_j.add(nxt)
            stack.append(nxt)
    if k in side_j:
        raise ValueError(f'the {j}-{k} bond is in a ring and does not split '
                         f'the molecule')
    side_k = set(range(len(adj))) - side_j
    return side_j, side_k


def rotatable_bonds(mol, adj=None, scale=CONFORMER_BOND_SCALE,
                    single_ratio=CONFORMER_SINGLE_BOND_RATIO):
    """The torsions a conformer search has to enumerate. Returns [Torsion].

    A bond qualifies when it is between two heavy atoms, is not in a ring, is
    long enough relative to the covalent radii to be single rather than double
    or conjugated-short, and carries at least one further heavy atom on each
    side. The last condition is what removes the methyl and hydroxyl rotors:
    turning them permutes hydrogens and produces no new heavy-atom skeleton, so
    they multiply the start count without adding a conformer.

    The reference atoms i and l are the lowest-indexed heavy neighbours of j and
    k, which makes the dihedral definition deterministic; any other choice
    differs by a constant offset and describes the same rotation.
    """
    adj = molecular_graph(mol, scale) if adj is None else adj
    z = np.asarray(mol.atom_charges())
    crd = np.asarray(mol.atom_coords())
    out = []
    for j in range(mol.natm):
        for k in sorted(adj[j]):
            if k <= j or z[j] == 1 or z[k] == 1:
                continue
            r_sum = radii.COVALENT[int(z[j])] + radii.COVALENT[int(z[k])]
            if np.linalg.norm(crd[j] - crd[k]) < single_ratio * r_sum:
                continue
            if bond_in_ring(adj, j, k):
                continue
            nb_j = sorted(a for a in adj[j] if a != k and z[a] > 1)
            nb_k = sorted(a for a in adj[k] if a != j and z[a] > 1)
            if not nb_j or not nb_k:
                continue
            side_j, side_k = split_bond(adj, j, k)
            moving, sense = ((tuple(sorted(side_j)), 1)
                             if len(side_j) <= len(side_k)
                             else (tuple(sorted(side_k)), -1))
            out.append(Torsion(nb_j[0], j, k, nb_k[0], moving, sense))
    return out


# ---------------------------------------------------------------------------
# torsion geometry
# ---------------------------------------------------------------------------

def dihedral(coords, i, j, k, l):
    """The i-j-k-l dihedral in degrees: anti is +-180, syn is 0.

    The signed angle between the projections of j->i and k->l on the plane
    normal to j->k.
    """
    crd = np.asarray(coords)
    b0 = crd[i] - crd[j]
    b1 = crd[k] - crd[j]
    b2 = crd[l] - crd[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(-np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def torsion_values(mol_or_coords, torsions):
    """The dihedrals of `torsions` in degrees, in their listed order."""
    crd = (np.asarray(mol_or_coords.atom_coords())
           if hasattr(mol_or_coords, 'atom_coords') else np.asarray(mol_or_coords))
    return np.array([dihedral(crd, t.i, t.j, t.k, t.l) for t in torsions])


def rotate_torsion(coords, tor, delta):
    """`coords` with tor's dihedral advanced by `delta` degrees.

    The smaller of the two fragments the j-k bond separates is carried rigidly
    about the bond axis, which leaves every bond length and angle untouched: the
    start is a pure torsional displacement and the relaxation that follows only
    has to fix what the torsion actually changed.
    """
    crd = np.asarray(coords, float).copy()
    u = crd[tor.k] - crd[tor.j]
    u = u / np.linalg.norm(u)
    ang = np.radians(delta) * tor.sense
    kx = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    rot = (np.eye(3) + np.sin(ang) * kx
           + (1.0 - np.cos(ang)) * (kx @ kx))          # Rodrigues
    mv = list(tor.moving)
    crd[mv] = (crd[mv] - crd[tor.j]) @ rot.T + crd[tor.j]
    return crd


def torsion_starts(mol, torsions, n_grid=CONFORMER_TORSION_GRID,
                   max_starts=CONFORMER_MAX_STARTS):
    """Coarse torsional grid. Returns [(coords in Bohr, offsets in degrees)].

    Each torsion takes `n_grid` values spaced 360/n_grid apart, as offsets from
    the value the INPUT geometry already has, so the input structure is always
    start 0 and no minimum already in hand can be lost. Three values per torsion
    is the anti/gauche+/gauche- pattern of an sp3-sp3 bond and puts a start
    inside every basin of a threefold barrier.

    THE COMBINATORICS IS n_grid ** n_torsions AND IT IS THE WHOLE COST: three
    torsions is 27 relaxations, five is 243, and each one is a full optimization
    on the surface -- hundreds of BSE@GW gradients apiece. `max_starts` truncates
    the product on a regular stride through it (always keeping the input), which
    samples the grid evenly rather than exhausting the first torsions and never
    reaching the last. A molecule with many soft torsions therefore gets a
    SAMPLE of its conformers, not the complete set, and the honest statement
    about such a search is that it found the minima it found.
    """
    total = n_grid ** len(torsions) if torsions else 1
    if total <= max_starts:
        indices = list(range(total))
    else:
        stride = np.linspace(0, total - 1, max_starts).round().astype(int)
        indices = sorted({0} | set(int(n) for n in stride))
    step = 360.0 / n_grid
    base = np.asarray(mol.atom_coords())
    out = []
    for idx in indices:
        rest = idx
        offsets = []
        for _ in torsions:
            offsets.append(step * (rest % n_grid))
            rest //= n_grid
        crd = base.copy()
        for tor, off in zip(torsions, offsets):
            if off:
                crd = rotate_torsion(crd, tor, off)
        out.append((crd, tuple(offsets)))
    return out


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------

def graph_automorphisms(mol, adj=None, max_perms=CONFORMER_MAX_AUTOMORPHISMS):
    """Heavy-atom permutations that leave the molecular graph unchanged.

    Returned as index arrays over the heavy-atom list: row p of a permuted
    coordinate block is the position the automorphism sends heavy atom p to.
    Atoms are first coloured by Weisfeiler-Lehman refinement of the FULL graph,
    hydrogens included, so a CH2 never maps onto a CH3; the backtracking then
    only tries same-colour images and checks adjacency against everything
    already assigned.

    This is what makes an RMSD comparison a comparison of STRUCTURES rather than
    of atom labellings: the two ends of a symmetric molecule are chemically the
    same, and a search that relabels them reports one minimum twice. The
    identity is always first, and the enumeration stops at `max_perms`, which
    can only leave duplicates in the output, never merge two real conformers.
    """
    adj = molecular_graph(mol) if adj is None else adj
    z = np.asarray(mol.atom_charges())
    colour = [int(x) for x in z]
    for _ in range(COLOUR_REFINEMENT_ROUNDS):
        sig = [(colour[a], tuple(sorted(colour[b] for b in adj[a])))
               for a in range(len(adj))]
        rank = {s: n for n, s in enumerate(sorted(set(sig)))}
        colour = [rank[s] for s in sig]

    heavy = heavy_atoms(mol)
    m = len(heavy)
    pos = {int(a): p for p, a in enumerate(heavy)}
    link = np.zeros((m, m), bool)
    for p, a in enumerate(heavy):
        for b in adj[int(a)]:
            if b in pos:
                link[p, pos[b]] = True
    col = [colour[int(a)] for a in heavy]
    order = sorted(range(m), key=lambda p: -int(link[p].sum()))

    perms, assign, used = [], [-1] * m, [False] * m

    def place(depth):
        if len(perms) >= max_perms:
            return
        if depth == m:
            perms.append(np.array(assign, int))
            return
        p = order[depth]
        for q in range(m):
            if used[q] or col[q] != col[p]:
                continue
            if any(link[p, order[d]] != link[q, assign[order[d]]]
                   for d in range(depth)):
                continue
            assign[p], used[q] = q, True
            place(depth + 1)
            assign[p], used[q] = -1, False

    place(0)
    identity = np.arange(m)
    perms.sort(key=lambda p: 0 if np.array_equal(p, identity) else 1)
    return perms if perms else [identity]


def kabsch_rmsd(a, b):
    """RMSD of b on a after the optimal PROPER superposition, in a's units.

    Proper because the reflection is enumerated separately in `conformer_rmsd`:
    letting the determinant go negative here would silently identify a molecule
    with its mirror image in every comparison, including the ones where that is
    a different molecule.
    """
    a0 = np.asarray(a, float) - np.asarray(a, float).mean(0)
    b0 = np.asarray(b, float) - np.asarray(b, float).mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    # A planar or collinear pair makes the determinant vanish, where np.sign
    # would return 0 and turn the rotation into a projection.
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((b0 @ rot - a0) ** 2).sum(1).mean()))


def conformer_rmsd(a, b, perms=None):
    """Smallest heavy-atom RMSD over superposition, reflection and symmetry.

    THE MIRROR IMAGE IS THE SAME CONFORMER for everything this module reports. A
    gauche+ and a gauche- minimum are enantiomers: identical energies, identical
    spectra, identical rates, and counting them separately would split one
    population into two. Their distinction is a chirality, which a torsional
    search on an achiral skeleton is not resolving.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0:
        return 0.0        # nothing heavy to tell two structures apart with
    mirror = np.array([-1.0, 1.0, 1.0])
    best = np.inf
    for perm in ([np.arange(len(a))] if perms is None else perms):
        bp = b[perm]
        best = min(best, kabsch_rmsd(a, bp), kabsch_rmsd(a, bp * mirror))
    return float(best)


def deduplicate(records, mol, perms=None, energy_tol=CONFORMER_ENERGY_TOL,
                rmsd_tol=CONFORMER_RMSD_TOL):
    """Group records that are the same minimum. Returns [(record, [indices])].

    `records` are dicts carrying at least 'mol' and 'energy'; the groups come
    back ordered by energy and each is represented by its lowest member.

    BOTH tests have to pass. Energy alone merges two genuinely different
    conformers that happen to be degenerate -- which symmetry makes common, not
    rare -- and RMSD alone merges structures that a loose relaxation left in the
    same region but which sit in different basins. Requiring both means a merge
    claims the two starts reached one minimum by two routes, which is what a
    duplicate is.
    """
    heavy = heavy_atoms(mol)
    perms = graph_automorphisms(mol) if perms is None else perms
    order = sorted(range(len(records)), key=lambda n: records[n]['energy'])
    reps, geoms, groups = [], [], []
    for n in order:
        rec = records[n]
        x = np.asarray(rec['mol'].atom_coords())[heavy] * BOHR_TO_ANGSTROM
        for g, rep in enumerate(reps):
            if (abs(rep['energy'] - rec['energy']) < energy_tol
                    and conformer_rmsd(geoms[g], x, perms) < rmsd_tol):
                groups[g].append(n)
                break
        else:
            reps.append(rec)
            geoms.append(x)
            groups.append([n])
    return list(zip(reps, groups))


# ---------------------------------------------------------------------------
# populations and the search
# ---------------------------------------------------------------------------

def boltzmann_weights(energies, temperature=CONFORMER_TEMPERATURE):
    """Populations exp(-E/kT) normalized to one, from TOTAL energies in Hartree.

    Electronic energies only. The conformational entropy -- the vibrational
    partition function of each minimum, and the multiplicity of the symmetry
    images merged into it -- is not in here, and for a pair of minima separated
    by a few k_B T those are corrections of the same order as the gap itself.
    """
    e = np.asarray(energies, float)
    w = np.exp(-(e - e.min()) / (BOLTZMANN_HARTREE_PER_KELVIN * temperature))
    return w / w.sum()


def _relax_start(surface, mol, engine, loose, relax_kw):
    """One relaxation through the existing optimizers, loose or converged.

    The two engines spell their convergence differently -- a threshold dict for
    the Cartesian optimizer, a named set for geomeTRIC -- so the engine is
    resolved here rather than left to `relax`'s own probe.
    """
    if engine == 'auto':
        engine = ('geometric' if importlib.util.find_spec('geometric')
                  else 'cartesian')
    kw = dict(relax_kw)
    kw.setdefault('verbose', False)
    if engine == 'geometric':
        kw['converge'] = 'GAU_LOOSE' if loose else 'GAU'
    elif loose:
        kw['conv'] = CONFORMER_SCREEN_CONV
    return relax(surface, mol, engine=engine, **kw)


def _mol_at(mol, coords):
    """`mol` rebuilt at `coords` (Bohr)."""
    out = mol.copy()
    out.set_geom_(np.asarray(coords), unit='Bohr')
    out.build(False, False)
    return out


def search_conformers(surface, mol=None, n_grid=CONFORMER_TORSION_GRID,
                      max_starts=CONFORMER_MAX_STARTS,
                      temperature=CONFORMER_TEMPERATURE, engine='auto',
                      refine=True, torsions=None,
                      energy_tol=CONFORMER_ENERGY_TOL,
                      rmsd_tol=CONFORMER_RMSD_TOL, verbose=True, map_fn=None,
                      **relax_kw):
    """Torsional minima of `surface`'s state. Returns [Conformer], lowest first.

    Enumerate a coarse grid over the rotatable bonds, relax every start on the
    surface at loose convergence, merge the duplicates, relax the survivors to
    full convergence, merge again and report Boltzmann populations.

    THE TWO PASSES ARE WHERE THE COST GOES. The screening pass runs once per
    start and only has to identify a basin, so it stops at
    CONFORMER_SCREEN_CONV; the converged pass runs once per SURVIVING conformer,
    which for a torsional grid is a small fraction of the starts, and the saving
    grows with the start count. Measured on 1,3-butadiene/HF/STO-3G with the
    Cartesian optimizer: 78 cycles for the two passes against 116 for converging
    every start, with the screening pass 5 cycles per start against 52 in the
    flat s-cis basin.

    NEITHER PASS IS OPTIONAL. The screening energies in that basin sit 3.7 mHa
    (100 meV) above the converged ones, so screening ranks basins and does not
    resolve gaps; `refine=False` reports minima whose ENERGY ORDERING is only
    good to that much, and their populations with it.

    A start that fails to evaluate -- a torsion rotated into a clash, a state
    that stops existing there -- is dropped with its status recorded rather than
    raised: the conformers that did relax are still the useful output.
    """
    mol = surface.mol0 if mol is None else mol
    adj = molecular_graph(mol)
    torsions = rotatable_bonds(mol, adj) if torsions is None else torsions
    perms = graph_automorphisms(mol, adj)
    starts = torsion_starts(mol, torsions, n_grid, max_starts)
    if verbose:
        print(f'  [conformers] {len(torsions)} rotatable bond(s), '
              f'{len(starts)} start(s) on {surface.label()}')

    # Every start relaxes on its own; `map_fn` (map's signature, default the
    # builtin) is where a process pool spreads them. The mapped function hands
    # back COORDINATES, not a Mole, so its result crosses a pickle boundary;
    # the record's Mole is rebuilt here from them.
    runner = map if map_fn is None else map_fn

    def relax_loose(item):
        n, (crd, offsets) = item
        try:
            mol_opt, info = _relax_start(surface, _mol_at(mol, crd), engine,
                                         True, relax_kw)
        except Exception as exc:                      # a start that dissociates
            return n, offsets, None, None, f'{type(exc).__name__}: {exc}'
        if info.get('energy') is None:
            return n, offsets, None, None, info.get('status', 'no energy')
        return n, offsets, np.asarray(mol_opt.atom_coords()), info, None

    records, failed = [], []
    for n, offsets, crd_opt, info, why in runner(relax_loose,
                                                 list(enumerate(starts))):
        if crd_opt is None:
            failed.append((offsets, why))
            continue
        records.append({'mol': _mol_at(mol, crd_opt), 'energy': float(info['energy']),
                        'offsets': offsets, 'index': n,
                        'converged': bool(info.get('converged')), 'info': info})
    if not records:
        raise RuntimeError(f'every conformer start failed to relax: {failed}')

    groups = deduplicate(records, mol, perms, energy_tol, rmsd_tol)
    if verbose:
        print(f'  [conformers] {len(records)} relaxed, {len(groups)} distinct '
              f'after screening ({len(failed)} failed)')

    if refine:
        def relax_tight(item):
            rep, members = item
            try:
                mol_opt, info = _relax_start(surface, rep['mol'], engine, False,
                                             relax_kw)
            except Exception as exc:
                return None, None, f'{type(exc).__name__}: {exc}'
            if info.get('energy') is None:
                return None, None, info.get('status', 'no energy')
            return np.asarray(mol_opt.atom_coords()), info, None

        refined = []
        for (rep, members), (crd_opt, info, why) in zip(
                groups, runner(relax_tight, list(groups))):
            if crd_opt is None:
                failed.append((rep['offsets'], why))
                continue
            refined.append(dict(rep, mol=_mol_at(mol, crd_opt),
                                energy=float(info['energy']),
                                converged=bool(info.get('converged')),
                                info=info,
                                index=[records[n]['index'] for n in members]))
        if not refined:
            raise RuntimeError(f'every screened conformer failed to converge: '
                               f'{failed}')
        groups = deduplicate(refined, mol, perms, energy_tol, rmsd_tol)
        records = refined

    reps = [rep for rep, _ in groups]
    weights = boltzmann_weights([r['energy'] for r in reps], temperature)
    out = []
    for (rep, members), w in zip(groups, weights):
        merged = []
        for n in members:
            idx = records[n]['index']
            merged.extend(idx if isinstance(idx, list) else [idx])
        out.append(Conformer(mol=rep['mol'], energy=rep['energy'],
                             weight=float(w),
                             torsions=torsion_values(rep['mol'], torsions),
                             start=rep['offsets'], starts=sorted(merged),
                             converged=rep['converged'],
                             info=dict(rep['info'], failed_starts=failed)))
    if verbose:
        print(f'  {"E - E0 / eV":>12s} {"weight":>8s}  {"torsions / deg":>s}')
        for c in out:
            tors = ' '.join(f'{t:7.1f}' for t in c.torsions)
            print(f'  {(c.energy - out[0].energy) * HARTREE_TO_EV:12.4f} '
                  f'{c.weight:8.4f}  {tors}')
    return out


def conformer_average(conformers, routine, surface=None):
    """Boltzmann average of a property routine over the ensemble.

    `routine(conformer)` is called at each minimum, or `routine(conformer,
    surface.refreeze(conformer.mol))` when a surface is given -- which is the
    form a cubic chain needs, since the quasiparticle set, the frame and the
    interpolation layout were frozen at one geometry and the conformers are as
    far apart as a geometry optimization ever moves.

    Returns the per-conformer values alongside the average, because the SPREAD
    is the result whenever it is large: a property that varies by more than its
    own error bar across the populated minima is not a property of the molecule
    at a single geometry, and reporting only the mean hides that.
    """
    values = [routine(c) if surface is None
              else routine(c, surface.refreeze(c.mol)) for c in conformers]
    w = np.array([c.weight for c in conformers])
    average = sum(wi * np.asarray(v, float) for wi, v in zip(w, values))
    return {'values': values, 'weights': w, 'average': average}
