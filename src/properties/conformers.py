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

BEYOND THE TORSIONAL GRID. Ring puckers and nitrogen inversions are extra
starts (`ring_conformers`, through `extra_starts`); a chiral minimum stands for
itself and its mirror image and is counted twice (`mirror_degeneracy`); the
vibrational and rotational entropy is `conformer_weights`; barriers and the
following of one state through a twist are `torsion_scan`; and
`excited_conformers` relaxes each ground-state conformer on an excited surface
and pairs them, the twisted-ICT case.

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

from src.Base.numerics.constants import (BOHR_TO_ANGSTROM, BOLTZMANN_HARTREE_PER_KELVIN,
                                         COLOUR_REFINEMENT_ROUNDS, CONFORMER_TWIST_TOL,
                                         CONFORMER_BOND_SCALE, CONFORMER_ENERGY_TOL,
                                         CONFORMER_MAX_AUTOMORPHISMS,
                                         CONFORMER_MAX_STARTS, CONFORMER_RMSD_TOL,
                                         CONFORMER_SCREEN_CONV,
                                         CONFORMER_SINGLE_BOND_RATIO,
                                         CONFORMER_TEMPERATURE, CONFORMER_TORSION_GRID,
                                         HARTREE_TO_EV)
from src.Base.geometry.body_frame import centred, kabsch_rotation
from src.Base.geometry.geometry import molecule_at
from src.Base.step_failed import StepFailed
from src.properties.optimize import relax



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
    index that relaxed into it. `degeneracy` is 2 for a chiral conformer,
    whose mirror image the deduplication merged into it (`mirror_degeneracy`),
    else 1; the weight counts it.
    """

    mol: object
    energy: float
    weight: float
    torsions: np.ndarray
    start: tuple
    starts: list = field(default_factory=list)
    converged: bool = True
    info: dict = field(default_factory=dict)
    degeneracy: int = 1


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

def dihedral_frame(x, i, j, k, l):
    """F, G, H and the plane normals A = F x G, B = H x G of i-j-k-l
    (Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996))."""
    f, g, h = x[i] - x[j], x[j] - x[k], x[l] - x[k]
    return f, g, h, np.cross(f, g), np.cross(h, g)


def dihedral_radians(coords, i, j, k, l):
    """The signed i-j-k-l dihedral in radians, in (-pi, pi]: anti is pi, syn
    is 0, and the sign is IUPAC's (positive when, looking along j->k, the
    bond k-l is turned clockwise from i-j)."""
    _, g, _, a, b = dihedral_frame(np.asarray(coords, float), i, j, k, l)
    return float(np.arctan2(np.cross(b, a) @ g / np.linalg.norm(g), a @ b))


def dihedral(coords, i, j, k, l):
    """The i-j-k-l dihedral in degrees as the conformer search counts it:
    anti is +-180, syn is 0, and the sign is the opposite of IUPAC's
    (`dihedral_radians`), the sense `rotate_torsion` advances."""
    return float(-np.degrees(dihedral_radians(coords, i, j, k, l)))


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
    a0, b0 = centred(a), centred(b)
    rot = kabsch_rotation(b, a)
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

def boltzmann_weights(energies, temperature=CONFORMER_TEMPERATURE,
                      degeneracies=None):
    """Populations g exp(-E/kT) normalized to one, from TOTAL energies in
    Hartree (or free energies) and the degeneracy g of each (1 when None).

    With electronic energies and the mirror degeneracy this is what
    `search_conformers` reports. The rest of the conformational entropy -- the
    vibrational and rotational partition functions of each minimum and its
    rotational symmetry number -- is `conformer_weights.free_energy_weights`,
    which needs a Hessian per conformer; for minima a few k_B T apart those
    are corrections of the same order as the gap itself. A temperature that
    is not positive is refused.
    """
    e = np.asarray(energies, float)
    g = np.ones_like(e) if degeneracies is None else np.asarray(degeneracies,
                                                                float)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    if kt <= 0.0:
        raise ValueError(f'temperature {temperature} K is not positive')
    w = g * np.exp(-(e - e.min()) / kt)
    return w / w.sum()


def mirror_degeneracy(mol, perms=None, rmsd_tol=CONFORMER_RMSD_TOL):
    """2 when `mol`'s mirror image is a different structure, 1 when it is the
    same one.

    The comparison `deduplicate` makes, without the reflection: the heavy-atom
    RMSD between the structure and its mirror image over proper superpositions
    and the graph's automorphisms. Above `rmsd_tol` the two are enantiomers,
    which `conformer_rmsd` merges into ONE conformer -- so that conformer
    stands for two minima and is counted twice in the populations (RT ln 2 =
    18 meV at 300 K). gauche-butane is 2, anti-butane 1.
    """
    heavy = heavy_atoms(mol)
    x = np.asarray(mol.atom_coords())[heavy] * BOHR_TO_ANGSTROM
    if len(x) < 4:
        return 1          # three points are always planar: achiral
    perms = graph_automorphisms(mol) if perms is None else perms
    mirror = x * np.array([-1.0, 1.0, 1.0])
    best = min(kabsch_rmsd(x, mirror[perm]) for perm in perms)
    return 2 if best > rmsd_tol else 1


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


def search_conformers(surface, mol=None, n_grid=CONFORMER_TORSION_GRID,
                      max_starts=CONFORMER_MAX_STARTS,
                      temperature=CONFORMER_TEMPERATURE, engine='auto',
                      refine=True, torsions=None,
                      energy_tol=CONFORMER_ENERGY_TOL,
                      rmsd_tol=CONFORMER_RMSD_TOL, verbose=True, map_fn=None,
                      extra_starts=(), **relax_kw):
    """Torsional minima of `surface`'s state. Returns [Conformer], lowest first.

    extra_starts: [(coords in Bohr, label)] relaxed beside the torsional grid:
        the ring puckers and inversions of `ring_conformers`
        (`puckering_starts`, `inversion_starts`), or any geometry the caller
        wants searched. Their Conformer.start is the label.

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

    A start whose walk fails a step it cannot shorten (`StepFailed`: a state
    lost, a broken Casida solve, conventions that cannot be matched) is dropped
    with its status recorded: the conformers that did relax are still the
    useful output. Any other exception is a defect and propagates.
    """
    mol = surface.mol0 if mol is None else mol
    adj = molecular_graph(mol)
    torsions = rotatable_bonds(mol, adj) if torsions is None else torsions
    perms = graph_automorphisms(mol, adj)
    starts = (torsion_starts(mol, torsions, n_grid, max_starts)
              + [(np.asarray(c, float), label) for c, label in extra_starts])
    if verbose:
        print(f'  [conformers] {len(torsions)} rotatable bond(s), '
              f'{len(extra_starts)} extra start(s), {len(starts)} start(s) on '
              f'{surface.label()}')

    # Every start relaxes on its own; `map_fn` (map's signature, default the
    # builtin) is where a process pool spreads them. The mapped function hands
    # back COORDINATES, not a Mole, so its result crosses a pickle boundary;
    # the record's Mole is rebuilt here from them.
    runner = map if map_fn is None else map_fn

    def relax_loose(item):
        n, (crd, offsets) = item
        try:
            mol_opt, info = _relax_start(surface, molecule_at(mol, crd), engine,
                                         True, relax_kw)
        except StepFailed as exc:
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
        records.append({'mol': molecule_at(mol, crd_opt), 'energy': float(info['energy']),
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
            except StepFailed as exc:
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
            refined.append(dict(rep, mol=molecule_at(mol, crd_opt),
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
    degeneracy = [mirror_degeneracy(r['mol'], perms, rmsd_tol) for r in reps]
    weights = boltzmann_weights([r['energy'] for r in reps], temperature,
                                degeneracy)
    out = []
    for (rep, members), w, g in zip(groups, weights, degeneracy):
        merged = []
        for n in members:
            idx = records[n]['index']
            merged.extend(idx if isinstance(idx, list) else [idx])
        out.append(Conformer(mol=rep['mol'], energy=rep['energy'],
                             weight=float(w),
                             torsions=torsion_values(rep['mol'], torsions),
                             start=rep['offsets'], starts=sorted(merged),
                             converged=rep['converged'],
                             info=dict(rep['info'], failed_starts=failed),
                             degeneracy=g))
    if verbose:
        print(f'  {"E - E0 / eV":>12s} {"weight":>8s} {"g":>2s}  '
              f'{"torsions / deg":>s}')
        for c in out:
            tors = ' '.join(f'{t:7.1f}' for t in c.torsions)
            print(f'  {(c.energy - out[0].energy) * HARTREE_TO_EV:12.4f} '
                  f'{c.weight:8.4f} {c.degeneracy:2d}  {tors}')
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


def excited_conformers(ground, surface, torsions=None, engine='auto',
                       refreeze=1, energy_tol=CONFORMER_ENERGY_TOL,
                       rmsd_tol=CONFORMER_RMSD_TOL,
                       twist_tol=CONFORMER_TWIST_TOL, verbose=True,
                       **relax_kw):
    """The excited minima reached from each ground-state conformer: the
    twisted-ICT case, where the emitting conformer need not be the absorbing
    one: search the ground state, relax each of its conformers on the
    excited surface, and pair the two sets.

    ground: [Conformer] of the ground state (`search_conformers` on it), with
        their populations. surface: the excited state's surface; each
        relaxation starts on `surface.refreeze(conformer.mol)` and refreezes
        its conventions `refreeze` more times at its own minimum (the outer
        loop of `optimize`), so no frozen convention is carried across the
        torsion. A surface that follows its root by overlap follows it along
        each walk.

    Returns (pairs, minima). `pairs`: one dict per ground conformer --
    'ground' (the Conformer), 'excited' (index into minima or None when the
    walk failed), 'gap' (E_excited - E_ground, total energies, Hartree: the
    adiabatic gap of that pair), 'twist' (degrees per torsion, excited minus
    ground, wrapped to [-180, 180)), 'tict' (any |twist| above `twist_tol`),
    'status'. `minima`: the distinct excited minima (`deduplicate`), each a
    dict with 'mol', 'energy', 'torsions', 'from' (ground indices), 'info'.

    A rate exponential in the gap is dominated by the SMALLEST gap of the
    populated pairs, not by their mean: read the pairs, not an average.
    """
    if not ground:
        raise ValueError('no ground-state conformers to start from')
    mol = ground[0].mol
    if torsions is None:
        torsions = rotatable_bonds(mol)
    perms = graph_automorphisms(mol)
    records, pairs = [], []
    for n, c in enumerate(ground):
        start = surface.refreeze(c.mol)
        try:
            mol_opt, info = _relax_start(start, c.mol, engine, False,
                                         dict(relax_kw, refreeze=refreeze))
            why = None if info.get('energy') is not None else info.get(
                'status', 'no energy')
        except StepFailed as exc:
            why = f'{type(exc).__name__}: {exc}'
        if why is not None:
            pairs.append({'ground': c, 'excited': None, 'gap': None,
                          'twist': None, 'tict': None, 'status': why})
            continue
        records.append({'mol': mol_opt, 'energy': float(info['energy']),
                        'info': info, 'ground': n})
        twist = ((torsion_values(mol_opt, torsions)
                  - torsion_values(c.mol, torsions) + 180.0) % 360.0 - 180.0)
        pairs.append({'ground': c, 'excited': None,
                      'gap': float(info['energy']) - c.energy,
                      'twist': twist,
                      'tict': bool(np.any(np.abs(twist) > twist_tol)),
                      'status': 'relaxed', 'record': len(records) - 1})
    minima = []
    if records:
        for rep, members in deduplicate(records, mol, perms, energy_tol,
                                        rmsd_tol):
            minima.append({'mol': rep['mol'], 'energy': rep['energy'],
                           'torsions': torsion_values(rep['mol'], torsions),
                           'from': [records[m]['ground'] for m in members],
                           'info': rep['info']})
            for p in pairs:
                if p.get('record') in members:
                    p['excited'] = len(minima) - 1
    for p in pairs:
        p.pop('record', None)
    if verbose:
        print(f'  {"ground":>6s} {"weight":>8s} {"gap / eV":>9s} {"TICT":>5s}  '
              f'twist / deg')
        for n, p in enumerate(pairs):
            if p['gap'] is None:
                print(f'  {n:6d} {p["ground"].weight:8.4f}  failed: '
                      f'{p["status"]}')
                continue
            print(f'  {n:6d} {p["ground"].weight:8.4f} '
                  f'{p["gap"] * HARTREE_TO_EV:9.4f} {str(p["tict"]):>5s}  '
                  + ' '.join(f'{t:7.1f}' for t in p['twist']))
    return pairs, minima
