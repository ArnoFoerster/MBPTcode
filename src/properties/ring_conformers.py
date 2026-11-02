"""Conformer starts a torsional grid cannot reach: ring puckering and
pyramidal inversion.

`conformers.search_conformers` turns rotatable bonds; a ring bond is not one
(rotating it would break another bond), and the umbrella of an amine or
phosphine centre is not a torsion at all. Cyclohexane's twist-boat, the
butterfly fold of an acridan or phenoxazine donor and the axial/equatorial
pair of an N-substituted ring are therefore invisible to the grid. This
module makes the starts for them; `search_conformers(extra_starts=...)` relaxes
them with the torsional ones, through the same optimizers and the same
deduplication.

PUCKERING by Cremer-Pople coordinates (J. Am. Chem. Soc. 97, 1354 (1975)). For
an N-membered ring with out-of-plane displacements z_j about its mean plane,

    q_m cos phi_m =  sqrt(2/N) sum_j z_j cos(2 pi m j / N)
    q_m sin phi_m = -sqrt(2/N) sum_j z_j sin(2 pi m j / N),   2 <= m < N/2
    q_{N/2}       =  sqrt(1/N) sum_j z_j (-1)^j               (N even)

A start sets one puckering mode to RING_PUCKER_AMPLITUDE: the phases
RING_PUCKER_PHASES of every m < N/2 (boat/twist-boat starts for N = 6,
envelope/twist ones for N = 5), off the symmetric values so that no start sits
on a saddle its symmetry keeps it on, and both signs of q_{N/2} (the chair and
the inverted chair).
Only the ring atoms' z changes; a substituent keeps its place in the local
frame of the ring atom it hangs from (an axial hydrogen stays axial), so bonds
to the ring are carried along and the relaxation only has to repair the ring
itself. Pure m >= 2 components leave the mean plane
and the centroid where they were, so the start has exactly the coordinates
asked for.

A ring is searched when it can pucker: it holds an atom with four neighbours,
or at least two bonds long enough to be single (RING_SINGLE_BOND_RATIO). An
aromatic ring is neither; a furan passes and costs starts that relax back to
planar and merge.

INVERSION reflects a three-coordinate N, P or As centre through the plane of
its neighbours when it stands more than INVERSION_MIN_HEIGHT out of it (a
planarized amide or carbazole nitrogen has no second umbrella). Where the
inverted structure is the mirror image of the input, deduplication merges the
two and `conformers.mirror_degeneracy` counts it twice in the weights; where
another stereo element makes it a diastereomer (an axial/equatorial N-alkyl)
the two stay separate conformers.

The combinations of puckers, inversions and torsions are not enumerated: each
extra start changes one coordinate of the input geometry.
"""
from collections import deque

import numpy as np
from pyscf.data import radii

from src.Base.numerics.constants import (BOHR_TO_ANGSTROM, INVERSION_MIN_HEIGHT, RING_PUCKER_AMPLITUDE,
                                         RING_PUCKER_PHASES, RING_SINGLE_BOND_RATIO, RING_SIZES)
from src.properties.conformers import molecular_graph

#: Elements that invert as pyramidal three-coordinate centres.
INVERSION_ELEMENTS = (7, 15, 33)


def smallest_rings(adj, sizes=RING_SIZES):
    """The rings of the graph, each as an ordered tuple of atoms around it.

    For every edge, the shortest cycle through it (the shortest path between
    its ends with the edge cut); duplicates removed. Every ring of a fused
    system is the shortest cycle through at least one of its bonds, so the
    set holds every ring an organic chemist would draw.
    """
    rings = {}
    for a in range(len(adj)):
        for b in adj[a]:
            if b < a:
                continue
            prev = {a: None}
            queue = deque([a])
            while queue and b not in prev:
                cur = queue.popleft()
                for nxt in adj[cur]:
                    if (cur, nxt) in ((a, b), (b, a)) or nxt in prev:
                        continue
                    prev[nxt] = cur
                    queue.append(nxt)
            if b not in prev:
                continue
            path, cur = [], b
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            if len(path) not in sizes:
                continue
            key = frozenset(path)
            if key not in rings:
                rings[key] = tuple(path)
    return sorted(rings.values(), key=lambda r: (len(r), sorted(r)))


def flexible_rings(mol, adj=None, single_ratio=RING_SINGLE_BOND_RATIO):
    """The rings that can pucker (module docstring): an atom with four
    neighbours, or two bonds long enough to be single."""
    adj = molecular_graph(mol) if adj is None else adj
    z = np.asarray(mol.atom_charges())
    x = np.asarray(mol.atom_coords())
    out = []
    for ring in smallest_rings(adj):
        if any(len(adj[a]) >= 4 for a in ring):
            out.append(ring)
            continue
        single = 0
        for a, b in zip(ring, ring[1:] + ring[:1]):
            r_sum = radii.COVALENT[int(z[a])] + radii.COVALENT[int(z[b])]
            single += np.linalg.norm(x[a] - x[b]) >= single_ratio * r_sum
        if single >= 2:
            out.append(ring)
    return out


def mean_plane(coords, ring):
    """(centroid, unit normal) of the Cremer-Pople mean plane of `ring`."""
    x = np.asarray(coords, float)[list(ring)]
    n = len(ring)
    c = x.mean(0)
    ang = 2.0 * np.pi * np.arange(n) / n
    r1 = ((x - c) * np.sin(ang)[:, None]).sum(0)
    r2 = ((x - c) * np.cos(ang)[:, None]).sum(0)
    normal = np.cross(r1, r2)
    return c, normal / np.linalg.norm(normal)


def cremer_pople(coords, ring):
    """Puckering coordinates of `ring` at `coords` (Bohr).

    Returns {'z': out-of-plane displacements (Bohr), 'q': {m: amplitude},
    'phi': {m: phase in degrees} (m < N/2), 'Q': total amplitude}; the
    amplitudes in Bohr.
    """
    c, normal = mean_plane(coords, ring)
    z = (np.asarray(coords, float)[list(ring)] - c) @ normal
    n = len(ring)
    j = np.arange(n)
    q, phi = {}, {}
    for m in range(2, (n - 1) // 2 + 1):
        a = np.sqrt(2.0 / n) * (z * np.cos(2.0 * np.pi * m * j / n)).sum()
        b = -np.sqrt(2.0 / n) * (z * np.sin(2.0 * np.pi * m * j / n)).sum()
        q[m] = float(np.hypot(a, b))
        phi[m] = float(np.degrees(np.arctan2(b, a)) % 360.0)
    if n % 2 == 0:
        q[n // 2] = float(np.sqrt(1.0 / n) * (z * (-1.0) ** j).sum())
    return {'z': z, 'q': q, 'phi': phi,
            'Q': float(np.sqrt((z ** 2).sum()))}


def pucker_profile(n, m, amplitude, phase_deg=0.0):
    """z_j of one Cremer-Pople mode m of an n-ring (the inverse transform);
    for m = n/2 the sign of `amplitude` picks the chair or its inverse."""
    j = np.arange(n)
    if 2 * m == n:
        return amplitude * np.sqrt(1.0 / n) * (-1.0) ** j
    return (amplitude * np.sqrt(2.0 / n)
            * np.cos(np.radians(phase_deg) + 2.0 * np.pi * m * j / n))


def riders(adj, ring):
    """{atom: ring atoms it hangs from}: every atom outside the ring with the
    ring atoms it reaches without passing through another ring atom."""
    members = set(ring)
    hang = {}
    for r in ring:
        seen, stack = {r}, [r]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt in members or nxt in seen:
                    continue
                seen.add(nxt)
                hang.setdefault(nxt, set()).add(r)
                stack.append(nxt)
    return hang


def local_frame(x, ring, k):
    """(origin, 3x3 rows e1, e2, e3) of ring atom ring[k]: e1 the bisector
    of its two ring bonds, e2 their plane's normal, e3 = e1 x e2."""
    r, p, n = ring[k], ring[k - 1], ring[(k + 1) % len(ring)]
    u, v = x[p] - x[r], x[n] - x[r]
    e1 = u / np.linalg.norm(u) + v / np.linalg.norm(v)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, v)
    e2 /= np.linalg.norm(e2)
    return x[r].copy(), np.array([e1, e2, np.cross(e1, e2)])


def puckered(coords, ring, adj, z_target):
    """`coords` with `ring`'s out-of-plane displacements set to `z_target`
    (Bohr) about its mean plane.

    A rider hanging from one ring atom keeps its place in that atom's local
    frame (the bisector and the normal of its two ring bonds), so an axial
    and an equatorial hydrogen stay axial and equatorial on the new ring and
    keep their bond lengths and angles; a rider hanging from several ring
    atoms (a fused ring) moves along the normal by their mean change.
    """
    x = np.asarray(coords, float).copy()
    c, normal = mean_plane(x, ring)
    z_now = (x[list(ring)] - c) @ normal
    dz = np.asarray(z_target, float) - z_now
    new = x.copy()
    for k, a in enumerate(ring):
        new[a] = x[a] + normal * dz[k]
    frames = [(local_frame(x, ring, k), local_frame(new, ring, k))
              for k in range(len(ring))]
    where = {a: k for k, a in enumerate(ring)}
    for atom, hosts in riders(adj, ring).items():
        if len(hosts) == 1:
            k = where[next(iter(hosts))]
            (o0, f0), (o1, f1) = frames[k]
            new[atom] = o1 + f1.T @ (f0 @ (x[atom] - o0))
        else:
            new[atom] = x[atom] + normal * np.mean([dz[where[h]]
                                                    for h in hosts])
    return new


def puckering_starts(mol, adj=None, amplitude=RING_PUCKER_AMPLITUDE):
    """[(coords in Bohr, label)]: one start per puckering mode and phase of
    every flexible ring (module docstring), each from `mol`'s geometry."""
    adj = molecular_graph(mol) if adj is None else adj
    x0 = np.asarray(mol.atom_coords(), float)
    amp = float(amplitude) / BOHR_TO_ANGSTROM
    out = []
    for ring in flexible_rings(mol, adj):
        n = len(ring)
        for m in range(2, n // 2 + 1):
            shapes = ([(amp, 0.0), (-amp, 0.0)] if 2 * m == n
                      else [(amp, p) for p in RING_PUCKER_PHASES])
            for a, phase in shapes:
                z = pucker_profile(n, m, a, phase)
                label = (f'ring {ring}: q{m} {"+" if a > 0 else "-"}'
                         if 2 * m == n else
                         f'ring {ring}: q{m} phi {phase:.0f}')
                out.append((puckered(x0, ring, adj, z), label))
    return out


def inversion_centres(mol, adj=None, min_height=INVERSION_MIN_HEIGHT):
    """[(centre, (three neighbours), height in Angstrom)] of every pyramidal
    three-coordinate N/P/As."""
    adj = molecular_graph(mol) if adj is None else adj
    z = np.asarray(mol.atom_charges())
    x = np.asarray(mol.atom_coords(), float)
    out = []
    for a in range(mol.natm):
        if int(z[a]) not in INVERSION_ELEMENTS or len(adj[a]) != 3:
            continue
        nb = tuple(sorted(adj[a]))
        p = x[list(nb)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        normal /= np.linalg.norm(normal)
        h = float((x[a] - p.mean(0)) @ normal) * BOHR_TO_ANGSTROM
        if abs(h) >= min_height:
            out.append((a, nb, h))
    return out


def inverted(coords, centre, neighbours):
    """`coords` with `centre` reflected through the plane of its three
    neighbours: the other umbrella."""
    x = np.asarray(coords, float).copy()
    p = x[list(neighbours)]
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    normal /= np.linalg.norm(normal)
    x[centre] -= 2.0 * ((x[centre] - p.mean(0)) @ normal) * normal
    return x


def inversion_starts(mol, adj=None, min_height=INVERSION_MIN_HEIGHT):
    """[(coords in Bohr, label)]: one start per pyramidal centre, inverted."""
    adj = molecular_graph(mol) if adj is None else adj
    x0 = np.asarray(mol.atom_coords(), float)
    return [(inverted(x0, a, nb), f'inversion at atom {a}')
            for a, nb, _ in inversion_centres(mol, adj, min_height)]
