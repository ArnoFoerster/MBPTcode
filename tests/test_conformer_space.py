"""Gates for what the torsional conformer search could not reach:
src/properties/ring_conformers.py (ring puckering, pyramidal inversion),
src/properties/conformer_weights.py and conformers.mirror_degeneracy
(conformational entropy), src/properties/torsion_scan.py (relaxed scans,
barriers, following a state through a sweep) and
conformers.excited_conformers (the twisted-ICT pairing).

Cheap tier: geometry, graph theory, point groups and a model surface, no
electronic structure. Expensive tier (HF/STO-3G through the repository's
optimizers): cyclohexane's twist-boat, butadiene's torsional barrier and
ethylene's twisted triplet. Run the cheap tier alone with
    -k "not search and not barrier and not twisted"
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (BOHR_TO_ANGSTROM,
                                         BOLTZMANN_HARTREE_PER_KELVIN,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         RING_PUCKER_AMPLITUDE)
from src.gradients.gwbse.state_following import RootFollowingLost
from src.gradients.mean_field_surface import MeanFieldSurface
from src.gradients.surface import PotentialEnergySurface
from src.properties.conformer_weights import (free_energy_terms,
                                              free_energy_weights,
                                              harmonic_free_energy,
                                              symmetry_number)
from src.properties.conformers import (Conformer, Torsion, boltzmann_weights,
                                       deduplicate, dihedral,
                                       excited_conformers, molecular_graph,
                                       mirror_degeneracy, rotatable_bonds,
                                       search_conformers, torsion_starts)
from src.properties.duschinsky import CartesianHessian
from src.properties.ring_conformers import (cremer_pople, flexible_rings,
                                            inversion_centres,
                                            inversion_starts, pucker_profile,
                                            puckered, puckering_starts,
                                            smallest_rings)
from src.properties.torsion_scan import (TorsionRestrained, barrier,
                                         dihedral_gradient, torsion_scan)
from tests.test_conformers import BUTADIENE, CYCLOHEXANE, scf_factory

BUTANE = ('C 0 0 0; C 1.53 0 0; C 2.04 1.44 0; C 3.57 1.44 0; '
          'H -0.36 -1.02 0; H -0.36 0.51 0.89; H -0.36 0.51 -0.89; '
          'H 1.89 -0.51 0.89; H 1.89 -0.51 -0.89; H 1.68 1.95 0.89; '
          'H 1.68 1.95 -0.89; H 3.93 0.42 0; H 3.93 1.95 0.89; '
          'H 3.93 1.95 -0.89')
AMMONIA = 'N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; H -0.81 -0.47 -0.27'
ETHYLENE = ('C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; '
            'H 0 0.923 -1.237; H 0 -0.923 -1.237')


def molecule(atoms):
    return gto.M(atom=atoms, basis='sto-3g', verbose=0)


def bond_lengths(x, adj):
    return {(a, b): np.linalg.norm(x[a] - x[b]) for a in range(len(adj))
            for b in adj[a] if a < b}


# --------------------------------------------------------------- puckering
def test_rings_are_found_and_aromatic_ones_left_alone():
    chex = molecule(CYCLOHEXANE)
    assert [sorted(r) for r in flexible_rings(chex)] == [list(range(6))]
    benzene = molecule('C 1.39 0 0; C 0.695 1.204 0; C -0.695 1.204 0; '
                       'C -1.39 0 0; C -0.695 -1.204 0; C 0.695 -1.204 0; '
                       'H 2.47 0 0; H 1.235 2.139 0; H -1.235 2.139 0; '
                       'H -2.47 0 0; H -1.235 -2.139 0; H 1.235 -2.139 0')
    assert len(smallest_rings(molecular_graph(benzene))) == 1
    assert flexible_rings(benzene) == []
    assert flexible_rings(molecule(BUTANE)) == []


def test_cremer_pople_round_trip():
    """A start has exactly the puckering asked for, every mode and sign, and
    its substituents keep their bonds to the ring."""
    chex = molecule(CYCLOHEXANE)
    adj = molecular_graph(chex)
    ring = flexible_rings(chex, adj)[0]
    x0 = chex.atom_coords()
    amp = RING_PUCKER_AMPLITUDE / BOHR_TO_ANGSTROM
    for m, a, phase in ((2, amp, 15.0), (2, amp, 75.0), (3, amp, 0.0),
                        (3, -amp, 0.0)):
        x = puckered(x0, ring, adj, pucker_profile(6, m, a, phase))
        cp = cremer_pople(x, ring)
        assert cp['q'][m] == pytest.approx(a if m == 3 else abs(a), abs=1e-10)
        other = 5 - m
        assert abs(cp['q'][other]) < 1e-10
        if m == 2:
            assert cp['phi'][2] == pytest.approx(phase, abs=1e-8)
        # every C-H keeps its length (the hydrogens ride in their carbon's
        # frame); only the ring bonds stretch
        b0, b1 = bond_lengths(x0, adj), bond_lengths(x, adj)
        ch = [k for k in b0 if not (k[0] in ring and k[1] in ring)]
        assert max(abs(b1[k] - b0[k]) for k in ch) < 1e-10
    labels = [label for _, label in puckering_starts(chex, adj)]
    assert len(labels) == 4
    # the chair the input already is: its q3 is that of a chair
    assert abs(cremer_pople(x0, ring)['q'][3]) * BOHR_TO_ANGSTROM > 0.5


# --------------------------------------------------------------- inversion
def test_inversion_reflects_a_pyramidal_centre_only():
    nh3 = molecule(AMMONIA)
    (centre, nb, h), = inversion_centres(nh3)
    assert centre == 0 and nb == (1, 2, 3) and abs(h) > 0.3
    (x, label), = inversion_starts(nh3)
    assert np.abs(x[1:] - nh3.atom_coords()[1:]).max() == 0.0
    (_, _, h_inv), = inversion_centres(molecule_at(nh3, x))
    assert h_inv == pytest.approx(-h, rel=1e-12)
    # bonds unchanged: a reflection through the neighbours' plane
    adj = molecular_graph(nh3)
    b0, b1 = bond_lengths(nh3.atom_coords(), adj), bond_lengths(x, adj)
    assert max(abs(b1[k] - b0[k]) for k in b0) < 1e-12
    flat = molecule('N 0 0 0; H 0 1.0 0; H 0.866 -0.5 0; H -0.866 -0.5 0')
    assert inversion_centres(flat) == [] and inversion_starts(flat) == []


# ------------------------------------------------------ mirror and entropy
def test_gauche_is_chiral_and_counted_twice():
    """anti-butane is its own mirror image; gauche+ and gauche- are one
    conformer standing for two minima."""
    but = molecule(BUTANE)
    tor = rotatable_bonds(but)
    starts = torsion_starts(but, tor)
    mols = [molecule_at(but, x) for x, _ in starts]
    assert [mirror_degeneracy(m) for m in mols] == [1, 2, 2]
    records = [{'mol': m, 'energy': e} for m, e in
               zip(mols, (0.0, 1e-3, 1e-3))]
    groups = deduplicate(records, but)
    assert [g for _, g in groups] == [[0], [1, 2]]
    w = boltzmann_weights([0.0, 0.0], degeneracies=[1, 2])
    assert w == pytest.approx([1.0 / 3.0, 2.0 / 3.0], rel=1e-14)
    with pytest.raises(ValueError, match='not positive'):
        boltzmann_weights([0.0, 1e-3], temperature=0.0)


def test_symmetry_numbers():
    chex = molecule(CYCLOHEXANE)
    ring = flexible_rings(chex)[0]
    adj = molecular_graph(chex)
    amp = 0.6 / BOHR_TO_ANGSTROM
    chair = molecule_at(chex, puckered(chex.atom_coords(), ring, adj,
                                       pucker_profile(6, 3, amp)))
    twist = molecule_at(chex, puckered(chex.atom_coords(), ring, adj,
                                       pucker_profile(6, 2, amp, 90.0)))
    boat = molecule_at(chex, puckered(chex.atom_coords(), ring, adj,
                                      pucker_profile(6, 2, amp, 0.0)))
    assert symmetry_number(chair) == (6, 'D3d')
    assert symmetry_number(boat) == (2, 'C2v')
    assert (mirror_degeneracy(chair), mirror_degeneracy(twist),
            mirror_degeneracy(boat)) == (1, 2, 1)
    assert symmetry_number(molecule(AMMONIA)) == (3, 'C3v')
    methane = molecule('C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; '
                       'H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629')
    assert symmetry_number(methane) == (12, 'Td')


def test_harmonic_free_energy_limits():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * 300.0
    stiff = 3000.0 / HARTREE_TO_CM
    assert harmonic_free_energy([stiff], 300.0) == pytest.approx(
        0.5 * stiff, rel=1e-6)
    soft = 1e-3 * kt
    assert harmonic_free_energy([soft], 300.0) == pytest.approx(
        kt * np.log(soft / kt), rel=1e-6)


def test_free_energy_weights_add_the_terms():
    """G = E + vib + rot + kT ln sigma - kT ln g, populations exp(-G/kT); the
    softer twist-boat-like conformer gains on the stiffer one."""
    chex = molecule(CYCLOHEXANE)
    ring = flexible_rings(chex)[0]
    adj = molecular_graph(chex)
    amp = 0.6 / BOHR_TO_ANGSTROM
    geoms = [puckered(chex.atom_coords(), ring, adj, pucker_profile(6, 3, amp)),
             puckered(chex.atom_coords(), ring, adj,
                      pucker_profile(6, 2, amp, 90.0))]
    rng = np.random.default_rng(4)
    a = rng.normal(size=(54, 54))
    base = a @ a.T / 54.0 * 0.3
    hess = [CartesianHessian(molecule_at(chex, g), base * s, 'model')
            for g, s in zip(geoms, (1.0, 0.6))]
    confs = [Conformer(mol=molecule_at(chex, g), energy=e, weight=0.5,
                       torsions=np.zeros(0), start=(), degeneracy=d)
             for g, e, d in zip(geoms, (0.0, 0.004), (1, 2))]
    out, terms = free_energy_weights(confs, 300.0, hessians=hess)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * 300.0
    for c, t in zip(confs, terms):
        own = free_energy_terms(c.mol, c.energy, 300.0, hess[confs.index(c)],
                                c.degeneracy)
        assert t['free_energy'] == pytest.approx(own['free_energy'],
                                                 abs=1e-14)
        assert t['free_energy'] == pytest.approx(
            t['energy'] + t['vibrational'] + t['rotational']
            + kt * np.log(t['symmetry_number']) - kt * np.log(t['degeneracy']),
            abs=1e-14)
    g = np.array([t['free_energy'] for t in terms])
    p = np.exp(-(g - g.min()) / kt)
    assert [c.weight for c in out] == pytest.approx(p / p.sum(), rel=1e-12)
    electronic = boltzmann_weights([0.0, 0.004], 300.0, [1, 2])
    assert out[1].weight > electronic[1]
    with pytest.raises(ValueError, match='some conformers'):
        free_energy_weights(confs, 300.0, hessians=[hess[0], None])


# ------------------------------------------------------- the restrained scan
def test_restraint_gradient_is_the_derivative_of_its_energy():
    but = molecule(BUTANE)
    tor = rotatable_bonds(but)[0]
    x = but.atom_coords() + np.random.default_rng(5).normal(0, 0.05, (14, 3))
    g = dihedral_gradient(x, tor)
    held = TorsionRestrained(ModelSurface(molecule_at(but, x), tor), tor, 75.0)
    _, gr = held.restraint(molecule_at(but, x))
    fd, h = np.zeros_like(x), 1e-5
    for a in range(len(x)):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a, c] += h
            xm[a, c] -= h
            fd[a, c] = ((held.restraint(molecule_at(but, xp))[0]
                         - held.restraint(molecule_at(but, xm))[0]) / (2 * h))
    assert np.abs(gr - fd).max() < 1e-8
    assert np.abs(g).max() > 0.1


class ModelSurface(PotentialEnergySurface):
    """Springs on every distance within each of the two fragments the
    torsion's bond separates (each rigid, the reference geometry's), plus
    V(phi) = A cos(phi) on the torsion, and a 'state' that is lost when the
    torsion jumps by more than `hold` degrees from the last committed
    geometry: the root following of an excited chain, reduced to what a
    sweep driver sees."""

    def __init__(self, mol, tor, hold=20.0, amplitude=0.004, followed=None):
        self.mol0, self.tor, self.hold, self.amp = mol, tor, hold, amplitude
        x = mol.atom_coords()
        moving = set(tor.moving) | {tor.k}
        rest = (set(range(mol.natm)) - set(tor.moving)) | {tor.j}
        self.r0 = {(a, b): np.linalg.norm(x[a] - x[b])
                   for side in (moving, rest) for a in side for b in side
                   if a < b}
        self.followed = followed
        self.commits = 0

    def phi(self, mol):
        t = self.tor
        return dihedral(mol.atom_coords(), t.i, t.j, t.k, t.l)

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        x = mol.atom_coords()
        phi = self.phi(mol)
        if self.followed is not None and abs(
                (phi - self.followed + 180.0) % 360.0 - 180.0) > self.hold:
            raise RootFollowingLost(f'model state lost at {phi:.1f} deg')
        e, g = self.amp * np.cos(np.radians(phi)), np.zeros_like(x)
        g += -self.amp * np.sin(np.radians(phi)) * dihedral_gradient(x,
                                                                     self.tor)
        for (a, b), r0 in self.r0.items():
            d = x[a] - x[b]
            r = np.linalg.norm(d)
            e += 0.05 * (r - r0) ** 2
            g[a] += 0.1 * (r - r0) * d / r
            g[b] -= 0.1 * (r - r0) * d / r
        return g, float(e), {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def commit(self, mol):
        self.followed = self.phi(mol)
        self.commits += 1

    def refreeze(self, mol, mf=None):
        return ModelSurface(self.mol0, self.tor, self.hold, self.amp,
                            followed=self.followed)

    def label(self):
        return 'model torsion surface'

    def identity(self):
        return None

    def mean_field(self, mol=None, mf=None):
        return mol, None


def test_sweep_halves_where_the_state_cannot_be_followed():
    """60 degrees in 30-degree increments on a state lost beyond 20: each
    increment is halved once and the sweep completes on the followed state,
    each point on its own refrozen surface that the followed torsion was
    handed over to; with no halving allowed it stops at the first increment
    and says why."""
    pytest.importorskip('geometric')
    but = molecule(BUTANE)
    tor = rotatable_bonds(but)[0]
    start = molecule_at(but, torsion_starts(but, [tor], n_grid=6)[1][0])
    phi0 = dihedral(start.atom_coords(), tor.i, tor.j, tor.k, tor.l)
    surface = ModelSurface(start, tor, followed=phi0)
    targets = [phi0 + 30.0, phi0 + 60.0]
    points, status = torsion_scan(surface, start, tor, targets,
                                  engine='geometric', verbose=False,
                                  force_constant=0.5, refreeze=1)
    assert status == 'complete'
    assert [round(p.target - phi0, 6) for p in points] == [15.0, 30.0, 45.0,
                                                          60.0]
    assert [p.halvings for p in points] == [1, 0, 1, 0]
    # every point refrozen at its minimum, the followed torsion handed over:
    # four surfaces, none the one the scan started on, and the last one
    # follows the last point
    surfaces = {id(p.surface) for p in points}
    assert len(surfaces) == 4 and id(surface) not in surfaces
    assert all(p.info['refreeze'] == 'measured' for p in points)
    assert points[-1].surface.followed == pytest.approx(points[-1].phi,
                                                        abs=1e-6)
    for p in points:
        assert abs(p.phi - p.target) < 1.0
    _, status = torsion_scan(ModelSurface(start, tor, followed=phi0), start,
                             tor, targets, engine='geometric', verbose=False,
                             max_halvings=0)
    assert status.startswith('stopped before') and 'RootFollowingLost' in status


def test_an_unconverged_point_is_halved_never_passed_on(monkeypatch):
    """A restrained relaxation that does not converge is not a scan point:
    the increment is halved and retried exactly as for a lost state, and
    with no halving left the scan stops and says the point did not
    converge. Model: the walk converges only within 20 degrees of the last
    committed torsion."""
    pytest.importorskip('geometric')
    from src.properties import torsion_scan as scan
    but = molecule(BUTANE)
    tor = rotatable_bonds(but)[0]
    start = molecule_at(but, torsion_starts(but, [tor], n_grid=6)[1][0])
    phi0 = dihedral(start.atom_coords(), tor.i, tor.j, tor.k, tor.l)
    real = scan.relax

    def relax(held, mol, **kw):
        # the increment from the last committed torsion, read before the
        # walk commits its own steps
        far = abs(scan.wrapped(held.target - held.surface.followed)) > 20.0
        mol_opt, info = real(held, mol, **kw)
        if far:
            info = dict(info, converged=False, status='cycle cap reached')
        return mol_opt, info
    monkeypatch.setattr(scan, 'relax', relax)
    targets = [phi0 + 30.0, phi0 + 60.0]
    points, status = torsion_scan(ModelSurface(start, tor, hold=360.0,
                                               followed=phi0),
                                  start, tor, targets, engine='geometric',
                                  verbose=False, force_constant=0.5,
                                  refreeze=1)
    assert status == 'complete'
    assert [round(p.target - phi0, 6) for p in points] == [15.0, 30.0, 45.0,
                                                          60.0]
    assert [p.halvings for p in points] == [1, 0, 1, 0]
    assert all(p.converged for p in points)
    points, status = torsion_scan(ModelSurface(start, tor, hold=360.0,
                                               followed=phi0),
                                  start, tor, targets, engine='geometric',
                                  verbose=False, max_halvings=0)
    assert points == []
    assert status.startswith('stopped before') and 'did not converge' in status
    assert 'cycle cap reached' in status


# ------------------------------------------------- electronic structure tier
def test_search_finds_the_twist_boat_through_a_pucker():
    """Cyclohexane, HF/STO-3G, from its chair: no rotatable bond, so the
    torsional grid is the input alone; the puckering starts reach the
    twist-boat (D2, chiral, counted twice) 0.26 eV up. Screening pass only.
    (The test's model geometry of a twist-boat is only C2: the relaxed one
    is D2.)"""
    chex = molecule(CYCLOHEXANE)
    assert rotatable_bonds(chex) == []
    surface = MeanFieldSurface(chex, scf_factory)
    found = search_conformers(surface, extra_starts=puckering_starts(chex),
                              refine=False, verbose=False)
    ring = flexible_rings(chex)[0]
    assert len(found) == 2
    chair, twist = found
    cp_c, cp_t = cremer_pople(chair.mol.atom_coords(), ring), \
        cremer_pople(twist.mol.atom_coords(), ring)
    assert abs(cp_c['q'][3]) > 3.0 * cp_c['q'][2]
    assert cp_t['q'][2] > 3.0 * abs(cp_t['q'][3])
    gap = (twist.energy - chair.energy) * HARTREE_TO_EV
    assert 0.2 < gap < 0.35
    assert (chair.degeneracy, twist.degeneracy) == (1, 2)
    assert symmetry_number(chair.mol) == (6, 'D3d')
    assert symmetry_number(twist.mol) == (4, 'D2')


def test_relaxed_scan_gives_the_butadiene_barrier():
    """s-trans -> 60 degrees on HF/STO-3G: the top of the relaxed scan sits
    near the perpendicular structure, above both minima of the search. Every
    point converged: on geomeTRIC's internal coordinates each restrained
    relaxation takes 12-14 cycles; the Cartesian walk under the default
    restraint collapses its trust radius from 116 degrees on, and an
    unconverged point is not a scan point."""
    pytest.importorskip('geometric')
    mol = molecule(BUTADIENE)
    tor = rotatable_bonds(mol)[0]
    surface = MeanFieldSurface(mol, scf_factory)
    points, status = torsion_scan(surface, mol, tor, [150, 120, 90, 60],
                                  engine='geometric', verbose=False)
    assert status == 'complete'
    assert all(p.converged and p.halvings == 0 for p in points)
    height, top = barrier(points)
    assert top.target == 90.0
    assert max(abs(p.phi - p.target) for p in points) < 0.5
    # the s-trans minimum is 0.08 eV below the s-cis one (test_conformers);
    # the barrier from the scan's own lowest point
    assert 0.15 < height * HARTREE_TO_EV < 0.3
    assert all(p.restraint < 1e-4 for p in points)


def triplet_factory(mol):
    mf = scf.UHF(mol.copy().set(spin=2).build(False, False))
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def test_twisted_triplet_is_a_different_conformer():
    """Ethylene: S0 planar, T1 (UHF) twisted by 90 degrees about the C=C
    torsion -- the twisted-ICT signature, flagged per pair with its
    adiabatic gap."""
    eth = molecule(ETHYLENE)
    ground = search_conformers(MeanFieldSurface(eth, scf_factory),
                               verbose=False)
    tor = Torsion(2, 0, 1, 4, (0, 2, 3), 1)
    pairs, minima = excited_conformers(ground, MeanFieldSurface(
        eth, triplet_factory), torsions=[tor], nudge=0.05, verbose=False)
    (pair,) = pairs
    assert pair['tict'] is True
    assert abs(abs(pair['twist'][0]) - 90.0) < 2.0
    assert len(minima) == 1 and minima[0]['from'] == [0]
    assert 1.0 < pair['gap'] * HARTREE_TO_EV < 3.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
