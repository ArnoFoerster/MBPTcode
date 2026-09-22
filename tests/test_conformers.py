"""Gates for the conformer search in src/properties/conformers.py.

Two tiers. The cheap one is pure geometry and graph theory -- rotatable-bond
detection, the rigid torsion rotation, the automorphism group and the
symmetry-aware RMSD -- and costs no electronic structure at all. The expensive
one is a real search on a mean-field surface: 1,3-butadiene at HF/STO-3G has two
torsional minima about its central C-C bond, s-trans and s-cis, and the search
has to find exactly those two, order them correctly, merge the two starts that
fall into the same one, and populate them by the Boltzmann factor of the gap it
measured itself.

Run the cheap tier alone with
    -k "not search and not weight and not average"
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.data import radii

from src.Base.constants import (BOHR_TO_ANGSTROM, BOLTZMANN_HARTREE_PER_KELVIN,
                                CONFORMER_RMSD_TOL,
                                CONFORMER_SINGLE_BOND_RATIO,
                                CONFORMER_TEMPERATURE, HARTREE_TO_EV)
from src.properties.conformers import (Conformer, Torsion, bond_in_ring,
                                       boltzmann_weights, conformer_average,
                                       conformer_rmsd, deduplicate, dihedral,
                                       graph_automorphisms, heavy_atoms,
                                       kabsch_rmsd, molecular_graph,
                                       rotatable_bonds, rotate_torsion,
                                       search_conformers, split_bond,
                                       torsion_starts, torsion_values)
from src.properties.optimize import MeanFieldSurface

# s-trans 1,3-butadiene, idealized and planar: C=C 1.34, C-C 1.46, CCC 122.8 deg.
# Atoms 0-3 are the carbon chain, so the one rotatable bond is 1-2.
BUTADIENE = ('C -0.726  1.126  0.000; C  0.000  0.000  0.000; '
             'C  1.460  0.000  0.000; C  2.186 -1.126  0.000; '
             'H -0.232  2.087  0.000; H -1.805  1.073  0.000; '
             'H -0.517 -0.948  0.000; H  1.692 -2.087  0.000; '
             'H  3.265 -1.073  0.000; H  1.977  0.948  0.000')

# Ideal cyclohexane chair, C-C 1.543: six single C-C bonds, all of them in the
# ring, which is the only reason none of them is rotatable.
CYCLOHEXANE = (
    'C  1.4600  0.0000  0.2500; C  0.7300  1.2644 -0.2500; '
    'C -0.7300  1.2644  0.2500; C -1.4600  0.0000 -0.2500; '
    'C -0.7300 -1.2644  0.2500; C  0.7300 -1.2644 -0.2500; '
    'H  1.4600  0.0000  1.3400; H  2.4888  0.0000 -0.1101; '
    'H  0.7300  1.2644 -1.3400; H  1.2444  2.1554  0.1101; '
    'H -0.7300  1.2644  1.3400; H -1.2444  2.1554 -0.1101; '
    'H -1.4600  0.0000 -1.3400; H -2.4888  0.0000  0.1101; '
    'H -0.7300 -1.2644  1.3400; H -1.2444 -2.1554 -0.1101; '
    'H  0.7300 -1.2644 -1.3400; H  1.2444 -2.1554  0.1101')

BASIS = 'sto-3g'


def scf_factory(mol):
    """RHF with the orbital gradient driven far below the force threshold."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


@pytest.fixture(scope='module')
def butadiene():
    return gto.M(atom=BUTADIENE, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def cyclohexane():
    return gto.M(atom=CYCLOHEXANE, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def butadiene_search(butadiene):
    """The full search on the HF/STO-3G ground state, run once."""
    surface = MeanFieldSurface(butadiene, scf_factory)
    return surface, search_conformers(surface, engine='cartesian',
                                      max_cycle=60, verbose=False)


def test_rotatable_bond_is_the_single_bond_only(butadiene):
    """Butadiene has exactly one rotatable bond and it is the central C-C."""
    tors = rotatable_bonds(butadiene)
    assert len(tors) == 1
    assert (tors[0].j, tors[0].k) == (1, 2)
    # the dihedral is defined by heavy reference atoms, not hydrogens
    assert (tors[0].i, tors[0].l) == (0, 3)
    # the terminal C=C bonds are in the graph and are rejected as too short
    adj = molecular_graph(butadiene)
    crd = np.asarray(butadiene.atom_coords())
    for a, b in ((0, 1), (2, 3)):
        assert b in adj[a]
        r_sum = 2.0 * radii.COVALENT[6]
        assert np.linalg.norm(crd[a] - crd[b]) < CONFORMER_SINGLE_BOND_RATIO * r_sum
    assert torsion_values(butadiene, tors) == pytest.approx([-180.0], abs=1e-6)


def test_ring_bonds_are_skipped(cyclohexane):
    """Every cyclohexane C-C passes the other tests and is skipped for the ring."""
    adj = molecular_graph(cyclohexane)
    crd = np.asarray(cyclohexane.atom_coords())
    z = np.asarray(cyclohexane.atom_charges())
    ring = [(a, b) for a in range(6) for b in sorted(adj[a]) if a < b < 6]
    assert len(ring) == 6
    for a, b in ring:
        # long enough to be a single bond ...
        assert (np.linalg.norm(crd[a] - crd[b])
                > CONFORMER_SINGLE_BOND_RATIO * 2.0 * radii.COVALENT[6])
        # ... with a heavy atom on each side ...
        assert [n for n in adj[a] if n != b and z[n] > 1]
        assert [n for n in adj[b] if n != a and z[n] > 1]
        # ... so ring membership is the only thing that disqualifies it.
        assert bond_in_ring(adj, a, b)
    assert rotatable_bonds(cyclohexane, adj) == []


def test_rotation_is_rigid_and_reaches_the_requested_dihedral(butadiene):
    """A torsion start changes the dihedral by exactly delta and nothing else."""
    tors = rotatable_bonds(butadiene)[0]
    crd = np.asarray(butadiene.atom_coords())
    d0 = np.linalg.norm(crd[:, None] - crd[None, :], axis=-1)
    phi0 = dihedral(crd, tors.i, tors.j, tors.k, tors.l)
    for delta in (60.0, 120.0, -75.0):
        new = rotate_torsion(crd, tors, delta)
        phi = dihedral(new, tors.i, tors.j, tors.k, tors.l)
        assert (phi - phi0 - delta + 180.0) % 360.0 - 180.0 == pytest.approx(
            0.0, abs=1e-9)
        # the moving fragment is rigid: every bonded distance is unchanged
        d1 = np.linalg.norm(new[:, None] - new[None, :], axis=-1)
        assert np.abs(d1 - d0)[d0 < 3.0].max() < 1e-12
    # carrying the OTHER fragment is the opposite rotation and the same result
    side_j, side_k = split_bond(molecular_graph(butadiene), tors.j, tors.k)
    assert side_j == {0, 1, 4, 5, 6} and side_k == {2, 3, 7, 8, 9}
    other = Torsion(tors.i, tors.j, tors.k, tors.l, tuple(sorted(side_k)), -1)
    phi = dihedral(rotate_torsion(crd, other, 60.0), *(tors.i, tors.j, tors.k,
                                                       tors.l))
    assert (phi - phi0 - 60.0 + 180.0) % 360.0 - 180.0 == pytest.approx(
        0.0, abs=1e-9)


def test_torsion_grid_counts_and_cap(butadiene):
    """n_grid ** n_torsions starts, truncated to max_starts, input always first."""
    tors = rotatable_bonds(butadiene)
    starts = torsion_starts(butadiene, tors, n_grid=3, max_starts=64)
    assert len(starts) == 3
    assert starts[0][1] == (0.0,)
    assert np.allclose(starts[0][0], butadiene.atom_coords())
    # the same bond listed twice exercises the product, not the chemistry
    pair = torsion_starts(butadiene, tors + tors, n_grid=3, max_starts=64)
    assert len(pair) == 9
    assert {o for _, o in pair} == {(a, b) for a in (0.0, 120.0, 240.0)
                                    for b in (0.0, 120.0, 240.0)}
    capped = torsion_starts(butadiene, tors + tors, n_grid=3, max_starts=4)
    assert len(capped) == 4
    assert capped[0][1] == (0.0, 0.0)


def test_graph_automorphisms(butadiene, cyclohexane):
    """The heavy-atom symmetry a structural comparison has to quotient out."""
    perms = graph_automorphisms(butadiene)
    assert [p.tolist() for p in perms] == [[0, 1, 2, 3], [3, 2, 1, 0]]
    # the chair's carbon skeleton is a six-cycle: six rotations, six reflections
    assert len(graph_automorphisms(cyclohexane)) == 12


def test_mirror_images_are_one_conformer(butadiene):
    """gauche+ and gauche- are enantiomers and must not be counted twice."""
    tors = rotatable_bonds(butadiene)[0]
    hv = heavy_atoms(butadiene)
    crd = np.asarray(butadiene.atom_coords())
    plus = rotate_torsion(crd, tors, 120.0)[hv] * BOHR_TO_ANGSTROM
    minus = rotate_torsion(crd, tors, -120.0)[hv] * BOHR_TO_ANGSTROM
    # no proper rotation superposes them, which is what makes this a real test
    assert kabsch_rmsd(plus, minus) > CONFORMER_RMSD_TOL
    assert conformer_rmsd(plus, minus) < 1e-8


def test_symmetry_equivalent_labelling_is_one_conformer(butadiene):
    """A structure and its relabelling under a graph automorphism are one."""
    hv = heavy_atoms(butadiene)
    ref = np.asarray(butadiene.atom_coords())[hv] * BOHR_TO_ANGSTROM
    # distorted so that no rigid motion maps the skeleton onto its relabelling
    distorted = ref + np.random.default_rng(4).normal(scale=0.3, size=ref.shape)
    swapped = distorted[[3, 2, 1, 0]]
    assert conformer_rmsd(distorted, swapped) > CONFORMER_RMSD_TOL
    assert conformer_rmsd(distorted, swapped,
                          graph_automorphisms(butadiene)) < 1e-8


def test_deduplicate_merges_and_keeps_the_lowest(butadiene):
    """Duplicates collapse onto their lowest member; a real gap survives."""
    tors = rotatable_bonds(butadiene)[0]
    crd = np.asarray(butadiene.atom_coords())

    def at(coords):
        m = butadiene.copy()
        m.set_geom_(coords, unit='Bohr')
        m.build(False, False)
        return m

    # the enantiomeric pair differs by the residual of two loose relaxations
    records = [{'mol': at(rotate_torsion(crd, tors, 120.0)), 'energy': -1.0},
               {'mol': at(rotate_torsion(crd, tors, -120.0)), 'energy': -1.00005},
               {'mol': at(crd), 'energy': -1.05}]
    groups = deduplicate(records, butadiene)
    assert len(groups) == 2
    assert [rep['energy'] for rep, _ in groups] == [-1.05, -1.00005]
    assert sorted(groups[1][1]) == [0, 1]
    # a pair that agrees geometrically but not in energy stays two minima
    apart = list(records)
    apart[1] = dict(apart[1], energy=-1.001)
    assert len(deduplicate(apart, butadiene)) == 3


def test_search_finds_both_butadiene_minima(butadiene_search):
    """s-trans and s-cis at HF/STO-3G, in that order, and nothing else."""
    _, confs = butadiene_search
    assert len(confs) == 2
    assert all(isinstance(c, Conformer) and c.converged for c in confs)
    assert confs[0].energy < confs[1].energy
    # s-trans is planar anti, s-cis planar syn
    assert abs(abs(float(confs[0].torsions[0])) - 180.0) < 5.0
    assert abs(float(confs[1].torsions[0])) < 20.0
    # the +120 and -120 starts fall into the same minimum and are merged
    assert confs[0].starts == [0]
    assert confs[1].starts == [1, 2]
    gap = (confs[1].energy - confs[0].energy) * HARTREE_TO_EV
    assert 0.02 < gap < 0.30           # HF/STO-3G puts it at 0.079 eV
    assert confs[0].energy == pytest.approx(-153.0203601, abs=1e-5)


def test_weights_are_the_boltzmann_factor_of_the_measured_gap(butadiene_search):
    """Populations sum to one and follow exp(-dE/kT) of the gap just measured."""
    _, confs = butadiene_search
    weights = np.array([c.weight for c in confs])
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert weights[0] > weights[1]
    kt = BOLTZMANN_HARTREE_PER_KELVIN * CONFORMER_TEMPERATURE
    gap = confs[1].energy - confs[0].energy
    assert weights[1] / weights[0] == pytest.approx(np.exp(-gap / kt), rel=1e-10)
    # and the standalone routine agrees with what the search reported
    assert boltzmann_weights([c.energy for c in confs]) == pytest.approx(weights)


def test_conformer_average_reruns_a_routine_on_refrozen_surfaces(butadiene_search):
    """The property hook: a routine over the ensemble, weighted by population."""
    surface, confs = butadiene_search
    out = conformer_average(confs, lambda c, s: s.total_energy(c.mol),
                            surface=surface)
    assert out['values'] == pytest.approx([c.energy for c in confs], abs=1e-8)
    assert out['average'] == pytest.approx(
        sum(c.weight * c.energy for c in confs), abs=1e-10)
    # and without a surface the routine sees the conformer alone
    torsion = conformer_average(confs, lambda c: abs(float(c.torsions[0])))
    assert torsion['average'] == pytest.approx(
        sum(c.weight * abs(float(c.torsions[0])) for c in confs), abs=1e-10)
