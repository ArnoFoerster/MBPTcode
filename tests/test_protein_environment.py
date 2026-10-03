"""A classical environment from a charged structure file, and the diabatic
gradient inside it.

A synthetic PQR: two ethylenes (QM, one fragment each), a propene whose
methyl group is excluded from the QM region (a covalent cut, capped with a
link atom), a magnesium ion with a water at coordination distance (a bond
that must NOT be cut), and TIP3P-charged waters around them.

WHAT EACH GATE IS FOR:

- the reader takes elements from atom names without mistaking a C-alpha for
  calcium, and the cut is capped on the bond at the X-H length;
- the boundary atom's charge is moved onto its neighbours, so the total
  charge is conserved exactly, and nothing is cut at the magnesium;
- environment residues enter whole, so a water is never split, and a
  polarizable site too close to the QM region is dropped and reported;
- in the composite environment the builder returns (fixed charges in the
  mean field, polarizable sites screening the interaction), every diabatic
  element's analytic gradient equals the relocalized finite difference. This
  is the finite-difference gate for polarizable sites on the diabatic route.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import scf

from src.Base.composite_environment import CompositeEnvironment
from src.Base.protein_environment import (LINK_BOND_LENGTH, embed, read_pqr)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.fragment_diabatic import DiabaticGradient
from src.properties.diabatic import diabatic_derivative

ETHYLENE = [('C1', 0, 0, 0.667), ('C2', 0, 0, -0.667),
            ('H1', 0, 0.923, 1.238), ('H2', 0, -0.923, 1.238),
            ('H3', 0, 0.923, -1.238), ('H4', 0, -0.923, -1.238)]
TIP3P = (-0.834, 0.417)


def water(o, resseq, chain='W'):
    o = np.asarray(o, float)
    return [('OW', 'HOH', chain, resseq, o, TIP3P[0], 1.77),
            ('HW1', 'HOH', chain, resseq, o + [0.757, 0.586, 0], TIP3P[1], 0.0),
            ('HW2', 'HOH', chain, resseq, o + [-0.757, 0.586, 0], TIP3P[1], 0.0)]


def write_pqr(path, records):
    with open(path, 'w') as fh:
        for k, (name, res, chain, seq, xyz, q, r) in enumerate(records, 1):
            fh.write(f'ATOM {k:6d} {name:>4s} {res:>3s} {chain} {seq:4d} '
                     f'{xyz[0]:9.4f} {xyz[1]:9.4f} {xyz[2]:9.4f} '
                     f'{q:8.4f} {r:7.4f}\n')


def dimer_records(offset=(4.0, 0.3, 0.2)):
    recs = [(n, 'ETH', 'A', 1, np.array([x, y, z]), 0.0, 1.7)
            for n, x, y, z in ETHYLENE]
    recs += [(n, 'ETH', 'A', 2, np.array([x, y, z]) + offset, 0.0, 1.7)
             for n, x, y, z in ETHYLENE]
    return recs


@pytest.fixture
def synthetic(tmp_path):
    recs = dimer_records()
    # propene, C3 (methyl) excluded from the QM region
    base = np.array([-6.0, 0.0, 0.0])
    prp = [('C1', base + [0, 0, 0], -0.40), ('C2', base + [1.33, 0, 0], -0.10),
           ('C3', base + [2.10, 1.25, 0], -0.30),
           ('H1', base + [-0.55, 0.93, 0], 0.15),
           ('H2', base + [-0.55, -0.93, 0], 0.15),
           ('H3', base + [1.88, -0.93, 0], 0.10),
           ('H4', base + [3.17, 1.0, 0], 0.13),
           ('H5', base + [1.85, 1.85, 0.88], 0.13),
           ('H6', base + [1.85, 1.85, -0.88], 0.14)]
    recs += [(n, 'PRP', 'A', 3, x, q, 1.7) for n, x, q in prp]
    mg = np.array([2.0, -6.0, 0.0])
    recs += [('MG', 'MG', 'A', 4, mg, 2.0, 1.2)]
    recs += water(mg + [2.1, 0, 0], 10)
    recs += water([2.0, 6.0, 0.0], 11)
    recs += water([2.0, 0.3, 4.5], 12)
    recs += water([30.0, 0.0, 0.0], 13)            # beyond the cutoff
    recs += [('CA', 'ALA', 'P', 20, np.array([-2.0, 5.0, 0.0]), 0.03, 1.9)]
    path = tmp_path / 'synthetic.pqr'
    write_pqr(path, recs)
    return path


def test_reader_and_elements(synthetic):
    atoms = read_pqr(synthetic)
    by = {(a.resname, a.name): a for a in atoms}
    assert by[('MG', 'MG')].element == 'Mg'
    assert by[('ALA', 'CA')].element == 'C'
    assert by[('HOH', 'OW')].element == 'O'


def test_cut_is_capped_and_charge_is_conserved(synthetic):
    atoms = read_pqr(synthetic)
    total = sum(a.charge for a in atoms)
    emb = embed(atoms, [('A', 1, 'ETH'), ('A', 2, 'ETH'), ('A', 3, 'PRP'),
                        ('A', 4, 'MG')], 'sto-3g', cutoff=50.0,
                exclude={('A', 3, 'PRP'): ['C3', 'H4', 'H5', 'H6']})
    rep = emb.report
    assert rep['n_links'] == 1
    assert rep['cut_bonds'] == [(('A', 3, 'PRP'), 'C2', 'C3')]
    # the link hydrogen sits on the C2-C3 bond at the C-H length
    c2 = next(a.xyz for a in atoms if a.resname == 'PRP' and a.name == 'C2')
    c3 = next(a.xyz for a in atoms if a.resname == 'PRP' and a.name == 'C3')
    link = emb.mol.atom_coords(unit='Angstrom')[emb.fragments[2][-1]]
    assert abs(np.linalg.norm(link - c2) - LINK_BOND_LENGTH['C']) < 1e-10
    u = (c3 - c2) / np.linalg.norm(c3 - c2)
    assert np.linalg.norm(np.cross(link - c2, u)) < 1e-10
    # nothing cut at the magnesium; charge conserved through the boundary
    assert all(b[0] != ('A', 4, 'MG') for b in rep['cut_bonds'])
    assert abs(rep['qm_charge_in_file'] + rep['environment_charge']
               - total) < 1e-12
    assert emb.mol.charge == round(rep['qm_charge_in_file'])
    assert [len(f) for f in emb.fragments] == [6, 6, 6, 1]


def test_cutoff_keeps_whole_residues_and_drops_close_sites(synthetic):
    atoms = read_pqr(synthetic)
    emb = embed(atoms, [('A', 1, 'ETH'), ('A', 2, 'ETH')], 'sto-3g',
                cutoff=7.0, polarizabilities={'O': 5.0, 'H': 1.5, 'C': 8.0,
                                              'Mg': 2.0})
    kept = emb.charges.charges
    # three charges per kept water, never one or two
    waters = [q for q in kept if abs(abs(q) - 0.417) < 1e-9
              or abs(q + 0.834) < 1e-9]
    assert len(waters) % 3 == 0
    assert isinstance(emb.environment, CompositeEnvironment)
    assert emb.report['closest_charge'] > 0.0
    for _, _, d in emb.report['dropped_sites']:
        assert d < 3.4


def factory(mol):
    mf = scf.RHF(mol).density_fit('cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-13, 1e-11, 300, 0
    return mf


def test_diabatic_gradient_in_charges_and_polarizable_sites(tmp_path):
    recs = dimer_records()
    for k, o in enumerate(([2.0, 5.5, 0.0], [2.0, -5.5, 0.5], [-4.0, 0.5, 0.0],
                           [9.0, 0.0, 0.5])):
        recs += water(o, 20 + k)
    path = tmp_path / 'dimer.pqr'
    write_pqr(path, recs)
    emb = embed(read_pqr(path), [('A', 1, 'ETH'), ('A', 2, 'ETH')], 'cc-pvdz',
                cutoff=10.0, polarizabilities={'O': 5.0, 'H': 1.5})
    assert emb.report['n_sites'] > 0 and not emb.report['dropped_sites']
    chain = ExcitedStateChain(emb.mol, factory, bse_tda=True,
                              auxbasis='cc-pvdz-ri',
                              environment=emb.environment)
    sites, ct = {0: 1, 1: 1}, {(0, 1): 1, (1, 0): 1}
    grad = DiabaticGradient(chain, emb.fragments, sites, ct, omega0=0.31)
    rng = np.random.default_rng(23)
    direction = rng.normal(size=(emb.mol.natm, 3))
    direction /= np.linalg.norm(direction)
    fd = [diabatic_derivative(chain, emb.fragments, sites, direction, ct,
                              omega0=0.31, step=h)[0]['a_eff']
          for h in (5e-4, 2.5e-4)]
    fd = (4.0 * fd[1] - fd[0]) / 3.0
    scale = np.abs(fd).max()
    n = len(grad.partition.labels)
    for a in range(n):
        for b in range(a, n):
            an = float((grad.element(a, b)[0] * direction).sum())
            assert abs(abs(an) - abs(fd[a, b])) < 1e-6 * scale, (a, b)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
