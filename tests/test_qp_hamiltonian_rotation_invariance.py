"""The quasiparticle Hamiltonian of a near-degenerate group does not depend on
how the group's orbitals rotate into each other.

Formaldehyde/cc-pVDZ (Hartree-Fock), planar, C-O stretched to the length
where the occupied B1 (pi) and B2 (n) orbitals 6 and 7 cross: an accidental
degeneracy (1.2e-10 Ha) of two orbitals of different symmetry and different
self-energies. There every rotation of the pair is an equally canonical
basis, so a surface that is a function of the geometry gives the same
excitation energies, and the same eigenvalues of the pair's quasiparticle
Hamiltonian, in all of them. The rotation tried mixes the pair by 0.3, as
the DMABN LUMO/LUMO+1 pair is mixed 3.45e-3 Bohr from its gas S1 minimum.

  - A chain frozen in each basis gives the same S1 and the same eigenvalues
    of the pair's H to well under 0.1 meV. Taking Sigma^c_pq at each
    orbital's own root on per-orbital poles, they moved by meV.
  - One chain frozen in the unrotated basis and evaluated on the rotated
    orbitals matches the pair as one group by its span and gives the same S1
    again; on the diagonal route the two orbitals are matched one by one,
    hold half their weight each, and the conventions are refused.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import HARTREE_TO_MEV
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_jkfit_differentiable_checked

#: Angstrom: the C-O length where orbitals 6 (B1) and 7 (B2) cross, at
#: R(C-H) = 1.10 Angstrom and H-C-H = 121.8 degrees
CROSSING_RCO = 1.421217189729214
RCH = 1.10
HCH = 121.8
PAIR = (6, 7)
WINDOW = list(range(4, 18))
BASIS = 'cc-pvdz'
#: Ha: how degenerate the pair is at CROSSING_RCO
DEGENERATE = 1e-9
#: the pair's mixing, sin^2 of the rotation angle
MIXING = 0.3
#: meV: what the S1 energy and the pair's eigenvalues may move by
INVARIANT_MEV = 1e-3


def formaldehyde():
    y = RCH * np.sin(np.deg2rad(HCH / 2))
    z = -RCH * np.cos(np.deg2rad(HCH / 2))
    return gto.M(atom=f'O 0 0 {CROSSING_RCO!r}; C 0 0 0; H 0 {y!r} {z!r}; '
                      f'H 0 {-y!r} {z!r}', basis=BASIS, verbose=0)


def rotated(mf, theta):
    """`mf` with the degenerate pair rotated by theta: as canonical as it was."""
    out = mf.copy()
    c = np.array(mf.mo_coeff, float)
    p, q = PAIR
    cp, cq = c[:, p].copy(), c[:, q].copy()
    c[:, p] = np.cos(theta) * cp + np.sin(theta) * cq
    c[:, q] = -np.sin(theta) * cp + np.cos(theta) * cq
    out.mo_coeff = c
    return out


@pytest.fixture(scope='module')
def crossing():
    warnings.simplefilter('ignore')
    mol = formaldehyde()
    mf = rhf_df_jkfit_differentiable_checked(mol)
    assert abs(mf.mo_energy[PAIR[1]] - mf.mo_energy[PAIR[0]]) < DEGENERATE
    return mol, mf


def chain(mol, mf, qp_hamiltonian=False):
    warnings.simplefilter('ignore')
    return ExcitedStateChain(
        mol, rhf_df_jkfit_differentiable_checked, mf=mf,
        numerics=g1_numerics(mol, Numerics(qp={'window': WINDOW, 'residue_route': 'sop', 'hamiltonian': qp_hamiltonian},
                                           bse={'solver': 'dense'})))


def pair_eigenvalues(ch, mol, mf):
    """S1 and the eigenvalues of the quasiparticle Hamiltonian on the pair."""
    fwd = ch.shared_forward(mol, mf)
    idx = list(PAIR)
    h = np.diag(fwd.eps_qp[idx])
    if fwd.qp_h is not None:
        h = h + fwd.qp_h.h[np.ix_(idx, idx)]
    return ch.excitation(mol, mf), np.linalg.eigvalsh(h)


def test_the_pair_eigenvalues_do_not_depend_on_its_basis(crossing):
    mol, mf = crossing
    theta = np.arcsin(np.sqrt(MIXING))
    om0, e0 = pair_eigenvalues(chain(mol, mf, qp_hamiltonian=True), mol, mf)
    turned = rotated(mf, theta)
    om1, e1 = pair_eigenvalues(chain(mol, turned, qp_hamiltonian=True), mol,
                               turned)
    assert abs(om1 - om0) * HARTREE_TO_MEV < INVARIANT_MEV, (om0, om1)
    assert np.abs(e1 - e0).max() * HARTREE_TO_MEV < INVARIANT_MEV, (e0, e1)


def test_a_chain_matches_the_pair_by_its_span(crossing):
    mol, mf = crossing
    theta = np.arcsin(np.sqrt(MIXING))
    ch = chain(mol, mf, qp_hamiltonian=True)
    om0 = ch.excitation(mol, mf)
    assert abs(ch.excitation(mol, rotated(mf, theta)) - om0) \
        * HARTREE_TO_MEV < INVARIANT_MEV


def test_the_diagonal_route_refuses_a_pair_mixed_half_and_half(crossing):
    mol, mf = crossing
    ch = chain(mol, mf)
    ch.excitation(mol, mf)
    with pytest.raises(Exception, match='left its group'):
        ch.excitation(mol, rotated(mf, np.pi / 4))
