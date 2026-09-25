"""Calibrating a site set against the molecule it replaces.

A classical model of a molecule is consistent with the QM molecule only if it
carries that molecule's polarizability: Li, D'Avino, Duchemin, Beljonne and
Blase, J. Phys. Chem. Lett. 7, 2814 (2016), "both models feature molecular
linear response and take as input the molecular polarizability tensor computed
at the desired level of accuracy", and Phys. Rev. B 97, 035108 (2018),
Technical details, "the molecular polarizability tensors are computed at the
DFT level". That is the precondition for moving a molecule across the QM/MM
partition: its polarizability has to leave chi0_11 and arrive in chi*_22
unchanged, or the polarization energy cannot compensate the gas-phase gap.

The reference here is the code's OWN polarizability, not experiment. cc-pVDZ
has no diffuse functions and HF has no correlation, so water comes out at 5.0
Bohr^3 against the experimental 9.64; what the calibration has to be right
about is the QM number of the calculation the sites will stand in for, in its
own basis.

AND AT ITS OWN LEVEL OF RESPONSE, which is the sharper half of that sentence.
A relaxed HF density answers a finite field, so `finite_field_polarizability`
is the CPHF (= TDHF at zero frequency) response, exchange kernel included,
while the W of every GW route here is built from the Hartree-only chi -- the
direct RPA. The two differ by 39 % for Ar and 61 % for water in this basis,
and calibrating a site on the wrong one over-screens by exactly that ratio,
which the partition test measures as a broken compensation.

WHY THE SCALE IS A ROOT SOLVE. The sites see one another, so their coupled
response exceeds sum(alpha) -- by 22 % for the three atoms of water here.
Dividing the molecular polarizability among the sites overshoots by that much.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.Base.constants import BOHR_TO_ANGSTROM, POLARIZABILITY_SCF_TOL
from src.Base.polarizable_sites import (calibrated_site_alphas,
                                        collective_polarizability,
                                        finite_field_polarizability,
                                        isotropic_polarizability,
                                        rpa_polarizability,
                                        sites_for_molecule)
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver

WATER = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
BASIS = 'cc-pvdz'
#: Bohr between the two oxygens of the dimer: far enough that the monomers
#: barely perturb one another, which is the papers' own non-overlap assumption.
DIMER_SEPARATION = 9.0
#: the papers' spread over MM models and parametrizations, and the tightest
#: tolerance any comparison of a classical model to a QM number should use
MODEL_SPREAD = 0.10


def water_dimer(separation, basis=BASIS):
    """Two waters related by inversion through the midpoint, so the monomers are
    equivalent and their dipoles cancel."""
    half = 0.5 * separation * BOHR_TO_ANGSTROM
    atoms = [f'O {half} 0 0.117', f'H {half} 0.757 -0.468', f'H {half} -0.757 -0.468',
             f'O {-half} 0 -0.117', f'H {-half} -0.757 0.468', f'H {-half} 0.757 0.468']
    return gto.M(atom='; '.join(atoms), basis=basis, verbose=0)


@pytest.fixture(scope='module')
def monomer():
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    return mol, finite_field_polarizability(mol)


@pytest.fixture(scope='module')
def dimer():
    mol = water_dimer(DIMER_SEPARATION)
    return mol, finite_field_polarizability(mol)


def test_the_finite_field_tensor_is_a_polarizability(monomer):
    """Symmetric and positive definite, as a linear response must be, and
    diagonal in the C2v frame. Water/cc-pVDZ/HF: 3.040 (out of plane), 6.899
    (H-H), 5.076 (C2 axis), isotropic 5.005 Bohr^3 -- half the experimental
    9.64, which is what a basis with no diffuse functions gives."""
    _, alpha = monomer
    assert np.abs(alpha - alpha.T).max() < 1e-8
    assert np.linalg.eigvalsh(alpha).min() > 0.0
    assert np.abs(alpha - np.diag(np.diag(alpha))).max() < 1e-8
    assert np.trace(alpha) / 3.0 == pytest.approx(5.005, abs=0.01)


def test_the_two_levels_of_response_are_not_interchangeable(monomer):
    """`rpa_polarizability` is the static limit of the SAME Hartree-only Casida
    problem the screened interaction is built from, so it is gated against the
    solver that builds it -- `build_casida_matrices(lBSE=False)` -- rather than
    left as a second spelling of the kernel: the two agree to 2e-15 Bohr^3.

    Against it, the finite field's CPHF response is 1.6130 times larger for
    water and 1.3923 for argon, the exchange kernel of the singlet A and B
    matrices being what separates them. Both are 'the molecular polarizability
    computed at the desired level of accuracy'; only one of them is the level
    this code screens at, and the factor is far outside the 10 % a site model
    is ever expected to be right to."""
    mol, ff = monomer
    rpa = rpa_polarizability(mol)
    mf = scf.RHF(mol)
    mf.conv_tol = POLARIZABILITY_SCF_TOL
    mf.kernel()
    nocc = mol.nelectron // 2
    c = mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.full(mol, c), c.shape[1])
    A, B = LinearResponseSolver(mf.mo_energy,
                                eri_chemist=eri).build_casida_matrices(nocc)
    d = np.einsum('mp,xmn,nq->xpq', c[:, :nocc], mol.intor('int1e_r', comp=3),
                  c[:, nocc:]).reshape(3, -1)
    assert np.abs(rpa - 4.0 * d @ np.linalg.solve(A + B, d.T)).max() < 1e-12
    assert np.trace(rpa) / 3.0 == pytest.approx(3.1029, abs=0.001)
    assert np.trace(ff) / np.trace(rpa) == pytest.approx(1.6130, abs=0.001)

    argon = gto.M(atom='Ar 0 0 0', basis=BASIS, verbose=0)
    ratio = (np.trace(finite_field_polarizability(argon))
             / np.trace(rpa_polarizability(argon)))
    assert ratio == pytest.approx(1.3923, abs=0.001)
    assert ratio - 1.0 > 3.0 * MODEL_SPREAD


def test_one_site_carries_the_isotropic_average_and_no_anisotropy(monomer):
    """One site per heavy atom is one site for water, and an isolated isotropic
    site can only answer isotropically: it reproduces the trace exactly and the
    anisotropy not at all. The negative control for the test below."""
    mol, alpha = monomer
    coords, alphas = sites_for_molecule(mol, target=float(np.trace(alpha) / 3.0))
    assert coords.shape == (1, 3)
    assert np.abs(coords[0] - mol.atom_coords()[0]).max() < 1e-12
    tensor = collective_polarizability(coords, alphas)
    assert np.trace(tensor) / 3.0 == pytest.approx(np.trace(alpha) / 3.0, rel=1e-8)
    assert np.abs(tensor - np.eye(3) * np.trace(tensor) / 3.0).max() < 1e-12
    qm_spread = np.diag(alpha).max() - np.diag(alpha).min()
    assert qm_spread / (np.trace(alpha) / 3.0) > 0.7      # the QM tensor is not


def test_sites_on_the_atoms_carry_the_anisotropy_through_the_coupling(monomer):
    """Isotropic sites summed without coupling are isotropic, so every bit of a
    site model's anisotropy is the dipole-dipole interaction. Three sites on
    water's atoms at one common scale put the three principal components in the
    right ORDER and within 12 % of the QM tensor (3.065 vs 3.040, 7.463 vs
    6.899, 4.487 vs 5.076), having been calibrated on the trace alone."""
    mol, alpha = monomer
    coords = mol.atom_coords()
    alphas = calibrated_site_alphas(coords, float(np.trace(alpha) / 3.0))
    tensor = collective_polarizability(coords, alphas)
    assert np.trace(tensor) / 3.0 == pytest.approx(np.trace(alpha) / 3.0, rel=1e-8)
    assert np.array_equal(np.argsort(np.diag(tensor)), np.argsort(np.diag(alpha)))
    for model, qm in zip(np.diag(tensor), np.diag(alpha)):
        assert abs(model - qm) / qm < 1.5 * MODEL_SPREAD


def test_the_common_scale_is_a_root_solve_not_a_division(monomer):
    """sum(alpha) is not the model's polarizability. The three sites of water
    respond 22 % above their own sum, so handing each site a third of the
    molecular value overshoots by that much; the calibration solves for the
    scale instead and lands on the target."""
    mol, alpha = monomer
    coords = mol.atom_coords()
    target = float(np.trace(alpha) / 3.0)
    naive = isotropic_polarizability(coords, np.full(len(coords), target / len(coords)))
    assert naive / target > 1.15
    alphas = calibrated_site_alphas(coords, target)
    assert alphas.sum() / target == pytest.approx(1.0 / 1.22, rel=0.05)
    assert isotropic_polarizability(coords, alphas) == pytest.approx(target, rel=1e-8)


def test_the_partner_is_the_same_molecule_qm_or_mm(monomer, dimer):
    """The calibration T7 needs. Two waters 9 Bohr apart: with both QM the
    finite-field polarizability is 9.955 Bohr^3, and replacing each by its own
    calibrated site gives 10.011 (one site per monomer, +0.6 %) or 9.967 (three
    sites per monomer, +0.1 %) -- inside the papers' 10 % model spread, so a
    monomer carries the same response whichever side of the partition it sits
    on. The QM pair is 0.6 % BELOW twice the monomer, which is the overlap the
    classical model cannot have and the reason the separation is 9 Bohr."""
    mono, alpha_mono = monomer
    mol, alpha_dimer = dimer
    iso_mono, iso_dimer = np.trace(alpha_mono) / 3.0, np.trace(alpha_dimer) / 3.0
    assert abs(iso_dimer / (2.0 * iso_mono) - 1.0) < 0.02

    heavy = mol.atom_coords()[mol.atom_charges() > 1]
    one_per_monomer = isotropic_polarizability(heavy, np.full(2, iso_mono))
    assert abs(one_per_monomer - iso_dimer) / iso_dimer < MODEL_SPREAD

    per_atom = calibrated_site_alphas(mono.atom_coords(), iso_mono)
    three_per_monomer = isotropic_polarizability(
        mol.atom_coords(), np.concatenate([per_atom, per_atom]))
    assert abs(three_per_monomer - iso_dimer) / iso_dimer < MODEL_SPREAD

    # the mutual coupling of the two sites is real but tiny at this separation:
    # it is the QM overlap, not the classical interaction, that moves the pair
    uncoupled = 2.0 * iso_mono
    assert abs(one_per_monomer - uncoupled) / uncoupled < 1e-3
