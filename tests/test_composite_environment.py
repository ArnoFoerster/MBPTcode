"""Permanent charges and polarizable sites through one `Environment`.

The two channels of Li, D'Avino, Duchemin, Beljonne and Blase, Phys. Rev. B 97,
035108 (2018): the permanent multipoles act on the ground state (Sec. II A) and
the induced ones on the charged excitation (Sec. II D), and neither may reach
the other's entry point. Gated here at the level of an energy, which is the
only place the separation can actually be seen -- `PointCharges.aux_kernel`
returning None says nothing about whether the reaction field stayed out of the
SCF or the crystal field out of Delta W.

THE COMPOSITE IS NOT TWO ATTACHMENTS. Eq. (18) of Duchemin, Guido, Jacquemin
and Blase, Chem. Sci. 9, 4430 (2018) is a self-element of the TOTAL
Delta W = W[v + vtilde] - W[v], and Delta W is not additive in vtilde: the
Dyson inversion mixes the members through the solute's own chi0. Two site
groups whose kernels are summed before the route sees them therefore give a
different shift from two shifts added, measurably so, and the composite is what
makes the first one available.

THE CRYSTAL FIELD SHIFTS THE LEVELS AND NOT THE GAP ONLY WHERE THE POTENTIAL IS
UNIFORM over the QM region, which is the condition PRB's Tables II/III rest on
("to first approximation, the superposition of the quadrupolar fields of MM
molecules as a uniform electrostatic potential acting on the QM region ...
implies a rigid shift"). A neutral shell pair delivers that; four charges
around a molecule with a diffuse virtual do not, and the counter-case is gated
too, because it is what tells a reader which property of the environment the
result depends on.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, qmmm, scf

from src.Base.composite_environment import CompositeEnvironment
from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import (Environment, NoEnvironment, PointCharges,
                                  attached_environment)
from src.Base.polarizable_sites import PolarizableSites
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

WATER = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
BASIS = 'cc-pvdz'
#: Bohr and Bohr^3, outside the QM density (7.2 Bohr from the nearest nucleus)
SITES = np.array([[0.0, 0.0, 9.0], [5.0, 2.0, -6.0], [-5.0, -6.0, 2.0]])
ALPHAS = np.array([5.0, 3.0, 8.0])
#: A neutral pair of octahedral shells, +q at 12 Bohr and -q at 24. Its
#: interior potential is the constant 6q(1/R_in - 1/R_out) up to the l = 4 term
#: of the discrete shell, which is the Madelung-like crystal field of PRB
#: Sec. II A rather than the field of any single multipole.
SHELL_IN, SHELL_OUT, SHELL_Q = 12.0, 24.0, 0.044
#: Four charges at 7 Bohr: neutral, zero potential AND zero field at the
#: centre, so the whole of what the molecule sees is the quadratic growth.
QUADRUPOLE_R, QUADRUPOLE_Q = 7.0, 0.4


def octahedron(radius):
    return radius * np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                              [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])


def shell_charges(q):
    """+q on an octahedron at SHELL_IN, -q on one at SHELL_OUT."""
    coords = np.vstack([octahedron(SHELL_IN), octahedron(SHELL_OUT)])
    return PointCharges(coords, np.concatenate([np.full(6, q), np.full(6, -q)]),
                        unit='Bohr')


def shell_potential(q):
    """The constant interior potential of that pair, in eV."""
    return 6.0 * q * (1.0 / SHELL_IN - 1.0 / SHELL_OUT) * HARTREE_TO_EV


def quadrupole_charges(q):
    """+q on the x axis, -q on the y axis, both at QUADRUPOLE_R."""
    r = QUADRUPOLE_R
    coords = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]])
    return PointCharges(coords, np.array([q, q, -q, -q]), unit='Bohr')


def rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    return mol, rhf(mol), mol.nelectron // 2


def frontier(mf, nocc, environment=None):
    """(HOMO, LUMO) quasiparticle energies in eV, in `environment`."""
    with attached_environment(mf, environment):
        out = calc_qp_energy(mf, mode='casida', state=[nocc - 1, nocc])
    return out[nocc - 1]['GW'], out[nocc]['GW']


# ------------------------------------------------------------------ contract
def test_the_composite_satisfies_the_protocol(water):
    """And is differentiable only if every member is: a chain asks once,
    before paying for a reverse pass, and a member that cannot differentiate
    its own entry makes the whole force unavailable rather than partial."""
    mol, _, _ = water
    sites = PolarizableSites(SITES, ALPHAS, unit='Bohr', mol=mol)
    env = CompositeEnvironment(quadrupole_charges(QUADRUPOLE_Q), sites)
    assert isinstance(env, Environment)
    assert env.differentiable is True
    assert env.for_geometry(mol) is env

    class Undifferentiable(NoEnvironment):
        differentiable = False

    assert CompositeEnvironment(Undifferentiable(), sites).differentiable is False


def test_the_charges_reach_the_ground_state_and_nothing_else(water):
    """A fixed charge has no response: it enters the mean field through pyscf's
    QM/MM wrapper and contributes nothing to vtilde, so the composite's kernel
    IS the sites' kernel and the Eq. (18) shift of the charges alone does not
    exist. This is PRB Sec. II D, 'fixed charges in the MM part ... do not
    contribute to the reaction field matrix'."""
    mol, _, _ = water
    sites = PolarizableSites(SITES, ALPHAS, unit='Bohr', mol=mol)
    charges = quadrupole_charges(QUADRUPOLE_Q)
    env = CompositeEnvironment(charges, sites)
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=BASIS + '-ri')

    assert charges.aux_kernel(auxmol) is None
    assert np.array_equal(env.aux_kernel(auxmol), sites.aux_kernel(auxmol))
    assert env.static_self_energy(None, mol) is None

    mf = env.mean_field(mol, rhf)
    assert isinstance(mf, qmmm.itrf.QMMM)
    assert mf.e_tot != pytest.approx(rhf(mol).e_tot, abs=1e-6)
    with attached_environment(mf, charges):
        assert environment_quasiparticle_shift(mf) is None


def test_one_shift_from_the_summed_kernel_is_not_two_shifts_added(water):
    """Delta W runs through a Dyson inversion, so it is not additive in vtilde:
    the members' kernels have to be summed BEFORE the shift is formed. Two
    single sites at +-8 Bohr, each alpha = 60 Bohr^3, give HOMO shifts summing
    to 383.38 meV while the composite's single shift is 382.40 -- 0.26 % apart,
    six times the 0.04 % at alpha = 10, i.e. one order higher in vtilde. That
    difference is the only thing the composite buys over two attachments, and
    it is what makes summing two shifts wrong rather than merely redundant."""
    mol, mf, nocc = water
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=BASIS + '-ri')
    for alpha, expected in ((10.0, 4e-4), (60.0, 26e-4)):
        a = PolarizableSites(np.array([[0.0, 0.0, 8.0]]), np.array([alpha]),
                             unit='Bohr', mol=mol)
        b = PolarizableSites(np.array([[0.0, 0.0, -8.0]]), np.array([alpha]),
                             unit='Bohr', mol=mol)
        with attached_environment(mf, a):
            shift_a = environment_quasiparticle_shift(mf)
        with attached_environment(mf, b):
            shift_b = environment_quasiparticle_shift(mf)
        with attached_environment(mf, CompositeEnvironment(a, b)):
            together = environment_quasiparticle_shift(mf)
        added = shift_a + shift_b
        assert np.array_equal(CompositeEnvironment(a, b).aux_kernel(auxmol),
                              a.aux_kernel(auxmol) + b.aux_kernel(auxmol))
        relative = abs(together[nocc - 1] / added[nocc - 1] - 1.0)
        assert relative == pytest.approx(expected, rel=0.25)
        assert abs(together[nocc - 1]) < abs(added[nocc - 1])


def test_the_charges_do_not_move_the_sites_shift_to_first_order(water):
    """The charges reach Eq. (18) only through the orbitals they polarize, and
    for a quadrupolar field at 7 Bohr that is 7e-4 of the shift: sites alone
    give HOMO +45.19 / LUMO -40.37 meV, the composite +45.22 / -40.38."""
    mol, mf, nocc = water
    sites = PolarizableSites(SITES, ALPHAS, unit='Bohr', mol=mol)
    charges = quadrupole_charges(QUADRUPOLE_Q)
    with attached_environment(mf, sites):
        alone = environment_quasiparticle_shift(mf)
    polarized = charges.mean_field(mol, rhf)
    with attached_environment(polarized, CompositeEnvironment(charges, sites)):
        together = environment_quasiparticle_shift(polarized)
    assert alone[nocc - 1] * HARTREE_TO_EV == pytest.approx(0.0452, abs=1e-3)
    assert alone[nocc] * HARTREE_TO_EV == pytest.approx(-0.0404, abs=1e-3)
    for p in (nocc - 1, nocc):
        assert abs(together[p] / alone[p] - 1.0) < 0.01


# -------------------------------------------------------------- crystal field
def test_a_uniform_crystal_field_shifts_the_levels_and_not_the_gap(water):
    """PRB Tables II/III: the crystal field moves HOMO and LUMO by 0.16-0.62 eV
    in the same direction and the gap by at most 0.03, and the direction flips
    with the sign of the multipole (PEN against PFP). A neutral pair of
    octahedral shells puts a constant 0.299 eV inside itself, and every
    quasiparticle level moves by exactly that: HOMO and LUMO both -299.3 meV,
    the gap -0.0. Reversing the shells reverses both. The Kohn-Sham spectrum
    moves rigidly too, over its whole range, which is what says the potential
    is uniform rather than the two frontier orbitals coincidentally agreeing."""
    mol, mf, nocc = water
    gas = frontier(mf, nocc)
    for sign in (1.0, -1.0):
        charges = shell_charges(sign * SHELL_Q)
        polarized = charges.mean_field(mol, rhf)
        homo, lumo = frontier(polarized, nocc)
        d_homo, d_lumo = homo - gas[0], lumo - gas[1]
        assert abs(d_homo + sign * shell_potential(SHELL_Q)) < 5e-3
        assert np.sign(d_homo) == np.sign(d_lumo) == -sign
        assert abs(d_homo) > 0.25 and abs(d_lumo) > 0.25
        assert abs(d_lumo - d_homo) < 0.1 * abs(d_homo)
        levels = (polarized.mo_energy - mf.mo_energy) * HARTREE_TO_EV
        assert np.abs(levels + sign * shell_potential(SHELL_Q)).max() < 1e-3


def test_a_four_charge_quadrupole_is_not_a_uniform_potential(water):
    """The counter-case, and the reason the gate above uses shells. Four
    charges at 7 Bohr put zero potential and zero field at the centre, so a
    level feels only the quadratic growth of the potential, weighted by how far
    the orbital reaches: <r^2> is 2.17 Bohr^2 for water's lone-pair HOMO and
    11.99 for its diffuse LUMO, and the first-order <p|V|p> is -75.8 meV
    against +232.6. The quasiparticle levels follow -- HOMO -18.7, LUMO +235.9,
    so the gap moves by +254.6 meV, thirteen times the HOMO shift. Neutrality
    is not the condition PRB's rigid shift rests on; uniformity over the QM
    region is."""
    mol, mf, nocc = water
    charges = quadrupole_charges(QUADRUPOLE_Q)
    fake = gto.fakemol_for_charges(charges.coords_bohr)
    # the electron carries charge -1, so the potential of the charges is -sum q/r
    v_ao = np.einsum('mnk,k->mn', pyscf_df.incore.aux_e2(
        mol, fake, intor='int3c2e', aosym='s1'), -charges.charges)
    v_mo = np.einsum('mp,mn,np->p', mf.mo_coeff, v_ao, mf.mo_coeff)
    r2 = np.einsum('mp,mn,np->p', mf.mo_coeff, mol.intor('int1e_r2'), mf.mo_coeff)
    assert r2[nocc] / r2[nocc - 1] > 4.0
    assert v_mo[nocc - 1] < 0.0 < v_mo[nocc]
    assert abs(v_mo[nocc]) > 2.0 * abs(v_mo[nocc - 1])

    gas = frontier(mf, nocc)
    polarized = charges.mean_field(mol, rhf)
    homo, lumo = frontier(polarized, nocc)
    d_homo, d_lumo = homo - gas[0], lumo - gas[1]
    assert abs(d_lumo) > 0.2 and abs(d_homo) < 0.05
    assert abs(d_lumo - d_homo) > 5.0 * abs(d_homo)
