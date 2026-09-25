"""The N+/-1 surface must be the derivative of its own energy.

`QuasiparticleSurface` exists so `optimize` can relax a cation or an anion on
E_0 -/+ eps^QP, which is what turns the 2024 paper's quasiparticle GRADIENTS
into its adiabatic ionization potentials. A surface is only usable for that if
its analytic gradient is the derivative of the energy the same object reports:
an optimizer walks on the gradient and reports the energy, so a mismatch
between them puts the minimum somewhere neither describes, and does it quietly.

This is deliberately checked against ITSELF rather than against the paper.
Agreement with Toelle sits at about 1 mHa/A across Tables I to III for reasons
not yet understood, and folding that in here would mean a genuine coding error
of the same size could never be told from the pre-existing residual.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import GroundState
from src.Base.environment import PointCharges, attach_environment
from src.Base.solvent_screening import (SolventScreening,
                                        attach_solvent_screening)
from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                               SCF_DIFFERENTIABLE_GRAD_TOL,
                               SCF_ENERGY_CONV_TOL,
                               SCF_ENERGY_GRAD_TOL,
                               XC_SHIFT_GRADIENT_TOL)
from src.gradients.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                                QuasiparticleSurface,
                                                kohn_sham_gradient_correction,
                                                mo_eri)
from src.gradients.quasi_boson_adjoint import (BSEqbAdjoint as BSEqb,
                                               QPqbAdjoint as QPqb)
from src.properties.optimize import MeanFieldSurface
from src.properties.surface import (FiniteDifferenceGradient,
                                    PotentialEnergySurface)

#: A two-point central difference has an O(h^2) truncation error, so comparing
#: at ONE step size cannot tell a correct gradient from one that is wrong by
#: less than that error. Measured here on H2/cc-pVDZ the difference falls by
#: exactly 4.00 per halving from 1e-4 at h = 0.02 to 3.9e-7 at h = 0.00125 --
#: no noise floor in sight -- so the pair below is Richardson-extrapolated and
#: the residual gated at the round-off level instead.
FD_STEPS = (5e-3, 2.5e-3)
FD_TOL = 1e-8
#: Successive halvings must divide the error by four. A constant error would
#: give one, and that is what a MISSING TERM looks like: it survives h -> 0.
FD_ORDER_TOL = 0.15


def richardson(surface, steps=FD_STEPS):
    """(gradient extrapolated to h -> 0, the ratio between the two errors).

    G(h) = G + c h^2, so (4 G(h/2) - G(h)) / 3 cancels the leading error and
    the ratio of the two residuals says whether the error really was O(h^2).
    """
    coarse, fine = (FiniteDifferenceGradient(surface, h=h).total_gradient()[0]
                    for h in steps)
    extrapolated = (4.0 * fine - coarse) / 3.0
    ratio = (np.max(np.abs(coarse - extrapolated))
             / max(np.max(np.abs(fine - extrapolated)), 1e-300))
    return extrapolated, ratio


def rhf(mol):
    """Tight enough to differentiate: the Lagrangian assumes the occupied-
    virtual Fock block vanishes, so a loose SCF biases the force rather than
    degrading it gracefully."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    return mf


def pbe0(mol):
    """A global hybrid: the exact-exchange fraction is what makes the
    Kohn-Sham repair and its gradient nonzero, and a pure functional would
    leave `exx_double_counting`'s two halves coinciding with the wrong ones."""
    mf = dft.RKS(mol, xc='pbe0')
    mf.grids.level = 5
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    return mf


def h2():
    return gto.M(atom='H 0 0 0; H 0 0 0.7413', basis='cc-pvdz', verbose=0)


def water():
    """Three atoms, so the gradient has directions the diatomics cannot test:
    a bond length alone never exercises the off-axis components."""
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis='sto-3g', verbose=0)


@pytest.mark.parametrize('charge_change', (-1, +1))
@pytest.mark.parametrize('screening', ('rpa', 'tda'))
def test_the_gradient_is_the_derivative_of_the_energy(charge_change, screening):
    surface = QuasiparticleSurface(h2(), rhf, charge_change=charge_change,
                                   screening=screening)
    g_analytic, e, _ = surface.total_gradient()
    g_fd, ratio = richardson(surface)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, (
        f'{surface.label()}: the finite-difference error fell by {ratio:.2f} '
        f'per halving, not 4 -- so it is not O(h^2) truncation, and a term '
        f'that survives h -> 0 is a term that is missing')
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL, (
        f'{surface.label()}: analytic {g_analytic[1, 2]:.10f} vs h -> 0 '
        f'{g_fd[1, 2]:.10f}')


def test_a_hartree_fock_reference_is_not_read_as_a_kohn_sham_one():
    """<p|Sigma_x - v_xc|p> is zero on Hartree-Fock analytically and round-off
    numerically, so the guard refusing to differentiate that shift has to ask
    its MAGNITUDE: asked as `delta.any()` it refuses the one reference this
    route can differentiate, and the excited-state gradient is then unreachable
    on every molecule.

    H2/sto-3g cannot see this. At two basis functions the shift cancels to
    exactly 0.0 and the guard stays quiet whichever way it is written, which is
    how the defect passed a suite that exercised this path. Water is the
    smallest case here whose round-off is real.
    """
    mol = water()
    surface = DenseBSESurface(mol, scf=rhf)
    shift = np.max(np.abs(np.asarray(surface.xc_shift(rhf(mol)))))
    assert 0.0 < shift < XC_SHIFT_GRADIENT_TOL, (
        f'this molecule no longer exercises the defect: the Hartree-Fock '
        f'shift is {shift:.3e} Ha, which is not round-off below the '
        f'{XC_SHIFT_GRADIENT_TOL:.0e} Ha threshold the guard asks against')
    g, e, info = surface.total_gradient(mol)
    assert np.all(np.isfinite(g)) and info['omega'] > 0.0


def test_a_shift_large_enough_to_carry_a_force_is_still_refused():
    """The other half: what the guard exists for must keep being refused. A
    tenth of a Hartree per orbital is a Kohn-Sham shift, and its nuclear
    derivative is not assembled on this route."""
    mol = water()
    mf = rhf(mol)
    b = BSEqb(mf, mo_eri(mf, mol), mol.nelectron // 2,
              delta=np.full(mol.nao, 0.1))
    with pytest.raises(NotImplementedError, match='Sigma_x - v_xc'):
        b.partials(0)


def test_the_excited_state_gradient_is_the_derivative_of_its_own_energy():
    """The excited surface an optimizer walks on, gated the way the
    quasiparticle surfaces above are: an optimizer reports the energy and steps
    on the gradient, so a mismatch puts the minimum where neither describes."""
    surface = DenseBSESurface(water(), scf=rhf)
    g_analytic, _, _ = surface.total_gradient()
    g_fd, ratio = richardson(surface)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, (
        f'the finite-difference error fell by {ratio:.2f} per halving, not 4')
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL, (
        f'analytic vs h -> 0: {np.max(np.abs(g_analytic - g_fd)):.3e}')


def test_the_gradient_survives_more_than_one_bond_direction():
    """A diatomic tests one component of one atom. Water tests nine."""
    surface = QuasiparticleSurface(water(), rhf, charge_change=-1)
    g_analytic, _, _ = surface.total_gradient()
    g_fd, ratio = richardson(surface)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, ratio
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL, (
        np.max(np.abs(g_analytic - g_fd)))


def test_the_gradient_sums_to_zero_over_atoms():
    """A translation cannot change the energy, so this is the cheapest honest
    check that no skeleton derivative is missing -- and unlike the finite
    difference it cannot be fooled by an error the energy shares."""
    for charge_change in (-1, +1):
        surface = QuasiparticleSurface(water(), rhf,
                                       charge_change=charge_change)
        g, _, _ = surface.total_gradient()
        assert np.max(np.abs(g.sum(axis=0))) < 1e-8, g.sum(axis=0)


def test_it_is_a_potential_energy_surface():
    """`optimize` takes anything with this shape, so the shape is the contract."""
    surface = QuasiparticleSurface(h2(), rhf)
    assert isinstance(surface, PotentialEnergySurface)
    assert 'G0W0' in surface.label()


def test_the_orbital_is_held_across_a_displacement():
    """The frozen convention: re-choosing the quasiparticle HOMO mid-walk would
    change WHICH ionization the surface describes, which is a discontinuity in
    the energy rather than a better answer."""
    surface = QuasiparticleSurface(h2(), rhf, charge_change=-1)
    chosen = surface.orbital
    stretched = gto.M(atom='H 0 0 0; H 0 0 0.9', basis='cc-pvdz', verbose=0)
    surface.total_energy(stretched)
    assert surface.orbital == chosen
    assert surface.refreeze(stretched).mol0 is stretched


def test_an_impossible_process_is_refused():
    for bad in (0, +2, -3):
        with pytest.raises(ValueError, match='charge_change'):
            QuasiparticleSurface(h2(), rhf, charge_change=bad)
    with pytest.raises(ValueError, match='screening'):
        QuasiparticleSurface(h2(), rhf, screening='gw')


def test_the_ground_state_surface_is_also_its_own_derivative():
    surface = DenseRPASurface(water(), rhf)
    g_analytic, _, _ = surface.total_gradient()
    g_fd, ratio = richardson(surface)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, ratio
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL
    assert isinstance(surface, PotentialEnergySurface)


@pytest.mark.parametrize('screening,neutral', (('rpa', DenseRPASurface),
                                               ('tda', MeanFieldSurface)))
def test_the_two_legs_of_an_ionization_share_one_ground_state(screening, neutral):
    """AT ONE GEOMETRY the difference of the two surfaces must be exactly the
    quasiparticle energy and nothing else.

    An adiabatic IP is a difference of two minima, so whatever E_0 sits inside
    E^{N-1} has to be the neutral surface too. The Tamm-Dancoff quasiparticle
    surface carries no correlation energy -- the convention Tables II and III
    are computed in -- so its partner is the bare mean field, and pairing it
    with E_HF + E_c instead leaves -E_c in the difference. That is +11 eV on
    benzene and it arrives wearing the units of an ionization potential, which
    is why this is checked rather than assumed: every gradient test here
    examines ONE surface, and this defect exists only between two.
    """
    mol = h2()
    cation = QuasiparticleSurface(mol, rhf, charge_change=-1,
                                  screening=screening)
    vertical = cation.total_energy(mol) - neutral(mol, rhf).total_energy(mol)
    _, _, diag = cation.total_gradient(mol)
    assert abs(vertical + diag['qp_energy_eV'] / HARTREE_TO_EV) < 1e-10, (
        f'{screening}: E^(N-1) - E^N is {vertical:.10f} Ha but the '
        f'quasiparticle energy is {diag["qp_energy_eV"] / HARTREE_TO_EV:.10f} '
        f'-- the two legs are not on the same ground state')


def test_the_excitation_is_reachable_on_a_kohn_sham_reference():
    """The letter's second starting point is BHLYP and its transition energies
    on it are single points, so an ENERGY must be reachable there. Two things
    make that safe, and both are asserted: E_0 is repaired to the plasmon
    formula's E_HF + E_c^dRPA rather than left as E_KS + E_c^dRPA, which would
    count the correlation in E_xc twice; and the GRADIENT still refuses,
    because the nuclear derivative of <Sigma_x - v_xc> is not assembled here.
    """
    def bhlyp(m):
        mf = dft.RKS(m)
        mf.xc = 'bhandhlyp'
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf

    mol = water()
    surface = DenseBSESurface(mol, scf=bhlyp)
    mf = surface.scf_factory(mol)
    shift = np.max(np.abs(np.asarray(surface.xc_shift(mf))))
    assert shift > XC_SHIFT_GRADIENT_TOL, (
        f'a Kohn-Sham reference must carry a real shift, not {shift:.3e} Ha')

    b, _, _, _, e0 = surface._solve(mol, mf)
    assert np.isfinite(e0) and b.Omega[0] > 0.0

    # What the repair is worth, asked as a starting-point SPREAD, since E_0 is
    # meant to be one functional of the density and not of the mean field that
    # produced it. Corrected, the two references land 11 mHa apart here, which
    # is the genuine difference between two densities; uncorrected they are
    # 338 mHa apart, and that gap is E_xc - E_x^exact rather than anything
    # physical.
    e0_hf = DenseBSESurface(mol, scf=rhf)._solve(mol, rhf(mol))[4]
    uncorrected = mf.e_tot + b.qp.qb.e_corr()
    assert abs(e0 - e0_hf) < 0.05, (
        f'E_0 moved {abs(e0 - e0_hf) * 1e3:.1f} mHa with the starting point')
    assert abs(uncorrected - e0_hf) > 0.3, (
        'this molecule no longer shows the double counting the correction '
        'removes, so the test has stopped testing it')
    with pytest.raises(NotImplementedError, match='Sigma_x - v_xc'):
        surface.total_gradient(mol, mf)


def test_repairing_e0_leaves_a_hartree_fock_reference_bit_identical():
    """The correction is what allows a Kohn-Sham reference through, and every
    cubic gate is a difference measured from this route's Hartree-Fock
    numbers. It is identically zero there, and `assert ==` rather than a
    tolerance is the point: a reference that moved by even a rounding would
    move all of those at once."""
    from src.gradients.isdf_derivatives import exx_double_counting

    mol = water()
    assert exx_double_counting(rhf(mol), mol) == 0.0


def test_the_energy_threshold_does_not_move_an_excitation_energy():
    """A mean field that will only be READ from does not need the convergence
    a mean field that will be DIFFERENTIATED needs, and on a Kohn-Sham
    reference the difference is most of the cost: the differentiable pair sits
    at or below the exchange-correlation grid's own noise floor, so the SCF
    spends its cycles chasing that noise.

    What must not change is the answer. Measured on acrolein/def2-SVP the two
    thresholds agree to 0.0012 meV over 915 roots while the loose one runs
    three times faster; this gates the same statement on a molecule small
    enough for a suite.
    """
    def ks(conv_tol, conv_tol_grad):
        def build(m):
            mf = dft.RKS(m)
            mf.xc = 'bhandhlyp'
            mf.grids.level = 5
            mf.conv_tol = conv_tol
            mf.conv_tol_grad = conv_tol_grad
            mf.max_cycle = 200
            mf.kernel()
            return mf
        return build

    mol = water()
    tight = DenseBSESurface(mol, scf=ks(SCF_DIFFERENTIABLE_CONV_TOL,
                                        SCF_DIFFERENTIABLE_GRAD_TOL))
    loose = DenseBSESurface(mol, scf=ks(SCF_ENERGY_CONV_TOL,
                                        SCF_ENERGY_GRAD_TOL))
    a = tight._solve(mol, tight.scf_factory(mol))[0].Omega
    b = loose._solve(mol, loose.scf_factory(mol))[0].Omega
    worst = np.max(np.abs(np.asarray(a) - np.asarray(b))) * HARTREE_TO_EV * 1e3
    assert worst < 0.05, (
        f'the energy threshold moved an excitation energy by {worst:.4f} meV; '
        f'it is meant to be invisible to everything but the wall clock')


@pytest.mark.parametrize('build', (
    lambda mol: DenseRPASurface(mol, pbe0),
    lambda mol: QuasiparticleSurface(mol, pbe0, charge_change=-1)))
def test_a_kohn_sham_reference_is_differentiated_and_not_refused(build):
    """The two ground-state surfaces on a Kohn-Sham starting point, gated the
    way their Hartree-Fock selves are: the force must be the derivative of the
    energy the same object reports.

    THREE TERMS MAKE IT ONE. E_0 is E_HF[rho], so the exact-exchange double
    counting rides along with its own orbital response and skeleton; the
    quasiparticle energy carries <p|Sigma_x - v_xc|p>, whose derivative enters
    weighted by the root's quasiparticle weight; and the Fock partial is folded
    on the REFERENCE'S Fock rather than the Hartree-Fock one
    (`kohn_sham_correlation_gradients`). Each was removed in turn and the
    residual here went from 5.4e-11 to 1.0e-02, 3.4e-10 to 5.0e-02 and 3.4e-10
    to 3.4e-02 respectively, and doubling the correction instead of dropping it
    is equally visible.

    H2 is enough to see all three -- the defects above are eight orders over the
    gate -- and it is what keeps this test at seconds. Water/sto-3g agrees to
    2.4e-09 on the ground state and 1.3e-07 on the quasiparticle surface, where
    the exchange-correlation QUADRATURE is the floor and not the assembly: the
    shift term weights one orbital's density, which the grid resolves worse than
    the total, and refining it moves that pair to 7.7e-10 and 1.3e-08 while the
    analytic force's own translational invariance tracks them (8.9e-10 and
    1.9e-09).
    """
    surface = build(h2())
    g_analytic, _, _ = surface.total_gradient()
    g_fd, ratio = richardson(surface)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, (
        f'{surface.label()}: the finite-difference error fell by {ratio:.2f} '
        f'per halving, not 4 -- a term that survives h -> 0 is one that is '
        f'missing')
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL, (
        f'{surface.label()}: analytic vs h -> 0 '
        f'{np.max(np.abs(g_analytic - g_fd)):.3e}')


def test_the_kohn_sham_correction_is_zero_on_hartree_fock_and_not_on_a_hybrid():
    """`assert ==` rather than a tolerance, and that is the point: every cubic
    gate is a difference measured from this route's Hartree-Fock numbers, so a
    correction that was merely small there would move all of them at once. On a
    hybrid the same two terms must be real, or the gate above would be passing
    on arithmetic that does nothing."""
    mol = water()
    nocc = mol.nelectron // 2
    weights = np.zeros(mol.nao)
    weights[nocc - 1] = 1.0

    y, g = kohn_sham_gradient_correction(mol, rhf(mol), nocc)
    assert np.array_equal(y, np.zeros_like(y)), 'orbital response'
    assert np.array_equal(g, np.zeros_like(g)), 'skeleton'

    # The quasiparticle half cancels NUMERICALLY rather than exactly -- it is a
    # difference of two Coulomb/exchange builds, and Sigma_x - v_xc is the zero
    # operator on Hartree-Fock only analytically. That is why no shift is
    # applied there in the first place: 1e-31 is not zero, and this route's
    # Hartree-Fock numbers are gated bitwise.
    y_qp, g_qp = kohn_sham_gradient_correction(mol, rhf(mol), nocc, weights)
    assert np.abs(y_qp).max() < 1e-12 and np.abs(g_qp).max() < 1e-12

    y, g = kohn_sham_gradient_correction(mol, pbe0(mol), nocc)
    assert np.abs(y).max() > 1e-3 and np.abs(g).max() > 1e-3
    y_qp, g_qp = kohn_sham_gradient_correction(mol, pbe0(mol), nocc, weights)
    assert np.abs(y_qp - y).max() > 1e-3, 'the quasiparticle shift is not in it'
    assert np.abs(g_qp - g).max() > 1e-3


def test_the_declaration_follows_the_reference_the_factory_builds():
    """E_0 is one functional of the density, but WHICH mean field produced that
    density is part of the declaration: two surfaces may only be differenced if
    they share it, and a surface that called its PBE0 reference Hartree-Fock
    would let that difference be taken silently."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    assert DenseRPASurface(mol, rhf).physics_ground_state == GroundState('rpa',
                                                                         'hf')
    assert DenseRPASurface(mol, pbe0).physics_ground_state == GroundState(
        'rpa', 'pbe0')
    assert QuasiparticleSurface(mol, pbe0).physics_ground_state == GroundState(
        'rpa', 'pbe0')


def test_the_quasiparticle_shift_reaches_a_kohn_sham_surface():
    """G0W0 takes out the static exchange-correlation potential the reference
    already counted, and a surface that dropped it would report an ionization
    potential eV out with a gradient that still passed every check above -- a
    force is the derivative of whatever its surface reports, right or wrong.

    Hartree-Fock is the other half: there v_xc IS Sigma_x, the shift is
    round-off, and it is not applied at all, which is what keeps this route's
    Hartree-Fock numbers the oracle they are gated as."""
    mol = h2()
    for factory, moves in ((pbe0, True), (rhf, False)):
        surface = QuasiparticleSurface(mol, factory, charge_change=-1)
        mf = surface.scf_factory(mol)
        _, _, info = surface.total_gradient(mol, mf)
        bare = QPqb(mf.mo_energy, mo_eri(mf, mol), mol.nelectron // 2,
                    screening='rpa')
        # the seeded Newton lands one ulp from the unseeded root, so the
        # Hartree-Fock statement is a tolerance and not an equality
        shift = abs(info['qp_energy_eV']
                    - bare.solve_diag(surface.orbital)[0] * HARTREE_TO_EV)
        assert (shift > 1.0) if moves else (shift < 1e-9), shift


def solvated_pair(mol):
    """(PCM-relaxed mean field, mean field with an `Environment` attached).

    The two markers of a continuum, and NEITHER IMPLIES THE OTHER: pyscf's PCM
    relaxes the orbitals inside the reaction field and leaves `with_solvent`,
    while an `Environment` is attached after the SCF as `with_screening` and
    dresses the interaction the post-SCF methods see. A guard that watched only
    one would let the other through.
    """
    return (SolventScreening(mol, solvent='water').mean_field(mol, rhf),
            attach_solvent_screening(rhf(mol), solvent='water', mol=mol))


def test_a_solvated_mean_field_is_refused_wherever_e_c_is_reported():
    """The dense (pq|rs) is the BARE interaction and consults no environment,
    so E_c^dRPA off it is a different functional from the one the cubic chain
    builds on the SAME mean field -- 5.3 mHa apart for water in water, against
    the 1e-4 Ha a relaxed geometry needs. Computing it silently is the defect;
    refusing is the fix.

    The guard is asserted through the mean field passed to the energy, not
    through the stored factory, because that is the path a dense-versus-cubic
    comparison takes: one SCF handed to both routes.
    """
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    for solvated in solvated_pair(mol):
        reporters = (
            lambda: DenseRPASurface(mol, rhf).total_energy(mol, solvated),
            lambda: DenseRPASurface(mol, rhf).total_gradient(mol, solvated),
            lambda: QuasiparticleSurface(mol, rhf).total_energy(mol, solvated),
            lambda: DenseBSESurface(mol, scf=rhf).total_energy(mol, solvated),
            lambda: DenseBSESurface(mol, scf=rhf).total_gradient(mol, solvated),
        )
        for report in reporters:
            with pytest.raises(ValueError, match='continuum'):
                report()


def test_the_refusal_spares_everything_that_carries_no_correlation_energy():
    """A guard that refused more widely would be a worse defect than the one it
    fixes, so what it must NOT touch is asserted too.

    The gas phase is the overwhelmingly common path; fixed point charges do not
    respond and so leave the interaction bare, which is exactly what this route
    computes; the Tamm-Dancoff surface is E_HF -/+ eps^QP and has no E_c in it;
    and an excitation energy is a difference that never sees E_0.
    """
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    gas = DenseRPASurface(mol, rhf).total_energy(mol)
    assert gas == DenseRPASurface(mol, rhf).total_energy(mol, rhf(mol))
    assert np.isfinite(DenseRPASurface(mol, rhf).total_gradient(mol)[1])

    charges = attach_environment(rhf(mol), PointCharges([[0.0, 0.0, 5.0]],
                                                        [0.4]))
    assert DenseRPASurface(mol, rhf).total_energy(mol, charges) == gas, (
        'a fixed charge screens nothing, so the bare (pq|rs) is right for it')

    pcm, _ = solvated_pair(mol)
    tda = QuasiparticleSurface(mol, rhf, screening='tda')
    assert np.isfinite(tda.total_energy(mol, pcm))
    omega = DenseBSESurface(mol, scf=rhf).excitation_gradient(mol, pcm)[1]
    assert np.isfinite(omega)


def test_a_triplet_surface_is_its_own_derivative_and_is_not_the_singlet():
    """The dense route solves the triplet BSE, and its gradient is complete.

    Kappa weights the bare exchange (ia|jb) in A and in B, and nothing else:
    the screened term W, the quasiparticle energies and every quasi-boson
    normalization are spin-independent. The gradient therefore needs no
    separate treatment -- the one place kappa appears in `partials` is the
    derivative of those two terms.

    Two assertions, because either alone passes on a broken kernel. The
    Richardson gate says the gradient is the derivative of THIS kernel, which a
    kappa applied consistently but wrongly would also satisfy; the separation
    from the singlet says kappa is actually being applied, which a gradient
    check cannot see.
    """
    triplet = DenseBSESurface(water(), scf=rhf, spin='triplet')
    g_analytic, _, _ = triplet.total_gradient()
    g_fd, ratio = richardson(triplet)
    assert abs(ratio - 4.0) < FD_ORDER_TOL * 4.0, (
        f'the finite-difference error fell by {ratio:.2f} per halving, not 4')
    assert np.max(np.abs(g_analytic - g_fd)) < FD_TOL, (
        f'analytic vs h -> 0: {np.max(np.abs(g_analytic - g_fd)):.3e}')

    om_t = DenseBSESurface(water(), scf=rhf,
                           spin='triplet').excitation_gradient()[1]
    om_s = DenseBSESurface(water(), scf=rhf,
                           spin='singlet').excitation_gradient()[1]
    # Dropping a positive-definite bare-exchange coupling can only lower the
    # root, so the ORDERING is physics and not a tolerance.
    assert om_t < om_s, f'triplet {om_t} is not below singlet {om_s}'
    assert om_s - om_t > 1e-3, (
        f'singlet and triplet differ by {om_s - om_t:.2e} Ha, which is what a '
        f'kappa that never reached the kernel would give')


def test_an_unknown_multiplicity_is_refused():
    """KAPPA has two entries; anything else is a typo, not a third spin state."""
    with pytest.raises(ValueError, match='spin'):
        DenseBSESurface(water(), scf=rhf, spin='quintet')
