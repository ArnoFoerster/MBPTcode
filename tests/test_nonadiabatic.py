"""Derivative couplings between BSE@GW states, by overlaps of the eigenvectors
at displaced geometries.

Formaldehyde/cc-pVDZ, RHF reference, BSE@G0W0 on the cubic ISDF/space-time
chain with a dense Casida, singlet. One displaced solve is 0.6 s, so a 4-atom
coupling is 25 solves, and with the four steps of the sweep the file runs in
about a minute.

WHAT MAKES THIS TESTABLE AT ALL is that the construction has four exact
properties and one asymptotic one, and they fail in different ways:

- S(R0, R0) is the identity EXACTLY, and only for the (XX - YY) combination.
  X alone or XX + YY miss it by 2-5%, which a finite difference turns into an
  O(1/h) term that is not a derivative of anything.
- the coupling is invariant under any orthogonal rotation of the displaced
  orbitals within the occupied and within the virtual space, because the BSE
  was solved in that basis. This is exact, and it is what the reference-index
  bug broke: contracting a bra-geometry occupied index against a ket-geometry
  amplitude looks like an O(h^2) difference and destroys the invariance.
- each coupling vector is a pure nuclear displacement of the irreducible
  representation Gamma_I x Gamma_J, and nothing in the construction is told
  the point group. Formaldehyde's A1/A2 ground-to-dark pair couples through
  the A2 out-of-plane hydrogen wag alone, its A2/B2 interstate pair through
  the B1 mode alone, each to 1e-9 of a coupling of order 0.3.
- sum_A d_IJ^A is the velocity-gauge transition moment, computable from
  one-electron integrals at the reference geometry alone. NOT zero -- the sum
  rule that vanishes belongs to the ETF-corrected coupling.
- antisymmetry d_IJ = -d_JI holds only to the stencil error, because the
  reference-anchored difference does not enforce it. It and the residual of
  the sum rule both fall as h^2 with no noise floor over a factor of eight in
  the step, which is what says the displaced solves are converged rather than
  merely consistent.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.SingleReference.LinearResponse.davidson import oscillator_strengths
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.state_manifold import StateManifold
from src.Base.constants import (NUCLEAR_FD_STEP,
                                ROOT_FOLLOW_MARGIN_MIN)
from src.properties.nonadiabatic import (aligned_overlap,
                                         follow_state,
                                         derivative_couplings, mo_overlap,
                                         nabla_mo, spectrum_solve,
                                         state_overlap, translational_sum)

#: The two lowest BSE roots: formaldehyde's dark n->pi* (A2) and the bright
#: B2 state above it. The pair is chosen so that one ground-to-excited sum rule
#: is zero by symmetry and the other is not.
STATES = (0, 1)
#: In the yz plane, so x is the out-of-plane direction and the C2 axis is z.
CH2O = 'C 0 0 0.0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
#: Steps of the 1/h sweep, as multiples of `NUCLEAR_FD_STEP`. A factor of eight,
#: over which an h^2 truncation falls by 64 while an O(eps/h) noise floor from
#: the displaced solves would instead RISE by 8.
SWEEP = (2.0, 1.0, 0.5, 0.25)


def rhf(mol):
    """The Lagrangian assumes the occupied-virtual Fock block vanishes."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)


@pytest.fixture(scope='module')
def chain(mol):
    return ExcitedStateChain(mol, rhf, solver='dense', mf=rhf(mol))


@pytest.fixture(scope='module')
def reference(chain, mol):
    """(mf, omega, X, Y, nocc) at the reference geometry, one forward pass."""
    mf, omega, x, y = spectrum_solve(chain)(mol)
    return mf, omega, x, y, int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))


@pytest.fixture(scope='module')
def nac(chain, mol):
    return derivative_couplings(chain, mol, states=STATES, step=NUCLEAR_FD_STEP)


@pytest.fixture(scope='module')
def sweep(chain, mol, nac):
    """`{h: DerivativeCouplings}` over `SWEEP`, reusing `nac` for its step."""
    out = {nac.step: nac}
    for scale in SWEEP:
        step = scale * NUCLEAR_FD_STEP
        if step not in out:
            out[step] = derivative_couplings(chain, mol, states=STATES,
                                             step=step)
    return out


# ------------------------------------------------------- the operator algebra
def test_the_overlap_derivative_operator_is_antisymmetric(mol, reference):
    """<p|grad|q> = -<q|grad|p> because <p|q> = delta_pq at every geometry.

    This is the whole reason a derivative coupling takes the X - Y combination
    of a skew-symmetric operator and not the X + Y of a symmetric one.
    """
    nab = nabla_mo(mol, reference[0].mo_coeff)
    assert np.abs(nab + nab.transpose(0, 2, 1)).max() < 1e-12


def test_the_overlap_at_zero_displacement_is_the_identity(mol, reference):
    """S(R0, R0) = 1 exactly, which is what makes the difference a derivative.

    A construction that missed it by delta would report delta/2h as a coupling,
    2e-2 / 2e-3 = 10 in this basis, with nothing raised anywhere.
    """
    mf, _, x, y, nocc = reference
    t = mo_overlap(mol, mf.mo_coeff, mol, mf.mo_coeff)
    s = state_overlap(t, nocc, x[:, :3], y[:, :3], x[:, :3], y[:, :3])
    assert np.abs(s - np.eye(4)).max() < 1e-12


def test_only_the_minus_combination_reaches_the_identity(reference):
    """XX - YY is the Casida metric; XX + YY and X alone are not.

    The plus combination is what a SYMMETRIC one-electron operator takes
    (Himmelsbach and Holzer, J. Chem. Phys. 161, 244105 (2024), Eq. (18)); the
    overlap derivative is skew-symmetric and takes the minus one, their
    Eq. (19). Both miss the identity by percents, so the choice is decided
    here rather than argued about.
    """
    _, _, x, y, _ = reference
    x3, y3 = x[:, :3], y[:, :3]
    minus = x3.T @ x3 - y3.T @ y3
    plus = x3.T @ x3 + y3.T @ y3
    assert np.abs(minus - np.eye(3)).max() < 1e-12
    assert np.abs(plus - np.eye(3)).max() > 1e-2
    assert np.abs(x3.T @ x3 - np.eye(3)).max() > 1e-2


def test_a_foreign_normalization_is_refused_not_rescaled(reference):
    """pySCF normalizes to 1/2; silently accepting it is a factor of two."""
    _, _, x, y, nocc = reference
    with pytest.raises(ValueError, match='from_pyscf'):
        state_overlap(np.eye(len(x)) * 0.0 + np.eye(len(x)), nocc,
                      x[:, :2] / np.sqrt(2.0), y[:, :2] / np.sqrt(2.0),
                      x[:, :2], y[:, :2])


# ------------------------------------------------- gauge freedom of the route
def test_orbital_rotations_of_the_displaced_solve_cancel(chain, mol, reference):
    """An orthogonal rotation within occ and within virt leaves S unchanged.

    T^oo -> T^oo U and X -> U^T X, because the BSE was solved in that basis.
    Exact, and it is why the near-degenerate orbital noise that dominates a
    finite-difference GRADIENT cannot reach a finite-difference coupling.
    """
    mf0, _, x0, y0, nocc = reference
    crd = np.asarray(mol.atom_coords()) + np.array([[2e-3, 0, 0]] * mol.natm)
    moved = mol.copy()
    moved.set_geom_(crd, unit='Bohr')
    moved.build(False, False)
    mfd, _, xd, yd = spectrum_solve(chain)(moved)

    nmo = mfd.mo_coeff.shape[1]
    nvir = nmo - nocc
    rng = np.random.default_rng(0)
    uo = np.linalg.qr(rng.normal(size=(nocc, nocc)))[0]
    uv = np.linalg.qr(rng.normal(size=(nvir, nvir)))[0]
    u = np.zeros((nmo, nmo))
    u[:nocc, :nocc], u[nocc:, nocc:] = uo, uv
    rot = [np.einsum('ij,ab,jbK->iaK', uo.T, uv.T, v.reshape(nocc, nvir, -1),
                     optimize=True).reshape(nocc * nvir, -1) for v in (xd, yd)]

    plain = state_overlap(mo_overlap(mol, mf0.mo_coeff, moved, mfd.mo_coeff),
                          nocc, x0[:, :2], y0[:, :2], xd, yd)
    turned = state_overlap(
        mo_overlap(mol, mf0.mo_coeff, moved, mfd.mo_coeff @ u), nocc,
        x0[:, :2], y0[:, :2], rot[0], rot[1])
    assert np.abs(turned - plain).max() < 1e-11


def test_root_tracking_undoes_a_permutation_and_a_sign(mol, reference):
    """A displaced solve reordered and sign-flipped is put back by the overlap.

    Assignment takes the largest |S| first, so an unambiguous root is never
    left with whatever column the scan order hands an ambiguous one.
    """
    mf, _, x, y, nocc = reference
    order = [2, 0, 1]
    flip = np.array([1.0, -1.0, -1.0])
    xd = x[:, order] * flip[None, :]
    yd = y[:, order] * flip[None, :]
    t = mo_overlap(mol, mf.mo_coeff, mol, mf.mo_coeff)
    s = state_overlap(t, nocc, x[:, :3], y[:, :3], xd, yd)
    aligned, recovered, weight = aligned_overlap(s, 3)
    assert list(recovered) == [1, 2, 0]
    assert weight > 1.0 - 1e-12
    assert np.abs(aligned - np.eye(4)).max() < 1e-12


def test_following_one_state_survives_a_reordering(mol, reference):
    """An optimizer follows a STATE; the energy order it comes back in is not
    the state's identity."""
    mf, _, x, y, nocc = reference
    order = [2, 0, 1]
    xd, yd = x[:, order], y[:, order]
    t = mo_overlap(mol, mf.mo_coeff, mol, mf.mo_coeff)
    s = state_overlap(t, nocc, x[:, :3], y[:, :3], xd, yd)
    for want, came_back_as in enumerate((1, 2, 0)):
        index, weight, margin = follow_state(s, want)
        assert index == came_back_as
        assert weight > 1.0 - 1e-12
        assert margin > 1.0 - 1e-12


def test_a_mixed_pair_shows_as_a_SMALL_MARGIN_not_a_small_weight():
    """The dangerous case is two roots sharing the reference character.

    Both overlaps are then moderate, so a weight threshold passes it and the
    follower picks whichever is larger -- a coin toss the size of the margin.
    Guarding on weight alone does not see this.
    """
    s = np.zeros((3, 3))
    s[1, 1], s[1, 2] = 0.70, 0.69          # one state split across two roots
    index, weight, margin = follow_state(s, 0)
    assert index == 0
    assert weight > 0.5, 'a weight guard would pass this'
    assert margin < 0.02, 'and the margin is what says it should not'


def test_a_state_that_left_the_window_shows_as_a_small_weight():
    s = np.zeros((3, 3))
    s[1, 1], s[1, 2] = 0.08, 0.03
    _, weight, margin = follow_state(s, 0)
    assert weight < 0.1
    assert margin < 0.1


def test_following_refuses_when_there_is_nothing_to_follow_into():
    with pytest.raises(ValueError, match='no displaced roots'):
        follow_state(np.zeros((2, 1)), 0)


# ------------------------------------------------------------- the entry points
def test_every_entry_point_drives_the_same_solve(chain, mol, reference):
    """A chain, a composed surface and a manifold hand back one spectrum.

    The composed surface reads its excitation off the chain in `excited` and
    its mean field off the ground-state half, so an adapter that drove the
    outer object would return root 0 for every state.
    """
    surface = RPABSESurface(mol, rhf, state=0, mf=reference[0], solver='dense')
    manifold = StateManifold(surface, states=STATES)
    base = spectrum_solve(chain)(mol)
    for source in (surface, manifold):
        got = spectrum_solve(source)(mol)
        assert np.abs(got[1] - base[1]).max() < 1e-10
        assert np.abs(np.abs(got[2]) - np.abs(base[2])).max() < 1e-8


def test_the_geometry_defaults_to_the_one_the_chain_was_frozen_at(
        chain, mol, reference):
    """`solve(None)` is the reference pair, so `mol` is optional throughout.

    A route that silently built a fresh mean field instead would difference one
    function against another, which is how the quasiparticle correction's
    forward and reverse halves once disagreed.
    """
    mf, omega, _, _ = spectrum_solve(chain)(None)
    assert mf.mol is mol
    assert np.abs(omega - reference[1]).max() < 1e-12


def test_a_root_beyond_the_solve_is_refused(chain, mol):
    with pytest.raises(ValueError, match='roots'):
        derivative_couplings(chain, mol, states=(0, 100000))


def test_the_displaced_geometries_keep_the_reference_conventions(chain, mol):
    """A factorization frozen at the reference is what a displacement differences.

    Refreezing per displaced point would rebuild the radii, the interpolation
    layout and the frames, and a convention that moves with the displacement is
    a discontinuity in the quantity being differenced.
    """
    frozen = chain.factorization
    moved = mol.copy()
    moved.set_geom_(np.asarray(mol.atom_coords()) + 1e-3, unit='Bohr')
    moved.build(False, False)
    spectrum_solve(chain)(moved)
    assert chain.factorization is frozen


# ----------------------------------------------------------- the couplings
def test_the_coupling_is_antisymmetric_to_the_stencil_error(nac):
    """d_IJ = -d_JI is NOT built into the stencil and so measures everything.

    The bra sits at the reference for both displacements, so antisymmetry holds
    only because the displaced states really are orthonormal. The two-sided
    Hammes-Schiffer/Tully form would make this a tautology.
    """
    assert nac.antisymmetry < 1e-6
    assert nac.diagnostics['identity_residual'] < 1e-12
    assert nac.diagnostics['root_swaps'] == 0
    assert nac.diagnostics['assignment_weight'] > 0.999


def test_the_diagonal_coupling_vanishes(nac):
    """<Psi_I|d/dR Psi_I> = 0 for real states, and d_00 needs det(T^oo)^2 to
    deliver it: tr K^oo = 0 only because K is antisymmetric."""
    for i in range(len(STATES) + 1):
        assert np.abs(nac[i, i]).max() < 1e-6


def test_the_translational_sum_rule_holds_for_the_electronic_coupling(
        mol, reference, nac):
    """sum_A d_IJ^A is the velocity-gauge transition moment, to the stencil error.

    An exact identity of the construction -- the orbitals ride with their
    centres, so d T_pq / d(translation) = -<phi_p|grad|phi_q> -- and the only
    check available that compares the whole displacement machinery against an
    analytic contraction at the reference geometry alone.
    """
    mf, _, x, y, nocc = reference
    predicted = translational_sum(mol, mf.mo_coeff, nocc, x, y, states=STATES)
    assert np.abs(nac.d.sum(axis=2) - predicted).max() < 1e-5


def test_the_sum_over_atoms_is_not_zero(nac):
    """The sum rule that VANISHES belongs to the ETF-corrected coupling.

    The bare derivative coupling of a bright pair sums to the transition
    moment, which is order one here; asserting sum_A d = 0 would be asserting
    the molecule has no oscillator strength.
    """
    bright = np.abs(nac.d.sum(axis=2)[0, 2]).max()
    dark = np.abs(nac.d.sum(axis=2)[0, 1]).max()
    assert bright > 0.2
    assert dark < 1e-8          # the n->pi* is A2: no transition moment at all


def test_each_coupling_is_a_pure_displacement_of_one_irrep(nac):
    """d_IJ is nonzero only along a mode of the symmetry Gamma_I x Gamma_J.

    In C2v with the molecule in the yz plane, the A1/A2 ground-to-dark pair
    couples through the A2 displacement -- the two hydrogens leaving the plane
    in opposite senses, C and O fixed -- and the A2/B2 interstate pair through
    the B1 one, all four atoms out of plane with the hydrogens equal. Nothing
    in the construction knows about the point group, so every component that
    vanishes here does so because the Casida vectors, the transported metric
    and the displaced solves are each right; a leaked in-plane component would
    be the signature of a bra index contracted against a ket amplitude.
    """
    dark = nac[0, 1]
    assert np.abs(dark[:2]).max() < 1e-9            # C and O do not move
    assert np.abs(dark[:, 1:]).max() < 1e-9         # out of plane only
    assert abs(dark[2, 0] + dark[3, 0]) < 1e-9      # antisymmetric: A2
    assert abs(dark[2, 0]) > 0.3

    interstate = nac[1, 2]
    assert np.abs(interstate[:, 1:]).max() < 1e-8   # out of plane only
    assert abs(interstate[2, 0] - interstate[3, 0]) < 1e-8   # symmetric: B1
    assert abs(interstate[0, 0]) > 0.2


def test_the_length_gauge_relation_does_not_hold(mol, reference, nac):
    """sum_A d_0J = -Omega_J <0|r|J> needs a hypervirial BSE@GW does not have.

    Its diagonal is a quasiparticle spectrum from a self-energy, not a one-body
    operator, so the commutator that turns velocity into length is broken -- on
    top of the finite basis, which breaks it for any method. The two forms of
    the same sum rule are asserted to DISAGREE, far outside the stencil error,
    because an implementation that made them agree would have lost the
    quasiparticle diagonal.
    """
    mf, omega, x, y, nocc = reference
    dip = oscillator_strengths(mf, mol, nocc, omega[:2], x[:, :2], y[:, :2])[1]
    velocity = nac.d.sum(axis=2)[0, 2]
    length = -omega[1] * dip[1]
    gap = np.linalg.norm(velocity - length) / np.linalg.norm(length)
    assert 1e-2 < gap < 0.5


def test_the_error_falls_as_the_step_squared_with_no_noise_floor(
        mol, reference, sweep):
    """A 1/h sweep over a factor of eight, on two independent exact identities.

    Antisymmetry and the translational sum rule are both residuals of something
    the construction satisfies exactly, so each is pure stencil error and each
    must fall as h^2. An error that FLATTENED or ROSE at the small-h end would
    be the displaced solves' own convergence entering as O(eps/h) -- the
    discriminator the gradient campaign uses -- and there is none of it down to
    a quarter of the default step. The coupling itself is then step-independent
    far below the level it is used at.
    """
    mf, _, x, y, nocc = reference
    predicted = translational_sum(mol, mf.mo_coeff, nocc, x, y, states=STATES)
    steps = sorted(sweep, reverse=True)
    skew = [sweep[h].antisymmetry for h in steps]
    drift = [np.abs(sweep[h].d.sum(axis=2) - predicted).max() for h in steps]
    for err in (skew, drift):
        for coarse, fine in zip(err[:-1], err[1:]):
            assert 3.5 < coarse / fine < 4.5
        assert abs(np.polyfit(np.log(steps), np.log(err), 1)[0] - 2.0) < 0.05

    finest = sweep[steps[-1]].d
    for h in steps:
        assert np.abs(sweep[h].d - finest).max() < 1e-4


def test_the_same_displaced_solves_reproduce_the_analytic_gradient(
        chain, mol, reference, nac):
    """dOmega/dR off the coupling's own displacements against the gated force.

    The coupling and the excitation gradient are two contractions of the same
    nuclear derivatives, so no identity relates them; what this gates is
    everything they share -- the displacement, the mean field at it, the frozen
    conventions and the root tracking -- against a quantity that is already
    finite-difference gated end to end.
    """
    mf = reference[0]
    for k, n in enumerate(STATES):
        chain.state = n
        analytic = chain.excitation_gradient(mol, mf)[0]
        differenced = nac.diagnostics['omega_gradient'][k]
        rel = np.abs(analytic - differenced).max() / np.abs(analytic).max()
        assert rel < 1e-4
    chain.state = 0


def test_the_pair_accessor_returns_one_cartesian_array(nac):
    """d_IJ is (natm, 3) per pair, with the ground state at index 0."""
    assert nac[0, 1].shape == (4, 3)
    assert np.allclose(nac[0, 1], -nac[1, 0], atol=1e-6)
    assert nac.states == STATES
    assert 'h = ' in nac.label()


# --------------------------------------------------- following on the surface
def test_tracking_off_is_bit_identical_to_the_index(mol):
    """`track=None` is the shipped behaviour and must not move."""
    plain = ExcitedStateChain(mol, rhf, solver='dense', mf=rhf(mol))
    tracked = ExcitedStateChain(mol, rhf, solver='dense', mf=rhf(mol),
                                track='overlap')
    assert plain.excitation(mol) == tracked.excitation(mol)
    assert plain.follow_log == []
    assert len(tracked.follow_log) == 1, 'the first pass anchors, it follows nothing'
    assert tracked.follow_log[0]['weight'] == 1.0


def test_the_first_pass_anchors_and_a_repeat_finds_the_same_root(mol):
    """Re-evaluating at the SAME geometry must return the same state with a
    unit overlap: anything less means the follower is reading noise."""
    chain = ExcitedStateChain(mol, rhf, solver='dense', mf=rhf(mol),
                              state=1, track='overlap')
    first = chain.excitation(mol)
    again = chain.excitation(mol)
    assert again == pytest.approx(first, abs=1e-12)
    step = chain.follow_log[-1]
    assert step['index'] == 1, 'it must not drift off its own state'
    assert step['weight'] > 1.0 - 1e-8
    assert step['margin'] > 0.5
    assert step['anchor'] > 1.0 - 1e-8


def test_a_displaced_geometry_keeps_the_state_and_logs_its_overlap(mol):
    """The state survives a real displacement, and the log carries the
    evidence rather than the caller having to trust it."""
    chain = ExcitedStateChain(mol, rhf, solver='dense', mf=rhf(mol),
                              track='overlap')
    chain.excitation(mol)
    moved = mol.copy()
    coords = moved.atom_coords().copy()
    coords[0, 2] += 0.02
    moved.set_geom_(coords, unit='Bohr')
    chain.excitation(moved)
    step = chain.follow_log[-1]
    assert step['weight'] > 0.9, f'a 0.02 Bohr step should be unambiguous, got {step}'
    assert step['margin'] > ROOT_FOLLOW_MARGIN_MIN


def _pieces_with(xn, yn):
    """The 16-tuple `_forward` returns, carrying only what following reads."""
    pieces = [None] * 16
    pieces[10], pieces[11] = xn, yn
    return tuple(pieces)


def test_the_surface_follows_the_state_through_an_index_SWAP(mol, reference):
    """The point of the whole mechanism: when the energy order changes, the
    followed index changes WITH it so the state does not.

    Without following this returns `self.state` unchanged and the optimizer
    walks onto whichever state is now n-th -- the acene 1La/1Lb failure.
    """
    mf, om, x, y, nocc = reference
    chain = ExcitedStateChain(mol, rhf, solver='dense', mf=mf, track='overlap')
    assert chain.nocc == nocc
    chain.tracked_state(mol, mf, om, _pieces_with(x, y))     # anchors on 0
    order = [2, 0, 1]                     # state 0 comes back as column 1
    index = chain.tracked_state(mol, mf, np.asarray(om)[order],
                                _pieces_with(x[:, order], y[:, order]))
    assert index == 1, 'the follower must move off the stale index'
    assert chain.follow_log[-1]['weight'] > 1.0 - 1e-10
    assert chain.state == 0, 'and must not mutate the state it was asked for'


def test_a_refrozen_chain_inherits_the_FOLLOWED_index(mol, reference):
    """`refreeze` starts a fresh overlap history, so it has to be handed the
    root that IS the state now -- rebuilding on the original index hands the
    outer loop the state that was crossed."""
    mf, om, x, y, _ = reference
    chain = ExcitedStateChain(mol, rhf, solver='dense', mf=mf, track='overlap')
    chain.tracked_state(mol, mf, om, _pieces_with(x, y))
    order = [2, 0, 1]
    chain.tracked_state(mol, mf, np.asarray(om)[order],
                        _pieces_with(x[:, order], y[:, order]))
    assert chain.refreeze(mol).state == 1
    assert chain.refreeze(mol).track == 'overlap'
