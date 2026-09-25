"""The ANALYTIC derivative coupling against the overlap finite difference.

Water/cc-pVDZ, RHF reference, BSE@G0W0 on the ISDF/space-time chain with a
dense Casida, singlet. The reference is
`src.properties.nonadiabatic.derivative_couplings`, whose own conventions are
gated in `test_nonadiabatic.py`; what is tested here is that the analytic route
reproduces it.

WHAT EACH GATE IS FOR:

- the interstate SEED is the adjoint of <m|dH|n> at fixed amplitudes, and that
  is checkable with no geometry, no orbital gauge and no configuration term at
  all: perturb the four inputs the Casida blocks are built from. It passes at
  1e-11, better than the diagonal path's own 1e-9, so a failure anywhere below
  is in the assembly and not in the contraction.
- the assembled coupling agrees with the finite difference to O(h^2) with NO
  noise floor over a factor of eight in the step. That is the gate that says
  no term is missing: a missing term would leave an h-independent residual, and
  the symmetric-gauge configuration term this route first carried left 62% of
  the coupling behind in exactly that way. That h^2 being the WHOLE residual is
  also why the assembled gate differences it away instead of tolerating it.
- antisymmetry d_mn = -d_nm is a measurement, not a tautology: the two are
  computed from separate calls and the interstate numerator is symmetrized over
  both orderings, so the sign comes from the gap alone.
- the excitation gradient must be untouched by the refactor that gave the
  interstate seed its own entry into the same reverse chain.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import AMU_TO_ME, NUCLEAR_FD_STEP
from src.gradients.bse_isdf import (bse_backward, bse_blocks, bse_solve,
                                    interstate_backward)
from src.gradients.derivative_coupling import (analytic_coupling,
                                               analytic_ground_coupling,
                                               canonical_overlap_derivative,
                                               configuration_coupling,
                                               state_to_state_density)
from src.gradients.excited_state import ExcitedStateChain
from src.properties import rates, vibronic
from src.properties.nonadiabatic import (derivative_couplings,
                                         translational_sum)
from src.properties.spin_orbit import EXCITED_SPIN_FACTOR

#: C2v, in the yz plane. The three lowest roots are well separated, so the
#: assignment the reference makes across a displacement is never in doubt.
WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
#: Formaldehyde, in the yz plane. A second molecule, and the one the
#: finite-difference reference's own conventions are gated on.
CH2O = 'C 0 0 0.0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
#: A factor of eight in the step: an h^2 truncation falls by 64 over it, while
#: a missing TERM does not move at all.
STEPS = (4.0, 2.0, 1.0, 0.5)
#: The analytic route carries no step and the reference is a central
#: difference, so comparing the two at one h measures the stencil:
#: (4 f(h/2) - f(h))/3 cancels its h^2 and takes the worst residual over the
#: three pairs from 3.9e-04 to 5.1e-07. An order of magnitude above that, and
#: ten times TIGHTER than gating the raw stencil, whose 1e-4 was a statement
#: about the step rather than about the assembly.
COUPLING_GATE = 1e-5


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def chain():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    return ExcitedStateChain(mol, factory, state=0, nroots=5,
                             auxbasis='cc-pvdz-ri')


@pytest.fixture(scope='module')
def extrapolated(chain):
    """d_mn between roots 0-2 from the reference, extrapolated to h -> 0.

    Every state pair comes off the SAME 6 natm + 1 displaced solves, so one
    call over the three roots carries all three pairs -- bitwise the blocks the
    per-pair calls return -- and the two steps together cost less than the
    three single-step calls they replace.
    """
    coarse = derivative_couplings(chain, states=(0, 1, 2),
                                  step=NUCLEAR_FD_STEP)
    fine = derivative_couplings(chain, states=(0, 1, 2),
                                step=0.5 * NUCLEAR_FD_STEP)
    return {(m, n): (4.0 * fine[1 + m, 1 + n] - coarse[1 + m, 1 + n]) / 3.0
            for m in range(3) for n in range(3) if m != n}


def test_interstate_seed_is_the_adjoint_of_the_element():
    """<m|dH|n> at fixed amplitudes, differentiated in all four inputs."""
    rng = np.random.default_rng(7)
    no, nv, naux, npts = 3, 5, 9, 14
    x = rng.normal(size=(npts, no + nv)) * 0.3
    d = rng.normal(size=(npts, naux)) * 0.2
    eps = np.sort(rng.normal(size=no + nv))
    w = rng.normal(size=(naux, naux)) * 0.1
    w = 0.5 * (w + w.T) + np.eye(naux)

    def blocks(x, d, eps, w):
        return bse_blocks(x, d, eps, w, no, spin='singlet', bse_tda=False)

    a, b, cache = blocks(x, d, eps, w)
    om, xn, yn = bse_solve(a, b)
    m, n = 0, 2

    def element(x, d, eps, w):
        a, b, _ = blocks(x, d, eps, w)
        ca = np.outer(xn[:, m], xn[:, n]) + np.outer(yn[:, m], yn[:, n])
        cb = np.outer(xn[:, m], yn[:, n]) + np.outer(yn[:, m], xn[:, n])
        return float((ca * a).sum() + (cb * b).sum())

    seeds = interstate_backward(m, n, x, d, eps, w, no, cache, xn, yn)
    h = 1e-6
    # seeds come back as (eps_qp_bar, X_bar, D_bar, W_aux_bar); `element` takes
    # (X, D, eps, W), so each seed is differentiated in its own argument slot.
    for k, (slot, base) in enumerate(((2, eps), (0, x), (1, d), (3, w))):
        dv = rng.normal(size=np.shape(base))
        if base is w:
            dv = 0.5 * (dv + dv.T)          # W_aux enters symmetrically
        up, dn = [x, d, eps, w], [x, d, eps, w]
        up[slot], dn[slot] = base + h * dv, base - h * dv
        fd = (element(*up) - element(*dn)) / (2 * h)
        assert abs(float((seeds[k] * dv).sum()) - fd) < 1e-8 * max(abs(fd), 1.0)


def test_diagonal_path_survives_the_interstate_refactor(chain):
    """The excitation gradient's own entry into the shared reverse chain.

    Not gated on an absolute energy: Omega is a function of the auxiliary
    basis, the grid counts and the radii, so a magic number here would fail
    whenever any of those moved and would say nothing about the wiring. What
    the refactor could break is the wiring itself, and that shows as a broken
    sum rule or as the two entry points into the chain disagreeing.
    """
    grad, diags = chain.excitation_gradient()
    assert diags['translation_residual'] < 1e-10
    assert diags['stationarity'] < 1e-9
    assert abs(diags['omega'] - chain.excitation()) < 1e-12
    assert np.abs(grad.sum(axis=0)).max() < 1e-10


@pytest.mark.parametrize('pair', ((0, 1), (1, 2), (0, 2)))
def test_analytic_coupling_matches_the_finite_difference(chain, extrapolated,
                                                         pair):
    """The assembled coupling against the reference with its stencil removed.

    Scaled by the REFERENCE's largest component and not by the analytic one, so
    that a defect which inflates the coupling cannot inflate its own tolerance.

    Pair (1, 2) is what forces the extrapolation. Those two roots are coupled
    only through the out-of-plane hydrogen components -- 6.0e-03, against 1.6
    and 0.9 for the two pairs involving root 0 -- so at NUCLEAR_FD_STEP the
    reference's truncation is the SMALLEST of the three in absolute terms,
    2.4e-06, and much the largest once divided by the coupling it sits on. The
    translational sum rule below is the corroboration that owes nothing to a
    finite difference at all.
    """
    m, n = pair
    ana, diags = analytic_coupling(chain, m, n)
    ref = extrapolated[pair]
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.abs(ana - ref).max() / scale < COUPLING_GATE
    # The configuration term is not a correction: it is the larger half here.
    assert diags['branch_configuration'] > 0.0


def test_configuration_term_z_vector_matches_displaced_solves(chain):
    """The Z-vector term against the same contraction by finite difference.

    The displaced route builds A from MO overlaps at displaced geometries --
    mean fields only -- and contracts it directly. It is the independent check
    on the Z-vector assembly, and it is where the gauge is visible: max|A_vv|
    runs to ~27 here, which is the canonical rotation of near-degenerate
    virtuals and is exactly what the symmetric-gauge form throws away.
    """
    mol, mf = chain.mean_field()
    _, pieces = chain._forward(mol, mf)
    xn, yn, nocc = pieces[10], pieces[11], chain.nocc
    goo, gvv = state_to_state_density(nocc, xn[:, 0], yn[:, 0],
                                      xn[:, 1], yn[:, 1])
    ana = configuration_coupling(mol, mf, nocc, goo, gvv)
    num = np.zeros_like(ana)
    for atom in range(mol.natm):
        for axis in range(3):
            a = canonical_overlap_derivative(mol, mf, chain.scf_factory,
                                             atom, axis)
            num[atom, axis] = EXCITED_SPIN_FACTOR * (
                np.einsum('ab,ab->', a[nocc:, nocc:], gvv, optimize=True)
                - np.einsum('ij,ji->', goo, a[:nocc, :nocc], optimize=True))
    assert np.abs(ana - num).max() < 1e-5 * max(np.abs(num).max(), 1.0)


def test_residual_is_the_reference_step_and_not_a_missing_term(chain):
    """The analytic route carries NO step, so the whole residual is the
    reference's own h^2 and must fall as such with no floor."""
    ana, _ = analytic_coupling(chain, 0, 1)
    ratios = []
    for k in STEPS:
        h = k * NUCLEAR_FD_STEP
        fd = derivative_couplings(chain, states=(0, 1), step=h)[1, 2]
        ratios.append(np.abs(ana - fd).max() / h ** 2)
    ratios = np.array(ratios)
    assert ratios.ptp() / ratios.mean() < 0.05


def test_antisymmetry_is_exact(chain):
    """d_mn = -d_nm to machine precision, from two independent calls.

    Exact, not approximate: the interstate numerator is symmetrized over both
    orderings and the gap changes sign, and with the configuration term
    analytic there is no stencil left to break it.
    """
    fwd, _ = analytic_coupling(chain, 0, 1)
    rev, _ = analytic_coupling(chain, 1, 0)
    assert np.abs(fwd + rev).max() < 1e-12 * np.abs(fwd).max()


def test_kohn_sham_reference(chain):
    """A hybrid reference, where the response kernel gains f_xc.

    The Z-vector runs through `response_kernel`, which is `vj - 0.5 vk` on
    Hartree-Fock and picks up the hybrid scaling and f_xc here. That is exactly
    the kind of term that goes wrong silently -- the coupling stays the right
    shape and the wrong size -- so it gets its own gate rather than riding on
    the Hartree-Fock one.
    """
    def ks(mol):
        mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
        mf.grids.level = 5
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
        mf.kernel()
        return mf

    mol = gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)
    ch = ExcitedStateChain(mol, ks, state=0, nroots=5, auxbasis='cc-pvdz-ri')
    ana, diags = analytic_coupling(ch, 0, 1)
    fd = derivative_couplings(ch, states=(0, 1))[1, 2]
    assert np.abs(ana - fd).max() / np.abs(fd).max() < 1e-4
    assert diags['branch_configuration'] > diags['branch_amplitude']


@pytest.mark.parametrize('n', (0, 1, 2))
def test_ground_state_coupling_matches_the_finite_difference(chain, n):
    """d_0n, where the amplitude term is identically absent.

    <Psi_0|Phi_ia> = 0, so the whole coupling is the occupied-virtual block of
    the derivative overlap -- the coupled-perturbed rotation itself, which the
    excited-excited coupling never touches.
    """
    ana, _ = analytic_ground_coupling(chain, n)
    fd = derivative_couplings(chain, states=(n,))[0, 1]
    assert np.abs(ana - fd).max() / np.abs(fd).max() < 1e-4


def test_translational_sum_rule(chain):
    """sum_A d_IJ^A is the velocity-gauge transition moment, from one-electron
    integrals at the reference geometry alone.

    The sharpest gate in this file, and independent of the finite difference:
    nothing in the analytic route is told this identity exists, and it holds to
    machine precision rather than to a stencil. It is NOT the sum rule that
    vanishes -- that one belongs to the electron-translation-factor-corrected
    coupling, a different object.
    """
    mol, mf = chain.mean_field()
    _, pieces = chain._forward(mol, mf)
    ref = translational_sum(mol, mf.mo_coeff, chain.nocc, pieces[10],
                            pieces[11], states=(0, 1, 2))
    for n in (0, 1, 2):
        d, _ = analytic_ground_coupling(chain, n)
        assert np.abs(d.sum(axis=0) - ref[0, 1 + n]).max() < 1e-10
    for m, n in ((0, 1), (0, 2), (1, 2)):
        d, _ = analytic_coupling(chain, m, n)
        assert np.abs(d.sum(axis=0) - ref[1 + m, 1 + n]).max() < 1e-10


def test_internal_conversion_rate_plumbing():
    """The golden-rule sum over modes: shape, scaling and the empty case.

    The rate is quadratic in the coupling and linear in nothing else, so
    doubling every d_k must quadruple it; a zero coupling must give exactly
    zero rather than a small number; and an imaginary frequency must be
    dropped rather than propagated as a negative sqrt.
    """
    omega = np.array([-0.001, 0.004, 0.009])
    d_k = np.array([5.0, 1e-3, 2e-3])
    s_k = np.array([0.0, 0.3, 0.2])

    def rho(de):
        return rates.fc_weighted_dos(de, s_k[1:], omega[1:], 300.0,
                                     lambda_classical=0.002)

    k, per = rates.internal_conversion_rate(0.05, d_k, omega, rho, 300.0)
    assert per[0] == 0.0                      # the imaginary mode is dropped
    assert k > 0.0 and np.isclose(k, per.sum())
    k2, _ = rates.internal_conversion_rate(0.05, 2.0 * d_k, omega, rho, 300.0)
    assert np.isclose(k2, 4.0 * k, rtol=1e-12)
    k0, _ = rates.internal_conversion_rate(0.05, np.zeros_like(d_k), omega,
                                           rho, 300.0)
    assert k0 == 0.0


def test_coupling_projects_onto_modes_like_a_gradient(chain):
    """`project_coupling` is the gradient projection, on the same convention.

    A derivative coupling is one d/dR per Cartesian component, so it
    mass-weights exactly as a gradient does; the two sharing a routine is what
    stops the Huang-Rhys spectrum and the promoting-mode coupling from ending
    up in different units.
    """
    mol, mf = chain.mean_field()
    d, _ = analytic_ground_coupling(chain, 0)
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True)) * AMU_TO_ME
    modes = np.eye(3 * mol.natm)
    got = vibronic.project_coupling(d, modes, masses)
    want = d.ravel() / np.repeat(np.sqrt(masses), 3)
    assert np.abs(got - want).max() < 1e-14


def test_coupling_rotates_with_the_molecule(chain):
    """A derivative coupling is a vector per atom IN ITS OWN FRAME.

    An excited-state optimizer is free to rotate, so the coupling computed at
    the relaxed geometry and the modes computed at the ground-state one live in
    different frames, and projecting one on the other without the rotation
    mixes the components while complaining about nothing. `align_to` reports
    the rotation for exactly this. Measured on acetaldehyde, the projection
    moves by 7.7e-04 against a coupling of 6e-03 when the rotation is dropped
    -- a 12% error that looks like a physical result.
    """
    mol, mf = chain.mean_field()
    d_a, _ = analytic_ground_coupling(chain, 0)
    t = 0.7
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    crd = np.asarray(mol.atom_coords()) @ rot
    turned = gto.M(atom=[(mol.atom_symbol(i), tuple(crd[i]))
                         for i in range(mol.natm)], basis='cc-pvdz',
                   unit='Bohr', verbose=0)
    ch_b = ExcitedStateChain(turned, factory, state=0, nroots=5,
                             auxbasis='cc-pvdz-ri')
    d_b, _ = analytic_ground_coupling(ch_b, 0)
    # UP TO THE GLOBAL SIGN, and that is not sloppiness. d_0n is LINEAR in the
    # Casida vector, whose overall sign each solve picks arbitrarily, so two
    # independent solves have no reason to agree on it -- water flips here and
    # acetaldehyde does not. Every rate is quadratic in the coupling so nothing
    # downstream cares; propagating dynamics would, and would have to fix the
    # phase against a reference the way `align_roots` does.
    turn = d_a @ rot
    assert min(np.abs(d_b - turn).max(), np.abs(d_b + turn).max()) < 1e-6
    _, r_back = vibronic.align_to(mol, turned, return_rotation=True)
    assert min(np.abs(d_b @ r_back - d_a).max(),
               np.abs(d_b @ r_back + d_a).max()) < 1e-6
