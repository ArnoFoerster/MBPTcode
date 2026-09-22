"""The quasi-boson supermatrix and the Casida route are one G0W0@HF.

`SingleReference.GW.quasi_boson.QPqb` downfolds the dense quasi-boson
supermatrix to w = eps_p + Sigma_pp(w) and solves it by an unguarded Newton on
the closed-form self-energy; `qp_energy.calc_qp_energy(mode='casida')` builds
the same Sigma_c as a Lehmann sum over the Casida spectrum at w + i.eta and
hands the residual to `Solvers.qp_equation`. This gate holds that the two are
the SAME functional -- one dense-RPA-screened G0W0 on a Hartree-Fock reference
-- and that the only thing separating the two numbers is the algebraic route
to the root.

WHAT MAKES THEM THE SAME BY CONSTRUCTION. Each of these is a choice on one
side that has to be matched on the other, not a tolerance:

  * df=False. `build_rpa_AB` holds the exact four-index (ia|jb), so the Casida
    route must not fit it; the two (A, B) pairs are then BITWISE equal.
  * screening='rpa' against the full Casida problem (no TDA on either side).
    Flipping `QPqb` to screening='tda' moves the root 5.1e-2 to 4.6e-1 eV.
  * eta = 0. `QPqb` has no broadening at all, so the Lorentzian denominator
    w - eps_q +- Om / ((w - eps_q +- Om)^2 + eta^2) has to be the bare pole.
    The default eta = 1e-3 Ha moves the root by 2.8e-8 to 8.6e-6 eV, above
    this gate, so eta=0 is load-bearing. `calc_qp_energy` accepts eta=0 on
    mode='casida' -- the refusal of a non-default eta is the imaginary-axis
    branch, which has no broadening to set -- and no denominator is singular
    at the roots measured here.
  * a Hartree-Fock reference, where <Sigma_x - v_xc> vanishes identically:
    `_static_correction` returns exact zeros and `QPqb.delta` is zeros, so
    NEITHER side adds an explicit exchange term. Both solve
    w = eps_p + Sigma_c(w) with Sigma_x already inside eps_p.
  * qp_solver='newton', the nearest-root rule, matching `QPqb`'s Newton
    started from eps_p. No root selection is in play: every Z below is > 0.94,
    so there is one quasiparticle and no satellite to choose between.

WHAT WAS MEASURED, in eV, on RHF references converged to conv_tol_grad 1e-11.
"tight" is the production residual driven to |f| < 1e-14 Ha by the
repository's own `solve_qp_equation_newton`; "driver" is `calc_qp_energy`.

  system         p  orbital  qb energy         |tight-qb|  |driver-qb|  Z
  water/sto-3g   4  HOMO      -8.999245032650    1.1e-14     2.0e-09    0.967283
  water/sto-3g   3  HOMO-1   -11.351370791786    4.5e-15     1.6e-10    0.968584
  water/sto-3g   5  LUMO      16.568202106750    3.0e-15     1.3e-06    0.982317
  water/cc-pVDZ  4  HOMO     -12.157969852912    2.3e-14     9.0e-10    0.950610
  water/cc-pVDZ  3  HOMO-1   -14.437172408718    2.7e-14     2.9e-10    0.951163
  water/cc-pVDZ  5  LUMO       4.706485853318    1.5e-14     1.2e-05    0.989217
  h4/sto-3g      1  HOMO      -6.720683425627    7.6e-16     2.8e-11    0.976713
  h4/sto-3g      0  HOMO-1   -17.925478630433    0.0e+00     2.5e-08    0.943808
  h4/sto-3g      2  LUMO       6.562794618638    1.2e-13     1.2e-13    0.971477

Z is the same on both sides to 3e-6 (the quasi-boson slope is analytic, the
production one a central difference of the residual at 1e-4 Ha).

THE VERDICT. The physics is identical: (A, B) bitwise, the RPA spectrum to
1.0e-13 Ha, Sigma_c to 1.1e-15 Ha at both eps_p and the root, and the two
quasiparticle roots to 2.7e-14 eV -- six orders below the 1e-8 eV the gate
asserts. Neither side is wrong.

THE PLAN'S 1e-8 eV IS NOT REACHABLE THROUGH `calc_qp_energy`, and the whole
residual is attributed: `solve_qp_equation_newton` stops at
|f(w)| < QP_NEWTON_TOL = 1e-6 Ha and `calc_qp_energy` exposes no way to ask
for less, so the driver returns a point w_drv off the root by exactly
Z * f(w_drv) -- reproduced below to 7.2e-14 eV on every row, worst case
1.2e-5 eV on the water/cc-pVDZ LUMO. That is a stopping rule, not a term:
`QPqb.solve_diag` converges the same equation to |dw| < 1e-12 Ha, and
qp_solver='pole_strength' (production's default, which bisects to
QP_GRAPHICAL_TOL = 1e-8 Ha) lands 5.8e-8 eV away instead of 1.2e-5.

A keyword the route does not have: `return_z=True` is REFUSED on
mode='casida', whose continuation is 'spectral' -- only the three contour
continuations return their Newton slope. The production Z below therefore
comes from `qp_equation.pole_strength` on the same residual.

THE GATE WAS SHOWN TO FAIL. `ScaledSigmaQPqb` below -- the production class
with its correlation self-energy and slope scaled by SIGMA_SCALE and nothing
else -- put in place of `QPqb` for one run moves the quasi-boson root by
9.9e-8 to 1.6e-6 eV, above GATE_EV on all nine rows, and fails
`test_the_correlation_self_energy_is_the_same_function` (6.3e-8 Ha where
SIGMA_TOL is 1e-12), `test_the_quasiparticle_root_is_the_same_number`
(1.6e-6 eV where GATE_EV is 1e-8) and
`test_the_driver_differs_only_by_its_newton_stopping_rule` (1.6e-6 eV left
unattributed). The three that stand on the screening alone -- the bitwise
(A, B) pair, the vanishing static shift and the broadening -- still pass,
which is the point of keeping them separate: they cannot see a self-energy
that drifted.
"""
import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.Base.constants import (DEFAULT_BROADENING_ETA, HARTREE_TO_EV,
                                QP_NEWTON_TOL)
from src.Base.pyscf_interface import get_orbital_energies
from src.SingleReference.GW.qp_energy import (_static_correction,
                                              _two_electron_integrals,
                                              calc_qp_energy)
from src.SingleReference.GW.quasi_boson import QPqb, build_rpa_AB
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.Solvers.qp_equation import pole_strength, solve_qp_equation_newton

#: The three references every number is measured on.
SYSTEMS = (('water/sto-3g', 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; '
            'H 0.0 -0.7572 -0.4692', 'sto-3g', 'Angstrom'),
           ('water/cc-pVDZ', 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; '
            'H 0.0 -0.7572 -0.4692', 'cc-pvdz', 'Angstrom'),
           ('h4/sto-3g', 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; '
            'H 2.52 1.62 0.9', 'sto-3g', 'Bohr'))

#: The acceptance on the physics, in eV. The measured spread is 2.7e-14.
GATE_EV = 1e-8

#: The RPA spectrum and Sigma_c compared in Hartree: one eigh of Abar against
#: the symplectic Casida solve, and one closed-form pole sum against a Lehmann
#: sum. Measured 1.0e-13 and 1.1e-15.
SPECTRUM_TOL = 1e-12
SIGMA_TOL = 1e-12

#: How well w_drv - w* = Z * f(w_drv) accounts for the driver's whole
#: departure from the root, in eV. Measured 7.2e-14.
ATTRIBUTION_EV = 1e-10

#: The residual the tight root is driven to, in Hartree, and the Newton cap
#: that reaches it. Six orders below QP_NEWTON_TOL, so the root it returns is
#: the equation's and not the stopping rule's.
TIGHT_TOL = 1e-14
TIGHT_MAX_ITER = 400

#: The perturbation the gate was shown to fail under.
SIGMA_SCALE = 1.0 + 1e-6


class ScaledSigmaQPqb(QPqb):
    """`QPqb` with the correlation self-energy and its slope scaled.

    Nothing else changes: the bosons, the couplings and the Newton are the
    production ones, so the root moves by the scale alone.
    """

    def sigma(self, p, w):
        return SIGMA_SCALE * super().sigma(p, w)

    def sigma_prime(self, p, w):
        return SIGMA_SCALE * super().sigma_prime(p, w)


def rhf_reference(atom, basis, unit):
    """(mol, mf, eri, nocc) on a tightly converged RHF and its full MO ERI."""
    mol = gto.M(atom=atom, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    norb, nocc = mol.nao, mol.nelectron // 2
    eri = ao2mo.general(mol, (mf.mo_coeff,) * 4,
                        compact=False).reshape((norb,) * 4)
    return mol, mf, eri, nocc


def casida_stack(mol, mf, nocc, eta):
    """The production Casida-route objects `calc_qp_energy` builds internally.

    Exactly the restricted GW@RPA, df=False path of the driver: the exact
    four-index integrals, the singlet (A, B) pair, its full spectrum, and the
    <Sigma_Hx - v_Hxc> diagonal that a Hartree-Fock reference makes zero.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    _, eri = _two_electron_integrals(mol, mf, df=False, is_uhf=False)
    lr = LinearResponseSolver(eps, coeff_df=None, eri_chemist=eri,
                              spin_mode='restricted', eta=eta)
    se = SelfEnergySolver(eps, df_coeff=None, eri_chemist=eri,
                          spin_mode='restricted', eta=eta)
    A, B = lr.build_casida_matrices(nocc, lBSE=False, W_aux=None,
                                    triplet=False)
    omega, X, Y = CasidaSolver(A, B).solve(tda=False)
    xc = _static_correction(mf, mol, se, None, None, 'alpha', False)
    return {'eps': np.asarray(eps, float), 'se': se, 'A': A, 'B': B,
            'omega': omega, 'X': X, 'Y': Y, 'xc': xc, 'nocc': nocc}


def production_sigma(stack, p):
    """Sigma_c(w) for orbital p, as the Casida route's Lehmann sum."""
    se, nocc = stack['se'], stack['nocc']
    chi_a = se.get_chi_a(nocc, stack['X'], stack['Y'], spin_channel='alpha',
                         p_state=p)
    return lambda w: se.calculate_self_energy(
        p, w, nocc, stack['omega'], chi_a, None, spin_channel='alpha',
        vertex_mode='GW', calc_imag=False)


def production_residual(stack, p):
    """f(w) = w - eps_p - <Sigma_x - v_xc>_pp - Sigma_c(w), the driver's own."""
    sigma = production_sigma(stack, p)
    return lambda w: w - stack['eps'][p] - stack['xc'][p] - sigma(w)


def orbitals(nocc):
    """HOMO, HOMO-1, LUMO -- the three rows measured per system."""
    return {'HOMO': nocc - 1, 'HOMO-1': nocc - 2, 'LUMO': nocc}


@pytest.fixture(scope='module')
def gate():
    """Both routes' numbers for every (system, orbital) row, computed once."""
    rows = {}
    for name, atom, basis, unit in SYSTEMS:
        mol, mf, eri, nocc = rhf_reference(atom, basis, unit)
        qp = QPqb(np.asarray(mf.mo_energy, float), eri, nocc, screening='rpa')
        stack = casida_stack(mol, mf, nocc, 0.0)
        states = sorted(orbitals(nocc).values())
        driver = calc_qp_energy(mf, df=False, eta=0.0, mode='casida',
                                qp_solver='newton', state=states)
        rows[name] = {'mol': mol, 'mf': mf, 'eri': eri, 'nocc': nocc,
                      'qp': qp, 'stack': stack,
                      'driver': {p: driver[p]['GW'] for p in states}}
    return rows


def test_the_two_routes_build_the_same_dense_rpa_problem(gate):
    """One (A, B) pair, bitwise, and one boson spectrum.

    `build_rpa_AB` and `LinearResponseSolver.build_casida_matrices(lBSE=False)`
    are separate routines over the same exact (ia|jb) with the same factor-2
    singlet convention, so the screening both self-energies stand on is not
    merely close. The spectra then differ only by the eigensolver: one eigh of
    Abar against the symplectic Casida solve.
    """
    for name, row in gate.items():
        qp, stack = row['qp'], row['stack']
        A, B, _ = build_rpa_AB(np.asarray(row['mf'].mo_energy, float),
                               row['eri'], row['nocc'])
        assert np.array_equal(A, stack['A']), name
        assert np.array_equal(B, stack['B']), name
        spread = np.abs(np.sort(qp.omega) - np.sort(stack['omega'])).max()
        assert spread < SPECTRUM_TOL, f'{name}: RPA spectrum {spread:.3e} Ha'


def test_the_hartree_fock_reference_leaves_no_static_shift(gate):
    """<Sigma_x - v_xc> is identically zero on both sides, not small.

    G0W0 removes the static potential the mean field already counted, and on a
    gas-phase Hartree-Fock reference v_xc IS Sigma_x. Both routes therefore
    solve w = eps_p + Sigma_c(w) with the exchange inside eps_p, and a gate
    that did not check this could be comparing two different equations.
    """
    for name, row in gate.items():
        assert not np.any(row['stack']['xc']), name
        assert not np.any(row['qp'].delta), name


def test_the_correlation_self_energy_is_the_same_function(gate):
    """Sigma_c agrees at the starting point and at the root, not just at one.

    The closed-form pole sum over (j, nu) and (b, nu) against the Lehmann sum
    over the Casida spectrum: the same singlet factor 2, the same dressed
    couplings, the same eta = 0 denominators.
    """
    for name, row in gate.items():
        qp, stack = row['qp'], row['stack']
        for label, p in orbitals(row['nocc']).items():
            sigma = production_sigma(stack, p)
            w_qb, _ = qp.solve_diag(p)
            for w in (stack['eps'][p], w_qb):
                d = abs(qp.sigma(p, w) - sigma(w))
                assert d < SIGMA_TOL, f'{name} {label}: {d:.3e} Ha at w={w}'


def test_the_quasiparticle_root_is_the_same_number(gate):
    """The two roots agree at GATE_EV; the pole strengths agree with them.

    The production residual is driven to TIGHT_TOL by the repository's own
    Newton, so what is compared is the equation and not a stopping rule. Every
    Z here is above 0.94: one quasiparticle per orbital, so the nearest-root
    and largest-weight rules cannot disagree and no root selection enters.
    """
    for name, row in gate.items():
        qp, stack = row['qp'], row['stack']
        for label, p in orbitals(row['nocc']).items():
            w_qb, z_qb = qp.solve_diag(p)
            f = production_residual(stack, p)
            w_tight = solve_qp_equation_newton(f, stack['eps'][p],
                                               tol=TIGHT_TOL,
                                               max_iter=TIGHT_MAX_ITER)
            d = abs(w_tight - w_qb) * HARTREE_TO_EV
            assert d <= GATE_EV, f'{name} {label}: {d:.3e} eV'
            z_casida = pole_strength(f, w_tight)
            assert z_qb > 0.9, f'{name} {label}: Z={z_qb:.3f}'
            assert abs(z_casida - z_qb) < 1e-5, f'{name} {label}'


def test_the_driver_differs_only_by_its_newton_stopping_rule(gate):
    """`calc_qp_energy` stops at QP_NEWTON_TOL, and that accounts for all of it.

    `solve_qp_equation_newton` returns the first iterate with
    |f(w)| < QP_NEWTON_TOL and `calc_qp_energy` passes no tolerance through, so
    the driver's number is a point on the residual rather than its zero. Near
    the root f(w) = (w - w*)/Z, so the whole departure is Z * f(w_drv) -- and
    it is, to ATTRIBUTION_EV. Nothing is left over for a missing term.
    """
    for name, row in gate.items():
        qp, stack = row['qp'], row['stack']
        for label, p in orbitals(row['nocc']).items():
            w_qb, _ = qp.solve_diag(p)
            f = production_residual(stack, p)
            w_drv = row['driver'][p] / HARTREE_TO_EV
            residual = f(w_drv)
            assert abs(residual) < QP_NEWTON_TOL, f'{name} {label}'
            predicted = pole_strength(f, w_drv) * residual
            left_over = abs((w_drv - w_qb) - predicted) * HARTREE_TO_EV
            assert left_over < ATTRIBUTION_EV, \
                f'{name} {label}: {left_over:.3e} eV unattributed'


def test_the_zero_broadening_is_load_bearing(gate):
    """eta = 0 is a construction choice, not a tolerance that could be relaxed.

    `QPqb` has no broadening, so the Casida route has to run without one. The
    default eta = 1e-3 Ha moves every root by more than GATE_EV -- up to
    8.6e-6 eV -- which is what makes eta=0 part of matching the two rather
    than a cosmetic argument.
    """
    for name, row in gate.items():
        broadened = casida_stack(row['mol'], row['mf'], row['nocc'],
                                 DEFAULT_BROADENING_ETA)
        for label, p in orbitals(row['nocc']).items():
            roots = [solve_qp_equation_newton(production_residual(s, p),
                                              s['eps'][p], tol=TIGHT_TOL,
                                              max_iter=TIGHT_MAX_ITER)
                     for s in (row['stack'], broadened)]
            shift = abs(roots[1] - roots[0]) * HARTREE_TO_EV
            assert shift > GATE_EV, f'{name} {label}: {shift:.3e} eV'


def test_a_scaled_correlation_self_energy_moves_the_root_past_the_gate(gate):
    """The perturbation the gate was shown to fail under, as a measurement.

    Scaling Sigma_c by SIGMA_SCALE and nothing else moves the quasi-boson root
    off the production one by more than GATE_EV on every row, so
    `test_the_quasiparticle_root_is_the_same_number` would fail rather than
    pass on a route that had drifted by that much.
    """
    for name, row in gate.items():
        scaled = ScaledSigmaQPqb(np.asarray(row['mf'].mo_energy, float),
                                 row['eri'], row['nocc'], screening='rpa')
        for label, p in orbitals(row['nocc']).items():
            w_qb, _ = row['qp'].solve_diag(p)
            w_scaled, _ = scaled.solve_diag(p)
            moved = abs(w_scaled - w_qb) * HARTREE_TO_EV
            assert moved > GATE_EV, f'{name} {label}: {moved:.3e} eV'
