"""The exact block fold of the dRPA correlation energy in a continuum.

`src/gradients/solvated_rpa_energy.py` evaluates three members of one
functional on ONE set of factors and ONE frequency grid: the bare interaction
(A), today's dressed one (B), and the fold F -- the solvent's single-pole
g(iw) = Omega_p^2/(w^2 + Omega_p^2) inside the logarithm with the BARE
interaction in the linear counter-term. The gates below pin it from both ends
and check the one term the other two cannot carry, the solute-solvent
dispersion.

Judge by OUTPUT: every check prints its measured number, and the asserts are at
the end of each test.

H2O/cc-pVDZ in water unless stated, seconds each:
  1. g = 1 with the dressed counter-term IS `RPAGroundStateChain`, bitwise
  2. g = 0 with the bare counter-term IS the same chain without an environment,
     on the same PCM-relaxed mean field
  3. the first-order term at constant g = 1 is the closed static sum
     sum_ia <ia|vtilde|ia> over spatial occupied-virtual pairs
  4. the dispersion term is negative, vanishes as Omega_p -> 0, and
     F(Omega_p -> infinity) = B + dispersion(g = 1)
  5. H2O in toluene: the correlation contribution to the solvation energy is
     NEGATIVE under F where today's dressed-both functional makes it positive
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import HARTREE_TO_EV, HARTREE_TO_KCAL
from src.Base.environment import NoEnvironment
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import SolventScreening
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.solvated_rpa_energy import (dispersion_energy,
                                               dynamic_screening_factor,
                                               screening_matrix,
                                               solvated_correlation_energies,
                                               solvent_plasmon_energy)
from src.gradients.space_time_adjoint import (rpa_energy_and_adjoint,
                                              three_index_ov)

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Far enough above every quadrature point (the grid tops out at 10.7 keV on
#: this system) that g(iw) is 1 to machine precision, which is the adiabatic
#: limit the identity F -> B + dispersion is stated in.
ADIABATIC_OMEGA_P = 1e8
#: Today's dressed-both functional puts water's correlation contribution to the
#: toluene solvation energy at +1.76 kcal/mol -- a continuum destabilizing a
#: neutral solute, which electrostatics alone cannot do. The fold has to land
#: on the other side of zero by more than the quadrature's own noise.
TOLUENE_SIGN_MARGIN_KCAL = 0.1
#: Ha/Bohr, absolute. The same bound the gas-phase and dressed-gauge chains are
#: gated at; what is left at this step is the four-point stencil's own O(h^4)
#: truncation on top of the fit adjoint's reproducibility floor.
FOLD_GRADIENT_TOL = 1e-6


def check(ok, label, detail=''):
    """Print a named check with its measured number and return the verdict."""
    print(f'  [{"ok" if ok else "FAIL"}] {label}' + (f'   {detail}' if detail else ''))
    return bool(ok)


def scf_factory(mol):
    """A density-fitted RHF converged tightly enough for a frozen factorization."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def solvated_chain(solvent, atom=H2O):
    """The dRPA ground-state chain of `atom` inside a named continuum."""
    mol = gto.M(atom=atom, basis=BASIS, verbose=0)
    return RPAGroundStateChain(mol, scf_factory,
                               environment=SolventScreening(mol, solvent=solvent))


def bare_twin(chain):
    """The same chain, same factorization and same quadrature, gas-phase gauge.

    It is handed the chain's own PCM-relaxed mean field, so the ONLY difference
    is which metric root builds D: this is option A on solvated orbitals, not a
    gas-phase calculation.
    """
    mol, mf = chain.mean_field()
    return RPAGroundStateChain(mol, scf_factory, mf=mf,
                               environment=NoEnvironment(),
                               factorization=chain.factorization,
                               ntau=chain.ntau,
                               nfreq=len(chain.grid.omega_points))


def static_reaction_field_sum(chain):
    """sum_ia <ia|vtilde|ia> over SPATIAL occupied-virtual pairs, in Hartree.

    The closed form of the first-order term at constant g: the Born-like
    reaction field of every transition density. Built from the density-fitting
    coefficients in the bare gauge, c = V^(-1/2) B, so it shares no arithmetic
    with the frequency quadrature it is compared against.
    """
    mol, mf = chain.mean_field()
    x_mo, _, eps, auxmol, crd, _ = chain.factors_at(mol, mf)
    environment = chain.environment_at(mol)
    V = auxmol.intor('int2c2e', aosym='s1')
    root = aux_metric_sqrt(auxmol, None, V=V)
    kernel = environment.aux_kernel(auxmol)
    whitened = np.linalg.solve(root, np.linalg.solve(root, kernel).T).T
    b_ov = three_index_ov(x_mo, chain.bare_factor(mol, auxmol, crd), eps,
                          chain.nocc)
    return float(np.einsum('pn,pq,qn->', b_ov, 0.5 * (whitened + whitened.T),
                           b_ov, optimize=True))


def correlation_solvation(solvent, atom=H2O, omega_p=None):
    """(mean field, A, B, F, dispersion) shares of the solvation energy, kcal/mol.

    The correlation shares are E_c(solvated) - E_c(gas) with the gas value a
    genuine gas-phase SCF carrying the bare interaction on its own orbitals,
    which is the partition the mean-field solvation energy is measured in.
    """
    chain = solvated_chain(solvent, atom)
    gas = RPAGroundStateChain(gto.M(atom=atom, basis=BASIS, verbose=0),
                              scf_factory)
    e_gas = gas.correlation_energy()
    res = solvated_correlation_energies(chain, omega_p=omega_p)
    return ((chain.mf0.e_tot - gas.mf0.e_tot) * HARTREE_TO_KCAL,
            (res.bare - e_gas) * HARTREE_TO_KCAL,
            (res.dressed - e_gas) * HARTREE_TO_KCAL,
            (res.fold - e_gas) * HARTREE_TO_KCAL,
            res.dispersion * HARTREE_TO_KCAL)


def test_dressed_limit_is_todays_chain():
    """g = 1 with the dressed counter-term reproduces E_c bitwise."""
    chain = solvated_chain('water')
    res = solvated_correlation_energies(chain)
    today = chain.correlation_energy()
    delta = abs(res.dressed - today)
    ok = check(delta == 0.0, 'g = 1 is RPAGroundStateChain.correlation_energy',
               f'{res.dressed:.12f} vs {today:.12f}, |diff| {delta:.1e}')
    assert delta < 1e-12
    assert ok


def test_bare_limit_is_the_undressed_chain():
    """g = 0 with the bare counter-term reproduces the undressed chain."""
    chain = solvated_chain('water')
    res = solvated_correlation_energies(chain)
    bare = bare_twin(chain).correlation_energy()
    delta = abs(res.bare - bare)
    ok = check(delta < 1e-12, 'g = 0 is the same chain in the bare gauge',
               f'{res.bare:.12f} vs {bare:.12f}, |diff| {delta:.1e}')
    # the counter-term matrix is the bare interaction in the dressed gauge
    mol, mf = chain.mean_field()
    auxmol = chain.auxmol(mol)
    environment = chain.environment_at(mol)
    n_mat = screening_matrix(auxmol, environment)
    V = auxmol.intor('int2c2e', aosym='s1')
    root = aux_metric_sqrt(auxmol, environment, V=V)
    gauged = np.linalg.solve(root, np.linalg.solve(root, V).T).T
    residual = float(np.abs(np.eye(n_mat.shape[0]) - n_mat - gauged).max())
    ok &= check(residual < 1e-10, 'I - N is V_d^(-1/2) v V_d^(-1/2)',
                f'max |diff| {residual:.1e}')
    spectrum = np.linalg.eigvalsh(n_mat)
    ok &= check(spectrum.max() < 1e-10 and spectrum.min() > -1.0,
                'N is negative semi-definite and above -1 (it screens, and '
                'v + vtilde stays positive)',
                f'eigenvalues in [{spectrum.min():.4f}, {spectrum.max():.1e}]')
    assert delta < 1e-12
    assert residual < 1e-10
    assert ok


def test_first_order_term_is_the_static_born_sum():
    """At constant g the dispersion integral closes on the static sum.

    int_0^inf dw Delta/(Delta^2 + w^2) = pi/2 per occupied-virtual pair, so the
    quadrature must reproduce sum_ia <ia|vtilde|ia> over spatial pairs -- half
    the sum over spin-orbital pairs. The agreement is limited by the frequency
    quadrature alone: the closed form is exact.
    """
    chain = solvated_chain('water')
    res = solvated_correlation_energies(chain)
    quadrature = dispersion_energy(res.traces, chain.grid, ADIABATIC_OMEGA_P)
    closed = static_reaction_field_sum(chain)
    rel = abs(quadrature - closed) / abs(closed)
    ok = check(rel < 1e-6, 'first order at g = 1 is sum_ia <ia|vtilde|ia>',
               f'{quadrature:.10f} vs {closed:.10f} Ha, relative {rel:.1e}')
    ok &= check(closed < 0.0, 'the static sum is negative (a reaction field '
                'stabilizes every transition density)',
                f'{closed * HARTREE_TO_KCAL:.3f} kcal/mol')
    assert rel < 1e-6
    assert ok


def test_dispersion_limits_and_the_fold_identity():
    """Negative, vanishing at Omega_p -> 0, and F -> B + dispersion at g = 1."""
    chain = solvated_chain('water')
    res = solvated_correlation_energies(chain)
    ok = check(res.dispersion < 0.0, 'the dispersion term is attractive',
               f'{res.dispersion * HARTREE_TO_KCAL:.3f} kcal/mol at Omega_p = '
               f'{res.omega_p * HARTREE_TO_EV:.1f} eV')

    # the pole energy enters only through g, which rises monotonically with it,
    # so the term deepens with Omega_p and goes to zero with the oscillator
    # strength the model gives the solvent
    small = [dispersion_energy(res.traces, chain.grid, w / HARTREE_TO_EV)
             for w in (1e-4, 1e-3, 1e-2)]
    monotone = all(0.0 >= small[i] > small[i + 1] for i in range(len(small) - 1))
    ok &= check(monotone and abs(small[-1]) < abs(res.dispersion),
                'it vanishes as Omega_p -> 0',
                '  '.join(f'{d:.2e}' for d in small) + ' Ha')

    adiabatic = solvated_correlation_energies(chain, omega_p=ADIABATIC_OMEGA_P)
    identity = abs(adiabatic.fold - res.dressed
                   - dispersion_energy(res.traces, chain.grid, ADIABATIC_OMEGA_P))
    ok &= check(identity < 1e-9,
                'F(Omega_p -> inf) = B + dispersion(g = 1)',
                f'residual {identity:.1e} Ha')

    between = res.dressed > res.fold > res.bare + res.dispersion - 1e-12
    ok &= check(res.fold < res.bare and res.fold < res.dressed,
                'F is below both A and B (dispersion outweighs the screening '
                'reduction)',
                f'A {res.bare:.8f}  F {res.fold:.8f}  B {res.dressed:.8f} Ha')
    ok &= check(res.beyond_first_order > 0.0,
                'what is left after the first order is the screening reduction',
                f'{res.beyond_first_order * HARTREE_TO_KCAL:+.3f} kcal/mol')
    assert res.dispersion < 0.0
    assert monotone
    assert identity < 1e-9
    assert between
    assert ok


def test_g_of_omega_is_the_single_pole_model():
    """g(iw) = Omega_p^2/(w^2 + Omega_p^2), and the tabulated pole energies."""
    omega_p = solvent_plasmon_energy('water', 'fit')
    grid = np.array([0.0, omega_p, 2.0 * omega_p])
    g = dynamic_screening_factor(grid, omega_p)
    ok = check(np.allclose(g, [1.0, 0.5, 0.2], atol=1e-14),
               'g(0) = 1, g(Omega_p) = 1/2, g(2 Omega_p) = 1/5',
               '  '.join(f'{x:.6f}' for x in g))
    fit = omega_p * HARTREE_TO_EV
    f_sum = solvent_plasmon_energy('water', 'f_sum') * HARTREE_TO_EV
    ok &= check(fit < f_sum, 'water: the visible-UV fit sits below the f-sum '
                'value that controls the tail', f'{fit:.1f} vs {f_sum:.1f} eV')
    with pytest.raises(KeyError):
        solvent_plasmon_energy('liquid helium')
    with pytest.raises(KeyError):
        solvent_plasmon_energy('toluene', 'fit')
    assert np.allclose(g, [1.0, 0.5, 0.2], atol=1e-14)
    assert ok


def test_toluene_solvation_sign():
    """The correlation contribution to the solvation energy changes sign.

    A PCM continuum always stabilizes a neutral charge distribution, so a
    positive electrostatics-only solvation energy is a defect and not a small
    one. Today's dressed-both functional puts +4.7 kcal/mol of correlation on
    top of a -2.9 kcal/mol mean field and leaves water destabilized by toluene.
    """
    mean_field, a, b, f, disp = correlation_solvation('toluene')
    ok = check(mean_field + b > 0.0, 'the dressed-both functional destabilizes '
               'water in toluene',
               f'{mean_field + b:+.3f} kcal/mol total, correlation {b:+.3f}')
    ok &= check(mean_field + f < -TOLUENE_SIGN_MARGIN_KCAL,
                'the fold puts it back below zero',
                f'{mean_field + f:+.3f} kcal/mol total, correlation {f:+.3f}')
    ok &= check(disp < 0.0, 'the dispersion term carries the sign change',
                f'{disp:+.3f} kcal/mol')
    print(f'    mean field {mean_field:+.3f}   dE_c: A {a:+.3f}   B {b:+.3f}   '
          f'F {f:+.3f}   (of which dispersion {disp:+.3f}) kcal/mol')
    assert mean_field + b > 0.0
    assert mean_field + f < -TOLUENE_SIGN_MARGIN_KCAL
    assert disp < 0.0
    assert ok


def test_the_new_terms_leave_the_default_gradient_alone():
    """S = I and C = I is the interaction this route has always used, so the
    reverse pass through the new branch has to reproduce it BITWISE -- a
    gradient that merely agreed to 1e-12 would be a second implementation of
    the same functional, and the fold's own gate could not tell a sign error in
    it from quadrature."""
    chain = solvated_chain('water')
    mol, mf = chain.mean_field()
    x_mo, d, eps, auxmol, _, _ = chain.factors_at(mol, mf)
    n_mat = screening_matrix(auxmol, chain.environment_at(mol))
    eye = np.eye(n_mat.shape[0])
    args = (x_mo, d, eps, chain.nocc, chain.grid)
    e0, eps0, x0, d0, fold0 = rpa_energy_and_adjoint(*args, want_grad=True)
    ok = check(fold0 is None, 'the default route carries no fold adjoint')
    # g == 1 makes S = I without taking the default branch: same functional,
    # same arithmetic, so the whole reverse pass has to be bitwise identical.
    e, eps_b, x_b, d_b, fold = rpa_energy_and_adjoint(
        *args, want_grad=True, screening=(n_mat, np.ones(chain.grid.nfreq)))
    ok &= check(e == e0 and np.array_equal(eps_b, eps0)
                and np.array_equal(x_b, x0) and np.array_equal(d_b, d0),
                'g == 1 is the default, bitwise')
    ok &= check(fold is not None, 'g == 1 still reports its own adjoint')
    # C = I is the same NUMBER by a different summation -- Tr(I c) as a full
    # contraction rather than a diagonal sum -- so its energy agrees to
    # rounding and not to the bit. The adjoints it returns are bitwise, which
    # is the half that a sign error would break.
    e, eps_b, x_b, d_b, fold = rpa_energy_and_adjoint(*args, want_grad=True,
                                                      counter_term=eye)
    ok &= check(abs(e - e0) < 1e-13 * abs(e0), 'C == I is the default energy',
                f'relative difference {abs(e - e0) / abs(e0):.2e}')
    ok &= check(np.array_equal(eps_b, eps0) and np.array_equal(x_b, x0)
                and np.array_equal(d_b, d0),
                'C == I leaves the factor adjoints bitwise')
    assert ok


def test_the_folded_gradient_matches_finite_difference():
    """dE_c/dR under the fold, against a four-point difference of the energy.

    THE GATE OF THE WHOLE TERM. N = R^-1 vtilde R^-1 moves with the geometry
    twice -- through vtilde, whose derivative is the cavity's, and through the
    dressed metric root, which D reads as well -- so a force that dropped
    either piece would still look like a force. The displaced energies rebuild
    their own PCM ground state, so this difference answers the reaction field
    everywhere it enters, the orbital response included.
    """
    chain = solvated_chain('water')
    folded = RPAGroundStateChain(chain.mol0, scf_factory, auxbasis='cc-pvdz-ri',
                                 environment=chain.environment, fold=True)
    g_fold, _, _ = folded.correlation_gradient()
    g_plain, _, _ = chain.correlation_gradient()
    x0 = folded.mol0.atom_coords()

    def energy_at(ia, k, dx):
        m = folded.mol0.copy()
        c = x0.copy()
        c[ia, k] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return folded.correlation_energy(m)

    ok, worst = True, 0.0
    for ia, k in ((0, 2), (1, 1)):
        h = 4e-3
        f = [energy_at(ia, k, s * h) for s in (1, -1, 2, -2)]
        fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
        worst = max(worst, abs(fd - g_fold[ia, k]))
        ok &= check(abs(fd - g_fold[ia, k]) < FOLD_GRADIENT_TOL,
                    f'atom {ia} component {k}',
                    f'analytic {g_fold[ia, k]:+.9f}  FD {fd:+.9f}  '
                    f'|diff| {abs(fd - g_fold[ia, k]):.2e} Ha/Bohr')
    # and the term is not a rounding correction to the force it replaces
    shift = float(np.abs(g_fold - g_plain).max())
    ok &= check(shift > 100 * worst, 'the fold moves the force it corrects',
                f'max |F_fold - F_dressed| = {shift:.2e} Ha/Bohr')
    assert worst < FOLD_GRADIENT_TOL
    assert shift > 100 * worst
    assert ok


def test_the_fold_gradient_needs_the_environments_own_derivative():
    """An environment with no nuclear derivative is refused before the sweep,
    not handed a gas-phase-shaped force: the whole new term IS the environment
    moving."""
    chain = solvated_chain('water')
    folded = RPAGroundStateChain(chain.mol0, scf_factory, auxbasis='cc-pvdz-ri',
                                 environment=chain.environment, fold=True)
    folded.environment.differentiable = False
    try:
        with pytest.raises(NotImplementedError, match='refuses forces'):
            folded.correlation_gradient()
    finally:
        del folded.environment.differentiable
    assert check(True, 'a non-differentiable environment refuses the fold force')


def test_the_solvent_carries_its_own_pole_energy():
    """Omega_p is a property of the SOLVENT, so it travels with the dielectric
    constants rather than being set per calculation, and a displaced geometry
    keeps it."""
    mol = gto.M(atom=H2O, basis='sto-3g', verbose=0)
    for name, ev in (('water', 32.5), ('toluene', 26.6),
                     ('carbon disulfide', 29.0)):
        env = SolventScreening(mol, solvent=name)
        assert env.omega_p * HARTREE_TO_EV == pytest.approx(ev, abs=0.05)
        w = np.array([0.0, env.omega_p, 1e6])
        # g(0) = 1, g at the pole = 1/2, and a transparent solvent far above it
        assert env.dynamic_factor(w) == pytest.approx([1.0, 0.5, 0.0], abs=1e-9)
    # an explicit eps has no spectrum to take a pole energy from
    anonymous = SolventScreening(mol, eps=1.78, eps_static=78.39)
    assert anonymous.omega_p is None
    assert anonymous.dynamic_factor(np.array([0.0, 5.0])) == pytest.approx(1.0)
    # and the cavity moving does not lose it
    moved = SolventScreening(mol, solvent='water').for_geometry(
        gto.M(atom=H2O.replace('0.96', '0.97'), basis='sto-3g', verbose=0))
    assert moved.omega_p == SolventScreening(mol, solvent='water').omega_p


def test_an_environment_that_does_not_respond_is_adiabatic():
    """g is part of the contract, and an environment with no dynamics of its
    own returns 1: fixed charges do not respond at any frequency."""
    from src.Base.environment import PointCharges
    w = np.array([0.0, 1.0, 100.0])
    for env in (NoEnvironment(), PointCharges([[0.0, 0.0, 3.0]], [1.0])):
        assert env.dynamic_factor(w) == pytest.approx(1.0)


def test_the_fold_is_off_by_default_and_exact_when_asked_for():
    """The rescaled logarithm and the bare counter-term are ONE choice.

    The linear term of log det(I - S c) is -Tr(S c), so a counter-term that
    does not match the logarithm's interaction leaves a first-order term of
    order Tr(vtilde P_1) uncancelled -- tens of kcal/mol. The fold makes them
    disagree deliberately, which is its content, and it stays off by default
    because the dispersion it restores belongs on the solvent-accessible
    surface rather than the electrostatic cavity.
    """
    chain = solvated_chain('water')
    folded = RPAGroundStateChain(chain.mol0, scf_factory, auxbasis='cc-pvdz-ri',
                                 environment=chain.environment, fold=True)
    ref = solvated_correlation_energies(chain)
    assert chain.correlation_energy() == ref.dressed          # bitwise
    assert folded.correlation_energy() == ref.fold            # bitwise
    # the two static members bracket any consistent member of the family;
    # the fold is outside because its counter-term is deliberately unmatched
    lo, hi = sorted((ref.bare, ref.dressed))
    assert not lo <= ref.fold <= hi


def test_refreeze_carries_the_fold_and_the_radii():
    """An optimizer refreezes at every geometry it walks to, so anything the
    rebuilt chain forgets is a functional that changed mid-walk: a folded
    surface that quietly unfolds, or a tailored grid that reverts to the atomic
    optimizer's."""
    chain = solvated_chain('water')
    folded = RPAGroundStateChain(chain.mol0, scf_factory, auxbasis='cc-pvdz-ri',
                                 environment=chain.environment, fold=True,
                                 radii=chain.factorization.radii)
    fresh = folded.refreeze(folded.mol0)
    ok = check(fresh.fold, 'the refrozen chain still folds')
    ok &= check(fresh.factorization.radii_tag == folded.factorization.radii_tag,
                'and still carries the radii it was given')
    ok &= check(not solvated_chain('water').refreeze(chain.mol0).fold,
                'while a chain that never folded does not start')
    assert ok


def test_the_gas_phase_never_sees_the_fold():
    """No environment, nothing to dress: asking for the fold changes nothing."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    plain = RPAGroundStateChain(mol, scf_factory, auxbasis='cc-pvdz-ri')
    asked = RPAGroundStateChain(mol, scf_factory, auxbasis='cc-pvdz-ri', fold=True)
    assert asked.correlation_energy() == plain.correlation_energy()
    # and a force is not refused, because no new term is active
    asked.correlation_gradient()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-q']))
