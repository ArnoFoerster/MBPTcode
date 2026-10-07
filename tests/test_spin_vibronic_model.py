"""Second-order spin-vibronic coupling on a model with known couplings.

A diabatic three-state model: a singlet S1 and two triplets T1, T2 a gap
Delta_TT apart, mixed by a linear vibronic coupling lambda Q along one
promoting mode, with <S1|H_SO|T1> = a and <S1|H_SO|T2> = b in the diabatic
basis. The adiabatic T1(Q) is the lower eigenvector of the 2x2 triplet block,
so everything a molecule hides -- the mixing, the coupling, the element along
the mode -- is computable here exactly.

WHAT EACH GATE IS FOR:

- the analytic derivative of Eq. (1) of `spin_vibronic` equals the finite
  difference of the adiabatic element, with the coupling taken from the
  eigenvector overlap and not from the formula it is meant to check;
- the rate from it equals Fermi's golden rule with the linear coupling summed
  over explicit vibrational states of an undisplaced promoting mode and a
  displaced accepting mode, which is the analytic second-order rate;
- the same golden rule over the EXACT adiabatic element on a grid converges to
  it as lambda / Delta_TT -> 0, so Eq. (1) is the leading term of the model and
  not merely consistent with itself;
- the trap: the norm over the S1 x {T1, T2} block, differenced over displaced
  geometries, is exactly invariant to the T1-T2 rotation and returns ZERO
  where the single element carries the whole coupling;
- with T2 far away only the first-order (direct) Herzberg-Teller term is left,
  and Eq. (1) reproduces the block-norm difference;
- the T2 channel of `photoluminescence` against the explicit three-level
  kinetics with fast T2 <-> T1 internal conversion obeying detailed balance.
"""
import numpy as np
import pytest
import scipy.linalg

from src.Base.constants import BOLTZMANN_HARTREE_PER_KELVIN
from src.properties import rates
from src.properties.spin_vibronic import (mode_derivatives,
                                          second_order_derivative)

#: The model, Hartree: T2 0.27 eV above T1, a 1100 cm^-1 promoting mode, an
#: El-Sayed-forbidden S1-T1 element and ~22 cm^-1 into T2.
GAP_TT = 0.01
LAMBDA = 1e-3
OMEGA_P = 0.005
A_SOC = np.zeros(3)
B_SOC = np.array([0.0, 3e-5, 1e-4])
#: the accepting mode and the bath
OMEGA_A = 0.007
S_A = 0.8
SIGMA = 0.002
TEMPERATURE = 300.0
#: the finite-difference step along Q (mass-weighted, masses 1)
DQ = 1e-3


def triplet_vectors(q, gap=GAP_TT, lam=LAMBDA):
    """(T1, T2) adiabatic eigenvectors in the diabatic basis, signs fixed."""
    h = np.array([[0.0, lam * q], [lam * q, gap]])
    _, vec = np.linalg.eigh(h)
    t1, t2 = vec[:, 0], vec[:, 1]
    return t1 * np.sign(t1[0]), t2 * np.sign(t2[1])


def adiabatic_element(q, direct=np.zeros(3), gap=GAP_TT, lam=LAMBDA,
                      triplet=0):
    """(3,) <S1|H_SO|T_n(q)> of the model, with a diabatic direct slope."""
    t = triplet_vectors(q, gap, lam)[triplet]
    return t[0] * (A_SOC + direct * q) + t[1] * B_SOC


def one_atom(d_q):
    """A coupling along Q as a (1, 3) Cartesian array, mode 0 along x."""
    return np.array([[d_q, 0.0, 0.0]])


def model_modes():
    """Unit masses, Cartesian modes; mode 0 is the promoting one."""
    return np.eye(3), np.ones(1), np.array([OMEGA_P, 1.0, 1.0])


def overlap_coupling(gap=GAP_TT, lam=LAMBDA):
    """d_21 = <T2|d/dQ T1> at Q = 0, by the eigenvector overlap."""
    t2 = triplet_vectors(0.0, gap, lam)[1]
    t1p = triplet_vectors(DQ, gap, lam)[0]
    t1m = triplet_vectors(-DQ, gap, lam)[0]
    return float(t2 @ (t1p - t1m) / (2 * DQ))


def analytic_dv_dq(gap=GAP_TT, lam=LAMBDA, direct=None):
    """|dV/dq_p| by Eq. (1), from the overlap coupling."""
    d21 = overlap_coupling(gap, lam)
    v12 = adiabatic_element(0.0, gap=gap, lam=lam, triplet=1)
    dv = second_order_derivative(
        {}, {'T2': v12}, {}, {'T2': one_atom(d21)},
        direct=None if direct is None else one_atom(1.0)[..., None] * direct)
    modes, masses, omega = model_modes()
    return mode_derivatives(dv, modes, masses, omega)[1][0]


def fock_ops(n):
    """(q, a) in an n-state harmonic-oscillator basis, q = (a + a^dag)/sqrt 2."""
    a = np.diag(np.sqrt(np.arange(1, n)), 1)
    return (a + a.T) / np.sqrt(2.0), a


def bose_weights(omega, n):
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    p = np.exp(-omega * np.arange(n) / kt)
    return p / p.sum()


def golden_rule_explicit(delta_e, v_of_q, n=40):
    """2 pi sum_if P_i |<f|V(q_p)|i>|^2 G_sigma(E_f - E_i), in s^-1.

    Explicit vibrational states: the accepting mode displaced by
    sqrt(2 S_A) between the two states, the promoting mode undisplaced, the
    coupling an arbitrary function of the promoting coordinate given as its
    matrix in the oscillator basis. Final minus initial electronic energy is
    delta_e.
    """
    q, a = fock_ops(n)
    disp = scipy.linalg.expm(np.sqrt(S_A) * (a.T - a))   # q -> q - sqrt(2 S)
    fc = np.abs(disp) ** 2                                # |<f|D|i>|^2
    vq = v_of_q(q)                                        # (3, n, n)
    vib2 = (np.abs(vq) ** 2).sum(axis=0)                  # sum over eta
    pa, pp = bose_weights(OMEGA_A, n), bose_weights(OMEGA_P, n)
    v = np.arange(n)
    # energy conservation delta_e + (f_a - i_a) w_a + (f_p - i_p) w_p = 0
    de = (delta_e + (v[:, None, None, None] - v[None, :, None, None]) * OMEGA_A
          + (v[None, None, :, None] - v[None, None, None, :]) * OMEGA_P)
    g = np.exp(-0.5 * (de / SIGMA) ** 2) / (np.sqrt(2 * np.pi) * SIGMA)
    w = (fc[:, :, None, None] * vib2[None, None, :, :]
         * pa[None, :, None, None] * pp[None, None, None, :])
    return rates._golden_rule(1.0, float((w * g).sum()))


def rho(delta_e):
    return rates.fc_weighted_dos(delta_e, [S_A], [OMEGA_A], TEMPERATURE,
                                 broadening=SIGMA)


def test_analytic_derivative_is_the_adiabatic_element_derivative():
    """Eq. (1) with the overlap coupling against the element's difference."""
    d21 = overlap_coupling()
    assert np.isclose(d21, -LAMBDA / GAP_TT, rtol=1e-5)
    v12 = adiabatic_element(0.0, triplet=1)
    dv = second_order_derivative({}, {'T2': v12}, {}, {'T2': one_atom(d21)})
    fd = (adiabatic_element(DQ) - adiabatic_element(-DQ)) / (2 * DQ)
    assert np.allclose(dv[0, 0], fd, rtol=1e-6, atol=1e-14)
    assert np.abs(dv[0, 1:]).max() == 0.0


def test_rate_equals_the_second_order_golden_rule():
    """spin_vibronic_rate on Eq. (1) against explicit vibrational states."""
    dv_dq = analytic_dv_dq()
    v1 = B_SOC * (-LAMBDA / GAP_TT) / np.sqrt(OMEGA_P)    # dV/dq, closed form
    assert np.isclose(dv_dq, np.linalg.norm(v1), rtol=1e-6)
    for delta_e in (-0.02, -0.005, 0.004):
        k, k_c, k_ht = rates.spin_vibronic_rate(
            delta_e, np.linalg.norm(A_SOC), [dv_dq], [OMEGA_P], rho,
            TEMPERATURE)
        ref = golden_rule_explicit(
            delta_e, lambda q: A_SOC[:, None, None] * np.eye(len(q))
            + v1[:, None, None] * q[None])
        assert k_c == 0.0
        assert np.isclose(k, ref, rtol=1e-6)


def test_exact_adiabatic_element_converges_to_it():
    """The golden rule over the exact element on a grid, lambda -> 0."""
    n = 40
    q_mat, _ = fock_ops(n)
    x, u = np.linalg.eigh(q_mat)                  # the promoting coordinate
    delta_e = -0.01
    ratio = []
    for lam in (2e-5, 1e-5):
        def exact(_q, lam=lam):
            # Q = q / sqrt(omega_p); V(Q) diagonal on the grid of q
            v = np.array([adiabatic_element(xi / np.sqrt(OMEGA_P), lam=lam)
                          for xi in x])
            return np.einsum('ik,ke,jk->eij', u, v, u)
        exact_k = golden_rule_explicit(delta_e, exact, n)
        dv_dq = analytic_dv_dq(lam=lam)
        k = rates.spin_vibronic_rate(delta_e, 0.0, [dv_dq], [OMEGA_P], rho,
                                     TEMPERATURE)[0]
        ratio.append(exact_k / k - 1.0)
    # the residual is O(lambda^2): a quarter at half the coupling
    assert abs(ratio[0]) < 1e-2
    assert np.isclose(ratio[1] / ratio[0], 0.25, rtol=0.05)


def block_norm_derivative(direct=np.zeros(3), gap=GAP_TT, lam=LAMBDA,
                          block=2):
    """|dV/dq| from the norm over S1 x T_block differenced along q,
    sqrt((B+ + B- - 2 B0) / (2 dq^2)) in the dimensionless coordinate."""
    dq = DQ * np.sqrt(OMEGA_P)

    def norm2(q):
        return sum(float((adiabatic_element(q / np.sqrt(OMEGA_P), direct,
                                            gap, lam, triplet=j) ** 2).sum())
                   for j in range(block))
    curv = norm2(dq) + norm2(-dq) - 2.0 * norm2(0.0)
    return np.sqrt(max(curv, 0.0) / (2.0 * dq ** 2))


def test_block_norm_cancels_the_mixing_the_single_element_keeps():
    """The trap: the S1 x {T1, T2} norm is invariant to the T1-T2 rotation."""
    single = block_norm_derivative(block=1)
    new = analytic_dv_dq()
    old = block_norm_derivative(block=2)
    assert np.isclose(new, single, rtol=1e-5)
    # the block norm returns the square root of the roundoff of a constant,
    # not the coupling
    assert old < 1e-3 * new
    assert not np.isclose(old, single, rtol=1e-5)


def test_far_t2_leaves_the_first_order_term_of_the_old_route():
    """T2 far away: Eq. (1) plus the direct slope equals the block norm."""
    direct = np.array([2e-5, 0.0, -1e-5])
    far = 50.0
    new = analytic_dv_dq(gap=far, direct=direct)
    old = block_norm_derivative(direct, gap=far, block=2)
    assert np.isclose(new, np.linalg.norm(direct) / np.sqrt(OMEGA_P),
                      rtol=1e-4)
    assert np.isclose(new, old, rtol=1e-4)


def three_level_yield(k_r, k_isc1, k_isc2, k_risc1, k_risc2, k_nr_t1,
                      k_nr_t2, k_ic, gap, temperature):
    """phi_total from the S1/T1/T2 rate matrix, T2 <-> T1 by detailed balance."""
    k_up = k_ic * np.exp(-gap / (BOLTZMANN_HARTREE_PER_KELVIN * temperature))
    m = np.array([[k_r + k_isc1 + k_isc2, -k_risc1, -k_risc2],
                  [-k_isc1, k_risc1 + k_nr_t1 + k_up, -k_ic],
                  [-k_isc2, -k_up, k_risc2 + k_nr_t2 + k_ic]])
    pops = np.linalg.solve(m, np.array([1.0, 0.0, 0.0]))
    return k_r * pops[0], np.sort(np.linalg.eigvals(m).real)


def test_t2_channel_against_three_level_kinetics():
    """Fast T2 <-> T1 conversion: the reservoir rates are Boltzmann-weighted."""
    gap = 0.004
    kw = dict(k_r=2e6, k_isc=1e7, k_risc=1e3, k_nr_t=1e2)
    t2 = dict(k_risc_t2=5e6, k_isc_t2=3e6, k_nr_t2=1e3)
    d = rates.photoluminescence(**kw, delta_e_t2t1=gap,
                                temperature=TEMPERATURE, **t2)
    phi, lam = three_level_yield(2e6, 1e7, 3e6, 1e3, 5e6, 1e2, 1e3, 1e13, gap,
                                 TEMPERATURE)
    assert np.isclose(d['phi_total'], phi, rtol=1e-5)
    assert np.isclose(d['k_delayed'], lam[0], rtol=1e-5)
    p2 = d['p_t2']
    assert np.isclose(d['k_risc_eff'], (1 - p2) * 1e3 + p2 * 5e6, rtol=1e-14)
    # without T2 the same call is the two-level one, bit for bit
    base = rates.photoluminescence(**kw)
    assert base == rates.photoluminescence(**kw, k_risc_t2=0.0)
    assert 'p_t2' not in base


def test_t2_rates_need_its_population():
    with pytest.raises(ValueError):
        rates.photoluminescence(1e6, 1e7, 1e3, k_risc_t2=1e6)
    with pytest.raises(ValueError):
        rates.photoluminescence(1e6, 1e7, 1e3, delta_e_t2t1=0.004)
