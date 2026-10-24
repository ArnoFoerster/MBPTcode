"""Non-radiative transfer rates from the quantities a surface layer produces.

    k = (2 pi / hbar) |V|^2 rho_FC

with V the electronic coupling between the two states -- a spin-orbit matrix
element <S1|H_SO|T1> for intersystem crossing, a transfer integral for electron
transfer -- and rho_FC the thermally averaged Franck-Condon weighted density of
states. Everything below is Fermi's golden rule with a different rho_FC:
Marcus-Levich-Jortner treats ONE effective quantum mode explicitly and the rest
of the bath classically, Marcus treats all of it classically.

The intersystem-crossing form follows Samanta, Kim, Coropceanu and Bredas,
J. Am. Chem. Soc. 139, 4042 (2017); the quantum-mode form is Jortner, J. Chem.
Phys. 64, 4860 (1976). The surface layer supplies the adiabatic gap from two
relaxed minima, lambda, the Huang-Rhys spectrum and its first-moment effective
mode; V is an argument.

WHAT THE SCALE DEMANDS. Reverse intersystem crossing in a multiresonance emitter
runs at 10^6 s^-1 on a gap of 0.02 eV and a coupling of a few tenths of a cm^-1.
50 meV of error in the adiabatic gap is an order of magnitude in the rate, and
so is a factor of three in the coupling: these formulas are exponentially
sensitive to their inputs and are only meaningful on DIFFERENCES between states
of one molecule, where the method's absolute error cancels.

Every energy in and out of this module is in Hartree, the temperature in Kelvin
and the rate in s^-1.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import (ATOMIC_TIME_SECONDS,
                                         BOLTZMANN_HARTREE_PER_KELVIN,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         HARTREE_TO_MEV,
                                         RATE_FC_LAMBDA_RATIO_MAX,
                                         RATE_LAMBDA_REMAINDER_MIN_MEV,
                                         RATE_MLJ_N_MAX,
                                         RATE_THERMAL_WINDOW_KT,
                                         SPEED_OF_LIGHT_AU, TRIPLET_SUBLEVELS)
from src.properties.franck_condon import (FCIntegrandNotDecayed, bose,
                                          fc_weighted_dos,
                                          require_vibrational_modes,
                                          thermal_energy)


@dataclass(frozen=True)
class Minimum:
    """What the evaluation at one relaxed excited minimum hands the rate table.

    emitter, reservoir: the singlet and triplet labels ('S1', 'T2') this
        evaluation reported; a walk that followed its root off the starting
        index names it differently at the other minimum.
    energy: label -> total energy E_0 + Omega in Hartree at this geometry; at
        least the emitter and the reservoir, and T(n+1) above the reservoir for
        the T2 channel.
    socme: the spin-orbit table here, rows S0, S1, S2, ..., columns T1, T2, ...
    dv_dq, omega_promoting: |d<emitter|H_SO|reservoir>/dq_k| on dimensionless
        modes at this geometry and those modes' frequencies, both Hartree;
        None for a Condon-only crossing.
    dv_dq_paths: name -> |dV/dq_k| of each contribution, for the record only.
    emission, dipole, f_osc: the emitter's energy, transition dipole and
        oscillator strength here (the S1 minimum's are the emission's).
    e_ground: the ground state's total energy here, Hartree, on the surface
        of the rate table's e0; E_S0(here) - e0 is the energy lambda that
        guards the decay to S0 from this minimum (`fc_lambda_guard`). None
        refuses that decay.
    """
    emitter: str
    reservoir: str
    energy: dict
    socme: np.ndarray
    dv_dq: np.ndarray = None
    omega_promoting: np.ndarray = None
    dv_dq_paths: dict = None
    emission: float = None
    dipole: np.ndarray = None
    f_osc: float = None
    e_ground: float = None


#: How the state manifold's pairs take their gaps (`state_levels`).
MANIFOLD_GAP_RULES = ('stacked', 'adiabatic')


@dataclass(frozen=True)
class ManifoldGaps:
    """The declared gap rule of the rate table's state manifold.

    rule: 'stacked', every state at its spin's adiabatic energy plus its
        vertical splitting at the crossing's initial minimum; or 'adiabatic',
        every state at the energy of its own relaxed minimum.
    minima: label -> total energy at that state's own minimum, Hartree, for
        'adiabatic' (the S1 and T1 references come from the table's minima);
        None for 'stacked'. Neither field has a default.
    """
    rule: str
    minima: dict

    def __post_init__(self):
        if self.rule not in MANIFOLD_GAP_RULES:
            raise ValueError(f'gap rule {self.rule!r} not in '
                             f'{MANIFOLD_GAP_RULES}')
        if self.rule == 'stacked' and self.minima is not None:
            raise ValueError('the stacked rule reads no other minima')
        if self.rule == 'adiabatic' and not isinstance(self.minima, dict):
            raise ValueError('the adiabatic rule needs label -> energy of '
                             'each state\'s own minimum')


def _golden_rule(coupling, rho_fc):
    """(2 pi / hbar) |V|^2 rho_FC in s^-1, from Hartree and Hartree^-1.

    hbar = 1 in atomic units, so the product is already a rate in inverse
    atomic time and only the unit of time has to be converted.
    """
    return (2.0 * np.pi * abs(coupling) ** 2 * rho_fc) / ATOMIC_TIME_SECONDS


def marcus_levich_jortner_rate(coupling, delta_e, lambda_m, s_eff, omega_eff,
                               temperature, n_max=50):
    """Marcus-Levich-Jortner rate in s^-1.

        rho_FC = sum_n e^-S S^n / n! (4 pi lambda_M k_B T)^-1/2
                 exp[ -(Delta_E + lambda_M + n hbar omega_eff)^2
                      / (4 lambda_M k_B T) ]

    Each term is the classical Marcus expression for a channel that also
    deposits n quanta of the effective high-frequency mode, weighted by the
    Poisson factor that mode's Huang-Rhys factor gives it. The quantum mode is
    what makes an activationless-looking process run at all in the inverted
    region, which classical Marcus cannot do.

    coupling: |V| in Hartree (a complex spin-orbit element is fine; only its
        modulus enters).
    delta_e: Delta_E = E_final - E_initial in Hartree, the reaction energy's own
        sign convention, so an exothermic step is NEGATIVE and the barrierless
        case is Delta_E + lambda_M = 0. For reverse intersystem crossing
        T1 -> S1 that makes Delta_E = +Delta_E_ST = E(S1) - E(T1) at the TRIPLET
        minimum, positive, which is why the process is thermally activated.
    lambda_m: the CLASSICAL reorganization energy, from the low-frequency modes
        only. The modes folded into (s_eff, omega_eff) must not be counted here
        as well -- double counting them lowers the barrier twice.
    s_eff, omega_eff: the effective quantum mode. `vibronic_analysis` reports
        both: S = sum_k S_k over the high-frequency modes and
        omega_eff = sum_k S_k omega_k / sum_k S_k, their Huang-Rhys-weighted
        first moment.
    n_max: quanta retained in the Poisson sum.
    """
    kt = thermal_energy(lambda_m, temperature)
    if s_eff < 0.0 or omega_eff < 0.0:
        raise ValueError(f'the effective mode needs S >= 0 and omega >= 0, got '
                         f'S = {s_eff}, omega = {omega_eff} Hartree')
    n = np.arange(int(n_max) + 1)
    # e^-S S^n/n! by recursion rather than a gamma function, so that S = 0
    # collapses to the single classical channel exactly.
    poisson = np.empty(len(n))
    w = np.exp(-s_eff)
    for k in range(len(n)):
        poisson[k] = w
        w *= s_eff / (k + 1)
    # The Poisson mass is 1; a truncation that loses weight loses rate with it,
    # and at S of order n_max that is a factor rather than a rounding error.
    if 1.0 - poisson.sum() > np.sqrt(np.finfo(float).eps):
        raise ValueError(f'n_max = {n_max} keeps only {poisson.sum():.6f} of '
                         f'the Poisson weight at S = {s_eff}; raise n_max')
    activation = (float(delta_e) + lambda_m + n * omega_eff) ** 2
    rho_fc = ((poisson * np.exp(-activation / (4.0 * lambda_m * kt))).sum()
              / np.sqrt(4.0 * np.pi * lambda_m * kt))
    return _golden_rule(coupling, rho_fc)


def marcus_rate(coupling, delta_e, lambda_total, temperature):
    """Classical Marcus rate in s^-1: the S -> 0 limit of the expression above.

        k = (2 pi / hbar) |V|^2 (4 pi lambda k_B T)^-1/2
            exp[ -(Delta_E + lambda)^2 / (4 lambda k_B T) ]

    `lambda_total` is the WHOLE reorganization energy here, high-frequency modes
    included, because none of them is treated quantum mechanically. Same sign
    convention: Delta_E = E_final - E_initial.
    """
    kt = thermal_energy(lambda_total, temperature)
    rho_fc = (np.exp(-(float(delta_e) + lambda_total) ** 2
                     / (4.0 * lambda_total * kt))
              / np.sqrt(4.0 * np.pi * lambda_total * kt))
    return _golden_rule(coupling, rho_fc)


def spin_vibronic_rate(delta_e, v0, dv_dq, omega_promoting, rho_fc,
                       temperature):
    """Condon + Herzberg-Teller intersystem-crossing rate in s^-1.

        k = 2 pi [ |V_0|^2 rho(dE)
                   + sum_k (1/2)|dV/dq_k|^2 ( (n_k+1) rho(dE + w_k)
                                              + n_k rho(dE - w_k) ) ]

    The Condon term alone (Samanta, Kim, Coropceanu and Bredas, J. Am. Chem.
    Soc. 139, 4042 (2017)) is nearly zero for an El-Sayed-forbidden pair,
    pi-pi* against pi-pi*, and the derivative carries the process: the
    promoting mode lends the pair the symmetry the coupling needs. Its channel
    deposits one quantum of mode k, so the remaining bath absorbs dE + w_k and
    rho is evaluated at the shifted gap.

    q_k are DIMENSIONLESS normal coordinates, in which <1|q|0> = 1/sqrt 2 and
    the 1/2 above is that squared; `dv_dq` must be |dV/dq| in the same
    convention, the norm over the three spin-orbit components of the
    element's derivative (`spin_vibronic.spin_vibronic_coupling` gives it
    analytically, through the higher states). Both v0 and dv_dq are Hartree.
    The Condon and Herzberg-Teller channels add without interference, which is
    exact when the promoting modes are not displaced between the two states.

    rho_fc: callable dE -> Hartree^-1. Pass `fc_weighted_dos` bound to the
    spectrum, or a Marcus/MLJ rho if that is the comparison being made.

    Returns (k_total, k_condon, k_herzberg_teller) in s^-1.
    """
    kt = thermal_energy(1.0, temperature)
    dv = np.atleast_1d(np.asarray(dv_dq, float))
    w = np.atleast_1d(np.asarray(omega_promoting, float))
    if dv.shape != w.shape:
        raise ValueError(f'{dv.shape} derivatives against {w.shape} '
                         f'frequencies; one per promoting mode')
    n = bose(w, kt)
    k_c = _golden_rule(v0, rho_fc(float(delta_e)))
    k_ht = 0.0
    for dvk, wk, nk in zip(dv, w, n):
        k_ht += _golden_rule(dvk * np.sqrt(0.5 * (nk + 1.0)),
                             rho_fc(float(delta_e) + float(wk)))
        k_ht += _golden_rule(dvk * np.sqrt(0.5 * nk),
                             rho_fc(float(delta_e) - float(wk)))
    return k_c + k_ht, k_c, k_ht


def internal_conversion_rate(delta_e, d_modes, omega, rho_fc, temperature):
    """Internal-conversion rate in s^-1 from the derivative coupling.

        k = 2 pi sum_k omega_k |d_k|^2 [ (1/2)(n_k+1) rho(dE + w_k)
                                         + (1/2) n_k   rho(dE - w_k) ]

    THE COUPLING IS NOT A SCALAR, which is the whole difference from
    intersystem crossing. The non-adiabatic operator is a nuclear DERIVATIVE,
    H' = -sum_k d_k d/dQ_k in mass-weighted coordinates, so every mode couples
    through its own momentum and the rate is a sum over channels rather than
    one |V|^2. With dimensionless q_k = sqrt(omega_k) Q_k the operator is
    -sum_k d_k sqrt(omega_k) d/dq_k, and <v+1|d/dq|v> = -sqrt((v+1)/2) has the
    same magnitude as <v+1|q|v> -- which is why this has exactly the shape of
    the Herzberg-Teller sum in `spin_vibronic_rate` and is evaluated by it,
    with no Condon term because there is none.

    d_modes: the coupling projected on the ground-state mass-weighted modes,
        `vibronic.project_coupling` of an (natm, 3) array in 1/Bohr.
    omega: those modes' frequencies in Hartree, every one a vibration
        (`require_vibrational_modes`); nothing in this model is meaningful at
        a saddle point.
    rho_fc: callable dE -> Hartree^-1, `fc_weighted_dos` bound to the
        Huang-Rhys spectrum. The promoting mode is NOT removed from the
        accepting bath, which is the usual approximation and is good while its
        own Huang-Rhys factor is small; at a mode that both promotes and
        accepts strongly this double counts one quantum.

    Returns (k_total, k_per_mode) so the channels that carry the process can be
    read off -- internal conversion is usually one or two high-frequency
    stretches, and a rate quoted without them says nothing about why.
    """
    d = np.atleast_1d(np.asarray(d_modes, float))
    w = np.atleast_1d(np.asarray(omega, float))
    if d.shape != w.shape:
        raise ValueError(f'{d.shape} couplings against {w.shape} frequencies; '
                         f'one per mode')
    require_vibrational_modes(w, 'internal_conversion_rate')
    k_per_mode = np.zeros_like(w)
    for k in range(len(w)):
        k_per_mode[k] = spin_vibronic_rate(delta_e, 0.0,
                                           [np.sqrt(w[k]) * d[k]], [w[k]],
                                           rho_fc, temperature)[0]
    return float(k_per_mode.sum()), k_per_mode


def radiative_rate(delta_e, dipole):
    """Spontaneous-emission rate in s^-1, the Einstein A coefficient.

        k_r = 4 omega^3 |mu|^2 / (3 c^3)

    the third electronic quantity, beside the spin-orbit element and the
    derivative coupling. `solve_bse_isdf` and `solve_bse_df` return
    `transition_dipole` per root in the length gauge.

    EVALUATE IT AT THE RELAXED EXCITED-STATE GEOMETRY, with the EMISSION
    energy. The oscillator strength a vertical spectrum reports belongs to
    ABSORPTION: it is the dipole at the Franck-Condon point at the vertical
    energy, and k_r is cubic in that energy, so using it for emission
    overestimates the rate by (vertical/emission)^3 -- a factor of 1.5 at the
    0.4 eV relaxation typical of these molecules.

    delta_e: the emission energy in Hartree.
    dipole: <0|r|n> in atomic units, either the (3,) vector or its magnitude.

    Condon only. A weakly allowed transition -- which a charge-transfer TADF
    state usually is -- also emits through dmu/dQ, and that term is NOT here;
    it is the same Herzberg-Teller structure `spin_vibronic_rate` carries for
    intersystem crossing, and it needs the transition dipole's nuclear
    derivative.
    """
    mu2 = float(np.sum(np.asarray(dipole, float) ** 2))
    w = float(delta_e)
    if w <= 0:
        raise ValueError(f'emission energy {w:.6g} Ha is not positive; a '
                         'radiative rate needs the state above the one it '
                         'falls to')
    return (4.0 * w ** 3 * mu2 / (3.0 * SPEED_OF_LIGHT_AU ** 3)
            / ATOMIC_TIME_SECONDS)


def triplet_reservoir(delta_e_t2t1, temperature):
    """(p_T1, p_T2), the Boltzmann populations of a T1/T2 pair in equilibrium.

    p_T2 / p_T1 = exp(-Delta E_T2T1 / k_B T), Delta E_T2T1 = E(T2) - E(T1) the
    adiabatic gap, Hartree. Both are threefold, so no degeneracy ratio enters,
    and the two minima are taken with equal vibrational partition functions.
    Internal conversion T2 -> T1 is assumed fast against every rate out of the
    pair, which is the regime of Etherington, Gibson, Monkman and Penfold,
    Nat. Commun. 7, 13680 (2016): there T2 is a thermal intermediate, not a
    second reservoir.
    """
    kt = thermal_energy(1.0, temperature)
    k_eq = np.exp(-float(delta_e_t2t1) / kt)
    return 1.0 / (1.0 + k_eq), k_eq / (1.0 + k_eq)


def photoluminescence(k_r, k_isc, k_risc, k_nr_s=0.0, k_nr_t=0.0,
                      delta_e_t2t1=None, temperature=None, k_risc_t2=0.0,
                      k_isc_t2=0.0, k_nr_t2=0.0):
    """The observables a PL experiment actually reports, from the four rates.

    Two coupled levels, the singlet prepared by absorption and the triplet fed
    only by intersystem crossing:

        d[S]/dt = -(k_r + k_nr_s + k_isc) [S] + k_risc [T]
        d[T]/dt =  k_isc [S] - (k_nr_t + k_risc) [T]

    whose two decay constants are the PROMPT and DELAYED components of the
    fluorescence -- not k_r and not k_risc. Emission comes from the singlet in
    both, which is why delayed fluorescence has the prompt spectrum and a
    lifetime set by the triplet reservoir.

        lambda_+- = (k_S + k_T)/2 -+ sqrt((k_S - k_T)^2 + 4 k_isc k_risc)/2

    with k_S = k_r + k_nr_s + k_isc and k_T = k_nr_t + k_risc. The photon yield
    is k_r times the singlet's time integral, which the 2x2 inverse gives in
    closed form; splitting it over the two exponentials gives the prompt and
    delayed yields whose RATIO is what a time-resolved measurement reads off.

    Every rate in s^-1. Returns a dict: `k_prompt`, `k_delayed` (s^-1),
    `tau_prompt`, `tau_delayed` (s), `phi_prompt`, `phi_delayed`, `phi_total`.

    A triplet that neither decays nor returns (k_nr_t = k_risc = 0) traps every
    molecule that crosses, and the delayed component vanishes rather than
    diverging -- the branch is handled, not assumed away.

    THE T2 CHANNEL. With `delta_e_t2t1` (E(T2) - E(T1), adiabatic, Hartree) and
    `temperature`, [T] is the T1/T2 pair in internal equilibrium
    (`triplet_reservoir`), so the rates out of it are population-weighted:

        k_risc,eff = p_T1 k_risc + p_T2 k_risc_t2
        k_nr_t,eff = p_T1 k_nr_t + p_T2 k_nr_t2

    and S1 -> T2 crossing, `k_isc_t2`, feeds the same reservoir. A RISC that
    runs through T2 -- a multiresonance emitter with T2 a few k_B T above T1
    and a large <S1|H_SO|T2> -- is invisible without it. The dict then also
    carries `k_risc_eff`, `k_nr_t_eff` and `p_t2`.
    """
    k_r, k_isc, k_risc = float(k_r), float(k_isc), float(k_risc)
    if min(k_r, k_isc, k_risc, k_nr_s, k_nr_t, k_risc_t2, k_isc_t2,
           k_nr_t2) < 0:
        raise ValueError('a rate cannot be negative')
    channel = {}
    if delta_e_t2t1 is not None:
        if temperature is None:
            raise ValueError('the T2 channel is thermally activated: pass '
                             'temperature with delta_e_t2t1')
        p1, p2 = triplet_reservoir(delta_e_t2t1, temperature)
        k_risc = p1 * k_risc + p2 * float(k_risc_t2)
        k_nr_t = p1 * float(k_nr_t) + p2 * float(k_nr_t2)
        k_isc = k_isc + float(k_isc_t2)
        channel = {'k_risc_eff': k_risc, 'k_nr_t_eff': k_nr_t, 'p_t2': p2}
    elif k_risc_t2 or k_isc_t2 or k_nr_t2:
        raise ValueError('rates through T2 need delta_e_t2t1 and temperature '
                         'to weight them by the T2 population')
    k_s = k_r + float(k_nr_s) + k_isc
    k_t = float(k_nr_t) + k_risc
    disc = np.sqrt((k_s - k_t) ** 2 + 4.0 * k_isc * k_risc)
    lam_p = 0.5 * (k_s + k_t + disc)
    det = k_s * k_t - k_isc * k_risc
    if det <= 0:
        # The triplet is a perfect trap: every molecule that crosses stays, so
        # only the prompt channel emits and the singlet integral is 1/k_S.
        return {'k_prompt': k_s, 'k_delayed': 0.0,
                'tau_prompt': 1.0 / k_s if k_s > 0 else float('inf'),
                'tau_delayed': float('inf'),
                'phi_prompt': k_r / k_s if k_s > 0 else 0.0,
                'phi_delayed': 0.0,
                'phi_total': k_r / k_s if k_s > 0 else 0.0, **channel}
    # THE SMALL ROOT COMES FROM THE PRODUCT, NOT FROM THE DIFFERENCE. The two
    # roots satisfy lam_+ lam_- = det exactly, and the subtracted form
    # 0.5 (k_S + k_T - disc) loses every significant digit once
    # k_isc k_risc << k_S^2, the ordinary case for a slow reverse crossing:
    # disc then equals k_S - k_T in double precision and lam_- would be zero.
    lam_d = det / lam_p
    # THE DELAYED AMPLITUDE HAS THE SAME DISEASE ONE LEVEL UP. a_d = 1 - a_p
    # with a_p = (k_S - lam_-)/(lam_+ - lam_-) can be 1 - (1 - 1e-21) in
    # double precision, which reads 0 and breaks the sum of the two yields
    # against the closed-form total. Both roots satisfy
    # (lambda - k_S)(lambda - k_T) = k_isc k_risc, which gives lam_+ - k_S
    # without subtracting two nearly equal numbers.
    gap = lam_p - k_t
    top = (k_isc * k_risc / gap) if gap > 0 else (lam_p - k_s)
    a_d = top / (lam_p - lam_d) if lam_p > lam_d else 0.0
    a_p = 1.0 - a_d
    return {'k_prompt': lam_p, 'k_delayed': lam_d,
            'tau_prompt': 1.0 / lam_p, 'tau_delayed': 1.0 / lam_d,
            'phi_prompt': k_r * a_p / lam_p,
            'phi_delayed': k_r * a_d / lam_d,
            'phi_total': k_r * k_t / det, **channel}


def marcus_rows(v, delta_e, lam, s_sum, omega_eff, temperature):
    """The Marcus (A1) and Marcus-Levich-Jortner (A2) rates of one crossing in
    s^-1, and `marcus_mlj`, the record of the lambdas they stand on.

    The classical remainder lambda - S omega_eff is what MLJ treats
    classically. Below RATE_LAMBDA_REMAINDER_MIN_MEV the Huang-Rhys sum rule
    disagrees with the energy-based lambda, the displacement model does not
    describe the crossing, and both rows are None with the record saying
    'refused' rather than run on a clamped remainder.
    """
    lam_rule = s_sum * omega_eff
    remainder = lam - lam_rule
    record = {'lambda_energy_eV': lam * HARTREE_TO_EV,
              'lambda_sum_rule_eV': lam_rule * HARTREE_TO_EV,
              'lambda_classical_eV': remainder * HARTREE_TO_EV,
              'remainder_min_meV': RATE_LAMBDA_REMAINDER_MIN_MEV}
    if remainder * HARTREE_TO_MEV < RATE_LAMBDA_REMAINDER_MIN_MEV:
        record['status'] = 'refused'
        return {'A1_marcus': None, 'A2_mlj': None, 'marcus_mlj': record}
    record['status'] = 'computed'
    return {'A1_marcus': marcus_rate(v, delta_e, lam, temperature),
            'A2_mlj': marcus_levich_jortner_rate(v, delta_e, remainder, s_sum,
                                                 omega_eff, temperature,
                                                 n_max=RATE_MLJ_N_MAX),
            'marcus_mlj': record}


def _root_number(name):
    """'S2' -> 2, 'T1' -> 1: the 1-based root of a state label."""
    return int(name[1:])


def _rms_cm(dv):
    """Root mean square of |dV/dq_k| over the modes, in cm^-1."""
    return float(np.sqrt((np.asarray(dv, float) ** 2).mean()) * HARTREE_TO_CM)


def fc_lambda_guard(lam_rule, lam):
    """Whether the shared Franck-Condon density describes a crossing: its
    harmonic sum rule `lam_rule` against the crossing's energy-based `lam`,
    both Hartree. 'refused' when they differ by more than
    RATE_FC_LAMBDA_RATIO_MAX either way, or when `lam` is None (no energy
    to check the density against)."""
    if lam is None:
        return {'status': 'refused', 'lambda_energy_eV': None,
                'lambda_sum_rule_eV': lam_rule * HARTREE_TO_EV,
                'ratio': None, 'ratio_max': RATE_FC_LAMBDA_RATIO_MAX,
                'reason': 'no ground-state energy at the minimum'}
    if lam > 0:
        ratio = lam_rule / lam
    elif lam == 0 and lam_rule == 0:
        ratio = 1.0          # no relaxation and no displacement agree
    else:
        ratio = float('inf')  # a negative lambda is no minimum to expand at
    ok = 1.0 / RATE_FC_LAMBDA_RATIO_MAX <= ratio <= RATE_FC_LAMBDA_RATIO_MAX
    return {'status': 'computed' if ok else 'refused',
            'lambda_energy_eV': lam * HARTREE_TO_EV,
            'lambda_sum_rule_eV': lam_rule * HARTREE_TO_EV,
            'ratio': ratio, 'ratio_max': RATE_FC_LAMBDA_RATIO_MAX}


def ground_lambda(at, e0):
    """E_S0 at the minimum `at` less e0 at R0, Hartree; None without the
    minimum's ground-state energy."""
    return None if at.e_ground is None else float(at.e_ground) - float(e0)


def manifold_crossings(at, initial, final, e_initial, e_final, rho,
                       temperature, average, k_reference, gaps):
    """Every crossing between the thermally reachable states of two spin
    manifolds, read at the initial state's minimum `at`, in s^-1.

    initial, final: the labels at `at` of the two states the adiabatic gap is
        taken between (its emitter and reservoir); e_initial, e_final: those
        states' energies at their own minima, Hartree. average: the initial
        sublevel average (TRIPLET_SUBLEVELS out of a triplet, else 1).
    k_reference: the rate of the initial -> final pair itself, which carries
        the Herzberg-Teller term the others lack. gaps: the declared
        `ManifoldGaps`, where each state stands (`state_levels`).

    The shared density `rho` holds the reorganization once. The initial
    states within RATE_THERMAL_WINDOW_KT k_B T of the lowest are Boltzmann
    populated; every final state up to that window above the higher of
    e_initial and e_final is a channel, Condon only with its own
    <S_n|H_SO|T_m> at `at`. 'complete' is False where a computed root of that
    spin lies within its window bound and the next, uncomputed one could too,
    or where a state has no level under the rule: the sum is then a lower
    bound.
    """
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    window = RATE_THERMAL_WINDOW_KT * kt
    socme = np.asarray(at.socme)

    def coupling(a, b):
        sing, trip = (a, b) if a[0] == 'S' else (b, a)
        row, col = _root_number(sing), _root_number(trip) - 1
        if row >= socme.shape[0] or col >= socme.shape[1]:
            return None
        return abs(float(socme[row, col]))

    lev_i, skip_i = state_levels(at, initial, e_initial, gaps)
    lev_f, skip_f = state_levels(at, final, e_final, gaps)
    up_i = {lab: e - e_initial for lab, e in lev_i.items()}
    up_f = {lab: e - e_final for lab, e in lev_f.items()}
    low = min(up_i.values())
    # the reference state stays in even when a lower root lies below it
    starts = {lab: x for lab, x in up_i.items()
              if x - low <= window or lab == initial}
    bound = max(0.0, e_initial - e_final) + window
    ends = {lab: x for lab, x in up_f.items() if x <= bound}
    weight = {lab: np.exp(-(x - low) / kt) for lab, x in starts.items()}
    z = sum(weight.values())
    pairs, k_eff, missing = [], 0.0, []
    for a in starts:
        for b in ends:
            de = lev_f[b] - lev_i[a]
            v = coupling(a, b)
            if v is None:
                missing.append(f'{a}->{b}')
                continue
            if (a, b) == (initial, final):
                k = k_reference
            else:
                k = spin_vibronic_rate(de, v / np.sqrt(average), [], [], rho,
                                       temperature)[0]
            pairs.append({'pair': f'{a}->{b}', 'p_initial': weight[a] / z,
                          'dE_eV': de * HARTREE_TO_EV,
                          'v_cm': v * HARTREE_TO_CM,
                          'herzberg_teller': (a, b) == (initial, final),
                          'k': k})
            k_eff += weight[a] / z * k
    return {'window_kT': RATE_THERMAL_WINDOW_KT, 'gap_rule': gaps.rule,
            'populations': {lab: w / z for lab, w in weight.items()},
            'levels': {lab: lev_i[lab] for lab in starts},
            'complete': {
                'initial': max(up_i.values()) - low > window and not skip_i,
                'final': (max(up_f.values()) > bound and not missing
                          and not skip_f)},
            'without_level': skip_i + skip_f,
            'missing_couplings': missing, 'pairs': pairs, 'k_eff': k_eff}


def state_levels(at, ref, e_ref, gaps):
    """Each state of `ref`'s spin computed at `at`, with the energy it stands
    at under `gaps`, Hartree, and the labels left without one.

    'stacked': its reference state's own-minimum energy `e_ref` plus its
    vertical splitting from `ref` at `at`. 'adiabatic': the energy of its own
    relaxed minimum, `gaps.minima`, `ref` at `e_ref`; a state with no minimum
    there is left out. The adiabatic rule reads the labels as the same states
    at every minimum; a root that swaps character between them is the
    caller's to map.
    """
    labels = [lab for lab in at.energy
              if lab[0] == ref[0] and _root_number(lab) > 0]
    if gaps.rule == 'stacked':
        return {lab: e_ref + at.energy[lab] - at.energy[ref]
                for lab in labels}, []
    level = {lab: float(gaps.minima[lab]) for lab in labels
             if lab in gaps.minima and lab != ref}
    level[ref] = e_ref
    return level, [lab for lab in labels if lab not in level]


def rate_table(s1, t1, e0, omega, s_isc, temperature, lambda_env, s_ic=None,
               d_ic=None, s_t=None, manifold_gaps=None):
    """Every rate of an emitter from its two relaxed minima, in s^-1.

    s1, t1: the `Minimum` of the emitting singlet and of the triplet
        reservoir. e0: E_0 at the ground-state minimum, Hartree.
    omega: the ground-state modes' frequencies, every one a vibration
        (`require_vibrational_modes`). s_isc: the Huang-Rhys factors of the
        S1 <-> T1 displacement on those modes, the bath of both crossings and
        of the T2 channel. lambda_env: the classical reorganization of the
        surroundings, Hartree.
    s_ic, d_ic: S1 -> S0 internal conversion, the S0 <-> S1 Huang-Rhys factors
        and the derivative coupling at the S1 minimum on the mass-weighted
        modes (`vibronic.project_coupling`). s_t: the S0 <-> T1 factors of the
        triplet's decay through <S0|H_SO|T1>. A channel left None is not
        computed; its rate is None and the kinetics omit it (`pl_omits`).

    Delta_E = E_final - E_initial throughout. A CROSSING READS ITS SPIN-ORBIT
    COUPLING AND ITS DERIVATIVE AT ITS INITIAL STATE'S MINIMUM: ISC (S1 -> T1,
    T2) at the S1 minimum, RISC (T1, T2 -> S1) at the T1 minimum. Its
    reorganization energy is its ACCEPTING state's, one way: lambda_T =
    E_T1(R_S1) - E_T1(R_T1) for ISC and lambda_S = E_S1(R_T1) - E_S1(R_S1)
    for RISC. Each minimum is read at the labels its own evaluation reported.

    |V| is the sum over the three triplet sublevels. A crossing out of a
    triplet (RISC from T1 and T2, T1 -> S0) averages over its thermalized
    initial sublevels and takes |V|^2 / TRIPLET_SUBLEVELS, |dV/dq|^2 alike;
    ISC sums over the final ones and takes |V|^2. 'v_condon_cm' records the
    summed |V| of every channel.

    The Franck-Condon rows (A3, A4) of a crossing are None where its
    'fc_lambda' record says 'refused' (`fc_lambda_guard`), and so is every
    kinetics built on them. k_ic and k_nr_t1 are guarded alike against
    S0's own relaxation, E_S0 at the S1 or T1 minimum (`Minimum.e_ground`)
    less e0 ('ic_fc_lambda', 't1_s0_fc_lambda'); a refused one is None and
    refuses the kinetics ('pl_refused_by'), while one not asked for (s_ic,
    d_ic or s_t None) is omitted from them ('pl_omits').

    'manifold' (`state_manifold`) holds ISC and RISC between every thermally
    reachable singlet and triplet under the declared `manifold_gaps`
    (`ManifoldGaps`), and 'pl_manifold' the kinetics on its effective rates;
    with no declaration it is 'not declared' and pl_manifold None.
    'pl_with_t2' is the T1/T2 special case on the old reading (T2's splitting
    at the T1 minimum for both directions); 'pl' and 'pl_with_t2' stand next
    to 'pl_manifold' whatever the rule. 'pl_manifold_complete' (None without
    pl_manifold) says whether that manifold was complete (`manifold_complete`):
    under 'adiabatic' with only S1 and T1 relaxed it is not, and pl_manifold is
    the two-state kinetics at best.

    The T2 channels (k_risc_t2, k_isc_t2, pl_with_t2) read the S1 <-> T1
    density as ISC and RISC do, and are None where either channel's
    Franck-Condon row is refused ('t2_status' 'refused', 't2_refused_by').

    'numerical_flags' names the numerical events the table stands on:
    'fc_integrand_not_decayed', the densities ('S1-T1', 'S0-S1', 'S0-T1')
    whose time integrand had not decayed (`FCIntegrandNotDecayed`, warned
    about; the rates are computed on the truncated density). Empty when
    clean (the policy: refuse inputs, warn numerics).
    """
    omega = np.asarray(omega, float)
    require_vibrational_modes(omega, 'rate_table')
    dens = _Densities(omega, temperature, lambda_env)
    s_name, t_name = s1.emitter, t1.reservoir
    # the singlet row and the triplet column of each minimum's own table
    row_s, col_s = _root_number(s_name), _root_number(s1.reservoir) - 1
    row_t, col_t = _root_number(t1.emitter), _root_number(t_name) - 1
    e_s1, e_t1 = s1.energy[s_name], t1.energy[t_name]
    de_st = e_s1 - e_t1
    lam_t = s1.energy[s1.reservoir] - e_t1
    lam_s = t1.energy[t1.emitter] - e_s1
    s_k = np.asarray(s_isc, float)
    rho = dens.rho_of(s_k)
    s_sum = float(s_k.sum())
    lam_rule = dens.sum_rule(s_k)
    om_eff = float((s_k * omega).sum() / max(s_sum, 1e-30))
    out = {'temperature_K': temperature,
           'lambda_env_eV': lambda_env * HARTREE_TO_EV,
           'emitter': s_name, 'reservoir': t_name,
           'dE_ST_adiabatic_eV': de_st * HARTREE_TO_EV,
           'labels': {'S1_minimum': [s_name, s1.reservoir],
                      'T1_minimum': [t1.emitter, t_name]},
           'S1_adiabatic_eV': (e_s1 - e0) * HARTREE_TO_EV,
           'T1_adiabatic_eV': (e_t1 - e0) * HARTREE_TO_EV,
           'lambda_T_eV': lam_t * HARTREE_TO_EV,
           'lambda_S_eV': lam_s * HARTREE_TO_EV,
           'huang_rhys_S1T1_sum': s_sum,
           'lambda_sum_rule_eV': lam_rule * HARTREE_TO_EV,
           'channels': _crossing_rows(
               (('ISC S1->T1', s1, -de_st, lam_t, row_s, col_s, 1),
                ('RISC T1->S1', t1, de_st, lam_s, row_t, col_t,
                 TRIPLET_SUBLEVELS)),
               rho, lam_rule, s_sum, om_eff, temperature)}
    decays = _decay_rows(s1, t1, e0, e_s1, e_t1, col_t, omega, dens, s_ic,
                         d_ic, s_t, temperature)
    out.update((k, v) for k, v in decays.items() if k not in _DECAY_STATE)
    out['manifold'] = manifold = state_manifold(
        s1, t1, e0, e_s1, e_t1, out['channels'], rho,
        dens.rho_of(s_t, 'S0-T1') if decays['k_nr_t1'] is not None else None,
        temperature, manifold_gaps)
    t2 = _t2_rows(s1, t1, e_s1, e_t1, row_s, col_s, row_t, col_t,
                  out['channels'], rho, temperature)
    out.update((k, v) for k, v in t2.items() if k != 'de_t2t1')
    out.update(_kinetics(decays, t2, out['channels'], manifold, s_name,
                         temperature))
    out['numerical_flags'] = ({'fc_integrand_not_decayed': dens.not_decayed}
                              if dens.not_decayed else {})
    return out


class _Densities:
    """The Franck-Condon weighted densities of one rate table: one per
    displacement, on the table's modes, temperature and classical
    reorganization, with the names of those whose time integrand had not
    decayed (`FCIntegrandNotDecayed`, warned about again here)."""

    def __init__(self, omega, temperature, lambda_env):
        self.w, self.temperature, self.lambda_env = omega, temperature, lambda_env
        self.not_decayed = []

    def rho_of(self, s, name='S1-T1'):
        s = np.asarray(s, float)

        def rho(e):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', FCIntegrandNotDecayed)
                value = fc_weighted_dos(e, s, self.w, self.temperature,
                                        lambda_classical=self.lambda_env)
            for caveat in caught:
                if (issubclass(caveat.category, FCIntegrandNotDecayed)
                        and name not in self.not_decayed):
                    self.not_decayed.append(name)
                warnings.warn_explicit(caveat.message, caveat.category,
                                       caveat.filename, caveat.lineno)
            return value
        return rho

    def sum_rule(self, s):
        """The harmonic reorganization sum_k S_k omega_k of one displacement."""
        return float((np.asarray(s, float) * self.w).sum())


def _crossing_rows(crossings, rho, lam_rule, s_sum, om_eff, temperature):
    """{name: row} of the S1 <-> T1 crossings, each (name, the `Minimum` it
    reads its coupling at, Delta_E, the accepting state's lambda, the socme
    row and column, the initial sublevels averaged over) on the S1 <-> T1
    density `rho`: the Marcus rows, the Franck-Condon rows A3 and A4, and
    the guard that refuses the latter (`fc_lambda_guard`)."""
    channels = {}
    for name, at, de, lam, row, col, average in crossings:
        v = abs(float(np.asarray(at.socme)[row, col]))
        condon = at.dv_dq is None
        dv = [] if condon else np.asarray(at.dv_dq, float)
        k_ht, k_c, k_h = spin_vibronic_rate(
            de, v / np.sqrt(average), dv / np.sqrt(average),
            [] if condon else at.omega_promoting, rho, temperature)
        fc = fc_lambda_guard(lam_rule, lam)
        if fc['status'] == 'refused':
            k_ht = k_c = k_h = None
        channels[name] = {
            'v_condon_cm': v * HARTREE_TO_CM,
            'initial_sublevel_average': average,
            'dv_dq_rms_by_path_cm': {path: _rms_cm(x) for path, x in
                                     (at.dv_dq_paths or {}).items()},
            'dv_dq_rms_cm': None if condon else _rms_cm(dv),
            **marcus_rows(v / np.sqrt(average), de, lam, s_sum, om_eff,
                          temperature),
            'fc_lambda': fc,
            'A3_fc_condon': k_c, 'A4_fc_condon_ht': k_ht, 'A4_ht_part': k_h}
    return channels


#: The entries of `_decay_rows` the kinetics read and the table does not keep.
_DECAY_STATE = ('ic_fc', 't0_fc')


def _decay_rows(s1, t1, e0, e_s1, e_t1, col_t, omega, dens, s_ic, d_ic, s_t,
                temperature):
    """The decays to S0: emission at the S1 minimum (its gap and transition
    dipole), S1 -> S0 internal conversion on the S0 <-> S1 displacement and
    T1 -> S0 through <S0|H_SO|T1> on the S0 <-> T1 one, each guarded by S0's
    own relaxation at that minimum; plus the guards' records the kinetics
    read (`_DECAY_STATE`)."""
    k_r = None
    if s1.emission is not None:
        k_r = radiative_rate(s1.emission, s1.dipole)
    k_ic, top, ic_fc = None, None, None
    if s_ic is not None and d_ic is not None:
        ic_fc = fc_lambda_guard(dens.sum_rule(s_ic), ground_lambda(s1, e0))
        if ic_fc['status'] == 'computed':
            k_ic, per_mode = internal_conversion_rate(
                e0 - e_s1, d_ic, omega, dens.rho_of(s_ic, 'S0-S1'),
                temperature)
            top = [float(omega[k] * HARTREE_TO_CM)
                   for k in np.argsort(-per_mode)[:3]]
    v_t0 = abs(float(np.asarray(t1.socme)[0, col_t]))
    k_nr_t, t0_fc = None, None
    if s_t is not None:
        t0_fc = fc_lambda_guard(dens.sum_rule(s_t), ground_lambda(t1, e0))
        if t0_fc['status'] == 'computed':
            k_nr_t = spin_vibronic_rate(
                e0 - e_t1, v_t0 / np.sqrt(TRIPLET_SUBLEVELS), [], [],
                dens.rho_of(s_t, 'S0-T1'), temperature)[0]
    return dict(emission_eV=(None if s1.emission is None
                             else s1.emission * HARTREE_TO_EV),
                f_emission=s1.f_osc, k_r=k_r,
                k_ic=k_ic, k_ic_modes_cm=top, ic_fc_lambda=ic_fc,
                k_nr_t1=k_nr_t, t1_s0_fc_lambda=t0_fc,
                v_s0t1_cm=v_t0 * HARTREE_TO_CM, ic_fc=ic_fc, t0_fc=t0_fc)


def _t2_rows(s1, t1, e_s1, e_t1, row_s, col_s, row_t, col_t, channels, rho,
             temperature):
    """T2 at the T1 minimum (vertically above T1 there), on the S1 <-> T1
    Franck-Condon spectrum: the thermally activated channel, refused with
    either crossing's Franck-Condon row; 'de_t2t1' (Hartree) is what the
    kinetics read."""
    t2_name = f'T{col_t + 2}'
    de_t2t1 = v_s1t2_t = k_risc_t2 = k_isc_t2 = None
    t2_refused = [name for name, ch in channels.items()
                  if ch['fc_lambda']['status'] == 'refused']
    if t2_name in t1.energy:
        de_t2t1 = t1.energy[t2_name] - e_t1
        e_t2 = e_t1 + de_t2t1
        v_s1t2_t = abs(float(np.asarray(t1.socme)[row_t, col_t + 1]))
        v_s1t2_s = abs(float(np.asarray(s1.socme)[row_s, col_s + 1]))
    if t2_name in t1.energy and not t2_refused:
        k_risc_t2 = spin_vibronic_rate(
            e_s1 - e_t2, v_s1t2_t / np.sqrt(TRIPLET_SUBLEVELS), [], [], rho,
            temperature)[0]
        k_isc_t2 = spin_vibronic_rate(e_t2 - e_s1, v_s1t2_s, [], [], rho,
                                      temperature)[0]
    return dict(
        dE_T2T1_vertical_eV=(None if de_t2t1 is None
                             else de_t2t1 * HARTREE_TO_EV),
        v_s1t2_cm=None if v_s1t2_t is None else v_s1t2_t * HARTREE_TO_CM,
        k_risc_t2=k_risc_t2, k_isc_t2=k_isc_t2,
        t2_status=('absent' if t2_name not in t1.energy
                   else 'refused' if t2_refused else 'computed'),
        t2_refused_by=t2_refused, de_t2t1=de_t2t1)


def _kinetics(decays, t2, channels, manifold, s_name, temperature):
    """The kinetics a PL experiment reads, with every channel computed: the
    two-state one, with T2, and on the thermalized manifolds. A channel not
    asked for is omitted ('pl_omits'); a refused one refuses the kinetics
    ('pl_refused_by')."""
    k_r, k_ic, k_nr_t = decays['k_r'], decays['k_ic'], decays['k_nr_t1']
    ic_fc, t0_fc = decays['ic_fc'], decays['t0_fc']
    omits = [k for k, fc in (('k_ic', ic_fc), ('k_nr_t1', t0_fc))
             if fc is None]
    refused = [k for k, fc in (('k_ic', ic_fc), ('k_nr_t1', t0_fc))
               if fc is not None and fc['status'] == 'refused']
    pl = pl_t2 = pl_all = None
    isc = channels['ISC S1->T1']['A4_fc_condon_ht']
    risc = channels['RISC T1->S1']['A4_fc_condon_ht']
    if (k_r is not None and isc is not None and risc is not None
            and not refused):
        nr = {'k_nr_s': 0.0 if k_ic is None else k_ic,
              'k_nr_t': 0.0 if k_nr_t is None else k_nr_t}
        pl = photoluminescence(k_r, isc, risc, **nr)
        if t2['k_risc_t2'] is not None:
            pl_t2 = photoluminescence(k_r, isc, risc, **nr,
                                      delta_e_t2t1=t2['de_t2t1'],
                                      temperature=temperature,
                                      k_risc_t2=t2['k_risc_t2'],
                                      k_isc_t2=t2['k_isc_t2'])
    if pl is not None and manifold['status'] == 'computed':
        # the thermalized manifolds: k_r and k_IC are the emitter's alone
        p_s = manifold['ISC']['populations'][s_name]
        pl_all = photoluminescence(
            p_s * k_r, manifold['ISC']['k_eff'], manifold['RISC']['k_eff'],
            k_nr_s=p_s * nr['k_nr_s'],
            k_nr_t=(nr['k_nr_t'] if manifold['k_nr_t_eff'] is None
                    else manifold['k_nr_t_eff']))
    return dict(pl=pl, pl_with_t2=pl_t2, pl_manifold=pl_all,
                pl_manifold_complete=(None if pl_all is None
                                      else manifold_complete(manifold)),
                pl_omits=omits, pl_refused_by=refused)


def manifold_complete(manifold):
    """Whether every side of a computed `state_manifold` reached past its
    thermal window with every level and coupling it needs (`complete` of
    each side's `manifold_crossings`); False makes its kinetics a lower
    bound on the crossings, not the manifold's."""
    return all(all(manifold[side]['complete'].values())
               for side in ('ISC', 'RISC'))


def state_manifold(s1, t1, e0, e_s1, e_t1, channels, rho, rho_t0,
                   temperature, gaps):
    """ISC from every thermally populated singlet at the S1 minimum and RISC
    from every populated triplet at the T1 minimum, each into every final
    state within reach (`manifold_crossings`), and the populated triplets'
    decay to S0 through <S0|H_SO|T_m> on the S0 <-> T1 density `rho_t0`.

    A side whose S1-T1 channel is refused by `fc_lambda_guard` is refused
    whole: every pair stands on the same density. The emitter-reservoir pair
    is the channel's own A4 rate, the others are Condon. `gaps` is the
    declared `ManifoldGaps`; None computes no manifold.
    """
    if gaps is None:
        return {'status': 'not declared', 'k_nr_t_eff': None}
    out = {'status': 'computed'}
    for side, at, initial, final, e_i, e_f, name, average in (
            ('ISC', s1, s1.emitter, s1.reservoir, e_s1, e_t1, 'ISC S1->T1',
             1),
            ('RISC', t1, t1.reservoir, t1.emitter, e_t1, e_s1, 'RISC T1->S1',
             TRIPLET_SUBLEVELS)):
        k_ref = channels[name]['A4_fc_condon_ht']
        if k_ref is None:
            out[side] = {'status': 'refused', 'k_eff': None,
                         'populations': {initial: 1.0}}
            continue
        out[side] = {'status': 'computed', **manifold_crossings(
            at, initial, final, e_i, e_f, rho, temperature, average, k_ref,
            gaps)}
    k_t0 = None
    if rho_t0 is not None and out['RISC']['status'] == 'computed':
        socme = np.asarray(t1.socme)
        k_t0 = 0.0
        for lab, p in out['RISC']['populations'].items():
            col = _root_number(lab) - 1
            e_m = out['RISC']['levels'][lab]
            v = abs(float(socme[0, col]))
            k_t0 += p * spin_vibronic_rate(
                e0 - e_m, v / np.sqrt(TRIPLET_SUBLEVELS), [], [], rho_t0,
                temperature)[0]
    out['k_nr_t_eff'] = k_t0
    return out
