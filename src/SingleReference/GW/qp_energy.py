"""Quasiparticle energies for single-reference states.

`calc_qp_energy` is the front end. Three routes produce Sigma_c and all solve
the same equation w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w):

  casida         diagonalize (A, B), build W from the spectrum.  O(N^6), and the
                 only route carrying vertex corrections or a CC polarizability.
  imagfrequency  chi0(i.omega) by direct particle-hole summation.  O(N^4).
  space-time     chi0(i.tau) from a separable ISDF factorization.  O(N^3).
"""
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyscf import scf
from threadpoolctl import threadpool_limits

from src.Base.constants import (get_method_info, DEFAULT_BROADENING_ETA,
                                HARTREE_TO_EV)
from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_density_fitting_coefficients,
                                      get_two_electron_integrals_chemist)
from src.Base.solvent_screening import solvent_static_selfenergy
from src.SingleReference.GW.reaction_field import (
    bare_self_energy, environment_quasiparticle_shift)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.cc_polarizability import GWCCSelfEnergy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
from src.SingleReference.GW import evGW  # module import: evGW imports this file
from src.SingleReference.GW import qsGW  # module import: qsGW imports this file
from src.Solvers.qp_equation import solve_qp_equation

IMAGINARY_AXIS_MODES = ('imagfrequency', 'imag-frequency', 'space-time')

#: `mode` values the evGW loop can drive, in its own naming.
EVGW_MODES = {'casida': 'casida', 'imagfrequency': 'imagfrequency',
              'imag-frequency': 'imagfrequency', 'space-time': 'space-time'}

#: `self_consistency` values that enter the loop, and the screening each keeps:
#: evGW rebuilds P0 and W from the iterate, evGW0 keeps the mean field's.
EVGW_SCREENING = {'evgw': 'updated', 'ev': 'updated',
                  'evgw0': 'fixed', 'ev0': 'fixed'}

#: `self_consistency` values that enter the quasiparticle-self-consistent loop,
#: and the screening each keeps: qsGW rebuilds W from the rotated orbitals,
#: qsGW0 keeps the mean field's.
QSGW_SCREENING = {'qsgw': 'updated', 'qs': 'updated',
                  'qsgw0': 'fixed', 'qs0': 'fixed'}


def _qp_energy_evgw(mf, mol, mode_key, selfenergy, polarizability, state,
                    spin_channel, screening, route_kwargs):
    """Quasiparticle energies in eV from the eigenvalue-self-consistent loop.

    The loop returns the whole converged spectrum, so the requested states are
    read straight out of it: an evGW quasiparticle energy IS the fixed point,
    not a further correction applied to one. `screening` is the loop's:
    'updated' for evGW, 'fixed' for evGW0.
    """
    if str(selfenergy).upper() != 'GW' or str(polarizability).upper() != 'RPA':
        raise NotImplementedError(
            f'self_consistency=evGW drives GW@RPA only, not '
            f'selfenergy={selfenergy!r} polarizability={polarizability!r}: the '
            f'loop reinjects eigenvalues into G and P0, and a vertex or a '
            f'non-RPA screening would need its own fixed point')
    if mode_key not in EVGW_MODES:
        raise ValueError(f"mode={mode_key!r} has no evGW loop; choose one of "
                         f"{sorted(set(EVGW_MODES))}")

    is_uhf = isinstance(mf, scf.uhf.UHF)
    eps_qp, info = evGW.evgw_eigenvalues(mf, mol, mode=EVGW_MODES[mode_key],
                                         screening=screening, **route_kwargs)
    if is_uhf:
        # the loop converges both channels at once; the caller asked for one
        nocc = mf.nelec[0 if spin_channel == 'alpha' else 1]
        eps_qp = eps_qp[0 if spin_channel == 'alpha' else 1]
    else:
        nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)
    out = {p: {'GW': float(eps_qp[p]) * HARTREE_TO_EV} for p in states}
    out['evgw_info'] = info
    if not isinstance(state, list):
        return out[states[0]]['GW']
    return out


def _qp_energy_qsgw(mf, mol, mode_key, selfenergy, polarizability, state,
                    screening, route_kwargs):
    """Quasiparticle energies in eV from the quasiparticle-self-consistent loop.

    The loop returns the whole converged spectrum and its orbitals; the
    requested states are read out of the spectrum and the orbitals ride in
    `info['mo_coeff']`, since a qsGW energy without its orbital is half the
    result. `screening` is the loop's: 'updated' for qsGW, 'fixed' for qsGW0.
    """
    if str(selfenergy).upper() != 'GW' or str(polarizability).upper() != 'RPA':
        raise NotImplementedError(
            f'self_consistency=qsGW drives GW@RPA only, not '
            f'selfenergy={selfenergy!r} polarizability={polarizability!r}: the '
            f'static self-energy is built from the plain GW amplitudes')
    if mode_key != 'casida':
        raise NotImplementedError(
            f"mode={mode_key!r} has no qsGW loop: the static self-energy is a "
            f"pole sum over the Casida spectrum, so the loop runs on mode='casida'")
    eps_qp, mo_coeff, info = qsGW.qsgw_eigenvalues(mf, mol, screening=screening,
                                                   **route_kwargs)
    info['mo_coeff'] = mo_coeff
    nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)
    out = {p: {'GW': float(eps_qp[p]) * HARTREE_TO_EV} for p in states}
    out['qsgw_info'] = info
    if not isinstance(state, list):
        return out[states[0]]['GW']
    return out


def calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True,
                   eta=DEFAULT_BROADENING_ETA, state='homo',
                   spin_channel='alpha', printSpectralFunction=False,
                   dm_correction=None, tda=False, qp_solver='pole_strength',
                   nroots=8, mode='casida', self_consistency='G0W0',
                   eps_anchor=None, n_workers=None, **route_kwargs):
    """Quasiparticle energies, in eV.

    selfenergy:     'GW'/'GWGammaInf'/'PSD1'...'PSD9', or a list of these.
    polarizability: screening of the Casida states feeding Sigma -- 'RPA'
                    (Hartree only), 'BSE' (RPA-screened exchange) or 'TDHF'
                    (bare exchange). Methods with force_rpa_casida, plain GW
                    among them, always use RPA; vertex screening always uses the
                    static RPA W. 'CCSD'/'CCSDT' replaces the Casida
                    polarizability with an EOM-CC one (G0W@CC, Lewis and
                    Berkelbach), a separate spin-orbital path in cc_polarizability.py.
    mode:           how Sigma_c is built; see the module docstring. The two
                    imaginary-axis routes implement GW@RPA only and reject every
                    other combination rather than silently downgrading it.
    state:          'homo', an orbital index, or a list of them.
    dm_correction:  AO 1RDM used in place of the mean-field density in the
                    static Sigma_Hx term, e.g. a CCSD or GW 1RDM.
    tda:            solve every Casida problem with Y = 0; the static screening
                    W_aux is unaffected.
    nroots:         EE states in the Lehmann sum, CC polarizability only.
    self_consistency: 'G0W0' (default) evaluates Sigma once on the mean field's
                    own eigenvalues. 'evGW' reinjects the quasiparticle
                    energies into G and P0 until they stop moving -- every
                    eigenvalue updated, DIIS-accelerated, convergence on the
                    HOMO and LUMO, quadrature frozen at the first cycle. 'evGW0'
                    reinjects them into G alone: the mean field's P0 and W
                    stay, the poles of Sigma_c move; Casida route only. Both
                    GW@RPA only, since that is what the loop drives.
                    'qsGW' / 'qsGW0' run the quasiparticle-self-consistent loop
                    of qsGW.py on the Casida route: orbitals and eigenvalues
                    reinjected, W rebuilt (qsGW) or the mean field's kept
                    (qsGW0); a list of states returns 'qsgw_info' with the
                    orbitals. The loops' own keywords (max_cycle, tol, mixing,
                    flow, ...) pass through **route_kwargs to evgw_eigenvalues
                    and qsgw_eigenvalues.
    eps_anchor:     the eps_p that anchors w = eps_p + <Sigma_x - v_xc> +
                    Re Sigma_c(w), when it differs from the spectrum that built
                    the screening. That is the evGW case and the only one;
                    None means the two coincide. The loops set their own and
                    refuse it.
    qp_solver:      root selection. 'pole_strength' (default) returns the root
                    of largest weight Z, which deep valence and semicore states
                    need -- a Z ~ 0.03 satellite can sit closer to eps than the
                    quasiparticle. 'graphical' returns the root nearest eps;
                    they agree wherever only one root exists. 'newton' and
                    'bisection' are also accepted.
    n_workers:      threads for the per-state root scan (Casida route, also
                    inside every evGW cycle on it). The
                    scan uses a thread pool by default of OMP_NUM_THREADS
                    threads, else one per CPU in the process's affinity mask,
                    never more than that mask holds. n_workers=1 gives the
                    serial scan.
    """
    mol = mf.mol
    mode_key = str(mode).lower().replace('_', '-')
    consistency = str(self_consistency).lower()
    if eps_anchor is not None and consistency in {**EVGW_SCREENING, **QSGW_SCREENING}:
        # the loops anchor on the mean field's eigenvalues (evGW) or on none
        # (qsGW); casida_evgw_step takes an anchor for a single step
        raise NotImplementedError(
            f'self_consistency={self_consistency!r} sets its own anchor; '
            f'eps_anchor has no place in it')
    if consistency in EVGW_SCREENING:
        # the loop takes the route's knobs by name, on the Casida route through
        # its own step and on the imaginary axis by calling back into this
        # function every cycle
        route_kwargs = dict(route_kwargs, df=df, eta=eta,
                            dm_correction=dm_correction, tda=tda,
                            qp_solver=qp_solver, n_workers=n_workers)
        return _qp_energy_evgw(mf, mol, mode_key, selfenergy, polarizability,
                               state, spin_channel, EVGW_SCREENING[consistency],
                               route_kwargs)
    if consistency in QSGW_SCREENING:
        # the loop builds its own density every cycle, so a density correction
        # has nowhere to enter; the imaginary-axis knobs mean nothing to it
        if dm_correction is not None:
            raise NotImplementedError(
                'self_consistency=qsGW builds the density from its own orbitals '
                'every cycle; dm_correction has no place in it')
        route_kwargs = dict(route_kwargs, df=df, eta=eta, tda=tda,
                            n_workers=n_workers)
        return _qp_energy_qsgw(mf, mol, mode_key, selfenergy, polarizability,
                               state, QSGW_SCREENING[consistency], route_kwargs)
    if mode_key in IMAGINARY_AXIS_MODES:
        # the anchor is a named argument here and a route keyword there
        if eps_anchor is not None:
            route_kwargs = dict(route_kwargs, eps_anchor=eps_anchor)
        return _qp_energy_imaginary_axis_route(
            mf, mol, mode_key, selfenergy, polarizability, df, state,
            spin_channel, qp_solver, dm_correction, tda, route_kwargs)
    if mode_key != 'casida':
        raise ValueError(
            f"mode='{mode}'; choose 'casida', 'imagfrequency' or 'space-time'.")
    if route_kwargs:
        raise TypeError(
            f"unexpected keyword(s) {sorted(route_kwargs)} for mode='casida'")

    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = mf.nelec if is_uhf else mol.nelectron // 2
    nocc_spin = _channel(nocc, spin_channel, is_uhf)
    states = _resolve_states(state, nocc_spin)
    shift, sigma_solvent = _reaction_field_terms(mf, mol, nocc_spin, spin_channel)

    if polarizability.upper() in ('CCSD', 'CCSDT'):
        results = _qp_energy_cc_polarizability(
            mf, states, polarizability.lower(), selfenergy, eta, nroots,
            sigma_solvent, shift, spin_channel, qp_solver, is_uhf)
        if not isinstance(state, list) and not isinstance(selfenergy, list):
            return results[states[0]]['GW']
        return results

    methods = selfenergy if isinstance(selfenergy, list) else [selfenergy]
    # Fail fast on typos rather than silently falling back to GW.
    method_infos = {m: get_method_info(m) for m in methods}

    setup = _casida_setup(mf, mol, df, eta, shift, is_uhf, nocc)
    se_solver = setup['se_solver']
    spectrum = _casida_spectrum(setup['lr_solver'], nocc, polarizability,
                                setup['w_aux'], tda, method_infos, methods, is_uhf,
                                df)
    eri_w_singlet, eri_w_triplet, xc_diagonal = _channel_terms(
        mf, mol, setup, df, dm_correction, sigma_solvent, spin_channel, is_uhf,
        nocc)

    # The equation is anchored on eps_p while the solvers above screen with
    # whatever spectrum `mf` carried; evGW is the case where the two differ.
    anchor = setup['eps'] if eps_anchor is None else np.asarray(eps_anchor, float)
    anchor_spin = _channel(anchor, spin_channel, is_uhf)
    energies = qp_energies_from_spectrum(
        se_solver, nocc, spectrum, method_infos, methods, spin_channel, states,
        eri_w_singlet, eri_w_triplet, is_uhf, df, anchor_spin, xc_diagonal,
        qp_solver=qp_solver, n_workers=n_workers)
    results = {p: {m: energies[p][m] * HARTREE_TO_EV for m in methods} for p in states}

    if printSpectralFunction:
        for p_state in states:
            amps = _self_energy_amplitudes(se_solver, nocc, spectrum, method_infos,
                                           methods, spin_channel, p_state,
                                           eri_w_singlet, eri_w_triplet, is_uhf, df)
            for method in methods:
                info = method_infos[method]
                omega_val, chi_a_val, chi_b_val, omega_t_val, chi_b_t_val = amps[method]
                qp_ev = results[p_state][method]
                _print_spectral_function(se_solver, p_state, method, qp_ev, nocc,
                                         omega_val, chi_a_val, chi_b_val,
                                         omega_t_val, chi_b_t_val, spin_channel,
                                         info)

    if not isinstance(state, list) and not isinstance(selfenergy, list):
        return results[states[0]][methods[0]]
    return results


def _resolve_states(state, nocc_spin):
    """'homo', an index, or a list of them -> list of orbital indices."""
    if state == 'homo':
        return [nocc_spin - 1]
    if isinstance(state, list):
        return state
    if isinstance(state, int):
        return [state]
    raise ValueError(f"Invalid state parameter: {state}")


def _two_electron_integrals(mol, mf, df, is_uhf):
    """(df_coeff, eri): the three-index DF factors, or the full chemist ERI."""
    if not df:
        return None, get_two_electron_integrals_chemist(mol, mf,
                                                        representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    if is_uhf:
        # The alpha-beta block needs the overlap between the two MO sets.
        df_a, df_b = df_coeff
        mo_a, mo_b = mf.mo_coeff
        S_ab = mo_a.T @ mol.intor_symmetric('int1e_ovlp') @ mo_b
        df_coeff = (df_a, df_b, np.einsum('ia, pik -> pka', S_ab, df_a))
    return df_coeff, None


def _channel(x, spin_channel, is_uhf):
    """One spin channel's entry of a per-channel pair; `x` itself when restricted."""
    if not is_uhf:
        return x
    return x[0] if spin_channel == 'alpha' else x[1]


def _reaction_field_terms(mf, mol, nocc_spin, spin_channel):
    """(shift, sigma_solvent): the environment's one-body terms, None in the gas
    phase.

    Duchemin et al. Eq. (18) where it is available -- the self-polarization of
    the orbital carrying the added charge in the SCREENED reaction field, built
    once per spectrum a mean field carries, so this route and the imaginary-axis
    ones read the same array, in every evGW cycle as well; `sigma_solvent` is
    then its diagonal. `cohsex_correction` is
    the unrestricted fallback: it sums the BARE vtilde over every orbital and
    differs by 0.39 eV of quasiparticle gap on water in water.
    """
    shift = environment_quasiparticle_shift(mf, mol, nocc_spin)
    if shift is not None:
        return shift, np.diag(shift)
    sigma_solvent = solvent_static_selfenergy(mf, mol)
    if isinstance(sigma_solvent, tuple):
        sigma_solvent = sigma_solvent[0 if spin_channel == 'alpha' else 1]
    return None, sigma_solvent


def _casida_setup(mf, mol, df, eta, shift, is_uhf, nocc):
    """What the Casida route builds from the mean field before a spectrum screens.

    A dict: 'eps', the mean field's own spectrum; 'df_coeff' and 'eri', the
    three-index factors or the chemist ERI, built with the environment off
    where `shift` carries it (`bare_self_energy`); 'spin_mode'; 'lr_solver' and
    'se_solver' on that spectrum; 'w_aux', the static RPA W_aux a vertex or a
    BSE screening contracts against. None of it moves with the eigenvalues, so
    an eigenvalue-self-consistent loop builds it once and rebuilds only the
    Casida problem and the poles of Sigma_c per cycle.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    with bare_self_energy(mf, shift):
        df_coeff, eri = _two_electron_integrals(mol, mf, df, is_uhf)
    spin_mode = 'unrestricted' if is_uhf else 'restricted'
    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, eri_chemist=eri,
                                     spin_mode=spin_mode, eta=eta)
    se_solver = SelfEnergySolver(eps, df_coeff=df_coeff, eri_chemist=eri,
                                 spin_mode=spin_mode, eta=eta)
    return {'eps': eps, 'df_coeff': df_coeff, 'eri': eri, 'spin_mode': spin_mode,
            'lr_solver': lr_solver, 'se_solver': se_solver,
            'w_aux': lr_solver.static_screening_aux(nocc)}


def _channel_terms(mf, mol, setup, df, dm_correction, sigma_solvent, spin_channel,
                   is_uhf, nocc, vertex=True):
    """(eri_w_singlet, eri_w_triplet, xc_diagonal) of one spin channel.

    A vertex contracts against W: the auxiliary form under DF, the explicit
    four-index one otherwise, built only when `vertex` asks for it and None
    without. <Sigma_Hx - v_Hxc> is one matrix for the whole spectrum; built per
    state it costs a J/K and a grid build per orbital, which dominates a
    cycle that asks for every orbital.
    """
    if df:
        eri_w_singlet = eri_w_triplet = setup['w_aux']
    elif not vertex:
        eri_w_singlet = eri_w_triplet = None
    else:
        eri_w_singlet, eri_w_triplet = setup['lr_solver'].construct_4d_w_rpa(
            nocc, spin_channel)
    xc_diagonal = _static_correction(mf, mol, setup['se_solver'], dm_correction,
                                     sigma_solvent, spin_channel, is_uhf)
    return eri_w_singlet, eri_w_triplet, xc_diagonal


def _move_poles(se_solver, eps):
    """The poles of Sigma_c at `eps`; the amplitudes, the spectrum's, stay."""
    if se_solver.spin_mode == 'unrestricted':
        se_solver.eps_a, se_solver.eps_b = eps
    else:
        se_solver.eps = eps


def _casida_spectrum(lr_solver, nocc, polarizability, w_aux, tda, method_infos,
                     methods, is_uhf, df):
    """Casida excitations feeding the self-energy, as (omega, X, Y) per channel.

    Holds the S_z = 0 solution; the triplet, or the two spin-flip channels when
    unrestricted, where a vertex needs it; and the RPA solution where a method
    forces RPA screening regardless of `polarizability`.
    """
    mode = polarizability.upper()
    if mode not in ('RPA', 'BSE', 'TDHF'):
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose "
                         "'RPA', 'BSE', 'TDHF', 'CCSD', or 'CCSDT'.")
    lBSE = mode != 'RPA'
    # BSE screens exchange with the static RPA W; TDHF uses bare exchange.
    w_casida = w_aux if mode == 'BSE' else None

    out = {'lBSE': lBSE,
           'singlet': CasidaSolver(*lr_solver.build_casida_matrices(
               nocc, lBSE=lBSE, W_aux=w_casida, triplet=False)).solve(tda=tda)}

    if any(method_infos[m]['needs_triplet'] for m in methods):
        if is_uhf:
            # w_casida is an (naux,naux) DF metric; build_spin_flip's full-ERI
            # branch needs a dense (alpha,alpha|beta,beta) 4-index W instead
            # (construct_4d_w_rpa never builds that cross block), so df=False
            # falls back to BARE exchange for the spin-flip channel only --
            # fine on a stable reference (test_unrestricted_neon's 4e-1 eV
            # PSD2 tolerance already prices this in), but on a reference with
            # a genuine triplet/spin-flip instability (e.g. C2, BN) the bare
            # eigenvalue is far more negative than the screened one and can
            # send the QP root arbitrarily far off. Use df=True there.
            if lBSE and not df:
                warnings.warn(
                    "needs_triplet method on a UHF reference with df=False: "
                    "the spin-flip channel uses BARE exchange, not BSE-"
                    "screened W (see comment above) -- use df=True on any "
                    "reference that may be triplet/spin-flip unstable.",
                    stacklevel=2)
            W_sf = w_casida if df else None
            channels = [CasidaSolver(*lr_solver.build_spin_flip_casida_matrices(
                            nocc, lBSE=lBSE, W_aux=W_sf, channel=c)).solve(tda=tda)
                        for c in ('ba', 'ab')]
            out['triplet'] = tuple(zip(*channels))
        else:
            out['triplet'] = CasidaSolver(*lr_solver.build_casida_matrices(
                nocc, lBSE=lBSE, W_aux=w_casida, triplet=True)).solve(tda=tda)
    else:
        out['triplet'] = (None, None, None)

    if lBSE and any(method_infos[m]['force_rpa_casida'] for m in methods):
        out['rpa'] = CasidaSolver(
            *lr_solver.build_casida_matrices(nocc, lBSE=False)).solve(tda=tda)
    else:
        out['rpa'] = out['singlet']
    return out


def _self_energy_amplitudes(se_solver, nocc, spectrum, method_infos, methods,
                            spin_channel, p_state, eri_w_singlet, eri_w_triplet,
                            is_uhf, df):
    """Per method, the (omega, chi_a, chi_b, omega_t, chi_b_t) Sigma is built from."""
    omega_s, X_s, Y_s = spectrum['singlet']
    omega_r, X_r, Y_r = spectrum['rpa']
    lBSE = spectrum['lBSE']

    chi_a_s = se_solver.get_chi_a(nocc, X_s, Y_s, spin_channel=spin_channel,
                                  p_state=p_state)
    force_rpa = lBSE and any(method_infos[m]['force_rpa_casida'] for m in methods)
    chi_a_rpa = (se_solver.get_chi_a(nocc, X_r, Y_r, spin_channel=spin_channel,
                                     p_state=p_state) if force_rpa else None)

    out = {}
    for method in methods:
        info = method_infos[method]
        use_rpa = info['force_rpa_casida'] and lBSE
        omega_val = omega_r if use_rpa else omega_s
        chi_a_val = chi_a_rpa if use_rpa else chi_a_s

        if not info['needs_vertex']:
            out[method] = (omega_val, chi_a_val, None, None, None)
            continue

        chi_b_val = se_solver.get_chi_b_vertex(nocc, X_s, Y_s,
                                               spin_channel=spin_channel,
                                               eri_w=eri_w_singlet,
                                               p_state=p_state)
        if not info['needs_triplet']:
            out[method] = (omega_val, chi_a_val, chi_b_val, None, None)
            continue

        omega_t, X_t, Y_t = spectrum['triplet']
        if is_uhf:
            get_sf = (se_solver.get_chi_b_vertex_sf_df if df
                      else se_solver.get_chi_b_vertex_sf_full)
            chi_b_t_val = tuple(
                get_sf(nocc, X_t[i], Y_t[i], spin_channel=spin_channel,
                       channel=channel, eri_w=eri_w_triplet, p_state=p_state)
                for i, channel in enumerate(('ba', 'ab')))
        else:
            chi_b_t_val = se_solver.get_chi_b_vertex(nocc, X_t, Y_t,
                                                     spin_channel=spin_channel,
                                                     eri_w=eri_w_triplet,
                                                     p_state=p_state)
        out[method] = (omega_val, chi_a_val, chi_b_val, omega_t, chi_b_t_val)
    return out


def _static_correction(mf, mol, se_solver, dm_correction, sigma_solvent,
                       spin_channel, is_uhf):
    """<Sigma_Hx - v_Hxc>_pp for every orbital p, plus the solvent reaction
    field: one array over the spectrum.

    Zero on a Hartree-Fock reference with no density correction, where Sigma_Hx
    is already the mean-field potential. Both potentials are one AO build each
    (the exchange-correlation one on the grid, the Hartree-Fock one a J and a
    K), so they are formed once here, not once per state.
    """
    beta = is_uhf and spin_channel != 'alpha'
    nmo = (mf.mo_coeff[1 if beta else 0] if is_uhf else mf.mo_coeff).shape[1]
    xc_correction = np.zeros(nmo)
    if hasattr(mf, 'xc') or dm_correction is not None:
        dm_mf = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        dm_for_hx = dm_correction if dm_correction is not None else dm_mf
        V_Hxc = mf.get_veff(mol, dm_mf)
        if is_uhf:
            mf_hf = scf.UHF(mol)
            mo = mf.mo_coeff[1 if beta else 0]
            V_Hxc_mo = mo.T @ V_Hxc[1 if beta else 0] @ mo
        else:
            mf_hf = scf.RHF(mol)
            V_Hxc_mo = mf.mo_coeff.T @ V_Hxc @ mf.mo_coeff
        V_Hx_mo = se_solver.calculate_sigma_hx(mol, mf_hf, dm_for_hx, mf.mo_coeff)
        if is_uhf:
            V_Hx_mo = V_Hx_mo[1 if beta else 0]
        xc_correction = np.diag(V_Hx_mo) - np.diag(V_Hxc_mo)
    if sigma_solvent is not None:
        xc_correction = xc_correction + np.diag(np.asarray(sigma_solvent))
    return xc_correction


def _positive_int_env(name):
    """The environment variable `name` as a positive int, or None."""
    value = os.environ.get(name, '').strip()
    return int(value) if value.isdigit() and int(value) > 0 else None


def _resolve_workers(n_workers, n_states):
    """Threads for the per-state scan, and for the qsGW static self-energy.

    The keyword, else OMP_NUM_THREADS, the per-process thread budget BLAS reads
    too, else the CPUs in this thread's affinity mask. Never more than those
    CPUs, since the pool threads inherit the mask, and never more than there
    are states. A mask narrower than the request warns: with OMP_PROC_BIND set,
    the OpenMP runtime binds this thread to one core as pyscf loads, and the
    whole pool would share that core.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    if n_workers is None:
        n_workers = _positive_int_env('OMP_NUM_THREADS') or cpus
    n_workers = int(n_workers)
    if n_workers > cpus:
        warnings.warn(
            f'{n_workers} threads requested for a thread pool, but this thread may '
            f'run on {cpus} CPU(s) and the pool inherits that; using {cpus}. '
            f'Unset OMP_PROC_BIND if it is set.', RuntimeWarning, stacklevel=3)
    return max(1, min(n_workers, cpus, n_states))


def qp_energies_from_spectrum(se_solver, nocc, spectrum, method_infos, methods,
                              spin_channel, states, eri_w_singlet, eri_w_triplet,
                              is_uhf, df, eps_spin, xc_correction,
                              qp_solver='pole_strength', n_workers=None):
    """Quasiparticle energies, in Hartree, for every state from one Casida spectrum.

    Solves w = eps_p + xc_p + Re Sigma_pp(w) per state and method, states in a
    thread pool. The first state runs before the pool so the p-independent
    amplitude caches exist before the threads read them; inside the pool every
    BLAS is pinned to one thread (threadpoolctl) and the per-state work is
    elementwise numpy, which releases the GIL.

    Parameters
    ----------
    se_solver : SelfEnergySolver
    nocc : int or (int, int)
        Occupied-orbital count of the spin channel, or (nocc_alpha,
        nocc_beta) when `is_uhf`.
    spectrum : dict, as returned by _casida_spectrum
    method_infos : dict {str: dict}
        Per-method entry from get_method_info: vertex_mode, force_rpa_casida,
        needs_vertex, needs_triplet.
    methods : list of str
        Self-energy method names, e.g. 'GW', 'GWGammaInf', 'PSD1'...'PSD9'.
    spin_channel : {'alpha', 'beta'}
    states : sequence of int, orbital indices
    eri_w_singlet, eri_w_triplet : ndarray, shape (naux, naux) or (norb,) * 4
        Screened interaction feeding the vertex correction: the auxiliary
        form when `df`, else the 4-index tensor. The two differ only for the
        unrestricted spin-flip vertex.
    is_uhf : bool
    df : bool
        Density fitting: the auxiliary form (True) or the explicit 4-index
        ERI (False).
    eps_spin : ndarray, shape (norb,)
        The eps_p the equation is anchored on, per orbital: the spin
        channel's orbital energies, or the mean-field anchor under evGW.
    xc_correction : float or ndarray, shape (norb,)
        <Sigma_Hx - v_Hxc>_pp indexed by orbital p, as _static_correction
        returns it; a float applies to every state.
    qp_solver : str, root selection as in solve_qp_equation
    n_workers : int or None
        None takes OMP_NUM_THREADS, else the CPUs in the affinity mask; any
        value is capped at that mask (see _resolve_workers); 1 is serial.

    Returns
    -------
    qp_energies : dict {p: {method: E_qp}}
        Quasiparticle energy per state and method, in Hartree.

    Notes
    -----
    Use one `se_solver` per spectrum: its amplitude cache keys on the identity
    of the X/Y arrays, so a solver reused after an earlier spectrum was freed
    can return stale amplitudes. Memory grows with n_workers because each
    in-flight state holds its self-energy weights (a few arrays of shape
    (nexciton, norb)) and amplitudes.
    """
    states = [int(p) for p in states]
    if not states:
        return {}
    xc = np.asarray(xc_correction, dtype=float)
    if xc.ndim == 0:
        xc = np.full(np.shape(eps_spin), float(xc))
    elif xc.shape != np.shape(eps_spin):
        raise ValueError(
            f'xc_correction has shape {xc.shape}; it is indexed by orbital, '
            f'so it needs the shape of eps_spin, {np.shape(eps_spin)}')
    vect_ok = qp_solver in ('pole_strength', 'graphical')
    grid_kw = {'vectorized': True} if vect_ok else {}

    def solve_state(i):
        p = states[i]
        amps = _self_energy_amplitudes(se_solver, nocc, spectrum, method_infos, methods,
                                       spin_channel, p, eri_w_singlet, eri_w_triplet,
                                       is_uhf, df)
        out = {}
        for method in methods:
            info = method_infos[method]
            omega_val, chi_a_val, chi_b_val, omega_t_val, chi_b_t_val = amps[method]
            sigma = se_solver.self_energy_evaluator(
                p, nocc, omega_val, chi_a_val, chi_b_val,
                eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
                spin_channel=spin_channel, vertex_mode=info['vertex_mode'])
            e0, x = eps_spin[p], xc[p]
            func = lambda w, e0=e0, x=x, sigma=sigma: w - e0 - x - sigma(w)
            out[method] = solve_qp_equation(func, e0, method=qp_solver, **grid_kw)
        return p, out

    results = {}
    p0, out0 = solve_state(0)
    results[p0] = out0
    rest = range(1, len(states))
    n_workers = _resolve_workers(n_workers, len(states))
    run_parallel = n_workers > 1 and len(states) > 1
    if run_parallel:
        with threadpool_limits(limits=1, user_api='blas'):
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for p, out in pool.map(solve_state, rest):
                    results[p] = out
    else:
        for i in rest:
            p, out = solve_state(i)
            results[p] = out
    return results


def casida_evgw_step(mf, mol, screening, df=True, eta=DEFAULT_BROADENING_ETA,
                     dm_correction=None, tda=False, qp_solver='pole_strength',
                     n_workers=None, eps_anchor=None, fixed_spectrum=None,
                     fixed_rho=None):
    """The map eps -> eps_new of an eigenvalue-self-consistent loop on the Casida
    route: GW@RPA, every orbital, in Hartree, its setup built once.

    screening: 'updated' (evGW) solves the RPA Casida problem on `eps` every
        cycle; 'fixed' (evGW0) keeps the mean field's. Both put the poles of
        Sigma_c at `eps` and anchor w = eps_p + <Sigma_x - v_xc> + Re Sigma_c(w)
        on the mean field's eps_p.
    eps_anchor: the eps_p the equation is anchored on, in place of the mean
        field's eigenvalues; what a quasiparticle-self-consistent fixed point
        needs, the Kohn-Sham Fock diagonal of its rotated density.
    fixed_spectrum, fixed_rho: under 'fixed', the Casida solution (a dict as
        _casida_spectrum returns) and its DF transition density (naux,
        nexciton) to screen with, in place of the one built from `mf`; the
        amplitudes are then that density contracted with `mf`'s own DF
        factors, which is how the mean field's W meets rotated orbitals.
        Both or neither; `mf` must carry `with_df`, whose auxiliary basis
        does not move with the orbitals.
    The other keywords are `calc_qp_energy`'s.

    Built once, since none of it moves with the eigenvalues: the DF factors,
    the static W (a vertex would read it; the loop refuses one) and
    <Sigma_Hx - v_Hxc> of every spin channel. 'fixed' also keeps the one
    self-energy solver, whose amplitude cache keys on the identity of X and Y;
    'updated' takes a fresh one per spectrum for the same reason. The reaction
    field of an attached environment does move: its Delta W screens with the
    eigenvalues it is built at, so 'updated' rebuilds its term every cycle at
    the iterate, as the imaginary-axis routes do through calc_qp_energy on the
    shifted view, and 'fixed' keeps the mean field's with the rest of W.
    """
    if screening not in ('updated', 'fixed'):
        raise ValueError(f"screening={screening!r}: choose 'updated' or 'fixed'")
    if screening != 'fixed' and (fixed_spectrum is not None or fixed_rho is not None):
        raise ValueError("fixed_spectrum and fixed_rho belong to screening='fixed'")
    if (fixed_spectrum is None) != (fixed_rho is None):
        raise ValueError('fixed_spectrum and fixed_rho come together: the Casida '
                         'solution and its DF transition density')
    if fixed_rho is not None and getattr(mf, 'with_df', None) is None:
        # without with_df the factors decompose the MO-basis ERI, and their
        # auxiliary index follows mf's orbitals, not the one fixed_rho was built in
        raise NotImplementedError(
            'fixed_rho is an auxiliary-space object and needs a fixed auxiliary '
            'basis: attach one with mf.density_fit() or mf.with_df = df.DF(mol)')
    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = mf.nelec if is_uhf else mol.nelectron // 2
    channels = ('alpha', 'beta') if is_uhf else ('alpha',)
    methods = ['GW']
    method_infos = {'GW': get_method_info('GW')}

    fields = {ch: _reaction_field_terms(mf, mol, _channel(nocc, ch, is_uhf), ch)
              for ch in channels}
    setup = _casida_setup(mf, mol, df, eta, fields['alpha'][0], is_uhf, nocc)
    eps0 = np.asarray(setup['eps'], float)
    states = list(range(eps0.shape[-1]))
    # the reaction field screens with the iterate under 'updated', so there it
    # is added per cycle and left out of the terms built once
    follow = screening == 'updated' and any(f[1] is not None for f in fields.values())
    # the loop refuses every vertex, so no four-index W is built for one
    terms = {ch: _channel_terms(mf, mol, setup, df, dm_correction,
                                None if follow else fields[ch][1], ch, is_uhf, nocc,
                                vertex=False)
             for ch in channels}
    if screening == 'fixed' and fixed_spectrum is None:
        spectrum_mf = _casida_spectrum(setup['lr_solver'], nocc, 'RPA',
                                       setup['w_aux'], tda, method_infos, methods,
                                       is_uhf, df)
    elif screening == 'fixed':
        spectrum_mf = fixed_spectrum
        _, X_f, Y_f = fixed_spectrum['singlet']
        setup['se_solver'].preset_transition_density(nocc, X_f, Y_f, fixed_rho)
    anchor = eps0 if eps_anchor is None else np.asarray(eps_anchor, float)

    def step(eps):
        eps = np.asarray(eps, float)
        if screening == 'fixed':
            spectrum, se_solver = spectrum_mf, setup['se_solver']
            _move_poles(se_solver, eps)
        else:
            lr_solver = LinearResponseSolver(eps, coeff_df=setup['df_coeff'],
                                             eri_chemist=setup['eri'],
                                             spin_mode=setup['spin_mode'], eta=eta)
            spectrum = _casida_spectrum(lr_solver, nocc, 'RPA', setup['w_aux'], tda,
                                        method_infos, methods, is_uhf, df)
            se_solver = SelfEnergySolver(eps, df_coeff=setup['df_coeff'],
                                         eri_chemist=setup['eri'],
                                         spin_mode=setup['spin_mode'], eta=eta)
        view = evGW.shifted_mean_field(mf, eps) if follow else None
        out = []
        for ch in channels:
            eri_w_singlet, eri_w_triplet, xc_diagonal = terms[ch]
            if follow:
                sigma_solvent = _reaction_field_terms(
                    view, mol, _channel(nocc, ch, is_uhf), ch)[1]
                xc_diagonal = xc_diagonal + np.diag(np.asarray(sigma_solvent))
            energies = qp_energies_from_spectrum(
                se_solver, nocc, spectrum, method_infos, methods, ch, states,
                eri_w_singlet, eri_w_triplet, is_uhf, df, _channel(anchor, ch, is_uhf),
                xc_diagonal, qp_solver=qp_solver, n_workers=n_workers)
            out.append([energies[p]['GW'] for p in states])
        return np.asarray(out, float).reshape(eps0.shape)

    return step


def _print_spectral_function(se_solver, p_state, method, qp_ev, nocc, omega_val,
                             chi_a_val, chi_b_val, omega_t_val, chi_b_t_val,
                             spin_channel, info):
    """A(omega) and Sigma on a window around the quasiparticle solution."""
    print(f"\nSpectral Function for State {p_state} ({method}):", flush=True)
    print(f"{'omega (eV)':>12s} | {'A(omega)':>12s} | {'Re Sigma (eV)':>14s} | "
          f"{'Im Sigma (eV)':>14s}", flush=True)
    print("-" * 60, flush=True)
    qp_ha = qp_ev / HARTREE_TO_EV
    omega_grid = np.linspace(qp_ha - 0.5, qp_ha + 0.5, 50)
    spec, sig_re, sig_im = se_solver.calculate_spectral_function(
        p_state, omega_grid, nocc, omega_val, chi_a_val, chi_b_val,
        eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
        spin_channel=spin_channel, vertex_mode=info['vertex_mode'])
    for w_val, spec_val, re_val, im_val in zip(omega_grid, spec, sig_re, sig_im):
        print(f"{w_val * HARTREE_TO_EV:12.6f} | {spec_val:12.6f} | "
              f"{re_val * HARTREE_TO_EV:14.6f} | "
              f"{im_val * HARTREE_TO_EV:14.6f}", flush=True)


def _qp_energy_cc_polarizability(mf, states, level, selfenergy, eta, nroots,
                                 sigma_solvent, shift, spin_channel, qp_solver,
                                 is_uhf):
    """G0W@CC: the same QP equation, with Sigma_c from an EOM-CC Lehmann sum.

    A separate branch because none of the Casida machinery applies -- the CC
    route is spin-orbital, full-ERI, and takes its transition densities from
    EOM-CC rather than from X/Y.
    """
    methods = selfenergy if isinstance(selfenergy, list) else [selfenergy]
    if methods != ['GW']:
        raise NotImplementedError(
            f"CC polarizability screens the plain GW self-energy only, got {selfenergy}. "
            "The vertex-corrected self-energies (GWGammaInf/PSDn) are built from Casida "
            "amplitudes, which the EOM-CC route does not produce.")
    if is_uhf:
        raise NotImplementedError("CC polarizability is restricted (closed-shell RHF) only.")
    if hasattr(mf, 'xc'):
        raise NotImplementedError(
            "CC polarizability assumes an HF reference; a KS starting point would "
            "additionally need a v_xc correction, which this route does not build.")

    # The CC screening is built from the BARE interaction, like the Casida
    # route's: the reaction field enters once, through the Eq. (18) static
    # shift below, and a self-energy screened with v + vtilde as well would
    # count the same polarization twice.
    with bare_self_energy(mf, shift):
        solver = GWCCSelfEnergy(mf, level=level, nroots=nroots)
    spin_offset = 0 if spin_channel == 'alpha' else 1

    results = {}
    for p_state in states:
        # Spatial orbital p -> spin orbital 2p (+1 for beta), interleaved.
        p_so = 2 * p_state + spin_offset
        static_shift = (0.0 if sigma_solvent is None
                        else sigma_solvent[p_state, p_state])
        qp_ha = solver.solve_qp(p_so, eta=eta, method=qp_solver,
                                static_shift=static_shift)
        results[p_state] = {'GW': qp_ha * HARTREE_TO_EV}
    return results


def _qp_energy_imaginary_axis_route(mf, mol, mode_key, selfenergy, polarizability,
                                    df, state, spin_channel, qp_solver,
                                    dm_correction, tda, route_kwargs):
    """Dispatch to the imaginary-frequency or space-time driver.

    Both implement GW@RPA on a restricted, density-fitted reference and nothing
    else, so every unsupported combination is rejected rather than quietly
    returning a GW@RPA number under another name.
    """
    if isinstance(selfenergy, list) or str(selfenergy).upper() != 'GW':
        raise ValueError(
            f"mode='{mode_key}' implements GW only, got selfenergy={selfenergy!r}. "
            f"Vertex corrections (GWGammaInf, PSDn) need the Casida route.")
    if str(polarizability).upper() != 'RPA':
        raise ValueError(
            f"mode='{mode_key}' implements RPA screening only, got "
            f"polarizability={polarizability!r}. BSE/TDHF/CCSD need mode='casida'.")
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(f"mode='{mode_key}' is restricted-spin only.")
    if not df:
        raise ValueError(f"mode='{mode_key}' requires density fitting (df=True).")
    if tda:
        raise ValueError(f"tda has no meaning for mode='{mode_key}'; it never "
                         f"forms a Casida problem.")

    nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)

    if mode_key == 'space-time':
        # One call for the whole window: chi0, the Dyson inversion, the tau
        # sweep and <Sigma_x - v_xc> are all shared across states.
        out = solve_qp_energy_space_time(mf, mol, nocc, np.asarray(states),
                                         solver_mode=qp_solver,
                                         dm_correction=dm_correction,
                                         **route_kwargs)
    else:
        out = [solve_qp_energy_imaginary_axis(mf, mol, nocc, p,
                                              solver_mode=qp_solver,
                                              dm_correction=dm_correction,
                                              **route_kwargs)
               for p in states]

    out = [e * HARTREE_TO_EV for e in out]
    return out[0] if not isinstance(state, list) else out
