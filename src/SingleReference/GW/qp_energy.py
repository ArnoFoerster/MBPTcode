"""Quasiparticle energies for single-reference states.

`calc_qp_energy` is the front end. Three routes produce Sigma_c and all solve
the same equation w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w):

  casida         diagonalize (A, B), build W from the spectrum.  O(N^6), and the
                 only route carrying vertex corrections or a CC polarizability.
  imagfrequency  chi0(i.omega) by direct particle-hole summation.  O(N^4).
  space-time     chi0(i.tau) from a separable ISDF factorization.  O(N^3).

`mode` is the chi0 realization; `continuation` is how that chi0 reaches the
REAL axis, which is a separate approximation and the one a deep state is lost
to. The two imaginary-axis realizations offer four of them -- Thiele-Pade of
Sigma_c(i.omega), the contour deformation of `GW.contour_deformation` with its
residues from the explicit chi0(w') or from the cosh transform of proj(tau),
and the auxiliary-pole model of `GW.sum_over_poles` -- and `MODE_CONTINUATIONS`
is the table of which pairs exist.
"""
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyscf import scf

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
from src.Solvers.qp_equation import solve_qp_equation

IMAGINARY_AXIS_MODES = ('imagfrequency', 'imag-frequency', 'space-time')

#: `mode` values the evGW loop can drive, in its own naming.
EVGW_MODES = {'casida': 'casida', 'imagfrequency': 'imagfrequency',
              'imag-frequency': 'imagfrequency', 'space-time': 'space-time'}

#: The `continuation` each `mode` can run; the first entry is that mode's
#: default. A pair missing here is refused rather than approximated by a
#: neighbour: the continuation is where a deep state is lost, so substituting
#: one for another returns a different functional under the name asked for.
MODE_CONTINUATIONS = {'casida': ('spectral',),
                      'imagfrequency': ('pade', 'cd'),
                      'imag-frequency': ('pade', 'cd'),
                      'space-time': ('pade', 'cd', 'laplace', 'sop')}

#: Route keywords only the contour driver reads.
CONTOUR_KEYWORDS = frozenset({'nfreq_cd', 'w0_cd', 'e_min_below_gap',
                              'pole_offset', 'tile_gb', 'diagnostics'})

#: ... and, on top of those, the two only the pole model reads. All three
#: contour continuations read the CD frequency grid, since the pole model is
#: FITTED on `wc` sampled there.
SOP_KEYWORDS = frozenset({'n_poles', 'sop_stride'})

#: Route keywords only a Pade continuation's drivers read.
PADE_KEYWORDS = frozenset({'nfreq', 'npade', 'w0', 'grid', 'greedy',
                           'tau_target', 'freq_block', 'scratch_dir', 'extras',
                           'screen_r_cut', 'distribute', 'timings'})

#: Route keywords naming the ISDF factorization or its imaginary-time grid,
#: which no Casida route has.
ISDF_KEYWORDS = frozenset({'ntau', 'counts', 'radii', 'factors', 'auxbasis',
                          'grid_accuracy', 'sigma_x'})

#: What `solver` accepts on the Casida route. 'auto' is resolved through
#: `bse.solver_choice`, the repository's one eigensolver rule.
CASIDA_SOLVERS = ('auto', 'dense', 'davidson')


def casida_pair_count(mf, mol):
    """nocc * nvirt, summed over the spin channels the Casida blocks span."""
    eps = np.asarray(mf.mo_energy, float)
    if eps.ndim == 1:
        nocc = mol.nelectron // 2
        return nocc * (eps.shape[-1] - nocc)
    return sum(int(n) * (eps.shape[-1] - int(n)) for n in mf.nelec)


def refuse_unbuilt_casida_solver(solver, mf, mol, tda):
    """The Casida route has ONE eigensolver; 'auto' above the memory rule refuses.

    THERE IS NO MATRIX-FREE CASIDA GW. Sigma_c here is the Lehmann sum over
    EVERY neutral excitation of the (A, B) problem -- `_self_energy_amplitudes`
    contracts the whole spectrum -- while `solve_casida_davidson` returns the
    lowest `nroots` of it, so routing there would truncate the self-energy at a
    root count nobody declared and return a different functional under this
    name. 'davidson' is therefore refused outright.

    'auto' is resolved through `solver_choice`, production's one rule, and its
    'davidson' answer is the rule saying the dense (A, B) pair no longer fits
    in BSE_DENSE_MAX_GB. That is refused too rather than paid silently: the
    memory is the whole content of the rule, and mode='space-time' is the route
    that reaches those sizes.
    """
    if solver not in CASIDA_SOLVERS:
        raise ValueError(f'solver {solver!r} not in {CASIDA_SOLVERS}')
    if solver == 'dense':
        return
    n_ov = casida_pair_count(mf, mol)
    if solver == 'auto':
        # cycle: bse.py drives this dispatcher for the quasiparticle diagonal.
        # LinearResponse.bse is a parallel, not-yet-landed port; solver='auto'
        # is calc_qp_energy's default, so a hard import here would break every
        # default mode='casida' call rather than only the ones this rule was
        # meant to catch. Fall back to proceeding (dense) with a warning until
        # it lands -- solver='dense'/'davidson' name the choice explicitly and
        # need no such fallback.
        try:
            from src.SingleReference.LinearResponse.bse import solver_choice
        except ImportError:
            warnings.warn(
                "LinearResponse.bse.solver_choice is not available yet, so "
                "the dense (A, B) pair's memory rule cannot be checked for "
                "solver='auto'; proceeding as if it fit. Pass solver='dense' "
                "once you have confirmed the pair fits, to silence this.",
                RuntimeWarning, stacklevel=3)
            return
        if solver_choice(n_ov, tda) == 'dense':
            return
    reached = ('the memory rule refuses the dense (A, B) pair at '
               f'n_ov={n_ov} ({2 * int(n_ov)**2 * 8 / 1e9:.1f} GB)'
               if solver == 'auto' else "solver='davidson' was asked for")
    raise NotImplementedError(
        f"{reached}, and mode='casida' has no matrix-free solve: its Sigma_c "
        f"is a Lehmann sum over EVERY Casida root and the Davidson returns the "
        f"lowest nroots of them. Take mode='space-time', whose self-energy "
        f"never forms the spectrum at all, or solver='dense' to pay the pair.")


def _refused_keywords(continuation):
    """The route keywords this continuation does not read."""
    if continuation == 'spectral':
        return CONTOUR_KEYWORDS | SOP_KEYWORDS | PADE_KEYWORDS | ISDF_KEYWORDS
    if continuation == 'pade':
        return CONTOUR_KEYWORDS | SOP_KEYWORDS
    if continuation in ('cd', 'laplace'):
        return PADE_KEYWORDS | SOP_KEYWORDS
    return PADE_KEYWORDS


def _continuation_of(mode_key, continuation, route_kwargs, return_z):
    """The continuation `mode` will run, with the validity table enforced.

    Decided and checked before any integral is built, because the alternative
    is a converged self-energy and then a refusal. The table is
    `MODE_CONTINUATIONS`; the keyword sets above say which route reads what, so
    a keyword aimed at another continuation is an error rather than a request
    that quietly does nothing.
    """
    allowed = MODE_CONTINUATIONS[mode_key]
    name = allowed[0] if continuation is None else str(continuation).lower()
    if name not in allowed:
        known = sorted(set().union(*MODE_CONTINUATIONS.values()))
        if name not in known:
            raise ValueError(f'continuation={continuation!r}; choose one of '
                             f'{known}')
        raise ValueError(
            f"mode={mode_key!r} cannot run continuation={name!r}; it accepts "
            f"{list(allowed)}. 'spectral' is the Casida route's Lehmann sum "
            f"over a spectrum it already holds, 'laplace' and 'sop' both read "
            f"proj(tau), which only mode='space-time' builds, and 'pade' and "
            f"'cd' need an imaginary-axis chi0.")
    refused = sorted(_refused_keywords(name) & set(route_kwargs))
    if refused:
        raise TypeError(f"continuation={name!r} does not read {refused}")
    if name == 'cd' and mode_key != 'space-time':
        raise NotImplementedError(
            f"mode={mode_key!r} with continuation='cd' is not built: the "
            f"contour needs wc[nu, q] on the CD quadrature and a real-frequency "
            f"backend, and this route's chi0 comes from "
            f"`solve_rpa_screening_df` on its OWN minimax grid, which it "
            f"inverts to W and continues. Nothing of the contour falls out of "
            f"that code path, so it would be a second driver with a second "
            f"grid. mode='space-time' with continuation='cd' is the built one.")
    if return_z and name in ('pade', 'spectral'):
        raise ValueError(
            f"return_z=True is refused for continuation={name!r}. A Thiele "
            f"continuation is not differentiable -- forward mode through its "
            f"recursion divides by inverse differences approaching zero -- so "
            f"the only Z available is a difference of the continued Sigma_c, "
            f"which is noise; and the Casida route's root finder "
            f"(`Solvers.qp_equation.solve_qp_equation`) returns the root "
            f"alone, its pole strength being an internal finite difference "
            f"that only qp_solver='pole_strength' forms at all. "
            f"The three contour continuations return Z from their own "
            f"Newton slope, which is the exact derivative of the equation "
            f"solved.")
    return name


def _qp_energy_evgw(mf, mol, mode_key, selfenergy, polarizability, state,
                    spin_channel, eta, route_kwargs):
    """Quasiparticle energies in eV from the eigenvalue-self-consistent loop.

    The loop returns the whole converged spectrum, so the requested states are
    read straight out of it: an evGW quasiparticle energy IS the fixed point,
    not a further correction applied to one.
    """
    # cycle: the loop asks this dispatcher for the spectrum every cycle
    from src.SingleReference.GW.evGW import evgw_eigenvalues

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
    # Every cycle re-enters this function, so eta travels as a route keyword.
    eps_qp, info = evgw_eigenvalues(mf, mol, mode=EVGW_MODES[mode_key],
                                    eta=eta, **route_kwargs)
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


def calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True,
                   eta=DEFAULT_BROADENING_ETA, state='homo',
                   spin_channel='alpha', printSpectralFunction=False,
                   dm_correction=None, tda=False, qp_solver='pole_strength',
                   nroots=8, mode='casida', self_consistency='G0W0',
                   eps_anchor=None, n_workers=None, continuation=None,
                   return_z=False, solver='auto', **route_kwargs):
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
    continuation:   how that Sigma_c reaches the REAL axis, None taking each
                    mode's default. `MODE_CONTINUATIONS` is the table of pairs
                    that exist; a keyword another continuation reads is a
                    TypeError naming both.
                    'spectral' the Lehmann sum at omega + i.eta over the Casida
                        spectrum; mode='casida' only, and its only choice.
                    'pade'     Thiele-Pade of Sigma_c(i.omega), the default on
                        both imaginary-axis modes. Reads nfreq/npade/w0 and the
                        rest of their quadrature; returns no Z, since
                        differentiating a Thiele recursion divides by inverse
                        differences approaching zero.
                    'cd'       contour deformation: the imaginary-axis integral
                        plus the residues of the poles of G the rotation sweeps.
                        Exact for any state and O(N^4) per residue, because each
                        one evaluates W at a real frequency. mode='space-time'.
                        Reads nfreq_cd, w0_cd, e_min_below_gap, pole_offset.
                    'laplace'  the same contour with each residue's W from the
                        cosh transform of proj(tau): O(N^3) instead of O(N^4),
                        and defined only below the particle-hole gap, where
                        that transform exists. A residue the imaginary-time
                        grid does not carry is refused by name rather than
                        served from the explicit backend. mode='space-time',
                        and it reads the same keywords as 'cd'.
                    'sop'      W modelled by n_poles auxiliary poles and
                        Sigma_c closed form: no real-frequency screening at all,
                        and a refusal for the states Eq. (27) of the pole paper
                        excludes, named with their reach. mode='space-time'.
                        Reads n_poles and sop_stride on top of the CD grid,
                        which is the grid the pole model is fitted on.
    diagnostics:    a dict that receives what the contour driver decided:
                    the CD grid it resolved, the residues swept, the pole
                    guard in force, the Newton seed, the residue treatment
                    that ran and the Laplace transform's representation error
                    where it did. 'cd', 'laplace' and 'sop' only.
    return_z:       return (energy, Z) instead of the energy. The three
                    contour continuations take Z from their own Newton slope,
                    the exact derivative of the equation they solved; the other
                    two refuse it rather than return a differenced one.
    solver:         the Casida eigensolver, mode='casida' only. 'auto'
                    (default) resolves through `bse.solver_choice`, the one
                    memory rule; 'dense' pays the (A, B) pair whatever its
                    size. There is no matrix-free route here -- the self-energy
                    sums over EVERY Casida root -- so 'davidson', and 'auto'
                    above the rule, are refused by name.
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
                    HOMO and LUMO, quadrature frozen at the first cycle. GW@RPA
                    only, since that is what the loop drives.
    eps_anchor:     the eps_p that anchors w = eps_p + <Sigma_x - v_xc> +
                    Re Sigma_c(w), when it differs from the spectrum that built
                    the screening. That is the evGW case and the only one;
                    None means the two coincide.
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
                    never more than that mask holds. n_workers=1, or
                    threadpoolctl not being installed, gives the serial scan.
    """
    mol = mf.mol
    mode_key = str(mode).lower().replace('_', '-')
    if mode_key not in MODE_CONTINUATIONS:
        raise ValueError(
            f"mode='{mode}'; choose 'casida', 'imagfrequency' or 'space-time'.")
    # THE TABLE IS ENFORCED FIRST. The alternative is a converged self-energy
    # and then a refusal.
    continuation = _continuation_of(mode_key, continuation, route_kwargs,
                                    return_z)
    # The eigensolver is decided before an integral is built, like the table
    # above: the alternative is a converged screening and then a refusal.
    if mode_key == 'casida':
        refuse_unbuilt_casida_solver(solver, mf, mol, tda)
    elif solver != 'auto':
        raise TypeError(
            f"mode={mode_key!r} does not read solver={solver!r}: it builds "
            f"chi0 on the imaginary axis and has no Casida eigenproblem to "
            f"choose a solver for. mode='casida' is the only route that has "
            f"one.")
    if str(self_consistency).lower() in ('evgw', 'ev'):
        if continuation != MODE_CONTINUATIONS[mode_key][0]:
            raise ValueError(
                f"self_consistency='evGW' drives continuation="
                f"{MODE_CONTINUATIONS[mode_key][0]!r} only, not "
                f"{continuation!r}: the loop freezes its quadrature at the "
                f"first cycle and reinjects eigenvalues, and every other "
                f"continuation would have to freeze its own branch as well.")
        # each evGW cycle calls back into this function, so the scan's thread
        # count travels as a route keyword
        if n_workers is not None:
            route_kwargs = dict(route_kwargs, n_workers=n_workers)
        return _qp_energy_evgw(mf, mol, mode_key, selfenergy, polarizability,
                               state, spin_channel, eta, route_kwargs)
    if mode_key in IMAGINARY_AXIS_MODES:
        # Neither route has a broadening to set: chi0 has the real denominator
        # -2d/(d^2 + w^2) on the imaginary axis, and Sigma_c reaches the real
        # axis by Pade or by contour deformation rather than at w + i.eta.
        if eta != DEFAULT_BROADENING_ETA:
            raise ValueError(
                f"mode='{mode_key}' has no broadening: eta={eta!r} never "
                f"reaches the self-energy, so the number returned would be the "
                f"one at eta={DEFAULT_BROADENING_ETA!r} under another name. "
                f"mode='casida' is the only route eta acts on.")
        # the anchor is a named argument here and a route keyword there
        if eps_anchor is not None:
            route_kwargs = dict(route_kwargs, eps_anchor=eps_anchor)
        return _qp_energy_imaginary_axis_route(
            mf, mol, mode_key, continuation, selfenergy, polarizability, df,
            state, spin_channel, qp_solver, dm_correction, tda, return_z,
            route_kwargs)
    if route_kwargs:
        raise TypeError(
            f"unexpected keyword(s) {sorted(route_kwargs)} for mode='casida'")

    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = mf.nelec if is_uhf else mol.nelectron // 2
    nocc_spin = (nocc[0] if spin_channel == 'alpha' else nocc[1]) if is_uhf else nocc
    eps = get_orbital_energies(mf, representation='spatial')
    states = _resolve_states(state, nocc_spin)

    # The reaction field's one-body term, None in the gas phase. Duchemin et
    # al. Eq. (18) where it is available -- the self-polarization of the
    # orbital carrying the added charge in the SCREENED reaction field, built
    # once per mean field so this route and the imaginary-axis ones read the
    # same array. `cohsex_correction` is the unrestricted fallback: it sums the
    # BARE vtilde over every orbital and differs by 0.39 eV of quasiparticle
    # gap on water in water.
    shift = environment_quasiparticle_shift(mf, mol, nocc_spin)
    if shift is not None:
        sigma_solvent = np.diag(shift)
    else:
        sigma_solvent = solvent_static_selfenergy(mf, mol)
        if isinstance(sigma_solvent, tuple):
            sigma_solvent = sigma_solvent[0 if spin_channel == 'alpha' else 1]

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

    with bare_self_energy(mf, shift):
        df_coeff, eri = _two_electron_integrals(mol, mf, df, is_uhf)
    spin_mode = 'unrestricted' if is_uhf else 'restricted'
    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, eri_chemist=eri,
                                     spin_mode=spin_mode, eta=eta)
    se_solver = SelfEnergySolver(eps, df_coeff=df_coeff, eri_chemist=eri,
                                 spin_mode=spin_mode, eta=eta)
    w_aux = lr_solver.static_screening_aux(nocc)
    spectrum = _casida_spectrum(lr_solver, nocc, polarizability, w_aux, tda,
                                method_infos, methods, is_uhf, df)

    # A vertex contracts against W: the auxiliary form under DF, the explicit
    # four-index one otherwise.
    if df:
        eri_w_singlet = eri_w_triplet = w_aux
    else:
        eri_w_singlet, eri_w_triplet = lr_solver.construct_4d_w_rpa(nocc, spin_channel)

    eps_spin = (eps[0] if spin_channel == 'alpha' else eps[1]) if is_uhf else eps
    # The equation is anchored on eps_p while the solvers above screen with
    # whatever spectrum `mf` carried; evGW is the case where the two differ.
    if eps_anchor is None:
        anchor_spin = eps_spin
    else:
        anchor = np.asarray(eps_anchor, float)
        anchor_spin = ((anchor[0] if spin_channel == 'alpha' else anchor[1])
                       if is_uhf else anchor)

    # <Sigma_Hx - v_Hxc> is one matrix for the whole spectrum; built per state
    # it was 85 % of an evGW cycle that asks for every orbital.
    xc_diagonal = _static_correction(mf, mol, se_solver, dm_correction,
                                     sigma_solvent, spin_channel, is_uhf)
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
    """Threads for the per-state scan.

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
            f'{n_workers} threads requested for the QP scan, but this thread may '
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
        value is capped at that mask (see _resolve_workers); 1 is serial;
        threadpoolctl not being installed also gives the serial scan.

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
    requested_workers = n_workers
    n_workers = _resolve_workers(n_workers, len(states))
    run_parallel = n_workers > 1 and len(states) > 1
    if run_parallel:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            if requested_workers is not None:
                raise ImportError(
                    "qp_energies_from_spectrum with n_workers > 1 needs "
                    "threadpoolctl (pip install threadpoolctl); n_workers=1 "
                    "is the serial scan") from exc
            warnings.warn(
                "threadpoolctl is not installed; running the per-state QP scan "
                "serially.")
            run_parallel = False
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


def _qp_energy_imaginary_axis_route(mf, mol, mode_key, continuation, selfenergy,
                                    polarizability, df, state, spin_channel,
                                    qp_solver, dm_correction, tda, return_z,
                                    route_kwargs):
    """Dispatch to the imaginary-frequency, space-time or contour driver.

    All implement GW@RPA on a restricted, density-fitted reference and nothing
    else, so every unsupported combination is rejected rather than quietly
    returning a GW@RPA number under another name. `eta` is one of them and is
    refused by the caller, since none of them broadens anything.

    The three drivers differ in the CHI0 they build and in the continuation
    that takes it to the real axis; `continuation` has already been checked
    against `MODE_CONTINUATIONS` and against the keywords it reads.
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

    if continuation in ('cd', 'laplace', 'sop'):
        # Deferred: contour_deformation.py's chi0 comes from
        # LinearResponse.space_time's polarizability_projected_sweep /
        # three_index_ov / three_index_slice / owned_frequency_blocks and
        # LinearResponse.imaginary_frequency's frequency_factor, which are a
        # parallel port and may not exist yet -- keeping this import out of
        # the module top level means mode='casida'/'imagfrequency'/'space-time'
        # with continuation='pade' stay unaffected either way.
        from src.SingleReference.GW.contour_deformation import \
            solve_qp_energy_contour

        # The two contours and the pole model share every step but the last:
        # one factorization, one proj(tau), one contour grid, one exchange
        # build.
        if qp_solver != 'pole_strength':
            raise ValueError(
                f"continuation={continuation!r} solves the quasiparticle "
                f"equation with the guarded Newton of `Solvers.qp_equation`, "
                f"which selects no root: qp_solver={qp_solver!r} would never "
                f"reach it. continuation='pade' is where the root selector "
                f"acts.")
        diagnostics_out = route_kwargs.pop('diagnostics', None)
        out, z, diagnostics = solve_qp_energy_contour(
            mf, mol, nocc, np.asarray(states), continuation=continuation,
            dm_correction=dm_correction, **route_kwargs)
        if diagnostics_out is not None:
            diagnostics_out.update(diagnostics)
        out, z = list(out), list(z)
    elif mode_key == 'space-time':
        # One call for the whole window: chi0, the Dyson inversion, the tau
        # sweep and <Sigma_x - v_xc> are all shared across states.
        out = solve_qp_energy_space_time(mf, mol, nocc, np.asarray(states),
                                         solver_mode=qp_solver,
                                         dm_correction=dm_correction,
                                         **route_kwargs)
        out, z = [e * HARTREE_TO_EV for e in out], None
    else:
        out = [solve_qp_energy_imaginary_axis(mf, mol, nocc, p,
                                              solver_mode=qp_solver,
                                              dm_correction=dm_correction,
                                              **route_kwargs)
               for p in states]
        out, z = [e * HARTREE_TO_EV for e in out], None

    if not isinstance(state, list):
        return (out[0], z[0]) if return_z else out[0]
    return (out, z) if return_z else out
