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

`self_consistency` wraps a route in a loop instead of evaluating Sigma once:
evGW/evGW0 (`GW.evGW`, eigenvalues reinjected) on any mode, and qsGW/qsGW0
(`GW.qsGW`, orbitals and eigenvalues reinjected) on mode='casida'. A loop runs
its mode's default continuation only; `calc_qp_energy`'s docstring gives the
order the dispatch decides in.
"""

import numpy as np
from pyscf import scf

from src.Base.numerics.constants import BYTES_PER_GB, DEFAULT_BROADENING_ETA, HARTREE_TO_EV
from src.SingleReference.GW.self_energy import get_method_info
from src.Base.pyscf.pyscf_interface import (require_closed_shell_or_unrestricted,
                                            unrestricted_reference_diagnostics,
                                            get_orbital_energies,
                                            get_density_fitting_coefficients,
                                            get_two_electron_integrals_chemist,
                                            spin_index)
from src.Base.environment_seam import environment_of
from src.Base.utils.parallel.threads import blas_single_threaded, row_map
from src.SingleReference.GW.reaction_field import (
    bare_self_energy, environment_quasiparticle_shift, equilibrium_level_shift,
    shift_after_solve)
from src.SingleReference.LinearResponse.casida import CasidaSolver, solver_choice
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.cc_polarizability import GWCCSelfEnergy
from src.SingleReference.GW.contour_deformation import solve_qp_energy_contour
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
from src.SingleReference.GW import evGW  # module import: evGW imports this file
from src.SingleReference.GW import qsGW  # module import: qsGW imports this file
from src.Solvers.qp_equation import solve_qp_equation_pole_strength

IMAGINARY_AXIS_MODES = ('imagfrequency', 'imag-frequency', 'space-time')
from src.Base.utils.parallel.threads import resolve_workers

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

#: The `continuation` each `mode` can run; the first entry is that mode's
#: default. A pair missing here is refused rather than approximated by a
#: neighbour: the continuation is where a deep state is lost, so substituting
#: one for another returns a different functional under the name asked for.
MODE_CONTINUATIONS = {'casida': ('spectral',),
                      'imagfrequency': ('pade', 'cd'),
                      'imag-frequency': ('pade', 'cd'),
                      'space-time': ('pade', 'cd', 'laplace', 'sop')}

#: Route keywords only the contour driver reads. Its grid size, its
#: e_min_below_gap and tile cap are `numerics` fields (`Numerics`).
CONTOUR_KEYWORDS = frozenset({'w0_cd', 'diagnostics'})
#: The pole model reads the same: its pole count and fit stride are
#: `numerics.qp` fields, and it is FITTED on the CD frequency grid.

#: Route keywords only a Pade continuation's drivers read.
PADE_KEYWORDS = frozenset({'w0', 'grid', 'greedy', 'freq_block', 'scratch_dir',
                           'extras', 'screen_r_cut', 'distribute', 'comm',
                           'timings'})

#: Grid sizes a route reads from `numerics` alone: passed as a keyword they
#: are refused by name.
NUMERICS_KEYWORDS = {'nfreq': 'imagfreq.nfreq', 'npade': 'quadrature.npade',
                     'tau_target': 'quadrature.tau_target',
                     'ntau': 'quadrature.ntau_st'}

#: Route keywords naming the ISDF factorization or its imaginary-time grid,
#: which no Casida route has. `numerics`, the run's one Numerics, is read by
#: every route and loop that has a setting in it (the space-time quadrature and
#: pole model, the imaginary-frequency grid, the evGW and qsGW loops).
ISDF_KEYWORDS = frozenset({'counts', 'radii', 'factors', 'auxbasis',
                          'grid_accuracy', 'sigma_x'})

#: What `solver` accepts on the Casida route. 'auto' is resolved through
#: `casida.solver_choice`, the repository's one eigensolver rule.
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
        # the one dense-or-Davidson rule of every Casida solve
        if solver_choice(n_ov) == 'dense':
            return
    reached = ('the memory rule refuses the dense (A, B) pair at '
               f'n_ov={n_ov} ({2 * int(n_ov)**2 * 8 / BYTES_PER_GB:.1f} GB)'
               if solver == 'auto' else "solver='davidson' was asked for")
    raise NotImplementedError(
        f"{reached}, and mode='casida' has no matrix-free solve: its Sigma_c "
        f"is a Lehmann sum over EVERY Casida root and the Davidson returns the "
        f"lowest nroots of them. Take mode='space-time', whose self-energy "
        f"never forms the spectrum at all, or solver='dense' to pay the pair.")


def _refused_keywords(continuation):
    """The route keywords this continuation does not read."""
    if continuation == 'spectral':
        return CONTOUR_KEYWORDS | PADE_KEYWORDS | ISDF_KEYWORDS
    if continuation == 'pade':
        return CONTOUR_KEYWORDS
    return PADE_KEYWORDS


def _continuation_of(mode_key, continuation, route_kwargs, return_z):
    """The continuation `mode` will run, with the validity table enforced.

    Decided and checked before any integral is built, because the alternative
    is a converged self-energy and then a refusal. The table is
    `MODE_CONTINUATIONS`; the keyword sets above say which route reads what, so
    a keyword aimed at another continuation is an error rather than a request
    that quietly does nothing.
    """
    given = sorted(set(NUMERICS_KEYWORDS) & set(route_kwargs))
    if given:
        raise TypeError(
            f"{given} are Numerics settings ("
            f"{', '.join(f'numerics.{NUMERICS_KEYWORDS[k]}' for k in given)}): "
            f"pass numerics=Numerics().derive(...) instead of a keyword")
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
            f"which is noise as a weight (the space-time driver's with_z "
            f"returns it, and the window selection reads it only as a coarse "
            f"admission gate); and the Casida route's root finder "
            f"(`Solvers.qp_equation.solve_qp_equation_pole_strength`) "
            f"returns the root alone, its pole strength being an internal "
            f"finite difference. "
            f"The three contour continuations return Z from their own "
            f"closed-form slope, which is the exact derivative of the "
            f"equation solved.")
    return name


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
        nocc = mf.nelec[spin_index(spin_channel)]
        eps_qp = eps_qp[spin_index(spin_channel)]
    else:
        nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)
    out = {p: {'GW': float(eps_qp[p]) * HARTREE_TO_EV} for p in states}
    out['evgw_info'] = info
    if not isinstance(state, list):
        return out[states[0]]['GW']
    return out


def _qp_energy_qsgw(mf, mol, mode_key, selfenergy, polarizability, state,
                    screening, route_kwargs, spin_channel='alpha'):
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
    if isinstance(mf, scf.uhf.UHF):
        # the loop converges both channels at once; the caller asked for one
        nocc = mf.nelec[spin_index(spin_channel)]
        eps_qp = eps_qp[spin_index(spin_channel)]
    else:
        nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)
    out = {p: {'GW': float(eps_qp[p]) * HARTREE_TO_EV} for p in states}
    out['qsgw_info'] = info
    if not isinstance(state, list):
        return out[states[0]]['GW']
    return out


def calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True,
                   eta=None, state='homo',
                   spin_channel='alpha', printSpectralFunction=False,
                   dm_correction=None, tda=False,
                   nroots=8, mode='casida', self_consistency='G0W0',
                   eps_anchor=None, n_workers=None, continuation=None,
                   return_z=False, solver='auto', equilibrium=False,
                   out=None, **route_kwargs):
    """Quasiparticle energies, in eV.

    The dispatch decides in this order, every refusal before an integral is
    built:
      1. `mode` must be a key of `MODE_CONTINUATIONS`, and `continuation` (None
         takes the mode's default: 'spectral' on 'casida', 'pade' on the two
         imaginary-axis modes) must be one it pairs with; a route keyword only
         another continuation reads is a TypeError (`_continuation_of`).
      2. `solver` is checked (mode='casida': the dense-pair memory rule; any
         other mode: 'auto' only), and a non-default `eta` is refused on the
         imaginary-axis modes, which broaden nothing.
      3. `self_consistency` picks the loop. A key of `EVGW_SCREENING` ('evGW',
         'evGW0') enters `evGW.evgw_eigenvalues` and runs only the mode's
         default continuation, so 'cd', 'laplace' and 'sop' are refused there;
         a key of `QSGW_SCREENING` ('qsGW', 'qsGW0') enters
         `qsGW.qsgw_eigenvalues`, mode='casida' only. Neither takes
         `eps_anchor`, and the loop's own keywords ride in **route_kwargs.
      4. Anything else is G0W0: an imaginary-axis mode goes to its driver with
         the chosen continuation (the Pade drivers of `imaginary_axis` and
         `space_time`, or `contour_deformation` for 'cd', 'laplace' and
         'sop'), and mode='casida' solves the Lehmann sum over its spectrum
         here, refusing any route keyword.

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
                        both imaginary-axis modes. Reads w0, grid and greedy;
                        its point counts are `numerics`' (imagfreq.nfreq,
                        quadrature.npade); returns no Z, since
                        differentiating a Thiele recursion divides by inverse
                        differences approaching zero.
                    'cd'       contour deformation: the imaginary-axis integral
                        plus the residues of the poles of G the rotation sweeps.
                        Exact for any state and O(N^4) per residue, because each
                        one evaluates W at a real frequency. mode='space-time'.
                        Reads w0_cd; its frequency grid is `numerics`'.
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
                        Its pole count and fit stride are `numerics.qp`, on top of the
                        CD grid, which is the grid the pole model is fitted on.
    diagnostics:    a dict that receives what the contour driver decided:
                    the CD grid it resolved, the residues swept, the pole
                    guard in force, the Newton seed, the residue treatment
                    that ran and the Laplace transform's representation error
                    where it did. 'cd', 'laplace' and 'sop' only.
    out:            a dict, or None: receives 'reference_diagnostics', the
                    unrestricted reference's spin contamination and stability
                    (`unrestricted_reference_diagnostics`, warned about;
                    None for a restricted one), for the caller's record.
    return_z:       return (energy, Z) instead of the energy. The three
                    contour continuations take Z from their own Newton slope,
                    the exact derivative of the equation they solved; the other
                    two refuse it rather than return a differenced one.
    solver:         the Casida eigensolver, mode='casida' only. 'auto'
                    (default) resolves through `casida.solver_choice`, the one
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
    n_workers:      threads for the per-state root scan (Casida route, also
                    inside every evGW cycle on it). The
                    scan uses a thread pool by default of OMP_NUM_THREADS
                    threads, else one per CPU in the process's affinity mask,
                    never more than that mask holds. n_workers=1 gives the
                    serial scan.
    equilibrium:    in a continuum with a static dielectric constant, add
                    the solvent's equilibrium relaxation around the charged
                    state to every level returned (`reaction_field.
                    equilibrium_level_shift`): an occupied level rises and a
                    virtual one falls by Eq18_p(eps_s) - Eq18_p(eps_inf), so
                    E^(N-/+1) = E_0 -/+ eps_p is the ion in equilibrium with
                    its solvent rather than the vertical one. A classical
                    relaxation after the quasiparticle equation, on every
                    route and on each self-consistency's converged levels;
                    refused without a static response.
    """
    # None is "not set": DEFAULT_BROADENING_ETA where a broadening acts, and
    # the self-consistent loops and the equilibrium levels call back with
    # what was given, so a route without a broadening refuses only a set one
    given_eta, eta = eta, DEFAULT_BROADENING_ETA if eta is None else eta
    if equilibrium:
        return _equilibrium_levels(
            mf, dict(selfenergy=selfenergy, polarizability=polarizability,
                     df=df, eta=given_eta, state=state, spin_channel=spin_channel,
                     printSpectralFunction=printSpectralFunction,
                     dm_correction=dm_correction, tda=tda,
                     nroots=nroots, mode=mode,
                     self_consistency=self_consistency, eps_anchor=eps_anchor,
                     n_workers=n_workers, continuation=continuation,
                     return_z=return_z, solver=solver, out=out,
                     **route_kwargs))
    mol = mf.mol
    mode_key = str(mode).lower().replace('_', '-')
    if mode_key not in MODE_CONTINUATIONS:
        raise ValueError(
            f"mode='{mode}'; choose 'casida', 'imagfrequency' or 'space-time'.")
    # THE TABLE IS ENFORCED FIRST. The alternative is a converged self-energy
    # and then a refusal.
    continuation = _continuation_of(mode_key, continuation, route_kwargs,
                                    return_z)
    require_closed_shell_or_unrestricted(mf, 'calc_qp_energy', mol=mol)
    # named flags for the caller's record (refuse inputs, warn numerics)
    diagnostics = unrestricted_reference_diagnostics(mf, 'calc_qp_energy')
    if out is not None:
        out['reference_diagnostics'] = diagnostics
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
    if mode_key in IMAGINARY_AXIS_MODES and given_eta is not None:
        # Neither route has a broadening to set: chi0 has the real denominator
        # -2d/(d^2 + w^2) on the imaginary axis, and Sigma_c reaches the real
        # axis by Pade or by contour deformation rather than at w + i.eta.
        # Refused here rather than on the route, so that an evGW loop, whose
        # every cycle calls back in with this eta, stops before its first one.
        raise ValueError(
            f"mode='{mode_key}' has no broadening: eta={eta!r} never "
            f"reaches the self-energy, so the number returned would be the "
            f"one at eta={DEFAULT_BROADENING_ETA!r} under another name. "
            f"mode='casida' is the only route eta acts on.")
    consistency = str(self_consistency).lower()
    if eps_anchor is not None and consistency in {**EVGW_SCREENING, **QSGW_SCREENING}:
        # the loops anchor on the mean field's eigenvalues (evGW) or on none
        # (qsGW); casida_evgw_step takes an anchor for a single step
        raise TypeError(
            f'self_consistency={self_consistency!r} sets its own anchor every '
            f'cycle, so eps_anchor would be overwritten by the loop rather '
            f'than read')
    if consistency in EVGW_SCREENING:
        if continuation != MODE_CONTINUATIONS[mode_key][0]:
            raise ValueError(
                f"self_consistency={self_consistency!r} drives continuation="
                f"{MODE_CONTINUATIONS[mode_key][0]!r} only, not "
                f"{continuation!r}: the loop freezes its quadrature at the "
                f"first cycle and reinjects eigenvalues, and every other "
                f"continuation would have to freeze its own branch as well.")
        # the loop takes the route's knobs by name, on the Casida route through
        # its own step and on the imaginary axis by calling back into this
        # function every cycle
        route_kwargs = dict(route_kwargs, df=df, eta=given_eta,
                            dm_correction=dm_correction, tda=tda,
                            solver=solver,
                            n_workers=n_workers)
        return _qp_energy_evgw(mf, mol, mode_key, selfenergy, polarizability,
                               state, spin_channel, EVGW_SCREENING[consistency],
                               route_kwargs)
    if consistency in QSGW_SCREENING:
        # the loop builds its own density every cycle, so a density correction
        # has nowhere to enter; the imaginary-axis knobs mean nothing to it.
        # Its mode refusal lives in _qp_energy_qsgw: on mode='casida' the table
        # above has already left 'spectral', the only continuation there is.
        if dm_correction is not None:
            raise NotImplementedError(
                'self_consistency=qsGW builds the density from its own orbitals '
                'every cycle; dm_correction has no place in it')
        if given_eta is not None:
            raise TypeError(
                'self_consistency=qsGW reads the broadening of its static '
                'self-energy from numerics.qsgw.eta, not from eta=')
        route_kwargs = dict(route_kwargs, df=df, tda=tda, n_workers=n_workers)
        return _qp_energy_qsgw(mf, mol, mode_key, selfenergy, polarizability,
                               state, QSGW_SCREENING[consistency], route_kwargs,
                               spin_channel=spin_channel)
    if mode_key in IMAGINARY_AXIS_MODES:
        # the anchor is a named argument here and a route keyword there
        if eps_anchor is not None:
            route_kwargs = dict(route_kwargs, eps_anchor=eps_anchor)
        return _qp_energy_imaginary_axis_route(
            mf, mol, mode_key, continuation, selfenergy, polarizability, df,
            state, spin_channel, dm_correction, tda, return_z,
            route_kwargs)
    # the Casida route has no setting in the run's Numerics
    route_kwargs = {k: v for k, v in route_kwargs.items() if k != 'numerics'}
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
            sigma_solvent, shift, spin_channel, is_uhf)
        if not isinstance(state, list) and not isinstance(selfenergy, list):
            return results[states[0]]['GW']
        return results

    methods = selfenergy if isinstance(selfenergy, list) else [selfenergy]
    # Fail fast on typos rather than silently falling back to GW.
    method_infos = {m: get_method_info(m) for m in methods}

    # W(0) is read by the BSE kernel and by a screened DF vertex, nothing else
    screened_vertex = (polarizability.upper() != 'TDHF' and df
                       and any(method_infos[m]['needs_vertex'] for m in methods))
    setup = _casida_setup(mf, mol, df, eta, shift, is_uhf, nocc,
                          static_w=polarizability.upper() == 'BSE' or screened_vertex)
    se_solver = setup['se_solver']
    spectrum = casida_spectrum(setup['lr_solver'], nocc, polarizability,
                                setup['w_aux'], tda, method_infos, methods, is_uhf,
                                df)
    # Eq. (18) is added to every G0W0 root whole, after the solve; only an
    # evGW step anchored elsewhere carries it in the equation
    # (`reaction_field.shift_after_solve`)
    after = shift is not None and shift_after_solve(eps_anchor)
    eri_w_singlet, eri_w_triplet, xc_diagonal = _channel_terms(
        mf, mol, setup, df, dm_correction, None if after else sigma_solvent,
        spin_channel, is_uhf, nocc,
        bare_vertex=polarizability.upper() == 'TDHF')

    # The equation is anchored on eps_p while the solvers above screen with
    # whatever spectrum `mf` carried; evGW is the case where the two differ.
    anchor = setup['eps'] if eps_anchor is None else np.asarray(eps_anchor, float)
    anchor_spin = _channel(anchor, spin_channel, is_uhf)
    energies = qp_energies_from_spectrum(
        se_solver, nocc, spectrum, method_infos, methods, spin_channel, states,
        eri_w_singlet, eri_w_triplet, is_uhf, df, anchor_spin, xc_diagonal,
        n_workers=n_workers)
    if after:
        energies = {p: {m: e + float(shift[p]) for m, e in by_method.items()}
                    for p, by_method in energies.items()}
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


def _equilibrium_levels(mf, kwargs):
    """`calc_qp_energy(**kwargs)` with the equilibrium level shift added.

    The shift is formed FIRST, so a continuum without a static response is
    refused before any self-energy is built. It is added to every numeric
    entry the route returns -- the level of each method on the Casida route --
    and never to a pole strength.
    """
    mol = mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)
    channel = kwargs['spin_channel']
    level = equilibrium_level_shift(mf, mol,
                                    auxbasis=kwargs.get('auxbasis'))
    level = level[0 if channel == 'alpha' else 1] if is_uhf else level
    out = calc_qp_energy(mf, equilibrium=False, **kwargs)

    def moved(p, value):
        return value + float(level[int(p)]) * HARTREE_TO_EV

    state = kwargs['state']
    nocc = _channel(mf.nelec if is_uhf else mol.nelectron // 2, channel,
                    is_uhf)
    states = _resolve_states(state, nocc)
    z = None
    if kwargs['return_z'] and not isinstance(out, dict):
        out, z = out
    if isinstance(out, dict):
        return {p: ({m: (moved(p, v) if isinstance(v, (float, np.floating))
                         else v) for m, v in entry.items()}
                    if isinstance(entry, dict) and p in states else entry)
                for p, entry in out.items()}
    if isinstance(state, list):
        out = [moved(p, e) for p, e in zip(states, out)]
    else:
        out = moved(states[0], out)
    return out if z is None else (out, z)


def _two_electron_integrals(mol, mf, df, is_uhf):
    """(df_coeff, eri): the three-index DF factors, or the full chemist ERI."""
    if not df:
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        if is_uhf:
            # the solvers take (aa|aa), (bb|bb), (aa|bb); the integrals come as aa, ab, bb
            eri_aa, eri_ab, eri_bb = eri
            eri = (eri_aa, eri_bb, eri_ab)
        return None, eri
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
    return x[spin_index(spin_channel)]


def _reaction_field_terms(mf, mol, nocc_spin, spin_channel):
    """(shift, sigma_solvent): the environment's one-body terms, None in the gas
    phase.

    Duchemin et al. Eq. (18) -- the self-polarization of the orbital carrying
    the added charge in the SCREENED reaction field, built once per spectrum a
    mean field carries, so this route and the imaginary-axis ones read the same
    array, in every evGW cycle as well; `sigma_solvent` is then its diagonal.
    Unrestricted, the screening is built once from both spins and this returns
    the requested channel's row. `cohsex_correction`, which sums the BARE vtilde
    over every orbital and differs by 0.39 eV of quasiparticle gap on water in
    water, is left only for an environment that dresses no interaction.
    """
    unrestricted = isinstance(mf, scf.uhf.UHF)
    shift = environment_quasiparticle_shift(mf, mol,
                                            None if unrestricted else nocc_spin)
    if shift is not None:
        if unrestricted:
            shift = shift[spin_index(spin_channel)]
        return shift, np.diag(shift)
    sigma_solvent = environment_of(mf).static_self_energy(mf, mol)
    if isinstance(sigma_solvent, tuple):
        sigma_solvent = sigma_solvent[spin_index(spin_channel)]
    return None, sigma_solvent


def _casida_setup(mf, mol, df, eta, shift, is_uhf, nocc, static_w=True):
    """What the Casida route builds from the mean field before a spectrum screens.

    A dict: 'eps', the mean field's own spectrum; 'df_coeff' and 'eri', the
    three-index factors or the chemist ERI, built with the environment off
    where `shift` carries it (`bare_self_energy`); 'spin_mode'; 'lr_solver' and
    'se_solver' on that spectrum; 'w_aux', the static RPA W_aux a vertex or a
    BSE screening contracts against, built only under `static_w` and None
    otherwise. None of it moves with the eigenvalues, so
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
            'w_aux': lr_solver.static_screening_aux(nocc) if static_w else None}


def _channel_terms(mf, mol, setup, df, dm_correction, sigma_solvent, spin_channel,
                   is_uhf, nocc, vertex=True, bare_vertex=False):
    """(eri_w_singlet, eri_w_triplet, xc_diagonal) of one spin channel.

    A vertex contracts against the interaction the Casida kernel screens its
    exchange with: the bare v (None) where `bare_vertex` says so, the TDHF
    kernel's; else the static W, the auxiliary form under DF and the explicit
    four-index one otherwise, built only when `vertex` asks for it and None
    without. <Sigma_Hx - v_Hxc> is one matrix for the whole spectrum; built per
    state it costs a J/K and a grid build per orbital, which dominates a
    cycle that asks for every orbital.
    """
    if bare_vertex:
        eri_w_singlet = eri_w_triplet = None
    elif df:
        eri_w_singlet = eri_w_triplet = setup['w_aux']
    elif not vertex:
        eri_w_singlet = eri_w_triplet = None
    else:
        eri_w_singlet, eri_w_triplet = setup['lr_solver'].construct_4d_w_rpa(
            nocc, spin_channel)
    xc_diagonal = _static_correction(mf, mol, dm_correction, sigma_solvent,
                                     spin_channel, is_uhf)
    return eri_w_singlet, eri_w_triplet, xc_diagonal


def _move_poles(se_solver, eps):
    """The poles of Sigma_c at `eps`; the amplitudes, the spectrum's, stay."""
    if se_solver.spin_mode == 'unrestricted':
        se_solver.eps_a, se_solver.eps_b = eps
    else:
        se_solver.eps = eps


def casida_spectrum(lr_solver, nocc, polarizability, w_aux, tda, method_infos,
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
            # Without DF the spin-flip blocks read the four-index (aa|bb) W.
            if df or w_casida is None:
                W_sf = w_casida
            else:
                W_sf = lr_solver.construct_4d_w_rpa(nocc)[1]
            # one coupled problem: ab excitations with ba de-excitations and back
            out['triplet'] = CasidaSolver(*lr_solver.build_spin_flip_casida_matrices(
                nocc, lBSE=lBSE, W_aux=W_sf)).solve(tda=tda)
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
            chi_b_t_val = get_sf(nocc, X_t, Y_t, spin_channel=spin_channel,
                                 eri_w=eri_w_triplet, p_state=p_state)
        else:
            chi_b_t_val = se_solver.get_chi_b_vertex(nocc, X_t, Y_t,
                                                     spin_channel=spin_channel,
                                                     eri_w=eri_w_triplet,
                                                     p_state=p_state)
        out[method] = (omega_val, chi_a_val, chi_b_val, omega_t, chi_b_t_val)
    return out


def _static_correction(mf, mol, dm_correction, sigma_solvent, spin_channel,
                       is_uhf):
    """<Sigma_Hx - v_Hxc>_pp for every orbital p, plus the solvent reaction
    field: one array over the spectrum.

    The static term every route reads (`static_exchange_mean_field_matrix`):
    the mean field's own J, K and v_xc, so on a fitted reference the fitting
    error of J and K cancels against the eigenvalues instead of entering the
    quasiparticle energy. One build for the whole spectrum, not one per state.
    """
    sigma_x = static_exchange_mean_field_matrix(mf, mol,
                                                dm_correction=dm_correction)
    if is_uhf:
        sigma_x = sigma_x[spin_index(spin_channel)]
    xc_correction = np.diag(sigma_x).copy()
    if sigma_solvent is not None:
        xc_correction = xc_correction + np.diag(np.asarray(sigma_solvent))
    return xc_correction


def qp_energies_from_spectrum(se_solver, nocc, spectrum, method_infos, methods,
                              spin_channel, states, eri_w_singlet, eri_w_triplet,
                              is_uhf, df, eps_spin, xc_correction,
                              n_workers=None):
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
    spectrum : dict, as returned by casida_spectrum
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
    n_workers : int or None
        None takes OMP_NUM_THREADS, else the CPUs in the affinity mask; any
        value is capped at that mask (see resolve_workers); 1 is serial.

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
            # the root of largest weight; no root at all is NaN with a
            # QPNoRoot warning (refuse inputs, warn numerics)
            out[method] = solve_qp_equation_pole_strength(func, e0,
                                                          vectorized=True)
        return p, out

    results = {}
    p0, out0 = solve_state(0)
    results[p0] = out0
    rest = range(1, len(states))
    n_workers = resolve_workers(n_workers, len(states))
    run_parallel = n_workers > 1 and len(states) > 1
    if run_parallel:
        # the states over the process's row pool, BLAS at one thread under it
        rest = list(rest)
        with blas_single_threaded(min_threads=1):
            parts = row_map(lambda r0, r1: [solve_state(i) for i in rest[r0:r1]],
                            len(rest), n_workers)
        for chunk in parts:
            for p, out in chunk:
                results[p] = out
    else:
        for i in rest:
            p, out = solve_state(i)
            results[p] = out
    return results


def casida_evgw_step(mf, mol, screening, df=True, eta=None,
                     dm_correction=None, tda=False,
                     n_workers=None, eps_anchor=None, fixed_spectrum=None,
                     fixed_rho=None, solver='auto'):
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
        casida_spectrum returns) and its DF transition density (naux,
        nexciton) to screen with, in place of the one built from `mf`; the
        amplitudes are then that density contracted with `mf`'s own DF
        factors, which is how the mean field's W meets rotated orbitals.
        Both or neither; `mf` must carry `with_df`, whose auxiliary basis
        does not move with the orbitals.
    The other keywords are `calc_qp_energy`'s; `solver` is refused here as
    there (`refuse_unbuilt_casida_solver`), before the loop's first cycle.

    Built once, since none of it moves with the eigenvalues: the DF factors,
    the static W only where something reads it (nothing: the loop refuses a
    vertex) and <Sigma_Hx - v_Hxc> of every spin channel. 'fixed' also keeps
    the one self-energy solver, whose amplitude cache keys on the identity of X
    and Y; 'updated' takes a fresh one per spectrum for the same reason. The
    reaction field of an attached environment does move: its Delta W screens
    with the eigenvalues it is built at, so 'updated' rebuilds its term every
    cycle at the iterate, as the imaginary-axis routes do through
    calc_qp_energy on the shifted view, and 'fixed' keeps the mean field's with
    the rest of W. Either way it stays inside the equation: the levels G is
    built on carry it (`reaction_field.shift_after_solve`).
    """
    eta = DEFAULT_BROADENING_ETA if eta is None else eta  # None: not set
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
    refuse_unbuilt_casida_solver(solver, mf, mol, tda)
    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = mf.nelec if is_uhf else mol.nelectron // 2
    channels = ('alpha', 'beta') if is_uhf else ('alpha',)
    methods = ['GW']
    method_infos = {'GW': get_method_info('GW')}

    fields = {ch: _reaction_field_terms(mf, mol, _channel(nocc, ch, is_uhf), ch)
              for ch in channels}
    # GW@RPA: neither the Casida kernel nor a vertex reads the static W
    setup = _casida_setup(mf, mol, df, eta, fields['alpha'][0], is_uhf, nocc,
                          static_w=False)
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
        spectrum_mf = casida_spectrum(setup['lr_solver'], nocc, 'RPA',
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
            spectrum = casida_spectrum(lr_solver, nocc, 'RPA', setup['w_aux'], tda,
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
                xc_diagonal, n_workers=n_workers)
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
                                 sigma_solvent, shift, spin_channel, is_uhf):
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
    if shift is not None:
        # Eq. (18) needs the static screening in the DRESSED and the bare
        # interaction; the only one available is the RPA's, and a CC-screened
        # Sigma beside an RPA-screened shift is two screenings for one
        # quasiparticle. The CC response in the dressed interaction is a CC
        # ground state in the continuum, which is not built.
        raise NotImplementedError(
            'CC polarizability in a continuum: the Eq. (18) shift would be '
            'screened by the RPA while Sigma is screened by EOM-CC -- two '
            'screenings for one quasiparticle -- and the CC response in the '
            "dressed interaction is not built. Use polarizability='RPA' in a "
            'continuum.')
    if hasattr(mf, 'xc'):
        raise NotImplementedError(
            "CC polarizability assumes an HF reference; a KS starting point would "
            "additionally need a v_xc correction, which this route does not build.")

    # The CC screening is built from the BARE interaction, like the Casida
    # route's; a continuum that dresses the interaction is refused above, and
    # one that does not arrives as the COHSEX `sigma_solvent`.
    with bare_self_energy(mf, shift):
        solver = GWCCSelfEnergy(mf, level=level, nroots=nroots)
    spin_offset = spin_index(spin_channel)

    results = {}
    for p_state in states:
        # Spatial orbital p -> spin orbital 2p (+1 for beta), interleaved.
        p_so = 2 * p_state + spin_offset
        static_shift = (0.0 if sigma_solvent is None
                        else sigma_solvent[p_state, p_state])
        qp_ha = solver.solve_qp(p_so, eta=eta,
                                static_shift=static_shift)
        results[p_state] = {'GW': qp_ha * HARTREE_TO_EV}
    return results


def _qp_energy_imaginary_axis_route(mf, mol, mode_key, continuation, selfenergy,
                                    polarizability, df, state, spin_channel,
                                    dm_correction, tda, return_z,
                                    route_kwargs):
    """Dispatch to the imaginary-frequency, space-time or contour driver.

    All implement GW@RPA on a density-fitted reference and nothing else, so
    every unsupported combination is rejected rather than quietly returning a
    GW@RPA number under another name. `eta` is one of them and is refused by
    the caller, since none of them broadens anything. An unrestricted
    reference runs on all of them: one W from both spins, the self-energy of
    `spin_channel`.

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
    if not df:
        raise ValueError(f"mode='{mode_key}' requires density fitting (df=True).")
    if tda:
        raise ValueError(f"tda has no meaning for mode='{mode_key}'; it never "
                         f"forms a Casida problem.")

    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = tuple(int(n) for n in mf.nelec) if is_uhf else mol.nelectron // 2
    states = _resolve_states(state, _channel(nocc, spin_channel, is_uhf))
    if is_uhf:
        route_kwargs = dict(route_kwargs, spin_channel=spin_channel)

    if continuation in ('cd', 'laplace', 'sop'):
        # The two contours and the pole model share every step but the last:
        # one factorization, one proj(tau), one contour grid, one exchange
        # build.
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
                                         dm_correction=dm_correction,
                                         **route_kwargs)
        out, z = [e * HARTREE_TO_EV for e in out], None
    else:
        # One call for the whole window: W(i omega), the exchange and v_xc are
        # built once and their rows sliced for every state.
        out = solve_qp_energy_imaginary_axis(mf, mol, nocc, np.asarray(states),
                                             dm_correction=dm_correction,
                                             **route_kwargs)
        out, z = [e * HARTREE_TO_EV for e in out], None

    if not isinstance(state, list):
        return (out[0], z[0]) if return_z else out[0]
    return (out, z) if return_z else out
