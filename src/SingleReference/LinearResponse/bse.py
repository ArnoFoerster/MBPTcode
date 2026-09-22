"""Bethe-Salpeter excitation energies: every route behind one front end.

`solve_bse` is the front end. Two independent axes choose the route, and the
same physics -- BSE@GW with a statically screened kernel -- comes out of all of
them:

  SOLVER     davidson  never forms A or B; the few lowest roots from a
                       matrix-free block action on a three-index factor.
             dense     builds A and B and diagonalizes; the whole spectrum,
                       O(N^6) and O(N^4) memory.
             auto      `solver_choice`: dense while the (A, B) pair fits in
                       BSE_DENSE_MAX_GB, davidson above it. One rule, in one
                       place, fed by one constant -- so a caller that cannot
                       know n_ov in advance still never asks for a matrix it
                       has no memory for.

  INTEGRALS  isdf      the separable (Duchemin-Blase) factorization's own
                       three-index factor; the only O(N^3) screening here.
             df        pyscf's cderi, in the mean field's auxiliary basis.
             full      the four-index (pq|rs) tensor, no fitting error at all.

Five of the six combinations run. davidson + full has no implementation: the
matrix-free action exists to contract a three-index factor, and a code that can
hold (pq|rs) can hold A and B as well.

Both three-index routes want a COULOMB-fitting auxiliary basis (<basis>-jkfit),
not the MP2 correlation-fitting <basis>-ri: the BSE direct term contracts
(ij|ab), a contraction MP2 never asks for. On water/cc-pVDZ, one quasiparticle
diagonal and the exact tensor as the reference, the kernel's fitting error on
the lowest excitation is 0.7 meV in -jkfit and 24 meV in -ri.
"""
import time

import numpy as np

from src.Base.constants import BSE_DENSE_MAX_GB, HARTREE_TO_EV, KAPPA
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.GW.evGW import evgw_eigenvalues, shifted_mean_field
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.space_time import (_unpack_factors,
                                               separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import (bse_pair_diagonal,
                                                         isdf_df_coefficients,
                                                         oscillator_strengths,
                                                         solve_bse_df,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

SOLVERS = ('davidson', 'dense')
#: What `solver` accepts: a route by name, or 'auto' for `solver_choice`, which
#: is resolved to one of SOLVERS before any other check runs.
SOLVER_CHOICES = SOLVERS + ('auto',)
INTEGRALS = ('isdf', 'df', 'full')

#: Keywords each half of the route matrix accepts on top of the front end's own.
DAVIDSON_KWARGS = ('conv_tol', 'max_cycle', 'guess_factor', 'progress')
DENSE_KWARGS = ('tda',)
ISDF_KWARGS = ('factors', 'auxbasis', 'radii', 'counts', 'n_start',
               'grid_accuracy', 'distribute', 'comm')
#: The two of those that only the matrix-free ISDF route can act on: it is the
#: Davidson block action and the GW tau axis that are divided over ranks, and
#: the separable fit itself takes neither keyword. The dense route refuses them
#: rather than forward a distributed request into a serial solve.
ISDF_MPI_KWARGS = ('distribute', 'comm')
SHARED_KWARGS = ('probe', 'gw_kwargs')


def solver_choice(n_ov, tda=False):
    """'dense' or 'davidson' for a particle-hole space of n_ov pairs.

    Dense while the Casida pair (A, B) fits in BSE_DENSE_MAX_GB -- the dense
    route's own storage, 2 * n_ov**2 * 8 bytes of float64 -- and matrix-free
    above it. That pair is what the rule protects: the block action holds a
    three-index factor and a handful of trial vectors instead, so nothing else
    on the dense route grows as n_ov**2.

    tda: not exempt, and it does not move the boundary. The Tamm-Dancoff
    problem diagonalizes A alone, but the dense route BUILDS both blocks either
    way (`build_casida_matrices` returns the pair; only `CasidaSolver.solve`
    drops B), so a TDA solve holds the same memory and flips to the matrix-free
    route at the same size. The argument is here to state that, not to change
    the answer: a rule that exempts TDA lets a dense solve ask for terabytes.
    """
    return 'dense' if 2 * int(n_ov)**2 * 8 / 1e9 <= BSE_DENSE_MAX_GB else 'davidson'


def solve_bse(mf, mol=None, nocc=None, nroots=5, solver='davidson',
              integrals='isdf', qp='G0W0', spin='singlet',
              self_consistency='G0W0', screen_at='mean-field', **route_kwargs):
    """BSE excitation energies: mean field in, (omega, X, Y, info) out.

    solver:     'davidson' (default), 'dense', or 'auto' to let `solver_choice`
                pick from the pair count nocc * nvirt and the memory the dense
                (A, B) pair would take; see the module docstring for what each
                route costs and returns. The resolved value, never 'auto', is
                what runs and what `info['solver']` reports.
    integrals:  'isdf' (default), 'df' or 'full'. The quasiparticle diagonal is
                built by the matching GW route -- space-time GW on the separable
                factors for 'isdf', the dense Casida GW for 'df' and 'full' --
                so a route is one level of theory throughout, integrals
                included, rather than a BSE kernel bolted onto a foreign
                spectrum. Pass `qp` as an array to share one diagonal across
                routes and isolate the kernel.
    qp:         'G0W0' (default) puts the route's own quasiparticle energies on
                the BSE diagonal; an ARRAY of energies is used directly;
                False/None solves BSE@mean-field.
    spin:       'singlet' (kappa = 2) or 'triplet' (kappa = 0, the bare-exchange
                term absent); the screened term is spin-independent. Every route
                supports both.
    self_consistency: 'G0W0' (default), or 'evGW' to converge the spectrum
                first and screen W with that fixed point.
    screen_at:  which spectrum builds W when `qp` is an explicit array,
                'mean-field' (default, the standard G0W0-BSE split) or 'qp'.
                Ignored otherwise. See `solve_bse_isdf` for the full rule.
    nroots:     roots wanted. None is the whole spectrum and is dense-only.
    probe:      measure min eig(A - B) and refuse the solve while it is
                negative, the mean-field instability regime where the Casida
                omega^2 reduction is invalid. Davidson estimates it matrix-free
                ('sign' stops as soon as the sign is proven); the dense route
                reads it exactly off the A and B it just built, and skips it
                under `tda`, which never makes that reduction.

    Route-specific keywords, rejected rather than ignored when they belong to a
    route that was not chosen: `conv_tol`, `max_cycle`, `guess_factor` and
    `progress` are the Davidson iteration's; `tda` (solve with Y = 0) is the
    dense solver's; `factors`, `auxbasis`, `radii`, `counts`, `n_start` and
    `grid_accuracy` belong to the separable fit, and `distribute`/`comm` to the
    matrix-free ISDF route that divides the tau axis and the rows of Zt over
    MPI ranks. `gw_kwargs` reaches the GW step on any route.

    Returns (omega, X, Y, info), with omega in Hartree ascending, X and Y of
    shape (n_pair, nroots) normalized <X|X> - <Y|Y> = 1, and info carrying
    eps / eps_mf / W_aux / min_eig_amb / evgw / timings on every route; factors
    (ISDF) or coeff_df (DF), None on the route that has neither; stats only
    where an iterative solver produced them and n_states only where a solver
    formed the whole spectrum; oscillator_strength and
    transition_dipole per root in the length gauge, None for a triplet, whose
    transitions are spin-forbidden and whose spatial-orbital formula would
    report a number that is not one.
    """
    mol = mf.mol if mol is None else mol
    nocc = mol.nelectron // 2 if nocc is None else nocc
    solver = str(solver).lower()
    integrals = str(integrals).lower()
    if solver not in SOLVER_CHOICES:
        raise ValueError(f"solver={solver!r}; choose one of {list(SOLVER_CHOICES)}.")
    if integrals not in INTEGRALS:
        raise ValueError(f"integrals={integrals!r}; choose one of {list(INTEGRALS)}.")
    if spin not in KAPPA:
        raise ValueError(f"spin={spin!r}; choose one of {sorted(KAPPA)}.")
    if np.asarray(mf.mo_coeff).ndim == 3:
        raise NotImplementedError(
            'every BSE route here is restricted-spin: the kernels are spin '
            'adapted through kappa and the quasiparticle diagonals come from '
            'the restricted GW routes. An open-shell reference needs the '
            'unrestricted Casida path (LinearResponseSolver with '
            "spin_mode='unrestricted'), which has no Davidson action.")
    if solver == 'auto':
        # Resolved here and nowhere else, so every check below -- and every
        # route, keyword and info key -- sees a real solver.
        n_ov = nocc * (np.asarray(mf.mo_coeff).shape[-1] - nocc)
        solver = solver_choice(n_ov, tda=bool(route_kwargs.get('tda', False)))
    if solver == 'davidson' and integrals == 'full':
        raise NotImplementedError(
            "solver='davidson' with integrals='full' is the one combination "
            "with no implementation: the matrix-free block action contracts a "
            "THREE-index factor (df_block_action, isdf_block_action) and a "
            "four-index (pq|rs) tensor offers none -- while a calculation that "
            "can hold (pq|rs) can hold A and B too, so the exact tensor pairs "
            "with solver='dense'. Use solver='dense' to keep the exact "
            "integrals, or integrals='df'/'isdf' to keep the matrix-free solver.")
    if nroots is None and solver != 'dense':
        raise ValueError("nroots=None asks for the whole spectrum, which only "
                         "solver='dense' computes; give Davidson a root count.")
    _check_route_kwargs(solver, integrals, route_kwargs)

    route = dict(nroots=nroots, qp=qp, spin=spin, screen_at=screen_at,
                 self_consistency=self_consistency, **route_kwargs)
    if solver == 'dense':
        omega, X, Y, info = _solve_bse_dense(mf, mol, nocc, integrals, **route)
    elif integrals == 'isdf':
        omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, **route)
    else:
        omega, X, Y, info = solve_bse_df(mf, mol, nocc, **route)

    # One shape for every route: the keys a route cannot fill are present and
    # None, never a stand-in that reads like a measurement.
    info = {'factors': None, 'coeff_df': None, 'stats': None, 'n_states': None,
            **info, 'solver': solver, 'integrals': integrals}
    if spin != 'singlet':
        info['oscillator_strength'] = info['transition_dipole'] = None
    return omega, X, Y, info


def _check_route_kwargs(solver, integrals, route_kwargs):
    """Refuse a keyword belonging to a route other than the one chosen."""
    allowed = set(SHARED_KWARGS)
    allowed |= set(DAVIDSON_KWARGS if solver == 'davidson' else DENSE_KWARGS)
    if integrals == 'isdf':
        allowed |= set(ISDF_KWARGS)
        if solver != 'davidson':
            allowed -= set(ISDF_MPI_KWARGS)
    unknown = sorted(set(route_kwargs) - allowed)
    if not unknown:
        return
    owner = {key: route for route, keys in (("solver='davidson'", DAVIDSON_KWARGS),
                                            ("solver='dense'", DENSE_KWARGS),
                                            ("integrals='isdf'", ISDF_KWARGS),
                                            ("solver='davidson' with "
                                             "integrals='isdf'", ISDF_MPI_KWARGS))
             for key in keys}
    detail = ', '.join(f'{k} (belongs to {owner[k]})' if k in owner else k
                       for k in unknown)
    raise TypeError(f"unexpected keyword(s) for solver={solver!r} "
                    f"integrals={integrals!r}: {detail}")


def _bse_setup(mf, mol, nocc, integrals, qp, self_consistency, screen_at,
               gw_kwargs=None, factors=None, **fit_kwargs):
    """The quasiparticle diagonal, the integrals and the static W of one route.

    Everything the dense solver needs before it can build A and B, under
    `solve_bse_isdf`'s conventions: one factorization shared by the GW and BSE
    stages, W built at the MEAN-FIELD spectrum while the diagonal carries `qp`
    (the G0W0-BSE split), and W never taken in a different auxiliary gauge than
    the kernel it screens.

    fit_kwargs: `separable_factors` keywords, ISDF only.
    """
    if integrals == 'isdf' and factors is None:
        factors = separable_factors(mf, mol, **fit_kwargs)

    evgw_info = None
    if str(self_consistency).lower() in ('evgw', 'ev'):
        if integrals == 'isdf':
            qp, evgw_info = evgw_eigenvalues(mf, mol, mode='space-time',
                                             factors=factors, **(gw_kwargs or {}))
        else:
            qp, evgw_info = evgw_eigenvalues(mf, mol, mode='casida',
                                             df=(integrals == 'df'),
                                             **(gw_kwargs or {}))
        # W must screen with the spectrum the self-energy converged on, not the
        # mean field's; shifting the reference is what carries that downstream.
        mf = shifted_mean_field(mf, qp)
    elif str(self_consistency).lower() not in ('g0w0', 'none'):
        raise ValueError(f"self_consistency={self_consistency!r}; choose "
                         f"'G0W0' or 'evGW'.")

    eps_mf = get_orbital_energies(mf, representation='spatial')
    gw_extras = {}
    if qp is None or qp is False:
        eps = eps_mf
    elif isinstance(qp, str):
        if qp.upper() != 'G0W0':
            raise ValueError(f"qp='{qp}'; choose 'G0W0', an energy array, or False.")
        if integrals == 'isdf':
            # One self-energy for the whole diagonal, and it carries W(0) out.
            eps, _ = solve_qp_diagonal_space_time(mf, mol, nocc, factors=factors,
                                                  extras=gw_extras,
                                                  **(gw_kwargs or {}))
        else:
            out = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                                 mode='casida', df=(integrals == 'df'),
                                 state=list(range(len(eps_mf))),
                                 **(gw_kwargs or {}))
            eps = np.array([out[p]['GW'] for p in range(len(eps_mf))]) / HARTREE_TO_EV
    else:
        eps = np.asarray(qp, dtype=float)
        if eps.shape != eps_mf.shape:
            raise ValueError(f'qp energies have shape {eps.shape}, the mean '
                             f'field has {eps_mf.shape}.')
        if screen_at not in ('qp', 'mean-field'):
            raise ValueError(f"screen_at={screen_at!r}; choose 'qp' or 'mean-field'.")
        if screen_at == 'qp':
            mf = shifted_mean_field(mf, eps)

    if integrals == 'isdf':
        # The factorization's implied three-index factor: the bridge that puts
        # the dense builders in the ISDF fit's own auxiliary gauge.
        X_mo, D = _unpack_factors(factors)[:2]
        coeff, eri = isdf_df_coefficients(X_mo, D), None
    elif integrals == 'df':
        coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        eri = None
    else:
        coeff = None
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    eps_screen = get_orbital_energies(mf, representation='spatial')
    # The space-time self-energy already inverted [1 - chi0] at omega = 0, so
    # the kernel's static W is one of its slots rather than a second sweep.
    W_aux = gw_extras.get('w_static')
    if W_aux is None:
        W_aux = LinearResponseSolver(eps_screen, coeff_df=coeff, eri_chemist=eri,
                                     spin_mode='restricted').static_screening_aux(nocc)
    return dict(eps=eps, eps_screen=eps_screen, eps_mf=eps_mf, factors=factors,
                coeff_df=coeff, eri=eri, W_aux=W_aux, evgw=evgw_info)


def _solve_bse_dense(mf, mol, nocc, integrals, nroots=5, qp='G0W0',
                     spin='singlet', self_consistency='G0W0',
                     screen_at='mean-field', probe=True, tda=False,
                     gw_kwargs=None, **fit_kwargs):
    """BSE by building A and B and diagonalizing them: the whole spectrum, O(N^6).

    The route the vertex sums and any spectrum-wide property need, and the only
    one the four-index tensor runs on.
    """
    t = {}
    t0 = time.time()
    setup = _bse_setup(mf, mol, nocc, integrals, qp, self_consistency, screen_at,
                       gw_kwargs=gw_kwargs, **fit_kwargs)
    t['setup'] = time.time() - t0

    t0 = time.time()
    lr = LinearResponseSolver(setup['eps_screen'], coeff_df=setup['coeff_df'],
                              eri_chemist=setup['eri'], spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=setup['W_aux'],
                                    triplet=(spin == 'triplet'))
    # The quasiparticle energies enter A through its diagonal alone, so the
    # matrices are built at the SCREENING spectrum and the diagonal is moved
    # onto `qp` afterwards. Building them at `qp` instead would rescreen the
    # full-ERI route's direct term at the quasiparticle energies, which is the
    # G0W0-BSE split broken -- 21 meV on water/cc-pVDZ. The diagonal each
    # spectrum contributes is `bse_pair_diagonal`, the one rule the matrix-free
    # route reads straight off `lr_solver.eps`.
    diag_qp = bse_pair_diagonal(setup['eps'], nocc)
    diag_screen = bse_pair_diagonal(setup['eps_screen'], nocc)
    A = A + np.diag((diag_qp - diag_screen).ravel())

    amb = None
    if probe and not tda:
        amb = float(np.linalg.eigvalsh(A - B).min())
        if amb <= 0:
            raise RuntimeError(
                f'BSE refused before the solve: min eig(A-B) = {amb:.6f} Ha '
                '<= 0 -- the mean-field reference is singlet/triplet unstable '
                'and the Casida omega^2 reduction is invalid there. Check '
                'first that the SCF is the LOWEST solution (vary the initial '
                'guess, follow instabilities); if it is, stabilize the '
                'reference or solve the non-Hermitian problem -- no shift '
                'gives physical roots in this regime.')

    omega, X, Y = CasidaSolver(A, B).solve(tda=tda)
    t['dense'] = time.time() - t0

    n_states = len(omega)
    order = np.argsort(omega)
    if nroots is not None:
        order = order[:nroots]
    omega, X, Y = omega[order], X[:, order], Y[:, order]
    f_osc, trans_dip = oscillator_strengths(mf, mol, nocc, omega, X, Y)
    info = dict(eps=setup['eps'], eps_mf=setup['eps_mf'], W_aux=setup['W_aux'],
                min_eig_amb=amb, oscillator_strength=f_osc,
                transition_dipole=trans_dip, timings=t, evgw=setup['evgw'],
                n_states=n_states)
    if integrals == 'isdf':
        info['factors'] = setup['factors']
    elif integrals == 'df':
        info['coeff_df'] = setup['coeff_df']
    return omega, X, Y, info
