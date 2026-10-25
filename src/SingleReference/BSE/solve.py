"""Bethe-Salpeter excitation energies: every route behind one front end, and
the two matrix-free drivers (`solve_bse_isdf`, `solve_bse_df`: mean field in,
excitations out, the same conventions on two representations of the
interaction) with the quasiparticle diagonal they share (`qp_diagonal`).

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

from src.Base.numerics.constants import (
    BSE_SOLVERS, HARTREE_TO_EV, KAPPA, SEPARABLE_AUX_GAUGE)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist,
                                            require_closed_shell_or_unrestricted)
from src.Base.factorization.sliced_factors import SlicedFactors, unpack_factors
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.Base.utils.parallel.mpi_grid import (current_comm, grid_comm,
                                              lockstep_mean_field, replicate)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.SingleReference.GW.evGW import evgw_eigenvalues, shifted_mean_field
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import (replicate_factors,
                                               separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.LinearResponse.block_actions import bse_pair_diagonal
from src.SingleReference.LinearResponse.casida import CasidaSolver, solver_choice
from src.SingleReference.LinearResponse.davidson import (
    CasidaBreakdown, bse_settings, casida_action, davidson_flags,
    probe_after_solve, run_davidson, solve_casida_davidson,
    unstable_reference)
from src.SingleReference.LinearResponse.davidson_shared import check_preconditioner
from src.SingleReference.LinearResponse.exciton_descriptors import exciton_descriptors
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, check_normalization)
from src.SingleReference.LinearResponse.probe import lowest_amb_eigenvalue
from src.SingleReference.LinearResponse.space_time import (
    static_screening_grid, static_screening_matrix)

INTEGRALS = ('isdf', 'df', 'full')

#: Keywords each half of the route matrix accepts on top of the front end's own.
DAVIDSON_KWARGS = ('progress',)
DENSE_KWARGS = ()
ISDF_KWARGS = ('factors', 'auxbasis', 'radii', 'counts', 'n_start',
               'grid_accuracy', 'distribute', 'comm')
#: The two of those that only the matrix-free ISDF route can act on: it is the
#: Davidson block action and the GW tau axis that are divided over ranks, and
#: the separable fit itself takes neither keyword. The dense route refuses them
#: rather than forward a distributed request into a serial solve.
ISDF_MPI_KWARGS = ('distribute', 'comm')
SHARED_KWARGS = ('probe', 'gw_kwargs', 'tda')




def solve_bse(mf, mol=None, nocc=None, numerics=None, solver='davidson',
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
                'mean-field' (default, the standard G0W0-BSE split) or 'qp',
                refused beside anything but an array. See `solve_bse_isdf`.
    numerics:   the choices (`Numerics`; None for the defaults): `bse.nroots`
                is the roots wanted, None the whole spectrum and dense-only;
                the Davidson routes read its residual, cycle cap, guess
                count and preconditioner (`davidson.bse_settings`).
    probe:      off (False) by default on every route. True or 'sign'
                measures min eig(A - B) and refuses the solve while it is
                negative, the mean-field instability regime where the Casida
                omega^2 reduction is invalid. Davidson estimates it matrix-free
                ('sign' stops as soon as the sign is proven) and on large runs
                it can cost more than the solve; the dense route reads it
                exactly off the A and B it just built. Without it the Davidson
                routes still refuse an unstable reference when their projected
                (A - B) breaks down. info['min_eig_amb'] is None when it did
                not run. Every route skips it under `tda`, which never makes
                that reduction.

    Route-specific keywords, rejected rather than ignored when they belong to a
    route that was not chosen: `progress` is the Davidson iteration's; `factors`, `auxbasis`, `radii`, `counts`, `n_start`
    and `grid_accuracy` belong to the separable fit, and `distribute`/`comm` to the
    matrix-free ISDF route that divides the tau axis and the rows of Zt over
    MPI ranks. `gw_kwargs` reaches the GW step on any route, and `tda`
    (the Tamm-Dancoff BSE, A X = omega X with Y = 0) every solver: the
    Davidson routes then diagonalize the A half of their block action.

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
    require_closed_shell_or_unrestricted(mf, 'solve_bse', mol=mol)
    nocc = mol.nelectron // 2 if nocc is None else nocc
    solver = str(solver).lower()
    integrals = str(integrals).lower()
    if solver not in BSE_SOLVERS:
        raise ValueError(f"solver={solver!r}; choose one of {list(BSE_SOLVERS)}.")
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
        solver = solver_choice(n_ov)
    if solver == 'davidson' and integrals == 'full':
        raise NotImplementedError(
            "solver='davidson' with integrals='full' is the one combination "
            "with no implementation: the matrix-free block action contracts a "
            "THREE-index factor (df_block_action, isdf_block_action) and a "
            "four-index (pq|rs) tensor offers none -- while a calculation that "
            "can hold (pq|rs) can hold A and B too, so the exact tensor pairs "
            "with solver='dense'. Use solver='dense' to keep the exact "
            "integrals, or integrals='df'/'isdf' to keep the matrix-free solver.")
    nroots = numerics_or_default(numerics).bse.nroots
    if nroots is None and solver != 'dense':
        raise ValueError("nroots=None asks for the whole spectrum, which only "
                         "solver='dense' computes; give Davidson a root count.")
    _check_route_kwargs(solver, integrals, route_kwargs)

    route = dict(qp=qp, spin=spin, screen_at=screen_at,
                 self_consistency=self_consistency, **route_kwargs)
    if solver == 'dense':
        omega, X, Y, info = _solve_bse_dense(mf, mol, nocc, integrals,
                                             nroots, numerics, **route)
    elif integrals == 'isdf':
        omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, numerics=numerics,
                                           **route)
    else:
        omega, X, Y, info = solve_bse_df(mf, mol, nocc, numerics=numerics,
                                         **route)

    # One shape for every route: the keys a route cannot fill are present and
    # None, never a stand-in that reads like a measurement.
    # A Davidson root left unconverged is returned as solved and named in
    # 'davidson_flags' (refuse inputs, warn numerics); the dense route has none to name.
    info = {'factors': None, 'coeff_df': None, 'stats': None, 'n_states': None,
            'davidson_flags': {}, **info, 'solver': solver,
            'integrals': integrals}
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
               numerics, gw_kwargs=None, factors=None, **fit_kwargs):
    """The quasiparticle diagonal, the integrals and the static W of one route.

    Everything the dense solver needs before it can build A and B, under
    `solve_bse_isdf`'s conventions: one factorization shared by the GW and BSE
    stages, W built at the MEAN-FIELD spectrum while the diagonal carries `qp`
    (the G0W0-BSE split), and W never taken in a different auxiliary gauge than
    the kernel it screens.

    fit_kwargs: `separable_factors` keywords, ISDF only. numerics: the
    choices the GW stage, the fit and the static W read alike.
    """
    numerics = numerics_or_default(numerics)
    # every GW stage reads the one object: the space-time route its grids,
    # the evGW loop its settings
    refuse_second_numerics(gw_kwargs)
    if integrals == 'isdf' and factors is None:
        factors = separable_factors(mf, mol, numerics=numerics, **fit_kwargs)

    diagonal = qp_diagonal(mf, mol, nocc, integrals, qp, self_consistency,
                           screen_at, dict(gw_kwargs or {}, numerics=numerics),
                           factors=factors)
    eps, eps_mf, mf = diagonal['eps'], diagonal['eps_mf'], diagonal['mf']
    gw_extras, evgw_info = diagonal['extras'], diagonal['evgw']

    if integrals == 'isdf':
        # The factorization's implied three-index factor: the bridge that puts
        # the dense builders in the ISDF fit's own auxiliary gauge.
        X_mo, D = unpack_factors(factors)[:2]
        coeff, eri = isdf_df_coefficients(X_mo, D), None
    elif integrals == 'df':
        coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        eri = None
    else:
        coeff = None
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    eps_screen = get_orbital_energies(mf, representation='spatial')
    # On the separable factors W(0) is the one static build the Davidson route
    # and the space-time GW stage read; the DF and four-index routes screen
    # exactly on their own integrals.
    W_aux = gw_extras.get('w_static')
    if W_aux is None and integrals == 'isdf':
        q = numerics.quadrature
        W_aux = static_screening_matrix(X_mo, D, eps_screen, nocc,
                                        static_screening_grid(
                                            eps_screen, nocc, ntau=q.ntau_w,
                                            tau_target=q.tau_target))
    elif W_aux is None:
        W_aux = LinearResponseSolver(eps_screen, coeff_df=coeff, eri_chemist=eri,
                                     spin_mode='restricted').static_screening_aux(nocc)
    return dict(eps=eps, eps_screen=eps_screen, eps_mf=eps_mf, factors=factors,
                coeff_df=coeff, eri=eri, W_aux=W_aux, evgw=evgw_info)


def _solve_bse_dense(mf, mol, nocc, integrals, nroots, numerics, qp='G0W0',
                     spin='singlet', self_consistency='G0W0',
                     screen_at='mean-field', probe=False, tda=False,
                     gw_kwargs=None, **fit_kwargs):
    """BSE by building A and B and diagonalizing them: the whole spectrum, O(N^6).

    The route the vertex sums and any spectrum-wide property need, and the only
    one the four-index tensor runs on.
    """
    t = {}
    t0 = time.time()
    setup = _bse_setup(mf, mol, nocc, integrals, qp, self_consistency, screen_at,
                       numerics, gw_kwargs=gw_kwargs, **fit_kwargs)
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
            raise unstable_reference(amb, 'before the solve')

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
                n_states=n_states,
                numerics_identity=numerics_or_default(numerics).identity())
    if integrals == 'isdf':
        info['factors'] = setup['factors']
    elif integrals == 'df':
        info['coeff_df'] = setup['coeff_df']
    return omega, X, Y, info


#: `GW/space_time.py`'s own `timings` keys, renamed for `info['timings']`'s
#: `qp_` prefix: `t_qp_static`/`t_qp_states` fold their `t_qp_` in with the
#: shared prefix rather than doubling it. `t_qp` -- `finish_qp`'s own total,
#: static + states -- carries nothing beyond the two keys it sums and is not
#: re-exposed under a third name.
QP_TIMING_RENAME = {'t_chi0': 'qp_chi0', 't_dyson': 'qp_dyson',
                    't_sigma': 'qp_sigma', 't_qp_static': 'qp_static',
                    't_qp_states': 'qp_states'}


class _QPStageTimings:
    """`timings=` target for the space-time QP solve, G0W0 or one evGW cycle.

    `solve_qp_energy_space_time`/`_qp_blocked`/`finish_qp` write each
    per-stage key once per call with a plain assignment, and evGW re-enters
    that call once per cycle on the same object (`gw_kwargs['timings']` is
    fixed for the whole loop in `evgw_eigenvalues`), so a plain dict would
    keep only the last cycle's numbers. `__setitem__` sums the per-stage keys
    instead: one call for `qp='G0W0'`, one per evGW cycle. Everything else
    (`nranks`, `ntau_auto`, `tau_fit_error`) takes the latest write. A
    caller-supplied dict (`gw_kwargs['timings']`) is written into directly,
    never copied, so it carries the final numbers under its own name.
    """
    _ADDITIVE = frozenset({'t_chi0', 't_dyson', 't_sigma', 't_qp_static',
                           't_qp_states', 't_qp'})

    def __init__(self, target=None):
        self.target = target if target is not None else {}

    def __setitem__(self, key, value):
        if key in self._ADDITIVE:
            self.target[key] = self.target.get(key, 0.0) + value
        else:
            self.target[key] = value


def refuse_second_numerics(gw_kwargs):
    """Refuse a `Numerics` inside `gw_kwargs`: an entry point holds ONE and
    hands it to its GW stage itself, so a second would be two answers to
    which grid the self-energy ran on."""
    if gw_kwargs and 'numerics' in gw_kwargs:
        raise TypeError('gw_kwargs holds a numerics of its own; pass the one '
                        "Numerics as the entry point's `numerics`, which its "
                        'GW stage reads too')


def isdf_df_coefficients(X_mo, D):
    """The (naux, norb, norb) DF factor the separable factorization implies,
    B[A,p,q] = sum_k X[k,p] X[k,q] D[k,A].

    O(M naux norb^2) to build and naux norb^2 to hold, so it defeats the point
    of the factorization at scale. It is the bridge to every DF consumer:
    `build_casida_matrices`, `static_screening_aux` and the dense CasidaSolver
    all speak B, and feeding them this B puts them in the ISDF factorization's
    own gauge, the only setting in which they and the matrix-free ISDF path
    compare the same operator.
    """
    return np.einsum('Pp,Pq,PA->Apq', X_mo, X_mo, D, optimize=True)


def oscillator_strengths(mf, mol, nocc, omega, X, Y):
    """Length-gauge oscillator strengths and transition dipoles for the
    solver's roots:

        f_n = (2/3) omega_n |<0|r|n>|^2,
        <0|r|n> = sqrt(2) sum_ia (X + Y)_{ia,n} <i|r|a>

    the spin-adapted singlet with this module's <X|X>-<Y|Y>=1 spatial
    normalization (validated against pyscf's TDHF `oscillator_strength` on the
    same DF operator). Transition moments are origin-independent, so the gauge
    origin is fixed at zero only for definiteness.

    Returns (f, dip) with shapes (nroots,) and (nroots, 3), atomic units.

    The normalization is checked, not assumed: the sqrt(2) above belongs to
    <X|X> - <Y|Y> = 1, pySCF returns 1/2, and f is quadratic in the vector, so
    the mismatch is a silent factor of two in every oscillator strength.
    """
    X, Y = check_normalization(X, Y)
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    mo = mf.mo_coeff
    d_ov = np.einsum('xmn,mi,na->xia', ao_dip, mo[:, occ], mo[:, virt],
                     optimize=True).reshape(3, -1)
    dip = np.sqrt(2.0) * (d_ov @ (X + Y)).T                  # (nroots, 3)
    f = (2.0 / 3.0) * np.asarray(omega) * np.einsum('nx,nx->n', dip, dip)
    return f, dip


def _takes_comm(distribute, comm):
    """Whether an entry point runs over a communicator: `distribute` None
    follows `comm`, True asks for one (COMM_WORLD failing `comm`), False
    takes none."""
    return bool(distribute) or (distribute is None and comm is not None)


def stage_clock(t, progress, tag):
    """(begin, end) of a driver's timed stages: begin(name) starts one and,
    with `progress`, prints it under `tag`; end(name, t0) files its wall
    time in t[name] and prints that."""
    def begin(name):
        if progress:
            print(f'[{tag} {time.strftime("%H:%M:%S")}] {name} ...', flush=True)
        return time.time()

    def end(name, t0):
        t[name] = time.time() - t0
        if progress:
            print(f'[{tag} {time.strftime("%H:%M:%S")}] {name} done, '
                  f'{t[name]:.1f} s', flush=True)
    return begin, end


def qp_diagonal(mf, mol, nocc, integrals, qp, self_consistency, screen_at,
                gw_kwargs, factors=None, clock=None, progress=False,
                tag='bse'):
    """{'eps', 'eps_mf', 'mf', 'extras', 'evgw'}: the quasiparticle diagonal
    of a BSE on one route, and the mean field whose spectrum its static W
    screens with -- the one rule of every BSE driver.

    integrals: 'isdf' runs the space-time GW on `factors`, `gw_kwargs` being
    its keywords (the run's Numerics among them, which the evGW loop reads on
    every route); 'df' and 'full' the Casida GW
    on pyscf's fitted or four-index integrals. self_consistency 'evGW' puts
    the loop's fixed point on the diagonal and screens there (`gw_kwargs`
    may not choose its 'screening': the kernel screens at the fixed point,
    which is evGW's W); 'G0W0' takes `qp`: 'G0W0' solves every state (the
    space-time solve carries W(0) out in 'extras'), an array is the diagonal
    as given, screened at it under screen_at='qp' and at the mean field's
    spectrum under 'mean-field' (the G0W0-BSE split), None/False is
    BSE@mean-field. clock: `stage_clock`'s (begin, end), timing 'evgw' and
    'qp'; progress prints the loop's verdict under `tag`.
    """
    begin, end = clock or (lambda name: None, lambda name, t0: None)
    gw_kwargs = gw_kwargs or {}
    if screen_at not in ('qp', 'mean-field'):
        raise ValueError(f"screen_at={screen_at!r}; choose 'qp' or "
                         f"'mean-field'.")
    explicit = not (qp is None or qp is False or isinstance(qp, str))
    if screen_at == 'qp' and (not explicit or str(self_consistency).lower()
                              in ('evgw', 'ev')):
        raise ValueError(
            "screen_at='qp' screens at an explicit energy array and is read "
            "only beside one: qp='G0W0' screens with its own chi0 and evGW at "
            "its fixed point, and BSE@mean-field at the mean field")
    evgw_info, extras = None, {}
    if str(self_consistency).lower() in ('evgw', 'ev'):
        if 'screening' in gw_kwargs:
            raise ValueError("gw_kwargs: self_consistency='evGW' screens the BSE "
                             "at the loop's fixed point, so 'screening' has no "
                             "place here")
        t0 = begin('evgw')
        if integrals == 'isdf':
            # the fit is geometry-only: the loop reuses it every cycle
            qp, evgw_info = evgw_eigenvalues(mf, mol, mode='space-time',
                                             factors=factors, **gw_kwargs)
        else:
            qp, evgw_info = evgw_eigenvalues(mf, mol, mode='casida',
                                             df=(integrals == 'df'),
                                             **gw_kwargs)
        # W must screen with the spectrum the self-energy converged on, not the
        # mean field's; shifting the reference is what carries that downstream.
        mf = shifted_mean_field(mf, qp)
        end('evgw', t0)
        if progress:
            print(f'[{tag} {time.strftime("%H:%M:%S")}] evGW '
                  f'{"converged" if evgw_info["converged"] else "NOT converged"}'
                  f' in {evgw_info["cycles"]} cycles', flush=True)
    elif str(self_consistency).lower() not in ('g0w0', 'none'):
        raise ValueError(f"self_consistency={self_consistency!r}; choose "
                         f"'G0W0' or 'evGW'.")

    eps_mf = get_orbital_energies(mf, representation='spatial')
    if qp is None or qp is False:
        eps = eps_mf
    elif isinstance(qp, str):
        if qp.upper() != 'G0W0':
            raise ValueError(f"qp='{qp}'; choose 'G0W0', an energy array, or False.")
        t0 = begin('qp')
        if integrals == 'isdf':
            # one self-energy for the whole diagonal, and it carries W(0) out
            eps, _ = solve_qp_diagonal_space_time(mf, mol, nocc, factors=factors,
                                                  extras=extras, **gw_kwargs)
        else:
            out = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                                 mode='casida', df=(integrals == 'df'),
                                 state=list(range(len(eps_mf))), **gw_kwargs)
            eps = (np.array([out[p]['GW'] for p in range(len(eps_mf))])
                   / HARTREE_TO_EV)
        end('qp', t0)
    else:
        eps = np.asarray(qp, dtype=float)
        if eps.shape != eps_mf.shape:
            raise ValueError(f'qp energies have shape {eps.shape}, the mean '
                             f'field has {eps_mf.shape}.')
        if screen_at == 'qp':
            # An array that is a self-consistent spectrum: W follows it, the
            # way evGW's does. The standard G0W0 split is the default.
            mf = shifted_mean_field(mf, eps)
    return {'eps': eps, 'eps_mf': eps_mf, 'mf': mf, 'extras': extras,
            'evgw': evgw_info}


def solve_bse_isdf(mf, mol, nocc, numerics=None, qp='G0W0', factors=None,
                   auxbasis=None, radii=None, counts=None, probe=False,
                   gw_kwargs=None, progress=None, n_start=None,
                   self_consistency='G0W0', screen_at='mean-field',
                   spin='singlet', grid_accuracy=None, distribute=None,
                   comm=None, sigma_x_matrix=None, W_aux=None,
                   W_aux_gauge=None, guess=None, tda=False,
                   refuse_unconverged=False):
    """BSE by the ISDF matrix-free Davidson: mean field in, excitations out.

    The calling sequence:

        mf = dft.RKS(mol, xc=...).density_fit(auxbasis=...); mf.kernel()
        omega, X, Y, info = solve_bse_isdf(mf, mol, nocc)

    The discipline the pieces demand lives here so the caller cannot violate
    it: one ISDF fit is built (or taken from `factors`) and shared by the GW
    and BSE stages; the static W comes from that fit at the eigenvalues the
    level of theory screens with (the mean field's for G0W0, the standard
    split, and the converged ones under evGW); and W and the block action
    cannot mix auxiliary gauges.

    self_consistency: 'G0W0' (default) evaluates the self-energy once on the
        mean field's eigenvalues. 'evGW' runs the eigenvalue-self-consistent
        loop first (all eigenvalues updated, DIIS, convergence on the HOMO
        and LUMO), puts its fixed point on the BSE diagonal and builds the
        static W from it, so the kernel screens with the same spectrum the
        self-energy did. It overrides `qp` and shares the ISDF fit.
    screen_at: which spectrum builds the static W when `qp` is an explicit
        energy array. The convention is the level of theory, not the
        diagonal: G0W0 screens W0 at the mean-field eigenvalues and evGW at
        the converged ones. 'mean-field' (default) is the standard G0W0-BSE
        split, for an array of G0W0 energies computed elsewhere. 'qp' screens
        at the array instead, for an array that is a self-consistent spectrum
        (evGW energies from a separate loop). The two differ: on water
        /cc-pVDZ the full-integral and DF routes agree to +0.6 meV when the
        screening and the diagonal match and drift -65 meV when they do not.

        Quasiparticle energies plus a further shift are neither: evGW
        energies carrying a static reaction-field correction on top are not
        the evGW spectrum, so screening at them satisfies no convention
        (119 meV on acrolein's n->pi*). Leave the default and let the
        reference object carry the spectrum.

        Neither value touches `qp='G0W0'`, whose W comes from the GW step's own
        chi0 at mean-field energies, nor `self_consistency='evGW'`, which always
        screens at its fixed point.

    qp: 'G0W0' (default) puts every quasiparticle energy from one space-time
    self-energy on the BSE diagonal (`solve_qp_diagonal_space_time`, which
    applies <Sigma_x - v_xc> per state for a KS reference); an array of
    energies is used as the diagonal directly (an evGW result, say, or the
    G0W0 diagonal a caller already solved, handed over with `W_aux`);
    False/None solves BSE@mean-field.
    spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0); the screened term is
    spin-independent, so both read the same W.
    probe: off (False) by default. True or 'sign' measures min eig(A-B) after
    the Davidson, on its own block action, and refuses the roots while it is
    negative (the instability regime, where the omega^2 reduction is
    invalid); on large DF/ISDF runs it can cost more than the solve. An
    unstable reference is refused anyway, probe or not, when the Davidson's
    projected (A-B) block stops being positive definite, and the measurement
    that breakdown runs names the reference. True converges the eigenvalue by
    the symmetric Davidson from 1/d. 'sign' stops once the sign is proven by the
    residual bound: first by Rayleigh-Ritz on the span of the roots' X - Y
    (which proves an eigenvalue in that span, not the minimum over pair
    irreps no root lies in), then by a Davidson from the lowest root with
    PROBE_START_MIX of 1/d at 1e-2, then from 1/d at escalating tolerances.
    The quantity is a property of the SCF solution, not the molecule, so
    probe once per reference that is in doubt.
    gw_kwargs: forwarded to `solve_qp_diagonal_space_time` (the shared factors
    are always passed).
    sigma_x_matrix: <p|Sigma_x - v_xc|q> already built for this mean field. It
    depends on the density and orbitals alone, so one build (one K plus the xc
    potential) serves this BSE, an earlier quasiparticle window and every
    evGW cycle. Left None, the GW stage builds it once here, timed as its own
    `sigma_x` stage.
    W_aux: the kernel's static screening W(omega = 0) on `factors`, already
    built: `extras['w_static']` of the `solve_qp_diagonal_space_time` call
    whose diagonal is handed in as `qp`, which then gives the same roots bit
    for bit without a second chi0 and self-energy sweep. Refused beside
    `qp='G0W0'`, `self_consistency='evGW'` and `screen_at='qp'`, which each
    build their own W.
    W_aux_gauge: the auxiliary gauge W_aux was built in, required beside it and
    refused unless SEPARABLE_AUX_GAUGE: a W from pyscf's cderi (`solve_bse_df`)
    has the factors' naux and a rotated auxiliary index.
    distribute, comm: split the M^2 work over the ranks of `comm`, by default
    the current `distributed` region's. None (the default) follows that
    communicator; True also falls back to COMM_WORLD; False gives this call no
    communicator of its own (no replication, no COMM_WORLD), while the kernels
    it calls still follow a `distributed` region around it -- a serial solve
    inside one is `with distributed(None):`. What is divided is the tau axis of
    chi0 and Sigma in the GW stage and of the static W, and the rows of Zt in
    the block action, so the one M^2 object's per-rank memory falls with the
    rank count. Exact up to summation order; one rank prints.

    The answer is rank 0's. Everything the ranks contract into a reduced sum
    is replicated from rank 0 first -- the mean field, the ISDF factors, the
    quasiparticle diagonal and the static W -- and their own copies are
    overwritten, because partials of two slightly different calculations do
    not add. The Davidson and the (A-B) probe then run on every rank on the
    block action's lockstepped output, and their results are lockstepped from
    rank 0 at the end, as is the probe's decision whether the roots alone
    proved the sign. The trial space is sized from the job's memory: inside a
    SLURM allocation from `mf.max_memory`, which a job sets to the
    allocation's share, less the block action's working set, counted on the
    pair rows a rank holds (`trial_space_memory`); outside one within
    DAVIDSON_SPACE_GB. On a slow interconnect weigh the GW stage's
    all-reduce of nfreq x naux^2 first: on 1 GbE it costs more than it saves
    (`Base.utils.parallel.mpi_grid`).
    grid_accuracy: the named way to ask for the interpolation grid: an
    accuracy level of `ISDF_GRID_ACCURACY`, measured on this quantity
    (the three lowest BSE roots against `solve_bse_df`), or four explicit shell
    counts. It reaches `separable_factors`, sets `counts` and `n_start`
    together, and aborts on anything the radii table has not got.
    Factors built here are refused where they miss an orbital density of
    `mf` (`separable_factors`, at the bar ISDF_ORBITAL_RESIDUAL_TOL); factors
    handed in were measured where they were built.

    progress: stamp each stage to stdout as it begins and ends, so a long job
    says which stage it is in while it is in it (the returned timings arrive
    only if the run finishes). None follows `mol.verbose`.

    numerics.bse.preconditioner: the diagonal the Davidson's corrections
    divide by, 'bare' (the pair energies d) or 'screened' (d - (ii|W|aa) in
    the fit's own gauge, one reduction of this rank's grid rows' partial);
    refused before the mean field is touched when it is neither
    (`run_davidson`).

    guess: (X, Y) of this spin to warm-start the Davidson from, carried into
    this mean field's orbitals (`DavidsonWarmStart.guess`); `run_davidson`'s.

    tda: the Tamm-Dancoff BSE, A X = omega X with B = 0 and Y returned zero,
    on the A half of the same block action (`solve_casida_davidson`);
    `probe` is skipped, as on the dense route, since that problem makes no
    omega^2 reduction.

    refuse_unconverged: False (the default) returns a root left above
    conv_tol as solved, warned about and named in info['davidson_flags']
    (`davidson_flags`); True raises `DavidsonNotConverged`.
    `run_davidson`'s.

    Returns (omega, X, Y, info); info carries eps / eps_mf / factors / W_aux /
    min_eig_amb (None when the probe did not run; under 'sign' the Ritz value
    whose sign was proven, an upper bound on the minimum), length-gauge
    oscillator_strength and transition_dipole per root
    (`oscillator_strengths`), exciton_descriptors, per-stage timings in
    seconds and `nranks`; `stats` holds the probe's record (`probe_source`,
    `probe_matvecs`). info['tracked'] holds 'omega', 'X' and 'Y' of the roots
    and the tracked Ritz pairs above them, what a `DavidsonWarmStart` carries
    to the next geometry; info['davidson_flags'] names roots left
    unconverged or unconfirmed (`davidson_flags`, empty when clean). The
    Davidson stage carries its breakdown
    (`run_davidson`); the `qp` stage, when the space-time solve runs,
    carries `qp_chi0`, `qp_dyson`, `qp_sigma`, `qp_static` and `qp_states`
    (`_QPStageTimings`, summed over evGW's `qp_cycles`).
    """
    require_closed_shell_or_unrestricted(mf, 'solve_bse_isdf', mol=mol)
    nroots, conv_tol, max_cycle, guess_factor, preconditioner = bse_settings(
        numerics, 'nroots', 'conv_tol', 'max_cycle', 'guess_factor',
        'preconditioner')
    t = {}
    stats = {}
    tracked = {}
    check_preconditioner(preconditioner)
    if W_aux is not None and (isinstance(qp, str) or screen_at == 'qp' or
                              str(self_consistency).lower() in ('evgw', 'ev')):
        raise ValueError(
            f'W_aux handed in beside qp={qp!r}, screen_at={screen_at!r}, '
            f'self_consistency={self_consistency!r}, each of which builds its '
            'own W; hand the G0W0 diagonal in as qp= with the W(0) its solve '
            'carried out.')
    if W_aux is not None and W_aux_gauge != SEPARABLE_AUX_GAUGE:
        raise ValueError(
            f'W_aux handed in with W_aux_gauge={W_aux_gauge!r}: the kernel '
            f'contracts it with the separable factors, whose auxiliary index '
            f'is in the {SEPARABLE_AUX_GAUGE!r} (V^-1/2) gauge, and a W from '
            f"pyscf's L^-1-whitened cderi has the same shape and is wrong. "
            f'Hand a W built on these factors (`isdf_bse_factors`, the '
            f"space-time GW's extras['w_static']) with "
            f'W_aux_gauge={SEPARABLE_AUX_GAUGE!r}.')
    if W_aux is None and W_aux_gauge is not None:
        raise TypeError('W_aux_gauge is read only beside W_aux')
    comm = current_comm() if comm is None else comm
    mpi_comm, rank, nranks = (grid_comm(comm) if _takes_comm(distribute, comm)
                              else (None, 0, 1))
    # Rank 0's mean field, before a grid size or an interpolation point count
    # is read off it anywhere below.
    if nranks > 1:
        lockstep_mean_field(mf, mpi_comm)
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0
    progress = bool(progress) and rank == 0        # lockstep ranks, one voice
    # the GW stage and the static W read the one object
    refuse_second_numerics(gw_kwargs)
    gw_kw = dict(gw_kwargs or {}, numerics=numerics)
    if nranks > 1:
        gw_kw.update(distribute=True, comm=mpi_comm)

    _begin, _end = stage_clock(t, progress, 'bse')

    if progress:
        print(f'[bse {time.strftime("%H:%M:%S")}] natm={mol.natm} '
              f'nao={mol.nao_nr()} nocc={nocc} nroots={nroots} qp={qp}',
              flush=True)

    if factors is None:
        t0 = _begin('factors')
        factors = separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                    counts=counts, n_start=n_start,
                                    grid_accuracy=grid_accuracy,
                                    numerics=numerics)
        _end('factors', t0)
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] ISDF M={factors[1].shape[0]} '
                  f'({factors[1].shape[0] // mol.natm}/atom)', flush=True)
    # Rank 0's fit, whether it was built here or handed in: every rank owns a
    # block of the rows of Zt, and rows built from two different point sets are
    # not partials of one matrix.
    if nranks > 1:
        replicate_factors(factors, mpi_comm)

    # <Sigma_x - v_xc> is a functional of the density and the orbitals, so it
    # is built once, from the mean field before any spectrum shifts it, and
    # handed to every quasiparticle call below -- the G0W0 diagonal, or each
    # evGW cycle, which would otherwise rebuild the same K and the same
    # DFT-grid potential per cycle. 'df-direct' forms no such matrix.
    runs_gw = (str(self_consistency).lower() in ('evgw', 'ev')
               or isinstance(qp, str))
    # chi0/Dyson/sigma/static-exchange/root-search timings from the space-time
    # QP solve below, merged into `t` under a `qp_` prefix once the spectrum
    # is settled. `_QPStageTimings` sums across evGW's cycles rather than
    # keeping only the last one's numbers.
    qp_stage = _QPStageTimings(gw_kw.get('timings')) if runs_gw else None
    if qp_stage is not None:
        gw_kw['timings'] = qp_stage
    if (runs_gw and sigma_x_matrix is None
            and gw_kw.get('sigma_x', 'mf') != 'df-direct'):
        t0 = _begin('sigma_x')
        sigma_x_matrix = static_exchange_mean_field_matrix(
            mf, mol, dm_correction=gw_kw.get('dm_correction'),
            exchange=gw_kw.get('sigma_x', 'mf'))
        _end('sigma_x', t0)
    if runs_gw and sigma_x_matrix is not None:
        gw_kw['sigma_x_matrix'] = sigma_x_matrix

    diagonal = qp_diagonal(mf, mol, nocc, 'isdf', qp, self_consistency,
                           screen_at, gw_kw, factors=factors,
                           clock=(_begin, _end), progress=progress, tag='bse')
    eps, eps_mf, mf = diagonal['eps'], diagonal['eps_mf'], diagonal['mf']
    gw_extras, evgw_info = diagonal['extras'], diagonal['evgw']

    if qp_stage is not None:
        for key, value in qp_stage.target.items():
            if key == 't_qp':
                continue
            t[QP_TIMING_RENAME.get(key, 'qp_' + key)] = value
        if evgw_info is not None:
            t['qp_cycles'] = evgw_info['cycles']

    t0 = _begin('W')
    # The GW route built W(0) with the same `static_screening_matrix` on the
    # same grid, so a second sweep would repeat it bit for bit.
    w_shared = gw_extras.get('w_static')
    if W_aux is not None:
        naux = (factors.naux if isinstance(factors, SlicedFactors)
                else factors[1].shape[1])
        if np.shape(W_aux) != (naux, naux):
            raise ValueError(f'W_aux has shape {np.shape(W_aux)}; the factors '
                             f'screen in naux = {naux}.')
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] static W handed in, not '
                  'rebuilt', flush=True)
    elif w_shared is not None:
        W_aux = w_shared
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] static W taken from the '
                  f"GW stage (ntau={gw_extras.get('w_static_ntau')}), "
                  'not rebuilt', flush=True)
    else:
        if progress and gw_extras.get('w_static_unavailable'):
            print(f'[bse {time.strftime("%H:%M:%S")}] rebuilding the static W: '
                  f"{gw_extras['w_static_unavailable']}", flush=True)
        _, _, W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors,
                                       numerics=numerics,
                                       distribute=nranks > 1, comm=mpi_comm)
    _end('W', t0)
    # The last two objects the block action reads: the BSE diagonal, and the
    # kernel each rank dresses its own rows of Zt with. Both come off a
    # replicated solve on reduced data -- a Dyson inversion, a Pade
    # continuation and a Newton -- which is bitwise across ranks only while
    # their dense arithmetic is. naux^2 and nmo doubles to make certain.
    if nranks > 1:
        eps, W_aux = replicate(np.asarray(eps, float), W_aux, comm=mpi_comm)

    lr = LinearResponseSolver(eps, spin_mode='restricted')
    # No communicator of the call's own leaves the kernels to the region's.
    action_comm = current_comm() if mpi_comm is None else mpi_comm
    t0 = _begin('davidson')
    # One block action for the Davidson and the (A-B) probe after it: Zt, and
    # the D and X_o sliced factors gather whole, are built once for both.
    apply_AB, diag_d = casida_action(lr, nocc, 'BSE', W_aux, factors,
                                      conv_tol, spin=spin, comm=action_comm,
                                      timings=t, tda=tda)
    try:
        omega, X, Y = run_davidson(apply_AB, diag_d, nroots, conv_tol,
                                    max_cycle, None, guess_factor, stats=stats,
                                    comm=action_comm, timings=t,
                                    max_memory=_job_max_memory(mf),
                                    preconditioner=preconditioner,
                                    guess=guess, tda=tda, tracked=tracked,
                                    refuse_unconverged=refuse_unconverged)
    except CasidaBreakdown as exc:
        # The projected (A-B) broke down and the measurement it ran says the
        # reference, not the subspace, is what failed; named with or without
        # `probe`, since the breakdown has already paid for it.
        if exc.min_eig_amb is not None and exc.min_eig_amb <= 0:
            raise unstable_reference(exc.min_eig_amb) from exc
        raise
    _end('davidson', t0)
    amb = None
    if probe and not tda:
        t0 = _begin('probe')
        try:
            amb = probe_after_solve(apply_AB, diag_d, omega, X, Y, probe,
                                    stats=stats, comm=action_comm)
        finally:
            _end('probe', t0)
    if progress and stats:
        print(f'[bse {time.strftime("%H:%M:%S")}] ' + '  '.join(
            f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
            for k, v in sorted(stats.items())), flush=True)
    f_osc, trans_dip = oscillator_strengths(mf, mol, nocc, omega, X, Y)
    info = dict(eps=eps, eps_mf=eps_mf, factors=factors, W_aux=W_aux,
                min_eig_amb=amb, oscillator_strength=f_osc,
                transition_dipole=trans_dip,
                exciton_descriptors=exciton_descriptors(mf, mol, nocc, X, Y),
                timings=t, stats=stats, evgw=evgw_info, nranks=nranks,
                tracked=tracked, davidson_flags=davidson_flags(t),
                numerics_identity=numerics_or_default(numerics).identity())
    return omega, X, Y, info


def solve_bse_df(mf, mol, nocc, numerics=None, qp='G0W0', probe=False,
                 gw_kwargs=None, progress=None, self_consistency='G0W0',
                 screen_at='mean-field', spin='singlet', tda=False,
                 refuse_unconverged=False):
    """BSE by the density-fitted matrix-free Davidson: mean field in, excitations out.

    The twin of `solve_bse_isdf` on pyscf's own Coulomb-fitted three-index
    factor, so the two routes share every convention and differ
    only in how the interaction is represented:

        mf = dft.RKS(mol, xc=...).density_fit(auxbasis=...); mf.kernel()
        omega, X, Y, info = solve_bse_df(mf, mol, nocc)

    The quasiparticle diagonal comes from the Casida GW route -- the dense RPA
    spectrum in the same auxiliary basis, `calc_qp_energy(mode='casida')` for
    every orbital -- and `self_consistency='evGW'` drives that route through
    the eigenvalue loop, `gw_kwargs` going to `evgw_eigenvalues` (its own
    keywords and the Casida step's; `screening` is refused, the loop here is
    evGW). The static W is built from the same three-index
    factor: at the mean-field spectrum for `qp='G0W0'`, the standard G0W0-BSE
    split, and at the fixed point under evGW, since that is the spectrum the
    self-energy was built with. `qp` as an energy array and `screen_at` follow
    the ISDF twin's rules (see there).

    An attached environment reaches the kernel through the dressed factor
    (`SolventScreening.whitened_transform`, v -> v + vtilde) and the diagonal
    through Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018)
    Eq. (18) inside the GW route, whose self-energy screens with the bare
    interaction. Both halves of the reaction field, the ground-state one at
    eps_static and the response at eps_inf, are the caller's two calls on one
    environment object: `env.mean_field(mol, factory)` then
    `attach_environment(mf, env)`.

    Inside a `distributed` region every rank runs the whole route: the DF
    action is not divided, and it locksteps its inputs so that every rank's
    Davidson is rank 0's (`df_block_action`).

    numerics.bse.preconditioner: 'bare' or 'screened', as for the ISDF twin;
    the screened diagonal comes from the factor's own C_oo[:, i, i] and
    C_vv[:, a, a].

    probe: off by default. True or 'sign' measures min eig(A-B) matrix-free
    BEFORE the Davidson and refuses an unstable reference up front; on large
    DF runs it can cost more than the solve. Without it an unstable reference
    is still refused when the Davidson's projected (A-B) breaks down.
    min_eig_amb is None when the probe did not run.

    tda: the Tamm-Dancoff BSE, A X = omega X with B = 0 and Y returned zero
    (`solve_casida_davidson`); `probe` is skipped, as on the dense route,
    since that problem makes no omega^2 reduction.

    refuse_unconverged: as for the ISDF twin.

    Returns (omega, X, Y, info) with the ISDF twin's fields: eps / eps_mf /
    coeff_df / W_aux / min_eig_amb, length-gauge oscillator_strength,
    transition_dipole and exciton_descriptors per root, per-stage timings (the
    Davidson's breakdown among them), the evGW record and the tracked Ritz
    pairs.
    """
    require_closed_shell_or_unrestricted(mf, 'solve_bse_df', mol=mol)
    nroots, conv_tol, max_cycle, guess_factor, preconditioner = bse_settings(
        numerics, 'nroots', 'conv_tol', 'max_cycle', 'guess_factor',
        'preconditioner')
    t = {}
    stats = {}
    tracked = {}
    check_preconditioner(preconditioner)
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0
    if getattr(mf, 'with_df', None) is None:
        raise ValueError('solve_bse_df reads the mean field\'s own density '
                         'fitting: build it with mf.density_fit(auxbasis=...) '
                         'or use solve_bse_isdf, which fits for itself')
    if np.asarray(mf.mo_coeff).ndim == 3:
        raise NotImplementedError('solve_bse_df is restricted-spin only, like '
                                  'the Casida GW route that feeds its diagonal')

    _begin, _end = stage_clock(t, progress, 'bse-df')

    if progress:
        print(f'[bse-df {time.strftime("%H:%M:%S")}] natm={mol.natm} '
              f'nao={mol.nao_nr()} nocc={nocc} nroots={nroots} qp={qp}',
              flush=True)

    # the GW stage reads the driver's one Numerics, never a second
    refuse_second_numerics(gw_kwargs)
    diagonal = qp_diagonal(mf, mol, nocc, 'df', qp, self_consistency,
                           screen_at, dict(gw_kwargs or {}, numerics=numerics),
                           clock=(_begin, _end),
                           progress=progress, tag='bse-df')
    eps, eps_mf, mf = diagonal['eps'], diagonal['eps_mf'], diagonal['mf']
    evgw_info = diagonal['evgw']

    t0 = _begin('W')
    # One factor for W and the kernel, dressed by whatever is attached to mf;
    # the screening spectrum is the one mf carries (mean field, or shifted).
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    W_aux = LinearResponseSolver(get_orbital_energies(mf, representation='spatial'),
                                 coeff_df=coeff,
                                 spin_mode='restricted').static_screening_aux(nocc)
    _end('W', t0)

    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    amb = None
    # Opt-in: before the solve, refusing an unstable reference up front.
    if probe and not tda:
        t0 = _begin('probe')
        amb = float(lowest_amb_eigenvalue(
            lr, nocc, polarizability='BSE', W_aux=W_aux,
            sign_only=(isinstance(probe, str) and probe.lower() == 'sign'),
            stats=stats)[0])
        _end('probe', t0)
        if amb <= 0:
            raise unstable_reference(amb, 'before the solve')

    t0 = _begin('davidson')
    try:
        omega, X, Y = solve_casida_davidson(
            lr, nocc, numerics=numerics, polarizability='BSE', W_aux=W_aux,
            stats=stats, spin=spin, timings=t,
            max_memory=_job_max_memory(mf),
            tda=tda, tracked=tracked, refuse_unconverged=refuse_unconverged)
    except CasidaBreakdown as exc:
        # As on the ISDF twin: the breakdown's own measurement names an
        # unstable reference whether or not `probe` was asked for.
        if exc.min_eig_amb is not None and exc.min_eig_amb <= 0:
            raise unstable_reference(exc.min_eig_amb) from exc
        raise
    _end('davidson', t0)
    f_osc, trans_dip = oscillator_strengths(mf, mol, nocc, omega, X, Y)
    info = dict(eps=eps, eps_mf=eps_mf, coeff_df=coeff, W_aux=W_aux,
                min_eig_amb=amb, oscillator_strength=f_osc,
                transition_dipole=trans_dip,
                exciton_descriptors=exciton_descriptors(mf, mol, nocc, X, Y),
                timings=t, stats=stats, evgw=evgw_info, tracked=tracked,
                davidson_flags=davidson_flags(t),
                numerics_identity=numerics_or_default(numerics).identity())
    return omega, X, Y, info


def isdf_bse_factors(mf, mol, nocc, eps=None,
                     auxbasis=None, radii=None, counts=None, factors=None,
                     screening='imaginary-time', numerics=None, n_start=None,
                     grid_accuracy=None, distribute=None, comm=None):
    """(X_mo, D, W_aux) for the ISDF Davidson BSE, all from ONE auxiliary fit.

    The point is the gauge. `LinearResponseSolver.static_screening_aux` returns
    W_aux in whatever gauge its `coeff_df` came in, so W_aux is built here from
    the separable factors themselves and never from pyscf's cderi -- see the
    module docstring for what pairing the two costs.

    eps: orbital energies the RPA screening is built at, defaulting to the mean
    field's. A BSE@GW run passes the quasiparticle energies to
    `solve_casida_davidson` through `lr_solver.eps`, but W is normally still the
    G0W0 one, i.e. RPA at the mean-field energies -- so the two are separate
    arguments on purpose.
    counts: Lebedev sub-shell replica counts, i.e. the grid size -- the knob the
    factorization error responds to. The radii come from the one shipped table,
    looked up per element at these counts; nothing is substituted for a caller
    who named no size, so the published Duchemin-Blase cc-pVTZ grids are
    reached by asking for their counts rather than by default.
    grid_accuracy: the same grid asked for by name: an accuracy level of
    `ISDF_GRID_ACCURACY` or four explicit shell counts (`resolve_isdf_grid`),
    which refuses instead of substituting.
    Factors built here are refused where they miss an orbital density of
    `mf` (`separable_factors`).
    factors: pre-built (X_mo, D), to reuse a factorization across states, or
    `SlicedFactors` over `comm`: the sweep gathers what it reads whole once,
    and the return is then (the sliced factors, None, W_aux) -- D is whole
    for W alone and is not handed back.

    screening:
      'imaginary-time'  chi0(i.0) built from the separable factors in imaginary
            time by `LinearResponse/space_time.py`, which is tiled and cubic and
            touches nothing bigger than (naux, naux): the route for large
            systems. `static_screening_matrix` on
            `static_screening_grid`, so any other consumer of the same static
            screening builds it identically.
      'df'  chi0 from the three-index factor these factors imply. Exact for the
            given factors, and the reference the imaginary-time route was
            checked against -- 1e-8 relative on the W entries at ntau=18 on
            water and ethene / cc-pVDZ -- but it forms B and then a
            (naux, n_occ, n_vir) array. Small systems only.

    numerics: the choices (`Numerics`; None for the defaults): the static
    W's `quadrature.ntau_w` imaginary-time points, None sizing the grid from
    the Kaltak-Klimes-Kresse test integral over the chi0 transition range at
    `quadrature.tau_target`, exactly as the GW space-time route does, on the
    one object `solve_bse_isdf` hands its GW stage too -- but over that one
    range alone, since nothing here transforms a self-energy. More points are
    monotonically safe and then stop paying: measured against the 'df'
    reference on water / cc-pVDZ, W(i.omega = 0) is off by 4.6e-7 at ntau=12,
    1.0e-8 at 18, 2.5e-10 at 24 and 1.8e-10 at 34, the last two sitting on the
    tabulated coefficients' own precision.
    distribute, comm: split the tau points over the ranks of `comm`, by
    default the current `distributed` region's (None follows it, True also
    falls back to COMM_WORLD, False takes none -- as in `solve_bse_isdf`),
    replicate the mean field, the factors and `eps` from rank 0 before the
    grid is sized from them, and all-reduce one (naux, naux) matrix. This is
    the best compute-per-byte sweep in the code: every tau point is an M^2
    sweep and the whole result is a single naux^2 block. Exact up to
    summation order.
    """
    comm = current_comm() if comm is None else comm
    mpi_comm = grid_comm(comm)[0] if _takes_comm(distribute, comm) else None
    nranks = mpi_comm.Get_size() if mpi_comm is not None else 1
    # Before the grid: `ntau='auto'` is an integer read off the spectrum.
    if nranks > 1:
        lockstep_mean_field(mf, mpi_comm)
    factors = (factors if factors is not None
               else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                      counts=counts, n_start=n_start,
                                      grid_accuracy=grid_accuracy,
                                      numerics=numerics))
    if nranks > 1:
        replicate_factors(factors, mpi_comm)
    sliced = isinstance(factors, SlicedFactors)
    if sliced:
        factors.require(mpi_comm)
        if screening == 'df':
            raise ValueError("screening='df' forms the three-index factor "
                             'from X_mo and D whole; sliced factors take the '
                             "'imaginary-time' route")
        # The sweep gathers the branches of X_mo itself; D is whole for it
        # alone and is not handed back.
        X_mo, D = factors, factors.gather('D')
    else:
        X_mo, D = unpack_factors(factors)[:2]
    if eps is None:
        eps = get_orbital_energies(mf, representation='spatial')
    elif nranks > 1:
        eps = replicate(np.asarray(eps, float), comm=mpi_comm)

    if screening == 'df':
        lr = LinearResponseSolver(eps, coeff_df=isdf_df_coefficients(X_mo, D),
                                  spin_mode='restricted')
        return X_mo, D, np.asarray(lr.static_screening_aux(nocc))
    if screening != 'imaginary-time':
        raise ValueError(f"screening='{screening}'; choose 'imaginary-time' or 'df'.")

    q = numerics_or_default(numerics).quadrature
    grid = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                 tau_target=q.tau_target)
    W_aux = static_screening_matrix(X_mo, D, eps, nocc, grid, comm=mpi_comm)
    return X_mo, (None if sliced else D), W_aux


def _job_max_memory(mf):
    """MB this rank's job gave it -- the mean field's max_memory, which a job
    sets from its allocation's share (`allocation_max_memory_mb`) -- inside a
    SLURM allocation; None outside one, where max_memory is pyscf's process
    default and says nothing of the memory the machine has."""
    return (float(mf.max_memory) if allocation_max_memory_mb() is not None
            else None)
