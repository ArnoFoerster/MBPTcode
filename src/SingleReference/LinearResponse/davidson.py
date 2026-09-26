"""Matrix-free Casida/RPA/BSE Davidson, with two ways of applying the same A/B.

`solve_bse_isdf` and `solve_bse_df` are the two production drivers on them --
mean field in, excitations out -- and share every convention, so they differ
only in how the interaction is represented.

The DF route contracts pyscf's three-index factor directly. What sets its cost
is `apply_exchange_direct`, which needs a (naux, nvirt, nvirt) intermediate --
the array that caps system size.

The ISDF (separable-RI) route replaces those contractions by Fock-like builds on
the interpolation grid. With

    (pq|W|rs) = sum_kk' X[k,p] X[k,q] Zt[k,k'] X[k',r] X[k',s],  Zt = D W_aux D^T

each block action is a few GEMMs and one Hadamard product,

    P[k,k']  = sum_jb X_o[k,j] z[j,b] X_v[k',b]     the trial vector as a
                                                    transition density on the grid
    [K z]_ia = sum_kk' X_o[k,i] (Zt * P)[k,k'] X_v[k',a]

so O(M^2 (n_occ + n_vir) + M n_occ n_vir) per trial vector, with no
(naux, nvirt, nvirt) array anywhere. This is the BSE counterpart of the GW
self-energy build in `GW/imaginary_time.py::self_energy_matrix_imaginary_time`,
taken statically at omega = 0. Foerster and Visscher, JCTC 2022,
doi 10.1021/acs.jctc.2c00531 do the same
with PADF in place of ISDF.

The auxiliary GAUGE is the trap. pyscf's cderi is L^-1-whitened while a
separable RI fits with the symmetric V^-1/2, and the two differ by an orthogonal
rotation of the auxiliary index. A W_aux from one paired with factors from the
other is silently wrong -- every intermediate still looks self-consistent, and
in the GW work the same mistake moved a quasiparticle energy by 1.4 eV while
Sigma(i.omega) still agreed to 1e-5. `isdf_bse_factors` returns X, D and W_aux
from a single fit, which is the only reliable way to keep them in one gauge.

The Davidson GUESS is the trap the two routes share, and it is silent in the
same way: one unit vector per requested root returns `nroots` roots that all
report converged and are not the lowest ones whenever the smallest
orbital-energy differences are degenerate. See GUESS_FACTOR.

Where this reaches. At a chlorophyllide-a hexamer / cc-pVTZ -- 474 atoms,
nao 10980, naux 28236, and M 117762 on the published grids -- the DF route
would need a 21 TB (naux, nvirt, nvirt) array and 5.5 Pflop per trial vector,
and its static W a 2 TB (naux, n_occ, n_vir) one. The separable route holds Zt
at 103 GB, costs 29 Tflop per trial vector, and takes its W from imaginary time
instead. Zt spread across ranks is the row split of `isdf_block_action`; what
is still missing at that size is the ISDF fit's own M x M Gram inversion, which
lives in `separable_ri.fit_M`, not here.

Over ranks the Davidson and the (A-B) probe run on every rank alike. MPI lives
inside the block action alone: it locksteps its trial vectors at entry and
gathers or all-reduces every term it returns, so its output, and with it every
decision the replicated iteration takes, is the same on every rank.
"""
import sys
import time
import warnings

import numpy as np
from pyscf.lib import logger, param
from pyscf.tdscf._lr_eig import MAX_SPACE_INC, real_eig
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, eigsh

from src.Base.constants import (AMB_LANCZOS_MAXITER_PER_DIM, AMB_LANCZOS_NCV,
                                BSE_DENSE_MAX_NOV,
                                DAVIDSON_FLOOR_EPS_MULTIPLE, DAVIDSON_LINDEP,
                                DAVIDSON_MIN_NEW_FRACTION,
                                DAVIDSON_SIZED_RESIDUAL, DAVIDSON_SPACE_CYCLES,
                                DAVIDSON_SPACE_GB, HARTREE_TO_EV,
                                ISDF_TILE_GB, KAPPA)
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (allgather_blocks, contiguous_block,
                                     current_comm, grid_comm, lockstep,
                                     lockstep_stats, partition, reduce_sum,
                                     replicate)
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.evGW import evgw_eigenvalues, shifted_mean_field
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import (DEFAULT_NTAU, _unpack_factors,
                                               replicate_factors,
                                               replicate_mean_field,
                                               separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.LinearResponse.exciton_descriptors import exciton_descriptors
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, check_normalization)
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency


#: Unit-vector guesses per requested root. One each is what a diagonal-dominant
#: argument suggests, and it silently returns the WRONG STATES whenever the
#: smallest orbital-energy differences are degenerate. A unit vector in the
#: occupied-virtual pair basis carries a definite pair symmetry and both A and B
#: are block diagonal over it, so the Davidson subspace never leaves the irreps
#: its guess started in: it converges, to `nroots` roots that are not the lowest
#: ones. benzene/cc-pVDZ RPA at nroots=5 puts a degenerate quartet plus half of
#: the next degenerate pair in the guess and returns
#:     0.483476 0.519125 0.523487 0.570116 0.570117
#: against a true lowest five of
#:     0.483476 0.519125 0.519134 0.523487 0.553819
#: -- with converged True on all five. So `converged` does not catch this; only
#: a better-spanning guess does, and the two fixes are independent.
#:
#: Over-requesting is a heuristic, not a guarantee -- it buys directions in more
#: irreps, and nothing here knows which irreps the true lowest roots live in.
#: Completing the degenerate set alone is NOT enough (measured: six guesses on
#: that benzene case still miss the fifth root); 2x plus completion is. A result
#: worth trusting is worth repeating at a raised `guess_factor`.
GUESS_FACTOR = 2
#: Ha. Two orbital-energy differences closer than this count as one degenerate
#: set, which the guess then takes whole. pyscf's tdscf uses the same value and
#: the same window (`deg_eia_thresh`); the splittings it has to absorb are the
#: DF and SCF noise on a true degeneracy, ~1e-5 Ha on the benzene case above.
GUESS_DEG_TOL = 1e-3

#: `GW/space_time.py`'s own `timings` keys, renamed for `info['timings']`'s
#: `qp_` prefix: `t_qp_static`/`t_qp_states` fold their `t_qp_` in with the
#: shared prefix rather than doubling it. `t_qp` -- `_finish_qp`'s own total,
#: static + states -- carries nothing beyond the two keys it sums and is not
#: re-exposed under a third name.
QP_TIMING_RENAME = {'t_chi0': 'qp_chi0', 't_dyson': 'qp_dyson',
                    't_sigma': 'qp_sigma', 't_qp_static': 'qp_static',
                    't_qp_states': 'qp_states'}

#: The collectives of a distributed block action, each timed apart as
#: `davidson_comm_<kind>`: the checked lockstep of the trial vectors (a digest,
#: and a broadcast only where one differs) and of the result; the all-gather of
#: z X_v^T, M x n_occ per vector; the all-reduce of X_o^T (Zt * P), n_occ x M
#: per vector; and those of p D, naux per vector, and of the batch's
#: (n_occ, n_vir) output slabs.
COMM_KINDS = ('lockstep', 'gather', 'reduce_grid', 'reduce_pairs')


class _QPStageTimings:
    """`timings=` target for the space-time QP solve, G0W0 or one evGW cycle.

    `solve_qp_energy_space_time`/`_qp_blocked`/`_finish_qp` write each
    per-stage key once per call with a plain assignment. evGW re-enters that
    call once per cycle on the SAME object -- `gw_kwargs['timings']` is fixed
    for the whole loop in `evgw_eigenvalues`, which is not edited to make this
    work -- so a plain dict would keep only the last cycle's numbers.
    `__setitem__` sums the per-stage keys instead, so after the loop the
    object holds the total spent in each stage over however many calls wrote
    to it: one for `qp='G0W0'`, one per evGW cycle. Everything else
    (`nranks`, `ntau_auto`, `tau_fit_error`) takes the latest write, since
    those describe the run rather than accumulate over it. A caller-supplied
    dict (`gw_kwargs['timings']`) is written into directly, never copied, so
    it still carries the final numbers under its own name.
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


class _CommClock:
    """Wall seconds one block action spends inside its collectives, read by
    the Davidson as `davidson_comm`, and the same seconds by collective in
    `by_kind` (COMM_KINDS); nothing is timed without a rank count."""

    __slots__ = ('seconds', 'by_kind')

    def __init__(self):
        self.seconds = 0.0
        self.by_kind = dict.fromkeys(COMM_KINDS, 0.0)


def solve_casida_davidson(lr_solver, nocc, nroots=3, polarizability='RPA',
                           W_aux=None, conv_tol=1e-5, max_cycle=100, orbsym=None,
                           isdf_factors=None, guess_factor=GUESS_FACTOR,
                           stats=None, spin='singlet', comm=None, timings=None,
                           refuse_unconverged=False):
    """Matrix-free Davidson solver for the `nroots` lowest Casida excitation energies (never forms dense A/B).

    For a handful of low-lying states; vertex-correction sums still need the
    full spectrum (use build_casida_matrices + CasidaSolver for that).
    Iteration via pyscf's real_eig; the A/B matrix-free action is ours,
    validated against build_casida_matrices to machine precision on both routes.

    polarizability: 'RPA' (Hartree-only), 'TDHF' (bare exchange), or 'BSE'
    (screened exchange, needs W_aux from solve_rpa_screening).
    orbsym: optional pyscf orbital irrep IDs; enables symmetry-block Davidson
    (validated correct, but measured no speedup on benzene).
    isdf_factors: (X_mo, D) from `isdf_bse_factors` -- switches the block action
    to the separable-RI Fock-like builds described in the module docstring.
    W_aux MUST come from the same call, or the auxiliary gauges disagree.
    guess_factor: guess vectors per requested root. The default spans enough
    symmetry blocks on everything tested here; raise it and check the roots stop
    moving when a spectrum is dense or highly degenerate. See GUESS_FACTOR.
    spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0, the bare-exchange
    term absent). The screened term is spin-independent.
    comm: an MPI communicator whose ranks all run this solver together, by
    default the current `distributed` region's; the ISDF block action divides
    the rows of Zt between them (`isdf_block_action`) and every rank runs the
    same iteration on its lockstepped output (`_run_davidson`); the DF action
    is not divided and runs whole on every rank on lockstepped inputs.
    timings: optional dict, filled with the `davidson_*` breakdown of
    `_run_davidson` -- the same keys serially and under a comm, so the two are
    comparable -- plus `davidson_setup`, the action's own build: the collocation
    slices and this rank's rows of the screened kernel, paid once before the
    first trial vector and therefore invisible in `davidson_block_action`.
    conv_tol: the residual |r| every root must reach, in Hartree. One below the
    round-off floor of the residual is refused before the action is built
    (`check_residual_floor`).
    refuse_unconverged: raise instead of warning when a root ends above
    conv_tol -- for a caller that differentiates the eigenvectors, which an
    unconverged root makes wrong by its residual.
    Returns (omega, X, Y) normalized <X|X>-<Y|Y>=1.
    """
    comm = current_comm() if comm is None else comm
    check_residual_floor(conv_tol, bse_pair_diagonal(lr_solver.eps, nocc))
    t0 = time.time()
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, spin=spin, comm=comm)
    setup_s = time.time() - t0
    if timings is not None:
        timings['davidson_setup'] = setup_s
        # Every rank arrives here off the same straight-line code, so the
        # gather is the one `_run_davidson` already makes, nranks doubles, and
        # it says which rank's row block cost what to build.
        if comm is not None and comm.Get_size() > 1:
            timings['davidson_setup_by_rank'] = _block_action_by_rank(setup_s,
                                                                      comm)
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    return _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                         _pair_symmetry(orbsym, occ, virt), guess_factor,
                         stats=stats, comm=comm, timings=timings,
                         refuse_unconverged=refuse_unconverged)


def check_residual_floor(conv_tol, diag_d):
    """The residual floor of the Davidson on the pair space `diag_d`, in
    Hartree, or a ValueError if `conv_tol` asks for less.

    The residual is A x - omega x from block actions good to eps * ||A||, and
    ||A|| is the largest pair energy (DAVIDSON_FLOOR_EPS_MULTIPLE). Asked for
    less, the solver spends its cycles on round-off and stops unconverged, or
    its projected (A-B) block loses definiteness; this refuses before either.
    """
    floor = _residual_floor(diag_d)
    if not conv_tol >= floor:
        raise ValueError(
            f'Davidson conv_tol={conv_tol:g} is below the residual this pair '
            f'space can reach in double precision, {floor:.1e} Ha '
            f'({DAVIDSON_FLOOR_EPS_MULTIPLE:g} eps times the largest pair '
            f'energy, {np.abs(diag_d).max():.2f} Ha); ask for at least that.')
    return floor


def _residual_floor(diag_d):
    """Hartree: the Casida residual that is round-off on the pair space `diag_d`."""
    return (DAVIDSON_FLOOR_EPS_MULTIPLE * np.finfo(float).eps
            * float(np.abs(diag_d).max()))


def _block_action(lr_solver, nocc, polarizability, W_aux, isdf_factors,
                  spin='singlet', comm=None):
    """Mode dispatch shared by the solver and the (A-B) instability probe."""
    comm = current_comm() if comm is None else comm
    if spin not in KAPPA:
        raise ValueError(f"spin {spin!r}: one of {', '.join(sorted(KAPPA))}")
    mode = polarizability.upper()
    if mode == 'RPA':
        lBSE, w = False, None
    elif mode == 'TDHF':
        lBSE, w = True, None
    elif mode == 'BSE':
        if W_aux is None:
            raise ValueError("polarizability='BSE' requires W_aux (see LinearResponseSolver.solve_rpa_screening).")
        lBSE, w = True, W_aux
    else:
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose 'RPA', 'TDHF', or 'BSE'.")

    if isdf_factors is None:
        return df_block_action(lr_solver, nocc, lBSE, w, spin=spin, comm=comm)
    return isdf_block_action(lr_solver, nocc, lBSE, w, isdf_factors, spin=spin,
                             comm=comm)


def lowest_amb_eigenvalue(lr_solver, nocc, polarizability='BSE', W_aux=None,
                          isdf_factors=None, k=1, tol=1e-6, sign_only=False,
                          stats=None, comm=None):
    """Lowest eigenvalue(s) of (A - B): the sign that decides whether the
    Casida omega^2 reduction is valid at all.

    Matrix-free -- a symmetric Lanczos on z -> (A-B)z through the same block
    action the solver uses -- so it runs at any size the action runs at. That
    makes it the instability CONTROL at production sizes, where nothing dense
    fits (at a11/cc-pVTZ the DF vv slice alone is 91.5 GB): min eig < 0 there
    confirms the regime `solve_casida_davidson` refuses; and comparing the DF
    and ISDF actions' values at a size where both fit tests whether the
    factorization, not the physics, pushed a near-zero mode negative.

    pyscf's mf.stability() is NOT a proxy for this: 90-degree twisted ethene
    reports internally stable there while (A-B) here is indefinite
    (-0.005 Ha) -- the SCF Hessian under real singlet rotations is a different
    condition from response-metric positive definiteness.

    The probe is a property of the REFERENCE, not of the molecule. Twisted
    ethene has (at least) two converged RHF solutions 31 mHa apart, and
    pyscf's default minao guess can land on the HIGHER one, where min eig(A-B)
    is far more negative (TDHF -0.067 vs -0.005 on the lowest solution; a
    "7x screening deepening" and a "sign flip" were both artifacts of
    comparing across solutions). On both solutions the screening moved the
    minimum UPWARD, consistently. WHICH guess finds the lowest solution is
    itself system-dependent: on the acenes (multi-solution from a6, spreads
    up to 107 mHa) minao lands on the LOWER of three and '1e' is the outlier
    -- the reverse of ethene. So compare E_SCF across initial guesses rather
    than trusting any one; probe the reference you will actually use, with
    the kernel you will actually use; and if the probe comes back negative,
    establish that the SCF is the lowest solution before concluding the
    system itself is unstable.

    Spin does not enter: kappa (ia|jb) is the same term in A and B, so it cancels
    out of the difference and a triplet has the same (A - B) as its singlet.

    comm: the communicator the ISDF action divides its rows over, by default
    the current `distributed` region's; every rank then runs the same Lanczos
    and returns rank 0's value (`_lowest_amb_from_action`).
    """
    comm = current_comm() if comm is None else comm
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, comm=comm)
    return _lowest_amb_from_action(apply_AB, diag_d, k=k, tol=tol,
                                   sign_only=sign_only, stats=stats, comm=comm)


def _davidson_counters():
    """Where one rank's Davidson time and traffic go.

    `action_s` is the block action, its collectives included, and `comm_s` the
    part of it spent in them -- the entry lockstep of the trial vectors, the
    head's gather and the reductions -- plus the lockstep of the converged
    result. `loop_s` is the wall of the iteration, so `loop_s - action_s` is
    the Rayleigh-Ritz, orthogonalisation, preconditioning and convergence
    checks, which every rank runs whole. `lockstep_bytes` is what this rank's
    locksteps moved, the volume ONE rank sends or receives, and
    `lockstep_skipped_bytes` what the checked ones did not move because every
    rank's digest agreed; every rank reports the same two numbers.
    `space_bound` is the trial pairs real_eig holds before it collapses its
    subspace (`_trial_space_memory`) and `collapses` how often it did.
    `comm_by_kind` splits `comm_s` by collective (COMM_KINDS).
    """
    return {'action_s': 0.0, 'comm_s': 0.0, 'loop_s': 0.0,
            'lockstep_bytes': 0, 'lockstep_skipped_bytes': 0, 'vectors': 0,
            'iterations': 0, 'space_max': 0, 'space_bound': 0, 'collapses': 0,
            'comm_by_kind': dict.fromkeys(COMM_KINDS, 0.0)}


def _block_action_by_rank(value, comm):
    """Every rank's `value` in rank order, through one all-reduce of nranks
    doubles -- a rank-indexed sum of zeros is the gather `mpi_grid` has not
    got, and at this size it costs nothing next to one trial block."""
    buf = np.zeros(comm.Get_size())
    buf[comm.Get_rank()] = value
    reduce_sum(buf, comm)
    return [float(x) for x in buf]


def _lowest_amb_from_action(apply_AB, diag_d, k=1, tol=1e-6, sign_only=False,
                            stats=None, comm=None):
    """The (A-B) probe, the same iteration on every rank, rank 0's value on each.

    Serially the Lanczos is ARPACK's. Under any rank count it is
    `_lanczos_lowest` on every rank, over the distributed action's lockstepped
    output: eigsh holds one process-wide lock across its whole iteration,
    matvecs included, so over ranks that are threads of one process
    (`run_simulated`) the first rank inside waits in its matvec's collective
    for a rank that waits for the lock -- and simulated ranks have to run the
    algorithm MPI ranks run. The value is lockstepped at the end;
    ranks whose iterations had fallen apart meet that lockstep against another
    rank's block-action lockstep and raise ValueError together instead of
    deadlocking.
    """
    replicated = comm is not None and comm.Get_size() > 1
    out = _lowest_amb_local(apply_AB, diag_d, k, tol, sign_only, stats,
                            replicated)
    return lockstep(out, comm) if replicated else out


def _lowest_amb_local(apply_AB, diag_d, k=1, tol=1e-6, sign_only=False,
                      stats=None, replicated=False):
    """The Lanczos itself: lowest eigenvalue(s) of (A - B), by ARPACK, or by
    `_lanczos_lowest` where the action is distributed (`replicated`)."""
    no, nv = diag_d.shape
    nmv = [0]
    t_action = [0.0]

    def matvec(z):
        t0 = time.time()
        Az, Bz = apply_AB(np.asarray(z, dtype=float).reshape(1, no, nv))
        t_action[0] += time.time() - t0
        nmv[0] += 1
        return (Az - Bz).ravel()

    op = LinearOperator((no * nv, no * nv), matvec=matvec, dtype=float)
    # A FIXED START VECTOR, not ARPACK's random one. ARPACK draws its start
    # from process-global state, which makes the iteration -- and so the
    # answer -- depend on what else in the process has drawn from it; the
    # start must be a function of the data alone for the probe to be
    # reproducible at all. It must ALSO carry weight in every pair irrep:
    # (A - B) is block diagonal over Gamma_i x Gamma_a and a unit vector would
    # confine the Krylov space to one block, returning that block's minimum as
    # if it were the global one. 1/d has both properties and leans on the low
    # pairs.
    v0 = 1.0 / diag_d.ravel()
    v0 /= np.linalg.norm(v0)

    def lowest(nroots, tol_used, vectors):
        """The nroots lowest Ritz values, with their vectors as columns if
        asked for."""
        if not replicated:
            return eigsh(op, k=nroots, which='SA', tol=tol_used, v0=v0,
                         return_eigenvectors=vectors)
        w, x = _lanczos_lowest(matvec, v0, nroots, tol_used)
        return (w, x.T) if vectors else w

    def _record(extra=None):
        if stats is not None:
            stats.update({'probe_matvecs': nmv[0],
                          'probe_action_s': t_action[0], **(extra or {})})

    if sign_only and k == 1 and no * nv > 1:
        # THE SIGN IS THE ANSWER, so stop once it is PROVEN -- which a Ritz
        # value alone does not do. Ritz values from a Krylov subspace bracket
        # the spectrum from the inside, so theta >= lambda_min and theta > 0
        # proves nothing on its own. The residual closes it: for a symmetric
        # operator |lambda_i - theta| <= ||r||, so theta - ||r|| > 0 is a
        # certificate, and theta + ||r|| < 0 is the certificate for the other
        # sign. Escalate the tolerance only until one of them holds, which for
        # a value far from zero is the first pass and a fraction of the
        # iterations a converged 'SA' solve would take.
        for tol_try in (1e-2, 1e-3, 1e-4, tol):
            w, v = lowest(1, tol_try, True)
            theta = float(w[0])
            vec = v[:, 0]
            resid = float(np.linalg.norm(matvec(vec) - theta * vec))
            if theta - resid > 0.0 or theta + resid < 0.0:
                _record({'probe_sign_proven': True, 'probe_tol_used': tol_try,
                         'probe_residual': resid})
                return np.array([theta])
        # No certificate at the tightest tolerance: the value sits inside its
        # own error bar, i.e. genuinely near zero. Say so rather than returning
        # a sign the arithmetic does not support.
        _record({'probe_sign_proven': False, 'probe_tol_used': tol,
                 'probe_residual': resid})
        return np.array([theta])

    # A Ritz value is good to |r|^2 / gap, so the replicated Lanczos converges
    # to |r| <= sqrt(eps) |theta| rather than stopping at tol: at water/cc-pVDZ
    # BSE@HF that is 88 matvecs where ARPACK takes 81, for a value 3e-15 Ha
    # from the dense eigenvalue where ARPACK's sits 7e-14 from it.
    if replicated:
        tol = min(tol, float(np.sqrt(np.finfo(float).eps)))
    out = np.sort(lowest(min(k, no * nv - 1), tol, False))
    _record()
    return out


def _lanczos_lowest(matvec, v0, k, tol):
    """(theta, x): the k lowest eigenpairs of the symmetric operator
    `matvec`, the vectors as rows, from the start vector `v0`.

    A Lanczos with full reorthogonalization, thick-restarted on the lower
    half of its Ritz vectors whenever the basis reaches eigsh's own size
    (Wu and Simon, SIAM J. Matrix Anal. Appl. 22, 602 (2000)), converged at
    ARPACK's test |r| <= tol |theta|. The Rayleigh quotient is formed from the
    stored products, never from a recurrence, so the iteration is a function
    of the operator's outputs alone and ranks handed the same outputs take the
    same steps; every array is this call's own, so rank threads can run it
    side by side. One start vector spans one direction of a degenerate
    eigenspace, so for k > 1 a repeated eigenvalue can appear once, as in
    ARPACK; the lowest value itself is found either way.
    """
    n = v0.size
    ncv = min(n, max(2 * k + 1, AMB_LANCZOS_NCV))
    V = np.zeros((ncv, n))
    AV = np.zeros((ncv, n))
    q = v0 / np.linalg.norm(v0)
    m = 0
    for _ in range(AMB_LANCZOS_MAXITER_PER_DIM * n):
        V[m] = q
        AV[m] = matvec(V[m])                 # a distributed action locksteps V[m]
        m += 1
        H = V[:m] @ AV[:m].T
        theta, s = np.linalg.eigh(0.5 * (H + H.T))
        nk = min(k, m)
        x = s[:, :nk].T @ V[:m]
        r = s[:, :nk].T @ AV[:m] - theta[:nk, None] * x
        rnorm = np.linalg.norm(r, axis=1)
        if m == n or (m >= k and np.all(rnorm <= tol * np.abs(theta[:k]))):
            return theta[:nk], x
        # In a Lanczos basis every Ritz residual is the same next direction;
        # the longest is its most accurate copy.
        q = r[np.argmax(rnorm)]
        if m == ncv:
            m = max(k, ncv // 2)
            V[:m] = s[:, :m].T @ V
            AV[:m] = s[:, :m].T @ AV
        q -= (V[:m] @ q) @ V[:m]
        q -= (V[:m] @ q) @ V[:m]             # twice is enough (Kahan, Parlett)
        norm = np.linalg.norm(q)
        if norm <= np.finfo(float).eps * np.abs(theta).max():
            return theta[:nk], x             # an invariant subspace
        q /= norm
    raise RuntimeError(
        f'the (A - B) Lanczos left roots above |r| <= {tol:g} |theta| after '
        f'{AMB_LANCZOS_MAXITER_PER_DIM * n} operator applications')




def isdf_df_coefficients(X_mo, D):
    """The (naux, norb, norb) DF factor the separable factorization implies,
    B[A,p,q] = sum_k X[k,p] X[k,q] D[k,A].

    O(M naux norb^2) to build and naux norb^2 to hold, so it defeats the point
    of the factorization for production. It exists as the bridge to every DF
    consumer:
    `build_casida_matrices`, `static_screening_aux` and the dense CasidaSolver
    all speak B, and feeding them THIS B is what puts them in the ISDF
    factorization's own gauge -- the only setting in which they and the
    matrix-free ISDF path are comparing the same operator.
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

    The normalization is CHECKED, not assumed: the sqrt(2) above belongs to
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


def solve_bse_isdf(mf, mol, nocc, nroots=5, qp='G0W0', factors=None,
                   auxbasis=None, radii=None, counts=None, probe=True,
                   conv_tol=1e-5, max_cycle=100, guess_factor=GUESS_FACTOR,
                   gw_kwargs=None, progress=None, n_start=1,
                   self_consistency='G0W0', screen_at='mean-field',
                   spin='singlet', grid_accuracy=None, distribute=None,
                   comm=None, sigma_x_matrix=None):
    """BSE by the ISDF matrix-free Davidson: mean field in, excitations out.

    The production calling sequence is three lines --

        mf = dft.RKS(mol, xc=...).density_fit(auxbasis=...); mf.kernel()
        omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=5)

    -- and the discipline the pieces demand lives HERE so the caller cannot
    violate it: one ISDF fit is built (or taken from `factors`) and shared by
    the GW and BSE stages; the static W comes from that fit at the eigenvalues
    the LEVEL OF THEORY screens with -- the mean field's for G0W0, which is the
    standard split, and the converged ones under evGW -- and W and the block
    action can never mix auxiliary gauges.

    self_consistency: 'G0W0' (default) evaluates the self-energy once on the
        mean field's eigenvalues. 'evGW' runs the eigenvalue-self-consistent
        loop first -- all eigenvalues updated, DIIS, convergence on the HOMO
        and LUMO -- puts its fixed point on the BSE diagonal AND builds the
        static W from it, so the kernel screens with the same spectrum the
        self-energy did. It overrides `qp`, and it shares the ISDF fit, so the
        factorization is built once for both stages.
    screen_at: which spectrum builds the static W when `qp` is an explicit
        ENERGY ARRAY. THE CONVENTION IS THE LEVEL OF THEORY, not the diagonal:
        G0W0 screens W0 at the MEAN-FIELD eigenvalues and evGW at the converged
        ones. 'mean-field' (default) is therefore the standard G0W0-BSE split,
        and is what an array of G0W0 energies computed elsewhere wants. 'qp'
        screens at the array instead, for an array that IS a self-consistent
        spectrum -- evGW energies obtained from a separate loop, say, where
        `self_consistency='evGW'` was not used. The two differ: on water
        /cc-pVDZ the full-integral and DF routes agree to +0.6 meV when the
        screening and the diagonal match and drift -65 meV when they do not,
        which is the size of the choice.

        QUASIPARTICLE ENERGIES PLUS A FURTHER SHIFT are neither, and are the
        easier mistake: evGW energies carrying a static reaction-field
        correction on top are not the evGW spectrum, so screening at them
        satisfies no convention -- 119 meV on acrolein's n->pi*. Leave the
        default and let the reference object carry the spectrum.

        Neither value touches `qp='G0W0'`, whose W comes from the GW step's own
        chi0 at mean-field energies, nor `self_consistency='evGW'`, which always
        screens at its fixed point.

    qp: 'G0W0' (default) puts every quasiparticle energy from ONE space-time
    self-energy on the BSE diagonal (`solve_qp_diagonal_space_time`, which
    applies <Sigma_x - v_xc> per state for a KS reference); an ARRAY of
    energies is used as the diagonal directly (an evGW result, say);
    False/None solves BSE@mean-field.
    spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0); the screened term is
    spin-independent, so both read the same W.
    probe: measure min eig(A-B) first and refuse the solve while it is
    negative -- the instability regime, diagnosed before the node-hours are
    spent rather than inside the eigensolver. True converges the eigenvalue;
    'sign' stops as soon as the sign is PROVEN by the Ritz residual bound,
    which is all the refusal above actually reads. At the chlorophyllide
    dimer/cc-pVDZ the converged probe was 24.3% of the whole run -- second only
    to the solve it guards -- to establish that a number was +0.060 Ha.
    False skips it, which is reasonable only on a reference already probed:
    the quantity is a property of the SCF SOLUTION, not the molecule.
    gw_kwargs: forwarded to `solve_qp_diagonal_space_time` (the shared factors
    are always passed).
    sigma_x_matrix: <p|Sigma_x - v_xc|q> already built for this mean field, the
    way `factors` is handed in. It is a functional of the density and the
    orbitals alone, so one build serves this BSE, a quasiparticle window the
    caller ran before it, and every evGW cycle -- and none of the three is what
    makes it expensive: one K plus the xc potential on the DFT grid, which no
    window size divides. Left None the GW stage builds it once here, before
    the spectrum starts moving, and hands it to every cycle; the build is
    timed as its own `sigma_x` stage.
    distribute, comm: split the M^2 work over the ranks of `comm`, by default
    the current `distributed` region's. None (the default) follows that
    communicator; True also falls back to COMM_WORLD; False gives this call no
    communicator of its own (no replication, no COMM_WORLD), while the kernels
    it calls still follow a `distributed` region around it -- a serial solve
    inside one is `with distributed(None):`. What is divided is the tau axis of
    chi0 and Sigma in the GW stage and of the static W, and the ROWS of Zt in
    the block action, so the one M^2 object's per-rank memory falls with the
    rank count. Exact up to summation order; one rank prints.

    THE ANSWER IS RANK 0's. Everything the ranks contract into a reduced sum
    is replicated from rank 0 first -- the mean field, the ISDF factors, the
    quasiparticle diagonal and the static W -- and their own copies are
    overwritten, because partials of two slightly different calculations do
    not add. The Davidson and the (A-B) probe then run on every rank on the
    block action's lockstepped output, and their results are lockstepped from
    rank 0 at the end. On a slow interconnect weigh the GW stage's all-reduce
    of nfreq x naux^2 first: on 1 GbE it cost more than it saved
    (`Base.utils.mpi_grid`).
    grid_accuracy: the named way to ask for the interpolation grid -- an
    accuracy level of `ISDF_GRID_ACCURACY`, measured on exactly this quantity
    (the three lowest BSE roots against `solve_bse_df`), or four explicit shell
    counts. It reaches `separable_factors`, sets `counts` and `n_start`
    together, and aborts on anything the radii table has not got.

    progress: stamp each stage to stdout as it begins and ends, so a job that
    is going to take node-hours says which stage it is in WHILE it is in it.
    The returned timings only arrive if the run finishes, which is exactly the
    case that does not need them. None follows `mol.verbose`, so the one knob
    that already turns on the mean field's output turns on this too rather
    than leaving a second one to forget.

    Returns (omega, X, Y, info); info carries eps / eps_mf / factors / W_aux /
    min_eig_amb, length-gauge oscillator_strength and transition_dipole per
    root (`oscillator_strengths`), exciton_descriptors, per-stage timings in
    seconds and `nranks`. The Davidson stage carries its own breakdown beside
    its total -- the block action, the subspace work, the collectives and the
    lockstep volume, and the iteration counts, the same keys serially and
    under a comm (`_run_davidson`). The `qp` stage carries its own breakdown
    too, whenever `qp='G0W0'` or `self_consistency='evGW'` runs the
    space-time solve: `qp_chi0`, `qp_dyson`, `qp_sigma`, the exchange build
    `qp_static` and the per-state root search `qp_states`
    (`_QPStageTimings`, summed over evGW's cycles, whose count is
    `qp_cycles`).
    """
    t = {}
    stats = {}
    comm = current_comm() if comm is None else comm
    mpi_comm, rank, nranks = (grid_comm(comm) if _takes_comm(distribute, comm)
                              else (None, 0, 1))
    # Rank 0's mean field, before a grid size or an interpolation point count
    # is read off it anywhere below.
    if nranks > 1:
        replicate_mean_field(mf, mpi_comm)
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0
    progress = bool(progress) and rank == 0        # lockstep ranks, one voice

    def _begin(name):
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] {name} ...', flush=True)
        return time.time()

    def _end(name, t0):
        t[name] = time.time() - t0
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] {name} done, '
                  f'{t[name]:.1f} s', flush=True)

    if progress:
        print(f'[bse {time.strftime("%H:%M:%S")}] natm={mol.natm} '
              f'nao={mol.nao_nr()} nocc={nocc} nroots={nroots} qp={qp}',
              flush=True)

    if factors is None:
        t0 = _begin('factors')
        factors = separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                    counts=counts, n_start=n_start,
                                    grid_accuracy=grid_accuracy)
        _end('factors', t0)
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] ISDF M={factors[1].shape[0]} '
                  f'({factors[1].shape[0] // mol.natm}/atom)', flush=True)
    # Rank 0's fit, whether it was built here or handed in: every rank owns a
    # block of the ROWS of Zt, and rows built from two different point sets are
    # not partials of one matrix.
    if nranks > 1:
        replicate_factors(factors, mpi_comm)

    # <Sigma_x - v_xc> is a functional of the density and the orbitals, so it
    # is built ONCE, from the mean field before any spectrum shifts it, and
    # handed to every quasiparticle call below -- the G0W0 diagonal, or each
    # evGW cycle, which would otherwise rebuild the same K and the same
    # DFT-grid potential per cycle. 'df-direct' forms no such matrix.
    gw_kw = dict(gw_kwargs or {})
    if nranks > 1:
        gw_kw.update(distribute=True, comm=mpi_comm)
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

    evgw_info = None
    if str(self_consistency).lower() in ('evgw', 'ev'):
        t0 = _begin('evgw')
        # The fit is geometry-only, so the loop reuses the one above rather
        # than rebuilding it every cycle.
        qp, evgw_info = evgw_eigenvalues(mf, mol, mode='space-time',
                                         factors=factors, **gw_kw)
        # W must screen with the spectrum the self-energy converged on, not the
        # mean field's; shifting the reference is what carries that downstream.
        mf = shifted_mean_field(mf, qp)
        _end('evgw', t0)
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] evGW '
                  f'{"converged" if evgw_info["converged"] else "NOT converged"}'
                  f' in {evgw_info["cycles"]} cycles', flush=True)
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
        t0 = _begin('qp')
        gw_extras = {}
        eps, _ = solve_qp_diagonal_space_time(mf, mol, nocc, factors=factors,
                                              extras=gw_extras, **gw_kw)
        _end('qp', t0)
    else:
        eps = np.asarray(qp, dtype=float)
        if eps.shape != eps_mf.shape:
            raise ValueError(f'qp energies have shape {eps.shape}, the mean '
                             f'field has {eps_mf.shape}.')
        if screen_at not in ('qp', 'mean-field'):
            raise ValueError(f"screen_at={screen_at!r}; choose 'qp' or "
                             f"'mean-field'.")
        if screen_at == 'qp':
            # An array that IS a self-consistent spectrum: W follows it, the
            # way evGW's does. The standard G0W0 split is the default above.
            mf = shifted_mean_field(mf, eps)

    if qp_stage is not None:
        for key, value in qp_stage.target.items():
            if key == 't_qp':
                continue
            t[QP_TIMING_RENAME.get(key, 'qp_' + key)] = value
        if evgw_info is not None:
            t['qp_cycles'] = evgw_info['cycles']

    t0 = _begin('W')
    # The GW route already inverted [1 - chi0] at every frequency it needed; if
    # it carried omega = 0 along, the static screening this kernel wants is one
    # of those slots and a second imaginary-time sweep buys nothing. The sweep
    # is the cost -- `polarizability_projected_tau` per tau -- and it was 1320 s
    # at the chlorophyllide dimer/cc-pVTZ for a single frequency.
    w_shared = gw_extras.get('w_static')
    if w_shared is not None:
        W_aux = w_shared
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] static W taken from the '
                  f"GW frequency axis (ntau={gw_extras.get('w_static_ntau')}), "
                  'not rebuilt', flush=True)
    else:
        if progress and gw_extras.get('w_static_unavailable'):
            print(f'[bse {time.strftime("%H:%M:%S")}] rebuilding the static W: '
                  f"{gw_extras['w_static_unavailable']}", flush=True)
        _, _, W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors,
                                       distribute=nranks > 1, comm=mpi_comm)
    _end('W', t0)
    # The last two objects the block action reads: the BSE diagonal, and the
    # kernel each rank dresses ITS OWN rows of Zt with. Both come off a
    # replicated solve on reduced data -- a Dyson inversion, a Pade
    # continuation and a Newton -- which is bitwise across ranks only while
    # their dense arithmetic is. naux^2 and nmo doubles to make certain.
    if nranks > 1:
        eps, W_aux = replicate(np.asarray(eps, float), W_aux, comm=mpi_comm)

    lr = LinearResponseSolver(eps, spin_mode='restricted')
    amb = None
    if probe:
        t0 = _begin('probe')
        amb = float(lowest_amb_eigenvalue(
            lr, nocc, polarizability='BSE', W_aux=W_aux, isdf_factors=factors,
            sign_only=(isinstance(probe, str) and probe.lower() == 'sign'),
            stats=stats, comm=mpi_comm)[0])
        _end('probe', t0)
        if amb <= 0:
            raise RuntimeError(
                f'BSE refused before the solve: min eig(A-B) = {amb:.6f} Ha '
                '<= 0 -- the mean-field reference is singlet/triplet unstable '
                'and the Casida omega^2 reduction is invalid there. Check '
                'first that the SCF is the LOWEST solution (vary the initial '
                'guess, follow instabilities); if it is, stabilize the '
                'reference or solve the non-Hermitian problem -- no shift '
                'gives physical roots in this regime.')

    t0 = _begin('davidson')
    omega, X, Y = solve_casida_davidson(lr, nocc, nroots=nroots,
                                        polarizability='BSE', W_aux=W_aux,
                                        isdf_factors=factors,
                                        conv_tol=conv_tol, max_cycle=max_cycle,
                                        guess_factor=guess_factor, stats=stats,
                                        spin=spin, comm=mpi_comm, timings=t)
    _end('davidson', t0)
    if progress and stats:
        print(f'[bse {time.strftime("%H:%M:%S")}] ' + '  '.join(
            f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
            for k, v in sorted(stats.items())), flush=True)
    f_osc, trans_dip = oscillator_strengths(mf, mol, nocc, omega, X, Y)
    info = dict(eps=eps, eps_mf=eps_mf, factors=factors, W_aux=W_aux,
                min_eig_amb=amb, oscillator_strength=f_osc,
                transition_dipole=trans_dip,
                exciton_descriptors=exciton_descriptors(mf, mol, nocc, X, Y),
                timings=t, stats=stats, evgw=evgw_info, nranks=nranks)
    return omega, X, Y, info


def solve_bse_df(mf, mol, nocc, nroots=5, qp='G0W0', probe=True, conv_tol=1e-5,
                 max_cycle=100, guess_factor=GUESS_FACTOR, gw_kwargs=None,
                 progress=None, self_consistency='G0W0', screen_at='mean-field',
                 spin='singlet'):
    """BSE by the density-fitted matrix-free Davidson: mean field in, excitations out.

    The twin of `solve_bse_isdf` on pyscf's OWN Coulomb-fitted three-index
    factor, so the two production routes share every convention and differ
    only in how the interaction is represented:

        mf = dft.RKS(mol, xc=...).density_fit(auxbasis=...); mf.kernel()
        omega, X, Y, info = solve_bse_df(mf, mol, nocc, nroots=5)

    The quasiparticle diagonal comes from the Casida GW route -- the dense RPA
    spectrum in the same auxiliary basis, `calc_qp_energy(mode='casida')` for
    every orbital -- and `self_consistency='evGW'` drives that route through
    the eigenvalue loop, `gw_kwargs` going to `evgw_eigenvalues` (its own
    keywords and the Casida step's; `screening` is refused, the loop here is
    evGW). The static W is built from the same three-index
    factor: at the MEAN-FIELD spectrum for `qp='G0W0'`, the standard G0W0-BSE
    split, and at the fixed point under evGW, since that is the spectrum the
    self-energy was built with. `qp` as an ENERGY ARRAY and `screen_at` follow
    the ISDF twin's rules exactly (see there).

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

    Returns (omega, X, Y, info) with the ISDF twin's fields: eps / eps_mf /
    coeff_df / W_aux / min_eig_amb, length-gauge oscillator_strength,
    transition_dipole and exciton_descriptors per root, per-stage timings (the
    Davidson's breakdown among them), and the evGW record.
    """
    t = {}
    stats = {}
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0
    if getattr(mf, 'with_df', None) is None:
        raise ValueError('solve_bse_df reads the mean field\'s own density '
                         'fitting: build it with mf.density_fit(auxbasis=...) '
                         'or use solve_bse_isdf, which fits for itself')
    if np.asarray(mf.mo_coeff).ndim == 3:
        raise NotImplementedError('solve_bse_df is restricted-spin only, like '
                                  'the Casida GW route that feeds its diagonal')

    def _begin(name):
        if progress:
            print(f'[bse-df {time.strftime("%H:%M:%S")}] {name} ...', flush=True)
        return time.time()

    def _end(name, t0):
        t[name] = time.time() - t0
        if progress:
            print(f'[bse-df {time.strftime("%H:%M:%S")}] {name} done, '
                  f'{t[name]:.1f} s', flush=True)

    if progress:
        print(f'[bse-df {time.strftime("%H:%M:%S")}] natm={mol.natm} '
              f'nao={mol.nao_nr()} nocc={nocc} nroots={nroots} qp={qp}',
              flush=True)

    evgw_info = None
    if str(self_consistency).lower() in ('evgw', 'ev'):
        if 'screening' in (gw_kwargs or {}):
            # the kernel below screens at the fixed point, which is evGW's W;
            # evGW0 keeps the mean field's, and this route does not build that
            raise ValueError("gw_kwargs: self_consistency='evGW' screens the BSE "
                             "at the loop's fixed point, so 'screening' has no "
                             "place here")
        t0 = _begin('evgw')
        qp, evgw_info = evgw_eigenvalues(mf, mol, mode='casida',
                                         **(gw_kwargs or {}))
        mf = shifted_mean_field(mf, qp)
        _end('evgw', t0)
        if progress:
            print(f'[bse-df {time.strftime("%H:%M:%S")}] evGW '
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
        t0 = _begin('qp')
        out = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                             mode='casida', state=list(range(len(eps_mf))),
                             **(gw_kwargs or {}))
        eps = np.array([out[p]['GW'] for p in range(len(eps_mf))]) / HARTREE_TO_EV
        _end('qp', t0)
    else:
        eps = np.asarray(qp, dtype=float)
        if eps.shape != eps_mf.shape:
            raise ValueError(f'qp energies have shape {eps.shape}, the mean '
                             f'field has {eps_mf.shape}.')
        if screen_at not in ('qp', 'mean-field'):
            raise ValueError(f"screen_at={screen_at!r}; choose 'qp' or "
                             f"'mean-field'.")
        if screen_at == 'qp':
            mf = shifted_mean_field(mf, eps)

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
    if probe:
        t0 = _begin('probe')
        amb = float(lowest_amb_eigenvalue(
            lr, nocc, polarizability='BSE', W_aux=W_aux,
            sign_only=(isinstance(probe, str) and probe.lower() == 'sign'),
            stats=stats)[0])
        _end('probe', t0)
        if amb <= 0:
            raise RuntimeError(
                f'BSE refused before the solve: min eig(A-B) = {amb:.6f} Ha '
                '<= 0 -- the mean-field reference is singlet/triplet unstable '
                'and the Casida omega^2 reduction is invalid there.')

    t0 = _begin('davidson')
    omega, X, Y = solve_casida_davidson(lr, nocc, nroots=nroots,
                                        polarizability='BSE', W_aux=W_aux,
                                        conv_tol=conv_tol, max_cycle=max_cycle,
                                        guess_factor=guess_factor, stats=stats,
                                        spin=spin, timings=t)
    _end('davidson', t0)
    f_osc, trans_dip = oscillator_strengths(mf, mol, nocc, omega, X, Y)
    info = dict(eps=eps, eps_mf=eps_mf, coeff_df=coeff, W_aux=W_aux,
                min_eig_amb=amb, oscillator_strength=f_osc,
                transition_dipole=trans_dip,
                exciton_descriptors=exciton_descriptors(mf, mol, nocc, X, Y),
                timings=t, stats=stats, evgw=evgw_info)
    return omega, X, Y, info


def minimax_points_for_bse(eps, nocc, tau_target=DEFAULT_TAU_TARGET):
    """(ntau, residual): smallest minimax time grid for the static-W build.

    The BSE counterpart of `minimax_points_for_gw`: only chi0(i.tau) is
    transformed here, so the chi0 transition range alone binds -- none of the
    self-energy ranges that widen the GW choice apply. Public so a harness that
    RECORDS the grid actually used can call the resolver `isdf_bse_factors`
    itself uses, instead of re-deriving a number that could silently drift from
    it.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    return minimax_points_for_accuracy(eps[virt].min() - eps[occ].max(),
                                       eps[virt].max() - eps[occ].min(),
                                       target=tau_target)


def static_screening_grid(eps, nocc, ntau=None, tau_target=DEFAULT_TAU_TARGET):
    """The one-point (omega = 0) minimax grid a static screening is built on.

    ONE frequency, omega = 0: the BSE kernel and the reaction field's
    self-polarization both want the STATIC screened interaction, so only the
    tau -> omega direction is ever evaluated. with_inverse=False skips the
    omega -> tau matrices, which a single input frequency cannot constrain --
    they would report an error of order one and warn on every call.

    ntau: None or 'auto' sizes the grid from the chi0 transition range through
    `minimax_points_for_bse`.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, _ = minimax_points_for_bse(eps, nocc, tau_target=tau_target)
    return TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, [0.0], [1.0],
                                           with_sine=False, with_inverse=False)


def static_screening_matrix(X, D, eps, nocc, grid=None, mu=None, comm=None):
    """W = [1 - chi0(i.omega = 0)]^-1 in the auxiliary basis, from the
    separable factors in imaginary time.

    THE ONE BUILD OF THE STATIC SCREENED INTERACTION: the BSE kernel's W_aux
    (`isdf_bse_factors`) and a static reaction field built on the same
    factors are this matrix, in the auxiliary gauge of the factors handed in.

    comm: an MPI communicator (or `simulated_world` rank), by default the
    current `distributed` region's, whose ranks all call this in lockstep, on
    X, D, eps and a grid that already agree bit for bit across them --
    `mpi_grid.replicate` at the entry point, which `isdf_bse_factors` does
    before it sizes the grid. Tau is the compute axis -- every point is an M^2
    sweep -- and there is ONE frequency, so the whole result is a single
    (naux, naux) accumulator to all-reduce, the best compute-per-byte sweep in
    the chain. The inversion afterwards is replicated. Exact up to summation
    order: chi0 accumulates as its tau points arrive, so a partition
    re-associates that sum and the serial path alone is bitwise.

    X may be `SlicedFactors` over `comm` (`chi0_imaginary_frequency` gathers
    its branches); D is then the whole, gathered array.
    """
    comm = current_comm() if comm is None else comm
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    chi0 = chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=mu,
                                    tau_indices=tau_mine)
    if nranks > 1:
        reduce_sum(chi0, comm)
    chi0 = chi0[0]
    return np.linalg.inv(np.eye(chi0.shape[-1]) - chi0)


def isdf_bse_factors(mf, mol, nocc, eps=None,
                     auxbasis=None, radii=None, counts=None, factors=None,
                     screening='imaginary-time', ntau=DEFAULT_NTAU,
                     tau_target=DEFAULT_TAU_TARGET, n_start=1,
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
    grid_accuracy: the same grid asked for by NAME -- an accuracy level of
    `ISDF_GRID_ACCURACY` or four explicit shell counts (`resolve_isdf_grid`),
    which refuses instead of substituting.
    factors: pre-built (X_mo, D), to reuse a factorization across states, or
    `SlicedFactors` over `comm`: the sweep gathers what it reads whole once,
    and the return is then (the sliced factors, None, W_aux) -- D is whole
    for W alone and is not handed back.

    screening:
      'imaginary-time'  chi0(i.0) built from the separable factors in imaginary
            time by `LinearResponse/space_time.py`, which is tiled and cubic and
            touches nothing bigger than (naux, naux). The only route that
            reaches production sizes. `static_screening_matrix` on
            `static_screening_grid`, so any other consumer of the same static
            screening builds it identically.
      'df'  chi0 from the three-index factor these factors imply. Exact for the
            given factors, and the reference the imaginary-time route was
            checked against -- 1e-8 relative on the W entries at ntau=18 on
            water and ethene / cc-pVDZ -- but it forms B and then a
            (naux, n_occ, n_vir) array, which at a chlorophyllide-a hexamer /
            cc-pVTZ are 25 TB and 2 TB. Small systems only.

    ntau: imaginary-time points; 'auto' (the default) sizes the grid from the
    Kaltak-Klimes-Kresse test integral over the chi0 transition range at
    `tau_target`, exactly as the GW space-time route does -- but over that one
    range alone, since nothing here transforms a self-energy. More points are
    monotonically safe and then stop paying: measured against the 'df'
    reference on water / cc-pVDZ, W(i.omega = 0) is off by 4.6e-7 at ntau=12,
    1.0e-8 at 18, 2.5e-10 at 24 and 1.8e-10 at 34, the last two sitting on the
    tabulated coefficients' own precision.
    distribute, comm: split the tau points over the ranks of `comm`, by
    default the current `distributed` region's (None follows it, True also
    falls back to COMM_WORLD, False takes none -- as in `solve_bse_isdf`),
    replicate the mean field, the factors and `eps` from rank 0 before the
    grid is sized from them, and all-reduce ONE (naux, naux) matrix. This is
    the best compute-per-byte sweep in the code: every tau point is an M^2
    sweep and the whole result is a single naux^2 block, 0.19 GB at
    dodecacene/cc-pVTZ, against 1320 s of serial sweep at the chlorophyllide
    dimer. Exact up to summation order.
    """
    comm = current_comm() if comm is None else comm
    mpi_comm = grid_comm(comm)[0] if _takes_comm(distribute, comm) else None
    nranks = mpi_comm.Get_size() if mpi_comm is not None else 1
    # Before the grid: `ntau='auto'` is an integer read off the spectrum.
    if nranks > 1:
        replicate_mean_field(mf, mpi_comm)
    factors = (factors if factors is not None
               else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                      counts=counts, n_start=n_start,
                                      grid_accuracy=grid_accuracy))
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
        X_mo, D = _unpack_factors(factors)[:2]
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

    grid = static_screening_grid(eps, nocc, ntau=ntau, tau_target=tau_target)
    W_aux = static_screening_matrix(X_mo, D, eps, nocc, grid, comm=mpi_comm)
    return X_mo, (None if sliced else D), W_aux


def _pair_symmetry(orbsym, occ, virt):
    """Irrep label of every occupied-virtual pair, for the symmetry-block guess."""
    if orbsym is None:
        return None
    orbsym_d2h = np.asarray(orbsym) % 10
    return (orbsym_d2h[occ][:, None] ^ orbsym_d2h[virt][None, :]).ravel()


def bse_pair_diagonal(eps, nocc):
    """eps_a - eps_i on every particle-hole pair, (n_occ, n_vir).

    THE ONE PLACE A BSE PUTS A SPECTRUM ON ITS DIAGONAL. A BSE@GW carries the
    QUASIPARTICLE energies here while its kernel screens at whatever spectrum
    built W -- the G0W0-BSE split -- so the two spectra enter the problem in
    two different places and each route has to take them from the same rule.
    The matrix-free action reads this straight off `lr_solver.eps`; the dense
    route builds A at the screening spectrum and moves the diagonal onto the
    quasiparticle one by the difference of two calls here. Both are this
    expression, and the pair layout -- occupied slow, virtual fast -- is what
    makes a vector of one route a vector of the other.

    The split is by INDEX (`get_occ_virt_indices`), so a quasiparticle array
    whose entries no longer ascend still pairs occupied i with virtual a.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    return eps[virt][None, :] - eps[occ][:, None]


def df_block_action(lr_solver, nocc, lBSE, W_aux,
                    tile_memory_gb=ISDF_TILE_GB, spin='singlet', comm=None):
    """(apply_AB, diag_d): the Casida blocks as an action on a batch of trial
    vectors, contracted straight out of pyscf's three-index factor.

    Returned rather than solved with, so the same action can be timed against
    `isdf_block_action` or driven by an eigensolver other than Davidson's.
    The exchange contractions go through an (nvec, naux, nvirt, nvirt)-shaped
    intermediate, so they are CHUNKED over trial vectors to `tile_memory_gb`;
    Davidson batches ~2-3x the requested roots, which at naphthalene/cc-pVTZ
    is a 24 GB temporary unchunked -- enough to take out the process, not just
    slow it. The naux*nvirt^2 inside one chunk is what cannot be chunked away,
    and is what `isdf_block_action` exists to remove.

    comm: by default the current `distributed` region's. Nothing here is
    divided over ranks: every rank applies the whole serial action, so the
    output is the same on every rank because the inputs are -- the factor
    slices, W_aux and the pair diagonal are lockstepped once here, in one
    call, and the trial vectors at entry, one `lockstep` per batch, in place.
    Both are checked (`mpi_grid.lockstep(check=True)`): only an array a
    digest shows apart from rank 0's is broadcast.
    No other collective runs, so the arithmetic is the serial one and the
    roots are bitwise the serial ones. The returned action carries the
    `comm_clock` its entry lockstep is timed on.
    """
    comm = current_comm() if comm is None else comm
    nranks = comm.Get_size() if comm is not None else 1
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)

    C_ov = lr_solver.df_coeff[:, occ[:, None], virt]
    C_oo = lr_solver.df_coeff[:, occ[:, None], occ]
    C_vv = lr_solver.df_coeff[:, virt[:, None], virt]
    diag_d = bse_pair_diagonal(lr_solver.eps, nocc)
    if nranks > 1:
        # checked: C_vv alone is naux n_vir^2
        diag_d, C_ov, C_oo, C_vv, W_aux = lockstep(
            (diag_d, C_ov, C_oo, C_vv, W_aux), comm, check=True)
    comm_clock = _CommClock()
    factor = KAPPA[spin]

    def apply_V(z):
        t = np.einsum('Pjb,njb->nP', C_ov, z, optimize=True)
        return np.einsum('nP,Pia->nia', t, C_ov, optimize=True)

    # Precompute the W_aux contraction once; redoing it per apply_exchange_* call
    # made each Davidson iteration ~naux times more expensive.
    if W_aux is not None:
        WC_vv = np.einsum('PQ,Qab->Pab', W_aux, C_vv, optimize=True)
        WC_ov = np.einsum('PQ,Qjb->Pjb', W_aux, C_ov, optimize=True)
    else:
        WC_vv = C_vv
        WC_ov = C_ov

    naux = C_ov.shape[0]
    no, nv = len(occ), len(virt)
    per_vec = 8 * naux * nv * max(no, nv)
    chunk = max(1, int(tile_memory_gb * 1e9 / max(per_vec, 1)))

    def apply_exchange_direct(z):
        # Explicit 2-step contraction (a single 3-operand einsum silently costs
        # O(naux*nocc^2*nvirt^2) instead of O(naux*nocc*nvirt*max(nocc,nvirt))).
        # optimize=True matters too: ~13x slower without it for these shapes.
        out = np.empty_like(z)
        for c0 in range(0, len(z), chunk):
            tmp = np.einsum('Pab,njb->nPja', WC_vv, z[c0:c0 + chunk], optimize=True)
            out[c0:c0 + chunk] = np.einsum('Pij,nPja->nia', C_oo, tmp, optimize=True)
        return out

    def apply_exchange_swap(z):
        out = np.empty_like(z)
        for c0 in range(0, len(z), chunk):
            tmp = np.einsum('Pja,njb->nPab', WC_ov, z[c0:c0 + chunk], optimize=True)
            out[c0:c0 + chunk] = np.einsum('Pib,nPab->nia', C_ov, tmp, optimize=True)
        return out

    def apply_AB(z):
        if nranks > 1:
            z = _timed(comm_clock, 'lockstep', lockstep, z, comm,
                       check=True)                     # rank 0's
        # The Hartree term enters A and B identically, so it is contracted once
        # per trial vector rather than once per block. A triplet has kappa = 0
        # and drops it entirely -- not merely scaled to zero, skipped, since it
        # is two of the step's contractions.
        if factor:
            v = factor * apply_V(z)
            Az = diag_d[None, :, :] * z + v
            Bz = v
        else:
            Az = diag_d[None, :, :] * z
            Bz = np.zeros_like(z)
        if lBSE:
            Az = Az - apply_exchange_direct(z)
            Bz = Bz - apply_exchange_swap(z)
        return Az, Bz

    apply_AB.comm_clock = comm_clock
    return apply_AB, diag_d


def _screened_rows(D_mine, D, W_aux, comm=None):
    """This rank's rows of the screened kernel Zt = D W D^T, the bare Z = D D^T
    where W_aux is None (TDHF).

    UNDER A RANK COUNT THE PRODUCT IS REASSOCIATED. `D_mine (W D^T)` forms the
    (naux, M) inner product whole on every rank -- naux^2 M multiply-adds and
    naux M doubles that no rank count divides, 9.4e13 and 26.6 GB at a
    chlorophyllide-a hexamer / cc-pVTZ (M 117762, naux 28236), the same order as
    the rows they are for. `(D_mine W) D^T` is the same matrix from
    nmine naux^2 + nmine naux M multiply-adds through an (nmine, naux)
    intermediate, both of which fall with the rank count: 7.0e13 fewer
    multiply-adds and 20 GB less per rank at four ranks on those shapes.

    The two associations are not bitwise equal, so the serial expression is
    left exactly as it was and the reassociation is taken only where there is a
    rank count to divide. Every rank reads the same size and takes the same
    branch, which the partials being summed across ranks requires.
    """
    if W_aux is None:
        return D_mine @ D.T
    if comm is not None and comm.Get_size() > 1:
        return (D_mine @ W_aux) @ D.T
    return D_mine @ (W_aux @ D.T)


def _timed(clock, kind, fn, *args, **kwargs):
    """fn(*args, **kwargs), its wall seconds added to `clock`, and to its
    `kind` of collective (COMM_KINDS), when there is one."""
    if clock is None:
        return fn(*args, **kwargs)
    t0 = time.time()
    out = fn(*args, **kwargs)
    dt = time.time() - t0
    clock.seconds += dt
    clock.by_kind[kind] += dt
    return out


def isdf_block_action(lr_solver, nocc, lBSE, W_aux, isdf_factors,
                      tile_memory_gb=ISDF_TILE_GB, spin='singlet', comm=None):
    """The same A/B action from a separable RI -- Fock-like builds on the grid.

    Nothing here carries a (naux, nvirt, nvirt) array, and the only object that
    scales as M^2 is the screened kernel Zt, built once. Everything else is
    TILED over grid rows, because every contraction is a sum over them:

        [K z]_ia = sum_k X_o[k,i] sum_k' (Zt * P)[k,k'] X_v[k',a]

    so a row block contributes X_o[blk]^T (Zt[blk] * P[blk]) X_v and the M x M
    Hadamard product is never formed. The tiling is free -- same flop count,
    same GEMM shapes -- and at a chlorophyllide-a hexamer / cc-pVTZ (nao 10980,
    M 117762) it is the difference between holding 103 GB and 310 GB. Zt itself
    is the wall past that, and the same row decomposition is what distributes
    it across ranks.

    comm: with an MPI communicator, by default the current `distributed`
    region's, this rank builds and holds ONLY ITS
    CONTIGUOUS BLOCK OF ROWS of Zt (`contiguous_block`), M^2 / nranks, and the
    row loop runs over that block alone; the product that builds those rows is
    reassociated so its intermediate follows them (`_screened_rows`), the setup
    being the one place a replicated object is as large as the rows themselves.
    Both exchange terms are sums over the row index -- X_o[blk]^T (...) for A
    and, through U, for B -- so each rank produces a partial (n_occ, n_vir) per
    trial vector and ONE all-reduce of the batch, nvec x n_occ x n_vir doubles
    per term, completes them: 1 MB per vector and term at dodecacene against
    the M^2 n_occ flops it stands for.

    THE HARTREE TERM RIDES THE SAME REDUCTION. Its last contraction,
    X_o^T (u X_v), is a sum over the same grid rows, and at anthracene/cc-pVTZ
    (M 3552, n_occ 47, n_vir 513) the whole term measures 7.7% of a serial
    action and 19% of a four-rank one, because it was the only piece a rank
    count did not divide. Split over the rows it costs one more (n_occ, n_vir)
    slab in the batch and no extra collective.

    NOTHING ELSE RUNS AT FULL GRID LENGTH EITHER, which is what a rank count
    has to divide for the action to keep scaling. Three pieces read or write
    the whole grid index and each takes the split that fits its shape:

      z X_v^T is an OUTPUT PARTITION -- a rank builds only the grid rows it
        owns, M n_occ n_vir / nranks flops, and one all-gather of M x n_occ
        doubles (`allgather_blocks`) gives everyone the whole thing, which the
        exchange needs. Gathered, not summed: every row is computed once, by
        its owner, so the ranks hold identical bits and no summation order
        enters.
      p and p D are a REDUCTION -- p is a per-grid-point quantity, so a rank
        forms its own rows of it and contracts them with its own rows of D;
        only the naux-long p D crosses, 11 kB at anthracene against the
        M naux flops it removes.
      X_o^T (Zt * P) X_v is a partial over the rows in its FIRST index and
        whole in its second, so the tail contraction over that second index
        ran at full length on every rank. One reduce of (n_occ, M) makes that
        intermediate the same everywhere and each rank then contracts its own
        columns of it: 1.3 MB per trial vector at anthracene against
        n_occ M n_vir flops, the cheap side at every size this route reaches.
        Contracting X_v first instead would need no reduce at all and costs
        M n_vir per row in place of n_occ M -- ten times more here, since the
        virtual space is the wide one.

    The partials sum in rank order rather than in one GEMM, so a distributed
    run agrees with the serial one to the last bits, not in them.

    THE OUTPUT IS THE SAME ON EVERY RANK BY CONSTRUCTION, which is what lets
    the Davidson and the probe run replicated. The trial vectors are
    lockstepped at entry, rank 0's written into every rank's in place by one
    `lockstep` per batch, and the pair diagonal once here, since d z is the
    one term no collective touches; every other term is gathered or
    all-reduced. A rank whose own iteration drifted by a last bit is handed
    back rank 0's vectors rather than applying its own. The batch lockstep is
    checked (`mpi_grid.lockstep(check=True)`): every rank iterated on this
    action's identical output, so a digest proves the batch is rank 0's and
    only a drifted one is broadcast.
    Serial (comm None or one rank) is bitwise unchanged: the row block is then
    the whole grid and every expression is the one it replaces.

    `SlicedFactors` in `isdf_factors`: this rank's rows are all the action
    reads of X_v and D once Zt is built, so it holds only those; D is gathered
    whole once for the Zt product and dropped with it, and X_o, which
    (Zt * P) X_o reads whole, is gathered once. At the chlorophyllide hexamer
    over 8 ranks the Davidson then holds 8.0 GB of factor arrays per rank
    instead of 57.6 (X_mo, X_ao and D whole plus the X_o and X_v copies), and
    the gathered arrays are the replicated ones, so the output is bitwise the
    replicated-factor action's.

    The returned action carries `comm_clock`, a `_CommClock` credited with the
    wall seconds of the lockstep, the gather and the reductions of every
    application, which the Davidson reports as `davidson_comm`.
    """
    if lr_solver.spin_mode == 'unrestricted':
        raise NotImplementedError("The ISDF Davidson route is restricted-spin only.")
    comm = current_comm() if comm is None else comm
    sliced = isinstance(isdf_factors, SlicedFactors)
    if sliced:
        isdf_factors.require(comm)
        nmo, naux = isdf_factors.nmo, isdf_factors.naux
    else:
        X_mo, D = _unpack_factors(isdf_factors)[:2]
        nmo, naux = X_mo.shape[1], D.shape[1]
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    if nmo != len(lr_solver.eps):
        raise ValueError(f"isdf_factors X has {nmo} orbitals, "
                         f"lr_solver.eps has {len(lr_solver.eps)}.")
    if W_aux is not None and W_aux.shape[0] != naux:
        raise ValueError(f"W_aux is ({W_aux.shape[0]}, ...) but D has "
                         f"{naux} auxiliary functions; they are not from "
                         "the same fit (see isdf_bse_factors).")

    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    diag_d = bse_pair_diagonal(lr_solver.eps, nocc)
    if nranks > 1:
        diag_d = lockstep(diag_d, comm)
    comm_clock = _CommClock()
    factor = KAPPA[spin]
    if sliced:
        # This rank's rows are all the action reads of X_v and D once Zt is
        # built; X_o is read whole by (Zt * P) X_o, and only by the exchange.
        npts, no = isdf_factors.npts, len(occ)
        r0, r1 = isdf_factors.rows
        X_o = (isdf_factors.gather_columns('X_mo', occ, 'X_o') if lBSE
               else None)
        X_o_mine = (X_o[r0:r1] if lBSE
                    else np.ascontiguousarray(isdf_factors.X_mo[:, occ]))
        X_v_mine = np.ascontiguousarray(isdf_factors.X_mo[:, virt])
        X_v = None
        D_mine = isdf_factors.D
    else:
        X_o = np.ascontiguousarray(X_mo[:, occ])
        X_v = np.ascontiguousarray(X_mo[:, virt])
        npts, no = X_o.shape
        r0, r1 = contiguous_block(npts, rank, nranks)  # this rank's rows
        X_o_mine, X_v_mine = X_o[r0:r1], X_v[r0:r1]    # contiguous row blocks
        D_mine = D[r0:r1]
    nmine = r1 - r0
    # Slabs of the reduced batch: the two exchange terms, then the Hartree one
    # where a rank holds only part of it.
    i_hartree = 2 if lBSE else 0                  # the slot after the exchange
    n_reduced = i_hartree + (1 if factor and nranks > 1 else 0)
    # z X_v^T with the grid index FIRST, which is the index the gather splits.
    # Its transpose is the (n_occ, M) operand every contraction below reads.
    zXv_rows = np.empty((npts, no)) if nranks > 1 else None

    # Zt = D W_aux D^T is `screened_interaction_imaginary_time`'s per-tau step,
    # wanted once and statically here. W_aux = None is TDHF, where the exchange
    # kernel is the bare Coulomb, i.e. Zt = Z = D D^T. Only the rows this rank
    # owns are ever formed, and under a rank count so is its inner product
    # (`_screened_rows`).
    if lBSE:
        if W_aux is not None:
            asym = np.abs(W_aux - W_aux.T).max()
            if asym > 1e-10 * np.abs(W_aux).max():
                raise ValueError(f"W_aux is not symmetric (max asymmetry "
                                 f"{asym:.2e}); the B block below relies on it.")
        # Sliced, D is whole only for this product and gone after it.
        Zt = _screened_rows(D_mine, isdf_factors.gather('D') if sliced else D,
                            W_aux, comm)                # (nmine, M)
        rows = max(1, min(max(nmine, 1),
                          int(tile_memory_gb * 1e9 / max(npts * 8, 1))))
        S = np.empty((rows, npts))            # one row block of Zt * P
        T = np.empty((no, npts))              # X_o^T (Zt * P), accumulated
        Tb = np.empty((no, npts))
        U = np.empty((nmine, no))             # (Zt * P) X_o, this rank's rows

    def apply_AB(z):
        if nranks > 1:
            # Rank 0's batch, computed on as a C-ordered array: MKL's GEMM
            # kernels follow operand layout, and a view into the driver's
            # buffer moved water's roots 2e-13 Ha off the root-driven solve's.
            z = np.require(_timed(comm_clock, 'lockstep', lockstep, z, comm,
                                  check=True),
                           requirements='C')
        Az = diag_d[None, :, :] * z
        Bz = np.zeros_like(z)
        if n_reduced:
            # the terms that are partial over this rank's rows, reduced once
            # for the whole batch after the vector loop
            ex = np.zeros((n_reduced,) + z.shape)
        for n, zn in enumerate(z):
            if nranks > 1:
                # This rank's grid rows of z X_v^T, then everyone's: an output
                # partition, so no row is formed twice and the gathered array
                # is the same bits on every rank.
                np.matmul(X_v_mine, zn.T, out=zXv_rows[r0:r1])
                _timed(comm_clock, 'gather', allgather_blocks, zXv_rows, comm)
                zXv = zXv_rows.T                          # (n_occ, M)
            else:
                zXv = zn @ X_v.T                          # (n_occ, M)
            if factor:
                # Hartree needs only P's DIAGONAL, and the bare Z = D D^T is
                # never formed either: two (M, naux) products instead of an
                # (M, M) one. A triplet has kappa = 0 and skips the whole term.
                # p is a per-grid-point quantity, so a rank builds its own rows
                # of it and only p D, naux long, crosses between them.
                p = (np.einsum('kj,kj->k', X_o_mine, zXv_rows[r0:r1],
                               optimize=True) if nranks > 1 else
                     np.einsum('kj,jk->k', X_o, zXv, optimize=True))
                pD = p @ D_mine
                if nranks > 1:
                    _timed(comm_clock, 'reduce_pairs', reduce_sum, pD, comm)
                u = D_mine @ pD
                hartree = factor * (X_o_mine.T @ (u[:, None] * X_v_mine))
                if nranks > 1:
                    ex[i_hartree, n] = hartree
                else:
                    Az[n] += hartree
                    Bz[n] = hartree
            if not lBSE:
                continue
            T[:] = 0.0
            for p0 in range(r0, r1, rows):
                p1 = min(p0 + rows, r1)
                blk = S[:p1 - p0]
                np.matmul(X_o[p0:p1], zXv, out=blk)       # P's rows
                blk *= Zt[p0 - r0:p1 - r0]                # the Hadamard product
                np.matmul(X_o[p0:p1].T, blk, out=Tb)
                np.add(T, Tb, out=T)          # not T += Tb: that rebinds T
                np.matmul(blk, X_o, out=U[p0 - r0:p1 - r0])
            # T is a partial over this rank's rows and WHOLE in the column
            # index, so contracting it with X_v runs at full grid length on
            # every rank. Reduced first, it is the same T everywhere and each
            # rank takes only its own columns of it.
            if nranks > 1:
                _timed(comm_clock, 'reduce_grid', reduce_sum, T, comm)
            ex[0, n] = T[:, r0:r1] @ X_v_mine
            # The B block wants Zt * P^T. Forming that Hadamard product directly
            # reads Zt against a transposed operand and measured as costly as
            # everything else in the step put together; with Zt symmetric --
            # checked above, and it is the screened interaction, so it must be --
            # it is (Zt * P)^T, and (Zt * P)^T X_o is the U already built. Its
            # contraction with X_v runs over the row index, so this rank's rows
            # of U meet this rank's rows of X_v and the sum over ranks is exact.
            ex[1, n] = U.T @ X_v_mine
        if n_reduced:
            if nranks > 1:
                _timed(comm_clock, 'reduce_pairs', reduce_sum, ex, comm)
                if factor:
                    Az += ex[i_hartree]
                    Bz += ex[i_hartree]
            if lBSE:
                Az -= ex[0]
                Bz -= ex[1]
        return Az, Bz

    apply_AB.comm_clock = comm_clock
    return apply_AB, diag_d


def _guess_indices(diag_d, nroots, guess_factor=GUESS_FACTOR, deg_tol=GUESS_DEG_TOL):
    """Occupied-virtual pairs carrying the unit-vector Davidson guess: more of
    them than `nroots`, and never half a degenerate set -- see GUESS_FACTOR for
    what one guess per root costs.
    """
    d = diag_d.ravel()
    n_pair = d.size
    order = np.argsort(d)
    # The +4 floor is for small nroots, where guess_factor alone leaves too
    # few directions to span anything -- 2 vectors at nroots=1.
    n = min(max(guess_factor * nroots, nroots + 4), n_pair)
    # The members of a degenerate set are exactly the pairs that sit in
    # DIFFERENT irreps, so a cut through one is what loses whole symmetry
    # blocks; extend past the cut for as long as the gaps stay degenerate.
    cut = d[order[n - 1]] + deg_tol
    # real_eig sizes its trial-space holders at max(4 * nroots, 2 * space_inc)
    # in the worst (memory-starved) case and a guess wider than that overruns
    # them, so stop extending there -- space_inc is at least min(20, n_pair//2).
    n_max = min(max(4 * nroots, 20, n), n_pair)
    while n < n_max and d[order[n]] <= cut:
        n += 1
    if n == n_max < n_pair and d[order[n]] <= cut:
        warnings.warn(
            f'Davidson guess truncated at {n} vectors inside a cluster of '
            f'orbital-energy differences degenerate to {deg_tol:g} Ha, so the '
            'cut may split a degenerate set and lose the symmetry blocks its '
            'other members would have opened. Ask for fewer roots, or raise '
            'guess_factor and check the roots stop moving.',
            RuntimeWarning, stacklevel=4)
    return order[:n]


def _real_eig_space(max_memory, nroots, n_pair):
    """(space_inc, max_space): the corrections real_eig adds per cycle and the
    trial pairs it holds before collapsing, by its own rule on `max_memory` MB.
    """
    space_inc = (nroots if MAX_SPACE_INC is None
                 else max(nroots, min(MAX_SPACE_INC, n_pair // 2)))
    max_space = int(max_memory * 1e6 / 8 / (4 * n_pair) / 2 - space_inc)
    if max_space < nroots * 4 < n_pair:
        max_space = space_inc * 2
    return space_inc, min(max(max_space, nroots * 4), n_pair)


def _trial_space_memory(nroots, n_pair):
    """MB handed to real_eig: pyscf's own MAX_MEMORY, raised until the trial
    space holds DAVIDSON_SPACE_CYCLES increments within DAVIDSON_SPACE_GB.

    real_eig's four holders take 32 n_pair bytes per trial pair, and its rule
    budgets half of max_memory for them.
    """
    space_inc = _real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]
    target = min(DAVIDSON_SPACE_CYCLES * space_inc,
                 int(DAVIDSON_SPACE_GB * 1e9 / (32 * n_pair)))
    # the inverse of real_eig's rule, half a pair clear of its floor
    return max(param.MAX_MEMORY,
               (target + space_inc + 0.5) * 2 * 32 * n_pair / 1e6)


def _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                  guess_factor=GUESS_FACTOR, stats=None, comm=None,
                  timings=None, refuse_unconverged=False):
    """Drive pyscf's real_eig from a block action, the same iteration on every
    rank.

    A DAVIDSON IS A CHAIN OF DECISIONS -- how many trial vectors the next batch
    carries, which roots have converged, when the space restarts -- each read
    off residuals in dense arithmetic, and on two nodes at
    water/cc-pVDZ an iteration replicated over unequal action outputs fell
    apart: the ranks called the batch reduction with buffers of different
    sizes (MPI_ERR_TRUNCATE), or one iterated on alone. The distributed action
    closes that at its source: it locksteps its trial vectors at entry and
    gathers or all-reduces every term it returns (`isdf_block_action`), so
    every rank's iteration reads the same numbers and takes the same decisions,
    with no protocol between them. The roots, vectors, flags, cycle count and
    pair-space record are lockstepped once at the end, so every rank returns
    rank 0's even had its own subspace eigensolve, or the dense completion of
    a solve whose pair space ran out, differed in a last bit; ranks whose
    iterations had fallen apart meet that lockstep against another rank's
    block-action lockstep and raise ValueError together instead of
    deadlocking.

    stats: optional dict, filled with where the time went. The Davidson is the
    largest stage of a production BSE (37.6% at the chlorophyllide dimer/
    cc-pVDZ) and the total alone does not say whether that is the block action
    doing necessary work or the solver taking too many iterations to get there.
    Counting the action separately from everything around it distinguishes
    "make the action faster" from "give it a better guess", which are different
    pieces of work.

    timings: optional dict, filled with the same `davidson_*` keys serially and
    under a comm, so a rank count's scaling can be read term by term rather
    than off one total -- `davidson_block_action` is what a rank count
    divides, `davidson_comm` the part of it and of the final lockstep spent in
    collectives, `davidson_lockstep_mb` what the locksteps moved and
    `davidson_lockstep_skipped_mb` what their digests proved every rank held
    already (the trial vectors and the result are checked locksteps, 357 MB a
    solve at pentacene/cc-pVTZ),
    `davidson_subspace` what every rank runs whole, and `davidson_iterations`
    with `davidson_vectors_applied` the work the guess asked for.
    `davidson_subspace_max` is the largest trial subspace in pairs,
    `davidson_max_space` the bound real_eig collapses it at and
    `davidson_collapses` how often it did (`_trial_space_memory`).
    `davidson_comm_<kind>` splits `davidson_comm` by collective (COMM_KINDS),
    which says whether the digest, a broadcast or the action's reductions
    carry it.
    `davidson_block_action_by_rank` (under a comm) lists every rank's action
    time, which is where load imbalance in the row split shows.

    refuse_unconverged: a root left above conv_tol raises rather than warns,
    on every rank, since every rank holds rank 0's `converged`.
    """
    nranks = comm.Get_size() if comm is not None else 1
    counters = _davidson_counters()
    comm_clock = getattr(apply_AB, 'comm_clock', None)   # the ISDF action's
    clock0 = comm_clock.seconds if comm_clock is not None else 0.0
    kinds0 = dict(comm_clock.by_kind) if comm_clock is not None else None
    stats0 = lockstep_stats()
    omega, X, Y, converged, cycles, limit = _davidson_root(
        apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym, guess_factor,
        stats, counters, comm=comm)
    if nranks > 1:
        t0 = time.time()
        omega, X, Y, converged, cycles, limit = lockstep(
            (omega, X, Y, converged, cycles, limit), comm, check=True)
        dt = time.time() - t0
        counters['comm_s'] += dt
        counters['comm_by_kind']['lockstep'] += dt
        if comm_clock is not None:
            counters['comm_s'] += comm_clock.seconds - clock0
            for kind in COMM_KINDS:
                counters['comm_by_kind'][kind] += (comm_clock.by_kind[kind]
                                                   - kinds0[kind])
        stats1 = lockstep_stats()
        counters['lockstep_bytes'] = stats1['bytes'] - stats0['bytes']
        counters['lockstep_skipped_bytes'] = (stats1['skipped_bytes']
                                              - stats0['skipped_bytes'])
    by_rank = (_block_action_by_rank(counters['action_s'], comm)
               if nranks > 1 else None)
    if timings is not None:
        timings.update({
            'davidson_block_action': counters['action_s'],
            'davidson_subspace': counters['loop_s'] - counters['action_s'],
            'davidson_comm': counters['comm_s'],
            'davidson_iterations': counters['iterations'],
            'davidson_subspace_max': counters['space_max'],
            'davidson_max_space': counters['space_bound'],
            'davidson_collapses': counters['collapses'],
            'davidson_vectors_applied': counters['vectors'],
            # MB as 1e6 bytes, the unit pyscf's max_memory is in.
            'davidson_lockstep_mb': counters['lockstep_bytes'] / 1e6,
            'davidson_lockstep_skipped_mb':
                counters['lockstep_skipped_bytes'] / 1e6,
            **{f'davidson_comm_{kind}': counters['comm_by_kind'][kind]
               for kind in COMM_KINDS}})
        if by_rank is not None:
            timings['davidson_block_action_by_rank'] = by_rank
    # A root that never converged still comes back with an energy attached, and
    # returning it unremarked is how a Davidson result silently stops meaning
    # anything. It does NOT flag the guess failure GUESS_FACTOR describes --
    # that one converges. Every rank warns, because every rank returns it.
    if not converged.all():
        stuck = np.flatnonzero(~converged)
        # real_eig also stops early, unconverged, when no correction survives
        # its linear-dependence test: the subspace could not grow, and more
        # cycles would not have helped. Where the pair space ran out first,
        # that is the stop, and no tolerance moves it (`_davidson_root`).
        if limit is not None and limit[2]:
            stop, advice = (f'after completing densely on all {diag_d.size} '
                            'pairs', 'That residual is the dense solve\'s '
                            'round-off; loosen conv_tol.')
        elif limit is not None:
            stop, advice = (
                f'at cycle {cycles} of {max_cycle}, where the trial subspace '
                f'held {limit[0]} of the {diag_d.size} particle-hole pairs and '
                f'the {limit[1]} corrections of a cycle could not be paired in '
                'what was left of them',
                'More cycles would not help, and above BSE_DENSE_MAX_NOV = '
                f'{BSE_DENSE_MAX_NOV} pairs the solve is not completed densely: '
                'ask for fewer roots, raise guess_factor, or use the dense '
                'solver.')
        elif cycles >= max_cycle:
            stop, advice = (f'at the cycle cap, max_cycle={max_cycle}',
                            'Raise max_cycle, or loosen conv_tol.')
        else:
            stop, advice = (f'at cycle {cycles} of {max_cycle}, where no new '
                            'direction survived the linear-dependence test',
                            'More cycles would not help; loosen conv_tol.')
        message = (f'Davidson left {stuck.size} of {nroots} roots unconverged '
                   f'at |r| <= {conv_tol:g} (roots {stuck.tolist()}), stopping '
                   f'{stop}; their energies are whatever the last subspace '
                   'happened to give.')
        if refuse_unconverged:
            raise RuntimeError(f'{message} Refused: the eigenvectors of an '
                               'unconverged root are wrong by its residual, '
                               f'and so is every derivative read off them. '
                               f'{advice}')
        warnings.warn(f'{message} {advice}', RuntimeWarning, stacklevel=3)
    return omega, X, Y


def _davidson_root(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                   guess_factor=GUESS_FACTOR, stats=None, counters=None,
                   comm=None):
    """One Davidson: (omega, X, Y, converged, cycles, limit), the same on
    every rank of `comm`, which reaches only the (A-B) probe run when the
    projected problem breaks down, and the dense completion when the pair
    space ran out; `limit` is None, or (pairs held, corrections asked, whether
    the solve was completed densely) at the last cycle that ran out."""
    no, nv = diag_d.shape
    n_pair = no * nv
    ncall = [0]
    nvec = [0]
    t_action = [0.0]
    t_total = [0.0]
    # real_eig's subspace is the trial pairs it has handed over since it last
    # collapsed the space, which it does when a cycle's kept corrections would
    # overrun max_space, handing over its nroots Ritz vectors instead. So a
    # batch that overruns the bound, or one of nroots after a cycle that asked
    # for more corrections than the bound had room for, starts the count again;
    # exact unless the partner test left exactly nroots of such a cycle's
    # corrections.
    max_memory = _trial_space_memory(nroots, n_pair)
    max_space = _real_eig_space(max_memory, nroots, n_pair)[1]
    space = [0]
    space_max = [0]
    collapses = [0]
    asked = [0]                      # corrections the last cycle asked for
    batch_max = [0]                  # the widest batch the action was handed
    # THE PAIR SPACE CAN RUN OUT BEFORE THE TOLERANCE DOES. real_eig adds each
    # correction with its x/y-swapped partner, so m trial pairs span 2m of the
    # 2 n_pair dimensions, and once one cycle's k corrections outnumber the
    # n_pair - m pairs left they span the whole remainder: every direction in
    # it is then its own partner (x = +-y), the partner test drops it, and the
    # iteration stops with "no new direction". water/6-31G (40 pairs) BSE@G0W0
    # on PBE, 3 roots: 19 to 20 corrections for the 6 pairs left at 34, none
    # kept, roots at |r| up to 1.2e-2 and 1.5e-5 Ha off the dense ones at
    # conv_tol 1e-8 and 1e-5 alike. Such a solve is completed densely
    # (`_dense_casida`) when it ends unconverged or breaks down, and only
    # then: water/cc-pVDZ (95 pairs) runs out on every one of 96 solves at 2
    # to 5 roots and converges on the complete space, so a rule that acted
    # whenever the space ran out would move the bits of solves that work.
    limit = [None]

    def vind(xys):
        t_enter = time.time()
        xys = np.asarray(xys).reshape(-1, 2, no, nv)
        ncall[0] += 1
        nvec[0] += 2 * xys.shape[0]          # a block action per X and per Y
        n = xys.shape[0]
        if ncall[0] > 1 and (space[0] + n > max_space or
                             (n == nroots and space[0] + asked[0] > max_space)):
            collapses[0] += 1
            space[0] = n
        else:
            space[0] += n
        space_max[0] = max(space_max[0], space[0])
        batch_max[0] = max(batch_max[0], xys.shape[0])
        t0 = time.time()
        Ax, Bx = apply_AB(xys[:, 0])
        Ay, By = apply_AB(xys[:, 1])
        t_action[0] += time.time() - t0
        top = (Ax + By).reshape(xys.shape[0], -1)
        bot = (Bx + Ay).reshape(xys.shape[0], -1)
        out = np.hstack([top, -bot])
        t_total[0] += time.time() - t_enter
        return out

    hdiag = np.hstack([diag_d.ravel(), -diag_d.ravel()])
    floor = _residual_floor(diag_d)

    def precond(dx, e):
        e = np.atleast_1d(e)
        asked[0] = len(e)
        if len(e) > n_pair - space[0]:
            limit[0] = (space[0], len(e))
        d = hdiag[None, :] - e[:, None]
        d[np.abs(d) < 1e-8] = 1e-8
        t = dx.reshape(len(e), -1) / d
        # THE LENGTH OF A CORRECTION IS ITS ACCEPTANCE TEST. real_eig keeps one
        # while its squared norm outside the subspace exceeds lindep, read
        # before normalizing, so at the raw length r / (d - omega) |r| stalls
        # near 1e-7 whatever conv_tol asks (2.4e-7 at naphthalene/cc-pVDZ).
        # Below DAVIDSON_SIZED_RESIDUAL a correction is sized so that it is
        # kept while its part outside the subspace, as a fraction of itself,
        # exceeds DAVIDSON_MIN_NEW_FRACTION, which keeps the subspace
        # conditioned, and floor / |r|, the relative round-off of the residual
        # it came from. A power of two scales exactly.
        r = np.linalg.norm(dx.reshape(len(e), -1), axis=1)
        sized = r < DAVIDSON_SIZED_RESIDUAL
        if sized.any():
            new_min = np.maximum(DAVIDSON_MIN_NEW_FRACTION, floor / r[sized])
            length = np.sqrt(DAVIDSON_LINDEP) / new_min
            raw = np.linalg.norm(t[sized], axis=1)
            t[sized] *= np.exp2(np.round(np.log2(length / raw)))[:, None]
        return t.reshape(dx.shape)

    order = _guess_indices(diag_d, nroots, guess_factor)
    x0 = np.zeros((len(order), 2 * n_pair))
    x0[np.arange(len(order)), order] = 1.0
    x0sym = x_sym[order] if x_sym is not None else None

    def action(z):
        t0 = time.time()
        out = apply_AB(z)
        t_action[0] += time.time() - t0
        return out

    def completes():
        """Whether a failed solve is finished densely: the pair space ran out,
        and it is one the dense route holds (BSE_DENSE_MAX_NOV)."""
        return limit[0] is not None and n_pair <= BSE_DENSE_MAX_NOV

    t_eig0 = time.time()
    try:
        converged, e, xy = real_eig(vind, x0, precond, tol_residual=conv_tol,
                                     nroots=nroots, x0sym=x0sym, max_cycle=max_cycle,
                                     max_memory=max_memory,
                                     lindep=DAVIDSON_LINDEP,
                                     verbose=logger.Logger(sys.stdout, 0))
    except np.linalg.LinAlgError as exc:
        # pyscf's real_eig Cholesky-factorizes the projected (A-B) block, so
        # this is the response subspace losing positive definiteness -- the
        # instability regime of the mean-field reference (long acenes reach it
        # near a11), where the omega^2 reduction is INVALID rather than merely
        # ill-conditioned. No shift or retry inside this solver gives physical
        # roots there: the dense CasidaSolver's eta-shift is a diagnostic that
        # changes the spectrum, not an answer. Stabilize the reference, or
        # solve the non-Hermitian problem directly. A subspace that ran out of
        # pair space breaks down here too -- water/6-31G BSE@HF, 4 roots at
        # 1e-5, after the partner test kept 3 pairs for the last one left --
        # and the dense completion tells the two apart.
        if not completes():
            raise _casida_breakdown(apply_AB, diag_d, conv_tol, comm) from exc
        converged = None
    dense = None
    if completes() and (converged is None or not np.all(converged)):
        try:
            dense = _dense_casida(action, diag_d, nroots, conv_tol,
                                  batch_max[0])
        except np.linalg.LinAlgError as exc:
            raise _casida_breakdown(apply_AB, diag_d, conv_tol, comm) from exc
        ncall[0] += 1
        nvec[0] += n_pair
        space_max[0] = n_pair
    t_eig = time.time() - t_eig0           # the iteration, action included
    if stats is not None:
        stats.update({'davidson_vind_calls': ncall[0],
                      'davidson_block_actions': nvec[0],
                      'davidson_action_s': t_action[0],
                      'davidson_vind_s': t_total[0],
                      'davidson_dense_completion': dense is not None})
    if counters is not None:
        counters['action_s'] = t_action[0]
        counters['loop_s'] = t_eig
        counters['iterations'] = ncall[0]
        counters['vectors'] = nvec[0]
        counters['space_max'] = min(space_max[0], n_pair)
        counters['space_bound'] = max_space
        counters['collapses'] = collapses[0]
    if limit[0] is not None:
        limit[0] = (int(limit[0][0]), int(limit[0][1]), dense is not None)
    if dense is not None:
        omega, X, Y, converged = dense
        return omega, X, Y, converged, ncall[0], limit[0]
    omega = np.asarray(e)
    X = np.zeros((n_pair, len(omega)))
    Y = np.zeros((n_pair, len(omega)))
    for k, z in enumerate(xy):
        x, y = z.reshape(2, no, nv)
        norm = np.sqrt(abs(np.sum(x**2) - np.sum(y**2)))
        X[:, k] = (x / norm).ravel()
        Y[:, k] = (y / norm).ravel()
    # One block action per cycle, so the action count IS the cycle count; a
    # dense completion counts as one more.
    return omega, X, Y, np.asarray(converged), ncall[0], limit[0]


def _casida_breakdown(apply_AB, diag_d, conv_tol, comm=None):
    """The RuntimeError for a projected (A-B) block that is not positive
    definite, telling a stable reference's numerical breakdown from an
    unstable reference by the (A-B) probe, run over `comm`."""
    try:
        amb = float(_lowest_amb_from_action(apply_AB, diag_d, comm=comm)[0])
        detail = f'measured min eig(A-B) = {amb:.6f} Ha'
    except Exception:
        amb = None
        detail = ('min eig(A-B) probe did not converge; smallest '
                  f'orbital-energy difference {diag_d.min():.4f} Ha')
    # A congruence of a positive definite (A-B) is positive definite, so
    # with the reference stable it is the SUBSPACE that failed: real_eig's
    # projected problem loses accuracy as |r| nears round-off.
    if amb is not None and amb > 0:
        return RuntimeError(
            'Casida-form Davidson broke down numerically: the projected '
            '(A-B) block is not positive definite although (A-B) itself '
            f'is ({detail}), so the reference is stable and the trial '
            'subspace is what failed, as it does when conv_tol '
            f'({conv_tol:g} Ha) asks for a residual approaching round-off '
            f'({_residual_floor(diag_d):.1e} Ha on this pair space). '
            'Loosen conv_tol.')
    return RuntimeError(
        'Casida-form Davidson failed: the projected (A-B) block is not '
        'positive definite, the signature of a singlet/triplet '
        f'instability of the mean-field reference ({detail}). The '
        'omega^2 reduction is invalid in this regime and shifted or '
        'retried roots would not be physical. Check first that the SCF is '
        'the LOWEST solution (vary the initial guess, follow '
        'instabilities) -- a converged non-minimum reference produces '
        'exactly this failure.')


def _dense_casida(apply_AB, diag_d, nroots, conv_tol, batch):
    """(omega, X, Y, converged): the `nroots` lowest Casida roots from A and B
    built whole out of the block action, one unit vector per pair.

    The completion of a Davidson whose pair space ran out (`_davidson_root`):
    the subspace and its remainder are then the whole space, so its dense
    problem is the one left to solve, and at water/6-31G it costs 40 actions
    after the Davidson's 68. The unit vectors go through the action in batches
    of `batch`, the widest trial batch the Davidson already handed it, and
    are lockstepped at its entry like any other, so every rank builds the same
    A and B. The roots come from the Hermitian form L^T (A + B) L = Z w^2 Z^T
    with A - B = L L^T, whose Cholesky raises LinAlgError where (A - B) is not
    positive definite; X + Y = L Z w^-1/2 and X - Y = L^-T Z w^1/2 carry
    <X|X> - <Y|Y> = 1. `converged` is each root's Casida residual against
    conv_tol, as the Davidson's is.
    """
    no, nv = diag_d.shape
    n_pair = no * nv
    batch = max(1, int(batch))
    A = np.empty((n_pair, n_pair))
    B = np.empty((n_pair, n_pair))
    for c0 in range(0, n_pair, batch):
        c1 = min(c0 + batch, n_pair)
        unit = np.zeros((c1 - c0, n_pair))
        unit[np.arange(c1 - c0), np.arange(c0, c1)] = 1.0
        Az, Bz = apply_AB(unit.reshape(c1 - c0, no, nv))
        A[c0:c1] = Az.reshape(c1 - c0, n_pair)
        B[c0:c1] = Bz.reshape(c1 - c0, n_pair)
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(A - B)
    w2, Z = np.linalg.eigh(L.T @ (A + B) @ L)
    if w2[0] <= 0.0:
        # (A - B) positive definite, (A + B) not: the other instability.
        raise RuntimeError(
            f'Casida problem completed densely has omega^2 = {w2[0]:.3e} '
            '<= 0 Ha^2: the mean-field reference is unstable and the omega^2 '
            'reduction is invalid there.')
    omega = np.sqrt(w2[:nroots])
    Z = Z[:, :nroots]
    xpy = (L @ Z) / np.sqrt(omega)[None, :]
    xmy = solve_triangular(L, Z, lower=True, trans='T') * np.sqrt(omega)[None, :]
    X = 0.5 * (xpy + xmy)
    Y = 0.5 * (xpy - xmy)
    r = np.sqrt(np.sum((A @ X + B @ Y - X * omega)**2, axis=0)
                + np.sum((B @ X + A @ Y + Y * omega)**2, axis=0))
    return omega, X, Y, r <= conv_tol
