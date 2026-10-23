"""Matrix-free Casida/RPA/BSE Davidson, with two ways of applying the same A/B.

The drivers on it (mean field in, excitations out) are
`BSE.solve.solve_bse_isdf` and `solve_bse_df`. The block actions are in
`block_actions`, the (A-B) probe in `probe` and what the loops share in
`davidson_shared`; the Casida loop and the Tamm-Dancoff solve (the one
symmetric Davidson of `Solvers.davidson`, on the A half of the block action)
are here, and so is the walk's warm start of either (`DavidsonWarmStart`).

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
doi 10.1021/acs.jctc.2c00531 do the same with PADF in place of ISDF.

The auxiliary gauge is the trap. pyscf's cderi is L^-1-whitened while a
separable RI fits with the symmetric V^-1/2, and the two differ by an orthogonal
rotation of the auxiliary index. A W_aux from one paired with factors from the
other is silently wrong -- every intermediate still looks self-consistent, and
in the GW work the same mistake moved a quasiparticle energy by 1.4 eV while
Sigma(i.omega) still agreed to 1e-5. `isdf_bse_factors` returns X, D and W_aux
from a single fit, which is the only reliable way to keep them in one gauge.

The Davidson guess is the trap the two routes share, and it is silent in the
same way: one unit vector per requested root returns `nroots` roots that all
report converged and are not the lowest ones whenever the smallest
orbital-energy differences are degenerate. See constants.DAVIDSON_GUESS_FACTOR.

On a large molecule the DF route needs an
(naux, nvirt, nvirt) array per trial vector and an (naux, n_occ, n_vir) one
for its static W; the separable route holds the (M, M) Zt instead and takes
its W from imaginary time. Zt spread across ranks is the row split of
`isdf_block_action`.

Over ranks the Davidson and the (A-B) probe run on every rank alike. MPI lives
inside the realization kernels alone: the block action locksteps its trial
vectors at entry and gathers or all-reduces every term it returns, so its
output is the same on every rank; and the Davidson's trial space is cut by
pair rows (`trial_space`), its sums over the pair index reduced and its new
batch gathered, so every decision the iteration takes is read off the same
small matrices on every rank.
"""
import sys
import time
import warnings

import numpy as np
from pyscf import gto
from pyscf.lib import logger, param
from scipy.sparse.linalg import ArpackNoConvergence

from src.Base.numerics.constants import (
    BYTES_PER_GB, BSE_DENSE_MAX_NOV, DAVIDSON_FLOOR_EPS_MULTIPLE,
    DAVIDSON_GUESS_FACTOR, DAVIDSON_LINDEP, DAVIDSON_MIN_NEW_FRACTION,
    DAVIDSON_PRECONDITIONER, DAVIDSON_SIZED_RESIDUAL, DAVIDSON_SPACE_FRACTION,
    DAVIDSON_SPACE_GB, DAVIDSON_WARM_START_MAX_SHIFT)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.step_failed import DavidsonNotConverged, DavidsonRefusal, StepFailed
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.Base.utils.parallel.mpi_grid import (
    broadcast, current_comm, lockstep, lockstep_stats, reduce_sum)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.Solvers.davidson import solve_symmetric
from src.SingleReference.LinearResponse.block_actions import (
    ACTION_PIECES, COMM_KINDS, block_action, bse_pair_diagonal)
from src.SingleReference.LinearResponse.casida import (CasidaInstability,
                                                       casida_eigensolve)
from src.SingleReference.LinearResponse.davidson_shared import (
    action_working_bytes, check_preconditioner, correction, guess_indices,
    preconditioner_diagonal, real_eig_space, tracked_roots,
    trial_space_memory)
from src.SingleReference.LinearResponse.probe import (
    AmbProbeNotConverged, lowest_amb_from_action)
from src.SingleReference.LinearResponse.trial_space import (SUBSPACE_PIECES,
                                                            PairRows,
                                                            casida_rows,
                                                            warm_trial_space)


class DavidsonWarmStart:
    """The last converged Casida vectors of a walk, with the orbitals and the
    geometry they are written in.

    One holder serves every Davidson along a walk (an optimization, a
    finite difference): each solve leaves its roots here under its spin
    (`keep`) and the next one of that spin starts from them (`guess`) where
    the atoms, the basis and the pair space are the same and no atom moved
    further than `max_shift` Bohr -- the rule `scf_convergence.WarmStart`
    applies to the density.

    THE VECTORS ARE CARRIED INTO THE NEW ORBITALS. X_ia and Y_ia are
    amplitudes on orbital pairs, and the orbitals of the next geometry come
    out of their own eigensolve with their own signs and, inside a
    degenerate set, their own rotation; a vector read in the old orbitals is
    a guess only up to those. With T = C_old^T S(R_old, R_new) C_new, the MO
    overlap across the step (`gto.intor_cross`, exact at any separation),
    phi_i^old = sum_j T_ij phi_j^new to first order in the step, so
    X_new = T_oo^T X T_vv and Y_new likewise: the sign flips and the
    rotations are undone, and what is left is the change of the states
    themselves, which is what the Davidson then converges.

    A guess is where the iteration starts, not what it converges to: the
    roots are the cold start's to the Davidson's tolerance. `used` and
    `declined` count the two outcomes.

    ALONG A WALK, FROM THE LAST ACCEPTED GEOMETRY. A walk evaluates the
    steps it rejects too, so the excited-state chain does not `keep` every
    solve: it `offer`s each solve's roots under its geometry, and the walk's
    accepted step (`ExcitedStateChain.commit`) makes the ones offered there
    the held ones (`accept`), every spin solved there at once. A rejected
    trial step, a Hessian displacement or an energy read elsewhere never
    seeds the next step.
    """

    def __init__(self, max_shift=DAVIDSON_WARM_START_MAX_SHIFT):
        self.max_shift = float(max_shift)
        # spin -> (mol, mo_coeff, nocc, X, Y): a chain and its spin views
        # share one holder, and each spin starts from its own roots
        self.held = {}
        # (geometry, spin) -> what `keep` would hold, offered since the last
        # accepted geometry and waiting for `accept`
        self.offered = {}
        self.used = self.declined = 0

    def to_record(self):
        """The held roots by spin, (coordinates, mo_coeff, nocc, X, Y) each,
        and `max_shift`: what a chain's record carries so that a chain
        rebuilt from it starts its next solve where this one would
        (`from_record`). Offers waiting for `accept` are not kept: a walk
        writes its checkpoint after its accepted step."""
        return {'max_shift': self.max_shift,
                'held': {spin: (np.array(mol.atom_coords(), float), mo, nocc,
                                X, Y)
                         for spin, (mol, mo, nocc, X, Y) in self.held.items()}}

    @classmethod
    def from_record(cls, record, mol):
        """The holder `to_record` wrote, its geometries placed on `mol`'s
        basis (`gto.Mole.set_geom_`, Bohr)."""
        out = cls(record['max_shift'])
        for spin, (coords, mo, nocc, X, Y) in record['held'].items():
            at = mol.set_geom_(np.asarray(coords, float), unit='Bohr',
                               inplace=False)
            out.keep(at, mo, nocc, spin, X, Y)
        return out

    def offer(self, mol, mo_coeff, nocc, spin, X, Y):
        """The roots (X, Y) of a solve of `spin` at `mol`, held as `keep`
        holds them once a walk accepts `mol` (`accept`)."""
        self.offered[(_nuclei(mol), spin)] = (
            mol, np.array(mo_coeff, dtype=float), int(nocc),
            np.array(X, dtype=float), np.array(Y, dtype=float))

    def accept(self, mol):
        """A walk accepted `mol`: the roots offered there, of every spin,
        are the held ones; every other offer is dropped. True when one was
        offered there."""
        here = _nuclei(mol)
        taken = {spin: v for (at, spin), v in self.offered.items()
                 if at == here}
        self.held.update(taken)
        self.offered = {}
        return bool(taken)

    def keep(self, mol, mo_coeff, nocc, spin, X, Y):
        """Hold the roots (X, Y), (n_pair, nroots) each, of a solve of
        `spin` in the orbitals `mo_coeff` of `mol`."""
        self.held[spin] = (mol, np.array(mo_coeff, dtype=float), int(nocc),
                           np.array(X, dtype=float), np.array(Y, dtype=float))

    def guess(self, mol, mo_coeff, nocc, spin):
        """The held roots of `spin` carried into `mo_coeff` of `mol`,
        (X, Y), where they belong to its atoms, basis and pair space near its
        geometry; else None."""
        held = self.held.get(spin)
        fits = (held is not None and int(nocc) == held[2]
                and np.shape(mo_coeff) == held[1].shape
                and mol.nao_nr() == held[0].nao_nr()
                and np.array_equal(mol.atom_charges(), held[0].atom_charges())
                and float(np.abs(mol.atom_coords()
                                 - held[0].atom_coords()).max())
                <= self.max_shift)
        if not fits:
            self.declined += 1
            return None
        self.used += 1
        mol0, mo0, no, X, Y = held
        t = (mo0.T @ gto.mole.intor_cross('int1e_ovlp', mol0, mol)
             @ np.asarray(mo_coeff, float))
        t_oo, t_vv = t[:no, :no], t[no:, no:]
        nv, k = t_vv.shape[0], X.shape[1]

        def carried(Z):
            Z = Z.reshape(no, nv, k)
            return np.einsum('ij,iak,ab->jbk', t_oo, Z, t_vv,
                             optimize=True).reshape(no * nv, k)

        return carried(X), carried(Y)


def _nuclei(mol):
    """The coordinates' bits in Bohr and the charges of `mol`: the geometry
    an offered root set belongs to."""
    return (np.asarray(mol.atom_coords(), float).tobytes(),
            np.asarray(mol.atom_charges()).tobytes())


class CasidaBreakdown(StepFailed):
    """A projected (A-B) block that is not positive definite, carrying the
    min eig(A-B) the probe measured on the whole pair space, or None where
    that probe did not converge (`_casida_breakdown`)."""

    def __init__(self, message, min_eig_amb):
        super().__init__(message)
        self.min_eig_amb = min_eig_amb


class UnstableReference(CasidaInstability):
    """The Casida problem refused for an unstable reference: min eig(A-B)
    <= 0 measured, or (A - B) or omega^2 not positive on the whole pair
    space; the dense solver's `CasidaInstability`, and the RuntimeError the
    BSE drivers raise. `min_eig_amb` is the measured minimum, or None."""

    def __init__(self, message, min_eig_amb=None):
        super().__init__(message)
        self.min_eig_amb = min_eig_amb


def bse_settings(numerics, *names):
    """The values of the Davidson's choices `names` -- 'nroots', 'conv_tol',
    'max_cycle', 'guess_factor', 'preconditioner' -- read off `numerics`'
    `bse` section (None: the defaults), as a tuple in the order asked: one
    default for every entry point and the chain."""
    bse = numerics_or_default(numerics).bse
    return tuple(getattr(bse, name) for name in names)


def solve_casida_davidson(lr_solver, nocc, numerics=None,
                           polarizability='RPA', W_aux=None, orbsym=None,
                           isdf_factors=None, stats=None, spin='singlet',
                           comm=None, timings=None,
                           refuse_unconverged=False, max_memory=None,
                           read_roots=None, read_tol=None, guess=None,
                           tda=False, one_body=None, tracked=None,
                           probe=False):
    """Matrix-free Davidson solver for the `nroots` lowest Casida excitation energies (never forms dense A/B).

    For a handful of low-lying states; vertex-correction sums still need the
    full spectrum (use build_casida_matrices + CasidaSolver for that).
    Iteration by our Casida Davidson (`run_davidson`); the A/B matrix-free action is ours,
    validated against build_casida_matrices to machine precision on both routes.

    polarizability: 'RPA' (Hartree-only), 'TDHF' (bare exchange), or 'BSE'
    (screened exchange, needs W_aux from solve_rpa_screening).
    orbsym: optional pyscf orbital irrep IDs; enables symmetry-block Davidson
    (validated correct, but measured no speedup on benzene).
    isdf_factors: (X_mo, D) from `isdf_bse_factors` -- switches the block action
    to the separable-RI Fock-like builds described in the module docstring.
    W_aux must come from the same call, or the auxiliary gauges disagree.
    numerics: the solver's choices (`Numerics.bse`, read by `bse_settings`;
    None for the defaults): nroots, conv_tol, max_cycle, guess_factor and
    preconditioner, below.
    guess_factor: guess vectors per requested root. The default spans enough
    symmetry blocks on everything tested here; raise it and check the roots stop
    moving when a spectrum is dense or highly degenerate. See DAVIDSON_GUESS_FACTOR.
    spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0, the bare-exchange
    term absent). The screened term is spin-independent.
    comm: an MPI communicator whose ranks all run this solver together, by
    default the current `distributed` region's; the ISDF block action divides
    the rows of Zt between them (`isdf_block_action`) and every rank runs the
    same iteration on its lockstepped output (`run_davidson`); the DF action
    is not divided and runs whole on every rank on lockstepped inputs.
    timings: optional dict, filled with the `davidson_*` breakdown of
    `run_davidson` -- the same keys serially and under a comm, so the two are
    comparable -- plus `davidson_setup`, the action's own build: the collocation
    slices and this rank's rows of the screened kernel, paid once before the
    first trial vector and therefore invisible in `davidson_block_action`.
    conv_tol: the residual |r| every root must reach, in Hartree. One below the
    round-off floor of the residual is refused before the action is built
    (`check_residual_floor`).
    refuse_unconverged: False by default: a root left above conv_tol is
    returned as solved with a RuntimeWarning and flagged in `timings`
    (`davidson_flags`), which the caller copies into its record
    (refuse inputs, warn numerics).
    True raises `DavidsonNotConverged`, since its energy and eigenvector are
    wrong by its residual.
    read_roots, read_tol: with refuse_unconverged, the lowest `read_roots`
    roots are the ones the caller differentiates and `read_tol` the residual
    it needs of them; `run_davidson`'s.
    max_memory: MB this rank may hold, which sizes the trial space;
    `run_davidson`'s.
    preconditioner: 'bare' (d) or 'screened' (d - (ii|W|aa)), the diagonal
    the corrections divide by; `run_davidson`'s.
    guess: (X, Y), (n_pair, k) each -- a previous geometry's roots carried
    into these orbitals (`DavidsonWarmStart.guess`) -- started from with
    the unit vectors; None starts from the unit vectors alone.
    `run_davidson`'s.
    tda: the Tamm-Dancoff problem A X = omega X with B = 0, solved by a
    Hermitian Davidson on the A half of the same block action
    (`_tda_davidson_root`); Y is returned zero. Its roots lie at or above
    the full ones for a stable reference.
    one_body: the off-diagonal one-body blocks a quasiparticle Hamiltonian
    adds to A (`with_one_body`), or None for the diagonal eps alone.
    tracked: optional dict, filled with the requested roots and the tracked
    Ritz pairs above them (`run_davidson`'s), for a warm start to carry.
    probe: off (False) by default. True or 'sign' measures min eig(A - B)
    after the solve on the same block action (`probe_after_solve`, the
    probe `solve_bse_isdf` runs) and refuses the roots (`UnstableReference`)
    where it is not positive; `stats` then also carries `min_eig_amb`.
    Skipped under `tda`, which makes no omega^2 reduction.
    Returns (omega, X, Y) normalized <X|X>-<Y|Y>=1.
    """
    nroots, conv_tol, max_cycle, guess_factor, preconditioner = bse_settings(
        numerics, 'nroots', 'conv_tol', 'max_cycle', 'guess_factor',
        'preconditioner')
    check_preconditioner(preconditioner)
    if tda and orbsym is not None:
        raise ValueError('the Tamm-Dancoff Davidson takes no pair symmetry: '
                         'its trial space is one Hermitian block')
    comm = current_comm() if comm is None else comm
    apply_AB, diag_d = casida_action(lr_solver, nocc, polarizability, W_aux,
                                      isdf_factors, conv_tol, spin=spin,
                                      comm=comm, timings=timings, tda=tda)
    apply_AB = with_one_body(apply_AB, one_body)
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    omega, X, Y = run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                                _pair_symmetry(orbsym, occ, virt),
                                guess_factor, stats=stats, comm=comm,
                                timings=timings,
                                refuse_unconverged=refuse_unconverged,
                                max_memory=max_memory,
                                preconditioner=preconditioner,
                                read_roots=read_roots, read_tol=read_tol,
                                guess=guess, tda=tda, tracked=tracked)
    if probe and not tda:
        amb = probe_after_solve(apply_AB, diag_d, omega, X, Y, probe,
                                stats=stats, comm=comm)
        if stats is not None:
            stats['min_eig_amb'] = amb
    return omega, X, Y


def probe_after_solve(apply_AB, diag_d, omega, X, Y, probe, stats=None,
                      comm=None):
    """min eig(A - B) on the block action a Davidson just ran on, after it:
    the converged Lanczos from 1/d for probe=True, or for 'sign' the sign
    alone, proven from the span of the roots' X - Y first
    (`_lowest_amb_local`). Nothing here reaches the roots. A value <= 0
    refuses them (`UnstableReference`): the reference is unstable and the
    Casida omega^2 reduction invalid. Returns the value."""
    sign_only = isinstance(probe, str) and probe.lower() == 'sign'
    # The roots' X - Y, lowest first: nothing here reaches the roots.
    order = np.argsort(omega, kind='stable')
    amb = float(lowest_amb_from_action(
        apply_AB, diag_d, sign_only=sign_only, stats=stats, comm=comm,
        start=(X - Y)[:, order] if sign_only else None)[0])
    if amb <= 0:
        raise unstable_reference(amb)
    return amb


def casida_action(lr_solver, nocc, polarizability, W_aux, isdf_factors,
                   conv_tol, spin='singlet', comm=None, timings=None,
                   tda=False):
    """(apply_AB, diag_d): the Davidson's block action, built once the
    residual floor admits `conv_tol`, its build timed as `davidson_setup`.

    The same action serves the (A-B) probe after the Davidson
    (`solve_bse_isdf`), so Zt and the whole arrays a sliced action gathers are
    built once for both.
    """
    check_residual_floor(conv_tol, bse_pair_diagonal(lr_solver.eps, nocc))
    t0 = time.time()
    apply_AB, diag_d = block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, spin=spin, comm=comm,
                                     tda=tda)
    setup_s = time.time() - t0
    if timings is not None:
        timings['davidson_setup'] = setup_s
        # Every rank arrives here off the same straight-line code, so the
        # gather is the one `run_davidson` already makes, nranks doubles, and
        # it says which rank's row block cost what to build.
        if comm is not None and comm.Get_size() > 1:
            timings['davidson_setup_by_rank'] = _block_action_by_rank(setup_s,
                                                                      comm)
    return apply_AB, diag_d


def with_one_body(apply_AB, one_body):
    """`apply_AB` with an off-diagonal one-body term added to A.

    one_body: ((o, H_oo), (v, H_vv)), occupied and virtual positions (into
    the pair space's occupied and virtual axes) with symmetric blocks whose
    diagonal is zero: A_ia,jb gains delta_ij H_ab - delta_ab H_ij, the
    quasiparticle Hamiltonian's coupling the diagonal eps leaves out. B is
    untouched. None returns `apply_AB` itself.
    """
    if one_body is None:
        return apply_AB
    (o, h_oo), (v, h_vv) = one_body

    def coupled(z):
        Az, Bz = apply_AB(z)
        z3 = np.asarray(z, float).reshape(Az.shape)
        if len(v):
            Az[:, :, v] += z3[:, :, v] @ h_vv
        if len(o):
            Az[:, o, :] -= np.einsum('ij,kja->kia', h_oo, z3[:, o, :])
        return Az, Bz

    coupled.__dict__.update(apply_AB.__dict__)
    return coupled


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


def _correction_length(r, floor):
    """The length a Davidson correction from a residual of norm `r` is sized
    to: kept by real_eig's lindep test while its part outside the subspace
    exceeds DAVIDSON_MIN_NEW_FRACTION of it, and floor / r, the relative
    round-off of that residual."""
    new_min = np.maximum(DAVIDSON_MIN_NEW_FRACTION, floor / r)
    return np.sqrt(DAVIDSON_LINDEP) / new_min


def _tda_davidson_root(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                       guess_factor=DAVIDSON_GUESS_FACTOR, stats=None,
                       counters=None, max_memory=None,
                       preconditioner=DAVIDSON_PRECONDITIONER, guess=None,
                       tracked=None, extra_roots=None):
    """(omega, X, Y, converged, cycles, None, unconfirmed): the `nroots`
    lowest roots of the Tamm-Dancoff problem A X = omega X, Y = 0,
    <X|X> = 1, by the one symmetric Davidson of the code base
    (`Solvers.davidson.solve_symmetric`) on the A half of the block action.

    A is real symmetric: the roots are bounded from above by their Ritz
    values and no (A-B) definiteness enters, so the solve has no breakdown
    regime. The operator is the A half of `apply_AB` on a batch of trial
    vectors, the preconditioner the Jacobi-Davidson one on the diagonal
    `preconditioner` names (`preconditioner_diagonal`), the first space the
    lowest diagonal pairs (`guess_indices`), led by the X columns of
    `guess` (X, Y). The subspace is held within the trial pairs the Casida
    loop's budget gives (`trial_space_memory`). Every decision is dense
    arithmetic on the action's output, which an MPI action locksteps, so
    every rank runs the same iteration.

    The solve converges `tracked_roots` pairs, more than `nroots`: a root
    whose Ritz value starts above the next root up is otherwise never
    corrected and the solve converges past it. After it, an extra pair left
    unconverged with theta - |r| below the highest requested root, where an
    eigenvalue it approximates may lie, marks the requested roots above
    that bound `unconfirmed`. `tracked`, a dict, receives 'omega', 'X' and
    'Y' of every tracked pair. `cycles` counts the batches the operator was
    applied to; the solver's restarts are its own (`collapses` 0).
    """
    no, nv = diag_d.shape
    n_pair = no * nv
    t0 = time.time()
    precond_d = preconditioner_diagonal(apply_AB, diag_d, preconditioner)
    t_precond = time.time() - t0 if precond_d is not diag_d else 0.0
    order = guess_indices(diag_d, nroots, guess_factor, counters=counters)
    space_inc = real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]
    if max_memory is None:
        max_memory = allocation_max_memory_mb()
    working = action_working_bytes(apply_AB, max(len(order), space_inc),
                                    n_pair, False)
    max_space = real_eig_space(
        trial_space_memory(nroots, n_pair, None, max_memory, working),
        nroots, n_pair)[1]
    track = tracked_roots(nroots, n_pair, extra_roots)
    applied = {'s': 0.0, 'calls': 0, 'vectors': 0}

    class _AHalf:
        """A on columns, through the block action."""

        def matmat(self, V):
            t = time.time()
            k = V.shape[1]
            out = apply_AB(np.ascontiguousarray(V.T).reshape(k, no, nv))[0]
            applied['s'] += time.time() - t
            applied['calls'] += 1
            applied['vectors'] += k
            return out.reshape(k, n_pair).T

        def matvec(self, v):
            return self.matmat(np.asarray(v, float).reshape(n_pair, 1))[:, 0]

    x0 = np.zeros((n_pair, len(order)))
    x0[order, np.arange(len(order))] = 1.0
    warm = 0
    if guess is not None:
        gx = np.asarray(guess[0], float).reshape(n_pair, -1)
        warm = gx.shape[1]
        x0 = np.hstack([gx, x0])
    t_loop0 = time.time()
    op = _AHalf()
    omega, X, converged = solve_symmetric(
        op, np.asarray(precond_d, float).ravel(), nroots=track, x0=x0,
        tol_residual=conv_tol, max_cycle=max_cycle, max_subspace=max_space,
        warn_unconverged=False, complete_lowest=False,
        label='Tamm-Dancoff Davidson', max_memory=max_memory)
    unconfirmed = np.zeros(nroots, bool)
    extra = np.flatnonzero(~converged[nroots:]) + nroots
    if extra.size:
        # the extra pairs left unconverged: no eigenvalue they approximate
        # lies below theta - |r|
        r = np.linalg.norm(op.matmat(X[:, extra]) - X[:, extra] * omega[extra],
                           axis=0)
        lower = np.min(omega[extra] - r)
        unconfirmed = omega[:nroots] > lower
    cycles = applied['calls']
    t_loop = time.time() - t_loop0         # the iteration, action included
    if stats is not None:
        stats.update({'davidson_vind_calls': cycles,
                      'davidson_block_actions': applied['vectors'],
                      'davidson_action_s': applied['s'],
                      'davidson_vind_s': applied['s'],
                      'davidson_dense_completion': False})
    if counters is not None:
        counters.update(
            action_s=applied['s'], loop_s=t_loop, precond_s=t_precond,
            warm_start=warm, iterations=cycles, vectors=applied['vectors'],
            space_max=max_space, space_bound=max_space, collapses=0,
            space_bytes=16 * n_pair * max_space,
            space_budget_bytes=(
                DAVIDSON_SPACE_GB * BYTES_PER_GB if max_memory is None
                else DAVIDSON_SPACE_FRACTION * max_memory * 1e6 - working),
            working_bytes=working)
    if tracked is not None:
        tracked.update(omega=omega.copy(), X=X.copy(),
                       Y=np.zeros((n_pair, len(omega))))
    X = X[:, :nroots].copy()
    return (omega[:nroots], X, np.zeros_like(X), converged[:nroots], cycles,
            None, unconfirmed)


def _davidson_counters():
    """Where one rank's Davidson time and traffic go.

    `action_s` is the block action, its collectives included, and `comm_s` the
    part of it spent in them -- the entry lockstep of the trial vectors, the
    head's gather and the reductions -- plus the lockstep of the converged
    result. `loop_s` is the wall of the iteration, so `loop_s - action_s` is
    the Rayleigh-Ritz, orthogonalisation, preconditioning and convergence
    checks, which every rank runs whole. `lockstep_bytes` is what this rank's
    locksteps moved, the volume one rank sends or receives, and
    `lockstep_skipped_bytes` what the checked ones did not move because every
    rank's digest agreed; every rank reports the same two numbers.
    `space_bound` is the trial pairs the loop holds before it collapses its
    subspace (`trial_space_memory`), rank 0's on every rank, and `collapses`
    how often it did. `space_budget_bytes` is what this rank's budget let its
    holders take (the whole holders' DAVIDSON_SPACE_GB without an allocation)
    and `working_bytes` the block action's working set it was sized beside.
    `comm_by_kind` splits `comm_s` by collective (COMM_KINDS), and
    `action_by_piece` the rest of `action_s` by piece of local work
    (ACTION_PIECES), zero for an action that does not time its pieces.
    `space_bytes` is what this rank's trial-space holders take,
    `subspace_by_piece` splits the pair-row loop's subspace work
    (`trial_space.SUBSPACE_PIECES`), zero under Tamm-Dancoff, and
    `trace` is that loop's decisions, one dict per cycle. `precond_s` is
    the preconditioner's diagonal (`preconditioner_diagonal`), formed once
    before the loop. `warm_start` counts the guess vectors the first batch
    kept, and `guess_truncated` whether the unit-vector guess was cut inside
    a degenerate cluster (`guess_indices`).
    """
    return {'action_s': 0.0, 'comm_s': 0.0, 'loop_s': 0.0, 'precond_s': 0.0,
            'warm_start': 0, 'guess_truncated': False,
            'lockstep_bytes': 0, 'lockstep_skipped_bytes': 0, 'vectors': 0,
            'iterations': 0, 'space_max': 0, 'space_bound': 0, 'collapses': 0,
            'space_bytes': 0, 'space_budget_bytes': 0, 'working_bytes': 0,
            'trace': [],
            'comm_by_kind': dict.fromkeys(COMM_KINDS, 0.0),
            'action_by_piece': dict.fromkeys(ACTION_PIECES, 0.0),
            'subspace_by_piece': dict.fromkeys(SUBSPACE_PIECES, 0.0)}


def _block_action_by_rank(value, comm):
    """Every rank's `value` in rank order, through one all-reduce of nranks
    doubles -- a rank-indexed sum of zeros is the gather `mpi_grid` has not
    got, and at this size it costs nothing next to one trial block."""
    buf = np.zeros(comm.Get_size())
    buf[comm.Get_rank()] = value
    reduce_sum(buf, comm)
    return [float(x) for x in buf]


def _pair_symmetry(orbsym, occ, virt):
    """Irrep label of every occupied-virtual pair, for the symmetry-block guess."""
    if orbsym is None:
        return None
    orbsym_d2h = np.asarray(orbsym) % 10
    return (orbsym_d2h[occ][:, None] ^ orbsym_d2h[virt][None, :]).ravel()


def run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                  guess_factor=DAVIDSON_GUESS_FACTOR, stats=None, comm=None,
                  timings=None, refuse_unconverged=False, pair_tile=None,
                  pair_rows=None, max_memory=None,
                  preconditioner=DAVIDSON_PRECONDITIONER, read_roots=None,
                  read_tol=None, guess=None, tda=False, tracked=None,
                  extra_roots=None):
    """The Casida Davidson from a block action (`_davidson_root`, the loop
    `trial_space.casida_rows`), the same iteration on every rank -- or, under
    `tda`, the Hermitian Davidson on its A half (`_tda_davidson_root`),
    replicated on every rank. Over more than one rank the Casida trial space
    is cut by pair rows (`pair_tile` rows a tile, DAVIDSON_PAIR_TILE by
    default; `pair_rows` True cuts it so on any comm, None included);
    serially it is one tile, pyscf's real_eig arithmetic, correcting the
    tracked Ritz pairs a cycle rather than real_eig's twenty.

    Both iterations track `tracked_roots` Ritz pairs, more than `nroots`,
    so a root whose Ritz value starts above the next root up is corrected
    rather than converged past: an extra pair is iterated on until it
    converges or its residual interval clears the highest requested root --
    theta - |r| for the Hermitian A, theta - r_S in the metric
    S = [[A, B], [B, A]] for the Casida problem (`casida_rows`).

    max_memory: MB this rank may hold, which sizes the trial space
    (`trial_space_memory`); None reads the SLURM allocation's share
    (`allocation_max_memory_mb`), and outside one the whole trial space is
    held within DAVIDSON_SPACE_GB.

    preconditioner: the diagonal the corrections divide by
    (DAVIDSON_PRECONDITIONERS, `preconditioner_diagonal`): 'bare', the pair
    energies d, or 'screened', d - (ii|W|aa) from the action's own factors.
    Only the path to the roots moves: A, B and the convergence test are the
    action's.

    A Davidson is a chain of decisions (how many trial vectors the next batch
    carries, which roots have converged, when the space restarts), each read
    off residuals in dense arithmetic, and an iteration replicated over unequal
    action outputs falls apart: the ranks call the batch reduction with buffers
    of different sizes (MPI_ERR_TRUNCATE), or one iterates on alone. The
    distributed action closes that at its source: it locksteps its trial
    vectors at entry and gathers or all-reduces every term it returns
    (`isdf_block_action`), so every rank's iteration reads the same numbers and
    takes the same decisions, with no protocol between them. The roots,
    vectors, flags, cycle count and pair-space record are lockstepped once at
    the end, so every rank returns rank 0's even had its own subspace
    eigensolve, or the dense completion of a solve whose pair space ran out,
    differed in a last bit; ranks whose iterations had fallen apart meet that
    lockstep against another rank's block-action lockstep and raise ValueError
    together instead of deadlocking.

    stats: optional dict, filled with where the time went: the block action
    counted separately from everything around it, which tells a slow action
    from too many iterations.

    timings: optional dict, filled with the same `davidson_*` keys serially and
    under a comm, so a rank count's scaling can be read term by term rather
    than off one total: `davidson_block_action` is what a rank count
    divides, `davidson_comm` the part of it and of the final lockstep spent in
    collectives, `davidson_lockstep_mb` what the locksteps moved and
    `davidson_lockstep_skipped_mb` what their digests proved every rank held
    already (the trial vectors and the result are checked locksteps),
    `davidson_subspace` what every rank runs whole, and `davidson_iterations`
    with `davidson_vectors_applied` the work the guess asked for.
    `davidson_subspace_max` is the largest trial subspace in pairs,
    `davidson_max_space` the bound the loop collapses it at, one number on
    every rank, and `davidson_collapses` how often it did
    (`trial_space_memory`); `davidson_space_budget_mb` is what this rank's
    budget let its holders take and `davidson_action_working_mb` the block
    action's working set it was sized beside.
    `davidson_comm_<kind>` splits `davidson_comm` by collective (COMM_KINDS),
    which says whether the digest, a broadcast or the action's reductions
    carry it. `davidson_action_<piece>` splits the rest of the block action
    by piece of local work (ACTION_PIECES: the GEMMs, the Hadamard product,
    the Hartree term, the owner-order write, the batch's element-wise
    terms), so the work no rank count divides can be told from the work it
    does; zero for an action that does not time its pieces.
    `davidson_block_action_by_rank` (under a comm) lists every rank's action
    time, which is where load imbalance in the row split shows.
    `davidson_subspace_<piece>` splits `davidson_subspace` of the pair-row
    loop (SUBSPACE_PIECES: the tile arithmetic, the replicated small solves,
    the batch and result gathers, the small reductions, the checked
    locksteps), zero under `tda`, and `davidson_trial_space_mb`
    is what this rank's four holders take, 32 bytes per pair row per trial
    pair. `davidson_precond_setup` is the preconditioner's diagonal, zero
    for the bare one.

    refuse_unconverged: False by default: a root left above conv_tol is
    returned as solved with a RuntimeWarning, and `timings` names it
    (`davidson_flags` reads them into a record's flags;
    refuse inputs, warn numerics). True raises
    `DavidsonNotConverged` naming the roots, their residuals and conv_tol, on
    every rank, since every rank holds rank 0's `converged`. Either way
    `timings` carries `davidson_residuals`, every root's |r| recomputed from
    the returned vectors where one is unconverged (None where all
    converged), and `davidson_unconverged` / `davidson_unconfirmed`, each
    root's own two flags.

    read_roots, read_tol: with refuse_unconverged, only the lowest
    `read_roots` roots (None: all) can refuse, and one of them left above
    conv_tol refuses only if its residual, recomputed from one block action
    on the returned vectors, exceeds `read_tol` (None: conv_tol). A caller
    that differentiates one root reads only its eigenvector, whose error is
    its own residual over the gap to the roots outside the trial space;
    the other roots' residuals enter it at second order. `timings` then
    carries `davidson_converged` (every root's flag), `davidson_verdict`
    ('converged', 'read_converged', 'read_within_tol', 'unconverged' or
    'unconfirmed') and `davidson_read_residuals` (the recomputed |r| of the
    read roots, None where none was needed). Under `tda`, a read root that an
    extra Ritz pair still unsettled at the stop may lie below
    (`_tda_davidson_root`) refuses whatever its residual, and warns without
    refuse_unconverged; its flag is False either way.

    guess: (X, Y), (n_pair, k) each, or None. A WARM START: the first trial
    batch is these vectors followed by the cold unit vectors, orthonormalized
    together with their partners (`trial_space.warm_trial_space`) and cut at
    the trial space's bound, so a root the guess does not span is still
    reachable from the unit vectors. Built whole and lockstepped, the batch
    is rank 0's at any rank count; the pair-row loop cuts it into its tiles.
    From the converged vectors of the same problem the solve ends in its
    first cycle. `timings` carries `davidson_warm_start` (the guess vectors
    kept in the first batch, 0 for a cold start). Carrying the tracked pairs
    of `tracked` as well spares the extra pairs' first intervals.

    tracked: optional dict, filled with 'omega', 'X' and 'Y' of the requested
    roots and the tracked Ritz pairs above them, rank 0's on every rank: the
    vectors a warm start of the next geometry carries (`DavidsonWarmStart`).

    extra_roots: the Ritz pairs carried above the nroots asked for (None:
    the `tracked_roots` rule), for a problem whose diagonal is a poorer
    guide to its spectrum than the screened pair diagonal.
    `preconditioner` is checked by the entry points that reach here.
    """
    nranks = comm.Get_size() if comm is not None else 1
    counters = _davidson_counters()
    comm_clock = getattr(apply_AB, 'comm_clock', None)   # the ISDF action's
    clock0 = comm_clock.seconds if comm_clock is not None else 0.0
    kinds0 = dict(comm_clock.by_kind) if comm_clock is not None else None
    piece_clock = getattr(apply_AB, 'piece_clock', None)
    pieces0 = dict(piece_clock.by_piece) if piece_clock is not None else None
    stats0 = lockstep_stats()
    unconfirmed = np.zeros(nroots, bool)
    held = {}
    if tda:
        omega, X, Y, converged, cycles, limit, unconfirmed = (
            _tda_davidson_root(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                               guess_factor, stats, counters,
                               max_memory=max_memory,
                               preconditioner=preconditioner, guess=guess,
                               tracked=held, extra_roots=extra_roots))
    else:
        omega, X, Y, converged, cycles, limit, unconfirmed = _davidson_root(
            apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym, guess_factor,
            stats, counters, comm=comm, pair_tile=pair_tile,
            pair_rows=pair_rows, max_memory=max_memory,
            preconditioner=preconditioner, guess=guess, tracked=held,
            extra_roots=extra_roots)
    if piece_clock is not None:
        for piece in ACTION_PIECES:
            counters['action_by_piece'][piece] = (piece_clock.by_piece[piece]
                                                  - pieces0[piece])
    if nranks > 1:
        t0 = time.time()
        omega, X, Y, converged, cycles, limit, unconfirmed, held = lockstep(
            (omega, X, Y, converged, cycles, limit, unconfirmed, held), comm,
            check=True)
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
    if tracked is not None:
        tracked.update(held)
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
               for kind in COMM_KINDS},
            **{f'davidson_action_{piece}': counters['action_by_piece'][piece]
               for piece in ACTION_PIECES},
            **{f'davidson_subspace_{piece}':
               counters['subspace_by_piece'][piece]
               for piece in SUBSPACE_PIECES},
            'davidson_trial_space_mb': counters['space_bytes'] / 1e6,
            'davidson_space_budget_mb': counters['space_budget_bytes'] / 1e6,
            'davidson_action_working_mb': counters['working_bytes'] / 1e6,
            'davidson_precond_setup': counters['precond_s'],
            'davidson_warm_start': counters['warm_start'],
            'davidson_guess_truncated': counters['guess_truncated']})
        if by_rank is not None:
            timings['davidson_block_action_by_rank'] = by_rank
    # A root that never converged still comes back with an energy attached, and
    # returning it unremarked is how a Davidson result silently stops meaning
    # anything. It does not flag the guess failure DAVIDSON_GUESS_FACTOR
    # describes: that one converges. Every rank refuses or warns alike, since
    # every rank holds rank 0's flags.
    verdict, residuals = 'converged', None
    read = nroots if read_roots is None else min(int(read_roots), nroots)
    if not converged.all():
        stuck = np.flatnonzero(~converged)
        # the loop also stops early, unconverged, when no correction survives
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
        every = _root_residuals(apply_AB, omega, X, Y, diag_d.shape, comm)
        message += (' Residuals: ' + ', '.join(
            f'root {k} |r| = {every[k]:.2e}' for k in stuck)
            + f' against conv_tol = {conv_tol:g}.')
        refuses = bool(refuse_unconverged) and not converged[:read].all()
        if refuses and read_tol is not None:
            residuals = _root_residuals(apply_AB, omega[:read], X[:, :read],
                                        Y[:, :read], diag_d.shape, comm)
            refuses = bool(np.any(residuals > read_tol))
            message += (f' The {read} roots read carry |r| = '
                        f'{np.array2string(residuals, precision=2)} against '
                        f'read_tol = {read_tol:g}.')
        if refuses:
            _record_verdict(timings, converged, 'refused', residuals, every,
                            raw=(converged, unconfirmed))
            refused = [k for k in stuck if k < read]
            raise DavidsonNotConverged(
                f'{message} Refused: the energies and eigenvectors of an '
                'unconverged root are wrong by its residual, and so is every '
                'derivative read off them; refuse_unconverged=False returns '
                f'them for diagnosis. {advice}', refused, every[refused],
                conv_tol if read_tol is None else read_tol)
        verdict = ('unconverged' if not refuse_unconverged
                   else 'read_within_tol' if residuals is not None
                   else 'read_converged')
        warnings.warn(f'{message} {advice}', RuntimeWarning, stacklevel=3)
    else:
        every = None
    if unconfirmed.any():
        # a tracked extra pair unsettled at the stop may stand for an
        # eigenvalue below these roots; small residuals do not rescue them
        message = (f'Davidson could not confirm that roots '
                   f'{np.flatnonzero(unconfirmed).tolist()} of {nroots} are '
                   'the lowest: a tracked extra Ritz pair was still '
                   f'unsettled when the solve stopped at cycle {cycles} of '
                   f'{max_cycle}, and an eigenvalue it approximates may lie '
                   'below them.')
        if refuse_unconverged and unconfirmed[:read].any():
            _record_verdict(timings, converged & ~unconfirmed, 'refused',
                            residuals, every, raw=(converged, unconfirmed))
            raise DavidsonRefusal(f'{message} Refused. Raise max_cycle, or ask '
                                  'for more roots; refuse_unconverged=False '
                                  'returns them for diagnosis.')
        warnings.warn(f'{message} Raise max_cycle, or ask for more roots.',
                      RuntimeWarning, stacklevel=3)
        if converged.all():
            verdict = 'unconfirmed'
    _record_verdict(timings, converged & ~unconfirmed, verdict, residuals,
                    every, raw=(converged, unconfirmed))
    return omega, X, Y


def davidson_flags(timings):
    """{flag: what it says} of one Davidson solve, read off what it recorded
    in `timings` (`_record_verdict`); empty for a converged solve or none
    recorded. 'davidson_not_converged': {roots, residuals (|r| recomputed
    from the returned vectors), verdict}; 'davidson_unconfirmed': {roots}, the
    roots an extra Ritz pair still unsettled at the stop may lie below. What
    a caller copies into its record where the roots were kept as solved.
    """
    if not timings:
        return {}
    flags = {}
    stuck = [k for k, u in enumerate(timings.get('davidson_unconverged')
                                     or ()) if u]
    if stuck:
        every = timings.get('davidson_residuals')
        flags['davidson_not_converged'] = {
            'roots': stuck,
            'residuals': (None if every is None
                          else [float(every[k]) for k in stuck]),
            'verdict': timings.get('davidson_verdict')}
    unsure = [k for k, u in enumerate(timings.get('davidson_unconfirmed')
                                      or ()) if u]
    if unsure:
        flags['davidson_unconfirmed'] = {'roots': unsure}
    return flags


def _record_verdict(timings, converged, verdict, residuals, every=None,
                    raw=None):
    """Every root's flag, the verdict, the read roots' recomputed residuals
    and, where a root is unconverged, every root's into `timings`, where
    there is one; `raw`, (converged, unconfirmed), each root's two flags
    apart as `davidson_unconverged` / `davidson_unconfirmed`."""
    if timings is not None:
        if raw is not None:
            timings.update({
                'davidson_unconverged': [not bool(c) for c in raw[0]],
                'davidson_unconfirmed': [bool(u) for u in raw[1]]})
        timings.update({
            'davidson_converged': [bool(c) for c in converged],
            'davidson_verdict': verdict,
            'davidson_read_residuals': (None if residuals is None
                                        else [float(r) for r in residuals]),
            'davidson_residuals': (None if every is None
                                   else [float(r) for r in every])})


def _root_residuals(apply_AB, omega, X, Y, shape, comm=None):
    """|r| of each Casida root (omega_k, X_k, Y_k): the norm of
    [A X + B Y - omega X, B X + A Y + omega Y], one block action on the X
    and one on the Y, rank 0's on every rank; |A X - omega X| for an
    action of the A half alone (Tamm-Dancoff, Y = 0)."""
    k = len(omega)
    Ax, Bx = apply_AB(X.T.reshape((k,) + shape))
    if Bx is None:
        rx = Ax.reshape(k, -1) - omega[:, None] * X.T
        residuals = np.sqrt(np.sum(rx**2, axis=1))
        if comm is not None and comm.Get_size() > 1:
            residuals = lockstep(residuals, comm)
        return residuals
    Ay, By = apply_AB(Y.T.reshape((k,) + shape))
    rx = (Ax + By).reshape(k, -1) - omega[:, None] * X.T
    ry = (Bx + Ay).reshape(k, -1) + omega[:, None] * Y.T
    residuals = np.sqrt(np.sum(rx**2, axis=1) + np.sum(ry**2, axis=1))
    if comm is not None and comm.Get_size() > 1:
        residuals = lockstep(residuals, comm)
    return residuals


def _davidson_root(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                   guess_factor=DAVIDSON_GUESS_FACTOR, stats=None, counters=None,
                   comm=None, pair_tile=None, pair_rows=None, max_memory=None,
                   preconditioner=DAVIDSON_PRECONDITIONER, guess=None,
                   tracked=None, extra_roots=None):
    """One Casida Davidson (`trial_space.casida_rows`): (omega, X, Y,
    converged, cycles, limit, unconfirmed), the same on every rank of
    `comm`, which reaches the (A-B) probe run when the projected problem
    breaks down, the dense completion when the pair space ran out, and over
    more than one rank (or with `pair_rows` True) the trial space cut by pair
    rows; serially the loop holds one tile, real_eig's arithmetic, and a
    cycle corrects the tracked pairs (`tracked_roots`) alone. `limit` is
    None, or (pairs held, corrections asked, whether the solve was completed
    densely) at the last cycle that ran out; `unconfirmed` marks the roots
    above the residual interval of a tracked extra Ritz pair
    (`tracked_roots`) still unsettled where the solve stopped. `tracked`, a
    dict, receives 'omega', 'X' and 'Y' of the requested roots and the
    tracked pairs above them. `max_memory`, `preconditioner` and `guess` are
    `run_davidson`'s."""
    no, nv = diag_d.shape
    n_pair = no * nv
    nranks = comm.Get_size() if comm is not None else 1
    t0 = time.time()
    precond_d = preconditioner_diagonal(apply_AB, diag_d, preconditioner)
    t_precond = time.time() - t0 if precond_d is not diag_d else 0.0
    split = nranks > 1 if pair_rows is None else pair_rows
    rows = (PairRows(n_pair, comm, pair_tile) if split
            else PairRows(n_pair, None, n_pair))
    trace = []                       # the loop's decisions, per cycle
    ncall = [0]
    nvec = [0]
    t_action = [0.0]
    t_total = [0.0]
    order = guess_indices(diag_d, nroots, guess_factor, counters=counters)
    # A cycle corrects the tracked Ritz pairs alone, not real_eig's
    # max(nroots, MAX_SPACE_INC = 20) lowest: a correction of a pair above
    # them costs a block action and settles nothing the solve reads.
    # benzene/cc-pVDZ BSE@RHF-DF, conv_tol 1e-8, screened: 2 roots 672 ->
    # 156 vectors applied, 5 roots 704 -> 242, 12 roots 604 -> 568, cycles
    # 19-21 either way; the fragment partition's full BSE (320 pairs, 2
    # roots, bare diagonal, 4 extra pairs) 642 -> 360, where 20 corrections
    # a cycle had filled its whole pair space.
    ntrack = tracked_roots(nroots, n_pair, extra_roots)
    if max_memory is None:
        max_memory = allocation_max_memory_mb()
    held_rows = rows.rows[1] - rows.rows[0]
    working = action_working_bytes(apply_AB, max(len(order), ntrack),
                                    n_pair, not split)
    eig_memory = trial_space_memory(nroots, n_pair, held_rows, max_memory,
                                     working)
    # The bound is a decision: every rank restarts at the same cycle only if
    # it holds the same bound, so rank 0's (the rank of the most pair rows
    # and kernel rows, whose budget binds first) is every rank's.
    if nranks > 1:
        eig_memory = broadcast(eig_memory, comm)
    max_space = real_eig_space(eig_memory, nroots, n_pair)[1]
    space = [0]                      # trial pairs held at the last cycle
    batch_max = [0]                  # the widest batch the action was handed
    # The pair space can run out before the tolerance does. The loop adds each
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

    def vind_rows(xys):
        """The whole batch [V, W] in, this rank's rows of U1 = A V + B W and
        U2 = A W + B V out."""
        t_enter = time.time()
        xys = np.asarray(xys).reshape(-1, 2, no, nv)
        n = xys.shape[0]
        ncall[0] += 1
        nvec[0] += 2 * n                     # a block action per X and per Y
        batch_max[0] = max(batch_max[0], n)
        t0 = time.time()
        Ax, Bx = apply_AB(xys[:, 0])
        Ay, By = apply_AB(xys[:, 1])
        t_action[0] += time.time() - t0
        p0, p1 = rows.rows
        U1 = Ax.reshape(n, -1)[:, p0:p1] + By.reshape(n, -1)[:, p0:p1]
        U2 = Bx.reshape(n, -1)[:, p0:p1] + Ay.reshape(n, -1)[:, p0:p1]
        t_total[0] += time.time() - t_enter
        return U1, U2

    floor = _residual_floor(diag_d)
    # hdiag's rows of each tile: the tile's X rows, then its Y rows
    d_flat = precond_d.ravel()
    hdiag_tiles = [np.hstack([d_flat[p0:p1], -d_flat[p0:p1]])
                   for p0, p1 in rows.tiles]

    def precond_rows(dx, e, size_all=False):
        """r / (d - omega) on this rank's tiles, (k, 2 tile rows) blocks of
        the residuals in and of the corrections out; the norms that size them
        are the whole vectors', reduced over the ranks. `size_all` sizes
        every correction, whatever its residual (`casida_rows`' resize)."""
        e = np.atleast_1d(e)
        space[0] = trace[-1]['m1']
        if len(e) > n_pair - space[0]:
            limit[0] = (space[0], len(e))
        t = [correction(block, h, e) for h, block in zip(hdiag_tiles, dx)]
        # The length of a correction is its acceptance test: the loop keeps
        # one while its squared norm outside the subspace exceeds lindep, read
        # before normalizing, so at the raw length r / (d - omega) |r| stalls
        # near 1e-7 whatever conv_tol asks (2.4e-7 at naphthalene/cc-pVDZ).
        # Below DAVIDSON_SIZED_RESIDUAL a correction is sized so that it is
        # kept while its part outside the subspace, as a fraction of itself,
        # exceeds DAVIDSON_MIN_NEW_FRACTION, which keeps the subspace
        # conditioned, and floor / |r|, the relative round-off of the residual
        # it came from. A power of two scales exactly.
        r = rows.norms([block.reshape(len(e), -1) for block in dx], len(e))
        sized = (r < DAVIDSON_SIZED_RESIDUAL) | bool(size_all)
        if sized.any():
            length = _correction_length(r[sized], floor)
            raw = rows.norms([b[sized] for b in t], int(sized.sum()))
            scale = np.exp2(np.round(np.log2(length / raw)))[:, None]
            for b in t:
                b[sized] *= scale
        return [b.reshape(block.shape) for b, block in zip(t, dx)]

    x0 = np.zeros((len(order), 2 * n_pair))
    x0[np.arange(len(order)), order] = 1.0
    x0sym = x_sym[order] if x_sym is not None else None
    warm = 0
    if guess is not None:
        if x_sym is not None:
            raise ValueError(
                'a warm-started Davidson takes no pair symmetry: the guess '
                'vectors carry no irrep label, and the symmetry-blocked '
                'subspace solve reads one per trial vector')
        gx, gy = (np.asarray(g, float).reshape(n_pair, -1) for g in guess)
        x0 = warm_trial_space(gx, gy, x0, DAVIDSON_LINDEP)[:max_space]
        if nranks > 1:
            x0 = lockstep(x0, comm, check=True)
        warm = min(gx.shape[1], len(x0))

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
    lower = np.inf
    try:
        converged, e, xy, lower = casida_rows(
            vind_rows, x0, precond_rows, rows, tol_residual=conv_tol,
            nroots=nroots, ntrack=ntrack, x0sym=x0sym,
            max_cycle=max_cycle, space_inc=ntrack, max_space=max_space,
            verbose=logger.Logger(sys.stdout, 0),
            trace=trace,
            resize=lambda dx, e: precond_rows(dx, e, size_all=True))
    except np.linalg.LinAlgError as exc:
        # The projected (A-B) block is Cholesky-factorized, so this is the
        # response subspace losing positive definiteness -- the instability
        # regime of the mean-field reference (long acenes reach it near a11),
        # where the omega^2 reduction is invalid rather than merely
        # ill-conditioned; the S-metric residual interval of a tracked pair
        # fails the same way. No shift or retry inside this solver gives
        # physical roots there: the dense CasidaSolver refuses the same
        # problem. Stabilize the reference, or solve the non-Hermitian problem
        # directly. A subspace that ran out of pair space breaks down here too
        # (water/6-31G BSE@HF, 4 roots at 1e-5, after the partner test kept 3
        # pairs for the last one left), and the dense completion tells the two
        # apart.
        if not completes():
            raise _casida_breakdown(apply_AB, diag_d, conv_tol, comm) from exc
        converged = None
    # the loop's own cycles and space: a restart applies no action, and a
    # residual interval is a block action of no cycle
    cycles = len(trace)
    space_max = max([step['m1'] for step in trace], default=0)
    collapses = sum(step['fresh'] for step in trace[1:])
    dense = None
    if completes() and (converged is None or not np.all(converged)):
        try:
            dense = _dense_casida(action, diag_d, nroots, conv_tol,
                                  batch_max[0])
        except CasidaInstability as exc:
            raise UnstableReference(f'completed densely: {exc}', None) from exc
        ncall[0] += 1
        cycles += 1
        nvec[0] += n_pair
        space_max = n_pair
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
        counters['precond_s'] = t_precond
        counters['warm_start'] = warm
        counters['iterations'] = cycles
        counters['vectors'] = nvec[0]
        counters['space_max'] = min(space_max, n_pair)
        counters['space_bound'] = max_space
        counters['collapses'] = collapses
        counters['space_bytes'] = rows.held_bytes
        counters['space_budget_bytes'] = (
            DAVIDSON_SPACE_GB * BYTES_PER_GB if max_memory is None
            else DAVIDSON_SPACE_FRACTION * max_memory * 1e6 - working)
        counters['working_bytes'] = working
        counters['subspace_by_piece'] = dict(rows.clock)
        counters['trace'] = trace
    if limit[0] is not None:
        limit[0] = (int(limit[0][0]), int(limit[0][1]), dense is not None)
    if dense is not None:
        omega, X, Y, converged = dense
        if tracked is not None:
            tracked.update(omega=omega, X=X, Y=Y)
        return (omega, X, Y, converged, cycles, limit[0],
                np.zeros(nroots, bool))
    omega = np.asarray(e)
    X = np.zeros((n_pair, len(omega)))
    Y = np.zeros((n_pair, len(omega)))
    for k, z in enumerate(xy):
        x, y = z.reshape(2, no, nv)
        norm = np.sqrt(abs(np.sum(x**2) - np.sum(y**2)))
        X[:, k] = (x / norm).ravel()
        Y[:, k] = (y / norm).ravel()
    if tracked is not None:
        tracked.update(omega=omega, X=X, Y=Y)
    unconfirmed = omega[:nroots] > lower
    return (omega[:nroots].copy(), X[:, :nroots].copy(),
            Y[:, :nroots].copy(), np.asarray(converged), cycles, limit[0],
            unconfirmed)


def _casida_breakdown(apply_AB, diag_d, conv_tol, comm=None):
    """The `CasidaBreakdown` for a projected (A-B) block that is not positive
    definite, telling a stable reference's numerical breakdown from an
    unstable reference (`UnstableReference`, min eig(A-B) <= 0 measured) by
    the (A-B) probe, run cold over `comm`. Only the probe's own failures --
    ARPACK or the distributed Lanczos out of iterations, or ranks whose
    iterations fell apart meeting at the lockstep (ValueError) -- become the
    "did not converge" breakdown; any other exception is a defect and raises
    as itself rather than being reported as an instability."""
    try:
        amb = float(lowest_amb_from_action(apply_AB, diag_d, comm=comm)[0])
        detail = f'measured min eig(A-B) = {amb:.6f} Ha'
    except (ArpackNoConvergence, AmbProbeNotConverged, ValueError):
        amb = None
        detail = ('min eig(A-B) probe did not converge; smallest '
                  f'orbital-energy difference {diag_d.min():.4f} Ha')
    # A congruence of a positive definite (A-B) is positive definite, so
    # with the reference stable it is the subspace that failed: the loop's
    # projected problem loses accuracy as |r| nears round-off.
    if amb is not None and amb > 0:
        return CasidaBreakdown(
            'Casida-form Davidson broke down numerically: the projected '
            '(A-B) block is not positive definite although (A-B) itself '
            f'is ({detail}), so the reference is stable and the trial '
            'subspace is what failed, as it does when conv_tol '
            f'({conv_tol:g} Ha) asks for a residual approaching round-off '
            f'({_residual_floor(diag_d):.1e} Ha on this pair space). '
            'Loosen conv_tol.', amb)
    if amb is not None:
        return unstable_reference(amb)
    return CasidaBreakdown(
        'Casida-form Davidson failed: the projected (A-B) block is not '
        'positive definite, the signature of a singlet/triplet '
        f'instability of the mean-field reference ({detail}). The '
        'omega^2 reduction is invalid in this regime and shifted or '
        'retried roots would not be physical. Check first that the SCF is '
        'the LOWEST solution (vary the initial guess, follow '
        'instabilities) -- a converged non-minimum reference produces '
        'exactly this failure.', amb)


def unstable_reference(amb, when='before the solve returned its roots'):
    """The `UnstableReference` an unstable reference is refused with,
    min eig(A-B) = `amb` <= 0, `when` (before any root is returned)."""
    return UnstableReference(
        f'BSE refused {when}: min eig(A-B) = '
        f'{amb:.6f} Ha <= 0 -- the mean-field reference is singlet/triplet '
        'unstable and the Casida omega^2 reduction is invalid there. Check '
        'first that the SCF is the LOWEST solution (vary the initial guess, '
        'follow instabilities); if it is, stabilize the reference or solve '
        'the non-Hermitian problem -- no shift gives physical roots in this '
        'regime.', amb)


def _dense_casida(apply_AB, diag_d, nroots, conv_tol, batch):
    """(omega, X, Y, converged): the `nroots` lowest Casida roots from A and B
    built whole out of the block action, one unit vector per pair.

    The completion of a Davidson whose pair space ran out (`_davidson_root`):
    the subspace and its remainder are then the whole space, so its dense
    problem is the one left to solve, and at water/6-31G it costs 40 actions
    after the Davidson's 68. The unit vectors go through the action in batches
    of `batch`, the widest trial batch the Davidson already handed it, and
    are lockstepped at its entry like any other, so every rank builds the same
    A and B. The roots are the one dense Casida eigensolve's
    (`casida.casida_eigensolve`, A - B factored, on every rank whole), which
    raises `CasidaInstability` where the reference is unstable.
    `converged` is each root's Casida residual against conv_tol, as the
    Davidson's is.
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
    omega, X, Y, _, _ = casida_eigensolve([A + B, A - B], 'amb', lowest=nroots,
                                          threshold=np.inf)
    r = np.sqrt(np.sum((A @ X + B @ Y - X * omega)**2, axis=0)
                + np.sum((B @ X + A @ Y + Y * omega)**2, axis=0))
    return omega, X, Y, r <= conv_tol
