"""Matrix-free Casida/RPA/BSE Davidson, with two ways of applying the same A/B.

`solve_bse_isdf` and `solve_bse_df` are the two drivers on them (mean
field in, excitations out) and share every convention, so they differ
only in how the interaction is represented. The block actions are in
`block_actions`, the (A-B) probe in `probe`, the Tamm-Dancoff loop in
`davidson_tda` and what both loops share in `davidson_shared`; the Casida
loop and the drivers stay here.

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
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import ArpackNoConvergence

from src.Base.constants import (BSE_DENSE_MAX_NOV,
                                DAVIDSON_FLOOR_EPS_MULTIPLE,
                                DAVIDSON_GUESS_FACTOR, DAVIDSON_LINDEP,
                                DAVIDSON_MIN_NEW_FRACTION,
                                DAVIDSON_PRECONDITIONER,
                                DAVIDSON_SIZED_RESIDUAL,
                                DAVIDSON_SPACE_FRACTION, DAVIDSON_SPACE_GB,
                                DAVIDSON_WARM_START_MAX_SHIFT,
                                HARTREE_TO_EV, SEPARABLE_AUX_GAUGE)
from src.Base.numerics import numerics_or_default
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies,
                                      require_closed_shell_or_unrestricted)
from src.Base.sliced_factors import SlicedFactors
from src.Base.step_failed import DavidsonNotConverged, DavidsonRefusal, StepFailed
from src.Base.utils.memory import allocation_max_memory_mb
from src.Base.utils.mpi_grid import (broadcast, current_comm, grid_comm,
                                     lockstep, lockstep_stats, reduce_sum,
                                     replicate)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.evGW import evgw_eigenvalues, shifted_mean_field
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import (_unpack_factors,
                                               replicate_factors,
                                               replicate_mean_field,
                                               separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.LinearResponse.block_actions import (
    ACTION_PIECES, COMM_KINDS, _block_action, bse_pair_diagonal)
from src.SingleReference.LinearResponse.casida import CasidaInstability
from src.SingleReference.LinearResponse.davidson_shared import (
    _action_working_bytes, _check_preconditioner, _correction, _guess_indices,
    _preconditioner_diagonal, _real_eig_space, _tracked_roots,
    _trial_space_memory)
from src.SingleReference.LinearResponse.davidson_tda import _tda_davidson_root
from src.SingleReference.LinearResponse.exciton_descriptors import exciton_descriptors
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, check_normalization)
from src.SingleReference.LinearResponse.probe import (
    AmbProbeNotConverged, _lowest_amb_from_action, lowest_amb_eigenvalue)
from src.SingleReference.LinearResponse.space_time import (
    static_screening_grid, static_screening_matrix)
from src.SingleReference.LinearResponse.trial_space import (SUBSPACE_PIECES,
                                                            PairRows,
                                                            casida_rows,
                                                            warm_trial_space)


#: `GW/space_time.py`'s own `timings` keys, renamed for `info['timings']`'s
#: `qp_` prefix: `t_qp_static`/`t_qp_states` fold their `t_qp_` in with the
#: shared prefix rather than doubling it. `t_qp` -- `_finish_qp`'s own total,
#: static + states -- carries nothing beyond the two keys it sums and is not
#: re-exposed under a third name.
QP_TIMING_RENAME = {'t_chi0': 'qp_chi0', 't_dyson': 'qp_dyson',
                    't_sigma': 'qp_sigma', 't_qp_static': 'qp_static',
                    't_qp_states': 'qp_states'}


class _QPStageTimings:
    """`timings=` target for the space-time QP solve, G0W0 or one evGW cycle.

    `solve_qp_energy_space_time`/`_qp_blocked`/`_finish_qp` write each
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


class _CasidaBreakdown(StepFailed):
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


def refuse_second_numerics(gw_kwargs):
    """Refuse a `Numerics` inside `gw_kwargs`: an entry point holds ONE and
    hands it to its GW stage itself, so a second would be two answers to
    which grid the self-energy ran on."""
    if gw_kwargs and 'numerics' in gw_kwargs:
        raise TypeError('gw_kwargs holds a numerics of its own; pass the one '
                        "Numerics as the entry point's `numerics`, which its "
                        'GW stage reads too')


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
    Iteration by our Casida Davidson (`_run_davidson`); the A/B matrix-free action is ours,
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
    refuse_unconverged: False by default: a root left above conv_tol is
    returned as solved with a RuntimeWarning and flagged in `timings`
    (`davidson_flags`), which the caller copies into its record
    (refuse inputs, warn numerics).
    True raises `DavidsonNotConverged`, since its energy and eigenvector are
    wrong by its residual.
    read_roots, read_tol: with refuse_unconverged, the lowest `read_roots`
    roots are the ones the caller differentiates and `read_tol` the residual
    it needs of them; `_run_davidson`'s.
    max_memory: MB this rank may hold, which sizes the trial space;
    `_run_davidson`'s.
    preconditioner: 'bare' (d) or 'screened' (d - (ii|W|aa)), the diagonal
    the corrections divide by; `_run_davidson`'s.
    guess: (X, Y), (n_pair, k) each -- a previous geometry's roots carried
    into these orbitals (`DavidsonWarmStart.guess`) -- started from with
    the unit vectors; None starts from the unit vectors alone.
    `_run_davidson`'s.
    tda: the Tamm-Dancoff problem A X = omega X with B = 0, solved by a
    Hermitian Davidson on the A half of the same block action
    (`_tda_davidson_root`); Y is returned zero. Its roots lie at or above
    the full ones for a stable reference.
    one_body: the off-diagonal one-body blocks a quasiparticle Hamiltonian
    adds to A (`with_one_body`), or None for the diagonal eps alone.
    tracked: optional dict, filled with the requested roots and the tracked
    Ritz pairs above them (`_run_davidson`'s), for a warm start to carry.
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
    _check_preconditioner(preconditioner)
    if tda and orbsym is not None:
        raise ValueError('the Tamm-Dancoff Davidson takes no pair symmetry: '
                         'its trial space is one Hermitian block')
    comm = current_comm() if comm is None else comm
    apply_AB, diag_d = _casida_action(lr_solver, nocc, polarizability, W_aux,
                                      isdf_factors, conv_tol, spin=spin,
                                      comm=comm, timings=timings, tda=tda)
    apply_AB = with_one_body(apply_AB, one_body)
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    omega, X, Y = _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle,
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
    amb = float(_lowest_amb_from_action(
        apply_AB, diag_d, sign_only=sign_only, stats=stats, comm=comm,
        start=(X - Y)[:, order] if sign_only else None)[0])
    if amb <= 0:
        raise _unstable_reference(amb)
    return amb


def _casida_action(lr_solver, nocc, polarizability, W_aux, isdf_factors,
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
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, spin=spin, comm=comm,
                                     tda=tda)
    setup_s = time.time() - t0
    if timings is not None:
        timings['davidson_setup'] = setup_s
        # Every rank arrives here off the same straight-line code, so the
        # gather is the one `_run_davidson` already makes, nranks doubles, and
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
    subspace (`_trial_space_memory`), rank 0's on every rank, and `collapses`
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
    the preconditioner's diagonal (`_preconditioner_diagonal`), formed once
    before the loop. `warm_start` counts the guess vectors the first batch
    kept, and `guess_truncated` whether the unit-vector guess was cut inside
    a degenerate cluster (`_guess_indices`).
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


def solve_bse_isdf(mf, mol, nocc, numerics=None, qp='G0W0', factors=None,
                   auxbasis=None, radii=None, counts=None, probe=False,
                   gw_kwargs=None, progress=None, n_start=1,
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
    the Lanczos from 1/d. 'sign' stops once the sign is proven by the
    residual bound: first by Rayleigh-Ritz on the span of the roots' X - Y
    (which proves an eigenvalue in that span, not the minimum over pair
    irreps no root lies in), then by a Lanczos from the lowest root with
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
    pair rows a rank holds (`_trial_space_memory`); outside one within
    DAVIDSON_SPACE_GB. On a slow interconnect weigh the GW stage's
    all-reduce of nfreq x naux^2 first: on 1 GbE it costs more than it saves
    (`Base.utils.mpi_grid`).
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
    (`_run_davidson`).

    guess: (X, Y) of this spin to warm-start the Davidson from, carried into
    this mean field's orbitals (`DavidsonWarmStart.guess`); `_run_davidson`'s.

    tda: the Tamm-Dancoff BSE, A X = omega X with B = 0 and Y returned zero,
    on the A half of the same block action (`solve_casida_davidson`);
    `probe` is skipped, as on the dense route, since that problem makes no
    omega^2 reduction.

    refuse_unconverged: False (the default) returns a root left above
    conv_tol as solved, warned about and named in info['davidson_flags']
    (`davidson_flags`); True raises `DavidsonNotConverged`.
    `_run_davidson`'s.

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
    (`_run_davidson`); the `qp` stage, when the space-time solve runs,
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
    _check_preconditioner(preconditioner)
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
        replicate_mean_field(mf, mpi_comm)
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0
    progress = bool(progress) and rank == 0        # lockstep ranks, one voice
    # the GW stage and the static W read the one object
    refuse_second_numerics(gw_kwargs)
    gw_kw = dict(gw_kwargs or {}, numerics=numerics)
    if nranks > 1:
        gw_kw.update(distribute=True, comm=mpi_comm)

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
            # An array that is a self-consistent spectrum: W follows it, the
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
    apply_AB, diag_d = _casida_action(lr, nocc, 'BSE', W_aux, factors,
                                      conv_tol, spin=spin, comm=action_comm,
                                      timings=t, tda=tda)
    try:
        omega, X, Y = _run_davidson(apply_AB, diag_d, nroots, conv_tol,
                                    max_cycle, None, guess_factor, stats=stats,
                                    comm=action_comm, timings=t,
                                    max_memory=_job_max_memory(mf),
                                    preconditioner=preconditioner,
                                    guess=guess, tda=tda, tracked=tracked,
                                    refuse_unconverged=refuse_unconverged)
    except _CasidaBreakdown as exc:
        # The projected (A-B) broke down and the measurement it ran says the
        # reference, not the subspace, is what failed; named with or without
        # `probe`, since the breakdown has already paid for it.
        if exc.min_eig_amb is not None and exc.min_eig_amb <= 0:
            raise _unstable_reference(exc.min_eig_amb) from exc
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
    _check_preconditioner(preconditioner)
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
    # Opt-in: before the solve, refusing an unstable reference up front.
    if probe and not tda:
        t0 = _begin('probe')
        amb = float(lowest_amb_eigenvalue(
            lr, nocc, polarizability='BSE', W_aux=W_aux,
            sign_only=(isinstance(probe, str) and probe.lower() == 'sign'),
            stats=stats)[0])
        _end('probe', t0)
        if amb <= 0:
            raise UnstableReference(
                f'BSE refused before the solve: min eig(A-B) = {amb:.6f} Ha '
                '<= 0 -- the mean-field reference is singlet/triplet unstable '
                'and the Casida omega^2 reduction is invalid there.', amb)

    t0 = _begin('davidson')
    try:
        omega, X, Y = solve_casida_davidson(
            lr, nocc, numerics=numerics, polarizability='BSE', W_aux=W_aux,
            stats=stats, spin=spin, timings=t,
            max_memory=_job_max_memory(mf),
            tda=tda, tracked=tracked, refuse_unconverged=refuse_unconverged)
    except _CasidaBreakdown as exc:
        # As on the ISDF twin: the breakdown's own measurement names an
        # unstable reference whether or not `probe` was asked for.
        if exc.min_eig_amb is not None and exc.min_eig_amb <= 0:
            raise _unstable_reference(exc.min_eig_amb) from exc
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
                     screening='imaginary-time', numerics=None, n_start=1,
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
        replicate_mean_field(mf, mpi_comm)
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

    q = numerics_or_default(numerics).quadrature
    grid = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                 tau_target=q.tau_target)
    W_aux = static_screening_matrix(X_mo, D, eps, nocc, grid, comm=mpi_comm)
    return X_mo, (None if sliced else D), W_aux


def _pair_symmetry(orbsym, occ, virt):
    """Irrep label of every occupied-virtual pair, for the symmetry-block guess."""
    if orbsym is None:
        return None
    orbsym_d2h = np.asarray(orbsym) % 10
    return (orbsym_d2h[occ][:, None] ^ orbsym_d2h[virt][None, :]).ravel()


def _job_max_memory(mf):
    """MB this rank's job gave it -- the mean field's max_memory, which a job
    sets from its allocation's share (`allocation_max_memory_mb`) -- inside a
    SLURM allocation; None outside one, where max_memory is pyscf's process
    default and says nothing of the memory the machine has."""
    return (float(mf.max_memory) if allocation_max_memory_mb() is not None
            else None)


def _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                  guess_factor=DAVIDSON_GUESS_FACTOR, stats=None, comm=None,
                  timings=None, refuse_unconverged=False, pair_tile=None,
                  pair_rows=None, max_memory=None,
                  preconditioner=DAVIDSON_PRECONDITIONER, read_roots=None,
                  read_tol=None, guess=None, tda=False, tracked=None):
    """The Casida Davidson from a block action (`_davidson_root`, the loop
    `trial_space.casida_rows`), the same iteration on every rank -- or, under
    `tda`, the Hermitian Davidson on its A half (`_tda_davidson_root`),
    replicated on every rank. Over more than one rank the Casida trial space
    is cut by pair rows (`pair_tile` rows a tile, DAVIDSON_PAIR_TILE by
    default; `pair_rows` True cuts it so on any comm, None included);
    serially it is one tile, pyscf's real_eig iteration.

    Both iterations track `_tracked_roots` Ritz pairs, more than `nroots`,
    so a root whose Ritz value starts above the next root up is corrected
    rather than converged past: an extra pair is iterated on until it
    converges or its residual interval clears the highest requested root --
    theta - |r| for the Hermitian A, theta - r_S in the metric
    S = [[A, B], [B, A]] for the Casida problem (`casida_rows`).

    max_memory: MB this rank may hold, which sizes the trial space
    (`_trial_space_memory`); None reads the SLURM allocation's share
    (`allocation_max_memory_mb`), and outside one the whole trial space is
    held within DAVIDSON_SPACE_GB.

    preconditioner: the diagonal the corrections divide by
    (DAVIDSON_PRECONDITIONERS, `_preconditioner_diagonal`): 'bare', the pair
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
    (`_trial_space_memory`); `davidson_space_budget_mb` is what this rank's
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
    """
    _check_preconditioner(preconditioner)
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
                               tracked=held))
    else:
        omega, X, Y, converged, cycles, limit, unconfirmed = _davidson_root(
            apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym, guess_factor,
            stats, counters, comm=comm, pair_tile=pair_tile,
            pair_rows=pair_rows, max_memory=max_memory,
            preconditioner=preconditioner, guess=guess, tracked=held)
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
                   tracked=None):
    """One Casida Davidson (`trial_space.casida_rows`): (omega, X, Y,
    converged, cycles, limit, unconfirmed), the same on every rank of
    `comm`, which reaches the (A-B) probe run when the projected problem
    breaks down, the dense completion when the pair space ran out, and over
    more than one rank (or with `pair_rows` True) the trial space cut by pair
    rows; serially the loop holds one tile, real_eig's iteration. `limit` is
    None, or (pairs held, corrections asked, whether the solve was completed
    densely) at the last cycle that ran out; `unconfirmed` marks the roots
    above the residual interval of a tracked extra Ritz pair
    (`_tracked_roots`) still unsettled where the solve stopped. `tracked`, a
    dict, receives 'omega', 'X' and 'Y' of the requested roots and the
    tracked pairs above them. `max_memory`, `preconditioner` and `guess` are
    `_run_davidson`'s."""
    no, nv = diag_d.shape
    n_pair = no * nv
    nranks = comm.Get_size() if comm is not None else 1
    t0 = time.time()
    precond_d = _preconditioner_diagonal(apply_AB, diag_d, preconditioner)
    t_precond = time.time() - t0 if precond_d is not diag_d else 0.0
    split = nranks > 1 if pair_rows is None else pair_rows
    rows = (PairRows(n_pair, comm, pair_tile) if split
            else PairRows(n_pair, None, n_pair))
    trace = []                       # the loop's decisions, per cycle
    ncall = [0]
    nvec = [0]
    t_action = [0.0]
    t_total = [0.0]
    order = _guess_indices(diag_d, nroots, guess_factor, counters=counters)
    space_inc = _real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]
    if max_memory is None:
        max_memory = allocation_max_memory_mb()
    held_rows = rows.rows[1] - rows.rows[0]
    working = _action_working_bytes(apply_AB, max(len(order), space_inc),
                                    n_pair, not split)
    eig_memory = _trial_space_memory(nroots, n_pair, held_rows, max_memory,
                                     working)
    # The bound is a decision: every rank restarts at the same cycle only if
    # it holds the same bound, so rank 0's (the rank of the most pair rows
    # and kernel rows, whose budget binds first) is every rank's.
    if nranks > 1:
        eig_memory = broadcast(eig_memory, comm)
    max_space = _real_eig_space(eig_memory, nroots, n_pair)[1]
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
        t = [_correction(block, h, e) for h, block in zip(hdiag_tiles, dx)]
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
            nroots=nroots, ntrack=_tracked_roots(nroots, n_pair), x0sym=x0sym,
            max_cycle=max_cycle, space_inc=space_inc, max_space=max_space,
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
        except np.linalg.LinAlgError as exc:
            raise _casida_breakdown(apply_AB, diag_d, conv_tol, comm) from exc
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
            DAVIDSON_SPACE_GB * 1e9 if max_memory is None
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
    """The `_CasidaBreakdown` for a projected (A-B) block that is not positive
    definite, telling a stable reference's numerical breakdown from an
    unstable reference (`UnstableReference`, min eig(A-B) <= 0 measured) by
    the (A-B) probe, run cold over `comm`. Only the probe's own failures --
    ARPACK or the distributed Lanczos out of iterations, or ranks whose
    iterations fell apart meeting at the lockstep (ValueError) -- become the
    "did not converge" breakdown; any other exception is a defect and raises
    as itself rather than being reported as an instability."""
    try:
        amb = float(_lowest_amb_from_action(apply_AB, diag_d, comm=comm)[0])
        detail = f'measured min eig(A-B) = {amb:.6f} Ha'
    except (ArpackNoConvergence, AmbProbeNotConverged, ValueError):
        amb = None
        detail = ('min eig(A-B) probe did not converge; smallest '
                  f'orbital-energy difference {diag_d.min():.4f} Ha')
    # A congruence of a positive definite (A-B) is positive definite, so
    # with the reference stable it is the subspace that failed: the loop's
    # projected problem loses accuracy as |r| nears round-off.
    if amb is not None and amb > 0:
        return _CasidaBreakdown(
            'Casida-form Davidson broke down numerically: the projected '
            '(A-B) block is not positive definite although (A-B) itself '
            f'is ({detail}), so the reference is stable and the trial '
            'subspace is what failed, as it does when conv_tol '
            f'({conv_tol:g} Ha) asks for a residual approaching round-off '
            f'({_residual_floor(diag_d):.1e} Ha on this pair space). '
            'Loosen conv_tol.', amb)
    if amb is not None:
        return _unstable_reference(amb)
    return _CasidaBreakdown(
        'Casida-form Davidson failed: the projected (A-B) block is not '
        'positive definite, the signature of a singlet/triplet '
        f'instability of the mean-field reference ({detail}). The '
        'omega^2 reduction is invalid in this regime and shifted or '
        'retried roots would not be physical. Check first that the SCF is '
        'the LOWEST solution (vary the initial guess, follow '
        'instabilities) -- a converged non-minimum reference produces '
        'exactly this failure.', amb)


def _unstable_reference(amb):
    """The `UnstableReference` an unstable reference is refused with,
    min eig(A-B) = `amb` <= 0, before any root is returned."""
    return UnstableReference(
        f'BSE refused before the solve returned its roots: min eig(A-B) = '
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
    try:
        L = np.linalg.cholesky(A - B)
    except np.linalg.LinAlgError as exc:
        raise UnstableReference(
            'Casida problem completed densely: (A - B) is not positive '
            'definite, the mean-field reference is unstable and the omega^2 '
            'reduction is invalid there.', None) from exc
    w2, Z = np.linalg.eigh(L.T @ (A + B) @ L)
    if w2[0] <= 0.0:
        # (A - B) positive definite, (A + B) not: the other instability.
        raise UnstableReference(
            f'Casida problem completed densely has omega^2 = {w2[0]:.3e} '
            '<= 0 Ha^2: the mean-field reference is unstable and the omega^2 '
            'reduction is invalid there.', None)
    omega = np.sqrt(w2[:nroots])
    Z = Z[:, :nroots]
    xpy = (L @ Z) / np.sqrt(omega)[None, :]
    xmy = solve_triangular(L, Z, lower=True, trans='T') * np.sqrt(omega)[None, :]
    X = 0.5 * (xpy + xmy)
    Y = 0.5 * (xpy - xmy)
    r = np.sqrt(np.sum((A @ X + B @ Y - X * omega)**2, axis=0)
                + np.sum((B @ X + A @ Y + Y * omega)**2, axis=0))
    return omega, X, Y, r <= conv_tol
