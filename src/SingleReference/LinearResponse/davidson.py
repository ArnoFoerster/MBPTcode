"""Matrix-free Casida/RPA/BSE Davidson, with two ways of applying the same A/B.

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
instead. What is still missing at that size is Zt spread across ranks -- the row
tiling in `isdf_block_action` is already that decomposition -- and the ISDF
fit's own M x M Gram inversion, which lives in `separable_ri.fit_M`, not here.
"""
import sys
import time
import warnings

import numpy as np
from pyscf.lib import logger
from pyscf.tdscf._lr_eig import real_eig
from scipy.sparse.linalg import LinearOperator, eigsh

from src.Base.pyscf_interface import get_orbital_energies
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.space_time import (DEFAULT_NTAU, _unpack_factors,
                                              separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.LinearResponse.exciton_descriptors import exciton_descriptors
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency


#: Budget for the ISDF block action's row tiles. Only the (rows, M) buffer is
#: sized by it; Zt and the (M, n_occ) accumulators are not tileable.
DEFAULT_TILE_MEMORY_GB = 4.0

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


def solve_casida_davidson(lr_solver, nocc, nroots=3, polarizability='RPA',
                           W_aux=None, conv_tol=1e-5, max_cycle=100, orbsym=None,
                           isdf_factors=None, guess_factor=GUESS_FACTOR,
                           stats=None):
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
    Returns (omega, X, Y) normalized <X|X>-<Y|Y>=1.
    """
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors)
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    return _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                         _pair_symmetry(orbsym, occ, virt), guess_factor,
                         stats=stats)


def _block_action(lr_solver, nocc, polarizability, W_aux, isdf_factors):
    """Mode dispatch shared by the solver and the (A-B) instability probe."""
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
        return df_block_action(lr_solver, nocc, lBSE, w)
    return isdf_block_action(lr_solver, nocc, lBSE, w, isdf_factors)


def lowest_amb_eigenvalue(lr_solver, nocc, polarizability='BSE', W_aux=None,
                          isdf_factors=None, k=1, tol=1e-6, sign_only=False,
                          stats=None):
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
    """
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors)
    return _lowest_amb_from_action(apply_AB, diag_d, k=k, tol=tol,
                                   sign_only=sign_only, stats=stats)


def _lowest_amb_from_action(apply_AB, diag_d, k=1, tol=1e-6, sign_only=False,
                            stats=None):
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
            w, v = eigsh(op, k=1, which='SA', tol=tol_try,
                         return_eigenvectors=True)
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

    out = np.sort(eigsh(op, k=min(k, no * nv - 1), which='SA', tol=tol,
                        return_eigenvectors=False))
    _record()
    return out


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
    """
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


def solve_bse_isdf(mf, mol, nocc, nroots=5, qp='G0W0', factors=None,
                   auxbasis=None, radii=None, counts=None, probe=True,
                   conv_tol=1e-5, max_cycle=100, guess_factor=GUESS_FACTOR,
                   gw_kwargs=None, progress=None, n_start=1):
    """BSE by the ISDF matrix-free Davidson: mean field in, excitations out.

    The production calling sequence is three lines --

        mf = dft.RKS(mol, xc=...).density_fit(auxbasis=...); mf.kernel()
        omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=5)

    -- and the discipline the pieces demand lives HERE so the caller cannot
    violate it: one ISDF fit is built (or taken from `factors`) and shared by
    the GW and BSE stages; the static W is always built from that fit at the
    MEAN-FIELD energies while the BSE diagonal carries `qp` -- the standard
    G0W0-BSE split; and W and the block action can never mix auxiliary gauges.

    qp: 'G0W0' (default) puts every quasiparticle energy from ONE space-time
    self-energy on the BSE diagonal (`solve_qp_diagonal_space_time`, which
    applies <Sigma_x - v_xc> per state for a KS reference); an ARRAY of
    energies is used as the diagonal directly (an evGW result, say);
    False/None solves BSE@mean-field.
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

    progress: stamp each stage to stdout as it begins and ends, so a job that
    is going to take node-hours says which stage it is in WHILE it is in it.
    The returned timings only arrive if the run finishes, which is exactly the
    case that does not need them. None follows `mol.verbose`, so the one knob
    that already turns on the mean field's output turns on this too rather
    than leaving a second one to forget.

    Returns (omega, X, Y, info); info carries eps / eps_mf / factors / W_aux /
    min_eig_amb, length-gauge oscillator_strength and transition_dipole per
    root (`oscillator_strengths`), and per-stage timings in seconds.
    """
    t = {}
    stats = {}
    if progress is None:
        progress = getattr(mol, 'verbose', 0) > 0

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
                                    counts=counts, n_start=n_start)
        _end('factors', t0)
        if progress:
            print(f'[bse {time.strftime("%H:%M:%S")}] ISDF M={factors[1].shape[0]} '
                  f'({factors[1].shape[0] // mol.natm}/atom)', flush=True)

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
                                              extras=gw_extras,
                                              **(gw_kwargs or {}))
        _end('qp', t0)
    else:
        eps = np.asarray(qp, dtype=float)
        if eps.shape != eps_mf.shape:
            raise ValueError(f'qp energies have shape {eps.shape}, the mean '
                             f'field has {eps_mf.shape}.')

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
        _, _, W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)
    _end('W', t0)

    lr = LinearResponseSolver(eps, spin_mode='restricted')
    amb = None
    if probe:
        t0 = _begin('probe')
        amb = float(lowest_amb_eigenvalue(
            lr, nocc, polarizability='BSE', W_aux=W_aux, isdf_factors=factors,
            sign_only=(isinstance(probe, str) and probe.lower() == 'sign'),
            stats=stats)[0])
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
                                        guess_factor=guess_factor, stats=stats)
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
                timings=t, stats=stats)
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


def isdf_bse_factors(mf, mol, nocc, eps=None,
                     auxbasis=None, radii=None, counts=None, factors=None,
                     screening='imaginary-time', ntau=DEFAULT_NTAU,
                     tau_target=DEFAULT_TAU_TARGET, n_start=1):
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
    factorization error responds to. `separable_ri.published_grids` is used
    automatically where it applies (H, C, N, O at cc-pVTZ) and the radii are
    optimized per element otherwise.
    factors: pre-built (X_mo, D), to reuse a factorization across states.

    screening:
      'imaginary-time'  chi0(i.0) built from the separable factors in imaginary
            time by `LinearResponse/space_time.py`, which is tiled and cubic and
            touches nothing bigger than (naux, naux). The only route that
            reaches production sizes.
      'df'  chi0 from the three-index factor these factors imply. Exact for the
            given factors, and the reference the imaginary-time route was
            checked against -- 1e-8 relative on the W entries at ntau=18 on
            water and ethene / cc-pVDZ -- but it forms B and then a
            (naux, n_occ, n_vir) array, which at a chlorophyllide-a hexamer /
            cc-pVTZ are 25 TB and 2 TB. Small systems only.

    ntau: imaginary-time points; 'auto' (the default) sizes the grid from the
    Kaltak-Klimes-Kresse test integral over the chi0 transition range at
    `tau_target`, exactly as the GW space-time route does -- but over that one
    range alone, since nothing here transforms a self-energy. A fixed value is
    not monotonically safe: the transform's Remez window closes above ~20 at
    molecular energy ranges and the fit COLLAPSES rather than saturating --
    measured against the 'df' reference on water / cc-pVDZ, 4.6e-7 at ntau=12,
    1.0e-8 at 18, and back up to 7.2e-4 at 24. The upward scan in
    `minimax_points_for_accuracy` stops at the first size that meets
    `tau_target` and never enters that region.
    """
    X_mo, D = _unpack_factors(
        factors if factors is not None
        else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                               counts=counts, n_start=n_start))[:2]
    if eps is None:
        eps = get_orbital_energies(mf, representation='spatial')

    if screening == 'df':
        lr = LinearResponseSolver(eps, coeff_df=isdf_df_coefficients(X_mo, D),
                                  spin_mode='restricted')
        return X_mo, D, np.asarray(lr.static_screening_aux(nocc))
    if screening != 'imaginary-time':
        raise ValueError(f"screening='{screening}'; choose 'imaginary-time' or 'df'.")

    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, _ = minimax_points_for_bse(eps, nocc, tau_target=tau_target)
    # ONE frequency, omega = 0: the BSE kernel wants the STATIC screening, so
    # only the tau -> omega direction is ever evaluated. with_inverse=False
    # skips the omega -> tau matrices, which cannot be fitted from a single
    # input point and would otherwise warn on every BSE run.
    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           [0.0], [1.0], with_sine=False,
                                           with_inverse=False)
    chi0 = chi0_imaginary_frequency(X_mo, D, eps, nocc, grid)[0]
    return X_mo, D, np.linalg.inv(np.eye(chi0.shape[-1]) - chi0)


def _pair_symmetry(orbsym, occ, virt):
    """Irrep label of every occupied-virtual pair, for the symmetry-block guess."""
    if orbsym is None:
        return None
    orbsym_d2h = np.asarray(orbsym) % 10
    return (orbsym_d2h[occ][:, None] ^ orbsym_d2h[virt][None, :]).ravel()


def df_block_action(lr_solver, nocc, lBSE, W_aux,
                    tile_memory_gb=DEFAULT_TILE_MEMORY_GB):
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
    """
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)

    C_ov = lr_solver.df_coeff[:, occ[:, None], virt]
    C_oo = lr_solver.df_coeff[:, occ[:, None], occ]
    C_vv = lr_solver.df_coeff[:, virt[:, None], virt]
    diag_d = (lr_solver.eps[virt][None, :] - lr_solver.eps[occ][:, None])
    factor = 2.0

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
        # The Hartree term enters A and B identically, so it is contracted once
        # per trial vector rather than once per block.
        v = factor * apply_V(z)
        Az = diag_d[None, :, :] * z + v
        Bz = v
        if lBSE:
            Az = Az - apply_exchange_direct(z)
            Bz = Bz - apply_exchange_swap(z)
        return Az, Bz

    return apply_AB, diag_d


def isdf_block_action(lr_solver, nocc, lBSE, W_aux, isdf_factors,
                      tile_memory_gb=DEFAULT_TILE_MEMORY_GB):
    """The same A/B action from a separable RI -- Fock-like builds on the grid.

    Nothing here carries a (naux, nvirt, nvirt) array, and the only object that
    scales as M^2 is the screened kernel Zt, built once. Everything else is
    TILED over grid rows, because every contraction is a sum over them:

        [K z]_ia = sum_k X_o[k,i] sum_k' (Zt * P)[k,k'] X_v[k',a]

    so a row block contributes X_o[blk]^T (Zt[blk] * P[blk]) X_v and the M x M
    Hadamard product is never formed. The tiling is free -- same flop count,
    same GEMM shapes -- and at a chlorophyllide-a hexamer / cc-pVTZ (nao 10980,
    M 117762) it is the difference between holding 103 GB and 310 GB. Zt itself
    is the wall past that, and this same row decomposition is what distributing
    it across ranks would use.
    """
    if lr_solver.spin_mode == 'unrestricted':
        raise NotImplementedError("The ISDF Davidson route is restricted-spin only.")
    X_mo, D = _unpack_factors(isdf_factors)[:2]
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    if X_mo.shape[1] != len(lr_solver.eps):
        raise ValueError(f"isdf_factors X has {X_mo.shape[1]} orbitals, "
                         f"lr_solver.eps has {len(lr_solver.eps)}.")
    if W_aux is not None and W_aux.shape[0] != D.shape[1]:
        raise ValueError(f"W_aux is ({W_aux.shape[0]}, ...) but D has "
                         f"{D.shape[1]} auxiliary functions; they are not from "
                         "the same fit (see isdf_bse_factors).")

    diag_d = (lr_solver.eps[virt][None, :] - lr_solver.eps[occ][:, None])
    factor = 2.0
    X_o = np.ascontiguousarray(X_mo[:, occ])
    X_v = np.ascontiguousarray(X_mo[:, virt])
    npts, no = X_o.shape

    # Zt = D W_aux D^T, the per-tau step of the space-time screening, wanted
    # once and statically here. W_aux = None is TDHF, where the exchange
    # kernel is the bare Coulomb, i.e. Zt = Z = D D^T.
    if lBSE:
        if W_aux is not None:
            asym = np.abs(W_aux - W_aux.T).max()
            if asym > 1e-10 * np.abs(W_aux).max():
                raise ValueError(f"W_aux is not symmetric (max asymmetry "
                                 f"{asym:.2e}); the B block below relies on it.")
        Zt = D @ D.T if W_aux is None else D @ (W_aux @ D.T)
        rows = max(1, min(npts, int(tile_memory_gb * 1e9 / max(npts * 8, 1))))
        S = np.empty((rows, npts))            # one row block of Zt * P
        T = np.empty((no, npts))              # X_o^T (Zt * P), accumulated
        Tb = np.empty((no, npts))
        U = np.empty((npts, no))              # (Zt * P) X_o, filled by block

    def apply_AB(z):
        Az = diag_d[None, :, :] * z
        Bz = np.empty_like(z)
        for n, zn in enumerate(z):
            zXv = zn @ X_v.T                              # (n_occ, M)
            # Hartree needs only P's DIAGONAL, and the bare Z = D D^T is never
            # formed either: two (M, naux) products instead of an (M, M) one.
            p = np.einsum('kj,jk->k', X_o, zXv, optimize=True)
            u = D @ (p @ D)
            hartree = factor * (X_o.T @ (u[:, None] * X_v))
            Az[n] += hartree
            Bz[n] = hartree
            if not lBSE:
                continue
            T[:] = 0.0
            for p0 in range(0, npts, rows):
                p1 = min(p0 + rows, npts)
                blk = S[:p1 - p0]
                np.matmul(X_o[p0:p1], zXv, out=blk)       # P's rows
                blk *= Zt[p0:p1]                          # the Hadamard product
                np.matmul(X_o[p0:p1].T, blk, out=Tb)
                np.add(T, Tb, out=T)          # not T += Tb: that rebinds T
                np.matmul(blk, X_o, out=U[p0:p1])
            Az[n] -= T @ X_v
            # The B block wants Zt * P^T. Forming that Hadamard product directly
            # reads Zt against a transposed operand and measured as costly as
            # everything else in the step put together; with Zt symmetric --
            # checked above, and it is the screened interaction, so it must be --
            # it is (Zt * P)^T, and (Zt * P)^T X_o is the U already built.
            Bz[n] -= U.T @ X_v
        return Az, Bz

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


def _run_davidson(apply_AB, diag_d, nroots, conv_tol, max_cycle, x_sym,
                  guess_factor=GUESS_FACTOR, stats=None):
    """Drive pyscf's real_eig from a block action, whatever produced it.

    stats: optional dict, filled with where the time went. The Davidson is the
    largest stage of a production BSE (37.6% at the chlorophyllide dimer/
    cc-pVDZ) and the total alone does not say whether that is the block action
    doing necessary work or the solver taking too many iterations to get there.
    Counting the action separately from everything around it distinguishes
    "make the action faster" from "give it a better guess", which are different
    pieces of work.
    """
    no, nv = diag_d.shape
    n_pair = no * nv
    ncall = [0]
    nvec = [0]
    t_action = [0.0]
    t_total = [0.0]

    def vind(xys):
        t_enter = time.time()
        xys = np.asarray(xys).reshape(-1, 2, no, nv)
        ncall[0] += 1
        nvec[0] += 2 * xys.shape[0]          # a block action per X and per Y
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

    def precond(dx, e):
        e = np.atleast_1d(e)
        d = hdiag[None, :] - e[:, None]
        d[np.abs(d) < 1e-8] = 1e-8
        return (dx.reshape(len(e), -1) / d).reshape(dx.shape)

    order = _guess_indices(diag_d, nroots, guess_factor)
    x0 = np.zeros((len(order), 2 * n_pair))
    x0[np.arange(len(order)), order] = 1.0
    x0sym = x_sym[order] if x_sym is not None else None

    try:
        converged, e, xy = real_eig(vind, x0, precond, tol_residual=conv_tol,
                                     nroots=nroots, x0sym=x0sym, max_cycle=max_cycle,
                                     verbose=logger.Logger(sys.stdout, 0))
    except np.linalg.LinAlgError as exc:
        # pyscf's real_eig Cholesky-factorizes the projected (A-B) block, so
        # this is the response subspace losing positive definiteness -- the
        # instability regime of the mean-field reference (long acenes reach it
        # near a11), where the omega^2 reduction is INVALID rather than merely
        # ill-conditioned. No shift or retry inside this solver gives physical
        # roots there: the dense CasidaSolver's eta-shift is a diagnostic that
        # changes the spectrum, not an answer. Stabilize the reference, or
        # solve the non-Hermitian problem directly.
        try:
            amb = float(_lowest_amb_from_action(apply_AB, diag_d)[0])
            detail = f'measured min eig(A-B) = {amb:.6f} Ha'
        except Exception:
            detail = ('min eig(A-B) probe did not converge; smallest '
                      f'orbital-energy difference {diag_d.min():.4f} Ha')
        raise RuntimeError(
            'Casida-form Davidson failed: the projected (A-B) block is not '
            'positive definite, the signature of a singlet/triplet '
            f'instability of the mean-field reference ({detail}). The '
            'omega^2 reduction is invalid in this regime and shifted or '
            'retried roots would not be physical. Check first that the SCF is '
            'the LOWEST solution (vary the initial guess, follow '
            'instabilities) -- a converged non-minimum reference produces '
            'exactly this failure.') from exc
    if stats is not None:
        stats.update({'davidson_vind_calls': ncall[0],
                      'davidson_block_actions': nvec[0],
                      'davidson_action_s': t_action[0],
                      'davidson_vind_s': t_total[0]})
    # A root that never converged still comes back with an energy attached, and
    # returning it unremarked is how a Davidson result silently stops meaning
    # anything. It does NOT flag the guess failure GUESS_FACTOR describes --
    # that one converges.
    converged = np.asarray(converged)
    if not converged.all():
        stuck = np.flatnonzero(~converged)
        warnings.warn(
            f'Davidson left {stuck.size} of {nroots} roots unconverged after '
            f'{max_cycle} cycles at |r| <= {conv_tol:g} (roots '
            f'{stuck.tolist()}); their energies are whatever the last subspace '
            'happened to give. Raise max_cycle, or loosen conv_tol.',
            RuntimeWarning, stacklevel=3)

    omega = np.asarray(e)
    X = np.zeros((n_pair, len(omega)))
    Y = np.zeros((n_pair, len(omega)))
    for k, z in enumerate(xy):
        x, y = z.reshape(2, no, nv)
        norm = np.sqrt(abs(np.sum(x**2) - np.sum(y**2)))
        X[:, k] = (x / norm).ravel()
        Y[:, k] = (y / norm).ravel()
    return omega, X, Y
