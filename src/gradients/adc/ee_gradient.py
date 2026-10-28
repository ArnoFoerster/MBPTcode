"""Nuclear force of a folded ADC(2) or SCS-ADC(2) state on an RHF reference,
density fitted.
The method-general engines are src.gradients.df_correlation_force (the
dense reference: every tensor whole; the gate the tiled route is checked
against) and src.gradients.df_correlation_force_tiled (occupied tiles,
auxiliary chunks and AO shell blocks over the fold's MPI ranks); this module
hands them the ADC(2) derivatives.

E_X = E_HF + E_MP2 + omega, omega = t1 Phi (ee_relaxed_density: Phi = y^T
A_eff(omega) y with y, omega held; t1 the singles weight). dPhi/df and
dPhi/dB^P_pq are ee_relaxed_density.dense_pieces (tiled_pieces on the tiled
route); the B_vv derivative is Gamma^P_ab = sum_k y_ka Xvv[(k, b), P],
assembled whole on the dense route and kept factored on the tiled one.

The LR term (singlets with a transition environment): Phi_LR = 2 t^T Q t,
t = V^T y the surface potential of the transition density (ee_ri_fold
.TransitionReactionField), adds to the orbital gradient

    Y_ri += sum_a lr_ia U_ra,   Y_ra += sum_i lr_ia U_ri,   lr = 4 y,

U = sum_k (Q t)_k v_k in the MO basis, and at fixed orbitals the derivative of
the medium's own quadratic form, 4 t1 d[(1/2) t^T Q t]/dR, which the medium
supplies (src.environments.reaction_field: transition_force) for the transition
density rho_t = (C_o y C_v^T + its transpose) / 2 returned here.

SCS-ADC(2) (spin_scaling 'scs', 'sos' or (c_os, c_ss); ee_ri_fold): E_X =
E_HF + E_SCS-MP2 + omega_SCS, the ground state scaled with the same factors
(Hellweg, Gruen and Haettig, Phys. Chem. Chem. Phys. 10, 4119 (2008)); the
fold, its root and Phi are the scaled operator's (op.scs), the MP2 pieces
SCS-MP2's, and the Z-vector solves for both together. The triplet is the
Ms = 0 component (spin-component scaling is not spin-rotation invariant),
Turbomole's spin-adapted triplet. (1, 1) is ADC(2) bit for bit.
"""
import numpy as np

from src.Base.numerics.constants import (DF_FORCE_AUX_CHUNK, DF_FORCE_BLOCK_GB,
                                 EE_FORCE_POLISH_STEPS, EE_FORCE_ROOT_RESIDUAL,
                                 EE_FORCE_ROOT_TOL, EE_RI_TILE_GB,
                                 ROOT_FOLLOW_WEIGHT_MIN)
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients import df_factors
from src.gradients.adc import fold_factors
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference.common.df_amplitudes import spin_scaling_factors
from src.SingleReference.ADC.eeADC.ee_follow import (flag_unconverged, pick_followed,
                                                     root_missing)
from src.Solvers.davidson import overlap_pick, solve_symmetric


def _lr_terms(pc, env, mol, C, nf, nocc, nmo, t1):
    """(Y_extra, rho_t) of the LR term (module docstring), (None, None)
    without one."""
    if pc.lr is None:
        return None, None
    o, v = slice(nf, nocc), slice(nocc, nmo)
    U = erd.lr_potential(env, mol, C, pc.q)
    Y_extra = np.zeros((nmo, nmo))
    Y_extra[:, o] = U[:, v] @ pc.lr.T
    Y_extra[:, v] = U[:, o] @ pc.lr
    Y_extra *= t1
    rt = C[:, o] @ pc.y @ C[:, v].T
    return Y_extra, 0.5 * (rt + rt.T)


def adc2_pieces(op, y, omega, t1, spin, nf, nocc, nmo, rf=None, env=None,
                mol=None, C=None):
    """(gamma, Gamma, Y_extra, rho_t) of omega = t1 Phi in the full MO layout:
    Y_extra the LR orbital term (None without one) and rho_t the AO transition
    density whose medium derivative carries the weight 4 t1."""
    pc = erd.dense_pieces(erd.dense_inputs(op), y, omega, spin, rf=rf, scaling=op.scs)
    no, nv = op.no, op.nv
    naux = pc.GL.shape[1]
    o, v = slice(nf, nocc), slice(nocc, nmo)
    gamma = np.zeros((nmo, nmo))
    gamma[o, o], gamma[v, v] = pc.g_oo, pc.g_vv
    Gam = np.zeros((nmo, nmo, naux))
    Gam[o, v] = pc.GL.reshape(no, nv, naux)
    Gam[o, o] = pc.Goo
    Gam[v, v] = np.einsum('ka,kbP->abP', pc.y, pc.Xvv.reshape(no, nv, naux))
    Y_extra, rho_t = _lr_terms(pc, env, mol, C, nf, nocc, nmo, t1)
    return t1 * gamma, t1 * Gam, Y_extra, rho_t


def solve_root(mol, mf, spin, nroots, frozen, auxbasis, env=None,
               conv_tol=EE_FORCE_ROOT_TOL, tile_gb=EE_RI_TILE_GB, comm=None, aux_chunk=None,
               factors=None, scaling=None):
    """(op, rf, {omega, y, t1}) of the folded ADC(2) roots on mf, the LR term
    of singlets when env (a transition environment) is given. Over comm's
    ranks (default: none, serial) the operator is distributed and rank 0
    iterates (ee_ri_fold.drive); every rank returns the same roots. aux_chunk:
    the width of the fold's auxiliary chunks (ee_ri_fold.aux_chunks).
    factors: the bare factors (ee_ri_fold.bare_factors) of the same mf,
    frozen, auxbasis, comm and aux_chunk when already built (the other spin
    of one geometry); None builds them. scaling: (c_os, c_ss) of SCS-ADC(2)
    or None."""
    f = factors if factors is not None else ee_ri_fold.bare_factors(
        mol, mf, frozen=frozen, auxbasis=auxbasis, comm=comm, aux_chunk=aux_chunk)
    rf = (ee_ri_fold.TransitionReactionField.from_environment(env, mol, mf, frozen)
          if env is not None and spin == 'singlet' else None)
    op = ee_ri_fold.RIFoldADC2(f, spin, level='adc2', reaction_field=rf,
                               tile_gb=tile_gb, comm=comm, aux_chunk=aux_chunk,
                               scaling=scaling)

    def solve(o):
        # the force's root is polished by Newton, named so a changed default
        # cannot switch it
        r = ee_fold.solve_folded(o.pieces(), nroots, spin=spin, dense_limit=0,
                                 solver='newton', tol_omega=conv_tol,
                                 tol_residual=max(conv_tol, EE_FORCE_ROOT_RESIDUAL))
        return dict(omega=np.asarray(r.omega), y=np.asarray(r.y),
                    t1=np.asarray(r.t1),
                    converged=np.array(np.atleast_1d(r.converged), bool),
                    flags=list(r.flags))

    out = ee_ri_fold.drive(op, solve)
    return op, rf, out


def polish_root(op, y, omega, max_steps=EE_FORCE_POLISH_STEPS):
    """The root of op near (omega, y) -- a root the call solved more loosely
    on the same operator -- converged as the force's roots are (EE_FORCE_ROOT_TOL,
    residual EE_FORCE_ROOT_RESIDUAL): Newton on lambda(w) = w, each step the
    eigenpair of A_eff(w) that follows the previous vector (Davidson from it).
    Returns {omega, y, t1, converged, overlap} on every rank (ee_ri_fold.drive),
    converged False when max_steps did not reach |lambda - w| < EE_FORCE_ROOT_TOL
    with the eigensolve converged; overlap |<y_start|y>| of the unit vectors."""
    y0 = np.asarray(y, float) / np.linalg.norm(y)

    def solve(o):
        pieces = o.pieces()
        x, w = y0, float(omega)
        lam, t1, done = w, 1.0, False
        for _ in range(max_steps):
            matvec, dmatvec, diag_s, _, _ = ee_fold.folded_operator(pieces, w, o.spin)
            e, X, conv = solve_symmetric(matvec, diag_s, nroots=1, x0=x,
                                         pick=overlap_pick(x),
                                         tol_residual=EE_FORCE_ROOT_RESIDUAL,
                                         warn_unconverged=False,
                                         label='ADC(2) force root polish')
            lam = float(e[0])
            x = np.asarray(X[:, 0], float)
            x = x / np.linalg.norm(x)
            t1 = 1.0 / (1.0 + dmatvec(x))
            if abs(lam - w) < EE_FORCE_ROOT_TOL and bool(np.all(conv)):
                done = True
                break
            w = w + (lam - w) * t1
        return dict(omega=lam, y=x, t1=t1, converged=done,
                    overlap=abs(float(y0 @ x)))

    return ee_ri_fold.drive(op, solve)


def _root(solved, afresh, root, y_ref):
    """(op, rf, k, (y, omega, t1), flags) of the force's root: the call's root
    polished on the call's operator when `solved` = (op, k, y, omega) is given
    and the polish holds (converged, overlap with the call's vector at least
    ROOT_FOLLOW_WEIGHT_MIN), else every root solved afresh (afresh() ->
    solve_root's (op, rf, out)) and picked (_pick); (y, omega, t1) None for a
    missing root."""
    if solved is not None:
        op, k, y, omega = solved
        out = polish_root(op, y, omega)
        if out['converged'] and out['overlap'] >= ROOT_FOLLOW_WEIGHT_MIN:
            return op, op.rf, k, (np.asarray(out['y']), float(out['omega']),
                                  float(out['t1'])), []
    op, rf, out = afresh()
    return (op, rf) + _pick(out, root, y_ref)


def _pick(out, root, y_ref):
    """(k, (y, omega, t1), flags) of the root asked for: index `root`, or the
    one overlapping y_ref (an MO singles vector of this geometry) most; a lost
    or mixed pick (ee_follow.pick_followed) or an unconverged root is warned
    about and flagged, beside the fold's own flags; a root index the solve did
    not return is flagged 'adc_root_missing' and k and the root are None
    (ee_follow.root_missing: the caller returns a NaN force)."""
    Y = np.asarray(out['y'])
    flags = list(out.get('flags', []))
    flag_unconverged(out['converged'], 'ADC(2) force root', flags)
    if y_ref is None:
        k = root
        if root_missing(k, Y.shape[1], 'ADC(2) force root', flags):
            return None, None, flags
    else:
        w = np.abs(y_ref @ Y) / (np.linalg.norm(y_ref) * np.linalg.norm(Y, axis=0))
        k = pick_followed(w, 'ADC(2) force root (overlap following)', flags)
    return k, (Y[:, k], float(out['omega'][k]), float(out['t1'][k])), flags


def _info(mf, C, e2, omega, gam, lr, asym, k, y):
    dm_corr = C @ gam @ C.T
    return dict(energy=mf.e_tot + e2 + omega, e_mp2=e2, omega=omega,
                dm_corr=dm_corr, dm_eff=mf.make_rdm1() + dm_corr, lr=lr,
                asymmetry=asym, index=k,
                y=None if y is None else y / np.linalg.norm(y))


def _missing(mf, mol, flags):
    """The NaN force and info of a root the solve did not return (_pick)."""
    nan = np.full_like(np.asarray(mf.make_rdm1(), float), np.nan)
    return np.full((mol.natm, 3), np.nan), dict(
        energy=np.nan, e_mp2=np.nan, omega=np.nan, dm_corr=nan, dm_eff=nan,
        lr=[], asymmetry=np.nan, index=None, y=None, flags=flags)


def dense_force(mf, mol, frozen, auxbasis, op=None, root=None, rf=None, env=None,
                numerics=None, flags=None, scaling=None):
    """(dE/dR (natm, 3), info) of E_HF + E_MP2 (+ omega of root = (y, omega,
    t1), a root of op) on the dense reference; numerics and flags the
    Z-vector's (df_correlation_force.mp2_force); scaling (c_os, c_ss) of
    SCS-MP2 (and op's, which must agree) or None."""
    _same_scaling(op, scaling)
    excited = None
    if root is not None:
        y, omega, t1 = root

        def excited(C, nocc, nmo):
            return adc2_pieces(op, y, omega, t1, op.spin, frozen, nocc, nmo, rf, env,
                               mol, C)
    F, (e2, gam, asym, rho_t) = dcf.mp2_force(mf, mol, frozen, auxbasis, excited,
                                              numerics, flags, scaling)
    return F, _record(e2, gam, asym, root, rho_t)


def _same_scaling(op, scaling):
    """Refuse a ground state scaled otherwise than the operator: the energy
    would be no method's."""
    if op is not None and op.scs != (None if scaling is None else tuple(scaling)):
        raise ValueError(f'the ground state is scaled by {scaling!r}, the operator by '
                         f'{op.scs!r}: SCS-ADC(2) takes the same factors in both')


def tiled_force(mf, mol, op, frozen, auxbasis, root=None, env=None,
                tile_gb=EE_RI_TILE_GB, block_gb=DF_FORCE_BLOCK_GB,
                width=DF_FORCE_AUX_CHUNK, numerics=None, flags=None):
    """dense_force on the fold's tiles and ranks: a collective over op's
    communicator, the same bits on every rank; the ground state scaled as op
    is (op.scs)."""
    excited = None
    if root is not None:
        y, omega, t1 = root

        def excited(d, C, nocc, nmo):
            pc = erd.tiled_pieces(op, y, omega, tile_gb)
            Y_extra, rho_t = _lr_terms(pc, env, mol, C, frozen, nocc, nmo, t1)
            return d.plus(t1, pc), Y_extra, rho_t
    F, (e2, gam, asym, rho_t) = tcf.force(mf, mol, fold_factors.FoldFactors(op), frozen,
                                          auxbasis, excited, tile_gb, block_gb, width,
                                          numerics, flags, op.scs)
    return F, _record(e2, gam, asym, root, rho_t)


def _record(e2, gam, asym, root, rho_t):
    """(e2, omega, gamma, lr, asymmetry, y) of a force, lr the LR term's
    [(4 t1, rho_t)]."""
    if root is None:
        return e2, 0.0, gam, [], asym, None
    y, omega, t1 = root
    lr = [] if rho_t is None else [(4.0 * t1, rho_t)]
    return e2, omega, gam, lr, asym, y


def adc2_force(mf, mol, spin, root=0, nroots=3, frozen=0, auxbasis=None,
               conv_tol=EE_FORCE_ROOT_TOL, ground=False, env=None, y_ref=None,
               numerics=None, solved=None, spin_scaling=None):
    """dE/dR (natm, 3) of E_HF + E_MP2 (+ omega of the root unless ground)
    on a closed-shell DF-RHF mf, with the external operator's own derivative
    left to the medium, and a dict: energy, omega, dm_corr (the relaxed
    correlation density, AO), dm_eff (D_HF + dm_corr), lr = [(weight,
    rho_t)], asymmetry (max |Y - Y^T|), index (the root taken: the one
    overlapping y_ref most, if given), flags (the root's and the Z-vector's
    numerical events). The dense reference; numerics the caller's (the
    Z-vector's tolerances). solved = (op, k, y, omega): the root k a call
    solved on this mf with the operator op (serial), polished there instead of
    solving every root again (_root); its operator must be scaled as asked.
    spin_scaling: None (ADC(2)), 'scs', 'sos' (ADC2_SPIN_SCALING) or (c_os,
    c_ss): SCS-ADC(2) on SCS-MP2 (module docstring)."""
    scaling = spin_scaling_factors(spin_scaling)
    C = np.asarray(mf.mo_coeff)
    op = rf = rt = k = None
    flags = []
    if not ground:
        # a root asked to another tolerance than the force's own is solved afresh
        op, rf, k, rt, flags = _root(
            solved if conv_tol == EE_FORCE_ROOT_TOL else None,
            lambda: solve_root(mol, mf, spin, nroots, frozen, auxbasis, env, conv_tol,
                               scaling=scaling),
            root, y_ref)
        if rt is None:
            return _missing(mf, mol, flags)
    F, (e2, omega, gam, lr, asym, y) = dense_force(mf, mol, frozen, auxbasis, op,
                                                   rt, rf, env, numerics, flags,
                                                   scaling)
    info = _info(mf, C, e2, omega, gam, lr, asym, k, y)
    info['flags'] = flags
    return F, info


def adc2_force_tiled(mf, mol, spin, root=0, nroots=3, frozen=0, auxbasis=None,
                     conv_tol=EE_FORCE_ROOT_TOL, ground=False, env=None, y_ref=None,
                     tile_gb=EE_RI_TILE_GB, max_memory=None,
                     block_gb=DF_FORCE_BLOCK_GB, width=DF_FORCE_AUX_CHUNK, comm=None,
                     aux_chunk=None, numerics=None, solved=None, spin_scaling=None):
    """adc2_force on the tiled route over comm's ranks (default: the current
    communicator), refused up front (MemoryError) when the per-rank estimate
    (df_correlation_force_tiled.estimate) exceeds max_memory (MB; default
    mol.max_memory). solved and spin_scaling as adc2_force's, op on these
    ranks."""
    scaling = spin_scaling_factors(spin_scaling)
    if ground:
        flags = []
        F, (e2, gam, asym, est) = tcf.mp2_force_tiled(mf, mol,
                                                      df_factors.mp2_factors,
                                                      frozen, auxbasis, tile_gb,
                                                      max_memory, block_gb, width,
                                                      comm, aux_chunk, numerics, flags,
                                                      scaling)
        info = _info(mf, np.asarray(mf.mo_coeff), e2, 0.0, gam, [], asym, None, None)
        info['estimate'], info['flags'] = est, flags
        return F, info
    comm, est = tcf.prepare(mf, mol, frozen, auxbasis, True, tile_gb, block_gb,
                            max_memory, comm)
    C = np.asarray(mf.mo_coeff)
    op, _, k, rt, flags = _root(
        solved if conv_tol == EE_FORCE_ROOT_TOL else None,
        lambda: solve_root(mol, mf, spin, nroots, frozen, auxbasis, env, conv_tol,
                           tile_gb, comm, aux_chunk, scaling=scaling),
        root, y_ref)
    _same_scaling(op, scaling)
    if rt is None:
        F, info = _missing(mf, mol, flags)
        info['estimate'] = est
        return F, info
    F, (e2, omega, gam, lr, asym, y) = tiled_force(mf, mol, op, frozen, auxbasis,
                                                   rt, env, tile_gb, block_gb, width,
                                                   numerics, flags)
    info = _info(mf, C, e2, omega, gam, lr, asym, k, y)
    info['estimate'], info['flags'] = est, flags
    return F, info
