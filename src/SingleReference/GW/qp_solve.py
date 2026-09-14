"""Pieces shared by every imaginary-axis quasiparticle route.

The imaginary-frequency and space-time routes differ ONLY in how they produce
Sigma_c sampled on the imaginary axis. Everything downstream of that -- the
static Sigma_x - v_xc term, the Pade continuation to the real axis, and the
root search on the QP equation -- is identical, so it lives here rather than
being duplicated in each driver.

A route is therefore expected to supply just `(z_fit, sigma_iw)` and hand off.

References
----------
Vidberg and Serene, J. Low Temp. Phys. 29, 179 (1977) -- Pade continuation of
a function sampled on the imaginary axis via Thiele's continued fraction, the
algorithm behind `Base/utils/analyticalContinuation.thiele_coefficients`.
"""
import numpy as np
from pyscf import df as _pyscf_df, scf as _pyscf_scf
from pyscf.df import incore as _df_incore

from src.Base.solvent_screening import solvent_static_selfenergy
from src.Base.utils.analyticalContinuation import (greedy_pade_order,
                                                   thiele_coefficients,
                                                   pade_eval)
from src.Solvers.qp_equation import solve_qp_equation


def static_exchange_matrix(mf, mol, dm_correction=None, exchange='mf',
                           reaction_field=None):
    """
    <p| Sigma_x - v_xc |q> over the whole MO basis.
    Zero by construction on a Hartree-Fock reference
    dm_correction: optional AO 1RDM used in place of the
    mean-field density for Sigma_x only (a CCSD or GW density, say).
    exchange: which K builds Sigma_x. 'mf' is the mean field's own, which on
        an ISDF mean field carries the grid's K error into the QP energy at
        FIRST order -- the v_xc part of that error cancels against the
        eigenvalue shift, the Sigma_x part does not (benzene HOMO 8.5 meV, a
        conjugated macrocycle ~30 meV). 'exact' builds the full-range K by
        direct integration; 'df' by density fitting with the STORED three-index
        tensor, which is naux nao^2/2 -- 84 GB at 1780 basis functions, 1.6 TB
        at 4736 -- so it is a small-molecule cross-check only. For production
        use `static_exchange_diagonal(exchange='df-direct')`, which holds no
        such tensor. v_xc is always the SCF's own operator: the eigenvalues
        came from it, and replacing it would break that cancellation.

    An attached solvent screening adds its first-order reaction-field
    (static COHSEX) operator here: the static self-energy is where the Born
    polarization energy lives, and this matrix is the one static object every
    imaginary-axis route shares. None attached (gas phase) adds nothing.

    Sigma_c is NOT negligible next to it. Its leading vtilde dependence is
    v chi0 vtilde, first order in each and not second order in vtilde, and it
    OPPOSES the Born term: on pyridine/cc-pVDZ in toluene it gives back 0.75 eV
    of the 3.08 eV that Sigma^solv closes the quasiparticle gap by, off a B3LYP
    starting point, and less off Hartree-Fock, since the size of the term
    follows chi0. Nothing double counts -- Sigma_c is built from W - (v +
    vtilde), so the instantaneous reaction field is counted once, here -- but
    the split is not a large term plus a small one.

    reaction_field REPLACES that operator when given; see
    `static_exchange_diagonal`.
    """
    dm = mf.make_rdm1()
    dm_for_hx = dm if dm_correction is None else dm_correction
    if exchange == 'mf':
        K = mf.get_k(mol, dm_for_hx)
    elif exchange == 'exact':
        K = _pyscf_scf.hf.get_jk(mol, dm_for_hx, hermi=1, with_j=False)[1]
    elif exchange == 'df':
        aux = getattr(mf.with_df, 'auxbasis', None) or (str(mol.basis) + '-ri')
        K = _pyscf_df.DF(mol, auxbasis=aux).get_jk(dm_for_hx, hermi=1,
                                                    with_j=False)[1]
    else:
        raise ValueError(f"exchange must be 'mf', 'exact' or 'df', got {exchange!r}")
    sig_x = -0.5 * K
    v_xc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
    mo = mf.mo_coeff
    out = mo.T @ (sig_x - v_xc) @ mo
    if reaction_field is not None:
        return out + np.diag(np.asarray(reaction_field, float))
    sigma_solvent = solvent_static_selfenergy(mf, mol)
    if sigma_solvent is not None:
        if isinstance(sigma_solvent, tuple):
            raise NotImplementedError(
                "solvent screening through static_exchange_matrix is "
                "restricted-only, like its consumers")
        out = out + sigma_solvent
    return out


def _df_direct_exchange_diagonal(mol, auxbasis, mo_states, dm, block_memory_gb):
    """<p| K[dm] |p> for the columns of `mo_states`, from ONE pass over the
    three-index integrals, never holding them.

    With dm = sum_k n_k v_k v_k^T (its natural orbitals),
        K[dm]_pp = sum_k n_k sum_PQ (p k|P) [V^-1]_PQ (Q|k p),
    so the only object that grows past one integral block is (p k|P): naux x
    nstates x nao while the AO index is still open, naux x nstates x nk once it
    is contracted. The pass is the same integral work as one integral-direct
    DF-J, which the isdf route already pays every SCF cycle. At nao = 4736 and
    naux = 17664 that is 0.67 GB per AO of block, ~6.7 GB for ten states and a
    2.5 GB metric, against 1.6 TB for the stored tensor.
    """
    auxmol = _pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    T = np.zeros((naux, mo_states.shape[1], nao))
    # Big blocks: each aux_e2 call rebuilds a shell-pair list over
    # nbas x auxnbas, so the CALL COUNT costs at large nbas.
    max_ao = max(1, int(block_memory_gb * 1e9 / (nao * naux * 8)))
    sh0 = 0
    while sh0 < mol.nbas:
        sh1 = sh0 + 1
        while sh1 < mol.nbas and ao_loc[sh1 + 1] - ao_loc[sh0] <= max_ao:
            sh1 += 1
        ints = _df_incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1',
                                 shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas))
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        T += np.einsum('mnP,mp->Ppn', ints, mo_states[a0:a1], optimize=True)
        del ints
        sh0 = sh1
    n, v = np.linalg.eigh(dm)
    keep = n > 1e-10 * n.max()
    B = np.einsum('Ppn,nk->Ppk', T, v[:, keep] * np.sqrt(n[keep]), optimize=True)
    del T
    V = auxmol.intor('int2c2e', aosym='s1')
    w, u = np.linalg.eigh(V)
    kp = w > 1e-12 * w.max()
    B = np.einsum('PQ,Qpk->Ppk', (u[:, kp] / np.sqrt(w[kp])) @ u[:, kp].T, B,
                  optimize=True)
    return np.einsum('Ppk,Ppk->p', B, B)


def static_exchange_diagonal(mf, mol, states, dm_correction=None, exchange='mf',
                             block_memory_gb=None, reaction_field=None):
    """<p| Sigma_x - v_xc |p> for the requested states only.

    reaction_field REPLACES the continuum's static term, it does not add to it.
    A route that has built W can form Duchemin et al.'s Eq. (18) shift, the
    self-polarization of the orbital carrying the added charge in the SCREENED
    reaction field; a route that has not (ADC, whose secular matrix starts at
    Sigma^(2) and never forms W) falls back to `cohsex_correction`, which is
    the same physics with the BARE vtilde summed over every orbital. Passing
    one while the other is also applied counts the polarization energy twice --
    on acrolein in water that is 2.65 + 2.35 eV of gap closure where 2.3 is
    right, which reads as plausible over-screening rather than as a bug.

    Every consumer of the static exchange on the QP routes needs diagonals for
    a few states, and that is what makes an exchange build that scales
    possible: 'df-direct' streams the three-index integrals once against the
    state vectors (`_df_direct_exchange_diagonal`) instead of storing them or
    forming a full K. The other choices go through `static_exchange_matrix`.
    v_xc is the SCF's own operator throughout; see `static_exchange_matrix`.

    block_memory_gb: for 'df-direct', the integral slab per aux_e2 call. None
        takes a quarter of mf.max_memory.
    """
    states = np.atleast_1d(states).astype(int)
    if exchange != 'df-direct':
        return np.diag(static_exchange_matrix(
            mf, mol, dm_correction=dm_correction, exchange=exchange,
            reaction_field=reaction_field))[states]
    dm = mf.make_rdm1()
    dm_for_hx = dm if dm_correction is None else dm_correction
    mo = mf.mo_coeff[:, states]
    aux = getattr(mf.with_df, 'auxbasis', None) or (str(mol.basis) + '-ri')
    if block_memory_gb is None:
        block_memory_gb = max(1.0, 0.25 * getattr(mf, 'max_memory', 4000) / 1e3)
    k_pp = _df_direct_exchange_diagonal(mol, aux, mo, dm_for_hx, block_memory_gb)
    v_xc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
    out = -0.5 * k_pp - np.einsum('mp,mn,np->p', mo, v_xc, mo, optimize=True)
    if reaction_field is not None:
        return out + np.asarray(reaction_field, float)[states]
    sigma_solvent = solvent_static_selfenergy(mf, mol)
    if sigma_solvent is not None:
        if isinstance(sigma_solvent, tuple):
            raise NotImplementedError(
                "solvent screening through static_exchange_diagonal is "
                "restricted-only, like its consumers")
        out = out + np.diag(sigma_solvent)[states]
    return out


def static_exchange_correction(mf, mol, p_state, dm_correction=None,
                               exchange='mf'):
    """
    <p| Sigma_x - v_xc |p>:
    static_exchange_diagonal for a single state.
    If you want several states, rather call static_exchange_diagonal directly
    """
    return float(static_exchange_diagonal(mf, mol, [p_state],
                                          dm_correction=dm_correction,
                                          exchange=exchange)[0])


def solve_qp_from_imaginary_axis(eps, p_state, xc_correction, z_fit, sigma_iw,
                                 greedy=True, solver_mode='pole_strength',
                                 max_order=None):
    """Pade-continue Sigma_c off the imaginary axis and solve the QP equation.

        w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w)

    eps           : KS orbitale energies
    p_state       : state of interest 
    xc_correction : static part of self-energy (corrected for DFT starting point)
    z_fit         : the sample points
    sigma_iw      : Sigma_c values on them
    greedy        : Default algorithm for analytical continuation
    solver_mode   : By default, look for solution with largest Z-factor.
    max_order     : cap on the number of Pade nodes, applied after the greedy
                    ordering. None keeps all of them, which is the T = 0 behaviour; pass
                    `matsubara.ir_continuation_order(beta, wmax)` for a metal.
    """
    z_ord, f_ord = z_fit, sigma_iw
    if greedy:
        order = greedy_pade_order(z_fit, sigma_iw)
        z_ord, f_ord = z_fit[order], sigma_iw[order]
    if max_order is not None and max_order < len(z_ord):
        # At finite temperature the imaginary-axis data can only resolve so
        # many independent structures and asking Pade for more nodes than
        # that is asking it to fit noise. Truncating AFTER the greedy ordering
        # keeps the most informative points, since that ordering puts them
        # first by construction.
        if max_order < 2:
            raise ValueError(
                f"max_order = {max_order} leaves too few points for a Pade "
                f"fit; at least two are needed.")
        z_ord, f_ord = z_ord[:max_order], f_ord[:max_order]
    pade_coeffs = thiele_coefficients(z_ord, f_ord)

    def residual(w):
        sigma_c_w = pade_eval(np.array([w], dtype=complex), z_ord, pade_coeffs)[0]
        return w - eps[p_state] - xc_correction - sigma_c_w.real

    return solve_qp_equation(residual, eps[p_state], method=solver_mode)


def imaginary_axis_sample_points(freq_points, nocc, p_state, mu):
    """The (z_fit, query) pair both routes sample on: the line Re z = mu, on the
    side of the gap the state sits on."""
    sign = -1.0 if p_state < nocc else 1.0
    iw_query = sign * np.asarray(freq_points)
    return mu + 1j * iw_query, iw_query
