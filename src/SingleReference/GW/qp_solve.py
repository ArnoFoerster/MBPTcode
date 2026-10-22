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

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.distributed_df import distributed_fock
from src.Base.environment import environment_of
from src.Base.utils.analyticalContinuation import (greedy_pade_order,
                                                   thiele_coefficients,
                                                   pade_eval)
from src.Base.utils.mpi_grid import (current_comm, lockstep,
                                     lockstep_mean_field)
from src.Base.utils.threads import blas_single_threaded
from src.Solvers.qp_equation import pole_strength, solve_qp_equation


def static_exchange_mean_field_matrix(mf, mol, dm_correction=None,
                                      exchange='mf', comm=None):
    """<p| Sigma_x - v_xc |q> from the mean field alone, before any environment
    term.

    THE OBJECT A DRIVER CACHES. It carries no state index and no spectrum: K
    and v_xc are functionals of the converged density and the orbitals, so one
    build serves a quasiparticle window, the whole BSE diagonal, and every
    cycle of an eigenvalue self-consistency, where only the eigenvalues move.
    Its cost is one K -- naux nao^2 nocc through the fit -- plus the
    exchange-correlation potential on the DFT grid, and on a hybrid the grid is
    the larger of the two. Neither part divides by the number of states
    asked for, which is what makes rebuilding it per window the waste it is.

    comm: defaults to `current_comm()`. Over more than one rank the matrix
        is rank 0's bit for bit on every rank, whichever way it is built.
        Through the slices a distributed SCF left on the mean field
        (`Base.distributed_df.distributed_fock`) every rank makes the same
        three pyscf calls it makes serially, each a collective that locksteps
        the density and adds every rank's rows of the fitted tensor and block
        of the grid. Without those handles, or for another `exchange`, it is
        the replicated build: each rank's own calls on rank 0's orbitals,
        occupations and `dm_correction`, locked at entry. The slices cost more
        to make than this whole stage, so a mean field converged one rank at a
        time gains nothing by paying for them here. Under a multi-rank context
        this is a COLLECTIVE in every branch: a caller running it on rank 0
        alone must do so under `distributed(None)`.

    The environment's static term is NOT included, so one cached matrix serves
    every environment; `static_exchange_matrix` adds the Eq. (18) shift it is
    handed.

    THE ONE BUILDER of the static term: the Casida, imaginary-frequency and
    space-time routes, the BSE diagonal and the gradient chain's forward all
    read it. Every piece is the mean field's own operator -- its J, its K
    (fitted or ISDF on a fitted reference), its v_xc -- so the Hartree term
    cancels exactly and the eigenvalues' own fitting error cancels against
    v_xc. Exact J and K against the fitted v_eff would instead leave
    (J_exact - J_fit) - (K_exact - K_fit)/2 in every quasiparticle energy.

    dm_correction: an AO 1RDM gamma' (a CCSD, MP2 or GW density) at which
        Sigma_Hx is taken, against v_Hxc at the mean-field density gamma:

            <p| J[gamma'] - J[gamma] - K[gamma']/2 - (v_eff[gamma] - J[gamma]) |q>,

        Hartree AND exchange, the static self-energy Sigma(infinity) of the
        correlated density. Unrestricted: (alpha, beta), each spin's K of its
        own block.

    Unrestricted mean fields return (2, nmo, nmo), one matrix per spin.

    `exchange` selects the K; see `static_exchange_matrix`.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        return _local_static_exchange(mf, mol, dm_correction, exchange)
    handles = None
    if exchange == 'mf':
        with distributed_fock(mf, comm, build=False) as handles:
            if handles is not None:
                out = _local_static_exchange(mf, mol, dm_correction, exchange)
    if handles is None:
        # A mean field converged on each rank holds that rank's own last bits.
        lockstep_mean_field(mf, comm)
        dm_correction = lockstep(dm_correction, comm)
        out = _local_static_exchange(mf, mol, dm_correction, exchange)
    # Each rank's own K, quadrature and final GEMM, even on rank 0's inputs,
    # are its own arithmetic; this makes the product rank 0's.
    return lockstep(out, comm)


def hartree_shift(mf, mol, dm_new, j_ref):
    """J[dm'] - J[dm]: how Sigma_H moves with the density, through the mean
    field's own Coulomb so that it vanishes at dm' = dm (`j_ref` = J[dm],
    total densities)."""
    return mf.get_j(mol, dm_new) - j_ref


def _local_static_exchange(mf, mol, dm_correction, exchange):
    """The matrix from this rank's own pyscf calls: K and v_xc built whole
    on the AO basis, then projected onto the MO basis.

    Serially that is the whole build; inside `distributed_fock` every Fock
    piece among them is a collective over the ranks.

    UNRESTRICTED: (2, nmo, nmo), each spin in its own orbitals. The exchange
    of a spin orbital is -K[D_s] of its own spin density alone, against
    -K[D]/2 of the total density when the two spins coincide, and v_xc_s is
    v_eff_s less the Coulomb potential of BOTH spin densities.
    """
    dm = mf.make_rdm1()
    unrestricted = np.ndim(dm) == 3
    dm_for_hx = dm if dm_correction is None else np.asarray(dm_correction)
    if unrestricted and np.shape(dm_for_hx) != np.shape(dm):
        raise ValueError(
            f'an unrestricted mean field needs dm_correction as (alpha, beta) '
            f'of shape {np.shape(dm)}, got {np.shape(dm_for_hx)}')
    # K and the exchange-correlation potential are pyscf's own OpenMP -- the
    # fit's contraction and `nr_rks` on the DFT grid -- so BLAS is held at one
    # thread across them and the two pools stop spinning against each other.
    with blas_single_threaded():
        K = _exchange_k(mf, mol, dm_for_hx, exchange)
        sig_x = -K if unrestricted else -0.5 * K
        dm_total = dm[0] + dm[1] if unrestricted else dm
        v_eff = mf.get_veff(mol, dm)
        j_mf = mf.get_j(mol, dm_total)
        op = sig_x - (v_eff - j_mf)
        if dm_correction is not None:
            dm_x_total = (dm_for_hx[0] + dm_for_hx[1] if unrestricted
                          else dm_for_hx)
            op = op + hartree_shift(mf, mol, dm_x_total, j_mf)
    mo = mf.mo_coeff
    if unrestricted:
        return np.stack([mo[s].T @ op[s] @ mo[s] for s in (0, 1)])
    return mo.T @ op @ mo                                # a GEMM: BLAS wide


def refuse_unshifted_environment(mf, mol, where):
    """Refuse an environment on `mf` that reached the static term without its
    Eq. (18) shift, unless the caller adds that shift to the root itself.

    Every route into `static_exchange_matrix` screens Sigma_c bare and carries
    the reaction field as Eq. (18) (Duchemin, Guido, Jacquemin, Blase, Chem.
    Sci. 9, 4430 (2018)): inside the static term on an evGW cycle, after the
    solve in G0W0 (`reaction_field.shift_after_solve`). Without either the
    only static term left is the COHSEX operator, which is a different one.
    Either kind of environment is refused: one that dresses the interaction,
    and one that offers a COHSEX operator without dressing it.
    """
    environment = environment_of(mf)
    if not (getattr(environment, 'screens', True)
            or environment.static_self_energy(mf, mol) is not None):
        return
    raise ValueError(
        f'{where} was reached with reaction_field=None on a mean field '
        f'carrying {environment!r}. A GW route that builds W screens Sigma_c '
        f'with the bare interaction and passes the continuum as Eq. (18) of '
        f'Duchemin et al., Chem. Sci. 9, 4430 (2018): '
        f'`reaction_field.environment_quasiparticle_shift`, or '
        f'`separable_quasiparticle_shift` on the route\'s own factors, inside '
        f'the static term on an evGW cycle, or added to the root after a '
        f'G0W0 solve (reaction_field_after_solve=True, `add_reaction_field`). '
        f'The '
        f'static COHSEX operator `cohsex_correction` is not that shift and is '
        f'not added here. A route that forms no W reads '
        f'`solvent_static_selfenergy` itself; the environment-free matrix is '
        f'`static_exchange_mean_field_matrix`.')


def _exchange_k(mf, mol, dm, exchange):
    """K[dm] by the build `exchange` names; dm may carry a leading spin axis."""
    if exchange == 'mf':
        return mf.get_k(mol, dm)
    if exchange == 'exact':
        return _pyscf_scf.hf.get_jk(mol, dm, hermi=1, with_j=False)[1]
    if exchange == 'df':
        aux = getattr(mf.with_df, 'auxbasis', None) or paired_auxbasis(mol.basis)
        return _pyscf_df.DF(mol, auxbasis=aux).get_jk(dm, hermi=1,
                                                      with_j=False)[1]
    raise ValueError(f"exchange must be 'mf', 'exact' or 'df', got {exchange!r}")


def _unrestricted_xc_potential(mf, mol, dm):
    """(2, nao, nao) v_xc_s = veff_s - J[D_alpha + D_beta], exact exchange
    included: the operator the unrestricted eigenvalues came from, less the
    Hartree term every spin sees alike."""
    vj = mf.get_j(mol, dm)
    return np.asarray(mf.get_veff(mol, dm)) - (vj[0] + vj[1])[None]


def _refuse_both_placements(reaction_field, after_solve):
    """Eq. (18) enters a quasiparticle energy once: in the static term or
    after the solve, never both."""
    if after_solve and reaction_field is not None:
        raise ValueError(
            'reaction_field_after_solve=True says the caller adds Eq. (18) to '
            'the root after the solve, so a reaction_field handed in as well '
            'would enter the quasiparticle energy twice')


def add_reaction_field(roots, reaction_field, states, spin=None):
    """G0W0 roots with Duchemin's Eq. (18) added WHOLE, after the solve.

    roots: one quasiparticle energy per state of `states` (or a scalar for a
        scalar state), solved from the static term WITHOUT the shift.
    reaction_field: the Eq. (18) array, (nmo,) or (2, nmo) with `spin`
        naming the channel; None (gas phase) returns `roots` unchanged.

    Eq. (18) is a static potential on the added charge, nearly the same for
    every occupied (+1/2 <ii|Delta W|ii>) and every virtual orbital, and a
    constant potential moves a level by its full value. Inside the G0W0
    equation w = eps_p + <Sigma_x - v_xc>_pp + shift_p + Sigma_c(w) the
    fixed Sigma_c(w) renormalizes it to Z_p x shift_p, and the (1 - Z)(P_e
    + P_h) left over is the 130-550 meV spurious blue shift of the solvated
    BSE (Duchemin's Eqs. (18)-(20) need the full shift to cancel the reaction
    field of the direct kernel). An evGW cycle keeps it in the equation,
    where G and W are built on levels that carry it.
    """
    if reaction_field is None:
        return roots
    shift = np.asarray(reaction_field, float)
    if spin is not None:
        shift = shift[spin]
    return roots + shift[np.atleast_1d(states).astype(int)].reshape(
        np.shape(roots))


def static_exchange_matrix(mf, mol, dm_correction=None, exchange='mf',
                           reaction_field=None, sigma_x_matrix=None,
                           comm=None, reaction_field_after_solve=False):
    """
    <p| Sigma_x - v_xc |q> over the whole MO basis.
    Zero by construction on a Hartree-Fock reference
    dm_correction: an AO 1RDM at which Sigma_Hx is taken (Hartree and
        exchange); see `static_exchange_mean_field_matrix`.
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

    reaction_field: the continuum's Eq. (18) shift of Duchemin, Guido,
        Jacquemin and Blase, Chem. Sci. 9, 4430 (2018), one value per
        orbital, added on the diagonal. Every route that reaches this matrix
        builds W and screens Sigma_c with the BARE interaction, so this is the
        whole of the reaction field it sees. It belongs here only on an evGW
        cycle; a G0W0 solve adds it to the root (`add_reaction_field`) and
        says so with reaction_field_after_solve=True, which builds the matrix
        without it. An environment attached to `mf` with reaction_field=None
        and no such declaration is REFUSED (`refuse_unshifted_environment`):
        the static COHSEX operator `SolventScreening.cohsex_correction` sums
        the bare vtilde over every orbital and is not Eq. (18), so it is never
        added here. The gas phase adds nothing.

    Unrestricted, reaction_field is the (2, nmo) Eq. (18) array and the
    matrix is (2, nmo, nmo), one per spin.

    sigma_x_matrix: a `static_exchange_mean_field_matrix` built earlier for
        this mean field, in place of rebuilding K and v_xc. The Eq. (18)
        term is still applied here, so the cached matrix stays environment-free
        and one of them serves the gas phase and every continuum alike.

    comm: builds that matrix over the ranks; see
        `static_exchange_mean_field_matrix`. The environment term is added on
        every rank afterwards, from the same replicated cavity, so it needs no
        collective of its own. A handed-in `sigma_x_matrix` enters no
        collective at all.
    """
    comm = current_comm() if comm is None else comm
    _refuse_both_placements(reaction_field, reaction_field_after_solve)
    if reaction_field is None and not reaction_field_after_solve:
        refuse_unshifted_environment(mf, mol, 'static_exchange_matrix')
    if sigma_x_matrix is None:
        out = static_exchange_mean_field_matrix(mf, mol,
                                                dm_correction=dm_correction,
                                                exchange=exchange, comm=comm)
    else:
        out = np.asarray(sigma_x_matrix, float)
    if reaction_field is None:
        return out
    shift = np.asarray(reaction_field, float)
    if out.ndim == 3:
        if shift.shape != out.shape[:2]:
            raise ValueError(
                f'an unrestricted static term takes one Eq. (18) value per '
                f'spin orbital, shape {out.shape[:2]}; got {shift.shape}')
        return out + np.array([np.diag(s) for s in shift])
    return out + np.diag(shift)


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
                             block_memory_gb=None, reaction_field=None,
                             sigma_x_matrix=None, comm=None, spin=None,
                             reaction_field_after_solve=False):
    """<p| Sigma_x - v_xc |p> for the requested states only.

    spin: on an unrestricted reference, the channel (0 alpha, 1 beta) the
        states belong to; required there and refused on a restricted one.
        `reaction_field` and `sigma_x_matrix` are then the whole (2, ...)
        arrays and are indexed here.

    reaction_field is the continuum's whole static term: Duchemin et al.'s
    Eq. (18) shift, the self-polarization of the orbital carrying the added
    charge in the SCREENED reaction field, which every route here can form
    because it has built W. With an environment attached it is required
    (`refuse_unshifted_environment`), here on an evGW cycle or after the
    solve in G0W0 (reaction_field_after_solve=True: the diagonal without it,
    `add_reaction_field` on the root). A route that never forms W (ADC, whose
    secular matrix starts at Sigma^(2)) reads `cohsex_correction`, the same
    physics with the BARE vtilde summed over every orbital, itself and does
    not come here; applying both counts the polarization energy twice -- on
    acrolein in water that is 2.65 + 2.35 eV of gap closure where 2.3 is
    right, which reads as plausible over-screening rather than as a bug.

    Every consumer of the static exchange on the QP routes needs diagonals for
    a few states, and that is what makes an exchange build that scales
    possible: 'df-direct' streams the three-index integrals once against the
    state vectors (`_df_direct_exchange_diagonal`) instead of storing them or
    forming a full K. The other choices go through `static_exchange_matrix`.
    v_xc is the SCF's own operator throughout; see `static_exchange_matrix`.

    THE STATES DO NOT DIVIDE THE COST on any route but 'df-direct': K and v_xc
    are built whole and then indexed, so a two-state window costs what the
    whole diagonal costs. `sigma_x_matrix` is the way out for a driver that
    needs both -- `static_exchange_mean_field_matrix` once, handed to every
    call -- and the diagonal it yields is the built one bit for bit, since
    indexing is all that is left to do.

    block_memory_gb: for 'df-direct', the integral slab per aux_e2 call. None
        takes a quarter of mf.max_memory.
    comm: builds the matrix over the ranks where the mean field carries the
        slices to do it with; see `static_exchange_mean_field_matrix`. Only
        the build is collective -- the diagonal is indexing, on every rank.
        'df-direct' streams its own integrals and takes no communicator.
    """
    comm = current_comm() if comm is None else comm
    states = np.atleast_1d(states).astype(int)
    unrestricted = isinstance(mf, _pyscf_scf.uhf.UHF)
    if unrestricted and spin not in (0, 1):
        raise ValueError(
            f'an unrestricted reference has one static exchange per spin; pass '
            f'spin=0 (alpha) or spin=1 (beta), got {spin!r}')
    if not unrestricted and spin is not None:
        raise ValueError(f'spin={spin!r} on a restricted reference')
    if sigma_x_matrix is not None and exchange == 'df-direct':
        raise ValueError(
            "exchange='df-direct' never forms <p|Sigma_x - v_xc|q>, so a "
            'handed-in sigma_x_matrix names a different static term than the '
            'one asked for')
    if exchange != 'df-direct':
        matrix = static_exchange_matrix(
            mf, mol, dm_correction=dm_correction, exchange=exchange,
            reaction_field=reaction_field,
            sigma_x_matrix=sigma_x_matrix, comm=comm,
            reaction_field_after_solve=reaction_field_after_solve)
        return np.diag(matrix[spin] if unrestricted else matrix)[states]
    _refuse_both_placements(reaction_field, reaction_field_after_solve)
    if reaction_field is None and not reaction_field_after_solve:
        refuse_unshifted_environment(mf, mol, 'static_exchange_diagonal')
    if unrestricted:
        return _df_direct_unrestricted(mf, mol, states, dm_correction,
                                       block_memory_gb, reaction_field, spin)
    dm = mf.make_rdm1()
    dm_for_hx = dm if dm_correction is None else dm_correction
    mo = mf.mo_coeff[:, states]
    aux = getattr(mf.with_df, 'auxbasis', None) or paired_auxbasis(mol.basis)
    if block_memory_gb is None:
        block_memory_gb = max(1.0, 0.25 * getattr(mf, 'max_memory', 4000) / 1e3)
    k_pp = _df_direct_exchange_diagonal(mol, aux, mo, dm_for_hx, block_memory_gb)
    v_eff = mf.get_veff(mol, dm)
    j_mf = mf.get_j(mol, dm)
    v_xc = v_eff - j_mf
    if dm_correction is not None:
        v_xc = v_xc - hartree_shift(mf, mol, dm_correction, j_mf)
    out = -0.5 * k_pp - np.einsum('mp,mn,np->p', mo, v_xc, mo, optimize=True)
    if reaction_field is None:
        return out
    return out + np.asarray(reaction_field, float)[states]


def _df_direct_unrestricted(mf, mol, states, dm_correction, block_memory_gb,
                            reaction_field, spin):
    """'df-direct' for spin `spin`: -K[D_s]_pp from one streamed pass against
    the channel's own state vectors, less v_xc,s."""
    dm = mf.make_rdm1()
    dm_for_hx = dm if dm_correction is None else np.asarray(dm_correction)
    if dm_for_hx.shape != dm.shape:
        raise ValueError(
            f'an unrestricted dm_correction is one density per spin, shape '
            f'{dm.shape}; got {dm_for_hx.shape}')
    mo = mf.mo_coeff[spin][:, states]
    aux = getattr(mf.with_df, 'auxbasis', None) or paired_auxbasis(mol.basis)
    if block_memory_gb is None:
        block_memory_gb = max(1.0, 0.25 * getattr(mf, 'max_memory', 4000) / 1e3)
    k_pp = _df_direct_exchange_diagonal(mol, aux, mo, dm_for_hx[spin],
                                        block_memory_gb)
    v_xc = _unrestricted_xc_potential(mf, mol, dm)[spin]
    if dm_correction is not None:
        vj = mf.get_j(mol, dm[0] + dm[1])
        v_xc = v_xc - hartree_shift(mf, mol, dm_for_hx[0] + dm_for_hx[1], vj)
    out = -k_pp - np.einsum('mp,mn,np->p', mo, v_xc, mo, optimize=True)
    if reaction_field is None:
        return out
    return out + np.asarray(reaction_field, float)[spin][states]


def solve_qp_from_imaginary_axis(eps, p_state, xc_correction, z_fit, sigma_iw,
                                 greedy=True, solver_mode='pole_strength',
                                 max_order=None, with_z=False):
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
    with_z        : return (root, Z), Z = 1/(1 - dRe Sigma_c/dw) of the
                    continued fraction at the root, NaN where it is flat.
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

    root = solve_qp_equation(residual, eps[p_state], method=solver_mode)
    if not with_z:
        return root
    return root, float(pole_strength(residual, root))


def imaginary_axis_sample_points(freq_points, nocc, p_state, mu):
    """The (z_fit, query) pair both routes sample on: the line Re z = mu, on the
    side of the gap the state sits on."""
    sign = -1.0 if p_state < nocc else 1.0
    iw_query = sign * np.asarray(freq_points)
    return mu + 1j * iw_query, iw_query
