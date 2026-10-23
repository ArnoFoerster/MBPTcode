"""Static (frequency-independent) corrections to the ADC(3) F block."""
import numpy as np
from pyscf import scf

from src.Base.numerics.constants import MPN_DENSITY_LAPLACE_NTAU
from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_block_eri,
                                            get_antisymmetrized_spin_eri, get_orbital_energies,
                                            get_two_electron_integrals_chemist,
                                            get_uhf_spin_orbital_arrays_blockstacked,
                                            get_uhf_spin_orbital_df_factor_blockstacked,
                                            uhf_blockstacked_order)
from src.environments.solvent_screening import solvent_static_selfenergy
from src.SingleReference.CC.pipeline import (compute_ccsd_density_matrix,
                                             compute_ccsd_density_matrix_uhf,
                                             compute_ccsdt_density_matrix)
from src.SingleReference.DensityMatrix.density_matrix import (
    build_dgamma_block, semicanonicalize_restricted, semicanonicalize_uhf,
    solve_cphf_relaxation, solve_cphf_relaxation_uhf, t1_singles_blocks)
from src.SingleReference.DensityMatrix.generated_mpn import (
    mpn_density_pieces as mpn_gen)
from src.SingleReference.DensityMatrix.mpn_density_driver import (
    g_vvoo_df, t1_2_numerator_df)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted, compute_delta_gamma2_df_streamed)
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import (
    MPnDensityDriverUnrestricted)
from src.SingleReference.EpsteinNesbet import (build_dressed_e_ai, build_dressed_e_abij,
                                                build_dressed_denoms_uhf,
                                                restricted_channel_shifts,
                                                validate_en_dress)
from src.SingleReference.EpsteinNesbet.shifts import (
    diag_ladder_df, diag_ladder_df_screened, epstein_nesbet_denominator)
from src.SingleReference.LinearResponse.linear_response import (
    static_screened_coulomb_aux_uhf)
from src.SingleReference.ADC.adc_r_utils import dress_is_streamable
from src.SingleReference.common.mp_amplitudes import orbital_denominator, to_l


def diagonalize_static_only(eps_spin, static_correction=None):
    """Diagonalize just diag(eps_spin) + static_correction (eigh: ascending)."""
    H = np.diag(eps_spin) if static_correction is None else np.diag(eps_spin) + static_correction
    return np.linalg.eigh(H)


def _embed_active(oo_act, ov_act, vv, nocc, ncore):
    """
    Zero-pad active-occ-only (oo, ov) blocks into full nocc-sized arrays.
    """
    if ncore == 0:
        return oo_act, ov_act, vv
    full_oo = np.zeros((nocc, nocc))
    full_oo[ncore:, ncore:] = oo_act
    full_ov = np.zeros((nocc, vv.shape[0]))
    full_ov[ncore:, :] = ov_act
    return full_oo, full_ov, vv


def _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b, nocc_a, nocc_b, norb_a, norb_b):
    """
    Assemble the block-stacked spin-orbital Delta_gamma matching
    get_uhf_spin_orbital_arrays_blockstacked's ordering, from two full spatial
    (norb_a,norb_a)/(norb_b,norb_b) MO-basis density corrections.
    """
    order = uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b)
    nso = norb_a + norb_b
    dgamma_unordered = np.zeros((nso, nso))
    dgamma_unordered[0:norb_a, 0:norb_a] = dgamma_a
    dgamma_unordered[norb_a:, norb_a:] = dgamma_b
    return dgamma_unordered[np.ix_(order, order)]


def _spin_orbital_static_correction(g_anti_spin, dgamma_spatial):
    """G[Delta_gamma] in spin-orbital form from a spatial-MO Delta_gamma (RHF
    interleaved alpha/beta convention: each spin channel gets half of the
    combined spatial density correction). Shared by build_mp2/mp3_static_correction's
    RHF branch and _static_correction_from_ao_density.
    """
    nmo = dgamma_spatial.shape[0]
    dgamma_spin = np.zeros((2 * nmo, 2 * nmo))
    dgamma_spin[0::2, 0::2] = dgamma_spatial / 2.0
    dgamma_spin[1::2, 1::2] = dgamma_spatial / 2.0
    return np.einsum('prqs,rs->pq', g_anti_spin, dgamma_spin)


def _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial):
    """
    Calculate static self-energy correction due to delta gamma, spatial
    """
    J = np.einsum('pqrs,rs->pq', eri_chemist, dgamma_spatial)
    K = np.einsum('prqs,rs->pq', eri_chemist, dgamma_spatial)
    return J - 0.5 * K


def _static_correction_from_dgamma_restricted_df(B_aa, dgamma_spatial):
    """
    DF/RI variant of _static_correction_from_dgamma_restricted
    """
    v = np.einsum('Qrs,rs->Q', B_aa, dgamma_spatial, optimize=True)
    J = np.einsum('Qpq,Q->pq', B_aa, v, optimize=True)
    W = np.einsum('Qqs,rs->Qrq', B_aa, dgamma_spatial, optimize=True)
    K = np.einsum('Qpr,Qrq->pq', B_aa, W, optimize=True)
    return J - 0.5 * K


def _ks_semicanonical_setup(mf, mol, nocc, ncore):
    """RHF/RKS: if mf is a KS object, semicanonicalize it (see
    density_matrix.semicanonicalize_restricted) and return everything needed to (a)
    evaluate the existing MP2/MP3DensityMatrixSolverUnrestricted machinery in the
    resulting basis and (b) fold in the new leading-order T1 singles contribution
    (density_matrix.t1_singles_blocks) -- see build_mp2_static_correction's KS note.

    Returns None if mf is not a KS object (no-op: canonical HF, nothing to do).
    Otherwise returns (mf_semi, U_oo, U_vv, t1_ov, dgamma_oo_t1, dgamma_vv_t1):
    mf_semi is a shallow copy of mf with mo_coeff/mo_energy replaced by the
    semicanonical ones (so get_orbital_energies/get_antisymmetrized_spin_block_eri
    called on it transparently give semicanonical-basis quantities); t1_ov/
    dgamma_oo_t1/dgamma_vv_t1 are active-space-sized (ncore already excluded, matching
    _embed_active's convention).
    """

    if not hasattr(mf, 'xc'):
        return None

    mo_coeff_semi, eps_semi, f_ov_semi, U_oo, U_vv = semicanonicalize_restricted(mf, mol, nocc=nocc)

    # PySCF's own shallow copy: copy.copy runs the pickle hooks, which drop the
    # direct-SCF optimizer that a J/K build on a large molecule reads
    mf_semi = mf.copy()
    mf_semi.mo_coeff = mo_coeff_semi
    mf_semi.mo_energy = eps_semi

    eps_o_act, eps_v = eps_semi[ncore:nocc], eps_semi[nocc:]
    f_ov_act = f_ov_semi[ncore:, :]
    t1_ov, dgamma_oo_t1, _, dgamma_vv_t1 = t1_singles_blocks(f_ov_act, eps_o_act, eps_v)

    return mf_semi, U_oo, U_vv, t1_ov, dgamma_oo_t1, dgamma_vv_t1


def _ks_semicanonical_setup_uhf(mf, mol, ncore):
    """UHF/UKS counterpart of _ks_semicanonical_setup, per spin channel.

    Returns None if mf is not a KS object. Otherwise returns
    (mf_semi, U_oo_a, U_vv_a, t1_ov_a, dgamma_oo_t1_a, dgamma_vv_t1_a,
             U_oo_b, U_vv_b, t1_ov_b, dgamma_oo_t1_b, dgamma_vv_t1_b)
    -- mf_semi is a shallow copy of mf with mo_coeff/mo_energy replaced by the
    per-spin semicanonical ones; the t1/dgamma pieces are active-space-sized.
    """

    if not hasattr(mf, 'xc'):
        return None

    (mo_a_semi, eps_a_semi, f_ov_a_semi, U_oo_a, U_vv_a,
     mo_b_semi, eps_b_semi, f_ov_b_semi, U_oo_b, U_vv_b) = semicanonicalize_uhf(mf, mol)

    mf_semi = mf.copy()  # keeps the direct-SCF optimizer, as above
    mf_semi.mo_coeff = (mo_a_semi, mo_b_semi)
    mf_semi.mo_energy = (eps_a_semi, eps_b_semi)

    nocc_a, nocc_b = mf.nelec
    t1_ov_a, dgamma_oo_t1_a, _, dgamma_vv_t1_a = t1_singles_blocks(
        f_ov_a_semi[ncore:, :], eps_a_semi[ncore:nocc_a], eps_a_semi[nocc_a:])
    t1_ov_b, dgamma_oo_t1_b, _, dgamma_vv_t1_b = t1_singles_blocks(
        f_ov_b_semi[ncore:, :], eps_b_semi[ncore:nocc_b], eps_b_semi[nocc_b:])

    return (mf_semi, U_oo_a, U_vv_a, t1_ov_a, dgamma_oo_t1_a, dgamma_vv_t1_a,
            U_oo_b, U_vv_b, t1_ov_b, dgamma_oo_t1_b, dgamma_vv_t1_b)


def _rotate_dgamma_to_original_basis(dgamma_semi, U_oo, U_vv, nocc):
    """Rotate a full (nmo,nmo) spatial-MO Delta_gamma built in the semicanonical
    basis (see _ks_semicanonical_setup) back to mf's own (canonical KS) MO basis --
    needed so it can be added directly to build_ks_static_correction's matrix and
    contracted with mf's own (non-semicanonical) ERI. Delta_gamma transforms as a
    matrix under U = block_diag(U_oo, U_vv), since semicanonicalization only mixes
    within the occ and within the virt subspace separately.
    """
    nmo = dgamma_semi.shape[0]
    U = np.zeros((nmo, nmo))
    U[:nocc, :nocc] = U_oo
    U[nocc:, nocc:] = U_vv
    return U @ dgamma_semi @ U.T


def _mpn_dgamma_spatial(mf, mol, nocc, order, relax, ncore, eri_chemist=None, B_aa=None,
                        u2_denom_dress=None, amplitudes=None,
                        laplace_ntau=MPN_DENSITY_LAPLACE_NTAU):
    """RHF-only: the MPn Delta_gamma = Delta_gamma^(2) + ... + Delta_gamma^(order)
    in the spatial MO basis (order 2, 3 or 4), the density whose G[Delta_gamma]
    is Sigma(infinity) through order+1; shared by every restricted and
    spin-orbital RHF builder. Correlation blocks come from the generated
    restricted pipeline (MPnDensityDriverRestricted, Laplace-fused order-3/4
    triples at laplace_ntau); x2 for alpha+beta.

    A KS reference is semicanonicalized (_ks_semicanonical_setup) and the
    leading-order T1 singles contribution is added once to the total (it is
    first order, so not per order; see density_matrix.t1_singles_blocks).
    relax=True solves the CPHF/CPKS Z-vector equation for the combined ov
    block (solve_cphf_relaxation; in the semicanonical basis for KS, then
    rotated back). Through MP2 that is the relaxed density; from MP3 on it
    relaxes the orbitals only (no amplitude response), which is why it misses
    the finite-field dipole (tests/test_mp3_finite_field.py); a KS reference
    with relax=True is refused from order 3.

    eri_chemist: mf's own spatial chemist ERI when the caller has it (reused
    only for a plain-HF reference). B_aa: the DF factor; with it and a plain
    HF reference the density runs g-free (orders 2 and 3). u2_denom_dress:
    the Epstein-Nesbet dressing of the T2^(1)/T1^(2) denominators (orders 2
    and 3). amplitudes: a foreign amplitude pool (cc_amplitudes_restricted's
    keying 't2_aaaa', 't2_abab', 't1_aa') replacing the MBPT T2^(1)/T1^(2) in
    Delta_gamma^(2) (order 2 only): an expectation-value density with bra =
    ket adjoint, the convention of the CC-ADC(3) supermatrix, not the CC
    response 1-RDM (build_ccsd_static_correction_restricted). Mutually
    exclusive with u2_denom_dress, refused for KS; the streamed DF path is
    skipped (it generates T2^(1) from B_aa, which is what is overridden).

    Order 2 with B_aa, a plain-HF reference and a streamable dressing (bare,
    or hh/pp spin-adapted) is rank-4-free end to end: T2^(1) streams from
    B_aa (compute_delta_gamma2_df_streamed), the EN shift enters as the small
    dh/dp matrices. 'hp'/spin-resolved dressing takes the materialized
    driver, the same exception as the ADC solver's dress_is_streamable gate."""
    if order not in (2, 3, 4):
        raise ValueError(f"order={order}: the MPn density is written for orders 2, 3, 4")
    if order == 4 and (relax or B_aa is not None or u2_denom_dress is not None):
        raise NotImplementedError(
            "the MP4 density is unrelaxed, dense and undressed only (an MP4 orbital "
            "relaxation needs a Lambda-based two-particle density)")
    if amplitudes is not None and order != 2:
        raise ValueError("amplitudes= replaces the second-order amplitudes; order 2 only")

    ks_setup = _ks_semicanonical_setup(mf, mol, nocc, ncore)
    if ks_setup is not None and relax and order >= 3:
        raise NotImplementedError(
            f"MP{order} relax=True is not implemented for a KS reference (needs a "
            "semicanonical-basis CPHF/Z-vector solve) -- use relax=False.")
    mf_eval = ks_setup[0] if ks_setup is not None else mf
    nmo = mf_eval.mo_coeff.shape[1]
    eps = get_orbital_energies(mf_eval, representation='spatial')
    # frozen core: a [ncore:] window on eps/g with nocc-ncore occupied
    w = slice(ncore, None)
    O, V = nocc - ncore, nmo - nocc

    foreign_t2 = foreign_t1_2 = None
    if amplitudes is not None:
        if u2_denom_dress is not None:
            raise ValueError(
                "amplitudes= and u2_denom_dress= are mutually exclusive: the EN "
                "dressing renormalizes the MBPT denominators, and a converged CC "
                "amplitude has no denominator left to dress.")
        if ks_setup is not None:
            raise NotImplementedError(
                "amplitudes= is not supported for a KS reference (its T1 singles "
                "blocks are built from the MBPT amplitudes too).")
        missing = {'t2_aaaa', 't2_abab', 't1_aa'} - set(amplitudes)
        if missing:
            raise KeyError(f"amplitudes= is missing {sorted(missing)}; expected "
                           "cc_amplitudes_restricted's keying")
        foreign_t2 = (amplitudes['t2_aaaa'], amplitudes['t2_abab'])
        foreign_t1_2 = amplitudes['t1_aa']
        want2, want1 = (V, V, O, O), (V, O)
        for name, arr, want in (('t2_aaaa', foreign_t2[0], want2),
                                ('t2_abab', foreign_t2[1], want2),
                                ('t1_aa', foreign_t1_2, want1)):
            if tuple(arr.shape) != want:
                raise ValueError(
                    f"amplitudes[{name!r}] has shape {tuple(arr.shape)}, expected "
                    f"{want} for nocc={nocc} ncore={ncore} nmo={nmo}; a frozen-core "
                    "CC pool must be frozen with the same ncore as this correction.")

    df = B_aa is not None and ks_setup is None
    streamed = (order == 2 and df and amplitudes is None
                and dress_is_streamable(u2_denom_dress))

    e_abij = e_ai = None
    if u2_denom_dress is not None:
        if df:
            B_aa_w = B_aa[:, w, w]
            if not streamed:
                e_abij = build_dressed_e_abij(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
            e_ai = build_dressed_e_ai(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
        else:
            reusable_eri = eri_chemist if ks_setup is None else None
            if reusable_eri is None:
                reusable_eri = get_two_electron_integrals_chemist(mol, mf_eval, representation='spatial')
            reusable_eri_w = reusable_eri[w, w, w, w]
            e_abij = build_dressed_e_abij(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)
            e_ai = build_dressed_e_ai(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)

    if streamed:
        B_aa_w = B_aa[:, w, w]
        dh = dp = None
        if u2_denom_dress:
            d_h, d_p, _ = restricted_channel_shifts(
                u2_denom_dress, B_aa_w, None, None, O, np.arange(O, nmo - ncore))
            dh = d_h[0] if d_h is not None else None
            dp = d_p[0] if d_p is not None else None
        parts = [compute_delta_gamma2_df_streamed(B_aa_w, eps[w], O, dh=dh, dp=dp, e_ai=e_ai)]
    elif df:
        B_aa_w = B_aa[:, w, w]
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), None, None, None, O,
                                            B_aa=B_aa_w, B_bb=B_aa_w, e_abij=e_abij, e_ai=e_ai)
        if order == 2:
            parts = [driver.compute_delta_gamma2_df(t2_1=foreign_t2, t1_2_aa=foreign_t1_2)]
        else:
            parts = list(driver.compute_delta_gamma23_df(laplace_ntau=laplace_ntau))
    else:
        reusable_eri = eri_chemist if ks_setup is None else None
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval, eri_chemist=reusable_eri)
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), g_aaaa[w, w, w, w],
                                            g_abab[w, w, w, w], g_bbbb[w, w, w, w],
                                            O, e_abij=e_abij, e_ai=e_ai)
        if order == 2:
            parts = [driver.compute_delta_gamma2(
                t2_1=foreign_t2, order2=None if foreign_t1_2 is None else (foreign_t1_2,))]
        else:
            parts = list(driver.compute_delta_gamma23(laplace_ntau=laplace_ntau))
            if order == 4:
                parts.append(driver.compute_delta_gamma4(laplace_ntau=laplace_ntau))
    oo_a, ov_a, vv_a = parts[0]
    for oo_m, ov_m, vv_m in parts[1:]:
        oo_a, ov_a, vv_a = oo_a + oo_m, ov_a + ov_m, vv_a + vv_m
    oo_act, ov_act, vv_act = 2.0 * oo_a, 2.0 * ov_a, 2.0 * vv_a

    if ks_setup is not None:
        _, U_oo, U_vv, t1_ov, dgamma_oo_t1, dgamma_vv_t1 = ks_setup
        oo_act = oo_act + dgamma_oo_t1
        ov_act = ov_act + t1_ov
        vv_act = vv_act + dgamma_vv_t1

    dgamma_oo, dgamma_ov, dgamma_vv = _embed_active(oo_act, ov_act, vv_act, nocc, ncore)
    if relax:
        dgamma_full = solve_cphf_relaxation(mf_eval, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    else:
        dgamma_full = build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    if ks_setup is not None:
        dgamma_full = _rotate_dgamma_to_original_basis(dgamma_full, U_oo, U_vv, nocc)
    return dgamma_full


def _mpn_dgamma_uhf(mf, mol, order, relax, ncore, u2_denom_dress=None, cphf_level_shift=0.0,
                    cphf_max_cycle=None, cphf_tol=None):
    """UHF: (dgamma_a, dgamma_b), the MP2 (order 2) or MP2+MP3 (order 3)
    Delta_gamma per spin in the spatial MO basis, from the generated
    unrestricted pipeline (MPnDensityDriverUnrestricted); frozen core is a
    [ncore:] window per spin channel. KS references are semicanonicalized and
    get the T1 singles blocks, as in _mpn_dgamma_spatial. relax=True solves
    the UHF CPHF/CPKS Z-vector equation (solve_cphf_relaxation_uhf;
    cphf_level_shift/max_cycle/tol forwarded, None keeping its defaults);
    from MP3 on it relaxes the orbitals only, as on the RHF route."""
    ks_setup = _ks_semicanonical_setup_uhf(mf, mol, ncore)
    mf_eval = ks_setup[0] if ks_setup is not None else mf
    nocc_a, nocc_b = mf.nelec
    eps_a, eps_b = get_orbital_energies(mf_eval, representation='spatial')
    norb_a, norb_b = len(eps_a), len(eps_b)
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval)
    w = slice(ncore, None)
    custom_denom = None
    if u2_denom_dress is not None:
        custom_denom = build_dressed_denoms_uhf(u2_denom_dress, eps_a, eps_b, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b, ncore, mol=mol, mf_eval=mf_eval)
    driver = MPnDensityDriverUnrestricted(np.diag(eps_a[w]), np.diag(eps_b[w]),
                                          g_aaaa[w, w, w, w], g_abab[w, w, w, w],
                                          g_bbbb[w, w, w, w], nocc_a - ncore, nocc_b - ncore,
                                          custom_denom=custom_denom)
    if order == 2:
        oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = driver.compute_delta_gamma2()
    elif order == 3:
        (oo2_a, oo2_b, ov2_a, ov2_b, vv2_a, vv2_b), (oo3_a, oo3_b, ov3_a, ov3_b, vv3_a, vv3_b) = \
            driver.compute_delta_gamma23()
        oo_a, ov_a, vv_a = oo2_a + oo3_a, ov2_a + ov3_a, vv2_a + vv3_a
        oo_b, ov_b, vv_b = oo2_b + oo3_b, ov2_b + ov3_b, vv2_b + vv3_b
    else:
        raise NotImplementedError(
            f"order={order}: the unrestricted density driver has orders 2 and 3")

    if ks_setup is not None:
        (_, U_oo_a, U_vv_a, t1_ov_a, dgamma_oo_t1_a, dgamma_vv_t1_a,
         U_oo_b, U_vv_b, t1_ov_b, dgamma_oo_t1_b, dgamma_vv_t1_b) = ks_setup
        oo_a, ov_a, vv_a = oo_a + dgamma_oo_t1_a, ov_a + t1_ov_a, vv_a + dgamma_vv_t1_a
        oo_b, ov_b, vv_b = oo_b + dgamma_oo_t1_b, ov_b + t1_ov_b, vv_b + dgamma_vv_t1_b

    dgamma_oo_a, dgamma_ov_a, dgamma_vv_a = _embed_active(oo_a, ov_a, vv_a, nocc_a, ncore)
    dgamma_oo_b, dgamma_ov_b, dgamma_vv_b = _embed_active(oo_b, ov_b, vv_b, nocc_b, ncore)
    if relax:
        cphf_kwargs = {'level_shift': cphf_level_shift}
        if cphf_max_cycle is not None:
            cphf_kwargs['max_cycle'] = cphf_max_cycle
        if cphf_tol is not None:
            cphf_kwargs['tol'] = cphf_tol
        dgamma_a, dgamma_b = solve_cphf_relaxation_uhf(
            mf_eval, nocc_a, nocc_b,
            dgamma_oo_a, dgamma_ov_a, dgamma_vv_a,
            dgamma_oo_b, dgamma_ov_b, dgamma_vv_b, **cphf_kwargs)
    else:
        dgamma_a = build_dgamma_block(norb_a, nocc_a, dgamma_oo_a, dgamma_ov_a, dgamma_vv_a)
        dgamma_b = build_dgamma_block(norb_b, nocc_b, dgamma_oo_b, dgamma_ov_b, dgamma_vv_b)
    if ks_setup is not None:
        dgamma_a = _rotate_dgamma_to_original_basis(dgamma_a, U_oo_a, U_vv_a, nocc_a)
        dgamma_b = _rotate_dgamma_to_original_basis(dgamma_b, U_oo_b, U_vv_b, nocc_b)
    return dgamma_a, dgamma_b


def build_mpn_static_correction(mf, order, mol=None, nocc=None, relax=False, ncore=0,
                                u2_denom_dress=None, B_aa=None, amplitudes=None,
                                restricted=False, laplace_ntau=MPN_DENSITY_LAPLACE_NTAU,
                                cphf_level_shift=0.0, cphf_max_cycle=None, cphf_tol=None):
    """The generalized-Fock 'static' correction to the ADC F block from the MPn
    1-RDM, Delta_gamma = Delta_gamma^(2) + ... + Delta_gamma^(order): F is
    linear in the density, so F[gamma_HF + Delta_gamma] = diag(eps) +
    G[Delta_gamma], Sigma(infinity) through order+1 (ADC(n) needs the
    MP(n-1) density, static_kind_for_level).

    restricted=False: the (nso, nso) spin-orbital G[Delta_gamma]_pq =
    sum_rs gamma_rs <pr||qs>, symmetric -- RHF interleaved alpha/beta
    (nso = 2 nmo), UHF block-stacked [occ_a, occ_b, virt_a, virt_b] (the
    ordering in which the occupied orbitals are contiguous for nocc_a !=
    nocc_b). restricted=True (RHF only): the spatial (nmo, nmo) J - K/2 form
    for ADCSolverRestricted, g-free with B_aa.

    ncore: frozen spatial core orbitals (both spin channels for UHF). relax,
    u2_denom_dress, amplitudes, laplace_ntau: see _mpn_dgamma_spatial;
    cphf_*: the UHF Z-vector solve's options (_mpn_dgamma_uhf)."""
    mol = mol if mol is not None else mf.mol
    if isinstance(mf, scf.uhf.UHF):
        if restricted:
            raise NotImplementedError(
                f"the restricted MP{order} static correction is RHF-only -- use "
                "restricted=False (spin-orbital, block-stacked) for UHF.")
        if B_aa is not None or amplitudes is not None:
            raise ValueError("B_aa= and amplitudes= are restricted-branch options; "
                             "the UHF g-free route is build_mp2_static_correction_uhf_df")
        dgamma_a, dgamma_b = _mpn_dgamma_uhf(
            mf, mol, order, relax, ncore, u2_denom_dress=u2_denom_dress,
            cphf_level_shift=cphf_level_shift, cphf_max_cycle=cphf_max_cycle,
            cphf_tol=cphf_tol)
        nocc_a, nocc_b = mf.nelec
        _, g_anti_spin, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
        dgamma_spin = _uhf_dgamma_to_spin_blockstacked(
            dgamma_a, dgamma_b, nocc_a, nocc_b, dgamma_a.shape[0], dgamma_b.shape[0])
        return np.einsum('prqs,rs->pq', g_anti_spin, dgamma_spin)

    nocc = nocc if nocc is not None else mol.nelectron // 2
    if restricted and B_aa is not None:
        dgamma_spatial = _mpn_dgamma_spatial(mf, mol, nocc, order, relax, ncore, B_aa=B_aa,
                                             u2_denom_dress=u2_denom_dress,
                                             amplitudes=amplitudes, laplace_ntau=laplace_ntau)
        return _static_correction_from_dgamma_restricted_df(B_aa, dgamma_spatial)
    if B_aa is not None:
        raise ValueError("B_aa= is the restricted branch's DF route (restricted=True)")
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    dgamma_spatial = _mpn_dgamma_spatial(mf, mol, nocc, order, relax, ncore,
                                         eri_chemist=eri_chemist, u2_denom_dress=u2_denom_dress,
                                         amplitudes=amplitudes, laplace_ntau=laplace_ntau)
    if restricted:
        return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)
    return _spin_orbital_static_correction(get_antisymmetrized_spin_eri(eri_chemist),
                                           dgamma_spatial)


def build_mp2_static_correction(mf, mol=None, nocc=None, relax=False, ncore=0, u2_denom_dress=None,
                                cphf_level_shift=0.0, cphf_max_cycle=None, cphf_tol=None):
    """build_mpn_static_correction at order 2, spin-orbital (RHF or UHF)."""
    return build_mpn_static_correction(
        mf, 2, mol, nocc, relax=relax, ncore=ncore, u2_denom_dress=u2_denom_dress,
        cphf_level_shift=cphf_level_shift, cphf_max_cycle=cphf_max_cycle, cphf_tol=cphf_tol)


def build_mp2_static_correction_restricted(mf, mol=None, nocc=None, relax=False, ncore=0, B_aa=None,
                                           u2_denom_dress=None, amplitudes=None):
    """build_mpn_static_correction at order 2, restricted (RHF): the spatial J - K/2
    form, g-free with B_aa; amplitudes: the CC-ADC(3) amplitude substitution."""
    return build_mpn_static_correction(
        mf, 2, mol, nocc, relax=relax, ncore=ncore, u2_denom_dress=u2_denom_dress,
        B_aa=B_aa, amplitudes=amplitudes, restricted=True)


def build_mp3_static_correction(mf, mol=None, nocc=None, relax=False, ncore=0, u2_denom_dress=None):
    """build_mpn_static_correction at order 3, spin-orbital (RHF or UHF). relax=True
    relaxes the orbitals only (see _mpn_dgamma_spatial)."""
    return build_mpn_static_correction(mf, 3, mol, nocc, relax=relax, ncore=ncore,
                                       u2_denom_dress=u2_denom_dress)


def build_mp3_static_correction_restricted(mf, mol=None, nocc=None, relax=False, ncore=0, B_aa=None,
                                           u2_denom_dress=None):
    """build_mpn_static_correction at order 3, restricted (RHF), g-free with B_aa."""
    return build_mpn_static_correction(mf, 3, mol, nocc, relax=relax, ncore=ncore,
                                       u2_denom_dress=u2_denom_dress, B_aa=B_aa,
                                       restricted=True)


def build_mp4_static_correction_restricted(mf, mol=None, nocc=None, ncore=0,
                                           laplace_ntau=MPN_DENSITY_LAPLACE_NTAU):
    """build_mpn_static_correction at order 4, restricted (RHF): Sigma(infinity)
    through fifth order, ADC(5)'s orbital block; unrelaxed, dense only."""
    return build_mpn_static_correction(mf, 4, mol, nocc, ncore=ncore, restricted=True,
                                       laplace_ntau=laplace_ntau)


def build_mp4_static_correction(mf, mol=None, nocc=None, ncore=0,
                                laplace_ntau=MPN_DENSITY_LAPLACE_NTAU):
    """build_mpn_static_correction at order 4, spin-orbital (RHF)."""
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(
            "build_mp4_static_correction is RHF-only (the spin-orbital form of the "
            "restricted MP4 density); the unrestricted density driver has no "
            "order-4 Delta_gamma.")
    return build_mpn_static_correction(mf, 4, mol, nocc, ncore=ncore,
                                       laplace_ntau=laplace_ntau)


def static_kind_for_level(level):
    """The static-correction kind whose Sigma(infinity) is order-consistent
    with an ADC level: the orbital block of ADC(n) carries Sigma(infinity)
    through order n, and G[Delta_gamma^(m)] is of order m+1, so ADC(n) needs
    the MP(n-1) density. A level with no such builder (above ADC(5)) returns
    None."""
    return {'adc2': 'mp2_unrelaxed', 'adc2x': 'mp2_unrelaxed',
            'adc3': 'mp2_unrelaxed', 'adc4': 'mp3_unrelaxed',
            'adc4t2': 'mp3_unrelaxed', 'adc5': 'mp4_unrelaxed'}.get(level)


def _dgamma_spatial_from_ao_density(mf, dm_cc_ao):
    """RHF-only: fold a full correlated AO 1-RDM (e.g. from
    compute_ccsd_density_matrix/compute_ccsdt_density_matrix) down to the spatial
    MO basis and subtract the HF (diag(2,...,2,0,...,0)) reference. Shared by
    _static_correction_from_ao_density and the CCSD/CCSDT *_restricted builders.
    """
    mo = mf.mo_coeff
    S = mf.get_ovlp()
    nmo = mo.shape[1]
    nocc = int(round(np.trace(mf.make_rdm1() @ S) / 2))

    mo_inv = mo.T @ S
    dm_cc_mo = mo_inv @ dm_cc_ao @ mo_inv.T
    dm_hf_mo = np.diag([2.0] * nocc + [0.0] * (nmo - nocc))
    return dm_cc_mo - dm_hf_mo


def _dgamma_spatial_from_ao_density_uhf(mf, dm_a_ao, dm_b_ao):
    """UHF counterpart of _dgamma_spatial_from_ao_density: fold correlated AO
    1-RDMs (alpha, beta; e.g. from compute_ccsd_density_matrix_uhf) down to
    their spatial MO bases and subtract the UHF reference (diag(1,...,1,
    0,...,0) per spin channel, occupation nocc_a/nocc_b)."""
    mo_a, mo_b = mf.mo_coeff
    S = mf.get_ovlp()
    nocc_a, nocc_b = mf.nelec
    norb_a, norb_b = mo_a.shape[1], mo_b.shape[1]

    mo_inv_a = mo_a.T @ S
    mo_inv_b = mo_b.T @ S
    dm_a_mo = mo_inv_a @ dm_a_ao @ mo_inv_a.T
    dm_b_mo = mo_inv_b @ dm_b_ao @ mo_inv_b.T
    dm_hf_a = np.diag([1.0] * nocc_a + [0.0] * (norb_a - nocc_a))
    dm_hf_b = np.diag([1.0] * nocc_b + [0.0] * (norb_b - nocc_b))
    return dm_a_mo - dm_hf_a, dm_b_mo - dm_hf_b


def build_mp2_static_correction_uhf_df(mf, mol, B_so, relax=True, u2_denom_dress=None,
                                       cphf_level_shift=0.0, cphf_max_cycle=None, cphf_tol=None):
    """g-FREE UHF counterpart of build_mp2_static_correction: the same
    MP2-relaxed (optionally EN-dressed) G[Delta_gamma] blockstacked
    spin-orbital correction, built entirely from the blockstacked DF factor
    B_so (get_uhf_spin_orbital_df_factor_blockstacked) -- no O(norb^4)
    per-spin g blocks and no O(nso^4) g_anti_spin anywhere. Validated
    identical to the dense route at an exact DF factor
    (tests/test_uhf_static_correction_df.py).

    Route: instead of the per-spin MPnDensityDriverUnrestricted pipeline,
    the density is built in the blockstacked SPIN-ORBITAL picture (same
    physics, one channel), where the generated MP2 density blocks
    (m2_oo_11/m2_vv_11/m2_ov_02) are PURE AMPLITUDE contractions -- their g
    dependence flows solely through T2^(1) (g_vvoo_df) and T1^(2)
    (t1_2_numerator_df), both rank<=3 from B. EN dressing enters through
    epstein_nesbet_denominator's B= path, exactly as in
    adc_u_utils.dressed_t2_amplitudes. CPHF relaxation reuses
    solve_cphf_relaxation_uhf unchanged (mf.gen_response is AO-side and
    automatically DF when mf carries with_df). The final
    G_pq = sum_rs dgamma_rs <pr||qs> contraction is two rank<=3 B
    contractions (direct: B-weighted trace; exchange: B @ dgamma @ B per Q).

    Restrictions (raise): UHF only (no KS semicanonicalization), ncore=0.
    """

    if not isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("build_mp2_static_correction_uhf_df is UHF-only")
    if hasattr(mf, 'xc'):
        raise NotImplementedError(
            "build_mp2_static_correction_uhf_df: KS references (semicanonical "
            "setup) not supported -- use the dense build_mp2_static_correction")

    nocc_a, nocc_b = mf.nelec
    eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
    norb_a, norb_b = len(eps_a), len(eps_b)
    order = uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b)
    eps_spin = np.concatenate([eps_a, eps_b])[order]
    nocc = nocc_a + nocc_b
    nso = norb_a + norb_b
    nvirt = nso - nocc
    o, v = slice(0, nocc), slice(nocc, nso)

    # ---- T2^(1) (EN-dressed if requested) + T1^(2), all from B ----
    D_bare = orbital_denominator(eps_spin[o], eps_spin[v], 2)
    d1 = orbital_denominator(eps_spin[o], eps_spin[v], 1)
    if u2_denom_dress:
        channels = {k: val for k, val in u2_denom_dress.items()
                    if k in ('hh', 'pp', 'hp')}
        D = epstein_nesbet_denominator(D_bare, None, nocc, layout='pphh',
                                       bare_sign=1.0, B=B_so, **channels)
        # The dense reference (build_dressed_denoms_uhf) gates the SINGLES
        # denominator shift on the 'singles' key (default True, see
        # -- both routes used to apply it
        # unconditionally). Spin-orbital form: d1 + diag ladder <ai||ai>[a,i]
        # (cross-spin (a,i) elements are irrelevant: those t1 numerators
        # vanish by spin symmetry).
        singles = u2_denom_dress.get('singles', True)
        if singles == 'screened':
            # Reconstruct each spin's own (naux, norb_s, norb_s) DF factor in
            # CANONICAL (energy-ordered, occ-then-virt) layout by slicing it
            # back out of the blockstacked B_so -- cheap index gymnastics,
            # no second DF build. alpha occ sits at [0:nocc_a) and alpha virt
            # at [nocc:nocc+nvirt_a) in blockstacked order (see
            # uhf_blockstacked_order); beta fills the remaining two slices.
            nvirt_a = norb_a - nocc_a
            idx_a = np.concatenate([np.arange(0, nocc_a),
                                    np.arange(nocc, nocc + nvirt_a)])
            idx_b = np.concatenate([np.arange(nocc_a, nocc),
                                    np.arange(nocc + nvirt_a, nso)])
            B_a_full = B_so[:, idx_a][:, :, idx_a]
            B_b_full = B_so[:, idx_b][:, :, idx_b]
            W_aux = static_screened_coulomb_aux_uhf(eps_a, eps_b, B_a_full, B_b_full,
                                                     nocc_a, nocc_b)
            d1 = d1 + diag_ladder_df_screened(B_so, W_aux, v, o)
        elif singles:
            d1 = d1 + diag_ladder_df(B_so, v, o)
    else:
        D = D_bare
    t2_1 = g_vvoo_df(B_so, nocc) / D
    t1_2 = t1_2_numerator_df(B_so, nocc, t2_1) / d1

    # ---- Delta_gamma^(2) blocks: generated m2_* are amplitude-only (their g
    # parameter is unused -- verified against the generated source), so the
    # dense driver's compute_delta_gamma2 is reproduced verbatim with g=None.
    kd = np.eye(nso)
    args = dict(g=None, kd=kd, o=o, v=v, nv=nvirt, no=nocc)
    l2, l1 = to_l(t2_1, 2), to_l(t1_2, 1)
    N2 = mpn_gen.overlap2(l_amp=l2, t_amp=t2_1)
    dg_oo = (mpn_gen.m2_oo_11(**args, l2=l2, t2=t2_1)
             + mpn_gen.m2_oo_20(**args, l1=l1) + mpn_gen.m2_oo_02(**args, t1=t1_2)
             - N2 * kd[o, o])
    dg_vv = (mpn_gen.m2_vv_11(**args, l2=l2, t2=t2_1)
             + mpn_gen.m2_vv_20(**args, l1=l1) + mpn_gen.m2_vv_02(**args, t1=t1_2))
    dg_ov = (mpn_gen.m2_ov_11(**args, l2=l2, t2=t2_1)
             + mpn_gen.m2_ov_20(**args, l1=l1) + mpn_gen.m2_ov_02(**args, t1=t1_2))

    # ---- per-spin spatial blocks (blockstacked spin-orbital -> alpha/beta) ----
    oo_a, oo_b = dg_oo[:nocc_a, :nocc_a], dg_oo[nocc_a:, nocc_a:]
    nv_a = norb_a - nocc_a
    vv_a, vv_b = dg_vv[:nv_a, :nv_a], dg_vv[nv_a:, nv_a:]
    ov_a, ov_b = dg_ov[:nocc_a, :nv_a], dg_ov[nocc_a:, nv_a:]

    if relax:
        cphf_kwargs = {'level_shift': cphf_level_shift}
        if cphf_max_cycle is not None:
            cphf_kwargs['max_cycle'] = cphf_max_cycle
        if cphf_tol is not None:
            cphf_kwargs['tol'] = cphf_tol
        dgamma_a, dgamma_b = solve_cphf_relaxation_uhf(
            mf, nocc_a, nocc_b, oo_a, ov_a, vv_a, oo_b, ov_b, vv_b, **cphf_kwargs)
    else:
        dgamma_a = build_dgamma_block(norb_a, nocc_a, oo_a, ov_a, vv_a)
        dgamma_b = build_dgamma_block(norb_b, nocc_b, oo_b, ov_b, vv_b)

    dgamma_spin = _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b,
                                                   nocc_a, nocc_b, norb_a, norb_b)

    # ---- G_pq = sum_rs dgamma_rs <pr||qs>, rank<=3 from B ----
    tQ = np.einsum('Qrs,rs->Q', B_so, dgamma_spin, optimize=True)
    direct = np.einsum('Qpq,Q->pq', B_so, tQ, optimize=True)
    M = np.einsum('Qrq,rs->Qqs', B_so, dgamma_spin, optimize=True)
    exchange = np.einsum('Qps,Qqs->pq', B_so, M, optimize=True)
    return direct - exchange


def _static_correction_from_ao_density(mf, dm_cc_ao, g_anti_spin):
    """Shared G[Delta_gamma] builder: dm_cc_ao is the full correlated AO 1-RDM (e.g.
    from compute_ccsd_density_matrix/compute_ccsdt_density_matrix); Delta_gamma is
    obtained by folding it down to the spatial MO basis and subtracting the HF
    (diag(2,...,2,0,...,0)) reference, then expanding to spin-orbital block-diagonal
    form. See build_mp2_static_correction for the Sigma(infinity) justification.
    """
    dgamma_spatial = _dgamma_spatial_from_ao_density(mf, dm_cc_ao)
    return _spin_orbital_static_correction(g_anti_spin, dgamma_spatial)


def build_ccsd_static_correction(mf, mol=None, ncore=0):
    """Build the generalized-Fock 'static' correction to the ADC(3) F block from the CCSD 1-RDM.

    Same G[Delta_gamma] construction as build_mp2_static_correction (see there for
    the Sigma(infinity) justification), but with Delta_gamma taken from the full
    CCSD ground-state 1-RDM (src/SingleReference/CC/pipeline.py:compute_ccsd_density_matrix,
    a thin wrapper around pyscf's CCSD Lambda density) instead of a perturbative
    (MPn) one.

    ncore: number of frozen spatial core orbitals (e.g. 2 for C+O 1s).
    Returns the (2*nmo, 2*nmo) spin-orbital correction G[Delta_gamma] for an
    RHF reference, or the (nso, nso) block-stacked spin-orbital correction
    (get_uhf_spin_orbital_arrays_blockstacked's ordering) for a UHF reference.
    """

    mol = mol if mol is not None else mf.mol

    if isinstance(mf, scf.uhf.UHF):
        nocc_a, nocc_b = mf.nelec
        norb_a, norb_b = mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1]
        dm_a_ao, dm_b_ao = compute_ccsd_density_matrix_uhf(mf, ncore=ncore)
        dgamma_a, dgamma_b = _dgamma_spatial_from_ao_density_uhf(mf, dm_a_ao, dm_b_ao)
        _, g_anti_spin, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
        dgamma_spin = _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b, nocc_a, nocc_b, norb_a, norb_b)
        return np.einsum('prqs,rs->pq', g_anti_spin, dgamma_spin)

    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    dm_cc_ao = compute_ccsd_density_matrix(mf, ncore=ncore)
    return _static_correction_from_ao_density(mf, dm_cc_ao, g_anti_spin)


def build_ccsd_static_correction_restricted(mf, mol=None, ncore=0, B_aa=None):
    """Restricted (spatial-MO) counterpart of build_ccsd_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted.

    ncore: number of frozen spatial core orbitals (e.g. 2 for C+O 1s).
    B_aa: DF/RI factor (naux, norb, norb); contracts the density with it
        instead of the dense four-index integrals.
    Returns the (nmo, nmo) restricted correction G[Delta_gamma], symmetric.
    """

    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("build_ccsd_static_correction_restricted is RHF-only.")

    mol = mol if mol is not None else mf.mol
    dm_cc_ao = compute_ccsd_density_matrix(mf, ncore=ncore)
    dgamma_spatial = _dgamma_spatial_from_ao_density(mf, dm_cc_ao)
    if B_aa is not None:
        return _static_correction_from_dgamma_restricted_df(B_aa, dgamma_spatial)
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)


def build_density_static_correction_restricted(dm1, nocc, B_aa, fock=None):
    """Static correction G[dm1 - dm_ref] of the restricted F block from a
    correlated spin-summed 1-RDM dm1 given in the MO basis of the reference
    determinant (dm_ref = diag(2,...,2,0,...,0)), contracted with the DF factor
    B_aa of the same orbitals. For a reference that is not canonical HF, fock
    is its Fock matrix in that basis: the part off its diagonal (the orbital
    energies the solver takes) is added, so F + correction = h + G[dm1] in
    either case.
    Returns the (nmo, nmo) correction, symmetric."""
    nmo = dm1.shape[0]
    dgamma = dm1 - np.diag([2.0] * nocc + [0.0] * (nmo - nocc))
    corr = _static_correction_from_dgamma_restricted_df(B_aa, 0.5 * (dgamma + dgamma.T))
    if fock is not None:
        corr = corr + fock - np.diag(np.diag(fock))
    return corr


def build_ccsdt_static_correction(mf, mol=None, **kwargs):
    """Build the generalized-Fock 'static' correction to the ADC(3) F block from the CCSDT 1-RDM.

    Same construction as build_ccsd_static_correction, but with Delta_gamma taken
    from the full CCSDT ground-state 1-RDM
    (src/SingleReference/CC/pipeline.py:compute_ccsdt_density_matrix, from-scratch
    spin-orbital T/Lambda-CCSDT). **kwargs are forwarded to
    compute_ccsdt_density_matrix (e.g. t_stopping_eps, l_stopping_eps, max_iter).

    Cost warning: full spin-orbital CCSDT with no frozen core, same caveat as
    compute_ccsdt_density_matrix -- can take tens of minutes to hours beyond a
    handful of atoms in anything larger than a minimal/double-zeta basis.

    Returns the (2*nmo, 2*nmo) spin-orbital correction G[Delta_gamma], symmetric.
    """

    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("build_ccsdt_static_correction is currently restricted-spin only.")

    mol = mol if mol is not None else mf.mol
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    dm_cc_ao = compute_ccsdt_density_matrix(mf, **kwargs)
    return _static_correction_from_ao_density(mf, dm_cc_ao, g_anti_spin)


def build_ccsdt_static_correction_restricted(mf, mol=None, **kwargs):
    """Restricted (spatial-MO) counterpart of build_ccsdt_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted. Same cost warning as
    build_ccsdt_static_correction -- can take tens of minutes to hours.

    Returns the (nmo, nmo) restricted correction G[Delta_gamma], symmetric.
    """

    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("build_ccsdt_static_correction_restricted is RHF-only.")

    mol = mol if mol is not None else mf.mol
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    dm_cc_ao = compute_ccsdt_density_matrix(mf, **kwargs)
    dgamma_spatial = _dgamma_spatial_from_ao_density(mf, dm_cc_ao)
    return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)


def _ks_hx_hxc_correction_restricted(mf, mol, dm=None):
    """Sigma_Hx[gamma] - Sigma_Hxc[gamma] in the spatial MO basis: the extra piece
    of Sigma(infinity) needed when mf is a KS (not HF) mean-field object.

    diag(eps) alone is only Sigma(infinity) (in the exact-exchange sense the F
    block assumes) at an HF reference, where eps_p = h_pp + Sigma_Hx_pp[gamma_HF]
    by Koopmans' theorem. At a KS reference eps_p = h_pp + Sigma_Hxc_pp[gamma_KS]
    instead (Hartree + xc, not Hartree + exact exchange), so diag(eps) must be
    corrected by Sigma_Hx[gamma] - Sigma_Hxc[gamma] to recover the same
    Sigma(infinity) reference point the F block's static_correction machinery is
    built around (see ADCSolver's docstring). Same construction as the GW
    quasiparticle layer's diagonal Sigma_x - v_xc
    (GW.qp_solve.static_exchange_diagonal), generalized here to the full matrix
    so it can be added directly to the F block like
    build_mp2_static_correction's G[Delta_gamma].

    dm: density evaluated at for both Sigma_Hx and Sigma_Hxc (defaults to mf's own
    SCF density); pass a correlated density to evaluate the correction there instead.
    """
    dm = dm if dm is not None else mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    V_Hxc = mf.get_veff(mol, dm)
    mf_hf = scf.RHF(mol)
    V_Hx = mf_hf.get_veff(mol, dm)
    mo = mf.mo_coeff
    return mo.T @ (V_Hx - V_Hxc) @ mo


def _ks_hx_hxc_correction_uhf(mf, mol, dm=None):
    """UHF/UKS counterpart of _ks_hx_hxc_correction_restricted: per-spin
    (norb_a,norb_a)/(norb_b,norb_b) spatial-MO Sigma_Hx-Sigma_Hxc matrices."""
    dm = dm if dm is not None else mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    V_Hxc_a, V_Hxc_b = mf.get_veff(mol, dm)
    mf_hf = scf.UHF(mol)
    V_Hx_a, V_Hx_b = mf_hf.get_veff(mol, dm)
    mo_a, mo_b = mf.mo_coeff
    corr_a = mo_a.T @ (V_Hx_a - V_Hxc_a) @ mo_a
    corr_b = mo_b.T @ (V_Hx_b - V_Hxc_b) @ mo_b
    return corr_a, corr_b


def build_ks_static_correction(mf, mol=None, dm=None):
    """Spin-orbital Sigma_Hx-Sigma_Hxc correction to ADCSolver's F block, needed
    for a KS (not HF) mean-field object -- see _ks_hx_hxc_correction_restricted.
    RHF/RKS: (2*nmo, 2*nmo), interleaved alpha/beta (matches
    get_antisymmetrized_spin_eri). UHF/UKS: (norb_a+norb_b, norb_a+norb_b),
    block-stacked [occ_alpha, occ_beta, virt_alpha, virt_beta] (matches
    get_uhf_spin_orbital_arrays_blockstacked). Returns None (no correction) if mf
    is not a KS object and dm is not given, so it is a no-op for a plain HF
    reference.

    Add the result to a Delta_gamma-based static_correction (e.g. from
    build_mp2_static_correction) rather than replacing it -- they are additive,
    non-overlapping pieces of Sigma(infinity): this term corrects the reference
    point from Sigma_Hxc(KS) to Sigma_Hx(HF), the Delta_gamma term adds the
    genuine correlation correction G[Delta_gamma] on top.
    """
    mol = mol if mol is not None else mf.mol
    if not hasattr(mf, 'xc') and dm is None:
        return None
    if isinstance(mf, scf.uhf.UHF):
        corr_a, corr_b = _ks_hx_hxc_correction_uhf(mf, mol, dm)
        nocc_a, nocc_b = mf.nelec
        norb_a, norb_b = corr_a.shape[0], corr_b.shape[0]
        return _uhf_dgamma_to_spin_blockstacked(corr_a, corr_b, nocc_a, nocc_b, norb_a, norb_b)
    corr = _ks_hx_hxc_correction_restricted(mf, mol, dm)
    nmo = corr.shape[0]
    corr_spin = np.zeros((2 * nmo, 2 * nmo))
    corr_spin[0::2, 0::2] = corr
    corr_spin[1::2, 1::2] = corr
    return corr_spin


def build_ks_static_correction_restricted(mf, mol=None, dm=None):
    """Restricted (spatial-MO) counterpart of build_ks_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted. RHF/RKS only -- use
    build_ks_static_correction (spin-orbital) for UHF/UKS. Returns None if mf is
    not a KS object and dm is not given.
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(
            "build_ks_static_correction_restricted is RHF-only -- use "
            "build_ks_static_correction (spin-orbital) for UHF.")
    mol = mol if mol is not None else mf.mol
    if not hasattr(mf, 'xc') and dm is None:
        return None
    return _ks_hx_hxc_correction_restricted(mf, mol, dm)


def build_solvent_static_correction(mf, mol=None):
    """Spin-orbital static COHSEX reaction-field correction to Sigma(infinity)
    for a mean field carrying a solvent screening, else None.

    Same packing conventions as build_ks_static_correction (RHF/RKS:
    interleaved (2*nmo, 2*nmo); UHF/UKS: block-stacked), and additive with it
    and with the Delta_gamma term for the same reason -- these are disjoint
    pieces of Sigma(infinity). This one is first order in the reaction field
    vtilde and carries essentially all of the solvation shift; the
    v -> v + vtilde substitution in the integrals themselves only reaches
    order vtilde*v. See SolventScreening.cohsex_correction.
    """
    corr = solvent_static_selfenergy(mf, mol)
    if corr is None:
        return None
    if isinstance(corr, tuple):
        corr_a, corr_b = corr
        nocc_a, nocc_b = mf.nelec
        return _uhf_dgamma_to_spin_blockstacked(corr_a, corr_b, nocc_a, nocc_b,
                                                corr_a.shape[0], corr_b.shape[0])
    nmo = corr.shape[0]
    corr_spin = np.zeros((2 * nmo, 2 * nmo))
    corr_spin[0::2, 0::2] = corr
    corr_spin[1::2, 1::2] = corr
    return corr_spin


def build_solvent_static_correction_restricted(mf, mol=None):
    """Restricted (spatial-MO) counterpart of build_solvent_static_correction."""
    corr = solvent_static_selfenergy(mf, mol)
    if isinstance(corr, tuple):
        raise NotImplementedError(
            "build_solvent_static_correction_restricted is RHF-only -- use "
            "build_solvent_static_correction (spin-orbital) for UHF.")
    return corr


def _add_ks(base, ks):
    """Fold an additive, non-overlapping Sigma(infinity) piece (the KS
    double-counting term, the solvent reaction field) into `base`."""
    if ks is None:
        return base
    return ks if base is None else base + ks


_STATIC_KINDS = (None, 'mp2_unrelaxed', 'mp2_relaxed', 'mp3_unrelaxed',
                 'mp3_relaxed', 'mp4_unrelaxed', 'ccsd', 'ccsdt')

#: EN-prefixed kind spellings (scan-script vocabulary): prefix -> dress dict
_EN_KIND_PREFIXES = (('full_en_hp_', {'hh': True, 'pp': True, 'hp': True}),
                     ('full_en_', {'hh': True, 'pp': True}),
                     ('hhen_', {'hh': True}))


def parse_static_kind(kind):
    """Split an EN-prefixed static kind ('hhen_mp2_relaxed',
    'full_en_mp3_unrelaxed', 'full_en_hp_mp2_relaxed', ...) into
    (base_kind, en_dress-dict-or-None) -- so a scan key yields the same dict
    the ADCSolver should be given for a consistent EN run."""
    if kind is None:
        return None, None
    for pre, dress in _EN_KIND_PREFIXES:
        if kind.startswith(pre):
            return kind[len(pre):], dict(dress)
    return kind, None


def build_static_correction(mf, mol=None, kind='mp2_relaxed', en_dress=None,
                            nocc=None, ncore=0, B_aa=None, spin='auto',
                            df=False, cphf_level_shift=0.0, cphf_max_cycle=None,
                            cphf_tol=None, amplitudes=None):
    """Sigma(infinity) static-correction dispatcher over the builders above;
    the KS double-counting correction is always added on top (zero-effect for
    plain HF references).

    kind: one of (None, 'mp2_unrelaxed', 'mp2_relaxed', 'mp3_unrelaxed',
        'mp3_relaxed', 'mp4_unrelaxed', 'ccsd', 'ccsdt'), optionally EN-prefixed with
        'hhen_' / 'full_en_' / 'full_en_hp_' (the scan-script vocabulary;
        see parse_static_kind).
    en_dress: Epstein-Nesbet dressing dict, threaded as u2_denom_dress into
        the MP2/MP3 density (the 'singles' key is consumed HERE); rejected
        for ccsd/ccsdt (a CC density has no EN hook). Pass the SAME dict to
        the ADCSolver for a consistent EN run. Mutually exclusive with an
        EN-prefixed kind.
    spin: 'auto' (restricted for RHF, spin-orbital for UHF) | 'restricted' |
        'spinorbital' -- must match the solver branch the correction feeds
        (spin-orbital corrections are (nso, nso)).
    df: spin-orbital ('spinorbital') branch ONLY, same convention as
        ADCSolver(..., df=True): mf must already be density-fitted
        (mf.with_df set, e.g. via mf.density_fit()); the blockstacked DF
        factor is then built here and build_mp2_static_correction_uhf_df is
        used instead of the dense build_mp2_static_correction -- no silent
        fallback to the dense route, and no auto-detection from mf.with_df:
        df must be requested explicitly, and any combination the DF builder
        doesn't support (kind other than mp2_unrelaxed/mp2_relaxed, ncore!=0,
        a restricted reference) raises immediately instead of silently
        computing something else. For spin='restricted', pass B_aa instead
        (the DF hook of the restricted MP2/MP3/CCSD builders).
    cphf_level_shift/cphf_max_cycle/cphf_tol: forwarded to the UHF CPHF/
        Z-vector solve (relax=True only, dense or df); see
        solve_cphf_relaxation_uhf.
    amplitudes: foreign (CC) amplitude pool for the MP2 density functional,
        the Sigma(inf) partner of a CC-ADC(3) amplitude substitution. Restricted
        MP2 kinds only ('mp2_unrelaxed'/'mp2_relaxed'); rejected elsewhere
        rather than silently ignored. See
        build_mp2_static_correction_restricted's parameter of the same name."""
    mol = mol if mol is not None else mf.mol
    kind, kind_dress = parse_static_kind(kind)
    if kind_dress is not None:
        if en_dress is not None:
            raise ValueError("give EITHER an EN-prefixed kind OR en_dress, "
                             "not both")
        en_dress = kind_dress
    is_uhf = isinstance(mf, scf.uhf.UHF)
    if spin == 'auto':
        spin = 'spinorbital' if is_uhf else 'restricted'
    if spin not in ('restricted', 'spinorbital'):
        raise ValueError(f"spin={spin!r}; expected 'auto', 'restricted', "
                         "or 'spinorbital'")
    if spin == 'restricted' and is_uhf:
        raise ValueError("spin='restricted' needs an RHF reference")
    if df and spin == 'restricted':
        raise NotImplementedError(
            "df=True is the spin-orbital (UHF) DF route; the restricted "
            "branch's DF hook is B_aa (build_mp2/mp3_static_correction_"
            "restricted's own parameter), not this flag")
    if kind not in _STATIC_KINDS:
        raise ValueError(f"kind={kind!r}; expected one of {_STATIC_KINDS}")
    if amplitudes is not None:
        if spin != 'restricted':
            raise NotImplementedError(
                "amplitudes= is implemented on the restricted branch only "
                f"(got spin={spin!r}); the CC-ADC(3) route it serves is restricted.")
        if kind not in ('mp2_unrelaxed', 'mp2_relaxed'):
            raise ValueError(
                f"amplitudes= applies to the MP2 density kinds only (got kind={kind!r}): "
                "an MP3/MP4/CC/None static has no second-order amplitude functional "
                "to substitute into.")
    en_dress = validate_en_dress(en_dress)
    if en_dress is not None and kind not in ('mp2_unrelaxed', 'mp2_relaxed',
                                             'mp3_unrelaxed', 'mp3_relaxed'):
        raise ValueError("en_dress applies to the MP2/MP3 density kinds only "
                         "(a CC/None static has no EN hook)")
    relax = kind in ('mp2_relaxed', 'mp3_relaxed')

    if spin == 'restricted':
        nocc = nocc if nocc is not None else mol.nelectron // 2
        if kind is None:
            base = None
        elif kind.startswith('mp2'):
            base = build_mp2_static_correction_restricted(
                mf, mol, nocc, relax=relax, ncore=ncore, B_aa=B_aa,
                u2_denom_dress=en_dress, amplitudes=amplitudes)
        elif kind.startswith('mp3'):
            base = build_mp3_static_correction_restricted(
                mf, mol, nocc, relax=relax, ncore=ncore, B_aa=B_aa,
                u2_denom_dress=en_dress)
        elif kind.startswith('mp4'):
            if B_aa is not None:
                raise NotImplementedError(
                    "kind='mp4_unrelaxed' has no DF route (the MP4 density driver "
                    "takes dense spin-block integrals); pass B_aa=None")
            base = build_mp4_static_correction_restricted(mf, mol, nocc, ncore=ncore)
        elif kind == 'ccsd':
            base = build_ccsd_static_correction_restricted(mf, mol, ncore=ncore,
                                                          B_aa=B_aa)
        else:
            base = build_ccsdt_static_correction_restricted(mf, mol)
        base = _add_ks(base, build_ks_static_correction_restricted(mf, mol))
        return _add_ks(base, build_solvent_static_correction_restricted(mf, mol))

    if is_uhf and kind == 'ccsdt':
        raise NotImplementedError(
            f"kind={kind!r} is not implemented for UHF (the CCSDT density "
            "pipeline is RHF-only)")
    if nocc is None:
        nocc = None if is_uhf else mol.nelectron // 2

    if df:
        if kind not in ('mp2_unrelaxed', 'mp2_relaxed'):
            raise NotImplementedError(
                f"df=True (spin-orbital) only has an MP2 static-correction "
                f"builder (build_mp2_static_correction_uhf_df); kind={kind!r} "
                "has no DF route -- use df=False (dense) for mp3/ccsd/ccsdt")
        if ncore != 0:
            raise NotImplementedError(
                "build_mp2_static_correction_uhf_df has no frozen-core hook "
                f"(ncore=0 only); got ncore={ncore}")
        if getattr(mf, 'with_df', None) is None:
            raise ValueError(
                "df=True needs a density-fitted mean field (mf.with_df) -- "
                "same requirement as ADCSolver(mf, df=True, ...)")
        B_spin = get_uhf_spin_orbital_df_factor_blockstacked(mol, mf)
        base = build_mp2_static_correction_uhf_df(
            mf, mol, B_spin, relax=relax, u2_denom_dress=en_dress,
            cphf_level_shift=cphf_level_shift, cphf_max_cycle=cphf_max_cycle,
            cphf_tol=cphf_tol)
        base = _add_ks(base, build_ks_static_correction(mf, mol))
        return _add_ks(base, build_solvent_static_correction(mf, mol))

    if kind is None:
        base = None
    elif kind.startswith('mp2'):
        base = build_mp2_static_correction(mf, mol, nocc, relax=relax,
                                           ncore=ncore, u2_denom_dress=en_dress,
                                           cphf_level_shift=cphf_level_shift,
                                           cphf_max_cycle=cphf_max_cycle,
                                           cphf_tol=cphf_tol)
    elif kind.startswith('mp3'):
        base = build_mp3_static_correction(mf, mol, nocc, relax=relax,
                                           ncore=ncore, u2_denom_dress=en_dress)
    elif kind.startswith('mp4'):
        base = build_mp4_static_correction(mf, mol, nocc, ncore=ncore)
    elif kind == 'ccsd':
        base = build_ccsd_static_correction(mf, mol, ncore=ncore)
    else:
        base = build_ccsdt_static_correction(mf, mol)
    base = _add_ks(base, build_ks_static_correction(mf, mol))
    return _add_ks(base, build_solvent_static_correction(mf, mol))
