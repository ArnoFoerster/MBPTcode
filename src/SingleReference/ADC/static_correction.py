"""Static (frequency-independent) corrections to the ADC(3) F block."""
import numpy as np
from pyscf import scf

from src.SingleReference.EpsteinNesbet import (_build_dressed_e_ai, _build_dressed_e_abij,
                                                _build_dressed_denoms_uhf,
                                                restricted_channel_shifts,
                                                validate_en_dress)
from src.Base.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist, get_antisymmetrized_spin_eri
from src.Base.pyscf_interface import uhf_blockstacked_order
from src.Base.solvent_screening import solvent_static_selfenergy
from src.SingleReference.DensityMatrix.density_matrix import (
    semicanonicalize_restricted,
    t1_singles_blocks,
)
from src.SingleReference.DensityMatrix.density_matrix import (
    semicanonicalize_uhf,
    t1_singles_blocks,
)
from src.SingleReference.DensityMatrix.density_matrix import (
    solve_cphf_relaxation,
    _build_dgamma_block,
)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted,
    compute_delta_gamma2_df_streamed,
)
from src.Base.pyscf_interface import get_antisymmetrized_spin_block_eri
from src.SingleReference.DensityMatrix.density_matrix import (
    solve_cphf_relaxation,
    solve_cphf_relaxation_uhf,
    _build_dgamma_block,
)
from src.Base.pyscf_interface import (
    get_antisymmetrized_spin_block_eri,
    get_uhf_spin_orbital_arrays_blockstacked,
    get_uhf_spin_orbital_df_factor_blockstacked,
)
from src.SingleReference.CC.pipeline import compute_ccsd_density_matrix
from src.SingleReference.CC.pipeline import compute_ccsd_density_matrix_uhf
from src.SingleReference.CC.pipeline import compute_ccsdt_density_matrix
from src.SingleReference.ADC.adc_r_utils import _dress_is_streamable
from src.SingleReference.DensityMatrix.generated_mpn import (
    mpn_density_pieces as mpn_gen)
from src.SingleReference.DensityMatrix.mpn_density_driver import (
    _denom, _to_l, g_vvoo_df, t1_2_numerator_df)
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import (
    MPnDensityDriverUnrestricted)
from src.SingleReference.EpsteinNesbet.shifts import (
    _diag_ladder_df, _diag_ladder_df_screened, epstein_nesbet_denominator)
from src.SingleReference.LinearResponse.linear_response import (
    static_screened_coulomb_aux_uhf)

def diagonalize_static_only(eps_spin, static_correction=None):
    """Diagonalize just diag(eps_spin) + static_correction"""
    norb = len(eps_spin)
    H = np.diag(eps_spin) if static_correction is None else np.diag(eps_spin) + static_correction
    eGF, Reigv = np.linalg.eigh(H)
    order = np.argsort(eGF)
    return eGF[order], Reigv[:, order]



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


def _mp2_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=None, B_aa=None, u2_denom_dress=None):
    """RHF-only: MP2-level Delta_gamma in the spatial MO basis, shared by
    build_mp2_static_correction's RHF branch and build_mp2_static_correction_restricted.

    Correlation blocks come from the generated restricted pipeline
    (MPnDensityDriverRestricted).

    If mf is a KS object, transparently semicanonicalizes it (_ks_semicanonical_setup)
    and adds the leading-order T1 singles contribution on top of the existing
    (Brillouin-theorem-derived) t2/t1_2 formulas -- see that helper and
    density_matrix.t1_singles_blocks for the physics and its known limitation.

    relax=True for a KS reference solves the CPHF/CPKS Z-vector equation in the
    *semicanonical* basis (passing the semicanonical mf copy -- same mo_occ, same
    .xc/grids for the response kernel, only mo_coeff/mo_energy rotated -- as the
    Hessian object to solve_cphf_relaxation) and rotates the relaxed result back;
    the T1 singles' own ov contribution is included in the un-relaxed dgamma_ov fed
    into that solve (i.e. it gets CPHF-relaxed along with the rest), consistent with
    how the existing t2-driven ov piece is already treated for the canonical-HF case.

    eri_chemist: if the caller already has mf's own spatial chemist ERI (not
    mf_eval's -- only used when mf is plain HF, i.e. mf_eval is mf itself, since
    a KS reference needs its own transform in the rotated semicanonical basis),
    pass it here to avoid a second norb^4 AO->MO transform/array -- see
    get_antisymmetrized_spin_block_eri's eri_chemist parameter.

    B_aa: DF/RI factor (naux, norb, norb), e.g. DFIntegrals.from_scf(mol, mf).B_aa
    when given AND mf is plain HF (ks_setup is None;
    routes through MPnDensityDriverRestricted's DF
    methods instead of building g_aaaa/g_abab/g_bbbb at all. Ignored (falls
    back to the dense eri_chemist path) for a KS reference.
    """

    ks_setup = _ks_semicanonical_setup(mf, mol, nocc, ncore)
    mf_eval = ks_setup[0] if ks_setup is not None else mf

    nmo = mf_eval.mo_coeff.shape[1]
    eps = get_orbital_energies(mf_eval, representation='spatial')

    w = slice(ncore, None)

    # The DF + plain-HF branch is rank-4-free end to end when the dressing
    # is streamable (bare, or hh/pp spin-adapted): the T2^(1) amplitudes
    # stream from B_aa (compute_delta_gamma2_df_streamed) with the EN shift
    # entering as the small dh/dp matrices, so neither the (V,V,O,O)
    # e_abij inverse denominator nor any amplitude array is ever built.
    # 'hp'/spin-resolved dressing is not chunk-representable and takes the
    # materialized driver path below -- the same documented exception as
    # the ADC solver's _dress_is_streamable gate.
    _mp2_streamable = (B_aa is not None and ks_setup is None
                       and _dress_is_streamable(u2_denom_dress))

    e_abij = e_ai = None
    if u2_denom_dress is not None:
        O = nocc - ncore
        V = nmo - nocc
        if B_aa is not None and ks_setup is None:
            B_aa_w = B_aa[:, w, w]
            if not _mp2_streamable:
                e_abij = _build_dressed_e_abij(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
            e_ai = _build_dressed_e_ai(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
        else:
            reusable_eri = eri_chemist if ks_setup is None else None
            if reusable_eri is None:
                reusable_eri = get_two_electron_integrals_chemist(mol, mf_eval, representation='spatial')
            reusable_eri_w = reusable_eri[w, w, w, w]
            e_abij = _build_dressed_e_abij(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)
            e_ai = _build_dressed_e_ai(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)

    if _mp2_streamable:
        B_aa_w = B_aa[:, w, w]
        O_act = nocc - ncore
        dh = dp = None
        if u2_denom_dress:
            vidx_abs = np.arange(O_act, nmo - ncore)
            d_h, d_p, _ = restricted_channel_shifts(
                u2_denom_dress, B_aa_w, None, None, O_act, vidx_abs)
            dh = d_h[0] if d_h is not None else None
            dp = d_p[0] if d_p is not None else None
        oo_a, ov_a, vv_a = compute_delta_gamma2_df_streamed(
            B_aa_w, eps[w], O_act, dh=dh, dp=dp, e_ai=e_ai)
    elif B_aa is not None and ks_setup is None:
        B_aa_w = B_aa[:, w, w]
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), None, None, None,
                                            nocc - ncore, B_aa=B_aa_w, B_bb=B_aa_w, e_abij=e_abij, e_ai=e_ai)
        oo_a, ov_a, vv_a = driver.compute_delta_gamma2_df()
    else:
        reusable_eri = eri_chemist if ks_setup is None else None
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval, eri_chemist=reusable_eri)
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), g_aaaa[w, w, w, w],
                                            g_abab[w, w, w, w], g_bbbb[w, w, w, w],
                                            nocc - ncore, e_abij=e_abij, e_ai=e_ai)
        oo_a, ov_a, vv_a = driver.compute_delta_gamma2()
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
        dgamma_full = _build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    if ks_setup is not None:
        dgamma_full = _rotate_dgamma_to_original_basis(dgamma_full, U_oo, U_vv, nocc)
    return dgamma_full


def _mp3_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=None, B_aa=None, u2_denom_dress=None):
    """RHF-only: MP3-level (2nd+3rd order) Delta_gamma in the spatial MO basis,
    shared by build_mp3_static_correction's RHF branch and
    build_mp3_static_correction_restricted.

    Correlation blocks come from the generated restricted pipeline
    (MPnDensityDriverRestricted, laplace_ntau=6 matching the hand-written
    solver's default).

    KS handling: same as _mp2_dgamma_spatial -- the T1 singles contribution is
    leading (first) order, so it is added once to the combined 2nd+3rd-order total,
    not separately per order (see t1_singles_blocks's docstring).

    eri_chemist: see _mp2_dgamma_spatial's parameter of the same name -- only
    reused when mf is plain HF (mf_eval is mf), skipping a second norb^4
    AO->MO transform/array.

    B_aa: see _mp2_dgamma_spatial's parameter of the same name -- routes
    through MPnDensityDriverRestricted.compute_delta_gamma23_df when given
    and mf is plain HF.
    """

    ks_setup = _ks_semicanonical_setup(mf, mol, nocc, ncore)
    if ks_setup is not None and relax:
        raise NotImplementedError(
            "_mp3_dgamma_spatial: relax=True is not yet implemented for a KS reference "
            "(needs a semicanonical-basis CPHF/Z-vector solve) -- use relax=False.")
    mf_eval = ks_setup[0] if ks_setup is not None else mf

    nmo = mf_eval.mo_coeff.shape[1]
    eps = get_orbital_energies(mf_eval, representation='spatial')
    # Frozen core: the hand-written solver's ncore is exactly a [ncore:]
    # window on eps/g with nocc-ncore occupied (its _slices() never touches
    # [0:ncore]), so slice before constructing. Driver returns the alpha
    # block; x2 for alpha+beta.
    w = slice(ncore, None)

    e_abij = e_ai = None
    if u2_denom_dress is not None:
        O = nocc - ncore
        V = nmo - nocc
        if B_aa is not None and ks_setup is None:
            B_aa_w = B_aa[:, w, w]
            e_abij = _build_dressed_e_abij(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
            e_ai = _build_dressed_e_ai(u2_denom_dress, eps[w], O, V, B_aa=B_aa_w)
        else:
            reusable_eri = eri_chemist if ks_setup is None else None
            if reusable_eri is None:
                reusable_eri = get_two_electron_integrals_chemist(mol, mf_eval, representation='spatial')
            reusable_eri_w = reusable_eri[w, w, w, w]
            e_abij = _build_dressed_e_abij(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)
            e_ai = _build_dressed_e_ai(u2_denom_dress, eps[w], O, V, eri_chemist=reusable_eri_w)

    if B_aa is not None and ks_setup is None:
        B_aa_w = B_aa[:, w, w]
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), None, None, None,
                                            nocc - ncore, B_aa=B_aa_w, B_bb=B_aa_w, e_abij=e_abij, e_ai=e_ai)
        (oo2_a, ov2_a, vv2_a), (oo3_a, ov3_a, vv3_a) = driver.compute_delta_gamma23_df(laplace_ntau=6)
    else:
        reusable_eri = eri_chemist if ks_setup is None else None
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval, eri_chemist=reusable_eri)
        driver = MPnDensityDriverRestricted(np.diag(eps[w]), g_aaaa[w, w, w, w],
                                            g_abab[w, w, w, w], g_bbbb[w, w, w, w],
                                            nocc - ncore, e_abij=e_abij, e_ai=e_ai)
        (oo2_a, ov2_a, vv2_a), (oo3_a, ov3_a, vv3_a) = driver.compute_delta_gamma23(laplace_ntau=6)
    oo_act = 2.0 * (oo2_a + oo3_a)
    ov_act = 2.0 * (ov2_a + ov3_a)
    vv_act = 2.0 * (vv2_a + vv3_a)

    if ks_setup is not None:
        _, U_oo, U_vv, t1_ov, dgamma_oo_t1, dgamma_vv_t1 = ks_setup
        oo_act = oo_act + dgamma_oo_t1
        ov_act = ov_act + t1_ov
        vv_act = vv_act + dgamma_vv_t1

    dgamma_oo, dgamma_ov, dgamma_vv = _embed_active(oo_act, ov_act, vv_act, nocc, ncore)

    if relax:
        return solve_cphf_relaxation(mf, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    dgamma_full = _build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    if ks_setup is not None:
        dgamma_full = _rotate_dgamma_to_original_basis(dgamma_full, U_oo, U_vv, nocc)
    return dgamma_full


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


def build_mp2_static_correction(mf, mol=None, nocc=None, relax=False, ncore=0, u2_denom_dress=None,
                                cphf_level_shift=0.0, cphf_max_cycle=None, cphf_tol=None):
    """Build the generalized-Fock 'static' correction to the ADC(3) F block from the MP2 1-RDM.

    cphf_level_shift/cphf_max_cycle/cphf_tol: forwarded to the UHF branch's
    solve_cphf_relaxation_uhf (None keeps that function's own defaults,
    CPHF_MAX_CYCLE=100/CPHF_TOL=1e-9). Strongly spin-contaminated UHF
    references (small occ-virt gaps) can fail to converge at those defaults;
    try level_shift~0.2-0.3 and max_cycle~500-1000.

    F is linear in the density, so F[gamma_HF + Delta_gamma] = diag(eps) +
    G[Delta_gamma] with G[gamma]_pq = sum_rs gamma_rs*<pr||qs> (spin-orbital
    antisymmetrized). CPHF relaxation runs in the spatial MO basis (pyscf
    machinery); the result is expanded to spin-orbital block-diagonal form.

    ncore: number of frozen spatial core orbitals. Returns the (nso, nso)
    spin-orbital correction G[Delta_gamma], symmetric -- RHF: nso=2*nmo,
    interleaved alpha/beta; UHF: nso=norb_a+norb_b, block-stacked
    [occ_alpha, occ_beta, virt_alpha, virt_beta] (required since occupied
    orbitals only form a contiguous slice in that ordering for a general
    nocc_a != nocc_b).
    """

    mol = mol if mol is not None else mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:

        ks_setup = _ks_semicanonical_setup_uhf(mf, mol, ncore)
        mf_eval = ks_setup[0] if ks_setup is not None else mf

        nocc_a, nocc_b = mf.nelec
        eps_a, eps_b = get_orbital_energies(mf_eval, representation='spatial')
        norb_a, norb_b = len(eps_a), len(eps_b)
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval)
        # UHF engine is the generated unrestricted pipeline (see
        # tests/test_mpn_density_unrestricted.py). Frozen core: ncore is a
        # [ncore:] window on eps/g per spin channel, sliced before
        # construction (same convention as _mp2_dgamma_spatial's RHF branch).
        w = slice(ncore, None)
        custom_denom = None
        if u2_denom_dress is not None:
            custom_denom = _build_dressed_denoms_uhf(u2_denom_dress, eps_a, eps_b, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b, ncore, mol=mol, mf_eval=mf_eval)
        driver = MPnDensityDriverUnrestricted(np.diag(eps_a[w]), np.diag(eps_b[w]),
                                              g_aaaa[w, w, w, w], g_abab[w, w, w, w],
                                              g_bbbb[w, w, w, w], nocc_a - ncore, nocc_b - ncore,
                                              custom_denom=custom_denom)
        oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = driver.compute_delta_gamma2()

        if ks_setup is not None:
            (_, U_oo_a, U_vv_a, t1_ov_a, dgamma_oo_t1_a, dgamma_vv_t1_a,
             U_oo_b, U_vv_b, t1_ov_b, dgamma_oo_t1_b, dgamma_vv_t1_b) = ks_setup
            oo_a, ov_a, vv_a = oo_a + dgamma_oo_t1_a, ov_a + t1_ov_a, vv_a + dgamma_vv_t1_a
            oo_b, ov_b, vv_b = oo_b + dgamma_oo_t1_b, ov_b + t1_ov_b, vv_b + dgamma_vv_t1_b

        dgamma_oo_a, dgamma_ov_a, dgamma_vv_a = _embed_active(oo_a, ov_a, vv_a, nocc_a, ncore)
        dgamma_oo_b, dgamma_ov_b, dgamma_vv_b = _embed_active(oo_b, ov_b, vv_b, nocc_b, ncore)

        if relax:
            # Solve Z-vector equation for relaxed ov blocks
            mf_hess = mf if ks_setup is None else ks_setup[0]
            cphf_kwargs = {'level_shift': cphf_level_shift}
            if cphf_max_cycle is not None:
                cphf_kwargs['max_cycle'] = cphf_max_cycle
            if cphf_tol is not None:
                cphf_kwargs['tol'] = cphf_tol
            dgamma_a, dgamma_b = solve_cphf_relaxation_uhf(
                mf_hess, nocc_a, nocc_b,
                dgamma_oo_a, dgamma_ov_a, dgamma_vv_a,
                dgamma_oo_b, dgamma_ov_b, dgamma_vv_b,
                **cphf_kwargs
            )
        else:
            dgamma_a = _build_dgamma_block(norb_a, nocc_a, dgamma_oo_a, dgamma_ov_a, dgamma_vv_a)
            dgamma_b = _build_dgamma_block(norb_b, nocc_b, dgamma_oo_b, dgamma_ov_b, dgamma_vv_b)

        if ks_setup is not None:
            dgamma_a = _rotate_dgamma_to_original_basis(dgamma_a, U_oo_a, U_vv_a, nocc_a)
            dgamma_b = _rotate_dgamma_to_original_basis(dgamma_b, U_oo_b, U_vv_b, nocc_b)

        eps_spin, g_anti_spin, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
        dgamma_spin = _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b, nocc_a, nocc_b, norb_a, norb_b)
        return np.einsum('prqs,rs->pq', g_anti_spin, dgamma_spin)

    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    dgamma_spatial = _mp2_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=eri_chemist, u2_denom_dress=u2_denom_dress)
    return _spin_orbital_static_correction(g_anti_spin, dgamma_spatial)


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
    D_bare = _denom(eps_spin[o], eps_spin[v], 2)
    d1 = _denom(eps_spin[o], eps_spin[v], 1)
    if u2_denom_dress:
        channels = {k: val for k, val in u2_denom_dress.items()
                    if k in ('hh', 'pp', 'hp')}
        D = epstein_nesbet_denominator(D_bare, None, nocc, layout='pphh',
                                       bare_sign=1.0, B=B_so, **channels)
        # The dense reference (_build_dressed_denoms_uhf) gates the SINGLES
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
            d1 = d1 + _diag_ladder_df_screened(B_so, W_aux, v, o)
        elif singles:
            d1 = d1 + _diag_ladder_df(B_so, v, o)
    else:
        D = D_bare
    t2_1 = g_vvoo_df(B_so, nocc) / D
    t1_2 = t1_2_numerator_df(B_so, nocc, t2_1) / d1

    # ---- Delta_gamma^(2) blocks: generated m2_* are amplitude-only (their g
    # parameter is unused -- verified against the generated source), so the
    # dense driver's compute_delta_gamma2 is reproduced verbatim with g=None.
    kd = np.eye(nso)
    args = dict(g=None, kd=kd, o=o, v=v, nv=nvirt, no=nocc)
    l2, l1 = _to_l(t2_1, 2), _to_l(t1_2, 1)
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
        dgamma_a = _build_dgamma_block(norb_a, nocc_a, oo_a, ov_a, vv_a)
        dgamma_b = _build_dgamma_block(norb_b, nocc_b, oo_b, ov_b, vv_b)

    dgamma_spin = _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b,
                                                   nocc_a, nocc_b, norb_a, norb_b)

    # ---- G_pq = sum_rs dgamma_rs <pr||qs>, rank<=3 from B ----
    tQ = np.einsum('Qrs,rs->Q', B_so, dgamma_spin, optimize=True)
    direct = np.einsum('Qpq,Q->pq', B_so, tQ, optimize=True)
    M = np.einsum('Qrq,rs->Qqs', B_so, dgamma_spin, optimize=True)
    exchange = np.einsum('Qps,Qqs->pq', B_so, M, optimize=True)
    return direct - exchange


def build_mp2_static_correction_restricted(mf, mol=None, nocc=None, relax=False, ncore=0, B_aa=None, u2_denom_dress=None):
    """Restricted (spatial-MO) counterpart of build_mp2_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted. RHF only -- use
    build_mp2_static_correction (spin-orbital) for UHF.

    Same Delta_gamma (via _mp2_dgamma_spatial, shared with the spin-orbital RHF
    branch above) but contracted with the plain spatial chemist ERI via
    _static_correction_from_dgamma_restricted's J-0.5*K formula instead of expanding
    to spin-orbital form -- avoids the spin-orbital solver's ~4x-per-dimension
    blowup while giving numerically the same physical correction (see that
    function's docstring for the equivalence).

    B_aa: DF/RI factor (naux, norb, norb), e.g. DFIntegrals.from_scf(mol, mf).B_aa
    when given, the whole
    dense eri_chemist norb^4 array is never built at all (both the density
    correlation piece, via _mp2_dgamma_spatial's own B_aa branch, and the
    final J-0.5*K contraction, via _static_correction_from_dgamma_restricted_df).
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(
            "build_mp2_static_correction_restricted is RHF-only -- use "
            "build_mp2_static_correction (spin-orbital) for UHF.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2

    if B_aa is not None:
        dgamma_spatial = _mp2_dgamma_spatial(mf, mol, nocc, relax, ncore, B_aa=B_aa, u2_denom_dress=u2_denom_dress)
        return _static_correction_from_dgamma_restricted_df(B_aa, dgamma_spatial)

    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    dgamma_spatial = _mp2_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=eri_chemist, u2_denom_dress=u2_denom_dress)
    return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)


def build_mp3_static_correction(mf, mol=None, nocc=None, relax=False, ncore=0, u2_denom_dress=None):
    """Build the generalized-Fock 'static' correction to the ADC(3) F block from the MP3 1-RDM.

    Same construction as build_mp2_static_correction (G[Delta_gamma], see there for
    the Sigma(infinity) justification), but with Delta_gamma = Delta_gamma^(2) +
    Delta_gamma^(3) -- the density correction through third order in Moller-Plesset
    perturbation theory, using MP3DensityMatrixSolver (src/SingleReference/
    density_matrix.py) for the additional Delta_gamma^(3) piece. Both pieces are
    combined *before* CPHF relaxation (relax=True): the ov block passed to
    solve_cphf_relaxation is Delta_gamma^(2)_ov + Delta_gamma^(3)_ov together, not
    Delta_gamma^(3) relaxed in isolation.

    ncore: number of frozen spatial core orbitals (e.g. 2 for C+O 1s; applied to
    both spin channels for UHF). Returns the (nso, nso) spin-orbital correction
    G[Delta_gamma], symmetric -- see build_mp2_static_correction's docstring for
    the RHF (interleaved) vs UHF (block-stacked) shape/ordering convention.

    Both spin cases run on the generated pipelines: RHF via
    _mp3_dgamma_spatial's MPnDensityDriverRestricted, UHF via
    MPnDensityDriverUnrestricted (see tests/test_mpn_density_unrestricted.py).
    relax=True is not yet implemented for UHF.

    KNOWN LIMITATION (relax=True, RHF): solve_cphf_relaxation does not
    produce a fully correct MP3-relaxed density -- MP2's 1-PDM-only
    Z-vector shortcut isn't valid for MP3, which would need an explicit
    Lambda2/2-particle-density-matrix formalism. See
    compute_mp3_density_matrix_ao's docstring and
    tests/test_mp3_finite_field.py for the regression check that pins this.
    """

    mol = mol if mol is not None else mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        if relax:
            raise NotImplementedError(
                "build_mp3_static_correction: relax=True is not yet implemented for UHF "
                "(needs a UHF CPHF/Z-vector solve) -- use relax=False.")
        ks_setup = _ks_semicanonical_setup_uhf(mf, mol, ncore)
        mf_eval = ks_setup[0] if ks_setup is not None else mf

        nocc_a, nocc_b = mf.nelec
        eps_a, eps_b = get_orbital_energies(mf_eval, representation='spatial')
        norb_a, norb_b = len(eps_a), len(eps_b)
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf_eval)
        # UHF engine is the generated unrestricted pipeline (see
        # tests/test_mpn_density_unrestricted.py). Frozen core: ncore is a
        # [ncore:] window on eps/g per spin channel, sliced before
        # construction.
        w = slice(ncore, None)
        custom_denom = None
        if u2_denom_dress is not None:
            custom_denom = _build_dressed_denoms_uhf(u2_denom_dress, eps_a, eps_b, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b, ncore, mol=mol, mf_eval=mf_eval)
        driver = MPnDensityDriverUnrestricted(np.diag(eps_a[w]), np.diag(eps_b[w]),
                                              g_aaaa[w, w, w, w], g_abab[w, w, w, w],
                                              g_bbbb[w, w, w, w], nocc_a - ncore, nocc_b - ncore,
                                              custom_denom=custom_denom)
        (oo2_a, oo2_b, ov2_a, ov2_b, vv2_a, vv2_b), (oo3_a, oo3_b, ov3_a, ov3_b, vv3_a, vv3_b) = \
            driver.compute_delta_gamma23()
        oo_a, ov_a, vv_a = oo2_a + oo3_a, ov2_a + ov3_a, vv2_a + vv3_a
        oo_b, ov_b, vv_b = oo2_b + oo3_b, ov2_b + ov3_b, vv2_b + vv3_b

        if ks_setup is not None:
            (_, U_oo_a, U_vv_a, t1_ov_a, dgamma_oo_t1_a, dgamma_vv_t1_a,
             U_oo_b, U_vv_b, t1_ov_b, dgamma_oo_t1_b, dgamma_vv_t1_b) = ks_setup
            oo_a, ov_a, vv_a = oo_a + dgamma_oo_t1_a, ov_a + t1_ov_a, vv_a + dgamma_vv_t1_a
            oo_b, ov_b, vv_b = oo_b + dgamma_oo_t1_b, ov_b + t1_ov_b, vv_b + dgamma_vv_t1_b

        dgamma_oo_a, dgamma_ov_a, dgamma_vv_a = _embed_active(oo_a, ov_a, vv_a, nocc_a, ncore)
        dgamma_oo_b, dgamma_ov_b, dgamma_vv_b = _embed_active(oo_b, ov_b, vv_b, nocc_b, ncore)
        dgamma_a = _build_dgamma_block(norb_a, nocc_a, dgamma_oo_a, dgamma_ov_a, dgamma_vv_a)
        dgamma_b = _build_dgamma_block(norb_b, nocc_b, dgamma_oo_b, dgamma_ov_b, dgamma_vv_b)

        if ks_setup is not None:
            dgamma_a = _rotate_dgamma_to_original_basis(dgamma_a, U_oo_a, U_vv_a, nocc_a)
            dgamma_b = _rotate_dgamma_to_original_basis(dgamma_b, U_oo_b, U_vv_b, nocc_b)

        eps_spin, g_anti_spin, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
        dgamma_spin = _uhf_dgamma_to_spin_blockstacked(dgamma_a, dgamma_b, nocc_a, nocc_b, norb_a, norb_b)
        return np.einsum('prqs,rs->pq', g_anti_spin, dgamma_spin)

    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    dgamma_spatial = _mp3_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=eri_chemist, u2_denom_dress=u2_denom_dress)
    return _spin_orbital_static_correction(g_anti_spin, dgamma_spatial)


def build_mp3_static_correction_restricted(mf, mol=None, nocc=None, relax=False, ncore=0, B_aa=None, u2_denom_dress=None):
    """Restricted (spatial-MO) counterpart of build_mp3_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted. RHF only -- use
    build_mp3_static_correction (spin-orbital) for UHF. Same
    KNOWN LIMITATION (relax=True) as build_mp3_static_correction -- see its docstring.

    B_aa: see build_mp2_static_correction_restricted's parameter of the same
    name -- when given, no dense eri_chemist norb^4 array is ever built.
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(
            "build_mp3_static_correction_restricted is RHF-only -- use "
            "build_mp3_static_correction (spin-orbital) for UHF.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2

    if B_aa is not None:
        dgamma_spatial = _mp3_dgamma_spatial(mf, mol, nocc, relax, ncore, B_aa=B_aa, u2_denom_dress=u2_denom_dress)
        return _static_correction_from_dgamma_restricted_df(B_aa, dgamma_spatial)

    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    dgamma_spatial = _mp3_dgamma_spatial(mf, mol, nocc, relax, ncore, eri_chemist=eri_chemist, u2_denom_dress=u2_denom_dress)
    return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)


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


def build_ccsd_static_correction_restricted(mf, mol=None, ncore=0):
    """Restricted (spatial-MO) counterpart of build_ccsd_static_correction, for
    ADCSolverRestricted/solve_ip_ea_restricted.

    ncore: number of frozen spatial core orbitals (e.g. 2 for C+O 1s).
    Returns the (nmo, nmo) restricted correction G[Delta_gamma], symmetric.
    """

    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("build_ccsd_static_correction_restricted is RHF-only.")

    mol = mol if mol is not None else mf.mol
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    dm_cc_ao = compute_ccsd_density_matrix(mf, ncore=ncore)
    dgamma_spatial = _dgamma_spatial_from_ao_density(mf, dm_cc_ao)
    return _static_correction_from_dgamma_restricted(eri_chemist, dgamma_spatial)


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
    built around (see ADCSolver's docstring). Same construction as
    qp_energy.py's diagonal-only xc_correction (built from
    SelfEnergySolver.calculate_sigma_hx), generalized here to the full matrix so
    it can be added directly to the F block like build_mp2_static_correction's
    G[Delta_gamma].

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
                 'mp3_relaxed', 'ccsd', 'ccsdt')

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
                            cphf_tol=None):
    """Sigma(infinity) static-correction dispatcher over the builders above;
    the KS double-counting correction is always added on top (zero-effect for
    plain HF references).

    kind: one of (None, 'mp2_unrelaxed', 'mp2_relaxed', 'mp3_unrelaxed',
        'mp3_relaxed', 'ccsd', 'ccsdt'), optionally EN-prefixed with
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
        (build_mp2/mp3_static_correction_restricted's existing DF hook).
    cphf_level_shift/cphf_max_cycle/cphf_tol: forwarded to the UHF CPHF/
        Z-vector solve (relax=True only, dense or df); see
        solve_cphf_relaxation_uhf /."""
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
                u2_denom_dress=en_dress)
        elif kind.startswith('mp3'):
            base = build_mp3_static_correction_restricted(
                mf, mol, nocc, relax=relax, ncore=ncore, B_aa=B_aa,
                u2_denom_dress=en_dress)
        elif kind == 'ccsd':
            base = build_ccsd_static_correction_restricted(mf, mol, ncore=ncore)
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
    elif kind == 'ccsd':
        base = build_ccsd_static_correction(mf, mol, ncore=ncore)
    else:
        base = build_ccsdt_static_correction(mf, mol)
    base = _add_ks(base, build_ks_static_correction(mf, mol))
    return _add_ks(base, build_solvent_static_correction(mf, mol))
