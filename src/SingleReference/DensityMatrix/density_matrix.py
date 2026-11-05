import numpy as np
from pyscf import scf
from pyscf.df import df_jk

from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_antisymmetrized_spin_block_eri,
)
from src.Base.numerics.constants import MPN_DENSITY_LAPLACE_NTAU
from src.Base.numerics.numerics import required_numerics
from src.Base.pyscf.cphf import solve as cphf_solve
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import (
    MPnDensityDriverUnrestricted)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted,
)


def relaxation_numerics(relax, numerics, where):
    """`numerics` where `relax` asks for the Z-vector solve, refused there
    when None (`required_numerics`); not read otherwise."""
    return required_numerics(numerics, where) if relax else numerics


def hf_kernel_mean_field(mf):
    """A bare RHF carrying a restricted `mf`'s orbitals, energies and
    occupations and its J/K backend, so that its `gen_response` is the HF
    kernel J - K/2 on `mf`'s own integrals: the Hessian object of an
    HF-kernel relaxation (`solve_cphf_relaxation`) on a gKS reference.

    The backend is the one `mf` builds J and K with: exact four-centre
    integrals where `mf` has no `with_df`; the same `with_df` object
    (density fit or ISDF-K, `isdf_jk`) where `mf` is a pyscf density-fitted
    mean field. Any other backend is refused rather than replaced by exact
    integrals. Only the kernel changes: no exchange-correlation kernel, full
    exact exchange (no fraction, no range separation)."""
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)) or not isinstance(mf, scf.hf.RHF):
        raise TypeError('hf_kernel_mean_field takes a restricted closed-shell '
                        f'mean field, got {type(mf).__name__}')
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        hf = scf.RHF(mf.mol)
    elif isinstance(mf, df_jk._DFHF):
        hf = scf.RHF(mf.mol).density_fit()
        hf.with_df = with_df
        hf.only_dfj = mf.only_dfj
    else:
        raise TypeError(
            f'hf_kernel_mean_field cannot reproduce the J/K backend of {type(mf).__name__} '
            f'(with_df {type(with_df).__name__}): only exact integrals and pyscf '
            'density fitting (DF or ISDF-K via isdf_jk) are carried over; exact '
            'integrals are not substituted for a fitted interaction.')
    hf.max_memory = mf.max_memory
    hf.mo_coeff = mf.mo_coeff
    hf.mo_energy = mf.mo_energy
    hf.mo_occ = mf.mo_occ
    return hf


def solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics):
    """
    Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov block; return the full relaxed MO-basis Delta_gamma.

    Generic over the correction's origin (GW, MP2, ...) -- only the
    spatial-MO oo/ov/vv blocks and a Hessian mean-field object are needed.
    Pass a KS mf for the gKS Hessian, or a bare RHF object sharing mf's
    orbitals/energies for the TDHF Hessian. numerics: the run's `Numerics`,
    required; its scf section's multiplier_tol, multiplier_max_iter and
    multiplier_residual_tol set the solve (src.Base.pyscf.cphf.solve).
    """
    mo_coeff = mf_hess.mo_coeff
    mo_energy = mf_hess.mo_energy
    mo_occ = mf_hess.mo_occ
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]
    no, nv = orbo.shape[1], orbv.shape[1]
    nmo = mo_energy.size
    e_ia = mo_energy[mo_occ == 0][None, :] - mo_energy[mo_occ > 0][:, None]
    vresp = mf_hess.gen_response(hermi=1)

    def fvind(x):
        x = x.reshape(nv, no)
        dm1 = orbv @ x @ orbo.T
        dm1 = dm1 + dm1.T
        return (2.0 * orbv.T @ vresp(dm1) @ orbo).ravel()

    D_corr = np.zeros((nmo, nmo))
    D_corr[:no, :no] = dgamma_oo
    D_corr[no:, no:] = dgamma_vv
    coupling_ov = orbo.T @ vresp(mo_coeff @ D_corr @ mo_coeff.T) @ orbv
    Y = e_ia * dgamma_ov - 2.0 * coupling_ov

    X_ai = cphf_solve(fvind, e_ia.T, Y.T, numerics=numerics)
    relaxed_ov = X_ai.reshape(nv, no).T

    dgamma = np.zeros((nmo, nmo))
    dgamma[:no, :no] = dgamma_oo
    dgamma[no:, no:] = dgamma_vv
    dgamma[:no, no:] = relaxed_ov
    dgamma[no:, :no] = relaxed_ov.T
    return dgamma


def solve_cphf_relaxation_uhf(mf_hess, nocc_a, nocc_b, dgamma_oo_a, dgamma_ov_a, dgamma_vv_a, dgamma_oo_b, dgamma_ov_b, dgamma_vv_b, numerics, level_shift=0.0):
    """Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov blocks (UHF); return the full relaxed MO-basis Delta_gamma for alpha and beta.

    The per-spin-channel equation is (D + K) z = D*dgamma_ov - K[dgamma_oo+vv]
    with coupling factor 1 (K couples the two spin channels through vresp) --
    the counterpart of solve_cphf_relaxation's factor 2.0, which is a spin
    SUMMATION factor belonging to the restricted route only. Reduces exactly
    to solve_cphf_relaxation on a closed-shell UHF=RHF reference and satisfies
    the finite-field dipole sum rule on open shells (both asserted in
    tests/test_uhf_mp2_relaxed_finite_field.py).

    The solve is src.Base.pyscf.cphf.solve on `numerics`, the run's, required
    (lgmres to its scf section's multiplier_tol relative to max(1, |rhs|), at
    most multiplier_max_iter iterations), checked on its true residual; a
    solve ending above the bar warns (CphfNotConverged) and returns its last
    iterate (pyscf's lib.krylov alone stops on its
    linear-dependence test on strongly spin-contaminated references).
    level_shift floors only the PRECONDITIONER's denominators (sign-
    preserving, |e_ia| >= level_shift); the operator keeps the exact e_ia, so
    it changes how fast the solve converges, never what it converges to: the
    knob for small occupied-virtual gaps.
    """
    mo_coeff_a, mo_coeff_b = mf_hess.mo_coeff
    mo_energy_a, mo_energy_b = mf_hess.mo_energy
    mo_occ_a, mo_occ_b = mf_hess.mo_occ

    orbo_a = mo_coeff_a[:, mo_occ_a > 0]
    orbv_a = mo_coeff_a[:, mo_occ_a == 0]
    orbo_b = mo_coeff_b[:, mo_occ_b > 0]
    orbv_b = mo_coeff_b[:, mo_occ_b == 0]

    no_a, nv_a = orbo_a.shape[1], orbv_a.shape[1]
    no_b, nv_b = orbo_b.shape[1], orbv_b.shape[1]
    nmo = mo_energy_a.size

    e_ia_a = mo_energy_a[mo_occ_a == 0][None, :] - mo_energy_a[mo_occ_a > 0][:, None]
    e_ia_b = mo_energy_b[mo_occ_b == 0][None, :] - mo_energy_b[mo_occ_b > 0][:, None]

    vresp = mf_hess.gen_response(hermi=1)

    def fvind(x):
        x_a = x[:nv_a * no_a].reshape(nv_a, no_a)
        x_b = x[nv_a * no_a:].reshape(nv_b, no_b)

        dm1_a = orbv_a @ x_a @ orbo_a.T
        dm1_b = orbv_b @ x_b @ orbo_b.T
        dm1_a = dm1_a + dm1_a.T
        dm1_b = dm1_b + dm1_b.T

        v_a, v_b = vresp((dm1_a, dm1_b))

        # Coupling factor 1 per spin channel (pyscf grad/ump2 _response_dm1
        # convention): the RHF route's 2.0 is a SPIN-SUMMATION factor and
        # must not appear in the spin-resolved UHF equation.
        resp_a = (orbv_a.T @ v_a @ orbo_a).ravel()
        resp_b = (orbv_b.T @ v_b @ orbo_b).ravel()
        return np.concatenate([resp_a, resp_b])

    D_corr_a = np.zeros((nmo, nmo))
    D_corr_a[:no_a, :no_a] = dgamma_oo_a
    D_corr_a[no_a:, no_a:] = dgamma_vv_a

    D_corr_b = np.zeros((nmo, nmo))
    D_corr_b[:no_b, :no_b] = dgamma_oo_b
    D_corr_b[no_b:, no_b:] = dgamma_vv_b

    dm_a = mo_coeff_a @ D_corr_a @ mo_coeff_a.T
    dm_b = mo_coeff_b @ D_corr_b @ mo_coeff_b.T

    v_a, v_b = vresp((dm_a, dm_b))

    coupling_ov_a = orbo_a.T @ v_a @ orbv_a
    coupling_ov_b = orbo_b.T @ v_b @ orbv_b

    Y_a = e_ia_a * dgamma_ov_a - coupling_ov_a
    Y_b = e_ia_b * dgamma_ov_b - coupling_ov_b

    Y_flat = np.concatenate([Y_a.T.ravel(), Y_b.T.ravel()])
    h_diag = np.concatenate([e_ia_a.T.ravel(), e_ia_b.T.ravel()])
    # Level-shifted, sign-preserving floor on the PRECONDITIONER only: the
    # operator keeps the exact h_diag, so the solution does not depend on it
    if level_shift:
        floor = np.sign(h_diag) * np.maximum(np.abs(h_diag), level_shift)
    else:
        floor = h_diag
    X_ai_flat = cphf_solve(lambda x: fvind(np.ravel(x)), h_diag, Y_flat,
                           numerics=numerics, precond=floor)

    X_ai_a = X_ai_flat[:nv_a * no_a].reshape(nv_a, no_a)
    X_ai_b = X_ai_flat[nv_a * no_a:].reshape(nv_b, no_b)

    relaxed_ov_a = X_ai_a.T
    relaxed_ov_b = X_ai_b.T

    dgamma_a = np.zeros((nmo, nmo))
    dgamma_a[:no_a, :no_a] = dgamma_oo_a
    dgamma_a[no_a:, no_a:] = dgamma_vv_a
    dgamma_a[:no_a, no_a:] = relaxed_ov_a
    dgamma_a[no_a:, :no_a] = relaxed_ov_a.T

    dgamma_b = np.zeros((nmo, nmo))
    dgamma_b[:no_b, :no_b] = dgamma_oo_b
    dgamma_b[no_b:, no_b:] = dgamma_vv_b
    dgamma_b[:no_b, no_b:] = relaxed_ov_b
    dgamma_b[no_b:, :no_b] = relaxed_ov_b.T

    return dgamma_a, dgamma_b


def _semicanonicalize_fock_blocks(F_mo, nocc):
    """
    Diagonalize the occ-occ and virt-virt blocks of a (possibly non-diagonal)
    spatial-MO Fock matrix F_mo separately, leaving a residual ov coupling.
    """
    eps_oo, U_oo = np.linalg.eigh(F_mo[:nocc, :nocc])
    eps_vv, U_vv = np.linalg.eigh(F_mo[nocc:, nocc:])
    eps_semi = np.concatenate([eps_oo, eps_vv])
    f_ov_semi = U_oo.T @ F_mo[:nocc, nocc:] @ U_vv
    return eps_semi, U_oo, U_vv, f_ov_semi


def semicanonicalize_restricted(mf, mol=None, dm=None, nocc=None):
    """
    MP2/MP3 ... can not be run in KS basis.
    To do so: RHF/RKS semicanonicalizationrotate exists.
    Rotate mf's own (KS) occupied and virtual MO subspaces
    separately so that F_HF[gamma] = h + Sigma_Hx[gamma] is block-diagonal
    in the new basis, with a residual ov coupling f_ov_semi
    that cannot be rotated away. See _semicanonicalize_fock_blocks.
    """
    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    dm = dm if dm is not None else mf.make_rdm1(mf.mo_coeff, mf.mo_occ)

    mf_hf = scf.RHF(mol)
    F_ao = mf_hf.get_hcore(mol) + mf_hf.get_veff(mol, dm)
    mo = mf.mo_coeff
    F_mo = mo.T @ F_ao @ mo

    eps_semi, U_oo, U_vv, f_ov_semi = _semicanonicalize_fock_blocks(F_mo, nocc)

    mo_coeff_semi = mo.copy()
    mo_coeff_semi[:, :nocc] = mo[:, :nocc] @ U_oo
    mo_coeff_semi[:, nocc:] = mo[:, nocc:] @ U_vv
    return mo_coeff_semi, eps_semi, f_ov_semi, U_oo, U_vv


def semicanonicalize_uhf(mf, mol=None, dm=None):
    """UHF/UKS counterpart of semicanonicalize_restricted: separate alpha/beta
    semicanonicalization (Sigma_Hx[gamma] is spin-diagonal, so there is no cross-spin
    coupling to remove or leave behind). Returns a tuple
    (mo_coeff_semi_a, eps_semi_a, f_ov_semi_a, U_oo_a, U_vv_a,
     mo_coeff_semi_b, eps_semi_b, f_ov_semi_b, U_oo_b, U_vv_b).
    """
    mol = mol if mol is not None else mf.mol
    nocc_a, nocc_b = mf.nelec
    dm = dm if dm is not None else mf.make_rdm1(mf.mo_coeff, mf.mo_occ)

    mf_hf = scf.UHF(mol)
    h_ao = mf_hf.get_hcore(mol)
    V_Hx_a, V_Hx_b = mf_hf.get_veff(mol, dm)
    mo_a, mo_b = mf.mo_coeff

    F_mo_a = mo_a.T @ (h_ao + V_Hx_a) @ mo_a
    F_mo_b = mo_b.T @ (h_ao + V_Hx_b) @ mo_b

    eps_semi_a, U_oo_a, U_vv_a, f_ov_semi_a = _semicanonicalize_fock_blocks(F_mo_a, nocc_a)
    eps_semi_b, U_oo_b, U_vv_b, f_ov_semi_b = _semicanonicalize_fock_blocks(F_mo_b, nocc_b)

    mo_coeff_semi_a = mo_a.copy()
    mo_coeff_semi_a[:, :nocc_a] = mo_a[:, :nocc_a] @ U_oo_a
    mo_coeff_semi_a[:, nocc_a:] = mo_a[:, nocc_a:] @ U_vv_a
    mo_coeff_semi_b = mo_b.copy()
    mo_coeff_semi_b[:, :nocc_b] = mo_b[:, :nocc_b] @ U_oo_b
    mo_coeff_semi_b[:, nocc_b:] = mo_b[:, nocc_b:] @ U_vv_b

    return (mo_coeff_semi_a, eps_semi_a, f_ov_semi_a, U_oo_a, U_vv_a,
            mo_coeff_semi_b, eps_semi_b, f_ov_semi_b, U_oo_b, U_vv_b)


def t1_singles_blocks(f_ov, eps_o, eps_v):
    """First-order T1 singles amplitude t_i^{a(1)} = f_ia/(ei-ea) for a
    non-Brillouin (e.g. KS) reference, and its density contributions.

    f_ov must be evaluated in a basis where eps_o/eps_v are themselves diagonal
    (i.e. the semicanonical basis from semicanonicalize_restricted/_uhf), or the
    (ei-ea) denominator is meaningless -- for an actual HF reference f_ov ~ 0 and
    this is a no-op.

    Returns (t1, dgamma_oo, dgamma_ov, dgamma_vv):
        dgamma_ov = t1 -- the direct, FIRST-order density contribution (new;
            does not exist for a canonical-HF/Brillouin reference).
        dgamma_oo/dgamma_vv = -t1.t1^T / +t1^T.t1 -- the singles-singles
            (CIS-density-like) SECOND-order contribution.
    """
    t1 = f_ov / (eps_o[:, None] - eps_v[None, :])
    dgamma_oo = -np.einsum('ia,ja->ij', t1, t1, optimize=True)
    dgamma_vv = np.einsum('ia,ib->ab', t1, t1, optimize=True)
    return t1, dgamma_oo, t1, dgamma_vv


def build_dgamma_block(nmo, nocc, oo, ov, vv):
    dgamma = np.zeros((nmo, nmo))
    dgamma[:nocc, :nocc] = oo
    dgamma[nocc:, nocc:] = vv
    dgamma[:nocc, nocc:] = ov
    dgamma[nocc:, :nocc] = ov.T
    return dgamma


def compute_mpn_density_matrix_ao(mf, order, mol=None, nocc=None, relax=True,
                                  relax_kernel='self', numerics=None):
    """The total AO 1-particle density gamma = gamma_mf + Delta_gamma^(2) [+
    Delta_gamma^(3)] of MP2 (order=2) or MP3 (order=3), from the generated
    MPn density drivers (restricted for RHF, unrestricted for UHF; a UHF
    returns (dm_a, dm_b)).

    relax: the CPHF/CPKS Z-vector solve for the ov block (solve_cphf_relaxation,
    solve_cphf_relaxation_uhf) instead of the direct second(+third)-order
    singles. relax_kernel: 'self' (mf's own Hessian) or 'tdhf' (bare Hartree +
    exact-exchange Hessian, RHF only). numerics: the run's `Numerics`, the
    Z-vector's; required with relax=True.

    relax=True at MP3 is NOT a fully relaxed density: the orbitals are relaxed
    by the Schirmer approach, the amplitudes never are. It therefore does NOT
    satisfy the finite-field/Hellmann-Feynman dipole sum rule (mu_z =
    -Tr[gamma @ z_AO] should equal -dE/dF_z), and what is left over is the
    size of the missing amplitude response: 4.3e-03 on HF/6-31g, against
    2.7e-06 for MP2 relaxed through the same CPHF solve. Pinned in
    tests/test_mp3_finite_field.py."""
    if order not in (2, 3):
        raise ValueError(f"order={order}: the AO MPn density is written for MP2 and MP3")
    if relax_kernel not in ('self', 'tdhf'):
        raise ValueError(f"relax_kernel={relax_kernel!r}; expected 'self' or 'tdhf'")
    numerics = relaxation_numerics(relax, numerics, 'compute_mpn_density_matrix_ao')
    mol = mol if mol is not None else mf.mol

    if isinstance(mf, scf.uhf.UHF):
        if relax and relax_kernel != 'self':
            raise ValueError("relax_kernel='tdhf' is the RHF route's option")
        nocc_a, nocc_b = mf.nelec
        eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        driver = MPnDensityDriverUnrestricted(np.diag(eps_a), np.diag(eps_b),
                                              g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
        if order == 2:
            oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = driver.compute_delta_gamma2()
        else:
            (oo2_a, oo2_b, ov2_a, ov2_b, vv2_a, vv2_b), (oo3_a, oo3_b, ov3_a, ov3_b, vv3_a, vv3_b) = \
                driver.compute_delta_gamma23()
            oo_a, ov_a, vv_a = oo2_a + oo3_a, ov2_a + ov3_a, vv2_a + vv3_a
            oo_b, ov_b, vv_b = oo2_b + oo3_b, ov2_b + ov3_b, vv2_b + vv3_b
        mo_a, mo_b = mf.mo_coeff
        if relax:
            dgamma_a, dgamma_b = solve_cphf_relaxation_uhf(
                mf, nocc_a, nocc_b, oo_a, ov_a, vv_a, oo_b, ov_b, vv_b, numerics)
        else:
            dgamma_a = build_dgamma_block(mo_a.shape[1], nocc_a, oo_a, ov_a, vv_a)
            dgamma_b = build_dgamma_block(mo_b.shape[1], nocc_b, oo_b, ov_b, vv_b)
        dm_a, dm_b = mf.make_rdm1()
        return dm_a + mo_a @ dgamma_a @ mo_a.T, dm_b + mo_b @ dgamma_b @ mo_b.T

    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
    driver = MPnDensityDriverRestricted(np.diag(eps), g_aaaa, g_abab, g_bbbb, nocc)
    if order == 2:
        oo_a, ov_a, vv_a = driver.compute_delta_gamma2()
        dgamma_oo, dgamma_ov, dgamma_vv = 2.0 * oo_a, 2.0 * ov_a, 2.0 * vv_a
    else:
        (oo2_a, ov2_a, vv2_a), (oo3_a, ov3_a, vv3_a) = driver.compute_delta_gamma23(
            laplace_ntau=MPN_DENSITY_LAPLACE_NTAU)
        dgamma_oo = 2.0 * (oo2_a + oo3_a)
        dgamma_ov = 2.0 * (ov2_a + ov3_a)
        dgamma_vv = 2.0 * (vv2_a + vv3_a)

    mo = mf.mo_coeff
    if not relax:
        dgamma = build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    else:
        if relax_kernel == 'tdhf':
            mf_hess = scf.RHF(mol)
            mf_hess.mo_coeff = mo
            mf_hess.mo_energy = mf.mo_energy
            mf_hess.mo_occ = mf.mo_occ
        else:
            mf_hess = mf
        dgamma = solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv,
                                       numerics)
    return mf.make_rdm1() + mo @ dgamma @ mo.T


def compute_mp2_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self',
                                  numerics=None):
    """compute_mpn_density_matrix_ao at order 2."""
    return compute_mpn_density_matrix_ao(mf, 2, mol, nocc, relax=relax, relax_kernel=relax_kernel,
                                         numerics=numerics)


def compute_mp3_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self',
                                  numerics=None):
    """compute_mpn_density_matrix_ao at order 3 (see its MP3 relaxation caveat)."""
    return compute_mpn_density_matrix_ao(mf, 3, mol, nocc, relax=relax, relax_kernel=relax_kernel,
                                         numerics=numerics)
