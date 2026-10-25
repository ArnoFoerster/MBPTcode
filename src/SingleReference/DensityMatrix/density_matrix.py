import numpy as np
from pyscf import scf

from src.SingleReference.GW.qp_solve import mean_field_static_operator
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.orbital_spaces import get_occ_virt_indices
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_density_fitting_coefficients,
    get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_block_eri,
)
from src.Base.numerics.constants import (
    DEFAULT_BROADENING_ETA,
    DEFAULT_BLOCK_SIZE,
    GW_DENSITY_SPIN_SUM,
    CPHF_MAX_CYCLE,
    CPHF_TOL,
    MPN_DENSITY_LAPLACE_NTAU,
)
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.cphf import solve as cphf_solve
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import (
    MPnDensityDriverUnrestricted)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted,
)


class GWDensityMatrixSolver(AmplitudeGenerator):
    """
    Builds the GW 1-particle density-matrix correction Delta_gamma^GW (unrelaxed and orbital-relaxed),
    restricted spin only.

    Screening reuses LinearResponseSolver/CasidaSolver and
    AmplitudeGenerator.get_chi_a, so the polarizability channel matches the
    one used for qp_energy.py's self-energy -- see build_screening.
    """

    def __init__(self, eps, df_coeff=None, eri_chemist=None, eta=DEFAULT_BROADENING_ETA, block_size=DEFAULT_BLOCK_SIZE):
        self.spin_mode = 'restricted'
        self.eta = eta
        self.block_size = block_size
        self.eps = eps
        self.df_coeff = df_coeff
        self.eri_chemist = eri_chemist
        self.norb = len(eps)
        if self.df_coeff is not None:
            self.naux = self.df_coeff.shape[0]

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)

    def build_screening(self, nocc, polarizability='RPA'):
        """Solve the Casida/BSE equation for the requested screening channel.

        polarizability: 'RPA' (Coulomb-only), 'TDHF' (bare exchange), or
        'BSE' (statically screened exchange, W_aux=W_RPA(0)). Returns (omega, X, Y).
        """
        lr_solver = LinearResponseSolver(
            self.eps, coeff_df=self.df_coeff, eri_chemist=self.eri_chemist,
            spin_mode='restricted', eta=self.eta
        )
        mode = polarizability.upper()
        if mode == 'RPA':
            build_kw = dict(lBSE=False)
        elif mode == 'TDHF':
            build_kw = dict(lBSE=True, W_aux=None)
        elif mode == 'BSE':
            w_aux = lr_solver.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]
            build_kw = dict(lBSE=True, W_aux=w_aux)
        else:
            raise ValueError(f"Unknown screening mode '{polarizability}'; choose 'RPA', 'TDHF', or 'BSE'.")

        omega, X, Y = CasidaSolver(
            *lr_solver.build_casida_matrices(nocc, **build_kw)).solve()
        return omega, X, Y

    def compute_unrelaxed_blocks(self, nocc, mf, mol, polarizability='RPA', omega=None, X=None, Y=None,
                                  spin_sum=GW_DENSITY_SPIN_SUM):
        """
        Spatial-MO blocks (dgamma_oo, dgamma_ov, dgamma_vv) of the unrelaxed GW density correction,
        scaled to add to mf.make_rdm1().

        ov block includes the static <i|Sigma_x - v_xc|a> term, so it vanishes in the HF limit.
        """
        if omega is None or X is None or Y is None:
            omega, X, Y = self.build_screening(nocc, polarizability=polarizability)

        eps = self.eps
        occ, virt = self._get_occ_virt_indices(eps, nocc)
        ei, ea = eps[occ], eps[virt]

        w = self.get_chi_a(nocc, X, Y, p_state=None)          # (nstate, norb, norb)
        w_ic = w[:, occ, :][:, :, virt]
        w_ak = w[:, virt, :][:, :, occ]
        w_ik = w[:, occ, :][:, :, occ]
        w_bc = w[:, virt, :][:, :, virt]

        den_ic = ei[None, :, None] - ea[None, None, :] - omega[:, None, None]
        den_ak = ea[None, :, None] - ei[None, None, :] + omega[:, None, None]
        t_ic = w_ic / den_ic
        t_ak = w_ak / den_ak

        dgamma_oo = -np.einsum('sic,sjc->ij', t_ic, t_ic)
        dgamma_vv = np.einsum('sak,sbk->ab', t_ak, t_ak)

        mo = mf.mo_coeff
        sx_ib = (mo.T @ mean_field_static_operator(mf, mol) @ mo)[np.ix_(occ, virt)]

        term1 = np.einsum('sik,sbk->ib', w_ik, t_ak)
        term2 = np.einsum('sic,sbc->ib', t_ic, w_bc)
        dgamma_ov = (term1 + term2 + sx_ib) / (ei[:, None] - ea[None, :])

        dgamma_oo = dgamma_oo * spin_sum
        dgamma_vv = dgamma_vv * spin_sum
        dgamma_ov = dgamma_ov * spin_sum
        return dgamma_oo, dgamma_ov, dgamma_vv

    def solve_relaxation(self, mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, max_cycle=CPHF_MAX_CYCLE, tol=CPHF_TOL):
        """Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov block; return the full relaxed MO-basis Delta_gamma.

        Pass a KS mf for the gKS Hessian, or a bare RHF object sharing mf's
        orbitals/energies for the TDHF Hessian.
        """
        return solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, max_cycle=max_cycle, tol=tol)


def solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, max_cycle=CPHF_MAX_CYCLE, tol=CPHF_TOL):
    """
    Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov block; return the full relaxed MO-basis Delta_gamma.

    Generic over the correction's origin (GW, MP2, ...) -- only the
    spatial-MO oo/ov/vv blocks and a Hessian mean-field object are needed.
    Pass a KS mf for the gKS Hessian, or a bare RHF object sharing mf's
    orbitals/energies for the TDHF Hessian.
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

    X_ai = cphf_solve(fvind, e_ia.T, Y.T, numerics=Numerics(
        scf={'cphf_tol': tol, 'cphf_max_cycle': max_cycle}))
    relaxed_ov = X_ai.reshape(nv, no).T

    dgamma = np.zeros((nmo, nmo))
    dgamma[:no, :no] = dgamma_oo
    dgamma[no:, no:] = dgamma_vv
    dgamma[:no, no:] = relaxed_ov
    dgamma[no:, :no] = relaxed_ov.T
    return dgamma


def solve_cphf_relaxation_uhf(mf_hess, nocc_a, nocc_b, dgamma_oo_a, dgamma_ov_a, dgamma_vv_a, dgamma_oo_b, dgamma_ov_b, dgamma_vv_b, max_cycle=CPHF_MAX_CYCLE, tol=CPHF_TOL, level_shift=0.0):
    """Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov blocks (UHF); return the full relaxed MO-basis Delta_gamma for alpha and beta.

    The per-spin-channel equation is (D + K) z = D*dgamma_ov - K[dgamma_oo+vv]
    with coupling factor 1 (K couples the two spin channels through vresp) --
    the counterpart of solve_cphf_relaxation's factor 2.0, which is a spin
    SUMMATION factor belonging to the restricted route only. Reduces exactly
    to solve_cphf_relaxation on a closed-shell UHF=RHF reference and satisfies
    the finite-field dipole sum rule on open shells (both asserted in
    tests/test_uhf_mp2_relaxed_finite_field.py).

    The solve is src.Base.pyscf.cphf.solve: pyscf's lib.krylov refined on the
    true residual; a solve ending above tol warns (CphfNotConverged) and
    returns its last iterate (lib.krylov alone stops on its
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
                           numerics=Numerics(scf={'cphf_tol': tol,
                                                  'cphf_max_cycle': max_cycle}),
                           precond=floor)

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


def compute_gw_density_matrix(mf, mol=None, nocc=None, polarizability='RPA', df=True,
                               relax=True, relax_kernel='self', eta=DEFAULT_BROADENING_ETA):
    """Build the total AO 1-particle density matrix gamma = gamma_mf + Delta_gamma^GW, restricted spin only.

    polarizability: 'RPA'/'TDHF'/'BSE' screening channel. df: use density
    fitting (default) vs full 4-center ERIs. relax: CPHF/CPKS-relaxed ov
    block (default) vs simple perturbative one. relax_kernel: 'self' (mf's
    own Hessian) or 'tdhf' (bare Hartree + exact-exchange Hessian).
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("compute_gw_density_matrix is currently restricted-spin only.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)

    if df:
        df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        eri_chemist = None
    else:
        df_coeff = None
        eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    solver = GWDensityMatrixSolver(eps, df_coeff=df_coeff, eri_chemist=eri_chemist, eta=eta)
    dgamma_oo, dgamma_ov, dgamma_vv = solver.compute_unrelaxed_blocks(
        nocc, mf, mol, polarizability=polarizability
    )

    mo = mf.mo_coeff
    if not relax:
        dgamma = np.zeros((nmo, nmo))
        dgamma[:nocc, :nocc] = dgamma_oo
        dgamma[nocc:, nocc:] = dgamma_vv
        dgamma[:nocc, nocc:] = dgamma_ov
        dgamma[nocc:, :nocc] = dgamma_ov.T
    else:
        if relax_kernel == 'tdhf':
            mf_hess = scf.RHF(mol)
            mf_hess.mo_coeff = mo
            mf_hess.mo_energy = mf.mo_energy
            mf_hess.mo_occ = mf.mo_occ
        else:
            mf_hess = mf
        dgamma = solver.solve_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao


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
                                  relax_kernel='self'):
    """The total AO 1-particle density gamma = gamma_mf + Delta_gamma^(2) [+
    Delta_gamma^(3)] of MP2 (order=2) or MP3 (order=3), from the generated
    MPn density drivers (restricted for RHF, unrestricted for UHF; a UHF
    returns (dm_a, dm_b)).

    relax: the CPHF/CPKS Z-vector solve for the ov block (solve_cphf_relaxation,
    solve_cphf_relaxation_uhf) instead of the direct second(+third)-order
    singles. relax_kernel: 'self' (mf's own Hessian) or 'tdhf' (bare Hartree +
    exact-exchange Hessian, RHF only).

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
                mf, nocc_a, nocc_b, oo_a, ov_a, vv_a, oo_b, ov_b, vv_b)
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
        dgamma = solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    return mf.make_rdm1() + mo @ dgamma @ mo.T


def compute_mp2_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self'):
    """compute_mpn_density_matrix_ao at order 2."""
    return compute_mpn_density_matrix_ao(mf, 2, mol, nocc, relax=relax, relax_kernel=relax_kernel)


def compute_mp3_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self'):
    """compute_mpn_density_matrix_ao at order 3 (see its MP3 relaxation caveat)."""
    return compute_mpn_density_matrix_ao(mf, 3, mol, nocc, relax=relax, relax_kernel=relax_kernel)
