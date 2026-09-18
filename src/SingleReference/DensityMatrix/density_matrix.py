import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from pyscf import lib, scf
from pyscf.scf import cphf, ucphf

from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.base import get_occ_virt_indices
from src.Base.pyscf_interface import (
    get_orbital_energies,
    get_density_fitting_coefficients,
    get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri,
    get_antisymmetrized_spin_block_eri,
)
from src.Base.constants import (
    DEFAULT_BROADENING_ETA,
    DEFAULT_BLOCK_SIZE,
    GW_DENSITY_SPIN_SUM,
    CPHF_MAX_CYCLE,
    CPHF_TOL,
)
from src.Base.utils.grids import minimax_time_grid
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
        dm = mf.make_rdm1()
        Sigx = -0.5 * mf.get_k(mol, dm)
        vxc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
        sx_ib = (mo.T @ (Sigx - vxc) @ mo)[np.ix_(occ, virt)]

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

    X_ai, _ = cphf.solve(fvind, mo_energy, mo_occ, -Y.T, max_cycle=max_cycle, tol=tol)
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

    DELIBERATELY NOT USED: routing this through pyscf's own
    ucphf.solve (the UHF counterpart of solve_cphf_relaxation's cphf.solve)
    looked like the natural fix for the stalls seen on strongly
    spin-contaminated systems (C2H3, C3H3) -- but pyscf's underlying
    lib.krylov solver terminates EARLY via its own lindep (linear-dependence)
    threshold on this equation, before actually satisfying its own x+aop(x)=b
    contract (verified directly: raw contract residual ~1e-3 relative,
    reproducibly, regardless of max_cycle). That's a real quirk of
    lib.krylov on this particular inhomogeneous-source CPHF variant, not a
    sign/shape bug in the call -- confirmed by building the dense (nv*no,
    nv*no) operator explicitly and comparing exact numpy.linalg.solve against
    both solvers. scipy gmres below is exact (residual ~1e-13 against that
    same dense ground truth) even though slow/non-convergent on the hardest
    cases, so kept as the base solver. level_shift here only floors the
    PRECONDITIONER's denominator (M_matvec below) -- it does not touch the
    actual operator A (fvind's full e_ia+coupling), so it cannot change what
    gmres converges to, only how fast; a small nonzero value (e.g. 0.02-0.1)
    is the safe knob to try when a system's occ-virt gaps get small enough
    to make the naive 1/e_ia preconditioner blow up.
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
        # Add the diagonal part (e_ia * x) because gmres solves A*x = Y, where A = diag(e_ia) + K
        return np.concatenate([resp_a, resp_b]) + x * np.concatenate([e_ia_a.T.ravel(), e_ia_b.T.ravel()])

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
    # Level-shifted, sign-preserving floor on the PRECONDITIONER only (not on
    # h_diag as fed into fvind's exact diagonal above): avoids 1/h_diag
    # blowing up for small occ-virt gaps without perturbing the operator A
    # itself, so the gmres solution is exact regardless of level_shift.
    if level_shift:
        floor = np.sign(h_diag) * np.maximum(np.abs(h_diag), level_shift)
    else:
        floor = h_diag

    def M_matvec(x):
        return x / floor

    A = LinearOperator((Y_flat.size, Y_flat.size), matvec=fvind)
    M = LinearOperator((Y_flat.size, Y_flat.size), matvec=M_matvec)

    X_ai_flat, info = gmres(A, Y_flat, M=M, rtol=tol, atol=tol, maxiter=max_cycle)
    if info > 0:
        print(f"Warning: gmres did not converge after {info} iterations.")
    elif info < 0:
        print(f"Warning: gmres failed with code {info}.")

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


class MP2DensityMatrixSolver:
    """
    Builds the unrelaxed MP2 1-particle density-matrix correction Delta_gamma^(2).

    Restricted spin only.

    Works in the interleaved spin-orbital basis with antisymmetrized <pq||rs>
    integrals, matching AmplitudeGenerator/get_antisymmetrized_spin_eri's
    convention. The ov block comes from the second-order singles amplitude
    t_i^{a(2)} (no CPHF/Z-vector solve), analogous to
    GWDensityMatrixSolver.compute_unrelaxed_blocks.
    """

    def __init__(self, eps_spin, g_anti_spin, nocc_spin, ncore_spin=0):
        self.eps = eps_spin
        self.g = g_anti_spin
        self.nocc = nocc_spin
        self.ncore = ncore_spin
        self.norb = len(eps_spin)

    def compute_t2(self):
        """t2[a,b,i,j] = t_ij^{ab(1)} = <ab||ij> / (ei+ej-ea-eb)."""
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        ei, ea = self.eps[occ], self.eps[virt]
        g_abij = self.g[virt, virt, occ, occ]
        denom = (ei[None, None, :, None] + ei[None, None, None, :]
                 - ea[:, None, None, None] - ea[None, :, None, None])
        return g_abij / denom

    def compute_blocks(self):
        """
        Spin-orbital blocks (dgamma_oo, dgamma_ov, dgamma_vv)
        of the unrelaxed MP2 density correction.

        When ncore_spin > 0, the returned oo/ov blocks are over active occupied
        orbitals only (dimensions nocc_active × nocc_active and nocc_active × nvirt).
        """
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        t2 = self.compute_t2()
        dgamma_oo = -0.5 * np.einsum('abik,abij->kj', t2, t2)
        dgamma_vv = 0.5 * np.einsum('abij,acij->bc', t2, t2)

        g_ajbc = g[virt, occ, virt, virt]
        g_jkib = g[occ, occ, occ, virt]
        term1 = 0.5 * np.einsum('ajbc,bcij->ia', g_ajbc, t2)
        term2 = 0.5 * np.einsum('jkib,abjk->ia', g_jkib, t2)
        dgamma_ov = (term1 - term2) / (ei[:, None] - ea[None, :])

        return dgamma_oo, dgamma_ov, dgamma_vv

    def compute_blocks_spatial(self):
        """
        Spatial-MO blocks, folding the interleaved spin-orbital blocks.
        restricted reference: alpha block == beta block, so spatial = 2*alpha-alpha).

        When ncore_spin > 0, the active-occ blocks are embedded into full
        (nocc_spatial, nocc_spatial) arrays with zeros in core rows/columns.
        """
        dgamma_oo_act, dgamma_ov_act, dgamma_vv_act = self.compute_blocks()
        oo_sp = dgamma_oo_act[0::2, 0::2] * 2.0
        ov_sp = dgamma_ov_act[0::2, 0::2] * 2.0
        vv_sp = dgamma_vv_act[0::2, 0::2] * 2.0

        ncore_spatial = self.ncore // 2
        if ncore_spatial == 0:
            return oo_sp, ov_sp, vv_sp

        # Embed active-occ blocks into full spatial-MO occupied space
        nocc_spatial = self.nocc // 2
        full_oo = np.zeros((nocc_spatial, nocc_spatial))
        full_oo[ncore_spatial:, ncore_spatial:] = oo_sp
        full_ov = np.zeros((nocc_spatial, vv_sp.shape[0]))
        full_ov[ncore_spatial:, :] = ov_sp
        return full_oo, full_ov, vv_sp


class MP3DensityMatrixSolver(MP2DensityMatrixSolver):
    """Builds the unrelaxed MP3 (third-order) 1-particle density-matrix correction Delta_gamma^(3)

    restricted spin only.
    """

    def compute_t1_second_order(self, t2_1=None):
        """
        t1_2[i,a] = t_i^{a(2)} -- same expression as MP2DensityMatrixSolver.compute_blocks's ov block.
        """
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term1 = 0.5 * np.einsum('ajbc,bcij->ia', g[virt, occ, virt, virt], t2_1, optimize=True)
        term2 = 0.5 * np.einsum('jkib,abjk->ia', g[occ, occ, occ, virt], t2_1, optimize=True)
        return (term1 - term2) / (ei[:, None] - ea[None, :])

    def compute_t2_second_order(self, t2_1=None):
        """
        t2_2[a,b,i,j] = t_ij^{ab(2)}.
        """
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term2 = 0.5 * np.einsum('klij,abkl->abij', g[occ, occ, occ, occ], t2_1, optimize=True)

        raw3 = np.einsum('kacj,cbik->abij', g[occ, virt, virt, occ], t2_1, optimize=True)
        term3 = (raw3 - raw3.transpose(1, 0, 2, 3)
                 - raw3.transpose(0, 1, 3, 2) + raw3.transpose(1, 0, 3, 2))

        term4 = 0.5 * np.einsum('abcd,cdij->abij', g[virt, virt, virt, virt], t2_1, optimize=True)

        denom = (ei[None, None, :, None] + ei[None, None, None, :]
                 - ea[:, None, None, None] - ea[None, :, None, None])
        return (term2 + term3 + term4) / denom

    def compute_t3_second_order(self, t2_1=None):
        """t3_2[a,b,c,i,j,k] = t_ijk^{abc(2)}."""
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        raw1 = np.einsum('lajk,bcil->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T1 = -(raw1 - raw1.transpose(1, 0, 2, 3, 4, 5)
               - raw1.transpose(0, 1, 2, 4, 3, 5) + raw1.transpose(1, 0, 2, 4, 3, 5))

        raw2 = np.einsum('abdk,dcij->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T2 = -(raw2 - raw2.transpose(0, 2, 1, 3, 4, 5)
               - raw2.transpose(0, 1, 2, 3, 5, 4) + raw2.transpose(0, 2, 1, 3, 5, 4))

        raw3 = np.einsum('laij,bckl->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T3 = -(raw3 - raw3.transpose(1, 0, 2, 3, 4, 5))

        raw4 = np.einsum('lcjk,abil->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T4 = -(raw4 - raw4.transpose(0, 1, 2, 4, 3, 5))

        raw5 = np.einsum('abdi,dcjk->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T5 = -(raw5 - raw5.transpose(0, 2, 1, 3, 4, 5))

        raw6 = np.einsum('bcdk,daij->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T6 = -(raw6 - raw6.transpose(0, 1, 2, 3, 5, 4))

        T7 = -np.einsum('lcij,abkl->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T8 = -np.einsum('bcdi,dajk->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        numerator = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8

        denom = (ei[None, None, None, :, None, None] + ei[None, None, None, None, :, None]
                 + ei[None, None, None, None, None, :] - ea[:, None, None, None, None, None]
                 - ea[None, :, None, None, None, None] - ea[None, None, :, None, None, None])
        return numerator / denom

    def compute_t1_third_order(self, t1_2=None, t2_2=None, t3_2=None):
        """
        t1_3[i,a] = t_i^{a(3)}.
        """
        if t1_2 is None:
            t1_2 = self.compute_t1_second_order()
        if t2_2 is None:
            t2_2 = self.compute_t2_second_order()
        if t3_2 is None:
            t3_2 = self.compute_t3_second_order()

        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term2 = np.einsum('jabi,jb->ia', g[occ, virt, virt, occ], t1_2, optimize=True)
        term3 = -0.5 * np.einsum('kjbi,bakj->ia', g[occ, occ, virt, occ], t2_2, optimize=True)
        term4 = -0.5 * np.einsum('jabc,bcij->ia', g[occ, virt, virt, virt], t2_2, optimize=True)

        # t3_2.transpose(3,5,4,1,2,0)[i,k,j,b,c,a] == t3_2[a,b,c,i,j,k] (a straight axis
        # relabeling, reordering t3_2's (a,b,c,i,j,k) axes to (i,k,j,b,c,a)).
        t3_2_reordered = t3_2.transpose(3, 5, 4, 1, 2, 0)
        term5 = -0.25 * np.einsum('kjbc,ikjbca->ia', g[occ, occ, virt, virt], t3_2_reordered, optimize=True)

        numerator = term2 + term3 + term4 + term5
        return numerator / (ei[:, None] - ea[None, :])

    def compute_gamma3_blocks(self):
        """Spin-orbital blocks (dgamma_oo, dgamma_ov, dgamma_vv) of Delta_gamma^(3)."""
        t2_1 = self.compute_t2()
        t1_2 = self.compute_t1_second_order(t2_1)
        t2_2 = self.compute_t2_second_order(t2_1)
        t3_2 = self.compute_t3_second_order(t2_1)
        t1_3 = self.compute_t1_third_order(t1_2, t2_2, t3_2)

        gamma3_vv_raw = np.einsum('acij,bcij->ab', t2_1, t2_2, optimize=True)
        dgamma_vv = 0.5 * (gamma3_vv_raw + gamma3_vv_raw.T)

        gamma3_oo_raw = np.einsum('abik,abjk->ij', t2_1, t2_2, optimize=True)
        dgamma_oo = -0.5 * (gamma3_oo_raw + gamma3_oo_raw.T)

        term_b = np.einsum('jc,acij->ia', t1_2, t2_1, optimize=True)
        term_c = 0.25 * np.einsum('bcjk,abcijk->ia', t2_1, t3_2, optimize=True)
        dgamma_ov = t1_3 + term_b + term_c

        return dgamma_oo, dgamma_ov, dgamma_vv

    def compute_gamma3_blocks_spatial(self):
        """Spatial-MO blocks, folding the interleaved spin-orbital blocks.

        restricted reference: alpha block == beta block, so spatial = 2*alpha-alpha.

        When ncore_spin > 0, the active-occ blocks are embedded into full
        (nocc_spatial, nocc_spatial) arrays with zeros in core rows/columns.
        """
        dgamma_oo_act, dgamma_ov_act, dgamma_vv_act = self.compute_gamma3_blocks()
        oo_sp = dgamma_oo_act[0::2, 0::2] * 2.0
        ov_sp = dgamma_ov_act[0::2, 0::2] * 2.0
        vv_sp = dgamma_vv_act[0::2, 0::2] * 2.0

        ncore_spatial = self.ncore // 2
        if ncore_spatial == 0:
            return oo_sp, ov_sp, vv_sp

        # Embed active-occ blocks into full spatial-MO occupied space
        nocc_spatial = self.nocc // 2
        full_oo = np.zeros((nocc_spatial, nocc_spatial))
        full_oo[ncore_spatial:, ncore_spatial:] = oo_sp
        full_ov = np.zeros((nocc_spatial, vv_sp.shape[0]))
        full_ov[ncore_spatial:, :] = ov_sp
        return full_oo, full_ov, vv_sp


class MP2DensityMatrixSolverUnrestricted:
    """Spin-case-resolved (alpha/beta) unrelaxed MP2 1-particle-density-matrix correction.

    Genuinely UHF-general: works for any (eps_a, eps_b, g_aaaa, g_bbbb,
    g_abab, nocc_a, nocc_b), RHF being the special case eps_a=eps_b with all
    three integral blocks built from the same restricted integrals (see
    get_antisymmetrized_spin_block_eri). Reproduces MP2DensityMatrixSolver's
    alpha spin-orbital sub-block exactly for a restricted reference

    Every oo/vv/ov formula below is the direct spin decomposition of
    MP2DensityMatrixSolver.compute_blocks's spin-orbital sum: the same-spin
    (aaaa/bbbb) piece carries over unchanged, and the opposite-spin sum
    contributes via g_abab/t2_1_abab with a factor of 2
    """

    def __init__(self, eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc_a, nocc_b, ncore_a=0, ncore_b=0):
        self.eps_a, self.eps_b = eps_a, eps_b
        self.g_aaaa, self.g_bbbb, self.g_abab = g_aaaa, g_bbbb, g_abab
        self.nocc_a, self.nocc_b = nocc_a, nocc_b
        self.ncore_a, self.ncore_b = ncore_a, ncore_b
        self.norb_a, self.norb_b = len(eps_a), len(eps_b)

    def _slices(self):
        oa = slice(self.ncore_a, self.nocc_a)
        va = slice(self.nocc_a, self.norb_a)
        ob = slice(self.ncore_b, self.nocc_b)
        vb = slice(self.nocc_b, self.norb_b)
        return oa, va, ob, vb

    def compute_t2_1(self):
        """
        t2_1_aaaa/bbbb/abab[a,b,i,j] = t_ij^{ab(1)} = <ab||ij> / (ei+ej-ea-eb), per spin sector.

        abab convention throughout this class: axis order (virt_alpha,
        virt_beta, occ_alpha, occ_beta) -- matches g_abab's own convention.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        den_aa = (ei_a[None, None, :, None] + ei_a[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_a[None, :, None, None])
        t2_1_aaaa = self.g_aaaa[va, va, oa, oa] / den_aa

        den_bb = (ei_b[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_b[:, None, None, None] - ea_b[None, :, None, None])
        t2_1_bbbb = self.g_bbbb[vb, vb, ob, ob] / den_bb

        den_ab = (ei_a[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_b[None, :, None, None])
        t2_1_abab = self.g_abab[va, vb, oa, ob] / den_ab
        return t2_1_aaaa, t2_1_bbbb, t2_1_abab

    def compute_t1_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        """
        t1_2_aa/bb[i,a] = t_i^{a(2)} -- ov block of Delta_gamma^(2) (== MP2's singles response).

        Derived symbolically: pq.add_commutator(1.0,['v'],['t2']) projected
        onto singles (e1), spin_labels={'i':'a','a':'a'} / {'i':'b','a':'b'}
        A single commutator [V,T2] is automatically fully connected, so no filtering
        of disconnected terms needed.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_aa_ovvv = self.g_aaaa[oa, va, va, va]
        g_aa_ooov = self.g_aaaa[oa, oa, oa, va]
        g_ab_ooov = self.g_abab[oa, ob, oa, vb]
        g_ab_vovv = self.g_abab[va, ob, va, vb]

        t1_2_aa = (-0.5 * np.einsum('jabc,bcij->ia', g_aa_ovvv, t2_1_aaaa, optimize=True)
                   - 0.5 * np.einsum('kjib,abkj->ia', g_ab_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jkib,abjk->ia', g_ab_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_1_aaaa, optimize=True)
                   + 0.5 * np.einsum('ajbc,bcij->ia', g_ab_vovv, t2_1_abab, optimize=True)
                   + 0.5 * np.einsum('ajcb,cbij->ia', g_ab_vovv, t2_1_abab, optimize=True))
        t1_2_aa = t1_2_aa / (ei_a[:, None] - ea_a[None, :])

        g_bb_ovvv = self.g_bbbb[ob, vb, vb, vb]
        g_ba_ooov = self.g_abab[oa, ob, va, ob]
        g_ba_ovvv = self.g_abab[oa, vb, va, vb]

        t1_2_bb = (-0.5 * np.einsum('kjbi,bakj->ia', g_ba_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jkbi,bajk->ia', g_ba_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_1_bbbb, optimize=True)
                   + 0.5 * np.einsum('jabc,bcji->ia', g_ba_ovvv, t2_1_abab, optimize=True)
                   + 0.5 * np.einsum('jacb,cbji->ia', g_ba_ovvv, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jabc,bcij->ia', g_bb_ovvv, t2_1_bbbb, optimize=True))
        t1_2_bb = t1_2_bb / (ei_b[:, None] - ea_b[None, :])
        return t1_2_aa, t1_2_bb

    def compute_blocks(self):
        """
        Spin-resolved (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) blocks of Delta_gamma^(2), active-space sized.
        """
        t2_1_aaaa, t2_1_bbbb, t2_1_abab = self.compute_t2_1()

        oo_a = -0.5 * (np.einsum('abik,abij->kj', t2_1_aaaa, t2_1_aaaa, optimize=True)
                       + 2.0 * np.einsum('abki,abji->kj', t2_1_abab, t2_1_abab, optimize=True))
        oo_b = -0.5 * (np.einsum('abik,abij->kj', t2_1_bbbb, t2_1_bbbb, optimize=True)
                       + 2.0 * np.einsum('abik,abij->kj', t2_1_abab, t2_1_abab, optimize=True))

        vv_a = 0.5 * (np.einsum('abij,acij->bc', t2_1_aaaa, t2_1_aaaa, optimize=True)
                      + 2.0 * np.einsum('baij,caij->bc', t2_1_abab, t2_1_abab, optimize=True))
        vv_b = 0.5 * (np.einsum('abij,acij->bc', t2_1_bbbb, t2_1_bbbb, optimize=True)
                      + 2.0 * np.einsum('abij,acij->bc', t2_1_abab, t2_1_abab, optimize=True))

        ov_a, ov_b = self.compute_t1_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        return oo_a, oo_b, ov_a, ov_b, vv_a, vv_b


class MP3DensityMatrixSolverUnrestricted(MP2DensityMatrixSolverUnrestricted):
    """
    Spin-case-resolved (alpha/beta) unrelaxed MP3 1-particle-density-matrix correction.
    Same concept as for MP2DensityMatrixSolverUnrestricted
    """

    def compute_t2_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        def antisym4(raw):
            return (raw - raw.transpose(0, 1, 3, 2)
                    - raw.transpose(1, 0, 2, 3) + raw.transpose(1, 0, 3, 2))

        t2_2_aaaa = 0.5 * np.einsum('lkij,ablk->abij', self.g_aaaa[oa, oa, oa, oa], t2_1_aaaa, optimize=True)
        t2_2_aaaa += antisym4(np.einsum('kacj,cbik->abij', self.g_aaaa[oa, va, va, oa], t2_1_aaaa, optimize=True))
        t2_2_aaaa += antisym4(-np.einsum('akjc,bcik->abij', self.g_abab[va, ob, oa, vb], t2_1_abab, optimize=True))
        t2_2_aaaa += 0.5 * np.einsum('abcd,cdij->abij', self.g_aaaa[va, va, va, va], t2_1_aaaa, optimize=True)
        den_aa = (ei_a[None, None, :, None] + ei_a[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_a[None, :, None, None])
        t2_2_aaaa = t2_2_aaaa / den_aa

        t2_2_bbbb = 0.5 * np.einsum('lkij,ablk->abij', self.g_bbbb[ob, ob, ob, ob], t2_1_bbbb, optimize=True)
        t2_2_bbbb += antisym4(-np.einsum('kacj,cbki->abij', self.g_abab[oa, vb, va, ob], t2_1_abab, optimize=True))
        t2_2_bbbb += antisym4(np.einsum('kacj,cbik->abij', self.g_bbbb[ob, vb, vb, ob], t2_1_bbbb, optimize=True))
        t2_2_bbbb += 0.5 * np.einsum('abcd,cdij->abij', self.g_bbbb[vb, vb, vb, vb], t2_1_bbbb, optimize=True)
        den_bb = (ei_b[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_b[:, None, None, None] - ea_b[None, :, None, None])
        t2_2_bbbb = t2_2_bbbb / den_bb

        t2_2_abab = 0.5 * np.einsum('lkij,ablk->abij', self.g_abab[oa, ob, oa, ob], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('klij,abkl->abij', self.g_abab[oa, ob, oa, ob], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('akcj,cbik->abij', self.g_abab[va, ob, va, ob], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('kbcj,caik->abij', self.g_abab[oa, vb, va, ob], t2_1_aaaa, optimize=True)
        t2_2_abab += np.einsum('kbcj,acik->abij', self.g_bbbb[ob, vb, vb, ob], t2_1_abab, optimize=True)
        t2_2_abab += np.einsum('kaci,cbkj->abij', self.g_aaaa[oa, va, va, oa], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('akic,cbjk->abij', self.g_abab[va, ob, oa, vb], t2_1_bbbb, optimize=True)
        t2_2_abab += -np.einsum('kbic,ackj->abij', self.g_abab[oa, vb, oa, vb], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('abcd,cdij->abij', self.g_abab[va, vb, va, vb], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('abdc,dcij->abij', self.g_abab[va, vb, va, vb], t2_1_abab, optimize=True)
        den_ab = (ei_a[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_b[None, :, None, None])
        t2_2_abab = t2_2_abab / den_ab
        return t2_2_aaaa, t2_2_bbbb, t2_2_abab

    def _is_restricted(self):
        """
        True when this is effectively an RHF calculation
        """
        return self.g_aaaa is self.g_bbbb

    def compute_t3_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        """
        triples amplitude of second-order wave function. Materialized 6-index intermediates,
        so only really reelevant for benchmarking, not useful for production runs.
        For production runs, we have the Laplace transform version of this routine that never
        materializes the tensor but contracts it directly with other intermediates to avoid
        6-index tensor entirely.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        def denom6(e_i, e_j, e_k, e_a, e_b, e_c):
            return (e_i[None, None, None, :, None, None] + e_j[None, None, None, None, :, None]
                    + e_k[None, None, None, None, None, :] - e_a[:, None, None, None, None, None]
                    - e_b[None, :, None, None, None, None] - e_c[None, None, :, None, None, None])

        def p_ij_ab(raw):
            return (raw - raw.transpose(0, 1, 2, 4, 3, 5)
                    - raw.transpose(1, 0, 2, 3, 4, 5) + raw.transpose(1, 0, 2, 4, 3, 5))

        def p_ab(raw):
            return raw - raw.transpose(1, 0, 2, 3, 4, 5)

        def p_ij(raw):
            return raw - raw.transpose(0, 1, 2, 4, 3, 5)

        def p_jk_bc(raw):
            return (raw - raw.transpose(0, 1, 2, 3, 5, 4)
                    - raw.transpose(0, 2, 1, 3, 4, 5) + raw.transpose(0, 2, 1, 3, 5, 4))

        def p_bc(raw):
            return raw - raw.transpose(0, 2, 1, 3, 4, 5)

        def p_jk(raw):
            return raw - raw.transpose(0, 1, 2, 3, 5, 4)

        # aaa
        t3_2_aaa = p_ij_ab(-np.einsum('lajk,bcil->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_ab(-np.einsum('laij,bckl->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += -np.einsum('lcij,abkl->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True)
        t3_2_aaa += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_bc(-np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_jk(-np.einsum('bcdk,daij->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += -np.einsum('bcdi,dajk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True)
        t3_2_aaa = t3_2_aaa / denom6(ei_a, ei_a, ei_a, ea_a, ea_a, ea_a)

        # RHF fast-path: on a restricted reference g_aaaa is g_bbbb and
        # t2_1_aaaa == t2_1_bbbb, so t3_2_bbb is identical to t3_2_aaa and
        # t3_2_abb[a,b,c,i,j,k] == t3_2_aab[b,c,a,j,k,i] (swap alpha<->beta
        # labels).  Skip building them explicitly -- just alias/transpose.
        if self._is_restricted():
            t3_2_bbb = t3_2_aaa  # identical arrays for RHF

            # aab (2 alpha virt/occ: a,b,i,j; 1 beta virt/occ: c,k)
            t3_2_aab = p_ij_ab(np.einsum('aljk,bcil->abcijk', self.g_abab[va, ob, oa, ob], t2_1_abab, optimize=True))
            t3_2_aab += p_ab(np.einsum('laij,bclk->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_abab, optimize=True))
            t3_2_aab += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_aaaa, optimize=True))
            t3_2_aab += np.einsum('acdk,dbij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
            t3_2_aab += np.einsum('abdj,dcik->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('acjd,bdik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
            t3_2_aab += np.einsum('acid,bdjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('bcdk,daij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
            t3_2_aab += np.einsum('bcjd,adik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('bcid,adjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab = t3_2_aab / denom6(ei_a, ei_a, ei_b, ea_a, ea_a, ea_b)

            # abb is a relabeling of aab under the full alpha<->beta spin flip
            # (valid since g_aaaa is g_bbbb and eps_a == eps_b here): the two
            # "pair" slots (a,b / i,j) of aab become abb's two beta slots
            # (b,c / j,k), and aab's single "unlike" slot (c / k) becomes
            # abb's single alpha slot (a / i) -- i.e.
            # t3_2_abb[a,b,c,i,j,k] = -t3_2_aab[b,c,a,k,j,i]  (axes: 2,0,1,5,4,3).
            # The extra minus sign comes from the antisymmetric-permutation
            # convention baked into t3_2_aab's own p_ij_ab/p_ab/... helpers,
            # which are not symmetric under this particular relabeling;
            t3_2_abb = -t3_2_aab.transpose(2, 0, 1, 5, 4, 3)

            return t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb

        # Full UHF path: compute all four sectors independently.
        # bbb (mirror of aaa with beta integrals/amplitudes)
        t3_2_bbb = p_ij_ab(-np.einsum('lajk,bcil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_ab(-np.einsum('laij,bckl->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += -np.einsum('lcij,abkl->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True)
        t3_2_bbb += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_bc(-np.einsum('abdi,dcjk->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_jk(-np.einsum('bcdk,daij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += -np.einsum('bcdi,dajk->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True)
        t3_2_bbb = t3_2_bbb / denom6(ei_b, ei_b, ei_b, ea_b, ea_b, ea_b)

        # aab (2 alpha virt/occ: a,b,i,j; 1 beta virt/occ: c,k)
        t3_2_aab = p_ij_ab(np.einsum('aljk,bcil->abcijk', self.g_abab[va, ob, oa, ob], t2_1_abab, optimize=True))
        t3_2_aab += p_ab(np.einsum('laij,bclk->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_abab, optimize=True))
        t3_2_aab += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_aaaa, optimize=True))
        t3_2_aab += np.einsum('acdk,dbij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
        t3_2_aab += np.einsum('abdj,dcik->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('acjd,bdik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
        t3_2_aab += np.einsum('acid,bdjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('bcdk,daij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
        t3_2_aab += np.einsum('bcjd,adik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('bcid,adjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab = t3_2_aab / denom6(ei_a, ei_a, ei_b, ea_a, ea_a, ea_b)

        # abb (1 alpha virt/occ: a,i; 2 beta virt/occ: b,c,j,k)
        t3_2_abb = np.einsum('lbjk,acil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('alik,bcjl->abcijk', self.g_abab[va, ob, oa, ob], t2_1_bbbb, optimize=True)
        t3_2_abb += np.einsum('lbik,aclj->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += np.einsum('alij,bckl->abcijk', self.g_abab[va, ob, oa, ob], t2_1_bbbb, optimize=True)
        t3_2_abb += -np.einsum('lbij,aclk->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('lcjk,abil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('lcik,ablj->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += np.einsum('lcij,ablk->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_abab[va, vb, va, ob], t2_1_abab, optimize=True))
        t3_2_abb += p_bc(np.einsum('abid,dcjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_bbbb, optimize=True))
        t3_2_abb += p_jk(np.einsum('bcdk,adij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_abab, optimize=True))
        t3_2_abb = t3_2_abb / denom6(ei_a, ei_b, ei_b, ea_a, ea_b, ea_b)

        return t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb

    def compute_t3_double_slice_aaa(self, k, j, t2_1_aaaa):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]

        g = self.g_aaaa

        ei_k = ei_a[k]
        ei_j = ei_a[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_a[None, :, None, None]
                    - ea_a[None, None, :, None])

        raw1_kj = np.einsum('la,bcil->abci', g[oa, va, j, k], t2_1_aaaa, optimize=True)
        raw1_alt_kj = np.einsum('lai,bcl->abci', g[oa, va, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T1_kj = -(raw1_kj - raw1_kj.transpose(1, 0, 2, 3)
                  - raw1_alt_kj + raw1_alt_kj.transpose(1, 0, 2, 3))

        raw2_kj = np.einsum('abd,dci->abci', g[va, va, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        raw2_alt_kj = np.einsum('abd,dci->abci', g[va, va, va, j], t2_1_aaaa[:, :, :, k], optimize=True)
        T2_kj = -(raw2_kj - raw2_kj.transpose(0, 2, 1, 3)
                  - raw2_alt_kj + raw2_alt_kj.transpose(0, 2, 1, 3))

        raw3_kj = np.einsum('lai,bcl->abci', g[oa, va, oa, j], t2_1_aaaa[:, :, k, :], optimize=True)
        T3_kj = -(raw3_kj - raw3_kj.transpose(1, 0, 2, 3))

        raw4_kj = np.einsum('lc,abil->abci', g[oa, va, j, k], t2_1_aaaa, optimize=True)
        raw4_alt_kj = np.einsum('lci,abl->abci', g[oa, va, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T4_kj = -(raw4_kj - raw4_alt_kj)

        raw5_kj = np.einsum('abdi,dc->abci', g[va, va, va, oa], t2_1_aaaa[:, :, j, k], optimize=True)
        T5_kj = -(raw5_kj - raw5_kj.transpose(0, 2, 1, 3))

        raw6_kj = np.einsum('bcd,dai->abci', g[va, va, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        raw6_alt_kj = np.einsum('bcd,dai->abci', g[va, va, va, j], t2_1_aaaa[:, :, :, k], optimize=True)
        T6_kj = -(raw6_kj - raw6_alt_kj)

        T7_kj = -np.einsum('lci,abl->abci', g[oa, va, oa, j], t2_1_aaaa[:, :, k, :], optimize=True)
        T8_kj = -np.einsum('bcdi,da->abci', g[va, va, va, oa], t2_1_aaaa[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_bbb(self, k, j, t2_1_bbbb):
        oa, va, ob, vb = self._slices()
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g = self.g_bbbb

        ei_k = ei_b[k]
        ei_j = ei_b[j]
        denom_kj = (ei_b[None, None, None, :] + ei_j + ei_k
                    - ea_b[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        raw1_kj = np.einsum('la,bcil->abci', g[ob, vb, j, k], t2_1_bbbb, optimize=True)
        raw1_alt_kj = np.einsum('lai,bcl->abci', g[ob, vb, ob, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T1_kj = -(raw1_kj - raw1_kj.transpose(1, 0, 2, 3)
                  - raw1_alt_kj + raw1_alt_kj.transpose(1, 0, 2, 3))

        raw2_kj = np.einsum('abd,dci->abci', g[vb, vb, vb, k], t2_1_bbbb[:, :, :, j], optimize=True)
        raw2_alt_kj = np.einsum('abd,dci->abci', g[vb, vb, vb, j], t2_1_bbbb[:, :, :, k], optimize=True)
        T2_kj = -(raw2_kj - raw2_kj.transpose(0, 2, 1, 3)
                  - raw2_alt_kj + raw2_alt_kj.transpose(0, 2, 1, 3))

        raw3_kj = np.einsum('lai,bcl->abci', g[ob, vb, ob, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T3_kj = -(raw3_kj - raw3_kj.transpose(1, 0, 2, 3))

        raw4_kj = np.einsum('lc,abil->abci', g[ob, vb, j, k], t2_1_bbbb, optimize=True)
        raw4_alt_kj = np.einsum('lci,abl->abci', g[ob, vb, ob, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T4_kj = -(raw4_kj - raw4_alt_kj)

        raw5_kj = np.einsum('abdi,dc->abci', g[vb, vb, vb, ob], t2_1_bbbb[:, :, j, k], optimize=True)
        T5_kj = -(raw5_kj - raw5_kj.transpose(0, 2, 1, 3))

        raw6_kj = np.einsum('bcd,dai->abci', g[vb, vb, vb, k], t2_1_bbbb[:, :, :, j], optimize=True)
        raw6_alt_kj = np.einsum('bcd,dai->abci', g[vb, vb, vb, j], t2_1_bbbb[:, :, :, k], optimize=True)
        T6_kj = -(raw6_kj - raw6_alt_kj)

        T7_kj = -np.einsum('lci,abl->abci', g[ob, vb, ob, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T8_kj = -np.einsum('bcdi,da->abci', g[vb, vb, vb, ob], t2_1_bbbb[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_aab(self, k, j, t2_1_aaaa, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_aaaa = self.g_aaaa
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_j = ei_a[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_a[None, :, None, None]
                    - ea_b[None, None, :, None])

        raw1_kj = np.einsum('al,bcil->abci', g_abab[va, ob, j, k], t2_1_abab, optimize=True)
        raw1_alt_kj = np.einsum('ali,bcl->abci', g_abab[va, ob, oa, k], t2_1_abab[:, :, j, :], optimize=True)
        T1_kj = raw1_kj - raw1_alt_kj - raw1_kj.transpose(1, 0, 2, 3) + raw1_alt_kj.transpose(1, 0, 2, 3)

        raw2_kj = np.einsum('lai,bcl->abci', g_aaaa[oa, va, oa, j], t2_1_abab[:, :, :, k], optimize=True)
        T2_kj = raw2_kj - raw2_kj.transpose(1, 0, 2, 3)

        raw3_kj = -np.einsum('lc,abil->abci', g_abab[oa, vb, j, k], t2_1_aaaa, optimize=True)
        raw3_alt_kj = -np.einsum('lci,abl->abci', g_abab[oa, vb, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T3_kj = raw3_kj - raw3_alt_kj

        T4_kj = np.einsum('acd,dbi->abci', g_abab[va, vb, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        T5_kj = np.einsum('abd,dci->abci', g_aaaa[va, va, va, j], t2_1_abab[:, :, :, k], optimize=True)
        T6_kj = -np.einsum('acd,bdi->abci', g_abab[va, vb, j, vb], t2_1_abab[:, :, :, k], optimize=True)
        T7_kj = -np.einsum('abdi,dc->abci', g_aaaa[va, va, va, oa], t2_1_abab[:, :, j, k], optimize=True)
        T8_kj = np.einsum('acid,bd->abci', g_abab[va, vb, oa, vb], t2_1_abab[:, :, j, k], optimize=True)
        T9_kj = -np.einsum('bcd,dai->abci', g_abab[va, vb, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        T10_kj = np.einsum('bcd,adi->abci', g_abab[va, vb, j, vb], t2_1_abab[:, :, :, k], optimize=True)
        T11_kj = -np.einsum('bcid,ad->abci', g_abab[va, vb, oa, vb], t2_1_abab[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj + T9_kj + T10_kj + T11_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_abb(self, k, j, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_bbbb = self.g_bbbb
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_j = ei_b[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        T1_kj = np.einsum('lb,acil->abci', g_bbbb[ob, vb, j, k], t2_1_abab, optimize=True)
        T2_kj = -np.einsum('ali,bcl->abci', g_abab[va, ob, oa, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T3_kj = np.einsum('lbi,acl->abci', g_abab[oa, vb, oa, k], t2_1_abab[:, :, :, j], optimize=True)
        T4_kj = np.einsum('ali,bcl->abci', g_abab[va, ob, oa, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T5_kj = -np.einsum('lbi,acl->abci', g_abab[oa, vb, oa, j], t2_1_abab[:, :, :, k], optimize=True)
        T6_kj = -np.einsum('lc,abil->abci', g_bbbb[ob, vb, j, k], t2_1_abab, optimize=True)
        T7_kj = -np.einsum('lci,abl->abci', g_abab[oa, vb, oa, k], t2_1_abab[:, :, :, j], optimize=True)
        T8_kj = np.einsum('lci,abl->abci', g_abab[oa, vb, oa, j], t2_1_abab[:, :, :, k], optimize=True)

        raw9_kj = -np.einsum('abd,dci->abci', g_abab[va, vb, va, k], t2_1_abab[:, :, :, j], optimize=True)
        raw9_alt_kj = -np.einsum('abd,dci->abci', g_abab[va, vb, va, j], t2_1_abab[:, :, :, k], optimize=True)
        T9_kj = raw9_kj - raw9_alt_kj - raw9_kj.transpose(0, 2, 1, 3) + raw9_alt_kj.transpose(0, 2, 1, 3)

        raw10_kj = np.einsum('abid,dc->abci', g_abab[va, vb, oa, vb], t2_1_bbbb[:, :, j, k], optimize=True)
        T10_kj = raw10_kj - raw10_kj.transpose(0, 2, 1, 3)

        raw11_kj = np.einsum('bcd,adi->abci', g_bbbb[vb, vb, vb, k], t2_1_abab[:, :, :, j], optimize=True)
        raw11_alt_kj = np.einsum('bcd,adi->abci', g_bbbb[vb, vb, vb, j], t2_1_abab[:, :, :, k], optimize=True)
        T11_kj = raw11_kj - raw11_alt_kj

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj + T9_kj + T10_kj + T11_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_abb_alpha(self, k, i, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_bbbb = self.g_bbbb
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_i = ei_a[i]
        denom_ki = (ei_i + ei_b[None, None, None, :] + ei_k
                    - ea_a[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        T1_ki = np.einsum('lbj,acl->abcj', g_bbbb[ob, vb, ob, k], t2_1_abab[:, :, i, :], optimize=True)
        T2_ki = -np.einsum('al,bcjl->abcj', g_abab[va, ob, i, k], t2_1_bbbb, optimize=True)
        T3_ki = np.einsum('lb,aclj->abcj', g_abab[oa, vb, i, k], t2_1_abab, optimize=True)
        T4_ki = np.einsum('alj,bcl->abcj', g_abab[va, ob, i, ob], t2_1_bbbb[:, :, k, :], optimize=True)
        T5_ki = -np.einsum('lbj,acl->abcj', g_abab[oa, vb, i, ob], t2_1_abab[:, :, :, k], optimize=True)
        T6_ki = -np.einsum('lcj,abl->abcj', g_bbbb[ob, vb, ob, k], t2_1_abab[:, :, i, :], optimize=True)
        T7_ki = -np.einsum('lc,ablj->abcj', g_abab[oa, vb, i, k], t2_1_abab, optimize=True)
        T8_ki = np.einsum('lcj,abl->abcj', g_abab[oa, vb, i, ob], t2_1_abab[:, :, :, k], optimize=True)

        raw9_ki = -np.einsum('abd,dcj->abcj', g_abab[va, vb, va, k], t2_1_abab[:, :, i, :], optimize=True)
        raw9_alt_ki = -np.einsum('abdj,dc->abcj', g_abab[va, vb, va, ob], t2_1_abab[:, :, i, k], optimize=True)
        T9_ki = raw9_ki - raw9_alt_ki - raw9_ki.transpose(0, 2, 1, 3) + raw9_alt_ki.transpose(0, 2, 1, 3)

        raw10_ki = np.einsum('abd,dcj->abcj', g_abab[va, vb, i, vb], t2_1_bbbb[:, :, :, k], optimize=True)
        T10_ki = raw10_ki - raw10_ki.transpose(0, 2, 1, 3)

        raw11_ki = np.einsum('bcd,adj->abcj', g_bbbb[vb, vb, vb, k], t2_1_abab[:, :, i, :], optimize=True)
        raw11_alt_ki = np.einsum('bcdj,ad->abcj', g_bbbb[vb, vb, vb, ob], t2_1_abab[:, :, i, k], optimize=True)
        T11_ki = raw11_ki - raw11_alt_ki

        numerator_ki = T1_ki + T2_ki + T3_ki + T4_ki + T5_ki + T6_ki + T7_ki + T8_ki + T9_ki + T10_ki + T11_ki
        return numerator_ki / denom_ki

    def compute_t1_3(self, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_bbbb, t2_2_abab,
                      t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        t1_3_aa = np.einsum('jabi,jb->ia', self.g_aaaa[oa, va, va, oa], t1_2_aa, optimize=True)
        t1_3_aa += np.einsum('ajib,jb->ia', self.g_abab[va, ob, oa, vb], t1_2_bb, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_2_aaaa, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjib,abkj->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jkib,abjk->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jabc,bcij->ia', self.g_aaaa[oa, va, va, va], t2_2_aaaa, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajbc,bcij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajcb,cbij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)

        t1_3_aa += 0.25 * np.einsum('kjbc,bcaikj->ia', self.g_aaaa[oa, oa, va, va], t3_2_aaa, optimize=True)
        t1_3_aa += -0.25 * np.einsum('kjbc,bacikj->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += -0.25 * np.einsum('jkbc,bacijk->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += 0.25 * np.einsum('kjcb,acbikj->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += 0.25 * np.einsum('jkcb,acbijk->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += -0.25 * np.einsum('kjbc,acbikj->ia', self.g_bbbb[ob, ob, vb, vb], t3_2_abb, optimize=True)
        t1_3_aa = t1_3_aa / (ei_a[:, None] - ea_a[None, :])

        t1_3_bb = np.einsum('jabi,jb->ia', self.g_abab[oa, vb, va, ob], t1_2_aa, optimize=True)
        t1_3_bb += np.einsum('jabi,jb->ia', self.g_bbbb[ob, vb, vb, ob], t1_2_bb, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jkbi,bajk->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_2_bbbb, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jabc,bcji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jacb,cbji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jabc,bcij->ia', self.g_bbbb[ob, vb, vb, vb], t2_2_bbbb, optimize=True)

        t1_3_bb += -0.25 * np.einsum('kjbc,bcajki->ia', self.g_aaaa[oa, oa, va, va], t3_2_aab, optimize=True)
        t1_3_bb += -0.25 * np.einsum('kjbc,bcakij->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('jkbc,bcajki->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += -0.25 * np.einsum('kjcb,cbakij->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('jkcb,cbajki->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('kjbc,bcaikj->ia', self.g_bbbb[ob, ob, vb, vb], t3_2_bbb, optimize=True)
        t1_3_bb = t1_3_bb / (ei_b[:, None] - ea_b[None, :])
        return t1_3_aa, t1_3_bb

    def _laplace_aaa_contribution(self, t2_1_aaaa, ntau):
        """
        Laplace-quadrature replacement for compute_gamma3_blocks's "Loop 1"
        (pure-alpha aaa sector)
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        g = self.g_aaaa
        no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
        g_aaaa_oooo = g[oa, oa, va, va]
        g_vvvo = g[va, va, va, oa]
        g_ovoo = g[oa, va, oa, oa]

        # e_min/e_max only need to bracket the true (positive) three-particle
        # gap ea+eb+ec-ei-ej-ek within the right order of magnitude for the
        # minimax table lookup (it's a ratio-based lookup, not a tight bound)
        gap_min = max(ea_a.min() - ei_a.max(), 1e-3)
        gap_max = ea_a.max() - ei_a.min()
        e_min, e_max = 3.0 * gap_min, 3.0 * gap_max
        tau, sigma = minimax_time_grid(ntau, e_min, e_max)

        # (k,j,b,c) -- feeds t1_3_aa
        outer1 = g_aaaa_oooo

        # (b,c,j,k) -- feeds term_c_a
        outer2 = t2_1_aaaa
        t1_3_aa_contrib = np.zeros((no_a, nv_a))
        term_c_a_contrib = np.zeros((no_a, nv_a))

        for tk in range(ntau):
            t = tau[tk]

            # 1/D = -sum_tau sigma_tau * exp(D*tau)
            w = -sigma[tk]

            Oe = np.exp(ei_a * t)
            Ve = np.exp(-ea_a * t)
            N_outer1 = np.zeros((no_a, nv_a))
            N_outer2 = np.zeros((no_a, nv_a))

            # Term 1: p_ij_ab(-einsum('lajk,bcil->abcijk', g_ovoo, t2))
            G1d = g_ovoo * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
            T1d = t2_1_aaaa * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, :, None]
            p1 = -np.einsum('kjbc,lajk,bcil->ia', outer1, G1d, T1d, optimize=True)
            p2 = -np.einsum('kjbc,laik,bcjl->ia', outer1, G1d, T1d, optimize=True)
            p3 = -np.einsum('kjbc,lbjk,acil->ia', outer1, G1d, T1d, optimize=True)
            p4 = -np.einsum('kjbc,lbik,acjl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += p1 - p2 - p3 + p4
            q1 = -np.einsum('bcjk,lajk,bcil->ia', outer2, G1d, T1d, optimize=True)
            q2 = -np.einsum('bcjk,laik,bcjl->ia', outer2, G1d, T1d, optimize=True)
            q3 = -np.einsum('bcjk,lbjk,acil->ia', outer2, G1d, T1d, optimize=True)
            q4 = -np.einsum('bcjk,lbik,acjl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += q1 - q2 - q3 + q4

            # Term 2: p_ab(-einsum('laij,bckl->abcijk', g_ovoo, t2)) -- same dressed legs as term 1
            r1 = -np.einsum('kjbc,laij,bckl->ia', outer1, G1d, T1d, optimize=True)
            r2 = -np.einsum('kjbc,lbij,ackl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += r1 - r2
            s1 = -np.einsum('bcjk,laij,bckl->ia', outer2, G1d, T1d, optimize=True)
            s2 = -np.einsum('bcjk,lbij,ackl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += s1 - s2

            # Term 3: p_ij(-einsum('lcjk,abil->abcijk', g_ovoo, t2)) -- same dressed legs
            u1 = -np.einsum('kjbc,lcjk,abil->ia', outer1, G1d, T1d, optimize=True)
            u2 = -np.einsum('kjbc,lcik,abjl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += u1 - u2
            v1 = -np.einsum('bcjk,lcjk,abil->ia', outer2, G1d, T1d, optimize=True)
            v2 = -np.einsum('bcjk,lcik,abjl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += v1 - v2

            # Term 4 (bare): -einsum('lcij,abkl->abcijk', g_ovoo, t2) -- same dressed legs
            N_outer1 += -np.einsum('kjbc,lcij,abkl->ia', outer1, G1d, T1d, optimize=True)
            N_outer2 += -np.einsum('bcjk,lcij,abkl->ia', outer2, G1d, T1d, optimize=True)

            # Term 5: p_jk_bc(-einsum('abdk,dcij->abcijk', g_vvvo, t2)) -- antisymmetry shortcut x4
            G5d = g_vvvo * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, None, :]
            T5d = t2_1_aaaa * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
            N_outer1 += 4.0 * (-np.einsum('kjbc,abdk,dcij->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 4.0 * (-np.einsum('bcjk,abdk,dcij->ia', outer2, G5d, T5d, optimize=True))

            # Term 6: p_bc(-einsum('abdi,dcjk->abcijk', g_vvvo, t2)) -- shortcut x2, same dressed legs
            N_outer1 += 2.0 * (-np.einsum('kjbc,abdi,dcjk->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 2.0 * (-np.einsum('bcjk,abdi,dcjk->ia', outer2, G5d, T5d, optimize=True))

            # Term 7: p_jk(-einsum('bcdk,daij->abcijk', g_vvvo, t2)) -- shortcut x2, same dressed legs
            N_outer1 += 2.0 * (-np.einsum('kjbc,bcdk,daij->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 2.0 * (-np.einsum('bcjk,bcdk,daij->ia', outer2, G5d, T5d, optimize=True))

            # Term 8 (bare): -einsum('bcdi,dajk->abcijk', g_vvvo, t2) -- same dressed legs
            N_outer1 += -np.einsum('kjbc,bcdi,dajk->ia', outer1, G5d, T5d, optimize=True)
            N_outer2 += -np.einsum('bcjk,bcdi,dajk->ia', outer2, G5d, T5d, optimize=True)

            t1_3_aa_contrib += w * (-0.25) * N_outer1
            term_c_a_contrib += w * (0.25) * N_outer2

        return t1_3_aa_contrib, term_c_a_contrib

    def compute_gamma3_blocks(self, laplace_ntau=6):
        """Spin-resolved (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) blocks of Delta_gamma^(3), active-space sized.

        Memory-efficient: the t3_2 tensor (V^3 x O^3) is never fully built.
        We compute slices on the fly to keep memory complexity to O(V^3 O^2).
        """
        if self.ncore_a or self.ncore_b:
            wa, wb = slice(self.ncore_a, None), slice(self.ncore_b, None)
            g_aaaa_w = self.g_aaaa[wa, wa, wa, wa]
            # preserve the g_aaaa-is-g_bbbb identity _is_restricted() keys on
            g_bbbb_w = g_aaaa_w if self.g_aaaa is self.g_bbbb else self.g_bbbb[wb, wb, wb, wb]
            sub = MP3DensityMatrixSolverUnrestricted(
                self.eps_a[wa], self.eps_b[wb], g_aaaa_w, g_bbbb_w,
                self.g_abab[wa, wb, wa, wb],
                self.nocc_a - self.ncore_a, self.nocc_b - self.ncore_b)
            return sub.compute_gamma3_blocks(laplace_ntau=laplace_ntau)

        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]
        no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
        no_b, nv_b = ei_b.shape[0], ea_b.shape[0]

        t2_1_aaaa, t2_1_bbbb, t2_1_abab = self.compute_t2_1()
        t1_2_aa, t1_2_bb = self.compute_t1_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        t2_2_aaaa, t2_2_bbbb, t2_2_abab = self.compute_t2_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)

        # 1. oo/vv blocks (t2_1 x t2_2 only -- no t3_2 involvement)
        vv_a_raw = (np.einsum('abij,acij->bc', t2_1_aaaa, t2_2_aaaa, optimize=True)
                    + 2.0 * np.einsum('baij,caij->bc', t2_1_abab, t2_2_abab, optimize=True))
        vv_a = 0.5 * (vv_a_raw + vv_a_raw.T)
        vv_b_raw = (np.einsum('abij,acij->bc', t2_1_bbbb, t2_2_bbbb, optimize=True)
                    + 2.0 * np.einsum('abij,acij->bc', t2_1_abab, t2_2_abab, optimize=True))
        vv_b = 0.5 * (vv_b_raw + vv_b_raw.T)

        oo_a_raw = (np.einsum('abik,abij->kj', t2_1_aaaa, t2_2_aaaa, optimize=True)
                    + 2.0 * np.einsum('abki,abji->kj', t2_1_abab, t2_2_abab, optimize=True))
        oo_a = -0.5 * (oo_a_raw + oo_a_raw.T)
        oo_b_raw = (np.einsum('abik,abij->kj', t2_1_bbbb, t2_2_bbbb, optimize=True)
                    + 2.0 * np.einsum('abik,abij->kj', t2_1_abab, t2_2_abab, optimize=True))
        oo_b = -0.5 * (oo_b_raw + oo_b_raw.T)

        # 2. t1_3 terms that do not depend on t3_2
        t1_3_aa = np.einsum('jabi,jb->ia', self.g_aaaa[oa, va, va, oa], t1_2_aa, optimize=True)
        t1_3_aa += np.einsum('ajib,jb->ia', self.g_abab[va, ob, oa, vb], t1_2_bb, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_2_aaaa, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjib,abkj->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jkib,abjk->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jabc,bcij->ia', self.g_aaaa[oa, va, va, va], t2_2_aaaa, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajbc,bcij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajcb,cbij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)

        t1_3_bb = np.einsum('jabi,jb->ia', self.g_abab[oa, vb, va, ob], t1_2_aa, optimize=True)
        t1_3_bb += np.einsum('jabi,jb->ia', self.g_bbbb[ob, vb, vb, ob], t1_2_bb, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jkbi,bajk->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_2_bbbb, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jabc,bcji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jacb,cbji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jabc,bcij->ia', self.g_bbbb[ob, vb, vb, vb], t2_2_bbbb, optimize=True)

        # 3. Contraction accumulators for t3_2-dependent terms (Double-loop batched)
        g_aaaa_oooo = self.g_aaaa[oa, oa, va, va]
        g_abab_oooo = self.g_abab[oa, ob, va, vb]
        g_bbbb_oooo = self.g_bbbb[ob, ob, vb, vb]

        term_c_a = np.zeros((no_a, nv_a))
        term_c_b = np.zeros((no_b, nv_b))

        restricted = self._is_restricted()

        # Loop 1: k_alpha, j_alpha (for aaa terms). laplace_ntau (default 6)
        # replaces this O(no_a^2) double loop with the Laplace quadrature from
        # _laplace_aaa_contribution; when restricted, the bbb-diagonal piece
        # of Loop 2a below is *also* skipped and filled in from this same
        # result (bbb is just a relabeled copy of aaa in that case).
        if laplace_ntau is not None:
            t1_3_aa_lap, term_c_a_lap = self._laplace_aaa_contribution(t2_1_aaaa, laplace_ntau)
            t1_3_aa += t1_3_aa_lap
            term_c_a += term_c_a_lap
            if restricted:
                t1_3_bb += t1_3_aa_lap
                term_c_b += term_c_a_lap
        else:
            for k in range(no_a):
                for j in range(no_a):
                    t3_aaa_kj = self.compute_t3_double_slice_aaa(k, j, t2_1_aaaa)
                    t1_3_aa += -0.25 * np.einsum('bc,abci->ia', g_aaaa_oooo[k, j], t3_aaa_kj, optimize=True)
                    term_c_a += 0.25 * np.einsum('bc,abci->ia', t2_1_aaaa[:, :, j, k], t3_aaa_kj, optimize=True)

        # Loop 2: k_beta (for bbb, aab, abb terms)
        for k in range(no_b):
            # 2a: j_beta (for bbb, abb terms). The bbb-diagonal accumulation
            # (t3_bbb_kj and its two accumulator lines) is skipped here when
            # restricted and laplace_ntau is set, since it was already added
            # above from the aaa Laplace result -- t3_abb_kj (cross-spin) is
            # not yet Laplace-accelerated and is still computed every time.
            skip_bbb_diagonal = restricted and laplace_ntau is not None
            for j in range(no_b):
                t3_abb_kj = self.compute_t3_double_slice_abb(k, j, t2_1_bbbb, t2_1_abab)
                t1_3_bb[j, :] += -0.25 * np.einsum('lbc,bcal->a', g_abab_oooo[:, k, :, :], t3_abb_kj, optimize=True)
                t1_3_bb[j, :] += -0.25 * np.einsum('lcb,cbal->a', g_abab_oooo[:, k, :, :], t3_abb_kj, optimize=True)
                term_c_a += 0.25 * np.einsum('bc,abci->ia', t2_1_bbbb[:, :, j, k], t3_abb_kj, optimize=True)

                if not skip_bbb_diagonal:
                    if restricted:
                        t3_bbb_kj = self.compute_t3_double_slice_aaa(k, j, t2_1_aaaa)
                    else:
                        t3_bbb_kj = self.compute_t3_double_slice_bbb(k, j, t2_1_bbbb)
                    t1_3_bb += -0.25 * np.einsum('bc,abci->ia', g_bbbb_oooo[k, j], t3_bbb_kj, optimize=True)
                    term_c_b += 0.25 * np.einsum('bc,abci->ia', t2_1_bbbb[:, :, j, k], t3_bbb_kj, optimize=True)

            # 2b: j_alpha (for aab, abb terms)
            for j in range(no_a):
                t3_aab_kj = self.compute_t3_double_slice_aab(k, j, t2_1_aaaa, t2_1_abab)
                t3_abb_ki = self.compute_t3_double_slice_abb_alpha(k, j, t2_1_bbbb, t2_1_abab)

                t1_3_aa += -0.25 * np.einsum('bc,baci->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += -0.25 * np.einsum('bc,baci->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += 0.25 * np.einsum('cb,acbi->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += 0.25 * np.einsum('cb,acbi->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa[j, :] += 0.25 * np.einsum('lbc,acbl->a', g_bbbb_oooo[k], t3_abb_ki, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lbc,bcal->a', g_aaaa_oooo[:, j, :, :], t3_aab_kj, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lbc,bcal->a', g_abab_oooo[j], t3_abb_ki, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lcb,cbal->a', g_abab_oooo[j], t3_abb_ki, optimize=True)

                term_c_a += np.einsum('bc,abci->ia', t2_1_abab[:, :, j, k], t3_aab_kj, optimize=True)
                term_c_b[k, :] += 0.25 * np.einsum('bcl,bcal->a', t2_1_aaaa[:, :, :, j], t3_aab_kj, optimize=True)
                term_c_b += np.einsum('bc,baci->ia', t2_1_abab[:, :, j, k], t3_abb_ki, optimize=True)

        t1_3_aa = t1_3_aa / (ei_a[:, None] - ea_a[None, :])
        t1_3_bb = t1_3_bb / (ei_b[:, None] - ea_b[None, :])

        # 4. Final ov construction
        term_b_a = (np.einsum('jc,acij->ia', t1_2_aa, t2_1_aaaa, optimize=True)
                    + np.einsum('jc,acij->ia', t1_2_bb, t2_1_abab, optimize=True))
        ov_a = t1_3_aa + term_b_a + term_c_a

        term_b_b = (np.einsum('jc,acij->ia', t1_2_bb, t2_1_bbbb, optimize=True)
                    + np.einsum('jc,acij->ia', t1_2_aa, t2_1_abab.transpose(1, 0, 3, 2), optimize=True))
        ov_b = t1_3_bb + term_b_b + term_c_b

        return oo_a, oo_b, ov_a, ov_b, vv_a, vv_b


def compute_mp3_density_matrix(mf, mol=None, nocc=None):
    """
    Build the spin-orbital blocks (oo, ov, vv) of the unrelaxed MP3
    1-particle density-matrix correction, restricted spin only.

    Spin-orbital basis is interleaved alpha/beta (even=alpha, odd=beta),
    matching compute_mp2_density_matrix; each block is to be added to the
    corresponding block of the reference spin-orbital density.
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("compute_mp3_density_matrix is currently restricted-spin only.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    solver = MP3DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
    return solver.compute_gamma3_blocks()


def _build_dgamma_block(nmo, nocc, oo, ov, vv):
    dgamma = np.zeros((nmo, nmo))
    dgamma[:nocc, :nocc] = oo
    dgamma[nocc:, nocc:] = vv
    dgamma[:nocc, nocc:] = ov
    dgamma[nocc:, :nocc] = ov.T
    return dgamma


def compute_mp3_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self'):
    """
    Build the total AO 1-particle density matrix

    gamma = gamma_mf + Delta_gamma^(2) + Delta_gamma^(3).

    relax: CPHF/CPKS-relaxed ov block (default) vs the direct second+third-order
    singles expression. relax_kernel: 'self' (mf's own Hessian) or 'tdhf' (bare
    Hartree + exact-exchange Hessian) -- RHF only; UHF does not support relax=True
    yet (no UHF CPHF/Z-vector solve implemented here).

    relax=True is NOT a fully relaxed density: the orbitals are relaxed by the
    Schirmer approach, the amplitudes never are. It therefore does NOT satisfy
    the finite-field/Hellmann-Feynman dipole sum rule (mu_z = -Tr[gamma @ z_AO]
    should equal -dE/dF_z), and what is left over is the size of the missing
    amplitude response: 4.3e-03 on HF/6-31g, against 2.7e-06 for MP2 relaxed
    through the same CPHF solve. Pinned in tests/test_mp3_finite_field.py.
    """

    mol = mol if mol is not None else mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        if relax:
            raise NotImplementedError(
                "compute_mp3_density_matrix_ao: relax=True is not yet implemented for UHF "
                "(needs a UHF CPHF/Z-vector solve) -- use relax=False.")
        nocc_a, nocc_b = mf.nelec
        eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)

        driver = MPnDensityDriverUnrestricted(np.diag(eps_a), np.diag(eps_b),
                                              g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
        (oo2_a, oo2_b, ov2_a, ov2_b, vv2_a, vv2_b), (oo3_a, oo3_b, ov3_a, ov3_b, vv3_a, vv3_b) = \
            driver.compute_delta_gamma23()

        mo_a, mo_b = mf.mo_coeff
        dgamma_a = _build_dgamma_block(mo_a.shape[1], nocc_a, oo2_a + oo3_a, ov2_a + ov3_a, vv2_a + vv3_a)
        dgamma_b = _build_dgamma_block(mo_b.shape[1], nocc_b, oo2_b + oo3_b, ov2_b + ov3_b, vv2_b + vv3_b)
        dm_a, dm_b = mf.make_rdm1()
        return dm_a + mo_a @ dgamma_a @ mo_a.T, dm_b + mo_b @ dgamma_b @ mo_b.T

    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)

    driver = MPnDensityDriverRestricted(np.diag(eps), g_aaaa, g_abab, g_bbbb, nocc)
    (oo2_a, ov2_a, vv2_a), (oo3_a, ov3_a, vv3_a) = driver.compute_delta_gamma23(laplace_ntau=6)
    dgamma_oo = 2.0 * (oo2_a + oo3_a)
    dgamma_ov = 2.0 * (ov2_a + ov3_a)
    dgamma_vv = 2.0 * (vv2_a + vv3_a)

    mo = mf.mo_coeff
    if not relax:
        dgamma = _build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    else:
        if relax_kernel == 'tdhf':
            mf_hess = scf.RHF(mol)
            mf_hess.mo_coeff = mo
            mf_hess.mo_energy = mf.mo_energy
            mf_hess.mo_occ = mf.mo_occ
        else:
            mf_hess = mf
        dgamma = solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao


def compute_mp2_density_matrix(mf, mol=None, nocc=None):
    """
    Build the spin-orbital blocks (oo, ov, vv) of the unrelaxed
    MP2 1-particle density-matrix correction, restricted spin only.

    Spin-orbital basis is interleaved alpha/beta (even=alpha, odd=beta);
    each block is to be added to the corresponding block of the reference
    spin-orbital density (2*delta_ij for occ-occ, 0 elsewhere).
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("compute_mp2_density_matrix is currently restricted-spin only.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    solver = MP2DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
    return solver.compute_blocks()


def compute_mp2_density_matrix_ao(mf, mol=None, nocc=None, relax=True, relax_kernel='self'):
    """
    Build the total AO 1-particle density matrix gamma = gamma_mf + Delta_gamma^MP2.
    """
    mol = mol if mol is not None else mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        if relax:
            raise NotImplementedError(
                "compute_mp2_density_matrix_ao: relax=True is not yet implemented for UHF "
                "(needs a UHF CPHF/Z-vector solve) -- use relax=False.")
        nocc_a, nocc_b = mf.nelec
        eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)

        driver = MPnDensityDriverUnrestricted(np.diag(eps_a), np.diag(eps_b),
                                              g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
        oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = driver.compute_delta_gamma2()

        mo_a, mo_b = mf.mo_coeff
        dgamma_a = _build_dgamma_block(mo_a.shape[1], nocc_a, oo_a, ov_a, vv_a)
        dgamma_b = _build_dgamma_block(mo_b.shape[1], nocc_b, oo_b, ov_b, vv_b)
        dm_a, dm_b = mf.make_rdm1()
        return dm_a + mo_a @ dgamma_a @ mo_a.T, dm_b + mo_b @ dgamma_b @ mo_b.T

    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)

    driver = MPnDensityDriverRestricted(np.diag(eps), g_aaaa, g_abab, g_bbbb, nocc)
    oo_a, ov_a, vv_a = driver.compute_delta_gamma2()
    dgamma_oo, dgamma_ov, dgamma_vv = 2.0 * oo_a, 2.0 * ov_a, 2.0 * vv_a

    mo = mf.mo_coeff
    if not relax:
        dgamma = _build_dgamma_block(nmo, nocc, dgamma_oo, dgamma_ov, dgamma_vv)
    else:
        if relax_kernel == 'tdhf':
            mf_hess = scf.RHF(mol)
            mf_hess.mo_coeff = mo
            mf_hess.mo_energy = mf.mo_energy
            mf_hess.mo_occ = mf.mo_occ
        else:
            mf_hess = mf
        dgamma = solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao
