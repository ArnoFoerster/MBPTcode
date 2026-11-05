import numpy as np
from pyscf import scf

from src.SingleReference.common.mean_field_operators import mean_field_static_operator
from src.SingleReference.LinearResponse.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.orbital_spaces import get_occ_virt_indices
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_density_fitting_coefficients,
    get_two_electron_integrals_chemist,
)
from src.Base.numerics.constants import (
    DEFAULT_BROADENING_ETA,
    DEFAULT_BLOCK_SIZE,
    GW_DENSITY_SPIN_SUM,
)
from src.SingleReference.DensityMatrix.density_matrix import (
    solve_cphf_relaxation,
    relaxation_numerics,
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

    def solve_relaxation(self, mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics):
        """Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov block; return the full relaxed MO-basis Delta_gamma.

        Pass a KS mf for the gKS Hessian, or a bare RHF object sharing mf's
        orbitals/energies for the TDHF Hessian; numerics as in
        solve_cphf_relaxation.
        """
        return solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics)


def compute_gw_density_matrix(mf, mol=None, nocc=None, polarizability='RPA', df=True,
                               relax=True, relax_kernel='self', eta=DEFAULT_BROADENING_ETA,
                               numerics=None):
    """Build the total AO 1-particle density matrix gamma = gamma_mf + Delta_gamma^GW, restricted spin only.

    polarizability: 'RPA'/'TDHF'/'BSE' screening channel. df: use density
    fitting (default) vs full 4-center ERIs. relax: CPHF/CPKS-relaxed ov
    block (default) vs simple perturbative one. relax_kernel: 'self' (mf's
    own Hessian) or 'tdhf' (bare Hartree + exact-exchange Hessian).
    numerics: the run's `Numerics`, the Z-vector's (solve_cphf_relaxation);
    required with relax=True.
    """
    numerics = relaxation_numerics(relax, numerics, 'compute_gw_density_matrix')
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
        dgamma = solver.solve_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv,
                                         numerics)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao
