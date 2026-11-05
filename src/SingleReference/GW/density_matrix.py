import numpy as np
from pyscf import scf

from src.SingleReference.common.mean_field_operators import mean_field_static_operator
from src.SingleReference.LinearResponse.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.orbital_spaces import get_occ_virt_indices, particle_hole_window
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid)
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_density_fitting_coefficients,
    get_two_electron_integrals_chemist,
)
from src.Base.numerics.constants import (
    DEFAULT_BROADENING_ETA,
    DEFAULT_BLOCK_SIZE,
)
from src.SingleReference.DensityMatrix.density_matrix import (
    hf_kernel_mean_field,
    solve_cphf_relaxation,
    relaxation_numerics,
)
from src.SingleReference.GW.imaginary_axis import (
    correlation_self_energy_matrix_imaginary_axis)
from src.SingleReference.GW.space_time import correlation_self_energy_matrix_space_time
from src.Base.environment_seam import NoEnvironment, environment_of
from src.Base.numerics.numerics import required_numerics
from src.Base.utils.grids.matsubara import beta_from_mf

#: The restricted spin sum: the spatial density is twice each channel's.
SPIN_SUM = 2.0

MODES = ('casida', 'imagfrequency', 'space-time')
RELAXATIONS = ('none', 'dyson', 'zvector')


def _choice(value, choices, keyword):
    """`value` normalized as calc_qp_energy's mode (lower case, '_' -> '-',
    'imag-frequency' -> 'imagfrequency'), refused unless one of `choices`."""
    name = str(value).lower().replace('_', '-')
    name = {'imag-frequency': 'imagfrequency'}.get(name, name)
    if name not in choices:
        raise ValueError(f"Unknown {keyword}={value!r}; choose one of "
                         + ', '.join(repr(c) for c in choices) + '.')
    return name


def static_ov_term(mf, mol, eps, nocc):
    """Per-channel static gKS ov term <i|Sigma_x - v_xc|a> / (e_i - e_a),
    (nocc, nvir), in closed form from `mean_field_static_operator` (the
    per-spin -K/2 - v_xc at mf's density); no spin factor. Zero on HF."""
    occ, virt = get_occ_virt_indices(eps, nocc)
    mo = mf.mo_coeff
    sx_ia = (mo.T @ mean_field_static_operator(mf, mol) @ mo)[np.ix_(occ, virt)]
    return sx_ia / (eps[occ][:, None] - eps[virt][None, :])


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

    def compute_unrelaxed_blocks(self, nocc, mf, mol, polarizability='RPA', omega=None, X=None, Y=None):
        """
        Per-channel MO blocks (dgamma_oo, dgamma_ov, dgamma_vv) of the unrelaxed GW density
        correction; the spatial density is SPIN_SUM times these.

        The ov block includes the static term (`static_ov_term`), so on HF it is purely dynamic.
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

        # get_chi_a builds the restricted w without its sqrt(2); the restricted
        # self-energy restores it with a prefactor 2 (SelfEnergySolver._spin,
        # GW/self_energy.py), and so do the w-quadratic terms here. The static
        # term below is already per channel and takes no such factor.
        w_norm = 2.0
        dgamma_oo = -w_norm * np.einsum('sic,sjc->ij', t_ic, t_ic)
        dgamma_vv = w_norm * np.einsum('sak,sbk->ab', t_ak, t_ak)

        term1 = np.einsum('sik,sbk->ib', w_ik, t_ak)
        term2 = np.einsum('sic,sbc->ib', t_ic, w_bc)
        dgamma_ov = (w_norm * (term1 + term2) / (ei[:, None] - ea[None, :])
                     + static_ov_term(mf, mol, eps, nocc))
        return dgamma_oo, dgamma_ov, dgamma_vv

    def solve_relaxation(self, mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics):
        """Solve the CPHF/CPKS Z-vector equation A.X=Y for the ov block; return the full relaxed MO-basis Delta_gamma.

        Pass mf for its own (gKS) Hessian, or hf_kernel_mean_field(mf) for the
        HF-kernel Hessian on mf's orbitals and J/K backend; numerics as in
        solve_cphf_relaxation.
        """
        return solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics)


def compute_gw_density_matrix(mf, mol=None, nocc=None, polarizability='RPA', df=True,
                               mode='casida', relaxation='zvector', eta=DEFAULT_BROADENING_ETA,
                               numerics=None, auxbasis=None, screen_r_cut=None):
    """Build the total AO 1-particle density matrix gamma = gamma_mf + Delta_gamma^GW, restricted spin only.

    polarizability: 'RPA'/'TDHF'/'BSE' screening channel. df: use density
    fitting (default) vs full 4-center ERIs.
    mode: how the unrelaxed Delta_gamma^GW is obtained; 'casida' (the
    analytic pole sum) or 'imagfrequency' (N^4: the full Sigma_c,pq(mu + i w)
    of the imaginary-axis QP route on the outer grid `numerics.gwdensity`,
    `imagfrequency_unrelaxed_blocks`) or 'space-time' (N^3: the same on the
    space-time QP route's Sigma, `space_time_unrelaxed_blocks`, on the ISDF
    grid numerics.isdf names).
    The numerical modes take RPA screening on a density-fitted three-index
    object (df=True) of a gas-phase, T = 0 mean field only, and always need
    numerics.
    relaxation: what is done with its ov block, always including the static
    gKS term (`static_ov_term`), which is relaxed with the rest:
      'none'    -- unrelaxed Delta_gamma^GW;
      'dyson'   -- iterated-Dyson density: one linear solve with the HF kernel
                   (bare Hartree + full exact exchange) on the gKS orbitals and
                   energies and mf's own J/K backend (`hf_kernel_mean_field`);
      'zvector' -- (default) the Z-vector solve with mf's own gKS kernel.
    Both coincide on an HF reference.
    numerics: the run's `Numerics`, the relaxation's (solve_cphf_relaxation)
    and the numerical modes' grids; required unless mode='casida' and
    relaxation='none'.
    auxbasis, screen_r_cut: the space-time mode's; refused with any other mode.
    """
    mode = _choice(mode, MODES, 'mode')
    relaxation = _choice(relaxation, RELAXATIONS, 'relaxation')
    for name, value in (('auxbasis', auxbasis), ('screen_r_cut', screen_r_cut)):
        if value is not None and mode != 'space-time':
            raise ValueError(f"{name} is a 'space-time' mode setting; got mode={mode!r}.")
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("compute_gw_density_matrix is currently restricted-spin only.")
    if mode == 'casida':
        numerics = relaxation_numerics(relaxation != 'none', numerics,
                                       'compute_gw_density_matrix')
    else:
        numerics = required_numerics(numerics, f"compute_gw_density_matrix mode={mode!r}")
        _refuse_numerical_mode(mf, mode, polarizability, df)

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)

    if mode == 'imagfrequency':
        blocks = imagfrequency_unrelaxed_blocks(mf, mol, eps, nocc, numerics)
    elif mode == 'space-time':
        blocks = space_time_unrelaxed_blocks(mf, mol, eps, nocc, numerics,
                                             auxbasis=auxbasis,
                                             screen_r_cut=screen_r_cut)
    else:
        if df:
            df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
            eri_chemist = None
        else:
            df_coeff = None
            eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        blocks = GWDensityMatrixSolver(
            eps, df_coeff=df_coeff, eri_chemist=eri_chemist, eta=eta
        ).compute_unrelaxed_blocks(nocc, mf, mol, polarizability=polarizability)
    dgamma_oo, dgamma_ov, dgamma_vv = (SPIN_SUM * block for block in blocks)

    mo = mf.mo_coeff
    if relaxation == 'none':
        dgamma = np.zeros((nmo, nmo))
        dgamma[:nocc, :nocc] = dgamma_oo
        dgamma[nocc:, nocc:] = dgamma_vv
        dgamma[:nocc, nocc:] = dgamma_ov
        dgamma[nocc:, :nocc] = dgamma_ov.T
    else:
        mf_hess = hf_kernel_mean_field(mf) if relaxation == 'dyson' else mf
        dgamma = solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv,
                                       numerics)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao


def _refuse_numerical_mode(mf, mode, polarizability, df):
    """The numerical modes' restrictions, each refused by name."""
    where = f"compute_gw_density_matrix mode={mode!r}"
    if str(polarizability).upper() != 'RPA':
        raise NotImplementedError(f"{where} screens with RPA only; got "
                                  f"polarizability={polarizability!r}.")
    if not df:
        raise NotImplementedError(f"{where} needs a three-index object; df=False "
                                  "(four-centre integrals) is not supported.")
    if not isinstance(environment_of(mf), NoEnvironment):
        raise NotImplementedError(f"{where} is gas phase only; mf carries an "
                                  f"environment ({type(environment_of(mf)).__name__}).")
    if beta_from_mf(mf) is not None:
        raise NotImplementedError(f"{where} is T = 0 only; mf is thermally smeared.")


def imagfrequency_unrelaxed_blocks(mf, mol, eps, nocc, numerics):
    """Per-channel (oo, ov, vv) of the unrelaxed Delta gamma^GW on the
    imaginary axis: the dynamic blocks of the full Sigma_c,pq(mu + i w)
    (`correlation_self_energy_matrix_imaginary_axis`, already per spin, so no
    w-normalization) on the outer grid, plus the closed-form static ov term."""
    points, weights = outer_frequency_grid(eps, nocc, numerics)
    sigma, mu = correlation_self_energy_matrix_imaginary_axis(mf, mol, nocc, points,
                                                              numerics)
    oo, ov, vv = dynamic_density_blocks(eps - mu, sigma, points, weights, nocc)
    return oo, ov + static_ov_term(mf, mol, eps, nocc), vv


def space_time_unrelaxed_blocks(mf, mol, eps, nocc, numerics, auxbasis=None,
                                screen_r_cut=None):
    """`imagfrequency_unrelaxed_blocks` on the space-time route's full
    Sigma_c,pq(mu + i w) (`correlation_self_energy_matrix_space_time`, per
    spin): separable factors on numerics.isdf's grid and the QP route's
    tau <-> omega transforms."""
    points, weights = outer_frequency_grid(eps, nocc, numerics)
    sigma, mu = correlation_self_energy_matrix_space_time(
        mf, mol, nocc, points, numerics, auxbasis=auxbasis,
        screen_r_cut=screen_r_cut)
    oo, ov, vv = dynamic_density_blocks(eps - mu, sigma, points, weights, nocc)
    return oo, ov + static_ov_term(mf, mol, eps, nocc), vv


def outer_frequency_grid(eps, nocc, numerics):
    """(points, weights) on omega >= 0 of the outer integral of the GW
    density, by `numerics.gwdensity`: 'gauss_legendre' (w0 the gap-scaled
    one when None) or 'minimax' (a tabulated size, over the particle-hole
    window of `eps`)."""
    cfg = numerics.gwdensity
    if cfg.grid == 'gauss_legendre':
        w0 = gap_scaled_w0(eps, nocc) if cfg.w0 is None else cfg.w0
        return gauss_legendre_grid(cfg.nfreq, w0)
    if cfg.grid == 'minimax':
        gap, span, _ = particle_hole_window(eps, nocc)
        return minimax_frequency_grid(cfg.nfreq, gap, span)
    raise ValueError(f"Unknown grid={cfg.grid!r}; choose one of "
                     "'gauss_legendre', 'minimax'.")


def dynamic_density_blocks(eps_mu, sigma, points, weights, nocc):
    """(oo, ov, vv): per channel, the real blocks of
    (1/2 pi) int_{-inf}^{inf} G_pp Sigma_pq G_qq d omega, G_pp = 1/(i omega -
    (eps_p - mu)), from the omega >= 0 nodes: Sigma(mu - i omega) is
    Sigma(mu + i omega)* for real orbitals, so the full axis is twice the
    real part of the half axis. `sigma` is (len(points), nmo, nmo)."""
    g = 1.0 / (1j * np.asarray(points)[:, None] - np.asarray(eps_mu)[None, :])
    integrand = g[:, :, None] * np.asarray(sigma) * g[:, None, :]
    gamma = np.einsum('w,wpq->pq', np.asarray(weights),
                      integrand.real) / np.pi
    return gamma[:nocc, :nocc], gamma[:nocc, nocc:], gamma[nocc:, nocc:]
