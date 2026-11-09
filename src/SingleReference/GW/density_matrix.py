import numpy as np

from src.SingleReference.common.mean_field_operators import mean_field_static_operator
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.orbital_spaces import get_occ_virt_indices, particle_hole_window
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid)
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_density_fitting_coefficients,
    get_spin_factor,
    get_two_electron_integrals_chemist,
    require_gas_phase_ground_state,
)
from src.SingleReference.DensityMatrix.density_matrix import (
    hf_kernel_mean_field,
    solve_cphf_relaxation,
    relaxation_numerics,
)
from src.SingleReference.GW.imaginary_axis import (
    self_energy_matrix_imaginary_axis)
from src.SingleReference.GW.space_time import self_energy_matrix_space_time
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.Base.numerics.numerics import required_numerics

MODES = ('casida', 'imagfrequency', 'space-time')
RELAXATIONS = ('none', 'dyson', 'zvector')


def compute_gw_density_matrix(mf, mol=None, polarizability='RPA', df=True,
                               mode='casida', relaxation='zvector',
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
    Every mode takes a closed-shell, restricted, gas-phase, T = 0 mean field
    only. The numerical modes take RPA screening on a density-fitted
    three-index object (df=True) only, and always need numerics.
    Every mode's unrelaxed ov block includes the static gKS term
    (`static_ov_term_unrelaxed`).
    relaxation: what is done with that ov block (`relax_gw_density`):
      'none'    -- unrelaxed Delta_gamma^GW;
      'dyson'   -- iterated-Dyson density: one linear solve with the HF kernel
                   (bare Hartree + full exact exchange) on the gKS orbitals and
                   energies and mf's own J/K backend (`hf_kernel_mean_field`);
      'zvector' -- (default) the Z-vector solve with mf's own gKS kernel.
    Both coincide on an HF reference.
    numerics: the run's `Numerics`, the relaxation's (`relax_gw_density`)
    and the numerical modes' grids; required unless mode='casida' and
    relaxation='none'.
    auxbasis, screen_r_cut: the space-time mode's; refused with any other mode.
    """
    mode = _choice(mode, MODES, 'mode')
    relaxation = _choice(relaxation, RELAXATIONS, 'relaxation')
    for name, value in (('auxbasis', auxbasis), ('screen_r_cut', screen_r_cut)):
        if value is not None and mode != 'space-time':
            raise ValueError(f"{name} is a 'space-time' mode setting; got mode={mode!r}.")
    where = f"compute_gw_density_matrix mode={mode!r}"
    # The reference's scope, one set for every mode (the numerical modes'
    # Sigma builders carry the same guards for their own callers).
    require_gas_phase_ground_state(mf, where, mol=mol)

    mol = mol if mol is not None else mf.mol
    nocc = mol.nelectron // 2
    # the spatial density sums the channels: restricted, so twice the one
    spin_factor = get_spin_factor('restricted')
    eps = get_orbital_energies(mf, representation='spatial')
    nmo = len(eps)

    if mode == 'casida':
        numerics = relaxation_numerics(relaxation != 'none', numerics, where)
        blocks = casida_unrelaxed_blocks(mf, mol, eps, nocc, polarizability, df)
    else:
        numerics = required_numerics(numerics, where)
        _refuse_casida_only_options(where, polarizability, df)
        if mode == 'imagfrequency':
            blocks = imagfrequency_unrelaxed_blocks(mf, mol, eps, nocc, numerics)
        else:
            blocks = space_time_unrelaxed_blocks(mf, mol, eps, nocc, numerics,
                                                 auxbasis=auxbasis, screen_r_cut=screen_r_cut)
    dgamma_oo, dgamma_ov, dgamma_vv = (spin_factor * block for block in blocks)

    mo = mf.mo_coeff
    if relaxation == 'none':
        dgamma = np.zeros((nmo, nmo))
        dgamma[:nocc, :nocc] = dgamma_oo
        dgamma[nocc:, nocc:] = dgamma_vv
        dgamma[:nocc, nocc:] = dgamma_ov
        dgamma[nocc:, :nocc] = dgamma_ov.T
    else:
        dgamma = relax_gw_density(mf, relaxation, nocc, dgamma_oo, dgamma_ov, dgamma_vv,
                                  numerics)

    dm_ao = mf.make_rdm1() + mo @ dgamma @ mo.T
    return dm_ao


# Shared by every mode.


def _choice(value, choices, keyword):
    """`value` normalized as calc_qp_energy's mode (lower case, '_' -> '-',
    'imag-frequency' -> 'imagfrequency'), refused unless one of `choices`."""
    name = str(value).lower().replace('_', '-')
    name = {'imag-frequency': 'imagfrequency'}.get(name, name)
    if name not in choices:
        raise ValueError(f"Unknown {keyword}={value!r}; choose one of "
                         + ', '.join(repr(c) for c in choices) + '.')
    return name


def static_ov_term_unrelaxed(mf, mol, eps, nocc):
    """Per-channel static gKS ov term <i|Sigma_x - v_xc|a> / (e_i - e_a),
    (nocc, nvir), in closed form from `mean_field_static_operator` (the
    per-spin -K/2 - v_xc at mf's density); no spin factor. Zero on HF.

    The same for every mode: each mode's unrelaxed ov block is its own
    dynamic (Sigma_c) part plus this term, and a relaxation relaxes both."""
    occ, virt = get_occ_virt_indices(eps, nocc)
    mo = mf.mo_coeff
    sx_ia = (mo.T @ mean_field_static_operator(mf, mol) @ mo)[np.ix_(occ, virt)]
    return sx_ia / (eps[occ][:, None] - eps[virt][None, :])


def relax_gw_density(mf, relaxation, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics):
    """The relaxed MO-basis Delta_gamma^GW (nmo, nmo) from the spatial
    unrelaxed blocks: `solve_cphf_relaxation` on the Hessian object the
    relaxation names. relaxation='none' does not come here.
      'zvector' -- `mf` itself: its own gKS kernel;
      'dyson'   -- `hf_kernel_mean_field(mf)`: the HF kernel J - K/2 on mf's
                   orbitals, energies and J/K backend.
    numerics: the run's `Numerics` (the Z-vector solve's tolerances)."""
    if relaxation == 'zvector':
        mf_hess = mf
    elif relaxation == 'dyson':
        mf_hess = hf_kernel_mean_field(mf)
    else:
        raise ValueError(f"relax_gw_density relaxes with 'zvector' or 'dyson'; "
                         f"got relaxation={relaxation!r}.")
    return solve_cphf_relaxation(mf_hess, nocc, dgamma_oo, dgamma_ov, dgamma_vv, numerics)


# mode='casida': the analytic sum over the Casida poles of the screening.


def casida_screening(eps, nocc, df_coeff=None, eri_chemist=None, polarizability='RPA'):
    """(omega, X, Y) of the restricted Casida problem of the screening:
    'RPA' (Coulomb only), 'TDHF' (bare exchange) or 'BSE' (statically
    screened exchange, W_aux = W_RPA(0)), on `df_coeff` or `eri_chemist`."""
    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, eri_chemist=eri_chemist,
                                     spin_mode='restricted')
    channel = str(polarizability).upper()
    if channel == 'RPA':
        build_kw = dict(lBSE=False)
    elif channel == 'TDHF':
        build_kw = dict(lBSE=True, W_aux=None)
    elif channel == 'BSE':
        w_aux = lr_solver.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]
        build_kw = dict(lBSE=True, W_aux=w_aux)
    else:
        raise ValueError(f"Unknown screening mode '{polarizability}'; choose 'RPA', 'TDHF', or 'BSE'.")
    return CasidaSolver(*lr_solver.build_casida_matrices(nocc, **build_kw)).solve()


def screened_coulomb_residues(eps, nocc, X, Y, df_coeff=None, eri_chemist=None):
    """w^s_pq (nstate, nmo, nmo): the residues of W at the Casida poles, from
    the self-energy's own `get_chi_a` (`SelfEnergySolver`), so the density
    and the QP Sigma share one convention: restricted w without its sqrt(2),
    restored by `get_spin_factor` where w enters squared."""
    return SelfEnergySolver(eps, df_coeff=df_coeff, eri_chemist=eri_chemist,
                            spin_mode='restricted').get_chi_a(nocc, X, Y, p_state=None)


def casida_unrelaxed_blocks(mf, mol, eps, nocc, polarizability='RPA', df=True,
                            screening=None):
    """Per-channel (oo, ov, vv) of the unrelaxed Delta gamma^GW from the
    Casida poles of `polarizability` screening, on the mean field's density
    fit (df) or on four-centre integrals; the spatial density is
    `get_spin_factor` times these. `screening`: a precomputed
    (omega, X, Y) of `casida_screening` on the same integrals, else solved here.

    The ov block includes the static term (`static_ov_term_unrelaxed`), so on HF it is
    purely dynamic."""
    if df:
        df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        eri_chemist = None
    else:
        df_coeff = None
        eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    if screening is None:
        screening = casida_screening(eps, nocc, df_coeff, eri_chemist, polarizability)
    omega, X, Y = screening

    occ, virt = get_occ_virt_indices(eps, nocc)
    ei, ea = eps[occ], eps[virt]
    w = screened_coulomb_residues(eps, nocc, X, Y, df_coeff, eri_chemist)
    w_ic = w[:, occ, :][:, :, virt]
    w_ak = w[:, virt, :][:, :, occ]
    w_ik = w[:, occ, :][:, :, occ]
    w_bc = w[:, virt, :][:, :, virt]

    t_ic = w_ic / (ei[None, :, None] - ea[None, None, :] - omega[:, None, None])
    t_ak = w_ak / (ea[None, :, None] - ei[None, None, :] + omega[:, None, None])

    # w enters squared, so the restricted sqrt(2) is restored by the spin
    # factor, as in the restricted self-energy (SelfEnergySolver._spin). The
    # static term is already per channel and takes no such factor.
    spin_factor = get_spin_factor('restricted')
    dgamma_oo = -spin_factor * np.einsum('sic,sjc->ij', t_ic, t_ic)
    dgamma_vv = spin_factor * np.einsum('sak,sbk->ab', t_ak, t_ak)
    dynamic_ov = (np.einsum('sik,sbk->ib', w_ik, t_ak)
                  + np.einsum('sic,sbc->ib', t_ic, w_bc))
    dgamma_ov = (spin_factor * dynamic_ov / (ei[:, None] - ea[None, :])
                 + static_ov_term_unrelaxed(mf, mol, eps, nocc))
    return dgamma_oo, dgamma_ov, dgamma_vv


# Shared by the numerical modes: the outer integral over the full Sigma_c,pq.


def _refuse_casida_only_options(where, polarizability, df):
    """Options only the Casida route takes, refused by name on the numerical
    modes; `where` names the call in the message."""
    if str(polarizability).upper() != 'RPA':
        raise NotImplementedError(f"{where} screens with RPA only; got "
                                  f"polarizability={polarizability!r}.")
    if not df:
        raise NotImplementedError(f"{where} needs a three-index object; df=False "
                                  "(four-centre integrals) is not supported.")


def outer_frequency_grid(eps, nocc, numerics):
    """(points, weights) on omega >= 0 of the outer integral of the GW
    density, by `numerics.gwdensity`: 'gauss_legendre' (w0_gl, or the gap-scaled
    one when None) or 'minimax' (a tabulated size, fitted to [gap/2, span]
    of the particle-hole window of `eps`).

    The minimax window starts at gap/2, not at the gap: each G_pp(i w) of
    the integrand is a Lorentzian of width |eps_p - mu|, which for the
    frontier orbitals around the mid-gap mu is gap/2. The upper end stays
    at span (not max |eps_p - mu|), since Sigma adds poles beyond the
    orbital energies."""
    cfg = numerics.gwdensity
    if cfg.grid == 'gauss_legendre':
        w0 = gap_scaled_w0(eps, nocc) if cfg.w0_gl is None else cfg.w0_gl
        return gauss_legendre_grid(cfg.nfreq, w0)
    if cfg.grid == 'minimax':
        gap, span, _ = particle_hole_window(eps, nocc)
        return minimax_frequency_grid(cfg.nfreq, gap / 2, span)
    # GWDensityNumerics admits IMAGINARY_FREQUENCY_GRIDS, a list shared with
    # other routes: a grid added there without a branch here stops here.
    raise ValueError(f"outer_frequency_grid has no {cfg.grid!r} branch; the GW "
                     "density's outer grid is 'gauss_legendre' or 'minimax'.")


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


# mode='imagfrequency'.


def imagfrequency_unrelaxed_blocks(mf, mol, eps, nocc, numerics):
    """Per-channel (oo, ov, vv) of the unrelaxed Delta gamma^GW on the
    imaginary axis: the dynamic blocks of the full Sigma_c,pq(mu + i w)
    (`self_energy_matrix_imaginary_axis`, already per spin, so no
    w-normalization) on the outer grid, plus the closed-form static ov term."""
    points, weights = outer_frequency_grid(eps, nocc, numerics)
    sigma, mu = self_energy_matrix_imaginary_axis(mf, mol, nocc, points, numerics)
    oo, ov, vv = dynamic_density_blocks(eps - mu, sigma, points, weights, nocc)
    return oo, ov + static_ov_term_unrelaxed(mf, mol, eps, nocc), vv


# mode='space-time'.


def space_time_unrelaxed_blocks(mf, mol, eps, nocc, numerics, auxbasis=None,
                                screen_r_cut=None):
    """`imagfrequency_unrelaxed_blocks` on the space-time route's full
    Sigma_c,pq(mu + i w) (`self_energy_matrix_space_time`, per
    spin): separable factors on numerics.isdf's grid and the QP route's
    tau <-> omega transforms."""
    points, weights = outer_frequency_grid(eps, nocc, numerics)
    sigma, mu = self_energy_matrix_space_time(
        mf, mol, nocc, points, numerics, auxbasis=auxbasis,
        screen_r_cut=screen_r_cut)
    oo, ov, vv = dynamic_density_blocks(eps - mu, sigma, points, weights, nocc)
    return oo, ov + static_ov_term_unrelaxed(mf, mol, eps, nocc), vv
