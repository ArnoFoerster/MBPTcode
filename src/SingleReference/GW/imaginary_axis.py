"""GW quasiparticle energies on the imaginary-frequency axis.

The RPA screened interaction W(i omega) on a minimax or Gauss-Legendre grid,
the correlation self-energy by numerical convolution over that grid
(`self_energy_imaginary_axis`), and its continuation to the real axis by Pade
(`solve_qp_energy_imaginary_axis`), restricted and unrestricted, at T = 0 or
at a finite temperature. And the full-matrix Sigma_c,pq(mu + i w) of the GW
density (`correlation_self_energy_matrix_imaginary_axis`), whose inner
integral goes through a Lorentzian fit of W_c rather than that quadrature.
"""
import time as _time

import numpy as np

from pyscf import scf

from src.Base.numerics.constants import (DEFAULT_BROADENING_ETA, GW_DENSITY_W_FIT_HIGH,
                                         GW_DENSITY_W_FIT_LOW, GW_DENSITY_W_FIT_TIKHONOV)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.utils.grids.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid
from src.Base.utils.grids.matsubara import (beta_from_mf, ir_continuation_order,
                                            self_energy_range, thermal_e_min)
from src.Base.pyscf.pyscf_interface import (get_orbital_energies,
                                     get_density_fitting_coefficients,
                                     get_df_coefficients_ov,
                                     require_closed_shell_or_unrestricted,
                                     require_gas_phase, require_restricted,
                                     require_zero_temperature, spin_index)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.orbital_spaces import (get_occ_virt_indices,
                                     particle_hole_window, transition_range)
from src.SingleReference.GW.reaction_field import (
    bare_self_energy, environment_quasiparticle_shift, shift_after_solve)
from src.SingleReference.GW.qp_solve import (finish_qp,
                                             imaginary_axis_sample_points)


def self_energy_imaginary_axis(df_coeff, eps, nocc, p_state, freq_points, freq_weights,
                                W_grid, query_freqs):
    """Correlation self-energy Sigma_c,pp(i*query_freqs) by numerical
    convolution over the imaginary-frequency grid:

    Sigma_pp(i*w) = -(1/2pi) sum_m sum_k weight_k * [G_m(i*(w+wk)) + G_m(i*(w-wk))] * w_pp^m(i*wk)

    with w_pp^m built from Wc = W - I, the correlation-only screened
    interaction (the exchange is the separate static Sigma_x - v_xc term).
    Returns a complex array of shape (len(query_freqs)).

    eps and query_freqs share an origin: passing eps - mu returns
    Sigma_pp(mu + i*w), i.e. samples on the vertical line Re z = mu, so mu
    must lie inside the gap.
    """
    
    # w_pm^m(i*w_k) for all m, all grid points k: shape (nfreq, norb)
    Cp = df_coeff[:, p_state, :] # (naux, norb)
    w_pm = np.empty((len(W_grid), Cp.shape[1]))
    for k in range(len(W_grid)):
        w_pm[k] = np.einsum('Pm,Pm->m', Cp, W_grid[k] @ Cp)
    w_pm -= np.einsum('Pm,Pm->m', Cp, Cp)[None, :]

    query_freqs = np.atleast_1d(query_freqs)
    sigma = np.zeros(len(query_freqs), dtype=complex)
    for iq, w in enumerate(query_freqs):
        for k in range(len(freq_points)):
            wk = freq_points[k]
            Gp = 1.0 / (1j * (w + wk) - eps)     # (norb,)
            Gm = 1.0 / (1j * (w - wk) - eps)
            sigma[iq] += freq_weights[k] * np.sum((Gp + Gm) * w_pm[k])
    sigma *= -1.0 / (2.0 * np.pi)
    return sigma


def screened_pair_weights(df_coeff, W):
    """w^m_pq(i w') = sum_PQ B_P,pm (W - I)_PQ B_Q,qm, as (nmo, nmo, nmo)
    indexed [m, p, q]: the full-matrix weight of the GW convolution at one
    W-grid point (`self_energy_imaginary_axis` takes its diagonal p = q)."""
    flat = df_coeff.reshape(df_coeff.shape[0], -1)
    WC = (W @ flat - flat).reshape(df_coeff.shape)
    return np.einsum('Ppm,Pqm->mpq', df_coeff, WC, optimize=True)


def lorentzian_fit_matrix(freq_points, poles, tikhonov):
    """P (len(poles), len(freq_points)): the Tikhonov-regularized least-squares
    fit f(i w') ~ sum_l A_l 2 Omega_l/(w'^2 + Omega_l^2) of samples f_k = f(i w'_k),
    A = P f. Columns normalized; `tikhonov` relative to the largest singular
    value squared (lambda = 0 is rank-deficient for overlapping poles)."""
    F = 2.0 * poles[None, :] / (np.asarray(freq_points)[:, None] ** 2 + poles[None, :] ** 2)
    norm = np.linalg.norm(F, axis=0)
    U, s, Vt = np.linalg.svd(F / norm, full_matrices=False)
    filt = s / (s ** 2 + tikhonov * s[0] ** 2)
    return ((Vt.T * filt) @ U.T) / norm[:, None]


def one_pole_convolution(eps_mu, poles, query_freqs):
    """K (nquery, npoles, nmo): the exact convolution of G_m with one even
    Lorentzian,
        -(1/2pi) int dw' 1/(i(w + w') - e_m) 2 Omega/(w'^2 + Omega^2)
            = -1/(i w - e_m - sign(e_m) Omega),
    e_m = eps_m - mu nonzero (mu inside the gap)."""
    e = np.asarray(eps_mu)[None, None, :]
    return -1.0 / (1j * np.atleast_1d(query_freqs)[:, None, None]
                   - e - np.sign(e) * np.asarray(poles)[None, :, None])


def self_energy_matrix_imaginary_axis(df_coeff, eps_mu, freq_points, W_grid, query_freqs,
                                      poles, tikhonov):
    """The full correlation self-energy Sigma_c,pq(i*query_freqs), complex
    (nquery, nmo, nmo), per spin channel, with the inner w' integral done
    through a Lorentzian fit of W_c = W - I on the W grid:

        W_c(i w') ~ sum_l A_l 2 Omega_l/(w'^2 + Omega_l^2),  A_l = sum_k P_lk W_c(i w'_k),
        Sigma_pq(i w) = sum_k sum_m M_km(w) w^m_pq(i w'_k),
        M_km(w) = sum_l P_lk K(e_m, Omega_l, w)

    (`lorentzian_fit_matrix`, `one_pole_convolution`, `screened_pair_weights`).
    The fit only changes the inner weights, so A is never formed. Unlike the
    QP route's plain quadrature (`self_energy_imaginary_axis`) it integrates the
    sharp G_m(i(w + w')) exactly. eps_mu and query_freqs share an origin:
    eps - mu gives Sigma(mu + i*w). Streams over the W grid: nfreq naux nmo^3.
    """
    df_coeff = np.asarray(df_coeff)
    nmo = df_coeff.shape[1]
    P = lorentzian_fit_matrix(freq_points, poles, tikhonov)
    M = np.einsum('lk,qlm->kqm', P, one_pole_convolution(eps_mu, poles, query_freqs))
    sigma = np.zeros((M.shape[1], nmo, nmo), dtype=complex)
    for k in range(len(freq_points)):
        sigma += np.einsum('qm,mpr->qpr', M[k], screened_pair_weights(df_coeff, W_grid[k]),
                           optimize=True)
    return sigma


def correlation_self_energy_matrix_imaginary_axis(mf, mol, nocc, query_freqs, numerics):
    """(Sigma_c,pq(mu + i*query_freqs), mu) of a restricted gas-phase mean
    field at T = 0. The three-index object of all orbitals (the mean field's
    density fit, or the exact integrals' decomposition without one) and the
    RPA W(i w') on the minimax grid of `numerics.imagfreq.nfreq` points over
    the particle-hole window, as `solve_qp_energy_imaginary_axis` builds them;
    the correlation part W - I only; mu the mid-gap. The inner integral is the
    Lorentzian fit of `self_energy_matrix_imaginary_axis`, its poles
    log-spaced over [GW_DENSITY_W_FIT_LOW * gap, GW_DENSITY_W_FIT_HIGH * span],
    one per W-grid point, regularized by GW_DENSITY_W_FIT_TIKHONOV."""
    where = 'correlation_self_energy_matrix_imaginary_axis'
    require_closed_shell_or_unrestricted(mf, where, mol=mol)
    require_restricted(mf, where)
    require_gas_phase(mf, where)
    require_zero_temperature(mf, where)
    nfreq = numerics.imagfreq.nfreq
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted',
                              eta=DEFAULT_BROADENING_ETA)
    e_min, e_max, mu = particle_hole_window(eps, nocc)
    freq_points, _ = minimax_frequency_grid(nfreq, e_min, e_max)
    W_grid = lr.solve_rpa_screening(freq_points, nocc, is_imaginary=True)
    poles = np.geomspace(GW_DENSITY_W_FIT_LOW * e_min, GW_DENSITY_W_FIT_HIGH * e_max,
                         len(freq_points))
    sigma = self_energy_matrix_imaginary_axis(df_coeff, eps - mu, freq_points, W_grid,
                                              query_freqs, poles, GW_DENSITY_W_FIT_TIKHONOV)
    return sigma, mu


def _unrestricted_screening(mol, mf, nocc, states, spin):
    """(alpha and beta spectra, the solver of the spin-summed W, the rows
    B[:, states, :] of channel `spin`).

    chi0 = chi0_alpha + chi0_beta, each spin's particle-hole block in its own
    orbitals, and no spin factor: that is `LinearResponseSolver`'s unrestricted
    screening, handed the two occupied-virtual blocks alone.
    """
    spectra = tuple(np.asarray(e, float)
                    for e in get_orbital_energies(mf, representation='spatial'))
    if getattr(mf, 'with_df', None) is None:
        coeff = get_density_fitting_coefficients(mol, mf,
                                                 representation='spatial')[:2]
        lr = LinearResponseSolver(spectra, coeff_df=coeff,
                                  spin_mode='unrestricted',
                                  eta=DEFAULT_BROADENING_ETA)
        return spectra, lr, coeff[spin][:, states, :]
    blocks, rows = [], None
    for s, (e, c) in enumerate(zip(spectra, mf.mo_coeff)):
        occ, virt = get_occ_virt_indices(e, nocc[s])
        C_ov, C_row = get_df_coefficients_ov(
            mol, mf, occ, virt, rows=list(states) if s == spin else None,
            mo_coeff=c)
        blocks.append(C_ov)
        rows = C_row if s == spin else rows
    lr = LinearResponseSolver(spectra, coeff_ov=tuple(blocks),
                              spin_mode='unrestricted',
                              eta=DEFAULT_BROADENING_ETA)
    return spectra, lr, rows


def solve_qp_energy_imaginary_axis(mf, mol, nocc, p_state, numerics=None, w0=None, grid='minimax',
                                    greedy=True,
                                    dm_correction=None, timings=None, beta=None,
                                    eps_anchor=None, spin_channel='alpha',
                                    with_z=False, pade_freq=None):
    """
    GW@RPA quasiparticle energy via the imaginary-frequency-axis route:
    RPA W(i*omega) -> convolution -> Pade continuation.

    numerics: the run's `Numerics`; its imagfreq section holds the W(i omega)
    quadrature's point count, nfreq.

    No broadening to set: chi0(i*omega) has the real denominator
    -2d/(d^2 + w^2) and Sigma_c reaches the real axis by Pade rather than at
    w + i.eta, so this signature advertises none -- a caller wanting eta
    reaches mode='casida' instead, the only route it acts on.

    beta: inverse temperature in inverse Hartree, for a system whose gap is too
    small for a T = 0 grid. Setting it does two things:
      * the grid's e_min (or w0) is floored at the first Matsubara frequency
        pi/beta. This makes the tabulated minimax and Gauss-Legendre
        quadratures defined at all when the gap closes;
      * the Pade continuation is capped at `ir_continuation_order(beta, wmax)`
        nodes: the number of structures the imaginary-axis data can actually
        resolve at that temperature and bandwidth, rather than however many
        sample points happen to exist.
    `beta=None` reads it off the mean field when that carries Fermi-Dirac
    smearing (`beta_from_mf`), and otherwise leaves everything at T = 0.

    UNRESTRICTED (nocc = (nalpha, nbeta)): ONE W from chi0_alpha + chi0_beta,
    and the self-energy, the static exchange and the Eq. (18) shift of the
    channel `spin_channel`, Sigma_s = -G_s W, sampled on the line through that
    channel's own mid-gap. The frequency grid spans the transitions of both
    spins, since both build the W it integrates. An `eps_anchor` of shape
    (2, nmo) is read at the channel's row.

    with_z: also return Z = 1/(1 - dRe Sigma_c/dw) of the continued fraction
    at each root, as the space-time route's Pade does: (roots, z).

    pade_freq: the imaginary frequencies Sigma_c is sampled at for the Pade
    continuation. None takes this route's own W nodes;
    `analytic_continuation.pade_nodes` is the space-time route's node rule.
    """

    require_closed_shell_or_unrestricted(mf, 'solve_qp_energy_imaginary_axis',
                                         mol=mol)
    nfreq = numerics_or_default(numerics).imagfreq.nfreq
    unrestricted = isinstance(mf, scf.uhf.UHF)
    spin = spin_index(spin_channel) if unrestricted else None
    eps = get_orbital_energies(mf, representation='spatial')
    if not unrestricted:
        occ_idx, virt_idx = get_occ_virt_indices(eps, nocc)

    # A WINDOW SHARES ONE SCREENING BUILD: the rows of the three-index object
    # are sliced for every requested state at once, and W below is built once.
    states = np.atleast_1d(p_state).astype(int)
    scalar = np.ndim(p_state) == 0

    # The self-energy screens bare and takes the continuum as the static Eq.
    # (18) shift, the same split the space-time route makes.
    reaction_field = environment_quasiparticle_shift(mf, mol, nocc)

    # Use B only as B[:, occ, virt] (for chi0) and B[:, states, :] (for Sigma)
    with bare_self_energy(mf, reaction_field):
        if unrestricted:
            spectra, lr, C_row = _unrestricted_screening(mol, mf, nocc, states,
                                                         spin)
        elif hasattr(mf, 'with_df') and mf.with_df is not None:
            C_ov, C_row = get_df_coefficients_ov(mol, mf, occ_idx, virt_idx,
                                                 rows=list(states))
            lr = LinearResponseSolver(eps, coeff_ov=C_ov,
                                      spin_mode='restricted',
                                      eta=DEFAULT_BROADENING_ETA)
        else:
            # Without with_df there is no three-index object to slice, so that
            # case still goes through the full builder.
            df_coeff = get_density_fitting_coefficients(
                mol, mf, representation='spatial')
            C_row = df_coeff[:, states, :]
            lr = LinearResponseSolver(eps, coeff_df=df_coeff,
                                      spin_mode='restricted',
                                      eta=DEFAULT_BROADENING_ETA)

    # inverse temperature
    if beta is None:
        beta = beta_from_mf(mf)

    # occ-virt ranges
    screening_nocc = nocc
    if unrestricted:
        e_min, e_max = transition_range(spectra, nocc)
        eps, nocc = spectra[spin], nocc[spin]
    else:
        e_min, e_max, _ = particle_hole_window(eps, nocc)
    if beta is not None:
        e_min = thermal_e_min(beta, e_min)

    # initialize frequency grids
    if grid == 'minimax':
        freq_points, freq_weights = minimax_frequency_grid(nfreq, e_min, e_max)
    elif grid == 'gauss_legendre':
        if w0 is None:
            w0 = gap_scaled_w0(eps, nocc, gap=e_min if beta is not None
                               or unrestricted else None)
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    else:
        raise ValueError(f"Unknown grid '{grid}'; choose 'minimax' or 'gauss_legendre'.")

    # calculate W on imaginary axis - This is the integration gird  
    _t = _time.time()
    W_grid = lr.solve_rpa_screening(freq_points, screening_nocc,
                                    is_imaginary=True)
    if timings is not None:
        timings['t_W'] = _time.time() - _t

    # chemical potential, mid gap
    mu = particle_hole_window(eps, nocc)[2]

    # imaginary axis sampling point
    w_sigma = self_energy_range(eps, mu, e_max)
    pade_order = None if beta is None else ir_continuation_order(beta, w_sigma)
    # A WINDOW SHARES ONE W. Everything above is state-independent; only the
    # sampling points, Sigma and the root solve carry a state index, so a
    # window costs one screening build and not one per orbital.
    # The equation is anchored on eps_p; evGW hands the mean-field spectrum
    # here while the screening above follows the corrected one.
    anchor = eps if eps_anchor is None else np.asarray(eps_anchor, float)
    if unrestricted and anchor.ndim == 2:
        anchor = anchor[spin]

    # Eq. (18) after the solve in G0W0, in the static term on an evGW cycle;
    # each state's Sigma_c sampled on its own branch of the axis
    z_points = freq_points if pade_freq is None else pade_freq
    sigma = [self_energy_imaginary_axis(
        C_row[:, [i], :], eps - mu, nocc, 0, freq_points, freq_weights, W_grid,
        imaginary_axis_sample_points(z_points, nocc, int(p), mu)[1])
        for i, p in enumerate(states)]
    return finish_qp(sigma[0] if scalar else sigma, anchor, nocc,
                     states[0] if scalar else states, mu, z_points, mf, mol,
                     dm_correction, greedy, timings,
                     reaction_field=reaction_field, spin=spin, with_z=with_z,
                     after_solve=shift_after_solve(eps_anchor),
                     max_order=pade_order, branch_sampled=True)

