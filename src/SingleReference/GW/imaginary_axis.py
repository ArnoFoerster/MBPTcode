import time as _time

import numpy as np

from pyscf import scf

from src.Base.constants import DEFAULT_BROADENING_ETA
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid, minimax_supported_sizes
from src.Base.utils.analyticalContinuation import greedy_pade_order, thiele_coefficients, pade_eval
from src.Base.utils.matsubara import (beta_from_mf, ir_continuation_order,
                                      self_energy_range, thermal_e_min)
from src.Base.pyscf_interface import (get_orbital_energies,
                                     get_density_fitting_coefficients,
                                     get_df_coefficients_ov,
                                     require_closed_shell_or_unrestricted,
                                     spin_index)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.base import get_occ_virt_indices, transition_range
from src.Solvers.qp_equation import solve_qp_equation
from src.SingleReference.GW.reaction_field import (
    bare_self_energy, environment_quasiparticle_shift)
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             solve_qp_from_imaginary_axis,
                                             imaginary_axis_sample_points)


def solve_screening_imaginary_axis(lr_solver, nocc, freq_points):
    """W(i*omega_k) = [I - P(i*omega_k)]^-1 on the imaginary-frequency grid"""
    return lr_solver.solve_rpa_screening(freq_points, nocc, is_imaginary=True)


def self_energy_imaginary_axis(df_coeff, eps, nocc, p_state, freq_points, freq_weights,
                                W_grid, query_freqs):
    """
    Correlation self-energy Sigma_c,pp(i*query_freqs) 
    by numerical convolution over the imaginary-frequency grid.

    Sigma_pp(i*w) = -(1/2pi) sum_m sum_k weight_k * [G_m(i*(w+wk)) + G_m(i*(w-wk))] * w_pp^m(i*wk)
    Wc = W - I is the correlation-only screened interaction
    exchange handled separately via the static Sigma_x - v_xc term). 
    Returns complex array of shape (len(query_freqs)).

    eps and query_freqs share an origin: passing eps - mu returns
    Sigma_pp(mu + i*w), i.e. samples on the vertical line Re z = mu. 
    Pick mu inside the gap!!!
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


def solve_qp_energy_imaginary_axis(mf, mol, nocc, p_state, nfreq=20, w0=None, grid='minimax',
                                    solver_mode='pole_strength', greedy=True,
                                    dm_correction=None, timings=None, beta=None,
                                    eps_anchor=None, spin_channel='alpha'):
    """
    GW@RPA quasiparticle energy via the imaginary-frequency-axis route:
    RPA W(i*omega) -> convolution -> Pade continuation.

    No broadening to set: chi0(i*omega) has the real denominator
    -2d/(d^2 + w^2) and Sigma_c reaches the real axis by Pade rather than at
    w + i.eta, so this signature advertises none -- a caller wanting eta
    reaches mode='casida' instead, the only route it acts on.

    beta: inverse temperature in inverse Hartree, for a system whose gap is too
    small for a T = 0 grid. 
    gap exceeds pi/beta:
      * the grid's e_min (or w0) is floored at the first Matsubara frequency
        pi/beta. This makes the tabulated minimax and Gauss-Legendre
        quadratures defined at all when the gap closes;
      * the Pade continuation is capped at `ir_continuation_order(beta, wmax)`
        nodes. This is the the number of structures the imaginary-axis data can 
        actually resolve at that temperature and bandwidth, rather than however 
        many sample points happen to exist.
    `beta=None` reads it off the mean field when that carries Fermi-Dirac
    smearing (`beta_from_mf`), and otherwise leaves everything at T = 0.

    UNRESTRICTED (nocc = (nalpha, nbeta)): ONE W from chi0_alpha + chi0_beta,
    and the self-energy, the static exchange and the Eq. (18) shift of the
    channel `spin_channel`, Sigma_s = -G_s W, sampled on the line through that
    channel's own mid-gap. The frequency grid spans the transitions of both
    spins, since both build the W it integrates. An `eps_anchor` of shape
    (2, nmo) is read at the channel's row.
    """

    require_closed_shell_or_unrestricted(mf, 'solve_qp_energy_imaginary_axis',
                                         mol=mol)
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
        occ, virt = get_occ_virt_indices(eps, nocc)
        e_min = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
    if beta is not None:
        e_min = thermal_e_min(beta, e_min)

    # initialize frequency grids
    if grid == 'minimax':
        freq_points, freq_weights = minimax_frequency_grid(nfreq, e_min, e_max)
    elif grid == 'gauss_legendre':
        if w0 is None:
            w0 = (0.5 * e_min if beta is not None or unrestricted
                  else gap_scaled_w0(eps, nocc))
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    else:
        raise ValueError(f"Unknown grid '{grid}'; choose 'minimax' or 'gauss_legendre'.")

    # calculate W on imaginary axis - This is the integration gird  
    _t = _time.time()
    W_grid = solve_screening_imaginary_axis(lr, screening_nocc, freq_points)
    if timings is not None:
        timings['t_W'] = _time.time() - _t

    # chemical potential, mid gap
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

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

    _t = _time.time()
    xc_diag = static_exchange_diagonal(mf, mol, states,
                                       dm_correction=dm_correction,
                                       reaction_field=reaction_field,
                                       spin=spin)
    out = []
    for i, p in enumerate(states):
        z_fit, iw_query = imaginary_axis_sample_points(freq_points, nocc,
                                                       int(p), mu)
        sigma_iw = self_energy_imaginary_axis(C_row[:, [i], :], eps - mu, nocc,
                                              0, freq_points, freq_weights,
                                              W_grid, iw_query)
        out.append(solve_qp_from_imaginary_axis(
            anchor, int(p), float(xc_diag[i]), z_fit, sigma_iw, greedy=greedy,
            solver_mode=solver_mode, max_order=pade_order))
    if timings is not None:
        timings['t_qp'] = _time.time() - _t
    return out[0] if scalar else np.asarray(out)

