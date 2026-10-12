"""Self-consistent GW (scGW): the dressed Green's function via Dyson.

`solve_qp_energy_space_time` only ever needs Sigma_c(i.omega) projected onto
one orbital diagonal, on a handful of Pade sample points -- enough to find
that orbital's QP pole, useless for a Dyson loop. Building an actual dressed
G(i.omega) needs the FULL Sigma_c(i.omega) matrix, on the SAME frequency grid
chi0/W already live on, so it round-trips cleanly to imaginary time for the
next iteration's polarizability. This module is deliberately kept apart from
space_time.py -- the O(N^3) route's tested, shared entry point -- rather than
threading a second output contract through it.

Sigma_c(i.omega) -> G(i.omega) here is the matrix analogue of the
W = [I - chi0]^-1 Dyson inversion `sigma_space_time` already does for the
screened interaction:

    G(i.omega) = [(i.omega + mu) I - diag(eps) - Sigma_c(i.omega)]^-1

Turning this into an actual self-consistency loop still needs G(i.omega)
transformed back to G(i.tau) and fed into a dressed replacement for
`chi0_imaginary_frequency` (which currently bakes in the bare-orbital
exponential form of G0) -- not implemented yet; this module stops at one
Dyson-dressed G.
"""
import numpy as np

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_frequency_grid, minimax_supported_sizes)
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_transform_weights,
                                           COSINE_TW, COSINE_WT, SINE_TW, SINE_WT)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import (self_energy_fit_ranges,
                                                   self_energy_matrix_imaginary_time,
                                                   sigma_ao_to_mo,
                                                   minimax_points_for_gw,
                                                   DEFAULT_TAU_TARGET)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency

DEFAULT_NTAU = 'auto'
DEFAULT_NFREQ = 'auto'

def get_freq_points(nfreq, ntau, e_min, e_max, w0):
    """ This function gets the frequency points either via the 
    minimax method, or via gauss-legendre method, just like
    in space_time.py
    """

    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = minimax_frequency_grid(ntau, e_min, e_max)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)

    return freq_points, freq_weights

def find_freq_points(nocc, eps, mu, ntau, e_min, e_max, w0, maxiter=5,
                     start_points=6, step=4, trace_tol=1e-8):
    """Sanity check that Tr[gamma] (built from the bare G0) actually
    converges as Gauss-Legendre points are added, THEN return the same
    minimax('auto') grid get_freq_points would have given anyway.

    Gauss-Legendre is only a cheap PROBE here, used because it can be built
    at any size (minimax cannot -- only the tabulated sizes in
    minimax_supported_sizes exist): a handful of GL sizes (maxiter, small by
    default) is enough to see whether the density integral is behaving, not
    to find a grid size to actually use. Regardless of the outcome, the
    frequency grid this function returns is unchanged: the minimax grid tied
    to ntau, exactly what get_freq_points('auto', ...) gives.
    """
    nmo = len(eps)
    occ_ref = np.zeros(nmo)
    occ_ref[:nocc] = 1.0

    n_freq = start_points
    freq_points_gl, freq_weights_gl = get_freq_points(n_freq, ntau, e_min, e_max, w0)
    sigma_0 = np.zeros((n_freq, nmo, nmo), dtype=complex)
    G0_gl = dyson_green_function(sigma_0, eps, mu, freq_points_gl)
    gamma_gl = density_matrix_scgw(G0_gl, freq_weights_gl)
    d_occ_old = np.max(np.abs(np.diag(gamma_gl) - occ_ref))

    converged = d_occ_old < trace_tol
    for _ in range(maxiter):
        if converged:
            break

        n_freq += step
        sigma_0 = np.zeros((n_freq, nmo, nmo), dtype=complex)
        freq_points_gl, freq_weights_gl = get_freq_points(n_freq, ntau, e_min, e_max, w0)
        G0_gl = dyson_green_function(sigma_0, eps, mu, freq_points_gl)
        gamma_gl = density_matrix_scgw(G0_gl, freq_weights_gl)
        d_occ = np.max(np.abs(np.diag(gamma_gl) - occ_ref))

        if d_occ_old - d_occ < 0:
            # stopped improving: not going to converge further this way
            break
        converged = d_occ < trace_tol
        d_occ_old = d_occ

    if converged:
        print(f'Gauss-Legendre density check converged: max|d_occ|={d_occ_old:.2e} '
             f'at nfreq={n_freq}')
    else:
        print(f'WARNING: Gauss-Legendre density check did NOT converge within '
             f'{maxiter} probes (max|d_occ|={d_occ_old:.2e} at nfreq={n_freq})')

    return get_freq_points('auto', ntau, e_min, e_max, w0)


def sigma_matrix_scgw_iteration1(mf, mol, nocc, ntau=DEFAULT_NTAU, nfreq=DEFAULT_NFREQ,
                      w0=1.0, auxbasis=None, radii=None, factors=None,
                      tau_target=DEFAULT_TAU_TARGET, timings=None):
    """Sigma_c(i.omega) as a full MO matrix, on chi0/W's own frequency grid.

    Same separable-factor/tau/frequency setup as `sigma_space_time`, but the
    self-energy is projected with `sigma_ao_to_mo` (full matrix) instead of
    the diagonal, and evaluated with `omega_out = freq_points` instead of a
    Pade grid -- the grid chi0/W already live on, since a Dyson loop needs
    Sigma_c and G comparable, frequency by frequency, to what built W.

    Returns (sigma_mo, eps, mu, freq_points, freq_weights); sigma_mo is
    (nfreq, nmo, nmo) complex. `freq_weights` are the matching minimax/Gauss-
    Legendre quadrature weights on `[0, infty)` -- not needed to build G
    itself (`dyson_green_function` closes the Dyson equation pointwise, no
    integral), but exactly what a frequency-integral quantity built from G,
    such as the density matrix, needs next.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, tau_err = minimax_points_for_gw(eps, nocc, mu=mu, target=tau_target)
        if timings is not None:
            timings['ntau_auto'] = ntau
            timings['tau_fit_error'] = tau_err

    X_mo, D, X_ao, coords = (factors if factors is not None
                             else separable_factors(mf, mol, auxbasis=auxbasis,
                                                    radii=radii))

    # find out if auto is better than just as many gauss-legendre points
    # (in reference to density trace)
    freq_points, freq_weights = find_freq_points(nocc, eps, mu, ntau, e_min, e_max, w0)

    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)
    _, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau_points = 0.5 * minimax_time_grid(ntau, *rS)[0]

    print(f'cos: {grid.cosft_tw.shape}',f'sin: {grid.sinft_tw.shape}')

    chi0 = chi0_imaginary_frequency(X_mo, D, eps, nocc, grid, mu=mu)
    eye = np.eye(chi0.shape[-1])
    for k in range(chi0.shape[0]):
        chi0[k] = np.linalg.inv(eye - chi0[k])
    W_omega = chi0

    # omega_out = freq_points, not a Pade grid: Sigma comes back on the SAME
    # axis W lives on, so G built from it stays on that axis too.
    sigma_ao = self_energy_matrix_imaginary_time(
        X_ao, D, W_omega, mf.mo_coeff, eps, nocc,
        tau_points, freq_points, freq_points, mu=mu)
    sigma_mo = sigma_ao_to_mo(sigma_ao, mf.mo_coeff)

    return sigma_mo, eps, mu, freq_points, freq_weights, X_mo, D, grid


def dyson_green_function(sigma_mo, eps, mu, freq_points):
    """G(i.omega) = [(i.omega + mu) I - diag(eps) - Sigma(i.omega)]^-1.

    The matrix Dyson equation, closed pointwise per frequency point -- the
    same structure as the W = [I - chi0]^-1 inversion `sigma_space_time`
    already does for the screened interaction, with the analytic
    (i.omega + mu) I - diag(eps) standing in for G0^-1 (MO basis, so S = I
    and F = diag(eps)) instead of the identity chi0 is inverted against.

    sigma_mo: (nfreq, nmo, nmo) complex, on `freq_points`. Returns G, same
    shape.
    """
    nmo = len(eps)
    eye = np.eye(nmo)
    diag_eps = np.diag(eps)
    G = np.empty_like(sigma_mo)

    for k, w in enumerate(freq_points):
        g0_inv = (1j * w + mu) * eye - diag_eps
        G[k] = np.linalg.inv(g0_inv - sigma_mo[k])
    return G


def density_matrix_scgw(G, freq_weights):
    """gamma_pq = (1/2) delta_pq + (1/pi) sum_k w_k Re[G_pq(i.omega_k)].

    The T=0 density matrix from a Matsubara Green's function, as a frequency
    integral over the SAME [0, infty) grid G was built on -- `freq_weights`
    must be the quadrature paired with the `freq_points` `G` lives on
    (`sigma_matrix_scgw`'s third return value), not an independent choice.

    Exact for the bare G0: Re[G0_pp(i.omega)] = -(eps_p-mu)/(omega^2+(eps_p-mu)^2)
    integrates to +-1/2, so gamma_pp comes out exactly 1 (occupied) or 0
    (virtual) -- the mean-field occupations, recovered without ever touching
    eps directly. For the dressed G this is only as good as the quadrature:
    Tr(gamma) should reproduce the electron count, and that check is NOT
    automatic just because the same grid integrates chi0/W well.

    G: (nfreq, nmo, nmo) complex, on `freq_points`. Returns gamma, (nmo, nmo)
    real, in the MO basis G was built in (mf.mo_coeff @ gamma @ mf.mo_coeff.T
    for the AO density).
    """
    nmo = G.shape[-1]
    averaged_term = (1.0 / np.pi) * np.einsum('k,kpq->pq', freq_weights, G.real)

    # with np.printoptions(precision=5, suppress=True, linewidth=120):
    #     print('(1/pi) sum_k w_k Re[G(i.omega_k)]:')
    #     print(averaged_term + 0.5*np.eye(nmo))

    return 0.5 * np.eye(nmo) + averaged_term

def density_matrix_scgw_split(G, G_hf, F, mu, freq_weights):
    """Method for calculating the desity like
    Grumet, Liu, Kaltak, Klimes, Kresse, Phys. Rev. B 98 , 155143 (2018)

    Should be more accurate than density_matrix_scgw
    since the gamma_hf part is solved analytically, without quadrature error

    F: the Fock matrix G_hf was built from. gamma_hf is diagonal in F's OWN
    eigenbasis, not necessarily the caller's original orbital basis -- once
    F is rebuilt from a scGW density (mu_cycle), it is no longer diagonal
    there, so its eigenvalues/eigenvectors (not a fixed `eps`) are what
    decide occupation. Reduces exactly to the old eps-based version when
    F = diag(eps): eigh of a diagonal matrix returns eps itself, in order,
    with the identity as eigenvectors.
    """

    eps_F, U = np.linalg.eigh(F)
    occ_F = (eps_F < mu).astype(float)
    gamma_hf = U @ np.diag(occ_F) @ U.T

    G_c = G - G_hf

    gamma_c = (1/np.pi)*np.einsum('k,kpq->pq', freq_weights, G_c.real)

    # print(np.trace(gamma_c+gamma_hf))

    return gamma_c + gamma_hf

def density_matrix_ir(G, grid_f):
    """gamma = -G_greater(tau -> beta), read directly off the tau-domain
    representation on the fermion IR grid.

    Exact, not a quadrature approximation: the fermion antiperiodic boundary
    condition G(tau-beta) = -G(tau), combined with the equal-time identity
    G(0-) = gamma, gives G(tau -> beta-) = -gamma. This is the analogue of
    density_matrix_scgw's frequency-integral, but IR's true two-sided basis
    lets it be evaluated exactly at the tau = beta boundary via
    grid_f.project_tau, rather than approximated by a [0, infty) quadrature.

    Cross-checked two ways on the bare limit: -G_greater(beta) and
    I - G_lesser(beta) agree with each other and with the exact bare density
    to ~4e-6 (H2/cc-pVDZ) -- G_lesser(tau) = Go(tau) + Gv(beta-tau) and
    G_greater(tau) = -[Gv(tau) + Go(beta-tau)] both reduce their mirror term
    to the full occupied/virtual projector exactly at tau = beta, since the
    non-mirror term (Go(beta) or Gv(beta)) is exponentially suppressed there.

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    """
    G_even = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)

    ntau, nmo = G_even.shape[0], G_even.shape[1]
    G_even_flat = G_even.reshape(ntau, nmo * nmo).T
    G_odd_flat = G_odd.reshape(ntau, nmo * nmo).T

    beta = grid_f.meta['beta']
    tau_beta = np.array([beta])
    G_even_beta = grid_f.project_tau(G_even_flat, tau_beta, parity='even').reshape(nmo, nmo)
    G_odd_beta = grid_f.project_tau(G_odd_flat, tau_beta, parity='odd').reshape(nmo, nmo)

    G_greater_beta = G_even_beta + G_odd_beta
    return -G_greater_beta

def build_new_F(gamma, V, h_mo):
    J_ij = np.einsum('ijkl,kl->ij', V, gamma)
    K_ij = np.einsum('ikjl,kl->ij', V, gamma)

    return h_mo + 2* J_ij - K_ij

def build_G_from_F(mu, freq_points, sigma_mo, F, return_G_hf=False):

    G = np.empty_like(sigma_mo)
    nmo = G.shape[-1]
    eye = np.eye(nmo)


    # IT MIGHT BE THAT THIS FORMULA ACTUALLY REQUIERES USING + sigma_mo[k]
    # INSTEAD OF -, BECAUSE THE SIGMA MIGHT HAVE BEEN CALCULATED AS iGW 
    # INSTEAD OF -iGW
    if not return_G_hf:
        for k, w in enumerate(freq_points):
            G[k] = np.linalg.inv((1j* w + mu)*eye - F - sigma_mo[k])
        return G
    
    else:
        G_hf = np.empty_like(sigma_mo)
        for k, w in enumerate(freq_points):
            G_hf_inv = (1j* w + mu)*eye - F 
            G[k] = np.linalg.inv(G_hf_inv - sigma_mo[k])
            G_hf[k] = np.linalg.inv((G_hf_inv))
        return G, G_hf

def _mu_cycle_find_trace(mu, F, freq_points, freq_weights, sigma_mo, grid_f=None, grid_kind='IR',
                         densmethod='full'):
    """One gamma(mu) evaluation at FIXED F. Returns (trace, gamma, G).

    F is held fixed for the whole mu bisection -- only mu moves -- so this
    is the inner "2.a Find mu" loop, which only ever sees a single, frozen
    set of poles. F is rebuilt from the mu-converged gamma exactly once per
    outer "Build G" iteration (by mu_cycle's caller, as "2.b Build F"), not
    once per trial mu.
    """
    if grid_kind.upper() == 'IR':
        G = build_G_from_F(mu, grid_f.omega_points, sigma_mo, F)
        gamma = density_matrix_ir(G, grid_f)
    elif densmethod.lower() == 'split':
        G, G_hf = build_G_from_F(mu, freq_points, sigma_mo, F, return_G_hf=True)
        gamma = density_matrix_scgw_split(G, G_hf, F, mu, freq_weights)
    else:
        G = build_G_from_F(mu, freq_points, sigma_mo, F)
        gamma = density_matrix_scgw(G, freq_weights)
    return np.trace(gamma), gamma, G

def mu_cycle(mu, G, gamma, nocc, freq_points, freq_weights, sigma_mo, V, h_mo,
            mu_step=0.000001, trace_tol=1e-5, mu_tol=1e-10,
            max_bracket_iter=50, max_bisect_iter=100, grid_f=None, grid_kind='IR', densmethod='full',
            stall_trace_tol=1e-4):
    """Bisection search for mu with Tr[gamma(mu)] = nocc, at FIXED F.

    Sigma (sigma_mo) AND F are both held fixed for the whole call: F is
    built once, from the incoming gamma, before the search starts, and
    never rebuilt from the trial gammas the bisection generates along the
    way. This is the inner "2.a Find mu" loop; it is the caller's job to
    rebuild F from the converged gamma ("2.b Build F") once this returns,
    before deciding whether another outer "Build G" iteration is needed.

    With F fixed, Tr[gamma(mu)] is the ordinary single-particle counting
    function of mu (monotonically non-decreasing: raising mu can only pull
    more poles below the Fermi level), which is what makes the direction
    convention below ("too many electrons -> lower mu") and plain bisection
    valid in the first place. Rebuilding F from gamma at every trial mu (the
    previous implementation) let the poles of G move during the search
    itself, which is what produced the non-monotonic, discontinuous
    trace(mu) seen on the IR route.

    Two phases: an expanding-step search to bracket the root (step grows
    while still hunting for the sign flip), then ordinary bisection, whose
    bracket width halves every iteration on its own.

    densmethod: passed straight through to _mu_cycle_find_trace at every
    trial mu -- 'split' builds G_hf alongside G and uses
    density_matrix_scgw_split; 'full' (default) uses the plain
    frequency-integral density_matrix_scgw.

    stall_trace_tol: the bracket can collapse below mu_tol before
    |trace-nocc| < trace_tol -- a numerical noise floor right at the root,
    harmless if |trace-nocc| is still small (< stall_trace_tol). Silently
    returning that as "converged" previously fed a wildly wrong gamma/F into
    the next scGW iteration. Now it raises instead.

    Returns (mu, G, gamma, F), where F is the single fixed F used
    throughout this call.
    """
    F = build_new_F(gamma, V, h_mo)
    trace_old, gamma, G = _mu_cycle_find_trace(
        mu, F, freq_points, freq_weights, sigma_mo, grid_f=grid_f, grid_kind=grid_kind,
        densmethod=densmethod)

    # print(f'Trace before mu-cycle: {trace_old}')

    if abs(trace_old - nocc) < trace_tol:
        print('mu-cycle already converged within trace tolerance')
        return mu, G, gamma, F

    # --- Phase 1: bracket the root ---
    step = mu_step
    direction = -1.0 if trace_old > nocc else 1.0 #original
    # direction = 1.0 if trace_old > nocc else -1.0
    mu_a, trace_a = mu, trace_old

    for _ in range(max_bracket_iter):
        mu_b = mu_a + direction * step
        trace_b, gamma_b, G_b = _mu_cycle_find_trace(
            mu_b, F, freq_points, freq_weights, sigma_mo, grid_f=grid_f, grid_kind=grid_kind,
            densmethod=densmethod)

        if (trace_a - nocc) * (trace_b - nocc) <= 0:
            break
        mu_a, trace_a = mu_b, trace_b
        step *= 1.5
    else:
        raise RuntimeError(f'mu_cycle: failed to bracket nocc={nocc} '
                           f'within {max_bracket_iter} probes')

    # --- Phase 2: bisection ---
    mu_lo, mu_hi, trace_lo = mu_a, mu_b, trace_a
    gamma_mid, G_mid = gamma_b, G_b
    for _ in range(max_bisect_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        trace_mid, gamma_mid, G_mid = _mu_cycle_find_trace(
            mu_mid, F, freq_points, freq_weights, sigma_mo, grid_f=grid_f, grid_kind=grid_kind,
            densmethod=densmethod)

        if abs(trace_mid - nocc) < trace_tol:
            print(f'mu_cycle converged: mu={mu_mid}, trace={trace_mid}')
            return mu_mid, G_mid, gamma_mid, F

        if abs(mu_hi - mu_lo) < mu_tol:
            if abs(trace_mid - nocc) < stall_trace_tol:
                print(f'mu_cycle converged (bracket collapsed, trace within '
                     f'stall_trace_tol): mu={mu_mid}, trace={trace_mid}')
                return mu_mid, G_mid, gamma_mid, F
            raise RuntimeError(
                f'mu_cycle: bisection bracket collapsed to width '
                f'{abs(mu_hi - mu_lo):.2e} (< mu_tol={mu_tol:.1e}) with '
                f'|trace-nocc| = {abs(trace_mid - nocc):.2e} at mu={mu_mid} -- '
                f'nowhere near trace_tol={trace_tol:.1e} or '
                f'stall_trace_tol={stall_trace_tol:.1e}.')

        if (trace_lo - nocc) * (trace_mid - nocc) <= 0:
            mu_hi = mu_mid
        else:
            mu_lo, trace_lo = mu_mid, trace_mid

    raise RuntimeError(f'mu_cycle: did not converge in {max_bisect_iter} '
                       f'iterations (|trace-nocc| = {abs(trace_mid - nocc):.2e})')

def calc_new_sigma(G, freq_points, freq_weights, tau_points, nocc, F, mu, X_mo, D, grid):
    """G(i.omega) -> the next iteration's Sigma_c(i.omega), via the dressed
    polarizability chi, the Dyson-inverted W, and Sigma = i G W as a
    pointwise product in imaginary time -- the same construction as
    chi0_imaginary_frequency/polarizability_imaginary_time and
    self_energy_matrix_imaginary_time, generalized from the bare-orbital G0
    to an arbitrary dressed G. Returns (G_tau, sigma_new); sigma_new is in
    MO basis, on the SAME freq_points G came in on, ready for the next
    build_G_from_F call.

    The cosine/sine transform weights are fit to F's OWN pole range, not
    chi0/Sigma's transition-energy range (e_min/e_max) -- reusing that range
    silently gives wrong results here (checked: ~0.29 vs ~0.0015 max error
    against the exact bare-G0 reference), since F is generally no longer
    diagonal in the original eps once it is rebuilt from a scGW density.

    Pi_PQ = Ghat_greater_PQ * Ghat_lesser_PQ (elementwise) is only meaningful
    in the real-space/ISDF (M, M) interpolation-point basis, NOT the MO
    basis G_lesser/G_greater come out of the transform in -- an elementwise
    product is not basis-covariant. X_mo projects each branch onto that
    basis first, exactly like the bare X_o/X_v collocation matrices do for
    the same purpose in polarizability_imaginary_time; D then projects the
    result on to the auxiliary basis.

    The prefactor here is +2.0, not the -2.0 polarizability_projected_tau
    uses on its Go*Gv: Gv = -G_greater (greens_function_imaginary_time puts
    an explicit minus sign on the virtual/greater branch that the bare
    Go*Gv product does not carry), so Go*Gv = G_lesser*(-G_greater), and
    that -1 flips polarizability_projected_tau's -2.0 into +2.0 here.
    Checked against chi0_imaginary_frequency in the bare-G0 limit: -2.0 was
    off by O(1) (not the ~1e-3 level the G-transform's own fit error would
    predict) and shrank the WRONG way with a more accurate ntau; +2.0 lines
    up with chi0_imaginary_frequency to 2e-3 at ntau=20 (the ntau=16 default
    here is right at the edge of accurate for this comparison specifically).
    """
    eps_F, _ = np.linalg.eigh(F)
    dG = np.abs(eps_F - mu)
    rG = (0.3 * dG.min(), 3.0 * dG.max())

    Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points, *rG, warn=False)
    Stw, _ = minimax_transform_weights(SINE_WT, tau_points, freq_points, *rG, warn=False)

    G_even = np.einsum('tk,kpq->tpq', Ctw, G.real)
    G_odd = np.einsum('tk,kpq->tpq', Stw, G.imag)

    G_lesser = G_even - G_odd
    G_greater = G_even + G_odd

    # MO basis -> ISDF/THC interpolation-point basis, per tau point.
    Ghat_lesser = np.einsum('pi,tij,qj->tpq', X_mo, G_lesser, X_mo, optimize=True)
    Ghat_greater = np.einsum('pi,tij,qj->tpq', X_mo, G_greater, X_mo, optimize=True)

    # F's own transition-energy range (generalized off F's eigenvalues
    # instead of the bare eps/occ/virt split, since F is no longer diagonal
    # in the original orbital basis once rebuilt from a scGW density).
    occ_mask = eps_F < mu
    e_min_F = eps_F[~occ_mask].min() - eps_F[occ_mask].max()
    e_max_F = eps_F[~occ_mask].max() - eps_F[occ_mask].min()

    grid_chi = TimeFrequencyGrid.minimax_split(len(tau_points), e_min_F, e_max_F,
                                           freq_points, freq_weights)

    chi_omega = chi0_imaginary_frequency(X_mo, D, eps_F, nocc, grid_chi, Go=Ghat_greater, Gv=-Ghat_lesser)

    # Dyson invert chi(i.omega) -> W(i.omega), same pointwise pattern as the
    # chi0 -> W inversion elsewhere.

    naux = chi_omega.shape[-1]
    eye_aux = np.eye(naux)
    W_omega = np.empty_like(chi_omega)
    print(eye_aux.shape, W_omega.shape)
    for k in range(chi_omega.shape[0]):
        W_omega[k] = np.linalg.inv(eye_aux - chi_omega[k])

    # omega -> tau: only the correlation part (W - I) decays and needs
    # transforming; W is real/even like chi, so cosine only. Padded, like
    # self_energy_fit_ranges' rW -- the screened interaction has weight
    # BELOW the smallest transition, so an unpadded range misfits exactly
    # where W is largest.

    rW_F = (0.3 * e_min_F, 3.0 * e_max_F)
    Ctw_W, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points, *rW_F, warn=False)
    Wt_tau = np.einsum('tk,kab->tab', Ctw_W, W_omega - eye_aux, optimize=True)

    # aux basis -> ISDF/THC grid, per tau point: Zt(tau) = D Wt(tau) D^T.

    Zt_tau = np.einsum('Pa,tab,Qb->tPQ', D, Wt_tau, D, optimize=True)

    # Sigma = i G W, pointwise product in imaginary time, THC grid. Unlike
    # the Pi step, NO extra sign flip is needed here: self_energy_matrix_
    # imaginary_time builds the raw (no-minus) virtual-branch product and
    # explicitly SUBTRACTS it into its greater accumulator, which is exactly
    # equivalent to using G_greater (with its own built-in minus) directly
    # with a plain +, as done here.


    sig_lesser_thc = Zt_tau * Ghat_lesser
    sig_greater_thc = Zt_tau * Ghat_greater

    # THC grid -> MO basis directly (X_mo for both indices -- Sigma comes
    # back in MO basis, no AO round trip needed, per self_energy_matrix_
    # imaginary_time's own documented X_ao/X_mo flexibility).

    sig_lesser = np.einsum('pi,tpq,qj->tij', X_mo, sig_lesser_thc, X_mo, optimize=True)
    sig_greater = np.einsum('pi,tpq,qj->tij', X_mo, sig_greater_thc, X_mo, optimize=True)

    even = sig_greater + sig_lesser
    odd = sig_greater - sig_lesser

    # tau -> omega: Sigma's own (padded) range, combining G's per-orbital
    # decay (dG) with W's transition range -- Sigma = G*W is a PRODUCT, so
    # its decay rates are SUMS of the two factors' own (self_energy_fit_
    # ranges' rS, generalized off F instead of the bare eps).

    rS_F = (0.3 * (dG.min() + e_min_F), 3.0 * (dG.max() + e_max_F))
    C, _ = minimax_transform_weights(COSINE_TW, tau_points, freq_points, *rS_F, warn=False)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, freq_points, *rS_F, warn=False)
    sigma_new = (np.einsum('wt,tij->wij', C, even, optimize=True)
                + 1j * np.einsum('wt,tij->wij', S, odd, optimize=True))

    # Sigma = -0.5 * (i G W), matching self_energy_matrix_imaginary_time's
    # own "out *= -0.5" (imaginary_time.py) -- checked directly: -0.5*sigma_new
    # (this function's raw, un-prefactored product) matches that already-
    # tested route's own Sigma to 8e-09 for the same G and W, on the bare-G0
    # limit. The lesser/greater branch signs alone (both routines build
    # G_greater with Gv's own built-in minus, both combine even=greater+lesser,
    # odd=greater-lesser) are NOT the whole story -- that trailing -0.5 is a
    # separate, otherwise easy to miss, overall normalization.
    return (G_lesser + 1j * G_greater), chi_omega, W_omega, -0.5 * sigma_new

def project_tau_cross(src_grid, dst_grid, f_src, dim, parity):
    """Move a quantity known on src_grid's own tau_points onto dst_grid's --
    src_grid and dst_grid are two IR grids built at the SAME beta but
    different statistics (fermion vs boson), so their native tau_points
    differ, and a pointwise product like chi = G_greater*G_lesser or
    Sigma = i*G*W needs every factor at the SAME tau before it can be taken.

    f_src: (ntau_src, dim, dim), on src_grid.tau_points.
    Returns (ntau_dst, dim, dim), on dst_grid.tau_points.
    """
    ntau_src = len(src_grid.tau_points)
    flat = f_src.reshape(ntau_src, dim * dim).T               # (dim*dim, ntau_src)
    return src_grid.project_tau(flat, dst_grid.tau_points, parity=parity).T.reshape(-1, dim, dim)

def calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D):
    """IR analogue of calc_new_sigma: ONE fixed (beta, omega_max) pair, baked
    into grid_f/grid_b when they were built, serves every stage here -- no
    per-call minimax refit to rG/rW_F/rS_F. That is the entire point of
    using IR grids.

    G and W live on DIFFERENT grids (fermion carries different basis
    functions/tau_points than boson), so moving a quantity from one grid's
    tau axis onto the other's, for the Sigma=GW pointwise product, goes
    through grid.project_tau -- the two-sided ("fit here, evaluate there")
    capability a minimax fit does not have.

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    Returns (G_tau, chi_omega, W_omega, sigma_new); G_tau is
    (G_lesser + 1j*G_greater) on grid_f.tau_points, same packing as
    calc_new_sigma. sigma_new is on grid_f.omega_points.
    """
    nmo = G.shape[-1]

    # G(i.omega) -> G(tau), fermion-native, sectors kept separate so each
    # can be projected onto grid_b's tau below.
    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)

    # Move both sectors onto the boson grid's own tau -- chi/W live there,
    # and the chi Hadamard product needs both factors at the SAME tau.
    G_even_b = project_tau_cross(grid_f, grid_b, G_even_f, nmo, 'even')
    G_odd_b = project_tau_cross(grid_f, grid_b, G_odd_f, nmo, 'odd')
    G_lesser_b = G_even_b - G_odd_b
    G_greater_b = G_even_b + G_odd_b

    # MO -> ISDF/THC grid, on the boson grid's tau.
    Ghat_lesser_b = np.einsum('pi,tij,qj->tpq', X_mo, G_lesser_b, X_mo, optimize=True)
    Ghat_greater_b = np.einsum('pi,tij,qj->tpq', X_mo, G_greater_b, X_mo, optimize=True)

    # chi(i.omega) on the boson grid -- same Go=Ghat_greater, Gv=-Ghat_lesser
    # convention as the minimax route (Gv = -G_greater, greens_function_
    # imaginary_time's built-in minus on the virtual/greater branch); eps/nocc
    # are unused when Go/Gv are given (chi0_imaginary_frequency's early
    # return), so None stands in for them here.
    chi_omega = chi0_imaginary_frequency(X_mo, D, None, None, grid_b,
                                         Go=Ghat_greater_b, Gv=-Ghat_lesser_b)

    # Dyson invert chi -> W, same pointwise pattern as everywhere else.
    naux = chi_omega.shape[-1]
    eye_aux = np.eye(naux)
    W_omega = np.empty_like(chi_omega)
    for k in range(chi_omega.shape[0]):
        W_omega[k] = np.linalg.inv(eye_aux - chi_omega[k])

    # W(i.omega) -> W(tau), boson-native. Only the correlation part (W - I)
    # decays; W is real/even like chi, so cosine (even sector) only.
    Wt_tau_b = np.einsum('tk,kab->tab', grid_b.cosft_tw, W_omega - eye_aux, optimize=True)

    # Move Wt from the boson grid's tau onto the fermion grid's tau -- the
    # Sigma=GW product needs W at the SAME tau as G (fermion-native here).
    Wt_tau_f = project_tau_cross(grid_b, grid_f, Wt_tau_b, naux, 'even')

    # aux basis -> ISDF/THC grid, per tau point, now fermion-native tau.
    Zt_tau = np.einsum('Pa,tab,Qb->tPQ', D, Wt_tau_f, D, optimize=True)

    # Ghat_lesser/Ghat_greater on the FERMION grid's own tau, for Sigma=GW --
    # built directly from G_even_f/G_odd_f already in hand, not by projecting
    # the grid_b versions back (which would add an extra round trip).
    Ghat_lesser_f = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f - G_odd_f, X_mo, optimize=True)
    Ghat_greater_f = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f + G_odd_f, X_mo, optimize=True)

    # Sigma = i G W, pointwise product in imaginary time, THC grid. No extra
    # sign flip here, same reasoning as calc_new_sigma's own docstring.
    sig_lesser_thc = Zt_tau * Ghat_lesser_f
    sig_greater_thc = Zt_tau * Ghat_greater_f

    sig_lesser = np.einsum('pi,tpq,qj->tij', X_mo, sig_lesser_thc, X_mo, optimize=True)
    sig_greater = np.einsum('pi,tpq,qj->tij', X_mo, sig_greater_thc, X_mo, optimize=True)

    even = sig_greater + sig_lesser
    odd = sig_greater - sig_lesser

    # Sigma(tau) -> Sigma(i.omega), fermion-native frequencies.
    sigma_new = (np.einsum('wt,tij->wij', grid_f.cosft_wt, even, optimize=True)
                + 1j * np.einsum('wt,tij->wij', grid_f.sinft_wt, odd, optimize=True))

    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    # Sigma = -0.5 * (i G W), matching self_energy_matrix_imaginary_time's
    # own trailing "out *= -0.5" (imaginary_time.py) -- checked directly:
    # -0.5*sigma_new (this function's raw, un-prefactored product) matches
    # that already-tested route's own Sigma to 8e-09 for the same G and W,
    # in the bare-G0 limit. The lesser/greater branch signs matching (both
    # routines build G_greater with Gv's own built-in minus, both combine
    # even=greater+lesser, odd=greater-lesser) is NOT the whole story --
    # that trailing -0.5 is a separate overall normalization.
    return (G_lesser_f + 1j * G_greater_f), chi_omega, W_omega, -0.5 * sigma_new

def solve_qp_energy_scgw(mf, mol, nocc, skip_scf=False, grid_kind='IR', densmethod='split', **kwargs):
    """One scGW iteration: the dressed Green's function via Dyson.

    In the future going to run the sc cycle

    `sigma_matrix_scgw` builds Sigma_c(i.omega) as a full matrix on chi0/W's
    own frequency grid; this closes the Dyson equation pointwise per
    frequency. See the module docstring for what is still missing to turn
    this into an actual self-consistency loop (the i.omega -> i.tau
    transform of G and a dressed replacement for `chi0_imaginary_frequency`).

    Returns (G, sigma_mo, eps, mu, freq_points, freq_weights, F_new). F_new is
    h_mo + J[gamma] - 0.5*K[gamma] (see `build_new_F`), the updated Fock
    matrix in mf.mo_coeff's basis built from the scGW density `gamma`.
    """

    if grid_kind.upper() == 'IR':
        # Step 1: build the two IR grids (fermion for G/Sigma, boson for
        # chi/W -- they carry different basis functions and different
        # tau_points, never one grid for both), gate on their own round-trip
        # fidelity BEFORE trusting them for anything, then build the bare G0
        # analytically, directly on the fermion grid's own Matsubara
        # frequencies -- no minimax freq_points, no chi0-from-bare-orbitals
        # bootstrap needed at all.
        eps = get_orbital_energies(mf, representation='spatial')
        print(eps[:nocc],'\n',eps[nocc:])
        occ, virt = get_occ_virt_indices(eps, nocc)
        mu = 0.5 * (eps[nocc - 1] + eps[nocc])
        F_bare = np.diag(eps)

        # omega_max: the widest range calc_new_sigma's minimax route ever
        # needed (rS_F's own logic -- G's decay plus W's, padded) -- ONE
        # choice has to cover every stage here, since there is no per-stage
        # refit later the way minimax has.
        dG = np.abs(eps - mu)
        e_min = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        omega_max = 3.0 * (dG.max() + e_max)

        # beta: large enough that the smallest gap is in the T = 0 regime
        # (thermal occupation exp(-beta * dG.min()) = exp(-25) ~ 1e-11), and
        # no larger. Lambda = beta * omega_max sets the conditioning of the
        # omega -> tau fit, not just the basis size: the Matsubara sampling
        # barely reaches past omega_max at large Lambda, so the 1/(i.omega)
        # tail that fixes G(tau -> 0, beta) -- the density -- is extrapolated
        # rather than sampled. Measured on Ne/cc-pVDZ: 100/dG.min() (Lambda
        # 1.7e4, odd-sector cond 4e5) lets the G -> Sigma -> G iteration blow
        # up within ~5 iterations; 25/dG.min() (Lambda 4e3, cond 4e4) is stable.
        beta = 100.0 / dG.min()

        # Matsubara search window: default_matsubara_sampling only offers
        # candidate frequencies up to
        #     omega_top / omega_max ~ 1 + factor * size * 2 pi / Lambda,
        # which at the default factor = 4 is 1.1 - 1.4 x omega_max -- the
        # extrema of the highest basis functions above that are never
        # sampled, and the omega -> tau fit is ill-conditioned in exactly the
        # directions that fix G(tau -> 0, beta). Measured on Ne/cc-pVDZ at
        # Lambda = 1.7e4: reach 1.12 -> odd-sector cond 4e5 and a divergent
        # G -> Sigma -> G iteration; reach 5.4 -> cond 2e2, stable. Solve the
        # relation above for the factor that gives `reach`; size_guess is an
        # upper estimate of the IR size (60 - 100 here), so the true reach
        # comes out somewhat below `reach` rather than above it.
        reach = 6.0
        size_guess = 100
        n_matsubara_factor = int(np.ceil(
            (reach - 1.0) * beta * omega_max / (2 * np.pi * size_guess)))

        grid_f = TimeFrequencyGrid.ir(beta, omega_max, statistics='fermion',
                                      n_matsubara_factor=n_matsubara_factor)
        
        grid_b = TimeFrequencyGrid.ir(beta, omega_max, statistics='boson',
                                      n_matsubara_factor=n_matsubara_factor)

        # print(dir(grid_f))

        print(f'IR grids: Lambda={beta * omega_max:.4g}, '
             f'fermion size={grid_f.meta["basis"].size}, '
             f'boson size={grid_b.meta["basis"].size}, '
             f'n_matsubara_factor={n_matsubara_factor}, '
             f'highest sampled omega = '
             f'{grid_f.omega_points.max() / omega_max:.2f} x omega_max')
        # duality_error, not roundtrip_error: the latter's probe is the
        # bosonic particle-hole bubble (periodic under tau -> beta-tau),
        # which is the right check for the boson/chi grid but not a valid
        # probe for the fermion/G grid (antiperiodic) -- duality_error tests
        # each grid's own represented subspace directly and is what the
        # class docstring itself says to trust within IR.
        print(f'fermion duality_error (even/odd) = '
             f'{grid_f.duality_error("even"):.2e} / {grid_f.duality_error("odd"):.2e}')
        print(f'boson   duality_error (even/odd) = '
             f'{grid_b.duality_error("even"):.2e} / {grid_b.duality_error("odd"):.2e}')

        X_mo, D, X_ao, coords = (kwargs.get('factors') if kwargs.get('factors') is not None
                                 else separable_factors(mf, mol, auxbasis=kwargs.get('auxbasis')))

        freq_points = grid_f.omega_points
        nmo = len(eps)
        eye = np.eye(nmo)
        G0 = np.array([np.linalg.inv((1j * w + mu) * eye - F_bare)
                       for w in freq_points])

        print(f'G0 (IR, bare): shape={G0.shape}, max|G0|={np.max(np.abs(G0)):.4e}')

        gamma = density_matrix_ir(G0, grid_f)
        print(f'Trace gamma_G0: {np.trace(gamma)}, expected nocc: {nocc}')

        G_tau, chi_omega, W_omega, sigma_new = calc_new_sigma_ir(
            G0, grid_f, grid_b, X_mo, D)

        h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
        V = get_two_electron_integrals_chemist(mol,mf)

        F = build_new_F(gamma, V, h_mo)
        G = build_G_from_F(mu, grid_f.omega_points, sigma_new, F)

        if skip_scf:
            return G, F, G0, mu, V, h_mo, grid_f, grid_b, X_mo, D

        delta_gamma = 999
        gamma_convergence = 1e-6

        alpha = 0.6

        gamma = density_matrix_ir(G, grid_f)
        _, _, _, sigma_new = calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D)

        while delta_gamma > gamma_convergence:
            gamma_old = gamma
            sigma_old = sigma_new
            mu_old = mu

            _, _, _, sigma_new = calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D)

            delta_sigma = np.max(np.abs(sigma_old - sigma_new))

            mu_pre_scf = mu


            delta_mu = 999
            mu_convergence = 1e-3

            while delta_mu > mu_convergence:
                mu_old = mu

                mu, G, gamma, F = mu_cycle(mu, G, gamma, nocc, freq_points=None, freq_weights=None,
                                sigma_mo=sigma_new, V=V, h_mo=h_mo, grid_f=grid_f, grid_kind='IR')

                delta_mu = abs(mu_old - mu)

            print(f'Mu post mu_scf: {mu}')

            gamma = alpha * gamma + (1 - alpha) * gamma_old

            delta_mu = abs(mu_old - mu)

            delta_gamma = np.max(np.abs(gamma - gamma_old))
            delta_mu = np.abs(mu - mu_old)
            print(f'Delta gamma: {delta_gamma}\nDelta mu: {delta_mu}\nDelta Sigma: {delta_sigma}\n\n')


        return G, F, G0, mu_pre_scf, gamma, sigma_new, V, h_mo, grid_f, grid_b, X_mo, D

        # This mu-cycle has to be fixed / implemented properly still
        # The mu-cycle does not converge, may be because the initial G0 density is so close to n_occ
        #


        # with np.printoptions(precision=3, suppress=False, linewidth=120):
        #     print(gamma)

        # calc_new_sigma_ir (dressed chi/W/Sigma from grid_f/grid_b) and the
        # IR density (gamma = -G(tau=beta)) are not built yet -- this is
        # only the grid setup and the analytic bare G0, per step 1.
        raise NotImplementedError(
            'IR route: grids and bare G0 built above; calc_new_sigma_ir and '
            'the IR density extraction are not implemented yet.')

    # from here the non-ir route, currently not continuing implementation
    sigma_mo, eps, mu, freq_points, freq_weights, X_mo, D, grid = sigma_matrix_scgw_iteration1(
        mf, mol, nocc, **kwargs)

    G = dyson_green_function(sigma_mo, eps, mu, freq_points)

    if densmethod.lower()=='split':
        G0 = dyson_green_function(np.zeros_like(sigma_mo), eps, mu, freq_points)
        gamma = density_matrix_scgw_split(G, G0, np.diag(eps), mu, freq_weights)

    elif densmethod.lower()=='full':
        gamma = density_matrix_scgw(G, freq_weights)

    else:
        raise KeyError('densemethod implemented "full" or "split"')

    #here
    print('\nDIAGNOSTICS\n')
    h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
    V = get_two_electron_integrals_chemist(mol,mf)

    # print(f'Testing mu-cycle: mu:{mu}, tr_gamma:{np.trace(gamma)}, expected 1')
    mu, G, gamma, F = mu_cycle(mu, G, gamma, nocc, freq_points,
                               freq_weights, sigma_mo, V, h_mo, densmethod=densmethod)

    # Evaluate now density of G

    n_occ_gamma = np.trace(gamma)
    print(f'n_occ_gamma {n_occ_gamma}, expected {nocc}')

    G_tau, chi_omega, W_omega, sigma_new = calc_new_sigma(
        G, freq_points, freq_weights, grid.tau_points, nocc, F, mu, X_mo, D, grid)

    return G, sigma_mo, eps, mu, freq_points, freq_weights, F
