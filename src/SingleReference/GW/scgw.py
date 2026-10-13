"""Self-consistent GW (scGW) on intermediate-representation (IR) grids.

Everything is done in the MO basis of the starting mean-field calculation.
Each iteration of the cycle is:

    G(i.omega) -> G(tau) -> chi(tau) -> chi(i.omega) -> W(i.omega) -> W(tau)
               -> Sigma(tau) = G(tau) W(tau) -> Sigma(i.omega)
    F = h + 2J[gamma] - K[gamma]
    G(i.omega) = [(i.omega + mu) I - F - Sigma(i.omega)]^-1

with mu adjusted so that Tr[gamma] equals the number of occupied orbitals.

Two IR grids are used, both at the same beta and omega_max: a fermion grid
for G and Sigma, and a boson grid for chi and W. Once converged, Sigma is
continued to the real axis with Pade and the quasiparticle energies are read
off the peaks of the spectral function.

`solve_qp_energy_scgw` is the entry point.
"""
import numpy as np
import scipy.signal

from pyscf.data.elements import chemcore
from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.Base.utils.time_frequency import (TimeFrequencyGrid)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency
from src.Base.utils.analyticalContinuation import pade_continue_elementwise

# --- Convergence parameters ---
# Outer scGW cycle
GAMMA_CONVERGENCE = 1e-6    # max |gamma - gamma_old| at which the cycle stops
GAMMA_MIXING = 0.6          # fraction of the new gamma kept in each iteration
MU_CONVERGENCE = 1e-3       # max change of mu between repeated mu searches
MAX_SCF_ITER = 100          # cap on scGW iterations
MAX_MU_ITER = 50            # cap on repeated mu searches per scGW iteration

# mu search (mu_cycle)
MU_STEP = 1e-6              # first step away from the starting mu
TRACE_TOL = 1e-5            # |Tr[gamma] - nocc| at which mu is converged
MU_TOL = 1e-10              # smallest bisection bracket width
STALL_TRACE_TOL = 1e-4      # |Tr[gamma] - nocc| still accepted at that width
MAX_BRACKET_ITER = 50
MAX_BISECT_ITER = 100

def density_matrix_ir(G, grid_f):
    """Density matrix from G: gamma = -G(tau = beta).

    Follows from G(0-) = gamma and fermion antiperiodicity,
    G(tau - beta) = -G(tau). G is transformed to imaginary time and the IR
    basis is evaluated at tau = beta, so no frequency quadrature is involved.

    Checked for the bare G0 on H2/cc-pVDZ: matches the exact density to ~4e-6.

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    Returns gamma, (nmo, nmo), in the MO basis.
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
    """Closed-shell Fock matrix F = h + 2J[gamma] - K[gamma], in the MO basis.

    gamma: (nmo, nmo) density matrix per spin (occupations between 0 and 1).
    V: two-electron integrals in chemist notation, (ij|kl).
    h_mo: core Hamiltonian.
    """
    J_ij = np.einsum('ijkl,kl->ij', V, gamma)
    K_ij = np.einsum('ikjl,kl->ij', V, gamma)

    return h_mo + 2* J_ij - K_ij

def build_G_from_F(mu, freq_points, sigma_mo, F, return_G_hf=False):
    """Dyson equation on the imaginary axis:
    G(i.omega) = [(i.omega + mu) I - F - Sigma(i.omega)]^-1.

    sigma_mo: (nfreq, nmo, nmo) complex, on freq_points.
    return_G_hf: also return G_hf, the same expression without Sigma.
    """

    G = np.empty_like(sigma_mo)
    nmo = G.shape[-1]
    eye = np.eye(nmo)

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

def build_Greal_from_F(freq_points, sigma_real, F, eta):
    """Dyson equation on the real axis:
    G(omega) = [(omega + i.eta) I - F - Sigma(omega)]^-1.

    No mu here: sigma_real is expected on absolute real frequencies.
    """

    G_w = np.empty_like(sigma_real)
    eye = np.eye(F.shape[0])

    for i,w in enumerate(freq_points):
        G_w[i] = np.linalg.inv((w + 1j*eta)*eye - F - sigma_real[i])

    return G_w

def _mu_cycle_find_trace(mu, F, sigma_mo, grid_f, grid_kind='IR'):
    """Build G and gamma for one trial mu, with F and Sigma fixed.

    Returns (Tr[gamma], gamma, G).
    """
    if grid_kind.upper() == 'IR':
        G = build_G_from_F(mu, grid_f.omega_points, sigma_mo, F)
        gamma = density_matrix_ir(G, grid_f)

    return np.trace(gamma), gamma, G

def mu_cycle(mu, G, gamma, nocc, sigma_mo, V, h_mo,
            mu_step=MU_STEP, trace_tol=TRACE_TOL, mu_tol=MU_TOL,
            max_bracket_iter=MAX_BRACKET_ITER, max_bisect_iter=MAX_BISECT_ITER,
            grid_f=None, grid_kind='IR', stall_trace_tol=STALL_TRACE_TOL):
    """Find the mu for which Tr[gamma(mu)] = nocc.

    F is built once from the incoming gamma and, like Sigma, kept fixed
    during the search. Only then is Tr[gamma(mu)] monotonic in mu, which
    bisection needs; rebuilding F at every trial mu made the trace jump.
    The caller rebuilds F from the returned gamma.

    Two phases:
      1. Step away from the starting mu, growing the step by 1.5x each time,
         until the trace crosses nocc.
      2. Bisect inside that bracket.

    Converged when |trace - nocc| < trace_tol. If the bracket shrinks below
    mu_tol first, the result is still accepted when
    |trace - nocc| < stall_trace_tol; otherwise a RuntimeError is raised.

    Returns (mu, G, gamma, F).
    """
    F = build_new_F(gamma, V, h_mo)
    trace_old, gamma, G = _mu_cycle_find_trace(
        mu, F, sigma_mo, grid_f=grid_f, grid_kind=grid_kind,)

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
            mu_b, F, sigma_mo, grid_f=grid_f, grid_kind=grid_kind)

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
            mu_mid, F, sigma_mo, grid_f=grid_f, grid_kind=grid_kind)

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

def project_tau_cross(src_grid, dst_grid, f_src, dim, parity):
    """Interpolate f from src_grid's tau points onto dst_grid's tau points.

    The fermion and boson grids share beta but sample different tau points,
    and the products chi = G*G and Sigma = G*W need both factors at the same
    tau.

    f_src: (ntau_src, dim, dim), on src_grid.tau_points.
    parity: 'even' or 'odd', the symmetry sector f_src belongs to.
    Returns (ntau_dst, dim, dim), on dst_grid.tau_points.
    """
    ntau_src = len(src_grid.tau_points)
    flat = f_src.reshape(ntau_src, dim * dim).T               # (dim*dim, ntau_src)
    return src_grid.project_tau(flat, dst_grid.tau_points, parity=parity).T.reshape(-1, dim, dim)

def calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D):
    """Correlation self-energy Sigma(i.omega) from a dressed G(i.omega).

    Steps: G -> chi = G*G -> W = [I - chi]^-1 -> Sigma = G*W. The products
    are taken pointwise in imaginary time on the ISDF/THC grid; the Dyson
    inversion for W is done in frequency. G and Sigma live on the fermion
    grid, chi and W on the boson grid, and `project_tau_cross` moves
    quantities between the two.

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    X_mo, D: separable (ISDF/THC) factors from `separable_factors`.

    Returns (G_tau, chi_omega, W_omega, sigma_new):
      G_tau      G_lesser + 1j*G_greater, on grid_f.tau_points
      chi_omega  on grid_b.omega_points, auxiliary basis
      W_omega    on grid_b.omega_points, auxiliary basis
      sigma_new  on grid_f.omega_points, MO basis
    """
    nmo = G.shape[-1]

    # G(i.omega) -> G(tau) on the fermion grid, even and odd parts separately.
    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)

    # chi lives on the boson grid, so move G onto the boson tau points.
    G_even_b = project_tau_cross(grid_f, grid_b, G_even_f, nmo, 'even')
    G_odd_b = project_tau_cross(grid_f, grid_b, G_odd_f, nmo, 'odd')
    G_lesser_b = G_even_b - G_odd_b
    G_greater_b = G_even_b + G_odd_b

    # MO basis -> ISDF/THC grid.
    Ghat_lesser_b = np.einsum('pi,tij,qj->tpq', X_mo, G_lesser_b, X_mo, optimize=True)
    Ghat_greater_b = np.einsum('pi,tij,qj->tpq', X_mo, G_greater_b, X_mo, optimize=True)

    # chi(i.omega) from the dressed G. The minus on Gv matches the sign
    # convention chi0_imaginary_frequency uses for its bare Go/Gv. eps and
    # nocc are not needed when Go/Gv are passed, hence None.
    chi_omega = chi0_imaginary_frequency(X_mo, D, None, None, grid_b,
                                         Go=Ghat_greater_b, Gv=-Ghat_lesser_b)

    # W = [I - chi]^-1 at each frequency.
    naux = chi_omega.shape[-1]
    eye_aux = np.eye(naux)
    W_omega = np.empty_like(chi_omega)
    for k in range(chi_omega.shape[0]):
        W_omega[k] = np.linalg.inv(eye_aux - chi_omega[k])

    # W(i.omega) -> W(tau). Only the correlation part W - I is transformed
    # (the constant does not decay), and it is even, so cosine only.
    Wt_tau_b = np.einsum('tk,kab->tab', grid_b.cosft_tw, W_omega - eye_aux, optimize=True)

    # Sigma lives on the fermion grid, so move W onto the fermion tau points.
    Wt_tau_f = project_tau_cross(grid_b, grid_f, Wt_tau_b, naux, 'even')

    # Auxiliary basis -> ISDF/THC grid.
    Zt_tau = np.einsum('Pa,tab,Qb->tPQ', D, Wt_tau_f, D, optimize=True)

    # G on the ISDF/THC grid at the fermion tau points, straight from the
    # fermion-grid G rather than projecting the boson-grid copy back.
    Ghat_lesser_f = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f - G_odd_f, X_mo, optimize=True)
    Ghat_greater_f = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f + G_odd_f, X_mo, optimize=True)

    # Sigma(tau) = G(tau) * W(tau), elementwise on the ISDF/THC grid. Unlike
    # chi, no extra sign is needed here.
    sig_lesser_thc = Zt_tau * Ghat_lesser_f
    sig_greater_thc = Zt_tau * Ghat_greater_f

    # ISDF/THC grid -> MO basis.
    sig_lesser = np.einsum('pi,tpq,qj->tij', X_mo, sig_lesser_thc, X_mo, optimize=True)
    sig_greater = np.einsum('pi,tpq,qj->tij', X_mo, sig_greater_thc, X_mo, optimize=True)

    even = sig_greater + sig_lesser
    odd = sig_greater - sig_lesser

    # Sigma(tau) -> Sigma(i.omega) on the fermion grid.
    sigma_new = (np.einsum('wt,tij->wij', grid_f.cosft_wt, even, optimize=True)
                + 1j * np.einsum('wt,tij->wij', grid_f.sinft_wt, odd, optimize=True))

    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    # Overall prefactor -0.5, the same normalization as
    # self_energy_matrix_imaginary_time (imaginary_time.py). Checked against
    # that routine for the bare G0: agreement to 8e-9.
    return (G_lesser_f + 1j * G_greater_f), chi_omega, W_omega, -0.5 * sigma_new

def solve_qp_energy_scgw(mf, mol, nocc, skip_scf=False, grid_kind='IR', save_file=None, **kwargs):
    """Self-consistent GW quasiparticle energies on IR grids.

    Starts from the mean-field G0, iterates Sigma, mu, the density and F to
    self-consistency (convergence is on the density matrix), then continues
    Sigma to the real axis with Pade and takes the quasiparticle energies
    from the peaks of the spectral function A(omega) = -Tr[Im G(omega)] / pi.

    mf, mol: converged pyscf mean-field object and its molecule.
    nocc: number of doubly occupied orbitals.
    skip_scf: stop after the first Dyson step (see Returns).
    grid_kind: only 'IR' is implemented.
    save_file: if given, (w_real, A) is written there as CSV.
    kwargs: `factors` (precomputed output of `separable_factors`) or
        `auxbasis` (used to build them).

    Returns (qpes, A, w_real): the peak positions, the spectral function and
    the real frequency axis it is on.
    With skip_scf=True, returns
    (G, F, G0, mu, V, h_mo, grid_f, grid_b, X_mo, D) instead.
    """

    if grid_kind.upper() != 'IR':
        raise NotImplementedError('The only method now implemented is the IR method')

    # --- Starting point: mean-field orbitals, mu in the middle of the gap ---
    eps = get_orbital_energies(mf, representation='spatial')

    occ, virt = get_occ_virt_indices(eps, nocc)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    F_bare = np.diag(eps)

    # --- IR grids: fermion for G/Sigma, boson for chi/W ---
    # omega_max must cover G, W and Sigma = G*W at once, so it is the sum of
    # the largest orbital energy (from mu) and the largest transition, padded.
    dG = np.abs(eps - mu)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    omega_max = 3.0 * (dG.max() + e_max)

    # beta must be large enough that the gap is effectively at T = 0, but
    # not larger: Lambda = beta * omega_max also sets how well conditioned
    # the omega -> tau transform is. Measured on Ne/cc-pVDZ, with the default
    # Matsubara sampling: 100/dG.min() diverged within ~5 iterations,
    # 25/dG.min() was stable.
    beta = 100.0 / dG.min()

    # The default Matsubara sampling only reaches 1.1 - 1.4 x omega_max,
    # which leaves the transform ill-conditioned near tau = 0 and beta, where
    # the density is read off. n_matsubara_factor is chosen so the sampling
    # reaches about `reach` x omega_max, from
    #     omega_top / omega_max ~ 1 + factor * size * 2 pi / Lambda.
    # size_guess overestimates the IR size, so the true reach is a bit lower.
    # Measured on Ne/cc-pVDZ at Lambda = 1.7e4: reach 1.12 diverges, 5.4 is
    # stable.
    reach = 6.0
    size_guess = 100
    n_matsubara_factor = int(np.ceil(
        (reach - 1.0) * beta * omega_max / (2 * np.pi * size_guess)))

    grid_f = TimeFrequencyGrid.ir(beta, omega_max, statistics='fermion',
                                    n_matsubara_factor=n_matsubara_factor)
    
    grid_b = TimeFrequencyGrid.ir(beta, omega_max, statistics='boson',
                                    n_matsubara_factor=n_matsubara_factor)

    print(f'IR grids: Lambda={beta * omega_max:.4g}, '
            f'fermion size={grid_f.meta["basis"].size}, '
            f'boson size={grid_b.meta["basis"].size}, '
            f'n_matsubara_factor={n_matsubara_factor}, '
            f'highest sampled omega = '
            f'{grid_f.omega_points.max() / omega_max:.2f} x omega_max')
    # Grid quality check. duality_error is used rather than roundtrip_error
    # because the latter's test function is bosonic and so not valid for
    # the fermion grid.
    print(f'\nfermion duality_error (even/odd) = '
            f'{grid_f.duality_error("even"):.2e} / {grid_f.duality_error("odd"):.2e}')
    print(f'boson   duality_error (even/odd) = '
            f'{grid_b.duality_error("even"):.2e} / {grid_b.duality_error("odd"):.2e}\n')

    X_mo, D, X_ao, coords = (kwargs.get('factors') if kwargs.get('factors') is not None
                                else separable_factors(mf, mol, auxbasis=kwargs.get('auxbasis')))

    # --- First iteration: G0 -> gamma, Sigma -> F -> G ---
    freq_points = grid_f.omega_points
    nmo = len(eps)
    eye = np.eye(nmo)
    G0 = np.array([np.linalg.inv((1j * w + mu) * eye - F_bare)
                    for w in freq_points])

    print(f'G0 (IR, bare): shape={G0.shape}, max|G0|={np.max(np.abs(G0)):.4e}')

    gamma = density_matrix_ir(G0, grid_f)
    print(f'Trace gamma_G0: {np.trace(gamma)}, expected nocc: {nocc}\n')

    G_tau, chi_omega, W_omega, sigma_new = calc_new_sigma_ir(
        G0, grid_f, grid_b, X_mo, D)

    h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
    V = get_two_electron_integrals_chemist(mol,mf)

    F = build_new_F(gamma, V, h_mo)
    G = build_G_from_F(mu, grid_f.omega_points, sigma_new, F)

    if skip_scf:
        return G, F, G0, mu, V, h_mo, grid_f, grid_b, X_mo, D

    # --- Self-consistency cycle ---
    # Per iteration: new Sigma from G, then repeat the mu search (which also
    # rebuilds F, G and gamma) until mu stops moving, then mix gamma with the
    # previous one (fraction alpha of the new).
    delta_gamma = 999
    gamma_convergence = GAMMA_CONVERGENCE

    alpha = GAMMA_MIXING

    gamma = density_matrix_ir(G, grid_f)

    print('\n----SCF----\n')

    for _ in range(MAX_SCF_ITER):
        if delta_gamma <= gamma_convergence:
            break

        gamma_old = gamma
        sigma_old = sigma_new

        _, _, _, sigma_new = calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D)

        delta_sigma = np.max(np.abs(sigma_old - sigma_new))

        mu_pre_scf = mu

        mu_convergence = MU_CONVERGENCE

        for _ in range(MAX_MU_ITER):
            mu_old = mu

            mu, G, gamma, F = mu_cycle(mu, G, gamma, nocc,
                            sigma_mo=sigma_new, V=V, h_mo=h_mo, grid_f=grid_f, grid_kind='IR')

            if abs(mu_old - mu) <= mu_convergence:
                break
        else:
            raise RuntimeError(f'scGW: mu did not settle within {MAX_MU_ITER} '
                               f'mu searches (last change {abs(mu_old - mu):.2e})')

        print(f'Mu post mu_scf: {mu}')

        gamma = alpha * gamma + (1 - alpha) * gamma_old

        delta_gamma = np.max(np.abs(gamma - gamma_old))
        delta_mu = abs(mu - mu_pre_scf)
        print(f'Delta gamma: {delta_gamma}\nDelta mu: {delta_mu}\nDelta Sigma: {delta_sigma}\n\n')
    else:
        if delta_gamma > gamma_convergence:
            raise RuntimeError(f'scGW: not converged in {MAX_SCF_ITER} iterations '
                               f'(delta gamma = {delta_gamma:.2e})')

    print('\n----END SCF----\n')

    # --- Real axis: Pade-continue Sigma, build G(omega), find peaks of A ---
    # The frequency window runs from the lowest valence level to the highest
    # level of F, with 0.5 Ha margin. eta is the broadening; the frequency
    # step is finer, so each peak (half-width ~eta) is well sampled.
    eta = 1e-2
    w_step = eta / 10

    eps_F = np.linalg.eigvalsh(F)
    w_real = np.arange(eps_F[chemcore(mol)] - 0.5, eps_F[-1] + 0.5, w_step)
    pade_max_order = 24

    z_fit = mu + 1j * grid_f.omega_points
    z_real = w_real + 1j * eta

    sigma_real = pade_continue_elementwise(z_fit, sigma_new, z_real, greedy=True,
                                        symmetric=True, max_order=pade_max_order) 

    print(f'\nSigma(w) on {len(w_real)} real frequencies in [{w_real[0]}, {w_real[-1]}] Ha, '
        f'eta = {eta}: shape {sigma_real.shape}')
    print(f'Pade check, max |Sigma_pade(i*w_n) - Sigma(i*w_n)|: '
        f'{np.max(np.abs(pade_continue_elementwise(z_fit, sigma_new, z_fit, symmetric=True, max_order=pade_max_order) - sigma_new))}\n')

    G_w = build_Greal_from_F(w_real, sigma_real, F, eta)
    A = -1/np.pi * np.trace(G_w, axis1=1, axis2=2).imag

    if save_file is not None:
        np.savetxt(save_file, (w_real, A), delimiter=',')

    # A pole of weight Z, broadened by eta, peaks at Z / (pi * eta), so the
    # height threshold is set as a minimum weight rather than a fixed number.
    min_peak_weight = 0.05
    peaks, _ = scipy.signal.find_peaks(A, height=min_peak_weight / (np.pi * eta))
    qpes = np.empty(len(peaks))

    # Refine each peak position with a parabola through the maximum and its
    # two neighbours, instead of taking the grid point itself.
    for i, peak in enumerate(peaks):
        a_left, a_mid, a_right = A[peak - 1], A[peak], A[peak + 1]
        shift = 0.5 * (a_left - a_right) / (a_left - 2 * a_mid + a_right)
        qpes[i] = w_real[peak] + shift * w_step

    return qpes, A, w_real