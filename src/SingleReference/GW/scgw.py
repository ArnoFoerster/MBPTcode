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
off the peaks of the spectral function. The total energy is the
Galitskii-Migdal energy of the converged G and Sigma
(`galitskii_migdal_energy`).

`solve_qp_energy_scgw` is the entry point.
"""
import numpy as np
import scipy.signal
import resource

from pyscf.data.elements import chemcore
from src.SingleReference.CC.diis import DIIS
from src.Base.pyscf_interface import get_orbital_energies
from src.Base.utils.time_frequency import (TimeFrequencyGrid)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency
from src.Base.utils.analyticalContinuation import pade_continue_elementwise

# --- Convergence parameters ---
# Outer scGW cycle
RESIDUAL_CONVERGENCE = 1e-7 # max |Sigma_out - Sigma_in|, |F_out - F_in| at which the cycle stops
LINEAR_MIXING = 1.0         # fraction of the new (Sigma, F) kept in linear-mixing steps
DIIS_SIZE = 8               # DIIS subspace; 0 mixes linearly throughout
DIIS_START = 3              # linear-mixing iterations before DIIS takes over
MAX_SCF_ITER = 150          # cap on scGW iterations

# mu search (mu_cycle)
MU_STEP = 1e-6              # first step away from the starting mu
TRACE_TOL = 1e-8            # |Tr[gamma] - nocc| at which mu is converged
MU_KEEP_TOL = 1e-6          # |Tr[gamma] - nocc| at the incoming mu below which no search is done
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

def galitskii_migdal_energy(G, sigma_mo, mf, h_mo, grid_f, e_nuc):
    """Closed-shell Galitskii-Migdal total energy from G and Sigma_c.

        E = E_nuc + Tr[gamma (h + F)] + (1/beta) sum_n Tr[Sigma_c(i.omega_n) G(i.omega_n)]

    with gamma per spin and F = h + 2J[gamma] - K[gamma]. The prefactor of
    the correlation term is 2 (spin) x 1/2 (GM).

    The Matsubara sum runs over all n, not over the IR sampling points. It is
    done as in `density_matrix_ir`: (1/beta) sum_n X(i.omega_n) = -X(tau = beta)
    for a fermionic X, evaluated through the IR basis. Tr[Sigma G] decays as
    1/omega^2, so X(tau) has no jump at tau = 0 and no convergence factor is
    needed. The trace is taken before the transform: the matrix Sigma G does
    not satisfy X(-i.omega) = X(i.omega)^dagger, which the even/odd split
    assumes, but its trace satisfies x(-i.omega) = x(i.omega)*.

    G is expected to be the Dyson solution for sigma_mo; at convergence
    sigma_mo = Sigma[G] as well. gamma and F are rebuilt from G, not taken
    from the (mixed) quantities of the scGW cycle.

    Checked on H2/cc-pVDZ: with Sigma = 0 and the HF G0 it reproduces the HF
    energy to 3e-8; for G0 and Sigma[G0] with the interaction scaled by
    lambda, E_c / lambda^2 tends to twice the direct MP2 energy.

    G, sigma_mo: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    mf: the mean-field object, for J and K (see `build_new_F`).
    Returns (E, E_c): the total energy and its correlation part, both real.
    """
    gamma = density_matrix_ir(G, grid_f)
    F = build_new_F(gamma, mf, h_mo)
    E_1b = np.einsum('pq,qp->', gamma, h_mo + F).real

    tr_sigma_G = np.einsum('wij,wji->w', sigma_mo, G)
    E_c = density_matrix_ir(tr_sigma_G[:, None, None], grid_f)[0, 0].real

    return e_nuc + E_1b + E_c, E_c

def build_new_F(gamma, mf, h_mo):
    """Closed-shell Fock matrix F = h + 2J[gamma] - K[gamma], in the MO basis.

    J and K are built in the AO basis by mf.get_jk, so the (nmo^4) MO
    integrals are never stored; whether pyscf keeps AO integrals in core,
    works integral-direct or uses density fitting is decided by mf. With
    hermi=0 pyscf's J and K equal the MO-basis contractions
    J_ij = (ij|kl) gamma_kl and K_ij = (ik|jl) gamma_kl for any gamma, also
    one that is not exactly symmetric.

    gamma: (nmo, nmo) density matrix per spin (occupations between 0 and 1).
    mf: the mean-field object whose mo_coeff define the MO basis.
    h_mo: core Hamiltonian.
    """
    C = mf.mo_coeff
    J_ao, K_ao = mf.get_jk(mf.mol, C @ gamma @ C.T, hermi=0)

    return h_mo + C.T @ (2* J_ao - K_ao) @ C

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

def build_Greal_from_F(freq_points, sigma_real, F, eta, trace_only=False):
    """Dyson equation just above the real axis, at z = omega + i.eta:
    G(z) = [z I - F - Sigma(z)]^-1.

    sigma_real must be Sigma evaluated at the same z = omega + i.eta, so eta
    enters both terms. This is not double broadening: it is G at one complex
    point, so every pole of G becomes a Lorentzian of half-width eta and height
    Z / (pi * eta). Taking Sigma at omega + i0 instead would give widths Z * eta
    and the same height 1 / (pi * eta) for every peak, whatever its weight.

    No mu here: sigma_real is expected on absolute real frequencies.

    trace_only: return Tr[G(z)] per frequency instead of the full G, so the
    (nfreq, nmo, nmo) array is never stored.
    """

    eye = np.eye(F.shape[0])

    if trace_only:
        tr_G = np.empty(len(freq_points), dtype=complex)
        for i, w in enumerate(freq_points):
            tr_G[i] = np.trace(np.linalg.inv((w + 1j*eta)*eye - F - sigma_real[i]))
        return tr_G

    G_w = np.empty_like(sigma_real)
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

def mu_cycle(mu, F, nocc, sigma_mo,
            mu_step=MU_STEP, trace_tol=TRACE_TOL, keep_tol=MU_KEEP_TOL, mu_tol=MU_TOL,
            max_bracket_iter=MAX_BRACKET_ITER, max_bisect_iter=MAX_BISECT_ITER,
            grid_f=None, grid_kind='IR', stall_trace_tol=STALL_TRACE_TOL):
    """Find the mu for which Tr[gamma(mu)] = nocc.

    F and Sigma are given by the caller and kept fixed during the search.
    Only then is Tr[gamma(mu)] monotonic in mu, which bisection needs;
    rebuilding F at every trial mu made the trace jump. Taking F as input
    lets the caller mix F before the search, so the returned G is the Dyson
    solution of the mixed F with the right particle number.

    Two phases:
      1. Step away from the starting mu, growing the step by 1.5x each time,
         until the trace crosses nocc.
      2. Bisect inside that bracket.

    If the trace at the incoming mu is already within keep_tol of nocc, mu
    is kept and no search is done. In a gapped system at large beta the
    trace is nearly flat in mu inside the gap, so a trace error at the
    1e-8 level of the IR density moves mu by ~1e-6 each search; mu then
    drifts every scGW iteration and G(i.omega), which depends on mu
    explicitly, never converges (seen on Ne/cc-pVDZ).

    Once searching, converged when |trace - nocc| < trace_tol. If the bracket shrinks below
    mu_tol first, the result is still accepted when
    |trace - nocc| < stall_trace_tol; otherwise a RuntimeError is raised.

    Returns (mu, G, gamma).
    """
    trace_old, gamma, G = _mu_cycle_find_trace(
        mu, F, sigma_mo, grid_f=grid_f, grid_kind=grid_kind,)

    # print(f'Trace before mu-cycle: {trace_old}')

    if abs(trace_old - nocc) < keep_tol:
        print(f'mu_cycle: mu kept, |trace - nocc| = {abs(trace_old - nocc):.2e}')
        return mu, G, gamma
    del G, gamma

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
        del G_b, gamma_b
    else:
        raise RuntimeError(f'mu_cycle: failed to bracket nocc={nocc} '
                           f'within {max_bracket_iter} probes')

    # --- Phase 2: bisection ---
    mu_lo, mu_hi, trace_lo = mu_a, mu_b, trace_a
    del G_b, gamma_b
    for _ in range(max_bisect_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        G_mid = gamma_mid = None  # free the previous trial before building the next
        trace_mid, gamma_mid, G_mid = _mu_cycle_find_trace(
            mu_mid, F, sigma_mo, grid_f=grid_f, grid_kind=grid_kind)

        if abs(trace_mid - nocc) < trace_tol:
            print(f'mu_cycle converged: mu={mu_mid}, trace={trace_mid}')
            return mu_mid, G_mid, gamma_mid

        if abs(mu_hi - mu_lo) < mu_tol:
            if abs(trace_mid - nocc) < stall_trace_tol:
                print(f'mu_cycle converged (bracket collapsed, trace within '
                     f'stall_trace_tol): mu={mu_mid}, trace={trace_mid}')
                return mu_mid, G_mid, gamma_mid
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

def calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D, return_all=False):
    """Correlation self-energy Sigma(i.omega) from a dressed G(i.omega).

    Steps: G -> chi = G*G -> W = [I - chi]^-1 -> Sigma = G*W. The products
    are taken pointwise in imaginary time on the ISDF/THC grid; the Dyson
    inversion for W is done in frequency. G and Sigma live on the fermion
    grid, chi and W on the boson grid, and `project_tau_cross` moves
    quantities between the two.

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    X_mo, D: separable (ISDF/THC) factors from `separable_factors`.
    return_all: also return the intermediates; otherwise only sigma_new is
        returned and chi, W and G(tau) are freed as soon as they are used.

    Returns sigma_new, or with return_all=True (G_tau, chi_omega, W_omega, sigma_new):
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

    # MO basis -> ISDF/THC grid (lesser = even - odd, greater = even + odd).
    Ghat_lesser_b = np.einsum('pi,tij,qj->tpq', X_mo, G_even_b - G_odd_b, X_mo, optimize=True)
    Ghat_greater_b = np.einsum('pi,tij,qj->tpq', X_mo, G_even_b + G_odd_b, X_mo, optimize=True)
    del G_even_b, G_odd_b

    # chi(i.omega) from the dressed G. The minus on Gv matches the sign
    # convention chi0_imaginary_frequency uses for its bare Go/Gv. eps and
    # nocc are not needed when Go/Gv are passed, hence None.
    chi_omega = chi0_imaginary_frequency(X_mo, D, None, None, grid_b,
                                         Go=Ghat_greater_b, Gv=-Ghat_lesser_b)
    del Ghat_lesser_b, Ghat_greater_b

    # W = [I - chi]^-1 at each frequency.
    naux = chi_omega.shape[-1]
    eye_aux = np.eye(naux)
    W_omega = np.empty_like(chi_omega)
    for k in range(chi_omega.shape[0]):
        W_omega[k] = np.linalg.inv(eye_aux - chi_omega[k])
    if not return_all:
        del chi_omega

    # W(i.omega) -> W(tau). Only the correlation part W - I is transformed
    # (the constant does not decay), and it is even, so cosine only.
    Wt_tau_b = np.einsum('tk,kab->tab', grid_b.cosft_tw, W_omega - eye_aux, optimize=True)
    if not return_all:
        del W_omega

    # Sigma lives on the fermion grid, so move W onto the fermion tau points.
    Wt_tau_f = project_tau_cross(grid_b, grid_f, Wt_tau_b, naux, 'even')
    del Wt_tau_b

    # Auxiliary basis -> ISDF/THC grid.
    Zt_tau = np.einsum('Pa,tab,Qb->tPQ', D, Wt_tau_f, D, optimize=True)
    del Wt_tau_f

    # Sigma(tau) = G(tau) * W(tau), elementwise on the ISDF/THC grid. Unlike
    # chi, no extra sign is needed here. G is taken on the ISDF/THC grid at
    # the fermion tau points straight from the fermion-grid G rather than
    # projecting the boson-grid copy back. Lesser and greater are done one
    # after the other and multiplied in place, so at most two
    # (ntau, ngrid, ngrid) arrays are alive at once.
    sig_thc = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f - G_odd_f, X_mo, optimize=True)
    sig_thc *= Zt_tau
    sig_lesser = np.einsum('pi,tpq,qj->tij', X_mo, sig_thc, X_mo, optimize=True)
    del sig_thc

    sig_thc = np.einsum('pi,tij,qj->tpq', X_mo, G_even_f + G_odd_f, X_mo, optimize=True)
    sig_thc *= Zt_tau
    sig_greater = np.einsum('pi,tpq,qj->tij', X_mo, sig_thc, X_mo, optimize=True)
    del sig_thc, Zt_tau

    even = sig_greater + sig_lesser
    odd = sig_greater - sig_lesser
    del sig_lesser, sig_greater

    # Sigma(tau) -> Sigma(i.omega) on the fermion grid.
    # Overall prefactor -0.5, the same normalization as
    # self_energy_matrix_imaginary_time (imaginary_time.py). Checked against
    # that routine for the bare G0: agreement to 8e-9.
    sigma_new = -0.5 * (np.einsum('wt,tij->wij', grid_f.cosft_wt, even, optimize=True)
                        + 1j * np.einsum('wt,tij->wij', grid_f.sinft_wt, odd, optimize=True))

    if not return_all:
        return sigma_new

    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    return (G_lesser_f + 1j * G_greater_f), chi_omega, W_omega, sigma_new

def pack_sigma_F(sigma, F, sigma_weight=1.0):
    """(Sigma, F) as one real vector, for DIIS.

    Real and imaginary parts of Sigma are stored separately, so the plain
    dot product of two such vectors is Re<e1|e2>, the DIIS inner product.
    sigma_weight scales the Sigma block: it has nfreq times more entries
    than F and would otherwise dominate the DIIS B-matrix. It is applied to
    the error vectors only; the iterates are packed unweighted, so DIIS
    combines them without unpacking the weight again.

    sigma: (nfreq, nmo, nmo) complex. F: (nmo, nmo) real.
    """
    return np.concatenate([sigma_weight * sigma.real.ravel(),
                           sigma_weight * sigma.imag.ravel(),
                           F.ravel()])

def unpack_sigma_F(x, sigma_shape):
    """Inverse of `pack_sigma_F` (for sigma_weight = 1)."""
    n = int(np.prod(sigma_shape))
    sigma = (x[:n] + 1j * x[n:2 * n]).reshape(sigma_shape)
    # Copy, so F does not keep the whole packed vector x alive as a view.
    F = x[2 * n:].reshape(sigma_shape[1:]).copy()
    return sigma, F

def pack_G(G):
    """G as one real vector for DIIS (real and imaginary parts, as in `pack_sigma_F`)."""
    return np.concatenate([G.real.ravel(), G.imag.ravel()])

def unpack_G(x, G_shape):
    """Inverse of `pack_G`."""
    n = int(np.prod(G_shape))
    return (x[:n] + 1j * x[n:]).reshape(G_shape)

def solve_qp_energy_scgw(mf, mol, nocc, skip_scf=False, grid_kind='IR', save_file=None, **kwargs):
    """Self-consistent GW quasiparticle energies on IR grids.

    Starts from the mean-field G0, iterates Sigma and F to self-consistency
    (DIIS on (Sigma, F), convergence on their residual), then continues
    Sigma to the real axis with Pade and takes the quasiparticle energies
    from the peaks of the spectral function A(omega) = -Tr[Im G(omega)] / pi.
    The total energy is the Galitskii-Migdal energy of the converged G and
    Sigma.

    mf, mol: converged pyscf mean-field object and its molecule.
    nocc: number of doubly occupied orbitals.
    skip_scf: stop after the first Dyson step (see Returns).
    grid_kind: only 'IR' is implemented.
    save_file: if given, (w_real, A) is written there as CSV.
    kwargs: `factors` (precomputed output of `separable_factors`) or
        `auxbasis` (used to build them); `beta` and `omega_max` override
        the IR grid parameters that are otherwise derived from mf's
        orbital energies (e.g. to run different starts on the same grid);
        `diis_size`, `diis_start` and `mixing` override DIIS_SIZE,
        DIIS_START and LINEAR_MIXING; `mix` picks what is mixed, 'sigma_F'
        (default) or 'G' (see the comment at the cycle).

    Returns (qpes, A, w_real, E_gm, mu): the peak positions, the spectral
    function, the real frequency axis it is on, the Galitskii-Migdal
    total energy of the converged G and Sigma (`galitskii_migdal_energy`)
    and the converged chemical potential.
    With skip_scf=True, returns
    (G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D) instead.
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
    omega_max = kwargs.get('omega_max') or 3.0 * (dG.max() + e_max)

    # beta must be large enough that the gap is effectively at T = 0, but
    # not larger: Lambda = beta * omega_max also sets how well conditioned
    # the omega -> tau transform is. Measured on Ne/cc-pVDZ, with the default
    # Matsubara sampling: 100/dG.min() diverged within ~5 iterations,
    # 25/dG.min() was stable.
    beta = kwargs.get('beta') or 25.0 / dG.min()

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

    X_mo, D, _, _ = (kwargs.get('factors') if kwargs.get('factors') is not None
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

    sigma_new = calc_new_sigma_ir(G0, grid_f, grid_b, X_mo, D)

    h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
    e_nuc = mol.energy_nuc()

    F = build_new_F(gamma, mf, h_mo)
    G = build_G_from_F(mu, grid_f.omega_points, sigma_new, F)

    if skip_scf:
        return G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D
    del G0

    # --- Self-consistency cycle ---
    # mix='sigma_F': the iterate is the Dyson input x_in = (Sigma_in, F_in).
    #   1. x_out = (Sigma[G], F[gamma[G]]) from the current G
    #   2. residual r = x_out - x_in; stop when it is below tolerance
    #   3. new x_in by linear mixing (first diis_start iterations, or when
    #      the DIIS B-matrix is singular) or DIIS
    #   4. mu search with x_in fixed -> G, gamma
    # The mu search comes last, so every G is the Dyson solution of x_in with
    # Tr[gamma] = nocc, and its GM energy is meaningful at every iteration.
    #
    # mix='G': the iterate is G_in (gamma_in = gamma[G_in] is linear in G,
    # so mixing G mixes gamma as well).
    #   1. Sigma[G_in], F[gamma_in], mu search -> G_out
    #   2. residual r = G_out - G_in; stop when it is below tolerance
    #   3. new G_in by linear mixing or DIIS
    # Every G_out has Tr[gamma] = nocc and the mixing coefficients sum to 1,
    # so G_in keeps the right particle number, but it is not the Dyson
    # solution of any Sigma; the GM energy is that of G_out.
    #
    # mu itself is not mixed in either: the search fixes it from Sigma and F.
    mix = kwargs.get('mix', 'sigma_F')
    if mix not in ('sigma_F', 'G'):
        raise ValueError(f"mix={mix!r}: choose 'sigma_F' or 'G'")
    alpha = kwargs.get('mixing', LINEAR_MIXING)
    diis_size = kwargs.get('diis_size', DIIS_SIZE)
    diis_start = kwargs.get('diis_start', DIIS_START)
    diis = DIIS(diis_size, start_iter=0) if diis_size > 0 else None

    # Weight of the Sigma block in the DIIS error, see pack_sigma_F.
    sigma_weight = 1.0 / np.sqrt(len(freq_points))
    sigma_shape = sigma_new.shape

    sigma_in, F_in = sigma_new, F
    del sigma_new, F, G, gamma
    mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                            grid_f=grid_f, grid_kind='IR')

    print(f'\n----SCF (mixing {mix})----\n')

    residual = np.inf
    for it in range(MAX_SCF_ITER):
        mu_old = mu
        if mix == 'sigma_F':
            # 1. Outputs from the current G
            sigma_out = calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D)
            F_out = build_new_F(gamma, mf, h_mo)

            # 2. Residual
            r_sigma = sigma_out - sigma_in
            r_F = F_out - F_in
            res_sigma = np.max(np.abs(r_sigma))
            res_F = np.max(np.abs(r_F))
            residual = max(res_sigma, res_F)
            print(f'Iteration {it}: residual Sigma: {res_sigma:.3e}, F: {res_F:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                del sigma_out, F_out, r_sigma, r_F
                break

            x_in = pack_sigma_F(sigma_in, F_in)
            x_out = pack_sigma_F(sigma_out, F_out)
            error = pack_sigma_F(r_sigma, r_F, sigma_weight)
            del sigma_out, F_out, r_sigma, r_F
        else:
            # 1. G_out from the current G_in
            sigma_in = calc_new_sigma_ir(G, grid_f, grid_b, X_mo, D)
            F_in = build_new_F(gamma, mf, h_mo)
            mu, G_out, gamma_out = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                            grid_f=grid_f, grid_kind='IR')

            # 2. Residual
            res_G = np.max(np.abs(G_out - G))
            res_gamma = np.max(np.abs(gamma_out - gamma))
            residual = res_G
            print(f'Iteration {it}: residual G: {res_G:.3e}, gamma: {res_gamma:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                G, gamma = G_out, gamma_out
                del G_out, gamma_out
                break

            x_in = pack_G(G)
            x_out = pack_G(G_out)
            error = x_out - x_in

        # 3. Mix
        x_new = None
        if diis is not None and it >= diis_start:
            try:
                x_new = diis.compute_new_vec(x_out, error)
                step = 'DIIS'
            except np.linalg.LinAlgError:
                # Near convergence the error vectors can become linearly
                # dependent; take a linear step instead.
                pass
        if x_new is None:
            x_new = x_in + alpha * (x_out - x_in)
            step = f'linear (alpha = {alpha})'
        del x_in, x_out, error

        if mix == 'sigma_F':
            sigma_in, F_in = unpack_sigma_F(x_new, sigma_shape)
            del x_new, G, gamma

            # 4. Dyson + particle number
            mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                    grid_f=grid_f, grid_kind='IR')
            G_dyson = G
        else:
            G = unpack_G(x_new, sigma_shape)
            gamma = density_matrix_ir(G, grid_f)
            G_dyson = G_out
            del x_new, G_out, gamma_out

        # G_dyson is the Dyson solution for sigma_in, so this is the GM
        # energy of the current iterate; it converges with the cycle.
        E_gm, E_c = galitskii_migdal_energy(G_dyson, sigma_in, mf, h_mo, grid_f, e_nuc)
        print(f'Step: {step}\nmu: {mu} (delta mu: {abs(mu - mu_old):.3e})\n'
              f'E_GM: {E_gm} (E_c: {E_c})\n\n')
        del G_dyson
    else:
        raise RuntimeError(f'scGW: not converged in {MAX_SCF_ITER} iterations '
                           f'(residual = {residual:.2e})')

    print('\n----END SCF----\n')

    # G is the Dyson solution of (sigma_in, F_in); continue those to the real
    # axis so G, F and Sigma below stay consistent.
    sigma_new, F = sigma_in, F_in

    E_gm, E_c = galitskii_migdal_energy(G, sigma_new, mf, h_mo, grid_f, e_nuc)
    print(f'Galitskii-Migdal energy: E = {E_gm} Ha, E_c = {E_c} Ha\n')

    # Nothing from the cycle is needed on the real axis; free it before the
    # (large) real-frequency arrays are allocated.
    del h_mo, X_mo, D, diis, G, gamma, grid_b, sigma_in, F_in

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

    tr_G_w = build_Greal_from_F(w_real, sigma_real, F, eta, trace_only=True)
    del sigma_real
    A = -1/np.pi * tr_G_w.imag

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

    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # ru_maxrss is in KiB on Linux
    print(f'\nPeak memory usage: {peak_gib} GiB\n')

    return qpes, A, w_real, E_gm, mu