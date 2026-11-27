"""Imaginary-axis Green's function side of scGW: the Fock matrix and Dyson
equation for G(i.omega), the density matrix read off G, the chemical
potential search that fixes the particle number, and the Galitskii-Migdal
total energy. Everything is in the MO basis of the starting mean-field
calculation, on the fermion IR grid.

    F = h + 2J[gamma] - K[gamma]
    G(i.omega) = [(i.omega + mu) I - F - Sigma(i.omega)]^-1
"""
import numpy as np

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
