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
import scipy.integrate
import scipy.signal
import resource

import h5py
from pyscf import df, lib
from pyscf.data.elements import chemcore
from src.Solvers.diis import DIIS
from src.Base.numerics.constants import SURFACE_GRID_ACCURACY
from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.SingleReference.GW.space_time import separable_factors
from src.Base.utils.grids.analytic_continuation import (pade_fit_elementwise,
                                                        pade_eval_elementwise)

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

# Real axis
PADE_EVAL_CHUNK = 16        # real frequencies per Sigma(w) / G(w) evaluation in trace_Greal_from_pade

# RI (build_B_ri, calc_new_sigma_ri)
RI_BLOCK_MB = 256           # size of one (block, nmo, nmo) slab of B; a few slabs are alive at once

def ri_blocks(naux, nmo, block_size=None):
    """Slices that split the auxiliary index 0..naux into consecutive blocks
    of block_size, by default as many P as fit one (block, nmo, nmo) slab of
    RI_BLOCK_MB."""
    if block_size is None:
        block_size = max(1, int(RI_BLOCK_MB * 2**20 // (8 * nmo * nmo)))
    return [slice(p0, min(p0 + block_size, naux)) for p0 in range(0, naux, block_size)]


def build_B_ri(mol, mo_coeff, auxbasis, block_size=None):
    """RI three-index tensor in the MO basis,
        B[P,p,q] = sum_mn L[P,mn] C[m,p] C[n,q],  (pq|rs) = sum_P B[P,p,q] B[P,r,s],
    with L the Cholesky factors of the AO integrals in `auxbasis`.

    L comes packed (lower triangle of mn). It is unpacked and transformed one
    block of P at a time into a preallocated B, so the unpacked AO tensor
    (naux, nao, nao) is never stored whole. Peak: B, plus L (about half of B)
    until return, plus a few blocks.
    """
    Lpq = df.incore.cholesky_eri(mol, auxbasis=auxbasis)
    naux, nmo = Lpq.shape[0], mo_coeff.shape[1]
    B = np.empty((naux, nmo, nmo))
    for blk in ri_blocks(naux, nmo, block_size):
        L = lib.unpack_tril(Lpq[blk])
        B[blk] = np.matmul(mo_coeff.T @ L, mo_coeff)
    return B

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

def trace_Greal_from_pade(freq_points, Z, A, F, eta, chunk=PADE_EVAL_CHUNK):
    """Tr[G(z)] at z = omega + i.eta, with Sigma(z) evaluated from its Pade fit.

    Same G as build_Greal_from_F with trace_only=True, but Sigma(z) comes from
    the fit (Z, A) of pade_fit_elementwise, chunk frequencies at a time, so
    neither the (nfreq, nmo, nmo) Sigma nor G is ever stored: memory is the
    fit plus a few (chunk, nmo, nmo) arrays, whatever the number of frequencies.
    """
    eye = np.eye(F.shape[0])
    tr_G = np.empty(len(freq_points), dtype=complex)
    for start in range(0, len(freq_points), chunk):
        z = np.asarray(freq_points[start:start + chunk]) + 1j*eta
        sigma = pade_eval_elementwise(z, Z, A)
        tr_G[start:start + chunk] = np.trace(
            np.linalg.inv(z[:, None, None]*eye - F - sigma), axis1=1, axis2=2)
    return tr_G

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
    Returns (ntau_dst, dim, dim), on dst_grid.tau_points, C-contiguous: the
    transposed view would make every [t] slice strided, which matmul cannot
    pass to BLAS (14 s instead of 0.5 s per Sigma build for CH4/cc-pVTZ).
    """
    ntau_src = len(src_grid.tau_points)
    flat = f_src.reshape(ntau_src, dim * dim).T               # (dim*dim, ntau_src)
    f_dst = src_grid.project_tau(flat, dst_grid.tau_points, parity=parity).T
    return np.ascontiguousarray(f_dst).reshape(-1, dim, dim)

def tau_projection_matrix(src_grid, dst_grid, parity):
    """The matrix P of `project_tau_cross`, f_dst[t'] = sum_t P[t, t'] f_src[t],
    shape (ntau_src, ntau_dst). The projection is linear in f, so P is its
    image of the identity."""
    return src_grid.project_tau(np.eye(len(src_grid.tau_points)),
                                dst_grid.tau_points, parity=parity)

def G_tau_slice(row_even, row_odd, G_flat, nmo):
    """Even and odd part of G at one tau point, sum_k row_even[k] Re G(i.omega_k)
    and sum_k row_odd[k] Im G(i.omega_k), as contiguous (nmo, nmo) arrays.
    G_flat: G as (nfreq, nmo*nmo) complex."""
    G_even = np.ascontiguousarray((row_even @ G_flat).real).reshape(nmo, nmo)
    G_odd = np.ascontiguousarray((row_odd @ G_flat).imag).reshape(nmo, nmo)
    return G_even, G_odd

def streaming_transforms(grid_f, grid_b):
    """The tau <-> i.omega transforms of the Sigma build, with the projection
    between the fermion and boson tau points multiplied in, so that each tau
    slice comes straight from the frequency axis (`G_tau_slice`):
        G(i.omega) -> G(tau_b), even / odd:   (ntau_b, nfreq_f) each
        (W - I)(i.omega_b) -> (W - I)(tau_f): (ntau_f, nfreq_b)
    G(i.omega) -> G(tau_f) is grid_f.cosft_tw / sinft_tw itself.
    Returns (G_to_b_even, G_to_b_odd, W_to_f)."""
    G_to_b_even = tau_projection_matrix(grid_f, grid_b, 'even').T @ grid_f.cosft_tw
    G_to_b_odd = tau_projection_matrix(grid_f, grid_b, 'odd').T @ grid_f.sinft_tw
    W_to_f = tau_projection_matrix(grid_b, grid_f, 'even').T @ grid_b.cosft_tw
    return G_to_b_even, G_to_b_odd, W_to_f

def W_minus_I_inplace(chi_omega):
    """W - I = [I - chi]^-1 - I at each frequency, written over chi_omega
    (nfreq, naux, naux), which is returned. Only the correlation part W - I
    is transformed to tau (the constant does not decay), and it is even, so
    cosine only."""
    eye_aux = np.eye(chi_omega.shape[-1])
    for k in range(len(chi_omega)):
        Wc = np.linalg.inv(eye_aux - chi_omega[k])
        Wc -= eye_aux
        chi_omega[k] = Wc
    return chi_omega

def calc_new_sigma_isdf(G, grid_f, grid_b, X_mo, D, return_all=False):
    """Correlation self-energy Sigma(i.omega) from a dressed G(i.omega).

    Steps: G -> chi = G*G -> W = [I - chi]^-1 -> Sigma = G*W. The products
    are taken pointwise in imaginary time on the ISDF/THC grid; the Dyson
    inversion for W is done in frequency. G and Sigma live on the fermion
    grid, chi and W on the boson grid, and the projection of
    `project_tau_cross` (as a matrix, `tau_projection_matrix`) moves
    quantities between the two.

    Both products are local in tau, so each is done in a loop over tau that
    takes the result straight back to a small basis, chi(tau) to the
    auxiliary basis and Sigma(tau) to the MO basis, before the transform to
    frequency. Only a few (ngrid, ngrid) slices are alive at once, never an
    (ntau, ngrid, ngrid) array: ngrid is several times nmo and grows with
    the number of atoms, so those arrays set the peak memory (2.2 GiB for
    CH4/cc-pVTZ, against ~0.7 GiB with the loops). The tau <-> i.omega
    transforms are streamed in the same loops (`streaming_transforms`), so
    the only arrays over a grid axis are G and Sigma(i.omega) (nfreq_f, nmo,
    nmo) and chi, then W - I in its place, (nfreq_b, naux, naux).

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
    nfreq_f = grid_f.cosft_wt.shape[0]
    nfreq_b = grid_b.cosft_wt.shape[0]
    G_flat = np.ascontiguousarray(G).reshape(len(G), nmo * nmo)

    # The tau <-> i.omega transforms and the fermion <-> boson projections are
    # streamed as in `calc_new_sigma_ri`: each tau slice of G and W is built
    # from the frequency axis when it is used (`streaming_transforms`), and
    # summed straight into chi(i.omega) and Sigma(i.omega).
    G_to_b_even, G_to_b_odd, W_to_f = streaming_transforms(grid_f, grid_b)

    # chi(i.omega) = -2 sum_tau cosft_wt[w, tau] D^T [Go(tau) * Gv(tau)] D,
    # the same as chi0_imaginary_frequency, with Go = Ghat_greater and
    # Gv = -Ghat_lesser for the dressed G. The two minus signs cancel.
    # Ghat = X_mo G X_mo^T is G on the ISDF/THC grid (lesser = even - odd,
    # greater = even + odd).
    #
    # The (ngrid, ngrid) products are written into buffers reused over tau,
    # and X^T, D^T are made contiguous: in benchmarks at ngrid = 740
    # (OpenBLAS), fresh arrays and transposed views as left operand made
    # these products up to 2-3x slower than the batched einsums.
    naux = D.shape[1]
    ngrid = X_mo.shape[0]
    XT = np.ascontiguousarray(X_mo.T)
    DT = np.ascontiguousarray(D.T)
    buf_a = np.empty((ngrid, ngrid))
    buf_b = np.empty((ngrid, ngrid))

    chi_omega = np.zeros((nfreq_b, naux, naux))
    for t in range(len(grid_b.tau_points)):
        G_even, G_odd = G_tau_slice(G_to_b_even[t], G_to_b_odd[t], G_flat, nmo)
        np.matmul(X_mo @ (G_even - G_odd), XT, out=buf_a)
        np.matmul(X_mo @ (G_even + G_odd), XT, out=buf_b)
        buf_a *= buf_b
        pi_t = (DT @ buf_a) @ D
        for k in range(nfreq_b):
            chi_omega[k] += (2.0 * grid_b.cosft_wt[k, t]) * pi_t
    del pi_t

    if return_all:
        chi_out = chi_omega.copy()
    Wc_omega = W_minus_I_inplace(chi_omega)
    del chi_omega

    # Sigma(tau) = G(tau) * Z(tau) elementwise on the ISDF/THC grid, with
    # Z = D (W - I) D^T; unlike chi, no extra sign is needed. G is taken at
    # the fermion tau points straight from the fermion-grid G rather than
    # projecting the boson-grid copy back. Sigma_greater +- Sigma_lesser,
    # the even and odd parts, need only the even and odd parts of G:
    #     even = 2 X^T [(X G_even X^T) * Z] X,  odd = 2 X^T [(X G_odd X^T) * Z] X.
    # Sigma(tau) -> Sigma(i.omega) with overall prefactor -0.5, the same
    # normalization as self_energy_matrix_imaginary_time (imaginary_time.py);
    # it cancels the 2 above, so even/odd below are the halves.
    # Checked against that routine for the bare G0: agreement to 8e-9.
    # Sigma(i.omega) = -sum_tau [cosft_wt even + 1j sinft_wt odd] is
    # accumulated one tau at a time.
    sigma_new = np.zeros((nfreq_f, nmo, nmo), dtype=complex)
    sigma_re, sigma_im = sigma_new.real, sigma_new.imag
    Z_t = np.empty((ngrid, ngrid))
    for t in range(len(grid_f.tau_points)):
        G_even, G_odd = G_tau_slice(grid_f.cosft_tw[t], grid_f.sinft_tw[t], G_flat, nmo)
        Wt = np.tensordot(W_to_f[t], Wc_omega, axes=(0, 0))
        np.matmul(D @ Wt, DT, out=Z_t)
        np.matmul(X_mo @ G_even, XT, out=buf_a)
        buf_a *= Z_t
        even = (XT @ buf_a) @ X_mo
        np.matmul(X_mo @ G_odd, XT, out=buf_a)
        buf_a *= Z_t
        odd = (XT @ buf_a) @ X_mo
        for k in range(nfreq_f):
            sigma_re[k] -= grid_f.cosft_wt[k, t] * even
            sigma_im[k] -= grid_f.sinft_wt[k, t] * odd
    del Z_t, buf_a, buf_b, Wt

    if not return_all:
        return sigma_new

    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)
    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    return (G_lesser_f + 1j * G_greater_f), chi_out, Wc_omega + np.eye(naux), sigma_new

def calc_new_sigma_ri(G, grid_f, grid_b, B, return_all=False, block_size=None):
    """Correlation self-energy Sigma(i.omega) from a dressed G(i.omega), with
    the Coulomb interaction in plain RI instead of the ISDF/THC factors.

    The same workflow as `calc_new_sigma_isdf`, G -> chi = G*G -> W = [I - chi]^-1
    -> Sigma = G*W, with the same grids, transforms and normalization. Only
    the two products in tau change: with B[P,p,q] ~ sum_g X_mo[g,p] X_mo[g,q] D[g,P]
    the elementwise products on the ISDF grid become contractions over MO
    indices,
        chi_PQ(tau)  = sum B[P,p,r] G_lesser[p,q] G_greater[r,s] B[Q,q,s]
        Sigma_pq(tau) = sum B[P,p,r] G[r,s] (W - I)[P,Q] B[Q,s,q],
    so chi and W are in the same auxiliary basis as with ISDF. Cost per tau
    point is O(naux nmo^3 + naux^2 nmo^2).

    Both products are summed over blocks of P (`ri_blocks`), so besides B
    only a few (block, nmo, nmo) slabs are alive, never a second B-sized
    array. The blocking does not change the flop count. The tau <-> i.omega
    transforms are streamed one tau point at a time, so the only arrays over
    a grid axis are G and Sigma(i.omega) (nfreq_f, nmo, nmo) and chi, then
    W - I in its place, (nfreq_b, naux, naux).

    G: (nfreq, nmo, nmo) complex, on grid_f.omega_points.
    B: (naux, nmo, nmo) RI three-index tensor in the MO basis, already
        contracted with the Coulomb metric, (pq|rs) = sum_P B[P,p,q] B[P,r,s].
    return_all: also return the intermediates; otherwise only sigma_new is
        returned.
    block_size: P per block; default from RI_BLOCK_MB.

    Returns sigma_new, or with return_all=True (G_tau, chi_omega, W_omega, sigma_new),
    as `calc_new_sigma_isdf`.
    """
    nmo = G.shape[-1]
    naux = B.shape[0]
    nfreq_f = grid_f.cosft_wt.shape[0]
    nfreq_b = grid_b.cosft_wt.shape[0]
    G_flat = np.ascontiguousarray(G).reshape(len(G), nmo * nmo)

    # Every transform between tau and i.omega, and the projection between
    # the fermion and boson tau points, is a small matrix along that axis
    # (`streaming_transforms`). Each tau slice of G and W is built when it is
    # used and summed straight into chi(i.omega) and Sigma(i.omega), so no
    # (ntau, nmo, nmo) or (ntau, naux, naux) array is stored.
    G_to_b_even, G_to_b_odd, W_to_f = streaming_transforms(grid_f, grid_b)

    # chi(tau): M_P = G_lesser^T B_P G_greater, then pi_PQ = sum_qs M_P[q,s] B_Q[q,s]
    # (lesser = even - odd, greater = even + odd, as in calc_new_sigma_isdf).
    # Each block of P fills its own rows pi[blk, :], so M is only ever one
    # block. chi(i.omega) = 2 sum_tau cosft_wt[omega, tau] pi(tau) is
    # accumulated one tau at a time.
    B_flat = B.reshape(naux, nmo * nmo)
    blocks = ri_blocks(naux, nmo, block_size)
    chi_omega = np.zeros((nfreq_b, naux, naux))
    pi_t = np.empty((naux, naux))
    for t in range(len(grid_b.tau_points)):
        G_even, G_odd = G_tau_slice(G_to_b_even[t], G_to_b_odd[t], G_flat, nmo)
        G_lesser = G_even - G_odd
        G_greater = G_even + G_odd
        for blk in blocks:
            M = np.matmul(G_lesser.T @ B[blk], G_greater)
            pi_t[blk] = M.reshape(-1, nmo * nmo) @ B_flat.T
        for k in range(nfreq_b):
            chi_omega[k] += (2.0 * grid_b.cosft_wt[k, t]) * pi_t
    del M, pi_t

    if return_all:
        chi_out = chi_omega.copy()

    Wc_omega = W_minus_I_inplace(chi_omega)
    del chi_omega

    # Sigma(tau): V_P = sum_Q (W - I)[P,Q] B_Q, then
    #     even = sum_P B_P G_even V_P,  odd = sum_P B_P G_odd V_P,
    # the RI form of 2 X^T [(X G X^T) * Z] X; the prefactor -0.5 is the same
    # as in calc_new_sigma_isdf. The sum over P is split into blocks: each
    # block needs only its own V_P (rows blk of W - I against all of B) and
    # adds sum_{P in blk, r} B[P,p,r] (G V_P)[r,q] to even / odd.
    # Sigma(i.omega) = -sum_tau [cosft_wt even + 1j sinft_wt odd] is
    # accumulated one tau at a time.
    sigma_new = np.zeros((nfreq_f, nmo, nmo), dtype=complex)
    sigma_re, sigma_im = sigma_new.real, sigma_new.imag
    for t in range(len(grid_f.tau_points)):
        G_even, G_odd = G_tau_slice(grid_f.cosft_tw[t], grid_f.sinft_tw[t], G_flat, nmo)
        Wt = np.tensordot(W_to_f[t], Wc_omega, axes=(0, 0))
        even = np.zeros((nmo, nmo))
        odd = np.zeros((nmo, nmo))
        for blk in blocks:
            V = (Wt[blk] @ B_flat).reshape(-1, nmo, nmo)
            even += np.tensordot(B[blk], np.matmul(G_even, V), axes=([0, 2], [0, 1]))
            odd += np.tensordot(B[blk], np.matmul(G_odd, V), axes=([0, 2], [0, 1]))
        for k in range(nfreq_f):
            sigma_re[k] -= grid_f.cosft_wt[k, t] * even
            sigma_im[k] -= grid_f.sinft_wt[k, t] * odd
    del V, Wt

    if not return_all:
        return sigma_new

    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)
    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    return (G_lesser_f + 1j * G_greater_f), chi_out, Wc_omega + np.eye(naux), sigma_new

def calc_new_sigma(G, grid_f, grid_b, X_mo=None, D=None, B=None, method='ISDF',
                   return_all=False):
    """Correlation self-energy Sigma(i.omega), by `calc_new_sigma_isdf`
    (method='ISDF', needs X_mo and D) or `calc_new_sigma_ri` (method='RI',
    needs B). Returns what the chosen routine returns.
    """
    method = method.upper()
    if method == 'ISDF':
        if X_mo is None or D is None:
            raise ValueError("method='ISDF' needs X_mo and D")
        return calc_new_sigma_isdf(G, grid_f, grid_b, X_mo, D, return_all=return_all)
    if method == 'RI':
        if B is None:
            raise ValueError("method='RI' needs B")
        return calc_new_sigma_ri(G, grid_f, grid_b, B, return_all=return_all)
    raise ValueError(f"Unknown method {method!r}, expected 'ISDF' or 'RI'")

def pack_sigma_F(sigma, F, sigma_weight=1.0):
    """(Sigma, F) as one real vector, for DIIS.

    Sigma is stored as its complex memory read as float64 (real and
    imaginary parts interleaved), so the plain dot product of two such
    vectors is Re<e1|e2>, the DIIS inner product, and `unpack_sigma_F` can
    return views instead of copies.
    sigma_weight scales the Sigma block: it has nfreq times more entries
    than F and would otherwise dominate the DIIS B-matrix. It is applied to
    the error vectors only; the iterates are packed unweighted, so DIIS
    combines them without unpacking the weight again.

    sigma: (nfreq, nmo, nmo) complex. F: (nmo, nmo) real.
    """
    n2 = 2 * sigma.size
    x = np.empty(n2 + F.size)
    x[:n2] = np.ascontiguousarray(sigma).view(np.float64).ravel()
    x[n2:] = F.ravel()
    if sigma_weight != 1.0:
        x[:n2] *= sigma_weight
    return x

def unpack_sigma_F(x, sigma_shape):
    """Inverse of `pack_sigma_F` (for sigma_weight = 1). Sigma and F are
    views into x, not copies: the packed DIIS vector and the Dyson input
    share memory, and x stays alive as long as either is used."""
    n2 = 2 * int(np.prod(sigma_shape))
    sigma = x[:n2].view(np.complex128).reshape(sigma_shape)
    F = x[n2:].reshape(sigma_shape[1:])
    return sigma, F

def pack_G(G):
    """G as one real vector for DIIS, laid out as in `pack_sigma_F`; a view
    of G's memory (no copy) when G is contiguous."""
    return np.ascontiguousarray(G).view(np.float64).ravel()

def unpack_G(x, G_shape):
    """Inverse of `pack_G`; a view of x."""
    return x.view(np.complex128).reshape(G_shape)

def pack_pade(Z, A, omega_points):
    """A symmetric `pade_fit_elementwise` fit (Z, A) in compact form for saving.

    Every node in Z is one of the fit points mu + i*omega_points, so it is
    stored as an index into omega_points (smallest integer type that fits)
    instead of a complex number, and only the upper triangle is kept.
    Elements below the fit's floor have all-zero coefficients, so their
    nodes do not matter; they get index 0.

    Returns {'node_index': (n_nodes, npair), 'coeff': (n_nodes, npair)}, with
    npair = nmo (nmo + 1) / 2 in np.triu_indices order; `unpack_pade` inverts it.
    """
    iu = np.triu_indices(Z.shape[-1])
    Z_u, A_u = Z[:, iu[0], iu[1]], A[:, iu[0], iu[1]]

    sorter = np.argsort(omega_points)
    pos = np.searchsorted(omega_points[sorter], Z_u.imag)
    index = sorter[np.clip(pos, 0, len(omega_points) - 1)]
    fitted = (A_u != 0).any(axis=0)
    index[:, ~fitted] = 0
    if not np.array_equal(omega_points[index][:, fitted], Z_u.imag[:, fitted]):
        raise ValueError('Pade nodes are not on omega_points')

    return {'node_index': index.astype(np.min_scalar_type(len(omega_points) - 1)),
            'coeff': A_u}

def unpack_pade(pade, omega_points, mu):
    """Inverse of `pack_pade`: the full (Z, A), each (n_nodes, nmo, nmo),
    ready for `pade_eval_elementwise` or `trace_Greal_from_pade`."""
    index, A_u = pade['node_index'], pade['coeff']
    npair = A_u.shape[1]
    nmo = int(round((np.sqrt(8 * npair + 1) - 1) / 2))
    iu = np.triu_indices(nmo)

    Z = np.zeros((len(A_u), nmo, nmo), dtype=complex)
    A = np.zeros((len(A_u), nmo, nmo), dtype=complex)
    Z[:, iu[0], iu[1]] = mu + 1j * np.asarray(omega_points)[index]
    A[:, iu[0], iu[1]] = A_u
    Z[:, iu[1], iu[0]] = Z[:, iu[0], iu[1]]
    A[:, iu[1], iu[0]] = A[:, iu[0], iu[1]]
    return Z, A

def _solve_qp_energy_scgw(mf, mol, nocc, skip_scf=False, method='ISDF', grid_kind='IR', save_file=None,
                          save_sigma=False, **kwargs):
    """Self-consistent GW on IR grids, up to (not including) the real axis.

    Starts from the mean-field G0 and iterates Sigma and F to
    self-consistency (DIIS on (Sigma, F), convergence on their residual).
    The total energy is the Galitskii-Migdal energy of the converged G and
    Sigma.

    mf, mol: converged pyscf mean-field object and its molecule.
    nocc: number of doubly occupied orbitals.
    skip_scf: stop after the first Dyson step (see Returns).
    grid_kind: only 'IR' is implemented.
    save_file: if given, the converged F_mo, mu, density (dm_ao, gamma_mo),
        mo_coeff, the Matsubara frequencies omega_points, E_gm, E_c, e_hf
        and method are written to save_file + '.h5' (pyscf chkfile, key
        'scgw', with mol).
    save_sigma: also write the converged Sigma (sigma_mo, on omega_points)
        to the h5. Off by default: it is (n_omega, nmo, nmo) complex and
        can be very large for large systems. Ignored without save_file.
    method: 'ISDF' (separable factors) or 'RI' (plain RI three-index
        tensor, needs `auxbasis`); see `calc_new_sigma`.
    kwargs: `factors` (precomputed output of `separable_factors`) or
        `auxbasis` and `grid_accuracy` (ISDF grid level, default
        SURFACE_GRID_ACCURACY; used to build them); `auxbasis` is also the
        RI auxiliary basis; `beta` and `omega_max` override
        the IR grid parameters that are otherwise derived from mf's
        orbital energies (e.g. to run different starts on the same grid);
        `diis_size`, `diis_start` and `mixing` override DIIS_SIZE,
        DIIS_START and LINEAR_MIXING; `mix` picks what is mixed, 'sigma_F'
        (default) or 'G' (see the comment at the cycle).

    Returns (sigma, F, mu, omega_points, E_gm): the converged Sigma(i w_n)
    and F, of which the converged G is the Dyson solution, the chemical
    potential, the fermion Matsubara frequencies Sigma is on, and the
    Galitskii-Migdal total energy (`galitskii_migdal_energy`).
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

    method = method.upper()
    if method == 'ISDF':
        X_mo, D, _, _ = (kwargs.get('factors') if kwargs.get('factors') is not None
                                    else separable_factors(mf, mol, auxbasis=kwargs.get('auxbasis'),
                                                        grid_accuracy=kwargs.get('grid_accuracy',
                                                                                    SURFACE_GRID_ACCURACY)))
        B = None
    elif method == 'RI':
        if kwargs.get('auxbasis') is None:
            raise ValueError("If method == 'RI' auxbasis must be specified")
        B = build_B_ri(mol, mf.mo_coeff, kwargs.get('auxbasis'))

        X_mo = None
        D = None
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'ISDF' or 'RI'")

    # --- First iteration: G0 -> gamma, Sigma -> F -> G ---
    freq_points = grid_f.omega_points
    nmo = len(eps)
    eye = np.eye(nmo)
    G0 = np.array([np.linalg.inv((1j * w + mu) * eye - F_bare)
                    for w in freq_points])

    print(f'G0 (IR, bare): shape={G0.shape}, max|G0|={np.max(np.abs(G0)):.4e}')

    gamma = density_matrix_ir(G0, grid_f)
    print(f'Trace gamma_G0: {np.trace(gamma)}, expected nocc: {nocc}\n')

    sigma_new = calc_new_sigma(G0, grid_f, grid_b, X_mo=X_mo, D=D, B=B, method=method)

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
    n_sigma = 2 * sigma_new.size    # length of the Sigma block of a packed vector

    # With mix='sigma_F', x_in is the only copy of the iterate: sigma_in and
    # F_in are views into it (`unpack_sigma_F`), so the Dyson input and the
    # DIIS vector are not stored twice.
    if mix == 'sigma_F':
        x_in = pack_sigma_F(sigma_new, F)
        sigma_in, F_in = unpack_sigma_F(x_in, sigma_shape)
    else:
        x_in = None
        sigma_in, F_in = sigma_new, F
    del sigma_new, F, G, gamma
    # print(f'Sigma_in: {sigma_in.nbytes / 2**30:.3f} GiB')
    mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                            grid_f=grid_f, grid_kind='IR')

    print(f'\n----SCF (mixing {mix})----\n')

    residual = np.inf
    for it in range(MAX_SCF_ITER):
        mu_old = mu
        if mix == 'sigma_F':
            # 1. Outputs from the current G
            sigma_out = calc_new_sigma(G, grid_f, grid_b, X_mo=X_mo, D=D, B=B, method=method)
            F_out = build_new_F(gamma, mf, h_mo)
            x_out = pack_sigma_F(sigma_out, F_out)
            del sigma_out, F_out

            # 2. Residual, x_out - x_in, read off the packed vectors and
            # then weighted in place into the DIIS error.
            error = x_out - x_in
            res_sigma = np.max(np.abs(error[:n_sigma].view(np.complex128)))
            res_F = np.max(np.abs(error[n_sigma:]))
            residual = max(res_sigma, res_F)
            print(f'Iteration {it}: residual Sigma: {res_sigma:.3e}, F: {res_F:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                del x_out, error
                break
            error[:n_sigma] *= sigma_weight
        else:
            # 1. G_out from the current G_in
            sigma_in = calc_new_sigma(G, grid_f, grid_b, X_mo=X_mo, D=D, B=B, method=method)
            F_in = build_new_F(gamma, mf, h_mo)
            mu, G_out, gamma_out = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                            grid_f=grid_f, grid_kind='IR')

            # 2. Residual. x_in and x_out are views of G and G_out, so only
            # the error is a new array.
            x_in = pack_G(G)
            x_out = pack_G(G_out)
            error = x_out - x_in
            res_G = np.max(np.abs(error.view(np.complex128)))
            res_gamma = np.max(np.abs(gamma_out - gamma))
            residual = res_G
            print(f'Iteration {it}: residual G: {res_G:.3e}, gamma: {res_gamma:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                G, gamma = G_out, gamma_out
                x_in = None
                del G_out, gamma_out, x_out, error
                break

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
            # x_in + alpha * (x_out - x_in), with one new array instead of three
            x_new = x_out - x_in
            x_new *= alpha
            x_new += x_in
            step = f'linear (alpha = {alpha})'
        del x_out, error

        if mix == 'sigma_F':
            x_in = x_new
            sigma_in, F_in = unpack_sigma_F(x_in, sigma_shape)
            del x_new, G, gamma

            # 4. Dyson + particle number
            mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                    grid_f=grid_f, grid_kind='IR')
            G_dyson = G
        else:
            # x_in is a view of the old G; drop it so that G can be freed.
            x_in = None
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

    # G is the Dyson solution of (sigma_in, F_in); these are what is returned
    # and continued to the real axis, so G, F and Sigma stay consistent.
    sigma_new, F = sigma_in, F_in

    E_gm, E_c = galitskii_migdal_energy(G, sigma_new, mf, h_mo, grid_f, e_nuc)
    print(f'Galitskii-Migdal energy: E = {E_gm} Ha, E_c = {E_c} Ha\n')

    if save_file is not None:
        # Read back with lib.chkfile.load(save_file + '.h5', 'scgw') and
        # load_mol. gamma is per spin in the MO basis of mf, so the total AO
        # density is 2 C gamma C^T. With save_sigma, sigma_mo, F_mo, mu and
        # omega_points are enough to redo the real-axis continuation without
        # the cycle; sigma_mo is (n_omega, nmo, nmo) complex, so it is off by
        # default.
        data = {
            'dm_ao': 2.0 * mf.mo_coeff @ gamma @ mf.mo_coeff.T,
            'gamma_mo': gamma, 'mo_coeff': mf.mo_coeff, 'F_mo': F,
            'omega_points': grid_f.omega_points,
            'mu': mu, 'E_gm': E_gm, 'E_c': E_c, 'e_hf': mf.e_tot,
            'method': method}
        if save_sigma:
            data['sigma_mo'] = sigma_new
        lib.chkfile.save(save_file + '.h5', 'scgw', data)
        lib.chkfile.save_mol(mol, save_file + '.h5')

    # Everything else from the cycle (G, the factors, DIIS, grid_b) is freed
    # on return, before the real-axis arrays are allocated.
    return sigma_new, F, mu, grid_f.omega_points, E_gm

def _finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm, save_file=None, **kwargs):
    """Real-axis part of `solve_qp_energy_scgw`, from a Pade fit of Sigma.

    Builds A(omega) = -Tr[Im G(omega + i.eta)] / pi with Sigma from the fit
    (pade_Z, pade_A) of `pade_fit_elementwise` (or `unpack_pade`) and takes
    the quasiparticle energies from its peaks. Shared by the run that has
    just fitted Sigma and the reconstruction from a saved fit.

    kwargs: `eta` the real-axis broadening; `A_window` and `A_range` the
        margin around and the range of the real frequency axis;
        `min_qp_weight` the smallest |weight| of a kept peak (default 0.7,
        see `analyze_peaks`).
    save_file: if given, (w_real, A, mu) is written there as CSV.

    Returns (qpes, A, w_real, E_gm, mu), as `solve_qp_energy_scgw`.
    """
    # The frequency window runs from the lowest valence level to the highest
    # level of F, with 0.5 Ha margin. eta is the broadening; the frequency
    # step is finer, so each peak (half-width ~eta) is well sampled.
    eta = kwargs.get('eta', 1e-2)
    w_step = eta / 10

    eps_F = np.linalg.eigvalsh(F)

    # the spectral function's window and range (kwargs A_window, A_range)
    A_window = kwargs.get('A_window') or 0.5
    eps_low = eps_F[max(chemcore(mol) - getattr(mf, "frozen_core", 0), 0)]
    A_range = kwargs.get('A_range') or [eps_low - A_window, 2 * mu - eps_low + A_window]

    w_real = np.arange(A_range[0], A_range[1], w_step)
    print(f'\nSigma(w) on {len(w_real)} real frequencies in [{w_real[0]}, {w_real[-1]}] Ha, '
          f'eta = {eta}: Pade fit {pade_Z.shape}')

    # Sigma(w) is evaluated a chunk of frequencies at a time inside
    # trace_Greal_from_pade, so (len(w_real), nmo, nmo) is never stored.
    tr_G_w = trace_Greal_from_pade(w_real, pade_Z, pade_A, F, eta)
    A = -1/np.pi * tr_G_w.imag

    if save_file is not None:
        np.savetxt(save_file, (w_real, A, np.full_like(w_real, mu)), delimiter=',')

    # Peaks and dips of A, filtered on their integrated weight (analyze_peaks).
    qpes, _, _ = analyze_peaks(w_real, A, eta, w_step,
                               min_qp_weight=kwargs.get('min_qp_weight', 0.7))

    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # ru_maxrss is in KiB on Linux
    print(f'\nPeak memory usage: {peak_gib} GiB\n')

    return qpes, A, w_real, E_gm, mu

def analyze_peaks(w_real, A, eta, w_step, min_qp_weight=0.7):
    """Quasiparticle peaks of the spectral function A(omega).

    Peaks are searched for in A and in -A: a quasiparticle pole can come out
    of the Pade continuation as a dip (e.g. the Ne HOMO at eta = 1e-3), which
    a search for maxima alone misses. Between neighbouring peaks the boundary
    is the point where |A| is smallest (the zero crossing between a peak and
    a dip), and the weight of a peak is the integral of A between its two
    boundaries. Peaks with |weight| < min_qp_weight (satellites, or spurious
    peaks from a wrong eta) are dropped.

    A is a trace over orbitals, so a degenerate level has weight ~ Z times
    its degeneracy.

    Returns (qpes, weights, negative) for the kept peaks: the positions,
    refined with a parabola through the extremum and its two neighbours,
    their weights, and whether each is a dip (negative peak) in A.
    """
    # A pole of weight Z, broadened by eta, peaks at Z / (pi * eta), so the
    # height threshold is set as a minimum weight rather than a fixed number.
    min_peak_weight = 0.05
    height = min_peak_weight / (np.pi * eta)
    peaks_pos, _ = scipy.signal.find_peaks(A, height=height)
    peaks_neg, _ = scipy.signal.find_peaks(-A, height=height)
    peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))
    negative = np.isin(peaks, peaks_neg)

    # Boundaries at the smallest |A| between neighbouring peaks, and the data
    # edges outside the first and last peak; the weights then add up to the
    # integral of A over the whole axis.
    bounds = ([0] + [p0 + np.argmin(np.abs(A[p0:p1])) for p0, p1 in zip(peaks[:-1], peaks[1:])]
              + [len(A) - 1])
    weights = np.array([scipy.integrate.trapezoid(A[l:r + 1], w_real[l:r + 1])
                        for l, r in zip(bounds[:-1], bounds[1:])])

    # Refine each peak position with a parabola through the extremum and its
    # two neighbours, instead of taking the grid point itself (the same
    # formula holds for a minimum).
    qpes = np.empty(len(peaks))
    for i, peak in enumerate(peaks):
        a_left, a_mid, a_right = A[peak - 1], A[peak], A[peak + 1]
        shift = 0.5 * (a_left - a_right) / (a_left - 2 * a_mid + a_right)
        qpes[i] = w_real[peak] + shift * w_step

    keep = np.abs(weights) >= min_qp_weight

    print(f'\nPeaks of A (|weight| >= {min_qp_weight} kept):')
    for qpe, weight, neg, kept in zip(qpes, weights, negative, keep):
        print(f'  w = {qpe:.6f} Ha  weight = {weight:7.3f}'
              f'{"  (negative peak)" if neg else ""}{"" if kept else "  dropped"}')

    return qpes[keep], weights[keep], negative[keep]


def solve_qp_energy_scgw(mf, mol, nocc, ref_file=None, method='ISDF', grid_kind='IR', skip_scf=False, save_file=None,
                         save_pade=False, **kwargs):
    """Self-consistent GW quasiparticle energies on IR grids.

    Runs the scGW cycle (`_solve_qp_energy_scgw`), then continues the
    converged Sigma to the real axis with Pade and takes the quasiparticle
    energies from the peaks of the spectral function
    A(omega) = -Tr[Im G(omega)] / pi (`_finish_qp_from_pade`).

    Arguments are those of `_solve_qp_energy_scgw`, plus:
    ref_file: an .h5 file written by an earlier run (its save_file + '.h5').
        If given, the cycle is skipped and the real-axis part is redone from
        the saved 'scgw/sigma_mo' (refitted), or else from 'scgw/pade'
        (the saved fit, used as is). mf and mol must be those of that run.
    save_file: if given, (w_real, A, mu) is also written there as CSV.
    save_sigma: passed on to `_solve_qp_energy_scgw` through kwargs.
    save_pade: also write the Pade fit of Sigma to save_file + '.h5' under
        'scgw/pade' (see `pack_pade`), enough to redo A(omega), e.g. at a
        smaller eta, without Sigma. Smaller than sigma_mo, but fixes the
        fit (max_order, node choice). Ignored without save_file.
    kwargs: `eta` the real-axis broadening; `A_window` and `A_range` the
        margin around and the range of the real frequency axis.

    Returns (qpes, A, w_real, E_gm, mu): the peak positions, the spectral
    function, the real frequency axis it is on, the Galitskii-Migdal
    total energy of the converged G and Sigma (`galitskii_migdal_energy`)
    and the converged chemical potential.
    With skip_scf=True (and no ref_file), returns
    (G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D) instead.
    """
    if ref_file is None:
        out = _solve_qp_energy_scgw(mf, mol, nocc, skip_scf=skip_scf, method=method,
                                    grid_kind=grid_kind, save_file=save_file, **kwargs)
        if skip_scf:
            return out
        sigma, F, mu, omega_points, E_gm = out

    else:
        if not ref_file.endswith('.h5'):
            raise ValueError(f"ref_file must be an .h5 file, got '{ref_file}'")

        with h5py.File(ref_file, 'r') as f:
            if 'scgw' not in f:
                raise ValueError(f"No 'scgw' group in {ref_file}")
            keys = f['scgw'].keys()
            has_sigma, has_pade = 'sigma_mo' in keys, 'pade' in keys

        if not (has_sigma or has_pade):
            raise ValueError('Ref. file was given to skip calculation, but no '
                             f'sigma or pade weights found in {ref_file}')

        F            = lib.chkfile.load(ref_file, 'scgw/F_mo')
        mu           = lib.chkfile.load(ref_file, 'scgw/mu')
        omega_points = lib.chkfile.load(ref_file, 'scgw/omega_points')
        E_gm         = lib.chkfile.load(ref_file, 'scgw/E_gm')

        if not has_sigma:
            # Saved fit, used as is: no Sigma(i*w_n) to refit or check against.
            pade_Z, pade_A = unpack_pade(lib.chkfile.load(ref_file, 'scgw/pade'),
                                         omega_points, mu)
            return _finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm,
                                        save_file=save_file, **kwargs)

        sigma = lib.chkfile.load(ref_file, 'scgw/sigma_mo')

    # --- Real axis: Pade-continue Sigma, build G(omega), find peaks of A ---
    pade_max_order = 24
    z_fit = mu + 1j * omega_points

    # Fit once; _finish_qp_from_pade then evaluates Sigma(w) from the fit.
    pade_Z, pade_A = pade_fit_elementwise(z_fit, sigma, greedy=True,
                                          symmetric=True, max_order=pade_max_order)

    print(f'Pade check, max |Sigma_pade(i*w_n) - Sigma(i*w_n)|: '
        f'{np.max(np.abs(pade_eval_elementwise(z_fit, pade_Z, pade_A) - sigma))}\n')
    del sigma

    if save_file is not None and save_pade:
        # Next to the cycle's results; unpack_pade(d['pade'], d['omega_points'],
        # d['mu']) with d = lib.chkfile.load(save_file + '.h5', 'scgw') gives
        # back (Z, A) for trace_Greal_from_pade, e.g. to redo A at another eta.
        lib.chkfile.save(save_file + '.h5', 'scgw/pade',
                         pack_pade(pade_Z, pade_A, omega_points))

    return _finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm,
                                save_file=save_file, **kwargs)
