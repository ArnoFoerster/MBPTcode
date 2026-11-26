"""Correlation self-energy of scGW on IR grids, Sigma(i.omega) from a dressed
G(i.omega):

    G(i.omega) -> G(tau) -> chi(tau) -> chi(i.omega) -> W(i.omega) -> W(tau)
               -> Sigma(tau) = G(tau) W(tau) -> Sigma(i.omega)

with the two-electron integrals either as separable ISDF/THC factors
(`calc_new_sigma_isdf`) or as a plain RI three-index tensor (`build_B_ri`,
`calc_new_sigma_ri`). `calc_new_sigma` picks one. G and Sigma live on the
fermion IR grid, chi and W on the boson grid.
"""
import numpy as np
from pyscf import df, lib

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


def tau_projection_matrix(src_grid, dst_grid, parity):
    """Matrix P that interpolates f from src_grid's tau points onto
    dst_grid's, f_dst[t'] = sum_t P[t, t'] f_src[t], shape (ntau_src, ntau_dst).

    The fermion and boson grids share beta but sample different tau points,
    and the products chi = G*G and Sigma = G*W need both factors at the same
    tau. parity: 'even' or 'odd', the symmetry sector of f. The projection is
    linear in f, so P is its image of the identity."""
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

def static_W(Wc_omega, grid_b):
    """W(i.omega = 0) in the auxiliary basis, a new (naux, naux) array, from
    W - I on the boson grid (`W_minus_I_inplace`). The boson IR sampling
    always contains omega = 0; it is looked up rather than assumed first."""
    k0 = int(np.argmin(np.abs(grid_b.omega_points)))
    if grid_b.omega_points[k0] != 0.0:
        raise ValueError('boson grid has no omega = 0 point')
    W0 = Wc_omega[k0].copy()
    W0[np.diag_indices_from(W0)] += 1.0
    return W0

def calc_new_sigma_isdf(G, grid_f, grid_b, X_mo, D, return_all=False, return_W0=False):
    """Correlation self-energy Sigma(i.omega) from a dressed G(i.omega).

    Steps: G -> chi = G*G -> W = [I - chi]^-1 -> Sigma = G*W. The products
    are taken pointwise in imaginary time on the ISDF/THC grid; the Dyson
    inversion for W is done in frequency. G and Sigma live on the fermion
    grid, chi and W on the boson grid, and `tau_projection_matrix` moves
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
    return_W0: also return the static W(i.omega = 0), (naux, naux) in the
        auxiliary basis of D (`static_W`). It costs one naux^2 copy, so it
        can be asked for on every call.

    Returns sigma_new, (sigma_new, W0) with return_W0=True, or with
    return_all=True (G_tau, chi_omega, W_omega, sigma_new):
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
    W0 = static_W(Wc_omega, grid_b) if return_W0 else None

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
        return (sigma_new, W0) if return_W0 else sigma_new

    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)
    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    return (G_lesser_f + 1j * G_greater_f), chi_out, Wc_omega + np.eye(naux), sigma_new

def calc_new_sigma_ri(G, grid_f, grid_b, B, return_all=False, block_size=None,
                      return_W0=False):
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
    return_W0: also return the static W(i.omega = 0), (naux, naux) in the
        auxiliary basis of B, so that (pq|W(0)|rs) = sum_PQ B[P,p,q] W0[P,Q] B[Q,r,s].
    block_size: P per block; default from RI_BLOCK_MB.

    Returns as `calc_new_sigma_isdf`.
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
    W0 = static_W(Wc_omega, grid_b) if return_W0 else None

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
        return (sigma_new, W0) if return_W0 else sigma_new

    G_even_f = np.einsum('tk,kpq->tpq', grid_f.cosft_tw, G.real)
    G_odd_f = np.einsum('tk,kpq->tpq', grid_f.sinft_tw, G.imag)
    G_lesser_f = G_even_f - G_odd_f
    G_greater_f = G_even_f + G_odd_f
    return (G_lesser_f + 1j * G_greater_f), chi_out, Wc_omega + np.eye(naux), sigma_new

def calc_new_sigma(G, grid_f, grid_b, X_mo=None, D=None, B=None, method='ISDF',
                   return_all=False, return_W0=False):
    """Correlation self-energy Sigma(i.omega), by `calc_new_sigma_isdf`
    (method='ISDF', needs X_mo and D) or `calc_new_sigma_ri` (method='RI',
    needs B). Returns what the chosen routine returns.
    """
    method = method.upper()
    if method == 'ISDF':
        if X_mo is None or D is None:
            raise ValueError("method='ISDF' needs X_mo and D")
        return calc_new_sigma_isdf(G, grid_f, grid_b, X_mo, D, return_all=return_all,
                                   return_W0=return_W0)
    if method == 'RI':
        if B is None:
            raise ValueError("method='RI' needs B")
        return calc_new_sigma_ri(G, grid_f, grid_b, B, return_all=return_all,
                                 return_W0=return_W0)
    raise ValueError(f"Unknown method {method!r}, expected 'ISDF' or 'RI'")
