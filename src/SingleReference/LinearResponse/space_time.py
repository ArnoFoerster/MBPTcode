"""Space-time RPA screening: the polarizability in imaginary time, then one
cosine transform to the imaginary-frequency axis.

With a separable (ISDF) ERI,

    (ia|jb) = sum_PQ X_o[P,i] X_v[P,a] Z[P,Q] X_o[Q,j] X_v[Q,b]

the particle-hole bubble separates into an occupied and a virtual half that each
carry only ONE orbital index,

    Pi_PQ(i.tau) = G^o_PQ(tau) * G^v_PQ(tau)          (elementwise)
    G^o_PQ(tau)  = sum_i X_o[P,i] X_o[Q,i] e^{+eps_i tau}
    G^v_PQ(tau)  = sum_a X_v[P,a] X_v[Q,a] e^{-eps_a tau}

so a tau point costs O(M^2 (n_occ + n_vir)) -- two GEMMs and a Hadamard product
-- against O(naux^2 n_occ n_vir) per frequency for the direct summation in
`imaginary_frequency.py`. That is the N^3-against-N^4 step, crossing over near
120 basis functions.

The transform is used one way only, on the model space it is fitted for
(Pi(i.tau) is a sum of e^{-Delta_ia tau} with Delta_ia in [e_min, e_max]), so
the minimax transform's lack of matrix duality costs nothing.

Sign convention follows `imaginary_frequency._f_rpa`: f(i.w) = -2 d/(d^2 + w^2)
and chi0 = 2 (C_ov f) C_ov^T; the cosine transform maps e^{-d tau} to
2d/(d^2 + w^2), hence the leading -2 below.

ONE KERNEL FOR BOTH SIDES OF THE CODE. `polarizability_projected_tau` is the
N^3 sweep, and it is the only implementation of it: the energy routes stream
it into chi0 here, and a gradient chain would run it over every tau point
through `polarizability_projected_sweep` and keep the result whole, because
every contour-deformation frequency is a fixed linear combination of proj(tau).
A change to the arithmetic or the tiling therefore reaches every caller.

`three_index_slice`, `b_block` and `three_index_ov` sit here for the same
reason: one bra state's pair density, an arbitrary index block and the
particle-hole block are the forward halves of the three-index chain, read by
the contour-deformation energy route and the downfolded model alike.

`rpa_correlation_energy_space_time` is the dRPA energy's own forward sweep, and
`want_tape=True` hands back what a reverse pass through it would consume --
proj(tau) and the two partitions it was built on.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- building the
response in imaginary time, where the occupied and virtual sums decouple.
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) -- the
minimax imaginary-time/frequency quadrature the transform below runs on.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) -- the separable RI
supplying X_o and X_v.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.constants import ISDF_TILE_GB
from src.SingleReference.base import get_occ_virt_indices


@dataclass(frozen=True)
class RPAEnergyTape:
    """The forward sweep of E_c^dRPA, as a reverse pass would read it.

    proj_tau is the one whole object of the N^3 route, (ntau, naux, naux), from
    which every frequency is a fixed linear combination; the two index sets are
    the tau points and the frequencies a distributed sweep would own -- both
    None in this serial-only build.
    """

    proj_tau: np.ndarray
    tau_indices: np.ndarray
    freq_indices: np.ndarray


def polarizability_imaginary_time(X_o, X_v, eps_o, eps_v, tau_points,
                                  out=None, beta=None):
    """Pi_PQ(i.tau) on the interpolation grid, shape (ntau, M, M).

    X_o, X_v :     (M, n_occ) and (M, n_vir) collocation, occupied and virtual.
    eps_o, eps_v : orbital energies SHIFTED so every eps_v - eps_o > 0; any
                   chemical potential inside the gap does this.
    beta :         inverse temperature, giving the bosonic periodic object
                   Pi(tau) + Pi(beta - tau) that a Matsubara/IR grid needs.
                   Omit for the T = 0 half-line function of the minimax grids.

    The mirror term is not a small correction: e^{i nu_n beta} = 1 for bosonic
    frequencies, so tau -> beta - tau maps the integral onto itself and the
    mirror contributes exactly as much as the direct term. Dropping it is a
    factor of two at every beta.
    """
    M = X_o.shape[0]
    ntau = len(tau_points)
    if out is None:
        out = np.empty((ntau, M, M))
    for k, tau in enumerate(tau_points):
        Go = (X_o * np.exp(eps_o * tau)) @ X_o.T
        Gv = (X_v * np.exp(-eps_v * tau)) @ X_v.T
        np.multiply(Go, Gv, out=out[k])
        if beta is not None:
            tb = beta - tau
            out[k] += (((X_o * np.exp(eps_o * tb)) @ X_o.T)
                       * ((X_v * np.exp(-eps_v * tb)) @ X_v.T))
    return out


def tile_rows(M, tile_memory_gb, per_row_bytes):
    """Grid rows per tile so that `per_row_bytes` x rows fits in
    tile_memory_gb: at least one row, at most all M of them."""
    return max(1, min(M, int(tile_memory_gb * 1e9 / max(per_row_bytes, 1))))


def polarizability_work(M, tile_memory_gb=ISDF_TILE_GB):
    """The two (rows, M) Green's-function tiles `polarizability_projected_tau`
    works in, sized for this M and budget, to be reused across tau points."""
    rows = tile_rows(M, tile_memory_gb, 3 * M * 8)
    return np.empty((rows, M)), np.empty((rows, M))


def split_branches(X, eps, nocc, mu=None):
    """(X_o, X_v, e_o, e_v, mu, occ, virt): the collocation split at the gap.

    The energies come back shifted by mu, which defaults to the middle of the
    gap. Any chemical potential inside the gap makes every e_v - e_o positive,
    which is what keeps both exponentials in the Green's functions decaying.
    The occupied and virtual blocks are made contiguous here, once, so the
    GEMMs downstream never copy them.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    return (np.ascontiguousarray(X[:, occ]), np.ascontiguousarray(X[:, virt]),
            eps[occ] - mu, eps[virt] - mu, mu, occ, virt)


def polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                 tile_memory_gb=ISDF_TILE_GB, out=None,
                                 work=None):
    """chi0 at ONE imaginary time, already projected to the auxiliary basis:

        proj_ab(tau) = -2 sum_PQ D[P,a] (Go_PQ Gv_PQ) D[Q,b]

    Tiled over grid rows, which is exact because the expression is a sum over
    them, so the M x M object never exists. Contract the Pi block with D first:
    the other order builds an (naux, M) intermediate instead.

    THE ONE KERNEL OF THE N^3 SWEEP, for the energy routes alike -- see the
    module docstring. The row budget here, 3 * M * 8 bytes per row, is a
    convention any reverse-mode kernel mirroring this tiling would have to
    match.

    out:  (naux, naux) buffer the result is written into, so a sweep over many
          tau points allocates nothing per point. Zeroed here; allocated when
          None.
    work: the two (rows, M) tiles from `polarizability_work`, reused across
          calls for the same reason. Their contents are never read.
    """
    M = X_o.shape[0]
    naux = D.shape[1]
    rows = tile_rows(M, tile_memory_gb, 3 * M * 8)
    if work is None:
        work = polarizability_work(M, tile_memory_gb)
    Go_buf, Gv_buf = work
    for buf in (Go_buf, Gv_buf):
        if buf.shape[0] < rows or buf.shape[1] != M:
            raise ValueError(f'work tile is {buf.shape}; this M and budget '
                             f'need at least ({rows}, {M})')
    if out is None:
        out = np.zeros((naux, naux))
    else:
        if out.shape != (naux, naux):
            raise ValueError(f'out is {out.shape}, need ({naux}, {naux})')
        out[...] = 0.0
    eo_t, ev_t = np.exp(e_o * tau), np.exp(-e_v * tau)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        b = p1 - p0
        Go, Gv = Go_buf[:b], Gv_buf[:b]
        np.matmul(X_o[p0:p1] * eo_t, X_o.T, out=Go)   # (b, M)
        np.matmul(X_v[p0:p1] * ev_t, X_v.T, out=Gv)   # (b, M)
        Go *= Gv                                       # Pi block, in Go's tile
        out += D[p0:p1].T @ (Go @ D)                   # (naux, naux)
    out *= -2.0
    return out


def polarizability_projected_sweep(X, D, eps, nocc, tau_points, mu=None,
                                   tau_indices=None,
                                   tile_memory_gb=ISDF_TILE_GB):
    """proj(tau) on every time point, (ntau, naux, naux): the one whole object
    of the N^3 route.

    `chi0_imaginary_frequency` folds each point into chi0 as it is built and
    never holds this array. A caller that needs it whole (e.g. a gradient
    chain, since every contour-deformation frequency is a fixed linear
    combination of it) gets the same kernel, kept instead of streamed.

    tau_indices: the points to compute; the other slots stay zero, so the sum
    over disjoint subsets is the full sweep.
    """
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    tau_points = np.asarray(tau_points, float)
    naux = D.shape[1]
    proj_tau = np.zeros((len(tau_points), naux, naux))
    work = polarizability_work(X.shape[0], tile_memory_gb)
    which = (range(len(tau_points)) if tau_indices is None
             else np.atleast_1d(tau_indices))
    for k in which:
        polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau_points[k],
                                     tile_memory_gb=tile_memory_gb,
                                     out=proj_tau[k], work=work)
    return proj_tau


def chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=None, stream=True,
                             tau_indices=None, tile_memory_gb=ISDF_TILE_GB):
    """chi0(i.omega) in the DF auxiliary basis, shape (nfreq, naux, naux).

    X :    (M, norb) collocation in the MO basis.
    D :    (M, naux) Coulomb factor, Z = D D^T; carries the result back to the
           auxiliary basis every DF consumer speaks.
    grid : TimeFrequencyGrid with an imaginary-time axis.

    Same chi0 as `solve_rpa_screening_df`, so W follows as [I - chi0]^-1.

    stream=True projects and accumulates each Pi(i.tau) immediately. Identical
    algebraically, since

        chi0(i.w) = -2 sum_tau cosft_wt[w,tau] (D^T Pi(tau) D),

    but the peak drops from (ntau, M, M) to one (M, M).
    """
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)

    if not stream:
        Pi_tau = polarizability_imaginary_time(X_o, X_v, e_o, e_v,
                                               grid.tau_points)
        Pi_w = np.tensordot(grid.cosft_wt, Pi_tau, axes=(1, 0))
        return -2.0 * np.einsum('Pa,wPQ,Qb->wab', D, Pi_w, D, optimize=True)

    naux = D.shape[1]
    chi0 = np.zeros((grid.nfreq, naux, naux))
    # One projected point and the two Green's-function tiles, allocated once
    # for the whole sweep and handed to the kernel every time.
    proj = np.empty((naux, naux))
    work = polarizability_work(X.shape[0], tile_memory_gb)
    which = range(grid.ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        polarizability_projected_tau(X_o, X_v, e_o, e_v, D, grid.tau_points[k],
                                     tile_memory_gb=tile_memory_gb, out=proj,
                                     work=work)
        chi0 += grid.cosft_wt[:, k, None, None] * proj
    return chi0


def frequency_blocks(nfreq, naux, tile_gb, live):
    """(k0, k1) ranges over the frequency axis such that `live` (naux, naux)
    arrays per frequency fit in tile_gb. The one knob for both the M-row tiles
    and the frequency blocks: a working-set budget, not a count."""
    nb = max(1, min(nfreq, int(tile_gb * 1e9 / max(live * naux * naux * 8, 1))))
    return [(k0, min(k0 + nb, nfreq)) for k0 in range(0, nfreq, nb)]


def three_index_slice(X, D, p, tile_gb=ISDF_TILE_GB):
    """B_p[P, q] = sum_k D[k,P] X[k,p] X[k,q]: one bra state's pair density in
    the auxiliary basis, (naux, norb), in O(M naux norb). Tiled over grid rows
    so the (M, norb) product X[:,p] * X never exists whole either."""
    M, norb = X.shape
    Bp = np.zeros((D.shape[1], norb))
    rows = tile_rows(M, tile_gb, norb * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Bp += D[p0:p1].T @ (X[p0:p1, p, None] * X[p0:p1])
    return Bp


def b_block(X, D, p_idx, q_idx, Y=None):
    """B[P, p, q] = sum_k X[k,p] Y[k,q] D[k,P] for one index block, Y = X by default.

    Written as one GEMM per bra function rather than a single einsum. The flops
    are identical; the library path is not. numpy's einsum falls off BLAS for
    a three-operand contraction carrying a batch index and runs it at ~2
    GFlop/s, where the same work as matrix products reaches 60-85 -- measured,
    a factor of FORTY. Looping the outer index also keeps the working set at
    (M, n_q) instead of the (M, n_p n_q) an einsum path materializes, which at
    production sizes is the difference between fitting and not.
    """
    Xp, Xq = X[:, p_idx], (X if Y is None else Y)[:, q_idx]
    out = np.empty((D.shape[1], Xp.shape[1], Xq.shape[1]))
    for a in range(Xp.shape[1]):
        out[:, a, :] = D.T @ (Xp[:, a, None] * Xq)
    return out


def three_index_ov(X, D, eps, nocc, tile_gb=ISDF_TILE_GB):
    """C_ov[P, (i,a)] = sum_k D[k,P] X[k,i] X[k,a], (naux, nocc*nvirt).

    The particle-hole block of the three-index tensor -- the O(N^4) object a
    contour deformation's RESIDUE term needs, because W at a real frequency has
    no imaginary-time form and is built from it explicitly. Frontier states
    sweep no residues and never need it. Tiled so the (rows, nocc*nvirt) pair
    block stays inside tile_gb.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    M = X.shape[0]
    n_ov = len(occ) * len(virt)
    C = np.zeros((D.shape[1], n_ov))
    rows = tile_rows(M, tile_gb, n_ov * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Xo, Xv = X[p0:p1][:, occ], X[p0:p1][:, virt]
        C += D[p0:p1].T @ (Xo[:, :, None] * Xv[:, None, :]).reshape(p1 - p0, n_ov)
    return C


def owned_frequency_blocks(proj_tau, cosft_wt, tile_gb, freq_indices=None,
                           live=3):
    """(ks, chi0 block) over the frequency axis, block by block.

    ks are the frequencies of the block this caller owns -- all of them when
    `freq_indices` is None -- and the block is sum_tau cosft_wt[ks, tau]
    proj_tau for exactly those, so a caller never transforms a frequency it
    will not use and a block with no owned frequency is skipped outright.
    """
    nfreq, naux = cosft_wt.shape[0], proj_tau.shape[-1]
    owned = (None if freq_indices is None
             else set(int(k) for k in np.atleast_1d(freq_indices)))
    for k0, k1 in frequency_blocks(nfreq, naux, tile_gb, live=live):
        ks = (list(range(k0, k1)) if owned is None
              else [k for k in range(k0, k1) if k in owned])
        if not ks:
            continue
        yield ks, np.tensordot(cosft_wt[ks], proj_tau, axes=(1, 0))


def laplace_representation_error(grid, eps, nocc, freq):
    """max over y = d +- freq of |y sum_k w_k e^{-y tau_k} - 1|: how well the
    grid's bare quadrature carries this real frequency, on the pair energies
    that actually occur. inf when freq reaches the gap (a real pole).

    The gate on the imaginary-time form of W at a REAL frequency, which is what
    a contour-deformation residue below the particle-hole gap asks for:
    -2d/(d^2 - w^2) = -int 2 cosh(w tau) e^{-d tau} dtau holds only where the
    grid still represents e^{-y tau} on every y = d -/+ w.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    d = (np.asarray(eps)[virt][None, :] - np.asarray(eps)[occ][:, None]).ravel()
    y = np.concatenate([d - freq, d + freq])
    if y.min() <= 0.0:
        return np.inf
    q = np.exp(-np.outer(y, grid.tau_points)) @ grid.tau_weights
    return float(np.abs(q * y - 1.0).max())


def screening_space_time(X, D, eps, nocc, grid, mu=None):
    """W(i.omega) = [I - chi0(i.omega)]^-1, via the imaginary-time route."""
    chi0 = chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=mu)
    eye = np.eye(chi0.shape[-1])
    return np.array([np.linalg.inv(eye - c) for c in chi0])


def rpa_correlation_energy_space_time(X, D, eps, nocc, grid, mu=None,
                                      tile_gb=ISDF_TILE_GB, screening=None,
                                      counter_term=None, want_tape=False):
    """dRPA correlation energy E_c = (1/2pi) int dw Tr{log(1 - chi0) + chi0}.

    The quadrature of `rpa_energy.rpa_correlation_energy_imaginary_axis`, with
    chi0 from the imaginary-time route. The frequencies come out of proj(tau)
    one block at a time (`owned_frequency_blocks`), so the axis is never whole
    and the peak is proj(tau) plus one block.

    screening = (N, g): the interaction that builds the LOGARITHM is rescaled
    to v + g(w) vtilde while D stays in the dressed gauge. In that gauge the
    rescaling is a similarity on chi0 alone -- with N = V_d^(-1/2) vtilde
    V_d^(-1/2), V_d = V + vtilde, and S_w = I - (1 - g_w) N,

        log det(I - S_w c_w) = log det(I - (v + g_w vtilde) P_1(iw)) ,

    so g == 1 is the dressed interaction (the default) and g == 0 the bare one.
    g is one scalar per frequency, N one naux x naux matrix.
    counter_term = C: the LINEAR term becomes Tr(C c_w) in place of Tr(c_w).
    C = I - N puts the BARE interaction there, which is what an exact block
    fold of the log-determinant over a non-overlapping solvent keeps, and what
    leaves the leading solute-solvent dispersion term in the energy instead of
    cancelling it (`solvated_rpa_energy.fold_terms` chooses the pair).

    want_tape: also return the `RPAEnergyTape` a reverse pass would read.
    """
    n_mat, g = (None, None) if screening is None else screening
    proj_tau = polarizability_projected_sweep(X, D, eps, nocc, grid.tau_points,
                                              mu=mu, tile_memory_gb=tile_gb)
    eye = np.eye(proj_tau.shape[-1])
    e_c = 0.0
    for ks, blk in owned_frequency_blocks(proj_tau, grid.cosft_wt, tile_gb):
        for m, k in enumerate(ks):
            c0 = blk[m]
            linear = (np.trace(c0) if counter_term is None
                      else float(np.einsum('pq,qp->', counter_term, c0)))
            # S_w c_w, the argument of the logarithm. `one_minus_g` is 0 for the
            # dressed interaction, so S = I and c is untouched there.
            one_minus_g = 0.0 if n_mat is None else 1.0 - g[k]
            c = c0 if n_mat is None else c0 - one_minus_g * (n_mat @ c0)
            _, logdet = np.linalg.slogdet(eye - c)
            e_c += grid.omega_weights[k] * (logdet + linear)
    e_c /= 2.0 * np.pi
    if not want_tape:
        return e_c
    return e_c, RPAEnergyTape(proj_tau, None, None)
