"""GW self-energy in imaginary time.

The standard space-time construction: Sigma = i G W is a pointwise product in
imaginary time, so the self-energy costs no convolution.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- the space-time
method itself.
Foerster and Visscher, J. Chem. Theory Comput. 16, 7381 (2020) -- the same
construction in a localized basis, with PADF as the factorization.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) and J. Chem. Theory
Comput. 17, 2383 (2021) -- the separable RI (ISDF) used here in place of PADF,
which is what makes the full self-energy build O(N^3).
"""
import os

import numpy as np

from src.Base.constants import ISDF_TILE_GB, SCREENED_CHUNK_BYTES
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (agreement, contiguous_block,
                                     current_comm, partition, reduce_sum)
from src.Base.utils.time_frequency import (DEFAULT_TAU_TARGET,
                                          minimax_transform_weights,
                                          minimax_points_for_accuracy,
                                          COSINE_TW, COSINE_WT, SINE_TW,
                                          SELF_ENERGY_PAD)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.space_time import (
    FrequencyBlock, ProjRows, owned_frequency_blocks,
    polarizability_projected_sweep, polarizability_projected_tau,
    split_branches, sweep_waves, three_index_slice, wave_items)


class SigmaPairs:
    """The (P block, Q block) pairs of the self-energy sweep over the
    interpolation points and what they contract: X_o, X_v and the states'
    columns X_s of the MO collocation X_ao C, and D. The blocks are the AO
    build's (`sigma_blocks`) and the far-field screen (coords, screen_r_cut)
    drops the same pairs, the points Morton-ordered first. X_ao is read once
    here and not kept.
    """

    def __init__(self, X_ao, D, mo_coeff, eps, nocc, states, mu,
                 block_memory_gb=ISDF_TILE_GB, coords=None, screen_r_cut=None):
        occ, virt = get_occ_virt_indices(eps, nocc)
        blocks = sigma_blocks(X_ao.shape[0], D.shape[1], block_memory_gb)
        far = None
        if coords is not None and screen_r_cut:
            idx = _morton_order(np.asarray(coords))
            X_ao = np.ascontiguousarray(X_ao[idx])
            D = np.ascontiguousarray(D[idx])
            far = _far_block_pairs(np.asarray(coords)[idx], blocks,
                                   screen_r_cut)
        X_mo = X_ao @ mo_coeff
        del X_ao
        # the branches, and the whole diagonal's states, are column ranges of
        # the one MO collocation: views, so a rank holds it once
        self.X_o, self.X_v = X_mo[:, :len(occ)], X_mo[:, len(occ):]
        states = np.atleast_1d(states)
        self.X_s = (X_mo if np.array_equal(states, np.arange(X_mo.shape[1]))
                    else np.ascontiguousarray(X_mo[:, states]))
        self.D, self.blocks = D, blocks
        self.e_o, self.e_v = eps[occ] - mu, eps[virt] - mu
        self.pairs = [(ip, iq) for ip in range(len(blocks))
                      for iq in range(len(blocks))
                      if far is None or not far[ip, iq]]

    def add(self, out, Wk, tau, js):
        """out[0] += Sigma^<_pp(tau) and out[1] += Sigma^>_pp(tau) of the
        pairs `js`, one after another in that order:

            Sigma^{<,>}_pp += sum_{P in p, Q in q} X_s[P,p] (Zt*G)_PQ X_s[Q,p]

        with Zt = D_p Wk D_q^T; D_p Wk is formed once per run of one P block.
        """
        eo_t, ev_t = np.exp(self.e_o * tau), np.exp(-self.e_v * tau)
        X_o, X_v, X_s, D = self.X_o, self.X_v, self.X_s, self.D
        last = None
        for j in js:
            ip, iq = self.pairs[j]
            (p0, p1), (q0, q1) = self.blocks[ip], self.blocks[iq]
            if ip != last:
                DW = D[p0:p1] @ Wk
                Xo_p, Xv_p = X_o[p0:p1] * eo_t, X_v[p0:p1] * ev_t
                last = ip
            Zt = DW @ D[q0:q1].T
            G = Xo_p @ X_o[q0:q1].T
            G *= Zt                                  # in place; Zt reused
            out[0] += np.einsum('Pp,Pp->p', X_s[p0:p1], G @ X_s[q0:q1])
            G = Xv_p @ X_v[q0:q1].T
            G *= Zt
            out[1] -= np.einsum('Pp,Pp->p', X_s[p0:p1], G @ X_s[q0:q1])
            del Zt, G


def greens_function_imaginary_time(X, eps, nocc, tau, mu=None):
    """
    (Ghat_lesser, Ghat_greater) projected onto the THC grid, each (M, M).
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    X_o, X_v = X[:, occ], X[:, virt]
    e_o, e_v = eps[occ] - mu, eps[virt] - mu
    G_lesser = (X_o * np.exp(e_o * tau)) @ X_o.T          # occupied branch
    G_greater = -(X_v * np.exp(-e_v * tau)) @ X_v.T       # virtual branch
    return G_lesser, G_greater


def self_energy_imaginary_time(X, D, W_tilde_aux_tau, eps, nocc, tau_points,
                               mu=None):
    """Sigma^c(i.tau) in the AO/MO basis X was built in, shape (ntau, n, n) x 2.

    W_tilde_aux_tau : (ntau, naux, naux) the correlation part of the screened
        interaction, W - V, already on the imaginary-time axis.

    Returns (Sigma_lesser, Sigma_greater). The frequency-axis self-energy
    follows by transforming their SUM with the cosine kernel and their
    DIFFERENCE with the sine kernel -- the even and odd parts of Sigma(i.tau)
    respectively. Both transforms are carried by TimeFrequencyGrid, which
    carries the reference for them.
    """
    n = X.shape[1]
    ntau = len(tau_points)
    sig_l = np.empty((ntau, n, n))
    sig_g = np.empty((ntau, n, n))
    for k, tau in enumerate(tau_points):
        G_l, G_g = greens_function_imaginary_time(X, eps, nocc, tau, mu=mu)
        Zt = D @ W_tilde_aux_tau[k] @ D.T                 # (M, M)
        sig_l[k] = X.T @ (Zt * G_l) @ X
        sig_g[k] = X.T @ (Zt * G_g) @ X
    return sig_l, sig_g


def self_energy_fit_ranges(eps, nocc, mu=None):
    """
    The two exponential-decay ranges the space-time self-energy needs.

    * Wt(i.tau): the RPA screened interaction has spectral weight BELOW the
      smallest independent-particle transition -- collective excitations sit
      under the HOMO-LUMO gap -- so a range starting at the gap misfits exactly
      where Wt is largest. Measured on the Wt(i.w) -> tau -> Wt(i.w) round trip
      at ntau=18: 1.3e-3 with [gap, e_max], 5.0e-8 with the widened range.
    * Sigma(i.tau) = -G Wt is a PRODUCT, so its decay rates are SUMS
      |eps_m - mu| + Omega_S and reach far beyond either factor's range.

    Returns ((w_lo, w_hi), (sig_lo, sig_hi)).
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    w_lo = eps[virt].min() - eps[occ].max()
    w_hi = eps[virt].max() - eps[occ].min()
    dG = np.abs(eps - mu)
    lo, hi = SELF_ENERGY_PAD
    return ((lo * w_lo, hi * w_hi),
            (lo * (dG.min() + w_lo), hi * (dG.max() + w_hi)))


def minimax_points_for_gw(eps, nocc, mu=None, target=DEFAULT_TAU_TARGET,
                          npoints_max=34):
    """Smallest minimax time grid that resolves every range the route integrates.

    The space-time route uses ONE ntau for three different fits, over three
    different energy ranges, and the widest one binds:

        chi0(i.tau) -> chi0(i.omega)   [e_min, e_max]        the bare gap ratio
        W(i.omega)  -> W(i.tau)        rW, widened below the gap
        Sigma(i.tau)-> Sigma(i.omega)  rS, widest -- Sigma is a PRODUCT, so its
                                       decay rates are SUMS (see
                                       `self_energy_fit_ranges`)

    R = e_max/e_min grows as the gap closes, so this necessarily returns more
    points for a longer acene than a shorter one at fixed accuracy: naphthalene
    needs 16 where hexacene needs 18. A hardcoded ntau is therefore wrong at one
    end or the other of any size series.

    Returns (npoints, worst_error). If no tabulated size reaches `target` the
    best available is returned instead, and the second value is the accuracy
    actually obtained -- callers should not assume `target` was met.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    ratios = ((eps[virt].max() - eps[occ].min())
              / (eps[virt].min() - eps[occ].max()),
              rW[1] / rW[0],
              rS[1] / rS[0])

    npoints, worst = 0, 0.0
    for R in ratios:
        n, err = minimax_points_for_accuracy(1.0, R, target=target,
                                             npoints_max=npoints_max)
        if n is None:                       # nothing tabulated resolved this R
            return npoints_max, float('inf')
        npoints, worst = max(npoints, n), max(worst, err)
    return npoints, worst


def screened_chunk(nfreq, naux, chunk_bytes=SCREENED_CHUNK_BYTES):
    """Frequencies per chunk of the omega -> tau transform of W - I."""
    return max(1, min(nfreq, int(chunk_bytes // max(naux * naux * 8, 1))))


def _transform_screened(Ctw, W_omega, chunk_bytes=SCREENED_CHUNK_BYTES):
    """
    sum_omega Ctw[.,omega] (W(i.omega) - I), without ever copying all of W.
    """
    nfreq, naux = W_omega.shape[0], W_omega.shape[-1]
    step = screened_chunk(nfreq, naux, chunk_bytes)
    dg = np.diag_indices(naux)
    out = np.zeros(Ctw.shape[:1] + (naux, naux))
    for k0 in range(0, nfreq, step):
        k1 = min(k0 + step, nfreq)
        blk = W_omega[k0:k1].copy()
        blk[(slice(None),) + dg] -= 1.0
        out += np.tensordot(Ctw[:, k0:k1], blk, axes=(1, 0))
        del blk
    return out


def screened_interaction_rows(W_minus_I, Ctw, naux, comm,
                              chunk_bytes=SCREENED_CHUNK_BYTES):
    """Wt(i.tau) = sum_w Ctw[t, w] (W(i.omega_w) - I) held by auxiliary rows,
    `ProjRows` (ntau, r1 - r0, naux), from each frequency's W - I whole on
    its round-robin owner (`partition`): `W_minus_I` maps this rank's
    frequencies to them.

    The frequencies are folded in the chunks of `_transform_screened`
    (`screened_chunk`, fixed by naux), the owners handing every rank its rows
    of the chunk and each rank adding the chunk into its rows one auxiliary
    row per call (`ProjRows.fold`): a call shape no rank count changes, so a
    rank's rows are the same bits at every rank count on any BLAS. No rank
    holds a whole (ntau, naux, naux) array or the product of a chunk.
    """
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    nfreq = Ctw.shape[1]
    r0, r1 = contiguous_block(naux, rank, size)
    Wt = ProjRows(np.zeros((Ctw.shape[0], r1 - r0, naux)), naux, comm)
    weights = np.ascontiguousarray(Ctw.T)                 # (nfreq, ntau)
    step = screened_chunk(nfreq, naux, chunk_bytes)
    for k0 in range(0, nfreq, step):
        block = list(range(k0, min(k0 + step, nfreq)))
        owners = [k % size for k in block]
        ks = [k for k, o in zip(block, owners) if o == rank]
        whole = [W_minus_I[k] for k in ks]
        if size == 1:
            whole = np.stack(whole)
        Wt.fold(weights, FrequencyBlock(block, owners, ks, whole))
    return Wt


def screened_interaction_tau_blocked(X, D, eps, nocc, grid, Ctw, mu=None,
                                     freq_block=None, scratch_dir=None,
                                     tile_memory_gb=ISDF_TILE_GB,
                                     wt_scratch=None, static_index=None,
                                     static_out=None, transform=None,
                                     tau_indices=None, tau_out_indices=None,
                                     comm=None):
    """
    Wt(i.tau) = sum_w Ctw[.,w] ( [I - chi0(i.w)]^-1 - I ), in one blocked pass.

    The in-core route builds all of chi0 (nfreq, naux, naux), inverts it in
    place, then transforms

    The catch is that every block needs all of proj(tau) again, and rebuilding
    those IS the N^3 cost of the method -- so recomputing them costs a factor
    nfreq/freq_block in time. `scratch_dir` avoids that by caching them.

    transform: `bare_gauge_transform`, when a reaction field dresses the
    factors. Wt is then the BARE screening -- the self-energy takes the
    continuum as Duchemin et al.'s static Eq. (18) shift instead, and screening
    it dynamically as well would count the same polarization twice. The
    captured `static_out['w_static']` stays DRESSED, because the BSE kernel it
    is carried for is built in the dressed gauge with the dressed factors.

    tau_indices / tau_out_indices / comm: THE TAU PARTITION, INSIDE EACH
    FREQUENCY BLOCK. This rank projects only the input points `tau_indices`
    into the block, the block is all-reduced over `comm` (nb x naux^2 per
    block, nfreq x naux^2 over the sweep -- the same volume the in-core route
    reduces, no more), every rank inverts the block, and this rank folds it
    into ONLY ITS OWN output rows `tau_out_indices`. So the M^2 sweep is
    divided, Wt is held rows-per-rank -- ntau/nranks x naux^2 instead of
    ntau x naux^2, which is what makes 55 GB at the 476-atom hexamer/cc-pVDZ
    fit beside anything -- and the proj(tau) cache holds this rank's points
    alone. The self-energy sweep downstream reads Wt[k] only for the tau
    points it owns, so the two partitions must coincide: hand it the same
    `tau_out_indices`. Serial (all None) is bitwise unchanged. With all three
    None the comm is `current_comm()`, and a context of several ranks splits
    the INPUT points here (`partition`) while every rank folds every output
    row, so the return is the whole Wt, identical on every rank; a partition
    handed in without a comm stays the caller's own partial, never reduced
    here. An audited run compares the inputs' digests, and on the way out
    those of W(omega = 0) and of a whole Wt -- rows held per rank differ by
    design and are not compared.

    X is the MO collocation, or `SlicedFactors` over `comm`, whose occupied
    and virtual columns `split_branches` gathers whole once for the sweep.

    Returns Wt(i.tau) with shape (Ctw.shape[0], naux, naux) when
    `tau_out_indices` is None, matching what `self_energy_matrix_imaginary_time`
    builds internally when Wt_tau is None; otherwise a dict
    {tau index: (naux, naux)} holding the owned rows, which every consumer
    indexes as Wt[k] exactly as before.
    """
    naux, nfreq, ntau = D.shape[1], grid.nfreq, grid.ntau
    nb = int(freq_block or nfreq)
    if comm is None and tau_indices is None and tau_out_indices is None:
        comm = current_comm()
        if comm is not None and comm.Get_size() > 1:
            tau_indices = partition(ntau, comm.Get_rank(), comm.Get_size())
    rank = 0 if comm is None else comm.Get_rank()
    nranks = 1 if comm is None else comm.Get_size()
    if isinstance(X, SlicedFactors):
        X.require(comm)
    X_o, X_v, e_o, e_v, mu, _, _ = split_branches(X, eps, nocc, mu)
    if nranks > 1:
        agreement((X_o, X_v, D, eps, Ctw, transform), comm, audit_only=True,
                  label='screened_interaction_tau_blocked inputs')
    which_in = range(ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    rows_out = (np.arange(Ctw.shape[0]) if tau_out_indices is None
                else np.atleast_1d(tau_out_indices))
    if wt_scratch is not None:
        os.makedirs(os.path.dirname(wt_scratch) or '.', exist_ok=True)
        Wt = np.lib.format.open_memmap(
            wt_scratch, mode='w+', dtype=np.float64,
            shape=(len(rows_out), naux, naux))
        Wt[:] = 0.0
    else:
        Wt = np.zeros((len(rows_out), naux, naux))
    eye = np.eye(naux)
    dg = np.diag_indices(naux)

    cache, path = None, None
    if scratch_dir is not None:
        os.makedirs(scratch_dir, exist_ok=True)
        # per rank: two ranks on one node must not share a cache file
        path = os.path.join(scratch_dir, f'proj_{ntau}_{naux}_r{rank}.npy')
        cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                          shape=(ntau, naux, naux))
    cached = False
    try:
        for k0 in range(0, nfreq, nb):
            k1 = min(k0 + nb, nfreq)
            blk = np.zeros((k1 - k0, naux, naux))
            for j in which_in:
                if cached:
                    proj = cache[j]
                else:
                    proj = polarizability_projected_tau(
                        X_o, X_v, e_o, e_v, D, grid.tau_points[j],
                        tile_memory_gb=tile_memory_gb)
                    if cache is not None:
                        cache[j] = proj
                blk += grid.cosft_wt[k0:k1, j, None, None] * proj
            if cache is not None:
                cached = True
            if nranks > 1:
                reduce_sum(blk, comm)         # every rank now holds chi0(block)
            for m in range(k1 - k0):
                b = blk[m]
                if static_index is not None and k0 + m == static_index:
                    static_out['w_static'] = np.linalg.inv(eye - b)   # dressed
                if transform is not None:
                    b[:] = transform.T @ b @ transform     # chi0 into the bare gauge
                b[:] = np.linalg.inv(eye - b)
                b[dg] -= 1.0                  # the correlation part, W - I
            # ONE OUTPUT TAU AT A TIME. `Wt += tensordot(Ctw[:, k0:k1], blk)`
            # materializes a temporary the size of Wt itself
            for i, t in enumerate(rows_out):
                Wt[i] += np.tensordot(Ctw[t, k0:k1], blk, axes=(0, 0))
            del blk
    finally:
        if cache is not None:
            del cache
            if path and os.path.exists(path):
                os.remove(path)
    if wt_scratch is not None:
        Wt.flush()
        Wt = np.lib.format.open_memmap(wt_scratch, mode='r')
    if nranks > 1:
        agreement(((static_out or {}).get('w_static'),
                   Wt if tau_out_indices is None else None), comm,
                  audit_only=True,
                  label='screened_interaction_tau_blocked outputs')
    if tau_out_indices is None:
        return Wt
    return {int(t): Wt[i] for i, t in enumerate(rows_out)}


def _morton_order(coords):
    """Z-order interpolation points so a contiguous block is spatially compact.

    The grid is built atom by atom, and atom order in a geometry file is not
    spatial, so contiguous index blocks otherwise straddle the whole molecule
    and no distance screen can bite.
    """
    c = coords - coords.min(axis=0)
    scale = c.max()
    if scale <= 0:
        return np.arange(len(c))
    bits = 21                                     # 3 * 21 fits one uint64
    q = np.minimum((c / scale * (2**bits - 1)).astype(np.uint64), 2**bits - 1)
    key = np.zeros(len(c), dtype=np.uint64)
    for b in range(bits):
        for d in range(3):
            key |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + d)
    return np.argsort(key)


def sigma_blocks(M, naux, block_memory_gb=ISDF_TILE_GB):
    """[(p0, p1)] interpolation-point blocks of the self-energy sweep, b rows
    each so that one (b, naux) and two (b, b) blocks fit the budget: set by
    M, naux and the budget alone."""
    b = int((-naux + np.sqrt(naux**2 + 8 * block_memory_gb * 1e9 / 8)) / 4)
    b = max(1, min(M, b))
    edges = list(range(0, M, b)) + [M]
    return list(zip(edges[:-1], edges[1:]))


def _far_block_pairs(points, blocks, screen_r_cut):
    """far[ip, iq]: blocks whose bounding spheres are further apart than
    screen_r_cut (Bohr), on Morton-ordered points."""
    cen = np.array([points[p0:p1].mean(axis=0) for p0, p1 in blocks])
    rad = np.array([np.linalg.norm(points[p0:p1] - c, axis=1).max()
                    for (p0, p1), c in zip(blocks, cen)])
    sep = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    return sep - rad[:, None] - rad[None, :] > screen_r_cut


def self_energy_matrix_imaginary_time(X_ao, D, W_omega, mo_coeff, eps, nocc,
                                      tau_points, omega_in, omega_out,
                                      mu=None, ranges=None, tau_indices=None,
                                      Wt_tau=None,
                                      block_memory_gb=ISDF_TILE_GB,
                                      coords=None, screen_r_cut=None):
    """Full Sigma^c_{mu nu}(i.omega) in the AO basis, shape (nfreq, nao, nao).

    The whole chain in one call, with the outer contraction keeping both AO
    indices:

        Sigma_{mu nu}(tau) = sum_PQ X_ao[P,mu] (Zt_PQ * Ghat_PQ) X_ao[Q,nu]

    Two collocation matrices are involved and they are NOT interchangeable.
    Ghat needs the occupied/virtual split, so it is built from the MO
    collocation X_mo = X_ao @ mo_coeff; the outer indices are AO, so they use
    X_ao. Passing an MO-basis X for both silently returns Sigma in the MO basis
    instead.

    Cost per tau is O(M^2 N + M N^2): two GEMMs and a Hadamard product for the
    WHOLE matrix, not per element. That is the O(N^3)-for-everything claim --
    the frequency route needs O(N^3) per state, so it only matches this for a
    handful of states.

    Wt_tau: the screened interaction already on the time grid. Pass it when the
    caller built W blockwise (`screened_interaction_tau_blocked`) and W_omega --
    (nfreq, naux, naux), the largest array on this route -- was never formed at
    all; W_omega is then ignored.

    block_memory_gb caps the per-block working set. It does not change the
    answer or the flop count, only the peak allocation.

    coords + screen_r_cut (Bohr) drop (P, Q) block pairs whose bounding spheres
    are further apart than the cutoff. Sigma^{<,>}_PQ decays in |r_P - r_Q| --
    G at a rate set by the gap, Wt because it is screened -- so the surviving
    fraction falls with system size, which is where the cubic scaling is
    actually realized. A magnitude bound is NOT usable here: Cauchy-Schwarz on
    ||G[P]|| ||G[Q]|| discards the row overlap that carries the decay, and
    measured on benzene it drops 34% of pairs where only 3.7% are negligible.

    tau_indices selects a subset of tau points and returns that subset's
    contribution; the full Sigma is the sum over subsets, since the tau -> omega
    transform is a sum over tau. Distributing tau across ranks and reducing is
    therefore exact, with each rank threading its own GEMMs -- the same
    two-level split `chi0_imaginary_frequency` supports.

    Memory is (len(omega_out), nao, nao) complex -- the returned array and
    little else; the time points are folded in as they are built. Project it
    down with one of

        sigma_ao_to_mo           full MO matrix     -> qsGW, scGW
        sigma_ao_to_mo_diagonal  MO diagonal        -> evGW, G0W0

    and discard, or loop over frequency blocks, if that does not fit.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = ranges or self_energy_fit_ranges(eps, nocc, mu=mu)

    if Wt_tau is None:
        Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, omega_in,
                                           *rW, warn=False)
        Wt_tau = _transform_screened(Ctw, W_omega)
    # else the caller built it blockwise and W_omega was never formed at all

    X_mo = X_ao @ mo_coeff
    X_o, X_v = X_mo[:, occ], X_mo[:, virt]
    e_o, e_v = eps[occ] - mu, eps[virt] - mu

    ntau, nao = len(tau_points), X_ao.shape[1]
    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS,
                                     warn=False)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS,
                                     warn=False)

    # STREAMED OVER TAU, not staged. The tau -> omega transform is a sum over
    # tau, so each time point can be folded into the output as soon as it is
    # built and never stored. Staging Sigma^{<,>}(tau) first would hold two
    # (ntau, nao, nao) arrays, and forming `sig_g +/- sig_l` for the transform
    # two more -- and with both grids on 'auto' nfreq == ntau, so those are the
    # same size as the result itself. Measured peak was 6.1 (n, nao, nao)
    # float64 stacks against the 2 this array actually needs.
    # zeros, not empty: with tau_indices set only the owned points contribute
    # and the rest must add nothing, since the caller reduces over subsets.
    out = np.zeros((len(omega_out), nao, nao), dtype=complex)
    # BLOCKED OVER THE INTERPOLATION INDEX. Zt, G_l and G_g are each (M, M),
    # and the unblocked form holds three of them plus the Hadamard temporary --
    # 159 GB at M = 70448 (the 476-atom hexamer/cc-pVDZ), which the OOM killer
    # ended. Every term is a sum over P, so cutting that index costs nothing:
    # the three GEMMs become (b, M) row slabs, the outer contraction
    # accumulates into a (nao, M) buffer and multiplies by X_ao once at the end.
    # Flop counts are identical term by term; only the working set changes.
    M, naux = X_ao.shape[0], D.shape[1]
    pairs = sigma_blocks(M, naux, block_memory_gb)
    # Spatial order first, then block: the screen is geometric, so the blocks
    # have to be. Permuting P is a relabelling of a summation index and leaves
    # Sigma unchanged.
    far = None
    if coords is not None and screen_r_cut:
        idx = _morton_order(np.asarray(coords))
        X_ao = np.ascontiguousarray(X_ao[idx])
        D = np.ascontiguousarray(D[idx])
        X_mo = X_ao @ mo_coeff
        X_o, X_v = X_mo[:, occ], X_mo[:, virt]
        far = _far_block_pairs(np.asarray(coords)[idx], pairs, screen_r_cut)

    which = range(ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        tau = tau_points[k]
        Wk = Wt_tau[k]
        Xo_t = X_o * np.exp(e_o * tau)
        Xv_t = X_v * np.exp(-e_v * tau)

        # Cauchy-Schwarz bound per interpolation point, so a block pair whose
        # product cannot reach the tolerance is skipped without being built.
        # |G^<_PQ| <= ||X_o[P] e^{e tau/2}|| ||X_o[Q] e^{e tau/2}|| and
        # |Zt_PQ| <= ||D[P]|| ||Wt||_F ||D[Q]||, the Frobenius norm standing in
        # for the spectral one so the bound stays cheap. Rigorous, not heuristic:
        # a skipped pair is bounded, never estimated.
        T_l = np.zeros((nao, M))
        T_g = np.zeros((nao, M))
        for ip, (p0, p1) in enumerate(pairs):
            DW = D[p0:p1] @ Wk
            Xa_p = X_ao[p0:p1].T
            for iq, (q0, q1) in enumerate(pairs):
                if far is not None and far[ip, iq]:
                    continue
                Zt = DW @ D[q0:q1].T
                G = Xo_t[p0:p1] @ X_o[q0:q1].T
                G *= Zt                              # in place; Zt is reused
                T_l[:, q0:q1] += Xa_p @ G
                G = Xv_t[p0:p1] @ X_v[q0:q1].T
                G *= Zt
                T_g[:, q0:q1] -= Xa_p @ G
                del Zt, G
            del DW
        s_l = T_l @ X_ao
        s_g = T_g @ X_ao
        del T_l, T_g
        even_k, odd_k = s_g + s_l, s_g - s_l
        # One frequency at a time: np.multiply.outer(C[:, k], even_k) would be
        # correct but allocates a whole (nfreq, nao, nao) temporary per tau,
        # which is the thing being avoided.
        for w in range(len(omega_out)):
            out.real[w] += C[w, k] * even_k
            out.imag[w] += S[w, k] * odd_k
    out *= -0.5
    return out


def self_energy_diagonal_rows(pairs, Wt, eps, nocc, tau_points, omega_out,
                              mu=None, ranges=None):
    """Sigma^c_pp(i.omega_out) for the states of `pairs` (`SigmaPairs`),
    (nstates, nfreq), whole on every rank, from Wt(i.tau) held by auxiliary
    rows (`ProjRows` over the ranks).

    `self_energy_matrix_imaginary_time`'s sweep with the (tau point, block
    pair) items split over the ranks in `sweep_waves` windows instead of the
    tau points alone, so it divides past ntau ranks: a rank gathers the at
    most two Wt(tau) slices its items read and adds each pair's branch sums
    straight to the diagonal (`SigmaPairs.add`), so neither the (nao, M)
    accumulators of the AO build nor the (nfreq, nao, nao) AO matrix is
    formed. The branch sums, (2, ntau, nstates), are the one reduction -- a
    sum re-associated over the ranks' items -- and the tau -> omega transform
    follows on every rank.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    _, rS = ranges or self_energy_fit_ranges(eps, nocc, mu=mu)
    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS,
                                     warn=False)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS,
                                     warn=False)
    sig_l, sig_g = self_energy_branch_sums(pairs, Wt, tau_points)
    return -0.5 * ((sig_g + sig_l).T @ C.T + 1j * ((sig_g - sig_l).T @ S.T))


def self_energy_branch_sums(pairs, Wt, tau_points):
    """(Sigma^<_pp(tau), Sigma^>_pp(tau)), (2, ntau, nstates), whole on every
    rank: the (tau point, block pair) items in `sweep_waves` windows, each
    rank's items added in their order and the ranks' partials reduced."""
    size, rank = Wt._size_rank()
    ntau = len(tau_points)
    sig = np.zeros((2, ntau, pairs.X_s.shape[1]))    # lesser, greater branch
    for k0, k1 in sweep_waves(ntau, size):
        mine = wave_items(k0, k1, len(pairs.pairs), rank, size)
        need = sorted({k for k, _ in mine})
        slabs = {k: slab.copy() for k, slab in Wt.gather_slices(need)}
        for k in need:
            pairs.add(sig[:, k], slabs.pop(k), tau_points[k],
                      [j for kk, j in mine if kk == k])
    return reduce_sum(sig, Wt.comm)


def sigma_ao_to_mo(sigma_ao, mo_coeff):
    """Full MO-basis Sigma, (nfreq, nmo, nmo) -- what qsGW and scGW need."""
    return np.einsum('mp,wmn,nq->wpq', mo_coeff, sigma_ao, mo_coeff,
                     optimize=True)


def sigma_ao_to_mo_diagonal(sigma_ao, mo_coeff, states=None):
    """MO-diagonal Sigma_pp, (nfreq, nstates) -- what evGW needs.

    Never forms the full MO matrix, so it stays O(nfreq nao^2 nstates).
    """
    C = mo_coeff if states is None else mo_coeff[:, states]
    return np.einsum('mp,wmn,np->wp', C, sigma_ao, C, optimize=True)


def screened_interaction_tau(proj_tau, grid, Ctw, tile_gb=ISDF_TILE_GB,
                             comm=None):
    """Wt(tau) = sum_w Ctw[t, w] ([I - chi0_w]^-1 - I) from proj(tau), (ntau_out, naux, naux).

    W - I is folded into imaginary time one frequency block at a time, so no
    (nfreq, naux, naux) array exists; the transform weights Ctw are the
    omega -> tau half of `sigma_transforms`.

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep. proj(tau) arrives whole, so the only work left is the
    naux^3 inversion at each frequency and frequency is the compute axis here;
    tau_out carries no inversion and splitting it would replicate every one of
    them. Unlike the contour-deformation contraction this axis is not
    reduction-free -- Wt(tau) is a SUM over frequencies -- so the split ends in
    one all-reduce of the result, ntau_out x naux^2. Each inversion is the
    serial one bitwise, its chi0 rows cut from the serial block
    (`owned_frequency_blocks`), so only that sum re-associates. None is
    `current_comm()`. proj(tau) arrives identical on every rank by
    construction and is not broadcast; an audited run compares its digest and
    Wt's (`mpi_grid.agreement`).
    """
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((proj_tau, Ctw), comm, audit_only=True,
                  label='screened_interaction_tau inputs')
    nu_mine = partition(grid.nfreq, rank, nranks) if nranks > 1 else None
    Wt_tau = np.zeros((Ctw.shape[0], naux, naux))
    for ks, blk in owned_frequency_blocks(proj_tau, grid.cosft_wt, tile_gb,
                                          nu_mine):
        for m in range(len(ks)):
            blk[m] = np.linalg.inv(eye - blk[m]) - eye
        Wt_tau += np.tensordot(Ctw[:, ks], blk, axes=(1, 0))
    if nranks > 1:
        reduce_sum(Wt_tau, comm)
        agreement(Wt_tau, comm, audit_only=True,
                  label='screened_interaction_tau outputs')
    return Wt_tau


def selfenergy_block(X, D, eps, nocc, grid, states, transforms, mu,
                     intermediate=None, tile_gb=ISDF_TILE_GB, comm=None):
    """Sigma^c_pq(i.omega_out) for p, q in `states`: the self-energy MATRIX on a
    block, (nfreq_out, nstates, nstates), by the space-time route.

    `selfenergy_diag`'s construction with both bra and ket free,

        Sigma^<_pq(tau) = sum_i e^{e_i tau} B_p[:,i]^T Wt(tau) B_q[:,i],
        Sigma^>_pq(tau) = -sum_a e^{-e_a tau} B_p[:,a]^T Wt(tau) B_q[:,a],

    one (naux, norb) GEMM per tau and bra state. `intermediate` restricts the
    summed index i, a to a set of orbitals: with the environment orbitals of an
    active window it is Sigma_c^{G^E W}, the embedding self-energy in which the
    intermediate state lies outside the window and the screening is the full
    system's (Sheng et al., JCTC 18, 3512 (2022)). Real orbitals make
    Sigma_pq(tau) real, so every element obeys Sigma(z*) = Sigma(z)* and is
    continued like a diagonal one.

    Returns (Sigma, cache) as `selfenergy_diag` does; the cache holds proj(tau),
    Wt(tau) and the state slices B_p, which is what the reverse pass
    (`gradients.space_time_adjoint.selfenergy_block_backward`) reads.

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep. Three sweeps, each on the axis that carries its own
    O(N^3) work: the M^2 polarizability sweep over tau (one reduction of
    proj(tau), ntau x naux^2, over disjoint slots and so bitwise), the Dyson
    inversions over frequency inside `screened_interaction_tau` (one reduction
    of Wt(tau), ntau x naux^2), and the naux^2 norb self-energy GEMMs over tau
    again (one reduction of Sigma^< and Sigma^> together, ntau x nstates^2).
    The cache leaves reduced, so every rank reverses through the same tape.
    None is `current_comm()`; the inputs are identical by construction and an
    audited run compares their digests and the outputs' (`mpi_grid.agreement`).
    """
    Ctw, C, S = transforms
    _, _, e_o, e_v, _, occ, virt = split_branches(X, eps, nocc, mu)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((X, D, eps, transforms, states, intermediate), comm,
                  audit_only=True, label='selfenergy_block inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    proj_tau = polarizability_projected_sweep(X, D, eps, nocc, grid.tau_points,
                                              mu=mu, tau_indices=tau_mine,
                                              tile_memory_gb=tile_gb)
    if nranks > 1:
        reduce_sum(proj_tau, comm)            # the others' slots are zero
    Wt_tau = screened_interaction_tau(proj_tau, grid, Ctw, tile_gb=tile_gb,
                                      comm=comm)

    states = np.atleast_1d(states)
    n_s = len(states)
    keep = np.ones(len(eps), bool)
    if intermediate is not None:
        keep[:] = False
        keep[np.atleast_1d(intermediate)] = True
    w_occ = np.where(keep[occ], 1.0, 0.0)
    w_virt = np.where(keep[virt], 1.0, 0.0)
    Bs = np.stack([three_index_slice(X, D, int(s), tile_gb=tile_gb)
                   for s in states])                        # (nstates, naux, norb)
    ntau = len(grid.tau_points)
    sig_l = np.zeros((ntau, n_s, n_s))
    sig_g = np.zeros((ntau, n_s, n_s))
    for k in (range(ntau) if tau_mine is None else tau_mine):
        tau = grid.tau_points[k]
        w, u = np.exp(e_o * tau) * w_occ, np.exp(-e_v * tau) * w_virt
        for s in range(n_s):
            Y = Wt_tau[k] @ Bs[s]                            # (naux, norb)
            for t in range(n_s):
                bYb = np.einsum('Pq,Pq->q', Bs[t], Y)
                sig_l[k, t, s] = bYb[occ] @ w
                sig_g[k, t, s] = -(bYb[virt] @ u)
    if nranks > 1:
        sig_l, sig_g = reduce_sum(np.stack([sig_l, sig_g]), comm)
    sigma = -0.5 * (np.tensordot(C, sig_g + sig_l, axes=(1, 0))
                    + 1j * np.tensordot(S, sig_g - sig_l, axes=(1, 0)))
    if nranks > 1:
        agreement((sigma, Bs), comm, audit_only=True,
                  label='selfenergy_block outputs')
    return sigma, (proj_tau, Wt_tau, Bs)


def selfenergy_diag(X, D, eps, nocc, grid, states, transforms, mu,
                    tile_gb=ISDF_TILE_GB, comm=None):
    """Sigma^c_pp(i.omega_out) by the space-time route, and the tape it needs.

    The diagonal of `self_energy_matrix_imaginary_time`, built in the
    auxiliary basis: W - I is folded into Wt(tau) one frequency block at a
    time, then per tau and state one (naux, norb) GEMM. Returns (Sigma,
    cache); the cache holds proj(tau), Wt(tau) and the state slices B_p --
    (naux, naux) and (naux, norb) objects only.

    comm: `selfenergy_block`'s three splits with one index of Sigma fixed --
    tau for the polarizability sweep and for the self-energy GEMMs, frequency
    for the Dyson inversions; the reductions are proj(tau), Wt(tau) and the
    two branch sums. None is `current_comm()`; an audited run compares the
    digests of the inputs and outputs, as `selfenergy_block` does.
    """
    Ctw, C, S = transforms
    _, _, e_o, e_v, _, occ, virt = split_branches(X, eps, nocc, mu)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((X, D, eps, transforms, states), comm, audit_only=True,
                  label='selfenergy_diag inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    proj_tau = polarizability_projected_sweep(X, D, eps, nocc, grid.tau_points,
                                              mu=mu, tau_indices=tau_mine,
                                              tile_memory_gb=tile_gb)
    if nranks > 1:
        reduce_sum(proj_tau, comm)            # the others' slots are zero
    Wt_tau = screened_interaction_tau(proj_tau, grid, Ctw, tile_gb=tile_gb,
                                      comm=comm)

    states = np.atleast_1d(states)
    Bs = np.stack([three_index_slice(X, D, int(s), tile_gb=tile_gb)
                   for s in states])                        # (nstates, naux, norb)
    ntau = len(grid.tau_points)
    sig_l = np.zeros((ntau, len(states)))
    sig_g = np.zeros((ntau, len(states)))
    for k in (range(ntau) if tau_mine is None else tau_mine):
        tau = grid.tau_points[k]
        w, u = np.exp(e_o * tau), np.exp(-e_v * tau)
        for s in range(len(states)):
            bYb = np.einsum('Pq,Pq->q', Bs[s], Wt_tau[k] @ Bs[s])
            sig_l[k, s] = bYb[occ] @ w
            sig_g[k, s] = -(bYb[virt] @ u)
    if nranks > 1:
        sig_l, sig_g = reduce_sum(np.stack([sig_l, sig_g]), comm)
    sigma = -0.5 * ((sig_g + sig_l).T @ C.T + 1j * ((sig_g - sig_l).T @ S.T))
    if nranks > 1:
        agreement((sigma, Bs), comm, audit_only=True,
                  label='selfenergy_diag outputs')
    return sigma, (proj_tau, Wt_tau, Bs)


