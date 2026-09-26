"""Space-time GW: the O(N^3) route to a quasiparticle energy.

The polarizability is built in imaginary time from a separable (ISDF)
factorization of the ERIs, where the occupied and virtual sums decouple, and
the self-energy Sigma = -G Wt is a pointwise product there rather than a
convolution.

Peer of the Casida route (`qp_energy.calc_qp_energy`) and the
imaginary-frequency route (`imaginary_axis.solve_qp_energy_imaginary_axis`);
all three end in `qp_solve.solve_qp_from_imaginary_axis`.

Four silent traps, all handled here:

  * W must come from the SAME factors as Sigma. pyscf's cderi is L^-1-whitened,
    a separable RI fits with the symmetric V^-1/2, and mixing the two gauges
    moves the QP energy by ~1.4 eV while staying self-consistent.
  * The tau grid uses the SELF-ENERGY's energy range, not the polarizability's:
    Sigma is a product, so its decay rates are sums.
  * The tau and frequency axes are decoupled; one grid cannot serve both the
    tau->omega transform and the Sigma quadrature.
  * Occupied states sample the negative branch, so the Pade input is conjugated.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- the space-time
method: Sigma = i G W as a pointwise product in imaginary time rather than a
convolution in frequency, which is what makes this route cubic.
Duchemin and Blase, J. Chem. Theory Comput. 17, 2383 (2021) -- the same
construction on a separable RI in a Gaussian basis, which this follows.
Foerster and Visscher, J. Chem. Theory Comput. 16, 7381 (2020) -- low-scaling
G0W0 in a localized basis, the pair-fitting ancestor of this route.
"""
import os
import time as _time
import warnings

import numpy as np
from pyscf import df as pyscf_df

from src.Base.constants import ISDF_RADII_MATCH_TOL, ISDF_TILE_GB
from src.Base.environment import environment_of
from src.Base.pyscf_interface import get_orbital_energies
from src.Base.separable_ri import (DEFAULT_PAIR_TOL, atomic_grid,
                                   aux_metric_sqrt,
                                   build_separable_ri, fit_M_streaming,
                                   molecular_points_covariant,
                                   optimize_atomic_radii, resolve_isdf_grid,
                                   shipped_radii_lookup)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_frequency_grid,
                                  minimax_supported_sizes)
from src.Base.utils.mpi_grid import (agreement, broadcast_rows,
                                     current_comm, grid_comm, lockstep,
                                     lockstep_mean_field, partition,
                                     reduce_sum)
from src.Base.utils.time_frequency import (TimeFrequencyGrid, COSINE_WT,
                                           minimax_transform_weights)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.imaginary_time import (SigmaPairs,
                                                   self_energy_matrix_imaginary_time,
                                                   self_energy_diagonal_rows,
                                                   sigma_ao_to_mo_diagonal,
                                                   self_energy_fit_ranges,
                                                   screened_interaction_rows,
                                                   screened_interaction_tau_blocked,
                                                   minimax_points_for_gw,
                                                   DEFAULT_TAU_TARGET)
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             solve_qp_from_imaginary_axis,
                                             imaginary_axis_sample_points)
from src.SingleReference.GW.reaction_field import (
    separable_gauge_transform, separable_quasiparticle_shift)
from src.SingleReference.LinearResponse.space_time import (
    chi0_frequency_rows, chi0_imaginary_frequency)

DEFAULT_NTAU = 'auto'
DEFAULT_NFREQ = 'auto'
DEFAULT_NPADE = 16
DEFAULT_COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}


def separable_factors(mf, mol, auxbasis=None, radii=None, counts=None,
                      block_memory_gb=4.0, pair_tol=DEFAULT_PAIR_TOL,
                      n_start=1, grid_accuracy=None, comm=None, timings=None,
                      sliced=False, fit='replicated', fit_block=None):
    """(X_mo, D, X_ao, coords) of the Duchemin-Blase separable RI, Z = D D^T.

    X_ao is the collocation the fit actually produces; X_mo = X_ao C is the
    form most consumers want. Both are returned because inverting one back to
    the other needs a square C, which a large basis does not guarantee.

    Interpolation points come from covariant atomic frames, so the answer does
    not depend on the orientation of the molecule.

    THE GRID IS ONE OBJECT, so a keyword naming it is never dropped. Three
    keywords can name it and each combination has one outcome:

      grid_accuracy alone   `resolve_isdf_grid` sets `counts` and `n_start`.
      grid_accuracy+counts  equal -> proceed; different -> ValueError. One
                            request cannot be two grids, and resolving to
                            either side hands back a factorization nobody
                            asked for.
      counts alone          tabulated -> the row; not tabulated -> a warning
                            and a run-time re-optimization onto another local
                            minimum of a multi-modal surface, which is a grid
                            no campaign scored.
      radii alone           the radii ARE the grid; no row is consulted.
      radii+counts          where a row exists for those counts the two must
                            agree to `ISDF_RADII_MATCH_TOL` or the call is
                            refused, and the row's `origin` is honoured -- it
                            places one extra point at the nucleus, so dropping
                            it builds a 306-point carbon grid where a
                            Duchemin-Blase row describes 307.
      nothing               `DEFAULT_COUNTS`, sized for double zeta.

    grid_accuracy:   an accuracy level of `ISDF_GRID_ACCURACY` or four explicit
                     shell counts, resolved by `resolve_isdf_grid`, which
                     refuses anything the radii table has not got. It sets
                     `counts` AND `n_start`, since a validated row is keyed on
                     both.
    radii:           per-element shell radii; optimized per element if omitted.
    block_memory_gb: caps the working set of the fit's blocked loops. It does
                     not change the answer, but it IS a speed knob as well as a
                     memory one -- one `aux_e2` call per block, each rebuilding
                     a shell-pair list over nbas x auxnbas. See `build_D_F`,
                     which carries the measurement; size it from the node.
    comm:            spreads the fit's three-centre pass over ranks
                     (`build_separable_ri`); `current_comm()` when None, serial
                     without either. The call is RANK 0's on every rank twice
                     over: at entry the mean-field arrays it reads and the
                     points it placed are locksteps of rank 0's -- each rank
                     converged its own SCF, and a run-time radius
                     re-optimization or the `eigh` of the atomic frames can
                     place the points elsewhere on another node -- and at exit
                     the factors are (`replicate_factors`), since the fit's
                     replicated Cholesky tail is dense arithmetic that need not
                     repeat bitwise across nodes. Every consumer downstream
                     therefore receives identical factors and does not
                     broadcast them again.
    timings:         dict, filled at phase boundaries on every rank, same
                     pattern as `solve_qp_energy_space_time`. `fit_points` is
                     this function's own interpolation-point build
                     (`molecular_points_covariant`); `fit_collocation`,
                     `fit_integrals`, `fit_integrals_reduce`, `fit_gram`,
                     `fit_cholesky`, `fit_solve`, `fit_blocks` and
                     `fit_blocks_total` come from `fit_M_streaming` through
                     `build_separable_ri`, unchanged; `fit_assembly` is that
                     function's X tail PLUS this function's own tail (the
                     dressed metric, the MO/AO projections, replication) --
                     the two add into one key rather than each keeping a
                     phase the call graph does not actually separate.
                     `fit_total` is this call, start to return, on whichever
                     rank asked. Radius lookup or run-time re-optimization,
                     before `fit_points`, has no phase of its own: it is a
                     table read on every tabulated grid and is not timed. The
                     clock reads touch no bit of the factors.
    sliced:          over more than one rank, return a `SlicedFactors`: this
                     rank's contiguous block of the grid rows of X_mo, D and
                     X_ao, cut from the whole, lockstepped products after the
                     whole arrays are formed exactly as above, so a consumer
                     that gathers one gets the replicated array bit for bit
                     (`sliced_factors` carries the table of who reads what).
                     Ignored serially and on one rank, where the tuple is
                     already all there is.
    fit:             'replicated' (the default above) or 'rows': the fit
                     distributed by grid rows (`separable_ri.fit_rows`), so
                     that no rank holds the Gram matrix, F D^T, the solve, the
                     collocation or any factor whole. It returns
                     `SlicedFactors` over more than one rank whatever `sliced`
                     says -- there is no whole array to cut -- and the tuple
                     on one rank. Its rows are bitwise identical at every rank
                     count and are not the replicated fit's bits: a different
                     realization of the same estimator, within a few of the
                     replicated fit's own reassociation responses. `timings`
                     carries the same keys; `fit_assembly` is the metric, the
                     projections and the move of D to the contiguous rows.
    fit_block:       grid points per tile of fit='rows' (`FIT_CHOLESKY_BLOCK`
                     when None); any fixed value is bitwise across rank
                     counts, two values differ by rounding.
    """
    if fit not in ('replicated', 'rows'):
        raise ValueError(f"fit={fit!r}: 'replicated' or 'rows'")
    _t0 = _time.time()
    comm = current_comm() if comm is None else comm
    auxbasis = auxbasis or (str(mol.basis) + '-ri')
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    if grid_accuracy is not None:
        level_counts, n_start = resolve_isdf_grid(grid_accuracy, mol.basis,
                                                  elements, auxbasis=auxbasis)
        if counts is not None and ({k: int(v) for k, v in dict(counts).items()}
                                   != {k: int(v) for k, v in level_counts.items()}):
            raise ValueError(
                f'two grids asked for: counts {dict(sorted(dict(counts).items()))} '
                f'and grid_accuracy {grid_accuracy!r}, which is '
                f'{dict(sorted(level_counts.items()))} at {mol.basis}. Pass one '
                f'or the other -- the level is not a hint that a count may '
                f'override, and every energy built on the grid not asked for '
                f'would be of a functional nobody requested.')
        counts = level_counts
    named_counts = counts is not None
    counts = counts or DEFAULT_COUNTS

    if radii is None:
        radii, origins = {}, {}
        for el in elements:
            # One table, one lookup, at the counts asked for. Rows carrying the
            # published cc-pVTZ grids live in it at their own counts, so
            # reaching them means asking for them.
            try:
                radii[el], origins[el] = atomic_grid(el, mol.basis, auxbasis,
                                                     counts)
            except KeyError as no_row:
                # No row: the request is honoured by re-optimizing, and the
                # warning carries the counts held for this atom because the
                # miss is almost always a tuple nobody optimized.
                # `resolve_isdf_grid` is the gate that refuses this outright; a
                # caller who reached here passed explicit counts.
                warnings.warn(
                    f'ISDF grid re-optimized at run time: {no_row.args[0]} '
                    f'Re-optimizing lands on another local minimum of a '
                    f'multi-modal surface, so this grid is not one any '
                    f'campaign scored and is not reproducible from a clean '
                    f'checkout.', RuntimeWarning, stacklevel=2)
                radii[el] = optimize_atomic_radii(el, mol.basis, auxbasis,
                                                  counts=counts,
                                                  n_start=n_start)[0]
                origins[el] = False
    else:
        # Explicit radii ARE the grid and are honoured. Where the caller ALSO
        # named counts the table may describe the same grid, and then the two
        # specifications have to agree: `origin` belongs to the row, not to the
        # recipe, so a matching row brings its nuclear point with it.
        origins = {el: False for el in radii}
        against_table = sorted(set(elements) & set(radii)) if named_counts else []
        for el in against_table:
            hit = shipped_radii_lookup(el, str(mol.basis), str(auxbasis), counts)
            if hit is None:
                continue                 # no row at these counts: nothing to contradict
            table_radii, _, origins[el] = hit
            for shell in sorted(set(table_radii) | set(radii[el])):
                mine = np.atleast_1d(np.asarray(radii[el].get(shell, []),
                                                dtype=float))
                theirs = np.atleast_1d(np.asarray(table_radii.get(shell, []),
                                                  dtype=float))
                if (mine.shape != theirs.shape
                        or np.any(np.abs(mine - theirs) > ISDF_RADII_MATCH_TOL)):
                    off = ('' if mine.shape != theirs.shape else
                           f', worst by {np.abs(mine - theirs).max():.3g} Bohr')
                    raise ValueError(
                        f'{el} radii contradict the shipped grid at counts '
                        f'{dict(sorted(dict(counts).items()))}: this call passes '
                        f'{shell}={np.array2string(mine, precision=12)} where the '
                        f'table row holds '
                        f'{np.array2string(theirs, precision=12)} (they must '
                        f'agree to {ISDF_RADII_MATCH_TOL:g} Bohr{off}). Drop '
                        f'`radii` to build the tabulated row, or drop `counts` '
                        f'to build the radii you passed; the two are different '
                        f'grids and nothing here can choose between them.')

    _t = _time.time()
    coords = molecular_points_covariant(mol, radii, origin_by_element=origins)
    if timings is not None:
        timings['fit_points'] = _time.time() - _t
    if comm is not None and comm.Get_size() > 1:
        # One packed call: the orbitals X_mo is projected on, and the grid in
        # which the radii and the frames this rank decided are realized.
        # Checked: only an array some rank holds differently is broadcast.
        mf.mo_energy, mf.mo_coeff, mf.mo_occ, coords = lockstep(
            (mf.mo_energy, mf.mo_coeff, mf.mo_occ, coords), comm, check=True)
    if fit == 'rows':
        return _row_factors(mf, mol, auxmol, coords, block_memory_gb, pair_tol,
                            comm, timings, fit_block, _t0)
    # Z = M^T V M is never read here: D = M^T V^1/2 below carries the metric.
    X, _, M = build_separable_ri(mol, coords, auxmol=auxmol,
                                 block_memory_gb=block_memory_gb,
                                 pair_tol=pair_tol, comm=comm, timings=timings,
                                 with_Z=False)

    # Symmetric V^1/2 gauge on the auxiliary index -- the one Sigma expects,
    # dressed by the environment attached to the mean field (v -> v + vtilde).
    _t = _time.time()
    V_half = aux_metric_sqrt(auxmol, environment_of(mf))
    out = replicate_factors((X @ mf.mo_coeff, M.T @ V_half, X, coords), comm)
    if sliced and comm is not None and comm.Get_size() > 1:
        del X, M
        out = SlicedFactors.from_whole(out, comm)
    if timings is not None:
        timings['fit_assembly'] = (timings.get('fit_assembly', 0.0)
                                   + (_time.time() - _t))
        timings['fit_total'] = _time.time() - _t0
    return out


def _row_factors(mf, mol, auxmol, coords, block_memory_gb, pair_tol, comm,
                 timings, fit_block, t0):
    """`separable_factors`' fit='rows' tail: the row-distributed fit, then
    X_mo, D = M^T V^1/2 and X_ao on this rank's contiguous rows, each read in
    the fit's own tiles; `SlicedFactors` over more than one rank."""
    row_fit = fit_M_streaming(mol, auxmol, coords,
                              block_memory_gb=block_memory_gb,
                              pair_tol=pair_tol, comm=comm, timings=timings,
                              fit='rows', block=fit_block)
    _t = _time.time()
    many = comm is not None and comm.Get_size() > 1
    X_mo = row_fit.mo_rows(mf.mo_coeff)
    # the metric root is formed on rank 0 alone and streamed in slabs
    D = row_fit.metric_root_rows(auxmol, environment_of(mf))
    X_ao = row_fit.ao_rows()
    out = (SlicedFactors.from_rows(X_mo, D, X_ao, coords, comm,
                                   fit_held=row_fit.held)
           if many else (X_mo, D, X_ao, coords))
    out = replicate_factors(out, comm)
    if timings is not None:
        timings['fit_assembly'] = _time.time() - _t
        timings['fit_total'] = _time.time() - t0
    return out


def _unpack_factors(factors):
    """(X_mo, D, X_ao, coords); the last two are None for a shorter tuple."""
    return tuple(factors) + (None,) * (4 - len(factors))


def _sliced_solve_factors(factors, comm, transform):
    """(X_mo, D, X_ao, coords) as one quasiparticle solve reads `SlicedFactors`.

    D is read whole by both sweeps, chi0 and the self-energy, so it is
    gathered once here and held for the solve. The collocations stay sliced
    and are handed on as the factors themselves: the chi0 sweep gathers its
    two branches (`split_branches`), the self-energy sweep X_ao
    (`_sigma_mo_diagonal`), each once and each gone when its sweep returns.
    The continuum's Eq. (18) shift reads X_mo whole beside them and is
    refused rather than paid for with a third collocation gather.
    """
    factors.require(comm)
    if transform is not None:
        raise ValueError(
            'sliced factors under a reaction field: the Eq. (18) shift and the '
            'bare-gauge D read X_mo and D whole beside both sweeps; build the '
            'factors unsliced for a solvated run')
    return factors, factors.gather('D'), factors, factors.coords


def replicate_mean_field(mf, comm):
    """`mpi_grid.lockstep_mean_field` under the name its callers import.

    The implementation lives in `mpi_grid` so that replicating a mean field
    imports nothing from GW. Here `comm` stays explicit: None or one rank is
    serial and never falls back to `current_comm()`.
    """
    if comm is None or comm.Get_size() == 1:
        return mf
    return lockstep_mean_field(mf, comm)


def replicate_factors(factors, comm):
    """Rank 0's separable fit onto every rank, in place: one `lockstep` of its
    arrays. None or one rank is serial.

    The factors are the kernel output every downstream kernel takes as
    identical by construction, and `separable_factors` makes them so here: the
    fit's replicated Cholesky tail is dense arithmetic, and rows of Zt reduced
    from two different fits are not a sum of anything.

    Checked (`mpi_grid.lockstep(check=True)`): a fit on lockstepped orbitals
    and points repeats bitwise across identical nodes -- the audited
    eight-node pentacene cc-pVTZ fit found no rank apart here -- so digests
    stand in for the 167 MB broadcast there (~47 GB at the chlorophyllide
    hexamer), and only an array some rank holds differently is sent.

    `SlicedFactors` differ between ranks by design: they are cut from arrays
    `separable_factors` already locked whole, so only the grid points they
    were cut on are checked, and a layout other than theirs is refused.
    """
    if isinstance(factors, SlicedFactors):
        lockstep(factors.require(comm).coords, comm, check=True)
        return factors
    if comm is None or comm.Get_size() == 1:
        return factors
    lockstep(tuple(a for a in _unpack_factors(factors)
                   if isinstance(a, np.ndarray)), comm, check=True)
    return factors


def _ao_collocation(X_mo, mf):
    """X_ao[k, mu] = chi_mu(r_k), inverted from the MO collocation X_mo = X_ao C.

    FALLBACK ONLY -- `separable_factors` now returns X_ao directly, because the
    inversion is not always available. It is exact where it works: MO
    coefficients are S-orthonormal, C^T S C = I, so C^-1 = C^T S. But it needs a
    square C, and a mean field that dropped linear dependencies gives only a
    left inverse, i.e. a silent projection. cc-pVQZ on the 178-atom
    chlorophyllide dimer is past that line -- cond(S) = 1.8e7 with four
    eigenvalues below 1e-6 -- so the AO route there must take X_ao from the fit.
    """
    C = mf.mo_coeff
    if C.shape[0] != C.shape[1]:
        raise ValueError(
            f'mo_coeff is {C.shape[0]}x{C.shape[1]}: the mean field dropped '
            f'{C.shape[0] - C.shape[1]} linearly dependent combinations, so the '
            'MO collocation cannot be inverted back to AO exactly. Build the '
            'factors in the AO representation instead.')
    return X_mo @ C.T @ mf.get_ovlp()


def _dyson_in_place(chi0, rows, static_index=None, transform=None):
    """W(i.omega_k) = [I - chi0(i.omega_k)]^-1 over chi0's rows `rows`, in place.

    Diagonal in frequency: each row is one naux^3 inversion that reads no other
    row. The omega = 0 passenger `static_index` is inverted once, DRESSED, since
    it is the BSE kernel's static screening and outside the Sigma quadrature;
    every other row takes chi0 into the bare gauge first when `transform` is set.
    """
    eye = np.eye(chi0.shape[-1])
    for k in rows:
        if k == static_index:
            chi0[k] = np.linalg.inv(eye - chi0[k])
            continue
        if transform is not None:
            chi0[k] = transform.T @ chi0[k] @ transform
        chi0[k] = np.linalg.inv(eye - chi0[k])
    return chi0


def _dyson_owned(chi0, static_index=None, transform=None):
    """({frequency: W - I}, W(omega = 0)) from chi0 held by auxiliary rows
    (`ProjRows` over the frequencies): W - I whole for this rank's round-robin
    frequencies alone, W(0) whole on every rank (None without a passenger).

    Each frequency's chi0 is gathered whole to its owner and inverted there as
    `_dyson_in_place` inverts it -- the passenger dressed, every other
    frequency in the bare gauge when `transform` is set -- so each W is the
    serial inversion of its chi0, bitwise. The passenger is broadcast from
    its owner; it is no part of the Sigma quadrature.
    """
    size, rank = chi0._size_rank()
    naux = chi0.naux
    eye, dg = np.eye(naux), np.diag_indices(naux)
    W_minus_I, w_static = {}, None
    for k, c in chi0.gather_slices(partition(chi0.shape[0], rank, size)):
        if k == static_index:
            w_static = np.linalg.inv(eye - c)        # dressed: the kernel's
            continue
        if transform is not None:
            c = transform.T @ c @ transform
        W = np.linalg.inv(eye - c)
        W[dg] -= 1.0                                  # the correlation part
        W_minus_I[k] = W
    c = W = eye = None                          # before the broadcast lands
    if static_index is not None:
        if w_static is None:
            w_static = np.empty((naux, naux))
        broadcast_rows(w_static, static_index % size, chi0.comm)
    return W_minus_I, w_static


def _qp_grid_rows(X_mo, D, X_ao, coords, mf, mol, eps, nocc, mu, grid,
                  tau_points, freq_points, pade_freq, p_state, want_static,
                  extras, ntau, comm, transform, solver_mode, dm_correction,
                  greedy, timings, screen_r_cut, sigma_x, eps_anchor,
                  sigma_x_matrix, tile_gb):
    """The in-core route over more than one rank, split over grid rows as
    well as tau points, so that it divides past ntau ranks and no rank holds a
    whole (M, M) block, a whole stack of (naux, naux) slices or the AO
    self-energy:

        chi0(i.nu)   this rank's auxiliary rows    `chi0_frequency_rows`
        W(i.nu)      whole on its owner alone      `_dyson_owned`
        Wt(i.tau)    this rank's auxiliary rows    `screened_interaction_rows`
        Sigma_pp     whole on every rank           `self_energy_diagonal_rows`

    Two sums are reduced and re-associate over the ranks: proj(tau) over the
    grid-row tiles and the self-energy's branch sums over its block pairs.
    Every other step is an output partition: chi0's rows the serial update of
    their proj rows, each W the serial inversion of its chi0, Wt's rows one
    call shape at every rank count, each root the serial solve of its Sigma.
    """
    _t = _time.time()
    chi0 = chi0_frequency_rows(X_mo, D, eps, nocc, grid, mu=mu, comm=comm,
                               tile_memory_gb=tile_gb)
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['nranks'] = comm.Get_size()

    _t = _time.time()
    W_minus_I, w_static = _dyson_owned(
        chi0, grid.nfreq - 1 if want_static else None, transform)
    del chi0
    if want_static:
        if extras is not None:
            extras['w_static'] = w_static
            extras['w_static_ntau'] = ntau
        freq_points = freq_points[:-1]
    rW, _ = self_energy_fit_ranges(eps, nocc, mu=mu)
    Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points,
                                       *rW, warn=False)
    Wt = screened_interaction_rows(W_minus_I, Ctw, D.shape[1], comm)
    if timings is not None:
        timings['dyson_frequencies'] = len(W_minus_I)
    del W_minus_I
    reaction_field = None
    if transform is not None:
        reaction_field = separable_quasiparticle_shift(X_mo, D, w_static,
                                                       transform, nocc)
        D = D @ transform
    if timings is not None:
        timings['t_dyson'] = _time.time() - _t

    _t = _time.time()
    if isinstance(X_ao, SlicedFactors):
        X_ao = X_ao.gather('X_ao')
    elif X_ao is None:
        X_ao = _ao_collocation(X_mo, mf)
    pairs = SigmaPairs(X_ao, D, mf.mo_coeff, eps, nocc,
                       np.atleast_1d(p_state), mu, block_memory_gb=tile_gb,
                       coords=coords, screen_r_cut=screen_r_cut)
    del X_ao
    sigma = self_energy_diagonal_rows(pairs, Wt, eps, nocc, tau_points,
                                      pade_freq, mu=mu)
    del pairs, Wt
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t

    return _finish_qp(sigma[0] if np.ndim(p_state) == 0 else sigma,
                      eps if eps_anchor is None else eps_anchor,
                      nocc, p_state, mu, pade_freq, mf, mol,
                      solver_mode, dm_correction, greedy, timings, sigma_x,
                      reaction_field=reaction_field, comm=comm,
                      sigma_x_matrix=sigma_x_matrix)


def _sigma_mo_diagonal(X_mo, D, W_omega, mf, eps, nocc, tau_points, freq_points,
                       pade_freq, mu, p_state, Wt_tau=None, tau_indices=None,
                       reduce_over=None, X_ao=None, coords=None,
                       screen_r_cut=None, block_memory_gb=ISDF_TILE_GB):
    """Sigma_c(i.omega) built in the AO basis, projected onto the p_state diagonal.

    X_ao may be `SlicedFactors`: the sweep reads it whole, so it is gathered
    here, once, and dropped with the sweep.

    Returns (nstates, nfreq), or a bare (nfreq,) for a scalar state.
    """
    if isinstance(X_ao, SlicedFactors):
        X_ao = X_ao.gather('X_ao')
    elif X_ao is None:
        X_ao = _ao_collocation(X_mo, mf)
    sigma_ao = self_energy_matrix_imaginary_time(
        X_ao, D, W_omega, mf.mo_coeff, eps, nocc,
        tau_points, freq_points, pade_freq, mu=mu, Wt_tau=Wt_tau,
        tau_indices=tau_indices, coords=coords, screen_r_cut=screen_r_cut,
        block_memory_gb=block_memory_gb)
    if reduce_over is not None:
        reduce_sum(sigma_ao, reduce_over)
    sigma = sigma_ao_to_mo_diagonal(sigma_ao, mf.mo_coeff,
                                    states=np.atleast_1d(p_state)).T
    del sigma_ao
    return sigma[0] if np.ndim(p_state) == 0 else sigma


def _root_quasiparticles(out, comm, extras=None):
    """Rank 0's quasiparticle energies on every rank (`lockstep`).

    `_finish_qp` already all-gathers the window, so every rank holds the same
    array before this; the lockstep makes that a guarantee of the kernel
    rather than a property of the ranks' arithmetic happening to agree, and an
    audited run counts whether it had anything to repair. These energies go on
    the BSE diagonal and decide which root the window follows, so the ranks
    take one answer rather than each their own.

    extras: the caller's dict; an audited run compares the digests of the
    W(omega=0) it carries, identical by construction -- the Dyson rows travel
    verbatim, or come out of one all-reduced block.
    """
    if comm is None or comm.Get_size() == 1:
        return out
    if extras is not None:
        agreement(extras.get('w_static'), comm, audit_only=True,
                  label='solve_qp_energy_space_time W(0)')
    return lockstep(out, comm)


def _finish_qp(sigma, eps, nocc, p_state, mu, pade_freq, mf, mol,
               solver_mode, dm_correction, greedy, timings, sigma_x='mf',
               reaction_field=None, comm=None, sigma_x_matrix=None):
    """Sigma_c on the imaginary axis -> quasiparticle energy, one per state.

    reaction_field is the continuum's Eq. (18) shift when the route built W,
    and REPLACES the COHSEX fallback inside `static_exchange_diagonal`.

    Two costs, and only one of them carries a state index. <Sigma_x - v_xc> is
    ONE exchange build for the whole window, so it precedes the loop and stays
    shared; the Pade fit of Sigma_pp(i.omega) and the root search on
    w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w) are per state and share
    nothing, since each state has its own sample line (occupied states sit on
    the other branch) and its own scalar equation. On a quasiparticle SET --
    the whole BSE diagonal -- that loop is the larger of the two and it is
    what `comm` splits.

    comm: rank r takes states r, r + nranks, ... and the roots are all-gathered
    back into state order, so every rank returns the whole window. The static
    term is replicated from rank 0 first: the loop must solve one calculation's
    equation wherever a state lands, and a K built independently per node
    agrees only to its last bits. Nothing else changes -- Sigma arrives
    all-reduced and identical, and each root is the same scalar iteration on
    the same numbers -- so a partitioned window is BITWISE the serial one,
    which `tests/test_qp_states_over_ranks.py` gates per state. Ranks with more
    ranks than states own an empty block and only serve the gather.

    NO WINDOW SIZE DIVIDES THE STATIC TERM: K and v_xc are built whole and
    then indexed, so a two-state window pays what the whole diagonal pays.
    `sigma_x_matrix` hands that build in from the caller, which removes it
    outright, and the diagonal it yields is the built one bit for bit; failing
    that the RANKS divide it, where the mean field carries the slices a
    distributed SCF left on it -- one K and one quadrature over the ranks
    instead of on each of them. A mean field converged one rank at a time has
    no such slices and the build stays replicated, which is what it was.
    """
    states = np.atleast_1d(p_state)
    scalar = np.ndim(p_state) == 0
    sig = np.atleast_2d(sigma)
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))

    _t = _time.time()
    # One exchange build for the whole window: <Sigma_x - v_xc> carries no state
    # index until it is indexed.
    xc_diag = static_exchange_diagonal(mf, mol, states,
                                       dm_correction=dm_correction,
                                       exchange=sigma_x,
                                       reaction_field=reaction_field,
                                       sigma_x_matrix=sigma_x_matrix,
                                       comm=comm)
    if nranks > 1:
        xc_diag = lockstep(np.ascontiguousarray(xc_diag, dtype=float), comm)
    if timings is not None:
        timings['t_qp_static'] = _time.time() - _t

    _t_states = _time.time()
    mine = (partition(len(states), rank, nranks) if nranks > 1
            else np.arange(len(states)))
    out = [None] * len(states)
    for i in mine:
        i = int(i)
        p = states[i]
        z_fit, _ = imaginary_axis_sample_points(pade_freq, nocc, p, mu)
        # Occupied states sit on the negative branch, where Sigma(-i w) = conj.
        s_p = np.conj(sig[i]) if p < nocc else sig[i]
        out[i] = solve_qp_from_imaginary_axis(eps, int(p), xc_diag[i],
                                              z_fit, s_p, greedy=greedy,
                                              solver_mode=solver_mode)
    if nranks > 1:
        for chunk in comm.allgather([(i, out[i]) for i in mine]):
            for i, root in chunk:
                out[int(i)] = root
    if timings is not None:
        timings['t_qp_states'] = _time.time() - _t_states
        timings['t_qp'] = _time.time() - _t
    return out[0] if scalar else np.asarray(out)


def solve_qp_energy_space_time(mf, mol, nocc, p_state,
                               ntau=DEFAULT_NTAU, nfreq=DEFAULT_NFREQ,
                               npade=DEFAULT_NPADE, w0=1.0, auxbasis=None,
                               radii=None, factors=None, greedy=True,
                               solver_mode='pole_strength', dm_correction=None,
                               timings=None, distribute=None, comm=None,
                               freq_block=None, scratch_dir=None,
                               tau_target=DEFAULT_TAU_TARGET, extras=None,
                               screen_r_cut=None, sigma_x='mf',
                               eps_anchor=None, sigma_x_matrix=None,
                               tile_gb=ISDF_TILE_GB):
    """GW@RPA quasiparticle energy by the space-time route; restricted, DF only.

    Same quantity as `calc_qp_energy(selfenergy='GW', polarizability='RPA')`.
    `p_state` is one orbital or a window; a window shares a single Sigma.

    Sigma is built in the AO basis and projected onto the requested MO diagonal.
    That contraction is flat in the number of states and leaves the AO matrix
    behind for evGW/qsGW, at (npade, nao, nao) complex of memory.

    ntau:        imaginary-time points, or 'auto' to size from the
                 Kaltak-Klimes-Kresse test integral to residual `tau_target`.
    factors:     pre-built (X_mo, D), to reuse the factorization across calls,
                 or `SlicedFactors` from `separable_factors(sliced=True)` over
                 this call's ranks: D is gathered whole once for the solve,
                 X_o/X_v once for the chi0 sweep and X_ao once for the
                 self-energy sweep, each dropped when its sweep returns, so
                 between the sweeps a rank holds its grid rows alone. The
                 gathered arrays are the replicated ones, so the energies are
                 bitwise those of the replicated factors.
    freq_block:  build W blockwise in frequency, so chi0 is never formed.
    scratch_dir: additionally cache the tau projection on disk.
    tile_gb:     the working-set budget of the chi0 sweep's grid-row tiles
                 and of the self-energy's block pairs, GB; memory and the
                 summation order only.
    distribute:  over more than one rank the in-core route is split over
                 grid rows as well as tau points (`_qp_grid_rows`): the
                 (tau, tile) and (tau, block pair) items of both M^2 sweeps
                 go over the ranks, so the stage divides past ntau ranks,
                 and chi0 and Wt(tau) are held by auxiliary rows, each
                 frequency's W whole on its owner alone, Sigma as the
                 states' diagonal -- no whole (M, M) block, stack of
                 (naux, naux) slices or AO self-energy on any rank. proj(tau)
                 and the self-energy's branch sums are reduced and
                 re-associate; every other step is an output partition. The
                 quasiparticle window's per-state Pade and Newton split over
                 the STATES (`_finish_qp`), which needs no reduction at all
                 and is bitwise. With freq_block or scratch_dir the tau
                 partition moves inside each frequency block and Wt is held
                 rows-per-rank (`_qp_blocked`); that path still splits tau
                 points alone and every rank there inverts every frequency.
                 None (the default) distributes over `comm`, or over
                 `current_comm()` when no comm is given, and runs serially
                 without either; True with neither probes MPI's world; False
                 keeps this call's own splits serial, while the kernels it
                 calls still read the context -- a serial reference inside a
                 distributed region belongs under `distributed(None)`.
                 THE ANSWER IS RANK 0's: the mean-field arrays and eps_anchor
                 are locksteps of rank 0's before anything is sized from them,
                 and the quasiparticle energies are locksteps of rank 0's on
                 the way out. The factors and sigma_x_matrix are identical by
                 construction -- `separable_factors` and the static-exchange
                 build leave their outputs identical on every rank -- so they
                 are not broadcast again; an audited run
                 (`distributed(comm, audit=True)`) compares their digests on
                 entry and W(omega=0)'s on exit (`mpi_grid.agreement`).
    comm:        the communicator `distribute` splits over.
    extras:      dict; receives W(omega=0) for a BSE on the same factors.
    eps_anchor:  the eps_p anchoring w = eps_p + <Sigma_x - v_xc> +
                 Re Sigma_c(w), when it differs from the spectrum that builds
                 G, P0 and W. That is the evGW case and the only one: the
                 screening follows the corrected eigenvalues while the equation
                 stays anchored on the mean field. Anchoring it on the ITERATE
                 instead adds each cycle's correction a second time, which
                 shows as a gap opening by the same amount every cycle and
                 never converging. None means the two coincide, which is G0W0
                 and leaves this route bitwise unchanged.
    sigma_x:     which K builds the static exchange, see
                 `static_exchange_diagonal`. 'mf' on an ISDF mean field puts
                 the grid's K error into every QP energy at first order;
                 'df-direct' removes it with one streamed three-index pass and
                 no stored tensor, 'exact' with a full direct K.
    sigma_x_matrix: <p|Sigma_x - v_xc|q> already built for this mean field
                 (`qp_solve.static_exchange_mean_field_matrix`), handed in the
                 way `factors` is. It is a functional of the density and the
                 orbitals alone, so the same matrix serves a two-state window,
                 the whole BSE diagonal and every evGW cycle, and the diagonal
                 it yields is the built one bit for bit. Building it costs one
                 K plus the xc potential on the DFT grid, and neither the
                 window size nor the rank count divides that.
    """
    comm = current_comm() if comm is None else comm
    if distribute is None:
        distribute = comm is not None
    mpi_comm, rank, nranks = (grid_comm(comm) if distribute else (None, 0, 1))
    # BEFORE the grid is sized: `ntau` is an integer read off the spectrum, so
    # ranks whose eigenvalues differ in the last bits can pick different point
    # counts and then all-reduce chi0 buffers of different lengths.
    if nranks > 1:
        mf.mo_energy, mf.mo_coeff, mf.mo_occ, eps_anchor = lockstep(
            (mf.mo_energy, mf.mo_coeff, mf.mo_occ, eps_anchor), mpi_comm)
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    # R = e_max/e_min grows as the gap closes, so a fixed ntau is wrong at one
    # end of any size series.
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        # Sized from the ANCHOR spectrum when one is given: an evGW iterate
        # opens the gap cycle by cycle, and a grid that followed it would make
        # each cycle integrate a different functional. The mean field has the
        # smallest gap, so its count is the conservative one.
        eps_size = eps if eps_anchor is None else np.asarray(eps_anchor, float)
        ntau, tau_err = minimax_points_for_gw(
            eps_size, nocc, mu=0.5 * (eps_size[nocc - 1] + eps_size[nocc]),
            target=tau_target)
        if timings is not None:
            timings['ntau_auto'] = ntau
            timings['tau_fit_error'] = tau_err

    factors = (factors if factors is not None
               else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                      comm=mpi_comm))
    sliced = isinstance(factors, SlicedFactors)
    if nranks > 1:
        agreement((factors.coords if sliced else factors, sigma_x_matrix),
                  mpi_comm, audit_only=True,
                  label='solve_qp_energy_space_time inputs')

    # THE SELF-ENERGY SCREENS BARE AND TAKES THE CONTINUUM AS A STATIC SHIFT.
    # Duchemin et al. build Sigma from the gas-phase W and put the whole
    # reaction field in their Eq. (18), the COHSEX approximation to
    # Sigma[W_solv] - Sigma[W_gas]; screening Sigma dynamically as well counts
    # that difference twice. The BSE kernel keeps the dressed factors.
    transform = separable_gauge_transform(mol, environment_of(mf), auxbasis)
    if sliced:
        X_mo, D, X_ao, coords = _sliced_solve_factors(factors, mpi_comm,
                                                      transform)
    else:
        X_mo, D, X_ao, coords = _unpack_factors(factors)

    # The frequency axis paired with the time axis: same size, minimax. It is
    # not a quadrature here -- chi0 is transformed onto it only so the Dyson
    # inversion, which is not diagonal in time, can be done, and W comes
    # straight back. A round trip on ntau points carries no more information.
    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = minimax_frequency_grid(ntau, e_min, e_max)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    pade_freq = gauss_legendre_grid(npade, w0=w0)[0]

    # Carry omega = 0 as a zero-weight passenger when the caller wants the
    # static screening, so a BSE need not repeat the tau sweep. Stripped again
    # below: omega = 0 is not part of the Sigma quadrature.
    want_static = extras is not None or transform is not None
    if want_static:
        freq_points = np.append(np.asarray(freq_points, float), 0.0)
        freq_weights = np.append(np.asarray(freq_weights, float), 0.0)

    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)

    _, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau_points = 0.5 * minimax_time_grid(ntau, *rS)[0]

    if freq_block or scratch_dir:
        return _root_quasiparticles(
            _qp_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points,
                        freq_points, pade_freq, p_state, want_static, extras,
                        ntau, (mpi_comm, rank, nranks), freq_block,
                        scratch_dir, solver_mode, dm_correction, greedy,
                        timings, X_ao, coords, screen_r_cut, sigma_x,
                        eps_anchor, transform, sigma_x_matrix, tile_gb),
            mpi_comm if nranks > 1 else None, extras)
    if nranks > 1:
        return _root_quasiparticles(
            _qp_grid_rows(X_mo, D, X_ao, coords, mf, mol, eps, nocc, mu, grid,
                          tau_points, freq_points, pade_freq, p_state,
                          want_static, extras, ntau, mpi_comm, transform,
                          solver_mode, dm_correction, greedy, timings,
                          screen_r_cut, sigma_x, eps_anchor, sigma_x_matrix,
                          tile_gb),
            mpi_comm, extras)

    _t = _time.time()
    chi0 = chi0_imaginary_frequency(X_mo, D, eps, nocc, grid, mu=mu,
                                    tile_memory_gb=tile_gb)
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['nranks'] = nranks

    # Dyson, in place one frequency at a time: a list comprehension would hold
    # both the list and the stacked copy on top of chi0.
    _t = _time.time()
    rows = range(chi0.shape[0])
    _dyson_in_place(chi0, rows, chi0.shape[0] - 1 if want_static else None,
                    transform)
    W_omega = chi0
    w_static = None
    if want_static:
        w_static = W_omega[-1].copy()                  # dressed: the kernel's
        if extras is not None:
            extras['w_static'] = w_static
            extras['w_static_ntau'] = ntau
        W_omega = W_omega[:-1]
        freq_points = freq_points[:-1]
        freq_weights = freq_weights[:-1]

    reaction_field = None
    if transform is not None:
        reaction_field = separable_quasiparticle_shift(X_mo, D, w_static,
                                                       transform, nocc)
        D = D @ transform
    if timings is not None:
        timings['t_dyson'] = _time.time() - _t
        timings['dyson_frequencies'] = len(rows)

    _t = _time.time()
    sigma = _sigma_mo_diagonal(X_mo, D, W_omega, mf, eps, nocc, tau_points,
                               freq_points, pade_freq, mu, p_state, X_ao=X_ao,
                               coords=coords, screen_r_cut=screen_r_cut,
                               block_memory_gb=tile_gb)
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t

    return _finish_qp(sigma, eps if eps_anchor is None else eps_anchor,
                      nocc, p_state, mu, pade_freq, mf, mol,
                      solver_mode, dm_correction, greedy, timings, sigma_x,
                      reaction_field=reaction_field,
                      sigma_x_matrix=sigma_x_matrix)


def _qp_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points, freq_points,
                pade_freq, p_state, want_static, extras, ntau, mpi,
                freq_block, scratch_dir, solver_mode, dm_correction, greedy,
                timings, X_ao, coords, screen_r_cut, sigma_x='mf',
                eps_anchor=None, transform=None, sigma_x_matrix=None,
                tile_gb=ISDF_TILE_GB):
    """Low-memory branch: chi0 is never formed.

    Frequencies are built, inverted and folded into Wt(i.tau) a block at a time,
    so the peak is Wt plus one block instead of the whole frequency axis.

    mpi is (comm, rank, nranks) from `grid_comm`, (None, 0, 1) when serial.
    Under MPI the tau partition sits INSIDE the frequency blocks: each rank
    projects its own tau points into a block, the block is all-reduced, and
    each rank folds it into its own rows of Wt alone -- so Wt costs
    ntau/nranks x naux^2 per rank and the self-energy sweep runs over those
    same rows (`screened_interaction_tau_blocked`). This is exactly the path
    production memory forces, so it is the one that most needs the ranks.
    """
    mpi_comm, rank, nranks = mpi
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    tau_mine_out = (partition(len(tau_points), rank, nranks)
                    if nranks > 1 else None)
    rW, _ = self_energy_fit_ranges(eps, nocc, mu=mu)

    # Fit the omega -> tau weights on the UNEXTENDED axis, then pad with a zero
    # column: every output tau is fitted from all input frequencies, so letting
    # omega = 0 into the fit refits every other coefficient.
    # A continuum wants W(0) even when the caller asked for no extras, so the
    # passenger needs somewhere to land either way.
    static_out = extras if extras is not None else {}
    static_idx = None
    fit_freqs = freq_points[:-1] if want_static else freq_points
    Ctw, w_fit_err = minimax_transform_weights(COSINE_WT, tau_points,
                                               fit_freqs, *rW, warn=False)
    if timings is not None:
        timings['w_fit_error'] = w_fit_err
    if want_static:
        static_idx = len(freq_points) - 1
        Ctw = np.hstack([Ctw, np.zeros((Ctw.shape[0], 1))])

    wt_path = (os.path.join(scratch_dir, f'wt_tau_r{rank}.npy')
               if scratch_dir else None)
    _t = _time.time()
    Wt_tau = screened_interaction_tau_blocked(
        X_mo, D, eps, nocc, grid, Ctw, mu=mu, freq_block=freq_block,
        scratch_dir=scratch_dir, wt_scratch=wt_path,
        static_index=static_idx, static_out=static_out, transform=transform,
        tau_indices=tau_mine, tau_out_indices=tau_mine_out,
        comm=mpi_comm if nranks > 1 else None, tile_memory_gb=tile_gb)
    if want_static:
        static_out['w_static_ntau'] = ntau
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['t_dyson'] = 0.0
        timings['nranks'] = nranks

    reaction_field = None
    if transform is not None:
        reaction_field = separable_quasiparticle_shift(
            X_mo, D, static_out['w_static'], transform, nocc)
        D = D @ transform

    _t = _time.time()
    sigma = _sigma_mo_diagonal(X_mo, D, None, mf, eps, nocc, tau_points,
                               freq_points, pade_freq, mu, p_state,
                               Wt_tau=Wt_tau, tau_indices=tau_mine_out,
                               reduce_over=mpi_comm if nranks > 1 else None,
                               X_ao=X_ao, coords=coords,
                               screen_r_cut=screen_r_cut,
                               block_memory_gb=tile_gb)
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t

    out = _finish_qp(sigma, eps if eps_anchor is None else eps_anchor,
                     nocc, p_state, mu, pade_freq, mf, mol,
                     solver_mode, dm_correction, greedy, timings, sigma_x,
                     reaction_field=reaction_field,
                     comm=mpi_comm if nranks > 1 else None,
                     sigma_x_matrix=sigma_x_matrix)
    del Wt_tau, sigma
    if wt_path and os.path.exists(wt_path):
        os.remove(wt_path)
    return out


def solve_qp_diagonal_space_time(mf, mol, nocc, states=None, **kwargs):
    """The whole QP diagonal from one self-energy, as a BSE@GW needs it.

    The per-tau work Zt = D Wt D^T is shared by every state, so n states cost
    one extra (M, M) x (M, n) product.

    states: which orbitals, default all. Returns (qp_energies, states), Hartree.
    """
    eps = get_orbital_energies(mf, representation='spatial')
    if states is None:
        states = np.arange(len(eps))
    states = np.atleast_1d(states)
    return solve_qp_energy_space_time(mf, mol, nocc, states, **kwargs), states
