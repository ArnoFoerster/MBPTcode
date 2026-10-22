"""Space-time GW: the O(N^3) route to a quasiparticle energy.

The polarizability is built in imaginary time from a separable (ISDF)
factorization of the ERIs, where the occupied and virtual sums decouple, and
the self-energy Sigma = -G Wt is a pointwise product there rather than a
convolution.

Peer of the Casida route (`qp_energy.calc_qp_energy`) and the
imaginary-frequency route (`imaginary_axis.solve_qp_energy_imaginary_axis`);
all three end in `qp_solve.solve_qp_from_imaginary_axis`.

Four silent traps, all handled here:

  * W must come from the same factors as Sigma. pyscf's cderi is L^-1-whitened,
    a separable RI fits with the symmetric V^-1/2, and mixing the two gauges
    moves the QP energy by ~1.4 eV while staying self-consistent.
  * The tau grid uses the self-energy's energy range, not the polarizability's:
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

import numpy as np
from pyscf import df as pyscf_df

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.constants import (FIT_CHOLESKY_BLOCK,
                                ISDF_RADII_MATCH_TOL, ISDF_TILE_GB)
from src.Base.numerics import numerics_or_default
from src.Base.environment import environment_of
from src.Base.geometry import elements_of
from pyscf import scf as pyscf_scf

from src.Base.pyscf_interface import (get_orbital_energies,
                                      require_closed_shell_or_unrestricted,
                                      spin_index, occupied_count)
from src.Base.separable_ri import (aux_metric_sqrt,
                                   build_separable_ri, fit_M_streaming,
                                   molecular_points_covariant,
                                   refuse_unrepresented_factors,
                                   require_grid, resolve_isdf_grid,
                                   runtime_atomic_radii, shipped_radii_lookup)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.grids import (gauss_legendre_grid, minimax_time_grid,
                                  minimax_supported_sizes)
from src.Base.utils.mpi_grid import (agreement,
                                     current_comm, grid_comm, lockstep,
                                     lockstep_mean_field, partition,
                                     reduce_sum)
from src.Base.utils.time_frequency import (TimeFrequencyGrid, COSINE_WT,
                                           minimax_transform_weights,
                                           sigma_fit_ranges)
from src.SingleReference.base import get_occ_virt_indices, transition_range
from src.SingleReference.GW.imaginary_time import (
    SigmaPairs, self_energy_matrix_imaginary_time, self_energy_diagonal_rows,
    sigma_ao_to_mo_diagonal, screening_frequency_grid,
    screened_interaction_rows, screened_interaction_tau_blocked,
    minimax_points_for_gw, minimax_points_for_gw_unrestricted,
    unrestricted_fit_ranges, unrestricted_screening_frequency_grid)
from src.SingleReference.GW.qp_solve import (add_reaction_field,
                                             static_exchange_diagonal,
                                             solve_qp_from_imaginary_axis,
                                             imaginary_axis_sample_points)
from src.SingleReference.GW.reaction_field import (
    separable_gauge_transform, separable_quasiparticle_shift,
    shift_after_solve)
from src.SingleReference.LinearResponse.space_time import (
    chi0_frequency_rows, chi0_imaginary_frequency, dyson_screening,
    minimax_points_for_bse, spin_summed, static_screening_grid,
    static_screening_matrix, static_screening_matrix_unrestricted)


def separable_factors(mf, mol, auxbasis=None, radii=None, counts=None,
                      numerics=None,
                      n_start=1, grid_accuracy=None, comm=None, timings=None,
                      sliced=False, fit='replicated', fit_block=None,
                      fit_tiles=None):
    """(X_mo, D, X_ao, coords) of the Duchemin-Blase separable RI, Z = D D^T,
    refused (`UnrepresentedOrbitals`) when they miss the density of any
    orbital of `mf` -- every occupied one and every virtual, the BSE pair
    space of every route here -- by more than the bar
    ISDF_ORBITAL_RESIDUAL_TOL, relative Coulomb residual against RI-V on the
    same auxiliary basis (`refuse_unrepresented_factors`), in the gauge of D:
    a BSE on such orbitals binds spurious roots that no other check refuses.
    The bar is not a keyword; a tool that measures factors the
    bar refuses builds them with `unrefused_separable_factors`. One
    three-centre pass over every orbital, divided over `comm`.

    X_ao is the collocation the fit produces; X_mo = X_ao C is the
    form most consumers want. Both are returned because X_mo gives back only
    X_ao's projection onto the span of C (`_ao_collocation`), which is all
    of it only where the mean field kept every AO combination.

    Interpolation points come from covariant atomic frames, so the answer does
    not depend on the orientation of the molecule.

    The grid is one object, so a keyword naming it is never dropped. Three
    keywords can name it and each combination has one outcome:

      grid_accuracy alone   `resolve_isdf_grid` sets `counts` and `n_start`.
      grid_accuracy+counts  equal -> proceed; different -> ValueError. One
                            request cannot be two grids, and resolving to
                            either side hands back a factorization nobody
                            asked for.
      counts alone          tabulated -> the row; not tabulated -> refused
                            (`UntabulatedISDFGrid`) unless the numerics
                            declare a run-time re-optimization
                            (`IsdfNumerics.runtime_radii`) onto another local
                            minimum of a multi-modal surface, which is a grid
                            nothing validated.
      radii alone           the radii are the grid; no row is consulted.
      radii+counts          where a row exists for those counts the two must
                            agree to `ISDF_RADII_MATCH_TOL` or the call is
                            refused, and the row's `origin` is honoured -- it
                            places one extra point at the nucleus, so dropping
                            it builds a 306-point carbon grid where a
                            Duchemin-Blase row describes 307.
      nothing               refused (`require_grid`).

    grid_accuracy:   an accuracy level of `ISDF_GRID_ACCURACY` or four explicit
                     shell counts, resolved by `resolve_isdf_grid`, which
                     refuses anything the radii table lacks. It sets `counts`
                     and `n_start`, since a validated row is keyed on both.
    radii:           per-element shell radii; optimized per element if omitted.
    numerics:        the fit's choices and the working set of its blocked
                     loops (`fit_settings`; None for the defaults). The
                     working set (`LayoutNumerics.fit_gb`) does not change the
                     answer, but it is a speed knob as well as a memory one:
                     one `aux_e2` call per block, each rebuilding a shell-pair
                     list over nbas x auxnbas. See `build_D_F`, which carries
                     the measurement; size it from the node.
    comm:            spreads the fit's three-centre pass over ranks
                     (`build_separable_ri`); `current_comm()` when None, serial
                     without either. The result is rank 0's on every rank: at
                     entry the mean-field arrays and the placed points are
                     locksteps of rank 0's (each rank converged its own SCF,
                     and a run-time radius re-optimization or the frames'
                     `eigh` can place points differently on another rank), and
                     at exit so are the factors (`replicate_factors`), since
                     the fit's replicated Cholesky tail need not repeat
                     bitwise across ranks. Consumers downstream receive
                     identical factors and do not broadcast them again.
    timings:         dict, filled at phase boundaries on every rank.
                     `fit_points` is the interpolation-point build
                     (`molecular_points_covariant`); `fit_collocation`,
                     `fit_integrals`, `fit_integrals_reduce`, `fit_gram`,
                     `fit_cholesky`, `fit_solve`, `fit_blocks` and
                     `fit_blocks_total` come from `fit_M_streaming` through
                     `build_separable_ri`; `fit_assembly` is that function's X
                     tail plus this function's own (the dressed metric, the
                     MO/AO projections, replication), in one key; `fit_total`
                     is this call, start to return. The radius lookup has no
                     phase of its own. The clock reads touch no bit of the
                     factors.
    sliced:          over more than one rank, return a `SlicedFactors`: this
                     rank's contiguous block of the grid rows of X_mo, D and
                     X_ao, cut from the whole, lockstepped products, so a
                     consumer that gathers one gets the replicated array bit
                     for bit (`sliced_factors` carries the table of who reads
                     what). Ignored serially and on one rank.
    fit:             'replicated' (the default) or 'rows': the fit
                     distributed by grid rows (`separable_ri.fit_rows`), so
                     that no rank holds the Gram matrix, F D^T, the solve, the
                     collocation or any factor whole. It returns
                     `SlicedFactors` over more than one rank whatever `sliced`
                     says, and the tuple on one rank. Its rows are bitwise
                     identical at every rank count but are not the replicated
                     fit's bits: another realization of the same estimator,
                     within a few of the replicated fit's reassociation
                     responses. `fit_assembly` is then the metric, the
                     projections and the move of D to the contiguous rows.
    fit_block:       grid points per tile of fit='rows' (`FIT_CHOLESKY_BLOCK`
                     when None); any fixed value is bitwise across rank
                     counts, two values differ by rounding.
    fit_tiles:       fit='rows' only: a `separable_ri.FitTiles` of this fit
                     held by another stage (the distributed ISDF-K SCF's,
                     `DistributedISDFJK.fit_tiles`), read instead of fitting
                     where its points and settings are this call's, ignored
                     otherwise; `timings['fit_reused_tiles']` says it was read.
    """
    return _separable_factors(
        mf, mol, True, auxbasis=auxbasis, radii=radii, counts=counts,
        numerics=numerics, n_start=n_start,
        grid_accuracy=grid_accuracy, comm=comm, timings=timings,
        sliced=sliced, fit=fit, fit_block=fit_block, fit_tiles=fit_tiles)


def unrefused_separable_factors(mf, mol, **keywords):
    """`separable_factors`' factors without the bar's refusal, for a tool or
    a test that measures what the bar refuses (`orbital_density_residual`)
    or that gates something other than the fit; never for a calculation."""
    return _separable_factors(mf, mol, False, **keywords)


def _separable_factors(mf, mol, refuse, auxbasis=None, radii=None,
                       counts=None, numerics=None, n_start=1,
                       grid_accuracy=None, comm=None, timings=None,
                       sliced=False, fit='replicated', fit_block=None,
                       fit_tiles=None):
    """The body of `separable_factors`; `refuse` False hands the factors
    over unmeasured (`unrefused_separable_factors`)."""
    if fit not in ('replicated', 'rows'):
        raise ValueError(f"fit={fit!r}: 'replicated' or 'rows'")
    _t0 = _time.time()
    comm = current_comm() if comm is None else comm
    auxbasis = auxbasis or paired_auxbasis(mol.basis)
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    elements = elements_of(mol)
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
    require_grid(counts, radii, 'separable_factors')
    named_counts = counts is not None

    if radii is None:
        radii, origins = {}, {}
        for el in elements:
            # One table, one lookup, at the counts asked for. Rows carrying the
            # published cc-pVTZ grids live in it at their own counts, so
            # reaching them means asking for them.
            # No row: `runtime_atomic_radii` refuses it unless the numerics
            # declare a run-time grid (`IsdfNumerics.runtime_radii`).
            radii[el], _, origins[el] = runtime_atomic_radii(
                el, mol.basis, auxbasis, counts, n_start=n_start,
                runtime_radii=numerics_or_default(numerics).isdf.runtime_radii)
    else:
        # Explicit radii are the grid and are honoured. Where the caller also
        # named counts the table may describe the same grid, and then the two
        # specifications have to agree: `origin` belongs to the row, not to the
        # recipe, so a matching row brings its nuclear point with it.
        origins = {el: False for el in radii}
        against_table = sorted(set(elements) & set(radii)) if named_counts else []
        for el in against_table:
            hit = shipped_radii_lookup(el, mol.basis, auxbasis, counts)
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
        out = _row_factors(mf, mol, auxmol, coords, numerics, comm, timings,
                           fit_block, _t0, fit_tiles)
        return _represented(out, mf, mol, auxmol, comm) if refuse else out
    # Z = M^T V M is never read here: D = M^T V^1/2 below carries the metric.
    X, _, M = build_separable_ri(mol, coords, auxmol=auxmol,
                                 numerics=numerics, comm=comm, timings=timings,
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
    return _represented(out, mf, mol, auxmol, comm, V_half) if refuse else out


def _represented(factors, mf, mol, auxmol, comm, root=None):
    """`factors` once every orbital density of `mf` passes
    `refuse_unrepresented_factors` at the bar, D measured in its own gauge:
    `root`, or the mean field's environment's metric root."""
    if root is None:
        root = aux_metric_sqrt(auxmol, environment_of(mf))
    if isinstance(factors, SlicedFactors):
        x_mo, d = factors, None
    else:
        x_mo, d = factors[0], factors[1]
    mo_coeff = np.asarray(mf.mo_coeff, float)
    if mo_coeff.ndim == 3:
        # unrestricted: each spin's orbitals in their own projection X_ao C_s
        for s in range(2):
            refuse_unrepresented_factors(
                mol, auxmol, mo_coeff[s], x_mo[s], d, root=root,
                nocc=int(np.count_nonzero(mf.mo_occ[s] > 0)),
                comm=comm)
        return factors
    refuse_unrepresented_factors(mol, auxmol, mo_coeff,
                                 x_mo, d, root=root,
                                 nocc=occupied_count(mf),
                                 comm=comm)
    return factors


def _row_factors(mf, mol, auxmol, coords, numerics, comm, timings, fit_block,
                 t0, fit_tiles=None):
    """`separable_factors`' fit='rows' tail: the row-distributed fit (or the
    tiles handed in of it), then X_mo, D = M^T V^1/2 and X_ao on this rank's
    contiguous rows, each read in the fit's own tiles; `SlicedFactors` over
    more than one rank."""
    row_fit = fit_M_streaming(mol, auxmol, coords,
                              numerics=numerics, comm=comm, timings=timings,
                              fit='rows', block=fit_block, tiles=fit_tiles)
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


def _sliced_solve_factors(factors, comm):
    """(X_mo, D, X_ao, coords) as one quasiparticle solve reads `SlicedFactors`.

    D is read whole by both sweeps, chi0 and the self-energy, so it is
    gathered once here and held for the solve. The collocations stay sliced
    and are handed on as the factors themselves: the chi0 sweep gathers its
    two branches (`split_branches`), the self-energy sweep X_ao
    (`_sigma_mo_diagonal`), each once and each gone when its sweep returns.
    A continuum's Eq. (18) reads the orbital densities D^T X_mo^2 off the
    rows (`_density_factor`), gathering nothing.
    """
    factors.require(comm)
    return factors, factors.gather('D'), factors, factors.coords


def _density_factor(X_mo, D):
    """The D Eq. (18)'s orbital densities D^T X_mo^2 are read from, in X_mo's
    layout: on `SlicedFactors` their own rows (the whole D the sweeps read is
    the gather of them), else D."""
    return X_mo if isinstance(X_mo, SlicedFactors) else D


def _bare_gauge(X_mo, D, transform):
    """D T, the bare-gauge factor the self-energy screens with. On
    `SlicedFactors` D is the solve's own gather (`_sliced_solve_factors`) and
    is carried over in place, a tile of rows at a time, so the continuum holds
    no second whole (M, naux) beside the gas phase's; else a new array, the
    caller's D left as it was."""
    if not isinstance(X_mo, SlicedFactors):
        return D @ transform
    for r0 in range(0, D.shape[0], FIT_CHOLESKY_BLOCK):
        rows = slice(r0, r0 + FIT_CHOLESKY_BLOCK)
        D[rows] = D[rows] @ transform
    return D


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
    and points repeats bitwise across ranks on identical hardware, so digests
    stand in for the broadcast, and only an array some rank holds differently
    is sent.

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
    """X_ao[k, mu] with X_ao C = X_mo, from the MO collocation alone:
    X_ao = X_mo C^T S.

    MO coefficients are S-orthonormal, C^T S C = I, so this X_ao reproduces
    X_mo exactly whether or not C is square. Where the mean field dropped
    linearly dependent combinations (nmo < nao, the augmented bases run in
    the full basis) it is the fit's collocation projected onto the span of
    C, P = C C^T S, which is not the fit's X_ao. Every consumer here reads
    X_ao only through X_ao C' and C'^T Sigma_ao C' for orbitals C' of this
    mean field (`self_energy_matrix_imaginary_time`, then
    `sigma_ao_to_mo_diagonal`), and every such C' lies in that span, so
    those numbers are the fit's to rounding. An AO quantity read outside the
    span would not be: take X_ao from `separable_factors` there.
    """
    C = mf.mo_coeff
    return X_mo @ C.T @ mf.get_ovlp()


def _dyson_in_place(chi0, rows, transform=None):
    """W(i.omega_k) = [I - chi0(i.omega_k)]^-1 over chi0's rows `rows`, in place.

    Diagonal in frequency: each row is one naux^3 inversion that reads no other
    row, chi0 taken into the bare gauge first when `transform` is set.
    """
    for k in rows:
        chi0[k] = dyson_screening(chi0[k], transform)
    return chi0


def _dyson_owned(chi0, transform=None):
    """{frequency: W - I} from chi0 held by auxiliary rows (`ProjRows` over the
    frequencies), whole for this rank's round-robin frequencies alone.

    Each frequency's chi0 is gathered whole to its owner and inverted there as
    `_dyson_in_place` inverts it, in the bare gauge when `transform` is set,
    so each W is the serial inversion of its chi0, bitwise.
    """
    size, rank = chi0._size_rank()
    return {k: dyson_screening(c, transform, correlation=True)
            for k, c in chi0.gather_slices(partition(chi0.shape[0], rank,
                                                     size))}


def _static_screening(X_mo, D, eps, nocc, extras, comm=None):
    """W(omega = 0), dressed, on the route's factors: `static_screening_matrix`
    on `static_screening_grid`, the one build the BSE kernel and the chain's
    reaction field read; recorded in `extras` with its point count."""
    grid = static_screening_grid(eps, nocc)
    w_static = static_screening_matrix(X_mo, D, eps, nocc, grid, comm=comm)
    if extras is not None:
        extras['w_static'] = w_static
        extras['w_static_ntau'] = grid.ntau
    return w_static


def _qp_grid_rows(X_mo, D, X_ao, coords, mf, mol, eps, nocc, mu, grid,
                  tau_points, freq_points, pade_freq, p_state, want_static,
                  extras, ntau, comm, transform, solver_mode, dm_correction,
                  greedy, timings, screen_r_cut, sigma_x, eps_anchor,
                  sigma_x_matrix, tile_gb, xc_diagonal=None, with_z=False,
                  failures=None):
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
    W_minus_I = _dyson_owned(chi0, transform)
    del chi0
    w_static = (_static_screening(X_mo, D, eps, nocc, extras, comm)
                if want_static else None)
    rW, _ = sigma_fit_ranges(eps, nocc, mu=mu)
    Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points,
                                       *rW, warn=False)
    Wt = screened_interaction_rows(W_minus_I, Ctw, D.shape[1], comm)
    if timings is not None:
        timings['dyson_frequencies'] = len(W_minus_I)
    del W_minus_I
    reaction_field = None
    if transform is not None:
        reaction_field = separable_quasiparticle_shift(
            X_mo, _density_factor(X_mo, D), w_static, transform, nocc)
        D = _bare_gauge(X_mo, D, transform)
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
                      sigma_x_matrix=sigma_x_matrix, xc_diagonal=xc_diagonal,
                      with_z=with_z, failures=failures,
                      after_solve=shift_after_solve(eps_anchor))


def _sigma_mo_diagonal(X_mo, D, W_omega, mf, eps, nocc, tau_points, freq_points,
                       pade_freq, mu, p_state, Wt_tau=None, tau_indices=None,
                       reduce_over=None, X_ao=None, coords=None,
                       screen_r_cut=None, block_memory_gb=ISDF_TILE_GB,
                       mo_coeff=None, ranges=None):
    """Sigma_c(i.omega) built in the AO basis, projected onto the p_state diagonal.

    X_ao may be `SlicedFactors`: the sweep reads it whole, so it is gathered
    here, once, and dropped with the sweep.

    mo_coeff: the orbitals G is built in, by default the mean field's -- one
    spin's on an unrestricted reference; ranges: the transforms' fit ranges,
    by default this spectrum's own (`sigma_fit_ranges`).

    Returns (nstates, nfreq), or a bare (nfreq,) for a scalar state.
    """
    if isinstance(X_ao, SlicedFactors):
        X_ao = X_ao.gather('X_ao')
    elif X_ao is None:
        X_ao = _ao_collocation(X_mo, mf)
    mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
    sigma_ao = self_energy_matrix_imaginary_time(
        X_ao, D, W_omega, mo_coeff, eps, nocc,
        tau_points, freq_points, pade_freq, mu=mu, Wt_tau=Wt_tau,
        tau_indices=tau_indices, coords=coords, screen_r_cut=screen_r_cut,
        block_memory_gb=block_memory_gb, ranges=ranges)
    if reduce_over is not None:
        reduce_sum(sigma_ao, reduce_over)
    sigma = sigma_ao_to_mo_diagonal(sigma_ao, mo_coeff,
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
    W(omega=0) it carries, identical by construction: `static_screening_matrix`
    sums its tau addends in one rank order.
    """
    if comm is None or comm.Get_size() == 1:
        return out
    if extras is not None:
        agreement(extras.get('w_static'), comm, audit_only=True,
                  label='solve_qp_energy_space_time W(0)')
    return lockstep(out, comm)


def _finish_qp(sigma, eps, nocc, p_state, mu, pade_freq, mf, mol,
               solver_mode, dm_correction, greedy, timings, sigma_x='mf',
               reaction_field=None, comm=None, sigma_x_matrix=None, spin=None,
               xc_diagonal=None, with_z=False, failures=None,
               after_solve=True):
    """Sigma_c on the imaginary axis -> quasiparticle energy, one per state.

    reaction_field is the continuum's Eq. (18) shift when the route built W,
    and replaces the COHSEX fallback inside `static_exchange_diagonal`.
    after_solve (G0W0, `reaction_field.shift_after_solve`) adds it to every
    root whole once the equation is solved without it; False (an evGW cycle)
    puts it in the static term. Z is the equation's either way.
    spin: the channel of an unrestricted reference, whose (2, ...)
    reaction_field and sigma_x_matrix the static term indexes.

    <Sigma_x - v_xc> is one exchange build for the whole window, so it
    precedes the loop; the Pade fit of Sigma_pp(i.omega) and the root search
    on w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w) are per state (each
    state has its own sample line and its own scalar equation). On a
    quasiparticle set (the whole BSE diagonal) that loop is the larger cost,
    and it is what `comm` splits.

    comm: rank r takes states r, r + nranks, ... and the roots are
    all-gathered back into state order, so every rank returns the whole
    window. The static term is replicated from rank 0 first, since a K built
    independently per rank agrees only to its last bits; Sigma arrives
    all-reduced and identical, so a partitioned window is bitwise the serial
    one (`tests/test_qp_states_over_ranks.py` gates it per state). Ranks
    beyond the number of states own an empty block and only serve the gather.

    No window size divides the static term: K and v_xc are built whole and
    then indexed. `sigma_x_matrix` hands that build in from the caller (the
    diagonal it yields is the built one bit for bit); otherwise the ranks
    divide it where the mean field carries the slices a distributed SCF left
    on it, and a mean field converged one rank at a time builds it
    replicated.

    xc_diagonal: the static term of the equation on the states from the
    caller, in place of the build: <p|Sigma_x - v_xc|p>, plus Eq. (18) only
    where the shift belongs in the equation (after_solve False). A
    reaction_field the route built is still added after the solve.
    with_z: return (roots, Z) of the continuation.
    failures: with with_z, receives {orbital: 'ExceptionClass: message'} for
    each state whose Pade continuation or root search failed numerically
    (ArithmeticError, LinAlgError); that state's root and Z are NaN, and any
    other exception propagates.
    """
    states = np.atleast_1d(p_state)
    scalar = np.ndim(p_state) == 0
    sig = np.atleast_2d(sigma)
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))

    _t = _time.time()
    after = after_solve and reaction_field is not None
    # One exchange build for the whole window: <Sigma_x - v_xc> carries no state
    # index until it is indexed.
    if xc_diagonal is not None:
        xc_diag = np.atleast_1d(np.asarray(xc_diagonal, float))
    else:
        xc_diag = static_exchange_diagonal(
            mf, mol, states, dm_correction=dm_correction, exchange=sigma_x,
            reaction_field=None if after else reaction_field,
            sigma_x_matrix=sigma_x_matrix, comm=comm, spin=spin,
            reaction_field_after_solve=after)
    if nranks > 1:
        xc_diag = lockstep(np.ascontiguousarray(xc_diag, dtype=float), comm)
    if timings is not None:
        timings['t_qp_static'] = _time.time() - _t

    _t_states = _time.time()
    mine = (partition(len(states), rank, nranks) if nranks > 1
            else np.arange(len(states)))
    out = [None] * len(states)
    failed = {}
    for i in mine:
        i = int(i)
        p = states[i]
        z_fit, _ = imaginary_axis_sample_points(pade_freq, nocc, p, mu)
        # Occupied states sit on the negative branch, where Sigma(-i w) = conj.
        s_p = np.conj(sig[i]) if p < nocc else sig[i]
        if not with_z:
            out[i] = solve_qp_from_imaginary_axis(eps, int(p), xc_diag[i],
                                                  z_fit, s_p, greedy=greedy,
                                                  solver_mode=solver_mode)
            continue
        # A caller that asks for Z judges each root itself, so a state whose
        # continuation fails numerically is reported, NaN with its exception
        # in `failures`; any other exception is a defect and raises.
        try:
            out[i] = solve_qp_from_imaginary_axis(eps, int(p), xc_diag[i],
                                                  z_fit, s_p, greedy=greedy,
                                                  solver_mode=solver_mode,
                                                  with_z=True)
        except (ArithmeticError, np.linalg.LinAlgError) as exc:
            out[i] = (np.nan, np.nan)
            failed[i] = f'{type(exc).__name__}: {exc}'
    if nranks > 1:
        for chunk in comm.allgather([(i, out[i], failed.get(i))
                                     for i in mine]):
            for i, root, why in chunk:
                out[int(i)] = root
                if why is not None:
                    failed[int(i)] = why
    if failures is not None:
        failures.update({int(states[i]): why for i, why in failed.items()})
    if timings is not None:
        timings['t_qp_states'] = _time.time() - _t_states
        timings['t_qp'] = _time.time() - _t
    shift = reaction_field if after else None
    if with_z:
        roots = add_reaction_field(np.array([r for r, _ in out], float),
                                   shift, states, spin)
        z = np.array([zz for _, zz in out], float)
        return (roots[0], z[0]) if scalar else (roots, z)
    roots = add_reaction_field(np.asarray(out, float), shift, states, spin)
    return roots[0] if scalar else roots


def solve_qp_energy_space_time(mf, mol, nocc, p_state, numerics=None,
                               auxbasis=None, radii=None, factors=None,
                               greedy=True, solver_mode='pole_strength',
                               dm_correction=None, timings=None,
                               distribute=None, comm=None, freq_block=None,
                               scratch_dir=None, extras=None,
                               screen_r_cut=None, sigma_x='mf',
                               eps_anchor=None, sigma_x_matrix=None,
                               spin_channel='alpha',
                               xc_diagonal=None, with_z=False, failures=None,
                               counts=None, grid_accuracy=None):
    """GW@RPA quasiparticle energy by the space-time route; DF only.

    Same quantity as `calc_qp_energy(selfenergy='GW', polarizability='RPA')`.
    `p_state` is one orbital or a window; a window shares a single Sigma.

    Sigma is built in the AO basis and projected onto the requested MO diagonal.
    That contraction is flat in the number of states and leaves the AO matrix
    behind for evGW/qsGW, at (npade, nao, nao) complex of memory.

    numerics:    the choices (`Numerics`; None for the defaults): the
                 quadrature's `ntau_st` imaginary-time points, or 'auto' to
                 size from the Kaltak-Klimes-Kresse test integral to residual
                 `tau_target`; `nfreq_st` chi0 frequencies ('auto': the
                 minimax grid of that size), `npade` Pade points on the
                 Gauss-Legendre grid of scale `w0_gl`; `layout.tile_gb`, the
                 working-set budget of the chi0 sweep's grid-row tiles and of
                 the self-energy's block pairs, GB (ISDF_TILE_GB where None;
                 memory and the summation order only); the fit's choices of
                 the factors built here.
    counts, grid_accuracy, radii: the interpolation grid of the factors
                 built here when `factors` is not given; one is required.
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
    distribute:  over more than one rank the in-core route is split over
                 grid rows as well as tau points (`_qp_grid_rows`): the
                 (tau, tile) and (tau, block pair) items of both M^2 sweeps
                 go over the ranks, so the stage divides past ntau ranks;
                 chi0 and Wt(tau) are held by auxiliary rows, each
                 frequency's W whole on its owner alone, Sigma as the
                 states' diagonal, with no whole (M, M) block, stack of
                 (naux, naux) slices or AO self-energy on any rank. proj(tau)
                 and the self-energy's branch sums are reduced and
                 re-associate; every other step is an output partition. The
                 window's per-state Pade and Newton split over the states
                 (`_finish_qp`), with no reduction and bitwise. With
                 freq_block or scratch_dir the tau partition moves inside
                 each frequency block and Wt is held rows-per-rank
                 (`_qp_blocked`); that path splits tau points alone and every
                 rank inverts every frequency. None (the default) distributes
                 over `comm`, or over `current_comm()` when no comm is given,
                 and runs serially without either; True with neither probes
                 MPI's world; False keeps this call's own splits serial,
                 while the kernels it calls still read the context (a serial
                 reference inside a distributed region belongs under
                 `distributed(None)`). The answer is rank 0's: the mean-field
                 arrays and eps_anchor are locksteps of rank 0's before
                 anything is sized from them, and so are the quasiparticle
                 energies on the way out. The factors and sigma_x_matrix are
                 identical by construction and not broadcast again; an
                 audited run (`distributed(comm, audit=True)`) compares their
                 digests on entry and W(omega=0)'s on exit
                 (`mpi_grid.agreement`).
    comm:        the communicator `distribute` splits over.
    extras:      dict; receives W(omega=0) for a BSE on the same factors, the
                 one static build (`static_screening_matrix` on
                 `static_screening_grid`) a BSE would otherwise rebuild.
    eps_anchor:  the eps_p anchoring w = eps_p + <Sigma_x - v_xc> +
                 Re Sigma_c(w), when it differs from the spectrum that builds
                 G, P0 and W: the evGW case, where the screening follows the
                 corrected eigenvalues while the equation stays anchored on
                 the mean field. Anchoring it on the iterate instead adds each
                 cycle's correction a second time (a gap opening by the same
                 amount every cycle). None means the two coincide (G0W0).
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
    spin_channel: 'alpha' or 'beta' on an unrestricted reference, nocc then
                 (nalpha, nbeta); see `_space_time_unrestricted`. Ignored on a
                 restricted one.
    xc_diagonal: <p|Sigma_x - v_xc (+ Sigma^env)|p> on `p_state` from the
                 caller; it replaces the static build and the Eq. (18) term,
                 so the roots differ from the caller's by the correlation part
                 alone. Restricted reference only.
    with_z:      return (energies, Z), Z = 1/(1 - dRe Sigma_c/dw) of the
                 continued fraction at each root. Restricted reference only.
    failures:    with with_z, a dict that receives {orbital: 'ExceptionClass:
                 message'} for each state whose continuation failed
                 numerically (ArithmeticError, LinAlgError); that state's
                 energy and Z are NaN.
    """
    numerics = numerics_or_default(numerics)
    q = numerics.quadrature
    ntau, nfreq, npade = q.ntau_st, q.nfreq_st, q.npade
    w0, tau_target = q.w0_gl, q.tau_target
    tile_gb = (ISDF_TILE_GB if numerics.layout.tile_gb is None
               else numerics.layout.tile_gb)
    comm = current_comm() if comm is None else comm
    if distribute is None:
        distribute = comm is not None
    mpi_comm, rank, nranks = (grid_comm(comm) if distribute else (None, 0, 1))
    # Before the grid is sized: `ntau` is an integer read off the spectrum, so
    # ranks whose eigenvalues differ in the last bits can pick different point
    # counts and then all-reduce chi0 buffers of different lengths.
    if nranks > 1:
        mf.mo_energy, mf.mo_coeff, mf.mo_occ, eps_anchor = lockstep(
            (mf.mo_energy, mf.mo_coeff, mf.mo_occ, eps_anchor), mpi_comm)
    require_closed_shell_or_unrestricted(mf, 'solve_qp_energy_space_time', mol=mol)
    if isinstance(mf, pyscf_scf.uhf.UHF):
        if xc_diagonal is not None or with_z:
            raise ValueError("xc_diagonal and with_z take a restricted "
                             "reference")
        return _space_time_unrestricted(
            mf, mol, nocc, p_state, spin_index(spin_channel), ntau, nfreq,
            npade, w0, auxbasis, radii, factors, greedy, solver_mode,
            dm_correction, timings, (mpi_comm, rank, nranks), freq_block,
            scratch_dir, tau_target, extras, screen_r_cut, sigma_x,
            eps_anchor, sigma_x_matrix, tile_gb, counts, grid_accuracy)
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    # R = e_max/e_min grows as the gap closes, so a fixed ntau is wrong at one
    # end of any size series.
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        # Sized from the anchor spectrum when one is given: an evGW iterate
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
                                      counts=counts,
                                      grid_accuracy=grid_accuracy,
                                      comm=mpi_comm, numerics=numerics))
    sliced = isinstance(factors, SlicedFactors)
    if nranks > 1:
        agreement((factors.coords if sliced else factors, sigma_x_matrix),
                  mpi_comm, audit_only=True,
                  label='solve_qp_energy_space_time inputs')

    # The self-energy screens bare and takes the continuum as a static shift:
    # Duchemin et al. build Sigma from the gas-phase W and put the whole
    # reaction field in their Eq. (18), the COHSEX approximation to
    # Sigma[W_solv] - Sigma[W_gas]; screening Sigma dynamically as well counts
    # that difference twice. The BSE kernel keeps the dressed factors.
    transform = separable_gauge_transform(mol, environment_of(mf), auxbasis)
    if sliced:
        X_mo, D, X_ao, coords = _sliced_solve_factors(factors, mpi_comm)
    else:
        X_mo, D, X_ao, coords = _unpack_factors(factors)

    # The frequency axis paired with the time axis: same size, minimax. It is
    # not a quadrature here -- chi0 is transformed onto it only so the Dyson
    # inversion, which is not diagonal in time, can be done, and W comes
    # straight back. A round trip on ntau points carries no more information.
    # It is built over the range W's way back to tau is fitted over, rW, not
    # over the bare window (`screening_frequency_grid`).
    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = screening_frequency_grid(ntau, eps, nocc,
                                                             mu=mu)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    pade_freq = gauss_legendre_grid(npade, w0=w0)[0]

    # W(omega = 0) for a BSE kernel or the Eq. (18) shift, from the one static
    # build: not on this axis, whose omega = 0 would be a grid of its own.
    want_static = extras is not None or transform is not None

    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)

    _, rS = sigma_fit_ranges(eps, nocc, mu=mu)
    tau_points = 0.5 * minimax_time_grid(ntau, *rS)[0]

    if freq_block or scratch_dir:
        return _root_quasiparticles(
            _qp_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points,
                        freq_points, pade_freq, p_state, want_static, extras,
                        ntau, (mpi_comm, rank, nranks), freq_block,
                        scratch_dir, solver_mode,
                        dm_correction, greedy, timings, X_ao, coords,
                        screen_r_cut, sigma_x, eps_anchor, auxbasis,
                        transform, sigma_x_matrix, tile_gb, xc_diagonal,
                        with_z, failures=failures),
            mpi_comm if nranks > 1 else None, extras)
    if nranks > 1:
        return _root_quasiparticles(
            _qp_grid_rows(X_mo, D, X_ao, coords, mf, mol, eps, nocc, mu, grid,
                          tau_points, freq_points, pade_freq, p_state,
                          want_static, extras, ntau, mpi_comm, transform,
                          solver_mode, dm_correction, greedy, timings,
                          screen_r_cut, sigma_x, eps_anchor, sigma_x_matrix,
                          tile_gb, xc_diagonal, with_z,
                          failures=failures),
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
    _dyson_in_place(chi0, rows, transform)
    W_omega = chi0
    w_static = (_static_screening(X_mo, D, eps, nocc, extras)
                if want_static else None)

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
                      sigma_x_matrix=sigma_x_matrix, xc_diagonal=xc_diagonal,
                      with_z=with_z, failures=failures,
                      after_solve=shift_after_solve(eps_anchor))


def _space_time_unrestricted(mf, mol, nocc, p_state, spin, ntau, nfreq, npade,
                             w0, auxbasis, radii, factors, greedy, solver_mode,
                             dm_correction, timings, mpi, freq_block,
                             scratch_dir, tau_target, extras, screen_r_cut,
                             sigma_x, eps_anchor, sigma_x_matrix, tile_gb,
                             counts=None, grid_accuracy=None):
    """`solve_qp_energy_space_time` on an unrestricted reference, channel `spin`.

    One W from chi0_alpha + chi0_beta (`spin_summed`), each spin in its own
    collocation X_mo,s = X_ao C_s off the one factorization, and then the
    restricted self-energy sweep in the channel's orbitals: Sigma_s = -G_s W
    carries no spin factor, so the restricted kernel on (X_mo,s, eps_s,
    nocc_s) is it. Both spins build W, so every grid spans both: chi0's time
    grid covers both spins' transitions, W's frequency grid covers the range
    its transform back to tau is fitted over
    (`unrestricted_screening_frequency_grid`), and the self-energy's fit
    ranges cover both channels (`unrestricted_fit_ranges`). Each channel
    samples Sigma on the line through its own mid-gap and carries its own
    static exchange and its own Eq. (18) shift, contracted against the one W.
    A closed shell run unrestricted therefore reads every grid the
    restricted route reads.

    eps_anchor, when given, is the (2, nmo) mean-field spectrum of the evGW
    loop; the channel's row anchors the equation.

    Over more than one rank it takes the restricted route's grid rows
    (`_qp_grid_rows`): chi0 summed over the spins in this rank's auxiliary
    rows, each W whole on its owner alone, Wt by rows and the channel's
    self-energy by block pairs.

    The blocked (`freq_block`, `scratch_dir`) and sliced paths are the
    restricted route's memory tiers and are not built for two spins: they are
    refused by name rather than run as something else.
    """
    mpi_comm, rank, nranks = mpi
    if freq_block or scratch_dir:
        raise NotImplementedError(
            'freq_block / scratch_dir on an unrestricted reference: the '
            'blocked space-time path is built for one spin; run it unblocked')
    if isinstance(factors, SlicedFactors):
        raise NotImplementedError(
            'sliced factors on an unrestricted reference: each spin projects '
            'the AO collocation on its own orbitals, which the sliced solve '
            'does not; build the factors unsliced')
    noccs = tuple(int(n) for n in nocc)
    spectra = tuple(np.asarray(e, float) for e in mf.mo_energy)
    eps, n_ch = spectra[spin], noccs[spin]
    mu = 0.5 * (eps[n_ch - 1] + eps[n_ch])
    if eps_anchor is None:
        anchors = spectra
    else:
        anchors = np.asarray(eps_anchor, float)
        if anchors.shape != (2, eps.size):
            raise ValueError(
                f'an unrestricted eps_anchor is one spectrum per spin, shape '
                f'(2, {eps.size}); got {anchors.shape}')
    e_min, e_max = transition_range(spectra, noccs)
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, tau_err = minimax_points_for_gw_unrestricted(
            anchors, noccs, target=tau_target)
        if timings is not None:
            timings['ntau_auto'] = ntau
            timings['tau_fit_error'] = tau_err
    factors = (factors if factors is not None
               else separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                      counts=counts,
                                      grid_accuracy=grid_accuracy,
                                      comm=mpi_comm))
    _, D, X_ao, coords = _unpack_factors(factors)
    if X_ao is None:
        raise ValueError(
            'an unrestricted reference needs the AO collocation of the '
            'factors, (X_mo, D, X_ao[, coords]): each spin projects it on its '
            'own orbitals')
    X_mos = [X_ao @ c for c in mf.mo_coeff]
    if nranks > 1:
        agreement((X_ao, D, sigma_x_matrix), mpi_comm, audit_only=True,
                  label='solve_qp_energy_space_time inputs')
    transform = separable_gauge_transform(mol, environment_of(mf), auxbasis)
    if nfreq is None or (isinstance(nfreq, str) and nfreq.lower() == 'auto'):
        if ntau not in minimax_supported_sizes():
            raise ValueError(
                f"nfreq='auto' needs a tabulated minimax frequency grid at "
                f'ntau = {ntau}; GreenX has {minimax_supported_sizes()}. Pass an '
                'explicit nfreq.')
        freq_points, freq_weights = unrestricted_screening_frequency_grid(
            ntau, spectra, noccs)
    else:
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    pade_freq = gauss_legendre_grid(npade, w0=w0)[0]
    # W(omega = 0) for a BSE kernel or the Eq. (18) shift, from the one static
    # build: not on this axis, whose omega = 0 would be a grid of its own.
    want_static = extras is not None or transform is not None
    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max,
                                           freq_points, freq_weights)
    ranges = unrestricted_fit_ranges(spectra, noccs)
    tau_points = 0.5 * minimax_time_grid(ntau, *ranges[1])[0]

    _t = _time.time()
    if nranks > 1:
        chi0 = spin_summed(chi0_frequency_rows, X_mos, D, spectra, noccs,
                           grid, comm=mpi_comm, tile_memory_gb=tile_gb)
    else:
        chi0 = spin_summed(chi0_imaginary_frequency, X_mos, D, spectra,
                           noccs, grid, tile_memory_gb=tile_gb)
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['nranks'] = nranks
    _t = _time.time()
    if nranks > 1:
        W_minus_I = _dyson_owned(chi0, transform)
        del chi0
        Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, freq_points,
                                           *ranges[0], warn=False)
        W_omega = screened_interaction_rows(W_minus_I, Ctw, D.shape[1],
                                            mpi_comm)
        n_dyson = len(W_minus_I)
        del W_minus_I
    else:
        _dyson_in_place(chi0, range(chi0.shape[0]), transform)
        W_omega, n_dyson = chi0, chi0.shape[0]
    w_static = None
    if want_static:
        w_static = static_screening_matrix_unrestricted(X_mos, D, spectra,
                                                        noccs)
        if extras is not None:
            extras['w_static'] = w_static
            extras['w_static_ntau'] = max(
                minimax_points_for_bse(e, n)[0] for e, n in zip(spectra, noccs))
    reaction_field = None
    if transform is not None:
        reaction_field = np.array([
            separable_quasiparticle_shift(X, D, w_static, transform, n)
            for X, n in zip(X_mos, noccs)])
        D = D @ transform
    if timings is not None:
        timings['t_dyson'] = _time.time() - _t
        timings['dyson_frequencies'] = n_dyson
    _t = _time.time()
    if nranks > 1:
        pairs = SigmaPairs(X_ao, D, mf.mo_coeff[spin], eps, n_ch,
                           np.atleast_1d(p_state), mu, block_memory_gb=tile_gb,
                           coords=coords, screen_r_cut=screen_r_cut)
        sigma = self_energy_diagonal_rows(pairs, W_omega, eps, n_ch,
                                          tau_points, pade_freq, mu=mu,
                                          ranges=ranges)
        del pairs
        if np.ndim(p_state) == 0:
            sigma = sigma[0]
    else:
        sigma = _sigma_mo_diagonal(X_mos[spin], D, W_omega, mf, eps, n_ch,
                                   tau_points, freq_points, pade_freq, mu,
                                   p_state, X_ao=X_ao, coords=coords,
                                   screen_r_cut=screen_r_cut,
                                   block_memory_gb=tile_gb,
                                   mo_coeff=mf.mo_coeff[spin], ranges=ranges)
    del W_omega
    if timings is not None:
        timings['t_sigma'] = _time.time() - _t
    return _root_quasiparticles(
        _finish_qp(sigma, anchors[spin], n_ch, p_state, mu, pade_freq, mf, mol,
                   solver_mode, dm_correction, greedy, timings, sigma_x,
                   reaction_field=reaction_field,
                   comm=mpi_comm if nranks > 1 else None,
                   sigma_x_matrix=sigma_x_matrix, spin=spin,
                   after_solve=shift_after_solve(eps_anchor)),
        mpi_comm if nranks > 1 else None, extras)


def _qp_blocked(X_mo, D, mf, mol, eps, nocc, mu, grid, tau_points, freq_points,
                pade_freq, p_state, want_static, extras, ntau, mpi,
                freq_block, scratch_dir, solver_mode, dm_correction, greedy,
                timings, X_ao, coords, screen_r_cut, sigma_x='mf',
                eps_anchor=None, auxbasis=None, transform=None,
                sigma_x_matrix=None, tile_gb=ISDF_TILE_GB, xc_diagonal=None,
                with_z=False, failures=None):
    """Low-memory branch: chi0 is never formed.

    Frequencies are built, inverted and folded into Wt(i.tau) a block at a time,
    so the peak is Wt plus one block instead of the whole frequency axis.

    mpi is (comm, rank, nranks) from `grid_comm`, (None, 0, 1) when serial.
    Under MPI the tau partition sits inside the frequency blocks: each rank
    projects its own tau points into a block, the block is all-reduced, and
    each rank folds it into its own rows of Wt alone, so Wt costs
    ntau/nranks x naux^2 per rank and the self-energy sweep runs over those
    same rows (`screened_interaction_tau_blocked`).
    """
    mpi_comm, rank, nranks = mpi
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    tau_mine_out = (partition(len(tau_points), rank, nranks)
                    if nranks > 1 else None)
    rW, _ = sigma_fit_ranges(eps, nocc, mu=mu)

    Ctw, w_fit_err = minimax_transform_weights(COSINE_WT, tau_points,
                                               freq_points, *rW, warn=False)
    if timings is not None:
        timings['w_fit_error'] = w_fit_err

    wt_path = (os.path.join(scratch_dir, f'wt_tau_r{rank}.npy')
               if scratch_dir else None)
    _t = _time.time()
    Wt_tau = screened_interaction_tau_blocked(
        X_mo, D, eps, nocc, grid, Ctw, mu=mu, freq_block=freq_block,
        scratch_dir=scratch_dir, wt_scratch=wt_path,
        transform=transform,
        tau_indices=tau_mine, tau_out_indices=tau_mine_out,
        comm=mpi_comm if nranks > 1 else None, tile_memory_gb=tile_gb)
    w_static = (_static_screening(X_mo, D, eps, nocc, extras,
                                  mpi_comm if nranks > 1 else None)
                if want_static else None)
    if timings is not None:
        timings['t_chi0'] = _time.time() - _t
        timings['t_dyson'] = 0.0
        timings['nranks'] = nranks

    reaction_field = None
    if transform is not None:
        reaction_field = separable_quasiparticle_shift(
            X_mo, _density_factor(X_mo, D), w_static, transform,
            nocc)
        D = _bare_gauge(X_mo, D, transform)

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
                     sigma_x_matrix=sigma_x_matrix, xc_diagonal=xc_diagonal,
                     with_z=with_z, failures=failures,
                     after_solve=shift_after_solve(eps_anchor))
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
    require_closed_shell_or_unrestricted(mf, 'solve_qp_diagonal_space_time',
                                         mol=mol)
    eps = get_orbital_energies(mf, representation='spatial')
    if states is None:
        states = np.arange(np.shape(eps)[-1])
    states = np.atleast_1d(states)
    return solve_qp_energy_space_time(mf, mol, nocc, states, **kwargs), states
