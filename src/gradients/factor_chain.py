"""The frozen ISDF factorization a cubic gradient chain differentiates, and the
three branches that carry adjoints on its factors to the nuclei.

Every target on the space-time route (a quasiparticle energy, a BSE
excitation, the dRPA correlation energy) is a function of the orbital
energies eps, the collocation X_mo = X_ao C and the auxiliary factor D, and its
reverse pass ends with adjoints (eps_bar, X_bar, D_bar). From there on nothing
depends on the target:

    eps_bar, and X_bar through C          -> the orbital-response Lagrangian
    X_bar through the interpolation points -> the collocation adjoint
    D_bar                                  -> the fit adjoint

The grid radii, pair layout and frames are decided once, at the reference
geometry: re-deciding any of them per geometry puts a step into the energy.
The surroundings enter through one object, the chain's `environment`
(src.Base.environment), which builds the mean field at each geometry, dresses
the auxiliary gauge, supplies the static one-body term and owns the adjoints
of both.

A chain carries no communicator. Inside `with distributed(comm):` every rank
runs it whole, the kernels divide their sweeps over the ranks, and the arrays
the chain decides itself (grid, placed points, pair layout, fit, assembled
gradient, mean-field force) are each one `lockstep`, so every rank holds rank
0's bits.

Sliced factors: a factorization built with `sliced=True` holds, over more than
one rank, each rank's `contiguous_block` of the grid rows of X_mo, D and X_ao
(`SlicedFactors`), cut from the whole products formed as the whole layout
forms them, and keeps no whole fit between geometries. Kernels that read a
factor whole gather it once per sweep or solve; the Davidson block action
reads the rows. The layout adds no difference of its own; pyscf's threaded K
builds and mean-field force do not repeat their bits from run to run (OpenMP
GEMM partial sums), so two evaluations agree to that reassociation and are
compared on an anchored bar. X_bar and D_bar are whole in both layouts
(all-reduced sums over the tau partition). Larger than any factor and
untouched by the layout: the fit's nk^2 Gram matrix, proj(tau) of the
quasiparticle solve (ntau naux^2) and the BSE adjoint's (naux, nocc, nvir)
blocks.

One estimator: both realizations of the fit solve
`separable_ri.fit_M_streaming`'s estimator on the frozen pair layout (Gram
matrix over every product pair, F D^T over the screened pairs), the one the
ISDF-K SCF, space-time GW and ISDF BSE run, so a force here is the derivative
of the energy those routes report. The replicated fit is `fit_M_streaming`, bit
for bit. The row fit (`fit='rows'`, with `sliced=True`) builds the rows with
`separable_ri.fit_rows` on the frozen points: the Gram matrix, F D^T, the
solve and the collocation exist only as each rank's tiles. Its peak is the
three-centre pass (one f shell's (mu nu|P) beside the metric's LU), which no
rank count lowers. The rows are bitwise the same at every rank count and lie
within a few reassociation responses (`FIT_REASSOCIATION_K`) of the
replicated fit's.

The row fit's nuclear assembly runs in the same tiles (`row_fit_branches`,
`separable_ri.fit_rows_adjoint`), the same bits at every rank count. Per rank
it peaks in its three-centre pass (the Gram tiles' kept factor, the metric's
LU, three (rows, naux) arrays and one block's broadcast coefficients) beside
X_bar, D_bar and the factor rows; rank 0 alone holds four metric-sized arrays
for the root's adjoint and reads D_bar whole. The pair layout is screened once
per reference on the whole AO collocation.

On a distributed ISDF-K SCF the mean field's handle holds this fit's M^T tiles
wherever the frozen points are its grid (the reference geometry), and the rows
are read from them (`separable_ri.FitTiles`). A force's row exchange skeletons
hand their fit-adjoint seeds to the assembly (`one_fit_adjoint`), which
contracts them with its own in one `fit_rows_adjoints` call. The chain's point
adjoint closes on the frozen frames and the skeletons' on the SCF's turning
frames, so the two share one pass but not one seed.
"""
import time
import warnings
import collections
import gc
import weakref
from contextlib import contextmanager

import numpy as np
from pyscf import df as pyscf_df

from src.Base.isdf_jk import ISDFJK, mean_field_skeleton_force
from src.Base.constants import (ENVIRONMENT_CACHE_SIZE, FIT_CHOLESKY_BLOCK,
                                FIT_REALIZATIONS, ISDF_DEFAULT_COUNTS,
                                ISDF_FIT_ERROR_FAILED, SCF_GRAD_TOL)
from src.Base.distributed_df import distributed_fock, distributed_mean_field
from src.Base.environment import dresses_interaction, resolve_environment
from src.Base.separable_ri import (atomic_frames, aux_metric_sqrt,
                                   default_auxbasis, fit_M_streaming,
                                   fit_M_whole, resolve_isdf_grid,
                                   runtime_atomic_radii, subshells,
                                   test_set_layout)
from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.mpi_grid import current_comm, lockstep, lockstep_mean_field
from src.SingleReference.LinearResponse.rpa_energy import reference_energy
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            continued_frames,
                                            dfactor_adjoint_gauges,
                                            GaugeAdjoint,
                                            eps_chain_gradient,
                                            exx_double_counting_Y,
                                            exx_double_counting_skeleton,
                                            isdf_scf_handle, one_fit_adjoint,
                                            orbital_rotation_rows,
                                            pending_fit_adjoint,
                                            point_layout, product_pairs,
                                            require_whole_fit_adjoint,
                                            row_fit_adjoint)


def converged_factory(scf_factory):
    """`scf_factory` completed by the chain when it returns a mean field that
    has been built but not run.

    The completion is `distributed_mean_field`: inside a `distributed`
    context every rank runs pyscf's SCF driver against the reduced J/K and
    quadrature, each contributing its block of the auxiliary index and of the
    xc grid, and rank 0's spectrum and orbitals end on all of them. Outside
    one it is `mf.kernel()`. Inside one the mean field must use pyscf's own DF
    (the split is over the rows of `cderi`); `distributed_mean_field` refuses
    any other.
    """
    def build(mol):
        mf = scf_factory(mol)
        if getattr(mf, 'mo_coeff', None) is None:
            distributed_mean_field(mf)
        return mf

    return build


def radii_tag(radii):
    """Hashable summary of an explicit radii set, or None when the atomic
    optimizer decides them: two factorizations differing only in their radii are
    different functionals and must not share."""
    if radii is None:
        return None
    return tuple((el, tuple(np.round(np.concatenate(
        [np.atleast_1d(radii[el][s]) for s in sorted(radii[el])]), 12)))
        for el in sorted(radii))


class FrozenFactorization:
    """The ISDF conventions fixed at one reference geometry, so that several
    chains are one functional rather than two that happen to agree.

    Radii, interpolation points, frames and pair layout are discrete choices:
    re-deciding them per geometry puts a step in the energy, and deciding them
    once per chain diverges silently when the settings differ (`counts`,
    `n_start`, or `frames='continued'`, which carries orientation history).
    Holding them in one object makes sharing structural, and a composed
    surface costs one fit per geometry.

    It owns what `separable_factors` consumes and nothing else. The
    quadratures are not here (dRPA integrates chi0 over frequency, the
    self-energy over imaginary time; each chain keeps its own), nor is the
    quasiparticle window. `refreeze` keeps a quadrature fixed between
    geometries, a different requirement from sharing between chains.

    `grid_accuracy` fixes `counts` and `n_start` together: a validated
    accuracy level or four shell counts, resolved against the shipped radii
    table by `resolve_isdf_grid`, which refuses an unoptimized grid.

    Under ranks the conventions are rank 0's: every rank decides them and one
    `lockstep` leaves rank 0's radii, fit errors, clouds, owners and frames on
    all of them; points, pair layout and fit are locksteps of their own.
    Conventions decided per rank put ranks on different machines on different
    surfaces (forces differing by 1e-3 Ha/Bohr and more). Serially every
    lockstep is a no-op.

    `sliced` is the layout of the factors, not part of `settings`: over more
    than one rank each rank keeps its grid rows of X_mo, D and X_ao
    (`FactorChain.factors_at`) and no whole fit outlives the cut.

    `fit` is the realization of the fit, also matched apart from `settings`;
    both solve `fit_M_streaming`'s estimator on the frozen `layout` (Gram
    matrix (A A^T) o (B B^T) + P P^T over every product pair, F D^T over the
    layout's pairs). 'replicated' is `_fit`, `fit_M_streaming` run whole by
    every rank. 'rows' (with `sliced=True`) is `separable_ri.fit_rows` on the
    frozen points, `fit_block` points per tile: no rank holds a whole Gram
    matrix, F D^T, solve, collocation or factor, and the rows are bitwise the
    same at every rank count and within a few reassociation responses
    (`FIT_REASSOCIATION_K`) of the replicated fit's.
    """

    def __init__(self, mol, basis=None, auxbasis=None, counts=None, n_start=1,
                 frames='frozen', radii=None, grid_accuracy=None,
                 sliced=False, fit='replicated', fit_block=None):
        # Refused before the radii are optimized or anything is placed.
        if fit not in FIT_REALIZATIONS:
            raise ValueError(f'fit={fit!r}: one of {FIT_REALIZATIONS}')
        if fit == 'rows' and not sliced:
            raise ValueError(
                "fit='rows' builds each rank's grid rows and never the whole "
                'factors; build the factorization with sliced=True (on one '
                'rank the rows are the whole grid and the flag lays nothing '
                'out)')
        if fit_block is not None and fit != 'rows':
            raise ValueError(
                f"fit_block={fit_block} is the row fit's tile edge; "
                f'fit={fit!r} has no tiles and would ignore it')
        self.basis = basis or mol.basis
        self.auxbasis = auxbasis or default_auxbasis(self.basis)
        elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
        # Resolved here and stored as plain counts, so `rebuilt_at`,
        # `settings` and `require_match` compare one kind of object and a
        # level cannot mean two grids.
        if grid_accuracy is not None:
            counts, n_start = resolve_isdf_grid(grid_accuracy, self.basis,
                                                elements, auxbasis=self.auxbasis)
        self.counts = counts or ISDF_DEFAULT_COUNTS
        self.n_start = n_start
        self.frames_mode = frames
        self.with_frames = frames == 'continued'
        # Explicit radii (e.g. `separable_ri.tailor_grid`'s) replace the
        # per-element atomic optimization and are frozen like every other
        # discrete choice.
        self.radii_tag = radii_tag(radii)
        if radii is not None and not set(elements) <= set(radii):
            raise ValueError(f'radii given for {sorted(radii)} but the molecule '
                             f'holds {elements}')
        # The fit error is kept: it is the only signal of whether this grid
        # resolves this basis. The default count is sized for double zeta and
        # fails for most elements at larger bases.
        fit_errors = {}
        # wall seconds of each convention's construction, for the record
        self.seconds = {}
        t0 = time.perf_counter()
        if radii is None:
            radii = {}
            for el in elements:
                radii[el], fit_errors[el], _ = runtime_atomic_radii(
                    el, self.basis, self.auxbasis, self.counts,
                    n_start=n_start)
        self.seconds['radii'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        # The radii come from a local descent on a multi-modal objective whose
        # minimum moves with the BLAS reduction order, the frames from an
        # `eigh`: rank 0's on every rank, before anything is placed. Clouds and
        # owners follow from the radii and travel in the same call.
        (self.radii, self.fit_errors, self.pts_local, self.owner,
         self.frames) = lockstep((radii, fit_errors,
                                  *point_layout(mol, radii),
                                  atomic_frames(mol)[0]))
        self.seconds['points'] = time.perf_counter() - t0
        if self.radii_tag is None:
            failed = {el: e for el, e in self.fit_errors.items()
                      if e >= ISDF_FIT_ERROR_FAILED}
            if failed:
                npts = sum(len(subshells()[n]) * len(r)
                           for n, r in next(iter(self.radii.values())).items())
                warnings.warn(
                    f'ISDF atomic grid FAILED its fit for '
                    f'{", ".join(f"{el} {e:.2f}" for el, e in sorted(failed.items()))} '
                    f'at {self.basis}/{npts} points per atom: a fit error near 1 '
                    f'is a failed factorization, not a coarse grid, and every '
                    f'quantity built on it is unreliable. The lever is the point '
                    f'COUNT -- pass counts= with a larger grid; the default is '
                    f'sized for double zeta.', RuntimeWarning, stacklevel=2)
        self.M = len(self.owner)
        self.naux = self.auxmol(mol).nao_nr()
        # The layout is a screening threshold on collocated products of rank
        # 0's placed points, locked too: a rank whose threshold kept another
        # column set raises on every rank (the shapes disagree).
        t0 = time.perf_counter()
        self.layout = lockstep(test_set_layout(mol, self.coords(mol)))
        self.seconds['layout'] = time.perf_counter() - t0
        # Weak on the Mole: the value is arrays and pins nothing, so an entry
        # dies with its geometry. A weak key is wrong wherever the value
        # references the key -- `_environment_cache` holds `env.mol is mol`.
        self._fit_cache = weakref.WeakKeyDictionary()
        self.sliced = bool(sliced)
        self.fit = fit
        # resolved, so that two factorizations naming one tile edge match
        self.fit_block = (None if fit != 'rows' else
                          FIT_CHOLESKY_BLOCK if fit_block is None
                          else int(fit_block))
        # Sliced, what the chains share per geometry is the rows, keyed on
        # the mean field whose orbitals X_mo = X_ao C carries; weak, so the
        # rows go with it.
        self._rows_cache = weakref.WeakKeyDictionary()

    def slice_comm(self):
        """The communicator the factors are cut into grid rows over, or None
        where every rank holds them whole: unsliced, serially, on one rank."""
        if not self.sliced:
            return None
        comm = current_comm()
        return comm if comm is not None and comm.Get_size() > 1 else None

    def cached_rows(self, mol, mf, gauge):
        """The factors already built at `mol` for `mf` in this gauge and over
        the ranks they are laid out over now (`slice_comm`), or None: a
        `SlicedFactors` over more than one rank, and on one rank the row
        fit's whole (X_mo, D, X_ao).

        A hit takes no collective where a miss fits and locksteps, so the
        ranks must agree on which it is. They do: an entry is made at the same
        call on every rank, and it is looked up only through a mean field the
        caller still holds, whose entry cannot have been collected on one rank
        and not on another.
        """
        hit = self._rows_cache.get(mf)
        if (hit is not None and hit[0] is mol and hit[1] is gauge
                and hit[2] is self.slice_comm()):
            return hit[3]
        return None

    def keep_rows(self, mol, mf, gauge, rows):
        """Hold `rows`, built at `mol` for `mf` in `gauge` over the ranks of
        `slice_comm`, while `mf` lives."""
        self._rows_cache[mf] = (mol, gauge, self.slice_comm(), rows)

    def rebuilt_at(self, mol):
        """The same conventions re-derived at `mol`.

        Every owned choice travels verbatim (explicit radii included; atomic
        radii are element-only and rebuild identically); only the placed
        points, frames and pair layout are derived again. Callers rebuild
        through here so that no setting is dropped by one of them.
        """
        return type(self)(mol, basis=self.basis, auxbasis=self.auxbasis,
                          counts=self.counts, n_start=self.n_start,
                          frames=self.frames_mode,
                          radii=(self.radii if self.radii_tag is not None
                                 else None),
                          sliced=self.sliced, fit=self.fit,
                          fit_block=self.fit_block)

    def auxmol(self, mol):
        """The auxiliary molecule of the frozen auxiliary basis at `mol`."""
        return pyscf_df.addons.make_auxmol(mol, auxbasis=self.auxbasis)

    def coords(self, mol):
        """Interpolation points r_g = p_g F_i + R_i on frozen or continued frames.

        Rank 0's points on every rank: the grid is part of the functional.
        Continued frames are re-derived at every geometry through an `eigh`,
        which is where a rank's own bits enter.
        """
        fr = continued_frames(mol, self.frames) if self.with_frames \
            else self.frames
        return lockstep(np.vstack([self.pts_local[ia] @ fr[ia]
                                   + mol.atom_coord(ia)
                                   for ia in range(mol.natm)]))

    def shareable_factors(self, mol, auxmol, crd):
        """(X_ao, Mfit, V): the fit at `mol` before the auxiliary gauge.

        The gauge is left out because `aux_metric_sqrt` dresses it with the
        chain's own environment; everything before it is
        environment-independent and cached, so the chains of one composed
        surface pay it once.

        Under ranks the fit is rank 0's, and the lockstep is taken on every
        call, hit or miss, because a weak-keyed cache does not expire in step
        across ranks. Sliced over ranks nothing is cached here (the chains
        share the rows, `cached_rows`). The row fit refuses: `_fit` is another
        realization.
        """
        if self.fit == 'rows':
            raise ValueError(
                "fit='rows' never forms the whole fit, and `_fit` is another "
                "realization of it; read the factors through "
                "FactorChain.factors_at")
        if self.slice_comm() is not None:
            return lockstep(self._fit(mol, auxmol, crd))
        hit = self._fit_cache.get(mol)
        if hit is None:
            hit = self._fit(mol, auxmol, crd)
            self._fit_cache[mol] = hit
        return lockstep(hit)

    def _fit(self, mol, auxmol, crd):
        """(X_ao, M, V) at `mol`: `fit_M_streaming` on the frozen layout."""
        # every rank runs the serial fit, which `shareable_factors` locksteps
        M = fit_M_whole(mol, auxmol, crd, self.layout)
        return (mol.eval_gto('GTOval_sph', crd), M,
                auxmol.intor('int2c2e', aosym='s1'))

    def settings(self):
        """What two chains must agree on to be allowed to share one of these."""
        return (self.basis, self.auxbasis, tuple(sorted(self.counts.items()))
                if isinstance(self.counts, dict) else tuple(np.ravel(self.counts)),
                self.n_start, self.frames_mode, self.radii_tag)

    def require_match(self, basis, auxbasis, counts, n_start, frames,
                      radii=None, sliced=None, fit=None, fit_block=None):
        """Refuse a chain whose own settings contradict this factorization.

        Resolving silently to one side would give numbers of a functional
        nobody requested. A requested layout (`sliced` not None) must match
        too (same numbers, different memory per rank), and so must a requested
        fit realization (`fit`, `fit_block`): two realizations agree only to
        rounding.
        """
        if sliced is not None and bool(sliced) != self.sliced:
            raise ValueError(
                f'this chain asks for sliced={bool(sliced)} factors but the '
                f'shared factorization holds sliced={self.sliced}; build the '
                'factorization with the layout you want and pass it to both '
                'chains')
        block = (None if fit_block is None else int(fit_block))
        if ((fit is not None and fit != self.fit)
                or (block is not None and block != self.fit_block)):
            raise ValueError(
                f'this chain asks for the fit={fit!r} (fit_block={fit_block}) '
                f'realization but the shared factorization holds '
                f'fit={self.fit!r} (fit_block={self.fit_block}); build the '
                'factorization with the fit you want and pass it to both '
                'chains')
        other = FrozenFactorization.__new__(FrozenFactorization)
        other.basis = basis or self.basis
        other.auxbasis = auxbasis or default_auxbasis(other.basis)
        other.counts = counts or ISDF_DEFAULT_COUNTS
        other.n_start, other.frames_mode = n_start, frames
        # a chain with no opinion on radii accepts the factorization's
        other.radii_tag = radii_tag(radii) if radii is not None else self.radii_tag
        if other.settings() != self.settings():
            raise ValueError(
                f'this chain asks for {other.settings()} but the shared '
                f'factorization was built for {self.settings()}; build the '
                f'factorization with the settings you want and pass it to '
                f'both chains, or pass none and let each build its own')


class FactorChain:
    """Reference-frozen radii, layout and frames; the factors at any geometry;
    adjoints on (eps, X_mo, D) carried to the nuclei.

    `scf_factory(mol) -> mf` supplies the mean field at a displaced geometry;
    it must converge the orbital gradient to ~1e-11, since the Lagrangian
    assumes the occupied-virtual Fock block vanishes. A mean field returned
    built but not run is converged by the chain over the ranks
    (`converged_factory`); a converged one is used as it is.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole:
    the frozen factorization is rank 0's, mean fields are locked to rank 0's
    orbitals, and the gradient is one `lockstep` at the end of
    `nuclear_gradient`.

    `sliced` asks for a factorization whose factors are grid rows over the
    ranks (`FrozenFactorization`); None takes the given factorization's
    layout, or whole factors. A chain whose kernels do not all take
    `SlicedFactors` (`READS_SLICED_FACTORS`) refuses a sliced factorization at
    construction, as does one that reads the bare gauge (`READS_BARE_GAUGE`)
    in an environment that dresses the interaction.

    `fit` and `fit_block` select the fit's realization the same way; None
    takes the given factorization's, or the replicated fit. The row fit is
    refused in an environment that dresses the interaction: its D is one
    gauge's rows, Eq. (18) reads the bare gauge beside the dressed one, and
    its tiled adjoint carries the bare gauge alone.
    """

    #: Whether every kernel this chain calls reads `SlicedFactors`.
    READS_SLICED_FACTORS = False
    #: Whether this chain screens with the bare gauge beside the dressed one
    #: where the environment dresses the interaction (`bare_factor`).
    READS_BARE_GAUGE = False

    def __init__(self, mol, scf_factory, basis=None, auxbasis=None, counts=None,
                 n_start=1, frames='frozen', mf=None, environment=None,
                 factorization=None, radii=None, grid_accuracy=None,
                 sliced=None, fit=None, fit_block=None):
        self.mol0, self.scf_factory = mol, scf_factory
        # wall seconds of each step of this constructor, for the record
        self.construction_seconds = {}
        t0 = time.perf_counter()
        # Resolved before the branch, so that building a factorization and
        # matching a shared one are handed the same counts and recipe.
        if grid_accuracy is not None:
            basis = basis or mol.basis
            counts, n_start = resolve_isdf_grid(
                grid_accuracy, basis,
                {mol.atom_pure_symbol(i) for i in range(mol.natm)},
                auxbasis=auxbasis)
        if factorization is None:
            factorization = FrozenFactorization(mol, basis=basis,
                                                auxbasis=auxbasis, counts=counts,
                                                n_start=n_start, frames=frames,
                                                radii=radii,
                                                sliced=bool(sliced),
                                                fit=fit or 'replicated',
                                                fit_block=fit_block)
        else:
            factorization.require_match(basis, auxbasis, counts, n_start, frames,
                                        radii, sliced=sliced, fit=fit,
                                        fit_block=fit_block)
        self.construction_seconds['factorization'] = time.perf_counter() - t0
        if factorization.sliced and not self.READS_SLICED_FACTORS:
            raise ValueError(
                f'{type(self).__name__} reads the factors whole in kernels '
                'that do not take SlicedFactors; build its factorization with '
                'sliced=False')
        self.factorization = factorization
        self.basis, self.auxbasis = factorization.basis, factorization.auxbasis
        self.counts, self.n_start = factorization.counts, factorization.n_start
        self.frames_mode = factorization.frames_mode
        self.with_frames = factorization.with_frames
        self.sliced = factorization.sliced
        self.fit, self.fit_block = factorization.fit, factorization.fit_block
        self.nocc = mol.nelectron // 2
        # the one given, else the one attached to the reference mean field,
        # else the gas phase; a given mean field is completed by it (charges)
        self.environment = resolve_environment(environment, mf)
        self._environment_cache = collections.OrderedDict()
        # a dict to accumulate phase timings into, or None for no timing
        self.timer = None
        if (factorization.sliced and self.READS_BARE_GAUGE
                and dresses_interaction(self.environment_at(mol),
                                        self.auxmol(mol))):
            raise ValueError(
                f'{type(self).__name__} on sliced factors in '
                f'{self.environment!r}: '
                'the Eq. (18) reaction field (`reaction_field_shift`, '
                '`reaction_field_backward`) reads X_mo and D in both gauges '
                'whole beside every sweep, and the bare gauge is a second D '
                'the slices do not carry; build the factorization with '
                'sliced=False for a solvated run')
        if (factorization.fit == 'rows'
                and dresses_interaction(self.environment_at(mol),
                                        self.auxmol(mol))):
            raise ValueError(
                f"{type(self).__name__} on the row fit (fit='rows') in "
                f'{self.environment!r}: its D is one gauge\'s rows, and the '
                "row fit's tiled adjoint carries the bare gauge alone; build "
                "the factorization with fit='replicated', the same "
                'estimator, for a solvated run')
        t0 = time.perf_counter()
        with self.phase('t_scf'):
            self.mf0 = self.environment.mean_field(
                mol, converged_factory(
                    scf_factory if mf is None else (lambda _mol: mf)))
        # One reference on every rank from the start: what the chain reads off
        # mf0 directly must be rank 0's, and a given `mf=` may have been
        # converged on each rank alone.
        lockstep_mean_field(self.mf0)
        self.construction_seconds['mean_field'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.scf_residual = check_scf_quality(self.mf0, self.nocc)
        self.construction_seconds['scf_quality'] = time.perf_counter() - t0

        # the same objects, not copies, so two chains on one factorization
        # have bitwise identical factors
        self.radii = factorization.radii
        self.pts_local, self.owner = factorization.pts_local, factorization.owner
        self.frames = factorization.frames
        self.M, self.naux = factorization.M, factorization.naux
        self.layout = factorization.layout

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the reference pair, or a fresh SCF at another geometry,
        built in the chain's environment and converged over the ranks where
        the factory leaves that to the chain (`converged_factory`).
        """
        mol = self.mol0 if mol is None else mol
        if mf is None:
            if mol is self.mol0:
                mf = self.mf0
            else:
                # an abandoned mean field is a reference cycle (see
                # `response_kernel`), and numpy-sized garbage never wakes the
                # cyclic collector
                gc.collect()
                with self.phase('t_scf'):
                    mf = self.environment.mean_field(
                        mol, converged_factory(self.scf_factory))
        # Rank 0's orbitals on every rank: the chain forms X_mo = X_ao C and
        # contracts the kernels' adjoints with this mean field's coefficients,
        # and another rank's SCF can differ from rank 0's by the phases and
        # degenerate rotations of its orbitals. A mean field the ranks
        # converged together arrives locked and this rewrites the same bits;
        # one each rank converged alone does not.
        lockstep_mean_field(mf)
        return mol, mf

    @contextmanager
    def phase(self, key):
        """Time one stage into `self.timer[key]` (accumulating), or do nothing
        when `timer` is None."""
        if self.timer is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timer[key] = self.timer.get(key, 0.0) + time.perf_counter() - t0

    def require_differentiable_environment(self):
        """Refuse a force this environment cannot complete before the reverse
        pass runs (the environment's adjoints would raise only at its last
        branch)."""
        if not self.environment.differentiable:
            raise NotImplementedError(
                f'{self.environment!r} has no nuclear derivative, so this chain '
                'reports energies and refuses forces: returning the gas-phase '
                'adjoints would give a force that is wrong by the whole '
                'environment response and looks perfectly reasonable. Finite '
                'differences of the energy are exact (each displaced geometry '
                'rebuilds its own environment).')

    @staticmethod
    def is_kohn_sham(mf):
        """True when this mean field's e_tot is a KS energy, not E_HF."""
        return getattr(mf, 'xc', None) is not None

    def reference_energy(self, mol, mf):
        """E_0^HF of the plasmon formula: E_HF at this density.

        `mf.e_tot` on a Hartree-Fock mean field; on a Kohn-Sham one the
        Hartree-Fock energy of its density (`rpa_energy.reference_energy`).
        Take it together with `kohn_sham_gradient_correction`, which moves the
        force the same way: a chain that adopts one and not the other is
        stationary for neither functional.
        """
        return reference_energy(mf, mol)

    def kohn_sham_gradient_correction(self, mol, mf):
        """(y_extra, g_extra): the gradient-side partner of `reference_energy`.

        The EXX double-counting orbital response (joins the Lagrangian before
        the multiplier solve, sharing Lambda) and its skeleton (joins the
        orbital branch). The `y_extra`/`g_extra` hooks of `nuclear_gradient`
        are shared: sum into what a chain already passes. Both vanish on a
        Hartree-Fock mean field, so callers add them unconditionally.

        `g_extra` is differenced against `mean_field_gradient`, which must
        carry the grid response. A chain with no EXX skeleton of its own
        should refuse a Kohn-Sham reference outright. The dense surfaces
        carry the same two terms instead
        (`dense_surfaces.kohn_sham_gradient_correction`), paired with a
        grid-response mean-field force (`mean_field_skeleton_force`).
        """
        return (exx_double_counting_Y(mf, self.nocc),
                exx_double_counting_skeleton(mf, mol))

    def mean_field_gradient(self, mf):
        """pyscf's force for this mean field, with the grid response paired.

        The half of `kohn_sham_gradient_correction` outside the Lagrangian:
        `grid_response=True` on a KS gradient is the partner of the
        double-counting skeleton, not an accuracy knob.

        Rank 0's force on every rank: pyscf blocks a DF gradient's auxiliary
        index by the process's free memory, so ranks re-associate the same
        sums differently.
        """
        if isinstance(getattr(mf, 'with_df', None), ISDFJK):
            # `isdf_mean_field_gradient` sets grid_response on the reference it
            # differentiates, so the quadrature's own motion is carried there.
            return lockstep(np.asarray(mean_field_skeleton_force(mf)))
        g0 = mf.Gradients()
        if self.is_kohn_sham(mf):
            g0.grid_response = True
        return lockstep(np.asarray(g0.kernel()))

    def environment_at(self, mol):
        """The environment around this geometry, built once per geometry.

        A cavity moves with the atoms, so each displaced geometry gets its
        own. The key is the exact bytes of the atomic charges and coordinates
        and the atom count: charges because a cavity follows element radii,
        unrounded because rounding lets two finite-difference displacements
        share a reaction field (a wrong force no gate would report). Content
        rather than `id(mol)`, which is safe only by accident.
        """
        key = (mol.atom_charges().tobytes(), mol.atom_coords().tobytes(),
               mol.natm)
        hit = self._environment_cache.get(key)
        if hit is not None:
            self._environment_cache.move_to_end(key)
            return hit
        hit = self.environment.for_geometry(mol)
        self._environment_cache[key] = hit
        while len(self._environment_cache) > ENVIRONMENT_CACHE_SIZE:
            self._environment_cache.popitem(last=False)
        return hit

    def auxmol(self, mol):
        """The auxiliary molecule of the frozen auxiliary basis at `mol`."""
        return self.factorization.auxmol(mol)

    def coords(self, mol):
        """Interpolation points r_g = p_g F_i + R_i on frozen or continued frames."""
        return self.factorization.coords(mol)

    def factors(self, mol, auxmol, crd):
        """(X_ao, D) of the separable RI on the frozen layout, D = M^T V^(1/2).

        The least-squares fit target keeps the bare metric; only the gauge
        V^(1/2) is dressed by the environment at this geometry, as in
        `space_time.separable_factors`.
        """
        X_ao, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return (X_ao,
                Mfit.T @ aux_metric_sqrt(auxmol, self.environment_at(mol), V=V))

    def bare_factor(self, mol, auxmol, crd):
        """D in the bare gauge, or None when nothing screens.

        The self-energy screens with the bare interaction and takes the
        continuum as Duchemin et al.'s static Eq. (18) shift, while the BSE
        kernel keeps the dressed one, so a solvated chain carries both factors.
        They share the fit M and differ only in the metric root, which is one
        naux eigendecomposition.
        """
        if not dresses_interaction(self.environment_at(mol), auxmol):
            return None
        _, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return Mfit.T @ aux_metric_sqrt(auxmol, None, V=V)

    def static_factor(self, mol, auxmol, crd):
        """(the static partner of this geometry's continuum, D in its gauge).

        The ion's equilibrium solvation reads Eq. (18) at eps_static on the
        same cavity (`SolventScreening.static_partner`). It shares the fit M
        with D and D_bare and differs only in the metric root,
        (V + vtilde_static)^(1/2): one more naux eigendecomposition.
        """
        environment = self.environment_at(mol)
        if not hasattr(environment, 'static_partner'):
            raise ValueError(
                f'equilibrium solvation needs a continuum with a static '
                f'response; this chain carries {environment!r}')
        partner = environment.static_partner()
        _, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return partner, Mfit.T @ aux_metric_sqrt(auxmol, partner, V=V)

    def factors_at(self, mol, mf):
        """(X_mo, D, eps, auxmol, coords, X_ao) at `mol`, under one timing phase.

        Sliced over ranks, X_mo, D and X_ao are one `SlicedFactors` holding
        this rank's rows of each (`sliced_factors_at`); kernels gather what
        they read whole. On the row fit they are the rows it built, and on one
        rank its whole arrays.
        """
        auxmol = self.auxmol(mol)
        crd = self.coords(mol)
        eps = np.asarray(mf.mo_energy, float)
        if (self.factorization.fit == 'rows'
                or self.factorization.slice_comm() is not None):
            with self.phase('t_factors'):
                rows = self.sliced_factors_at(mol, mf, auxmol, crd)
            if isinstance(rows, SlicedFactors):
                return rows, rows, eps, auxmol, crd, rows
            x_mo, d, x_ao = rows
            return x_mo, d, eps, auxmol, crd, x_ao
        with self.phase('t_factors'):
            x_ao, d = self.factors(mol, auxmol, crd)
        return (x_ao @ mf.mo_coeff, d, eps, auxmol, crd, x_ao)

    def sliced_factors_at(self, mol, mf, auxmol, crd):
        """This rank's grid rows of (X_mo, D, X_ao) at `mol`: `SlicedFactors`,
        or on the row fit over one rank its whole (X_mo, D, X_ao).

        On the replicated fit the rows are cut from the whole products
        X_mo = X_ao C and D = M^T V^(1/2), built as the whole layout builds
        them, since a row block of a GEMM is not the same bits as those rows
        of the whole GEMM; the whole arrays are dropped on return. On the row
        fit they are `row_fit_factors`' output and nothing whole exists.

        The rows are shared through the factorization with every chain asking
        at the same geometry, mean field and gauge, cut for the ranks of the
        current region (`slice_comm`).
        """
        env = self.environment_at(mol)
        # The gauge is bare wherever the environment dresses nothing, so every
        # such chain shares one D.
        gauge = env if getattr(env, 'screens', True) else None
        hit = self.factorization.cached_rows(mol, mf, gauge)
        if hit is not None:
            return hit
        if self.factorization.fit == 'rows':
            rows = self.row_fit_factors(mol, mf, auxmol, crd)
        else:
            x_ao, d = self.factors(mol, auxmol, crd)
            rows = SlicedFactors.from_whole((x_ao @ mf.mo_coeff, d, x_ao, crd),
                                            self.factorization.slice_comm())
        self.factorization.keep_rows(mol, mf, gauge, rows)
        return rows

    def row_fit_factors(self, mol, mf, auxmol, crd):
        """(X_mo, D, X_ao) of the row-distributed fit on the frozen points:
        `SlicedFactors` over more than one rank, whole arrays on one.

        `separable_ri.fit_rows` solves the balanced, regularized estimator on
        the frozen pair layout, the reference geometry's screen that the fit
        adjoint differentiates (screened again here, a pair crossing the
        threshold would step the energy under a force that cannot see it), in
        fixed tiles of `fit_block` grid points, so each rank holds only its
        own tiles and its rows are bitwise the one-rank rows. The metric root
        is rank 0's, since every tile owner projects with it. Not
        `separable_factors(fit='rows')`, which places its own points on frames
        recomputed per geometry.

        Where `mf` is a distributed ISDF-K SCF whose grid is these points and
        whose own screen kept the frozen layout's pairs (the reference
        geometry), the rows are read from its handle's M^T tiles
        (`separable_ri.FitTiles`) instead of fitted again.
        """
        comm = self.factorization.slice_comm()
        handle = isdf_scf_handle(mf)
        fit = fit_M_streaming(mol, auxmol, crd, fit='rows',
                              block=self.factorization.fit_block,
                              layout=self.factorization.layout,
                              tiles=None if handle is None
                              else handle.fit_tiles())
        x_mo = fit.mo_rows(mf.mo_coeff)
        d = fit.metric_root_rows(auxmol, self.environment_at(mol))
        x_ao = fit.ao_rows()
        if comm is None:
            return x_mo, d, x_ao
        return SlicedFactors.from_rows(x_mo, d, x_ao, crd, comm,
                                       fit_held=fit.held)

    def nuclear_gradient(self, mol, mf, auxmol, crd, x_mo, eps_bar, x_bar,
                         d_bar, y_extra=None, g_extra=None, d_bar_bare=None,
                         root_bar=None, kernel_bar=None, extra_gauges=()):
        """(natm, 3) gradient from adjoints on (eps, X_mo, D), with diagnostics.

        X_mo = X_ao C depends on the geometry through C (the orbital-rotation
        gradient X_mo^T X_bar in the Lagrangian) and through X_ao (adjoint
        X_bar C^T).

        eps_bar: adjoint on the orbital energies, or a full symmetric MO Fock
            partial dE/dF_pq when the target's one-body term contracts F off
            the diagonal (an active-space model does).
        y_extra: orbital-rotation gradient of a term that is not a function
            of the factors (e.g. the Kohn-Sham static correction); joins Y
            before the multiplier solve, sharing Lambda.
        g_extra: that term's skeleton, added to the orbital branch.
        d_bar_bare: adjoint on the bare-gauge factor when the target screens
            with both (`bare_factor`); the cavity's motion reaches the force
            through the dressed factor only.
        root_bar, kernel_bar: adjoints on the dressed metric root
            (V + vtilde)^(1/2) and on vtilde alone, from a term that reads the
            screened metric without D (the fold's N = R^-1 vtilde R^-1). They
            join the dressed gauge so one Frechet solve and one cavity
            derivative serve both.
        extra_gauges: further `GaugeAdjoint`s of the same fit (e.g. the
            static partner's factor of an equilibrium-solvated ion,
            `static_factor`).

        x_mo may be `SlicedFactors`: X_mo is gathered whole for X_mo^T X_bar
        alone (on the row fit its tiles stream, `orbital_rotation_rows`), and
        the diagnostics carry `factor_gathers` and, on the row fit, `fit_held`
        (rank 0's peak bytes per fit array). x_bar and d_bar may be
        `GridTileRows`: the row fit reads its own tiles, the replicated fit
        gathers them.

        The fit adjoint differentiates the one estimator on both
        realizations: whole (`dfactor_adjoint_gauges` over `product_pairs`) on
        the replicated fit, in the fit's tiles (`row_fit_branches`) on the row
        fit, where `adjoint_held` reports rank 0's peak bytes per array.

        The assembly is a `one_fit_adjoint` window on `mf`: row exchange
        skeletons inside it (the relaxed density's, the caller's
        Sigma_x - v_xc) leave their seeds to the row fit's adjoint, one
        `fit_rows_adjoints` call per fit, and their share joins the orbital
        branch.
        """
        with one_fit_adjoint(mf) as pending:
            if self.factorization.fit != 'rows':
                x_bar, d_bar, d_bar_bare = (
                    a.gather() if isinstance(a, GridTileRows) else a
                    for a in (x_bar, d_bar, d_bar_bare))
            if self.factorization.fit == 'rows':
                with self.phase('t_orbital'):
                    y = orbital_rotation_rows(x_mo, x_bar, self.fit_block)
            else:
                # refused before the orbital response rather than after it
                require_whole_fit_adjoint(mol.nao_nr(), auxmol.nao_nr(),
                                          len(crd), len(product_pairs(mol)[0]),
                                          'FactorChain.nuclear_gradient')
                y = (x_mo.require(current_comm()).gather('X_mo')
                     if isinstance(x_mo, SlicedFactors) else x_mo).T @ x_bar
            if y_extra is not None:
                y = y + y_extra
            with self.phase('t_orbital'):
                g_orb, diags = eps_chain_gradient(mf, eps_bar, self.nocc,
                                                  Y_extra=y)
            if g_extra is not None:
                g_orb = g_orb + g_extra
            held = None
            if self.factorization.fit == 'rows':
                if extra_gauges:
                    raise NotImplementedError(
                        "a further gauge on the row fit: fit='rows' refuses "
                        'every environment that dresses the interaction')
                g_coll, g_fit, held = self.row_fit_branches(
                    mol, mf, auxmol, crd, x_bar, d_bar, d_bar_bare=d_bar_bare,
                    root_bar=root_bar, kernel_bar=kernel_bar)
            else:
                g_coll, g_fit = self.whole_fit_branches(
                    mol, mf, auxmol, crd, x_bar, d_bar, d_bar_bare=d_bar_bare,
                    root_bar=root_bar, kernel_bar=kernel_bar,
                    extra_gauges=extra_gauges)
            with self.phase('t_fit'):
                g_orb = g_orb + pending.settle()
        return self._assembled(g_orb, g_coll, g_fit, diags, x_mo,
                               adjoint_held=held)

    def one_fit_adjoint(self, mf, mean_field=False):
        """The window of one force on `mf` whose row exchange skeletons hand
        their fit adjoint's seeds to the assembly
        (`isdf_derivatives.one_fit_adjoint`); mean_field: this chain's
        `mean_field_gradient(mf)` follows inside it and rides the same
        call."""
        return one_fit_adjoint(mf, mean_field=mean_field)

    def whole_fit_branches(self, mol, mf, auxmol, crd, x_bar, d_bar,
                           d_bar_bare=None, root_bar=None, kernel_bar=None,
                           extra_gauges=()):
        """(g_collocation, g_fit) on the replicated fit, both formed whole:
        `collocation_adjoint` of X_bar C^T and `dfactor_adjoint_gauges` over
        every product pair; `extra_gauges` as `nuclear_gradient`'s."""
        with self.phase('t_collocation'):
            g_coll = collocation_adjoint(mol, crd, x_bar @ mf.mo_coeff.T,
                                         self.pts_local, self.owner,
                                         frames=self.frames,
                                         with_frames=self.with_frames)
        with self.phase('t_fit'):
            # Both gauges of one fit in one pass: the dressed factor the kernel
            # uses and the bare one the self-energy screens with share the test
            # set, the three-centre integrals and M, and differ only in the
            # metric root.
            gauges = [GaugeAdjoint(d_bar, self.environment_at(mol),
                                   root_bar=root_bar, kernel_bar=kernel_bar)]
            if d_bar_bare is not None:
                gauges.append(GaugeAdjoint(d_bar_bare, None))
            gauges.extend(extra_gauges)
            g_fit = dfactor_adjoint_gauges(mol, auxmol, crd, gauges,
                                           self.layout, self.pts_local,
                                           self.owner, frames=self.frames,
                                           with_frames=self.with_frames,
                                           gram_layout=product_pairs(mol))
        return g_coll, g_fit

    def row_fit_branches(self, mol, mf, auxmol, crd, x_bar, d_bar,
                         d_bar_bare=None, root_bar=None, kernel_bar=None):
        """(g_collocation, g_fit, held) on the row fit: the fit adjoint of
        `separable_ri.fit_rows` on the frozen pair layout and the X_mo
        collocation adjoint, in the fit's tiles
        (`isdf_derivatives.row_fit_adjoint`). No rank forms the Gram matrix,
        the test-set collocation, (mu nu|P), M, F D^T or an adjoint of their
        size whole, and the bits are the same at every rank count. held: the
        peak bytes of each array this rank held.

        One gauge, the bare metric (the chain refuses the row fit in a
        dressing environment). Inside the force's `one_fit_adjoint` window the
        call also carries the skeletons' seeds (`PendingFitAdjoint.contract`).
        """
        if not (d_bar_bare is None and root_bar is None
                and kernel_bar is None):
            raise ValueError(
                'the row fit differentiates one bare gauge; a bare-gauge '
                'adjoint beside it or one on a dressed metric root belongs '
                "to a dressing environment, which fit='rows' refuses")
        with self.phase('t_fit'):
            adjoint = row_fit_adjoint(mol, auxmol, crd, d_bar, x_bar,
                                      mf.mo_coeff, self.layout, self.pts_local,
                                      self.owner, frames=self.frames,
                                      with_frames=self.with_frames,
                                      block=self.fit_block,
                                      pending=pending_fit_adjoint(mf))
        return adjoint

    def _assembled(self, g_orb, g_coll, g_fit, diags, x_mo,
                   adjoint_held=None):
        """(grad, diagnostics) of `nuclear_gradient` from its three
        branches, rank 0's on every rank."""
        grad = g_orb + g_coll + g_fit
        # an exact gradient sums to zero over the atoms: a free check on every branch
        diags = dict(diags,
                     translation_residual=float(np.abs(grad.sum(axis=0)).max()),
                     branch_orbital=float(np.abs(g_orb).max()),
                     branch_collocation=float(np.abs(g_coll).max()),
                     branch_fit=float(np.abs(g_fit).max()))
        if isinstance(x_mo, SlicedFactors):
            diags['factor_gathers'] = dict(x_mo.gathers)
        if getattr(x_mo, 'fit_held', None) is not None:
            diags['fit_held'] = dict(x_mo.fit_held)
        if adjoint_held is not None:
            diags['adjoint_held'] = dict(adjoint_held)
        # Rank 0's gradient on every rank: the kernels returned one set of
        # adjoints, and the orbital response, collocation and fit branches each
        # rank forms from them carry its own last bits, so without this the
        # ranks would disagree by what they added alone.
        return lockstep((grad, diags))


def check_scf_quality(mf, nocc, tol=SCF_GRAD_TOL, raise_on_fail=False):
    """max |F_ia| in the MO basis -- the quantity `conv_tol_grad` controls.

    The Lagrangian assumes this block vanishes, what is left is amplified in
    the force, and symmetry hides it: a symmetric molecule looks converged and
    is wrong in the fourth digit. Under ranks the Fock matrix is the
    distributed SCF's own (`distributed_fock`).
    """
    c = mf.mo_coeff
    with distributed_fock(mf, build=False):
        fock = mf.get_fock()
    resid = float(np.abs((c.T @ fock @ c)[:nocc, nocc:]).max())
    if resid > tol:
        msg = (f'the mean field carries |F_ia| = {resid:.2e}, above {tol:.0e}; '
               'the gradient Lagrangian assumes it vanishes. Set '
               'conv_tol_grad=1e-11.')
        if raise_on_fail:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
    return resid
