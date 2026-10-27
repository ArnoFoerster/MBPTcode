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
(src.environments.environment), which builds the mean field at each geometry, dresses
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
`isdf_fit_adjoint.fit_rows_adjoint`), the same bits at every rank count. Per rank
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
import weakref

import numpy as np
from pyscf import df as pyscf_df

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.geometry.body_frame import BodyFrame
from src.Base.factorization.isdf_jk import ISDFJK
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.gradients.surface_identity import chain_identity
from src.Base.numerics.constants import (ENVIRONMENT_CACHE_SIZE,
                                         EVALUATION_CACHE_SIZE, FIT_CHOLESKY_BLOCK,
                                         FIT_REALIZATIONS,
                                         ISDF_FIT_ERROR_FAILED)
from src.Base.numerics.numerics import (Numerics, numerics_or_default,
                                        require_same_numerics)
from src.environments.environment import dresses_interaction
from src.Base.environment_seam import resolve_environment
from src.Base.geometry.geometry import elements_of
from src.Base.factorization.separable_ri import (atomic_frames, aux_metric_sqrt,
                                                 fit_M_streaming, fit_M_whole, fit_settings,
                                                 refuse_unrepresented_factors,
                                                 resolve_isdf_grid, runtime_atomic_radii,
                                                 subshells, fit_test_set_layout)
from src.Base.factorization.separable_ri import require_grid
from src.Base.factorization.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.Base.utils.timing import stage
from src.SingleReference.LinearResponse.rpa_energy import reference_energy
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            dfactor_adjoint_gauges,
                                            GaugeAdjoint, isdf_scf_handle,
                                            one_fit_adjoint,
                                            orbital_rotation_rows,
                                            pending_fit_adjoint, point_layout,
                                            product_pairs,
                                            require_whole_fit_adjoint,
                                            row_fit_adjoint)
from src.gradients.mean_field_response import eps_chain_gradient
from src.gradients.rpa.exchange_double_counting import (exx_double_counting_Y,
                                                        exx_double_counting_skeleton)
from src.gradients.reference_mean_field import ReferenceMeanField

#: What `FrozenFactorization.to_record` keeps beside the body frame.
FACTORIZATION_RECORD = ('basis', 'auxbasis', 'counts', 'n_start', 'radii_tag',
                        'radii', 'fit_errors', 'seconds', 'pts_local', 'owner',
                        'frames', 'M', 'naux', 'layout', 'sliced', 'fit',
                        'fit_block')


def radii_tag(radii):
    """Hashable summary of an explicit radii set, or None when the atomic
    optimizer decides them: two factorizations differing only in their radii are
    different functionals and must not share."""
    if radii is None:
        return None
    return tuple((el, tuple(np.round(np.concatenate(
        [np.atleast_1d(radii[el][s]) for s in sorted(radii[el])]), 12)))
        for el in sorted(radii))


class GaugeRows:
    """Every gauge's factor rows built at one geometry for one mean field
    over one rank layout: {gauge: rows}, a gauge being the environment that
    dresses the metric root, or None for the bare one.

    Each gauge D_g = M^T (V + vtilde_g)^1/2 of the one fit is a right
    product of the same rows of M^T, so a gauge asked after the first is
    formed from `source`, the row fit's M^T tiles kept for that, with no
    second fit; on the replicated fit `source` is None and the whole fit is
    re-run for it. The X_mo and X_ao rows are the first gauge's, shared.
    `layout` is the communicator the rows were cut over (`slice_comm`), the
    entry's key beside the geometry, never one to reach the ranks through.
    """

    def __init__(self, mol, layout, source, serial):
        self.mol, self.layout, self.source = mol, layout, source
        self.serial = serial
        self.gauges = []

    def get(self, gauge):
        """The rows of `gauge`, or None."""
        for g, rows in self.gauges:
            if g is gauge:
                return rows
        return None

    def put(self, gauge, rows):
        self.gauges.append((gauge, rows))

    def first(self):
        """The rows built first, whose X_mo and X_ao every gauge shares."""
        return self.gauges[0][1]


class FrozenFactorization:
    """The ISDF conventions fixed at ONE reference geometry, so that several
    chains are ONE functional rather than two that happen to agree.

    Radii, interpolation points, frames and pair layout are discrete choices.
    Re-deciding them per geometry puts a step in the energy; deciding them
    twice, once per chain, is subtler -- `separable_factors` is deterministic
    given the settings, so two chains handed matching settings produce bitwise
    identical factors and merely pay twice. They diverge silently the moment
    the settings stop matching: different `counts`, `n_start` or radii.

    Holding them in one object makes the sharing structural rather than a
    property of the caller passing the same keywords, and a composed surface
    then costs ONE fit per geometry instead of one per chain.

    It owns exactly what `separable_factors` consumes and nothing else. The
    QUADRATURES ARE NOT HERE: E_c^dRPA integrates chi0 over frequency and the
    self-energy over imaginary time, different integrands with no reason to
    share a grid, so each chain keeps its own. Nor is the quasiparticle window,
    which belongs to whichever chain has one.

    The distinction that keeps this boundary from drifting: `refreeze` carries
    a quadrature so that it does not CHANGE BETWEEN GEOMETRIES, which is not
    the same requirement as being SHARED BETWEEN CHAINS. Only the second is
    this object's business.

    Its choices are `numerics`' (`Numerics`): the `isdf` section's basis,
    auxiliary basis, grid, radii and fit realization, the `layout` section's
    `sliced` and `fit_block`. The level (`isdf.level`) is the named way to
    fix `counts` and `n_start` together -- a validated accuracy level or four
    explicit shell counts, resolved against the shipped radii table by
    `resolve_isdf_grid`, which refuses a grid nobody optimized rather than
    re-optimizing one per geometry. The fit's own choices (l_max_second,
    pair_tol, regularization) are refused where they are not the defaults:
    the chains' fit adjoints differentiate the default fit, so a forward on
    another would carry a force of a different function.

    UNDER RANKS THE CONVENTIONS ARE RANK 0'S, which is the same requirement
    one step further out. Every rank decides them, as serial code, and one
    `lockstep` then leaves rank 0's radii, clouds, owners and frames on all of
    them; the placed points, the pair layout and the fit are locksteps of
    their own, taken where they are built. Deciding them per rank is invisible
    on one machine -- two processes there run the same arithmetic on the same
    libraries and agree bit for bit -- and across NODES it puts the ranks on
    different surfaces: the route gates pass on every rank while the
    end-to-end forces of ranks that froze their own miss by up to 6e-3
    Ha/Bohr. Serially every lockstep is a no-op and the arithmetic is what it
    was.

    `sliced` is the LAYOUT of the factors, not a choice of functional, and is
    not among `settings`: over more than one rank each rank keeps its grid
    rows of X_mo, D and X_ao (`FactorChain.factors_at`) and no whole fit
    outlives the cut. Serially, and on one rank, it changes nothing.

    `fit` is the REALIZATION of the fit, and like `sliced` it is matched
    apart from `settings`; both solve ONE estimator, `fit_M_streaming`'s on
    the frozen `layout`: the Gram matrix over every product pair
    (A A^T) o (B B^T) + P P^T, F D^T over the layout's pairs. 'replicated'
    is `_fit`, `fit_M_streaming` itself, run whole by every rank.
    'rows' (with `sliced=True`) is `separable_ri.fit_rows` on the frozen
    points, `fit_block` points per tile: no rank holds the Gram matrix,
    F D^T, the solve, the collocation or a factor whole, and the rows are
    bitwise the same at every rank count, one rank included -- not the
    replicated fit's bits, but within a few of its reassociation responses
    (`FIT_REASSOCIATION_K`). `FactorChain.nuclear_gradient` differentiates
    that estimator on either.
    """

    def __init__(self, mol, numerics=None):
        numerics = numerics_or_default(numerics)
        # Refused before the radii are optimized or anything is placed.
        self._refuse_fit_layout(numerics)
        elements = elements_of(mol)
        self._resolve_grid(mol, numerics.isdf, elements)
        self._place_points(mol, numerics.isdf, elements)
        self._lay_out(mol)
        self._open_caches(numerics)

    @staticmethod
    def _realization(numerics):
        """(fit, sliced, fit_block) as `numerics` asks for them."""
        layout = numerics.layout
        return (numerics.isdf.fit or 'replicated', bool(layout.sliced),
                layout.fit_block)

    @classmethod
    def _refuse_fit_layout(cls, numerics):
        """Refuse a fit that is not the default one, and a realization or
        tile edge the layout cannot carry."""
        fit, sliced, fit_block = cls._realization(numerics)
        require_default_fit(numerics)
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

    def _resolve_grid(self, mol, isdf, elements):
        """The basis pair and the grid's counts (a level resolved into them);
        the explicit radii are checked against the molecule's elements."""
        basis, auxbasis, counts, radii = (isdf.basis, isdf.auxbasis,
                                          isdf.counts, isdf.radii)
        n_start = 1 if isdf.n_start is None else isdf.n_start
        grid_accuracy = isdf.level
        self.basis = basis or str(mol.basis)
        self.auxbasis = auxbasis or paired_auxbasis(self.basis)
        # Resolved here and stored as plain counts, so `rebuilt_at`,
        # `settings` and `require_match` compare one kind of object and a
        # level cannot mean two grids.
        if grid_accuracy is not None:
            counts, n_start = resolve_isdf_grid(grid_accuracy, self.basis,
                                                elements,
                                                auxbasis=self.auxbasis)
        require_grid(counts, radii, 'FrozenFactorization')
        self.counts = counts
        self.n_start = n_start
        # Explicit radii replace the
        # per-element atomic optimization. They are frozen here like every other
        # discrete choice, so a grid tailored at the reference geometry is the
        # one the whole walk uses.
        self.radii_tag = radii_tag(radii)
        if radii is not None and not set(elements) <= set(radii):
            raise ValueError(f'radii given for {sorted(radii)} but the molecule '
                             f'holds {elements}')

    def _place_points(self, mol, isdf, elements):
        """The radii (`isdf.radii`, or optimized per element where none were
        given), the points they place, their owners and the frames, rank 0's
        on every rank; a warning where an element's fit failed, and the
        element in `fit_failed`, which `held_grid` reports."""
        radii = isdf.radii
        # The fit error is kept: it is the only signal of whether this grid
        # resolves this basis. The default count is sized for double zeta and
        # fails for most elements at larger bases.
        fit_errors = {}
        # the elements whose fit failed (a numerical flag, warned about)
        self.fit_failed = []
        # wall seconds of each convention's construction, for the record
        self.seconds = {}
        t0 = time.perf_counter()
        if radii is None:
            radii = {}
            for el in elements:
                radii[el], fit_errors[el], _ = runtime_atomic_radii(
                    el, self.basis, self.auxbasis, self.counts,
                    n_start=self.n_start, runtime_radii=isdf.runtime_radii)
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
        # Frozen frames are frozen in the molecule: they turn with the Kabsch
        # rotation of each geometry onto this one, so a rigid motion moves
        # the points with the atoms and the energy cannot see it.
        self.body = BodyFrame(lockstep(np.array(mol.atom_coords())))
        if self.radii_tag is None:
            failed = {el: e for el, e in self.fit_errors.items()
                      if e >= ISDF_FIT_ERROR_FAILED}
            if failed:
                npts = sum(len(subshells()[n]) * len(r)
                           for n, r in next(iter(self.radii.values())).items())
                self.fit_failed = sorted(failed)
                warnings.warn(
                    f'ISDF atomic grid FAILED its fit for '
                    f'{", ".join(f"{el} {e:.2f}" for el, e in sorted(failed.items()))} '
                    f'at {self.basis}/{npts} points per atom: a fit error near 1 '
                    f'is a failed factorization, not a coarse grid, and every '
                    f'quantity built on it is unreliable. The lever is the point '
                    f'COUNT -- pass counts= with a larger grid; the default is '
                    f'sized for double zeta.', RuntimeWarning, stacklevel=3)

    def _lay_out(self, mol):
        """The point count, the auxiliary size and the AO-pair layout."""
        self.M = len(self.owner)
        self.naux = self.auxmol(mol).nao_nr()
        # The layout is a screening threshold on collocated products of rank
        # 0's placed points, locked too: a rank whose threshold kept another
        # column set raises on every rank (the shapes disagree).
        t0 = time.perf_counter()
        self.layout = lockstep(fit_test_set_layout(mol, self.coords(mol)))
        self.seconds['layout'] = time.perf_counter() - t0

    def _open_caches(self, numerics):
        """The fit and row caches, and the realization the factors are built
        in."""
        fit, sliced, fit_block = self._realization(numerics)
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
        # rows go with it. One `GaugeRows` per mean field, every gauge's.
        self._rows_cache = weakref.WeakKeyDictionary()
        self._kept = 0

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
        entry = self.rows_entry(mol, mf)
        return None if entry is None else entry.get(gauge)

    def keep_rows(self, mol, mf, gauge, rows, source=None):
        """Hold `rows`, built at `mol` for `mf` in `gauge` over the ranks of
        `slice_comm`, while `mf` lives; `source`, the row fit's M^T tiles
        further gauges are formed from (`GaugeRows`)."""
        entry = self.rows_entry(mol, mf)
        if entry is None:
            self._kept += 1
            entry = GaugeRows(mol, self.slice_comm(), source, self._kept)
            self._rows_cache[mf] = entry
        entry.put(gauge, rows)

    def rows_entry(self, mol, mf=None):
        """The `GaugeRows` held at `mol` for `mf` over the current ranks, or
        None; mf None takes the entry kept last at `mol`."""
        layout = self.slice_comm()
        if mf is not None:
            entry = self._rows_cache.get(mf)
            return (entry if entry is not None and entry.mol is mol
                    and entry.layout is layout else None)
        found = [e for e in self._rows_cache.values()
                 if e.mol is mol and e.layout is layout]
        return max(found, key=lambda e: e.serial) if found else None

    def rebuilt_at(self, mol):
        """The same conventions re-derived at `mol`.

        Every owned choice travels verbatim (explicit radii included; atomic
        radii are element-only and rebuild identically); only the placed
        points, frames and pair layout are derived again. Callers rebuild
        through here so that no setting is dropped by one of them.
        """
        return type(self)(mol, Numerics(
            isdf={'basis': self.basis, 'auxbasis': self.auxbasis,
                  'counts': self.counts, 'n_start': self.n_start,
                  'radii': (self.radii if self.radii_tag is not None
                            else None),
                  'fit': self.fit},
            layout={'sliced': self.sliced, 'fit_block': self.fit_block}))

    def to_record(self):
        """Every convention this object owns -- the settings, radii,
        placed points, owners, frames, body frame and pair layout -- for
        `from_record` (`record_io` writes it)."""
        return {name: getattr(self, name) for name in FACTORIZATION_RECORD} | {
            'body_reference': self.body.reference}

    @classmethod
    def from_record(cls, record):
        """The factorization `to_record` wrote, with nothing placed or
        optimized again, and empty caches."""
        out = cls.__new__(cls)
        for name in FACTORIZATION_RECORD:
            setattr(out, name, record[name])
        out.body = BodyFrame(record['body_reference'])
        out._fit_cache = weakref.WeakKeyDictionary()
        out._rows_cache = weakref.WeakKeyDictionary()
        out._kept = 0
        return out

    def auxmol(self, mol):
        """The auxiliary molecule of the frozen auxiliary basis at `mol`."""
        return pyscf_df.addons.make_auxmol(mol, auxbasis=self.auxbasis)

    def coords(self, mol):
        """Interpolation points r_g = p_g F_i + R_i on the frozen frames: the
        reference's turned by the body frame, F0_i Q (`BodyFrame`), and the
        reference's own bits at the reference.

        Rank 0's points on every rank: the grid is part of the functional.
        Continued frames are re-derived at every geometry through an `eigh`,
        which is where a rank's own bits enter.
        """
        q = self.body.rotation(mol.atom_coords())
        fr = self.frames if q is None else self.frames @ q
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
                self.n_start, self.radii_tag)

    def require_match(self, numerics):
        """Refuse a chain whose own settings (`numerics`) contradict this
        factorization.

        Resolving silently to one side would give numbers of a functional
        nobody requested. A requested layout (`sliced` not None) must match
        too (same numbers, different memory per rank), and so must a requested
        fit realization (`fit`, `fit_block`): two realizations agree only to
        rounding.
        """
        require_default_fit(numerics)
        isdf, layout = numerics.isdf, numerics.layout
        basis, auxbasis, counts, radii = (isdf.basis, isdf.auxbasis,
                                          isdf.counts, isdf.radii)
        n_start, fit = isdf.n_start, isdf.fit
        sliced, fit_block = layout.sliced, layout.fit_block
        # the named way in resolves here as it does where one is built
        if isdf.level is not None:
            elements = {mol_el for mol_el in self.radii}
            counts, n_start = resolve_isdf_grid(
                isdf.level, basis or self.basis, elements,
                auxbasis=auxbasis or paired_auxbasis(basis or self.basis))
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
        other.auxbasis = auxbasis or paired_auxbasis(other.basis)
        # a chain with no opinion on the grid accepts the factorization's
        other.counts = self.counts if counts is None else counts
        # a chain with no opinion on the radii recipe accepts the
        # factorization's (a level resolves its own, ISDF_GRID_N_START)
        other.n_start = self.n_start if n_start is None else n_start
        # a chain with no opinion on radii accepts the factorization's
        other.radii_tag = radii_tag(radii) if radii is not None else self.radii_tag
        if other.settings() != self.settings():
            raise ValueError(
                f'this chain asks for {other.settings()} but the shared '
                f'factorization was built for {self.settings()}; build the '
                f'factorization with the settings you want and pass it to '
                f'both chains, or pass none and let each build its own')


class OrbitalCapture(list):
    """The orbital branch's inputs (eps_bar, Y) each `FactorChain.nuclear_gradient`
    of a reverse pass appends, for a relaxed density. density_only: the pass
    stops there (the multiplier, collocation and fit branches are a force's)
    and returns no gradient."""

    def __init__(self, density_only=False):
        super().__init__()
        self.density_only = bool(density_only)


def require_default_fit(numerics):
    """Refuse fit choices (`IsdfNumerics` l_max_second, pair_tol,
    regularization) other than the defaults on a chain's factorization: the
    chains' fit adjoints (`dfactor_adjoint_gauges`, `row_fit_adjoint`) and the
    fit adjoint's pooling key (`FitKey`) differentiate the default fit, so a
    forward on another would carry a force of a different function."""
    asked = fit_settings(numerics, 'l_max_second', 'pair_tol',
                         'regularization')
    if asked != fit_settings(None, 'l_max_second', 'pair_tol',
                             'regularization'):
        raise NotImplementedError(
            f'fit choices (l_max_second, pair_tol, regularization) = {asked}: '
            "a chain's fit adjoints differentiate the default fit only, so "
            'its force would be of another function than its energy')


class FactorChain(ReferenceMeanField):
    """Reference-frozen radii, layout and frames; the factors at any geometry;
    adjoints on (eps, X_mo, D) carried to the nuclei.

    `scf_factory(mol) -> mf` supplies the mean field at a displaced geometry;
    it must converge the orbital gradient to SCF_DIFFERENTIABLE_GRAD_TOL,
    since the Lagrangian assumes the occupied-virtual Fock block vanishes. A
    mean field returned built but not run is converged by the chain over the
    ranks (`converged_factory`); a converged one is used as it is.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole:
    the frozen factorization is rank 0's, mean fields are locked to rank 0's
    orbitals, and the gradient is one `lockstep` at the end of
    `nuclear_gradient`.

    `numerics` (`Numerics`) holds every choice; the chain reads its `isdf`
    section -- basis, auxiliary basis, grid by level or counts, radii, fit --
    and `layout` -- sliced, fit_block -- here, and holds the object as
    `settings`. `layout.sliced` asks for a factorization whose factors are
    grid rows over the ranks (`FrozenFactorization`); None takes the given
    factorization's
    layout, or whole factors when the chain builds its own. A chain whose
    kernels do not all take `SlicedFactors` says so in
    `READS_SLICED_FACTORS` and refuses a sliced factorization at
    construction, before any SCF.

    `fit` and `fit_block` ask for the fit's realization the same way
    (`FrozenFactorization`); None takes the given factorization's, or the
    replicated fit. In an environment that dresses the interaction every
    gauge a chain reads -- the dressed D, the bare one (`bare_factor`), any
    further one -- is a metric root on the one fit, formed by rows in the
    factors' layout (`gauge_factor`), and the row fit's adjoint carries them
    all in one pass (`row_fit_branches`).

    The factors at every mean field are refused (`UnrepresentedOrbitals`)
    when they miss the density of any of its orbitals -- occupied and
    virtual, the pair space of every target here -- by more than the bar
    ISDF_ORBITAL_RESIDUAL_TOL, relative Coulomb residual against RI-V
    (`refuse_unrepresented_factors`, measured once per mean field in the
    gauge of D, `orbital_residuals_of`). The near-null AO combinations of an
    augmented basis carry such orbitals, and a BSE on them binds a spurious
    root no other check refuses. The bar is not a keyword.
    """

    #: Whether every kernel this chain calls reads `SlicedFactors`.
    READS_SLICED_FACTORS = False
    #: Whether the chain holds an explicit quasiparticle set that a refreeze
    #: can be handed a floor for (`refreeze`'s qp_floor).
    HOLDS_QP_SET = False

    def __init__(self, mol, scf_factory, numerics=None, mf=None,
                 environment=None, factorization=None):
        self.mol0, self.scf_factory = mol, scf_factory
        self.settings = numerics = numerics_or_default(numerics)
        # (weak reference to a mean field, relative residual of each orbital
        # density) measured: a memo that keeps no dead geometry alive
        self._orbital_residuals = []
        # wall seconds of each step of this constructor, for the record
        self.construction_seconds = {}
        t0 = time.perf_counter()
        # n_start None: the radii recipe's single start where the chain builds
        # its own factorization, which a table lookup ignores; a shared
        # factorization keeps its own. Both resolve a level the same way.
        if factorization is None:
            factorization = FrozenFactorization(mol, numerics)
        else:
            factorization.require_match(numerics)
        self.construction_seconds['factorization'] = time.perf_counter() - t0
        if factorization.sliced and not self.READS_SLICED_FACTORS:
            raise ValueError(
                f'{type(self).__name__} reads the factors whole in kernels '
                'that do not take SlicedFactors; build its factorization with '
                'sliced=False')
        self.factorization = factorization
        self.basis, self.auxbasis = factorization.basis, factorization.auxbasis
        self.counts, self.n_start = factorization.counts, factorization.n_start
        self.sliced = factorization.sliced
        self.fit, self.fit_block = factorization.fit, factorization.fit_block
        self.nocc = mol.nelectron // 2
        # the one given, else the one attached to the reference mean field,
        # else the gas phase; a given mean field is completed by it (charges)
        self.environment = resolve_environment(environment, mf)
        self._environment_cache = collections.OrderedDict()
        # a dict to accumulate phase timings into, or None for no timing
        self.timer = None
        self._init_reference(mol, scf_factory, mf)

        # the same objects, not copies, so two chains on one factorization
        # have bitwise identical factors
        self.radii = factorization.radii
        self.pts_local, self.owner = factorization.pts_local, factorization.owner
        self.frames = factorization.frames
        self.body = factorization.body
        self.M, self.naux = factorization.M, factorization.naux
        self.layout = factorization.layout

    def refreeze(self, mol, factorization=None, mf=None, walked=None,
                 qp_floor=None):
        """The same surface with every frozen convention rebuilt at `mol`:
        the one signature every factor chain refreezes through.

        factorization: the new one, shared with another chain (a composed
        surface's halves); else this chain's own rebuilt at `mol` by
        `FrozenFactorization.rebuilt_at`, the one place its settings travel,
        so no chain carries its own copy of the keyword list. mf: the new
        reference mean field at `mol`, converged by the caller; else the chain
        converges its own. walked: the mean field at `mol` the walk evaluated
        this chain on, which a chain whose conventions are keyed to orbitals
        carries them over on. qp_floor: states the new explicit set must
        hold; refused by a chain that holds no such set.
        """
        if qp_floor is not None and not self.HOLDS_QP_SET:
            raise ValueError(f'{type(self).__name__} holds no explicit '
                             'quasiparticle set: qp_floor has nothing to '
                             'carry')
        if factorization is None:
            factorization = self.factorization.rebuilt_at(mol)
        out = self._refrozen(mol, factorization, mf, walked, qp_floor)
        # only the geometry differs: a refrozen chain on other numerics is
        # another surface, whose energies do not continue this one's
        require_same_numerics(self.settings, out.settings, 'refreeze')
        return out

    def _refrozen(self, mol, factorization, mf, walked, qp_floor):
        """The chain's own rebuild at `mol` on `factorization` (`refreeze`)."""
        raise NotImplementedError(f'{type(self).__name__} cannot refreeze')

    def identity(self):
        """What a force of this chain is a force of
        (`surface_identity.chain_identity`); a chain with frozen
        conventions names them."""
        return chain_identity(self)

    def settings_record(self):
        """The chain's choices besides its factorization and conventions, for
        `surface_identity.force_identity`; a chain adds its own."""
        return {}

    def phase(self, key):
        """Time one stage into `self.timer[key]` (accumulating), or do nothing
        when `timer` is None."""
        return stage(self.timer, key)

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
        self.require_scf_quality(mf, 'mean_field_gradient')
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
        """Interpolation points r_g = p_g F_i + R_i on the frozen frames,
        turned with the molecule (`FrozenFactorization.coords`)."""
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

        On the row fit, and sliced over ranks, it is that gauge's rows in the
        factors' own layout (`gauge_factor`).
        """
        if not dresses_interaction(self.environment_at(mol), auxmol):
            return None
        if self.laid_out_in_rows():
            return self.gauge_factor(mol, auxmol, crd, None)
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

    def laid_out_in_rows(self):
        """Whether the factors are this rank's grid rows (`factors_at`): on
        the row fit, or sliced over more than one rank."""
        return (self.factorization.fit == 'rows'
                or self.factorization.slice_comm() is not None)

    def gauge_factor(self, mol, auxmol, crd, environment, mf=None):
        """D = M^T (V + vtilde)^1/2 dressed by `environment` (None: bare) in
        the factors' row layout: `SlicedFactors` sharing the X_mo and X_ao
        rows of the factors built at `mol` for `mf` (the last built there
        when None), or on the row fit over one rank the whole array.

        The same fit, another metric root. On the row fit the rows are formed
        from its M^T tiles, kept for this where the environment dresses the
        interaction (`separable_ri.RowFit.metric_root_rows`, the root on rank
        0 alone), so nothing whole exists; on the replicated fit the whole
        fit is re-run and this rank's rows cut from the whole product.
        """
        if not self.laid_out_in_rows():
            raise ValueError('the factors are whole; read a gauge whole '
                             'through shareable_factors')
        gauge = (environment if dresses_interaction(environment, auxmol)
                 else None)
        entry = self.factorization.rows_entry(mol, mf)
        if entry is None:
            raise ValueError('no factors are held at this geometry: build '
                             'them first (factors_at)')
        hit = entry.get(gauge)
        if hit is not None:
            return hit
        comm = self.factorization.slice_comm()
        base = entry.first()
        if entry.source is not None:
            d = entry.source.metric_root_rows(auxmol, gauge)
        else:
            _, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
            d = Mfit.T @ aux_metric_sqrt(auxmol, gauge, V=V)
            del Mfit
            if comm is not None:
                r0, r1 = base.rows
                d = d[r0:r1].copy()
        if comm is None:
            rows = d
        elif entry.source is not None:
            rows = SlicedFactors.from_rows(base.X_mo, d, base.X_ao, crd, comm,
                                           fit_held=entry.source.held)
        else:
            rows = SlicedFactors(base.X_mo, d, base.X_ao, crd, comm)
        entry.put(gauge, rows)
        return rows

    @staticmethod
    def zero_adjoint(factor, block=None):
        """A zero adjoint on the D of `factor` (an array or `SlicedFactors`):
        whole where `block` is None, else `GridTileRows` in tiles of `block`
        over the current ranks, so no rank forms the whole (npts, naux)."""
        if block is None:
            return np.zeros(np.shape(factor))
        npts, naux = ((factor.npts, factor.naux)
                      if isinstance(factor, SlicedFactors) else factor.shape)
        return GridTileRows.zeros(npts, naux, block, current_comm())

    def split_factor(self, mol, auxmol, crd):
        """(D_s, R_inf, R_s, T, static environment) of a split equilibrium
        continuum, or None for any other environment or where it collapses.

        D_s = M^T R_s is the factor dressed at eps_static on the same fit as
        the optical D = M^T R_inf, so D_s = D T with T = R_inf^-1 R_s
        (`split_equilibrium`). On the row fit, and sliced over ranks, D_s is
        that gauge's rows in the factors' layout (`gauge_factor`).
        """
        environment = self.environment_at(mol)
        if (getattr(environment, 'equilibrium', False) != 'split'
                or environment.collapses):
            return None
        static = environment.static_response()
        if self.laid_out_in_rows():
            V = auxmol.intor('int2c2e', aosym='s1')
            d_s = self.gauge_factor(mol, auxmol, crd, static)
        else:
            _, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
            d_s = Mfit.T @ aux_metric_sqrt(auxmol, static, V=V)
        r_inf = aux_metric_sqrt(auxmol, environment, V=V)
        r_s = aux_metric_sqrt(auxmol, static, V=V)
        return (d_s, r_inf, r_s, np.linalg.solve(r_inf, r_s), static)

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
                    self.orbital_residuals_of(mol, mf, auxmol, rows, None)
                else:
                    self.orbital_residuals_of(mol, mf, auxmol, rows[0],
                                              rows[1])
            if isinstance(rows, SlicedFactors):
                return rows, rows, eps, auxmol, crd, rows
            x_mo, d, x_ao = rows
            return x_mo, d, eps, auxmol, crd, x_ao
        with self.phase('t_factors'):
            x_ao, d = self.factors(mol, auxmol, crd)
            x_mo = x_ao @ mf.mo_coeff
            self.orbital_residuals_of(mol, mf, auxmol, x_mo, d)
        return (x_mo, d, eps, auxmol, crd, x_ao)

    def measured_orbital_residual(self, mf):
        """The largest relative residual of the factors on an orbital density
        of `mf`, as measured (`orbital_residuals_of`), or None."""
        for held, rel in self._orbital_residuals:
            if held() is mf:
                return float(np.max(rel))
        return None

    def orbital_residuals_of(self, mol, mf, auxmol, x_mo, d):
        """The relative residual of the factors on each orbital density of
        `mf` at `mol` (`refuse_unrepresented_factors`, which refuses one
        above the bar), measured once per mean field on the factors
        (x_mo, d) it is handed -- whole, or this rank's rows (`SlicedFactors`,
        d None) -- in the gauge of D.
        """
        for held, rel in self._orbital_residuals:
            if held() is mf:
                return rel
        rel, _ = refuse_unrepresented_factors(
            mol, auxmol, np.asarray(mf.mo_coeff, float), x_mo, d,
            root=aux_metric_sqrt(auxmol, self.environment_at(mol)),
            nocc=self.nocc, comm=current_comm())
        self._orbital_residuals = ([(weakref.ref(mf), rel)] + [
            (held, r) for held, r in self._orbital_residuals
            if held() is not None])[:EVALUATION_CACHE_SIZE + 1]
        return rel

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
        # where the interaction is dressed the fit's M^T tiles stay with the
        # rows: the other gauges are right products of them (`gauge_factor`)
        kept = [] if dresses_interaction(env, auxmol) else None
        if self.factorization.fit == 'rows' and kept is not None:
            rows = self.row_fit_factors(mol, mf, auxmol, crd, keep=kept)
        elif self.factorization.fit == 'rows':
            rows = self.row_fit_factors(mol, mf, auxmol, crd)
        else:
            x_ao, d = self.factors(mol, auxmol, crd)
            rows = SlicedFactors.from_whole((x_ao @ mf.mo_coeff, d, x_ao, crd),
                                            self.factorization.slice_comm())
        self.factorization.keep_rows(mol, mf, gauge, rows,
                                     source=kept[0] if kept else None)
        return rows

    def row_fit_factors(self, mol, mf, auxmol, crd, keep=None):
        """(X_mo, D, X_ao) of the row-distributed fit on the frozen points:
        `SlicedFactors` over more than one rank, whole arrays on one.
        keep: a list the fit's `RowFit` is appended to, its M^T tiles kept
        for further gauges (`gauge_factor`) and its collocation dropped.

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
        if keep is not None:
            fit.X = None
            keep.append(fit)
        if comm is None:
            return x_mo, d, x_ao
        return SlicedFactors.from_rows(x_mo, d, x_ao, crd, comm,
                                       fit_held=fit.held)

    def nuclear_gradient(self, mol, mf, auxmol, crd, x_mo, eps_bar, x_bar,
                         d_bar, y_extra=None, g_extra=None, d_bar_bare=None,
                         root_bar=None, kernel_bar=None, gauges=()):
        """(natm, 3) gradient from adjoints on (eps, X_mo, D), with diagnostics.

        X_mo = X_ao C depends on the geometry twice: through C, which enters the
        Lagrangian as the orbital-rotation gradient X_mo^T X_bar, and through the
        collocation X_ao, whose adjoint is X_bar C^T.

        eps_bar: the adjoint on the orbital energies, or a full symmetric MO
            Fock partial dE/dF_pq when the target's one-body term contracts F
            off the diagonal (an active-space model does).
        y_extra: orbital-rotation gradient of a term that is not a function of
            the factors -- the Kohn-Sham static correction, say. It joins Y
            BEFORE the multiplier solve, because it shares Lambda.
        g_extra: that term's own skeleton, added to the orbital branch.
        d_bar_bare: the adjoint on the BARE-gauge factor, when the target
            screens with both (`bare_factor`). It rides the same fit with no
            environment, so the cavity's motion reaches the force through
            the DRESSED factor only -- which is where vtilde actually sits.
        root_bar, kernel_bar: adjoints on the DRESSED gauge's metric root
            (V + vtilde)^(1/2) and on vtilde alone, from a term that reads the
            screened metric WITHOUT going through D. The fold's
            N = R^-1 vtilde R^-1 is that term. They join the dressed gauge so
            that one Frechet solve and one cavity derivative serve both it and
            D; differentiating them apart is the same number and twice the
            reaction-field work.
        gauges: further `GaugeAdjoint`s on the same fit -- the static
            partner's factor of an equilibrium-solvated ion (`static_factor`)
            or the static-dressed factor of a split equilibrium continuum
            (`split_factor`).

        x_mo may be `SlicedFactors`: the orbital-rotation gradient contracts
        the grid index, so X_mo is gathered whole for that product alone (on
        the row fit its tiles stream instead, `orbital_rotation_rows`), and
        the diagnostics carry `factor_gathers`, the gathers those factors have
        made, the same on every rank, and on the row fit `fit_held`, the most
        of each array of the fit rank 0 held at once, in bytes.

        The fit adjoint differentiates the one estimator -- the Gram matrix
        over every product pair, F over the frozen test set -- on both
        realizations: formed whole (`dfactor_adjoint_gauges` over
        `product_pairs`) on the replicated fit, and on the row fit with the
        collocation adjoint in the fit's tiles (`row_fit_branches`): no rank
        forms an array of the grid by a factor's or the fit's width beyond
        the X_bar and D_bar it is handed, and the diagnostics carry
        `adjoint_held`, the most of each of its arrays rank 0 held at once,
        in bytes.

        x_bar and d_bar may be `GridTileRows`, the adjoints in grid tiles
        over the ranks: the row fit reads its own tiles of them, and the
        replicated fit's whole-form adjoint gathers them whole first.

        ONE FIT ADJOINT. The assembly is a `one_fit_adjoint` window on `mf`
        (the caller's, where it opened one around the skeletons it hands in
        as g_extra): every row exchange skeleton inside it -- the relaxed
        density's here, the caller's Sigma_x - v_xc -- leaves its seeds to
        the row fit's adjoint the branches run, one `fit_rows_adjoints`
        call per fit, and their share joins the orbital branch they belong
        to.
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
            capture = getattr(self, '_orbital_capture', None)
            if capture is not None:
                # the orbital branch's inputs, for a relaxed density
                capture.append((np.array(eps_bar, float), np.array(y)))
                if getattr(capture, 'density_only', False):
                    # the multiplier, collocation and fit branches are the
                    # force's: a relaxed density alone reads nothing past here
                    pending.discard()
                    return None, {}
            with self.phase('t_orbital'):
                g_orb, diags = eps_chain_gradient(mf, eps_bar, self.nocc,
                                                  Y_extra=y, numerics=self.settings)
            if g_extra is not None:
                g_orb = g_orb + g_extra
            held = None
            if self.factorization.fit == 'rows':
                g_coll, g_fit, held = self.row_fit_branches(
                    mol, mf, auxmol, crd, x_bar, d_bar, d_bar_bare=d_bar_bare,
                    root_bar=root_bar, kernel_bar=kernel_bar, gauges=gauges)
            else:
                g_coll, g_fit = self.whole_fit_branches(
                    mol, mf, auxmol, crd, x_bar, d_bar, d_bar_bare=d_bar_bare,
                    root_bar=root_bar, kernel_bar=kernel_bar, gauges=gauges)
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
                           gauges=()):
        """(g_collocation, g_fit) on the replicated fit, both formed whole:
        `collocation_adjoint` of X_bar C^T and `dfactor_adjoint_gauges` over
        every product pair."""
        with self.phase('t_collocation'):
            g_coll = collocation_adjoint(mol, crd, x_bar @ mf.mo_coeff.T,
                                         self.pts_local, self.owner,
                                         frames=self.frames,
                                         with_frames=False, body=self.body)
        with self.phase('t_fit'):
            # Both gauges of one fit in one pass: the dressed factor the kernel
            # uses and the bare one the self-energy screens with share the test
            # set, the three-centre integrals and M, and differ only in the
            # metric root.
            every = [GaugeAdjoint(d_bar, self.environment_at(mol),
                                  root_bar=root_bar, kernel_bar=kernel_bar)]
            if d_bar_bare is not None:
                every.append(GaugeAdjoint(d_bar_bare, None))
            for g in gauges:
                whole = (g.d_bar.gather() if isinstance(g.d_bar, GridTileRows)
                         else g.d_bar)
                every.append(GaugeAdjoint(whole, g.environment,
                                          root_bar=g.root_bar,
                                          kernel_bar=g.kernel_bar))
            g_fit = dfactor_adjoint_gauges(mol, auxmol, crd, every,
                                           self.layout, self.pts_local,
                                           self.owner, frames=self.frames,
                                           with_frames=False,
                                           gram_layout=product_pairs(mol),
                                           body=self.body)
        return g_coll, g_fit

    def row_fit_branches(self, mol, mf, auxmol, crd, x_bar, d_bar,
                         d_bar_bare=None, root_bar=None, kernel_bar=None,
                         gauges=()):
        """(g_collocation, g_fit, held) on the row fit: the fit adjoint of
        `separable_ri.fit_rows` on the frozen pair layout and the X_mo
        collocation adjoint, computed in the fit's tiles
        (`isdf_derivatives.row_fit_adjoint`), so no rank forms the Gram
        matrix, the test set's collocation, (mu nu|P), M, F D^T or an
        adjoint of their size whole; the same bits at every rank count.
        held: the most bytes of each of its arrays this rank held at once.

        Every gauge of one fit rides the one call, as in
        `whole_fit_branches`: where the environment dresses the interaction
        d_bar is the dressed gauge's (with root_bar and kernel_bar on its
        root and vtilde), d_bar_bare the bare one's, and `gauges` further
        dressed factors (a split continuum's static one); in the gas phase
        d_bar is the bare gauge's and the call is the one-gauge call.

        Inside the force's `one_fit_adjoint` window the call carries the
        skeletons' seeds too (`PendingFitAdjoint.contract`), and `held` is
        that one call's ledger.
        """
        if any(g.environment is None for g in gauges):
            # scheme B's static half is the bare root; its root adjoint is
            # carried on the replicated fit (`dfactor_adjoint_gauges`) only
            raise NotImplementedError(
                'a bare gauge with a root adjoint (the static half of '
                'split_equilibrium.ClassicalStateSpecific) is differentiated '
                "on the replicated fit; build the factorization with "
                "fit='replicated' for a scheme B force or relaxed density")
        env = self.environment_at(mol)
        if dresses_interaction(env, auxmol):
            bare = d_bar_bare
            dressed = [GaugeAdjoint(d_bar, env, root_bar=root_bar,
                                    kernel_bar=kernel_bar), *gauges]
        else:
            if not (d_bar_bare is None and root_bar is None
                    and kernel_bar is None):
                raise ValueError(
                    f'{env!r} dresses nothing, so there is no bare gauge '
                    'beside D and no dressed root or vtilde for root_bar '
                    'and kernel_bar to land on')
            bare, dressed = d_bar, list(gauges)
        with self.phase('t_fit'):
            adjoint = row_fit_adjoint(mol, auxmol, crd, bare, x_bar,
                                      mf.mo_coeff, self.layout, self.pts_local,
                                      self.owner, frames=self.frames,
                                      with_frames=False,
                                      body=self.body,
                                      block=self.fit_block,
                                      pending=pending_fit_adjoint(mf),
                                      gauges=dressed)
        return adjoint

    def _assembled(self, g_orb, g_coll, g_fit, diags, x_mo,
                   adjoint_held=None):
        """(grad, diagnostics) of `nuclear_gradient` from its three
        branches, rank 0's on every rank."""
        grad = g_orb + g_coll + g_fit
        # an exact gradient sums to zero over the atoms: a free check on every branch
        diags = dict(diags, numerics_identity=self.settings.identity(),
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
