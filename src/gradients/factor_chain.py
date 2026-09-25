"""The frozen ISDF factorization a cubic gradient chain differentiates, and the
three branches that carry adjoints on its factors to the nuclei.

Every target on the space-time route -- a quasiparticle energy, a BSE
excitation, the dRPA correlation energy -- is a function of the orbital
energies eps, the collocation X_mo = X_ao C and the auxiliary factor D, and its
reverse pass ends with adjoints (eps_bar, X_bar, D_bar). From there on nothing
depends on the target:

    eps_bar, and X_bar through C          -> the orbital-response Lagrangian
    X_bar through the interpolation points -> the collocation adjoint
    D_bar                                  -> the fit adjoint

Neither does what has to be decided ONCE, at the reference geometry, for the
surface to be smooth at all: the grid radii, the pair layout and the frames.
Re-deciding any of them per geometry puts a step into the energy that no
gradient can follow.

The surroundings enter through ONE object, the chain's `environment`
(src.Base.environment): it builds the mean field at each geometry, dresses the
auxiliary gauge, supplies the static one-body term production adds, and owns
the adjoints of both. A chain therefore cannot see half of a reaction field.

A chain carries no communicator. Inside `with distributed(comm):` every rank
runs it whole, the kernels it calls divide their sweeps over the ranks, and
the few arrays the chain decides itself -- the grid, the placed points, the
pair layout, the fit, the assembled nuclear gradient and the mean field's
own force (`mean_field_gradient`) -- are one `lockstep` each, so every rank
holds rank 0's bits of them.

SLICED FACTORS. A factorization built with `sliced=True` holds, over more
than one rank, each rank's `contiguous_block` of the grid rows of X_mo, D and
X_ao (`SlicedFactors`), cut from the whole products after they are formed
exactly as the whole layout forms them, and it keeps no whole fit between
geometries. Every kernel that reads a factor whole gathers it once per sweep
or solve and drops it on return; the Davidson block action reads the rows.
The gathered arrays are the whole ones verbatim, so the LAYOUT adds no
difference: on one mean field, and on a pyscf that repeats its bits, every
output is the whole layout's bit for bit. pyscf's own threaded work inside a
chain -- its K builds and the mean field's force -- does not repeat its bits
from run to run (its OpenMP GEMM adds the partial sums in thread-arrival
order), so two evaluations, of either layout, agree to that re-association
and are compared on an anchored bar.

Per rank, factor arrays only, at the chlorophyllide hexamer/cc-pVTZ over 8
ranks (M 117762, nmo = nao 10980, nocc 972, naux
28236; 14721 rows): X_mo and X_ao 10.34 GB whole and 1.29 GB as rows, D 26.60
and 3.33, X_o 0.92, X_v 9.43, the fit's (P|Q) 6.38; whole -> sliced:

  stage                   read whole              GB whole -> sliced
  between steps           -                       43.3 -> 5.9 per geometry
  factor build            X_ao, M, (P|Q), X_mo, D 80.3 -> 86.2, then 5.9
  factor build, row fit   one shell's (mu nu|P)   86.2 -> ~51 peak, then 5.9
  static W, chi0 sweep    X_o, X_v, D             90.6 -> 42.9
  QP self-energy solve    X_mo, D (+ branches)    90.6 -> 53.2
  BSE Davidson setup      D (Zt rows), X_o        90.6 -> 34.6
  BSE Davidson, per trial X_o                     90.6 -> 8.0
  BSE cache               X_mo, D                 80.3 -> 42.9
  BSE adjoint             X_mo, D + adjoints      117.2 -> 79.8
  chi0/W adjoint sweep    X_o, X_v, D + adjoints  127.6 -> 79.8
  dRPA energy + adjoint   X_mo, D (+ branches)    127.6 -> 90.1
  nuclear assembly        X_mo + adjoints         117.2 -> 53.2
  assembly on the row fit X_bar, D_bar            9.4e4 -> 72.4 (60.0 at 16)

Whole, the fit (X_ao, M, (P|Q)) stays cached for every live geometry and the
chain holds X_mo and D for the whole gradient. The adjoints X_bar and D_bar
(36.9 GB) are whole in both layouts: they are all-reduced sums over the tau
partition, and the orbital response and the fit adjoint contract them over
the grid. Larger than any factor, and untouched by the layout: the fit's own
nk^2 Gram matrix (111 GB), proj(tau) of the quasiparticle solve (ntau naux^2,
6.4 GB per tau point) and the BSE adjoint's (naux, nocc, nvir) blocks (2.2
TB), which is what bounds the excited-state force at this size.

THE ROW FIT. `fit='rows'` (with `sliced=True`) builds the rows with
`separable_ri.fit_rows` on the frozen points instead of cutting them from a
whole fit: the Gram matrix, F D^T, the solve and the collocation exist only as
each rank's tiles, and nothing whole is formed or cached at any geometry. At
the hexamer over 8 ranks the build then peaks at about 51 GB per rank in the
three-centre pass, every array of the fit counted (the 86.2 above counts the
factor arrays alone; the replicated fit's own pass peaks near 300): one f
shell's (mu nu|P) over every nu and P beside the metric's LU sets it, the
fit's blocking decides that and no rank count lowers it, against 21 GB for the
Gram tiles and 13 for the Cholesky (the table in `separable_ri`), and the same
5.9 GB of rows remain. The rows are bitwise the same at every rank count, one
rank included, and they realize the row fit's own estimator
(`FrozenFactorization`), which the fit adjoint then differentiates.

The row fit's nuclear assembly runs in the same tiles (`row_fit_branches`,
`separable_ri.fit_rows_adjoint`, whose table has the arrays): the fit adjoint,
the X_mo collocation adjoint and X_mo^T X_bar, the same bits at every rank
count. Where the whole adjoint formed on every rank the Gram matrix (111 GB),
the test set's collocation over every product pair (nk x 1.0e8 doubles) and
the dense (mu nu|P), it peaks per rank at 29.6 GB over 8 ranks and 20.1 over
16, in its three-centre pass: the Gram tiles' kept factor (7.2 / 3.7), the
metric's LU (6.4, replicated as in the forward pass), three (rows, naux)
arrays (3 x 3.4 / 3 x 1.7) and one block's coefficients broadcast (3.4, the
fit's blocking); beside it the X_bar and D_bar it is handed (36.9) and the
factor rows (5.9 / 3.0) make the row above, the whole adjoint's 9.4e4 GB
being its test-set collocation. Rank 0 alone holds four metric-sized
arrays for the root's adjoint (25.5, at any rank count), and reads D_bar
whole for M D_bar. What stays whole, and why: X_bar and D_bar, the kernels'
all-reduced sums over the tau partition; and the pair layout, screened once
per reference on the whole AO collocation (10.3 GB, when the factorization
is built, never at a force).
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
                                ISDF_FIT_ERROR_FAILED, SCF_GRAD_TOL,
                                THREE_CENTER_BLOCK_BYTES)
from src.Base.distributed_df import distributed_mean_field
from src.Base.environment import dresses_interaction, resolve_environment
from src.Base.separable_ri import (atomic_frames, aux_metric_sqrt, fit_M_stable,
                                   fit_M_streaming, optimize_atomic_radii,
                                   resolve_isdf_grid, subshells, test_set_D,
                                   test_set_layout)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import current_comm, lockstep, lockstep_mean_field
from src.SingleReference.GW.space_time import DEFAULT_COUNTS
from src.SingleReference.LinearResponse.rpa_energy import reference_energy
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            continued_frames,
                                            dfactor_adjoint_gauges,
                                            GaugeAdjoint,
                                            shell_blocks,
                                            eps_chain_gradient,
                                            exx_double_counting_Y,
                                            exx_double_counting_skeleton,
                                            orbital_rotation_rows,
                                            point_layout, row_fit_adjoint)

#: How the fit is realized (`FrozenFactorization`): 'replicated', the whole
#: fit on every rank, or 'rows', `separable_ri.fit_rows` by grid rows.
FIT_REALIZATIONS = ('replicated', 'rows')


def converged_factory(scf_factory):
    """`scf_factory` completed by the chain when it returns a mean field that
    has been built but not run, so a factory may hand back the cheaper of the
    two and let the chain finish the SCF.

    The completion is `distributed_mean_field`, which reads the current
    `distributed` context: inside one, every rank runs pyscf's own SCF driver
    against the reduced J/K and the reduced quadrature, each contributing its
    block of the auxiliary index and of the exchange-correlation grid, and
    rank 0's spectrum and orbitals end on all of them. That is the one stage
    of a displaced geometry that does not otherwise divide -- every rank
    converges the same SCF, so its wall is what it was at one rank however
    many there are, 128 s of a rank's 227 s at pentacene/cc-pVTZ on four.
    Outside a region it is `mf.kernel()`, the call the factory would have made
    itself, so the serial arithmetic is bit for bit what it was. Inside one the
    mean field it builds must be density-fitted with pyscf's own DF, since the
    split is over the rows of `cderi`; `distributed_mean_field` refuses any
    other by name.
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
    different functionals and must not be allowed to share."""
    if radii is None:
        return None
    return tuple((el, tuple(np.round(np.concatenate(
        [np.atleast_1d(radii[el][s]) for s in sorted(radii[el])]), 12)))
        for el in sorted(radii))


class FrozenFactorization:
    """The ISDF conventions fixed at ONE reference geometry, so that several
    chains are ONE functional rather than two that happen to agree.

    Radii, interpolation points, frames and pair layout are discrete choices.
    Re-deciding them per geometry puts a step in the energy; deciding them
    twice, once per chain, is subtler -- `separable_factors` is deterministic
    given the settings, so two chains handed matching settings produce bitwise
    identical factors and merely pay twice. They diverge silently the moment
    the settings stop matching: different `counts` or `n_start`, or
    `frames='continued'`, which carries orientation history from wherever each
    chain started.

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

    `grid_accuracy` is the named way to fix `counts` and `n_start` together --
    a validated accuracy level or four explicit shell counts, resolved against
    the shipped radii table by `resolve_isdf_grid`, which refuses a grid nobody
    optimized rather than re-optimizing one per geometry.

    UNDER RANKS THE CONVENTIONS ARE RANK 0'S, which is the same requirement
    one step further out. Every rank decides them, as serial code, and one
    `lockstep` then leaves rank 0's radii, fit errors, clouds, owners and
    frames on all of them; the placed points, the pair layout and the fit are
    locksteps of their own, taken where they are built. Deciding them per rank
    is invisible on one machine -- two processes there run the same arithmetic
    on the same libraries and agree bit for bit -- and across NODES it puts
    the ranks on different surfaces: the route gates passed on every rank of
    two and four nodes while the end-to-end forces missed by 1.5e-3 and
    5.7e-3 Ha/Bohr on the ranks that had frozen their own. Serially every
    lockstep is a no-op and the arithmetic is what it was.

    `sliced` is the LAYOUT of the factors, not a choice of functional, and is
    not among `settings`: over more than one rank each rank keeps its grid
    rows of X_mo, D and X_ao (`FactorChain.factors_at`) and no whole fit
    outlives the cut. Serially, and on one rank, it changes nothing.

    `fit` is the REALIZATION of the fit, and like `sliced` it is matched
    apart from `settings`. 'replicated' is `_fit`: every rank forms the
    frozen test set, the Gram matrix over its pairs and the Cholesky solve
    whole. 'rows' (with `sliced=True`) is `separable_ri.fit_rows` on the
    frozen points, `fit_block` points per tile: no rank holds the Gram
    matrix, F D^T, the solve, the collocation or a factor whole, the rows
    are bitwise the same at every rank count, one rank included, and they
    are not `_fit`'s bits. Nor is it `_fit`'s estimator wherever the pair
    screen drops a pair: its Gram matrix is the unscreened product
    (A A^T) o (B B^T), F D^T the screened pairs. On ethylene/cc-pVDZ (72 of
    2304 pairs dropped) that is 2.0e-4 of D, which moves the composed
    singlet force 6.3e-7 Ha/Bohr and its root 2.6e-6 eV, 55 to 6e5 times
    what one reassociation of the row fit's estimator moves them;
    water/cc-pVDZ, which keeps every pair, cannot tell the two apart.
    `FactorChain.nuclear_gradient` differentiates that estimator on the row
    fit. The row fit screens its F columns at each geometry where `_fit`
    freezes the reference's (`layout`): the two sets agree at the reference
    by construction, and a pair crossing the tolerance on the way steps the
    row fit's surface by that pair's share of the fit.
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
        self.basis = basis or str(mol.basis)
        self.auxbasis = auxbasis or (self.basis + '-ri')
        elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
        # The named way in. It is resolved HERE and stored as plain counts, so
        # the whole walk -- `rebuilt_at`, `settings`, `require_match` -- keeps
        # comparing one kind of object and a level cannot mean two grids.
        if grid_accuracy is not None:
            counts, n_start = resolve_isdf_grid(grid_accuracy, self.basis,
                                                elements, auxbasis=self.auxbasis)
        self.counts = counts or DEFAULT_COUNTS
        self.n_start = n_start
        self.frames_mode = frames
        self.with_frames = frames == 'continued'
        # Explicit radii -- `separable_ri.tailor_grid`'s, say -- replace the
        # per-element atomic optimization. They are frozen here like every other
        # discrete choice, so a grid tailored at the reference geometry is the
        # one the whole walk uses.
        self.radii_tag = radii_tag(radii)
        if radii is not None and not set(elements) <= set(radii):
            raise ValueError(f'radii given for {sorted(radii)} but the molecule '
                             f'holds {elements}')
        # The fit error is kept, not discarded: it is the only thing that says
        # whether this grid resolves this basis, and the default count is sized
        # for double zeta. A caller who does not pass `counts` gets a FAILED fit
        # for most elements at any larger basis, and nothing else would say so.
        fit_errors = {}
        if radii is None:
            radii = {}
            for el in elements:
                radii[el], fit_errors[el] = optimize_atomic_radii(
                    el, self.basis, self.auxbasis, counts=self.counts,
                    n_start=n_start)
        # The radii come out of a local descent on a multi-modal objective
        # whose minimum moves with the BLAS reduction order, the frames out of
        # an `eigh`: rank 0's, on every rank, before anything is placed. The
        # clouds and owners follow from the radii and travel in the same call.
        (self.radii, self.fit_errors, self.pts_local, self.owner,
         self.frames) = lockstep((radii, fit_errors,
                                  *point_layout(mol, radii),
                                  atomic_frames(mol)[0]))
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
        # The layout is a screening threshold on collocated products, taken on
        # rank 0's placed points, and locked as well: its column SET must be
        # one set, and a rank whose threshold kept another raises on every
        # rank (the shapes disagree) rather than differentiating its own.
        self.layout = lockstep(test_set_layout(mol, self.coords(mol)))
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

        Everything this object OWNS is a choice and travels verbatim, radii
        included: an explicit set IS the choice and has nothing to re-derive,
        where atomic radii are element-only and rebuild identically. Only what
        depends on the geometry -- the placed points, the frames, the pair
        layout -- is derived again. Callers rebuild through here so that a
        setting added above cannot be silently dropped by one of them.
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

        Rank 0's points on every rank of a distributed run: the grid IS the
        functional, and ranks whose points differ even in the last bits carry
        forward quantities and adjoints of two different surfaces. Continued
        frames are re-derived at every geometry through an `eigh`, which is
        where a node's own bits enter.
        """
        fr = continued_frames(mol, self.frames) if self.with_frames \
            else self.frames
        return lockstep(np.vstack([self.pts_local[ia] @ fr[ia]
                                   + mol.atom_coord(ia)
                                   for ia in range(mol.natm)]))

    def shareable_factors(self, mol, auxmol, crd):
        """(X_ao, Mfit, V): the fit at `mol` before the auxiliary gauge.

        The split is drawn before the gauge because `aux_metric_sqrt` dresses
        it with the chain's own environment, so two chains sharing a layout and
        carrying different continua must not share a D. Everything above it is
        environment-independent and cached here, which is most of the
        per-geometry cost, so several chains of one composed surface pay it once.

        Under ranks the fit is rank 0's on every rank, and the lockstep that
        makes it so is taken on EVERY call, hit or miss. Every rank must reach
        the same collective at the same call, and a weak-keyed cache does not
        expire in step across ranks: a rank whose Mole was collected would
        recompute -- and enter the lockstep -- while the others returned from
        their cache.

        Sliced over ranks nothing is cached here: the fit is whole only while
        the rows are cut from its products, and the rows are what the chains
        share (`cached_rows`). The row fit has no whole fit to share and
        refuses: `_fit` is another estimator than the one it realizes.
        """
        if self.fit == 'rows':
            raise ValueError(
                "fit='rows' never forms the whole fit, and `_fit` is not its "
                "estimator; read the factors through FactorChain.factors_at")
        if self.slice_comm() is not None:
            return lockstep(self._fit(mol, auxmol, crd))
        hit = self._fit_cache.get(mol)
        if hit is None:
            hit = self._fit(mol, auxmol, crd)
            self._fit_cache[mol] = hit
        return lockstep(hit)

    def _fit(self, mol, auxmol, crd):
        """The least-squares fit itself: (X_ao, M, V) at `mol` on the frozen layout."""
        mu_i, nu_i, wc_l = self.layout
        naux = auxmol.nao_nr()
        Dt = test_set_D(mol, auxmol, crd, self.layout)
        V = auxmol.intor('int2c2e', aosym='s1')
        e3 = test_set_three_center(mol, auxmol, mu_i, nu_i)
        F = np.hstack([np.linalg.solve(V, e3.T) * wc_l[None, :],
                       np.eye(naux)])
        return (mol.eval_gto('GTOval_sph', crd), fit_M_stable(Dt, F), V)

    def settings(self):
        """What two chains must agree on to be allowed to share one of these."""
        return (self.basis, self.auxbasis, tuple(sorted(self.counts.items()))
                if isinstance(self.counts, dict) else tuple(np.ravel(self.counts)),
                self.n_start, self.frames_mode, self.radii_tag)

    def require_match(self, basis, auxbasis, counts, n_start, frames,
                      radii=None, sliced=None, fit=None, fit_block=None):
        """Refuse a chain whose own settings contradict this factorization.

        Silently resolving to one side is the failure the object exists to
        prevent: the caller asked for a factorization it is not getting, and
        every number downstream would be of a functional nobody requested.
        A layout asked for (`sliced` not None) must be this one's too: the
        numbers agree, the memory a rank holds does not. So must a fit
        realization (`fit`, `fit_block` not None): two realizations agree
        to rounding and no closer.
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
        other.auxbasis = auxbasis or (other.basis + '-ri')
        other.counts = counts or DEFAULT_COUNTS
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
    assumes the occupied-virtual Fock block vanishes. A factory that returns a
    mean field it has BUILT AND NOT RUN hands the convergence to the chain,
    which divides its Fock build over the ranks (`converged_factory`); one that
    converges its own is used as it is.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole.
    Its frozen factorization is rank 0's on every rank, its mean fields are
    locked to rank 0's orbitals, and its share of the gradient -- the orbital
    response, the collocation and the fit branch it forms itself -- is one
    `lockstep` at the end of `nuclear_gradient`. Without these each rank would
    decide its own grid and add its own node's last bits, which agrees between
    two processes of one machine and not between two nodes.

    `sliced` asks for a factorization whose factors are grid rows over the
    ranks (`FrozenFactorization`); None takes the given factorization's
    layout, or whole factors when the chain builds its own. A chain whose
    kernels do not all take `SlicedFactors` says so in
    `READS_SLICED_FACTORS` and refuses a sliced factorization at
    construction, before any SCF, as does one that reads the bare gauge
    (`READS_BARE_GAUGE`) in an environment that dresses the interaction.

    `fit` and `fit_block` ask for the fit's realization the same way
    (`FrozenFactorization`); None takes the given factorization's, or the
    replicated fit. The row fit ('rows') is refused in an environment that
    dresses the interaction, for every chain: its D is one gauge's rows,
    where Eq. (18) reads the bare gauge beside the dressed one, and the
    gauge adjoint of its estimator is gated in the gas phase alone.
    """

    #: Whether every kernel this chain calls reads `SlicedFactors`.
    READS_SLICED_FACTORS = False
    #: Whether this chain screens with the BARE gauge beside the dressed one
    #: where the environment dresses the interaction (`bare_factor`).
    READS_BARE_GAUGE = False

    def __init__(self, mol, scf_factory, basis=None, auxbasis=None, counts=None,
                 n_start=1, frames='frozen', mf=None, environment=None,
                 factorization=None, radii=None, grid_accuracy=None,
                 sliced=None, fit=None, fit_block=None):
        self.mol0, self.scf_factory = mol, scf_factory
        # Resolved before the branch, so that building a factorization and
        # matching a shared one are handed the same counts and recipe.
        if grid_accuracy is not None:
            basis = basis or str(mol.basis)
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
                "gauge adjoint of the row fit's estimator is gated in the "
                "gas phase alone; build the factorization with "
                "fit='replicated' for a solvated run")
        with self.phase('t_scf'):
            self.mf0 = self.environment.mean_field(
                mol, converged_factory(
                    scf_factory if mf is None else (lambda _mol: mf)))
        # One reference on every rank from the start: what the chain reads off
        # mf0 directly must be rank 0's, and a given `mf=` may have been
        # converged on each rank alone.
        lockstep_mean_field(self.mf0)
        self.scf_residual = check_scf_quality(self.mf0, self.nocc)

        # the SAME objects, not copies: two chains on one factorization share
        # the arrays, which is what makes their factors bitwise identical
        self.radii = factorization.radii
        self.pts_local, self.owner = factorization.pts_local, factorization.owner
        self.frames = factorization.frames
        self.M, self.naux = factorization.M, factorization.naux
        self.layout = factorization.layout

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the reference pair, or a fresh SCF at another geometry,
        built in the chain's environment.

        The fresh one is converged over the ranks wherever the factory leaves
        it to them (`converged_factory`): a displaced geometry is a whole SCF,
        and it is the one stage of a walk that every rank would otherwise
        compute alone.
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
        # contracts the kernels' adjoints with THIS mean field's coefficients,
        # and another node's SCF differs from rank 0's by the phases and
        # degenerate rotations of its orbitals. A mean field the ranks
        # converged together arrives locked and this rewrites the same bits;
        # one each rank converged alone does not.
        lockstep_mean_field(mf)
        return mol, mf

    @contextmanager
    def phase(self, key):
        """Time one stage into `self.timer[key]` (accumulating), or do nothing.

        The forward and reverse halves of a gradient are only comparable when
        timed apart, so every stage of a chain runs inside one of these; a
        runner that wants the split sets `timer` to a dict, everyone else pays
        one attribute read.
        """
        if self.timer is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timer[key] = self.timer.get(key, 0.0) + time.perf_counter() - t0

    def require_differentiable_environment(self):
        """Refuse a force this environment cannot complete, before paying for it.

        The refusal itself comes from the environment's own adjoints, which
        raise; asking here means a production-sized reverse pass is not run
        first only to raise at its last branch.
        """
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
        """E_0^HF of the plasmon formula: production's E_HF AT this density.

        `mf.e_tot` on a Hartree-Fock mean field, and on a Kohn-Sham one the
        Hartree-Fock energy of ITS density, which is what the plasmon formula
        wants; the physics is in the production routine. TAKE IT WITH ITS
        GRADIENT: `kohn_sham_gradient_correction` below carries the two terms
        that move the force the same way, and a chain that adopts one and not
        the other is stationary for neither functional.
        """
        return reference_energy(mf, mol)

    def kohn_sham_gradient_correction(self, mol, mf):
        """(y_extra, g_extra): what `reference_energy` costs on the gradient side.

        `reference_energy` moves the surface from E_KS to E_HF at the same
        density. These are the two terms that move its DERIVATIVE the same way
        -- the EXX double-counting orbital response, which enters the Lagrangian
        before the multiplier solve because it shares Lambda, and its skeleton,
        which enters the orbital branch. They are the `y_extra` and `g_extra`
        hooks of `nuclear_gradient`, and those hooks are SHARED: sum into
        whatever a chain already passes rather than replacing it.

        Both are identically zero on a Hartree-Fock mean field, so callers add
        them unconditionally and that path stays bitwise unchanged.

        USE `mean_field_gradient` FOR THE OTHER HALF. `g_extra` is differenced
        against pyscf's own mean-field force, so that force must carry the grid
        response or the cancellation tears.

        A chain with no EXX skeleton of its own does not need these at all: it
        should REFUSE a Kohn-Sham reference outright, which makes the
        inconsistent state unreachable instead of merely documented.
        """
        return (exx_double_counting_Y(mf, self.nocc),
                exx_double_counting_skeleton(mf, mol))

    def mean_field_gradient(self, mf):
        """pyscf's own force for THIS mean field, with the grid response paired.

        The half of `kohn_sham_gradient_correction` that lives outside the
        Lagrangian. `grid_response=True` on a KS gradient is not an accuracy
        knob here: it is the partner of the double-counting skeleton, and
        setting it on one and not the other tears a cancellation the two are
        supposed to complete.

        Rank 0's force on every rank. Every rank computes it, and pyscf
        blocks a density-fitted gradient's auxiliary index by the free
        memory of the process it runs in (`max_memory` less the resident
        size), so ranks whose resident sizes differ re-associate the same
        sums differently and the force a chain returns would differ in its
        last bits from rank to rank.
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
        """The environment around THIS geometry, built once per geometry.

        A cavity moves with the atoms, so each displaced geometry gets its own,
        and a finite difference of the energy is the derivative of the real
        surface only because the two displaced points do not share one.

        THE KEY IS THE EXACT BYTES of the atomic charges and coordinates, and
        the nuclear count. Charges belong in it because a cavity follows the
        element radii, so coordinates alone do not determine one. The bytes are
        unrounded because rounding is what lets two finite-difference
        displacements collide and share a reaction field, which is a wrong force
        that no energy or gradient gate would report.

        Content rather than `id(mol)` because identity is safe only by
        accident of what the environments happen to return.
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

        The least-squares fit target keeps the BARE metric; only the gauge
        V^(1/2) is dressed by the environment at this geometry, as in
        `space_time.separable_factors`.
        """
        X_ao, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return (X_ao,
                Mfit.T @ aux_metric_sqrt(auxmol, self.environment_at(mol), V=V))

    def bare_factor(self, mol, auxmol, crd):
        """D in the BARE gauge, or None when nothing screens.

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

    def factors_at(self, mol, mf):
        """(X_mo, D, eps, auxmol, coords, X_ao) at `mol`.

        The factorization every chain starts from, under the one timing phase,
        so the reference-frozen radii, layout and frames are consulted in
        exactly one place rather than once per target.

        Sliced over ranks, X_mo, D and X_ao are ONE `SlicedFactors` holding
        this rank's rows of each (`sliced_factors_at`), and every kernel the
        chain hands them to gathers what it reads whole. On the row fit they
        are the rows it built, and on one rank its whole arrays.
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

        On the replicated fit the rows are CUT FROM THE WHOLE PRODUCTS, never
        formed as row products: X_mo = X_ao C and D = M^T V^(1/2) are built
        exactly as the whole layout builds them, and only then is this rank's
        `contiguous_block` kept, since a row block of a GEMM is not the same
        bits as those rows of the whole GEMM. The whole fit and products are
        dropped when this returns. On the row fit they are what it built
        (`row_fit_factors`), and nothing whole exists at any stage.

        The rows are shared through the factorization with every chain that
        asks at the same geometry, for the same mean field and in the same
        gauge -- the two halves of a composed surface pay one fit. The rows are
        cut for the ranks of the current region (`slice_comm`).
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

        `separable_ri.fit_rows` solves the balanced, regularized estimator
        with the grid index in fixed tiles of `fit_block` points, so each
        rank holds only the tiles it owns and its rows are bitwise the
        one-rank rows at any rank count. X_mo, D = M^T V^(1/2) and X_ao are
        read off it on this rank's rows, in the fit's own tiles; the metric
        root is rank 0's, because every tile owner projects with it.

        Not `separable_factors(fit='rows')`: that call places its own points
        on frames recomputed at each geometry, and the chain differentiates
        the frozen ones.
        """
        comm = self.factorization.slice_comm()
        fit = fit_M_streaming(mol, auxmol, crd, fit='rows',
                              block=self.factorization.fit_block)
        x_mo = fit.mo_rows(mf.mo_coeff)
        d = fit.metric_root_rows(auxmol, self.environment_at(mol))
        x_ao = fit.ao_rows()
        if comm is None:
            return x_mo, d, x_ao
        return SlicedFactors.from_rows(x_mo, d, x_ao, crd, comm,
                                       fit_held=fit.held)

    def nuclear_gradient(self, mol, mf, auxmol, crd, x_mo, eps_bar, x_bar,
                         d_bar, y_extra=None, g_extra=None, d_bar_bare=None,
                         root_bar=None, kernel_bar=None):
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

        x_mo may be `SlicedFactors`: the orbital-rotation gradient contracts
        the grid index, so X_mo is gathered whole for that product alone (on
        the row fit its tiles stream instead, `orbital_rotation_rows`), and
        the diagnostics carry `factor_gathers`, the gathers those factors have
        made, the same on every rank, and on the row fit `fit_held`, the most
        of each array of the fit rank 0 held at once, in bytes.

        On the row fit the fit adjoint differentiates the row fit's own
        estimator -- the Gram matrix over every product pair, F over the
        frozen test set -- and runs with the collocation adjoint in the
        fit's tiles (`row_fit_branches`): no rank forms an array of the grid
        by a factor's or the fit's width beyond the X_bar and D_bar it is
        handed, and the diagnostics carry `adjoint_held`, the most of each
        of its arrays rank 0 held at once, in bytes.
        """
        if self.factorization.fit == 'rows':
            with self.phase('t_orbital'):
                y = orbital_rotation_rows(x_mo, x_bar, self.fit_block)
        else:
            y = (x_mo.require(current_comm()).gather('X_mo')
                 if isinstance(x_mo, SlicedFactors) else x_mo).T @ x_bar
        if y_extra is not None:
            y = y + y_extra
        with self.phase('t_orbital'):
            g_orb, diags = eps_chain_gradient(mf, eps_bar, self.nocc, Y_extra=y)
        if g_extra is not None:
            g_orb = g_orb + g_extra
        if self.factorization.fit == 'rows':
            g_coll, g_fit, held = self.row_fit_branches(
                mol, mf, auxmol, crd, x_bar, d_bar, d_bar_bare=d_bar_bare,
                root_bar=root_bar, kernel_bar=kernel_bar)
            return self._assembled(g_orb, g_coll, g_fit, diags, x_mo,
                                   adjoint_held=held)
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
            g_fit = dfactor_adjoint_gauges(mol, auxmol, crd, gauges,
                                           self.layout, self.pts_local,
                                           self.owner, frames=self.frames,
                                           with_frames=self.with_frames)
        return self._assembled(g_orb, g_coll, g_fit, diags, x_mo)

    def row_fit_branches(self, mol, mf, auxmol, crd, x_bar, d_bar,
                         d_bar_bare=None, root_bar=None, kernel_bar=None):
        """(g_collocation, g_fit, held) on the row fit: the adjoint of
        `separable_ri.fit_rows`' own estimator on the frozen pair layout and
        the X_mo collocation adjoint, computed in the fit's tiles
        (`isdf_derivatives.row_fit_adjoint`), so no rank forms the Gram
        matrix, the test set's collocation, (mu nu|P), M, F D^T or an
        adjoint of their size whole; the same bits at every rank count.
        held: the most bytes of each of its arrays this rank held at once.

        One gauge, the bare metric: the chain refuses the row fit where the
        environment dresses the interaction, so a second gauge or an
        adjoint on a dressed root has nowhere to land.
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
                                      block=self.fit_block)
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
        # rank forms from them carry its node's last bits, so without this the
        # ranks would disagree by what they added alone.
        return lockstep((grad, diags))


def test_set_three_center(mol, auxmol, mu_i, nu_i,
                          max_bytes=THREE_CENTER_BLOCK_BYTES):
    """(mu_i nu_i|P) on the test-set pairs alone, blocked over the mu shells.

    The fit reads one row per test-set pair, of order a percent of the dense
    (nao, nao, naux) three-centre tensor. Building that tensor in order to index
    it costs nao^2 naux doubles, so it, not the flop count, is what ends the
    chain at large size. Blocking the first index holds one (block, nao, naux)
    slab at a time and returns the same rows to the bit.
    """
    nao, naux = mol.nao, auxmol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    mu_i, nu_i = np.asarray(mu_i), np.asarray(nu_i)
    out = np.empty((mu_i.size, naux))
    for sh0, sh1 in shell_blocks(mol, int(nao) * int(naux) * 8, max_bytes):
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        take = np.flatnonzero((mu_i >= a0) & (mu_i < a1))
        if not take.size:
            continue
        blk = pyscf_df.incore.aux_e2(
            mol, auxmol, intor='int3c2e', aosym='s1',
            shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas)
        ).reshape(a1 - a0, nao, naux)
        out[take] = blk[mu_i[take] - a0, nu_i[take], :]
        del blk
    return out


def check_scf_quality(mf, nocc, tol=SCF_GRAD_TOL, raise_on_fail=False):
    """max |F_ia| in the MO basis -- the quantity `conv_tol_grad` controls.

    The Lagrangian assumes this block vanishes, what is left is amplified in
    the force, and symmetry hides it: a symmetric molecule looks converged and
    is wrong in the fourth digit.
    """
    c = mf.mo_coeff
    resid = float(np.abs((c.T @ mf.get_fock() @ c)[:nocc, nocc:]).max())
    if resid > tol:
        msg = (f'the mean field carries |F_ia| = {resid:.2e}, above {tol:.0e}; '
               'the gradient Lagrangian assumes it vanishes. Set '
               'conv_tol_grad=1e-11.')
        if raise_on_fail:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
    return resid
