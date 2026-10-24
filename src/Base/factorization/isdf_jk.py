"""ISDF (separable-RI) J and K for the SCF, in place of a density-fitted cderi.

pyscf's DF builds `cderi`, naux x nao(nao+1)/2, and streams it once per SCF
iteration; a range-separated hybrid builds a second one for the
erf-attenuated operator (`pyscf/df/df.py::range_coulomb`). The separable RI of
Duchemin and Blase (`src/Base/factorization/separable_ri.py`), which the GW/BSE code runs
on, has far smaller factors X (M, nao) and D (naux, M):

    (mu nu | la si) ~= sum_{PQ} X_{P mu} X_{P nu} Z_{PQ} X_{Q la} X_{Q si}

with X[P, mu] = chi_mu(r_P) on the interpolation grid and Z = M^T V M. Both
Coulomb matrices follow by contraction, with no three-index tensor anywhere:

    rho_Q = [X Dm X^T]_QQ                    diagonal only
    J     = X^T diag(Z rho) X                                    O(M nao^2)
    K     = X^T [ Z .* (X Dm X^T) ] X        .* elementwise      O(M^2 nao)

K costs 4 M^2 nao, or 3 with the hermitian shortcut; the largest object ever
formed is a block of rows of an (M, M) matrix.

Only K is usable. In K, mu and nu sit on different interpolation points, each
paired there with the density (a sum over occupied orbitals, therefore
compact). In J they sit on the same point as a co-density product
X_{P mu} X_{P nu} of an arbitrary AO pair, diffuse x diffuse included, while
`separable_ri.build_D_F` fits M against test co-densities {all AOs} x {AOs with
l <= l_max_second = 2}, so an f x f product is never in the test set. On
benzene/cc-pVTZ the worst |dJ| element (a carbon 4f with itself) is 2.4e-1
against 1.1e-3 for the f block of K, and the SCF collapses by 129 eV; widening
the test set to l <= 3 takes that element to 2.9e-2 and the SCF still
collapses (by 142 eV), so the failure is variational. So `j_route` defaults
to 'df-direct': J from pyscf's integral-direct DF-J, which stores no
three-index tensor either. ISDF-K on that footing lands within 17 meV of DF-K
on the same auxiliary basis, the size of the RI error it sits on top of.

Z is (M, M), far smaller than the cderi, and `z_mode='dense'` holds it;
`z_mode='factored'` keeps L = V^{1/2} M, (naux, M), and rebuilds Z's rows per
block, trading time for resident memory by a ratio that grows with naux/nao.
`z_mode='auto'` picks from max_memory.

Range separation. The ISDF ansatz approximates the co-density,
rho_{mu nu}(r) ~= sum_P X_{P mu} X_{P nu} xi_P(r), and the operator only enters
through Z_PQ = (xi_P | w | xi_Q), so a second operator needs a second Z and
nothing else: same grid, X and M, with V replaced by the attenuated
two-centre metric. Z is linear in V and V_SR + V_LR = V, so
K_SR(omega) + K_LR(omega) = K_bare to machine precision (tests/test_isdf_jk.py);
a refitted M_omega would forfeit that identity (it was 3-4x better on water
and no better on benzene, for a second full factorization, and is gone).
Reuse is also numerically safer: the attenuated metric is singular where the
bare one is merely ill-conditioned, and Z_w = M^T V_w M only multiplies by
it, while refitting has to invert it.

References
----------
Lu and Ying, J. Comput. Phys. 302, 329 (2015) -- interpolative separable
density fitting, where the name ISDF and the collocation-plus-interpolation
structure come from.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) -- the separable RI that
is actually fitted here (`src/Base/factorization/separable_ri.py`): the interpolation
coefficients reproduce RI-V fitting coefficients rather than the orbital
products themselves, which is what keeps the per-molecule step cubic.
"""
import contextlib
import threading
import types

import numpy as np
import scipy.linalg
from pyscf import df, scf
from pyscf.lib import logger

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.numerics.constants import (BYTES_PER_GB, ISDF_JK_CHECK_TOL,
                                         ISDF_SCF_KERNEL_MEMORY_FRACTION)
from src.Base.geometry.geometry import elements_of
from src.Base.numerics.numerics import (ISDF_JK_REQUEST_FIELDS, IsdfNumerics,
                                        numerics_or_default, requested, same_setting)
from src.Base.factorization.separable_ri import (ao_values, aux_metric_sqrt,
                                                 fit_M_streaming,
                                                 molecular_points_covariant,
                                                 refuse_unrepresented_factors, require_grid,
                                                 resolve_grid, resolve_isdf_grid)
from src.Base.geometry.body_frame import BodyFrame
from src.Base.factorization.separable_ri import atomic_frames
from src.Base.utils.parallel import memory
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.Base.factorization.separable_ri import points_on_frames


#: The pair layout an ISDFJK built inside `frozen_pair_layout` fits on,
#: and the frames one built inside `frozen_frames` places its points on.
_FROZEN = threading.local()


def omega_key(omega):
    """The key one attenuation omega is stored under (None is the bare 0)."""
    return '%.6f' % (0.0 if omega is None else omega)

@contextlib.contextmanager
def frozen_pair_layout(layout):
    """Every ISDFJK built inside fits on `layout`, a reference geometry's
    `fit_test_set_layout` (mu, nu, weight), in place of the screen at its own
    points, and keeps it (`ISDFJK.pair_layout`); None leaves each its own.

    The screen is a discrete choice: re-taken at every geometry of a walk,
    the energy steps wherever a pair crosses `pair_tol`, while the force
    differentiates one fixed column set and cannot see the step.
    """
    previous = getattr(_FROZEN, 'layout', None)
    _FROZEN.layout = layout
    try:
        yield
    finally:
        _FROZEN.layout = previous


class FrozenFrames:
    """The atomic frames of a reference geometry, turned with the molecule.

    The covariant frames of `atomic_frames` are eigenvectors of each atom's
    neighbour tensor, and where two of its eigenvalues are close -- an sp2
    atom with three similar neighbours: formaldehyde's carbon (relative gap
    3.2e-2), DMABN's planar amine nitrogen (1.0e-2) -- the in-plane axes turn
    at 1/gap: 18.9 and 16.2 rad/Bohr. The ISDF energy depends on each cloud's
    orientation at 1e-5 Ha with sub-5-degree structure, so a cloud turning that
    fast makes the surface rough on a 1e-3 Bohr scale and a finite-difference
    Hessian across it asymmetric (formaldehyde LRC-wPBEh/cc-pVDZ on GH, h =
    1e-3 Bohr: 2.85e-2 Ha/Bohr^2, 2.7 percent of the largest force constant).

    Frozen here, every geometry places its clouds on the reference's frames
    F0_i turned by the Kabsch body frame Q of the whole molecule
    (`body_frame.BodyFrame`): F_i = F0_i Q. The energy stays invariant under a
    rigid motion, and Q turns at the rate of the whole molecule (the same
    Hessian: 2.4e-7). At the reference F_i = F0_i bit for bit, so the energy
    and the points there are those of turning frames; the force differs by
    the frame-turning term, which is the derivative of the orientation
    artifact. A convention, like the pair layout: frozen within a
    finite-difference stencil (`properties.hessian`), recorded where it is.
    """

    def __init__(self, mol):
        self.reference = np.array(mol.atom_coords(), dtype=float)
        # rank 0's on every rank: eigh's last bits differ between nodes
        self.frames = lockstep(atomic_frames(mol)[0])
        self.body = BodyFrame(self.reference)

    def at(self, coords):
        """(natm, 3, 3) frames, rows the axes, at `coords` (Bohr)."""
        q = self.body.rotation(coords)
        return self.frames if q is None else self.frames @ q

    def to_record(self):
        """The convention and the reference geometry, for a record."""
        return {'convention': 'reference-body-fixed',
                'reference': self.reference.tolist()}


@contextlib.contextmanager
def frozen_frames(frames):
    """Every ISDFJK built inside places its points on `frames`, a
    `FrozenFrames`, and keeps them (`ISDFJK.frozen_frames`), its force's point
    chain then turning them with the body frame only; None leaves each the
    covariant frames of its own geometry."""
    previous = getattr(_FROZEN, 'frames', None)
    _FROZEN.frames = frames
    try:
        yield
    finally:
        _FROZEN.frames = previous


def reference_frames(mf):
    """The `FrozenFrames` an ISDF-K mean field at a chain's reference stands
    on, kept on its fit (`ISDFJK.frozen_frames`) so its own force turns its
    clouds the way every other geometry's will; None for a mean field without
    an ISDF-K fit. At the reference the frozen frames ARE the covariant ones,
    so its points and energy are unchanged. A mean field already on frames
    frozen at another geometry is refused: it is a point of another anchor's
    surface, not this reference's."""
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        return None
    coords = np.asarray(mf.mol.atom_coords(), dtype=float)
    held = with_df.frozen_frames
    if held is None:
        if with_df.coords is not None and with_df.grid_radii is None:
            return None          # points handed in: no frames placed them
        held = FrozenFrames(mf.mol)
        # rank 0's frames against this rank's placement: equal to round-off
        if with_df.coords is not None and not np.allclose(
                points_on_frames(mf.mol, with_df.grid_radii,
                                 with_df.grid_origins, held.frames),
                with_df.coords, rtol=0.0, atol=1e-10):
            raise ValueError(
                'this ISDF-K mean field\'s points are not on the covariant '
                'frames of its own geometry, so no frames frozen here are '
                'the ones its energy was taken on')
        with_df.frozen_frames = held
    elif not np.array_equal(held.reference, coords):
        raise ValueError(
            'this ISDF-K mean field places its points on frames frozen at '
            'another geometry: it belongs to that anchor\'s surface and '
            'cannot be the reference of a new one')
    return held


def isdf_grid(mol, counts=None, radii=None, auxbasis=None, n_start=1,
              return_info=False, grid_accuracy=None, frames=None,
              runtime_radii=False):
    """Interpolation points for `mol`, the same way `space_time` picks them.

    The grid is one object: `separable_ri.resolve_grid` decides which grid
    the keywords name, the same decision as `space_time.separable_factors`'.

    Shells are rotated into covariant atomic frames so the grid rotates with
    the molecule.

    grid_accuracy:   an accuracy level of `ISDF_GRID_ACCURACY` or four explicit
                     shell counts, resolved by `resolve_isdf_grid`, which
                     refuses anything the radii table has not got. It sets
                     `counts` and `n_start`, since a validated row is keyed on
                     both.
    radii:           per-element shell radii; optimized per element if omitted.
    n_start: descents per element in `optimize_atomic_radii`. The single
        default descent is the worst of the starting shapes on carbon; six cut
        benzene's exchange-energy error 23x at the same point count.
    return_info: also return the per-element radii and cusp-sample flags the
        points were placed from. A nuclear derivative needs them, because a
        bare point cloud does not say which atom owns which row.
    frames: a `FrozenFrames` the clouds are placed on; None is the covariant
        frames of this geometry (`atomic_frames`).
    runtime_radii: True declares a grid for counts the radii table holds no
        row for, re-optimized here; False refuses one.
    """
    radii, origins, _, _ = resolve_grid(
        mol, IsdfNumerics(level=grid_accuracy, counts=counts, radii=radii,
                          n_start=n_start, auxbasis=auxbasis,
                          runtime_radii=runtime_radii), where='isdf_grid')
    if frames is None:
        pts = molecular_points_covariant(mol, radii, origin_by_element=origins)
    else:
        pts = points_on_frames(mol, radii, origins,
                               frames.at(mol.atom_coords()))
    return (pts, radii, origins) if return_info else pts


# ---------------------------------------------------------------------------
# The contractions
# ---------------------------------------------------------------------------

class _ZRows:
    """Rows of Z, either sliced out of a dense (M, M) array or rebuilt from L.

    The two `z_mode`s differ only here, which is the point of the indirection:
    every kernel below asks for a block of rows and never learns which it got.
    """

    def __init__(self, Z=None, L=None):
        self.Z, self.L = Z, L
        self.nk = Z.shape[0] if Z is not None else L.shape[1]

    def rows(self, p0, p1, q0=0, q1=None):
        q1 = self.nk if q1 is None else q1
        if self.Z is not None:
            return self.Z[p0:p1, q0:q1]          # a view, no copy
        return self.L[:, p0:p1].T @ self.L[:, q0:q1]

    def matvec(self, rho):
        if self.Z is not None:
            return self.Z @ rho
        return self.L.T @ (self.L @ rho)

    @property
    def nbytes(self):
        return (self.Z if self.Z is not None else self.L).nbytes


def _row_blocks(nk, block):
    return [(p0, min(p0 + block, nk)) for p0 in range(0, nk, block)]


def isdf_j(X, zrows, dms, block):
    """J for a stack of density matrices.

    Only the diagonal of X Dm X^T is needed, so J never touches an (M, M)
    object: it is O(M nao^2), and the memory question is K's.
    """
    nk, nao = X.shape
    nset = len(dms)
    rho = np.empty((nset, nk))
    for p0, p1 in _row_blocks(nk, block):
        Xb = X[p0:p1]
        for i, dm in enumerate(dms):
            rho[i, p0:p1] = np.einsum('kp,kp->k', Xb @ dm, Xb)
    v = np.array([zrows.matvec(r) for r in rho])
    vj = np.zeros((nset, nao, nao))
    for p0, p1 in _row_blocks(nk, block):
        Xb = X[p0:p1]
        for i in range(nset):
            vj[i] += Xb.T @ (v[i, p0:p1, None] * Xb)
    return vj


def isdf_k(X, zrows, dms, block):
    """K = X^T [Z .* (X Dm X^T)] X, blocked over the interpolation index.

    The Hadamard product is why K cannot be reassociated into something
    cheaper: Z's elements are needed, not its action. What blocking buys is
    that only `block` rows of it exist at a time, (block, M) against (M, M).

    General in Dm: no hermiticity is assumed, so this is also the hermi=0 path.
    """
    nk, nao = X.shape
    nset = len(dms)
    vk = np.zeros((nset, nao, nao))
    for i, dm in enumerate(dms):
        Y = X @ dm                                   # (nk, nao)
        for p0, p1 in _row_blocks(nk, block):
            A = Y[p0:p1] @ X.T                       # (b, nk)
            A *= zrows.rows(p0, p1)
            vk[i] += X[p0:p1].T @ (A @ X)
            del A
    return vk


def isdf_k_symmetric(X, zrows, dms, block):
    """`isdf_k` over the lower triangle only, for hermitian Dm.

    W = Z .* (X Dm X^T) is then symmetric, so half the blocks determine the
    other half. Saves 25% of the flops in dense mode and 50% of the Z-rebuild
    in factored mode, at the cost of one more (M, nao) buffer.
    """
    nk, nao = X.shape
    nset = len(dms)
    blocks = _row_blocks(nk, block)
    vk = np.zeros((nset, nao, nao))
    for i, dm in enumerate(dms):
        Y = X @ dm
        T = np.zeros((nk, nao))
        for bi, (p0, p1) in enumerate(blocks):
            # columns past p1 belong to the upper triangle and are never read;
            # asking for them would redo the rebuild work the triangular loop
            # avoids.
            Zp = zrows.rows(p0, p1, 0, p1)
            for q0, q1 in blocks[:bi + 1]:
                A = Y[p0:p1] @ X[q0:q1].T
                A *= Zp[:, q0:q1]
                T[p0:p1] += A @ X[q0:q1]
                if q0 != p0:
                    T[q0:q1] += A.T @ X[p0:p1]
                del A
            del Zp
        vk[i] = X.T @ T
    return vk


# ---------------------------------------------------------------------------
# The with_df-like object
# ---------------------------------------------------------------------------

class ISDFJK(df.df.DF):
    """A `with_df` that answers `get_jk` from ISDF factors and stores no cderi.

    Subclasses pyscf's DF purely for the plumbing `_DFHF` expects (auxbasis,
    auxmol, mol, max_memory, reset). Everything that would touch a three-index
    tensor is overridden: `build` factorizes instead, and `loop`/`_cderi` raise
    rather than let a caller silently fall back onto the dense cderi.
    """

    def __init__(self, mol, auxbasis=None, counts=None, radii=None,
                 z_mode=None, block=None, j_route=None,
                 check_tol=ISDF_JK_CHECK_TOL,
                 numerics=None, progress=None, n_start=None):
        # The grid level, the J route, the Z mode and the row block are
        # `numerics`' (scf.grid_level, scf.j_route, layout.z_mode,
        # layout.block) where the keyword is None; a keyword naming another
        # value is refused (`requested`).
        req = requested(numerics, 'ISDFJK', ISDF_JK_REQUEST_FIELDS,
                        auxbasis=auxbasis, grid_accuracy=None, j_route=j_route,
                        z_mode=z_mode, block=block)
        auxbasis = req['auxbasis'] or paired_auxbasis(mol.basis)
        if req['grid_accuracy'] is not None:
            level_counts, level_n_start = resolve_isdf_grid(
                req['grid_accuracy'], mol.basis, elements_of(mol),
                auxbasis=auxbasis)
            if counts is not None and not same_setting(
                    {k: int(v) for k, v in dict(counts).items()}, level_counts):
                raise ValueError(
                    f'ISDFJK: counts {dict(sorted(dict(counts).items()))} and '
                    f'numerics.scf.grid_level {req["grid_accuracy"]!r} name two '
                    f'grids; pass one')
            counts = level_counts
            n_start = level_n_start if n_start is None else n_start
        n_start = 1 if n_start is None else n_start
        j_route = req['j_route'] or 'df-direct'
        z_mode, block = req['z_mode'] or 'auto', req['block']
        super().__init__(mol, auxbasis=auxbasis)
        # The fit's choices (`IsdfNumerics`: the test set's l_max_second, the
        # Tikhonov shift) and the working set of its blocked loops
        # (`LayoutNumerics.fit_gb`). The working set reaches here because the
        # fit is where the peak is: the Gram matrix is n_k^2, and everything else
        # in the build is small beside it.
        # It does not change the answer, but it IS a speed knob: one aux_e2
        # call per block, each rebuilding a shell-pair list over nbas x
        # auxnbas -- see `separable_ri.build_D_F` for the measurement. The
        # l_max_second of 2 is the published scheme's and is right for K,
        # whose co-densities always carry an occupied orbital. It is exactly
        # why J fails: an f x f AO product is never in the test set. Derive a
        # larger one to fit what J needs, at a larger test set and a
        # proportionally longer fit.
        self.numerics = numerics_or_default(numerics)
        # None follows mol.verbose, so the knob that turns on the mean field's
        # output turns on the factorization's too, which can run for minutes
        # to hours.
        self.progress = (mol.verbose > 0) if progress is None else progress
        require_grid(counts, radii, 'ISDFJK')
        self.counts = counts
        # Whether the caller named counts: `isdf_grid` refuses explicit radii
        # that contradict the table row at named counts, so a caller who gave
        # radii alone, for whom the radii are the grid, must not pass counts.
        self._named_counts = counts is not None
        self.radii = radii
        self.n_start = n_start
        self.z_mode = z_mode
        self.block = block
        # j_route: 'isdf' is the pure method, for validation; 'df-direct', the
        # default, takes J from pyscf's integral-direct DF-J (`df_jk.get_j`),
        # which stores only the (naux, naux) metric and removes ISDF-J's
        # variational failure (benzene/cc-pVDZ on the fallback grid: -128810
        # meV with ISDF J, -225 meV with DF-J, the same ISDF exchange in both;
        # water/PBE0 HOMO +25.7 -> -0.6 meV on a good grid). J is the cheap
        # term, O(M nao^2) against K's O(M^2 nao).
        self.j_route = j_route
        self._jdf = {}
        # Relative Coulomb-energy tolerance for the one-off `check`, run on the
        # first density the SCF asks about (ISDF_JK_CHECK_TOL). None disables
        # the check.
        self.check_tol = check_tol
        self._checked = False
        self.grid_warning = None
        # The AO-pair columns the fit keeps, frozen at a reference geometry
        # (`frozen_pair_layout`); None screens at this geometry's own points.
        self.pair_layout = None
        # The frames its points are placed on, frozen at a reference geometry
        # (`frozen_frames`); None is this geometry's covariant frames.
        self.frozen_frames = None

        self.coords = None
        self.X = None            # (nk, nao)
        self.M = None            # (naux, nk)
        # What `build` placed the points from, which a nuclear derivative needs
        # to split them into atom-local clouds. None when `coords` was set from
        # outside, and the gradient refuses that case rather than guessing.
        self.grid_radii = None
        self.grid_origins = None
        self._z = {}             # omega key -> _ZRows
        self._built = False
        # True once a `DistributedISDFJK` divides this handle over ranks
        self._distributed_twin = False

    # -- construction --------------------------------------------------------

    @property
    def nk(self):
        return 0 if self.coords is None else len(self.coords)

    @property
    def built(self):
        """Whether the factors (points, collocation, fit) are built."""
        return self._built

    @property
    def named_counts(self):
        """Whether the caller named the grid's counts (else a level's)."""
        return self._named_counts

    def mark_distributed(self):
        """A `DistributedISDFJK` divides this handle over the ranks: building
        it whole is then refused (`build`)."""
        self._distributed_twin = True

    def place_grid(self):
        """(points, radii, origins) of this handle's interpolation grid: its
        counts (when named), radii, auxiliary basis, frames and the run's
        `runtime_radii`, in the one `isdf_grid` call the serial build and the
        distributed one (`DistributedISDFJK`) both make, so the two fit one
        grid."""
        return isdf_grid(
            self.mol, counts=self.counts if self._named_counts else None,
            radii=self.radii,
            auxbasis=self.auxbasis, n_start=self.n_start, return_info=True,
            frames=self.fit_frames(),
            runtime_radii=self.numerics.isdf.runtime_radii)

    def build(self):
        if self._built:
            return self
        comm = current_comm()
        if (self._distributed_twin and comm is not None
                and comm.Get_size() > 1):
            raise RuntimeError(
                'this ISDFJK is divided over the ranks by a DistributedISDFJK, '
                'and building it whole here would form the (M, M) Gram matrix '
                'and hold X and M whole on every rank. A J/K request on this '
                'mean field under ranks belongs inside '
                'distributed_fock(mf, build=False), which answers it from the '
                "distributed SCF's tiles.")
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        if self.auxmol is None:
            self.auxmol = df.addons.make_auxmol(mol, auxbasis=self.auxbasis)
        if self.coords is None:
            self.coords, self.grid_radii, self.grid_origins = self.place_grid()
        self.X = ao_values(mol, self.coords)
        self.M = fit_M_streaming(mol, self.auxmol, self.coords,
                                 numerics=self.numerics,
                                 progress=self.progress,
                                 layout=self.fit_layout())
        log.timer('ISDF factorization (M = %d points, nao = %d, naux = %d)'
                  % (self.nk, mol.nao_nr(), self.auxmol.nao_nr()), *t0)
        if self.j_route == 'isdf':
            # Not a tolerance question: on benzene/cc-pVDZ a basin-hopping grid
            # takes the probe error from 8.5e-3 to 3.3e-4 and the SCF still
            # collapses by 5.6 eV, because the variational optimization seeks
            # out the modes where the interpolated Coulomb operator is weak.
            log.warn('j_route=\'isdf\' builds the Coulomb matrix from the '
                     'interpolation. That is measurably unsafe in an SCF: the '
                     'minimization finds the modes the fit underestimates, and '
                     'benzene/cc-pVDZ collapses by 5.6 eV even on a grid whose '
                     'probe error is 3.3e-4. Keep it for measuring the pure '
                     "method; use j_route='df-direct' to compute with.")
        self._built = True
        return self

    def fit_layout(self):
        """The pair layout the fit keeps: this handle's frozen one, else the
        one a `frozen_pair_layout` block holds now, which it then keeps; None
        screens at its own points."""
        if self.pair_layout is None:
            self.pair_layout = getattr(_FROZEN, 'layout', None)
        return self.pair_layout

    def fit_frames(self):
        """The frames the points are placed on: this handle's frozen ones,
        else the ones a `frozen_frames` block holds, which it then keeps;
        None is the covariant frames of its geometry."""
        if self.frozen_frames is None:
            self.frozen_frames = getattr(_FROZEN, 'frames', None)
        return self.frozen_frames

    def _resolve_z_mode(self):
        if self.z_mode != 'auto':
            return self.z_mode
        nk = self.nk
        free = memory.left_mb(self.max_memory) * 1e6
        # Dense Z plus one row block plus the (nk, nao) buffers; a third of the
        # remaining memory is the budget, so an SCF still has room to breathe.
        return ('dense' if nk * nk * 8 < ISDF_SCF_KERNEL_MEMORY_FRACTION * free
                else 'factored')

    def _zrows(self, omega):
        """Z for one operator, built once and cached by omega."""
        key = omega_key(omega)
        if key in self._z:
            return self._z[key]
        if not self._built:
            self.build()
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        om = 0.0 if omega is None else omega

        M = self.M
        with self.mol.with_range_coulomb(om), self.auxmol.with_range_coulomb(om):
            V = self.auxmol.intor('int2c2e', aosym='s1')

        # Resolved once: it reads the process's memory, which the dense branch
        # then changes, so asking twice can report a mode that was not used.
        mode = self._resolve_z_mode()
        if mode == 'dense':
            zr = _ZRows(Z=M.T @ (V @ M))
        else:
            zr = _ZRows(L=aux_metric_sqrt(V=V) @ M)
        self._z[key] = zr
        log.timer('ISDF Z for omega=%s (%.2f GB, %s)'
                  % (key, zr.nbytes / BYTES_PER_GB, mode), *t0)
        return zr

    def _block(self):
        if self.block:
            return self.block
        nk, nao = self.X.shape
        free = max(memory.left_mb(self.max_memory) * 1e6, 2e8)
        # A row block costs b*nk (the Hadamard argument) and, in factored mode,
        # b*nk again for the rebuilt Z rows.
        b = int(0.25 * free / (2 * nk * 8))
        return int(np.clip(b, 16, nk))

    def probe_densities(self):
        """Densities to test the interpolation against, cheapest first.

        Not the SCF's own density: the failure this catches is a J operator
        with bad modes that the SCF then finds variationally, which a physical
        density misses (on benzene/cc-pVDZ with the fallback grid, the exact HF
        density gives a Coulomb error of 3.5e-4 while the SCF collapses by
        129 eV). The core-Hamiltonian guess, far too diffuse to be physical,
        gives 8.5e-3 there and 4.8e-5 on the published grid.
        """
        out = {}
        for key in ('minao', '1e'):
            try:
                out[key] = scf.hf.SCF(self.mol).get_init_guess(key=key)
            except Exception:                  # a guess unavailable for this mol
                pass
        return out

    def check(self, dms=None):
        """max relative Coulomb-energy error against integral-direct DF-J.

        The reference is `df_jk.get_j`, which stores no three-index tensor
        either, so this costs about one J build per probe.

        J is the probe rather than K because J reads the part of the fit
        nobody constrained, co-densities of two arbitrary AOs including the
        high-l products `l_max_second` leaves out of the test set; a bad grid
        shows up in J one to two orders of magnitude before K. It is also the
        cheap term.

        A pass is necessary, not sufficient: it does not license
        `j_route='isdf'`, which fails variationally at probe values far below
        any tolerance (see `build`).
        """
        if not self._built:
            self.build()
        if dms is None:
            dms = self.probe_densities()
        if not isinstance(dms, dict):
            dms = {'dm': dms}
        nao = self.mol.nao_nr()
        worst = 0.0
        for dm in dms.values():
            dm = np.asarray(dm).reshape(-1, nao, nao)[0]
            vj_ref = self._df_j(dm[None], 1, 1e-13, None)[0]
            vj = isdf_j(self.X, self._zrows(None), dm[None], self._block())[0]
            e_ref = 0.5 * np.einsum('ij,ji->', vj_ref, dm)
            worst = max(worst, abs(0.5 * np.einsum('ij,ji->', vj, dm) - e_ref)
                        / max(abs(e_ref), 1e-12))
        return float(worst)

    def check_k(self, dms=None, omega=None):
        """Exchange-energy error of ISDF-K against stored-cderi DF-K, in Hartree.

        `check` probes J because a broken grid shows there first, but with
        `j_route='df-direct'` the only term the grid touches is K, and a K
        error too small to trip the J probe is not small in absolute terms on
        a large molecule (1e-3 relative can be a tenth of a Hartree). Returns
        (max |dE_x| in Ha, max relative), with E_x = -1/4 tr(D K) for a
        closed-shell D.

        The reference builds a three-index tensor, naux x nao^2 / 2, so this is a
        small-molecule calibration tool for choosing grids, not a check to run
        inside a production SCF. `omega` probes the attenuated K a range-
        separated functional also builds.
        """
        if not self._built:
            self.build()
        if dms is None:
            dms = self.probe_densities()
        if not isinstance(dms, dict):
            dms = {'dm': dms}
        ref = df.DF(self.mol, auxbasis=self.auxmol.basis)
        ref.build()
        nao = self.mol.nao_nr()
        worst_abs, worst_rel = 0.0, 0.0
        for dm in dms.values():
            dm = np.asarray(dm).reshape(-1, nao, nao)[0]
            vk_ref = ref.get_jk(dm, hermi=1, with_j=False, omega=omega)[1]
            vk = self.get_jk(dm, hermi=1, with_j=False, omega=omega)[1]
            ex_ref = -0.25 * np.einsum('ij,ji->', vk_ref, dm)
            d = abs(-0.25 * np.einsum('ij,ji->', vk, dm) - ex_ref)
            worst_abs = max(worst_abs, d)
            worst_rel = max(worst_rel, d / max(abs(ex_ref), 1e-12))
        return float(worst_abs), float(worst_rel)

    # -- the interface _DFHF calls ------------------------------------------

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        dms = np.asarray(dm)
        shape = dms.shape
        nao = shape[-1]
        dms = dms.reshape(-1, nao, nao)
        if np.iscomplexobj(dms):
            raise NotImplementedError('ISDF J/K takes real density matrices')

        vj = vk = None
        if with_j and self.j_route == 'df-direct':
            vj = self._df_j(dms, hermi, direct_scf_tol, omega).reshape(shape)
            with_j = False
        if (self.check_tol is not None and not self._checked
                and (with_k or self.j_route == 'isdf')):
            self._checked = True                 # set first: check() calls back in
            rel = self.check()
            if rel > self.check_tol:
                remedy = ("J is already taken from integral-direct DF-J so the "
                          "Coulomb term is protected, but the same grid builds "
                          "K." if self.j_route == 'df-direct' else
                          "Set j_route='df-direct', which removes this failure "
                          "mode entirely at no memory cost.")
                logger.warn(self, 'ISDF interpolation grid is weak: the Coulomb '
                            'energy of a probe density is %.2e off integral-'
                            'direct DF-J (tolerance %.1e), on M = %d points '
                            'over %d atoms (%d per atom). Radii come from the '
                            'shipped table, or from optimize_atomic_radii where '
                            'it holds no row: raise the point COUNT '
                            '(counts=) or the number of starts (n_start=), NOT '
                            'basin_hopping, which re-optimizes radii at fixed '
                            'count and was measured at 11x the optimizer time '
                            'for ~6%% (C/cc-pVTZ 0.2326 -> 0.2191) against a '
                            '~10x gap to the published tables at matched '
                            'count. %s',
                            rel, self.check_tol, self.nk, self.mol.natm,
                            self.nk // max(self.mol.natm, 1), remedy)
                self.grid_warning = rel
        if not (with_j or with_k):
            return vj, vk
        # Only now is the factorization needed. A pure functional on the
        # default j_route asks for J alone, and never pays for it.
        if not self._built:
            self.build()
        zrows = self._zrows(omega)
        block = self._block()
        if with_j:
            vj = isdf_j(self.X, zrows, dms, block).reshape(shape)
        if with_k:
            kern = isdf_k_symmetric if hermi == 1 else isdf_k
            vk = kern(self.X, zrows, dms, block).reshape(shape)
        return vj, vk

    def _df_j(self, dms, hermi, direct_scf_tol, omega):
        """J from pyscf's integral-direct DF-J -- which builds no cderi.

        `df_jk.get_jk` routes to `df_jk.get_j` whenever `with_k` is false and
        `_cderi is None`, and that path is two screened passes over int3c2e
        with only the (naux, naux) metric held: no three-index tensor on disk
        or in core, and J from the RI-V fit rather than the interpolation.
        """
        key = omega_key(omega)
        if key not in self._jdf:
            d = df.DF(self.mol, auxbasis=self.auxbasis)
            d.max_memory = self.max_memory
            self._jdf[key] = d           # deliberately never .build()-ed
        return self._jdf[key].get_jk(dms, hermi, True, False,
                                     direct_scf_tol, omega)[0]

    def get_naoaux(self):
        return self.nk

    def reset(self, mol=None):
        super().reset(mol)
        # `pair_layout` and `frozen_frames` survive: frozen conventions are a
        # choice of the walk, like the counts and radii, and a scanner's next
        # geometry fits on them too
        if mol is not None:
            self.coords = None
            self.X = self.M = None
            self.grid_radii = self.grid_origins = None
            self._z = {}
            self._jdf = {}
            self._checked = False
            self.grid_warning = None
            self._built = False
        return self

    # -- the cderi path, closed off -----------------------------------------

    def loop(self, blksize=None):
        raise NotImplementedError(
            'ISDFJK stores no three-index tensor -- that is the point. '
            'A caller reaching loop()/_cderi wants pyscf DF; give it a '
            'pyscf.df.DF instead of silently materializing the tensor.')

    @property
    def _cderi(self):
        raise NotImplementedError('ISDFJK stores no cderi; see loop().')

    @_cderi.setter
    def _cderi(self, x):
        if x is not None:
            raise NotImplementedError('ISDFJK stores no cderi; see loop().')


def separable_factors_from_jk(mf):
    """(X_mo, D, X_ao, coords) for GW/BSE, from an SCF that already fitted.

    `space_time.separable_factors` and `ISDFJK.build` construct the same
    factorization (the same published tables, the same resolver and covariant
    frames, and the one metric root `aux_metric_sqrt`), so a BSE run on top
    of an ISDF SCF takes the SCF's rather than building it twice.

    Returns the same pair `separable_factors` would, in the same auxiliary
    gauge, which is the part that must not drift: pairing factors from one fit
    with a W from another is a silent gauge error.

    Refused, as `separable_factors` refuses, where they miss an orbital
    density of `mf` by more than the bar (`refuse_unrepresented_factors`).
    """
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        raise TypeError(
            'separable_factors_from_jk needs an SCF whose with_df is an ISDFJK '
            f'(got {type(with_df).__name__}); a mean field that did not fit '
            'one needs its factors built by build_separable_ri.')
    # cycle: pyscf_interface -> distributed_df -> this module
    from src.Base.pyscf.pyscf_interface import occupied_count
    if not with_df.built:
        with_df.build()
    V = with_df.auxmol.intor('int2c2e', aosym='s1')
    root = aux_metric_sqrt(V=V)
    out = (with_df.X @ mf.mo_coeff, with_df.M.T @ root, with_df.X,
           with_df.coords)
    refuse_unrepresented_factors(
        mf.mol, with_df.auxmol, np.asarray(mf.mo_coeff, float), out[0],
        out[1], root=root, nocc=occupied_count(mf))
    return out


def canonical_basis(S, threshold):
    """(nao, m) canonical-orthogonal basis X = U w^-1/2 of the overlap
    eigenvectors with eigenvalue above `threshold` (Lowdin's canonical
    orthogonalization); X^T S X = 1."""
    w, U = np.linalg.eigh(S)
    keep = w > threshold
    return U[:, keep] / np.sqrt(w[keep])


def solve_in_span(mf, basis):
    """`mf` with its orbitals and its DIIS confined to the AO span `basis`
    (nao, m), orthonormal in the overlap (basis^T S basis = 1): the mean field
    then carries m orbitals, none outside the span. `canonical_basis` gives the
    canonical cut; `separable_ri.represented_span` the directions the
    separable-RI factors represent. The span belongs to one geometry.

    Both hooks are set: up to pyscf 2.12 the kernel calls the eigensolver
    `_eigh(h, s)`; from 2.13 it takes its orthogonalizer from
    `check_linear_dependency` and hands it to the eigensolver and to DIIS
    alike, so an eigensolver hook alone leaves DIIS in the larger space and the
    SCF stalls (|g| 1e-4 to 1e-2). Returns `mf`.
    """
    basis = np.asarray(basis, float)

    def eigh_in_span(h, s, *_ignored, **_ignored_kw):
        e, c = scipy.linalg.eigh(basis.T @ h @ basis)
        return e, basis @ c

    mf._eigh = eigh_in_span
    mf.check_linear_dependency = lambda s1e, log=None: basis
    return mf


def canonical_span_size(mol, cut):
    """How many overlap eigenvectors of `mol` lie above `cut`, or None when
    none lies below it and the full basis is the span."""
    w = np.linalg.eigvalsh(mol.intor('int1e_ovlp'))
    kept = int((w > cut).sum())
    return None if kept == len(w) else kept


@contextlib.contextmanager
def frozen_span(basis):
    """Every `span_confined` mean field built inside is solved in the span of
    `basis` (nao, m), AO coefficients frozen at a reference geometry, instead
    of a canonical cut taken at its own; None leaves each its own cut.

    With the coefficients fixed the span moves with the atoms through the AO
    functions alone, so every orbital variation stays inside it and the
    ordinary force is the exact slope of the energy: on benzene aug-cc-pVDZ
    (13 of 192 directions dropped at 1e-4) the DF-HF force matches the finite
    difference to 1.8e-7 Ha/Bohr, where a cut re-taken at every geometry
    misses it by 4.9e-6. A refreeze cuts the span again at its new reference;
    the jump is part of the drift it measures (-0.004 meV for a 0.02 Bohr
    step of that benzene).
    """
    previous = getattr(_FROZEN, 'span', None)
    _FROZEN.span = basis
    try:
        yield
    finally:
        _FROZEN.span = previous


def span_confined(mf, n_kept, cut=None):
    """`mf` solved in a span of `n_kept` overlap eigenvector directions of its
    molecule (`solve_in_span`), or `mf` untouched where n_kept is None.

    The near-null AO combinations of an augmented basis carry virtual orbitals
    with coefficient norms of 30-400, whose densities the separable-RI factors
    miss and on which a BSE binds spurious roots. Inside `frozen_span` the
    span is the frozen basis, orthonormalized in this geometry's overlap;
    outside it the `n_kept` largest overlap eigenvectors here (canonical
    orthogonalization). A count, not a threshold, is what a launch fixes: at a
    fixed threshold an eigenvalue crossing it changes the span's dimension.
    `cut` is the threshold the count was taken at and is required with a
    count: it is what mf.span records. mf.span_basis
    is the basis's AO coefficients, which a chain freezes; mf.span records the
    cut, the counts, whether the span was frozen, and the overlap eigenvalues
    on either side of the boundary here.
    """
    if n_kept is None:
        return mf
    if cut is None:
        raise ValueError('a span count names the cut it was taken at')
    S = mf.mol.intor('int1e_ovlp')
    w = np.linalg.eigvalsh(S)
    nao = len(w)
    if not 0 < n_kept <= nao:
        raise ValueError(f'n_kept={n_kept}: between 1 and nao = {nao}')
    first = nao - n_kept
    frozen = getattr(_FROZEN, 'span', None)
    if frozen is not None:
        if frozen.shape != (nao, n_kept):
            raise ValueError(f'the frozen span is {frozen.shape}, this mean '
                             f'field keeps {n_kept} of {nao}')
        # the same span, orthonormal in this geometry's overlap
        m, V = np.linalg.eigh(frozen.T @ S @ frozen)
        basis = frozen @ (V / np.sqrt(m)) @ V.T
        mf.span_basis = frozen
    else:
        w, U = np.linalg.eigh(S)
        basis = mf.span_basis = U[:, first:] / np.sqrt(w[first:])
    mf.span = {'cut': cut, 'n_kept': int(n_kept), 'n_dropped': int(first),
               'nao': int(nao), 'frozen': frozen is not None,
               'smallest_kept_eigenvalue': float(w[first]),
               'largest_dropped_eigenvalue': (float(w[first - 1]) if first
                                              else 0.0)}
    return solve_in_span(mf, basis)


def isdf_jk(mf, auxbasis=None, counts=None, radii=None, z_mode=None,
            block=None,
            j_route=None, check_tol=ISDF_JK_CHECK_TOL, numerics=None,
            progress=None, n_start=None):
    """Give `mf` an ISDF `with_df`. Returns the modified SCF object.

    Routes through pyscf's own `density_fit` so the `_DFHF` mixin (which is
    what dispatches `get_jk(..., omega=...)` for a range-separated hybrid) is
    installed as usual, then swaps the DF object underneath it.

    The returned mean field carries no force (`without_force`): pyscf's own
    gradient differentiates the fitted interaction, 8.4e-5 Ha/Bohr from a
    finite difference of this route's own energy on water/cc-pVDZ/B3LYP where
    the ISDF force sits at 7e-8, so it is refused by name until the gradient
    layer attaches the ISDF force (`isdf_jk_with_force`).

    The settings `numerics` holds (scf.grid_level, auxbasis, j_route;
    layout.z_mode, block) are read where the keyword is None (`ISDFJK`).
    """
    auxbasis = requested(numerics, 'isdf_jk', ISDF_JK_REQUEST_FIELDS,
                         auxbasis=auxbasis)['auxbasis']
    out = mf.density_fit(auxbasis=auxbasis or paired_auxbasis(mf.mol.basis))
    out.with_df = ISDFJK(mf.mol, auxbasis=auxbasis, counts=counts, radii=radii,
                         z_mode=z_mode, block=block, j_route=j_route,
                         check_tol=check_tol, numerics=numerics,
                         progress=progress,
                         n_start=n_start)
    out.with_df.max_memory = mf.max_memory
    return without_force(out)


class ISDFForceNotAttached(RuntimeError):
    """pySCF's own gradient asked of an ISDF mean field that carries no force."""


def without_force(mf):
    """`mf` with pySCF's own gradient refused by name and no force hook.

    pySCF's gradient of an ISDF mean field differentiates the FITTED
    interaction, not the interpolated one whose energy the mean field reports.
    The ISDF force is handed to the mean field as an object: whatever attaches
    it binds the force and leaves itself on the mean field as `force_hook`, so
    a continuum wrapped around it later (`SolventScreening`) rebinds it
    without knowing where the force comes from.
    """
    def refuse(m):
        raise ISDFForceNotAttached(
            'this ISDF mean field carries no force: pySCF\'s own gradient '
            'differentiates the fitted interaction, not the interpolated one '
            'its energy comes from. Attach an ISDF force to the mean field '
            '(attach_isdf_gradient), or build it with isdf_jk_with_force')
    mf.nuc_grad_method = types.MethodType(refuse, mf)
    mf.Gradients = mf.nuc_grad_method
    mf.force_hook = None
    return mf
