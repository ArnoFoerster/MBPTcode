"""ISDF (separable-RI) J and K for the SCF, in place of a density-fitted cderi.

pyscf's DF builds `cderi`, naux x nao(nao+1)/2, and streams it once per SCF
iteration; a range-separated hybrid builds a second one for the
erf-attenuated operator (`pyscf/df/df.py::range_coulomb`). The separable RI of
Duchemin and Blase (`src/Base/separable_ri.py`), which the GW/BSE code runs
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
a refitted M_omega would forfeit that identity. M is minimized only in the
bare Coulomb norm; `refit_omega=True` builds an independent M_omega from
attenuated three-centre integrals for comparison (3-4x better on water, no
gain on benzene, for a second full factorization). Reuse is also numerically
safer: the attenuated metric is singular where the bare one is merely
ill-conditioned, and Z_w = M^T V_w M only multiplies by it, while refitting has
to invert it.

References
----------
Lu and Ying, J. Comput. Phys. 302, 329 (2015) -- interpolative separable
density fitting, where the name ISDF and the collocation-plus-interpolation
structure come from.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) -- the separable RI that
is actually fitted here (`src/Base/separable_ri.py`): the interpolation
coefficients reproduce RI-V fitting coefficients rather than the orbital
products themselves, which is what keeps the per-molecule step cubic.
"""
import contextlib
import warnings

import numpy as np
import scipy.linalg
from pyscf import df, dft, gto, lib, scf
from pyscf.lib import logger
from pyscf.scf.dispersion import parse_disp

from src.Base.constants import ISDF_RADII_MATCH_TOL
from src.Base.pcm_derivatives import solvation_gradient
from src.Base.separable_ri import (ANGULAR_WEIGHTS, DEFAULT_REGULARIZATION,
                                   _ao_l_labels, atomic_grid, build_D_F,
                                   fit_M_stable, fit_M_streaming,
                                   molecular_points_covariant,
                                   optimize_atomic_radii, resolve_isdf_grid,
                                   shipped_radii_lookup, default_auxbasis)
from src.Base.utils import memory
from src.Base.utils.mpi_grid import current_comm

#: `space_time.separable_factors`' grid, so a J/K built here and a GW run share
#: one factorization when the caller wants that. 148 points per atom.
DEFAULT_COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}

#: Hartree-Fock minus its own exchange, for `exchange_free_reference`: the
#: empty functional, so a pure Hartree-Fock mean field keeps none of it.
_NO_FUNCTIONAL = '0*LDA'

#: Eigenvalue floor for V^{1/2}. The attenuated metric is far more
#: rank-deficient than the bare one -- erf(omega r)/r is smooth, so its
#: two-centre matrix decays fast in the aux basis -- and a relative floor keeps
#: the square root real without discarding anything that carries weight.
_METRIC_EIG_TOL = 1e-12


@contextlib.contextmanager
def range_coulomb(mol, auxmol, omega):
    """`omega` on both molecules, the way `pyscf/df/df.py::range_coulomb` does.

    omega > 0 is the long-range erf(omega r)/r, omega < 0 the short-range
    complement, 0 or None the bare operator -- pyscf's convention, which
    `dft/rks.py::get_veff` relies on when it asks for `omega=-omega`.

    Both molecules, because a three-centre integral is driven by a concatenated
    Mole whose _env inherits the first argument's range-omega slot; setting only
    the auxiliary one silently leaves int3c2e bare.
    """
    if omega is None:
        omega = 0.0
    saved = (mol.omega, None if auxmol is None else auxmol.omega)
    mol.omega = omega
    if auxmol is not None:
        auxmol.omega = omega
    try:
        yield
    finally:
        mol.omega = saved[0]
        if auxmol is not None:
            auxmol.omega = saved[1]


def isdf_grid(mol, counts=None, radii=None, auxbasis=None, n_start=1,
              return_info=False, grid_accuracy=None):
    """Interpolation points for `mol`, the same way `space_time` picks them.

    The grid is one object, so a keyword naming it is never dropped; the
    same decision table as `space_time.separable_factors`:

      grid_accuracy alone   `resolve_isdf_grid` sets `counts` and `n_start`.
      grid_accuracy+counts  equal -> proceed; different -> ValueError. One
                            request cannot be two grids.
      counts alone          tabulated -> the row; not tabulated -> a warning
                            and a run-time re-optimization onto another local
                            minimum of a multi-modal surface, which is a grid
                            nothing validated.
      radii alone           the radii are the grid; no row is consulted.
      radii+counts          where a row exists for those counts the two must
                            agree to `ISDF_RADII_MATCH_TOL` or the call is
                            refused, and the row's `origin` is honoured (it
                            places one extra point at the nucleus: 307 points
                            for a Duchemin-Blase carbon row, not 306).
      nothing               `DEFAULT_COUNTS`, sized for double zeta.

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
    """
    auxbasis = auxbasis or default_auxbasis(mol.basis)
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
            # One table lookup at the counts asked for: a published grid is a
            # row at its own counts, never substituted for other counts (a
            # grid-convergence study would then return the same number at
            # every count).
            try:
                radii[el], origins[el] = atomic_grid(el, mol.basis, auxbasis,
                                                     counts)
            except KeyError as no_row:
                warnings.warn(
                    f'ISDF grid re-optimized at run time: {no_row.args[0]} '
                    f'Re-optimizing lands on another local minimum of a '
                    f'multi-modal surface, so this grid is not a validated '
                    f'one and is not reproducible from a clean checkout.',
                    RuntimeWarning, stacklevel=2)
                radii[el] = optimize_atomic_radii(el, mol.basis, auxbasis,
                                                  counts=counts,
                                                  n_start=n_start)[0]
                origins[el] = False
    else:
        # Explicit radii are the grid and are honoured. Where the caller also
        # named counts the table may describe the same grid, and then the two
        # specifications have to agree: `origin` belongs to the row, not to
        # the recipe, so a matching row brings its nuclear point with it.
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
    pts = molecular_points_covariant(mol, radii, origin_by_element=origins)
    return (pts, radii, origins) if return_info else pts


def metric_sqrt(V, tol=_METRIC_EIG_TOL):
    """V^{1/2} through a truncated eigendecomposition.

    V is a Coulomb (or attenuated Coulomb) two-centre matrix, positive
    semidefinite by construction, but the attenuated one is numerically
    singular, so a Cholesky is not available and the small modes are dropped.
    """
    w, v = np.linalg.eigh(V)
    keep = w > tol * max(w.max(), 1e-300)
    return (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T


def fit_M_omega(mol, auxmol, coords, omega, l_max_second=2,
                regularization=DEFAULT_REGULARIZATION, metric_rcond=1e-10):
    """M refitted against the attenuated metric, the "second factorization".

    Same least-squares estimator as `separable_ri.fit_M`, but the RI fitting
    coefficients it targets are taken with the range-separated operator:

        F^w_{beta,rho} = sum_gamma [V_w^{-1}]_{beta gamma} (gamma | w | rho)

    A pseudo-inverse, not an LU solve: the attenuated two-centre metric is
    numerically singular where the bare one is merely ill-conditioned (on
    water/cc-pVDZ-RI, cond(V) = 2.5e5 against cond(V_w) = 2.5e18 at
    omega = 0.3), and `separable_ri.build_D_F`'s `lu_solve` on it returns
    noise. The reuse route has no such fragility: Z_w = M^T V_w M only
    multiplies by the metric.
    """
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(coords)
    ao = mol.eval_gto('GTOval_sph', coords)
    aux_on_grid = auxmol.eval_gto('GTOval_sph', coords)
    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])

    with range_coulomb(mol, auxmol, omega):
        V = auxmol.intor('int2c2e', aosym='s1')
        e3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                               aosym='s1').reshape(nao, nao, naux)
    Vinv = np.linalg.pinv(V, rcond=metric_rcond, hermitian=True)

    D = (ao[:, :, None] * ao[:, None, second]).reshape(nk, -1)
    D *= np.tile(w, nao)[None, :]
    F = (e3c[:, second, :].reshape(-1, naux) @ Vinv).T
    F *= np.tile(w, nao)[None, :]
    # The auxiliary block of the test set: F^w(gamma) = V_w^{-1} V_w, which is
    # the identity only on the retained modes, so it is written out rather
    # than assumed, unlike the bare case where `build_D_F` puts an eye there.
    D = np.hstack([D, aux_on_grid])
    F = np.hstack([F, Vinv @ V])
    # The Cholesky solve, not the explicit inverse: cond(G) reaches ~2e8 on a
    # real grid, where forming G^-1 costs ~2e-9 relative against the same
    # estimator solved stably.
    return fit_M_stable(D, F, regularization)


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
                 z_mode='auto', block=None, refit_omega=False,
                 use_symmetry=True, j_route='df-direct', check_tol=1e-3,
                 l_max_second=2, regularization=DEFAULT_REGULARIZATION,
                 block_memory_gb=4.0, progress=None, n_start=1):
        super().__init__(mol, auxbasis=auxbasis or default_auxbasis(mol.basis))
        # Caps the working set of the fit's blocked loops, where the build's
        # peak is (the Gram matrix is n_k^2). It does not change the answer
        # but is a speed knob: one aux_e2 call per block, each rebuilding a
        # shell-pair list over nbas x auxnbas (`separable_ri.build_D_F`).
        self.block_memory_gb = block_memory_gb
        # None follows mol.verbose, so the knob that turns on the mean field's
        # output turns on the factorization's too, which can run for minutes
        # to hours.
        self.progress = (mol.verbose > 0) if progress is None else progress
        self.counts = counts or DEFAULT_COUNTS
        # Whether the caller named counts. `isdf_grid` refuses explicit radii
        # that contradict the table row at named counts, so passing the default
        # on would refuse a caller who gave radii alone, for whom the radii are
        # the grid.
        self._named_counts = counts is not None
        self.radii = radii
        self.n_start = n_start
        self.z_mode = z_mode
        self.block = block
        self.refit_omega = refit_omega
        self.use_symmetry = use_symmetry
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
        # first density the SCF asks about: benzene/cc-pVDZ on
        # `optimize_atomic_radii`'s default single descent puts 0.31 Ha of
        # error into a single J element and collapses the SCF by 129 eV, while
        # K stays good to 5e-3. None disables the check.
        self.check_tol = check_tol
        self._checked = False
        self.grid_warning = None
        # Angular momentum cutoff on the second index of the test co-densities
        # M is fitted against (`separable_ri.build_D_F`). The default of 2 is
        # the published scheme's and is right for K, whose co-densities always
        # carry an occupied orbital; it is why J fails (an f x f AO product is
        # never in the test set). Raising it enlarges the test set and the fit.
        self.l_max_second = l_max_second
        self.regularization = regularization

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
            self.coords, self.grid_radii, self.grid_origins = isdf_grid(
                mol, counts=self.counts if self._named_counts else None,
                radii=self.radii,
                auxbasis=self.auxbasis, n_start=self.n_start, return_info=True)
        self.X = mol.eval_gto('GTOval_sph', self.coords)
        self.M = fit_M_streaming(mol, self.auxmol, self.coords,
                                 l_max_second=self.l_max_second,
                                 regularization=self.regularization,
                                 block_memory_gb=self.block_memory_gb,
                                 progress=self.progress)
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

    def _resolve_z_mode(self):
        if self.z_mode != 'auto':
            return self.z_mode
        nk = self.nk
        free = (self.max_memory - memory.current_memory_mb()) * 1e6
        # Dense Z plus one row block plus the (nk, nao) buffers; a third of the
        # remaining memory is the budget, so an SCF still has room to breathe.
        return 'dense' if nk * nk * 8 < 0.33 * free else 'factored'

    def _zrows(self, omega):
        """Z for one operator, built once and cached by omega."""
        key = '%.6f' % (0.0 if omega is None else omega)
        if key in self._z:
            return self._z[key]
        if not self._built:
            self.build()
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        om = 0.0 if omega is None else omega

        if om != 0.0 and self.refit_omega:
            M = fit_M_omega(self.mol, self.auxmol, self.coords, om,
                            l_max_second=self.l_max_second,
                            regularization=self.regularization)
        else:
            M = self.M
        with range_coulomb(self.mol, self.auxmol, om):
            V = self.auxmol.intor('int2c2e', aosym='s1')

        # Resolved once: it reads the process's memory, which the dense branch
        # then changes, so asking twice can report a mode that was not used.
        mode = self._resolve_z_mode()
        if mode == 'dense':
            zr = _ZRows(Z=M.T @ (V @ M))
        else:
            zr = _ZRows(L=metric_sqrt(V) @ M)
        self._z[key] = zr
        log.timer('ISDF Z for omega=%s (%.2f GB, %s)'
                  % (key, zr.nbytes / 1e9, mode), *t0)
        return zr

    def _block(self):
        if self.block:
            return self.block
        nk, nao = self.X.shape
        free = max((self.max_memory - memory.current_memory_mb()) * 1e6, 2e8)
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
            kern = isdf_k_symmetric if (self.use_symmetry and hermi == 1) else isdf_k
            vk = kern(self.X, zrows, dms, block).reshape(shape)
        return vj, vk

    def _df_j(self, dms, hermi, direct_scf_tol, omega):
        """J from pyscf's integral-direct DF-J -- which builds no cderi.

        `df_jk.get_jk` routes to `df_jk.get_j` whenever `with_k` is false and
        `_cderi is None`, and that path is two screened passes over int3c2e
        with only the (naux, naux) metric held: no three-index tensor on disk
        or in core, and J from the RI-V fit rather than the interpolation.
        """
        key = '%.6f' % (0.0 if omega is None else omega)
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
    factorization (the same published tables, `optimize_atomic_radii`
    fallback and covariant frames, and `metric_sqrt`'s tolerance is
    `separable_factors`' 1e-12), so a BSE run on top of an ISDF SCF takes the
    SCF's rather than building it twice.

    Returns the same pair `separable_factors` would, in the same auxiliary
    gauge, which is the part that must not drift: pairing factors from one fit
    with a W from another is a silent gauge error.
    """
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        raise TypeError(
            'separable_factors_from_jk needs an SCF whose with_df is an ISDFJK '
            f'(got {type(with_df).__name__}); use space_time.separable_factors '
            'for a mean field that did not fit one.')
    if not with_df._built:
        with_df.build()
    V = with_df.auxmol.intor('int2c2e', aosym='s1')
    return (with_df.X @ mf.mo_coeff, with_df.M.T @ metric_sqrt(V),
            with_df.X, with_df.coords)


def isdf_jk(mf, auxbasis=None, counts=None, radii=None, z_mode='auto',
            block=None, refit_omega=False, use_symmetry=True,
            j_route='df-direct', check_tol=1e-3, l_max_second=2,
            block_memory_gb=4.0, progress=None, n_start=1):
    """Give `mf` an ISDF `with_df`. Returns the modified SCF object.

    Routes through pyscf's own `density_fit` so the `_DFHF` mixin (which is
    what dispatches `get_jk(..., omega=...)` for a range-separated hybrid) is
    installed as usual, then swaps the DF object underneath it.

    The returned mean field carries `attach_isdf_gradient`, so its
    `Gradients()` is the ISDF force: pyscf's own differentiates the fitted
    interaction, 8.4e-5 Ha/Bohr from a finite difference of this route's own
    energy on water/cc-pVDZ/B3LYP where the ISDF force sits at 7e-8.
    """
    out = mf.density_fit(auxbasis=auxbasis or default_auxbasis(mf.mol.basis))
    out.with_df = ISDFJK(mf.mol, auxbasis=auxbasis, counts=counts, radii=radii,
                         z_mode=z_mode, block=block, refit_omega=refit_omega,
                         use_symmetry=use_symmetry, j_route=j_route,
                         check_tol=check_tol, l_max_second=l_max_second,
                         block_memory_gb=block_memory_gb, progress=progress,
                         n_start=n_start)
    out.with_df.max_memory = mf.max_memory
    # Every ISDF mean field gets the ISDF force rather than each caller asking
    # for it: pyscf's own gradient differentiates the fitted interaction.
    # cycle: gradients.isdf_mean_field imports ISDFJK from this module
    from src.gradients.isdf_mean_field import attach_isdf_gradient
    return attach_isdf_gradient(out)


class _NoExactExchange(dft.numint.NumInt):
    """A numint reporting no exact exchange, with the functional untouched.

    pyscf's Kohn-Sham `get_veff` and its gradient both read the exchange
    fractions from `rsh_and_hybrid_coeff` and never by re-parsing the string,
    so zeroing them here removes the exact exchange and leaves `eval_xc` as it
    was, which for a range-separated functional is the short-range DFT
    exchange libxc holds under the same name.

    Appending "- a_x*HF" to the functional instead could not express a
    range-separated hybrid and could not survive a dispersion suffix
    (`b3lyp-d3bj - 0.2*HF` is not a name pyscf can split).
    """

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        return 0.0, 0.0, 0.0

    def hybrid_coeff(self, xc_code, spin=0):
        return 0.0

    def rsh_coeff(self, xc_code):
        return 0.0, 0.0, 0.0


def _base_functional(xc):
    """`xc` with any empirical-dispersion suffix removed.

    Keyed on the dispersion version, not on `parse_disp`'s first return: that
    element is "xc_code_for_dftd3", the name to hand the DFTD3 library, and
    can be None for a plain functional; with no dispersion suffix `xc` itself
    is returned.

    The dispersion is a function of the geometry alone and the caller that
    takes this reference's gradient adds its force back separately.
    """
    base, disp, _ = parse_disp(xc)
    out = xc if disp is None else base
    if not out:
        raise ValueError(
            f'could not read a functional out of {xc!r}: parse_disp gave '
            f'{(base, disp)}. The exchange-free reference cannot be built '
            f'without one, and pyscf fails far from here if it is handed None.')
    return out


def exchange_free_reference(mf):
    """`mf` at its own orbitals with the exact-exchange fraction taken out.

    Everything the ISDF route did not interpolate, in one pyscf mean field: the
    functional minus its a_x*HF term, density-fitted on the same auxiliary basis
    (which is where `j_route='df-direct'` takes J from) and carrying the
    converged coefficients, orbital energies and occupations, so that its
    gradient's one-electron, Coulomb, exchange-correlation and energy-weighted
    overlap terms are the ISDF route's own.

    The same grid objects as the mean field's: a Kohn-Sham energy is a property
    of its quadrature, and a finite difference taken against a different grid
    would be differencing two functionals.

    An empirical dispersion correction is left out of the name (see
    `_base_functional`); the caller adds its force back separately.
    """
    # Kohn-Sham iff the reference carries a functional, the same test
    # `mean_field_skeleton_force` makes below.
    is_ks = hasattr(mf, 'xc')
    # Hartree-Fock minus its exchange is the empty functional, which is what
    # `_NO_FUNCTIONAL` already is. A Kohn-Sham reference keeps its own
    # functional and has the exact-exchange fractions zeroed instead of
    # subtracted by name (`_NoExactExchange`).
    xc = _base_functional(mf.xc) if is_ks else _NO_FUNCTIONAL
    ref = dft.RKS(mf.mol, xc=xc).density_fit(auxbasis=mf.with_df.auxbasis)
    if is_ks:
        ref._numint = _NoExactExchange()
        ref.grids, ref.nlcgrids = mf.grids, mf.nlcgrids
    ref.mo_coeff, ref.mo_energy = mf.mo_coeff, mf.mo_energy
    ref.mo_occ, ref.converged = mf.mo_occ, True
    return ref


def mean_field_skeleton_force(mf):
    """dE/dR of the energy this mean field reported, whatever built its exchange.

    One call site for every correlated chain that adds a Lagrangian on top of
    the mean-field force. pyscf's gradient is right for a fitted reference and
    wrong for an interpolated one, and the correct interpolated force exists,
    so the choice is a dispatch and not a refusal.
    """
    # circular import: gradients.isdf_mean_field needs ISDFJK from this module
    from src.gradients.isdf_mean_field import isdf_mean_field_gradient

    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        force = np.asarray(isdf_mean_field_gradient(mf))
        # The ISDF force is assembled here, not by pyscf's gradient class, so
        # pyscf's PCM mixin never adds the reaction field's Hellmann-Feynman
        # part: the energy-weighted density already carries V_PCM through the
        # orbital energies, but dE_PCM/dR at fixed density does not come for
        # free. Without it the force is off by 1.0e-2 Ha/Bohr on water/cc-pVDZ
        # in water.
        with_solvent = getattr(mf, 'with_solvent', None)
        if with_solvent is not None:
            force = force + np.asarray(solvation_gradient(with_solvent,
                                                          mf.make_rdm1()))
        return force
    g0 = mf.Gradients()
    if hasattr(mf, 'xc'):
        g0.grid_response = True
    return np.asarray(g0.kernel())


def refuse_isdf_jk_gradient(mf, what):
    """Raise if `mf` answers its exchange from ISDF factors and a nuclear force
    is about to be taken from pyscf's own gradient.

    pyscf's density-fitted gradient differentiates the integrals of a fitted
    interaction and knows nothing of the interpolation points or the fit
    matrix this route builds K from, so it returns a gradient of a different
    function than the energy reported.

    The ground-state force itself is built elsewhere
    (`src.gradients.isdf_mean_field.isdf_mean_field_gradient`). What remains
    fitted, and why every caller of this guard is refused, is the Lagrangian a
    correlated chain adds on top: `fock_partial_skeleton` routes a
    density-fitted mean field to `fock_partial_skeleton_df`, whose exchange
    half is built from the auxiliary basis, and `exx_double_counting_skeleton`
    differences two fitted pyscf gradients. Energies are unaffected.
    """
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        return
    raise NotImplementedError(
        f'{what} takes a nuclear force from a mean field whose exchange comes '
        f'from ISDF factors, and pyscf\'s gradient does not know about them: '
        f'it differentiates the FITTED interaction rather than the '
        f'interpolated one this route actually built. Use '
        f'`src.gradients.isdf_mean_field.isdf_mean_field_gradient` for the SCF '
        f'force; a correlated (GW/BSE/dRPA) force on this reference is not '
        f'built at all.')
