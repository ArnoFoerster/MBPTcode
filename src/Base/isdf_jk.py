"""ISDF (separable-RI) J and K for the SCF, in place of a density-fitted cderi.

WHY THIS EXISTS
---------------
pyscf's DF builds `cderi`, naux x nao(nao+1)/2, and streams it once per SCF
iteration. For the chlorophyllide-a dimer (178 atoms) at cc-pVTZ that is
nao = 4068, naux = 10420 -> 690 GB; and a range-separated hybrid such as
LRC-wPBEh builds a SECOND one for the erf-attenuated operator (see
`pyscf/df/df.py::range_coulomb`, which caches an entire parallel DF object per
omega), so 1.38 TB of node-local scratch, read every iteration.

The GW/BSE half of this pipeline already runs on the separable RI of Duchemin
and Blase (`src/Base/separable_ri.py`), where the same system costs 1.4 GB for
X and 3.6 GB for D. The SCF was the only consumer of the dense cderi left.

THE SAME FACTORIZATION, CONTRACTED FOR J AND K
----------------------------------------------
    (mu nu | la si) ~= sum_{PQ} X_{P mu} X_{P nu} Z_{PQ} X_{Q la} X_{Q si}

with X[P, mu] = chi_mu(r_P) on the interpolation grid and Z = M^T V M. Both
Coulomb matrices follow by contraction, with no three-index tensor anywhere:

    rho_Q = [X Dm X^T]_QQ                    diagonal only
    J     = X^T diag(Z rho) X                                    O(M nao^2)
    K     = X^T [ Z .* (X Dm X^T) ] X        .* elementwise      O(M^2 nao)

K is the expensive one and its cost is 4 M^2 nao, or 3 with the hermitian
shortcut -- the M^2-nao contraction appears twice, once forming the Hadamard
argument and once contracting it back. At the dimer/cc-pVTZ, M = 43076, that is
25.5 Tflop against DF-K's 112, and the largest object ever formed is a block of
rows of an (M, M) matrix.

AND ONLY K IS USABLE, WHICH IS THE POINT OF THIS MODULE
-------------------------------------------------------
The two contractions do not ask the interpolation for the same thing. In K,
mu and nu sit on DIFFERENT interpolation points, each paired there with the
density -- a sum over occupied orbitals, therefore compact. In J they sit on the
SAME point as a co-density product X_{P mu} X_{P nu}, an arbitrary AO pair,
diffuse x diffuse included. `separable_ri.build_D_F` fits M against test
co-densities {all AOs} x {AOs with l <= l_max_second}, and `l_max_second` is 2,
so an f x f product is never in the test set at all.

Measured, benzene/cc-pVTZ on the published grids: the worst |dJ| element is a
carbon 4f with itself at 2.4e-1, against 1.1e-3 for the f block of K -- 222x --
and the SCF collapses by 129 eV. Widening the test set to l <= 3 takes that
element to 2.9e-2 and leaves K alone, confirming the mechanism, and the SCF then
collapses by 142 eV instead: the failure is variational, not one bad element.
So `j_route` defaults to 'df-direct': J from
pyscf's integral-direct DF-J, which stores no three-index tensor either.
ISDF-K on that footing lands within 17 meV of DF-K on the same auxiliary
basis, the size of the RI error it sits on top of.

WHAT IS STORED
--------------
Z itself is (M, M): 14.9 GB at the dimer. That fits in a node, unlike 690 GB,
so `z_mode='dense'` holds it. `z_mode='factored'` instead keeps L = V^{1/2} M,
shape (naux, M) = 3.6 GB, and rebuilds Z's rows per block. Measured on
benzene and naphthalene at cc-pVTZ, that is 1.6x the time for 2.4x less
resident memory -- and the ratio moves with naux/nao, so it widens as the
auxiliary basis does. `z_mode='auto'` picks from max_memory.

RANGE SEPARATION
----------------
The ISDF ansatz approximates the CO-DENSITY, rho_{mu nu}(r) ~= sum_P X_{P mu}
X_{P nu} xi_P(r), and the operator only enters through Z_PQ = (xi_P | w | xi_Q).
So a second operator needs a second Z and NOTHING ELSE: same grid, same X, same
fitting coefficients M, with V replaced by the attenuated two-centre metric.

It is also EXACT in the metric: Z is linear in V and V_SR + V_LR = V, so
K_SR(omega) + K_LR(omega) = K_bare to machine precision (9.8e-15, see
tests/test_isdf_jk.py). A refitted M_omega would forfeit that identity.

The residual worry is that M is fitted against the BARE Coulomb metric
(`separable_ri.fit_error_coulomb` is the objective the grid radii were optimized
against), so it is only minimized in that norm. `refit_omega=True` builds an
independent M_omega from attenuated three-centre integrals for comparison; the
measurement says the refit buys 3-4x on water and NOTHING on benzene (1.778e-2
either way), for a second full factorization. Reuse is also the numerically
safer route: the attenuated metric is singular where the bare one is merely
ill-conditioned -- 214 of 420 auxiliary functions in the null space on
benzene/cc-pVDZ-RI at omega = 0.2 -- and Z_w = M^T V_w M only ever multiplies
BY it, while refitting has to invert it.

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

import numpy as np
import scipy.linalg
from pyscf import df, gto, lib, scf
from pyscf.lib import logger

from src.Base.separable_ri import (ANGULAR_WEIGHTS, DEFAULT_REGULARIZATION,
                                   _ao_l_labels, build_D_F,
                                   fit_M_stable, fit_M_streaming,
                                   molecular_points_covariant,
                                   optimize_atomic_radii, published_grids)

#: `space_time.separable_factors`' grid, so a J/K built here and a GW run share
#: one factorization when the caller wants that. 148 points per atom.
DEFAULT_COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}

#: Eigenvalue floor for V^{1/2}. The attenuated metric is far more
#: rank-deficient than the bare one -- erf(omega r)/r is smooth, so its
#: two-centre matrix decays fast in the aux basis -- and a relative floor keeps
#: the square root real without discarding anything that carries weight.
_METRIC_EIG_TOL = 1e-12


@contextlib.contextmanager
def range_coulomb(mol, auxmol, omega):
    """`omega` on BOTH molecules, the way `pyscf/df/df.py::range_coulomb` does.

    omega > 0 is the long-range erf(omega r)/r, omega < 0 the short-range
    complement, 0 or None the bare operator -- pyscf's convention, which
    `dft/rks.py::get_veff` relies on when it asks for `omega=-omega`.

    Both molecules, because a three-centre integral is driven by a concatenated
    Mole whose _env inherits the FIRST argument's range-omega slot; setting only
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


def isdf_grid(mol, counts=None, radii=None, auxbasis=None, n_start=1):
    """Interpolation points for `mol`, the same way `space_time` picks them.

    Published Duchemin-Blase tables where they apply (H/C/N/O at cc-pVTZ),
    `optimize_atomic_radii` otherwise -- which is cached on disk per
    (element, basis, auxbasis, counts, n_start). Shells are rotated into
    covariant atomic frames so the grid rotates with the molecule.

    n_start: descents per element in `optimize_atomic_radii`. The single
        default descent is the worst of the starting shapes on carbon; six cut
        benzene's exchange-energy error 23x at the same point count.
    """
    # The published tables come at ONE size, so they can only be substituted for
    # a caller who did not ask for a size. Asking for `counts` and silently
    # getting the published grid instead makes a grid-convergence study return
    # the same number for every count, which reads as convergence.
    asked_for_counts = counts is not None
    counts = counts or DEFAULT_COUNTS
    auxbasis = auxbasis or (str(mol.basis) + '-ri')
    if radii is None:
        pub = published_grids()
        radii, origins = {}, {}
        for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
            if el in pub and str(mol.basis).lower() == 'cc-pvtz' \
                    and not asked_for_counts:
                radii[el], origins[el] = pub[el]
            else:
                radii[el] = optimize_atomic_radii(el, mol.basis, auxbasis,
                                                  counts=counts,
                                                  n_start=n_start)[0]
                origins[el] = False
    else:
        origins = {el: False for el in radii}
    return molecular_points_covariant(mol, radii, origin_by_element=origins)


def metric_sqrt(V, tol=_METRIC_EIG_TOL):
    """V^{1/2} through a truncated eigendecomposition.

    V is a Coulomb (or attenuated Coulomb) two-centre matrix, positive
    semidefinite by construction, but the attenuated one is numerically
    singular -- so a Cholesky is not available and the small modes are dropped.
    """
    w, v = np.linalg.eigh(V)
    keep = w > tol * max(w.max(), 1e-300)
    return (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T


def fit_M_omega(mol, auxmol, coords, omega, l_max_second=2,
                regularization=DEFAULT_REGULARIZATION, metric_rcond=1e-10):
    """M refitted against the ATTENUATED metric -- the "second factorization".

    Same least-squares estimator as `separable_ri.fit_M`, but the RI fitting
    coefficients it targets are taken with the range-separated operator:

        F^w_{beta,rho} = sum_gamma [V_w^{-1}]_{beta gamma} (gamma | w | rho)

    A PSEUDO-inverse, not an LU solve. The attenuated two-centre metric is
    numerically singular where the bare one is merely ill-conditioned -- on
    water/cc-pVDZ-RI, cond(V) = 2.5e5 against cond(V_w) = 2.5e18 at
    omega = 0.3, with 14 of 84 eigenvalues at the level of the smallest
    negative one. `separable_ri.build_D_F`'s `lu_solve` on that returns noise,
    and a DF reconstruction of the attenuated ERI built the same way is wrong
    in the third significant figure (1.6e-1 absolute) purely from the inverse.
    That fragility is the refit route's, not the reuse route's: Z_w = M^T V_w M
    only ever multiplies BY the metric.
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
    # the identity only on the retained modes -- so it is written out rather
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

    Only the DIAGONAL of X Dm X^T is needed, so J never touches an (M, M)
    object at all -- it is O(M nao^2) and would run happily on a laptop at the
    dimer. The whole memory question is K's.
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
    cheaper: Z's ELEMENTS are needed, not its action. What blocking buys is
    that only `block` rows of it exist at a time -- 177 MB at the dimer with
    block = 512, against 14.9 GB for the whole thing.

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
            # asking for them would hand back exactly the rebuild work the
            # triangular loop exists to avoid.
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
    rather than let a caller silently fall back onto the 690 GB path.
    """

    def __init__(self, mol, auxbasis=None, counts=None, radii=None,
                 z_mode='auto', block=None, refit_omega=False,
                 use_symmetry=True, j_route='df-direct', check_tol=1e-3,
                 l_max_second=2, regularization=DEFAULT_REGULARIZATION,
                 block_memory_gb=4.0, progress=None, n_start=1):
        super().__init__(mol, auxbasis=auxbasis or (str(mol.basis) + '-ri'))
        # Caps the working set of the fit's blocked loops. It reaches here
        # because the fit is where the peak is: the Gram matrix is n_k^2, 14.9
        # GB at the dimer/cc-pVTZ, and everything else in the build is small
        # beside it. It does not change the answer, but it IS a speed knob:
        # one aux_e2 call per block, each rebuilding a shell-pair list over
        # nbas x auxnbas -- see `separable_ri.build_D_F` for the measurement.
        self.block_memory_gb = block_memory_gb
        # None follows mol.verbose, so the knob that turns on the mean field's
        # output turns on the factorization's too. It is minutes to hours here
        # and was previously silent throughout.
        self.progress = (mol.verbose > 0) if progress is None else progress
        self.counts = counts or DEFAULT_COUNTS
        self.radii = radii
        self.n_start = n_start
        self.z_mode = z_mode
        self.block = block
        self.refit_omega = refit_omega
        self.use_symmetry = use_symmetry
        # j_route: 'isdf' is the pure method and what the validation sweep
        # measures; 'df-direct' is the DEFAULT because the measurements say so.
        # It takes J from pyscf's INTEGRAL-DIRECT DF-J (`df_jk.get_j`), which
        # stores nothing but the (naux, naux) metric -- so the no-cderi
        # property is untouched -- and it removes the entire failure mode.
        # Benzene/cc-pVDZ on the fallback grid: -128810 meV with ISDF J,
        # -225 meV with DF-J, IDENTICAL ISDF exchange in both. On grids that
        # are fine it still buys an order of magnitude on orbital energies
        # (water/PBE0 HOMO +25.7 -> -0.6 meV). J is also the cheap term --
        # O(M nao^2) here against K's O(M^2 nao) -- so the trade is not
        # symmetric.
        self.j_route = j_route
        self._jdf = {}
        # Relative Coulomb-energy tolerance for the one-off `check`, run on the
        # first density the SCF asks about. A bad interpolation grid is not a
        # hypothetical: benzene/cc-pVDZ on `optimize_atomic_radii`'s default
        # local descent puts 0.31 Ha of error into a single J element and
        # collapses the SCF by 129 eV, while K stays good to 5e-3. Silent is
        # the wrong failure mode for that. None disables the check.
        self.check_tol = check_tol
        self._checked = False
        self.grid_warning = None
        # Angular momentum cutoff on the SECOND index of the test co-densities
        # M is fitted against (`separable_ri.build_D_F`). The default of 2 is
        # the published scheme's and is right for K, whose co-densities always
        # carry an occupied orbital. It is exactly why J fails: an f x f AO
        # product is never in the test set. Raise it to fit what J needs, at a
        # larger test set and a proportionally longer fit.
        self.l_max_second = l_max_second
        self.regularization = regularization

        self.coords = None
        self.X = None            # (nk, nao)
        self.M = None            # (naux, nk)
        self._z = {}             # omega key -> _ZRows
        self._built = False

    # -- construction --------------------------------------------------------

    @property
    def nk(self):
        return 0 if self.coords is None else len(self.coords)

    def build(self):
        if self._built:
            return self
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        if self.auxmol is None:
            self.auxmol = df.addons.make_auxmol(mol, auxbasis=self.auxbasis)
        if self.coords is None:
            self.coords = isdf_grid(mol, counts=self.counts, radii=self.radii,
                                    auxbasis=self.auxbasis, n_start=self.n_start)
        self.X = mol.eval_gto('GTOval_sph', self.coords)
        self.M = fit_M_streaming(mol, self.auxmol, self.coords,
                                 l_max_second=self.l_max_second,
                                 regularization=self.regularization,
                                 block_memory_gb=self.block_memory_gb,
                                 progress=self.progress)
        log.timer('ISDF factorization (M = %d points, nao = %d, naux = %d)'
                  % (self.nk, mol.nao_nr(), self.auxmol.nao_nr()), *t0)
        if self.j_route == 'isdf':
            # Not a tolerance question. On benzene/cc-pVDZ, re-optimizing the
            # radii with basin hopping takes the probe error from 8.5e-3 to
            # 3.3e-4 -- inside any reasonable tolerance -- and the SCF still
            # collapses by 5.6 eV, because the variational optimization seeks
            # out the modes where the interpolated Coulomb operator is weak.
            # There is no probe value at which j_route='isdf' becomes safe.
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
        free = (self.max_memory - lib.current_memory()[0]) * 1e6
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

        # Resolved ONCE: it reads current_memory(), which the dense branch
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
        free = max((self.max_memory - lib.current_memory()[0]) * 1e6, 2e8)
        # A row block costs b*nk (the Hadamard argument) and, in factored mode,
        # b*nk again for the rebuilt Z rows.
        b = int(0.25 * free / (2 * nk * 8))
        return int(np.clip(b, 16, nk))

    def probe_densities(self):
        """Densities to test the interpolation against, cheapest first.

        NOT the SCF's own density. The failure this catches is a J operator
        with bad modes that the SCF then VARIATIONALLY FINDS, so testing at a
        physical density misses it: on benzene/cc-pVDZ with the fallback grid,
        the exact HF density gives a Coulomb error of 3.5e-4 while the SCF
        still collapses by 129 eV. The core-Hamiltonian guess -- far too
        diffuse to be physical, which is the point -- gives 8.5e-3 there and
        4.8e-5 on the published grid, a factor of 175 apart.
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
        either -- two screened integral passes and the (naux, naux) metric --
        so this costs about one J build per probe and is affordable once per
        SCF even at the sizes this method exists for.

        J is the right probe rather than K, because J is what reads the part
        of the fit nobody constrained -- co-densities of two arbitrary AOs,
        including the high-l products that `l_max_second` leaves out of the
        test set entirely. In every case measured here a grid bad enough to
        matter shows up in J one to two orders of magnitude before it shows up
        in K: benzene/cc-pVDZ on the fallback grid puts 0.31 Ha into a single J
        element while K stays good to 5e-3. It is also the cheap term.

        A pass is necessary, not sufficient. It says the grid is not obviously
        broken; it does NOT license `j_route='isdf'`, which fails variationally
        at probe values far below any tolerance. See `build`.
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

        `check` probes J because a broken grid shows there first -- but J is
        protected by `j_route='df-direct'`, so on the production route the ONLY
        term the grid touches is K, and a K error too small to trip the J probe
        is not small in absolute terms: 1e-3 relative on a 178-atom molecule is
        0.12 Ha in the SCF energy, i.e. 30 meV on every BSE root. This is the
        instrument for that. It returns (max |dE_x| in Ha, max relative), with
        E_x = -1/4 tr(D K) for a closed-shell D.

        The reference builds a three-index tensor, naux x nao^2 / 2, so this is a
        SMALL-MOLECULE calibration tool for choosing grids, not a check to run
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
                            'over %d atoms (%d per atom). Grids for elements or '
                            'bases outside published_grids() come from '
                            'optimize_atomic_radii: raise the point COUNT '
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
        with only the (naux, naux) metric held. So this keeps the whole point
        of the exercise -- no three-index tensor on disk or in core -- while
        taking J from the RI-V fit rather than the interpolation.
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
            'pyscf.df.DF instead of silently materializing 690 GB.')

    @property
    def _cderi(self):
        raise NotImplementedError('ISDFJK stores no cderi; see loop().')

    @_cderi.setter
    def _cderi(self, x):
        if x is not None:
            raise NotImplementedError('ISDFJK stores no cderi; see loop().')


def separable_factors_from_jk(mf):
    """(X_mo, D, X_ao, coords) for GW/BSE, from an SCF that already fitted.

    `space_time.separable_factors` and `ISDFJK.build` construct the SAME
    factorization -- `isdf_grid` and the grid block in `separable_factors` are
    the same published tables, the same `optimize_atomic_radii` fallback and the
    same covariant frames, and `metric_sqrt`'s tolerance is `separable_factors`'
    1e-12 -- so a BSE run on top of an ISDF SCF was building it twice. At the
    chlorophyllide dimer/cc-pVTZ the second build cost 7772 s, on top of the
    6166 s the SCF had already spent.

    Returns the same pair `separable_factors` would, in the same auxiliary
    gauge, which is the part that must not drift: pairing factors from one fit
    with a W from another is the silent gauge error this project has paid for
    before.
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
    installed exactly as usual, then swaps the DF object underneath it.
    """
    out = mf.density_fit(auxbasis=auxbasis or (str(mf.mol.basis) + '-ri'))
    out.with_df = ISDFJK(mf.mol, auxbasis=auxbasis, counts=counts, radii=radii,
                         z_mode=z_mode, block=block, refit_omega=refit_omega,
                         use_symmetry=use_symmetry, j_route=j_route,
                         check_tol=check_tol, l_max_second=l_max_second,
                         block_memory_gb=block_memory_gb, progress=progress,
                         n_start=n_start)
    out.with_df.max_memory = mf.max_memory
    return out
