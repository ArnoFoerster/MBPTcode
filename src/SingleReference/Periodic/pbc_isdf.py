"""Periodic ISDF / THC factorization on a uniform real-space grid: the Gamma
point, q-resolved k-point sampling, and the symmetry-adapted variant.

This is the Lu-Ying / Yeh-Morales line of ISDF, NOT the Duchemin-Blase
separable RI of `Base/separable_ri.py`. The two produce the same
factorization and share nothing else: there are no atom-centred grids, no
auxiliary basis, and no RI-V fit here.

    rho_ij(r) = phi_i^*(r) phi_j(r) ~= sum_mu rho_ij(r_mu) zeta_mu(r)
    (ij|kl)  ~= sum_{mu nu} X_{i mu}^* X_{j mu} V_{mu nu} X_{k nu}^* X_{l nu}
    X_{i mu} = phi_i(r_mu)                       plain collocation, no weight

Three steps, and the middle one is the reason the whole scheme is cubic:

  1. Collocate the orbitals on a uniform grid.
  2. Pick {r_mu} and fit {zeta_mu}. Both act on the pair-density Gram matrix
     S_{r r'} = sum_{ij} rho_ij^*(r) rho_ij(r'), which NEVER has to be formed
     as an N_r x N_r object, and never has to touch the N^2 pair index,
     because it factorizes into single-particle density matrices:

         S_{r r'} = A_a(r, r') * conj(A_b(r, r')),   A(r, r') = sum_i phi_i(r) phi_i^*(r')

     That identity is the whole ISDF trick. Point selection is a pivoted
     Cholesky on S (Yeh-Morales 2023 eq 11, a reformulation of QRCP that is
     cheaper and deterministic); the interpolating vectors are the
     least-squares solution of Z zeta = C with C = S[points, :] and
     Z = S[points, points].
  3. Solve Poisson for each zeta and take inner products -> V.

Step 2 is grid-agnostic black-box linear algebra; only steps 1 and 3 know what
the grid is (Zhu, Yeh, Morales et al., JCTC 2026, 22, 2904, Sec. 2). That is
why `coulomb_matrix` takes a `coulG_fn` callback rather than owning the
kernel: an adaptive-grid Poisson backend, a damped kernel, or a truncated
low-dimensional kernel all enter here and nowhere else.

WHY A UNIFORM GRID IS ADMISSIBLE HERE AND NOT IN THE MOLECULAR CODE.
It is admissible exactly to the extent that the basis is pseudized. Zhu et al.
Table 1: resolving an ALL-ELECTRON cc-pVTZ TiO to 1e-5 on a uniform grid needs
somewhere between 65536^3 and 131072^3 points, against 3.4e6 on an adaptive
octree. With an ECP the same molecule needs between 512^3 and 1024^3. The
periodic stack here uses gth-* pseudopotential bases, which is the regime the
FFT route is built for; an all-electron periodic ISDF would need the adaptive
Poisson solver, which is why `coulG_fn` is a seam and not a hard-coded FFT.
"""
import numpy as np
import scipy.linalg
from pyscf.pbc import tools
from pyscf.pbc.dft import numint

#: Ridge on the ISDF normal equations. The Gram matrix Z is Hermitian PSD by
#: construction but is deliberately driven towards rank deficiency -- more
#: interpolation points means a better fit AND a worse-conditioned Z -- so the
#: solve is regularized rather than assumed safe. Relative to trace(Z)/n.
DEFAULT_REGULARIZATION = 1e-10


def uniform_grid(cell, mesh=None):
    """(coords, weight) for the uniform FFT grid. weight is the scalar vol/N."""
    mesh = cell.mesh if mesh is None else mesh
    coords = cell.get_uniform_grids(mesh)
    return coords, cell.vol / len(coords)


def resolve_npoints(npoints, alpha, nmo):
    """The interpolation rank, given either an absolute count or `alpha`.

    The literature and every convergence study here talk in `alpha = N_mu /
    N_orb` (8-12 for chemical accuracy, ~16 before HF total energies settle),
    but the code only ever accepted an absolute `npoints`, so every caller
    wrote `ALPHA * nmo` by hand and the convention lived in test-file
    constants. Two consequences worth avoiding: the ratio that actually
    transfers between systems was not the thing being passed, and `alpha`
    silently meant alpha*nmo in the tests while memory sizing is written in
    alpha*nao. Identical while nothing is frozen; not under an orbital window.
    """
    if (npoints is None) == (alpha is None):
        raise ValueError("give npoints= or alpha=, not both and not neither "
                         "(alpha is the rank RATIO N_mu / n_orb).")
    if alpha is not None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        npoints = int(round(alpha * nmo))
    return int(npoints)


def resolve_grid(cell, mesh=None, ke_cutoff=None):
    """A cell whose `mesh` IS the grid this factorization will use.

    Two things this fixes, both of which bite hardest on a slab.

    FIRST, a bare `mesh=` override would leave `cell.mesh` stale: the grid
    would be overridden for `get_uniform_grids` and `tools.fft` while anything
    reading `cell.mesh` -- including pyscf internals -- still saw the old one.
    So the cell is rebuilt, as the DF builders do (`pbc_rpa`,
    `pbc_damped_integrals`).

    SECOND, `ke_cutoff` is exposed because the ISDF cost is LINEAR in N_r and
    pyscf's default mesh for a slab is set by the vacuum, not by the physics: a
    4-layer slab in 20 A of vacuum asks for ~1800x the point count of the
    equivalent bulk cell. Choosing the cutoff is therefore a first-order cost
    decision, not tuning, and it should be made explicitly rather than
    inherited.

    Pass one or the other; passing both is refused rather than silently
    resolved, since which one wins is exactly the kind of thing that is assumed
    wrongly.
    """
    if mesh is not None and ke_cutoff is not None:
        raise ValueError("give mesh= or ke_cutoff=, not both -- they are two "
                         "ways to say the same thing and the precedence would "
                         "have to be guessed.")
    if mesh is None and ke_cutoff is None:
        return cell, cell.mesh
    if ke_cutoff is not None:
        mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)
    out = cell.copy()
    out.mesh = list(mesh)
    out.build(False, False)
    return out, out.mesh


def check_isdf_low_dim(cell, kpts, coulG_fn=None):
    """Refuse the two ways a low-dimensional cell corrupts V^q in SILENCE.

    Neither of these raises on its own, and neither looks wrong downstream --
    which is the whole reason they are checked here rather than trusted.

    1. `low_dim_ft_type = 'inf_vacuum'`. Under it pyscf's `get_Gv_weights`
       returns a NON-UNIFORM Gauss-Chebyshev base along the vacuum axis, while
       `coulomb_matrix_q` FFTs on the uniform grid. The two G-orderings then no
       longer correspond, and because pyscf forces an even vacuum mesh the
       LENGTHS still match -- so `coulG[G]` is silently paired with the wrong
       `ahat[G]` and no shape error is ever raised. The DF builders survive
       this because they take Gv and the weights from the same
       `get_Gv_weights` call and never FFT. (This setting also breaks pyscf's
       own periodic SCF: E(H2 slab) = -3830 Ha against -0.84.) The slab route
       is dimension=2 + the default ft type + a damped kernel.

    2. A damped kernel whose real-space support does not fit the cell. That is
       `check_low_dim_support`, which the four DF builders call and this route
       -- a fifth consumer of the same `coulG_fn` seam -- did not.
    """
    from src.SingleReference.Periodic.pbc_rpa_damping import assert_damping_fits

    if getattr(cell, 'low_dim_ft_type', None) == 'inf_vacuum':
        raise ValueError(
            "cell.low_dim_ft_type='inf_vacuum' is not usable with the ISDF "
            "Coulomb build: get_Gv_weights then returns a non-uniform "
            "z-quadrature while coulomb_matrix_q FFTs on the uniform grid, so "
            "coulG is paired with the wrong ahat -- same length, wrong "
            "correspondence, no error. Use cell.dimension=2 with the default "
            "ft type and a damped kernel (pbc_rpa_damping.make_coulG_damped).")
    assert_damping_fits(cell, kpts, coulG_fn)


def collocation(cell, coords, mo_coeff, kpt=None):
    """phi_i(r) on the grid, shape (ngrid, nmo). No quadrature weight.

    The weight belongs to the Coulomb integrals in `coulomb_matrix`, not to
    the collocation matrix: X_{i mu} = phi_i(r_mu) is a point VALUE, and
    putting a weight on it would be absorbed into zeta by the fit and then
    double counted in V.
    """
    kpt = np.zeros(3) if kpt is None else np.asarray(kpt)
    ao = numint.eval_ao(cell, coords, kpt=kpt)
    return ao @ mo_coeff


def _density_matrix_column(Phi, p):
    """A(:, p) = sum_i phi_i(r) phi_i^*(r_p), shape (ngrid,). O(ngrid nmo)."""
    return Phi @ Phi[p].conj()


def _gram_diagonal(Phi_a, Phi_b):
    """diag(S) = rho_a(r) rho_b(r), real and non-negative."""
    return np.einsum('ri,ri->r', Phi_a, Phi_a.conj()).real * \
           np.einsum('ri,ri->r', Phi_b, Phi_b.conj()).real


def _gram_column(Phi_a, Phi_b, p):
    """S[:, p] = A_a(:, p) * conj(A_b(:, p))."""
    return _density_matrix_column(Phi_a, p) * _density_matrix_column(Phi_b, p).conj()


def select_points_cholesky(Phi_a, Phi_b=None, npoints=None, tol=1e-10):
    """Interpolation points as the first `npoints` pivots of a pivoted Cholesky
    on the pair-density Gram matrix S.

    Returns (points, residuals) with residuals[k] the Schur-complement
    diagonal at the moment pivot k was taken -- a monotone decreasing curve
    whose decay IS the numerical rank of the pair densities, so it is returned
    rather than discarded: it is the only cheap diagnostic of whether npoints
    was chosen sensibly.

    S is never formed. One column costs O(ngrid * nmo); the whole selection
    costs O(npoints^2 ngrid + npoints ngrid nmo).
    """
    Phi_b = Phi_a if Phi_b is None else Phi_b
    ngrid = Phi_a.shape[0]
    if npoints is None or npoints > ngrid:
        npoints = ngrid

    d = _gram_diagonal(Phi_a, Phi_b)
    R = np.zeros((npoints, ngrid), dtype=np.complex128)
    points = np.empty(npoints, dtype=int)
    residuals = np.empty(npoints)

    dmax0 = d.max()
    for k in range(npoints):
        p = int(np.argmax(d))
        residuals[k] = d[p]
        if d[p] <= tol * dmax0:
            points, residuals, R = points[:k], residuals[:k], R[:k]
            break
        points[k] = p
        col = _gram_column(Phi_a, Phi_b, p)
        if k:
            col = col - R[:k].conj().T @ R[:k, p]
        R[k] = col / np.sqrt(d[p])
        d = d - (R[k].real ** 2 + R[k].imag ** 2)
        np.maximum(d, 0.0, out=d)
    return points, residuals


def interpolating_vectors(Phi_a, Phi_b, points, regularization=DEFAULT_REGULARIZATION):
    """zeta, shape (npoints, ngrid): the least-squares solution of Z zeta = C.

    C = S[points, :], Z = S[points, points]. Hermitian PSD, solved by Cholesky
    with a relative ridge (see DEFAULT_REGULARIZATION).
    """
    Phi_b = Phi_a if Phi_b is None else Phi_b
    A_a = Phi_a[points] @ Phi_a.conj().T          # (npts, ngrid)
    A_b = Phi_b[points] @ Phi_b.conj().T
    C = A_a * A_b.conj()
    Z = C[:, points]

    if regularization:
        Z = Z + np.eye(len(points)) * (regularization * np.trace(Z).real / len(points))
    return scipy.linalg.solve(Z, C, assume_a='pos')


def coulomb_matrix(cell, zeta, weight, coulG_fn=None, kpt=None, mesh=None,
                   check_kernel=True):
    """V_{mu nu} = int int zeta_mu^*(r) v(r - r') zeta_nu(r').

    With pyscf's unweighted FFT convention (fft = sum_r f e^{-iGr},
    ifft = (1/N) sum_G) and coulG = 4 pi / |k+G|^2,

        V = (weight / N) sum_G coulG(G) zetahat_mu(G)^* zetahat_nu(G)

    ONE quadrature weight, not two. This is worth stating because two is the
    natural guess and it is wrong: a double integral over r and r' has two
    volume elements, but coulG multiplies the Fourier COEFFICIENT
    rho(G) = (1/vol) int dr rho(r) e^{-iGr}, whose 1/vol has already consumed
    the inner one. Equivalently, `ifft(coulG * fft(rho))` IS the potential
    v(r) with no weight applied -- which is why pyscf's `fft_jk.get_j_kpts`
    multiplies vR by vol/ngrids exactly once, for the outer integral, before
    contracting it with the AO pair.

    Getting this wrong is a pure scale error on every ERI element (measured:
    a factor of vol/N = 7.4e-4 on diamond/gth-szv), so it cannot hide behind
    a factorization error -- but it also cannot be caught by a normalization
    check on the collocation matrix, which stays exact to 8e-13 either way.
    """
    mesh = cell.mesh if mesh is None else mesh
    ngrid = np.prod(mesh)
    kpt = np.zeros(3) if kpt is None else np.asarray(kpt)
    if coulG_fn is None:
        Gv = cell.get_Gv(mesh)
        coulG = tools.get_coulG(cell, k=kpt, mesh=mesh, Gv=Gv)
    else:
        Gv = cell.get_Gv(mesh)
        coulG = coulG_fn(cell, kpt, Gv)
    if check_kernel:
        assert_kernel_admissible(coulG, kpt)

    zhat = tools.fft(np.ascontiguousarray(zeta), mesh)      # (npts, ngrid)
    return ((zhat.conj() * coulG) @ zhat.T) * (weight / ngrid)


def build_isdf_gamma(cell, mo_coeff, npoints=None, mesh=None, coulG_fn=None,
                     regularization=DEFAULT_REGULARIZATION, nmo_a=None,
                     ke_cutoff=None, check_kernel=True, alpha=None):
    """(X, V, info) for the Gamma-point THC factorization.

    X has shape (npoints, nmo) with X[mu, i] = phi_i(r_mu); V is (npoints,
    npoints). `nmo_a` restricts the FIRST orbital set of the pair densities
    (e.g. occupied only) while the second stays full -- the point-selection
    knob, off by default.
    """
    npoints = resolve_npoints(npoints, alpha, np.shape(mo_coeff)[1])
    cell, mesh = resolve_grid(cell, mesh, ke_cutoff)
    check_isdf_low_dim(cell, np.zeros((1, 3)), coulG_fn)
    coords, weight = uniform_grid(cell, mesh)
    Phi = collocation(cell, coords, mo_coeff)
    Phi_a = Phi[:, :nmo_a] if nmo_a else Phi

    points, residuals = select_points_cholesky(Phi_a, Phi, npoints)
    zeta = interpolating_vectors(Phi_a, Phi, points, regularization)
    V = coulomb_matrix(cell, zeta, weight, coulG_fn=coulG_fn, mesh=mesh,
                       check_kernel=check_kernel)

    nmo = np.shape(mo_coeff)[1]
    info = {'points': points, 'residuals': residuals, 'ngrid': len(coords),
            'weight': weight, 'mesh': mesh, 'npoints': npoints, 'nmo': nmo,
            'alpha': npoints / nmo}
    return Phi[points], V, info


def thc_eri(X, V):
    """(ij|kl) = sum_{mu nu} X_{mu i}^* X_{mu j} V_{mu nu} X_{nu k}^* X_{nu l}.

    Validation only -- it materializes the rank-4 tensor the factorization
    exists to avoid, and is O(npoints^2 nmo^2). `optimize=True` is not
    optional: without it numpy plans the three-operand contraction naively and
    the cost goes as npoints^2 nmo^4.
    """
    P = np.einsum('mi,mj->mij', X.conj(), X, optimize=True)
    return np.einsum('mij,mn,nkl->ijkl', P, V, P, optimize=True)


# ---------------------------------------------------------------------------
# q-resolved ISDF with k-point sampling
# ---------------------------------------------------------------------------
#
# The pair density carries crystal momentum. With pyscf's periodic AOs, which
# already include the Bloch phase, phi^k(r+T) = e^{ikT} phi^k(r), so
#
#     rho^q_{k,ij}(r) = phi^{(k-q)*}_i(r) phi^k_j(r)      obeys   rho^q(r+T) = e^{iqT} rho^q(r)
#
# and so does the interpolating vector zeta^q fitted to it. Everything else
# follows from that one fact:
#
#   * the FIT is unchanged except for a sum over k, because the Gram matrix
#     still factorizes into single-particle density matrices,
#
#         C^q[mu, r] = sum_k A^{k-q}(r_mu, r) * conj(A^k(r_mu, r))
#
#     with A^k(r, r') = sum_i phi^k_i(r) phi^{k*}_i(r'). Note the k-sum runs
#     over the FULL mesh for every q, which is what makes N_mu independent of
#     N_k rather than proportional to it;
#   * the COULOMB step must strip the Bloch phase before the FFT and use
#     coulG(q+G), since e^{-iqr} zeta^q is the lattice-periodic part whose
#     plane-wave coefficients sit at q+G.
#
# TIME REVERSAL IS NOT ASSUMED. The ERI needs the pair at momentum -q as well
# as +q, and zeta^{-q} = conj(zeta^q) EXACTLY -- not up to a gauge, and not
# only on a time-reversal-symmetric mesh. Relabelling k -> k+q in the k-sum,
# which is a bijection because the sum covers the whole mesh,
#
#     C^{-q}[mu,r] = sum_k A^{k+q} conj(A^k) = sum_k' A^{k'} conj(A^{k'-q})
#                  = conj( sum_k' A^{k'-q} conj(A^{k'}) ) = conj(C^q[mu,r])
#
# and Z^{-q} = conj(Z^q) likewise, so the (real-ridged) solve conjugates too.
# This matters: the alternative is to assume phi^{-k} = phi^{k*}, which holds
# only up to a rotation inside degenerate blocks and would have been a silent
# error at exactly the high-symmetry k-points where degeneracies live.


def kpoint_minus_map(cell, kpts, tol=1e-8):
    """kminus[q, k] = index of (kpts[k] - kpts[q]) folded back into the mesh.

    The inverse permutation of `get_momentum_transfer_map`'s rows.
    """
    from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
    kplus = get_momentum_transfer_map(cell, kpts, tol=tol)
    kminus = np.empty_like(kplus)
    for q in range(len(kpts)):
        kminus[q, kplus[q]] = np.arange(len(kpts))
    return kminus


def collocation_kpts(cell, coords, mo_coeff, kpts):
    """[phi^k_i(r)] per k-point, each (ngrid, nmo), complex."""
    return [np.asarray(numint.eval_ao(cell, coords, kpt=k)) @ c
            for k, c in zip(np.asarray(kpts), mo_coeff)]


def select_points_cholesky_kpts(Phi_list, npoints=None, tol=1e-10):
    """Interpolation points from the q=0 Gram matrix summed over k.

    S^0[r, r'] = sum_k |A^k(r, r')|^2, real and PSD. Same pivoted Cholesky as
    the Gamma case, with S never formed; one column costs O(N_k ngrid nmo).

    These points are then used at EVERY q. That is an empirical claim (Yeh &
    Morales 2023, Sec. 3.1), not a theorem, and it is the load-bearing one:
    q-dependent points would put a q index on X and destroy the separability
    of the k-indices, which is the entire reason for doing this. It is checked
    directly in tests/test_pbc_isdf_kpts.py.
    """
    ngrid = Phi_list[0].shape[0]
    if npoints is None or npoints > ngrid:
        npoints = ngrid

    d = np.zeros(ngrid)
    for Phi in Phi_list:
        d += np.einsum('ri,ri->r', Phi, Phi.conj(), optimize=True).real ** 2

    R = np.zeros((npoints, ngrid))
    points = np.empty(npoints, dtype=int)
    residuals = np.empty(npoints)
    dmax0 = d.max()
    for k in range(npoints):
        p = int(np.argmax(d))
        residuals[k] = d[p]
        if d[p] <= tol * dmax0:
            points, residuals, R = points[:k], residuals[:k], R[:k]
            break
        points[k] = p
        col = np.zeros(ngrid)
        for Phi in Phi_list:
            a = Phi @ Phi[p].conj()
            col += a.real ** 2 + a.imag ** 2
        if k:
            col = col - R[:k].T @ R[:k, p]
        R[k] = col / np.sqrt(d[p])
        d = np.maximum(d - R[k] ** 2, 0.0)
    return points, residuals


def interpolating_vectors_q(Phi_list, kminus, q, points,
                            regularization=DEFAULT_REGULARIZATION):
    """zeta^q on the grid, shape (npoints, ngrid), complex.

    Solves Z^q zeta = C^q with C^q[mu, r] = sum_k A^{k-q}(r_mu, r) conj(A^k(r_mu, r)).
    """
    npts = len(points)
    C = np.zeros((npts, Phi_list[0].shape[0]), dtype=np.complex128)
    for k, Phi_k in enumerate(Phi_list):
        Phi_kq = Phi_list[kminus[q, k]]
        C += (Phi_kq[points] @ Phi_kq.conj().T) * (Phi_k[points] @ Phi_k.conj().T).conj()
    Z = C[:, points]
    if regularization:
        Z = Z + np.eye(npts) * (regularization * np.trace(Z).real / npts)
    return scipy.linalg.solve(Z, C, assume_a='her')


def assert_kernel_admissible(coulG, q=None, tol=1e-10):
    """A negative Coulomb kernel makes V^q indefinite. Refuse it here.

    V^q = sum_G v(q+G) ahat_mu(G) conj(ahat_nu(G)) is a nonnegative combination
    of rank-1 PSD terms, so it is PSD **iff v >= 0 everywhere**. Checking the
    kernel is both the sharper diagnosis and the cheap one: eigendecomposing
    V^q would cost O(N_mu^3) per transfer to learn something O(N_r) of
    arithmetic already knows.

    The case this exists for: `tools.get_coulG` returns v(G=0) = -2 pi L_z^2
    for cell.dimension == 2 with the default ft type, and it is the LARGEST
    entry in magnitude. The DF route catches the same thing one step later, on
    the RI-V metric (`pbc_rpa.coulomb_metric_inv_sqrt`, which raises on an
    indefinite J). This route has no RI metric, so without this check the
    negative head propagates all the way into the RPA logdet, where the only
    guard is a determinant-sign test that fires on a symptom and only
    sometimes.
    """
    lo = float(np.min(coulG.real))
    if lo < -tol * max(float(np.max(coulG.real)), 1.0):
        where = 'at G = 0' if int(np.argmin(coulG.real)) == 0 else 'off G = 0'
        qs = '' if q is None else f' at q = {np.asarray(q).round(6).tolist()}'
        raise ValueError(
            f"Coulomb kernel is negative {where}{qs}: min v = {lo:.6g}. V^q "
            f"would be indefinite, and every consumer here assumes it is not. "
            f"For a 2D cell this is pyscf's v(G=0) = -pi L_z^2/2, which is a "
            f"DISCARDED DIVERGENCE rather than the kernel: it keeps the finite "
            f"part of the G_par -> 0 branch and drops the 2 pi L/q head. Two "
            f"fixes, and they want opposite things -- "
            f"(a) pbc_smallq.make_coulG_2d_head restores that head as its "
            f"mini-BZ average, needs the mesh FINE relative to the vacuum "
            f"(R < 8/L_z), and leaves the vacuum free; "
            f"(b) pbc_rpa_damping.make_coulG_damped damps the kernel instead, "
            f"needs the vacuum TALL relative to the mesh (its support must fit "
            f"in L_z/2), which chains L_z to the k-mesh. Pass either as "
            f"coulG_fn.")


def coulomb_matrix_q(cell, zeta, coords, weight, q, coulG_fn=None, mesh=None,
                     check_kernel=True):
    """V^q_{mu nu} = int dr dr' zeta^q_mu(r) v(r - r') zeta^{-q}_nu(r').

    This is the object the ERI contracts, so it is returned directly rather
    than the paper's `int zeta^* v zeta` -- see the module note: the two
    differ by a conjugation on the second index, and zeta^{-q} = conj(zeta^q)
    turns one into the other exactly.

        V^q = (weight / N) sum_G coulG(q+G) ahat_mu(G) conj(ahat_nu(G))
        ahat_mu = fft( e^{-i q.r} zeta^q_mu )

    Hermitian by construction. Reduces to `coulomb_matrix` at q = 0.
    """
    mesh = cell.mesh if mesh is None else mesh
    ngrid = np.prod(mesh)
    q = np.asarray(q)
    Gv = cell.get_Gv(mesh)
    coulG = (tools.get_coulG(cell, k=q, mesh=mesh, Gv=Gv) if coulG_fn is None
             else coulG_fn(cell, q, Gv))
    if check_kernel:
        assert_kernel_admissible(coulG, q)

    periodic = zeta * np.exp(-1j * (coords @ q))
    ahat = tools.fft(np.ascontiguousarray(periodic), mesh)
    return ((ahat * coulG) @ ahat.conj().T) * (weight / ngrid)


def build_isdf_kpts(cell, mo_coeff, kpts, npoints=None, mesh=None,
                    coulG_fn=None, regularization=DEFAULT_REGULARIZATION,
                    ke_cutoff=None, check_kernel=True, alpha=None):
    """(X, V, info) for the k-point THC factorization.

    X[k] is (npoints, nmo) with X[k][mu, i] = phi^k_i(r_mu) -- ONE k index and
    no q index, which is what makes the k-indices separable. V[q] is
    (npoints, npoints).

    `mesh` / `ke_cutoff` select the real-space grid (see `resolve_grid`); the
    low-dimensional guards are `check_isdf_low_dim` and, per transfer,
    `assert_kernel_admissible`.
    """
    ctx = isdf_prepare(cell, mo_coeff, kpts, npoints=npoints, alpha=alpha,
                       mesh=mesh, ke_cutoff=ke_cutoff, coulG_fn=coulG_fn)
    V = [isdf_vq(ctx, q, coulG_fn=coulG_fn, regularization=regularization,
                 check_kernel=check_kernel) for q in range(len(kpts))]
    return ctx['X'], V, ctx['info']


def isdf_prepare(cell, mo_coeff, kpts, npoints=None, alpha=None, mesh=None,
                 ke_cutoff=None, coulG_fn=None):
    """The q-INDEPENDENT half of the factorization: collocation and points.

    Split out from `build_isdf_kpts` so that `V^q` can be built ONE TRANSFER AT
    A TIME (`isdf_vq`). Two consumers need that and neither can use the all-q
    list:

      * the streaming RPA (`pbc_isdf_rpa.rpa_ecorr_thc_streaming`), where
        V for every q and, above all, Pi for every (q, omega) are the objects
        the whole memory discipline exists to avoid;
      * a distributed run, where a rank owns a subset of transfers and should
        build only its own.

    The context keeps `Phi`, the full (ngrid, nmo) collocation per k-point.
    That is the LARGEST object in the build -- (ngrid, nmo) per k-point
    against V^q's (M, M) -- and it cannot be avoided, because the fit at
    transfer q reads Phi at every k. Free the context once the V's are built
    if the consumer does not need more.
    """
    npoints = resolve_npoints(npoints, alpha, np.shape(mo_coeff[0])[1])
    cell, mesh = resolve_grid(cell, mesh, ke_cutoff)
    check_isdf_low_dim(cell, kpts, coulG_fn)
    coords, weight = uniform_grid(cell, mesh)
    Phi = collocation_kpts(cell, coords, mo_coeff, kpts)
    kminus = kpoint_minus_map(cell, kpts)
    points, residuals = select_points_cholesky_kpts(Phi, npoints)

    nmo = np.shape(mo_coeff[0])[1]
    info = {'points': points, 'residuals': residuals, 'kminus': kminus,
            'ngrid': len(coords), 'weight': weight, 'coords': coords,
            'mesh': mesh, 'npoints': npoints, 'nmo': nmo,
            'alpha': npoints / nmo}
    return {'cell': cell, 'kpts': np.asarray(kpts), 'Phi': Phi, 'coords': coords,
            'weight': weight, 'mesh': mesh, 'points': points, 'kminus': kminus,
            'X': [P[points] for P in Phi], 'info': info}


def isdf_vq(ctx, q, coulG_fn=None, regularization=DEFAULT_REGULARIZATION,
            check_kernel=True):
    """V^q for ONE transfer, from an `isdf_prepare` context. (npoints, npoints)."""
    zeta = interpolating_vectors_q(ctx['Phi'], ctx['kminus'], q, ctx['points'],
                                   regularization)
    return coulomb_matrix_q(ctx['cell'], zeta, ctx['coords'], ctx['weight'],
                            ctx['kpts'][q], coulG_fn=coulG_fn, mesh=ctx['mesh'],
                            check_kernel=check_kernel)


def build_isdf_kpts_per_q(cell, mo_coeff, kpts, npoints, mesh=None,
                          coulG_fn=None, regularization=DEFAULT_REGULARIZATION,
                          ke_cutoff=None, check_kernel=True):
    """DIAGNOSTIC ONLY: the same factorization with q-DEPENDENT points.

    Returns (X_by_q, V, info) where X_by_q[q][k] is (npoints_q, nmo). The
    extra q index on X is the whole reason this cannot be used in production:
    it is exactly the separability that the q=0-points assumption buys, and
    losing it would put back the O(N_k^2) the method exists to remove. This
    exists so the cost of that assumption can be MEASURED rather than
    believed.

    Note the per-q point counts need not agree with each other or with q=0:
    the pair densities at different transfers have different numerical ranks,
    which is itself worth seeing.
    """
    cell, mesh = resolve_grid(cell, mesh, ke_cutoff)
    check_isdf_low_dim(cell, kpts, coulG_fn)
    coords, weight = uniform_grid(cell, mesh)
    Phi = collocation_kpts(cell, coords, mo_coeff, kpts)
    kminus = kpoint_minus_map(cell, kpts)

    X_by_q, V, counts = [], [], []
    for q in range(len(kpts)):
        pts = (select_points_cholesky_kpts(Phi, npoints)[0] if q == 0
               else _select_points_q(Phi, kminus, q, npoints))
        zeta = interpolating_vectors_q(Phi, kminus, q, pts, regularization)
        X_by_q.append([P[pts] for P in Phi])
        V.append(coulomb_matrix_q(cell, zeta, coords, weight, kpts[q],
                                  coulG_fn=coulG_fn, mesh=mesh))
        counts.append(len(pts))
    return X_by_q, V, {'kminus': kminus, 'counts': counts, 'coords': coords,
                       'weight': weight,
                       'mesh': cell.mesh if mesh is None else mesh}


def _select_points_q(Phi_list, kminus, q, npoints, tol=1e-10):
    """Pivoted Cholesky on S^q for q != 0 -- diagnostic only (per_q_points)."""
    ngrid = Phi_list[0].shape[0]
    rho = [np.einsum('ri,ri->r', P, P.conj(), optimize=True).real for P in Phi_list]
    d = np.zeros(ngrid)
    for k in range(len(Phi_list)):
        d += rho[kminus[q, k]] * rho[k]
    R = np.zeros((npoints, ngrid), dtype=np.complex128)
    points = np.empty(npoints, dtype=int)
    dmax0 = d.max()
    for j in range(npoints):
        p = int(np.argmax(d))
        if d[p] <= tol * dmax0:
            points = points[:j]
            break
        points[j] = p
        col = np.zeros(ngrid, dtype=np.complex128)
        for k, Phi_k in enumerate(Phi_list):
            Phi_kq = Phi_list[kminus[q, k]]
            col += (Phi_kq @ Phi_kq[p].conj()) * (Phi_k @ Phi_k[p].conj()).conj()
        if j:
            col = col - R[:j].conj().T @ R[:j, p]
        R[j] = col / np.sqrt(d[p])
        d = np.maximum(d - (R[j].real ** 2 + R[j].imag ** 2), 0.0)
    return points


def thc_eri_kpts(X, V, kminus, k1, k2, k3, k4):
    """(i k1, j k2 | k k3, l k4) from the THC factors.

    q = k2 - k1 is the momentum of the first pair density; the second pair
    carries -q, and momentum conservation k1 - k2 + k3 - k4 = 0 is the
    caller's responsibility (the contraction is meaningless otherwise).
    """
    q = kminus[k1, k2]
    P12 = np.einsum('mi,mj->mij', X[k1].conj(), X[k2], optimize=True)
    P34 = np.einsum('mk,ml->mkl', X[k3].conj(), X[k4], optimize=True)
    return np.einsum('mij,mn,nkl->ijkl', P12, V[q], P34, optimize=True)


def thc_eri_kpts_per_q(X_by_q, V, kminus, k1, k2, k3, k4):
    """`thc_eri_kpts` for the q-dependent-points diagnostic."""
    q = kminus[k1, k2]
    Xq = X_by_q[q]
    P12 = np.einsum('mi,mj->mij', Xq[k1].conj(), Xq[k2], optimize=True)
    P34 = np.einsum('mk,ml->mkl', Xq[k3].conj(), Xq[k4], optimize=True)
    return np.einsum('mij,mn,nkl->ijkl', P12, V[q], P34, optimize=True)


# ---------------------------------------------------------------------------
# Symmetry-adapted ISDF (Yeh & Morales, JCTC 2024, 20, 3184, Sec. 4)
# ---------------------------------------------------------------------------
#
# The interpolating vectors zeta^q are fitted only for q in the IRREDUCIBLE
# wedge. For a general q, pick S with qbar = S q inside the wedge; then
#
#     phi^{k-q*}_j(r) phi^k_i(r) = sum_mu phi^{k-q*}_j(S r_mu) phi^k_i(S r_mu)
#                                         zeta^qbar_mu(S^-1 r)
#
# (their eq 11c), so the ERI keeps its THC form with the collocation matrix
# evaluated at ROTATED interpolation points:
#
#     X^k_{i mu}(S) = phi^k_i(S r_mu)
#
# The saving is on the expensive step -- the least-squares fit and the Coulomb
# matrix, both O(N_mu^2 N_r), run |IBZ_q| times instead of N_q. The added cost
# is nop x nk collocations at N_mu points, which is negligible against N_r.
#
# WHAT IS NOT DETERMINED BY ALGEBRA. For a point group {R} and {R^T} are the
# same SET, so "is the rotation R or R^T" has no answer in isolation -- both
# choices are valid group elements, merely relabelled. What IS determined is
# the PAIRING: the op used to map q -> qbar and the op used to rotate r_mu must
# be consistent with each other. That pairing is fixed here by measurement
# against the already-validated non-symmetric route, not by reading a
# convention off the paper.
#
# A TRAP. Eq 9 -- that MF orbitals at symmetry-related k-points are
# related by a d-matrix -- requires WHOLE DEGENERATE SETS in the orbital list.
# An energy window that cuts through a degenerate shell breaks it silently.
# `check_degenerate_blocks_intact` is the guard; it is not optional the moment
# anyone truncates.


def cartesian_rotations(cell):
    """(nop, 3, 3) Cartesian rotations of the cell's space group.

    pyscf gives `op.rot` in the lattice basis, acting on fractional
    coordinates. With A the matrix whose ROWS are the lattice vectors,
    x_cart = A^T f, so a fractional rotation R becomes A^T R A^-T in Cartesian.
    Orthogonality of the result is asserted rather than assumed -- it is the
    cheapest check that the basis conversion is right, and it fails loudly if
    pyscf's convention ever moves.
    """
    from pyscf.pbc.symm import Symmetry

    sym = Symmetry(cell).build()
    if not sym.symmorphic:
        raise NotImplementedError(
            "symmetry-adapted ISDF here assumes a symmorphic space group "
            "(zero fractional translations); this cell is non-symmorphic. The "
            "extension is a phase factor on the rotated collocation, but it is "
            "not implemented and would be silently wrong if ignored.")
    AT = cell.lattice_vectors().T
    ATinv = np.linalg.inv(AT)
    R = np.array([AT @ np.asarray(op.rot, dtype=float) @ ATinv for op in sym.ops])
    err = np.abs(np.einsum('sij,skj->sik', R, R) - np.eye(3)).max()
    if err > 1e-8:
        raise RuntimeError(f"Cartesian rotations are not orthogonal (max "
                           f"deviation {err:.2e}) -- the lattice-basis "
                           f"conversion is wrong.")
    return R


def mesh_invariant_operations(cell, kpts, R_cart, tol=1e-8):
    """Indices of the operations that map the k-MESH onto itself.

    NOT the same as the space group, and using the space group instead is a
    silent accuracy floor rather than an error. The Gram matrix that defines
    zeta is a sum over the whole mesh,

        C^q = sum_k A^{k-q} conj(A^k),

    so under S it becomes a sum over the ROTATED mesh S{k}. Unless S{k} = {k}
    the two Gram matrices are not related at all and the symmetry relation
    zeta^q(r) = zeta^qbar(S^-1 r) is simply false -- the fit still runs, still
    looks reasonable, and stops converging.

    Measured on diamond: with the full 24-operation cubic group applied to a
    2x2x1 mesh (which does NOT have cubic symmetry -- the mesh is coarser in
    one direction than the lattice), the symmetry-adapted ERI plateaus at
    1.7e-2 relative while the plain route reaches 3.8e-7. Restricted to the
    invariant subgroup it tracks the plain route instead.

    Checking that each q maps INTO the mesh is not sufficient and was the
    original bug here: individual momenta can land on mesh points while the
    mesh as a whole is not invariant.
    """
    scaled = cell.get_scaled_kpts(kpts)
    scaled = scaled - scaled[0]
    binv = np.linalg.inv(cell.reciprocal_vectors())
    keep = []
    for s, R in enumerate(R_cart):
        rot = (kpts @ R.T) @ binv
        hit = np.zeros(len(kpts), dtype=bool)
        for i, r in enumerate(rot):
            d = scaled - r
            m = np.where(np.linalg.norm(np.round(d) - d, axis=1) < tol)[0]
            if len(m) == 1:
                hit[m[0]] = True
        if hit.all():
            keep.append(s)
    return np.asarray(keep)


def q_star_map(cell, kpts, R_cart, tol=1e-8):
    """(ibz, rep, op) for the transfer mesh.

    ibz : indices of the irreducible transfers, ascending.
    rep[q] : position within `ibz` of the representative of q.
    op[q]  : index s with R_cart[s] mapping kpts[q] onto that representative.
    """
    b = cell.reciprocal_vectors()
    binv = np.linalg.inv(b)
    scaled = cell.get_scaled_kpts(kpts)
    scaled = scaled - scaled[0]
    nq = len(kpts)

    rep = -np.ones(nq, dtype=int)
    op = -np.ones(nq, dtype=int)
    ibz = []
    for q in range(nq):
        if rep[q] >= 0:
            continue
        here = len(ibz)
        ibz.append(q)
        rep[q], op[q] = here, 0
        for s, R in enumerate(R_cart):
            rot = (R @ kpts[q]) @ binv                  # scaled coordinates
            d = scaled - rot
            hit = np.where(np.linalg.norm(np.round(d) - d, axis=1) < tol)[0]
            if len(hit) == 1 and rep[hit[0]] < 0:
                rep[hit[0]], op[hit[0]] = here, s
    if (rep < 0).any():
        raise RuntimeError("q-star map left transfers unassigned; the k-mesh "
                           "is not closed under the space group.")
    return np.asarray(ibz), rep, op


def check_degenerate_blocks_intact(mo_energy, nmo_kept=None, tol=1e-6):
    """Refuse an orbital list that splits a degenerate shell.

    Symmetry-adapted ISDF rests on eq 9 -- orbitals at symmetry-related
    k-points related by a d-matrix -- which holds only for COMPLETE degenerate
    sets. Truncating mid-shell breaks it silently, and degeneracies live at
    exactly the high-symmetry k-points the IBZ reduction leans on.
    """
    e = np.asarray(mo_energy)
    n = e.shape[1] if nmo_kept is None else nmo_kept
    if n >= e.shape[1]:
        return
    for k in range(e.shape[0]):
        if abs(e[k][n] - e[k][n - 1]) < tol:
            raise ValueError(
                f"orbital window of {n} splits a degenerate shell at k-point "
                f"{k} (eps[{n-1}]={e[k][n-1]:.8f}, eps[{n}]={e[k][n]:.8f}). "
                f"Symmetry-adapted ISDF needs whole degenerate sets; widen or "
                f"narrow the window to a shell boundary.")


def build_isdf_kpts_symm(cell, mo_coeff, kpts, npoints, mesh=None,
                         coulG_fn=None, regularization=DEFAULT_REGULARIZATION,
                         ke_cutoff=None, check_kernel=True):
    """(X_by_op, V_ibz, info) -- the factorization with the IBZ reduction.

    X_by_op[s][k] is (npoints, nmo) with entries phi^k_i(R_s r_mu).
    V_ibz[j] is the Coulomb matrix at the j-th irreducible transfer.
    """
    cell, mesh = resolve_grid(cell, mesh, ke_cutoff)
    check_isdf_low_dim(cell, kpts, coulG_fn)
    coords, weight = uniform_grid(cell, mesh)
    Phi = collocation_kpts(cell, coords, mo_coeff, kpts)
    kminus = kpoint_minus_map(cell, kpts)
    points, residuals = select_points_cholesky_kpts(Phi, npoints)
    r_mu = coords[points]

    R_all = cartesian_rotations(cell)
    keep = mesh_invariant_operations(cell, kpts, R_all)
    R_cart = R_all[keep]
    ibz, rep, op = q_star_map(cell, kpts, R_cart)

    V_ibz = []
    for q in ibz:
        zeta = interpolating_vectors_q(Phi, kminus, q, points, regularization)
        V_ibz.append(coulomb_matrix_q(cell, zeta, coords, weight, kpts[q],
                                      coulG_fn=coulG_fn, mesh=mesh))

    X_by_op = []
    for R in R_cart:
        rot = r_mu @ R.T
        X_by_op.append(collocation_kpts(cell, rot, mo_coeff, kpts))

    info = {'points': points, 'residuals': residuals, 'kminus': kminus,
            'ibz': ibz, 'rep': rep, 'op': op, 'R_cart': R_cart,
            'ngrid': len(coords), 'weight': weight, 'coords': coords,
            'nq_solved': len(ibz), 'nq_total': len(kpts),
            'nop_group': len(R_all), 'nop_mesh_invariant': len(keep),
            'mesh': cell.mesh if mesh is None else mesh}
    return X_by_op, V_ibz, info


def thc_eri_kpts_symm(X_by_op, V_ibz, info, k1, k2, k3, k4):
    """(i k1, j k2 | k k3, l k4) from the symmetry-adapted factors."""
    q = info['kminus'][k1, k2]
    s = info['op'][q]
    Xs = X_by_op[s]
    P12 = np.einsum('mi,mj->mij', Xs[k1].conj(), Xs[k2], optimize=True)
    P34 = np.einsum('mk,ml->mkl', Xs[k3].conj(), Xs[k4], optimize=True)
    return np.einsum('mij,mn,nkl->ijkl', P12, V_ibz[info['rep'][q]], P34,
                     optimize=True)
