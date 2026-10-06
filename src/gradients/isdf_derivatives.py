"""Nuclear derivatives of the ISDF factors, as adjoint contractions.

The space-time route returns sensitivities with respect to the factors,
(eps_bar, X_bar, D_bar), and this module turns the X and D parts into forces.
The derivative tensor dX/dR is never formed, only its contraction against a
given adjoint.

On the row-distributed fit both parts run in the fit's own tiles
(`row_fit_adjoint` over `separable_ri.fit_rows_adjoints`, and X_mo^T X_bar by
`orbital_rotation_rows`), so no rank forms an array of the grid by a factor's
or the fit's width, and the result is the same bits at every rank count. The
whole forms (`dfactor_adjoint_gauges`, `collocation_adjoint`) serve the
replicated fit; the whole fit adjoint is refused above
`WHOLE_FIT_ADJOINT_MAX_GB` (`require_whole_fit_adjoint`).

One fit adjoint per force: the interpolated exchange skeletons of a force (the
relaxed density's, Sigma_x - v_xc's and the ISDF-K mean field's own)
differentiate the same row fit as the chain wherever their grid is its points,
and that adjoint is linear in its seeds. Inside a force's `one_fit_adjoint`
window each skeleton leaves its seeds (`PendingFitAdjoint`), and the chain's
assembly contracts them with its own in one `fit_rows_adjoints` call per fit:
the seed-free work (Gram factor, three-centre pass, collocations, derivative
integrals) once, the seed-linear part per point chain and per separately
reported force.

X depends on the geometry three times: through the fit and the collocation,
through the points translating with their atom, and through the local frames
turning with the environment. All three are needed to gate below 1e-4.

An axis of a frame carries a pure gauge sign, and no sign convention is
continuous everywhere, so `continued_frames` carries the convention over from
a reference and differentiates that.

Derivative-integral signs are measured, not transcribed: pyscf's int2c2e_ip1,
int3c2e_ip1 and int3c2e_ip2 are all +(grad .), so every nuclear derivative
built from them carries a minus.
"""
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from src.Base.constants import (FIT_CHOLESKY_BLOCK, FIT_REALIZATIONS,
                                ORBITAL_MULTIPLIER_MAX_ITER,
                                ORBITAL_MULTIPLIER_TOL,
                                THREE_CENTER_BLOCK_BYTES,
                                WHOLE_FIT_ADJOINT_MAX_GB)
from src.Base.dispersion import refuse_dispersion_under_rpa
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import scf as pyscf_scf
from pyscf.grad import rhf as grad_rhf

from src.Base.distributed_df import distributed_fock, distributed_handles
from src.Base.distributed_isdf_jk import DistributedISDFJK
from src.Base.environment import dresses_interaction
from src.Base.isdf_jk import (ISDFJK, isdf_grid, mean_field_skeleton_force,
                              range_coulomb)
from src.Base.pcm_derivatives import reaction_field_fock_skeleton
from src.Base.pyscf_interface import fock_mo, response_kernel
from src.Base.separable_ri import (ANGULAR_WEIGHTS, DEFAULT_PAIR_TOL,
                                   DEFAULT_REGULARIZATION, _FRAME_DECAY,
                                   AdjointSeeds, _ao_l_labels, _content_key,
                                   atomic_frames, atomic_points,
                                   aux_metric_sqrt, build_D_F, fit_M_stable,
                                   fit_rows_adjoints,
                                   rows_transpose_product, screened_layout,
                                   subshells, test_set_D, test_set_layout)
from src.Base.skeleton_tiles import (fitted_fock_skeleton,
                                     isdf_exchange_seeds, xc_grid_skeleton)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import current_comm, lockstep
from src.gradients.multipliers import solve_orbital_multipliers
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.LinearResponse.rpa_energy import (
    exchange_channels, exx_double_counting, rsh_split, xc_hybrid_coeff)

#: The row-fit adjoint pending on this thread (`one_fit_adjoint`): each
#: simulated rank is a thread assembling its own force.
_PENDING = threading.local()


class FitKey:
    """What a row fit's adjoint is a function of, by content: the molecule,
    the auxiliary basis, the points, the pair layout, the tile edge and the
    estimator's settings. Seeds of one key share one
    `separable_ri.fit_rows_adjoints` call (`contract`)."""

    def __init__(self, mol, auxmol, coords, layout, block, l_max_second=2,
                 regularization=DEFAULT_REGULARIZATION, block_memory_gb=4.0):
        self.mol, self.auxmol = mol, auxmol
        self.coords, self.layout = np.asarray(coords), layout
        self.block = int(block)
        self.settings = dict(l_max_second=int(l_max_second),
                             regularization=float(regularization),
                             block_memory_gb=float(block_memory_gb))
        self.key = (_content_key(mol), _content_key(auxmol),
                    self.coords.tobytes(),
                    tuple(np.asarray(a, dtype=kind).tobytes() for a, kind
                          in zip(layout, (np.int64, np.int64, np.float64))),
                    self.block, tuple(sorted(self.settings.items())))

    def __eq__(self, other):
        return isinstance(other, FitKey) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def contract(self, targets):
        """[RowFitAdjoint] of `targets` (`AdjointSeeds`) on this fit."""
        return fit_rows_adjoints(self.mol, self.auxmol, self.coords,
                                 self.layout, targets, block=self.block,
                                 **self.settings)


class PointChain:
    """How a point adjoint P[g] = dE/dr_g reaches the nuclei for one grid:
    r_g = p_g F_i + R_i over the clouds `pts_local` and their owners, the
    frames F_i turning with the geometry or not (`with_frames`)."""

    def __init__(self, pts_local, owner, frames, with_frames,
                 decay=_FRAME_DECAY):
        self.pts_local = [np.asarray(p) for p in pts_local]
        self.owner = np.asarray(owner)
        self.frames = None if frames is None else np.asarray(frames)
        self.with_frames, self.decay = bool(with_frames), float(decay)

    def same(self, other):
        """Whether `other` takes a point adjoint to the same force."""
        if (self.with_frames != other.with_frames or self.decay != other.decay
                or len(self.pts_local) != len(other.pts_local)
                or not np.array_equal(self.owner, other.owner)
                or not all(np.array_equal(a, b) for a, b in
                           zip(self.pts_local, other.pts_local))):
            return False
        if not self.with_frames:
            return True
        if self.frames is None or other.frames is None:
            return self.frames is None and other.frames is None
        return np.array_equal(self.frames, other.frames)

    def __call__(self, mol, P):
        return point_chain(mol, P, self.pts_local, self.owner,
                           frames=self.frames, decay=self.decay,
                           with_frames=self.with_frames)


class PendingFitAdjoint:
    """The row fit's adjoint seeds the exchange skeletons of one force leave
    for its assembly instead of contracting them (`one_fit_adjoint`).

    A skeleton inside the window deposits its seeds (`deposit`) and returns
    what reaches the nuclei outside the fit. The assembly contracts them with
    its own fit adjoint (`contract`, from `row_fit_adjoint`) in one
    `fit_rows_adjoints` call per fit: deposits on the same fit and point chain
    pool into one target, their seeds summed in deposit order
    (`separable_ri.summed_seeds`), the chain's own first where its points
    close the same way; the rest (the chain's frozen frames against the mean
    field's turning ones, and the mean field's own force, reported apart)
    ride the same call as targets of their own, sharing every seed-free part.
    The deposits' share of the force is `settle`'s; the mean field's
    skeleton, contracted ahead where its force follows (`mean_field`), waits
    in `stash` for that force (`take`).
    """

    def __init__(self, mf, mean_field=False):
        self.mf = mf
        self.mean_field = bool(mean_field)
        self.open = True
        self.deposits = []
        self.share = None
        self.stash = None

    def deposit(self, key, chain, seeds):
        """Hold one skeleton's seeds for the assembly's call."""
        self.deposits.append((key, chain, seeds))

    def take(self, request):
        """The mean field's exchange skeleton for `request` (dm, dm_other,
        prefactor, channels) where it was contracted ahead, once; else
        None."""
        if self.stash is None or not _same_request(self.stash[0], request):
            return None
        value, self.stash = self.stash[1], None
        return value

    def contract(self, key=None, seeds=None, chain=None):
        """The chain's own target's `RowFitAdjoint` (None without one), and
        every deposit contracted in the call of its fit; the window closes.

        A deposit whose fit and point chain are the chain's pools into the
        chain's target, so its share is inside the chain's branches; the
        others' (natm, 3) add into `share`. Where the mean field's force
        follows and its exchange skeleton is on a fit this call contracts,
        it rides along and waits in `stash`.
        """
        self.open = False
        entries = []
        if seeds is not None:
            entries.append({'key': key, 'chain': chain, 'seeds': [seeds],
                            'role': 'own'})
        for k, c, s in self.deposits:
            for entry in entries:
                if entry['key'] == k and entry['chain'].same(c):
                    entry['seeds'].append(s)
                    break
            else:
                entries.append({'key': k, 'chain': c, 'seeds': [s],
                                'role': 'deposit'})
        self.deposits = []
        if self.mean_field and self.stash is None:
            ahead = _mean_field_exchange_target(
                self.mf, {e['key'] for e in entries})
            if ahead is not None:
                entries.append(ahead)
        own, share = None, None
        keys = []
        for entry in entries:
            if entry['key'] not in keys:
                keys.append(entry['key'])
        for k in keys:
            group = [e for e in entries if e['key'] == k]
            outs = k.contract([e['seeds'] for e in group])
            for entry, out in zip(group, outs):
                if entry['role'] == 'own':
                    own = out
                    continue
                points = entry['chain'](k.mol, out.fit_points
                                        + out.coll_points)
                if entry['role'] == 'deposit':
                    value = out.fit_centre + out.coll_centre + points
                    share = value if share is None else share + value
                else:
                    centre = out.fit_centre + out.coll_centre + entry['two']
                    self.stash = (entry['request'], lockstep(centre + points))
        if share is not None:
            share = lockstep(share)
            self.share = share if self.share is None else self.share + share
        return own

    def settle(self):
        """(natm, 3): the deposits' share of the force, contracted now where
        no chain target took them; 0.0 without any. The window closes."""
        if self.deposits:
            self.contract()
        self.open = False
        share, self.share = self.share, None
        return 0.0 if share is None else share


def point_layout(mol, radii_by_element, origin_by_element=None):
    """(pts_local, atom_of_point): the atom-local clouds and their owners.

    The half of `separable_ri.molecular_points_covariant` that a derivative
    needs and that one does not return. It stacks the placed points and
    keeps only the coordinates, while a chain rule on r_g = p_g F_i + R_i needs
    p_g and i separately. The two share the `atomic_points` call and the
    stacking order, which is what fixes the row order of X.
    """
    pts_local, owner = [], []
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        use_origin = (origin_by_element or {}).get(sym, False)
        p = atomic_points(radii_by_element[sym], centre=(0.0, 0.0, 0.0),
                          origin=use_origin)
        pts_local.append(p)
        owner.append(np.full(len(p), ia, dtype=int))
    return pts_local, np.concatenate(owner)


def continued_frames(mol, frames_ref, decay=_FRAME_DECAY):
    """`atomic_frames` with each axis matched to a reference frame.

    Axis k is the raw axis with the largest |overlap| with frames_ref[i, k],
    signed to make that overlap positive, so the convention varies continuously
    with the geometry instead of following `eigh`'s arbitrary output sign. A
    proper rotation is restored at the end, since matching may leave an odd
    number of flips.
    """
    raw = atomic_frames(mol, decay=decay)[0]
    out = np.empty_like(raw)
    for i in range(mol.natm):
        taken = []
        for k in range(3):
            ov = raw[i] @ frames_ref[i, k]
            cand = [j for j in range(3) if j not in taken]
            j = cand[int(np.argmax(np.abs(ov[cand])))]
            taken.append(j)
            out[i, k] = raw[i, j] * (1.0 if ov[j] >= 0 else -1.0)
        if np.linalg.det(out[i]) < 0:
            # the reference is a proper rotation, so an odd number of flips
            # means the least-determined axis should carry the parity
            ov = np.abs([out[i, k] @ frames_ref[i, k] for k in range(3)])
            out[i, int(np.argmin(ov))] *= -1.0
    return out


def frame_adjoint(mol, W, decay=_FRAME_DECAY, frames=None, degenerate=None):
    """(natm, 3) gradient of sum_i sum_ab W[i,a,b] F_i[a,b], F_i the atomic frame.

    F_i's rows are the eigenvectors of T_i = sum_j w_j dhat_j dhat_j^T, sorted
    by descending eigenvalue and sign-fixed. With ordering and signs frozen,
    Tbar_i is the standard non-degenerate eigenvector adjoint, symmetrized
    because dT is symmetric, and T_i's own chain to the coordinates closes it.
    """
    coords = np.asarray(mol.atom_coords())
    natm = len(coords)
    if frames is None or degenerate is None:
        f, d = atomic_frames(mol, decay=decay)
        frames = f if frames is None else frames
        degenerate = d if degenerate is None else degenerate
    grad = np.zeros((natm, 3))
    for i in range(natm):
        if not np.any(W[i]):
            continue
        if degenerate[i]:
            raise ValueError(
                f"atom {i} has a degenerate local frame: its axes are not a "
                f"differentiable function of the geometry, and the construction "
                f"takes a branch there. Displace the reference, or exclude the "
                f"atom's points from the adjoint.")
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        keep = r > 1e-8
        idx = np.flatnonzero(keep)
        dj = d[keep] / r[keep, None]
        w = np.exp(-r[keep] / decay)
        T = np.einsum('j,ja,jb->ab', w, dj, dj)
        lam, V = np.linalg.eigh(T)
        # Match each frame axis to its eigenvector by overlap, not by
        # eigenvalue order. Continued frames may be ordered differently from
        # eigh's, and an order-based match would pair the wrong axes.
        vbar = np.zeros((3, 3))                       # vbar[a] for eigenvector a
        taken = []
        for k in range(3):
            ov = V.T @ frames[i, k]
            cand = [j for j in range(3) if j not in taken]
            a = cand[int(np.argmax(np.abs(ov[cand])))]
            taken.append(a)
            vbar[a] = np.sign(ov[a]) * W[i, k]
        Tbar = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                gap = lam[a] - lam[b]
                if abs(gap) < 1e-10:
                    raise ValueError(
                        f"atom {i}: eigenvalue gap {gap:.2e} is too small for a "
                        f"frame derivative; the construction is on a branch.")
                Tbar += ((vbar[a] @ V[:, b]) / gap) * np.outer(V[:, a], V[:, b])
        Tbar = 0.5 * (Tbar + Tbar.T)
        # T = sum_j w(r_j) dhat_j dhat_j^T,  d_j = R_j - R_i
        for n, j in enumerate(idx):
            dh, rj, wj = dj[n], r[j], w[n]
            q = Tbar @ dh
            g = (-wj / decay) * float(dh @ q) * dh + (2.0 * wj / rj) * (q - float(dh @ q) * dh)
            grad[j] += g
            grad[i] -= g
    return grad


def basis_centre_forces(basis_mol, coords, bar):
    """(centre gradient, dE/dr_g) for sum_{g,mu} bar[g,mu] chi_mu(r_g).

    Splits the two ways a collocation depends on the geometry, so that several
    bases on the same points can each contribute their own centre term while
    sharing one point chain.

    pyscf's GTOval_ip_sph is +d(chi)/dr, and chi_mu(r) depends on a nuclear
    coordinate only through its own centre, where the derivative is -d(chi)/dr.
    """
    g = basis_mol.eval_gto('GTOval_ip_sph', coords)        # (3, M, nbas_ao)
    centre = np.zeros((basis_mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(basis_mol.aoslice_by_atom()):
        centre[ia] -= np.einsum('xgm,gm->x', g[:, :, p0:p1], bar[:, p0:p1])
    return centre, np.einsum('xgm,gm->gx', g, bar)         # (natm,3), (M,3)


def point_chain(mol, P, pts_local, atom_of_point, frames=None,
                decay=_FRAME_DECAY, with_frames=True):
    """(natm, 3) from an adjoint P[g] = dE/dr_g on the interpolation points.

    The points translate with the atom that owns them and turn with that atom's
    frame: r_g = p_g F_i + R_i, so dE/dF_i[a,b] = sum_{g in i} p_g[a] P[g,b].
    """
    grad = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        grad[ia] += P[atom_of_point == ia].sum(axis=0)
    if with_frames:
        W = np.zeros((mol.natm, 3, 3))
        for ia in range(mol.natm):
            W[ia] = pts_local[ia].T @ P[atom_of_point == ia]
        grad += frame_adjoint(mol, W, decay=decay, frames=frames)
    return grad


def collocation_adjoint(mol, coords, Xao_bar, pts_local, atom_of_point,
                        frames=None, decay=_FRAME_DECAY, with_frames=True):
    """(natm, 3) gradient of sum_{g,mu} Xao_bar[g,mu] chi_mu(r_g).

    The orbital-basis case: AO centres plus the shared point chain.
    `Xao_bar` is the adjoint on the AO-basis collocation; from an adjoint on
    X_mo = X_ao C it is X_bar C^T.
    """
    centre, P = basis_centre_forces(mol, coords, Xao_bar)
    return centre + point_chain(mol, P, pts_local, atom_of_point, frames=frames,
                                decay=decay, with_frames=with_frames)


# ---------------------------------------------------------------------------
# the fit: adjoint of the regularized least-squares estimator
# ---------------------------------------------------------------------------

def fit_adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
    """(D_bar, F_bar) of sum_{beta k} M_bar[beta,k] M[beta,k], M from `fit_M`.

    `fit_M` row-balances D, forms a regularized Gram matrix and solves

        s_k = ||D[k,:]||,   d = 1/s,   Dt = d D
        G   = Dt Dt^T + reg I
        M   = (F Dt^T) G^-1 d

    so the reverse pass is that read backwards. With B = A G^-1 (A = F Dt^T)
    and A_bar = B_bar G^-1, the inverse contributes

        G_bar = -G^-1 A^T B_bar G^-1 = -B^T A_bar,

    low rank and formed from those two single solves -- the row fit's
    G_bar = -W Z^T (`separable_ri.fit_rows_adjoints`). No dM/dR is ever
    formed, and G is the matrix the forward pass already factorized.

    Nothing here forms G^-1. G is floored by `regularization` and reaches a
    large condition number on a large basis. Applying G^-1 to the
    (nk, nk) product A^T B_bar and again to its result carries a last-bit
    change of M_bar about 1e4 times further than the low-rank form does, as
    on the row fit (tests/test_fit_adjoint_stability.py), and an explicit
    inverse loses most of double precision. G is symmetric positive definite,
    so one Cholesky factorization serves every solve.
    `fit_adjoint_conditioning` reports the amplification.
    """
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    d = 1.0 / s
    Dt = D * d[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    cho = scipy.linalg.cho_factor(G, lower=True)
    del G
    A = F @ Dt.T
    B = scipy.linalg.cho_solve(cho, A.T).T      # A G^-1, G symmetric

    B_bar = M_bar * d[None, :]
    d_bar = np.einsum('bk,bk->k', M_bar, B)
    A_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
    del A, B_bar, cho
    # G_bar = -G^-1 A^T B_bar G^-1 = -B^T A_bar: low rank, from the two
    # single solves already made, never G^-1 applied to an (nk, nk) product
    G_bar = -(B.T @ A_bar)
    del B
    F_bar = A_bar @ Dt
    # Dt_bar = A_bar^T F + (G_bar + G_bar^T) Dt, the Dt term formed first so
    # that Dt is gone before the F term: three arrays of D's shape at most
    S = G_bar + G_bar.T
    del G_bar
    Dt_bar = S @ Dt
    del S, Dt
    Dt_bar += A_bar.T @ F
    del A_bar

    d_bar = d_bar + np.einsum('kr,kr->k', Dt_bar, D)
    s_bar = -d_bar * d ** 2                    # d = 1/s
    D_bar = Dt_bar
    D_bar *= d[:, None]
    c = s_bar / s                              # s = ||D[k,:]||
    for r0 in range(0, len(c), FIT_CHOLESKY_BLOCK):
        r1 = r0 + FIT_CHOLESKY_BLOCK
        D_bar[r0:r1] += c[r0:r1, None] * D[r0:r1]
    return D_bar, F_bar


def fit_adjoint_conditioning(D, regularization=DEFAULT_REGULARIZATION):
    """(cond(G), reg, amplification) for the Gram matrix the fit adjoint inverts.

    The adjoint's error scales as cond(G)^2 because G^-1 enters twice, so this
    is the number that says whether a converged-looking fit gradient can be
    trusted.
    """
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    Dt = D * (1.0 / s)[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    w = np.linalg.eigvalsh(G)
    cond = float(w.max() / max(w.min(), 1e-300))
    return cond, regularization, cond ** 2


# ---------------------------------------------------------------------------
# the grid itself: the fit error differentiated in the interpolation points
# ---------------------------------------------------------------------------

def fit_error_coulomb_grad(mol, auxmol, coords, layout=None, F=None, V=None,
                           l_max_second=2, pair_tol=DEFAULT_PAIR_TOL,
                           regularization=DEFAULT_REGULARIZATION):
    """(E, dE/dr_g) for `separable_ri.fit_error_coulomb`, the grid's objective.

    E = ||M D - F||_V, and of the three only D knows where the points are.
    E^2 = Tr[R^T V R] with R = M D - F reverses as R_bar = 2 V R,
    M_bar = R_bar D^T, D_bar = M^T R_bar plus the fit's own adjoint, and

        dE/dr_g = (1/2E) sum_c D_bar[g,c] dD[g,c]/dr_g,

    one GTOval_ip_sph evaluation. Row g of D depends on r_g alone, so the points
    do not couple and the result is (n_k, 3), not a Jacobian.

    E comes from `fit_M_stable`, so value and gradient are the same realization
    of the estimator. `layout` freezes which AO pairs the test set holds, since
    screening is a discrete choice and a derivative is only defined at fixed
    column set. It has to travel with its F, whose columns are those same pairs.

    The fit adjoint applies G^-1 twice, so the accuracy here follows cond(G).
    See `fit_adjoint_conditioning`.
    """
    if (layout is None) != (F is None):
        raise ValueError('layout and F describe the same columns and must be '
                         'passed together; F built by screening at these '
                         'coordinates does not match a frozen layout.')
    if V is None:
        V = auxmol.intor('int2c2e', aosym='s1')
    if layout is None:
        layout = test_set_layout(mol, coords, l_max_second=l_max_second,
                                 pair_tol=pair_tol)
        F = build_D_F(mol, auxmol, coords, l_max_second=l_max_second,
                      pair_tol=pair_tol)[1]
    D = test_set_D(mol, auxmol, coords, layout)
    M = fit_M_stable(D, F, regularization)
    R = M @ D - F
    E = float(np.sqrt(np.einsum('br,bc,cr->', R, V, R)))

    R_bar = 2.0 * (V @ R)                                  # d(E^2)/dR
    D_bar = M.T @ R_bar                                    # R's explicit D
    D_bar += fit_adjoint(D, F, R_bar @ D.T, regularization)[0]   # and M's

    mu, nu, w = layout
    nao = mol.nao_nr()
    ao = mol.eval_gto('GTOval_sph', coords)
    g = mol.eval_gto('GTOval_ip_sph', coords)              # (3, nk, nao)
    gaux = auxmol.eval_gto('GTOval_ip_sph', coords)
    pair_bar = D_bar[:, :len(mu)] * w[None, :]

    # Fold the pair adjoint onto the AO index before meeting the gradients.
    # Each column contributes chi_mu' chi_nu + chi_mu chi_nu'. The direct
    # einsum over the column list instead materializes (3, n_k, n_pair), which
    # is gigabytes on a molecule.
    S = np.zeros((len(coords), nao))
    for first, second in ((mu, nu), (nu, mu)):
        order = np.argsort(first, kind='stable')
        edges = np.searchsorted(first[order], np.arange(nao + 1))
        for m in range(nao):
            cols = order[edges[m]:edges[m + 1]]
            if len(cols):
                S[:, m] += np.einsum('kc,kc->k', pair_bar[:, cols], ao[:, second[cols]])
    P = (np.einsum('xkm,km->kx', g, S)
         + np.einsum('kc,xkc->kx', D_bar[:, len(mu):], gaux))
    return E, P / (2.0 * E)


def atomic_radii_adjoint(radii, P, origin=False):
    """{shell: dE/dr_i} from an adjoint P[g] = dE/dr_g on one atom's points.

    The reverse of `atomic_points`, walking the point list in that function's
    order. r_g = r_i u_g with u_g a unit direction, so dE/dr_i sums u_g . P[g]
    over the sub-shell. The cusp sample sits at the nucleus and carries no
    radius.
    """
    shells, out, k = subshells(), {}, (1 if origin else 0)
    for name, rs in radii.items():
        u = shells[name]
        grad = np.empty(len(np.atleast_1d(rs)))
        for i in range(len(grad)):
            grad[i] = np.einsum('gx,gx->', u, P[k:k + len(u)])
            k += len(u)
        out[name] = grad
    return out


# ---------------------------------------------------------------------------
# the D factor: D = M^T V^(1/2), all the way back to the nuclei
# ---------------------------------------------------------------------------

def sqrtm_adjoint(V, Vh_bar, tol=1e-12):
    """V_bar for an adjoint on V^(1/2), V symmetric positive semidefinite.

    The Frechet derivative L of the square root solves Vh L + L Vh = E, so in
    the eigenbasis it is the Hadamard multiplier 1/(sqrt(w_i) + sqrt(w_j)).
    That multiplier is symmetric, hence the map is self-adjoint and the reverse
    pass is the same operation. The truncation mirrors `separable_factors`,
    which drops eigenvalues below tol x max before taking the root.
    """
    w, U = np.linalg.eigh(V)
    keep = w > tol * w.max()
    Uk, sk = U[:, keep], np.sqrt(w[keep])
    E = Uk.T @ (0.5 * (Vh_bar + Vh_bar.T)) @ Uk
    return Uk @ (E / (sk[:, None] + sk[None, :])) @ Uk.T


def two_centre_adjoint(auxmol, V_bar, omega=0.0):
    """(natm, 3) gradient of sum_PQ V_bar[P,Q] (P|Q), V_bar symmetrized here.

    omega != 0 takes the integrals with the erf-attenuated operator, the
    long-range channel of a range-separated hybrid.

    pyscf's int2c2e_ip1 is +(grad P|Q), so the nuclear derivative is minus it,
    and both index positions contribute equally once V_bar is symmetric.
    """
    with auxmol.with_range_coulomb(omega):
        v1 = auxmol.intor('int2c2e_ip1', comp=3)           # +(grad P|Q)
    Vb = 0.5 * (V_bar + V_bar.T)
    t = -2.0 * np.einsum('xPQ,PQ->xP', v1, Vb, optimize=True)
    grad = np.zeros((auxmol.natm, 3))
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += t[:, q0:q1].sum(axis=1)
    return grad


def shell_blocks(mol, per_ao_bytes, max_bytes):
    """Shell ranges whose slab, `per_ao_bytes` for each AO in it, fits the cap.

    Always yields at least one shell, so a single shell wider than the cap is
    still attempted rather than silently skipped.
    """
    ao_loc = mol.ao_loc_nr()
    blocks, sh0 = [], 0
    while sh0 < mol.nbas:
        sh1 = sh0 + 1
        # int() because ao_loc_nr() is int32 while a slab is nao*naux*8 bytes or
        # more. The product wraps negative on a large system, the test then
        # passes for every block, and the loop asks for the whole tensor.
        while (sh1 < mol.nbas
               and int(ao_loc[sh1 + 1] - ao_loc[sh0]) * per_ao_bytes <= max_bytes):
            sh1 += 1
        blocks.append((sh0, sh1))
        sh0 = sh1
    return blocks


def three_centre_adjoint(mol, auxmol, G3, omega=0.0, max_bytes=THREE_CENTER_BLOCK_BYTES):
    """(natm, 3) gradient of sum_{mn P} G3[m,n,P] (mn|P).

    omega != 0 takes the erf-attenuated operator, set on both molecules since
    `aux_e2` reads the range parameter from the combined environment.

    G3 is not symmetric in (m, n), because the ISDF test set pairs every AO with
    only the low-l ones, so the two orbital positions are contracted separately.
    The derivative integrals are contracted where they are made, blocked over
    the shells of the differentiated index, since a whole (3, nao, nao, naux)
    tensor is gigabytes on a large system.
    """
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    t_first, t_second = np.zeros((3, nao)), np.zeros((3, nao))
    t_aux = np.zeros((3, naux))
    per_ao = 3 * int(nao) * int(naux) * 8
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        for sh0, sh1 in shell_blocks(mol, per_ao, max_bytes):
            a0, a1 = ao_loc[sh0], ao_loc[sh1]
            sl = (sh0, sh1, 0, mol.nbas, 0, auxmol.nbas)
            d1 = pyscf_df.incore.aux_e2(
                mol, auxmol, intor='int3c2e_ip1', aosym='s1', comp=3,
                shls_slice=sl).reshape(3, a1 - a0, nao, naux)
            # minus, for the same reason as the two-centre case
            t_first[:, a0:a1] = -np.einsum('xmnP,mnP->xm', d1, G3[a0:a1],
                                           optimize=True)
            t_second[:, a0:a1] = -np.einsum('xnmP,mnP->xn', d1, G3[:, a0:a1],
                                            optimize=True)
            del d1
            d2 = pyscf_df.incore.aux_e2(
                mol, auxmol, intor='int3c2e_ip2', aosym='s1', comp=3,
                shls_slice=sl).reshape(3, a1 - a0, nao, naux)
            t_aux -= np.einsum('xmnP,mnP->xP', d2, G3[a0:a1], optimize=True)
            del d2
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] += t_first[:, p0:p1].sum(axis=1) + t_second[:, p0:p1].sum(axis=1)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += t_aux[:, q0:q1].sum(axis=1)
    return grad


@dataclass(frozen=True)
class GaugeAdjoint:
    """One gauge of one fit: the D adjoint, its environment, and any adjoint a
    term outside D put on the same metric.

    d_bar: (nk, naux) adjoint on this gauge's factor.
    environment: what dresses this gauge's metric, None for the bare one.
    root_bar: an adjoint on the metric root R = (V + vtilde)^(1/2) from a term
        that reads R without going through D. It is added to D's own root
        adjoint before the Frechet solve, so the two halves cannot drift apart.
    kernel_bar: an adjoint on vtilde alone, which does not reach V and so cannot
        join `root_bar`. It goes to the environment's own adjoint instead.

    Both extras are refused on a bare gauge, which has no root and no kernel.
    """

    d_bar: np.ndarray
    environment: object = None
    root_bar: np.ndarray = None
    kernel_bar: np.ndarray = None


def dfactor_adjoint(mol, auxmol, coords, D_bar, layout, pts_local,
                    atom_of_point, frames=None, decay=_FRAME_DECAY,
                    regularization=DEFAULT_REGULARIZATION, with_frames=True,
                    environment=None):
    """(natm, 3) gradient of sum_{gP} D_bar[g,P] D[g,P], D = M^T V_env^(1/2)."""
    return dfactor_adjoint_gauges(mol, auxmol, coords,
                                  [(D_bar, environment)], layout, pts_local,
                                  atom_of_point, frames=frames, decay=decay,
                                  regularization=regularization,
                                  with_frames=with_frames)


def product_pairs(mol, l_max_second=2):
    """(mu, nu, weight) of every pair of the test set's product basis, (all
    AOs) x (AOs with l <= l_max_second), in `test_set_layout`'s order.

    These are the columns the Gram matrix of `fit_M_streaming` and
    `fit_rows` sums over, screened or not: S = (A A^T) o (B B^T) is the
    product form of D D^T over all of them, while F D^T keeps the screened
    ones. A frozen `test_set_layout` is a subsequence of this list.
    """
    l_ao = _ao_l_labels(mol)
    second = np.flatnonzero(l_ao <= l_max_second)
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    nao = mol.nao_nr()
    return (np.repeat(np.arange(nao), len(second)), np.tile(second, nao),
            np.tile(w, nao))


def whole_fit_adjoint_gb(nao, naux, npoint, ngram):
    """GB the fit adjoint formed whole holds at once on one rank: the test
    set over `ngram` product pairs and the auxiliaries, (npoint, ngram +
    naux), three times (D_test, its adjoint, one temporary), F of that
    width, and the (nao, nao, naux) three-centre tensor with its adjoint."""
    width = ngram + naux
    return 8 * (3 * npoint * width + naux * width
                + 2 * nao * nao * naux) / 1e9


def require_whole_fit_adjoint(nao, naux, npoint, ngram, what):
    """`whole_fit_adjoint_gb`, refused above `WHOLE_FIT_ADJOINT_MAX_GB`
    before any of those arrays exists; the row fit's adjoint serves the same
    estimator in grid-row tiles."""
    gb = whole_fit_adjoint_gb(nao, naux, npoint, ngram)
    if gb > WHOLE_FIT_ADJOINT_MAX_GB:
        raise MemoryError(
            f'{what} forms the fit adjoint whole: the test set over every '
            f'product pair, ({npoint}, {ngram} + {naux}), three times, F and '
            f'the ({nao}, {nao}, {naux}) three-centre tensor with its '
            f'adjoint, {gb:.3g} GB on every rank, above '
            f'WHOLE_FIT_ADJOINT_MAX_GB = {WHOLE_FIT_ADJOINT_MAX_GB:g} GB. Use '
            f"the row fit, fit='rows' (a FrozenFactorization with "
            f"sliced=True, fit='rows'; the default of the exchange "
            f'skeletons), whose adjoint runs in the grid-row tiles of '
            f'`separable_ri.fit_rows_adjoint`.')
    return gb


def dfactor_adjoint_gauges(mol, auxmol, coords, gauges, layout, pts_local,
                           atom_of_point, frames=None, decay=_FRAME_DECAY,
                           regularization=DEFAULT_REGULARIZATION,
                           with_frames=True, gram_layout=None):
    """(natm, 3) gradient of sum over `gauges` of sum_{gP} D_bar[g,P] D[g,P].

    The whole D branch. Split D into the fit and the metric root, run the fit's
    Z-vector (`fit_adjoint`), then land the test-set adjoints on integrals.

        D[g,P] = sum_Q M[Q,g] Vh[Q,P]      Mbar = Vh Dbar^T,  Vhbar = M Dbar
        Mbar  --fit_adjoint-->  (Dtest_bar, F_bar)
        Dtest  columns are collocations: AO pairs, then auxiliaries
        F      columns are w_c V^-1 (mu nu|.) for pairs, and the
               identity for auxiliaries, which therefore contributes nothing
        Vh     contributes through the square-root Frechet adjoint, and V^-1
               inside F contributes a second term to V_bar

    Each gauge is a `GaugeAdjoint`, or the bare (D_bar, environment) pair. A
    solvated target carries two gauges of one fit, since the self-energy screens
    bare while the kernel keeps the cavity. Everything below the metric root is
    common to them, so the fit is built once and each gauge contributes its own
    adjoints to shared accumulators. Both bases collocate on the same points, so
    their centre terms differ but the point and frame chain is shared.

    gram_layout: the (mu, nu, weight) pairs the Gram matrix and the row
    balancing sum over; None is every product pair (`product_pairs`), the
    estimator of `fit_M_streaming` and `fit_rows`. Its columns beyond
    `layout`'s enter D_test and carry F = 0, so they reach the force through
    the collocation alone. Refused above `WHOLE_FIT_ADJOINT_MAX_GB`
    (`require_whole_fit_adjoint`).
    """
    if gram_layout is None:
        gram_layout = product_pairs(mol)
    require_whole_fit_adjoint(mol.nao_nr(), auxmol.nao_nr(), len(coords),
                              len(gram_layout[0]), 'dfactor_adjoint_gauges')
    mu, nu, wc = layout
    naux = auxmol.nao_nr()
    ao = mol.eval_gto('GTOval_sph', coords)
    V = auxmol.intor('int2c2e', aosym='s1')

    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    e3c = e3c.reshape(mol.nao_nr(), mol.nao_nr(), naux)
    g_cols = e3c[mu, nu, :].T                              # (naux, npair)
    # the adjoint on (mu nu|P) in the integrals' own memory order, which the
    # derivative contraction's BLAS calls read; zeros stay unmapped until used
    G3 = np.zeros_like(e3c)
    del e3c
    F_pairs = np.linalg.solve(V, g_cols) * wc[None, :]
    del g_cols
    gmu, gnu, gw = gram_layout
    cols = pair_positions(layout, gram_layout, mol.nao_nr())
    F = np.zeros((naux, len(gmu) + naux))
    F[:, cols] = F_pairs
    F[:, len(gmu):] = np.eye(naux)
    del F_pairs
    ngram = len(gmu)
    D_test = test_set_D(mol, auxmol, coords, (gmu, gnu, gw))
    M = fit_M_stable(D_test, F, regularization)

    V_bar = np.zeros_like(V)
    ao_bar = np.zeros_like(ao)
    aux_bar = np.zeros((len(coords), auxmol.nao_nr()))
    env_grad = None
    for gauge in gauges:
        gauge = gauge if isinstance(gauge, GaugeAdjoint) else GaugeAdjoint(*gauge)
        D_bar, environment = gauge.d_bar, gauge.environment
        dressed = dresses_interaction(environment, auxmol)
        if not dressed and (gauge.root_bar is not None
                            or gauge.kernel_bar is not None):
            raise ValueError(
                f'{environment!r} dresses nothing, so this gauge has no '
                'vtilde and no dressed root for the extra adjoints to land '
                'on; they belong to the gauge that carries the environment')
        V_gauge = V + environment.aux_kernel(auxmol) if dressed else V
        Vh = aux_metric_sqrt(auxmol, environment, V=V)

        M_bar = Vh @ D_bar.T                               # (naux, nk)
        Vh_bar = M @ D_bar                                 # (naux, naux)
        if gauge.root_bar is not None:
            Vh_bar = Vh_bar + gauge.root_bar
        Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, regularization)

        # --- the metric: the root of the dressed gauge, and the bare V^-1
        # inside F's pair columns. Y is the adjoint on V_env and reaches V and
        # vtilde alike.
        Y = sqrtm_adjoint(V_gauge, Vh_bar)
        Fp_bar = F_bar[:, cols]
        VinvFb = np.linalg.solve(V, Fp_bar)                # (naux, npair)
        V_bar += Y - F[:, cols] @ VinvFb.T
        g_bar = VinvFb * wc[None, :]                       # adjoint on (mu nu|P)
        np.add.at(G3, (mu, nu), g_bar.T)
        if dressed:
            # V_gauge = V + vtilde, so Y is the adjoint on both; a term that
            # differentiated vtilde alone adds only here.
            kernel_bar = (Y if gauge.kernel_bar is None
                          else Y + gauge.kernel_bar)
            k_grad = environment.aux_kernel_adjoint(auxmol, kernel_bar)
            env_grad = k_grad if env_grad is None else env_grad + k_grad

        # --- the test set's collocations, sharing one point chain, in blocks
        # of grid rows: each element still adds its pairs in the same order
        for r0 in range(0, len(coords), FIT_CHOLESKY_BLOCK):
            r1 = r0 + FIT_CHOLESKY_BLOCK
            pair_bar = Dtest_bar[r0:r1, :ngram] * gw[None, :]
            np.add.at(ao_bar[r0:r1].T, gmu,
                      (pair_bar * ao[r0:r1, gnu]).T)       # chi_mu's slot
            np.add.at(ao_bar[r0:r1].T, gnu,
                      (pair_bar * ao[r0:r1, gmu]).T)       # chi_nu's slot
            del pair_bar
        aux_bar += Dtest_bar[:, ngram:]
        # nothing of this gauge's test-set width outlives it into the next
        del Dtest_bar, F_bar, Fp_bar, VinvFb, g_bar
    del D_test, F, M

    grad = two_centre_adjoint(auxmol, V_bar) + three_centre_adjoint(mol, auxmol, G3)
    if env_grad is not None:
        grad = grad + env_grad
    c_ao, P = basis_centre_forces(mol, coords, ao_bar)
    c_aux, P_aux = basis_centre_forces(auxmol, coords, aux_bar)
    grad += c_ao + c_aux
    grad += point_chain(mol, P + P_aux, pts_local, atom_of_point, frames=frames,
                        decay=decay, with_frames=with_frames)
    return grad


def row_fit_adjoint(mol, auxmol, coords, d_bar, x_bar, mo_coeff, layout,
                    pts_local, atom_of_point, frames=None, decay=_FRAME_DECAY,
                    with_frames=True, block=None, pending=None):
    """(g_collocation, g_fit, held): the collocation branch of X_bar and the
    fit branch of D_bar on the row-distributed fit, `fit_rows`' estimator on
    the frozen `layout` (its Gram matrix over every product pair).

    `separable_ri.fit_rows_adjoints` runs both in the fit's tiles and
    returns their centre terms and point adjoints, the same bits on every
    rank at every rank count; the point and frame chain closes them here, on
    the (nk, 3) point adjoints, which are not grid-by-width.
    `dfactor_adjoint_gauges` with `gram_layout=product_pairs(mol)` and
    `collocation_adjoint` are the same derivatives formed whole, to the
    rounding of the fit's balanced Gram matrix.

    pending: the force's `PendingFitAdjoint`, whose skeletons' seeds ride
    the same call (`PendingFitAdjoint.contract`); a skeleton pooled into
    this target arrives inside these two branches.
    """
    key = FitKey(mol, auxmol, coords, layout,
                 FIT_CHOLESKY_BLOCK if block is None else block)
    chain = PointChain(pts_local, atom_of_point, frames, with_frames, decay)
    seeds = AdjointSeeds(d_bar=d_bar, x_bar=x_bar, mo_coeff=mo_coeff)
    adjoint = (key.contract([seeds])[0] if pending is None
               else pending.contract(key, seeds, chain))
    g_coll = adjoint.coll_centre + chain(mol, adjoint.coll_points)
    g_fit = adjoint.fit_centre + chain(mol, adjoint.fit_points)
    return g_coll, g_fit, adjoint.held


def orbital_rotation_rows(x_mo, x_bar, block=None):
    """Y = X_mo^T X_bar, the orbital-rotation gradient of an adjoint on
    X_mo = X_ao C, from grid rows: `SlicedFactors` over the current ranks,
    or X_mo whole on one. `separable_ri.rows_transpose_product` in fixed
    tiles of `block` points: no rank holds X_mo whole, and Y is the same
    bits at every rank count."""
    if isinstance(x_mo, SlicedFactors):
        comm = x_mo.require(current_comm()).comm
        return rows_transpose_product(x_mo.X_mo, x_mo.rows[0], x_bar,
                                      block=block, comm=comm)
    return rows_transpose_product(x_mo, 0, x_bar, block=block)

def pair_positions(layout, gram_layout, nao):
    """Where each (mu, nu) of `layout` sits in `gram_layout`, both in
    ascending (mu, nu) order; a pair the Gram layout lacks is refused."""
    key = np.asarray(gram_layout[0]) * nao + np.asarray(gram_layout[1])
    want = np.asarray(layout[0]) * nao + np.asarray(layout[1])
    pos = np.searchsorted(key, want)
    if (np.any(pos >= len(key))
            or not np.array_equal(key[np.minimum(pos, len(key) - 1)], want)):
        raise ValueError('the F test set holds pairs the Gram layout does not')
    return pos


# ---------------------------------------------------------------------------
# the orbital-energy chain, without any norb^4 tensor
# ---------------------------------------------------------------------------

def require_no_range_separation(mf, what):
    """Refuse a derivative that would silently drop the long-range exchange.

    The skeletons in this module (`fock_partial_skeleton`, its fitted twin,
    `qp_xc_correction_skeleton`, `isdf_exchange_skeleton`) carry every
    channel and do not need it; it is the refusal a single-channel route owes
    its caller.

    The forward quantities are safe on a range-separated hybrid, since `v_xc` is
    the SCF's own `get_veff - get_j`. A skeleton that weights one full-range K
    build by `xc_hybrid_coeff` is not, because that scalar is alpha and the
    whole beta K_lr(omega) term goes missing without announcing itself.
    """
    omega, alpha, beta = rsh_split(mf)
    if omega != 0.0 and beta != 0.0:
        raise NotImplementedError(
            f'{what} does not support the range-separated hybrid '
            f'{mf.xc!r} (omega={omega:g}, alpha={alpha:g}, beta={beta:g}): '
            f'the skeleton derivative carries one full-range exchange build '
            f'weighted by alpha, so the beta K_lr(omega) term would be '
            f'silently dropped. ENERGIES on this reference are correct -- '
            f'v_xc comes from the SCF itself -- so use them, or finite '
            f'differences of them, until the split lands.')


def fock_partial_Y(mf, gamma, nocc):
    """Orbital-rotation gradient of Tr[gamma F], gamma a symmetric MO partial.

    Tr[gamma F] depends on the coefficients through the two-sided transform and
    through the density inside F, and both come out as

        Y[u,p] = 2 (F gamma)[u,p] + 4 delta_{p occupied} (C^T G(gamma_ao) C)[u,p]

    with G(x) the closed-shell response kernel and gamma_ao = C gamma C^T. The
    same expression serves the target's own partial and the multiplier's. It
    needs a Coulomb and an exchange build and nothing else, so with a fitted or
    ISDF mean field it is cubic and no norb^4 tensor appears.
    """
    C = mf.mo_coeff
    gs = 0.5 * (gamma + gamma.T)
    g_ao = C @ gs @ C.T
    G = response_kernel(mf)(g_ao)
    F_mo = fock_mo(mf)
    Y = 2.0 * (F_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ G @ C)[:, :nocc]
    return Y


def qp_xc_correction(mf, states=None, reaction_field=None):
    """`GW.qp_solve.static_exchange_diagonal` at exchange='mf', all states by
    default: the gradient chain and `calc_qp_energy` differentiate and evaluate
    one function, so there is nothing here but the default."""
    states = (np.arange(np.shape(mf.mo_coeff)[-1]) if states is None
              else states)
    return static_exchange_diagonal(mf, mf.mol, states, exchange='mf',
                                    reaction_field=reaction_field)


def _delta_o_response(mf):
    """x -> d(Sigma_x - v_xc)/dD . x, the response of the correction's operator.

    `v_xc` here is `get_veff - get_j`, which on a hybrid already contains
    -a_x K/2, so the operator is

        O[D] = -((1 - a_x)/2) K - v_xc^DFT

    and its response, using f_xc(x) = G_KS(x) - J(x) + (a_x/2) K(x), is

        dO(x) = J(x) - K(x)/2 - G_KS(x).

    On Hartree-Fock that is identically zero, as it must be. Getting the a_x
    bookkeeping wrong gives -K(x)/2, which coincides with the right answer on a
    pure functional and is wrong by the whole gradient on a hybrid. Test a
    hybrid, not just PBE.
    """
    # continuum-free: O itself carries no reaction field, so neither may its
    # response, and the J - K/2 - G identity holds only for the gas-phase G
    kern = response_kernel(mf, environment=False)

    def apply(x):
        with distributed_fock(mf, build=False):
            vj, vk = mf.get_jk(mf.mol, x, hermi=1)
        return vj - 0.5 * vk - kern(x)
    return apply


def sigma_x_minus_vxc(mf, dm):
    """Sigma_x - v_xc in the AO basis, -K/2 - (get_veff - J), at density dm;
    under ranks from the distributed SCF's handles."""
    with distributed_fock(mf, build=False):
        return -0.5 * mf.get_k(mf.mol, dm) - (mf.get_veff(mf.mol, dm)
                                              - mf.get_j(mf.mol, dm))


def qp_xc_correction_Y(mf, weights, nocc):
    """Orbital-rotation gradient of sum_p w_p <p|Sigma_x - v_xc|p>.

    The same shape as `fock_partial_Y`, with the Fock replaced by this
    operator: two terms, one from the two-sided MO transform and one from the
    operator's own density dependence. It must enter the Lagrangian before the
    multiplier solve, because it shares Lambda with every other contribution.
    """
    C = mf.mo_coeff
    gs = np.diag(np.asarray(weights, float))
    g_ao = C @ gs @ C.T
    dm = mf.make_rdm1()
    o_ao = sigma_x_minus_vxc(mf, dm)
    o_mo = C.T @ o_ao @ C
    Y = 2.0 * (o_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ _delta_o_response(mf)(g_ao) @ C)[:, :nocc]
    return Y


def qp_xc_correction_skeleton(mf, weights, nocc):
    """(natm, 3) of d/dR sum_p w_p <p|Sigma_x - v_xc|p>, coefficients fixed.

    The operator is -((1 - a_x)/2) K - v_xc^DFT (see `_delta_o_response`), so
    the exchange half is `fock_partial_skeleton`'s K structure at weight
    -(1 - a_x) with no Coulomb term, and the rest is minus `xc_skeleton`. There
    is no one-electron part. On Hartree-Fock both halves vanish.
    """
    is_ks, _ = xc_hybrid_coeff(mf)
    if not is_ks:
        return np.zeros((mf.mol.natm, 3))
    C = mf.mo_coeff
    gs = np.diag(np.asarray(weights, float))
    # Sigma_x is the full-range exact exchange whatever the reference is, and
    # v_xc carries the reference's own channels, so Sigma_x - v_xc is one
    # full-range unit minus each of them. On a global hybrid that collapses to
    # the single weight 1 - a_x.
    delta = [(0.0, 1.0)] + [(o, -w) for o, w in exchange_channels(mf)]
    # A density-fitted mean field must go through the fitted skeleton, for the
    # same reason `fock_partial_skeleton` routes there. pyscf's conventional and
    # density-fitted derivative J/K do not share a convention, and the mismatch
    # shows up as a sharply degraded translational invariance.
    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        # Sigma_x is the interpolated exchange here, because that is what the
        # SCF built. Taking it from the auxiliary basis would differentiate one
        # operator while the energy carries another.
        return (isdf_fock_partial_exchange(mf, gs, channels=delta)
                - xc_skeleton(mf, gs))
    if getattr(mf, 'with_df', None) is not None:
        grad = fock_partial_skeleton_df(mf, _auxmol_of(mf), gs, nocc,
                                        coulomb=False, channels=delta)
        return grad - xc_skeleton(mf, gs)
    g_ao = C @ gs @ C.T
    D = mf.make_rdm1()
    grad = np.zeros((mf.mol.natm, 3))
    for omega, weight in delta:
        if weight != 0.0:
            grad = grad + exchange_channel_skeleton(mf, g_ao, D, omega, weight)
    return grad - xc_skeleton(mf, gs)


def exx_double_counting_skeleton(mf, mol=None):
    """(natm, 3) of d/dR (E_x^exact - E_xc), MO coefficients held fixed.

    Taken as a difference of two pyscf gradients at one density. Both gradient
    objects read the same mo_coeff/mo_energy/mo_occ, so the one-electron,
    overlap, Coulomb and nuclear terms are bit-identical and cancel, leaving
    full exchange against a_x-exchange-plus-xc. Weighting derivative K builds by
    hand instead is where the coefficients go wrong.

    A continuum is stripped off the Kohn-Sham side first. The reaction field is
    the same functional of the same density in both members, so it belongs to
    neither, and leaving it in would leave -dE_PCM/dR behind. The mean field's
    own force carries it once, in `FactorChain.mean_field_gradient`.

    Each member goes through `isdf_jk.mean_field_skeleton_force`, the one
    dispatch that answers with the force of the energy a given mean field
    reported.
    """
    mol = mf.mol if mol is None else mol
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((mol.natm, 3))
    ks = mf.undo_solvent() if hasattr(mf, 'undo_solvent') else mf
    hf = pyscf_scf.RHF(mol)
    with_df = getattr(ks, 'with_df', None)
    if with_df is not None:
        hf = hf.density_fit(auxbasis=with_df.auxbasis)
    hf.mo_coeff, hf.mo_energy = mf.mo_coeff, mf.mo_energy
    hf.mo_occ, hf.converged = mf.mo_occ, True
    refuse_dispersion_under_rpa(mf, 'the Kohn-Sham-to-Hartree-Fock skeleton')
    if isinstance(with_df, ISDFJK):
        # Both sides must be the same fit, so the Hartree-Fock partner is given
        # the very same ISDFJK object. The points, collocation and fit matrix
        # are then bit-identical and everything but the exchange fraction
        # cancels. A fresh fit would leave its own realization in the
        # difference, and so would the other realization of it, so the
        # distributed SCF's handle travels with the fit.
        hf.with_df = with_df
        handles = getattr(mf, '_distributed', None)
        if handles is not None:
            for partner in (hf, ks):
                if getattr(partner, '_distributed', None) is None:
                    partner._distributed = handles
    return (np.asarray(mean_field_skeleton_force(hf))
            - np.asarray(mean_field_skeleton_force(ks)))


def exx_double_counting_Y(mf, nocc):
    """Orbital-rotation gradient of (E_x^exact - E_xc).

    One term, where `qp_xc_correction_Y` needs two. The functional derivative of
    this energy with respect to the density is the operator
    `qp_xc_correction` carries, dE/dD = Sigma_x - v_xc, and an energy does not
    depend on the coefficients through the states as <p|O[D]|p> does.

    The factor and the occupied restriction follow `fock_partial_Y`.
    """
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((np.shape(mf.mo_coeff)[-1],) * 2)
    C = mf.mo_coeff
    dm = mf.make_rdm1()
    o_ao = sigma_x_minus_vxc(mf, dm)
    Y = np.zeros((C.shape[1], C.shape[1]))
    Y[:, :nocc] = 4.0 * (C.T @ o_ao @ C)[:, :nocc]
    return Y


def solve_lambda(mf, Y_E, nocc, tol=ORBITAL_MULTIPLIER_TOL,
                 max_iter=ORBITAL_MULTIPLIER_MAX_ITER, verbose=False):
    """Symmetric zero-diagonal Lambda with antisym(Y_E + Y[Lambda]) = 0.

    The cubic route's entry to `solve_orbital_multipliers`: the matvec is
    `fock_partial_Y` -- a Coulomb and an exchange build -- instead of the
    four-index tensor, and the orbital energies come from the mean field.
    """
    return solve_orbital_multipliers(
        lambda Lam: fock_partial_Y(mf, Lam, nocc),
        Y_E, mf.mo_energy, tol=tol, max_iter=max_iter, verbose=verbose)


def _auxmol_of(mf):
    """The auxiliary molecule behind a density-fitted mean field."""
    aux = getattr(mf.with_df, 'auxmol', None)
    if aux is None:
        aux = pyscf_df.addons.make_auxmol(mf.mol, auxbasis=mf.with_df.auxbasis)
    return aux


def reaction_field_skeleton(mf, gamma):
    """(natm, 3) the ground-state continuum's entry in the skeleton of Tr[gamma F].

    V_PCM(eps_static) is part of this mean field's Fock, so a folded Fock
    partial owes it a derivative term of the same kind as the Coulomb one, with
    the relaxed density on one side and the SCF density on the other. Zero
    without a continuum, which is why callers add it unconditionally.
    """
    pcm = getattr(mf, 'with_solvent', None)
    if pcm is None:
        return 0.0
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return reaction_field_fock_skeleton(pcm, mf.make_rdm1(), g_ao)


# ---------------------------------------------------------------------------
# the interpolated exchange: E_K = -(a_x/4) sum_PQ Z_PQ W_PQ^2, W = X D X^T
# ---------------------------------------------------------------------------

def isdf_exchange_adjoints(X, Z, dm, a_x, dm_other=None):
    """(X_bar, Z_bar) of the ISDF exchange form

        E = -(a_x/4) sum_PQ Z_PQ (X dm X^T)_PQ (X dm_other X^T)_PQ

    dm_other = None is the SCF's own E_K = -(a_x/4) Tr[dm K[dm]]. W enters
    squared, so W_bar is -(a_x/2) Z .* W and X, sitting on both sides, collects
    twice that.

    A folded Fock partial needs the two-density form. Its exchange half is
    -(gamma vk[D] + D vk[gamma]), and K is linear in the density it is built
    from, so the same contraction serves with W^2 replaced by W1 W2.
    """
    W1 = X @ dm @ X.T
    other = dm if dm_other is None else dm_other
    W2 = W1 if dm_other is None else X @ other @ X.T
    Z_bar = -0.25 * a_x * (W1 * W2)
    X_bar = -0.5 * a_x * ((Z * W2) @ (X @ dm) + (Z * W1) @ (X @ other))
    return X_bar, Z_bar


def isdf_fock_partial_exchange(mf, gamma, channels=None, fit=None):
    """(natm, 3) exchange half of a folded Fock partial, from ISDF factors.

    The folded two-particle density Gamma_abcd = gamma_ab D_cd - gamma_ac D_bd/2
    contributes -(w/2) Tr[gamma K[D]] per channel, while the SCF energy is
    -(a_x/4) Tr[D K[D]]. So the bilinear coefficient is twice the SCF's, not
    eight times.

    A finite difference cannot fix that factor, since the reference and the
    analytic side share it. What pins it is agreement with
    `fock_partial_skeleton_df` at the same gamma.

    `channels` overrides the reference's own, because
    `qp_xc_correction_skeleton` needs Sigma_x - v_xc, one full-range unit minus
    each of the reference's channels. `fit` is `isdf_exchange_skeleton`'s.
    """
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return isdf_exchange_skeleton(mf, dm=g_ao, dm_other=mf.make_rdm1(),
                                  prefactor=2.0, channels=channels, fit=fit)


def exchange_fit(mf, fit=None):
    """The realization of the fit an ISDF exchange skeleton differentiates,
    both of the SCF's estimator: `fit` as given, else 'rows', the row fit's
    tiles on any rank count (the distributed ISDF-K SCF's own where the mean
    field carries its handle). The whole form ('replicated') differentiates
    the same estimator through the dense Gram matrix, whose conditioning
    holds it to 1e-8 of the SCF energy's derivative on ethylene/cc-pVDZ
    PBE0 (one reassociation of it moves the force 2.1e-8) where the rows
    reach 5e-10."""
    if fit is None:
        return 'rows'
    if fit not in FIT_REALIZATIONS:
        raise ValueError(f'fit={fit!r}: one of {FIT_REALIZATIONS}')
    return fit


def isdf_scf_handle(mf):
    """The `DistributedISDFJK` a distributed ISDF-K SCF left on `mf` for the
    current ranks, or None."""
    handles = distributed_handles(mf)
    if handles is None or not isinstance(handles[0], DistributedISDFJK):
        return None
    return handles[0]


def isdf_exchange_skeleton(mf, dm=None, dm_other=None, prefactor=1.0,
                           channels=None, fit=None):
    """(natm, 3) of d/dR E_K^ISDF with the density matrix held fixed.

    The whole ISDF branch of the force: the collocation's adjoint, the fit's
    Z-vector, and the derivative integrals underneath them. Z = M^T V M is
    differentiated as it is built in `z_mode='dense'`. `z_mode='factored'` holds
    L = V^(1/2) M instead, and its Z differs by the modes the square root
    truncates, which is negligible on a bare Coulomb metric.

    fit: 'replicated' rebuilds the fit whole below, W and Z (M, M) and the
    Gram matrix (M, M) on every rank, and is refused above
    `WHOLE_FIT_ADJOINT_MAX_GB`; 'rows' is `_exchange_skeleton_rows`, in the
    row fit's grid-row tiles over the ranks; None is `exchange_fit`'s choice.
    """
    fit = exchange_fit(mf, fit)
    dm = mf.make_rdm1() if dm is None else np.asarray(dm)
    channels = exchange_channels(mf) if channels is None else list(channels)
    if fit == 'rows':
        return _exchange_skeleton_rows(mf, dm, dm_other, prefactor, channels)
    with_df = mf.with_df
    mol, auxmol, crd = with_df.mol, with_df.auxmol, with_df.coords
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    reg = with_df.regularization

    # The fit is rebuilt, not taken from the mean field, so that the function
    # differentiated is a single realization of the SCF's estimator
    # (`fit_M_streaming`: the Gram matrix over every product pair, F over the
    # screened ones): `fit_adjoint` reverses `fit_M_stable` on D over every
    # product pair with F zero on the screened ones. Taking the SCF's own M
    # instead mixes two realizations, which differ in their near-null space
    # and move this force above the gradient's reproducibility floor.
    gram = product_pairs(mol, l_max_second=with_df.l_max_second)
    gmu, gnu, gw = gram
    ngram = len(gmu)
    require_whole_fit_adjoint(nao, naux, len(crd), ngram,
                              "isdf_exchange_skeleton(fit='replicated')")
    layout = test_set_layout(mol, crd, l_max_second=with_df.l_max_second)
    mu, nu, wc = layout
    cols = pair_positions(layout, gram, nao)
    D_test = test_set_D(mol, auxmol, crd, gram)
    V = auxmol.intor('int2c2e', aosym='s1')
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                                 aosym='s1').reshape(nao, nao, naux)
    F = np.zeros((naux, ngram + naux))
    F[:, cols] = np.linalg.solve(V, e3c[mu, nu, :].T) * wc[None, :]
    F[:, ngram:] = np.eye(naux)
    M = fit_M_stable(D_test, F, reg)
    X = mol.eval_gto('GTOval_sph', crd)

    # One channel per exchange operator, sharing the fit. The ISDF route keeps
    # the bare M and builds Z_w = M^T V_w M, which is what holds
    # K_SR + K_LR = K_bare, so a range-separated hybrid adds metrics and not
    # fits. W = X dm X^T is the same for every channel and is never rebuilt.
    X_bar = np.zeros((len(crd), nao))
    M_bar = np.zeros_like(M)
    V_bar = np.zeros_like(V)
    attenuated = []
    for omega, weight in channels:
        if weight == 0.0:
            continue
        if omega == 0.0:
            V_w = V
        else:
            with range_coulomb(mol, auxmol, omega):
                V_w = auxmol.intor('int2c2e', aosym='s1')
        xb, Z_bar = isdf_exchange_adjoints(X, M.T @ (V_w @ M), dm,
                                           prefactor * weight, dm_other)
        # Z = M^T V M: M sits on both sides of a symmetric adjoint, V between.
        X_bar += xb
        M_bar += 2.0 * (V_w @ M) @ Z_bar
        if omega == 0.0:
            V_bar += M @ Z_bar @ M.T
        else:
            attenuated.append((omega, M @ Z_bar @ M.T))
    Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, reg)

    # F's pair columns are w_c V^-1 (mu nu|P), so they reach the metric a
    # second time and the three-centre integrals once; its screened columns
    # are zero and its auxiliary block the identity, and neither
    # contributes.
    VinvFb = np.linalg.solve(V, F_bar[:, cols])
    V_bar -= F[:, cols] @ VinvFb.T
    G3 = np.zeros_like(e3c)
    np.add.at(G3, (mu, nu), (VinvFb * wc[None, :]).T)

    # One collocation adjoint for both slots: the exchange's own X and the
    # test set's AO pairs are the same evaluation on the same points.
    ao_bar = X_bar.copy()
    pair_bar = Dtest_bar[:, :ngram] * gw[None, :]
    np.add.at(ao_bar.T, gmu, (pair_bar * X[:, gnu]).T)
    np.add.at(ao_bar.T, gnu, (pair_bar * X[:, gmu]).T)

    # The attenuated metrics reach the fit nowhere else -- F is built on the
    # bare operator -- so they contribute through their own derivative alone.
    grad = (two_centre_adjoint(auxmol, V_bar)
            + three_centre_adjoint(mol, auxmol, G3))
    for omega, vb in attenuated:
        grad = grad + two_centre_adjoint(auxmol, vb, omega=omega)
    c_ao, P = basis_centre_forces(mol, crd, ao_bar)
    c_aux, P_aux = basis_centre_forces(auxmol, crd, Dtest_bar[:, ngram:])
    grad += c_ao + c_aux
    pts_local, owner = point_layout(mol, with_df.grid_radii,
                                    with_df.grid_origins)
    return grad + point_chain(mol, P + P_aux, pts_local, owner,
                              frames=atomic_frames(mol)[0], with_frames=True)


def _exchange_skeleton_rows(mf, dm, dm_other, prefactor, channels):
    """`isdf_exchange_skeleton` on the row fit (`skeleton_tiles`), rank 0's
    on every rank.

    The grid is the mean field's own, from its placed points or, where the
    distributed SCF left its ISDFJK unbuilt, from the same `isdf_grid` call
    that SCF's handle placed its points with; the handle's M^T and
    collocation tiles are reused rather than refitted. The pair layout is
    the row fit's own screen at these points.

    Inside a force's `one_fit_adjoint` window the fit adjoint is not run
    here: the seeds go to the window and this returns the attenuated
    metrics' term alone, the rest arriving with the assembly's one call; a
    mean-field skeleton that call contracted ahead is read from it. Where
    every operator's channels sum to a zero weight -- a pure functional's
    exchange, or Sigma_x - v_xc on Hartree-Fock -- E_K is zero and nothing
    is run.
    """
    weights = {}
    for omega, weight in channels:
        weights[omega] = weights.get(omega, 0.0) + weight
    if not any(weight != 0.0 for weight in weights.values()):
        return np.zeros((mf.mol.natm, 3))
    pending = pending_fit_adjoint(mf)
    if pending is not None:
        ahead = pending.take((dm, dm_other, prefactor, channels))
        if ahead is not None:
            return ahead
    key, chain, mt, X = _exchange_rows_fit(mf)
    seeds, two, _ = isdf_exchange_seeds(
        key.mol, key.auxmol, key.coords, key.layout, dm, dm_other=dm_other,
        prefactor=prefactor, channels=channels, mt=mt, X=X, block=key.block,
        **key.settings)
    if pending is not None and pending.open:
        pending.deposit(key, chain, seeds)
        return lockstep(two)
    adjoint = key.contract([seeds])[0]
    centre = adjoint.fit_centre + adjoint.coll_centre + two
    return lockstep(centre + chain(key.mol, adjoint.fit_points
                                   + adjoint.coll_points))


def _exchange_rows_fit(mf):
    """(FitKey, PointChain, M^T tiles, X tiles) of the row fit the ISDF-K
    mean field's exchange skeleton differentiates: its grid, the distributed
    SCF's tiles where it carries them (None elsewhere), and its points'
    chain, which follows the atomic frames of each geometry."""
    with_df = mf.with_df
    mol = with_df.mol
    handle = isdf_scf_handle(mf)
    if with_df.coords is not None and with_df.grid_radii is not None:
        coords, radii, origins = (with_df.coords, with_df.grid_radii,
                                  with_df.grid_origins)
    else:
        coords, radii, origins = lockstep(isdf_grid(
            mol, counts=with_df.counts if with_df._named_counts else None,
            radii=with_df.radii, auxbasis=with_df.auxbasis,
            n_start=with_df.n_start, return_info=True))
    if handle is not None and not np.array_equal(handle.coords, coords):
        raise RuntimeError(
            "the distributed SCF's interpolation points are not this mean "
            "field's grid, so its fit is not the one the energy used")
    auxmol = handle.auxmol if handle is not None else _auxmol_of(mf)
    block = handle.tile if handle is not None else FIT_CHOLESKY_BLOCK
    layout = screened_layout(mol, coords, l_max_second=with_df.l_max_second,
                             block=block)
    key = FitKey(mol, auxmol, coords, layout, block,
                 l_max_second=with_df.l_max_second,
                 regularization=with_df.regularization,
                 block_memory_gb=with_df.block_memory_gb)
    pts_local, owner = point_layout(mol, radii, origins)
    chain = PointChain(pts_local, owner, atomic_frames(mol)[0], True)
    if handle is None:
        return key, chain, None, None
    return key, chain, handle.MT, handle.X


def mean_field_exchange_wanted(mf):
    """Whether the ISDF-K mean-field force of `mf` carries an interpolated
    exchange skeleton: a hybrid's, the one predicate that force and the
    pending call's stash both read."""
    return (isinstance(getattr(mf, 'with_df', None), ISDFJK)
            and xc_hybrid_coeff(mf)[1] != 0.0)


def _mean_field_exchange_target(mf, keys):
    """The mean-field force's exchange skeleton as a target of a pending
    call -- its key, chain, seeds, attenuated term and the request it
    answers -- where its fit is among `keys`; None otherwise."""
    if not mean_field_exchange_wanted(mf):
        return None
    key, chain, mt, X = _exchange_rows_fit(mf)
    if key not in keys:
        return None
    dm = mf.make_rdm1()
    channels = exchange_channels(mf)
    seeds, two, _ = isdf_exchange_seeds(
        key.mol, key.auxmol, key.coords, key.layout, dm, prefactor=1.0,
        channels=channels, mt=mt, X=X, block=key.block, **key.settings)
    return {'key': key, 'chain': chain, 'seeds': [seeds], 'role': 'mean_field',
            'two': two, 'request': (dm, None, 1.0, channels)}


def _same_request(a, b):
    """Whether two exchange-skeleton requests (dm, dm_other, prefactor,
    channels) ask for the same number."""
    (dm_a, other_a, p_a, ch_a), (dm_b, other_b, p_b, ch_b) = a, b
    if (other_a is None) != (other_b is None):
        return False
    return (np.array_equal(dm_a, dm_b)
            and (other_a is None or np.array_equal(other_a, other_b))
            and float(p_a) == float(p_b)
            and [tuple(map(float, c)) for c in ch_a]
            == [tuple(map(float, c)) for c in ch_b])


def pending_fit_adjoint(mf):
    """The `PendingFitAdjoint` of the force this thread assembles on `mf`,
    or None."""
    pending = getattr(_PENDING, 'current', None)
    return pending if pending is not None and pending.mf is mf else None


@contextmanager
def one_fit_adjoint(mf, mean_field=False):
    """A window in which the exchange skeletons of one force on `mf` leave
    their row-fit adjoint seeds for its assembly, which contracts them with
    its own in one `fit_rows_adjoints` call per fit (`PendingFitAdjoint`).

    Opened around every skeleton that reaches a nuclear gradient and that
    gradient itself; a window already open on `mf` is joined. The first
    `PendingFitAdjoint.contract` or `settle` closes it, so a skeleton called
    after the assembly runs its own adjoint again. mean_field: the mean
    field's own force follows inside this block, and its exchange skeleton
    rides the same call, read back when that force asks for it. Leaving the
    block with seeds no assembly contracted raises: that force would miss
    them.
    """
    outer = getattr(_PENDING, 'current', None)
    if outer is not None and outer.mf is mf and outer.open:
        outer.mean_field = outer.mean_field or bool(mean_field)
        yield outer
        return
    pending = PendingFitAdjoint(mf, mean_field)
    _PENDING.current = pending
    try:
        yield pending
    finally:
        _PENDING.current = outer
    if pending.deposits:
        raise RuntimeError(
            f'{len(pending.deposits)} exchange skeleton(s) left row-fit '
            'adjoint seeds that no nuclear assembly contracted: the force '
            'would miss their fit and collocation terms')
    if pending.stash is not None:
        warnings.warn('the mean-field exchange skeleton was contracted ahead '
                      'of a force that never asked for it', RuntimeWarning)


def fock_partial_skeleton(mf, gamma, nocc):
    """(natm, 3) two-electron skeleton for a folded Fock partial, tensor-free.

    Folding Tr[gamma F] gives the separable two-particle density

        Gamma_abcd = gamma_ab D_cd - (1/2) gamma_ac D_bd,   D the SCF density,

    whose four-permutation sum contracts against derivative integrals as
    Coulomb- and exchange-shaped terms only,

        dE_2/dR_A = sum_{a in A} [ 2 (gamma.vj[D] + D.vj[gamma])
                                   - (gamma.vk[D] + D.vk[gamma]) ]_a

    gamma and D enter E_2 symmetrically, which forces one coefficient per
    channel. A free least-squares fit cannot recover them, since the four
    contractions are linearly dependent and return unphysical weights that
    still fit exactly. Constraining by that symmetry gives 2 and -1 exactly.

    A density-fitted mean field is routed to `fock_partial_skeleton_df`
    instead, and must be. pyscf's conventional and density-fitted derivative
    J/K do not share a convention, and feeding the latter through the weights
    below gives a gradient large enough to ruin a geometry while looking
    plausible.
    """
    is_ks, a_x = xc_hybrid_coeff(mf)
    if getattr(mf, 'with_df', None) is not None:
        # An interpolated mean field splits the two halves. Its Coulomb is
        # fitted and integral-direct, so the fitted skeleton differentiates it.
        # Its exchange comes from the ISDF factors, which the auxiliary-basis
        # skeleton knows nothing about.
        if isinstance(mf.with_df, ISDFJK):
            g = fock_partial_skeleton_df(mf, _auxmol_of(mf), gamma, nocc,
                                         channels=(), coulomb=True)
            g = g + isdf_fock_partial_exchange(mf, gamma)
        else:
            g = fock_partial_skeleton_df(mf, _auxmol_of(mf), gamma, nocc,
                                         channels=exchange_channels(mf))
        return (g + (xc_skeleton(mf, gamma) if is_ks else 0.0)
                + reaction_field_skeleton(mf, gamma))
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    vj_D = grad_rhf.get_jk(mf.mol, D)[0]
    vj_g = grad_rhf.get_jk(mf.mol, g_ao)[0]
    # The Coulomb half is never range-separated and is built once. The exchange
    # half is one build per channel, so a range-separated hybrid carries its
    # erf-attenuated term instead of dropping it.
    t = 2.0 * (np.einsum('ab,xab->xa', g_ao, vj_D, optimize=True)
               + np.einsum('ab,xab->xa', D, vj_g, optimize=True))
    grad = np.zeros((mf.mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
        grad[ia] = t[:, p0:p1].sum(axis=1)
    for omega, weight in (exchange_channels(mf) if is_ks else [(0.0, 1.0)]):
        if weight != 0.0:
            grad = grad + exchange_channel_skeleton(mf, g_ao, D, omega, weight)
    if is_ks:
        grad = grad + xc_skeleton(mf, gamma)
    return grad + reaction_field_skeleton(mf, gamma)


def xc_skeleton(mf, gamma):
    """(natm, 3) of d/dR Tr[gamma v_xc^DFT], MO coefficients held fixed and the
    Becke grid moving with the atoms, as the energy's grid does.

    The exchange-correlation half of the Kohn-Sham Fock partial. It appears
    once, not twice. The Coulomb and exchange terms come from a four-index
    object in which gamma and the density enter symmetrically, while v_xc is
    not bilinear in the density.

    The grid's own motion is part of it. The fixed-grid derivative (the AOs
    and the density moving through the kernel) misses the points riding with
    their atoms and the Becke weights' response, which is the whole of an
    excitation force's translation residual on a Kohn-Sham reference, larger
    on a range-separated functional than on a global hybrid;
    `skeleton_tiles.xc_grid_skeleton` carries all three terms.
    """
    if mf.do_nlc():
        raise NotImplementedError(
            f'{mf.xc!r} carries a VV10 kernel, whose skeleton is not built')
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return xc_grid_skeleton(mf.mol, mf.grids, mf._numint, mf.xc, g_ao,
                            np.asarray(mf.make_rdm1()))


def eps_chain_gradient(mf, eps_bar, nocc, Y_extra=None, verbose=False):
    """(natm, 3) of sum_pq gamma_pq dF_pq/dR, with no norb^4 tensor anywhere.

    The dense engine's Lagrangian, rebuilt on Coulomb/exchange builds. Fold the
    adjoint into a Fock partial, solve for the canonical-condition multiplier,
    read the orthonormality multiplier off the stationary orbital gradient, and
    contract the skeleton. `Y_extra` carries any orbital-rotation dependence
    that is not expressible as an (F, ERI) partial.

    eps_bar: the adjoint on the orbital energies, gamma = diag(eps_bar), or a
        full symmetric MO Fock partial gamma_pq = dE/dF_pq.
    """
    # cycle: grad_engine imports this module at module level.
    from src.gradients.grad_engine import one_electron_skeleton
    C = mf.mo_coeff
    eps_bar = np.asarray(eps_bar, float)
    gamma = np.diag(eps_bar) if eps_bar.ndim == 1 else 0.5 * (eps_bar + eps_bar.T)
    Y_E = fock_partial_Y(mf, gamma, nocc)
    if Y_extra is not None:
        Y_E = Y_E + Y_extra
    Lam, res = solve_lambda(mf, Y_E, nocc, verbose=verbose)
    Y_tot = Y_E + fock_partial_Y(mf, Lam, nocc)
    Mmat = -0.25 * (Y_tot + Y_tot.T)
    gamma_tot = gamma + Lam
    g2 = fock_partial_skeleton(mf, gamma_tot, nocc)
    g1 = one_electron_skeleton(mf.mol, mf, [C @ gamma_tot @ C.T],
                               [C @ Mmat @ C.T])[0]
    return g1 + g2, {'stationarity': np.abs(Y_tot - Y_tot.T).max(),
                     'multiplier_res': res}


def exchange_channel_skeleton(mf, g_ao, D, omega, weight):
    """(natm, 3) of weight * d/dR of one exchange channel, four-centre.

    omega = 0 is the ordinary full-range build, and a nonzero one is the
    erf-attenuated channel of a range-separated hybrid.

    The long-range channel is not density fitted, and that is forced. Its
    auxiliary metric (P|erf(omega r)/r|Q) goes rank-deficient, and the
    derivative of a truncated pseudo-inverse is not -V^-1 dV V^-1, so a fitted
    skeleton is wrong there at any threshold. Four-centre derivative integrals
    have no metric and no such failure, and the term costs one exchange build
    either way.

    `pyscf.grad.rhf.get_jk` is used directly, not `mf.Gradients()`, which on a
    fitted mean field returns the fitted derivative J/K.
    """
    mol = mf.mol
    with mol.with_range_coulomb(omega):
        vk_D = grad_rhf.get_jk(mol, D)[1]
        vk_g = grad_rhf.get_jk(mol, g_ao)[1]
    t = -weight * (np.einsum('ac,xac->xa', g_ao, vk_D, optimize=True)
                   + np.einsum('ad,xad->xa', D, vk_g, optimize=True))
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] = t[:, p0:p1].sum(axis=1)
    return grad


def fock_partial_skeleton_df(mf, auxmol, gamma, nocc, channels=((0.0, 1.0),),
                             coulomb=True):
    """(natm, 3) two-electron skeleton for a density-fitted mean field.

    pyscf's conventional derivative J/K and its density-fitted one do not share
    a convention, so this skeleton is built from the three- and two-centre
    adjoints directly, both gated against finite differences of the integrals.

    With (ab|cd) = sum_PQ J[ab,P] Vinv[P,Q] J[cd,Q] and the folded density
    Gamma_abcd = gamma_ab D_cd - gamma_ac D_bd / 2,

        E_2 = a^T Vinv b - (1/2) sum_PQ Vinv[P,Q] T[P,Q]
        a_P = sum_ab gamma_ab J[ab,P],   b_P = sum_cd D_cd J[cd,P]
        T_PQ = tr[gamma J^P D J^Q]      (symmetric)

    whose adjoints are

        Jbar[ab,P] = gamma_ab (Vinv b)_P + D_ab (Vinv a)_P - [gamma K^P D]_ab
        Vbar       = -(Vinv a)(Vinv b)^T + (1/2) Vinv T Vinv      (symmetrized)

    with K^P = sum_Q Vinv[P,Q] J^Q. Nothing of the (nao, nao, naux) shape is
    formed: `skeleton_tiles.fitted_fock_skeleton` streams the three-centre
    integrals and their derivatives over fixed auxiliary tiles, the tiles
    divided over the current ranks and their (natm, 3) addends reduced once.
    Every omega = 0 channel is one exchange at the channels' summed weight;
    a long-range one is the four-centre `exchange_channel_skeleton`, rank 0's
    on every rank. The exchange term costs naux nao^2 nocc, what
    density-fitted exchange costs anywhere.

    `channels` is `exchange_channels(mf)`, and a range-separated hybrid's
    second channel carries its own density fit. `coulomb=False` leaves the
    exchange alone, which is what the Sigma_x - v_xc correction needs. On a
    Kohn-Sham reference the XC half comes from `xc_skeleton`, which the caller
    adds.
    """
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    full = 0.0
    for omega, weight in channels:
        if omega == 0.0:
            full += weight
    occ = mf.mo_occ
    grad = fitted_fock_skeleton(
        mf.mol, auxmol, g_ao, D,
        occ=C[:, occ > 0] * np.sqrt(occ[occ > 0]) if full else None,
        coulomb=coulomb, exchange=full)
    for omega, weight in channels:
        if omega != 0.0 and weight != 0.0:
            # pyscf's threaded derivative K: one rank's bits on all of them
            grad = grad + lockstep(exchange_channel_skeleton(mf, g_ao, D,
                                                             omega, weight))
    return grad
