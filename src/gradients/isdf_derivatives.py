"""Nuclear derivatives of the ISDF factors, as adjoint contractions.

The space-time route returns sensitivities with respect to the factors,
(eps_bar, X_bar, D_bar), and this module turns the X and D parts into forces.
The derivative tensor dX/dR is never formed, only its contraction against a
given adjoint.

On the row-distributed fit both parts run in the fit's own tiles
(`row_fit_adjoint` over `isdf_fit_adjoint.fit_rows_adjoints`, and X_mo^T X_bar by
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
a reference and differentiates that. Frozen frames turn with the whole
molecule instead: F_i = F0_i Q, with Q the Kabsch rotation of the geometry
onto the reference (`body_frame.BodyFrame`), so a rigid rotation does not move
the molecule against its own interpolation clouds. The point chain carries
dE/dQ = sum_i F0_i^T dE/dF_i through the rotation's derivative, so the force
is the derivative of the energy and carries no net torque.

Derivative-integral signs are measured, not transcribed: pyscf's int2c2e_ip1,
int3c2e_ip1 and int3c2e_ip2 are all +(grad .), so every nuclear derivative
built from them carries a minus.
"""
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from src.Base.numerics.constants import (AUX_METRIC_ROOT_FLOOR, FIT_CHOLESKY_BLOCK,
                                         FIT_REALIZATIONS,
                                         THREE_CENTER_BLOCK_BYTES,
                                         WHOLE_FIT_ADJOINT_MAX_GB)
import scipy.linalg
from pyscf import df as pyscf_df

from src.Base.factorization.distributed_df import distributed_handles
from src.Base.factorization.distributed_isdf_jk import DistributedISDFJK
from src.environments.environment import dresses_interaction
from src.Base.factorization.isdf_jk import ISDFJK, isdf_grid
from src.Base.factorization.separable_ri import (ANGULAR_WEIGHTS, FRAME_DECAY, ao_l_labels,
                                                 content_key, atomic_frames, atomic_points,
                                                 fit_settings, aux_metric_sqrt, fit_M_stable,
                                                 metric_eigensystem, rows_transpose_product,
                                                 screened_layout, fit_test_set_D,
                                                 fit_test_set_layout)
from src.gradients.isdf_fit_adjoint import AdjointSeeds, GaugeSeeds, fit_rows_adjoints
from src.gradients.skeleton_tiles import isdf_exchange_seeds, shell_runs
from src.Base.factorization.sliced_factors import SlicedFactors
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.Base.pyscf.pyscf_interface import exchange_channels

#: The row-fit adjoint pending on this thread (`one_fit_adjoint`): each
#: simulated rank is a thread assembling its own force.
_PENDING = threading.local()


class FitKey:
    """What a row fit's adjoint is a function of, by content: the molecule,
    the auxiliary basis, the points, the pair layout, the tile edge and the
    estimator's settings. Seeds of one key share one
    `isdf_fit_adjoint.fit_rows_adjoints` call (`contract`)."""

    def __init__(self, mol, auxmol, coords, layout, block, numerics=None):
        l_max_second, regularization, block_memory_gb = fit_settings(
            numerics, 'l_max_second', 'regularization', 'block_memory_gb')
        self.mol, self.auxmol = mol, auxmol
        self.coords, self.layout = np.asarray(coords), layout
        self.block = int(block)
        self.numerics = numerics
        self.settings = dict(l_max_second=int(l_max_second),
                             regularization=float(regularization),
                             block_memory_gb=float(block_memory_gb))
        self.key = (content_key(mol), content_key(auxmol),
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
                                 numerics=self.numerics)


class PointChain:
    """How a point adjoint P[g] = dE/dr_g reaches the nuclei for one grid:
    r_g = p_g F_i + R_i over the clouds `pts_local` and their owners, the
    frames F_i turning with the geometry (`with_frames`), or frozen and
    turned with the whole molecule by a `body` frame (`point_chain`)."""

    def __init__(self, pts_local, owner, frames, with_frames,
                 decay=FRAME_DECAY, body=None):
        self.pts_local = [np.asarray(p) for p in pts_local]
        self.owner = np.asarray(owner)
        self.frames = None if frames is None else np.asarray(frames)
        self.with_frames, self.decay = bool(with_frames), float(decay)
        self.body = None if self.with_frames else body

    def same(self, other):
        """Whether `other` takes a point adjoint to the same force."""
        if (self.with_frames != other.with_frames or self.decay != other.decay
                or len(self.pts_local) != len(other.pts_local)
                or not np.array_equal(self.owner, other.owner)
                or not all(np.array_equal(a, b) for a, b in
                           zip(self.pts_local, other.pts_local))):
            return False
        if not self.with_frames and self.body is None:
            return other.body is None
        if self.frames is None or other.frames is None:
            return self.frames is None and other.frames is None
        return (np.array_equal(self.frames, other.frames)
                and (self.body is None or self.body.same(other.body)))

    def __call__(self, mol, P):
        return point_chain(mol, P, self.pts_local, self.owner,
                           frames=self.frames, decay=self.decay,
                           with_frames=self.with_frames, body=self.body)


class PendingFitAdjoint:
    """The row fit's adjoint seeds the exchange skeletons of one force leave
    for its assembly instead of contracting them (`one_fit_adjoint`).

    A skeleton inside the window deposits its seeds (`deposit`) and returns
    what reaches the nuclei outside the fit. The assembly contracts them with
    its own fit adjoint (`contract`, from `row_fit_adjoint`) in one
    `fit_rows_adjoints` call per fit: deposits on the same fit and point chain
    pool into one target, their seeds summed in deposit order
    (`isdf_fit_adjoint.summed_seeds`), the chain's own first where its points
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

    def discard(self):
        """Drop the deposits uncontracted and close the window: the caller
        wanted the orbital branch's inputs only (a relaxed density), so no
        force is assembled and nothing of the fit reaches the nuclei."""
        self.deposits = []
        self.open = False

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


def continued_frames(mol, frames_ref, decay=FRAME_DECAY):
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


def frame_adjoint(mol, W, decay=FRAME_DECAY, frames=None, degenerate=None):
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
                decay=FRAME_DECAY, with_frames=True, body=None):
    """(natm, 3) from an adjoint P[g] = dE/dr_g on the interpolation points.

    The points translate with the atom that owns them and turn with that atom's
    frame: r_g = p_g F_i + R_i, so dE/dF_i[a,b] = sum_{g in i} p_g[a] P[g,b].
    Continued frames turn with each atom's environment (`frame_adjoint`);
    frozen ones, F_i = F0_i Q, with the body frame Q of the whole molecule
    (`body_frame.BodyFrame`), so dE/dQ = sum_i F0_i^T dE/dF_i.
    """
    grad = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        grad[ia] += P[atom_of_point == ia].sum(axis=0)
    if with_frames or body is not None:
        W = np.zeros((mol.natm, 3, 3))
        for ia in range(mol.natm):
            W[ia] = pts_local[ia].T @ P[atom_of_point == ia]
    if with_frames:
        grad += frame_adjoint(mol, W, decay=decay, frames=frames)
    elif body is not None:
        q_bar = np.einsum('iak,iab->kb', np.asarray(frames), W)
        grad += body.rotation_adjoint(mol.atom_coords(), q_bar)
    return grad


def collocation_adjoint(mol, coords, Xao_bar, pts_local, atom_of_point,
                        frames=None, decay=FRAME_DECAY, with_frames=True,
                        body=None):
    """(natm, 3) gradient of sum_{g,mu} Xao_bar[g,mu] chi_mu(r_g).

    The orbital-basis case: AO centres plus the shared point chain.
    `Xao_bar` is the adjoint on the AO-basis collocation; from an adjoint on
    X_mo = X_ao C it is X_bar C^T.
    """
    centre, P = basis_centre_forces(mol, coords, Xao_bar)
    return centre + point_chain(mol, P, pts_local, atom_of_point, frames=frames,
                                decay=decay, with_frames=with_frames,
                                body=body)


# ---------------------------------------------------------------------------
# the fit: adjoint of the regularized least-squares estimator
# ---------------------------------------------------------------------------

def fit_adjoint(D, F, M_bar, numerics=None):
    """(D_bar, F_bar) of sum_{beta k} M_bar[beta,k] M[beta,k], M from `fit_M`.

    `fit_M` row-balances D, forms a regularized Gram matrix and solves

        s_k = ||D[k,:]||,   d = 1/s,   Dt = d D
        G   = Dt Dt^T + reg I
        M   = (F Dt^T) G^-1 d

    so the reverse pass is that read backwards. With B = A G^-1 (A = F Dt^T)
    and A_bar = B_bar G^-1, the inverse contributes

        G_bar = -G^-1 A^T B_bar G^-1 = -B^T A_bar,

    low rank and formed from those two single solves -- the row fit's
    G_bar = -W Z^T (`isdf_fit_adjoint.fit_rows_adjoints`). No dM/dR is ever
    formed, and G is the matrix the forward pass already factorized.

    Nothing here forms G^-1. G is floored by `regularization` and reaches a
    large condition number on a large basis. Applying G^-1 to the
    (nk, nk) product A^T B_bar and again to its result carries a last-bit
    change of M_bar about 1e4 times further than the low-rank form does, as
    on the row fit (tests/test_fit_adjoint_stability.py), and an explicit
    inverse loses most of double precision. G is symmetric positive definite,
    so one Cholesky factorization serves every solve.
    """
    regularization, = fit_settings(numerics, 'regularization')
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


# ---------------------------------------------------------------------------
# the grid itself: the fit error differentiated in the interpolation points
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the D factor: D = M^T V^(1/2), all the way back to the nuclei
# ---------------------------------------------------------------------------

def sqrtm_adjoint(V, Vh_bar, tol=AUX_METRIC_ROOT_FLOOR):
    """V_bar for an adjoint on V^(1/2), V symmetric positive semidefinite.

    The Frechet derivative L of the square root solves Vh L + L Vh = E, so in
    the eigenbasis it is the Hadamard multiplier 1/(sqrt(w_i) + sqrt(w_j)).
    That multiplier is symmetric, hence the map is self-adjoint and the reverse
    pass is the same operation. The truncation is the forward's
    (`separable_ri.metric_spectrum`, which drops eigenvalues below
    AUX_METRIC_ROOT_FLOOR x max before taking the root), so energy and force
    keep one spectrum.
    """
    entry = metric_eigensystem(V)      # the root's own, when it was formed
    w, U = entry['w'], entry['v']
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
        for sh0, sh1 in shell_runs(mol.ao_loc_nr(), max_bytes // int(per_ao)):
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
                    atom_of_point, frames=None, decay=FRAME_DECAY,
                    numerics=None, with_frames=True,
                    environment=None, body=None):
    """(natm, 3) gradient of sum_{gP} D_bar[g,P] D[g,P], D = M^T V_env^(1/2):
    one gauge of `dfactor_adjoint_gauges`. A test gate, read by no
    production route."""
    return dfactor_adjoint_gauges(mol, auxmol, coords,
                                  [(D_bar, environment)], layout, pts_local,
                                  atom_of_point, frames=frames, decay=decay,
                                  numerics=numerics,
                                  with_frames=with_frames, body=body)


def product_pairs(mol, numerics=None):
    """(mu, nu, weight) of every pair of the test set's product basis, (all
    AOs) x (AOs with l <= l_max_second), in `fit_test_set_layout`'s order.

    These are the columns the Gram matrix of `fit_M_streaming` and
    `fit_rows` sums over, screened or not: S = (A A^T) o (B B^T) is the
    product form of D D^T over all of them, while F D^T keeps the screened
    ones. A frozen `fit_test_set_layout` is a subsequence of this list.
    """
    l_max_second, = fit_settings(numerics, 'l_max_second')
    l_ao = ao_l_labels(mol)
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
            f'`isdf_fit_adjoint.fit_rows_adjoint`.')
    return gb


def dfactor_adjoint_gauges(mol, auxmol, coords, gauges, layout, pts_local,
                           atom_of_point, frames=None, decay=FRAME_DECAY,
                           numerics=None,
                           with_frames=True, gram_layout=None, body=None):
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
    D_test = fit_test_set_D(mol, auxmol, coords, (gmu, gnu, gw))
    M = fit_M_stable(D_test, F, numerics)

    V_bar = np.zeros_like(V)
    ao_bar = np.zeros_like(ao)
    aux_bar = np.zeros((len(coords), auxmol.nao_nr()))
    env_grad = None
    for gauge in gauges:
        gauge = gauge if isinstance(gauge, GaugeAdjoint) else GaugeAdjoint(*gauge)
        D_bar, environment = gauge.d_bar, gauge.environment
        dressed = dresses_interaction(environment, auxmol)
        # a bare gauge's root V^(1/2) takes an adjoint as a dressed one's
        # does (the split partition's static half on the bare interaction);
        # only a vtilde adjoint has nowhere to land
        if not dressed and gauge.kernel_bar is not None:
            raise ValueError(
                f'{environment!r} dresses nothing, so this gauge has no '
                'vtilde for a kernel adjoint to land on; it belongs to the '
                'gauge that carries the environment')
        V_gauge = V + environment.aux_kernel(auxmol) if dressed else V
        Vh = aux_metric_sqrt(auxmol, environment, V=V)

        M_bar = Vh @ D_bar.T                               # (naux, nk)
        Vh_bar = M @ D_bar                                 # (naux, naux)
        if gauge.root_bar is not None:
            Vh_bar = Vh_bar + gauge.root_bar
        Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, numerics)

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
                        decay=decay, with_frames=with_frames, body=body)
    return grad


def row_fit_adjoint(mol, auxmol, coords, d_bar, x_bar, mo_coeff, layout,
                    pts_local, atom_of_point, frames=None, decay=FRAME_DECAY,
                    with_frames=True, block=None, pending=None, body=None,
                    gauges=()):
    """(g_collocation, g_fit, held): the collocation branch of X_bar and the
    fit branch of D_bar on the row-distributed fit, `fit_rows`' estimator on
    the frozen `layout` (its Gram matrix over every product pair).

    d_bar is the BARE gauge's adjoint (None where no target reads it);
    gauges: `GaugeAdjoint`s of dressed factors on the same fit, each with
    the environment that dresses its metric root, which ride the same call
    as `isdf_fit_adjoint.GaugeSeeds` -- `dfactor_adjoint_gauges`' gauges, in
    the fit's tiles. A gauge whose environment dresses nothing belongs in
    d_bar and is refused here.

    `isdf_fit_adjoint.fit_rows_adjoints` runs both in the fit's tiles and
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
    chain = PointChain(pts_local, atom_of_point, frames, with_frames, decay,
                       body=body)
    dressed = []
    for g in gauges:
        if not dresses_interaction(g.environment, auxmol):
            raise ValueError(
                f'{g.environment!r} dresses nothing: its adjoint is on the '
                'bare gauge and belongs in d_bar')
        dressed.append(GaugeSeeds(g.d_bar, g.environment,
                                  root_bar=g.root_bar,
                                  kernel_bar=g.kernel_bar))
    seeds = AdjointSeeds(d_bar=d_bar, x_bar=x_bar, mo_coeff=mo_coeff,
                         gauges=dressed)
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


def auxmol_of(mf):
    """The auxiliary molecule behind a density-fitted mean field."""
    aux = getattr(mf.with_df, 'auxmol', None)
    if aux is None:
        aux = pyscf_df.addons.make_auxmol(mf.mol, auxbasis=mf.with_df.auxbasis)
    return aux


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


def exchange_fit(fit=None):
    """The realization of the fit an ISDF exchange skeleton differentiates,
    both of the SCF's estimator: `fit` as given, else 'rows', the row fit's
    tiles on any rank count (the distributed ISDF-K SCF's own). The whole form
    ('replicated') differentiates the same estimator through the dense Gram
    matrix, whose conditioning holds it to 1e-8 of the SCF energy's derivative
    on ethylene/cc-pVDZ PBE0 (one reassociation of it moves the force 2.1e-8)
    where the rows reach 5e-10."""
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
    fit = exchange_fit(fit)
    dm = mf.make_rdm1() if dm is None else np.asarray(dm)
    channels = exchange_channels(mf) if channels is None else list(channels)
    if fit == 'rows':
        return _exchange_skeleton_rows(mf, dm, dm_other, prefactor, channels)
    with_df = mf.with_df
    require_placed_grid(with_df, 'the ISDF exchange skeleton')
    mol, auxmol, crd = with_df.mol, with_df.auxmol, with_df.coords
    nao, naux = mol.nao_nr(), auxmol.nao_nr()

    # The fit is rebuilt, not taken from the mean field, so that the function
    # differentiated is a single realization of the SCF's estimator
    # (`fit_M_streaming`: the Gram matrix over every product pair, F over the
    # screened ones): `fit_adjoint` reverses `fit_M_stable` on D over every
    # product pair with F zero on the screened ones. Taking the SCF's own M
    # instead mixes two realizations, which on water/cc-pVDZ differ by
    # 1.1e-08 relative in the fit's near-null space and move this force by
    # 3.5e-08 Ha/Bohr on Hartree-Fock, above the 1e-8 floor.
    gram = product_pairs(mol, numerics=with_df.numerics)
    gmu, gnu, gw = gram
    ngram = len(gmu)
    require_whole_fit_adjoint(nao, naux, len(crd), ngram,
                              "isdf_exchange_skeleton(fit='replicated')")
    # the pair columns the SCF's fit kept: a walk's frozen layout, else the
    # screen at these points
    layout = (with_df.pair_layout if with_df.pair_layout is not None else
              fit_test_set_layout(mol, crd, numerics=with_df.numerics))
    mu, nu, wc = layout
    cols = pair_positions(layout, gram, nao)
    D_test = fit_test_set_D(mol, auxmol, crd, gram)
    V = auxmol.intor('int2c2e', aosym='s1')
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                                 aosym='s1').reshape(nao, nao, naux)
    F = np.zeros((naux, ngram + naux))
    F[:, cols] = np.linalg.solve(V, e3c[mu, nu, :].T) * wc[None, :]
    F[:, ngram:] = np.eye(naux)
    M = fit_M_stable(D_test, F, with_df.numerics)
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
            with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
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
    Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, with_df.numerics)

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
    return grad + mean_field_point_chain(with_df, mol, pts_local,
                                         owner)(mol, P + P_aux)


def _exchange_skeleton_rows(mf, dm, dm_other, prefactor, channels):
    """`isdf_exchange_skeleton` on the row fit (`skeleton_tiles`), rank 0's
    on every rank.

    The grid is the mean field's own, from its placed points or, where the
    distributed SCF left its ISDFJK unbuilt, from the same `isdf_grid` call
    that SCF's handle placed its points with; the handle's M^T and
    collocation tiles are reused rather than refitted. The pair layout is
    the one the SCF's fit kept: a walk's frozen layout (`ISDFJK.pair_layout`),
    else the row fit's own screen at these points.

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
        numerics=key.numerics)
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
    chain, which follows the atomic frames of each geometry, or the frozen
    ones the fit was placed on (`mean_field_point_chain`)."""
    with_df = mf.with_df
    require_placed_grid(with_df, 'the ISDF exchange skeleton')
    mol = with_df.mol
    handle = isdf_scf_handle(mf)
    if with_df.coords is not None:
        coords, radii, origins = (with_df.coords, with_df.grid_radii,
                                  with_df.grid_origins)
    else:
        coords, radii, origins = lockstep(isdf_grid(
            mol, counts=with_df.counts if with_df.named_counts else None,
            radii=with_df.radii, auxbasis=with_df.auxbasis,
            n_start=with_df.n_start, return_info=True,
            frames=with_df.frozen_frames,
            runtime_radii=with_df.numerics.isdf.runtime_radii))
    if handle is not None and not np.array_equal(handle.coords, coords):
        raise RuntimeError(
            "the distributed SCF's interpolation points are not this mean "
            "field's grid, so its fit is not the one the energy used")
    auxmol = handle.auxmol if handle is not None else auxmol_of(mf)
    block = handle.tile if handle is not None else FIT_CHOLESKY_BLOCK
    layout = (with_df.pair_layout if with_df.pair_layout is not None else
              screened_layout(mol, coords, numerics=with_df.numerics,
                              block=block))
    key = FitKey(mol, auxmol, coords, layout, block,
                 numerics=with_df.numerics)
    pts_local, owner = point_layout(mol, radii, origins)
    chain = mean_field_point_chain(with_df, mol, pts_local, owner)
    if handle is None:
        return key, chain, None, None
    return key, chain, handle.MT, handle.X


def mean_field_point_chain(with_df, mol, pts_local, owner):
    """The `PointChain` of an ISDF-K mean field's points: the covariant
    frames of `mol`, turning with each atom's environment, or the frames its
    fit was placed on inside `isdf_jk.frozen_frames`, turned by their body
    frame alone."""
    frozen = getattr(with_df, 'frozen_frames', None)
    if frozen is None:
        return PointChain(pts_local, owner, atomic_frames(mol)[0], True)
    return PointChain(pts_local, owner, frozen.frames, False,
                      body=frozen.body)


def require_placed_grid(with_df, what):
    """Raise if the ISDF factorization's points were handed in from outside
    (`coords` set, `grid_radii` None): the point chain needs the radii that
    placed them and which atom owns which point, and neither is known."""
    if with_df.coords is not None and with_df.grid_radii is None:
        raise NotImplementedError(
            f'{what} needs the interpolation points as atom-local clouds, and '
            f'this factorization was handed its `coords` from outside, so the '
            f'radii that placed them -- and which atom owns which point -- are '
            f'unknown. Let ISDFJK.build pick the grid.')


def require_fitted_coulomb(mf, what):
    """Raise if `mf` builds J from the interpolation (`j_route` not
    'df-direct'): every ISDF-K skeleton takes the Coulomb derivative from the
    density-fitted one, which is the derivative of another function then."""
    with_df = getattr(mf, 'with_df', None)
    if isinstance(with_df, ISDFJK) and with_df.j_route != 'df-direct':
        raise NotImplementedError(
            f'{what} takes the Coulomb derivative from the density-fitted '
            f'skeleton, which is the right function only for '
            f"j_route='df-direct'; this mean field builds J from the "
            f'interpolation (j_route={with_df.j_route!r}), whose derivative is '
            'not built. That route is measurably unsafe in an SCF anyway -- '
            'see ISDFJK.build.')


def mean_field_exchange_wanted(mf):
    """Whether the ISDF-K mean-field force of `mf` carries an interpolated
    exchange skeleton: a hybrid's, the one predicate that force and the
    pending call's stash both read. Every exchange channel counts, so a
    screened hybrid with no long-range exact exchange (alpha = 0) still has
    its short-range term."""
    return (isinstance(getattr(mf, 'with_df', None), ISDFJK)
            and any(w != 0.0 for _, w in exchange_channels(mf)))


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
        channels=channels, mt=mt, X=X, block=key.block,
        numerics=key.numerics)
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
