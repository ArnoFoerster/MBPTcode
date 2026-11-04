"""The pieces every ISDF fit adjoint shares: how a point adjoint reaches the
nuclei (`PointChain`, `point_chain`: the points translating with their atom
and turning with its local frame, or with the whole molecule's body frame),
the adjoints of the metric's root and of the two-centre integrals, a gauge's
seeds (`GaugeAdjoint`) and the ISDF exchange form's adjoints on X and Z.

The fit adjoints of both test sets (`isdf_derivatives`,
`ao_product_fit_adjoint`) and the tiled skeletons build on them, so they sit
below all three. Derivative-integral signs as in `isdf_derivatives`: pyscf's
int2c2e_ip1 is +(grad .), so the nuclear derivative carries a minus.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.geometry.geometry import grid_key
from src.Base.numerics.constants import AUX_METRIC_ROOT_FLOOR
from src.Base.factorization.shell_runs import functions_to_atoms
from src.Base.factorization.separable_ri import (FRAME_DECAY, atomic_frames,
                                                 atomic_points,
                                                 metric_eigensystem)


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
        sym = grid_key(mol, ia)
        use_origin = (origin_by_element or {}).get(sym, False)
        p = atomic_points(radii_by_element[sym], centre=(0.0, 0.0, 0.0),
                          origin=use_origin)
        pts_local.append(p)
        owner.append(np.full(len(p), ia, dtype=int))
    return pts_local, np.concatenate(owner)


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


def two_centre_rows(auxmol, V_rows, shells=None, omega=None):
    """(3, rows) of -2 sum_Q (grad P|Q) V_rows[P, Q] for the auxiliary
    functions P of the auxiliary shells shells = (k0, k1) (None: all), V_rows
    their rows of a symmetric V_bar: the per-function gradient of
    sum_PQ V_bar[P,Q] (P|Q), the one two-centre adjoint.

    pyscf's int2c2e_ip1 is +(grad P|Q), so the nuclear derivative is minus it,
    and both index positions contribute equally once V_bar is symmetric.
    omega: the operator, set on auxmol while the integrals are made (nonzero:
    the erf-attenuated long-range channel of a range-separated hybrid); None
    takes the one auxmol carries.
    """
    sl = None if shells is None else (shells[0], shells[1], 0, auxmol.nbas)
    if omega is None:
        v1 = auxmol.intor('int2c2e_ip1', comp=3, shls_slice=sl)
    else:
        with auxmol.with_range_coulomb(omega):
            v1 = auxmol.intor('int2c2e_ip1', comp=3, shls_slice=sl)
    return -2.0 * np.einsum('xPQ,PQ->xP', v1, V_rows, optimize=True)


def two_centre_adjoint(auxmol, V_bar, omega=0.0):
    """(natm, 3) gradient of sum_PQ V_bar[P,Q] (P|Q), V_bar symmetrized here
    (`two_centre_rows` folded onto the atoms); omega != 0 the erf-attenuated
    operator's."""
    Vb = 0.5 * (V_bar + V_bar.T)
    return functions_to_atoms(two_centre_rows(auxmol, Vb, omega=omega), auxmol)


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
