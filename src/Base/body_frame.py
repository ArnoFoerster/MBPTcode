"""The orientation of a molecule against a reference geometry, and its derivative.

A convention frozen at a reference geometry R0 that carries a direction -- the
local frames the ISDF interpolation clouds are turned by -- is a function of
the geometry only if it turns with the molecule. Held in the lab frame it is
not: a rigid rotation then moves the molecule against it, the energy changes
under a motion that changes no physics, and its gradient carries a net torque
that an optimizer follows.

The body frame is the proper rotation Q that superposes R0 on R in the least-
squares sense (Kabsch, Acta Cryst. A 32, 922 (1976); A 34, 827 (1978)),

    Q = argmax_{Q in SO(3)} < Q, A >,   A = (R0 - c0)^T (R - c),

so that (R - c) ~ (R0 - c0) Q with c, c0 the centroids. A lab-frame direction
frozen at R0, a row vector f, becomes f Q at R: invariant under any rigid
motion of R, the identity at R0, and smooth wherever the superposition is
unique.

THE DERIVATIVE. Q is stationary on SO(3), so K = Q^T A is symmetric. Varying A
and writing dQ = Q Omega with Omega antisymmetric gives

    K Omega + Omega K = Q^T dA - dA^T Q,

solved in K's eigenbasis by Omega'_ab = C'_ab / (k_a + k_b). The adjoint of
sum Q_bar . Q is then A_bar = 2 Q Y, with Y the same solve applied to the
antisymmetric part of Q^T Q_bar, and dA = (R0 - c0)^T dR closes it on the
nuclei. Both sums k_a + k_b stay positive on a planar molecule (one k is
zero) and vanish only for a linear one, whose rotation about its axis no
superposition can fix.
"""
import numpy as np

from src.Base.constants import BODY_FRAME_MIN_SPREAD


class BodyFrame:
    """The rotation of a geometry against the reference it was frozen at.

    `rotation(coords)` is None at the reference itself, bit for bit, so a
    convention read there is the one frozen there without a product by an
    identity; elsewhere it is the Kabsch Q. A linear reference determines no
    body frame, and its conventions stay where they were frozen (`fixed`).
    """

    def __init__(self, reference):
        self.reference = np.array(reference, dtype=float)
        self.fixed = is_linear(self.reference)

    def same(self, other):
        """Whether `other` turns conventions the same way."""
        return (other is not None and self.fixed == other.fixed
                and np.array_equal(self.reference, other.reference))

    def rotation(self, coords):
        """Q at `coords`, or None where it is the identity by construction."""
        coords = np.asarray(coords, float)
        if self.fixed or np.array_equal(coords, self.reference):
            return None
        return kabsch_rotation(self.reference, coords)

    def rotation_adjoint(self, coords, q_bar):
        """(natm, 3) gradient of sum_ab q_bar[a, b] Q[a, b] at `coords`."""
        coords = np.asarray(coords, float)
        if self.fixed:
            return np.zeros_like(coords)
        a0 = centred(self.reference)
        a = a0.T @ centred(coords)
        q = self.rotation(coords)
        q = np.eye(3) if q is None else q
        k = q.T @ a
        lam, w = np.linalg.eigh(0.5 * (k + k.T))
        pair = lam[:, None] + lam[None, :]
        np.fill_diagonal(pair, 1.0)
        if pair.min() <= BODY_FRAME_MIN_SPREAD * max(abs(lam).max(), 1.0):
            raise ValueError(
                'the geometry is (nearly) linear or turned by 180 degrees '
                'against the reference: the superposing rotation is not '
                'unique there and has no derivative')
        b = q.T @ np.asarray(q_bar, float)
        y = w @ ((w.T @ (0.5 * (b - b.T)) @ w) / pair) @ w.T
        return a0 @ (2.0 * q @ y)


def centred(coords):
    """Coordinates about their centroid."""
    coords = np.asarray(coords, float)
    return coords - coords.mean(axis=0)


def kabsch_rotation(reference, coords):
    """Proper Q with (coords - c) ~ (reference - c0) Q, least squares."""
    a = centred(reference).T @ centred(coords)
    u, _, vt = np.linalg.svd(a)
    # det 0 (a planar pair) keeps +1: np.sign would make Q a projection
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def aligned_displacement(reference, coords):
    """Largest atomic displacement of `coords` from `reference` once the
    rigid motion between them is removed (Kabsch superposition)."""
    q = kabsch_rotation(reference, coords)
    return float(np.abs(centred(coords) @ q.T - centred(reference)).max())


def is_linear(coords):
    """True when the atoms lie on a line, where no rotation is determined."""
    s = np.linalg.svd(centred(coords), compute_uv=False)
    return len(s) < 2 or s[1] <= BODY_FRAME_MIN_SPREAD * max(s[0], 1.0)
