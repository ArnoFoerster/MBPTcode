"""Excitation-transfer couplings between two site diabats of a fragment
partition (`src.fragments.partition`), for any method behind its
`SinglesOperator` protocol.

THE COUPLING. A_eff(Omega_0) is the effective Hamiltonian of the diabats in
P with everything else eliminated at Omega_0. The transfer from a donor site
diabat d to an acceptor site diabat a sees the other diabats of P (the
charge-transfer ones above all) as intermediates, so the two-state matrix is
the Loewdin fold of those at the transfer energy E,

    H(E) = A_ss + A_sf (E - A_ff)^-1 A_fs ,   A = A_eff(Omega_0) ,

s = (d, a), f the folded diabats, and the coupling is V = H(E)_da. Charge
transfer is counted once: kept explicit and folded here, or left in Q and
folded at Omega_0 by the partition. At E = Omega_0 the two are the same
matrix (the quotient property of Schur complements), which is the identity
the tests check; at another E only the explicit block moves, Q stays at
Omega_0, and `slope` (the element of dA_eff/dOmega) says how far the rest
would move to first order.

The decomposition the literature reads: `bare` is the element of K_PP (the
operator between the two diabats, nothing eliminated), `dressed` that of
A_eff(Omega_0) (Q eliminated), `superexchange` = V - dressed what the folded
diabats add. In the triplet channel the GW-BSE kernel has no exchange term
2v, so `bare` is the screened direct term -W between the two fragments plus
the one-electron couplings of the local orbitals; ADC(2) adds its folded
doubles. A triplet-triplet (Dexter) coupling is `dexter_coupling`, which
refuses a singlet partition.

ITS GRADIENT. With E held fixed, dH_da = u_d^T dA u_a, u_c = e_c +
(E - A_ff)^-1 A_fc on the folded components: the coupling's nuclear gradient
is `DiabaticGradient.gradient(W)` with W = sym(u_d u_a^T) (`weights`), one
reverse pass of the method.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import FRAGMENT_POLE_MARGIN


@dataclass(frozen=True)
class TransferCoupling:
    """The effective coupling between two site diabats (module docstring).
    Energies in Hartree.

    donor, acceptor: their labels; energy: E of the fold; omega0: the
    partition's; folded: the labels folded at E; coupling: V = H(E)_da;
    site_energies: (H(E)_dd, H(E)_aa); bare, dressed, superexchange: the
    decomposition of V; slope: dA_eff/dOmega_da at Omega_0; weights: the
    (n_p, n_p) W whose contraction with dA_eff/dR is dV/dR; spin: the
    partition's channel."""
    donor: str
    acceptor: str
    energy: float
    omega0: float
    folded: tuple
    coupling: float
    site_energies: tuple
    bare: float
    dressed: float
    superexchange: float
    slope: float
    weights: np.ndarray
    spin: str


def _index(labels, label):
    if isinstance(label, (int, np.integer)):
        return int(label)
    if label not in labels:
        raise ValueError(f'{label!r} is not a diabat of this partition '
                         f'({labels})')
    return labels.index(label)


def require_below(a_ff, energy, margin=FRAGMENT_POLE_MARGIN):
    """Refuse an `energy` within `margin` of, or above, an eigenvalue of the
    folded block `a_ff`. A folded diabat at or below the transfer energy is a
    real intermediate of a two-step transfer, not a virtual one."""
    a_ff = np.asarray(a_ff, float)
    if not len(a_ff):
        return
    lowest = float(np.linalg.eigvalsh(a_ff)[0])
    if not energy < lowest - margin:
        raise ValueError(
            f'E = {energy:.6f} Ha is not below the lowest folded diabat '
            f'({lowest:.6f} Ha) by the margin {margin}: a folded state at '
            f'or below the transfer energy is a real intermediate, not a '
            f'virtual one. Lower E or keep that diabat in the model.')


def fold_diabats(a, keep, energy, margin=FRAGMENT_POLE_MARGIN):
    """(H(E), U): the Loewdin fold of every diabat outside `keep` out of the
    symmetric matrix `a` at `energy`, H = U^T (a - E) U + E with U (n, k) the
    columns u_c of the module docstring (the kept diabats' unit vectors plus
    (E - a_ff)^-1 a_fc on the folded ones).

    Refused: an E within `margin` of, or above, an eigenvalue of the folded
    block (`require_below`)."""
    a = np.asarray(a, float)
    n = len(a)
    keep = [int(k) for k in keep]
    fold = [c for c in range(n) if c not in keep]
    u = np.zeros((n, len(keep)))
    u[keep, np.arange(len(keep))] = 1.0
    if fold:
        a_ff = a[np.ix_(fold, fold)]
        require_below(a_ff, energy, margin)
        u[fold] = np.linalg.solve(energy * np.eye(len(fold)) - a_ff,
                                  a[np.ix_(fold, keep)])
    h = u.T @ (a - energy * np.eye(n)) @ u + energy * np.eye(len(keep))
    return 0.5 * (h + h.T), u


def site_coupling(partition, donor, acceptor, energy=None, fold=None):
    """`TransferCoupling` between the site diabats `donor` and `acceptor`
    (labels such as 'site 0.0', or indices) of `partition`, at the transfer
    energy `energy` (default Omega_0). `fold`: the labels of the diabats
    folded at E, default every other diabat of P; the rest of P (if any) is
    dropped, which the caller asks for by naming `fold`."""
    labels = list(partition.labels)
    d, a = _index(labels, donor), _index(labels, acceptor)
    for c in (d, a):
        if not labels[c].startswith('site'):
            raise ValueError(f'{labels[c]} is not a site diabat: a transfer '
                             f'coupling is between two sites')
    if d == a:
        raise ValueError('donor and acceptor are the same diabat')
    energy = partition.omega0 if energy is None else float(energy)
    if fold is None:
        fold_idx = [c for c in range(len(labels)) if c not in (d, a)]
    else:
        fold_idx = [_index(labels, c) for c in fold]
        if d in fold_idx or a in fold_idx:
            raise ValueError('the donor and the acceptor cannot be folded')
    sub = [d, a] + fold_idx
    h, u = fold_diabats(partition.a_eff[np.ix_(sub, sub)], (0, 1), energy)
    w = np.zeros((len(labels), len(labels)))
    ud, ua = u[:, 0], u[:, 1]
    w[np.ix_(sub, sub)] = 0.5 * (np.outer(ud, ua) + np.outer(ua, ud))
    return TransferCoupling(
        donor=labels[d], acceptor=labels[a], energy=energy,
        omega0=float(partition.omega0),
        folded=tuple(labels[c] for c in fold_idx), coupling=float(h[0, 1]),
        site_energies=(float(h[0, 0]), float(h[1, 1])),
        bare=float(partition.a_pp[d, a]),
        dressed=float(partition.a_eff[d, a]),
        superexchange=float(h[0, 1] - partition.a_eff[d, a]),
        slope=float(partition.dsigma[d, a]), weights=w,
        spin=getattr(partition.operator, 'spin', None))


def dexter_coupling(partition, donor, acceptor, energy=None, fold=None):
    """The triplet-triplet (Dexter) coupling T1(D) S0(A) -> S0(D) T1(A):
    `site_coupling` on a triplet partition; a singlet one is refused."""
    spin = getattr(partition.operator, 'spin', None)
    if spin != 'triplet':
        raise ValueError(f'a Dexter coupling is between triplet diabats; this '
                         f'partition is of the {spin!r} channel')
    return site_coupling(partition, donor, acceptor, energy=energy, fold=fold)
