"""The tiles every density-fitted ISR-ADC(2) density pass walks: an axis cut
into tiles that fit a memory budget, the first-order amplitudes and a root's
doubles of one occupied tile, the two tile contractions, and `Pass`, which
sums one addend per tile in tile order over the ranks (ordered_chain_sum), so
a density is the same bits at every rank count. The difference density
(ee_ri_density), the transition densities (ee_ri_transition) and the relaxed
density (ee_relaxed_density) are written on them.
"""
import numpy as np

from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import tile_width
from src.Base.utils.parallel.mpi_grid import ordered_chain_sum
from src.SingleReference.common.df_amplitudes import doubles_denominator


def tile_ranges(n, per, tile_gb, ranks):
    """(start, stop) tiles of an axis of n, each holding at most tile_gb at
    `per` bytes per index and no more than gives every rank one."""
    k = tile_width(per, -(-n // ranks), Numerics(layout={'tile_gb': tile_gb}))
    return [(k0, min(k0 + k, n)) for k0 in range(0, n, k)]


class Pass:
    """Addends of one pass, one flat array per tile, summed in tile order.

    The running sum travels from each tile's rank to the next tile's
    (ordered_chain_sum): no rank receives another rank's addends, which at
    many tiles of a large pass would be more than one gather can carry."""

    def __init__(self, op, shapes):
        self.op, self.shapes = op, shapes
        self.size = sum(int(np.prod(s)) for s in shapes.values())
        self.parts = []

    def add(self, t, blocks):
        flat = np.zeros(self.size)
        p = 0
        for name, shape in self.shapes.items():
            m = int(np.prod(shape))
            if name in blocks:
                flat[p:p + m] = np.asarray(blocks[name]).ravel()
            p += m
        self.parts.append((t, flat))

    def total(self):
        comm = self.op.comm
        mine = [t for t, _ in self.parts]
        held = (comm.allgather(mine) if comm is not None and comm.Get_size() > 1
                else [mine])
        owner = {t: r for r, ts in enumerate(held) for t in ts}
        order = sorted(owner)
        at = {t: k for k, t in enumerate(order)}
        tot = ordered_chain_sum({at[t]: a for t, a in self.parts},
                                [owner[t] for t in order], comm, np.zeros(self.size))
        out, p = {}, 0
        for name, shape in self.shapes.items():
            m = int(np.prod(shape))
            out[name] = tot[p:p + m].reshape(shape)
            p += m
        return out


def tile_amplitudes(op, k0, k1):
    """(tau, ts) of the occupied rows k0..k1-1, shape (kt, nv, no, nv)."""
    f = op.f
    I = op.ints(k0, k1)
    D = doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
    tau = I / D
    del I
    return tau, tau - tau.transpose(0, 3, 2, 1)


def tile_doubles(op, BtT, k0, k1, omega, scale, s):
    """(Z, W) of the occupied rows k0..k1-1 (ee_ri_density's docstring), from the
    auxiliary-major dressed factor BtT (naux, no nv)."""
    f = op.f
    H, HT = op.pair(BtT, k0, k1)
    D = doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
    w = scale / (omega - D)
    del D
    if s > 0:
        Z = w * (H + HT)
        del H, HT
        W = Z - Z.transpose(0, 3, 2, 1)
    else:
        Z = w * (H - HT)
        S = H + HT
        del H, HT
        W = w * (S - S.transpose(0, 3, 2, 1))
    return Z, W


def root_doubles(op, y, omega, t1):
    """(Z, W) of every occupied row of one root (ee_ri_density's docstring), shape
    (no, nv, no, nv) each: its doubles held whole, for molecules whose
    o^2 v^2 fits in memory."""
    s = 1.0 if op.spin == 'singlet' else -1.0
    return tile_doubles(op, op.dressed(np.asarray(y, float).ravel()), 0, op.no,
                    omega, np.sqrt(t1) / (2.0 * np.sqrt(2.0)), s)


def oo_contract(A, B):
    """sum_{k in tile} sum_ab A[k,b,i,a] B[k,b,j,a] -> (i, j)."""
    no = A.shape[2]
    a = A.transpose(2, 0, 1, 3).reshape(no, -1)
    b = B.transpose(2, 0, 1, 3).reshape(no, -1)
    return a @ b.T


def vv_contract(A, B):
    """sum_{i in tile} sum_jc A[i,a,j,c] B[i,b,j,c] -> (a, b)."""
    nv = A.shape[1]
    out = np.zeros((nv, nv))
    for i in range(A.shape[0]):
        out += A[i].reshape(nv, -1) @ B[i].reshape(nv, -1).T
    return out
