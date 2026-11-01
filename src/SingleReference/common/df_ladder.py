"""The particle-particle ladder through a density-fitted factor:

    L[i, j, a, b] = sum_cd (ac|bd) Y[i, j, c, d],   (ac|bd) = sum_Q Bl[Q, a, c] Br[Q, b, d]

for several amplitude tensors Y (i, j, c, d) at once (the occupied indices are
spectators). Two evaluations, the same numbers to a few ulps:

    'aux'   through the auxiliary index in chunks of Q: two products per chunk,
            2 naux N va vb (va + vb) / 2 flops for N rows of Y
    'slab'  (ac|bd) formed in slabs of a few a and contracted with every Y in one
            matrix product per slab: va^2 vb^2 (naux + N) flops, the integrals
            built once for all the Y

'auto' takes the cheaper by that count: 'slab' whenever the rows outnumber about
v/2 (no^2 > nv/2 for doubles), 'aux' for few occupied and many virtual orbitals.
`mem_gb` bounds the chunk (aux) or slab (two copies of va vb^2 per a) in memory.
"""
import numpy as np

KERNELS = ('auto', 'aux', 'slab')


def choose_kernel(naux, rows, va, vb, kernel='auto'):
    """'aux' or 'slab' for `rows` rows of Y (the flop count above)."""
    if kernel not in KERNELS:
        raise ValueError(f'ladder kernel {kernel!r}: one of {KERNELS}')
    if kernel != 'auto':
        return kernel
    aux = naux * rows * va * vb * (va + vb)
    slab = va * va * vb * vb * (naux + rows)
    return 'slab' if slab < aux else 'aux'


def ladder_slabs(Bl, Br, M, mem_gb=2.0, pair_factor=None):
    """Yield (a0, a1, X) with X[r, a - a0, b] = sum_cd (ac|bd) M[r, c vb + d] for
    every slab a0 <= a < a1; M is (rows, va vb). The building block of `ladders`,
    for callers that keep only part of each slab. pair_factor: for Bl is Br, the
    factor with the pair index leading, (va va, naux) C-contiguous with row a va + c
    = Bl[:, a, c], when the caller holds it already."""
    naux, va = Bl.shape[0], Bl.shape[1]
    vb = Br.shape[1]
    if pair_factor is None:
        Bf = np.ascontiguousarray(Br).reshape(naux, vb * vb)
    width = max(1, min(va, int(mem_gb * 1.25e8 / (2 * va * vb * vb))))
    for a0 in range(0, va, width):
        a1 = min(a0 + width, va)
        if pair_factor is None:
            G = (np.ascontiguousarray(Bl[:, a0:a1]).reshape(naux, -1).T @ Bf)   # (a c, b d)
        else:
            G = pair_factor[a0 * va:a1 * va] @ pair_factor.T                      # (a c, b d)
        G = G.reshape(a1 - a0, va, vb, vb).transpose(0, 2, 1, 3)              # (a, b, c, d)
        G = np.ascontiguousarray(G).reshape((a1 - a0) * vb, va * vb)
        yield a0, a1, (M @ G.T).reshape(M.shape[0], a1 - a0, vb)


def ladders(Bl, Br, Ys, mem_gb=2.0, kernel='auto'):
    """[L(Y) for Y in Ys] (module docstring), each Y (i, j, va, vb) with va, vb
    the virtual counts of Bl and Br."""
    naux, va, vb = Bl.shape[0], Bl.shape[1], Br.shape[1]
    rows = [Y.shape[0] * Y.shape[1] for Y in Ys]
    if choose_kernel(naux, sum(rows), va, vb, kernel) == 'aux':
        return [_aux(Bl, Br, Y, mem_gb) for Y in Ys]
    M = np.concatenate([np.reshape(Y, (n, va * vb)) for Y, n in zip(Ys, rows)])
    out = np.empty((M.shape[0], va, vb))
    for a0, a1, X in ladder_slabs(Bl, Br, M, mem_gb):
        out[:, a0:a1] = X
    del M
    edges = np.cumsum([0] + rows)
    return [out[e0:e1].reshape(Y.shape[:2] + (va, vb))
            for Y, e0, e1 in zip(Ys, edges[:-1], edges[1:])]


def _aux(Bl, Br, Y, mem_gb):
    """The 'aux' evaluation of one Y (i, j, c, d), chunks of Q inside mem_gb."""
    naux, va, vb = Bl.shape[0], Bl.shape[1], Br.shape[1]
    no2 = Y.shape[0] * Y.shape[1]
    n = max(1, min(naux, int(mem_gb * 1.25e8 / max(no2 * va * vb, 1))))
    out = np.zeros(Y.shape[:2] + (va, vb))
    for k in range(0, naux, n):
        Bc, Bd = Bl[k:k + n], Br[k:k + n]
        U = np.einsum('Qac,ijcd->Qijad', Bc, Y, optimize=True)
        out += np.einsum('Qijad,Qbd->ijab', U, Bd, optimize=True)
    return out
