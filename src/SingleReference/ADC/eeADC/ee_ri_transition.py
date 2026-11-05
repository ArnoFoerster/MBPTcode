"""ISR-ADC(2) one-particle transition densities between folded roots, on
density-fitted tiles: the doubles of every root are rebuilt one occupied tile
at a time from its singles, never held whole, and so are the amplitudes.

ee_transition_density writes the transition densities once, in spin orbitals,
and evaluates them with every doubles-sized array whole (o^2 v^2), the route
of small molecules and the oracle of this one. Here the same terms (its
names) are evaluated for one spin channel of a closed shell from the tiled
operators of ee_ri_fold; ee_ri_density is the pair (k, k) of a root with itself.

Spin cases. A root of parity s (+1 singlet, -1 Ms = 0 triplet) is a flip
eigenvector, so the beta block of <l|a^dag_p a_q|r> is s_l s_r times the
alpha block gamma, and

    charge (alpha + beta) = (1 + s_l s_r) gamma
    spin   (alpha - beta) = (1 - s_l s_r) gamma

(the ground state counts as a singlet): a singlet-triplet pair has spin
density only, two roots of one spin charge only. With x the alpha singles
(sqrt(T1 / 2) y), Z, W the opposite- and same-spin doubles of ee_ri_density
(W and the amplitudes tau, ts symmetric under the pair swap (i a) <-> (j b),
Z of parity s), T_p v = (ts + p tau) v and the tile contractions
oo_contract, vv_contract of ee_ri_tiles, the alpha block of a pair (l, r)
is

    oo_dd   -2 [oo(W^r, W^l) + 2 s_l s_r oo(Z^r, Z^l)]
    vv_dd    2 [vv(W^l, W^r) + 2 vv(Z^l, Z^r)]
    ru       r_n = T_{s_n} x^n:   oo -r_l r_r^T,  vv r_r^T r_l
    tru      oo -[x^r (T_{s_l} r_l)^T + (T_{s_r} r_r) x^l^T] / 2
             vv  [(T_{s_l} r_l)^T x^r + x^l^T T_{s_r} r_r] / 2
    tt       ee_ri_density's sums with p1 of the pair, every term whose p1
             is a beta block times s_l s_r, none symmetrized
    ov_sd   -2 q(r, l),      q(n, m) = (W^n + s_m Z^n) x^m
    ov_tsd   2 T_{s_l s_r} q(l, r)
    ov_tdv  -x^r TU_vv(l)^T,  ov_tdo -TU_oo(l) x^r
             TU_oo(n) = oo(ts, W^n) + 2 s_n oo(tau, Z^n)
             TU_vv(n) = vv(ts, W^n) + 2 vv(tau, Z^n)
    vo_*     the ov terms of (r, l), transposed

and the rest (ss0, p0 terms) products of small matrices. The ground row of a
root n (u1 = x):

    ov_ring  T_s r_n / 2,  vo_t1 -r_n^T,  oo_d -TU_oo(n)^T,  vv_d TU_vv(n)
    vo_t2   -[(ts2 + s tau2) x]^T

with tau2, ts2 the opposite- and same-spin second-order amplitudes, in the
(i j a b) layout N / D with T = tau, S = ts (i j a b):

    N_os = -PP(T) - HH(T) + P[-RC(S + T) + RX(T) + T_ikcb (kj|ac)]
    N_ss = -PP(S) - HH(S) + A[RX(S) - RC(S + T)]
    PP(T) = sum_cd (ac|bd) T_ijcd     HH(T) = sum_kl (ki|lj) T_klab
    RC(T) = sum_kc T_ikac (kc|jb)     RX(T) = sum_kc T_ikac (kj|bc)

(P the pair swap's sum X + X_jiba, A the antisymmetrizer); RC goes through
the factors, RX and HH through occupied-tile pairs, PP through the particle
ladder of common.df_ladder on B_vv, which this pass reads whole, one copy
per node.

Spin-component scaling (an operator's `scs`, (c_os, c_ss)) is
ee_transition_density's: tau and Z carry sqrt(c_os), ts and W sqrt(c_ss),
t1_2 (p0_ov) c, and the second-order amplitudes are built from the scaled
first-order ones.

Every pass deals tiles over the operators' ranks and sums them in tile order
(ee_ri_tiles.Pass, ordered_chain_sum): the same bits at every rank count at one
tile size, and the memory per rank a few tiles (plus the factors and, for
the ground row, B_vv on its node).
"""
import numpy as np

from src.Base.numerics.constants import (EE_RI_TILE_GB, EE_RI_TRANSITION_GROUND_TILE_ARRAYS,
                                         EE_RI_TRANSITION_TILE_ARRAYS,
                                         EE_RI_TRANSITION_VIRT_TILE_ARRAYS)
from src.Base.utils.parallel.mpi_grid import NodeShared, broadcast_rows
from src.SingleReference.common import df_ladder
from src.SingleReference.common.df_amplitudes import doubles_denominator
from src.SingleReference.ADC.eeADC.ee_transition_density import assemble
from src.SingleReference.ADC.eeADC.ee_ri_tiles import (Pass, tile_amplitudes, tile_doubles,
                                                     oo_contract, tile_ranges, vv_contract)

GROUND = ('ground', 0)


class _Root:
    """One folded root: its operator, parity, unit channel singles y,
    frequency, singles weight and alpha singles x."""

    def __init__(self, op, y, omega, t1):
        if op.level != 'adc2':
            raise ValueError('the tiled transition densities are ADC(2)\'s; '
                             f'an operator at {op.level!r} has no doubles')
        self.op, self.omega, self.t1 = op, float(omega), float(t1)
        self.s = 1.0 if op.spin == 'singlet' else -1.0
        self.y = np.asarray(y, float).ravel()
        self.x = (np.sqrt(self.t1 / 2.0) * self.y).reshape(op.no, op.nv)


def _weights(op):
    """(a_os, a_ss): the square roots of the operator's spin-component
    scaling, (1, 1) for ADC(2)."""
    return (1.0, 1.0) if op.scs is None else tuple(float(np.sqrt(c)) for c in op.scs)


def _scaled_amplitudes(op, k0, k1, w):
    """(tau, ts) of the rows k0..k1-1, each times its class weight."""
    tau, ts = tile_amplitudes(op, k0, k1)
    return w[0] * tau, w[1] * ts


def _tt_terms(tau, ts, p1_oo, p1_vv, slr):
    """oo_tt_v, vv_tt_v, vv_tt_o of one occupied tile and one pair: the terms
    whose p1 is a beta block carry slr = s_l s_r."""
    Rs = np.einsum('db,kbjc->kdjc', p1_vv, ts, optimize=True)
    Rt = np.einsum('db,kbjc->kdjc', p1_vv, tau, optimize=True)
    Rx = np.einsum('kcjb,db->kcjd', tau, p1_vv, optimize=True)
    oo_v = (oo_contract(ts, Rs) + slr * oo_contract(tau, Rt)
            + np.einsum('kcid,kcjd->ij', tau, Rx, optimize=True))
    Rs = np.einsum('kald,cd->kalc', ts, p1_vv, optimize=True)
    Rt = np.einsum('kald,cd->kalc', tau, p1_vv, optimize=True)
    vv_v = -0.5 * (vv_contract(Rs, ts) + 2.0 * slr * vv_contract(Rt, tau))
    Rs = np.einsum('jk,lckb->lcjb', p1_oo, ts, optimize=True)
    Rt = np.einsum('jk,lckb->lcjb', p1_oo, tau, optimize=True)
    vv_o = (np.einsum('lcjb,lcja->ab', Rs, ts, optimize=True)
            + np.einsum('lcjb,lcja->ab', Rt, tau, optimize=True)
            + slr * np.einsum('lbjc,lajc->ab', Rt, tau, optimize=True))
    return oo_v, vv_v, vv_o


def _p1(left, right):
    """(p1_oo, p1_vv) alpha blocks of the singles of <left| |right>."""
    return -right.x @ left.x.T, left.x.T @ right.x


class _Job:
    """The roots, pairs and ground kets of one call, and its tiling."""

    def __init__(self, roots, pairs, grounds, tile_gb):
        self.roots = {k: _Root(*v) for k, v in roots.items()}
        ops = [r.op for r in self.roots.values()]
        self.op = ops[0]
        f = self.op.f
        for o in ops[1:]:
            if o.f is not f or o.scs != self.op.scs or o.comm is not self.op.comm:
                raise ValueError('the roots of one call share the factors, the '
                                 'spin-component scaling and the communicator')
        self.no, self.nv, self.f = f.no, f.nv, f
        self.w = _weights(self.op)
        for bra, ket in pairs:
            for k in (bra, ket):
                if k not in self.roots:
                    raise ValueError(f'{k!r} is not one of the roots given')
        for k in grounds:
            if k not in self.roots:
                raise ValueError(f'{k!r} is not one of the roots given')
        self.pairs, self.grounds = list(pairs), list(grounds)
        comm = self.op.comm
        self.ranks = 1 if comm is None else comm.Get_size()
        self.rank = 0 if comm is None else comm.Get_rank()
        self.tile_gb = tile_gb
        # the roots any term needs the doubles of
        self.used = sorted({k for p in self.pairs for k in p} | set(self.grounds))
        per = 8.0 * self.nv * self.no * self.nv * (
            EE_RI_TRANSITION_TILE_ARRAYS + 2 * len(self.used))
        tiles = tile_ranges(self.no, per, tile_gb, self.ranks)
        self.mine = [(t, kk) for t, kk in enumerate(tiles) if t % self.ranks == self.rank]


def _pass_amplitudes(job):
    """Pass 1, on the operator's own tiles (whose p0_ov numerator its B_vv
    chunk owners back-transform): r_n per root, the p0 blocks and the
    pairs' tt terms."""
    op, no, nv, w, f = job.op, job.no, job.nv, job.w, job.f
    shapes = {'p0_oo': (no, no), 'p0_vv': (nv, nv), 'p0_ov': (no, nv)}
    for k in job.used:
        shapes[('r', k)] = (no, nv)
    p1s = {p: _p1(job.roots[p[0]], job.roots[p[1]]) for p in job.pairs}
    for p in job.pairs:
        shapes.update({('oo_tt_v', p): (no, no), ('vv_tt_v', p): (nv, nv),
                       ('vv_tt_o', p): (nv, nv)})
    acc = Pass(op, shapes)
    gams = {}
    for t, (k0, k1) in op.my_tiles:
        kt = k1 - k0
        tau, ts = _scaled_amplitudes(op, k0, k1, w)
        blocks = {}
        for k in job.used:
            root = job.roots[k]
            r = np.zeros((no, nv))
            r[k0:k1] = ((ts + root.s * tau).reshape(kt * nv, -1) @ root.x.ravel()
                        ).reshape(kt, nv)
            blocks[('r', k)] = r
        # t1_2: c_os tau + c_ss ts, the scaled ones once more weighted
        G = (w[1] * ts + w[0] * tau).reshape(kt * nv, -1) @ f.L
        blocks['p0_ov'] = 2.0 * op.back_oo(G, k0, k1).reshape(no, nv)
        gams[t] = G[None]
        del G
        blocks['p0_oo'] = -0.5 * (oo_contract(ts, ts) + 2.0 * oo_contract(tau, tau))
        blocks['p0_vv'] = 0.5 * (vv_contract(ts, ts) + 2.0 * vv_contract(tau, tau))
        for p in job.pairs:
            slr = job.roots[p[0]].s * job.roots[p[1]].s
            oo_v, vv_v, vv_o = _tt_terms(tau, ts, *p1s[p], slr)
            blocks.update({('oo_tt_v', p): oo_v, ('vv_tt_v', p): vv_v,
                           ('vv_tt_o', p): vv_o})
        del tau, ts
        acc.add(t, blocks)
    a = acc.total()
    num = a['p0_ov'] + 2.0 * op.back_vv(gams, 1)[0].reshape(no, nv)
    del gams
    d_ia = f.eo[:, None] - f.ev[None, :]
    a['p0_ov'] = num / (2.0 * d_ia)
    return a


def _pass_doubles(job):
    """Pass 2, the doubles of every root on this rank's tiles: the pairs' dd
    blocks and q vectors, and TU_oo, TU_vv of every root."""
    no, nv, w = job.no, job.nv, job.w
    shapes = {}
    for p in job.pairs:
        l, r = p
        shapes.update({('oo_dd', p): (no, no), ('vv_dd', p): (nv, nv),
                       ('q', r, l): (no, nv), ('q', l, r): (no, nv)})
    for k in job.used:
        shapes.update({('tu_oo', k): (no, no), ('tu_vv', k): (nv, nv)})
    acc = Pass(job.op, shapes)
    dressed = {k: job.roots[k].op.dressed(job.roots[k].y) for k in job.used}
    for t, (k0, k1) in job.mine:
        kt = k1 - k0
        dbl = {}
        for k in job.used:
            root = job.roots[k]
            Z, W = tile_doubles(root.op, dressed[k], k0, k1, root.omega,
                            np.sqrt(root.t1) / (2.0 * np.sqrt(2.0)), root.s)
            dbl[k] = (w[0] * Z, w[1] * W)
            del Z, W
        tau, ts = _scaled_amplitudes(job.op, k0, k1, w)
        blocks = {}
        for k in job.used:
            Z, W = dbl[k]
            blocks[('tu_oo', k)] = (oo_contract(ts, W)
                                    + 2.0 * job.roots[k].s * oo_contract(tau, Z))
            blocks[('tu_vv', k)] = vv_contract(ts, W) + 2.0 * vv_contract(tau, Z)
        del tau, ts
        for p in job.pairs:
            l, r = p
            sl, sr = job.roots[l].s, job.roots[r].s
            (Zl, Wl), (Zr, Wr) = dbl[l], dbl[r]
            blocks[('oo_dd', p)] = -2.0 * (oo_contract(Wr, Wl)
                                           + 2.0 * sl * sr * oo_contract(Zr, Zl))
            blocks[('vv_dd', p)] = 2.0 * (vv_contract(Wl, Wr) + 2.0 * vv_contract(Zl, Zr))
            for n, m in ((r, l), (l, r)):
                Zn, Wn = dbl[n]
                q = np.zeros((no, nv))
                q[k0:k1] = ((Wn + job.roots[m].s * Zn).reshape(kt * nv, -1)
                            @ job.roots[m].x.ravel()).reshape(kt, nv)
                blocks[('q', n, m)] = q
        del dbl
        acc.add(t, blocks)
    del dressed
    return acc.total()


def _pass_t_products(job, vectors):
    """Pass 3: {name: T_p v} of vectors {name: (v, p)}, on this rank's tiles."""
    no, nv = job.no, job.nv
    acc = Pass(job.op, {name: (no, nv) for name in vectors})
    for t, (k0, k1) in job.mine:
        kt = k1 - k0
        tau, ts = _scaled_amplitudes(job.op, k0, k1, job.w)
        blocks = {}
        for name, (v, p) in vectors.items():
            out = np.zeros((no, nv))
            out[k0:k1] = ((ts + p * tau).reshape(kt * nv, -1) @ v.ravel()).reshape(kt, nv)
            blocks[name] = out
        del tau, ts
        acc.add(t, blocks)
    return acc.total()


def _pass_virtual_tiles(job):
    """Pass 4: oo_tt_o of every pair, summed over tiles of one virtual index."""
    f, no, nv, w = job.f, job.no, job.nv, job.w
    vtiles = tile_ranges(nv, 8.0 * no * nv * no * EE_RI_TRANSITION_VIRT_TILE_ARRAYS,
                         job.tile_gb, job.ranks)
    acc = Pass(job.op, {p: (no, no) for p in job.pairs})
    p1s = {p: _p1(job.roots[p[0]], job.roots[p[1]])[0] for p in job.pairs}
    L3 = f.L.reshape(no, nv, -1)
    for t, (d0, d1) in enumerate(vtiles):
        if t % job.ranks != job.rank:
            continue
        dt = d1 - d0
        Ld = np.ascontiguousarray(L3[:, d0:d1]).reshape(no * dt, -1)
        I = (f.L @ Ld.T).reshape(no, nv, no, dt)                 # (ic|kd)
        D = (f.ev[None, :, None, None] + f.ev[None, None, None, d0:d1]
             - f.eo[:, None, None, None] - f.eo[None, None, :, None])
        tau = I / D
        del I, D
        ts = w[1] * (tau - tau.transpose(2, 1, 0, 3))            # - tau_idkc
        tau *= w[0]
        blocks = {}
        for p, p1 in p1s.items():
            slr = job.roots[p[0]].s * job.roots[p[1]].s
            S = np.zeros((no, no))
            for X, wgt in ((ts, 1.0), (tau, 2.0 * slr)):
                R = np.einsum('lk,jcld->jckd', p1, X, optimize=True)
                S += wgt * (X.reshape(no, -1) @ R.reshape(no, -1).T)
            blocks[p] = -0.5 * S
        del tau, ts
        acc.add(t, blocks)
    return acc.total()


def _bvv_whole(job):
    """(holder, B_vv as [P, a, b]) on every rank, one copy per node: the
    operator's auxiliary chunks broadcast from their owners."""
    op, f, nv = job.op, job.f, job.nv
    holder = NodeShared((f.naux, nv, nv), op.comm)
    for c, (p0, p1) in enumerate(op.chunks):
        if op.owners[c] == job.rank:
            buf = np.ascontiguousarray(f.bvv(c, p0, p1).transpose(1, 0, 2))
        else:
            buf = np.empty((p1 - p0, nv, nv))
        broadcast_rows(buf, op.owners[c], op.comm)
        if holder.fills:
            holder.array[p0:p1] = buf
        del buf
    holder.ready()
    return holder, holder.array


def _exchange_block(f, j0, j1, Bvv):
    """(kj|bc) for every k and j0 <= j < j1, shape (no, jt, nv, nv)."""
    no, nv = f.no, f.nv
    Boj = np.ascontiguousarray(f.BooP[:, :, j0:j1]).reshape(f.naux, -1)
    return (Boj.T @ Bvv.reshape(f.naux, -1)).reshape(no, j1 - j0, nv, nv)


def _pass_second_order(job):
    """Pass 5, the ground row's (ts2 + s tau2) x of every ground ket, on this
    rank's tiles (module docstring)."""
    f, no, nv, w = job.f, job.no, job.nv, job.w
    op = job.op
    gtiles = tile_ranges(no, 8.0 * nv * no * nv * EE_RI_TRANSITION_GROUND_TILE_ARRAYS,
                         job.tile_gb, job.ranks)
    gmine = [(t, kk) for t, kk in enumerate(gtiles) if t % job.ranks == job.rank]
    # Q^P_ia = sum_kc (S + T)_ikac B^P_kc of every row, whole on every rank:
    # a tile's rows broadcast from the rank that made them (one copy, not one
    # per tile)
    Q = np.empty((no * nv, f.naux))
    for t, (k0, k1) in enumerate(gtiles):
        rows = Q[k0 * nv:k1 * nv]
        if t % job.ranks == job.rank:
            tau, ts = _scaled_amplitudes(op, k0, k1, w)
            rows[...] = (ts + tau).reshape((k1 - k0) * nv, -1) @ f.L
            del tau, ts
        broadcast_rows(rows, t % job.ranks, op.comm)
    holder, Bvv = _bvv_whole(job)
    L = f.L
    acc = Pass(op, {k: (no, nv) for k in job.grounds})
    for t, (i0, i1) in gmine:
        it = i1 - i0
        To, Ts = _scaled_amplitudes(op, i0, i1, w)              # (i a k c)
        # PP through the particle ladder, (i j a b)
        pp_o, pp_s = df_ladder.ladders(
            Bvv, Bvv, [np.ascontiguousarray(X.transpose(0, 2, 1, 3)) for X in (To, Ts)])
        Nos = -pp_o.transpose(0, 2, 1, 3)
        Nss = -pp_s.transpose(0, 2, 1, 3)
        del pp_o, pp_s
        # P[RC(S + T)] = Q_ia B_jb + B_ia Q_jb
        rows = slice(i0 * nv, i1 * nv)
        y_rc = (Q[rows] @ L.T + L[rows] @ Q.T).reshape(it, nv, no, nv)
        Nos -= y_rc
        y_x = -y_rc
        del y_rc
        Kx_i = _exchange_block(f, i0, i1, Bvv)                  # (k, i, a, c)
        for j0, j1 in gtiles:
            Jo, Js = _scaled_amplitudes(op, j0, j1, w)
            Kx_j = _exchange_block(f, j0, j1, Bvv)              # (k, j, b, c)
            jt = j1 - j0
            # HH: sum_{k in J, l} (ki|lj) T_kalb
            O = (np.ascontiguousarray(f.BooP[:, j0:j1, i0:i1]).reshape(f.naux, -1).T
                 @ f.BooP.reshape(f.naux, -1)).reshape(jt, it, no, no)
            Nos -= np.einsum('kilj,kalb->iajb', O, Jo, optimize=True)
            Nss -= np.einsum('kilj,kalb->iajb', O, Js, optimize=True)
            del O
            # RX(T) on (I, J) and its swap partner, and the T_ikcb (kj|ac) pair
            KJ = np.ascontiguousarray(Kx_j.transpose(0, 3, 1, 2)).reshape(no * nv, -1)
            for X, Y, target in ((To, Jo, 'o'), (Ts, Js, 's')):
                rx = (X.reshape(it * nv, -1) @ KJ).reshape(it, nv, jt, nv)
                rx += np.einsum('jbkc,kiac->iajb', Y, Kx_i, optimize=True)
                if target == 'o':
                    Nos[:, :, j0:j1] += rx
                else:
                    y_x[:, :, j0:j1] += rx
                del rx
            Nos[:, :, j0:j1] += (np.einsum('ickb,kjac->iajb', To, Kx_j, optimize=True)
                                 + np.einsum('jcka,kibc->iajb', Jo, Kx_i, optimize=True))
            del Jo, Js, Kx_j, KJ
        del Kx_i, To, Ts
        Nss += y_x - y_x.transpose(0, 3, 2, 1)
        del y_x
        D = doubles_denominator(f.eo, f.ev, f.eo[i0:i1])
        Nos /= D
        Nss /= D
        del D
        blocks = {}
        for k in job.grounds:
            root = job.roots[k]
            v = np.zeros((no, nv))
            v[i0:i1] = ((Nss + root.s * Nos).reshape(it * nv, -1) @ root.x.ravel()
                        ).reshape(it, nv)
            blocks[k] = v
        del Nos, Nss
        acc.add(t, blocks)
    del Bvv, holder
    return acc.total()


def _pair_terms(job, p, a, b, c, tt_o):
    """{term: alpha block} of the pair p = (l, r)."""
    l, r = p
    L, R = job.roots[l], job.roots[r]
    p0 = {'oo': a['p0_oo'], 'vv': a['p0_vv'], 'ov': a['p0_ov']}
    p1_oo, p1_vv = _p1(L, R)
    rl, rr = a[('r', l)], a[('r', r)]
    out = {'oo_ss0': p1_oo, 'vv_ss0': p1_vv,
           'oo_dd': b[('oo_dd', p)], 'vv_dd': b[('vv_dd', p)],
           'oo_ru': -rl @ rr.T, 'vv_ru': rr.T @ rl,
           'oo_p0': 0.5 * (p1_oo @ p0['oo'] + p0['oo'] @ p1_oo),
           'vv_p0': -0.5 * (p1_vv @ p0['vv'] + p0['vv'] @ p1_vv),
           'oo_tt_o': tt_o[p], 'oo_tt_v': a[('oo_tt_v', p)],
           'vv_tt_v': a[('vv_tt_v', p)], 'vv_tt_o': a[('vv_tt_o', p)],
           'oo_tru': -0.5 * (R.x @ c[('rr', l)].T + c[('rr', r)] @ L.x.T),
           'vv_tru': 0.5 * (c[('rr', l)].T @ R.x + L.x.T @ c[('rr', r)])}
    for side, (m, n) in (('ov', (l, r)), ('vo', (r, l))):
        M, N = job.roots[m], job.roots[n]
        q1_oo, q1_vv = _p1(M, N)
        ov = {'sd': -2.0 * b[('q', n, m)],
              'tsd': 2.0 * c[('tq', m, n)],
              'p0v': -p0['ov'] @ q1_vv,
              'p0o': q1_oo @ p0['ov'],
              'tdv': -N.x @ b[('tu_vv', m)].T,
              'tdo': -b[('tu_oo', m)] @ N.x}
        for k, v in ov.items():
            out[f'{side}_{k}'] = v if side == 'ov' else v.T
    return out


def _ground_terms(job, k, a, b, c, g):
    """{term: alpha block} of <Psi_0|..|k>."""
    root = job.roots[k]
    x = root.x
    return {'ov_s': x,
            'ov_p0v': -0.5 * x @ a['p0_vv'], 'ov_p0o': 0.5 * a['p0_oo'] @ x,
            'ov_ring': 0.5 * c[('rr', k)],
            'vo_t1': -a[('r', k)].T, 'vo_t2': -g[k].T,
            'oo_s': -x @ a['p0_ov'].T, 'oo_d': -b[('tu_oo', k)].T,
            'vv_s': a['p0_ov'].T @ x, 'vv_d': b[('tu_vv', k)]}


def transition_densities(roots, pairs=(), grounds=(), terms=False, tile_gb=EE_RI_TILE_GB):
    """{(bra, ket): (charge, spin)} of the ISR-ADC(2) transition densities
    <bra|a^dag_p a_q|ket> (module docstring), each (no + nv) square in the
    operators' active orbitals.

    roots: {key: (op, y, omega, t1)}, a root of a tiled operator (ee_ri_fold
        .RIFoldADC2 at adc2, one per spin, the same factors, communicator
        and scaling): its unit channel singles y, frequency (Hartree) and
        singles weight.
    pairs: [(bra key, ket key)] of two excited states.
    grounds: [ket key]: the pairs (('ground', 0), ket).
    terms=True returns ({term: charge block}, {term: spin block}) per pair
    instead. A collective over the operators' ranks: every rank calls it with
    the same arguments and gets the same bits."""
    job = _Job(roots, pairs, grounds, tile_gb)
    a = _pass_amplitudes(job)
    b = _pass_doubles(job)
    vectors = {('rr', k): (a[('r', k)], job.roots[k].s) for k in job.used}
    for l, r in job.pairs:
        slr = job.roots[l].s * job.roots[r].s
        vectors[('tq', l, r)] = (b[('q', l, r)], slr)
        vectors[('tq', r, l)] = (b[('q', r, l)], slr)
    c = _pass_t_products(job, vectors)
    tt_o = _pass_virtual_tiles(job) if job.pairs else {}
    g = _pass_second_order(job) if job.grounds else {}
    no, nv = job.no, job.nv
    out = {}
    for key, alpha, sl, sr in (
            [(p, _pair_terms(job, p, a, b, c, tt_o), job.roots[p[0]].s, job.roots[p[1]].s)
             for p in job.pairs]
            + [((GROUND, k), _ground_terms(job, k, a, b, c, g), 1.0, job.roots[k].s)
               for k in job.grounds]):
        charge = {n: (1.0 + sl * sr) * v for n, v in alpha.items()}
        spin = {n: (1.0 - sl * sr) * v for n, v in alpha.items()}
        out[key] = (charge, spin) if terms else (assemble(charge, no, nv),
                                                 assemble(spin, no, nv))
    return out
