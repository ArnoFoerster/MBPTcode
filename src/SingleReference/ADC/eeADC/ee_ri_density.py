"""The ISR-ADC(2) difference density of a folded root on density-fitted tiles.

ee_density writes the unrelaxed ISR-ADC(2) difference density once, in spin
orbitals, for the dense routes. Here the same terms (the names of
ee_density.TERMS) are evaluated for one spin channel of a closed shell from
the tiled operator of ee_ri_fold: the doubles of the root are rebuilt one
occupied tile at a time from its singles, never held whole, and so are the
first-order amplitudes.

Spin cases. The state is a flip eigenvector of parity s (+1 singlet, -1
Ms = 0 triplet), so its density is the same for both spins and the spatial
difference density is twice the alpha block. With the alpha singles x
(x = sqrt(T1) y / sqrt 2, y the unit channel vector, T1 the singles weight),
the spin-orbital doubles are, in the (i a j b) layout of a tile,

    Z_iajb = u2[i_a j_b a_a b_b]      W_iajb = u2[i_a j_a a_a b_a]
    singlet  Z = sqrt(T1) w (H + H^T) / (2 sqrt 2),   W = Z - Z~
    triplet  Z = sqrt(T1) w (H - H^T) / (2 sqrt 2),
             W = sqrt(T1) w (S - S~) / (2 sqrt 2),    S = H + H^T

with H, w = 1/(omega - D) and X~_iajb = X_ibja as in ee_ri_fold, so that
sum_all u2^2 = T1 |Y|^2 is the doubles weight of the full eigenvector; the
first-order amplitudes are tau_iajb = (ia|jb)/(e_a + e_b - e_i - e_j)
(opposite spins) and ts = tau - tau~ (same spin). Under the pair swap
(i a) <-> (j b) tau, ts and W are symmetric and Z has parity s, which moves
every contraction onto the tile's own first index. The spin sums are

    oo_dd   -2 sum_kab [W_iakb W_jakb + 2 Z_iakb Z_jakb]
    vv_dd    2 sum_ijc [W_iajc W_ibjc + 2 Z_iajc Z_ibjc]
    ov_sd   -2 sum_jb x_jb (W_iajb + s Z_iajb)
    r        T_s x,   (T_s v)_ia = sum_jb (ts + s tau)_iajb v_jb
    p0_oo   -1/2 sum_kab [ts_iakb ts_jakb + 2 tau_iakb tau_jakb]
    p0_vv    1/2 sum_ijc [ts_iajc ts_ibjc + 2 tau_iajc tau_ibjc]
    p0_ov    sum_P [sum_j B_jiP G_jaP - sum_c G_icP B_cPa] / (e_i - e_a),
             G_icP = sum_jb (ts + tau)_icjb B_jbP
    oo_tt_o -1/2 sym sum_klcd [ts_ickd ts_jcld + 2 tau_ickd tau_jcld] p1_lk
    oo_tt_v  sym sum_kcdb [ts_ickd ts_jckb + tau_ickd tau_jckb
                           + tau_idkc tau_jbkc] p1vv_db
    vv_tt_v -1/2 sym sum_klcd [ts_kblc ts_kald + 2 tau_kblc tau_kald] p1vv_cd
    vv_tt_o  sym sum_klcj [ts_kblc ts_jalc + tau_kblc tau_jalc
                           + tau_lbkc tau_lajc] p1_jk
    TU_oo    sum_kab [ts_iakb W_jakb + 2 tau_iakb Z_jakb]   (ov_tdo = -TU_oo x)
    TU_vv    sum_ijc [ts_iajc W_ibjc + 2 tau_iajc Z_ibjc]   (ov_tdv = -x TU_vv^T)
    ov_tsd  -T_+ p2_ov   (p2_ov is spin-symmetric whatever the state)

and the rest (oo_ss0, vv_ss0, oo_ru, vv_ru, oo_p0, vv_p0, oo_tru, vv_tru,
ov_p0v, ov_p0o) are products of these no-by-nv and square matrices.

Every term but one contracts the tile's first occupied index away or keeps
it, so it is a sum over occupied tiles. oo_tt_o couples two occupied indices
of DIFFERENT amplitudes through p1_lk, which no occupied tile holds together;
it is a sum over tiles of one VIRTUAL index d instead, (ic|kd) for every i,
c, k and the tile's d. Four passes in all (amplitudes; amplitudes and
doubles; the T products of r and p2_ov; the virtual tiles), each about one
folded product's work, the tiles dealt over the operator's ranks and summed
in tile order (ordered_sum), so the density is the same bits at every rank
count. tests/test_ee_adc_state_specific.py pins every term against
ee_density on the dense spin-free route's full eigenvector.
"""
import numpy as np

from src.Base.constants import EE_RI_TILE_GB
from src.Base.utils.mpi_grid import ordered_sum
from src.SingleReference.ADC.eeADC.ee_density import assemble

_GB = 1024.0 ** 3


def _tiles(n, per, tile_gb, ranks):
    """(start, stop) tiles of an axis of n, each holding at most tile_gb at
    `per` bytes per index and no more than gives every rank one."""
    k = int(max(1, min(-(-n // ranks), tile_gb * _GB // per)))
    return [(k0, min(k0 + k, n)) for k0 in range(0, n, k)]


class _Pass:
    """Addends of one pass, one flat array per tile, summed in tile order."""

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
        tot = ordered_sum(self.parts, self.op.comm, shape=(self.size,),
                          onto=np.zeros(self.size))
        out, p = {}, 0
        for name, shape in self.shapes.items():
            m = int(np.prod(shape))
            out[name] = tot[p:p + m].reshape(shape)
            p += m
        return out


def _amplitudes(op, k0, k1):
    """(tau, ts) of the occupied rows k0..k1-1, shape (kt, nv, no, nv)."""
    f = op.f
    I = op._ints(k0, k1)
    D = (f.ev[None, :, None, None] + f.ev[None, None, None, :]
         - f.eo[k0:k1, None, None, None] - f.eo[None, None, :, None])
    tau = I / D
    del I
    return tau, tau - tau.transpose(0, 3, 2, 1)


def _doubles(op, Bt, k0, k1, omega, scale, s):
    """(Z, W) of the occupied rows k0..k1-1 (module docstring)."""
    f, nv, no = op.f, op.nv, op.no
    kt = k1 - k0
    rows = slice(k0 * nv, k1 * nv)
    H = (Bt[rows] @ f.L.T).reshape(kt, nv, no, nv)
    HT = (f.L[rows] @ Bt.T).reshape(kt, nv, no, nv)
    D = (f.ev[None, :, None, None] + f.ev[None, None, None, :]
         - f.eo[k0:k1, None, None, None] - f.eo[None, None, :, None])
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


def _oo(A, B):
    """sum_{k in tile} sum_ab A[k,b,i,a] B[k,b,j,a] -> (i, j)."""
    no = A.shape[2]
    a = A.transpose(2, 0, 1, 3).reshape(no, -1)
    b = B.transpose(2, 0, 1, 3).reshape(no, -1)
    return a @ b.T


def _vv(A, B):
    """sum_{i in tile} sum_jc A[i,a,j,c] B[i,b,j,c] -> (a, b)."""
    nv = A.shape[1]
    out = np.zeros((nv, nv))
    for i in range(A.shape[0]):
        out += A[i].reshape(nv, -1) @ B[i].reshape(nv, -1).T
    return out


def difference_density(op, y, omega, t1, terms=False, tile_gb=EE_RI_TILE_GB):
    """Delta rho (no + nv square, active spatial orbitals, both spins) of one
    root of the tiled operator `op` (ee_ri_fold.RIFoldADC2): y its unit
    singles in the channel basis, omega its frequency (Hartree), t1 its
    singles weight. terms=True returns {term: spatial block} (ee_density's
    names) instead. A collective over op's ranks: every rank calls it with
    the same arguments and gets the same bits."""
    f, no, nv = op.f, op.no, op.nv
    s = 1.0 if op.spin == 'singlet' else -1.0
    y = np.asarray(y, float).ravel()
    x = (np.sqrt(t1 / 2.0) * y).reshape(no, nv)
    p1_oo = -x @ x.T
    p1_vv = x.T @ x
    out = {'oo_ss0': p1_oo, 'vv_ss0': p1_vv}
    if op.level == 'adc1':
        return _finish(out, no, nv, terms)
    ranks = 1 if op.comm is None else op.comm.Get_size()
    rank = 0 if op.comm is None else op.comm.Get_rank()
    # occupied tiles: about nine (kt, nv, no, nv) arrays live in the doubles
    # pass (tau, ts, Z, W and their temporaries)
    otiles = _tiles(no, 8.0 * nv * no * nv * 9, tile_gb, ranks)
    mine = [(t, kk) for t, kk in enumerate(otiles) if t % ranks == rank]

    # ---- pass 1: amplitudes alone ------------------------------------------
    p1 = _Pass(op, {'r': (no, nv), 'p0_oo': (no, no), 'p0_vv': (nv, nv),
                    'p0_ov': (no, nv), 'oo_tt_v': (no, no),
                    'vv_tt_v': (nv, nv), 'vv_tt_o': (nv, nv)})
    for t, (k0, k1) in mine:
        kt = k1 - k0
        tau, ts = _amplitudes(op, k0, k1)
        r = np.zeros((no, nv))
        r[k0:k1] = ((ts + s * tau).reshape(kt * nv, -1) @ x.ravel()).reshape(kt, nv)
        G = (ts + tau).reshape(kt * nv, -1) @ f.L                  # G_icP
        num = 2.0 * op._back(G, k0, k1).reshape(no, nv)
        del G
        blocks = {'r': r, 'p0_ov': num,
                  'p0_oo': -0.5 * (_oo(ts, ts) + 2.0 * _oo(tau, tau)),
                  'p0_vv': 0.5 * (_vv(ts, ts) + 2.0 * _vv(tau, tau))}
        # oo_tt_v: k is the tile; R[k,d,j,c] = sum_b p1vv_db X[k,b,j,c]
        Rs = np.einsum('db,kbjc->kdjc', p1_vv, ts, optimize=True)
        Rt = np.einsum('db,kbjc->kdjc', p1_vv, tau, optimize=True)
        Rx = np.einsum('kcjb,db->kcjd', tau, p1_vv, optimize=True)
        blocks['oo_tt_v'] = _oo(ts, Rs) + _oo(tau, Rt) + \
            np.einsum('kcid,kcjd->ij', tau, Rx, optimize=True)
        # vv_tt_v: R[k,a,l,c] = sum_d X[k,a,l,d] p1vv_cd
        Rs = np.einsum('kald,cd->kalc', ts, p1_vv, optimize=True)
        Rt = np.einsum('kald,cd->kalc', tau, p1_vv, optimize=True)
        blocks['vv_tt_v'] = -0.5 * (_vv(Rs, ts) + 2.0 * _vv(Rt, tau))
        # vv_tt_o: l is the tile; R[l,c,j,b] = sum_k p1_jk X[l,c,k,b]
        Rs = np.einsum('jk,lckb->lcjb', p1_oo, ts, optimize=True)
        Rt = np.einsum('jk,lckb->lcjb', p1_oo, tau, optimize=True)
        blocks['vv_tt_o'] = (np.einsum('lcjb,lcja->ab', Rs, ts, optimize=True)
                             + np.einsum('lcjb,lcja->ab', Rt, tau, optimize=True)
                             + np.einsum('lbjc,lajc->ab', Rt, tau, optimize=True))
        del Rs, Rt, Rx, tau, ts
        p1.add(t, blocks)
    a = p1.total()
    r = a['r']
    d_ia = f.eo[:, None] - f.ev[None, :]
    p0_ov = a['p0_ov'] / (2.0 * d_ia)
    sym = _sym
    out.update({
        'oo_ru': -r @ r.T, 'vv_ru': r.T @ r,
        'oo_p0': sym(p1_oo @ a['p0_oo']), 'vv_p0': -sym(p1_vv @ a['p0_vv']),
        'oo_tt_v': sym(a['oo_tt_v']), 'vv_tt_v': sym(a['vv_tt_v']),
        'vv_tt_o': sym(a['vv_tt_o']),
        'ov_p0v': -p0_ov @ p1_vv, 'ov_p0o': p1_oo @ p0_ov,
    })

    # ---- pass 2: amplitudes and the doubles ---------------------------------
    p2 = _Pass(op, {'oo_dd': (no, no), 'vv_dd': (nv, nv), 'p2_ov': (no, nv),
                    'tu_oo': (no, no), 'tu_vv': (nv, nv)})
    Bt = op._dressed(y)
    scale = np.sqrt(t1) / (2.0 * np.sqrt(2.0))
    for t, (k0, k1) in mine:
        kt = k1 - k0
        Z, W = _doubles(op, Bt, k0, k1, omega, scale, s)
        p2_ov = np.zeros((no, nv))
        p2_ov[k0:k1] = -2.0 * ((W + s * Z).reshape(kt * nv, -1) @ x.ravel()
                               ).reshape(kt, nv)
        blocks = {'oo_dd': -2.0 * (_oo(W, W) + 2.0 * _oo(Z, Z)),
                  'vv_dd': 2.0 * (_vv(W, W) + 2.0 * _vv(Z, Z)),
                  'p2_ov': p2_ov}
        tau, ts = _amplitudes(op, k0, k1)
        blocks['tu_oo'] = _oo(ts, W) + 2.0 * s * _oo(tau, Z)
        blocks['tu_vv'] = _vv(ts, W) + 2.0 * _vv(tau, Z)
        del Z, W, tau, ts
        p2.add(t, blocks)
    del Bt
    b = p2.total()
    p2_ov = b['p2_ov']
    out.update({'oo_dd': b['oo_dd'], 'vv_dd': b['vv_dd'], 'ov_sd': p2_ov,
                'ov_tdv': -x @ b['tu_vv'].T, 'ov_tdo': -b['tu_oo'] @ x})

    # ---- pass 3: T_s r and T_+ p2_ov -----------------------------------------
    p3 = _Pass(op, {'tr': (no, nv), 'tp': (no, nv)})
    for t, (k0, k1) in mine:
        kt = k1 - k0
        tau, ts = _amplitudes(op, k0, k1)
        tr, tp = np.zeros((no, nv)), np.zeros((no, nv))
        tr[k0:k1] = ((ts + s * tau).reshape(kt * nv, -1) @ r.ravel()).reshape(kt, nv)
        tp[k0:k1] = ((ts + tau).reshape(kt * nv, -1) @ p2_ov.ravel()).reshape(kt, nv)
        del tau, ts
        p3.add(t, {'tr': tr, 'tp': tp})
    c = p3.total()
    out.update({'oo_tru': -sym(x @ c['tr'].T), 'vv_tru': sym(c['tr'].T @ x),
                'ov_tsd': -c['tp']})

    # ---- pass 4: oo_tt_o on virtual tiles -----------------------------------
    vtiles = _tiles(nv, 8.0 * no * nv * no * 6, tile_gb, ranks)
    p4 = _Pass(op, {'s': (no, no)})
    L3 = f.L.reshape(no, nv, -1)
    for t, (d0, d1) in enumerate(vtiles):
        if t % ranks != rank:
            continue
        dt = d1 - d0
        Ld = np.ascontiguousarray(L3[:, d0:d1]).reshape(no * dt, -1)
        I = (f.L @ Ld.T).reshape(no, nv, no, dt)                 # (ic|kd)
        D = (f.ev[None, :, None, None] + f.ev[None, None, None, d0:d1]
             - f.eo[:, None, None, None] - f.eo[None, None, :, None])
        tau = I / D
        del I, D
        ts = tau - tau.transpose(2, 1, 0, 3)                     # - tau_idkc
        S = np.zeros((no, no))
        for X, wgt in ((ts, 1.0), (tau, 2.0)):
            R = np.einsum('lk,jcld->jckd', p1_oo, X, optimize=True)
            S += wgt * (X.reshape(no, -1) @ R.reshape(no, -1).T)
        del tau, ts
        p4.add(t, {'s': S})
    out['oo_tt_o'] = -0.5 * sym(p4.total()['s'])
    return _finish(out, no, nv, terms)


def _sym(A):
    return 0.5 * (A + A.T)


def _finish(alpha, no, nv, terms):
    """Both spins: twice the alpha blocks, as terms or as one matrix."""
    both = {k: 2.0 * v for k, v in alpha.items()}
    return both if terms else assemble(both, no, nv)
