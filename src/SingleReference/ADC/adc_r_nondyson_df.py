"""Restricted non-Dyson IP- or EA-ADC(3), matrix-free and density-fitted, built
from the blocks of the Dyson DF operator (adc_r_sigma_df.build_pieces).

Non-Dyson ADC(3) keeps one sector of the Dyson supermatrix explicitly and folds
the other into a static orbital block. For the IP sector (holes i, j):

    H = [ M        U_(o,2h1p) ]      M = eps + Sigma(inf) + (G + G^T) / 2
        [ U^T      K + C      ]

with U, K and the first-order C of the 2h1p sector exactly as in Dyson
ADC(3), Sigma(inf) the static self-energy (unrelaxed MP2 density for the
standard method, Sigma(3)), and G[:, j] the fold of the 2p1h sector at
omega = eps_j, through third order:

    G[:, j] = U1 R U1_j + U1 R U2_j + U2 R U1_j + U1 R C1 R U1_j,
    R = (eps_j - K)^-1,

U1 and U2 the first- and second-order parts of the coupling, U_j its row j.
Averaging over omega = eps_i and omega = eps_j is the symmetrization of
Schirmer, Trofimov and Stelter, J. Chem. Phys. 109, 4734 (1998). The EA sector
is the mirror image (particles a, b; 2p1h explicit, 2h1p folded). With
Sigma(inf) from the unrelaxed MP2 density this is the standard non-Dyson
ADC(3) of the intermediate-state representation (nondyson_restricted) and of
pyscf.adc, to rounding.

Cost: the fold takes one application of the folded sector's C per orbital of
the kept sector (occupied orbitals for IP, virtual ones for EA), the MP3-like
step of non-Dyson ADC(3); every block is streamed as in the Dyson operator.
The fold works on batches of columns at once, forms the particle-particle ladder
from (ac|bd) slabs built once per batch, and, when memory allows, stores the rows
of U of the kept sector so that U acts on a batch as one matrix product.

Parallelism: static_block splits the fold columns over the ranks of `comm` in
contiguous blocks and gathers G; the matvec of the kept sector is distributed
like the Dyson one (adc_r_sigma_df).
"""
import math

import numpy as np

from src.Base.utils.parallel.mpi_grid import (NodeShared, allgather_blocks, contiguous_block, current_comm,
                                              distributed)
from src.Base.utils.cached_einsum import einsum as _einsum
from src.SingleReference.common.df_ladder import ladder_slabs
from src.SingleReference.common.mp_amplitudes import occ_chunk
from src.SingleReference.ADC.adc_r_sigma_df import c2h1p_df_chunks, c2p1h_df_chunks, build_pieces
from src.SingleReference.ADC.adc_utils import b_chunk_einsum
from src.SingleReference.ADC import adc_r_nondyson_otf as otf
from src.SingleReference.common.mp_amplitudes import (u2_2h1p_amplitude_chunks,
                                                      u2_2p1h_amplitude_chunks)

SECTORS = ('IP', 'EA')


def _sector_maps(P, U, sector):
    """(adj, fwd) of the coupling U between the orbitals and one satellite
    sector ('2h1p' or '2p1h'): adj maps an orbital vector to the sector's three
    CSF blocks, fwd maps three blocks back to an orbital vector."""
    U_I, U_Ip, f2h, a2h, f2p, a2p = U
    O, V, nP_o, nP_v = P.O, P.V, P.nP_o, P.nP_v
    if sector == '2h1p':
        def adj(zp):
            u2, u3 = a2h(zp)
            return ((U_I.T @ zp).reshape(O, V), u2.reshape(nP_o, V), u3.reshape(nP_o, V))

        def fwd(z):
            return U_I @ z[0].reshape(-1) + f2h(z[1], z[2])
    else:
        def adj(zp):
            u2, u3 = a2p(zp)
            return ((U_Ip.T @ zp).reshape(O, V), u2.reshape(O, nP_v), u3.reshape(O, nP_v))

        def fwd(z):
            return U_Ip @ z[0].reshape(-1) + f2p(z[1], z[2])
    return adj, fwd


def _sector_layout(P, sector):
    """K diagonals (three blocks, shaped) and the C applier of one sector."""
    O, V, nP_o, nP_v = P.O, P.V, P.nP_o, P.nP_v
    if sector == '2h1p':
        K = (P.e_I.reshape(O, V), P.e_II.reshape(nP_o, V), P.e_III.reshape(nP_o, V))
        return K, P.c_2h1p
    K = (P.e_Ip.reshape(O, V), P.e_IIp.reshape(O, nP_v), P.e_IIIp.reshape(O, nP_v))
    return K, P.c_2p1h


# ---------------------------------------------------------------------------
# Column-batched appliers for the fold. They repeat the single-vector appliers of
# adc_r_sigma_df term by term with a trailing column axis 'w', so one pass over
# the streamed amplitude and integral chunks serves a whole batch of columns.
# Bare route only (no screening, no EN shifts), as build_operator requires.
# ---------------------------------------------------------------------------

def _pairs_to_full(z, iu, ju, n, sign):
    """(n_pair, m, w) on i<j -> (n, n, m, w) with the (j, i) entries sign * z."""
    Z = np.zeros((n, n) + z.shape[1:])
    Z[iu, ju] = z
    Z[ju, iu] = sign * z
    return Z


def _u_2h1p_batch(P, chunk=None):
    """(adj, fwd) of the streamed coupling U between the orbitals and the 2h1p
    sector for column batches: adj (norb, w) -> three blocks (O,V,w),
    (nP_o,V,w), (nP_o,V,w); fwd the reverse."""
    c, O, V, norb = P.ctx, P.O, P.V, P.norb
    U_I = P.U[0]
    s32, s12 = math.sqrt(1.5), math.sqrt(0.5)
    specs = ((s32, 'A_anti', -1.0, +1.0, True), (s12, 'W1', +1.0, -1.0, False))
    naux = c.B.shape[0]
    qc = 32

    def chunks():
        return u2_2h1p_amplitude_chunks(c.B, O, V, norb, c.eps_o, c.eps_v, chunk or occ_chunk(O))

    def fwd(z):
        zI, zII, zIII = z
        nw = zI.shape[-1]
        y = U_I @ zI.reshape(O * V, nw)
        Zs = [_pairs_to_full(zz, c.iu_o, c.ju_o, O, -1.0 if anti else 1.0)
              for (_, _, _, _, anti), zz in zip(specs, (zII, zIII))]
        for (sc, _, _, _, _), Zf in zip(specs, Zs):
            M = _einsum('Qja,ijaw->Qiw', c.Bov_full, Zf, optimize=True)
            y += sc * b_chunk_einsum('Qip,Qiw->pw', (c.Bo_full, M), (True, True), c.Bo_full)
        for lo, hi, t_same_c, _t_opp, W1c, _W2, A_mix_c, X_ijcd_c in chunks():
            amps = {'A_anti': t_same_c, 'W1': W1c}
            for (sc, akey, qsign, ysign, _), Zf in zip(specs, Zs):
                Z = Zf[lo:hi]
                for st in range(0, naux, qc):
                    sl = slice(st, st + qc)
                    T = _einsum('Qda,ijaw->Qijdw', c.Bv_pp[sl], Z, optimize=True)
                    Pc = _einsum('ijcd,Qijdw->Qcw', X_ijcd_c, T, optimize=True)
                    y -= sc * _einsum('Qcp,Qcw->pw', c.Bv_full[sl], Pc, optimize=True)
                S = _einsum('ikca,ijaw->kcjw', amps[akey], Z, optimize=True)
                R = _einsum('Qjk,kcjw->Qcw', c.Boo_full, S, optimize=True)
                y += qsign * sc * b_chunk_einsum('Qcp,Qcw->pw', (c.Bv_full, R), (True, True), c.Bv_full)
                S4 = _einsum('ikca,ijaw->kcjw', A_mix_c, Z, optimize=True)
                R4 = _einsum('Qck,kcjw->Qjw', c.Bvo_full, S4, optimize=True)
                y += ysign * sc * b_chunk_einsum('Qjp,Qjw->pw', (c.Bo_full, R4), (True, True), c.Bo_full)
        return y

    def adj(zp):
        nw = zp.shape[-1]
        bzo = b_chunk_einsum('Qip,pw->Qiw', (c.Bo_full, zp), (True, False), c.Bo_full, (naux, O, nw))
        bzv = b_chunk_einsum('Qcp,pw->Qcw', (c.Bv_full, zp), (True, False), c.Bv_full, (naux, V, nw))
        gz1 = _einsum('Qiw,Qja->ijaw', bzo, c.Bov_full, optimize=True)
        gvo = _einsum('Qcw,Qjk->cjkw', bzv, c.Boo_full, optimize=True)
        gvh = _einsum('Qck,Qjw->cjkw', c.Bvo_full, bzo, optimize=True)
        tp = np.zeros((O, O, V, nw))
        y1 = np.zeros((O, O, V, nw))
        q3 = {'A_anti': np.zeros((O, O, V, nw)), 'W1': np.zeros((O, O, V, nw))}
        for lo, hi, t_same_c, _t_opp, W1c, _W2, A_mix_c, X_ijcd_c in chunks():
            amps = {'A_anti': t_same_c, 'W1': W1c}
            for st in range(0, naux, qc):
                sl = slice(st, st + qc)
                Xb = _einsum('ijcd,Qcw->Qijdw', X_ijcd_c, bzv[sl], optimize=True)
                tp[lo:hi] += _einsum('Qijdw,Qda->ijaw', Xb, c.Bv_pp[sl], optimize=True)
            y1[lo:hi] = _einsum('ikca,cjkw->ijaw', A_mix_c, gvh, optimize=True)
            for akey in ('A_anti', 'W1'):
                q3[akey][lo:hi] = _einsum('ikca,cjkw->ijaw', amps[akey], gvo, optimize=True)
        out = [(U_I.T @ zp).reshape(O, V, nw)]
        for sc, akey, qsign, ysign, anti in specs:
            sgn = -1.0 if anti else 1.0
            tot = gz1 - tp + qsign * q3[akey] + ysign * y1
            tot = sc * (tot + sgn * tot.transpose(1, 0, 2, 3))
            out.append(tot[c.iu_o, c.ju_o])
        return tuple(out)
    return adj, fwd


def _u_2p1h_batch(P, chunk=None):
    """(adj, fwd) of the streamed coupling U between the orbitals and the 2p1h
    sector for column batches: blocks (O,V,w), (O,nP_v,w), (O,nP_v,w)."""
    c, O, V, norb = P.ctx, P.O, P.V, P.norb
    U_Ip = P.U[1]
    s32, s12 = math.sqrt(1.5), math.sqrt(0.5)
    specs = ((s32, 'anti', 'A_anti_p', -1.0, True), (s12, 'sym', 'W1p', +1.0, False))
    gk_sign = {'anti': -1.0, 'sym': +1.0}
    naux = c.B.shape[0]

    def chunks():
        return u2_2p1h_amplitude_chunks(c.B, O, V, norb, c.eps_o, c.eps_v, chunk or occ_chunk(O))

    def unfold(z, anti):
        Z = np.zeros((O, V, V) + z.shape[2:])
        Z[:, c.iu_v, c.ju_v] = z
        Z[:, c.ju_v, c.iu_v] = -z if anti else z
        return Z

    def fwd(z):
        zIp, zIIp, zIIIp = z
        nw = zIp.shape[-1]
        y = U_Ip @ zIp.reshape(O * V, nw)
        Zs = [unfold(zz, anti) for (_, _, _, _, anti), zz in zip(specs, (zIIp, zIIIp))]
        for (sc, _, _, _, _), Zf in zip(specs, Zs):
            M = _einsum('Qbi,iabw->Qaw', c.Bvo_full, Zf, optimize=True)
            y += sc * b_chunk_einsum('Qap,Qaw->pw', (c.Bv_full, M), (True, True), c.Bv_full)
        for lo, hi, tp_same_c, _tp_opp, W1p_c, W2p_c, X_abkl_c in chunks():
            amps = {'A_anti_p': tp_same_c, 'W1p': W1p_c}
            for (sc, gkey, akey, hsign, _), Zf in zip(specs, Zs):
                Pk = _einsum('abkl,iabw->kliw', X_abkl_c, Zf, optimize=True)
                R1 = _einsum('Qli,kliw->Qkw', c.Boo_full[:, lo:hi, :], Pk, optimize=True)
                yA = b_chunk_einsum('Qkp,Qkw->pw', (c.Bo_full, R1), (True, True), c.Bo_full)
                R2 = _einsum('Qki,kliw->Qlw', c.Boo_full, Pk, optimize=True)
                yB = b_chunk_einsum('Qlp,Qlw->pw', (c.Bo_full[:, lo:hi], R2), (True, True), c.Bo_full)
                y += 0.5 * sc * (yA + gk_sign[gkey] * yB)
                Z = Zf[lo:hi]
                Y = _einsum('acki,iabw->ckbw', amps[akey], Z, optimize=True)
                R = b_chunk_einsum('Qbc,ckbw->Qkw', (c.Bv_pp, Y), (True, False), c.Bv_pp, (naux, O, nw))
                y += sc * b_chunk_einsum('Qkw,Qkp->pw', (R, c.Bo_full), (True, True), c.Bo_full)
                Y2 = _einsum('acki,iabw->ckbw', W2p_c, Z, optimize=True)
                R2 = b_chunk_einsum('Qkc,ckbw->Qbw', (c.Bov_full, Y2), (True, False), c.Bov_full, (naux, V, nw))
                y += hsign * sc * b_chunk_einsum('Qbw,Qbp->pw', (R2, c.Bv_full), (True, True), c.Bv_full)
        return y

    def adj(zp):
        nw = zp.shape[-1]
        bz = b_chunk_einsum('Qcp,pw->Qcw', (c.Bv_full, zp), (True, False), c.Bv_full, (naux, V, nw))
        boz = b_chunk_einsum('Qkp,pw->Qkw', (c.Bo_full, zp), (True, False), c.Bo_full, (naux, O, nw))
        gz = _einsum('Qaw,Qbi->abiw', bz, c.Bvo_full, optimize=True)
        gzr = _einsum('Qkw,Qbc->kbcw', boz, c.Bv_pp, optimize=True)
        gzm = _einsum('Qkc,Qbw->kbcw', c.Bov_full, bz, optimize=True)
        direct = _einsum('Qkw,Qli->kliw', boz, c.Boo_full, optimize=True)
        exch = _einsum('Qki,Qlw->kliw', c.Boo_full, boz, optimize=True)
        gkz = {g: direct + gk_sign[g] * exch for g in ('anti', 'sym')}
        abkl = {a: np.zeros((V, V, O, nw)) for a in ('A_anti_p', 'W1p')}
        Gz = {a: np.zeros((V, V, O, nw)) for a in ('A_anti_p', 'W1p')}
        Hz = {a: np.zeros((V, V, O, nw)) for a in ('A_anti_p', 'W1p')}
        for lo, hi, tp_same_c, _tp_opp, W1p_c, W2p_c, X_abkl_c in chunks():
            amps = {'A_anti_p': tp_same_c, 'W1p': W1p_c}
            for sc, gkey, akey, hsign, anti in specs:
                abkl[akey] += _einsum('abkl,kliw->abiw', X_abkl_c, gkz[gkey][:, lo:hi], optimize=True)
                Gz[akey][:, :, lo:hi] = _einsum('acki,kbcw->abiw', amps[akey], gzr, optimize=True)
                Hz[akey][:, :, lo:hi] = _einsum('acki,kbcw->abiw', W2p_c, gzm, optimize=True)
        out = [(U_Ip.T @ zp).reshape(O, V, nw)]
        for sc, gkey, akey, hsign, anti in specs:
            sgn = -1.0 if anti else 1.0
            sw = (1, 0, 2, 3)
            tot = sc * (gz + sgn * gz.transpose(sw))
            tot = tot + sc * abkl[akey]
            tot = tot + sc * (Gz[akey] + sgn * Gz[akey].transpose(sw))
            tot = tot + hsign * sc * (Hz[akey] + sgn * Hz[akey].transpose(sw))
            out.append(tot.transpose(2, 0, 1, 3)[:, c.iu_v, c.ju_v])
        return tuple(out)
    return adj, fwd


def _c_2h1p_batch(P, chunk=None):
    """First-order C of the 2h1p sector (no K) on column batches; chunk: occupied
    indices per streamed chunk (default occ_chunk)."""
    c, O, V, norb = P.ctx, P.O, P.V, P.norb
    iu, ju = c.iu_o, c.ju_o
    Boo = c.B[:, :O, :O]
    g_oooo = _einsum('Qik,Qjl->ijkl', Boo, Boo, optimize=True)     # O^4, small

    def apply(z_I, z_II, z_III):
        nw = z_I.shape[-1]
        Vfull_II = _pairs_to_full(z_II, iu, ju, O, -1.0)
        Vup_II = np.zeros((O, O, V, nw))
        Vup_II[iu, ju] = z_II
        Vup_III = np.zeros((O, O, V, nw))
        Vup_III[iu, ju] = z_III
        T1 = np.zeros((O, V, O, nw)); T2 = np.zeros((O, V, O, nw))
        BV1 = np.zeros((O, O, V, nw)); BV2 = np.zeros((O, O, V, nw))
        t1_I_II = np.zeros((O, V, nw)); t2_I_II = np.zeros((O, V, nw))
        W_II_I = np.zeros((O, O, V, nw))
        t1_I_III = np.zeros((O, V, nw)); t2_I_III = np.zeros((O, V, nw))
        Wb_III_I = np.zeros((O, O, V, nw))
        Tu_III = np.zeros((O, O, V, nw)); Tl_III = np.zeros((O, O, V, nw))
        Tu_II = np.zeros((O, O, V, nw)); Tl_II = np.zeros((O, O, V, nw))
        for lo, hi, m1a_c, m1b_c in c2h1p_df_chunks(c.B, O, V, norb, chunk_size=chunk):
            bracket = m1a_c - 0.5 * m1b_c
            M1 = -m1a_c + 1.5 * m1b_c
            M2 = -1.5 * m1b_c + m1a_c
            Vc = Vfull_II[:, lo:hi]
            T1 += _einsum('iamb,kmbw->iakw', M1, Vc, optimize=True)
            T2 += _einsum('jamb,kmbw->jakw', M2, Vc, optimize=True)
            V3a = Vup_III[:, lo:hi]
            BV1 += _einsum('xamb,Kmbw->Kxaw', bracket, V3a, optimize=True)
            BV2 += _einsum('xamb,mKbw->Kxaw', bracket, Vup_III[lo:hi], optimize=True)
            t1_I_III += _einsum('ialc,ilcw->iaw', bracket, V3a, optimize=True)
            t2_I_III += _einsum('iaxc,xicw->iaw', bracket, Vup_III[lo:hi], optimize=True)
            Wb_III_I[:, lo:hi] = _einsum('iaxc,iaw->ixcw', bracket, z_I, optimize=True)
            V2a = Vup_II[:, lo:hi]
            t1_I_II += _einsum('ialc,ilcw->iaw', m1b_c, V2a, optimize=True)
            t2_I_II += _einsum('iaxc,xicw->iaw', m1b_c, Vup_II[lo:hi], optimize=True)
            W_II_I[:, lo:hi] = _einsum('iaxc,iaw->ixcw', m1b_c, z_I, optimize=True)
            Tu_III += _einsum('xanc,Kncw->Kxaw', m1b_c, V3a, optimize=True)
            Tl_III += _einsum('xamc,mKcw->Kxaw', m1b_c, Vup_III[lo:hi], optimize=True)
            Tu_II += _einsum('xanc,Kncw->Kxaw', m1b_c, V2a, optimize=True)
            Tl_II += _einsum('xamc,mKcw->Kxaw', m1b_c, Vup_II[lo:hi], optimize=True)
        d_II = _einsum('ijkl,klaw->ijaw', g_oooo, Vfull_II, optimize=True)
        e_II = _einsum('ijlk,klaw->ijaw', g_oooo, Vfull_II, optimize=True)
        c_II_II = -0.5 * (d_II - e_II)[iu, ju] + T1[iu, :, ju] + T2[ju, :, iu]
        d_III = _einsum('ijkl,klaw->ijaw', g_oooo, Vup_III, optimize=True)
        e_III = _einsum('ijlk,klaw->ijaw', g_oooo, Vup_III, optimize=True)
        c_III_III = (-d_III[iu, ju] - e_III[iu, ju] + BV1[iu, ju] + BV2[iu, ju]
                     + BV1[ju, iu] + BV2[ju, iu])
        c_I_II = -c.s32 * (t1_I_II - t2_I_II)
        c_II_I = -c.s32 * (W_II_I[iu, ju] - W_II_I[ju, iu])
        c_I_III = (-c.s2 * _einsum('ip,paw->iaw', c.g_ii_pair, z_III, optimize=True)
                   + c.s2 * t1_I_III + c.s2 * t2_I_III)
        c_III_I = (-c.s2 * _einsum('ip,iaw->paw', c.g_ii_pair, z_I, optimize=True)
                   + c.s2 * Wb_III_I[iu, ju] + c.s2 * Wb_III_I[ju, iu])
        c_II_III = 0.5 * c.s3 * (Tu_III[ju, iu] - Tu_III[iu, ju] + Tl_III[ju, iu] - Tl_III[iu, ju])
        c_III_II = (-0.5 * c.s3 * (Tu_II[iu, ju] + Tu_II[ju, iu])
                    + 0.5 * c.s3 * (Tl_II[ju, iu] + Tl_II[iu, ju]))
        c_I_I = (-_einsum('ik,kaw->iaw', c.g_ii_kk, z_I, optimize=True)
                 + _einsum('iac,icw->iaw', c.inner_I, z_I, optimize=True))
        return (c_I_I + c_I_II + c_I_III, c_II_II + c_II_I + c_II_III,
                c_III_III + c_III_I + c_III_II)
    return apply


def _pp_pair_factor(Bv):
    """(V V, naux) contiguous: the vir-vir DF factor with the pair index (a c) leading,
    the operand of every (ac|bd) slab of _pp_ladders."""
    Bv = np.ascontiguousarray(Bv)     # a strided stack sends matmul down its slow path
    naux, V = Bv.shape[0], Bv.shape[1]
    return np.ascontiguousarray(Bv.transpose(1, 2, 0).reshape(V * V, naux))


def _pp_ladders(Bv, jobs, slab_gb=2.0, B2=None):
    """Particle-particle ladders of several column batches in one pass:
    X[i, a, b, w] = sum_{c,d} (ac|bd) M[i, c, d, w] on the pairs a < b (or a <= b),
    with (ac|bd) = sum_q Bv[q, a, c] Bv[q, b, d]. jobs: list of (z, iu, ju,
    antisymmetric), z (O, n_pair, w); returns one (O, n_pair, w) per job.
    The (ac|bd) integrals are formed once per slab of a (df_ladder.ladder_slabs) and
    contracted with the columns of every job as one matrix product.
    B2: _pp_pair_factor(Bv) when the caller holds it already (built once per fold)."""
    V = Bv.shape[1]
    O = jobs[0][0].shape[0]
    Ms = []
    for z, iu, ju, anti in jobs:
        nw = z.shape[-1]
        M = np.zeros((O, nw, V, V))                                 # (i, w, c, d)
        zt = z.transpose(0, 2, 1)
        M[:, :, iu, ju] = zt
        M[:, :, ju, iu] = -zt if anti else zt
        Ms.append(M.reshape(O * nw, V * V))
    Mall = np.concatenate(Ms) if len(Ms) > 1 else Ms[0]             # rows (job, i, w)
    del Ms
    edges = np.cumsum([0] + [O * j[0].shape[-1] for j in jobs])
    outs = [np.empty((O, j[1].size, j[0].shape[-1])) for j in jobs]
    for a0, a1, X in ladder_slabs(Bv, Bv, Mall, slab_gb, pair_factor=B2):          # X[row, a - a0, b]
        for (z, iu, ju, anti), e0, e1, out in zip(jobs, edges[:-1], edges[1:], outs):
            lo, hi = np.searchsorted(iu, a0), np.searchsorted(iu, a1)
            if hi > lo:
                Xj = X[e0:e1].reshape(O, -1, a1 - a0, V)
                out[:, lo:hi] = Xj[:, :, iu[lo:hi] - a0, ju[lo:hi]].transpose(0, 2, 1)
    return outs


def _c_2p1h_batch(P, pp_pair=None, chunk=None):
    """First-order C of the 2p1h sector (no K) on column batches. pp_pair: the
    pair-major vir-vir factor (_pp_pair_factor) when the caller holds it; chunk:
    occupied indices per streamed chunk (default occ_chunk)."""
    c, O, V, norb = P.ctx, P.O, P.V, P.norb
    A, Bi = c.iu_v, c.ju_v

    def upper(z):
        Z = np.zeros((O, V, V) + z.shape[2:])
        Z[:, A, Bi] = z
        return Z

    def apply(z_Ip, z_IIp, z_IIIp):
        nw = z_Ip.shape[-1]
        Vk2, Vk3 = upper(z_IIp), upper(z_IIIp)
        A2 = [np.zeros((O, V, V, nw)) for _ in range(4)]
        A3 = [np.zeros((O, V, V, nw)) for _ in range(4)]
        t1_2 = np.zeros((O, V, nw)); t2_2 = np.zeros((O, V, nw))
        t1_3 = np.zeros((O, V, nw)); t2_3 = np.zeros((O, V, nw))
        Wc = np.zeros((O, V, V, nw)); Wd1 = np.zeros((O, V, V, nw))
        for lo, hi, g_ovov_c, g_ovvo_c in c2p1h_df_chunks(c.B, O, V, norb, chunk_size=chunk):
            for Apps, Vk in ((A2, Vk2[lo:hi]), (A3, Vk3[lo:hi])):
                Apps[0] += _einsum('kXin,kYnw->iXYw', g_ovov_c, Vk, optimize=True)
                Apps[1] += _einsum('kXni,kYnw->iXYw', g_ovvo_c, Vk, optimize=True)
                Apps[2] += _einsum('kXim,kmYw->iXYw', g_ovov_c, Vk, optimize=True)
                Apps[3] += _einsum('kXmi,kmYw->iXYw', g_ovvo_c, Vk, optimize=True)
            Vk2c, Vk3c = Vk2[lo:hi], Vk3[lo:hi]
            t1_2 += c.s32 * _einsum('kaDi,kaDw->iaw', g_ovvo_c, Vk2c, optimize=True)
            t2_2 += c.s32 * _einsum('kaCi,kCaw->iaw', g_ovvo_c, Vk2c, optimize=True)
            Wc[lo:hi] = _einsum('kXYi,iXw->kXYw', g_ovvo_c, z_Ip, optimize=True)
            t1_3 += c.s2 * (-_einsum('kaiD,kaDw->iaw', g_ovov_c, Vk3c, optimize=True)
                            + 0.5 * _einsum('kaDi,kaDw->iaw', g_ovvo_c, Vk3c, optimize=True))
            t2_3 += c.s2 * (-_einsum('kaiC,kCaw->iaw', g_ovov_c, Vk3c, optimize=True)
                            + 0.5 * _einsum('kaCi,kCaw->iaw', g_ovvo_c, Vk3c, optimize=True))
            Wd1[lo:hi] = (-_einsum('kXiY,iXw->kXYw', g_ovov_c, z_Ip, optimize=True)
                          + 0.5 * _einsum('kXYi,iXw->kXYw', g_ovvo_c, z_Ip, optimize=True))
        c_Ip = ((t1_2 - t2_2) + c.s2 * _einsum('ipw,ap->iaw', z_IIIp, c.g_aa_cd, optimize=True)
                + t1_3 + t2_3)
        c_IIp = c.s32 * Wc[:, A, Bi] - c.s32 * Wc[:, Bi, A]
        c_IIIp = (c.s2 * _einsum('iaw,ap->ipw', z_Ip, c.g_aa_cd, optimize=True)
                  + c.s2 * Wd1[:, A, Bi] + c.s2 * Wd1[:, Bi, A])
        c_Ip_Ip = (_einsum('icw,ac->iaw', z_Ip, c.g_aabb, optimize=True)
                   + _einsum('iak,kaw->iaw', c.inner_Ip, z_Ip, optimize=True))
        L_II, L_III = _pp_ladders(c.Bv_pp, [(z_IIp, A, Bi, True), (z_IIIp, A, Bi, False)],
                                  B2=pp_pair)
        a1, a2, a3, a4 = A2
        IIp_IIp = (L_II
                   - a3[:, A, Bi] + a1[:, A, Bi] + 1.5 * a2[:, Bi, A] - a1[:, Bi, A]
                   - 1.5 * a4[:, Bi, A] + a3[:, Bi, A] - 1.5 * a2[:, A, Bi] + 1.5 * a4[:, A, Bi])
        IIIp_IIp = 0.5 * c.s3 * (a2[:, Bi, A] + a2[:, A, Bi]) - 0.5 * c.s3 * (a4[:, A, Bi] + a4[:, Bi, A])
        b1, b2, b3, b4 = A3
        IIIp_IIIp = (L_III
                     - (b1[:, A, Bi] + b1[:, Bi, A]) - (b3[:, A, Bi] + b3[:, Bi, A])
                     + 0.5 * (b2[:, A, Bi] + b2[:, Bi, A]) + 0.5 * (b4[:, A, Bi] + b4[:, Bi, A]))
        IIp_IIIp = 0.5 * c.s3 * (b2[:, Bi, A] - b2[:, A, Bi]) + 0.5 * c.s3 * (b4[:, Bi, A] - b4[:, A, Bi])
        return (c_Ip_Ip + c_Ip, c_IIp + IIp_IIp + IIp_IIIp, c_IIIp + IIIp_IIp + IIIp_IIIp)
    return apply


def _fold_chunk(P, memory_gb):
    """Occupied indices per streamed chunk inside the fold: occ_chunk, capped so that
    the about ten chunk arrays of O V^2 per occupied index alive at once (amplitude and
    integral chunks) take at most half the fold's memory."""
    per_index = 10 * P.O * P.V * P.V * 8
    return max(1, min(occ_chunk(P.O), int(0.5 * memory_gb * 1e9 // per_index)))


def _batch_size(P, folded, memory_gb):
    """Columns per pass from the memory budget: about twenty arrays of the
    folded sector's size live per column (unfolded blocks and accumulators)."""
    per_col = (P.O * P.V * P.V if folded == '2p1h' else P.O * P.O * P.V) * 8 * 20
    return max(1, int(memory_gb * 1e9 // per_col))


def static_block(P, sector, batch=None, memory_gb=40.0, comm=None, share=False):
    """M of the kept orbitals (occupied for IP, virtual for EA): F plus the
    symmetrized third-order fold of the other satellite sector. batch: columns
    per pass over the streamed chunks (None: from memory_gb; 1: the
    single-vector appliers of the Dyson operator, one column at a time). comm: the
    columns are split over its ranks (default: the enclosing `distributed` one); each
    rank folds its own columns serially, so the chunk split of the Dyson appliers is
    switched off inside. share: hold the pair-major factor of the 2p1h ladders once per
    node (an MPI-3 shared window) instead of once per rank: less memory, but most ranks
    then read it from another NUMA domain."""
    comm = comm if comm is not None else current_comm()
    shared = None
    if sector == 'IP':
        # the pair-major vir-vir factor of the 2p1h ladders, built once per fold (every
        # rank's fold reads it whole; with share, one copy per node)
        naux, V = P.ctx.Bv_pp.shape[0], P.V
        shared = NodeShared((V * V, naux), comm if share else None)
        if shared.fills:
            shared.array[...] = _pp_pair_factor(P.ctx.Bv_pp)
        shared.ready()
    with distributed(None):
        return _static_block(P, sector, batch, memory_gb, comm,
                             None if shared is None else shared.array)


def _static_block(P, sector, batch, memory_gb, comm, pp_pair=None):
    if P.ctx.screened or P.ctx.dressed:
        raise ValueError("the fold is built for the bare blocks: no screening, no EN shifts")
    folded = '2p1h' if sector == 'IP' else '2h1p'
    rows = np.arange(P.O) if sector == 'IP' else np.arange(P.O, P.norb)
    eps = P.eps                     # the fold reads bare eps (F carries Sigma(inf))
    K, C = _sector_layout(P, folded)
    adj1, fwd1 = _sector_maps(P, P.u_first_order(), folded)
    G = np.empty((rows.size, rows.size))
    if batch == 1:
        adj, fwd = _sector_maps(P, P.U, folded)
        e = np.zeros(P.norb)
        for jj, j in enumerate(rows):
            e[j] = 1.0
            u, u1 = adj(e), adj1(e)
            e[j] = 0.0
            R = tuple(1.0 / (eps[j] - k) for k in K)
            a = tuple(r * x for r, x in zip(R, u1))
            b = tuple(r * (x - x1) for r, x, x1 in zip(R, u, u1))
            c = tuple(r * x for r, x in zip(R, C(*a)))
            col = fwd1(tuple(x + y for x, y in zip(b, c))) + fwd(a)
            G[:, jj] = col[rows]
        return P.F[np.ix_(rows, rows)] + 0.5 * (G + G.T), rows
    nb = batch or _batch_size(P, folded, memory_gb)
    chunk = _fold_chunk(P, memory_gb)
    adjB, fwdB = (_u_2p1h_batch if folded == '2p1h' else _u_2h1p_batch)(P, chunk)
    CB = _c_2p1h_batch(P, pp_pair, chunk) if folded == '2p1h' else _c_2h1p_batch(P, chunk)
    # Columns are split over the ranks in contiguous blocks; every rank folds its own
    # and the columns of G travel verbatim, so G is the same on every rank.
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    c_lo, c_hi = contiguous_block(rows.size, rank, size)
    starts = list(range(c_lo, c_hi, nb))
    sizes = [k.size for k in K]
    n_sat = sum(sizes)
    # U applied to the vectors a is the contraction of a with the rows of U, and the
    # rows of the kept orbitals are the adjoint images u of the unit vectors computed
    # below anyway: with room for all of them, the streamed forward coupling becomes
    # one matrix product.
    keep_rows = rows.size * n_sat * 8 <= 0.4 * memory_gb * 1e9
    u_all = None
    if keep_rows:
        u_all = np.zeros((rows.size, n_sat))
        for c0 in starts:
            nw = min(nb, c_hi - c0)
            u_all[c0:c0 + nw] = np.concatenate(
                [x.reshape(-1, nw) for x in adjB(_units(P.norb, rows[c0:c0 + nw]))], axis=0).T
        allgather_blocks(u_all, comm)
    G = np.zeros((rows.size, rows.size))
    for c0 in starts:
        cols = rows[c0:c0 + nb] if c0 + nb <= c_hi else rows[c0:c_hi]
        nw = cols.size
        if keep_rows:
            u = _split_blocks(u_all[c0:c0 + nw].T, K)
        else:
            u = adjB(_units(P.norb, cols))
        u1 = tuple(np.stack([adj1(_unit(P.norb, j))[k] for j in cols], axis=-1) for k in range(3))
        R = tuple(1.0 / (eps[cols] - k[..., None]) for k in K)
        a = tuple(r * x for r, x in zip(R, u1))
        b = tuple(r * (x - x1) for r, x, x1 in zip(R, u, u1))
        cC = tuple(r * x for r, x in zip(R, CB(*a)))
        bc = tuple(x + y for x, y in zip(b, cC))
        col1 = np.stack([fwd1(tuple(z[..., w] for z in bc)) for w in range(nw)], axis=-1)
        if keep_rows:
            a_flat = np.concatenate([x.reshape(-1, nw) for x in a], axis=0)
            G[:, c0:c0 + nw] = col1[rows] + u_all @ a_flat
        else:
            G[:, c0:c0 + nw] = (fwdB(a) + col1)[rows]
    if size > 1:
        GT = np.ascontiguousarray(G.T)
        allgather_blocks(GT, comm)
        G = GT.T
    return P.F[np.ix_(rows, rows)] + 0.5 * (G + G.T), rows


def _units(n, cols):
    """(n, len(cols)) with a unit vector in each column."""
    E = np.zeros((n, len(cols)))
    E[cols, np.arange(len(cols))] = 1.0
    return E


def _split_blocks(mat, K):
    """(n_sat, nw) -> the blocks shaped like the K diagonals with a trailing column axis."""
    out, off = [], 0
    for k in K:
        out.append(np.ascontiguousarray(mat[off:off + k.size]).reshape(k.shape + (mat.shape[1],)))
        off += k.size
    return tuple(out)


def _unit(n, j):
    e = np.zeros(n)
    e[j] = 1.0
    return e


def build_operator(s, nocc, sector, static_correction=None, ip_block='precomputed',
                   share_df=False, fold_memory_gb=40.0):
    """(aop, diag, dims) of non-Dyson ADC(3) in one sector on a restricted DF
    solver s (ADCSolver(mf, level='adc3', df=True), bare). The operator is in
    excitation convention: eigenvalues E(N-1) - E0 = IP for 'IP' and
    E(N+1) - E0 = -EA for 'EA', so the first IP or EA is the lowest root.
    dims: n_orb (kept orbitals), n_sat (satellite configurations), M (the
    static block, Dyson sign convention; None on the fly).
    ip_block: 'precomputed' (the symmetrized fold, built once) or 'on_the_fly' (IP only:
    the 1h-1h block of Dempwolff et al. applied to every vector, adc_r_nondyson_otf).
    share_df: the fold's pair factor held once per node (static_block(share=True)).
    fold_memory_gb: the fold's working memory per rank (static_block(memory_gb=)); with
    several ranks per node, at most the node memory over the ranks minus the streamed
    chunk arrays."""
    if sector not in SECTORS:
        raise ValueError(f"sector must be 'IP' or 'EA', got {sector!r}")
    if getattr(s, 'B_aa', None) is None:
        raise ValueError("non-Dyson DF ADC(3) needs a DF solver (df=True, B_aa)")
    if (s.level != 'adc3' or s.pdaggerq or s.en_dress is not None or s.u2_denom_dress
            or s.amplitudes is not None or s.W_aux is not None or s.W_chemist is not None):
        raise ValueError("non-Dyson DF ADC(3) is built for the bare hand-written ADC(3) "
                         "blocks: level='adc3', pdaggerq=False, no en_dress, no amplitude "
                         "substitution, no screening")
    if ip_block not in ('precomputed', 'on_the_fly'):
        raise ValueError(f"ip_block must be 'precomputed' or 'on_the_fly', got {ip_block!r}")
    if ip_block == 'on_the_fly' and sector != 'IP':
        raise ValueError("the on-the-fly 1h-1h block exists for the IP sector only")
    P = build_pieces(s, nocc, static_correction)
    if ip_block == 'on_the_fly':
        sc = np.zeros((P.norb, P.norb)) if static_correction is None else static_correction
        S = otf.static_part(s.B_aa, P.eps, P.O, sc[:P.O, :P.O])
        rows, M = np.arange(P.O), None

        def M_apply(z):     # Dyson sign, as the precomputed M
            return -(S @ z + otf.third_order_apply(s.B_aa, P.eps, P.O, z))
        M_diag = -np.diag(S)
    else:
        M, rows = static_block(P, sector, memory_gb=fold_memory_gb, share=share_df)
        M_apply, M_diag = M.__matmul__, np.diag(M)
    kept = '2h1p' if sector == 'IP' else '2p1h'
    K, C = _sector_layout(P, kept)
    adj, fwd = _sector_maps(P, P.U, kept)
    shapes = [k.shape for k in K]
    sizes = [k.size for k in K]
    n_r = rows.size
    sign = -1.0 if sector == 'IP' else 1.0
    norb = P.norb

    def aop(z):
        z_r = z[:n_r]
        parts, off = [], n_r
        for shp, n in zip(shapes, sizes):
            parts.append(z[off:off + n].reshape(shp))
            off += n
        zp = np.zeros(norb)
        zp[rows] = z_r
        y_r = M_apply(z_r) + fwd(parts)[rows]
        u, c = adj(zp), C(*parts)
        y_sat = [(ui + k * x + ci).reshape(-1) for ui, k, x, ci in zip(u, K, parts, c)]
        return sign * np.concatenate([y_r] + y_sat)

    diag = sign * np.concatenate([M_diag] + [k.reshape(-1) for k in K])
    return aop, diag, dict(n_orb=n_r, n_sat=int(sum(sizes)), M=M)
