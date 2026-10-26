"""The orbital-relaxed difference density of a folded ADC(2) root: the
derivative of its excitation energy with respect to a one-electron
perturbation, Delta rho_rel = d omega / d h.

ee_ri_density gives the unrelaxed ISR-ADC(2) difference density, an
expectation value. The density a field actually sees is the derivative of
omega, which adds the response of the reference orbitals (the Z-vector of
Handy and Schaefer, J. Chem. Phys. 81, 5031 (1984)) and the response of the
first-order amplitudes. The construction is the ADC(2) Lagrangian of
Haettig (Adv. Quantum Chem. 50, 37 (2005); Koehn and Haettig, J. Chem. Phys.
119, 5021 (2003)), written here against the folded operator of ee_ri_fold.

THE FUNCTIONAL. For a root (y unit singles of one spin channel, omega, t1
its singles weight) the folded eigenvalue satisfies omega = y^T A_eff(omega)
y with A_eff(w) = M - C (D - w)^-1 C^T; with y and omega held,

    d omega = t1 d Phi,     Phi = y^T A_eff(omega) y

(Hellmann-Feynman for the fold: d omega / d omega of A_eff is -|Y|^2, and
t1 = 1 / (1 + |Y|^2)). Phi is a function of the orbital energies and the
density-fitted factors B^P_pq of the operator,

    Phi = Phi_0 + Phi_1 + Phi_LR + Phi_2 + Phi_D

    Phi_0   sum_ia y_ia^2 (e_a - e_i)
    Phi_1   singlet 2 (ia|jb) y y - (ij|ab) y y;  triplet -(ij|ab) y y
    Phi_LR  2 q^T Q q, q = V^T y (the optical reaction field, singlets)
    Phi_2   y^T (xo y + y xv) - (K y) . (T y)          (ee_ri_fold, M2)
    Phi_D   y^T C (omega - D)^-1 C^T y = sum R H        (the doubles)

THE DERIVATIVES. With gamma = dPhi/df (the Fock matrix in a non-canonical
basis, active orbitals) and Gamma^P_pq = dPhi/dB^P_pq, a rotation C -> C
(1 + kappa) of the orbitals moves Phi by

    dPhi/dkappa_rp = W_rp - W_pr,
    W_rp = sum_q Gs^P_pq B^P_rq + (the reaction field's own term),

Gs^P_pq = Gamma^P_pq + Gamma^P_qp, plus the change of the Fock matrix through
the occupied orbitals, sum_pq gamma_pq [4 (pq|ai) - (pa|qi) - (pi|qa)].
Stationarity of L = Phi + sum z_ai f_ai (all occupied i, frozen core
included) is the CPHF equation (solved by src.Base.pyscf.cphf)

    (e_a - e_i) z_ai + sum_bj [4 (ai|bj) - (ab|ij) - (aj|ib)] z_bj = -X_ai,

and a frozen core adds the core/active-occupied rotations, z'_ci =
X_ci / (e_i - e_c), whose f_ci enters gamma. The relaxed density is

    Delta rho_rel = t1 [gamma + sym(z') + sym(z)],

spin-summed, in the full MO basis (frozen core included). Its h-contraction
is d omega / d h for any symmetric one-electron h.

THE CONTINUUM. The ground state is the SCF relaxed in PCM(eps_static); in the
PTE convention its surface charges q0 are FROZEN when the orbitals respond,
so the CPHF kernel carries no reaction field (`pcm_response=False`, the
default): the solvent's whole linear answer to the excitation is then the
state-specific energy (1/2) dV Q dV of ee_state_specific, and letting the
continuum respond inside the CPHF as well would count the mutual
polarization about twice. pcm_response=True adds the equilibrium (eps_static)
response to the kernel, as a bound on that choice.

This module holds the dense reference (`dense_pieces`, small molecules, every
tensor whole) and the orbital response shared with the tiled route
(`relaxed_density`).
"""
import numpy as np
from pyscf import lib, scf
from pyscf import df as pyscf_df

from src.Base.numerics.constants import (CORRELATION_CPHF_TOL, CPHF_MAX_CYCLE,
                                         DF_FORCE_PIECES_TILE_ARRAYS, EE_RI_TILE_GB)
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf import cphf
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, ordered_chain_sum,
                                              reduce_sum)
from src.SingleReference.common.df_amplitudes import (PAIR, amplitude_fock,
                                                      doubles_denominator,
                                                      tile_amplitude_fock)
from src.SingleReference.ADC.eeADC.ee_ri_density import (Pass, oo_contract, tile_ranges,
                                                     vv_contract)
# symmetrize is also read from here (df_correlation_force_tiled: erd.symmetrize)
from src.SingleReference.ADC.eeADC.ee_utils import symmetrize

SWAP = (2, 3, 0, 1)        # X^T_kcld = X_ldkc


class Pieces:
    """dPhi/d(inputs) of one root, active orbitals, Phi-level (not times t1).

    g_oo (no, no), g_vv (nv, nv): dPhi/df, symmetric
    GL   (no nv, naux): dPhi/dB^P_ia, rows (i, a)
    Goo  (no, no, naux): dPhi/dB^P_ki as stored, [k, i, P]
    Xvv  (no nv, naux): dPhi/dB^P_ab = sum_k y_ka Xvv[(k, b), P]
    y    (no, nv): the unit singles that factor Gamma_vv
    lr   (no, nv) or None: 4 y, the transition density's weight against the
         reaction-field potential U (singlets in a continuum)
    q    (ng,) or None: Q V^T y, the surface charges U is built from
    """

    def __init__(self, g_oo, g_vv, GL, Goo, Xvv, y, lr=None, q=None):
        self.g_oo, self.g_vv = g_oo, g_vv
        self.GL, self.Goo, self.Xvv, self.y = GL, Goo, Xvv, y
        self.lr, self.q = lr, q


# ----------------------------------------------------------------------
# the dense reference
# ----------------------------------------------------------------------

def dense_inputs(op):
    """(eo, ev, L (no, nv, naux), Boo (no, no, naux), Bvv (nv, nv, naux)) of a
    serial RIFoldADC2, every factor whole."""
    f = op.f
    if f.Bvv is not None:
        Bvv = f.Bvv.transpose(0, 2, 1)
    else:
        if op.comm is not None and op.comm.Get_size() > 1:
            raise ValueError('the dense reference is serial')
        Bvv = np.concatenate([f.bvv(c, *op.chunks[c]) for c in
                              range(len(op.chunks))], axis=1).transpose(0, 2, 1)
    L = f.LT.transpose(1, 2, 0)
    return (np.array(f.eo), np.array(f.ev), np.ascontiguousarray(L),
            np.ascontiguousarray(f.Boo), np.ascontiguousarray(Bvv))


def dense_phi(inp, y, omega, spin, level='adc2', rf=None):
    """Phi = y^T A_eff(omega) y from whole tensors (module docstring); rf a
    TransitionReactionField or None."""
    eo, ev, L, Boo, Bvv = inp
    no, nv = eo.size, ev.size
    y = np.asarray(y, float).reshape(no, nv)
    singlet = spin == 'singlet'
    phi = float(np.sum(y * y * (ev[None, :] - eo[:, None])))
    c = np.einsum('iaP,ia->P', L, y)
    ex = np.einsum('ia,ijP,abP,jb->', y, Boo, Bvv, y)
    phi += (2.0 * c @ c - ex) if singlet else -ex
    if rf is not None and singlet:
        q = rf.V.T @ y.ravel()
        phi += 2.0 * q @ (rf.Q @ q)
    if level == 'adc1':
        return phi
    I = np.einsum('iaP,jbP->iajb', L, L)
    D = doubles_denominator(eo, ev)
    tau = (2.0 * I - I.transpose(PAIR)) / D
    x_oo = 0.5 * np.einsum('kdic,kdjc->ij', I, tau)
    x_vv = 0.5 * np.einsum('kalc,kblc->ab', I, tau)
    xo, xv = x_oo + x_oo.T, x_vv + x_vv.T
    if singlet:
        K, T = 2.0 * I - I.transpose(PAIR), tau
    else:
        K, T = -I.transpose(PAIR), -I.transpose(PAIR) / D
    Ky = np.einsum('iakc,kc->ia', K, y)
    Ty = np.einsum('iakc,kc->ia', T, y)
    phi += float(np.sum(y * (xo @ y + y @ xv)) - np.sum(Ky * Ty))
    R, H, _ = _dense_doubles(L, Boo, Bvv, eo, ev, y, omega, singlet)
    return phi + float(np.sum(R * H))


def _dense_doubles(L, Boo, Bvv, eo, ev, y, omega, singlet):
    """(R, H, w): H_kcld = sum_P Bt_kcP L_ldP, R the code's doubles weight
    (ee_ri_fold._doubles_tile), w = 1/(omega - D)."""
    Bt = np.einsum('kiP,ic->kcP', Boo, y) - np.einsum('ka,acP->kcP', y, Bvv)
    H = np.einsum('kcP,ldP->kcld', Bt, L)
    w = 1.0 / (omega - doubles_denominator(eo, ev))
    if singlet:
        G = H + H.transpose(SWAP)
        R = w * (2.0 * G - G.transpose(PAIR))
    else:
        S = H + H.transpose(SWAP)
        R = w * (2.0 * H - S.transpose(PAIR))
    return R, H, w


def _dense_dd(H, w, singlet):
    """The doubles' Fock derivative (oo, vv), spin-summed, Phi-level: the
    oo_dd and vv_dd terms of ee_ri_density at unit singles weight."""
    scale = 1.0 / (2.0 * np.sqrt(2.0))
    if singlet:
        Z = scale * w * (H + H.transpose(SWAP))
        W = Z - Z.transpose(PAIR)
    else:
        Z = scale * w * (H - H.transpose(SWAP))
        S = H + H.transpose(SWAP)
        W = scale * w * (S - S.transpose(PAIR))
    oo = -2.0 * (np.einsum('kbia,kbja->ij', W, W)
                 + 2.0 * np.einsum('kbia,kbja->ij', Z, Z))
    vv = 2.0 * (np.einsum('iajc,ibjc->ab', W, W)
                + 2.0 * np.einsum('iajc,ibjc->ab', Z, Z))
    return 2.0 * oo, 2.0 * vv


def dense_pieces(inp, y, omega, spin, level='adc2', rf=None):
    """Pieces of dPhi (module docstring) from whole tensors."""
    eo, ev, L, Boo, Bvv = inp
    no, nv, naux = eo.size, ev.size, L.shape[2]
    y = np.asarray(y, float).reshape(no, nv)
    singlet = spin == 'singlet'
    g_oo, g_vv = -y @ y.T, y.T @ y
    # Phi_1
    c = np.einsum('iaP,ia->P', L, y)
    GL = (4.0 * np.einsum('P,ia->iaP', c, y) if singlet
          else np.zeros((no, nv, naux)))
    Goo = -np.einsum('ia,abP,jb->ijP', y, Bvv, y)
    Xvv = -np.einsum('kiP,ib->kbP', Boo, y)           # Gamma_ab = sum_k y_ka X_kb
    lr = q = None
    if rf is not None and singlet:
        q = rf.Q @ (rf.V.T @ y.ravel())
        lr = 4.0 * y
    if level == 'adc2':
        I = np.einsum('iaP,jbP->iajb', L, L)
        D = doubles_denominator(eo, ev)
        tau = (2.0 * I - I.transpose(PAIR)) / D
        ro, rv = y @ y.T, y.T @ y
        if singlet:
            K, T = 2.0 * I - I.transpose(PAIR), tau
        else:
            K, T = -I.transpose(PAIR), -I.transpose(PAIR) / D
        Ky = np.einsum('iakc,kc->ia', K, y)
        Ty = np.einsum('iakc,kc->ia', T, y)
        # explicit dPhi_2/dI (tau, T, K held)
        A = (np.einsum('kdjc,ij->kdic', tau, ro)
             + np.einsum('kblc,ab->kalc', tau, rv))
        LK = -np.einsum('ia,kc->iakc', Ty, y)
        A += (2.0 * LK - LK.transpose(PAIR)) if singlet else -LK.transpose(PAIR)
        Ltau = (np.einsum('kdic,ij->kdjc', I, ro)
                + np.einsum('kalc,ab->kblc', I, rv))
        LT = -np.einsum('ia,kc->iakc', Ky, y)
        if singlet:
            Ltau = Ltau + LT
        lam = 0.5 * (Ltau + Ltau.transpose(SWAP)) / D
        A += 2.0 * lam - lam.transpose(PAIR)
        o, v = amplitude_fock(lam, tau)
        g_oo, g_vv = g_oo + o, g_vv + v
        if not singlet:
            lt = 0.5 * (LT + LT.transpose(SWAP)) / D
            A -= lt.transpose(PAIR)
            o, v = amplitude_fock(lt, T)
            g_oo, g_vv = g_oo + o, g_vv + v
        GL += np.einsum('kcld,ldP->kcP', A + A.transpose(SWAP), L)
        # Phi_D
        R, H, w = _dense_doubles(L, Boo, Bvv, eo, ev, y, omega, singlet)
        Bt = np.einsum('kiP,ic->kcP', Boo, y) - np.einsum('ka,acP->kcP', y, Bvv)
        Gam = np.einsum('kcld,ldP->kcP', R, L)
        GL += 2.0 * np.einsum('kcld,kcP->ldP', R, Bt)
        Goo += 2.0 * np.einsum('kcP,ic->kiP', Gam, y)
        Xvv -= 2.0 * Gam
        o, v = _dense_dd(H, w, singlet)
        g_oo, g_vv = g_oo + o, g_vv + v
    sym = lambda X: 0.5 * (X + X.T)
    return Pieces(sym(g_oo), sym(g_vv), GL.reshape(no * nv, naux), Goo,
                  Xvv.reshape(no * nv, naux), y, lr, q)


# ----------------------------------------------------------------------
# the tiled route
# ----------------------------------------------------------------------

def tiled_pieces(op, y, omega, tile_gb=EE_RI_TILE_GB):
    """Pieces of dPhi on the density-fitted tiles of `op` (ee_ri_fold.
    RIFoldADC2): the dense_pieces arithmetic with every doubles-shaped tensor
    one occupied tile at a time. A collective over op's ranks; the result is
    the same bits on every rank and at every rank count.

    Everything a tile contributes to rows it holds is exact under any sum;
    what it contributes to every row (the second factor of (kc|ld) and of the
    doubles' H) is summed in tile order (ordered_chain_sum). The two
    contractions that would need another tile's first index are moved onto
    this tile's: the amplitude adjoint's swap partner through the one-index
    transformed factor (rho_o L), and the occupied Fock block through the
    swap symmetry of the adjoint and the amplitudes."""
    f, no, nv, naux = op.f, op.no, op.nv, op.f.naux
    singlet = op.spin == 'singlet'
    y = np.asarray(y, float).reshape(no, nv)
    comm = op.comm
    ranks = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    L = f.L                                                   # (no nv, naux)
    g_oo, g_vv = -y @ y.T, y.T @ y
    c = L.T @ y.ravel()
    GL = (4.0 * np.outer(y.ravel(), c) if singlet else np.zeros((no * nv, naux)))
    # Gamma_oo of Phi_1, -(y B^P_vv y^T), from the auxiliary chunks' owners
    GooT = np.zeros((naux, no, no))
    for ch in op.my_chunks:
        p0, p1 = op.chunks[ch]
        Bc = f.bvv(ch, p0, p1)                                # [a, P, b]
        GooT[p0:p1] = -np.einsum('ia,aPb,jb->Pij', y, Bc, y, optimize=True)
    GooT = allgather_ranges(GooT, op.chunk_ranges, comm)
    Goo = np.ascontiguousarray(GooT.transpose(1, 2, 0))
    Xvv = -np.einsum('Pki,ib->kbP', f.BooP, y, optimize=True).reshape(no * nv, naux)
    lr = q = None
    if op.rf is not None and singlet:
        q = op.rf.Q @ (op.rf.V.T @ y.ravel())
        lr = 4.0 * y
    if op.level == 'adc1':
        return Pieces(symmetrize(g_oo), symmetrize(g_vv), GL, Goo, Xvv, y, lr, q)
    ro, rv = y @ y.T, y.T @ y
    Ky = op.k_apply(y.reshape(-1, 1))[:, 0].reshape(no, nv)
    roL = (ro @ L.reshape(no, -1)).reshape(no * nv, naux)     # sum_i ro_ki L_id
    tiles = tile_ranges(no, 8.0 * nv * no * nv * DF_FORCE_PIECES_TILE_ARRAYS, tile_gb, ranks)
    mine = [(t, kk) for t, kk in enumerate(tiles) if t % ranks == rank]
    # T y: rows of the tiles, exact under any sum
    Ty = np.zeros((no, nv))
    for t, (k0, k1) in mine:
        I = _tile_ints(op, k0, k1)
        D = doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
        T = (2.0 * I - I.transpose(PAIR)) / D if singlet else -I.transpose(PAIR) / D
        Ty[k0:k1] = (T.reshape((k1 - k0) * nv, -1) @ y.ravel()).reshape(k1 - k0, nv)
    reduce_sum(Ty, comm)
    # the doubles' dressed factor, every rank
    BtT = op.dressed(y.ravel())
    BtR = np.ascontiguousarray(BtT.T)                         # (no nv, naux)
    # sums over the tiles, in tile order; rows of one tile each, any order
    fock = Pass(op, {'g_oo': (no, no), 'g_vv': (nv, nv)})
    rows = {'Goo': np.zeros((no, no, naux)), 'Xvv': np.zeros((no * nv, naux))}
    # the GL terms that reach every row are summed in tile order a round of
    # ranks at a time, so a rank holds one tile's term, not all of its own
    running = np.zeros((no * nv, naux))
    for m in range(-(-len(tiles) // ranks)):
        ids = list(range(m * ranks, min((m + 1) * ranks, len(tiles))))
        terms = {}
        for t in ids:
            if t % ranks != rank:
                continue
            k0, k1 = tiles[t]
            terms[t - ids[0]], blk = _tile_terms(op, k0, k1, y, omega, ro, rv, Ty, Ky,
                                                 roL, BtT, BtR, rows)
            fock.add(t, blk)
        running = ordered_chain_sum(terms, [t % ranks for t in ids], comm, running)
        del terms
    del BtT, BtR
    GL += running
    del running
    tot = fock.total()
    # every element of these from one tile, the rest zeros: exact in any order
    for a in rows.values():
        reduce_sum(a, comm)
    return Pieces(symmetrize(g_oo + tot['g_oo']), symmetrize(g_vv + tot['g_vv']),
                  GL, Goo + rows['Goo'], Xvv + rows['Xvv'], y, lr, q)


def _tile_terms(op, k0, k1, y, omega, ro, rv, Ty, Ky, roL, BtT, BtR, rows):
    """(term, blk) of one occupied tile of tiled_pieces: its GL term, reaching
    every row, and its share of the Fock blocks; its own rows of Goo and Xvv are
    added to rows."""
    f, no, nv, naux = op.f, op.no, op.nv, op.f.naux
    singlet = op.spin == 'singlet'
    L = f.L
    scale = 1.0 / (2.0 * np.sqrt(2.0))
    blk = {'g_oo': np.zeros((no, no)), 'g_vv': np.zeros((nv, nv))}
    kt = k1 - k0
    r0, r1 = k0 * nv, k1 * nv
    I = _tile_ints(op, k0, k1)
    D = doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
    tau = (2.0 * I - I.transpose(PAIR)) / D
    # explicit dPhi_2/dI and the amplitude adjoints (dense_pieces)
    A = (np.einsum('kdjc,ij->kdic', tau, ro, optimize=True)
         + np.einsum('kblc,ab->kalc', tau, rv, optimize=True))
    LK = -Ty[k0:k1, :, None, None] * y[None, None]
    A += (2.0 * LK - LK.transpose(PAIR)) if singlet else -LK.transpose(PAIR)
    del LK
    Lt = (np.einsum('kdic,ij->kdjc', I, ro, optimize=True)
          + np.einsum('kalc,ab->kblc', I, rv, optimize=True)
          + (roL[r0:r1] @ L.T).reshape(kt, nv, no, nv)
          + np.einsum('kbla,ca->kblc', I, rv, optimize=True))
    LT = (-Ky[k0:k1, :, None, None] * y[None, None]
          - y[k0:k1, :, None, None] * Ky[None, None])
    if singlet:
        Lt += LT
    lam = 0.5 * Lt / D
    del Lt
    A += 2.0 * lam - lam.transpose(PAIR)
    o, v = tile_amplitude_fock(lam, tau)
    blk['g_oo'] += o
    blk['g_vv'] += v
    del lam
    if not singlet:
        lt = 0.5 * LT / D
        A -= lt.transpose(PAIR)
        o, v = tile_amplitude_fock(lt, -I.transpose(PAIR) / D)
        blk['g_oo'] += o
        blk['g_vv'] += v
        del lt
    del LT, I, tau
    A2 = A.reshape(kt * nv, -1)
    term = np.zeros((no * nv, naux))
    term[r0:r1] = A2 @ L
    term += A2.T @ L[r0:r1]
    del A, A2
    # the doubles (ee_ri_fold._doubles_tile's R)
    H, HT = op.pair(BtT, k0, k1)
    w = 1.0 / (omega - D)
    if singlet:
        G = H + HT
        R = w * (2.0 * G - G.transpose(PAIR))
        del G
    else:
        S = H + HT
        R = w * (2.0 * H - S.transpose(PAIR))
        del S
    R2 = R.reshape(kt * nv, -1)
    Gam = R2 @ L                                          # rows of the tile
    term += 2.0 * (R2.T @ BtR[r0:r1])
    del R, R2
    Gam3 = Gam.reshape(kt, nv, naux)
    rows['Goo'][k0:k1] += 2.0 * np.einsum('kcP,ic->kiP', Gam3, y, optimize=True)
    rows['Xvv'][r0:r1] -= 2.0 * Gam
    del Gam, Gam3
    # the doubles' Fock block, ee_ri_density's oo_dd and vv_dd
    if singlet:
        Z = scale * w * (H + HT)
        W = Z - Z.transpose(PAIR)
    else:
        Z = scale * w * (H - HT)
        S = H + HT
        W = scale * w * (S - S.transpose(PAIR))
        del S
    del H, HT, w, D
    blk['g_oo'] += -4.0 * (oo_contract(W, W) + 2.0 * oo_contract(Z, Z))
    blk['g_vv'] += 4.0 * (vv_contract(W, W) + 2.0 * vv_contract(Z, Z))
    del Z, W
    return term, blk


def _tile_ints(op, k0, k1):
    """(kc|ld) of the tile's rows, (kt, nv, no, nv)."""
    return op.ints(k0, k1)


# ----------------------------------------------------------------------
# the orbital response
# ----------------------------------------------------------------------

def core_factors(mol, mf, frozen, auxbasis=None):
    """B^P_cq (naux, frozen, nmo - frozen): the frozen core against every
    active orbital, in the operator's fit (ee_ri_fold.bare_factors)."""
    if auxbasis is None:
        with_df = mf.with_df
    else:
        with_df = pyscf_df.DF(mol, auxbasis=auxbasis)
        with_df.max_memory = mf.max_memory
        with_df.build()
    C = np.asarray(mf.mo_coeff)
    Cc, Ca = C[:, :frozen], C[:, frozen:]
    out = []
    for blk in with_df.loop():
        ao = lib.unpack_tril(blk)
        out.append(np.einsum('Pmn,mc,nq->Pcq', ao, Cc, Ca, optimize=True))
    return np.concatenate(out, axis=0)


def orbital_gradient(pc, op, Bc, back_vv):
    """X (nv, nocc_all) of the vo rotations and Xco (frozen, no) of the core/
    active-occupied ones, from the Gamma pieces alone (no Fock term):
    back_vv(G) = sum_bP G^P_ib B^P_ab (no, nv) for G (no nv, naux)."""
    f = op.f
    no, nv, naux = op.no, op.nv, f.naux
    nf = 0 if Bc is None else Bc.shape[1]
    L3 = f.LT.transpose(1, 2, 0)                              # [i, a, P]
    GL3 = pc.GL.reshape(no, nv, naux)
    Gs = pc.Goo + pc.Goo.transpose(1, 0, 2)                   # [i, j, P]
    X3 = pc.Xvv.reshape(no, nv, naux)
    y = pc.y
    # W[a, i] = sum_j Gs_ij L_ja + sum_b GL_ib B_ab
    W_ai = np.einsum('ijP,jaP->ai', Gs, L3) + back_vv(pc.GL).T
    # W[i, a] = sum_j GL_ja B_ij + sum_b Gs_ab L_ib,
    # sum_b Gs_ab B_rb = sum_k y_ka (X_k . B_r) + sum_k X_ka (B_r . y_k)
    W_ia = (np.einsum('jaP,ijP->ia', GL3, f.Boo)
            + np.einsum('ka,kbP,ibP->ia', y, X3, L3, optimize=True)
            + np.einsum('kaP,ibP,kb->ia', X3, L3, y, optimize=True))
    X = np.zeros((nv, nf + no))
    X[:, nf:] = W_ai - W_ia.T
    Xco = np.zeros((nf, no))
    if nf:
        Bco = Bc[:, :, :no].transpose(1, 2, 0)                # [c, j, P]
        Bcv = Bc[:, :, no:].transpose(1, 2, 0)                # [c, b, P]
        # W[c, i] = sum_j Gs_ij B_cj + sum_b GL_ib B_cb ;  W[i, c] = 0
        Xco = (np.einsum('ijP,cjP->ci', Gs, Bco)
               + np.einsum('ibP,cbP->ci', GL3, Bcv))
        # W[c, a] = sum_j GL_ja B_cj + sum_b Gs_ab B_cb ;  W[a, c] = 0
        W_ca = (np.einsum('jaP,cjP->ca', GL3, Bco)
                + np.einsum('ka,kbP,cbP->ca', y, X3, Bcv, optimize=True)
                + np.einsum('kaP,cbP,kb->ca', X3, Bcv, y, optimize=True))
        X[:, :nf] = -W_ca.T
    return X, Xco


def lr_potential(env, mol, C, q):
    """U = sum_k q_k v_k in the MO basis of C."""
    U = np.zeros((mol.nao, mol.nao))
    for k0, k1 in env.grid_blocks(mol.nao, live=1):
        v = env.ao_grid_potential(mol, grids=slice(k0, k1))
        U += np.einsum('mnk,k->mn', v, q[k0:k1], optimize=True)
    return C.T @ U @ C


def _gas_jk(mf):
    """get_jk of mf's interaction without any continuum: the JK the SCF
    Fock matrix is built from (density fitting included)."""
    m = scf.RHF(mf.mol)
    if getattr(mf, 'with_df', None) is not None:
        m = m.density_fit(auxbasis=mf.with_df.auxbasis)
        m.with_df = mf.with_df
    return m.get_jk


def relaxed_density(pc, op, mf, mol, frozen, t1, Bc=None, back_vv=None,
                    env=None, pcm_response=False, tol=CORRELATION_CPHF_TOL,
                    max_cycle=CPHF_MAX_CYCLE):
    """Delta rho_rel (nmo, nmo), the full MO basis of mf (frozen core
    included), spin-summed, times t1: d omega / d h_pq. Bc core_factors when
    frozen; back_vv as in orbital_gradient (default: the operator's own B_vv
    back-transformation, a collective over its ranks); env the continuum of
    the reaction field (its potentials for pc.lr); pcm_response adds mf's
    continuum (its eps, the static one of a solvated SCF) to the orbital
    response (module docstring)."""
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy, float)
    nmo = C.shape[1]
    nocc = mol.nelectron // 2
    nf = frozen
    if back_vv is None:
        back_vv = lambda G: operator_back_vv(op, G)
    X, Xco = orbital_gradient(pc, op, Bc, back_vv)
    if pc.lr is not None:
        U = lr_potential(env, mol, C, pc.q)
        o, v = slice(nf, nocc), slice(nocc, nmo)
        # W_LR[r, i] = sum_a lr_ia U_ra ; W_LR[r, a] = sum_i lr_ia U_ri
        X[:, nf:] += U[v, v] @ pc.lr.T - (U[o, o] @ pc.lr).T
        if nf:
            X[:, :nf] -= (U[:nf, o] @ pc.lr).T
            Xco = Xco + U[:nf, v] @ pc.lr.T
    # gamma in the full MO basis, the core/active rotations folded in
    G = np.zeros((nmo, nmo))
    G[nf:nocc, nf:nocc] = pc.g_oo
    G[nocc:, nocc:] = pc.g_vv
    if nf:
        zc = Xco / (eps[None, nf:nocc] - eps[:nf, None])
        G[:nf, nf:nocc] += 0.5 * zc
        G[nf:nocc, :nf] += 0.5 * zc.T
    get_jk = _gas_jk(mf)
    Co, Cv = C[:, :nocc], C[:, nocc:]
    P = C @ G @ C.T
    vj, vk = get_jk(mol, P, hermi=1)
    v = 4.0 * vj - 2.0 * vk
    pcm = None
    if pcm_response:
        pcm = getattr(mf, 'with_solvent', None)
        if pcm is None:
            raise ValueError('pcm_response needs a mean field solvated in PCM')
        v = v + 4.0 * pcm._B_dot_x(P)
    X = X + Cv.T @ v @ Co

    def fvind(z):
        z = z.reshape(nocc_v)
        dm = Cv @ z @ Co.T
        dm = dm + dm.T
        vj, vk = get_jk(mol, dm, hermi=1)
        v = 2.0 * vj - vk
        if pcm is not None:
            v = v + 2.0 * pcm._B_dot_x(dm)
        return (Cv.T @ v @ Co).ravel()

    nocc_v = (nmo - nocc, nocc)
    e_ai = eps[nocc:, None] - eps[None, :nocc]
    z = cphf.solve(fvind, e_ai, -X, numerics=Numerics(
        scf={'cphf_tol': tol, 'cphf_max_cycle': max_cycle})).reshape(nocc_v)
    G[nocc:, :nocc] += 0.5 * z
    G[:nocc, nocc:] += 0.5 * z.T
    return t1 * G


def operator_back_vv(op, G):
    """sum_bP G^P_ib B^P_ab (no, nv) for G (no nv, naux), on the operator's
    tiles and B_vv chunks (ee_ri_fold.RIFoldADC2.back_vv): a collective."""
    nv = op.nv
    gams = {t: G[k0 * nv:k1 * nv][None] for t, (k0, k1) in op.my_tiles}
    return -op.back_vv(gams, 1)[0].reshape(op.no, nv)
