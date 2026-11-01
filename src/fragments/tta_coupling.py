"""Triplet-triplet annihilation (triplet fusion) T1(A) T1(B) -> S1 S0: the
coupling of the singlet-coupled triplet pair 1(TT) to a site singlet, by the
charge-transfer-mediated model of the singlet-fission literature (Smith and
Michl, Chem. Rev. 110, 6891 (2010); Berkelbach, Hybertsen and Reichman,
J. Chem. Phys. 138, 114102 (2013)),

    V_eff(E) = V_direct + v_TT^T (E - A_cc)^-1 v_S ,

v_TT the couplings of 1(TT) to the charge-transfer diabats, v_S theirs to
the site singlet, A_cc their block of A_eff. 1(TT) is a double excitation,
which no singles operator holds as a diabat, so its couplings are built here
from the diabats of two fragment partitions on ONE set of local orbitals
(`src.fragments.partition`): the singlet partition (both site singlets and
both charge-transfer diabats explicit) and the triplet one (both site
triplets), for any method behind the `SinglesOperator` protocol.

THE STATES, on the closed-shell determinant |0> of the local occupied
orbitals:
- a singlet single S(x) = sum_ia x_ia E_ai |0> / sqrt 2 (x unit): the site
  singlets and the charge-transfer diabats, x their partition amplitudes;
- the site triplets' amplitudes t^A, t^B (unit, each on its fragment) and the
  pair coupled to total spin zero, 1(TT) = (T+ T- + T- T+ - T0 T0) / sqrt 3,
  in spin-free form

      1(TT) = (1 / (2 sqrt 3)) sum t^A_ia t^B_jb (E_ai E_bj + 2 E_aj E_bi) |0> ,

  unit for unit amplitudes on two fragments.

THE ELEMENTS:
- V(TT -> S) for any singlet single, V_direct (S the site singlet) and v_TT
  (S a charge-transfer diabat) alike: the Slater-Condon element of the
  electronic Hamiltonian,

      <S(x)|H|1(TT)> = sqrt(3/2) sum_{(u,w) = (t^A,t^B), (t^B,t^A)}
                       x . [ w f_vo u - K(w)^T u + w L(u) ] ,
      K(w)_ji = sum_kb (jb|ki) w_kb ,  L(u)_ca = sum_jb (jc|ba) u_jb ,

  f the Fock matrix of |0>'s density (h + 2J - K), the integrals density
  fitted. The Coulomb contractions of the two transition densities cancel in
  the singlet pairing of two triplets: only exchange-type terms remain, and
  the one-electron term f_vo vanishes for a Hartree-Fock reference
  (Brillouin). Gated against the full configuration interaction vectors of a
  small dimer, with a reference that is not Hartree-Fock as well
  (tests/test_tta_coupling.py).
- v_S, A_cc and the site energies: the singlet partition's A_eff(Omega_0),
  screened for GW-BSE, with the folded doubles for ADC(2): its site-charge
  transfer elements are the local orbitals' hole and electron transfer
  couplings (the Fock elements f_ij, f_ab between fragments) dressed by the
  kernel. The bare Slater-Condon couplings <S|H|S'> are in the record for
  comparison (`bare_singles`).
- E_TT = E(T1, A) + E(T1, B), the triplet partition's site diagonals;
  E = (E_S1 + E_TT) / 2 by default, the middle of the two diabats the
  coupling connects.

WHAT IS NOT HERE. The triplet channel 3(TT) -> T_n S0 and the quintet;
the doubly charge-transfer and the other doubly excited configurations;
a full-BSE partition (the configuration model reads excitation amplitudes:
the Tamm-Dancoff kernel or ADC(2)); more than two fragments.
"""
from dataclasses import dataclass

import numpy as np
from pyscf import df, lib

from src.fragments.transfer import require_below

#: Megabytes of three-index factors held at once when the fitted integrals
#: are streamed over the auxiliary index.
DF_BLOCK_MB = 2000


def local_factors(mol, c, auxbasis, max_memory_mb=DF_BLOCK_MB):
    """Yield blocks B^Q_pq (nQ, n, n) of the density-fitted integrals
    (pq|rs) = sum_Q B^Q_pq B^Q_rs in the orbitals `c` (nao, n), over the
    auxiliary index of `auxbasis`."""
    fit = df.DF(mol, auxbasis=auxbasis)
    fit.build()
    nao, n = c.shape
    blk = max(1, int(max_memory_mb * 1e6 / (8 * (nao * nao + n * nao
                                                  + n * n))))
    for packed in fit.loop(blksize=blk):
        b = lib.unpack_tril(packed)
        yield np.einsum('Qmn,mp,nq->Qpq', b, c, c, optimize=True)


def triplet_pair_elements(mol, hcore, c_occ, c_vir, auxbasis, t_a, t_b,
                          singles, max_memory_mb=DF_BLOCK_MB):
    """The configuration model's elements on the determinant of `c_occ`
    (module docstring). t_a, t_b (nocc, nvir): the two triplets' unit
    amplitudes on two fragments; singles (k, nocc, nvir): unit singlet
    amplitudes. Returns a dict:

    tt: (k,) <S_m|H|1(TT)>;
    ss: (k, k) the bare <S_m|H|S_n> less E_0 <S_m|S_n>;
    fock: (nmo, nmo) f of the reference density in [c_occ, c_vir].
    """
    c_occ, c_vir = np.asarray(c_occ, float), np.asarray(c_vir, float)
    no, nv = c_occ.shape[1], c_vir.shape[1]
    t_a, t_b = (np.asarray(t, float).reshape(no, nv) for t in (t_a, t_b))
    singles = np.asarray(singles, float).reshape(-1, no, nv)
    if np.any((t_a != 0) & (t_b != 0)):
        raise ValueError('the two triplets must sit on different fragments: '
                         'their amplitudes share pairs')
    c = np.hstack([c_occ, c_vir])
    o, v = slice(0, no), slice(no, no + nv)
    h = c.T @ np.asarray(hcore) @ c
    j = np.zeros_like(h)
    k = np.zeros_like(h)
    kw = {key: np.zeros((no, no)) for key in ('a', 'b')}
    lu = {key: np.zeros((nv, nv)) for key in ('a', 'b')}
    rho_s = []
    ex_s = np.zeros_like(singles)
    for b in local_factors(mol, c, auxbasis, max_memory_mb):
        boo, bov, bvv = b[:, o, o], b[:, o, v], b[:, v, v]
        j += np.einsum('Q,Qpq->pq', 2.0 * np.einsum('Qkk->Q', boo), b)
        k += np.einsum('Qpk,Qkq->pq', b[:, :, o], b[:, o, :])
        for key, t in (('a', t_a), ('b', t_b)):
            kw[key] += np.einsum('Qjb,kb,Qki->ji', bov, t, boo, optimize=True)
            lu[key] += np.einsum('Qjc,jb,Qba->ca', bov, t, bvv, optimize=True)
        rho_s.append(np.einsum('Qia,mia->Qm', bov, singles))
        ex_s += np.einsum('Qij,mjb,Qba->mia', boo, singles, bvv, optimize=True)
    f = h + j - k
    f = 0.5 * (f + f.T)
    fov = f[o, v]
    term = np.zeros((no, nv))
    for u, w, ku, kw_w in ((t_a, t_b, 'a', 'b'), (t_b, t_a, 'b', 'a')):
        term += w @ fov.T @ u - kw[kw_w].T @ u + w @ lu[ku]
    tt = np.sqrt(1.5) * np.einsum('mia,ia->m', singles, term)
    rho_s = np.vstack(rho_s)
    one = (np.einsum('mia,nib,ab->mn', singles, singles, f[v, v])
           - np.einsum('mia,nja,ji->mn', singles, singles, f[o, o]))
    ss = one + 2.0 * rho_s.T @ rho_s - np.einsum('mia,nia->mn', singles, ex_s)
    return dict(tt=tt, ss=0.5 * (ss + ss.T), fock=f)


@dataclass(frozen=True)
class TTACoupling:
    """The 1(TT) -> S1 S0 coupling of one site singlet (module docstring).
    Energies in Hartree.

    singlet: its label; triplets: the two triplet labels; energy: E;
    e_s1, e_tt: the two diabats' energies; e_ct: the charge-transfer diagonals;
    ct: their labels; v_direct: <S1 S0|H|1(TT)>; v_tt_ct: <CT|H|1(TT)>;
    v_ct_s: the A_eff couplings CT - S1 S0; v_eff: V_eff(E);
    mediated: v_eff - v_direct; bare_singles: the bare <S|H|S'> of
    [S1 S0, CT...]; spin: the channel of the pair (always 'singlet')."""
    singlet: str
    triplets: tuple
    energy: float
    e_s1: float
    e_tt: float
    e_ct: np.ndarray
    ct: tuple
    v_direct: float
    v_tt_ct: np.ndarray
    v_ct_s: np.ndarray
    v_eff: float
    mediated: float
    bare_singles: np.ndarray
    spin: str = 'singlet'


def _amplitudes(partition, label):
    if not partition.operator.tda:
        raise ValueError('the configuration model reads excitation '
                         'amplitudes: take the Tamm-Dancoff kernel (or ADC(2)), '
                         'not the full BSE')
    o = partition.orbitals
    return partition.p_local[:, partition.labels.index(label)].reshape(
        o.nocc, o.nvir)


def tta_coupling(singlet, triplet, mf, auxbasis, energy=None,
                 triplets=('site 0.0', 'site 1.0'), max_memory_mb=DF_BLOCK_MB):
    """{site singlet label: TTACoupling} from a singlet partition (both
    site singlets and both charge-transfer diabats explicit) and a triplet
    partition (both site triplets) on the same local orbitals of the mean
    field `mf`; `auxbasis` fits the integrals. `energy`: E of the fold,
    default (E_S1 + E_TT) / 2 per singlet; refused at or above a
    charge-transfer level of A_eff."""
    if getattr(singlet.operator, 'spin', None) != 'singlet' or getattr(
            triplet.operator, 'spin', None) != 'triplet':
        raise ValueError('give a singlet partition and a triplet partition, '
                         'in that order')
    o, ot = singlet.orbitals, triplet.orbitals
    if o.nfrag != 2:
        raise ValueError(f'the pair model is for two fragments, not {o.nfrag}')
    if not (np.array_equal(o.c_occ, ot.c_occ)
            and np.array_equal(o.c_vir, ot.c_vir)):
        raise ValueError('the two partitions must share their local orbitals: '
                         'build both on one FragmentOrbitals')
    sites = [lab for lab in singlet.labels if lab.startswith('site')]
    cts = [lab for lab in singlet.labels if lab.startswith('ct')]
    if not cts:
        raise ValueError('the singlet partition holds no charge-transfer '
                         'diabat: keep both explicit (ct=...)')
    t_a, t_b = (_amplitudes(triplet, lab) for lab in triplets)
    order = sites + cts
    singles = np.array([_amplitudes(singlet, lab) for lab in order])
    el = triplet_pair_elements(o.mol, mf.get_hcore(), o.c_occ, o.c_vir,
                               auxbasis, t_a, t_b, singles, max_memory_mb)
    a = singlet.a_eff
    ic = [singlet.labels.index(lab) for lab in cts]
    a_cc = a[np.ix_(ic, ic)]
    it = [triplet.labels.index(lab) for lab in triplets]
    e_tt = float(np.trace(triplet.a_eff[np.ix_(it, it)]))
    v_tt_ct = el['tt'][len(sites):]
    out = {}
    for n, lab in enumerate(sites):
        i = singlet.labels.index(lab)
        e_s1 = float(a[i, i])
        e = 0.5 * (e_s1 + e_tt) if energy is None else float(energy)
        require_below(a_cc, e)
        v_ct_s = a[ic, i]
        mediated = float(v_tt_ct @ np.linalg.solve(e * np.eye(len(ic)) - a_cc,
                                                   v_ct_s))
        rows = [n] + list(range(len(sites), len(order)))
        out[lab] = TTACoupling(
            singlet=lab, triplets=tuple(triplets), energy=e, e_s1=e_s1,
            e_tt=e_tt, e_ct=np.diag(a_cc).copy(), ct=tuple(cts),
            v_direct=float(el['tt'][n]), v_tt_ct=v_tt_ct.copy(),
            v_ct_s=np.asarray(v_ct_s, float).copy(),
            v_eff=float(el['tt'][n]) + mediated, mediated=mediated,
            bare_singles=el['ss'][np.ix_(rows, rows)])
    return out
