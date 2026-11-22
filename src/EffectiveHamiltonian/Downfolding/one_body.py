"""The one-body term of the downfolded Hamiltonian, per spin channel.

h_eff^sigma = f^sigma - t^DC,sigma (+ Sigma_bar^sigma): the Fock operator of
the frozen inactive core on the active orbitals, minus the double counting of
the effective interaction vtilde with the active reference density.

Every quantity is per spin channel, and vtilde is (v,) or (aa, ab, bb): the
convention of spin.py.
"""
import numpy as np
from pyscf.scf.hf import get_jk

from .options import choice
from .spin import spin_weight

DC_MODES = ('full', 'hartree')


def inactive_jk(mol, mo, parts):
    """(dms, j, k): the inactive densities D_I^sigma = C_I C_I^T of the spin
    channels, one electron per spin orbital, as one (nchannel, nao, nao)
    array; the Coulomb matrix J[D_I] of the whole core, D_I = D_I^a + D_I^b
    (2 D_I^sigma of the one restricted channel); and the exchange matrices
    K[D_I^sigma], one per channel.

    J and K are the exact 4-centre ones of `mol`: the bare v of the effective
    Hamiltonian is never density fitted. One build serves both `core_fock`
    and the environment energy, which take it as `jk`; with no inactive
    orbital nothing is built. mo = (C,) or (C_a, C_b), parts likewise.
    """
    weight = spin_weight(mo)
    dms = np.array([c[:, p.env_occ] @ c[:, p.env_occ].T
                    for c, p in zip(mo, parts)])
    if not any(len(p.env_occ) for p in parts):
        return dms, np.zeros_like(dms[0]), tuple(np.zeros_like(d) for d in dms)
    vj, vk = get_jk(mol, dms)
    return dms, weight * sum(vj), tuple(vk)


def core_fock(mol, hcore_ao, mo, parts, jk=None):
    """One (n_act, n_act) array per spin channel: the Fock operator of the
    inactive core on the active orbitals.

    f^sigma = h + J[D_I] - K[D_I^sigma] with D_I^sigma = C_I C_I^T the
    inactive density of one spin (one electron per spin orbital) and D_I the
    whole core's, so a restricted reference gives
    h_tu + sum_{i in I} [2(tu|ii) - (ti|iu)]. J and K are the exact ones of
    `inactive_jk`, built here unless handed in as `jk`. mo = (C,) or
    (C_a, C_b), parts likewise; hcore_ao is the AO core Hamiltonian.
    """
    _, j, vk = inactive_jk(mol, mo, parts) if jk is None else jk
    return tuple(c[:, p.act].T @ (hcore_ao + j - k) @ c[:, p.act]
                 for c, p, k in zip(mo, parts, vk))


def double_counting(vtilde, n_act_occ):
    """(hartree, exchange), each one (n_act, n_act) array per spin channel:
    the Hartree and exchange parts of the double counting of vtilde with the
    active reference density.

    hartree^s_tu  = sum_{k in A_o^s} vt^{ss}_tu,kk + sum_{k in A_o^s'} vt^{ss'}_tu,kk
    exchange^s_tu = sum_{k in A_o^s} vt^{ss}_tk,ku
    so t^DC = hartree - exchange. Two channels: vtilde = (aa, ab, bb) with
    ab[t,u,v,w] = (t_a u_a | v_b w_b), the beta channel reading the
    opposite-spin block as ab.transpose(2, 3, 0, 1), and n_act_occ =
    (n_a, n_b). One restricted channel: vtilde = (v,), n_act_occ = (n,), and
    the opposite spin sees the same block, hartree_tu = 2 sum_k vt_tu,kk.
    """
    if spin_weight(n_act_occ) == 2:
        (v,), (n,) = vtilde, n_act_occ
        return ((2 * np.einsum('tukk->tu', v[:, :, :n, :n]),),
                (np.einsum('tkku->tu', v[:, :n, :n, :]),))
    aa, ab, bb = vtilde
    n_a, n_b = n_act_occ
    channels = ((aa, ab, n_a, n_b), (bb, ab.transpose(2, 3, 0, 1), n_b, n_a))
    hartree, exchange = [], []
    for same, other, n_own, n_other in channels:
        hartree.append(np.einsum('tukk->tu', same[:, :, :n_own, :n_own])
                       + np.einsum('tukk->tu', other[:, :, :n_other, :n_other]))
        exchange.append(np.einsum('tkku->tu', same[:, :n_own, :n_own, :]))
    return tuple(hartree), tuple(exchange)


def effective_one_body(f, hartree, exchange, dc, sigma_bar=None):
    """The one-body part of H_eff, one array per spin channel.

    dc='full' subtracts the whole double counting, f - hartree + exchange;
    dc='hartree' subtracts the Hartree part only, f - hartree, the convention
    that goes with the static self-energy Sigma_bar (which then carries the
    exchange). sigma_bar, one array per channel when given, is added.
    """
    dc = choice(dc, DC_MODES, 'dc')
    h = []
    for s in range(len(f)):
        h_s = f[s] - hartree[s]
        if dc == 'full':
            h_s = h_s + exchange[s]
        if sigma_bar is not None:
            h_s = h_s + sigma_bar[s]
        h.append(h_s)
    return tuple(h)
