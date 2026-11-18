"""The one-body term of the downfolded Hamiltonian, per spin channel.

h_eff^sigma = f^sigma - t^DC,sigma (+ Sigma_bar^sigma): the Fock operator of
the frozen inactive core on the active orbitals, minus the double counting of
the effective interaction vtilde with the active reference density. Every
channel quantity is a tuple (alpha, beta); a restricted reference passes two
identical channels. Active indices are local to `Partition.act` (occupied
first), so the active occupied orbitals of a channel are its first n_act_occ.
"""
import numpy as np
from pyscf.scf.hf import get_jk

from .terms import choice

DC_MODES = ('full', 'hartree')


def inactive_jk(mol, mo, parts):
    """(dms, j, k): the inactive densities D_I^sigma = C_I C_I^T of the two
    spin channels as one (2, nao, nao) array, the Coulomb matrix
    J[D_I^a + D_I^b] of the whole core, and the exchange matrices
    (K[D_I^a], K[D_I^b]).

    J and K are the exact 4-centre ones of `mol`: the bare v of the effective
    Hamiltonian is never density fitted. mo = (C_a, C_b), parts = (part_a,
    part_b).
    """
    dms = np.array([c[:, p.inactive] @ c[:, p.inactive].T
                    for c, p in zip(mo, parts)])
    vj, vk = get_jk(mol, dms)
    return dms, vj[0] + vj[1], (vk[0], vk[1])


def core_fock(mol, hcore_ao, mo, parts):
    """(f_a, f_b), each (n_act, n_act): the Fock operator of the inactive core
    on the active orbitals of each spin channel.

    f^sigma = h + J[D_I^a + D_I^b] - K[D_I^sigma] with D_I^sigma = C_I C_I^T
    the inactive density of one spin (one electron per spin orbital), so a
    restricted reference gives h_tu + sum_{i in I} [2(tu|ii) - (ti|iu)].
    J and K are the exact ones of `inactive_jk`. mo = (C_a, C_b), parts =
    (part_a, part_b); hcore_ao is the AO core Hamiltonian.
    """
    _, j, vk = inactive_jk(mol, mo, parts)
    return tuple(c[:, p.act].T @ (hcore_ao + j - k) @ c[:, p.act]
                 for c, p, k in zip(mo, parts, vk))


def double_counting(vtilde, n_act_occ):
    """(hartree, exchange), each (a, b) of (n_act, n_act): the Hartree and
    exchange parts of the double counting of vtilde with the active
    reference density.

    hartree^s_tu  = sum_{k in A_o^s} vt^{ss}_tu,kk + sum_{k in A_o^s'} vt^{ss'}_tu,kk
    exchange^s_tu = sum_{k in A_o^s} vt^{ss}_tk,ku
    so t^DC = hartree - exchange. vtilde = (aa, ab, bb) with
    ab[t,u,v,w] = (t_a u_a | v_b w_b); the beta channel reads the
    opposite-spin block as ab.transpose(2, 3, 0, 1). n_act_occ = (n_a, n_b).
    """
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
    """(h_a, h_b): the one-body part of H_eff in each spin channel.

    dc='full' subtracts the whole double counting, f - hartree + exchange;
    dc='hartree' subtracts the Hartree part only, f - hartree, the convention
    that goes with the static self-energy Sigma_bar (which then carries the
    exchange). sigma_bar, when given as (a, b), is added.
    """
    dc = choice(dc, DC_MODES, 'dc')
    h = []
    for s in (0, 1):
        h_s = f[s] - hartree[s]
        if dc == 'full':
            h_s = h_s + exchange[s]
        if sigma_bar is not None:
            h_s = h_s + sigma_bar[s]
        h.append(h_s)
    return tuple(h)
