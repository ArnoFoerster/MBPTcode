"""Density-fitted assembly of the correlation gradient's two-electron term.

The dense route contracts the correlation two-particle density against
d(mu nu|lam sig)/dR, which costs 3 N^4 derivative integrals and an N^4 tensor
to hold. Under RI the same term needs only three- and two-centre derivatives,

    (pq|rs) = sum_PQ J_pq,P [V^-1]_PQ J_rs,Q,     V_PQ = (P|Q)

    dE_2 = 2 sum_pq,P Gamma3_pq,P dJ_pq,P - sum_PQ Gamma2_PQ dV_PQ
    Gamma3 = Gamma_s K,   Gamma2 = K^T Gamma3,    K = J V^-1

with Gamma_s the pair-swap-symmetrised coefficient tensor. The derivative
integrals drop from 3 N^4 to 3 N^2 naux and the N^4 tensor is replaced by an
(N^2, naux) one -- the shape a separable/ISDF factorization also produces, so
this is the assembly both routes feed.

The FORWARD side of the same factorization -- `df_integrals` and the fitted
four-index `df_eri_mo` the gradient here differentiates -- is `Base.eri_blocks`.

pyscf sign convention, verified against finite differences in Bohr:
int3c2e_ip1 = -(grad mu nu|P), int3c2e_ip2 = -(mu nu|grad P),
int2c2e_ip1 = -(grad P|Q).
"""
import numpy as np
from pyscf import df as pyscf_df

from src.Base.eri_blocks import df_eri_mo, df_integrals  # noqa: F401


def df_densities(Gamma4, C, mol, auxmol):
    """(Gamma3_ao, Gamma2) for an MO two-particle coefficient tensor.

    Gamma4 is symmetrised over both intra-pair swaps and the pair swap first,
    which is free -- (pq|rs) already has all three symmetries -- and makes the
    AO three-index density symmetric in (mu, nu).
    """
    norb = Gamma4.shape[0]
    G = 0.5 * (Gamma4 + Gamma4.transpose(1, 0, 2, 3))
    G = 0.5 * (G + G.transpose(0, 1, 3, 2))
    G = 0.5 * (G + G.transpose(2, 3, 0, 1))
    J, V = df_integrals(mol, auxmol)
    naux = V.shape[0]
    J_mo = np.einsum('mp,nq,mnP->pqP', C, C, J, optimize=True)
    K_mo = np.linalg.solve(V, J_mo.reshape(-1, naux).T).T.reshape(J_mo.shape)
    Gamma3 = np.einsum('pqrs,rsP->pqP', G, K_mo, optimize=True)
    Gamma2 = np.einsum('pqP,pqQ->PQ', K_mo, Gamma3, optimize=True)
    Gamma3_ao = np.einsum('mp,nq,pqP->mnP', C, C, Gamma3, optimize=True)
    return Gamma3_ao, 0.5 * (Gamma2 + Gamma2.T)


def two_electron_skeleton_df(mol, auxmol, Gamma3_list, Gamma2_list):
    """sum_pqrs Gamma_pqrs d(pq|rs)/dR_A under RI, one derivative sweep.

    Gamma3 must be symmetric in (mu, nu), so the two AO-derivative terms of
    dJ are equal and enter once with a factor 2.
    """
    nao, naux = mol.nao, auxmol.nao
    d1 = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', aosym='s1',
                                comp=3).reshape(3, nao, nao, naux)
    d2 = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2', aosym='s1',
                                comp=3).reshape(3, nao, nao, naux)
    v1 = auxmol.intor('int2c2e_ip1', comp=3)
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()

    out = np.zeros((len(Gamma3_list), mol.natm, 3))
    for k, (G3, G2) in enumerate(zip(Gamma3_list, Gamma2_list)):
        # dJ on an AO centre, both index positions, and on the auxiliary centre
        t_ao = -2.0 * np.einsum('xmnP,mnP->xm', d1, G3, optimize=True)
        t_aux = -np.einsum('xmnP,mnP->xP', d2, G3, optimize=True)
        # -Gamma2 dV, symmetric in (P, Q)
        t_v = 2.0 * np.einsum('xPQ,PQ->xP', v1, G2, optimize=True)
        for iatm in range(mol.natm):
            p0, p1 = aoslices[iatm, 2:]
            q0, q1 = auxslices[iatm, 2:]
            out[k, iatm] = (2.0 * (t_ao[:, p0:p1].sum(axis=1)
                                   + t_aux[:, q0:q1].sum(axis=1))
                            + t_v[:, q0:q1].sum(axis=1))
    return out
