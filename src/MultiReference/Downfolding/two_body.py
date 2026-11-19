"""The two-body term of the downfolded Hamiltonian, per spin channel.

v_eff = v + vtilde: the bare interaction on the active orbitals, from exact
4-centre integrals (`active_eri`), plus the correction the low-level method
hands back (`effective_interaction`). All the physics of the screening is the
method's; what is general is only the bare v and the sum. Spin blocks as in
one_body.py: (v,) for the one channel of a restricted reference, or
(aa, ab, bb).
"""
import numpy as np
from pyscf import ao2mo


def active_eri(mol, mo_act):
    """The exact chemist integrals (tu|vw) of the active orbitals, each block
    (n_act,)*4: (v,) for the one channel of a restricted reference, from one
    integral transformation, or (aa, ab, bb) for two channels, with
    ab[t,u,v,w] = (t_a u_a | v_b w_b).

    mo_act = (C_A,) or (C_Aa, C_Ab) are the active orbital coefficients of the
    spin channels. The integrals are the exact 4-centre ones of `mol`, never
    from a density-fitted object, because the bare v must not carry a fitting
    error. An empty active space gives zero-size blocks without a
    transformation.
    """
    if len(mo_act) == 1:
        return (_block(mol, mo_act[0], mo_act[0]),)
    ca, cb = mo_act
    return (_block(mol, ca, ca), _block(mol, ca, cb), _block(mol, cb, cb))


def _block(mol, c1, c2):
    n1, n2 = c1.shape[1], c2.shape[1]
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n1, n2, n2))
    eri = ao2mo.general(mol, (c1, c1, c2, c2), compact=False)
    return eri.reshape(n1, n1, n2, n2)


def effective_interaction(mol, mo_act, vtilde):
    """v_eff = v + vtilde, one array per spin block: the two-body part of
    H_eff in chemist order (tu|vw) on the active orbitals.

    v is the exact bare interaction of `active_eri` on mo_act = (C_A,) or
    (C_Aa, C_Ab); vtilde is the method's correction on the same spin blocks,
    (v,) or (aa, ab, bb). A vtilde with another number of blocks, or of
    another shape, is refused.
    """
    bare = active_eri(mol, mo_act)
    if len(vtilde) != len(bare) or any(np.shape(vt) != v.shape
                                       for v, vt in zip(bare, vtilde)):
        raise ValueError(
            f'vtilde has blocks of shape {[np.shape(vt) for vt in vtilde]}; '
            f'the active interaction of {len(mo_act)} spin channel(s) has '
            f'{[v.shape for v in bare]}.')
    return tuple(v + vt for v, vt in zip(bare, vtilde))
