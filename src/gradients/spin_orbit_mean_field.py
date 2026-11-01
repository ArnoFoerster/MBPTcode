"""The two-electron term of the spin-orbit mean-field operator, built once over
the ranks.

The spin-orbit mean-field reduction of Neese (J. Chem. Phys. 122, 034107
(2005), Eq. 19) needs J - (3/2) K over `int2e_p1vxp1` at the ground-state
density: a pass over nao^4 x 3 integrals (or sum_A nao_A^4 in the atomic
mean-field approximation), the one step of `properties.spin_orbit` whose cost
grows with the fourth power of the basis. Under a communicator rank 0 runs it
from its own density and `lockstep` sends the (3, nao, nao) result to every
rank; the other ranks run no integral pass, and every rank holds rank 0's
operator to the bit whatever density it was handed. Outside a communicator it
is the plain serial build, the same bits.

This is a realization kernel: `comm` defaults to `current_comm()`, and the
property code that calls it never sees a communicator.
"""
import copy

import numpy as np
from pyscf import lib
from pyscf.scf import _vhf

from src.Base.utils.parallel.mpi_grid import current_comm, lockstep

#: The two-electron reductions `mean_field_term` builds.
TWO_ELECTRON_TERMS = ('somf', 'amfi')


def _soc_jk(mol, dm):
    """(vj, vk) of the two-electron SOC operator at density `dm`, (3, nao, nao).

    K here is the sum of BOTH exchange contractions, (pi|iq) + (iq|pi), which is
    what the SOMF factor of 3/2 multiplies.
    """
    vj, vk, vk1 = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                                    'a4ij', ('lk->s2ij', 'jk->s1il', 'li->s1kj'),
                                    dm, 3, mol._atm, mol._bas, mol._env)
    for i in range(3):
        lib.hermi_triu(vj[i], hermi=2, inplace=True)
    return vj, vk + vk1


def _soc_jk_amfi(mol, dm):
    """(vj, vk) in the atomic-mean-field approximation: one-centre blocks only.

    The two-electron SOC operator is short-ranged and dominated by the nucleus
    it is centred on, so the multi-centre blocks are discarded. Cost falls from
    nao^4 to sum_A nao_A^4, which is what makes a 60-90 atom emitter reachable.
    """
    nao = mol.nao_nr()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    for b0, b1, p0, p1 in mol.aoslice_by_atom():
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = _soc_jk(atom, np.asarray(dm)[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def mean_field_term(mol, dm, two_electron='somf', comm=None):
    """(3, nao, nao) J - (3/2) K of the two-electron spin-orbit operator at the
    AO density `dm`, without the alpha^2/4 prefactor.

    two_electron: 'somf' (exact two-electron integrals) or 'amfi' (one-centre
        blocks only).
    comm: defaults to `current_comm()`. With more than one rank, rank 0 builds
        the term from ITS `dm` and every rank returns rank 0's array; the `dm`
        of the other ranks is not read. Collective, as `lockstep` is.
    """
    if two_electron not in TWO_ELECTRON_TERMS:
        raise ValueError(f"two_electron={two_electron!r}: one of "
                         f"{TWO_ELECTRON_TERMS}")
    comm = current_comm() if comm is None else comm
    if comm is not None and comm.Get_size() > 1 and comm.Get_rank() != 0:
        nao = mol.nao_nr()
        return lockstep(np.zeros((3, nao, nao)), comm)
    jk = _soc_jk_amfi if two_electron == 'amfi' else _soc_jk
    vj, vk = jk(mol, dm)
    return lockstep(np.ascontiguousarray(vj - 1.5 * vk), comm)
