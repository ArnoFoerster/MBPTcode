"""The split of the canonical orbitals into the active space and its environment.

`Partition` carries the index sets every downfolding step reads, and the two
masks that pick out the particle-hole parts of a two-body object. All indices
are canonical spatial MO indices unless a function says they are local to the
active orbitals.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import QP_DEGENERACY_TOL


@dataclass(frozen=True)
class Partition:
    """Active and environment orbitals of a closed-shell reference.

    inactive: the occupied orbitals outside the active space.
    act_occ, act_virt: the active occupied and virtual orbitals.
    env_virt: the virtual orbitals outside the active space.
    nocc, norb: the occupied count and the orbital count of the reference.
    """
    inactive: np.ndarray
    act_occ: np.ndarray
    act_virt: np.ndarray
    env_virt: np.ndarray
    nocc: int
    norb: int

    @property
    def act(self):
        """The active orbitals, occupied first."""
        return np.concatenate([self.act_occ, self.act_virt])

    @property
    def n_act(self):
        return len(self.act_occ) + len(self.act_virt)

    @property
    def n_act_occ(self):
        return len(self.act_occ)

    @property
    def env(self):
        """The environment orbitals: the inactive occupied, then the environment virtual."""
        return np.concatenate([self.inactive, self.env_virt])


def partition(active_space, eps):
    """The Partition of `active_space`, refusing a boundary that cuts a degenerate pair.

    An active and an environment orbital of the same kind (both occupied or
    both virtual) closer than QP_DEGENERACY_TOL would make the energy
    denominators between the two sets vanish, so the active space has to take
    the whole degenerate set or none of it. An occupied orbital close to a
    virtual one is not a cut and is never compared.
    """
    eps = np.asarray(eps)
    for act, env in ((active_space.occ_active, active_space.env_occ),
                     (active_space.virt_active, active_space.env_virt)):
        for a in act:
            for e in env:
                if abs(eps[a] - eps[e]) < QP_DEGENERACY_TOL:
                    raise ValueError(
                        f"the active space cuts a degenerate pair: active "
                        f"orbital {a} (eps = {eps[a]:.8f}) and environment "
                        f"orbital {e} (eps = {eps[e]:.8f}) differ by less "
                        f"than {QP_DEGENERACY_TOL:g}. Take the whole set, "
                        f"e.g. with ActiveSpace.from_irreps or "
                        f"ActiveSpace.from_population.")
    return Partition(inactive=active_space.env_occ,
                     act_occ=active_space.occ_active,
                     act_virt=active_space.virt_active,
                     env_virt=active_space.env_virt,
                     nocc=active_space.nocc, norb=active_space.norb)


def active_pair_mask(part):
    """Boolean mask over the nocc*nvirt (i, a) pairs, True where i and a are both active.

    The order is C order over (i, a), i over all occupied orbitals and a over
    all virtual orbitals counted from nocc.
    """
    occ = np.zeros(part.nocc, dtype=bool)
    occ[part.act_occ] = True
    virt = np.zeros(part.norb - part.nocc, dtype=bool)
    virt[part.act_virt - part.nocc] = True
    return np.outer(occ, virt).ravel()


def ph_element_mask(part):
    """Boolean (n_act,)*4 mask over chemist elements (tu|vw), True where both pairs are particle-hole.

    t, u, v, w are local indices into `part.act` (occupied first). A pair is
    particle-hole when it holds one active occupied and one active virtual
    index, in either order, so all of (ia|jb), (ai|jb), (ia|bj) and (ai|bj)
    are covered.
    """
    is_occ = np.arange(part.n_act) < part.n_act_occ
    pair = is_occ[:, None] ^ is_occ[None, :]
    return pair[:, :, None, None] & pair[None, None, :, :]
