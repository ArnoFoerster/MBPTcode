"""The particle-hole masks of an active space.

The downfolding reads the active space of each spin channel as an
`ActiveSpace` (src/EffectiveHamiltonian/active_space.py). The two masks pick
out the particle-hole parts of a pair list and of a two-body object; all
indices are canonical spatial MO indices unless a function says they are
local to the active orbitals.
"""
import numpy as np


def active_pair_mask(space):
    """Boolean mask over the nocc*nvirt (i, a) pairs, True where i and a are both active.

    The order is C order over (i, a), i over all occupied orbitals and a over
    all virtual orbitals counted from nocc.
    """
    occ = np.zeros(space.nocc, dtype=bool)
    occ[space.occ_active] = True
    virt = np.zeros(space.norb - space.nocc, dtype=bool)
    virt[space.virt_active - space.nocc] = True
    return np.outer(occ, virt).ravel()


def ph_element_mask(space, second=None):
    """Boolean (n_act,)*4 mask over chemist elements (tu|vw), True where both pairs are particle-hole.

    t, u, v, w are local indices into `space.act` (occupied first). A pair is
    particle-hole when it holds one active occupied and one active virtual
    index, in either order, so all of (ia|jb), (ai|jb), (ia|bj) and (ai|bj)
    are covered. `second`: the ActiveSpace of the spin channel of the second
    pair, v and w, for the opposite-spin block of an unrestricted reference
    (default: both pairs in the channel of `space`).
    """
    def pair(p):
        is_occ = np.arange(p.n_act) < p.n_act_occ
        return is_occ[:, None] ^ is_occ[None, :]

    first = pair(space)
    last = first if second is None else pair(second)
    return first[:, :, None, None] & last[None, None, :, :]
