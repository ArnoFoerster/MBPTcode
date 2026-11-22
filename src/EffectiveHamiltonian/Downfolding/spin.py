"""The spin channels of the downfolding: one convention, written down once.

A restricted reference (RHF/RKS) is ONE spin channel, whose orbitals each hold
two electrons, so nothing is computed twice; an unrestricted one (UHF/UKS) is
two, (alpha, beta), each with its own orbitals, occupied count and active
space, also when the two coincide. Every per-channel quantity is a tuple over
the channels: (x,) or (x_alpha, x_beta).

A two-body quantity on the active orbitals is a tuple of spin blocks in
chemist order: (v,), one spin-independent block, for one channel, or
(aa, ab, bb) for two, with the opposite-spin block oriented
ab[t,u,v,w] = (t_a u_a | v_b w_b); `spin_block` reads the block of any pair
of channels from it. Active indices are local to `ActiveSpace.act`, occupied
first, so the active occupied orbitals of a channel are its first n_act_occ.

The one-channel formulas are the two-channel ones with both channels equal,
written once with the spin weight g (`spin_weight`), the number of electrons
an occupied orbital of one channel holds: 2 for a restricted reference, 1
for each of two.
"""

#: The names of the two channels of an unrestricted reference.
SPINS = ('alpha', 'beta')

#: The channel pairs (first pair, second pair) of the two-body blocks: the
#: one block (v,) of one channel, or (aa, ab, bb) of two.
BLOCKS = {1: ((0, 0),), 2: ((0, 0), (0, 1), (1, 1))}


def spin_weight(channels):
    """g, the electrons an occupied orbital of one channel holds: 2 for the
    one channel of a restricted reference, 1 for each of (alpha, beta).
    `channels` is any per-channel tuple."""
    if len(channels) not in (1, 2):
        raise ValueError(f'{len(channels)} spin channels; a restricted '
                         f'reference has one, an unrestricted one two.')
    return 2 if len(channels) == 1 else 1


def channels_of(space):
    """The spin channels of `space` as a tuple: (space,) for one ActiveSpace,
    the one channel of a restricted reference, or the pair (alpha, beta) of
    an unrestricted one."""
    return (space,) if not isinstance(space, (tuple, list)) else tuple(space)


def spin_block(blocks, s, t):
    """The block of `blocks` ((v,) or (aa, ab, bb)) with the first pair in
    spin channel s and the second in channel t: the one block of one channel
    for any s, t; for two, the beta-alpha block is the alpha-beta one with
    its pairs exchanged."""
    if len(blocks) == 1:
        return blocks[0]
    aa, ab, bb = blocks
    return ((aa, ab), (ab.transpose(2, 3, 0, 1), bb))[s][t]
