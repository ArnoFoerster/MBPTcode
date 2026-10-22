"""SCW83 Appendix-A resummation of the seven-line-cut quartet class.

Census fact (order 4, all four external blocks, gated below): every skeleton
with a seven-line (4p3h/4h3p) omega-cut has EXACTLY four such time orderings,
with cut signatures

    O1 [3w 7w 3w]   O2 [3w 7w 4f]   O3 [4f 7w 3w]   O4 [4f 7w 4f]

and all four share the same middle bipartition {bra-vertex, W} | {W', ket-
vertex}; hence one common seven-line cut energy Em, one bra-side three-line
energy Ea, one ket-side Eb, and the two 2p2h block energies

    E_f  = Em - Ea   (the W  block),   E_f' = Em - Eb   (the W' block),

which are the free-cut energies of O2/O3/O4.  Partial-fractioning the four
orderings in omega and summing:

  * the 4p3h-pole residues cancel PER INDEX TUPLE:
        (1 - 1 - 1 + 1) / (E_f E_f') = 0,
  * the survivors combine to the manifestly regular closed form (u = dir*w)

        G(u) = (u + Em) / ((u + Ea)(u + Eb) E_f E_f'),

    with no dangerous (Eb - Ea) denominator and the degenerate tuples
    (Ea == Eb, which forces E_f == E_f') included automatically.

G is the explicit-denominator content the ISR-ADC(4) representation cannot
produce: (u+Em)/((u+Ea)(u+Eb)) splits into a
C^(2)-shaped double-resolvent piece plus U^(3)-shaped single poles, with
chained 1/(E_f E_f') weights in place of the amplitude-factorized ones.

This module is validation-scale dense numerics, like goldstone.py; the
symbolic emission consumes the same group data.
"""
import numpy as np

from src.CodeGeneration.diagrammatic.goldstone import (
    _BLOCKS, build_skeletons, ordering_sector, ordering_sign, _ordering_cuts,
    sevenline_filter, BlockEvaluator)


def _signature(skel, ordering):
    return tuple((('w' if d != 0 else 'f'), len(h) + len(p))
                 for d, h, p in _ordering_cuts(skel, ordering))


_ROLE = {(('w', 3), ('w', 7), ('w', 3)): 'O1',
         (('w', 3), ('w', 7), ('f', 4)): 'O2',
         (('f', 4), ('w', 7), ('w', 3)): 'O3',
         (('f', 4), ('w', 7), ('f', 4)): 'O4'}


class QuartetGroup:
    """One skeleton's four seven-line-cut orderings, role-tagged."""

    def __init__(self, skel):
        self.skel = skel
        self.orderings = {}
        for o in skel.orderings:
            if not sevenline_filter(skel, o):
                continue
            sig = _signature(skel, o)
            role = _ROLE.get(sig)
            if role is None or role in self.orderings:
                raise ValueError(
                    f"quartet census violated for {skel.raw}: sig {sig}")
            self.orderings[role] = o
        if set(self.orderings) != {'O1', 'O2', 'O3', 'O4'}:
            raise ValueError(
                f"incomplete quartet for {skel.raw}: {set(self.orderings)}")
        # cut data from O1: [a-cut, m-cut, b-cut]; sector from any member
        self.cuts = _ordering_cuts(skel, self.orderings['O1'])
        self.sector = ordering_sector(skel, self.orderings['O1'])

    def consistency(self):
        """The structural identities the closed form relies on: all four
        orderings share the middle seven-line cut (same crossed lines), and
        O1's outer cuts reappear as O2/O3's omega-cuts.  Returns True or
        raises."""
        mid = frozenset(self.cuts[1][1] + self.cuts[1][2])
        for role in ('O2', 'O3', 'O4'):
            cuts = _ordering_cuts(self.skel, self.orderings[role])
            m = frozenset(cuts[1][1] + cuts[1][2])
            if m != mid:
                raise ValueError(f"middle cut differs in {role}: {self.skel.raw}")
        a = frozenset(self.cuts[0][1] + self.cuts[0][2])
        b = frozenset(self.cuts[2][1] + self.cuts[2][2])
        a2 = _ordering_cuts(self.skel, self.orderings['O2'])[0]
        b3 = _ordering_cuts(self.skel, self.orderings['O3'])[2]
        if frozenset(a2[1] + a2[2]) != a or frozenset(b3[1] + b3[2]) != b:
            raise ValueError(f"outer cuts differ across roles: {self.skel.raw}")
        return True


def quartet_groups(block, order=4):
    """All QuartetGroups of one external block."""
    skels, _ = build_skeletons(order, block)
    return [QuartetGroup(s) for s in skels
            if any(sevenline_filter(s, o) for o in s.orderings)]


class QuartetEvaluator(BlockEvaluator):
    """Closed-form (resummed) evaluation of the quartet class."""

    def _cut_energy(self, cut, dummies):
        """omega-free part of a cut denominator, broadcast like
        _cut_denominator (sum eps_holes - sum eps_particles)."""
        d, holes, particles = cut
        return self._cut_denominator((0, holes, particles), dummies, 0.0, 0)

    def eval_group(self, grp, omega):
        """w * sum_tuples N * (u+Em)/((u+Ea)(u+Eb) E_f E_f'), u = dir*omega.

        dir is the shared omega-cut direction of the group's orderings (all
        cuts of one ordering share dir; O1's three omega-cuts pin it)."""
        skel = grp.skel
        num, dummies = self._numerator(skel)
        d = grp.cuts[1][0]
        u = d * omega
        Ea = self._cut_energy(grp.cuts[0], dummies)
        Em = self._cut_energy(grp.cuts[1], dummies)
        Eb = self._cut_energy(grp.cuts[2], dummies)
        Ef = Em - Ea
        Efp = Em - Eb
        w = (skel.coeff / skel.n_ext) * ordering_sign(skel, grp.sector)
        val = num * (u + Em) / ((u + Ea) * (u + Eb) * Ef * Efp)
        return w * val.sum(axis=tuple(range(2, val.ndim)))

    def eval_groups(self, groups, omega):
        out = None
        for grp in groups:
            contrib = self.eval_group(grp, omega)
            out = contrib if out is None else out + contrib
        return out
