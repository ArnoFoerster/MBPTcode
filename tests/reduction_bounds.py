"""Rounding bounds of a sum reduced over ranks, from the one-rank kernel's own
addends: the gate the tests put on every such sum -- K over the ISDF row
tiles, proj(tau) over the polarizability's grid-row tiles, the self-energy's
branch sums over its block pairs -- in place of sampled regroupings of it.

Three parts, none of them sampled:
  * the one-rank addends added in the one-rank order are the one-rank result,
    bitwise (`regrouped`);
  * every rank's partial is its own addends added in that order onto zeros,
    bitwise, zeros where a rank owns none (`rank_partials`);
  * the reduced result lies within a bound of an exact sum at every element
    (`exact_offset`, `math.fsum`), the bound floored at one ulp of the
    one-rank result's largest element.

An addition rounds by at most half an ulp of its result and not at all onto
zeros, and every join of two ranks' partial sums is below sum_r |partial_r|
in magnitude whatever tree the reduction takes, so the bounds are the most ANY
order of the reduction can move the sum -- worst cases, not measured
responses, and no multiplier goes on them. `rounding_bound` is taken against
the addends' exact sum: half an ulp of each partial sum a rank forms and of
each join. `join_bound` is taken against the exact sum of the ranks' partials,
which the second part pins bitwise: the joins' half-ulps alone. Where one rank
sums a whole tau point's sixteen tiles on water the first is 8.75 ulp of
|proj|max and passes proj moved 8 ulp at its largest element (0.78 of it);
the second is one ulp there and fails it eightfold.
"""
import math
from typing import NamedTuple

import numpy as np

ULP = np.finfo(float).eps


class Verdict(NamedTuple):
    """One rank's three parts for one reduced sum (`reduced_sum_verdict`)."""

    serial: bool       # the addends in order are the one-rank result
    partial: bool      # this rank's partial is its own addends in order
    ratio: float       # worst |reduced - exact| over the floored join bound
    bound_ulp: float   # that bound's largest element, in ulp of |whole|max


def regrouped(addends, groups):
    """The addends summed run by run, and the runs' sums added in turn."""
    total = np.zeros_like(addends[0])
    for group in groups:
        run = np.zeros_like(addends[0])
        for i in group:
            run = run + addends[i]
        total = total + run
    return total


def rank_partials(addends, owners):
    """Every rank's partial as its reduction receives it: the addends of its
    items `owners[r]` added in order onto zeros, zeros for a rank that owns
    none."""
    return [regrouped(addends, [own]) for own in owners]


def rounding_bound(addends, owners):
    """Per element, the most a sum of the tile addends over ranks can lie
    from their exact sum, whatever order its reduction takes: rank r adds
    the addends of its tiles `owners[r]` in order onto zeros, and the
    reduction joins the m nonzero partials in some tree.

    An addition rounds by at most half an ulp of its result, and not at all
    onto zeros -- a rank's first tile, a join with a tile-less rank -- so the
    bound is half an ulp of each partial sum a rank forms after its first
    tile, and m - 1 half-ulps of sum_r |partial_r|, which every join's result
    is below in magnitude (by 1 + m eps, the joins' own roundings)."""
    half = np.zeros_like(addends[0])
    parts = []
    for tiles in owners:
        total = np.zeros_like(addends[0])
        for k, t in enumerate(tiles):
            total = total + addends[t]
            if k:
                half += np.spacing(np.abs(total))
        if len(tiles):
            parts.append(total)
    m = len(parts)
    if m > 1:
        reach = (1 + 2 * m * ULP) * sum(np.abs(p) for p in parts)
        half += (m - 1) * np.spacing(reach)
    return half / 2


def join_bound(partials):
    """Per element, the most joining the m partials of the ranks that own
    addends can lie from their exact sum, whatever tree the reduction takes:
    m - 1 half-ulps of sum_r |partial_r|, which every join's result is below
    in magnitude (by 1 + 2 m eps, the joins' own roundings). A join with a
    rank that owns none adds zeros exactly, so one partial is the sum."""
    m = len(partials)
    if m < 2:
        return np.zeros_like(partials[0]) if m else 0.0
    reach = (1 + 2 * m * ULP) * sum(np.abs(p) for p in partials)
    return (m - 1) * np.spacing(reach) / 2


def exact_offset(value, addends):
    """value minus the exact sum of the addends, element by element, rounded
    once: `math.fsum` of the value and the negated addends."""
    cols = np.stack([np.asarray(value)] + [-np.asarray(a) for a in addends])
    cols = cols.reshape(len(addends) + 1, -1).T
    return np.array([math.fsum(c) for c in cols]).reshape(np.shape(value))


def reduced_sum_verdict(whole, addends, owners, rank, partial, got,
                        rows=None):
    """Rank `rank`'s three parts for a sum over ranks of the one-rank
    `addends`, whose one-rank result is `whole`: `owners` every rank's items
    in order, rank-ordered; `partial` what this rank handed the reduction;
    `got` the reduced sum, or the rows [r0, r1) = `rows` of its leading axis
    this rank received. The bound is `join_bound` of the ranks' partials
    (`rank_partials`, which the partial part pins on every rank), floored at
    one ulp of |whole|max."""
    parts = rank_partials(addends, owners)
    cut = slice(None) if rows is None else slice(*rows)
    held = [p[cut] for p, own in zip(parts, owners) if len(own)]
    ulp = np.spacing(np.abs(whole).max())
    bound = np.maximum(join_bound(held), ulp)
    off = np.abs(exact_offset(got, held))
    return Verdict(
        np.array_equal(regrouped(addends, [range(len(addends))]), whole),
        np.array_equal(partial, parts[rank]),
        float((off / bound).max(initial=0.0)),
        float(bound.max(initial=0.0) / ulp))
