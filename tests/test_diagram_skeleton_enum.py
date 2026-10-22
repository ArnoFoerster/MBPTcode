"""Gates for the graph-native skeleton enumerator (route B).

The enumerator replaces the pdaggerq fixed-end stream whose internal Wick
space made order 5 cost 247 GB.  Its correctness rests on:

  G-LAW  the zero-parameter coefficient law on the pdaggerq streams:
         per exact-iso class, sum_i coeff_i*parity_i
         == sigma_ket * (1/2)^n_ep * n_ext  (no violations, no null classes);
  G-SET  enumerated iso-class sets == stream iso-class sets;
  G-VAL  BlockEvaluator fields from enumerated skeletons == fields from
         the stream skeletons at complex omega on H4, machine precision.

All three at orders 2-4, every external block.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

from src.Base.pdaggerq_interface import pdaggerq_available

needs_pdaggerq = pytest.mark.skipif(not pdaggerq_available(),
                                    reason='the pdaggerq stream is optional')

from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.CodeGeneration.diagrammatic.goldstone import (
    _BLOCKS, BlockEvaluator, build_skeletons)
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    _canonical, arrangement_parity, enumerate_skeletons)

ORDERS = (2, 3, 4)


@lru_cache(maxsize=None)
def _h4():
    mol, mf = EO.h4_chain()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    return eps, np.asarray(g), nocc


def _canon_of_skel(skel):
    m = {skel.bra_vertex: 0, skel.ket_vertex: 1}
    nxt = 2
    for v in range(skel.order):
        if v not in m:
            m[v] = nxt
            nxt += 1
    return _canonical(skel.order,
                      tuple((m[ln.src], m[ln.dst], ln.kind)
                            for ln in skel.lines))


def _n_equiv_pairs(skel):
    g = defaultdict(int)
    for ln in skel.lines:
        g[(ln.src, ln.dst, ln.kind)] += 1
    return sum(1 for mlt in g.values() if mlt == 2)


@needs_pdaggerq
@pytest.mark.parametrize('order', ORDERS)
def test_coefficient_law_on_stream(order):
    for block in _BLOCKS:
        skels, _ = build_skeletons(order, block)
        sigma = 1.0 if block[1] == 'v' else -1.0
        classes = defaultdict(list)
        for s in skels:
            classes[_canon_of_skel(s)].append(s)
        for canon, members in classes.items():
            m0 = members[0]
            T = sum(m.coeff * arrangement_parity(m.factors, m.lines,
                                                 m.bra_label, m.ket_label)
                    for m in members)
            pred = sigma * m0.n_ext * float(Fraction(1, 2 ** _n_equiv_pairs(m0)))
            assert abs(T - pred) < 1e-10, (order, block, canon)


def test_published_energy_diagram_counts():
    """Literature cross-gate (Li thesis Tab. 5.1 / Li-Tichai-Schwenk-
    Smirnova 2026): non-bubble two-body ground-state-energy diagram counts
    1, 3, 39, 840 at orders 2-5 == labeled connected 2-in/2-out arrow
    multigraphs (each is one time-ordered diagram).  Validates the
    enumeration core at order 5 independently of pdaggerq."""
    from src.CodeGeneration.diagrammatic.skeleton_enum import (
        _closure)

    def labeled_arrow_count(n):
        out_cap, in_cap = [2] * n, [2] * n
        symbols = [(s, d) for s in range(n) for d in range(n) if s != d]
        last = {s: i for i, (s, d) in enumerate(symbols)}
        edges = []
        cnt = 0

        def connected():
            adj = [0] * n
            for s, d in edges:
                adj[s] |= 1 << d
                adj[d] |= 1 << s
            return _closure(adj, 1, n) == (1 << n) - 1

        def rec(i):
            nonlocal cnt
            if i == len(symbols):
                if not any(out_cap) and not any(in_cap) and connected():
                    cnt += 1
                return
            s, d = symbols[i]
            for m in range(min(out_cap[s], in_cap[d], 2), -1, -1):
                out_cap[s] -= m
                in_cap[d] -= m
                edges.extend([symbols[i]] * m)
                if not (last.get(s) == i and out_cap[s] > 0):
                    rec(i + 1)
                del edges[len(edges) - m:]
                out_cap[s] += m
                in_cap[d] += m

        rec(0)
        return cnt

    assert [labeled_arrow_count(n) for n in (2, 3, 4, 5)] == [1, 3, 39, 840]


def test_kept_class_census_pins():
    """Frozen kept-class counts of the typed self-energy enumeration
    (block-independent); orders 2-4 are pdaggerq-gated elsewhere, order 5
    pins the production regime."""
    from src.CodeGeneration.diagrammatic.skeleton_enum import (
        kept_topologies)
    assert [len(kept_topologies(n)) for n in (2, 3, 4, 5)] == [2, 12, 170,
                                                               3006]


@needs_pdaggerq
@pytest.mark.parametrize('order', ORDERS)
def test_class_sets_and_fields(order):
    eps, g, nocc = _h4()
    ev = BlockEvaluator(eps, g, nocc)
    oms = (0.41 + 0.77j, -1.13 + 0.6j)
    for block in _BLOCKS:
        stream, _ = build_skeletons(order, block)
        enum = enumerate_skeletons(order, block)
        assert ({_canon_of_skel(s) for s in enum}
                == {_canon_of_skel(s) for s in stream}), (order, block)
        for w in oms:
            a = ev.eval_block(enum, w)
            b = ev.eval_block(stream, w)
            assert np.max(np.abs(a - b)) < 1e-13, (order, block, w)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
