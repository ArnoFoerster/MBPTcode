"""Codegen script (run once, output committed): derives restricted
(spin-blocked, closed-shell RHF) CCSDT T-residual, Lambda-residual, and 1-RDM
equations with pdaggerq's spin-integration feature (pq.strings(spin_labels=...)).

Scope of this generator: T1/T2/T3 (full CCSDT) + Lambda1/Lambda2/Lambda3
(full Lambda-CCSDT) + the 1-RDM built from those.

pdaggerq's add_st_operator hard-truncates its automatic BCH expansion at
quadruple commutators (see its own paper, arXiv:2501.08882), which is exact
for a <=2-body target (plain H) but silently drops every T3-dependent
diagram once a rank-3 projector is folded into the same target list, as
needed for the Lambda-commutator construction at rank 3. Confirmed by
reading pdaggerq's C++ source (pq_helper.cc::get_st_operator_terms): for
`ops` of length dim, add_st_operator enumerates, at each order n=0..4, every
non-decreasing multi-index combination of size n from range(dim) (i.e.
itertools.combinations_with_replacement), with prefactor
1/(m0! * m1! * ...) where mi is the multiplicity of ops[i] in that
combination, calling get_double/triple/quadruple_commutator_terms per
combination -- a truncated multinomial BCH expansion, exact only up to
order 4. `_add_order_n_commutator` below replicates that exact pattern at
order 5 (and, as a symbolic zero-check, order 6) via pdaggerq's
add_quintuple_commutator/add_hextuple_commutator, which are NOT wrapped by
any add_st_operator-like convenience function. Order 5 is needed (and
sufficient) only when the projector itself is rank 3 (Lambda3): a rank-3
projector combined with a <=2-body H (f or v) is already 5-body, so 5th
order is the first order at which T3-dependent terms get dropped by
add_st_operator's own 4th-order cap; order 6 is verified (not merely
assumed) to vanish identically for this target/ops combination -- see
`_check_order6_vanishes` in this module.

Spin-block storage convention (closed-shell Ms=0 symmetry -- see
src/SingleReference/CC/restricted_solver.py):
  T1/L1:  only '_aa' stored ('_bb' is numerically identical, not generated)
  T2/L2:  '_aaaa' and '_abab' stored ('_bbbb' == '_aaaa')
  T3/L3:  '_aaaaaa', '_aabaab', '_abbabb' stored ('_bbbbbb' == '_aaaaaa')
  D1:     only '_aa' block per sector (oo/vv/ov/vo) -- off-diagonal spin
          blocks of a closed-shell 1-RDM are exactly zero, and the final
          spatial density is 2 * this block (alpha + numerically-identical
          beta).

Bra/ket conventions:
  T-residual: bra = e_n(occ..., vir...REVERSED) as an EXTERNAL projector
    (set_left_operators), target = H (f, v) ALONE -- exactly the pattern
    pdaggerq's own official, tested examples/ccsdt_with_spin.py uses. Never
    hits the BCH-truncation issue since the target stays <=2-body regardless
    of the projector's rank.
  Lambda-residual: commutator [Hbar, e_projector] built by folding the
    projector into add_st_operator's target list, taken verbatim from
    pdaggerq's own official examples/lambda_singles.py / lambda_doubles.py,
    PLUS (rank 3 only) the hand-built order-5 correction terms described
    above, added alongside every existing add_st_operator call with the
    same coefficient/targets. Validated when written: reproduces this repo's
    checked-in lambda1_residual.py/lambda2_residual.py to ~1e-9 relative
    (float-roundoff level) on random amplitudes for ranks 1/2.
  Rank-3 (Lambda3) projector: 'e3(d,e,f,n,m,l)' -- vir d,e,f in normal
    (alphabetical) order, occ n,m,l in REVERSED (alphabetical) order,
    pairing (d,l),(e,m),(f,n) -- direct analogue of LAMBDA_PROJ[2]'s
    'e2(e,f,n,m)' (vir normal e,f / occ reversed n,m, pairing (e,m),(f,n))
    and of T_BRA[3]'s 'e3(i,j,k,c,b,a)' (occ normal i,j,k / vir reversed
    c,b,a, pairing (i,a),(j,b),(k,c)). Letters chosen from pdaggerq's own
    safe sets (PDAGGERQ_SAFE_VIR = a-f, PDAGGERQ_SAFE_OCC = i-n in
    src/CodeGeneration/pdaggerq_interface.py) -- NOT g/o as a naive continuation of the
    rank-2 letters might suggest; g is outside the safe virtual set and o is
    outside the safe occupied set, so d (vir) and l (occ) are used instead.
  D1 blocks: insert_op = e1(...) alone (rank 1, always exact), same pattern
    as generate_eom_ccsdt.py's density generation, with left = [1, l1, l2,
    l3].
"""
import math
import os
from itertools import combinations_with_replacement

from src.CodeGeneration.pdaggerq_interface import pq_helper, assemble_function, VIR_LETTERS
from src.CodeGeneration.codegen_common import (
    CC_OUT_DIR, DENSITY_BLOCKS, EINSUM_IMPORTS, density_insert, density_out_idx, density_shape,
    density_spin_labels, generated_header, write_module)

OUT_DIR = os.path.join(CC_OUT_DIR, 'generated_restricted')


def _spin_grouped_order(spin_labels, type_order=('vir', 'occ')):
    """Canonical output axis order for a spin-blocked equation: for each type
    in type_order, its letters grouped by spin ascending (alpha before
    beta), stable within a spin group (ties broken by spin_labels' own
    insertion order, i.e. the bra's own listed argument order).

    Deliberately NOT inferred from pdaggerq's raw pq.strings() term order
    (via infer_free_indices/infer_output_order) -- that order is an
    accident of pdaggerq's internal Wick-contraction bookkeeping for
    whichever term happens to be scanned, not a real convention. Confirmed
    the hard way: it silently produced a T3 aab-block residual with the
    unique-spin occupied index in the wrong slot (caught only because T3's
    required antisymmetry gives something to numerically check -- T1/T2/L1/L2
    would have failed with no structural check to catch it). This directly
    computed order is instead a FIXED choice from spin_labels alone, so every
    separate generation call for T1/T2/T3/L1/L2 uses a mutually consistent
    convention regardless of whether any given equation happens to
    self-reference its own leaf tensor.
    """
    groups = {'vir': [], 'occ': []}
    for letter in spin_labels:
        groups['vir' if letter in VIR_LETTERS else 'occ'].append(letter)
    order = []
    for t in type_order:
        order.extend(sorted(groups[t], key=lambda l: spin_labels[l]))
    return order

# Every '_bb'/'_bbbb'/'_bbbbbb' name below is a genuine argument the generated
# equations reference internally (pdaggerq's own spin-summed Wick contraction
# produces terms with a same-spin-but-beta leaf, e.g. a bb-loop diagram) --
# NOT something under our control. For a closed-shell reference these are
# numerically IDENTICAL to their '_aa'/'_aaaa'/'_aaaaaa' counterparts (see
# generate_restricted_ccsdt.py's module docstring), so restricted_solver.py
# passes the same array in twice at call time rather than storing/solving a
# redundant duplicate.
T_ARGS = ['t1_aa', 't1_bb', 't2_aaaa', 't2_abab', 't2_bbbb',
         't3_aaaaaa', 't3_aabaab', 't3_abbabb', 't3_bbbbbb',
         'f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'o', 'v']
L_ARGS = T_ARGS + ['l1_aa', 'l1_bb', 'l2_aaaa', 'l2_abab', 'l2_bbbb',
                   'l3_aaaaaa', 'l3_aabaab', 'l3_abbabb', 'l3_bbbbbb']
D_ARGS = L_ARGS + ['d_aa']

T_BRA = {
    1: ('e1(i,a)', {'aa': {'i': 'a', 'a': 'a'}}),
    2: ('e2(i,j,b,a)', {
        'aaaa': {'i': 'a', 'j': 'a', 'a': 'a', 'b': 'a'},
        'abab': {'i': 'a', 'j': 'b', 'a': 'a', 'b': 'b'},
    }),
    3: ('e3(i,j,k,c,b,a)', {
        'aaaaaa': {'i': 'a', 'j': 'a', 'k': 'a', 'a': 'a', 'b': 'a', 'c': 'a'},
        'aabaab': {'i': 'a', 'j': 'a', 'k': 'b', 'a': 'a', 'b': 'a', 'c': 'b'},
        'abbabb': {'i': 'a', 'j': 'b', 'k': 'b', 'a': 'a', 'b': 'b', 'c': 'b'},
    }),
}

LAMBDA_PROJ = {
    1: ('e1(e,m)', {'aa': {'e': 'a', 'm': 'a'}}),
    2: ('e2(e,f,n,m)', {
        'aaaa': {'e': 'a', 'f': 'a', 'n': 'a', 'm': 'a'},
        'abab': {'e': 'a', 'f': 'b', 'n': 'b', 'm': 'a'},
    }),
    # vir d,e,f normal order; occ n,m,l REVERSED order; pairing (d,l),(e,m),
    # (f,n) -- see module docstring for why d/l (not g/o, which are outside
    # pdaggerq's safe index-letter sets) are used here.
    3: ('e3(d,e,f,n,m,l)', {
        'aaaaaa': {'d': 'a', 'e': 'a', 'f': 'a', 'l': 'a', 'm': 'a', 'n': 'a'},
        'aabaab': {'d': 'a', 'e': 'a', 'f': 'b', 'l': 'a', 'm': 'a', 'n': 'b'},
        'abbabb': {'d': 'a', 'e': 'b', 'f': 'b', 'l': 'a', 'm': 'b', 'n': 'b'},
    }),
}



def _t_residual_terms(bra, spin_labels):
    pq = pq_helper('fermi')
    pq.set_left_operators([[bra]])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2', 't3'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def _add_order_n_commutator(pq, order, coeff, targets, ops):
    """Replicate pdaggerq's add_st_operator commuting-operator multinomial
    BCH pattern (pq_helper.cc::get_st_operator_terms, do_operators_commute=
    True branch) at `order` (5 or 6) -- an order add_st_operator itself
    never reaches (hard 4th-order truncation). For every non-decreasing
    combination of `order` indices into range(len(ops))
    (itertools.combinations_with_replacement, matching the C++ enumeration
    exactly), add an order-`order` commutator scaled by
    coeff / (m0! * m1! * ...), mi = multiplicity of ops[i] in the
    combination -- the same multinomial coefficient the C++ pattern uses at
    orders 2-4. `order` selects add_quintuple_commutator (5 op-slots after
    targets) or add_hextuple_commutator (6 op-slots after targets); each
    slot takes a length-1 list, e.g. ['t2'].
    """
    dim = len(ops)
    add_fn = {5: pq.add_quintuple_commutator, 6: pq.add_hextuple_commutator}[order]
    for combo in combinations_with_replacement(range(dim), order):
        mult = [combo.count(i) for i in range(dim)]
        prefactor = coeff
        for m in mult:
            prefactor /= math.factorial(m)
        op_args = [[ops[i]] for i in combo]
        add_fn(prefactor, targets, *op_args)


def _lambda_residual_terms(proj, spin_labels, l_ops=('l1', 'l2', 'l3'), rank=None):
    """rank selects whether the hand-built order-5 BCH correction (see
    _add_order_n_commutator) is needed: only when `proj` is itself a rank-3
    (Lambda3) projector does folding it into add_st_operator's target list
    make the combined target >2-body, which is where add_st_operator's own
    4th-order truncation starts silently dropping T3-dependent diagrams.
    For rank 1/2 this is plain add_st_operator -- l_ops defaults to
    including l3 (for Lambda3 feedback into Lambda1/Lambda2), which changes
    nothing structurally since l3 simply may or may not appear as a leaf in
    the resulting terms depending on what the diagrams actually connect to.
    """
    # Overall sign of the rank-3 (Lambda3) projector convention: rank 1/2
    # here are taken verbatim from pdaggerq's own official
    # examples/lambda_singles.py / lambda_doubles.py (no ambiguity). Rank 3
    # has no official template -- the 'e3(d,e,f,n,m,l)' bra with pairing
    # (d,l),(e,m),(f,n) was extended by direct analogy to LAMBDA_PROJ[2]'s
    # 'e2(e,f,n,m)'. Cross-checked numerically against the trusted
    # generalized lambda3_residual.py on random spin-consistent amplitudes
    # (see tests/test_ccsdt_lambda.py-style validation harness): the
    # analogy reproduces the correct residual up to a clean, uniform
    # overall factor of -1 (max abs diff ~1e-13 after negating), consistent
    # with an odd-vs-even-rank fermion-reordering sign picked up by the
    # 3-pair (d,l),(e,m),(f,n) bra construction that a 2-pair bra doesn't
    # see. Corrected here by flipping the sign of every proj-involving term
    # for rank==3 only; rank 1/2 are untouched (s stays +1).
    s = -1.0 if rank == 3 else 1.0

    pq = pq_helper('fermi')
    pq.set_left_operators([['1']])
    pq.set_right_operators([['1']])
    pq.add_st_operator(s * 1.0, ['f', proj], ['t1', 't2', 't3'])
    pq.add_st_operator(s * 1.0, ['v', proj], ['t1', 't2', 't3'])
    if rank == 3:
        _add_order_n_commutator(pq, 5, s * 1.0, ['f', proj], ['t1', 't2', 't3'])
        _add_order_n_commutator(pq, 5, s * 1.0, ['v', proj], ['t1', 't2', 't3'])
    pq.set_left_operators([[lop] for lop in l_ops])
    pq.add_st_operator(s * 1.0, ['f', proj], ['t1', 't2', 't3'])
    pq.add_st_operator(s * 1.0, ['v', proj], ['t1', 't2', 't3'])
    pq.add_st_operator(s * -1.0, [proj, 'f'], ['t1', 't2', 't3'])
    pq.add_st_operator(s * -1.0, [proj, 'v'], ['t1', 't2', 't3'])
    if rank == 3:
        _add_order_n_commutator(pq, 5, s * 1.0, ['f', proj], ['t1', 't2', 't3'])
        _add_order_n_commutator(pq, 5, s * 1.0, ['v', proj], ['t1', 't2', 't3'])
        _add_order_n_commutator(pq, 5, s * -1.0, [proj, 'f'], ['t1', 't2', 't3'])
        _add_order_n_commutator(pq, 5, s * -1.0, [proj, 'v'], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def _check_order6_vanishes(proj, spin_labels, l_ops=('l1', 'l2', 'l3')):
    """Safety check (not part of the production equations): a rank-3
    projector combined with a <=2-body H (f or v) and T1/T2/T3 is at most
    5-body, so the 6th-order (hextuple) commutator against T must vanish
    identically by Wick's theorem -- confirmed here symbolically (0 terms
    after pq.simplify()) rather than merely assumed. Mirrors the same four
    add_st_operator call sites in _lambda_residual_terms's rank==3 branch,
    at order 6 instead of 5.
    """
    pq = pq_helper('fermi')
    pq.set_left_operators([[lop] for lop in l_ops])
    pq.set_right_operators([['1']])
    _add_order_n_commutator(pq, 6, 1.0, ['f', proj], ['t1', 't2', 't3'])
    _add_order_n_commutator(pq, 6, 1.0, ['v', proj], ['t1', 't2', 't3'])
    _add_order_n_commutator(pq, 6, -1.0, [proj, 'f'], ['t1', 't2', 't3'])
    _add_order_n_commutator(pq, 6, -1.0, [proj, 'v'], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def _density_terms(insert_op, spin_labels, l_ops=('l1', 'l2', 'l3')):
    pq = pq_helper('fermi')
    pq.set_left_operators([['1']] + [[lop] for lop in l_ops])
    pq.set_right_operators([['1']])
    pq.add_st_operator(1.0, [insert_op], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def generate_t_residuals():
    funcs = []
    for rank, (bra, blocks) in T_BRA.items():
        leaf = f't{rank}'
        for block_name, spin_labels in blocks.items():
            terms = _t_residual_terms(bra, spin_labels)
            out_idx = _spin_grouped_order(spin_labels, type_order=('vir', 'occ'))
            shape = '(' + ', '.join(['nv'] * rank + ['no'] * rank) + ')'
            print(f"t{rank}_{block_name}: {len(terms)} terms, out_indices={out_idx}")
            funcs.append(assemble_function(f't{rank}_{block_name}_residual', T_ARGS,
                                           f'{leaf}_res', shape, terms, out_idx,
                                           shape_source='t1_aa'))
    return funcs


def generate_lambda_residuals():
    """Same infer_free_indices-not-infer_output_order reasoning as
    generate_t_residuals applies here too: l1_aa/l2_aaaa/l2_abab's
    cross-consistency requirement (each is fed as an INPUT into every OTHER
    Lambda-residual's equation, and into the D1 blocks) is a
    cross-generation-call constraint that a single equation's own raw-term
    leaf-scan cannot see or guarantee -- only a fixed, bra-position-based
    convention (identical bra string 'e2(e,f,n,m)' for every block) can.
    """
    funcs = []
    for rank, (proj, blocks) in LAMBDA_PROJ.items():
        leaf = f'l{rank}'
        for block, spin_labels in blocks.items():
            terms = _lambda_residual_terms(proj, spin_labels, rank=rank)
            out_idx = _spin_grouped_order(spin_labels, type_order=('vir', 'occ'))
            shape = '(' + ', '.join(['nv'] * rank + ['no'] * rank) + ')'
            print(f"l{rank}_{block}: {len(terms)} terms, out_indices={out_idx}")
            funcs.append(assemble_function(f'l{rank}_{block}_residual', L_ARGS,
                                           f'{leaf}_res', shape, terms, out_idx,
                                           shape_source='t1_aa'))
    return funcs


def generate_density():
    funcs = []
    for block in DENSITY_BLOCKS:
        # the alpha block (off-diagonal spin blocks vanish); out_indices are the
        # insertion's own argument order
        terms = _density_terms(density_insert(block), density_spin_labels(block, 'a'))
        out_idx = density_out_idx(block)
        print(f"d1_{block}_aa: {len(terms)} terms, out_indices={out_idx}")
        funcs.append(assemble_function(f'd1_{block}_aa', D_ARGS, f'd1_{block}',
                                       density_shape(block), terms, out_idx,
                                       shape_source='t1_aa'))
    return funcs


def _energy_terms():
    """CCSDT correlation energy: T3 never appears (a 2-body H cannot connect
    the reference to a triple excitation by Slater-Condon rules, so <0|H
    T3|0> is identically zero) -- same reason amplitudes.py::cc_energy takes
    no t3 argument."""
    pq = pq_helper('fermi')
    pq.set_left_operators([['1']])
    pq.set_right_operators([['1']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    terms = pq.strings(spin_labels={})
    pq.clear()
    return terms


def generate_energy():
    terms = _energy_terms()
    print(f"cc_energy_restricted: {len(terms)} terms")
    func = assemble_function('cc_energy_restricted',
                             ['t1_aa', 't1_bb', 't2_aaaa', 't2_abab', 't2_bbbb',
                              'f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'o', 'v'],
                             'energy', '()', terms, [], shape_source='t1_aa')
    return [func]


MKL_THREADING_PREAMBLE = (
    "# allow numpy built with MKL to consume more threads for tensordot -- but only\n"
    "# if the caller hasn't already pinned the thread count (same convention as\n"
    "# amplitudes.py's generalized-pipeline counterpart)\n"
    "import os\n"
    "os.environ.setdefault(\"MKL_NUM_THREADS\", \"{}\".format(os.cpu_count() - 1))\n\n"
)


def _write(fname, funcs, *notes, mkl_threading=False):
    write_module(os.path.join(OUT_DIR, fname), funcs,
                 generated_header('generate_restricted_ccsdt.py', *notes,
                                  "See that module's docstring for the equations/conventions."),
                 imports=(MKL_THREADING_PREAMBLE if mkl_threading else '') + EINSUM_IMPORTS,
                 package_init=True)


def generate():
    """Write amplitudes_restricted.py, lambda_residual_restricted.py and
    d1_blocks_restricted.py into OUT_DIR, after checking that the order-6
    correction of the rank-3 Lambda projector vanishes."""
    # Safety check first (cheap, symbolic): the order-6 (hextuple) commutator
    # correction for the rank-3 Lambda projector must vanish identically --
    # verify this rather than merely assume it before generating/trusting
    # the order-5 production equations below.
    proj3, blocks3 = LAMBDA_PROJ[3]
    order6_terms = _check_order6_vanishes(proj3, blocks3['aaaaaa'])
    if len(order6_terms) != 0:
        raise RuntimeError(
            f"order-6 (hextuple) correction for the rank-3 Lambda projector "
            f"did NOT vanish symbolically ({len(order6_terms)} terms survived "
            f"pq.simplify()) -- the '5-body combined, so 6th order vanishes' "
            f"physical argument was wrong somewhere; investigate before "
            f"trusting the order-5 equations below.")
    print(f"order-6 (hextuple) correction for rank-3 Lambda projector: "
          f"{len(order6_terms)} terms (vanishes, as expected) -- OK")

    t_funcs = generate_energy() + generate_t_residuals()
    _write('amplitudes_restricted.py', t_funcs,
           'Restricted (spin-blocked) CCSDT energy and T-residuals.', mkl_threading=True)

    l_funcs = generate_lambda_residuals()
    _write('lambda_residual_restricted.py', l_funcs,
           'Restricted (spin-blocked) Lambda-CCSDT residuals (Lambda1/Lambda2/Lambda3).',
           mkl_threading=True)

    d_funcs = generate_density()
    _write('d1_blocks_restricted.py', d_funcs,
           'Restricted 1-RDM oo/vv/ov/vo blocks, _aa only (off-diagonal spin blocks vanish; '
           'the spatial density is', '2x this block), with the full Lambda1/Lambda2/Lambda3 feedback.')


if __name__ == '__main__':
    generate()
