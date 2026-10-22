"""generateADC(order) -- the master-formula-driven ADC(n) generation driver.

The design target: ``generateADC(order=n)`` shall
  (a) use the master formula (block_order(n, mu_i, mu_j) = n - mu_i - mu_j,
      class mu present iff 2*mu <= n) to decide WHICH blocks are needed,
  (b) derive every needed block via the ISR + pdaggerq machinery,
  (c) emit the spin-free (restricted) twins,
  (d) wire in DF and Laplace (rule: NO materialized intermediate may carry
      more than 4 indices => every rank>=3 amplitude is Laplace-fused),
  (e) emit everything matrix-free (apply form),
  (f) wire the even-order closed-form downfold + diagonal approximation,
  (g) package as a callable the production solve entry can consume,
  incrementally: ADC(n) generates only what ADC(n-1) did not already emit.

THIS MODULE is the driver skeleton: plan (a) is fully implemented and
order-general; (b,c,d,e) dispatch to the existing per-family generators via a
REGISTRY, skipping module files that already exist (the generatedADC/ files
ARE the incremental cache); everything the registry cannot yet produce is
reported as a PRECISE gap, never silently skipped (the l3f lesson: a missing
module is indistinguishable from a pruned one at runtime unless generation
asserts coverage).

Current state:
  * plan/coverage/dispatch: order-general, this file.
  * per-family generators: order-general for (0,1) [generate_adc_u_restricted];
    per-order functions for the rest (general emitters are Phase 1 work).
  * runtime assembly: hand-wired per order up to ADC(4)+ADC(5)-blocks; the
    order-general series assembly (sequential class-GS + Loewdin convolutions
    mirroring validation/lambda_oracle/exact_isr_order5.py) is Phase 2.
  * amplitudes: numerator generator is order-general
    (t_numerator_terms_restricted(rank, order, block) works at any order);
    the recursion driver is wired through order 3 (+t1_4) -- Phase 1 wires
    the general loop.
"""
import ast
import importlib
import os
import re

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.SingleReference.ADC.adc_order_rule import (
    kmax, block_order, blocks_at, u_order_terms, s_order_terms,
    amplitude_rank_surviving, cross_blocks_at, cross_block_leading_order,
    refuse_above_max_order)
from src.SingleReference.ADC.precursor_classes import nondyson_module_name
from src.SingleReference.ADC.generated_modules import (RUNTIME_MODULES, is_generated,
                                                    dotted as generated_dotted)

_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'SingleReference', 'ADC', 'generatedADC')
_MPN_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'SingleReference', 'DensityMatrix',
                            'generated_mpn')


def _module_path(mod):
    """Path to a generatedADC module, mirroring generate_adcn._module_subdir
    (adc{N}_....py lives in generatedADC/adc{N}/, never the flat root --
    this used to be a flat lookup and reported every already-generated module
    as a gap)."""
    m = re.match(r'adc(\d+)_', mod)
    if not m:
        raise ValueError(f"cannot derive an adc{{N}} subpackage from module "
                         f"name {mod!r} (expected an 'adc<N>_...' prefix)")
    return os.path.join(_GEN_DIR, f'adc{m.group(1)}', mod + '.py')


def _amp_census():
    """{order: ranks or None} of generated amplitude numerators. Orders 1-3
    are the legacy mpn_density_pieces chain (census pinned to match
    ranks_present at its generation); higher orders come from the
    RANKS_PRESENT manifests of generated_mpn/mpn_amp_numerators_o*.py --
    parsed TEXTUALLY (ast.literal_eval) so this driver never imports
    SingleReference modules (repo convention). A rank absent from a listed
    order's census is pdaggerq-PRUNED, not a gap."""
    census = {1: [2], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4]}
    if os.path.isdir(_MPN_GEN_DIR):
        for fn in sorted(os.listdir(_MPN_GEN_DIR)):
            if not (fn.startswith('mpn_amp_numerators_') and
                    fn.endswith('.py')):
                continue
            with open(os.path.join(_MPN_GEN_DIR, fn)) as fh:
                for line in fh:
                    if line.startswith('RANKS_PRESENT'):
                        census.update(ast.literal_eval(
                            line.split('=', 1)[1].strip()))
                        break
    return census


# ---------------------------------------------------------------------------
# REGISTRY: (family, order) -> what implements it today.
#   family keys: U_k{mu}   orbital<->class-mu coupling at a given order
#                C_k{mu}   class-mu self block at a given order
#                C_{i}_{j} cross-class coupling (i<j) at a given order
#                S_{i}_{j} cross-class overlap (ISR input) -- NEVER visited by
#                          plan()/coverage() (_family() has no S_ branch);
#                          kept for documentation, dispatched today only via
#                          whichever REAL family's 'calls' also produces its
#                          module (see ('U_k2',3) below).
#   'modules': generated module basenames that must exist for coverage.
#   'calls': ordered list of (dotted_callable, kwargs) pairs that together
#            produce every listed module -- the real dispatch metadata
#            GenerateADC executes. Resolved lazily (importlib) and deduped by
#            (callable identity, frozen kwargs) across the whole REGISTRY, so
#            a producer shared by two keys (e.g. generate_adc4_base_k2 under
#            both ('C_1_2',1) and ('U_k2',2)) runs exactly once per
#            GenerateADC pass. Every entry here was confirmed by reading the
#            real producer's _write_module/_write_apply_module call and
#            matching its output filename against 'modules' -- nothing here
#            is a guess (2026-07-24 GenerateADC audit).
#   'generator': legacy display string (coverage()'s printout only).
#   'note': honest one-liner (what representation, what is missing).
# ---------------------------------------------------------------------------
REGISTRY = {
    ('U_k1', 1): dict(modules=['adc2_blocks', 'adc2_blocks_restricted',
                               'adc2_blocks_restricted_df'],
                      generator='generate_adcn.generate_adc2', note='U^(1)',
                      calls=[('generate_adcn.generate_adc2', {}),
                             ('generate_adcn.generate_adc2_restricted', {}),
                             ('generate_adcn.generate_adc2_restricted',
                              {'df': True})]),
    # the _df twins are not optional extras: the g-free DF ADC(3) route
    # (ADCSolverRestricted with df=True + amplitudes=, the only one that fits
    # aug-cc-pVQZ) builds EVERY block from them.
    ('U_k1', 2): dict(modules=['adc3_u_blocks', 'adc3_u_blocks_restricted',
                               'adc3_u_blocks_restricted_df'],
                      generator='generate_adcn.generate_adc3_u', note='U^(2)+S^(2)',
                      calls=[('generate_adcn.generate_adc3_u', {}),
                             ('generate_adcn.generate_adc3_u_restricted',
                              {'df': False}),
                             ('generate_adcn.generate_adc3_u_restricted',
                              {'df': True})]),
    ('U_k1', 3): dict(modules=['adc4_u3_blocks', 'adc4_u3_blocks_restricted'],
                      generator='generate_adcn.generate_adc4_u3',
                      note='U^(3)+S^(3); laplace l3 twin exists',
                      calls=[('generate_adcn.generate_adc4_u3', {}),
                             ('generate_adcn.generate_adc4_u3_restricted',
                              {'df': False})]),
    ('U_k1', 4): dict(modules=['adc5_u4_blocks_restricted',
                               'adc5_u4_blocks_restricted_df'],
                      generator='generate_adcn.generate_adc_u_restricted',
                      note='ORDER-GENERAL generator (u_order arg); restricted+DF '
                           '(the _df twin is sigma_codegen.generate_adc5_u4_'
                           'restricted_df -- generate_adc_u_restricted has no '
                           'df flag)',
                      calls=[('generate_adcn.generate_adc_u_restricted',
                              {'u_order': 4}),
                             ('sigma_codegen.generate_adc5_u4_restricted_df', {})]),
    ('C_k1', 0): dict(modules=['adc2_blocks'], generator='generate_adcn.generate_adc2',
                      note='K (f part)',
                      calls=[('generate_adcn.generate_adc2', {})]),
    ('C_k1', 1): dict(modules=['adc2_blocks'], generator='generate_adcn.generate_adc2',
                      note='C^(1) (v part)',
                      calls=[('generate_adcn.generate_adc2', {})]),
    ('C_k1', 2): dict(modules=['adc4_c2_blocks', 'adc4_c2_blocks_restricted'],
                      generator='generate_adcn.generate_adc4_base_c2',
                      note='C^(2)+s2c; apply+df twins exist',
                      calls=[('generate_adcn.generate_adc4_base_c2', {}),
                             ('generate_adcn.generate_adc4_base_c2_restricted', {})]),
    ('C_k1', 3): dict(modules=['adc5_c3_blocks', 'adc5_c3_blocks_restricted',
                               'adc5_c3_apply_restricted_blocks'],
                      generator='generate_adcn.generate_adc5_c3',
                      note='C^(3); runtime adds orbital-GS cross (n=3 only); '
                           'apply twin is sigma_codegen.generate_adc5_c3_'
                           'apply_restricted',
                      calls=[('generate_adcn.generate_adc5_c3', {}),
                             ('generate_adcn.generate_adc5_c3_restricted', {}),
                             ('sigma_codegen.generate_adc5_c3_apply_restricted',
                              {'df': False})]),
    ('C_1_2', 1): dict(modules=['adc4_k2_blocks', 'adc4_k2_blocks_restricted'],
                      generator='generate_adcn.generate_adc4_base_k2',
                      note='k1<->k2 C^(1) (+ U^(2)_k2/S^(2)_k2 legs)',
                      calls=[('generate_adcn.generate_adc4_base_k2', {}),
                             ('generate_adcn.generate_adc4_base_k2_restricted', {})]),
    ('C_1_2', 2): dict(modules=['adc5_c12_blocks'],
                      generator='generate_adcn.generate_adc5_c12',
                      note='k1<->k2 order-2 (l1f/l2v/l2ft2) + runtime '
                           'c12_o2_generated (sequential class-GS assembly, '
                           'arbiter 6e-10/6e-9) -- the gap generateADC(5) '
                           'found, closed 2026-07-21.',
                      calls=[('generate_adcn.generate_adc5_c12', {})]),
    ('U_k2', 2): dict(modules=['adc4_k2_blocks', 'adc4_k2_blocks_restricted'],
                      generator='generate_adcn.generate_adc4_base_k2',
                      note='U^(2)_k2 + S^(2)_k2 (module SHARED with '
                           "('C_1_2',1) -- same calls, deduped by GenerateADC)",
                      calls=[('generate_adcn.generate_adc4_base_k2', {}),
                             ('generate_adcn.generate_adc4_base_k2_restricted', {})]),
    ('U_k2', 3): dict(modules=['adc5_u3_k2_blocks', 'adc5_s_k1k2_blocks'],
                      generator='generate_adcn.generate_adc5_u3_k2',
                      note='U^(3)_k2 incl. S_{k1,k2} cross term (arbiter 1e-10); '
                           "adc5_s_k1k2_blocks is generate_adc5_s_k1k2 (also "
                           "listed under ('S_1_2',2), which plan()/coverage() "
                           "never visits -- this key is its only real dispatch "
                           "path)",
                      calls=[('generate_adcn.generate_adc5_u3_k2', {}),
                             ('generate_adcn.generate_adc5_s_k1k2', {})]),
    ('S_1_2', 2): dict(modules=['adc5_s_k1k2_blocks'],
                      generator='generate_adcn.generate_adc5_s_k1k2',
                      note='cross-class overlap, orders 2+3 in one module '
                           "(documentation only -- see ('U_k2',3))",
                      calls=[('generate_adcn.generate_adc5_s_k1k2', {})]),
    ('C_k2', 0): dict(modules=['adc5_k2k2_blocks'],
                      generator='generate_adcn.generate_adc5_k2k2',
                      note='K_k2 (diag; also runtime diag_K)',
                      calls=[('generate_adcn.generate_adc5_k2k2', {})]),
    ('C_k2', 1): dict(modules=['adc5_k2k2_blocks',
                               'adc5_k2k2_apply_restricted_blocks'],
                      generator='generate_adcn.generate_adc5_k2k2',
                      note='(2,2) C^(1): the ADC(5) block; every required property (apply twin '
                           'is sigma_codegen.generate_adc5_k2k2_apply_'
                           "restricted; dense call SHARED with ('C_k2',0))",
                      calls=[('generate_adcn.generate_adc5_k2k2', {}),
                             ('sigma_codegen.generate_adc5_k2k2_apply_restricted',
                              {'df': False})]),
    # ---- union-space D x C cross blocks (X_{kI}_{kJ}: (kI+1)h kI p-dagger bra,
    # (kJ+1)p kJ h ket), the Hamiltonian blocks the two-sector convention
    # deletes; needed by construction='union'/'diagrammatic' only, so the ones
    # not emitted are reported as union-only, never as gaps of the two-sector
    # production ----
    ('X_1_1', 2): dict(modules=['adc4_x2_k1k1_blocks', 'adc4_x2_k1k1_blocks_restricted',
                               'adc4_x2_k1k1_blocks_restricted_df',
                               'adc4_x2_k1k1_apply_restricted_blocks',
                               'adc4_x2_k1k1_apply_restricted_df_blocks'],
                      generator='generate_adcn.generate_adc_cross_pair',
                      note='X^(2) 2h1p x 2p1h (union-space ADC(4)); dense, restricted, '
                           'restricted DF, apply and apply DF twins',
                      union_only=True,
                      calls=[('generate_adcn.generate_adc_cross_pair',
                              {'order': 2, 'pair': 'k1k1'}),
                             ('generate_adcn.generate_adc_cross_pair_restricted',
                              {'order': 2, 'pair': 'k1k1', 'df': False}),
                             ('generate_adcn.generate_adc_cross_pair_restricted',
                              {'order': 2, 'pair': 'k1k1', 'df': True}),
                             ('sigma_codegen.generate_adc_cross_pair_apply_restricted',
                              {'order': 2, 'pair': 'k1k1', 'df': False}),
                             ('sigma_codegen.generate_adc_cross_pair_apply_restricted',
                              {'order': 2, 'pair': 'k1k1', 'df': True})]),
    ('X_1_1', 3): dict(modules=['adc5_x3_k1k1_blocks'], generator=None, union_only=True,
                      note='X^(3) 2h1p x 2p1h (union-space ADC(5)); not emitted'),
    ('X_1_2', 2): dict(modules=['adc5_x2_k1k2_blocks'], generator=None, union_only=True,
                      note='X^(2) 2h1p x 3p2h (union-space ADC(5)); not emitted, must be '
                           'matrix-free'),
    ('X_2_1', 2): dict(modules=['adc5_x2_k2k1_blocks'], generator=None, union_only=True,
                      note='X^(2) 3h2p x 2p1h (union-space ADC(5)); not emitted'),
}

# ---- non-Dyson (single-sector, ordinary-metric) ISR blocks, ND_{k_i}_{k_j} at
# a given order, both sectors in one module (generate_adcn.generate_nondyson_pair;
# the generated-ADC runtime). Emitted through ADC(5); listed by
# nondyson_plan()/nondyson_coverage(), never by the Dyson plan() ----
_NONDYSON_EMITTED = [(m, 0, 0) for m in range(6)] + [(m, 0, 1) for m in range(1, 5)] + \
    [(m, 1, 1) for m in range(4)] + [(m, 0, 2) for m in range(1, 4)] + \
    [(m, 1, 2) for m in range(1, 3)] + [(m, 2, 2) for m in range(2)]
for _m, _ki, _kj in _NONDYSON_EMITTED:
    REGISTRY[(f'ND_{_ki}_{_kj}', _m)] = dict(
        modules=[nondyson_module_name(_m, _ki, _kj)[:-3]],
        generator='generate_adcn.generate_nondyson_pair',
        note=f'non-Dyson class ({_ki},{_kj}) block at order {_m}, both sectors',
        nondyson=True,
        calls=[('generate_adcn.generate_nondyson_pair',
                {'order': _m, 'k_i': _ki, 'k_j': _kj})])


def nondyson_plan(order):
    """Block plan of non-Dyson ADC(order): every class pair (k_i <= k_j,
    2k <= order) with the orders it is carried through (the same manifold
    rule as the Dyson supermatrix; same-class blocks start at order 0,
    couplings at order 1)."""
    refuse_above_max_order(order, 'nondyson_plan')
    out = []
    for (i, j, o) in blocks_at(order):
        start = 0 if i == j else 1
        out.append(dict(i=i, j=j, family=f'ND_{i}_{j}', carried=o,
                        orders=list(range(start, o + 1))))
    return out


def nondyson_coverage(order, verbose=True):
    """(have, gaps) of the non-Dyson plan by existing generated modules."""
    have, gaps = [], []
    for blk in nondyson_plan(order):
        for m in blk['orders']:
            key = (blk['family'], m)
            reg = REGISTRY.get(key)
            if reg is None:
                gaps.append((key, 'NO REGISTRY ENTRY (never derived)'))
                continue
            missing = [mod for mod in reg['modules'] if not os.path.exists(_module_path(mod))]
            if missing:
                gaps.append((key, f"modules missing: {missing} (generator: {reg['generator']})"))
            else:
                have.append((key, reg['note']))
    if verbose:
        print(f"=== non-Dyson ADC({order}): {len(nondyson_plan(order))} blocks ===")
        for k, note in have:
            print(f"  [have] {str(k):16s} {note}")
        for k, why in gaps:
            print(f"  [GAP ] {str(k):16s} {why}")
    return have, gaps


def _family(i, j):
    if i == 0 and j == 0:
        return None                      # orbital block: runtime diag(eps)
    if i == 0:
        return f'U_k{j}'
    if i == j:
        return f'C_k{i}'
    return f'C_{i}_{j}'


def plan(order):
    """The master-formula block plan for ADC(order): every (family, block
    order, per-order entries 0..carried) with its registry key list."""
    refuse_above_max_order(order, 'plan')
    out = []
    for (i, j, o) in blocks_at(order):
        fam = _family(i, j)
        if fam is None:
            continue
        # a block carried to order o needs its content at EVERY order <= o
        # (couplings start at 1 for k=1 / 2 for k=2 / 3 for k=3; class blocks
        # at 0). Lower-order content lives in lower-order modules -- that is
        # the incremental-cache property.
        if fam.startswith('U_k'):
            mu = int(fam[3:])
            start = {1: 1, 2: 2, 3: 3}.get(mu, mu)
        else:
            start = 0 if (i == j) else 1
        orders = [m for m in range(start, o + 1)]
        out.append(dict(i=i, j=j, family=fam, carried=o, orders=orders))
    # union-space cross blocks: D_kI x C_kJ, carried from their leading order
    for (k_i, k_j, o) in cross_blocks_at(order):
        orders = list(range(cross_block_leading_order(k_i, k_j), o + 1))
        out.append(dict(i=k_i, j=k_j, family=f'X_{k_i}_{k_j}', carried=o, orders=orders,
                        union_only=True))
    return out


def amplitude_requirements(order):
    """(rank, mp-order) pairs any block of ADC(order) consumes, from the order
    rule (superset; pdaggerq pruning only removes)."""
    need = set()
    for blk in plan(order):
        for m in blk['orders']:
            for t in u_order_terms(m):
                _, jj, _, kk, a, b = t
                if jj:
                    need.add((jj, a))
                if kk:
                    need.add((kk, b))
            for t in s_order_terms(m):
                _, jj, kk, a, b = t
                if jj:
                    need.add((jj, a))
                if kk:
                    need.add((kk, b))
    return sorted(need)


def coverage(order, verbose=True):
    """Coverage of the plan by EXISTING generated modules + precise gaps."""
    gaps, have = [], []
    for blk in plan(order):
        for m in blk['orders']:
            key = (blk['family'], m)
            reg = REGISTRY.get(key)
            if reg is None:
                gaps.append((key, 'NO REGISTRY ENTRY (family/order never '
                                  'derived or catalogued)'))
                continue
            missing = [mod for mod in reg['modules']
                       if not os.path.exists(_module_path(mod))]
            if reg.get('runtime'):
                # satisfied by a runtime construction (e.g. diag_K) -- no
                # generated module is the CORRECT state, not a gap
                have.append((key, reg['note']))
            elif reg.get('union_only') and (reg['generator'] is None or missing):
                # a cross block of the union construction that is not emitted:
                # the two-sector production is complete without it
                have.append((key, '[union-only, not emitted] ' + reg['note']))
            elif reg['generator'] is None:
                gaps.append((key, f"no generator: {reg['note']}"))
            elif missing:
                gaps.append((key, f"modules missing: {missing} "
                                  f"(generator: {reg['generator']})"))
            else:
                have.append((key, reg['note']))
    # amplitude coverage: the recursion driver is ORDER-GENERAL
    # (compute_order_amplitudes_general); a gap is an order with no generated
    # numerator module, closable by one generate_amplitude_numerator_module
    # call. Rank>4 pairs are never materialized (the <=4-index rule) and are
    # reported separately.
    amp_need = amplitude_requirements(order)
    census = _amp_census()
    amp_gaps = [(r, o) for (r, o) in amp_need
                if r <= 4 and o not in census]
    rank_cap = [(r, o) for (r, o) in amp_need if r > 4]
    if verbose:
        print(f"=== generateADC({order}): kmax={kmax(order)}, "
              f"{len(plan(order))} blocks ===")
        for k, note in have:
            print(f"  [have] {str(k):16s} {note}")
        for k, why in gaps:
            print(f"  [GAP ] {str(k):16s} {why}")
        print(f"  amplitudes needed (rank,order): {amp_need}")
        if amp_gaps:
            orders_missing = sorted(set(o for _, o in amp_gaps))
            print(f"  [GAP ] amplitude numerators ungenerated for: {amp_gaps} "
                  f"(one generate_amplitude_numerator_module({orders_missing}, "
                  f"...) call; the recursion loop is order-general)")
        if rank_cap:
            print(f"  [rule] rank>4 amplitudes {sorted(set(r for r,_ in rank_cap))} "
                  f"must NEVER materialize (<=4-index rule): consume via "
                  f"Laplace-fused numerators only")
        surv = amplitude_rank_surviving(order)
        print(f"  surviving amplitude rank at ADC({order}): {surv} "
              f"(indices: {2*surv})")
    return have, gaps, amp_gaps


def _resolve_call(dotted):
    """Resolve 'module.func' (module in {generate_adcn, sigma_codegen, ...})
    to the real callable, importing lazily so generate_adc.py itself stays
    import-cheap (matches the module's existing 'dispatched lazily'
    convention)."""
    mod_name, func_name = dotted.rsplit('.', 1)
    mod = importlib.import_module(
        f'src.CodeGeneration.{mod_name}')
    return getattr(mod, func_name)


def _run_calls(calls, done):
    """Execute each (dotted_callable, kwargs) in `calls`, skipping any call
    whose (resolved callable, frozen kwargs) already ran THIS PASS -- the
    dedup that lets a producer shared by two REGISTRY keys (e.g.
    generate_adc4_base_k2 under both ('C_1_2',1) and ('U_k2',2)) run exactly
    once per GenerateADC pass instead of twice."""
    for dotted, kwargs in calls:
        func = _resolve_call(dotted)
        sig = (func, tuple(sorted(kwargs.items())))
        if sig in done:
            continue
        func(**kwargs)
        done.add(sig)


def generateADC(order, dry_run=True):
    """Walk the plan; (re)generate every missing module with a registered
    generator; report what cannot be generated yet. Incremental: existing
    module files are the cache and are skipped (coverage() re-run after
    dispatch confirms this). dry_run=True only reports (unchanged behavior).

    dry_run=False actually invokes the REGISTRY 'calls' for every gap that has
    them, then re-checks coverage: any gap still open after dispatch --
    because no 'calls' are registered for it or because a call ran but its
    module still didn't materialize
    -- is raised as a precise, itemized error. Never a silent no-op or a
    partial success reported as done (the project's standing 'never silently
    under-compute' rule)."""
    have, gaps, amp_gaps = coverage(order, verbose=False)
    actionable = [(k, g) for k, g in gaps if REGISTRY.get(k, {}).get('calls')]
    if dry_run:
        return dict(have=have, gaps=gaps, amp_gaps=amp_gaps,
                    actionable=actionable)

    done = set()
    for key, _ in gaps:
        reg = REGISTRY.get(key)
        if reg and reg.get('calls'):
            _run_calls(reg['calls'], done)

    if amp_gaps:
        from src.CodeGeneration.generate_mpn_density import (
            generate_amplitude_numerator_module)
        for o in sorted(set(o for _, o in amp_gaps)):
            generate_amplitude_numerator_module([o], f'mpn_amp_numerators_o{o}')

    have2, gaps2, amp_gaps2 = coverage(order, verbose=False)
    if gaps2 or amp_gaps2:
        lines = [f"  {k}: {w}" for k, w in gaps2] + (
            [f"  amplitude numerators still missing after dispatch: "
             f"{amp_gaps2}"] if amp_gaps2 else [])
        raise RuntimeError(
            f"generateADC({order}, dry_run=False): {len(gaps2) + bool(amp_gaps2)} "
            f"gap(s) could not be closed -- never silently under-computed:\n"
            + "\n".join(lines))
    return dict(have=have2, gaps=[], amp_gaps=[], actionable=[])


def GenerateADC(orders=(2, 3, 4, 5)):
    """One-shot setup entry point: generate every module production runs need
    at the given orders, so a fresh machine only needs to call this once
    before any solve_adc(...) run (generated code is not committed to git --
    see README.md). Incremental: skips whatever generatedADC/ already has.
    Thin loop over generateADC(order, dry_run=False) -- no new machinery.

    Fails fast at the first order it cannot fully close (never silently
    continues past a failed order, and never silently no-ops an order it
    can't do at all). Orders above 5 are refused outright
    (adc_order_rule.MAX_ADC_ORDER)."""
    results = {}
    for n in orders:
        results[n] = generateADC(n, dry_run=False)
    return results


def GenerateRuntime(orders=(2, 3, 4), verbose=True):
    """Write every generated module the charged-ADC runtime imports at the
    given orders (src.SingleReference.ADC.generated_modules.RUNTIME_MODULES:
    dense blocks, sigma-vector apply modules, their DF twins, the Laplace-fused
    and Goldstone-diagram modules, the non-Dyson pairs, and with ADC(5) the
    order-4 MP amplitude numerators). GenerateADC covers the
    master-formula plan; this covers what adc_autogenerated, sigma_driver,
    nondyson_isr and delta_u3_dressing load. Incremental: a module that exists
    is skipped, and one producer call that writes several modules runs once.
    Raises, naming them, when modules are still missing afterwards.

    Order 5 writes the ADC(5) Laplace modules (tens to hundreds of MB each,
    hours of derivation): run it once per machine, on a cluster node if need be.
    """
    # the diagram emitter pulls in the Goldstone engine (networkx): imported
    # here so generate_adc stays import-cheap, like every producer (_resolve_call)
    from src.CodeGeneration.diagrammatic.emit import default_out_dir
    orders = tuple(orders)
    for n in orders:
        refuse_above_max_order(n, 'GenerateRuntime')
    wanted = [m for m, (n, _) in RUNTIME_MODULES.items() if n in orders]
    done = set()
    for m in wanted:
        if is_generated(m):
            continue
        func_name, kwargs = RUNTIME_MODULES[m][1]
        sig = (func_name, tuple(sorted(kwargs.items())))
        if sig in done:
            continue
        if func_name.startswith('diagrammatic.emit.'):
            kwargs = dict(kwargs, out_dir=default_out_dir())
        if verbose:
            print(f'GenerateRuntime: {m} <- {func_name}({kwargs})', flush=True)
        _resolve_call(func_name)(**kwargs)
        importlib.invalidate_caches()
        done.add(sig)
    # MP order o amplitude numerators (generated_mpn/mpn_amp_numerators_o{o}),
    # first consumed by ADC(o + 1): the order-4 amplitudes of the ADC(5)
    # accessors (norm series, non-Dyson blocks); orders <= 3 are hand-chained
    numerators = [o for o in range(4, max(orders))]
    for o in numerators:
        if not os.path.exists(os.path.join(_MPN_GEN_DIR, f'mpn_amp_numerators_o{o}.py')):
            from src.CodeGeneration.generate_mpn_density import (  # lazy: see _resolve_call
                generate_amplitude_numerator_module)
            if verbose:
                print(f'GenerateRuntime: mpn_amp_numerators_o{o}', flush=True)
            generate_amplitude_numerator_module([o], f'mpn_amp_numerators_o{o}')
            importlib.invalidate_caches()
    missing = [generated_dotted(m) for m in wanted if not is_generated(m)]
    missing += [f'generated_mpn/mpn_amp_numerators_o{o}' for o in numerators
                if not os.path.exists(os.path.join(_MPN_GEN_DIR, f'mpn_amp_numerators_o{o}.py'))]
    if missing:
        raise RuntimeError('GenerateRuntime: still missing after their producers ran:\n  '
                           + '\n  '.join(missing))
    return wanted


if __name__ == '__main__':
    for n in (2, 3, 4, 5):
        generateADC(n, dry_run=True)
        coverage(n)
        print()
