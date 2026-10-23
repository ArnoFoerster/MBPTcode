import os
import re
import sys
import itertools
import argparse
import networkx as nx
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.CodeGeneration.pdaggerq_interface import pq_helper

_PQ = []


def _pq():
    """The one pq_helper, built on first use so the module imports without pdaggerq."""
    if not _PQ:
        _PQ.append(pq_helper('fermi'))
    return _PQ[0]

INDEX_MAP = {
    'e': 'p', 'f': 'q', 'a1': 'p', 'a2': 'q',
    'a3': 'r', 'a4': 's', 'a5': 't', 'a6': 'u',
    'a7': 'v', 'a8': 'w','a9': 'x', 'a10': 'z'
}

# The index spaces of the QDPT terms (after INDEX_MAP): core (occupied),
# virtual and active labels; generate_qdpt_evaluators reads them from here.
VIRS  = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
CORES = ['i', 'j', 'k', 'l', 'm', 'n', 'o']
ACTS  = ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z']


def index_space(idx):
    """'occ', 'vir' or 'act' (anything not core or virtual is active)."""
    if idx in CORES:
        return 'occ'
    if idx in VIRS:
        return 'vir'
    return 'act'


@dataclass
class Term:
    """Structured representation of one Wick-contracted term."""
    prefactor:    float        # numeric coefficient
    operators:    List[str]    # e.g. ['g(p,i,q,j)', 'F(p,q)']
    denom_slices: List[str]    # algebraic denominator per time-slice
                               # e.g. ['E_corr + e_i - e_a']
    sector:       int          # 0, 1, 2, or 3
    raw_str:      str          # full formatted string (for display)
    skeleton_hash: str = ""    # Hash representing the Goldstone skeleton diagram


# ── helper functions ─────────────────────────────────────────────────────────

def is_connected(operators):
    if len(operators) <= 1: return True
    op_indices = [set(re.findall(r'[a-z0-9]+', op.split('(')[1])) for op in operators]
    connected = {0}
    to_visit = [0]
    while to_visit:
        curr = to_visit.pop()
        for i, indices in enumerate(op_indices):
            if i not in connected and not op_indices[curr].isdisjoint(indices):
                connected.add(i)
                to_visit.append(i)
    return len(connected) == len(operators)

def has_internal_loop(op_str):
    indices = re.findall(r'[a-z0-9]+', op_str.split('(')[1])
    env_indices = [idx for idx in indices if idx in 'ijklmnabcd']
    return len(env_indices) != len(set(env_indices))

def is_reducible(term, target_n):
    operators = term[1:]
    if len(operators) <= 1: return False
    g_count = sum(1 for op in operators if op.lstrip().startswith('g'))
    if g_count < 2: return False
    for op in operators:
        if has_internal_loop(op): return True
    return False

def get_indices(op_str):
    match = re.search(r'\((.*?)\)', op_str)
    if not match: return [], []
    indices = [idx.strip() for idx in match.group(1).split(',')]
    n = len(indices)
    return indices[:n//2], indices[n//2:]

def build_denom_string(net_c, net_a, target_n, is_mpn=False, external_indices=frozenset()):
    """Constructs the E_corr algebraic string for a single time slice.
    The on-shell approximation (dropping active indices for n=1) has been removed.

    external_indices: indices that are free (bra/ket) labels rather than
    genuinely internal/summed dummies -- an external index has the same
    value in the initial state and any intermediate state (it is a
    "spectator" line untouched by the inserted interaction vertices), so its
    energy contribution cancels identically and it must NEVER appear in a
    denominator. Excluded here before building the string. Needed for
    amplitude-type (bra-H-...-ket) derivations where bra/ket indices can end
    up as the "leftover" (net_c/net_a) indices at a topological-sort cut
    purely as a bookkeeping artifact, not because they are genuinely
    internal -- see get_denominator's docstring for the concrete failure
    mode this fixes. Default (empty set) preserves prior behavior exactly
    for existing callers (the QDPT active-space effective-Hamiltonian
    sectors in run_wick_contractions/run_mpn, where this was not an issue).
    """
    net_c = [idx for idx in net_c if idx not in external_indices]
    net_a = [idx for idx in net_a if idx not in external_indices]
    denom_parts = []
    for idx in net_a: denom_parts.append(f"+ e_{idx}")
    for idx in net_c: denom_parts.append(f"- e_{idx}")
    denom_parts.sort(key=lambda x: x.startswith('-'))

    orb_str = " ".join(denom_parts).strip()
    
    if is_mpn:
        if orb_str.startswith("+ "): return f"({orb_str[2:]})"
        elif orb_str.startswith("- "): return f"({orb_str})"
        return "()"
    
    if orb_str.startswith("+ "): return f"(E_corr + {orb_str[2:]})"
    elif orb_str.startswith("- "): return f"(E_corr {orb_str})"
    return "(E_corr)"

def get_denominator(term_str, target_n, is_mpn=False, external_indices=frozenset()):
    """external_indices: see build_denom_string's docstring -- passed through
    to every per-cut denominator built here, and also used to exclude
    Kronecker-delta factors (d(...)) from the vertex/topological-sort graph
    entirely when both of a delta's indices are external (a bra/ket-to-
    bra/ket identification carries no interaction-vertex structure and must
    not be treated as a graph node). Default preserves prior behavior for
    existing (non-amplitude) callers, where d(...) factors are filtered out
    upstream by format_and_filter before this function ever sees them.
    """
    factors = [f.strip() for f in term_str.split('*') if '(' in f]
    # P(x,y) antisymmetrization markers are not vertex/tensor factors -- they
    # never represent an interaction and must never be treated as a
    # topological-sort node (this silently injected spurious extra "cuts"
    # with empty denominators before this fix, since P(...)'s bra/ket
    # indices are always external and dropped by the filter below anyway,
    # but the *factor itself* was still being counted as an extra vertex).
    factors = [f for f in factors if not f.startswith('P(')]
    if external_indices:
        factors = [f for f in factors if not (
            f.startswith('d(') and all(
                idx.strip() in external_indices
                for idx in re.search(r'\((.*?)\)', f).group(1).split(',')))]
    if len(factors) <= 1: return ""

    from collections import defaultdict
    edges = defaultdict(list)
    in_degree = {i: 0 for i in range(len(factors))}
    idx_map = defaultdict(lambda: {'c': [], 'a': []})
    
    for i, f in enumerate(factors):
        c, a = get_indices(f)
        for x in c: idx_map[x]['c'].append(i)
        for x in a: idx_map[x]['a'].append(i)
            
    for x, data in idx_map.items():
        for c_idx in data['c']:
            for a_idx in data['a']:
                if c_idx == a_idx: continue
                is_core = x in CORES
                if not is_core:
                    edges[c_idx].append(a_idx)
                    in_degree[a_idx] += 1
                else:
                    edges[a_idx].append(c_idx)
                    in_degree[c_idx] += 1
                    
    # Topological sort
    queue = [i for i in range(len(factors)) if in_degree[i] == 0]
    sequence = []
    
    while queue:
        u = queue.pop(0)
        sequence.append(factors[u])
        for v in edges[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(sequence) != len(factors):
        sequence = factors # Fallback to original order if there's a cycle

    denoms = []
    running_c = []
    running_a = []
    for factor in sequence[:-1]:
        c, a = get_indices(factor)
        running_c.extend(c)
        running_a.extend(a)
        
        net_c = []
        net_a = running_a.copy()
        for x in running_c:
            if x in net_a:
                net_a.remove(x)
            else:
                net_c.append(x)
        denoms.append(build_denom_string(net_c, net_a, target_n, is_mpn, external_indices))

    return " * ".join(f"[{d}]" for d in denoms)


def term_to_graph(operators):
    """
    Converts a list of vertex operators into a NetworkX directed multigraph.
    Nodes represent the interaction vertices.
    Edges represent internal contractions (holes or particles).
    """
    G = nx.MultiDiGraph()
    
    # 1. Create a node for each vertex
    for idx, op in enumerate(operators):
        G.add_node(idx, type='vertex')
        
    # 2. Map every internal index to the vertices it connects
    internal_map = {}
    
    for v_idx, op in enumerate(operators):
        created, annihilated = get_indices(op)
        
        ext_p = 0
        ext_h = 0
        
        for i_c in created:
            if i_c in CORES or i_c in VIRS: # It's an internal line
                if i_c not in internal_map: internal_map[i_c] = {}
                internal_map[i_c]['created_at'] = v_idx
            else:
                ext_p += 1
                
        for i_a in annihilated:
            if i_a in CORES or i_a in VIRS:
                if i_a not in internal_map: internal_map[i_a] = {}
                internal_map[i_a]['annihilated_at'] = v_idx
            else:
                ext_h += 1
                
        G.nodes[v_idx]['ext_p'] = ext_p
        G.nodes[v_idx]['ext_h'] = ext_h

    # 3. Add edges between vertices for every internal contraction
    for idx, connection in internal_map.items():
        if 'created_at' in connection and 'annihilated_at' in connection:
            u = connection['created_at']
            v = connection['annihilated_at']
            edge_type = 'hole' if idx in CORES else 'particle'
            
            # Add directed edge from creator to annihilator
            G.add_edge(u, v, type=edge_type)
            
    return G

def get_skeleton_hash(operators):
    """Returns a unique hash for the base antisymmetrized Goldstone skeleton."""
    multi_G = term_to_graph(operators)
    
    # The WL hash doesn't support MultiDiGraph, so we convert edges into nodes
    G = nx.DiGraph()
    for n, d in multi_G.nodes(data=True):
        node_type = f"{d.get('type', 'vertex')}_{d.get('ext_p', 0)}_{d.get('ext_h', 0)}"
        G.add_node(f"v_{n}", type=node_type)
        
    edge_idx = 0
    for u, v, k, d in multi_G.edges(keys=True, data=True):
        edge_node = f"e_{edge_idx}"
        edge_idx += 1
        G.add_node(edge_node, type=d.get('type', 'unknown'))
        G.add_edge(f"v_{u}", edge_node)
        G.add_edge(edge_node, f"v_{v}")
        
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='type')


def format_and_filter(terms, target_n, print_unique, is_mpn=False):
    """
    Return a dictionary grouping Term objects by their unique antisymmetrized Goldstone skeleton.
    """
    skeleton_groups = {}
    
    for term in terms:
        if any(op.strip().startswith('d(') for op in term): continue
        if not is_connected(term[1:]): continue
        if is_reducible(term, target_n): continue

        try:
            prefactor = float(term[0])
        except ValueError:
            prefactor = 1.0

        mod_term = []
        for piece in term:
            for old_idx, new_idx in INDEX_MAP.items():
                piece = re.sub(rf'\b{old_idx}\b', new_idx, piece)
            piece = re.sub(r'\bq\(', 'F(', piece)
            piece = re.sub(r'\bf\(', 'F(', piece)
            piece = re.sub(r'\bh\(', 'h1(', piece)
            mod_term.append(piece)

        term_str = " * ".join(mod_term)
        operators = [op.strip() for op in mod_term[1:]]

        skel_hash = get_skeleton_hash(operators)

        # If we only care about unique terms, skip the denominator for duplicates.
        if print_unique and skel_hash in skeleton_groups:
            t = Term(
                prefactor=prefactor,
                operators=operators,
                denom_slices=[],
                sector=target_n,
                raw_str=term_str,
                skeleton_hash=skel_hash,
            )
            skeleton_groups[skel_hash].append(t)
            continue

        denom = get_denominator(term_str, target_n, is_mpn)

        raw = f"{term_str} / {denom}" if denom else term_str

        denom_slices: List[str] = []
        if denom:
            for m in re.finditer(r'\[\(([^)]+)\)\]', denom):
                denom_slices.append(m.group(1).strip())

        t = Term(
            prefactor=prefactor,
            operators=operators,
            denom_slices=denom_slices,
            sector=target_n,
            raw_str=raw,
            skeleton_hash=skel_hash,
        )
        
        if skel_hash not in skeleton_groups:
            skeleton_groups[skel_hash] = []
            
        skeleton_groups[skel_hash].append(t)
        
    return skeleton_groups


def get_external_signature(term_str):
    indices = []
    op_part = term_str.split('/')[0]
    for m in re.finditer(r'\((.*?)\)', op_part):
        for idx in m.group(1).split(','):
            idx = idx.strip()
            if idx in ACTS:
                indices.append(idx)
    return tuple(indices)

def print_grouped_terms(grouped_terms, print_unique):
    print(f"-> Generated {sum(len(v) for v in grouped_terms.values())} total diagrams.")
    print(f"-> Collapsed into {len(grouped_terms)} unique antisymmetrized Goldstone skeletons.")
    
    for skel_id, terms_list in grouped_terms.items():
        if not print_unique:
            print(f"   Skeleton {skel_id[:8]}... contains {len(terms_list)} Pauli permutations.")
            for t in terms_list:
                print(f"      {t.raw_str}")
        else:
            base_t = terms_list[0]
            print(f"\n      Base term: {base_t.raw_str}")
            if len(terms_list) > 1:
                print(f"      Antisymmetrizer:")
                base_pref = base_t.prefactor
                for t in terms_list:
                    sig = get_external_signature(t.raw_str)
                    rel_sign = t.prefactor / base_pref if base_pref != 0 else 1.0
                    sign_str = "+1" if rel_sign > 0 else "-1"
                    sig_str = "(" + ", ".join(sig) + ")"
                    print(f"         {sign_str} * {sig_str}")


def evaluate_sector(left_ops, right_ops, name, n, max_pt_order, include_f, print_unique, is_mpn=False, spin_free=False):
    print(f"\n=== {name} (n={n}) ===")
    
    all_terms = []

    _pq().set_left_operators([left_ops])
    _pq().set_right_operators([right_ops])

    # 0th order
    print("--- 0th Order ---")
    _pq().add_operator_product(1.0,    ['h'])
    _pq().add_operator_product(0.25,   ['g'])
    
    _pq().simplify()
    
    spin_labels = {}
    if spin_free:
        all_ext_indices = []
        for op_list in [left_ops, right_ops]:
            for op in op_list:
                match = re.search(r'\((.*?)\)', op)
                if match:
                    indices = [idx.strip() for idx in match.group(1).split(',')]
                    all_ext_indices.extend(indices)
        if n == 1:
            for idx in all_ext_indices:
                spin_labels[idx] = 'a'
        else:
            def sort_key(x):
                match = re.search(r'\d+', x)
                return int(match.group()) if match else x
            sorted_ext = sorted(list(set(all_ext_indices)), key=sort_key)
            # Assign spins per-position-within-half (not a flat i%2 alternation
            # over the whole list): the bra half (left_ops) and ket half
            # (right_ops) each contribute n indices, and bra-position-i must
            # get the SAME spin as ket-position-i for the representative term
            # to conserve total Sz (required for a nonzero pdaggerq result)
            # and to match active_space.spatial_to_spin_orbitals' lift, which
            # assumes bra-position-i and ket-position-i share a spin. A flat
            # i%2 alternation only happens to satisfy this for n=2 (period 2
            # coincides with the half-length); for n=3 it desyncs the halves
            # (bra gets a,b,a while ket gets b,a,b -- 2 alpha vs 1 alpha,
            # Sz-nonconserving) and pdaggerq silently returns zero terms.
            n_per_half = len(sorted_ext) // 2
            for i, idx in enumerate(sorted_ext):
                spin_labels[idx] = 'a' if (i % n_per_half) % 2 == 0 else 'b'
                
    raw_strings = _pq().strings(spin_labels=spin_labels) if spin_free else _pq().strings()
    
    grouped_terms_0 = format_and_filter(raw_strings, n, print_unique, is_mpn)
    
    print_grouped_terms(grouped_terms_0, print_unique)
            
    for t_list in grouped_terms_0.values():
        all_terms.extend(t_list)
    _pq().clear()

    for order in range(2, max_pt_order + 1):
        _pq().set_left_operators([left_ops])
        _pq().set_right_operators([right_ops])

        order_str = f"{order}th"
        if order == 2: order_str = "2nd"
        elif order == 3: order_str = "3rd"
        
        print(f"\n--- {order_str} Order ---")
        if not include_f:
            ops = ['g'] * order
            pref = (0.25) ** order
            _pq().add_operator_product(pref, ops)
        else:
            # V = 0.25 g - 1.0 f
            terms_components = [ (0.25, 'g'), (-1.0, 'f') ]
            for prod in itertools.product(terms_components, repeat=order):
                pref = 1.0
                ops = []
                for p, op in prod:
                    pref *= p
                    ops.append(op)
                _pq().add_operator_product(pref, ops)

        _pq().simplify()
        
        raw_strings = _pq().strings(spin_labels=spin_labels) if spin_free else _pq().strings()
        
        grouped_terms_k = format_and_filter(raw_strings, n, print_unique, is_mpn)
        
        print_grouped_terms(grouped_terms_k, print_unique)
                
        for t_list in grouped_terms_k.values():
            all_terms.extend(t_list)
        _pq().clear()

    print(f"\n=== Total {name} diagrams: {len(all_terms)} ===")
    return all_terms


def run_mpn(max_pt_order=2, include_f=False, print_unique=False, spin_free=False):
    """Generate only the pure MPn (0-body) terms without active space denominators."""
    results = {}
    results["0-Body"] = evaluate_sector(['1'], ['1'], "0-Body", 0, max_pt_order, include_f, print_unique, is_mpn=True, spin_free=spin_free)
    return results

def run_wick_contractions(max_pt_order=2, max_vertex=2, include_f=False, print_unique=False, spin_free=False):
    """Run all sectors and return structured Term objects grouped by sector name."""
    results = {}
    if max_vertex >= 0:
        results["0-Body"] = evaluate_sector(['1'],                                      ['1'],                                          "0-Body", 0, max_pt_order, include_f, print_unique, spin_free=spin_free)
    if max_vertex >= 1:
        results["1-Body"] = evaluate_sector(['a(e)'],                                   ['a*(f)'],                                      "1-Body", 1, max_pt_order, include_f, print_unique, spin_free=spin_free)
    if max_vertex >= 2:
        results["2-Body"] = evaluate_sector(['a(a1)', 'a(a2)'],                         ['a*(a4)', 'a*(a3)'],                           "2-Body", 2, max_pt_order, include_f, print_unique, spin_free=spin_free)
    if max_vertex >= 3:
        results["3-Body"] = evaluate_sector(['a(a1)', 'a(a2)', 'a(a3)'],                ['a*(a6)', 'a*(a5)', 'a*(a4)'],                 "3-Body", 3, max_pt_order, include_f, print_unique, spin_free=spin_free)
    if max_vertex >= 4:
        results["4-Body"] = evaluate_sector(['a(a1)', 'a(a2)', 'a(a3)', 'a(a4)'],       ['a*(a8)', 'a*(a7)', 'a*(a6)', 'a*(a5)'],       "4-Body", 4, max_pt_order, include_f, print_unique, spin_free=spin_free)
    if max_vertex >= 5:
        results["5-Body"] = evaluate_sector(['a(a1)', 'a(a2)', 'a(a3)', 'a(a4)', 'a(a5)'], ['a*(a10)', 'a*(a9)', 'a*(a8)', 'a*(a7)', 'a*(a6)'], "5-Body", 5, max_pt_order, include_f, print_unique, spin_free=spin_free)
    return results


# ── standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wick contractions for downfolding.")
    parser.add_argument("--max-pt-order", type=int, default=2, help="Maximum order in perturbation theory.")
    parser.add_argument("--max-vertex", type=int, default=2, help="Maximum vertex (n-body sector) to evaluate.")
    parser.add_argument("--include-f", action="store_true", help="Include the 'f' operator in the perturbation (use V = g - f).")
    parser.add_argument("--print-unique", action="store_true", help="Only print unique base terms with their antisymmetrizers.")
    parser.add_argument("--mpn-only", action="store_true", help="Only generate pure MPn terms (0-body sector without E_corr).")
    
    args = parser.parse_args()
    
    print(f"Configuration: Max PT Order = {args.max_pt_order}, Max Vertex = {args.max_vertex}, Include F = {args.include_f}, Print Unique = {args.print_unique}, MPn Only = {args.mpn_only}")
    
    if args.mpn_only:
        run_mpn(max_pt_order=args.max_pt_order, include_f=args.include_f, print_unique=args.print_unique)
    else:
        run_wick_contractions(max_pt_order=args.max_pt_order, max_vertex=args.max_vertex, include_f=args.include_f, print_unique=args.print_unique)