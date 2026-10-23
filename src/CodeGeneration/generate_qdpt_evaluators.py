"""QDPT effective-Hamiltonian evaluator generator: turns the Wick-contracted
MP2/MP3 downfolding terms of qdpt_wick into the committed
src/MultiReference/QDPT/generated_evaluators[_spinfree].py
(eval_pt{2,3}_v{2,3}[_spinfree], called by perturbative.build_effective_hamiltonian).

Every multi-vertex term is contracted pairwise in the operator order that
minimizes the largest intermediate (find_optimal_operator_ordering), each
energy denominator applied at the first step that carries all its indices;
indices beyond the rank-6 budget (and all but one virtual, for the cached
emitter) are looped over instead of held. Terms that are pure external-index
permutations of one skeleton share one intermediate (generate_term_code_cached,
cached per canonical key and freed after its last use).
"""
import os
import sys
import re
import itertools
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.CodeGeneration.codegen_common import QDPT_OUT_DIR, generated_header, write_source
from src.CodeGeneration.qdpt_wick import (
    run_wick_contractions, get_skeleton_hash, get_external_signature, Term, CORES, VIRS, ACTS,
    index_space)

TARGET_EXT_INDICES = {
    0: [],
    1: ['p', 'q'],
    2: ['p', 'q', 'r', 's'],
    3: ['p', 'q', 'r', 's', 't', 'u'],
    4: ['p', 'q', 'r', 's', 't', 'u', 'v', 'w'],
    5: ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z'],
}

_SLICE_NAME = {'occ': 'occ_idx', 'vir': 'virt_idx', 'act': 'act_idx'}


def get_slice_str_normal(idx):
    """The index-array name of a label's space (occ_idx / act_idx / virt_idx)."""
    if idx not in CORES and idx not in VIRS and idx not in ACTS:
        raise ValueError(f"Unknown index {idx}")
    return _SLICE_NAME[index_space(idx)]


def parse_operator_str(op_str):
    op_name, indices_str = op_str.split('(')
    indices = [idx.strip() for idx in indices_str.replace(')', '').split(',')]
    return op_name, indices


def _denom_indices(denom_str):
    return [m.group(1) for m in re.finditer(r'e_([a-z0-9_]+)', denom_str)]


def term_indices(term):
    """Every label of a term's operators and denominators, sorted."""
    all_indices = set()
    for op in term.operators:
        all_indices.update(parse_operator_str(op)[1])
    for denom in term.denom_slices:
        all_indices.update(_denom_indices(denom))
    return sorted(all_indices)


def _char_map(all_indices):
    return {idx: chr(65 + i) for i, idx in enumerate(all_indices)}


def sliced_indices_for(all_indices, target_ext, max_rank, single_virtual):
    """The labels looped over instead of held: max_rank - 6 of them (the
    rank-6 intermediate budget), with single_virtual at least all virtuals
    but one; virtuals first, then core, then active, external targets last."""
    nvir = sum(1 for idx in all_indices if idx in VIRS)
    if single_virtual:
        if not (max_rank > 6 or nvir > 1):
            return []
        num = max(max_rank - 6, nvir - 1)
    else:
        if not max_rank > 6:
            return []
        num = max_rank - 6

    def sort_key(idx):
        if idx in target_ext:
            return (3, idx)
        if idx in VIRS:
            return (0, idx)
        if idx in CORES:
            return (1, idx)
        return (2, idx)
    return sorted(all_indices, key=sort_key)[:num]


def _array_name(op_name, spin_free):
    if spin_free:
        return op_name
    return "g_anti_spin" if op_name == 'g' else "h1_spin"


def _block_setup_lines(ordered_ops, get_slice_str_fn, spin_free, full):
    """b{i} (or b{i}_full when indices are sliced) = the operator's block."""
    lines, ops_info = [], []
    for i, op in enumerate(ordered_ops):
        op_name, idxs = parse_operator_str(op)
        slice_str = ", ".join(get_slice_str_fn(idx) for idx in idxs)
        lines.append(f"    b{i}{'_full' if full else ''} = {_array_name(op_name, spin_free)}[np.ix_({slice_str})]")
        ops_info.append({'name': f"b{i}", 'indices': idxs})
    return lines, ops_info


def _slice_parts(idxs, sliced):
    return [f"i_{idx}" if idx in sliced else ":" for idx in idxs]


def _denoms_at(term, denom_steps, keep):
    """[{'str', 'indices'}] of the denominators whose step satisfies keep(step)."""
    return [{'str': term.denom_slices[d], 'indices': _denom_indices(term.denom_slices[d])}
            for d, step in denom_steps.items() if keep(step)]


def _denom_lines(denoms, axes, get_slice_str_fn, sliced, next_denom_var, target, indent="    "):
    """d{k} = <denominator broadcast over axes>, guarded, and target /= d{k}."""
    lines = []
    for denom in denoms:
        expr = generate_denom_expr(denom['str'], axes, get_slice_str_fn, sliced)
        dvar = next_denom_var()
        lines.append(f"{indent}{dvar} = {expr}")
        lines.append(f"{indent}{dvar}[np.abs({dvar}) < 1e-12] = 1e-12")
        lines.append(f"{indent}{target} = {target} / {dvar}")
    return lines


def _contraction_step(term, ops_info, i, current, target_ext, denom_steps, char_map, sliced,
                      get_slice_str_fn, next_denom_var, pref_str):
    """Lines of pairwise step i (current x ops_info[i] -> I{i}), with the
    step's denominators applied and the indices nothing later needs summed;
    returns (lines, the indices of I{i})."""
    nxt = ops_info[i]
    a_set = set(current['indices']) | set(nxt['indices'])
    future = set(target_ext)
    for j in range(i + 1, len(ops_info)):
        future.update(ops_info[j]['indices'])
    for d in _denoms_at(term, denom_steps, lambda step: step > i):
        future.update(d['indices'])
    cur_rem = [idx for idx in current['indices'] if idx not in sliced]
    nxt_rem = [idx for idx in nxt['indices'] if idx not in sliced]
    a_rem = sorted(idx for idx in a_set if idx not in sliced)
    lhs = "".join(char_map[idx] for idx in cur_rem) + "," + "".join(char_map[idx] for idx in nxt_rem)
    lines = [f"    num = {pref_str}np.einsum('{lhs}->{''.join(char_map[idx] for idx in a_rem)}', "
             f"{current['name']}, {nxt['name']}, optimize=True)"]
    denoms = _denoms_at(term, denom_steps, lambda step: step == i)
    a_denom = set(a_set)
    for denom in denoms:
        a_denom.update(denom['indices'])
    a_denom_rem = [idx for idx in sorted(a_denom) if idx not in sliced]
    if denoms:
        a_idx_rem = [idx for idx in a_set if idx not in sliced]
        view = [":" if idx in a_idx_rem else "None" for idx in a_denom_rem]
        if "None" in view:
            lines.append(f"    num = num[{', '.join(view)}]")
        lines += _denom_lines(denoms, a_denom_rem, get_slice_str_fn, sliced, next_denom_var, 'num')
    out_list = sorted(a_denom - (a_denom - future))
    out_rem = [idx for idx in out_list if idx not in sliced]
    if not set(a_denom_rem) - set(out_rem):
        lines.append(f"    I{i} = num")
    else:
        lines.append(f"    I{i} = np.einsum('{''.join(char_map[idx] for idx in a_denom_rem)}->"
                     f"{''.join(char_map[idx] for idx in out_rem)}', num, optimize=True)")
    return lines, out_list


def _loop_nest(sliced, body, get_slice_str_fn, pass_if_empty=False):
    """for i_x in range(len(<space>)): ... headers for every sliced index, and
    the body re-indented under them."""
    headers, indent = [], ""
    for idx in sliced:
        headers.append(f"    {indent}for i_{idx} in range(len({get_slice_str_fn(idx)})):")
        indent += "    "
    out = []
    for line in body:
        if line.strip():
            out.append(f"    {indent}{line[4:] if line.startswith('    ') else line}")
        else:
            out.append(line)
    if pass_if_empty and (not out or all(not line.strip() for line in out)):
        out.append(f"    {indent}pass")
    return headers + out


def _counter():
    n = [0]

    def nxt():
        name = f"d{n[0]}"
        n[0] += 1
        return name
    return nxt


def generate_denom_expr(denom_str, remaining_indices, get_slice_str_fn, sliced_indices=None):
    if sliced_indices is None:
        sliced_indices = []
    parts = []
    matches = re.finditer(r'([+-])\s*e_([a-z0-9_]+)', denom_str)
    num_holes = len(re.findall(r'\+\s*e_', denom_str))
    for match in matches:
        sign = match.group(1)
        idx = match.group(2)

        # If it's sliced, it's a scalar in this scope
        if idx in sliced_indices:
            slice_name = get_slice_str_fn(idx)
            slice_str = f"{slice_name}[i_{idx}]"
            broadcast = ""
        else:
            slice_str = get_slice_str_fn(idx)
            axis = remaining_indices.index(idx)
            colons = ["None"] * len(remaining_indices)
            colons[axis] = ":"
            broadcast = f"[{', '.join(colons)}]"

        if sign == '+':
            parts.append(f"+ (eps_spin[{slice_str}] + (shift / {num_holes} if shift is not None else 0.0)){broadcast}")
        else:
            parts.append(f"- eps_spin[{slice_str}]{broadcast}")

    expr = " ".join(parts)
    if expr.startswith("+ "): expr = expr[2:]
    return expr


def generate_term_code(term, target_dict_name, get_slice_str_fn, spin_free=False):
    """One term as a single dense einsum over all its indices (no factorization)."""
    all_indices = term_indices(term)
    char_map = _char_map(all_indices)
    lines, ops_info = _block_setup_lines(term.operators, get_slice_str_fn, spin_free, full=False)
    block_names = [op['name'] for op in ops_info]
    block_einsums = ["".join(char_map[idx] for idx in op['indices']) for op in ops_info]
    num_einsum_str = "".join(char_map[idx] for idx in all_indices)
    out_str = "".join(char_map[idx] for idx in TARGET_EXT_INDICES[term.sector])
    ops_str = ", ".join(block_names)

    if len(term.denom_slices) == 0:
        rhs_einsum = ",".join(block_einsums) + "->" + out_str
        if len(block_names) == 0:
            lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor}")
        else:
            lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor} * np.einsum('{rhs_einsum}', {ops_str}, optimize=True)")
    else:
        rhs_einsum = ",".join(block_einsums) + "->" + num_einsum_str
        lines.append(f"    num = {term.prefactor} * np.einsum('{rhs_einsum}', {ops_str}, optimize=True)")
        for denom_idx, denom in enumerate(term.denom_slices):
            expr = generate_denom_expr(denom, all_indices, get_slice_str_fn)
            if expr:
                lines.append(f"    d{denom_idx} = {expr}")
                lines.append(f"    d{denom_idx}[np.abs(d{denom_idx}) < 1e-12] = 1e-12")
                lines.append(f"    num = num / d{denom_idx}")
        lines.append(f"    {target_dict_name}[{term.sector}] += np.einsum('{num_einsum_str}->{out_str}', num, optimize=True)")

    return "\n".join(lines)


def _split_loop_lines(candidate_lines, loop_dependent_vars, setup_lines, sliced_indices, lhs_indexed):
    """Partition lines into loop-invariant (appended to setup_lines) and
    loop-dependent ones (returned): a line depends on the loop when its
    right-hand side (or, without '=', any word) names a loop-dependent
    variable; its left-hand side then becomes loop-dependent, and a later
    loop-invariant assignment to it makes it invariant again. lhs_indexed: an
    indexed assignment to a loop-dependent variable is loop-dependent."""
    loop_body = []
    loop_idx = set(f"i_{idx}" for idx in sliced_indices)
    for line in candidate_lines:
        words = re.findall(r'[a-zA-Z0-9_]+', line)
        if lhs_indexed and '=' in line:
            is_dep = any(w in loop_dependent_vars for w in re.findall(r'[a-zA-Z0-9_]+', line.split('=', 1)[1]))
        else:
            is_dep = any(w in loop_dependent_vars for w in words)
        if lhs_indexed and ('[' in line.split('=', 1)[0] if '=' in line else False):
            if re.match(r'^\s*([a-zA-Z0-9_]+)', line).group(1) in loop_dependent_vars:
                is_dep = True
        match = re.match(r'^\s*([a-zA-Z0-9_]+)\s*=', line)
        if match:
            lhs = match.group(1)
            if is_dep:
                loop_dependent_vars.add(lhs)
            elif lhs in loop_dependent_vars and lhs not in loop_idx:
                loop_dependent_vars.remove(lhs)
        (loop_body if is_dep else setup_lines).append(line)
    return loop_body


def _accumulate_line(target_dict_name, sector, target_slice_str, pref_str, rhs):
    sl = f"[{target_slice_str}]" if target_slice_str else ""
    return f"    {target_dict_name}[{sector}]{sl} += {pref_str}{rhs}"


def generate_term_code_factorized(term, target_dict_name, get_slice_str_fn, spin_free=False):
    """One term as a chain of pairwise contractions in the optimal operator
    order, denominators applied at their earliest step, indices beyond the
    rank-6 budget looped over."""
    all_indices = term_indices(term)
    char_map = _char_map(all_indices)
    target_ext = TARGET_EXT_INDICES[term.sector]
    next_denom_var = _counter()
    if len(term.operators) == 0:
        return f"    {target_dict_name}[{term.sector}] += {term.prefactor}"

    opt_perm, denom_steps, max_rank = find_optimal_operator_ordering(term, target_ext)
    ordered_ops = [term.operators[p] for p in opt_perm]
    sliced_indices = sliced_indices_for(all_indices, target_ext, max_rank, single_virtual=False)
    setup_lines, ops_info = _block_setup_lines(ordered_ops, get_slice_str_fn, spin_free,
                                               full=bool(sliced_indices))
    candidate_lines = []
    if sliced_indices:
        for i, op in enumerate(ordered_ops):
            parts = _slice_parts(parse_operator_str(op)[1], sliced_indices)
            candidate_lines.append(f"    b{i} = b{i}_full[{', '.join(parts)}]"
                                   if any(p.startswith("i_") for p in parts) else f"    b{i} = b{i}_full")
    target_slice_str = ", ".join(_slice_parts(target_ext, sliced_indices))
    final_einsum_str = "".join(char_map[idx] for idx in target_ext if idx not in sliced_indices)

    if len(term.denom_slices) == 0:
        rhs_einsum = ",".join("".join(char_map[idx] for idx in op['indices'] if idx not in sliced_indices)
                              for op in ops_info)
        ops_str = ", ".join(op['name'] for op in ops_info)
        if len(ops_info) == 0:
            candidate_lines.append(_accumulate_line(target_dict_name, term.sector, target_slice_str,
                                                    '', term.prefactor))
        else:
            candidate_lines.append(_accumulate_line(
                target_dict_name, term.sector, target_slice_str, f"{term.prefactor} * ",
                f"np.einsum('{rhs_einsum}->{final_einsum_str}', {ops_str}, optimize=True)"))
    else:
        denoms0 = _denoms_at(term, denom_steps, lambda step: step == 0)
        if denoms0:
            b0_rem = [idx for idx in ops_info[0]['indices'] if idx not in sliced_indices]
            candidate_lines += _denom_lines(denoms0, b0_rem, get_slice_str_fn, sliced_indices,
                                            next_denom_var, 'b0')
        current = ops_info[0]
        for i in range(1, len(ops_info)):
            lines, out_list = _contraction_step(
                term, ops_info, i, current, target_ext, denom_steps, char_map, sliced_indices,
                get_slice_str_fn, next_denom_var, f"{term.prefactor} * " if i == 1 else "")
            candidate_lines += lines
            current = {'name': f"I{i}", 'indices': out_list}
        cur_rem = "".join(char_map[idx] for idx in current['indices'] if idx not in sliced_indices)
        candidate_lines.append(_accumulate_line(
            target_dict_name, term.sector, target_slice_str, '',
            f"np.einsum('{cur_rem}->{final_einsum_str}', {current['name']}, optimize=True)"))

    loop_body = _split_loop_lines(candidate_lines, set(f"i_{idx}" for idx in sliced_indices),
                                  setup_lines, sliced_indices, lhs_indexed=True)
    if sliced_indices:
        return "\n".join(setup_lines + _loop_nest(sliced_indices, loop_body, get_slice_str_fn))
    return "\n".join(setup_lines + loop_body)


def generate_term_code_cached(term, target_dict_name, get_slice_str_fn, intermediate_id, is_first_in_group, use_counts, spin_free=False):
    """generate_term_code_factorized for a term whose first pairwise
    intermediate I1 is shared with other terms: I1 (without the prefactor,
    applied at the final accumulation) is built once into
    _cache['int_<id>'] by the first term of its group and freed after its last
    use (_use_count). Sliced indices of the first step are looped inside the
    cache build; those surviving into I1 are looped over the rest of the
    chain."""
    all_indices = term_indices(term)
    char_map = _char_map(all_indices)
    target_ext = TARGET_EXT_INDICES[term.sector]
    is_i1_sliced = False
    next_denom_var = _counter()
    if len(term.operators) == 0:
        return f"    {target_dict_name}[{term.sector}] += {term.prefactor}"

    opt_perm, denom_steps, max_rank = find_optimal_operator_ordering(term, target_ext)
    ordered_ops = [term.operators[p] for p in opt_perm]
    sliced_indices = sliced_indices_for(all_indices, target_ext, max_rank, single_virtual=True)
    # the full blocks are always defined; b0/b1 are cut from them in the cache build
    setup_lines, ops_info = _block_setup_lines(ordered_ops, get_slice_str_fn, spin_free,
                                               full=bool(sliced_indices))
    denoms0 = _denoms_at(term, denom_steps, lambda step: step == 0)

    cache_sliced_indices, loop_sliced_indices = [], []
    candidate_lines = []
    current = ops_info[0]
    for i in range(1, len(ops_info)):
        if i == 1:
            # I1's indices before its denominators: the sliced ones among the two
            # first operators are looped in the cache build, those surviving into
            # I1 over the rest of the chain
            a_set = set(current['indices']) | set(ops_info[1]['indices'])
            future = set(target_ext)
            for j in range(2, len(ops_info)):
                future.update(ops_info[j]['indices'])
            for d in _denoms_at(term, denom_steps, lambda step: step > 1):
                future.update(d['indices'])
            i1_out_list = sorted(a_set & future)
            if sliced_indices:
                cache_sliced_indices = [idx for idx in sliced_indices if idx in a_set]
                loop_sliced_indices = [idx for idx in sliced_indices if idx in i1_out_list]
        step_sliced = cache_sliced_indices if i < 2 else loop_sliced_indices
        if sliced_indices and i >= 2:
            parts = _slice_parts(parse_operator_str(ordered_ops[i])[1], loop_sliced_indices)
            candidate_lines.append(f"    b{i} = b{i}_full[{', '.join(parts)}]"
                                   if any(p.startswith("i_") for p in parts) else f"    b{i} = b{i}_full")
        lines, out_list = _contraction_step(term, ops_info, i, current, target_ext, denom_steps,
                                            char_map, step_sliced, get_slice_str_fn, next_denom_var, "")
        candidate_lines += lines
        out_rem = [idx for idx in out_list if idx not in step_sliced]
        current = {'name': f"I{i}", 'indices': out_list}
        if i == 1 and loop_sliced_indices:
            parts = _slice_parts(out_list, loop_sliced_indices)
            if any(p.startswith("i_") for p in parts):
                candidate_lines.append(f"    I1 = _cache['int_{intermediate_id}'][{', '.join(parts)}]")
                is_i1_sliced = True
                current = {'name': "I1", 'indices': out_rem}

    i1_line_idx = next((k for k, line in enumerate(candidate_lines) if re.search(r'\bI1\s*=', line)), -1)
    if i1_line_idx == -1:
        raise ValueError(f"Could not find I1 definition in term {term.raw_str}")
    i1_comp_lines = candidate_lines[:i1_line_idx + 1]
    rest_comp_lines = candidate_lines[i1_line_idx + 1:]

    target_slice_str = ", ".join(_slice_parts(target_ext, loop_sliced_indices))
    final_einsum_str = "".join(char_map[idx] for idx in target_ext if idx not in loop_sliced_indices)
    cur_rem = "".join(char_map[idx] for idx in current['indices'] if idx not in loop_sliced_indices)
    rest_comp_lines.append(_accumulate_line(
        target_dict_name, term.sector, target_slice_str, f"{term.prefactor} * ",
        f"np.einsum('{cur_rem}->{final_einsum_str}', {current['name']}, optimize=True)"))

    setup_body = []
    if is_first_in_group:
        setup_body.append(f"    if 'int_{intermediate_id}' not in _cache:")
        indent = "        "
        if cache_sliced_indices:
            shape_str = ", ".join(f"len({get_slice_str_fn(idx)})" for idx in i1_out_list)
            setup_body.append(f"        I1 = np.zeros(({shape_str}))")
            for idx in cache_sliced_indices:
                setup_body.append(f"{indent}for i_{idx} in range(len({get_slice_str_fn(idx)})):")
                indent += "    "
        for i in range(2):
            if sliced_indices:
                parts = _slice_parts(parse_operator_str(ordered_ops[i])[1], cache_sliced_indices)
                setup_body.append(f"{indent}b{i} = b{i}_full[{', '.join(parts)}]")
            else:
                setup_body.append(f"{indent}b{i} = b{i}")
        if denoms0:
            b0_rem = [idx for idx in ops_info[0]['indices'] if idx not in cache_sliced_indices]
            setup_body += _denom_lines(denoms0, b0_rem, get_slice_str_fn, cache_sliced_indices,
                                       next_denom_var, 'b0', indent=indent)
        for line in i1_comp_lines:
            if denoms0 and (re.search(r'\bb0 / d\d+\b', line) or re.search(r'\bd\d+\[np\.abs', line)
                            or "d = generate" in line):
                continue
            if cache_sliced_indices and re.match(r'^\s*I1\s*=', line):
                parts = _slice_parts(i1_out_list, cache_sliced_indices)
                if any(p != ":" for p in parts):
                    line = line.replace("I1 =", f"I1[{', '.join(parts)}] +=")
                else:
                    line = line.replace("I1 =", "I1 +=")
            setup_body.append(f"{indent}{line.strip()}")
        setup_body.append(f"        _cache['int_{intermediate_id}'] = I1")
    if not is_i1_sliced:
        setup_body.append(f"    I1 = _cache['int_{intermediate_id}']")

    post_setup_lines = []
    loop_body = _split_loop_lines(rest_comp_lines, set(f"i_{idx}" for idx in loop_sliced_indices),
                                  post_setup_lines, loop_sliced_indices, lhs_indexed=False)
    cleanup_lines = [f"    _use_count['int_{intermediate_id}'] -= 1",
                     f"    if _use_count['int_{intermediate_id}'] == 0:",
                     f"        del _cache['int_{intermediate_id}']"]
    if loop_sliced_indices:
        loop_body = _loop_nest(loop_sliced_indices, loop_body, get_slice_str_fn, pass_if_empty=True)
    return "\n".join(setup_lines + setup_body + post_setup_lines + loop_body + cleanup_lines)


def find_optimal_operator_ordering(term, target_ext):
    operators = term.operators
    denom_slices = term.denom_slices
    
    if len(operators) == 0:
        return [], {}, 0
        
    if len(operators) == 1:
        denom_steps = {d_idx: 0 for d_idx in range(len(denom_slices))}
        return [0], denom_steps, len(operators)
        
    best_max_rank = 999
    best_cost = 99999999
    best_perm = None
    best_denom_steps = None
    
    for perm in itertools.permutations(range(len(operators))):
        ordered_ops = [operators[p] for p in perm]
        
        # Map denominators to the earliest step
        denom_steps = {}
        for d_idx, denom in enumerate(denom_slices):
            matches = re.finditer(r'e_([a-z0-9_]+)', denom)
            d_idxs = {m.group(1) for m in matches}
            
            found_step = len(ordered_ops) - 1
            for step in range(len(ordered_ops)):
                acc_indices = set()
                for s in range(step + 1):
                    _, idxs = parse_operator_str(ordered_ops[s])
                    acc_indices.update(idxs)
                if d_idxs.issubset(acc_indices):
                    found_step = step
                    break
            denom_steps[d_idx] = found_step
            
        # Calculate intermediate ranks and cost
        current_indices = set()
        _, first_idxs = parse_operator_str(ordered_ops[0])
        current_indices = set(first_idxs)
        
        ranks = []
        cost = 0
        for step in range(1, len(ordered_ops)):
            _, next_idxs = parse_operator_str(ordered_ops[step])
            
            denom_indices = set()
            for d_idx, step_associated in denom_steps.items():
                if step_associated == step:
                    matches = re.finditer(r'e_([a-z0-9_]+)', denom_slices[d_idx])
                    denom_indices.update(m.group(1) for m in matches)
                    
            A_denom_set = current_indices | set(next_idxs) | denom_indices
            rank = len(A_denom_set)
            ranks.append(rank)
            cost += 2**rank
            
            # Future indices needed
            future_indices = set(target_ext)
            for j in range(step + 1, len(ordered_ops)):
                _, future_idxs = parse_operator_str(ordered_ops[j])
                future_indices.update(future_idxs)
            for d_idx, step_associated in denom_steps.items():
                if step_associated > step:
                    matches = re.finditer(r'e_([a-z0-9_]+)', denom_slices[d_idx])
                    future_indices.update(m.group(1) for m in matches)
                    
            sum_indices = A_denom_set - future_indices
            current_indices = A_denom_set - sum_indices
            
        max_rank = max(ranks) if ranks else len(current_indices)
        
        if (max_rank < best_max_rank) or (max_rank == best_max_rank and cost < best_cost):
            best_max_rank = max_rank
            best_cost = cost
            best_perm = perm
            best_denom_steps = denom_steps
            
    return list(best_perm), best_denom_steps, best_max_rank


def canonical_intermediate_key(term, opt_perm, denom_steps, sliced_indices):
    ordered_ops = [term.operators[p] for p in opt_perm]
    step1_ops = [ordered_ops[0], ordered_ops[1]]
    
    # denominators at step 0 and 1
    step1_denoms = []
    for d_idx, step_associated in denom_steps.items():
        if step_associated in [0, 1]:
            step1_denoms.append(term.denom_slices[d_idx])
            
    # Find all indices in step1_ops and step1_denoms
    step1_indices = set()
    for op in step1_ops:
        _, idxs = parse_operator_str(op)
        step1_indices.update(idxs)
    for denom in step1_denoms:
        matches = re.finditer(r'e_([a-z0-9_]+)', denom)
        for m in matches:
            step1_indices.add(m.group(1))
            
    step1_indices = sorted(list(step1_indices))
    
    # Rename map
    rename_map = {}
    counters = {'occ': 0, 'vir': 0, 'act': 0}
    for idx in step1_indices:
        t = index_space(idx)
        if idx not in rename_map:
            rename_map[idx] = f"{t[0]}{counters[t]}"
            counters[t] += 1
            
    # Canonicalize ops
    canon_ops = []
    for op in step1_ops:
        name, idxs = parse_operator_str(op)
        canon_idxs = [rename_map[idx] for idx in idxs]
        canon_ops.append(f"{name}({','.join(canon_idxs)})")
    canon_ops.sort()
    
    # Canonicalize denoms
    canon_denoms = []
    for denom in step1_denoms:
        matches = re.findall(r'e_([a-z0-9_]+)', denom)
        canon_denom = denom
        for m in sorted(list(set(matches)), key=len, reverse=True):
            canon_denom = re.sub(r'\be_' + m + r'\b', f"e_{rename_map[m]}", canon_denom)
        canon_denoms.append(canon_denom)
    canon_denoms.sort()
    
    # Find canonical names of sliced indices that are in step 1
    canon_sliced = []
    for idx in sliced_indices:
        if idx in rename_map:
            canon_sliced.append(rename_map[idx])
    canon_sliced.sort()
    
    return f"{' * '.join(canon_ops)} / ({' * '.join(canon_denoms)}) | sliced: {','.join(canon_sliced)}"


class SpatialTerm:
    def __init__(self, prefactor, operators, denom_slices, sector):
        self.prefactor = prefactor
        self.operators = operators
        self.denom_slices = denom_slices
        self.sector = sector

def expand_operator(op_name, idxs):
    idxs = tuple(idxs)
    if op_name in ['h_aa', 'h_bb', 'f_aa', 'f_bb', 'h1_aa', 'h1_bb']:
        return [(1.0, ('h1', idxs))]
    elif op_name in ['g_aaaa', 'g_bbbb']:
        return [
            (1.0, ('g', idxs)),
            (-1.0, ('g', (idxs[0], idxs[1], idxs[3], idxs[2])))
        ]
    elif op_name in ['g_abab', 'g_baba']:
        return [(1.0, ('g', idxs))]
    elif op_name in ['g_abba', 'g_baab']:
        return [(-1.0, ('g', (idxs[0], idxs[1], idxs[3], idxs[2])))]
    elif op_name == 'g':
        return [(1.0, ('g', idxs))]
    elif op_name == 'h1':
        return [(1.0, ('h1', idxs))]
    else:
        return []

def expand_wick_term(term):
    op_expansions = []
    for op in term.operators:
        op_name, idxs = parse_operator_str(op)
        expansion = expand_operator(op_name, idxs)
        if not expansion:
            return []
        op_expansions.append(expansion)

    expanded_terms = []
    for prod in itertools.product(*op_expansions):
        coef = term.prefactor
        ops = []
        for c, op_tuple in prod:
            coef *= c
            ops.append(op_tuple)
        expanded_terms.append(SpatialTerm(
            prefactor=coef,
            operators=ops,
            denom_slices=term.denom_slices,
            sector=term.sector
        ))
    return expanded_terms

def canonical_spatial_term_key(operators, denom_slices):
    sorted_ops = tuple(sorted([(op[0], tuple(op[1])) for op in operators], key=lambda x: (x[0], x[1])))
    
    def canonical_slice(denom_str):
        matches = re.finditer(r'([+-]?)\s*e_([a-z0-9_]+)', denom_str)
        pos = []
        neg = []
        for m in matches:
            sign = m.group(1)
            idx = m.group(2)
            if sign == '-':
                neg.append(idx)
            else:
                pos.append(idx)
        return (tuple(sorted(pos)), tuple(sorted(neg)))
        
    canonical_denoms = tuple(sorted([canonical_slice(d) for d in denom_slices]))
    return (sorted_ops, canonical_denoms)

def optimize_spatial_terms(terms):
    expanded = []
    for t in terms:
        expanded.extend(expand_wick_term(t))
        
    groups = {}
    for t in expanded:
        key = canonical_spatial_term_key(t.operators, t.denom_slices)
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    optimized = []
    for key, group in groups.items():
        pref = sum(t.prefactor for t in group)
        if abs(pref) < 1e-12:
            continue
        template = group[0]
        ops_str = [f"{op[0]}({','.join(op[1])})" for op in template.operators]
        skel_hash = get_skeleton_hash(ops_str)
        
        ops_joined = " * ".join(ops_str)
        denom_joined = " / " + " * ".join(template.denom_slices) if template.denom_slices else ""
        raw_str = f"{pref:+.4f} * {ops_joined}{denom_joined}"
        
        optimized.append(Term(
            prefactor=pref,
            operators=ops_str,
            denom_slices=template.denom_slices,
            sector=template.sector,
            raw_str=raw_str,
            skeleton_hash=skel_hash
        ))
    return optimized

def generate_missing_mpn_code(terms, max_vertex, max_pt_order, optimize_factorization=False, spin_free=False):
    lines = []
    lines.append("    # --- Missing MPn Shift ---")
    lines.append("    missing_shift = 0.0")
    
    for term in terms:
        unique_indices = term_indices(term)
        index_choices = {}
        for idx in unique_indices:
            t = index_space(idx)
            if t == 'occ':
                index_choices[idx] = ['occ_idx', 'act_occ_idx']
            elif t == 'vir':
                index_choices[idx] = ['virt_idx', 'act_vir_idx']
            else:
                index_choices[idx] = ['act_idx']
                
        keys = list(index_choices.keys())
        values = list(index_choices.values())
        
        for p in itertools.product(*values):
            mapping = dict(zip(keys, p))
            has_active = any(val in ['act_occ_idx', 'act_vir_idx', 'act_idx'] for val in mapping.values())
            if not has_active:
                continue
                
            def get_slice_assign(idx):
                return mapping[idx]
                
            lines.append(f"    # Term: {term.raw_str}")
            lines.append(f"    # Assignment: {mapping}")
            if len(unique_indices) > 6 and optimize_factorization:
                term_code = generate_term_code_factorized(term, "dummy", get_slice_assign, spin_free=spin_free)
            else:
                term_code = generate_term_code(term, "dummy", get_slice_assign, spin_free=spin_free)
            lines.append(term_code.replace("dummy[0] +=", "missing_shift +="))
            
    lines.append("    g_eff[0] += missing_shift")
    return "\n".join(lines)

def generate_function(pt_order, max_vertex, wick_terms, optimize_factorization=False, spin_free=False):
    all_terms = []
    for sector, term_list in wick_terms.items():
        n = int(sector.split('-')[0])
        if n > max_vertex: continue
        filtered = [t for t in term_list if len(t.operators) <= max(1, pt_order)]
        if spin_free:
            optimized = optimize_spatial_terms(filtered)
            all_terms.extend(optimized)
        else:
            all_terms.extend(filtered)

    skeleton_groups = {}
    uncached_terms = []
    
    def remap_denom_slices(denom_slices, mapping):
        def repl(m):
            return f"e_{mapping.get(m.group(1), m.group(1))}"
        return [re.sub(r'\be_([a-z0-9_]+)\b', repl, d) for d in denom_slices]

    def is_pure_external_permutation(base_term, term):
        base_sig = get_external_signature(base_term.raw_str)
        term_sig = get_external_signature(term.raw_str)
        if len(base_sig) != len(term_sig):
            return False
        if len(base_sig) == 0:
            mapping = {}
        else:
            mapping = {}
            for t_idx, b_idx in zip(term_sig, base_sig):
                mapping[t_idx] = b_idx
        mapped_ops = []
        for op in term.operators:
            op_name, idxs = parse_operator_str(op)
            mapped_idxs = tuple(mapping.get(idx, idx) for idx in idxs)
            mapped_ops.append(f"{op_name}({','.join(mapped_idxs)})")
        if sorted(mapped_ops) != sorted(base_term.operators):
            return False
        # Operators alone aren't enough: two terms can expand to identical
        # operator strings (e.g. via expand_operator's antisymmetrization
        # split) while carrying genuinely different denominators inherited
        # from different upstream diagrams. Reusing one term's cached
        # numerator under the other's (different) denominator silently
        # substitutes the wrong energy denominator, so require the mapped
        # denominators to match too before treating this as a pure
        # external-index permutation eligible for the transpose shortcut.
        mapped_denoms = remap_denom_slices(term.denom_slices, mapping)
        return sorted(mapped_denoms) == sorted(base_term.denom_slices)

    for term in all_terms:
        if len(term.operators) < 2:
            uncached_terms.append(term)
        else:
            skel = term.skeleton_hash or "default_skel"
            if skel not in skeleton_groups:
                skeleton_groups[skel] = []
            
            joined = False
            for sub_group in skeleton_groups[skel]:
                base_term = sub_group[0]
                if is_pure_external_permutation(base_term, term):
                    sub_group.append(term)
                    joined = True
                    break
            if not joined:
                skeleton_groups[skel].append([term])
                
    all_groups = []
    for skel, sub_groups in skeleton_groups.items():
        for sub_group in sub_groups:
            all_groups.append(sub_group)

    term_intermediates = {}
    intermediate_counts = {}
    intermediate_keys = []
    
    group_bases = {}
    for group_idx, group in enumerate(all_groups):
        base_term = group[0]
        group_bases[group_idx] = base_term
        
        target_ext = TARGET_EXT_INDICES[base_term.sector]
        opt_perm, denom_steps, max_rank = find_optimal_operator_ordering(base_term, target_ext)
        
        sliced_indices = sliced_indices_for(term_indices(base_term), target_ext, max_rank,
                                            single_virtual=True)
        canon_key = canonical_intermediate_key(base_term, opt_perm, denom_steps, sliced_indices)
        term_intermediates[id(base_term)] = canon_key
        intermediate_counts[canon_key] = intermediate_counts.get(canon_key, 0) + 1
        if canon_key not in intermediate_keys:
            intermediate_keys.append(canon_key)
            
    key_to_id = {key: idx for idx, key in enumerate(intermediate_keys)}
    
    lines = []
    if spin_free:
        lines.append(f"def eval_pt{pt_order}_v{max_vertex}_spinfree(g_spatial, h1_spatial, eps_spatial, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):")
        lines.append("    import numpy as np")
        lines.append("    g = g_spatial")
        lines.append("    h1 = h1_spatial")
        lines.append("    eps_spin = eps_spatial")
    else:
        lines.append(f"def eval_pt{pt_order}_v{max_vertex}(g_anti_spin, h1_spin, eps_spin, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):")
        lines.append("    import numpy as np")
        
    lines.append(f"    g_eff = {{n: np.zeros((len(act_idx),) * (2*n)) for n in range(1, {max_vertex + 1})}}")
    lines.append("    g_eff[0] = 0.0")
    lines.append("    _cache = {}")
    lines.append("    _use_count = {")
    for key, count in intermediate_counts.items():
        lines.append(f"        'int_{key_to_id[key]}': {count},")
    lines.append("    }")
    
    for term in uncached_terms:
        lines.append(f"\n    # Sector: {term.sector}-Body | Term: {term.raw_str}")
        lines.append(generate_term_code(term, "g_eff", get_slice_str_normal, spin_free=spin_free))

    generated_intermediates = set()

    for group_idx, group in enumerate(all_groups):
        base_term = group_bases[group_idx]
        skel = base_term.skeleton_hash or "default_skel"
        canon_key = term_intermediates[id(base_term)]
        int_id = key_to_id[canon_key]
        is_first = int_id not in generated_intermediates
        generated_intermediates.add(int_id)
        
        lines.append(f"\n    # --- Skeleton {skel[:8]} contains {len(group)} permutations ---")
        lines.append(f"    # Base Term: {base_term.raw_str}")
        
        lines.append(f"    g_temp = {{{base_term.sector}: np.zeros((len(act_idx),) * {2 * base_term.sector})}}")
        
        term_code = generate_term_code_cached(
            base_term,
            "g_temp",
            get_slice_str_normal,
            int_id,
            is_first,
            intermediate_counts,
            spin_free=spin_free
        )
        lines.append(term_code)
        
        base_sig = get_external_signature(base_term.raw_str)
        
        for term in group:
            lines.append(f"    # Projecting permutation: {term.raw_str}")
            term_sig = get_external_signature(term.raw_str)
            
            if len(base_sig) > 0 and len(base_sig) == len(term_sig):
                try:
                    target_ext = TARGET_EXT_INDICES[base_term.sector]
                    axes = [target_ext.index(base_sig[term_sig.index(idx)]) for idx in target_ext]
                    rel_sign = term.prefactor / base_term.prefactor if base_term.prefactor != 0 else 1.0
                    axes_str = ", ".join(map(str, axes))
                    lines.append(f"    g_eff[{term.sector}] += {rel_sign:.12f} * np.transpose(g_temp[{base_term.sector}], ({axes_str}))")
                except ValueError:
                    lines.append(f"    # Fallback direct generation due to signature index mismatch")
                    lines.append(generate_term_code(term, "g_eff", get_slice_str_normal, spin_free=spin_free))
            else:
                rel_sign = term.prefactor / base_term.prefactor if base_term.prefactor != 0 else 1.0
                lines.append(f"    g_eff[{term.sector}] += {rel_sign:.12f} * g_temp[{base_term.sector}]")

    # missing mpn (0-body terms)
    if '0-Body' in wick_terms:
        mpn_terms = [t for t in wick_terms['0-Body'] if len(t.operators) > max_vertex and len(t.operators) <= max(1, pt_order)]
        if mpn_terms:
            if spin_free:
                opt_mpn = optimize_spatial_terms(mpn_terms)
                lines.append(generate_missing_mpn_code(opt_mpn, max_vertex, pt_order, optimize_factorization=optimize_factorization, spin_free=True))
            else:
                lines.append(generate_missing_mpn_code(mpn_terms, max_vertex, pt_order, optimize_factorization=optimize_factorization, spin_free=False))
        
    lines.append("    return g_eff")
    return "\n".join(lines)

COMBINATIONS = ((2, 2), (2, 3), (3, 2), (3, 3))   # (PT order, max vertex rank)


def generate(out_dir=None):
    """Write generated_evaluators.py and generated_evaluators_spinfree.py
    (eval_pt{pt}_v{v}[_spinfree] for every COMBINATIONS entry) into out_dir
    (default: src/MultiReference/QDPT)."""
    # every denominator gets its own variable d0, d1, ...: a shared name could be
    # hoisted out of a sliced-index loop by the dependency classifier while a
    # later same-named occurrence stayed inside it
    root = out_dir or QDPT_OUT_DIR
    header = ('"""QDPT active-space effective Hamiltonian g_eff (sectors 0 to max_vertex), {what}.\n\n'
              'Machine-generated from the Wick-contracted MP2/MP3 downfolding diagrams, for\n'
              'max vertex rank 2 or 3{extra}. Do not edit by hand. Called from the QDPT driver,\n'
              'perturbative.build_effective_hamiltonian.\n"""\nimport numpy as np\n\n')
    text = generated_header('generate_qdpt_evaluators.py') + header.format(what='spin-orbital', extra='')
    text_sf = generated_header('generate_qdpt_evaluators.py') + header.format(
        what='spin-free (spatial)',
        extra=',\nwith the antisymmetrized-diagram terms collapsed to spatial-orbital\n'
              'tensors and summed over spin')
    for pt, v in COMBINATIONS:
        for spin_free in (False, True):
            t_start = time.time()
            what = 'spin-free' if spin_free else 'spin-orbital'
            print(f"Evaluating Wick's diagrams for PT={pt}, max_vertex={v} ({what})...", flush=True)
            terms = run_wick_contractions(max_pt_order=pt, max_vertex=v, include_f=False,
                                          print_unique=False, spin_free=spin_free)
            print(f"Diagram evaluation took {time.time() - t_start:.2f}s", flush=True)
            code = generate_function(pt, v, terms, optimize_factorization=True, spin_free=spin_free) + "\n\n"
            if spin_free:
                text_sf += code
            else:
                text += code
    write_source(os.path.join(root, 'generated_evaluators.py'), text)
    write_source(os.path.join(root, 'generated_evaluators_spinfree.py'), text_sf)


if __name__ == '__main__':
    generate()
