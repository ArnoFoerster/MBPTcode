import os
import sys
import re
import itertools
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.CodeGeneration.qdpt_wick import run_wick_contractions, Term

VIRS  = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
CORES = ['i', 'j', 'k', 'l', 'm', 'n', 'o']
ACTS  = ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z']
TARGET_EXT_INDICES = {
    0: [],
    1: ['p', 'q'],
    2: ['p', 'q', 'r', 's'],
    3: ['p', 'q', 'r', 's', 't', 'u'],
    4: ['p', 'q', 'r', 's', 't', 'u', 'v', 'w'],
    5: ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z'],
}

def get_slice_str_normal(idx):
    if idx in CORES: return 'occ_idx'
    if idx in ACTS:  return 'act_idx'
    if idx in VIRS:  return 'virt_idx'
    raise ValueError(f"Unknown index {idx}")

def parse_operator_str(op_str):
    op_name, indices_str = op_str.split('(')
    indices = [idx.strip() for idx in indices_str.replace(')', '').split(',')]
    return op_name, indices

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
    # Collect all unique indices
    all_indices = set()
    for op in term.operators:
        _, idxs = parse_operator_str(op)
        all_indices.update(idxs)
    for denom in term.denom_slices:
        matches = re.finditer(r'e_([a-z0-9_]+)', denom)
        for m in matches:
            all_indices.add(m.group(1))
    all_indices = sorted(list(all_indices))
    char_map = {idx: chr(65 + i) for i, idx in enumerate(all_indices)}
    
    lines = []
    
    # generate blocks
    block_names = []
    block_einsums = []
    for i, op in enumerate(term.operators):
        op_name, idxs = parse_operator_str(op)
        slices = [get_slice_str_fn(idx) for idx in idxs]
        slice_str = ", ".join(slices)
        if spin_free:
            array_name = op_name
        else:
            array_name = "g_anti_spin" if op_name == 'g' else "h1_spin"
        lines.append(f"    b{i} = {array_name}[np.ix_({slice_str})]")
        block_names.append(f"b{i}")
        block_einsums.append("".join(char_map[idx] for idx in idxs))
        
    num_einsum_str = "".join(char_map[idx] for idx in all_indices)
    
    if len(term.denom_slices) == 0:
        target_ext = TARGET_EXT_INDICES[term.sector]
        out_str = "".join(char_map[idx] for idx in target_ext)
        rhs_einsum = ",".join(block_einsums) + "->" + out_str
        ops_str = ", ".join(block_names)
        
        if len(block_names) == 0:
            lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor}")
        else:
            lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor} * np.einsum('{rhs_einsum}', {ops_str}, optimize=True)")
    else:
        rhs_einsum = ",".join(block_einsums) + "->" + num_einsum_str
        ops_str = ", ".join(block_names)
        lines.append(f"    num = {term.prefactor} * np.einsum('{rhs_einsum}', {ops_str}, optimize=True)")
        
        for denom_idx, denom in enumerate(term.denom_slices):
            expr = generate_denom_expr(denom, all_indices, get_slice_str_fn)
            if expr:
                lines.append(f"    d{denom_idx} = {expr}")
                lines.append(f"    d{denom_idx}[np.abs(d{denom_idx}) < 1e-12] = 1e-12")
                lines.append(f"    num = num / d{denom_idx}")
            
        target_ext = TARGET_EXT_INDICES[term.sector]
        out_str = "".join(char_map[idx] for idx in target_ext)
        lines.append(f"    {target_dict_name}[{term.sector}] += np.einsum('{num_einsum_str}->{out_str}', num, optimize=True)")
        
    return "\n".join(lines)


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

def generate_term_code_factorized(term, target_dict_name, get_slice_str_fn, spin_free=False):
    all_indices = set()
    for op in term.operators:
        _, idxs = parse_operator_str(op)
        all_indices.update(idxs)
    for denom in term.denom_slices:
        matches = re.finditer(r'e_([a-z0-9_]+)', denom)
        for m in matches:
            all_indices.add(m.group(1))
    all_indices = sorted(list(all_indices))
    char_map = {idx: chr(65 + i) for i, idx in enumerate(all_indices)}
    
    target_ext = TARGET_EXT_INDICES[term.sector]
    denom_var_counter = [0]

    def next_denom_var():
        name = f"d{denom_var_counter[0]}"
        denom_var_counter[0] += 1
        return name

    if len(term.operators) == 0:
        lines = []
        lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor}")
        return "\n".join(lines)

    # Get optimal ordering and denominator assignments
    opt_perm, denom_steps, max_rank = find_optimal_operator_ordering(term, target_ext)
    ordered_ops = [term.operators[p] for p in opt_perm]

    sliced_indices = []
    if max_rank > 6:
        num_sliced = max_rank - 6
        candidates = set(all_indices) # Include target_ext!
        
        def sort_key(idx):
            if idx in target_ext:
                return (3, idx)
            if idx in VIRS:
                return (0, idx)
            if idx in CORES:
                return (1, idx)
            return (2, idx)
            
        sorted_candidates = sorted(list(candidates), key=sort_key)
        sliced_indices = sorted_candidates[:num_sliced]
        
    setup_lines = []
    
    ops_info = []
    for i, op in enumerate(ordered_ops):
        op_name, idxs = parse_operator_str(op)
        slices = [get_slice_str_fn(idx) for idx in idxs]
        slice_str = ", ".join(slices)
        if spin_free:
            array_name = op_name
        else:
            array_name = "g_anti_spin" if op_name == 'g' else "h1_spin"
        if len(sliced_indices) > 0:
            setup_lines.append(f"    b{i}_full = {array_name}[np.ix_({slice_str})]")
        else:
            setup_lines.append(f"    b{i} = {array_name}[np.ix_({slice_str})]")
        ops_info.append({'name': f"b{i}", 'indices': idxs})
        
    candidate_lines = []
    if len(sliced_indices) > 0:
        for i, op in enumerate(ordered_ops):
            op_name, idxs = parse_operator_str(op)
            slice_parts = []
            for idx in idxs:
                if idx in sliced_indices:
                    slice_parts.append(f"i_{idx}")
                else:
                    slice_parts.append(":")
            if any(part.startswith("i_") for part in slice_parts):
                slice_str = ", ".join(slice_parts)
                candidate_lines.append(f"    b{i} = b{i}_full[{slice_str}]")
            else:
                candidate_lines.append(f"    b{i} = b{i}_full")
                
    # Build target slice strings
    target_slice_parts = []
    for idx in target_ext:
        if idx in sliced_indices:
            target_slice_parts.append(f"i_{idx}")
        else:
            target_slice_parts.append(":")
    target_slice_str = ", ".join(target_slice_parts)
    
    target_ext_remaining = [idx for idx in target_ext if idx not in sliced_indices]
    final_einsum_str = "".join(char_map[idx] for idx in target_ext_remaining)
                
    if len(term.denom_slices) == 0:
        rhs_parts = []
        for op in ops_info:
            op_rem = [idx for idx in op['indices'] if idx not in sliced_indices]
            rhs_parts.append("".join(char_map[idx] for idx in op_rem))
        rhs_einsum = ",".join(rhs_parts)
        ops_str = ", ".join(op['name'] for op in ops_info)
        if len(ops_info) == 0:
            if target_slice_str:
                candidate_lines.append(f"    {target_dict_name}[{term.sector}][{target_slice_str}] += {term.prefactor}")
            else:
                candidate_lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor}")
        else:
            if target_slice_str:
                candidate_lines.append(f"    {target_dict_name}[{term.sector}][{target_slice_str}] += {term.prefactor} * np.einsum('{rhs_einsum}->{final_einsum_str}', {ops_str}, optimize=True)")
            else:
                candidate_lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor} * np.einsum('{rhs_einsum}->{final_einsum_str}', {ops_str}, optimize=True)")
    else:
        # Check if there are any denoms associated with step 0
        denoms_at_step_0 = []
        for d_idx, step_associated in denom_steps.items():
            if step_associated == 0:
                denom_str = term.denom_slices[d_idx]
                matches = re.finditer(r'e_([a-z0-9_]+)', denom_str)
                d_idxs = [m.group(1) for m in matches]
                denoms_at_step_0.append({'str': denom_str, 'indices': d_idxs})
                
        if denoms_at_step_0:
            b0_indices_remaining = [idx for idx in ops_info[0]['indices'] if idx not in sliced_indices]
            for denom in denoms_at_step_0:
                expr = generate_denom_expr(denom['str'], b0_indices_remaining, get_slice_str_fn, sliced_indices)
                dvar = next_denom_var()
                candidate_lines.append(f"    {dvar} = {expr}")
                candidate_lines.append(f"    {dvar}[np.abs({dvar}) < 1e-12] = 1e-12")
                candidate_lines.append(f"    b0 = b0 / {dvar}")
                
        current_tensor = ops_info[0]
        
        for i in range(1, len(ops_info)):
            next_op = ops_info[i]
            
            A_indices_set = set(current_tensor['indices']) | set(next_op['indices'])
            
            future_indices = set(target_ext)
            for j in range(i+1, len(ops_info)):
                future_indices.update(ops_info[j]['indices'])
            for d_idx, step_associated in denom_steps.items():
                if step_associated > i:
                    matches = re.finditer(r'e_([a-z0-9_]+)', term.denom_slices[d_idx])
                    future_indices.update(m.group(1) for m in matches)
                    
            sum_indices = A_indices_set - future_indices
            out_indices = A_indices_set - sum_indices
            out_list = sorted(list(out_indices))
            
            current_remaining = [idx for idx in current_tensor['indices'] if idx not in sliced_indices]
            next_remaining = [idx for idx in next_op['indices'] if idx not in sliced_indices]
            A_ops_remaining = [idx for idx in A_indices_set if idx not in sliced_indices]
            A_ops_list_remaining = sorted(A_ops_remaining)
            
            lhs_einsum = "".join(char_map[idx] for idx in current_remaining) + "," + "".join(char_map[idx] for idx in next_remaining)
            num_einsum_str = "".join(char_map[idx] for idx in A_ops_list_remaining)
            
            pref_str = f"{term.prefactor} * " if i == 1 else ""
            candidate_lines.append(f"    num = {pref_str}np.einsum('{lhs_einsum}->{num_einsum_str}', {current_tensor['name']}, {next_op['name']}, optimize=True)")
            
            denoms_at_step = []
            for d_idx, step_associated in denom_steps.items():
                if step_associated == i:
                    denom_str = term.denom_slices[d_idx]
                    matches = re.finditer(r'e_([a-z0-9_]+)', denom_str)
                    d_idxs = [m.group(1) for m in matches]
                    denoms_at_step.append({'str': denom_str, 'indices': d_idxs})
                    
            A_denom_set = A_indices_set.copy()
            for denom in denoms_at_step:
                A_denom_set.update(denom['indices'])
                
            A_denom_list = sorted(list(A_denom_set))
            A_denom_list_remaining = [idx for idx in A_denom_list if idx not in sliced_indices]
            
            if denoms_at_step:
                view_slices = []
                A_indices_remaining = [idx for idx in A_indices_set if idx not in sliced_indices]
                for idx in A_denom_list_remaining:
                    if idx in A_indices_remaining:
                        view_slices.append(":")
                    else:
                        view_slices.append("None")
                if "None" in view_slices:
                    candidate_lines.append(f"    num = num[{', '.join(view_slices)}]")
                    
                for denom in denoms_at_step:
                    expr = generate_denom_expr(denom['str'], A_denom_list_remaining, get_slice_str_fn, sliced_indices)
                    dvar = next_denom_var()
                    candidate_lines.append(f"    {dvar} = {expr}")
                    candidate_lines.append(f"    {dvar}[np.abs({dvar}) < 1e-12] = 1e-12")
                    candidate_lines.append(f"    num = num / {dvar}")

            sum_indices = A_denom_set - future_indices
            out_indices = A_denom_set - sum_indices
            out_list = sorted(list(out_indices))
            
            out_list_remaining = [idx for idx in out_list if idx not in sliced_indices]
            sum_indices_remaining = set(A_denom_list_remaining) - set(out_list_remaining)
            
            if len(sum_indices_remaining) == 0:
                candidate_lines.append(f"    I{i} = num")
            else:
                in_einsum_str = "".join(char_map[idx] for idx in A_denom_list_remaining)
                out_einsum_str = "".join(char_map[idx] for idx in out_list_remaining)
                candidate_lines.append(f"    I{i} = np.einsum('{in_einsum_str}->{out_einsum_str}', num, optimize=True)")
                
            current_tensor = {'name': f"I{i}", 'indices': out_list}
            
        current_remaining = [idx for idx in current_tensor['indices'] if idx not in sliced_indices]
        if target_slice_str:
            candidate_lines.append(f"    {target_dict_name}[{term.sector}][{target_slice_str}] += np.einsum('{''.join(char_map[idx] for idx in current_remaining)}->{final_einsum_str}', {current_tensor['name']}, optimize=True)")
        else:
            candidate_lines.append(f"    {target_dict_name}[{term.sector}] += np.einsum('{''.join(char_map[idx] for idx in current_remaining)}->{final_einsum_str}', {current_tensor['name']}, optimize=True)")
        
    loop_dependent_vars = set(f"i_{idx}" for idx in sliced_indices)
    loop_body_lines = []
    
    for line in candidate_lines:
        words = re.findall(r'[a-zA-Z0-9_]+', line)
        if '=' in line:
            rhs = line.split('=', 1)[1]
            rhs_words = re.findall(r'[a-zA-Z0-9_]+', rhs)
            is_dep = any(w in loop_dependent_vars for w in rhs_words)
        else:
            is_dep = any(w in loop_dependent_vars for w in words)
            
        if '[' in line.split('=', 1)[0] if '=' in line else False:
            lhs_var = re.match(r'^\s*([a-zA-Z0-9_]+)', line).group(1)
            if lhs_var in loop_dependent_vars:
                is_dep = True
                
        match = re.match(r'^\s*([a-zA-Z0-9_]+)\s*=', line)
        if match:
            lhs = match.group(1)
            if is_dep:
                loop_dependent_vars.add(lhs)
            else:
                if lhs in loop_dependent_vars and lhs not in (f"i_{idx}" for idx in sliced_indices):
                    loop_dependent_vars.remove(lhs)
                    
        if is_dep:
            loop_body_lines.append(line)
        else:
            setup_lines.append(line)
        
    if len(sliced_indices) > 0:
        loop_headers = []
        indent = ""
        for idx in sliced_indices:
            slice_name = get_slice_str_fn(idx)
            loop_headers.append(f"    {indent}for i_{idx} in range(len({slice_name})):")
            indent += "    "
            
        indented_body = []
        for line in loop_body_lines:
            if line.strip():
                stripped = line[4:] if line.startswith("    ") else line
                indented_body.append(f"    {indent}{stripped}")
            else:
                indented_body.append(line)
                
        return "\n".join(setup_lines + loop_headers + indented_body)
        
    return "\n".join(setup_lines + loop_body_lines)

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
        t = get_index_type(idx)
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

def get_index_type(idx):
    if idx in CORES: return 'occ'
    if idx in VIRS:  return 'vir'
    return 'act'

def generate_term_code_cached(term, target_dict_name, get_slice_str_fn, intermediate_id, is_first_in_group, use_counts, spin_free=False):
    all_indices = set()
    for op in term.operators:
        _, idxs = parse_operator_str(op)
        all_indices.update(idxs)
    for denom in term.denom_slices:
        matches = re.finditer(r'e_([a-z0-9_]+)', denom)
        for m in matches:
            all_indices.add(m.group(1))
    all_indices = sorted(list(all_indices))
    char_map = {idx: chr(65 + i) for i, idx in enumerate(all_indices)}
    
    target_ext = TARGET_EXT_INDICES[term.sector]
    is_i1_sliced = False
    denom_var_counter = [0]

    def next_denom_var():
        name = f"d{denom_var_counter[0]}"
        denom_var_counter[0] += 1
        return name

    if len(term.operators) == 0:
        lines = []
        lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor}")
        return "\n".join(lines)

    opt_perm, denom_steps, max_rank = find_optimal_operator_ordering(term, target_ext)
    ordered_ops = [term.operators[p] for p in opt_perm]
    
    # Calculate number of virtual indices in the term
    vir_indices = [idx for idx in all_indices if idx in VIRS]
    
    sliced_indices = []
    if max_rank > 6 or len(vir_indices) > 1:
        num_sliced = max(max_rank - 6, len(vir_indices) - 1)
        candidates = set(all_indices)
        def sort_key(idx):
            if idx in target_ext:
                return (3, idx)
            if idx in VIRS:
                return (0, idx)
            if idx in CORES:
                return (1, idx)
            return (2, idx)
        sorted_candidates = sorted(list(candidates), key=sort_key)
        sliced_indices = sorted_candidates[:num_sliced]
        
    setup_lines = []
    
    ops_info = []
    for i, op in enumerate(ordered_ops):
        op_name, idxs = parse_operator_str(op)
        slices = [get_slice_str_fn(idx) for idx in idxs]
        slice_str = ", ".join(slices)
        if spin_free:
            array_name = op_name
        else:
            array_name = "g_anti_spin" if op_name == 'g' else "h1_spin"
        
        # We always want the full versions defined for external operators (e.g. b2)
        if len(sliced_indices) > 0:
            setup_lines.append(f"    b{i}_full = {array_name}[np.ix_({slice_str})]")
        else:
            setup_lines.append(f"    b{i} = {array_name}[np.ix_({slice_str})]")
        ops_info.append({'name': f"b{i}", 'indices': idxs})
        
    def make_slice_line(idx_in_ordered, op_str, sliced_indices_list):
        op_name, idxs = parse_operator_str(op_str)
        slice_parts = []
        for idx in idxs:
            if idx in sliced_indices_list:
                slice_parts.append(f"i_{idx}")
            else:
                slice_parts.append(":")
        if any(part.startswith("i_") for part in slice_parts):
            slice_str = ", ".join(slice_parts)
            return f"    b{idx_in_ordered} = b{idx_in_ordered}_full[{slice_str}]"
        else:
            return f"    b{idx_in_ordered} = b{idx_in_ordered}_full"

    # Define variables to hold split sliced indices
    cache_sliced_indices = []
    loop_sliced_indices = []
    
    candidate_lines = []
    # If we have sliced indices, we will define b0 and b1 from full inside cache check (or loop)
    # We will compute loop_sliced_indices and cache_sliced_indices below
            
    # Generate the computation logic lines
    # For cached term, we assume term.operators >= 2
    denoms_at_step_0 = []
    for d_idx, step_associated in denom_steps.items():
        if step_associated == 0:
            denom_str = term.denom_slices[d_idx]
            matches = re.finditer(r'e_([a-z0-9_]+)', denom_str)
            d_idxs = [m.group(1) for m in matches]
            denoms_at_step_0.append({'str': denom_str, 'indices': d_idxs})
            
    if denoms_at_step_0:
        b0_indices_remaining = ops_info[0]['indices'] # handled later dynamically
        
    current_tensor = ops_info[0]
    
    for i in range(1, len(ops_info)):
        next_op = ops_info[i]
        
        # We will split sliced_indices on the first step (i=1)
        A_indices_set = set(current_tensor['indices']) | set(next_op['indices'])
        future_indices = set(target_ext)
        for j in range(i+1, len(ops_info)):
            future_indices.update(ops_info[j]['indices'])
        for d_idx, step_associated in denom_steps.items():
            if step_associated > i:
                matches = re.finditer(r'e_([a-z0-9_]+)', term.denom_slices[d_idx])
                future_indices.update(m.group(1) for m in matches)
                
        sum_indices = A_indices_set - future_indices
        out_indices = A_indices_set - sum_indices
        out_list = sorted(list(out_indices))
        
        if i == 1:
            i1_out_list = out_list.copy() if isinstance(out_list, list) else list(out_indices)
            if len(sliced_indices) > 0:
                # Any sliced index in the first step is looped inside the cache block to populate I1 slice-by-slice
                step1_all_indices = set(current_tensor['indices']) | set(next_op['indices'])
                cache_sliced_indices = [idx for idx in sliced_indices if idx in step1_all_indices]
                loop_sliced_indices = [idx for idx in sliced_indices if idx in out_list]
            else:
                cache_sliced_indices = []
                loop_sliced_indices = []
                
        # step_sliced_indices represents indices sliced at this step
        step_sliced_indices = cache_sliced_indices if i < 2 else loop_sliced_indices
        
        if len(sliced_indices) > 0:
            if i < 2:
                # We define b0 and b1 inside the cache check (potentially using cache_sliced_indices)
                pass
            else:
                candidate_lines.append(make_slice_line(i, ordered_ops[i], loop_sliced_indices))
            
        current_remaining = [idx for idx in current_tensor['indices'] if idx not in step_sliced_indices]
        next_remaining = [idx for idx in next_op['indices'] if idx not in step_sliced_indices]
        A_ops_remaining = [idx for idx in A_indices_set if idx not in step_sliced_indices]
        A_ops_list_remaining = sorted(A_ops_remaining)
        
        lhs_einsum = "".join(char_map[idx] for idx in current_remaining) + "," + "".join(char_map[idx] for idx in next_remaining)
        num_einsum_str = "".join(char_map[idx] for idx in A_ops_list_remaining)
        
        # Omit prefactor from intermediate! We multiply it at the very end when updating target
        candidate_lines.append(f"    num = np.einsum('{lhs_einsum}->{num_einsum_str}', {current_tensor['name']}, {next_op['name']}, optimize=True)")
        
        denoms_at_step = []
        for d_idx, step_associated in denom_steps.items():
            if step_associated == i:
                denom_str = term.denom_slices[d_idx]
                matches = re.finditer(r'e_([a-z0-9_]+)', denom_str)
                d_idxs = [m.group(1) for m in matches]
                denoms_at_step.append({'str': denom_str, 'indices': d_idxs})
                
        A_denom_set = A_indices_set.copy()
        for denom in denoms_at_step:
            A_denom_set.update(denom['indices'])
            
        A_denom_list = sorted(list(A_denom_set))
        A_denom_list_remaining = [idx for idx in A_denom_list if idx not in step_sliced_indices]
        
        if denoms_at_step:
            view_slices = []
            A_indices_remaining = [idx for idx in A_indices_set if idx not in step_sliced_indices]
            for idx in A_denom_list_remaining:
                if idx in A_indices_remaining:
                    view_slices.append(":")
                else:
                    view_slices.append("None")
            if "None" in view_slices:
                candidate_lines.append(f"    num = num[{', '.join(view_slices)}]")
                
            for denom in denoms_at_step:
                expr = generate_denom_expr(denom['str'], A_denom_list_remaining, get_slice_str_fn, step_sliced_indices)
                dvar = next_denom_var()
                candidate_lines.append(f"    {dvar} = {expr}")
                candidate_lines.append(f"    {dvar}[np.abs({dvar}) < 1e-12] = 1e-12")
                candidate_lines.append(f"    num = num / {dvar}")
                
        sum_indices = A_denom_set - future_indices
        out_indices = A_denom_set - sum_indices
        out_list = sorted(list(out_indices))
        
        out_list_remaining = [idx for idx in out_list if idx not in step_sliced_indices]
        sum_indices_remaining = set(A_denom_list_remaining) - set(out_list_remaining)
        
        if len(sum_indices_remaining) == 0:
            candidate_lines.append(f"    I{i} = num")
        else:
            in_einsum_str = "".join(char_map[idx] for idx in A_denom_list_remaining)
            out_einsum_str = "".join(char_map[idx] for idx in out_list_remaining)
            candidate_lines.append(f"    I{i} = np.einsum('{in_einsum_str}->{out_einsum_str}', num, optimize=True)")
            
        # Update current tensor. If we just computed I1 and have sliced indices, we append the slice line
        # and then current_tensor's indices become the remaining ones.
        if i == 1 and len(loop_sliced_indices) > 0:
            slice_parts = []
            for idx in out_list:
                if idx in loop_sliced_indices:
                    slice_parts.append(f"i_{idx}")
                else:
                    slice_parts.append(":")
            if any(part.startswith("i_") for part in slice_parts):
                slice_str = ", ".join(slice_parts)
                candidate_lines.append(f"    I1 = _cache['int_{intermediate_id}'][{slice_str}]")
                is_i1_sliced = True
                current_tensor = {'name': f"I{i}", 'indices': out_list_remaining}
            else:
                current_tensor = {'name': f"I{i}", 'indices': out_list}
        else:
            current_tensor = {'name': f"I{i}", 'indices': out_list}
        
    # Split candidate_lines at I1's definition
    i1_line_idx = -1
    for idx, line in enumerate(candidate_lines):
        if re.search(r'\bI1\s*=', line):
            i1_line_idx = idx
            break
            
    if i1_line_idx == -1:
        raise ValueError(f"Could not find I1 definition in term {term.raw_str}")
        
    i1_comp_lines = candidate_lines[:i1_line_idx+1]
    rest_comp_lines = candidate_lines[i1_line_idx+1:]
    
    # Define setup_lines for b0_full, b1_full if loop sliced (already added above)
    # Define target slice string using loop_sliced_indices only
    target_slice_parts = []
    for idx in target_ext:
        if idx in loop_sliced_indices:
            target_slice_parts.append(f"i_{idx}")
        else:
            target_slice_parts.append(":")
    target_slice_str = ", ".join(target_slice_parts)
    
    target_ext_remaining = [idx for idx in target_ext if idx not in loop_sliced_indices]
    final_einsum_str = "".join(char_map[idx] for idx in target_ext_remaining)
    
    # Add final target accumulation to rest_comp_lines (with prefactor!)
    current_remaining = [idx for idx in current_tensor['indices'] if idx not in loop_sliced_indices]
    if target_slice_str:
        rest_comp_lines.append(f"    {target_dict_name}[{term.sector}][{target_slice_str}] += {term.prefactor} * np.einsum('{''.join(char_map[idx] for idx in current_remaining)}->{final_einsum_str}', {current_tensor['name']}, optimize=True)")
    else:
        rest_comp_lines.append(f"    {target_dict_name}[{term.sector}] += {term.prefactor} * np.einsum('{''.join(char_map[idx] for idx in current_remaining)}->{final_einsum_str}', {current_tensor['name']}, optimize=True)")
        
    # Process i1_comp_lines (which are loop-invariant) into setup_lines if first in group
    setup_body = []
    if is_first_in_group:
        setup_body.append(f"    if 'int_{intermediate_id}' not in _cache:")
        
        # If cache_sliced_indices is not empty, we initialize I1 to zeros and loop
        indent = "        "
        if cache_sliced_indices:
            shape_str = ", ".join(f"len({get_slice_str_fn(idx)})" for idx in i1_out_list)
            setup_body.append(f"        I1 = np.zeros(({shape_str}))")
            for idx in cache_sliced_indices:
                slice_name = get_slice_str_fn(idx)
                setup_body.append(f"{indent}for i_{idx} in range(len({slice_name})):")
                indent += "    "
                
        # Inside the cache check block (potentially indented)
        # Define b0 and b1
        if len(sliced_indices) > 0:
            for i in range(2):
                op_name, idxs = parse_operator_str(ordered_ops[i])
                slice_parts = []
                for idx in idxs:
                    if idx in cache_sliced_indices:
                        slice_parts.append(f"i_{idx}")
                    else:
                        slice_parts.append(":")
                slice_str = ", ".join(slice_parts)
                setup_body.append(f"{indent}b{i} = b{i}_full[{slice_str}]")
        else:
            for i in range(2):
                setup_body.append(f"{indent}b{i} = b{i}")
                
        # Add denoms_at_step_0 evaluation if any
        if denoms_at_step_0:
            b0_indices_remaining = [idx for idx in ops_info[0]['indices'] if idx not in cache_sliced_indices]
            for denom in denoms_at_step_0:
                expr = generate_denom_expr(denom['str'], b0_indices_remaining, get_slice_str_fn, cache_sliced_indices)
                dvar = next_denom_var()
                setup_body.append(f"{indent}{dvar} = {expr}")
                setup_body.append(f"{indent}{dvar}[np.abs({dvar}) < 1e-12] = 1e-12")
                setup_body.append(f"{indent}b0 = b0 / {dvar}")

        for line in i1_comp_lines:
            # Skip denoms_at_step_0 if we evaluated it above
            if denoms_at_step_0 and (re.search(r'\bb0 / d\d+\b', line) or re.search(r'\bd\d+\[np\.abs', line) or "d = generate" in line):
                continue
            if cache_sliced_indices and re.match(r'^\s*I1\s*=', line):
                slice_parts = []
                for idx in i1_out_list:
                    if idx in cache_sliced_indices:
                        slice_parts.append(f"i_{idx}")
                    else:
                        slice_parts.append(":")
                if any(part != ":" for part in slice_parts):
                    slice_str = ", ".join(slice_parts)
                    line = line.replace("I1 =", f"I1[{slice_str}] +=")
                else:
                    line = line.replace("I1 =", "I1 +=")
            setup_body.append(f"{indent}{line.strip()}")
            
        setup_body.append(f"        _cache['int_{intermediate_id}'] = I1")
        
    if not is_i1_sliced:
        setup_body.append(f"    I1 = _cache['int_{intermediate_id}']")
    
    # Process rest_comp_lines (which are loop-dependent)
    loop_dependent_vars = set(f"i_{idx}" for idx in loop_sliced_indices)
    loop_body_lines = []
    post_setup_lines = []
    
    for line in rest_comp_lines:
        words = re.findall(r'[a-zA-Z0-9_]+', line)
        is_dep = any(w in loop_dependent_vars for w in words)
                
        match = re.match(r'^\s*([a-zA-Z0-9_]+)\s*=', line)
        if match:
            lhs = match.group(1)
            if is_dep:
                loop_dependent_vars.add(lhs)
            else:
                if lhs in loop_dependent_vars and lhs not in (f"i_{idx}" for idx in loop_sliced_indices):
                    loop_dependent_vars.remove(lhs)
                    
        if is_dep:
            loop_body_lines.append(line)
        else:
            post_setup_lines.append(line)
            
    # Add cleanup to end of term block
    cleanup_lines = []
    cleanup_lines.append(f"    _use_count['int_{intermediate_id}'] -= 1")
    cleanup_lines.append(f"    if _use_count['int_{intermediate_id}'] == 0:")
    cleanup_lines.append(f"        del _cache['int_{intermediate_id}']")
    
    if len(loop_sliced_indices) > 0:
        loop_headers = []
        indent = ""
        for idx in loop_sliced_indices:
            slice_name = get_slice_str_fn(idx)
            loop_headers.append(f"    {indent}for i_{idx} in range(len({slice_name})):")
            indent += "    "
            
        indented_body = []
        for line in loop_body_lines:
            if line.strip():
                stripped = line[4:] if line.startswith("    ") else line
                indented_body.append(f"    {indent}{stripped}")
            else:
                indented_body.append(line)
        if not indented_body or all(not line.strip() for line in indented_body):
            indented_body.append(f"    {indent}pass")
                 
        return "\n".join(setup_lines + setup_body + post_setup_lines + loop_headers + indented_body + cleanup_lines)
        
    return "\n".join(setup_lines + setup_body + post_setup_lines + loop_body_lines + cleanup_lines)

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
        
    import itertools
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
        
    from src.CodeGeneration.qdpt_wick import get_skeleton_hash, Term
    
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
        all_indices = set()
        for op in term.operators:
            _, idxs = parse_operator_str(op)
            all_indices.update(idxs)
        for denom in term.denom_slices:
            matches = re.finditer(r'e_([a-z0-9_]+)', denom)
            for m in matches:
                all_indices.add(m.group(1))
                
        from itertools import product
        unique_indices = sorted(list(all_indices))
        
        index_choices = {}
        for idx in unique_indices:
            t = get_index_type(idx)
            if t == 'occ':
                index_choices[idx] = ['occ_idx', 'act_occ_idx']
            elif t == 'vir':
                index_choices[idx] = ['virt_idx', 'act_vir_idx']
            else:
                index_choices[idx] = ['act_idx']
                
        keys = list(index_choices.keys())
        values = list(index_choices.values())
        
        for p in product(*values):
            mapping = dict(zip(keys, p))
            has_active = any(val in ['act_occ_idx', 'act_vir_idx', 'act_idx'] for val in mapping.values())
            if not has_active:
                continue
                
            def get_slice_assign(idx):
                return mapping[idx]
                
            lines.append(f"    # Term: {term.raw_str}")
            lines.append(f"    # Assignment: {mapping}")
            if len(all_indices) > 6 and optimize_factorization:
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
        from src.CodeGeneration.qdpt_wick import get_external_signature
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
        
        all_indices = set()
        for op in base_term.operators:
            _, idxs = parse_operator_str(op)
            all_indices.update(idxs)
        for denom in base_term.denom_slices:
            matches = re.finditer(r'e_([a-z0-9_]+)', denom)
            for m in matches:
                all_indices.add(m.group(1))
        all_indices = sorted(list(all_indices))
        
        vir_indices = [idx for idx in all_indices if idx in VIRS]
        
        sliced_indices = []
        if max_rank > 6 or len(vir_indices) > 1:
            num_sliced = max(max_rank - 6, len(vir_indices) - 1)
            candidates = set(all_indices)
            def sort_key(idx):
                if idx in target_ext:
                    return (3, idx)
                if idx in VIRS:
                    return (0, idx)
                if idx in CORES:
                    return (1, idx)
                return (2, idx)
            sorted_candidates = sorted(list(candidates), key=sort_key)
            sliced_indices = sorted_candidates[:num_sliced]
            
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
    from src.CodeGeneration.qdpt_wick import get_external_signature
    
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

if __name__ == '__main__':
    combinations = [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ]

    # 2026-07-17: generate_term_code_cached's per-step denominators all shared
    # the variable name `d`, which the loop-splitting dependency classifier
    # could hoist one occurrence of out of a sliced-index loop while leaving a
    # later same-named occurrence inside it -- the loop-invariant `d` got
    # clobbered by the loop-dependent one after the first iteration. Fixed by
    # giving every denominator its own d0, d1, ... variable. Verified: fresh
    # regeneration now matches the committed generated_evaluators.py to 1e-16
    # across (pt,max_vertex) in {2,3}x{2,3}, shift in {None, 0.1}.
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../MultiReference/QDPT/generated_evaluators.py'))
    output_path_spinfree = os.path.abspath(os.path.join(os.path.dirname(__file__), '../MultiReference/QDPT/generated_evaluators_spinfree.py'))
    
    header = ('"""QDPT active-space effective Hamiltonian g_eff (sectors 0 to max_vertex), {what}.\n\n'
              'Machine-generated from the Wick-contracted MP2/MP3 downfolding diagrams, for\n'
              'max vertex rank 2 or 3{extra}. Do not edit by hand. Called from the QDPT driver,\n'
              'perturbative.build_effective_hamiltonian.\n"""\nimport numpy as np\n\n')
    with open(output_path, 'w') as f:
        f.write(header.format(what='spin-orbital', extra=''))
    with open(output_path_spinfree, 'w') as f:
        f.write(header.format(what='spin-free (spatial)',
                              extra=',\nwith the antisymmetrized-diagram terms collapsed to spatial-orbital\n'
                                    'tensors and summed over spin'))

    for pt, v in combinations:
        import time
        t_start = time.time()
        print(f"Evaluating Wick's diagrams for PT={pt}, max_vertex={v} (spin-orbital)...", flush=True)
        terms = run_wick_contractions(max_pt_order=pt, max_vertex=v, include_f=False, print_unique=False, spin_free=False)
        print(f"Diagram evaluation took {time.time() - t_start:.2f}s", flush=True)
        
        print(f"Generating spin-orbital code for PT={pt}, max_vertex={v}...", flush=True)
        code_str = generate_function(pt, v, terms, optimize_factorization=True, spin_free=False)
        with open(output_path, 'a') as f:
            f.write(code_str + "\n\n")
            
        t_start = time.time()
        print(f"Evaluating Wick's diagrams for PT={pt}, max_vertex={v} (spin-free)...", flush=True)
        terms_sf = run_wick_contractions(max_pt_order=pt, max_vertex=v, include_f=False, print_unique=False, spin_free=True)
        print(f"Diagram evaluation took {time.time() - t_start:.2f}s", flush=True)
        
        print(f"Generating spin-free code for PT={pt}, max_vertex={v}...", flush=True)
        code_str_sf = generate_function(pt, v, terms_sf, optimize_factorization=True, spin_free=True)
        with open(output_path_spinfree, 'a') as f:
            f.write(code_str_sf + "\n\n")

    print(f"Generated evaluators saved to {output_path} and {output_path_spinfree}")
