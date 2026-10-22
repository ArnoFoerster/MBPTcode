"""Print the einsum code of the MP amplitude equations T2^(2), T3^(2) and
T1^(3) (stdout only; nothing is written)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.Base.pdaggerq_interface import pq_helper

def get_slice_name(idx_char):
    if idx_char in ['i', 'j', 'k', 'l', 'm', 'n', 'o']: return 'occ'
    if idx_char in ['a', 'b', 'c', 'd', 'e', 'f']: return 'virt'
    raise ValueError(f"Unknown index type: {idx_char}")

def generate_einsum(term, target_ext, var_name, is_gamma=False):
    pref_str = term[0]
    pref = float(pref_str)
    
    perms = []
    ops = []
    for piece in term[1:]:
        if piece.startswith('P('):
            idx1, idx2 = piece[2:-1].split(',')
            perms.append((idx1, idx2))
        else:
            ops.append(piece)
            
    op_vars = []
    einsum_in = []
    for op in ops:
        if op.startswith('<'):
            indices = op[1:-1].split('||')
            p, q = indices[0].split(',')
            r, s = indices[1].split(',')
            idxs = [p, q, r, s]
            slices = [get_slice_name(idx) for idx in idxs]
            slice_str = ", ".join(slices)
            op_vars.append(f"g_anti[{slice_str}]")
            einsum_in.append(idxs)
        elif op.startswith('t1('):
            idxs = op[3:-1].split(',')
            op_vars.append("t1_2") 
            einsum_in.append(idxs)
        elif op.startswith('t2('):
            idxs = op[3:-1].split(',')
            op_vars.append("t2_1")
            einsum_in.append(idxs)
        elif op.startswith('t3('):
            idxs = op[3:-1].split(',')
            op_vars.append("t3_2")
            einsum_in.append(idxs)

    all_idx = []
    for idxs in einsum_in:
        all_idx.extend(idxs)
    all_idx.extend(target_ext)
    all_idx = sorted(list(set(all_idx)))
    char_map = {orig: chr(65 + i) for i, orig in enumerate(all_idx)}
    
    in_str = ",".join(["".join([char_map[c] for c in idxs]) for idxs in einsum_in])
    out_str = "".join([char_map[c] for c in target_ext])
    
    args_str = ", ".join(op_vars)
    base_expr = f"{pref} * np.einsum('{in_str}->{out_str}', {args_str}, optimize=True)"
    
    lines = []
    lines.append(f"    raw = {base_expr}")
    
    perm_expr = "raw"
    for p1, p2 in perms:
        ax1 = target_ext.index(p1)
        ax2 = target_ext.index(p2)
        axes = list(range(len(target_ext)))
        axes[ax1] = ax2
        axes[ax2] = ax1
        perm_expr = f"({perm_expr} - {perm_expr}.transpose({', '.join(map(str, axes))}))"
        
    lines.append(f"    {var_name} += {perm_expr}")
    return lines

def run_target(name, left_op, target_ext):
    print(f"--- {name} ---")
    pq = pq_helper('fermi')
    pq.set_print_level(0)
    pq.set_left_operators([left_op])
    pq.set_right_operators([[]])
    
    pq.add_commutator(1.0, ['v'], ['t2'])
    if name == 'T1^(3)':
        pq.add_commutator(1.0, ['v'], ['t1'])
        pq.add_commutator(1.0, ['v'], ['t3'])

    pq.simplify()
    terms = pq.strings()
    
    print(f"def evaluate_{name.lower().replace('^', '').replace('(', '').replace(')', '')}(g_anti, t1_2, t2_1, t3_2):")
    print(f"    # target indices: {target_ext}")
    print(f"    result = 0")
    for i, term in enumerate(terms):
        print(f"    # {term}")
        lines = generate_einsum(term, target_ext, "result")
        for l in lines:
            print(l)
    print("    return result\n")

if __name__ == "__main__":
    run_target("T2^(2)", ['e2(i,j,a,b)'], ['a', 'b', 'i', 'j'])
    run_target("T3^(2)", ['e3(i,j,k,a,b,c)'], ['a', 'b', 'c', 'i', 'j', 'k'])
    run_target("T1^(3)", ['e1(i,a)'], ['i', 'a'])
