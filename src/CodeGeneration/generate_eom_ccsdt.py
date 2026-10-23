"""Codegen script (run once, output committed): derives EOM-CCSDT sigma-vector equations with pdaggerq.

Uses pdaggerq_interface's raw_terms_to_einsum/assemble_function, same
pipeline that generated lambda*_residual.py/d1_blocks.py. Not imported at
runtime; re-run only to regenerate/extend these equations.

For each sector (IP/EA/EE) and rank R (1/2/3 = S/D/T-type manifold), generates
    sigma_R      = <mu_R| Hbar |r1+r2+r3>   (right matvec)
    sigma_R_left = <l1+l2+l3| Hbar |nu_R>   (left matvec)
consumed by eom.py's Davidson iterations. r0/l0 never contribute once the
manifold excludes rank 0 (those components are the ground-state T equations).

r/l array shapes (locked in by infer_free_indices + tests/test_eom_ccsdt.py):
    IP: r1(no,) r2(nv,no,no) r3(nv,nv,no,no,no); l1(no,) l2(no,no,nv) l3(no,no,no,nv,nv)
    EA: r1(nv,) r2(nv,nv,no) r3(nv,nv,nv,no,no); l1(nv,) l2(no,nv,nv) l3(no,no,nv,nv,nv)
    EE: r1(nv,no) r2(nv,nv,no,no) r3(nv,nv,nv,no,no,no); l analogous with occ/vir swapped

IP/EA ket projectors are the adjoint of the matching bra (reversed operator
order, creation<->annihilation flipped). EE uses the same e1/e2/e3 syntax for
both bra and ket (self-dual in pdaggerq's convention).
"""
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.pdaggerq_interface import pq_helper, assemble_function, infer_output_order
from src.CodeGeneration.codegen_common import (
    CC_OUT_DIR, DENSITY_BLOCKS, density_insert, density_out_idx, density_shape, generated_header, write_module)

OUT_DIR = os.path.join(CC_OUT_DIR, 'generated_eom')

_IMPORTS = ("import numpy as np\n\n"
            "from ..cached_einsum import einsum\n"
            "from ..parallel_accum import pacc\n")
_HEADER = generated_header('generate_eom_ccsdt.py', "See that module's docstring for the equations/conventions.")

R_ARGS = ['r1', 'r2', 'r3', 't1', 't2', 't3', 'f', 'g', 'o', 'v']
L_ARGS = ['l1', 'l2', 'l3', 't1', 't2', 't3', 'f', 'g', 'o', 'v']

IP_BRA = {
    1: ['a*(i)'],
    2: ['a*(i)', 'a*(j)', 'a(a)'],
    3: ['a*(i)', 'a*(j)', 'a*(k)', 'a(b)', 'a(a)'],
}
IP_KET = {
    1: ['a(i)'],
    2: ['a*(a)', 'a(j)', 'a(i)'],
    3: ['a*(a)', 'a*(b)', 'a(k)', 'a(j)', 'a(i)'],
}
EA_BRA = {
    1: ['a(a)'],
    2: ['a*(i)', 'a(b)', 'a(a)'],
    3: ['a*(i)', 'a*(j)', 'a(c)', 'a(b)', 'a(a)'],
}
EA_KET = {
    1: ['a*(a)'],
    2: ['a*(a)', 'a*(b)', 'a(i)'],
    3: ['a*(a)', 'a*(b)', 'a*(c)', 'a(j)', 'a(i)'],
}
EE_BRA = {
    # e1(p,q) = a*_p a_q always; a bra needs (occ,vir) here or it's the
    # trivially-zero <0|a*_vir... (a virtual creation can never survive to
    # the left of <0|). Confirmed empirically: e1(a,i) as bra -> 0 terms,
    # e1(i,a) -> 27 terms.
    1: ['e1(i,a)'],
    2: ['e2(i,j,a,b)'],
    3: ['e3(i,j,k,a,b,c)'],
}
EE_KET = {
    # As a ket, (vir,occ) is the physically sensible excitation operator
    # a*_vir a_occ |0> = |Phi_occ^vir>.
    1: ['e1(a,i)'],
    2: ['e2(a,b,i,j)'],
    3: ['e3(a,b,c,i,j,k)'],
}


def _generate(op_type, bra_or_none, right_or_none):
    """One pdaggerq call: <bra_or_none| Hbar |right_or_none>. Exactly one side is
    a manifold projector; the other is the generic r1/r2/r3 or l1/l2/l3 superposition."""
    pq = pq_helper('fermi')
    if op_type is not None:
        if bra_or_none is not None:
            pq.set_right_operators_type(op_type)   # generic R lives on the right
        else:
            pq.set_left_operators_type(op_type)    # generic L lives on the left
    if bra_or_none is not None:
        pq.set_left_operators([bra_or_none])
        pq.set_right_operators([['r1'], ['r2'], ['r3']])
    else:
        pq.set_left_operators([['l1'], ['l2'], ['l3']])
        pq.set_right_operators([right_or_none])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2', 't3'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings()
    pq.clear()
    return terms


def _right_sigma(op_type, bra_ops, rank):
    terms = _generate(op_type, bra_ops, None)
    # anchor axis order to how r{rank} is referenced as a leaf elsewhere --
    # NOT infer_free_indices' naive first-occurrence-in-term[0] heuristic,
    # which is unrelated to r{rank}'s own canonical slot convention and
    # silently produces a self-inconsistent sigma function (see
    # infer_output_order's docstring for how this was caught).
    out_idx = infer_output_order(terms, f'r{rank}', type_order=('vir', 'occ'))
    return terms, out_idx


def _left_sigma(op_type, ket_ops, rank):
    terms = _generate(op_type, None, ket_ops)
    out_idx = infer_output_order(terms, f'l{rank}', type_order=('occ', 'vir'))
    return terms, out_idx


SECTORS = {
    'ip': dict(op_type='IP', bra=IP_BRA, ket=IP_KET,
              right_shapes={1: '(no,)', 2: '(nv,no,no)', 3: '(nv,nv,no,no,no)'},
              left_shapes={1: '(no,)', 2: '(no,no,nv)', 3: '(no,no,no,nv,nv)'}),
    'ea': dict(op_type='EA', bra=EA_BRA, ket=EA_KET,
              right_shapes={1: '(nv,)', 2: '(nv,nv,no)', 3: '(nv,nv,nv,no,no)'},
              left_shapes={1: '(nv,)', 2: '(no,nv,nv)', 3: '(no,no,nv,nv,nv)'}),
    'ee': dict(op_type=None, bra=EE_BRA, ket=EE_KET,
              right_shapes={1: '(nv,no)', 2: '(nv,nv,no,no)', 3: '(nv,nv,nv,no,no,no)'},
              left_shapes={1: '(no,nv)', 2: '(no,no,nv,nv)', 3: '(no,no,no,nv,nv,nv)'}),
}


def generate_sector(name):
    spec = SECTORS[name]
    funcs = []
    for rank in (1, 2, 3):
        terms, out_idx = _right_sigma(spec['op_type'], spec['bra'][rank], rank)
        print(f"{name} right sigma{rank}: {len(terms)} terms, out_indices={out_idx}")
        funcs.append(assemble_function(f'{name}_sigma{rank}', R_ARGS, 'sigma',
                                       spec['right_shapes'][rank], terms, out_idx,
                                       accumulator='pacc'))

    for rank in (1, 2, 3):
        terms, out_idx = _left_sigma(spec['op_type'], spec['ket'][rank], rank)
        print(f"{name} left sigma{rank}: {len(terms)} terms, out_indices={out_idx}")
        funcs.append(assemble_function(f'{name}_sigma{rank}_left', L_ARGS, 'sigma',
                                       spec['left_shapes'][rank], terms, out_idx,
                                       accumulator='pacc'))

    write_module(os.path.join(OUT_DIR, f'{name}_sigma.py'), funcs, _HEADER, imports=_IMPORTS,
                 package_init=True)


# ---------------------------------------------------------------------------
# EE transition densities (for polarizabilities / G0W@CC, see
# src/SingleReference/GW/gw_cc.py). CC 1-body density matrices never
# involve f/g at all (only t/l amplitudes + the identity 'kd') -- the density
# operator is transformed by the T-similarity transform alone, unlike the
# sigma-vector equations above which sandwich the full Hamiltonian.
#
#   rho_n(pq)      = <(1+Lambda_gs)| e^-T p^dag q e^T |R_n>   (ground bra,
#                     excited ket -- Lambda_gs is the GROUND-STATE Lambda
#                     multiplier, R_n the n-th EOM-EE right eigenvector)
#   rho_star_n(pq) = <L_n| e^-T p^dag q e^T |1>                (excited bra
#                     -- L_n the n-th EOM-EE left eigenvector -- ground ket)
#
# Both use the SAME (vir,occ)-type r/l storage convention as the 'ee' sigma
# equations (confirmed: r1(a,i), l1(i,a) leaf patterns match exactly), so
# they consume EOMResult's own EE vectors directly with no conversion.
# ---------------------------------------------------------------------------
DENSITY_ARGS_RHO = ['lam1', 'lam2', 'lam3', 'r1', 'r2', 'r3', 't1', 't2', 't3', 'kd', 'o', 'v']
DENSITY_ARGS_RHO_STAR = ['l1', 'l2', 'l3', 't1', 't2', 't3', 'kd', 'o', 'v']



def _density_generate(left_ops, right_ops, insert_op):
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops)
    pq.set_right_operators(right_ops)
    pq.add_st_operator(1.0, [insert_op], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings()
    pq.clear()
    return terms


def generate_density():
    funcs = []
    # out_indices are the e1(p,q) insertion's own argument order (density_out_idx),
    # not a first-occurrence guess: the 'vv' block's two free indices are both
    # virtual, so letter type alone cannot tell rho[e,f] from rho[f,e]
    for block in DENSITY_BLOCKS:
        terms = _density_generate([['1'], ['l1'], ['l2'], ['l3']], [['r1'], ['r2'], ['r3']],
                                  density_insert(block))
        out_idx = density_out_idx(block)
        print(f"rho_{block}: {len(terms)} terms, out_indices={out_idx}")
        # rename l1/l2/l3 -> lam1/lam2/lam3 inside the raw terms too (ground
        # Lambda, distinct from the EOM left-eigenvector l1/l2/l3 used by
        # rho_star below and by the sigma equations elsewhere)
        renamed_terms = [[tok.replace('l1(', 'lam1(').replace('l2(', 'lam2(').replace('l3(', 'lam3(')
                          for tok in term] for term in terms]
        funcs.append(assemble_function(f'rho_{block}', DENSITY_ARGS_RHO, 'rho',
                                       density_shape(block), renamed_terms, out_idx,
                                       accumulator='pacc'))

    for block in DENSITY_BLOCKS:
        terms = _density_generate([['l1'], ['l2'], ['l3']], [['1']], density_insert(block))
        out_idx = density_out_idx(block)
        print(f"rho_star_{block}: {len(terms)} terms, out_indices={out_idx}")
        funcs.append(assemble_function(f'rho_star_{block}', DENSITY_ARGS_RHO_STAR, 'rho',
                                       density_shape(block), terms, out_idx,
                                       accumulator='pacc'))

    write_module(os.path.join(OUT_DIR, 'ee_density.py'), funcs, _HEADER, imports=_IMPORTS,
                 package_init=True)


def generate():
    """Write the ip/ea/ee sigma modules and ee_density.py into OUT_DIR."""
    for name in ('ip', 'ea', 'ee'):
        generate_sector(name)
    generate_density()


if __name__ == '__main__':
    generate()
