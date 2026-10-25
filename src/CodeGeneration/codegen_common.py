"""What the hand-written generators in src/CodeGeneration share.

One module writer and one header for every generated file, the index letters,
the spin-blocked leaf parameter lists, the one-particle density blocks, and
the small term utilities more than one generator needs. Generators import
these public names from here, never each other's private helpers.

    write_module(path, funcs, generated_header('generate_adcn.py', 'what it is'))

Letters: OCC_LETTERS / VIR_LETTERS are the ordered label pools
(precursor_classes, unbounded: i, j, ..., n, i7, i8, ...) from which
generators spell pdaggerq indices; OCC_FAMILY / VIR_FAMILY
(pdaggerq_interface) answer "is this label occupied / virtual" for any label
pdaggerq prints, including fresh einsum subscripts.
"""
import os
import re

from src.CodeGeneration.pdaggerq_interface import (
    OCC_LETTERS as OCC_FAMILY, VIR_LETTERS as VIR_FAMILY, OCC_SUBSCRIPTS, VIR_SUBSCRIPTS,
    parse_factor, array_expr)
from src.SingleReference.ADC.precursor_classes import OCC_LETTERS, VIR_LETTERS

# Output roots, read at call time (a test may point one at a temporary directory).
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADC_OUT_DIR = os.path.join(SRC_DIR, 'SingleReference', 'ADC', 'generatedADC')    # not committed
DM_OUT_DIR = os.path.join(SRC_DIR, 'SingleReference', 'DensityMatrix')          # generated_mpn*/
CC_OUT_DIR = os.path.join(SRC_DIR, 'SingleReference', 'CC')                     # generated_*/
QDPT_OUT_DIR = os.path.join(SRC_DIR, 'MultiReference', 'QDPT')

__all__ = [
    'OCC_LETTERS', 'VIR_LETTERS', 'OCC_FAMILY', 'VIR_FAMILY', 'OCC_SUBSCRIPTS', 'VIR_SUBSCRIPTS',
    'parse_factor', 'array_expr', 'SRC_DIR', 'ADC_OUT_DIR', 'DM_OUT_DIR', 'CC_OUT_DIR',
    'QDPT_OUT_DIR', 'EINSUM_IMPORTS', 'LAPLACE_IMPORTS', 'ERI_LEAVES', 'DF_LEAVES', 'R_LEAVES',
    'R_LEAVES_DF', 'R_DIMS', 'ADC_DIMS', 'U_DIMS', 'eri_leaves', 'r_leaves', 'DENSITY_BLOCKS',
    'density_insert', 'density_out_idx', 'density_shape', 'density_spin_labels',
    'generated_header', 'write_source', 'write_module', 'module_subdir', 'write_adc_module',
    'shape_expr', 'amp_leaves', 'signed_terms', 'term_has_factor', 'pq_terms']

EINSUM_IMPORTS = "import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
# Laplace-fused modules build their minimax grid in the function body.
LAPLACE_IMPORTS = EINSUM_IMPORTS + "from src.Base.utils.grids.grids import minimax_time_grid\n"

# Spin-blocked leaves. g_ssss are the antisymmetrized-within-spin / bare
# two-electron blocks; their DF/RI twin is the pair of 3-index factors B_aa/B_bb
# (df_codegen.dress_integral_factor). f_ss Fock, d_ss the Kronecker delta.
ERI_LEAVES = ['g_aaaa', 'g_abab', 'g_bbbb']
DF_LEAVES = ['B_aa', 'B_bb']
R_LEAVES = ['f_aa', 'f_bb'] + ERI_LEAVES + ['d_aa', 'd_bb']
R_LEAVES_DF = ['f_aa', 'f_bb'] + DF_LEAVES + ['d_aa', 'd_bb']
R_DIMS = ['o', 'v', 'nv', 'no']        # MPn / Laplace signatures
ADC_DIMS = ['o', 'v', 'no', 'nv']      # ADC block and apply signatures
U_DIMS = ['o_a', 'v_a', 'o_b', 'v_b', 'nv_a', 'no_a', 'nv_b', 'no_b']


def eri_leaves(df):
    """The two-electron leaves of a restricted signature: B_aa/B_bb with DF."""
    return list(DF_LEAVES if df else ERI_LEAVES)


def r_leaves(df):
    """f, two-electron and delta leaves of a restricted ADC block signature."""
    return list(R_LEAVES_DF if df else R_LEAVES)


# One-particle density blocks: block -> its two letters; the pdaggerq
# insertion is e1(<letters>), the output axes are the letters.
DENSITY_BLOCKS = {'oo': ('m', 'n'), 'vv': ('e', 'f'), 'ov': ('m', 'e'), 'vo': ('e', 'm')}


def density_insert(block):
    """'e1(m,n)' for 'oo', ..."""
    return 'e1(' + ','.join(DENSITY_BLOCKS[block]) + ')'


def density_out_idx(block):
    return list(DENSITY_BLOCKS[block])


def density_shape(block, spin=None):
    """'(no, nv)' for 'ov'; with spin 'a' the unrestricted '(no_a, nv_a)'."""
    dims = [('no' if s == 'o' else 'nv') + (f'_{spin}' if spin else '') for s in block]
    return '(' + ', '.join(dims) + ')'


def density_spin_labels(block, spin):
    """Both letters of a density block on one spin (off-diagonal spin blocks vanish)."""
    return {l: spin for l in DENSITY_BLOCKS[block]}


def generated_header(producer, *notes):
    """The comment header of every generated module: who wrote it and, one
    comment line each, what it is. `producer` is a path under src/CodeGeneration."""
    lines = ["# GENERATED CODE -- do not edit by hand.",
             f"# Produced by src/CodeGeneration/{producer}."]
    lines += [f"# {n}" if n else "#" for n in notes]
    return "\n".join(lines) + "\n"


def write_source(path, text, package_init=False):
    """Write one generated file (directories created; with package_init an
    empty __init__.py next to it if there is none)."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    if package_init and not os.path.exists(os.path.join(d, '__init__.py')):
        open(os.path.join(d, '__init__.py'), 'w').close()
    with open(path, 'w') as fh:
        fh.write(text)
    print(f"wrote {path}", flush=True)
    return path


def write_module(path, funcs, header, imports=EINSUM_IMPORTS, package_init=False):
    """header, imports, then the function sources two blank lines apart."""
    return write_source(path, header + imports + "\n\n" + "\n\n".join(funcs) + "\n",
                        package_init=package_init)


def module_subdir(fname):
    """generatedADC/adc{N}/ (created, with its __init__.py) for a module named
    adc{N}_....py: the layout the generated-ADC runtime and generate_adc's
    coverage() import from."""
    m = re.match(r'adc(\d+)_', fname)
    if not m:
        raise ValueError(f"cannot derive an adc{{N}} subpackage from module name {fname!r} "
                         "(expected an 'adc<N>_...' prefix)")
    for d in (ADC_OUT_DIR, os.path.join(ADC_OUT_DIR, f'adc{m.group(1)}')):
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, '__init__.py')):
            open(os.path.join(d, '__init__.py'), 'w').close()
    return os.path.join(ADC_OUT_DIR, f'adc{m.group(1)}')


def write_adc_module(fname, funcs, header, imports=EINSUM_IMPORTS):
    """write_module into generatedADC/adc{N}/<fname>."""
    return write_module(os.path.join(module_subdir(fname), fname), funcs, header, imports)


def shape_expr(letters):
    """'(no, nv, ...,)' for a list of occupied/virtual labels."""
    return "(" + ", ".join(('no' if l in OCC_LETTERS else 'nv') for l in letters) + ",)"


_AMP_LEAF = re.compile(r'^([tl]\d+_[ab]+)\(')


def amp_leaves(raw_terms):
    """Sorted spin-blocked amplitude leaf names (t2_abab, l1_aa, ...) that raw
    pq.strings() terms reference: a restricted block function's amplitude
    arguments (which spin blocks survive is pdaggerq's decision, so they are
    detected, not listed)."""
    leaves = set()
    for term in raw_terms:
        for fac in term[1:]:
            m = _AMP_LEAF.match(fac.strip())
            if m:
                leaves.add(m.group(1))
    return sorted(leaves)


def signed_terms(terms, sign):
    """Raw terms with every coefficient multiplied by `sign`."""
    from src.CodeGeneration.pdaggerq_interface import exact_coefficient
    return [[f"{exact_coefficient(t[0]) * sign:+.15g}"] + list(t[1:]) for t in terms]


def term_has_factor(term, tag):
    """True when `tag` (e.g. 'l3_aaaaaa') is one of a raw term's factors."""
    return any(f.strip().startswith(tag + '(') for f in term[1:])


def pq_terms(left_ops, products, spin_labels=None, right_ops=(('1',),)):
    """Raw pq.strings() of <left_ops| sum_k c_k product_k |right_ops>, one
    pdaggerq helper per call. `products` is a list of (coefficient, operator
    token list); spin_labels (a dict, possibly empty) asks for the spin-blocked
    strings."""
    from src.CodeGeneration.pdaggerq_interface import pq_helper
    pq = pq_helper('fermi')
    pq.set_left_operators([list(op) for op in left_ops])
    pq.set_right_operators([list(op) for op in right_ops])
    for coef, ops in products:
        pq.add_operator_product(coef, list(ops))
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms
