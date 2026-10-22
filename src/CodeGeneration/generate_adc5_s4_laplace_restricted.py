"""Laplace-fused rank>=3 S^(4) OVERLAP stitch -- the s4 l4t3 structure
(l4^(2)*t3^(2), a NESTED overlap with NO hamiltonian) that u_dot_orb_isr's
class-GS subtraction -eps_p * S_eff consumes at n=4 (Ss[4]). It is the only
surviving rank>=3 S^(4) structure (s_order_terms(4): ('l4t3',4,3,2,2), the
lower-order rank>=3 S survivors l3t2 live in the order-3 module and are handled
by adc4_u3_l3_laplace). Same nested double-tau machinery as the U^(4) l4ft3
(generate_adc5_u4_laplace_restricted): l4 -> tau1, t3 -> tau2, one Laplace
integral per amplitude KIND, so l4/t3 are NEVER materialized. The only
difference from the U^(4) generator is the term source -- overlap_element_terms
(no ham) instead of superop_element_terms_dressed. Writes
generatedADC/adc5/adc5_s4_laplace_restricted[_df].py (generated on the user's
machine, never committed).
"""
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.SingleReference.ADC.precursor_classes import (
    OCC_LETTERS, VIR_LETTERS, class_letters, precursor_ket_string,
    class_index_convention)
from src.CodeGeneration.generate_adcn import (
    overlap_element_terms, _shape_expr, _module_subdir)
from src.CodeGeneration.laplace_codegen import (
    assemble_laplace_fused)
from src.CodeGeneration.generate_adc5_u4_laplace_restricted import (
    _fuse_highrank, _highrank_tags, PARAMS, PARAMS_DF, ALIASES)

_S4_SPECS = [('l4t3', [['l4']], ['t3'])]   # only surviving rank>=3 S^(4)


def generate(df=False):
    from itertools import product as _iprod
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'],
                                   vir_offset=conv['np'])
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'],
                               vir_offset=conv['np'])
        ket_letters = kh + kp
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' \
            else (lambda l: [f'a({l})'])
        for split, lo, kd in _S4_SPECS:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                for spins in _iprod('ab', repeat=len(out_idx)):
                    sl = dict(zip(out_idx, spins))
                    outer = overlap_element_terms(
                        p_tok(p_letter), ket, left_ops=lo, ket_dressing=kd,
                        spin_labels=sl)
                    if not outer:
                        continue
                    fused, nslots = _fuse_highrank(outer, df=df)
                    if not fused:
                        continue
                    suffix = ''.join(spins)
                    fname = f's4_{tag}_{row}_{split}_{suffix}'
                    print(f"{fname}: {len(outer)} outer -> {len(fused)} fused "
                          f"(nslots={nslots})", flush=True)
                    funcs.append(assemble_laplace_fused(
                        fname, PARAMS_DF if df else PARAMS, ALIASES, 'out',
                        _shape_expr(out_idx), fused, out_idx,
                        eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                        ntau_default=6, nslots=nslots))
    fname = 'adc5_s4_laplace_restricted%s.py' % ('_df' if df else '')
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Laplace-fused rank>=3 S^(4) overlap (l4t3); l4/t3 never\n"
                 "# materialized. From generate_adc5_s4_laplace_restricted.py.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"\nwrote {path}  ({len(funcs)} functions)", flush=True)


if __name__ == '__main__':
    generate()
