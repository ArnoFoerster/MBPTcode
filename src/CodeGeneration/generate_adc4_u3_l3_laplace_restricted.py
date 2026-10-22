"""Laplace-fused l3-consuming U^(3)/S^(3) stitches for the Dyson-ADC(4)
downfold -- eliminates the last O(no^3 nv^3) materialized tensors (l3, and
its t3 precursor) from the restricted matrix-free U^(3) coupling.

Scope: the l3ft2 (U^(3)) and l3t2 (S^(3)) tag structures of
generate_adc4_u3_restricted, for BOTH sectors (2h1p/2p1h) and BOTH rows
(occ/vir), every Ms-conserving output spin pattern. Every other U^(3)/S^(3)
tag structure (l2vt2, l2ft1, l1v, ... -- rank<=2 amplitudes) is untouched:
those never form a rank>4 tensor, so the ordinary dense generated blocks
(adc4_u3_blocks_restricted) still handle them.

Mechanism (numerical, convergeable, stays Dyson):
l3 = _to_l_restricted(t3) is a signed
axis-transpose of t3, and t3 = numerator/D3. The generic Laplace fuser
(laplace_codegen.fuse_amplitude_into_consumer) substitutes l3's numerator --
i.e. t3's numerator relabeled into l3's occ-first external order -- directly
into each stitch's contraction, one minimax-quadrature tau point at a time,
so l3 (and t3) are never materialized. The occ-first external order and the
_to_l sign are supplied via the fuser's `canonical` override:
  same-spin (aaaaaa/bbbbbb): canonical i,j,k,c,b,a ; term sign -1
  mixed-spin (aabaab/abbabb): canonical i,j,k,a,b,c ; term sign +1
(canonical is a reordering of the numerator's own external legs {a,b,c,i,j,k}
encoding _to_l_restricted's transpose; validated block-by-block to machine
precision at large ntau vs the dense adc4_u3_blocks_restricted twins.)

Writes generatedADC/adc4/adc4_u3_l3_laplace_restricted[_df].py (generated on
the user's machine, never committed).
"""
import os
import re

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import exact_coefficient

from src.SingleReference.ADC.precursor_classes import (
    OCC_LETTERS, VIR_LETTERS, class_letters, precursor_ket_string,
    class_index_convention)
from src.CodeGeneration.generate_adcn import (
    superop_element_terms_dressed, overlap_element_terms, _shape_expr,
    _module_subdir)
from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, T_BRA)
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_into_consumer, assemble_laplace_fused, lambda_canonical_and_sign)
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'SingleReference', 'ADC', 'generatedADC'))

# Amplitude rank consumed by the l-stitches. 3 = l3 (ADC(4)'s U^(3)); the whole
# generator is rank-parametric via lambda_canonical_and_sign, so ADC(5)'s l4/l5
# U-couplings reuse it by bumping RANK and supplying their own split defs.
RANK = 3
# occ-first external orders + signs realizing _to_l_restricted's transpose,
# derived rank-generally (not hard-coded to 6 indices).
_LCS = lambda_canonical_and_sign(RANK)
CANON_SAME, SIGN_SAME = _LCS['same']
CANON_MIX, SIGN_MIX = _LCS['mixed']

# uniform param signature (superset; unused leaves harmless). eps_a drives the
# quadrature grid; t2_1_* are l3's numerator inputs; f_aa/f_bb only used by the
# l3ft2 (U^(3)) family, g/d by the numerator + outer integrals.
PARAMS = ['f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb',
          'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab']
# DF/RI twin: the three 4-index g blocks give way to the 3-index B factors.
PARAMS_DF = ['f_aa', 'f_bb', 'B_aa', 'B_bb', 'd_aa', 'd_bb',
             'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab']
ALIASES = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab', 't2_bbbb': 't2_1_aaaa'}


def _l_tags(outer, rank=RANK):
    """l{rank} factor tags present in the consumer terms (l3_aaaaaa, ...)."""
    pat = re.compile(rf'^(l{rank}_[ab]{{{2 * rank}}})\(')
    tags = set()
    for term in outer:
        for tok in term[1:]:
            m = pat.match(tok.strip())
            if m:
                tags.add(m.group(1))
    return sorted(tags)


def _signed(terms, sign):
    return [[f"{exact_coefficient(t[0]) * sign:+.15g}"] + list(t[1:]) for t in terms]


def _numerator_block(block, rank=RANK):
    """Map an l{rank} spin block string to a stored T_BRA numerator key. The
    stored keys are the alpha-majority representatives; a block whose numerator
    isn't stored (e.g. the all-beta bb..b) is its closed-shell a<->b twin, whose
    numerator arrays (g/t2) alias the stored ones -- so the same numerator terms
    reproduce it numerically (verified by the block-by-block validation)."""
    keys = T_BRA[rank][1]
    if block in keys:
        return block
    flipped = block.translate(str.maketrans('ab', 'ba'))
    if flipped in keys:
        return flipped
    raise KeyError(f"no numerator key for l{rank} block {block!r}")


def _fuse_all_l(outer, rank=RANK, df=False):
    """Fuse every l{rank}_xxxx factor present in `outer` with its (signed,
    occ-first-canonical) numerator. Returns the combined fused raw-term list.
    Rank-general: canonical/sign come from lambda_canonical_and_sign(rank), the
    numerator from t_numerator_terms_restricted(rank, 2, block)."""
    # DF x LAPLACE composition: dress BOTH the consumer (outer) and the
    # amplitude numerator (inner) BEFORE fusing, exactly as
    # generate_mp3_t3_laplace_restricted does. It composes because
    # dress_integral_factor's B_xx(Q,p,r) output is just another plain-paren
    # factor to laplace_codegen's parser (no bracket syntax survives dressing),
    # and the fresh auxiliary letters get renamed like any other dummy during
    # fusion. Tags are read from the DRESSED outer: dressing rewrites bracket
    # factors only, so the l{rank} amplitude tags are untouched.
    if df:
        outer = dress_integral_factor(outer)
    fused = []
    for tag in _l_tags(outer, rank):
        block = tag[len(f'l{rank}_'):]
        same = len(set(block)) == 1
        canon, sign = (CANON_SAME, SIGN_SAME) if same else (CANON_MIX, SIGN_MIX)
        inner = _signed(t_numerator_terms_restricted(rank, 2, _numerator_block(block, rank)), sign)
        if df:
            inner = dress_integral_factor(inner)
        fused += fuse_amplitude_into_consumer(inner, rank, outer, tag, canonical=canon)
    return fused


def generate(df=False):
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_letters = kh + kp
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        # (split, family, outer_fn) -- l3ft2 is U^(3) (superop, ham 'f'),
        # l3t2 is S^(3) (overlap, no ham).
        specs = [
            ('l3ft2', 'u3',
             lambda pl, sl: superop_element_terms_dressed(
                 p_tok(pl), ket, sector, ('f',), left_ops=[['l3']],
                 ket_dressing=['t2'], spin_labels=sl)),
            ('l3t2', 's3',
             lambda pl, sl: overlap_element_terms(
                 p_tok(pl), ket, left_ops=[['l3']], ket_dressing=['t2'],
                 spin_labels=sl)),
        ]
        for split, fam, outer_fn in specs:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                from itertools import product as _iprod
                for spins in _iprod('ab', repeat=len(out_idx)):
                    sl = dict(zip(out_idx, spins))
                    outer = outer_fn(p_letter, sl)
                    if not outer:
                        continue
                    fused = _fuse_all_l(outer, df=df)
                    if not fused:
                        continue
                    suffix = ''.join(spins)
                    # NO '_lap' suffix: the separate module name already
                    # distinguishes these from the dense adc4_u3_blocks_restricted
                    # twins, and _stitch_k1r looks up `{base}_{suffix}` verbatim.
                    fname = f'{fam}_{tag}_{row}_{split}_{suffix}'
                    print(f"{fname}: {len(outer)} outer -> {len(fused)} fused terms "
                          f"(l{RANK} {_l_tags(outer)})")
                    funcs.append(assemble_laplace_fused(
                        fname, PARAMS_DF if df else PARAMS, ALIASES, 'out', _shape_expr(out_idx),
                        fused, out_idx, eps_o_expr='eps_a[o]',
                        eps_v_expr='eps_a[v]', ntau_default=6))

    fname = 'adc4_u3_l3_laplace_restricted%s.py' % ('_df' if df else '')
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Laplace-fused l3-consuming U^(3)/S^(3) stitches; produced by\n"
                 "# src/CodeGeneration/generate_adc4_u3_l3_laplace_restricted.py\n"
                 "# (see that module's docstring). l3/t3 are never materialized.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"\nwrote {path}  ({len(funcs)} functions)")


if __name__ == '__main__':
    generate()
