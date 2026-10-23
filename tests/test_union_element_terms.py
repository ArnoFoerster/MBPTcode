"""Regression gate for the union-space element builder
(generate_adcn.union_element_terms): before it is used for any D x C cross
block it must reproduce the two-sector elements it generalizes.

The two-sector convention (superop_element_terms, validated numerically
against adc_spinorbital.py) evaluates IP-class elements with the reversed
commutator [C_J, H]; the union-space form uses the normal commutator for
every class and represents an IP class by D = E^dag. The two agree through
{E_J, [H, E_K^dag]} = ({E_J^dag, [E_K, H]})^dag, which under Omega-dressing
exchanges the l- and t-dressed splits, so the comparison is per ORDER (the sum
over all dressing splits at fixed amplitude orders), not per split:

  * k1-k1 (both sectors): K (order 0), C^(1) (order 1), the order-2 raw sum
    (every u_order_terms(2) assignment at first/second-order amplitudes) and
    the order-2 within-class overlap sum -- 6-index tensors, elementwise;
  * orbital-k1 couplings (both sectors, occupied and virtual orbital):
    U^(1) and the order-2 raw sum;
  * k1-k2 C^(1) (both sectors) and, on H4, the k2-k2 K and C^(1).

The EA term lists are identical strings (asserted); the IP lists are the
adjoint relabelling and are compared numerically. Also pinned: the D x C
element has no terms at orders 0 and 1 and the D x C metric has none at all.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect

import numpy as np
import pytest

from src.CodeGeneration.generate_adcn import (
    superop_element_terms, overlap_element_terms,
    union_element_terms, union_element_terms_ops, union_overlap_terms,
    union_overlap_terms_ops, assemble_block)
from src.CodeGeneration.codegen_common import shape_expr
from src.SingleReference.ADC.adc_order_rule import u_order_terms, s_order_terms
from src.SingleReference.ADC.precursor_classes import (
    class_index_convention, class_letters, precursor_bra_string, precursor_ket_string,
    union_operator_string, pair_index_convention, OCC_LETTERS, VIR_LETTERS)
from src.SingleReference.DensityMatrix.mpn_density_driver import MPnDensityDriver
from src.SingleReference.common.mp_amplitudes import to_l
from tests.test_isr_assembly import _build
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)

SYSTEMS = [('H4/sto-3g', 'H 0 0 0; H 0 0 0.80; H 0 0 1.75; H 0 0 2.60'),
           ('LiH/sto-3g', 'Li 0 0 0; H 0 0 1.6')]
TOL = 1e-14           # relative to max(1, |block|): pure einsum re-association noise


def _evaluate(terms, out_idx, pool, extra_args=()):
    """Numerical value of a raw term list as the dense out_idx tensor."""
    src = assemble_block('blk', shape_expr(out_idx), terms, out_idx, '', extra_args=extra_args)
    ns = {}
    exec('import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n' + src, ns)
    fn = ns['blk']
    return fn(**{p: pool[p] for p in inspect.signature(fn).parameters})


def _pool(dd):
    eps, g, nocc = dd['eps'], dd['g'], dd['nocc']
    norb = len(eps)
    mpn = MPnDensityDriver(eps, g, nocc)
    amps = mpn.compute_order_amplitudes_general(2)
    pool = dict(f=np.diag(eps), g=g, kd=np.eye(norb), o=slice(0, nocc), v=slice(nocc, norb),
                no=nocc, nv=norb - nocc)
    return pool, amps


def _dressed_sum(order, two_sector_fn, union_fn, pool, amps, kind='u'):
    """(two-sector, union) order-`order` raw sums over every dressing split."""
    two = uni = 0.0
    n_terms = 0
    terms_iter = u_order_terms(order) if kind == 'u' else s_order_terms(order)
    for t in terms_iter:
        if kind == 'u':
            nm, j, ham, k, a, b = t
        else:
            nm, j, k, a, b = t
            ham = None
        if (j and (j, a) not in amps) or (k and (k, b) not in amps):
            continue
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        extra = {}
        if j:
            extra[f'l{j}'] = to_l(amps[(j, a)], j)
        if k:
            extra[f't{k}'] = amps[(k, b)]
        t2s = two_sector_fn(ham, lo, kd)
        tun = union_fn(ham, lo, kd)
        if not t2s and not tun:
            continue
        n_terms += len(t2s) + len(tun)
        p = dict(pool)
        p.update(extra)
        if t2s:
            two = two + _evaluate(t2s, two_sector_fn.out_idx, p, extra_args=list(extra))
        if tun:
            uni = uni + _evaluate(tun, union_fn.out_idx, p, extra_args=list(extra))
    return two, uni, n_terms


def _report(label, two, uni, require_nonzero=True):
    scale = float(np.max(np.abs(two))) if np.size(two) else 0.0
    d = float(np.max(np.abs(np.asarray(two) - np.asarray(uni))))
    print(f"  {label:48s} |block|={scale:.3e}  |two-sector - union|={d:.2e}")
    if require_nonzero:
        assert scale > 1e-10, f"{label}: two-sector block is zero -- no content to compare"
    assert d <= TOL * max(1.0, scale), (label, d, scale)


@pytest.mark.parametrize('label,atom', SYSTEMS)
def test_same_sector_class_pairs(label, atom):
    dd = _build(atom)
    pool, amps = _pool(dd)
    pairs = [(1, 1), (1, 2)] + ([(2, 2)] if label.startswith('H4') else [])
    for bra_k, ket_k in pairs:
        for sector in ('IP', 'EA'):
            conv = pair_index_convention(bra_k, sector, ket_k, sector)
            bra_dag = precursor_bra_string(bra_k, sector)
            ket = precursor_ket_string(ket_k, sector, occ_offset=conv['ket_occ_offset'],
                                       vir_offset=conv['ket_vir_offset'])
            out_idx = conv['out_idx']

            def two_fn(ham, lo, kd, bra_dag=bra_dag, ket=ket, sector=sector):
                if ham is None:
                    return overlap_element_terms(bra_dag, ket, left_ops=lo, ket_dressing=kd)
                return superop_element_terms(bra_dag, ket, sector, (ham,),
                                                     left_ops=lo, ket_dressing=kd)

            def uni_fn(ham, lo, kd, bk=bra_k, kk=ket_k, sector=sector):
                if ham is None:
                    return union_overlap_terms(bk, sector, kk, sector, left_ops=lo, ket_dressing=kd)
                return union_element_terms(bk, sector, kk, sector, hams=(ham,),
                                           left_ops=lo, ket_dressing=kd)
            two_fn.out_idx = uni_fn.out_idx = out_idx
            tag = f"{label} {sector} k{bra_k}k{ket_k}"
            if sector == 'EA':
                assert two_fn('f', None, None) == uni_fn('f', None, None)
                assert two_fn('v', None, None) == uni_fn('v', None, None)
            for order, ham in ((0, 'f'), (1, 'v')):
                t2s, tun = two_fn(ham, None, None), uni_fn(ham, None, None)
                if not t2s and not tun:
                    print(f"  {tag} order {order}: both prune to zero")
                    continue
                # the k1-k2 coupling has no f-part (first order only): the f
                # element is emitted and evaluates to zero in both conventions
                _report(f"{tag} order {order} ({ham})", _evaluate(t2s, out_idx, pool),
                        _evaluate(tun, out_idx, pool),
                        require_nonzero=not (bra_k != ket_k and ham == 'f'))
            if (bra_k, ket_k) == (1, 1):
                two, uni, nt = _dressed_sum(2, two_fn, uni_fn, pool, amps, 'u')
                _report(f"{tag} order-2 raw sum ({nt} terms)", two, uni)
                two, uni, nt = _dressed_sum(2, two_fn, uni_fn, pool, amps, 's')
                _report(f"{tag} order-2 overlap sum ({nt} terms)", two, uni)


@pytest.mark.parametrize('label,atom', SYSTEMS)
def test_orbital_k1_couplings(label, atom):
    dd = _build(atom)
    pool, amps = _pool(dd)
    for sector in ('IP', 'EA'):
        conv = class_index_convention(1, sector)
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        y = union_operator_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        for p_letter in (OCC_LETTERS[2 * conv['nh']], VIR_LETTERS[2 * conv['np']]):
            out_idx = kh + kp + [p_letter]
            p_tok = [f'a*({p_letter})'] if sector == 'IP' else [f'a({p_letter})']

            def two_fn(ham, lo, kd, p_tok=p_tok, ket=ket, sector=sector):
                if ham is None:
                    return overlap_element_terms(p_tok, ket, left_ops=lo, ket_dressing=kd)
                return superop_element_terms(p_tok, ket, sector, (ham,),
                                                     left_ops=lo, ket_dressing=kd)

            def uni_fn(ham, lo, kd, x=[f'a*({p_letter})'], y=y):
                if ham is None:
                    return union_overlap_terms_ops(x, y, left_ops=lo, ket_dressing=kd)
                return union_element_terms_ops(x, y, hams=(ham,), left_ops=lo, ket_dressing=kd)
            two_fn.out_idx = uni_fn.out_idx = out_idx
            tag = f"{label} {sector} orb({p_letter})-k1"
            _report(f"{tag} U^(1)", _evaluate(two_fn('v', None, None), out_idx, pool),
                    _evaluate(uni_fn('v', None, None), out_idx, pool))
            two, uni, nt = _dressed_sum(2, two_fn, uni_fn, pool, amps, 'u')
            _report(f"{tag} order-2 raw sum ({nt} terms)", two, uni)
            two, uni, nt = _dressed_sum(2, two_fn, uni_fn, pool, amps, 's')
            # the virtual-orbital row of the orbital-k1 overlap has no order-2 content
            _report(f"{tag} order-2 overlap sum ({nt} terms)", two, uni, require_nonzero=nt > 0)


def test_cross_block_vanishes_at_orders_0_and_1_and_has_no_metric():
    assert union_element_terms(1, 'IP', 1, 'EA', hams=('f',)) == []
    assert union_element_terms(1, 'IP', 1, 'EA', hams=('v',)) == []
    assert union_element_terms(1, 'EA', 1, 'IP', hams=('f', 'v')) == []
    for j, k in ((0, 0), (2, 0), (0, 2), (2, 2), (1, 1), (3, 0), (0, 3)):
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        assert union_overlap_terms(1, 'IP', 1, 'EA', left_ops=lo, ket_dressing=kd) == []
    # the order-2 Hamiltonian cross block is NOT empty (it is the deleted block)
    n2 = sum(len(union_element_terms(1, 'IP', 1, 'EA', hams=(ham,), left_ops=lo, ket_dressing=kd))
             for nm, j, ham, k, a, b in u_order_terms(2)
             for lo, kd in [([[f'l{j}']] if j else None, [f't{k}'] if k else None)])
    print(f"  D(k1) x C(k1) order-2 structures: {n2} terms in total")
    assert n2 > 0


def main():
    for label, atom in SYSTEMS:
        test_same_sector_class_pairs(label, atom)
        test_orbital_k1_couplings(label, atom)
    test_cross_block_vanishes_at_orders_0_and_1_and_has_no_metric()
    print('\nALL OK')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
