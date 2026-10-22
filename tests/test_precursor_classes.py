"""Regression test for src/SingleReference/ADC/precursor_classes.py (Phase 1
of the automated ADC(n) generator):

class_configs(nocc, norb, k, sector) must reproduce base.py's hand-written
_configs_2h1p/_configs_3h2p (sector='IP', k=1/2) and _configs_2p1h/_configs_3p2h
(sector='EA', k=1/2) BYTE-FOR-BYTE -- this is the cheap, high-signal check that
the generic combinatorics/flattening convention matches the existing,
already-validated ADC(3)/(4) configuration-index bookkeeping before any new
physics is built on top of it. Also checks class_dimension against
dimensions()/dimensions_adc4()'s n2h1p/n2p1h/n3h2p/n3p2h formulas, and
precursor_bra_string/precursor_ket_string against
generate_eom_ccsdt.py's hand-written IP_BRA/IP_KET/EA_BRA/EA_KET token tables.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.SingleReference.ADC.base import ADCSolverUnrestricted
from src.SingleReference.ADC.precursor_classes import (
    class_configs, class_dimension, dimensions_adcn,
    precursor_bra_string, precursor_ket_string)

def main():
    all_ok = True

    for norb, nocc in [(10, 3), (14, 5), (7, 2)]:
        nvirt = norb - nocc
        base = ADCSolverUnrestricted.from_arrays(np.arange(norb, dtype=float), np.zeros((norb,) * 4))

        checks = [
            ('IP k=1 vs _configs_2h1p', class_configs(nocc, norb, 1, 'IP'), base._configs_2h1p(nocc)),
            ('IP k=2 vs _configs_3h2p', class_configs(nocc, norb, 2, 'IP'), base._configs_3h2p(nocc)),
            ('EA k=1 vs _configs_2p1h', class_configs(nocc, norb, 1, 'EA'), base._configs_2p1h(nocc)),
            ('EA k=2 vs _configs_3p2h', class_configs(nocc, norb, 2, 'EA'), base._configs_3p2h(nocc)),
        ]
        for name, got, expect in checks:
            ok = len(got) == len(expect) and all(np.array_equal(g, e) for g, e in zip(got, expect))
            print(f"[{'OK' if ok else 'FAIL'}] nocc={nocc},norb={norb} {name}")
            all_ok &= ok

        d = base.dimensions(nocc)
        d3x = base.dimensions_adc4(nocc)
        dim_checks = [
            ('IP k=1 dim vs n2h1p', class_dimension(nocc, nvirt, 1, 'IP'), d['n2h1p']),
            ('EA k=1 dim vs n2p1h', class_dimension(nocc, nvirt, 1, 'EA'), d['n2p1h']),
            ('IP k=2 dim vs n3h2p', class_dimension(nocc, nvirt, 2, 'IP'), d3x['n3h2p']),
            ('EA k=2 dim vs n3p2h', class_dimension(nocc, nvirt, 2, 'EA'), d3x['n3p2h']),
            ('IP k=0 dim == nocc', class_dimension(nocc, nvirt, 0, 'IP'), nocc),
            ('EA k=0 dim == nvirt', class_dimension(nocc, nvirt, 0, 'EA'), nvirt),
        ]
        for name, got, expect in dim_checks:
            ok = got == expect
            print(f"[{'OK' if ok else 'FAIL'}] nocc={nocc},norb={norb} {name}: {got} vs {expect}")
            all_ok &= ok

        dn = dimensions_adcn(nocc, norb, n=1, max_k=1)
        ok = (dn[('IP', 1)] == d['n2h1p'] and dn[('EA', 1)] == d['n2p1h']
              and dn['nH'] == d['n2h1p'] + d['n2p1h'] + norb)
        print(f"[{'OK' if ok else 'FAIL'}] nocc={nocc},norb={norb} dimensions_adcn nH consistency")
        all_ok &= ok

    # Operator-string convention vs. generate_eom_ccsdt.py's hand-written tables
    # (IP_KET/IP_BRA/EA_KET/EA_BRA at rank R=k+1).
    string_checks = [
        ('IP k=0 ket', precursor_ket_string(0, 'IP'), ['a(i)']),
        ('IP k=0 bra', precursor_bra_string(0, 'IP'), ['a*(i)']),
        ('IP k=1 ket', precursor_ket_string(1, 'IP'), ['a*(a)', 'a(j)', 'a(i)']),
        ('IP k=1 bra', precursor_bra_string(1, 'IP'), ['a*(i)', 'a*(j)', 'a(a)']),
        ('IP k=2 ket', precursor_ket_string(2, 'IP'), ['a*(a)', 'a*(b)', 'a(k)', 'a(j)', 'a(i)']),
        ('IP k=2 bra', precursor_bra_string(2, 'IP'), ['a*(i)', 'a*(j)', 'a*(k)', 'a(b)', 'a(a)']),
        ('EA k=0 ket', precursor_ket_string(0, 'EA'), ['a*(a)']),
        ('EA k=0 bra', precursor_bra_string(0, 'EA'), ['a(a)']),
        ('EA k=1 ket', precursor_ket_string(1, 'EA'), ['a*(a)', 'a*(b)', 'a(i)']),
        ('EA k=1 bra', precursor_bra_string(1, 'EA'), ['a*(i)', 'a(b)', 'a(a)']),
        ('EA k=2 ket', precursor_ket_string(2, 'EA'), ['a*(a)', 'a*(b)', 'a*(c)', 'a(j)', 'a(i)']),
        ('EA k=2 bra', precursor_bra_string(2, 'EA'), ['a*(i)', 'a*(j)', 'a(c)', 'a(b)', 'a(a)']),
    ]
    for name, got, expect in string_checks:
        ok = got == expect
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {got} vs {expect}")
        all_ok &= ok

    print("ALL OK" if all_ok else "SOME FAILED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
