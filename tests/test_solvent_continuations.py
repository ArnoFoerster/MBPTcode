"""The continuum reaches every continuation of the space-time route the same way.

`calc_qp_energy(mode='space-time')` leaves the imaginary axis by Pade, by
contour deformation ('cd'), by the Laplace-transformed contour ('laplace') or
by the sum-over-poles model ('sop'). In a continuum each of them must screen
the self-energy with the BARE interaction and add Duchemin et al.'s Eq. (18)
shift -- the construction the Pade route and the gradient chains use. A
continuation that screened Sigma with the dressed interaction and added the
static COHSEX term instead would describe a different solvated functional
under the same name: on water in water the two differ by tenths of an eV in
the gap.

What is compared is the SOLVENT SHIFT of the HOMO, IP(solvated) - IP(gas),
per continuation. The continuations differ in how they reach the real axis,
by a few meV on this molecule, and that difference cancels in the shift; the
static term does not cancel, so a continuation on the wrong static term shows
up here as a shift tens to hundreds of meV away from Pade's.

Run as a script: `python tests/test_solvent_continuations.py`.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.separable_ri import optimize_atomic_radii
from src.Base.solvent_screening import (attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
CONTINUATIONS = ('pade', 'cd', 'laplace', 'sop')
#: The continuation error of the dynamic part cancels in the solvent shift;
#: what is left is the static-grid difference of the two Eq. (18) builds.
SHIFT_TOL_EV = 0.005


def check(ok, label, detail=''):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def build():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
    auxbasis = str(mol.basis) + '-ri'
    radii = {el: optimize_atomic_radii(el, mol.basis, auxbasis)[0]
             for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})}
    return mol, mf, radii


def solvent_shifts(mf, radii):
    """{continuation: (IP gas, IP solvated)} for the HOMO, in eV."""
    def ip(continuation):
        return -float(np.atleast_1d(calc_qp_energy(
            mf, selfenergy='GW', df=True, state='homo', mode='space-time',
            continuation=continuation, radii=radii))[0])

    out = {}
    for c in CONTINUATIONS:
        gas = ip(c)
        attach_solvent_screening(mf, solvent='water')
        try:
            sol = ip(c)
        finally:
            detach_solvent_screening(mf)
        out[c] = (gas, sol)
        print(f'       {c:8s}: IP {gas:8.4f} -> {sol:8.4f} eV '
              f'(shift {sol - gas:+.4f})')
    return out


def check_every_continuation_carries_eq18(mf, radii):
    ips = solvent_shifts(mf, radii)
    shift = {c: sol - gas for c, (gas, sol) in ips.items()}
    ok = check(all(s < -0.1 for s in shift.values()),
               'every continuation reports a lower IP in solvent')
    for c in CONTINUATIONS[1:]:
        d = abs(shift[c] - shift['pade'])
        ok &= check(d < SHIFT_TOL_EV,
                    f"continuation='{c}' carries the same solvent shift as pade",
                    f'{d * 1000:.2f} meV')
    return ok


def run():
    import warnings
    warnings.simplefilter('ignore')
    mol, mf, radii = build()
    print('\n-- the solvent shift of the HOMO on every space-time continuation')
    all_ok = check_every_continuation_carries_eq18(mf, radii)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_solvent_continuations_checks():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
