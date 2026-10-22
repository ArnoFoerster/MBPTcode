"""The solvation loop over MPI ranks (src.Base.solvation_loop with the state
models of src.SingleReference.state_models and eeADC.ee_state_model): every
rank runs the loop with its own molecule, medium and state objects, and every
rank must come back with the serial loop's numbers to a few ulp (each rank
computes them). Two and three simulated ranks
(mpi_grid.run_simulated; the fold and the relaxed densities shared out over
them), water 6-31G (bent, non-symmetric), frozen 1, cc-pvdz-ri /
cc-pvdz-jkfit, toluene IEF-PCM.

  1. G_eq, E(q), the charges, the relaxed density, the cycle count and the
     force of G_eq, for MP2 S0, ADC(2) S1 (with its LR term) and T1, and the
     S1 absorption and emission (vertical_energies: two equilibrium and two
     non-equilibrium loops, the slow charges handed between them): every
     rank holds rank 0's numbers to a few ulp, and 2 ranks give 3 ranks'.
  2. MP2 S0 equals the serial run to a few ulp. ADC(2) equals it to round-off
     only, below ROUNDOFF: the distributed fold sums the auxiliary index in
     chunks (ee_ri_fold.bare_factors), the serial one in one piece
     (measured, largest over G, E(q), q, density and force: 2.3e-12 S1,
     9.2e-12 T1, 5.7e-14 the S1 vertical energies; same cycle counts).

Run: python tests/test_solvation_loop_ranks.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base import solvation_loop as sl
from src.Base.reaction_field import ContinuumField
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC.ee_state_model import ADC2State
from tests.adc_common import (ATOMS, AUX, NF, ROOT_CONV_TOL, SOLVATION_TIGHT, check,
                              make_scf_loop, molecule)
from tests.pins import RECORDING_BLAS, matches

KINDS = ('S0', 'S1', 'T1')
ROUNDOFF = 1e-10    # Ha, Ha/Bohr, e: distributed vs serial ADC(2)


def build(kind):
    """(field, state) on a molecule of their own (pyscf moves a shared mol's
    rinv origin under a force)."""
    mol = molecule(ATOMS)
    field = ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))

    def state():
        mp2 = MP2State(HFState(mol, make_scf_loop), NF, AUX, force_route='dense')
        if kind == 'S0':
            return mp2
        lr = field.transition_environment() if kind == 'S1' else None
        return ADC2State(mp2, 'singlet' if kind == 'S1' else 'triplet', 0, 3, lr,
                         conv_tol=ROOT_CONV_TOL, follow='index')
    return field, state


def loop_and_force(comm=None):
    out = {}
    for kind in KINDS:
        field, make = build(kind)
        st = make()
        r = sl.equilibrate(st, field, **SOLVATION_TIGHT)
        out[kind] = dict(G=r.energy, E=r.state_energy, q=r.q, dm=r.dm_eff,
                         cycles=len(r.history), converged=r.converged,
                         F=sl.force(st, field, r))
    field, make_s1 = build('S1')
    _, make_s0 = build('S0')
    v = sl.vertical_energies(make_s0, make_s1, field, **SOLVATION_TIGHT)
    out['vertical'] = dict(absorption=v['absorption'], emission=v['emission'],
                           G=v['G'], converged=v['converged'])
    return out


def same(a, b):
    """Two runs' records: counts and flags equal, numbers to a few ulp."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(same(a[k], b[k]) for k in a))
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind in 'biu' or y.dtype.kind in 'biu':
        return np.array_equal(x, y)
    return matches(x, y, RECORDING_BLAS)


def largest_difference(a, b):
    worst = 0.0
    for k in a:
        if k in ('cycles', 'converged'):
            continue
        x, y = (a[k], b[k]) if not isinstance(a[k], dict) else (
            list(a[k].values()), list(b[k].values()))
        worst = max(worst, float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
    return worst


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref = loop_and_force()
        for kind in KINDS:
            all_ok &= check(ref[kind]['converged'], f'{kind}: serial loop converged',
                            f"{ref[kind]['cycles']} cycles, G {ref[kind]['G']:.12f}")
        all_ok &= check(ref['vertical']['converged'],
                        'S1 absorption / emission loops converged (serial)',
                        f"{ref['vertical']['absorption']:.10f} / "
                        f"{ref['vertical']['emission']:.10f} Ha")
        first = {}
        for size in (2, 3):
            outs = run_simulated(loop_and_force, size)
            first[size] = outs[0]
            for kind in KINDS + ('vertical',):
                all_ok &= check(all(same(o[kind], outs[0][kind]) for o in outs),
                                f'{size} ranks, {kind}: every rank holds rank 0\'s numbers')
        for kind in KINDS + ('vertical',):
            all_ok &= check(same(first[2][kind], first[3][kind]),
                            f'{kind}: 2 ranks = 3 ranks to a few ulp')
            d = largest_difference(first[2][kind], ref[kind])
            if kind == 'S0':
                all_ok &= check(same(first[2][kind], ref[kind]),
                                'S0: ranks = serial to a few ulp')
            else:
                all_ok &= check(d < ROUNDOFF and first[2][kind].get(
                    'cycles') == ref[kind].get('cycles'),
                                f'{kind}: ranks = serial to round-off (same cycles)',
                                f'largest difference {d:.1e}')
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_solvation_loop_ranks():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
