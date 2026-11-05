"""ISR-ADC(2) transition densities on density-fitted tiles
(eeADC.ee_ri_transition).

Water, 6-31G, DF-RHF on cc-pvdz-jkfit, the fold on cc-pvdz-ri, the 1s core
frozen; tiles of one occupied orbital (tile_gb 1e-9), so every pass runs over
many tiles. Checks:

  1. term by term, charge and spin, the tiled densities equal the
     whole-doubles evaluation of ee_transition_density on the same roots and
     factors (fold_state, reference_from_factors) to 1e-14: singlet-triplet,
     triplet-triplet and singlet-singlet pairs and the ground row of singlets
     and triplets, for ADC(2), SCS-ADC(2) and an arbitrary (c_os, c_ss).
  2. a singlet-triplet pair has no charge density, a pair of one spin no
     spin density, exactly.
  3. over 2 and 3 simulated ranks the densities are the serial ones to a few
     ulp.
  4. refusals: an ADC(1) operator, roots of operators on other factors, a
     pair naming a root not given.
  5. a pass (ee_ri_tiles.Pass) over 2 and 3 simulated ranks sums its tiles
     to the serial bits, and no rank receives another rank's addends.

Run: python tests/test_ee_ri_transition.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace

import numpy as np

from src.Base.numerics.constants import ADC2_SPIN_SCALING
from src.Base.utils.parallel.simulated_comm import SimulatedComm, run_simulated
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold, ee_ri_tiles
from src.SingleReference.ADC.eeADC import ee_ri_transition as ert
from src.SingleReference.ADC.eeADC import ee_transition_density as etd
from tests.adc_common import AUX, WATER_ORIGIN, check, rhf_df
from tests.pins import RECORDING_BLAS, matches

TILE_GB = 1e-9
SCALINGS = (None, ADC2_SPIN_SCALING['scs'], (0.7, 1.9))
PAIRS = [(('singlet', 0), ('triplet', 0)), (('singlet', 1), ('triplet', 0)),
         (('singlet', 0), ('triplet', 1)), (('triplet', 0), ('triplet', 1)),
         (('singlet', 0), ('singlet', 1))]
GROUNDS = [('singlet', 0), ('triplet', 0), ('triplet', 1)]


def solved_roots(f, scaling, comm=None):
    """{(spin, k): (op, y, omega, t1)} of the two lowest roots of each spin."""
    roots = {}
    for spin in ee_ri_fold.SPINS:
        op = ee_ri_fold.RIFoldADC2(f, spin, scaling=scaling, tile_gb=TILE_GB, comm=comm)
        r = ee_fold.solve_folded(op.pieces(), 2, spin=spin, tol_omega=1e-11,
                                 tol_residual=1e-9)
        for k in range(2):
            roots[spin, k] = (op, np.asarray(r.y)[:, k], float(r.omega[k]),
                              float(r.t1[k]))
    return roots


def check_against_dense(f):
    """Checks 1 and 2."""
    ok = True
    eps = np.concatenate([f.eo, f.ev])
    for scaling in SCALINGS:
        roots = solved_roots(f, scaling)
        ref = etd.reference_from_factors(eps, etd.factors_of(roots['singlet', 0][0]),
                                          f.no, ground=True, scaling=scaling)
        state = {k: etd.fold_state(*v) for k, v in roots.items()}
        got = ert.transition_densities(roots, PAIRS, GROUNDS, terms=True, tile_gb=TILE_GB)
        err, zero = 0.0, 0.0
        for p in PAIRS + [(ert.GROUND, k) for k in GROUNDS]:
            dense = (etd.ground_transition_density(ref, state[p[1]], terms=True)
                     if p[0] == ert.GROUND else
                     etd.transition_density(ref, state[p[0]], state[p[1]], terms=True))
            for d, t in zip(dense, got[p]):
                assert set(d) == set(t), sorted(set(d) ^ set(t))
                err = max([err] + [float(np.abs(np.asarray(d[n]) - t[n]).max()) for n in d])
            mixed = (p[0][0] == 'triplet') != (p[1][0] == 'triplet')
            unused = got[p][0] if mixed else got[p][1]
            zero = max([zero] + [float(np.abs(v).max()) for v in unused.values()])
        ok &= check(err < 1e-14, f'{scaling}: tiled densities equal the whole-doubles '
                    'ones term by term', f'{err:.1e}')
        ok &= check(zero == 0.0, f'{scaling}: a mixed-spin pair has no charge, a '
                    'pair of one spin no spin density')
    return ok


def check_ranks(mol, mf):
    """Check 3."""
    scaling = ADC2_SPIN_SCALING['scs']
    f = ee_ri_fold.bare_factors(mol, mf, frozen=1, auxbasis=AUX)
    ser_roots = solved_roots(f, scaling)
    want = {k: (v[1], v[2], v[3]) for k, v in ser_roots.items()}
    ser = ert.transition_densities(ser_roots, PAIRS, GROUNDS, tile_gb=TILE_GB)

    def rank(comm):
        fr = ee_ri_fold.bare_factors(mol, mf, frozen=1, auxbasis=AUX, comm=comm)
        ops = {s: ee_ri_fold.RIFoldADC2(fr, s, scaling=scaling, tile_gb=TILE_GB,
                                        comm=comm) for s in ee_ri_fold.SPINS}
        roots = {k: (ops[k[0]],) + v for k, v in want.items()}
        return ert.transition_densities(roots, PAIRS, GROUNDS, tile_gb=TILE_GB)

    ok = True
    for size in (2, 3):
        outs = run_simulated(rank, size)
        same = all(matches(o[p][c], ser[p][c], RECORDING_BLAS)
                   for o in outs for p in ser for c in (0, 1))
        ok &= check(same, f'{size} simulated ranks give the serial densities')
    return ok


def check_refusals(mol, mf, f):
    """Check 4."""
    ok = True
    roots = solved_roots(f, None)
    op1 = ee_ri_fold.RIFoldADC2(f, 'singlet', level='adc1')
    other = ee_ri_fold.RIFoldADC2(ee_ri_fold.bare_factors(mol, mf, frozen=1, auxbasis=AUX),
                                  'triplet')
    cases = (({('singlet', 0): (op1,) + roots['singlet', 0][1:]}, [], [('singlet', 0)],
              'an ADC(1) operator'),
             ({('singlet', 0): roots['singlet', 0], ('triplet', 0): (other,) +
               roots['triplet', 0][1:]}, [(('singlet', 0), ('triplet', 0))], [],
              'roots on other factors'),
             ({('singlet', 0): roots['singlet', 0]}, [(('singlet', 0), ('triplet', 0))],
              [], 'a pair naming a root not given'))
    for r, pairs, grounds, what in cases:
        try:
            ert.transition_densities(r, pairs, grounds, tile_gb=TILE_GB)
            ok &= check(False, f'{what} is refused')
        except ValueError as exc:
            ok &= check(True, f'{what} is refused', str(exc)[:60])
    return ok


class _GatherLog(SimulatedComm):
    """A simulated rank that counts the arrays its allgathers carry."""

    def __init__(self, comm):
        super().__init__(comm._world, comm._rank)
        self.arrays = 0

    def allgather(self, obj):
        stack = [obj]
        while stack:
            x = stack.pop()
            if isinstance(x, np.ndarray):
                self.arrays += 1
            elif isinstance(x, (list, tuple)):
                stack.extend(x)
        return super().allgather(obj)


def check_pass():
    """Check 5."""
    shapes, ntiles = {'a': (3, 4), 'b': (5,)}, 7
    rng = np.random.default_rng(5)
    adds = [{'a': rng.standard_normal((3, 4)), 'b': rng.standard_normal(5)}
            for _ in range(ntiles)]

    def total(comm):
        log = None if comm is None else _GatherLog(comm)
        ranks = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        acc = ee_ri_tiles.Pass(SimpleNamespace(comm=log), shapes)
        for t in range(ntiles):
            if t % ranks == rank:
                acc.add(t, adds[t])
        return acc.total(), 0 if log is None else log.arrays

    ser = total(None)[0]
    ok = True
    for size in (2, 3):
        outs = run_simulated(total, size)
        same = all(np.array_equal(o[n], ser[n]) for o, _ in outs for n in shapes)
        ok &= check(same, f'a pass over {size} simulated ranks gives the serial bits')
        ok &= check(all(k == 0 for _, k in outs),
                    f'over {size} ranks no addend is gathered',
                    str([k for _, k in outs]))
    return ok


def run():
    """Every check; True when all passed."""
    mf = rhf_df(WATER_ORIGIN, '6-31g')
    mol = mf.mol
    f = ee_ri_fold.bare_factors(mol, mf, frozen=1, auxbasis=AUX)
    ok = check_against_dense(f)
    ok &= check_ranks(mol, mf)
    ok &= check_refusals(mol, mf, f)
    ok &= check_pass()
    return ok


def test_ee_ri_transition():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
