"""The state solvers a solvation loop drives converge below its charge
tolerance (`Numerics.solved_for_the_loop`, LOOP_SOLVER_TOL_FRACTION): the
charges answer the state's density, whose error is linear in the roots'
residual, while the roots' energies need only its square.

  1. the rule: the named solver's residual -- the ADC roots'
     (adc.residual_tol) or the BSE's (bse.conv_tol) -- becomes
     LOOP_SOLVER_TOL_FRACTION of loop_tol_q, derived with its reason, and
     only that section moves (a realization refuses a setting it does not
     read); a Numerics already that tight is returned as it is.
  2. the drivers: the solvated driver's ADC(2) and GW-BSE Numerics carry the
     rule at the photophysics loop tolerance; the stage command line carries
     it under scheme B and refuses a --bse-conv-tol above the bar; the GW-BSE
     realization reads every setting the rule moves.
  3. the stall it removes: the ADC(2) singlet of water aug-cc-pVDZ in toluene
     (aug-cc-pvdz-ri, frozen 1; the fold iterated as on a molecule of DMABN's
     size, adc.dense_limit 0), loop_tol_q 1e-9: with the roots at the bar of
     their energies alone its charges still move above loop_tol_q after G has
     settled below loop_tol_energy, and the loop runs to its cap (12 cycles);
     with the rule it converges well before it.

Run: python -m pytest -s tests/test_loop_solver_tolerance.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.numerics.constants import (LOOP_SOLVER_TOL_FRACTION,
                                         PHOTOPHYSICS_LOOP_TOL_Q)
from src.Base.numerics.numerics import Numerics
from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.gwbse.surface_factory import dispatch_table, reads_field
from src.gradients.state_models import HFState, MP2State
from src.photophysics import cli, solvated
from tests.adc_common import ATOMS, NF, check, make_scf_loop, molecule
from tests.photophysics_cases import argv, parser

LOOP = {'loop_tol_q': 1e-9, 'loop_tol_energy': 1e-11, 'loop_max_cycle': 12}


def test_the_rule():
    plain = Numerics(solvent={'loop_tol_q': 1e-9})
    bar = LOOP_SOLVER_TOL_FRACTION * 1e-9
    for solver, field in (('adc', 'adc.residual_tol'), ('bse', 'bse.conv_tol')):
        loop = plain.solved_for_the_loop(solver)
        assert loop.parent == plain.identity() and 'charge tolerance' in loop.reason
        assert [c[0] for c in loop.change] == [field]
        assert loop.solved_for_the_loop(solver) is loop
    assert plain.solved_for_the_loop('adc').bse.conv_tol == plain.bse.conv_tol
    assert plain.solved_for_the_loop('bse').adc.residual_tol is None
    tight = Numerics(solvent={'loop_tol_q': 1e-9},
                     adc={'residual_tol': 1e-12}, bse={'conv_tol': 1e-12})
    assert all(tight.solved_for_the_loop(s) is tight for s in ('adc', 'bse'))
    with pytest.raises(ValueError, match='solver'):
        plain.solved_for_the_loop('cis')


#: the realization the GW-BSE drivers build (ISDF space-time on a DFT mean field)
GWBSE_ROW = next(r for r in dispatch_table()
                 if r.key() == ('dft', 'Excitation', 'space-time', 'isdf'))


def _read_by_gwbse(num):
    """Every field the rule moved is one the GW-BSE realization reads (it
    refuses a setting it does not read)."""
    return num.change and all(reads_field(GWBSE_ROW, c[0]) for c in num.change)


def test_the_drivers_carry_it(tmp_path):
    xyz = tmp_path / 'w.xyz'
    xyz.write_text('3\n\n' + '\n'.join(f'{a} {x} {y} {z}' for a, (x, y, z) in ATOMS) + '\n')
    bar = LOOP_SOLVER_TOL_FRACTION * PHOTOPHYSICS_LOOP_TOL_Q
    for extra in ([], ['--method', 'gwbse', '--isdf-grid', 'G1']):
        args = solvated._parse_args(['--xyz', str(xyz), '--basis', 'cc-pvdz', '--solvent',
                                     'toluene', '--out', str(tmp_path / 'r.json'), *extra])
        solvated._resolve_basis(args)
        mol = molecule(ATOMS, basis='cc-pvdz')
        solvated._check_args(args, mol)
        num = (solvated.adc2_numerics(args) if args.method == 'adc2'
               else solvated.gwbse_numerics(args, mol))
        if args.method == 'adc2':
            assert num.adc.residual_tol == bar
        else:
            assert num.bse.conv_tol == bar and _read_by_gwbse(num)
    parse = parser().parse_args
    loop = ['--environment', 'toluene', '--solvent-scheme', 'scheme-b-eq']
    num = cli.production_numerics(parse(argv(*loop)))
    assert num.bse.conv_tol == bar and _read_by_gwbse(num)
    assert cli.production_numerics(parse(argv())).bse.conv_tol > bar
    with pytest.raises(SystemExit, match='bse-conv-tol'):
        cli.production_numerics(parse(argv(*loop, '--bse-conv-tol', '1e-8')))


def _loop(numerics):
    mol = molecule(ATOMS, basis='aug-cc-pvdz')
    field = ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))
    state = ADC2State(MP2State(HFState(mol, make_scf_loop), NF, 'aug-cc-pvdz-ri',
                               numerics=numerics), 'singlet', 0, 3, None, follow='index')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sl.SolvationNotConverged)
        return sl.equilibrate(state, field, numerics=numerics)


def test_the_stall_it_removes():
    plain = Numerics(solvent=LOOP, adc={'dense_limit': 0})
    stalled, solved = _loop(plain), _loop(plain.solved_for_the_loop('adc'))
    dq = [h[2] for h in stalled.history]
    dG = abs(stalled.history[-1][1] - stalled.history[-2][1])
    print(f'roots at the energy bar: {len(dq)} cycles, last max|dq| {dq[-3:]}, |dG| {dG:.1e}; '
          f'below the loop: {len(solved.history)} cycles')
    assert not stalled.converged and len(dq) == LOOP['loop_max_cycle']
    assert dG < LOOP['loop_tol_energy'] and min(dq[-3:]) > LOOP['loop_tol_q']
    assert check(solved.converged and len(solved.history) <= 8,
                 'with the roots below the loop its charges converge',
                 f'{len(solved.history)} cycles')
    assert abs(solved.energy - stalled.energy) < 1e-10


if __name__ == '__main__':
    sys.exit(pytest.main(['-s', __file__]))
