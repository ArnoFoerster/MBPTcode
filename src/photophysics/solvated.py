"""Solvated absorption, emission and adiabatic energies of the lowest ADC(2)
or GW-BSE singlet and triplet of one molecule, in the gas phase and in a
continuum, written as one JSON record.

    python -m src.photophysics solvated --xyz mol.xyz \
        --basis cc-pvdz --auxbasis cc-pvdz-ri --jk-auxbasis cc-pvdz-jkfit \
        --environment toluene --saddle-check 1 --out record.json
    python -m src.photophysics solvated --method gwbse --refreeze 1 \
        --saddle-check 1 --grid-accuracy G1 --xyz mol.xyz --basis cc-pvdz \
        --environment toluene --out record.json

from the repository root (or with it on PYTHONPATH); `main(argv)` is the
same run from Python. The threads are the environment's (no --threads). The
options this driver shares with the TADF run (`cli`: --basis, --basis-for,
--environment, --grid-accuracy, --isdf-test-set, --pivot-points, --xc-grid,
--refreeze, --saddle-check, --loop-tol-q, --loop-tol-energy,
--loop-tolerance) have one name, type and meaning on both, and a required
one has no default here either.

--environment names the continuum: a solvent of pyscf's SMD table, or any
medium by its two constants, eps=EPS_S,n2=N2 (src.environments.medium), on
the cavity --cavity names; 'gas' runs the gas phase alone. Every state
model, loop and SCF of the run reads that one medium.

For each medium (gas, and the solvent if given):

    S0 minimum      the ground state (MP2; the mean-field reference for
                    GW-BSE) optimized (equilibrium-solvated: its
                    reaction field self-consistent with its relaxed density)
    absorption      at the S0 minimum, each state in the slow charges of S0
                    (src.environments.solvation_loop.vertical_energies)
    X minimum       S1 and T1 optimized on their equilibrium-solvated surface
                    from the S0 minimum (a seeded nudge breaks its symmetry)
    emission        at the X minimum, S0 in the slow charges of X
    adiabatic       G_eq(X at its minimum) - G_eq(S0 at its minimum); dEST =
                    adiabatic S1 - adiabatic T1 (no zero-point energies)

S1 and T1 are the lowest root of each spin at every geometry (--track index,
the adiabatic states); --track overlap follows one root's character instead.
The S1 / T1 walks start START_NUDGE_BOHR off the S0 minimum (a symmetric
start can only reach the symmetric stationary point). Every minimum (S0, S1,
T1) is saddle-checked under --saddle-check 1 (required, no default): the
Hessian of G_eq from 6N central differences of its force, and at a saddle a
step along its negative mode and a second walk (properties.excitations.saddle_check, the check the
photophysics stages take); the record keeps the check. The loops of
the walks and of the Hessian run at --loop-tol-q / --loop-tol-energy; with
--loop-tolerance gradient-tied a walk's loop stops instead once its
estimated force error is LOOP_FORCE_FRACTION of the walk's gradient
(SolvatedSurface's tie, the photophysics stages' option): the first gradient
of every surface and any below the walk's convergence bar run to the
tolerances, and the Hessian of the saddle check takes its forces on the
untied twin (SolvatedSurface.untied), every loop to the tolerances. Each
minimum's record keeps the walk's loop log (loop_log). The option is a
setting of the record (and its resume key); a record made before it existed
ran 'fixed'.

ADC(2) runs on the folded route (eeADC), forces from the dense reference
(src.gradients.df_correlation_force), so this is for molecules of up to
a few hundred basis functions. The singlet carries the LR term at eps_inf.
--spin-scaling scs|sos runs SCS- or SOS-ADC(2) on SCS- or SOS-MP2 with the
same factors (ADC2_SPIN_SCALING): every state model of the run, its loop,
relaxed density and force are the scaled ones (the triplet is the Ms = 0
component, MP2State's spin_scaling).

--solvent-arm names how an ADC(2) state meets the continuum (SOLVENT_ARMS):
'ss' (the default) the state-specific loop alone (no LR term: the singles
block sees the bare interaction; the medium enters through the loop's
charges only), 'ss+lr' the loop with the singlet's LR term at eps_inf. A
triplet has no LR term, so its numbers are the same in both arms. The arm is
a setting of a record in a solvent (and its resume key); a record made
before the option existed ran 'ss+lr'. The gas phase has no arm.

--method gwbse: G0W0-BSE on the --reference mean field (scheme B in the loop,
src.gradients.gwbse.state_model), on the --grid-accuracy level (no
default grid), the Casida step by --bse-solver, on the Tamm-Dancoff kernel
(--bse-kernel tda, the default for walks: a full-BSE triplet on a Hartree-Fock
reference turns (A - B)-unstable as a walk distorts the molecule; full is a
comparison). A GW-BSE surface is fixed by
the chain its first evaluation builds: a walk (and its saddle check) moves
that one model to every geometry (GWBSEState.at), and each vertical energy
builds its own at its geometry. --track index is the
chain's index (bse.track None), overlap its state following.

Refreeze (--refreeze 1; --refreeze 0 switches it off; required with --method
gwbse, no default): a GW-BSE
S1 / T1 walk freezes the chain's conventions at its start, so each minimum
is refrozen as the gas walks are (properties.optimize.relax(refreeze=1)):
the surface is rebuilt at the minimum (SolvatedSurface.refreeze: a fresh
chain there, as a vertical's), the drift E_new(R*) - E_old(R*) is measured
at that one geometry and the state walks again on the rebuilt surface while
|drift| > REFREEZE_TOL_MEV, at most REFREEZE_MAX_PASSES passes. The minimum
records refreeze ('measured', 'not measured', 'nothing frozen' or 'not converged'),
refreeze_drift_meV, refreeze_shift_bohr, refreeze_converged and
refreeze_passes (each pass's drift, shift and walk convergence); the saddle
check and G are taken on the last rebuilt surface. A drift still above the
tolerance after the last pass is warned about, the minimum kept and flagged
(`flags`: 'refreeze_not_converged'; the policy: refuse inputs, warn numerics).
ADC(2), and the GW-BSE reference (S0), freeze nothing: their
walks are not refrozen and are what they were.
walk_vs_vertical_G_Ha in the record is G_eq of the state at its minimum on
the walk's surface minus the same on the vertical's (for ADC(2) a
loop-tolerance number). On a refrozen GW-BSE walk it is the drift a further
pass would measure (the last surface was built at the previous pass's
minimum): flagged (walk_vs_vertical_flag) and warned about above
REFREEZE_TOL_MEV, the bar of the refreeze loop; on a walk that was not
refrozen above WALK_VERTICAL_G_FLAG_FACTOR loop tolerances.
Restartable: each finished piece is written to the record as it ends, and a
rerun skips what the record holds. A rerun on a record made with other
settings (RESUME_KEY: geometry -- the sha256 of the xyz file's content, not
its path --, bases, solvent, frozen core, roots,
following, saddle check, loop tolerances, code) is refused: its pieces and
the new ones would not belong to one calculation. The code is
`src.Base.utils.record_io.code_id`: the short commit, with '+dirty:' and a digest
of the edit on an edited tree; a record stamped with a full commit
resumes on a clean checkout of that commit.
Numerical events warn, are flagged and the run carries on (the policy: refuse
inputs, warn numerics). A walk that does not reach its
minimum is kept as `S0` / `minimum` with converged false and the flag
'walk_not_converged' (no saddle check is taken there); a solvation loop that
does not converge, an SCF that stops short, an unconverged or lost ADC(2)
root are flagged by the surface (SolvatedSurface.flags) and a minimum's
`flags` names those at its last geometry. Every number carries the names of
the flags it is built on: absorption_flags, emission_flags, adiabatic_flags,
dEST_adiabatic_flags (empty when clean). A rerun walks an unconverged
minimum again (e.g. with a larger --maxiter) and drops what was built on it.

A walk interrupted part way (a job's time limit) is not lost: every accepted
step writes the walk's state to the store `<out>.walks/` (`store.Store.at`),
one progress record per medium and state (`gas_S1_progress`,
`solvent_T1_progress`, ...), and a rerun starts that walk again from its
last accepted geometry, on its optimizer Hessian, not nudged, on a surface
built there (`walks.restart_from`, as the photophysics stages restart a
scheme-B walk; a GW-BSE surface freezes its conventions there, so the
restarted walk stands on another surface than the first; the refreeze loop
at its minimum measures and removes that, as for any walk). The minimum
records `restarted_from` (the pass, the accepted steps and G at the restart
geometry); the progress is set aside into `<out>.walks/finished/` once the
minimum is in the record. Progress of another calculation (settings, medium
or state: the record's settings, the code apart, and the state are its
stamp) is refused (`store.StoreRefused`); progress from another start is
walked again from this one. A checkpoint of the earlier form beside the
record (`<out>.walk.<medium>.<state>.json`) is refused: remove it, and that
walk starts again.

A Kohn-Sham reference is integrated on --xc-grid (pyscf's grid level; omitted
is the Numerics default, XC_GRID_LEVEL). GW-BSE reads the frozen shift
beyond the explicit quasiparticle set (--outside-scissor) and the residue
route (--residues) from the command line, each the Numerics default when
omitted. Every molecule's memory budget (MB per rank, its max_memory, which
pyscf and the tiled forces' memory check read) is --max-memory, else this
rank's share of the SLURM allocation (allocation_max_memory_mb), else
PYSCF_MAX_MEMORY; with none of them the budget is unknown: pyscf's default is
kept, warned about and flagged 'memory_budget_unknown' (the record's
`run_flags`), since a tiled force above it is then refused.

GW-BSE takes the AO-product fit, which keeps the diffuse products of an
augmented basis, on points pivoted from the --grid-accuracy level's parent grid:
--pivot-points of them, else the default count (`ao_product_fit.pivot_count`).
The Duchemin-Blase fit on the tabulated points is --isdf-test-set
duchemin_blase:

    python -m src.photophysics solvated --method gwbse --reference lrc-wpbeh \
        --basis aug-cc-pvdz --jk-auxbasis aug-cc-pvdz-jkfit --grid-accuracy G3 \
        --pivot-points 3500 \
        --refreeze 1 --saddle-check 1 --xyz mol.xyz --environment toluene \
        --out record.json

The mean field stays density-fitted (--jk-auxbasis); only the GW-BSE factors
are interpolated. The points are pivoted where a surface is built, so a walk
keeps them (they move with their atoms) and a refreeze pivots again at the
minimum.
--distribute runs the calculation over the ranks of the launch (rank 0
writes the record and the walks' progress, every rank holds rank 0's
numbers).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import time

from pyscf import dft, gto, lib, scf

from src.SingleReference.common.active_space import frozen_core
from src.Base.basis.basis_map import atom_basis_names, molecule_basis, parse_basis_for
from src.Base.basis.ri_fallback import paired_auxbasis, paired_jk_auxbasis
from src.Base.numerics.constants import (ADC2_SPIN_SCALING, HARTREE_TO_EV,
                                         ISDF_DEFAULT_TEST_SET, ISDF_TEST_SETS,
                                         LOOP_FORCE_FRACTION,
                                         PHOTOPHYSICS_LOOP_TOL_ENERGY,
                                         PHOTOPHYSICS_LOOP_TOL_Q,
                                         SCF_MAX_CYCLE, SCF_OPTIMIZATION_CONV_TOL,
                                         STATE_MODEL_SCF_GRAD_TOL,
                                         START_NUDGE_BOHR, XC_GRID_LEVEL)
from src.Base.numerical_flags import flag
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm, is_root
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_vocabulary import GAS
from src.Base.utils.record_io import code_id, same_code
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.environments.medium import (CAVITY_CONVENTIONS, DEFAULT_CAVITY,
                                     parse as parse_medium)
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from src.properties.excitations import relaxation_fields, relaxation_flags
from src.properties.optimize import derived_walk, relax, resolved_conv
from src.gradients.solvated_surface import SolvatedSurface
from src.Base.geometry.geometry import molecule_at
from src.photophysics import cli
from src.photophysics.store import Store, on_root
from src.photophysics.walks import (checked_minimum, progress_writer,
                                    restart_from, walk_progress)
from src.photophysics.state_specific import (gwbse_states, vertical,
                                             walk_vs_vertical)

#: How an ADC(2) state meets the continuum: the state-specific loop with the
#: singlet's LR term, or alone (module docstring).
SOLVENT_ARMS = ('ss', 'ss+lr')
#: The walks' convergence bar on the force (`relax`'s default), below which a
#: loop tied to the gradient runs to its tolerances (module docstring).
WALK_GRAD_MAX = resolved_conv(None)['opt_grad_max']
#: The ADC(2) options' values when not given (GW-BSE takes none of them).
ADC2_DEFAULTS = dict(auxbasis='cc-pvdz-ri', nroots=3, solvent_arm=SOLVENT_ARMS[0])
#: The options of one method only, refused with the other.
ADC2_ONLY = ('auxbasis', 'nroots', 'frozen', 'spin_scaling', 'solvent_arm')
GWBSE_ONLY = ('reference', 'grid_accuracy', 'bse_solver', 'bse_kernel', 'refreeze',
              'outside_scissor', 'residues', 'xc_grid', 'isdf_test_set',
              'pivot_points')
#: The settings a record may lack, and the value a record that lacks one was
#: made with.
LATER_SETTINGS = dict(outside_scissor=None, residues=None, xc_grid=None,
                      isdf_test_set=None, pivot_points=None, cavity=DEFAULT_CAVITY,
                      spin_scaling=None, solvent_arm=None, loop_tolerance='fixed')


def _parse_args(argv=None):
    """The command-line options of `argv` (omitted: the command line); the
    options the TADF run also takes come from its groups (`cli`: one name,
    one type and one meaning). args.solvent is the continuum --environment
    names, None in the gas phase; the loop's tolerances omitted are
    PHOTOPHYSICS_LOOP_TOL_Q, PHOTOPHYSICS_LOOP_TOL_ENERGY and 'fixed'."""
    # no abbreviations: a retired option (--solvent, --xc) is refused, not
    # read as the one it begins (--solvent-arm, --xc-grid)
    p = argparse.ArgumentParser(prog='python -m src.photophysics solvated',
                                description=__doc__.split('\n')[0], allow_abbrev=False)
    p.add_argument('--xyz', required=True)
    p.add_argument('--method', choices=('adc2', 'gwbse'), default='adc2')
    p.add_argument('--reference', default=None,
                   help='gwbse: the mean field (default hf, or a functional)')
    cli.add_factor_options(p, required=False)
    p.add_argument('--bse-solver', default=None, help="gwbse: the chain's Casida solver")
    p.add_argument('--bse-kernel', choices=('tda', 'full'), default=None,
                   help='gwbse: the BSE kernel (default tda: walks run on the Tamm-Dancoff '
                        'BSE, whose triplets have no (A - B) instability; full is a comparison)')
    p.add_argument('--outside-scissor', default=None,
                   help='gwbse: the frozen shift beyond the explicit set')
    p.add_argument('--residues', default=None,
                   help="gwbse: the route of Sigma's real-axis residues")
    cli.add_xc_grid_option(p)
    p.add_argument('--distribute', action='store_true',
                   help='run over the ranks of the launch')
    p.add_argument('--max-memory', type=int, default=None,
                   help='memory budget of one rank in MB (default: its share of '
                        'the SLURM allocation, else PYSCF_MAX_MEMORY)')
    cli.add_basis_options(p)
    p.add_argument('--auxbasis', default=None,
                   help="adc2: default cc-pvdz-ri; with --basis-for each element's -ri")
    p.add_argument('--jk-auxbasis', default=None,
                   help="default cc-pvdz-jkfit; with --basis-for each element's -jkfit")
    cli.add_environment_option(p)
    p.add_argument('--cavity', default=None, choices=sorted(CAVITY_CONVENTIONS),
                   help='the cavity convention, radius set and scale (default '
                        'the one --environment names, else DEFAULT_CAVITY)')
    p.add_argument('--frozen', type=int, default=None,
                   help='adc2: default the core shells')
    p.add_argument('--nroots', type=int, default=None, help='adc2: default 3')
    p.add_argument('--spin-scaling', choices=tuple(ADC2_SPIN_SCALING), default=None,
                   help='adc2: spin-component scaled ADC(2) on the same-scaled MP2 '
                        '(default none: ADC(2))')
    p.add_argument('--solvent-arm', choices=SOLVENT_ARMS, default=None,
                   help="adc2: 'ss' (default) the state-specific loop alone, "
                        "'ss+lr' the loop with the singlet's LR term")
    p.add_argument('--maxiter', type=int, default=100)
    p.add_argument('--states', nargs='+', default=['singlet', 'triplet'])
    p.add_argument('--track', choices=('index', 'overlap'), default='index',
                   help='root picking (index: adiabatic, the lowest root of each '
                        'spin; overlap: follow one state)')
    p.add_argument('--media', nargs='+', choices=('gas', 'solvent'),
                   default=['gas', 'solvent'],
                   help='which of gas and the --environment continuum to run (one '
                        'record file each when they run in separate processes)')
    cli.add_walk_policies(p, refreeze_required=False)
    cli.add_loop_options(p)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    environment = cli.environment_of(args)
    args.solvent = None if environment == GAS else environment
    if args.loop_tol_q is None:
        args.loop_tol_q = PHOTOPHYSICS_LOOP_TOL_Q
    if args.loop_tol_energy is None:
        args.loop_tol_energy = PHOTOPHYSICS_LOOP_TOL_ENERGY
    if args.loop_tolerance is None:
        args.loop_tolerance = 'fixed'
    args.saddle_check = bool(args.saddle_check)
    return args


RESUME_KEY = ('xyz_sha256', 'basis', 'auxbasis', 'jk_auxbasis', 'solvent', 'frozen',
              'nroots', 'follow', 'saddle_check', 'tol_q', 'tol_energy', 'code',
              'state_method', 'xc', 'isdf_grid', 'bse_solver', 'bse_kernel',
              'refreeze') + tuple(LATER_SETTINGS)


def _settings(args):
    """The run's settings that every number of the record depends on; the
    geometry by the xyz file's content (xyz_sha256), its path for the reader."""
    with open(args.xyz, 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return dict(xyz=os.path.abspath(args.xyz), xyz_sha256=digest, basis=_basis_setting(args),
                auxbasis=args.auxbasis, jk_auxbasis=args.jk_auxbasis,
                solvent=args.solvent, frozen=args.frozen, nroots=args.nroots,
                follow=args.track, saddle_check=args.saddle_check,
                tol_q=args.loop_tol_q, tol_energy=args.loop_tol_energy, code=code_id(),
                state_method=args.method, xc=args.reference, isdf_grid=args.grid_accuracy,
                bse_kernel=args.bse_kernel,
                bse_solver=args.bse_solver, refreeze=args.refreeze,
                **{k: getattr(args, k, LATER_SETTINGS[k]) for k in LATER_SETTINGS
                   if k not in ('cavity', 'solvent_arm')},
                cavity=_cavity(args),
                solvent_arm=None if args.solvent is None else getattr(args, 'solvent_arm', None))


def _older_setting(rec, key):
    """The value of a later setting `key` that a record lacking it was made
    with: an ADC(2) record in a solvent ran the singlet with its LR term."""
    if key == 'solvent_arm' and rec.get('state_method') == 'adc2' and rec.get('solvent'):
        return 'ss+lr'
    return LATER_SETTINGS.get(key, '<absent>')


def _basis_setting(args):
    """The basis a record is keyed on: the name or per-element map, or each
    atom's set where a per-atom basis labels the atoms (the xyz file holds
    the geometry without them)."""
    if isinstance(args.basis, dict) and any('@' in key for key in args.basis):
        return atom_basis_names(args.atom, args.basis)
    return args.basis


def _cavity(args):
    """The run's cavity convention: --environment's or --cavity, else the default."""
    medium = None if args.solvent is None else parse_medium(args.solvent, getattr(args, 'cavity', None))
    return medium.cavity if medium is not None else getattr(args, 'cavity', None) or DEFAULT_CAVITY


def _start_record(args):
    """The record: the file's content if present (refused when made with
    other settings, RESUME_KEY), with the run's settings."""
    settings = _settings(args)
    rec = on_root(lambda: _read_json(args.out))
    if rec is None:
        rec = {}
    else:
        made = {k: rec[k] if k in rec else _older_setting(rec, k) for k in RESUME_KEY}
        differ = {k: (made[k], settings[k]) for k in RESUME_KEY
                  if made[k] != settings[k]
                  and not (k == 'code' and same_code(made[k], settings[k]))}
        if differ:
            raise SystemExit(f'{args.out} was made with other settings (record, this '
                             f'run): {differ}; write to another --out')
    # an unnamed test set is the AO-product fit (ISDF_DEFAULT_TEST_SET)
    test_set = getattr(args, 'isdf_test_set', None) or ISDF_DEFAULT_TEST_SET
    fit = (f', {test_set} fit'
           + ('' if test_set != 'ao_products' else
              ' on the default count of pivoted points'
              if getattr(args, 'pivot_points', None) is None else
              f' on {args.pivot_points} pivoted points'))
    scaled = getattr(args, 'spin_scaling', None)
    adc = ('ADC(2) (folded, DF) on DF-MP2' if scaled is None else
           f'{scaled.upper()}-ADC(2) (folded, DF) on DF-{scaled.upper()}-MP2, '
           f'(c_os, c_ss) = {tuple(ADC2_SPIN_SCALING[scaled])}')
    if args.method == 'adc2' and args.solvent is not None:
        adc += ('; singlets with the LR term at eps_inf'
                if (getattr(args, 'solvent_arm', None) or SOLVENT_ARMS[0]) == 'ss+lr'
                else '; no LR term (state-specific loop only)')
    what = (adc if args.method == 'adc2' else
            f'G0W0-BSE (scheme B, ISDF {args.grid_accuracy}{fit}) on {args.reference}')
    medium = None if args.solvent is None else parse_medium(args.solvent, getattr(args, 'cavity', None))
    rec.update(settings, host=platform.node(),
               medium=None if medium is None else medium.record(),
               method=f'{what}, '
               'equilibrium/non-equilibrium state-specific PCM with relaxed '
               'densities (src.environments.solvation_loop)')
    return rec


def _read_json(path):
    """The JSON tree at path, or None where there is no file."""
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _save(rec, out):
    """Write the record to out atomically: rank 0 writes, and a failure
    raises on every rank (`store.on_root`)."""
    def act():
        with open(out + '.tmp', 'w') as fh:
            json.dump(rec, fh, indent=1)
        os.replace(out + '.tmp', out)
    on_root(act)


def _make_scf(m, jk_auxbasis, xc='hf', xc_grid=None, converged=None):
    """The density-fitted RHF (or RKS of xc on pyscf's XC grid level xc_grid)
    of m, converged as `converged` (a Numerics' scf section: conv_tol,
    conv_tol_grad, max_cycle; the ADC(2) route's, adc2_numerics) or, without
    one, tightly (1e-12, orbital gradient 1e-10, 200 cycles)."""
    mf = (scf.RHF(m) if xc == 'hf' else dft.RKS(m, xc=xc)).density_fit(
        auxbasis=jk_auxbasis)
    if xc != 'hf':
        mf.grids.level = xc_grid
    if converged is None:
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    else:
        mf.conv_tol, mf.conv_tol_grad = converged.conv_tol, converged.conv_tol_grad
        mf.max_cycle = converged.max_cycle
    mf.verbose = 0
    return mf


def _loop_settings(args):
    """The loop's section of the run's Numerics: its tolerances, and a walk's
    tie to its gradient under --loop-tolerance gradient-tied
    (SolventNumerics.loop_force_fraction; module docstring)."""
    tied = getattr(args, 'loop_tolerance', 'fixed') == 'gradient-tied'
    return {'loop_tol_q': args.loop_tol_q, 'loop_tol_energy': args.loop_tol_energy,
            'loop_force_fraction': LOOP_FORCE_FRACTION if tied else None}


def adc2_numerics(args):
    """The one Numerics of an ADC(2) run: the loop's tolerances and the mean
    field's convergence of a walk (the GW-BSE walk route's energy test
    SCF_OPTIMIZATION_CONV_TOL, the orbital gradient a state model needs at
    most, STATE_MODEL_SCF_GRAD_TOL, SCF_MAX_CYCLE cycles; a cap reached is
    warned about and flagged by the state model); the ADC roots converged
    below the loop's charge tolerance (`Numerics.solved_for_the_loop`)."""
    return Numerics(solvent=_loop_settings(args),
                    scf={'conv_tol': SCF_OPTIMIZATION_CONV_TOL,
                         'conv_tol_grad': STATE_MODEL_SCF_GRAD_TOL,
                         'max_cycle': SCF_MAX_CYCLE}).solved_for_the_loop('adc')


def _resolve_basis(args):
    """--basis with --basis-for as the molecule's per-element or per-atom
    basis (`molecule_basis`; args.atom the xyz file's atoms, labelled for a
    per-atom one), and the auxiliary sets that go with it: under a map each
    atom's -jkfit and -ri set follow its own basis, and one named set for
    every atom is refused, since it is matched to none of them. Without
    --basis-for the sets are the named ones, else cc-pvdz-jkfit (and
    ADC(2)'s default -ri)."""
    args.atom = args.xyz
    if getattr(args, 'basis_for', None):
        with open(args.xyz) as fh:
            lines = fh.read().splitlines()
        atoms = '; '.join(' '.join(line.split()[:4])
                          for line in lines[2:2 + int(lines[0].split()[0])])
        try:
            args.atom, args.basis = molecule_basis(atoms, args.basis,
                                                   *parse_basis_for(args.basis_for))
        except ValueError as err:
            raise SystemExit(f'--basis-for: {err}') from err
    if isinstance(args.basis, dict):
        named = [f'--{n.replace("_", "-")}' for n in ('auxbasis', 'jk_auxbasis')
                 if getattr(args, n) is not None]
        if named:
            raise SystemExit(f'{", ".join(named)} names one set for every element of '
                             f'a per-element basis; omit it to pair each element\'s '
                             f'auxiliary set with its own basis')
        args.jk_auxbasis = paired_jk_auxbasis(args.basis)
        if args.method == 'adc2':
            args.auxbasis = paired_auxbasis(args.basis)
    elif args.jk_auxbasis is None:
        args.jk_auxbasis = 'cc-pvdz-jkfit'


def _check_args(args, mol0):
    """Refuse the other method's options; fill this method's defaults."""
    if args.solvent is not None:
        try:
            parse_medium(args.solvent, getattr(args, 'cavity', None))
        except ValueError as err:
            raise SystemExit(f'--environment {args.solvent}: {err}') from err
    other = GWBSE_ONLY if args.method == 'adc2' else ADC2_ONLY
    given = [name for name in other if getattr(args, name, None) is not None]
    if given:
        raise SystemExit(f'--method {args.method} takes no '
                         + ', '.join('--' + g.replace('_', '-') for g in given))
    if args.method == 'adc2':
        for name, value in ADC2_DEFAULTS.items():
            if getattr(args, name, None) is None:
                setattr(args, name, value)
        if args.frozen is None:
            args.frozen = frozen_core(mol0)
        return
    if args.grid_accuracy is None:
        raise SystemExit('--method gwbse needs --grid-accuracy (no default grid)')
    if args.reference is None:
        args.reference = 'hf'
    if args.reference == 'hf' and getattr(args, 'xc_grid', None) is not None:
        raise SystemExit('--xc-grid sets the grid of a Kohn-Sham reference; '
                         'the Hartree-Fock reference has none')
    if args.reference != 'hf' and getattr(args, 'xc_grid', None) is None:
        args.xc_grid = XC_GRID_LEVEL
    if args.bse_kernel is None:
        args.bse_kernel = 'tda'
    if args.refreeze is None:
        raise SystemExit('--method gwbse needs --refreeze 0 or 1 (no default)')
    args.refreeze = bool(args.refreeze)


def gwbse_numerics(args, mol0):
    """The one Numerics of a GW-BSE run: the chain's grid (refused here when
    the radii table lacks an element) and its fit's test set and pivoted
    points, the quasiparticle options and solver, the reference's XC grid, the loop's
    tolerances; an option not given is the Numerics default (a pivot without
    the AO-product test set is refused there)."""
    bse = {'track': 'overlap' if args.track == 'overlap' else None}
    if args.bse_solver is not None:
        bse['solver'] = args.bse_solver
    counts = resolve_isdf_grid(args.grid_accuracy, args.basis,
                               elements=tuple(sorted(set(mol0.elements))))[0]
    qp = {key: getattr(args, name) for key, name in
          (('outside_scissor', 'outside_scissor'), ('residue_route', 'residues'))
          if getattr(args, name, None) is not None}
    scf_ = {} if getattr(args, 'xc_grid', None) is None else {'xc_grid': args.xc_grid}
    isdf = {'counts': counts}
    isdf.update({key: getattr(args, key) for key in ('isdf_test_set', 'pivot_points')
                 if getattr(args, key, None) is not None})
    if 'isdf_test_set' in isdf:
        isdf['test_set'] = isdf.pop('isdf_test_set')
    return Numerics(isdf=isdf, bse=bse, qp=qp, scf=scf_,
                    solvent=_loop_settings(args)).solved_for_the_loop('bse')


def _factories(args, medium, loop_kw):
    """(make_field, make_state) for one medium; make_state(kind) builds a state."""
    if medium == 'gas':
        make_field = NoMedium
    else:
        continuum = parse_medium(medium, getattr(args, 'cavity', None))
        make_field = lambda m: ContinuumField.from_environment(continuum.screening(m))

    def make_scf(m):
        return _make_scf(m, args.jk_auxbasis, args.reference or 'hf',
                         getattr(args, 'xc_grid', None))

    if args.method == 'gwbse':
        return make_field, gwbse_states(make_scf, loop_kw['numerics'],
                                        args.bse_kernel == 'tda')

    def make_adc2_scf(m):
        return _make_scf(m, args.jk_auxbasis, converged=loop_kw['numerics'].scf)

    arm = getattr(args, 'solvent_arm', None) or SOLVENT_ARMS[0]

    def make_state(kind):
        def build(m, field):
            mp2 = MP2State(HFState(m, make_adc2_scf), args.frozen, args.auxbasis,
                           numerics=loop_kw['numerics'],
                           spin_scaling=getattr(args, 'spin_scaling', None))
            if kind == 'S0':
                return mp2
            lr = (field.transition_environment()
                  if kind == 'singlet' and arm == 'ss+lr' else None)
            return ADC2State(mp2, kind, 0, args.nroots, lr, follow=args.track)
        return build
    return make_field, make_state


def _geom(m):
    """The atoms of m as [symbol, x, y, z] in Angstrom."""
    return [[m.atom_symbol(i), *map(float, m.atom_coord(i, unit='Angstrom'))]
            for i in range(m.natm)]


def _mol_of(g, basis, max_memory):
    """The molecule of a geometry list from _geom, its budget max_memory (MB)."""
    return gto.M(atom=[(a[0], tuple(a[1:])) for a in g], basis=basis, verbose=0,
                 max_memory=max_memory)


def memory_budget(asked=None, flags=None):
    """The memory budget of one rank (MB, module docstring): `asked`
    (--max-memory), else this rank's share of the SLURM allocation, else
    PYSCF_MAX_MEMORY; unknown otherwise: pyscf's default, warned about and
    flagged 'memory_budget_unknown' into `flags`."""
    if asked is not None:
        return int(asked)
    share = allocation_max_memory_mb(default=None)
    if share is not None:
        return int(share)
    if os.environ.get('PYSCF_MAX_MEMORY'):
        return int(lib.param.MAX_MEMORY)
    flag('memory_budget_unknown',
         f'no --max-memory, SLURM allocation or PYSCF_MAX_MEMORY: each rank keeps '
         f"pyscf's default of {lib.param.MAX_MEMORY} MB, and a tiled force whose "
         'estimate exceeds it is refused', RuntimeWarning, into=flags)
    return int(lib.param.MAX_MEMORY)


def _refreezes(args, kind):
    """1 when a walk of `kind` runs the refreeze loop (module docstring): a
    GW-BSE excited state's surface freezes its chain's conventions; ADC(2)
    and the GW-BSE reference freeze nothing."""
    return int(bool(args.refreeze) and args.method == 'gwbse' and kind != 'S0')


def _walk_record(info, asked):
    """A walk's entries in the record: the optimizer's fields under their one
    naming (`excitations.relaxation_fields`) and the refreeze passes, or
    'nothing frozen' where the surface freezes nothing (module docstring)."""
    out = relaxation_fields(info)
    if asked is None:
        out['refreeze'] = 'nothing frozen'
    elif asked:
        out['refreeze_passes'] = [dict(p, shift=float(p['shift']))
                                  for p in info.get('refreeze_passes') or []]
    return out


def _walk_store(args):
    """The store of the walks' progress: the directory <out>.walks (module
    docstring)."""
    return Store.at(args.out + '.walks')


def _progress_name(medium, key):
    """The progress record of the walk of state `key` ('S0', 'S1', 'T1') in
    `medium`: the gas phase's or the solvent's."""
    return f'{"gas" if medium == "gas" else "solvent"}_{key}_progress'


def _walk_stamp(args, kind):
    """What a walk's progress belongs to: the record's settings (the code
    apart, which the record itself checks) and the state; its start is
    checked apart (`walks.walk_progress`)."""
    settings = _settings(args)
    return dict({k: settings[k] for k in RESUME_KEY if k != 'code'}, kind=kind)


def _set_progress_aside(store, name):
    """Set the progress `name` aside into finished/ once its minimum is in
    the record: a rerun that walks that minimum again starts from its
    start."""
    store.set_aside([n for n in store.names() if n.split('.')[0] == name],
                    'finished')


def _refuse_retired_checkpoints(args):
    """Refuse a walk checkpoint of the earlier form beside the record
    (`<out>.walk.<medium>.<state>.json`), which this driver does not read
    (module docstring)."""
    old = on_root(lambda: sorted(glob.glob(glob.escape(args.out) + '.walk.*.json')))
    if old:
        raise SystemExit(f'{old[0]} is a walk checkpoint of an earlier form, which '
                         f'this driver does not read (the walks keep their progress '
                         f'in {args.out}.walks/): remove it, and that walk starts '
                         f'again from its start')


def _minimum(args, surf, start, nudge, kind, progress=None):
    """(mol, the record) of a walk on surf from start, refrozen where its
    surface freezes conventions (`_refreezes`) and saddle-checked when it
    converged (`excitations.saddle_check`, the photophysics stages' check);
    `flags` the names flagged at its last geometry (module docstring): the
    surface's, and the walk's own 'walk_not_converged',
    'refreeze_not_converged', 'saddle_check' and 'hessian_asymmetric', each warned about
    (`excitations.relaxation_flags`, `saddle_verdict`). The saddle check, its
    escape walk (refrozen as the first) and G are taken on the surface the
    last walk ran on.
    progress: (store, name) of the progress record the first walk writes
    after every accepted step and restarts from (module docstring); None
    keeps none."""
    t = time.time()
    refreeze = _refreezes(args, kind)
    asked = None if (args.method == 'adc2' or kind == 'S0') else refreeze
    first, restarted = {}, None
    if progress is not None:
        store, name = progress
        stamp = _walk_stamp(args, kind)
        first['checkpoint'] = progress_writer(store, name, stamp, start)
        state = walk_progress(store, name, stamp, start)
        if state is not None:
            template = start
            start, first['hess_init'], restarted = restart_from(
                state, lambda coords: molecule_at(template, coords))
            nudge = 0

    def walk(s, at, nudge=0, **kw):
        return relax(s, at, nudge=nudge, refreeze=refreeze,
                     numerics=derived_walk(s, 'the command line caps the walk',
                                           max_cycle=args.maxiter), **kw)
    m, info = walk(surf, start, nudge, **first)
    final = info.get('final_surface', surf)
    out = _walk_record(info, asked)
    if restarted is not None:
        out['restarted_from'] = restarted
    flags = relaxation_flags(kind, info)
    if args.saddle_check and out['converged']:
        m, again, chk, checked = checked_minimum(final, m, walk, kind)
        first = chk['first']
        out['saddle_check'] = dict(
            status=chk['status'],
            lowest_curvature=first['lowest_curvature'],
            negative_modes=first['negative_modes'],
            tolerance=first['tolerance'],
            hessian_asymmetry_rel=chk.get('hessian_asymmetry_rel'),
            hessian_asymmetric=chk.get('hessian_asymmetric'),
            energy_drop=chk.get('energy_drop'),
            second_lowest_curvature=(chk['second'] or {}).get('lowest_curvature'))
        if again is None:
            flags = dict(flags, **checked)
        else:
            out['first_walk'] = dict(out)
            out.update(_walk_record(again, asked))
            final = again.get('final_surface', final)
            flags = checked
    out.update(geometry=_geom(m), G=final.total_energy(m), wall_s=time.time() - t)
    # one log for the walk, its refreezes, the Hessian's twin and an escape walk
    out['loop_log'] = getattr(final, 'loop_log', [])
    # a refrozen surface shares the walk's flag log (SolvatedSurface.refreeze)
    out['flags'] = sorted(set(final.flags_at(m)) | set(flags))
    out['flagged_evaluations'] = len(final.flags)
    return m, out


def _store_minimum(rec, out, R, key, entry):
    """Store a walk's entry under R[key]; one that did not converge was warned
    about by `_minimum` and carries 'walk_not_converged' (module docstring)."""
    R[key] = entry
    R.pop(key + '_failed', None)
    _save(rec, out)


def _flag_names(*parts):
    """The sorted union of flag names: lists of names, or vertical_energies'
    {loop: names}."""
    out = set()
    for part in parts:
        for v in (part.values() if isinstance(part, dict) else [part]):
            out.update(v or [])
    return sorted(out)


# what a rerun drops with an unconverged minimum: everything built on it
_EMISSION_KEYS = ('emission_eV', 'emission_G', 'emission_wall_s', 'emission_flags',
                  'adiabatic_eV', 'adiabatic_flags', 'walk_vs_vertical_G_Ha',
                  'walk_vs_vertical_flag')


def _drop_unconverged(R):
    """A rerun walks an unconverged minimum again: drop it and what was built
    on it (module docstring); an old record's `_failed` entries go too."""
    for key in [k for k in R if k.endswith('_failed')]:
        R.pop(key)
    for X in (R.get('S1'), R.get('T1')):
        if isinstance(X, dict):
            X.pop('minimum_failed', None)
    if 'S0' in R and not R['S0'].get('converged'):
        R.clear()
        return
    for key in ('S1', 'T1'):
        X = R.get(key)
        if isinstance(X, dict) and 'minimum' in X and not X['minimum'].get('converged'):
            for k in ('minimum',) + _EMISSION_KEYS:
                X.pop(k, None)
            R.pop('dEST_adiabatic_eV', None)
            R.pop('dEST_adiabatic_flags', None)


def _ground_state(args, rec, R, medium, mol0, make_field, make_state, loop_kw,
                  store):
    """S0 minimum of one medium into R, unless the record holds it; its
    walk's progress in `store`."""
    if 'S0' not in R:
        surf = SolvatedSurface(mol0, make_field, make_state('S0'),
                               label=f'S0 {medium}', final_grad_max=WALK_GRAD_MAX,
                               **loop_kw)
        name = _progress_name(medium, 'S0')
        _store_minimum(rec, args.out, R, 'S0',
                       _minimum(args, surf, mol0, 0, 'S0', progress=(store, name))[1])
        _set_progress_aside(store, name)


def _absorption(args, rec, R, X, kind, mS0, make_field, make_state, loop_kw):
    """Vertical absorption at the S0 minimum into X, unless the record holds it."""
    if 'absorption_eV' not in X:
        t = time.time()
        r = vertical(make_field, make_state, kind, mS0, loop_kw)
        X.update(absorption_eV=r['absorption'] * HARTREE_TO_EV,
                 absorption_G=r['G'], absorption_wall_s=time.time() - t,
                 absorption_flags=_flag_names(R['S0'].get('flags'), r['flags']))
        _save(rec, args.out)


def _excited_minimum(args, rec, X, key, kind, medium, mS0, make_field, make_state,
                     loop_kw, store):
    """Minimum of the state on its surface from the S0 minimum, unless
    recorded; its walk's progress in `store`."""
    if 'minimum' not in X:
        surf = SolvatedSurface(mS0, make_field, make_state(kind),
                               label=f'{key} {medium}', final_grad_max=WALK_GRAD_MAX,
                               make_ground=make_state('S0'), **loop_kw)
        name = _progress_name(medium, key)
        _store_minimum(rec, args.out, X, 'minimum',
                       _minimum(args, surf, mS0, START_NUDGE_BOHR, kind,
                                progress=(store, name))[1])
        _set_progress_aside(store, name)


def _emission(args, rec, R, X, kind, make_field, make_state, loop_kw):
    """Vertical emission at the X minimum and the adiabatic energy, unless recorded."""
    if 'emission_eV' not in X:
        t = time.time()
        mX = _mol_of(X['minimum']['geometry'], args.basis, args.max_memory)
        r = vertical(make_field, make_state, kind, mX, loop_kw)
        X.update(emission_eV=r['emission'] * HARTREE_TO_EV,
                 emission_G=r['G'], emission_wall_s=time.time() - t,
                 emission_flags=_flag_names(X['minimum'].get('flags'), r['flags']))
        X['adiabatic_eV'] = (X['minimum']['G'] - R['S0']['G']) * HARTREE_TO_EV
        X['adiabatic_flags'] = _flag_names(X['minimum'].get('flags'),
                                           R['S0'].get('flags'))
        refrozen = X['minimum'].get('refreeze') in ('measured', 'not converged')
        X['walk_vs_vertical_G_Ha'], X['walk_vs_vertical_flag'] = walk_vs_vertical(
            kind, X['minimum']['G'], r['G']['eq_state'], refrozen, args.loop_tol_energy)
        _save(rec, args.out)


def _run_medium(args, rec, medium, mol0, loop_kw):
    """All pieces of one medium: S0, then absorption, minimum, emission per state."""
    R = rec.setdefault(medium, {})
    _drop_unconverged(R)
    make_field, make_state = _factories(args, medium, loop_kw)
    store = _walk_store(args)
    _ground_state(args, rec, R, medium, mol0, make_field, make_state, loop_kw,
                  store)
    mS0 = _mol_of(R['S0']['geometry'], args.basis, args.max_memory)
    for kind in args.states:
        key = 'S1' if kind == 'singlet' else 'T1'
        X = R.setdefault(key, {})
        _absorption(args, rec, R, X, kind, mS0, make_field, make_state, loop_kw)
        _excited_minimum(args, rec, X, key, kind, medium, mS0, make_field,
                         make_state, loop_kw, store)
        _emission(args, rec, R, X, kind, make_field, make_state, loop_kw)
    if 'S1' in R and 'T1' in R and 'adiabatic_eV' in R['S1'] and 'adiabatic_eV' in R['T1']:
        R['dEST_adiabatic_eV'] = R['S1']['adiabatic_eV'] - R['T1']['adiabatic_eV']
        R['dEST_adiabatic_flags'] = _flag_names(R['S1'].get('adiabatic_flags'),
                                                R['T1'].get('adiabatic_flags'))
        _save(rec, args.out)


def _print_summary(rec, media):
    """Print the eV entries of the record for the media run (rank 0)."""
    if not is_root():
        return
    print(json.dumps({m: {k: {kk: vv for kk, vv in v.items() if kk.endswith('eV')}
                          if isinstance(v, dict) else v
                          for k, v in rec[m].items()} for m in media}, indent=1))


def main(argv=None):
    """Parse `argv` (omitted: the command line) and run; the exit status."""
    args = _parse_args(argv)
    run_flags = []
    args.max_memory = memory_budget(args.max_memory, run_flags)
    _resolve_basis(args)
    mol0 = gto.M(atom=args.atom, basis=args.basis, verbose=0, max_memory=args.max_memory)
    _check_args(args, mol0)
    if args.method == 'gwbse':
        # one object: every GW-BSE model holds it, the loops are handed the same
        loop_kw = dict(numerics=gwbse_numerics(args, mol0))
    else:
        loop_kw = dict(numerics=adc2_numerics(args))
    comm = None
    if args.distribute:
        world, _, size = grid_comm()
        comm = None if size == 1 else world
    with distributed(comm):
        rec = _start_record(args)
        _refuse_retired_checkpoints(args)
    rec.update(max_memory_mb=args.max_memory, numerics=loop_kw['numerics'].to_record(),
               run_flags=sorted({f['flag'] for f in run_flags}))
    media = ((['gas'] if 'gas' in args.media else [])
             + ([args.solvent] if args.solvent and 'solvent' in args.media else []))
    with distributed(comm):
        for medium in media:
            _run_medium(args, rec, medium, mol0, loop_kw)
        _print_summary(rec, media)
    return 0
