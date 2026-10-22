"""Solvated absorption, emission and adiabatic energies of the lowest ADC(2)
or GW-BSE singlet and triplet of one molecule, in the gas phase and in a
continuum, written as one JSON record.

    python tools/solvation/solvated_photophysics.py --xyz mol.xyz \
        --basis cc-pvdz --auxbasis cc-pvdz-ri --jk-auxbasis cc-pvdz-jkfit \
        --solvent toluene --out record.json
    python tools/solvation/solvated_photophysics.py --method gwbse \
        --isdf-grid G1 --xyz mol.xyz --solvent toluene --out record.json

For each medium (gas, and the solvent if given):

    S0 minimum      the ground state (MP2; the mean-field reference for
                    GW-BSE) optimized (equilibrium-solvated: its
                    reaction field self-consistent with its relaxed density)
    absorption      at the S0 minimum, each state in the slow charges of S0
                    (src.Base.solvation_loop.vertical_energies)
    X minimum       S1 and T1 optimized on their equilibrium-solvated surface
                    from the S0 minimum (a seeded nudge breaks its symmetry)
    emission        at the X minimum, S0 in the slow charges of X
    adiabatic       G_eq(X at its minimum) - G_eq(S0 at its minimum); dEST =
                    adiabatic S1 - adiabatic T1 (no zero-point energies)

S1 and T1 are the lowest root of each spin at every geometry (--follow index,
the adiabatic states); --follow overlap follows one root's character instead.
The S1 / T1 walks start START_NUDGE_BOHR off the S0 minimum (a symmetric
start can only reach the symmetric stationary point). Every minimum (S0, S1,
T1) is saddle-checked (--saddle-check, default on): the Hessian of G_eq from
6N central differences of its force (properties.hessian.surface_hessian), and
at a saddle a step along its negative mode and a second walk
(properties.optimize.escape_saddle); the record keeps the check. The loops of
the walks and of the Hessian run at --tol-q / --tol-energy.

ADC(2) runs on the folded route (eeADC), forces from the dense reference
(src.SingleReference.df_correlation_force), so this is for molecules of up to
a few hundred basis functions. The singlet carries the LR term at eps_inf.

--method gwbse: G0W0-BSE on the --xc reference (scheme B in the loop,
src.gradients.gwbse_state_model), on the --isdf-grid accuracy level (no
default grid), the Casida step by --bse-solver, on the Tamm-Dancoff kernel
(--bse-kernel tda, the default for walks: a full-BSE triplet on a Hartree-Fock
reference turns (A - B)-unstable as a walk distorts the molecule; full is a
comparison). A GW-BSE surface is fixed by
the chain its first evaluation builds: a walk (and its saddle check) moves
that one model to every geometry (GWBSEState.at), and each vertical energy
builds its own at its geometry. --follow index is the
chain's index (bse.track None), overlap its state following.

Refreeze (--refreeze, the default; --no-refreeze switches it off): a GW-BSE
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
(`flags`: 'refreeze_not_converged'; refuse inputs, warn
numerics). ADC(2), and the GW-BSE reference (S0), freeze nothing: their
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
the new ones would not belong to one calculation.
Numerical events warn, are flagged and the run carries on
(refuse inputs, warn numerics). A walk that does not reach its
minimum is kept as `S0` / `minimum` with converged false and the flag
'walk_not_converged' (no saddle check is taken there); a solvation loop that
does not converge, an SCF that stops short, an unconverged or lost ADC(2)
root are flagged by the surface (SolvatedSurface.flags) and a minimum's
`flags` names those at its last geometry. Every number carries the names of
the flags it is built on: absorption_flags, emission_flags, adiabatic_flags,
dEST_adiabatic_flags (empty when clean). A rerun walks an unconverged
minimum again (e.g. with a larger --maxiter) and drops what was built on it.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings


def _repo_root(start):
    d = os.path.abspath(start)
    while not os.path.isdir(os.path.join(d, 'src')):
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit('no src/ above ' + start)
        d = parent
    return d


ROOT = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyscf import dft, gto, scf                                      # noqa: E402

from src.Base import solvation_loop as sl                            # noqa: E402
from src.Base.constants import (HARTREE_TO_EV, HARTREE_TO_MEV,     # noqa: E402
                                REFREEZE_TOL_MEV, START_NUDGE_BOHR,
                                WALK_VERTICAL_G_FLAG_FACTOR)
from src.Base.numerics import Numerics                               # noqa: E402
from src.Base.reaction_field import ContinuumField, NoMedium         # noqa: E402
from src.Base.separable_ri import resolve_isdf_grid                  # noqa: E402
from src.Base.solvent_screening import SolventScreening              # noqa: E402
from src.SingleReference.state_models import HFState, MP2State       # noqa: E402
from src.SingleReference.ADC.eeADC.ee_state_model import ADC2State   # noqa: E402
from src.gradients.gwbse_state_model import GWBSEState               # noqa: E402
from src.properties.hessian import surface_hessian                   # noqa: E402
from src.properties.optimize import escape_saddle, relax            # noqa: E402
from src.properties.solvated_surface import SolvatedSurface          # noqa: E402
from tools.common import code_id, frozen_core                        # noqa: E402

#: The ADC(2) options' values when not given (GW-BSE takes none of them).
ADC2_DEFAULTS = dict(auxbasis='cc-pvdz-ri', nroots=3)
#: The options of one method only, refused with the other.
ADC2_ONLY = ('auxbasis', 'nroots', 'frozen')
GWBSE_ONLY = ('xc', 'isdf_grid', 'bse_solver', 'bse_kernel', 'refreeze')


def _parse_args():
    """The command-line options."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--xyz', required=True)
    p.add_argument('--method', choices=('adc2', 'gwbse'), default='adc2')
    p.add_argument('--xc', default=None,
                   help='gwbse: the reference (default hf, or a functional)')
    p.add_argument('--isdf-grid', default=None,
                   help='gwbse: the ISDF accuracy level (G1, G2, G3) or shell counts')
    p.add_argument('--bse-solver', default=None, help="gwbse: the chain's Casida solver")
    p.add_argument('--bse-kernel', choices=('tda', 'full'), default=None,
                   help='gwbse: the BSE kernel (default tda: walks run on the Tamm-Dancoff '
                        'BSE, whose triplets have no (A - B) instability; full is a comparison)')
    p.add_argument('--basis', default='cc-pvdz')
    p.add_argument('--auxbasis', default=None, help='adc2: default cc-pvdz-ri')
    p.add_argument('--jk-auxbasis', default='cc-pvdz-jkfit')
    p.add_argument('--solvent', default=None, help="a name of pyscf's SMD table")
    p.add_argument('--frozen', type=int, default=None,
                   help='adc2: default the core shells')
    p.add_argument('--nroots', type=int, default=None, help='adc2: default 3')
    p.add_argument('--maxiter', type=int, default=100)
    p.add_argument('--states', nargs='+', default=['singlet', 'triplet'])
    p.add_argument('--follow', choices=('index', 'overlap'), default='index',
                   help='root picking (index: adiabatic, the lowest root of each '
                        'spin; overlap: follow one state)')
    p.add_argument('--media', nargs='+', choices=('gas', 'solvent'),
                   default=['gas', 'solvent'],
                   help='which of gas and --solvent to run (one record file each '
                        'when they run in separate processes)')
    p.add_argument('--saddle-check', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--refreeze', action=argparse.BooleanOptionalAction, default=None,
                   help='gwbse: refreeze each S1 / T1 minimum (default on; the outer '
                        'loop of properties.optimize.relax); ADC(2) and S0 freeze nothing')
    p.add_argument('--tol-q', type=float, default=1e-9,
                   help='loop: largest charge change (e) at convergence')
    p.add_argument('--tol-energy', type=float, default=1e-11,
                   help='loop: change of G (Ha) at convergence')
    p.add_argument('--out', required=True)
    return p.parse_args()


RESUME_KEY = ('xyz_sha256', 'basis', 'auxbasis', 'jk_auxbasis', 'solvent', 'frozen',
              'nroots', 'follow', 'saddle_check', 'tol_q', 'tol_energy', 'code',
              'state_method', 'xc', 'isdf_grid', 'bse_solver', 'bse_kernel',
              'refreeze')


def _settings(args):
    """The run's settings that every number of the record depends on; the
    geometry by the xyz file's content (xyz_sha256), its path for the reader."""
    with open(args.xyz, 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return dict(xyz=os.path.abspath(args.xyz), xyz_sha256=digest, basis=args.basis,
                auxbasis=args.auxbasis, jk_auxbasis=args.jk_auxbasis,
                solvent=args.solvent, frozen=args.frozen, nroots=args.nroots,
                follow=args.follow, saddle_check=args.saddle_check,
                tol_q=args.tol_q, tol_energy=args.tol_energy, code=code_id(ROOT),
                state_method=args.method, xc=args.xc, isdf_grid=args.isdf_grid,
                bse_kernel=args.bse_kernel,
                bse_solver=args.bse_solver, refreeze=args.refreeze)


def _start_record(args):
    """The record: the file's content if present (refused when made with
    other settings, RESUME_KEY), with the run's settings."""
    rec, settings = {}, _settings(args)
    if os.path.isfile(args.out):
        with open(args.out) as fh:
            rec = json.load(fh)
        differ = {k: (rec.get(k, '<absent>'), settings[k]) for k in RESUME_KEY
                  if rec.get(k, '<absent>') != settings[k]}
        if differ:
            raise SystemExit(f'{args.out} was made with other settings (record, this '
                             f'run): {differ}; write to another --out')
    what = ('ADC(2) (folded, DF) on DF-MP2' if args.method == 'adc2' else
            f'G0W0-BSE (scheme B, ISDF {args.isdf_grid}) on {args.xc}')
    rec.update(settings, host=platform.node(), method=f'{what}, '
               'equilibrium/non-equilibrium state-specific PCM with relaxed '
               'densities (src.Base.solvation_loop)')
    return rec


def _save(rec, out):
    """Write the record to out atomically."""
    with open(out + '.tmp', 'w') as fh:
        json.dump(rec, fh, indent=1)
    os.replace(out + '.tmp', out)


def _make_scf(m, jk_auxbasis, xc='hf'):
    """The tightly converged density-fitted RHF (or RKS of xc) of m."""
    mf = (scf.RHF(m) if xc == 'hf' else dft.RKS(m, xc=xc)).density_fit(
        auxbasis=jk_auxbasis)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-12, 1e-10, 200, 0
    return mf


def _check_args(args, mol0):
    """Refuse the other method's options; fill this method's defaults."""
    other = GWBSE_ONLY if args.method == 'adc2' else ADC2_ONLY
    given = [name for name in other if getattr(args, name) is not None]
    if given:
        raise SystemExit(f'--method {args.method} takes no '
                         + ', '.join('--' + g.replace('_', '-') for g in given))
    if args.method == 'adc2':
        for name, value in ADC2_DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if args.frozen is None:
            args.frozen = frozen_core(mol0)
        return
    if args.isdf_grid is None:
        raise SystemExit('--method gwbse needs --isdf-grid (no default grid)')
    if args.xc is None:
        args.xc = 'hf'
    if args.bse_kernel is None:
        args.bse_kernel = 'tda'
    if args.refreeze is None:
        args.refreeze = True


def gwbse_numerics(args, mol0):
    """The one Numerics of a GW-BSE run: the chain's grid (refused here when
    the radii table lacks an element) and solver, the loop's tolerances."""
    bse = {'track': 'overlap' if args.follow == 'overlap' else None}
    if args.bse_solver is not None:
        bse['solver'] = args.bse_solver
    counts = resolve_isdf_grid(args.isdf_grid, args.basis,
                               elements=tuple(sorted(set(mol0.elements))))[0]
    return Numerics(isdf={'counts': counts}, bse=bse,
                    solvent={'loop_tol_q': args.tol_q, 'loop_tol_energy': args.tol_energy})


def _gwbse_states(args, make_scf, numerics):
    """make_state(kind) of GW-BSE: each call is one surface, its first model
    built at the first geometry and moved to every later one (module
    docstring)."""
    def make_state(kind):
        first = []

        def build(m, field):
            if kind == 'S0':
                return GWBSEState(m, make_scf, numerics=numerics)
            tda = args.bse_kernel == 'tda'
            lr = field.transition_environment() if kind == 'singlet' else None
            if first:
                return first[0].at(m, make_scf, lr)
            first.append(GWBSEState(m, make_scf, kind, 0, bse_tda=tda,
                                    lr_environment=lr, numerics=numerics))
            return first[0]
        return build
    return make_state


def _factories(args, medium, loop_kw):
    """(make_field, make_state) for one medium; make_state(kind) builds a state."""
    if medium == 'gas':
        make_field = NoMedium
    else:
        make_field = lambda m: ContinuumField.from_environment(
            SolventScreening(m, solvent=medium))

    def make_scf(m):
        return _make_scf(m, args.jk_auxbasis, args.xc or 'hf')

    if args.method == 'gwbse':
        return make_field, _gwbse_states(args, make_scf, loop_kw['numerics'])

    def make_state(kind):
        def build(m, field):
            mp2 = MP2State(HFState(m, make_scf), args.frozen, args.auxbasis)
            if kind == 'S0':
                return mp2
            lr = field.transition_environment() if kind == 'singlet' else None
            return ADC2State(mp2, kind, 0, args.nroots, lr, follow=args.follow)
        return build
    return make_field, make_state


def _geom(m):
    """The atoms of m as [symbol, x, y, z] in Angstrom."""
    return [[m.atom_symbol(i), *map(float, m.atom_coord(i, unit='Angstrom'))]
            for i in range(m.natm)]


def _mol_of(g, basis):
    """The molecule of a geometry list from _geom."""
    return gto.M(atom=[(a[0], tuple(a[1:])) for a in g], basis=basis, verbose=0)


def _refreezes(args, kind):
    """1 when a walk of `kind` runs the refreeze loop (module docstring): a
    GW-BSE excited state's surface freezes its chain's conventions; ADC(2)
    and the GW-BSE reference freeze nothing."""
    return int(bool(args.refreeze) and args.method == 'gwbse' and kind != 'S0')


def _relax(surf, start, maxiter, nudge, refreeze):
    """relax(surf, start) with or without the refreeze loop: (mol, info). A
    drift above REFREEZE_TOL_MEV after the last pass is warned about by
    properties.optimize (RefreezeNotConverged) and the walk's minimum kept,
    its record's refreeze 'not converged' (module docstring)."""
    return relax(surf, start, maxiter=maxiter, nudge=nudge, refreeze=refreeze)


def _refreeze_record(info, refreeze):
    """The refreeze entries of a walk's record (module docstring)."""
    if not refreeze:
        return dict(refreeze='nothing frozen' if refreeze is None else
                    info.get('refreeze'))
    passes = [dict(p, shift=float(p['shift'])) for p in info.get('refreeze_passes') or []]
    denergy = info.get('refreeze_denergy')
    return dict(refreeze=info.get('refreeze'),
                refreeze_drift_meV=info.get('refreeze_drift_meV'),
                refreeze_shift_bohr=info.get('refreeze_shift'),
                refreeze_converged=info.get('refreeze_converged'),
                refreeze_denergy_meV=(None if denergy is None
                                      else denergy * HARTREE_TO_MEV),
                refreeze_passes=passes)


def _refreeze_flags(out):
    """The refreeze's flag of a walk's record (module docstring)."""
    return ['refreeze_not_converged'] if out['refreeze'] == 'not converged' else []


def _minimum(args, surf, start, nudge, kind):
    """(mol, the record) of a walk on surf from start, refrozen where its
    surface freezes conventions (`_refreezes`) and saddle-checked when it
    converged; `flags` the names flagged at its last geometry (module
    docstring), with 'walk_not_converged' for a walk that stopped short and
    'refreeze_not_converged'. The saddle check, its escape walk (refrozen as
    the first) and G are taken on the surface the last walk ran on."""
    t = time.time()
    refreeze = _refreezes(args, kind)
    asked = None if (args.method == 'adc2' or kind == 'S0') else refreeze
    m, info = _relax(surf, start, args.maxiter, nudge, refreeze)
    final = info.get('final_surface', surf)
    out = dict(converged=bool(info.get('converged')), cycles=info.get('cycles'),
               opt_grad_max=info.get('opt_grad_max'), status=info.get('status'),
               **_refreeze_record(info, asked))
    if args.saddle_check and out['converged']:
        def walk(s, at):
            return _relax(s, at, args.maxiter, 0, refreeze)
        hinfo = {}
        hess = surface_hessian(final, m, info=hinfo)
        m, again, _, chk = escape_saddle(final, m, hess, walk,
                                         lambda s, at: surface_hessian(s, at))
        out['saddle_check'] = dict(
            status=chk['status'],
            lowest_curvature=chk['first']['lowest_curvature'],
            negative_modes=chk['first']['negative_modes'],
            tolerance=chk['first']['tolerance'],
            hessian_asymmetry_rel=hinfo.get('asymmetry_rel'),
            energy_drop=chk.get('energy_drop'),
            second_lowest_curvature=(chk['second'] or {}).get('lowest_curvature'))
        if again is not None:
            out['first_walk'] = dict(out)
            out.update(converged=bool(again.get('converged')),
                       cycles=again.get('cycles'),
                       opt_grad_max=again.get('opt_grad_max'),
                       **_refreeze_record(again, asked))
            final = again.get('final_surface', final)
    out.update(geometry=_geom(m), G=final.total_energy(m), wall_s=time.time() - t)
    # a refrozen surface shares the walk's flag log (SolvatedSurface.refreeze)
    out['flags'] = sorted(set(final.flags_at(m) + _refreeze_flags(out)
                              + ([] if out['converged'] else ['walk_not_converged'])))
    out['flagged_evaluations'] = len(final.flags)
    return m, out


def _store_minimum(rec, out, R, key, entry, what):
    """Store a walk's entry under R[key]; one that did not converge is
    warned about and carries 'walk_not_converged' (module docstring)."""
    R[key] = entry
    R.pop(key + '_failed', None)
    _save(rec, out)
    if not entry.get('converged'):
        why = f'; {entry["status"]}' if entry.get('status') else ''
        warnings.warn(f'{what}: the walk did not converge ({entry.get("cycles")} '
                      f'cycles, opt_grad_max {entry.get("opt_grad_max")}{why}); '
                      'the run carries on from its last geometry, flagged '
                      'walk_not_converged', RuntimeWarning, stacklevel=2)


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


def _vertical(make_field, make_state, kind, m, loop_kw):
    """Vertical S0 -> kind energies of m in the slow charges of S0 (and the reverse)."""
    f = make_field(m)
    return sl.vertical_energies(lambda: make_state('S0')(m, f),
                                lambda: make_state(kind)(m, f), f, **loop_kw)


def _ground_state(args, rec, R, medium, mol0, make_field, make_state, loop_kw):
    """S0 minimum of one medium into R, unless the record holds it."""
    if 'S0' not in R:
        surf = SolvatedSurface(mol0, make_field, make_state('S0'),
                               label=f'S0 {medium}', **loop_kw)
        _store_minimum(rec, args.out, R, 'S0', _minimum(args, surf, mol0, 0, 'S0')[1],
                       f'S0 minimum ({medium})')


def _absorption(args, rec, R, X, kind, mS0, make_field, make_state, loop_kw):
    """Vertical absorption at the S0 minimum into X, unless the record holds it."""
    if 'absorption_eV' not in X:
        t = time.time()
        r = _vertical(make_field, make_state, kind, mS0, loop_kw)
        X.update(absorption_eV=r['absorption'] * HARTREE_TO_EV,
                 absorption_G=r['G'], absorption_wall_s=time.time() - t,
                 absorption_flags=_flag_names(R['S0'].get('flags'), r['flags']))
        _save(rec, args.out)


def _excited_minimum(args, rec, X, key, kind, medium, mS0, make_field, make_state,
                     loop_kw):
    """Minimum of the state on its surface from the S0 minimum, unless recorded."""
    if 'minimum' not in X:
        surf = SolvatedSurface(mS0, make_field, make_state(kind),
                               label=f'{key} {medium}', **loop_kw)
        _store_minimum(rec, args.out, X, 'minimum',
                       _minimum(args, surf, mS0, START_NUDGE_BOHR, kind)[1],
                       f'{key} minimum ({medium})')


def _emission(args, rec, R, X, kind, make_field, make_state, loop_kw):
    """Vertical emission at the X minimum and the adiabatic energy, unless recorded."""
    if 'emission_eV' not in X:
        t = time.time()
        mX = _mol_of(X['minimum']['geometry'], args.basis)
        r = _vertical(make_field, make_state, kind, mX, loop_kw)
        X.update(emission_eV=r['emission'] * HARTREE_TO_EV,
                 emission_G=r['G'], emission_wall_s=time.time() - t,
                 emission_flags=_flag_names(X['minimum'].get('flags'), r['flags']))
        X['adiabatic_eV'] = (X['minimum']['G'] - R['S0']['G']) * HARTREE_TO_EV
        X['adiabatic_flags'] = _flag_names(X['minimum'].get('flags'),
                                           R['S0'].get('flags'))
        gap = X['minimum']['G'] - r['G']['eq_state']
        refrozen = X['minimum'].get('refreeze') in ('measured', 'not converged')
        bar = (REFREEZE_TOL_MEV / HARTREE_TO_MEV if refrozen
               else WALK_VERTICAL_G_FLAG_FACTOR * args.tol_energy)
        X['walk_vs_vertical_G_Ha'], X['walk_vs_vertical_flag'] = gap, abs(gap) > bar
        if abs(gap) > bar:
            warnings.warn(f'{kind} at its minimum: G_eq on the walk\'s surface and on '
                          f'the vertical\'s differ by {gap:.2e} Ha (> {bar:.0e}): the two '
                          'surfaces froze their conventions at different geometries',
                          RuntimeWarning, stacklevel=2)
        _save(rec, args.out)


def _run_medium(args, rec, medium, mol0, loop_kw):
    """All pieces of one medium: S0, then absorption, minimum, emission per state."""
    R = rec.setdefault(medium, {})
    _drop_unconverged(R)
    make_field, make_state = _factories(args, medium, loop_kw)
    _ground_state(args, rec, R, medium, mol0, make_field, make_state, loop_kw)
    mS0 = _mol_of(R['S0']['geometry'], args.basis)
    for kind in args.states:
        key = 'S1' if kind == 'singlet' else 'T1'
        X = R.setdefault(key, {})
        _absorption(args, rec, R, X, kind, mS0, make_field, make_state, loop_kw)
        _excited_minimum(args, rec, X, key, kind, medium, mS0, make_field,
                         make_state, loop_kw)
        _emission(args, rec, R, X, kind, make_field, make_state, loop_kw)
    if 'S1' in R and 'T1' in R and 'adiabatic_eV' in R['S1'] and 'adiabatic_eV' in R['T1']:
        R['dEST_adiabatic_eV'] = R['S1']['adiabatic_eV'] - R['T1']['adiabatic_eV']
        R['dEST_adiabatic_flags'] = _flag_names(R['S1'].get('adiabatic_flags'),
                                                R['T1'].get('adiabatic_flags'))
        _save(rec, args.out)


def _print_summary(rec, media):
    """Print the eV entries of the record for the media run."""
    print(json.dumps({m: {k: {kk: vv for kk, vv in v.items() if kk.endswith('eV')}
                          if isinstance(v, dict) else v
                          for k, v in rec[m].items()} for m in media}, indent=1))


def main():
    args = _parse_args()
    mol0 = gto.M(atom=args.xyz, basis=args.basis, verbose=0)
    _check_args(args, mol0)
    if args.method == 'gwbse':
        # one object: every GW-BSE model holds it, the loops are handed the same
        loop_kw = dict(numerics=gwbse_numerics(args, mol0))
    else:
        loop_kw = dict(numerics=Numerics(solvent={'loop_tol_q': args.tol_q,
                                                  'loop_tol_energy': args.tol_energy}))
    rec = _start_record(args)
    media = ((['gas'] if 'gas' in args.media else [])
             + ([args.solvent] if args.solvent and 'solvent' in args.media else []))
    for medium in media:
        _run_medium(args, rec, medium, mol0, loop_kw)
    _print_summary(rec, media)
    return 0


if __name__ == '__main__':
    sys.exit(main())
