"""The command line of a TADF run: arguments into one `Declaration`, then
`run_tadf`. Nothing else happens here.

    python -m src.photophysics --system formaldehyde \\
        --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \\
        --basis cc-pvdz --environment gas --temperature 300 --lambda-env 0.05 \\
        --nudge 0.01 --refreeze 1 --saddle-check 0 --df-check 0 \\
        --hessian analytic-df --relaxed-solvation equilibrium --span full \\
        --manifold-gaps stacked --out tadf

    srun --mpi=pmi2 -n 8 python -m src.photophysics --xyz dmabn.xyz ... --distribute

The physics policies (--temperature, --lambda-env, --nudge, --far-start or
--no-far-start, --refreeze, --saddle-check, --df-check, --hessian,
--relaxed-solvation, --span, --manifold-gaps) have NO default: the run is refused until each is
given. The numerical settings default to the production chain's
(`production_numerics`), and the help names each default.

The thread environment (--threads) is set before this module is imported
(`launch`, from `__main__` and the launcher): numpy sizes its pools at
import.
"""
import argparse
import pathlib

from src.Base.numerics.constants import (ISDF_HESSIAN_MIN_GRID,
                                         SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_OPTIMIZATION_CONV_TOL,
                                         SCF_OPTIMIZATION_GRAD_TOL, TADF_ALLOCATION_MB,
                                         TADF_GRID_ACCURACY, TADF_N_STATES,
                                         TADF_NROOTS, TADF_OUTSIDE_SCISSOR,
                                         TADF_QP_HALF_WIDTH, TADF_REFERENCE,
                                         TADF_THREADS, UISC_BLOCKS, UISC_DQ,
                                         UISC_LAMBDA_SAMANTA_EV, UISC_N_MODES)
from src.Base.declaration import QPStates
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm
from src.photophysics.declaration import (ENGINES, GAS, GROUNDS,
                                          HESSIAN_ROUTES, RELAXED_SOLVATION,
                                          RESIDUES, SCF_FINISH, SCF_ROUTES,
                                          Declaration, FarStart, Policies,
                                          System, TadfModel, UiscModel)
from src.photophysics.stages import run_tadf
from src.photophysics.rates import MANIFOLD_GAP_RULES
from src.photophysics.store import Store

#: The --span word for the full basis; any other value is a declared cut.
SPAN_FULL = 'full'


# ------------------------------------------------------------- arguments
def parser():
    """The command line's arguments."""
    ap = argparse.ArgumentParser(
        prog='python -m src.photophysics', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_system(ap)
    add_policies(ap)
    add_physics(ap)
    add_numerics(ap)
    add_launch(ap)
    return ap


def add_system(ap):
    """The molecule, its basis and its environment."""
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument('--atom', help='pyscf atom string, Angstrom')
    where.add_argument('--xyz', metavar='PATH', help='an xyz file, Angstrom')
    ap.add_argument('--system', default=None,
                    help='the run\'s name (its store directory); omitted is '
                         'the xyz file\'s stem')
    ap.add_argument('--basis', required=True)
    ap.add_argument('--environment', required=True,
                    help="'gas', a solvent of pyscf's SMD table, or "
                         "'EPS:EPS_STATIC'")


def add_policies(ap):
    """The physics policies: required, no default."""
    pol = ap.add_argument_group('physics policies (required)')
    pol.add_argument('--temperature', type=float, required=True, help='K')
    pol.add_argument('--lambda-env', type=float, required=True,
                     help='eV of reorganization of the surroundings')
    pol.add_argument('--nudge', type=float, required=True,
                     help='Bohr, the seeded symmetry-breaking start of every '
                          'excited walk; 0 hands each walk the evaluation at '
                          'its start (START_NUDGE_BOHR is the production '
                          'value)')
    far = pol.add_mutually_exclusive_group(required=True)
    far.add_argument('--far-start', metavar='XYZ',
                     help='start the S1 walk at this geometry, on a chain '
                          'built there')
    far.add_argument('--no-far-start', action='store_true',
                     help='the S1 walk starts at R0')
    pol.add_argument('--refreeze', type=int, choices=(0, 1), required=True)
    pol.add_argument('--saddle-check', type=int, choices=(0, 1),
                     required=True)
    pol.add_argument('--df-check', type=int, choices=(0, 1), required=True)
    pol.add_argument('--hessian', choices=HESSIAN_ROUTES, required=True)
    pol.add_argument('--relaxed-solvation', required=True,
                     choices=RELAXED_SOLVATION)
    pol.add_argument('--span', required=True, type=span_option,
                     help="'full': every SCF in the full basis, and a "
                          'molecule whose orbital densities the factors miss '
                          'is refused (run it on the DF route); a number '
                          'between 0 and 1: every SCF in the canonical span '
                          'above that overlap eigenvalue (a declared cut, '
                          'never chosen by refusal)')
    pol.add_argument('--manifold-gaps', required=True,
                     choices=MANIFOLD_GAP_RULES,
                     help="the rate table's state manifold: 'stacked', each "
                          "state at its spin's adiabatic energy plus its "
                          "vertical splitting; 'adiabatic', each state at its "
                          'own minimum (S1 and T1 here)')


def add_physics(ap):
    """The physics with production values: the functional, the kernel, the
    states, the quasiparticle set and the rate model."""
    ap.add_argument('--reference', default=TADF_REFERENCE,
                    help=f'the Kohn-Sham functional (default {TADF_REFERENCE})')
    ap.add_argument('--tda', action='store_true',
                    help='the Tamm-Dancoff kernel (default full BSE)')
    ap.add_argument('--singlet', type=int, default=1)
    ap.add_argument('--triplet', type=int, default=1)
    ap.add_argument('--n-states', type=int, default=TADF_N_STATES)
    ap.add_argument('--ground', default='dft', choices=GROUNDS,
                    help="E_0: E_KS ('dft', default) or E_HF + E_c^dRPA "
                         "('rpa')")
    ap.add_argument('--qp-states', default='adaptive',
                    choices=('admitted', 'frontier', 'valence', 'all',
                             'adaptive'))
    ap.add_argument('--qp-tol-mev', type=float, default=None,
                    help='the adaptive set\'s budget; omitted is '
                         'ADAPTIVE_QP_TOL_MEV')
    ap.add_argument('--qp-window', type=int, default=TADF_QP_HALF_WIDTH,
                    help='the frontier set\'s half-width')
    ap.add_argument('--residues', default='sop', choices=RESIDUES)
    ap.add_argument('--rate-model', default='tadf', choices=('tadf', 'uisc'))
    ap.add_argument('--uisc-modes', type=int, default=UISC_N_MODES)
    ap.add_argument('--uisc-dq', type=float, default=UISC_DQ)
    ap.add_argument('--uisc-blocks', type=int, nargs=2, default=UISC_BLOCKS,
                    metavar=('S', 'T'))


def add_numerics(ap):
    """The numerics, at the production chain's values."""
    num = ap.add_argument_group('numerics (production defaults)')
    num.add_argument('--scf', default='isdf', choices=SCF_ROUTES)
    num.add_argument('--scf-grid', default=None,
                     help='omitted is ISDF_HESSIAN_MIN_GRID')
    num.add_argument('--grid-accuracy', default=TADF_GRID_ACCURACY,
                     help=f'the factors\' grid level (default '
                          f'{TADF_GRID_ACCURACY})')
    num.add_argument('--hessian-grid', default=None,
                     help='the numerical Hessian\'s grid; omitted is '
                          'ISDF_HESSIAN_MIN_GRID')
    num.add_argument('--nroots', type=int, default=TADF_NROOTS)
    num.add_argument('--track', default='overlap',
                     choices=('overlap', 'index'))
    num.add_argument('--qp-hamiltonian', action='store_true')
    num.add_argument('--outside-scissor', default=TADF_OUTSIDE_SCISSOR)
    num.add_argument('--bse-conv-tol', type=float, default=None,
                     help='omitted is BSE_DAVIDSON_CONV_TOL')
    num.add_argument('--no-probe', action='store_true')
    num.add_argument('--xc-grid', type=int, default=None,
                     help='pyscf\'s XC grid level of every Kohn-Sham mean '
                          'field; omitted is XC_GRID_LEVEL')
    num.add_argument('--scf-conv-tol', type=float, default=None,
                     help='omitted is SCF_OPTIMIZATION_CONV_TOL')
    num.add_argument('--scf-grad-tol', type=float, default=None,
                     help='omitted is SCF_OPTIMIZATION_GRAD_TOL')
    num.add_argument('--no-scf-warm-start', action='store_true')
    num.add_argument('--scf-finish', default='diis', choices=SCF_FINISH)
    num.add_argument('--engine', default='auto', choices=ENGINES)


def add_launch(ap):
    """The launch: stages, ranks, threads, memory and the store."""
    ap.add_argument('--stages', default=None,
                    help='comma-separated; omitted is every stage of the '
                         'rate model (vib under uisc only)')
    ap.add_argument('--distribute', action='store_true')
    ap.add_argument('--threads', type=int, default=TADF_THREADS,
                    help='per rank, set before numpy loads (`launch`)')
    ap.add_argument('--max-memory', type=int, default=None,
                    help=f'MB; omitted is the allocation, else '
                         f'{TADF_ALLOCATION_MB}')
    ap.add_argument('--out', required=True, help='the store\'s root')
    ap.add_argument('--dry-run', action='store_true')


# ----------------------------------------------------------- declaration
def production_numerics(args):
    """The `Numerics` of the production chain with the command line's
    choices: the factors on --grid-accuracy fitted by rows and sliced, the
    grid adjoint, the ISDF-K mean field on --scf-grid converged for a walk,
    the span cut --span declares (`span_cut_of`)."""
    bse = {'nroots': args.nroots, 'adjoint': 'grid',
           'track': 'overlap' if args.track == 'overlap' else None}
    if args.bse_conv_tol is not None:
        bse['conv_tol'] = args.bse_conv_tol
    if args.no_probe:
        bse['probe'] = False
    return Numerics(
        isdf={'level': args.grid_accuracy, 'fit': 'rows'},
        layout={'sliced': True},
        scf={'route': args.scf,
             'grid_level': (None if args.scf == 'df'
                            else args.scf_grid or ISDF_HESSIAN_MIN_GRID),
             'conv_tol': (SCF_OPTIMIZATION_CONV_TOL if args.scf_conv_tol is None
                          else args.scf_conv_tol),
             'conv_tol_grad': (SCF_OPTIMIZATION_GRAD_TOL
                               if args.scf_grad_tol is None
                               else args.scf_grad_tol),
             **({} if args.xc_grid is None else {'xc_grid': args.xc_grid})},
        span={'cut': span_cut_of(args)},
        qp={'outside_scissor': args.outside_scissor,
            'hamiltonian': args.qp_hamiltonian,
            'residue_route': args.residues},
        bse=bse)


def span_option(text):
    """--span's value: SPAN_FULL, or a cut between 0 and 1."""
    if text == SPAN_FULL:
        return text
    try:
        cut = float(text)
    except ValueError:
        cut = None
    if cut is None or not 0 < cut < 1:
        raise argparse.ArgumentTypeError(
            f"{text!r}: '{SPAN_FULL}', or a cut between 0 and 1")
    return text


def span_cut_of(args):
    """The cut --span declares: None for SPAN_FULL, else the cut."""
    return None if args.span == SPAN_FULL else float(args.span)


def system_of(args):
    """The `System` of --xyz (named by its stem unless --system) or --atom
    (named by --system)."""
    if args.xyz is not None:
        path = pathlib.Path(args.xyz)
        return System(args.system or path.stem, xyz_atoms(path.read_text()),
                      args.basis)
    if args.system is None:
        raise SystemExit('--atom carries no name: pass --system')
    return System(args.system, args.atom, args.basis)


def hessian_numerics_of(args, numerics):
    """The numerical Hessian's numerics: the run's own where they already
    are the Hessian's (its grid, SCF gradient converged for a derivative),
    else derived from them; None for the analytic route."""
    if args.hessian != 'numerical':
        return None
    scf = {'grid_level': args.hessian_grid or ISDF_HESSIAN_MIN_GRID,
           'conv_tol_grad': SCF_DIFFERENTIABLE_GRAD_TOL}
    if all(getattr(numerics.scf, k) == v for k, v in scf.items()):
        return numerics
    return numerics.derive('the finite-difference S0 Hessian: displaced '
                           'forces converged for a derivative, on the '
                           'Hessian grid', scf=scf)


def policies_of(args):
    """The `Policies` of the command line's required options."""
    far = (None if args.far_start is None else
           FarStart(xyz_atoms(pathlib.Path(args.far_start).read_text())))
    return Policies(
        nudge_bohr=args.nudge, far_start=far, refreeze=bool(args.refreeze),
        saddle_check=bool(args.saddle_check), df_check=bool(args.df_check),
        hessian=args.hessian, relaxed_solvation=args.relaxed_solvation,
        scf_warm_start=not args.no_scf_warm_start, engine=args.engine,
        scf_finish=args.scf_finish)


def qp_states_of(args):
    """The `QPStates` of --qp-states: an adaptive set's targets are the
    followed singlet and triplet, a frontier set's half-width --qp-window."""
    if args.qp_tol_mev is not None and args.qp_states != 'adaptive':
        raise SystemExit('--qp-tol-mev is the budget of --qp-states adaptive')
    if args.qp_states == 'adaptive':
        return QPStates('adaptive', tol_meV=args.qp_tol_mev,
                        targets=(('singlet', args.singlet - 1),
                                 ('triplet', args.triplet - 1)))
    return QPStates(kind=args.qp_states, half_width=args.qp_window)


def rate_model_of(args):
    """`TadfModel()`, or the `UiscModel` of the --uisc-* options."""
    if args.rate_model == 'tadf':
        return TadfModel()
    return UiscModel(n_modes=args.uisc_modes, dq=args.uisc_dq,
                     s_block=args.uisc_blocks[0], t_block=args.uisc_blocks[1],
                     lambda_samanta=UISC_LAMBDA_SAMANTA_EV)


def declaration_of(args):
    """The `Declaration` the parsed command line `args` says."""
    numerics = production_numerics(args)
    return Declaration(
        system=system_of(args), ground=args.ground, reference=args.reference,
        environment=(GAS if args.environment.lower() in ('', 'none', GAS)
                     else args.environment),
        kernel='bse-tda' if args.tda else 'bse', singlet=args.singlet,
        triplet=args.triplet, n_states=args.n_states,
        qp_states=qp_states_of(args), numerics=numerics,
        hessian_numerics=hessian_numerics_of(args, numerics),
        policies=policies_of(args), temperature=args.temperature,
        lambda_env_ev=args.lambda_env, rate_model=rate_model_of(args),
        manifold_gaps=args.manifold_gaps)


def xyz_atoms(text):
    """The atoms of an xyz file's text as a pyscf atom string, Angstrom."""
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    return '; '.join(' '.join(line.split()[:4]) for line in lines[2:2 + n])


# ------------------------------------------------------------------ main
def main(argv=None):
    """Parse `argv` and run the declared stages."""
    args = parser().parse_args(argv)
    declaration = declaration_of(args)
    comm = None
    if args.distribute:
        world, _, size = grid_comm()
        comm = None if size == 1 else world
    max_memory = (args.max_memory if args.max_memory is not None
                  else allocation_max_memory_mb(default=TADF_ALLOCATION_MB))
    stages = (None if args.stages is None else
              tuple(s.strip() for s in args.stages.split(',') if s.strip()))
    with distributed(comm):
        store = Store(args.out, declaration)
        run_tadf(declaration, store, stages=stages, max_memory=max_memory,
                 dry_run=args.dry_run)
    return 0
