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

The factors are the AO-product fit, which keeps the diffuse AO products of an
augmented basis, on points pivoted from the parent grid of the
--grid-accuracy level (required: no ISDF grid is a default), --pivot-points
of them (omitted, the default count), fitted by rows over the ranks; the
Duchemin-Blase set on the tabulated grid is --isdf-test-set duchemin_blase.
On augmented bases the mean field is best density-fitted (--scf df, J and K
on <basis>-jkfit):

    srun --mpi=pmi2 -n 8 python -m src.photophysics --xyz emitter.xyz \\
        --basis aug-cc-pvdz --environment toluene ... --tda --scf df \\
        --grid-accuracy G3 --pivot-points 9000 \\
        --scf-grad-tol 1e-9 --out tadf --distribute

A basis can differ by element or by atom: --basis names the set of every
atom, --basis-for EL=NAME another for an element and atoms=I-J:NAME another
for atoms counted from 1 (repeatable; an atom range wins over its element).
Each atom's -jkfit and -ri sets and its grid follow its own basis, and a
--grid-accuracy level is used where every named basis gives it the same
counts (aug-cc-pVDZ and cc-pVDZ do at G1-G3):

    python -m src.photophysics --xyz emitter.xyz --basis aug-cc-pvdz \\
        --basis-for H=cc-pvdz --environment gas ... --scf df --grid-accuracy G3
    python -m src.photophysics --xyz film.xyz --basis cc-pvdz \\
        --basis-for atoms=1-41:aug-cc-pvdz ... --scf df --grid-accuracy G3

In a continuum the run names how its states are solvated (--solvent-scheme,
required there, no default): 'scheme-a', the solvated BSE of Duchemin et
al.; 'scheme-b-eq' or 'scheme-b-noneq' (each also '-bare', without the
singlets' linear response), every state self-consistent with its
own relaxed density in the state-specific loop, the excited minima's
energies in their own equilibrium or in S0's slow charges (`stages` module
docstring), whose loops converge to --loop-tol-q / --loop-tol-energy. A
scheme-B run of the AO-product route:

    srun --mpi=pmi2 -n 8 python -m src.photophysics --xyz emitter.xyz \\
        --basis aug-cc-pvdz --environment toluene --solvent-scheme scheme-b-eq \\
        --relaxed-solvation equilibrium --df-check 0 --no-far-start ... \\
        --tda --qp-states frontier --scf df --grid-accuracy G3 \\
        --isdf-test-set ao_products --pivot-points 5000 \\
        --scf-conv-tol 1e-10 --scf-grad-tol 1e-9 \\
        --hessian external --hessian-file s0.hess --out tadf --distribute

The vertical energies alone may take the analytic continuation instead of
the pole model (--residues pade): the Pade continuation of the space-time
self-energy, with no residues and no adjoint, on a set declared by kind and
without the quasiparticle Hamiltonian, in the gas phase or scheme A, on the
stages s0, hess and vert. Its record holds the energies, oscillator strengths
and spin-orbit table and no derivative coupling; every walk, the rates and
scheme B are refused. With --r0 input R0 is the input geometry as it is (a film
frame), recorded by the s0 stage with no SCF:

    srun --mpi=pmi2 -n 32 python -m src.photophysics --xyz film.xyz \\
        --basis cc-pvdz --basis-for atoms=1-41:aug-cc-pvdz --environment gas \\
        ... --residues pade --qp-states admitted --scf isdf --scf-grid G3 \\
        --scf-pivot-points 60000 --grid-accuracy G3 --pivot-points 60000 \\
        --r0 input --stages vert --out tadf --distribute

The S0 Hessian may come from elsewhere (--hessian external): an ORCA .hess
file at R0, the geometry the s0 stage writes to s0.xyz; another geometry is
refused.

The thread environment (--threads) is set before this module is imported
(`launch`, from `__main__` and the launcher): numpy sizes its pools at
import.
"""
import argparse
import pathlib

from src.Base.numerics.constants import (HESSIAN_SCF_GRAD_TOL,
                                         ISDF_TEST_SETS, LOOP_FORCE_FRACTION,
                                         LOOP_SOLVER_TOL_FRACTION,
                                         PHOTOPHYSICS_LOOP_TOL_ENERGY,
                                         PHOTOPHYSICS_LOOP_TOL_Q,
                                         SCF_OPTIMIZATION_CONV_TOL,
                                         SCF_OPTIMIZATION_GRAD_TOL, TADF_ALLOCATION_MB,
                                         TADF_N_STATES,
                                         TADF_NROOTS, TADF_OUTSIDE_SCISSOR,
                                         TADF_QP_HALF_WIDTH, TADF_REFERENCE,
                                         TADF_THREADS, UISC_BLOCKS, UISC_DQ,
                                         UISC_LAMBDA_SAMANTA_EV, UISC_N_MODES)
from src.Base.basis.basis_map import molecule_basis, parse_basis_for
from src.Base.declaration import QPStates
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm
from src.environments.solvent_vocabulary import GAS, SCHEME_B, SOLVENT_SCHEMES
from src.photophysics.declaration import (ENGINES, GROUNDS,
                                          HESSIAN_ROUTES, RELAXED_SOLVATION,
                                          PROPERTY_SETS, R0_ROUTES, RESIDUES,
                                          SCF_FINISH,
                                          SCF_ROUTES,
                                          Declaration,
                                          ExternalHessian, FarStart, Policies,
                                          System, TadfModel, TdaMinimum,
                                          UiscModel)
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
    ap.add_argument('--basis', required=True,
                    help='the orbital basis of every atom --basis-for does '
                         'not name')
    ap.add_argument('--basis-for', action='append', default=None,
                    metavar='EL=NAME',
                    help='another basis for one element (H=cc-pvdz) or for '
                         'atoms counted from 1 (atoms=1-41:aug-cc-pvdz, over '
                         'the element\'s), repeatable; each atom\'s auxiliary '
                         'sets (-jkfit, -ri) follow its own basis')
    ap.add_argument('--environment', required=True,
                    help="'gas', a solvent of pyscf's SMD table, "
                         "'eps=EPS_S,n2=N2[,label=NAME]' or 'N2:EPS_STATIC', "
                         "each optionally ',cavity=NAME' (src.environments.medium)")
    ap.add_argument('--solvent-scheme', default=None,
                    choices=tuple(s for s in SOLVENT_SCHEMES if s != GAS),
                    help='in a continuum, required: how every state stands '
                         "in it ('scheme-a', 'scheme-b-eq' or "
                         "'scheme-b-noneq', each also '-bare'); the gas phase "
                         "takes none")


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
    pol.add_argument('--tda-t1-minimum', metavar='XYZ', default=None,
                     help='scheme-b-* on the full BSE (no --tda), only there: '
                          'the T1 minimum of the Tamm-Dancoff run '
                          '(its t1.xyz), where the T1 single point is taken '
                          'when the full kernel\'s triplet walk stops at the '
                          'instability')
    pol.add_argument('--refreeze', type=int, choices=(0, 1), required=True)
    pol.add_argument('--saddle-check', type=int, choices=(0, 1),
                     required=True)
    pol.add_argument('--df-check', type=int, choices=(0, 1), required=True)
    pol.add_argument('--hessian', choices=HESSIAN_ROUTES, required=True)
    pol.add_argument('--hessian-file', metavar='HESS', default=None,
                     help='with --hessian external: the ORCA .hess file of '
                          'the S0 Hessian at R0')
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
    ap.add_argument('--r0', default='walk', choices=R0_ROUTES,
                    help="where R0 comes from: 'walk' relaxes the ground "
                         "state (the s0 stage), 'input' takes the input "
                         'geometry as it is (a film or trajectory frame); '
                         'the s0 stage then only records it')
    ap.add_argument('--properties', default='all', choices=PROPERTY_SETS,
                    help="what the vertical stage computes besides the "
                         "energies: 'all' (every property a rate reads) or "
                         "'energies' (the energies, oscillator strengths and "
                         'transition dipoles; no spin-orbit table, no '
                         'coupling; only the stages s0, hess and vert)')


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
    ap.add_argument('--residues', default='sop', choices=RESIDUES,
                    help='how Sigma reaches the real axis: the pole model, '
                         'the explicit residues, or the analytic (Pade) '
                         'continuation, which has no adjoint and serves the '
                         'stages s0, hess and vert alone')
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
                     help='the ISDF-K mean field\'s grid level; required '
                          'with --scf isdf (no ISDF grid is a default)')
    num.add_argument('--grid-accuracy', required=True,
                     help='required: the factors\' grid level; no ISDF '
                          'grid is a default')
    num.add_argument('--isdf-test-set', default=None, choices=ISDF_TEST_SETS,
                     help='the products the factors\' fit is trained on; '
                          'omitted is ao_products; duchemin_blase, the pair '
                          'set on the tabulated grid, by name')
    num.add_argument('--pivot-points', type=int, default=None,
                     help='the AO-product fit\'s points pivoted from the '
                          '--grid-accuracy level\'s parent grid; omitted is '
                          'ao_product_fit.pivot_count')
    num.add_argument('--scf-isdf-test-set', default=None,
                     choices=ISDF_TEST_SETS,
                     help='the ISDF-K mean field\'s own fit, apart from the '
                          'factors\': omitted is the --scf-grid level\'s '
                          'tabulated fit')
    num.add_argument('--scf-pivot-points', type=int, default=None,
                     help='the ISDF-K mean field\'s AO-product points pivoted '
                          'from the --scf-grid level\'s parent (alone, it '
                          'means --scf-isdf-test-set ao_products); omitted, '
                          'the tabulated fit')
    num.add_argument('--hessian-grid', default=None,
                     help='the numerical Hessian\'s grid level; required '
                          'with --hessian numerical (no ISDF grid is a '
                          'default)')
    num.add_argument('--nroots', type=int, default=TADF_NROOTS)
    num.add_argument('--track', default='overlap',
                     choices=('overlap', 'index'))
    num.add_argument('--qp-hamiltonian', action=argparse.BooleanOptionalAction,
                     default=None,
                     help='solve the quasiparticle set\'s near-degenerate '
                          'groups as one Hamiltonian; omitted is on under '
                          'a scheme-b-* solvent scheme (the loop moves the '
                          'charges under which a near-degenerate explicit '
                          'pair mixes) and off otherwise')
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
    num.add_argument('--loop-tol-q', type=float, default=None,
                     help='scheme-b-*: the loop\'s largest charge change (e) at '
                          'convergence; omitted is PHOTOPHYSICS_LOOP_TOL_Q')
    num.add_argument('--loop-tol-energy', type=float, default=None,
                     help='scheme-b-*: the loop\'s change of G (Ha) at '
                          'convergence; omitted is the larger of '
                          'PHOTOPHYSICS_LOOP_TOL_ENERGY and the SCF\'s '
                          'conv_tol (G settles no further)')
    num.add_argument('--loop-tolerance', default=None,
                     choices=('fixed', 'gradient-tied'),
                     help='scheme-b-*: every walk loop to the tolerances above '
                          '(fixed, the default), or stopped once its estimated '
                          'force error is LOOP_FORCE_FRACTION of the walk\'s '
                          'gradient (the first gradient of each surface and '
                          'the converged one at the tolerances above)')


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
    choices: the factors on --grid-accuracy fitted by rows and sliced (the
    AO-product fit on pivoted points, or the Duchemin-Blase set named by
    --isdf-test-set, `isdf_numerics_of`), the grid adjoint, the ISDF-K mean field
    on --scf-grid converged for a walk, the span cut --span declares
    (`span_cut_of`), and under scheme B the loop's tolerances
    (`loop_numerics_of`) with the BSE roots converged below its charge
    tolerance (`Numerics.solved_for_the_loop`; a --bse-conv-tol above that
    bar is refused)."""
    bse = {'nroots': args.nroots, 'adjoint': 'grid',
           'track': 'overlap' if args.track == 'overlap' else None}
    if args.bse_conv_tol is not None:
        bse['conv_tol'] = args.bse_conv_tol
    if args.no_probe:
        bse['probe'] = False
    loop = loop_numerics_of(args)
    if loop and args.bse_conv_tol is not None:
        bar = LOOP_SOLVER_TOL_FRACTION * loop['solvent']['loop_tol_q']
        if args.bse_conv_tol > bar:
            raise SystemExit(f'--bse-conv-tol {args.bse_conv_tol:g} is above '
                             f'{bar:g}, LOOP_SOLVER_TOL_FRACTION of the loop\'s charge '
                             'tolerance: the loop could not converge its charges')
    numerics = Numerics(
        isdf=isdf_numerics_of(args),
        layout={'sliced': True},
        scf={'route': args.scf,
             'grid_level': None if args.scf == 'df' else scf_grid_of(args),
             'conv_tol': scf_conv_tol_of(args),
             'conv_tol_grad': (SCF_OPTIMIZATION_GRAD_TOL
                               if args.scf_grad_tol is None
                               else args.scf_grad_tol),
             **({} if args.xc_grid is None else {'xc_grid': args.xc_grid}),
             **scf_fit_of(args)},
        span={'cut': span_cut_of(args)},
        qp={'outside_scissor': args.outside_scissor,
            'hamiltonian': (args.solvent_scheme in SCHEME_B
                            if args.qp_hamiltonian is None
                            else args.qp_hamiltonian),
            'residue_route': args.residues},
        bse=bse, **loop)
    return numerics.solved_for_the_loop('bse') if loop else numerics


def scf_grid_of(args):
    """--scf-grid, the ISDF-K mean field's grid level, refused when omitted:
    no ISDF grid is a default."""
    if args.scf_grid is None:
        raise SystemExit(f'--scf {args.scf} builds the ISDF-K mean field on '
                         f'--scf-grid, which is required: no ISDF grid is a '
                         f'default')
    return args.scf_grid


def loop_numerics_of(args):
    """{'solvent': the loop's tolerances} under a scheme-b-* --solvent-scheme
    (omitted: PHOTOPHYSICS_LOOP_TOL_Q, and the larger of
    PHOTOPHYSICS_LOOP_TOL_ENERGY and the SCF's conv_tol; a walk's loop tied
    to its gradient by --loop-tolerance gradient-tied), else nothing:
    another scheme runs no loop, and a tolerance given to it is refused."""
    given = [o for o in ('loop_tol_q', 'loop_tol_energy', 'loop_tolerance')
             if getattr(args, o) is not None]
    if args.solvent_scheme not in SCHEME_B:
        if given:
            raise SystemExit(', '.join('--' + o.replace('_', '-')
                                       for o in given)
                             + ' set the loop of a scheme-b-* solvent scheme')
        return {}
    return {'solvent': {
        'loop_tol_q': (PHOTOPHYSICS_LOOP_TOL_Q if args.loop_tol_q is None
                       else args.loop_tol_q),
        'loop_tol_energy': (max(PHOTOPHYSICS_LOOP_TOL_ENERGY, scf_conv_tol_of(args))
                            if args.loop_tol_energy is None
                            else args.loop_tol_energy),
        'loop_force_fraction': (LOOP_FORCE_FRACTION
                                if args.loop_tolerance == 'gradient-tied'
                                else None)}}


def scf_conv_tol_of(args):
    """The SCF's conv_tol: --scf-conv-tol, else SCF_OPTIMIZATION_CONV_TOL."""
    return (SCF_OPTIMIZATION_CONV_TOL if args.scf_conv_tol is None
            else args.scf_conv_tol)


def solvent_scheme_of(args, environment):
    """The solvent scheme: 'gas' in the gas phase (where --solvent-scheme is
    refused), --solvent-scheme in a continuum (where it is required)."""
    if environment == GAS:
        if args.solvent_scheme is not None:
            raise SystemExit('--solvent-scheme names how states stand in a '
                             'continuum; the gas phase takes none')
        return GAS
    if args.solvent_scheme is None:
        raise SystemExit(f'--environment {environment} needs --solvent-scheme '
                         f'(scheme-a or a scheme-b-* one)')
    return args.solvent_scheme


def scf_fit_of(args):
    """The ISDF-K mean field's own fit, --scf-isdf-test-set and
    --scf-pivot-points, refused beside --scf df; empty where neither is
    given (the --scf-grid level's tabulated fit)."""
    fit = {k: v for k, v in (('test_set', args.scf_isdf_test_set),
                             ('pivot_points', args.scf_pivot_points))
           if v is not None}
    if fit and args.scf != 'isdf':
        raise SystemExit('--scf-isdf-test-set and --scf-pivot-points are the '
                         'ISDF-K mean field\'s fit; --scf df has none')
    return fit


def isdf_numerics_of(args):
    """The factors' fit, by rows over the ranks, on --grid-accuracy: the
    AO-product fit on --pivot-points points pivoted from that level's parent
    grid (omitted: `ao_product_fit.pivot_count`), or with --isdf-test-set
    duchemin_blase the Duchemin-Blase set on the level's grid as it is. A
    pivot count beside the Duchemin-Blase set is refused."""
    isdf = {'level': args.grid_accuracy, 'fit': 'rows'}
    if args.isdf_test_set == 'duchemin_blase':
        if args.pivot_points is not None:
            raise SystemExit('--pivot-points pivots the points of the '
                             'AO-product fit, not --isdf-test-set '
                             'duchemin_blase')
        return dict(isdf, test_set='duchemin_blase')
    if args.isdf_test_set is not None:
        isdf['test_set'] = args.isdf_test_set
    if args.pivot_points is not None:
        isdf['pivot_points'] = args.pivot_points
    return isdf


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
        return System(args.system or path.stem,
                      *basis_of(args, xyz_atoms(path.read_text())))
    if args.system is None:
        raise SystemExit('--atom carries no name: pass --system')
    return System(args.system, *basis_of(args, args.atom))


def basis_of(args, atom):
    """(atom, basis): `atom` and --basis, or with --basis-for the
    molecule's per-element or per-atom basis and its atoms labelled for it
    (`molecule_basis`: the plain name where every atom ends up on one
    set)."""
    if not args.basis_for:
        return atom, args.basis
    try:
        return molecule_basis(atom, args.basis, *parse_basis_for(args.basis_for))
    except ValueError as err:
        raise SystemExit(f'--basis-for: {err}')


def hessian_numerics_of(args, numerics):
    """The numerical Hessian's numerics: the run's own where they already
    are the Hessian's (its grid, SCF gradient converged to the run's threshold
    or HESSIAN_SCF_GRAD_TOL, whichever is tighter), else derived from them;
    None for the analytic route."""
    if args.hessian != 'numerical':
        return None
    if args.hessian_grid is None:
        raise SystemExit('--hessian numerical needs --hessian-grid, the '
                         'displaced forces\' grid level: no ISDF grid is a '
                         'default')
    scf = {'grid_level': args.hessian_grid,
           'conv_tol_grad': min(numerics.scf.conv_tol_grad,
                                HESSIAN_SCF_GRAD_TOL)}
    if all(getattr(numerics.scf, k) == v for k, v in scf.items()):
        return numerics
    return numerics.derive('the finite-difference S0 Hessian: displaced '
                           'forces converged for a derivative, on the '
                           'Hessian grid', scf=scf)


def policies_of(args):
    """The `Policies` of the command line's required options."""
    far = (None if args.far_start is None else
           FarStart(xyz_atoms(pathlib.Path(args.far_start).read_text())))
    if (args.hessian == 'external') != (args.hessian_file is not None):
        raise SystemExit('--hessian external reads --hessian-file, and only '
                         'it does')
    return Policies(
        nudge_bohr=args.nudge, far_start=far, refreeze=bool(args.refreeze),
        saddle_check=bool(args.saddle_check), df_check=bool(args.df_check),
        hessian=args.hessian, relaxed_solvation=args.relaxed_solvation,
        scf_warm_start=not args.no_scf_warm_start, engine=args.engine,
        scf_finish=args.scf_finish,
        external_hessian=(None if args.hessian_file is None
                          else ExternalHessian.of(args.hessian_file)),
        tda_t1_minimum=(None if args.tda_t1_minimum is None else TdaMinimum(
            xyz_atoms(pathlib.Path(args.tda_t1_minimum).read_text()))),
        r0=args.r0, properties=args.properties)


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
    environment = (GAS if args.environment.lower() in ('', 'none', GAS)
                   else args.environment)
    return Declaration(
        system=system_of(args), ground=args.ground, reference=args.reference,
        environment=environment,
        solvent_scheme=solvent_scheme_of(args, environment),
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
