"""What a photophysics run is: `Declaration`, one frozen record.

A run of the TADF chain -- the ground-state minimum, its modes, the vertical
spectrum, the relaxed S1 and T1 and the rate table -- is a function of what is
declared here and of nothing else. The declaration holds

    system      the molecule (atoms in Angstrom) and its basis
    physics     E_0 and its functional, the environment and how a state is
                solvated in it (`solvent_scheme`), the BSE kernel,
                the states followed, the quasiparticle set, temperature and
                lambda_env, the rate model the records feed and the gap
                rule of its state manifold
    numerics    the ONE `Numerics` of the run: grids, tolerances, the BSE
                Davidson, the span cut, the walk; derived numerics (the
                finite-difference Hessian's) carry it as their parent
    policies    every choice of PROCEDURE that moves a number: the start
                nudge, the far start, the refreeze, the saddle check, the
                ISDF-vs-DF acceptance check, the Hessian route, how relaxed
                states are solvated, the SCF warm start, the walk engine

Every field is required. A physics policy has no default anywhere in this
package: a declaration that leaves one out is refused at construction, so
two runs cannot differ by a default nobody wrote down. Defaults belong to the
command line (`photophysics.cli`), which says what it fills in.

`identity()` is the sha256 of the whole declaration; `stages.Stamps` holds
what each stage's record is a function of (a stage that reads no policy is
not invalidated by changing it), and the checkpoint store refuses a record
whose stamp is not the one the reading declaration makes.
"""
import hashlib
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Optional

from src.Base.numerics.constants import (RESIDUE_ROUTES_ALL,
                                         STATE_MODEL_SCF_GRAD_TOL)
from src.Base.declaration import KERNELS, QPStates
from src.Base.numerics.numerics import Numerics
from src.Base.utils.record_io import record_hash
from src.environments.medium import parse as parse_medium
from src.photophysics.rates import MANIFOLD_GAP_RULES

#: Where the relaxed states stand in a continuum.
RELAXED_SOLVATION = ('equilibrium', 'nonequilibrium')
#: The S0 Hessian's routes: pyscf's analytic Hessian of a density-fitted twin,
#: central differences of the declared mean field's force, or a Hessian
#: computed elsewhere at R0 and read from its file (`ExternalHessian`).
HESSIAN_ROUTES = ('analytic-df', 'numerical', 'external')
#: How the states stand in the environment, each by name. 'gas': no medium.
#: 'scheme-a': the solvated BSE of Duchemin, Guido, Jacquemin and Blase
#: (Chem. Sci. 9, 4430 (2018)): the ground state in PCM at eps_static, the
#: response screened by the continuum (non-equilibrium at R0), the relaxed
#: states in the split or non-equilibrium response (`relaxed_solvation`).
#: 'scheme-b-eq' and 'scheme-b-noneq': every state self-consistent with its
#: own orbital-relaxed density in the continuum
#: (`src.environments.solvation_loop`, the state-specific loop of
#: `src.gradients.gwbse.state_model`), its singlets' linear response at
#: eps_inf, the walks on G_eq; absorption and emission are the loop's
#: non-equilibrium energies. The two differ in the energy an excited state
#: has at its minimum (`stages` module docstring, THE ENERGIES): in its own
#: equilibrium ('eq', the solvent relaxed around it) or in the slow charges
#: of S0's equilibrium there ('noneq', the solvent left as S0 had it).
SOLVENT_SCHEMES = ('gas', 'scheme-a', 'scheme-b-eq', 'scheme-b-noneq')
#: The schemes of the state-specific loop, and the relaxed states' solvation
#: each declares (`Policies.relaxed_solvation`).
SCHEME_B = {'scheme-b-eq': 'equilibrium', 'scheme-b-noneq': 'nonequilibrium'}
#: The geometry optimizers (`properties.optimize.relax`).
ENGINES = ('auto', 'cartesian', 'geometric')
#: How each SCF is finished.
SCF_FINISH = ('diis', 'newton')
#: E_0 under the excited state: E_KS of the reference ('dft'), or
#: E_HF + E_c^dRPA on it ('rpa').
GROUNDS = ('dft', 'rpa')
#: How the self-energy's real-axis residues are taken on the chain: the two
#: of the one vocabulary (`RESIDUE_ROUTES_ALL`) a production run declares,
#: the pole model first (the default).
RESIDUES = ('sop', 'explicit')
if not set(RESIDUES) <= set(RESIDUE_ROUTES_ALL):
    raise ImportError(f'declaration residues {RESIDUES} leave the one '
                      f'vocabulary {RESIDUE_ROUTES_ALL}')
#: The mean-field routes of the production chain.
SCF_ROUTES = ('isdf', 'df')
#: What the gas phase is called.
GAS = 'gas'
#: The fields that admit None ('Class.field'); every other is required.
OPTIONAL = ('Policies.far_start', 'Policies.external_hessian',
            'Declaration.hessian_numerics')


class DeclarationRefused(ValueError):
    """A declaration that leaves a field out or declares an impossible run."""


@dataclass(frozen=True)
class System:
    """The molecule: a name (the store's directory), the pyscf atom string
    in Angstrom, and the orbital basis."""
    name: str
    atom: str
    basis: str


@dataclass(frozen=True)
class FarStart:
    """A start of the S1 walk away from R0: the geometry (Angstrom, the atom
    order of the input) as a pyscf atom string. The start is its own
    reference: its chain is built there (`stages.Run.started_at`)."""
    atom: str

    def sha256(self):
        """The digest every record of a walk from this start carries
        (`record_hash` of the atom string)."""
        return record_hash(self.atom)


@dataclass(frozen=True)
class ExternalHessian:
    """The S0 Hessian at R0 computed elsewhere: an ORCA .hess file
    (`properties.duschinsky.read_orca_hessian`), declared by its path and
    the sha256 of its content (`of`), which the hess stage checks again; its
    geometry must be R0's (`stages.ground_modes`)."""
    path: str
    sha256: str

    @classmethod
    def of(cls, path):
        """The declaration of the file at `path`, its content hashed now."""
        path = pathlib.Path(path).resolve()
        return cls(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass(frozen=True)
class Policies:
    """The procedural choices that move a number, each one explicit.

    nudge_bohr: the largest atomic displacement of the seeded
        symmetry-breaking start of every excited walk (`start_nudge`); 0
        starts each walk exactly at its geometry and hands it the evaluation
        there as its first point.
    far_start: a `FarStart` for the S1 walk, or None to walk from R0.
    refreeze: rebuild the frozen conventions at each excited minimum and walk
        again while the drift exceeds REFREEZE_TOL_MEV (`refreeze_pass`).
    saddle_check: difference the state's force into its Hessian at each
        excited minimum and leave a saddle along its negative mode.
    df_check: after vert, s1 and t1, the ISDF-vs-DF acceptance check at the
        stage's geometry (`df_check.check_stage`); the rate refuses a failed
        one.
    hessian: the S0 Hessian's route (HESSIAN_ROUTES).
    external_hessian: the `ExternalHessian` of the 'external' route, else
        None.
    relaxed_solvation: in a continuum, where the relaxed states and their
        walks stand (RELAXED_SOLVATION); the vertical stage is
        non-equilibrium either way.
    scf_warm_start: start each walk's SCF from the previous geometry's
        density.
    engine: the walk's optimizer (ENGINES).
    scf_finish: 'diis', or second-order steps at the end ('newton').
    """
    nudge_bohr: float
    far_start: Optional[FarStart]
    refreeze: bool
    saddle_check: bool
    df_check: bool
    hessian: str
    relaxed_solvation: str
    scf_warm_start: bool
    engine: str
    scf_finish: str
    external_hessian: Optional[ExternalHessian]


@dataclass(frozen=True)
class TadfModel:
    """The TADF rate table (`rate_models.tadf_rate_table`): every
    spin-orbit derivative analytic, from the minima's own evaluations."""
    name: str = 'tadf'


@dataclass(frozen=True)
class UiscModel:
    """The UISC protocol of Samanta, Kim, Coropceanu and Bredas, J. Am.
    Chem. Soc. 139, 4042 (2017), and ours one approximation at a time
    (`uisc`): |dV/dq| by finite differences of |V|^2 along the n_modes
    promoting modes of largest Huang-Rhys weight, `dq` (dimensionless) each
    way, over the lowest `s_block` singlets and `t_block` triplets at each
    crossing's initial minimum; `lambda_samanta` the reorganization energies
    (eV) their Sec. 2 scans with S = 0."""
    n_modes: int
    dq: float
    s_block: int
    t_block: int
    lambda_samanta: tuple
    name: str = 'uisc'


@dataclass(frozen=True)
class Declaration:
    """One run (module docstring). Every field is required.

    ground: GROUNDS, E_0 under the excited state.
    reference: the Kohn-Sham functional of the mean field and of E_0.
    environment: 'gas' or a continuum as `src.environments.medium.parse`
        reads it: a solvent of pyscf's SMD table, 'eps=EPS_S,n2=N2', or the
        older 'N2:EPS_STATIC', each optionally ',cavity=NAME'.
    solvent_scheme: SOLVENT_SCHEMES, how every state of the run stands in the
        environment: 'gas' in the gas phase and only there; in a continuum
        'scheme-a', 'scheme-b-eq' or 'scheme-b-noneq'. One scheme per run:
        every record holds it and a record of another scheme is refused.
    kernel: KERNELS. singlet, triplet: the emitting singlet and the triplet
        reservoir, 1 = lowest. n_states: the roots of each spin every
        evaluation reports, above both followed states.
    qp_states: the `QPStates` declaration of the explicit set; an adaptive
        set's targets are the followed singlet and triplet.
    rate_model: `TadfModel()` or a `UiscModel`.
    numerics: the run's `Numerics`; its scf route, mean-field grid, the
        factors' grid level, the BSE Davidson and the span cut are read from
        it. numerics.span.cut None solves every SCF in the full basis, where
        a molecule whose orbital densities the factors miss is refused (no
        cut is chosen by refusal); a declared cut solves it in the
        canonical span above that overlap eigenvalue, at the input geometry's
        count. hessian_numerics: the finite-difference Hessian's numerics,
        derived from `numerics` (`Numerics.derive`), or `numerics` itself
        where they already are the Hessian's; required for the numerical
        route and refused for the analytic one.
    temperature: K. lambda_env_ev: the classical reorganization of the
        surroundings, eV.
    manifold_gaps: MANIFOLD_GAP_RULES, the gaps of the rate table's state
        manifold (`rates.ManifoldGaps`): 'stacked', each state at its spin's
        adiabatic energy plus its vertical splitting at the crossing's
        initial minimum; 'adiabatic', each state at its own minimum, which
        the run relaxes for S1 and T1 only (the manifold then holds those
        two and is marked incomplete).
    """
    system: System
    ground: str
    reference: str
    environment: str
    solvent_scheme: str
    kernel: str
    singlet: int
    triplet: int
    n_states: int
    qp_states: QPStates
    numerics: Numerics
    hessian_numerics: Optional[Numerics]
    policies: Policies
    temperature: float
    lambda_env_ev: float
    rate_model: object
    manifold_gaps: str

    def __post_init__(self):
        refuse_missing(self)
        refuse_choices(self)
        refuse_environment(self)
        refuse_states(self)
        refuse_grids(self)
        refuse_hessian_numerics(self)
        refuse_scheme(self)

    # --------------------------------------------------------------- reads
    @property
    def gas(self):
        """Whether the run stands in the gas phase."""
        return self.environment == GAS

    @property
    def solvation(self):
        """What the relaxed states stand in: 'gas', or the continuum's
        mode."""
        return GAS if self.gas else self.policies.relaxed_solvation

    @property
    def scheme_b(self):
        """Whether every state stands in the state-specific loop."""
        return self.solvent_scheme in SCHEME_B

    @property
    def environment_dir(self):
        """The environment as a directory name ('gas', 'toluene',
        '2.2_7.6')."""
        return self.environment.replace(':', '_')

    @property
    def handover(self):
        """Whether a walk starts where an evaluation already holds its force,
        which it takes as its first point; a nudged walk starts elsewhere."""
        return self.policies.nudge_bohr == 0.0

    @property
    def mean_field_name(self):
        """The mean field's route and, for ISDF-K, its grid."""
        scf = self.numerics.scf
        return 'DF jkfit' if scf.route == 'df' else f'ISDF-K {scf.grid_level}'

    def to_record(self):
        """The declaration as plain values (`record_io`), numerics by their
        own record."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out['system'] = asdict(self.system)
        out['policies'] = asdict(self.policies)
        out['qp_states'] = asdict(self.qp_states)
        out['rate_model'] = asdict(self.rate_model)
        out['numerics'] = self.numerics.to_record()
        out['hessian_numerics'] = (None if self.hessian_numerics is None
                                   else self.hessian_numerics.to_record())
        return out

    def identity(self):
        """sha256 of the whole declaration; the numerics by their identity
        (provenance is not a choice)."""
        out = self.to_record()
        out['numerics'] = self.numerics.identity()
        out['hessian_numerics'] = (None if self.hessian_numerics is None
                                   else self.hessian_numerics.identity())
        return record_hash(out)


def refuse_missing(obj):
    """Refuse a declaration (or one of its parts) with a field left None
    where the field admits no None (OPTIONAL), or of the wrong kind."""
    parts = {'system': System, 'policies': Policies}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is None and f'{type(obj).__name__}.{f.name}' not in OPTIONAL:
            raise DeclarationRefused(
                f'{type(obj).__name__}.{f.name} is not declared: every field '
                f'is explicit, nothing defaults')
        if f.name in parts and not isinstance(value, parts[f.name]):
            raise DeclarationRefused(f'{f.name} is a {parts[f.name].__name__}')
        if f.name in parts:
            refuse_missing(value)
    if isinstance(obj, Declaration) and not isinstance(obj.numerics, Numerics):
        raise DeclarationRefused('numerics is the run\'s one Numerics')


def choose(name, value, allowed):
    """Refuse `value` of field `name` outside `allowed`."""
    if value not in allowed:
        raise DeclarationRefused(f'{name}={value!r} not in {allowed}')


def refuse_environment(d):
    """Refuse an environment that names no medium (`medium.parse`): an unknown
    solvent, a medium with only one of its two constants, an unknown cavity."""
    try:
        parse_medium(d.environment)
    except ValueError as err:
        raise DeclarationRefused(f'environment {d.environment!r}: {err}') from err


def refuse_choices(d):
    """Refuse a named choice outside its allowed values, and a part of the
    wrong kind."""
    p = d.policies
    choose('kernel', d.kernel, KERNELS)
    choose('ground', d.ground, GROUNDS)
    choose('numerics.qp.residue_route', d.numerics.qp.residue_route,
           RESIDUES)
    if not isinstance(d.qp_states, QPStates):
        raise DeclarationRefused('qp_states is a QPStates')
    if not isinstance(d.rate_model, (TadfModel, UiscModel)):
        raise DeclarationRefused('rate_model is a TadfModel or a UiscModel')
    choose('manifold_gaps', d.manifold_gaps, MANIFOLD_GAP_RULES)
    choose('policies.relaxed_solvation', p.relaxed_solvation,
           RELAXED_SOLVATION)
    choose('policies.hessian', p.hessian, HESSIAN_ROUTES)
    choose('policies.engine', p.engine, ENGINES)
    choose('policies.scf_finish', p.scf_finish, SCF_FINISH)
    choose('solvent_scheme', d.solvent_scheme, SOLVENT_SCHEMES)
    choose('numerics.scf.route', d.numerics.scf.route, SCF_ROUTES)
    cut = d.numerics.span.cut
    if cut is not None and not 0 < cut < 1:
        raise DeclarationRefused(
            f'numerics.span.cut={cut!r}: None (the full basis) or an overlap '
            f'eigenvalue between 0 and 1')
    if p.nudge_bohr < 0:
        raise DeclarationRefused(f'policies.nudge_bohr={p.nudge_bohr}: '
                                 'a displacement is not negative')


def refuse_states(d):
    """Refuse followed states the evaluations cannot reach: the adaptive
    set's targets, the Davidson's roots and the state above each followed
    one."""
    targets = (('singlet', d.singlet - 1), ('triplet', d.triplet - 1))
    if d.qp_states.kind == 'adaptive' and \
            tuple(d.qp_states.targets or ()) != targets:
        raise DeclarationRefused(
            f'the adaptive set holds its budget for the followed states '
            f'{targets}, not {d.qp_states.targets}')
    nroots = d.numerics.bse.nroots
    if nroots is None or d.n_states >= nroots:
        raise DeclarationRefused(
            f'n_states={d.n_states} must lie below numerics.bse.nroots'
            f'={nroots}: the Davidson refuses an unconverged root up to '
            f'the highest one read')
    if max(d.singlet, d.triplet) + 1 > d.n_states:
        raise DeclarationRefused(
            'n_states must include the state above the singlet and the '
            'triplet followed: the second-order coupling runs through it')
    if min(d.singlet, d.triplet) < 1:
        raise DeclarationRefused('singlet and triplet count from 1')


def refuse_grids(d):
    """Refuse a factor grid or an ISDF-K mean field not declared by its
    level."""
    if d.numerics.isdf.level is None:
        raise DeclarationRefused('numerics.isdf.level: the factors\' grid '
                                 'is declared by its level')
    if d.numerics.scf.route == 'isdf' and d.numerics.scf.grid_level is None:
        raise DeclarationRefused('numerics.scf.grid_level: an ISDF-K mean '
                                 'field is declared on a grid level')


def refuse_hessian_numerics(d):
    """Refuse Hessian numerics the route does not read, or that are neither
    the run's numerics nor derived from them, and an external Hessian file
    on another route or none on the external one."""
    external = d.policies.external_hessian
    if (d.policies.hessian == 'external') != (external is not None):
        raise DeclarationRefused(
            'policies.external_hessian is the file of the external Hessian '
            'route and of no other: declare both or neither')
    if external is not None and not isinstance(external, ExternalHessian):
        raise DeclarationRefused('policies.external_hessian is an '
                                 'ExternalHessian')
    if d.policies.hessian != 'numerical':
        if d.hessian_numerics is not None:
            raise DeclarationRefused('hessian_numerics is read by the '
                                     'numerical Hessian only')
        return
    h = d.hessian_numerics
    if h is None:
        raise DeclarationRefused(
            'the numerical Hessian needs hessian_numerics, derived from '
            'numerics (Numerics.derive) with its grid')
    if h.parent != d.numerics.identity() and \
            h.identity() != d.numerics.identity():
        raise DeclarationRefused(
            'hessian_numerics is neither numerics nor derived from them (its '
            'parent is not numerics.identity())')


#: What scheme B does not run, each with the reason (`refuse_scheme`).
SCHEME_B_REFUSES = (
    (lambda d: d.ground != 'dft',
     'ground: the loop\'s E_0 is the reference\'s energy in its charges (dft)'),
    (lambda d: d.numerics.scf.route != 'df',
     'numerics.scf.route: the loop converges a density-fitted mean field (df)'),
    (lambda d: d.numerics.span.cut is not None,
     'numerics.span.cut: the loop solves every SCF in the full basis'),
    (lambda d: d.numerics.scf.conv_tol_grad > STATE_MODEL_SCF_GRAD_TOL,
     f'numerics.scf.conv_tol_grad: a state model\'s SCF converges to '
     f'{STATE_MODEL_SCF_GRAD_TOL:.0e} or tighter'),
    (lambda d: d.numerics.solvent.loop_tol_energy < d.numerics.scf.conv_tol,
     'numerics.solvent.loop_tol_energy: below the SCF\'s conv_tol the loop\'s G '
     'cannot settle and every loop runs to its cycle cap'),
    (lambda d: d.policies.far_start is not None,
     'policies.far_start: a far start\'s chain is not a loop\'s surface'),
    (lambda d: d.policies.df_check,
     'policies.df_check: scheme B has no density-fitted route to check '
     'against (its forces are gated against their own differences)'),
    (lambda d: d.policies.scf_finish != 'diis',
     'policies.scf_finish: the state model converges its SCF by DIIS'),
    (lambda d: d.policies.relaxed_solvation != SCHEME_B[d.solvent_scheme],
     'policies.relaxed_solvation: scheme-b-eq declares the relaxed states in '
     'equilibrium, scheme-b-noneq in non-equilibrium'),
    (lambda d: d.qp_states.kind in ('admitted', 'adaptive')
     or (d.qp_states.kind == 'valence' and d.qp_states.filter_z),
     'qp_states: every scheme-B surface (each vertical, each refreeze) '
     'builds its own chain, and a set selected at the geometry it is built '
     'at differs from surface to surface; declare one fixed by orbital '
     'index (frontier, valence without the Z filter, or all)'),
    (lambda d: not isinstance(d.rate_model, TadfModel),
     'rate_model: the UISC stage rebuilds chains outside the loop'),
)


def refuse_scheme(d):
    """Refuse a solvent scheme that does not match the environment, and under
    scheme B every choice the loop does not run (SCHEME_B_REFUSES)."""
    if d.gas != (d.solvent_scheme == GAS):
        raise DeclarationRefused(
            f'solvent_scheme={d.solvent_scheme!r} in environment '
            f'{d.environment!r}: \'gas\' is the scheme of the gas phase, '
            f'and of it alone')
    if not d.scheme_b:
        return
    for refused, why in SCHEME_B_REFUSES:
        if refused(d):
            raise DeclarationRefused(f'{d.solvent_scheme}: {why}')
