"""Every numerical choice of a calculation, passed once: `Numerics`.

A grid, a count, a tolerance, a quadrature size, a route that changes the
bits -- each is decided ONCE, by the driver, into one immutable object, and
every product computed from it carries its identity (`Numerics.identity`).
Two products of different numerics are two answers to two questions; a seam
that combines them refuses. A task that genuinely needs other numerics (a
finite-difference Hessian on a finer grid) DERIVES them (`derive`): the child
records its parent's identity, what changed and why, so the difference is
written down where it happens and nowhere else.

What is NOT here: physics constants and the bars runs are judged by
(`src.Base.numerics.constants`), and the frozen RESULTS of a calculation at its
reference geometry (`gradients.gwbse.conventions.FrozenConventions`: the resolved
grids, the explicit set, the scissor), which are derived from these numerics
and the reference, not chosen.

The fields:

    isdf        the post-SCF interpolation grid and its fit
    scf         the reference mean field's route and convergence, and of
                its orbital response (the CPHF / Z-vector solve)
    span        the canonical span cut of an augmented basis
    qp          which quasiparticle energies are solved, and how
    quadrature  the imaginary-time, frequency and contour grids
    bse         the Casida eigensolver and the root's adjoint
    solvent     the continuum, by label, the fold, and the self-consistent
                solvation loop's convergence
    walk        the geometry optimizer and the finite-difference steps
    layout      memory and tiling: recorded, not in the identity

The excited-state chain and the composed surfaces hold one and read every
setting from it (`chain_settings.chain_setting`); the drivers build it
(`gradients.gwbse.surface_factory.excited_kwargs`). A section may be given as a dict of
its fields, `Numerics(bse={'solver': 'dense'})`, which the constructor turns
into the section's record.
"""
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

from src.Base.numerics.constants import (BSE_ADJOINTS, BSE_DAVIDSON_CONV_TOL,
                                         BSE_DAVIDSON_NROOTS, BSE_FORCE_MAX_CYCLE,
                                         BSE_SOLVERS, CD_NTAU, CPHF_MAX_CYCLE,
                                         CPHF_REFINEMENTS, CPHF_TOL,
                                         DAVIDSON_GUESS_FACTOR,
                                         DEFAULT_BROADENING_ETA,
                                         DEFAULT_NPADE, DEFAULT_TAU_TARGET,
                                         EVGW_DAMPING, EVGW_DIIS_SIZE, EVGW_DIIS_START,
                                         EVGW_MAX_CYCLE, EVGW_TOL, IMAGFREQ_NFREQ,
                                         QSGW_BLOCK_ELEMS, QSGW_DIIS_SIZE, QSGW_DM_TOL,
                                         QSGW_MIXING_LAMBDA, QSGW_SRG_FLOW,
                                         GL_FREQUENCY_SCALE,
                                         DAVIDSON_PRECONDITIONER,
                                         DAVIDSON_PRECONDITIONERS,
                                         FIT_REALIZATIONS, ISDF_FIT_L_MAX_SECOND,
                                         ISDF_FIT_PAIR_TOL, ISDF_FIT_REGULARIZATION,
                                         NUCLEAR_FD_STEP,
                                         OUTSIDE_SCISSOR_MODES,
                                         QP_DEGENERACY_TOL, ROOT_FOLLOWING,
                                         RPA_ENERGY_NFREQ, RPA_ENERGY_NFREQ_MINIMAX,
                                         SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL, SCF_MAX_CYCLE,
                                         SOLVATION_LOOP_MAX_CYCLE,
                                         SOLVATION_LOOP_TOL_ENERGY,
                                         SOLVATION_LOOP_TOL_Q,
                                         SOP_N_POLES, WALK_MAX_CYCLE, WALK_TRUST_BOHR,
                                         WALK_TRUST_MAX_BOHR, WALK_TRUST_MIN_BOHR,
                                         XC_GRID_LEVEL, XC_GRID_LEVEL_UNRECORDED)
from src.Base.numerics.constants import (WALK_GROUND_MAX_CYCLE, WALK_GROUND_TRUST_BOHR)
from src.Base.numerics.constants import (ORBITAL_MULTIPLIER_MAX_ITER,
                                         ORBITAL_MULTIPLIER_RESIDUAL_TOL,
                                         ORBITAL_MULTIPLIER_TOL)
from src.Base.utils.record_io import record_hash


@dataclass(frozen=True)
class IsdfNumerics:
    """The interpolation grid of the post-SCF factorization and its fit.

    level: a validated accuracy level ('G1', 'G2', 'G3', 'GH'), or None for
        explicit counts; counts: the shell counts it resolved to (or as
        given); radii: explicit per-element shell radii, None for the shipped
        table's; n_start: the radii recipe's multi-start count, None for no
        opinion (the single start where a chain builds its own
        factorization, a shared one's own otherwise); basis,
        auxbasis: the basis the grid is fitted in and its auxiliary basis,
        None for the mean field's and its '-ri' partner.
    l_max_second, pair_tol, regularization: the fit's test co-densities,
        pair screening and Tikhonov epsilon (`separable_ri`).
    fit: the fit's realization ('replicated' or 'rows'), which agree to
        rounding and no closer; None takes a given factorization's, else
        'replicated' (`FrozenFactorization`). Its tile edge and the layout
        over ranks are `LayoutNumerics`; the bound on the factors' orbital
        residual is a bar (ISDF_ORBITAL_RESIDUAL_TOL), not a choice.
    runtime_radii: True declares a grid the shipped radii table holds no
        row for, re-optimized at run time (`runtime_atomic_radii`); False
        refuses one (`UntabulatedISDFGrid`). Not part of `identity`: where a
        row exists it changes nothing, and where none does the undeclared
        calculation does not exist, so it never tells two calculations
        apart; the grid itself is the factorization's radii_tag.
    """
    level: Optional[str] = None
    counts: Optional[object] = None
    radii: Optional[dict] = None
    n_start: Optional[int] = None
    basis: Optional[str] = None
    auxbasis: Optional[str] = None
    l_max_second: int = ISDF_FIT_L_MAX_SECOND
    pair_tol: float = ISDF_FIT_PAIR_TOL
    regularization: float = ISDF_FIT_REGULARIZATION
    fit: Optional[str] = None
    runtime_radii: bool = False

    def __post_init__(self):
        if self.fit is not None and self.fit not in FIT_REALIZATIONS:
            raise ValueError(f'fit={self.fit!r} not in {FIT_REALIZATIONS}')


@dataclass(frozen=True)
class ScfNumerics:
    """The reference mean field: its J/K route ('df', 'isdf', 'exact'), the
    route's auxiliary basis or ISDF-K grid level, pyscf's
    exchange-correlation grid level of a Kohn-Sham one (xc_grid,
    XC_GRID_LEVEL), and its convergence; and
    the orbital response of that mean field (`src.Base.pyscf.cphf.solve`):
    the true residual its solution reaches (cphf_tol), the Krylov cycles of
    one solve (cphf_max_cycle) and the rounds of solving again for the
    residual left (cphf_refinements); and the orbital-response (Z-vector)
    solve of every gradient (`src.gradients.multipliers`): its lgmres
    tolerance (multiplier_tol), iteration ceiling (multiplier_max_iter) and
    the true residual it must reach (multiplier_residual_tol), each relative
    to max(1, |rhs|)."""
    route: Optional[str] = None
    auxbasis: Optional[str] = None
    grid_level: Optional[str] = None
    j_route: Optional[str] = None
    xc_grid: int = XC_GRID_LEVEL
    conv_tol: float = SCF_DIFFERENTIABLE_CONV_TOL
    conv_tol_grad: float = SCF_DIFFERENTIABLE_GRAD_TOL
    max_cycle: int = SCF_MAX_CYCLE
    cphf_tol: float = CPHF_TOL
    cphf_max_cycle: int = CPHF_MAX_CYCLE
    cphf_refinements: int = CPHF_REFINEMENTS
    multiplier_tol: float = ORBITAL_MULTIPLIER_TOL
    multiplier_max_iter: int = ORBITAL_MULTIPLIER_MAX_ITER
    multiplier_residual_tol: float = ORBITAL_MULTIPLIER_RESIDUAL_TOL


@dataclass(frozen=True)
class SpanNumerics:
    """The overlap eigenvalue below which an augmented basis is solved in the
    canonical span above it; None solves the full basis."""
    cut: Optional[float] = None


@dataclass(frozen=True)
class QPNumerics:
    """Which quasiparticle energies are solved explicitly and how the rest
    are shifted, with ONE tolerance below which two orbitals are one
    degenerate block, for the window and the set's shared corrections alike.

    window: the explicitly solved set -- a frontier half-width, 'all', or
        the orbital indices a `QPStates` declaration resolved to; select:
        QPStates('adaptive') to select the set out of those candidates at
        the first forward.
    residue_route: how the real-axis residues of Sigma^c are taken
        ('explicit', 'laplace', 'sop', or 'none' where there are none).
    scissor: 'calibrate' freezes a shift for each set member the pole model
        does not admit, from its root at the reference geometry; None keeps
        those on the fallback route.
    outside_scissor: the shape of the frozen shift beyond the set
        (OUTSIDE_SCISSOR_MODES).
    hamiltonian: solve the set's near-degenerate groups as one
        quasiparticle Hamiltonian each.
    degeneracy_tol: orbitals closer than this are one block, of the frontier
        set and of the explicit set, where a block whose own corrections
        agree within the bar QP_BLOCK_CORRECTION_TOL shares one correction.
    n_poles, sop_stride: the pole model's poles and the column stride of
        their fit.
    """
    window: object = 2
    select: Optional[object] = None
    residue_route: str = 'explicit'
    scissor: Optional[str] = None
    outside_scissor: str = 'tiers'
    hamiltonian: bool = False
    degeneracy_tol: float = QP_DEGENERACY_TOL
    n_poles: int = SOP_N_POLES
    sop_stride: Optional[int] = None

    def __post_init__(self):
        if self.residue_route == 'auto':
            raise ValueError(
                "residue_route='auto' decides the residue backend per state at "
                'every geometry, so a walk could change it: name one '
                "('explicit', 'laplace' or 'sop')")
        if self.scissor not in (None, 'calibrate'):
            raise ValueError(
                f"scissor={self.scissor!r}: 'calibrate' or None. A fixed shift "
                'for the states the pole model excludes is refused: it costs '
                'up to 7.5 meV and 3e-4 Ha/Bohr where the calibrated one costs '
                'nothing')
        if self.outside_scissor not in OUTSIDE_SCISSOR_MODES:
            raise ValueError(f'outside_scissor={self.outside_scissor!r} not '
                             f'in {OUTSIDE_SCISSOR_MODES}')
        if self.select is not None and getattr(
                self.select, 'kind', None) != 'adaptive':
            raise ValueError(f"select is a QPStates(kind='adaptive'), not "
                             f'{self.select!r}')
        object.__setattr__(self, 'hamiltonian', bool(self.hamiltonian))


@dataclass(frozen=True)
class QuadratureNumerics:
    """The grids: the self-energy's tau points (ntau_gw), the static W's
    (ntau_w, None for the minimax count of the gap), the contour's
    frequencies (nfreq_cd, None for the size the set needs, a floor when
    given), how far below the gap the tau axis is fitted, and the dRPA
    ground state's tau and frequency counts: None derives them from the
    excited state's; a value of their own comes only through `derive`.
    The space-time self-energy (`solve_qp_energy_space_time`)
    reads its own: `ntau_st` tau points and `nfreq_st` chi0 frequencies
    ('auto': sized from the spectrum), `npade` Pade points on the
    Gauss-Legendre grid of scale `w0_gl`; every 'auto' tau grid -- the
    self-energy's, the static W's (`ntau_w` None, `isdf_bse_factors`) -- is
    sized at the one minimax target `tau_target`."""
    ntau_gw: int = CD_NTAU
    ntau_w: Optional[int] = None
    nfreq_cd: Optional[int] = None
    e_min_below_gap: Optional[float] = None
    ntau_rpa: Optional[int] = None
    nfreq_rpa: Optional[int] = None
    tau_target: float = DEFAULT_TAU_TARGET
    ntau_st: object = 'auto'
    nfreq_st: object = 'auto'
    npade: int = DEFAULT_NPADE
    w0_gl: float = GL_FREQUENCY_SCALE


@dataclass(frozen=True)
class BSENumerics:
    """The Casida eigensolver (BSE_SOLVERS), its roots and residual, the
    Davidson's cycle cap, guess count per root and preconditioner, the
    (A - B) probe, the root's adjoint and how the state is followed. The
    Davidson's entry points (`solve_casida_davidson`, `solve_bse_isdf`,
    `solve_bse_df`, `solve_bse`) and the chain read the solver settings from
    here, so a stand-alone solve and the chain's converge to one residual.

    probe: the gradient chains' (A - B) probe (`excited_state`, the surface
        factory, the ISDF-vs-DF check). After every full-BSE matrix-free solve
        of a chain, True converges min eig(A - B) by the symmetric Davidson
        from 1/d and refuses the step (`UnstableReference`) when it is not
        positive; 'sign' only proves its sign, from the roots' span first,
        which cannot see a minimum in a pair irrep no root reaches; False
        skips it, only where asked for by name. The stand-alone drivers
        (`solve_bse`, `solve_bse_isdf`, `solve_bse_df`) do not read it: they
        probe only when called with `probe=True` or `'sign'`, since on a large
        DF run the probe can cost more than the solve. The dense solver
        refuses an indefinite (A - B) by its Cholesky whatever this says, and
        the Tamm-Dancoff kernel is not probed.
    adjoint: how the root's adjoint is formed (BSE_ADJOINTS); track: None
        follows the state's index, a mode of ROOT_FOLLOWING its Casida
        vector.
    """
    solver: str = 'auto'
    nroots: Optional[int] = BSE_DAVIDSON_NROOTS
    conv_tol: float = BSE_DAVIDSON_CONV_TOL
    max_cycle: int = BSE_FORCE_MAX_CYCLE
    guess_factor: int = DAVIDSON_GUESS_FACTOR
    preconditioner: str = DAVIDSON_PRECONDITIONER
    probe: object = True
    adjoint: str = 'explicit'
    track: Optional[str] = None

    def __post_init__(self):
        if self.solver not in BSE_SOLVERS:
            raise ValueError(f'solver={self.solver!r} not in {BSE_SOLVERS}')
        if self.adjoint not in BSE_ADJOINTS:
            raise ValueError(f'adjoint={self.adjoint!r} not in '
                             f'{BSE_ADJOINTS}')
        if self.track is not None and self.track not in ROOT_FOLLOWING:
            raise ValueError(f'track={self.track!r} not in {ROOT_FOLLOWING}')
        if self.preconditioner not in DAVIDSON_PRECONDITIONERS:
            raise ValueError(f'preconditioner={self.preconditioner!r} not in '
                             f'{DAVIDSON_PRECONDITIONERS}')


@dataclass(frozen=True)
class SolventNumerics:
    """How a continuum is computed: the cavity's scale and discretization,
    the reaction field's set, and whether the correlation energy folds it.
    None takes the environment's own. The continuum's PHYSICS (model, eps)
    is the environment's and is hashed through it. The self-consistent
    solvation loop (`src.environments.solvation_loop.equilibrate`) stops when the
    largest change of an induced charge is below loop_tol_q (e) and the
    change of G below loop_tol_energy (Ha), within loop_max_cycle cycles."""
    cavity_scale: Optional[float] = None
    discretization: Optional[object] = None
    reaction_field: Optional[object] = None
    fold: bool = False
    loop_tol_q: float = SOLVATION_LOOP_TOL_Q
    loop_tol_energy: float = SOLVATION_LOOP_TOL_ENERGY
    loop_max_cycle: int = SOLVATION_LOOP_MAX_CYCLE


@dataclass(frozen=True)
class WalkNumerics:
    """The geometry optimizer and the finite-difference steps; ground_*:
    the step cap and first trust radius of a mean-field ground state's own
    relaxation (`optimize.relax_ground_state`)."""
    max_cycle: int = WALK_MAX_CYCLE
    trust: float = WALK_TRUST_BOHR
    trust_max: float = WALK_TRUST_MAX_BOHR
    trust_min: float = WALK_TRUST_MIN_BOHR
    fd_step: float = NUCLEAR_FD_STEP
    ground_max_cycle: int = WALK_GROUND_MAX_CYCLE
    ground_trust: float = WALK_GROUND_TRUST_BOHR


@dataclass(frozen=True)
class LayoutNumerics:
    """Memory and tiling: the working-set cap of the tiled intermediates, the
    row fit's tile edge, the factors cut over ranks, the ISDF-K Z mode and
    row block, and the working set of the fit's blocked loops (`fit_gb`,
    ISDF_TILE_GB where None; its block edge moves the fit's last bits).
    Recorded, and NOT in the identity: the bitwise gates fix
    them explicitly. sliced None takes a given factorization's, else the
    whole factors on every rank."""
    tile_gb: Optional[float] = None
    fit_block: Optional[int] = None
    fit_gb: Optional[float] = None
    sliced: Optional[bool] = None
    z_mode: Optional[str] = None
    block: Optional[int] = None


@dataclass(frozen=True)
class EvGWNumerics:
    """The eigenvalue-self-consistent loop (`GW.evGW.evgw_eigenvalues`).

    tol: convergence on max |delta eps| over the tested orbitals, Hartree;
    max_cycle: cycles before the last iterate is returned, warned and
    flagged; diis_size: DIIS subspace, 0 for linear mixing by `damping`
    (eps <- (1 - d) eps_new + d eps_old); diis_start: the cycle DIIS starts on.
    """
    tol: float = EVGW_TOL
    max_cycle: int = EVGW_MAX_CYCLE
    diis_size: int = EVGW_DIIS_SIZE
    diis_start: int = EVGW_DIIS_START
    damping: float = EVGW_DAMPING


@dataclass(frozen=True)
class QsGWNumerics:
    """The quasiparticle-self-consistent loop (`GW.qsGW.qsgw_eigenvalues`).

    tol: HOMO/LUMO movement, Hartree; dm_tol: density change ||D' - D||_F /
    nmo; max_cycle; diis_size; mixing_lambda: the new Hamiltonian's weight in
    linear mixing; flow: the SRG flow parameter s, Hartree^-2; block_elems:
    the static self-energy builder's buffer size, in doubles; eta: the
    broadening of its denominators, Hartree.
    """
    tol: float = EVGW_TOL
    max_cycle: int = EVGW_MAX_CYCLE
    dm_tol: float = QSGW_DM_TOL
    diis_size: int = QSGW_DIIS_SIZE
    mixing_lambda: float = QSGW_MIXING_LAMBDA
    flow: float = QSGW_SRG_FLOW
    block_elems: int = QSGW_BLOCK_ELEMS
    eta: float = DEFAULT_BROADENING_ETA


@dataclass(frozen=True)
class ImagFreqNumerics:
    """The O(N^4) imaginary-frequency GW route (`GW.imaginary_axis`): nfreq
    points of its W(i omega) quadrature."""
    nfreq: int = IMAGFREQ_NFREQ


#: The sections of `Numerics`, by field name.
SECTIONS = {'isdf': IsdfNumerics, 'scf': ScfNumerics, 'span': SpanNumerics,
            'qp': QPNumerics, 'quadrature': QuadratureNumerics,
            'bse': BSENumerics, 'solvent': SolventNumerics,
            'walk': WalkNumerics, 'layout': LayoutNumerics,
            'evgw': EvGWNumerics, 'qsgw': QsGWNumerics,
            'imagfreq': ImagFreqNumerics}

#: Sections added after records and identities were first written: a record
#: without one reads its defaults, and one at its defaults leaves the identity
#: what it was before the section existed.
LATER_SECTIONS = ('evgw', 'qsgw', 'imagfreq')

#: The chain's flat setting names (the keywords a `SurfaceSpec` row and the
#: chain's attributes use) and the field of `Numerics` each one is.
FLAT_FIELDS = {
    'basis': 'isdf.basis', 'auxbasis': 'isdf.auxbasis',
    'counts': 'isdf.counts', 'n_start': 'isdf.n_start',
    'radii': 'isdf.radii', 'fit': 'isdf.fit',
    'sliced': 'layout.sliced', 'fit_block': 'layout.fit_block',
    'tile_gb': 'layout.tile_gb',
    'ntau_gw': 'quadrature.ntau_gw', 'ntau_w': 'quadrature.ntau_w',
    'nfreq_cd': 'quadrature.nfreq_cd',
    'e_min_below_gap': 'quadrature.e_min_below_gap',
    'qp_window': 'qp.window', 'qp_select': 'qp.select',
    'residue_route': 'qp.residue_route', 'scissor': 'qp.scissor',
    'outside_scissor': 'qp.outside_scissor',
    'qp_hamiltonian': 'qp.hamiltonian',
    'degeneracy_tol': 'qp.degeneracy_tol', 'n_poles': 'qp.n_poles',
    'sop_stride': 'qp.sop_stride',
    'solver': 'bse.solver', 'nroots': 'bse.nroots',
    'bse_conv_tol': 'bse.conv_tol', 'probe': 'bse.probe',
    'bse_adjoint': 'bse.adjoint', 'track': 'bse.track',
    'grid_accuracy': 'isdf.level', 'ntau_rpa': 'quadrature.ntau_rpa',
    'nfreq_rpa': 'quadrature.nfreq_rpa', 'fold': 'solvent.fold'}


class NumericsMismatch(ValueError):
    """Two products of different numerics met at a seam that combines them."""


def differing_fields(a, b):
    """['section.name', ...]: the choices, layout and provenance aside, in
    which two `Numerics` differ."""
    va, vb = a.values(), b.values()
    return [f'{section}.{name}' for section in IDENTITY_FIELDS
            for name in va[section] if va[section][name] != vb[section][name]]


def require_same_numerics(a, b, where):
    """Refuse two `Numerics` of different identity at `where`, naming the
    fields: two products of different numerics are two answers to two
    questions, and what combines them says neither."""
    if a.identity() != b.identity():
        raise NumericsMismatch(
            f'{where}: the two sides stand on different numerics '
            f'({", ".join(differing_fields(a, b))}); build both from one '
            'Numerics, or derive the other with a reason')


def require_same_identity(a, b, where):
    """Refuse two identity hashes that differ at `where`; a None is a
    product that carries none (a hand-built one) and is not refused."""
    if a is not None and b is not None and a != b:
        raise NumericsMismatch(
            f'{where}: the two products were computed on different numerics '
            f'(identity {a[:12]} against {b[:12]}); build both from one '
            'Numerics, or derive the other with a reason')


def require_same_section(a, b, section, where):
    """Refuse two `Numerics` whose `section` differs at `where`, naming the
    fields."""
    va, vb = a.values()[section], b.values()[section]
    differ = [f'{section}.{name}' for name in va if va[name] != vb[name]]
    if differ:
        raise NumericsMismatch(
            f'{where}: the two sides stand on different {section} numerics '
            f'({", ".join(differ)}); build both from one Numerics')


class Stamped(tuple):
    """A tuple result (a spectrum's (omega, X, Y)) that names the identity of
    the `Numerics` it was computed on, as `numerics_identity`."""

    def __new__(cls, values, numerics_identity):
        out = super().__new__(cls, values)
        out.numerics_identity = numerics_identity
        return out

    def __getnewargs__(self):
        return tuple(self), self.numerics_identity


#: Where a surface, a composed surface or a manifold holds its `Numerics`.
NUMERICS_HOLDERS = (None, 'chain', 'excited', 'ground')


def surface_numerics(surface):
    """The `Numerics` `surface` stands on (a chain, a composed surface -- both
    halves hold one object -- or a manifold); None for a surface that holds
    none (a dense one, a mean-field one)."""
    for name in NUMERICS_HOLDERS:
        part = surface if name is None else getattr(surface, name, None)
        held = getattr(part, '__dict__', {}).get('settings')
        if isinstance(held, Numerics):
            return held
    return None


def numerics_identity(surface):
    """The identity of `surface_numerics(surface)`; None where it holds none."""
    held = surface_numerics(surface)
    return None if held is None else held.identity()


def stamp_of(product):
    """The identity a product carries, None if it carries none."""
    if isinstance(product, dict):
        return product.get('numerics_identity')
    return getattr(product, 'numerics_identity', None)


def require_one_stamp(products, where):
    """Refuse products stamped with different identities at `where`."""
    held = [(p, stamp_of(p)) for p in products]
    stamps = [h for _, h in held if h is not None]
    for other in stamps[1:]:
        require_same_identity(stamps[0], other, where)


def rpa_quadrature(numerics, grid='gauss_legendre'):
    """(ntau, nfreq) of a dRPA correlation energy on `numerics`: its own
    counts where `derive` gave it some, else derived from the excited
    state's -- the static screening's tau count `ntau_w` (None:
    the minimax count of the gap, the same rule) and the energy integral's
    frequencies, the excited half having no frequency quadrature of that
    integrand: RPA_ENERGY_NFREQ on a Gauss-Legendre grid,
    RPA_ENERGY_NFREQ_MINIMAX on a minimax one (`grid='minimax'`), whose
    tables stop at 34."""
    q = numerics.quadrature
    if grid == 'minimax':
        nfreq = RPA_ENERGY_NFREQ_MINIMAX
    elif grid == 'gauss_legendre':
        nfreq = RPA_ENERGY_NFREQ
    else:
        raise ValueError(f"Unknown grid '{grid}'; choose 'minimax' or 'gauss_legendre'.")
    return (q.ntau_w if q.ntau_rpa is None else q.ntau_rpa,
            nfreq if q.nfreq_rpa is None else q.nfreq_rpa)


def numerics_or_default(numerics):
    """`numerics`, or every setting at its default where none is given."""
    return Numerics() if numerics is None else numerics


#: The keywords of the ISDF forward (`space_time.separable_factors`, the BSE
#: and QP drivers built on it) and the `Numerics` field each one names.
ISDF_REQUEST_FIELDS = {k: FLAT_FIELDS[k] for k in (
    'auxbasis', 'radii', 'counts', 'n_start', 'grid_accuracy', 'fit', 'sliced',
    'fit_block')}

#: The same for the ISDF-K mean field (`isdf_jk.ISDFJK`, `isdf_jk.isdf_jk`).
ISDF_JK_REQUEST_FIELDS = {
    'auxbasis': 'scf.auxbasis', 'grid_accuracy': 'scf.grid_level',
    'j_route': 'scf.j_route', 'z_mode': 'layout.z_mode',
    'block': 'layout.block'}


def same_setting(a, b):
    """Whether two settings name one value: dicts key by key, arrays and
    sequences element by element."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_setting(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or hasattr(a, 'shape'):
        return (len(a) == len(b) if hasattr(b, '__len__') else False) and all(
            same_setting(x, y) for x, y in zip(a, b))
    return bool(a == b)


def requested(numerics, where, fields=ISDF_REQUEST_FIELDS, **given):
    """{keyword: value}: the settings a call stands on, one per keyword in
    `given`, each named by its field of `numerics` in `fields`.

    A keyword left None reads its field (None: every field at its default),
    so a Numerics passed alone keeps its grid; a keyword that names another
    value than a field `numerics` sets is refused, naming `where` -- one call
    cannot stand on two settings, and neither side may silently win.
    """
    held = numerics_or_default(numerics)
    out = {}
    for name, value in given.items():
        section, key = fields[name].split('.')
        field_value = getattr(getattr(held, section), key)
        if value is None:
            out[name] = field_value
            continue
        if field_value is not None and not same_setting(value, field_value):
            raise ValueError(
                f'{where}: {name}={value!r} and numerics.{section}.{key}='
                f'{field_value!r} name two settings; pass the Numerics alone, '
                f'or derive one with the value')
        out[name] = value
    return out


def numerics_of(choices):
    """The `Numerics` holding `choices`, a mapping of `FLAT_FIELDS` names to
    values, every other field at its default: what a driver that collected
    a row's keywords builds the chain's object from. A name that is no
    field is refused."""
    unknown = set(choices) - set(FLAT_FIELDS)
    if unknown:
        raise ValueError(f'no setting {sorted(unknown)} in Numerics')
    sections = {}
    for name, value in choices.items():
        section, key = FLAT_FIELDS[name].split('.')
        sections.setdefault(section, {})[key] = value
    return Numerics(**sections)


#: The sections the identity covers: all but `layout`, which is recorded
#: and not hashed.
IDENTITY_FIELDS = tuple(name for name in SECTIONS if name != 'layout')
# declarations that permit a calculation without changing any (the reason is
# at each field)
IDENTITY_EXCLUDED = (('isdf', 'runtime_radii'),)


@dataclass(frozen=True)
class Numerics:
    """Every numerical choice of one calculation (module docstring).

    parent, change, reason: where a derived object came from -- its parent's
    identity, the fields that changed ('section.name', old, new) and why --
    None on an object the driver built. A section given as a dict of its
    fields is built into its record here.
    """
    isdf: IsdfNumerics = field(default_factory=IsdfNumerics)
    scf: ScfNumerics = field(default_factory=ScfNumerics)
    span: SpanNumerics = field(default_factory=SpanNumerics)
    qp: QPNumerics = field(default_factory=QPNumerics)
    quadrature: QuadratureNumerics = field(default_factory=QuadratureNumerics)
    bse: BSENumerics = field(default_factory=BSENumerics)
    solvent: SolventNumerics = field(default_factory=SolventNumerics)
    walk: WalkNumerics = field(default_factory=WalkNumerics)
    layout: LayoutNumerics = field(default_factory=LayoutNumerics)
    evgw: EvGWNumerics = field(default_factory=EvGWNumerics)
    qsgw: QsGWNumerics = field(default_factory=QsGWNumerics)
    imagfreq: ImagFreqNumerics = field(default_factory=ImagFreqNumerics)
    parent: Optional[str] = None
    change: tuple = ()
    reason: Optional[str] = None

    def __post_init__(self):
        for name, cls in SECTIONS.items():
            if isinstance(getattr(self, name), dict):
                object.__setattr__(self, name, cls(**getattr(self, name)))
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f'{name} is a {cls.__name__}, not '
                                f'{type(getattr(self, name)).__name__}')

    def values(self):
        """{section: {name: value}}: the choices, without provenance."""
        return {name: asdict(getattr(self, name)) for name in SECTIONS}

    def to_record(self):
        """The choices and the provenance, for `from_record`."""
        return dict(self.values(), parent=self.parent,
                    change=[list(c) for c in self.change],
                    reason=self.reason)

    @classmethod
    def from_record(cls, record):
        """The object `to_record` wrote. A record whose scf section names no
        xc_grid was written before the field existed, on pyscf's default
        grid (XC_GRID_LEVEL_UNRECORDED), and is read as that."""
        sections = {name: dict(record.get(name, {})) if name in LATER_SECTIONS
                    else dict(record[name]) for name in SECTIONS}
        sections['scf'].setdefault('xc_grid', XC_GRID_LEVEL_UNRECORDED)
        return cls(**{name: SECTIONS[name](**sections[name])
                      for name in SECTIONS},
                   parent=record.get('parent'),
                   change=tuple(tuple(c) for c in record.get('change', ())),
                   reason=record.get('reason'))

    def identity(self):
        """sha256 of the choices `IDENTITY_FIELDS` covers: two products of
        one calculation carry the same one, whatever object built it and
        however it was derived (the provenance is not a choice)."""
        values = self.values()
        for section, name in IDENTITY_EXCLUDED:
            values[section].pop(name)
        return record_hash({name: values[name] for name in IDENTITY_FIELDS
                            if name not in LATER_SECTIONS
                            or values[name] != asdict(SECTIONS[name]())})

    def derive(self, reason, **sections):
        """Other numerics for another task, written down: `sections` maps a
        section name to {field: value}; the child holds this object's
        identity, every changed (field, old, new) and `reason`. A change of
        nothing, or without a reason, is refused."""
        if not reason or not str(reason).strip():
            raise ValueError('derive needs a reason: other numerics are a '
                             'different calculation, and the record must say '
                             'why it was asked for')
        unknown = set(sections) - set(SECTIONS)
        if unknown:
            raise ValueError(f'no section {sorted(unknown)} in {list(SECTIONS)}')
        new, change = {}, []
        for name, updates in sections.items():
            current = getattr(self, name)
            bad = set(updates) - {f.name for f in fields(current)}
            if bad:
                raise ValueError(f'{name} has no field {sorted(bad)}')
            new[name] = replace(current, **updates)
            change += [(f'{name}.{key}', getattr(current, key), value)
                       for key, value in sorted(updates.items())
                       if getattr(current, key) != value]
        if not change:
            raise ValueError('derive changes nothing: use the object itself')
        return replace(self, **new, parent=self.identity(),
                       change=tuple(change), reason=str(reason))
