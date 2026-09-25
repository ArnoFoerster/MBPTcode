"""ONE entry point to a potential-energy surface, and the refusals that keep
two of them comparable.

A surface is two things that were never separated before: the PHYSICS it
declares -- which functional E_0 is, which state sits on it, which environment
it stands in (`src.Base.declaration.SurfacePhysics`) -- and the REALIZATION
that computes it: how chi0 is built, how the residues of the self-energy are
taken, which eigensolver runs, how the four-index integral is factorized, and
which orbitals carry an explicitly solved quasiparticle energy. Two surfaces of
the same physics and different realizations may be differenced; two of
different physics may not, whatever their realizations agree on.

WHAT THIS CURES. Two relaxations of the same molecule disagreed by up to 0.9 eV
on an adiabatic energy because one surface's E_0 was `mf.e_tot` and the other's
`mf.e_tot + (E_x^HF - E_xc) + E_c^dRPA`, and no object said which; two
incompatible quasiparticle windows ran under one name; `solver` had three
different defaults in three constructors; and an ISDF grid nobody validated
came in silently at 148 points per atom whenever `counts` was omitted. Each of
those is a realization choice that was never written down. Here every one of
them is resolved at construction and recorded on `surface.realization`, and a
combination that has no realizing class is refused by name rather than
approximated by the nearest one.

    surface = potential_energy_surface(mol, rhf, ground_state=GroundState('rpa', 'hf'),
                                       excitation=Excitation('singlet'))
    print(surface.describe())
    compare_surfaces(surface, other)     # PhysicsMismatch, or what differs

The call returns the realizing class's OWN instance -- `RPABSESurface`,
`ExcitedStateChain`, `DenseBSESurface` and the rest -- not a wrapper, so every
existing reach-through into a chain keeps working and `refreeze`, the gradient
and the property layer are untouched.
"""
import functools
import inspect
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

from src.Base.constants import OUTSIDE_TREATMENTS, SURFACE_GRID_ACCURACY
from src.Base.declaration import (ChargedExcitation, Excitation,
                                  PhysicsMismatch, QPStates, SurfacePhysics)
from src.Base.environment import environment_label, resolve_environment
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.utils.mpi_grid import lockstep_mean_field
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.SingleReference.GW.sum_over_poles import compressible
from src.SingleReference.LinearResponse.bse import solver_choice
from src.SingleReference.LinearResponse.rpa_energy import (declared_ground_state,
                                                           ground_state_energy)
from src.properties.optimize import MeanFieldSurface

#: How the polarizability is built. 'space-time' is the cubic imaginary-time
#: route, 'dense-qb' the quasi-boson route that holds the whole (pq|rs),
#: 'casida' the explicit particle-hole spectrum, 'imagfrequency' the
#: analytically continued imaginary-frequency route.
CHI0_ROUTES = ('casida', 'imagfrequency', 'space-time', 'dense-qb')

#: How the real-axis residues of Sigma^c are taken on the space-time route.
#: 'auto' picks per orbital and records what it picked; it is never a default,
#: because a surface whose residue backend is decided per geometry is not one
#: surface.
RESIDUE_ROUTES = ('explicit', 'laplace', 'sop', 'auto')

#: The Casida/BSE eigensolver. 'auto' is resolved through `solver_choice`.
SOLVERS = ('auto', 'dense', 'davidson')

#: How the four-index integral is represented.
FACTORIZATIONS = ('isdf', 'df', 'four-index')

#: Table key for "no state on E_0".
NO_EXCITATION = 'none'

#: Numeric keywords that name the ISDF interpolation grid and the basis it is
#: fitted in, `sliced`, how the factors are laid out over the ranks (each
#: rank its grid rows; the numbers are the whole layout's bit for bit), and
#: `fit` and `fit_block`, how the fit is realized ('rows': by grid rows, no
#: rank forming it whole, bitwise across rank counts and not the replicated
#: fit's bits). Shared by every space-time/ISDF row, because they all reach
#: the same `FrozenFactorization`.
GRID_NUMERICS = frozenset({'basis', 'auxbasis', 'counts', 'radii', 'n_start',
                           'grid_accuracy', 'sliced', 'fit', 'fit_block'})

#: Numeric keywords `ExcitedStateChain` reads: the two imaginary-time grids,
#: the contour-deformation quadrature, the pole model, the Casida step, and
#: `bse_adjoint`, how the root's adjoint is realized (the same surface either
#: way; 'grid' forms no three-index block).
EXCITED_NUMERICS = frozenset({'ntau_gw', 'ntau_w', 'nfreq_cd', 'n_poles',
                              'sop_stride', 'bse_conv_tol', 'degeneracy_tol',
                              'dense_max_nov', 'nroots', 'e_min_below_gap',
                              'cd_pole_resolution', 'tile_gb', 'bse_adjoint'})

#: Numeric keywords `RPAGroundStateChain` reads. Its imaginary-time count and
#: its frequency quadrature are named apart from the GW ones: they integrate
#: different integrands and a single `ntau` would silently tie them together.
RPA_GROUND_NUMERICS = frozenset({'ntau_rpa', 'nfreq_rpa', 'tile_gb'})

#: Combinations that have a class but cannot carry the declaration, with the
#: reason. A refusal with a reason is the point: silently building the nearest
#: row is what the declaration exists to stop.
DENSE_BSE_REFUSAL = (
    'the dense quasi-boson BSE surface puts E_HF + E_c^dRPA under every '
    'excitation (Toelle Eq. 15) and has no mean-field ground state to offer. '
    "Declare GroundState('rpa', xc) for it, or keep GroundState('dft', xc) "
    "and take chi0='space-time'.")

REFUSED_ROWS = {
    ('dft', 'Excitation', 'dense-qb', 'four-index'): DENSE_BSE_REFUSAL,
    ('dft', 'Excitation', 'dense-qb', 'df'): DENSE_BSE_REFUSAL,
}


@dataclass(frozen=True)
class Realization:
    """HOW a surface computes what its `SurfacePhysics` declares, resolved.

    Not one field here is a request: `solver` is the eigensolver that will run,
    `grid` the interpolation grid that was looked up, `qp_explicit` the orbital
    indices the declaration came out as at the reference spectrum. A field that
    a route has no answer for is None -- a mean-field surface has no residue
    backend, a dense one has no interpolation grid -- and None is a difference
    like any other.

    outside_treatment: what the orbitals NOT in `qp_explicit` carry on the BSE
        diagonal. 'scissor' is one frozen shift per state, calibrated at the
        reference geometry on the explicitly solved roots
        (`GW.qp_states.calibrate_scissor`); 'mean-field' is the bare
        eigenvalue. The two cubic rows take the scissor -- explicit inside the
        window, the frozen shift outside it -- and the dense quasi-boson rows
        record 'mean-field' because they have no scissor of their own. The two
        treatments are not the same surface, which is why `compare_surfaces`
        reports the field.

    How many ranks evaluated the surface is not a field: a surface carries no
    communicator, every rank reads the same numbers, and a record that needs
    the rank count takes it where the run was launched.
    """
    chi0: Optional[str]
    residues: Optional[str]
    solver: Optional[str]
    factorization: Optional[str]
    grid: Optional[str]
    qp_states: Optional[QPStates]
    qp_explicit: Optional[Tuple[int, ...]]
    outside_treatment: Optional[str]
    realizing_class: str

    def __post_init__(self) -> None:
        if self.outside_treatment not in OUTSIDE_TREATMENTS + (None,):
            raise ValueError(
                f'outside_treatment={self.outside_treatment!r} not in '
                f'{OUTSIDE_TREATMENTS}: what the orbitals outside the '
                f'quasiparticle set carry is part of the surface, so a row may '
                f'not invent a third answer for it.')


@dataclass(frozen=True)
class Row:
    """One line of the dispatch table.

    The first four fields are the key -- the physics kind, the state's type,
    and the two realization axes that decide which class can compute it. The
    rest is what that class needs: the numeric keywords it READS (anything else
    is a TypeError naming both), the builder that maps this entry point's
    vocabulary onto its constructor, and the realization fields it fixes.
    """
    ground_state: str
    excitation: str
    chi0: str
    factorization: str
    cls: type
    numerics: frozenset
    build: object
    qp_keyword: Optional[str] = None
    outside_treatment: Optional[str] = None
    reads_solver: bool = False
    reads_residues: bool = False
    isdf_grid: bool = False
    #: The only eigensolver this row has, where the class offers no choice.
    fixed_solver: Optional[str] = None
    #: False for a surface with no response function at all, whose chi0 and
    #: factorization are recorded as None rather than as what was asked for.
    post_scf: bool = True

    def key(self):
        """The four-part table key, with '*' where the row takes any value."""
        return (self.ground_state, self.excitation, self.chi0,
                self.factorization)


@dataclass(frozen=True)
class Setup:
    """Everything a builder needs, with every realization choice already resolved."""
    mol: object
    scf_factory: object
    mf: object
    environment: object
    excitation: object
    numerics: dict
    counts: Optional[dict]
    n_start: Optional[int]
    qp_explicit: Optional[Tuple[int, ...]]
    solver: Optional[str]
    residues: Optional[str]


@functools.lru_cache(maxsize=1)
def realizing_classes():
    """The classes the table dispatches to.

    cycle: src.gradients.excited_state imports src.properties.nonadiabatic, so
    src.properties cannot import src.gradients at module scope.
    """
    from src.gradients.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                              QuasiparticleSurface)
    from src.gradients.excited_state import ExcitedStateChain
    from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
    from src.gradients.rpa_ground_state import RPAGroundStateChain
    return {'DenseBSESurface': DenseBSESurface,
            'DenseRPASurface': DenseRPASurface,
            'ExcitedStateChain': ExcitedStateChain,
            'MeanFieldSurface': MeanFieldSurface,
            'QuasiparticleSurface': QuasiparticleSurface,
            'RPABSESurface': RPABSESurface,
            'RPAGroundStateChain': RPAGroundStateChain,
            'RPAQPSurface': RPAQPSurface}


def qp_set_degeneracy_tol(numerics):
    """The orbital-energy separation two orbitals count as one degenerate block at.

    Read off `ExcitedStateChain`'s own default rather than respelled, so the
    set this entry point resolves and the set the chain would have chosen
    cannot be two different sets.
    """
    if 'degeneracy_tol' in numerics:
        return float(numerics['degeneracy_tol'])
    chain = realizing_classes()['ExcitedStateChain']
    return float(inspect.signature(chain.__init__)
                 .parameters['degeneracy_tol'].default)


def _shell_counts(counts):
    """A counts mapping as plain integers in shell order, for comparing two of them."""
    counts = dict(counts)
    return tuple((name, int(counts[name])) for name in sorted(counts))


def _grid_label(level, counts, n_start, radii):
    """The printable resolved grid: the level, its shell counts and its recipe."""
    if radii is not None:
        return 'radii given'
    if counts is None:
        return None
    shells = ','.join(f'{name}={n}' for name, n in _shell_counts(counts))
    named = '' if level is None else f'{str(level).upper()} '
    return f'{named}{shells} (n_start={1 if n_start is None else int(n_start)})'


def resolve_grid(mol, numerics):
    """(counts, n_start, label) for the ISDF rows: the grid this surface fits on.

    With nothing asked for the level is `SURFACE_GRID_ACCURACY`, resolved
    through `resolve_isdf_grid`, which refuses a basis or an element the shipped
    radii table has not got. THE 148-POINT DEFAULT IS NEVER REACHED FROM HERE:
    a grid nobody optimized is a different factorization, not a coarse one, and
    it arrived silently whenever `counts` was omitted.

    An explicit `counts`, `radii` or `grid_accuracy` is honoured as given, and
    a `counts` that contradicts a named level is refused rather than resolved
    to either side -- `separable_factors`' rule, in the same words.
    """
    basis = numerics.get('basis') or str(mol.basis)
    auxbasis = numerics.get('auxbasis') or (basis + '-ri')
    elements = {mol.atom_pure_symbol(i) for i in range(mol.natm)}
    counts, n_start = numerics.get('counts'), numerics.get('n_start')
    radii, level = numerics.get('radii'), numerics.get('grid_accuracy')
    if level is None and counts is None and radii is None:
        level = SURFACE_GRID_ACCURACY
    if level is not None:
        level_counts, level_n_start = resolve_isdf_grid(level, basis, elements,
                                                        auxbasis=auxbasis)
        if counts is not None and _shell_counts(counts) != _shell_counts(level_counts):
            raise ValueError(
                f'two grids asked for: counts {dict(sorted(dict(counts).items()))} '
                f'and grid_accuracy {level!r}, which is '
                f'{dict(sorted(level_counts.items()))} at {basis}. Pass one or '
                f'the other -- the level is not a hint that a count may '
                f'override, and every energy built on the grid not asked for '
                f'would be of a functional nobody requested.')
        counts = level_counts
        n_start = level_n_start if n_start is None else n_start
    return counts, n_start, _grid_label(level, counts, n_start, radii)


def factorization_kwargs(setup):
    """The basis, auxiliary basis, interpolation grid, factor layout and fit
    realization every ISDF row shares."""
    kw = {}
    for name in ('basis', 'auxbasis', 'radii', 'sliced', 'fit', 'fit_block'):
        if name in setup.numerics:
            kw[name] = setup.numerics[name]
    if setup.counts is not None:
        kw['counts'] = setup.counts
    if setup.n_start is not None:
        kw['n_start'] = setup.n_start
    return kw


def excited_kwargs(setup, row):
    """The keywords `ExcitedStateChain` takes this entry point's realization through.

    The explicit quasiparticle set goes in as `qp_window`, which accepts a
    sequence of orbital indices; `scissor='calibrate'` is what makes the states
    INSIDE that set whose pole model is inadmissible take a frozen shift built
    at the reference geometry instead of a real-axis solve, and `outside`
    decides what the orbitals outside it carry -- the row's own
    `outside_treatment`, so the record and the chain cannot disagree.
    """
    kw = {'residue_route': setup.residues}
    # A row with no Casida eigenproblem -- the charged surface -- resolves no
    # solver, and handing the chain None is not the same as not asking.
    if setup.solver is not None:
        kw['solver'] = setup.solver
    if row.qp_keyword is not None:
        kw['qp_window'] = setup.qp_explicit
        kw['scissor'] = 'calibrate'
    if row.outside_treatment == 'scissor':
        kw['outside'] = 'scissor'
    for name in sorted(EXCITED_NUMERICS):
        if name in setup.numerics:
            kw[name] = setup.numerics[name]
    return kw


def excited_numerics(chain):
    """The grids and caps an `ExcitedStateChain` resolved, as numbers, and
    the realization of its adjoint."""
    return {'ntau_gw': int(chain.ntau_gw), 'ntau_w': int(chain.ntau_w),
            'nfreq_cd': int(chain.nfreq_cd), 'n_poles': int(chain.n_poles),
            'sop_stride': chain.sop_stride, 'nroots': int(chain.nroots),
            'dense_max_nov': int(chain.dense_max_nov),
            'bse_conv_tol': float(chain.bse_conv_tol),
            'degeneracy_tol': float(chain.degeneracy_tol),
            'cd_pole_resolution': float(chain.cd_pole_resolution),
            'bse_adjoint': chain.bse_adjoint}


def build_rpa_ground(row, setup):
    """E_HF + E_c^dRPA on the space-time/ISDF chain."""
    kw = factorization_kwargs(setup)
    for named, own in (('ntau_rpa', 'ntau'), ('nfreq_rpa', 'nfreq'),
                       ('tile_gb', 'tile_gb')):
        if named in setup.numerics:
            kw[own] = setup.numerics[named]
    chain = row.cls(setup.mol, setup.scf_factory, mf=setup.mf,
                    environment=setup.environment, **kw)
    return chain, {'ntau_rpa': int(chain.ntau),
                   'nfreq_rpa': int(len(chain.grid.omega_points))}


def build_rpa_bse(row, setup):
    """E_HF + E_c^dRPA + Omega_nu: the dRPA ground state and the BSE@GW state on it."""
    surface = row.cls(setup.mol, setup.scf_factory, mf=setup.mf,
                      environment=setup.environment,
                      state=setup.excitation.root - 1,
                      spin=setup.excitation.spin,
                      bse_tda=setup.excitation.kernel == 'bse-tda',
                      **factorization_kwargs(setup), **excited_kwargs(setup, row))
    return surface, excited_numerics(surface.excited)


def build_rpa_qp(row, setup):
    """E_HF + E_c^dRPA -/+ eps^QP_p, the charged state on the dRPA ground state."""
    nocc = setup.mol.nelectron // 2
    surface = row.cls(setup.mol, setup.scf_factory, mf=setup.mf,
                      environment=setup.environment,
                      state=setup.excitation.orbital - (nocc - 1),
                      **factorization_kwargs(setup), **excited_kwargs(setup, row))
    return surface, excited_numerics(surface.excited)


def build_excited_chain(row, setup):
    """E_KS + Omega: the mean field's own ground state with a BSE@GW state on it."""
    chain = row.cls(setup.mol, setup.scf_factory, mf=setup.mf,
                    environment=setup.environment,
                    state=setup.excitation.root - 1,
                    spin=setup.excitation.spin,
                    bse_tda=setup.excitation.kernel == 'bse-tda',
                    **factorization_kwargs(setup), **excited_kwargs(setup, row))
    return chain, excited_numerics(chain)


def build_dense_rpa(row, setup):
    """E_HF + E_c^dRPA on the dense quasi-boson route. It holds no settings."""
    return row.cls(setup.mol, setup.scf_factory), {}


def build_dense_bse(row, setup):
    """E_HF + E_c^dRPA + Omega_nu on the dense quasi-boson route.

    `filter_z=False` keeps exactly the set this entry point resolved: the
    surface's own Z > 0.5 filter would re-decide it and the two routes would
    then be compared on two different quasiparticle sets.
    """
    variant = 'BSEtda@GW' if setup.excitation.kernel == 'bse-tda' else 'BSE@GW'
    kw = {} if 'eri_blocks' not in setup.numerics else {
        'eri_blocks': setup.numerics['eri_blocks']}
    if 'auxbasis' in setup.numerics:
        kw['auxbasis'] = setup.numerics['auxbasis']
    surface = row.cls(setup.mol, variant, setup.excitation.root - 1,
                      qp_orbs=setup.qp_explicit, filter_z=False,
                      scf=setup.scf_factory, spin=setup.excitation.spin, **kw)
    return surface, {}


def build_dense_qp(row, setup):
    """E_HF + E_c^dRPA -/+ eps^QP_p on the dense quasi-boson route."""
    surface = row.cls(setup.mol, setup.scf_factory,
                      charge_change=setup.excitation.charge_change,
                      screening='rpa', orbital=setup.excitation.orbital)
    return surface, {}


def build_mean_field(row, setup):
    """E_KS: the mean field's own energy, with no post-SCF step at all.

    The reference mean field travels with it so that reading the declaration
    off the surface does not converge a second SCF.
    """
    return row.cls(setup.mol, setup.scf_factory, mf=setup.mf), {}


@functools.lru_cache(maxsize=1)
def dispatch_table():
    """(ground-state kind, state type, chi0, factorization) -> the class that realizes it.

    Every row is a combination someone has a class for. A combination that is
    not here is refused by name: the nearest row is a different functional or a
    different approximation, and substituting one is how two relaxations ended
    up 0.9 eV apart under one label.
    """
    cls = realizing_classes()
    return (
        Row('rpa', NO_EXCITATION, 'space-time', 'isdf',
            cls['RPAGroundStateChain'], GRID_NUMERICS | RPA_GROUND_NUMERICS,
            build_rpa_ground, isdf_grid=True),
        Row('rpa', NO_EXCITATION, 'dense-qb', 'four-index',
            cls['DenseRPASurface'], frozenset(), build_dense_rpa),
        Row('rpa', NO_EXCITATION, 'dense-qb', 'df',
            cls['DenseRPASurface'], frozenset(), build_dense_rpa),
        Row('rpa', 'Excitation', 'space-time', 'isdf',
            cls['RPABSESurface'], GRID_NUMERICS | EXCITED_NUMERICS,
            build_rpa_bse, qp_keyword='qp_window', outside_treatment='scissor',
            reads_solver=True, reads_residues=True, isdf_grid=True),
        Row('rpa', 'Excitation', 'dense-qb', 'four-index',
            cls['DenseBSESurface'], frozenset({'eri_blocks'}), build_dense_bse,
            qp_keyword='qp_orbs', outside_treatment='mean-field',
            fixed_solver='dense'),
        Row('rpa', 'Excitation', 'dense-qb', 'df',
            cls['DenseBSESurface'], frozenset({'eri_blocks', 'auxbasis'}),
            build_dense_bse, qp_keyword='qp_orbs',
            outside_treatment='mean-field', fixed_solver='dense'),
        Row('rpa', 'ChargedExcitation', 'space-time', 'isdf',
            cls['RPAQPSurface'], GRID_NUMERICS | EXCITED_NUMERICS,
            build_rpa_qp, outside_treatment='mean-field', reads_residues=True,
            isdf_grid=True),
        Row('rpa', 'ChargedExcitation', 'dense-qb', 'four-index',
            cls['QuasiparticleSurface'], frozenset(), build_dense_qp,
            outside_treatment='mean-field'),
        Row('rpa', 'ChargedExcitation', 'dense-qb', 'df',
            cls['QuasiparticleSurface'], frozenset(), build_dense_qp,
            outside_treatment='mean-field'),
        Row('dft', NO_EXCITATION, '*', '*',
            cls['MeanFieldSurface'], frozenset(), build_mean_field,
            post_scf=False),
        Row('dft', 'Excitation', 'space-time', 'isdf',
            cls['ExcitedStateChain'], GRID_NUMERICS | EXCITED_NUMERICS,
            build_excited_chain, qp_keyword='qp_window',
            outside_treatment='scissor', reads_solver=True,
            reads_residues=True, isdf_grid=True),
    )


def excitation_key(excitation):
    """The table's name for the state on E_0."""
    if excitation is None:
        return NO_EXCITATION
    return type(excitation).__name__


def resolved_factorization(row, factorization):
    """The representation of (pq|rs) the row actually builds.

    `DenseRPASurface` and `QuasiparticleSurface` hold the exact tensor and take
    no auxiliary basis, so 'df' resolves to 'four-index' there rather than
    reporting a fit that never happened; a mean-field surface has no
    post-SCF integral to represent at all.
    """
    if not row.post_scf:
        return None
    if row.cls.__name__ in ('DenseRPASurface', 'QuasiparticleSurface'):
        return 'four-index'
    return factorization


def find_row(ground_state, excitation, chi0, factorization):
    """The one row that realizes this combination, or a ValueError listing them all."""
    key = (ground_state.kind, excitation_key(excitation), chi0, factorization)
    for row in dispatch_table():
        if all(b in (a, '*') for a, b in zip(key, row.key())):
            return row
    refusal = REFUSED_ROWS.get(key)
    if refusal is not None:
        raise ValueError(f'{key} is refused: {refusal}')
    rows = '\n'.join(f'    {r.key()} -> {r.cls.__name__}'
                     for r in dispatch_table())
    raise ValueError(
        f'no surface realizes {key} (ground state, state, chi0, '
        f'factorization). The combinations that have a class:\n{rows}')


def refuse_missing_adjoints(excitation, chi0):
    """evGW and the imaginary-frequency chi0 have no adjoint, and are refused here.

    Neither is answered with a finite difference: a gradient that quietly costs
    6N energies and carries the step size as an error is a different object
    from the analytic one, and nothing downstream would say which it held.
    """
    if isinstance(excitation, Excitation) and excitation.qp == 'evgw':
        raise NotImplementedError(
            "Excitation(qp='evgw') has no adjoint: the eigenvalue-self-"
            'consistent quasiparticle energies re-enter the screening, and no '
            'reverse pass in src/gradients differentiates that fixed point. '
            "Only qp='g0w0' has one.")
    if chi0 == 'imagfrequency':
        raise NotImplementedError(
            "chi0='imagfrequency' has no adjoint: the reverse pass in "
            'src/gradients/space_time_adjoint.py differentiates the '
            'imaginary-TIME polarizability, and the analytically continued '
            "imaginary-frequency chi0 has none. Use chi0='space-time'.")


def refuse_residues_outside_validity(residues, states, eps, nocc):
    """A pole model is only defined where Eq. (27) admits the state.

    `compressible` is the wall both 'sop' and 'laplace' stand behind: a state
    within one particle-hole gap of the frontier has a compressible self-energy
    AND a cubic residue route, a deeper one has neither. Refusing names the
    state and its reach instead of falling back to the explicit O(N^4) build,
    which would make the surface's cost and its error a function of the
    geometry.
    """
    if residues not in ('laplace', 'sop'):
        return
    eps = np.asarray(eps, float)
    out = [(p, compressible(float(eps[p]), eps, nocc)[1]) for p in states]
    worst = [(p, r) for p, r in out if r >= 1.0]
    if not worst:
        return
    named = ', '.join(f'orbital {p} at reach {r:.2f}' for p, r in sorted(
        worst, key=lambda pr: -pr[1]))
    raise ValueError(
        f"residues={residues!r} does not reach {named}: the compression "
        f'condition |omega_p - eps_q| < E_g is violated (reach >= 1 counts the '
        f'worst swept pole in units of the particle-hole gap), so no pole '
        f'model represents this state at any number of poles. Take '
        f"residues='explicit', or declare a state the model covers.")


def refuse_foreign_numerics(row, numerics):
    """A numeric keyword the realizing class does not read is a TypeError naming both."""
    foreign = sorted(set(numerics) - set(row.numerics))
    if foreign:
        accepted = ', '.join(sorted(row.numerics)) or 'none'
        raise TypeError(
            f'{row.cls.__name__} does not read {", ".join(foreign)}: the '
            f'realization for {row.key()} accepts {accepted}. A keyword a '
            f'realization ignores is a setting the caller believes is in '
            f'force and is not.')


def refuse_undeclared_state(excitation, mol):
    """Refuse a state no realization can honour as declared."""
    if isinstance(excitation, Excitation) and excitation.irrep is not None:
        raise NotImplementedError(
            f'Excitation(irrep={excitation.irrep!r}): no surface here selects '
            f'a root by irreducible representation; they follow a root by '
            f'index or by overlap. Select the irrep from the spectrum with '
            f'src.properties.characters and declare its root.')
    if isinstance(excitation, ChargedExcitation):
        nocc = mol.nelectron // 2
        occupied = excitation.orbital < nocc
        if occupied != (excitation.charge_change == -1):
            raise ValueError(
                f'ChargedExcitation(orbital={excitation.orbital}, '
                f'charge_change={excitation.charge_change}) disagrees with the '
                f'reference occupation: orbital {excitation.orbital} is '
                f'{"occupied" if occupied else "virtual"} at nocc={nocc}, so '
                f'the process is an '
                f'{"ionization (charge_change=-1)" if occupied else "attachment (charge_change=+1)"}.')


def declare_physics(surface, physics):
    """CHECK the realizing class's own declaration, or stamp one where it has none.

    A class that says what it computes is not LABELLED by this entry point: the
    two declarations are compared, and a disagreement means the dispatcher and
    the class describe different surfaces. An `Excitation(kernel='bse-tda')`
    answered by a chain that solves the full kernel would otherwise come back
    under the declared name carrying the other approximation's number -- the
    same disease as the two E_0 conventions, one level down.

    A class with no declaration of its own keeps the stamp, which is what makes
    `compare_surfaces` and `describe` work for it at all.
    """
    if getattr(type(surface), 'physics', None) is None:
        surface.physics = physics
        return
    own = surface.physics
    if own != physics:
        raise ValueError(
            f'{type(surface).__name__} declares {own.label()!r} with state '
            f'{own.excitation!r} in {own.environment}, and '
            f'{physics.label()!r} with state {physics.excitation!r} in '
            f'{physics.environment} was asked for. The realizing class and the '
            f'declaration describe different surfaces, so the number would be '
            f"one surface's under the other's name.")


def resolve_solver(solver, row, mf, excitation):
    """The eigensolver that will run, resolving 'auto' through `solver_choice`.

    The rule is the repository's one rule -- dense while the Casida pair (A, B)
    fits in BSE_DENSE_MAX_GB -- rather than a third constructor default. A row
    with no eigenproblem records None and refuses to be told one.
    """
    if solver not in SOLVERS:
        raise ValueError(f'solver {solver!r} not in {SOLVERS}')
    if not row.reads_solver:
        if solver != 'auto':
            raise TypeError(
                f'{row.cls.__name__} does not read solver={solver!r}: the '
                f'realization for {row.key()} has no Casida eigenproblem to '
                f'choose a solver for.')
        return None
    eps = np.asarray(mf.mo_energy, float)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    n_ov = nocc * (len(eps) - nocc)
    tda = isinstance(excitation, Excitation) and excitation.kernel == 'bse-tda'
    return solver_choice(n_ov, tda) if solver == 'auto' else solver


def resolve_residues(residues, row):
    """The residue backend the row will run, or None where it has none."""
    if residues not in RESIDUE_ROUTES:
        raise ValueError(f'residues {residues!r} not in {RESIDUE_ROUTES}')
    if row.reads_residues:
        return residues
    if residues != 'explicit':
        raise TypeError(
            f'{row.cls.__name__} does not read residues={residues!r}: the '
            f'realization for {row.key()} takes no real-axis residues of the '
            f'self-energy.')
    return None


def resolve_states(qp_states, row, mol, mf, numerics, excitation):
    """(the spec the row honours, its explicit orbitals) at the reference spectrum.

    The set is decided ONCE, here, on the reference mean field, and handed to
    the class as indices -- so two routes that resolve the same declaration
    cannot be running two different windows under one name. A row that has no
    set (a ground state, a single charged state) refuses a declaration it
    cannot honour rather than dropping it.
    """
    if row.qp_keyword is None:
        if qp_states != QPStates():
            raise TypeError(
                f'{row.cls.__name__} does not read qp_states={qp_states!r}: '
                f'the realization for {row.key()} has no quasiparticle SET -- '
                f'it solves the one orbital it is given, or none at all.')
        if isinstance(excitation, ChargedExcitation):
            return None, (int(excitation.orbital),)
        return None, None
    eps = np.asarray(mf.mo_energy, float)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    resolved = resolve_qp_states(qp_states, eps, nocc, mol=mol,
                                 degeneracy_tol=qp_set_degeneracy_tol(numerics))
    return qp_states, resolved.explicit


def reference_mean_field(mol, scf_factory, environment):
    """The mean field every resolution above reads, built once, in the environment.

    A surface standing in an environment has the environment's ground state for
    its mean field, and the quasiparticle set, the pair count and the declared
    functional all follow from that spectrum rather than from the raw factory's.

    IT IS READ BEFORE ANY SURFACE EXISTS -- the solver, the quasiparticle set
    and the declared functional all come off this spectrum -- so a factory that
    hands back a mean field it has BUILT AND NOT RUN has to be converged here
    as well as inside the chain, and over the same ranks
    (`converged_factory`); under ranks its spectrum is rank 0's before any of
    those is decided from it, since a factory that converges its own does so
    on each rank alone. Serially, and for a factory that converges its own,
    this is the call the factory would have made.
    """
    # cycle: src.gradients.excited_state imports src.properties.nonadiabatic
    from src.gradients.factor_chain import converged_factory

    mf = resolve_environment(environment, None).mean_field(
        mol, converged_factory(scf_factory))
    return lockstep_mean_field(mf)


def refuse_undeclared_functional(ground_state, mf):
    """Refuse a declaration the reference mean field does not carry.

    `GroundState('rpa', 'pbe0')` on a Hartree-Fock factory is a surface nobody
    can build: the double-counting term (E_x^HF - E_xc)[rho] is a property of
    the reference, so the declaration and the number would describe two
    different functionals.
    """
    carried = declared_ground_state(mf, ground_state.kind)
    if carried != ground_state:
        raise ValueError(
            f'{ground_state.label()} was declared, but the reference mean '
            f'field is {carried.xc.upper()} and carries {carried.label()}. '
            f'The exchange-correlation double counting is a property of the '
            f'reference, so the declaration and the number would be two '
            f'different functionals.')


def correlation_energy(surface):
    """E_c^dRPA as the realizing class computes it, at the reference geometry.

    The cubic chains report it directly; the dense quasi-boson surfaces read it
    off the RPA block they already hold, which is the same trace formula on the
    same reference.
    """
    mol, mf = surface.mean_field()
    chain = getattr(surface, 'ground', surface)
    reported = getattr(chain, 'correlation_energy', None)
    if reported is not None:
        return float(reported(mol, mf))
    solve = getattr(surface, '_solve', None)
    if solve is not None:
        return float(solve(mol, mf)[0].qp.qb.e_corr())
    return float(surface._build(mol, mf)[3].qb.e_corr())


def ground_state_terms(surface):
    """E_0 and its additive terms at the reference geometry, through the ONE assembly.

    `ground_state_energy` is that assembly: it takes the declaration and the
    mean field and reports E_ref, the exact-exchange double counting and
    E_c^dRPA separately, because an E_0 given as one number cannot be checked
    against another route's.
    """
    mol, mf = surface.mean_field()
    ground = surface.physics.ground_state
    if ground.kind == 'dft':
        return ground_state_energy(ground, mf, mol)
    return ground_state_energy(ground, mf, mol,
                               e_corr=correlation_energy(surface))


def describe(surface):
    """What this surface computes, what E_0 is worth at its geometry, and how.

    The E_0 terms are evaluated, not named: the disease this entry point cures
    was two surfaces whose E_0 differed by E_c^dRPA with nothing printing
    either number.
    """
    physics = surface.physics
    e0 = ground_state_terms(surface)
    lines = [physics.label(),
             f'E_0 = {physics.ground_state.label()}',
             f'    {"total":<22s} {e0.total:.12f} Ha']
    for name in physics.ground_state.terms():
        lines.append(f'    {name:<22s} {e0.terms[name]:.12f} Ha')
    lines.append(f'state: {physics.excitation!r}')
    lines.append(f'environment: {physics.environment}')
    lines.append('realization:')
    for field in fields(Realization):
        lines.append(f'    {field.name:<22s} '
                     f'{getattr(surface.realization, field.name)!r}')
    lines.append('numerics:')
    for name in sorted(surface.numerics):
        lines.append(f'    {name:<22s} {surface.numerics[name]!r}')
    return '\n'.join(lines)


def potential_energy_surface(mol, scf_factory, *, ground_state, excitation=None,
                             environment=None, chi0='space-time',
                             residues='explicit', solver='auto',
                             factorization='isdf', qp_states=QPStates(),
                             **numerics):
    """The one way to build a potential-energy surface with its physics declared.

    ground_state/excitation/environment are the PHYSICS
        (`src.Base.declaration`): which functional E_0 is, which state sits on
        it, what it stands in. Two surfaces may be differenced only if these
        agree up to the excitation -- which is what a gap or an IP is.
    chi0/residues/solver/factorization/qp_states are the REALIZATION: how the
        polarizability is built, how the self-energy's real-axis residues are
        taken, which eigensolver runs, how (pq|rs) is represented, and which
        orbitals carry an explicitly solved quasiparticle energy.
    **numerics are the grids, caps and tolerances the chosen realization reads.
        One it does not read is a TypeError naming both, because a keyword that
        is silently dropped is a setting the caller believes is in force.

    Returns the realizing class's own instance, carrying `physics`,
    `realization`, `numerics` and `describe()`. It carries no communicator:
    inside `with distributed(comm):` every rank builds it and the kernels it
    reaches divide their sweeps over the ranks.
    """
    if chi0 not in CHI0_ROUTES:
        raise ValueError(f'chi0 {chi0!r} not in {CHI0_ROUTES}')
    if factorization not in FACTORIZATIONS:
        raise ValueError(f'factorization {factorization!r} not in {FACTORIZATIONS}')
    physics = SurfacePhysics(ground_state, excitation,
                             environment_label(environment))
    refuse_missing_adjoints(excitation, chi0)
    refuse_undeclared_state(excitation, mol)
    row = find_row(ground_state, excitation, chi0, factorization)
    refuse_foreign_numerics(row, numerics)
    mf = reference_mean_field(mol, scf_factory, environment)
    refuse_undeclared_functional(ground_state, mf)
    used_solver = resolve_solver(solver, row, mf, excitation)
    used_residues = resolve_residues(residues, row)
    spec, explicit = resolve_states(qp_states, row, mol, mf, numerics, excitation)
    refuse_residues_outside_validity(used_residues, explicit or (),
                                     mf.mo_energy,
                                     int(np.count_nonzero(np.asarray(mf.mo_occ) > 0)))
    counts, n_start, grid = (resolve_grid(mol, numerics) if row.isdf_grid
                             else (None, None, None))
    setup = Setup(mol, scf_factory, mf, environment, excitation, numerics,
                  counts, n_start, explicit, used_solver, used_residues)
    surface, resolved = row.build(row, setup)
    declare_physics(surface, physics)
    surface.realization = Realization(
        chi0=chi0 if row.post_scf else None, residues=used_residues,
        solver=row.fixed_solver or used_solver,
        factorization=resolved_factorization(row, factorization),
        grid=grid, qp_states=spec, qp_explicit=explicit,
        outside_treatment=row.outside_treatment,
        realizing_class=type(surface).__name__)
    surface.numerics = resolved
    surface.describe = functools.partial(describe, surface)
    return surface


def compare_surfaces(a, b):
    """What two surfaces share and what they do not, or a `PhysicsMismatch`.

    A differing EXCITATION is allowed and is the whole point: a gap and an
    ionization potential are differences of two states on one ground state. A
    differing ground-state functional or environment is not, and raises --
    E_c^dRPA is 6.3 eV on water/cc-pVDZ and does not cancel out of a difference
    that only one side carries.

    The realization differences are REPORTED, not refused: a dense oracle and a
    cubic production route are meant to be differenced, and the list is what
    says which of the two numbers to trust where they disagree.
    """
    for name, surface in (('a', a), ('b', b)):
        if not hasattr(surface, 'physics'):
            raise TypeError(
                f'{name} carries no declared physics: it was not built through '
                f'potential_energy_surface, so what its E_0 is cannot be read '
                f'off it and the comparison would be a guess.')
    if not a.physics.comparable_with(b.physics):
        raise PhysicsMismatch(a.physics, b.physics)
    names = [field.name for field in fields(Realization)]
    side_by_side = {name: (getattr(a.realization, name),
                           getattr(b.realization, name)) for name in names}
    return {'physics': (a.physics.label(), b.physics.label()),
            'ground_state': a.physics.ground_state,
            'environment': a.physics.environment,
            'excitation': (a.physics.excitation, b.physics.excitation),
            'realization': side_by_side,
            'realization_differs': [name for name in names
                                    if side_by_side[name][0] != side_by_side[name][1]]}
