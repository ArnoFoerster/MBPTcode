"""The excited-state potential-energy surface of the cubic BSE@GW gradient.

The gradient chain in `qp_space_time`/`bse_isdf`/`isdf_derivatives` returns
dOmega/dR, the derivative of an EXCITATION energy. A geometry optimization
needs the derivative of the excited state's TOTAL energy,

    E_ex(R) = E_0(R) + Omega(R),     dE_ex/dR = dE_0/dR + dOmega/dR

with E_0 the mean-field ground state, so the ground-state gradient is added
here and never inside the chain. That split matters for what comes after it:
in an ADIABATIC singlet-triplet gap the two states sit at DIFFERENT minima, so
E_0 does not cancel between them and has to be carried consistently.

THE FROZEN CONVENTIONS. The chain fixes its discrete choices at the
reference geometry -- the quasiparticle set, the Newton branch with the pole
guard it steps around and the seed it starts from, the pole model's
positions, the contour-deformation quadrature, the residue backend, the
degenerate blocks and quasiparticle-Hamiltonian groups, and the scissor the
orbitals OUTSIDE the set carry -- because each is a discontinuity in the
surface otherwise. They live in ONE object, `self.conventions`
(`gradients.conventions.FrozenConventions`), keyed by orbital identity: each
forward resolves them at its geometry by orbital overlap, a near-degenerate
group by its span, and refuses where an orbital has left its unit; the
kernels read them as `self.resolved`, in that geometry's indices. The frame
orientation and the interpolation pair layout are the factorization's
(`FrozenFactorization`). `refreeze` rebuilds them at a new geometry with the
same settings, which is how the optimizer in `src/properties/optimize.py`
measures the drift of the surface it walked from the one the fit would
choose at the end.

THE SURROUNDINGS enter through the chain's `environment`
(`src.environments.environment`): a polarizable continuum dresses the auxiliary gauge,
and the static reaction field that dressing implies (Eq. (18)) enters the
quasiparticle equation; fixed point charges enter the mean field; the gas
phase does nothing. The energy sees the whole of an environment or none of
it. The force takes the adjoint of the dressing from the environment and
carries Eq. (18) through the factors it is built from, so a solvated force
is analytic, for a restricted reference and a molecule. An environment
without a derivative says so through `differentiable`, and the chain then
reports energies and refuses forces rather than returning the gas-phase
adjoints, which would be wrong by the whole reaction-field response while
looking reasonable.

THE CASIDA STEP is dense while the (A, B) pair fits in BSE_DENSE_MAX_GB --
every root, in a fixed order, the only form a finite-difference check should
difference -- and production's ISDF Davidson (`solve_casida_davidson`) above
it, for either spin and either kernel (`bse_tda`: a Hermitian Davidson on
the action's A half, Y = 0). The adjoint is matrix-free either way, on the
static W the Davidson solved with; 'explicit' contracts the eigenvectors
against the three-index blocks of `bse_cache`, built at the first reverse
call, so an energy never pays for them; 'grid' (`isdf_bse_adjoint`) reverses
the ISDF block action and forms no three-index block. The two agree to
rounding, not in the bits.

SLICED FACTORS (`sliced=True`, over more than one rank) reach every stage
as ONE `SlicedFactors`: the Davidson block action reads its rows, and the
static W, the quasiparticle solve, the BSE cache and adjoint, the chi0 adjoint
and the nuclear assembly each gather what they read whole once and drop it,
so between geometries a rank holds its grid rows alone and every number is
the whole layout's bit for bit; in a continuum the bare gauge Eq. (18)
needs beside the dressed one is laid out in the same rows (`gauge_factor`).
On the row fit (`fit='rows'`) the same stages read the rows the
fit itself built, never a whole fit: bitwise the same at every rank count,
and another realization -- and, where the pair screen drops a pair, another
estimator -- than the replicated fit's (`FrozenFactorization`).

ONE FORWARD, SEVERAL STATES. Everything `forward` computes before the Casida
step -- the factors, the static screening, the reaction field, the
quasiparticle set and its tape, the outside scissor -- reads neither the spin
nor the root (`shared_forward`, a `SharedForward`); `solve_casida` solves one
spin on it (a `CasidaSolution`), and `root_gradient`, `interstate_gradient_of`
and `fold_to_nuclei` carry a seed on the two back to the nuclei. `spin_view` is a shallow
copy of a chain whose conventions are frozen, with another spin and root, so
a singlet and a triplet, or several roots, are solved and differentiated on
the same objects (`src.gradients.gwbse.state_manifold.StateManifold.evaluate`).

THE ADAPTIVE EXPLICIT SET (`qp_select=QPStates('adaptive')`). `qp_window` is
then the admitted window, the candidates. At the first forward one
analytic-continuation GW over the window (`qp_selection`) decides which of
them are solved explicitly; every other candidate (a HOLE) borrows the
explicit shift of the solved orbital its frozen tier map names. The
continuation SELECTS only: the energy and the force read the explicit roots
and the frozen map, never an AC number, so the force is the adjoint of the
explicit set exactly as for any other set. The partition is checked against
the production eigenvectors at the reference geometry and frozen for the
walk (`qp_partition`); `refreeze` selects it again at the new geometry,
GROW-ONLY: the selection there starts from the set the last surface solved
(`qp_floor`), so a refreeze pass may add states to the explicit set and
never drops one (`qp_growth` names what each pass carried and added).

Everything computed FROM this surface -- geometry optimization, normal modes,
Huang-Rhys factors, adiabatic gaps, reorganization energies, rates -- lives in
`src/properties/` and knows only the `PotentialEnergySurface` protocol.
"""
import copy
import warnings
from dataclasses import replace

import numpy as np

from src.Base.numerics.constants import KAPPA, CD_NFREQ
from src.gradients.rpa.chain_settings import CarriedGrids, chain_setting
from src.Base.numerics.numerics import (numerics_identity, require_same_identity,
                                        stamp_of)
from src.Base.declaration import Excitation, SurfacePhysics
from src.environments.environment import environment_label
from src.Base.environment_seam import attached_environment
from src.Base.utils.record_io import record_hash
from src.Base.geometry.geometry import molecule_at
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.LinearResponse.casida import solver_choice
from src.SingleReference.LinearResponse.davidson import DavidsonWarmStart
from src.SingleReference.LinearResponse.rpa_energy import (declared_ground_state,
                                                           xc_hybrid_coeff)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.gradients.evaluation import GeometryCache
from src.gradients.surface import PotentialEnergySurface
from src.gradients.surface_identity import force_identity
from src.gradients.rpa.factor_chain import (FactorChain, FrozenFactorization,
                                            OrbitalCapture)
from src.gradients.gwbse.reaction_field_adjoint import (added_reaction_field_adjoint,
                                                        reaction_field_backward,
                                                        reaction_field_shift,
                                                        static_screening)
from src.gradients.isdf_derivatives import GaugeAdjoint
from src.gradients.mean_field_response import (qp_xc_correction_Y,
                                               qp_xc_correction_skeleton,
                                               relaxed_density_mo)
from src.gradients.gwbse.conventions import FrozenConventions, ResolvedConventions
from src.gradients.surface import followed_root
from src.gradients.gwbse.qp_space_time import qp_gradient_space_time
from src.SingleReference.GW.qp_solve import (add_reaction_field,
                                             static_exchange_diagonal)
from src.gradients.gwbse.adaptive_selection import QuasiparticleSelection
from src.gradients.gwbse.qp_set import QuasiparticleSet
from src.gradients.gwbse.reverse_fold import ReversePass
from src.gradients.gwbse.state_following import StateFollower


#: The records `surface_of` stamps on the surface it returns, which
#: `ExcitedStateChain.to_record` keeps beside the chain's own.
STAMPED_RECORDS = ('numerics', 'realization')


class HandedOverPoint:
    """The `first_point` attribute of a surface: the `FirstPoint` it was
    handed, stamped on arrival with the hash of the conventions the surface
    holds then, which are the ones the handed-over force was evaluated on."""

    def __get__(self, surface, owner):
        if surface is None:
            return self
        return surface.__dict__.get('_first_point')

    def __set__(self, surface, point):
        # a force taken on other numerics is a force of another surface
        if point is not None:
            require_same_identity(stamp_of(point.result[-1]),
                                  numerics_identity(surface),
                                  'first point handed over')
        if point is not None and point.conventions is None:
            point = replace(point, conventions=conventions_hash(surface))
        surface.__dict__['_first_point'] = point


def dipole_change(mol, drho):
    """(3,) the dipole change -Tr[r drho] of an AO density difference `drho`
    (electrons negative), in atomic units."""
    return -np.einsum('xmn,mn->x', mol.intor('int1e_r'), drho)


def held_part(chain, name, cls):
    """The part `cls` of `chain` held under `name`, built on first use.

    A part holds its chain and no state of its own: everything it reads or
    writes is the chain's. A shallow copy of the chain (a spin view, a walk's
    surface) shares the cached part, whose chain is the original, so the part
    is rebuilt for the copy; what was set on the old part (a probe) carries
    over.
    """
    key = f'_part_{name}'
    part = chain.__dict__.get(key)
    if part is None or part.chain is not chain:
        built = cls(chain)
        if part is not None:
            built.__dict__.update({k: v for k, v in part.__dict__.items()
                                   if k != 'chain'})
        chain.__dict__[key] = part = built
    return part


class ExcitedStateChain(FactorChain, PotentialEnergySurface):
    """Frozen conventions, and the energy and gradient that follow from them.

    The physics are the arguments: spin, root (`state`, zero-based), kernel
    (`bse_tda`), environment and reference mean field. How the surface is
    computed is ONE `Numerics` (`numerics`, held as `settings`): the
    factorization, the grids, caps and tolerances, the explicit set,
    residues and scissor, the eigensolver, adjoint and root following;
    `refreeze` carries it. Drivers build the chain through
    `properties.excitations.SurfaceSpec`. `scf_factory(mol) -> mf` gives a
    displaced geometry its mean field, converged to an orbital gradient of
    ~1e-11 (`check_scf_quality`).

    The eigensolver 'auto' is production's memory rule (`bse.solver_choice`,
    `solver_used`): dense while the Casida pair (A, B) fits in
    BSE_DENSE_MAX_GB, matrix-free above it, so at finite-difference sizes it
    is the dense route and costs those cases neither a number nor an
    ordering. The two root adjoints (BSE_ADJOINTS) are one surface: 'explicit'
    contracts the three-index blocks of `bse_cache`, 'grid' closes over the
    ISDF grid (`isdf_bse_adjoint`); they agree to rounding.

    The default explicit set is the frontier half-width 2, widened over
    degenerate blocks. Every orbital outside it carries the frozen shift
    `calibrate_scissor` reads off the explicit roots at the reference
    geometry, a NUMBER, so the adjoint carries no term for it; 'curve'
    (`outside_scissor`) gives the orbitals beyond the window Delta(eps_p) of
    a frozen `OutsideCurve` instead, whose slope the adjoint carries. The
    inside scissor ('calibrate') is a different thing: a frozen shift for the
    set's states the pole model does not admit. An adaptive set (`qp_select`)
    is selected among the candidates at the first forward (module
    docstring); `qp_partition` is one an earlier stage froze.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole
    and the kernels it calls divide the M^2 sweeps of the quasiparticle
    solve and of the static screening -- the tau points of proj(tau) and of
    the quasiparticle solve's adjoint, the contour-deformation frequencies,
    the rows of Zt in the Davidson block action -- and hand every rank the
    same bits, so the frozen conventions decided here are the same on all
    of them. Over ranks the static-W adjoint (`chi0_backward_rows`) and
    the quasiparticle reverse sweep run tile-major instead: every rank
    sweeps every tau point for its grid tiles and holds the factors'
    adjoints by those tiles alone (`_adjoint_rows`).
    """

    #: The ingredients this surface provides (`surface.INGREDIENTS`).
    INGREDIENTS = frozenset({'conventions', 'energy', 'excitation',
                             'excited_states', 'mean_field', 'phase',
                             'physics', 'shared_forward', 'solver_used',
                             'spin', 'spin_view', 'state_specific_correction',
                             'to_record'})

    #: The root index (`surface.root_attribute`).
    ROOT_ATTR = 'state'

    READS_SLICED_FACTORS = True
    HOLDS_QP_SET = True

    first_point = HandedOverPoint()

    # The settings the kernels read, each one choice of the chain's
    # `Numerics` (`settings`), written by replacing it and nowhere
    # else (`chain_setting`): `track='overlap'` follows the STATE, None the
    # index; `scissor='calibrate'` freezes a shift for each set member Eq.
    # (27) excludes, from its root at the reference geometry (uncalibrated,
    # up to 7.5 meV and 3e-4 Ha/Bohr); `probe` takes min eig(A - B) after
    # every matrix-free full-BSE solve; `qp_hamiltonian` solves the set's
    # frozen groups as one quasiparticle Hamiltonian each.
    ntau_gw = chain_setting('quadrature.ntau_gw')
    e_min_below_gap = chain_setting('quadrature.e_min_below_gap')
    n_poles = chain_setting('qp.n_poles')
    sop_stride = chain_setting('qp.sop_stride')
    nroots = chain_setting('bse.nroots')
    bse_conv_tol = chain_setting('bse.conv_tol')
    probe = chain_setting('bse.probe')
    degeneracy_tol = chain_setting('qp.degeneracy_tol')
    tile_gb = chain_setting('layout.tile_gb')
    qp_window = chain_setting('qp.window')
    residue_route = chain_setting('qp.residue_route')
    scissor = chain_setting('qp.scissor')
    outside_scissor = chain_setting('qp.outside_scissor')
    qp_hamiltonian = chain_setting('qp.hamiltonian')
    solver = chain_setting('bse.solver')
    bse_adjoint = chain_setting('bse.adjoint')
    track = chain_setting('bse.track')

    def __init__(self, mol, scf_factory, spin='singlet', state=0, bse_tda=False,
                 environment=None, mf=None, numerics=None,
                 factorization=None, qp_partition=None, carried=None):
        # every choice as asked, held as `settings` and carried verbatim by
        # `refreeze`; not `numerics`/`realization`, which `surface_of` stamps
        # with its own resolved records on the surface it returns
        super().__init__(mol, scf_factory, numerics=numerics, mf=mf,
                         environment=environment, factorization=factorization)
        numerics = self.settings
        self.spin, self.state, self.bse_tda = spin, state, bse_tda
        # every discrete choice frozen at the reference geometry, keyed by
        # orbital identity (`FrozenConventions`)
        self.conventions = FrozenConventions(
            key_mol=self.mol0, key_mo=np.asarray(self.mf0.mo_coeff, float),
            nocc=self.nocc, outside_scissor=numerics.qp.outside_scissor)
        # the conventions in the indices of the geometry being evaluated
        self._rv = None
        self._followed = None          # (mol, mo_coeff, X, Y) last committed
        self._anchor = None            # ... and of the first, for drift
        # what this chain computed at the last geometries (`Evaluation`),
        # with the state matched at each until one is committed
        self._evaluations = GeometryCache()
        self.follow_log = []
        # each quasiparticle root kept on its branch while the largest weight
        # was elsewhere, by orbital and geometry (`_log_qp_branches`)
        self.qp_branch_log = []
        # each quasiparticle-Hamiltonian group kept with Z outside (0, 1] or
        # carried on short of self-consistency, by group and geometry
        # (`_log_qp_hamiltonian`)
        self.qp_hamiltonian_log = []
        # each matrix-free Casida solve whose roots the force reads were left
        # above read_tol or unconfirmed, kept as solved (`_flag_casida`)
        self.casida_log = []
        # where each entry of `follow_log` was evaluated (`followed_root`)
        self.follow_coords = []
        self.is_ks = xc_hybrid_coeff(self.mf0)[0]
        # each Davidson's {'stats', 'timings'}, kept while a stage timer is set
        self.davidson_solves = []
        # the roots of the last geometry a walk committed, which the next
        # matrix-free Casida solve starts from (`_casida`), shared with the
        # spin views; None solves every geometry cold
        self.davidson_warm_start = DavidsonWarmStart()
        # None: a set with a state on the quadrature takes CD_NFREQ and a
        # pole-model set CD_NFREQ_SOP (`_fix_cd_grid`); a size asked for is a
        # floor under both
        carried = CarriedGrids() if carried is None else carried
        asked = (numerics.quadrature.nfreq_cd if carried.nfreq_cd is None
                 else carried.nfreq_cd)
        self.conventions.nfreq_cd_floor = 0 if asked is None else int(asked)
        self.conventions.nfreq_cd = CD_NFREQ if asked is None else int(asked)

        eps = np.asarray(self.mf0.mo_energy, float)
        occ, virt = get_occ_virt_indices(eps, self.nocc)
        gap = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        self.gap = gap
        self.quasiparticles._build_cd_grid(self.conventions.nfreq_cd, eps)
        ntau_w = (numerics.quadrature.ntau_w if carried.ntau_w is None
                  else carried.ntau_w)
        self.conventions.ntau_w = (
            minimax_points_for_accuracy(gap, e_max,
                                        target=DEFAULT_TAU_TARGET)[0]
            if ntau_w is None else int(ntau_w))
        self.w_grid = TimeFrequencyGrid.minimax_split(
            self.ntau_w, gap, e_max, [0.0], [1.0],
            with_sine=False, with_inverse=False)
        self.conventions.qp_set = self.quasiparticles._qp_set(
            eps, self.nocc,
            numerics.qp.window if carried.window is None else carried.window,
            numerics.qp.degeneracy_tol)
        self.conventions.window = tuple(int(p) for p in self.resolved.qp_set)
        # A token per chain, renewed by every shallow copy (`__copy__`): what
        # owns the adaptive selection's working state, so a spin view never
        # refines the selection it shares, and the state matched in a held
        # evaluation, so copies sharing the cache never adopt each other's;
        # a freed chain's identity, reused by another object, claims neither.
        self._owner = object()
        self.selection._init_adaptive(numerics.qp.select, qp_partition)

    @property
    def follower(self):
        """The Casida step and the following of the state (`StateFollower`)."""
        return held_part(self, 'follower', StateFollower)

    def forward(self, mol, mf):
        """`StateFollower.forward`: the forward to the Casida step."""
        return self.follower.forward(mol, mf)

    def verified(self, mol, mf, shared, solved, solve):
        """`StateFollower.verified`: the solved roots, checked."""
        return self.follower.verified(mol, mf, shared, solved, solve)

    def solve_casida(self, shared, warm=True):
        """`StateFollower.solve_casida`."""
        return self.follower.solve_casida(shared, warm=warm)

    def shared_forward(self, mol, mf):
        """`StateFollower.shared_forward`: one forward below the Casida step."""
        return self.follower.shared_forward(mol, mf)

    def tracked_state(self, mol, mf, solution):
        """`StateFollower.tracked_state`."""
        return self.follower.tracked_state(mol, mf, solution)

    def tracked_root(self, mol, mf, shared, solution, widen=True):
        """`StateFollower.tracked_root`."""
        return self.follower.tracked_root(mol, mf, shared, solution, widen=widen)

    def following(self):
        """`StateFollower.following`."""
        return self.follower.following()

    def hand_over(self, held):
        """`StateFollower.hand_over`."""
        return self.follower.hand_over(held)

    def commit(self, mol):
        """`StateFollower.commit`: a walk accepted `mol`."""
        return self.follower.commit(mol)

    @property
    def selection(self):
        """The adaptive selection of the explicit quasiparticle set
        (`QuasiparticleSelection`)."""
        return held_part(self, 'selection', QuasiparticleSelection)

    @property
    def qp_growth(self):
        """What a grow-only refreeze did to the explicit set
        (`QuasiparticleSelection.qp_growth`), which a walk's record reads."""
        return self.selection.qp_growth

    def selection_record(self):
        """`QuasiparticleSelection.selection_record`, for the records."""
        return self.selection.selection_record()

    def posteriori_check(self, mol, mf, spectra, targets):
        """`QuasiparticleSelection.posteriori_check` at the end of a walk."""
        return self.selection.posteriori_check(mol, mf, spectra, targets)

    @property
    def quasiparticles(self):
        """The quasiparticle set (`QuasiparticleSet`): its frozen grid, its
        branches and groups, the subspace Hamiltonian and their seeds."""
        return held_part(self, 'quasiparticles', QuasiparticleSet)

    @property
    def reverse(self):
        """The reverse pass (`ReversePass`): the Casida-level adjoints carried
        back to the nuclei."""
        return held_part(self, 'reverse', ReversePass)

    @property
    def ntau_w(self):
        """The static W's tau count, frozen in the conventions."""
        return self.conventions.ntau_w

    @property
    def resolved(self):
        """The frozen conventions in the orbital indices of the geometry
        evaluated last (`ResolvedConventions`), the stored ones before any:
        what every kernel of the forward and the reverse pass reads."""
        view = self.__dict__.get('_rv')
        return view if view is not None else self.conventions.live()

    def _resolve_at(self, mol, mf):
        """Resolve the conventions at (mol, mf) for the passes that follow."""
        self._rv = self.conventions.at(mol, mf)
        return self._rv

    def __copy__(self):
        """A shallow copy that owns no selection and no match: every
        attribute shared by reference, its own `_owner`, and once the
        conventions are anchored its own keys (`keyed_copy`), so walking one
        copy (a spin view, a root, a walk's surface) re-keys no other; before
        that the copies share them whole, so what the first forward freezes
        is every copy's."""
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out._owner = object()
        if self.conventions.anchored:
            out.conventions = self.conventions.keyed_copy()
            view = self.__dict__.get('_rv')
            if view is not None:
                out._rv = ResolvedConventions(out.conventions, view.perm,
                                              view.rotations, view.mol)
        return out

    def to_record(self):
        """Everything this surface froze, for `from_record`: the reference
        geometry, the physics, the `Numerics` as asked (and the records
        `surface_of` stamped), the factorization, the conventions, the screening's
        tau grid, the reference mean field's ISDF-K pair layout and the
        root-following state (the committed and the anchor eigenvectors with
        their orbitals, and the log), and the committed Davidson roots of the
        warm start (`DavidsonWarmStart.to_record`, None where it is off)."""
        def followed(held):
            if held is None:
                return None
            mol, mo, xn, yn = held
            return (np.array(mol.atom_coords(), float), mo, xn, yn)
        warm = self.__dict__.get('davidson_warm_start')
        return {
            'class': f'{type(self).__module__}:{type(self).__qualname__}',
            'mol0_coords': np.array(self.mol0.atom_coords(), float),
            **self.settings_record(),
            'factorization': self.factorization.to_record(),
            'conventions': self.conventions.to_record(),
            'carried': self.carried_grids(
                tuple(int(p) for p in self.conventions.window)
                if np.ndim(self.qp_window) == 1 else None),
            'ntau_w': self.ntau_w, 'w_grid': self.w_grid, 'gap': self.gap,
            'scf_layout': self.scf_layout(),
            'followed': followed(self._followed),
            'anchor': followed(self._anchor),
            'follow_log': copy.deepcopy(self.follow_log),
            'follow_coords': list(self.follow_coords),
            'qp_branch_log': copy.deepcopy(self.qp_branch_log),
            'qp_hamiltonian_log': copy.deepcopy(self.qp_hamiltonian_log),
            'casida_log': copy.deepcopy(self.__dict__.get('casida_log', [])),
            # the committed roots the next Davidson starts from, so a chain
            # read back (a resumed walk) solves as this one would
            'davidson_warm_start': (None if warm is None
                                    else warm.to_record())}

    def settings_record(self):
        """The physics and the `Numerics` as asked, and the records
        `surface_of` stamped: what this chain was asked to compute, beside
        what it froze."""
        return {'spin': self.spin, 'state': int(self.state),
                'bse_tda': self.bse_tda,
                'settings': self.settings,
                'stamped': {k: self.__dict__[k] for k in STAMPED_RECORDS
                            if k in self.__dict__}}

    @classmethod
    def from_record(cls, record, mol, scf_factory, environment=None, mf=None,
                    factorization=None):
        """The surface `to_record` wrote, rebuilt without deriving any
        convention again: one SCF at its reference geometry (or `mf`, the
        mean field there), and every frozen choice read back.

        mol: the template molecule (basis, charge, spin) of the surface;
        scf_factory and environment: the declaration's, as `surface_of`
        hands them; factorization: the recorded one already read back by
        another half of a composed surface, which this chain shares.
        """
        mol0 = molecule_at(mol, record['mol0_coords'])
        chain = cls(mol0, scf_factory, spin=record['spin'],
                    state=record['state'], bse_tda=record['bse_tda'],
                    environment=environment, mf=mf,
                    numerics=record['settings'],
                    carried=record['carried'],
                    factorization=(FrozenFactorization.from_record(
                        record['factorization']) if factorization is None
                        else factorization))
        chain.__dict__.update(record['stamped'])
        chain.conventions = FrozenConventions.from_record(
            record['conventions'], mol)
        chain.conventions.ntau_w = record['ntau_w']
        chain.w_grid = record['w_grid']
        chain.gap = record['gap']
        chain._scf_layout = {'layout': record['scf_layout']}

        def followed(held):
            if held is None:
                return None
            coords, mo, xn, yn = held
            return (molecule_at(mol, coords), mo, xn, yn)
        chain._followed = followed(record['followed'])
        chain._anchor = followed(record['anchor'])
        chain.follow_log = list(record['follow_log'])
        chain.follow_coords = list(record['follow_coords'])
        chain.qp_branch_log = list(record.get('qp_branch_log', []))
        chain.qp_hamiltonian_log = list(record.get('qp_hamiltonian_log', []))
        chain.casida_log = list(record.get('casida_log', []))
        warm = record.get('davidson_warm_start', {})
        if warm is None or warm:
            chain.davidson_warm_start = (
                None if warm is None else
                DavidsonWarmStart.from_record(warm, mol))
        return chain

    # ------------------------------------------------------------ declaration
    @property
    def physics_ground_state(self):
        """E_0 = E_KS[xc], the mean field's OWN energy.

        `total_energy` is `mf.e_tot + Omega` and carries no correlation term of
        its own, so the functional under the state is the mean field's,
        whatever that mean field is: GroundState('dft', xc) for a Kohn-Sham
        factory and GroundState('dft', 'hf') for Hartree-Fock. The dRPA ground
        state of Toelle Eq. (15) is a DIFFERENT surface and is composed in
        `RPABSESurface`, which declares its own.
        """
        return declared_ground_state(self.mf0, 'dft')

    @property
    def physics_excitation(self):
        """The `Excitation` this chain's settings mean.

        Its spin, the kernel `bse_tda` selects, and the followed state counted
        from one -- `state` is a zero-based index into the energy-ordered
        manifold and `Excitation.root` is one-based. `qp` and `screening` are
        G0W0 and dRPA, which is the only pair this chain realizes.
        """
        return Excitation(spin=self.spin, root=self.state + 1,
                          kernel='bse-tda' if self.bse_tda else 'bse')

    @property
    def physics(self):
        """What this chain computes: E_KS + Omega, in its own environment.

        DECLARED HERE, not stamped from outside. A surface built through
        `potential_energy_surface` is CHECKED against this rather than labelled
        by it, so a dispatcher that built the chain with settings the caller
        did not ask for is a refusal instead of a mislabelled number.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.environment))

    def solver_used(self, n_ov):
        """'dense' or 'davidson' for a pair space of n_ov, resolving 'auto'
        through production's ONE rule, `bse.solver_choice`.

        The rule is dense while the Casida pair (A, B) fits in
        BSE_DENSE_MAX_GB; `solver_choice`'s docstring is where it is stated and
        why the Tamm-Dancoff form is not exempt from it (the dense route builds
        both blocks either way).

        A Tamm-Dancoff problem above the rule goes matrix-free like the full
        one: `solve_casida_davidson(tda=True)` diagonalizes A alone. Triplets
        go through Davidson too -- kappa = 0 removes the bare-exchange term
        from the block action and nothing else changes.
        """
        if self.solver != 'auto':
            return self.solver
        return solver_choice(n_ov, self.bse_tda)

    def _xc_chain(self, mf, weights):
        """(Y_extra, skeleton) of sum_p w_p <p|Sigma_x - v_xc|p>, or (None,
        zero) on Hartree-Fock, where the operator vanishes identically.

        On a Kohn-Sham reference the static correction is part of the
        quasiparticle energy, so its dependence on the orbitals and on the
        nuclei is carried like any other: the Y piece before the multiplier
        solve, because it shares Lambda, and the skeleton in the assembly.

        The environment's static term has no channel here. In a continuum it
        is Eq. (18), whose adjoint reaches the nuclei through (eps, X, D) in
        the caller; adding the COHSEX operator's own adjoint would enter the
        reaction field twice. An environment that does not dress the
        interaction (gas phase, fixed charges) has no static term.
        """
        if not self.is_ks:
            return None, np.zeros((mf.mol.natm, 3))
        y = qp_xc_correction_Y(mf, weights, self.nocc)
        if getattr(getattr(self, '_orbital_capture', None), 'density_only', False):
            # a relaxed density reads Y only; the skeleton is the force's
            return y, np.zeros((mf.mol.natm, 3))
        return y, qp_xc_correction_skeleton(mf, weights, self.nocc)

    def xc_correction(self, mf, states, reaction_field=None):
        """<p|Sigma_x - v_xc|p> for `states`: production's static term,
        evaluated with THIS geometry's environment attached.

        reaction_field: this geometry's Eq. (18) array, or None in the gas
        phase. It is NOT part of the term: every quasiparticle equation here
        is G0W0, so the caller adds the shift to its roots after the solve
        (`add_reaction_field`), whole rather than Z x shift; handing it says
        that the caller does so.

        Computed on Hartree-Fock too, where it is round-off rather than an
        exact zero. The previous attachment is put back afterwards.
        """
        with attached_environment(mf, self.environment_at(mf.mol)):
            return static_exchange_diagonal(
                mf, mf.mol, states, exchange='mf',
                reaction_field_after_solve=reaction_field is not None)

    def _factors_for(self, mol, mf):
        """(X_mo, D, eps, mu, auxmol, coords, X_ao, D_bare) at one geometry.

        D_bare is None in the gas phase; where it is not, it is what the
        self-energy screens with while D carries the cavity for the kernel.
        """
        x_mo, d, eps, auxmol, crd, x_ao = self.factors_at(mol, mf)
        mu = 0.5 * (eps[:self.nocc].max() + eps[self.nocc:].min())
        return (x_mo, d, eps, mu, auxmol, crd, x_ao,
                self.bare_factor(mol, auxmol, crd))

    def _reaction_field(self, x_mo, d, d_bare, eps, screening=None):
        """(Eq. (18) on this geometry's factors, the static screening it was
        built from), or (None, None) in the gas phase.

        Built on the SAME axis as the BSE kernel's static W, so the two
        screenings a solvated run needs come off one grid -- and the dressed
        member IS the kernel's W, so a caller that already has it passes the
        pair rather than rebuilding chi0(0) in the dressed gauge twice.
        """
        if d_bare is None:
            return None, None
        if screening is None:
            screening = (
                static_screening(x_mo, d, eps, self.nocc, self.w_grid),
                static_screening(x_mo, d_bare, eps, self.nocc, self.w_grid))
        return (reaction_field_shift(x_mo, d, d_bare, eps, self.nocc,
                                     grid=self.w_grid, screening=screening),
                screening)

    @property
    def conventions_frozen(self):
        """Whether the first forward has fixed what every later one reads:
        the contour-deformation quadrature sized and, outside the set, the
        scissor calibrated."""
        view = self.resolved
        return (self.conventions.cd_sized and view.outside_shift is not None
                and (self.qp_select is None or view.qp_partition is not None))

    def spin_view(self, spin, state=None, track=None):
        """This chain with another spin and root, on the SAME frozen objects.

        A shallow copy, refused until `conventions_frozen`: the quadrature
        (`nu`, `wt`, `gw_grid`), `cd_sized` and `outside_shift` are REBOUND by
        the first forward, so a copy made earlier would size its own grid and
        read a tape built on another frequency axis. Made afterwards it shares
        every convention, the factorization, the environment and its cache by
        reference, and keeps its own root-following history and its own
        record of the Davidsons it runs. `state` and `track` default to this
        chain's; the stage timer is shared.
        """
        if spin not in KAPPA:
            raise ValueError(f'spin={spin!r} not in {tuple(KAPPA)}')
        if not self.conventions_frozen:
            raise RuntimeError(
                'a spin view shares the conventions the first forward pass '
                'freezes (the contour-deformation grid, the outside scissor); '
                'evaluate the chain once before taking one')
        view = copy.copy(self)
        view.spin = spin
        view.state = self.state if state is None else int(state)
        view.track = self.track if track is None else track
        view._followed = view._anchor = None
        view._evaluations = GeometryCache()
        view.follow_log = []
        view.follow_coords = []
        view.casida_log = []
        view.davidson_solves = []
        return view

    def spectrum(self, mol=None, mf=None):
        """Every excitation energy the Casida step returned, ascending, in Hartree."""
        mol, mf = self.mean_field(mol, mf)
        return self.forward(mol, mf)[1].omega

    def excited_states(self, mol=None, mf=None):
        """(omega, X, Y, nocc): every root the Casida step solved at `mol`,
        not just `state` (the `excited_states` ingredient of
        `src.gradients.surface`)."""
        mol, mf = self.mean_field(mol, mf)
        solution = self.forward(mol, mf)[1]
        return solution.omega, solution.x, solution.y, self.nocc

    def excitation(self, mol=None, mf=None):
        """Omega_state in Hartree."""
        return self.energy(mol, mf)[2]

    def energy(self, mol=None, mf=None):
        """(E_ex, E_0, Omega) in Hartree, with E_ex = E_0 + Omega."""
        mol, mf = self.mean_field(mol, mf)
        root, _, solution = self.follower._tracked(mol, mf)
        om = float(solution.omega[root])
        return mf.e_tot + om, mf.e_tot, om

    def excitation_gradient(self, mol=None, mf=None):
        """(dOmega/dR, diagnostics). Nothing larger than three-index; taken
        once per geometry and root, and read back after that."""
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        self.require_scf_quality(mf, 'excitation_gradient')
        root, shared, solution = self.follower._tracked(mol, mf)
        held = self.follower._held(mol, mf)
        if held is not None and ('omega', root) in held.forces:
            return held.forces[('omega', root)]
        out = self.reverse.root_gradient(shared, solution, root)
        if held is not None:
            held.forces[('omega', root)] = out
        return out

    def excitation_density(self, mol=None, mf=None, relaxed=True,
                           frozen_field=True):
        """The followed state's density difference in the AO basis, nao^2.

        relaxed=True: d Omega / d h, the orbital-relaxed (Lagrangian) density
        of the excitation energy, gamma + Lambda from the inputs the force
        hands its orbital branch (`relaxed_density_mo`), so that
        Tr[drho h] is the first-order change of Omega under any one-electron
        perturbation h. frozen_field=True solves the multiplier with the
        ground state's surface charges held (the state-specific convention);
        False lets the continuum respond, as the force's multiplier does. It
        costs one excitation gradient.

        relaxed=False: the amplitude density of the Casida vectors (norm
        |X|^2 - |Y|^2 = 1), dP_ab = sum_i (X_ia X_ib + Y_ia Y_ib) and
        dP_ij = -sum_a (X_ia X_ja + Y_ia Y_ja), trace zero.

        The relaxed density is a force's Lagrangian: a mean field a force
        would flag is warned about and flagged (`flag_scf_quality`).
        """
        mol, mf = self.mean_field(mol, mf)
        if relaxed:
            self.flag_scf_quality(mf, 'excitation_density(relaxed=True)')
        root, shared, solution = self.follower._tracked(mol, mf)
        C = np.asarray(mf.mo_coeff, float)
        nocc = self.nocc
        if not relaxed:
            nvir = C.shape[1] - nocc
            x = np.asarray(solution.x)[:, root].reshape(nocc, nvir)
            y = np.asarray(solution.y)[:, root].reshape(nocc, nvir)
            p_mo = np.zeros((C.shape[1],) * 2)
            p_mo[nocc:, nocc:] = x.T @ x + y.T @ y
            p_mo[:nocc, :nocc] = -(x @ x.T + y @ y.T)
            return C @ p_mo @ C.T
        # the reverse pass up to the orbital branch's inputs only: the
        # multiplier, collocation and fit branches past them are the force's
        capture = OrbitalCapture(density_only=True)
        self._orbital_capture = capture
        try:
            self.reverse.root_gradient(shared, solution, root, release_tape=False)
        finally:
            self._orbital_capture = None
        gammas = [np.diag(e) if e.ndim == 1 else e for e, _ in capture]
        p_mo, _ = relaxed_density_mo(mf, sum(gammas), nocc,
                                     Y_extra=sum(y for _, y in capture),
                                     environment=not frozen_field)
        return C @ p_mo @ C.T

    def state_specific_correction(self, mol=None, mf=None):
        """Scheme B's record at one geometry (`ClassicalStateSpecific`):
        omega_B, E_SS of the density the environment names, and E_B =
        omega_B + E_SS, in Hartree, with the density's dipole change.

        E_SS is taken in the optical continuum (vertical) or its static
        partner (equilibrium), as the environment declares. Refused on any
        other environment: the Duchemin scheme carries its state-specific
        part inside omega already.
        """
        mol, mf = self.mean_field(mol, mf)
        env = self.environment_at(mol)
        if getattr(env, 'scheme', None) != 'classical-ss':
            raise ValueError(
                f'a classical state-specific term belongs to scheme B '
                f'(environment.solvent_scheme(env, "classical-ss")); this '
                f'chain stands in {environment_label(env)}, whose excitation '
                f'carries its state-specific part already')
        omega = float(self.excitation(mol, mf))
        drho = self.excitation_density(mol, mf,
                                       relaxed=env.ss_density == 'relaxed')
        e_ss, _ = env.ss_continuum().state_specific_energy(mol, drho)
        dip = dipole_change(mol, drho)
        return {'omega_B': omega, 'E_SS': e_ss, 'E_B': omega + e_ss,
                'ss_density': env.ss_density,
                'ss_equilibrium': env.ss_equilibrium,
                'delta_dipole': dip.tolist()}

    def interstate_gradient(self, m, n, mol=None, mf=None):
        """(d<m|H|n>/dR, diagnostics) between two BSE roots, m != n.

        The numerator of the derivative coupling. The element itself vanishes
        -- H is diagonal in its own eigenbasis -- so this is the derivative of
        a zero, and it is not a gradient of anything: it depends on how the
        orbital basis moves, and only the CSF term of
        `src.gradients.gwbse.derivative_coupling` restores that gauge. Take the
        coupling from there rather than from this alone.

        Everything below the Casida step is the excitation gradient's own
        chain, so the orbital relaxation and the Pulay terms arrive with it.
        """
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        self.require_scf_quality(mf, 'interstate_gradient')
        return self.reverse.interstate_gradient_of(*self.forward(mol, mf), m, n)

    def _refuse_split_ion(self, mol):
        """A charged state has no transition density, so its equilibrium is
        Eq. (18) at eps_static whole: the full object, not the split one."""
        if getattr(self.environment_at(mol), 'equilibrium', False) == 'split':
            raise ValueError(
                "a quasiparticle level in equilibrium is Eq. (18) at "
                "eps_static, which the full object gives: use "
                "equilibrium_environment(env, 'full')")

    def _quasiparticle_solve(self, offset, mol, mf, want_grad):
        """(the route's output, its route_out, (X_mo, D, eps, auxmol, coords,
        D_bare, shift, screening)) of the orbital `offset` from the HOMO at
        (mol, mf), on the set's quadrature widened by that orbital, repeated
        until its Newton branch is frozen.

        The static correction belongs in the forward as much as in the
        gradient: omitting it from one makes the two describe different
        functions, and a finite difference of one against a gradient of the
        other then looks exactly like a 10% gradient error.
        """
        self._refuse_split_ion(mol)
        self.follower._anchored_first(mol, lambda: self._quasiparticle_solve(
            offset, self.mol0, self.mf0, False))
        self._resolve_at(mol, mf)
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        shift, screening = self._reaction_field(x_mo, d, d_bare, eps)
        orb = self.nocc - 1 + offset
        d_sigma = d if d_bare is None else d_bare
        # the grid is the set's, widened by the orbital asked for, so a chain
        # reaches the same quadrature through a BSE excitation or one state
        self.quasiparticles._fix_cd_grid(np.union1d(self.resolved.qp_set, [orb]), eps)
        xc_orb = float(self.xc_correction(mf, [orb], shift)[0])
        while True:
            qp_out = {}
            with self.phase('t_qp_backward' if want_grad else 't_qp'):
                grid = self.conventions
                out = qp_gradient_space_time(
                    x_mo, d_sigma, eps, self.nocc, grid.gw_grid, grid.nu,
                    grid.wt, orb, mu=mu, want_grad=want_grad,
                    xc_correction=xc_orb, route_out=qp_out, **self.quasiparticles._qp_kw())
            if not self.quasiparticles._freeze_qp_branch(qp_out, eps, xc_orb, [orb]):
                break
        # Eq. (18) whole, on the root solved without it
        out = (float(add_reaction_field(out[0], shift, orb)),) + tuple(out[1:])
        self.conventions.anchor(mol)
        return out, qp_out, (x_mo, d, eps, auxmol, crd, d_bare, shift,
                             screening)

    def quasiparticle(self, offset=0, mol=None, mf=None):
        """eps^QP for the orbital `offset` from the HOMO (0 = HOMO, +1 = LUMO)."""
        mol, mf = self.mean_field(mol, mf)
        return float(self._quasiparticle_solve(offset, mol, mf, False)[0][0])

    def quasiparticle_gradient(self, offset=0, mol=None, mf=None):
        """(d eps^QP/dR, diagnostics) for one quasiparticle, its energy among
        the diagnostics (`qp_energy`).

        A separate entry point from `excitation_gradient` because it is a
        different physical quantity with a different chain: no BSE adjoint and
        no screening adjoint, one state rather than a whole set.
        """
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        self.require_scf_quality(mf, 'quasiparticle_gradient')
        (w_star, z_fac, eps_bar, x_bar, d_bar), qp_out, factors = \
            self._quasiparticle_solve(offset, mol, mf, True)
        x_mo, d, eps, auxmol, crd, d_bare, shift, screening = factors
        orb = self.nocc - 1 + offset
        # THE CORRECTION CARRIES Z, NOT 1. The Newton condition is
        # w = eps_p + Delta_p + Sigma_c(w), so every term on the right is
        # renormalized by Z = [1 - dSigma_c/dw]^-1 on its way into dw --
        # including Delta_p. Weighting it by 1 instead leaves an error of
        # (1 - Z) times the correction's own gradient, which on a valence
        # quasiparticle is 10-20%: exactly the size seen on water/PBE0.
        # Eq. (18) is added to the root after the solve, so it rides at 1.
        w_corr = np.zeros(len(eps))
        w_corr[orb] = float(z_fac)
        w_env = np.zeros(len(eps))
        w_env[orb] = 1.0
        d_bar_bare = None
        if shift is not None:
            # Sigma screened bare, so the adjoint above is on the BARE factor;
            # the dressed one is reached only through Eq. (18).
            block = self.reverse._adjoint_rows(x_mo)
            d_bar_bare, d_bar = d_bar, self.zero_adjoint(d, block)
            if block is not None:
                x_bar = self.reverse._as_rows(x_bar, block)
                d_bar_bare = self.reverse._as_rows(d_bar_bare, block)
            with self.phase('t_reaction_field_backward'):
                eps_bar = added_reaction_field_adjoint(
                    eps_bar, x_bar, d_bar, d_bar_bare,
                    reaction_field_backward(
                        w_env, x_mo, d, d_bare, eps, self.nocc,
                        grid=self.w_grid, screening=screening, block=block))
        with self.one_fit_adjoint(mf):
            y_xc, g_xc = self._xc_chain(mf, w_corr)
            grad, diags = self.nuclear_gradient(mol, mf, auxmol, crd, x_mo,
                                                eps_bar, x_bar, d_bar,
                                                y_extra=y_xc, g_extra=g_xc,
                                                d_bar_bare=d_bar_bare)
        # Z enters dw/dR and not w, so a route with Sigma's value right and
        # its slope wrong returns an exact energy with a wrong force. Without
        # Z in the record nothing says so.
        return grad, dict(diags, qp_energy=float(w_star), qp_z=float(z_fac),
                          qp_route=qp_out.get('residue_route'))

    def _equilibrium_pieces(self, mol, mf):
        """The factors and the three static screenings Delta eps^eq reads.

        The optical (dressed) and the static partner's gauge each screen the
        one chi0 in their own metric; the bare screening is shared by both
        Eq. (18) terms, so three chi0(0) sweeps serve the difference.
        """
        if self.laid_out_in_rows():
            raise ValueError(
                "the equilibrium shift of a charged state reads its three "
                "gauges as whole factors; build the factorization unsliced "
                "with fit='replicated' for it")
        x_mo, d, eps, _, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        if d_bare is None:
            raise ValueError(
                'equilibrium solvation needs a continuum that dresses the '
                'interaction; this chain screens in the gas phase')
        partner, d_static = self.static_factor(mol, auxmol, crd)
        with self.phase('t_screening'):
            bare = static_screening(x_mo, d_bare, eps, self.nocc, self.w_grid)
            optical = (static_screening(x_mo, d, eps, self.nocc, self.w_grid),
                       bare)
            relaxed = (static_screening(x_mo, d_static, eps, self.nocc,
                                        self.w_grid), bare)
        return (x_mo, d, d_static, d_bare, eps, auxmol, crd, partner, optical,
                relaxed)

    def equilibrium_shift(self, offset=0, mol=None, mf=None):
        """Delta eps^eq = Eq18_p(eps_s) - Eq18_p(eps_inf) on this chain's factors.

        The orbital `offset` from the HOMO, in Hartree: how much further the
        level moves once the solvent has relaxed around the charged state.
        The ion's energy E_0 -/+ eps^QP moves by -/+ this, <= 0. On the same
        factors and grid as the chain's own Eq. (18), so its gradient
        (`equilibrium_shift_gradient`) is of this number and not of the
        density-fitted one `calc_qp_energy(equilibrium=True)` adds.
        """
        mol, mf = self.mean_field(mol, mf)
        (x_mo, d, d_static, d_bare, eps, _, _, _, optical,
         relaxed) = self._equilibrium_pieces(mol, mf)
        orb = self.nocc - 1 + offset
        return float(
            reaction_field_shift(x_mo, d_static, d_bare, eps, self.nocc,
                                 grid=self.w_grid, screening=relaxed)[orb]
            - reaction_field_shift(x_mo, d, d_bare, eps, self.nocc,
                                   grid=self.w_grid, screening=optical)[orb])

    def equilibrium_shift_gradient(self, offset=0, mol=None, mf=None):
        """(d Delta eps^eq / dR, diagnostics) for the orbital `offset`.

        Two passes of the Eq. (18) adjoint, at the static partner and at the
        optical response, and their difference: the static one lands its D
        adjoint on the partner's metric, which carries the cavity's motion in
        the static response, the optical one on the chain's own dressed
        gauge, and both on the one bare factor they share.
        """
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        self.flag_scf_quality(mf, 'equilibrium_shift_gradient')
        (x_mo, d, d_static, d_bare, eps, auxmol, crd, partner, optical,
         relaxed) = self._equilibrium_pieces(mol, mf)
        weights = np.zeros(len(eps))
        weights[self.nocc - 1 + offset] = 1.0
        with self.phase('t_reaction_field_backward'):
            e_s, x_s, d_s, b_s = reaction_field_backward(
                weights, x_mo, d_static, d_bare, eps, self.nocc,
                grid=self.w_grid, screening=relaxed)
            e_o, x_o, d_o, b_o = reaction_field_backward(
                weights, x_mo, d, d_bare, eps, self.nocc, grid=self.w_grid,
                screening=optical)
        return self.nuclear_gradient(
            mol, mf, auxmol, crd, x_mo, e_s - e_o, x_s - x_o, -d_o,
            d_bar_bare=b_s - b_o,
            gauges=[GaugeAdjoint(d_s, partner)])

    def total_gradient(self, mol=None, mf=None):
        """(dE_ex/dR, E_ex, diagnostics) -- the quantity an optimizer steps on.

        The ground-state gradient is pyscf's own for whatever mean field the
        chain was handed, so a density-fitted reference gets the density-fitted
        gradient including its auxiliary-basis response.
        """
        replay = replayed_first_point(self, mol)
        if replay is not None:
            return replay
        mol, mf = self.mean_field(mol, mf)
        self.require_differentiable_environment()
        self.require_scf_quality(mf, 'total_gradient')
        root, shared, solution = self.follower._tracked(mol, mf)
        held = self.follower._held(mol, mf)
        if held is not None and ('total', root) in held.forces:
            return held.forces[('total', root)]
        # one fit adjoint for both: the mean field's exchange skeleton rides
        # the excitation's assembly, which is therefore taken here afresh
        with self.one_fit_adjoint(mf, mean_field=True):
            g_om, diags = self.reverse.root_gradient(shared, solution, root)
            g_0 = self.mean_field_gradient(mf)
        out = self.composed_total(mf, g_0, g_om, diags)
        if held is not None:
            held.forces[('total', root)] = out
        return out

    @staticmethod
    def composed_total(mf, g_0, g_om, diags):
        """(dE_ex/dR, E_ex, diagnostics) from the mean field's force and
        (dOmega/dR, diagnostics): E_ex = E_0 + Omega, one assembly."""
        diags = dict(diags, e_scf=mf.e_tot, grad_scf_max=float(np.abs(g_0).max()),
                     grad_omega_max=float(np.abs(g_om).max()))
        return g_0 + g_om, mf.e_tot + diags['omega'], diags

    # ------------------------------------------- the potential-energy surface
    def total_energy(self, mol=None, mf=None):
        """E_ex = E_0 + Omega_state in Hartree: the surface, not the excitation."""
        return self.energy(mol, mf)[0]

    def _refrozen(self, mol, factorization, mf, walked, qp_floor):
        """The same surface with every frozen convention rebuilt at `mol`
        (`FactorChain.refreeze`).

        The factorization (layout, frames; radii given explicitly travel as
        the choice they are), the Newton branch, the poles, the scissor and
        both quadratures are re-derived from `mol` and its own mean field
        (`FrozenConventions.rebuild`); the physics, the realization and the
        numerics travel verbatim -- the `Numerics` stays what was asked, its
        identity is the same before and after -- and the tau count and the
        contour-deformation size this chain grew to travel beside it
        (`carried_grids`), so only the geometry differs and the quadrature
        never steps back. A set given as orbital
        indices -- what a `QPStates` declaration resolved to -- travels as
        those ORBITALS, matched at `mol` by overlap, never re-resolved; a
        half-width or 'all' is taken again at `mol`. The environment is the
        same object; it rebuilds itself around the atoms.

        An adaptive set is selected again at `mol`, GROW-ONLY: the selection
        starts from this surface's explicit set and its own floor, so every
        state explicit here is explicit there and a walk's passes stand on
        one set once it stops growing (re-selected from scratch, formaldehyde
        T1's hole moved from orbital 5 at R0 to 10 at the S1 minimum and 11
        at T1's). qp_floor: more states to carry, the explicit set of a
        surface this chain was not refrozen from. The admitted window stays
        the outer bound.

        mf: the new reference mean field at `mol`, converged by the caller
        (a composed surface's ground half), else the chain converges its own.
        walked: the mean field at `mol` the walk evaluated this chain on,
        which the conventions are carried over on; else this chain's own
        (`mean_field`). A composed surface hands both, so its refreeze
        stands on one new SCF and the one the walk already converged.
        """
        # the tracked index as the starting guess; a followed state is handed
        # over as its eigenvector below
        state = followed_root(self, mol)
        reselect = self.qp_select is not None
        _, walked = self.mean_field(mol, walked)
        view = self.conventions.at(mol, walked)
        carried = self.conventions.rebuild(mol, walked)
        floor = set(int(p) for p in (qp_floor or ()))
        if view.qp_partition is not None:
            floor |= set(carried['explicit'])
        if carried['floor'] is not None:
            floor |= set(carried['floor'])
        # an explicit set travels as the orbitals it names, in the indices at
        # `mol`; a half-width or 'all' is the asked one and is taken again
        window = carried['window'] if np.ndim(self.qp_window) == 1 else None
        chain = type(self)(
            mol, self.scf_factory, spin=self.spin, state=state,
            bse_tda=self.bse_tda, mf=mf, environment=self.environment,
            numerics=self.settings, carried=self.carried_grids(window),
            factorization=factorization,
            qp_partition=None if reselect else view.qp_partition)
        if not reselect:
            chain.conventions.qp_selection = self.selection.selection_record()
        elif floor:
            chain.conventions.qp_floor = tuple(sorted(floor))
        # The shifts are not carried: this geometry calibrates its own, on its
        # own explicit roots. The old ones ride along only so the record can
        # say how far the frozen convention moved.
        chain.conventions.outside_shift_before = carried[
            'outside_shift_before']
        chain.conventions.outside_curve_before = carried[
            'outside_curve_before']
        # THE FOLLOWED STATE, NOT ITS INDEX: the new surface's first
        # evaluation matches this one's committed eigenvector by overlap, so
        # at a near-degenerate pair it cannot start on the other state, and
        # its follow_log says how well the two surfaces' states agree.
        if self.following() is not None:
            chain.hand_over(self.following())
        # what shared a quasiparticle Hamiltonian here still does
        chain.conventions.carried_subspaces = carried['subspaces']
        return chain

    def carried_grids(self, window=None):
        """The grids this chain resolved, which a refrozen or restored chain
        starts from: the tau count of W, the contour-deformation size it grew
        to (a floor) -- a refrozen chain never steps back to a quadrature
        found too coarse -- and the explicit set `window`, if any."""
        return CarriedGrids(ntau_w=int(self.ntau_w),
                            nfreq_cd=int(self.conventions.nfreq_cd),
                            window=window)

    def label(self):
        """The state and the method, for a log line or a relaxation record."""
        kernel = 'BSE(TDA)@GW' if self.bse_tda else 'BSE@GW'
        return (f'{kernel} {self.spin} state {self.state} / {self.basis} / '
                f'outside scissor / {self.environment!r}')


def conventions_hash(surface):
    """sha256 of what a force of `surface` is a force of (`force_identity`):
    the conventions, grid, settings and environment of each of its halves."""
    return record_hash(force_identity(surface))


def replayed_first_point(surface, mol):
    """The result of the `FirstPoint` `surface` holds, if it answers `mol`
    and the surface's spin, root and conventions, else None; taken off the
    surface at the first call either way, since it is the start of a walk and
    a later visit to the same geometry is evaluated afresh. A point that does
    not answer is discarded with a warning: its force is evaluated again."""
    first = surface.__dict__.pop('_first_point', None)
    if first is None:
        return None
    mol = surface.mol0 if mol is None else mol
    differs = first.mismatch(mol, surface.spin, surface.state,
                             conventions_hash(surface))
    if not differs:
        return first.result
    warnings.warn(
        f'the first point handed to this surface ({first.spin} root '
        f'{first.state}) is discarded: it differs in {", ".join(differs)} '
        f'from the first force asked of it, which is evaluated afresh')
    return None
