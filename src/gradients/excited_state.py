"""The excited-state potential-energy surface of the cubic BSE@GW gradient.

The gradient chain in `qp_space_time`/`bse_isdf`/`isdf_derivatives` returns
dOmega/dR, the derivative of an excitation energy. A geometry optimization
needs the derivative of the excited state's total energy,

    E_ex(R) = E_0(R) + Omega(R),     dE_ex/dR = dE_0/dR + dOmega/dR

with E_0 the mean-field ground state, so the ground-state gradient is added
here and never inside the chain. In an adiabatic singlet-triplet gap the two
states sit at different minima, so E_0 does not cancel between them.

Frozen conventions: the chain fixes seven discrete choices at the reference
geometry, each a discontinuity in the surface otherwise: the quasiparticle
set, the frame orientation, the interpolation pair layout, the Newton branch
(pole guard and seed), the contour-deformation quadrature (its scale w0 is
read off the reference gap), the residue backend, and the scissor the orbitals
outside the set carry. `refreeze` rebuilds all of them at a new geometry with
the same settings; the optimizer in `src/properties/optimize.py` uses it to
measure how far the walked surface drifted from the one the fit would choose
at the end.

The surroundings enter through the chain's `environment`
(`src.Base.environment`): a polarizable continuum dresses the auxiliary gauge
and adds its static reaction-field operator to the quasiparticle equation,
fixed point charges enter the mean field, the gas phase does nothing. The
gradient asks the environment for its own adjoints; a continuum supplies them,
so a solvated force is analytic for a restricted reference and a molecule. An
environment without a derivative says so through `differentiable`, and the
chain then reports energies and refuses forces.

The Casida step follows `bse.solver_choice` by default (`solver='auto'`):
dense while the (A, B) pair fits in BSE_DENSE_MAX_GB (every root, in a fixed
order, the form a finite-difference check should difference), matrix-free
through the ISDF Davidson (`solve_casida_davidson`) above it, for either spin
(kappa = 0 drops the bare-exchange term from the block action). A named
`solver` is not vetoed by the rule. The adjoint is matrix-free either way and
differentiates the static W the Davidson solved with. `bse_adjoint` names its
realization: 'explicit' (`bse_backward`) contracts the eigenvectors against
the three-index blocks of `bse_cache`, built at the first reverse call off a
forward pass and kept with it, so an energy never pays for them; 'grid'
(`isdf_bse_adjoint`) is the reverse of the ISDF block action and forms no
three-index block. The two agree to rounding, not in the bits.

Sliced factors (`sliced=True`, over more than one rank) reach every stage as
one `SlicedFactors`: the Davidson block action reads its rows, and the other
stages each gather what they read whole once and drop it, so between
geometries a rank holds its grid rows alone and every number is the whole
layout's bit for bit. A continuum that dresses the interaction is refused on
them (Eq. (18) needs the bare gauge beside the dressed one). On the row fit
(`fit='rows'`) the same stages read the rows the fit built: bitwise the same
at every rank count, and another realization (and, where the pair screen
drops a pair, another estimator) than the replicated fit
(`FrozenFactorization`).

One forward, several states. Everything `_forward` computes before the
Casida step (the factors, the static screening, the reaction field, the
quasiparticle set and its tape, the outside scissor) reads neither the spin
nor the root (`_shared_forward`, a `SharedForward`). `spin_view` is a shallow
copy of a chain whose conventions are frozen, with another spin and root, so
a singlet and a triplet, or several roots, are solved and differentiated on
the same objects (`src.gradients.state_manifold.StateManifold.evaluate`).

The adaptive explicit set (`qp_select=QPStates('adaptive')`). `qp_window` is
then the admitted window, the candidates. At the first forward one
analytic-continuation GW over the window (`qp_selection`) decides which of
them are solved explicitly; every other candidate (a hole) borrows the
explicit shift of the solved orbital its frozen tier map names. The
continuation only selects: the energy and the force read the explicit roots
and the frozen map, never an AC number, so the force is the adjoint of the
explicit set as for any other set. The partition is checked against the
production eigenvectors at the reference geometry and frozen for the walk.

Everything computed from this surface (geometry optimization, normal modes,
Huang-Rhys factors, adiabatic gaps, reorganization energies, rates) lives in
`src/properties/` and knows only the `PotentialEnergySurface` protocol.
"""
import copy
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.sliced_factors import GridTileRows, SlicedFactors, whole_factor
from src.Base.constants import (FIT_CHOLESKY_BLOCK, KAPPA,
                                ROOT_FOLLOW_MARGIN_MIN,
                                ROOT_FOLLOW_WEIGHT_MIN)
from src.Base.constants import (BSE_ADJOINTS, BSE_DAVIDSON_CONV_TOL,
                                BSE_DAVIDSON_NROOTS, BSE_FORCE_MAX_CYCLE,
                                BSE_FORCE_RESIDUAL_TOL,
                                BSE_DENSE_MAX_NOV, CASIDA_PHASE_TIE_TOL,
                                CD_NFREQ, CD_NFREQ_SOP, HARTREE_TO_EV,
                                OUTSIDE_TREATMENTS, SOP_N_POLES)
from src.Base.constants import (ADAPTIVE_AC_CALIBRATION_MAX_EV,
                                ADAPTIVE_AC_SHIFT_MAX_EV, ADAPTIVE_HOLE_MAX_EV,
                                ADAPTIVE_MAX_ROUNDS, ADAPTIVE_QP_TOL_MEV,
                                ADAPTIVE_ROOT_MIX_EV, ADAPTIVE_SELECT_MARGIN,
                                HARTREE_TO_MEV)
from src.Base.declaration import Excitation, SurfacePhysics
from src.Base.environment import attached_environment, environment_label
from src.Base.utils.mpi_grid import current_comm, lockstep
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        cd_grid_range)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.qp_selection import (AdaptivePartition,
                                                 ac_quasiparticle_shifts,
                                                 ac_unjudgeable,
                                                 calibrate_ac,
                                                 calibrated_selection,
                                                 capped_why,
                                                 casida_instability,
                                                 degenerate_blocks,
                                                 error_terms,
                                                 first_order_weights,
                                                 frontier_residual,
                                                 hole_errors,
                                                 mandatory_members,
                                                 near_root_targets,
                                                 select_explicit)
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              calibrate_scissor_tiers,
                                              frozen_scissor,
                                              is_quasiparticle_root)
from src.SingleReference.LinearResponse.bse import solver_choice
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_bse_backward_rows, isdf_interstate_backward)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.SingleReference.base import get_occ_virt_indices
from src.gradients.bse_isdf import (bse_backward, bse_blocks, bse_cache,
                                    bse_solve, interstate_backward,
                                    screening_backward)
from src.gradients.factor_chain import FactorChain
from src.gradients.factor_chain import check_scf_quality  # noqa: F401
from src.gradients.reaction_field_adjoint import (reaction_field_backward,
                                                   reaction_field_shift,
                                                   static_screening)
from src.gradients.isdf_derivatives import (GaugeAdjoint, qp_xc_correction,
                                            qp_xc_correction_Y,
                                            qp_xc_correction_skeleton,
                                            xc_hybrid_coeff)
from src.properties.nonadiabatic import (follow_state, mo_overlap,
                                         state_overlap)
from src.gradients.qp_space_time import (carried_analytically,
                                         qp_gradient_space_time,
                                         qp_set_gradient, static_term)
from src.gradients.space_time_adjoint import (chi0_backward,
                                              chi0_backward_rows)


class ForwardPieces(tuple):
    """The seventeen pieces of `ExcitedStateChain._forward`, with the static
    correction <p|Sigma_x - v_xc + Sigma^env|p> the quasiparticle solve read
    on the set beside them (`xc_correction`, None at the mean field), so the
    reverse solve reads the forward's own bits."""

    def __new__(cls, pieces, xc_correction=None):
        out = super().__new__(cls, pieces)
        out.xc_correction = xc_correction
        return out


@dataclass(eq=False)
class SharedForward:
    """What one geometry's forward pass holds before the Casida step: the
    factors, eps, eps^QP, the static W_aux, mu, the bare factor, the Eq. (18)
    shift and its screening pair, the quasiparticle set's tape and the static
    correction it read. No spin and no root enters any of them."""

    mol: object
    mf: object
    auxmol: object
    crd: object
    x_mo: object
    d: object
    eps: np.ndarray
    eps_qp: np.ndarray
    w_aux: np.ndarray
    mu: float
    d_bare: object
    shift: object
    screening: object
    qp_tape: object
    xc_correction: object

    def pieces(self, cache, xn, yn):
        """The `ForwardPieces` of one Casida solve on this forward."""
        return ForwardPieces(
            (self.mol, self.mf, self.auxmol, self.crd, self.x_mo, self.d,
             self.eps, self.eps_qp, self.w_aux, cache, xn, yn, self.mu,
             self.d_bare, self.shift, self.screening, self.qp_tape),
            self.xc_correction)

    def release(self):
        """Drop the quasiparticle tape's proj(tau) rows and slices."""
        if self.qp_tape is not None:
            self.qp_tape.release()


@dataclass(eq=False)
class FirstPoint:
    """(dE/dR, E, diagnostics) of one state already evaluated at one
    geometry, handed to the surface that walks from there: its first
    `total_gradient` at that geometry, for that spin and root, is this one
    (`replayed_first_point`), so a T1 relaxation started where the S1 force
    was taken does not evaluate the T1 force there again."""

    coords: np.ndarray
    charges: np.ndarray
    spin: str
    state: int
    result: tuple

    def answers(self, mol, spin, state):
        """Whether this is the point asked for: same nuclei, spin and root."""
        return (spin == self.spin and int(state) == self.state
                and np.array_equal(mol.atom_charges(), self.charges)
                and np.array_equal(mol.atom_coords(), self.coords))


class ExcitedStateChain(FactorChain):
    """Frozen conventions, and the energy and gradient that follow from them.

    `scf_factory(mol) -> mf` is how a displaced geometry gets a mean field; it
    must converge the orbital gradient to ~1e-11 (see `check_scf_quality`).

    `solver='auto'` is `bse.solver_choice`'s memory rule (resolved in
    `solver_used`): dense while the Casida pair (A, B) fits in
    BSE_DENSE_MAX_GB, matrix-free above it. A pair-count rule would pick dense
    for every Tamm-Dancoff problem; at the sizes of a finite-difference gate
    the memory rule resolves to dense anyway.

    `residue_route='explicit'` takes the real-axis residues of Sigma^c as they
    stand: a residue backend decided per state, or per geometry, is not one
    surface.

    `bse_adjoint` realizes the Hellmann-Feynman adjoint of the root
    (BSE_ADJOINTS): 'explicit' through the (naux, nocc, nvir) three-index
    blocks of `bse_cache`, built at the first reverse call; 'grid' closed over
    the ISDF grid (`isdf_bse_adjoint`), which forms none. The forward pass
    does not read it, and the two forces agree to rounding.

    `qp_window=2` is the frontier set, QPStates('frontier', 2): the two
    occupied and two virtual orbitals around the gap, widened over degenerate
    blocks.

    `outside` sets what the orbitals with no quasiparticle equation of their
    own carry on the BSE diagonal: 'mean-field' leaves each at its eigenvalue,
    'scissor' adds the frozen shift `calibrate_scissor` reads off the
    explicitly solved roots at the reference geometry. That shift is a
    constant, so d eps^QP_p/dR = d eps_p/dR for those orbitals. The `scissor`
    keyword is different: a tier for the states inside the set whose pole
    model is inadmissible.

    `qp_select=QPStates('adaptive', ...)` selects the explicit set out of
    the candidates `qp_window` names at the first forward (module docstring);
    `qp_partition` is a partition already frozen -- by `refreeze`, or by an
    earlier stage of the same run -- and selects nothing. Either needs the
    scissor outside the set: the holes carry it.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole
    and the kernels divide the M^2 sweeps (the tau points of proj(tau) and of
    the quasiparticle adjoint, the contour-deformation frequencies, the rows
    of Zt in the Davidson block action), handing every rank the same bits.
    The static-W adjoint (`chi0_backward_rows`) and the quasiparticle reverse
    sweep run tile-major, each rank holding the factors' adjoints by its grid
    tiles (`_adjoint_rows`). The nuclear assembly ends in the `lockstep` of
    `FactorChain.nuclear_gradient`, and the mean-field force is
    `FactorChain.mean_field_gradient`'s, so both forces are rank 0's on every
    rank.
    """

    READS_SLICED_FACTORS = True
    READS_BARE_GAUGE = True

    def __init__(self, mol, scf_factory, spin='singlet', state=0, bse_tda=False,
                 track=None,
                 basis=None, auxbasis=None, counts=None, n_start=1,
                 qp_window=2, degeneracy_tol=1e-4, ntau_gw=24, ntau_w=None,
                 nfreq_cd=None, e_min_below_gap=None, frames='frozen',
                 residue_route='explicit', tile_gb=None, at_mean_field=False,
                 scissor=None, outside='mean-field', n_poles=SOP_N_POLES,
                 sop_stride=None,
                 solver='auto', dense_max_nov=BSE_DENSE_MAX_NOV,
                 nroots=BSE_DAVIDSON_NROOTS, bse_conv_tol=BSE_DAVIDSON_CONV_TOL,
                 mf=None, environment=None, factorization=None, radii=None,
                 sliced=None, fit=None, fit_block=None,
                 bse_adjoint='explicit', qp_select=None, qp_partition=None):
        super().__init__(mol, scf_factory, basis=basis, auxbasis=auxbasis,
                         counts=counts, n_start=n_start, frames=frames, mf=mf,
                         environment=environment, factorization=factorization,
                         radii=radii, sliced=sliced, fit=fit,
                         fit_block=fit_block)
        self.spin, self.state, self.bse_tda = spin, state, bse_tda
        # `track='overlap'` follows the state; None follows the index.
        self.track = track
        self._followed = None          # (mol, mo_coeff, X, Y) of the last step
        self._anchor = None            # ... and of the first, for drift
        self.follow_log = []
        self.at_mean_field = at_mean_field
        self.residue_route, self.tile_gb = residue_route, tile_gb
        # Frozen quasiparticle shifts for the states Eq. (27) excludes, so the
        # pole route does not fall back to a real-axis solve for them. Frozen
        # here and never re-derived: a tier boundary decided per geometry steps
        # the surface the way a per-geometry residue set does.
        self.scissor = scissor
        self.n_poles = n_poles
        self.sop_stride = sop_stride
        # `scissor='calibrate'` asks for the tier to be built at the reference
        # geometry: the states Eq. (27) excludes are solved properly there by
        # the fallback route, and their converged roots are the calibration.
        # Uncalibrated (a shift of zero) costs meaningfully more in both the
        # energy and the force.
        self.scissor_map = {}
        if outside not in OUTSIDE_TREATMENTS:
            raise ValueError(
                f'outside={outside!r} not in {OUTSIDE_TREATMENTS}: an orbital '
                f'with no quasiparticle equation of its own either keeps its '
                f"mean-field eigenvalue ('mean-field') or carries the frozen "
                f"shift calibrated on the explicit roots ('scissor')")
        if outside == 'scissor' and at_mean_field:
            raise ValueError(
                "outside='scissor' calibrates its shifts on explicitly solved "
                'quasiparticle roots, and at_mean_field=True solves none')
        self.outside = outside
        # The frozen scissor of the orbitals outside the set: None until the
        # first forward calibrates it, a {orbital: shift} mapping afterwards,
        # which is what `frozen_scissor` reads.
        self.outside_shift = None
        # {probe: the shift it lends}, frozen with `outside_shift`.
        self.outside_lent = None
        # Whether a forward pass has run and frozen the Newton branch.
        self.forward_ran = False
        # What the chain this one was refrozen from had calibrated, so the
        # record can say how far the convention moved with the geometry.
        self.outside_shift_before = None
        self.is_ks = xc_hybrid_coeff(self.mf0)[0]
        if solver not in ('dense', 'davidson', 'auto'):
            raise ValueError(f"solver {solver!r}: 'dense', 'davidson' or 'auto'")
        if solver == 'davidson' and bse_tda:
            raise NotImplementedError('the ISDF Davidson route solves the full '
                                      'BSE only, not the Tamm-Dancoff form; '
                                      'use solver=\'dense\'')
        self.solver, self.dense_max_nov = solver, dense_max_nov
        if bse_adjoint not in BSE_ADJOINTS:
            raise ValueError(f'bse_adjoint={bse_adjoint!r} not in '
                             f'{BSE_ADJOINTS}')
        self.bse_adjoint = bse_adjoint
        self.nroots, self.bse_conv_tol = nroots, bse_conv_tol
        # each Davidson's {'stats', 'timings'}, kept while a stage timer is set
        self.davidson_solves = []
        # Kept verbatim so `refreeze` can rebuild the same surface elsewhere:
        # a setting that is re-derived instead of preserved makes the refrozen
        # surface a different one, and its energy incomparable with this one's.
        self.qp_window = qp_window
        self.degeneracy_tol, self.e_min_below_gap = degeneracy_tol, e_min_below_gap
        self.ntau_gw = ntau_gw
        # None: a set with a state on the quadrature takes CD_NFREQ and a
        # pole-model set CD_NFREQ_SOP (`_fix_cd_grid`); a size asked for is a
        # floor under both
        self._nfreq_cd_floor = 0 if nfreq_cd is None else int(nfreq_cd)
        self.nfreq_cd = CD_NFREQ if nfreq_cd is None else int(nfreq_cd)

        eps = np.asarray(self.mf0.mo_energy, float)
        occ, virt = get_occ_virt_indices(eps, self.nocc)
        gap = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        self.gap = gap
        self._build_cd_grid(self.nfreq_cd, eps)
        self.ntau_w = (minimax_points_for_accuracy(gap, e_max,
                                                   target=DEFAULT_TAU_TARGET)[0]
                       if ntau_w is None else int(ntau_w))
        self.w_grid = TimeFrequencyGrid.minimax_split(
            self.ntau_w, gap, e_max, [0.0], [1.0],
            with_sine=False, with_inverse=False)
        self.qp_set = self._qp_set(eps, self.nocc, qp_window, degeneracy_tol)
        # The orbitals the declared set named whose reference-geometry root
        # was no quasiparticle (`_settle_qp_set`), each with why and what the
        # rejected root was; they carry the outside treatment instead. The
        # set is settled by the first solve of it and frozen from then on.
        self.qp_demoted = {}
        self.qp_set_settled = False
        # The Newton's pole guard per orbital, filled by the first
        # quasiparticle solve and held from then on. A root inside the guard
        # makes the iteration shrink it, and an offset decided per geometry is
        # a piece of the Newton branch that moves under the optimizer;
        # `refreeze` re-derives it at the new geometry like the rest.
        self.pole_offsets = {}
        # The reference geometry's converged roots, handed back as the Newton
        # seed at every displaced geometry so that the iteration lands on the
        # branch this surface was built on rather than on whichever one the
        # mean-field eigenvalue is nearest to.
        self.qp_seeds = {}
        # The pole model's positions per orbital, fitted at the reference
        # geometry and held: the SOP adjoint differentiates the amplitudes at
        # fixed poles, so a set re-fitted per geometry puts the poles' motion
        # into the energy and not into the force.
        self.sop_poles = {}
        # Whether the contour-deformation quadrature has been fixed for the
        # set (`_fix_cd_grid`); that happens on the first quasiparticle solve
        # and is then frozen with everything else.
        self.cd_sized = False
        # Who owns the adaptive selection's working state: a token per chain,
        # renewed by every shallow copy (`__copy__`), so a spin view never
        # refines the selection it shares and a freed chain's identity, reused
        # by another object, cannot claim it.
        self._selection_owner = object()
        self._init_adaptive(qp_select, qp_partition)

    def __copy__(self):
        """A shallow copy that owns no selection: every attribute shared by
        reference, its own `_selection_owner`."""
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out._selection_owner = object()
        return out

    def _owns_selection(self, sel):
        """Whether the selection state `sel` was started by this chain."""
        return sel is not None and sel['owner'] is self._selection_owner

    def _init_adaptive(self, qp_select, qp_partition):
        """The adaptive explicit set's state: what selects, what was selected.

        qp_partition, when given, is the set already: `qp_set` becomes its
        explicit orbitals and nothing is selected. Otherwise `qp_set` holds the
        candidates until the first forward selects.
        """
        self.qp_select = qp_select
        self.qp_partition = qp_partition
        # the quadrature asked for, which a re-solve of the selection starts
        # from again
        self._nfreq_cd_asked = self.nfreq_cd
        # the record of how the partition was chosen (`selection_record`)
        self.qp_selection = None
        # the selection's working state at the reference geometry
        self._selecting = None
        if qp_select is None and qp_partition is None:
            return
        if qp_select is not None and qp_select.kind != 'adaptive':
            raise ValueError(f"qp_select is a QPStates(kind='adaptive'), not "
                             f'{qp_select!r}')
        if self.at_mean_field or self.outside != 'scissor':
            raise ValueError(
                "the adaptive explicit set needs outside='scissor' and a "
                'quasiparticle solve: the orbitals it leaves unsolved carry a '
                'scissor calibrated on the ones it solves')
        if qp_partition is not None:
            if tuple(int(p) for p in self.qp_set) != qp_partition.candidates:
                raise ValueError(
                    'a frozen partition belongs to its candidates: qp_window '
                    f'{[int(p) for p in self.qp_set]} is not '
                    f'{list(qp_partition.candidates)}')
            self.qp_set = np.asarray(qp_partition.explicit, int)
            self.qp_selection = dict(qp_partition.provenance) or None

    # ------------------------------------------------------------ declaration
    @property
    def physics_ground_state(self):
        """E_0 = E_KS[xc], the mean field's own energy.

        `total_energy` is `mf.e_tot + Omega` and carries no correlation term of
        its own, so the functional under the state is the mean field's,
        whatever that mean field is: GroundState('dft', xc) for a Kohn-Sham
        factory and GroundState('dft', 'hf') for Hartree-Fock. The dRPA ground
        state of Toelle Eq. (15) is a different surface and is composed in
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
        if self.at_mean_field:
            raise ValueError(
                'at_mean_field=True puts the MEAN-FIELD eigenvalues on the BSE '
                'diagonal, and `Excitation` has no vocabulary for that: its '
                "`qp` is a quasiparticle method, and declaring 'g0w0' for a "
                'surface that solves no quasiparticle equation would name a '
                'functional this chain does not compute.')
        return Excitation(spin=self.spin, root=self.state + 1,
                          kernel='bse-tda' if self.bse_tda else 'bse')

    @property
    def physics(self):
        """What this chain computes: E_KS + Omega, in its own environment.

        Declared here rather than stamped from outside: a surface built
        through `potential_energy_surface` is checked against it, so a chain
        built with settings the caller did not ask for is refused rather than
        mislabelled.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.environment))

    # ------------------------------------------------------------------ setup
    def _build_cd_grid(self, nfreq_cd, eps):
        """The contour-deformation quadrature and the imaginary-time grid whose
        frequency axis it is, at `nfreq_cd` points.

        `cd_frequency_grid` builds it, so a gradient walks the quadrature an
        energy route integrates on, including the refusal when
        e_min_below_gap eats the whole gap.

        ntau_gw is 24 rather than the 18 the imaginary axis alone needs: the
        cosine transform wants the grid's 1/y fit on [gap, e_max], the cosh
        transform of a residue at y = d -/+ w', down to gap - w', which is why
        e_min sits below the gap. On benzene/cc-pVDZ with e_min = 0.5 gap,
        every DFT starting point fails the Laplace backend's representation
        gate at 18 points and passes at 24; Hartree-Fock passes either way
        (smaller quasiparticle corrections, hence smaller residue
        frequencies).
        """
        self.nfreq_cd = int(nfreq_cd)
        # the range the tau axis is fitted on, recorded beside the grid it
        # built: an audit reads the range a surface was frozen at
        self.e_min_gw, self.e_max = cd_grid_range(eps, self.nocc,
                                                  self.e_min_below_gap)
        self.nu, self.wt, self.gw_grid, self.w0_cd = cd_frequency_grid(
            eps, self.nocc, ntau=self.ntau_gw, nfreq_cd=self.nfreq_cd,
            e_min_below_gap=self.e_min_below_gap)

    def _fix_cd_grid(self, states, eps):
        """Fix the contour-deformation grid for the set, once: `CD_NFREQ_SOP`
        points when the pole model carries it whole, `CD_NFREQ` otherwise,
        never below an `nfreq_cd` the caller asked for (a refrozen chain does
        not step back).

        The grid is never resized against the roots: the Lorentzian a pole of
        G puts on the imaginary-frequency integrand at omega = eps_q is
        integrated in closed form (`contour_deformation.cd_integral_weights`),
        so a root beside an orbital energy asks nothing more of the
        quadrature than any other. A set the pole model carries whole has a
        Sigma analytic in omega, and its grid only feeds the fits, which
        converge at fewer points.
        """
        if self.cd_sized:
            return
        kw = self._qp_kw()
        analytic = carried_analytically(
            states, eps, self.nocc, kw['residue_route'], kw['scissor'],
            kw['w0'], kw['pole_offset'], kw['sop_poles'])
        nfreq = max(CD_NFREQ_SOP if analytic else CD_NFREQ,
                    self._nfreq_cd_floor)
        if self.nfreq_cd != nfreq:
            self._build_cd_grid(nfreq, eps)
        self.cd_sized = True

    @staticmethod
    def _qp_set(eps, nocc, window, tol):
        """The quasiparticle set, widened so no degenerate block is split.

        Inside a degenerate subspace the eigenvectors are not determined, so an
        individual orbital is not a smooth function of the geometry; giving one
        half of a pair a quasiparticle energy and leaving the other at the mean
        field differentiates across an arbitrary rotation.

        `window` is a half-width, 'all', or a sequence of orbital indices:
        the set a `QPStates` declaration resolved to, which is already the
        whole answer and is taken as given rather than widened again.
        """
        if str(window).lower() == 'all':
            return np.arange(len(eps))
        if np.ndim(window) == 1:
            return np.sort(np.asarray(window, int))
        w = int(window)
        lo, hi = max(nocc - w, 0), min(nocc + w, len(eps))
        while lo > 0 and eps[lo] - eps[lo - 1] < tol:
            lo -= 1
        while hi < len(eps) and eps[hi] - eps[hi - 1] < tol:
            hi += 1
        return np.arange(lo, hi)

    def _qp_kw(self):
        kw = {'residue_route': self.residue_route,
              'pole_offset': self.pole_offsets,
              'w0': self.qp_seeds,
              'sop_poles': self.sop_poles,
              'n_poles': self.n_poles,
              'sop_stride': self.sop_stride,
              'scissor': self.scissor_map or self.scissor}
        if self.tile_gb is not None:
            kw['tile_gb'] = self.tile_gb
        return kw

    def _freeze_newton_branch(self, route_out, eps, xc_correction, states):
        """Keep the guard band, the root and the pole set each quasiparticle
        solve resolved to, once. True when the solve has to be repeated: a
        root was frozen here for the first time, or a calibrated shift was
        added and is read by the next solve.

        The first solve is the reference geometry's in every path that then
        displaces it, so `setdefault` freezes that one: a displaced geometry
        reports what it used, starts from the reference root, evaluates the
        pole model on the reference poles and changes none of them.

        THE REFERENCE ROOT IS THE ONE ITS FROZEN SEED REACHES. The first
        solve starts at eps_p pushed by the guard; every later one at this
        geometry starts at the root it froze, and a Newton started elsewhere
        lands an ulp away. Repeating the freezing solve from its own seed,
        guard and poles makes every evaluation at the reference geometry the
        same arithmetic, forward and reverse.

        eps, xc_correction, states: what the solve read, xc_correction per
        state of `states` (or one scalar), so the calibrated shift is taken
        off the same numbers the 'scissor' route adds it back to.

        The calibrated shift is Sigma_c,pp(w_p) alone: the 'scissor' route
        returns eps_p + xc_p + s_p with the static term
        xc_p = <p|Sigma_x - v_xc|p> + Sigma^env_pp, and the root solved here is
        w_p = eps_p + xc_p + Sigma_c,pp(w_p), so s_p = w_p - (eps_p + xc_p)
        puts the state on its own root at this geometry.
        """
        for p, off in route_out.get('pole_offsets', {}).items():
            self.pole_offsets.setdefault(int(p), float(off))
        roots = route_out.get('roots', {})
        froze = any(int(p) not in self.qp_seeds for p in roots)
        for p, w in roots.items():
            self.qp_seeds.setdefault(int(p), float(w))
        for p, poles in route_out.get('sop_poles', {}).items():
            self.sop_poles.setdefault(int(p), np.array(poles, float))
        if self.scissor != 'calibrate':
            return froze
        # Tier the states the route sent to the real axis. A second reading of
        # Eq. (27) would test the root where the route tested the start, and
        # disagree on the marginal states.
        static = {int(p): static_term(xc_correction, si)
                  for si, p in enumerate(np.atleast_1d(states))}
        grew = False
        for p, route in self._routes_taken(route_out).items():
            if route in ('sop', 'scissor') or p in self.scissor_map:
                continue
            self.scissor_map[p] = float(roots[p]) - float(eps[p] + static[p])
            grew = True
        return grew or froze

    def _settle_qp_set(self, route_out, states):
        """Reject the roots of the set that are no quasiparticle: the mask of
        the states kept if any was rejected, else None.

        A declaration resolves to orbital indices before anything is solved,
        so the pole strength Z of each root is first known here. A root with Z
        outside (0, 1] (`is_quasiparticle_root`) sits on a branch only a
        negative weight makes (the pole model reaches one through a fitted
        amplitude), and a BSE built on it has the energy and the force of a
        state that does not exist. Its orbital leaves the set and carries the
        outside treatment (the frozen scissor, calibrated on the roots that
        remain), and `qp_demoted` records the orbital, the reason, the route
        and the rejected root and Z.

        Decided on the first solve of the set, which is the reference
        geometry's, and then frozen with the rest of the Newton branch: a
        displaced geometry that demoted would move an orbital between two
        treatments along the walk and step the surface, so a rejected root
        there is refused instead. The pole strengths are rank 0's on every
        rank (`lockstep`), so every rank keeps the same set. The singular part
        of the contour deformation is integrated in closed form
        (`cd_integral_weights`), so the slope, and Z, is the self-energy's own
        on the plain grid.
        """
        z_of = route_out.get('z')
        if z_of is None:
            return None
        z_of = lockstep(np.array(z_of, float))
        rejected = {int(p): float(z) for p, z in zip(states, z_of)
                    if not is_quasiparticle_root(z)}
        roots = route_out.get('roots', {})
        if not rejected:
            return None
        if self.qp_set_settled:
            at = {p: roots.get(p) for p in rejected}
            raise RuntimeError(
                f'quasiparticle roots with a pole strength outside (0, 1] at '
                f'a displaced geometry, orbital: Z {rejected}, roots (Ha) '
                f'{at}. The set was '
                f'settled at the reference geometry and is frozen, so the '
                f'orbital cannot be moved to the scissor here; refreeze the '
                f'surface at this geometry to settle the set again.')
        routes = self._routes_taken(route_out)
        for p, z in rejected.items():
            self.qp_demoted[p] = {'reason': 'pole strength Z outside (0, 1]',
                                  'z': z, 'root': float(roots[p]),
                                  'route': routes.get(p)}
        keep = np.array([int(p) not in rejected for p in states])
        self.qp_set = np.asarray(states)[keep]
        warnings.warn(
            f'quasiparticle roots with a pole strength outside (0, 1], '
            f'orbital: Z {rejected}. Those orbitals leave the explicit set '
            f'and carry the {self.outside} treatment instead.',
            RuntimeWarning, stacklevel=3)
        return keep

    @staticmethod
    def _routes_taken(route_out):
        """{orbital: route} from either route_out shape.

        A set solve reports 'routes' per orbital; a single-orbital solve
        reports one 'residue_route' and one root.
        """
        routes = route_out.get('routes')
        if routes is not None:
            return {int(p): r for p, r in routes.items()}
        route = route_out.get('residue_route')
        return {int(p): route for p in route_out.get('roots', {})}

    def _qp_set_solve(self, x_mo, d_sigma, eps, mu, xc_correction, weights,
                      states=None, tape=None, rows_block=None):
        """((eps^QP values, adjoints), route_out) for a quasiparticle set, on
        the contour-deformation grid fixed for it before the first pass
        (`_fix_cd_grid`).

        tape: an earlier solve's `QPSetTape` on the same factors, and each
        repeat reads the one before it, so proj(tau) and the slices are
        swept once however often the solve repeats (`QPSetTape.reads`).
        rows_block: the adjoints in grid tiles over the ranks
        (`qp_set_gradient`).

        A solve of the whole set (states=None) settles it
        (`_settle_qp_set`): an orbital whose root is no quasiparticle leaves
        the set and the solve is repeated without it. Its weight and static
        correction leave with it; route_out['xc_correction'] is the correction
        the returned roots were solved with.
        """
        whole_set = states is None
        states = self.qp_set if whole_set else states
        self._fix_cd_grid(states, eps)
        while True:
            route_out = {}
            out = qp_set_gradient(x_mo, d_sigma, eps, self.nocc, self.gw_grid,
                                  self.nu, self.wt, states, weights,
                                  mu=mu, xc_correction=xc_correction,
                                  route_out=route_out, tape=tape,
                                  rows_block=rows_block, **self._qp_kw())
            tape = route_out.get('tape', tape)
            if whole_set:
                keep = self._settle_qp_set(route_out, states)
                if keep is not None:
                    states = self.qp_set
                    if tape is not None:
                        tape = tape.restricted(keep)
                    weights = np.asarray(weights)[keep]
                    if np.ndim(xc_correction):
                        xc_correction = np.asarray(xc_correction)[keep]
                    continue
            # A root frozen here, or a calibrated shift, is read by the solve
            # after the one that froze it, so either triggers one repeat.
            if not self._freeze_newton_branch(
                    route_out, eps, xc_correction, states):
                break
        if whole_set:
            self.qp_set_settled = True
        if route_out.get('tape') is not None:
            route_out['tape'].drop_residues()
        route_out['xc_correction'] = xc_correction
        # What the last solve took, state by state, for the record: the route
        # and the pole strength Z, which the route does not keep on the chain.
        # A solve that reports no Z (a stand-in route) records None for it.
        z_of = route_out.get('z')
        self.qp_diagnostics = {
            'routes': self._routes_taken(route_out),
            'z': None if z_of is None else
            {int(p): float(z) for p, z in zip(states, z_of)}}
        return out, route_out

    def _tile_kw(self):
        return {} if self.tile_gb is None else {'tile_gb': self.tile_gb}

    def solver_used(self, n_ov):
        """'dense' or 'davidson' for a pair space of n_ov, resolving 'auto'
        through `bse.solver_choice`: dense while the Casida pair (A, B) fits in
        BSE_DENSE_MAX_GB (the Tamm-Dancoff form is not exempt; the dense route
        builds both blocks either way). `dense_max_nov` is an extra cap that
        may only tighten the rule; at its default it is the rule's own
        boundary.

        A Tamm-Dancoff problem above the rule is refused:
        `solve_casida_davidson` solves the full Casida problem and takes no
        `tda`, and staying dense pays the memory the rule refuses. Triplets do
        go through Davidson (kappa = 0 removes the bare-exchange term from the
        block action). The refusal is here, where the pair count is known, so
        'auto' is not rejected at sizes where it picks dense.
        """
        if self.solver != 'auto':
            return self.solver
        choice = solver_choice(n_ov, self.bse_tda)
        if choice == 'dense' and n_ov > self.dense_max_nov:
            choice = 'davidson'
        if choice == 'davidson' and self.bse_tda:
            raise NotImplementedError(
                f'the Tamm-Dancoff form has no matrix-free route: '
                f'`solve_casida_davidson` solves the full Casida problem and '
                f'takes no tda, and the dense (A, B) pair at n_ov={n_ov} is '
                f'{2 * int(n_ov)**2 * 8 / 1e9:.1f} GB, which the memory rule '
                f'refuses. Declare the full kernel, or raise '
                f'BSE_DENSE_MAX_GB knowing what it costs.')
        return choice

    # -------------------------------------------------------- forward/reverse
    def _outside_window(self, norb):
        """Mask of the orbitals with no quasiparticle equation of their own:
        everything outside the window, and everything when `at_mean_field`."""
        mask = np.ones(norb, bool)
        if not self.at_mean_field:
            mask[self.qp_set] = False
        return mask

    def _freeze_outside_shift(self, eps, ws, env=None):
        """Calibrate the scissor the orbitals outside the set carry, once.

        `calibrate_scissor` gives every outside orbital the quasiparticle
        correction of the explicitly solved orbital nearest it in energy, so
        occupied orbitals take an occupied probe's shift and virtual ones a
        virtual probe's. An error on a core state moves a frontier
        quasiparticle by a fraction of a meV, the inner valence by orders of
        magnitude more per eV.

        Frozen at the first forward (the reference geometry's) and held, like
        the Newton branch and the quadrature: a shift re-derived per geometry
        steps the surface wherever the nearest probe changes and would put a
        d(shift)/dR into a force that has no term for it. Its constancy is
        what lets `_fold_to_nuclei` send an outside orbital's seed straight to
        the mean-field eigenvalue.

        In a continuum the shift is the probe's GW correction alone: the
        probe's root solves w = eps_q + <q|Sigma_x - v_xc|q> + Sigma_c,qq(w)
        + Sigma^env_qq, and every orbital outside the set gets its own
        Sigma^env_pp on the diagonal (`_env_static_outside`), so the scissor
        transfers w - eps_q - Sigma^env_qq (Duchemin, Jacquemin and Blase,
        J. Chem. Phys. 144, 164106 (2016), Eq. (18)). In the gas phase `env`
        is None and the shift is the root minus eps.

        A hole of an adaptive set (a candidate left unsolved) takes its frozen
        probe's shift instead, without the probe's own Eq. (18) term.
        """
        if self.outside_shift is not None:
            return
        roots = {int(p): float(w) for p, w in zip(self.qp_set, ws)}
        free = self._env_free(roots, env)
        # the shift each probe lends, which is how the record names it
        self.outside_lent = {q: w - float(eps[q]) for q, w in free.items()}
        outside = np.flatnonzero(self._outside_window(len(eps)))
        part = self.qp_partition
        if part is None or not part.tier_of:
            self.outside_shift = calibrate_scissor(eps, self.nocc, free,
                                                   outside)
            return
        # Beyond the window the nearest explicit orbital, which is a window
        # edge; inside it each hole's frozen probe (`calibrate_scissor_tiers`).
        holes = {p: b for p, b in part.tier_of.items() if p in set(outside)}
        shift = calibrate_scissor(eps, self.nocc, free,
                                  [p for p in outside if int(p) not in holes])
        shift.update(calibrate_scissor_tiers(eps, free, holes))
        self.outside_shift = shift

    def outside_record(self):
        """What the orbitals outside the quasiparticle set carry, in eV.

        The shifts themselves, and after a `refreeze` the ones the geometry
        this surface was rebuilt from had: the drift of a frozen convention is
        what says whether the surface an optimizer walked is the one the
        calibration would choose at the end of the walk.
        """
        out = {'outside': self.outside}
        if self.outside_shift is None:
            return out
        now = {int(p): float(s) * HARTREE_TO_EV
               for p, s in sorted(self.outside_shift.items())}
        out['outside_shift_ev'] = now
        if self.outside_shift_before is not None:
            before = {int(p): float(s) * HARTREE_TO_EV
                      for p, s in sorted(self.outside_shift_before.items())}
            out['outside_shift_before_ev'] = before
            out['outside_shift_moved_ev'] = max(
                (abs(now[p] - before[p]) for p in now if p in before),
                default=0.0)
        return out

    def _env_static_outside(self, shift, norb):
        """<p|Sigma^env|p> on the orbitals the quasiparticle window leaves out,
        zero inside it (where `_xc_correction` already carries it).

        The window approximates GW, not a reaction field. Leaving an orbital
        at its mean-field eigenvalue is defensible for
        Sigma_x - v_xc + Sigma_c, which varies smoothly across the spectrum
        and largely cancels in eps_a - eps_i. Sigma^solv is ~ +lambda/2 on
        every occupied and ~ -lambda/2 on every virtual orbital, and the
        direct kernel's vtilde reaches every pair; the two cancel for a
        compact electron-hole pair only if both reach the same orbitals.
        Without this term a pair outside the window keeps a whole lambda of
        spurious blue shift, and those are the local roots a continuum should
        barely move (on C2H4...F2/cc-pVDZ at eps = 2.5, window 2, the lowest
        local root changes sign and magnitude without it).

        Eq. (18) is one vector over the whole spectrum, so this is free. Gas
        phase passes None and the diagonal is untouched, bitwise.
        """
        if shift is None:
            return None
        return np.where(self._outside_window(norb), np.asarray(shift, float),
                        0.0)

    def _xc_chain(self, mf, weights_full, env_weights=None,
                  on_the_factors=False):
        """(Y_extra, skeleton) of sum_p w_p <p|Sigma_x - v_xc|p>
        + sum_p w^env_p <p|Sigma^env|p>.

        On a Kohn-Sham reference the static correction is part of the
        quasiparticle energy, so its dependence on the orbitals and nuclei is
        carried: the Y piece before the multiplier solve (it shares Lambda),
        the skeleton in the assembly. The gate is on Sigma_x - v_xc alone,
        which vanishes on Hartree-Fock.

        The environment's static term rides the same slot with its own two
        pieces (`static_self_energy_adjoint`): zero for the gas phase and
        fixed charges, the reaction field's orbital response and skeleton for
        a continuum, a refusal for an unrestricted reference.

        The weights can differ: outside the quasiparticle window
        `_env_static_outside` puts Sigma^env on the BSE diagonal with no
        Newton equation to renormalize it, so those orbitals carry their bare
        adjoint in `env_weights` and zero in `weights_full`. By default the
        two coincide (the single-quasiparticle case).

        on_the_factors: the continuum rides Eq. (18), whose adjoint reaches
        the nuclei through (eps, X, D) in the caller; the COHSEX operator's
        own adjoint must not be added too, or the reaction field enters the
        force twice.
        """
        if on_the_factors:
            if not self.is_ks:
                return None, np.zeros((mf.mol.natm, 3))
            return (qp_xc_correction_Y(mf, weights_full, self.nocc),
                    qp_xc_correction_skeleton(mf, weights_full, self.nocc))
        if env_weights is None:
            env_weights = weights_full
        y_env, g_env = self.environment_at(mf.mol).static_self_energy_adjoint(
            mf, env_weights)
        if not self.is_ks:
            return y_env, g_env
        return (qp_xc_correction_Y(mf, weights_full, self.nocc) + y_env,
                qp_xc_correction_skeleton(mf, weights_full, self.nocc) + g_env)

    def _xc_correction(self, mf, states, reaction_field=None):
        """<p|Sigma_x - v_xc|p> + <p|Sigma^env|p> for `states`, evaluated with
        this geometry's environment attached (the previous attachment is put
        back afterwards).

        Computed on Hartree-Fock too, where the first part is round-off rather
        than an exact zero; the environment's part is first order in vtilde
        and zeroth in v and survives there.
        """
        with attached_environment(mf, self.environment_at(mf.mol)):
            return qp_xc_correction(mf, states, reaction_field=reaction_field)

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

        Built on the same axis as the BSE kernel's static W, so the two
        screenings a solvated run needs come off one grid; the dressed member
        is the kernel's W, so a caller that already has it passes the pair
        rather than rebuilding chi0(0) in the dressed gauge.
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

    def _casida(self, x_mo, d, eps_qp, w_aux):
        """(Omega, X, Y, cache): every root densely, or the lowest `nroots`
        matrix-free on the same static W, sorted.

        The cache is what the explicit adjoint reads. The dense route's blocks
        are the ones its Casida matrices were built from and ride along on the
        explicit route; the matrix-free route returns it empty and
        `_casida_seeds` fills it at the first reverse call, so an energy
        never builds a three-index block. On the grid route it stays empty.
        """
        n_ov = self.nocc * (len(eps_qp) - self.nocc)
        if self.solver_used(n_ov) == 'dense':
            a, b, cache = bse_blocks(x_mo, d, eps_qp, w_aux, self.nocc,
                                     spin=self.spin, bse_tda=self.bse_tda)
            om, xn, yn = casida_phase(*bse_solve(a, b))
            return om, xn, yn, (cache if self.bse_adjoint == 'explicit'
                                else {})
        if self.nroots <= self.state:
            raise ValueError(f'nroots={self.nroots} does not reach state {self.state}')
        lr = LinearResponseSolver(np.asarray(eps_qp, float), spin_mode='restricted')
        # Sliced, the block action reads the rows themselves: D and X_o are
        # gathered once per build, never per trial vector.
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        # Hellmann-Feynman reads the eigenvectors, so an unconverged root is a
        # wrong force rather than a slightly wrong energy: refused, not warned
        # -- but only a root the force reads (the state and those below it,
        # every root when the state is followed by overlap), and only above
        # the residual the force needs, BSE_FORCE_RESIDUAL_TOL.
        solve = ({'stats': {}, 'timings': {}} if self.timer is not None
                 else {'stats': None, 'timings': None})
        if self.timer is not None:
            self.davidson_solves.append(solve)
        om, xn, yn = solve_casida_davidson(
            lr, self.nocc, nroots=self.nroots, polarizability='BSE',
            W_aux=w_aux, isdf_factors=factors, conv_tol=self.bse_conv_tol,
            max_cycle=BSE_FORCE_MAX_CYCLE, spin=self.spin,
            stats=solve['stats'], timings=solve['timings'],
            refuse_unconverged=True,
            read_roots=None if self.track is not None else self.state + 1,
            read_tol=max(self.bse_conv_tol, BSE_FORCE_RESIDUAL_TOL))
        order = np.argsort(om)
        om, xn, yn = casida_phase(om[order], xn[:, order], yn[:, order])
        return om, xn, yn, {}

    def _forward(self, mol, mf):
        shared = self._shared_forward(mol, mf)
        om, pieces = self._casida_forward(shared)
        # an adaptive set is checked on the production eigenvectors at the
        # reference geometry, and re-solved if they say so
        while self.verify_selection(shared, {self.spin: (om, pieces)}):
            shared.release()
            shared = self._shared_forward(mol, mf)
            om, pieces = self._casida_forward(shared)
        return om, pieces

    def _casida_forward(self, shared):
        """(Omega, `ForwardPieces`): this chain's spin solved on `shared`."""
        with self.phase('t_casida'):
            om, xn, yn, cache = self._casida(shared.x_mo, shared.d,
                                             shared.eps_qp, shared.w_aux)
        # The quasiparticle set's tape rides with the pieces: the reverse
        # solve reads proj(tau) and the slices from it instead of sweeping
        # them again, bitwise.
        return om, shared.pieces(cache, xn, yn)

    def kernel_pieces(self, mol, mf):
        """Everything `_forward` builds before the Casida solve, in its layout.

        The same seventeen-long `ForwardPieces` `_forward` returns, with an
        empty cache and no roots (`xn`, `yn` are None): the quasiparticle
        energies, the static screening and the factors the BSE matrix is made
        of. A quantity that needs the kernel but not the supermolecular roots
        -- the diabatic elements of `src.properties.fragment_bse` -- starts
        here, and so does its gradient: `_fold_to_nuclei` reads nothing from
        the roots.
        """
        return self._shared_forward(mol, mf).pieces({}, None, None)

    def _shared_forward(self, mol, mf):
        """The `SharedForward` at one geometry: everything before the Casida
        step, which reads neither `spin` nor `state`."""
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        # One static screening: the BSE kernel's W and the dressed half of
        # Eq. (18) are the same [1 - chi0(0)]^-1 on the same axis; the orbital
        # densities are Eq. (18)'s alone.
        with self.phase('t_screening'):
            dressed = static_screening(x_mo, d, eps, self.nocc, self.w_grid,
                                       densities=d_bare is not None)
            w_aux = dressed[1]
            pair = (None if d_bare is None else
                    (dressed, static_screening(x_mo, d_bare, eps, self.nocc,
                                               self.w_grid)))
        # The self-energy screens bare: Eq. (18) is the static approximation to
        # Sigma[W_solv] - Sigma[W_gas], so screening Sigma dynamically as well
        # counts the reaction field twice; the kernel below keeps D.
        shift, screening = self._reaction_field(x_mo, d, d_bare, eps,
                                                screening=pair)
        d_sigma = d if d_bare is None else d_bare
        qp_tape = xc = None
        if self.at_mean_field:
            eps_qp = eps
        else:
            with self.phase('t_qp'):
                if self.qp_select is not None and self.qp_partition is None \
                        and self._selecting is None:
                    self._select_qp_set(mol, mf, x_mo, d, d_sigma, eps,
                                        w_aux, shift)
                while True:
                    xc = self._xc_correction(mf, self.qp_set, shift)
                    out, route_out = self._qp_set_solve(
                        x_mo, d_sigma, eps, mu, xc, np.zeros(len(self.qp_set)))
                    # the set the roots belong to: settling may have shrunk it
                    xc = route_out['xc_correction']
                    if not self._refine_selection(eps, out[0], shift):
                        break
                    tape = route_out.get('tape')
                    if tape is not None:
                        tape.release()
            ws, qp_tape = out[0], route_out.get('tape')
            eps_qp = eps.copy()
            eps_qp[self.qp_set] = ws
            # Outside the set, the frozen scissor, calibrated on these roots
            # at the reference geometry and spent unchanged afterwards; the
            # environment's static term below is added on top of it, not
            # instead of it.
            if self.outside == 'scissor':
                self._freeze_outside_shift(eps, ws, shift)
                for p in np.flatnonzero(self._outside_window(len(eps))):
                    scissor_p = frozen_scissor(self.outside_shift, p)
                    if scissor_p is not None:
                        eps_qp[p] = eps[p] + scissor_p
        env_outside = self._env_static_outside(shift, len(eps))
        if env_outside is not None:
            eps_qp = eps_qp + env_outside
        self.forward_ran = True
        if self.qp_partition is not None:
            # what the check at a walk's end reads at this geometry
            self.last_forward = {'coords': np.array(mol.atom_coords()),
                                 'eps': eps, 'eps_qp': eps_qp, 'shift': shift}
        return SharedForward(mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux,
                             mu, d_bare, shift, screening, qp_tape, xc)

    @property
    def conventions_frozen(self):
        """Whether the first forward has fixed what every later one reads:
        the Newton branch of the quasiparticle set and, outside the set, the
        scissor calibrated. At the mean field there is nothing to fix."""
        if self.at_mean_field:
            return True
        return (self.forward_ran and (self.outside != 'scissor'
                                      or self.outside_shift is not None)
                and (self.qp_select is None or self.qp_partition is not None))

    def spin_view(self, spin, state=None, track=None):
        """This chain with another spin and root, on the same frozen objects.

        A shallow copy, refused until `conventions_frozen`: `outside_shift`
        is rebound by the first forward, so a copy made earlier would
        calibrate its own scissor. Made afterwards it shares
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
                'freezes (the Newton branch, the outside scissor); '
                'evaluate the chain once before taking one')
        view = copy.copy(self)
        view.spin = spin
        view.state = self.state if state is None else int(state)
        view.track = self.track if track is None else track
        view._followed = view._anchor = None
        view.follow_log = []
        view.davidson_solves = []
        return view

    def spectrum(self, mol=None, mf=None):
        """Every excitation energy the Casida step returned, ascending, in Hartree."""
        mol, mf = self.mean_field(mol, mf)
        return self._forward(mol, mf)[0]

    def excitation(self, mol=None, mf=None):
        """Omega_state in Hartree."""
        mol, mf = self.mean_field(mol, mf)
        om, pieces = self._forward(mol, mf)
        return float(om[self.tracked_state(mol, mf, om, pieces)])

    def energy(self, mol=None, mf=None):
        """(E_ex, E_0, Omega) in Hartree, with E_ex = E_0 + Omega."""
        mol, mf = self.mean_field(mol, mf)
        om_all, pieces = self._forward(mol, mf)
        om = float(om_all[self.tracked_state(mol, mf, om_all, pieces)])
        return mf.e_tot + om, mf.e_tot, om

    def tracked_state(self, mol, mf, om, pieces):
        """Which root of this geometry is the state being followed.

        `self.state` is an index into an energy-ordered manifold, and an
        optimizer that keeps it follows whatever is n-th at each step rather
        than one state. Following the overlap with the previous step keeps the
        identity through a crossing, where the index is what changes.

        Propagates step to step rather than against the first geometry: the
        overlap with a distant reference decays, and a relaxation moves far.
        The price is that a sequence of small misassignments can ratchet onto
        a different state, so the overlap with the original anchor is logged
        beside each step to show it.
        """
        xn, yn = pieces[10], pieces[11]
        if self.track is None:
            return self.state
        here = (mol, np.asarray(mf.mo_coeff, float))
        if self._followed is None:
            self._followed = here + (xn[:, [self.state]].copy(),
                                     yn[:, [self.state]].copy())
            self._anchor = self._followed
            self.follow_log.append(dict(index=self.state, weight=1.0,
                                        margin=1.0, anchor=1.0))
            return self.state
        mol0, mo0, x0, y0 = self._followed
        t = mo_overlap(mol0, mo0, mol, here[1])
        index, weight, margin = follow_state(
            state_overlap(t, self.nocc, x0, y0, xn, yn), 0)
        ta = mo_overlap(self._anchor[0], self._anchor[1], mol, here[1])
        anchor = float(np.abs(state_overlap(
            ta, self.nocc, self._anchor[2], self._anchor[3],
            xn[:, [index]], yn[:, [index]])[1, 1]))
        if weight < ROOT_FOLLOW_WEIGHT_MIN:
            warnings.warn(
                f'the tracked state overlaps its own previous step by only '
                f'{weight:.2f}: it has left the {xn.shape[1]} roots solved '
                f'here, and the root being returned is not it. Raise nroots '
                f'or shorten the step.', RuntimeWarning, stacklevel=2)
        elif margin < ROOT_FOLLOW_MARGIN_MIN:
            warnings.warn(
                f'two roots share the tracked state: overlaps differ by only '
                f'{margin:.2f}. They have mixed, so which one is followed is '
                f'decided by that difference and the state carried past this '
                f'geometry may not be the one asked for.',
                RuntimeWarning, stacklevel=2)
        self._followed = here + (xn[:, [index]].copy(), yn[:, [index]].copy())
        self.follow_log.append(dict(index=int(index), weight=weight,
                                    margin=margin, anchor=anchor,
                                    omega=float(om[index])))
        return int(index)

    def _casida_args(self, pieces):
        """(X_mo, D, eps_qp, W_aux, nocc, cache, Xn, Yn) out of `_forward`.

        The positional tail every Casida-level adjoint takes, named once so a
        seed cannot be wired to the wrong member of a seventeen-long tuple.
        """
        (_, _, _, _, x_mo, d, _, eps_qp, w_aux, cache, xn, yn,
         _, _, _, _, _) = pieces
        return x_mo, d, eps_qp, w_aux, self.nocc, cache, xn, yn

    def _casida_seeds(self, pieces, n, m=None):
        """(eps_qp_bar, X_bar, D_bar, W_aux_bar) of Omega_n, or of the
        interstate element <m| dH |n>, by the realization `bse_adjoint` names.

        The explicit route fills the forward pass's cache in place the first
        time it is asked, so every root reversed off one pinned forward pass
        reads the same blocks and builds them once.
        """
        x_mo, d, eps_qp, w_aux, nocc, cache, xn, yn = self._casida_args(pieces)
        block = self._adjoint_rows(x_mo)
        if self.bse_adjoint == 'grid' and block is not None:
            # the grid adjoint's own rows, moved into the adjoint's tiles:
            # X_bar and D_bar are never whole on any rank
            comm = current_comm()
            e_bar, x_rows, d_rows, w_bar = isdf_bse_backward_rows(
                n, x_mo, d, eps_qp, w_aux, nocc, xn, yn, spin=self.spin,
                bse_tda=self.bse_tda, bra=m, comm=comm)
            npts = (x_mo.npts if isinstance(x_mo, SlicedFactors)
                    else len(x_mo))
            return (e_bar,
                    GridTileRows.from_blocks(x_rows, npts, block, comm),
                    GridTileRows.from_blocks(d_rows, npts, block, comm),
                    w_bar)
        if self.bse_adjoint == 'grid':
            kw = dict(spin=self.spin, bse_tda=self.bse_tda)
            if m is None:
                return isdf_bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, xn,
                                         yn, **kw)
            return isdf_interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc,
                                            xn, yn, **kw)
        if not cache:
            cache.update(bse_cache(x_mo, d, eps_qp, w_aux, nocc,
                                   spin=self.spin, bse_tda=self.bse_tda))
        if m is None:
            return bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, cache, xn, yn,
                                **self._tile_kw())
        return interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc, cache,
                                   xn, yn, **self._tile_kw())

    def _adjoint_rows(self, x_mo):
        """The tile edge the factors' adjoints are held in over the ranks,
        or None serially, where they are whole.

        Over more than one rank every adjoint of X_mo and D is `GridTileRows`
        in the row fit's tiles (`fit_block`, `FIT_CHOLESKY_BLOCK` on the
        replicated fit), owned t % size: the sweeps are tile-major and the
        fit adjoint reads the same tiles, so no rank holds a whole pair. The
        factors' own layout, sliced or whole, reaches the same calls, so the
        two layouts keep one force.
        """
        comm = current_comm()
        if comm is None or comm.Get_size() < 2:
            return None
        return FIT_CHOLESKY_BLOCK if self.fit_block is None else self.fit_block

    @staticmethod
    def _as_rows(a, block):
        """`a` as `GridTileRows` in tiles of `block`: a whole array cut to
        this rank's tiles, verbatim, and tiles handed back as they are."""
        if isinstance(a, GridTileRows):
            return a
        return GridTileRows.from_whole(np.asarray(a), block, current_comm())

    def _fold_to_nuclei(self, pieces, eqp_bar, x_bar, d_bar, w_bar,
                        release_tape=True):
        """(natm, 3) from the four Casida-level adjoints, and diagnostics.

        Everything below the Casida step is linear in the seed, so the same
        sequence (quasiparticle, screening, reaction field, Kohn-Sham
        correction, integrals) carries dOmega_n and the interstate element
        <m| dH |n> alike.

        x_bar and d_bar are consumed: every stage's adjoint is added into them
        in place, in a fixed order, so the reverse pass holds one pair of the
        factors' shape beside the stage at work.

        Over ranks the pair is `GridTileRows` (`_adjoint_rows`): the seeds are
        cut to this rank's tiles, and the quasiparticle and chi0 sweeps return
        their adjoints in the same tiles, tile-major.

        release_tape: drop the forward's quasiparticle tape after reading it.
        False where more reverse passes follow off the same forward
        (`StateManifold.evaluate`), which releases it once at the end; each
        of them then reads proj(tau) and the slices instead of sweeping them.
        The static correction the forward read is reused where the pieces
        carry it (`ForwardPieces`).
        """
        (mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux, cache, xn, yn,
         mu, d_bare, shift, screening, qp_tape) = pieces
        xc = getattr(pieces, 'xc_correction', None)
        block = self._adjoint_rows(x_mo)
        if block is not None:
            x_bar, d_bar = self._as_rows(x_bar, block), self._as_rows(d_bar,
                                                                      block)
        d_sigma = d if d_bare is None else d_bare
        d_bar_bare = None if d_bare is None else np.zeros(d_bare.shape)
        if d_bar_bare is not None and block is not None:
            d_bar_bare = self._as_rows(d_bar_bare, block)
        eps_bar = np.zeros_like(eps)
        mask = self._outside_window(len(eps))
        if self.at_mean_field:
            eps_bar += eqp_bar
        else:
            # the whole set folds through one proj(tau) sweep
            with self.phase('t_qp_backward'):
                if xc is None:
                    xc = self._xc_correction(mf, self.qp_set, shift)
                (_, e_qp, x_qp, d_qp), qp_out = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu, xc, eqp_bar[self.qp_set],
                    tape=qp_tape, rows_block=block)
            # read once: its proj(tau) rows and slices are not held through
            # the rest of the reverse pass unless another reverse follows
            held = (qp_tape,) if release_tape else ()
            for tape in held + (qp_out.pop('tape', None),):
                if tape is not None:
                    tape.release()
            eps_bar += e_qp
            # Outside the set eps^QP_p is eps_p, or eps_p plus a frozen
            # scissor: a constant either way, so the seed passes straight to
            # the mean-field eigenvalue and the shift adds no term of its own.
            eps_bar[mask] += eqp_bar[mask]
            x_bar += x_qp
            # Sigma screened with the bare factor, so its adjoint lands there
            if d_bare is None:
                d_bar += d_qp
            else:
                d_bar_bare += d_qp
            del x_qp, d_qp
        with self.phase('t_screening_backward'):
            chi0_bar = screening_backward(w_aux, w_bar)
            if block is None:
                e2, x2, d2 = chi0_backward(chi0_bar[None, :, :], x_mo, d, eps,
                                           self.nocc, self.w_grid)
            else:
                e2, x2, d2 = chi0_backward_rows(chi0_bar[None, :, :], x_mo, d,
                                                eps, self.nocc, self.w_grid,
                                                block=block)
        eps_bar = eps_bar + e2
        x_bar += x2
        d_bar += d2
        del x2, d2

        w_corr = np.zeros(len(eps))
        if not self.at_mean_field:
            # Z per state, as on the single-quasiparticle path: each orbital in
            # the set has its own pole strength and its own correction.
            w_corr[self.qp_set] = qp_out['z'] * eqp_bar[self.qp_set]
        # Sigma^env reaches further than Sigma_x - v_xc does: outside the
        # window `_env_static_outside` put it straight on the BSE diagonal,
        # with no Newton equation to renormalize it, so it rides at the bare
        # adjoint there and at Z inside.
        w_env = w_corr.copy()
        w_env[mask] = eqp_bar[mask]
        if shift is not None:
            with self.phase('t_reaction_field_backward'):
                e_rf, x_rf, dd_rf, db_rf = reaction_field_backward(
                    w_env, x_mo, d, d_bare, eps, self.nocc, grid=self.w_grid,
                    screening=screening)
            eps_bar = eps_bar + e_rf
            x_bar += x_rf
            d_bar += dd_rf
            d_bar_bare += db_rf
            del x_rf, dd_rf, db_rf
        # the Sigma_x - v_xc skeleton's fit adjoint rides the assembly's
        with self.one_fit_adjoint(mf):
            y_xc, g_xc = self._xc_chain(mf, w_corr, w_env,
                                        on_the_factors=shift is not None)
            grad, diags = self.nuclear_gradient(mol, mf, auxmol, crd, x_mo,
                                                eps_bar, x_bar, d_bar,
                                                y_extra=y_xc, g_extra=g_xc,
                                                d_bar_bare=d_bar_bare)
        return grad, dict(diags, **self.outside_record())

    def excitation_gradient(self, mol=None, mf=None):
        """(dOmega/dR, diagnostics). Nothing larger than three-index."""
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        om, pieces = self._forward(mol, mf)
        root = self.tracked_state(mol, mf, om, pieces)
        return self._root_gradient(pieces, om, root)

    def _root_gradient(self, pieces, om, root, release_tape=True):
        """(dOmega_root/dR, diagnostics) off one forward's pieces."""
        with self.phase('t_bse_backward'):
            seeds = self._casida_seeds(pieces, root)
        grad, diags = self._fold_to_nuclei(pieces, *seeds,
                                           release_tape=release_tape)
        return grad, dict(diags, omega=float(om[root]), root=int(root))

    def _interstate_gradient(self, pieces, om, m, n, release_tape=True):
        """(d<m|H|n>/dR, diagnostics) off one forward's pieces."""
        with self.phase('t_bse_backward'):
            seeds = self._casida_seeds(pieces, n, m)
        grad, diags = self._fold_to_nuclei(pieces, *seeds,
                                           release_tape=release_tape)
        return grad, dict(diags, omega_m=float(om[m]), omega_n=float(om[n]),
                          gap=float(om[n] - om[m]))

    def interstate_gradient(self, m, n, mol=None, mf=None):
        """(d<m|H|n>/dR, diagnostics) between two BSE roots, m != n.

        The numerator of the derivative coupling. The element itself vanishes
        -- H is diagonal in its own eigenbasis -- so this is the derivative of
        a zero, and it is not a gradient of anything: it depends on how the
        orbital basis moves, and only the CSF term of
        `src.gradients.derivative_coupling` restores that gauge. Take the
        coupling from there rather than from this alone.

        Everything below the Casida step is the excitation gradient's own
        chain, so the orbital relaxation and the Pulay terms arrive with it.
        """
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        om, pieces = self._forward(mol, mf)
        return self._interstate_gradient(pieces, om, m, n)

    def quasiparticle(self, offset=0, mol=None, mf=None):
        """eps^QP for the orbital `offset` from the HOMO (0 = HOMO, +1 = LUMO)."""
        mol, mf = self.mean_field(mol, mf)
        x_mo, d, eps, mu, _, _, _, d_bare = self._factors_for(mol, mf)
        shift, _ = self._reaction_field(x_mo, d, d_bare, eps)
        orb = self.nocc - 1 + offset
        # The correction belongs in the forward as in the gradient: without it
        # here the energy and the force describe different functions, which a
        # finite-difference check shows as a small gradient error. On
        # Hartree-Fock it is zero.
        d_sigma = d if d_bare is None else d_bare
        # the grid is the set's, widened by the orbital asked for, so a chain
        # reaches the same quadrature through a BSE excitation or one state
        self._fix_cd_grid(np.union1d(self.qp_set, [orb]), eps)
        xc_orb = float(self._xc_correction(mf, [orb], shift)[0])
        while True:
            qp_out = {}
            with self.phase('t_qp'):
                out = qp_gradient_space_time(x_mo, d_sigma,
                                             eps, self.nocc, self.gw_grid,
                                             self.nu, self.wt, orb, mu=mu,
                                             want_grad=False,
                                             xc_correction=xc_orb,
                                             route_out=qp_out,
                                             **self._qp_kw())
            if not self._freeze_newton_branch(qp_out, eps, xc_orb, [orb]):
                break
        return float(out[0])

    def quasiparticle_gradient(self, offset=0, mol=None, mf=None):
        """(d eps^QP/dR, diagnostics) for one quasiparticle.

        A separate entry point from `excitation_gradient` because it is a
        different physical quantity with a different chain: no BSE adjoint and
        no screening adjoint, one state rather than a whole set.
        """
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        shift, screening = self._reaction_field(x_mo, d, d_bare, eps)
        orb = self.nocc - 1 + offset
        d_sigma = d if d_bare is None else d_bare
        # the grid is the set's, widened by the orbital asked for, so a chain
        # reaches the same quadrature through a BSE excitation or one state
        self._fix_cd_grid(np.union1d(self.qp_set, [orb]), eps)
        xc_orb = float(self._xc_correction(mf, [orb], shift)[0])
        while True:
            qp_out = {}
            with self.phase('t_qp_backward'):
                w_star, z_fac, eps_bar, x_bar, d_bar = qp_gradient_space_time(
                    x_mo, d_sigma, eps, self.nocc,
                    self.gw_grid, self.nu, self.wt, orb,
                    mu=mu, want_grad=True, xc_correction=xc_orb,
                    route_out=qp_out, **self._qp_kw())
            if not self._freeze_newton_branch(qp_out, eps, xc_orb, [orb]):
                break
        # The correction carries Z, not 1: the Newton condition is
        # w = eps_p + Delta_p + Sigma_c(w), so every term on the right,
        # Delta_p included, is renormalized by Z = [1 - dSigma_c/dw]^-1 on its
        # way into dw. Weighting it by 1 leaves an error of (1 - Z) times the
        # correction's gradient, a sizeable fraction of a valence total.
        w_corr = np.zeros(len(eps))
        w_corr[orb] = float(z_fac)
        d_bar_bare = None
        if shift is not None:
            # Sigma screened bare, so the adjoint above is on the bare factor;
            # the dressed one is reached only through Eq. (18).
            d_bar_bare, d_bar = d_bar, np.zeros(d.shape)
            with self.phase('t_reaction_field_backward'):
                e_rf, x_rf, dd_rf, db_rf = reaction_field_backward(
                    w_corr, x_mo, d, d_bare, eps, self.nocc, grid=self.w_grid,
                    screening=screening)
            eps_bar, x_bar = eps_bar + e_rf, x_bar + x_rf
            d_bar, d_bar_bare = d_bar + dd_rf, d_bar_bare + db_rf
        with self.one_fit_adjoint(mf):
            y_xc, g_xc = self._xc_chain(mf, w_corr,
                                        on_the_factors=shift is not None)
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
            extra_gauges=[GaugeAdjoint(d_s, partner)])

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
        # one fit adjoint for both: the mean field's exchange skeleton rides
        # the excitation's assembly
        with self.one_fit_adjoint(mf, mean_field=True):
            g_om, diags = self.excitation_gradient(mol, mf)
            g_0 = self.mean_field_gradient(mf)
        return self.composed_total(mf, g_0, g_om, diags)

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

    def refreeze(self, mol, factorization=None):
        """The same surface with every frozen convention rebuilt at `mol`.

        The pair layout, frames, quasiparticle set, Newton branch, pole-model
        poles, outside scissor and both quadrature grids are re-derived from
        `mol` and its own mean field, as are atomic radii (element-only).
        Every setting that is a choice travels verbatim, explicit radii and
        the resolved tau count included, so only the geometry differs. The
        contour-deformation grid this chain was fixed at is a floor for the
        new one. The environment is the same object and rebuilds itself
        around the atoms.
        """
        # The tracked index, not the constructor's. A refrozen chain starts a
        # fresh overlap history, so it has to be told which root is the state
        # at this geometry: following moved off `self.state` precisely when a
        # crossing was passed, and rebuilding on the old index would hand the
        # outer loop the state that was crossed rather than the one followed.
        state = self.state if not self.follow_log \
            else int(self.follow_log[-1]['index'])
        chain = type(self)(
            mol, self.scf_factory, spin=self.spin, state=state,
            track=self.track,
            bse_tda=self.bse_tda, basis=self.basis, auxbasis=self.auxbasis,
            counts=self.counts, n_start=self.n_start,
            # an explicit radii set is the choice itself, so it travels like
            # `counts`; atomic radii are element-only and re-derive identically
            radii=(self.radii if self.factorization.radii_tag is not None
                   else None),
            qp_window=self.qp_window, degeneracy_tol=self.degeneracy_tol,
            ntau_gw=self.ntau_gw, ntau_w=self.ntau_w, nfreq_cd=self.nfreq_cd,
            e_min_below_gap=self.e_min_below_gap, frames=self.frames_mode,
            residue_route=self.residue_route, tile_gb=self.tile_gb,
            at_mean_field=self.at_mean_field, solver=self.solver,
            dense_max_nov=self.dense_max_nov, nroots=self.nroots,
            bse_conv_tol=self.bse_conv_tol,
            outside=self.outside,
            scissor=self.scissor, n_poles=self.n_poles,
            sop_stride=self.sop_stride,
            environment=self.environment, factorization=factorization,
            sliced=self.sliced, fit=self.fit, fit_block=self.fit_block,
            bse_adjoint=self.bse_adjoint, qp_select=self.qp_select,
            qp_partition=self.qp_partition)
        chain.qp_selection = self.selection_record()
        # The adaptive set's partition IS carried: which orbitals are solved
        # and which probe each hole borrows is frozen for the relaxation, and
        # no continuation runs here. The probes' shifts are this geometry's.
        # The shifts themselves are not carried: this geometry calibrates its
        # own, on its own explicit roots. The old ones ride along only so the
        # record can say how far the frozen convention moved.
        chain.outside_shift_before = self.outside_shift
        return chain

    # --------------------------------------------- the adaptive explicit set
    def _target_states(self):
        """The (spin, root) pairs the selection budget holds for."""
        spec = self.qp_select
        if spec is not None and spec.targets:
            return tuple(spec.targets)
        return ((self.spin, int(self.state)),)

    def _tol_mev(self):
        spec = self.qp_select
        if spec is not None and spec.tol_meV is not None:
            return float(spec.tol_meV)
        if self.qp_partition is not None:
            return float(self.qp_partition.tol_meV)
        return ADAPTIVE_QP_TOL_MEV

    @staticmethod
    def _env_free(values, env):
        """{p: v_p - <p|Sigma^env|p>}: a shift without its orbital's own
        Eq. (18) term, which every orbital outside the set gets on its own."""
        if env is None:
            return dict(values)
        return {int(p): float(v) - float(env[int(p)])
                for p, v in values.items()}

    def _window_weights(self, x_mo, d, eps, eps_window, w_aux, window, spins):
        """({(spin, root): n over every orbital}, {spin: Omega}, {spin: the
        matrix that was not positive definite, or None}) of the BSE
        restricted to the pairs inside `window`, on `eps_window` (the AC
        energies): the first-order weights the selection reads before any
        explicit root exists. Pairs with a partner outside the window are
        missing here; the production eigenvectors check that afterwards.
        Where the window's full BSE is unstable the weights are Tamm-Dancoff
        (`casida_instability`)."""
        occ = [int(p) for p in window if p < self.nocc]
        cols = occ + [int(p) for p in window if p >= self.nocc]
        xw = whole_factor(x_mo, 'X_mo')[:, cols]
        dw = whole_factor(d, 'D')
        norb = len(eps)
        weights, omegas, unstable = {}, {}, {}
        for spin in spins:
            a, b, _ = bse_blocks(xw, dw, np.asarray(eps_window)[cols], w_aux,
                                 len(occ), spin=spin, bse_tda=self.bse_tda)
            # Tamm-Dancoff weights where the window's full BSE is unstable
            unstable[spin] = casida_instability(a, b)
            om, xn, yn = bse_solve(a, None if unstable[spin] else b)
            n = first_order_weights(xn, yn, len(occ), len(cols))
            omegas[spin] = om
            for k in range(len(om)):
                full = np.zeros(norb)
                full[cols] = n[k]
                weights[(spin, k)] = full
        return weights, omegas, unstable

    def _select_qp_set(self, mol, mf, x_mo, d, d_sigma, eps, w_aux, shift):
        """Choose the explicit set out of the candidates, before any root.

        One analytic-continuation GW over the window on this chain's own
        factors and static term (`ac_quasiparticle_shifts`), the
        window-restricted BSE on those energies for the first-order weights
        of every target (and of every root within ADAPTIVE_ROOT_MIX_EV of
        one), then the greedy rule on the raw AC shifts to
        ADAPTIVE_SELECT_MARGIN of the budget. `qp_set` becomes that set; the
        roots solved next calibrate the continuation (`_refine_selection`).
        """
        cands = np.asarray(self.qp_set, int)
        with self.phase('qp_select.ac'):
            xc_c = self._xc_correction(mf, cands, shift)
            # Sigma screens bare: sliced, X_mo's own D is the dressed kernel's,
            # and W_solv would count the continuum again beside Eq. (18)
            factors = (d_sigma if isinstance(d_sigma, SlicedFactors)
                       else (x_mo, d_sigma))
            a, z = ac_quasiparticle_shifts(mf, mol, self.nocc, cands,
                                           factors=factors, xc_diagonal=xc_c)
            a, z = lockstep((a, z))
        bad = ac_unjudgeable(a, z, ADAPTIVE_AC_SHIFT_MAX_EV)
        why = mandatory_members(eps, self.nocc, cands, self.degeneracy_tol,
                                unjudgeable=bad)
        targets = self._target_states()
        with self.phase('qp_select.bse'):
            eps_ac = np.array(eps, float)
            for p in cands:
                if int(p) not in bad:
                    eps_ac[p] += a[int(p)]
            weights, omegas, unstable = self._window_weights(
                x_mo, d, eps, eps_ac, w_aux, cands,
                tuple(dict.fromkeys(t[0] for t in targets)))
        targets = near_root_targets(omegas, targets, ADAPTIVE_ROOT_MIX_EV)
        free = self._env_free(a, shift)
        sel_weights = {t: weights[t] for t in targets if t in weights}
        first = select_explicit(eps, self.nocc, cands, free, sel_weights,
                                self._tol_mev() * ADAPTIVE_SELECT_MARGIN,
                                start=why, degeneracy_tol=self.degeneracy_tol)
        for i, (block, target, term) in enumerate(first.order):
            for p in block:
                why.setdefault(p, {'reason': 'budget', 'order': i,
                                   'target': list(target),
                                   'term_meV': term * HARTREE_TO_MEV})
        for p, w in capped_why(first.capped).items():
            why.setdefault(p, w)
        self._selecting = {
            'owner': self._selection_owner, 'candidates': tuple(int(p) for p in cands),
            'ac': a, 'z': z, 'ac_free': free, 'unjudgeable': bad, 'why': why,
            'targets': targets, 'weights': sel_weights, 'omegas': omegas,
            'window_bse': unstable, 'rounds': 0, 'fell_back': None,
            'verify_pending': False,
            'selection_budget': {t: b * HARTREE_TO_MEV
                                 for t, b in first.budget.items()}}
        self.qp_set = np.asarray(first.explicit, int)

    def _refine_selection(self, eps, ws, shift):
        """After a solve of the selected set at the reference geometry:
        calibrate the continuation on the explicit roots and either grow the
        set (True: solve again) or freeze the partition (False).

        A frontier residual beyond ADAPTIVE_AC_CALIBRATION_MAX_EV means the
        continuation is unfit to select, and the whole window is solved. A
        root the solve rejected stays a hole (`qp_demoted`); if its own term
        alone breaks the budget the record says so and a warning is raised.
        """
        sel = self._selecting
        if self.qp_partition is not None or not self._owns_selection(sel):
            return False
        cands = sel['candidates']
        explicit = [int(p) for p in self.qp_set]
        demoted = set(self.qp_demoted) & set(cands)
        roots = {p: float(w) for p, w in zip(explicit, ws)}
        s_free = self._env_free({p: roots[p] - float(eps[p])
                                 for p in explicit}, shift)
        tol = self._tol_mev()
        # the production eigenvectors' weights once a check has read them
        weights = (sel['verification']['weights'] if 'verification' in sel
                   else sel['weights'])
        selection, calibration = calibrated_selection(
            eps, self.nocc, cands, sel['ac_free'], s_free, weights,
            tol * ADAPTIVE_SELECT_MARGIN, degeneracy_tol=self.degeneracy_tol,
            gate_ev=ADAPTIVE_AC_CALIBRATION_MAX_EV, barred=demoted)
        if selection.fell_back == 'calibration':
            sel['fell_back'] = sel['fell_back'] or 'calibration'
        grown = [p for p in selection.explicit
                 if p not in set(explicit) and p not in demoted]
        if grown:
            sel['rounds'] += 1
            if sel['rounds'] > ADAPTIVE_MAX_ROUNDS:
                sel['fell_back'] = sel['fell_back'] or 'window'
                grown = [p for p in cands if p not in set(explicit)
                         and p not in demoted]
            for p, w in capped_why(selection.capped,
                                   after='calibration').items():
                sel['why'].setdefault(p, w)
            for i, p in enumerate(grown):
                sel['why'].setdefault(
                    p, 'calibration_fallback' if sel['fell_back']
                    else {'reason': 'budget', 'order': i,
                          'after': 'calibration'})
            self._reopen_selection(sorted(set(explicit) | set(grown)))
            return True
        blocks = degenerate_blocks(eps, cands, self.degeneracy_tol)
        tier_of = {p: b for p, b in selection.tier_of.items()}
        sel.update(calibration=calibration, s_free=s_free, blocks=blocks,
                   tier_of=tier_of, verify_pending=True,
                   budget_meV={t: b * HARTREE_TO_MEV
                               for t, b in selection.budget.items()},
                   signed_meV={t: b * HARTREE_TO_MEV
                               for t, b in selection.signed.items()})
        self._warn_unmet_by_demoted(sel, selection, demoted, tol)
        self.qp_partition = AdaptivePartition(
            explicit=explicit, tier_of=tier_of, candidates=cands,
            targets=sel['targets'], tol_meV=tol)
        self.qp_selection = self.selection_record()
        return False

    def _warn_unmet_by_demoted(self, sel, selection, demoted, tol):
        """Record, and warn, when a rejected root's hole alone keeps a target
        over the budget: the admitted window carries the same error, unseen."""
        unmet = {}
        for t, terms in (error_terms(sel['weights'], sel['calibration'].a_tilde,
                                     sel['calibration'].u, sel['s_free'],
                                     selection.tier_of, sel['blocks'])[2]
                         .items() if selection.tier_of else ()):
            for b, term in terms.items():
                if set(b) & demoted and term * HARTREE_TO_MEV > tol:
                    unmet[f'{t[0]} {t[1]}'] = {'demoted': list(b),
                                               'term_meV': term * HARTREE_TO_MEV}
        sel['budget_unmet_by'] = unmet or None
        if unmet:
            warnings.warn(
                f'a rejected quasiparticle root alone keeps the adaptive set '
                f'over its {tol} meV budget: {unmet}. The hole carries the '
                f'scissor, as it would in the admitted window.',
                RuntimeWarning, stacklevel=3)

    def _reopen_selection(self, explicit):
        """Unfreeze what the last solve of the reference geometry froze, so
        the set `explicit` is solved there again from cold: the quadrature
        from its asked size, the Newton branch, the poles and the scissors.
        A warm repeat would put the earlier roots' last bits into this one,
        and the surface would then depend on how many rounds selected it."""
        self.qp_partition = None
        self.qp_set = np.asarray(sorted(int(p) for p in explicit), int)
        self.qp_set_settled = False
        self.outside_shift = None
        self.scissor_map = {}
        self.pole_offsets.clear()
        self.qp_seeds.clear()
        self.sop_poles.clear()
        self._build_cd_grid(self._nfreq_cd_asked,
                            np.asarray(self.mf0.mo_energy, float))
        self.cd_sized = False

    def verify_selection(self, shared, solved):
        """Check a just-frozen adaptive set on the production eigenvectors at
        the reference geometry; True when the set grew and the forward must
        be repeated.

        solved: {spin: (Omega, pieces)} of the Casida solves on `shared`; a
        target spin not among them is solved once here on the same forward.
        The exact Hellmann-Feynman weights of every target give its budget
        B and signed first-order error; every other solved root's are
        recorded. A target over ADAPTIVE_QP_TOL_MEV adds the blocks that
        break it, at most ADAPTIVE_MAX_ROUNDS times, then the whole window.
        """
        sel = self._selecting
        if not self._owns_selection(sel) or not sel.get('verify_pending'):
            return False
        with self.phase('qp_select.verify'):
            spectra = {spin: (om, pieces[10], pieces[11])
                       for spin, (om, pieces) in solved.items()}
            for spin in dict.fromkeys(t[0] for t in sel['targets']):
                if spin not in spectra:
                    om, pieces = self.spin_view(spin)._casida_forward(shared)
                    spectra[spin] = (om, pieces[10], pieces[11])
            check = self._exact_budgets(shared.eps, spectra, sel)
        sel['verify_pending'] = False
        sel['verification'] = check
        tol = self._tol_mev()
        over = [t for t in sel['targets']
                if check['budget'].get(t, 0.0) * HARTREE_TO_MEV > tol]
        if not over:
            self.qp_selection = self.selection_record()
            return False
        cands = sel['candidates']
        demoted = set(self.qp_demoted) & set(cands)
        explicit = [int(p) for p in self.qp_set]
        sel['rounds'] += 1
        if sel['rounds'] > ADAPTIVE_MAX_ROUNDS:
            sel['fell_back'] = sel['fell_back'] or 'verification'
            grown = [p for p in cands if p not in set(explicit)
                     and p not in demoted]
        else:
            cal = sel['calibration']
            more = select_explicit(
                shared.eps, self.nocc, cands, cal.a_tilde, check['weights'],
                tol * ADAPTIVE_SELECT_MARGIN, start=explicit,
                degeneracy_tol=self.degeneracy_tol,
                probe_shift=sel['s_free'], u=cal.u, barred=demoted)
            grown = [p for p in more.explicit if p not in set(explicit)]
        if not grown:
            self.qp_selection = self.selection_record()
            return False
        for p in grown:
            sel['why'].setdefault(p, 'verification')
        self._reopen_selection(sorted(set(explicit) | set(grown)))
        return True

    def _exact_budgets(self, eps, spectra, sel):
        """The budget and signed error of every solved root, and the exact
        weights of the targets, on the production eigenvectors."""
        norb, cands = len(eps), set(sel['candidates'])
        cal, part = sel['calibration'], self.qp_partition
        out = {'weights': {}, 'budget': {}, 'signed': {}, 'outside': {},
               'omega': {}}
        for spin, (om, xn, yn) in spectra.items():
            # the reported roots and every target, not a dense solve's all
            top = max([int(self.nroots)] + [int(t[1]) + 1
                                            for t in sel['targets']
                                            if t[0] == spin])
            om, xn, yn = om[:top], xn[:, :top], yn[:, :top]
            n_all = np.atleast_2d(first_order_weights(xn, yn, self.nocc, norb))
            weights = {(spin, k): n_all[k] for k in range(len(om))}
            signed, budget, _ = error_terms(weights, cal.a_tilde, cal.u,
                                            sel['s_free'], part.tier_of,
                                            sel['blocks'])
            for k, t in enumerate(weights):
                out['budget'][t], out['signed'][t] = budget[t], signed[t]
                out['omega'][t] = float(om[k])
                out['outside'][t] = float(sum(abs(n_all[k][p])
                                              for p in range(norb)
                                              if p not in cands))
                if t in sel['targets']:
                    out['weights'][t] = n_all[k]
        return out

    def selection_record(self):
        """`qp_bookkeeping['adaptive']`: the candidates, the explicit set and
        why each member is in it, the holes with their probes, the
        continuation and its calibration, every target's budget, and the
        fallback, energies in eV and errors in meV. None for another set."""
        sel, part = self._selecting, self.qp_partition
        if sel is None:
            return self.qp_selection
        ev, mev = HARTREE_TO_MEV / 1000.0, HARTREE_TO_MEV
        cands = list(sel['candidates'])
        explicit = [int(p) for p in self.qp_set]
        holes = [p for p in cands if p not in set(explicit)]
        cal = sel.get('calibration')
        record = {
            'candidates': cands, 'n_candidates': len(cands),
            'explicit': explicit, 'n_explicit': len(explicit),
            'why': {int(p): w for p, w in sorted(sel['why'].items())
                    if p in set(explicit)},
            'holes': holes,
            'ac': {'route': 'space-time pade',
                   'shift_eV': {p: sel['ac'][p] * ev for p in cands},
                   'z': {p: sel['z'][p] for p in cands},
                   'unjudgeable': dict(sel['unjudgeable'])},
            'tol_meV': self._tol_mev(), 'margin': ADAPTIVE_SELECT_MARGIN,
            'rounds': sel['rounds'], 'fell_back': sel['fell_back'],
            'budget_unmet_by': sel.get('budget_unmet_by'),
            # the states solved because, as holes, their own error |delta| + u
            # would have exceeded the cap whatever their weight
            'hole_cap': {'max_eV': ADAPTIVE_HOLE_MAX_EV, 'kept_explicit': [
                p for p in explicit if isinstance(sel['why'].get(p), dict)
                and sel['why'][p].get('reason') == 'hole_cap']},
            # the form of the window BSE the selection weights came from:
            # Tamm-Dancoff where the full one was unstable (casida_instability)
            'window_bse': {
                spin: {'weights': 'tda' if self.bse_tda or bad else 'full',
                       'tda_fallback': bad is not None,
                       'not_positive_definite': bad}
                for spin, bad in sel.get('window_bse', {}).items()}}
        if part is not None:
            record['tier_of'] = {p: list(b) for p, b in part.tier_of.items()}
            # what a later stage of the same run reads the partition back from
            record['partition'] = part.as_record()
        if cal is not None:
            front = [max((p for p in explicit if p < self.nocc), default=None),
                     min((p for p in explicit if p >= self.nocc),
                         default=None)]
            record['holes_detail'] = {
                p: {'tier': list(sel['tier_of'].get(p, ())),
                    'delta_eV': (float(np.mean(
                        [sel['s_free'][q] for q in sel['tier_of'][p]]))
                        - cal.a_tilde[p]) * ev if p in sel['tier_of'] else None,
                    'u_eV': cal.u.get(p, 0.0) * ev} for p in holes}
            record['calibration'] = {
                'residual_eV': {q: r * ev for q, r in cal.residual.items()},
                'frontier_max_abs_eV': max(
                    (abs(cal.residual[q]) * ev for q in front
                     if q is not None and q in cal.residual), default=None),
                'gate_eV': ADAPTIVE_AC_CALIBRATION_MAX_EV,
                'passed': sel['fell_back'] != 'calibration'}
            eps = np.asarray(self.mf0.mo_energy, float)
            record['hole_margin_meV'] = min(
                (abs(eps[p] - eps[q]) * mev for p in holes for q in explicit),
                default=None)
            record['hole_cap']['largest_hole_meV'] = max(
                (e * mev for e in hole_errors(
                    cal.a_tilde, cal.u, sel['s_free'], sel['tier_of'],
                    sel['blocks']).values()), default=None)
        check = sel.get('verification')
        targets = []
        for t in sel['targets']:
            row = {'spin': t[0], 'root': int(t[1]),
                   'selection_budget_meV': sel['selection_budget'].get(t, 0.0),
                   'calibrated_budget_meV': sel.get('budget_meV', {}).get(t)}
            if check is not None and t in check['budget']:
                row.update(omega_eV=check['omega'][t] * ev,
                           predicted_error_meV=check['signed'][t] * mev,
                           budget_meV=check['budget'][t] * mev,
                           weight_outside_window=check['outside'][t])
            targets.append(row)
        record['targets'] = targets
        if check is not None:
            other = {}
            for t in check['budget']:
                if t in sel['targets']:
                    continue
                other.setdefault(t[0], []).append(
                    {'root': int(t[1]), 'omega_eV': check['omega'][t] * ev,
                     'predicted_error_meV': check['signed'][t] * mev,
                     'budget_meV': check['budget'][t] * mev})
            record['other_roots'] = other
        record['verified'] = check is not None
        return record

    def adopt_partition(self, partition):
        """Take a partition an earlier stage froze at this reference
        geometry, before the first forward: nothing is selected again."""
        if self.qp_partition is not None or self.qp_set_settled:
            raise RuntimeError('a partition is adopted before the first '
                               'forward, by a chain that has none')
        self._init_adaptive(self.qp_select, partition)

    def posteriori_check(self, mol, mf, spectra, targets):
        """The selection's first-order budget at the end of a walk.

        One AC GW at `mol` on this geometry's factors (`last_forward`, which
        the last evaluation here left), calibrated on this geometry's
        explicit roots, and the exact weights of the walked `targets` from
        `spectra` ({spin: (Omega, X, Y)}): delta_p = a~_p(R*) - s_t(p)(R*)
        with the frozen map. Recorded and warned about, never acted on. Also
        whether the overlap with the reference orbitals maps any window
        orbital onto another, which a frozen hole could not follow.
        """
        part, last = self.qp_partition, getattr(self, 'last_forward', None)
        if part is None:
            return None
        if last is None or not np.array_equal(last['coords'],
                                              mol.atom_coords()):
            raise RuntimeError('the a posteriori check reads the last forward '
                               'at this geometry: evaluate the surface here '
                               'first')
        eps, eps_qp, env = last['eps'], last['eps_qp'], last['shift']
        cands = np.asarray(part.candidates, int)
        x_mo, d, _, _, _, _, _, d_bare = self._factors_for(mol, mf)
        d_sigma = d if d_bare is None else d_bare
        with self.phase('qp_select.ac'):
            xc_c = self._xc_correction(mf, cands, env)
            # Sigma screens bare: sliced, X_mo's own D is the dressed kernel's,
            # and W_solv would count the continuum again beside Eq. (18)
            factors = (d_sigma if isinstance(d_sigma, SlicedFactors)
                       else (x_mo, d_sigma))
            a, z = ac_quasiparticle_shifts(mf, mol, self.nocc, cands,
                                           factors=factors, xc_diagonal=xc_c)
            a, z = lockstep((a, z))
        explicit = [int(p) for p in self.qp_set]
        # an explicit root carries its Eq. (18) term once, through the
        # static correction it was solved with
        s_free = self._env_free({q: float(eps_qp[q] - eps[q])
                                 for q in explicit}, env)
        cal = calibrate_ac(eps, self.nocc, self._env_free(a, env), s_free,
                           cands)
        gate = frontier_residual(cal, eps, self.nocc, explicit,
                                 self.degeneracy_tol) * HARTREE_TO_MEV / 1000.0
        blocks = degenerate_blocks(eps, cands, self.degeneracy_tol)
        norb = len(eps)
        weights = {}
        for spin, root in targets:
            om, xn, yn = spectra[spin]
            weights[(spin, int(root))] = first_order_weights(
                xn[:, int(root)], yn[:, int(root)], self.nocc, norb)
        signed, budget, terms = error_terms(weights, cal.a_tilde, cal.u,
                                            s_free, part.tier_of, blocks)
        tol = self._tol_mev()
        moved = self._orbital_map_changed(mol, mf, cands)
        out = {'ac_shift_eV': {int(p): a[int(p)] * HARTREE_TO_MEV / 1000.0
                               for p in cands},
               'targets': [], 'orbital_map_changed': moved,
               'calibration': {
                   'residual_eV': {q: r * HARTREE_TO_MEV / 1000.0
                                   for q, r in cal.residual.items()},
                   'frontier_max_abs_eV': gate,
                   'passed': bool(gate <= ADAPTIVE_AC_CALIBRATION_MAX_EV)}}
        exceeds = False
        for t in weights:
            worst = sorted(terms[t].items(), key=lambda bt: -bt[1])[:3]
            row = {'spin': t[0], 'root': t[1],
                   'budget_meV': budget[t] * HARTREE_TO_MEV,
                   'predicted_error_meV': signed[t] * HARTREE_TO_MEV,
                   'largest_terms_meV': [[list(b), v * HARTREE_TO_MEV]
                                         for b, v in worst]}
            out['targets'].append(row)
            if budget[t] * HARTREE_TO_MEV > tol:
                exceeds = True
                warnings.warn(
                    f'the adaptive set frozen at the reference geometry is '
                    f'over its {tol} meV budget at the end of the walk: '
                    f'{t[0]} root {t[1]} B = {budget[t] * HARTREE_TO_MEV:.2f} '
                    f'meV, largest terms {row["largest_terms_meV"]}',
                    RuntimeWarning, stacklevel=2)
        out['exceeds_tol'] = exceeds
        return out

    def _orbital_map_changed(self, mol, mf, cands):
        """{p: q} for every window orbital whose largest overlap at `mol` is
        with another orbital than itself, or {}."""
        t = mo_overlap(self.mol0, np.asarray(self.mf0.mo_coeff, float), mol,
                       np.asarray(mf.mo_coeff, float))
        out = {}
        for p in cands:
            q = int(np.argmax(np.abs(t[int(p)])))
            if q != int(p):
                out[int(p)] = q
        return out

    def label(self):
        """The state and the method, for a log line or a relaxation record."""
        kernel = 'BSE(TDA)@GW' if self.bse_tda else 'BSE@GW'
        if self.at_mean_field:
            kernel += '(mean-field eps)'
        return (f'{kernel} {self.spin} state {self.state} / {self.basis} / '
                f'outside {self.outside} / {self.environment!r}')


def casida_phase(om, xn, yn):
    """(Omega, X, Y) with each root's sign fixed by its largest |X| element,
    made positive; elements within CASIDA_PHASE_TIE_TOL of it are a tie,
    broken by the lowest index.

    An eigensolver returns each root with whatever sign its last Rayleigh-Ritz
    step produced, and a Davidson's path follows the last bits of its input:
    on water/cc-pVDZ Hartree-Fock, 34 of 48 one-ulp moves of a single eps_QP
    flip at least one of four roots. Energies and state gradients do not see
    it; a coupling between two roots, d<m|H|n>/dR and the derivative
    coupling, and a spin-orbit element and its derivative change sign with
    it, so the sign is a convention of the vector alone.
    """
    xn, yn = np.array(xn, copy=True), np.array(yn, copy=True)
    for k in range(xn.shape[1]):
        mag = np.abs(xn[:, k])
        lead = int(np.flatnonzero(mag >= (1.0 - CASIDA_PHASE_TIE_TOL)
                                  * mag.max())[0])
        if xn[lead, k] < 0.0:
            xn[:, k] = -xn[:, k]
            yn[:, k] = -yn[:, k]
    return om, xn, yn


def replayed_first_point(surface, mol):
    """The result of the `FirstPoint` `surface` holds, if it answers `mol`
    and the surface's spin and root, else None; taken off the surface at the
    first call either way, since it is the start of a walk and a later visit
    to the same geometry is evaluated afresh."""
    first = surface.__dict__.pop('first_point', None)
    if first is None:
        return None
    mol = surface.mol0 if mol is None else mol
    if first.answers(mol, surface.spin, surface.state):
        return first.result
    return None
