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

Everything computed from this surface (geometry optimization, normal modes,
Huang-Rhys factors, adiabatic gaps, reorganization energies, rates) lives in
`src/properties/` and knows only the `PotentialEnergySurface` protocol.
"""
import warnings

import numpy as np

from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.constants import (FIT_CHOLESKY_BLOCK, ROOT_FOLLOW_MARGIN_MIN,
                                ROOT_FOLLOW_WEIGHT_MIN)
from src.Base.constants import (BSE_ADJOINTS, BSE_DAVIDSON_CONV_TOL,
                                BSE_DAVIDSON_NROOTS, BSE_FORCE_MAX_CYCLE,
                                BSE_FORCE_RESIDUAL_TOL,
                                BSE_DENSE_MAX_NOV, CD_NFREQ, HARTREE_TO_EV,
                                OUTSIDE_TREATMENTS, SOP_N_POLES)
from src.Base.declaration import Excitation, SurfacePhysics
from src.Base.environment import attached_environment, environment_label
from src.Base.utils.mpi_grid import current_comm
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        cd_grid_range)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              frozen_scissor)
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
from src.gradients.qp_space_time import (qp_gradient_space_time,
                                         qp_set_gradient, static_term)
from src.gradients.space_time_adjoint import (chi0_backward,
                                              chi0_backward_rows)


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
                 nfreq_cd=CD_NFREQ, e_min_below_gap=None, frames='frozen',
                 residue_route='explicit', tile_gb=None, at_mean_field=False,
                 scissor=None, outside='mean-field', n_poles=SOP_N_POLES,
                 sop_stride=None,
                 solver='auto', dense_max_nov=BSE_DENSE_MAX_NOV,
                 nroots=BSE_DAVIDSON_NROOTS, bse_conv_tol=BSE_DAVIDSON_CONV_TOL,
                 mf=None, environment=None, factorization=None, radii=None,
                 sliced=None, fit=None, fit_block=None,
                 bse_adjoint='explicit'):
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
        self.ntau_gw, self.nfreq_cd = ntau_gw, nfreq_cd

        eps = np.asarray(self.mf0.mo_energy, float)
        occ, virt = get_occ_virt_indices(eps, self.nocc)
        gap = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        self.gap = gap
        self._build_cd_grid(nfreq_cd, eps)
        self.ntau_w = (minimax_points_for_accuracy(gap, e_max,
                                                   target=DEFAULT_TAU_TARGET)[0]
                       if ntau_w is None else int(ntau_w))
        self.w_grid = TimeFrequencyGrid.minimax_split(
            self.ntau_w, gap, e_max, [0.0], [1.0],
            with_sine=False, with_inverse=False)
        self.qp_set = self._qp_set(eps, self.nocc, qp_window, degeneracy_tol)
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
        solve resolved to, once. True when a calibrated shift was added and
        the solve has to be repeated to read it.

        The first solve is the reference geometry's in every path that then
        displaces it, so `setdefault` freezes that one: a displaced geometry
        reports what it used, starts from the reference root, evaluates the
        pole model on the reference poles and changes none of them.

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
        for p, w in roots.items():
            self.qp_seeds.setdefault(int(p), float(w))
        for p, poles in route_out.get('sop_poles', {}).items():
            self.sop_poles.setdefault(int(p), np.array(poles, float))
        if self.scissor != 'calibrate':
            return False
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
        return grew

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
        """((eps^QP values, adjoints), route_out) for a quasiparticle set.

        tape: an earlier solve's `QPSetTape` on the same factors, and each
        repeat reads the one before it, so proj(tau) and the slices are
        swept once however often the solve repeats (`QPSetTape.reads`).
        rows_block: the adjoints in grid tiles over the ranks
        (`qp_set_gradient`).
        """
        states = self.qp_set if states is None else states
        while True:
            route_out = {}
            out = qp_set_gradient(x_mo, d_sigma, eps, self.nocc, self.gw_grid,
                                  self.nu, self.wt, states, weights,
                                  mu=mu, xc_correction=xc_correction,
                                  route_out=route_out, tape=tape,
                                  rows_block=rows_block, **self._qp_kw())
            tape = route_out.get('tape', tape)
            # A calibrated shift is read by the solve after the one that
            # measured it, so the map triggers one repeat.
            if not self._freeze_newton_branch(
                    route_out, eps, xc_correction, states):
                break
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
        """
        if self.outside_shift is not None:
            return
        roots = {int(p): float(w) for p, w in zip(self.qp_set, ws)}
        outside = np.flatnonzero(self._outside_window(len(eps)))
        self.outside_shift = calibrate_scissor(
            eps, self.nocc, self._env_free(roots, env), outside)

    @staticmethod
    def _env_free(values, env):
        """{p: v_p - <p|Sigma^env|p>}: a value without its orbital's own
        Eq. (18) term, which every orbital outside the set gets on its own."""
        if env is None:
            return dict(values)
        return {int(p): float(v) - float(env[int(p)])
                for p, v in values.items()}

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
            om, xn, yn = bse_solve(a, b)
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
        return om[order], xn[:, order], yn[:, order], {}

    def _forward(self, mol, mf):
        pieces = self.kernel_pieces(mol, mf)
        x_mo, d, eps_qp, w_aux = pieces[4], pieces[5], pieces[7], pieces[8]
        with self.phase('t_casida'):
            om, xn, yn, cache = self._casida(x_mo, d, eps_qp, w_aux)
        return om, pieces[:9] + (cache, xn, yn) + pieces[12:]

    def kernel_pieces(self, mol, mf):
        """Everything `_forward` builds before the Casida solve, in its layout.

        The same seventeen-long tuple `_forward` returns, with an empty cache
        and no roots (`xn`, `yn` are None): the quasiparticle energies, the
        static screening and the factors the BSE matrix is made of. A quantity
        that needs the kernel but not the supermolecular roots -- the diabatic
        elements of `src.properties.fragment_bse` -- starts here, and so does
        its gradient: `_fold_to_nuclei` reads nothing from the roots.
        """
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
        qp_tape = None
        if self.at_mean_field:
            eps_qp = eps
        else:
            with self.phase('t_qp'):
                out, route_out = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu,
                    self._xc_correction(mf, self.qp_set, shift),
                    np.zeros(len(self.qp_set)))
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
        # The quasiparticle set's tape rides with the pieces: the reverse
        # solve reads proj(tau) and the slices from it instead of sweeping
        # them again, bitwise.
        return (mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux, {},
                None, None, mu, d_bare, shift, screening, qp_tape)

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

    def _fold_to_nuclei(self, pieces, eqp_bar, x_bar, d_bar, w_bar):
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
        """
        (mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux, cache, xn, yn,
         mu, d_bare, shift, screening, qp_tape) = pieces
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
                (_, e_qp, x_qp, d_qp), qp_out = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu,
                    self._xc_correction(mf, self.qp_set, shift),
                    eqp_bar[self.qp_set], tape=qp_tape, rows_block=block)
            # read once: its proj(tau) rows and slices are not held through
            # the rest of the reverse pass (a later reverse off the same
            # forward sweeps them again)
            for tape in (qp_tape, qp_out.pop('tape', None)):
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
        with self.phase('t_bse_backward'):
            seeds = self._casida_seeds(pieces, root)
        grad, diags = self._fold_to_nuclei(pieces, *seeds)
        return grad, dict(diags, omega=float(om[root]), root=int(root))

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
        with self.phase('t_bse_backward'):
            seeds = self._casida_seeds(pieces, n, m)
        grad, diags = self._fold_to_nuclei(pieces, *seeds)
        return grad, dict(diags, omega_m=float(om[m]), omega_n=float(om[n]),
                          gap=float(om[n] - om[m]))

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
        xc_orb = float(self._xc_correction(mf, [orb], shift)[0])
        qp_out = {}
        with self.phase('t_qp'):
            out = qp_gradient_space_time(x_mo, d_sigma,
                                         eps, self.nocc, self.gw_grid,
                                         self.nu, self.wt, orb, mu=mu,
                                         want_grad=False,
                                         xc_correction=xc_orb,
                                         route_out=qp_out,
                                         **self._qp_kw())
        self._freeze_newton_branch(qp_out, eps, xc_orb, [orb])
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
        xc_orb = float(self._xc_correction(mf, [orb], shift)[0])
        qp_out = {}
        with self.phase('t_qp_backward'):
            w_star, z_fac, eps_bar, x_bar, d_bar = qp_gradient_space_time(
                x_mo, d_sigma, eps, self.nocc,
                self.gw_grid, self.nu, self.wt, orb,
                mu=mu, want_grad=True, xc_correction=xc_orb,
                route_out=qp_out, **self._qp_kw())
        self._freeze_newton_branch(qp_out, eps, xc_orb, [orb])
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
        mol, mf = self.mean_field(mol, mf)
        # one fit adjoint for both: the mean field's exchange skeleton rides
        # the excitation's assembly
        with self.one_fit_adjoint(mf, mean_field=True):
            g_om, diags = self.excitation_gradient(mol, mf)
            g_0 = self.mean_field_gradient(mf)
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
        contour-deformation grid starts from the size this chain grew to, so
        refreezing never returns to a quadrature found too coarse; the new
        geometry may grow it further. The environment is the same object and
        rebuilds itself around the atoms.
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
            bse_adjoint=self.bse_adjoint)
        # The shifts themselves are not carried: this geometry calibrates its
        # own, on its own explicit roots. The old ones ride along only so the
        # record can say how far the frozen convention moved.
        chain.outside_shift_before = self.outside_shift
        return chain

    def label(self):
        """The state and the method, for a log line or a relaxation record."""
        kernel = 'BSE(TDA)@GW' if self.bse_tda else 'BSE@GW'
        if self.at_mean_field:
            kernel += '(mean-field eps)'
        return (f'{kernel} {self.spin} state {self.state} / {self.basis} / '
                f'outside {self.outside} / {self.environment!r}')
