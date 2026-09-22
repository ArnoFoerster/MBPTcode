"""The excited-state potential-energy surface of the cubic BSE@GW gradient.

The gradient chain in `qp_space_time`/`bse_isdf`/`isdf_derivatives` returns
dOmega/dR, the derivative of an EXCITATION energy. A geometry optimization
needs the derivative of the excited state's TOTAL energy,

    E_ex(R) = E_0(R) + Omega(R),     dE_ex/dR = dE_0/dR + dOmega/dR

with E_0 the mean-field ground state, so the ground-state gradient is added
here and never inside the chain. That split matters for what comes after it:
in an ADIABATIC singlet-triplet gap the two states sit at DIFFERENT minima, so
E_0 does not cancel between them and has to be carried consistently.

THE FROZEN CONVENTIONS. The chain fixes seven discrete choices at the
reference geometry -- the quasiparticle set, the frame orientation, the
interpolation pair layout, the Newton branch with the pole guard it steps
around and the seed it starts from, the contour-deformation quadrature sized
against the root-to-pole distance, the residue backend, and the scissor the
orbitals OUTSIDE the set carry -- because each is a discontinuity in the
surface otherwise.
`refreeze` rebuilds all of them at a new geometry with the same settings,
which is how the optimizer in `src/properties/optimize.py` measures the drift
of the surface it walked from the one the fit would choose at the end.

THE SURROUNDINGS enter through the chain's `environment`
(`src.Base.environment`): a polarizable continuum dresses the auxiliary gauge
and adds its static reaction-field operator to the quasiparticle equation,
fixed point charges enter the mean field, the gas phase does nothing. The
energy sees the whole of an environment or none of it, and the gradient asks
the environment for its own adjoints. A continuum supplies both, so a solvated
force is analytic, for a restricted reference and a molecule. An environment
without a derivative says so through `differentiable`, and the chain then
reports energies and refuses forces rather than returning the gas-phase
adjoints, which would be wrong by the whole reaction-field response while
looking reasonable.

THE CASIDA STEP follows production's one memory rule by default
(`solver='auto'`, `bse.solver_choice`): dense while the (A, B) pair fits in
BSE_DENSE_MAX_GB -- every root, in a fixed order, the only form a
finite-difference check should difference -- and matrix-free through
production's ISDF Davidson (`solve_casida_davidson`) above it, for either
spin -- kappa = 0 drops the bare-exchange term from the block action and the
screened term is spin-independent. A named `solver` is not vetoed by the rule:
'dense' at size is a caller paying the memory deliberately. The
adjoint is matrix-free either way: `bse_backward`
reads the eigenvectors and the three-index blocks of `bse_cache`, and the
static W it differentiates is the one the Davidson solved with.

Everything computed FROM this surface -- geometry optimization, normal modes,
Huang-Rhys factors, adiabatic gaps, reorganization energies, rates -- lives in
`src/properties/` and knows only the `PotentialEnergySurface` protocol.
"""
import warnings

import numpy as np

from src.Base.isdf_jk import mean_field_skeleton_force
from src.Base.constants import (ROOT_FOLLOW_MARGIN_MIN,
                                ROOT_FOLLOW_WEIGHT_MIN)
from src.Base.constants import (BSE_DAVIDSON_CONV_TOL, BSE_DAVIDSON_NROOTS,
                                BSE_DENSE_MAX_NOV, CD_NFREQ, CD_NFREQ_MAX,
                                CD_POLE_RESOLUTION, HARTREE_TO_EV,
                                OUTSIDE_TREATMENTS)
from src.Base.declaration import Excitation, SurfacePhysics
from src.Base.environment import attached_environment, environment_label
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        cd_grid_range,
                                                        cd_grid_resolves)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              frozen_scissor)
from src.SingleReference.LinearResponse.bse import solver_choice
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.SingleReference.base import get_occ_virt_indices
from src.gradients.bse_isdf import (bse_backward, bse_blocks, bse_cache,
                                    bse_solve, interstate_backward,
                                    screening_backward)
from src.gradients.contour_deformation import root_pole_distance
from src.gradients.factor_chain import FactorChain
from src.gradients.factor_chain import check_scf_quality  # noqa: F401
from src.gradients.reaction_field_adjoint import (reaction_field_backward,
                                                   reaction_field_shift,
                                                   static_screening)
from src.gradients.isdf_derivatives import (qp_xc_correction,
                                            qp_xc_correction_Y,
                                            qp_xc_correction_skeleton,
                                            xc_hybrid_coeff)
from src.gradients.sum_over_poles import SOP_N_POLES
from src.properties.nonadiabatic import (follow_state, mo_overlap,
                                         state_overlap)
from src.gradients.qp_space_time import (qp_gradient_space_time,
                                         qp_set_gradient)
from src.gradients.space_time_adjoint import chi0_backward


class ExcitedStateChain(FactorChain):
    """Frozen conventions, and the energy and gradient that follow from them.

    `scf_factory(mol) -> mf` is how a displaced geometry gets a mean field; it
    must converge the ORBITAL gradient to ~1e-11 (see `check_scf_quality`).

    `solver='auto'` is production's one memory rule (`bse.solver_choice`,
    resolved in `solver_used`): dense while the Casida pair (A, B) fits in
    BSE_DENSE_MAX_GB, matrix-free above it. It is the default because a pair
    count alone decides nothing -- at a hundred atoms the dense pair is
    terabytes, and under the Tamm-Dancoff form a count rule picks dense
    always -- while at the sizes a finite-difference gate runs the rule
    resolves to the dense route anyway, so the default costs those cases
    neither a number nor an ordering.

    `residue_route='explicit'` takes the real-axis residues of Sigma^c as they
    stand: a residue backend decided per state, or per geometry, is not one
    surface.

    `qp_window=2` is the frontier set, QPStates('frontier', 2) by the declared
    name: the two occupied and two virtual orbitals around the gap, widened
    over degenerate blocks. It is what this class solves explicitly, and the
    orbitals outside it carry what `outside` says.

    `outside` decides what the orbitals with no quasiparticle equation of
    their own carry on the BSE diagonal: 'mean-field' leaves each at its
    eigenvalue, 'scissor' adds the frozen shift `calibrate_scissor` reads off
    the explicitly solved roots at the reference geometry. That shift is a
    NUMBER, so d eps^QP_p/dR = d eps_p/dR for those orbitals and the adjoint
    chain carries no term for it. The `scissor` keyword is a different thing:
    a tier for the states INSIDE the set whose pole model is inadmissible.
    """

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
                 cd_pole_resolution=CD_POLE_RESOLUTION,
                 mf=None, environment=None, factorization=None, radii=None):
        super().__init__(mol, scf_factory, basis=basis, auxbasis=auxbasis,
                         counts=counts, n_start=n_start, frames=frames, mf=mf,
                         environment=environment, factorization=factorization,
                         radii=radii)
        self.spin, self.state, self.bse_tda = spin, state, bse_tda
        # `track='overlap'` follows the STATE; None follows the index.
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
        # `scissor='calibrate'` asks for the tier to be BUILT at the reference
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
        # What the chain this one was refrozen FROM had calibrated, so the
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
        self.nroots, self.bse_conv_tol = nroots, bse_conv_tol
        # Kept verbatim so `refreeze` can rebuild the same surface elsewhere:
        # a setting that is re-derived instead of preserved makes the refrozen
        # surface a different one, and its energy incomparable with this one's.
        self.qp_window = qp_window
        self.degeneracy_tol, self.e_min_below_gap = degeneracy_tol, e_min_below_gap
        self.ntau_gw, self.nfreq_cd = ntau_gw, nfreq_cd
        self.cd_pole_resolution = cd_pole_resolution

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
        # SEED at every displaced geometry so that the iteration lands on the
        # branch this surface was built on rather than on whichever one the
        # mean-field eigenvalue is nearest to.
        self.qp_seeds = {}
        # Whether the contour-deformation quadrature has been sized against the
        # root-to-pole distance yet; that happens on the first quasiparticle
        # solve and is then frozen with everything else.
        self.cd_sized = False

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

        DECLARED HERE, not stamped from outside. A surface built through
        `potential_energy_surface` is CHECKED against this rather than labelled
        by it, so a dispatcher that built the chain with settings the caller
        did not ask for is a refusal instead of a mislabelled number.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.environment))

    # ------------------------------------------------------------------ setup
    def _build_cd_grid(self, nfreq_cd, eps):
        """The contour-deformation quadrature and the imaginary-time grid whose
        frequency axis it is, at `nfreq_cd` points.

        Production's `cd_frequency_grid` builds it, so the surface a gradient
        walks and the quadrature an energy route integrates on are one choice
        made in one place -- including the refusal when e_min_below_gap eats
        the whole gap, which no imaginary-time grid represents.

        ntau_gw is 24 rather than the 18 the imaginary axis alone would need.
        The cosine transform wants the grid's 1/y fit at y = d only, so
        [gap, e_max]; the cosh transform a residue needs it at y = d -/+ w',
        which reaches down to gap - w' and is why e_min sits BELOW the gap.
        Widening the range that far costs points. Measured on benzene/cc-pVDZ
        with e_min = 0.5 gap: every DFT starting point failed the Laplace
        backend's representation gate at 18 points and passed cleanly at 24;
        Hartree-Fock passes either way because its quasiparticle corrections,
        and hence its residue frequencies, are much smaller.
        """
        self.nfreq_cd = int(nfreq_cd)
        # the range the tau axis is fitted on, recorded beside the grid it
        # built: an audit reads the range a surface was frozen at
        self.e_min_gw, self.e_max = cd_grid_range(eps, self.nocc,
                                                  self.e_min_below_gap)
        self.nu, self.wt, self.gw_grid, self.w0_cd = cd_frequency_grid(
            eps, self.nocc, ntau=self.ntau_gw, nfreq_cd=self.nfreq_cd,
            e_min_below_gap=self.e_min_below_gap)

    def _grow_cd_grid(self, roots, eps, states):
        """Double the contour-deformation grid while its first frequency does
        not resolve the root-to-pole spike; True when the grid changed.

        The spike is the Lorentzian of half-width d = |eps^QP_p - eps_q| that
        the pole of G at a NEIGHBOURING orbital energy puts on the
        imaginary-frequency integrand at nu = 0 (`root_pole_distance`). A grid
        whose smallest node sits at a comparable frequency integrates the
        wrong function there, and the Newton then converges on whatever zero
        the truncated self-energy has -- a satellite, or nothing at all.

        Sized ONCE, at the reference geometry, and frozen: a quadrature
        re-decided per geometry is a step in the surface of the same family as
        a re-decided residue route, and the displaced roots sit closer to the
        pole than the reference one anyway, so the reference is the conservative
        place to measure from only if what is measured there is held.
        """
        if self.cd_sized:
            return False
        d = root_pole_distance(eps, roots, states)
        resolved = cd_grid_resolves(self.nu, eps, roots, states,
                                    resolution=self.cd_pole_resolution)
        if not resolved and self.nfreq_cd < CD_NFREQ_MAX:
            self._build_cd_grid(min(2 * self.nfreq_cd, CD_NFREQ_MAX), eps)
            # the frozen branch belongs to the grid it was found on
            self.pole_offsets.clear()
            self.qp_seeds.clear()
            return True
        self.cd_sized = True
        if not resolved:
            warnings.warn(
                f'a quasiparticle root of this set lies {d:.2e} Ha from a '
                f'neighbouring orbital energy, which the contour-deformation '
                f'quadrature does not resolve at its {CD_NFREQ_MAX}-point '
                f'ceiling (smallest frequency {self.nu.min():.2e} Ha). The '
                f'self-energy of that orbital is integrated over a spike the '
                f'grid steps across, so its energy and its force answer '
                f'different functions.', RuntimeWarning, stacklevel=2)
        return False

    @staticmethod
    def _qp_set(eps, nocc, window, tol):
        """The quasiparticle set, widened so no degenerate block is split.

        Inside a degenerate subspace the eigenvectors are not determined, so an
        individual orbital is not a smooth function of the geometry; giving one
        half of a pair a quasiparticle energy and leaving the other at the mean
        field differentiates across an arbitrary rotation.

        `window` is a half-width, 'all', or a SEQUENCE of orbital indices --
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
              'n_poles': self.n_poles,
              'sop_stride': self.sop_stride,
              'scissor': self.scissor_map or self.scissor}
        if self.tile_gb is not None:
            kw['tile_gb'] = self.tile_gb
        return kw

    def _freeze_newton_branch(self, route_out):
        """Keep the guard band and the root each quasiparticle solve resolved
        to, once. True when a calibrated shift was added and the solve has to
        be repeated to read it.

        The first solve is the reference geometry's in every path that then
        displaces it, so `setdefault` freezes that one: a displaced geometry
        reports what it used, starts from the reference root and changes
        neither.
        """
        for p, off in route_out.get('pole_offsets', {}).items():
            self.pole_offsets.setdefault(int(p), float(off))
        roots = route_out.get('roots', {})
        for p, w in roots.items():
            self.qp_seeds.setdefault(int(p), float(w))
        if self.scissor != 'calibrate':
            return False
        # Tier the states the route ACTUALLY sent to the real axis. A second
        # reading of Eq. (27) would test the root where the route tested the
        # start, and disagree on exactly the marginal states.
        eps0 = np.asarray(self.mf0.mo_energy, float)
        grew = False
        for p, route in self._routes_taken(route_out).items():
            if route in ('sop', 'scissor') or p in self.scissor_map:
                continue
            self.scissor_map[p] = float(roots[p]) - float(eps0[p])
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
                      states=None):
        """((eps^QP values, adjoints), route_out) for a quasiparticle set, with
        the contour-deformation grid sized on the first pass.

        The sizing needs the roots and the roots need a grid, so the solve is
        repeated while `_grow_cd_grid` doubles the quadrature under it. That
        costs one extra pass at the reference geometry and nothing afterwards,
        because everything it decided is frozen on the chain.
        """
        states = self.qp_set if states is None else states
        while True:
            route_out = {}
            out = qp_set_gradient(x_mo, d_sigma, eps, self.nocc, self.gw_grid,
                                  self.nu, self.wt, states, weights,
                                  mu=mu, xc_correction=xc_correction,
                                  route_out=route_out,
                                  **self._qp_kw())
            grew = self._grow_cd_grid(out[0], eps, states)
            # A calibrated shift is read by the solve AFTER the one that
            # measured it, so the map triggers one repeat.
            #
            # The SHORT CIRCUIT is load-bearing: freezing on a pass that grew
            # the grid would `setdefault` the roots an under-resolved
            # quadrature produced. Hoisting this call out of the `or` reads as
            # a tidy-up and is a silent accuracy regression.
            if not (grew or self._freeze_newton_branch(route_out)):
                break
        return out, route_out

    def _size_cd_grid(self, x_mo, d_sigma, eps, mu, mf, shift, orb):
        """Size the contour-deformation quadrature before a single-orbital
        solve reads it, on the frozen set widened by the orbital asked for.

        One quasiparticle carries no adjoint here: the grid is a property of
        the whole set, so a chain reaches the same quadrature whether it is
        driven through a BSE excitation or through one quasiparticle.
        """
        if self.cd_sized:
            return
        states = np.union1d(self.qp_set, [orb])
        self._qp_set_solve(x_mo, d_sigma, eps, mu,
                           self._xc_correction(mf, states, shift),
                           np.zeros(len(states)), states=states)

    def _tile_kw(self):
        return {} if self.tile_gb is None else {'tile_gb': self.tile_gb}

    def solver_used(self, n_ov):
        """'dense' or 'davidson' for a pair space of n_ov, resolving 'auto'
        through production's ONE rule, `bse.solver_choice`.

        The rule is dense while the Casida pair (A, B) fits in
        BSE_DENSE_MAX_GB; `solver_choice`'s docstring is where it is stated and
        why the Tamm-Dancoff form is not exempt from it (the dense route builds
        both blocks either way). `dense_max_nov` is an explicit cap ON TOP of
        that rule and may only tighten it: at its default it IS the rule's own
        boundary, so the two agree exactly, and a smaller value forces the
        matrix-free route earlier than memory alone would.

        A TAMM-DANCOFF PROBLEM ABOVE THE RULE IS REFUSED, not served densely.
        `solve_casida_davidson` solves the full Casida problem and takes no
        `tda`, so routing there would return roots of a different kernel from
        the declared one; and staying dense pays exactly the memory the rule
        exists to refuse. Triplets DO go through Davidson -- kappa = 0 removes
        the bare-exchange term from the block action and nothing else changes.
        Refusing at CONSTRUCTION on the string `solver` would reject 'auto' at
        sizes where it was going to pick dense anyway, which is why the refusal
        is here, where the pair count is known.
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

    def _freeze_outside_shift(self, eps, ws):
        """Calibrate the scissor the orbitals outside the set carry, once.

        `calibrate_scissor` gives every outside orbital the quasiparticle
        correction of the explicitly solved orbital NEAREST it in orbital
        energy, so the occupied ones take an occupied probe's shift and the
        virtual ones a virtual probe's -- two tiers, which is where the
        sensitivity splits: an outsize error on a core state moves a frontier
        quasiparticle by a fraction of a meV, the inner valence by orders of
        magnitude more per eV.

        Frozen at the FIRST forward, which is the reference geometry's in
        every path that then displaces it, and held from then on like the
        Newton branch and the quadrature. A shift re-derived per geometry
        steps the surface wherever the nearest probe changes, and it would put
        a d(shift)/dR into a force that has no term for it -- the shift being
        a constant is exactly what lets `_fold_to_nuclei` send the seed of an
        outside orbital straight to the mean-field eigenvalue.
        """
        if self.outside_shift is not None:
            return
        roots = {int(p): float(w) for p, w in zip(self.qp_set, ws)}
        outside = np.flatnonzero(self._outside_window(len(eps)))
        self.outside_shift = calibrate_scissor(eps, self.nocc, roots, outside)

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
        """<p|Sigma^env|p> on the orbitals the quasiparticle window LEFT OUT,
        zero on the ones inside it (where `_xc_correction` already carries it).

        THE WINDOW IS AN APPROXIMATION TO GW, NOT TO A REACTION FIELD. Leaving
        an orbital at its mean-field eigenvalue is defensible for
        Sigma_x - v_xc + Sigma_c: that correction varies smoothly across the
        spectrum, so most of it cancels in the eps_a - eps_i the BSE diagonal
        actually uses. It is not defensible for an environment. Sigma^solv is
        ~ +lambda/2 on EVERY occupied orbital and ~ -lambda/2 on EVERY virtual
        one, and the direct kernel's vtilde reaches EVERY pair; the two cancel
        for a compact electron-hole pair only if both halves reach the same
        orbitals. A pair that gets the kernel's half and not the diagonal's
        keeps a whole lambda of spurious BLUE shift -- and since the window is
        centred on the HOMO/LUMO, the roots it drops are exactly the LOCAL
        ones, which a continuum should barely move.

        Measured on C2H4...F2 / cc-pVDZ at eps = 2.5, window 2 (orbitals
        15-18 of 76): the lowest local root, 11 -> LUMO with CT weight 0.01,
        shifted sign and magnitude without this term and settled close to its
        converged value with it, while the charge-transfer root, whose
        orbitals both sit inside the window, barely moved either way. Nothing
        outside the window ever got Sigma^solv, so the defect grew with how
        far a root reached from the gap.

        Free: Eq. (18) is one vector over the whole spectrum whatever the
        window is. Gas phase passes None and the diagonal is untouched,
        bitwise.
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
        quasiparticle energy, so its own dependence on the orbitals and on the
        nuclei has to be carried like any other -- the Y piece BEFORE the
        multiplier solve, because it shares Lambda, and the skeleton piece in
        the assembly. The gate is on Sigma_x - v_xc alone, which vanishes
        identically on Hartree-Fock.

        The environment's static term rides the same slot and supplies its own
        two pieces (`static_self_energy_adjoint`) -- zero for the gas phase and
        for fixed charges, the reaction field's own orbital response and
        skeleton for a continuum, and a refusal for an unrestricted reference,
        so a dressed energy can never come back with a gas-phase force.

        IT DOES NOT ALWAYS RIDE AT THE SAME WEIGHTS. Outside the quasiparticle
        window `_env_static_outside` puts Sigma^env on the BSE diagonal where
        Sigma_x - v_xc does not go, and there is no Newton equation there to
        renormalize it, so those orbitals carry their bare adjoint in
        `env_weights` and zero in `weights_full`. Default: the two coincide,
        which is the single-quasiparticle case.

        on_the_factors: the continuum rides Eq. (18), whose adjoint reaches the
        nuclei through (eps, X, D) in the caller. The COHSEX operator's own
        adjoint must NOT be added on top, or the reaction field enters the
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
        """<p|Sigma_x - v_xc|p> + <p|Sigma^env|p> for `states`: production's
        static term, evaluated with THIS geometry's environment attached.

        Computed on Hartree-Fock too, where the first part is round-off rather
        than an exact zero; the environment's part is first order in vtilde
        and zeroth in v and survives there. The previous attachment is put
        back afterwards.
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

    def _casida(self, x_mo, d, eps_qp, w_aux):
        """(Omega, X, Y, cache): every root densely, or the lowest `nroots`
        matrix-free on the SAME static W, sorted; the cache is what the adjoint
        needs either way."""
        n_ov = self.nocc * (len(eps_qp) - self.nocc)
        if self.solver_used(n_ov) == 'dense':
            a, b, cache = bse_blocks(x_mo, d, eps_qp, w_aux, self.nocc,
                                     spin=self.spin, bse_tda=self.bse_tda)
            om, xn, yn = bse_solve(a, b)
            return om, xn, yn, cache
        if self.nroots <= self.state:
            raise ValueError(f'nroots={self.nroots} does not reach state {self.state}')
        lr = LinearResponseSolver(np.asarray(eps_qp, float), spin_mode='restricted')
        om, xn, yn = solve_casida_davidson(lr, self.nocc, nroots=self.nroots,
                                           polarizability='BSE', W_aux=w_aux,
                                           isdf_factors=(x_mo, d),
                                           conv_tol=self.bse_conv_tol,
                                           spin=self.spin)
        order = np.argsort(om)
        cache = bse_cache(x_mo, d, eps_qp, w_aux, self.nocc, spin=self.spin,
                          bse_tda=self.bse_tda)
        return om[order], xn[:, order], yn[:, order], cache

    def _forward(self, mol, mf):
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        # ONE STATIC SCREENING. The BSE kernel's W and the dressed half of
        # Eq. (18) are the same [1 - chi0(0)]^-1 on the same axis.
        with self.phase('t_screening'):
            dressed = static_screening(x_mo, d, eps, self.nocc, self.w_grid)
            w_aux = dressed[1]
            pair = (None if d_bare is None else
                    (dressed, static_screening(x_mo, d_bare, eps, self.nocc,
                                               self.w_grid)))
        # THE SELF-ENERGY SCREENS BARE. Eq. (18) is the static approximation to
        # Sigma[W_solv] - Sigma[W_gas], so screening Sigma dynamically as well
        # counts the reaction field twice; the kernel below keeps D.
        shift, screening = self._reaction_field(x_mo, d, d_bare, eps,
                                                screening=pair)
        d_sigma = d if d_bare is None else d_bare
        if self.at_mean_field:
            eps_qp = eps
        else:
            with self.phase('t_qp'):
                ws = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu,
                    self._xc_correction(mf, self.qp_set, shift),
                    np.zeros(len(self.qp_set)))[0][0]
            eps_qp = eps.copy()
            eps_qp[self.qp_set] = ws
            # OUTSIDE THE SET, THE FROZEN SCISSOR. Calibrated on these roots
            # at the reference geometry and spent unchanged afterwards; the
            # environment's static term below is added on top of it, not
            # instead of it.
            if self.outside == 'scissor':
                self._freeze_outside_shift(eps, ws)
                for p in np.flatnonzero(self._outside_window(len(eps))):
                    scissor_p = frozen_scissor(self.outside_shift, p)
                    if scissor_p is not None:
                        eps_qp[p] = eps[p] + scissor_p
        env_outside = self._env_static_outside(shift, len(eps))
        if env_outside is not None:
            eps_qp = eps_qp + env_outside
        with self.phase('t_casida'):
            om, xn, yn, cache = self._casida(x_mo, d, eps_qp, w_aux)
        return om, (mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux, cache,
                    xn, yn, mu, d_bare, shift, screening)

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
        """Which root of THIS geometry is the state being followed.

        `self.state` is an index into an energy-ordered manifold, and an
        optimizer that keeps it follows whatever is n-th at each step rather
        than one state. Following the overlap with the previous step keeps the
        identity through a crossing, where the index is exactly what changes.

        Propagates step to step rather than against the first geometry: the
        overlap with a distant reference decays, and a relaxation moves far.
        The price is that a sequence of small misassignments can ratchet onto
        a different state, so the overlap with the ORIGINAL anchor is logged
        beside each step and is what shows that if it happens.
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
        seed cannot be wired to the wrong member of a sixteen-long tuple.
        """
        (_, _, _, _, x_mo, d, _, eps_qp, w_aux, cache, xn, yn,
         _, _, _, _) = pieces
        return x_mo, d, eps_qp, w_aux, self.nocc, cache, xn, yn

    def _fold_to_nuclei(self, pieces, eqp_bar, x_bar, d_bar, w_bar):
        """(natm, 3) from the four Casida-level adjoints, and diagnostics.

        Everything below the Casida step is LINEAR in the seed, so the same
        sequence -- quasiparticle, screening, reaction field, Kohn-Sham
        correction, integrals -- carries dOmega_n and the interstate element
        <m| dH |n> without knowing which it is holding.
        """
        (mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux, cache, xn, yn,
         mu, d_bare, shift, screening) = pieces
        d_sigma = d if d_bare is None else d_bare
        d_bar_bare = None if d_bare is None else np.zeros(d_bare.shape)
        eps_bar = np.zeros_like(eps)
        mask = self._outside_window(len(eps))
        if self.at_mean_field:
            eps_bar += eqp_bar
        else:
            # the whole set folds through ONE proj(tau) sweep
            with self.phase('t_qp_backward'):
                (_, e_qp, x_qp, d_qp), qp_out = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu,
                    self._xc_correction(mf, self.qp_set, shift),
                    eqp_bar[self.qp_set])
            eps_bar += e_qp
            # Outside the set eps^QP_p is eps_p, or eps_p plus a FROZEN
            # scissor: a constant either way, so the seed passes straight to
            # the mean-field eigenvalue and the shift adds no term of its own.
            eps_bar[mask] += eqp_bar[mask]
            x_bar = x_bar + x_qp
            # Sigma screened with the BARE factor, so its adjoint lands there
            if d_bare is None:
                d_bar = d_bar + d_qp
            else:
                d_bar_bare = d_bar_bare + d_qp
        with self.phase('t_screening_backward'):
            chi0_bar = screening_backward(w_aux, w_bar)
            e2, x2, d2 = chi0_backward(chi0_bar[None, :, :], x_mo, d, eps,
                                       self.nocc, self.w_grid)
        eps_bar, x_bar, d_bar = eps_bar + e2, x_bar + x2, d_bar + d2

        w_corr = np.zeros(len(eps))
        if not self.at_mean_field:
            # Z PER STATE, as on the single-quasiparticle path: each orbital in
            # the set has its own pole strength and its own correction, so one
            # shared factor would be as wrong as none.
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
            eps_bar, x_bar = eps_bar + e_rf, x_bar + x_rf
            d_bar, d_bar_bare = d_bar + dd_rf, d_bar_bare + db_rf
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
            seeds = bse_backward(root, *self._casida_args(pieces),
                                 **self._tile_kw())
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
            seeds = interstate_backward(m, n, *self._casida_args(pieces),
                                        **self._tile_kw())
        grad, diags = self._fold_to_nuclei(pieces, *seeds)
        return grad, dict(diags, omega_m=float(om[m]), omega_n=float(om[n]),
                          gap=float(om[n] - om[m]))

    def quasiparticle(self, offset=0, mol=None, mf=None):
        """eps^QP for the orbital `offset` from the HOMO (0 = HOMO, +1 = LUMO)."""
        mol, mf = self.mean_field(mol, mf)
        x_mo, d, eps, mu, _, _, _, d_bare = self._factors_for(mol, mf)
        shift, _ = self._reaction_field(x_mo, d, d_bare, eps)
        orb = self.nocc - 1 + offset
        # The correction belongs HERE as much as in the gradient. Omitting it
        # from the forward while the reverse carries it makes the two describe
        # different functions -- and a finite difference of THIS against a
        # gradient of THAT looks exactly like a small but nonzero gradient
        # error, which is how it was found. On Hartree-Fock it is zero and the
        # two agree by construction rather than by accident.
        d_sigma = d if d_bare is None else d_bare
        self._size_cd_grid(x_mo, d_sigma, eps, mu, mf, shift, orb)
        qp_out = {}
        with self.phase('t_qp'):
            out = qp_gradient_space_time(x_mo, d_sigma,
                                         eps, self.nocc, self.gw_grid,
                                         self.nu, self.wt, orb, mu=mu,
                                         want_grad=False,
                                         xc_correction=float(
                                             self._xc_correction(
                                                 mf, [orb], shift)[0]),
                                         route_out=qp_out,
                                         **self._qp_kw())
        self._freeze_newton_branch(qp_out)
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
        self._size_cd_grid(x_mo, d_sigma, eps, mu, mf, shift, orb)
        qp_out = {}
        with self.phase('t_qp_backward'):
            w_star, z_fac, eps_bar, x_bar, d_bar = qp_gradient_space_time(
                x_mo, d_sigma, eps, self.nocc,
                self.gw_grid, self.nu, self.wt, orb,
                mu=mu, want_grad=True,
                xc_correction=float(self._xc_correction(mf, [orb], shift)[0]),
                route_out=qp_out, **self._qp_kw())
        self._freeze_newton_branch(qp_out)
        # THE CORRECTION CARRIES Z, NOT 1. The Newton condition is
        # w = eps_p + Delta_p + Sigma_c(w), so every term on the right is
        # renormalized by Z = [1 - dSigma_c/dw]^-1 on its way into dw --
        # including Delta_p. Weighting it by 1 instead leaves an error of
        # (1 - Z) times the correction's own gradient, which on a valence
        # quasiparticle is a sizeable fraction of the total.
        w_corr = np.zeros(len(eps))
        w_corr[orb] = float(z_fac)
        d_bar_bare = None
        if shift is not None:
            # Sigma screened bare, so the adjoint above is on the BARE factor;
            # the dressed one is reached only through Eq. (18).
            d_bar_bare, d_bar = d_bar, np.zeros(d.shape)
            with self.phase('t_reaction_field_backward'):
                e_rf, x_rf, dd_rf, db_rf = reaction_field_backward(
                    w_corr, x_mo, d, d_bare, eps, self.nocc, grid=self.w_grid,
                    screening=screening)
            eps_bar, x_bar = eps_bar + e_rf, x_bar + x_rf
            d_bar, d_bar_bare = d_bar + dd_rf, d_bar_bare + db_rf
        y_xc, g_xc = self._xc_chain(mf, w_corr,
                                    on_the_factors=shift is not None)
        grad, diags = self.nuclear_gradient(mol, mf, auxmol, crd, x_mo, eps_bar,
                                            x_bar, d_bar, y_extra=y_xc,
                                            g_extra=g_xc, d_bar_bare=d_bar_bare)
        # Z enters dw/dR and not w, so a route with Sigma's value right and
        # its slope wrong returns an exact energy with a wrong force. Without
        # Z in the record nothing says so.
        return grad, dict(diags, qp_energy=float(w_star), qp_z=float(z_fac),
                          qp_route=qp_out.get('residue_route'))

    def total_gradient(self, mol=None, mf=None):
        """(dE_ex/dR, E_ex, diagnostics) -- the quantity an optimizer steps on.

        The ground-state gradient is pyscf's own for whatever mean field the
        chain was handed, so a density-fitted reference gets the density-fitted
        gradient including its auxiliary-basis response.
        """
        mol, mf = self.mean_field(mol, mf)
        g_om, diags = self.excitation_gradient(mol, mf)
        g_0 = mean_field_skeleton_force(mf)
        diags = dict(diags, e_scf=mf.e_tot, grad_scf_max=float(np.abs(g_0).max()),
                     grad_omega_max=float(np.abs(g_om).max()))
        return g_0 + g_om, mf.e_tot + diags['omega'], diags

    # ------------------------------------------- the potential-energy surface
    def total_energy(self, mol=None, mf=None):
        """E_ex = E_0 + Omega_state in Hartree: the surface, not the excitation."""
        return self.energy(mol, mf)[0]

    def refreeze(self, mol, factorization=None):
        """The same surface with every frozen convention rebuilt at `mol`.

        The pair layout, frames, quasiparticle set, the Newton branch, the
        scissor the orbitals outside the set carry and both quadrature grids
        are re-derived from `mol` and its own mean field, and so are atomic
        radii, which depend only on the elements; radii given
        explicitly are the choice itself and travel verbatim, since a grid
        tailored at the reference geometry is the one the whole walk uses.
        every setting that was a CHOICE is carried over verbatim, including the
        resolved tau count, so that only the geometry differs between the two
        surfaces. The contour-deformation grid starts from the size this chain
        grew to rather than from the constructor's, so refreezing never steps
        BACK to a quadrature already found too coarse; the new geometry may
        grow it further. The environment is the same object; it rebuilds itself
        around the atoms.
        """
        # The tracked index, not the constructor's. A refrozen chain starts a
        # fresh overlap history, so it has to be told which root IS the state
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
            cd_pole_resolution=self.cd_pole_resolution, outside=self.outside,
            scissor=self.scissor, n_poles=self.n_poles,
            sop_stride=self.sop_stride,
            environment=self.environment, factorization=factorization)
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
