"""The cubic nuclear gradient of the dRPA ground state on the space-time/ISDF
factorization.

    E_c = (1/2pi) int dw Tr[ ln(1 - chi0(iw)) + chi0(iw) ],   chi0 from proj(tau)

is a closed-form functional of (eps, X_mo, D): in the quasi-boson language it
is variational in the RPA amplitudes, so no amplitude multiplier exists and
the only response is the mean field's, which the orbital Lagrangian carries.
`space_time_adjoint.rpa_energy_and_adjoint` supplies E_c and its adjoints at
the forward pass's own cost; `FactorChain.nuclear_gradient` puts them on the
nuclei. The mean-field gradient is added here and never inside the chain:

    E(R) = E_HF(R) + E_c(R),      dE/dR = dE_HF/dR + dE_c/dR.

The imaginary-time grid and the frequency quadrature are fixed parameters of
the surface, chosen once for the reference gap and range.

On sliced factors (`sliced=True`, over more than one rank) the energy and its
adjoint gather X_mo and D once for their sweeps and the nuclear assembly X_mo
once more, so between geometries a rank holds its grid rows alone, and the
layout adds no difference: on one mean field, and on a pyscf that repeats its
bits, the force is the whole layout's bit for bit. pyscf's threaded K builds
and mean-field force do not repeat their bits from run to run, so two
evaluations are compared on an anchored bar. On the row fit (`fit='rows'`) the
rows are the fit's own and no rank forms the fit whole; the force is then
the derivative of that fit's estimator (`FrozenFactorization`).
"""
import numpy as np

from src.gradients.rpa.chain_settings import CarriedGrids, chain_setting, rpa_counts
from src.Base.declaration import SurfacePhysics
from src.environments.environment import environment_label
from src.Base.utils.grids.grids import gap_scaled_w0, gauss_legendre_grid
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.Base.orbital_spaces import get_occ_virt_indices
from src.SingleReference.LinearResponse.rpa_energy import (
    declared_ground_state, ground_state_energy)
from src.SingleReference.LinearResponse.solvated_rpa_energy import fold_terms
from src.Base.pyscf.dispersion import refuse_dispersion_under_rpa
from src.Base.geometry.geometry import molecule_at
from src.gradients.rpa.factor_chain import FactorChain, FrozenFactorization
from src.gradients.rpa.solvated_energy import fold_bar_to_gauge
from src.gradients.rpa.space_time_adjoint import rpa_energy_and_adjoint
from src.gradients.surface import PotentialEnergySurface


class RPAGroundStateChain(FactorChain, PotentialEnergySurface):
    """E_HF + E_c^dRPA and its nuclear gradient on a frozen factorization and quadrature.

    Under ranks (`with distributed(comm):`) every rank runs the chain whole
    and `rpa_energy_and_adjoint` divides its frequency loop over them.

    AN ABSOLUTE TOTAL ENERGY IN A CONTINUUM IS NOT TRUSTWORTHY under the
    default interaction. The exact block fold of the ACFDT log-determinant
    keeps the BARE interaction in the linear counter-term while the default
    here dresses it, and the dressing throws away the leading solute-solvent
    dispersion term -- enough to leave water DESTABILIZED by toluene.
    Excitation energies, quasiparticle levels and any difference taken at a
    FIXED geometry are free of it, as is every gas-phase total energy.
    `fold=True` restores it (the solvated dRPA energy of
    `src/SingleReference/LinearResponse/solvated_rpa_energy.py`, whose
    gradient is `gradients.rpa.solvated_energy`), on the cavity the caller
    gives: on the
    electrostatic one the term is several times too large, and it belongs on
    the solvent-accessible surface.

    In an EQUILIBRIUM continuum E_0 and its force are those of the
    non-equilibrium one bit for bit (`environment_at`): the ground state
    relaxes in eps_static in both modes, and the equilibrium switch moves only
    an excitation's response.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'energy', 'mean_field', 'phase', 'physics',
                             'scf_flags', 'to_record'})

    READS_SLICED_FACTORS = True

    # the working-set cap of the tiled intermediates, and whether the energy
    # folds the continuum's frequency dependence: choices of `settings`
    tile_gb = chain_setting('layout.tile_gb')
    fold = chain_setting('solvent.fold')

    def __init__(self, mol, scf_factory, numerics=None, mf=None,
                 environment=None, factorization=None, carried=None):
        super().__init__(mol, scf_factory, numerics=numerics, mf=mf,
                         environment=environment, factorization=factorization)
        # the quadrature: the one a chain before this resolved (`carried`),
        # else its own where derived, else the excited state's
        ntau, nfreq = rpa_counts(self.settings, carried)
        refuse_dispersion_under_rpa(self.mf0, type(self).__name__)
        eps = np.asarray(self.mf0.mo_energy, float)
        occ, virt = get_occ_virt_indices(eps, self.nocc)
        self.gap = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        # the tau count follows the Laplace test error of the reference range
        self.ntau = (minimax_points_for_accuracy(self.gap, e_max,
                                                 target=DEFAULT_TAU_TARGET)[0]
                     if ntau is None else int(ntau))
        # the frequency axis is the quadrature of the energy integral itself
        nu, wt = gauss_legendre_grid(nfreq, w0=gap_scaled_w0(eps, self.nocc))
        self.grid = TimeFrequencyGrid.minimax_split(
            self.ntau, self.gap, e_max, nu, wt, with_sine=False,
            with_inverse=False)

    def _tile_kw(self):
        return {} if self.tile_gb is None else {'tile_gb': self.tile_gb}

    def environment_at(self, mol):
        """The screening E_c reads at THIS geometry: the solvent's electronic
        response, in either solvation mode.

        E_c is the zero-point energy of the solute's density fluctuations, and
        only the solvent's electrons follow them -- with eps_inf and its pole
        g(iw) (Duchemin, Amblard and Blase, J. Chem. Theory Comput. 20, 9072
        (2024)). The slow orientational polarization follows the mean density,
        which the mean field already carries at eps_static, and in equilibrium
        it relaxes around the excitation's density difference, which only the
        excitation carries. So an equilibrium continuum is read through its
        non-equilibrium partner: E_0 and its force are one function in both
        modes, while the declared environment (and its mean field) stays the
        one given. eps_static inside E_c would be a different functional.
        """
        return correlation_screening(super().environment_at(mol))

    def _fold_terms(self, mol, auxmol):
        """(screening, counter_term) of the frequency-dependent fold, or
        (None, None) for the interaction this surface uses by default.

        The fold is v + g(iw) vtilde inside the logarithm with the BARE
        interaction in the linear counter-term, which is what an exact block
        fold of the log-determinant over a non-overlapping solvent keeps. The
        two are ONE choice and `fold_terms` makes it in one place for every
        chain that folds.

        It is off unless asked for, because the dispersion term it restores
        belongs on the solvent-accessible surface and not on the electrostatic
        cavity, where it is several times too large.
        """
        if not self.fold:
            return None, None
        return fold_terms(self.environment_at(mol), auxmol,
                          self.grid.omega_points)

    def _forward(self, mol, mf, want_grad):
        """E_c (and its adjoints) on the factors at `mol`, plus the reverse pass's inputs."""
        x_mo, d, eps, auxmol, crd, _ = self.factors_at(mol, mf)
        screening, counter_term = self._fold_terms(mol, auxmol)
        with self.phase('t_rpa_backward' if want_grad else 't_rpa'):
            out = rpa_energy_and_adjoint(x_mo, d, eps, self.nocc, self.grid,
                                         want_grad=want_grad,
                                         screening=screening,
                                         counter_term=counter_term,
                                         **self._tile_kw())
        return out, (auxmol, crd, x_mo)

    def correlation_energy(self, mol=None, mf=None):
        """E_c^dRPA in Hartree."""
        mol, mf = self.mean_field(mol, mf)
        return float(self._forward(mol, mf, False)[0])

    @property
    def physics_ground_state(self):
        """E_0 = E_HF + E_c^dRPA, on whatever functional the mean field is."""
        return declared_ground_state(self.mf0, 'rpa')

    @property
    def physics(self):
        """What this chain computes: E_0 alone, with no state on it.

        Declared by the chain, not stamped on it from outside, so a surface
        built directly carries the same declaration as one dispatched.
        """
        return SurfacePhysics(self.physics_ground_state, None,
                              environment_label(self.environment))

    def ground_state(self, mol=None, mf=None):
        """E_0 with its terms, through the one production assembly."""
        mol, mf = self.mean_field(mol, mf)
        e_c = float(self._forward(mol, mf, False)[0])
        return ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                   e_corr=e_c)

    def energy(self, mol=None, mf=None):
        """(E, E_HF, E_c) in Hartree, E = E_HF + E_c."""
        e0 = self.ground_state(mol, mf)
        # the double-counting term is stored as E_HF - E_ref, so the two
        # recover E_HF[rho] exactly rather than to a rounding
        e_hf = e0.terms['E_ref'] + e0.terms['E_x^HF - E_xc']
        return e0.total, e_hf, e0.terms['E_c^dRPA']

    def correlation_gradient(self, mol=None, mf=None):
        """(dE_c/dR, E_c, diagnostics). Nothing larger than three-index."""
        self.require_differentiable_environment()
        mol, mf = self.mean_field(mol, mf)
        self.flag_scf_quality(mf, 'correlation_gradient')
        (e_c, eps_bar, x_bar, d_bar, fold_bar), (auxmol, crd, x_mo) = (
            self._forward(mol, mf, True))
        # THE EXACT-EXCHANGE / DOUBLE-COUNTING TERM RIDES ALONG. E_0 is
        # E_HF + E_c^dRPA whatever the starting point, so on a Kohn-Sham
        # reference the surface carries E_x^exact - E_xc as well. Its Y joins
        # the Lagrangian BEFORE the multiplier solve because it shares Lambda,
        # and its skeleton is added to the orbital branch -- the hooks
        # `nuclear_gradient` documents for exactly this. On Hartree-Fock both
        # are identically zero and the gradient is bitwise what it always was.
        y_extra, g_extra = self.kohn_sham_gradient_correction(mol, mf)
        # THE FOLD'S OWN ADJOINTS. N = R^-1 vtilde R^-1 is not a function of the
        # factors, so its adjoint leaves the frequency loop separately and lands
        # on the dressed metric: on the root R, which D reads too, and on vtilde,
        # which it does not. Both are None unless the fold is active.
        root_bar, kernel_bar = fold_bar_to_gauge(auxmol,
                                                 self.environment_at(mol),
                                                 fold_bar)
        grad, diags = self.nuclear_gradient(
            mol, mf, auxmol, crd, x_mo, eps_bar, x_bar, d_bar,
            y_extra=y_extra, g_extra=g_extra,
            root_bar=root_bar, kernel_bar=kernel_bar)
        return grad, float(e_c), dict(diags, e_c=float(e_c))

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics): the mean field's own gradient plus dE_c/dR.

        A density-fitted reference gets pyscf's density-fitted gradient,
        auxiliary-basis response included, so E_HF and its derivative are the
        same functional.
        """
        mol, mf = self.mean_field(mol, mf)
        g_c, e_c, diags = self.correlation_gradient(mol, mf)
        # paired with the double-counting skeleton by `mean_field_gradient`:
        # the two are differenced, so the grid response cannot go on one alone
        g_0 = self.mean_field_gradient(mf)
        e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                 e_corr=e_c)
        e_hf = e0.terms['E_ref'] + e0.terms['E_x^HF - E_xc']
        diags = dict(diags, e_scf=mf.e_tot, e_hf=e_hf, e0_terms=e0.terms,
                     grad_scf_max=float(np.abs(g_0).max()),
                     grad_corr_max=float(np.abs(g_c).max()))
        return g_0 + g_c, e0.total, diags

    # ------------------------------------------- the potential-energy surface
    def total_energy(self, mol=None, mf=None):
        """E_HF + E_c^dRPA in Hartree: the surface a property routine steps on."""
        return self.energy(mol, mf)[0]

    def _refrozen(self, mol, factorization, mf, walked, qp_floor):
        """The same surface with radii, layout, frames and grids rebuilt at
        `mol` (`FactorChain.refreeze`). Nothing here is keyed to orbitals, so
        `walked` carries nothing over.

        The tau count and the frequency quadrature are carried over as resolved
        integers rather than re-derived: a quadrature that changes size between
        two geometries makes them two different functionals whose energies do
        not compare. The environment is the same object; it rebuilds itself
        around the new atoms.

        The fold travels for the same reason the quadrature does: it is a
        choice of FUNCTIONAL, and a walk that changed it at the first
        refreeze would compare two surfaces. The factorization's settings
        travel inside `factorization`.
        """
        return type(self)(mol, self.scf_factory, numerics=self.settings,
                          mf=mf, environment=self.environment,
                          factorization=factorization,
                          carried=self.carried_grids())

    def carried_grids(self):
        """The resolved dRPA quadrature a refreeze or a restore starts from."""
        return CarriedGrids(ntau_rpa=int(self.ntau),
                            nfreq_rpa=len(self.grid.omega_points))

    def settings_record(self):
        """The chain's choices besides its factorization: the resolved
        quadratures, the fold and the tile cap."""
        return {'ntau': int(self.ntau), 'nfreq': len(self.grid.omega_points),
                'fold': self.fold, 'tile_gb': self.tile_gb}

    def to_record(self):
        """Everything this surface froze, for `from_record`: the reference
        geometry, its settings, the factorization and the reference mean
        field's ISDF-K pair layout. The `Numerics` is the one asked for and
        the resolved quadrature is `carried`, as a refreeze carries it."""
        return {'class': f'{type(self).__module__}:{type(self).__qualname__}',
                'mol0_coords': np.array(self.mol0.atom_coords(), float),
                'settings': self.settings_record(),
                'numerics': self.settings,
                'carried': self.carried_grids(),
                'factorization': self.factorization.to_record(),
                'scf_layout': self.scf_layout()}

    @classmethod
    def from_record(cls, record, mol, scf_factory, environment=None, mf=None,
                    factorization=None):
        """The surface `to_record` wrote, rebuilt without deriving any
        convention again: one SCF at its reference geometry (or `mf`), the
        recorded factorization (or `factorization`, the same one another half
        already read back) and the resolved quadratures."""
        chain = cls(molecule_at(mol, record['mol0_coords']), scf_factory,
                    environment=environment, mf=mf,
                    factorization=(FrozenFactorization.from_record(
                        record['factorization']) if factorization is None
                        else factorization),
                    numerics=record['numerics'], carried=record.get('carried'))
        chain.hold_scf_layout(record['scf_layout'])
        return chain

    def label(self):
        """The state and the method, for a log line or a relaxation record."""
        return f'dRPA ground state / {self.basis}'


def correlation_screening(environment):
    """`environment` as a correlation energy's fluctuations see it: a continuum
    in equilibrium answers with its non-equilibrium partner (response at
    eps_inf, pole g(iw)); everything else is returned as it is."""
    partner = getattr(environment, 'non_equilibrium_partner', None)
    return environment if partner is None else partner()
