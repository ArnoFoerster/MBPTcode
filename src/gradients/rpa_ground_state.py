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

from src.Base.constants import RPA_ENERGY_NFREQ
from src.Base.declaration import SurfacePhysics
from src.Base.environment import environment_label
from src.Base.utils.grids import gap_scaled_w0, gauss_legendre_grid
from src.Base.utils.time_frequency import (TimeFrequencyGrid,
                                           minimax_points_for_accuracy)
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.rpa_energy import (
    declared_ground_state, ground_state_energy)
from src.SingleReference.LinearResponse.solvated_rpa_energy import fold_terms
from src.Base.dispersion import refuse_dispersion_under_rpa
from src.gradients.factor_chain import FactorChain
from src.gradients.solvated_rpa_energy import fold_bar_to_gauge
from src.gradients.space_time_adjoint import rpa_energy_and_adjoint


class RPAGroundStateChain(FactorChain):
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
    `fold=True` restores it (`src/SingleReference/LinearResponse/
    solvated_rpa_energy.py`), on the cavity the caller gives: on the
    electrostatic one the term is several times too large, and it belongs on
    the solvent-accessible surface.
    """

    READS_SLICED_FACTORS = True

    def __init__(self, mol, scf_factory, basis=None, auxbasis=None, counts=None,
                 n_start=1, ntau=None, nfreq=RPA_ENERGY_NFREQ, frames='frozen',
                 tile_gb=None, mf=None, environment=None, factorization=None,
                 fold=False, radii=None, sliced=None, fit=None,
                 fit_block=None):
        super().__init__(mol, scf_factory, basis=basis, auxbasis=auxbasis,
                         counts=counts, n_start=n_start, frames=frames, mf=mf,
                         environment=environment, factorization=factorization,
                         radii=radii, sliced=sliced, fit=fit,
                         fit_block=fit_block)
        self.tile_gb = tile_gb
        self.fold = bool(fold)
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

    def refreeze(self, mol, factorization=None):
        """The same surface with radii, layout, frames and grids rebuilt at `mol`.

        The tau count and the frequency quadrature are carried over as resolved
        integers rather than re-derived: a quadrature that changes size between
        two geometries makes them two different functionals whose energies do
        not compare. The environment is the same object; it rebuilds itself
        around the new atoms.

        The fold and an explicit radii set travel for the same reason the
        quadrature does: each is a choice of FUNCTIONAL, and a walk that
        changed one at the first refreeze would compare two surfaces.
        """
        return type(self)(mol, self.scf_factory, basis=self.basis,
                          auxbasis=self.auxbasis, counts=self.counts,
                          n_start=self.n_start, ntau=self.ntau,
                          nfreq=len(self.grid.omega_points),
                          frames=self.frames_mode, tile_gb=self.tile_gb,
                          environment=self.environment,
                          factorization=factorization, fold=self.fold,
                          radii=(self.radii
                                 if self.factorization.radii_tag is not None
                                 else None),
                          sliced=self.sliced, fit=self.fit,
                          fit_block=self.fit_block)

    def label(self):
        """The state and the method, for a log line or a relaxation record."""
        return f'dRPA ground state / {self.basis}'
