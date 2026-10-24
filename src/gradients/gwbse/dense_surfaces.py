"""Potential energy surfaces on the DENSE quasi-boson route.

The N-electron ground state, the N+/-1 quasiparticle states and the excited
states of the Bethe-Salpeter equation, each behind the `PotentialEnergySurface`
protocol so `src.properties.optimize` can relax them. Their cubic counterparts
are `RPAGroundStateChain` and `ExcitedStateChain`; these hold the full (pq|rs)
instead of factorizing it, which bounds them at a few hundred basis functions
and makes them the reference the cubic route is checked against.

A DIFFERENCE OF TWO MINIMA MUST COME OFF ONE ROUTE. An adiabatic ionization
potential is min E^{N-1} - min E^N, and whatever ground-state energy sits
inside the first has to be the second: the Tamm-Dancoff quasiparticle surface
carries no correlation energy, so its partner is the bare mean field, not
E_HF + E_c. Pairing them wrongly leaves -E_c in the difference, which wears the
units of an ionization potential while being several eV of correlation energy.

UNDER RANKS every rank runs these surfaces whole: nothing here divides, and
every term -- the mean field's force, the four-index transform, the
integral derivatives -- is pyscf's threaded arithmetic, whose OpenMP GEMM adds
its partial sums in thread-arrival order and so does not repeat its bits from
one rank to the next. Each energy and force a surface returns is therefore one
`lockstep` of what the call computed: rank 0's on every rank, and serially the
value itself.
"""
import warnings

import numpy as np
from pyscf import gto, scf
from pyscf.df.addons import make_auxmol

from src.Base.numerics.constants import (HARTREE_TO_EV, QP_BLOCK_CORRECTION_TOL,
                                         QP_DEGENERACY_TOL, QP_ORDER_SEARCH,
                                         SCF_DIFFERENTIABLE_GRAD_TOL, SCF_MAX_CYCLE,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.declaration import (ChargedExcitation, Excitation,
                                  SurfacePhysics)
from src.SingleReference.common.eri_blocks import MOEriBlocks, df_eri_mo, mo_eri
from src.Base.environment_seam import environment_of
from src.Base.factorization.isdf_jk import ISDFJK
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.Base.utils.parallel.mpi_grid import lockstep
from src.Base.utils.parallel.threads import blas_single_threaded
from src.SingleReference.GW.qp_selection import true_degenerate_blocks
from src.SingleReference.GW.qp_states import valence_qp_states
from src.SingleReference.LinearResponse.rpa_energy import (declared_ground_state,
                                                           ground_state_energy)
from src.Base.pyscf.pyscf_interface import xc_hybrid_coeff
from src.gradients.grad_engine import correlation_gradients
from src.gradients.surface import PotentialEnergySurface
from src.gradients.gwbse.diagnostics import chain_diagnostics
from src.gradients.rpa.exchange_double_counting import (exx_double_counting_Y,
                                                        exx_double_counting_skeleton)
from src.gradients.gwbse.static_correction import (qp_xc_correction_Y,
                                                   qp_xc_correction_skeleton)
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.gradients.gwbse.quasi_boson_adjoint import (BSEqbAdjoint as BSEqb,
                                                     QPqbAdjoint as QPqb)
from src.gradients.gwbse.targets import (add_z_contribution, rpa_partials,
                                         qp_partials_with_shift, solve_Z)

#: How many virtual orbitals above the Fermi level are searched for the
#: quasiparticle LUMO; see `constants.QP_ORDER_SEARCH`.
ATTACHMENT_SEARCH = QP_ORDER_SEARCH

# Paper Table I: how the four variants are named and what each one means.
# (screening of BOTH the G0W0 step and the BSE kernel, BSE eigenproblem)
VARIANTS = {
    'BSE@GW':          dict(screening='rpa', bse_tda=False),
    'BSEtda@GW':       dict(screening='rpa', bse_tda=True),
    'BSE@GWtda':       dict(screening='tda', bse_tda=False),
    'BSEtda@GWtda':    dict(screening='tda', bse_tda=True),
}

# The ground state each variant's excitation sits on: E_0 = E_HF + E_c^dRPA for
# RPA screening (letter Eq. 15/17), E_0 = E_HF for TDA screening (Eq. 19),
# where the correlation part of the plasmon formula vanishes.
GROUND_OF = {'rpa': 'rpa', 'tda': 'hf'}


def tight_rhf(mol):
    """The default reference: unfitted, and converged tightly enough to
    differentiate.

    The gradient Lagrangian assumes the occupied-virtual Fock block vanishes,
    so a loose SCF biases the force rather than degrading it gracefully; and a
    density-fitted mean field gives different orbitals, which the quasiparticle
    energy inherits directly. An SCF that stops short is warned about and
    kept where it stopped, carrying the flag 'scf_not_converged' in
    `mf.flags`.
    """
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = SCF_MAX_CYCLE
    with blas_single_threaded():
        mf.kernel()
    if not mf.converged:
        text = (f'the SCF did not converge in {mf.max_cycle} cycles; it is kept '
                f'where it stopped, and what follows carries its residual')
        warnings.warn(text, RuntimeWarning, stacklevel=2)
        mf.flags = [{'flag': 'scf_not_converged', 'message': text}]
    return mf


#: The letter's quasiparticle window: all valence occupied, nocc+10 virtual.
#: Production owns the set a declaration resolves to, so this route and the
#: cubic one cannot drift into two windows under one name.
qp_window = valence_qp_states


def kohn_sham_gradient_correction(mol, mf, nocc, qp_weights=None):
    """(Y_extra, g_extra): what a Kohn-Sham reference adds to a dense gradient.

    Two energies on these surfaces are not the mean field's own when the
    reference is Kohn-Sham, and both terms are carried here.

    E_0 is E_HF[rho], not E_KS: the exact-exchange double counting
    E_x^exact - E_xc rides along with it, `exx_double_counting_Y` being its
    orbital response -- which enters the Lagrangian BEFORE the multiplier solve,
    since it shares Lambda with everything else -- and
    `exx_double_counting_skeleton` its skeleton.

    A quasiparticle energy carries the static shift <p|Sigma_x - v_xc|p> as
    well; `qp_weights` is d eps^QP / d delta_p per orbital, the root's own
    quasiparticle weight, and None where no shift was applied.

    These are the production routines the cubic chain differentiates, so the
    two routes mean the same thing by them; the first pair is the one
    `FactorChain.kohn_sham_gradient_correction` hands its chain. It tests the
    reference's exchange fraction and returns EXACT zeros on Hartree-Fock, so
    this route's Hartree-Fock numbers -- the oracle every cubic gate is
    measured from -- are bit-unchanged. The shift's pair is a difference of
    Coulomb/exchange builds and cancels only numerically there, which is why no
    shift is applied to a Hartree-Fock reference in the first place.
    """
    y = exx_double_counting_Y(mf, nocc)
    g = exx_double_counting_skeleton(mf, mol)
    if qp_weights is None:
        return y, g
    return (y + qp_xc_correction_Y(mf, qp_weights, nocc),
            g + qp_xc_correction_skeleton(mf, qp_weights, nocc))


def refuse_a_solvated_mean_field(mf, surface, force=False):
    """Refuse a mean field carrying a continuum where the dense route cannot see it.

    Two quantities on this route are blind to one. `mo_eri` is a raw
    four-index transform of the BARE interaction and consults no environment,
    so E_c^dRPA here is the correlation energy of v alone, while a polarizable
    environment dresses every post-SCF interaction, v -> v + vtilde, and
    `RPAGroundStateChain` on the SAME mean field builds E_c from the dressed
    one: the two are different functionals, against the tolerance a relaxed
    geometry needs. And every force here (`force=True`) differentiates a
    Lagrangian with no reaction-field response: the orbital multipliers solve
    against the gas-phase Fock response and the skeleton reads the bare
    one-electron Hamiltonian, so the cavity's answer to the moving density and
    nuclei is missing from every dense gradient, whatever the screening.

    TWO MARKERS, EITHER OF WHICH IS A CONTINUUM. pyscf's PCM sits on the mean
    field as `with_solvent` and has already relaxed the orbitals inside the
    reaction field; an `Environment` reaches post-SCF code as the attached
    `with_screening` and dresses the interaction. The second is asked through
    the environment's `screens`, the basis-free half of `dresses_interaction`:
    this route expands (pq|rs) in no auxiliary basis, and building a kernel in
    one only to learn that it exists would be evaluating the reaction field to
    discover there is one. Fixed point charges screen nothing and are left
    alone.

    The check sits where E_c or the force is formed rather than on the stored
    factory, so a mean field handed straight to a surface cannot walk past it.
    """
    environment = environment_of(mf)
    if hasattr(mf, 'with_solvent'):
        carries = 'its SCF was relaxed inside a PCM reaction field'
    elif getattr(environment, 'screens', True):
        carries = f'{environment!r} is attached to it and dresses v'
    else:
        return
    if force:
        raise ValueError(
            f'{surface} differentiates a Lagrangian with no reaction-field '
            f'response -- gas-phase orbital multipliers and the bare '
            f'one-electron skeleton -- but this mean field carries a '
            f'continuum: {carries}. The force would omit the cavity\'s response '
            f'to the moving density and nuclei, and would not be the '
            f'derivative of any energy this surface reports. Use '
            f'ExcitedStateChain, or RPABSESurface / RPAQPSurface, which '
            f'differentiate an environment through the auxiliary metric. '
            f'The energies (`total_energy`, `excitation`) are not '
            f'refused by this.')
    raise ValueError(
        f'{surface} reports E_HF + E_c^dRPA built from the BARE (pq|rs), but '
        f'this mean field carries a continuum: {carries}. The dense route has '
        f'no way to see it, so the correlation energy it would return is a '
        f'different functional from the one the cubic chain returns on the '
        f'same mean field. Use RPAGroundStateChain, or RPABSESurface / '
        f'RPAQPSurface composed on it, which carry the environment through '
        f'the auxiliary metric. Excitation energies and quasiparticle levels '
        f'are differences and are unaffected.')


def refuse_a_foreign_mean_field(mf, mol, surface):
    """Refuse a mean field converged at another geometry than `mol`.

    The post-SCF integrals and their skeleton derivatives are built at `mol`,
    while the core Hamiltonian and the mean field's own force read `mf.mol`:
    on two geometries the energy and force are a mixture of both, of the right
    shape and meaning neither.
    """
    if not np.array_equal(mf.mol.atom_coords(), mol.atom_coords()):
        raise ValueError(
            f'{surface} was handed a mean field converged at another geometry '
            f'than the molecule it evaluates: converge the mean field at this '
            f'geometry, or pass mf=None.')


def mean_field_integrals(mf, mol):
    """How the mean field's two-electron integrals are represented: ('exact',
    None), ('df', its auxiliary molecule at `mol`) for a pyscf density-fitted
    SCF, or ('isdf', None) for an ISDF-K one (DF Coulomb, interpolated
    exchange), which no dense post-SCF step reproduces."""
    with_df = getattr(mf, 'with_df', None)
    if isinstance(with_df, ISDFJK):
        return 'isdf', None
    if with_df is not None:
        return 'df', make_auxmol(mol, with_df.auxbasis)
    return 'exact', None


def refuse_mismatched_integrals(mf, mol, auxmol, surface, auxbasis=True):
    """Refuse a force whose mean field and post-SCF step use different (pq|rs).

    The gradient engine (`correlation_gradients`) folds the Fock matrix's
    dependence on the orbitals, and solves the orbital multipliers, with the
    post-SCF integrals `eri_mo`, while the mean field's own force comes from
    the mean field. Both are the derivative of one energy only when the mean
    field was converged with those same integrals: exact on exact, or
    density-fitted in the auxiliary basis the post-SCF step fits in. A
    DF-RHF reference under exact post-SCF integrals, or an exact one under
    `auxbasis=`, returns a force 6e-5 Ha/Bohr off its own energy on water/
    cc-pVDZ; `DenseRPASurface` and `QuasiparticleSurface`, which always use
    the exact tensor, return forces 2.2e-6 and 6.7e-7 Ha/Bohr off on a
    DF-RHF there (unfitted: 2e-9 to 4e-9, a central difference's
    truncation). The energy is a well-defined functional either way (the RI
    step of the dense-vs-cubic decomposition is an exact reference with an
    `auxbasis`), so only the force is refused.

    auxmol: the post-SCF auxiliary molecule, None for exact integrals.
    auxbasis: whether `surface` takes an `auxbasis=` (`DenseBSESurface`);
              `DenseRPASurface` and `QuasiparticleSurface` hold the exact
              tensor only.
    """
    kind, mf_aux = mean_field_integrals(mf, mol)
    fitted_in = (getattr(getattr(mf, 'with_df', None), 'auxbasis', None)
                 or "pyscf's default JK-fitting set")
    if kind == 'isdf':
        why = ('the mean field is ISDF-K (interpolated exchange), which no '
               'dense post-SCF integral reproduces')
    elif kind == 'df' and auxmol is None:
        why = (f'the mean field is density-fitted in '
               f'{fitted_in} '
               f'and the post-SCF integrals are exact (auxbasis=None)')
    elif kind == 'exact' and auxmol is not None:
        why = (f'the mean field is unfitted and the post-SCF integrals are '
               f'fitted in {auxmol.basis!r}')
    elif kind == 'df' and mf_aux._basis != auxmol._basis:
        why = (f'the mean field is density-fitted in '
               f'{fitted_in} '
               f'and the post-SCF integrals in {auxmol.basis!r}')
    else:
        return
    use = (' with auxbasis=None, or a density-fitted one with auxbasis= its '
           'own auxiliary basis.' if auxbasis else
           ': this surface transforms the exact (pq|rs) and takes no '
           'auxiliary basis.')
    raise ValueError(
        f'{surface} differentiates its post-SCF step with the mean field\'s '
        f'orbital response built from the post-SCF integrals, but {why}: the '
        f'force would not be the derivative of the energy this surface '
        f'reports (6e-5 Ha/Bohr off on water/cc-pVDZ). Use an unfitted mean '
        f'field{use} The energies are not refused by this.')


class DenseRPASurface(PotentialEnergySurface):
    """E_0 = E_HF + E_c^dRPA on the dense quasi-boson route.

    The N-electron partner of `QuasiparticleSurface`, and the reason it exists:
    an ADIABATIC ionization potential is the difference of two minima, one on
    each surface, and the two have to come off the same route or the difference
    carries the route's error instead of the ionization. `RPAGroundStateChain`
    is the same energy on the cubic space-time factorization -- correct, and not
    interchangeable with this one at the tolerance a relaxed geometry needs.

    E_c is variational in the RPA amplitudes, so there is no amplitude
    multiplier here and no Z vector: the only response is the mean field's,
    which the orbital Lagrangian in `correlation_gradients` already carries.

    The starting point is free. On a Kohn-Sham reference E_0 is E_HF[rho] and
    not E_KS -- adding E_c^dRPA to the latter would count the correlation
    inside E_xc twice -- and the exact-exchange double counting that repairs it
    carries its own gradient, `kohn_sham_gradient_correction`.

    A mean field carrying a continuum is REFUSED rather than ignored: the
    interaction here is bare and a polarizable environment dresses it, so
    silently computing E_c[v] on one is `refuse_a_solvated_mean_field`'s job.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'mean_field', 'physics'})

    def __init__(self, mol, scf_factory=None):
        self.mol0 = mol
        self._scf = scf_factory or tight_rhf
        # The starting point is free -- Kohn-Sham included -- so the functional
        # is read off the mean field rather than assumed.
        self.physics_ground_state = declared_ground_state(self._scf(mol), 'rpa')

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on -- a gas-phase
        surface, so the factory's own; `mf` given is used as it is."""
        mol = self.mol0 if mol is None else mol
        return mol, (self._scf(mol) if mf is None else mf)

    def scf_factory(self, mol):
        return self._scf(mol)

    def _build(self, mol, mf=None):
        mf = self._scf(mol) if mf is None else mf
        refuse_a_foreign_mean_field(mf, mol, 'DenseRPASurface')
        refuse_a_solvated_mean_field(mf, 'DenseRPASurface')
        nocc = mol.nelectron // 2
        eri = mo_eri(mf, mol)
        return mf, eri, nocc, QPqb(mf.mo_energy, eri, nocc, screening='rpa')

    @property
    def physics(self):
        """What this surface computes: E_0 alone, in the gas phase.

        `refuse_a_solvated_mean_field` refuses a dressed mean field, so the
        environment of this declaration is never anything but
        `SurfacePhysics`'s default.
        """
        return SurfacePhysics(self.physics_ground_state, None)

    def total_energy(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        mf, _, _, qp = self._build(mol, mf)
        return lockstep(ground_state_energy(declared_ground_state(mf, 'rpa'),
                                            mf, mol,
                                            e_corr=qp.qb.e_corr()).total)

    def total_gradient(self, mol=None, mf=None):
        """(dE_0/dR, E_0, diagnostics), fully analytic."""
        mol = self.mol0 if mol is None else mol
        mf = self._scf(mol) if mf is None else mf
        refuse_mismatched_integrals(mf, mol, None, 'DenseRPASurface',
                                    auxbasis=False)
        mf, eri, nocc, qp = self._build(mol, mf)
        e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                 e_corr=qp.qb.e_corr())
        g_fock, g_eri = rpa_partials(qp.qb, nocc, mol.nao)
        y_extra, g_extra = kohn_sham_gradient_correction(mol, mf, nocc)
        (g_corr,), (diag,) = correlation_gradients(mol, mf,
                                                   [(g_fock, g_eri, y_extra)],
                                                   eri_mo=eri)
        grad = mean_field_skeleton_force(mf) + g_corr + g_extra
        return lockstep((grad, e0.total, {
            'e_corr': float(qp.qb.e_corr()),
            'e0_terms': e0.terms,
            'stationarity': float(diag['stationarity'])}))

    def refreeze(self, mol, mf=None):
        """Nothing is frozen on this surface; the copy is for the protocol.
        mf: not read; the surface converges its own SCF at `mol`."""
        return DenseRPASurface(mol, self._scf)

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        return 'E_HF + E_c(dRPA)'


class QuasiparticleSurface(PotentialEnergySurface):
    """E_0 -/+ eps^QP as a `PotentialEnergySurface`, for one added or removed
    electron.

    charge_change: -1 for an ionization (E^{N-1}), +1 for an attachment
        (E^{N+1}). The sign convention follows the electron count, so the
        quasiparticle energy enters with the OPPOSITE sign: removing an
        electron costs -eps_h, which is positive because eps_h is.
    screening: 'rpa' for G0W0, 'tda' for G0W0-TDA. The correlation energy E_c
        belongs to the RPA screening alone -- the Tamm-Dancoff surface is
        E_HF -/+ eps^QP with no ground-state correlation of its own, which is
        the convention the 2024 paper's Tables II and III are computed in.

    The starting point is free. A Kohn-Sham reference moves E_0 to E_HF[rho]
    and the quasiparticle energy by <p|Sigma_x - v_xc|p>, and both travel with
    their nuclear derivatives (`kohn_sham_gradient_correction`), so the force
    stays the derivative of the energy this object reports.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'mean_field', 'physics'})

    def __init__(self, mol, scf_factory, charge_change=-1, screening='rpa',
                 orbital=None):
        if charge_change not in (-1, +1):
            raise ValueError(f'charge_change {charge_change}: -1 removes an '
                             f'electron, +1 adds one')
        if screening not in ('rpa', 'tda'):
            raise ValueError(f"screening {screening!r}: 'rpa' or 'tda'")
        self.mol0 = mol
        self._scf = scf_factory
        self.charge_change = charge_change
        self.screening = screening
        # eps enters with the sign of the electron count change reversed.
        self.sign = -1.0 if charge_change == -1 else +1.0
        self.orbital = orbital
        mf = self._scf(mol)
        # The starting point is free -- Kohn-Sham included -- so the functional
        # is read off the mean field rather than assumed.
        self.physics_ground_state = declared_ground_state(mf, 'rpa')
        # The Newton seed is the reference root, frozen: a seed carried from
        # the last evaluation makes the root at a geometry depend on the call
        # history, and from a far step it can land on a satellite.
        if orbital is None:
            self.orbital, self._seed = self._choose(mol, mf)
        else:
            self._seed = self._build(mol, mf)[3].solve_diag(orbital)[0]

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on -- a gas-phase
        surface, so the factory's own; `mf` given is used as it is."""
        mol = self.mol0 if mol is None else mol
        return mol, (self._scf(mol) if mf is None else mf)

    def scf_factory(self, mol):
        return self._scf(mol)

    def _build(self, mol, mf=None):
        mf = self._scf(mol) if mf is None else mf
        refuse_a_foreign_mean_field(mf, mol, 'QuasiparticleSurface')
        nocc = mol.nelectron // 2
        eri = mo_eri(mf, mol)
        # G0W0 replaces the reference's static exchange-correlation potential by
        # the self-energy, so a Kohn-Sham starting point carries the shift
        # <p|Sigma_x - v_xc|p>. On Hartree-Fock v_xc IS Sigma_x: the shift is
        # zero analytically and round-off numerically, and it is not applied
        # there, these numbers being the oracle every cubic gate is measured
        # from.
        is_ks, _ = xc_hybrid_coeff(mf)
        return mf, eri, nocc, QPqb(mf.mo_energy, eri, nocc,
                                   screening=self.screening,
                                   delta=(static_exchange_diagonal(
                                       mf, mf.mol,
                                       np.arange(mf.mo_coeff.shape[1]),
                                       exchange='mf') if is_ks else None))

    def _choose(self, mol, mf=None):
        """(orbital, quasiparticle energy) of the lowest-energy process.

        The largest quasiparticle energy among the occupied is the easiest
        electron to remove; the smallest among the virtual is the easiest to
        add. Both are read from the QUASIPARTICLE ordering rather than the mean
        field's, because G0W0 reorders states -- on N2 the mean-field HOMO and
        the quasiparticle HOMO are different orbitals, which is the whole
        discrepancy on that row of the 2024 paper's Table II.
        """
        mf, _, nocc, qp = self._build(mol, mf)
        if self.charge_change == -1:
            w = [qp.solve_diag(p)[0] for p in range(nocc)]
            p = int(np.argmax(w))
            return p, w[p]
        hi = min(mol.nao, nocc + ATTACHMENT_SEARCH)
        w = [qp.solve_diag(p)[0] for p in range(nocc, hi)]
        k = int(np.argmin(w))
        return nocc + k, w[k]

    @property
    def physics_excitation(self):
        """E^{N-/+1}: the orbital eps^QP is taken at, and which way the
        electron count moves."""
        return ChargedExcitation(int(self.orbital), self.charge_change)

    @property
    def physics(self):
        """What this surface computes: E_HF + E_c^dRPA -/+ eps^QP_p, gas phase.

        The TDA-screened surface is refused rather than declared: its E_0 is
        E_HF[rho] with E_c^dRPA = 0, which no `GroundState` names, and calling
        it kind='rpa' would let `compare_surfaces` pair it with
        `DenseRPASurface` and leave -E_c^dRPA in an adiabatic ionization.
        """
        if self.screening != 'rpa':
            raise ValueError(
                f'screening={self.screening!r} screens W in the Tamm-Dancoff '
                f'approximation, whose E_0 carries no E_c^dRPA: declaring it '
                f'on the dRPA ground state would put two functionals under one '
                f'name.')
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation)

    def _state(self, mol, mf=None):
        """(E, quasiparticle energy, mf, eri, nocc, qp, E_0) at `mol`."""
        mf, eri, nocc, qp = self._build(mol, mf)
        if self.screening == 'rpa':
            # E_c^dRPA enters the total energy here and nowhere else: the
            # Tamm-Dancoff surface is E_HF -/+ eps^QP and carries none.
            refuse_a_solvated_mean_field(mf, 'QuasiparticleSurface')
        w, _ = qp.solve_diag(self.orbital, w0=self._seed)
        self._log_branch(mol, qp)
        e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                 e_corr=(qp.qb.e_corr()
                                         if self.screening == 'rpa' else 0.0))
        return e0.total + self.sign * w, w, mf, eri, nocc, qp, e0

    def _log_branch(self, mol, qp):
        """Keep in `branch_log`, once per geometry, the followed root the
        largest weight left at `mol` (`QPqb.solve_diag`, warned there)."""
        log = self.__dict__.setdefault('branch_log', [])
        coords = np.asarray(mol.atom_coords(), float).tolist()
        for entry in qp.branch_log:
            entry = dict(entry, coords=coords)
            if entry not in log:
                log.append(entry)

    def total_energy(self, mol=None, mf=None):
        return lockstep(float(self._state(self.mol0 if mol is None else mol,
                                          mf)[0]))

    def quasiparticle_energy(self, mol=None, mf=None):
        """eps^QP of the surface's orbital at `mol`, in Hartree, without a
        force: on any mean field, including one a force is refused on
        (`refuse_mismatched_integrals`)."""
        return lockstep(float(self._state(self.mol0 if mol is None else mol,
                                          mf)[1]))

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics) for the N-/+1 state, fully analytic."""
        mol = self.mol0 if mol is None else mol
        mf = self._scf(mol) if mf is None else mf
        refuse_a_solvated_mean_field(mf, 'QuasiparticleSurface', force=True)
        refuse_mismatched_integrals(mf, mol, None, 'QuasiparticleSurface',
                                    auxbasis=False)
        energy, w, mf, eri, nocc, qp, e0 = self._state(mol, mf)
        norb = mol.nao

        g_fock, g_eri, target, weight = qp_partials_with_shift(qp, self.orbital,
                                                               w)
        g_fock, g_eri = self.sign * g_fock, self.sign * g_eri
        if self.screening == 'rpa':
            # The ground-state correlation energy rides along, and the
            # quasi-boson amplitudes respond to the added or removed electron:
            # the Z vector is what carries that response into the density.
            g_fock_c, g_eri_c = rpa_partials(qp.qb, nocc, norb)
            g_fock += g_fock_c
            g_eri += g_eri_c
            add_z_contribution(qp.qb, solve_Z(qp.qb, self.sign * target),
                               nocc, norb, g_fock, g_eri)
        # The static shift reaches the root through one matrix element, so it
        # reaches the force through one orbital, with that root's own weight.
        shift = None
        if qp.delta.any():
            shift = np.zeros(norb)
            shift[self.orbital] = self.sign * weight
        y_extra, g_extra = kohn_sham_gradient_correction(mol, mf, nocc, shift)
        (g_corr,), (diag,) = correlation_gradients(mol, mf,
                                                   [(g_fock, g_eri, y_extra)],
                                                   eri_mo=eri)
        grad = mean_field_skeleton_force(mf) + g_corr + g_extra
        return lockstep((grad, float(energy), {
            'qp_energy_eV': float(w) * HARTREE_TO_EV,
            'orbital': int(self.orbital),
            'e0_terms': e0.terms,
            'stationarity': float(diag['stationarity'])}))

    def refreeze(self, mol, mf=None):
        """The same process with the orbital re-chosen at `mol`. mf: not
        read; the surface converges its own SCF at `mol`."""
        return QuasiparticleSurface(mol, self._scf, self.charge_change,
                                    self.screening)

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        kind = 'E^(N-1)' if self.charge_change == -1 else 'E^(N+1)'
        method = 'G0W0' if self.screening == 'rpa' else 'G0W0-TDA'
        return f'{kind} {method}, orbital {self.orbital}'


class DenseBSESurface(PotentialEnergySurface):
    """E_0 + Omega_nu(R) on the dense quasi-boson BSE@G0W0 route.

    variant: one of VARIANTS. state: which BSE root, by index at the reference
    geometry; the root is followed by index thereafter, which is why the paper's
    systems are chosen so the state of interest is the lowest singlet.

    The quasiparticle set and its roots are frozen at `mol` on construction and
    pinned everywhere else, so the surface is continuous. `refreeze` rebuilds
    them and is the honest error bar on any minimum found here. The frozen
    roots are the largest-weight ones at `mol`, and every geometry
    follows each from its frozen root between the two poles that bracket it
    there, wherever they have moved (`QPqb.solve_diag` with a seed and its
    labels; the interval holding the frozen energy would land on a satellite
    once a pole drifts across it): a largest-weight root
    decided per geometry jumps branch where two roots share the weight
    (formaldehyde/cc-pVDZ orbital 22, Z 0.19: 20 mHa within 1e-3 Bohr).

    Truly degenerate orbitals of the set (closer than QP_DEGENERACY_TOL
    at `mol`, with their own reference corrections within
    QP_BLOCK_CORRECTION_TOL) form blocks frozen with it, each carrying one
    shared correction (`BSEqb` blocks); `qp_block_record` says why every
    candidate was kept or rejected.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'diagnostics', 'excitation', 'excited_states',
                             'mean_field', 'physics', 'qp_set', 'spin'})

    #: The root index (`surface.root_attribute`).
    ROOT_ATTR = 'state'

    def __init__(self, mol, variant='BSE@GW', state=0, qp_orbs=None,
                 filter_z=False, scf=None, auxbasis=None, eri_blocks=False,
                 spin='singlet'):
        """eri_blocks: build only the MO blocks the energy needs.

        The full (pq|rs) is nao^4, which is what puts the letter's own basis
        out of reach on a workstation. The energy touches (ia|jb), (ij|ab) and
        (pq|ia) alone, about a factor nao^2 / (nocc * nvirt) smaller. Energies
        only: gradients need the whole tensor and refuse.

        filter_z: True drops from the set the orbitals whose root has a pole
        strength at or below QP_WINDOW_Z_MIN at `mol` (warned and flagged);
        False, the default, keeps the whole window with every root.
        """
        self.eri_blocks = eri_blocks
        self.mol0 = mol
        self.variant = variant
        # Kappa weights the bare exchange alone; the screening and the
        # quasiparticle set are spin-independent, so a triplet surface differs
        # from its singlet only in the BSE kernel and reuses everything above.
        self.spin = spin
        self.state = state
        self.filter_z = filter_z
        # The starting point is free here too, the letter's second one being
        # Kohn-Sham. A Kohn-Sham reference moves E_0 to E_HF[rho], which
        # `_solve` repairs, and puts <p|Sigma_x - v_xc|p> on every solved
        # orbital: a BSE root reads that shift through every orbital of the
        # quasiparticle set at once, each at its weight in the root
        # (`_omega_partials`), and both terms carry their nuclear derivatives.
        self._scf = scf or tight_rhf
        self.auxbasis = auxbasis
        cfg = VARIANTS[variant]
        self.screening = cfg['screening']
        self.bse_tda = cfg['bse_tda']
        self._qp_orbs_in = qp_orbs
        mf = self.scf_factory(mol)
        # The starting point is free here -- Kohn-Sham included -- so the
        # functional is read off the mean field rather than assumed.
        self.physics_ground_state = declared_ground_state(mf, 'rpa')
        nocc = mol.nelectron // 2
        eri = self._eri(mol, mf)
        window = (qp_window(mol, mol.nao, nocc) if qp_orbs is None
                  else list(qp_orbs))
        # ONE plain Z > 0.5 solve decides the set; everything downstream pins
        # it. filter_z=False keeps the whole window instead, which is what the
        # cubic chain does and what a like-for-like route comparison needs.
        probe = BSEqb(mf, eri, nocc, screening=self.screening,
                      bse_tda=self.bse_tda, qp_orbs=window,
                      pinned=not filter_z, delta=self.xc_shift(mf),
                      spin=spin)
        self.qp_set = sorted(probe.qp_roots)
        # The reference roots, followed elsewhere between the poles that
        # bracket them here.
        self.seeds = dict(probe.qp_roots)
        self.seed_labels = dict(probe.qp_labels)
        self.qp_z = dict(probe.qp_z)
        # the probe solves without blocks: its roots are each orbital's own
        blocks, record = true_degenerate_blocks(
            mf.mo_energy, {p: w - mf.mo_energy[p] for p, w in
                           self.seeds.items()},
            self.qp_set, QP_DEGENERACY_TOL, QP_BLOCK_CORRECTION_TOL)
        self.qp_blocks = tuple(blocks)
        self.qp_block_record = dict(
            record, energy_tol_Ha=QP_DEGENERACY_TOL,
            correction_tol_Ha=QP_BLOCK_CORRECTION_TOL)

    @property
    def physics_excitation(self):
        """The `Excitation` this surface's variant and settings mean.

        The two TDA-SCREENED variants are refused rather than declared: the
        declaration's `screening` names how W was built and knows dRPA only, so
        calling 'BSE@GWtda' an RPA-screened state would put two different
        kernels under one name. Their E_0 is still the same functional with
        E_c^dRPA = 0 (letter Eq. 19), which is why `physics_ground_state` does
        not refuse them.
        """
        if self.screening != 'rpa':
            raise ValueError(
                f'variant={self.variant!r} screens W in the Tamm-Dancoff '
                f'approximation, and `Excitation.screening` names dRPA only: '
                f'declaring one for the other would put two kernels under one '
                f'name.')
        return Excitation(spin=self.spin, root=self.state + 1,
                          kernel='bse-tda' if self.bse_tda else 'bse')

    @property
    def physics(self):
        """What this surface computes: E_HF + E_c^dRPA + Omega_nu, gas phase.

        `refuse_a_solvated_mean_field` guards the ground state, so the
        environment is never anything but `SurfacePhysics`'s default.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation)

    def xc_shift(self, mf):
        """<p|Sigma_x - v_xc|p>, the static shift a Kohn-Sham reference needs.

        Zero on Hartree-Fock to round-off, where v_xc IS Sigma_x, so this is
        safe to apply unconditionally: the HF route stays what it was. It is
        production's own function, the one the cubic chain differentiates, so
        both routes mean the same thing by it.
        """
        return static_exchange_diagonal(mf, mf.mol,
                                        np.arange(mf.mo_coeff.shape[1]),
                                        exchange='mf')

    def _eri(self, mol, mf):
        """(pq|rs), exact or through the same auxiliary basis the cubic route
        fits in -- the RI step of the dense-vs-cubic decomposition.

        With eri_blocks the same integrals come back as the three blocks the
        energy needs, which is the same numbers without the nao^4 array.
        """
        auxmol = self._auxmol(mol)
        if self.eri_blocks:
            return MOEriBlocks.from_mol(mol, mf.mo_coeff,
                                        mol.nelectron // 2, auxmol=auxmol)
        if auxmol is None:
            return mo_eri(mf, mol)
        return df_eri_mo(mol, auxmol, mf.mo_coeff)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on -- a gas-phase
        surface, so the factory's own; `mf` given is used as it is."""
        mol = self.mol0 if mol is None else mol
        return mol, (self._scf(mol) if mf is None else mf)

    def excited_states(self, mol=None, mf=None):
        """(omega, X, Y, nocc): every root the dense BSE solved at `mol` (the
        `excited_states` ingredient of `src.gradients.surface`)."""
        mol, mf = self.mean_field(mol, mf)
        solved = self._solve(mol, mf)[0]
        return solved.Omega, solved.X, solved.Y, mol.nelectron // 2

    def scf_factory(self, mol):
        return self._scf(mol)

    def _e0(self, mol, mf, b):
        """The plasmon ground state this excitation sits on, with its terms.

        E_0 is E_HF + E_c^dRPA, and `mf.e_tot` is E_HF only on a Hartree-Fock
        reference: on a Kohn-Sham one the exchange is not the exact one and the
        correlation inside E_xc would be counted a second time by E_c^dRPA.
        The repair is identically zero on Hartree-Fock, so that route is
        bit-unchanged.

        Under TDA screening the plasmon formula's correlation vanishes (letter
        Eq. 19), so E_c^dRPA is zero and the two screening variants sit on
        different ground states.
        """
        return ground_state_energy(
            declared_ground_state(mf, 'rpa'), mf, mol,
            e_corr=b.qp.qb.e_corr() if self.screening == 'rpa' else 0.0)

    def _solve(self, mol, mf):
        refuse_a_foreign_mean_field(mf, mol, 'DenseBSESurface')
        nocc = mol.nelectron // 2
        eri = self._eri(mol, mf)
        # the largest-weight roots decided at the reference, followed
        b = BSEqb(mf, eri, nocc, screening=self.screening, bse_tda=self.bse_tda,
                  qp_orbs=self.qp_set, pinned=True, seeds=self.seeds,
                  seed_labels=self.seed_labels,
                  delta=self.xc_shift(mf), spin=self.spin,
                  blocks=self.qp_blocks)
        return b, eri, nocc, mol.nao, self._e0(mol, mf, b).total

    def total_energy(self, mol=None, mf=None):
        """E_0 + Omega_nu. E_0 carries E_c^dRPA under RPA screening, so a
        continuum is refused there and not on `excitation`, which
        reports Omega alone."""
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        if self.screening == 'rpa':
            refuse_a_solvated_mean_field(mf, 'DenseBSESurface')
        b, _, _, _, e0 = self._solve(mol, mf)
        return lockstep(e0 + float(b.Omega[self.state]))

    def excitation(self, mol=None, mf=None):
        """Omega_nu alone, without a force.

        Omega carries no E_c, so a continuum is not refused here, while the
        force on it is (`excitation_gradient`).
        """
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        b, _, _, _, _ = self._solve(mol, mf)
        return lockstep(float(b.Omega[self.state]))

    def excitation_gradient(self, mol=None, mf=None):
        """(dOmega/dR, Omega, diagnostics) -- the excitation energy alone.

        The quantity to compare against `ExcitedStateChain.excitation_gradient`:
        Omega carries no E_HF and no E_c, so the two routes mean the same thing
        by it, which is not true of their total energies.

        Refuses a Kohn-Sham `mf` through `BSEqbAdjoint.partials`, per the class
        docstring: the shift's gradient on a BSE root is not carried here.
        """
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        refuse_a_solvated_mean_field(mf, 'DenseBSESurface', force=True)
        refuse_mismatched_integrals(mf, mol, self._auxmol(mol),
                                    'DenseBSESurface')
        b, eri, nocc, norb, _ = self._solve(mol, mf)
        gF, G4, tg, shift = self._omega_partials(b, mf)
        if self.screening == 'rpa':
            add_z_contribution(b.qp.qb, solve_Z(b.qp.qb, tg), nocc, norb,
                               gF, G4)
        target, g_extra = (gF, G4), 0.0
        if shift is not None:
            # Omega alone: the shift's terms, not E_0's double counting
            target += (qp_xc_correction_Y(mf, shift, nocc),)
            g_extra = qp_xc_correction_skeleton(mf, shift, nocc)
        aux = self._auxmol(mol)
        Gs, diags = correlation_gradients(mol, mf, [target], eri_mo=eri,
                                          auxmol=aux)
        omega = float(b.Omega[self.state])
        return lockstep((Gs[0] + g_extra, omega, {
            'omega': omega, 'stationarity': float(diags[0]['stationarity']),
            'n_qp': len(self.qp_set)}))

    def _auxmol(self, mol):
        """The auxiliary basis on THIS geometry, or None without one.

        Built from `atom_coords()`, the coordinates the integrals are evaluated
        at, so the energy (`_eri`) and the gradient fit in one auxiliary basis
        by construction rather than by two spellings of the same geometry.
        """
        if self.auxbasis is None:
            return None
        return gto.M(atom=[(mol.atom_pure_symbol(i), tuple(c)) for i, c
                           in enumerate(mol.atom_coords())],
                     unit='Bohr', basis=self.auxbasis, verbose=0)

    def total_gradient(self, mol=None, mf=None):
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        refuse_a_solvated_mean_field(mf, 'DenseBSESurface', force=True)
        refuse_mismatched_integrals(mf, mol, self._auxmol(mol),
                                    'DenseBSESurface')
        b, eri, nocc, norb, e0 = self._solve(mol, mf)
        gF, G4, tg, shift = self._omega_partials(b, mf)
        if self.screening == 'rpa':
            gFd, G4d = rpa_partials(b.qp.qb, nocc, norb)
            gF += gFd
            G4 += G4d
            Zm = solve_Z(b.qp.qb, tg)
            add_z_contribution(b.qp.qb, Zm, nocc, norb, gF, G4)
        target, g_extra = (gF, G4), 0.0
        if shift is not None:
            # E_0 = E_HF[rho] and the shift, each with its own derivative
            y_extra, g_extra = kohn_sham_gradient_correction(mol, mf, nocc,
                                                             shift)
            target += (y_extra,)
        Gs, diags = correlation_gradients(mol, mf, [target], eri_mo=eri,
                                          auxmol=self._auxmol(mol))
        omega = float(b.Omega[self.state])
        return lockstep((mean_field_skeleton_force(mf) + Gs[0] + g_extra,
                         e0 + omega,
                         {'omega': omega, 'e0': e0,
                          'e0_terms': self._e0(mol, mf, b).terms,
                          'stationarity': float(diags[0]['stationarity']),
                          'n_qp': len(self.qp_set)}))

    def _omega_partials(self, b, mf):
        """(gammaF, Gamma4, t_grad, shift weights) of Omega_state; the weights
        are None on Hartree-Fock, where the shift is round-off and the
        (F, ERI, t) partials are the whole derivative."""
        if not xc_hybrid_coeff(mf)[0]:
            return b.partials(self.state) + (None,)
        return b.partials_with_shift(self.state)

    def diagnostics(self, mol, mf=None, posteriori=False):
        """This surface's bookkeeping of its state at `mol`, the record fields
        beside the energies (`diagnostics.chain_diagnostics`)."""
        return chain_diagnostics(self, mol, mf, posteriori)

    def refreeze(self, mol, mf=None):
        """The same surface with the quasiparticle set rebuilt at `mol`
        (mf: not read; the surface converges its own SCF at `mol`).

        EVERY constructor setting travels. A refreeze that dropped one would
        change which functional the walk is on halfway through it: without
        `spin` a triplet comes back a singlet, 0.75 eV below itself on water,
        and without `eri_blocks` a surface that fits in memory rebuilds the
        full nao^4 tensor at the next geometry. The largest-weight roots are
        decided again at `mol`, the branch the walk follows from there.
        """
        return DenseBSESurface(mol, self.variant, self.state,
                               qp_orbs=self._qp_orbs_in,
                               filter_z=self.filter_z, scf=self._scf,
                               auxbasis=self.auxbasis,
                               eri_blocks=self.eri_blocks, spin=self.spin)

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        return f'{self.variant}[S{self.state + 1}]'
