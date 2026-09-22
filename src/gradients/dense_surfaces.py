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
"""
import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import (ChargedExcitation, Excitation,
                                  SurfacePhysics)
from src.Base.eri_blocks import MOEriBlocks, df_eri_mo, mo_eri
from src.Base.environment import dresses_interaction, environment_of
from src.Base.isdf_jk import mean_field_skeleton_force
from src.SingleReference.GW.qp_states import valence_qp_states
from src.SingleReference.LinearResponse.rpa_energy import (
    declared_ground_state, ground_state_energy, xc_hybrid_coeff)
from src.gradients.grad_engine import correlation_gradients
from src.gradients.isdf_derivatives import (exx_double_counting_Y,
                                            exx_double_counting_skeleton,
                                            qp_xc_correction,
                                            qp_xc_correction_Y,
                                            qp_xc_correction_skeleton)
from src.gradients.quasi_boson_adjoint import (BSEqbAdjoint as BSEqb,
                                               QPqbAdjoint as QPqb)
from src.gradients.targets import (add_z_contribution, rpa_partials,
                                   qp_partials_with_shift, solve_Z)

#: How many virtual orbitals above the Fermi level are searched for the
#: quasiparticle LUMO. G0W0 reorders states relative to the mean field, so the
#: lowest attachment is not always the first virtual, but it is never far.
ATTACHMENT_SEARCH = 4

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
    energy inherits directly.
    """
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF did not converge; nothing follows from it')
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


def refuse_a_solvated_mean_field(mf, mol, surface):
    """Refuse a mean field carrying a continuum on a route that reports E_c^dRPA.

    `mo_eri` is a raw four-index transform of the BARE interaction and consults
    no environment, so E_c^dRPA here is the correlation energy of v alone. A
    polarizable environment dresses every post-SCF interaction, v -> v + vtilde,
    and `RPAGroundStateChain` on the SAME mean field builds E_c from the dressed
    one: the two are different functionals, against the tolerance a relaxed
    geometry needs.

    TWO MARKERS, EITHER OF WHICH IS A CONTINUUM. pyscf's PCM sits on the mean
    field as `with_solvent` and has already relaxed the orbitals inside the
    reaction field; a wicks environment reaches post-SCF code as the attached
    `with_screening` and dresses the interaction. The second is asked through
    `dresses_interaction`, whose answer is a property of the environment and not
    of the basis it is asked in, so `mol` -- the basis this route expands
    (pq|rs) in, there being no auxiliary one -- is the honest thing to ask it
    about. Fixed point charges screen nothing and are left alone.

    The check sits where E_c is formed rather than on the stored factory, so a
    mean field handed straight to `total_energy` cannot walk past it.
    """
    if hasattr(mf, 'with_solvent'):
        carries = 'its SCF was relaxed inside a PCM reaction field'
    elif dresses_interaction(environment_of(mf), mol):
        carries = f'{environment_of(mf)!r} is attached to it and dresses v'
    else:
        return
    raise ValueError(
        f'{surface} reports E_HF + E_c^dRPA built from the BARE (pq|rs), but '
        f'this mean field carries a continuum: {carries}. The dense route has '
        f'no way to see it, so the correlation energy it would return is a '
        f'different functional from the one the cubic chain returns on the '
        f'same mean field. Use RPAGroundStateChain, or RPABSESurface / '
        f'RPAQPSurface composed on it, which carry the environment through '
        f'the auxiliary metric. Excitation energies and quasiparticle levels '
        f'are differences and are unaffected.')


class DenseRPASurface:
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
        refuse_a_solvated_mean_field(mf, mol, 'DenseRPASurface')
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
        return ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                   e_corr=qp.qb.e_corr()).total

    def total_gradient(self, mol=None, mf=None):
        """(dE_0/dR, E_0, diagnostics), fully analytic."""
        mol = self.mol0 if mol is None else mol
        mf, eri, nocc, qp = self._build(mol, mf)
        e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                 e_corr=qp.qb.e_corr())
        g_fock, g_eri = rpa_partials(qp.qb, nocc, mol.nao)
        y_extra, g_extra = kohn_sham_gradient_correction(mol, mf, nocc)
        (g_corr,), (diag,) = correlation_gradients(mol, mf,
                                                   [(g_fock, g_eri, y_extra)],
                                                   eri_mo=eri)
        grad = mean_field_skeleton_force(mf) + g_corr + g_extra
        return grad, e0.total, {
            'e_corr': float(qp.qb.e_corr()),
            'e0_terms': e0.terms,
            'stationarity': float(diag['stationarity'])}

    def refreeze(self, mol):
        """Nothing is frozen on this surface; the copy is for the protocol."""
        return DenseRPASurface(mol, self._scf)

    def label(self):
        return 'E_HF + E_c(dRPA)'


class QuasiparticleSurface:
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
        self._seed = None
        mf = self._scf(mol)
        # The starting point is free -- Kohn-Sham included -- so the functional
        # is read off the mean field rather than assumed.
        self.physics_ground_state = declared_ground_state(mf, 'rpa')
        if orbital is None:
            self.orbital, self._seed = self._choose(mol, mf)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on -- a gas-phase
        surface, so the factory's own; `mf` given is used as it is."""
        mol = self.mol0 if mol is None else mol
        return mol, (self._scf(mol) if mf is None else mf)

    def scf_factory(self, mol):
        return self._scf(mol)

    def _build(self, mol, mf=None):
        mf = self._scf(mol) if mf is None else mf
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
                                   delta=qp_xc_correction(mf) if is_ks else None)

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
        """What this surface computes: E_HF + E_c^dRPA -/+ eps^QP_p, gas phase."""
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation)

    def _state(self, mol, mf=None):
        """(E, quasiparticle energy, mf, eri, nocc, qp, E_0) at `mol`."""
        mf, eri, nocc, qp = self._build(mol, mf)
        if self.screening == 'rpa':
            # E_c^dRPA enters the total energy here and nowhere else: the
            # Tamm-Dancoff surface is E_HF -/+ eps^QP and carries none.
            refuse_a_solvated_mean_field(mf, mol, 'QuasiparticleSurface')
        w, _ = qp.solve_diag(self.orbital, w0=self._seed)
        self._seed = w
        e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, mol,
                                 e_corr=(qp.qb.e_corr()
                                         if self.screening == 'rpa' else 0.0))
        return e0.total + self.sign * w, w, mf, eri, nocc, qp, e0

    def total_energy(self, mol=None, mf=None):
        return float(self._state(self.mol0 if mol is None else mol, mf)[0])

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics) for the N-/+1 state, fully analytic."""
        mol = self.mol0 if mol is None else mol
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
        return grad, float(energy), {
            'qp_energy_eV': float(w) * HARTREE_TO_EV,
            'orbital': int(self.orbital),
            'e0_terms': e0.terms,
            'stationarity': float(diag['stationarity'])}

    def refreeze(self, mol):
        """The same process with the orbital re-chosen at `mol`."""
        return QuasiparticleSurface(mol, self._scf, self.charge_change,
                                    self.screening)

    def label(self):
        kind = 'E^(N-1)' if self.charge_change == -1 else 'E^(N+1)'
        method = 'G0W0' if self.screening == 'rpa' else 'G0W0-TDA'
        return f'{kind} {method}, orbital {self.orbital}'


class DenseBSESurface:
    """E_0 + Omega_nu(R) on the dense quasi-boson BSE@G0W0 route.

    variant: one of VARIANTS. state: which BSE root, by index at the reference
    geometry; the root is followed by index thereafter, which is why the paper's
    systems are chosen so the state of interest is the lowest singlet.

    The quasiparticle set and its roots are frozen at `mol` on construction and
    pinned everywhere else, so the surface is continuous. `refreeze` rebuilds
    them and is the honest error bar on any minimum found here.

    THIS SURFACE'S GRADIENT REFUSES A KOHN-SHAM REFERENCE. A BSE root reads a
    Kohn-Sham static shift through EVERY orbital of the quasiparticle window at
    once rather than through the single matrix element a quasiparticle energy
    reads it through, and `BSEqb.partials` (`quasi_boson_adjoint.py`) refuses to
    differentiate that shift rather than silently dropping it -- an
    acknowledged open item of the dense route, not a porting defect, and left
    as it stands here.
    """

    def __init__(self, mol, variant='BSE@GW', state=0, qp_orbs=None,
                 filter_z=True, scf=None, auxbasis=None, eri_blocks=False,
                 spin='singlet'):
        """eri_blocks: build only the MO blocks the energy needs.

        The full (pq|rs) is nao^4, which is what puts the letter's own basis
        out of reach on a workstation. The energy touches (ia|jb), (ij|ab) and
        (pq|ia) alone, about a factor nao^2 / (nocc * nvirt) smaller. Energies
        only: gradients need the whole tensor and refuse.
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
        # Kohn-Sham. What a Kohn-Sham reference breaks is E_0, which `_solve`
        # repairs, and the GRADIENT, which `BSEqb.partials` refuses by the size
        # of the shift it would have to differentiate: a BSE root reads the
        # shift through every orbital of the quasiparticle window at once, not
        # through the single matrix element a quasiparticle energy reads it
        # through.
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
        self.seeds = dict(probe.qp_roots)

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
        return qp_xc_correction(mf)

    def _eri(self, mol, mf):
        """(pq|rs), exact or through the same auxiliary basis the cubic route
        fits in -- the RI step of the dense-vs-cubic decomposition.

        With eri_blocks the same integrals come back as the three blocks the
        energy needs, which is the same numbers without the nao^4 array.
        """
        auxmol = (None if self.auxbasis is None
                  else gto.M(atom=mol.atom, basis=self.auxbasis,
                             unit=mol.unit, verbose=0))
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
        nocc = mol.nelectron // 2
        eri = self._eri(mol, mf)
        b = BSEqb(mf, eri, nocc, screening=self.screening, bse_tda=self.bse_tda,
                  qp_orbs=self.qp_set, seeds=self.seeds, pinned=True,
                  delta=self.xc_shift(mf), spin=self.spin)
        return b, eri, nocc, mol.nao, self._e0(mol, mf, b).total

    def total_energy(self, mol=None, mf=None):
        """E_0 + Omega_nu. E_0 carries E_c^dRPA under RPA screening, so a
        continuum is refused there and not on `excitation_gradient`, which
        reports Omega alone."""
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        if self.screening == 'rpa':
            refuse_a_solvated_mean_field(mf, mol, 'DenseBSESurface')
        b, _, _, _, e0 = self._solve(mol, mf)
        return e0 + float(b.Omega[self.state])

    def excitation_gradient(self, mol=None, mf=None):
        """(dOmega/dR, Omega, diagnostics) -- the excitation energy alone.

        The quantity to compare against `ExcitedStateChain.excitation_gradient`:
        Omega carries no E_HF and no E_c, so the two routes mean the same thing
        by it, which is not true of their total energies.

        Refuses a Kohn-Sham `mf` through `BSEqb.partials`, per the class
        docstring: the shift's gradient on a BSE root is not carried here.
        """
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        b, eri, nocc, norb, _ = self._solve(mol, mf)
        gF, G4, tg = b.partials(self.state)
        if self.screening == 'rpa':
            add_z_contribution(b.qp.qb, solve_Z(b.qp.qb, tg), nocc, norb,
                               gF, G4)
        aux = self._auxmol(mol)
        Gs, diags = correlation_gradients(mol, mf, [(gF, G4)], eri_mo=eri,
                                          auxmol=aux)
        omega = float(b.Omega[self.state])
        return Gs[0], omega, {'omega': omega,
                              'stationarity': float(diags[0]['stationarity']),
                              'n_qp': len(self.qp_set)}

    def _auxmol(self, mol):
        """The auxiliary basis on THIS geometry.

        Built from `atom_coords()` rather than `mol.atom`: a molecule that has
        been through `set_geom_` carries new coordinates but the string it was
        created from, so reading `mol.atom` would silently fit the reference
        geometry at every displaced one.
        """
        if self.auxbasis is None:
            return None
        return gto.M(atom=[(mol.atom_pure_symbol(i), tuple(c)) for i, c
                           in enumerate(mol.atom_coords())],
                     unit='Bohr', basis=self.auxbasis, verbose=0)

    def total_gradient(self, mol=None, mf=None):
        mol = mol or self.mol0
        mf = mf or self.scf_factory(mol)
        if self.screening == 'rpa':
            refuse_a_solvated_mean_field(mf, mol, 'DenseBSESurface')
        b, eri, nocc, norb, e0 = self._solve(mol, mf)
        gF, G4, tg = b.partials(self.state)
        if self.screening == 'rpa':
            gFd, G4d = rpa_partials(b.qp.qb, nocc, norb)
            gF += gFd
            G4 += G4d
            Zm = solve_Z(b.qp.qb, tg)
            add_z_contribution(b.qp.qb, Zm, nocc, norb, gF, G4)
        Gs, diags = correlation_gradients(mol, mf, [(gF, G4)], eri_mo=eri,
                                          auxmol=self._auxmol(mol))
        omega = float(b.Omega[self.state])
        return (mf.Gradients().kernel() + Gs[0], e0 + omega,
                {'omega': omega, 'e0': e0,
                 'e0_terms': self._e0(mol, mf, b).terms,
                 'stationarity': float(diags[0]['stationarity']),
                 'n_qp': len(self.qp_set)})

    def refreeze(self, mol):
        """The same surface with the quasiparticle set rebuilt at `mol`.

        EVERY constructor setting travels. A refreeze that dropped one would
        change which functional the walk is on halfway through it: without
        `spin` a triplet comes back a singlet, well off itself, and without
        `eri_blocks` a surface that fits in memory rebuilds the full nao^4
        tensor at the next geometry.
        """
        return DenseBSESurface(mol, self.variant, self.state,
                               qp_orbs=self._qp_orbs_in,
                               filter_z=self.filter_z, scf=self._scf,
                               auxbasis=self.auxbasis,
                               eri_blocks=self.eri_blocks, spin=self.spin)

    def label(self):
        return f'{self.variant}[S{self.state + 1}]'
