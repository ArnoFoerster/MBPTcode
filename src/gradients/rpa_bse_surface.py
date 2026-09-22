"""E_nu = E_0^dRPA + Omega_nu, the excited-state surface of Toelle Eq. (14).

Tolle, Kitsaras and Loos (arXiv:2507.02160) Eq. (14) puts the BSE@G0W0 energy
of state nu at E_nu = E_0 + Omega_nu, and Eq. (15) fixes E_0 by the plasmon
(Klein) trace formula

    E_0 = E_0^HF + (1/2) sum_beta Omega_beta^RPA - (1/2) Tr(A^RPA)
        = E_HF + E_c^dRPA ,

not the mean field's own energy. `ExcitedStateChain` returns `mf.e_tot + Omega`
and so drops E_c^dRPA; these surfaces compose it with `RPAGroundStateChain` and
add the two gradients.

E_c^dRPA depends on the geometry, so omitting it moves the surface rather than
only its zero. By Eq. (19) it vanishes when the TDA is enforced at the RPA
level, which puts the two screening variants on different ground states.

    surface = RPABSESurface(mol, scf_factory, state=0, spin='singlet')
    g, e, diags = surface.total_gradient(mol)     # analytic, cubic

E_0^HF is the Hartree-Fock energy, which a Kohn-Sham mean field does not
supply, so the ground-state chain also carries E_x^exact - E_xc -- exact
exchange restored, the density-functional double counting removed -- with its
orbital response in the shared multiplier solve. Every starting point reaches
the same E_0, and an excited-state geometry on this surface is on the dRPA
ground state by construction.

Both halves hold one `FrozenFactorization`, so a geometry costs one fit.
"""
import numpy as np

from src.Base.declaration import ChargedExcitation, SurfacePhysics
from src.Base.environment import environment_label
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FrozenFactorization
from src.gradients.rpa_ground_state import RPAGroundStateChain


def _one_factorization(mol, shared, ground, excited, given):
    """The single FrozenFactorization both halves of a composed surface use.

    `separable_factors` is deterministic given its settings, so two chains
    handed matching keywords produce bitwise identical factors and merely pay
    for the fit twice. One object makes the sharing structural rather than a
    property of this constructor passing the same keywords to both.

    An explicitly given one wins; otherwise an already-built half's is adopted,
    so a surface reassembled from existing chains does not manufacture a second
    factorization behind them.
    """
    if given is not None:
        return given
    for half in (ground, excited):
        if half is not None and getattr(half, 'factorization', None) is not None:
            return half.factorization
    return FrozenFactorization(
        mol, basis=shared.get('basis'), auxbasis=shared.get('auxbasis'),
        counts=shared.get('counts'), n_start=shared.get('n_start', 1),
        frames=shared.get('frames', 'frozen'), radii=shared.get('radii'))


class RPAQPSurface:
    """E^{N-/+1} = E_0^dRPA -/+ eps_p^QP: the same ruling, for a QUASIPARTICLE.

    A G0W0 ionization or attachment geometry optimization has exactly the
    ground-state problem the BSE one has -- the total energy of the N-/+1
    system is E_0 -/+ eps^QP, and E_0 is Toelle Eq. (15)'s, not the mean
    field's. Optimizing a cation on `mf.e_tot - eps^QP` puts it on a different
    ground-state surface from the neutral it will be compared with.

    THE QUASIPARTICLE ENERGY ENTERS WITH THE ELECTRON COUNT'S SIGN REVERSED.
    Removing an electron leaves E^{N-1} = E_0 - eps_h, which is above E_0
    because eps_h is negative; adding one gives E^{N+1} = E_0 + eps_l. `state`
    is the offset from the HOMO, so state <= 0 is an ionization and state > 0
    an attachment -- the same convention `QuasiparticleSurface` carries as
    `charge_change`, and a surface on E_0 + eps for every state would report a
    negative ionization potential and relax the cation on the wrong sign of the
    quasiparticle force.

    AN ABSOLUTE TOTAL ENERGY IN A CONTINUUM IS NOT TRUSTWORTHY. The exact block
    fold of the ACFDT log-determinant keeps the BARE interaction in the linear
    counter-term while this code dresses it, and the dressing throws away the
    leading solute-solvent dispersion term. Quasiparticle levels, excitation
    energies and any difference taken at a FIXED geometry are free of it, as is
    every gas-phase total energy.
    """

    def __init__(self, mol, scf_factory, state=0, mf=None, ground=None,
                 excited=None, factorization=None, **kw):
        # `radii` is SHARED, not excited-only: it decides the factorization
        # both halves read, so leaving it in kw builds the factorization
        # without it and the chain then refuses the mismatch.
        shared = {k: kw.pop(k) for k in
                  ('basis', 'auxbasis', 'counts', 'n_start', 'frames',
                   'environment', 'radii') if k in kw}
        self.mol0, self._scf, self.state = mol, scf_factory, state
        shared['factorization'] = _one_factorization(mol, shared, ground,
                                                     excited, factorization)
        self.ground = ground or RPAGroundStateChain(mol, scf_factory, mf=mf,
                                                    **shared)
        self.excited = excited or ExcitedStateChain(
            mol, scf_factory, mf=self.ground.mf0, **shared, **kw)

    def scf_factory(self, mol):
        return self._scf(mol)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf) through the environment, on the ground-state half.

        The composed surface's own entry to it, so a property routine or a
        manifold reaches the mean field the two halves are actually evaluated
        on rather than `scf_factory`'s, which carries no environment.
        """
        return self.ground.mean_field(mol, mf)

    @property
    def physics_ground_state(self):
        """E_0's functional, which is the ground-state half's: a composed
        surface is ONE functional plus an excitation on it."""
        return self.ground.physics_ground_state

    @property
    def physics_excitation(self):
        """E^{N-/+1}: the orbital eps^QP is taken at, and which way the
        electron count moves. `state` is the offset from the HOMO."""
        return ChargedExcitation(self.ground.nocc - 1 + self.state,
                                 self.charge_change)

    @property
    def physics(self):
        """What this surface computes: E_HF + E_c^dRPA -/+ eps^QP_p.

        The environment is the ground-state half's, which owns the mean field
        both halves are evaluated on.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.ground.environment))

    @property
    def charge_change(self):
        """-1 for the ionization E^{N-1}, +1 for the attachment E^{N+1}."""
        return -1 if self.ground.nocc - 1 + self.state < self.ground.nocc else +1

    @property
    def sign(self):
        """The sign eps^QP enters the total energy with."""
        return -1.0 if self.charge_change == -1 else +1.0

    def energy(self, mol=None, mf=None):
        """(E^{N-/+1}, E_0, eps^QP) in Hartree."""
        mol, mf = self.ground.mean_field(mol, mf)
        e_0 = self.ground.total_energy(mol, mf)
        qp = float(self.excited.quasiparticle(self.state, mol, mf))
        return e_0 + self.sign * qp, e_0, qp

    def total_energy(self, mol=None, mf=None):
        return self.energy(mol, mf)[0]

    def total_gradient(self, mol=None, mf=None):
        mol, mf = self.ground.mean_field(mol, mf)
        g_0, e_0, d0 = self.ground.total_gradient(mol, mf)
        g_qp, d1 = self.excited.quasiparticle_gradient(self.state, mol, mf)
        qp = float(self.excited.quasiparticle(self.state, mol, mf))
        sign = self.sign
        return (np.asarray(g_0) + sign * np.asarray(g_qp), e_0 + sign * qp,
                dict(d1, e_0=e_0, e_c=d0.get('e_c'), qp=qp,
                     e0_terms=d0.get('e0_terms'),
                     charge_change=self.charge_change))

    def refreeze(self, mol):
        """Both halves rebuilt at `mol` on ONE NEW factorization.

        A factorization is frozen AT a geometry, so refreezing must build a new
        one -- carrying the old object forward would keep the old geometry's
        radii, points and frames, which is the discontinuity refreeze exists to
        remove. But both halves must get the SAME new one, or the surface
        silently reverts to two factorizations after the first refreeze and the
        sharing quietly stops holding for the rest of the optimization.

        The five settings are read off the existing factorization, which is
        exactly what it owns -- no chain's own keyword list is duplicated here,
        so adding one upstream cannot make this drift.
        """
        fac = self.ground.factorization.rebuilt_at(mol)
        return type(self)(mol, self._scf, state=self.state, factorization=fac,
                          ground=self.ground.refreeze(mol, factorization=fac),
                          excited=self.excited.refreeze(mol, factorization=fac))

    def label(self):
        kind = 'E^(N-1)' if self.charge_change == -1 else 'E^(N+1)'
        return f'{kind} G0W0 state {self.state} on E_HF + E_c^dRPA'


class RPABSESurface:
    """A `PotentialEnergySurface` whose ground state is dRPA, not the mean field.

    AN ABSOLUTE TOTAL ENERGY IN A CONTINUUM IS NOT TRUSTWORTHY. The exact block
    fold of the ACFDT log-determinant keeps the BARE interaction in the linear
    counter-term while this code dresses it, and the dressing throws away the
    leading solute-solvent dispersion term. Excitation energies, quasiparticle
    levels and any difference taken at a FIXED geometry are free of it, as is
    every gas-phase total energy.
    """

    def __init__(self, mol, scf_factory, state=0, spin='singlet', mf=None,
                 basis=None, auxbasis=None, counts=None, n_start=1,
                 frames='frozen', environment=None, ground=None, excited=None,
                 factorization=None, radii=None, outside='mean-field',
                 **excited_kw):
        # `radii` is named rather than left to **excited_kw: it decides the
        # factorization BOTH halves read, so it has to reach the shared one.
        shared = dict(basis=basis, auxbasis=auxbasis, counts=counts,
                      n_start=n_start, frames=frames, environment=environment,
                      radii=radii)
        self.mol0, self._scf = mol, scf_factory
        shared['factorization'] = _one_factorization(mol, shared, ground,
                                                     excited, factorization)
        # ONE mean field for both chains: two SCFs at the same geometry would
        # be two solutions, and every energy below is a difference of them.
        self.ground = ground or RPAGroundStateChain(mol, scf_factory, mf=mf,
                                                    **shared)
        # `outside` is named rather than left to **excited_kw so that what
        # the orbitals outside the quasiparticle set carry is part of this
        # surface's own signature: it is a different surface, not a tuning.
        self.excited = excited or ExcitedStateChain(
            mol, scf_factory, state=state, spin=spin, mf=self.ground.mf0,
            outside=outside, **shared, **excited_kw)
        self.state, self.spin = state, spin

    @property
    def outside(self):
        """What the orbitals outside the quasiparticle set carry, which is the
        excited half's own choice: the BSE diagonal lives there."""
        return self.excited.outside

    # ---------------------------------------------------- the surface protocol
    def scf_factory(self, mol):
        return self._scf(mol)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf) through the environment, on the ground-state half.

        The composed surface's own entry to it, so a property routine or a
        manifold reaches the mean field the two halves are actually evaluated
        on rather than `scf_factory`'s, which carries no environment.
        """
        return self.ground.mean_field(mol, mf)

    @property
    def physics_ground_state(self):
        """E_0's functional, which is the ground-state half's: a composed
        surface is ONE functional plus an excitation on it."""
        return self.ground.physics_ground_state

    @property
    def physics_excitation(self):
        """The state on E_0, which is the excited half's: the spin, the kernel
        and the root are its settings and nothing here re-spells them."""
        return self.excited.physics_excitation

    @property
    def physics(self):
        """What this surface computes: E_HF + E_c^dRPA + Omega_nu.

        The two halves declare the two pieces and this composes them; the
        environment is the ground-state half's, which owns the mean field both
        are evaluated on.
        """
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.ground.environment))

    def energy(self, mol=None, mf=None):
        """(E_nu, E_0, E_HF, E_c, Omega) in Hartree."""
        mol, mf = self.ground.mean_field(mol, mf)
        e_0, e_hf, e_c = self.ground.energy(mol, mf)
        omega = self.excited.excitation(mol, mf)
        return e_0 + omega, e_0, e_hf, e_c, omega

    def total_energy(self, mol=None, mf=None):
        return self.energy(mol, mf)[0]

    def excitation(self, mol=None, mf=None):
        return self.excited.excitation(mol, mf)

    def total_gradient(self, mol=None, mf=None):
        """(dE_nu/dR, E_nu, diagnostics), fully analytic.

        `RPAGroundStateChain.total_gradient` already carries the mean field's
        own force plus dE_c/dR, so what is added here is dOmega/dR ALONE --
        `excitation_gradient`, not `ExcitedStateChain.total_gradient`, which
        would add the mean-field force a second time.
        """
        mol, mf = self.ground.mean_field(mol, mf)
        g_0, e_0, d0 = self.ground.total_gradient(mol, mf)
        g_om, d1 = self.excited.excitation_gradient(mol, mf)
        omega = float(d1['omega'])
        return (np.asarray(g_0) + np.asarray(g_om), e_0 + omega,
                dict(d1, e_0=e_0, e_c=d0.get('e_c'), omega=omega,
                     e0_terms=d0.get('e0_terms'),
                     grad_ground_max=float(np.abs(g_0).max()),
                     grad_omega_max=float(np.abs(g_om).max())))

    def refreeze(self, mol):
        """Both halves rebuilt at `mol` on ONE NEW factorization.

        A factorization is frozen AT a geometry, so refreezing must build a new
        one -- carrying the old object forward would keep the old geometry's
        radii, points and frames, which is the discontinuity refreeze exists to
        remove. But both halves must get the SAME new one, or the surface
        silently reverts to two factorizations after the first refreeze and the
        sharing quietly stops holding for the rest of the optimization.

        The five settings are read off the existing factorization, which is
        exactly what it owns -- no chain's own keyword list is duplicated here,
        so adding one upstream cannot make this drift.
        """
        fac = self.ground.factorization.rebuilt_at(mol)
        # The outside treatment travels with the refrozen excited half, which
        # recalibrates its scissor at `mol` like every other frozen convention.
        return type(self)(mol, self._scf, state=self.state, spin=self.spin,
                          factorization=fac, outside=self.outside,
                          ground=self.ground.refreeze(mol, factorization=fac),
                          excited=self.excited.refreeze(mol, factorization=fac))

    def label(self):
        return f'{self.excited.label()} on E_HF + E_c^dRPA'
