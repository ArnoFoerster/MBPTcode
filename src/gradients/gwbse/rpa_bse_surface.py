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

Both halves hold one `FrozenFactorization`, so a geometry costs one fit; on
sliced factors (`sliced=True`) they share one set of grid rows per geometry,
cut from the replicated fit or built by the row fit (`fit='rows'`).
"""
import numpy as np

from src.Base.numerics.numerics import (Numerics, numerics_or_default,
                                        require_same_numerics)
from src.Base.declaration import ChargedExcitation, SurfacePhysics
from src.environments.environment import environment_label
from src.gradients.gwbse.excited_state import (ExcitedStateChain, HandedOverPoint,
                                               replayed_first_point)
from src.gradients.rpa.factor_chain import FrozenFactorization
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.state_identity import orbital_fingerprint, track_orbital
from src.gradients.surface import PotentialEnergySurface
from src.gradients.surface_identity import composed_identity


def _halves(mol, scf_factory, mf, environment, numerics, ground,
            excited, factorization):
    """(numerics, factorization, ground half) of a composed surface: ONE
    `Numerics`, ONE FrozenFactorization both halves read and ONE mean field.

    `separable_factors` is deterministic given its settings, so two chains
    handed matching settings produce bitwise identical factors and merely pay
    for the fit twice. One object makes the sharing structural. An explicitly
    given factorization wins; otherwise an already-built half's is adopted,
    so a surface reassembled from existing chains does not manufacture a
    second factorization behind them. Two SCFs at one geometry would be two
    solutions, and every energy of the surface is a difference of them.
    """
    numerics = numerics_or_default(numerics)
    if factorization is None:
        held = [h.factorization for h in (ground, excited)
                if h is not None and getattr(h, 'factorization', None)
                is not None]
        factorization = held[0] if held else FrozenFactorization(mol, numerics)
    # the ground half reads the same object: its factorization settings, its
    # tile cap and fold, and the quadrature derived from the excited half's
    # where it ran on its own defaults before
    ground = ground or RPAGroundStateChain(
        mol, scf_factory, numerics=numerics, mf=mf, environment=environment,
        factorization=factorization)
    return numerics, factorization, ground


class _FollowsOrbital:
    """Which orbital a charged surface takes its quasiparticle at.

    The declared one, `state` from the HOMO, at every geometry -- or, with
    `track`, the orbital that continues it: fingerprinted on the reference
    geometry's Pipek-Mezey localized orbitals and followed through the
    cross-geometry overlap (`state_identity.track_orbital`). An index
    is the wrong label near a crossing: the canonical orbitals reorder and a
    surface following the index steps onto the other state. Each followed
    orbital is logged in `follow_log`, a match two candidates nearly tie for
    with its 'ambiguous' entry (`track_orbital(out=)`); the declaration stays
    the reference's.
    """

    def offset_at(self, mol, mf):
        """The quasiparticle's orbital at (mol, mf), as an offset from the HOMO."""
        reference = self.nocc - 1 + self.state
        if not self.track:
            return self.state
        if self._fingerprint is None:
            _, mf0 = self.mean_field()
            self._fingerprint = orbital_fingerprint(mf0, reference)
        if mol is self.mol0:
            return self.state
        # an ambiguous match is warned about by track_orbital and flagged in
        # the entry (the policy: refuse inputs, warn numerics)
        seen = {}
        orbital, score = track_orbital(self._fingerprint, mf, out=seen)
        entry = {'orbital': int(orbital), 'reference': reference,
                 'similarity': float(score[orbital])}
        if 'ambiguous' in seen:
            entry['ambiguous'] = seen['ambiguous']
        self.follow_log.append(entry)
        return int(orbital) - (self.nocc - 1)

    def _level(self, mol, mf, offset):
        """eps^QP, plus Delta eps^eq when the solvent is in equilibrium."""
        qp = float(self.excited.quasiparticle(offset, mol, mf))
        if self.equilibrium:
            qp += self.excited.equilibrium_shift(offset, mol, mf)
        return qp

    def _level_gradient(self, mol, mf, offset):
        """(d level / dR, the quasiparticle's diagnostics)."""
        g_qp, diagnostics = self.excited.quasiparticle_gradient(offset, mol, mf)
        if self.equilibrium:
            g_eq, _ = self.excited.equilibrium_shift_gradient(offset, mol, mf)
            g_qp = np.asarray(g_qp) + np.asarray(g_eq)
        return np.asarray(g_qp), diagnostics


def require_one_numerics(ground, excited):
    """Refuse halves that hold different `Numerics`: E_0 and the excitation
    are differenced and added, so a half on another grid or tolerance makes
    the sum a number of two surfaces. A half that holds none (a stub) is not
    checked."""
    held = [h.__dict__.get('settings') for h in (ground, excited)]
    if all(isinstance(n, Numerics) for n in held):
        require_same_numerics(*held, 'the halves of a composed surface')


class ComposedSurface:
    """The refreeze and the record of a surface composed of a dRPA
    ground-state half and an excited half on one factorization: ONE body for
    every composed surface, which says only what it carries besides its
    halves (`carried`)."""

    #: The half that owns the spectrum (`surface.driven_chain`) and the
    #: ground-state half (`surface.ground_half`).
    DRIVEN_HALF = 'excited'
    GROUND_HALF = 'ground'
    #: The outer root index the composition keeps beside its half's.
    ROOT_ATTR = 'state'

    def to_record(self):
        """Both halves' records and what the surface carries besides them,
        for `from_record`: a walk's checkpoint restores the composed surface
        with the excited half's committed root following."""
        return {'class': f'{type(self).__module__}:{type(self).__qualname__}',
                'carried': self.carried(),
                'ground': self.ground.to_record(),
                'excited': self.excited.to_record()}

    @classmethod
    def from_record(cls, record, mol, scf_factory, environment=None, mf=None):
        """The surface `to_record` wrote: the ground half read back on one SCF
        at its reference geometry, the excited half on that mean field and on
        the SAME factorization, as `_halves` builds them."""
        ground = RPAGroundStateChain.from_record(
            record['ground'], mol, scf_factory, environment=environment,
            mf=mf)
        excited = ExcitedStateChain.from_record(
            record['excited'], mol, scf_factory, environment=environment,
            mf=ground.mf0, factorization=ground.factorization)
        return cls(ground.mol0, scf_factory, factorization=ground.factorization,
                   ground=ground, excited=excited, **record['carried'])

    def refreeze(self, mol, mf=None):
        """Both halves rebuilt at `mol` on ONE NEW factorization.

        mf: the mean field at `mol` the ground half stands on, where the
        caller holds one; else the ground half converges it.

        A factorization is frozen AT a geometry, so refreezing must build a new
        one -- carrying the old object forward would keep the old geometry's
        radii, points and frames, which is the discontinuity refreeze exists to
        remove. But both halves must get the SAME new one, or the surface
        silently reverts to two factorizations after the first refreeze and the
        sharing quietly stops holding for the rest of the optimization.

        The settings are read off the existing factorization, which is
        exactly what it owns -- no chain's own keyword list is duplicated here,
        so adding one upstream cannot make this drift.

        ONE new SCF: the ground half converges it and the excited half is
        built on it (`_halves`), and the excited half carries its conventions
        over on the mean field the walk evaluated at `mol`, which the ground
        half holds. Each half refreezes through `FactorChain.refreeze`.
        """
        fac = self.ground.factorization.rebuilt_at(mol)
        walked = self.ground.mean_field(mol)[1]
        ground = self.ground.refreeze(mol, factorization=fac, mf=mf)
        excited = self.excited.refreeze(mol, factorization=fac, mf=ground.mf0,
                                        walked=walked)
        return type(self)(mol, self._scf, factorization=fac, ground=ground,
                          excited=excited, **self.carried_at(mol))

    def identity(self):
        """Both halves' identities under the surface's class
        (`surface_identity.composed_identity`)."""
        return composed_identity(self, ground=self.ground, excited=self.excited)

    def carried_at(self, mol):
        """What the surface refrozen at `mol` carries: `carried`, unless a
        subclass follows something that moves with the geometry."""
        return self.carried()


class RPAQPSurface(ComposedSurface, _FollowsOrbital, PotentialEnergySurface):
    """E^{N-/+1} = E_0^dRPA -/+ eps_p^QP: the same rule, for a QUASIPARTICLE.

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
    leading solute-solvent dispersion term -- about 29.6 kcal/mol for water in
    water. Quasiparticle levels, excitation energies and any difference taken
    at a FIXED geometry are free of it, as is every gas-phase total energy.

    equilibrium=True puts the ion in equilibrium with its solvent: the level
    carries the chain's `equilibrium_shift`, Eq18(eps_s) - Eq18(eps_inf) on
    the same factors, and the force its analytic gradient, so E^(N-/+1) =
    E_0 -/+ (eps^QP + Delta eps^eq) -- the adiabatic ion a Marcus four-point
    scheme compares with the vertical one. It is added after the quasiparticle
    equation, unrenormalized by Z. Refused without a continuum carrying a
    static dielectric constant.

    track=True follows the orbital's character rather than its index
    (`_FollowsOrbital`).
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'energy', 'mean_field', 'physics', 'to_record'})

    def __init__(self, mol, scf_factory, state=0, mf=None, environment=None,
                 ground=None, excited=None, factorization=None, numerics=None,
                 equilibrium=False, track=False):
        self.mol0, self._scf, self.state = mol, scf_factory, state
        numerics, factorization, self.ground = _halves(
            mol, scf_factory, mf, environment, numerics, ground, excited,
            factorization)
        self.excited = excited or ExcitedStateChain(
            mol, scf_factory, mf=self.ground.mf0, environment=environment,
            numerics=numerics, factorization=factorization)
        self.equilibrium = bool(equilibrium)
        if self.equilibrium:
            environment = self.excited.environment
            if not hasattr(environment, 'static_partner'):
                raise ValueError(
                    f'equilibrium=True needs a continuum with a static '
                    f'response; this surface carries {environment!r}')
            environment.static_partner()     # raises without eps_static
        self.track, self._fingerprint, self.follow_log = bool(track), None, []
        require_one_numerics(self.ground, self.excited)

    @property
    def nocc(self):
        return self.ground.nocc

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
                                 self.charge_change,
                                 equilibrium=self.equilibrium)

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
        qp = self._level(mol, mf, self.offset_at(mol, mf))
        return e_0 + self.sign * qp, e_0, qp

    def total_energy(self, mol=None, mf=None):
        return self.energy(mol, mf)[0]

    def total_gradient(self, mol=None, mf=None):
        mol, mf = self.ground.mean_field(mol, mf)
        g_0, e_0, d0 = self.ground.total_gradient(mol, mf)
        # the solve the gradient took carries the energy: one Newton per force
        offset = self.offset_at(mol, mf)
        g_qp, d1 = self._level_gradient(mol, mf, offset)
        qp = float(d1['qp_energy'])
        if self.equilibrium:
            qp += self.excited.equilibrium_shift(offset, mol, mf)
        sign = self.sign
        return (np.asarray(g_0) + sign * np.asarray(g_qp), e_0 + sign * qp,
                dict(d1, e_0=e_0, e_c=d0.get('e_c'), qp=qp,
                     e0_terms=d0.get('e0_terms'),
                     charge_change=self.charge_change))

    def carried(self):
        """What the refrozen surface is built with besides its halves."""
        return {'state': self.state, 'equilibrium': self.equilibrium,
                'track': self.track}

    def carried_at(self, mol):
        """`carried`, with the followed orbital at `mol` as the state."""
        return dict(self.carried(),
                    state=self.offset_at(*self.mean_field(mol)))

    def label(self):
        kind = 'E^(N-1)' if self.charge_change == -1 else 'E^(N+1)'
        solvent = ', solvent in equilibrium' if self.equilibrium else ''
        return f'{kind} G0W0 state {self.state} on E_HF + E_c^dRPA{solvent}'


class MeanFieldQPSurface(_FollowsOrbital, PotentialEnergySurface):
    """E^{N-/+1} = E_KS -/+ eps_p^QP: the charged state on the mean field's energy.

    The ground state is the mean field's own, as on `ExcitedStateChain`, so in
    a continuum E_KS is the SCF's in PCM at eps_static -- the neutral in
    equilibrium with its solvent -- and no dRPA correlation energy enters.
    That removes the caveat `RPAQPSurface` carries on absolute energies in a
    continuum, where the plasmon fold keeps the bare interaction in its linear
    counter-term; this surface's total energy is trustworthy there, and its
    difference from the neutral's E_KS is the quasiparticle level itself.

    eps^QP carries the vertical (optical) solvation through Eq. (18);
    `equilibrium=True` adds the ion's relaxed solvent as on `RPAQPSurface`,
    E_KS -/+ (eps^QP + Delta eps^eq), with its analytic gradient. `state` is
    the offset from the HOMO, and `track` follows the orbital's character, as
    there.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'energy', 'mean_field', 'physics', 'to_record'})

    #: The half that owns the spectrum, and the outer root index
    #: (`src.gradients.surface`).
    DRIVEN_HALF = 'excited'
    ROOT_ATTR = 'state'

    def __init__(self, mol, scf_factory, state=0, mf=None, excited=None,
                 equilibrium=False, track=False, **kw):
        self.mol0, self._scf, self.state = mol, scf_factory, state
        self.excited = excited or ExcitedStateChain(mol, scf_factory, mf=mf,
                                                    **kw)
        self.equilibrium = bool(equilibrium)
        if self.equilibrium:
            environment = self.excited.environment
            if not hasattr(environment, 'static_partner'):
                raise ValueError(
                    f'equilibrium=True needs a continuum with a static '
                    f'response; this surface carries {environment!r}')
            environment.static_partner()     # raises without eps_static
        self.track, self._fingerprint, self.follow_log = bool(track), None, []

    def scf_factory(self, mol):
        return self._scf(mol)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf) through the environment: the chain's own mean field."""
        return self.excited.mean_field(mol, mf)

    @property
    def nocc(self):
        return self.excited.nocc

    @property
    def physics_ground_state(self):
        """E_0 = E_KS[xc], the mean field's own energy (`ExcitedStateChain`)."""
        return self.excited.physics_ground_state

    @property
    def physics_excitation(self):
        """E^{N-/+1} at the orbital `state` from the HOMO."""
        return ChargedExcitation(self.nocc - 1 + self.state,
                                 self.charge_change,
                                 equilibrium=self.equilibrium)

    @property
    def physics(self):
        """What this surface computes: E_KS -/+ eps^QP_p, in its environment."""
        return SurfacePhysics(self.physics_ground_state,
                              self.physics_excitation,
                              environment_label(self.excited.environment))

    @property
    def charge_change(self):
        """-1 for the ionization E^{N-1}, +1 for the attachment E^{N+1}."""
        return -1 if self.state <= 0 else +1

    @property
    def sign(self):
        """The sign eps^QP enters the total energy with."""
        return -1.0 if self.charge_change == -1 else +1.0

    def energy(self, mol=None, mf=None):
        """(E^{N-/+1}, E_KS, the level it takes) in Hartree."""
        mol, mf = self.mean_field(mol, mf)
        qp = self._level(mol, mf, self.offset_at(mol, mf))
        return mf.e_tot + self.sign * qp, mf.e_tot, qp

    def total_energy(self, mol=None, mf=None):
        return self.energy(mol, mf)[0]

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics): pyscf's force for the mean field, which
        carries its own PCM term, and the quasiparticle's with the sign the
        electron count gives it."""
        mol, mf = self.mean_field(mol, mf)
        g_0 = self.excited.mean_field_gradient(mf)
        offset = self.offset_at(mol, mf)
        g_qp, d1 = self._level_gradient(mol, mf, offset)
        qp = self._level(mol, mf, offset)
        return (np.asarray(g_0) + self.sign * np.asarray(g_qp),
                mf.e_tot + self.sign * qp,
                dict(d1, e_0=mf.e_tot, qp=qp, charge_change=self.charge_change))

    def refreeze(self, mol, mf=None):
        """The same surface with the chain's frozen conventions rebuilt at
        `mol`, on `mf` where given."""
        return type(self)(mol, self._scf,
                          state=self.offset_at(*self.mean_field(mol, mf)),
                          excited=self.excited.refreeze(mol, mf=mf),
                          equilibrium=self.equilibrium, track=self.track)

    def identity(self):
        """The chain's identity under the surface's class
        (`surface_identity.composed_identity`)."""
        return composed_identity(self, excited=self.excited)

    def carried(self):
        """What the surface is built with besides its chain."""
        return {'state': self.state, 'equilibrium': self.equilibrium,
                'track': self.track}

    def to_record(self):
        """The chain's record and what the surface carries besides it, for
        `from_record`."""
        return {'class': f'{type(self).__module__}:{type(self).__qualname__}',
                'carried': self.carried(),
                'excited': self.excited.to_record()}

    @classmethod
    def from_record(cls, record, mol, scf_factory, environment=None, mf=None):
        """The surface `to_record` wrote, on its chain read back."""
        excited = ExcitedStateChain.from_record(
            record['excited'], mol, scf_factory, environment=environment,
            mf=mf)
        return cls(excited.mol0, scf_factory, excited=excited,
                   **record['carried'])

    def label(self):
        kind = 'E^(N-1)' if self.charge_change == -1 else 'E^(N+1)'
        solvent = ', solvent in equilibrium' if self.equilibrium else ''
        return f'{kind} G0W0 state {self.state} on E_KS{solvent}'




class RPABSESurface(ComposedSurface, PotentialEnergySurface):
    """A `PotentialEnergySurface` whose ground state is dRPA, not the mean field.

    AN ABSOLUTE TOTAL ENERGY IN A CONTINUUM IS NOT TRUSTWORTHY. The exact block
    fold of the ACFDT log-determinant keeps the BARE interaction in the linear
    counter-term while this code dresses it, and the dressing throws away the
    leading solute-solvent dispersion term -- about 29.6 kcal/mol for water in
    water. Excitation energies, quasiparticle levels and any difference taken
    at a FIXED geometry are free of it, as is every gas-phase total energy.

    In an EQUILIBRIUM continuum (`environment.equilibrium_environment`) only
    Omega relaxes in the slow field: E_0 and its force stay the
    non-equilibrium ones (`RPAGroundStateChain.environment_at`), because the
    ground state already relaxes in eps_static and its correlation energy is
    screened by the solvent's electrons alone. The surface declares the
    equilibrium environment it was given.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'composed', 'energy', 'excitation', 'mean_field',
                             'physics', 'spin', 'to_record'})

    first_point = HandedOverPoint()

    def __init__(self, mol, scf_factory, state=0, spin='singlet',
                 bse_tda=False, mf=None, environment=None, ground=None,
                 excited=None, factorization=None, numerics=None):
        self.mol0, self._scf = mol, scf_factory
        numerics, factorization, self.ground = _halves(
            mol, scf_factory, mf, environment, numerics, ground, excited,
            factorization)
        self.excited = excited or ExcitedStateChain(
            mol, scf_factory, state=state, spin=spin, bse_tda=bse_tda,
            mf=self.ground.mf0, environment=environment, numerics=numerics,
            factorization=factorization)
        require_one_numerics(self.ground, self.excited)
        self.state, self.spin = state, spin

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
        replay = replayed_first_point(self, mol)
        if replay is not None:
            return replay
        mol, mf = self.ground.mean_field(mol, mf)
        ground = self.ground.total_gradient(mol, mf)
        return self.composed(ground, self.excited.excitation_gradient(mol, mf))

    @staticmethod
    def composed(ground, excitation):
        """(dE_nu/dR, E_nu, diagnostics) from the ground half's
        (dE_0/dR, E_0, diagnostics) and the excited half's (dOmega/dR,
        diagnostics): one assembly for this surface and for a manifold that
        reuses one ground-state force under several states."""
        g_0, e_0, d0 = ground
        g_om, d1 = excitation
        omega = float(d1['omega'])
        return (np.asarray(g_0) + np.asarray(g_om), e_0 + omega,
                dict(d1, e_0=e_0, e_c=d0.get('e_c'), omega=omega,
                     e0_terms=d0.get('e0_terms'),
                     grad_ground_max=float(np.abs(g_0).max()),
                     grad_omega_max=float(np.abs(g_om).max())))

    def carried(self):
        """What the refrozen surface is built with besides its halves."""
        return {'state': self.state, 'spin': self.spin}

    def label(self):
        return f'{self.excited.label()} on E_HF + E_c^dRPA'
