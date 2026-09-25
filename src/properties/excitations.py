"""The vertical, emission and adiabatic energies of one state, with BOTH
surfaces of every difference built from one declaration.

An adiabatic quantity is a difference of two total energies at two different
geometries, and nothing about the number says which two surfaces produced it.
Two relaxations of one molecule were subtracted with E_KS + Omega on one side
and E_HF + (E_x^HF - E_xc)[rho] + E_c^dRPA + Omega on the other, 0.9 eV apart,
and no object could refuse it. Here a `SurfaceSpec` IS the declaration and it
builds both surfaces of the difference, so the same-functional guarantee is
structural rather than a convention the caller is asked to keep:

    spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',
                       factorization='four-index')
    record = calc_adiabatic_excitation(spec, Excitation('singlet'), mol, rhf)

Every record carries the physics it is of, the realization that computed it,
the resolved numerics, E_0's additive terms AT EACH GEOMETRY, the residual
|dE/dR| at the minimum and the driving force at the reference geometry under
names that cannot be confused, the energy the refreeze moved, and the commit
it was produced on. `grad_max` is a field name nowhere here: it means the
driving force at R0 in one half of the repository and the residual at R* in
the other, and an adiabatic record must not inherit that.

A state is declared by spin and root index, never by irreducible
representation: `Excitation(irrep=...)` has no realization that follows it and
`potential_energy_surface` refuses it, which these routines inherit unchanged.
Take the irrep from the spectrum with `src.properties.characters` and declare
the root it came out as.
"""
import datetime
import functools
import inspect
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.Base.constants import (GIT_PROVENANCE_TIMEOUT, HARTREE_TO_EV,
                                HARTREE_TO_MEV)
from src.Base.declaration import GroundState, PhysicsMismatch, SurfacePhysics
from src.SingleReference.LinearResponse.rpa_energy import (ground_state_energy,
                                                           reference_energy)
from src.properties.surface import evaluate, surface_mean_field
from src.properties.surfaces import (compare_surfaces, environment_label,
                                     find_row, potential_energy_surface)
from src.properties.vibronic import adiabatic_gap, relax_state

#: Everything `potential_energy_surface` takes beyond the molecule, the SCF
#: factory, the state and the numerics: the physics axes first, then the
#: realization ones.
SPEC_AXES = ('ground_state', 'environment', 'chi0', 'residues', 'solver',
             'factorization', 'qp_states')

#: The axes a surface with NO state on it reads. The residue backend, the
#: eigensolver and the quasiparticle set belong to a state: E_0 has no
#: self-energy, no Casida eigenproblem and no set of explicitly solved
#: orbitals, so they are dropped for the ground surface rather than refused.
GROUND_AXES = ('ground_state', 'environment', 'chi0', 'factorization')

#: What a record whose relaxation never ran says instead of a shift. A null
#: that does not say why it is null reads exactly like a measured zero.
NOT_RELAXED = 'not measured: no relaxation ran'


@dataclass(frozen=True)
class SurfaceSpec:
    """One declaration, both surfaces of an adiabatic quantity.

    The fields are `potential_energy_surface`'s own, minus the molecule, the
    SCF factory and the state: the physics (which functional E_0 is, what it
    stands in) and the realization (how chi0 is built, how the self-energy's
    real-axis residues are taken, which eigensolver runs, how (pq|rs) is
    represented, which orbitals carry an explicitly solved quasiparticle
    energy). A quantity that differences two geometries takes ONE spec, so the
    functional under the excited state and the functional under the ground
    state cannot be two functionals.

    An axis left None is the entry point's OWN default, read off its signature
    at construction and stored: a spec that spelled the defaults again would be
    a second set of them, which is what put three `solver` defaults in three
    constructors.

    numerics: the grids, caps and tolerances the realization reads, as a sorted
        tuple of (key, value) pairs so that the spec hashes and two specs
        written in different orders compare equal. A mapping is accepted and
        normalized. One value that is itself a mapping -- an explicit ISDF
        `counts` -- travels verbatim and makes that spec unhashable rather than
        being rewritten into something the realizing class does not read.

    A spec carries no communicator: how many ranks added the same terms up is
    not part of what a surface computes. Inside `with distributed(comm):`
    every rank builds both surfaces of the difference from the one spec, and
    the kernels they reach divide their sweeps over the ranks.
    """
    ground_state: GroundState
    environment: object = None
    chi0: Optional[str] = None
    residues: Optional[str] = None
    solver: Optional[str] = None
    factorization: Optional[str] = None
    qp_states: object = None
    numerics: tuple = ()

    def __post_init__(self) -> None:
        for name, default in entry_point_defaults().items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        object.__setattr__(self, 'numerics', normalized_numerics(self.numerics))

    def as_kwargs(self) -> dict:
        """The keywords `potential_energy_surface` takes, numerics expanded."""
        kwargs = {name: getattr(self, name) for name in SPEC_AXES}
        kwargs.update(self.numerics)
        return kwargs

    def physics(self, excitation) -> SurfacePhysics:
        """What a surface of this spec carrying `excitation` computes.

        Built without touching a molecule or an SCF, so two specs can be
        refused against each other before either is realized.
        """
        return SurfacePhysics(self.ground_state, excitation,
                              environment_label(self.environment))


@functools.lru_cache(maxsize=1)
def entry_point_defaults():
    """The realization defaults `potential_energy_surface` itself resolves.

    Read off its signature rather than repeated here, so a spec and the entry
    point cannot drift into two different defaults; `ground_state` has none and
    is required.
    """
    params = inspect.signature(potential_energy_surface).parameters
    return {name: params[name].default for name in SPEC_AXES
            if params[name].default is not inspect.Parameter.empty}


def normalized_numerics(numerics):
    """Numeric keywords as a sorted tuple of (key, value) pairs."""
    items = numerics.items() if hasattr(numerics, 'items') else numerics
    return tuple(sorted(((str(key), value) for key, value in items),
                        key=lambda pair: pair[0]))


def surface_of(spec, excitation, mol, scf_factory):
    """The surface `spec` declares with `excitation` on it; None gives E_0's own.

    Both come from one spec, which is what makes their difference a difference
    of one functional with itself at two geometries. The axes only a state has
    -- the residue backend, the eigensolver, the quasiparticle set -- and the
    numerics the ground row does not read are dropped on the way to the ground
    surface: E_0 has nothing for them to resolve, and passing them would be
    refused by the entry point rather than honoured.
    """
    kwargs = spec.as_kwargs()
    if excitation is not None:
        return potential_energy_surface(mol, scf_factory,
                                        excitation=excitation, **kwargs)
    row = find_row(spec.ground_state, None, spec.chi0, spec.factorization)
    ground = {name: kwargs[name] for name in GROUND_AXES}
    ground.update((key, value) for key, value in spec.numerics
                  if key in row.numerics)
    return potential_energy_surface(mol, scf_factory, **ground)


def git_commit():
    """The short commit these numbers were produced on, '+dirty' if the tree moved."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short',
                               'HEAD'], capture_output=True, text=True,
                              timeout=GIT_PROVENANCE_TIMEOUT)
        moved = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                               capture_output=True, text=True,
                               timeout=GIT_PROVENANCE_TIMEOUT)
    except Exception:
        return 'unknown'
    return (head.stdout.strip() or 'unknown') + ('+dirty' if moved.stdout.strip()
                                                 else '')


def provenance(seconds):
    """Which tree produced a record, and how long it took.

    The commit is what makes a number re-runnable. The clock is under `timing`
    and nowhere else, so that a record read for its physics carries no field
    that changes between two identical runs; the host is not recorded at all,
    because a record is of a method and not of a machine.
    """
    return {'commit': git_commit(),
            'timing': {'wall_seconds': float(seconds),
                       'utc': datetime.datetime.now(datetime.timezone.utc)
                       .strftime('%Y-%m-%dT%H:%M:%SZ')}}


def physics_record(surface):
    """The declared physics of a surface, as its printable label and the record."""
    return {'label': surface.physics.label(), 'record': surface.physics}


def quasiparticle_diagnostics(info):
    """(pole strength, residue route taken) as the surface's gradient reports them.

    Z and the backend that produced the residues are properties of the
    quasiparticle solve, not of the declaration: `residues='auto'` picks per
    orbital, and a root with Z near zero is a satellite the state was built on.
    A route that exposes neither -- the dense quasi-boson surface has no
    residue backend, and a neutral excitation folds its whole set through one
    solve -- reports None, which is not a Z of zero.
    """
    z = info.get('qp_z')
    return (None if z is None else float(z)), info.get('qp_route')


def ground_state_at(surface, mol, total=None):
    """E_0(mol) and its additive terms on the ground surface's frozen conventions.

    The total is the surface's own (`vibronic.energy_at`'s number, on the mean
    field the surface itself evaluates on) and the terms are that total
    decomposed through the ONE assembly, `ground_state_energy`: E_c^dRPA is
    taken as E_0 - E_HF[rho] so that the three terms sum to the number the
    surface reported rather than to a second evaluation of it.

    THE CONVENTIONS ARE THE REFERENCE GEOMETRY'S, wherever `mol` is. A ground
    surface refrozen at each geometry would make E_0(R*) and E_0(R0) two
    surfaces, and their difference the drift between them.
    """
    mf = surface_mean_field(surface, mol)
    declared = surface.physics.ground_state
    if declared.kind == 'dft':
        return ground_state_energy(declared, mf, mol)
    total = surface.total_energy(mol, mf) if total is None else float(total)
    return ground_state_energy(declared, mf, mol,
                               e_corr=total - reference_energy(mf, mol))


def relaxation_fields(info, prefix=''):
    """What an optimizer's record contributes, renamed away from `grad_max`.

    `info['grad_max']` and `info['opt_grad_max']` are ONE number under two
    names -- the residual |dE/dR| at the converged geometry -- and the first is
    what the driving force at the INPUT geometry is called everywhere else.
    Only the unambiguous name travels.

    The refreeze shift is reported twice because it is two measurements of one
    approximation: how far the geometry moved when the frozen conventions were
    rebuilt (Bohr) and how much energy that motion was worth (meV). Either is
    None only when the outer loop did not run, and `refreeze` then says why.
    """
    residual = info.get('opt_grad_max')
    shift = info.get('refreeze_shift')
    moved = info.get('refreeze_denergy')
    return {prefix + 'opt_grad_max': None if residual is None else float(residual),
            prefix + 'refreeze_shift_bohr': None if shift is None else float(shift),
            prefix + 'refreeze_shift_meV': (None if moved is None
                                            else float(moved) * HARTREE_TO_MEV),
            prefix + 'refreeze': info.get('refreeze'),
            prefix + 'optimizer': info.get('optimizer'),
            prefix + 'converged': bool(info.get('converged')),
            prefix + 'cycles': int(info.get('cycles', 0)),
            prefix + 'status': info.get('status')}


def driving_force(info):
    """max |dE/dR| at the FIRST geometry of a relaxation, from its own history."""
    history = info.get('history') or []
    return float(history[0]['grad_max']) if history else None


def excitation_energy(info, e_n, e_0):
    """Omega_n at one geometry: the surface's OWN number wherever it reports one.

    A neutral excitation's Omega is a root of the Casida problem and every
    route that solves one hands it back with the gradient; rebuilding it as a
    difference of two -76 Hartree totals would move it by their rounding. A
    charged surface reports no Omega -- its excitation IS -/+ eps^QP -- and
    there the difference of the two totals is the quantity exactly.
    """
    omega = info.get('omega')
    return float(e_n - e_0) if omega is None else float(omega)


def vertical_block(excited, ground, mol):
    """Omega_n(R0), the two total energies it is the difference of, and the
    force that says how far R0 is from the excited state's own minimum.

    Omega is the excited surface's OWN number, taken off the gradient that
    measures the driving force rather than recomputed: this layer records and
    differences, and may not move a number by so much as a rounding. The
    gradient is `surface.evaluate`'s, rank 0's on every rank.

    On sliced factors the record carries `factor_gathers`, the whole-array
    gathers the factors at R0 made up to that force, by name: one per sweep
    or solve, whatever the tau count or the Davidson's iterations; on the row
    fit also `fit_held`, the most of each fit array rank 0 held at once, in
    bytes.
    """
    force, e_n, info = evaluate(excited, mol)
    e_0 = ground_state_at(ground, mol)
    z, route = quasiparticle_diagnostics(info)
    omega = excitation_energy(info, e_n, e_0.total)
    layout = {name: dict(info[name]) for name in ('factor_gathers', 'fit_held')
              if name in info}
    return {**layout, 'omega_eV': omega * HARTREE_TO_EV,
            'e0_hartree': float(e_0.total), 'e0_terms': dict(e_0.terms),
            'en_hartree': float(e_n),
            'driving_force_max': float(np.abs(force).max()),
            'opt_grad_max': None, 'refreeze_shift_bohr': None,
            'refreeze_shift_meV': None, 'refreeze': NOT_RELAXED,
            'optimizer': None, 'converged': None, 'cycles': 0,
            'status': NOT_RELAXED,
            'qp_z': z, 'residue_route_taken': route,
            'physics': physics_record(excited),
            'realization': excited.realization,
            'numerics': dict(excited.numerics),
            'mol_reference': mol}


def emission_block(excited, ground, mol, refreeze, engine, optimizer):
    """(the vertical record extended to the relaxed state, the relaxation record).

    E_0 AT THE EXCITED STATE'S MINIMUM, not at R0: the emission energy is the
    vertical gap of the RELAXED geometry, and the ground state is not at its
    own minimum there. Both energies come off the same pair of frozen
    conventions, the reference geometry's, so their difference is one surface
    pair evaluated twice and not two surfaces compared.
    """
    record = vertical_block(excited, ground, mol)
    relaxed = relax_state(excited, mol, engine=engine, refreeze=refreeze,
                          **optimizer)
    e_0 = ground_state_at(ground, relaxed['mol'])
    e_n = float(relaxed['e_total'])
    record.update(
        emission_eV=(e_n - e_0.total) * HARTREE_TO_EV,
        relaxation_depth_eV=(record['en_hartree'] - e_n) * HARTREE_TO_EV,
        en_hartree_at_excited_minimum=e_n,
        e0_hartree_at_excited_minimum=float(e_0.total),
        e0_terms_at_excited_minimum=dict(e_0.terms),
        mol_excited_minimum=relaxed['mol'],
        **relaxation_fields(relaxed['info']))
    return record, relaxed


def refuse_incomparable(physics_a, physics_b):
    """Refuse two declarations whose difference would be of two functionals."""
    if not physics_a.comparable_with(physics_b):
        raise PhysicsMismatch(physics_a, physics_b)


def adiabatic_energy(surface_excited, surface_ground, mol, refreeze=1,
                     engine='auto', **optimizer):
    """E_n(R*_n) - E_0(R*_0) for two surfaces already built, with the refusal.

    The two-surface form of `calc_adiabatic_excitation`, for a pair somebody
    else constructed. `compare_surfaces` is the refusal: a differing
    ground-state functional or environment raises `PhysicsMismatch` BEFORE
    either relaxation runs, and a surface with no declared physics is refused
    too, since what its E_0 is cannot be read off it.

    `ground_realization_differs` is what it reports rather than refuses: the
    two halves of an adiabatic energy are realized differently by construction
    -- E_0 has no residue backend, no eigensolver and no quasiparticle set --
    and the list names those fields.
    """
    report = compare_surfaces(surface_excited, surface_ground)
    started = time.perf_counter()
    record, _ = emission_block(surface_excited, surface_ground, mol, refreeze,
                               engine, optimizer)
    relaxed = relax_state(surface_ground, mol, engine=engine,
                          refreeze=refreeze, **optimizer)
    e_0 = ground_state_at(surface_ground, relaxed['mol'],
                          total=relaxed['e_total'])
    record.update(
        adiabatic_eV=((record['en_hartree_at_excited_minimum'] - e_0.total)
                      * HARTREE_TO_EV),
        e0_hartree_at_ground_minimum=float(e_0.total),
        e0_terms_at_ground_minimum=dict(e_0.terms),
        mol_ground_minimum=relaxed['mol'],
        ground_realization=surface_ground.realization,
        ground_driving_force_max=driving_force(relaxed['info']),
        ground_realization_differs=report['realization_differs'],
        provenance=provenance(time.perf_counter() - started),
        **relaxation_fields(relaxed['info'], prefix='ground_'))
    return record


def calc_vertical_excitation(spec, excitation, mol, scf_factory):
    """Omega_n(R0) in eV: the excitation energy at the geometry as given.

    No geometry moves, so the residual at a minimum and the refreeze drift do
    not exist and are None with the marker saying so; the driving force at R0
    is the gradient that does exist, and is what says whether this vertical
    number stands at a stationary point of anything.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = surface_of(spec, None, mol, scf_factory)
    record = vertical_block(excited, ground, mol)
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_emission_energy(spec, excitation, mol, scf_factory, refreeze=1,
                         engine='auto', **optimizer):
    """Omega_n(R*_n) in eV: the gap at the relaxed geometry of the state.

    The state is relaxed from `mol` and the ground state is NOT: emission
    leaves the excited minimum for the ground surface at that same geometry,
    which is why E_0 is evaluated there and not at R0. `relaxation_depth_eV`
    is what the state shed getting there, and `refreeze_shift_meV` is what the
    frozen conventions were worth in the same units -- the honest error bar on
    the number above it.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = surface_of(spec, None, mol, scf_factory)
    record, _ = emission_block(excited, ground, mol, refreeze, engine, optimizer)
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_adiabatic_excitation(spec, excitation, mol, scf_factory, refreeze=1,
                              engine='auto', **optimizer):
    """E_n(R*_n) - E_0(R*_0) in eV: both states at their own minima.

    The quantity a 0-0 energy is built on, and the one the same-functional
    guarantee exists for: the two minima are on two surfaces, so E_0 does not
    cancel and every term of it has to be the same term on both sides. Carries
    the vertical and emission blocks with it, and both refreeze shifts, since
    an adiabatic energy is only as converged as the looser of its two
    relaxations.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = surface_of(spec, None, mol, scf_factory)
    record = adiabatic_energy(excited, ground, mol, refreeze=refreeze,
                              engine=engine, **optimizer)
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_adiabatic_gap(spec_a, excitation_a, spec_b, excitation_b, mol,
                       scf_factory, refreeze=1, engine='auto', **optimizer):
    """E_a(R*_a) - E_b(R*_b) in eV: the adiabatic gap of two states, each at
    its own minimum.

    Delta-E_ST when a is the singlet and b the triplet. REFUSED unless the two
    specs declare the same ground-state functional and the same environment:
    the two states relax by different amounts, so E_0 does not cancel out of
    the difference, and a mean-field E_0 differenced against E_HF + E_c^dRPA is
    6.3 eV of correlation energy on water that only one side carries. The
    refusal is taken on the DECLARATIONS, before an integral is computed.

    The realizations may differ and are reported side by side: a dense oracle
    and a cubic production route are meant to be differenced, and
    `realization_differs` is what says which of the two to trust where they do.
    """
    started = time.perf_counter()
    refuse_incomparable(spec_a.physics(excitation_a), spec_b.physics(excitation_b))
    surface_a = surface_of(spec_a, excitation_a, mol, scf_factory)
    surface_b = surface_of(spec_b, excitation_b, mol, scf_factory)
    report = compare_surfaces(surface_a, surface_b)
    record_a, relaxed_a = emission_block(
        surface_a, surface_of(spec_a, None, mol, scf_factory), mol, refreeze,
        engine, optimizer)
    record_b, relaxed_b = emission_block(
        surface_b, surface_of(spec_b, None, mol, scf_factory), mol, refreeze,
        engine, optimizer)
    return {'gap_eV': adiabatic_gap(relaxed_a, relaxed_b) * HARTREE_TO_EV,
            'physics': {'label': report['physics'],
                        'record': (surface_a.physics, surface_b.physics)},
            'realization': report['realization'],
            'realization_differs': report['realization_differs'],
            'numerics': (dict(surface_a.numerics), dict(surface_b.numerics)),
            'state_a': record_a, 'state_b': record_b,
            'provenance': provenance(time.perf_counter() - started)}
