"""The vertical, emission and adiabatic energies of one state, with both
surfaces of every difference built from one declaration.

An adiabatic quantity is a difference of two total energies at two geometries,
and nothing about the number says which two surfaces produced it: E_KS + Omega
against E_HF + (E_x^HF - E_xc)[rho] + E_c^dRPA + Omega can differ by 0.9 eV
for one molecule. A `SurfaceSpec` is the declaration and builds both surfaces
of the difference, so the same-functional guarantee is structural rather than
a convention the caller keeps:

    spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',
                       factorization='four-index')
    record = calc_adiabatic_excitation(spec, Excitation('singlet'), mol, rhf)

Every record carries the physics, the realization that computed it, the
resolved numerics, E_0's additive terms at each geometry, the residual |dE/dR|
at the minimum and the driving force at the reference geometry under distinct
names, the energy the refreeze moved, and the commit it was produced on. No
field here is named `grad_max`: elsewhere that name means the driving force at
R0 in some places and the residual at R* in others.

A state is declared by spin and root index, never by irreducible
representation: `Excitation(irrep=...)` has no realization that follows it and
`potential_energy_surface` refuses it.

Every vertical block also carries what the method reports of its own state
through the `diagnostics` ingredient: for GW-BSE (`gradients.gwbse.diagnostics`)
which orbitals were solved explicitly, which carry the frozen scissor and what
shift, which route each explicit state took, how many poles the sum-over-poles
model used, which interpolation grid the factorization held, and its warning
logs. A surface whose method provides no diagnostics records None there.
"""
import datetime
import functools
import inspect
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.numerics.constants import (HARTREE_TO_EV, HARTREE_TO_MEV,
                                         SURFACE_GRID_ACCURACY)
from src.Base.declaration import (Excitation, GroundState, PhysicsMismatch,
                                  SurfacePhysics)
from src.Base.utils.record_io import git_commit
from src.Base.numerics.numerics import (Numerics, numerics_identity,
                                        numerics_or_default, require_same_identity,
                                        surface_numerics)
from src.Base.geometry.geometry import elements_of
from src.Base.pyscf.scf_convergence import scf_record
from src.Base.factorization.separable_ri import grid_identity, grid_points_per_atom
from src.Base.utils.parallel.mpi_grid import lockstep
from src.Base.utils.timing import stage
from src.SingleReference.LinearResponse.rpa_energy import (ground_state_energy,
                                                           reference_energy)
from src.gradients.gwbse.state_manifold import StateManifold
from src.properties.hessian import surface_hessian
from src.properties.optimize import escape_saddle
from src.gradients.surface import (evaluate, lockstep_geometry,
                                   surface_mean_field)
from src.gradients.surface import driven_chain, ground_half, provides, require
from src.gradients.gwbse.surface_factory import (compare_surfaces, environment_label,
                                                 find_row, numerics_read_by,
                                                 potential_energy_surface,
                                                 reference_mean_field, resolve_grid)
from src.properties.vibronic import adiabatic_gap, relax_state

#: Everything `potential_energy_surface` takes beyond the molecule, the SCF
#: factory, the state, the numerics and the options: the physics axes first,
#: then the realization ones.
SPEC_AXES = ('ground_state', 'environment', 'chi0', 'residues', 'solver',
             'factorization', 'qp_states')

#: The axes a surface with no state on it reads. The residue backend, the
#: eigensolver and the quasiparticle set belong to a state: E_0 has no
#: self-energy, no Casida eigenproblem and no set of explicitly solved
#: orbitals, so they are dropped for the ground surface rather than refused.
GROUND_AXES = ('ground_state', 'environment', 'chi0', 'factorization')

#: What a record whose relaxation never ran says instead of a shift. A null
#: that does not say why it is null reads like a measured zero.
NOT_RELAXED = 'not measured: no relaxation ran'


class NoSharedForward(TypeError):
    """Several states were asked off one evaluation of a surface whose
    forward pass is not shared below the Casida step."""


@dataclass(frozen=True)
class SurfaceSpec:
    """One declaration, both surfaces of an adiabatic quantity.

    The fields are `potential_energy_surface`'s own, minus the molecule, the
    SCF factory and the state: the physics (which functional E_0 is, what it
    stands in) and the realization (how chi0 is built, how the self-energy's
    real-axis residues are taken, which eigensolver runs, how (pq|rs) is
    represented, which orbitals carry an explicitly solved quasiparticle
    energy). A quantity that differences two geometries takes one spec, so the
    functional under the excited state and the functional under the ground
    state cannot be two functionals.

    An axis left None is the entry point's own default, read off its signature
    at construction and stored, so the spec holds no second set of defaults.

    numerics: the one `Numerics` the realization stands on (None: every
        choice at its default), handed whole to `potential_energy_surface`.
        Two specs compare by its choices (`Numerics.values`, the provenance
        aside) and hash by its `identity`.
    options: the row options that are no choice of `Numerics` ('eri_blocks',
        'qp_partition'), as a sorted tuple of (key, value) pairs; a mapping
        is accepted and normalized. A value that does not hash (a frozen
        partition) makes that spec unhashable rather than being rewritten.

    A spec carries no communicator: the rank count is not part of what a
    surface computes. Inside `with distributed(comm):`
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
    numerics: Optional[Numerics] = None
    options: tuple = ()

    def __post_init__(self) -> None:
        for name, default in entry_point_defaults().items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        numerics = numerics_or_default(self.numerics)
        if not isinstance(numerics, Numerics):
            raise TypeError(f'SurfaceSpec.numerics is one Numerics, not a '
                            f'{type(numerics).__name__}: build it with '
                            f'Numerics(...) or numerics_of({{...}})')
        object.__setattr__(self, 'numerics', numerics)
        options = (self.options.items() if hasattr(self.options, 'items')
                   else self.options)
        object.__setattr__(self, 'options', tuple(sorted(
            ((str(key), value) for key, value in options),
            key=lambda pair: pair[0])))

    def _axes(self):
        return tuple(getattr(self, name) for name in SPEC_AXES)

    def __eq__(self, other):
        if not isinstance(other, SurfaceSpec):
            return NotImplemented
        return (self._axes(), self.numerics.values(), self.options) == (
            other._axes(), other.numerics.values(), other.options)

    def __hash__(self):
        return hash((self._axes(), self.numerics.identity(), self.options))

    def as_kwargs(self) -> dict:
        """The keywords `potential_energy_surface` takes: the axes, the one
        `numerics` and the options."""
        kwargs = {name: getattr(self, name) for name in SPEC_AXES}
        kwargs['numerics'] = self.numerics
        kwargs.update(self.options)
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


def surface_of(spec, excitation, mol, scf_factory, mf=None, settings=None):
    """The surface `spec` declares with `excitation` on it; None gives E_0's own.

    `mf` is a reference mean field already converged for this declaration at
    `mol`; None converges one. `settings` is the `Numerics` E_0's chain stands
    on (`ground_surface_of`); an excited surface builds its own.

    Both surfaces come from one spec, so their difference is of one
    functional with itself at two geometries. The axes only a state has (the
    residue backend, the eigensolver, the quasiparticle set), the parts of
    the numerics the ground row does not read (`numerics_read_by`) and the
    options it does not take are dropped for the ground surface, since the
    entry point would refuse them.
    """
    kwargs = spec.as_kwargs()
    if excitation is not None:
        return potential_energy_surface(mol, scf_factory, mf=mf,
                                        excitation=excitation, **kwargs)
    row = find_row(spec.ground_state, None, spec.chi0, spec.factorization)
    ground = {name: kwargs[name] for name in GROUND_AXES}
    ground.update((key, value) for key, value in spec.options
                  if key in row.options)
    return potential_energy_surface(
        mol, scf_factory, mf=mf, settings=settings,
        numerics=numerics_read_by(row, spec.numerics), **ground)


def ground_surface_of(spec, excited, mol, scf_factory):
    """E_0's surface of `spec` at `mol`, on the excited surface's reference.

    One SCF per record: a surface that holds its converged reference (`mf0`,
    the excited chain's) hands it to the ground surface, which declares its
    functional off it and, in the gas phase, evaluates E_0 at `mol` on it.

    A ground row with a chain (dRPA) stands on the excited surface's own
    `Numerics`, so that E_c^dRPA is taken on the quadrature the excited
    surface's ground half uses (`rpa_quadrature`) and the two energies of a
    record differ by the state alone; a ground choice of its own would be
    `Numerics.derive` of that object, and the seam refuses it.
    """
    kwargs = {}
    held = surface_numerics(excited)
    row = find_row(spec.ground_state, None, spec.chi0, spec.factorization)
    if held is not None and row.reads_settings:
        kwargs['settings'] = held
    return surface_of(spec, None, mol, scf_factory,
                      mf=getattr(excited, 'mf0', None), **kwargs)


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


def time_stages(surface, timer):
    """Hand `timer` to the chain that owns `surface`'s spectrum, whose stages
    (`FactorChain.phase`) then accumulate into it; a surface with no stages
    is left as it is."""
    chain = driven_chain(surface)
    if timer is not None and provides(chain, 'phase'):
        chain.timer = timer


def diagnostics_of(surface, mol, mf=None, posteriori=False):
    """The `diagnostics` ingredient of the chain that owns `surface`'s state
    at `mol` (a dict of record fields), {} for a method that provides none."""
    chain = driven_chain(surface)
    if not provides(chain, 'diagnostics'):
        return {}
    return chain.diagnostics(mol, mf=mf, posteriori=posteriori)


def merged_logs(name, *diagnostics):
    """The entries of log `name` across several `diagnostics_of` results,
    each once by the key its method gave it, in the order logged."""
    out, seen = [], set()
    for d in diagnostics:
        for key, entry in (d.get('logs') or {}).get(name, []):
            if key not in seen:
                seen.add(key)
                out.append(dict(entry))
    return out


def physics_record(surface):
    """The declared physics of a surface, as its printable label and the record."""
    return {'label': surface.physics.label(), 'record': surface.physics}


def quasiparticle_diagnostics(info):
    """(pole strength, residue route taken) as the surface's gradient reports them.

    Z and the backend that produced the residues are properties of the
    quasiparticle solve, not of the declaration: `residues='auto'` picks per
    orbital, and a root with Z near zero is a satellite the state was built on.
    A route that exposes neither (the dense quasi-boson surface has no
    residue backend, and a neutral excitation folds its whole set through one
    solve) reports None, which is not a Z of zero.
    """
    z = info.get('qp_z')
    return (None if z is None else float(z)), info.get('qp_route')


def grid_record(spec, mol):
    """The interpolation grid `spec` fits on at `mol`, resolved as the entry point does.

    The level asked for (the entry point's own default when the spec names
    neither a level, counts nor radii), its shell counts, the points per atom
    they come to, the multi-start recipe and `grid_identity`: the level with
    each element's counts and the digest of the radii they place, the stamp
    that tells a grid apart from another under the same name (a resume or a
    comparison checks it with `refuse_other_grid`). A grid nobody validated is
    refused here, before an integral is computed; a row with no interpolation
    grid reports None throughout.
    """
    row = find_row(spec.ground_state, Excitation('singlet'), spec.chi0,
                   spec.factorization)
    if not row.isdf_grid:
        return dict(grid_level=None, grid_counts=None,
                    grid_points_per_atom=None, grid_n_start=None,
                    grid_label=None, grid_identity=None)
    isdf = spec.numerics.isdf
    counts, n_start, label = resolve_grid(mol, spec.numerics)
    level = isdf.level
    if level is None and isdf.counts is None and isdf.radii is None:
        level = SURFACE_GRID_ACCURACY
    basis = isdf.basis or str(mol.basis)
    identity = grid_identity(
        level, basis, elements_of(mol),
        auxbasis=isdf.auxbasis or paired_auxbasis(basis), counts=counts,
        radii=isdf.radii)
    return dict(grid_level=level,
                grid_counts=None if counts is None else dict(counts),
                grid_points_per_atom=(None if counts is None
                                      else grid_points_per_atom(counts)),
                grid_n_start=n_start, grid_label=label,
                grid_identity=identity)


def spectrum_of(surface, mol, mf):
    """(Omega, X, Y, nocc): every root the surface's Casida step solved at `mol`."""
    return require(driven_chain(surface), 'excited_states', 'spectrum_of')(
        mol, mf)


def ground_state_at(surface, mol, total=None):
    """E_0(mol) and its additive terms on the ground surface's frozen conventions.

    The total is the surface's own (`vibronic.energy_at`'s number, on the mean
    field the surface itself evaluates on) and the terms are that total
    decomposed through `ground_state_energy`: E_c^dRPA is taken as
    E_0 - E_HF[rho] so that the three terms sum to the number the surface
    reported rather than to a second evaluation of it.

    The conventions are the reference geometry's wherever `mol` is; a ground
    surface refrozen at each geometry would make E_0(R*) and E_0(R0) two
    surfaces.
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

    A walk's record names the residual |dE/dR| at the converged geometry
    `opt_grad_max`; `grad_max` is the driving force at the INPUT geometry
    everywhere else, so only the first travels.

    The refreeze is reported as the drift of the conventions at the minimum,
    E_new(R*) - E_old(R*) at one geometry (meV, `optimize.refreeze_pass`),
    the geometry move of the walk on the rebuilt surface (Bohr) and that
    walk's whole energy change (meV, drift and descent). Each is None only
    when the refreeze did not run, and `refreeze` then says why;
    `refreeze_converged` is |drift| <= REFREEZE_TOL_MEV.

    `converged`, `cycles`, `status` and `opt_grad_max` are the walk whose
    geometry the record reports -- after a refreeze the refrozen walk's --
    so `converged` False means that geometry is not a minimum, whatever the
    walk before it did. A relaxed number needs both `converged` and, where
    the refreeze ran, `refreeze_converged`.
    """
    residual = info.get('opt_grad_max')
    shift = info.get('refreeze_shift')
    moved = info.get('refreeze_denergy')
    drift = info.get('refreeze_drift_meV')
    return {prefix + 'opt_grad_max': None if residual is None else float(residual),
            prefix + 'refreeze_drift_meV': None if drift is None else float(drift),
            prefix + 'refreeze_shift_bohr': None if shift is None else float(shift),
            prefix + 'refreeze_denergy_meV': (None if moved is None
                                              else float(moved) * HARTREE_TO_MEV),
            prefix + 'refreeze_converged': info.get('refreeze_converged'),
            prefix + 'refreeze': info.get('refreeze'),
            # the symmetry-breaking start (`optimize.start_nudge`)
            prefix + 'start_nudge': nudge_record(info.get('start_nudge')),
            prefix + 'optimizer': info.get('optimizer'),
            prefix + 'converged': bool(info.get('converged')),
            prefix + 'cycles': int(info.get('cycles', 0)),
            prefix + 'status': info.get('status')}


class WalkNotRelaxed(RuntimeWarning):
    """A walk that stopped short of a minimum: its last pass did not
    converge, its refreeze drift stayed above REFREEZE_TOL_MEV, or its saddle
    check did not end on a minimum. The stage carries on where the walk
    stopped and its record says so (`numerical_flags`)."""


def relaxation_flags(stage, info):
    """{flag: what it says} of the walk of `stage` (a name: 's0', 's1', 't1',
    or a saddle check's escape walk) whose record is `info`, each warned about
    (`WalkNotRelaxed`): 'walk_not_converged' (the last walk's status) and
    'refreeze_not_converged' (the drift, meV). Empty for a relaxed walk; the stage
    carries on where the walk stopped (the policy: refuse inputs, warn
    numerics)."""
    name = stage.upper()
    last = info.get('refreeze_info') or info
    flags = {}
    if not last.get('converged'):
        flags['walk_not_converged'] = str(last.get('status'))
        warnings.warn(f'{name} did not relax: {last.get("status")}; the '
                      f'stage carries on where the walk stopped and its '
                      f'record is flagged', WalkNotRelaxed, stacklevel=2)
    if info.get('refreeze') == 'not converged':
        drift = info.get('refreeze_drift_meV')
        flags['refreeze_not_converged'] = drift
        warnings.warn(f'{name}: the conventions rebuilt at its minimum move '
                      f'its energy by {drift:+.3f} meV, above '
                      f'REFREEZE_TOL_MEV; the stage carries on and its '
                      f'record is flagged', WalkNotRelaxed, stacklevel=2)
    return flags

def saddle_verdict(stage, record):
    """{flag: what it says} of a saddle check's `record` (`saddle_check`):
    'hessian_asymmetric' where either Hessian was (its relative asymmetry by
    Hessian, already warned about by `hessian.check_hessian_noise`), and a
    check that did not end on a minimum (a second saddle, an escape walk that
    did not converge) is warned about (`WalkNotRelaxed`) and flagged
    'saddle_check'; empty when it escaped or found a minimum on Hessians
    within the asymmetry tolerance. The caller carries on where the escape
    walk stopped."""
    flags = {}
    if record.get('hessian_asymmetric'):
        flags['hessian_asymmetric'] = record['hessian_asymmetric']
    if record.get('status') in ('minimum', 'escaped'):
        return flags
    warnings.warn(f'{stage.upper()}: {record["status"]} (saddle check); the '
                  f'walk carries on where the escape walk stopped and its '
                  f'record is flagged', WalkNotRelaxed, stacklevel=2)
    return dict(flags, saddle_check=record['status'])

def nudge_record(nudge):
    """A walk's `start_nudge` record with its displacement as nested lists
    (Bohr), or None for a walk that kept none."""
    if nudge is None:
        return None
    d = nudge.get('displacement_bohr')
    return dict(nudge, displacement_bohr=(None if d is None
                                          else np.asarray(d).tolist()))


def driving_force(info):
    """max |dE/dR| at the first geometry of a relaxation, from its own history."""
    history = info.get('history') or []
    return float(history[0]['grad_max']) if history else None


def excitation_energy(info, e_n, e_0):
    """Omega_n at one geometry: the surface's own number wherever it reports one.

    A neutral excitation's Omega is a Casida root handed back with the
    gradient; rebuilding it as a difference of two large totals would move it
    by their rounding. A charged surface reports no Omega (its excitation is
    -/+ eps^QP), and there the difference of the two totals is the quantity.
    """
    omega = info.get('omega')
    return float(e_n - e_0) if omega is None else float(omega)


def state_specific_record(surface, mol):
    """Scheme B's classical state-specific term at `mol`, in eV, or None.

    A surface whose medium answers a classical state-specific term
    (`state_specific_term`, scheme B) adds E_SS = 1/2 dV^T Q dV of its
    declared density difference to omega_B (the
    `state_specific_correction` ingredient). It is a single-point term: the
    surface's energy and force stay omega_B's, so the record carries E_SS and
    E_B beside them and moves nothing else. Any other medium carries its
    state-specific part inside omega (or has none): None.
    """
    chain = driven_chain(surface)
    if not provides(chain, 'state_specific_correction'):
        return None
    if chain.environment_at(mol).state_specific_term() is None:
        return None
    rec = chain.state_specific_correction(mol)
    return {'omega_B_eV': rec['omega_B'] * HARTREE_TO_EV,
            'E_SS_eV': rec['E_SS'] * HARTREE_TO_EV,
            'E_B_eV': rec['E_B'] * HARTREE_TO_EV,
            'ss_density': rec['ss_density'],
            'ss_equilibrium': rec['ss_equilibrium'],
            'delta_dipole_au': rec['delta_dipole']}


def vertical_block(excited, ground, mol, timer=None, gradient=True,
                   energy=None):
    """Omega_n(R0), the two total energies it is the difference of, and the
    force that says how far R0 is from the excited state's own minimum.

    Omega is the excited surface's own number, taken off the gradient that
    measures the driving force rather than recomputed. The gradient is
    `surface.evaluate`'s, rank 0's on every rank.

    gradient=False is the energy alone: no force is evaluated, Omega is the
    difference of the two totals, and the force fields are None. It is the
    only vertical block a surface whose gradient is refused can have -- the
    dense route on a Kohn-Sham reference has energies and no force. `energy`
    is then the excited total at `mol` where a shared evaluation already
    holds it (`calc_vertical_states`), None to evaluate it here.

    On sliced factors the record carries `factor_gathers`, the whole-array
    gathers the factors at R0 made up to that force, by name: one per sweep
    or solve, whatever the tau count or the Davidson's iterations; on the row
    fit also `fit_held` and `adjoint_held`, the most of each array of the fit
    and of its adjoint rank 0 held at once, in bytes.

    `gradient` is dE_n/dR itself, (natm, 3) Ha/Bohr, whose largest component
    is `driving_force_max`; `translation_residual` is max |sum_A dOmega/dR_A|
    as the chain reports it (None where it does not), zero for an exact
    derivative, and `total_translation_residual` the same of `gradient`.
    `davidson_solves` is the excited chain's record of each Davidson it ran
    with a stage timer set, None for a chain that keeps none (`diagnostics`). `timer` times
    the force and E_0 as 'record_force' and 'record_ground'.

    `qp_bookkeeping` and `held_grid` are what the quasiparticle set and the
    interpolation grid were (the `diagnostics` ingredient, None for a method
    that provides none); both are read after the evaluation, when the chain
    has frozen its conventions at R0. `scf_flags`: every mean field the two
    surfaces' chains read with |F_ia| above `scf_force_tol`, warned about and
    carried on (`scf_flags`); `casida_warnings`: every Casida solve whose read
    roots were left above read_tol or unconfirmed, kept as solved;
    `numerical_flag_warnings`: every named flag the excited chain's forward
    raised (`numerical_flags.flag`), with the geometry it was first raised at.
    """
    refuse_other_numerics(excited, ground)
    with stage(timer, 'record_force'):
        if gradient:
            force, e_n, info = evaluate(excited, mol)
        else:
            force, info = None, {}
            # a shared evaluation's total at `mol`, else this surface's own
            e_n = lockstep(float(
                energy if energy is not None
                else excited.total_energy(lockstep_geometry(mol))))
    with stage(timer, 'record_ground'):
        e_0 = ground_state_at(ground, mol)
    z, route = quasiparticle_diagnostics(info)
    omega = excitation_energy(info, e_n, e_0.total)
    layout = {name: dict(info[name]) for name in ('factor_gathers', 'fit_held',
                                                  'adjoint_held')
              if name in info}
    residual = info.get('translation_residual')
    diagnostics = diagnostics_of(excited, mol)
    forces = {'driving_force_max': None, 'gradient': None,
              'total_translation_residual': None}
    if force is not None:
        force = np.asarray(force, dtype=float)
        forces = {'driving_force_max': float(np.abs(force).max()),
                  'gradient': force.tolist(),
                  'total_translation_residual': float(np.abs(force.sum(axis=0))
                                                      .max())}
    return {**layout, 'omega_eV': omega * HARTREE_TO_EV,
            'e0_hartree': float(e_0.total), 'e0_terms': dict(e_0.terms),
            'en_hartree': float(e_n), **forces,
            'translation_residual': (None if residual is None
                                     else float(residual)),
            'davidson_solves': diagnostics.get('davidson_solves'),
            'opt_grad_max': None, 'refreeze_drift_meV': None,
            'refreeze_shift_bohr': None, 'refreeze_denergy_meV': None,
            'refreeze_converged': None, 'refreeze': NOT_RELAXED,
            'optimizer': None, 'converged': None, 'cycles': 0,
            'status': NOT_RELAXED,
            'qp_z': z, 'residue_route_taken': route,
            'qp_bookkeeping': diagnostics.get('qp_bookkeeping'),
            'held_grid': diagnostics.get('held_grid'),
            'scf_flags': scf_flags(excited, ground),
            'casida_warnings': merged_logs('casida_warnings', diagnostics),
            'numerical_flag_warnings': merged_logs('numerical_flag_warnings',
                                                   diagnostics),
            'physics': physics_record(excited),
            'state_specific': state_specific_record(excited, mol),
            'realization': excited.realization,
            'numerics': dict(excited.numerics),
            'mol_reference': mol}


def saddle_check(surface, mol, walk, verbose=False, cache_dir=None):
    """(minimum, the re-walk's record or None, the check's record) at a
    walk's minimum `mol` on `surface`: the state's Hessian there (6N forces,
    `surface_hessian`) and a saddle left along its negative mode and walked
    again by `walk(surface, start) -> (mol, info)` (`escape_saddle`). The
    record carries the Hessian's asymmetry, 'hessian_asymmetric' ({'first' /
    'second': relative asymmetry} of each Hessian `check_hessian_noise`
    flagged, None when neither was) and how many forces each Hessian read
    back from `cache_dir` (its force cache; the second Hessian's is its
    subdirectory 'second')."""
    info, second = {}, {}
    hess = surface_hessian(surface, mol, verbose=verbose, info=info,
                           cache_dir=cache_dir)
    moved, again, _, record = escape_saddle(
        surface, mol, hess, walk,
        lambda at_surface, at, cache_dir=None: surface_hessian(
            at_surface, at, verbose=verbose, info=second,
            cache_dir=cache_dir),
        cache_dir=cache_dir)
    record['hessian_asymmetry'] = info.get('asymmetry')
    record['hessian_asymmetry_rel'] = info.get('asymmetry_rel')
    record['hessian_asymmetric'] = {
        which: h['asymmetry_rel'] for which, h in (('first', info), ('second', second))
        if h.get('hessian_asymmetric')} or None
    record['forces_reused'] = {'first': info.get('forces_reused'),
                               'second': second.get('forces_reused')}
    return moved, again, record


def saddle_checked_state(relaxed, engine='auto', **walk_kw):
    """(the relaxed state, the saddle check's record) for a `relax_state`
    record `relaxed`: the state's Hessian at the walk's minimum on the
    surface the walk ended on (`surface_hessian`, 6N forces), and at a
    saddle a step along its negative mode and a second walk (`escape_saddle`).

    The second walk is `relax_state` with the first one's `engine` and
    `walk_kw`, so it has the same refreeze passes and the same nudge. Where
    it escaped, its record replaces the first.

    Every event is numerical and is warned about and flagged, the state kept
    where its walk stopped (the policy: refuse inputs, warn numerics; the same
    policy as the photophysics stages): a first walk that
    did not converge is not checked (a Hessian at a point with a residual
    force tests nothing; status 'not run: the walk did not relax'); a second
    saddle, or an escape walk that does not converge, returns the escape
    walk's state. The record's 'numerical_flags' names them
    (`relaxation_flags`, `saddle_verdict`).

    The record (`saddle_check`): 'first' and 'second'
    (`saddle_test`), 'status', 'hessian_asymmetry'. After an escape it also
    has 'step_bohr', 'direction', 'energies', 'energy_at_saddle',
    'energy_drop' (Hartree) and 'saddle_walk', which holds the first walk's
    `relaxation_fields` with its 'converged' and 'cycles'.
    """
    first = relaxed['info']
    if not first.get('converged'):
        # a walk stopped by a collapsed trust radius ends where the surface
        # is not quadratic: its 6N forces read that roughness as asymmetry
        return relaxed, {'status': 'not run: the walk did not relax',
                         'numerical_flags': relaxation_flags('walk', first)}
    walked = {}

    def walk(surface, start):
        state = relax_state(surface, start, engine=engine, **walk_kw)
        walked['state'] = state
        return state['mol'], dict(state['info'],
                                  final_surface=state['surface'])

    _, again, record = saddle_check(relaxed['surface'], relaxed['mol'], walk)
    if again is None:
        record['numerical_flags'] = saddle_verdict('saddle check', record)
        return relaxed, record
    record['saddle_walk'] = dict(relaxation_fields(first),
                                 converged=first.get('converged'),
                                 cycles=first.get('cycles'))
    record['numerical_flags'] = dict(
        relaxation_flags('escape walk', walked['state']['info']),
        **saddle_verdict('saddle check', record))
    return walked['state'], record


def emission_block(excited, ground, mol, refreeze, engine, optimizer,
                   hess_init=None, saddle_check=False):
    """(the vertical record extended to the relaxed state, the relaxation record).

    E_0 is taken at the excited state's minimum, not at R0: the emission
    energy is the vertical gap at the relaxed geometry. Both energies at the
    minimum come off one pair of conventions, the ones the walk ended on:
    after a refreeze the excited surface's were rebuilt at the minimum
    (`relax_state`), and E_0 is taken on the ground surface rebuilt there too.
    The vertical block at R0 is the pair frozen at R0, so
    `relaxation_depth_eV` is the difference of the route's own energies at
    the two geometries.

    hess_init: the excited relaxation's starting Cartesian Hessian (another
    state's `hessian_at_excited_minimum`, say), or None for the optimizer's
    own guess; the record's `hessian_at_excited_minimum` is the one the
    relaxation ended with, None where the optimizer reports none.

    saddle_check: test the excited minimum's Hessian and leave a saddle
    (`saddle_checked_state`). The record carries `saddle_check`, and
    everything at the minimum is read at the minimum the check returned.
    """
    record = vertical_block(excited, ground, mol)
    given = {} if hess_init is None else {'hess_init': hess_init}
    relaxed = relax_state(excited, mol, engine=engine, refreeze=refreeze,
                          **optimizer, **given)
    if saddle_check:
        relaxed, record['saddle_check'] = saddle_checked_state(
            relaxed, engine=engine, refreeze=refreeze, **optimizer)
    e_0 = ground_state_at(ground_at_minimum(ground, excited, relaxed),
                          relaxed['mol'])
    e_n = float(relaxed['e_total'])
    # the adaptive set the last walk ran on, judged where it ended, before
    # the logs are read: its forward logs too
    at_minimum = diagnostics_of(relaxed['surface'], relaxed['mol'], relaxed['mf'],
                                posteriori=True)
    check = at_minimum.get('adaptive_check')
    logs = (diagnostics_of(excited, mol), at_minimum)
    adaptive = (record.get('qp_bookkeeping') or {}).get('adaptive')
    if check is not None and adaptive is not None:
        adaptive['posteriori'] = check
    record.update(
        emission_eV=(e_n - e_0.total) * HARTREE_TO_EV,
        relaxation_depth_eV=(record['en_hartree'] - e_n) * HARTREE_TO_EV,
        en_hartree_at_excited_minimum=e_n,
        e0_hartree_at_excited_minimum=float(e_0.total),
        e0_terms_at_excited_minimum=dict(e_0.terms),
        mol_excited_minimum=relaxed['mol'],
        state_specific_at_excited_minimum=state_specific_record(
            relaxed['surface'], relaxed['mol']),
        hessian_at_excited_minimum=relaxed['info'].get('hessian'),
        qp_branch_warnings=merged_logs('qp_branch_warnings', *logs),
        qp_hamiltonian_warnings=merged_logs('qp_hamiltonian_warnings', *logs),
        scf_flags=scf_flags(excited, relaxed['surface'], ground),
        casida_warnings=merged_logs('casida_warnings', *logs),
        numerical_flag_warnings=merged_logs('numerical_flag_warnings', *logs),
        **relaxation_fields(relaxed['info']))
    return record, relaxed


def scf_flags(*surfaces):
    """Every mean field the chains of `surfaces` read with |F_ia| above
    `scf_force_tol` (`ReferenceMeanField.flag_scf_quality`): where, the
    residual, the tolerance and the nuclei (Bohr), once each, in the order
    flagged. The run carried on; a number off a flagged mean field carries
    the residual in its force."""
    out, seen = [], set()
    # a composed surface's ground half reads mean fields of its own
    holders = [h for s in surfaces for h in (driven_chain(s), ground_half(s))
               if h is not None and provides(h, 'scf_flags')]
    for holder in holders:
        for flag in holder.scf_flags or []:
            coords = np.asarray(flag['coords'], float)
            key = (flag['where'], flag['residual'], coords.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(dict(flag, coords=coords.tolist()))
    return out


def ground_at_minimum(ground, excited, relaxed):
    """The ground surface E_0 is read on at an excited minimum: `ground`
    itself where the walk ended on `excited`, else `ground` rebuilt at the
    minimum, as the excited surface was (`relax_state`), on the walk's own
    mean field there where its `refreeze` takes one: a second SCF at R*
    would put E_0 and the excited energy on two references."""
    if relaxed['surface'] is excited:
        return ground
    return ground.refreeze(relaxed['mol'], mf=relaxed['mf'])


def refuse_incomparable(physics_a, physics_b):
    """Refuse two declarations whose difference would be of two functionals."""
    if not physics_a.comparable_with(physics_b):
        raise PhysicsMismatch(physics_a, physics_b)


def refuse_other_numerics(excited, ground):
    """Refuse an E_0 surface and an excited surface on different numerics: their
    difference would carry the difference of the quadratures. A surface that
    holds no `Numerics` (mean field, dense) is not refused."""
    require_same_identity(numerics_identity(excited), numerics_identity(ground),
                          'E_0 against the excited energy')


def adiabatic_energy(surface_excited, surface_ground, mol, refreeze=1,
                     engine='auto', hess_init=None, saddle_check=False,
                     **optimizer):
    """E_n(R*_n) - E_0(R*_0) for two surfaces already built, with the refusal.

    The two-surface form of `calc_adiabatic_excitation`, for a pair somebody
    else constructed. `compare_surfaces` is the refusal: a differing
    ground-state functional or environment raises `PhysicsMismatch` before
    either relaxation runs, and a surface with no declared physics is refused
    too.

    `ground_realization_differs` is reported rather than refused: E_0 has no
    residue backend, eigensolver or quasiparticle set, so the two halves are
    realized differently by construction; the list names those fields.
    `hess_init` starts the excited relaxation alone, and `saddle_check` tests
    the excited minimum alone (`emission_block`).
    """
    report = compare_surfaces(surface_excited, surface_ground)
    refuse_other_numerics(surface_excited, surface_ground)
    started = time.perf_counter()
    record, _ = emission_block(surface_excited, surface_ground, mol, refreeze,
                               engine, optimizer, hess_init=hess_init,
                               saddle_check=saddle_check)
    relaxed = relax_state(surface_ground, mol, engine=engine,
                          refreeze=refreeze, **optimizer)
    e_0 = ground_state_at(relaxed['surface'], relaxed['mol'],
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


def calc_vertical_excitation(spec, excitation, mol, scf_factory, timer=None,
                             gradient=True):
    """Omega_n(R0) in eV: the excitation energy at the geometry as given.

    No geometry moves, so the residual at a minimum and the refreeze drift are
    None with the marker saying so; the driving force at R0 says how far the
    geometry is from a stationary point.

    timer: a dict the stages accumulate their seconds into (the surface's
    construction with its SCF as 'record_surface', then the excited chain's
    own stages and `vertical_block`'s), or None for no timing. With a timer
    the record also carries what 'record_surface' was made of
    (`surface_seconds`, `surface_parts`) and the reference SCF's cycles
    (`reference_scf`, `scf_record`).

    gradient=False evaluates the energies alone (`vertical_block`).
    """
    started = time.perf_counter()
    parts = {}
    with stage(timer, 'record_surface'):
        # the reference mean field `potential_energy_surface` would converge
        # first, converged here so that its seconds are its own
        with stage(parts, 'reference_scf'):
            mf = reference_mean_field(mol, scf_factory, spec.environment)
        with stage(parts, 'excited_surface'):
            excited = surface_of(spec, excitation, mol, scf_factory, mf=mf)
        with stage(parts, 'ground_surface'):
            ground = ground_surface_of(spec, excited, mol, scf_factory)
    time_stages(excited, timer)
    record = vertical_block(excited, ground, mol, timer=timer,
                            gradient=gradient)
    if timer is not None:
        record.update(surface_seconds=parts,
                      surface_parts=diagnostics_of(excited, mol).get('surface_parts'),
                      reference_scf=scf_record(mf))
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_vertical_states(spec, excitations, mol, scf_factory, timer=None,
                         gradient=True):
    """{excitation: vertical record} for several states of one declaration at
    R0, off ONE shared evaluation (`StateManifold.evaluate`).

    One reference SCF, one excited surface and one ground surface, as for one
    state; then the factors, the static W and the quasiparticle set once,
    one Casida solve per spin, one reverse pass per state and the
    ground-state force once. Each record is `vertical_block`'s on its state's
    own surface, whose force at R0 is the evaluation's (`FirstPoint`): the
    record `calc_vertical_excitation` writes for that state alone, on the
    same mean field, apart from the clocks. The evaluation is timed as
    'record_states'.

    The excitations may differ in spin and root only.

    gradient=False evaluates the energies alone, still once: no reverse pass
    and no ground-state force, each record `vertical_block`'s energy-only
    block on the shared evaluation's total. A surface whose forward is not
    shared below the Casida step (the dense quasi-boson route, which takes
    root indices) raises `NoSharedForward` once its surfaces are built, so
    that the caller takes the states one at a time.
    """
    excitations = tuple(excitations)
    if not excitations:
        raise ValueError('calc_vertical_states needs at least one excitation')
    first = excitations[0]
    for other in excitations[1:]:
        if (other.irrep, other.kernel, other.qp, other.screening) != (
                first.irrep, first.kernel, first.qp, first.screening):
            raise ValueError(
                f'{other} and {first} differ in more than spin and root; one '
                f'shared evaluation serves one kernel on one quasiparticle '
                f'and screening treatment')
    started = time.perf_counter()
    parts = {}
    with stage(timer, 'record_surface'):
        with stage(parts, 'reference_scf'):
            mf = reference_mean_field(mol, scf_factory, spec.environment)
        with stage(parts, 'excited_surface'):
            excited = surface_of(spec, first, mol, scf_factory, mf=mf)
        with stage(parts, 'ground_surface'):
            ground = ground_surface_of(spec, excited, mol, scf_factory)
    if not StateManifold(excited).shares_forward:
        raise NoSharedForward(
            f'{type(driven_chain(excited)).__name__} solves one root per '
            'forward pass; take the states one at a time')
    time_stages(excited, timer)
    targets = tuple((x.spin, x.root - 1) for x in excitations)
    manifold = StateManifold(excited, states=targets)
    with stage(timer, 'record_states'):
        shared = manifold.evaluate(lockstep_geometry(mol),
                                   gradients=targets if gradient else ())
    records = {}
    for excitation, target in zip(excitations, targets):
        if gradient:
            surface = manifold.surface(target, first_point=shared)
            record = vertical_block(surface, ground, mol, timer=timer)
        else:
            record = vertical_block(manifold.surface(target), ground, mol,
                                    timer=timer, gradient=False,
                                    energy=shared.energy[target])
        if timer is not None:
            record.update(surface_seconds=parts,
                          surface_parts=diagnostics_of(excited, mol).get(
                              'surface_parts'),
                          reference_scf=scf_record(mf))
        record['provenance'] = provenance(time.perf_counter() - started)
        records[excitation] = record
    return records


def calc_adiabatic_excitation(spec, excitation, mol, scf_factory, refreeze=1,
                              engine='auto', hess_init=None,
                              saddle_check=False, **optimizer):
    """E_n(R*_n) - E_0(R*_0) in eV: both states at their own minima.

    The two minima are on two surfaces, so E_0 does not cancel and every term
    of it has to be the same on both sides. Carries the vertical and emission
    blocks and both refreeze shifts, since an adiabatic energy is only as
    converged as the looser of its two relaxations.

    hess_init: the excited relaxation's starting Cartesian Hessian (another
    state's `hessian_at_excited_minimum`, the singlet's for the triplet
    started at the singlet minimum), or None for the optimizer's own guess;
    the ground relaxation keeps its own guess.

    saddle_check: the excited minimum's Hessian is tested and a saddle left
    (`saddle_checked_state`), recorded as `saddle_check`.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = ground_surface_of(spec, excited, mol, scf_factory)
    record = adiabatic_energy(excited, ground, mol, refreeze=refreeze,
                              engine=engine, hess_init=hess_init,
                              saddle_check=saddle_check, **optimizer)
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_adiabatic_gap(spec_a, excitation_a, spec_b, excitation_b, mol,
                       scf_factory, refreeze=1, engine='auto', **optimizer):
    """E_a(R*_a) - E_b(R*_b) in eV: the adiabatic gap of two states, each at
    its own minimum.

    Delta-E_ST when a is the singlet and b the triplet. Refused unless the two
    specs declare the same ground-state functional and environment: the two
    states relax by different amounts, so E_0 does not cancel (a mean-field
    E_0 against E_HF + E_c^dRPA is 6.3 eV of correlation energy on water). The
    refusal is taken on the declarations, before an integral is computed.

    The realizations may differ and are reported side by side
    (`realization_differs`), e.g. a dense oracle against a cubic route.
    The numerics may not: two cubic surfaces on different grids, tolerances or
    quadratures are refused (`NumericsMismatch`) before a relaxation starts.
    """
    started = time.perf_counter()
    refuse_incomparable(spec_a.physics(excitation_a), spec_b.physics(excitation_b))
    surface_a = surface_of(spec_a, excitation_a, mol, scf_factory)
    surface_b = surface_of(spec_b, excitation_b, mol, scf_factory)
    # the two minima are on two surfaces: other numerics on one side move the
    # gap by the difference of the numerics, so the relaxations are not started
    require_same_identity(numerics_identity(surface_a),
                          numerics_identity(surface_b), 'adiabatic gap')
    report = compare_surfaces(surface_a, surface_b)
    record_a, relaxed_a = emission_block(
        surface_a, ground_surface_of(spec_a, surface_a, mol, scf_factory),
        mol, refreeze, engine, optimizer)
    record_b, relaxed_b = emission_block(
        surface_b, ground_surface_of(spec_b, surface_b, mol, scf_factory),
        mol, refreeze, engine, optimizer)
    return {'gap_eV': adiabatic_gap(relaxed_a, relaxed_b) * HARTREE_TO_EV,
            'physics': {'label': report['physics'],
                        'record': (surface_a.physics, surface_b.physics)},
            'realization': report['realization'],
            'realization_differs': report['realization_differs'],
            'numerics': (dict(surface_a.numerics), dict(surface_b.numerics)),
            'state_a': record_a, 'state_b': record_b,
            'provenance': provenance(time.perf_counter() - started)}
