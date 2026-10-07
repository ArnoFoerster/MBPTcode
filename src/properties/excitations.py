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
`potential_energy_surface` refuses it. `root_by_irrep` takes the irrep from
the spectrum at the reference geometry, the lowest root of that state
symmetry, and returns the root to declare.

Every vertical block also says what its quasiparticle treatment was
(`qp_bookkeeping`): which orbitals were solved explicitly, which carry the
frozen scissor and what shift, which route each explicit state took, how many
poles the sum-over-poles model used; and which interpolation grid the
factorization held (`held_grid`).
"""
import datetime
import functools
import inspect
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pyscf import gto

from src.Base.constants import (GIT_PROVENANCE_TIMEOUT, HARTREE_TO_EV,
                                HARTREE_TO_MEV, SURFACE_GRID_ACCURACY)
from src.Base.declaration import (Excitation, GroundState, PhysicsMismatch,
                                  SurfacePhysics)
from src.Base.scf_convergence import scf_record
from src.Base.separable_ri import grid_points_per_atom
from src.Base.utils.mpi_grid import lockstep
from src.SingleReference.LinearResponse.rpa_energy import (ground_state_energy,
                                                           reference_energy)
from src.properties import characters
from src.properties.surface import (driven_chain, evaluate, lockstep_geometry,
                                    surface_mean_field)
from src.properties.surfaces import (compare_surfaces, environment_label,
                                     find_row, potential_energy_surface,
                                     reference_mean_field, resolve_grid)
from src.properties.vibronic import adiabatic_gap, relax_state

#: Everything `potential_energy_surface` takes beyond the molecule, the SCF
#: factory, the state and the numerics: the physics axes first, then the
#: realization ones.
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

    numerics: the grids, caps and tolerances the realization reads, as a sorted
        tuple of (key, value) pairs so that the spec hashes and two specs
        written in different orders compare equal. A mapping is accepted and
        normalized. A value that is itself a mapping (an explicit ISDF
        `counts`) travels verbatim and makes that spec unhashable rather than
        being rewritten into something the realizing class does not read.

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


def surface_of(spec, excitation, mol, scf_factory, mf=None):
    """The surface `spec` declares with `excitation` on it; None gives E_0's own.

    `mf` is a reference mean field already converged for this declaration at
    `mol`; None converges one.

    Both surfaces come from one spec, so their difference is of one
    functional with itself at two geometries. The axes only a state has (the
    residue backend, the eigensolver, the quasiparticle set) and the numerics
    the ground row does not read are dropped for the ground surface, since the
    entry point would refuse them.
    """
    kwargs = spec.as_kwargs()
    if excitation is not None:
        return potential_energy_surface(mol, scf_factory, mf=mf,
                                        excitation=excitation, **kwargs)
    row = find_row(spec.ground_state, None, spec.chi0, spec.factorization)
    ground = {name: kwargs[name] for name in GROUND_AXES}
    ground.update((key, value) for key, value in spec.numerics
                  if key in row.numerics)
    return potential_energy_surface(mol, scf_factory, mf=mf, **ground)


def ground_surface_of(spec, excited, mol, scf_factory):
    """E_0's surface of `spec` at `mol`, on the excited surface's reference.

    One SCF per record: a surface that holds its converged reference (`mf0`,
    the excited chain's) hands it to the ground surface, which declares its
    functional off it and, in the gas phase, evaluates E_0 at `mol` on it.
    """
    return surface_of(spec, None, mol, scf_factory,
                      mf=getattr(excited, 'mf0', None))


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


@contextmanager
def stage(timer, key):
    """Time the block into `timer[key]` (accumulating) by one assignment at its
    end, as `FactorChain.phase` does; nothing when `timer` is None."""
    if timer is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer[key] = timer.get(key, 0.0) + time.perf_counter() - t0


def time_stages(surface, timer):
    """Hand `timer` to the chain that owns `surface`'s spectrum, whose stages
    (`FactorChain.phase`) then accumulate into it; a surface with no stages
    is left as it is."""
    chain = driven_chain(surface)
    if timer is not None and hasattr(chain, 'phase'):
        chain.timer = timer


def surface_parts(surface):
    """What building the chain behind `surface` spent its seconds on (its
    constructor's steps and its factorization's conventions) and the reference
    mean field's max |F_ia| it measured; None for a surface with no chain."""
    chain = driven_chain(surface)
    seconds = getattr(chain, 'construction_seconds', None)
    if seconds is None:
        return None
    factorization = getattr(chain, 'factorization', None)
    residual = getattr(chain, 'scf_residual', None)
    return {'chain': dict(seconds),
            'factorization': dict(getattr(factorization, 'seconds', {}) or {}),
            'scf_residual': None if residual is None else float(residual)}


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
    they come to and the multi-start recipe. A grid nobody validated is
    refused here, before an integral is computed; a row with no interpolation
    grid reports None throughout.
    """
    row = find_row(spec.ground_state, Excitation('singlet'), spec.chi0,
                   spec.factorization)
    if not row.isdf_grid:
        return dict(grid_level=None, grid_counts=None,
                    grid_points_per_atom=None, grid_n_start=None,
                    grid_label=None)
    numerics = dict(spec.numerics)
    counts, n_start, label = resolve_grid(mol, numerics)
    level = numerics.get('grid_accuracy')
    if level is None and numerics.get('counts') is None \
            and numerics.get('radii') is None:
        level = SURFACE_GRID_ACCURACY
    return dict(grid_level=level,
                grid_counts=None if counts is None else dict(counts),
                grid_points_per_atom=(None if counts is None
                                      else grid_points_per_atom(counts)),
                grid_n_start=n_start, grid_label=label)


def held_grid(surface):
    """The grid the factorization of `surface`'s chain actually holds, or None.

    Read off the frozen factorization rather than the request: the counts and
    recipe that fitted the radii, and the Coulomb-metric fit error each
    element's grid reached, which is what says whether the grid resolves the
    basis for that element.
    """
    factorization = getattr(driven_chain(surface), 'factorization', None)
    if factorization is None:
        return None
    return {'counts': dict(factorization.counts),
            'points_per_atom': grid_points_per_atom(factorization.counts),
            'n_start': factorization.n_start,
            'fit_errors': {el: float(err) for el, err
                           in factorization.fit_errors.items()}}


def scissor_tiers(shifts, lent):
    """The frozen outside shifts grouped by the explicit root that supplied them.

    `calibrate_scissor` gives every outside orbital the shift of the probe
    nearest it in orbital energy, so the orbitals sharing one shift are one
    tier and the probe is the explicit orbital that lent it (`lent`, {probe:
    shift}: its root minus eps, and minus its own Eq. (18) term in a
    continuum). Shifts in eV.
    """
    by_shift = {}
    for p, shift in shifts.items():
        by_shift.setdefault(float(shift), []).append(int(p))
    tiers = []
    for shift, orbitals in sorted(by_shift.items()):
        probe = min(lent, key=lambda q: abs(lent[q] - shift), default=None)
        tiers.append({'probe': None if probe is None else int(probe),
                      'shift_eV': shift * HARTREE_TO_EV,
                      'n_orbitals': len(orbitals), 'orbitals': sorted(orbitals)})
    return tiers


def explicit_routes(chain):
    """{orbital: route} of every explicitly solved state.

    'sop' is the pole model; 'scissor' a state inside the set that Eq. (27)
    does not admit, solved once at the reference geometry by the fallback and
    frozen as a shift; anything else is the chain's own residue route, which
    is what a state carries when no pole model or calibration claimed it. The
    chain's own record of the routes its last solve took wins where it keeps
    one.
    """
    taken = {int(p): r for p, r in
             ((getattr(chain, 'qp_diagnostics', None) or {})
              .get('routes') or {}).items()}
    routes = {}
    for p in (int(q) for q in chain.qp_set):
        if p in taken:
            routes[p] = taken[p]
        elif p in chain.sop_poles:
            routes[p] = 'sop'
        elif p in chain.scissor_map:
            routes[p] = 'scissor'
        else:
            routes[p] = chain.residue_route
    return routes


def qp_bookkeeping(surface, mol):
    """What the quasiparticle treatment of `surface`'s state was, orbital by orbital.

    The explicitly solved set (with the root each carries and the route it
    took), the orbitals outside it and what they carry -- the frozen scissor,
    in tiers, or the bare mean-field eigenvalue on the dense route, which has
    no scissor -- and for the sum-over-poles model the number of auxiliary
    poles each state used and where they sit. Read after the first evaluation,
    when the chain has frozen its conventions at the reference geometry;
    `explicit` and `outside` partition the orbitals. None for a surface with
    no quasiparticle set.

    `demoted`: the orbitals the declaration made explicit whose reference
    root had a pole strength outside (0, 1], outside the set, with the
    reason, the route and the rejected root and Z.

    Shifts and roots are in eV; `z` is the pole strength per explicit state,
    None until the chain keeps it; `qp_root` (dense only) which root of the
    quasiparticle equation each explicit state carries, the nearest by Newton
    or the one of largest pole strength.
    """
    chain = driven_chain(surface)
    if not hasattr(chain, 'qp_set'):
        return None
    cubic = hasattr(chain, 'outside_shift')
    norb = int(len(chain.mf0.mo_energy)) if cubic else int(mol.nao)
    explicit = sorted(int(p) for p in chain.qp_set)
    outside = [p for p in range(norb) if p not in set(explicit)]
    roots = {int(p): float(w) for p, w in
             (chain.qp_seeds if cubic else chain.seeds).items()}
    record = {'kind': 'cubic' if cubic else 'dense', 'n_orbitals': norb,
              'explicit': explicit, 'n_explicit': len(explicit),
              'outside': outside, 'n_outside': len(outside),
              'qp_root_eV': {p: w * HARTREE_TO_EV for p, w in roots.items()}}
    if not cubic:
        record.update(outside_treatment='mean-field', n_scissor=0,
                      scissor_tiers=[], route=None, inside_scissor_eV={},
                      sop=None, demoted={}, qp_root=chain.qp_root,
                      z={int(p): float(v) for p, v in chain.qp_z.items()})
        return record
    eps = np.asarray(chain.mf0.mo_energy, float)
    scissored = chain.outside == 'scissor'
    shifts = (chain.outside_shift or {}) if scissored else {}
    diagnostics = getattr(chain, 'qp_diagnostics', None) or {}
    routes = explicit_routes(chain)
    sop = None
    if chain.residue_route == 'sop':
        poles = {int(p): np.asarray(v, float) for p, v in chain.sop_poles.items()}
        sop = {'n_poles_asked': int(chain.n_poles),
               'sop_stride': chain.sop_stride,
               'poles_per_state': {p: int(len(v)) for p, v in poles.items()},
               'poles_hartree': {p: v.tolist() for p, v in poles.items()}}
    record.update(
        outside_treatment=chain.outside,
        n_scissor=len(outside) if scissored else 0,
        scissor_tiers=scissor_tiers(shifts, chain.outside_lent or {}),
        scissor_eV={int(p): float(s) * HARTREE_TO_EV
                    for p, s in sorted(shifts.items())},
        route=routes, n_sop=sum(r == 'sop' for r in routes.values()),
        n_inside_scissor=sum(r == 'scissor' for r in routes.values()),
        inside_scissor_eV={int(p): float(s) * HARTREE_TO_EV
                           for p, s in sorted(chain.scissor_map.items())},
        eps_mean_field_eV={p: float(eps[p]) * HARTREE_TO_EV for p in explicit},
        sop=sop, z=diagnostics.get('z'),
        # declared explicit, but the reference root was no quasiparticle
        # (`_settle_qp_set`): outside the set, with why and the root rejected
        demoted={int(p): {'reason': d['reason'], 'z': d['z'],
                          'route': d['route'],
                          'root_eV': d['root'] * HARTREE_TO_EV}
                 for p, d in sorted(getattr(chain, 'qp_demoted',
                                            {}).items())})
    return record


def spectrum_of(surface, mol, mf):
    """(Omega, X, Y, nocc): every root the surface's Casida step solved at `mol`."""
    chain = driven_chain(surface)
    if hasattr(chain, '_forward'):
        return characters.roots(chain, mol, mf)
    solved = chain._solve(mol, mf)[0]
    return solved.Omega, solved.X, solved.Y, mol.nelectron // 2


def symmetric_copy(mol):
    """`mol` rebuilt with its point group detected, from its CURRENT coordinates.

    Read off `atom_coords` and not `mol.atom`: a molecule that went through
    `set_geom_` keeps the string it was created from, so rebuilding from that
    would label the reference geometry's orbitals at a displaced one.
    """
    return gto.M(atom=[(mol.atom_pure_symbol(i), tuple(c)) for i, c
                       in enumerate(mol.atom_coords())],
                 unit='Bohr', basis=mol.basis, symmetry=True,
                 max_memory=mol.max_memory, verbose=0)


def labelled_spectrum(spec, excitation, mol, scf_factory, listed=8):
    """The roots the route solves at `mol`, each labelled by its state irrep.

    A state's symmetry is the direct product Gamma_i (x) Gamma_a over its
    transition amplitude (`characters.root_irreps`), so the spectrum is the
    realization's own, solved on a copy of `mol` built with its point group.
    Returns (omega in Hartree, the per-root labels, the point group, the
    `listed` lowest roots as records with energy, irrep and purity).
    """
    symmetric = symmetric_copy(mol)
    surface = surface_of(spec, excitation, symmetric, scf_factory)
    symmetric, mf = surface.mean_field(symmetric)
    omega, x, y, nocc = spectrum_of(surface, symmetric, mf)
    labels = characters.root_irreps(symmetric, mf.mo_coeff, nocc, x, y)
    order = np.argsort(np.asarray(omega))[:listed]
    roots = [{'root': int(n), 'omega_eV': float(omega[n]) * HARTREE_TO_EV,
              'irrep': labels[int(n)]['irrep'],
              'purity': labels[int(n)]['purity']} for n in order]
    return np.asarray(omega), labels, symmetric.groupname, roots


def root_by_irrep(spec, excitation, mol, scf_factory, irrep, listed=8):
    """The lowest root of state irrep `irrep` at the reference geometry, as a record.

    THE INDEX IS THIS ROUTE'S OWN: two realizations that order near roots
    differently are each asked for the state, not for the other one's index.
    An irrep absent from the roots solved is an error naming the roots, never
    a fallback to root 1. `index` is zero-based; the record lists the `listed`
    lowest roots with their labels and purity.
    """
    omega, labels, group, roots = labelled_spectrum(spec, excitation, mol,
                                                    scf_factory, listed)
    index = next((int(n) for n in np.argsort(omega)
                  if labels[int(n)]['irrep'] == irrep), None)
    if index is None:
        raise ValueError(
            f'no {irrep!r} root among the {len(omega)} solved at {group}: '
            + ', '.join(f"{r['irrep']} {r['omega_eV']:.3f} eV" for r in roots)
            + '. Raise nroots, or name an irrep the point group has.')
    return {'target': irrep, 'index': index, 'point_group': group,
            'roots': roots, 'impure': characters.impure(labels)}


def followed_state(spec, excitation, mol, scf_factory, irrep, listed=8):
    """Whether the root a relaxation followed by index is still the state at `mol`.

    A walk keeps the n-th root of an energy-ordered manifold, so a crossing
    along the path hands it another state without saying so. At the relaxed
    geometry the root `excitation` indexes is labelled and compared with the
    lowest root of `irrep`: `same_state` is False when they differ, and None
    when the point group is lost and no label means anything. The surface is
    built afresh at `mol`, so its energies are not the walk's; the labels are
    what is read.
    """
    omega, labels, group, roots = labelled_spectrum(spec, excitation, mol,
                                                    scf_factory, listed)
    held = excitation.root - 1
    lowest = next((int(n) for n in np.argsort(omega)
                   if labels[int(n)]['irrep'] == irrep), None)
    lost = group in ('C1',)
    return {'target': irrep, 'point_group': group, 'followed_root': held,
            'followed_irrep': labels[held]['irrep'],
            'followed_purity': labels[held]['purity'],
            'lowest_of_target': lowest, 'roots': roots,
            'same_state': None if lost else lowest == held}


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

    `info['grad_max']` and `info['opt_grad_max']` are one number under two
    names (the residual |dE/dR| at the converged geometry), and the first is
    what the driving force at the input geometry is called elsewhere. Only the
    unambiguous name travels.

    The refreeze shift is reported as how far the geometry moved when the
    frozen conventions were rebuilt (Bohr) and how much energy that motion was
    worth (meV). Either is None only when the outer loop did not run, and
    `refreeze` then says why.
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
    with a stage timer set, None for a chain that keeps none. `timer` times
    the force and E_0 as 'record_force' and 'record_ground'.

    `qp_bookkeeping` and `held_grid` are what the quasiparticle set and the
    interpolation grid were; both are read after the evaluation, when the
    chain has frozen its conventions at R0.
    """
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
    solves = getattr(driven_chain(excited), 'davidson_solves', None)
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
            'davidson_solves': None if solves is None else list(solves),
            'opt_grad_max': None, 'refreeze_shift_bohr': None,
            'refreeze_shift_meV': None, 'refreeze': NOT_RELAXED,
            'optimizer': None, 'converged': None, 'cycles': 0,
            'status': NOT_RELAXED,
            'qp_z': z, 'residue_route_taken': route,
            'qp_bookkeeping': qp_bookkeeping(excited, mol),
            'held_grid': held_grid(excited),
            'physics': physics_record(excited),
            'realization': excited.realization,
            'numerics': dict(excited.numerics),
            'mol_reference': mol}


def emission_block(excited, ground, mol, refreeze, engine, optimizer,
                   hess_init=None):
    """(the vertical record extended to the relaxed state, the relaxation record).

    E_0 is taken at the excited state's minimum, not at R0: the emission
    energy is the vertical gap at the relaxed geometry. Both energies come off
    the reference geometry's frozen conventions, so their difference is one
    surface pair evaluated twice.

    hess_init: the excited relaxation's starting Cartesian Hessian (another
    state's `hessian_at_excited_minimum`, say), or None for the optimizer's
    own guess; the record's `hessian_at_excited_minimum` is the one the
    relaxation ended with, None where the optimizer reports none.
    """
    record = vertical_block(excited, ground, mol)
    given = {} if hess_init is None else {'hess_init': hess_init}
    relaxed = relax_state(excited, mol, engine=engine, refreeze=refreeze,
                          **optimizer, **given)
    e_0 = ground_state_at(ground, relaxed['mol'])
    e_n = float(relaxed['e_total'])
    record.update(
        emission_eV=(e_n - e_0.total) * HARTREE_TO_EV,
        relaxation_depth_eV=(record['en_hartree'] - e_n) * HARTREE_TO_EV,
        en_hartree_at_excited_minimum=e_n,
        e0_hartree_at_excited_minimum=float(e_0.total),
        e0_terms_at_excited_minimum=dict(e_0.terms),
        mol_excited_minimum=relaxed['mol'],
        hessian_at_excited_minimum=relaxed['info'].get('hessian'),
        **relaxation_fields(relaxed['info']))
    return record, relaxed


def refuse_incomparable(physics_a, physics_b):
    """Refuse two declarations whose difference would be of two functionals."""
    if not physics_a.comparable_with(physics_b):
        raise PhysicsMismatch(physics_a, physics_b)


def adiabatic_energy(surface_excited, surface_ground, mol, refreeze=1,
                     engine='auto', hess_init=None, **optimizer):
    """E_n(R*_n) - E_0(R*_0) for two surfaces already built, with the refusal.

    The two-surface form of `calc_adiabatic_excitation`, for a pair somebody
    else constructed. `compare_surfaces` is the refusal: a differing
    ground-state functional or environment raises `PhysicsMismatch` before
    either relaxation runs, and a surface with no declared physics is refused
    too.

    `ground_realization_differs` is reported rather than refused: E_0 has no
    residue backend, eigensolver or quasiparticle set, so the two halves are
    realized differently by construction; the list names those fields.
    `hess_init` starts the excited relaxation alone (`emission_block`).
    """
    report = compare_surfaces(surface_excited, surface_ground)
    started = time.perf_counter()
    record, _ = emission_block(surface_excited, surface_ground, mol, refreeze,
                               engine, optimizer, hess_init=hess_init)
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
        record.update(surface_seconds=parts, surface_parts=surface_parts(excited),
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
    # cycle: state_manifold imports src.properties, whose __init__ imports this
    from src.gradients.state_manifold import StateManifold
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
                          surface_parts=surface_parts(excited),
                          reference_scf=scf_record(mf))
        record['provenance'] = provenance(time.perf_counter() - started)
        records[excitation] = record
    return records


def calc_emission_energy(spec, excitation, mol, scf_factory, refreeze=1,
                         engine='auto', hess_init=None, **optimizer):
    """Omega_n(R*_n) in eV: the gap at the relaxed geometry of the state.

    The state is relaxed from `mol` and the ground state is not: emission
    leaves the excited minimum for the ground surface at the same geometry,
    so E_0 is evaluated there. `relaxation_depth_eV` is what the state shed
    getting there, and `refreeze_shift_meV` is the frozen conventions' error
    bar on the result.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = ground_surface_of(spec, excited, mol, scf_factory)
    record, _ = emission_block(excited, ground, mol, refreeze, engine, optimizer,
                               hess_init=hess_init)
    record['provenance'] = provenance(time.perf_counter() - started)
    return record


def calc_adiabatic_excitation(spec, excitation, mol, scf_factory, refreeze=1,
                              engine='auto', hess_init=None, **optimizer):
    """E_n(R*_n) - E_0(R*_0) in eV: both states at their own minima.

    The two minima are on two surfaces, so E_0 does not cancel and every term
    of it has to be the same on both sides. Carries the vertical and emission
    blocks and both refreeze shifts, since an adiabatic energy is only as
    converged as the looser of its two relaxations.

    hess_init: the excited relaxation's starting Cartesian Hessian (another
    state's `hessian_at_excited_minimum`, the singlet's for the triplet
    started at the singlet minimum), or None for the optimizer's own guess;
    the ground relaxation keeps its own guess.
    """
    started = time.perf_counter()
    excited = surface_of(spec, excitation, mol, scf_factory)
    ground = ground_surface_of(spec, excited, mol, scf_factory)
    record = adiabatic_energy(excited, ground, mol, refreeze=refreeze,
                              engine=engine, hess_init=hess_init, **optimizer)
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
    """
    started = time.perf_counter()
    refuse_incomparable(spec_a.physics(excitation_a), spec_b.physics(excitation_b))
    surface_a = surface_of(spec_a, excitation_a, mol, scf_factory)
    surface_b = surface_of(spec_b, excitation_b, mol, scf_factory)
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
