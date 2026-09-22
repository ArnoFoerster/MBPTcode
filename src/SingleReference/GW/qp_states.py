"""Which orbitals carry an explicitly solved quasiparticle energy, and what the
rest carry instead.

A BSE diagonal is built from eps^QP_p, but solving the quasiparticle equation
for every orbital is neither affordable nor meaningful: the deep states have no
compressible self-energy, and a root that has walked past a neighbouring
orbital energy is a satellite as often as a quasiparticle. So a surface names a
SET. Inside it every orbital is solved; outside it every orbital keeps its
mean-field eigenvalue plus a frozen scissor. Four declarations of that set are
in use:

    admitted   the orbitals Eq. (27) admits: |omega_p - eps_q| < Om_1 for every
               pole q the contour deformation sweeps, Om_1 the lowest neutral
               dRPA excitation. threshold='gap' tests against the particle-hole
               gap E_g, which the dRPA guarantees is a lower bound on Om_1 and
               which needs no screening, no fit and no solve;
               threshold='omega1' tests against a caller-supplied Om_1, which
               is the condition itself and is tighter.
    frontier   [nocc - half_width, nocc + half_width), widened until no
               degenerate block is split. Inside a degenerate subspace the
               eigenvectors are not determined, so giving one member of a pair
               a quasiparticle energy and leaving the other at the mean field
               differentiates across an arbitrary rotation.
    valence    every valence occupied orbital plus extra_virtuals virtuals,
               optionally filtered on pole strength Z > QP_WINDOW_Z_MIN. The
               core is read off the molecule, so the window is chemical rather
               than a count of orbitals.
    all        every orbital, which is the reference the other three are
               approximations to.

THE SET IS DECIDED ONCE, AT THE REFERENCE GEOMETRY, AND FROZEN. Membership is
a discrete function of the mean-field spectrum, so a set re-decided at each
geometry steps whenever an orbital crosses the criterion -- a state at reach
1.04 at one geometry and 0.87 at the next would be solved in one and shifted in
the other, and the energy difference between the two treatments lands in the
surface as a discontinuity with no derivative. For the same reason `admitted`
takes omega_p = eps_p: the root is not known before the solve, and a criterion
evaluated on the converged root would depend on where the iteration started.

The orbitals outside the set carry a SCISSOR: one frozen shift per state,
calibrated at the reference geometry on the explicitly solved roots
(`calibrate_scissor`) and spent unchanged at every displaced geometry
(`scissor_route`). The shift is a number, not a functional, so it contributes
nothing to the force beyond moving the diagonal -- which is what makes it safe
for the deep states, whose leverage on a frontier quasiparticle is 0.05 meV per
20 eV for the core and 3.6-10 meV per eV for the inner valence.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from pyscf.data.elements import chemcore

from src.Base.constants import QP_WINDOW_Z_MIN
from src.Base.declaration import QPStates
from src.SingleReference.GW.contour_deformation import residue_set
from src.SingleReference.GW.real_screening import ov_energies
from src.SingleReference.GW.sum_over_poles import compressible


@dataclass(frozen=True)
class ResolvedQPStates:
    """A `QPStates` declaration turned into orbital indices at one spectrum.

    explicit: the orbitals solved explicitly, sorted.
    outside:  the orbitals carrying the scissor; with `explicit` a partition of
              range(norb).
    reach:    kind='admitted' only, and for EVERY orbital, not just the
              excluded ones: the worst swept pole in units of the threshold, so
              a record shows why an orbital was left out and by how much.
    """
    explicit: Tuple[int, ...]
    outside: Tuple[int, ...]
    spec: QPStates
    reach: Dict[int, float] = field(default_factory=dict, hash=False)
    label: str = ''


def admitted_by_pole_condition(eps, nocc, p, limit):
    """(ok, reach) of Eq. (27) for orbital p against a caller-supplied Om_1.

    `compressible` is this test with limit fixed to the particle-hole gap, the
    dRPA lower bound on Om_1 that needs no solve. Passing the lowest dRPA root
    instead tests the condition as it is written, which is tighter: the gap
    admits states the true Om_1 does not.

    reach: the worst swept pole in units of `limit`. ok is reach < 1.
    """
    omega = float(eps[int(p)])
    swept = residue_set(eps, nocc, omega)
    reach = max((abs(eps[q] - omega) for q, _ in swept), default=0.0) / float(limit)
    return bool(reach < 1.0), float(reach)


def particle_hole_gap(eps, nocc):
    """E_g = min(eps_a - eps_i), the threshold `compressible` tests against."""
    return float(ov_energies(eps, nocc).min())


def frontier_qp_states(eps, nocc, half_width, degeneracy_tol):
    """The frontier quasiparticle set, widened so no degenerate block is split.

    Inside a degenerate subspace the eigenvectors are not determined, so an
    individual orbital is not a smooth function of the geometry; giving one
    half of a pair a quasiparticle energy and leaving the other at the mean
    field differentiates across an arbitrary rotation.
    """
    if str(half_width).lower() == 'all':
        return np.arange(len(eps))
    w = int(half_width)
    lo, hi = max(nocc - w, 0), min(nocc + w, len(eps))
    while lo > 0 and eps[lo] - eps[lo - 1] < degeneracy_tol:
        lo -= 1
    while hi < len(eps) and eps[hi] - eps[hi - 1] < degeneracy_tol:
        hi += 1
    return np.arange(lo, hi)


def valence_qp_states(mol, norb, nocc, extra_virtuals=10, filter_z=False,
                      z=None):
    """The valence quasiparticle window: all valence occupied, nocc+extra_virtuals virtual.

    The core comes from the molecule (`chemcore`), so the window is the same
    chemical set whatever basis carries it.

    filter_z: keep only the orbitals whose pole strength exceeds
              QP_WINDOW_Z_MIN. A root below it is a satellite, and the state is
              better served by the frozen scissor than by the number the Newton
              converged on.
    """
    ncore = chemcore(mol)
    n_vir = min(norb - nocc, nocc + extra_virtuals)
    window = list(range(ncore, nocc)) + list(range(nocc, nocc + n_vir))
    if not filter_z:
        return window
    if z is None:
        raise ValueError(
            'filter_z=True needs the pole strengths: pass z= (an array over '
            'orbitals), or ask for the unfiltered window')
    return [p for p in window if z[p] > QP_WINDOW_Z_MIN]


def resolve_qp_states(spec: QPStates, eps, nocc, *, degeneracy_tol,
                      mol=None, omega1: Optional[float] = None,
                      z=None) -> ResolvedQPStates:
    """Turn a `QPStates` declaration into the orbital indices one spectrum gives it.

    This is the ONE place a declaration becomes a set: two routes that resolve
    the same spec here cannot disagree about which orbitals were solved, which
    is what makes their energies differenceable.

    degeneracy_tol: required, and only read by kind='frontier'. A surface's own
                    degeneracy tolerance is a frozen convention of that
                    surface, so there is no default here to inherit silently.
    omega1:         the lowest neutral dRPA excitation, required by
                    threshold='omega1'.
    mol/z:          the molecule kind='valence' reads its core off, and the
                    pole strengths its Z filter reads.
    """
    eps = np.asarray(eps, float)
    norb = len(eps)
    nocc = int(nocc)
    reach: Dict[int, float] = {}
    if spec.kind == 'admitted':
        if spec.threshold == 'gap':
            tested = [compressible(float(eps[p]), eps, nocc)
                      for p in range(norb)]
        else:
            if omega1 is None:
                raise ValueError(
                    "QPStates(threshold='omega1') needs the lowest dRPA "
                    "excitation: pass omega1= (in Hartree). Only "
                    "threshold='gap' can be resolved from the spectrum alone")
            tested = [admitted_by_pole_condition(eps, nocc, p, omega1)
                      for p in range(norb)]
        reach = {p: r for p, (_, r) in enumerate(tested)}
        explicit = [p for p, (ok, _) in enumerate(tested) if ok]
        name = f'admitted({spec.threshold})'
    elif spec.kind == 'frontier':
        explicit = frontier_qp_states(eps, nocc, spec.half_width,
                                      degeneracy_tol)
        name = f'frontier(half_width={spec.half_width})'
    elif spec.kind == 'valence':
        if mol is None:
            raise ValueError(
                "QPStates(kind='valence') reads its core off the molecule: "
                'pass mol=')
        explicit = valence_qp_states(mol, norb, nocc, spec.extra_virtuals,
                                     spec.filter_z, z)
        name = f'valence(+{spec.extra_virtuals} virtuals)'
        if spec.filter_z:
            name += f', Z > {QP_WINDOW_Z_MIN}'
    else:
        explicit = range(norb)
        name = 'all'
    explicit = tuple(sorted(int(p) for p in explicit))
    outside = tuple(p for p in range(norb) if p not in set(explicit))
    return ResolvedQPStates(
        explicit=explicit, outside=outside, spec=spec, reach=reach,
        label=f'{name}: {len(explicit)} of {norb} orbitals explicit')


def frozen_scissor(scissor, p):
    """The frozen quasiparticle shift for orbital p, or None if it has none.

    A scalar applies to every state the pole model cannot carry; a mapping
    applies per orbital, and an orbital absent from it has no shift.
    """
    if scissor is None or isinstance(scissor, str):
        # A string is a REQUEST to calibrate, carried until the reference
        # geometry has solved the excluded states and turned it into a mapping.
        return None
    if isinstance(scissor, Mapping):
        got = scissor.get(int(p))
        return None if got is None else float(got)
    return float(scissor)


def scissor_route(scissor, p, eps, nocc, start):
    """(route, shift) for orbital p: a frozen shift, or the pole model.

    A MAPPING is the frozen tier assignment and is authoritative -- a state in
    it is shifted whatever its reach, so the tier cannot move between
    geometries. That matters at the boundary: benzene's orbital 9 reads reach
    1.04 at the Newton start and 0.87 at its converged root, because the
    quasiparticle correction lifts it 2.2 eV, so a per-geometry test puts it on
    either side depending on where the iteration began.

    A scalar is auto-detected against `compressible` at the start instead,
    which is convenient and only safe when no state sits near reach = 1.
    """
    shift = frozen_scissor(scissor, p)
    if isinstance(scissor, str):
        return 'sop', None
    if isinstance(scissor, Mapping):
        return ('scissor', shift) if shift is not None else ('sop', None)
    if shift is not None and not compressible(start, eps, nocc)[0]:
        return 'scissor', shift
    return 'sop', None


def calibrate_scissor(eps, nocc, roots, excluded):
    """Frozen shifts for the states Eq. (27) excludes, from explicit roots.

    `roots` maps a few probe orbitals to quasiparticle energies solved the
    expensive way at the reference geometry; every excluded orbital takes the
    shift of the probe nearest it in orbital energy. Two tiers are enough
    because the sensitivity splits that way: a 20 eV error on a core state
    moves a frontier quasiparticle by 0.05 meV, while the inner valence costs
    3.6-10 meV per eV, so the inner valence is what the probes must bracket.

    The shifts are frozen here and never re-derived per geometry: a tier
    boundary re-decided at a displaced geometry puts a step in the surface for
    the same reason a per-geometry residue set does.
    """
    eps = np.asarray(eps, float)
    probes = sorted(roots)
    if not probes:
        raise ValueError('a scissor needs at least one explicitly solved root')
    deltas = {q: float(roots[q]) - eps[q] for q in probes}
    out = {}
    for p in excluded:
        near = min(probes, key=lambda q: abs(eps[q] - eps[int(p)]))
        out[int(p)] = deltas[near]
    return out
