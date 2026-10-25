"""The GW-BSE chains' own bookkeeping of a state, the `diagnostics`
ingredient (`src.gradients.surface.INGREDIENT_DOCS`) a record reads beside the
energies: what the quasiparticle treatment was, orbital by orbital
(`qp_bookkeeping`), which interpolation grid the factorization holds
(`held_grid`), what building the chain spent its seconds on
(`surface_parts`), the Davidson solves it ran, and its warning logs, each
entry with the key a record merges them by (`chain_diagnostics`). A record
takes these through the ingredient, so a property never reads a chain's
internals and a method without them simply provides no diagnostics.
"""
import numpy as np

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.factorization.separable_ri import grid_points_per_atom
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.gradients.surface import driven_chain, followed_root, provides


def gw_qp_energies(**route):
    """The G0W0 removal and attachment energies `characters
    .quasiparticle_order` reads: energies(mf, states, spin_channel) ->
    ({orbital: eV}, the reference's diagnostics or None) through
    `calc_qp_energy(**route)`, by default the space-time route, whose ISDF
    grid `route` must name (`counts=` or `grid_accuracy=`)."""
    route = dict({'mode': 'space-time'}, **route)

    def energies(mf, states, spin_channel=None):
        kw = dict(route) if spin_channel is None else dict(route, spin_channel=spin_channel)
        seen = {}
        out = calc_qp_energy(mf, selfenergy='GW', df=True, state=list(states),
                             out=seen, **kw)
        if isinstance(out, dict):
            out = [out[p]['GW'] for p in states]
        return (dict(zip(states, (float(e) for e in out))),
                seen.get('reference_diagnostics'))
    return energies


def log_entries(chain):
    """{log: [(key, entry)]}: the chain's warning logs in the order logged,
    each entry with the key one record merges them by -- quasiparticle-
    Hamiltonian groups kept with Z outside (0, 1] or short of
    self-consistency by kind, group and geometry
    (`ExcitedStateChain.qp_hamiltonian_log`); Casida solves whose read roots
    were left above read_tol or unconfirmed by kind, roots, spin, state and
    geometry (`casida_log`); quasiparticles kept on their followed branch by
    orbital and geometry (`qp_branch_log`); named flags a forward raised by
    flag and message (`numerical_flag_log`)."""
    return {
        'qp_hamiltonian_warnings': [
            ((e['kind'], tuple(e['orbitals']), e['geometry']), e)
            for e in (getattr(chain, 'qp_hamiltonian_log', None) or [])],
        'casida_warnings': [
            ((e['kind'], tuple(e['roots']), e['spin'], e['state'],
              e['geometry']), e)
            for e in (getattr(chain, 'casida_log', None) or [])],
        'qp_branch_warnings': [
            ((e['orbital'], e['geometry']), e)
            for e in (getattr(chain, 'qp_branch_log', None) or [])],
        'numerical_flag_warnings': [
            ((e['flag'], e['message']), e)
            for e in (getattr(chain, 'numerical_flag_log', None) or [])]}


def chain_diagnostics(chain, mol, mf=None, posteriori=False):
    """The `diagnostics` ingredient of a GW-BSE chain at `mol` (module
    docstring): 'qp_bookkeeping', 'held_grid', 'surface_parts',
    'davidson_solves' (None for a chain that keeps none) and 'logs'
    (`log_entries`); posteriori=True adds 'adaptive_check', the adaptive
    explicit set's check at `mol` (`adaptive_check`, a forward there), taken
    first so that the logs hold what its forward logged."""
    out = {}
    if posteriori:
        out['adaptive_check'] = adaptive_check(chain, mol, mf)
    solves = getattr(chain, 'davidson_solves', None)
    out.update({'qp_bookkeeping': qp_bookkeeping(chain, mol),
                'held_grid': held_grid(chain),
                'surface_parts': surface_parts(chain),
                'davidson_solves': None if solves is None else list(solves),
                'logs': log_entries(chain)})
    return out


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
    out = {'counts': dict(factorization.counts),
           'points_per_atom': grid_points_per_atom(factorization.counts),
           'n_start': factorization.n_start,
           'fit_errors': {el: float(err) for el, err
                          in factorization.fit_errors.items()}}
    failed = getattr(factorization, 'fit_failed', None)
    if failed:
        # the warned-about failed fits, as a named flag
        out['fit_failed'] = list(failed)
    return out


def scissor_tiers(shifts, probes):
    """The frozen outside shifts grouped by the explicit root that supplied them.

    `probes` is the frozen map {orbital: the explicit orbitals it borrows
    from}: one probe for an orbital `calibrate_scissor` gave the nearest
    probe's shift, a degenerate block for a hole lent the block's mean. The
    orbitals sharing one shift are one tier. Shifts in eV.
    """
    by_shift = {}
    for p, shift in shifts.items():
        by_shift.setdefault(float(shift), []).append(int(p))
    tiers = []
    for shift, orbitals in sorted(by_shift.items()):
        lent = (probes or {}).get(min(orbitals))
        probe = (None if lent is None else int(lent[0]) if len(lent) == 1
                 else [int(q) for q in lent])
        tiers.append({'probe': probe, 'shift_eV': shift * HARTREE_TO_EV,
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
    view = chain.resolved
    routes = {}
    for p in (int(q) for q in view.qp_set):
        if p in taken:
            routes[p] = taken[p]
        elif p in view.sop_poles:
            routes[p] = 'sop'
        elif p in view.scissor_map:
            routes[p] = 'scissor'
        else:
            routes[p] = chain.residue_route
    return routes


def subspaces_at(chain, frozen, mol):
    """The frozen quasiparticle-Hamiltonian record, with `dropped_here`: the
    members that left their groups at `mol` when it is the geometry the
    chain evaluated last and some did."""
    record = getattr(frozen, 'qp_subspace_record', None)
    here = getattr(chain, 'qp_subspace_drops', None)
    if (not here or not here['dropped']
            or not np.array_equal(here['coords'], mol.atom_coords())):
        return record
    return dict(record or {}, dropped_here=here['dropped'])


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

    `z_outside`: the explicit orbitals whose reference root has a pole
    strength outside (0, 1], kept as solved, with the reason, the route and
    the root and Z.

    Shifts and roots are in eV; `z` is the pole strength per explicit state,
    None until the chain keeps it; `nfreq_cd` (cubic only) the
    contour-deformation quadrature the chain settled on, which the numerics
    record as asked. `blocks`: the orbitals sharing one correction (members,
    energy and correction spreads in Ha) and the candidate blocks rejected,
    with why; None before the cubic chain's first forward. `subspaces`
    (qp_hamiltonian only): the subspaces the quasiparticle Hamiltonian
    couples, the ones carried from the surface refrozen from, and `dropped`,
    each member that took no pole model at this geometry, with the subspace
    it left, its route and whether it was carried, and `groups`, each group's
    Hamiltonian at the last forward (poles, pole strengths, Newton steps);
    None otherwise. `conventions` (cubic only): `FrozenConventions.record`,
    the groups, the holes promoted into them, and the keys' moves.
    """
    chain = driven_chain(surface)
    # the cubic chain's set is a frozen convention, the dense surface's its own
    cubic = provides(chain, 'conventions')
    if not cubic and not provides(chain, 'qp_set'):
        return None
    view, frozen = ((chain.resolved, chain.conventions) if cubic
                    else (chain, chain))
    norb = int(len(chain.mf0.mo_energy)) if cubic else int(mol.nao)
    explicit = sorted(int(p) for p in view.qp_set)
    outside = [p for p in range(norb) if p not in set(explicit)]
    roots = {int(p): float(w) for p, w in
             (view.qp_seeds if cubic else chain.seeds).items()}
    record = {'kind': 'cubic' if cubic else 'dense', 'n_orbitals': norb,
              'explicit': explicit, 'n_explicit': len(explicit),
              'outside': outside, 'n_outside': len(outside),
              'qp_root_eV': {p: w * HARTREE_TO_EV for p, w in roots.items()},
              'blocks': getattr(frozen, 'qp_block_record', None),
              'subspaces': subspaces_at(chain, frozen, mol)}
    if not cubic:
        record.update(outside_treatment='mean-field', n_scissor=0,
                      scissor_tiers=[], route=None, inside_scissor_eV={},
                      sop=None, z_outside={},
                      z={int(p): float(v) for p, v in chain.qp_z.items()})
        return record
    eps = np.asarray(chain.mf0.mo_energy, float)
    # a charged state's chain never calibrates a scissor: no BSE diagonal
    scissored = view.outside_shift is not None
    shifts = view.outside_shift or {}
    diagnostics = getattr(chain, 'qp_diagnostics', None) or {}
    routes = explicit_routes(chain)
    sop = None
    if chain.residue_route == 'sop':
        poles = {int(p): np.asarray(v, float)
                 for p, v in view.sop_poles.items()}
        sop = {'n_poles_asked': int(chain.n_poles),
               'sop_stride': chain.sop_stride,
               'poles_per_state': {p: int(len(v)) for p, v in poles.items()},
               'poles_hartree': {p: v.tolist() for p, v in poles.items()}}
    record.update(
        outside_treatment='scissor' if scissored else None,
        outside_scissor=getattr(chain, 'outside_scissor', 'tiers'),
        # outside_scissor='curve': the frozen Delta(eps) beyond the window,
        # its knots, probes and the probes it dropped; the tiers then cover
        # only the orbitals off both curves
        scissor_curve=frozen.curve_record(),
        n_scissor=len(outside) if scissored else 0,
        scissor_tiers=scissor_tiers(shifts, view.outside_probe),
        scissor_eV={int(p): float(s) * HARTREE_TO_EV
                    for p, s in sorted(shifts.items())},
        route=routes, n_sop=sum(r == 'sop' for r in routes.values()),
        n_inside_scissor=sum(r == 'scissor' for r in routes.values()),
        inside_scissor_eV={int(p): float(s) * HARTREE_TO_EV
                           for p, s in sorted(view.scissor_map.items())},
        eps_mean_field_eV={p: float(eps[p]) * HARTREE_TO_EV for p in explicit},
        sop=sop, z=diagnostics.get('z'),
        # quasiparticles kept on their followed branch where the largest
        # weight had moved to another root, by orbital and geometry
        qp_branch_warnings=list(getattr(chain, 'qp_branch_log', [])),
        # quasiparticle-Hamiltonian groups kept with Z outside (0, 1] or
        # carried on short of self-consistency, by group and geometry
        # (`QuasiparticleSet.log_qp_hamiltonian`)
        qp_hamiltonian_warnings=list(getattr(chain, 'qp_hamiltonian_log', [])),
        # explicit roots of pole strength outside (0, 1] at the reference
        # (`_flag_z_outside`), kept as solved
        z_outside={int(p): {'reason': d['reason'], 'z': d['z'],
                            'route': d['route'],
                            'root_eV': d['root'] * HARTREE_TO_EV}
                   for p, d in sorted(view.qp_z_outside.items())},
        # the contour-deformation quadrature the first forward settled on,
        # doubled from the asked count where a root sat near an eps_q
        nfreq_cd=int(frozen.nfreq_cd),
        # kind='adaptive': how the explicit set was selected out of the
        # admitted candidates, and what that costs each target state
        adaptive=(chain.selection_record()
                  if view.qp_partition is not None else None),
        # the frozen conventions keyed by orbital identity: the groups, the
        # holes promoted into them and how the keys moved along the walk
        conventions=frozen.record())
    return record


def adaptive_check(surface, mol, mf=None):
    """The adaptive explicit set's check at the end of a walk, or None for
    another set.

    One forward at `mol` on the surface that walked, then one analytic-
    continuation GW there (`ExcitedStateChain.posteriori_check`): the
    first-order budget of the followed state with the partition frozen at
    the reference geometry, its probes' shifts at `mol` and the exact weights
    there. It warns when the budget is exceeded and records it; it neither
    reselects nor reruns the walk.
    """
    chain = driven_chain(surface)
    if (not provides(chain, 'conventions')
            or chain.resolved.qp_partition is None):
        return None
    mol, mf = chain.mean_field(mol, mf)
    solution = chain.forward(mol, mf)[1]
    root = (followed_root(chain, mol) if chain.track is not None
            else int(chain.state))
    return chain.posteriori_check(
        mol, mf, {chain.spin: (solution.omega, solution.x, solution.y)},
        ((chain.spin, root),))
