"""Self-consistent state-specific solvation, independent of both the medium and
the electronic-structure method.

Two contracts meet here and nowhere else:

    the medium (src.environments.reaction_field.ReactionField): the potential of a
        density on its sites, the induced charges q = Q V (static or fast), and
        the one-electron operator of given charges;
    the state (src.gradients.state_models): called with an external
        one-electron operator h and a constant e, it returns its energy E in
        H_vac + h + e and its orbital-relaxed density D_eff = dE/dh (AO,
        spin-summed). Hartree-Fock, MP2 and ADC(2) states are all the same thing
        to this module.

The functional. With E(q) the state's energy in the field of fixed charges q,

    G(q) = E(q) - (1/2) q^T Q_s^-1 q

is stationary where q = Q_s V_tot(D_eff) (Hellmann-Feynman holds for E(q)
because D_eff is the relaxed density), and there G = E - (1/2) q . V_tot. Each
cycle reports G(q) itself, at the charges q the state was solved in: its
error is second order in q - q*, where E - (1/2) q . V_tot(D(q)) would be
first order. This
is the PTED coupling of Lunkenheimer and Koehn (J. Chem. Theory Comput. 9, 977
(2013)) and Haettig and Pausch (J. Phys. Chem. A 129, 6155 (2025)), with the
relaxed density; its stationarity is what lets a nuclear force add only the
derivative of the medium's own terms.

Non-equilibrium. The charges split into fast (Q_f) and slow (Q_s - Q_f) parts
(the Marcus partition), each answering the solute's potential:

    G(q_f, q_s) = E(q_f + q_s) - (1/2) q_f^T Q_f^-1 q_f - (1/2) q_s^T (Q_s - Q_f)^-1 q_s.

A state B in the slow charges of state A's equilibrium, q_s^A = (Q_s - Q_f) V_A,
held, is stationary in q_f = Q_f V_B:

    G_neq(B | A) = E(q_s^A + q_f) - (1/2) q_f . V_B - (1/2) q_s^A . V_A.

G_neq(A | A) = G_eq(A); with Q_s = Q_f there are no slow charges. At one
geometry,

    absorption   G_neq(X | S0) - G_eq(S0)      (at the S0 minimum)
    emission     G_eq(X) - G_neq(S0 | X)       (at the X minimum)

(Cammi and Mennucci, J. Chem. Phys. 110, 9877 (1999); Cossi and Barone,
J. Chem. Phys. 112, 2427 (2000)). Adiabatic energies are differences of G_eq
at each state's own minimum.

The force of G_eq (stationary in q and in the state's own parameters):

    dG/dR = F_state(q)  +  d/dR [(1/2) V(D_eff)^T Q_s V(D_eff)]|_D_eff
            + sum_LR w d/dR [(1/2) V_e(rho_t)^T Q_f V_e(rho_t)]|_rho_t,

F_state the method's force in the fixed charges with the external operator's
own derivative left out (state.force), the second term the medium's at the
fixed relaxed density, the last the medium's part of a method's LR terms
(Haettig and Pausch Eqs. 40-47). Equilibrium states only: a minimum of G_eq is
the structure an emission or an adiabatic energy starts from.

The loop. q from the state's gas-phase density; then the state in the field of
q, its relaxed density, the medium's answer; DIIS on q; stop when the largest
change of a charge and of G are below their tolerances. A loop that reaches
loop_max_cycle first is a numerical event (the policy: refuse inputs, warn
numerics): it warns (SolvationNotConverged, a RuntimeWarning) and
returns its last cycle -- G at the charges the state was last solved in --
with converged False and the flag 'solvation_loop_not_converged' in its
`flags`, beside the state model's own flags of that cycle (info['flags']); a
force taken on it warns again and is computed there. A state model that
returns no number (a NaN energy: a root the solve did not return,
ee_follow.root_missing) stops the loop at once: the result is NaN, converged
False, with the model's flags and 'solvation_loop_stopped', and its force is
NaN; a walk rejects that step
(properties.optimize.NonFiniteEvaluation). A state model with a linear-response term (its lr_environment) must read
the field's own transition_environment(), the same object: the loop refuses
any other (another cavity or eps_inf would answer the transition density).

Checkpoint. equilibrate(..., checkpoint=path) writes, after every cycle,
what the next cycle starts from (src.Base.utils.record_io, atomic): the charges, the
cycle history and G, the DIIS steps taken (replayed into a fresh DIIS on
resume, so its extrapolation is the same to the bit) and the state model's own
`checkpoint()` (its SCF warm start; a GW-BSE model also the first call that
fixed its surface). A loop started again on the same path resumes after the
last cycle written and gives the uninterrupted run's G, charges and density
bit for bit; a file written for another loop is refused: other nuclei,
medium (`field.identity()`: eps_s, eps_inf, model and the cavity's sites and
areas), state (`state.identity()`: model, method settings, spin, root), slow
charges, start or tolerances. Only rank 0 reads and writes the file (a path
may be node-local); the other ranks take rank 0's. A state model without
checkpoint() / restore() / identity() is refused when a checkpoint is asked
for.

Numerics. The loop's tolerances are those of the `numerics` handed to it, or
else of the state model's own (`state.numerics`, a GW-BSE model's), so one
object is passed once; two different objects are refused.

Over MPI ranks (src.Base.utils.parallel.mpi_grid.current_comm) every rank runs the
loop with its own state and medium objects; the charges, the potential, the
state's energy and relaxed density are rank 0's on every rank each cycle
(lockstep, checked: a broadcast only where the bits differ), the
polarization term too, so every rank takes the same DIIS step and stops at
the same cycle, and the force and the slow charges are rank 0's. Serially the
locksteps are no-ops.
"""
import os
import warnings

import numpy as np
from pyscf import lib

from src.Base.numerics.constants import SOLVATION_LOOP_DIIS_SPACE
from src.Base.numerics.numerics import numerics_or_default
from src.Base.utils.record_io import read_record, write_record
from src.Base.utils.parallel.mpi_grid import broadcast, current_comm, is_root, lockstep

#: The version of the checkpoint a loop writes (module docstring).
CHECKPOINT_FORMAT = 1


class SolvationNotConverged(RuntimeWarning):
    """A loop that reached max_cycle with its charges or G still moving more
    than their tolerances (warned, flagged, the last cycle returned)."""


class LoopResult:
    """The last cycle of a loop.

    energy  G (Hartree); state_energy E(q); polarization the term subtracted
            ((1/2) q^T Q_s^-1 q, or (1/2) q_f^T Q_f^-1 q_f + c_s out of
            equilibrium)
    q       the induced charges; dm_eff the state's relaxed density (AO)
    info    what the state returned for this cycle (method-specific)
    history [(cycle, G, max |dq|)]; converged False for a loop that reached
            max_cycle (module docstring)
    flags   the numerical events G is built on, {'flag': name, 'message':
            text} each: the state model's of the last cycle and
            'solvation_loop_not_converged' or 'solvation_loop_stopped'
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def slow_charges(field, dm_eff):
    """(q_s, c_s) of an equilibrium with relaxed density dm_eff: the slow
    charges (Q_s - Q_f) V_tot and the constant (1/2) q_s . V_tot that G_neq of
    another state in them carries."""
    V = field.potential(dm_eff)
    q_s = field.charges(V, 'static') - field.charges(V, 'fast')
    q_s, V = lockstep((q_s, V), check=True)
    return q_s, 0.5 * float(q_s @ V)


def polarization(field, q, slow=None):
    """The quadratic term of G at the charges q: (1/2) q^T Q_s^-1 q, or with
    slow = (q_s, c_s) (1/2) q_f^T Q_f^-1 q_f + c_s, q_f = q - q_s."""
    if slow is None:
        return 0.5 * float(q @ field.response_solve(q, 'static'))
    q_f = q - slow[0]
    return 0.5 * float(q_f @ field.response_solve(q_f, 'fast')) + slow[1]


def free_energy(state, field, q, slow=None):
    """G(q) of the module docstring at any charges q: the state's energy in
    the charges q minus the quadratic polarization term; with slow = (q_s,
    c_s), q is the total and only q - q_s is free. What each loop cycle
    reports, there with the energy of that cycle's state call."""
    h, e = field.operator(q)
    return state(h, e).energy - polarization(field, q, slow)


def require_field_lr(state, field):
    """Refuse a state whose linear-response term reads a continuum other
    than the field's own (module docstring)."""
    lr = getattr(state, 'lr_environment', None)
    if lr is not None and lr is not field.transition_environment():
        raise ValueError('the state\'s lr_environment is not its field\'s '
                         'transition_environment(): its linear-response term would '
                         'answer on another cavity or eps_inf than the loop\'s '
                         'charges; build the state with field.transition_environment()')


def _identity(state, field, slow, q0, tol_q, tol_energy):
    """What a checkpoint belongs to: the nuclei, the cavity's size, the slow
    charges, the start asked for and the tolerances."""
    mol = state.mol
    return dict(coords=np.asarray(mol.atom_coords(), float),
                charges=np.asarray(mol.atom_charges()), nsites=int(field.nsites),
                medium=field.identity(), state=state.identity(),
                slow=None if slow is None else (np.asarray(slow[0], float),
                                                float(slow[1])),
                q0=None if q0 is None else np.asarray(q0, float),
                tol=(float(tol_q), float(tol_energy)))


def _same(a, b):
    """Two identity trees equal to the bit."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)):
        return (isinstance(b, (tuple, list)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def _resume(path, state, identity):
    """The checkpoint at `path` for this loop, or None when there is none;
    the state model takes its part back (module docstring)."""
    saved = read_record(path) if is_root() and os.path.exists(path) else None
    saved = broadcast(saved, current_comm())
    if saved is None:
        return None
    if saved.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{path}: checkpoint format {saved.get("format")!r}, '
                         f'this loop reads {CHECKPOINT_FORMAT}')
    if not _same(saved['identity'], identity):
        raise ValueError(f'{path} was written for another loop (geometry, cavity, '
                         'slow charges, start or tolerances): resume it with its '
                         'own settings or write to another path')
    state.restore(saved['state'])
    return saved


def loop_numerics(state, numerics):
    """The Numerics a loop reads: the one handed to it, else the state
    model's own (module docstring); two different objects are refused."""
    own = getattr(state, 'numerics', None)
    if numerics is None:
        return own
    if own is not None and own is not numerics and own != numerics:
        raise ValueError('the loop was handed a Numerics other than its state '
                         "model's: pass one object, once (the model's)")
    return numerics


def equilibrate(state, field, slow=None, q0=None, numerics=None, log=None,
                checkpoint=None):
    """The loop of the module docstring: G_eq of `state` in `field`, or with
    slow = (q_s, c_s) (slow_charges of another state) G_neq. q0: starting
    charges (default: the answer to the state's gas-phase density).
    numerics: a `src.Base.numerics.numerics.Numerics`, whose solvent section holds the
    loop's tolerances loop_tol_q, loop_tol_energy and its loop_max_cycle; None
    takes the state model's own, else the defaults (`loop_numerics`). checkpoint: a path (JSON, its arrays
    beside it) written after every cycle and resumed from when present
    (module docstring)."""
    numerics = loop_numerics(state, numerics)
    loop = numerics_or_default(numerics).solvent
    tol_q, tol_energy, max_cycle = (loop.loop_tol_q, loop.loop_tol_energy,
                                    loop.loop_max_cycle)
    say = log or (lambda *a: None)
    if max_cycle < 1:
        raise ValueError(f'max_cycle={max_cycle}; the loop needs at least one cycle')
    require_field_lr(state, field)
    if checkpoint is not None and not all(hasattr(state, name) for name in (
            'checkpoint', 'restore', 'identity')):
        raise ValueError(f'{type(state).__name__} has no checkpoint() / restore() / '
                         'identity(): its loop cannot be resumed; run it without '
                         'checkpoint')

    def answer(dm):
        V = field.potential(dm)
        if slow is None:
            return field.charges(V, 'static'), V
        return slow[0] + field.charges(V, 'fast'), V

    identity = (None if checkpoint is None
                else _identity(state, field, slow, q0, tol_q, tol_energy))
    saved = None if checkpoint is None else _resume(checkpoint, state, identity)
    diis = lib.diis.DIIS()
    diis.space = SOLVATION_LOOP_DIIS_SPACE
    if saved is None:
        if q0 is None:
            r = state(*field.operator(np.zeros(field.nsites)))
            if not np.isfinite(lockstep(r.energy)):
                return _stopped(r, np.zeros(field.nsites), [], 0)
            q0 = answer(r.dm_eff)[0]
        q = lockstep(np.array(q0, float), check=True)
        history, G_old, steps, start = [], None, [], 1
    else:
        q, G_old = saved['q'], saved['G_old']
        history, steps = list(saved['history']), list(saved['steps'])
        for x, err in steps:
            diis.update(x, xerr=err)
        start = history[-1][0] + 1
    dq, dG = (history[-1][2] if history else float('nan')), None
    last = None
    for cycle in range(start, max_cycle + 1):
        r = state(*field.operator(q))
        q_new, V = answer(r.dm_eff)
        q_new, V, dm_eff, E = lockstep((q_new, V, np.array(r.dm_eff), r.energy),
                                       check=True)
        r.dm_eff, r.energy = dm_eff, E
        if not np.isfinite(E):
            return _stopped(r, q, history, cycle)
        pol = lockstep(polarization(field, q, slow))
        G = r.energy - pol
        dq = float(np.max(np.abs(q_new - q))) if q.size else 0.0
        dG = None if G_old is None else abs(G - G_old)
        history.append((cycle, G, dq))
        say(f'solvation cycle {cycle}: G {G:.12f}  max|dq| {dq:.2e}'
            + ('' if dG is None else f'  |dG| {dG:.2e}'))
        if dq < tol_q and dG is not None and dG < tol_energy:
            return LoopResult(energy=G, state_energy=r.energy,
                              polarization=pol, q=q, dm_eff=r.dm_eff,
                              info=r.info, history=history, converged=True,
                              flags=list((r.info or {}).get('flags', [])))
        last = (r, q, pol, G)
        G_old = G
        if q.size:
            # every step, not the last DIIS space: replayed alone they sit in
            # other slots of DIIS's ring and the extrapolation's bits move
            steps.append((q_new, q_new - q))
            q = diis.update(q_new, xerr=q_new - q)
        else:
            q = q_new
        if checkpoint is not None and is_root():
            write_record(checkpoint, dict(
                format=CHECKPOINT_FORMAT, identity=identity, q=q, G_old=G_old,
                history=history, steps=steps, state=state.checkpoint()))
    if last is None:
        # resumed from a checkpoint already at max_cycle: its charges, solved once
        r = state(*field.operator(q))
        r.dm_eff, r.energy = lockstep((np.array(r.dm_eff), r.energy), check=True)
        pol = lockstep(polarization(field, q, slow))
        last = (r, q, pol, r.energy - pol)
    r, q, pol, G = last
    dG_txt = float('nan') if dG is None else dG
    text = (f'the solvation loop did not converge in {max_cycle} cycles: last '
            f'max|dq| {dq:.2e} (tol_q {tol_q:.0e}), |dG| {dG_txt:.2e} Ha '
            f'(tol_energy {tol_energy:.0e}); its last cycle is returned')
    warnings.warn(text, SolvationNotConverged, stacklevel=2)
    flags = list((r.info or {}).get('flags', [])) + [
        {'flag': 'solvation_loop_not_converged', 'message': text}]
    return LoopResult(energy=G, state_energy=r.energy, polarization=pol, q=q,
                      dm_eff=r.dm_eff, info=r.info, history=history,
                      converged=False, flags=flags)


def _stopped(r, q, history, cycle):
    """The NaN LoopResult of a state model that returned no number at cycle
    `cycle` (module docstring)."""
    text = (f'the solvation loop stopped at cycle {cycle}: the state model '
            'returned no number (NaN energy); G is NaN')
    warnings.warn(text, SolvationNotConverged, stacklevel=3)
    flags = list((r.info or {}).get('flags', [])) + [
        {'flag': 'solvation_loop_stopped', 'message': text}]
    return LoopResult(energy=np.nan, state_energy=np.nan, polarization=np.nan,
                      q=q, dm_eff=r.dm_eff, info=r.info, history=history,
                      converged=False, flags=flags)


def vertical_energies(make_ground, make_state, field, **loop_kw):
    """Absorption and emission of one state at one geometry (module
    docstring). make_ground() / make_state() -> fresh state models (each loop
    warm-starts its own). Returns the four loops, their G, the two vertical
    energies (Hartree), whether every loop converged and, by loop, the names
    of the flags each carries (none for a clean run)."""
    g0 = equilibrate(make_ground(), field, **loop_kw)
    gx = equilibrate(make_state(), field, **loop_kw)
    neq_x = equilibrate(make_state(), field, slow=slow_charges(field, g0.dm_eff),
                        **loop_kw)
    neq_0 = equilibrate(make_ground(), field, slow=slow_charges(field, gx.dm_eff),
                        **loop_kw)
    loops = dict(eq_ground=g0, eq_state=gx, neq_state=neq_x, neq_ground=neq_0)
    return dict(loops=loops, G={k: v.energy for k, v in loops.items()},
                absorption=neq_x.energy - g0.energy,
                emission=gx.energy - neq_0.energy,
                converged=all(v.converged for v in loops.values()),
                # the names of each loop's flags (LoopResult.flags): the
                # caveats the two vertical energies are built on
                flags={k: sorted({f['flag'] for f in v.flags})
                       for k, v in loops.items() if v.flags})


def force(state, field, result, flags=None):
    """dG_eq/dR (natm, 3) at an equilibrium loop `result` (module docstring):
    state.force(h, e) -> (F, info with dm_eff and lr [(w, rho_t)], and
    optionally the force's own flags, which are appended to `flags`). A loop
    that did not converge is warned about again (SolvationNotConverged) and
    its force taken at its last cycle: G is stationary only to the loop's
    residual there. A loop that stopped on a NaN state (`_stopped`) has a NaN
    force, with no solve."""
    if not np.isfinite(result.energy):
        return np.full((state.mol.natm, 3), np.nan)
    if not result.converged:
        warnings.warn('the force of G_eq is taken on a solvation loop that did '
                      'not converge (its last cycle)', SolvationNotConverged,
                      stacklevel=2)
    require_field_lr(state, field)
    F, info = state.force(*field.operator(result.q))
    if flags is not None:
        flags.extend(info.get('flags', []))
    F = F + field.force(info['dm_eff'])
    for w, rho_t in info.get('lr', ()):
        F = F + w * field.transition_force(rho_t)
    return lockstep(np.array(F), check=True)
