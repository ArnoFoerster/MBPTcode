"""An equilibrium-solvated state as a potential-energy surface
(src.gradients.surface.PotentialEnergySurface), for optimizations and walks.

The surface is G_eq of src.environments.solvation_loop: at every geometry a medium
(src.environments.reaction_field) and a state model (src.gradients.state_models,
src.gradients.adc.ee_state_model) are built by the caller's factories, the loop
equilibrates the state with its own relaxed density, and the force is
solvation_loop.force. A minimum of this surface is the solvated structure an
emission (solvation_loop.vertical_energies there) or an adiabatic energy
(differences of G_eq at each state's own minimum) is taken at.

    surface = SolvatedSurface(mol, make_field, make_state, label='S1 toluene')
    mol_min, info = optimize.relax(surface, mol, nudge=0)

make_field(mol) -> a ReactionField around mol; make_state(mol, field) -> a
fresh state model; make_ground(mol, field) -> a fresh S0 model of the same
method (None where the surface's model freezes nothing). Between geometries the surface carries the last converged
charges as the next loop's start when the cavity keeps its number of sites,
and an excited root by the overlap of its AO transition amplitudes
(state.y_follow, eeADC.ee_follow.FollowedRoot), so a walk stays on one state,
and an ADC(2) model's roots as the next geometry's fold start
(state.fold_start). Only these carry over (with the surface key and the flag
log below): `last` (mol, field, state model, loop result) holds the latest
geometry's evaluation, which commit, mean_field and refreeze read, and
`last_ground` (S0's model and loop out of equilibrium, below) the S0 loop the
force reads; both are dropped when the next geometry's evaluation starts, so
a walk holds one step's models at a time.
A loop that does not converge warns and returns its last cycle
(solvation_loop.SolvationNotConverged); every evaluation whose loop, state
model or force carried a numerical flag is logged in `flags` ({'coords':
Bohr, 'flags': names, 'messages': texts} per evaluation), and `flags_at(mol)`
returns the names logged at one geometry, so a record built on a walk's last
geometry says which caveats it stands on.

A state model whose surface is fixed by an object it carries (`surface_key`,
a GW-BSE model's chain: gradients.gwbse.state_model) is built in S0's
equilibrium charges (`solvation_loop`, "Frozen surfaces"): where its loop has
no charges to start from -- the first geometry, the first geometry of a
refrozen surface, a cavity of another number of sites -- the surface runs
S0's loop there first (`make_ground`, the same loop settings) and starts
from its charges; without make_ground the loop refuses a model whose first
call would freeze its surface. Such a model must stay on that one
surface: make_state then moves the first model
(`reference.at(mol, make_scf, field.transition_environment())`)
and a model on another surface is refused before its loop runs. The walkers'
accepted steps (gradients.surface.commit) reach the model's own
`commit(mol)`, which moves a chain's conventions and followed state there.
A walk's converged geometry (gradients.surface.confirm) reaches the model's
own `confirm(mol)` (ADC(2): the fold's full completeness check there); its
flags are logged at that geometry and returned.

Refreeze (`refreeze(mol)`, the outer loop of `optimize.relax(...,
refreeze=1)`): a model on a surface of its own is rebuilt at `mol` by its
`refrozen(mol, lr_environment)` (a GW-BSE model: a fresh chain, built at
`mol` in S0's equilibrium charges there, exactly as the model of a
vertical energy there is), and the new surface moves that model to every
later geometry. It starts its first loop from S0's charges at `mol`, as a
vertical's loop does, and carries the followed root (ADC(2)'s `y_follow`) over. A surface whose model
carries no `surface_key` -- HF, MP2 and ADC(2), which freeze nothing (an
ADC(2) root followed by overlap is matched at every geometry, not frozen),
and the GW-BSE reference -- is its own refreeze.

Out of equilibrium (`noneq=True`, with make_ground(mol, field) -> the
ground state's model): the surface is G_neq(X | S0) of the loop, X in the
slow charges of S0's own equilibrium at every geometry. Each geometry runs
S0's equilibrium loop first (its charges warm-started from the last
geometry's, at the loop's full tolerances), then X's loop in S0's slow
charges (`solvation_loop.slow_charges`), started at a surface's first
geometry from S0's equilibrium charges; the force is
`solvation_loop.force_neq`, which carries the response of S0's loop. The
ground state's model freezes nothing; a refreeze rebuilds X's model only.

A loop tied to the gradient. With `numerics.solvent.loop_force_fraction`
set (f, 1/3 in production) a geometry's loop stops at the first cycle whose
estimated force error is at most f times the current gradient (max |dG/dR|
of the last force this surface took), at least one cycle in
(`solvation_loop.equilibrate`'s stop); the loop is then converged at that
criterion and its force is taken without a warning. The estimate is kappa
times `solvation_loop.residual_force`, the cycle's charge residual's direct
electrostatic force; kappa >= 1 is measured once per surface, at its first
gradient: that loop runs to the full tolerances and the force taken at its
first cycle against the converged one gives the ratio of the true error to
the estimate (the state's own response to the charges is in the first, not
in the second: 5-12 for an ADC(2) singlet, below 1 for Hartree-Fock). The
full tolerances (`loop_tol_q`, `loop_tol_energy`) hold for the first
gradient of every surface (a refreeze's too), for any force below
`final_grad_max` (the walk's convergence bar, so the gradient its final test
reads is a fully converged loop's), and again when the force a tied loop
gave is so much smaller than the gradient the loop was tied to that the
estimate exceeds f times it: the loop is then continued from its charges to
the bar of the new force (or the full tolerances below `final_grad_max`) and
the force taken again. Every gradient's loop is logged in `loop_log`
({'coords', 'mode' ('full', 'calibration' or 'tied'), 'cycles', 'bar',
'estimate', 'kappa'}), one log for a walk and its refreezes. A
finite-difference Hessian takes its forces on `untied()`, the same surface
with every loop to the full tolerances: a displaced force lies above
`final_grad_max` and a tied loop would let it miss by up to f of itself,
which the difference of two displaced forces does not cancel.
"""
import copy

import numpy as np

from src.Base.numerics.numerics import numerics_or_default
from src.environments import solvation_loop as sl
from src.gradients.surface import PotentialEnergySurface


class SolvatedSurface(PotentialEnergySurface):
    """See the module docstring. loop_kw go to solvation_loop.equilibrate;
    final_grad_max (Ha/Bohr): the walk's convergence bar on max |dG/dR|,
    required when the loop is tied to the gradient; make_ground(mol, field):
    the ground state's model (its loop seeds a frozen model's first loop);
    noneq: G_neq(X | S0) in place of G_eq, which needs make_ground."""

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'mean_field'})

    def __init__(self, mol0, make_field, make_state, label='solvated state',
                 make_ground=None, final_grad_max=None, noneq=False,
                 **loop_kw):
        self.mol0 = mol0
        self.make_field, self.make_state = make_field, make_state
        self.make_ground = make_ground
        if noneq and make_ground is None:
            raise ValueError(f'{label}: a surface out of equilibrium needs '
                             'make_ground, the ground state\'s model')
        self.noneq = noneq
        self._q_ground = None
        self.last_ground, self.last_slow = None, None
        self._label = label
        self.final_grad_max = final_grad_max
        self.loop_kw = loop_kw
        # the tie to the gradient (module docstring): max |F| of the last
        # force, kappa, and the first cycle's (estimate, force) it is measured on
        self._scale = None
        self._kappa = None
        self._probe = None
        self.loop_log = []
        self._q = None
        self.y_follow = None
        # an ADC(2) model's last roots, the next geometry's fold start
        self.fold_start = None
        self.last = None
        self._surface_key = None
        self.flags = []
        self._untied = False

    def _log_flags(self, mol, flags):
        if flags:
            self.flags.append(dict(
                coords=mol.atom_coords().tolist(),
                flags=sorted({f['flag'] for f in flags}),
                messages=[f['message'] for f in flags]))

    def flags_at(self, mol):
        """The flag names logged at the nuclei of `mol` (module docstring)."""
        x = mol.atom_coords().tolist()
        return sorted({name for entry in self.flags if entry['coords'] == x
                       for name in entry['flags']})

    def untied(self):
        """A twin of this surface on which every loop runs to its full
        tolerances, for the forces of a finite-difference Hessian (module
        docstring). It shares this surface's flags and loop log; its charges,
        roots and gradient scale move on the twin alone."""
        twin = copy.copy(self)
        twin._untied = True
        return twin

    def _fraction(self, state):
        """f of the tie to the gradient, or None for a loop to its tolerances
        alone (module docstring)."""
        if self._untied:
            return None
        numerics = sl.loop_numerics(state, self.loop_kw.get('numerics'))
        f = numerics_or_default(numerics).solvent.loop_force_fraction
        if f is not None and self.final_grad_max is None:
            raise ValueError(f'{self._label}: a loop tied to the gradient needs the '
                             "walk's convergence bar (final_grad_max), below which "
                             'every loop runs to its full tolerances')
        return f

    def _tied(self, field, bar, seen):
        """The stop criterion of a loop tied to `bar` (Ha/Bohr); `seen` holds
        the last estimate."""
        def stop(cycle, q, q_new, r):
            seen[:] = [self._kappa * sl.residual_force(field, r.dm_eff, q, q_new)]
            return seen[0] <= bar
        return stop

    def _calibrating(self, state, field):
        """A criterion that never stops, and at the first cycle keeps the
        estimate and the force taken there (module docstring)."""
        def stop(cycle, q, q_new, r):
            if cycle == 1:
                est = sl.residual_force(field, r.dm_eff, q, q_new)
                F = self._force(state, field, sl.LoopResult(
                    energy=r.energy, q=q, converged=True), None)
                self._probe = (est, F)
            return False
        return stop

    def _force(self, state, field, r, flags):
        """The force of this surface's G at the loop `r` of `state`."""
        if not self.noneq:
            return sl.force(state, field, r, flags)
        return sl.force_neq(state, self.last_ground['state'], field, r,
                            self.last_ground['result'], flags,
                            self.loop_kw.get('numerics'))

    def _ground_loop(self, mol, field):
        """S0's equilibrium loop at `mol` (module docstring): (slow, q0 for
        X's loop at a surface's first geometry)."""
        model = self.make_ground(mol, field)
        q0 = (self._q_ground if self._q_ground is not None
              and self._q_ground.size == field.nsites else None)
        g = sl.equilibrate(model, field, q0=q0, **self.loop_kw)
        self._log_flags(mol, g.flags)
        self._q_ground = g.q
        self.last_ground = dict(state=model, result=g)
        return sl.slow_charges(field, g.dm_eff), g.q

    def _solve(self, mol, calibrate=False):
        mol = self.mol0 if mol is None else mol
        # the last geometry's models are released before this one's are built
        # (module docstring)
        self.last, self.last_ground = None, None
        field = self.make_field(mol)
        slow, q_ground = (self._ground_loop(mol, field) if self.noneq
                          else (None, None))
        state = self.make_state(mol, field)
        key = getattr(state, 'surface_key', None)
        if self._surface_key is not None and key is not self._surface_key:
            raise ValueError(
                f'{self._label}: make_state returned a model on another surface '
                '(its surface_key is not the first geometry\'s): build the model '
                'once and move it, make_state = lambda m, f: reference.at(\n'
                'm, make_scf, f.transition_environment())')
        if self.y_follow is not None and hasattr(state, 'y_follow'):
            state.y_follow = self.y_follow
        if self.fold_start is not None and hasattr(state, 'fold_start'):
            state.fold_start = self.fold_start
        q0 = (self._q if self._q is not None and self._q.size == field.nsites
              else q_ground)
        if q0 is None and self.make_ground is not None and (
                getattr(state, 'freezes_on_first_call', False)
                or getattr(state, 'surface_key', None) is not None):
            # S0's equilibrium charges here (module docstring)
            seed = sl.equilibrate(self.make_ground(mol, field), field,
                                  **self.loop_kw)
            self._log_flags(mol, seed.flags)
            q0 = seed.q
        f = self._fraction(state)
        stop, tie = None, None
        if f is not None and self._kappa is None and calibrate:
            stop, tie = self._calibrating(state, field), dict(mode='calibration')
        elif f is not None and self._kappa is None:
            tie = dict(mode='full')
        elif f is not None:
            tied = self._scale is not None and self._scale >= self.final_grad_max
            bar = f * self._scale if tied else None
            seen = []
            stop = self._tied(field, bar, seen) if tied else None
            tie = dict(mode='tied' if tied else 'full', bar=bar, seen=seen,
                       fraction=f)
        r = sl.equilibrate(state, field, slow=slow, q0=q0, stop=stop,
                           **self.loop_kw)
        self.last_slow = slow
        self._log_flags(mol, r.flags)
        self._q = r.q
        if getattr(state, 'y_follow', None) is not None:
            self.y_follow = state.y_follow
        if getattr(state, 'fold_start', None) is not None:
            self.fold_start = state.fold_start
        if self._surface_key is None:
            self._surface_key = getattr(state, 'surface_key', None)
        self.last = dict(mol=mol, field=field, state=state, result=r, tie=tie)
        return field, state, r

    def total_energy(self, mol=None, mf=None):
        return self._solve(mol)[2].energy

    def total_gradient(self, mol=None, mf=None):
        field, state, r = self._solve(mol, calibrate=True)
        flags = []
        F = self._force(state, field, r, flags)
        tie = self.last['tie']
        if tie is not None:
            r, F = self._tie_to(F, field, state, r, tie, flags)
        self._log_flags(self.last['mol'], flags)
        if getattr(state, 'y_follow', None) is not None:
            self.y_follow = state.y_follow
        info = dict(r.info or {})
        diag = dict(cycles=len(r.history), polarization=r.polarization)
        if 'omega' in info:
            diag['omega'] = info['omega']
        return F, r.energy, diag

    def _tie_to(self, F, field, state, r, tie, flags):
        """(result, force) of a geometry whose loop is tied to the gradient:
        kappa measured on a calibration loop, a tied loop continued where its
        estimate exceeds f times its own force or that force is below
        final_grad_max, and the gradient scale and the log updated (module
        docstring)."""
        g = float(np.max(np.abs(F)))
        if tie['mode'] == 'calibration' and self._probe is not None:
            est, F1 = self._probe
            self._probe = None
            miss = float(np.max(np.abs(F1 - F)))
            self._kappa = max(1.0, miss / est) if est > 0 else 1.0
            tie = dict(tie, kappa=self._kappa)
        elif tie['mode'] == 'tied' and r.stopped_early:
            final = g < self.final_grad_max
            est = tie['seen'][0]
            if final or est > tie['fraction'] * g:
                bar = None if final else tie['fraction'] * g
                seen = []
                stop = None if final else self._tied(field, bar, seen)
                again = sl.equilibrate(state, field, slow=self.last_slow,
                                       q0=r.q, stop=stop, **self.loop_kw)
                again.history = r.history + again.history
                r = again
                self._q = r.q
                self.last['result'] = r
                F = self._force(state, field, r, flags)
                g = float(np.max(np.abs(F)))
                tie = dict(tie, mode='tied' if stop else 'full', bar=bar,
                           seen=seen or tie['seen'], continued=True)
        self._scale = g
        seen = tie.get('seen') or [None]
        self.loop_log.append(dict(
            coords=self.last['mol'].atom_coords().tolist(),
            mode=('calibration' if tie['mode'] == 'calibration'
                  else 'tied' if r.stopped_early else 'full'),
            cycles=len(r.history), bar=tie.get('bar'),
            estimate=seen[-1] if r.stopped_early else None,
            kappa=self._kappa, continued=bool(tie.get('continued'))))
        return r, F

    def mean_field(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        if self.last is None or self.last['mol'] is not mol:
            self._solve(mol)
        st = self.last['state']
        hf = getattr(getattr(st, 'mp2', st), 'hf', st)
        return mol, hf.mf

    def commit(self, mol):
        """A walk accepted `mol` (gradients.surface.commit): the state model
        built last hears it (module docstring); one that follows nothing has
        no commit."""
        state = None if self.last is None else self.last['state']
        accept = getattr(state, 'commit', None)
        if accept is not None:
            accept(mol)

    def confirm(self, mol):
        """Flags that the walk's converged geometry `mol` is not on the state
        followed (gradients.surface.confirm): the state model's
        `confirm(mol)` on its evaluation at `mol` (made again when the last
        evaluation was elsewhere, as for a walk resumed at its minimum),
        logged at `mol`; empty for a model without one."""
        if self.last is None or not np.allclose(
                self.last['mol'].atom_coords(), mol.atom_coords(), rtol=0, atol=1e-12):
            self._solve(mol)
        check = getattr(self.last['state'], 'confirm', None)
        if check is None:
            return []
        flags = list(check(mol))
        self._log_flags(mol, flags)
        return flags

    def refreeze(self, mol, mf=None):
        """This surface with its model's frozen conventions rebuilt at `mol`
        (module docstring); a surface not yet evaluated is evaluated at `mol`
        first, to learn which model it is on. mf: not read; the loop
        converges its own mean field in its field."""
        if self.last is None:
            self._solve(mol)
        state = self.last['state']
        if getattr(state, 'surface_key', None) is None:
            return self
        built = []

        def make_state(m, field):
            lr = None if state.lr_environment is None else field.transition_environment()
            if built:
                return built[0].at(m, state.make_scf, lr)
            built.append(state.refrozen(m, lr))
            return built[0]
        out = SolvatedSurface(self.mol0, self.make_field, make_state,
                              label=self._label, make_ground=self.make_ground,
                              final_grad_max=self.final_grad_max,
                              noneq=self.noneq, **self.loop_kw)
        out.y_follow = self.y_follow
        # one flag and loop log for the walk and its refreezes (flags_at
        # reads both); kappa is measured again on the new surface
        out.flags = self.flags
        out.loop_log = self.loop_log
        return out

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        return self._label
