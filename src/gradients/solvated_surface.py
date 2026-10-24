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
fresh state model. Between geometries the surface carries the last converged
charges as the next loop's start when the cavity keeps its number of sites,
and an excited root by the overlap of its AO transition amplitudes
(state.y_follow, eeADC.ee_follow.FollowedRoot), so a walk stays on one state.
A loop that does not converge warns and returns its last cycle
(solvation_loop.SolvationNotConverged); every evaluation whose loop, state
model or force carried a numerical flag is logged in `flags` ({'coords':
Bohr, 'flags': names, 'messages': texts} per evaluation), and `flags_at(mol)`
returns the names logged at one geometry, so a record built on a walk's last
geometry says which caveats it stands on.

A state model whose surface is fixed by an object it carries (`surface_key`,
a GW-BSE model's chain: gradients.gwbse.state_model) must stay on that one
surface: make_state then moves the first model
(`reference.at(mol, make_scf, field.transition_environment())`)
and a model on another surface is refused before its loop runs. The walkers'
accepted steps (gradients.surface.commit) reach the model's own
`commit(mol)`, which moves a chain's conventions and followed state there.

Refreeze (`refreeze(mol)`, the outer loop of `optimize.relax(...,
refreeze=1)`): a model on a surface of its own is rebuilt at `mol` by its
`refrozen(mol, lr_environment)` (a GW-BSE model: a fresh chain, built at
`mol` in the zero-charge mean field, exactly as the model of a vertical
energy there is), and the new surface moves that model to every later
geometry. It starts its first loop from no charges, as a vertical's loop
does, and carries the followed root (ADC(2)'s `y_follow`) over. A surface whose model
carries no `surface_key` -- HF, MP2 and ADC(2), which freeze nothing (an
ADC(2) root followed by overlap is matched at every geometry, not frozen),
and the GW-BSE reference -- is its own refreeze.
"""
from src.environments import solvation_loop as sl
from src.gradients.surface import PotentialEnergySurface


class SolvatedSurface(PotentialEnergySurface):
    """See the module docstring. loop_kw go to solvation_loop.equilibrate."""

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'mean_field'})

    def __init__(self, mol0, make_field, make_state, label='solvated state',
                 **loop_kw):
        self.mol0 = mol0
        self.make_field, self.make_state = make_field, make_state
        self._label = label
        self.loop_kw = loop_kw
        self._q = None
        self.y_follow = None
        self.last = None
        self._surface_key = None
        self.flags = []

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

    def _solve(self, mol):
        mol = self.mol0 if mol is None else mol
        field = self.make_field(mol)
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
        q0 = self._q if (self._q is not None and self._q.size == field.nsites) else None
        r = sl.equilibrate(state, field, q0=q0, **self.loop_kw)
        self._log_flags(mol, r.flags)
        self._q = r.q
        if getattr(state, 'y_follow', None) is not None:
            self.y_follow = state.y_follow
        if self._surface_key is None:
            self._surface_key = getattr(state, 'surface_key', None)
        self.last = dict(mol=mol, field=field, state=state, result=r)
        return field, state, r

    def total_energy(self, mol=None, mf=None):
        return self._solve(mol)[2].energy

    def total_gradient(self, mol=None, mf=None):
        field, state, r = self._solve(mol)
        flags = []
        F = sl.force(state, field, r, flags)
        self._log_flags(self.last['mol'], flags)
        if getattr(state, 'y_follow', None) is not None:
            self.y_follow = state.y_follow
        info = dict(r.info or {})
        diag = dict(cycles=len(r.history), polarization=r.polarization)
        if 'omega' in info:
            diag['omega'] = info['omega']
        return F, r.energy, diag

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
                              label=self._label, **self.loop_kw)
        out.y_follow = self.y_follow
        # one flag log for the walk and its refreezes (flags_at reads both)
        out.flags = self.flags
        return out

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        return self._label
