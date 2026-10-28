"""The GW-BSE states of the self-consistent state-specific solvation loop
(scheme B), as the photophysics drivers build them: one surface per state,
and the vertical energies of one state at one geometry.

    make_state = gwbse_states(make_scf, numerics, bse_tda)  # or with
                                        # roots=, build_chain=
    build = make_state('singlet')       # one surface
    model = build(mol, field)           # the first model, its chain built
                                        # on its first call
    model = build(mol2, field2)         # the same chain at other nuclei

    r = vertical(make_field, make_state, 'singlet', mol, loop_kw)

`make_state(kind)` ('S0', 'singlet' or 'triplet') returns `build(mol,
field)`, the state model of `src.gradients.gwbse.state_model` at `mol` in the
medium `field` (`src.environments.reaction_field`): the reference for 'S0'
(it freezes nothing), else the root `roots[kind]` of the spin (zero-based,
the lowest by default), on the chain `build_chain` builds where it is given
(`GWBSEState`). Each call of
`make_state` is one surface: its first model builds the chain on its first
evaluation and every later model is that one moved (`GWBSEState.at`), so a
walk (`src.gradients.solvated_surface.SolvatedSurface`) stays on the
conventions frozen at its first geometry. A singlet carries the
linear-response term of the field's `transition_environment()`, a triplet
none.

`vertical` takes the absorption and emission of one state at one geometry
(`solvation_loop.vertical_energies`): both loops of the state stand on one
surface, built and frozen there, so each vertical energy's model is a chain
of its own geometry. Its result holds the medium (`field`) and the model of
each loop (`models`: 'eq_ground', 'eq_state', 'neq_state', 'neq_ground'),
whose mean field is the one converged in that loop's last charges; `ground`
and `state` are the first two (the state's holds the chain).

`walk_vs_vertical` compares G_eq of a state at a walk's minimum, on the
walk's surface, with the vertical's there, and flags a difference above its
bar.

`python -m src.photophysics solvated` (`solvated`) is the driver of these
states.
"""
import warnings

from src.Base.numerics.constants import (HARTREE_TO_MEV, REFREEZE_TOL_MEV,
                                         WALK_VERTICAL_G_FLAG_FACTOR)
from src.environments import solvation_loop as sl
from src.gradients.gwbse.state_model import GWBSEState


def gwbse_states(make_scf, numerics, bse_tda, roots=None, build_chain=None):
    """make_state(kind) of GW-BSE (module docstring): each call is one
    surface, its first model built at the first geometry and moved to every
    later one. make_scf(mol): the reference's configured, unconverged mean
    field; numerics: the one `Numerics` every model holds; bse_tda: the
    Tamm-Dancoff kernel; roots: {'singlet': k, 'triplet': k}, the zero-based
    root of each spin (None: the lowest); build_chain: the chain's builder
    (`GWBSEState`), None for its own."""
    roots = dict({'singlet': 0, 'triplet': 0}, **(roots or {}))

    def make_state(kind):
        first = []

        def build(m, field):
            if kind == 'S0':
                return GWBSEState(m, make_scf, numerics=numerics)
            lr = field.transition_environment() if kind == 'singlet' else None
            if first:
                return first[0].at(m, make_scf, lr)
            first.append(GWBSEState(m, make_scf, kind, roots[kind],
                                    bse_tda=bse_tda, lr_environment=lr,
                                    numerics=numerics, build_chain=build_chain))
            return first[0]
        return build
    return make_state


def vertical(make_field, make_state, kind, m, loop_kw):
    """Vertical S0 -> kind energies of m in the slow charges of S0 (and the reverse).
    Both loops of the state stand on one surface: its second model is the first's moved
    to the same nuclei (a GW-BSE chain is built and its conventions frozen once).
    The result holds the medium and the first model of each (module docstring)."""
    f = make_field(m)
    build = make_state(kind)
    held = {'ground': [], 'state': []}

    def ground():
        held['ground'].append(make_state('S0')(m, f))
        return held['ground'][-1]

    def state():
        held['state'].append(build(m, f))
        return held['state'][-1]
    r = sl.vertical_energies(ground, state, f, **loop_kw)
    # the loops run in the order eq_ground, eq_state, neq_state, neq_ground
    models = dict(eq_ground=held['ground'][0], eq_state=held['state'][0],
                  neq_state=held['state'][1], neq_ground=held['ground'][1])
    return dict(r, field=f, models=models, ground=models['eq_ground'],
                state=models['eq_state'])


def walk_vs_vertical(kind, g_walk, g_vertical, refrozen, tol_energy):
    """(gap, flagged): G_eq of `kind` at a walk's minimum on the walk's
    surface minus the same on the vertical's; flagged and warned about above
    REFREEZE_TOL_MEV for a refrozen walk (the refreeze loop's bar), above
    WALK_VERTICAL_G_FLAG_FACTOR loop tolerances `tol_energy` for one that was
    not (the policy: refuse inputs, warn numerics)."""
    gap = g_walk - g_vertical
    bar = (REFREEZE_TOL_MEV / HARTREE_TO_MEV if refrozen
           else WALK_VERTICAL_G_FLAG_FACTOR * tol_energy)
    if abs(gap) > bar:
        warnings.warn(f'{kind} at its minimum: G_eq on the walk\'s surface and on '
                      f'the vertical\'s differ by {gap:.2e} Ha (> {bar:.0e}): the two '
                      'surfaces froze their conventions at different geometries',
                      RuntimeWarning, stacklevel=2)
    return gap, abs(gap) > bar
