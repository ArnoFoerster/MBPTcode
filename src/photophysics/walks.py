"""What every walk of the photophysics drivers does around `relax`: its
progress kept in a `Store` after every accepted step, a relaunch that starts
a new walk from the last accepted geometry, and the saddle check at its
minimum. The stages (`stages.Run.walk`, `Run.saddle_checked`) and the
solvated driver (`solvated._minimum`) take these.

    walk_progress     the walk state a progress record holds for a walk from
                      this start, or None; another stamp refuses
                      (`Store.read`), another start is walked again
    restart_from      a new walk from a walk state's last accepted geometry,
                      on its optimizer Hessian, not nudged, recorded as
                      `restarted_from`
    checked_minimum   the saddle check on the surface the walk ended on, its
                      Hessian's forces on the surface's untied twin, and the
                      flags of its verdict
"""
import numpy as np

from src.photophysics.records import log
from src.properties.excitations import (relaxation_flags, saddle_check,
                                        saddle_verdict)


def walk_progress(store, name, stamp, start):
    """The walk state the progress record `name` of `store` holds for a walk
    from `start` under `stamp`, or None where there is none. Another stamp
    refuses the run (`Store.read`); a walk from another start is walked
    again and its progress overwritten."""
    if not store.has(name):
        return None
    held = store.read(name, stamp)
    if not np.array_equal(held['start_bohr'], start.atom_coords()):
        log(f'WARNING {name} is a walk from another start; walking again '
            f'from this one')
        return None
    return held['state']


def progress_writer(store, name, stamp, start):
    """`checkpoint(state)` for `relax`: the walk state after every accepted
    step written as the progress record `name`, with the walk's start."""
    start_bohr = np.array(start.atom_coords(), float)

    def checkpoint(state):
        store.write(name, {'start_bohr': start_bohr, 'state': state}, stamp)
    return checkpoint


def restart_from(state, molecule_at):
    """(molecule, optimizer Hessian or None, restarted_from) of a new walk
    from the last accepted geometry of the walk state `state`;
    `molecule_at(coords)` is the run's molecule at coords (Bohr). The new
    walk is not nudged, and the record's `restarted_from` names the pass,
    the accepted steps and G where it restarted."""
    restarted = {'walk_pass': int(state['pass']),
                 'accepted_steps': len(state['history']),
                 'G': float(state['energy'])}
    hessian = state.get('hessian')
    return (molecule_at(np.asarray(state['coords'], float)),
            None if hessian is None else np.asarray(hessian, float),
            restarted)


def checked_minimum(surface, mol, walk, stage, **check):
    """(minimum, the escape walk's info or None, the check's record, its
    flags) of the saddle check (`excitations.saddle_check`, `check` its
    keywords) at the walk's minimum `mol` on `surface`, the surface the
    walk ended on. The Hessian's forces are taken on the surface's untied
    twin where it has one (`SolvatedSurface.untied`: every loop to its
    tolerances). The flags are the check's verdict (`saddle_verdict`) and,
    after an escape, the escape walk's own (`relaxation_flags` of
    '<stage> escape'), each warned about."""
    untied = getattr(surface, 'untied', None)
    moved, again, record = saddle_check(
        surface if untied is None else untied(), mol, walk, **check)
    flags = saddle_verdict(stage, record)
    if again is not None:
        flags = dict(relaxation_flags(f'{stage} escape', again), **flags)
    return moved, again, record, flags
