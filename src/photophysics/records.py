"""Small pieces every stage shares: state labels, the log line, and what a
walk's record keeps of its optimizer's."""
import time
import warnings

from src.Base.utils.mpi_grid import is_root
from src.properties.excitations import relaxation_fields


class StageRefused(RuntimeError):
    """A stage that cannot run on what the store holds."""


def label(key):
    """'S1', 'T2' for a (spin, root) key, root counted from 0."""
    spin, root = key
    return f'{spin[0].upper()}{int(root) + 1}'


def key_of(name):
    """The (spin, root) key of a label 'S1', 'T2'."""
    return ({'S': 'singlet', 'T': 'triplet'}[name[0]], int(name[1:]) - 1)


def following(spin, root):
    """The (spin, root) key and the one above it, which the second-order
    coupling runs through."""
    return (spin, int(root)), (spin, int(root) + 1)


def log(msg, t0=None):
    """A time-stamped line from rank 0 of the enclosing distribution, with
    the seconds since `t0`."""
    if not is_root():
        return
    print(f'[{time.strftime("%H:%M:%S")}] {msg}'
          + ('' if t0 is None else f'   [{time.time() - t0:.1f} s]'),
          flush=True)


def walk_fields(info):
    """What a walk's record keeps of its optimizer's: no per-cycle Hessian.

    The optimizer's own fields under their one naming
    (`excitations.relaxation_fields`: convergence, cycles, residual, the
    refreeze's drift, move and energy change), and beside them the walk's
    rejected steps, its energy and omega (the last surface's), whether the
    first point was handed over, the first walk's history, the refrozen
    walk's rigid move, the explicit states the rebuild added (grow-only
    adaptive set), what the walks leave undetermined (`walk_resolution`,
    meV), the numerical events flagged on it (`relaxation_flags`,
    `numerical_flags`), and the refrozen walk's own convergence again under
    `last_walk`."""
    last = info.get('refreeze_info') or info
    return dict(
        relaxation_fields(info),
        rejected=info.get('rejected'), energy=info.get('energy'),
        omega=info.get('omega'),
        first_point_replayed=info.get('first_point_replayed'),
        history=[dict(h) for h in info.get('history') or []],
        refreeze_rigid_shift_bohr=info.get('refreeze_rigid_shift'),
        refreeze_qp_growth=info.get('refreeze_qp_growth'),
        first_walk_de_meV=info.get('first_walk_de_meV'),
        numerical_flags=dict(info.get('numerical_flags') or {}),
        last_walk={k: last.get(k) for k in ('converged', 'cycles',
                                            'opt_grad_max', 'status')})


def refreeze_summary(record):
    """One log line of a walk's refreeze."""
    drift = record.get('refreeze_drift_meV')
    if drift is None:
        return f'refreeze {record.get("refreeze")}'
    return (f'refreeze drift at the minimum {drift:+.4f} meV, '
            f'converged {record.get("refreeze_converged")}'
            + (' (FLAGGED: above REFREEZE_TOL_MEV)'
               if record.get('refreeze') == 'not converged' else ''))


class WalkNotRelaxed(RuntimeWarning):
    """A walk that stopped short of a minimum: its last pass did not
    converge, its refreeze drift stayed above REFREEZE_TOL_MEV, or its saddle
    check did not end on a minimum. The stage carries on where the walk
    stopped and its record says so (`numerical_flags`)."""


def relaxation_flags(stage, info):
    """{flag: what it says} of the walk of `stage` ('s0', 's1', 't1', or a
    saddle check's escape walk) whose record is `info`, each warned about
    (`WalkNotRelaxed`): 'walk_not_converged' (the last walk's status) and
    'refreeze_not_converged' (the drift, meV). Empty for a relaxed walk; the stage
    carries on where the walk stopped (refuse
    inputs, warn numerics)."""
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
