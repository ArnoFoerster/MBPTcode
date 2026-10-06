"""Force constants of any surface along a few chosen normal modes, by central
differences of its analytic gradient.

Displace along mass-weighted ground-state mode k by +-h_k and project each
displaced gradient on ALL the modes:

    R[k, l] = (g_l(+h_k) - g_l(-h_k)) / (2 h_k) = d^2E / dq_k dq_l.

The m x m block on the chosen modes gives the target surface's frequencies
and the mixing of those modes on it (the Duschinsky rotation inside the
subspace). That costs 2m gradients, against 6N for a full Hessian.

The rest of each row is kept, for two diagnostics:

- The couplings to the modes left out give a second-order estimate of the
  frequency shift those modes would cause. The estimate takes them at their
  ground-state curvature, so it is a warning and not a correction.
- The block's asymmetry R_kl - R_lk, read before symmetrizing, is the
  gradient's noise floor at this step.

Off a stationary point, removing the rigid-body motions from the Hessian is
not exact, because the gradient couples to rotations. The rows here
live on the vibrational modes only. A vibrationally relaxed excited state is
probed at its own minimum, so evaluate there when the frequency is meant to
be compared with a measured one.
"""
import numpy as np

from src.Base.constants import (HARTREE_TO_CM, HESSIAN_FD_ASYMMETRY_TOL,
                                NUCLEAR_FD_STEP)
from src.properties.surface import evaluate


def cartesian_mode(mode, masses):
    """(natm, 3) Cartesian displacement of one mass-weighted mode, x = M^-1/2 q."""
    w = np.repeat(np.asarray(masses), 3) ** -0.5
    return (np.asarray(mode) * w).reshape(-1, 3)


def stretch_modes(mol, modes, masses, bonds, top=1):
    """The modes that stretch the given bonds most, most stretching first.

    bonds: [(ia, ja), ...], e.g. the C=O of a carbonyl.

    The weight is |P u|^2 / |u|^2 for the Cartesian displacement u, with P the
    projector on the bond-stretch patterns (-e_ij on atom i, +e_ij on atom j).
    Unequal masses keep a pure stretch below 1. Returns (indices into the
    columns of `modes`, weights), `top` long.
    """
    coords = np.asarray(mol.atom_coords())
    patterns = np.zeros((len(bonds), coords.size))
    for b, (ia, ja) in enumerate(bonds):
        e = coords[ja] - coords[ia]
        e = e / np.linalg.norm(e)
        s = np.zeros_like(coords)
        s[ia], s[ja] = -e, e
        patterns[b] = s.ravel()
    basis, _ = np.linalg.qr(patterns.T)
    weights = np.zeros(modes.shape[1])
    for k in range(modes.shape[1]):
        u = cartesian_mode(modes[:, k], masses).ravel()
        weights[k] = float(np.sum((basis.T @ u) ** 2) / (u @ u))
    order = np.argsort(-weights)[:top]
    return order, weights[order]


def mode_step(mode, masses, step=NUCLEAR_FD_STEP):
    """h_k in the mass-weighted coordinate at which the furthest atom moves `step` Bohr."""
    u = cartesian_mode(mode, masses)
    return step / float(np.linalg.norm(u, axis=1).max())


def displaced_along(mol, mode, masses, dq):
    """`mol` moved by `dq` along one mass-weighted mode."""
    out = mol.copy()
    coords = np.asarray(mol.atom_coords()) + dq * cartesian_mode(mode, masses)
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def mode_gradient(grad, modes, masses):
    """A Cartesian gradient projected on the mass-weighted modes, d/dq = M^-1/2 d/dx."""
    w = np.repeat(np.asarray(masses), 3) ** -0.5
    return modes.T @ (np.asarray(grad, float).ravel() * w)


def mode_hessian(surface, mol, omega, modes, masses, selected,
                 step=NUCLEAR_FD_STEP, map_fn=None, verbose=False,
                 asymmetry_tol=HESSIAN_FD_ASYMMETRY_TOL):
    """Frequencies of `surface` in the subspace of the `selected` ground-state modes.

    omega, modes, masses: `vibronic.normal_modes` of the ground state; `modes`
        is only a basis, so `mol` need not be its minimum.
    step: Bohr moved by the furthest atom along each mode.
    map_fn: anything with `map`'s signature, for the 2m independent gradients.

    Returns a dict: `omega_cm` (ascending, negative for a negative curvature)
    and `vectors` (in the selected-mode basis) from the symmetrized `block`;
    `rows`, the unsymmetrized rows over every mode; `leakage_cm`, the
    second-order shift from the modes left out; `omega_ground_cm`;
    `asymmetry`; `steps` (h_k, mass-weighted Bohr); `gradients`.
    """
    selected = [int(k) for k in selected]
    hs = [mode_step(modes[:, k], masses, step) for k in selected]
    jobs = [(i, sign) for i in range(len(selected)) for sign in (+1, -1)]
    runner = map if map_fn is None else map_fn

    def one(job):
        i, sign = job
        k = selected[i]
        moved = displaced_along(mol, modes[:, k], masses, sign * hs[i])
        grad, energy, _ = evaluate(surface, moved)
        if verbose:
            print(f'  [mode {k:3d} {"+" if sign > 0 else "-"}] '
                  f'E {energy:.10f}  |g|max {np.abs(grad).max():.3e}',
                  flush=True)
        return job, mode_gradient(grad, modes, masses)

    projected = dict(runner(one, jobs))
    rows = np.array([(projected[(i, +1)] - projected[(i, -1)]) / (2.0 * hs[i])
                     for i in range(len(selected))])
    block_raw = rows[:, selected]
    asymmetry = float(np.abs(block_raw - block_raw.T).max())
    scale = max(float(np.abs(block_raw).max()), 1e-30)
    if asymmetry / scale > asymmetry_tol:
        raise RuntimeError(
            f'the mode-projected force constants are asymmetric by '
            f'{asymmetry:.2e} ({asymmetry / scale:.2e} of the largest), above '
            f'{asymmetry_tol:.1e}: the displaced gradients carry more noise '
            f'than curvature at this step. Check the gradient reproducibility '
            f'at this geometry, or take a larger step through `step_ladder`.')
    block = 0.5 * (block_raw + block_raw.T)
    lam, vec = np.linalg.eigh(block)
    freq = np.sign(lam) * np.sqrt(np.abs(lam))

    outside = [l for l in range(modes.shape[1]) if l not in set(selected)]
    leakage = np.zeros(len(lam))
    if outside:
        coupling = vec.T @ rows[:, outside]
        gap = lam[:, None] - (np.asarray(omega)[outside] ** 2)[None, :]
        dlam = (coupling ** 2 / np.where(np.abs(gap) > 1e-30, gap,
                                         1e-30)).sum(axis=1)
        # d(omega) = d(lambda) / (2 omega), at the subspace frequency
        leakage = dlam / (2.0 * np.where(np.abs(freq) > 1e-30, np.abs(freq),
                                         1e-30))
    return {'omega_cm': freq * HARTREE_TO_CM,
            'vectors': vec,
            'omega_ground_cm': np.asarray(omega)[selected] * HARTREE_TO_CM,
            'block': block,
            'rows': rows,
            'leakage_cm': leakage * HARTREE_TO_CM,
            'asymmetry': asymmetry,
            'steps': np.asarray(hs),
            'gradients': len(jobs)}


def step_ladder(surface, mol, omega, modes, masses, selected,
                steps=(4.0 * NUCLEAR_FD_STEP, 2.0 * NUCLEAR_FD_STEP,
                       NUCLEAR_FD_STEP),
                **kw):
    """`mode_hessian` at halved steps, and the convergence order of each frequency.

    Returns (results, ratios) with ratios[j] = (w_j - w_{j+1}) / (w_{j+1} - w_{j+2}).
    A clean central difference gives 4. A ratio near 2 means an error first
    order in h, i.e. a gradient that is not the derivative of its own energy
    at the displaced geometries, and no step makes that frequency usable.
    """
    steps = tuple(steps)
    halved = all(np.isclose(steps[j] / steps[j + 1], 2.0)
                 for j in range(len(steps) - 1))
    if not halved:
        raise ValueError(f'the ladder reads the order off halved steps; got {steps}')
    results = [mode_hessian(surface, mol, omega, modes, masses, selected,
                            step=h, **kw) for h in steps]
    w = [r['omega_cm'] for r in results]
    ratios = [(w[j] - w[j + 1]) / np.where(np.abs(w[j + 1] - w[j + 2]) > 1e-12,
                                           w[j + 1] - w[j + 2], 1e-12)
              for j in range(len(w) - 2)]
    return results, ratios
