"""_pole_sums against the explicit double sum, at chunk bounds that give one
chunk, several full chunks with a short last one, exactly one full chunk, and
one excitation per chunk; real and imaginary parts; a grid and a single w."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.SingleReference.GW.self_energy import _pole_sums


def check(ok, label, detail=''):
    """Print an [ok]/[FAIL] verdict line for `label` and return `ok`."""
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" +
          (f'   ({detail})' if detail else ''))
    return bool(ok)


def explicit(weights, omegas, w_grid, eps, nocc_spin, eta, calc_imag):
    """The double sum written out, one (S, q) term at a time."""
    out = np.zeros((len(weights), len(w_grid)))
    for k, (wt, om) in enumerate(zip(weights, omegas)):
        for s in range(len(om)):
            for q in range(len(eps)):
                sign = 1.0 if q < nocc_spin else -1.0
                x = w_grid - eps[q] + sign * om[s]
                g = -sign * eta / (x**2 + eta**2) if calc_imag else x / (x**2 + eta**2)
                out[k] += wt[s, q] * g
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    norb, nocc_spin, eta = 6, 3, 0.01
    eps = np.sort(rng.normal(size=norb))
    nex = (37, 5)
    weights = [rng.normal(size=(n, norb)) for n in nex]
    omegas = [np.sort(rng.uniform(0.1, 2.0, size=n)) for n in nex]
    grids = {'grid nw=7': np.linspace(-1.5, 0.5, 7), 'single w': np.array([0.2])}

    all_ok = True
    for grid_name, w_grid in grids.items():
        nw = len(w_grid)
        bounds = {'one chunk': 2**19,
                  'chunks of 10, last of 7': nw * norb * 10,
                  'exactly one full chunk': nw * norb * 37,
                  'one excitation per chunk': 1}
        for calc_imag in (False, True):
            ref = explicit(weights, omegas, w_grid, eps, nocc_spin, eta, calc_imag)
            scale = np.max(np.abs(ref))
            for bound_name, block_elems in bounds.items():
                got = _pole_sums(weights, omegas, w_grid, eps, nocc_spin, eta,
                                 calc_imag, block_elems=block_elems)
                d = np.max(np.abs(got - ref)) / scale
                tag = f'{grid_name}, {"Im" if calc_imag else "Re"}, {bound_name}'
                all_ok &= check(got.shape == (len(weights), nw) and d < 1e-13,
                                tag, f'rel {d:.1e}')
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
