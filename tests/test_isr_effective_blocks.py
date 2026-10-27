"""isr_assembly.isr_effective_blocks and norm_divide against the exact
blockwise orthogonalization, as linear algebra on random series.

Reference: with the full H(lambda), S(lambda) (classes in order, S^(0) = 1,
divided by the ground-state norm N(lambda) when one is given), each class
family is the class unit block Gram-Schmidt-projected off the earlier
classes' S-orthonormal families and then Loewdin-orthonormalized
(W^{-1/2}, scipy's principal fractional power); the effective blocks are
eff_i^T H eff_j. Sampled on a complex-lambda circle, the FFT gives the
Taylor coefficients of the blocks and of the families, which the
order-by-order series must reproduce through nmax. Dividing H and S by the
same N(lambda) leaves the blocks unchanged (eff scales by N^{1/2}); the norm
is seen by the families only, so both are compared. The generated-ADC raw
blocks (collect_spinorbital_raw) arrive with the generated runtime (E3).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from src.SingleReference.ADC.isr_assembly import isr_effective_blocks, norm_divide

DIMS = {0: 3, 1: 4, 2: 2}
NMAX = 4
TOL = 1e-12
RADIUS = 0.3
N_POINTS = 32
NORM = {0: 1.0, 2: 0.06, 3: -0.02, 4: 0.01}


def _random_series(rng):
    """Raw H, S series for ci <= cj: S^(0) = 1 within a class and 0 across."""
    H, S = {}, {}
    classes = list(DIMS)
    for a, ci in enumerate(classes):
        for cj in classes[a:]:
            shape = (DIMS[ci], DIMS[cj])
            h = {m: rng.standard_normal(shape) * 0.1 for m in range(NMAX + 1)}
            s = {m: rng.standard_normal(shape) * 0.05 for m in range(1, NMAX + 1)}
            if ci == cj:
                h = {m: 0.5 * (x + x.T) for m, x in h.items()}
                s = {m: 0.5 * (x + x.T) for m, x in s.items()}
                s[0] = np.eye(DIMS[ci])
            H[(ci, cj)], S[(ci, cj)] = h, s
    return H, S


def _full(series, lam):
    off = np.cumsum([0] + [DIMS[c] for c in DIMS])
    n = off[-1]
    out = np.zeros((n, n), dtype=complex)
    for a, ci in enumerate(DIMS):
        for b, cj in enumerate(DIMS):
            if b < a:
                continue
            blk = sum(lam ** m * x for m, x in series[(ci, cj)].items())
            out[off[a]:off[a + 1], off[b]:off[b + 1]] = blk
            out[off[b]:off[b + 1], off[a]:off[a + 1]] = blk.T
    return out, off


def _exact_blocks(H, S, lam, norm):
    Hf, off = _full(H, lam)
    Sf, _ = _full(S, lam)
    if norm:
        nv = sum(lam ** m * v for m, v in norm.items())
        Hf, Sf = Hf / nv, Sf / nv
    fams = []
    for a, ci in enumerate(DIMS):
        X = np.zeros((off[-1], DIMS[ci]), dtype=complex)
        X[off[a]:off[a + 1]] = np.eye(DIMS[ci])
        for F in fams:
            X = X - F @ (F.T @ Sf @ X)
        fams.append(X @ fractional_matrix_power(X.T @ Sf @ X, -0.5))
    blocks = {(ci, cj): fams[a].T @ Hf @ fams[b]
              for a, ci in enumerate(DIMS) for b, cj in enumerate(DIMS)}
    eff = {(ci, c1): fams[a][off[b]:off[b + 1]]
           for a, ci in enumerate(DIMS) for b, c1 in enumerate(DIMS)}
    return blocks, eff


def _worst(series, samples):
    """Largest deviation of {key: {order: array}} from the Taylor
    coefficients of the sampled exact {key: array}."""
    worst = 0.0
    for key in samples[0]:
        coefs = np.fft.fft(np.array([s[key] for s in samples]), axis=0) / N_POINTS
        for m in range(NMAX + 1):
            ref = coefs[m] / RADIUS ** m
            got = series.get(key, {}).get(m, np.zeros(ref.shape))
            worst = max(worst, float(np.max(np.abs(got - ref))))
    return worst


@pytest.mark.parametrize('norm', [None, NORM])
def test_effective_blocks_equal_the_exact_orthogonalization(norm):
    rng = np.random.default_rng(3)
    H, S = _random_series(rng)
    blocks, eff = isr_effective_blocks(H, S, DIMS, NMAX, norm=norm)
    eff = {(ci, c1): ser for ci, fam in eff.items() for c1, ser in fam.items()}
    lams = RADIUS * np.exp(2j * np.pi * np.arange(N_POINTS) / N_POINTS)
    exact = [_exact_blocks(H, S, lam, norm) for lam in lams]
    wb = _worst(blocks, [e[0] for e in exact])
    we = _worst(eff, [e[1] for e in exact])
    print(f"norm={'yes' if norm else 'no'}: worst coefficient vs exact, blocks {wb:.1e}, "
          f"families {we:.1e}")
    assert wb < TOL, wb
    assert we < TOL, we
    if norm:
        # the families must see the norm: the unnormalized reference differs
        ref_plain = [_exact_blocks(H, S, lam, None)[1] for lam in lams]
        assert _worst(eff, ref_plain) > 1e-3


def test_norm_divide_is_the_series_quotient():
    rng = np.random.default_rng(4)
    a = {m: rng.standard_normal((2, 3)) for m in range(NMAX + 1)}
    q = norm_divide(a, NORM, NMAX)
    back = {m: sum(NORM.get(j, 0.0) * q.get(m - j, 0.0) for j in range(m + 1))
            for m in range(NMAX + 1)}
    for m in range(NMAX + 1):
        assert np.max(np.abs(back[m] - a[m])) < 1e-14, m
