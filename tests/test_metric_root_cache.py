"""The auxiliary metric's root and eigensystem kept by content
(`separable_ri.metric_eigensystem`): a repeat is the first call's bits, the
root a caller gets is its own copy, a dressed metric that is indefinite is
refused on every call, and a metric past the byte budget is formed and not
kept. Water/cc-pVDZ-JKFIT, under a second.

Run: python tests/test_metric_root_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

import src.Base.factorization.separable_ri as separable_ri
from src.Base.numerics.constants import AUX_METRIC_ROOT_FLOOR
from src.Base.factorization.separable_ri import aux_metric_sqrt, metric_eigensystem
from src.gradients.isdf_derivatives import sqrtm_adjoint

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


class Kernel:
    """An environment whose vtilde is a fixed matrix."""

    def __init__(self, k):
        self.k = k

    def aux_kernel(self, auxmol):
        return self.k


def setup():
    auxmol = gto.M(atom=WATER, basis='cc-pvdz-jkfit', verbose=0)
    V = auxmol.intor('int2c2e', aosym='s1')
    rng = np.random.default_rng(7)
    a = rng.standard_normal(V.shape)
    return auxmol, V, Kernel(1e-3 * (a @ a.T) / len(V))


def root_uncached(M):
    """The root as `aux_metric_sqrt` formed it before the cache."""
    w, v = np.linalg.eigh(M)
    keep = w > AUX_METRIC_ROOT_FLOOR * w.max()
    return (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T


def test_a_repeat_is_the_first_calls_bits():
    separable_ri._METRIC_ROOTS.clear()
    auxmol, V, env = setup()
    first = aux_metric_sqrt(auxmol, env, V=V)
    assert np.array_equal(first, root_uncached(V + env.k))
    assert len(separable_ri._METRIC_ROOTS) == 1
    first[:] = 0.0                       # the caller's copy, not the kept one
    assert np.array_equal(aux_metric_sqrt(auxmol, env, V=V),
                          root_uncached(V + env.k))
    # the bare metric is another key; V=None forms the same matrix
    assert np.array_equal(aux_metric_sqrt(auxmol, None), root_uncached(V))
    assert len(separable_ri._METRIC_ROOTS) == 2


def test_the_adjoint_reads_the_roots_eigensystem():
    separable_ri._METRIC_ROOTS.clear()
    auxmol, V, env = setup()
    M = V + env.k
    bar = np.random.default_rng(3).standard_normal(V.shape)
    w, U = np.linalg.eigh(M)
    keep = w > AUX_METRIC_ROOT_FLOOR * w.max()
    Uk, sk = U[:, keep], np.sqrt(w[keep])
    E = Uk.T @ (0.5 * (bar + bar.T)) @ Uk
    want = Uk @ (E / (sk[:, None] + sk[None, :])) @ Uk.T
    aux_metric_sqrt(auxmol, env, V=V)
    assert np.array_equal(sqrtm_adjoint(M, bar), want)
    assert len(separable_ri._METRIC_ROOTS) == 1
    entry = metric_eigensystem(M)
    with pytest.raises(ValueError):
        entry['v'][0, 0] = 1.0           # read-only


def test_an_indefinite_dressed_metric_is_refused_every_time():
    separable_ri._METRIC_ROOTS.clear()
    auxmol, V, _ = setup()
    w, U = np.linalg.eigh(V)
    over = Kernel(-2.0 * w.min() * np.outer(U[:, 0], U[:, 0])
                  - 1e-3 * w.max() * np.outer(U[:, 0], U[:, 0]))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='indefinite'):
            aux_metric_sqrt(auxmol, over, V=V)


def test_a_metric_past_the_budget_is_not_kept(monkeypatch):
    separable_ri._METRIC_ROOTS.clear()
    auxmol, V, env = setup()
    monkeypatch.setattr(separable_ri, 'METRIC_ROOT_CACHE_BYTES', V.nbytes)
    assert np.array_equal(aux_metric_sqrt(auxmol, env, V=V),
                          root_uncached(V + env.k))
    assert len(separable_ri._METRIC_ROOTS) == 0
    # a budget for one: the older is dropped when a second arrives
    monkeypatch.setattr(separable_ri, 'METRIC_ROOT_CACHE_BYTES',
                        3 * V.nbytes + V.shape[0] * 8)
    aux_metric_sqrt(auxmol, env, V=V)
    aux_metric_sqrt(auxmol, None, V=V)
    assert len(separable_ri._METRIC_ROOTS) == 1
    assert np.array_equal(aux_metric_sqrt(auxmol, env, V=V),
                          root_uncached(V + env.k))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
