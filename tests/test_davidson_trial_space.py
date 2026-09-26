"""The Casida Davidson's trial space: the bound real_eig collapses it at, and
the count of those collapses.

pyscf's `real_eig` sizes its trial space from its process-wide MAX_MEMORY, 103
pairs at the chlorophyllide dimer/cc-pVDZ, and a collapse keeps only the nroots
Ritz vectors, so a large pair space collapsed every few cycles and its top
roots stalled. The Davidson hands real_eig a budget that holds
DAVIDSON_SPACE_CYCLES increments within DAVIDSON_SPACE_GB, never less than
pyscf's own, and counts the collapses that remain.

The model is a Casida problem of the RPA shape -- A = D + 2K, B = 2K with
K = V^T V -- whose roots are known densely.
"""
import io
import re
import warnings

import numpy as np
import pytest
from pyscf.lib import logger

import src.SingleReference.LinearResponse.davidson as davidson
from src.Base.constants import DAVIDSON_SPACE_CYCLES, DAVIDSON_SPACE_GB
from src.SingleReference.LinearResponse.davidson import bse_pair_diagonal

NOCC, NVIR, NAUX, NROOTS, CONV_TOL = 8, 150, 40, 12, 1e-6
FORCED_SPACE = 60
DIMER_PAIRS = 354 * 1426


@pytest.fixture()
def model():
    """(apply_AB, diag, dense roots) of a model Casida problem."""
    rng = np.random.default_rng(11)
    eps = np.concatenate([-np.sort(rng.uniform(0.3, 2.0, NOCC))[::-1],
                          np.sort(rng.uniform(0.05, 3.0, NVIR))])
    diag = bse_pair_diagonal(eps, NOCC)
    n_ov = diag.size
    V = rng.standard_normal((NAUX, n_ov)) / np.sqrt(n_ov) * 0.5

    def apply_AB(z):
        f = z.reshape(len(z), -1)
        v = (2 * (f @ V.T) @ V).reshape(z.shape)
        return diag[None] * z + v, v

    sqrt_d = np.sqrt(diag.ravel())
    m = sqrt_d[:, None] * (np.diag(diag.ravel()) + 4 * V.T @ V) * sqrt_d[None, :]
    return apply_AB, diag, np.sqrt(np.linalg.eigvalsh(m))


def logged_real_eig(monkeypatch):
    """Patch real_eig to log at debug level; returns the log buffer."""
    buf = io.StringIO()
    real_eig = davidson.real_eig

    def with_log(*args, **kwargs):
        kwargs['verbose'] = logger.Logger(buf, logger.DEBUG)
        return real_eig(*args, **kwargs)

    monkeypatch.setattr(davidson, 'real_eig', with_log)
    return buf


def solve(model, timings):
    apply_AB, diag, _ = model
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        return davidson._run_davidson(apply_AB, diag, NROOTS, CONV_TOL, 100,
                                      None, timings=timings)


@pytest.mark.parametrize('n_pair', [40, 95, 9353, 50000, DIMER_PAIRS, 10**7])
def test_the_bound_is_never_below_pyscfs_own(n_pair):
    """Raised to the cycles-or-memory target where pyscf's own is smaller,
    and pyscf's own, bit for bit, everywhere else."""
    own = davidson._real_eig_space(davidson.param.MAX_MEMORY, NROOTS, n_pair)
    space_inc = own[0]
    bound = davidson._real_eig_space(
        davidson._trial_space_memory(NROOTS, n_pair), NROOTS, n_pair)[1]
    target = min(DAVIDSON_SPACE_CYCLES * space_inc,
                 int(DAVIDSON_SPACE_GB * 1e9 / (32 * n_pair)))
    assert bound == min(max(own[1], target, 4 * NROOTS), n_pair)
    assert bound >= own[1]


def test_the_dimer_pair_space_is_no_longer_collapsed_every_few_cycles():
    """At the pair space where pyscf's own bound is a handful of cycles, the
    bound is DAVIDSON_SPACE_GB's worth of trial pairs."""
    own = davidson._real_eig_space(davidson.param.MAX_MEMORY, NROOTS,
                                   DIMER_PAIRS)
    bound = davidson._real_eig_space(
        davidson._trial_space_memory(NROOTS, DIMER_PAIRS), NROOTS,
        DIMER_PAIRS)[1]
    assert own[1] < 10 * NROOTS
    assert bound == int(DAVIDSON_SPACE_GB * 1e9 / (32 * DIMER_PAIRS))


def test_collapses_are_counted_as_real_eig_logs_them(model, monkeypatch):
    """A bound forced small: the count and the largest subspace are the ones
    real_eig's own log shows, and the roots are the dense ones."""
    monkeypatch.setattr(davidson.param, 'MAX_MEMORY', 1)
    monkeypatch.setattr(davidson, 'DAVIDSON_SPACE_GB',
                        (FORCED_SPACE + 0.5) * 32 * model[1].size / 1e9)
    buf = logged_real_eig(monkeypatch)
    t = {}
    omega, _, _ = solve(model, t)
    m1 = [int(m.group(1)) for m in
          re.finditer(r'real_lr_eig \d+ (\d+)', buf.getvalue())]
    logged = sum(1 for a, b in zip(m1, m1[1:]) if b < a)
    assert t['davidson_max_space'] == FORCED_SPACE
    assert t['davidson_collapses'] == logged > 0
    assert max(m1) <= t['davidson_subspace_max'] <= FORCED_SPACE
    assert np.abs(omega - model[2][:NROOTS]).max() <= CONV_TOL


def test_the_default_bound_holds_the_solve_whole(model):
    """The model's pair space fits: no collapse, and the dense roots."""
    t = {}
    omega, _, _ = solve(model, t)
    assert t['davidson_collapses'] == 0
    assert t['davidson_subspace_max'] <= t['davidson_max_space']
    assert np.abs(omega - model[2][:NROOTS]).max() <= CONV_TOL
