"""A defect inside the (A-B) probe raises; only the probe's non-convergence is a breakdown.

When the Casida-form Davidson's projected (A-B) block is not positive definite,
`_casida_breakdown` runs the (A-B) probe to tell a stable reference's numerical
breakdown from an unstable reference. A probe that runs out of iterations is
reported as "did not converge" and the step is refused as an instability. Any
other exception from the probe (here a planted TypeError) is a defect: catching
it
with `except Exception` would report it as that instability, which names the
wrong cause.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence

import src.SingleReference.LinearResponse.davidson as davidson
from src.SingleReference.LinearResponse.probe import AmbProbeNotConverged

DIAG = np.array([[0.5, 0.7], [0.9, 1.1]])


def _probe_raising(exc):
    def probe(*args, **kwargs):
        raise exc
    return probe


def _no_action(Z):
    raise AssertionError('the planted probe never applies the action')


def test_a_defect_in_the_probe_raises_as_itself(monkeypatch):
    monkeypatch.setattr(davidson, 'lowest_amb_from_action',
                        _probe_raising(TypeError('planted defect')))
    with pytest.raises(TypeError, match='planted defect'):
        davidson._casida_breakdown(_no_action, DIAG, 1e-6)


@pytest.mark.parametrize('exc', [
    ArpackNoConvergence('planted', np.array([]), np.array([])),
    'AmbProbeNotConverged',
    ValueError('ranks apart at the lockstep'),
])
def test_a_probe_that_did_not_converge_is_a_breakdown(monkeypatch, exc):
    if exc == 'AmbProbeNotConverged':
        exc = AmbProbeNotConverged('planted')
    monkeypatch.setattr(davidson, 'lowest_amb_from_action',
                        _probe_raising(exc))
    err = davidson._casida_breakdown(_no_action, DIAG, 1e-6)
    assert isinstance(err, davidson.CasidaBreakdown)
    assert err.min_eig_amb is None
    assert 'did not converge' in str(err)
