"""Two iterative solves that stop short are flagged under their own names.

WHAT BREAKS IF THIS FAILS. A numerical shortfall is warned about and carried
as a named flag into the record (docs/warnings.md); a flag that names another
solver sends the reader to the wrong solve, and a shortfall that is never
checked is not reported at all.

- The quasi-boson Z solve (`gradients.gwbse.targets.solve_Z`) whose
  conjugate gradients stop above tolerance raises
  'quasi_boson_z_not_converged', not the orbital-response flag.
- A Pipek-Mezey Newton step above the dense limit
  (`fragments.localization._newton_polish`) whose MINRES stops short raises
  'fragment_localization_step_not_converged' and keeps the step.

Both solvers are synthetic (a random positive quasi-boson block, a quadratic
functional) and the Krylov solve is stubbed to report a shortfall.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.Base.numerics.constants import FRAGMENT_PM_POLISH_MAX
from src.fragments import localization
from src.gradients.gwbse import targets


def short(op, rhs, **kw):
    return np.zeros_like(rhs), 7


def test_an_unconverged_quasi_boson_z_solve_has_its_own_flag(monkeypatch):
    rng = np.random.default_rng(3)
    n = 3
    a = rng.standard_normal((n, n))
    qb = SimpleNamespace(wt=np.array([-0.2, 0.1, 0.3]),
                         Vt=np.linalg.qr(rng.standard_normal((n, n)))[0],
                         A=a @ a.T + n * np.eye(n), B=0.1 * np.eye(n))
    monkeypatch.setattr(targets, 'cg', short)
    flags, mine = [], []
    with pytest.warns(targets.QuasiBosonZNotConverged), collecting(flags):
        targets.solve_Z(qb, rng.standard_normal((n, n)), flags=mine)
    assert [f['flag'] for f in flags] == ['quasi_boson_z_not_converged']
    assert [f['flag'] for f in mine] == ['quasi_boson_z_not_converged']


def test_a_short_minres_newton_step_is_flagged(monkeypatch):
    n = 2
    while n * (n - 1) // 2 <= FRAGMENT_PM_POLISH_MAX:
        n += 16
    npar = n * (n - 1) // 2

    class Functional:
        mo_coeff = None

        def gen_g_hop(self, u):
            return np.ones(npar), (lambda x: x), None

        def extract_rotation(self, x):
            return np.eye(n)

    monkeypatch.setattr(localization, 'minres', short)
    flags = []
    with pytest.warns(localization.LocalizationStepNotConverged), \
            collecting(flags):
        localization._newton_polish(Functional(), np.eye(n), maxiter=1)
    assert [f['flag'] for f in flags] == [
        'fragment_localization_step_not_converged']
