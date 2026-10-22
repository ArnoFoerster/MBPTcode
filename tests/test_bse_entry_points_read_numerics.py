"""The Davidson's entry points read their choices off the one `Numerics`.

`solve_casida_davidson`, `solve_bse_isdf`, `solve_bse_df` and `solve_bse`
must not take the root count, the residual, the cycle cap, the guess count
and the preconditioner as keywords with defaults of their own -- a 1e-5
residual where the chain converges to BSE_DAVIDSON_CONV_TOL (1e-8), 100
cycles where it allows BSE_FORCE_MAX_CYCLE, 50 in `casida_rows`, 3 roots in
one entry point and 5 in the others -- since a stand-alone solve and the
chain's would be two answers to one question. They take `numerics` and read
`numerics.bse` (`davidson.bse_settings`), whose defaults are the chain's;
`casida_rows` is handed the values and reads its linear-dependence floor
from DAVIDSON_LINDEP, and the guess factor lives in constants.py.

Water/STO-3G RHF with density fitting; one small Casida solve.
"""
import inspect

import numpy as np
from pyscf import gto, scf

from src.Base import constants
from src.Base.numerics import Numerics
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.LinearResponse import (block_actions, bse, davidson,
                                                trial_space)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)

CHOICES = ('nroots', 'conv_tol', 'max_cycle', 'guess_factor',
           'preconditioner')
H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def test_the_entry_points_take_the_object():
    for fn in (davidson.solve_casida_davidson, davidson.solve_bse_isdf,
               davidson.solve_bse_df, bse.solve_bse):
        params = inspect.signature(fn).parameters
        assert 'numerics' in params, fn.__name__
        assert not set(CHOICES) & set(params), fn.__name__


def test_one_default_per_choice():
    b = Numerics().bse
    assert davidson.bse_settings(None, *CHOICES) == (
        constants.BSE_DAVIDSON_NROOTS, constants.BSE_DAVIDSON_CONV_TOL,
        constants.BSE_FORCE_MAX_CYCLE, constants.DAVIDSON_GUESS_FACTOR,
        constants.DAVIDSON_PRECONDITIONER)
    assert davidson.bse_settings(None, *CHOICES) == tuple(
        getattr(b, name) for name in CHOICES)
    rows = inspect.signature(trial_space.casida_rows).parameters
    for name in ('tol_residual', 'max_cycle', 'nroots'):
        assert rows[name].default is inspect.Parameter.empty, name
    assert 'lindep' not in rows
    assert not hasattr(davidson, 'GUESS_FACTOR')


def test_a_stand_alone_solve_reaches_the_chains_residual():
    """In numbers: the default solve's roots stand at the chain's
    residual, where the 1e-5 default left them above it."""
    mol = gto.M(atom=H2O, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='def2-svp-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    lr = LinearResponseSolver(
        get_orbital_energies(mf, representation='spatial'),
        coeff_df=get_density_fitting_coefficients(mol, mf,
                                                  representation='spatial'),
        spin_mode='restricted')
    om, X, Y = davidson.solve_casida_davidson(lr, nocc)
    assert len(om) == constants.BSE_DAVIDSON_NROOTS
    act, diag = block_actions._block_action(lr, nocc, 'RPA', None, None)
    shape = (-1,) + diag.shape
    Ax, Bx = act(X.T.reshape(shape))
    Ay, By = act(Y.T.reshape(shape))
    top = (Ax + By).reshape(len(om), -1) - om[:, None] * X.T
    bot = (Bx + Ay).reshape(len(om), -1) + om[:, None] * Y.T
    r = np.sqrt(np.sum(top**2, axis=1) + np.sum(bot**2, axis=1))
    assert r.max() <= constants.BSE_DAVIDSON_CONV_TOL, r
