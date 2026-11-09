"""What compute_gw_density_matrix accepts, and what it refuses by name.

Accepted: `mode` and `relaxation` in any case; mode='casida' with
relaxation='none' without a Numerics.

Refused, each before any expensive work, with the exception type and a word
of the message that names the reason:

- option values that do not exist (mode, relaxation, polarizability);
- options that belong to another route: auxbasis and screen_r_cut outside
  'space-time'; TDHF/BSE screening and df=False (four-centre integrals) on
  the numerical modes, which screen with RPA on a three-index object;
- a missing Numerics where one is required: the numerical modes always, the
  Casida route whenever it relaxes;
- references outside the scope, in every mode: UHF, an open-shell (ROHF)
  reference, an environment and a thermally smeared mean field;
- the removed keywords `relax`, `relax_kernel` and `eta` (no route of the
  density has a real-axis denominator for a broadening to act on).

NH3/cc-pVDZ HF, density-fitted (`gw_density_routes.mean_field`).
"""

import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import scf

from src.Base.environment_seam import attach_environment
from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix
from src.environments.solvent_screening import SolventScreening
from tests.gw_density_routes import mean_field

NUMERICAL = ('imagfrequency', 'space-time')
MODES = ('casida',) + NUMERICAL


@pytest.fixture(scope='module')
def hf():
    return mean_field('HF')


def refused(exc, words, mf, **kwargs):
    """compute_gw_density_matrix(mf, **kwargs) raises `exc` naming `words`."""
    kwargs.setdefault('numerics', Numerics())
    with pytest.raises(exc) as err:
        compute_gw_density_matrix(mf, **kwargs)
    message = str(err.value)
    print(f"{kwargs.get('mode', 'casida')}: {type(err.value).__name__}: {message}")
    assert all(word in message for word in words), message


def test_keywords_are_case_insensitive(hf):
    lower = compute_gw_density_matrix(hf, mode='casida', relaxation='none')
    mixed = compute_gw_density_matrix(hf, mode='Casida', relaxation='NONE')
    assert np.array_equal(lower, mixed)


def test_unrelaxed_casida_needs_no_numerics(hf):
    dm = compute_gw_density_matrix(hf, mode='casida', relaxation='none', numerics=None)
    assert dm.shape == (hf.mol.nao, hf.mol.nao)


@pytest.mark.parametrize('keyword, value, choices', [
    ('mode', 'realfrequency', MODES),
    ('relaxation', 'tdhf', ('none', 'dyson', 'zvector'))])
def test_unknown_option(hf, keyword, value, choices):
    refused(ValueError, [keyword, repr(value), *choices], hf, **{keyword: value})


@pytest.mark.parametrize('keyword, value', [('relax', True), ('relax_kernel', 'tdhf'),
                                            ('eta', 1e-3)])
def test_removed_keywords(hf, keyword, value):
    refused(TypeError, [keyword], hf, **{keyword: value})


def test_unknown_polarizability(hf):
    refused(ValueError, ['XYZ'], hf, mode='casida', relaxation='none',
            polarizability='XYZ')


@pytest.mark.parametrize('mode', NUMERICAL)
@pytest.mark.parametrize('polarizability', ['TDHF', 'BSE'])
def test_numerical_modes_screen_with_rpa_only(hf, mode, polarizability):
    refused(NotImplementedError, ['RPA', polarizability], hf, mode=mode,
            relaxation='none', polarizability=polarizability)


@pytest.mark.parametrize('mode', NUMERICAL)
def test_numerical_modes_need_density_fitting(hf, mode):
    refused(NotImplementedError, ['df=False'], hf, mode=mode, relaxation='none',
            df=False)


@pytest.mark.parametrize('mode', ['casida', 'imagfrequency'])
@pytest.mark.parametrize('keyword, value', [('auxbasis', 'cc-pvdz-ri'),
                                            ('screen_r_cut', 10.0)])
def test_space_time_options_elsewhere(hf, mode, keyword, value):
    refused(ValueError, [keyword, 'space-time'], hf, mode=mode, relaxation='none',
            **{keyword: value})


@pytest.mark.parametrize('mode, relaxation',
                         [(mode, 'none') for mode in NUMERICAL]
                         + [('casida', 'dyson'), ('casida', 'zvector')])
def test_missing_numerics(hf, mode, relaxation):
    refused(TypeError, ['Numerics'], hf, mode=mode, relaxation=relaxation,
            numerics=None)


@pytest.mark.parametrize('mode', MODES)
def test_unrestricted_reference(hf, mode):
    refused(NotImplementedError, ['restricted'], scf.UHF(hf.mol), mode=mode,
            relaxation='none')


@pytest.mark.parametrize('mode', MODES)
def test_open_shell_reference(hf, mode):
    cation = hf.mol.copy()
    cation.charge, cation.spin = 1, 1
    cation.build()
    refused(NotImplementedError, ['closed-shell'], scf.ROHF(cation), mode=mode,
            relaxation='none')


@pytest.mark.parametrize('mode', MODES)
def test_environment(hf, mode):
    solvated = attach_environment(copy.copy(hf),
                                  SolventScreening(hf.mol, eps=2.0, eps_static=2.0))
    refused(NotImplementedError, ['gas phase', 'SolventScreening'], solvated,
            mode=mode, relaxation='none')


@pytest.mark.parametrize('mode', MODES)
def test_smeared_mean_field(hf, mode):
    smeared = copy.copy(hf)
    smeared.sigma, smeared.smearing_method = 0.01, 'fermi'
    refused(NotImplementedError, ['T = 0'], smeared, mode=mode, relaxation='none')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
