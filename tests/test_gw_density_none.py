"""The unrelaxed GW density (relaxation='none') on all three routes.

- 'imagfrequency' and 'space-time' reproduce the 'casida' density per block
  (oo, ov, vv) of the MO-basis density.
- Every route's natural occupations sum to N and lie in [0, 2].
- The numerical routes reproduce the Casida occupations.
- The Casida occupations reproduce MOLGW's: HF to every printed digit,
  PBE0 to 3.0e-6 (gw_density_routes.MOLGW_OCCUPATIONS).
- With TDHF and BSE screening (Casida only), the natural occupations sum to
  N and lie in [0, 2], on CH4/cc-pVDZ (why CH4: gw_density_routes).

Setup and the checks: gw_density_routes. NH3/cc-pVDZ (CH4/cc-pVDZ for TDHF
and BSE), DF on cc-pvdz-ri, HF and PBE0; default quadratures (outer minimax
34), space-time on the G1 ISDF grid. Run with `-s` for the printed
occupations and differences.

Casida occupations (largest five ... smallest; sum 10 to machine precision):

    HF    1.999849 1.983427 1.974139 1.971642 1.971642 ... 8.98e-05
    PBE0  1.999836 1.974083 1.957993 1.954030 1.954030 ... 1.64e-04

max |route - casida| per block of the MO density (max |Delta gamma| 2.8e-2
HF, 4.6e-2 PBE0), on the occupations, and |sum - N|:

                         oo       ov       vv       occ      sum
    HF   imagfrequency   4.0e-10  2.6e-10  2.3e-10  3.9e-10  1.8e-10
    HF   space-time      1.9e-7   2.4e-6   5.5e-7   2.4e-7   1.0e-9
    PBE0 imagfrequency   3.6e-10  5.5e-10  3.3e-10  3.6e-10  2.2e-10
    PBE0 space-time      3.3e-7   5.5e-6   1.1e-6   6.1e-7   2.0e-9

The ISDF grid GH (the finest for cc-pVDZ) gives space-time ov 1.8e-6 HF and
3.7e-6 PBE0: little closer than G1.

CH4, TDHF and BSE screening: smallest natural occupation (sum N to 2e-14,
largest at least 2.5e-4 below 2):

    HF    TDHF  1.41e-4        BSE  1.34e-4
    PBE0  TDHF  2.71e-4        BSE  2.48e-4
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from tests.gw_density_routes import (MODES, NUMERICAL_MODES, REFERENCES,
                                     check_blocks_match_casida,
                                     check_occupations_match_casida,
                                     check_casida_matches_molgw,
                                     check_occupations_physical,
                                     check_screened_occupations_physical, mean_field,
                                     route_densities, SCREENINGS)

RELAXATION = 'none'
#: Per block of the MO density: the worst reached (PBE0) with a factor ~10.
BLOCK_BAR = {'imagfrequency': 2e-8, 'space-time': 5e-5}
#: On the occupations: the worst reached (PBE0) with a factor ~10.
OCCUPATION_BAR = {'imagfrequency': 1e-8, 'space-time': 5e-6}


@pytest.fixture(scope='module', params=REFERENCES)
def routes(request):
    return route_densities(request.param, RELAXATION)


@pytest.mark.parametrize('mode', NUMERICAL_MODES)
def test_blocks_match_casida(routes, mode):
    mf, densities = routes
    check_blocks_match_casida(mf, densities, mode, RELAXATION, BLOCK_BAR[mode])


@pytest.mark.parametrize('mode', MODES)
def test_natural_occupations_physical(routes, mode):
    mf, densities = routes
    check_occupations_physical(mf, densities, mode, RELAXATION)


@pytest.mark.parametrize('mode', NUMERICAL_MODES)
def test_natural_occupations_match_casida(routes, mode):
    mf, densities = routes
    check_occupations_match_casida(mf, densities, mode, RELAXATION, OCCUPATION_BAR[mode])


def test_casida_matches_molgw(routes):
    mf, densities = routes
    check_casida_matches_molgw(mf, densities, RELAXATION)



@pytest.fixture(scope='module', params=REFERENCES)
def ch4(request):
    return mean_field(request.param, 'CH4')


@pytest.mark.parametrize('polarizability', SCREENINGS)
def test_screened_occupations_physical(ch4, polarizability):
    check_screened_occupations_physical(ch4, polarizability, RELAXATION)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
