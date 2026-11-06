"""The iterated-Dyson GW density (relaxation='dyson') on all three routes.

- 'imagfrequency' and 'space-time' reproduce the 'casida' density per block
  (oo, ov, vv) of the MO-basis density.
- Every route's natural occupations sum to N and lie in [0, 2].
- The numerical routes reproduce the Casida occupations.
- The Casida occupations reproduce MOLGW's relaxed (iterated-Dyson) ones:
  HF to every printed digit, PBE0 to 6.6e-6 (gw_density_routes.MOLGW_OCCUPATIONS).
- With TDHF and BSE screening (Casida only), the natural occupations sum to
  N and lie in [0, 2], on CH4/cc-pVDZ (why CH4: gw_density_routes).

Setup and the checks: gw_density_routes. NH3/cc-pVDZ (CH4/cc-pVDZ for TDHF
and BSE), DF on cc-pvdz-ri, HF and PBE0; default quadratures, space-time on
the G1 ISDF grid. Run with `-s` for the printed occupations and differences.

Casida occupations (largest five ... smallest; sum 10 to machine precision):

    HF    1.999849 1.983440 1.974183 1.971676 1.971676 ... 8.52e-05
    PBE0  1.999841 1.973983 1.958326 1.953983 1.953983 ... 1.52e-04

On HF these are the 'zvector' occupations: both relaxations use the HF kernel
there (test_gw_density_relaxation checks the densities coincide).

max |route - casida| per block of the MO density (max |Delta gamma| 2.8e-2
HF, 4.6e-2 PBE0), on the occupations, and |sum - N|:

                         oo       ov       vv       occ      sum
    HF   imagfrequency   3.3e-9   1.0e-9   2.3e-10  3.3e-9   3.0e-9
    HF   space-time      1.9e-7   4.5e-6   5.5e-7   2.4e-7   4.8e-9
    PBE0 imagfrequency   8.0e-8   7.6e-8   6.4e-10  8.1e-8   8.6e-8
    PBE0 space-time      3.3e-7   5.1e-5   1.1e-6   1.9e-6   7.7e-8

PBE0 ov is the one entry well above the other relaxations' (space-time 5.1e-5
against 6.5e-6 'zvector' and 5.5e-6 'none'). The cause is the conditioning of
Eq. 23a, not the route: the HF-kernel matrix on the PBE0 orbitals and
energies has its lowest eigenvalue at 0.034 Ha (0.32 Ha on HF; lowest ov gap
0.34 Ha), so the solve magnifies the error in its right-hand side by ~29
against ~3 on HF. The exact-integral route shows the same factor on its own
smaller error (imagfrequency ov 6.0e-9 'none' -> 7.6e-8 'dyson'). The ISDF
grid GH (the finest for cc-pVDZ) gives space-time ov 4.0e-5 PBE0.

CH4, TDHF and BSE screening: smallest natural occupation (sum N to 2e-14,
largest at least 2.5e-4 below 2):

    HF    TDHF  1.41e-4        BSE  1.34e-4
    PBE0  TDHF  2.60e-4        BSE  2.40e-4

NH3 is not used for TDHF/BSE because there, on PBE0, 'dyson' leaves [0, 2]:
one natural occupation is -1.13e-3 (TDHF) and -2.56e-4 (BSE), with N kept
to 1e-14 and no convergence flag ('none' and 'zvector' stay in [0, 2]). The
solve is right: a dense solve of Eq. 23 reproduces the ov block to 1e-12. It
is the same soft mode as above: TDHF and BSE project the right-hand side
onto the 0.034 Ha eigenvector of the HF kernel 6x and 4x more than RPA
does (-4.5e-3 and -3.2e-3 against -7.4e-4), so that mode carries 84% and
77% of |ov|^2, and the ov block reaches 0.136 (TDHF) against 0.015 for
'zvector'.
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

RELAXATION = 'dyson'
#: Per block of the MO density: the worst reached (PBE0) with a factor ~4-10;
#: space-time is looser than for the other relaxations (see above).
BLOCK_BAR = {'imagfrequency': 1e-6, 'space-time': 2e-4}
#: On the occupations: the worst reached (PBE0) with a factor ~10.
OCCUPATION_BAR = {'imagfrequency': 1e-6, 'space-time': 2e-5}


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
