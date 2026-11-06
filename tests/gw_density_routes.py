"""Shared setup for the per-relaxation GW density tests
(test_gw_density_none.py, test_gw_density_dyson.py, test_gw_density_zvector.py).

Each of those files runs one `relaxation` on all three routes ('casida',
'imagfrequency', 'space-time'), on HF and PBE0 references of NH3/cc-pVDZ, and
checks:

- the numerical routes reproduce the Casida density per block (oo, ov, vv) of
  the MO-basis density, to within their quadrature/ISDF error;
- the natural occupations (the eigenvalues of the total density in an
  orthonormal basis) of every route sum to N and lie in [0, 2], the
  restricted (spin-summed) bound; the GW density is restricted-spin only, so
  the unrestricted bound [0, 1] has no case;
- the numerical routes reproduce the Casida occupations;
- the Casida occupations reproduce MOLGW's (MOLGW_OCCUPATIONS).

The block comparison is the sharper test of the two: an error in the density
moves the occupations only at second order through its off-diagonal elements
(PBE0 'none', space-time: 5.5e-6 in the ov block, 6.1e-7 in the occupations),
and the occupations do not see the natural orbitals at all.

DF on cc-pvdz-ri; default quadratures (W grid minimax 20, outer
Gauss-Legendre 48); space-time on the G1 ISDF grid.
"""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix
from tests.isdf_test_grid import g1_numerics

NH3 = """
N -0.000000  0.068952 -0.000000
H  0.939731 -0.319345 -0.000000
H -0.469865 -0.319346  0.813831
H -0.469865 -0.319346 -0.813831
"""
AUXBASIS = 'cc-pvdz-ri'
REFERENCES = ('HF', 'PBE0')
MODES = ('casida', 'imagfrequency', 'space-time')
NUMERICAL_MODES = ('imagfrequency', 'space-time')
#: Restricted spin: each spatial natural orbital holds at most two electrons.
MAX_OCCUPATION = 2.0
TOL = 1e-10
#: |sum - N|: the analytic route is exact; the numerical routes reach ~1e-7
#: from their quadratures.
SUM_TOL = {'casida': 1e-8, 'imagfrequency': 1e-6, 'space-time': 1e-6}


def _occupations(text):
    return np.array(text.split(), dtype=float)


#: MOLGW's natural occupations of NH3/cc-pVDZ at this geometry, to the 6
#: decimals MOLGW prints, keyed (reference, relaxation). MOLGW's relaxed
#: density is the iterated-Dyson one: on PBE0 it matches 'dyson' to 6.6e-6
#: and differs from 'zvector' by 4.9e-4; on HF the two coincide.
#: The HF sets agree with 'casida' here to every printed digit, so MOLGW's
#: density fit is equivalent to cc-pvdz-ri (exact integrals would move the
#: occupations by up to 2.7e-5). On PBE0 MOLGW splits the degenerate pairs
#: by up to 5e-6 where this code keeps them to 1e-8, and differs by 3e-6
#: ('none') and 6.6e-6 ('dyson', amplified by the HF-kernel solve, whose
#: lowest eigenvalue on PBE0 is 0.034 Ha): a difference in the PBE0 mean field
#: (its XC grid or SCF convergence), not in the GW density, which agrees
#: exactly on HF.
MOLGW_OCCUPATIONS = {
    ('HF', 'none'): _occupations("""
        1.999849 1.983427 1.974139 1.971642 1.971642 0.016417 0.016417 0.014209
        0.010398 0.006375 0.006375 0.005535 0.005535 0.005464 0.001780 0.001305
        0.001299 0.001299 0.001153 0.001153 0.000912 0.000875 0.000875 0.000550
        0.000550 0.000541 0.000098 0.000098 0.000090"""),
    ('PBE0', 'none'): _occupations("""
        1.999836 1.974083 1.957991 1.954031 1.954027 0.028561 0.028560 0.024332
        0.016228 0.009698 0.009698 0.008103 0.008102 0.008043 0.002656 0.001986
        0.001986 0.001844 0.001711 0.001711 0.001390 0.001294 0.001294 0.000779
        0.000779 0.000778 0.000173 0.000164 0.000164"""),
    ('HF', 'dyson'): _occupations("""
        1.999849 1.983440 1.974183 1.971676 1.971676 0.016417 0.016417 0.014191
        0.010395 0.006375 0.006375 0.005536 0.005536 0.005458 0.001780 0.001305
        0.001283 0.001283 0.001152 0.001152 0.000895 0.000863 0.000863 0.000546
        0.000546 0.000533 0.000098 0.000098 0.000085"""),
    ('PBE0', 'dyson'): _occupations("""
        1.999841 1.973982 1.958333 1.953983 1.953978 0.028662 0.028658 0.024180
        0.016172 0.009648 0.009647 0.008107 0.008106 0.008062 0.002643 0.001987
        0.001987 0.001844 0.001713 0.001713 0.001368 0.001292 0.001292 0.000778
        0.000774 0.000773 0.000162 0.000162 0.000152"""),
}
#: On HF 'zvector' is 'dyson' (the reference's own kernel is the HF kernel).
MOLGW_OCCUPATIONS[('HF', 'zvector')] = MOLGW_OCCUPATIONS[('HF', 'dyson')]
#: max |n_casida - n_MOLGW|: HF to MOLGW's rounding (5e-7, reached 4.9e-7);
#: PBE0 the worst reached ('none' 3.0e-6, 'dyson' 6.6e-6) with a factor ~3.
MOLGW_BAR = {('HF', 'none'): 6e-7, ('HF', 'dyson'): 6e-7, ('HF', 'zvector'): 6e-7,
             ('PBE0', 'none'): 1e-5, ('PBE0', 'dyson'): 2e-5}


def nh3():
    return gto.M(atom=NH3, basis='cc-pvdz', verbose=0)


def mean_field(reference):
    """The converged density-fitted HF or PBE0 mean field of NH3, tagged
    with `reference_name`."""
    mol = nh3()
    mf = scf.RHF(mol) if reference == 'HF' else dft.RKS(mol, xc='pbe0')
    mf = mf.density_fit(auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    mf.kernel()
    assert mf.converged
    mf.reference_name = reference
    return mf


def route_numerics(mf, mode, relaxation):
    """Default quadratures; space-time on the G1 ISDF grid; none for the
    unrelaxed Casida route, which needs none."""
    if mode == 'space-time':
        return g1_numerics(mf)
    if mode == 'casida' and relaxation == 'none':
        return None
    return Numerics()


def mo_density(mf, mode, relaxation):
    """The total GW density on `mode` with `relaxation` in the MO basis,
    C^T S gamma S C, after checking the AO density is symmetric."""
    dm = compute_gw_density_matrix(mf, mode=mode, relaxation=relaxation,
                                   numerics=route_numerics(mf, mode, relaxation))
    assert np.allclose(dm, dm.T, atol=TOL)
    sc = mf.mol.intor('int1e_ovlp') @ mf.mo_coeff
    return sc.T @ dm @ sc


def route_densities(reference, relaxation):
    """{mode: MO-basis density} for every route, with the mean field."""
    mf = mean_field(reference)
    return mf, {mode: mo_density(mf, mode, relaxation) for mode in MODES}


def natural_occupations(gamma_mo):
    """Eigenvalues of the MO-basis density, descending."""
    return np.linalg.eigvalsh(gamma_mo)[::-1]


def check_blocks_match_casida(mf, densities, mode, relaxation, bar):
    """max |gamma_mode - gamma_casida| per block (oo, ov, vv) below `bar`."""
    nocc = mf.mol.nelectron // 2
    o, v = slice(0, nocc), slice(nocc, None)
    diff = np.abs(densities[mode] - densities['casida'])
    errs = {'oo': diff[o, o].max(), 'ov': diff[o, v].max(), 'vv': diff[v, v].max()}
    scale = np.abs(densities['casida'] - np.diag(mf.mo_occ)).max()
    print(f"\nNH3/cc-pVDZ {mf.reference_name} {relaxation!r}: {mode} - casida per block: "
          + ', '.join(f'{k} {e:.2e}' for k, e in errs.items())
          + f'  (max |dgamma| {scale:.2e}, bar {bar:.0e})')
    for block, err in errs.items():
        assert err < bar, f'{block} block: {err:.2e} >= {bar:.0e}'


def check_occupations_physical(mf, densities, mode, relaxation):
    """Sum N, every occupation in [0, 2]; correlation moves density out of
    the reference, so none is exactly 0 or 2, and the nocc largest are the
    valence pairs."""
    occ = natural_occupations(densities[mode])
    nelec = mf.mol.nelectron
    print(f"\nNH3/cc-pVDZ {mf.reference_name}: GW natural occupations "
          f"({mode}, relaxation={relaxation!r})")
    print(np.array2string(occ, precision=8, max_line_width=100))
    print(f"sum {occ.sum():.12f}  (N = {nelec})  "
          f"min {occ.min():.3e}  max {occ.max():.10f}")
    assert abs(occ.sum() - nelec) < SUM_TOL[mode]
    assert occ.min() >= -TOL
    assert occ.max() <= MAX_OCCUPATION + TOL
    assert occ.max() < MAX_OCCUPATION and occ.min() > 0.0
    assert np.all(occ[:nelec // 2] > 1.9)
    assert np.all(occ[nelec // 2:] < 0.1)


def check_occupations_match_casida(mf, densities, mode, relaxation, bar):
    """max |n_mode - n_casida| below `bar`."""
    diff = np.abs(natural_occupations(densities[mode])
                  - natural_occupations(densities['casida'])).max()
    print(f"\nNH3/cc-pVDZ {mf.reference_name} {relaxation!r}: "
          f"max |n_{mode} - n_casida| {diff:.2e}  (bar {bar:.0e})")
    assert diff < bar



def check_casida_matches_molgw(mf, densities, relaxation):
    """max |n_casida - n_MOLGW| below MOLGW_BAR; skipped where MOLGW gives
    no reference (PBE0 'zvector': MOLGW's relaxation is 'dyson')."""
    key = (mf.reference_name, relaxation)
    if key not in MOLGW_OCCUPATIONS:
        pytest.skip(f'no MOLGW occupations for {key}')
    ours = natural_occupations(densities['casida'])
    diff = np.abs(ours - MOLGW_OCCUPATIONS[key]).max()
    print(f"\nNH3/cc-pVDZ {mf.reference_name} {relaxation!r}: "
          f"max |n_casida - n_MOLGW| {diff:.2e}  (bar {MOLGW_BAR[key]:.0e})")
    assert diff < MOLGW_BAR[key]
