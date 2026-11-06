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
- the numerical routes reproduce the Casida occupations.

The block comparison is the sharper test of the two: an error in the density
moves the occupations only at second order through its off-diagonal elements
(PBE0 'none', space-time: 5.5e-6 in the ov block, 6.1e-7 in the occupations),
and the occupations do not see the natural orbitals at all.

DF on cc-pvdz-ri; default quadratures (W grid minimax 20, outer
Gauss-Legendre 48); space-time on the G1 ISDF grid.
"""
import numpy as np
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

