"""Gates for the open-shell (UHF) EE-ADC route.

The ADC equations are spin-orbital and assume only a canonical reference, so
the open-shell route is the SAME code fed different arrays: block-stacked
[occ_a, occ_b, virt_a, virt_b] spin orbitals, which is the ordering that
keeps the occupied/virtual split contiguous when nocc_a != nocc_b.

Two things have to hold, and the second is the one that bites:

  1. A closed-shell molecule run through UHF must give the RHF spectrum.
     That pins the block-stacked plumbing without needing any reference.
  2. The Delta-Ms = 0 sector must be imposed. Sz commutes with H, so an
     unrestricted spin-orbital space also carries the Ms-changing sectors --
     including the one that maps the reference onto its OWN other Ms
     component, which shows up as a spurious ~0 eV root. Restricting to the
     spin-conserving sector is what makes the spectrum match pyscf's UADC.
"""
import numpy as np
import pytest

from pyscf import gto, scf, adc

from src.SingleReference.ADC.eeADC import ee_utils
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

RADICALS = [('O 0 0 0; H 0 0 0.97', 1, 'OH'),
            ('N 0 0 0; H 0 0 1.04; H 1.0 0 -0.3', 1, 'NH2')]
CLOSED = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6']


@pytest.mark.parametrize('atom', CLOSED)
@pytest.mark.parametrize('level', ['adc2', 'adc2x', 'adc3'])
def test_uhf_reproduces_rhf_on_closed_shell(atom, level):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    r, u = scf.RHF(mol).run(), scf.UHF(mol).run()
    assert abs(r.e_tot - u.e_tot) < 1e-8
    # BOTH in the Delta Ms = 0 sector, so the comparison is like for like:
    # the RHF spin-orbital space otherwise also carries the Ms = +-1 partners
    # of every triplet, and its lowest four roots would then be four copies
    # of one state rather than four states.
    kw = dict(level=level, nroots=4, route='spinorbital', matrix_free=False,
              ms_sector=0)
    er = np.sort(solve_ee_adc(r, **kw)[0])
    eu = np.sort(solve_ee_adc(u, **kw)[0])
    # tolerance is set by the two SCF solutions, which agree in energy to
    # 1e-8 but whose orbitals differ by ~1e-6
    assert np.abs(er - eu).max() < 1e-5


@pytest.mark.parametrize('atom,spin,label', RADICALS)
@pytest.mark.parametrize('level,method', [('adc2', 'adc(2)'),
                                          ('adc2x', 'adc(2)-x'),
                                          ('adc3', 'adc(3)')])
def test_open_shell_matches_pyscf_uadc(atom, spin, label, level, method):
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    a = adc.ADC(mf); a.method = method; a.method_type = 'ee'; a.verbose = 0
    ref = np.sort(np.array(a.kernel(nroots=4)[0]))
    e = np.sort(solve_ee_adc(mf, level=level, nroots=8, route='spinorbital',
                             matrix_free=False)[0])
    for r in ref:
        assert np.abs(e - r).min() < 1e-8, (label, level, r)


@pytest.mark.parametrize('atom,spin,label', RADICALS)
@pytest.mark.parametrize('level', ['adc2', 'adc2x', 'adc3'])
def test_ms_mask_block_diagonalizes_exactly(atom, spin, label, level):
    """The invariant behind the sector restriction: Sz commutes with H, so
    the mask must split the supermatrix with ZERO coupling. If it did not,
    the restricted spectrum would not be a subset of the full one and the
    'sector' would be a silent approximation."""
    from src.SingleReference.ADC.eeADC import ee_u_dense_full as ppd
    from src.SingleReference.ADC.eeADC.ee_driver import spin_orbital_arrays
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    eps, g, nocc = spin_orbital_arrays(mf, mol)
    norb = len(eps)
    sz = ee_utils.spin_labels(mf, nocc, norb)
    mask = ee_utils.ms_sector_mask(sz, nocc, norb, 0)
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    assert np.abs(H[np.ix_(mask, ~mask)]).max() == 0.0
    assert 0 < mask.sum() < len(mask)


@pytest.mark.parametrize('atom,spin,label', RADICALS)
def test_ms_sector_removes_spurious_near_zero_roots(atom, spin, label):
    """Without the restriction the spectrum carries roots that are not
    excitations -- Ms-changing configurations, including ones that map the
    reference onto its own other Ms component."""
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    kw = dict(level='adc2', nroots=8, route='spinorbital', matrix_free=False)
    restricted = np.sort(solve_ee_adc(mf, **kw)[0])
    full = np.sort(solve_ee_adc(mf, ms_sector=None, **kw)[0])
    assert (np.abs(full) < 1e-2).sum() > (np.abs(restricted) < 1e-2).sum()


@pytest.mark.parametrize('atom,spin,label', RADICALS)
def test_spin_labels_and_mask(atom, spin, label):
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    eps, g, nocc = __import__(
        'src.SingleReference.ADC.eeADC.ee_driver', fromlist=['x']
    ).spin_orbital_arrays(mf, mol)
    norb = len(eps)
    sz = ee_utils.spin_labels(mf, nocc, norb)
    na, nb = mf.nelec
    assert (sz[:na] == 1).all() and (sz[na:nocc] == -1).all()
    assert sz.sum() == (na - nb) + (mf.mo_coeff[0].shape[1] - na) - \
        (mf.mo_coeff[1].shape[1] - nb)
    mask = ee_utils.ms_sector_mask(sz, nocc, norb, 0)
    assert mask.any() and not mask.all()


def test_spinfree_route_refuses_uhf():
    mol = gto.M(atom=RADICALS[0][0], basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).run()
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc2', nroots=1, route='spinfree')
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc2', nroots=1, route='spinorbital',
                     spin='singlet')
