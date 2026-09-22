"""The empirical dispersion correction: what it moves, and who may carry one.

The term is a function of the nuclear coordinates alone, so it must move the
potential energy surface and leave every orbital, and therefore every
excitation energy, exactly where it was. And it must never be added to a
direct-RPA ground state, whose correlation energy already contains dispersion.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.dispersion import (DISPERSION_VERSIONS, dispersion_energy,
                                 dispersion_gradient, dispersion_version,
                                 refuse_dispersion_under_rpa)
from src.gradients.rpa_ground_state import RPAGroundStateChain

#: Fragments in van der Waals contact, which is the case dispersion decides.
ETHANE = ('C 0 0 0; C 0 0 1.53; H 0 1.02 -0.36; H 0.88 -0.51 -0.36; '
          'H -0.88 -0.51 -0.36; H 0 -1.02 1.89; H 0.88 0.51 1.89; '
          'H -0.88 0.51 1.89')


def backend():
    """Skip where the D3/D4 libraries are not installed; the parsing and the
    refusal below need no backend and always run."""
    pytest.importorskip('pyscf.dispersion',
                        reason='pip install pyscf-dispersion')


def mean_field(xc, basis='sto-3g'):
    mol = gto.M(atom=ETHANE, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    return mf


def test_the_functional_name_carries_the_choice():
    assert 'd4' in DISPERSION_VERSIONS and 'd3bj' in DISPERSION_VERSIONS
    mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    assert dispersion_version(dft.RKS(mol, xc='pbe0')) is None
    assert dispersion_version(dft.RKS(mol, xc='pbe0-d4')) == 'd4'
    assert dispersion_version(dft.RKS(mol, xc='b3lyp-d3bj')) == 'd3bj'
    # a Hartree-Fock reference has no xc at all and carries nothing
    from pyscf import scf
    assert dispersion_version(scf.RHF(mol)) is None


def test_it_moves_the_surface_and_not_the_spectrum():
    """The term depends on the nuclei alone, so the orbitals must be bitwise
    unchanged: an excitation energy computed on either reference is the same
    number, and only the total energy moves."""
    backend()
    plain, corrected = mean_field('pbe0'), mean_field('pbe0-d4')
    e_disp = dispersion_energy(corrected)
    assert e_disp < 0.0
    assert corrected.e_tot - plain.e_tot == pytest.approx(e_disp, abs=1e-12)
    assert np.array_equal(corrected.mo_energy, plain.mo_energy)
    assert np.array_equal(corrected.mo_coeff, plain.mo_coeff)
    assert dispersion_energy(plain) == 0.0


def test_the_dispersion_force_is_the_difference_of_the_two_gradients():
    """And it is translation invariant, being a sum over interatomic distances."""
    backend()
    plain, corrected = mean_field('pbe0'), mean_field('pbe0-d4')
    g = dispersion_gradient(corrected)
    assert g.shape == (plain.mol.natm, 3)
    assert np.abs(g.sum(axis=0)).max() < 1e-12
    from pyscf.grad import rks as rks_grad
    got = (np.asarray(rks_grad.Gradients(corrected).kernel())
           - np.asarray(rks_grad.Gradients(plain).kernel()))
    assert got == pytest.approx(g, abs=1e-10)
    assert np.array_equal(dispersion_gradient(plain), np.zeros_like(g))


def test_direct_rpa_refuses_a_dispersion_corrected_reference():
    """dRPA already contains dispersion. The gradient is the sharper reason:
    the Kohn-Sham-to-Hartree-Fock skeleton differences two pyscf gradients at
    one density and only the Kohn-Sham one carries the dispersion force, so
    the cancellation tears and a force survives with the wrong sign."""
    backend()                      # a converged reference is needed to get past
    corrected = mean_field('pbe0-d4')   # the chain's own quality check
    mol = corrected.mol
    with pytest.raises(ValueError, match='already contains dispersion'):
        RPAGroundStateChain(mol, lambda m: corrected, auxbasis='sto-3g')
    with pytest.raises(ValueError, match='counted twice'):
        refuse_dispersion_under_rpa(corrected, 'a caller')
    # the skeleton guards itself, for a chain built before the reference moved
    from src.gradients.isdf_derivatives import exx_double_counting_skeleton
    with pytest.raises(ValueError, match='skeleton'):
        exx_double_counting_skeleton(corrected, mol)
    # and a plain functional passes both
    refuse_dispersion_under_rpa(mean_field('pbe0'), 'a caller')


def test_a_functional_fitted_with_dispersion_refuses_a_second_helping():
    """Unlike the direct-RPA case, nothing downstream would notice this one:
    the correction would simply be added twice and the geometry would come out
    over-bound."""
    from src.Base.dispersion import (functional_carries_dispersion,
                                     refuse_double_dispersion)
    for name in ('wB97X-D3', 'wB97X-D4', 'wB97X-V', 'wB97M-V', 'r2SCAN-3c',
                 'B97-3c'):
        assert functional_carries_dispersion(name), name
        with pytest.raises(ValueError, match='already includes dispersion'):
            refuse_double_dispersion(name, 'D3BJ')
        refuse_double_dispersion(name, None)      # the functional alone is fine
        refuse_double_dispersion(name, 'none')
    for name in ('B3LYP', 'PBE0', 'wB97X', 'M06-2X'):
        assert not functional_carries_dispersion(name), name
        refuse_double_dispersion(name, 'D3BJ')    # these need the correction


def test_the_pcm_factorization_changes_nothing_it_computes():
    """K is built from the cavity and the dielectric constant alone, so it does
    not move while the density iterates and its factorization belongs outside
    the loop. The gate is against pyscf's own unpatched routine, so a change
    upstream shows as a disagreement rather than as silence."""
    from pyscf.solvent import pcm

    from src.Base.pcm_factorization import factorize_once

    mol = gto.M(atom='O 0 0 0; C 0 0 1.21; H 0 0.94 1.80; H 0 -0.94 1.80',
                basis='def2-svp', verbose=0)
    dm = np.asarray(dft.RKS(mol, xc='b3lyp').get_init_guess())
    plain, fast = pcm.PCM(mol), pcm.PCM(mol)
    for obj in (plain, fast):
        obj.eps, obj.method, obj.lebedev_order, obj.verbose = 2.374, 'C-PCM', 29, 0
        obj.build()
    assert factorize_once(fast) is fast
    assert factorize_once(fast) is fast          # idempotent

    e0, v0 = plain._get_vind(dm)
    e1, v1 = fast._get_vind(dm)
    assert e1 == pytest.approx(e0, abs=1e-14)
    assert np.abs(v1 - v0).max() < 1e-12

    # a rebuild replaces K, and the cache must not survive it
    before = fast._intermediates['K']
    fast.eps = 78.4
    fast.build()
    assert fast._intermediates['K'] is not before
    e2, _ = fast._get_vind(dm)
    plain.eps = 78.4
    plain.build()
    e3, _ = plain._get_vind(dm)
    assert e2 == pytest.approx(e3, abs=1e-14)
    assert e2 != pytest.approx(e0, abs=1e-6)     # the new eps really took


def test_the_drpa_ground_state_energy_is_exact_on_a_range_separated_reference():
    """`exx_double_counting` sums one exact-exchange build per channel
    (`exchange_channels`), so a range-separated hybrid's erf-attenuated term
    and its full-range term are both added back and E_0 is exact on it too,
    the same as on a global hybrid."""
    from pyscf import scf

    from src.gradients.isdf_derivatives import exx_double_counting

    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='cc-pvdz',
                verbose=0)
    hf = scf.RHF(mol)
    range_separated = {'lrc-wpbeh', 'wb97x'}
    for xc in ('pbe0', 'b3lyp', 'lrc-wpbeh', 'wb97x'):
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level, mf.conv_tol = 5, 1e-12
        mf.kernel()
        if xc in range_separated:
            assert mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)[0] > 0.0
        got = mf.e_tot + exx_double_counting(mf, mol)
        assert got == pytest.approx(hf.energy_tot(dm=mf.make_rdm1()), abs=1e-10)


