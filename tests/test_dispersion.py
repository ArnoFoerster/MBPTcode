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
from pyscf import dft, gto, scf

from src.Base.dispersion import (DISPERSION_VERSIONS, dispersion_energy,
                                 dispersion_gradient, dispersion_version,
                                 refuse_dispersion_under_rpa)
from src.Base.isdf_jk import isdf_jk
from src.SingleReference.LinearResponse.rpa_energy import reference_energy
from src.gradients.dense_surfaces import kohn_sham_gradient_correction
from src.gradients.isdf_derivatives import (
    exx_double_counting, exx_double_counting_skeleton, rsh_split)
from src.gradients.rpa_ground_state import RPAGroundStateChain

#: Fragments in van der Waals contact, which is the case dispersion decides.
ETHANE = ('C 0 0 0; C 0 0 1.53; H 0 1.02 -0.36; H 0.88 -0.51 -0.36; '
          'H -0.88 -0.51 -0.36; H 0 -1.02 1.89; H 0.88 0.51 1.89; '
          'H -0.88 0.51 1.89')
WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: C1, so that no symmetry zeroes a component a missing term would show in.
H2O_C1 = 'O 0.03 0.02 0.117; H 0.10 0.757 -0.468; H -0.05 -0.80 -0.40'
#: The dRPA force gate's step pair, in Bohr: the finer step's h^2 error, 3.6e-8
#: Ha/Bohr on H2O_C1, has to stand above the interpolated route's 1.5e-8 floor.
FD_STEPS = (1e-3, 5e-4)
#: Central-difference step for the fixed-density skeleton gate, in Bohr.
STEP = 1e-4


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


def test_the_exact_exchange_repair_carries_every_range_separated_channel():
    """E_KS -> E_HF at one density, and its skeleton, on range-separated hybrids.

    The repair adds back each exact-exchange channel E_xc holds, the
    erf-attenuated one included; one full-range build weighted by alpha, the
    scalar `xc_hybrid_coeff` returns, misses E_HF by 6.3 Ha on LRC-wPBEh. The
    skeleton is the repair's derivative at fixed density with the grid moving
    with the atoms, so the dRPA ground state has a force on such a reference.
    """
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    for xc in ('pbe0', 'b3lyp', 'lrc-wpbeh', 'wb97x'):
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level, mf.conv_tol = 5, 1e-12
        mf.kernel()
        got = mf.e_tot + exx_double_counting(mf, mol)
        want = scf.RHF(mol).energy_tot(dm=mf.make_rdm1())
        assert got == pytest.approx(want, abs=1e-10), xc
    assert rsh_split(mf)[0] > 0.0
    crd = mol.atom_coords()
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for sign in (-1.0, 1.0):
                c = crd.copy()
                c[ia, x] += sign * STEP
                m = gto.M(atom=[(mol.atom_symbol(i), tuple(c[i]))
                                for i in range(mol.natm)],
                          basis='cc-pvdz', unit='Bohr', verbose=0)
                # the same AO density, so only the integrals and the grid move
                moved = dft.RKS(m, xc=mf.xc)
                moved.grids.level = mf.grids.level
                moved.mo_coeff, moved.mo_occ = mf.mo_coeff, mf.mo_occ
                v.append(exx_double_counting(moved, m))
            fd[ia, x] = (v[1] - v[0]) / (2.0 * STEP)
    assert np.abs(exx_double_counting_skeleton(mf, mol) - fd).max() < 1e-8


def isdf_kohn_sham(xc, level):
    """A factory of converged ISDF-exchange Kohn-Sham mean fields at XC grid `level`.

    conv_tol_grad 1e-11 because the Lagrangian assumes F_ia = 0. Every
    displaced SCF starts from the first one's density, which takes LRC-wPBEh at
    level 6 from many more cycles to fewer and lands on the same minimum.
    """
    guess = []

    def factory(m):
        mf = isdf_jk(dft.RKS(m, xc=xc), auxbasis='cc-pvdz-ri')
        mf.grids.level = level
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
        mf.kernel(dm0=guess[0] if guess else None)
        assert mf.converged
        if not guess:
            guess.append(mf.make_rdm1())
        return mf
    return factory


def richardson_force(chain, mol, steps=FD_STEPS):
    """(D_R, truncation): the Richardson limit of central differences of the
    chain's REPORTED E_0 at two steps, on every Cartesian component, and the
    finer step's own h^2 error, (D(h1) - D(h2)) / ((h1/h2)^2 - 1), at its largest.
    """
    diff = {}
    for h in steps:
        diff[h] = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            for x in range(3):
                e = []
                for sign in (-1.0, 1.0):
                    shift = np.zeros((mol.natm, 3))
                    shift[ia, x] = sign * h
                    m = mol.copy()
                    m.set_geom_(mol.atom_coords() + shift, unit='Bohr')
                    m.build(False, False)
                    e.append(chain.energy(m)[0])
                diff[h][ia, x] = (e[1] - e[0]) / (2.0 * h)
    h1, h2 = steps
    ratio = (h1 / h2) ** 2
    return ((ratio * diff[h2] - diff[h1]) / (ratio - 1.0),
            np.abs(diff[h1] - diff[h2]).max() / (ratio - 1.0))


def test_the_reported_hartree_fock_energy_is_the_mean_fields_own():
    """E_HF[rho] on the interaction each mean field converged with.

    An ISDF Kohn-Sham reference is given its own ISDFJK, so E_HF is E_KS plus
    the double counting built from the same operators, the functional the
    gradient pair differentiates; a fresh density fit of the same auxiliary
    basis is 5.5e-4 Ha away on this water. Hartree-Fock on ISDF reports its own
    e_tot and a fitted Kohn-Sham reference its fitted mirror, both bitwise, and
    on Hartree-Fock the gradient pair is exactly zero.
    """
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    nocc = mol.nelectron // 2
    for xc in ('pbe0', 'lrc-wpbeh'):
        mf = isdf_jk(dft.RKS(mol, xc=xc), auxbasis='cc-pvdz-ri')
        mf.kernel()
        want = mf.e_tot + exx_double_counting(mf, mol)
        assert reference_energy(mf, mol) == pytest.approx(want, abs=1e-12), xc
    hf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri')
    hf.kernel()
    assert reference_energy(hf, mol) == float(hf.e_tot)
    y, g = kohn_sham_gradient_correction(mol, hf, nocc)
    assert not np.any(y) and not np.any(g)
    fitted = dft.RKS(mol, xc='lrc-wpbeh').density_fit(auxbasis='cc-pvdz-ri')
    fitted.kernel()
    mirror = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    assert reference_energy(fitted, mol) == float(
        mirror.energy_tot(dm=fitted.make_rdm1()))


@pytest.mark.parametrize('xc, level', [('pbe0', 3), ('lrc-wpbeh', 6)])
def test_the_drpa_force_follows_its_reported_energy_on_an_interpolated_kohn_sham_reference(
        xc, level):
    """dE_0/dR against a difference of the E_0 the chain REPORTS, per component.

    E_0 takes E_HF from the mean field's own ISDF exchange. With a
    density-fitted mirror in its place the force missed the reported surface by
    9.46e-4 Ha/Bohr on both functionals; now the Richardson limit is met to
    1.5e-8 on PBE0 at XC grid level 3, where a density-fitted PBE0 meets its
    own to 2.0e-8, inside the finer step's measured truncation of 3.6e-8.

    LRC-wPBEh is gated at level 6. At level 3 it misses by 3.7e-7, and a
    density-fitted reference by 3.9e-7: the XC grid response the force does
    not carry, not the reference energy, and at level 6 the miss is 1.7e-8.
    """
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    factory = isdf_kohn_sham(xc, level)
    chain = RPAGroundStateChain(mol, factory, mf=factory(mol))
    g, e, _ = chain.total_gradient()
    assert e == pytest.approx(chain.energy()[0], abs=1e-12)
    fd, truncation = richardson_force(chain, mol)
    assert np.abs(g - fd).max() < truncation, (np.abs(g - fd).max(), truncation)

