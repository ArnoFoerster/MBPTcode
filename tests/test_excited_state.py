"""Gates for src/gradients/excited_state.py -- the BSE@GW surface itself.

Fast tier (seconds to a couple of minutes on H2O, cc-pVDZ): the total
excited-state gradient against finite differences of E_0 + Omega, the energy
decomposition that split rests on, and the quasiparticle gradient -- a
different chain with no BSE and no screening adjoint -- on both a Hartree-Fock
and a Kohn-Sham reference.

Everything computed FROM the surface (optimizer, normal modes, Huang-Rhys
factors, adiabatic gaps, rates) is gated in tests/test_properties.py; the full
workflow is minutes and lives in scripts/gw_bse_gradient_isdf/.
"""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.gradients.excited_state import ExcitedStateChain

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


def scf_factory(mol):
    """A mean field converged for GRADIENT work: conv_tol_grad 1e-11, because
    the Lagrangian assumes the occupied-virtual Fock block vanishes."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def ks_scf_factory(mol):
    """A PBE0 reference, converged the same way. The static correction
    <p|Sigma_x - v_xc|p> is identically zero on Hartree-Fock, so a Kohn-Sham
    reference is the only one that exercises it and its two derivative pieces
    at all."""
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


@pytest.fixture(scope='module')
def water_pbe0():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, ks_scf_factory(mol)


# ------------------------------------------------------------------ gradients
@pytest.mark.parametrize('spin,tda', [('singlet', False), ('triplet', True)])
def test_total_gradient_vs_finite_difference(water, spin, tda):
    """dE_ex/dR = dE_0/dR + dOmega/dR, against a finite difference of E_ex.

    The ground-state gradient is added HERE and never inside the chain, and
    this is what proves the split: the reference differences the SUM.
    """
    mol, mf = water
    ch = ExcitedStateChain(mol, scf_factory, spin=spin, bse_tda=tda, mf=mf)
    g, _, _ = ch.total_gradient()
    h, worst = 1e-4, 0.0
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                d = np.zeros((mol.natm, 3))
                d[ia, x] = k * h
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(ch.energy(m)[0])
            fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
            worst = max(worst, abs(fd - g[ia, x]))
    assert worst / np.abs(g).max() < 1e-6


def test_energy_decomposition_is_consistent(water):
    """E_ex = E_0 + Omega exactly, and E_0 is the mean field's own energy."""
    mol, mf = water
    ch = ExcitedStateChain(mol, scf_factory, mf=mf)
    e_ex, e_0, om = ch.energy()
    assert e_0 == pytest.approx(mf.e_tot, abs=1e-12)
    assert e_ex == pytest.approx(e_0 + om, abs=1e-12)
    assert om == pytest.approx(ch.excitation(), abs=1e-12)


# --------------------------------------------------------- quasiparticles
def test_the_davidson_route_gives_the_dense_gradient(water):
    """The matrix-free Casida solver and the dense one give the SAME force.

    This is what makes the route usable past the size where a dense (n_ov, n_ov)
    block can be formed at all. The adjoint never sees the choice: `bse_cache`
    hands `bse_backward` the same three-index blocks either way, and the rank-two
    structure of the Casida density means nothing pair-space is built in the
    reverse pass. What the solver decides is only WHICH eigenvector is handed
    over, so agreement here is a statement about the eigenvectors, and the
    residual is the Davidson tolerance rather than anything in the chain.

    Measured here on water/cc-pVDZ: the excitation energies agree to 6e-17 Ha
    and the forces to 1.5e-08 Ha/Bohr on a gradient of 0.11, i.e. 1.3e-07
    relative. That is the PLATEAU of this comparison rather than a convergence
    of it -- tightening the Davidson tolerance does not move it, because it is
    the fit adjoint's own reproducibility floor, not the eigensolver.
    """
    mol, mf = water
    dense = ExcitedStateChain(mol, scf_factory, mf=mf, solver='dense')
    davidson = ExcitedStateChain(mol, scf_factory, mf=mf, solver='davidson',
                                 nroots=5)
    assert davidson.solver_used(10 ** 6) == 'davidson'
    om_d, om_v = dense.excitation(), davidson.excitation()
    assert om_d == pytest.approx(om_v, abs=1e-8)
    g_d = dense.excitation_gradient()[0]
    g_v = davidson.excitation_gradient()[0]
    assert np.abs(g_d - g_v).max() < 1e-7    # the floor is 1.5e-08
    assert np.abs(g_d).max() > 1e-3          # there is a force to compare


def test_quasiparticle_gradient_vs_finite_difference(water):
    """d eps^QP/dR for the HOMO -- a different chain from the excitation one:
    no BSE adjoint, no screening adjoint, one state instead of a set."""
    mol, mf = water
    ch = ExcitedStateChain(mol, scf_factory, mf=mf)
    g, d = ch.quasiparticle_gradient(0)
    h, worst = 1e-4, 0.0
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                dsp = np.zeros((mol.natm, 3))
                dsp[ia, x] = k * h
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + dsp, unit='Bohr')
                m.build(False, False)
                v.append(ch.quasiparticle(0, m))
            fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
            worst = max(worst, abs(fd - g[ia, x]))
    assert worst / np.abs(g).max() < 1e-6
    assert d['translation_residual'] < 1e-10


def test_quasiparticle_gradient_vs_finite_difference_on_pbe0(water_pbe0):
    """The same chain on a KOHN-SHAM reference, where the static correction is
    not zero: <HOMO|Sigma_x - v_xc|HOMO> = -0.190 Ha at PBE0/H2O/cc-pVDZ.

    Every other gate here is Hartree-Fock, where that term and both of its
    derivative pieces vanish identically -- so all of them pass with the
    correction missing, misweighted, or carrying 1 instead of the pole strength
    Z, which is a 10-20% error on the force. This is the only one that measures
    them.

    ONE Cartesian component, because each finite difference is four Kohn-Sham
    SCFs. The residual is the exchange-correlation quadrature, not the chain:
    it does not move with the step size (2.1e-7 Ha/Bohr at h = 1e-4, 3e-4 and
    1e-3 alike) and falls to 6e-9 on a level-6 grid, so 1e-5 here is a grid
    tolerance five orders looser than the defect it guards against.
    """
    mol, mf = water_pbe0
    ch = ExcitedStateChain(mol, ks_scf_factory, mf=mf)
    g, _ = ch.quasiparticle_gradient(0)
    ia, x, h = 0, 2, 1e-4                       # the oxygen along the C2 axis
    v = []
    for k in (-2, -1, 1, 2):
        dsp = np.zeros((mol.natm, 3))
        dsp[ia, x] = k * h
        m = mol.copy()
        m.set_geom_(mol.atom_coords() + dsp, unit='Bohr')
        m.build(False, False)
        v.append(ch.quasiparticle(0, m))
    fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    assert abs(fd - g[ia, x]) / np.abs(g).max() < 1e-5
