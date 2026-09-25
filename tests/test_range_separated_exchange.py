"""Range-separated hybrids in the skeleton derivatives.

pyscf builds K_eff = hyb K_full + (alpha - hyb) K_lr(omega), and the whole
gradient chain used to refuse an RSH because every skeleton weighted ONE
full-range exchange build by `xc_hybrid_coeff`. That scalar is ALPHA, so the
refusal was protecting against two errors at once: a dropped beta K_lr(omega)
term AND a wrong weight on the term it kept -- 1.0 on LRC-wPBEh where the
full-range weight is 0.2.

WHAT EACH GATE IS FOR:

- the channel decomposition must reproduce the SCF's OWN exchange operator, to
  machine precision, on a global hybrid and on two range-separated ones. Every
  skeleton is built from it, so it is the single point of failure.
- the long-range channel is NOT density fitted, and that is forced: its
  auxiliary metric (P|erf(omega r)/r|Q) goes rank-deficient -- cc-pVDZ-JKFIT on
  water keeps 62 of 116 functions at omega = 0.2, smallest eigenvalue NEGATIVE
  -- and the derivative of a truncated pseudo-inverse is not -V^-1 dV V^-1.
  Fitted, the long-range skeleton is wrong by 45%; on four-centre integrals it
  is exact.
- a GLOBAL hybrid must be untouched by all of this, which is what the
  single-channel path asserts.
- the ISDF mean-field force carries an RSH too, and its translation residual
  is the tell: it sat at 6.6e-12 while the exchange-free reference still
  removed the wrong exchange, and fell to 6.3e-15 -- machine precision, the
  same as a global hybrid -- once it removed the right one. A force that sums
  to zero over the atoms is not thereby correct, but one that stops doing so
  by three orders of magnitude has changed.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.isdf_jk import isdf_jk
from src.gradients.isdf_derivatives import (exchange_channel_skeleton,
                                            exchange_channels, rsh_split,
                                            xc_hybrid_coeff)
from src.gradients.isdf_mean_field import (isdf_mean_field_gradient,
                                           require_isdf_gradient_support)

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: Displacement for the finite differences below, in Bohr.
STEP = 1e-4


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=WATER, basis='sto-3g', verbose=0)


def rks(mol, xc):
    mf = dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


@pytest.mark.parametrize('xc', ('b3lyp', 'pbe0', 'camb3lyp', 'wb97x',
                                'lrc-wpbeh'))
def test_channels_reproduce_the_scf_exchange_operator(mol, xc):
    """sum_ch w K(omega_ch) IS what `get_veff` built, to machine precision."""
    mf = rks(mol, xc)
    dm = mf.make_rdm1()
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    want = hyb * mf.get_k(mol, dm)
    if omega != 0:
        want = want + (alpha - hyb) * mf.get_k(mol, dm, omega=omega)
    got = sum(w * (mf.get_k(mol, dm) if o == 0 else mf.get_k(mol, dm, omega=o))
              for o, w in exchange_channels(mf))
    assert np.abs(got - want).max() < 1e-12


def test_hybrid_coeff_is_alpha_and_must_not_be_used_as_a_weight(mol):
    """The trap the refusal was really guarding, pinned so it cannot return."""
    mf = rks(mol, 'lrc-wpbeh')
    assert xc_hybrid_coeff(mf)[1] == pytest.approx(1.0)
    channels = dict((round(o, 6), w) for o, w in exchange_channels(mf))
    assert channels[0.0] == pytest.approx(0.2)      # NOT 1.0
    assert channels[0.2] == pytest.approx(0.8)
    omega, alpha, beta = rsh_split(mf)
    assert alpha + beta == pytest.approx(channels[0.0])
    assert -beta == pytest.approx(channels[0.2])


@pytest.mark.parametrize('omega,weight', ((0.2, 0.8), (0.33, 0.46), (1.0, 1.0)))
def test_exchange_channel_skeleton_against_exact_integrals(mol, omega, weight):
    """One channel's derivative, against a finite difference of its own energy.

    E_x = -(w/2) sum_abcd gamma_ac D_bd (ab|cd) with the erf-attenuated
    operator. No fit in the reference and none in the routine.
    """
    mf = rks(mol, 'b3lyp')
    rng = np.random.default_rng(2)
    n = mol.nao_nr()
    gam = rng.normal(size=(n, n))
    gam = 0.5 * (gam + gam.T)
    g_ao = mf.mo_coeff @ gam @ mf.mo_coeff.T
    D = mf.make_rdm1()

    def energy(m):
        with m.with_range_coulomb(omega):
            eri = m.intor('int2e', aosym='s1')
        return -0.5 * weight * float(np.einsum('ac,bd,abcd->', g_ao, D, eri,
                                               optimize=True))

    ana = exchange_channel_skeleton(mf, g_ao, D, omega, weight)
    crd = np.asarray(mol.atom_coords())
    fd = np.zeros_like(ana)
    for atom in range(mol.natm):
        for axis in range(3):
            v = []
            for sign in (-1.0, 1.0):
                c = crd.copy()
                c[atom, axis] += sign * STEP
                v.append(energy(gto.M(
                    atom=[(mol.atom_symbol(i), tuple(c[i]))
                          for i in range(mol.natm)],
                    basis='sto-3g', unit='Bohr', verbose=0)))
            fd[atom, axis] = (v[1] - v[0]) / (2 * STEP)
    assert np.abs(ana - fd).max() / np.abs(fd).max() < 1e-7


def test_long_range_auxiliary_metric_is_rank_deficient(mol):
    """Why the long-range channel may never be density fitted.

    Not a property of this code -- a property of the operator. It is asserted
    so that a future attempt to fit that channel fails here first.
    """
    from pyscf import df as pyscf_df
    auxmol = pyscf_df.addons.make_auxmol(mol, 'cc-pvdz-jkfit')
    ranks = {}
    for omega in (0.0, 0.2):
        with auxmol.with_range_coulomb(omega):
            v = auxmol.intor('int2c2e', aosym='s1')
        w = np.linalg.eigvalsh(0.5 * (v + v.T))
        ranks[omega] = int((w > 1e-10 * w.max()).sum())
    assert ranks[0.0] == auxmol.nao_nr()
    assert ranks[0.2] < 0.7 * auxmol.nao_nr()


@pytest.mark.parametrize('xc', ('b3lyp', 'camb3lyp', 'lrc-wpbeh'))
def test_isdf_mean_field_force_on_a_range_separated_hybrid(mol, xc):
    """The whole ISDF force against a finite difference of its own energy.

    The factorization is rebuilt at each displaced geometry, so what is
    differenced is the same realization of the estimator the analytic route
    differentiates. The translation residual is checked too: it is what caught
    the exchange-free reference removing the wrong exchange, sitting at 6.6e-12
    where a correct force gives 1e-15.
    """
    def scf(m):
        mf = isdf_jk(dft.RKS(m, xc=xc), auxbasis='cc-pvdz-ri')
        mf.grids.prune = None
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
        mf.kernel()
        return mf

    mf = scf(mol)
    require_isdf_gradient_support(mf, 'the ISDF mean-field gradient')
    ana = isdf_mean_field_gradient(mf)
    crd = np.asarray(mol.atom_coords())
    fd = np.zeros_like(ana)
    for atom in range(mol.natm):
        for axis in range(3):
            v = []
            for sign in (-1.0, 1.0):
                c = crd.copy()
                c[atom, axis] += sign * STEP
                v.append(scf(gto.M(
                    atom=[(mol.atom_symbol(i), tuple(c[i]))
                          for i in range(mol.natm)],
                    basis='sto-3g', unit='Bohr', verbose=0)).e_tot)
            fd[atom, axis] = (v[1] - v[0]) / (2 * STEP)
    assert np.abs(ana - fd).max() < 1e-7
    assert np.abs(ana.sum(axis=0)).max() < 1e-13
