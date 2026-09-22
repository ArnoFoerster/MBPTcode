"""Fast gates for src/gradients (Toelle-route analytic GW/BSE gradients).

H4/sto-3g tier (seconds): the closed-form RPA layer against wicks
production RPA, the diagonal qb-EOM QP solve against the dense supermatrix
and the production graphical G0W0, the static BSE kernel against
static_screened_coulomb_chemist, and a parameter-space FD gate on the
RPA partials. The full nuclear-gradient FD battery (minutes) lives in
scripts/gw_bse_toelle_gradients/.
"""
import numpy as np
import pytest
from pyscf import gto, scf, ao2mo

from src.Base.constants import KAPPA
from src.gradients.qb_core import build_rpa_AB, RPA
from src.gradients.qp_qb import QPqb
from src.gradients.bse_qb import BSEqb
from src.Base.eri_blocks import MOEriBlocks
from src.gradients.targets import rpa_partials, qp_partials
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_screened_coulomb_chemist)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.rpa_energy import rpa_correlation_energy_casida


@pytest.fixture(scope='module')
def h4():
    mol = gto.M(atom='H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9',
                unit='Bohr', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    norb = mol.nao
    nocc = mol.nelectron // 2
    eri = ao2mo.general(mol, (mf.mo_coeff,) * 4, compact=False).reshape((norb,) * 4)
    return mol, mf, eri, nocc, norb


def test_rpa_closed_form(h4):
    mol, mf, eri, nocc, norb = h4
    A, B, _ = build_rpa_AB(mf.mo_energy, eri, nocc)
    qb = RPA(A, B)
    assert np.abs(qb.residual()).max() < 1e-12          # J(t) = 0
    assert abs(qb.e_corr() - qb.e_corr_plasmon()) < 1e-13
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=eri, spin_mode='restricted')
    assert abs(qb.e_corr() - rpa_correlation_energy_casida(lr, nocc)) < 1e-11
    Aw, Bw = lr.build_casida_matrices(nocc, lBSE=False)
    om_wicks, _, _ = CasidaSolver(Aw, Bw).solve()
    assert np.abs(np.sort(qb.omega) - np.sort(om_wicks)).max() < 1e-10


def test_qp_vs_dense_and_wicks(h4):
    mol, mf, eri, nocc, norb = h4
    from src.SingleReference.GW.self_energy import SelfEnergySolver
    from src.Solvers.qp_equation import solve_qp_equation_newton
    qp = QPqb(mf.mo_energy, eri, nocc, screening='rpa')
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=eri,
                              spin_mode='restricted', eta=1e-9)
    Aw, Bw = lr.build_casida_matrices(nocc, lBSE=False)
    om, X, Y = CasidaSolver(Aw, Bw).solve()
    se = SelfEnergySolver(mf.mo_energy, eri_chemist=eri,
                          spin_mode='restricted', eta=1e-9)
    for p in range(norb):
        w_mine, Z = qp.solve_diag(p)
        w_dense, _ = qp.solve_diag_dense(p)
        assert abs(w_mine - w_dense) < 1e-10
        chi_a = se.get_chi_a(nocc, X, Y, p_state=p)
        func = lambda w: w - mf.mo_energy[p] - np.real(
            se.calculate_self_energy(p, w, nocc, om, chi_a, None, vertex_mode='GW'))
        w_wicks = solve_qp_equation_newton(func, mf.mo_energy[p], tol=1e-12,
                                           max_iter=200)
        assert abs(w_mine - w_wicks) < 1e-9


@pytest.fixture(scope='module')
def h4_bhlyp(h4):
    """The same geometry on a Kohn-Sham reference, to exercise the static shift."""
    from pyscf import dft
    mol = h4[0]
    mf = dft.RKS(mol)
    mf.xc = 'bhandhlyp'
    mf.grids.level = 5
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    eri = ao2mo.general(mol, (mf.mo_coeff,) * 4,
                        compact=False).reshape((mol.nao,) * 4)
    return mol, mf, eri, mol.nelectron // 2, mol.nao


def test_xc_shift_vanishes_on_hartree_fock(h4):
    """v_xc IS Sigma_x on a gas-phase HF reference, so the shift must be zero.

    This is what lets the shift be applied unconditionally: were it not zero
    here, every Hartree-Fock number this route has ever produced would move.
    """
    mol, mf, eri, nocc, norb = h4
    delta = static_exchange_diagonal(mf, mol, np.arange(norb), exchange='mf')
    assert np.abs(delta).max() < 1e-12

    plain = QPqb(mf.mo_energy, eri, nocc, screening='rpa')
    shifted = QPqb(mf.mo_energy, eri, nocc, screening='rpa', delta=delta)
    for p in range(norb):
        assert abs(plain.solve_diag(p)[0] - shifted.solve_diag(p)[0]) < 1e-13


def test_qp_on_a_kohn_sham_reference_carries_the_static_shift(h4_bhlyp):
    """eps_QP = eps_KS + <p|Sigma_x - v_xc|p> + Sigma_c(w), against production.

    Dropping the shift is not a small error: on BHLYP it is electron-volts, and
    it is the whole reason a Kohn-Sham starting point needs more than swapping
    the SCF.
    """
    mol, mf, eri, nocc, norb = h4_bhlyp
    from src.SingleReference.GW.self_energy import SelfEnergySolver
    from src.Solvers.qp_equation import solve_qp_equation_newton

    delta = static_exchange_diagonal(mf, mol, np.arange(norb), exchange='mf')
    assert np.abs(delta).max() > 1e-2          # the reference really is not HF

    # BOTH screenings: the shift is a property of the reference, not of W, so
    # it must enter a TDA-screened quasiparticle exactly as it enters an
    # RPA-screened one. That is the combination the four BSE@GW variants make
    # unavoidable, and the one an untested delta would break silently.
    se = SelfEnergySolver(mf.mo_energy, eri_chemist=eri,
                          spin_mode='restricted', eta=1e-9)
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=eri,
                              spin_mode='restricted', eta=1e-9)
    Aw, Bw = lr.build_casida_matrices(nocc, lBSE=False)
    for screening in ('rpa', 'tda'):
        qp = QPqb(mf.mo_energy, eri, nocc, screening=screening, delta=delta)
        B = Bw if screening == 'rpa' else np.zeros_like(Bw)
        om, X, Y = CasidaSolver(Aw, B).solve()
        for p in range(norb):
            w_mine, _ = qp.solve_diag(p)
            w_dense, _ = qp.solve_diag_dense(p)
            assert abs(w_mine - w_dense) < 1e-10, (screening, p)
            chi_a = se.get_chi_a(nocc, X, Y, p_state=p)
            func = lambda w: w - mf.mo_energy[p] - delta[p] - np.real(
                se.calculate_self_energy(p, w, nocc, om, chi_a, None,
                                         vertex_mode='GW'))
            w_prod = solve_qp_equation_newton(func, mf.mo_energy[p] + delta[p],
                                              tol=1e-12, max_iter=200)
            assert abs(w_mine - w_prod) < 1e-9, (screening, p)
            # and it is nowhere near the unshifted answer
            assert abs(w_mine - QPqb(mf.mo_energy, eri, nocc,
                                     screening=screening).solve_diag(p)[0]) > 1e-3


def test_kohn_sham_gradients_refuse_rather_than_drop_the_shift(h4_bhlyp):
    """d(shift)/dR is not assembled on this route, so the partials must say so."""
    mol, mf, eri, nocc, norb = h4_bhlyp
    delta = static_exchange_diagonal(mf, mol, np.arange(norb), exchange='mf')
    qp = QPqb(mf.mo_energy, eri, nocc, screening='rpa', delta=delta)
    w, _ = qp.solve_diag(nocc - 1)
    with pytest.raises(NotImplementedError, match='Sigma_x - v_xc'):
        qp_partials(qp, nocc - 1, w)
    b = BSEqb(mf, eri, nocc, screening='rpa', bse_tda=False, qp_orbs='all',
              delta=delta)
    with pytest.raises(NotImplementedError, match='Sigma_x - v_xc'):
        b.partials(0)


def test_blocks_reproduce_the_dense_route(h4):
    """Only three MO blocks matter; building just those must change nothing.

    The full (pq|rs) is nao^4 and is what keeps aug-cc-pVTZ off a workstation,
    while the energy path touches (ia|jb), (ij|ab) and (pq|ia) alone. Slicing
    those out of a dense tensor is exact by construction, so this pins the
    OTHER constructor: blocks transformed straight from the AO integrals.
    """
    mol, mf, eri, nocc, norb = h4
    sliced = MOEriBlocks.from_dense(eri, nocc)
    built = MOEriBlocks.from_mol(mol, mf.mo_coeff, nocc)
    for name in ('ovov', 'oovv', 'pqov'):
        a, b = getattr(sliced, name), getattr(built, name)
        assert np.abs(a - b).max() < 1e-12, name
    # the slice that is taken through the eight-fold symmetry rather than stored
    assert np.abs(sliced.vovo - eri[nocc:, :nocc, nocc:, :nocc]).max() < 1e-13

    for screening in ('rpa', 'tda'):
        for bse_tda in (False, True):
            ref = BSEqb(mf, eri, nocc, screening=screening, bse_tda=bse_tda,
                        qp_orbs='all')
            new = BSEqb(mf, built, nocc, screening=screening, bse_tda=bse_tda,
                        qp_orbs='all')
            assert np.abs(ref.eps_qp - new.eps_qp).max() < 1e-11
            assert np.abs(ref.Omega - new.Omega).max() < 1e-11
            assert abs(ref.qp.qb.e_corr() - new.qp.qb.e_corr()) < 1e-12


def test_blocks_refuse_gradients_rather_than_guess(h4):
    """The skeleton sweep needs the whole tensor; blocks must say so."""
    from src.gradients.grad_engine import correlation_gradients
    mol, mf, eri, nocc, norb = h4
    built = MOEriBlocks.from_mol(mol, mf.mo_coeff, nocc)
    b = BSEqb(mf, built, nocc, screening='rpa', bse_tda=False, qp_orbs='all')
    gF, G4, _ = b.partials(0)
    with pytest.raises(NotImplementedError, match='full'):
        correlation_gradients(mol, mf, [(gF, G4)], eri_mo=built)


def test_bse_kernel_vs_wicks(h4):
    mol, mf, eri, nocc, norb = h4
    b = BSEqb(mf, eri, nocc, screening='rpa', bse_tda=False, qp_orbs='all')
    V = b.qp.Vbare
    W_mine = eri - 4.0 * np.einsum('pqI,IJ,rsJ->pqrs', V, b.Pinv, V, optimize=True)
    W_wicks = np.asarray(static_screened_coulomb_chemist(mf.mo_energy, eri, nocc))
    assert np.abs(W_mine - W_wicks).max() < 1e-10


def test_rpa_partials_param_fd(h4):
    mol, mf, eri, nocc, norb = h4
    rng = np.random.default_rng(11)
    A, B, _ = build_rpa_AB(mf.mo_energy, eri, nocc)
    qb = RPA(A, B)
    gammaF, Gamma4 = rpa_partials(qb, nocc, norb)
    dF = rng.standard_normal((norb, norb)); dF = 0.5 * (dF + dF.T)
    dV = rng.standard_normal((norb,) * 4)
    dV = dV + dV.transpose(1, 0, 2, 3)
    dV = dV + dV.transpose(0, 1, 3, 2)
    dV = dV + dV.transpose(2, 3, 0, 1)
    dF /= np.linalg.norm(dF); dV /= np.linalg.norm(dV)

    def ec_at(s):
        occ, virt = np.arange(nocc), np.arange(nocc, norb)
        n_ov = nocc * (norb - nocc)
        epsF = np.diag(mf.mo_energy) + s * dF
        eriP = eri + s * dV
        Ax = (np.einsum('ij,ab->iajb', np.eye(nocc), epsF[np.ix_(virt, virt)])
              - np.einsum('ab,ij->iajb', np.eye(norb - nocc),
                          epsF[np.ix_(occ, occ)])).reshape(n_ov, n_ov)
        Ax += 2.0 * eriP[np.ix_(occ, virt, occ, virt)].reshape(n_ov, n_ov)
        Bx = 2.0 * eriP[np.ix_(occ, virt, occ, virt)].reshape(n_ov, n_ov)
        return RPA(Ax, Bx).e_corr()

    h = 1e-4
    fd = (ec_at(h) - ec_at(-h)) / (2 * h)
    an = np.einsum('pq,pq->', gammaF, dF) + np.einsum('pqrs,pqrs->', Gamma4, dV)
    assert abs(an - fd) < 1e-7 * max(1.0, abs(fd))


def test_dense_route_takes_its_spin_from_the_kappa_table(h4):
    """kappa is a table lookup, not the hard-coded 2 this route once refused
    triplets over. What must still be refused is an UNKNOWN spin, since that is
    what would otherwise be handed the singlet kernel.

    H4 is triplet-unstable at this geometry, and the two solver paths part
    company there: the omega^2 reduction tests A-B and refuses, while TDA
    diagonalizes A alone, has no such test, and returns a NEGATIVE excitation
    energy. Reading a TDA triplet gets no warning from the code.
    """
    mol, mf, eri, nocc, norb = h4
    assert KAPPA == {'singlet': 2.0, 'triplet': 0.0}

    singlet = BSEqb(mf, eri, nocc, screening='rpa')
    assert singlet.spin == 'singlet'
    assert singlet.kappa == KAPPA['singlet']

    with pytest.raises(ValueError, match='one of'):
        BSEqb(mf, eri, nocc, screening='rpa', spin='quintet')

    with pytest.raises(ValueError, match='BSE instability'):
        BSEqb(mf, eri, nocc, screening='rpa', spin='triplet')

    tda = BSEqb(mf, eri, nocc, screening='rpa', spin='triplet', bse_tda=True)
    assert tda.kappa == KAPPA['triplet']
    assert tda.Omega[0] < 0.0
