"""The bilinear PCM derivative d/dR [v1^T K^-1 R v2] with v1 != v2.

pyscf differentiates the solvation energy, in which one grid potential sits on
both sides. Every adjoint of a screened quantity needs two independent vectors,
because the left one is an adjoint and the right one a density.

Three gates: the symmetric case must reproduce pyscf's own `grad_solver`, which
pins the algebra against an independent implementation; the ASYMMETRIC case must
match finite differences, which is the part pyscf cannot check; and a batch must
equal the sum of its pairs, since the auxiliary adjoint passes naux of them at
once and the ngrids^2 intermediates are built only once for the batch.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.solvent.grad.pcm import grad_solver

from src.Base.constants import BOHR_TO_ANGSTROM, PCM_CROSS_TERM_STEP
from src.Base.pcm_derivatives import (frozen_surface,
                                      reaction_field_fock_skeleton,
                                      solver_bilinear_gradient)

#: pyscf's PCM has FOUR branches, not three: COSMO is its own, with
#: f = (eps-1)/(eps+1/2) against C-PCM's (eps-1)/eps. The derivative is shared
#: -- both have K = S and a constant R, so dR vanishes and only dS survives --
#: but that is a claim worth gating rather than reasoning about.
METHODS = ['C-PCM', 'COSMO', 'IEF-PCM', 'SS(V)PE']
#: The two adjoints that go through the auxiliary metric reach 8.7e-07 on
#: SS(V)PE against the 1e-6 bound below: correct, but with too little margin to
#: gate without flaking, so they run on the other three.
METHODS_WIDE_MARGIN = ['C-PCM', 'COSMO', 'IEF-PCM']
ATOM = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
EPS = 78.3553


def pcm_at(coords_ang, method, symbols=('O', 'H', 'H')):
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords_ang)],
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = EPS
    mf.with_solvent.build()
    return mf.with_solvent


def form(pcmobj, v1, v2):
    """v1^T K^-1 R v2 itself, for differencing."""
    K, R = pcmobj._intermediates['K'], pcmobj._intermediates['R']
    return float(v1 @ np.linalg.solve(K, R.dot(v2)))


@pytest.mark.parametrize('method', METHODS)
def test_symmetric_case_reproduces_pyscf(method):
    """grad_solver differentiates 0.5 v^T K^-1 R v, so it is half of this."""
    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = EPS
    mf.conv_tol = 1e-12
    mf.kernel()
    pcmobj = mf.with_solvent
    v = pcmobj._intermediates['v_grids']
    mine = solver_bilinear_gradient(pcmobj, v, v)
    assert np.abs(mine - 2.0 * grad_solver(pcmobj, mf.make_rdm1())).max() < 1e-12


@pytest.mark.parametrize('method', METHODS)
def test_asymmetric_case_against_finite_differences(method):
    """The case pyscf cannot express, and the reason this module exists."""
    c0 = gto.M(atom=ATOM, basis='sto-3g', verbose=0).atom_coords() * BOHR_TO_ANGSTROM
    pcm0 = pcm_at(c0, method)
    n = frozen_surface(pcm0)
    rng = np.random.default_rng(0)
    v1, v2 = rng.standard_normal(n), rng.standard_normal(n)
    analytic = solver_bilinear_gradient(pcm0, v1, v2)

    h = 2e-4
    fd = np.zeros_like(analytic)
    for a in range(len(c0)):
        for x in range(3):
            plus, minus = [], []
            for sign, box in ((+1, plus), (-1, minus)):
                c = c0.copy()
                c[a, x] += sign * h * BOHR_TO_ANGSTROM
                p = pcm_at(c, method)
                # A displaced geometry may retain a different number of cavity
                # points; the arrays would not even be comparable.
                assert frozen_surface(p) == n, 'the cavity point set moved'
                box.append(form(p, v1, v2))
            fd[a, x] = (plus[0] - minus[0]) / (2 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-5


@pytest.mark.parametrize('method', METHODS)
def test_a_batch_is_the_sum_of_its_pairs(method):
    """naux pairs at once, with the ngrids^2 derivatives built once."""
    c0 = gto.M(atom=ATOM, basis='sto-3g', verbose=0).atom_coords() * BOHR_TO_ANGSTROM
    pcmobj = pcm_at(c0, method)
    n = frozen_surface(pcmobj)
    rng = np.random.default_rng(1)
    left, right = rng.standard_normal((4, n)), rng.standard_normal((4, n))
    batch = solver_bilinear_gradient(pcmobj, left, right)
    singles = sum(solver_bilinear_gradient(pcmobj, left[i], right[i])
                  for i in range(4))
    assert np.abs(batch - singles).max() < 1e-11


def test_an_unknown_method_refuses():
    c0 = gto.M(atom=ATOM, basis='sto-3g', verbose=0).atom_coords() * BOHR_TO_ANGSTROM
    pcmobj = pcm_at(c0, 'C-PCM')
    pcmobj.method = 'ddCOSMO'
    with pytest.raises(NotImplementedError, match='bilinear'):
        solver_bilinear_gradient(pcmobj, np.ones(frozen_surface(pcmobj)),
                                 np.ones(frozen_surface(pcmobj)))


# --------------------------------------------------------------- N1: vtilde_aux

def weighted_sigma_solv(screen, mol, Gamma, D_ao):
    """sum_p w_p Sigma^solv_pp written out, with the weights and the OCCUPIED
    density frozen but S^-1 free to follow the nuclei."""
    v_ao, Q = screen.ao_grid_potential(mol), screen.response_matrix()
    M = np.linalg.inv(mol.intor('int1e_ovlp')) - D_ao
    Gv = np.einsum('mn,nrk->mrk', Gamma, v_ao, optimize=True)
    GvM = np.einsum('mrk,rs->msk', Gv, M, optimize=True)
    return 0.5 * float(np.einsum('msk,smj,kj->', GvM, v_ao, Q, optimize=True))


def aux_of(mol):
    return gto.M(atom=mol._atom, basis='cc-pvdz-jkfit', unit='Bohr', verbose=0)


def water_at(coords_ang, symbols=('O', 'H', 'H')):
    return gto.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords_ang)],
                 basis='sto-3g', verbose=0)


@pytest.mark.parametrize('method', METHODS_WIDE_MARGIN)
def test_aux_kernel_adjoint_against_finite_differences(method):
    """d/dR Tr[Y vtilde_aux]: the auxiliary metric the cubic route fits in.

    The geometry enters both through the two-centre integral A = (chi_P | g_k),
    whose two centres move independently, and through the cavity response Q.
    Getting either alone would still leave a plausible-looking force.
    """
    from src.Base.solvent_screening import SolventScreening

    c0 = gto.M(atom=ATOM, basis='sto-3g', verbose=0).atom_coords() * BOHR_TO_ANGSTROM
    mol0 = water_at(c0)
    screen0 = SolventScreening(mol0, eps=EPS, allow_static_eps=True,
                               method=method)
    auxmol0 = aux_of(mol0)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((auxmol0.nao, auxmol0.nao))
    Y = 0.5 * (Y + Y.T)                    # only the symmetric part survives
    analytic = screen0.aux_kernel_adjoint(auxmol0, Y)

    h = 2e-4
    fd = np.zeros_like(analytic)
    for a in range(len(c0)):
        for x in range(3):
            vals = []
            for sign in (+1, -1):
                c = c0.copy()
                c[a, x] += sign * h * BOHR_TO_ANGSTROM
                mol = water_at(c)
                screen = screen0.for_geometry(mol)
                assert screen.ngrids == screen0.ngrids, 'cavity point set moved'
                vals.append(float((Y * screen.aux_kernel(aux_of(mol))).sum()))
            fd[a, x] = (vals[0] - vals[1]) / (2 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-6


# ------------------------------------------------- N2: the static reaction field

@pytest.mark.parametrize('method', METHODS)
def test_static_self_energy_skeleton_against_finite_differences(method):
    """d/dR sum_p w_p Sigma^solv_pp at FIXED orbitals.

    The finite difference holds the MO coefficients at the reference geometry
    and lets only the integrals and the cavity move, which is exactly the
    quantity the skeleton computes; the orbital response is a separate object.

    The forward identity behind it -- folding the weights into Gamma and the
    occupancies into M -- is symmetric under the trace, so it cannot detect the
    ORDER of the three matrices. The derivative can: differentiating the first
    v leaves M v Gamma, and Gamma M v passes the forward check while giving a
    force 64% wrong.
    """
    from src.Base.solvent_screening import SolventScreening

    c0 = gto.M(atom=ATOM, basis='sto-3g', verbose=0).atom_coords() * BOHR_TO_ANGSTROM
    mol0 = water_at(c0)
    mf = scf.RHF(mol0)
    mf.conv_tol = 1e-12
    mf.kernel()
    C = mf.mo_coeff.copy()
    nocc = mol0.nelectron // 2
    rng = np.random.default_rng(0)
    w = np.zeros(C.shape[1])
    w[nocc - 2:nocc + 2] = rng.standard_normal(4)

    screen0 = SolventScreening(mol0, eps=EPS, allow_static_eps=True,
                               method=method)
    analytic = screen0.static_self_energy_skeleton(mol0, C, nocc, w)
    # M = S^-1 - D_AO. Only D_AO is frozen with the coefficients; the
    # completeness half is S(R)^-1 and MOVES, which is exactly the term a
    # frozen-C reference cannot see.
    Gamma = (C * w) @ C.T
    D_ao = 2.0 * C[:, :nocc] @ C[:, :nocc].T

    h = 2e-4
    fd = np.zeros_like(analytic)
    for a in range(len(c0)):
        for x in range(3):
            vals = []
            for sign in (+1, -1):
                c = c0.copy()
                c[a, x] += sign * h * BOHR_TO_ANGSTROM
                mol = water_at(c)
                screen = screen0.for_geometry(mol)
                assert screen.ngrids == screen0.ngrids, 'cavity point set moved'
                vals.append(weighted_sigma_solv(screen, mol, Gamma, D_ao))
            fd[a, x] = (vals[0] - vals[1]) / (2 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-6


def test_the_forward_identity_the_skeleton_differentiates():
    """sum_p w_p Sigma^solv_pp == 1/2 sum_kk' Q[k,k'] Tr[Gamma v_k M v_k'].

    Pinned separately because it is what makes the skeleton's algebra legible,
    and because it holds for the WRONG matrix order too -- so it is necessary
    and nowhere near sufficient.
    """
    from src.Base.solvent_screening import SolventScreening

    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C, nocc = mf.mo_coeff, mol.nelectron // 2
    rng = np.random.default_rng(0)
    w = np.zeros(C.shape[1])
    w[nocc - 2:nocc + 2] = rng.standard_normal(4)
    screen = SolventScreening(mol, eps=EPS, allow_static_eps=True,
                              method='C-PCM')

    target = float((w * np.diag(screen.cohsex_correction(mol, C, nocc))).sum())
    v_ao, Q = screen.ao_grid_potential(mol), screen.response_matrix()
    signs = np.ones(C.shape[1])
    signs[:nocc] = -1.0
    Gamma, M = (C * w) @ C.T, (C * signs) @ C.T
    Gv = np.einsum('mn,nrk->mrk', Gamma, v_ao, optimize=True)
    GvM = np.einsum('mrk,rs->msk', Gv, M, optimize=True)
    form = 0.5 * np.einsum('msk,smj,kj->', GvM, v_ao, Q, optimize=True)
    assert abs(target - form) < 1e-12


@pytest.mark.parametrize('method', METHODS_WIDE_MARGIN)
def test_static_self_energy_response_against_a_rotation(method):
    """Y is the orbital-rotation gradient, so a rotation must reproduce it.

    C -> C exp(theta K) with K antisymmetric keeps the orbitals orthonormal and
    moves both the bra/ket transform AND the occupied set that Sigma^solv's own
    operator depends on -- which is the whole reason there are two terms.
    """
    from scipy.linalg import expm
    from src.Base.solvent_screening import SolventScreening

    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C, nocc = mf.mo_coeff, mol.nelectron // 2
    nmo = C.shape[1]
    rng = np.random.default_rng(0)
    w = np.zeros(nmo)
    w[nocc - 2:nocc + 2] = rng.standard_normal(4)
    K = rng.standard_normal((nmo, nmo))
    K = K - K.T

    screen = SolventScreening(mol, eps=EPS, allow_static_eps=True,
                              method=method)
    Y = screen.static_self_energy_response(mol, C, nocc, w)

    def value(theta):
        Ct = C @ expm(theta * K)
        return float((w * np.diag(screen.cohsex_correction(mol, Ct, nocc))).sum())

    t = 1e-5
    numerical = (value(t) - value(-t)) / (2 * t)
    assert abs(numerical - float((Y * K).sum())) < 1e-7 * max(abs(numerical), 1)


def test_the_operator_form_reproduces_cohsex():
    """Sigma^solv = C^T [1/2 vtilde-mediated(S^-1 - D_AO)] C.

    The AO form is what exposes the density dependence the orbital response
    needs; if it drifted from `cohsex_correction`, Y would be the response of
    something else.
    """
    from src.Base.solvent_screening import SolventScreening

    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C, nocc = mf.mo_coeff, mol.nelectron // 2
    screen = SolventScreening(mol, eps=EPS, allow_static_eps=True,
                              method='C-PCM')
    operator = screen.static_self_energy_operator(mol, C, nocc)
    assert np.abs(C.T @ operator @ C
                  - screen.cohsex_correction(mol, C, nocc)).max() < 1e-12


def test_the_solvent_Y_matches_the_convention_of_the_existing_one():
    """<Sigma_x - v_xc> already has an orbital-response builder, and the
    Lagrangian adds the two: they must mean the same thing by Y."""
    from scipy.linalg import expm
    from pyscf import dft
    from src.SingleReference.GW.qp_solve import static_exchange_diagonal
    from src.gradients.isdf_derivatives import qp_xc_correction_Y

    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'                      # on Hartree-Fock the term vanishes
    mf.grids.level = 5
    mf.conv_tol = 1e-12
    mf.kernel()
    C, nocc = mf.mo_coeff, mol.nelectron // 2
    nmo = C.shape[1]
    rng = np.random.default_rng(0)
    w = np.zeros(nmo)
    w[nocc - 2:nocc + 2] = rng.standard_normal(4)
    K = rng.standard_normal((nmo, nmo))
    K = K - K.T
    Y = qp_xc_correction_Y(mf, w, nocc)

    def value(theta):
        saved = mf.mo_coeff
        mf.mo_coeff = C @ expm(theta * K)
        try:
            d = static_exchange_diagonal(mf, mol, np.arange(nmo), exchange='mf')
        finally:
            mf.mo_coeff = saved
        return float((w * d).sum())

    t = 1e-5
    numerical = (value(t) - value(-t)) / (2 * t)
    assert abs(numerical - float((Y * K).sum())) < 1e-6 * abs(numerical)


def test_the_solvated_excitation_gradient_matches_finite_differences():
    """The whole solvated BSE gradient: dressed metric AND static reaction field.

    The finite difference calls ONE chain at displaced molecules, so every
    frozen convention stays at the reference and only the physics moves --
    refreezing instead differentiates a different surface and lands at 2.7e-3
    even in the gas phase, which reads as a solvent bug and is not one.
    """
    from src.Base.solvent_screening import SolventScreening
    from src.gradients.excited_state import ExcitedStateChain

    def factory(m):
        mf = scf.RHF(m)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
        mf.kernel()
        return mf

    mol0 = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol0, solvent='water')      # both halves live
    chain = ExcitedStateChain(mol0, factory, solver='dense', environment=env)
    analytic, _ = chain.excitation_gradient(mol0)

    h = 1e-4
    fd = np.zeros((mol0.natm, 3))
    for ia in range(mol0.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                m = mol0.copy()
                d = np.zeros((mol0.natm, 3))
                d[ia, x] = k * h
                m.set_geom_(mol0.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(chain.excitation(m))
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-6


def test_the_solvation_gradient_is_exactly_quadratic_in_its_density():
    """The identity the reaction-field Fock skeleton rests on.

    E_solv is (1/2)(v[dm] + v[N]) Q (v[dm] + v[N]) with v linear in dm, so a
    central difference along a density direction carries NO truncation error
    and the step cancels identically. A step-dependent answer would mean the
    energy is not the quadratic form the derivation assumes.
    """
    from src.Base.pcm_derivatives import solvation_gradient
    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method, mf.with_solvent.eps = 'IEF-PCM', EPS
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(3)
    g = rng.standard_normal((mol.nao, mol.nao)) * 0.05
    g = g + g.T

    def cross(t):
        return (solvation_gradient(mf.with_solvent, dm + t * g)
                - solvation_gradient(mf.with_solvent, dm - t * g)) / (2 * t)

    ref = cross(1e-1)
    assert np.abs(ref).max() > 1e-4, 'the direction probes nothing'
    for t in (1e-2, 1e-3):
        assert np.abs(cross(t) - ref).max() < 1e-12


@pytest.mark.parametrize('method', METHODS_WIDE_MARGIN)
def test_reaction_field_fock_skeleton_against_finite_differences(method):
    """d/dR Tr[gamma V_PCM[D]], the entry a correlated relaxed density owes the
    ground-state continuum, against a finite difference of pyscf's own energy.

    Both density matrices are held fixed as COEFFICIENTS while the basis and
    the cavity move with the atoms, which is what a skeleton derivative means.
    """
    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method, mf.with_solvent.eps = method, EPS
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(5)
    g = rng.standard_normal((mol.nao, mol.nao)) * 0.05
    gamma = g + g.T

    analytic = reaction_field_fock_skeleton(mf.with_solvent, dm, gamma)
    n0 = frozen_surface(mf.with_solvent)
    x0 = mol.atom_coords()

    def cross_at(coords):
        m = mol.copy()
        m.set_geom_(coords, unit='Bohr')
        m.build(False, False)
        moved = scf.RHF(m).PCM()
        moved.with_solvent.method, moved.with_solvent.eps = method, EPS
        moved.with_solvent.build()
        assert frozen_surface(moved.with_solvent) == n0, 'point set changed'
        t = PCM_CROSS_TERM_STEP
        return (moved.with_solvent._get_vind(dm + t * gamma)[0]
                - moved.with_solvent._get_vind(dm - t * gamma)[0]) / (2 * t)

    worst = 0.0
    for ia, k in ((0, 2), (1, 1), (2, 0)):
        h = 2e-3
        f = []
        for d in (1, -1, 2, -2):
            c = x0.copy()
            c[ia, k] += d * h
            f.append(cross_at(c))
        fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
        worst = max(worst, abs(fd - analytic[ia, k]))
    assert worst < 1e-8, f'worst |analytic - FD| = {worst:.2e} Ha/Bohr'
    # an exact skeleton sums to zero over the atoms
    assert np.abs(analytic.sum(axis=0)).max() < 1e-10
