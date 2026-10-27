"""The Tamm-Dancoff BSE on the matrix-free routes: A X = omega X, B = 0.

The Davidson routes solve it on the A half of the same block action the full
Casida solve applies (`isdf_block_action(tda=True)`, `df_block_action(tda=True)`),
with a Hermitian Davidson (`_tda_davidson_root`). The references owe nothing to
that solver: the dense route's own Tamm-Dancoff roots (`bse_blocks` with
`bse_tda`, `CasidaSolver.solve(tda=True)`), and eigh of A built column by
column out of the action.

What a caller changes is ONE keyword: `Excitation(kernel='bse-tda')` through
`potential_energy_surface`, `bse_tda=True` on the chain, `tda=True` on
`solve_bse` / `solve_bse_isdf` / `solve_bse_df`. No solver switch.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import (HARTREE_TO_EV, SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.declaration import Excitation, GroundState
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.environments.solvent_screening import SolventScreening
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.surface_factory import potential_energy_surface
from src.SingleReference.GW.space_time import unrefused_separable_factors
from src.SingleReference.BSE.solve import solve_bse
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.BSE.solve import isdf_bse_factors
from src.SingleReference.LinearResponse.block_actions import (df_block_action,
                                                              isdf_block_action)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from tests.isdf_test_grid import g1, g1_numerics
from tests.pins import assert_same

BASIS = 'cc-pvdz'
MOLECULES = {
    'water': 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24',
    'formaldehyde': 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.59; H 0 -0.94 -0.59'}
BENZENE = ('C 0 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0 -1.396 0; '
           'C -1.209 -0.698 0; C -1.209 0.698 0; H 0 2.479 0; '
           'H 2.147 1.240 0; H 2.147 -1.240 0; H 0 -2.479 0; '
           'H -2.147 -1.240 0; H -2.147 1.240 0')
#: eV: the Davidson roots against the dense Tamm-Dancoff ones
ROOT_TOL_EV = 1e-6
NROOTS = 4


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module', params=sorted(MOLECULES))
def system(request):
    mol = gto.M(atom=MOLECULES[request.param], basis=BASIS, verbose=0)
    return request.param, mol, scf_factory(mol)


def lowest_roots(chain, mol, mf):
    """The chain's lowest NROOTS roots and its Y amplitudes."""
    solution = chain.forward(mol, mf)[1]
    om = solution.omega
    order = np.argsort(om)[:NROOTS]
    return np.asarray(om)[order], solution.y[:, order]


@pytest.mark.parametrize('environment', ['gas', 'toluene'])
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_davidson_tda_matches_the_dense_tda(system, environment, spin):
    """The chain's matrix-free Tamm-Dancoff roots are the dense route's, Y is
    zero, and each lies at or above the full kernel's root of that index."""
    name, mol, mf = system
    env = (None if environment == 'gas'
           else SolventScreening(mol, solvent='toluene'))
    kw = dict(mf=mf, spin=spin, environment=env)
    dense, _ = lowest_roots(ExcitedStateChain(
        mol, scf_factory, bse_tda=True,
        numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})), **kw), mol, mf)
    tight = g1_numerics(mol, Numerics(bse={'solver': 'davidson',
                                           'nroots': NROOTS, 'conv_tol': 1e-9}))
    tda, y = lowest_roots(ExcitedStateChain(
        mol, scf_factory, bse_tda=True, numerics=tight, **kw), mol, mf)
    full, _ = lowest_roots(ExcitedStateChain(
        mol, scf_factory, numerics=tight, **kw), mol, mf)
    assert np.abs(tda - dense).max() * HARTREE_TO_EV < ROOT_TOL_EV, name
    assert not np.any(y)
    assert np.all(tda >= full - 1e-9), (name, (tda - full) * HARTREE_TO_EV)
    assert (tda - full).max() * HARTREE_TO_EV > 1e-3      # B does something


@pytest.fixture(scope='module')
def benzene():
    """A pair space (1953 pairs) the Davidson has to iterate in, with a
    stand-in quasiparticle diagonal and the static W of the mean field."""
    mol = gto.M(atom=BENZENE, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    w_aux = LinearResponseSolver(eps, coeff_df=coeff,
                                 spin_mode='restricted').static_screening_aux(nocc)
    qp = eps.copy()
    qp[:nocc] -= 0.02
    qp[nocc:] += 0.03
    return mol, mf, nocc, LinearResponseSolver(qp, coeff_df=coeff,
                                               spin_mode='restricted'), w_aux


def dense_a(apply_a, no, nv):
    """A built whole from the action, one unit vector per pair."""
    n = no * nv
    unit = np.eye(n).reshape(n, no, nv)
    a = np.vstack([apply_a(unit[c:c + 256])[0].reshape(-1, n)
                   for c in range(0, n, 256)])
    return 0.5 * (a + a.T)


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_iterated_tda_davidson_is_eigh_of_a(benzene, spin):
    """Benzene: the Hermitian Davidson converges in a space far smaller than
    the pair space, on the eigenvalues of A, with |A X - omega X| <= conv_tol."""
    mol, mf, nocc, lr, w_aux = benzene
    full, d = df_block_action(lr, nocc, True, w_aux, spin=spin)
    half, _ = df_block_action(lr, nocc, True, w_aux, spin=spin, tda=True)
    no, nv = d.shape
    z = np.random.default_rng(7).standard_normal((3, no, nv))
    az, bz = half(z)
    # two actions: A z to a few ulp
    assert bz is None
    assert_same(az, full(z)[0], quantity='root', label=('DF', spin))
    a = dense_a(half, no, nv)
    timings = {}
    om, x, y = solve_casida_davidson(lr, nocc, numerics=Numerics(bse={'nroots': 6, 'conv_tol': 1e-8}), polarizability='BSE',
                                     W_aux=w_aux, spin=spin,
                                     tda=True, timings=timings)
    ref = np.linalg.eigvalsh(a)[:6]
    assert np.abs(np.sort(om) - ref).max() * HARTREE_TO_EV < ROOT_TOL_EV
    assert np.linalg.norm(a @ x - x * om, axis=0).max() <= 1e-8
    assert np.allclose(np.linalg.norm(x, axis=0), 1.0) and not np.any(y)
    assert timings['davidson_verdict'] == 'converged'
    assert timings['davidson_vectors_applied'] < no * nv // 10


def test_isdf_action_tda_is_the_a_half(benzene):
    """`isdf_block_action(tda=True)`: A z to a few ulp (two actions), B z
    never formed."""
    mol, mf, nocc, lr, _ = benzene
    # benzene's virtual 86 is missed by 0.12 on the default grid and the
    # factors would be refused; this gate is the solver's, not the fit's
    factors = isdf_bse_factors(
        mf, mol, nocc,
        factors=unrefused_separable_factors(mf, mol, counts=g1(mol))[:2])
    x_mo, d_fac, w_aux = factors
    lr_isdf = LinearResponseSolver(lr.eps, spin_mode='restricted')
    for spin in ('singlet', 'triplet'):
        full, diag = isdf_block_action(lr_isdf, nocc, True, w_aux,
                                       (x_mo, d_fac), spin=spin)
        half, _ = isdf_block_action(lr_isdf, nocc, True, w_aux,
                                    (x_mo, d_fac), spin=spin, tda=True)
        z = np.random.default_rng(3).standard_normal((2,) + diag.shape)
        az, bz = half(z)
        assert bz is None
        assert_same(az, full(z)[0], quantity='root', label=('ISDF', spin))


@pytest.mark.parametrize('integrals', ['df', 'isdf'])
def test_solve_bse_takes_tda_on_the_davidson_route(integrals):
    """`solve_bse(tda=True)` is one keyword on either solver, same roots."""
    mol = gto.M(atom=MOLECULES['water'], basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    kw = dict(integrals=integrals, qp=None, tda=True)
    if integrals == 'isdf':
        kw['factors'] = unrefused_separable_factors(mf, mol,
                                                    counts=g1(mol))[:2]
    om_d = solve_bse(mf, mol, solver='dense',
                     numerics=Numerics(bse={'nroots': 3}), **kw)[0]
    om_v, _, y, info = solve_bse(
        mf, mol, solver='davidson',
        numerics=Numerics(bse={'nroots': 3, 'conv_tol': 1e-9}), **kw)
    gap = np.abs(np.sort(om_v) - np.sort(om_d)[:3]).max() * HARTREE_TO_EV
    assert gap < ROOT_TOL_EV       # ISDF: 3.3e-7 eV, the dense route's build
    assert not np.any(y) and info['min_eig_amb'] is None


def test_the_kernel_keyword_alone_reaches_the_matrix_free_tda():
    """Through `potential_energy_surface`, only `kernel` changes: the declared
    Tamm-Dancoff state comes back from the Davidson route with the dense
    route's energy, above the full kernel's, and declares itself."""
    mol = gto.M(atom=MOLECULES['formaldehyde'], basis=BASIS, verbose=0)
    mf = scf_factory(mol)

    def surface(kernel, solver='davidson'):
        return potential_energy_surface(
            mol, scf_factory, ground_state=GroundState('dft', 'hf'),
            excitation=Excitation('singlet', root=1, kernel=kernel),
            solver=solver, mf=mf)

    tda, full = surface('bse-tda'), surface('bse')
    assert tda.physics.excitation.kernel == 'bse-tda'
    assert tda.solver_used(10 ** 6) == 'davidson'
    om_tda, om_full = tda.excitation(mol, mf), full.excitation(mol, mf)
    om_dense = surface('bse-tda', solver='dense').excitation(mol, mf)
    assert abs(om_tda - om_dense) * HARTREE_TO_EV < ROOT_TOL_EV
    assert om_tda > om_full


def test_the_tda_davidson_gives_the_dense_tda_force():
    """The Hellmann-Feynman adjoint reads X with Y = 0 either way, so the
    matrix-free Tamm-Dancoff force is the dense one's."""
    mol = gto.M(atom=MOLECULES['water'], basis=BASIS, verbose=0)
    mf = scf_factory(mol)
    dense = ExcitedStateChain(
        mol, scf_factory, mf=mf, bse_tda=True,
        numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    david = ExcitedStateChain(
        mol, scf_factory, mf=mf, bse_tda=True,
        numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'nroots': 5})))
    g_d = dense.excitation_gradient()[0]
    g_v = david.excitation_gradient()[0]
    assert np.abs(g_d - g_v).max() < 1e-7
    assert np.abs(g_d).max() > 1e-3
