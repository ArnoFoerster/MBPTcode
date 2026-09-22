"""The dense quasi-boson route computes the same numbers from production.

The forward physics of the dense oracle -- the closed-form dRPA ground state,
its bare boson couplings, the diagonal EOM G0W0 quasiparticle solve and the
BSE@G0W0 eigenproblem, together with the MO integral blocks and the fitted
interaction all four are built from -- lives in `SingleReference.GW.quasi_boson`,
`SingleReference.LinearResponse.quasi_boson_bse` and `Base.eri_blocks`. The
reverse mode stays in `gradients.quasi_boson_adjoint` as three subclasses, and
the five old modules are re-export shims.

WHAT IS GATED. That the shim names ARE the production or adjoint objects
(identity, not a copy); that every recorded quantity is bitwise the same
through either import root; the four dedup verdicts as measured numbers; and
that a fresh interpreter can import all three production modules without
`src.gradients` appearing in sys.modules.

THE GATES WERE SHOWN TO FAIL. Four perturbations, each on a backup copy of
the tree, restored and verified with `cmp` afterwards:

  * the screened term of the BSE direct kernel flipped in sign
    (`quasi_boson_bse`, Wd4 = eri4.oovv + S_d) fails
    `test_the_route_still_computes_the_same_physics` on both systems, and
    moves the frozen water DenseBSESurface[singlet] R0 row by 7.6e-2 Ha in
    energy and 9.7e-3 Ha/Bohr in the gradient.
  * the pole-strength filter dropped (`quasi_boson_bse`, every root accepted
    regardless of Z) fails `test_the_pole_strength_filter_is_the_declared_constant`
    and the water physics pin; the baseline row moves 2.3e-5 Ha and 1.7e-4
    Ha/Bohr, and the quasiparticle set grows from 15 states to 19.
  * the boson seam reverted (`QPqbAdjoint.boson` removed, so a gradient object
    would carry the forward-only RPA) fails
    `test_the_three_classes_the_shims_export_are_the_adjoint_subclasses`,
    `test_the_bse_partials_only_exist_on_the_adjoint_object` and
    `tests/test_gradients_qb.py::test_blocks_refuse_gradients_rather_than_guess`.
  * a shim re-DEFINING `build_rpa_AB` instead of re-exporting it fails
    `test_every_shim_name_is_the_production_or_adjoint_object`.

The last one is why the pinned-physics gate is here at all: a bitwise gate
between the shim and production compares two import roots of ONE body of code,
so it sees a shim that drifted and CANNOT see a kernel that changed sign.
Physics is held by the pinned numbers below, by the frozen baseline record and
by the existing surface tests.

Recorded reference numbers are from H4/sto-3g and water/cc-pVDZ, both RHF
converged to conv_tol_grad 1e-11.
"""
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

import src.Base.eri_blocks as production_eri
import src.SingleReference.GW.quasi_boson as production_qb
import src.SingleReference.LinearResponse.quasi_boson_bse as production_bse
import src.gradients.bse_qb as shim_bse
import src.gradients.df_assembly as shim_df
import src.Base.eri_blocks as shim_eri
import src.gradients.qb_core as shim_core
import src.gradients.qp_qb as shim_qp
import src.gradients.quasi_boson_adjoint as adjoint
from src.Base.constants import QP_WINDOW_Z_MIN
from src.Base.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.GW.qp_energy import _two_electron_integrals
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import rpa_correlation_energy_casida
from src.Solvers.qp_equation import (solve_qp_equation_newton,
                                     solve_qp_equation_newton_guarded)

#: The two systems every number below is measured on.
H4_ATOM = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'
WATER_ATOM = 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692'

#: `e_corr_plasmon` against `rpa_correlation_energy_casida`, in Hartree. Not
#: bitwise: the same trace formula off two different eigensolvers.
PLASMON_VS_CASIDA = {'h4': 1.78e-15, 'water': 5.68e-14}

#: `MOEriBlocks.from_mol` against the four-index builder production's
#: quasiparticle driver uses, largest element of (ij|ab), in Hartree. Zero on
#: H4/sto-3g -- the smallest molecule does not see this one.
FROM_MOL_VS_DENSE = {'h4': 0.0, 'water': 4.56e-15}

#: The guarded Newton of `Solvers.qp_equation` against `QPqb.solve_diag` on
#: water: bitwise on every root above this pole strength, and this far apart in
#: Hartree on the satellites below it. The threshold is not the guarded
#: solver's own Z floor -- it converges onto a DIFFERENT root there, whose
#: weight is 0.003 where this loop's is 0.233 -- so it is set between the
#: lowest root the two share (Z = 0.376) and the highest they do not.
NEWTON_BITWISE_ABOVE_Z = 0.3
NEWTON_SATELLITE_GAP = 7.92e-2

#: The physics itself, in Hartree: E_c^dRPA, the lowest BSE@GW singlet
#: excitation, the HOMO quasiparticle energy and the size of the quasiparticle
#: set the pole-strength filter admits. A gate comparing two import roots of
#: ONE body of code cannot see a kernel that changed sign; this one can.
#: Pinned to PHYSICS_TOL, which is far below any change worth catching (the
#: screened term of the direct kernel flipped moves Omega by 1e-2) and far
#: above the last digit the SCF path carries.
PHYSICS_TOL = 1e-9
PINNED = {
    'h4': {'e_corr': -0.07460064347212232, 'omega0': 0.11445588148996017,
           'homo_qp': -0.24698056044895061, 'n_qp': 4},
    'water': {'e_corr': -0.2313009545750493, 'omega0': 0.3104456904334518,
              'homo_qp': -0.44679715112656565, 'n_qp': 16},
}


def build(atom, basis, unit='Angstrom'):
    """(mol, mf, eri, nocc, norb) on a tightly converged RHF reference."""
    mol = gto.M(atom=atom, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    norb, nocc = mol.nao, mol.nelectron // 2
    eri = ao2mo.general(mol, (mf.mo_coeff,) * 4,
                        compact=False).reshape((norb,) * 4)
    return mol, mf, eri, nocc, norb


@pytest.fixture(scope='module')
def h4():
    return build(H4_ATOM, 'sto-3g', unit='Bohr')


@pytest.fixture(scope='module')
def water():
    return build(WATER_ATOM, 'cc-pvdz')


def systems(h4, water):
    return {'h4': h4, 'water': water}


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


# ---------------------------------------------------------------------------
# the shims re-export the objects themselves
# ---------------------------------------------------------------------------

def test_every_shim_name_is_the_production_or_adjoint_object():
    """Identity, not equality: a copy would drift the next time one is edited."""
    assert shim_core.build_rpa_AB is production_qb.build_rpa_AB
    assert shim_core.couplings_V is production_qb.couplings_V
    assert shim_core.eigh_sym is production_qb.eigh_sym
    assert shim_core.funm_sym is production_qb.funm_sym
    assert shim_core.sqrtm_sym is production_qb.sqrtm_sym
    assert shim_core.invsqrtm_sym is production_qb.invsqrtm_sym
    assert shim_core.frechet_funm_sym is adjoint.frechet_funm_sym
    assert shim_qp.qp_energy_general is production_qb.qp_energy_general
    assert shim_eri.MOEriBlocks is production_eri.MOEriBlocks
    assert shim_eri.as_blocks is production_eri.as_blocks
    assert shim_df.df_eri_mo is production_eri.df_eri_mo
    assert shim_df.df_integrals is production_eri.df_integrals


def test_the_three_classes_the_shims_export_are_the_adjoint_subclasses():
    """A gradient caller reaches the reverse mode under the old names.

    `RPA`, `QPqb` and `BSEqb` have always meant the differentiable objects on
    the gradient side; the forward-only classes of the same names carry no
    Frechet map and no partials, and a caller handed one would fail at its
    first chain rule rather than return a different number.
    """
    assert shim_core.RPA is adjoint.RPAAdjoint
    assert shim_qp.QPqb is adjoint.QPqbAdjoint
    assert shim_bse.BSEqb is adjoint.BSEqbAdjoint
    assert issubclass(adjoint.RPAAdjoint, production_qb.RPA)
    assert issubclass(adjoint.QPqbAdjoint, production_qb.QPqb)
    assert issubclass(adjoint.BSEqbAdjoint, production_bse.BSEqb)
    # the composition seam: a forward object builds forward parts
    assert production_qb.QPqb.boson is production_qb.RPA
    assert adjoint.QPqbAdjoint.boson is adjoint.RPAAdjoint
    assert production_bse.BSEqb.quasiparticle is production_qb.QPqb
    assert adjoint.BSEqbAdjoint.quasiparticle is adjoint.QPqbAdjoint


# ---------------------------------------------------------------------------
# the numbers are bitwise through either import root
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_rpa_ground_state_is_bitwise_through_both_roots(tag, h4, water):
    _, mf, eri, nocc, _ = systems(h4, water)[tag]
    A, B, d = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc)
    As, Bs, ds = shim_core.build_rpa_AB(mf.mo_energy, eri, nocc)
    assert bitwise(A, As) and bitwise(B, Bs) and bitwise(d, ds)
    prod, shim = production_qb.RPA(A, B), shim_core.RPA(A, B)
    for name in ('t', 'wt', 'Vt', 'exp_t', 'exp_mt', 'cosh_t', 'sinh_t',
                 'Abar', 'omega'):
        assert bitwise(getattr(prod, name), getattr(shim, name)), name
    assert bitwise(prod.eigAbar[1], shim.eigAbar[1])
    assert prod.e_corr() == shim.e_corr()
    assert prod.e_corr_plasmon() == shim.e_corr_plasmon()
    assert bitwise(production_qb.couplings_V(eri, nocc),
                   shim_core.couplings_V(eri, nocc))


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_quasiparticle_solve_is_bitwise_through_both_roots(tag, h4, water):
    _, mf, eri, nocc, norb = systems(h4, water)[tag]
    prod = production_qb.QPqb(mf.mo_energy, eri, nocc, screening='rpa')
    shim = shim_qp.QPqb(mf.mo_energy, eri, nocc, screening='rpa')
    assert bitwise(prod.Wnu, shim.Wnu) and bitwise(prod.omega, shim.omega)
    for p in range(norb):
        assert prod.solve_diag(p) == shim.solve_diag(p)
        assert prod.sigma(p, mf.mo_energy[p]) == shim.sigma(p, mf.mo_energy[p])
    p = nocc - 1
    assert bitwise(prod.supermatrix_diag(p), shim.supermatrix_diag(p))
    assert prod.solve_diag_dense(p) == shim.solve_diag_dense(p)
    fock = np.diag(np.asarray(mf.mo_energy, float))
    assert (production_qb.qp_energy_general(fock, eri, nocc, p)
            == shim_qp.qp_energy_general(fock, eri, nocc, p))


@pytest.mark.parametrize('screening,bse_tda', [('rpa', False), ('rpa', True),
                                               ('tda', False), ('tda', True)])
def test_the_bse_eigenpairs_are_bitwise_through_both_roots(screening, bse_tda,
                                                           water):
    _, mf, eri, nocc, _ = water
    kw = dict(screening=screening, bse_tda=bse_tda)
    for spin in ('singlet', 'triplet'):
        prod = production_bse.BSEqb(mf, eri, nocc, spin=spin, **kw)
        shim = shim_bse.BSEqb(mf, eri, nocc, spin=spin, **kw)
        assert bitwise(prod.A_bse, shim.A_bse)
        assert (prod.B_bse is None) == (shim.B_bse is None)
        if prod.B_bse is not None:
            assert bitwise(prod.B_bse, shim.B_bse)
        assert bitwise(prod.Pinv, shim.Pinv)
        assert bitwise(prod.eps_qp, shim.eps_qp)
        assert prod.qp_roots == shim.qp_roots
        assert bitwise(prod.Omega, shim.Omega)
        assert bitwise(prod.X, shim.X) and bitwise(prod.Y, shim.Y)


def test_the_pole_strength_filter_is_the_declared_constant(water):
    """The BSE diagonal is dressed with the roots the QP window admits.

    Dropping the filter (`pinned=True`) dresses it with satellites as well, so
    the two constructions disagree on which orbitals carry a quasiparticle and
    the set the filter selects is exactly the Z > QP_WINDOW_Z_MIN one.
    """
    _, mf, eri, nocc, norb = water
    filtered = production_bse.BSEqb(mf, eri, nocc)
    everything = production_bse.BSEqb(mf, eri, nocc, pinned=True)
    qp = production_qb.QPqb(mf.mo_energy, eri, nocc)
    expected = {p for p in range(norb) if qp.solve_diag(p)[1] > QP_WINDOW_Z_MIN}
    assert set(filtered.qp_roots) == expected
    assert set(everything.qp_roots) == set(range(norb))
    assert expected != set(range(norb))


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_integral_blocks_are_bitwise_through_both_roots(tag, h4, water):
    mol, mf, eri, nocc, _ = systems(h4, water)[tag]
    prod = production_eri.MOEriBlocks.from_dense(eri, nocc)
    shim = shim_eri.MOEriBlocks.from_dense(eri, nocc)
    for name in ('ovov', 'oovv', 'pqov', 'vovo'):
        assert bitwise(getattr(prod, name), getattr(shim, name)), name
    assert shim_eri.as_blocks(prod, nocc) is prod
    assert bitwise(production_eri.mo_eri(mf, mol), eri)
    auxmol = gto.M(atom=[(mol.atom_pure_symbol(i), tuple(c)) for i, c
                         in enumerate(mol.atom_coords())],
                   unit='Bohr', basis='cc-pvdz-ri', verbose=0)
    assert bitwise(production_eri.df_eri_mo(mol, auxmol, mf.mo_coeff),
                   shim_df.df_eri_mo(mol, auxmol, mf.mo_coeff))


def test_the_bse_partials_only_exist_on_the_adjoint_object(water):
    """The reverse mode is what `gradients` keeps, and it is bitwise its own."""
    _, mf, eri, nocc, _ = water
    forward = production_bse.BSEqb(mf, eri, nocc)
    assert not hasattr(forward, 'partials')
    assert not hasattr(forward.qp.qb, 'gamma_A')
    b = shim_bse.BSEqb(mf, eri, nocc)
    gF, G4, tg = b.partials(0)
    gF2, G42, tg2 = adjoint.BSEqbAdjoint(mf, eri, nocc).partials(0)
    assert bitwise(gF, gF2) and bitwise(G4, G42) and bitwise(tg, tg2)


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_route_still_computes_the_same_physics(tag, h4, water):
    """E_c, the lowest singlet excitation, the HOMO quasiparticle, the QP set."""
    _, mf, eri, nocc, _ = systems(h4, water)[tag]
    want = PINNED[tag]
    A, B, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc)
    assert production_qb.RPA(A, B).e_corr() == pytest.approx(
        want['e_corr'], abs=PHYSICS_TOL)
    qp = production_qb.QPqb(mf.mo_energy, eri, nocc)
    assert qp.solve_diag(nocc - 1)[0] == pytest.approx(want['homo_qp'],
                                                       abs=PHYSICS_TOL)
    b = production_bse.BSEqb(mf, eri, nocc)
    assert float(b.Omega[0]) == pytest.approx(want['omega0'], abs=PHYSICS_TOL)
    assert len(b.qp_roots) == want['n_qp']


# ---------------------------------------------------------------------------
# the dedup verdicts, as the numbers they were decided on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_build_rpa_AB_is_the_casida_builder_bitwise_and_still_stays(tag, h4,
                                                                   water):
    """Same blocks to the bit, kept separate for two reasons the gate names.

    `LinearResponseSolver` needs the full nao^4 `eri_chemist`, which is the
    array `MOEriBlocks` exists to avoid, and its `triplet=True` zeroes the
    exchange in A rather than zeroing B -- the TDA SCREENING variant this
    route needs has no spelling there.
    """
    _, mf, eri, nocc, _ = systems(h4, water)[tag]
    A, B, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc)
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=eri,
                              spin_mode='restricted')
    Aw, Bw = lr.build_casida_matrices(nocc, lBSE=False)
    assert bitwise(A, Aw) and bitwise(B, Bw)

    # tda_screening zeroes B and leaves A alone; triplet zeroes the exchange
    # in A and leaves B zero as well. Same B, different A: two variants.
    A_tda, B_tda, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc,
                                                 tda_screening=True)
    Awt, Bwt = lr.build_casida_matrices(nocc, lBSE=False, triplet=True)
    assert np.abs(B_tda).max() == 0.0 and np.abs(Bwt).max() == 0.0
    assert bitwise(A_tda, A) and not bitwise(A_tda, Awt)

    blocks = production_eri.MOEriBlocks.from_mol(mf.mol, mf.mo_coeff, nocc)
    assert blocks.dense is None
    Ab, _, _ = production_qb.build_rpa_AB(mf.mo_energy, blocks, nocc)
    assert Ab.shape == A.shape and np.abs(Ab - A).max() < 1e-13
    with pytest.raises(NotImplementedError):
        blocks.require_dense('the Casida builder')


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_plasmon_energy_is_not_the_casida_one_bitwise(tag, h4, water):
    """One trace formula, two eigensolvers, and the difference is recorded.

    E_c = (sum Omega - Tr A)/2 either way, but Omega comes from an eigh of
    Abar here and from the symplectic Casida solver there, so the two are not
    interchangeable at the bit level and both stay.
    """
    _, mf, eri, nocc, _ = systems(h4, water)[tag]
    A, B, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc)
    ours = production_qb.RPA(A, B).e_corr_plasmon()
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=eri,
                              spin_mode='restricted')
    theirs = float(rpa_correlation_energy_casida(lr, nocc))
    diff = abs(ours - theirs)
    assert diff != 0.0
    assert 0.3 * PLASMON_VS_CASIDA[tag] < diff < 3.0 * PLASMON_VS_CASIDA[tag]


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_from_mol_is_not_the_dense_four_index_builder_bitwise(tag, h4, water):
    """Block transforms and a full one are different `ao2mo.general` calls.

    H4/sto-3g agrees exactly and water/cc-pVDZ does not, which is why the
    verdict is not taken on the smaller molecule: `from_mol` stays its own
    builder, and a route that needs the dense tensor asks for it.
    """
    mol, mf, eri, nocc, norb = systems(h4, water)[tag]
    _, eri_prod = _two_electron_integrals(mol, mf, df=False, is_uhf=False)
    assert bitwise(eri_prod, production_eri.mo_eri(mf, mol))
    blocks = production_eri.MOEriBlocks.from_mol(mol, mf.mo_coeff, nocc)
    occ, virt = slice(0, nocc), slice(nocc, norb)
    diff = np.abs(blocks.oovv - eri_prod[occ, occ, virt, virt]).max()
    expected = FROM_MOL_VS_DENSE[tag]
    if expected == 0.0:
        assert diff == 0.0
    else:
        assert 0.3 * expected < diff < 3.0 * expected


def test_mo_eri_is_the_bare_interaction_and_not_the_environment_aware_one(h4):
    """They agree bitwise in the gas phase and mean different things.

    `pyscf_interface.get_two_electron_integrals_chemist`, which
    `_two_electron_integrals` returns, substitutes v -> v + vtilde wherever an
    environment is attached; `mo_eri` never consults one, which is what lets
    the dense route refuse a solvated mean field instead of silently reporting
    E_c[v + vtilde] under E_c[v].
    """
    mol, mf, _, _, _ = h4
    _, environment_aware = _two_electron_integrals(mol, mf, df=False,
                                                   is_uhf=False)
    assert bitwise(production_eri.mo_eri(mf, mol), environment_aware)
    assert production_eri.mo_eri is not get_two_electron_integrals_chemist


def test_the_quasiparticle_newton_keeps_its_own_loop(water):
    """The guarded Newton is the same root, except where it is a different one.

    `Solvers.qp_equation.solve_qp_equation_newton_guarded` returns this loop's
    root BITWISE on every water/cc-pVDZ orbital whose pole strength clears
    NEWTON_BITWISE_ABOVE_Z, and replaces the low-weight satellites by its
    linearized fallback -- up to NEWTON_SATELLITE_GAP Hartree away. Those
    satellite roots are exactly what `QP_WINDOW_Z_MIN` reads to decide which
    states carry a quasiparticle, so rerouting would change the QP set rather
    than move a root by an ulp. The undamped plain Newton is not bitwise even
    on the quasiparticles.
    """
    _, mf, eri, nocc, norb = water
    eps = np.asarray(mf.mo_energy, float)
    qp = production_qb.QPqb(eps, eri, nocc, screening='rpa')
    worst_satellite = 0.0
    plain_differs = 0
    for p in range(norb):
        w, z = qp.solve_diag(p)
        with warnings.catch_warnings():
            # the guarded Newton warns on every satellite it replaces
            warnings.simplefilter('ignore', RuntimeWarning)
            wg, _ = solve_qp_equation_newton_guarded(
                lambda x, p=p: qp.sigma(p, x),
                lambda x, p=p: qp.sigma_prime(p, x),
                eps, p, nocc, xc_correction=float(qp.delta[p]),
                w0=eps[p] + qp.delta[p], tol=1e-12, max_iter=200)
        if z > NEWTON_BITWISE_ABOVE_Z:
            assert wg == w, f'orbital {p}, Z={z:.3f}'
        else:
            worst_satellite = max(worst_satellite, abs(wg - w))
        wp = solve_qp_equation_newton(
            lambda x, p=p: x - eps[p] - qp.delta[p] - qp.sigma(p, x),
            eps[p] + qp.delta[p],
            deriv_func=lambda x, p=p: 1.0 - qp.sigma_prime(p, x),
            tol=1e-12, max_iter=200)
        plain_differs += int(wp != w)
    assert 0.3 * NEWTON_SATELLITE_GAP < worst_satellite
    assert plain_differs > 0


# ---------------------------------------------------------------------------
# production stands on its own
# ---------------------------------------------------------------------------

def test_production_imports_nothing_from_the_gradients_package():
    """A fresh interpreter reaches all three production modules without it."""
    code = ('import sys; '
            'import src.SingleReference.GW.quasi_boson; '
            'import src.SingleReference.LinearResponse.quasi_boson_bse; '
            'import src.Base.eri_blocks; '
            "print(sorted(m for m in sys.modules if m.startswith('src.gradients')))")
    repo = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=str(repo))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith('[]'), out.stdout
