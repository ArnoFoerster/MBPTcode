"""The dense quasi-boson route computes its numbers in production.

The forward physics of the dense oracle -- the closed-form dRPA ground state,
its bare boson couplings, the diagonal EOM G0W0 quasiparticle solve and the
BSE@G0W0 eigenproblem, together with the MO integral blocks and the fitted
interaction all four are built from -- lives in `SingleReference.GW.quasi_boson`,
`SingleReference.LinearResponse.quasi_boson_bse` and `Base.eri_blocks`. The
reverse mode lives in `gradients.quasi_boson_adjoint` as three subclasses of
production's classes.

WHAT IS GATED. That the three adjoint classes subclass production's and build
their own adjoint parts, and that `gradients.df_assembly` fits with
production's builders rather than a copy; that every recorded forward quantity
is the same to a few ulp on the adjoint class as on the forward one (two
objects computing it); the physics
itself as pinned numbers; the dedup verdicts as measured numbers; and that a
fresh interpreter can import all three production modules without
`src.gradients` appearing in sys.modules.

What breaks each gate:

  * the screened term of the BSE direct kernel flipped in sign
    (`quasi_boson_bse`, Wd4 = eri4.oovv + S_d):
    `test_the_route_still_computes_the_same_physics` on both systems; it
    moves the frozen water DenseBSESurface[singlet] R0 row by 7.6e-2 Ha in
    energy and 9.7e-3 Ha/Bohr in the gradient.
  * the pole-strength filter dropped (`quasi_boson_bse`, every root accepted
    regardless of Z): `test_the_pole_strength_filter_is_the_declared_constant`
    and the water physics pin; the baseline row moves 2.3e-5 Ha and 1.7e-4
    Ha/Bohr, and the quasiparticle set grows from 15 states to 19.
  * `QPqbAdjoint.boson` removed, so a gradient object carries the forward-only
    RPA:
    `test_the_adjoint_classes_subclass_production_and_build_their_own_parts`,
    `test_the_bse_partials_only_exist_on_the_adjoint_object` and
    `tests/test_gradients_qb.py::test_blocks_refuse_gradients_rather_than_guess`.

A few-ulp gate between the forward and the adjoint class compares one body of
forward code reached two ways, so it sees an adjoint subclass that overrode
the forward arithmetic and CANNOT see a kernel that changed sign. Physics is
held by the pinned numbers below, by the frozen baseline record and by the
existing surface tests.

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

import src.SingleReference.common.eri_blocks as production_eri
import src.SingleReference.GW.quasi_boson as production_qb
import src.SingleReference.BSE.quasi_boson_bse as production_bse
import src.gradients.df_assembly as df_assembly
import src.gradients.gwbse.quasi_boson_adjoint as adjoint
from src.Base.numerics.constants import (QP_WINDOW_Z_MIN, SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.pyscf.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.GW.qp_energy import _two_electron_integrals
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.dense_qp import dense_max_weight_root
from src.SingleReference.LinearResponse.rpa_energy import rpa_correlation_energy_casida
from tests.pins import (assert_same, bitwise, blas_identity, pin_bar,
                        recorded_blas)

#: The two systems every number below is measured on.
H4_ATOM = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'
WATER_ATOM = 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692'

#: `e_corr_plasmon` against `rpa_correlation_energy_casida`, in Hartree. Not
#: bitwise: the same trace formula off two different eigensolvers. The values
#: are the difference on the BLAS of tests/one_fit_pins.json (the workstation);
#: another BLAS rounds the two eigensolves differently (water 2.3e-13 on a cluster's
#: MKL 2025.3), and the difference then only has to stay under the energy bar
#: of tests/pins.py.
PLASMON_VS_CASIDA = {'h4': 1.78e-15, 'water': 5.68e-14}
RECORDING_BLAS = recorded_blas(pathlib.Path(__file__).resolve().parent
                               / 'one_fit_pins.json')

#: `MOEriBlocks.from_mol` against the four-index builder production's
#: quasiparticle driver uses, largest element of (ij|ab), in Hartree. Zero on
#: H4/sto-3g -- the smallest molecule does not see this one.
FROM_MOL_VS_DENSE = {'h4': 0.0, 'water': 4.56e-15}

#: `QPqb.solve_diag` is the exact global solve: the largest-weight root, or
#: the root bracketed by the poles around a seed. A Newton from eps_p + delta_p
#: lands on a satellite on water/cc-pVDZ orbitals 15 and 17-23 (Z 0.01-0.38,
#: up to 4.0 eV above the global root, whose Z is 0.2-0.53); orbital 15's
#: global weight is pinned here.
GLOBAL_Z_ORBITAL_15 = 0.53

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
    # the global quasiparticle solve takes orbital 15's largest-weight root
    # (Z 0.530), not the Newton satellite (Z 0.376, 0.546 eV higher), so the
    # Z > QP_WINDOW_Z_MIN set contains it; the Newton satellite would put the
    # lowest singlet 0.837 meV higher (0.3104456904334518).
    'water': {'e_corr': -0.2313009545750493, 'omega0': 0.31041494986123275,
              'homo_qp': -0.44679715112656565, 'n_qp': 17},
}


def build(atom, basis, unit='Angstrom'):
    """(mol, mf, eri, nocc, norb) on a tightly converged RHF reference."""
    mol = gto.M(atom=atom, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
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


# ---------------------------------------------------------------------------
# the gradient side holds no second copy of the forward
# ---------------------------------------------------------------------------

def test_df_assembly_reuses_the_production_fit():
    """Identity, not equality: a copy would drift the next time one is edited."""
    assert df_assembly.df_integrals is production_eri.df_integrals


def test_the_adjoint_classes_subclass_production_and_build_their_own_parts():
    """The forward-only classes carry no Frechet map and no partials, so a
    gradient object that built a forward part would fail at its first chain
    rule rather than return a different number."""
    assert issubclass(adjoint.RPAAdjoint, production_qb.RPA)
    assert issubclass(adjoint.QPqbAdjoint, production_qb.QPqb)
    assert issubclass(adjoint.BSEqbAdjoint, production_bse.BSEqb)
    # the composition seam: a forward object builds forward parts
    assert production_qb.QPqb.boson is production_qb.RPA
    assert adjoint.QPqbAdjoint.boson is adjoint.RPAAdjoint
    assert production_bse.BSEqb.quasiparticle is production_qb.QPqb
    assert adjoint.BSEqbAdjoint.quasiparticle is adjoint.QPqbAdjoint


# ---------------------------------------------------------------------------
# the adjoint classes compute the forward numbers to a few ulp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_adjoint_rpa_is_bitwise_the_forward_one(tag, h4, water):
    _, mf, eri, nocc, _ = systems(h4, water)[tag]
    A, B, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc)
    prod, adj = production_qb.RPA(A, B), adjoint.RPAAdjoint(A, B)
    for name in ('t', 'wt', 'Vt', 'exp_t', 'exp_mt', 'cosh_t', 'sinh_t',
                 'Abar', 'omega'):
        assert_same(getattr(adj, name), getattr(prod, name), quantity='root',
                    label=name)
    assert_same(adj.eigAbar[1], prod.eigAbar[1], unit='1', quantity='vector',
                label='eigenvectors of Abar')
    assert_same(adj.e_corr(), prod.e_corr(), label='E_c')
    assert_same(adj.e_corr_plasmon(), prod.e_corr_plasmon(),
                label='E_c plasmon')


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_the_adjoint_quasiparticle_solve_is_bitwise_the_forward_one(tag, h4,
                                                                    water):
    _, mf, eri, nocc, norb = systems(h4, water)[tag]
    prod = production_qb.QPqb(mf.mo_energy, eri, nocc, screening='rpa')
    adj = adjoint.QPqbAdjoint(mf.mo_energy, eri, nocc, screening='rpa')
    assert_same(adj.Wnu, prod.Wnu, quantity='root', label='Wnu')
    assert_same(adj.omega, prod.omega, quantity='root', label='omega')
    for p in range(norb):
        assert_same(adj.solve_diag(p), prod.solve_diag(p), label=('root, Z', p))
        assert_same(adj.sigma(p, mf.mo_energy[p]),
                    prod.sigma(p, mf.mo_energy[p]), label=('Sigma', p))
    p = nocc - 1
    assert_same(adj.supermatrix_diag(p), prod.supermatrix_diag(p),
                quantity='root', label='supermatrix diagonal')
    assert_same(dense_max_weight_root(adj, p), dense_max_weight_root(prod, p),
                label='dense root, Z')


@pytest.mark.parametrize('screening,bse_tda', [('rpa', False), ('rpa', True),
                                               ('tda', False), ('tda', True)])
def test_the_adjoint_bse_eigenpairs_are_bitwise_the_forward_ones(screening,
                                                                 bse_tda,
                                                                 water):
    _, mf, eri, nocc, _ = water
    kw = dict(screening=screening, bse_tda=bse_tda)
    for spin in ('singlet', 'triplet'):
        prod = production_bse.BSEqb(mf, eri, nocc, spin=spin, **kw)
        adj = adjoint.BSEqbAdjoint(mf, eri, nocc, spin=spin, **kw)
        assert_same(adj.A_bse, prod.A_bse, quantity='root', label='A')
        assert (prod.B_bse is None) == (adj.B_bse is None)
        if prod.B_bse is not None:
            assert_same(adj.B_bse, prod.B_bse, quantity='root', label='B')
        assert_same(adj.Pinv, prod.Pinv, quantity='root', label='Pinv')
        assert_same(adj.eps_qp, prod.eps_qp, label='eps_qp')
        assert prod.qp_roots == adj.qp_roots
        assert_same(adj.Omega, prod.Omega, quantity='root', label='Omega')
        assert_same(adj.X, prod.X, unit='1', quantity='vector', label='X')
        assert_same(adj.Y, prod.Y, unit='1', quantity='vector', label='Y')


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
def test_as_blocks_passes_blocks_through_and_mo_eri_is_the_dense_transform(
        tag, h4, water):
    mol, mf, eri, nocc, _ = systems(h4, water)[tag]
    prod = production_eri.MOEriBlocks.from_dense(eri, nocc)
    assert production_eri.as_blocks(prod, nocc) is prod
    assert_same(production_eri.mo_eri(mf, mol), eri, quantity='root',
                label='mo_eri')


def test_the_bse_partials_only_exist_on_the_adjoint_object(water):
    """The reverse mode is what `gradients` keeps, and it is its own to a few
    ulp."""
    _, mf, eri, nocc, _ = water
    forward = production_bse.BSEqb(mf, eri, nocc)
    assert not hasattr(forward, 'partials')
    assert not hasattr(forward.qp.qb, 'gamma_A')
    b = adjoint.BSEqbAdjoint(mf, eri, nocc)
    gF, G4, tg = b.partials(0)
    gF2, G42, tg2 = adjoint.BSEqbAdjoint(mf, eri, nocc).partials(0)
    assert_same(gF2, gF, quantity='root', label='gF')
    assert_same(G42, G4, quantity='root', label='G4')
    assert_same(tg2, tg, quantity='root', label='tg')


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
    """Same blocks to a few ulp, kept separate for two reasons the gate names.

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
    assert_same(Aw, A, quantity='root', label='A')
    assert_same(Bw, B, quantity='root', label='B')

    # tda_screening zeroes B and leaves A alone; triplet zeroes the exchange
    # in A and leaves B zero as well. Same B, different A: two variants.
    A_tda, B_tda, _ = production_qb.build_rpa_AB(mf.mo_energy, eri, nocc,
                                                 tda_screening=True)
    Awt, Bwt = lr.build_casida_matrices(nocc, lBSE=False, triplet=True)
    assert np.abs(B_tda).max() == 0.0 and np.abs(Bwt).max() == 0.0
    assert_same(A_tda, A, quantity='root', label='tda_screening leaves A')
    assert not bitwise(A_tda, Awt)

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
    if RECORDING_BLAS == blas_identity():
        assert diff != 0.0
        assert 0.3 * PLASMON_VS_CASIDA[tag] < diff < 3.0 * PLASMON_VS_CASIDA[tag]
    else:
        assert diff < pin_bar(RECORDING_BLAS)


@pytest.mark.parametrize('tag', ('h4', 'water'))
def test_from_mol_is_not_the_dense_four_index_builder_bitwise(tag, h4, water):
    """Block transforms and a full one are different `ao2mo.general` calls.

    H4/sto-3g agrees exactly and water/cc-pVDZ does not, which is why the
    verdict is not taken on the smaller molecule: `from_mol` stays its own
    builder, and a route that needs the dense tensor asks for it.
    """
    mol, mf, eri, nocc, norb = systems(h4, water)[tag]
    _, eri_prod = _two_electron_integrals(mol, mf, df=False, is_uhf=False)
    assert_same(eri_prod, production_eri.mo_eri(mf, mol), quantity='root',
                label='mo_eri')
    blocks = production_eri.MOEriBlocks.from_mol(mol, mf.mo_coeff, nocc)
    occ, virt = slice(0, nocc), slice(nocc, norb)
    diff = np.abs(blocks.oovv - eri_prod[occ, occ, virt, virt]).max()
    expected = FROM_MOL_VS_DENSE[tag]
    if expected != 0.0 and RECORDING_BLAS == blas_identity():
        assert 0.3 * expected < diff < 3.0 * expected
    else:
        # the recorded difference is the recording BLAS's rounding; elsewhere
        # the two builders are the same integrals within the class bar
        assert_same(blocks.oovv, eri_prod[occ, occ, virt, virt],
                    quantity='root', label='oovv')


def test_mo_eri_is_the_bare_interaction_and_not_the_environment_aware_one(h4):
    """They agree to a few ulp in the gas phase and mean different things.

    `pyscf_interface.get_two_electron_integrals_chemist`, which
    `_two_electron_integrals` returns, substitutes v -> v + vtilde wherever an
    environment is attached; `mo_eri` never consults one, which is what lets
    the dense route refuse a solvated mean field instead of silently reporting
    E_c[v + vtilde] under E_c[v].
    """
    mol, mf, _, _, _ = h4
    _, environment_aware = _two_electron_integrals(mol, mf, df=False,
                                                   is_uhf=False)
    assert_same(production_eri.mo_eri(mf, mol), environment_aware,
                quantity='root', label='gas phase')
    assert production_eri.mo_eri is not get_two_electron_integrals_chemist


def test_the_quasiparticle_root_is_the_global_one(water):
    """solve_diag is the largest-weight root of every root, and with that
    root as a seed it returns it again: the seeded rule follows a root, it
    does not move it. Following the bare level eps_p + delta_p instead would
    not: on orbital 15 its bracket holds a satellite below QP_WINDOW_Z_MIN."""
    _, mf, eri, nocc, norb = water
    eps = np.asarray(mf.mo_energy, float)
    qp = production_qb.QPqb(eps, eri, nocc, screening='rpa')
    for p in range(norb):
        w, z = qp.solve_diag(p)
        assert_same((w, z), qp.quasiparticle_root(p), label=('root, Z', p))
        w_seeded, z_seeded = qp.solve_diag(p, w0=w)
        assert abs(w_seeded - w) < 1e-10 and abs(z_seeded - z) < 1e-8
    assert abs(qp.solve_diag(15)[1] - GLOBAL_Z_ORBITAL_15) < 5e-3
    _, z_bare = qp.solve_diag(15, w0=eps[15] + qp.delta[15])
    assert z_bare < QP_WINDOW_Z_MIN < qp.solve_diag(15)[1]


# ---------------------------------------------------------------------------
# production stands on its own
# ---------------------------------------------------------------------------

def test_production_imports_nothing_from_the_gradients_package():
    """A fresh interpreter reaches all three production modules without it."""
    code = ('import sys; '
            'import src.SingleReference.GW.quasi_boson; '
            'import src.SingleReference.BSE.quasi_boson_bse; '
            'import src.SingleReference.common.eri_blocks; '
            "print(sorted(m for m in sys.modules if m.startswith('src.gradients')))")
    repo = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=str(repo))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith('[]'), out.stdout
