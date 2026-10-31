"""The ISDF fit on every AO product in the Coulomb metric, and its pivoted points
(`src.Base.factorization.ao_product_fit`).

Gates, on water (C1 geometry):
  * the fit's equations are those of the explicit test set: the balanced Gram
    matrix and the projection equal the products formed over all nao^2 AO
    products, M is `fit_M_stable` on that test set, and D = M^T V^1/2 is the
    Coulomb-metric least-squares fit V^-1/2 (P|mn) ~ D^T (X_m X_n), to the
    rounding of the balanced Gram matrix (cc-pVDZ);
  * the selection, its Gram columns formed on demand, is the greedy pivoted
    Cholesky of the whole Gram matrix (largest remaining residual first), its
    prefixes nested; a count above the parent and a pivot above its memory
    bound are refused, a parent of lower numerical rank keeps that rank and
    flags it; three simulated ranks give the one-rank selection bit for bit
    and its projection and fit to rounding, the same bits on every rank;
  * the Numerics fields refuse what no route realizes; a named grid with no
    test set is the AO-product fit on `pivot_count` points (600 on water at
    G1), in its identity too, and the Duchemin-Blase fit, named, keeps the
    identity it had before the fields existed (as does the ISDF-K mean
    field's grid named alone, which keeps its fit);
  * a selection frozen at a reference geometry and carried by the body frame
    gives the same fitted Coulomb matrix after a rigid rotation and
    translation (1e-10 relative), and travels through a record;
  * on aug-cc-pVDZ (HF, mean-field diagonal) 800 pivoted points put the lowest
    three singlets and triplets within 0.5 meV of DF-RI (measured 0.13);
  * in a toluene continuum (the fit in the bare metric, the gauge dressed by
    `aux_metric_sqrt(auxmol, environment)`) the same roots stay within
    0.2 meV of DF-RI on the mean-field diagonal (measured 0.09); with G0W0
    within 2 meV (measured 1.24), the space-time against the Casida GW, which
    the G3 grid's factors show too (1.6 meV).
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.environment_seam import attach_environment
from src.Base.factorization import ao_product_fit as apf
from src.Base.factorization.separable_ri import (atomic_frames, aux_metric_sqrt,
                                                 fit_M_stable, fit_settings,
                                                 resolve_grid)
from src.Base.numerical_flags import collecting
from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.numerics.numerics import IDENTITY_RETIRED, Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.Base.utils.record_io import record_hash
from src.SingleReference.BSE.solve import solve_bse_df, solve_bse_isdf
from src.environments.solvent_screening import SolventScreening
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.rpa.factor_chain import FrozenFactorization

H2O_C1 = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
REG, = fit_settings(None, 'regularization')
#: Rounding of the fit through its balanced Gram matrix, relative.
FIT_ROUNDING = 1e-8
AO_NUMERICS = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                             'pivot_points': 150})


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    frames = atomic_frames(mol)[0]
    pts, owner, _ = apf.pivoted_points(mol, radii, origins, frames, 150)
    return mol, auxmol, apf.placed(mol, pts, frames)


def explicit(mol, auxmol, coords):
    """(D_test, (P|mn)) over every AO product: (nk, nao^2), (naux, nao^2)."""
    X = mol.eval_gto('GTOval_sph', coords)
    nao = mol.nao_nr()
    D = (X[:, :, None] * X[:, None, :]).reshape(len(coords), -1)
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    return D, e3c.reshape(nao * nao, -1).T


def test_the_equations_are_the_explicit_ao_products(water):
    mol, auxmol, coords = water
    D, pq = explicit(mol, auxmol, coords)
    X = mol.eval_gto('GTOval_sph', coords)
    G, s = apf.balanced_gram(X)
    Dt = D / np.linalg.norm(D, axis=1)[:, None]
    assert np.abs(s - np.linalg.norm(D, axis=1)).max() < 1e-12 * s.max()
    assert np.abs(G - (Dt @ Dt.T + REG * np.eye(len(D)))).max() < 1e-12
    E = apf.ao_product_projection(mol, auxmol, X, Numerics().derive(
        'many shell runs', layout={'fit_gb': 1e-3}))
    ref = pq @ D.T
    assert np.abs(E - ref).max() < 1e-12 * np.abs(ref).max()
    _, M, V = apf.ao_product_fit(mol, auxmol, coords)
    want = fit_M_stable(D, np.linalg.solve(V, pq))
    assert np.abs(M - want).max() < FIT_ROUNDING * np.abs(want).max()


def test_the_factor_is_the_coulomb_metric_least_squares_fit(water):
    """D^T minimizes sum_mn |V^-1/2 (P|mn) - D^T (X_m X_n)|^2 on balanced rows
    with the Tikhonov floor: Mt = T Dt^T (Dt Dt^T + reg)^-1, D = (Mt / s)^T."""
    mol, auxmol, coords = water
    D_test, pq = explicit(mol, auxmol, coords)
    s = np.linalg.norm(D_test, axis=1)
    Dt = D_test / s[:, None]
    T = np.linalg.solve(aux_metric_sqrt(auxmol), pq)          # V^-1/2 (P|mn)
    A = Dt @ Dt.T + REG * np.eye(len(Dt))
    Mt = scipy.linalg.solve(A, (T @ Dt.T).T, assume_a='pos').T
    want = (Mt / s[None, :]).T
    _, M, V = apf.ao_product_fit(mol, auxmol, coords)
    D = M.T @ aux_metric_sqrt(auxmol, None, V=V)
    assert np.abs(D - want).max() < FIT_ROUNDING * np.abs(want).max()


def greedy(G, count):
    """The textbook pivoted Cholesky: the largest remaining residual first."""
    R = G.copy()
    order = []
    for _ in range(count):
        k = int(np.argmax(np.diag(R)))
        order.append(k)
        R = R - np.outer(R[:, k], R[k, :]) / R[k, k]
    return np.array(order)


def parent(mol):
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    frames = atomic_frames(mol)[0]
    return apf.placed(mol, apf.parent_layout(mol, radii, origins)[0], frames)


def test_the_pivot_is_the_greedy_pivot_on_the_whole_gram_matrix(water):
    """Columns formed on demand pick what the textbook pivot picks on the
    whole balanced Gram matrix of the parent (1776 points)."""
    mol = water[0]
    coords = parent(mol)
    G, _ = apf.balanced_gram(mol.eval_gto('GTOval_sph', coords))
    # the diagonal at its exact value 1 + reg, as the pivot starts from: the
    # first step is a tie of every point, broken by the lowest index
    np.fill_diagonal(G, 1.0 + REG)
    want = greedy(G, 150)
    got = apf.column_pivots(mol, coords, 150)
    assert np.array_equal(got, want)
    # every count's selection is a prefix of the next one's
    assert np.array_equal(apf.column_pivots(mol, coords, 60), want[:60])


def test_a_count_beyond_the_parent_is_refused_and_a_short_rank_flagged():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    radii = {'H': {'A1': [0.5, 1.0, 2.0], 'A2': [1.0]}}
    frames = atomic_frames(mol)[0]
    n_parent = len(apf.parent_layout(mol, radii)[1])
    with pytest.raises(ValueError, match='exceeds'):
        apf.pivoted_points(mol, radii, None, frames, n_parent + 1)
    # without the floor the Gram matrix of two s functions' three products
    # has rank 3: the pivot keeps 3 points and says so
    bare = Numerics().derive('a Gram matrix without its floor',
                             isdf={'regularization': 0.0})
    flags = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', apf.ISDFPivotShort)
        with collecting(flags):
            _, owner, _ = apf.pivoted_points(mol, radii, None, frames, 20, bare)
    assert len(owner) == 3
    assert [f['flag'] for f in flags] == ['isdf_pivot_rank_short']


def test_the_pivot_and_the_fit_over_ranks(water):
    """Three ranks: the parent in fixed tiles gives the one-rank selection bit
    for bit; the shell runs divided over the ranks give the one-rank
    projection and fit to rounding (their partials group the runs
    otherwise), the same bits on every rank."""
    mol, auxmol, coords = water
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    frames = atomic_frames(mol)[0]
    big = np.vstack([coords] * 4)        # several point tiles

    def run(comm):
        owner = apf.pivoted_points(mol, radii, origins, frames, 150)[1]
        return owner, apf.ao_product_projection(mol, auxmol, mol.eval_gto(
            'GTOval_sph', big), Numerics().derive('several shell runs',
                                                  layout={'fit_gb': 1e-3})), \
            apf.ao_product_fit(mol, auxmol, coords)[1]
    serial = run(None)
    ranks = run_simulated(run, 3)
    for owner, E, M in ranks:
        assert np.array_equal(owner, serial[0])
        assert np.abs(E - serial[1]).max() < 1e-13 * np.abs(serial[1]).max()
        assert np.abs(M - serial[2]).max() < FIT_ROUNDING * np.abs(serial[2]).max()
        assert np.array_equal(E, ranks[0][1]) and np.array_equal(M, ranks[0][2])


def test_the_ranks_agree_on_a_pivot_only_one_of_them_holds(water):
    """A pivot kept from before on one rank alone (the others' entries
    dropped: simulated ranks share one table) is not read there: every rank
    pivots again, in the same collectives, and gets the one-rank selection.
    Rank 0's stale entry is a wrong selection, so reading it would show; the
    group runs under a time limit, since a rank skipping the collectives
    hangs the others."""
    mol = water[0]
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    frames = atomic_frames(mol)[0]
    serial = apf.pivoted_points(mol, radii, origins, frames, 150)[1]
    parent = apf.placed(mol, apf.parent_layout(mol, radii, origins)[0], frames)

    def run(comm):
        if comm.Get_rank() == 0:
            stale = apf.pivot_key(mol, parent, 150, None, comm)
            with apf._MADE_LOCK:
                apf._MADE[stale] = np.arange(150)
        return apf.pivoted_points(mol, radii, origins, frames, 150)[1]
    out = []
    group = threading.Thread(target=lambda: out.append(run_simulated(run, 2)),
                             daemon=True)
    group.start()
    group.join(300)
    assert not group.is_alive(), 'the ranks entered different collectives'
    for owner in out[0]:
        assert np.array_equal(owner, serial)


def test_a_pivot_beyond_its_memory_bound_is_refused(water, monkeypatch):
    mol = water[0]
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    monkeypatch.setattr(apf, 'ISDF_PIVOT_MAX_GB', 1e-3)
    with pytest.raises(MemoryError, match='ISDF_PIVOT_MAX_GB'):
        apf.pivoted_points(mol, radii, origins, atomic_frames(mol)[0], 150)


def test_the_numerics_refuse_what_no_route_realizes():
    with pytest.raises(ValueError, match='test_set'):
        Numerics(isdf={'test_set': 'orbital_pairs'})
    with pytest.raises(ValueError, match="test_set='ao_products'"):
        Numerics(isdf={'test_set': 'duchemin_blase', 'pivot_points': 100})
    with pytest.raises(ValueError, match='positive'):
        Numerics(isdf={'test_set': 'ao_products', 'pivot_points': 0})
    Numerics(isdf={'level': 'G3', 'pivot_points': 100})   # the AO-product fit


def old_identity(numerics):
    """The identity written before test_set and pivot_points existed."""
    values = numerics.values()
    for name in ('test_set', 'pivot_points', 'runtime_radii'):
        values['isdf'].pop(name)
    values['solvent'].pop('loop_force_fraction')   # left out at its default
    for section, name, held in IDENTITY_RETIRED:   # written then, at the value held
        values[section][name] = held
    return record_hash({k: values[k] for k in ('isdf', 'scf', 'span', 'qp',
                                               'quadrature', 'bse', 'solvent',
                                               'walk')})


def test_a_named_grid_is_the_ao_product_fit_in_the_identity():
    """The Duchemin-Blase fit named keeps the identity it had before the
    fields existed (it was what a named level meant); a level named alone is
    the AO-product fit and says so; nothing named keeps its identity."""
    legacy = Numerics(isdf={'level': 'G3', 'test_set': 'duchemin_blase'})
    assert legacy.identity() == old_identity(legacy)
    plain = Numerics(isdf={'level': 'G3'})
    assert plain.isdf.test_set_named() == 'ao_products'
    assert plain.identity() != old_identity(plain)
    assert plain.identity() == Numerics(
        isdf={'level': 'G3', 'test_set': 'ao_products'}).identity()
    assert Numerics().identity() == old_identity(Numerics())
    assert Numerics().isdf.test_set_named() is None
    # the ISDF-K mean field's grid alone keeps its fit and its identity
    isdf_k = Numerics(scf={'grid_level': 'G1'})
    assert isdf_k.identity() == old_identity(isdf_k)


def test_the_pivot_count_rule():
    """max(ISDF_PIVOT_FRACTION x parent, ISDF_PIVOT_POINTS_PER_ATOM x atoms),
    at most the parent: the counts the validated emitters get at the G3
    parent (888 points per atom)."""
    assert apf.pivot_count(62, 55056) == 12400        # CBP
    assert apf.pivot_count(60, 53280) == 12000        # PXZ-TRZ
    assert apf.pivot_count(21, 18648) == 4200         # DMABN
    assert apf.pivot_count(3, 2664) == 600            # water
    assert apf.pivot_count(1, 150) == 150             # never above the parent
    assert apf.pivot_count(2, 20000) == 3200          # the fraction decides


def test_a_named_level_pivots_the_ao_product_fit(water):
    mol = water[0]
    radii, origins, _, _ = resolve_grid(mol, Numerics(isdf={'level': 'G1'}).isdf)
    n_parent = len(apf.parent_layout(mol, radii, origins)[1])
    fz = FrozenFactorization(mol, Numerics(isdf={'level': 'G1'}))
    assert fz.test_set == 'ao_products' and fz.layout is None
    assert fz.M == apf.pivot_count(mol.natm, n_parent) == 600
    old = FrozenFactorization(mol, Numerics(isdf={'level': 'G1',
                                                  'test_set': 'duchemin_blase'}))
    assert old.layout is not None and 2 * old.M == n_parent   # the grid as is
    # a chain naming a grid asks for the AO-product fit; one naming none
    # accepts whatever it is handed
    with pytest.raises(ValueError, match='was built for'):
        old.require_match(Numerics(isdf={'level': 'G1'}))
    old.require_match(Numerics())


def moved(mol, rot, shift):
    x = mol.atom_coords()
    c = x.mean(axis=0)
    m = mol.copy()
    m.set_geom_((x - c) @ rot.T + c + shift, unit='Bohr')
    m.build(False, False)
    return m


def test_a_frozen_selection_moves_with_the_molecule(water):
    mol = water[0]
    fz = FrozenFactorization(mol, AO_NUMERICS)
    assert fz.M == 150 and fz.layout is None and fz.parent_size > fz.M
    rot = scipy.linalg.expm(np.array([[0, -0.3, 0.2], [0.3, 0, -0.5],
                                      [-0.2, 0.5, 0]]))
    shift = np.array([0.3, -0.2, 0.4])
    m2 = moved(mol, rot, shift)
    c0, c2 = fz.coords(mol), fz.coords(m2)
    centre = mol.atom_coords().mean(axis=0)
    assert np.abs((c0 - centre) @ rot.T + centre + shift - c2).max() < 1e-12

    def coulomb(m):
        aux = fz.auxmol(m)
        _, M, V = fz._fit(m, aux, fz.coords(m))
        D = M.T @ aux_metric_sqrt(aux, None, V=V)
        return D @ D.T
    z0, z2 = coulomb(mol), coulomb(m2)
    assert np.abs(z2 - z0).max() < 1e-10 * np.abs(z0).max()


def test_the_selection_travels_through_a_record(water):
    mol = water[0]
    fz = FrozenFactorization(mol, AO_NUMERICS)
    back = FrozenFactorization.from_record(fz.to_record())
    assert back.test_set == 'ao_products' and back.pivot_points == 150
    assert np.array_equal(back.coords(mol), fz.coords(mol))
    old = {k: v for k, v in fz.to_record().items()
           if k not in ('test_set', 'pivot_points')}
    plain = FrozenFactorization.from_record(old)
    assert plain.test_set == 'duchemin_blase' and plain.pivot_points is None


def test_the_chain_settings_must_name_the_same_fit(water):
    mol = water[0]
    fz = FrozenFactorization(mol, AO_NUMERICS)
    fz.require_match(Numerics())                 # no opinion: accepted
    fz.require_match(AO_NUMERICS)
    with pytest.raises(ValueError, match='was built for'):
        fz.require_match(Numerics(isdf={'test_set': 'ao_products',
                                        'pivot_points': 160}))


def test_augmented_bse_against_df():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='aug-cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-ri')
    mf.conv_tol = 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    bse = Numerics(bse={'nroots': 3, 'conv_tol': 1e-9})
    factors = separable_factors(mf, mol, numerics=Numerics(
        isdf={'level': 'G3', 'test_set': 'ao_products', 'pivot_points': 800}))
    assert len(factors[3]) == 800
    for spin in ('singlet', 'triplet'):
        w_df = solve_bse_df(mf, mol, nocc, numerics=bse, qp=None, spin=spin)[0]
        w = solve_bse_isdf(mf, mol, nocc, numerics=bse, qp=None, spin=spin,
                           factors=factors)[0]
        err = 1e3 * HARTREE_TO_EV * (np.sort(w)[:3] - np.sort(w_df)[:3])
        assert np.abs(err).max() < 0.5, (spin, err)



def test_augmented_bse_in_a_continuum_against_df():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='aug-cc-pvdz', verbose=0)

    def factory(m):
        mf = scf.RHF(m).density_fit(auxbasis='aug-cc-pvdz-ri')
        mf.conv_tol = 1e-11
        mf.kernel()
        return mf
    env = SolventScreening(mol, solvent='toluene')
    mf = env.mean_field(mol, factory)
    attach_environment(mf, env)
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, numerics=Numerics(
        isdf={'level': 'G3', 'test_set': 'ao_products', 'pivot_points': 800}))
    for qp, bar in ((None, 0.2), ('G0W0', 2.0)):
        for spin in ('singlet', 'triplet'):
            kw = dict(qp=qp, spin=spin, probe=False,
                      numerics=Numerics(bse={'nroots': 3, 'conv_tol': 1e-9}))
            w_df = solve_bse_df(mf, mol, nocc, **kw)[0]
            w = solve_bse_isdf(mf, mol, nocc, factors=factors, **kw)[0]
            err = 1e3 * HARTREE_TO_EV * (np.sort(w)[:3] - np.sort(w_df)[:3])
            assert np.abs(err).max() < bar, (qp, spin, err)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
