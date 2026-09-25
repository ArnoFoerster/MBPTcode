"""One factorization, shared by several chains, instead of one per chain.

`separable_factors` is deterministic given its settings, so two chains handed
matching keywords already produce bitwise identical factors -- they simply pay
for them twice. This object makes the sharing STRUCTURAL rather than a property
of the caller remembering to pass the same keywords, and halves the cost of any
composed surface.

THE GATE IS BITWISE EQUALITY, NOT A FINITE DIFFERENCE. A re-measured FD at 1e-8
would silently accept a divergence; bitwise catches any setting that quietly
stopped being shared. If it is ever not bitwise, that is a bug to find rather
than a number to re-baseline.

The quadratures are deliberately NOT here: the correlation energy integrates
chi0 over frequency and the self-energy over imaginary time, different
integrands with no reason to share a grid.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import BOHR_TO_ANGSTROM
from src.gradients import factor_chain as fc_module
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.rpa_ground_state import RPAGroundStateChain


def rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis='cc-pvdz', verbose=0)


@pytest.fixture(scope='module')
def displaced(mol):
    """Where two independently frozen factorizations could diverge, unlike the
    reference geometry where they agree by construction."""
    crd = mol.atom_coords() * BOHR_TO_ANGSTROM
    crd[1, 0] += 0.05 * BOHR_TO_ANGSTROM
    m = mol.copy()
    m.set_geom_(crd, unit='Angstrom')
    m.build(False, False)
    return m


def test_a_chain_with_no_factorization_builds_its_own(mol):
    """The default path must not change: every existing chain constructs
    exactly as before and simply owns a factorization now."""
    chain = RPAGroundStateChain(mol, rhf)
    assert isinstance(chain.factorization, FrozenFactorization)
    assert chain.basis == chain.factorization.basis
    assert chain.radii is chain.factorization.radii
    assert chain.layout is chain.factorization.layout


def test_two_chains_on_one_factorization_share_the_arrays(mol):
    """Not equal -- the SAME objects, which is what makes the factors bitwise
    identical rather than merely close."""
    shared = FrozenFactorization(mol)
    a = RPAGroundStateChain(mol, rhf, factorization=shared)
    b = RPAGroundStateChain(mol, rhf, factorization=shared)
    assert a.radii is b.radii is shared.radii
    assert a.pts_local is b.pts_local
    assert a.frames is b.frames
    assert a.layout is b.layout
    assert a.M == b.M and a.naux == b.naux


def test_shared_factors_are_bitwise_identical_at_a_displaced_geometry(mol, displaced):
    """The gate. At the reference the two agree by construction, so the
    measurement has to be made where they could differ."""
    shared = FrozenFactorization(mol)
    a = RPAGroundStateChain(mol, rhf, factorization=shared)
    b = RPAGroundStateChain(mol, rhf, factorization=shared)
    mf = rhf(displaced)
    xa, da, epsa, _, crda, _ = a.factors_at(displaced, mf)
    xb, db, epsb, _, crdb, _ = b.factors_at(displaced, mf)
    assert np.array_equal(crda, crdb), 'interpolation points'
    assert np.array_equal(xa, xb), 'X_mo'
    # D moves first if a fit setting quietly stops being shared
    assert np.array_equal(da, db), 'D'
    assert np.array_equal(epsa, epsb), 'eps'


def test_an_unshared_pair_also_agrees_today_but_is_not_guaranteed_to(mol, displaced):
    """The premise behind the change: sharing is a COST optimization, not a
    physics fix. Two chains with matching settings already agree bitwise, which
    is why the shared object may not alter any number."""
    a = RPAGroundStateChain(mol, rhf)
    b = RPAGroundStateChain(mol, rhf)
    mf = rhf(displaced)
    xa, da, _, _, crda, _ = a.factors_at(displaced, mf)
    xb, db, _, _, crdb, _ = b.factors_at(displaced, mf)
    assert np.array_equal(crda, crdb)
    assert np.array_equal(xa, xb)
    assert np.array_equal(da, db)


def test_the_shared_factorization_is_built_once_not_per_chain(mol, monkeypatch):
    """The cost claim, counted rather than timed. Radii optimization is the
    expensive part of freezing a factorization."""
    calls = {'n': 0}
    original = fc_module.optimize_atomic_radii

    def counted(*a, **kw):
        calls['n'] += 1
        return original(*a, **kw)

    monkeypatch.setattr(fc_module, 'optimize_atomic_radii', counted)

    shared = FrozenFactorization(mol)
    once = calls['n']
    RPAGroundStateChain(mol, rhf, factorization=shared)
    RPAGroundStateChain(mol, rhf, factorization=shared)
    assert calls['n'] == once, 'a shared factorization must not re-optimize radii'

    calls['n'] = 0
    RPAGroundStateChain(mol, rhf)
    RPAGroundStateChain(mol, rhf)
    assert calls['n'] == 2 * once, 'unshared chains each pay for their own'


def test_a_contradicting_setting_is_refused_not_resolved(mol):
    """Silently resolving to one side would hand the caller a functional it did
    not ask for, with every number downstream wrong for a reason nothing
    reports."""
    shared = FrozenFactorization(mol, n_start=1)
    with pytest.raises(ValueError, match='shared factorization was built for'):
        RPAGroundStateChain(mol, rhf, factorization=shared, n_start=3)
    with pytest.raises(ValueError, match='shared factorization was built for'):
        RPAGroundStateChain(mol, rhf, factorization=shared, frames='continued')


def test_settings_that_agree_are_accepted(mol):
    """The refusal must not fire on a caller who passes the same values
    explicitly -- that is a legitimate way to write it."""
    shared = FrozenFactorization(mol)
    chain = RPAGroundStateChain(mol, rhf, factorization=shared,
                                basis=shared.basis, auxbasis=shared.auxbasis,
                                n_start=shared.n_start,
                                frames=shared.frames_mode)
    assert chain.factorization is shared


def test_the_quadrature_is_not_part_of_the_factorization(mol):
    """Two chains may share a factorization and still integrate differently:
    the correlation energy's frequency grid and the self-energy's imaginary-time
    grid are quadrature for different integrands."""
    shared = FrozenFactorization(mol)
    assert not hasattr(shared, 'ntau')
    assert not hasattr(shared, 'grid')
    a = RPAGroundStateChain(mol, rhf, factorization=shared, ntau=12)
    b = RPAGroundStateChain(mol, rhf, factorization=shared, ntau=14)
    assert a.ntau != b.ntau
    assert a.factorization is b.factorization


def test_refreeze_can_carry_a_factorization_built_at_the_new_geometry(mol, displaced):
    """A factorization is frozen AT a geometry, so refreeze must build a NEW one
    rather than carry the old. The keyword lets a COMPOSED surface build one at
    the new geometry and hand it to both halves, which is the only way the
    sharing survives the first optimizer step."""
    shared_new = FrozenFactorization(displaced)
    a = RPAGroundStateChain(mol, rhf).refreeze(displaced, factorization=shared_new)
    b = RPAGroundStateChain(mol, rhf).refreeze(displaced, factorization=shared_new)
    assert a.factorization is shared_new
    assert b.factorization is shared_new
    assert a.radii is b.radii


def test_refreeze_without_one_behaves_as_before(mol, displaced):
    """The default must not move: every existing caller passes no factorization
    and must get a chain that builds its own at the new geometry."""
    plain = RPAGroundStateChain(mol, rhf).refreeze(displaced)
    assert isinstance(plain.factorization, FrozenFactorization)
    assert plain.factorization.basis == plain.basis


def test_the_per_geometry_fit_is_paid_once_for_all_chains(mol):
    """WHAT SHARING BUYS, and the one piece that stays per chain.

    `FrozenFactorization.shareable_factors` holds everything the ENVIRONMENT
    does not touch -- the test-set overlaps, the two- and three-centre
    integrals and the least-squares fit, 96.6% of the per-geometry cost -- so
    several chains of one composed surface pay it ONCE.

    `factors` is still entered per chain, and must be: its last step dresses
    the auxiliary gauge with that chain's own environment. Two chains sharing a
    layout and carrying different continua have different D, so a cache drawn
    after the gauge would hand one chain the other's screening.
    """
    shared = FrozenFactorization(mol)
    a = RPAGroundStateChain(mol, rhf, factorization=shared)
    b = RPAGroundStateChain(mol, rhf, factorization=shared)

    entered, fits = {'n': 0}, {'n': 0}
    orig_factors, orig_fit = FactorChain.factors, FrozenFactorization.shareable_factors

    def count_factors(self, m, auxmol, crd):
        entered['n'] += 1
        return orig_factors(self, m, auxmol, crd)

    def count_fits(self, m, auxmol, crd):
        before = len(self._fit_cache)
        out = orig_fit(self, m, auxmol, crd)
        if len(self._fit_cache) > before:
            fits['n'] += 1
        return out

    FactorChain.factors = count_factors
    FrozenFactorization.shareable_factors = count_fits
    try:
        ea = a.total_energy()
        eb = b.total_energy()
    finally:
        FactorChain.factors = orig_factors
        FrozenFactorization.shareable_factors = orig_fit

    assert entered['n'] == 2, 'the gauge is per chain and stays per chain'
    assert fits['n'] == 1, 'the shareable fit must be computed once, not twice'
    assert ea == eb, 'sharing the fit must not change a digit'


def test_an_unshared_pair_still_pays_for_two_fits(mol):
    """Otherwise the test above would pass on a factorization that was shared
    by accident of there being only one."""
    fits = {'n': 0}
    orig_fit = FrozenFactorization.shareable_factors

    def count_fits(self, m, auxmol, crd):
        before = len(self._fit_cache)
        out = orig_fit(self, m, auxmol, crd)
        if len(self._fit_cache) > before:
            fits['n'] += 1
        return out

    FrozenFactorization.shareable_factors = count_fits
    try:
        RPAGroundStateChain(mol, rhf).total_energy()
        RPAGroundStateChain(mol, rhf).total_energy()
    finally:
        FrozenFactorization.shareable_factors = orig_fit
    assert fits['n'] == 2


def test_the_fit_cache_does_not_retain_dead_geometries(mol):
    """A WeakKeyDictionary keyed on the Mole is safe here ONLY because the
    value is arrays and holds no reference back to the key. The same container
    keyed on a value that owns its key never expires -- which is why the
    environment cache cannot be fixed this way."""
    import gc

    shared = FrozenFactorization(mol)
    chain = RPAGroundStateChain(mol, rhf, factorization=shared)
    crd = mol.atom_coords() * BOHR_TO_ANGSTROM
    for i in range(4):
        c = crd.copy()
        c[1, 0] += (i + 1) * 1e-3 * BOHR_TO_ANGSTROM
        m = mol.copy()
        m.set_geom_(c, unit='Angstrom')
        m.build(False, False)
        chain.total_energy(m, rhf(m))
        del m
    gc.collect()
    assert len(shared._fit_cache) <= 2, (
        f'{len(shared._fit_cache)} entries retained after four displaced '
        f'geometries went out of scope')
