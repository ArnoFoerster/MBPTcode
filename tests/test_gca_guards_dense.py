"""Guards on the dense gradient surfaces and the record files."""
import numpy as np
import pytest
from pyscf import gto

from src.Base.utils.record_io import read_record, write_record
from src.Base.geometry.geometry import molecule_at
from src.gradients.gwbse.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                                QuasiparticleSurface, tight_rhf)
from src.properties.hessian import surface_hessian
from src.gradients.mean_field_surface import MeanFieldSurface
from src.gradients.gwbse.surface_factory import compare_surfaces
from src.gradients.surface_identity import composed_identity, grid_record


def test_write_record_keeps_arrays_of_a_dotted_sibling_stem(tmp_path):
    """Writing `chain.json` must not delete the arrays of `chain.hess.json`."""
    write_record(tmp_path / 'chain.hess.json', {'h': np.eye(2)})
    write_record(tmp_path / 'chain.json', {'g': np.ones(3)})
    write_record(tmp_path / 'chain.json', {'g': np.zeros(3)})
    assert np.array_equal(read_record(tmp_path / 'chain.hess.json')['h'],
                          np.eye(2))
    assert np.array_equal(read_record(tmp_path / 'chain.json')['g'],
                          np.zeros(3))
    assert len(list(tmp_path.glob('chain.*.npz'))) == 2


@pytest.fixture(scope='module')
def h2():
    """H2/6-31G, the smallest closed shell with a virtual space to screen."""
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)


def test_tda_quasiparticle_surface_declares_no_rpa_ground_state(h2):
    """A G0W0-TDA surface carries E_c = 0 and must not pair with dRPA's E_0."""
    qp = QuasiparticleSurface(h2, tight_rhf, charge_change=-1,
                              screening='tda')
    with pytest.raises(ValueError, match='Tamm-Dancoff'):
        qp.physics
    with pytest.raises(ValueError, match='Tamm-Dancoff'):
        compare_surfaces(qp, DenseRPASurface(h2))


def test_dense_surfaces_refuse_a_mean_field_of_another_geometry(h2):
    """A mean field converged at one geometry cannot evaluate another."""
    moved = molecule_at(h2, h2.atom_coords() + np.array([[0, 0, 0],
                                                          [0, 0, 0.05]]))
    mf = tight_rhf(h2)
    rpa = DenseRPASurface(h2)
    qp = QuasiparticleSurface(h2, tight_rhf, charge_change=-1)
    bse = DenseBSESurface(h2)
    for call in (rpa.total_energy, rpa.total_gradient, qp.total_energy,
                 qp.total_gradient, bse.total_energy, bse.total_gradient,
                 bse.excitation):
        with pytest.raises(ValueError, match='another geometry'):
            call(moved, mf)


class Half:
    """One half of a composed surface: the excited half writes a record."""

    def __init__(self, record):
        self.record = record
        self._forward = None

    def to_record(self):
        return self.record

    def identity(self):
        return {'class': type(self).__name__, 'record': self.record}


class Composed:
    """A ground half and an excited half, as `RPABSESurface` holds them."""

    def __init__(self, surface, record):
        self.surface, self.mol0 = surface, surface.mol0
        self.ground, self.excited = Half({'ground': 0}), Half(record)

    def total_gradient(self, mol=None, mf=None):
        return self.surface.total_gradient(mol)

    def identity(self):
        return composed_identity(self, ground=self.ground, excited=self.excited)


def test_force_cache_tells_two_composed_surfaces_apart(h2, tmp_path):
    """A refrozen composed surface must not read its predecessor's forces."""
    surface = MeanFieldSurface(h2, tight_rhf)
    surface_hessian(Composed(surface, {'conventions': 1}), h2,
                    cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match='another Hessian.*conventions'):
        surface_hessian(Composed(surface, {'conventions': 2}), h2,
                        cache_dir=tmp_path)


class Factorization:
    """The frozen factorization a single `FactorChain` surface holds."""

    def __init__(self, record):
        self.record = record

    def to_record(self):
        return self.record


class Single:
    """A single FactorChain surface (`RPAGroundStateChain`): no record of its
    own, no halves, a factorization that writes one."""

    def __init__(self, surface, record):
        self.surface, self.mol0 = surface, surface.mol0
        self.factorization = Factorization(record)

    def total_gradient(self, mol=None, mf=None):
        return self.surface.total_gradient(mol)

    def identity(self):
        return {'class': type(self).__name__,
                'grid': grid_record(self.factorization)}


def test_force_cache_tells_two_single_chain_surfaces_apart(h2, tmp_path):
    """A refrozen single chain must not read its predecessor's forces."""
    surface = MeanFieldSurface(h2, tight_rhf)
    surface_hessian(Single(surface, {'points': 1}), h2, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match='another Hessian.*conventions'):
        surface_hessian(Single(surface, {'points': 2}), h2,
                        cache_dir=tmp_path)


def test_force_cache_refuses_a_chain_that_records_nothing(h2, tmp_path):
    """A FactorChain surface whose factorization writes no record is refused
    a cache instead of stamping ''."""
    surface = MeanFieldSurface(h2, tight_rhf)
    chain = Single(surface, None)
    chain.factorization = object()
    with pytest.raises(ValueError, match='cache_dir=None'):
        surface_hessian(chain, h2, cache_dir=tmp_path)


def test_force_cache_ignores_the_fit_wall_time(h2, tmp_path):
    """Two factorizations that differ only in the seconds their fit took are
    one set of conventions: a rebuilt chain reads its own forces."""
    surface = MeanFieldSurface(h2, tight_rhf)
    surface_hessian(Single(surface, {'points': 1, 'seconds': 1.0}), h2,
                    cache_dir=tmp_path)
    surface_hessian(Single(surface, {'points': 1, 'seconds': 2.0}), h2,
                    cache_dir=tmp_path)
