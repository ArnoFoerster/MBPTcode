"""One SCF and one forward pass per geometry, and a force on an SCF that
stopped short warned about and flagged.

WHAT BREAKS IF THIS FAILS. If every accessor of a chain ran its own SCF and
its own forward pass (`relax_state` converging a fresh SCF at the minimum and
running a fresh forward there, the ground state's record a third SCF), the
energy a walk reports (`info['energy']`) and the one its record carries
(`e_total`) would be two numbers from up to three SCFs at one geometry, since
an SCF is free to land on another solution. The chain holds what it computed
at the last EVALUATION_CACHE_SIZE geometries (`GeometryCache`, keyed by the
nuclei): the mean field, the roots and, once asked, the force. The SCF counts
below are one and `e_total` is the walk's energy to the bit.

`check_scf_quality` runs on every mean field a force reads: a displaced SCF
that stopped short of its convergence would put |F_ia| into the force
unannounced. A force on a mean field whose |F_ia| exceeds what it was
converged to (`scf_force_tol`) warns, is flagged in `scf_flags` and carries
on, as an energy does (refuse inputs, warn
numerics).

Formaldehyde S1, HF/cc-pVDZ, the chain's defaults.
"""
import collections

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import SCF_DIFFERENTIABLE_GRAD_TOL, SCF_TIGHT_CONV_TOL
from src.Base.numerics.numerics import Numerics
from src.gradients.evaluation import GeometryCache, geometry_key
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.vibronic import relax_state
from tests.isdf_test_grid import g1_numerics

CH2O = ('C 0 0 -0.529; O 0 0 0.6746; H 0 0.9376 -1.1188; '
        'H 0 -0.9376 -1.1188')
#: Bohr: the displacement the single-geometry checks are taken at.
STEP = 0.02


class CountingFactory:
    """HF converged for gradient work, counting the SCFs run per geometry;
    `short` geometries stop after two cycles."""

    def __init__(self):
        self.runs = collections.Counter()
        self.short = set()

    def __call__(self, mol):
        key = geometry_key(mol)
        self.runs[key] += 1
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
        mf.conv_tol = SCF_TIGHT_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        if key in self.short:
            mf.max_cycle = 2
        mf.kernel()
        return mf


@pytest.fixture(scope='module')
def ch2o():
    return gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)


def displaced(mol, step=STEP):
    out = mol.copy()
    coords = mol.atom_coords()
    coords[1, 2] += step
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def test_the_cache_holds_the_last_geometries(ch2o):
    cache = GeometryCache(size=2)
    a, b, c = ch2o, displaced(ch2o), displaced(ch2o, 2 * STEP)
    cache.put(a, 'a')
    cache.put(b, 'b')
    assert cache.get(a.copy()) == 'a', 'the key is the nuclei, not the object'
    cache.put(c, 'c')
    assert cache.get(b) is None and cache.get(a) == 'a'
    assert cache.get(c) == 'c' and len(cache) == 2


def test_an_energy_and_a_force_at_one_geometry_are_one_evaluation(ch2o):
    factory = CountingFactory()
    chain = ExcitedStateChain(ch2o, factory, spin='singlet', numerics=g1_numerics(ch2o))
    forwards = []
    shared = chain.shared_forward

    def counted(mol, mf):
        forwards.append(geometry_key(mol))
        return shared(mol, mf)

    chain.shared_forward = counted
    at = displaced(ch2o)
    energy = chain.total_energy(at)
    grad, total, _ = chain.total_gradient(displaced(ch2o))
    assert total == energy
    assert factory.runs[geometry_key(at)] == 1
    assert forwards.count(geometry_key(at)) == 1
    again, _, _ = chain.total_gradient(at)
    assert again is grad, 'the force is taken once per geometry'


def test_a_relaxed_state_reads_the_walks_own_evaluation(ch2o):
    factory = CountingFactory()
    chain = ExcitedStateChain(
        ch2o, factory, spin='singlet',
        numerics=g1_numerics(ch2o, Numerics(bse={'track': 'overlap'})))
    record = relax_state(chain, ch2o, engine='cartesian', max_cycle=3,
                         nudge=0, verbose=False)
    minimum = geometry_key(record['mol'])
    assert record['e_total'] == record['info']['energy']
    assert factory.runs[minimum] <= 1, 'more than one SCF at the minimum'
    assert record['mf'] is chain.mean_field(record['mol'])[1]


def test_a_force_on_an_scf_that_stopped_short_is_flagged(ch2o):
    factory = CountingFactory()
    chain = ExcitedStateChain(ch2o, factory, spin='singlet', numerics=g1_numerics(ch2o))
    at = displaced(ch2o)
    factory.short.add(geometry_key(at))
    with pytest.warns(RuntimeWarning, match='F_ia'):
        energy = chain.total_energy(at)
    assert np.isfinite(energy)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        grad, _, _ = chain.total_gradient(at)
    assert np.all(np.isfinite(grad))
    assert {f['where'] for f in chain.scf_flags} >= {
        'the SCF at this geometry stopped short', 'total_gradient'}
