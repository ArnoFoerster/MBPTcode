"""A chain's conventions are frozen at its reference geometry, whatever
geometry it is first evaluated at, and the record says where.

WHAT BREAKS IF THIS FAILS. A first forward pass that froze the explicit set,
the outside scissor, the quadrature and the Newton seeds at whatever geometry
it ran, with the first `at` re-keying the conventions there, would put a walk
on a fresh chain on conventions frozen at the nudged start: `optimize` starts
an excited-state walk `START_NUDGE_BOHR` from the reference (`start_nudge`),
with the window and the quadrature range taken from the reference mean
field, and nothing recorded which. A first evaluation elsewhere is
preceded by the reference's (`ExcitedStateChain._anchored_first`), and
`FrozenConventions.anchored_at` holds the geometry they were frozen at.

H2/cc-pVDZ Hartree-Fock on the dense Casida route.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics import Numerics
from src.gradients.evaluation import geometry_hash
from src.gradients.excited_state import ExcitedStateChain
from src.properties.optimize import optimize
from tests.isdf_test_grid import g1_numerics
from tests.pins import RECORDING_BLAS, matches
from tests.reference_scf import rhf_df_differentiable

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
H2_DISPLACED = 'H 0 0 0; H 0 0 0.77'
#: the conventions a first forward freezes; two chains solve each of them
#: apart, so they agree to a few ulp (`matches` of tests/pins.py), not bit
#: for bit
FROZEN = ('qp_seeds', 'qp_seed_weights', 'sop_poles', 'scissor_map',
          'outside_shift', 'nfreq_cd')


def molecule(atom=H2):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def reference():
    """(mol, mf, a chain evaluated at the reference first)."""
    mol = molecule()
    mf = rhf_df_differentiable(mol)
    chain = fresh(mol, mf)
    chain.total_energy(mol)
    chain.quasiparticle(0, mol)
    return mol, mf, chain


def fresh(mol, mf):
    return ExcitedStateChain(mol, rhf_df_differentiable, mf=mf,
                             numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))


def same(a, b):
    """Whether two frozen values agree: keys and shapes exactly, numbers to
    a few ulp, since each chain solved its roots for itself."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind not in 'fc' and b.dtype.kind not in 'fc':
        return np.array_equal(a, b)
    if a.dtype.kind == 'c' or b.dtype.kind == 'c':
        return (matches(a.real, b.real, RECORDING_BLAS, quantity='root')
                and matches(np.imag(a), np.imag(b), RECORDING_BLAS,
                            quantity='root'))
    return matches(a, b, RECORDING_BLAS, quantity='root')


def test_a_walk_on_a_fresh_chain_freezes_at_the_reference(reference):
    mol, mf, ref = reference
    walked = fresh(mol, mf)
    optimize(walked, max_cycle=1, refreeze=0)
    for name in FROZEN:
        got, want = (getattr(c.conventions, name) for c in (walked, ref))
        if name == 'qp_seeds':
            # the reference chain also solved the HOMO alone
            want = {p: w for p, w in want.items() if p in got}
        assert same(got, want), name
    assert walked.conventions.anchored_at == geometry_hash(mol)


def test_a_first_quasiparticle_elsewhere_freezes_at_the_reference(reference):
    mol, mf, ref = reference
    chain = fresh(mol, mf)
    chain.quasiparticle(0, molecule(H2_DISPLACED))
    homo = chain.nocc - 1
    assert same(chain.conventions.qp_seeds[homo], ref.conventions.qp_seeds[homo])
    assert chain.conventions.nfreq_cd == ref.conventions.nfreq_cd
    assert chain.conventions.anchored_at == geometry_hash(mol)


def test_matching_before_anchoring_leaves_the_keys_at_the_reference(
        reference):
    mol, mf, _ = reference
    chain = fresh(mol, mf)
    away = molecule(H2_DISPLACED)
    chain.conventions.at(away, rhf_df_differentiable(away))
    assert chain.conventions.key_mol is chain.mol0
    assert np.array_equal(chain.conventions.key_mo, mf.mo_coeff)
    assert not chain.conventions.anchored


def test_the_record_carries_the_anchor(reference):
    mol, mf, ref = reference
    record = ref.to_record()
    assert record['conventions']['anchored_at'] == geometry_hash(mol)
