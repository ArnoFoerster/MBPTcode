"""One SCF per vertical record: the ground surface is declared on the excited
chain's reference mean field and E_0 at R0 is read off it.

On a 'dft' ground state the excited chain's `mf0` serves as the excited
chain's reference, the ground surface's reference (read only for its
declaration) and the mean field of E_0 at R0 (`ground_state_at` ->
`MeanFieldSurface`).

Gated on water/cc-pVDZ, PBE0, the production flag set (`--scf isdf`,
space-time, SOP on the frontier states, sliced row-fit factors, grid BSE
adjoint, G1), serially and at 2 and 3 simulated ranks:
- the factory runs once per record;
- every record field equals the one assembled on a separately converged
  ground surface (pyscf's SCF from the same factory is deterministic within
  one process, so `e0_hartree` and `e0_terms` are equal too).
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from pyscf import gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.utils.mpi_grid import distributed, run_simulated
from src.properties.excitations import (SurfaceSpec, calc_vertical_excitation,
                                        surface_of, vertical_block)
from tests.test_state_pair_force_distributed_ks import (BASIS, GRID_ACCURACY,
                                                        MAX_MEMORY, WATER,
                                                        scf_factory)

XC = 'pbe0'
SIZES = (2, 3)
TIMELESS = ('provenance', 'mol_reference', 'physics')
_LOCK = threading.Lock()


def counted(calls):
    """The ISDF-K factory of the production flag set, each call counted per
    thread."""
    build = scf_factory(XC)

    def factory(mol):
        with _LOCK:
            key = threading.get_ident()
            calls[key] = calls.get(key, 0) + 1
        return build(mol)
    return factory


def spec_and_state():
    spec = SurfaceSpec(GroundState('dft', XC), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics={'grid_accuracy': GRID_ACCURACY, 'sliced': True,
                                 'fit': 'rows', 'bse_adjoint': 'grid'})
    return spec, Excitation('singlet', root=1, kernel='bse')


def molecule():
    warnings.simplefilter('ignore')
    return gto.M(atom=WATER, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)


def counted_record(calls):
    """(the vertical record, the SCFs it converged on this thread)."""
    spec, state = spec_and_state()
    rec = calc_vertical_excitation(spec, state, molecule(), counted(calls))
    return rec, calls[threading.get_ident()]


def comparable(rec):
    """The record's fields that do not name a clock or an object."""
    return {k: v for k, v in rec.items() if k not in TIMELESS}


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if key == 'realization':
            assert repr(a[key]) == repr(b[key]), key
        else:
            assert a[key] == b[key], key


def test_the_serial_record_converges_one_scf():
    """One factory call, and the record is the one on a second SCF's E_0."""
    calls = {}
    with distributed(None):
        rec, count = counted_record(calls)
        spec, state = spec_and_state()
        mol = molecule()
        factory = counted({})
        excited = surface_of(spec, state, mol, factory)
        separate = vertical_block(excited, surface_of(spec, None, mol,
                                                      factory), mol)
    assert count == 1, count
    assert_same(comparable(rec), comparable(separate))
    assert np.isfinite(rec['e0_hartree'])


@pytest.mark.parametrize('size', SIZES)
def test_every_rank_converges_one_scf(size):
    """Under simulated ranks: one distributed SCF per rank per record, and
    every rank's record rank 0's."""
    calls = {}
    out = run_simulated(lambda comm: counted_record(calls), size)
    assert [count for _, count in out] == [1] * size
    for rec, _ in out[1:]:
        assert_same(comparable(rec), comparable(out[0][0]))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
