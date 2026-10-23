"""A chain written to disk and rebuilt from it is the same surface, to a few
ulp (its own SCF and solves: `assert_same` of tests/pins.py).

`ExcitedStateChain.to_record` keeps everything the chain froze -- the
factorization (radii, points, frames, body frame, pair layout), the frozen
conventions with their keys after a committed step, the screening's tau
grid, the reference mean field's ISDF-K pair layout, the root-following
state and the Davidson warm start's committed roots -- and `from_record` rebuilds the chain from it with one SCF at its
reference geometry and nothing derived again. On the production
configuration of the walk gate (formaldehyde/cc-pVDZ, LRC-wPBEh, ISDF-K,
G3 factors, sum-over-poles residues, Davidson, the admitted set with the
calibrated scissor and the outside curve, the state followed by overlap,
SCF converged for differentiation and not warm-started), in the gas phase
and in toluene at split equilibrium, the rebuilt chain's energy and force
at two geometries past a committed step are the original chain's to a few
ulp, and its record hashes as the original's.
"""
import numpy as np
import pytest

from src.Base.utils.record_io import read_record, record_hash, write_record
from src.Base.geometry.geometry import molecule_at
from src.properties.optimize import nudge_vector
from src.gradients.surface import commit
from tests.photophysics_cases import production
from tests.pins import assert_same

#: (environment, relaxed solvation) of the two surfaces.
SETTINGS = {'gas': ('gas', 'nonequilibrium'),
            'toluene-split': ('toluene', 'equilibrium')}
#: Bohr: the largest atomic move of the displaced geometries.
MOVE = 0.02


def displaced(mol, seed):
    """`mol` moved by a seeded internal displacement of MOVE Bohr."""
    return molecule_at(mol, mol.atom_coords()
                       + nudge_vector(mol.atom_coords(), MOVE, seed))


@pytest.mark.parametrize('setting', list(SETTINGS))
def test_a_chain_read_back_is_the_same_surface(setting, tmp_path):
    run, chain = production(tmp_path / 'store', *SETTINGS[setting])
    mol0 = run.mol_in
    chain.total_gradient(mol0)
    # a committed step moves the keys and the followed state
    first = displaced(mol0, 1)
    chain.total_gradient(first)
    commit(chain, first)
    assert chain.conventions.commits == 1
    write_record(tmp_path / 'chain.json', chain.to_record())
    record = read_record(tmp_path / 'chain.json')
    rebuilt = type(chain).from_record(record, mol0, chain.scf_factory,
                                      environment=chain.environment)
    assert record_hash(rebuilt.to_record()) == record_hash(chain.to_record())
    assert rebuilt.conventions.commits == 1
    # The record carries the warm start (its committed roots are in the hash
    # above); the energies are compared on cold Davidsons. A seeded solve
    # starts from the committed roots carried through each chain's own SCF
    # orbitals, which differ in their last bits, and stops elsewhere inside
    # BSE_DAVIDSON_CONV_TOL: in toluene-split 4.3e-11 and 8.6e-11 Ha apart
    # (two cluster runs), above the energy bar; two cold solves
    # start from the same unit vectors.
    chain.davidson_warm_start = rebuilt.davidson_warm_start = None
    for seed in (2, 3):
        at = displaced(first, seed)
        g, e, _ = chain.total_gradient(at)
        g_back, e_back, _ = rebuilt.total_gradient(molecule_at(
            at, at.atom_coords()))
        print(f'{setting} seed {seed}: E {e:.12f}, |g|max '
              f'{np.abs(g).max():.3e}, differences {e_back - e:.1e} Ha, '
              f'{np.abs(g_back - g).max():.1e} Ha/Bohr')
        assert_same(e_back, e, label=(setting, seed))
        assert_same(g_back, g, quantity='gradient', label=(setting, seed))
