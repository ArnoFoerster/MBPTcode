"""The adaptive set's partition carried between the stages of one run.

A driver may store the partition its reference evaluation selected (the
JSON of `selection_record()['partition']`) and hand it to a later stage as the
numeric `qp_partition` of the same declaration rather than selecting again.
Water/cc-pVDZ (a DF Hartree-Fock mean field; G1, SOP, Davidson, grid
adjoint): the partition read back through JSON is the selected one, the
surface on it repeats the selected surface's S1 and T1 and S1 force bit for
bit, and no continuation runs.
"""
import json
import warnings

import numpy as np
from pyscf import gto, scf

import src.gradients.excited_state as excited_state
from src.Base.declaration import Excitation, GroundState, QPStates
from src.SingleReference.GW.qp_selection import AdaptivePartition
from src.gradients.state_manifold import StateManifold
from src.properties.excitations import SurfaceSpec, surface_of

H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
KEYS = (('singlet', 0), ('triplet', 0))


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def spec(partition=None):
    numerics = {'grid_accuracy': 'G1', 'bse_adjoint': 'grid', 'nroots': 4}
    if partition is not None:
        numerics['qp_partition'] = partition
    return SurfaceSpec(GroundState('dft', 'hf'), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates('adaptive', targets=KEYS),
                       numerics=numerics)


def evaluated(mol, mf, partition=None):
    surface = surface_of(spec(partition), Excitation('singlet'), mol,
                         factory, mf=mf)
    man = StateManifold(surface, states=KEYS)
    return man, man.evaluate(mol, mf, gradients=(KEYS[0],))


def test_partition_round_trips_through_the_vert_checkpoint(monkeypatch):
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mf = factory(mol)
    man, ev = evaluated(mol, mf)
    part = man.driven.qp_partition
    assert part.holes, 'the gate needs holes'
    stored = json.loads(json.dumps(
        man.driven.selection_record()['partition']))
    back = AdaptivePartition.from_record(stored)
    assert back == part and back.tier_of == part.tier_of

    def refused(*args, **kwargs):
        raise AssertionError('a carried partition selected again')
    monkeypatch.setattr(excited_state, 'ac_quasiparticle_shifts', refused)
    man2, ev2 = evaluated(mol, mf, back)
    assert man2.driven.qp_partition == part
    for k in KEYS:
        assert ev2.omega[k] == ev.omega[k]
    assert np.array_equal(ev2.gradient[KEYS[0]], ev.gradient[KEYS[0]])
