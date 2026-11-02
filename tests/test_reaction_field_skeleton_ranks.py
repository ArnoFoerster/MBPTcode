"""The continuum's entry in a folded Fock partial's skeleton
(`mean_field_response.reaction_field_skeleton`) divides its cavity blocks
over the ranks of the enclosing distribution.

Water (bent, non-symmetric), cc-pVDZ, Hartree-Fock in IEF-PCM toluene, a
random symmetric gamma; the cavity cut into blocks of 20 points
(`cavity_blocks`) so there are several:

  1. over two and three simulated ranks the skeleton is the serial one on
     every rank: bit for bit under the recording BLAS, to CROSS_PATH_RTOL
     under any other, where the ranks' own mean fields (each simulated rank
     solves its SCF in its own thread) round differently from the serial
     solve in their last bits;
  2. the ranks together evaluate each block's three-centre integrals once,
     as the serial run does, not once per rank.

Run: python -m pytest tests/test_reaction_field_skeleton_ranks.py
"""
import os
import sys
import threading

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
import src.environments.pcm_derivatives as pcm_derivatives
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.mean_field_response import reaction_field_skeleton
from tests.pins import CROSS_PATH_RTOL, RECORDING_BLAS, blas_identity

ATOMS = 'O 0.0 0.05 0.1173; H 0.02 0.7572 -0.4692; H 0.0 -0.7672 -0.4592'
#: cavity points per block, so that the cavity is several blocks
POINTS = 20


def mean_field():
    mol = gto.M(atom=ATOMS, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method, mf.with_solvent.eps = 'IEF-PCM', 2.3741
    mf.conv_tol, mf.verbose = 1e-12, 0
    return mf.run()


def gamma(nmo):
    g = np.random.default_rng(3).standard_normal((nmo, nmo))
    return g + g.T


@pytest.fixture
def counted(monkeypatch):
    """The cavity in blocks of POINTS; the derivative integral blocks
    (`int3c2e_ip1`, `_ip2`: the skeleton's, not the SCF's) evaluated, by
    thread."""
    blocks, n = pcm_derivatives.cavity_blocks, {}
    lock = threading.Lock()

    def few(nao, ngrids, comp=3, budget_gb=None):
        return [(k0, min(k0 + POINTS, ngrids))
                for k0 in range(0, ngrids, POINTS)]
    aux_e2 = pcm_derivatives.pyscf_df.incore.aux_e2

    def count(*a, **kw):
        if '_ip' in str(kw.get('intor', '')):
            with lock:
                key = threading.get_ident()
                n[key] = n.get(key, 0) + 1
        return aux_e2(*a, **kw)
    monkeypatch.setattr(pcm_derivatives, 'cavity_blocks', few)
    monkeypatch.setattr(pcm_derivatives.pyscf_df.incore, 'aux_e2', count)
    return n


@pytest.mark.parametrize('ranks', [2, 3])
def test_the_ranks_share_the_cavity_blocks(counted, ranks):
    mf = mean_field()
    ngrids = mf.with_solvent.surface['grid_coords'].shape[0]
    assert ngrids > ranks * POINTS, ngrids
    serial = reaction_field_skeleton(mf, gamma(mf.mo_coeff.shape[1]))
    once = sum(counted.values())
    counted.clear()

    def one(comm):
        mf = mean_field()
        return reaction_field_skeleton(mf, gamma(mf.mo_coeff.shape[1]))
    out = run_simulated(one, ranks)
    exact = RECORDING_BLAS == blas_identity()
    bar = 0.0 if exact else CROSS_PATH_RTOL * np.abs(serial).max()
    for g in out:
        assert np.abs(g - serial).max() <= bar, (np.abs(g - serial).max(), bar)
    assert sum(counted.values()) == once, (sum(counted.values()), once)
