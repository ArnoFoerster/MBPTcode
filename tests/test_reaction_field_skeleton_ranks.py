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


@pytest.fixture
def potential_passes(monkeypatch):
    """The surface cut into tiles of POINTS (`pcm_factorization`), none of
    them kept between calls (so that each call evaluates its own), and the
    potential integral tiles (`int3c2e`, the charge solves' own) evaluated
    inside the skeleton, by thread."""
    import src.environments.pcm_factorization as pcm_factorization
    monkeypatch.setattr(pcm_factorization, 'PCM_CAVITY_CACHE_GB', 0.0)

    def few(nao, ngrids, comp=1, budget_gb=None):
        return [(k0, min(k0 + POINTS, ngrids))
                for k0 in range(0, ngrids, POINTS)]
    aux_e2 = pcm_derivatives.pyscf_df.incore.aux_e2
    n, lock, inside = {}, threading.Lock(), threading.local()

    def count(*a, **kw):
        if getattr(inside, 'on', False) and str(kw.get('intor', '')) in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart'):
            with lock:
                key = threading.get_ident()
                n[key] = n.get(key, 0) + 1
        return aux_e2(*a, **kw)

    def skeleton(mf, g):
        inside.on = True
        try:
            return reaction_field_skeleton(mf, g)
        finally:
            inside.on = False
    monkeypatch.setattr(pcm_factorization, 'cavity_blocks', few)
    monkeypatch.setattr(pcm_derivatives.pyscf_df.incore, 'aux_e2', count)
    return n, skeleton


def tiled_mean_field():
    """`mean_field` on a continuum whose surface potential is tiled
    (`factorize_once`), as the distributed SCF leaves it."""
    from src.environments.pcm_factorization import factorize_once
    mol = gto.M(atom=ATOMS, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method, mf.with_solvent.eps = 'IEF-PCM', 2.3741
    factorize_once(mf.with_solvent)
    mf.conv_tol, mf.verbose = 1e-12, 0
    return mf.run()


@pytest.mark.parametrize('ranks', [2, 3])
def test_the_ranks_share_the_charge_solves_surface_potential(potential_passes, ranks):
    """On a tiled continuum the charge solves of the two displaced densities
    divide their surface tiles over the ranks: together the ranks evaluate
    each tile as often as the serial skeleton does, not once per rank. The
    skeleton is the serial one on every rank (bit for bit under the
    recording BLAS), and the charge cache is left as it was on entry."""
    counted, skeleton = potential_passes
    mf = tiled_mean_field()
    ngrids = mf.with_solvent.surface['grid_coords'].shape[0]
    assert ngrids > ranks * POINTS, ngrids
    serial = skeleton(mf, gamma(mf.mo_coeff.shape[1]))
    once = sum(counted.values())
    assert once > 0
    counted.clear()

    def one(comm):
        mf = tiled_mean_field()
        mf.with_solvent._get_vind(mf.make_rdm1())
        before = {k: np.copy(v) for k, v in mf.with_solvent._intermediates.items()
                  if k in ('q', 'q_sym', 'v_grids')}
        g = skeleton(mf, gamma(mf.mo_coeff.shape[1]))
        after = mf.with_solvent._intermediates
        assert all(np.array_equal(before[k], after[k]) for k in before)
        return g
    out = run_simulated(one, ranks)
    exact = RECORDING_BLAS == blas_identity()
    bar = 0.0 if exact else CROSS_PATH_RTOL * np.abs(serial).max()
    for g in out:
        assert np.abs(g - serial).max() <= bar, (np.abs(g - serial).max(), bar)
    assert sum(counted.values()) <= once, (sum(counted.values()), once)
