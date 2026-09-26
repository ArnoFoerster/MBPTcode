"""The two k-mesh index conventions, pinned against EACH OTHER.

The polarizability's k-sum (pbc_isdf_rpa) is a CORRELATION,

    Pi^q = sum_k Go^k * Gv^{k+q}        -> FFT with an index negation,

and the self-energy's q-sum (pbc_isdf_gw) is a CONVOLUTION,

    Sigma^k = sum_q G^{k-q} * Wt^q      -> plain product of forward FFTs.

Each is already pinned against its own direct sum, in its own file. That is
NOT sufficient, and this file exists because it was found not to be:

**every other ISDF test runs on a mesh that cannot tell the two apart.** The
meshes in use are [1,1,1], [2,1,1], [2,2,1] and [2,2,2] -- all subsets of
{1,2}^3, where every k-point is its own inverse, so k+q == k-q identically.
There `kplus` and `kminus` are the same array, the `neg` reindex in
`polarizability_tau_all_q_fft` is the identity permutation, and (proved below
by measurement, not assertion) the correlation and the convolution return the
SAME number. Swapping the two routines wholesale would have passed the entire
suite.

So the payload arrays here are random -- both identities are pure index
algebra over the mesh and hold for any arrays -- while the index maps are the
real ones built from a real cell, which is the part that can actually be
wrong. The physics lives in test_pbc_isdf_rpa.py and test_pbc_isdf_gw.py.
"""
import os
import sys

import numpy as np
import pytest
from pyscf.pbc import gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
from src.SingleReference.Periodic.pbc_isdf import kpoint_minus_map
from src.SingleReference.Periodic.pbc_isdf_gw import (self_energy_tau_convolution,
                                                      self_energy_tau_direct)
from src.SingleReference.Periodic.pbc_isdf_rpa import (polarizability_tau,
                                                       polarizability_tau_all_q_fft)

# Meshes on which k+q != k-q, i.e. where the two conventions are distinguishable.
CHIRAL_MESHES = [[3, 1, 1], [2, 3, 1], [3, 3, 1], [4, 1, 1]]
# Meshes used by every other ISDF test -- every k-point is its own inverse.
BLIND_MESHES = [[1, 1, 1], [2, 1, 1], [2, 2, 1], [2, 2, 2]]

NPTS = 6


def _cell():
    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.diag([3.0, 3.0, 2.8])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.build()
    return cell


def _maps(kmesh):
    """(kplus, kminus) from a real cell -- no SCF, the maps are pure geometry."""
    cell = _cell()
    kpts = cell.make_kpts(kmesh)
    return (get_momentum_transfer_map(cell, kpts), kpoint_minus_map(cell, kpts))


def _payload(nk, seed):
    rng = np.random.default_rng(seed)
    shape = (nk, NPTS, NPTS)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    b = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return a, b


@pytest.mark.parametrize('kmesh', BLIND_MESHES)
def test_blind_meshes_cannot_distinguish_the_conventions(kmesh):
    """Why this file exists: on {1,2}^3 meshes the two conventions COINCIDE.

    Every k is its own inverse there, so kplus == kminus elementwise and the
    correlation and the convolution of the same pair of arrays are equal. Any
    test of either routine on such a mesh is blind to a swap. Pinned so that
    the blindness is a recorded property rather than a surprise.
    """
    kplus, kminus = _maps(kmesh)
    nk = kplus.shape[0]
    assert np.array_equal(kplus, kminus), kmesh

    a, b = _payload(nk, seed=0)
    corr = polarizability_tau_all_q_fft(a, b, kmesh)
    conv = self_energy_tau_convolution(a, b, kmesh)
    assert np.allclose(corr, conv, atol=1e-12), (
        f"{kmesh}: expected the conventions to coincide on a self-inverse mesh")


@pytest.mark.parametrize('kmesh', CHIRAL_MESHES)
def test_maps_differ_where_it_matters(kmesh):
    """On these meshes k+q != k-q, so the rest of the file is not a tautology."""
    kplus, kminus = _maps(kmesh)
    assert not np.array_equal(kplus, kminus), kmesh
    # kminus is still the row-wise inverse permutation of kplus.
    nk = kplus.shape[0]
    for q in range(nk):
        assert np.array_equal(kminus[q, kplus[q]], np.arange(nk))


@pytest.mark.parametrize('kmesh', CHIRAL_MESHES)
def test_polarizability_fft_is_the_correlation(kmesh):
    """(v) sum_k Go^k * Gv^{k+q}: FFT == direct k-sum, at machine precision."""
    kplus, _ = _maps(kmesh)
    nk = kplus.shape[0]
    Go, Gv = _payload(nk, seed=1)

    fft = polarizability_tau_all_q_fft(Go, Gv, kmesh)
    direct = np.asarray([polarizability_tau(Go, Gv, kplus, q) for q in range(nk)])
    dev = np.abs(fft - direct).max() / np.abs(direct).max()
    assert dev < 1e-13, (kmesh, dev)


@pytest.mark.parametrize('kmesh', CHIRAL_MESHES)
def test_self_energy_fft_is_the_convolution(kmesh):
    """(ix) sum_q G^{k-q} * Wt^q: FFT == direct q-sum, at machine precision."""
    _, kminus = _maps(kmesh)
    nk = kminus.shape[0]
    G, Wt = _payload(nk, seed=2)

    fft = self_energy_tau_convolution(G, Wt, kmesh)
    direct = np.asarray([self_energy_tau_direct(G, Wt, kminus, k) for k in range(nk)])
    dev = np.abs(fft - direct).max() / np.abs(direct).max()
    assert dev < 1e-13, (kmesh, dev)


@pytest.mark.parametrize('kmesh', CHIRAL_MESHES)
def test_a_swapped_convention_is_detectable(kmesh):
    """THE CROSS-PIN: the two routines must return DIFFERENT numbers here.

    This is the assertion the suite was missing. Without it, both routines
    could compute the same thing -- or one could be substituted for the other
    -- and every other test would still pass, because they only ever run where
    the conventions coincide (see test_blind_meshes_...).

    The bar is a large relative separation, not merely non-equality: the point
    is that a swap changes the answer by O(1), so it cannot hide inside a
    tolerance.
    """
    kplus, kminus = _maps(kmesh)
    nk = kplus.shape[0]
    a, b = _payload(nk, seed=3)

    corr = polarizability_tau_all_q_fft(a, b, kmesh)   # sum_k a_k b_{k+q}
    conv = self_energy_tau_convolution(a, b, kmesh)    # sum_q a_{k-q} b_q
    sep = np.abs(corr - conv).max() / np.abs(corr).max()
    assert sep > 0.1, (kmesh, sep)

    # and each still equals its OWN direct sum, so the separation is the
    # convention and not a bug in one of them.
    d_corr = np.asarray([polarizability_tau(a, b, kplus, q) for q in range(nk)])
    d_conv = np.asarray([self_energy_tau_direct(a, b, kminus, k) for k in range(nk)])
    assert np.abs(corr - d_corr).max() / np.abs(d_corr).max() < 1e-13
    assert np.abs(conv - d_conv).max() / np.abs(d_conv).max() < 1e-13


if __name__ == '__main__':
    print('test_pbc_isdf_kindex:')
    for m in BLIND_MESHES:
        test_blind_meshes_cannot_distinguish_the_conventions(m)
    print(f'  blind meshes {BLIND_MESHES}: conventions coincide, as expected')
    for m in CHIRAL_MESHES:
        test_maps_differ_where_it_matters(m)
        test_polarizability_fft_is_the_correlation(m)
        test_self_energy_fft_is_the_convolution(m)
        test_a_swapped_convention_is_detectable(m)
    print(f'  chiral meshes {CHIRAL_MESHES}: both FFTs exact, and separable')
    print('ALL PASSED')
