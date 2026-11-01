"""The orbital-rotation gradient builds the exchange it reads, K C_occ, only.

`multipliers.rotation_gradient` reads the response potential G of a rotation
density only through its occupied columns, (C^T G C)[:, :nocc]. A multiplier
with virtual-virtual pairs makes a density of full rank, so the exchange of a
general density, 2 naux nao^3 per build, was the cost of every Z-vector matvec.
The density now carries the occupied orbitals (`read_columns`), and a
distributed JK fit builds K C_occ from its rows at 3 naux nao^2 nocc and hands
back (K C_occ) C_occ^+ in place of K (`DistributedDF._exchange_on_columns`).

Gated here, under 2 and 3 simulated ranks on a density-fitted LRC-wPBEh
water, so that both the bare rows and the long-range rows take the route:
  * the tagged exchange read on the occupied columns is the whole K's, for
    the bare and the attenuated operator;
  * the rotation gradient is the one formed from the whole kernel, to the
    reduction's last bits, and the same bits on every rank;
  * serially nothing reads the tag: pyscf's kernel returns the whole G, the
    same bits as for the untagged density.

What breaks each gate: dropping the transpose of L_Q (an asymmetric
contraction) or the pseudo-inverse moves the columns by order one.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import lib

from tests.pins import assert_within_class
from src.Base.factorization.distributed_df import (distributed_fock,
                                                   distributed_mean_field)
from src.Base.pyscf.pyscf_interface import fock_mo, response_kernel
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.multipliers import rotation_gradient
from tests.test_distributed_df import SIZES, WATER, mean_field

XC = 'lrc-wpbeh'
BASIS = 'cc-pvdz'
#: The attenuation of LRC-wPBEh, the long-range rows the response asks for.
OMEGA = 0.3


def symmetric_partial(norb, seed=11):
    """A symmetric MO partial with every block filled, virtual-virtual too."""
    g = np.random.default_rng(seed).standard_normal((norb, norb))
    return g + g.T


def whole_kernel_gradient(mf, gamma, nocc, kernel):
    """`rotation_gradient` with the kernel applied to the untagged density."""
    C = mf.mo_coeff
    gs = 0.5 * (gamma + gamma.T)
    G = kernel(C @ gs @ C.T)
    Y = 2.0 * (fock_mo(mf) @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ G @ C)[:, :nocc]
    return Y


def on_ranks(fn, size):
    """`fn(comm, mf)` on every rank, each with the mean field the ranks
    converged together."""
    mfs = [mean_field(WATER, BASIS, XC) for _ in range(size)]

    def one_rank(comm):
        mf = mfs[comm.Get_rank()]
        distributed_mean_field(mf, comm)
        return fn(comm, mf)

    return run_simulated(one_rank, size)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('omega', [None, OMEGA])
def test_tagged_exchange_is_k_on_the_columns(size, omega):
    def both(comm, mf):
        nocc = mf.mol.nelectron // 2
        Co = mf.mo_coeff[:, :nocc]
        x = mf.mo_coeff @ symmetric_partial(mf.mo_coeff.shape[1]) \
            @ mf.mo_coeff.T
        with distributed_fock(mf, build=False):
            vk = mf.get_k(mf.mol, x, hermi=1, omega=omega)
            vk_tag = mf.get_k(mf.mol, lib.tag_array(x, read_columns=Co),
                              hermi=1, omega=omega)
        return vk @ Co, vk_tag @ Co

    out = on_ranks(both, size)
    for whole, columns in out:
        assert_within_class(columns, whole, 'root', label='K C_occ')
        assert np.array_equal(columns, out[0][1])


@pytest.mark.parametrize('size', SIZES)
def test_rotation_gradient_is_the_whole_kernels(size):
    def both(comm, mf):
        nocc = mf.mol.nelectron // 2
        gamma = symmetric_partial(mf.mo_coeff.shape[1])
        kernel = response_kernel(mf, environment=False)
        return (rotation_gradient(mf.mo_coeff, fock_mo(mf), gamma, nocc,
                                  kernel),
                whole_kernel_gradient(mf, gamma, nocc, kernel))

    out = on_ranks(both, size)
    for columns, whole in out:
        assert_within_class(columns, whole, 'root', label='Y')
        assert np.array_equal(columns, out[0][0])


def test_serial_kernel_ignores_the_tag():
    mf = mean_field(WATER, BASIS, XC)
    mf.kernel()
    nocc = mf.mol.nelectron // 2
    gamma = symmetric_partial(mf.mo_coeff.shape[1])
    kernel = response_kernel(mf, environment=False)
    got = rotation_gradient(mf.mo_coeff, fock_mo(mf), gamma, nocc, kernel)
    assert np.array_equal(got, whole_kernel_gradient(mf, gamma, nocc, kernel))
