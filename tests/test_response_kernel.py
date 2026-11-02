"""The orbital-response kernel G(x) = J[x] - K[x]/2 of a density-fitted mean
field (src.Base.pyscf.cphf.ResponseKernel) and the Z-vector it serves over the
ranks. Water cc-pVDZ, cc-pvdz-jkfit, a converged DF-RHF; ADC(2) S1 tiled
force on water 6-31G (frozen 1, cc-pvdz-ri / -jkfit).

  1. a general symmetric x: pyscf's get_jk of the same fit to 1e-12.
  2. factored(P), x = Co P^T + P Co^T: pyscf's get_jk of that x to 1e-12.
  3. 2 and 3 simulated ranks: every rank returns the serial G, bit for bit,
     for both, and builds only its own auxiliary blocks (round-robin).
  4. rotation_gradient hands a gamma without a virtual-virtual block to
     factored, one with such a block to the general build.
  5. the tiled force's Z-vector is solved by every rank (each building its
     share of the Fock matrices), not by rank 0 alone, over 2 simulated ranks.

The same checks under real MPI: tests/test_response_kernel_mpi.py.
Run: python -m pytest -q tests/test_response_kernel.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.cphf import ResponseKernel
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force as dcf
from src.gradients.multipliers import rotation_gradient

ATOM = 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467'
JK = 'cc-pvdz-jkfit'
# auxiliary functions per kernel block: the water fit in several blocks
BLOCK = 24


def mean_field():
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=JK)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


MOL, MF = mean_field()
NOCC = MOL.nelectron // 2
RNG = np.random.default_rng(7)
X = RNG.standard_normal((MOL.nao, MOL.nao))
X = X + X.T
P = RNG.standard_normal((MOL.nao, NOCC))
CO = MF.mo_coeff[:, :NOCC]


def pyscf_g(x):
    m = scf.RHF(MOL).density_fit(auxbasis=JK)
    m.with_df = MF.with_df
    vj, vk = m.get_jk(MOL, x, hermi=1)
    return vj - 0.5 * vk


def test_general_density():
    d = np.max(np.abs(ResponseKernel(MF)(X) - pyscf_g(X)))
    assert d < 1e-12, d


def test_factored_density():
    d = np.max(np.abs(ResponseKernel(MF).factored(P) - pyscf_g(CO @ P.T + P @ CO.T)))
    assert d < 1e-12, d


def test_ranks_bitwise_and_own_blocks():
    serial = ResponseKernel(MF, block=BLOCK)
    want = (serial(X), serial.factored(P))
    nblocks = serial._nblocks()
    assert nblocks >= 3, 'the fit needs blocks for every rank'

    def run(comm):
        k = ResponseKernel(MF, comm, block=BLOCK)
        return k(X), k.factored(P), list(k.mine)

    for size in (2, 3):
        outs = run_simulated(run, size)
        for r, (g, gf, mine) in enumerate(outs):
            assert np.array_equal(g, want[0]) and np.array_equal(gf, want[1]), (size, r)
            assert mine == list(range(r, nblocks, size)), (size, r, mine)


class Recording:
    """A kernel that records which build each call took."""

    def __init__(self):
        self.inner, self.calls = ResponseKernel(MF), []

    def __call__(self, x):
        self.calls.append('general')
        return self.inner(x)

    def factored(self, p):
        self.calls.append('factored')
        return self.inner.factored(p)


def test_rotation_gradient_routes():
    C = MF.mo_coeff
    nmo = C.shape[1]
    fock = np.diag(MF.mo_energy)
    lam = np.zeros((nmo, nmo))
    lam[NOCC:, :NOCC] = RNG.standard_normal((nmo - NOCC, NOCC))
    lam = lam + lam.T
    rec = Recording()
    Y = rotation_gradient(C, fock, lam, NOCC, rec)
    Y_ref = rotation_gradient(C, fock, lam, NOCC, pyscf_g)
    gam = lam.copy()
    gam[NOCC:, NOCC:] = 1e-3
    rotation_gradient(C, fock, gam, NOCC, rec)
    assert rec.calls == ['factored', 'general'], rec.calls
    assert np.max(np.abs(Y - Y_ref)) < 1e-11


def test_force_zvector_on_every_rank():
    from tests.adc_common import ATOMS, AUX, NF, molecule
    from src.gradients.adc import ee_gradient as eg
    mol = molecule(ATOMS)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    fn = dcf.relax_gradient
    seen = []

    def counted(*a, **k):
        from src.Base.utils.parallel.mpi_grid import current_comm
        c = current_comm()
        seen.append(0 if c is None else c.Get_rank())
        return fn(*a, **k)

    dcf.relax_gradient = counted
    try:
        run_simulated(lambda comm: eg.adc2_force_tiled(
            mf, mol, 'singlet', frozen=NF, auxbasis=AUX, comm=comm, tile_gb=1e-9,
            numerics=Numerics())[0], 2)
    finally:
        dcf.relax_gradient = fn
    assert sorted(seen) == [0, 1], seen
