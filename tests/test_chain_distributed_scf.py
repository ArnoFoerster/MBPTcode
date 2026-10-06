"""The chain's displaced-geometry SCF divided over its ranks.

A walk over geometries rebuilds the mean field at every step, and that stage
is the one a chain does not divide: the sweeps it distributes -- the frequency
loop, the tau partition, the Davidson -- leave the SCF replicated, so every
rank converges the same one and its wall is what it was at a single rank.
`converged_factory`
hands it to `distributed_mean_field` instead: a factory that hands back a mean
field it has BUILT AND NOT RUN leaves the convergence to the chain, every rank
runs pyscf's SCF driver against the reduced J/K and the reduced quadrature,
and each contributes its block of the auxiliary index and of the
exchange-correlation grid.

What that costs in agreement, and why none of it is a tuned tolerance:

  * the reductions re-associate the sums over the auxiliary index and over the
    grid points, so J, K and V_xc differ from the serial ones in the last bits
    and the SCF converges to a mean field that differs AT ITS CONVERGENCE
    TOLERANCE -- not more, because every rank takes every decision in the
    iteration from rank 0's density.
    MEASURED here, water/cc-pVDZ at conv_tol 1e-14, distributed against the
    serial factory's own SCF at the displaced geometry: dE 2.8e-14 Ha and
    max |ddm| 1.1e-12 at two ranks, 5.7e-14 and 8.2e-13 at three, 2.8e-14
    and 6.7e-13 at eight, with the orbital energies 2.2e-13 Ha apart. Two
    SCF runs are compared, so the bars are the convergence they were run to:
    1e-10 Ha for the energy and 1e-8 for the density (`ENERGY_FLOOR`,
    `DM_FLOOR`), where tests/test_distributed_df.py and
    tests/test_mpi_routes.py put the same comparison.
  * the gradient at that geometry is then what the Lagrangian carries of
    the difference between two mean fields each converged to
    SCF_DIFFERENTIABLE_GRAD_TOL, so it is compared at
    CONVERGED_SCF_FORCE_TOL (src/Base/constants.py, which derives it from
    that convergence). Measured: 2.4e-13 Ha/Bohr at two ranks and 3.1e-13
    at three, against a force of 2.21e-02.
  * a build-only factory with NO communicator is `mf.kernel()`, the call the
    factory would have made itself: ONE run of it on the object the factory
    built, with no initial guess and no distributed handles left behind, and
    the chain holding that run's energy, orbitals and orbital energies bit for
    bit. The run is compared with itself rather than with a second factory's
    SCF, which a threaded pyscf would not repeat.

THE GATE WAS SHOWN TO FAIL. Two threads of one process compute bitwise the
same thing from the same inputs, so a rank left to converge alone cannot be
caught here (tests/test_simulated_ranks.py documents that blind spot at
length) -- what this file catches is a distributed SCF that is not the serial
one. With `DistributedDF._reduce_jk` made a no-op (the collective still
called, on a copy) every rank iterates against its own auxiliary rows alone,
the ranks' energies part, and the run raises rather than returning: at two
ranks they stop at different cycles and the next lockstep refuses on every
rank, at three one rank's DIIS meets a singular matrix.

ONE MOLE PER RANK, for the reason tests/test_simulated_ranks.py gives: the
nuclear gradient evaluates the nuclear-attraction derivative inside
`mol.with_rinv_at_nucleus`, an in-place write to the shared `mol._env`, and
two rank THREADS sharing a Mole overwrite each other's rinv origin.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import CONVERGED_SCF_FORCE_TOL
from src.Base.utils.mpi_grid import run_simulated
from src.gradients.rpa_ground_state import RPAGroundStateChain
from tests.test_mpi_routes import recorded

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: The displaced geometry: one hydrogen moved 0.03 A along y, which is where
#: the chain calls its factory. The reference geometry reads `mf0` and runs no
#: SCF at all, so it would gate nothing here.
H2O_DISPLACED = 'O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468'
SIZES = [2, 3]
#: The least of the distributed SCF's bars, as tests/test_mpi_routes.py states
#: them: the energy at the convergence tolerance, the density an order above.
ENERGY_FLOOR = 1e-10
DM_FLOOR = 1e-8


def built(mol):
    """A density-fitted mean field BUILT AND NOT RUN: the chain converges it.

    Tolerances for gradient work (`conv_tol_grad` 1e-11), since the Lagrangian
    assumes the occupied-virtual Fock block vanishes, and they belong to the
    object rather than to whoever calls `kernel`.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    return mf


def converged(mol):
    """The same mean field, converged by the factory itself -- the reference."""
    mf = built(mol)
    mf.kernel()
    return mf


def own_chain(factory):
    """A chain on this rank's OWN Mole, its reference mean field the user's.

    `mf=` is the converged reference in every case: what is being compared is
    the SCF at the DISPLACED geometry, so the two chains must start from one.
    The chain carries no communicator; inside `run_simulated` the context
    reaches `converged_factory` through `distributed_mean_field`.
    """
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return RPAGroundStateChain(mol, factory, mf=converged(mol))


def own_displaced():
    """This rank's own Mole at the displaced geometry."""
    return gto.M(atom=H2O_DISPLACED, basis=BASIS, verbose=0)


def scf_record(chain):
    """What a chain's displaced-geometry mean field ended holding."""
    _, mf = chain.mean_field(own_displaced())
    return dict(e=mf.e_tot, dm=np.asarray(mf.make_rdm1()),
                mo=np.asarray(mf.mo_coeff), eps=np.asarray(mf.mo_energy),
                converged=bool(mf.converged),
                with_df=type(mf.with_df).__name__)


@pytest.mark.parametrize('size', SIZES)
def test_displaced_mean_field_matches_the_serial_chain(size):
    """The distributed SCF is the serial one, and the ranks hold one copy.

    `with_df` is checked back to pyscf's own object: a row slice is not a
    cderi, and everything downstream of the SCF -- the fit, the screening --
    iterates `mf.with_df.loop()` and must meet the whole tensor.
    """
    ref = scf_record(own_chain(converged))
    bar_e, bar_dm = ENERGY_FLOOR, DM_FLOOR
    out = run_simulated(lambda comm: scf_record(own_chain(built)), size)
    d_e = max(abs(rank['e'] - ref['e']) for rank in out)
    d_dm = max(np.abs(rank['dm'] - ref['dm']).max() for rank in out)
    print(f'[info] {size} ranks: dE {d_e:.2e} of {bar_e:.0e} Ha, ddm '
          f'{d_dm:.2e} of {bar_dm:.0e}')
    for rank in out:
        assert rank['converged']
        assert rank['with_df'] == 'DF'
        assert abs(rank['e'] - ref['e']) < bar_e
        assert np.abs(rank['dm'] - ref['dm']).max() < bar_dm
    for rank in out[1:]:
        assert rank['e'] == out[0]['e']
        assert np.array_equal(rank['mo'], out[0]['mo'])
        assert np.array_equal(rank['eps'], out[0]['eps'])


@pytest.mark.parametrize('size', SIZES)
def test_gradient_at_the_displaced_geometry_matches_the_serial_chain(size):
    """The force on the mean field the ranks converged together.

    The energy gate above says the two SCFs agree; this one says the
    Lagrangian carries what is left of the difference no further than two
    converged mean fields may move a force, CONVERGED_SCF_FORCE_TOL.
    """
    ref_g, ref_e, _ = own_chain(converged).total_gradient(own_displaced())
    bar_g, bar_e = CONVERGED_SCF_FORCE_TOL, ENERGY_FLOOR

    def one_rank(comm):
        g, e, _ = own_chain(built).total_gradient(own_displaced())
        return e, g

    out = run_simulated(one_rank, size)
    d_g = max(np.abs(np.asarray(g) - np.asarray(ref_g)).max() for _, g in out)
    print(f'[info] {size} ranks: force |d| {d_g:.2e} = {d_g / bar_g:.4f} '
          f'of CONVERGED_SCF_FORCE_TOL {bar_g:.0e} Ha/Bohr')
    for e, g in out:
        assert abs(e - ref_e) < bar_e
        assert np.abs(np.asarray(g) - np.asarray(ref_g)).max() < bar_g
    assert np.abs(ref_g).max() > 1e-3          # there is a force to compare


def test_without_a_communicator_a_build_only_factory_is_the_factorys_kernel():
    """The serial path is untouched: outside a distributed region
    `distributed_mean_field(mf)` is `mf.kernel()` on the object the factory
    built -- ONE run of it, with no initial guess and no distributed handles
    left behind -- and the chain holds that run's bits, not a tolerance of
    them. The run is compared with itself: a second SCF run is not repeated
    bit for bit by a threaded pyscf."""
    runs = []
    _, mine = own_chain(recorded(built, runs)).mean_field(own_displaced())
    assert len(runs) == 1
    run = runs[0]
    assert run['mf'] is mine and not run['args']
    assert set(run['kwargs']) <= {'dm0'} and run['kwargs'].get('dm0') is None
    assert not hasattr(mine, '_distributed')
    assert not hasattr(mine, '_distributed_timings')
    assert mine.e_tot == run['e_tot']
    assert np.array_equal(mine.mo_coeff, run['mo_coeff'])
    assert np.array_equal(mine.mo_energy, run['mo_energy'])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
