"""The two-electron spin-orbit term (SOMF / AMFI) is built once over the ranks:
rank 0 runs the integral pass and every rank receives its operator.

Run through `run_simulated` (ranks are threads; mpirun cannot start in the
sandboxed test runner; tests/test_spin_orbit_mpi.py is the real-MPI script).
Gated:
  * serially, the kernel's term is J - (3/2) K of the integral pass, bit for
    bit, and `soc_operator` is the one-electron operator plus that term times
    alpha^2/4, bit for bit: the move into the kernel changes no number;
  * at 2, 3 and 8 ranks, the integral pass runs on rank 0 and on no other rank,
    for SOMF and for AMFI (counted per call);
  * every rank returns rank 0's operator BITWISE, and that operator is the
    serial one, even when the other ranks hold a different density;
  * the real-MPI script's checks pass over 2 and 3 thread-ranks.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.parallel.mpi_grid import current_comm
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import spin_orbit_mean_field
from src.gradients.spin_orbit_mean_field import mean_field_term
from src.properties.spin_orbit import (SOC_PREFACTOR, soc_operator,
                                       x2c_one_electron)
from tests.test_spin_orbit_mpi import main

H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf.make_rdm1()


@pytest.mark.parametrize('two_electron', ['somf', 'amfi'])
def test_the_serial_term_is_the_integral_pass(water, two_electron):
    mol, dm = water
    jk = (spin_orbit_mean_field._soc_jk_amfi if two_electron == 'amfi'
          else spin_orbit_mean_field._soc_jk)
    vj, vk = jk(mol, dm)
    term = mean_field_term(mol, dm, two_electron)
    assert np.array_equal(term, vj - 1.5 * vk)
    assert np.array_equal(soc_operator(mol, dm, two_electron),
                          x2c_one_electron(mol) + term * SOC_PREFACTOR)


def test_an_unknown_term_is_refused(water):
    mol, dm = water
    with pytest.raises(ValueError, match="'somf', 'amfi'"):
        mean_field_term(mol, dm, 'koseki')


@pytest.mark.parametrize('two_electron', ['somf', 'amfi'])
@pytest.mark.parametrize('size', [2, 3, 8])
def test_only_rank_zero_builds_the_two_electron_term(water, monkeypatch,
                                                    size, two_electron):
    mol, dm = water
    serial = soc_operator(mol, dm, two_electron)
    builders = []
    real = spin_orbit_mean_field._soc_jk

    def counted(m, d):
        builders.append(current_comm().Get_rank())
        return real(m, d)
    monkeypatch.setattr(spin_orbit_mean_field, '_soc_jk', counted)

    def one_rank(comm):
        # every rank holds its own, slightly different density
        mine = dm + 1e-9 * comm.Get_rank()
        return soc_operator(mol, mine, two_electron)
    out = run_simulated(one_rank, size)

    # the builds are counted on every rank-thread: one in all (SOMF) or one
    # per atom (AMFI), every one of them on rank 0
    assert set(builders) == {0}
    assert len(builders) == (mol.natm if two_electron == 'amfi' else 1)
    for r, h in enumerate(out):
        assert np.array_equal(h, out[0]), f'rank {r} differs from rank 0'
    assert np.array_equal(out[0], serial)


@pytest.mark.parametrize('size', [2, 3])
def test_the_mpi_script_passes_over_thread_ranks(monkeypatch, size):
    # the script installs its counting pass; put the real one back afterwards
    monkeypatch.setattr(spin_orbit_mean_field, '_soc_jk',
                        spin_orbit_mean_field._soc_jk)
    assert run_simulated(main, size) == [0] * size
