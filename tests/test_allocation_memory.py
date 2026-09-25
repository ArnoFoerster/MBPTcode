"""`Base.utils.memory`: what a SLURM allocation holds, and what pyscf did with it.

`allocation_max_memory_mb` is the fix for the pentacene cc-pVTZ regression:
pyscf's `max_memory` defaults to 4000 MB regardless of what the job was given,
and on a 16-core node that default kept pentacene's 6.4 GB DF tensor out of core
while anthracene's 1.8 GB one, at the same default, stayed in. These tests
gate the environment parsing in isolation (monkeypatched, so they say nothing
about an actual SLURM allocation) and `describe_df_storage` against a real DF
mean field.

Run as a script, this file hands itself to pytest and exits with its verdict.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import gto, scf

from src.Base.constants import ALLOCATION_MEMORY_FRACTION
from src.Base.utils.memory import allocation_max_memory_mb, describe_df_storage

#: SLURM_* variables the parser reads; cleared before every case so a test
#: run under an actual allocation does not leak into these, and cleared after
#: it so the values a case sets do not leak into the rest of a pytest session.
SLURM_VARS = ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK')


@pytest.fixture(autouse=True)
def _clean_slurm_env(monkeypatch):
    for var in SLURM_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in SLURM_VARS:
        os.environ.pop(var, None)


def test_per_node_wins_over_per_cpu():
    """SLURM_MEM_PER_NODE is read first, even with the per-cpu pair also set."""
    os.environ['SLURM_MEM_PER_NODE'] = '128000'
    os.environ['SLURM_MEM_PER_CPU'] = '99999'
    os.environ['SLURM_CPUS_PER_TASK'] = '99'
    assert allocation_max_memory_mb(fraction=0.6) == int(128000 * 0.6)


def test_per_cpu_times_cpus_per_task():
    os.environ['SLURM_MEM_PER_CPU'] = '4000'
    os.environ['SLURM_CPUS_PER_TASK'] = '16'
    assert allocation_max_memory_mb(fraction=0.6) == int(4000 * 16 * 0.6)


def test_per_cpu_without_cpus_per_task_assumes_one_cpu():
    os.environ['SLURM_MEM_PER_CPU'] = '4000'
    assert allocation_max_memory_mb(fraction=0.6) == int(4000 * 1 * 0.6)


def test_neither_variable_returns_the_default_unscaled():
    """No allocation in the environment (a laptop): `default` comes back
    exactly, never scaled by `fraction`."""
    assert allocation_max_memory_mb() is None
    assert allocation_max_memory_mb(default=8000) == 8000
    assert allocation_max_memory_mb(fraction=0.1, default=8000) == 8000


def test_fraction_is_applied_to_the_allocation():
    os.environ['SLURM_MEM_PER_NODE'] = '100000'
    assert allocation_max_memory_mb(fraction=0.5) == 50000
    assert allocation_max_memory_mb(fraction=0.25) == 25000


def test_result_is_an_integer_number_of_mb():
    os.environ['SLURM_MEM_PER_NODE'] = '100001'      # odd, so 0.6x is fractional
    result = allocation_max_memory_mb(fraction=0.6)
    assert isinstance(result, int)
    assert result == 60000                            # int() truncates, not rounds


def test_the_shipped_fraction_is_the_constants_module_value():
    """The argparse-style default `fraction=ALLOCATION_MEMORY_FRACTION` is
    read from `constants.py`, not re-spelled here."""
    os.environ['SLURM_MEM_PER_NODE'] = '100000'
    assert allocation_max_memory_mb() == int(100000 * ALLOCATION_MEMORY_FRACTION)


@pytest.fixture(scope='module')
def water_df():
    """Water/cc-pVDZ DF-RHF at pyscf's own default max_memory (4000 MB):
    far too small a system for the 900-MB-per-.9 threshold `DF.build` gates
    on, so this is the in-core branch."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
               basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    return mf


def test_describe_df_storage_water_is_in_core(water_df):
    info = describe_df_storage(water_df)
    assert info['cderi_in_core'] is True
    assert info['max_memory_mb'] == water_df.max_memory == 4000


def test_describe_df_storage_cderi_gb_matches_the_formula(water_df):
    nao = water_df.mol.nao_nr()
    naux = water_df.with_df.auxmol.nao_nr()
    expected_gb = nao * (nao + 1) // 2 * naux * 8 / 1e9
    info = describe_df_storage(water_df)
    assert info['cderi_gb'] == pytest.approx(expected_gb, rel=0.01)


def test_describe_df_storage_without_with_df_is_none():
    """A conventional (non-density-fitted) mean field has no `with_df`: the
    tensor question does not apply, and the helper says so rather than
    guessing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    info = describe_df_storage(mf)
    assert info['cderi_gb'] is None
    assert info['cderi_in_core'] is None
    assert info['max_memory_mb'] == mf.max_memory


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
