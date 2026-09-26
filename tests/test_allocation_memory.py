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
from src.Base.utils.memory import (allocation_max_memory_mb, describe_df_storage,
                                   tasks_per_node)

#: SLURM_* variables the parser reads; cleared before every case so a test
#: run under an actual allocation does not leak into these, and cleared after
#: it so the values a case sets do not leak into the rest of a pytest session.
SLURM_VARS = ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK',
              'SLURM_JOB_ID', 'SLURM_NTASKS_PER_NODE', 'SLURM_TASKS_PER_NODE')


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


def _sysconf(page_size, phys_pages):
    """A monkeypatched `os.sysconf` returning a known page size/count pair."""
    values = {'SC_PAGE_SIZE': page_size, 'SC_PHYS_PAGES': phys_pages}
    return lambda name: values[name]


def test_whole_node_allocation_inside_slurm_both_unset(monkeypatch):
    """A whole-node `--mem=0` allocation can leave both SLURM_MEM_PER_NODE and
    SLURM_MEM_PER_CPU unset; inside a SLURM job (`SLURM_JOB_ID` set) that
    reads the node's own physical memory rather than falling through to
    `default`, which is what let a whole-node job read pyscf's 4000 MB
    default and die building its DF tensor on a node with hundreds of GB
    free."""
    os.environ['SLURM_JOB_ID'] = '12345'
    monkeypatch.setattr(os, 'sysconf', _sysconf(4096, 2 ** 20))
    physical_mb = 4096 * 2 ** 20 / 2 ** 20     # = 4096 MB, by construction
    assert (allocation_max_memory_mb(fraction=0.6, default=8000)
            == int(physical_mb * 0.6))


def test_whole_node_allocation_inside_slurm_node_var_zero(monkeypatch):
    """SLURM can also report SLURM_MEM_PER_NODE=0 for a whole-node request:
    the same whole-node case, not literally zero memory."""
    os.environ['SLURM_JOB_ID'] = '12345'
    os.environ['SLURM_MEM_PER_NODE'] = '0'
    monkeypatch.setattr(os, 'sysconf', _sysconf(4096, 2 ** 20))
    physical_mb = 4096 * 2 ** 20 / 2 ** 20
    assert (allocation_max_memory_mb(fraction=0.6, default=8000)
            == int(physical_mb * 0.6))


def test_per_node_memory_is_shared_by_the_ranks_on_the_node():
    """SLURM_MEM_PER_NODE is the node's memory; with --ntasks-per-node=8 eight
    processes share it, so each one's max_memory is an eighth. With one
    rank per node (the case every earlier job had) nothing changes."""
    os.environ['SLURM_MEM_PER_NODE'] = '122880'
    os.environ['SLURM_NTASKS_PER_NODE'] = '8'
    assert allocation_max_memory_mb(fraction=0.6) == int(122880 / 8 * 0.6)
    os.environ['SLURM_NTASKS_PER_NODE'] = '1'
    assert allocation_max_memory_mb(fraction=0.6) == int(122880 * 0.6)


def test_tasks_per_node_reads_the_slurm_layout_string():
    """Without SLURM_NTASKS_PER_NODE the count comes from the leading integer
    of SLURM_TASKS_PER_NODE, which SLURM writes as `8(x2)` for two nodes of
    eight; unset or unparsable means one."""
    assert tasks_per_node() == 1
    os.environ['SLURM_TASKS_PER_NODE'] = '8(x2)'
    assert tasks_per_node() == 8
    os.environ['SLURM_MEM_PER_NODE'] = '122880'
    assert allocation_max_memory_mb(fraction=0.6) == int(122880 / 8 * 0.6)
    os.environ['SLURM_TASKS_PER_NODE'] = 'nonsense'
    assert tasks_per_node() == 1


def test_whole_node_physical_memory_is_shared_by_the_ranks_on_the_node(monkeypatch):
    """The whole-node fallback is the node's physical memory, shared the same
    way: eight ranks on an exclusive --mem=0 node each get an eighth."""
    os.environ['SLURM_JOB_ID'] = '12345'
    os.environ['SLURM_NTASKS_PER_NODE'] = '8'
    monkeypatch.setattr(os, 'sysconf', _sysconf(4096, 2 ** 20))
    assert (allocation_max_memory_mb(fraction=0.6, default=8000)
            == int(4096 / 8 * 0.6))


def test_per_cpu_form_is_already_per_task():
    """SLURM_MEM_PER_CPU times the task's CPUs is one process's share
    already; the ranks per node do not divide it again."""
    os.environ['SLURM_MEM_PER_CPU'] = '7680'
    os.environ['SLURM_CPUS_PER_TASK'] = '16'
    os.environ['SLURM_NTASKS_PER_NODE'] = '8'
    assert allocation_max_memory_mb(fraction=0.6) == int(7680 * 16 * 0.6)


def test_outside_slurm_both_unset_still_returns_the_default():
    """No `SLURM_JOB_ID` at all (a laptop): the whole-node fallback must not
    fire, so `default` still comes back unscaled, same as
    `test_neither_variable_returns_the_default_unscaled`."""
    assert 'SLURM_JOB_ID' not in os.environ
    assert allocation_max_memory_mb(default=8000) == 8000


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
