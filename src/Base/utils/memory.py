"""What a cluster allocation actually holds, read from the job environment.

pyscf's `max_memory` defaults to 4000 MB regardless of what SLURM handed the
job. On a 16-core node (`-c 16`) that default made anthracene
cc-pVTZ keep its 1.8 GB Cholesky-fitted DF tensor in core (SCF 32 s, static
exchange build 2.3 s) while pentacene cc-pVTZ's 6.4 GB tensor no longer fit,
so pyscf streamed it from a scratch file on GPFS at every J/K build instead --
a 13x SCF and a 24x static exchange build for 1.5x the basis. `max_memory`
(`Mole.max_memory`, inherited by `mf.max_memory`) is the one knob both `DF.build`
(`pyscf/df/df.py`) and this repository's own ISDF block budget
(`SingleReference.GW.qp_solve.static_exchange_diagonal`'s `exchange='df-direct'`
path, which reads `0.25 * mf.max_memory` into `block_memory_gb` when the caller
leaves it unset) size themselves against, so raising it is the whole fix.
"""
import os

import numpy as np

from src.Base.constants import ALLOCATION_MEMORY_FRACTION


def allocation_max_memory_mb(fraction=ALLOCATION_MEMORY_FRACTION, default=None):
    """The MB pyscf's `max_memory` should target on this job, or `default` off one.

    Reads what SLURM actually gave the job -- `SLURM_MEM_PER_NODE`, else
    `SLURM_MEM_PER_CPU * (SLURM_CPUS_PER_TASK or 1)` -- and scales it by
    `fraction`, since `max_memory` bounds only pyscf's own buffers (the DF
    tensor, the numint grid) and not the mo_coeff/ISDF-factor arrays and
    python overhead the rest of the process holds beside them: the probe that
    found the pentacene regression measured 7.8 GB RSS at anthracene with
    `max_memory` capped at 4000 MB. Neither SLURM variable set (no allocation,
    or a laptop) returns `default` unchanged, so an off-cluster run keeps
    whatever pyscf's own default or a caller's own value was.
    """
    node_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if node_mb is not None:
        total_mb = float(node_mb)
    else:
        cpu_mb = os.environ.get('SLURM_MEM_PER_CPU')
        if cpu_mb is None:
            return default
        cpus = float(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        total_mb = float(cpu_mb) * cpus
    return int(total_mb * fraction)


def describe_df_storage(mf):
    """Whether this mean field's DF tensor lives in RAM or streams from disk.

    `cderi_gb` is the Cholesky-fitted 3-index tensor's own size, nao_pair x
    naux x 8 bytes with nao_pair = nao*(nao+1)/2 -- the same product
    `pyscf.df.df.DF.build` compares against `max_memory` to choose
    `incore.cholesky_eri` (an ndarray) over `outcore.cholesky_eri` (a temp-file
    object streamed from disk), so `cderi_in_core` reads that decision back
    rather than re-deriving a threshold of its own. That switch is what turned
    pentacene's static exchange build 24x slower than anthracene's for 1.5x
    the basis.
    """
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        return dict(cderi_gb=None, cderi_in_core=None,
                    max_memory_mb=mf.max_memory)
    auxmol = getattr(with_df, 'auxmol', None)
    naux = auxmol.nao_nr() if auxmol is not None else None
    nao = with_df.mol.nao_nr()
    nao_pair = nao * (nao + 1) // 2
    cderi_gb = None if naux is None else nao_pair * naux * 8 / 1e9
    cderi = getattr(with_df, '_cderi', None)
    cderi_in_core = None if cderi is None else isinstance(cderi, np.ndarray)
    return dict(cderi_gb=cderi_gb, cderi_in_core=cderi_in_core,
                max_memory_mb=mf.max_memory)
