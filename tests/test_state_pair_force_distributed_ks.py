"""The vertical state-pair record on a Kohn-Sham reference converged by the
distributed ISDF-K SCF, under simulated ranks, on the production flag set.

`calc_vertical_excitation` with space-time chi0, ISDF factors, SOP residues
on the frontier states, sliced factors on the row fit, the grid BSE adjoint
and grid accuracy G1, the mean field an ISDF-K SCF that `converged_factory`
converges over the ranks. Every rank must hold rank 0's grid after that SCF:
otherwise rank 1 re-prunes its own on its first Fock build and the xc
grid-response skeleton, which gathers its tiles over the ranks, refuses the
mismatch.

Gated, water/cc-pVDZ, PBE0 and LRC-wPBEh, at 2 and 3 simulated ranks: the
record completes, the mean field's serial ISDFJK is never built on any rank
(the orbital response runs on the distributed SCF's handle), every rank's
force is rank 0's bitwise, and the force lies within the anchored bar of the
serial record's: `COMPOSED_GRAD_K` times what the serial force moves when the
whole record runs again with BLAS held to one thread (`one_thread`: its SCF
converges to a mean field a round-off apart, as the distributed SCF's does,
and every threaded GEMM reassociates), floored at `COMPOSED_GRAD_FLOOR`.
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (COMPOSED_GRAD_K, SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.utils.mpi_grid import current_comm, distributed, run_simulated
from src.properties import excitations
from src.properties.excitations import SurfaceSpec, calc_vertical_excitation
from tests.test_mpi_routes import COMPOSED_GRAD_FLOOR, one_thread

WATER = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
BASIS = 'cc-pvdz'
GRID_ACCURACY = 'G1'
FUNCTIONALS = ('pbe0', 'lrc-wpbeh')
SIZES = (2, 3)
MAX_MEMORY = 4000
_LOCK = threading.Lock()
MEAN_FIELDS = {}


def scf_factory(xc):
    """The driver's `--scf isdf` mean field: converged here serially, left
    unrun inside a region for the distributed SCF."""
    def build(mol):
        mf = dft.RKS(mol, xc=xc)
        mf.max_memory = MAX_MEMORY
        counts, n_start = resolve_isdf_grid(GRID_ACCURACY, BASIS, ['H', 'O'],
                                            auxbasis=BASIS + '-ri')
        mf = isdf_jk(mf, auxbasis=BASIS + '-ri', counts=counts,
                     n_start=n_start)
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        if current_comm() is None:
            mf.kernel()
        with _LOCK:
            MEAN_FIELDS[threading.get_ident()] = mf
        return mf
    return build


def record(xc, forces):
    """The vertical record of S1; its driving force into `forces`, and
    whether the mean field's serial ISDFJK was built."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)
    spec = SurfaceSpec(GroundState('dft', xc), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics={'grid_accuracy': GRID_ACCURACY, 'sliced': True,
                                 'fit': 'rows', 'bse_adjoint': 'grid'})
    rec = calc_vertical_excitation(spec, Excitation('singlet', root=1,
                                                    kernel='bse'),
                                   mol, scf_factory(xc))
    tid = threading.get_ident()
    return rec['omega_eV'], forces[tid], MEAN_FIELDS[tid].with_df._built


@pytest.fixture
def forces(monkeypatch):
    """{thread: the force `evaluate` returned on it}."""
    seen, evaluate = {}, excitations.evaluate

    def kept(*args, **kwargs):
        out = evaluate(*args, **kwargs)
        with _LOCK:
            seen[threading.get_ident()] = np.array(out[0], copy=True)
        return out

    monkeypatch.setattr(excitations, 'evaluate', kept)
    return seen


@pytest.mark.parametrize('xc', FUNCTIONALS)
def test_record_over_ranks_is_the_serial_one(xc, forces):
    """At 2 and 3 ranks: completes, one force on every rank, within the
    anchored bar of serial."""
    with distributed(None):
        omega, ref, _ = record(xc, forces)
    again = one_thread(lambda: record(xc, forces)[1])
    scatter = 0.0 if again is None else float(np.abs(again - ref).max())
    bar = max(COMPOSED_GRAD_FLOOR, COMPOSED_GRAD_K * scatter)
    lines = [f'serial omega {omega:.8f} eV, one-thread scatter '
             f'{scatter:.2e}, bar {bar:.2e} Ha/Bohr']
    for size in SIZES:
        out = run_simulated(lambda comm: record(xc, forces), size)
        # the orbital response ran on the distributed SCF's handle
        assert [built for _, _, built in out] == [False] * size
        for om, force, _ in out[1:]:
            assert om == out[0][0]
            assert np.array_equal(force, out[0][1])
        d = float(np.abs(out[0][1] - ref).max())
        lines.append(f'{size} ranks: omega {out[0][0]:.8f} eV, |d| {d:.2e} '
                     f'= {d / bar:.2f} of the bar')
        assert d < bar, lines
    assert np.abs(ref).max() > 1e-3
    print(f'\n{xc}: ' + '; '.join(lines))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
