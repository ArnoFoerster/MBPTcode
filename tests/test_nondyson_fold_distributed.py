"""The non-Dyson static block over ranks (adc_r_nondyson_df.static_block): the columns of the
fold are split in contiguous blocks, the rows of U and the columns of G travel verbatim, so every
rank holds the same block and it equals the serial one. Simulated ranks (threads, real reductions)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.ADC import ADCSolverRestricted, build_static_correction
from src.SingleReference.ADC.adc_r_nondyson_df import static_block
from src.SingleReference.ADC.adc_r_sigma_df import build_pieces


def _pieces(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    eps = get_orbital_energies(mf, representation='spatial')
    nocc, n = mol.nelectron // 2, len(eps)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n).reshape(n * n, n * n)
    w, V = np.linalg.eigh(eri)
    keep = w > 1e-14 * w.max()
    B = (np.sqrt(w[keep]) * V[:, keep]).T.reshape(-1, n, n)
    s = ADCSolverRestricted.from_arrays(eps, B_aa=B, nocc=nocc, level="adc3")
    sc = build_static_correction(mf, kind='mp2_unrelaxed', B_aa=B)
    return build_pieces(s, nocc, sc)


@pytest.fixture(scope='module')
def pieces():
    return _pieces('N 0 0 0.1; H 0.94 0 -0.25; H -0.47 0.81 -0.25; H -0.47 -0.81 -0.25', 'cc-pvdz')


@pytest.mark.parametrize('sector', ['IP', 'EA'])
@pytest.mark.parametrize('size', [2, 3, 7])
@pytest.mark.parametrize('memory_gb', [40.0, 1e-9])    # rows of U gathered / streamed forward coupling
def test_static_block_over_ranks_equals_serial(pieces, sector, size, memory_gb):
    M0, _ = static_block(pieces, sector, batch=2, memory_gb=memory_gb)
    out = run_simulated(lambda comm: static_block(pieces, sector, batch=2, memory_gb=memory_gb)[0], size)
    for r, M in enumerate(out):
        assert np.max(np.abs(M - M0)) < 1e-13, f'rank {r}'
        assert np.array_equal(M, out[0]), f'rank {r} differs from rank 0'
