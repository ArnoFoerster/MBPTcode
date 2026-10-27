"""The third-order quadruples T4^(3) Laplace-fused and never formed
(laplace_amplitudes.LaplaceAmplitude over the generated numerator term tables
of mpn_order4_ground_state_restricted) against the materialized T4^(3), on
H2O/6-31G (all four quadruples blocks nonzero).

1. The term table is the numerator: summed without the denominator it equals
   the generated t4_3 numerator to rounding (the emission, independent of the
   quadrature).
2. The Laplace blocks and their lambda twins converge to the materialized
   blocks with the minimax grid size (6 -> 24 points).
3. Consumers: T2^(4), T3^(4) (and so E^(5)) built on the Laplace T4^(3) agree
   with the materialized route to the quadrature accuracy.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri
from src.SingleReference.CC.cached_einsum import factorized_einsum
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted, STORED_BLOCKS, _blocks, _call, order4_ground_state_module)
from src.SingleReference.common.mp_amplitudes import to_l
from src.SingleReference.DensityMatrix.generated_mpn_restricted import \
    mpn_density_pieces_restricted as gen

ATOM = 'O 0 0 0; H 0 0 0.96; H 0.96 0 0'
NTAUS = (6, 10, 16, 24)
TOL_TABLE = 1e-12
TOL_LAPLACE = 1e-9


def _ground_state():
    """The generated order-4 ground-state module; absent (not generated on this
    machine), the test is skipped."""
    try:
        return order4_ground_state_module()
    except ImportError as err:
        pytest.skip(str(err))


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))


@pytest.fixture(scope='module')
def chain():
    mol = gto.M(atom=ATOM, basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.run()
    eps = get_orbital_energies(mf, representation='spatial')
    ga, gb, gab = get_antisymmetrized_spin_block_eri(mol, mf)
    mpn = MPnDensityDriverRestricted(np.diag(eps), ga, gab, gb, mol.nelectron // 2)
    t2a, t2b = mpn.compute_t2_1()
    E2 = mpn.compute_E(t2a, t2b)
    o2 = mpn.compute_order2_amplitudes(t2a, t2b)
    o3 = mpn.compute_order3_amplitudes(*o2, t2a, t2b, E2)
    return mpn, t2a, t2b, E2, o2, o3


def test_term_table_is_the_numerator(chain):
    mpn, _, _, _, o2, _ = chain
    pool = {**mpn.kernel_args(), 't2_2_aaaa': o2[1], 't2_2_abab': o2[2],
            **_blocks('t3_2', 3, o2[3:6]), **_blocks('t4_2', 4, o2[6:10])}
    tables = _ground_state()
    for b in STORED_BLOCKS[4]:
        ref = _call(getattr(gen, f't4_3_{b}_numerator'), pool)
        got = np.zeros_like(ref)
        for c, pieces in _call(getattr(tables, f't4_3_{b}_terms'), pool):
            expr = ','.join(s for s, _ in pieces) + '->abcdijkl'
            got += c * factorized_einsum(expr, *[a for _, a in pieces])
        d = _rel(got, ref)
        print(f"t4_3 {b}: term table vs numerator {d:.1e}")
        assert d <= TOL_TABLE, (b, d)


def test_laplace_converges_to_the_materialized_blocks(chain):
    _ground_state()
    mpn, _, _, _, o2, _ = chain
    dense = mpn.compute_t4_3(o2[1], o2[2], o2[3:6], o2[6:10])
    errs = []
    for ntau in NTAUS:
        lap = mpn.compute_t4_3(o2[1], o2[2], o2[3:6], o2[6:10], laplace_ntau=ntau)
        e = max(_rel(l.dense(), d) for l, d in zip(lap, dense))
        e_l = max(_rel(to_l(l, 4, b).dense(), to_l(d, 4, b))
                  for l, d, b in zip(lap, dense, STORED_BLOCKS[4]))
        print(f"ntau={ntau}: Laplace T4^(3) vs materialized {e:.1e} (lambda twin {e_l:.1e})")
        errs.append(max(e, e_l))
    assert errs[-1] <= TOL_LAPLACE, errs
    assert errs[-1] < errs[0], errs


def test_order4_consumers(chain):
    _ground_state()
    mpn, t2a, t2b, E2, o2, o3 = chain
    E3 = mpn.compute_E(o2[1], o2[2])
    dense = mpn.compute_order4_amplitudes(
        o2, o3, mpn.compute_t4_3(o2[1], o2[2], o2[3:6], o2[6:10]), t2a, t2b, E2, E3)
    lap = mpn.compute_order4_amplitudes(
        o2, o3, mpn.compute_t4_3(o2[1], o2[2], o2[3:6], o2[6:10], laplace_ntau=NTAUS[-1]),
        t2a, t2b, E2, E3)
    worst = max(_rel(l, d) for l, d in zip(lap, dense))
    e5 = float(mpn.compute_E(lap[0], lap[1]))
    e5_d = float(mpn.compute_E(dense[0], dense[1]))
    print(f"T2^(4)/T3^(4) Laplace vs materialized T4^(3): {worst:.1e}; E^(5) {e5:.12e} "
          f"vs {e5_d:.12e}")
    assert worst <= TOL_LAPLACE, worst
    assert abs(e5 - e5_d) <= TOL_LAPLACE * abs(e5_d)
