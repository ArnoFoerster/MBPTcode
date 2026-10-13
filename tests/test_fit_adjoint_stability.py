"""The whole fit's adjoint carries the last bits of its seeds no further than
the row fit's.

`isdf_derivatives.fit_adjoint` forms the Gram matrix's adjoint in the
low-rank form G_bar = -B^T A_bar from its two single solves, as the row fit's
adjoint does (G_bar = -W Z^T, `separable_ri.fit_rows_adjoints`). On
water/cc-pVDZ RHF, the BSE@GW excitation force of a sliced Davidson chain:
the four Casida-level seeds at its reverse call, every element moved one ulp
up or down by `ULP_DRAWS` fixed draws, are folded through the chain's own
fold on its own forward pieces (which, unmoved, give the force bitwise), and
the force moves at most `FIT_ADJOINT_ULP_RESPONSE_MAX`, on the whole fit and
on the row fit alike (6.5e-13 to 1.0e-12 and 2.7e-13 to 5.2e-13 Ha/Bohr).
pyscf's OpenMP is held to one thread, so every number here repeats bit for
bit. The form G_bar = -G^-1 (A^T B_bar) G^-1 by two more solves
(`two_solve_fit_adjoint`) moves the force about 1e4 times further and fails
the gate.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, lib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (FIT_ADJOINT_ULP_RESPONSE_MAX,
                                FIT_CHOLESKY_BLOCK)
from src.Base.separable_ri import DEFAULT_REGULARIZATION
from src.gradients import isdf_derivatives
from src.gradients.excited_state import ExcitedStateChain
from tests.test_chain_sliced_factors import BASIS, H2O, chain_scf

#: Fixed draws of the one-ulp seed changes.
ULP_DRAWS = (3, 5, 7)
#: The row fit's tile edge: water's 444 points in 7 tiles.
ROW_FIT_BLOCK = 64


@pytest.fixture
def pyscf_one_thread():
    """pyscf's OpenMP GEMM on one thread, so its K partials are summed in one
    order and a fold repeats its bits."""
    threads = lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(1)
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(threads)


def two_solve_fit_adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
    """`fit_adjoint` with G_bar = -G^-1 (A^T B_bar) G^-1 by two more Cholesky
    solves on (nk, nk) arrays: the same quantity in a less stable form."""
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    d = 1.0 / s
    Dt = D * d[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    cho = scipy.linalg.cho_factor(G, lower=True)
    A = F @ Dt.T
    B = scipy.linalg.cho_solve(cho, A.T).T
    B_bar = M_bar * d[None, :]
    d_bar = np.einsum('bk,bk->k', M_bar, B)
    A_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
    Y = scipy.linalg.cho_solve(cho, A.T @ B_bar)
    G_bar = -scipy.linalg.cho_solve(cho, Y.T).T
    F_bar = A_bar @ Dt
    Dt_bar = (G_bar + G_bar.T) @ Dt + A_bar.T @ F
    d_bar = d_bar + np.einsum('kr,kr->k', Dt_bar, D)
    s_bar = -d_bar * d ** 2
    D_bar = Dt_bar * d[:, None]
    c = s_bar / s
    for r0 in range(0, len(c), FIT_CHOLESKY_BLOCK):
        r1 = r0 + FIT_CHOLESKY_BLOCK
        D_bar[r0:r1] += c[r0:r1, None] * D[r0:r1]
    return D_bar, F_bar


def ulp_responses(**fit):
    """(the force, [max |force moved| per draw]) of the water chain on the
    fit `fit` names, its seeds moved one ulp per element."""
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    chain = ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol),
                              solver='davidson', sliced=True, **fit)
    rec = {}
    seeds_of = chain._casida_seeds

    def probed(pieces, n, m=None):
        out = seeds_of(pieces, n, m)
        rec.update(pieces=pieces,
                   seeds=tuple(np.array(a, copy=True) for a in out))
        return out

    chain._casida_seeds = probed
    g = np.asarray(chain.excitation_gradient()[0])

    def fold(seeds):
        return np.asarray(chain._fold_to_nuclei(
            rec['pieces'], *[np.array(a, copy=True) for a in seeds])[0])

    assert np.array_equal(fold(rec['seeds']), g)
    eps = np.finfo(float).eps
    moved = []
    for draw in ULP_DRAWS:
        rng = np.random.default_rng(draw)
        seeds = [a * (1 + eps * rng.choice((-1.0, 1.0), a.shape))
                 for a in rec['seeds']]
        moved.append(float(np.abs(fold(seeds) - g).max()))
    return g, moved


def test_one_ulp_seeds_move_the_force_by_a_last_bit(pyscf_one_thread):
    """The whole fit and the row fit: one-ulp seed changes move the force at
    most FIT_ADJOINT_ULP_RESPONSE_MAX."""
    g_whole, whole = ulp_responses()
    g_rows, rows = ulp_responses(fit='rows', fit_block=ROW_FIT_BLOCK)
    print(f'[info] |F| {np.abs(g_whole).max():.4f} Ha/Bohr; one-ulp seeds '
          f'move it {max(whole):.2e} on the whole fit, {max(rows):.2e} on '
          f'the row fit: {max(whole) / FIT_ADJOINT_ULP_RESPONSE_MAX:.4f} and '
          f'{max(rows) / FIT_ADJOINT_ULP_RESPONSE_MAX:.4f} of '
          f'{FIT_ADJOINT_ULP_RESPONSE_MAX:.0e}; the two realizations of the '
          f'one estimator {np.abs(g_whole - g_rows).max():.2e} apart')
    assert max(whole) <= FIT_ADJOINT_ULP_RESPONSE_MAX, whole
    assert max(rows) <= FIT_ADJOINT_ULP_RESPONSE_MAX, rows


def test_the_two_solve_form_fails_the_gate(pyscf_one_thread, monkeypatch):
    """The same gate with the whole fit's G_bar by two more solves fails."""
    monkeypatch.setattr(isdf_derivatives, 'fit_adjoint',
                        two_solve_fit_adjoint)
    _, moved = ulp_responses()
    print(f'[info] the two-solve form: one-ulp seeds move the force '
          f'{min(moved):.2e} to {max(moved):.2e} Ha/Bohr, '
          f'{max(moved) / FIT_ADJOINT_ULP_RESPONSE_MAX:.0f} x '
          f'{FIT_ADJOINT_ULP_RESPONSE_MAX:.0e}')
    assert max(moved) > FIT_ADJOINT_ULP_RESPONSE_MAX


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
