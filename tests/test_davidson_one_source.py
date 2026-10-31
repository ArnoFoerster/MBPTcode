"""The ADC root-following Davidson reads its settings from the one solver.

  1. davidson_follow carries no tolerance or subspace literal of its own: the
     subspace it hands the solver is the solver's default, and a tolerance is
     asked for, never assumed.
  2. the C2H4 IP-ADC(3) Koopmans root (STO-3G) converges at that default and
     is the dense supermatrix's root of the largest weight on the HOMO.
  3. one seed rule: `diagonal_seeds` extends its cut through a degenerate
     cluster, as the linear-response guess does, and says so when its limit
     stops it inside one.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import (get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.ADC import ADCSolverRestricted
from src.SingleReference.ADC import solve as adc_solve
from src.SingleReference.LinearResponse.davidson_shared import guess_indices
from src.Solvers import davidson as D

ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; '
            'H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')


def test_davidson_follow_has_no_settings_of_its_own():
    sig = inspect.signature(adc_solve.davidson_follow)
    assert sig.parameters['max_subspace'].default is None
    assert sig.parameters['tol'].default is inspect.Parameter.empty


def test_ethylene_koopmans_root_converges_at_the_solver_default():
    mol = gto.M(atom=ETHYLENE, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    solver = ADCSolverRestricted.from_arrays(eps, eri, nocc=nocc, level='adc3')
    H = solver.build_supermatrix(nocc)
    w, v = np.linalg.eigh(H)
    homo_weight = v[nocc - 1, :] ** 2
    dense = w[np.argmax(homo_weight)]
    aop, diag, dims = solver.build_matrix_free_operator(nocc)
    e, Z, _, conv = adc_solve.davidson_follow(aop, diag, dims['nH'], solver.norb,
                                              nocc - 1, None, 1, tol=1e-8)
    assert conv.all()
    assert abs(e[0] - dense) < 1e-7


def test_the_seed_cut_does_not_split_a_degenerate_cluster():
    diag = np.concatenate([[0.0], 1.0 + 1e-7 * np.arange(6), [5.0, 6.0, 7.0, 8.0]])
    cols = [int(np.argmax(s)) for s in D.diagonal_seeds(diag, 2, limit=20)]
    assert sorted(cols) == list(range(7))
    # the linear-response guess is the same rule
    assert sorted(guess_indices(diag, 1, guess_factor=2)) == list(range(7))


def test_a_limit_that_stops_inside_a_cluster_is_warned_about():
    diag = np.concatenate([[0.0], np.ones(30)])
    counters = {}
    with pytest.warns(RuntimeWarning, match='cluster'):
        seeds = D.diagonal_seeds(diag, 2, limit=4, counters=counters)
    assert len(seeds) == 4 and counters['guess_truncated']
