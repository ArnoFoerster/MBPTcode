"""The active/environment partition: index sets, masks and the cut-degeneracy refusal.

Every later downfolding step reads these index conventions, so the tests pin
them: the pair order of `active_pair_mask`, the local-index meaning of
`ph_element_mask`, and which orbital pairs `partition` refuses.
"""
import numpy as np
import pytest

from src.Base.numerics.constants import QP_DEGENERACY_TOL
from src.MultiReference.Downfolding.partition import (
    active_pair_mask, partition, ph_element_mask)
from src.SingleReference.common.active_space import ActiveSpace

EPS = np.array([-1.0, -0.6, -0.4, 0.3, 0.7, 1.1])


def test_masks():
    part = partition(ActiveSpace.from_indices(6, 3, [1, 2], [3, 5]), EPS)
    assert part.act.tolist() == [1, 2, 3, 5]
    assert part.inactive.tolist() == [0]
    assert part.env.tolist() == [0, 4]
    assert (part.n_act, part.n_act_occ) == (4, 2)
    assert active_pair_mask(part).reshape(3, 3).tolist() == [
        [0, 0, 0], [1, 0, 1], [1, 0, 1]]
    m = ph_element_mask(part)
    assert m.shape == (4,) * 4
    assert m[0, 2, 3, 1] and m[2, 0, 0, 3] and m[2, 0, 3, 1] and m[0, 2, 1, 3]
    assert not m[0, 1, 2, 3] and not m[0, 0, 2, 2]
    # exactly (2 * n_occ * n_virt)^2 elements
    assert m.sum() == (2 * 2 * 2) ** 2


def test_refuses_a_cut_degenerate_pair():
    eps = EPS.copy()
    eps[2] = eps[1] + 5e-5
    space = ActiveSpace.from_indices(6, 3, [2], [3])
    with pytest.raises(ValueError, match="from_irreps") as err:
        partition(space, eps)
    assert "from_population" in str(err.value)


def test_accepts_a_degenerate_pair_taken_whole():
    eps = EPS.copy()
    eps[2] = eps[1] + 5e-5
    partition(ActiveSpace.from_indices(6, 3, [1, 2], [3]), eps)


def test_gap_degeneracy_is_not_a_cut():
    eps = EPS.copy()
    eps[3] = eps[2] + 0.5 * QP_DEGENERACY_TOL     # HOMO/LUMO nearly equal
    partition(ActiveSpace.from_indices(6, 3, [2], [3]), eps)
    partition(ActiveSpace.from_indices(6, 3, [1], [4]), eps)


def test_n2_contiguous_window_cut():
    from pyscf import gto, scf
    mol = gto.M(atom='N 0 0 0; N 0 0 0.90', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    eps, nocc = np.asarray(mf.mo_energy), mol.nelectron // 2
    space = ActiveSpace.from_counts(eps, nocc, 3, 3)
    gaps = [abs(eps[a] - eps[e])
            for act, env in ((space.occ_active, space.env_occ),
                             (space.virt_active, space.env_virt))
            for a in act for e in env]
    if min(gaps) < QP_DEGENERACY_TOL:
        with pytest.raises(ValueError, match="from_irreps"):
            partition(space, eps)
    else:
        partition(space, eps)
