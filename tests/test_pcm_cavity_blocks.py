"""The PCM cavity's three-centre derivative integrals stay inside libcint's index.

libcint strides the components of a (comp, nao, nao, nk) integral block by a C
int, so a block past 2^31 - 1 integrals writes below its array; pyscf's
`grad_qv` blocks the cavity grid by `max_memory` alone. The gates here lower
the limit (`LIBCINT_BLOCK_LIMIT` to a few cavity points' worth) and count what
reaches libcint: every comp-3 cavity block of the S0 reaction-field force and
of the static self-energy skeleton must fit, and the blocked forces must equal
the unblocked ones.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import gto, scf
from pyscf.solvent.grad import pcm as pcm_grad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments import pcm_derivatives
from src.environments.pcm_derivatives import (LIBCINT_BLOCK_LIMIT, PYSCF_PCM_MAX_MEMORY_MB,
                                              PYSCF_PCM_MIN_BLOCK, bound_pyscf_pcm_blocks,
                                              cavity_blocks, libcint_block_fits,
                                              require_pyscf_pcm_hessian_fits,
                                              solvation_gradient)
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.environments.solvent_screening import SolventScreening
from tests.molecules import WATER_ROUNDED

H2O = WATER_ROUNDED
#: cavity points per block once the limit is lowered
POINTS = 7


def water(max_memory=4000):
    return gto.M(atom=H2O, basis='cc-pvdz', verbose=0, max_memory=max_memory)


@pytest.fixture
def libcint_calls(monkeypatch):
    """Every comp-3 three-centre block handed to libcint, as nao_i nao_j nk
    comp, with LIBCINT_BLOCK_LIMIT lowered to POINTS cavity points' worth for
    water/cc-pVDZ."""
    calls = []
    real = pyscf_df.incore.aux_e2

    def counted(mol, auxmol, intor='int3c2e', *args, **kwargs):
        if '_ip' in intor:
            calls.append(mol.nao * mol.nao * auxmol.nao * 3)
        return real(mol, auxmol, intor, *args, **kwargs)

    monkeypatch.setattr(pyscf_df.incore, 'aux_e2', counted)
    nao = water().nao
    monkeypatch.setattr(pcm_derivatives, 'LIBCINT_BLOCK_LIMIT',
                        nao * nao * 3 * POINTS, raising=False)
    return calls, nao * nao * 3 * POINTS


@pytest.mark.parametrize('nao, ngrids', [(1028, 2573), (1220, 3281),
                                         (1780, 4000)])
def test_cavity_blocks_fit_libcint_and_cover_the_grid(nao, ngrids):
    """Sizes past the int index in one block (nao ~ 1000-1800, a few
    thousand cavity points), with no memory bound."""
    blocks = cavity_blocks(nao, ngrids, comp=3, budget_gb=1e6)
    assert blocks[0][0] == 0 and blocks[-1][1] == ngrids
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
    assert all(nao * nao * (k1 - k0) * 3 <= LIBCINT_BLOCK_LIMIT
               for k0, k1 in blocks)


def test_the_s0_reaction_field_force_fits_libcint(libcint_calls):
    """`solvation_gradient`, the reaction-field term the S0 force of a
    PCM-wrapped ISDF-K mean field adds (`isdf_jk`'s gradient), equals pyscf's
    three terms and never hands libcint a block past the limit."""
    mol = water()
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.eps = 2.3741
    mf.with_solvent.lebedev_order = 11
    mf.with_solvent.build()
    dm = mf.get_init_guess()
    pcm = mf.with_solvent
    reference = (pcm_grad.grad_nuc(pcm, dm) + pcm_grad.grad_qv(pcm, dm)
                 + pcm_grad.grad_solver(pcm, dm))
    calls, limit = libcint_calls
    calls.clear()
    force = solvation_gradient(pcm, dm)
    assert pcm.surface['grid_coords'].shape[0] > 3 * POINTS
    assert calls and max(calls) <= limit, (max(calls), limit)
    assert np.abs(force - reference).max() < 1e-12


def test_the_static_self_energy_skeleton_fits_libcint(libcint_calls):
    """The solvated excited-state force's grid-potential derivative: the same
    value in blocks as whole."""
    mol = water()
    env = SolventScreening(mol, solvent='toluene')
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    nocc = mol.nelectron // 2
    weights = np.zeros(mol.nao)
    weights[nocc - 1], weights[nocc] = 1.0, -1.0
    calls, limit = libcint_calls
    calls.clear()
    blocked = env.static_self_energy_skeleton(mol, mf.mo_coeff, nocc, weights)
    assert calls and max(calls) <= limit, (max(calls), limit)
    whole_limit = 2 ** 31 - 1
    pcm_derivatives.LIBCINT_BLOCK_LIMIT = whole_limit
    whole = env.static_self_energy_skeleton(mol, mf.mo_coeff, nocc, weights)
    assert np.abs(blocked - whole).max() < 1e-12


def test_a_force_holds_one_small_block_at_a_time(monkeypatch):
    """The continuum terms of a force hold one cavity block at a time, and a
    block is at most PCM_CAVITY_BLOCK_GB: at CBP's nao (1090) and a few
    thousand cavity points, not the 4 GB tile budget of the ISDF routes."""
    from src.Base.numerics.constants import BYTES_PER_GB, PCM_CAVITY_BLOCK_GB
    nao, ngrids = 1090, 5000
    assert max((k1 - k0) * nao * nao * 3 * 8 for k0, k1 in
               cavity_blocks(nao, ngrids)) <= PCM_CAVITY_BLOCK_GB * BYTES_PER_GB
    budgets, plan = [], pcm_derivatives.cavity_blocks

    def recorded(nao, ngrids, comp=3, **kw):
        budgets.append(kw.get('budget_gb', PCM_CAVITY_BLOCK_GB))
        return plan(nao, ngrids, comp, **kw)
    monkeypatch.setattr(pcm_derivatives, 'cavity_blocks', recorded)
    mf = scf.RHF(water()).PCM()
    mf.with_solvent.eps = 2.3741
    mf.with_solvent.build()
    solvation_gradient(mf.with_solvent, mf.get_init_guess())
    assert budgets and set(budgets) == {PCM_CAVITY_BLOCK_GB}, budgets


def test_pyscf_own_pcm_routines_are_bounded_too():
    """pyscf's PCM force and Hessian, run by pyscf itself on the mean field a
    continuum hands out, block by `max_memory`: capped, no block of theirs can
    pass the limit at nao = 1028."""
    mol = water(max_memory=120000)
    env = SolventScreening(mol, solvent='toluene')
    mf = env.mean_field(mol, lambda m: scf.RHF(m).run(conv_tol=1e-10))
    for pcm in (mf.with_solvent, env._pcm):
        assert pcm.max_memory <= PYSCF_PCM_MAX_MEMORY_MB
    nao = 1028
    for comp in (1, 3, 9):
        nk = int(PYSCF_PCM_MAX_MEMORY_MB * .9e6 / 8 / nao ** 2 / comp)
        assert nao * nao * nk * comp <= LIBCINT_BLOCK_LIMIT



@pytest.mark.parametrize('nao, fits', [(1028, True), (2317, True), (2318, False),
                                       (2546, False), (5629, False)])
def test_pyscf_floor_passes_the_limit_above_nao_2317(nao, fits):
    """pyscf's floor of PYSCF_PCM_MIN_BLOCK points alone: inside libcint's
    index up to nao 2317, past it from 2318 (the film clusters' 2546-5629).
    One component is never strided."""
    assert libcint_block_fits(nao, PYSCF_PCM_MIN_BLOCK, 3) is fits
    assert libcint_block_fits(nao, PYSCF_PCM_MIN_BLOCK, 9) is fits
    assert libcint_block_fits(nao, PYSCF_PCM_MIN_BLOCK, 1)


def _uhf_in_toluene():
    mf = scf.UHF(water(max_memory=120000)).PCM()
    mf.with_solvent.eps = 2.3741
    mf.with_solvent.lebedev_order = 11
    bound_pyscf_pcm_blocks(mf.with_solvent)
    return mf.run(conv_tol=1e-10)


def test_a_pyscf_gradient_past_the_floor_runs_on_cavity_blocks(libcint_calls, monkeypatch):
    """The force of a mean field `fitted_force_supported` does not cover (here
    UHF) is pyscf's `Gradients()`, whose PCM term is the continuum's `grad`.
    Once pyscf's block passes the limit that term runs on `cavity_blocks`:
    no block past the limit, and the force pyscf gives where it fits."""
    mf = _uhf_in_toluene()
    calls, limit = libcint_calls
    calls.clear()
    blocked = mean_field_skeleton_force(mf)
    assert calls and max(calls) <= limit, (max(calls), limit)
    monkeypatch.setattr(pcm_derivatives, 'LIBCINT_BLOCK_LIMIT', 2 ** 31 - 1)
    calls.clear()
    whole = mean_field_skeleton_force(mf)
    assert max(calls) > limit       # pyscf's own block: the whole grid
    assert np.abs(blocked - whole).max() < 1e-10


def test_pyscf_pcm_hessian_past_the_limit_is_refused(monkeypatch):
    """pyscf's analytic PCM Hessian sizes its blocks itself (at least
    PYSCF_PCM_MIN_BLOCK points, one atom's points whole): refused once they
    pass the limit, run below it."""
    mf = scf.RHF(water()).PCM()
    mf.with_solvent.eps = 2.3741
    mf.with_solvent.lebedev_order = 11
    bound_pyscf_pcm_blocks(mf.with_solvent)
    assert require_pyscf_pcm_hessian_fits(mf) is mf
    assert require_pyscf_pcm_hessian_fits(scf.RHF(water())) is not None
    nao = mf.mol.nao
    monkeypatch.setattr(pcm_derivatives, 'LIBCINT_BLOCK_LIMIT', nao * nao * POINTS)
    with pytest.raises(ValueError, match='analytic PCM Hessian'):
        require_pyscf_pcm_hessian_fits(mf)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
