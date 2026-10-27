"""Near-degenerate orbitals of the explicit quasiparticle set share ONE
correction on the cubic SOP chain, frozen with the set, and the force
differentiates exactly that.

Diagonal G0W0 gives each orbital index its own Sigma_pp. Two virtuals
4.2e-5 Ha apart (thioformaldehyde/cc-pVTZ, orbitals 18/19) rotate into each
other across an avoided crossing within 1e-4 Bohr and trade corrections
8 mHa apart, so eps^QP moves at 5 Ha/Bohr and the surface bends on a scale
below the saddle check's step. A block of orbitals closer than the chain's
`degeneracy_tol` (QP_DEGENERACY_TOL) at the reference geometry now
gives every member eps_p + mean_q (eps^QP_q - eps_q), symmetric in its
members.

The blocks are true degeneracies only: closer than `degeneracy_tol` in energy
AND with their own reference corrections within the bar
QP_BLOCK_CORRECTION_TOL, which no run sets and which the tests here vary by
patching it (`correction_gate`). Orbitals merely close in energy keep their own
corrections: formaldehyde/cc-pVDZ (HF) orbitals 12/13 are 4.2e-3 Ha apart
with corrections 370 meV apart, and a block there would put each 185 meV off
its own analytic-continuation shift; two orbitals 5e-5 Ha apart whose
corrections differ by mHa are an accidental near-degeneracy all the same.
Every candidate block is recorded, kept or rejected with why.

NH3/cc-pVDZ, HF, SOP residues, window 1-9, dense Casida. Near C3v the E
pairs are split by 2e-5 to 3e-5 Ha and form blocks. The force check widens
`degeneracy_tol` and the correction gate on a distorted NH3 so the blocks join
orbitals whose own corrections differ by up to 15 mHa: an adjoint that seeded each root with
its own weight misses the energy's slope there by 3e-3 Ha/Bohr.
"""
import contextlib
import warnings
from unittest import mock

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.numerics import numerics_of
import src.gradients.gwbse.qp_set as qp_set_module
from src.Base.numerics.constants import (HARTREE_TO_MEV, QP_BLOCK_CORRECTION_TOL,
                                         QP_DEGENERACY_TOL)
from src.gradients.gwbse.conventions import FrozenConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.diagnostics import qp_bookkeeping
from src.SingleReference.GW.qp_selection import ac_quasiparticle_shifts
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_jkfit_differentiable_checked

NEAR_C3V = ('N 0 0 0.1173; H 0 0.9377 -0.2738; H 0.8121 -0.4689 -0.2738; '
            'H -0.8121 -0.4689 -0.2738')
DISTORTED = ('N 0 0 0.1173; H 0 0.9377 -0.2738; H 0.8521 -0.4689 -0.2738; '
             'H -0.8121 -0.4689 -0.2938')
BASIS = 'cc-pvdz'
WINDOW = tuple(range(1, 10))
#: Ha. Joins orbitals 2-4 and 5-7 of the distorted molecule into blocks.
WIDE_BLOCKS = 0.25
#: Bohr, and the force/energy agreement it resolves (measured 5e-9).
FD_STEP = 1e-4
FD_TOL = 1e-7
#: Ha: (eps_p + c) - eps_p is c to this.
ROUND_OFF = 1e-14
#: Bohr: moves the E pairs apart by more than QP_DEGENERACY_TOL.
SPLIT_STEP = 6e-2
FORMALDEHYDE = 'O 0 0 1.205; C 0 0 0; H 0 0.943 -0.587; H 0 -0.943 -0.587'
#: orbitals 12/13 of FORMALDEHYDE: 4.2e-3 Ha apart, not degenerate
CLOSE_PAIR = (12, 13)
#: meV: the pole model's own distance from the continuation on the pair
#: (measured 3.0 and 0.3), against 188/185 when the pair shares a correction
SOP_AC_TOL_MEV = 10.0
#: meV: how far apart the pair's own corrections are at least (measured 370)
OWN_SPLIT_MEV = 300.0
#: Ha: a degeneracy_tol that joins the pair, the tolerance that used to apply
JOINING_TOL = 5e-3
#: a correction gate that admits any candidate: no gate
NO_GATE = np.inf
#: Ha: orbital energies of a synthetic set; 1 and 2 are 5e-5 Ha apart
SYNTHETIC_EPS = np.array([-0.5, 0.1, 0.10005, 0.3])
#: Ha: own corrections of the synthetic set; 1 and 2 differ by 5 mHa
ACCIDENTAL = np.array([-0.02, 0.010, 0.015, 0.03])
#: Ha: ... and agree to 1e-7, as a symmetry-degenerate pair's do
DEGENERATE = np.array([-0.02, 0.012, 0.012 + 1e-7, 0.03])
#: Ha: three members within 1e-4 Ha; 1 and 2 agree, 3 is 5 mHa off
TRIPLE_EPS = np.array([-0.5, 0.1, 0.10003, 0.10006])
TRIPLE = np.array([-0.02, 0.012, 0.012 + 1e-7, 0.017])


def correction_gate(value):
    """The bar QP_BLOCK_CORRECTION_TOL as the chain reads it, set to `value`
    inside the context (a test's lever; no run sets a bar)."""
    if value is None:
        return contextlib.nullcontext()
    return mock.patch.object(qp_set_module, 'QP_BLOCK_CORRECTION_TOL',
                             value)


def chain(atom, qp_window=WINDOW, **tolerances):
    """The pole-model chain on `atom`, its `degeneracy_tol` where given."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=atom, basis=BASIS, verbose=0)
    return ExcitedStateChain(
        mol, rhf_df_jkfit_differentiable_checked, mf=rhf_df_jkfit_differentiable_checked(mol), numerics=g1_numerics(mol, numerics_of({'qp_window': list(qp_window), 'residue_route': 'sop', 'solver': 'dense', **tolerances})))


def displaced(mol, atom, axis, step):
    out = mol.copy()
    coords = out.atom_coords().copy()
    coords[atom, axis] += step
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def corrections(ch, mol):
    """eps^QP - eps per orbital of the set, and the set's blocks as orbitals."""
    fwd = ch.shared_forward(mol, ch.mean_field(mol)[1])
    states = np.asarray(ch.resolved.qp_set)
    return (fwd.eps_qp - fwd.eps)[states], [states[list(b)]
                                            for b in ch.resolved.qp_blocks[1]], fwd


def bare_chain(**kw):
    """An `ExcitedStateChain` holding only what deciding blocks reads."""
    ch = ExcitedStateChain.__new__(ExcitedStateChain)
    ch.settings = Numerics()
    ch.conventions = FrozenConventions(qp_set=np.arange(len(SYNTHETIC_EPS)))
    ch.qp_hamiltonian = False
    ch.degeneracy_tol = kw.get('degeneracy_tol', QP_DEGENERACY_TOL)
    return ch


def test_an_accidental_near_degeneracy_keeps_its_own_corrections():
    ch = bare_chain()
    ws = SYNTHETIC_EPS + ACCIDENTAL
    out = ch.quasiparticles.block_shared(SYNTHETIC_EPS, ws)
    assert np.abs(out - ws).max() < ROUND_OFF
    assert ch.resolved.qp_blocks[1] == ()
    rec = ch.conventions.qp_block_record
    assert rec['kept'] == [] and len(rec['rejected']) == 1
    rej = rec['rejected'][0]
    assert rej['members'] == [1, 2] and rej['split_into'] == []
    assert abs(rej['energy_spread_Ha'] - 5e-5) < 1e-12
    assert abs(rej['correction_spread_Ha'] - 5e-3) < 1e-12
    assert 'accident' in rej['why']
    # with the correction gate off, the old rule averages them
    off = bare_chain()
    with correction_gate(NO_GATE):
        shared = off.quasiparticles.block_shared(SYNTHETIC_EPS, ws) - SYNTHETIC_EPS
    assert np.ptp(shared[[1, 2]]) < ROUND_OFF


def test_a_true_degeneracy_shares_one_correction():
    ch = bare_chain()
    ws = SYNTHETIC_EPS + DEGENERATE
    out = ch.quasiparticles.block_shared(SYNTHETIC_EPS, ws) - SYNTHETIC_EPS
    assert ch.resolved.qp_blocks[1] == ((1, 2),)
    assert np.ptp(out[[1, 2]]) < ROUND_OFF
    assert abs(out[1] - DEGENERATE[[1, 2]].mean()) < ROUND_OFF
    assert np.abs(out[[0, 3]] - DEGENERATE[[0, 3]]).max() < ROUND_OFF
    kept = ch.conventions.qp_block_record['kept']
    assert [k['members'] for k in kept] == [[1, 2]]
    assert kept[0]['correction_spread_Ha'] < QP_BLOCK_CORRECTION_TOL


def test_a_candidate_splits_into_the_members_whose_corrections_agree():
    ch = bare_chain()
    ws = TRIPLE_EPS + TRIPLE
    out = ch.quasiparticles.block_shared(TRIPLE_EPS, ws) - TRIPLE_EPS
    assert ch.resolved.qp_blocks[1] == ((1, 2),)
    assert np.ptp(out[[1, 2]]) < ROUND_OFF
    assert abs(out[3] - TRIPLE[3]) < ROUND_OFF
    rej = ch.conventions.qp_block_record['rejected']
    assert [r['members'] for r in rej] == [[1, 2, 3]]
    assert rej[0]['split_into'] == [[1, 2]]


def test_the_quasiparticle_hamiltonian_rejects_every_block():
    ch = bare_chain()
    ch.qp_hamiltonian = True
    ws = SYNTHETIC_EPS + DEGENERATE
    out = ch.quasiparticles.block_shared(SYNTHETIC_EPS, ws)
    assert np.abs(out - ws).max() < ROUND_OFF
    rej = ch.conventions.qp_block_record['rejected']
    assert [r['members'] for r in rej] == [[1, 2]]
    assert 'qp_hamiltonian' in rej[0]['why']


@pytest.fixture(scope='module')
def near():
    ch = chain(NEAR_C3V)
    ch.excitation()
    return ch


def test_near_degenerate_members_carry_one_correction(near):
    assert near.degeneracy_tol == QP_DEGENERACY_TOL
    shift, blocks, fwd = corrections(near, near.mol0)
    assert len(blocks) == 3
    pos = {int(p): i for i, p in enumerate(near.resolved.qp_set)}
    kept = near.conventions.qp_block_record['kept']
    assert sorted(k['members'] for k in kept) == sorted(
        [int(p) for p in b] for b in blocks)
    assert near.conventions.qp_block_record['rejected'] == []
    for b in blocks:
        assert np.ptp(fwd.eps[b]) < QP_DEGENERACY_TOL
        own = np.array([near.resolved.qp_seeds[int(p)] for p in b]) - fwd.eps[b]
        assert np.ptp(own) < QP_BLOCK_CORRECTION_TOL
        mine = shift[[pos[int(p)] for p in b]]
        assert np.ptp(mine) < ROUND_OFF, f'block {b}: {mine}'
        roots = np.array([near.resolved.qp_seeds[int(p)] for p in b]) - fwd.eps[b]
        assert abs(mine[0] - roots.mean()) < ROUND_OFF


def test_the_record_names_every_block(near):
    """A block is never invisible: members, spreads, tolerances."""
    blocks = qp_bookkeeping(near, near.mol0)['blocks']
    assert blocks is near.conventions.qp_block_record
    assert blocks['energy_tol_Ha'] == QP_DEGENERACY_TOL
    assert blocks['correction_tol_Ha'] == QP_BLOCK_CORRECTION_TOL
    assert len(blocks['kept']) == len(near.resolved.qp_blocks[1]) == 3
    for k in blocks['kept']:
        assert k['energy_spread_Ha'] < QP_DEGENERACY_TOL
        assert k['correction_spread_Ha'] < QP_BLOCK_CORRECTION_TOL


def test_the_blocks_are_frozen_with_the_set(near):
    frozen = near.resolved.qp_blocks
    mol = displaced(near.mol0, 3, 0, SPLIT_STEP)
    shift, blocks, fwd = corrections(near, mol)
    assert near.resolved.qp_blocks == frozen
    pos = {int(p): i for i, p in enumerate(near.resolved.qp_set)}
    gaps = [np.ptp(fwd.eps[b]) for b in blocks]
    assert max(gaps) > QP_DEGENERACY_TOL, 'the step must split a pair'
    for b in blocks:
        assert np.ptp(shift[[pos[int(p)] for p in b]]) < ROUND_OFF


def test_the_force_differentiates_the_shared_correction():
    with correction_gate(WIDE_BLOCKS):
        ch = chain(DISTORTED, degeneracy_tol=WIDE_BLOCKS)
        grad = ch.excitation_gradient()[0]
    shift, blocks, _ = corrections(ch, ch.mol0)
    assert max(len(b) for b in blocks) == 3
    mol = ch.mol0
    for atom, axis in ((3, 1), (1, 2), (0, 2)):
        ep = ch.excitation(displaced(mol, atom, axis, FD_STEP))
        em = ch.excitation(displaced(mol, atom, axis, -FD_STEP))
        fd = (ep - em) / (2 * FD_STEP)
        assert abs(fd - grad[atom, axis]) < FD_TOL, (atom, axis, fd,
                                                     grad[atom, axis])


def test_close_but_not_degenerate_orbitals_keep_their_own_corrections():
    """Formaldehyde 12/13: no block at the default, each correction its own
    root's, and each within the pole model's error of the continuation; an
    energy tolerance that joins them is turned down by the correction gate,
    and only with the gate off do both end half their split off."""
    window = list(range(4, 18))
    out = {}
    for tol, gate in ((QP_DEGENERACY_TOL, QP_BLOCK_CORRECTION_TOL),
                      (JOINING_TOL, QP_BLOCK_CORRECTION_TOL),
                      (JOINING_TOL, NO_GATE)):
        with correction_gate(gate):
            ch = chain(FORMALDEHYDE, qp_window=window, degeneracy_tol=tol)
            ch.excitation()
        mol, mf = ch.mean_field(ch.mol0)
        fwd = ch.shared_forward(mol, mf)
        out[tol, gate] = (ch, fwd)
    ch, fwd = out[QP_DEGENERACY_TOL, QP_BLOCK_CORRECTION_TOL]
    pair = list(CLOSE_PAIR)
    assert np.ptp(fwd.eps[pair]) < JOINING_TOL
    assert ch.resolved.qp_blocks[1] == ()
    own = fwd.eps_qp[pair] - fwd.eps[pair]
    roots = np.array([ch.resolved.qp_seeds[p] for p in pair]) - fwd.eps[pair]
    assert np.abs(own - roots).max() < ROUND_OFF
    assert np.ptp(own) * HARTREE_TO_MEV > OWN_SPLIT_MEV
    xc = ch.xc_correction(fwd.mf, pair, None)
    ac, _ = ac_quasiparticle_shifts(fwd.mf, fwd.mol, ch.nocc, pair,
                                    factors=(fwd.x_mo, fwd.d), xc_diagonal=xc,
                                    numerics=ch.settings)
    ac = np.array([ac[p] for p in pair])
    assert np.abs(own - ac).max() * HARTREE_TO_MEV < SOP_AC_TOL_MEV
    gated, fwd_g = out[JOINING_TOL, QP_BLOCK_CORRECTION_TOL]
    assert gated.resolved.qp_blocks[1] == ()
    assert [r['members'] for r in gated.conventions.qp_block_record['rejected']] == [pair]
    assert np.abs(fwd_g.eps_qp[pair] - [gated.resolved.qp_seeds[p] for p in pair]
                  ).max() < ROUND_OFF
    joined, fwd_j = out[JOINING_TOL, NO_GATE]
    assert len(joined.resolved.qp_blocks[1]) == 1
    shared = fwd_j.eps_qp[pair] - fwd_j.eps[pair]
    assert np.abs(shared - ac).min() * HARTREE_TO_MEV > OWN_SPLIT_MEV / 2 - \
        SOP_AC_TOL_MEV
