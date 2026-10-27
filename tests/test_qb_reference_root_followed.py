"""The dense QB reference: the largest-weight root is decided at the reference
geometry and followed from there, and a near-degenerate block of the set
carries one shared correction.

`DenseBSESurface` must not take each orbital's root of largest pole strength
afresh at every geometry. For a continuum virtual no
root holds most of the weight, so that root changes branch between
geometries 1e-3 Bohr apart (formaldehyde/cc-pVDZ T1, orbital 22, Z 0.19:
+20 mHa), the energy steps and the saddle check's Hessian comes out
asymmetric. The largest-weight roots are decided at the surface's own
geometry (and again at each refreeze) and every geometry follows each from
its frozen root through the interval between the poles that bracket it
(`QPqb.solve_diag` with a seed, the one root rule of `find_qp_root`).

Orbitals closer than QP_DEGENERACY_TOL at the reference whose own
corrections agree within QP_BLOCK_CORRECTION_TOL form blocks frozen with
the set; each member keeps its own eps_p and takes the mean of
the block's eps^QP_q - eps_q, and the force differentiates exactly that.

NH3/6-31g, HF, dense QB on every orbital, T1. Near C3v its E pairs are split
by 6e-6 to 3e-5 Ha, so four blocks form; distorted further, none does.
Orbital 13, 4.7e-4 Ha from 12, is not degenerate and stays out of them.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (QP_BLOCK_CORRECTION_TOL,
                                         QP_DEGENERACY_TOL)
from src.gradients.gwbse import dense_surfaces
from src.gradients.gwbse.dense_surfaces import DenseBSESurface
from src.gradients.gwbse.diagnostics import qp_bookkeeping
from tests.blas_bar import bar_with_drift

NEAR_C3V = ('N 0 0 0.1173; H 0 0.9377 -0.2738; H 0.8121 -0.4689 -0.2738; '
            'H -0.8121 -0.4689 -0.2738')
DISTORTED = ('N 0 0 0.1173; H 0 0.9377 -0.2738; H 0.8521 -0.4689 -0.2738; '
             'H -0.8121 -0.4689 -0.2938')
#: Bohr, and twice it: small enough that the avoided crossings' curvature
#: leaves the h^2 truncation below FD_TOL. Orbital 13 sits 4.7e-4 Ha above the
#: block (11, 12) and keeps its own correction, so the surface bends on that
#: pair's avoided crossing: at 1e-4 Bohr the truncation is 8e-6 Ha/Bohr.
FD_STEP = 1e-5
FD_TOL = 1e-6
#: Ha: a root resolved by two solvers, each to QB_QP_ROOT_TOL.
ROOT_TOL = 1e-10
#: Ha: (eps_p + c) - eps_p is c to this.
ROUND_OFF = 1e-14
#: The same bar on another BLAS than the pin files': ROUND_OFF plus its drift.
ROUND_BAR = bar_with_drift(ROUND_OFF)


def surface(atom):
    mol = gto.M(atom=atom, basis='6-31g', verbose=0)
    return mol, DenseBSESurface(mol, spin='triplet',
                                qp_orbs=list(range(mol.nao)), filter_z=False)


def displaced(mol, atom, axis, step):
    out = mol.copy()
    coords = out.atom_coords().copy()
    coords[atom, axis] += step
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


@pytest.fixture
def spy(monkeypatch):
    """Every BSEqb the surface builds, with the seeds it followed."""
    calls = []

    class Spy(dense_surfaces.BSEqb):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            calls.append((kw.get('seeds'), self))

    monkeypatch.setattr(dense_surfaces, 'BSEqb', Spy)
    return calls


def test_the_largest_weight_root_is_decided_at_the_reference_only(spy):
    mol, surf = surface(DISTORTED)
    assert surf.qp_blocks == ()
    probe_seeds, probe = spy[0]
    # the set and its roots are decided once, unseeded, at the reference ...
    assert probe_seeds is None
    assert surf.seeds == probe.qp_roots
    for p in surf.qp_set:
        assert probe.qp_roots[p] == probe.qp.quasiparticle_root(p)[0]
    # ... and every evaluation follows them, the reference included
    surf.total_energy(mol)
    seeds, at_ref = spy[-1]
    assert seeds == surf.seeds
    for p in surf.qp_set:
        assert abs(at_ref.qp_roots[p] - surf.seeds[p]) < ROOT_TOL, p
    surf.total_energy(displaced(mol, 1, 2, 1e-3))
    seeds, there = spy[-1]
    assert seeds == surf.seeds
    for p in surf.qp_set:
        w, _ = there.qp.solve_diag(p, w0=surf.seeds[p])
        assert w == there.qp_roots[p]


def test_a_refreeze_decides_the_roots_again_at_its_geometry(spy):
    mol, surf = surface(DISTORTED)
    moved = displaced(mol, 1, 2, 2e-2)
    before = len(spy)
    again = surf.refreeze(moved)
    probe_seeds, probe = spy[before]
    assert probe_seeds is None
    assert again.seeds == probe.qp_roots
    assert set(again.seeds) == set(again.qp_set)
    again.total_energy(moved)
    seeds, at_ref = spy[-1]
    assert seeds == again.seeds
    for p in again.qp_set:
        assert abs(at_ref.qp_roots[p] - again.seeds[p]) < ROOT_TOL, p


def test_near_degenerate_orbitals_share_one_correction():
    mol, surf = surface(NEAR_C3V)
    eps = surf.scf_factory(mol).mo_energy
    assert len(surf.qp_blocks) == 4
    kept = surf.qp_block_record['kept']
    assert [tuple(k['members']) for k in kept] == list(surf.qp_blocks)
    assert qp_bookkeeping(surf, mol)['blocks'] is surf.qp_block_record
    for b in surf.qp_blocks:
        assert np.ptp(eps[list(b)]) < QP_DEGENERACY_TOL
        own = np.array([surf.seeds[p] for p in b]) - eps[list(b)]
        assert np.ptp(own) < QP_BLOCK_CORRECTION_TOL
    b, _, _, _, _ = surf._solve(mol, surf.scf_factory(mol))
    for block in surf.qp_blocks:
        block = list(block)
        shift = b.eps_qp[block] - eps[block]
        assert np.ptp(shift) < ROUND_BAR, f'block {block}: {shift}'
        raw = np.array([b.qp_roots[p] for p in block]) - eps[block]
        assert abs(shift[0] - raw.mean()) < ROUND_BAR
        # the members' own corrections differ by far more than that
        assert np.ptp(raw) > 1e3 * ROUND_OFF


def test_the_force_is_the_derivative_of_the_shared_correction():
    mol, surf = surface(NEAR_C3V)
    grad = surf.total_gradient(mol)[0]
    for atom, axis in ((3, 1), (1, 2)):
        err = []
        for h in (2 * FD_STEP, FD_STEP):
            ep = surf.total_energy(displaced(mol, atom, axis, h))
            em = surf.total_energy(displaced(mol, atom, axis, -h))
            err.append(abs((ep - em) / (2 * h) - grad[atom, axis]))
        assert max(err) < FD_TOL, (atom, axis, err)
