"""The grid Hellmann-Feynman adjoint of a BSE root (`bse_adjoint='grid'`,
`LinearResponse.isdf_bse_adjoint`) against the explicit one (`bse_backward`
on the three-index blocks of `bse_cache`), on RHF/cc-pVDZ at 148 points per
atom:

  * THE KERNEL, per adjoint on the chain's own roots and vectors: eps_bar
    exact, X_bar, D_bar and the symmetric part of W_bar within ANCHOR_K times
    the explicit route's own reassociation response (the explicit route
    re-run with the occupied and virtual orbitals each reversed, so every
    bra loop runs backwards, its virtual-virtual tiles one grid row wide and
    W symmetrized), for dOmega and the interstate element, singlet and triplet,
    full and Tamm-Dancoff, HF and PBE0 references, water and ethylene;
  * THE FORCE: the composed excitation gradient and the interstate element
    with 'grid' against the default route within the routes test's anchored
    bar (`COMPOSED_GRAD_K` times the one-thread repeat of the serial force,
    floored at `COMPOSED_GRAD_FLOOR`), and the total gradient against a
    five-point finite difference of E_0 + Omega on water beside the default
    route's own miss;
  * THE CACHE: on the explicit route built once, at the first reverse call
    off a forward pass (an energy builds none, two roots off one pinned
    forward pass one); on the grid route never;
  * THE MEMORY SCAN: every frame under src/ traced line by line through a
    whole force (excitation gradient and interstate element): on the grid
    route no array of the shape of a three-index block, (naux, nocc, nvir) or
    (naux, nocc, nocc) in any order or (naux, nocc*nvir), exists at any line;
    the default route trips the same scan in `b_block` (which `bse_cache`
    calls) and `bse_backward`;
  * the setting is carried by the constructor, `refreeze`, the dispatcher and
    its record, and refused where it means nothing;
  * benzene: the kernel's agreement and its time against the explicit
    route's, cache included.

Measured on the laptop at 2 threads, the kernel in its fixed (tile, tile)
blocks: within 4.5e-16 to 3.3e-15 of the explicit route, at most 6.8
anchors (water's singlet interstate X_bar); the grid force 0.34 of the
anchored bar (5.3e-8 Ha/Bohr) from the default, the interstate element
0.17, the Davidson route's 0.21; the five-point misses 8.5e-8 (grid) and
8.8e-8 (default) relative; benzene's adjoint 4.0 s with its cache against
0.31 s. The kernel over ranks -- the same bits at every rank count, no
whole-grid array on any rank -- is gated in
tests/test_bse_grid_adjoint_ranks.py.

The residue backend of the scanned force is 'sop': the quasiparticle set
solve's 'explicit' residues build C_ov, the same (naux, nocc*nvir) block, in
a stage this setting does not reach.

SHOWN TO FAIL, then restored and byte-compared (`cmp`): `_casida` building
the cache in the forward pass on the grid route too, as the default route
once did. The scan failed on water, 24 block-shaped arrays seen in the
`b_block`, `_casida_args` and `_casida_seeds` frames, and so did the cache
count, an energy having built it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.declaration import Excitation, GroundState
from src.SingleReference.LinearResponse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_interstate_backward)
from src.gradients import excited_state
from src.gradients.bse_isdf import (bse_backward, bse_cache,
                                    interstate_backward)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from src.properties.surfaces import potential_energy_surface
from tests.test_chain_sliced_factors import (BASIS, H2O, chain_scf,
                                             reachable_arrays)
from tests.test_mpi_routes import (COMPOSED_GRAD_FLOOR, COMPOSED_GRAD_K,
                                   one_thread_scatter)

ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
BENZENE = ('C 0 1.3915 0; C 1.2051 0.6958 0; C 1.2051 -0.6958 0; '
           'C 0 -1.3915 0; C -1.2051 -0.6958 0; C -1.2051 0.6958 0; '
           'H 0 2.4715 0; H 2.1404 1.2358 0; H 2.1404 -1.2358 0; '
           'H 0 -2.4715 0; H -2.1404 -1.2358 0; H -2.1404 1.2358 0')
MOLECULES = {'water': H2O, 'ethylene': ETHYLENE, 'benzene': BENZENE}
#: How many times the explicit route's own reassociation response the grid
#: route may sit from it: the two share no summation order at all, where the
#: anchor reorders two of the explicit route's sums.
ANCHOR_K = 10
#: The five-point stencil's step (Bohr) and the relative miss the excited-state
#: surface's own gate allows (tests/test_excited_state.py).
FD_STEP = 1e-4
FD_REL = 1e-6
#: The package whose frames the scan traces.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'src') + os.sep


def pbe0_scf(mol):
    """A PBE0 reference converged for gradient work."""
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def molecule(name):
    return gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)


def rel(a, b):
    """max |a - b| relative to max |a|."""
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


def symmetric(w):
    return 0.5 * (w + w.T)


def casida_inputs(chain):
    """(X_mo, D, eps_qp, W_aux, nocc, cache, Xn, Yn) at the reference, the
    cache built as the explicit route builds it."""
    om, pieces = chain._forward(chain.mol0, chain.mf0)
    x, d, eq, w, no, cache, xn, yn = chain._casida_args(pieces)
    if not cache:
        cache = bse_cache(x, d, eq, w, no, spin=chain.spin,
                          bse_tda=chain.bse_tda)
    return x, d, eq, w, no, cache, xn, yn


def distances(ref, got):
    """(eps_bar, X_bar, D_bar, sym W_bar) relative distances."""
    return (rel(ref[0], got[0]), rel(ref[1], got[1]), rel(ref[2], got[2]),
            rel(symmetric(ref[3]), symmetric(got[3])))


class BlockScan:
    """Every array shaped like a three-index block that any frame under src/
    holds at any line of a call: its locals, and the containers they hold."""

    def __init__(self, naux, nocc, nvir):
        self.three = {tuple(sorted((naux, nocc, nvir))),
                      tuple(sorted((naux, nocc, nocc)))}
        self.two = tuple(sorted((naux, nocc * nvir)))
        self.found = {}

    def is_block(self, a):
        shape = tuple(sorted(a.shape))
        return ((a.ndim == 3 and shape in self.three)
                or (a.ndim == 2 and shape == self.two))

    def visit(self, frame, value, depth=0):
        if isinstance(value, np.ndarray):
            if self.is_block(value):
                key = (frame.f_code.co_name,
                       os.path.basename(frame.f_code.co_filename), value.shape)
                self.found.setdefault(key, frame.f_lineno)
        elif depth < 3 and isinstance(value, dict):
            for v in value.values():
                self.visit(frame, v, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for v in value:
                self.visit(frame, v, depth + 1)

    def local(self, frame, event, arg):
        if event in ('line', 'return'):
            for value in frame.f_locals.values():
                self.visit(frame, value)
        return self.local

    def enter(self, frame, event, arg):
        return (self.local if frame.f_code.co_filename.startswith(SRC)
                else None)

    def run(self, fn, *args, **kwargs):
        held = sys.gettrace()
        sys.settrace(self.enter)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.settrace(held)


def reassociated_explicit(n, x, d, eq, w, no, cache, xn, yn, bra=None):
    """The explicit adjoint with the occupied and the virtual orbitals each in
    reverse order -- every bra loop and every orbital sum run backwards --
    its virtual-virtual term tiled one grid row at a time and W symmetrized:
    one reordering of its own sums, handed back in the original order."""
    nmo = x.shape[1]
    order = np.r_[np.arange(no)[::-1], np.arange(no, nmo)[::-1]]
    nv = nmo - no

    def flipped(v):
        return v.reshape(no, nv, -1)[::-1, ::-1].reshape(no * nv, -1)

    xr, eqr = np.ascontiguousarray(x[:, order]), np.asarray(eq)[order]
    spin = 'singlet' if cache['kappa'] else 'triplet'
    cr = bse_cache(xr, d, eqr, w, no, spin=spin,
                   bse_tda=cache['B_vo'] is None)
    one_row = 3 * x.shape[0] * 8 / 1e9
    e, xb, db, wb = bse_backward(n, xr, d, eqr, symmetric(w), no, cr,
                                 flipped(xn), flipped(yn), tile_gb=one_row,
                                 bra=bra)
    back = np.argsort(order)
    return e[back], xb[:, back], db, wb


# ------------------------------------------------------------- the setting
def test_the_setting_is_carried_and_refused():
    """Constructor, refreeze and the dispatcher carry `bse_adjoint`; the
    record's numerics name it; an unknown realization and a dense row, which
    has no BSE adjoint of this kind, refuse it."""
    mol = molecule('water')
    mf = chain_scf(mol)
    chain = ExcitedStateChain(mol, chain_scf, mf=mf, bse_adjoint='grid')
    assert chain.refreeze(mol).bse_adjoint == 'grid'
    assert ExcitedStateChain(mol, chain_scf, mf=mf).bse_adjoint == 'explicit'
    with pytest.raises(ValueError, match='bse_adjoint'):
        ExcitedStateChain(mol, chain_scf, mf=mf, bse_adjoint='blocks')
    ground = GroundState('rpa', 'hf')
    surface = potential_energy_surface(mol, chain_scf, ground_state=ground,
                                       excitation=Excitation('singlet'),
                                       bse_adjoint='grid')
    assert surface.excited.bse_adjoint == 'grid'
    assert surface.numerics['bse_adjoint'] == 'grid'
    assert surface.refreeze(mol).excited.bse_adjoint == 'grid'
    with pytest.raises(TypeError, match='bse_adjoint'):
        potential_energy_surface(mol, chain_scf, ground_state=ground,
                                 excitation=Excitation('singlet'),
                                 chi0='dense-qb', factorization='four-index',
                                 bse_adjoint='grid')


def test_the_cache_is_built_once_at_the_first_reverse_call(monkeypatch):
    """An energy builds no block on either route; two roots reversed off one
    pinned forward pass build the explicit route's cache once, and the grid
    route never builds it."""
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return bse_cache(*args, **kwargs)

    monkeypatch.setattr(excited_state, 'bse_cache', counted)
    mol = molecule('water')
    mf = chain_scf(mol)
    for adjoint, builds in (('explicit', 1), ('grid', 0)):
        calls.clear()
        chain = ExcitedStateChain(mol, chain_scf, mf=mf, solver='davidson',
                                  bse_adjoint=adjoint)
        chain.energy()
        assert calls == [], f'{adjoint}: an energy built the cache'
        StateManifold(chain, states=(0, 1)).gradients()
        assert len(calls) == builds, (adjoint, len(calls))


# -------------------------------------------------------------- the kernel
KERNEL_CASES = [
    # (molecule, reference, spin, tda, solver)
    ('water', 'hf', 'singlet', False, 'davidson'),
    ('water', 'hf', 'triplet', True, 'dense'),
    ('water', 'hf', 'singlet', True, 'dense'),
    ('water', 'pbe0', 'singlet', False, 'davidson'),
    ('ethylene', 'hf', 'singlet', False, 'davidson'),
    ('ethylene', 'hf', 'triplet', True, 'dense'),
]


@pytest.mark.parametrize('name,reference,spin,tda,solver', KERNEL_CASES)
def test_the_grid_adjoint_is_the_explicit_one(name, reference, spin, tda,
                                              solver):
    """eps_bar exact; X_bar, D_bar and sym(W_bar) within ANCHOR_K times the
    explicit route's own reassociation response, for dOmega_0 and for the
    one-sided and symmetrized interstate elements between roots 0 and 1."""
    mol = molecule(name)
    factory = chain_scf if reference == 'hf' else pbe0_scf
    chain = ExcitedStateChain(mol, factory, mf=factory(mol), spin=spin,
                              bse_tda=tda, solver=solver)
    args = casida_inputs(chain)
    x, d, eq, w, no, cache, xn, yn = args
    kw = dict(spin=spin, bse_tda=tda)
    for label, bra in (('dOmega', None), ('<1|dH|0>', 1)):
        ref = bse_backward(0, *args, bra=bra)
        anchor = distances(ref, reassociated_explicit(0, *args, bra=bra))
        got = distances(ref, isdf_bse_backward(0, x, d, eq, w, no, xn, yn,
                                               bra=bra, **kw))
        print(f'[info] {name}/{reference} {spin} tda={tda} {label}: '
              f'X {got[1]:.1e} D {got[2]:.1e} Wsym {got[3]:.1e} '
              f'(anchor {anchor[1]:.1e} {anchor[2]:.1e} {anchor[3]:.1e})')
        assert got[0] == 0.0, got
        for g, a in zip(got[1:], anchor[1:]):
            assert g < ANCHOR_K * a, (label, got, anchor)
    ref = interstate_backward(0, 1, *args)
    anchor = distances(ref, reassociated_explicit(0, *args, bra=1))
    got = distances(ref, isdf_interstate_backward(0, 1, x, d, eq, w, no, xn,
                                                  yn, **kw))
    print(f'[info] {name}/{reference} {spin} tda={tda} symmetrized '
          f'interstate: X {got[1]:.1e} D {got[2]:.1e} Wsym {got[3]:.1e}')
    assert got[0] == 0.0
    for g, a in zip(got[1:], anchor[1:]):
        assert g < ANCHOR_K * a, ('interstate', got, anchor)


def test_benzene_agreement_and_timing():
    """The kernel on benzene against the explicit route, and both timed: the
    explicit one with the cache it needs, the grid one alone."""
    mol = molecule('benzene')
    chain = ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol),
                              solver='davidson')
    om, pieces = chain._forward(mol, chain.mf0)
    x, d, eq, w, no, _, xn, yn = chain._casida_args(pieces)
    t0 = time.perf_counter()
    cache = bse_cache(x, d, eq, w, no)
    ref = bse_backward(0, x, d, eq, w, no, cache, xn, yn)
    t1 = time.perf_counter()
    got = isdf_bse_backward(0, x, d, eq, w, no, xn, yn)
    t2 = time.perf_counter()
    dist = distances(ref, got)
    anchor = distances(ref, reassociated_explicit(0, x, d, eq, w, no, cache,
                                                  xn, yn))
    print(f'[info] benzene M {x.shape[0]} naux {d.shape[1]} nocc {no}: '
          f'X {dist[1]:.1e} D {dist[2]:.1e} Wsym {dist[3]:.1e} (anchor '
          f'{anchor[1]:.1e} {anchor[2]:.1e} {anchor[3]:.1e}); explicit '
          f'{t1 - t0:.2f} s with its cache, grid {t2 - t1:.2f} s')
    assert dist[0] == 0.0
    for g, a in zip(dist[1:], anchor[1:]):
        assert g < ANCHOR_K * a


# --------------------------------------------------------------- the force
def five_point_gradient(chain, mol):
    """d(E_0 + Omega)/dR by the five-point stencil, every component."""
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for c in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                step = np.zeros((mol.natm, 3))
                step[ia, c] = k * FD_STEP
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + step, unit='Bohr')
                m.build(False, False)
                v.append(chain.energy(m)[0])
            fd[ia, c] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
    return fd


def test_the_grid_force_is_the_default_force():
    """Water, the default Casida solver: the excitation gradient and the
    interstate element within the routes test's anchored bar of the default
    route's, the total gradient against a five-point finite difference of
    E_0 + Omega beside the default route's miss; the Davidson route's forces
    beside it."""
    mol = molecule('water')
    mf = chain_scf(mol)
    default = ExcitedStateChain(mol, chain_scf, mf=mf)
    grid = ExcitedStateChain(mol, chain_scf, mf=mf, bse_adjoint='grid')
    g_def = default.excitation_gradient()[0]
    g_grid = grid.excitation_gradient()[0]
    bar = max(COMPOSED_GRAD_FLOOR,
              COMPOSED_GRAD_K * one_thread_scatter(mol, mf, g_def))
    d_ex = np.abs(g_grid - g_def).max()
    c_def = default.interstate_gradient(0, 1)[0]
    c_grid = grid.interstate_gradient(0, 1)[0]
    d_in = np.abs(c_grid - c_def).max()
    print(f'[info] water excitation gradient |d| {d_ex:.2e} = '
          f'{d_ex / bar:.3f} of the anchored bar {bar:.2e} Ha/Bohr; '
          f'interstate element |d| {d_in:.2e} = {d_in / bar:.3f} of it')
    assert d_ex < bar and d_in < bar
    fd = five_point_gradient(default, mol)
    scale = np.abs(fd).max()
    miss_def = np.abs(default.total_gradient()[0] - fd).max() / scale
    miss_grid = np.abs(grid.total_gradient()[0] - fd).max() / scale
    print(f'[info] water five-point miss (relative): grid {miss_grid:.2e}, '
          f'default {miss_def:.2e}')
    assert miss_grid < FD_REL
    davidson = {adjoint: ExcitedStateChain(mol, chain_scf, mf=mf,
                                           solver='davidson',
                                           bse_adjoint=adjoint)
                for adjoint in ('explicit', 'grid')}
    d_dav = np.abs(davidson['grid'].excitation_gradient()[0]
                   - davidson['explicit'].excitation_gradient()[0]).max()
    i_dav = np.abs(davidson['grid'].interstate_gradient(0, 1)[0]
                   - davidson['explicit'].interstate_gradient(0, 1)[0]).max()
    print(f'[info] water Davidson route: excitation |d| {d_dav:.2e}, '
          f'interstate |d| {i_dav:.2e} Ha/Bohr')
    assert d_dav < bar and i_dav < bar


# ---------------------------------------------------------- the memory scan
def scanned_force(adjoint, residues='sop'):
    """(BlockScan, chain) of one excitation gradient and one interstate
    element on water, every src/ frame traced."""
    mol = molecule('water')
    mf = chain_scf(mol)
    chain = ExcitedStateChain(mol, chain_scf, mf=mf, solver='davidson',
                              residue_route=residues, bse_adjoint=adjoint)
    naux = chain.naux
    nocc = chain.nocc
    scan = BlockScan(naux, nocc, mf.mo_coeff.shape[1] - nocc)
    scan.run(chain.excitation_gradient)
    scan.run(chain.interstate_gradient, 0, 1)
    for a in reachable_arrays(chain):
        if scan.is_block(a):
            scan.found.setdefault(('reachable', 'chain', a.shape), 0)
    return scan, chain


def test_no_three_index_block_exists_in_a_grid_force():
    """Grid route: no three-index-shaped array at any line of any src/ frame
    of a force, nor reachable from the chain after it. The default route
    trips the same scan, in the blocks' own builders and consumers."""
    scan, _ = scanned_force('grid')
    print(f'[info] grid route: {len(scan.found)} block-shaped arrays seen')
    assert scan.found == {}, sorted(scan.found)
    scan, _ = scanned_force('explicit')
    where = sorted({key[0] for key in scan.found})
    print(f'[info] explicit route: seen in {where}')
    assert {'b_block', 'bse_backward'} <= set(where), where


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
