"""The grid Hellmann-Feynman adjoint of a BSE root (`bse_adjoint='grid'`,
`LinearResponse.isdf_bse_adjoint`) against the explicit one (`bse_backward`
on the three-index blocks of `bse_cache`), on RHF/cc-pVDZ at 148 points per
atom:

  * THE KERNEL, per adjoint on the chain's own roots and vectors: eps_bar
    exact, X_bar, D_bar and the symmetric part of W_bar within a DERIVED
    rounding bound of the explicit route's. Both routes evaluate one
    polynomial of the factors, W and the Casida vectors by products and sums
    alone, so each lies within gamma_K of the sum of its own terms' absolute
    values (Higham, Accuracy and Stability of Numerical Algorithms, 2002,
    Sec. 3.1), K the most roundings any term passes through on either route
    (`roundings`); that sum is each route's own code run on the inputs'
    absolute values with every term made positive (`magnitudes`), and the
    explicit route's reading of W unsymmetrized adds the terms of W's
    antisymmetric part. No measured response enters. For dOmega and the
    interstate element, singlet and triplet, full and Tamm-Dancoff, HF and
    PBE0 references, water and ethylene, and benzene; shown to fail with the
    grid route's swap term dropped and with D_bar's largest element moved
    1e-10 relative (`test_the_bound_fails_a_wrong_kernel`);
  * THE FORCE, the composed excitation gradient and the interstate element
    on water, the default and the Davidson solver, both routes on one mean
    field with pyscf's OpenMP on one thread: the roots bitwise (one forward
    pass), the default chain's own fold of the grid route's seeds the grid
    force bitwise (one fold), so the routes differ in their seeds alone, and
    those seeds within the kernel's derived bound at that very reverse call.
    The two forces are printed, not compared: the fold's exact image of the
    seed difference is ~1e-16 Ha/Bohr, but the fold's own rounding, through
    a fit adjoint of Gram condition 2e8, moves its result by ~1e-8 Ha/Bohr
    when the last bits of its input change -- a bar on the forces would gate
    that draw. And the total gradient against a five-point finite difference
    of E_0 + Omega on water beside the default route's own miss;
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

Measured on a two-thread workstation: the kernel at most 0.006 of its
derived bound on every case and benzene's 0.001 (on water the bound is
4e-13 of D_bar's largest element, so moving that element 1e-10 relative is
238 bounds, dropping the swap term 1.9e11); at the chains' reverse calls the
seeds at most 0.006 of it, the forces 6.8e-9 to 1.8e-8 Ha/Bohr apart while
the fold's image of their seed difference is 2.9e-16 to 9.5e-16; the
five-point misses 8.5e-8 (grid) and 8.8e-8 (default) relative; benzene's
adjoint 3.6 s with its cache against 0.29 s. The one-fold gate fails with the
grid chain's W_bar moved 1e-12 relative inside its fold, the forces then
still 9.9e-9 apart, which no bar on them could tell from the clean run. The
kernel over ranks -- the same bits at every rank count, no whole-grid array
on any rank -- is gated in tests/test_bse_grid_adjoint_ranks.py.

The residue backend of the scanned force is 'sop': the quasiparticle set
solve's 'explicit' residues build C_ov, the same (naux, nocc*nvir) block, in
a stage this setting does not reach.

SHOWN TO FAIL, then restored and byte-compared (`cmp`): `_casida` building
the cache in the forward pass on the grid route too, as the default route
once did. The scan failed on water, 24 block-shaped arrays seen in the
`b_block`, `_casida_args` and `_casida_seeds` frames, and so did the cache
count, an energy having built it.
"""
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf

from src.Base.constants import BSE_ADJOINT_TILE_ROWS, KAPPA
from src.Base.declaration import Excitation, GroundState
from src.SingleReference.LinearResponse import isdf_bse_adjoint
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

ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
BENZENE = ('C 0 1.3915 0; C 1.2051 0.6958 0; C 1.2051 -0.6958 0; '
           'C 0 -1.3915 0; C -1.2051 -0.6958 0; C -1.2051 0.6958 0; '
           'H 0 2.4715 0; H 2.1404 1.2358 0; H 2.1404 -1.2358 0; '
           'H 0 -2.4715 0; H -2.1404 -1.2358 0; H -2.1404 1.2358 0')
MOLECULES = {'water': H2O, 'ethylene': ETHYLENE, 'benzene': BENZENE}
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


def roundings(npts, naux, nmo, nocc, tile_rows=BSE_ADJOINT_TILE_ROWS):
    """The most roundings one term of a seed passes through on either route.

    A product adds one to its factors' counts, a sum over n terms at most
    n - 1 in any order or blocking. The grid route contracts the grid index
    twice (Zt's columns and the column pass, or D^T (S D)), the auxiliary
    index twice (D W D^T, S D W) and the orbitals inside P_T and against T,
    and adds the column tiles' partials; the explicit route contracts the
    grid twice through its B blocks, the auxiliary index twice and the pairs,
    n_ov in the bare kernel and nocc^2 in the direct W_bar term. The last 64
    cover every elementwise product and every accumulation into an output, a
    dozen on either route.
    """
    nvir = nmo - nocc
    return (2 * npts + 2 * naux + 2 * nmo + nocc * nvir + nocc * nocc
            + 2 * math.ceil(npts / tile_rows) + 64)


def gamma(k):
    """gamma_k = k u / (1 - k u), u the unit roundoff: the relative error of
    k compounded roundings (Higham 2002, Lemma 3.1)."""
    ku = k * np.finfo(float).eps / 2
    return ku / (1 - ku)


def magnitudes(route, n, x, d, eq, w, no, xn, yn, spin, tda, bra=None,
               symmetrized=False, tile_rows=BSE_ADJOINT_TILE_ROWS):
    """(X_bar, D_bar, W_bar) of `route` ('grid' or 'explicit') run on the
    absolute values of its inputs with every term made positive: the sum of
    the absolute values of the terms that route adds.

    Every screened term carries -omega_bar and the triplet's zero kappa drops
    the bare kernel, so omega_bar = -1 turns them all positive; the bare
    kernel carries +kappa omega_bar, and W = 0 (with no B_oo and B_vo on the
    explicit route) leaves it alone. W_bar has no bare term. `symmetrized`:
    the explicit route's interstate element, both orderings averaged;
    `tile_rows`: the grid route's tile edge.
    """
    ax, ad, aw, axn, ayn = (np.abs(a) for a in (x, d, w, xn, yn))
    if route == 'grid':
        def run(kind, w_abs, omega_bar):
            return isdf_bse_backward(n, ax, ad, eq, w_abs, no, axn, ayn,
                                     spin=kind, bse_tda=tda,
                                     omega_bar=omega_bar, bra=bra,
                                     tile_rows=tile_rows)
    else:
        screened = bse_cache(ax, ad, eq, aw, no, spin='triplet', bse_tda=tda)
        bare = dict(screened, kappa=KAPPA[spin], B_vo=None,
                    B_oo=np.zeros_like(screened['B_oo']))

        def run(kind, w_abs, omega_bar):
            cache = screened if kind == 'triplet' else bare
            if symmetrized:
                return interstate_backward(bra, n, ax, ad, eq, w_abs, no,
                                           cache, axn, ayn,
                                           omega_bar=omega_bar)
            return bse_backward(n, ax, ad, eq, w_abs, no, cache, axn, ayn,
                                omega_bar=omega_bar, bra=bra)
    _, x_abs, d_abs, w_bar_abs = run('triplet', aw, -1.0)
    if KAPPA[spin]:
        _, x_bare, d_bare, _ = run(spin, np.zeros_like(aw), 1.0)
        x_abs, d_abs = x_abs + x_bare, d_abs + d_bare
    return x_abs, d_abs, w_bar_abs


def bound_ratios(ref, got, inputs, spin, tda, n, bra=None, symmetrized=False):
    """(X_bar, D_bar, sym W_bar): the worst |grid - explicit| over the derived
    bound, element by element; `ref` the explicit route's seeds of root `n`
    (and `bra`), `got` the grid route's, `inputs` (X_mo, D, eps_qp, W_aux,
    nocc, Xn, Yn) the arrays both read.

    Each route lies within gamma_K of its own magnitudes from the one exact
    value, K = `roundings` and two more for the symmetric parts formed; the
    magnitudes' own roundings are the 1 / (1 - gamma_K). The explicit route
    reads W as it is and the grid route its symmetric part, so the terms of
    the antisymmetric part, linear in it, are added in full.
    """
    x, d, eq, w, no, xn, yn = inputs
    g = gamma(roundings(x.shape[0], d.shape[1], x.shape[1], no) + 2)
    g = g / (1 - g)
    kw = dict(spin=spin, tda=tda, bra=bra)
    mag_g = magnitudes('grid', n, x, d, eq, symmetric(w), no, xn, yn, **kw)
    mag_e = magnitudes('explicit', n, x, d, eq, w, no, xn, yn,
                       symmetrized=symmetrized, **kw)
    anti = magnitudes('explicit', n, x, d, eq, 0.5 * (w - w.T), no, xn, yn,
                      spin='triplet', tda=tda, bra=bra,
                      symmetrized=symmetrized)
    pairs = ((ref[1], got[1], mag_e[0] + mag_g[0], anti[0]),
             (ref[2], got[2], mag_e[1] + mag_g[1], anti[1]),
             (symmetric(ref[3]), got[3], symmetric(mag_e[2]) + mag_g[2], 0.0))
    return [worst_ratio(a, b, g * mag + (1 + g) * asym)
            for a, b, mag, asym in pairs]


def grid_bound_ratios(ref, got, inputs, spin, tda, n, bra=None,
                      tiles=(BSE_ADJOINT_TILE_ROWS, BSE_ADJOINT_TILE_ROWS)):
    """(X_bar, D_bar, W_bar): the worst |got - ref| over the derived bound of
    two blockings of the grid route, `ref` in tiles[0] and `got` in
    tiles[1]: one polynomial, each within gamma_K of its own magnitudes, K
    counted at the smaller tile."""
    x, d, eq, w, no, xn, yn = inputs
    g = gamma(roundings(x.shape[0], d.shape[1], x.shape[1], no, min(tiles))
              + 2)
    g = g / (1 - g)
    mag_a, mag_b = (magnitudes('grid', n, x, d, eq, w, no, xn, yn, spin, tda,
                               bra=bra, tile_rows=t) for t in tiles)
    return [worst_ratio(a, b, g * (ma + mb))
            for a, b, ma, mb in zip(ref[1:], got[1:], mag_a, mag_b)]


def worst_ratio(a, b, bound):
    """max |a - b| / bound element by element; infinite where a zero bound
    meets a nonzero difference."""
    off = np.abs(a - b)
    if np.any(off[bound == 0] > 0):
        return np.inf
    return float((off / np.maximum(bound, np.finfo(float).tiny)).max())


@pytest.fixture
def pyscf_one_thread():
    """pyscf's OpenMP GEMM on one thread, where it adds its K partials in one
    order, so two chains on one mean field repeat each other's forward pass
    and fold bit for bit."""
    threads = lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(1)
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(threads)


def captured(chain):
    """Every call of the chain's Casida-level adjoint: its forward pieces,
    root, bra and copies of its seeds, which the fold consumes."""
    calls = []
    seeds_of = chain._casida_seeds

    def probed(pieces, n, m=None):
        out = seeds_of(pieces, n, m)
        calls.append(dict(pieces=pieces, n=n, m=m,
                          seeds=tuple(np.array(a, copy=True) for a in out)))
        return out

    chain._casida_seeds = probed
    return calls


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
    """eps_bar exact; X_bar, D_bar and sym(W_bar) within the derived rounding
    bound of the explicit route's, for dOmega_0 and for the one-sided and
    symmetrized interstate elements between roots 0 and 1."""
    mol = molecule(name)
    factory = chain_scf if reference == 'hf' else pbe0_scf
    chain = ExcitedStateChain(mol, factory, mf=factory(mol), spin=spin,
                              bse_tda=tda, solver=solver)
    x, d, eq, w, no, cache, xn, yn = casida_inputs(chain)
    inputs = (x, d, eq, w, no, xn, yn)
    kw = dict(spin=spin, bse_tda=tda)
    cases = [(label, 0, bra, False,
              bse_backward(0, x, d, eq, w, no, cache, xn, yn, bra=bra),
              isdf_bse_backward(0, x, d, eq, w, no, xn, yn, bra=bra, **kw))
             for label, bra in (('dOmega', None), ('<1|dH|0>', 1))]
    cases.append(('symmetrized interstate', 1, 0, True,
                  interstate_backward(0, 1, x, d, eq, w, no, cache, xn, yn),
                  isdf_interstate_backward(0, 1, x, d, eq, w, no, xn, yn,
                                           **kw)))
    for label, n, bra, sym, ref, got in cases:
        ratios = bound_ratios(ref, got, inputs, spin, tda, n, bra=bra,
                              symmetrized=sym)
        print(f'[info] {name}/{reference} {spin} tda={tda} {label}: X '
              f'{ratios[0]:.3f} D {ratios[1]:.3f} Wsym {ratios[2]:.3f} of '
              'the derived bound')
        assert np.array_equal(ref[0], got[0]), label
        assert max(ratios) <= 1, (label, ratios)


def test_the_bound_fails_a_wrong_kernel(monkeypatch):
    """The derived bound passes no wrong kernel on water: D_bar's largest
    element moved 1e-10 relative, and the grid route with its swap term
    dropped from S, each fail it."""
    mol = molecule('water')
    chain = ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol),
                              solver='davidson')
    x, d, eq, w, no, cache, xn, yn = casida_inputs(chain)
    inputs = (x, d, eq, w, no, xn, yn)
    ref = bse_backward(0, x, d, eq, w, no, cache, xn, yn)
    got = list(isdf_bse_backward(0, x, d, eq, w, no, xn, yn))
    clean = bound_ratios(ref, got, inputs, 'singlet', False, 0)
    d_bar = got[2].copy()
    d_bar[np.unravel_index(np.abs(d_bar).argmax(), d_bar.shape)] *= 1 + 1e-10
    moved = bound_ratios(ref, [got[0], got[1], d_bar, got[3]], inputs,
                         'singlet', False, 0)
    direct = tuple(t for t in isdf_bse_adjoint.S_TERMS[False]
                   if t[0] == t[2])
    monkeypatch.setitem(isdf_bse_adjoint.S_TERMS, False, direct)
    dropped = bound_ratios(ref, isdf_bse_backward(0, x, d, eq, w, no, xn, yn),
                           inputs, 'singlet', False, 0)
    print(f'[info] water dOmega, X / D / Wsym of the derived bound: clean '
          f'{clean[0]:.3f} / {clean[1]:.3f} / {clean[2]:.3f}; D_bar moved '
          f'1e-10: {moved[1]:.2f}; swap term dropped: {dropped[0]:.1e} / '
          f'{dropped[1]:.1e} / {dropped[2]:.1e}')
    assert max(clean) <= 1 and moved[1] > 1 and min(dropped[1:]) > 1


def test_benzene_agreement_and_timing():
    """The kernel on benzene against the explicit route within the derived
    bound, and both timed: the explicit one with the cache it needs, the
    grid one alone."""
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
    ratios = bound_ratios(ref, got, (x, d, eq, w, no, xn, yn), 'singlet',
                          False, 0)
    print(f'[info] benzene M {x.shape[0]} naux {d.shape[1]} nocc {no}: '
          f'X {ratios[0]:.3f} D {ratios[1]:.3f} Wsym {ratios[2]:.3f} of the '
          f'derived bound; explicit {t1 - t0:.2f} s with its cache, grid '
          f'{t2 - t1:.2f} s')
    assert np.array_equal(ref[0], got[0])
    assert max(ratios) <= 1, ratios


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


def test_the_grid_force_is_the_default_force(pyscf_one_thread):
    """Water, the default and the Davidson solver, the excitation gradient
    and the interstate element on one mean field: the two routes' roots
    bitwise, the default chain's fold of the grid seeds the grid force
    bitwise, and the two routes' seeds at that reverse call within the
    kernel's derived bound -- the forces differ by the fold's own rounding
    of those seeds alone, printed beside the fold's exact image of their
    difference; the total gradient against a five-point finite difference
    of E_0 + Omega beside the default route's miss."""
    mol = molecule('water')
    mf = chain_scf(mol)
    chains = {}
    for solver in (None, 'davidson'):
        kw = {} if solver is None else {'solver': solver}
        default = ExcitedStateChain(mol, chain_scf, mf=mf, **kw)
        grid = ExcitedStateChain(mol, chain_scf, mf=mf, bse_adjoint='grid',
                                 **kw)
        chains[solver] = (default, grid)
        calls = (captured(default), captured(grid))
        for element in ('excitation', 'interstate'):
            out = []
            for chain, log in zip((default, grid), calls):
                if element == 'excitation':
                    g, info = chain.excitation_gradient()
                    roots = (info['omega'],)
                else:
                    g, info = chain.interstate_gradient(0, 1)
                    roots = (info['omega_m'], info['omega_n'])
                out.append((np.asarray(g), roots, log[-1]))
            (g_d, om_d, c_d), (g_g, om_g, c_g) = out
            label = f"{solver or 'default'} solver, {element}"
            assert om_d == om_g, (label, om_d, om_g)
            fold = default._fold_to_nuclei
            g_x = np.asarray(fold(c_d['pieces'], *[np.array(a, copy=True)
                                                   for a in c_g['seeds']])[0])
            image = np.abs(fold(c_d['pieces'], *[
                a - b for a, b in zip(c_g['seeds'], c_d['seeds'])])[0]).max()
            x, d, eq, w, no, _, xn, yn = default._casida_args(c_d['pieces'])
            ratios = bound_ratios(c_d['seeds'], c_g['seeds'],
                                  (x, d, eq, w, no, xn, yn), default.spin,
                                  default.bse_tda, c_d['n'], bra=c_d['m'],
                                  symmetrized=c_d['m'] is not None)
            print(f'[info] water {label}: forces apart '
                  f'{np.abs(g_g - g_d).max():.2e} Ha/Bohr, the fold\'s image '
                  f'of the seed difference {image:.1e}; seeds X / D / Wsym '
                  f'at {ratios[0]:.3f} / {ratios[1]:.3f} / {ratios[2]:.3f} '
                  'of the derived bound')
            assert np.array_equal(g_x, g_g), label
            assert np.array_equal(c_d['seeds'][0], c_g['seeds'][0]), label
            assert max(ratios) <= 1, (label, ratios)
    default, grid = chains[None]
    fd = five_point_gradient(default, mol)
    scale = np.abs(fd).max()
    miss_def = np.abs(default.total_gradient()[0] - fd).max() / scale
    miss_grid = np.abs(grid.total_gradient()[0] - fd).max() / scale
    print(f'[info] water five-point miss (relative): grid {miss_grid:.2e}, '
          f'default {miss_def:.2e}')
    assert miss_grid < FD_REL


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
