"""What the whole-fit gradient holds at once, read off tracemalloc.

The replicated fit's adjoint (`isdf_derivatives.fit_adjoint`) and the test set
it differentiates (`separable_ri.test_set_D`) are the widest arrays of a
state-pair force: D is (M, npair + naux). On ethylene/cc-pVDZ at 148 points
per atom each is compared with a whole-array form written out below, which
gives the same bits. The blocked adjoint holds two arrays of D's shape beside
D where the whole form holds four and a transient, and its last update holds
D_bar, F_bar and one block of `FIT_CHOLESKY_BLOCK` grid rows; the test set
holds D and one block of pair columns instead of three whole-width products.

The mean field abandoned at each geometry is a reference cycle
(`pyscf_interface.response_kernel` caches pyscf's closure on it), freed only
by the cyclic collector, which `FactorChain.mean_field` runs before the next
SCF; with automatic collection off, consecutive excitation gradients at
displaced geometries peak flat on the whole fit and on the row fit.

The whole form refuses above `WHOLE_FIT_ADJOINT_MAX_GB`, estimated from the
shapes before any array exists (`isdf_derivatives.whole_fit_adjoint_gb`):
the test set over every product pair three times, F and the three-centre
tensor with its adjoint. The chlorophyllide dimer is refused with the row fit
named; with the cap below water's estimate, `dfactor_adjoint_gauges`,
`isdf_exchange_skeleton(fit='replicated')` and the chain's `nuclear_gradient`
refuse before the three-centre integrals or the test set are built.
"""
import gc
import inspect
import os
import re
import sys
import tracemalloc
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf

from src.Base.constants import FIT_CHOLESKY_BLOCK, WHOLE_FIT_ADJOINT_MAX_GB
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import DEFAULT_REGULARIZATION
# aliased on import: pytest collects a module-level `test_*` callable
from src.Base.separable_ri import test_set_D as build_test_set
from src.gradients import isdf_derivatives
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.isdf_derivatives import (dfactor_adjoint_gauges,
                                            fit_adjoint,
                                            isdf_exchange_skeleton,
                                            product_pairs,
                                            require_whole_fit_adjoint,
                                            whole_fit_adjoint_gb)

BASIS = 'cc-pvdz'
ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: The molecules the whole-form gates run on.
CAP_MOLECULES = {
    'water': WATER,
    'ethylene': ETHYLENE,
    'formaldehyde': ('C 0 0 -0.5296; O 0 0 0.6763; H 0 0.9357 -1.1172; '
                     'H 0 -0.9357 -1.1172'),
    'benzene': '; '.join(
        f'{el} {r * np.cos(k * np.pi / 3):.6f} {r * np.sin(k * np.pi / 3):.6f} 0'
        for el, r in (('C', 1.3915), ('H', 2.4715)) for k in range(6))}
#: The chlorophyllide dimer, (nao, AOs with l <= 2, naux of the -RI set,
#: grid points) at cc-pVDZ and cc-pVTZ, and the band the estimate must fall
#: in, in GB.
DIMER_SHAPES = ((1780, 1780, 6648, 52688, 4.4e3, 4.6e3),
                (4068, 3382, 10420, 79032, 2.9e4, 3.1e4))
#: Headroom on every pinned bound: numpy's own bookkeeping, not an array.
HEADROOM = 1.05
#: Bytes a pinned bound allows beyond the arrays it names.
SLACK = 5e5
#: Bytes four consecutive gradients may peak apart and still be flat: half
#: of what one abandoned geometry's mean field and fit hold here.
FLAT = 2e6


class Traced:
    """tracemalloc on for the block, left as it was found."""

    def __enter__(self):
        self.started = not tracemalloc.is_tracing()
        if self.started:
            tracemalloc.start()
        return self

    def __exit__(self, *exc):
        if self.started:
            tracemalloc.stop()


def chain_scf(mol):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def hstacked_test_set(mol, auxmol, coords, layout):
    """`test_set_D` as the three whole-width products and a concatenation."""
    mu, nu, w = layout
    ao = mol.eval_gto('GTOval_sph', coords)
    pairs = ao[:, mu] * ao[:, nu] * w[None, :]
    return np.hstack([pairs, auxmol.eval_gto('GTOval_sph', coords)])


def whole_fit_adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
    """`fit_adjoint` as whole arrays, every intermediate alive to the end."""
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
    G_bar = -(B.T @ A_bar)
    F_bar = A_bar @ Dt
    Dt_bar = A_bar.T @ F + (G_bar + G_bar.T) @ Dt
    D_bar = d[:, None] * Dt_bar
    d_bar = d_bar + np.einsum('kr,kr->k', Dt_bar, D)
    s_bar = -d_bar * d ** 2
    D_bar += (s_bar / s)[:, None] * D
    return D_bar, F_bar


def traced_increment(fn, *args):
    """(fn(*args), the traced peak above the bytes held at the call)."""
    gc.collect()
    tracemalloc.reset_peak()
    start = tracemalloc.get_traced_memory()[0]
    out = fn(*args)
    return out, tracemalloc.get_traced_memory()[1] - start


def line_increments(fn, *args):
    """(fn(*args), {line: traced peak above the call's start}) over fn's own
    lines, callees folded into the line that called them."""
    code, peaks, last = fn.__code__, {}, {}

    def local(frame, event, arg):
        if event in ('line', 'return'):
            peak = tracemalloc.get_traced_memory()[1] - start
            if 'line' in last:
                peaks[last['line']] = max(peaks.get(last['line'], 0), peak)
            tracemalloc.reset_peak()
            last['line'] = frame.f_lineno
        return local

    def enter(frame, event, arg):
        return local if event == 'call' and frame.f_code is code else None

    gc.collect()
    tracemalloc.reset_peak()
    start = tracemalloc.get_traced_memory()[0]
    previous = sys.gettrace()
    sys.settrace(enter)
    try:
        out = fn(*args)
    finally:
        sys.settrace(previous)
    return out, peaks


def source_line(fn, pattern):
    """The line number of the one line of fn's source matching `pattern`."""
    lines, first = inspect.getsourcelines(fn)
    hits = [first + i for i, text in enumerate(lines)
            if re.search(pattern, text)]
    assert len(hits) == 1, f'{pattern!r} matches lines {hits} of {fn.__name__}'
    return hits[0]


def bitwise(a, b):
    """Same shape, same memory order, same bytes."""
    return (a.shape == b.shape and a.strides == b.strides
            and a.tobytes() == b.tobytes())


@pytest.fixture(scope='module')
def fit_inputs():
    """Ethylene's frozen layout, points and auxiliary molecule."""
    mol = gto.M(atom=ETHYLENE, basis=BASIS, verbose=0)
    fz = FrozenFactorization(mol)
    return mol, fz.auxmol(mol), fz.coords(mol), fz.layout


def test_the_test_set_is_filled_in_column_blocks(fit_inputs):
    """D and one block of pair columns, not three whole-width products; the
    same bits and the same memory order as the concatenation."""
    mol, auxmol, crd, layout = fit_inputs
    with Traced():
        whole, peak_whole = traced_increment(hstacked_test_set, mol, auxmol,
                                             crd, layout)
        D, peak = traced_increment(build_test_set, mol, auxmol, crd, layout)
    assert bitwise(D, whole)
    M, npair = D.shape[0], len(layout[0])
    block = 3 * M * min(npair, FIT_CHOLESKY_BLOCK) * 8
    collocations = M * (mol.nao_nr() + auxmol.nao_nr()) * 8
    bound = HEADROOM * (D.nbytes + block + collocations) + SLACK
    print(f'\ntest_set_D: whole-array {peak_whole / 1e6:.1f} MB, blocked '
          f'{peak / 1e6:.1f} MB (bound {bound / 1e6:.1f}), |D| '
          f'{D.nbytes / 1e6:.2f} MB')
    assert peak <= bound, (peak, bound)
    # the concatenation holds the whole pair block beside D at the least
    assert peak_whole > D.nbytes + M * npair * 8 > bound, (peak_whole, bound)


def test_fit_adjoint_holds_two_arrays_of_d_beside_it(fit_inputs):
    """The adjoint beside its input holds two arrays of D's shape, and its
    last update one D_bar and a block of rows; the same bits as the whole
    form, which holds four and a transient."""
    mol, auxmol, crd, layout = fit_inputs
    D = build_test_set(mol, auxmol, crd, layout)
    M, W = D.shape
    assert M > FIT_CHOLESKY_BLOCK, 'the last update needs more than one block'
    naux = auxmol.nao_nr()
    rng = np.random.default_rng(0)
    F = rng.standard_normal((naux, W))
    M_bar = rng.standard_normal((naux, M))
    with Traced():
        (D_bar_w, F_bar_w), peak_whole = traced_increment(
            whole_fit_adjoint, D, F, M_bar)
        (D_bar, F_bar), lines = line_increments(fit_adjoint, D, F, M_bar)
    assert bitwise(D_bar, D_bar_w) and bitwise(F_bar, F_bar_w)
    n_D, n_G, n_F, n_A = D.nbytes, M * M * 8, F.nbytes, M_bar.nbytes
    peak = max(lines.values())
    # two of D's shape and one of F's at the Dt term of Dt_bar, or four
    # (M, M) at the Cholesky solves, with the (naux, M) intermediates
    bound = HEADROOM * max(2 * n_D + n_G + n_F + n_A,
                           n_D + 4 * n_G + 3 * n_A) + SLACK
    tail = lines[source_line(fit_adjoint, r'D_bar.*\+=.*\* D')]
    tail_bound = HEADROOM * (n_D + n_F + FIT_CHOLESKY_BLOCK * W * 8) + SLACK
    print(f'\nfit_adjoint: whole-array {peak_whole / 1e6:.1f} MB, blocked '
          f'{peak / 1e6:.1f} MB (bound {bound / 1e6:.1f}); its last update '
          f'{tail / 1e6:.1f} MB (bound {tail_bound / 1e6:.1f}); |D| '
          f'{n_D / 1e6:.2f} MB')
    assert peak <= bound, (peak, bound)
    assert tail <= tail_bound, (tail, tail_bound)
    assert peak_whole > 4 * n_D > bound, (peak_whole, bound)


def four_gradient_peaks(chain, displaced):
    """(start, peak) traced bytes of consecutive excitation gradients, the
    automatic collector off so that only the chain's own collection runs."""
    rows = []
    gc.collect()
    gc.disable()
    try:
        for atom in displaced:
            here = gto.M(atom=atom, basis=BASIS, verbose=0)
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            chain.excitation_gradient(here)
            del here
            rows.append((start, tracemalloc.get_traced_memory()[1]))
    finally:
        gc.enable()
    return rows


def test_consecutive_gradients_peak_flat():
    """Four displaced geometries in a row, on the whole fit and on the row
    fit: each gradient peaks where the last one did, since the mean field
    and fit it abandoned are collected before its SCF."""
    mol = gto.M(atom=ETHYLENE, basis=BASIS, verbose=0)
    mf = chain_scf(mol)
    displaced = [ETHYLENE.replace('C 0 0 0.6695',
                                  f'C 0 0 {0.6795 + 0.005 * k:.4f}')
                 for k in range(4)]
    peaks = {}
    with Traced():
        for tag, kw in (('whole', {}),
                        ('rows', dict(sliced=True, fit='rows'))):
            chain = ExcitedStateChain(mol, chain_scf, mf=mf,
                                      solver='davidson', **kw)
            rows = four_gradient_peaks(chain, displaced)
            del chain
            starts = [r[0] for r in rows]
            peaks[tag] = [r[1] for r in rows]
            print(f'\n{tag} fit: starts '
                  + ' '.join(f'{s / 1e6:.1f}' for s in starts) + ' MB, peaks '
                  + ' '.join(f'{p / 1e6:.1f}' for p in peaks[tag]) + ' MB')
            # the first start holds no abandoned geometry yet, every later
            # one exactly one, not yet collected
            assert max(starts[1:]) - min(starts[1:]) < FLAT, (tag, starts)
            assert max(peaks[tag]) - min(peaks[tag]) < FLAT, (tag, peaks[tag])
    # the row fit forms no array of the test set's width
    assert max(peaks['rows']) < min(peaks['whole']), peaks


def test_the_whole_fit_adjoint_refuses_the_dimer():
    """The chlorophyllide dimer's whole-form adjoint, estimated from its
    shapes alone: refused, and the refusal names the row fit."""
    for nao, n2, naux, npoint, low, high in DIMER_SHAPES:
        gb = whole_fit_adjoint_gb(nao, naux, npoint, nao * n2)
        print(f'\ndimer nao {nao}: {gb:.3g} GB')
        assert low < gb < high, gb
        with pytest.raises(MemoryError) as refusal:
            require_whole_fit_adjoint(nao, naux, npoint, nao * n2, 'probe')
        message = str(refusal.value)
        assert 'WHOLE_FIT_ADJOINT_MAX_GB' in message
        assert "fit='rows'" in message


def test_the_whole_fit_adjoint_refuses_before_allocating(monkeypatch):
    """With the cap below water's estimate, every whole form refuses before
    it builds the three-centre tensor or the test set."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    fz = FrozenFactorization(mol)
    auxmol, crd = fz.auxmol(mol), fz.coords(mol)

    def built(*args, **kwargs):
        raise AssertionError('an array of the whole form was built')

    mf = isdf_jk(scf.RHF(mol), auxbasis=BASIS + '-ri')
    mf.with_df.build()
    monkeypatch.setattr(isdf_derivatives, 'WHOLE_FIT_ADJOINT_MAX_GB', 1e-4)
    monkeypatch.setattr(isdf_derivatives, 'test_set_D', built)
    monkeypatch.setattr(isdf_derivatives.pyscf_df.incore, 'aux_e2', built)
    d_bar = np.zeros((len(crd), auxmol.nao_nr()))
    stub = SimpleNamespace(factorization=SimpleNamespace(fit='replicated'))
    calls = {
        'dfactor_adjoint_gauges': lambda: dfactor_adjoint_gauges(
            mol, auxmol, crd, [(d_bar, None)], fz.layout, fz.pts_local,
            fz.owner),
        'isdf_exchange_skeleton': lambda: isdf_exchange_skeleton(
            mf, dm=np.eye(mol.nao_nr()), channels=[(0.0, 1.0)],
            fit='replicated'),
        'FactorChain.nuclear_gradient': lambda: FactorChain.nuclear_gradient(
            stub, mol, None, auxmol, crd, None, None, None, None)}
    for what, call in calls.items():
        with pytest.raises(MemoryError, match='WHOLE_FIT_ADJOINT_MAX_GB'):
            call()


@pytest.mark.parametrize('name', ['water', 'ethylene', 'formaldehyde',
                                  'benzene'])
def test_the_test_molecules_pass_the_cap(name):
    """The molecules the whole form's gates run on sit orders under it."""
    mol = gto.M(atom=CAP_MOLECULES[name], basis=BASIS, verbose=0)
    fz = FrozenFactorization(mol)
    gb = require_whole_fit_adjoint(mol.nao_nr(), fz.auxmol(mol).nao_nr(),
                                   len(fz.coords(mol)),
                                   len(product_pairs(mol)[0]), name)
    print(f'\n{name}: {gb:.3g} GB')
    assert gb < 1e-2 * WHOLE_FIT_ADJOINT_MAX_GB


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
