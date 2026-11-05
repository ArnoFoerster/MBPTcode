"""Threaded operand copies (Base/utils/threaded_copies, used by cached_einsum): the contractions
agree with plain numpy einsum, the copies are exact, and the switch is reversible."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils import threaded_copies as tc
from src.Base.utils import cached_einsum as ce


@pytest.fixture
def threaded():
    tc.use_threaded_transposes(4)
    old = tc._COPY_MIN_ELEMENTS
    tc._COPY_MIN_ELEMENTS = 1          # force the threaded path on small arrays
    yield
    tc._COPY_MIN_ELEMENTS = old
    tc.use_threaded_transposes(1)


EXPRESSIONS = [
    ('Qac,icdw->qiadw'.replace('q', 'Q', 1), ((7, 5, 6), (3, 6, 6, 4))),
    ('Qiadw,Qbd->iabw', ((7, 3, 5, 6, 4), (7, 5, 6))),
    ('ikca,ijaw->kcjw', ((3, 4, 6, 5), (3, 3, 5, 4))),
    ('Qjk,kcjw->Qcw', ((7, 3, 4), (4, 6, 3, 5))),
    ('Qcp,Qcw->pw', ((7, 6, 9), (7, 6, 5))),
    ('ijcd,Qijdw->Qcw', ((3, 3, 6, 6), (7, 3, 3, 6, 4))),
]


@pytest.mark.parametrize('sub,shapes', EXPRESSIONS)
def test_threaded_einsum_matches_numpy(threaded, sub, shapes):
    rng = np.random.default_rng(1)
    ops = [rng.standard_normal(s) for s in shapes]
    ref = np.einsum(sub, *ops, optimize=True)
    got = ce.einsum(sub, *ops)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-12 * max(1.0, np.max(np.abs(ref)))


def test_threaded_tensordot_equals_numpy_bitwise(threaded):
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((6, 5, 7, 4)), rng.standard_normal((7, 3, 6, 5))
    axes = ((0, 2), (2, 0))
    assert np.array_equal(tc.threaded_tensordot(a, b, axes), np.tensordot(a, b, axes=axes))


def test_switch_is_reversible():
    tc.use_threaded_transposes(1)
    assert tc._COPY_POOL is None
    a = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(ce.einsum('ij,jk->ik', a, a.T), a @ a.T)


@pytest.mark.parametrize('shape,perm', [((6, 5, 7), (2, 0, 1)), ((2, 3, 400), (1, 2, 0)), ((300, 4), (1, 0)),
                                         ((1, 1, 90), (2, 0, 1)), ((5, 6), (0, 1))])
def test_threaded_contiguous_copy_is_exact(threaded, shape, perm):
    x = np.random.default_rng(2).standard_normal(shape).transpose(perm)
    y = tc.ascontiguous_threaded(x)
    assert y.flags.c_contiguous and y.shape == x.shape
    assert np.array_equal(y, x)


def test_g_slice_df_equals_numpy_einsum(threaded):
    from src.SingleReference.common.eri_blocks import g_slice_df
    rng = np.random.default_rng(5)
    B = rng.standard_normal((9, 14, 14))
    p, q, r, s = slice(0, 4), slice(4, 14), slice(4, 14), slice(0, 5)
    ref = np.einsum('Qpr,Qqs->pqrs', B[:, p, r], B[:, q, s], optimize=True)
    got = g_slice_df(B, p, q, r, s)
    assert got.flags.c_contiguous and np.max(np.abs(got - ref)) < 1e-13 * np.max(np.abs(ref))
