"""Path-caching drop-in replacement for numpy.einsum, used by generated CC
residual/density code.

Every generated function calls numpy.einsum(..., optimize=True) many times
per DIIS iteration on operands whose shapes never change across iterations
(no/nv are fixed for a given solve). Plain numpy re-runs its full
einsum_path search on every single call regardless of that -- profiling
showed this costs ~20% of restricted CCSDT/Lambda3 wall time on Ne/cc-pVDZ.
This wrapper memoizes the contraction path by (subscripts, operand shapes)
and reuses it; the arithmetic performed is identical to plain
optimize=True, just without recomputing the path each call.

'optimal' path search is combinatorial in operand count (brute-force over
every contraction order) -- fine for the <=4-operand contractions ordinary
generated CC/density code has, but laplace_codegen.py's Laplace-fused terms
routinely carry 7-9 operands (a couple of real tensors plus one Oe/Ve
dressing vector per amplitude external index) and 'optimal' search there
was measured to hang (still not finished after 2+ minutes) where 'greedy'
finds a path in under a millisecond. Only the ONE-TIME path-finding cost
differs between the two strategies (memoized afterward same as before);
'greedy' can in principle pick a very slightly less optimal contraction
order, but for these shapes the difference is not measurable and "finishes
at all" dominates.

Path search uses the standalone opt_einsum package's contract_path, NOT
numpy's own built-in einsum_path -- found (profiling DF-ADC(2)-x
MP3-density rollout on CO/aug-cc-pvdz) that numpy's own 'optimal' path
search silently returns a badly wrong (non-optimal, despite the name) path
for "triangle"-connectivity contractions: three operands where each PAIR
shares exactly one summed index but no single index is shared by all three
(e.g. 'Qac,Qbd,cdij->abij', the DF-dressed vvvv-type term df_codegen.py
produces -- two B-factors sharing the auxiliary axis Q, each also sharing a
different axis with the amplitude). Measured on that exact term
(naux=144, nv=39, no=7): numpy's own optimizer reports "Optimized scaling"
identical to "Naive scaling" (no improvement found at all) and picks a
single 5e10-FLOP direct contraction, timing at 12+ seconds; opt_einsum's
own contract_path finds the correct 2-step factorization (8.9e8 FLOPs, a
measured 55x reduction) in milliseconds. This pattern never arose before
the DF rewrite (a dense g_xxxx factor was always a single
pre-formed operand, never two separately-contracted DF factors sharing an
auxiliary axis), so it was never exercised by earlier CC/CCSD(T) profiling.
opt_einsum.contract_path's returned path is a bare list of index-pair
tuples (NOT numpy's own ['einsum_path', ...]-prefixed format) -- prepend
'einsum_path' before handing it to np.einsum(optimize=path); verified this
combination reproduces opt_einsum.contract's own numeric result exactly and
runs at opt_einsum's (not numpy's) speed.

A FactorizedOperand (an operand known only as a sum of products of smaller
arrays, e.g. the Laplace-fused third-order quadruples of laplace_amplitudes)
is expanded term by term, exactly, by factorized_einsum -- also when a
generated module calls numpy.einsum directly, through numpy's
__array_function__ dispatch.
"""
import itertools
import math
import os

import numpy as np
import opt_einsum

_PATH_CACHE = {}
_OPTIMAL_MAX_OPERANDS = 4
# path strategy for the products a factorized operand expands into: exact
# dynamic programming without outer products (greedy can pick an outer product
# of two doubles, i.e. the o^4 v^4 intermediate the factorization exists to avoid)
_EXPANDED_STRATEGY = 'dp'

# Debug-only rank ceiling, gated by an env var so it costs nothing (and
# doesn't need opt_einsum) when unset -- set MBPT_MAX_EINSUM_RANK=4 to
# enforce, e.g., the MP2/MP3 density paths' "no rank>4 array" invariant
# (see src/SingleReference/DensityMatrix/mpn_density_driver_restricted.py/
# mpn_density_driver_unrestricted.py's Laplace-fusion docstrings). Checks
# BOTH the ndim of every operand passed in AND the ndim of every intermediate
# opt_einsum's own contraction path would materialize -- a fused multi-
# operand einsum whose optimal path happens to materialize a larger
# intermediate than any single operand is just as much a violation as a
# literal rank>4 array showing up as a Python variable. Only runs once per
# distinct (subscripts, shapes) key, the same lifetime as the path cache
# itself, so it doesn't add per-call overhead in a hot loop.
_ASSERT_MAX_RANK = os.environ.get('MBPT_MAX_EINSUM_RANK')


_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class FactorizedOperand:
    """An einsum operand held as a sum of products of smaller arrays and never
    formed: coef times a view (axis k of the view is canonical axis axes[k]) of
    the canonical tensor. Subclasses set `shape`, `axes`, `coef` and
    `fresh_letters` (summation letters one expansion term may introduce) and
    define `_view` and `_canonical_terms`. numpy.einsum dispatches here; every
    other numpy function refuses it, so nothing materializes it by accident."""

    shape = ()
    axes = ()
    coef = 1.0
    fresh_letters = 0
    # numpy scalars defer to the operand's own reflected arithmetic
    __array_ufunc__ = None

    @property
    def ndim(self):
        return len(self.shape)

    def _view(self, axes, coef):
        raise NotImplementedError

    def _canonical_terms(self, letters, fresh):
        """[(coef, ((sub, array), ...))] of the canonical tensor whose axis k
        carries letters[k]; new summation letters are taken from `fresh`."""
        raise NotImplementedError

    def einsum_terms(self, subscript, fresh=()):
        """[(coef, ((sub, array), ...)), ...]: this operand indexed by
        `subscript` equals the sum over terms of coef * prod(array[sub])."""
        if self.coef == 0:
            return []
        letters = [None] * len(self.axes)
        for k, c in enumerate(subscript):
            letters[self.axes[k]] = c
        return [(self.coef * c, pieces) for c, pieces in self._canonical_terms(letters, list(fresh))]

    def transpose(self, *axes):
        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        return self._view(tuple(self.axes[a] for a in axes), self.coef)

    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return self._view(self.axes, self.coef * scalar)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def dense(self):
        """The materialized tensor -- validation-scale systems only."""
        sub = _LETTERS[:self.ndim]
        return factorized_einsum(f'{sub}->{sub}', self)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.einsum and kwargs.get('out') is None:
            return factorized_einsum(*args)
        return NotImplemented

    def __array__(self, *args, **kwargs):
        raise TypeError(f"{type(self).__name__} is never materialized implicitly")


def _assert_ranks_ok(subscripts, operands, strategy, max_rank):
    for op in operands:
        if op.ndim > max_rank:
            raise AssertionError(
                f"cached_einsum: operand of rank {op.ndim} > {max_rank} in '{subscripts}'")
    _, info = opt_einsum.contract_path(subscripts, *operands, optimize=strategy)
    for step in info.contraction_list:
        einsum_str = step[2]
        out_rank = len(einsum_str.split('->')[1])
        if out_rank > max_rank:
            raise AssertionError(
                f"cached_einsum: intermediate of rank {out_rank} > {max_rank} in "
                f"'{subscripts}' (contraction step '{einsum_str}')")


def _cached_path(subscripts, operands, strategy):
    key = (subscripts, tuple(op.shape for op in operands), strategy)
    path = _PATH_CACHE.get(key)
    if path is None:
        oe_path, _ = opt_einsum.contract_path(subscripts, *operands, optimize=strategy)
        path = ['einsum_path'] + list(oe_path)
        _PATH_CACHE[key] = path
        if _ASSERT_MAX_RANK is not None:
            _assert_ranks_ok(subscripts, operands, strategy, int(_ASSERT_MAX_RANK))
    return path


def einsum(subscripts, *operands, optimize=True):
    if any(isinstance(op, FactorizedOperand) for op in operands):
        return factorized_einsum(subscripts, *operands)
    if optimize is True and len(operands) > 1:
        strategy = 'optimal' if len(operands) <= _OPTIMAL_MAX_OPERANDS else 'greedy'
        optimize = _cached_path(subscripts, operands, strategy)
    return np.einsum(subscripts, *operands, optimize=optimize)


def _canonical_contraction(pieces, output):
    """(key, output in canonical letters): the contraction graph with every letter
    renamed by first appearance, so two products differing only by a relabelling of
    the output letters share a key and differ by a transpose."""
    ren = {}
    canon = []
    for sub, arr in pieces:
        for ch in sub:
            ren.setdefault(ch, _LETTERS[len(ren)])
        canon.append((''.join(ren[ch] for ch in sub), id(arr)))
    out = ''.join(ren[c] for c in output)
    return (tuple(canon), frozenset(out)), out


def factorized_einsum(subscripts, *operands):
    """einsum with every FactorizedOperand replaced by its sum of products, exact:
    each distinct product (up to a relabelling of the output letters, which the
    permutation copies of a generated term table are) is contracted once and added
    in every orientation it occurs in; a term that is itself factorized is expanded
    in turn. Each operand draws its new summation letters from its own slice of the
    letters the expression leaves free, so no two expansions can collide."""
    if '->' not in subscripts:
        raise ValueError(f"factorized_einsum needs an explicit output: {subscripts!r}")
    inputs, output = subscripts.replace(' ', '').split('->')
    subs = inputs.split(',')
    dims = {}
    for sub, op in zip(subs, operands):
        dims.update(zip(sub, op.shape))
    result = np.zeros(tuple(dims[c] for c in output))
    # a vanishing dense factor (the Brillouin-zero f[o,v] block) zeroes the product
    if any(not isinstance(op, FactorizedOperand) and np.ndim(op) <= 2 and not np.any(op)
           for op in operands):
        return result
    free = [c for c in _LETTERS if c not in set(inputs) | set(output)]
    choices = []
    for sub, op in zip(subs, operands):
        if not isinstance(op, FactorizedOperand):
            choices.append(((1.0, ((sub, op),)),))
            continue
        if op.fresh_letters > len(free):
            raise ValueError(f"factorized_einsum: no einsum letters left for {subscripts!r}")
        mine, free = free[:op.fresh_letters], free[op.fresh_letters:]
        choices.append(op.einsum_terms(sub, mine))
    groups = {}
    for combo in itertools.product(*choices):
        pieces = [piece for _, term in combo for piece in term]
        key, out = _canonical_contraction(pieces, output)
        entry = groups.setdefault(key, [pieces, []])
        entry[1].append((math.prod(c for c, _ in combo), out))
    for (canon, outset), (pieces, members) in groups.items():
        arrays = [arr for _, arr in pieces]
        target = ''.join(sorted(outset))
        expr = ','.join(sub for sub, _ in canon) + '->' + target
        if any(isinstance(a, FactorizedOperand) for a in arrays):
            value = factorized_einsum(expr, *arrays)
        else:
            strategy = 'optimal' if len(arrays) <= _OPTIMAL_MAX_OPERANDS else _EXPANDED_STRATEGY
            path = _cached_path(expr, arrays, strategy) if len(arrays) > 1 else False
            value = np.einsum(expr, *arrays, optimize=path)
        for coef, out in members:
            result += coef * value.transpose([target.index(c) for c in out])
    return result
