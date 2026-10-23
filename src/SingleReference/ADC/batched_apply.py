"""Batched (several trial vectors at once) twins of the generated restricted
apply modules, and the class-block matmat built on them.

A generated apply function y = fn(..., x) folds ONE trial tensor x through a few
hundred to a few thousand einsums, each taking x as its last operand
(pdaggerq_interface.raw_terms_to_sigma_einsum). Applied column by column, a
block of d columns costs d times that many einsum dispatches; for the small
spatial tensors of the 1h and 2h1p classes the dispatch, not the arithmetic,
is the cost. The twin carries one extra trailing axis (the batch) on x and on y:

    einsum('...,jlb->ija', ..., x, optimize=True)
        -> einsum('...,jlbZ->ijaZ', ..., x, optimize=True)
    y = np.zeros((no, no, nv,))  ->  y = np.zeros((no, no, nv,) + x.shape[-1:])

so one call applies the structure to every column. The twin is made from the
module's source text when first needed (nothing is regenerated on disk); a
module with a line of any other shape is refused instead of being batched
wrongly.
"""
import inspect
import re
import types

import numpy as np

from src.SingleReference.ADC.sigma_driver import restricted_class_matvec

BATCH_LETTER = 'Z'
# elements of one batched spatial trial tensor (columns x spatial size); 2**25 doubles
# = 256 MB, so the einsum intermediates, a few times that, stay well inside a task
BATCH_ELEMENTS = 2 ** 25

_APPLY = re.compile(r"^(    y [+-]= [0-9.e+-]+ \* einsum\(')([^']*)(', .*, x, optimize=True\))$")
_ZEROS = re.compile(r"^(    y = np\.zeros\()(\([^()]*\))(\))$")
_PLAIN = re.compile(r'^(def |    """|    return y$|#|import |from |$)')

_twins = {}


def _batch_subscript(sub):
    """'t1,...,tk,x->out' with the batch letter on the trial (last) operand and
    the output."""
    if BATCH_LETTER in sub:
        raise ValueError(f"batched_apply: subscript {sub!r} already uses {BATCH_LETTER!r}")
    lhs, out = sub.split('->')
    return f"{lhs}{BATCH_LETTER}->{out}{BATCH_LETTER}"


def batched_source(src):
    """The batched twin's source text of a generated apply module."""
    out = []
    for ln, line in enumerate(src.splitlines(), 1):
        m = _APPLY.match(line)
        if m:
            out.append(f"{m.group(1)}{_batch_subscript(m.group(2))}{m.group(3)}")
            continue
        z = _ZEROS.match(line)
        if z:
            out.append(f"{z.group(1)}{z.group(2)} + x.shape[-1:]{z.group(3)}")
            continue
        if not _PLAIN.match(line):
            raise ValueError(f"batched_apply: unexpected generated line {ln}: {line.strip()[:80]!r}")
        out.append(line)
    return '\n'.join(out) + '\n'


def batched_twin(module):
    """The batched twin of a generated apply module (cached per module)."""
    name = module.__name__
    if name not in _twins:
        twin = types.ModuleType(name + '_batched')
        twin.__file__ = getattr(module, '__file__', None)
        exec(compile(batched_source(inspect.getsource(module)), name + '_batched', 'exec'),
             twin.__dict__)
        _twins[name] = twin
    return _twins[name]


def restricted_class_matmat(module, prefix, V, cfg_bra, pool, no, nv, cfg_ket=None,
                            batch_elements=BATCH_ELEMENTS):
    """Y = block @ V for the columns of V (ket configurations x columns):
    sigma_driver.restricted_class_matvec on the batched twin, the columns folded
    in chunks whose spatial trial tensor holds at most batch_elements numbers."""
    if cfg_ket is None:
        cfg_ket = cfg_bra
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        return restricted_class_matvec(module, prefix, V, cfg_bra, pool, no, nv, cfg_ket)
    twin = batched_twin(module)
    size = max(no ** cfg['hole_sp'].shape[1] * nv ** cfg['part_sp'].shape[1]
               for cfg in (cfg_bra, cfg_ket))
    chunk = max(1, batch_elements // size)
    return np.concatenate(
        [restricted_class_matvec(twin, prefix, V[:, c0:c0 + chunk], cfg_bra, pool, no, nv,
                                 cfg_ket) for c0 in range(0, V.shape[1], chunk)], axis=1)
