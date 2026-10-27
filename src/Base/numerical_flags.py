"""Named numerical flags: a shortfall warns and leaves its name where the caller collects.

A solver that stops short of what it was asked (a factorization that falls back, a
residual above its tolerance, a root of low weight) keeps its result, warns with a
warning class of its own, and calls `flag`: inside a `collecting(flags)` block the
entry {'flag': name, 'message': text} is appended to that list, which the caller
copies into its record. Outside any block the warning alone is left.

One block per result: the function that returns a result or record opens it and
writes `gathered(flags)` into what it returns. A block opened inside another
hands its entries on to the outer one as it closes, the worker threads of
`threads.row_map` raise into the block of the thread that started them, and
`gathered` joins every rank's entries, so a flag raised on any thread or rank of
the computation reaches the one record.
"""
import contextlib
import contextvars
import functools
import inspect
import warnings

from src.Base.utils.parallel.mpi_grid import current_comm

# the flag list of the `collecting` block in flight, else None
_FLAGS = contextvars.ContextVar('numerical_flags', default=None)


@contextlib.contextmanager
def collecting(flags):
    """Every `flag` raised inside the block appends its entry to `flags`; on
    leaving it, the entries it added are appended to the enclosing block's
    list too, if there is one."""
    outer = _FLAGS.get()
    start = len(flags)
    token = _FLAGS.set(flags)
    try:
        yield flags
    finally:
        _FLAGS.reset(token)
        if outer is not None and outer is not flags:
            outer.extend(flags[start:])


def gathered(flags, comm=None):
    """The entries of `flags` on every rank of `comm` (default the enclosing
    distribution's, `mpi_grid.current_comm`), each distinct entry once, sorted:
    the same list on every rank, whatever rank or thread raised it."""
    comm = current_comm() if comm is None else comm
    parts = [list(flags)] if comm is None else comm.allgather(list(flags))
    seen = {(e['flag'], e['message']) for part in parts for e in part}
    return [{'flag': name, 'message': text} for name, text in sorted(seen)]


def flag(name, text, category, stacklevel=2, into=None):
    """Warn `text` as `category` from the caller's caller (`stacklevel` as for
    warnings.warn, counted from the function calling this one) and append
    {'flag': name, 'message': text} to the list of the `collecting` block in
    flight, if any, and to `into` (a list a caller handed in), unless that is
    the block's own list."""
    warnings.warn(text, category, stacklevel=stacklevel + 1)
    entry = {'flag': name, 'message': text}
    flags = _FLAGS.get()
    if flags is not None:
        flags.append(entry)
    if into is not None and into is not flags:
        into.append(entry)


def flags_into_out(fn):
    """`fn` run in one `collecting` block, whose entries over every rank
    (`gathered`) are written to out['numerical_flags'] when the call passes
    a dict as its `out` argument: the record of a function that hands its
    record out through `out`."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def recorded(*args, **kwargs):
        out = signature.bind_partial(*args, **kwargs).arguments.get('out')
        flags = []
        with collecting(flags):
            result = fn(*args, **kwargs)
        if out is not None:
            out['numerical_flags'] = gathered(flags)
        return result
    return recorded
