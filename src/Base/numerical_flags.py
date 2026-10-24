"""Named numerical flags: a shortfall warns and leaves its name where the caller collects.

A solver that stops short of what it was asked (a factorization that falls back, a
residual above its tolerance, a root of low weight) keeps its result, warns with a
warning class of its own, and calls `flag`: inside a `collecting(flags)` block the
entry {'flag': name, 'message': text} is appended to that list, which the caller
copies into its record. Outside any block the warning alone is left.
"""
import contextlib
import contextvars
import warnings

# the flag list of the `collecting` block in flight, else None
_FLAGS = contextvars.ContextVar('numerical_flags', default=None)


@contextlib.contextmanager
def collecting(flags):
    """Every `flag` raised inside the block appends its entry to `flags`."""
    token = _FLAGS.set(flags)
    try:
        yield flags
    finally:
        _FLAGS.reset(token)


def flag(name, text, category, stacklevel=2):
    """Warn `text` as `category` from the caller's caller (`stacklevel` as for
    warnings.warn, counted from the function calling this one) and append
    {'flag': name, 'message': text} to the list of the `collecting` block in
    flight, if any."""
    warnings.warn(text, category, stacklevel=stacklevel + 1)
    flags = _FLAGS.get()
    if flags is not None:
        flags.append({'flag': name, 'message': text})
