"""Records that survive a relaunch: a JSON tree beside an .npz of its arrays.

A record is a tree of Python values -- None, bools, numbers, strings, lists,
tuples, dicts with any hashable keys, sets, numpy arrays and scalars, and the
dataclasses and plain objects of this package -- written so that reading it
back gives the same values to the bit: a float goes through its shortest
round-trip repr, an array through the .npz untouched, a tuple stays a tuple
and an integer key an integer, so a frozen convention read back compares equal
to the one written and a kernel reading it computes the same bits.

`write_record` is atomic as a pair: the arrays go to a file named by a fresh
token, the JSON naming that file replaces the old JSON in one rename, and only
then are older array files of the same stem removed, so a crash at any point
leaves a JSON whose arrays are on disk. Only classes of this package
(`src.`) are rebuilt on reading; anything else in a tree is refused when it is
written.

`git_commit` names the tree a record was produced on.
"""
import dataclasses
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import uuid
import warnings

import numpy as np

from src.Base.constants import GIT_PROVENANCE_TIMEOUT

#: The package whose classes a record may hold.
RECORD_PACKAGE = 'src.'
#: The checkout this file belongs to (src/Base/record_io.py).
CHECKOUT = pathlib.Path(__file__).resolve().parents[2]


def write_record(path, value):
    """Write `value` to the JSON `path` and its arrays beside it, atomically."""
    path = pathlib.Path(path)
    tree, arrays = encode(value)
    token = uuid.uuid4().hex[:12]
    npz = path.with_name(f'{path.stem}.{token}.npz')
    tmp = npz.with_name(npz.name + '.tmp')
    with open(tmp, 'wb') as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, npz)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps({'arrays': npz.name, 'record': tree}))
    os.replace(tmp, path)
    # Only this stem's own token files: a sibling stem `<stem>.x` keeps its.
    hex_token = ''.join('[0-9a-f]' for _ in token)
    for old in path.parent.glob(f'{path.stem}.{hex_token}.npz'):
        if old.name != npz.name:
            old.unlink()


def read_record(path):
    """The value `write_record` wrote to `path`."""
    path = pathlib.Path(path)
    held = json.loads(path.read_text())
    with np.load(path.with_name(held['arrays']), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return decode(held['record'], arrays)


def record_hash(value):
    """sha256 of a record: its JSON tree and every array's dtype, shape and
    bytes, so two values that read back alike hash alike."""
    tree, arrays = encode(value)
    h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode())
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(f'{key}:{a.dtype.str}:{a.shape}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def git_commit(root=None):
    """The short commit of the checkout `root` (omitted: this one), with
    '+dirty' when its tree differs from it.

    Only the tree: a record is of a method, not of a machine, so the host is
    not part of it (a job row adds where it ran itself). Where git cannot
    answer -- no git, no repository, no reply within GIT_PROVENANCE_TIMEOUT --
    the stamp is 'unknown' and a RuntimeWarning says why, since a number
    without its commit cannot be re-run.
    """
    root = str(CHECKOUT if root is None else root)
    try:
        head = subprocess.run(['git', '-C', root, 'rev-parse', '--short',
                               'HEAD'], capture_output=True, text=True,
                              timeout=GIT_PROVENANCE_TIMEOUT, check=True)
        moved = subprocess.run(['git', '-C', root, 'status', '--porcelain'],
                               capture_output=True, text=True,
                               timeout=GIT_PROVENANCE_TIMEOUT, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f'no commit for the record: git at {root} failed '
                      f'({exc}); stamped unknown', RuntimeWarning)
        return 'unknown'
    return head.stdout.strip() + ('+dirty' if moved.stdout.strip() else '')


def encode(value):
    """(JSON tree, {key: array}) of `value`."""
    arrays = {}
    return _encode(value, arrays, 'record'), arrays


def decode(tree, arrays):
    """The value `encode` gave (tree, arrays) for."""
    return _decode(tree, arrays)


def _class_name(obj):
    cls = type(obj)
    return f'{cls.__module__}:{cls.__qualname__}'


def _encode(value, arrays, where):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, np.generic):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f'{where}: an object array has no record form')
        key = f'a{len(arrays)}'
        arrays[key] = value
        return {'__array__': key}
    if isinstance(value, np.generic):
        return {'__scalar__': value.dtype.str,
                'value': _encode(np.asarray(value).reshape(1), arrays, where)}
    if isinstance(value, tuple):
        return {'__tuple__': [_encode(v, arrays, f'{where}[{i}]')
                              for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_encode(v, arrays, f'{where}[{i}]')
                for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        kind = '__set__' if isinstance(value, set) else '__frozenset__'
        return {kind: [_encode(v, arrays, f'{where}{{}}') for v in value]}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith('__') for k in value):
            return {k: _encode(v, arrays, f'{where}.{k}')
                    for k, v in value.items()}
        return {'__dict__': [[_encode(k, arrays, f'{where} key'),
                              _encode(v, arrays, f'{where}[{k!r}]')]
                             for k, v in value.items()]}
    name = _class_name(value)
    if not name.startswith(RECORD_PACKAGE):
        raise TypeError(f'{where}: a {name} has no record form; records hold '
                        f'plain values and the classes of {RECORD_PACKAGE}')
    if dataclasses.is_dataclass(value):
        return {'__dataclass__': name,
                'fields': {f.name: _encode(getattr(value, f.name), arrays,
                                           f'{where}.{f.name}')
                           for f in dataclasses.fields(value)}}
    return {'__object__': name,
            'state': _encode(dict(vars(value)), arrays, f'{where}.__dict__')}


def record_class(name):
    """The class a record names, only from this package."""
    if not name.startswith(RECORD_PACKAGE):
        raise TypeError(f'a record names {name}, outside {RECORD_PACKAGE}')
    module, qualname = name.split(':')
    out = importlib.import_module(module)
    for part in qualname.split('.'):
        out = getattr(out, part)
    return out


def _decode(tree, arrays):
    if isinstance(tree, list):
        return [_decode(v, arrays) for v in tree]
    if not isinstance(tree, dict):
        return tree
    if '__array__' in tree:
        return arrays[tree['__array__']]
    if '__scalar__' in tree:
        return _decode(tree['value'], arrays).astype(tree['__scalar__'])[0]
    if '__tuple__' in tree:
        return tuple(_decode(v, arrays) for v in tree['__tuple__'])
    if '__set__' in tree:
        return {_decode(v, arrays) for v in tree['__set__']}
    if '__frozenset__' in tree:
        return frozenset(_decode(v, arrays) for v in tree['__frozenset__'])
    if '__dict__' in tree:
        return {_decode(k, arrays): _decode(v, arrays)
                for k, v in tree['__dict__']}
    if '__dataclass__' in tree:
        cls = record_class(tree['__dataclass__'])
        values = {k: _decode(v, arrays) for k, v in tree['fields'].items()}
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        out = cls(**{k: v for k, v in values.items() if k in init})
        for k, v in values.items():
            if k not in init:
                object.__setattr__(out, k, v)
        return out
    if '__object__' in tree:
        cls = record_class(tree['__object__'])
        out = cls.__new__(cls)
        out.__dict__.update(_decode(tree['state'], arrays))
        return out
    return {k: _decode(v, arrays) for k, v in tree.items()}
