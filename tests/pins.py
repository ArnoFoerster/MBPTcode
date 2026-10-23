"""Pins to a few ulp on the machine that recorded them, a bar on any other BLAS.

A pin file (tests/one_fit_pins.json, tests/baseline_forward_routes.json)
carries a 'blas' field: the identity of the BLAS libraries of the process
that wrote it (`blas_identity`). A pin read under the same identity holds to
PIN_SAME_ULPS ulp of its magnitude (floored at one unit, `same_bar`): two runs
of one code on one machine are not bitwise, a reduction in another order
moving the last bit. `==`
is kept for a value and a stored copy of that same value (a broadcast, a
checkpoint read back, a cache hit: `bitwise`). Under another identity the same SCF and
the same contractions round differently in their last bits (a cluster's
MKL against a workstation's MKL on its SSE4.2 path moved the cc-pVDZ mean
fields by 6e-14 to 4e-13 Ha), so the pin holds to the bar of its quantity
class instead (`PIN_CROSS_BLAS_BARS`): an energy, a force or gradient, or a
quasiparticle energy or pole strength read off a Pade continuation, which
amplifies the fit's last bits.

The identity has two parts, since numpy and pyscf's C kernels need not
link the same library: numpy's BLAS as threadpoolctl reports it (vendor and
version, MKL's instruction-set code path, OpenBLAS's core type), and the
BLAS pyscf's libnp_helper links.
"""
import ctypes
import functools
import json
import pathlib
import platform
import re
import subprocess

import numpy as np
import pyscf.lib
from threadpoolctl import threadpool_info

from src.Base.numerics.constants import (HARTREE_TO_EV, PIN_CROSS_BLAS_BAR,
                                         PIN_CROSS_BLAS_CONTINUED_BAR,
                                         PIN_CROSS_BLAS_GRADIENT_BAR,
                                         PIN_CROSS_BLAS_ROOT_BAR,
                                         PIN_CROSS_BLAS_VECTOR_BAR,
                                         PIN_DAVIDSON_ROOT_FACTOR, PIN_SAME_ULPS)

#: What a pin of each quantity class may miss by when the running BLAS is not
#: the recording one: Hartree, Ha/Bohr on a gradient, the same number of a
#: dimensionless pole strength, and for 'vector' (vectors and orbital
#: coefficients, compared after `align_signs`) and 'root' (roots and matrix
#: elements, Ha) the class bars.
PIN_CROSS_BLAS_BARS = {'energy': PIN_CROSS_BLAS_BAR,
                       'gradient': PIN_CROSS_BLAS_GRADIENT_BAR,
                       'continued': PIN_CROSS_BLAS_CONTINUED_BAR,
                       'vector': PIN_CROSS_BLAS_VECTOR_BAR,
                       'root': PIN_CROSS_BLAS_ROOT_BAR}
#: The bar per pin unit.
PIN_UNIT_SCALE = {'Ha': 1.0, 'eV': HARTREE_TO_EV, '1': 1.0}
#: The thread caps of a subprocess gate: one thread count for every pool.
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: The same caps without OMP_NUM_THREADS, for a gate that sets it per probe.
THREAD_CAPS_WITHOUT_OMP = {name: cap for name, cap in THREAD_CAPS.items()
                           if name != 'OMP_NUM_THREADS'}
#: What names a BLAS among a library's linked libraries.
BLAS_NAME = re.compile(r'openblas|mkl|accelerate|flexiblas|blis|libblas',
                       re.IGNORECASE)


class MKLVersion(ctypes.Structure):
    """mkl_types.h MKLVersion, filled by MKL_Get_Version."""
    _fields_ = [('MajorVersion', ctypes.c_int), ('MinorVersion', ctypes.c_int),
                ('UpdateVersion', ctypes.c_int),
                ('ProductStatus', ctypes.c_char_p), ('Build', ctypes.c_char_p),
                ('Processor', ctypes.c_char_p), ('Platform', ctypes.c_char_p)]


def mkl_code_path(filepath):
    """The instruction set MKL at `filepath` dispatched to, or None."""
    try:
        version = MKLVersion()
        ctypes.CDLL(filepath).MKL_Get_Version(ctypes.byref(version))
        return version.Processor.decode()
    except (OSError, AttributeError):
        return None


def numpy_blas():
    """numpy's BLAS: vendor, version and code path, one string per library."""
    out = []
    for lib in threadpool_info():
        if lib.get('user_api') != 'blas':
            continue
        name = f"{lib['internal_api']} {lib.get('version')}"
        if lib['internal_api'] == 'mkl':
            name += f" ({mkl_code_path(lib['filepath'])})"
        elif lib.get('architecture'):
            name += f" ({lib['architecture']})"
        out.append(name)
    return '; '.join(sorted(out)) or 'none found'


def pyscf_blas():
    """The BLAS libraries pyscf's libnp_helper links, by file name."""
    helper = sorted(pathlib.Path(pyscf.lib.__file__).parent.glob(
        'libnp_helper.*'))
    if not helper:
        return 'no libnp_helper'
    tool = ['otool', '-L'] if platform.system() == 'Darwin' else ['ldd']
    try:
        listed = subprocess.run(tool + [str(helper[0])], capture_output=True,
                                text=True).stdout
    except OSError:
        return 'not readable'
    names = {pathlib.Path(line.split()[0]).name for line in
             listed.splitlines()[1:] if line.split()
             and BLAS_NAME.search(line.split()[0])}
    return '; '.join(sorted(names)) or 'none linked'


@functools.lru_cache(maxsize=None)
def _identity():
    return (('numpy', numpy_blas()), ('pyscf', pyscf_blas()),
            ('machine', platform.machine()))


def blas_identity():
    """The BLAS identity of this process, as a pin file records it."""
    return dict(_identity())


def recorded_blas(path):
    """The 'blas' field of the pin file at `path`; a file without one is
    refused, since its pins cannot say where they hold bitwise."""
    record = json.loads(pathlib.Path(path).read_text())
    if 'blas' not in record:
        raise KeyError(f'{path} records no BLAS identity')
    return record['blas']


def align_signs(x, ref):
    """`x` with the sign of each column (of a 1-D array, the whole) fixed to
    `ref`'s: an eigenvector or an orbital has no sign of its own. Arrays of
    another shape than `ref`'s, and arrays beyond two dimensions, come back
    as they were."""
    x, ref = np.asarray(x), np.asarray(ref)
    if x.shape != ref.shape or x.ndim not in (1, 2) or x.dtype.kind != 'f':
        return x
    return x * np.where(np.sum(x * ref, axis=0) < 0.0, -1.0, 1.0)


def class_bar(quantity='energy', unit='Ha'):
    """The bar of `quantity` in `unit`, whatever the BLAS: for two numbers that
    are the same quantity by two routes which reassociate their sums (ranks,
    layouts) and so are never bitwise."""
    return PIN_CROSS_BLAS_BARS[quantity] * PIN_UNIT_SCALE[unit]


def assert_within_class(got, want, quantity='root', unit='Ha', label=None):
    """Raise unless every element of `got` is within the class bar of `want`
    (vectors after sign alignment), under any BLAS."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape, (label, got.shape, want.shape)
    if quantity == 'vector':
        got = align_signs(got, want)
    gap = np.abs(got - want).max(initial=0.0)
    bar = class_bar(quantity, unit)
    assert gap <= bar, (label, f'max |got - want| {gap:.3e} {unit} against '
                        f'the {quantity} bar {bar:.1e}')


def same_bar(got, want, unit='Ha'):
    """What two computations of one number may differ by on one machine:
    PIN_SAME_ULPS ulp of the larger of their largest magnitude and one `unit`
    (module docstring)."""
    scale = max(PIN_UNIT_SCALE[unit],
                float(np.abs(np.asarray(got, float)).max(initial=0.0)),
                float(np.abs(np.asarray(want, float)).max(initial=0.0)))
    return PIN_SAME_ULPS * np.finfo(float).eps * scale


def pin_bar(blas, unit='Ha', quantity='energy'):
    """0 where `blas` is this process's identity (the comparison is then
    `same_bar`'s few ulp), else the cross-BLAS bar of `quantity` ('energy',
    'gradient' or 'continued') in `unit` ('Ha' and Ha/Bohr, 'eV', or '1')."""
    if blas == blas_identity():
        return 0.0
    return class_bar(quantity, unit)


def bar_under(blas, tight, unit='Ha', quantity='energy'):
    """`tight`, the bar measured on the recording workstation, where `blas` is
    this process's identity; the cross-BLAS bar of `quantity` elsewhere."""
    return tight if blas == blas_identity() else pin_bar(
        blas, unit, quantity=quantity)


def matches(got, pin, blas, unit='Ha', quantity='energy'):
    """Whether `got` is the pin: within `same_bar` (a few ulp) under the
    recording BLAS, within `pin_bar` of its quantity class elementwise under
    another. A NaN matches nothing."""
    bar = pin_bar(blas, unit, quantity=quantity)
    got, pin = np.asarray(got, float), np.asarray(pin, float)
    if got.shape != pin.shape:
        return False
    if quantity == 'vector':
        got = align_signs(got, pin)
    if bar == 0.0:
        bar = same_bar(got, pin, '1' if quantity == 'vector' else unit)
    return bool(np.all(np.abs(got - pin) <= bar))


def miss(got, pin, blas, unit='Ha', quantity='energy'):
    """One line on how `got` stands against the pin: max |got - pin| against
    the bar, or the two shapes where they differ."""
    got, pin = np.asarray(got, float), np.asarray(pin, float)
    if got.shape != pin.shape:
        return f'shape {got.shape} against the pin\'s {pin.shape}'
    bar = pin_bar(blas, unit, quantity=quantity)
    if quantity == 'vector':
        got = align_signs(got, pin)
    if bar == 0.0:
        bar = same_bar(got, pin, '1' if quantity == 'vector' else unit)
    return (f'max |got - pin| {np.abs(got - pin).max():.3e} {unit} against '
            f'the {quantity} bar {bar:.1e}')


def assert_pin(got, pin, blas, unit='Ha', quantity='energy', label=None):
    """Raise unless `matches`, naming `label` and the `miss`."""
    assert matches(got, pin, blas, unit, quantity=quantity), (
        label, miss(got, pin, blas, unit, quantity=quantity))


#: The BLAS identity the bitwise claims of the test suite were recorded on
#: (the pin file of the one fit, which is the workstation's).
RECORDING_BLAS = recorded_blas(
    pathlib.Path(__file__).resolve().parent / 'one_fit_pins.json')


def assert_same(got, want, unit='Ha', quantity='energy', label=None):
    """`got` is `want` where the claim is that two routes, two layouts, two
    runs or two builds of one mean field return the SAME number: to a few ulp
    (`same_bar`) under the recording BLAS (`RECORDING_BLAS`), within the
    cross-BLAS bar of its quantity class elsewhere, where one library rounds
    by call shape, by free memory or by thread count and two SCFs of the same
    molecule differ in their last bits. A value against a stored copy of
    itself is `bitwise` instead."""
    assert_pin(got, want, RECORDING_BLAS, unit, quantity=quantity,
               label=label)


def assert_same_tree(got, want, unit='Ha', quantity='energy', label=None):
    """`assert_same` through the dicts, lists and tuples of a record: every
    number as `assert_same` reads it (a NaN is the same as a NaN), everything
    else with `==`."""
    if isinstance(got, dict) and isinstance(want, dict):
        assert got.keys() == want.keys(), label
        for key in got:
            assert_same_tree(got[key], want[key], unit, quantity,
                             label=(label, key))
        return
    if (isinstance(got, (list, tuple)) and isinstance(want, (list, tuple))
            and not isinstance(got, np.ndarray)):
        assert len(got) == len(want), label
        for i, (a, b) in enumerate(zip(got, want)):
            assert_same_tree(a, b, unit, quantity, label=(label, i))
        return
    try:
        x, y = np.asarray(got, float), np.asarray(want, float)
    except (TypeError, ValueError):
        assert got == want, label
        return
    both = np.isnan(x) & np.isnan(y)
    assert_same(np.where(both, 0.0, x), np.where(both, 0.0, y), unit, quantity,
                label=label)


def assert_archive_same(got, archived, unit='Ha', quantity='energy',
                        label=None):
    """`got` is what an archived tree (`git archive` of a baseline commit)
    returned: to a few ulp where the BLAS is the recording workstation's, within
    the class bar elsewhere, the archive having run under the same library as
    the tree it is compared with. Every archive comparison goes through this
    one routine; `quantity` is the class of what is compared ('root' for roots
    and matrix elements, 'vector' for vectors and orbital coefficients)."""
    assert_same(got, archived, unit, quantity, label=label)


def solution_pairs(got, want):
    """(class, got, want) of each array of two eigen-solutions: a 1-D array
    is roots, a 2-D array vectors."""
    assert len(got) == len(want)
    return [('root' if np.ndim(y) == 1 else 'vector', x, y)
            for x, y in zip(got, want)]


def same_solution(got, want):
    """Whether two eigen-solutions (tuples of arrays) are one: to a few ulp
    under the recording BLAS, elsewhere each root array within the 'root' bar and
    each vector array within the 'vector' bar after sign alignment."""
    return all(matches(x, y, RECORDING_BLAS, '1' if q == 'vector' else 'Ha',
                       quantity=q) for q, x, y in solution_pairs(got, want))


def solution_miss(got, want):
    """`miss` of each array of two eigen-solutions, one line."""
    return '; '.join(miss(x, y, RECORDING_BLAS, '1' if q == 'vector' else 'Ha',
                          quantity=q) for q, x, y in solution_pairs(got, want))


def davidson_root_bar(conv_tol):
    """Ha: what two Davidson solves of one problem, split over ranks or not,
    may differ by in a root. Each root is within its residual of an eigenvalue
    (|r| <= `conv_tol`), so two of them differ by at most
    PIN_DAVIDSON_ROOT_FACTOR conv_tol; the bar follows the tolerance the
    solve was asked for and is no fit to a measured miss."""
    return PIN_DAVIDSON_ROOT_FACTOR * conv_tol


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included: for a
    value and a stored copy of that same value only (a broadcast, a lockstep,
    a checkpoint read back, a cache hit), never for two computations
    (`assert_same`)."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def bitwise_arrays(a, b):
    """Every array of `a` holds the bits of the same array of `b` (copies
    only, as `bitwise`)."""
    return len(a) == len(b) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        for x, y in zip(a, b))
