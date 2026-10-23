"""The GPU doubles kernels of ee_ri_fold (_DoublesGPU) against the CPU path,
spin-component scaled and not, singlet and triplet, one tile and one tile
per occupied index: apply with |Y|^2, fold_images with its derivative image,
and doubles_dot of one column with itself and of two columns at two
frequencies. Water 6-31G, cc-pvdz-ri.

Without CuPy (this workstation) the kernels run on the CPU: CPUKernels is a
stand-in for the part of CuPy _DoublesGPU uses, NumPy arrays and each
ElementwiseKernel's CUDA text compiled as the body of a C loop. This checks
the kernel text's arithmetic, not the device; with CuPy the same checks run
on the GPU as well.

Note: doubles_dot on the GPU dropped the spin-component
weights (kernel/CPU 0.848 singlet, 0.865 triplet at (1.2, 0.33)).
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import ADC2_SPIN_SCALING, EE_RI_TILE_GB
from src.Base.pyscf.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_ri_fold
from tests.adc_common import WATER_ORIGIN, rhf_df

CC = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
_C_TYPES = {'float64': ('double', ctypes.c_double),
            'int64': ('long long', ctypes.c_longlong),
            'bool': ('bool', ctypes.c_bool)}


def _params(text):
    """[(raw, dtype, name)] of an ElementwiseKernel parameter list."""
    out = []
    for p in text.split(','):
        words = p.split()
        raw = words[0] == 'raw'
        out.append((raw, words[-2], words[-1]))
    return out


class _Kernel:
    """One ElementwiseKernel compiled for the CPU: the body runs once per
    element i of the output, as on the device."""

    def __init__(self, workdir, in_params, out_params, body, name):
        self.ins, (self.out,) = _params(in_params), _params(out_params)
        args = ['long long n__']
        for raw, dtype, pname in self.ins:
            c = _C_TYPES[dtype][0]
            args.append(f'const {c}* {pname}' if raw else f'{c} {pname}')
        args.append(f'{_C_TYPES[self.out[1]][0]}* out__')
        oc, oname = _C_TYPES[self.out[1]][0], self.out[2]
        src = ('#include <stdbool.h>\n'
               f'void {name}({", ".join(args)}) {{\n'
               '  for (long long i = 0; i < n__; ++i) {\n'
               f'    {oc} {oname};\n{body}\n    out__[i] = {oname};\n  }}\n}}\n')
        c_file = os.path.join(workdir, name + '.c')
        so = os.path.join(workdir, name + '.so')
        with open(c_file, 'w') as fh:
            fh.write(src)
        subprocess.run([CC, '-O1', '-shared', '-fPIC', '-o', so, c_file], check=True,
                       capture_output=True)
        self.fn = getattr(ctypes.CDLL(so), name)
        self.fn.restype = None

    def __call__(self, *args):
        *ins, out = args
        assert len(ins) == len(self.ins) and out.flags.c_contiguous
        cargs = [ctypes.c_longlong(out.size)]
        for (raw, dtype, _), a in zip(self.ins, ins):
            if raw:
                a = np.ascontiguousarray(a, float)
                cargs.append(a.ctypes.data_as(ctypes.c_void_p))
            else:
                cargs.append(_C_TYPES[dtype][1](a))
        cargs.append(out.ctypes.data_as(ctypes.c_void_p))
        self.fn(*cargs)
        return out


def cpu_kernels(workdir):
    """The CuPy stand-in (module docstring)."""
    return SimpleNamespace(
        asarray=lambda a: np.ascontiguousarray(a, float),
        asnumpy=np.asarray,
        empty_like=np.empty_like,
        ElementwiseKernel=lambda i, o, b, n: _Kernel(workdir, i, o, b, n))


@pytest.fixture(scope='module')
def factors():
    mf = rhf_df(WATER_ORIGIN, '6-31g')
    no = mf.mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    B = DFIntegrals.from_scf(mf.mol, mf).B_aa
    return ee_ri_fold.RIFactors.from_full(eps, B, no)


def _compare(F, spin, scaling, tile_gb):
    """Largest relative difference of the device path from the CPU path over
    the products of the module docstring."""
    cpu = ee_ri_fold.RIFoldADC2(F, spin, scaling=scaling, tile_gb=tile_gb)
    dev = ee_ri_fold.RIFoldADC2(F, spin, scaling=scaling, tile_gb=tile_gb,
                                device='gpu')
    rng = np.random.default_rng(3)
    U = rng.standard_normal((F.no * F.nv, 2))
    w1, w2 = 0.5 * cpu.d_min, 0.8 * cpu.d_min
    err = {}

    def rel(a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).max()
                     / max(np.abs(np.asarray(b)).max(), 1e-300))

    err['apply'] = max(rel(x, y) for x, y in zip(dev.apply(U, [w1, w2]),
                                                 cpu.apply(U, [w1, w2])))
    fd = dev.fold_images(U, [w1, w1], derivative=[False, True])
    fc = cpu.fold_images(U, [w1, w1], derivative=[False, True])
    err['fold_images'] = rel(fd[1][0], fc[1][0])
    err['derivative image'] = rel(fd[1][1], fc[1][1])
    u, v = U[:, 0], U[:, 1]
    err['doubles_dot (u, u)'] = rel(dev.doubles_dot(u, w1, u, w1),
                                    cpu.doubles_dot(u, w1, u, w1))
    err['doubles_dot (u, v)'] = rel(dev.doubles_dot(u, w1, v, w2),
                                    cpu.doubles_dot(u, w1, v, w2))
    return err


CASES = [(spin, scaling, tile_gb)
         for spin in ee_ri_fold.SPINS
         for scaling in (None, ADC2_SPIN_SCALING['scs'], (1.2, 0.33))
         for tile_gb in (EE_RI_TILE_GB, 1e-9)]


@pytest.mark.skipif(CC is None, reason='no C compiler for the kernel stand-in')
@pytest.mark.parametrize('spin,scaling,tile_gb', CASES)
def test_kernel_text_equals_the_cpu_path(factors, monkeypatch, spin, scaling,
                                         tile_gb):
    """The CUDA text, run on the CPU, gives the CPU path's products to 1e-12."""
    with tempfile.TemporaryDirectory() as work:
        monkeypatch.setattr(ee_ri_fold, 'cp', cpu_kernels(work))
        err = _compare(factors, spin, scaling, tile_gb)
    assert max(err.values()) < 1e-12, err


@pytest.mark.skipif(ee_ri_fold.cp is None, reason='CuPy is not importable')
@pytest.mark.parametrize('spin,scaling,tile_gb', CASES)
def test_device_equals_the_cpu_path(factors, spin, scaling, tile_gb):
    """On the GPU: the CPU path's products to 1e-10 (sums in another order)."""
    err = _compare(factors, spin, scaling, tile_gb)
    assert max(err.values()) < 1e-10, err
