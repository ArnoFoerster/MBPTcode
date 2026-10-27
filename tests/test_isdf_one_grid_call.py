"""The serial ISDF-K handle and the distributed one place their grid with ONE call
(`ISDFJK.place_grid`): the same counts, radii, auxiliary basis, frames and the run's
`runtime_radii`. Before, the distributed build left `runtime_radii` out, so a run that
declared a run-time grid optimized radii serially and was refused over the ranks
(`UntabulatedISDFGrid`). Both builds are stopped at the grid call and its arguments
compared; nothing is fitted.
"""
import warnings

import pytest
from pyscf import gto, scf

from src.Base.factorization import isdf_jk as isdf_module
from src.Base.factorization.distributed_isdf_jk import DistributedISDFJK
from src.Base.factorization.isdf_jk import isdf_jk
from src.Base.numerics.numerics import Numerics
from tests.isdf_test_grid import g1


class Placed(Exception):
    """Raised by the recording grid call: the build stops there."""


def grid_call(build, monkeypatch):
    """The keyword arguments a build hands `isdf_grid`."""
    seen = {}

    def record(mol, **kwargs):
        seen.update(kwargs)
        raise Placed

    monkeypatch.setattr(isdf_module, 'isdf_grid', record)
    with pytest.raises(Placed):
        build()
    seen.pop('frames')
    return seen


@pytest.mark.parametrize('runtime_radii', [False, True])
def test_serial_and_distributed_builds_place_one_grid(runtime_radii, monkeypatch):
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz', verbose=0)
    numerics = Numerics().derive('a run-time grid', isdf={'runtime_radii': runtime_radii}) \
        if runtime_radii else Numerics()
    mf = scf.RHF(mol)
    handle = isdf_jk(mf, auxbasis='cc-pvdz-ri', counts=g1(mf), numerics=numerics).with_df
    serial = grid_call(handle.build, monkeypatch)
    distributed = grid_call(DistributedISDFJK(handle, comm=None).build, monkeypatch)
    assert serial == distributed
    assert serial['runtime_radii'] is runtime_radii
