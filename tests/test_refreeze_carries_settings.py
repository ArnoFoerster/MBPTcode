"""`ExcitedStateChain.refreeze` carries EVERY setting.

A refrozen chain is the same surface at a new geometry: only the frozen
conventions are re-derived, every CHOICE travels verbatim. The choices are
the physics arguments and the one `Numerics`; the refrozen chain holds the
same object -- the resolved tau count and the grown quadrature travel beside
it (`CarriedGrids`), so its identity is the same before and after -- and a
setting added to it travels without a line in `refreeze`. A `refreeze` that
mirrors the constructor's forty keywords would drop `scissor`, `n_poles`,
`sop_stride` and `comm`, so a surface built with `scissor='calibrate'`
would silently fall back to no scissor tier at the first refreeze of a
relaxation.
"""

import numpy as np
from pyscf import gto, scf

from src.Base.numerics import FLAT_FIELDS, Numerics
from src.gradients.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics


def _water(shift=0.0):
    return gto.M(atom=f'O 0 0 {shift}; H 0 0.757 0.587; H 0 -0.757 0.587',
                 basis='cc-pvdz', verbose=0)


def _factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    return mf


def test_every_choice_travels():
    numerics = g1_numerics(_water(), Numerics(
        qp={'n_poles': 7, 'sop_stride': 3, 'window': 1, 'scissor': 'calibrate',
            'residue_route': 'sop'},
        quadrature={'nfreq_cd': 32, 'ntau_gw': 18}, bse={'solver': 'dense'}))
    chain = ExcitedStateChain(_water(), _factory, spin='triplet',
                              bse_tda=True, numerics=numerics)
    fresh = chain.refreeze(_water(0.02))
    assert (fresh.spin, fresh.bse_tda) == ('triplet', True)
    assert fresh.environment is chain.environment
    assert fresh.settings is numerics
    assert fresh.settings.identity() == chain.settings.identity()
    assert fresh.ntau_w == chain.ntau_w
    assert fresh.conventions.nfreq_cd >= chain.conventions.nfreq_cd
    for name in FLAT_FIELDS:
        if name != 'nfreq_cd' and hasattr(chain, name):
            np.testing.assert_equal(getattr(fresh, name),
                                    getattr(chain, name), err_msg=name)
    assert fresh.scissor == 'calibrate' and fresh.n_poles == 7 \
        and fresh.sop_stride == 3
