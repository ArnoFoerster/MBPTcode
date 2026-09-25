"""`ExcitedStateChain.refreeze` carries EVERY constructor setting.

A refrozen chain is the same surface at a new geometry: only the frozen
conventions are re-derived, every CHOICE travels verbatim. Before this test,
`refreeze` dropped `scissor`, `n_poles`, `sop_stride` and `comm`, so a surface
built with `scissor='calibrate'` silently fell back to no scissor tier at the
first refreeze of a relaxation -- a different surface from the one the record
named. Negative control: with the four keywords removed from the constructor
call in `refreeze` (the pre-fix code), `test_every_constructor_choice_travels`
fails on `scissor` ('calibrate' vs None).
"""
import inspect

import numpy as np
import pytest
from pyscf import gto, scf

from src.gradients.excited_state import ExcitedStateChain

#: Constructor parameters that are not carried by design: the geometry, the
#: factory, the reference mean field (it belongs to the old geometry), the
#: factorization (rebuilt at the new geometry) and the state (refreeze hands
#: over the FOLLOWED root, tested elsewhere).
NOT_CARRIED = {'self', 'mol', 'scf_factory', 'mf', 'factorization', 'state'}
#: Parameters stored under another attribute name.
STORED_AS = {'frames': 'frames_mode'}


def _water(shift=0.0):
    return gto.M(atom=f'O 0 0 {shift}; H 0 0.757 0.587; H 0 -0.757 0.587',
                 basis='cc-pvdz', verbose=0)


def _factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    return mf


def test_every_constructor_choice_travels():
    chain = ExcitedStateChain(_water(), _factory, spin='triplet', qp_window=1,
                              scissor='calibrate', outside='scissor',
                              n_poles=7, sop_stride=3, residue_route='sop',
                              nfreq_cd=32, ntau_gw=18, solver='dense')
    fresh = chain.refreeze(_water(0.02))
    params = [n for n in inspect.signature(ExcitedStateChain.__init__).parameters
              if n not in NOT_CARRIED]
    for name in params:
        attr = STORED_AS.get(name, name)
        if not hasattr(chain, attr):
            pytest.fail(f'{name} is stored under no attribute; the test cannot see it')
        # dicts of arrays (radii, counts) and plain values alike
        np.testing.assert_equal(getattr(chain, attr), getattr(fresh, attr),
                                err_msg=name)
    assert fresh.scissor == 'calibrate' and fresh.n_poles == 7 \
        and fresh.sop_stride == 3 and fresh.outside == 'scissor'
