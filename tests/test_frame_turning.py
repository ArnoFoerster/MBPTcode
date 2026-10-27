"""The turning report of the covariant ISDF frames.

`separable_ri.frame_turning` is first-order perturbation of each atom's
neighbour tensor; it must equal the angle the frames of `atomic_frames`
actually turn by under a small nuclear step. Measured: the ladder
formaldehyde's carbon (relative gap 3.15e-2) turns at 18.92 rad/Bohr, the
planar DMABN amine nitrogen at 16.22, both above FRAME_TURNING_RATE_BAR;
the oxygen at 0.37.
"""
import numpy as np
from pyscf import gto

from src.Base.numerics.constants import FRAME_TURNING_RATE_BAR
from src.Base.factorization.separable_ri import (atomic_frames, frame_turning,
                                                 frame_turning_record)

LADDER = ('C 0 0 -0.5296; O 0 0 0.6742; '
          'H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123')


def measured_rates(mol, h=1e-6):
    """The largest angle per Bohr any frame axis turns by under one step."""
    f0 = atomic_frames(mol)[0]
    rate = np.zeros(mol.natm)
    for j in range(mol.natm):
        for x in range(3):
            c = mol.atom_coords().copy()
            c[j, x] += h
            f1 = atomic_frames(mol.set_geom_(c, unit='Bohr',
                                             inplace=False))[0]
            dots = np.abs(np.einsum('iab,iab->ia', f0, f1)).clip(0.0, 1.0)
            rate = np.maximum(rate, np.arccos(dots).max(axis=1) / h)
    return rate


def test_the_perturbative_rate_is_the_frames_turning():
    mol = gto.M(atom=LADDER, basis='sto-3g', verbose=0)
    gap, rate = frame_turning(mol.atom_coords())
    assert np.allclose(rate, measured_rates(mol), rtol=1e-3)
    assert abs(rate[0] - 18.92) < 0.01 and abs(gap[0] - 3.15e-2) < 1e-4


def test_the_report_names_the_fast_atom_only():
    mol = gto.M(atom=LADDER, basis='sto-3g', verbose=0)
    record = frame_turning_record(mol)
    assert record['bar_rad_per_bohr'] == FRAME_TURNING_RATE_BAR
    assert [a['atom'] for a in record['above_bar']] == [0]
    assert record['above_bar'][0]['symbol'] == 'C'
    assert abs(record['max_rate_rad_per_bohr'] - 18.92) < 0.01
