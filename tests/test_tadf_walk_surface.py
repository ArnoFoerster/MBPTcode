"""The walks run on a mean-field surface smooth at their step.

The stages relax S0, S1 and T1 on E_KS + Omega with an ISDF-K mean field.
The ISDF-K exchange turns each atom's interpolation cloud with its covariant
frame, so on a grid below ISDF_HESSIAN_MIN_GRID its energy is rough on the
scale of a walk's last steps although its force is exact: the force
differences and the energy differences then disagree on the curvature, the
quasi-Newton model built from the forces mispredicts the energy, and the walk
stalls with steps along the force raising the energy (DMABN S1 at G1:
|g|max ~7e-4 Ha/Bohr for 70 steps).

The check: along formaldehyde's in-plane C_y at NUCLEAR_FD_STEP, through the
run's own mean-field factory on the command line's default mean-field grid,
the curvature from the forces (F(+h) - F(-h)) / 2h and from the energies
(E(+h) + E(-h) - 2 E(0)) / h^2 agree to HESSIAN_FD_ASYMMETRY_TOL relative.
Measured, LRC-wPBEh/cc-pVDZ: 1.6e-2 at G1, 5.9e-6 at G3.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (HESSIAN_FD_ASYMMETRY_TOL,
                                         ISDF_HESSIAN_MIN_GRID, NUCLEAR_FD_STEP)
from src.photophysics.stages import ground_minimum
from src.photophysics.store import StoreRefused
from src.properties.hessian import require_hessian_grid
from src.gradients.mean_field_surface import mean_field_force
from tests.photophysics_cases import run_of

#: formaldehyde off its minimum, C2v: C_y breaks the symmetry
CH2O = ('C 0.0 0.0 -0.6035; O 0.0 0.0 0.7349; H 0.0 0.9753 -1.1057; '
        'H 0.0 -0.9753 -1.1057')


def curvatures(factory, mol, ia, x, h):
    """(force-difference, energy-difference) curvature along (ia, x)."""
    out = {}
    for sign in (-1, 0, 1):
        coords = mol.atom_coords().copy()
        coords[ia, x] += sign * h
        m = mol.copy()
        m.set_geom_(coords, unit='Bohr')
        m.build(False, False)
        mf = factory(m)
        out[sign] = (mf.e_tot, np.asarray(mean_field_force(mf))[ia, x])
    k_force = (out[1][1] - out[-1][1]) / (2 * h)
    k_energy = (out[1][0] + out[-1][0] - 2 * out[0][0]) / h ** 2
    return k_force, k_energy, mf


def relative_mismatch(k_force, k_energy):
    """|k_force - k_energy| relative to k_force."""
    return abs(k_force - k_energy) / abs(k_force)


def test_walk_mean_field_is_smooth_at_the_step(tmp_path):
    run = run_of(tmp_path)
    assert run.declaration.numerics.scf.grid_level == ISDF_HESSIAN_MIN_GRID
    mol = gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)
    k_force, k_energy, mf = curvatures(run.factory, mol, 0, 1,
                                       NUCLEAR_FD_STEP)
    rel = relative_mismatch(k_force, k_energy)
    assert rel < HESSIAN_FD_ASYMMETRY_TOL, (
        f'the walks\' mean field is rough at h = {NUCLEAR_FD_STEP}: curvature '
        f'{k_force:.5f} from the forces, {k_energy:.5f} from the energies '
        f'(relative {rel:.2e})')
    require_hessian_grid(mf)


def test_the_coarse_grid_is_still_asked_for_by_name(tmp_path):
    """--scf-grid G1 asks for the coarse grid by name, and the check above
    fails on it."""
    run = run_of(tmp_path, '--scf-grid', 'G1')
    mol = gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)
    k_force, k_energy, _ = curvatures(run.factory, mol, 0, 1,
                                      NUCLEAR_FD_STEP)
    assert relative_mismatch(k_force, k_energy) > HESSIAN_FD_ASYMMETRY_TOL


def test_a_record_on_another_mean_field_is_refused(tmp_path):
    """R0, the Hessian and the walks stand on the mean field s0 relaxed on,
    so a run declaring another one refuses the store rather than walking a
    GH surface from a G1 minimum."""
    coarse = run_of(tmp_path, '--scf-grid', 'G1')
    assert coarse.declaration.mean_field_name == 'ISDF-K G1'
    coarse.write('s0', {'coords_bohr': coarse.mol_in.atom_coords()})
    run = run_of(tmp_path)
    assert run.declaration.mean_field_name == f'ISDF-K {ISDF_HESSIAN_MIN_GRID}'
    with pytest.raises(StoreRefused, match='mean_field'):
        ground_minimum(run)
