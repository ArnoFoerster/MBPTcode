"""Shared helpers of the ADC and solvation test files: verdict printing, water geometries, tight SCF
builders, finite differences, a stand-in for a killed job."""
import numpy as np
from pyscf import gto, scf

from src.Base.constants import BOHR_TO_ANGSTROM
from src.Base.numerics import Numerics

#: Water, bent and non-symmetric so that every nuclear coordinate moves (Angstrom).
ATOMS = [('O', (0.02, -0.01, 0.1173)), ('H', (0.0, 0.7672, -0.4592)),
         ('H', (0.03, -0.7472, -0.4792))]
#: Water C2v, Angstrom, as an atom string: the 0.1173 / 0.7572 / -0.4692 geometry.
WATER_C2V = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Water C2v, Angstrom, as an atom string: oxygen at the origin.
WATER_ORIGIN = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
#: Formaldehyde, Angstrom, as an atom string.
H2CO = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.587; H 0 -0.94 -0.587'

JKFIT = 'cc-pvdz-jkfit'
AUX = 'cc-pvdz-ri'
NF = 1  # frozen core orbitals of water

# SCF settings: convergence on the energy and on the orbital gradient.
SCF_TOL_TIGHT, SCF_GRAD_TIGHT = 1e-15, 1e-12   # forces, finite-difference oracles
SCF_TOL_LOOP, SCF_GRAD_LOOP = 1e-13, 1e-10     # solvation loop, state models
SCF_GRAD_TILED = 1e-11                         # tiled force oracle
SCF_TOL_DF, SCF_GRAD_DF = 1e-12, 1e-8          # DF-RHF mean fields of the fold tests
SCF_MAX_CYCLE = 200
SCF_TOL_NEWTON = 1e-16                         # second-order polish after the first-order SCF

ROOT_CONV_TOL = 1e-12                          # ADC(2) state-model roots
SOLVATION_TIGHT_LOOP = dict(loop_tol_q=1e-10, loop_tol_energy=1e-12)  # solvation loop: charges, energy
SOLVATION_TIGHT = dict(numerics=Numerics(solvent=SOLVATION_TIGHT_LOOP))  # equilibrate's keywords

# Finite-difference steps: nuclear displacements (Bohr), field strength (a.u.).
FD_STEP_FORCE = 1e-4          # gas-phase forces
FD_STEP_SOLVATION = 5e-4      # solvated forces: every displaced point a full loop
FD_STEP_DENSITY = 1e-3        # five-point derivative in the field strength


def five_point(fn, step=FD_STEP_DENSITY):
    """The five-point central derivative of fn at 0."""
    return (fn(-2 * step) - 8 * fn(-step) + 8 * fn(step) - fn(2 * step)) / (12 * step)


def central(fn, step):
    """The two-point central derivative of fn at 0."""
    return (fn(step) - fn(-step)) / (2 * step)


class Stop(Exception):
    """Stands in for a killed job."""


def stop_after(k):
    """A solvation-loop `log` that kills the loop as cycle k + 1 reports, so
    its checkpoint holds k cycles."""
    def log(line):
        if line.startswith(f'solvation cycle {k + 1}:'):
            raise Stop
    return log


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def molecule(atoms, shift=None, basis='6-31g'):
    """Molecule of (symbol, xyz) atoms, xyz in Angstrom, moved by `shift` (Bohr, natm x 3)."""
    coords = np.array([xyz for _, xyz in atoms])
    if shift is not None:
        coords = coords + shift * BOHR_TO_ANGSTROM
    return gto.M(atom=[(s, tuple(c)) for (s, _), c in zip(atoms, coords)],
                 basis=basis, verbose=0)


def df_rhf(mol, conv_tol, conv_tol_grad):
    """DF-RHF (jkfit) with the given convergence, not yet run."""
    mf = scf.RHF(mol).density_fit(auxbasis=JKFIT)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = (
        conv_tol, conv_tol_grad, SCF_MAX_CYCLE, 0)
    return mf


def make_scf_tight(mol):
    """DF-RHF converged to an orbital gradient of 1e-12, not yet run."""
    return df_rhf(mol, SCF_TOL_TIGHT, SCF_GRAD_TIGHT)


def make_scf_loop(mol):
    """DF-RHF converged to an orbital gradient of 1e-10, not yet run."""
    return df_rhf(mol, SCF_TOL_LOOP, SCF_GRAD_LOOP)


def df_rhf_converged(mol):
    """Run DF-RHF (default auxiliary basis) to the fold tests' tolerances."""
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = SCF_TOL_DF
    mf.conv_tol_grad = SCF_GRAD_DF
    mf.kernel()
    return mf


def rhf_df(atom, basis):
    """A tightly converged DF-RHF mean field."""
    return df_rhf_converged(gto.M(atom=atom, basis=basis, verbose=0))


def converge_newton(mf, dm0=None):
    """mf converged to an orbital gradient of 1e-12 (first order, then second order)."""
    mf.conv_tol, mf.conv_tol_grad = SCF_TOL_TIGHT, SCF_GRAD_TIGHT
    mf.kernel(dm0=dm0)
    m = mf.newton()
    m.conv_tol, m.conv_tol_grad = SCF_TOL_NEWTON, SCF_GRAD_TIGHT
    m.kernel(mf.mo_coeff, mf.mo_occ)
    return m
