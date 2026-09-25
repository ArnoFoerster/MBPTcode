"""A vibrational analysis built by differencing the analytic gradient

`numerical_hessian` central-differences the analytic FORCE, atom by atom and
Cartesian direction by direction, rather than differencing the energy twice:
the noise being divided by the step is the gradient's, which sits far below
the energy's own noise, so a comfortably large step (`NUCLEAR_FD_STEP`) still
gives a clean Hessian. It exists for routes that have no analytic Hessian at
all -- an ISDF mean field's force is exact but pyscf's analytic Hessian
differentiates the FITTED interaction twice and knows nothing of the
interpolation points -- but it needs no ISDF to be used, as this plain RHF
run below shows, cross-checked against pyscf's own analytic Hessian on the
same mean field.

    python examples/16_numerical_hessian_from_gradient.py
"""
import os
import sys

from pyscf import gto, scf
from pyscf.hessian import thermo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_CM
from src.properties.hessian import numerical_hessian
from src.properties.vibronic import normal_modes

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)


def factory(m):
    mf = scf.RHF(m)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


mf = factory(mol)
hess = numerical_hessian(mol, factory)                  # 6*natm displaced gradients
omega, modes, masses, _ = normal_modes(mf, mol=mol, hess=hess)
ours = sorted(omega * HARTREE_TO_CM)

ref = sorted(thermo.harmonic_analysis(
    mol, mf.Hessian().kernel())['freq_wavenumber'])

print(f"{'mode':>6s} {'from the gradient':>20s} {'pyscf analytic':>16s}")
for i, (a, b) in enumerate(zip(ours, ref)):
    print(f"{i + 1:6d} {a:17.2f} cm-1 {b:13.2f} cm-1")
