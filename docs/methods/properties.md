# Potential-energy surfaces and properties

**Potential-energy surfaces** — `src/properties/` computes from a surface
rather than by one. `potential_energy_surface(mol, scf_factory, *,
ground_state, excitation=None, environment=None, ...)` dispatches on the
declared physics (`src.Base.declaration`:
`GroundState`, `Excitation`, `ChargedExcitation`, `QPStates`) to the gradient
chain that realizes it and records the realization; `compare_surfaces` refuses
to difference two surfaces that do not share a ground-state functional and
environment. `calc_vertical_excitation`, `calc_emission_energy`,
`calc_adiabatic_excitation` and `calc_adiabatic_gap` build both surfaces of a
difference from one `SurfaceSpec`, so the functional under an excited state
and under its ground state cannot silently be two different functionals.
`optimize`/`relax` walk Cartesian RFO/BFGS with the rigid-body directions
projected out, or [geomeTRIC](https://github.com/leeping/geomeTRIC) when it is
installed. A surface keeps a few choices fixed while it is walked (which
orbitals get a full quasiparticle solve, for one); after the walk converges
they can be chosen again at the minimum and the walk repeated until the
energy stops moving (`refreeze`). An excited-state walk starts from a tiny
fixed random displacement of the input geometry, so a symmetric start can
still reach a minimum of lower symmetry. `vibronic` gives normal modes and
Huang-Rhys factors by two independent routes; `conformers` the Boltzmann-weighted average over torsional
minima a soft emitter has; `spin_orbit` and `nonadiabatic` the two couplings a
`rates` Marcus-Levich-Jortner or golden-rule rate needs; `characters` labels a
BSE root by its charge-transfer weight; `hessian` the nuclear Hessian by
central differences of the analytic gradient, for a route pyscf's own
analytic Hessian cannot serve (on an interpolated mean field it asks for a
fine enough interpolation grid and refuses a coarse one, whose surface is
too rough for a finite difference); `mode_hessian` the force constants
along a few chosen normal modes only, which gives the frequency of one local
vibration on an excited state for a handful of gradients. See
`examples/15_excited_state_geometry_optimization.py` for the one entry point,
`calc_adiabatic_excitation`, driving an excited-state relaxation end to end,
and `examples/17_spin_orbit_coupling.py` for `spin_orbit` read at the two
minima `calc_adiabatic_gap` already relaxes (El-Sayed's rule, computed rather
than assumed) plus the Herzberg-Teller dV/dq scan over `vibronic`'s
ground-state modes that `rates.spin_vibronic_rate` needs for an
El-Sayed-forbidden pair, where the Condon term alone is not the whole story.

**Spin-vibronic coupling** — intersystem crossing between a singlet and a
triplet of the same orbital character (both n -> pi*, or both pi -> pi*) is
nearly forbidden by El-Sayed's rule: their spin-orbit coupling is close to
zero at a fixed geometry. A vibration can still switch it on, by mixing a
nearby second triplet (or singlet) of a different character into the pair.
`spin_vibronic_coupling` gives how fast the coupling between S1 and T1
changes along each nuclear motion, through those higher states, from one
excited-state calculation and without displaced geometries. This matters when
the direct coupling is small and T2 lies close to T1: forward intersystem
crossing of El-Sayed-forbidden pairs, and reverse intersystem crossing in
TADF emitters, where T2 a few k_B T above T1 often carries the process. The
derivative, projected on normal modes, is the Herzberg-Teller input of
`rates.spin_vibronic_rate`, and `rates.photoluminescence` takes the T2 rates
and the T1-T2 gap to weight a thermally populated T2.

```python
import numpy as np
from pyscf import gto, scf

from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from src.properties.spin_orbit import soc_operator_mo
from src.properties.spin_vibronic import spin_vibronic_coupling


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


# twisted formaldehyde with a stretched C=O: T2 lies close above T1
mol = gto.M(atom='C 0 0 0; O 0.05 -0.03 1.45; H 0.12 0.943 -0.588; '
                 'H -0.20 -0.943 -0.588', basis='cc-pvdz')
chain = ExcitedStateChain(mol, rhf, mf=rhf(mol), solver='davidson',
                          bse_adjoint='grid', nroots=3)
S1, S2, T1, T2 = ('singlet', 0), ('singlet', 1), ('triplet', 0), ('triplet', 1)
ev = StateManifold(chain, states=(S1, S2, T1, T2)).evaluate(
    couplings=((S2, S1), (T2, T1)))
out = spin_vibronic_coupling(ev, soc_operator_mo(ev.mf, ev.mol), S1, T1,
                             singlet_paths=(S2,), triplet_paths=(T2,))
print('|<S1|H_SO|T1>| =', out['v0'], 'Hartree')
print('|dV/dR| =', np.linalg.norm(out['dv_cart']), 'Hartree/Bohr')
```

`out['dv_cart']` holds the derivative for every atom and direction; pass
`modes`, `masses` and `omega` from `vibronic.normal_modes` at a minimum to get
it per normal mode as well.

**Vibronic band shapes** — `band_shape(s_k, omega_k, e00, temperature,
energies, *, gaussian_fwhm, lorentzian_fwhm)` gives the normalized absorption
(E x FC) and emission (E^3 x FC) spectra of the displaced-oscillator model
from `vibronic`'s Huang-Rhys factors, the 0-0 energy and the temperature, with
their peaks, FWHMs, the Stokes shift, and the emission per unit wavelength with
its FWHM in nm. Both broadenings are required arguments: a computed width is
only comparable with a measured one when the broadening added to it is stated.
The Franck-Condon density is `rates.fc_weighted_dos`, the generating function
the golden-rule rates use, which takes a homogeneous Lorentzian (integrated on
the real time axis, with an Euler-Maclaurin correction at the kink of
e^{-gamma |t|}) and returns an exact zero where the saddle-point exponent has
underflowed, the far side of a cold band. `band_grid` builds an energy grid
that covers and resolves both bands, and `gaussian_limit_fwhm` the width of
the second cumulant alone. See `examples/20_vibronic_band_shape.py` for
formaldehyde S1 end to end, both Huang-Rhys routes.
