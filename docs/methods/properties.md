# Potential-energy surfaces and properties

**Potential-energy surfaces** — `src/properties/` computes FROM a surface
rather than BY one. `potential_energy_surface(mol, scf_factory, *,
ground_state, excitation=None, environment=None, ...)` dispatches on the
DECLARED physics (`src.Base.declaration`:
`GroundState`, `Excitation`, `ChargedExcitation`, `QPStates`) to the gradient
chain that realizes it and records the realization; `compare_surfaces` refuses
to difference two surfaces that do not share a ground-state functional and
environment. `calc_vertical_excitation`, `calc_emission_energy`,
`calc_adiabatic_excitation` and `calc_adiabatic_gap` build both surfaces of a
difference from one `SurfaceSpec`, so the functional under an excited state
and under its ground state cannot silently be two different functionals.
`optimize`/`relax` walk Cartesian RFO/BFGS with the rigid-body directions
projected out, or [geomeTRIC](https://github.com/leeping/geomeTRIC) when it is
installed; `vibronic` gives normal modes and Huang-Rhys factors by two
independent routes; `conformers` the Boltzmann-weighted average over torsional
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
