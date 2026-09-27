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
analytic Hessian cannot serve. See
`examples/15_excited_state_geometry_optimization.py` for the one entry point,
`calc_adiabatic_excitation`, driving an excited-state relaxation end to end,
and `examples/17_spin_orbit_coupling.py` for `spin_orbit` read at the two
minima `calc_adiabatic_gap` already relaxes (El-Sayed's rule, computed rather
than assumed) plus the Herzberg-Teller dV/dq scan over `vibronic`'s
ground-state modes that `rates.spin_vibronic_rate` needs for an
El-Sayed-forbidden pair, where the Condon term alone is not the whole story.
