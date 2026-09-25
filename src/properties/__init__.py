"""Everything that is computed FROM a potential-energy surface rather than by
one: geometries, vibronic spectra, state labels and transfer rates.

The layer knows one interface, `PotentialEnergySurface` -- a total energy over
the nuclei, its gradient, and the method's own way of rebuilding what it froze
at a reference geometry -- and nothing about how that energy is obtained. A
BSE@GW excitation, a dRPA ground state and a downfolded active-space root are
the same object here, and a surface that has no analytic gradient yet is driven
through `FiniteDifferenceGradient` instead of being excluded.

    surface     the protocol, and the finite-difference gradient wrapper
    optimize    Cartesian RFO/BFGS with the rigid-body directions projected
                out, geomeTRIC when it is installed, and the refreeze outer
                loop that measures the frozen conventions' own drift
    vibronic    normal modes, Huang-Rhys factors by two independent routes,
                adiabatic gaps and the Marcus reorganization energy
    conformers  the torsional minima a soft emitter has, their Boltzmann
                populations, and a property averaged over them
    characters  the whole BSE spectrum at one geometry and its charge-transfer
                weights, for labelling roots
    rates       Marcus-Levich-Jortner and Marcus golden-rule rates, with the
                electronic coupling as an argument
    spin_orbit  <S_I|H_SO|T_J> over any solver's Casida vectors -- the coupling
                `rates` was written to receive
    nonadiabatic
                <Psi_I|d/dR Psi_J> from overlaps of those same vectors at
                displaced geometries -- the other coupling `rates` needs, and
                the only one here that costs a gradient rather than a property
    surfaces    the ONE entry point that builds a surface from its declared
                physics, records the realization that computed it, and refuses
                a comparison of two that cannot be differenced
    excitations the vertical, emission and adiabatic energies of a state, with
                BOTH surfaces of every difference built from one `SurfaceSpec`
                -- so the functional under the excited state and the functional
                under the ground state cannot be two functionals
    hessian     the nuclear Hessian by central differences of the analytic
                gradient, for a route pyscf's own analytic Hessian cannot serve

`src.Embedding` and its cRPA-embedded surfaces are out of scope for this
repository, so no row here dispatches to one; a spec that names such a surface
is simply not among the combinations `potential_energy_surface` realizes.
"""
from src.Base.declaration import (ChargedExcitation, Excitation, GroundState,
                                  PhysicsMismatch, QPStates, SurfacePhysics)
from src.properties.surface import (FiniteDifferenceGradient,
                                    PotentialEnergySurface,
                                    surface_mean_field)
from src.properties.optimize import (ground_state_residual_force, optimize,
                                     optimize_geometric, relax,
                                     relax_ground_state,
                                     translation_rotation_basis)
from src.properties.vibronic import (adiabatic_gap, align_to, energy_at,
                                     huang_rhys_from_displacement,
                                     huang_rhys_from_gradient, normal_modes,
                                     relax_state, reorganization,
                                     reorganization_from_huang_rhys,
                                     vibronic_analysis)
from src.properties.conformers import (Conformer, Torsion, boltzmann_weights,
                                       conformer_average, conformer_rmsd,
                                       deduplicate, rotatable_bonds,
                                       search_conformers, torsion_starts,
                                       torsion_values)
from src.properties.characters import ct_character, roots
from src.properties.rates import marcus_levich_jortner_rate, marcus_rate
from src.properties.spin_orbit import (chain_manifolds, isdf_manifolds,
                                       qdpt_hamiltonian, qdpt_spectrum,
                                       soc_operator, soc_operator_mo,
                                       socme_table, spin_orbit_couplings)
from src.properties.nonadiabatic import (DerivativeCouplings,
                                         derivative_couplings, mo_overlap,
                                         state_overlap, translational_sum)
from src.properties.surfaces import (Realization, compare_surfaces,
                                     potential_energy_surface)
from src.properties.excitations import (SurfaceSpec, adiabatic_energy,
                                        calc_adiabatic_excitation,
                                        calc_adiabatic_gap,
                                        calc_emission_energy,
                                        calc_vertical_excitation, surface_of)
from src.properties.hessian import numerical_hessian
