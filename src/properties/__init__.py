"""Everything that is computed FROM a potential-energy surface rather than by
one: geometries, vibronic spectra, state labels and transfer rates.

The layer knows one interface, `PotentialEnergySurface` -- a total energy over
the nuclei, its gradient, and the method's own way of rebuilding what it froze
at a reference geometry -- and nothing about how that energy is obtained. A
BSE@GW excitation, a dRPA ground state and a downfolded active-space root are
the same object here, and a surface that has no analytic gradient is driven
through `FiniteDifferenceGradient` instead of being excluded.

    surface     the protocol, and the finite-difference gradient wrapper (the
                root-selection vocabulary is `src.gradients.surface`)
    optimize    Cartesian RFO/BFGS with the rigid-body directions projected
                out, geomeTRIC when it is installed, and the refreeze outer
                loop that measures the frozen conventions' own drift
    vibronic    normal modes, Huang-Rhys factors by two independent routes,
                adiabatic gaps and the Marcus reorganization energy
    conformers  the torsional minima a soft emitter has, their Boltzmann
                populations, and a property averaged over them
    characters  the whole BSE spectrum at one geometry and its charge-transfer
                weights, for labelling roots; the fragment character of a
                quasiparticle orbital (Pipek-Mezey) and whether it is the
                lowest-energy process (following an orbital or a state across
                geometries is `src.gradients.state_identity`)
    franck_condon
                the Franck-Condon weighted density of states of displaced
                oscillators, which the golden-rule rates of the physics layer
                (`src.photophysics.rates`) multiply by the squared coupling
    spin_orbit  <S_I|H_SO|T_J> over any solver's Casida vectors -- the coupling
                `rates` was written to receive
    spin_vibronic
                the nuclear derivative of one <S|H_SO|T> element through the
                higher states, for the Herzberg-Teller term of `rates`
    mode_hessian
                force constants along chosen normal modes by differencing the
                analytic gradient: excited-state frequencies of local modes
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
    diabatic    finite differences of the fragment-diabatic BSE matrix
                (`src.gradients.gwbse.fragment_bse`), relocalized at every displaced
                geometry: the reference the analytic gradient
                (`src.gradients.gwbse.fragment_diabatic`) is gated against, and the
                linear vibronic coupling model built on that gradient

`src.Embedding` and its cRPA-embedded surfaces are out of scope for this
repository, so no row here dispatches to one; a spec that names such a surface
is simply not among the combinations `potential_energy_surface` realizes.

The names below are imported on first use, so importing one module of this
package does not import the drivers.
"""
import importlib

_EXPORTS = {
    'src.Base.declaration': ('ChargedExcitation', 'Excitation', 'GroundState',
                             'PhysicsMismatch', 'QPStates', 'SurfacePhysics'),
    'src.gradients.surface': ('FiniteDifferenceGradient',
                               'PotentialEnergySurface', 'surface_mean_field'),
    'src.properties.optimize': ('ground_state_residual_force',
                                'optimize_geometric', 'relax',
                                'relax_ground_state',
                                'translation_rotation_basis'),
    'src.properties.vibronic': ('adiabatic_gap', 'align_to', 'energy_at',
                                'huang_rhys_from_displacement',
                                'huang_rhys_from_gradient', 'normal_modes',
                                'relax_state', 'reorganization',
                                'reorganization_four_point',
                                'reorganization_from_huang_rhys',
                                'vibronic_analysis'),
    'src.properties.conformers': ('Conformer', 'Torsion', 'boltzmann_weights',
                                  'conformer_average', 'conformer_rmsd',
                                  'deduplicate', 'rotatable_bonds',
                                  'search_conformers', 'torsion_starts',
                                  'torsion_values'),
    'src.properties.characters': ('ct_character', 'quasiparticle_character',
                                  'quasiparticle_order', 'roots',
                                  'surface_order'),
    'src.properties.spin_orbit': ('chain_manifolds', 'isdf_manifolds',
                                  'qdpt_hamiltonian', 'qdpt_spectrum',
                                  'soc_operator', 'soc_operator_mo',
                                  'socme_table', 'spin_orbit_couplings'),
    'src.properties.nonadiabatic': ('DerivativeCouplings',
                                    'derivative_couplings',
                                    'translational_sum'),
    'src.gradients.gwbse.surface_factory': ('Realization', 'compare_surfaces',
                                'potential_energy_surface'),
    'src.properties.excitations': ('SurfaceSpec', 'adiabatic_energy',
                                   'calc_adiabatic_excitation',
                                   'calc_adiabatic_gap',
                                   'calc_vertical_excitation', 'surface_of'),
    'src.properties.hessian': ('numerical_hessian',),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items()
              for name in names}


def __getattr__(name):
    """The exported routine or class `name`, imported from its module."""
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_MODULE_OF))
