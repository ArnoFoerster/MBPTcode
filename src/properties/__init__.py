"""Everything that is computed FROM a potential-energy surface rather than by
one: geometries, vibronic spectra, state labels and couplings.

The layer knows one interface, `src.gradients.surface.PotentialEnergySurface`
-- a total energy over the nuclei, its gradient, and the method's own way of
rebuilding what it froze at a reference geometry -- and the ingredients a
surface declares beyond it (`src.gradients.surface.INGREDIENT_DOCS`), and nothing
about how that energy is obtained. A BSE@GW excitation, a dRPA ground state
and an ADC(2) state are the same object here, and a surface that has no
analytic gradient is driven through `FiniteDifferenceGradient` instead of
being excluded.

    optimize    Cartesian RFO/BFGS with the rigid-body directions projected
                out, geomeTRIC when it is installed, and the refreeze outer
                loop that measures the frozen conventions' own drift
    vibronic    normal modes, Huang-Rhys factors by two independent routes,
                adiabatic gaps and the Marcus reorganization energy
    curvilinear Huang-Rhys factors in curvilinear normal coordinates
    band_shape  the vibronic band shape of a transition
    conformers  the torsional minima a soft emitter has, their Boltzmann
                populations, and a property averaged over them
    characters  the whole spectrum at one geometry and its charge-transfer
                weights, for labelling roots; the fragment character of a
                quasiparticle orbital (Pipek-Mezey) and whether it is the
                lowest-energy process (following an orbital or a state across
                geometries is `src.gradients.state_identity`)
    franck_condon
                the Franck-Condon weighted density of states of displaced
                oscillators, which the golden-rule rates of the physics layer
                (`src.photophysics.rates`) multiply by the squared coupling
    spin_orbit  <S_I|H_SO|T_J> over any solver's Casida vectors -- the coupling
                `photophysics.rates` receives
    spin_vibronic
                the nuclear derivative of one <S|H_SO|T> element through the
                higher states, for the Herzberg-Teller term of the rates
    mode_hessian
                force constants along chosen normal modes by differencing the
                analytic gradient: excited-state frequencies of local modes
    nonadiabatic
                <Psi_I|d/dR Psi_J> from overlaps of those same vectors at
                displaced geometries -- the other coupling the rates need, and
                the only one here that costs a gradient rather than a property
    excitations the vertical, emission and adiabatic energies of a state, with
                BOTH surfaces of every difference built from one `SurfaceSpec`
                -- so the functional under the excited state and the functional
                under the ground state cannot be two functionals
    hessian     the nuclear Hessian by central differences of the analytic
                gradient, for a route pyscf's own analytic Hessian cannot serve
"""
