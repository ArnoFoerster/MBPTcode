# Source layout

The code is stacked in layers. A module may import from its own layer and the
layers below it, never from a layer above:

| layer | packages | what it holds |
|---|---|---|
| physics | `src/photophysics` | composite runs of a molecule: rates, the staged TADF run, solvated spectra |
| properties | `src/properties` | geometry optimization, vibrations, couplings, band shapes: anything computed from a potential-energy surface |
| gradients | `src/gradients` and its `rpa/`, `gwbse/`, `adc/` | the surfaces and their analytic forces |
| electronic structure | `src/SingleReference`, `src/MultiReference`, `src/fragments` | the methods: ADC, CC, GW, BSE, linear response, ... |
| environments | `src/environments` | solvent continua, point charges, polarizable sites |
| foundation | `src/Base`, `src/Solvers` | the pySCF interface, numerical settings, factorizations, grids, eigensolvers |

Inside the gradient layer, `src/gradients` itself is the part every method
uses; `rpa/`, `gwbse/` and `adc/` hold one method each. `gwbse/` may import
`rpa/` (GW-BSE is built on RPA screening), and `adc/` imports neither and is
imported by neither. `src/SingleReference/Periodic` is used by no other
package. The code generators in `src/CodeGeneration` are never imported by the
code that runs a calculation. `tests/test_layering.py` checks all of these
rules on every module.

```
src/Base/                      the foundation
    numerics/                  numerics.py: the one Numerics object of a run
                               (grids, quadratures, solver settings);
                               constants.py: every named number
    pyscf/                     pyscf_interface.py (integrals, the mean field's
                               pieces), scf_convergence.py, dispersion.py
                               (empirical D3/D4), cphf.py (the CPHF /
                               Z-vector solve of the relaxed densities; the
                               gradient layer's multiplier solve is
                               gradients/multipliers.py)
    factorization/             separable_ri.py (ISDF / separable RI of the
                               ERIs, with its radii table in data/),
                               isdf_jk.py (ISDF Coulomb and exchange for the
                               SCF), the distributed DF and ISDF builds,
                               sliced_factors.py
    geometry/                  displaced geometries, body frames, Kabsch
    basis/                     CP2K's aug-MOLOPT orbital and RI sets, read at
                               run time (see basis-sets.md)
    utils/grids/               minimax and Gauss-Legendre imaginary-axis
                               grids, the time-frequency grid object,
                               finite-temperature (IR) grids, analytic
                               continuation
    utils/linear_algebra/      matrix functions, orthogonalization, Krylov
                               solvers, the distributed ELPA eigensolve
    utils/parallel/            the MPI region (mpi_grid.py), thread pools,
                               memory budgets
    utils/record_io.py         JSON records and checkpoints, the code id a
                               record carries
    utils/timing.py            wall-clock seconds of a named stage,
                               accumulated into a dict
    declaration.py             GroundState, Excitation, ChargedExcitation,
                               QPStates: the physics a surface is asked for
    environment_seam.py        the one contract every route reads its
                               surroundings through
    numerical_flags.py         named numerical flags: a solver that stops
                               short warns and raises a flag, which the
                               function returning the record collects (see
                               warnings.md)
    orbital_spaces.py          the occupied/virtual split
    step_failed.py             StepFailed, which a walk catches to shorten a
                               step
src/Solvers/                   the quasiparticle root finder (one for every
                               route), the symmetric Davidson eigensolver,
                               and the active-space solvers (FCI,
                               determinant basis, DMRG)
src/environments/              solvent_screening.py (PCM reaction field and
                               its adjoint), pcm_factorization.py and
                               pcm_derivatives.py (the cavity's cached solve
                               and nuclear derivatives), environment.py
                               (point charges, the solvation schemes),
                               split_equilibrium.py, reaction_field.py and
                               solvation_loop.py (state-specific solvation),
                               polarizable_sites.py, cppe_interface.py,
                               composite_environment.py,
                               protein_environment.py (a QM region and its
                               QM/MMPol surroundings from a PQR file)
src/SingleReference/
    ADC/                       charged (IP/EA) ADC; see ADC/__init__.py for
                               the map
    ADC/eeADC/                 excitation-energy ADC; `python -m
                               src.SingleReference.ADC.eeADC` runs the
                               vertical ADC(2) singlets and triplets of a
                               molecule
    CC/                        CCSDT amplitudes, Lambda, EOM-CCSDT
    DensityMatrix/             MPn, GW and CC correlated 1-RDMs
    EpsteinNesbet/             EN denominators and shifts
    GW/                        self-energy, the QP equation, the evGW and
                               qsGW loops, contour deformation and
                               sum-over-poles, the reaction field's shift,
                               the dense quasi-boson route
    LinearResponse/            Casida, RPA, the Casida Davidson, RPA
                               correlation energies
    BSE/                       solve.py: the BSE drivers (solve_bse,
                               solve_bse_isdf, solve_bse_df); the dense
                               quasi-boson BSE; the upfolded BSE
    Periodic/                  k-point RPA, BSE and GW (see
                               methods/periodic.md)
    common/                    pieces several methods share: MP1 amplitudes,
                               ERI blocks, DF amplitudes, ActiveSpace
src/MultiReference/QDPT/       active-space downfolding: generated MP2/MP3
                               diagram evaluators and the effective
                               Hamiltonian built from them
src/fragments/                 fragment-localized orbitals, the basis of the
                               fragment-partitioned BSE
src/gradients/                 the method-neutral core: surface.py (the
                               PotentialEnergySurface base class and the
                               ingredients a property may ask of a surface),
                               the mean-field response and Z-vector,
                               derivative integrals (one-electron, DF, ISDF),
                               state and surface identity, state models and
                               the solvated surface, the force of a
                               density-fitted correlated energy
    rpa/                       the dRPA ground state and the RPA screening
                               chain GW-BSE is built on
    gwbse/                     GW-BSE forces: the excited-state chain, the
                               quasiparticle set, the adjoints of each
                               continuation, the state manifold, derivative
                               couplings, the fragment BSE, and
                               surface_factory.py, the one entry point
                               (potential_energy_surface)
    adc/                       ADC(2) forces and the ADC(2) state model
src/properties/                optimize.py (walks, refreeze), vibronic.py,
                               hessian.py, mode_hessian.py (force constants
                               along chosen normal modes), curvilinear.py
                               (displacements in valence internal
                               coordinates, for Huang-Rhys factors),
                               conformers.py, spin_orbit.py,
                               spin_vibronic.py, nonadiabatic.py,
                               characters.py, band_shape.py,
                               franck_condon.py, excitations.py (vertical
                               and adiabatic energies from one SurfaceSpec)
src/photophysics/              rates.py (golden-rule and Marcus rates, the
                               rate table) and the composite runs: the
                               staged TADF run (`python -m src.photophysics`)
                               and the gas and continuum absorption, emission
                               and adiabatic energies (`python -m
                               src.photophysics solvated`)
src/autograd_oracle/           an optional torch reference for the GW-BSE
                               forces, used by the tests only
src/CodeGeneration/            the pdaggerq code generators (see
                               installation.md)
tools/ci/                      the test tiers the CI runs (see testing.md)
```

A checkpoint written before this layout names its classes by their old
modules; `src/Base/utils/record_names.json` maps them to the new ones, so old
checkpoints still load.
