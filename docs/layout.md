# Source layout

```
src/Base/               PySCF interface, constants, linear algebra
    declaration.py      the GroundState/Excitation/QPStates vocabulary a
                        surface is built from and validated against
    separable_ri.py     ISDF / separable-RI factorization of the ERIs
    isdf_jk.py          ISDF Coulomb and exchange for the SCF
    environment.py      what the surroundings do, in one contract
    solvent_screening.py  PCM reaction field, with its analytic adjoint
    dispersion.py        the empirical D3/D4 correction
    composite_environment.py, polarizable_sites.py, cppe_interface.py
                        QM/MMPol: permanent charges plus induced dipoles
    pcm_factorization.py, pcm_derivatives.py
                        the PCM cavity's cached solve and its nuclear
                        derivative primitives
    basis/cp2k_basis.py CP2K's aug-MOLOPT orbital and RI sets, read at run time
    utils/grids.py      minimax and Gauss-Legendre imaginary-axis grids
    utils/time_frequency.py  one grid object carrying both axes
    utils/matsubara.py  finite-temperature (IR) grids
src/SingleReference/
    ADC/                the ADC solvers (see ADC/__init__.py for the map)
    CC/                 CCSD/CCSDT amplitudes, lambda, EOM
    DensityMatrix/      MPn / GW / CC correlated 1-RDMs
    EpsteinNesbet/      EN denominators and shifts
    GW/                 self-energy, QP equation, imaginary axis/time,
                        the reaction field's shift, the evGW and qsGW
                        loops, contour deformation and sum-over-poles
                        continuations, the dense quasi-boson route
    LinearResponse/     Casida, RPA, BSE, Davidson, the dense quasi-boson BSE
    Periodic/           k-point RPA, BSE and GW: the GDF and ISDF/THC routes,
                        metals, 2D slabs and planar-interface screening
    BSE/                the upfolded (non-perturbative) BSE
src/Solvers/            quasiparticle root finders, including the
                        pole-guarded Newton solve contour deformation uses,
                        and a matrix-free Davidson eigensolver
src/gradients/          analytic nuclear gradients: one adjoint module per
                        forward one, differentiating production's own objects
src/properties/         the ONE surface dispatcher, geometry optimization,
                        vibronic analysis, conformers, rates and the
                        couplings they need
```
