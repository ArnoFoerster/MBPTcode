"""Photophysics of an emitter from its relaxed states: the staged TADF run.

    declaration    `Declaration`: what a run is -- the system, the physics,
                   the one `Numerics`, and every procedural policy, each
                   explicit (no defaults)
    store          `Store`: one directory, one record per stage, atomic,
                   stamped and refused under another declaration
    chain          the production BSE@GW chain of a declaration: its
                   `SurfaceSpec`, mean field and continuum
    stages         the stages as functions of a `Run` (ground_minimum,
                   ground_modes, vertical, s1_minimum, t1_minimum, rates) and
                   `run_tadf`, which runs them in order
    minimum        `StateMinimum`: what the rate layer reads of a geometry,
                   filled by GW-BSE or ADC(2) alike
    rate_models    the TADF rate table off three minima and the S0 modes
    uisc           the UISC protocol of Samanta et al.: the promoting-mode
                   derivative stage and its rate rows
    vertical_arms  the vertical spectrum of one geometry under three
                   quasiparticle treatments
    df_check       the ISDF-vs-DF acceptance check of a stage's geometry
    cli            the command line: arguments into a declaration
    launch         the thread environment, set before numpy loads
    solvated       gas and continuum absorption, emission and adiabatic
                   S1/T1 energies of one molecule, ADC(2) or GW-BSE, as one
                   JSON record (`python -m src.photophysics solvated`)

A run, in Python:

    from src.Base.utils.parallel.mpi_grid import distributed
    from src.photophysics.stages import run_tadf
    from src.photophysics.store import Store

    with distributed(None):
        records = run_tadf(declaration, Store('tadf', declaration),
                           max_memory=4000)

MPI enters at the top only (`with distributed(comm):` around `run_tadf`); no
stage takes a communicator.
"""
