# Coupled cluster

**Coupled cluster** — CCSDT amplitudes and Lambda equations, restricted and
spin-orbital, and the CCSDT one-particle density, plus EOM-CCSDT (IP/EA/EE).
Generated with pdaggerq. CCSD and EOM-CCSD are not implemented here: use
pyscf's (`pyscf.cc.CCSD`, `pyscf.cc.eom_rccsd` / `eom_gccsd`); the CCSD
density wrappers in `CC/pipeline.py` call pyscf, and `EOMCC` refuses
`level='ccsd'` with a pointer to pyscf.

The CCSDT density and EOM-CCSDT take an RHF reference; a UHF one is refused.
An amplitude or Lambda solve that runs out of iterations warns
(`CCNotConverged`) and reports `converged=False` instead of raising, and an
EOM solve with unconverged roots warns (`UnconvergedEOMRoots`) and says which
in `EOMResult.converged` ([Warnings, flags and refusals](../warnings.md)).

`EOMCC.kernel(sector, nroots)` with `nroots` at or above `manifold_dim(sector)`
diagonalizes the whole projected Hbar densely, so every degenerate cluster is
complete; below that it runs the Davidson. `EOMResult.transition_densities()`
pairs the left eigenvectors with the right ones per eigenvalue cluster, so
degenerate states (spin multiplets, p shells) work. A state whose left
eigenvector is not found warns (`UnpairedEOMStates`), is listed in
`EOMResult.unpaired_states` and is left out of the sums built from the
densities: the polarizability and the G0W@CC self-energy. The G0W@CC
self-energy is the Lehmann sum over the `nroots` states it is given; the
paper's numbers need all of them.
