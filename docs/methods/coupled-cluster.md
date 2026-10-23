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
