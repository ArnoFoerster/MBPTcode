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

## High-order coupled cluster in an active space

An active space too large for FCI or DMRG (tens of electrons in tens of
orbitals) can be solved by CCSDT or CCSDTQ instead. `CoupledClusterSolver`
runs pyscf's implementation of these methods by Jin, Hillenbrand, Berkelbach
and Zhai (J. Chem. Phys. 165, 144109 (2026)). It works like the other
active-space solvers in `Solvers/active_space_solver.py`. It takes a one-body
matrix `h1` and a two-body tensor `(pq|rs)` over the active orbitals, and
returns the energy without the core constant.

It first solves Hartree-Fock inside the active space, starting from the
lowest orbitals of the basis the Hamiltonian is written in. Then it runs
coupled cluster on that reference. `order=3` gives CCSDT and `order=4` gives
CCSDTQ. A closed-shell sector uses the spin-adapted (spin-free) code. An
open-shell sector (for example the triplet, `spin=2`) uses spin-unrestricted
CCSDT on a UHF reference. There is no open-shell CCSDTQ.

It gives energies only, and one state per call: the lowest state of the
spin sector asked for (`spin=0` the lowest singlet, `spin=2` the lowest
triplet), so `nroots` is 1. Excited states of the same spin (S1 above S0) are
out of reach, and there are no density matrices: `rdms` raises an error.
Through the driver, paired electrons give the lowest singlet even where a
triplet lies lower; `'pyscf_fci'` there returns the triplet. A downfolded Hamiltonian usually lacks the p <-> q symmetry
of bare integrals. The solver takes it as it is: any real Hermitian two-body
operator works, and nothing is averaged.

The downfolding driver picks it by name: `active_space.solve('ccsdt', g_eff,
...)` or `'ccsdtq'`. Like `'pyscf_fci'` and `'dmrg'`, it refuses a three-body
vertex and a non-Hermitian `g_eff` rather than dropping either. The cost grows
as N^8 for CCSDT and N^10 for CCSDTQ, and memory is mostly the highest
amplitudes. If the reference or the amplitudes do not converge, the solver
warns and records it in `solver.last_result`
([Warnings, flags and refusals](../warnings.md)).

```python
from pyscf import ao2mo, gto, mcscf, scf
from src.Solvers.active_space_solver import CoupledClusterSolver

mol = gto.M(atom='O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471',
            basis='6-31g', verbose=0)
mc = mcscf.CASCI(scf.RHF(mol).run(), 6, 4)
h1, ecore = mc.get_h1eff()
eri = ao2mo.restore(1, mc.get_h2eff(), 6)

solver = CoupledClusterSolver(order=4)                 # CCSDTQ
energy = solver.roots(h1, eri, 6, (2, 2))[0][0] + ecore
print(energy, mc.kernel()[0])        # four electrons: CCSDTQ is exact here
print(solver.last_result['converged'])
```
