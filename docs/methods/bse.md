# BSE

**BSE** — the iterative (Davidson) Bethe-Salpeter equation, singlet or
triplet, in two interchangeable flavours that share every convention:
`solve_bse_isdf` on ISDF factors (one factorization for the GW that feeds it
and the kernel) and `solve_bse_df` on pyscf's own Coulomb fit. Both take
`self_consistency='evGW'`.

Which eigenvalues build the static W is set by the LEVEL OF THEORY: G0W0
screens W₀ at the mean-field eigenvalues, which is the standard split, and evGW
at its converged ones. An explicit `qp=` array therefore follows the G0W0
convention unless `screen_at='qp'` says the array is itself a self-consistent
spectrum.

## The Davidson

**Preconditioner.** The Davidson divides its residuals by a diagonal,
`preconditioner='screened'` by default: the pair energies d = eps_a - eps_i
less the diagonal of the screened direct term, (ii|W|aa), formed in the fit's
own gauge from the block action's factors (on the ISDF route each rank adds
its grid rows' tiles and one reduction completes them; on the DF route from
the factor's own diagonals). It is closer to diag(A) where the screening binds
the pairs, so a BSE solve takes fewer cycles and block actions to the same
roots. `preconditioner='bare'` divides by d alone; it moves the iteration path
only, never the operator or the convergence test, and is what a solve that
must reproduce a bare-preconditioned iteration bit for bit asks for. RPA's
diagonal is d either way.

**The (A-B) probe.** `probe=` measures min eig(A-B) and refuses the roots
while it is negative, the singlet/triplet instability where the omega^2
reduction is invalid. It runs AFTER the Davidson, on the Davidson's own block
action, so the action is built once. An unstable reference is usually refused
sooner, by the Davidson itself: its projected (A-B) block stops being
positive definite, and the probe that breakdown runs names the reference.
`probe='sign'` stops as soon as the sign is proven: first by Rayleigh-Ritz of
(A - B) on the span of the converged roots' X - Y, one block action per root
(the lowest Ritz value theta and its residual r prove an eigenvalue within r
of theta), then by a Lanczos from the lowest root with `PROBE_START_MIX` of
the cold start 1/d, then by the Lanczos from 1/d at escalating tolerances.
The roots' certificate proves an eigenvalue in their span, not the minimum over
pair irreps no root reaches; `probe=True` converges the general probe, the
Lanczos from 1/d. `info['stats']['probe_source']` says which tier decided
('roots', 'lanczos-warm', 'lanczos-cold').

**The trial space.** pyscf's `real_eig` collapses its trial space to the Ritz
vectors when it fills, and a solve that collapses takes more cycles. The space
is raised to `DAVIDSON_SPACE_CYCLES` increments, within what this rank's
memory holds beside the block action's working set: inside a SLURM allocation
`DAVIDSON_SPACE_FRACTION` of `mf.max_memory` (which a job sets to its share of
the allocation), elsewhere `DAVIDSON_SPACE_GB` of the whole holders. Over more
than one rank the four pair-space-long holders are cut into fixed tiles of
`DAVIDSON_PAIR_TILE` pair rows (`LinearResponse.trial_space`): each rank holds
its run of tiles, the projected blocks, norms and Gram-Schmidt overlaps are
reduced once each, and only the new batch is gathered for the block action,
so the Rayleigh-Ritz work divides by the rank count. Serially `real_eig` runs
unchanged. Every decision is read off reduced or lockstepped small matrices,
so every rank returns rank 0's roots bitwise; the reduced sums re-associate
with the rank count, so distributed roots agree with the serial ones within
the Davidson's resolution rather than bit for bit. `info['timings']` splits
the stage into the block action's pieces (`davidson_action_<piece>`), the
pair-row loop's (`davidson_subspace_<piece>`), the collectives, the trial
space's bytes and its budget.
