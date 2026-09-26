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
