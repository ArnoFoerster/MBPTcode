# Tests

Standalone scripts, not a pytest suite. Each one runs a calculation, prints a
verdict per check and exits non-zero if any of them failed:

```bash
python tests/test_adc3.py
```

Almost nothing here is collected by `pytest`: the checks live under
`if __name__ == '__main__'`, so `pytest tests/` reports a directory of empty
files as passing. Run them as scripts, and **read the exit code** — a script
that prints `FAILURES DETECTED` is telling you something a green summary line
would not.

`test_imports.py` is the cheap one to run first: it imports every module under
`src/` in about a second and separates a missing `src.` module from a missing
optional third-party one.

## What covers what

| area | tests |
|---|---|
| imports, constants, interfaces | `test_imports`, `test_constants_registry`, `test_base` |
| basis sets read from CP2K | `test_cp2k_basis` |
| ADC solvers | `test_adc3`, `test_adc3_restricted`, `test_adc2x_df`, `test_adc3_df_memory_fix`, `test_spin_adapt`, `test_screened_adc2x`, `test_downfolded_seeds`, `test_unrestricted_neon`, `test_bn_unrestricted_excitations` |
| Epstein–Nesbet and static corrections | `test_uhf_static_correction_df`, `test_uhf_ccsd_static_correction`, `test_amplitudes_consistency` |
| coupled cluster | `test_restricted_ccsdt`, `test_ccsdt_lambda`, `test_ccsdt_density_matrix`, `test_eom_ccsdt`, `test_cc_polarizability` |
| MPn densities and Laplace | `test_mp2_density_matrix`, `test_mp3_density_matrix`, `test_mp2_density_df`, `test_mp3_density_df`, `test_mpn_density_restricted`, `test_mpn_density_unrestricted`, `test_mp4_laplace_restricted`, `test_density_matrix_small` |
| response derivatives (finite field) | `test_mp3_finite_field`, `test_uhf_mp2_relaxed_finite_field` |
| GW self-energy and QP equation | `test_self_energy_formulas`, `test_self_energy_diagonal_batch`, `test_self_energy_mode_matrix`, `test_analytical_continuation`, `test_construct_4d_w_rpa`, `test_rpa_correlation_energy` |
| eigenvalue self-consistency | `test_evgw` |
| imaginary axis and time | `test_imaginary_axis_gw`, `test_imaginary_axis_gw_dft`, `test_sigma_blocking_and_screening`, `test_mpi_grid_distribution` |
| grids | `test_grids`, `test_minimax_tau_grid`, `test_time_frequency_grid`, `test_matsubara_ir` |
| ISDF factorization | `test_isdf_jk`, `test_frame_sign_convention`, `test_grid_radii_optimizer`, `test_static_exchange_routes` |
| BSE | `test_davidson_casida`, `test_davidson_isdf_bse`, `test_davidson_benzene_bse`, `test_bse_isdf_driver`, `test_bse_df_driver`, `test_bse_screening_energies`, `test_davidson_triplet`, `test_casida_normalization` |
| environment and solvent | `test_environment`, `test_solvent_screening`, `test_solvent_mean_field`, `test_reaction_field` |
| distributed linear algebra | `test_numroc`, `test_elpa_casida` |

## The ISDF and BSE tests, in the order they build on each other

These four are worth reading as a sequence, because each one guards a property
the next one assumes:

- `test_isdf_jk` — the J/K builder. Structure first (no three-index tensor is
  ever formed, `loop()` refuses), then accuracy against DF.
- `test_frame_sign_convention` — the interpolation grid is placed in per-atom
  frames whose axis SIGNS are pure gauge: negating one permutes grid rows and
  moves no point. Deterministic, continuous through planar geometries, and the
  energy invariant under every sign pattern.
- `test_grid_radii_optimizer` — the radii themselves, against Duchemin & Blase's
  published tables. The objective is multi-modal, so a single descent is a
  lottery; this pins that multi-start fixes it, that `n_start=1` still means the
  old single descent, and that recipes coexist in the shipped table.
- `test_static_exchange_routes` — the QP step's static exchange. `Sigma_x` built
  from the mean field's own K inherits the SCF route's error at first order, so
  this pins the streamed density-fitted build against the stored-tensor one and
  against the routing.

`test_bse_screening_energies` guards the other half of the same question: not
which basis W is expressed in, but which ENERGIES it is screened at. The BSE
diagonal carries the quasiparticle energies while W is screened at the
mean-field ones, and a 4-center solver has to be told that with `eps_screen=`
because it rebuilds the direct term itself. Getting it wrong is worth more than
the auxiliary basis error it looks like, and does not shrink under refinement.

`test_davidson_isdf_bse` is the gauge test of the pair: the matrix-free BSE
action against the dense Casida solver, plus the negative control of pairing
ISDF factors with a cderi-gauge `W_aux`, which stays self-consistent and gives
the wrong spectrum.

## The two tests that assert a FAILURE, and why

Both guard properties that nothing else in the suite would notice going away.

`test_evgw` drives the loop by hand with the quasiparticle equation anchored on
the ITERATE instead of the mean field, and asserts that it DIVERGES — the gap
opening by more than an eV every cycle. That version still runs and still
prints plausible numbers, so without the assertion, dropping `eps_anchor` would
leave every other check in the file passing.

`test_casida_normalization` hands `oscillator_strengths` a raw pySCF vector and
asserts it is refused by name. pySCF normalizes to ⟨X|X⟩ − ⟨Y|Y⟩ = 1/2 where
this repo uses 1, the difference cancels out of every excitation energy, and
the consumer is quadratic in the vector — so the mistake is a silent factor of
two in every oscillator strength with every root at the right energy.

## The environment seam

`test_environment` pins the rule the three solvent tests rest on: an
environment enters in two places and only one of them is a choice. Whether it
dresses the interaction is asked in exactly one place (`dresses_interaction`),
and the Eq. (18) quasiparticle shift follows from that answer rather than from
a switch of its own. The negative control is `PointCharges`, which does not
respond, therefore screens nothing, and must move the quasiparticle energy
through the mean field alone.

`test_reaction_field` then checks the economy that makes the shift affordable:
ΔW needs ONE χ₀, the bare screening following from the dressed one by a
congruence in the auxiliary gauge, and the compact contraction that never
builds ΔW must equal the explicit form that does.

## Reference data

`reference_data.json` and `adc_refactor_pins.json` hold pinned numbers several
tests compare against. Regenerating a pin is a deliberate act: it changes what
the suite considers correct, so the reason belongs in the commit that does it.

Optimized ISDF grid radii ship in `src/Base/data/optimized_radii.json`. The
per-machine scratch cache beside it (`src/Base/data/radii_cache/`) is
gitignored, because the radii optimizer is a numerically differentiated descent
under threaded BLAS and does not reproduce across thread counts — the shipped
table is what makes a clean clone reproduce the suite.

## The multi-rank test

`test_elpa_casida` is the one script meant for several MPI ranks:

```bash
srun -n 4 --mpi=pmix python tests/test_elpa_casida.py
```

It solves the three Casida branches and ADC's dense matrix once with every rank
taking part and once with the workers parked in `serve_distributed_solves`, and
compares each with the serial solve. On more than one rank a cell that fell back
to `eigh` fails, so a missing `pyelpa` shows as a failure, not as a pass. Run
serially, as the suite does, it checks the comparisons only.
