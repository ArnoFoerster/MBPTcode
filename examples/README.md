# Examples

Each file is self-contained and runnable: `python examples/01_adc3.py`.
All use H2O/cc-pVDZ unless noted, so the numbers are directly comparable
(`11_bse_davidson_vs_dense.py` is cc-pVTZ).

| file | what it shows |
|---|---|
| `01_adc3.py` | standard ADC(3), spin-free, matrix-free — the default route |
| `02_adc3_df.py` | the same with density fitting, for large bases |
| `03_en_adc3.py` | EN-ADC(3), spin-adapted — the production EN variant |
| `06_open_shell.py` | UHF / spin-orbital ADC(3) |
| `07_several_states.py` | several ionization states with pole strengths |
| `08_en_adc3_screened_singles.py` | EN-ADC(3) singles shift, bare vs RPA-screened (BSE-kernel channel split) |
| `09_isdf_gw_space_time.py` | cubic-scaling GW: an ISDF SCF, then the space-time self-energy on its factors |
| `10_gw_routes.py` | the four routes to one G0W0 energy: Casida (full ERIs / DF), imaginary frequency, space-time |
| `11_bse_davidson_vs_dense.py` | BSE two ways (Davidson, dense) on three integral flavours (ISDF, DF, full ERIs) |
| `12_evgw.py` | the eigenvalue-self-consistent loop: cycle 1 is G0W0, the fixed point sits above it |
| `13_solvated_gw_bse.py` | GW and BSE in a continuum, with the ground-state and response halves separated |
| `14_cp2k_aug_molopt.py` | CP2K's aug-SZV-MOLOPT-ae basis with its RI tier, read from CP2K at run time; G0W0 and dense BSE |

The auxiliary basis is a choice, not a detail. `<basis>-ri` is an MP2
correlation-fitting set for occupied-virtual products, while J, K and the BSE
direct term contract `(ij|ab)` — occupied-occupied against virtual-virtual.
Fitting those in `-ri` puts 21 meV between DF and the exact tensor at cc-pVDZ
and 6 meV at cc-pVTZ, where `-jkfit` gives 2 meV. `11` therefore passes
`-jkfit`, and passes it to the separable fit as well: that side does not read
the mean field's auxiliary basis, and mismatching the two costs more than
either route's own error.

`09` and `10` still use `-ri`. They print no DF-against-exact comparison, so the
choice is invisible there, but it is the same trade. The library default stays
`<basis>-ri` because the shipped ISDF radii are keyed on the auxiliary basis and
`-jkfit` rows exist only for H and O at cc-pVDZ; changing the default would
silently drop every other configuration back to an unoptimized grid.

## Choosing a route

**Quasiparticle energies** → `calc_qp_energy(mf, state='homo')`: the Casida
route with density fitting, `polarizability='RPA'`, i.e. plain G0W0. `mode=`
swaps in the low-scaling routes (`'imagfrequency'`, `'space-time'`), both of
which require `df=True`; `df=False` gives the 4-center Casida reference. See
`10_gw_routes.py`.

**Excitation energies** → `solve_bse_isdf(mf, mol, nocc)`: a matrix-free
Davidson BSE on a separable (ISDF) factorization, `qp='G0W0'`, so it runs its
own GW and returns BSE@G0W0. It is ISDF whatever the mean field is; the dense
A/B route is the reference, not the production path. See
`11_bse_davidson_vs_dense.py`.

**Closed shell** → `ADCSolverRestricted` (spin-free). Add `B_aa=` for DF once the
dense ERI stops fitting. **Open shell** → `ADCSolver` (spin-orbital).

**EN** is ADC(3) only. Pass the same `u2_denom_dress` dict to both the solver and
the static correction — dressing one and not the other silently mixes methods.

- `spin_adapted=True` + `shift='sum'` (2J−K): production. `'mean'` (J−K/2) and
  `'opposite'` (J) are the other weightings.
- `singles=False` dresses the doubles only — the calibrated variant.
- `singles='screened'` replaces the bare CIS/BSE diagonal J−K by the
  RPA-screened J_W−K (restricted only): see `08_en_adc3_screened_singles.py`.

`ADCSolver.u2_denom_dress` raises: its U^(2) blocks are a merged form that is
only valid for the bare amplitude.

## Expected output

```
01  IP = 12.225 eV   Z = 0.936
02  IP = 12.226 eV   Z = 0.936   (DF)
03  IP = 11.875 eV   Z = 0.929   (EN, spin-adapted)
06  IP =  8.965 eV   Z = 0.953   (UHF / spin-orbital)
09  E(ISDF-SCF) = -76.02774739 Ha   G0W0 HOMO = -12.155 eV   (space-time)
10  G0W0 HOMO = -12.159 eV on all three DF routes, -12.158 eV on space-time
11  BSE@G0W0 = 8.492, 10.540, 10.966, 13.024 eV  (cc-pVTZ, cc-pVTZ-JKFIT)
    all five routes within 2 meV: Davidson == dense exactly, ISDF-DF 1 meV,
    DF-full 2 meV
14  G0W0 HOMO = -9.754 eV   LUMO = 1.659 eV   BSE@G0W0 = 2.733, 6.158, 7.199 eV
    (formaldehyde, aug-SZV-MOLOPT-ae, 116 auxiliary functions at Delta-I 1e-4;
    that tier is 27 meV from the exact tensor on the singlets, the tightest 1.6 meV)
```

## Large systems

`09_isdf_gw_space_time.py` is the production route once the dense `cderi` stops
fitting: the SCF and the GW share ONE separable (ISDF) factorization, which is
both the cheaper choice and the correct one — pairing factors from one fit with
a screened interaction from another stays self-consistent and silently moves the
spectrum. Pass `freq_block=` or `scratch_dir=` to `solve_qp_energy_space_time`
when even the frequency-axis W does not fit; the answer is unchanged.
