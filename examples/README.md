# Examples

Each file is self-contained and runnable: `python examples/01_adc3.py`.
All use H2O/cc-pVDZ unless noted, so the numbers are directly comparable
(`11_bse_davidson_vs_dense.py` is cc-pVTZ; `18_periodic_slab_gw.py` is a
periodic H2 slab).

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
| `15_excited_state_geometry_optimization.py` | two high-level entry points: S1 relaxed against S0 (`calc_adiabatic_excitation`) and the adiabatic S1-T1 gap (`calc_adiabatic_gap`), dense vs cubic-scaling ISDF/SOP |
| `16_numerical_hessian_from_gradient.py` | a vibrational analysis built by central-differencing the analytic gradient, cross-checked against pyscf's own analytic Hessian |
| `17_spin_orbit_coupling.py` | <S1\|H_SO\|T1/T2> at both relaxed minima from `15`'s adiabatic gap (El-Sayed's rule), plus the Herzberg-Teller dV/dq scan over the ground-state modes that finds the promoting mode |
| `18_periodic_slab_gw.py` | the continuum of `13` for a periodic **slab** — electrolyte above, metal electrode below |
| `19_ee_adc.py` | electronic-excitation ADC(2)/ADC(3): spin-free, matrix-free, DF, and the singlet and triplet channels |

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
15  S1 vertical/emission/adiabatic = 4.518/3.546/4.125 eV (dense); Delta-E_ST
    (adiabatic, S1-T1) = 0.805 eV; ISDF/SOP agrees to ~10 meV on S1 and ~1 meV
    on Delta-E_ST (formaldehyde/cc-pVDZ, n->pi* S1 and T1)
16  frequencies 1802.87, 3959.80, 4056.93 cm-1, exact against pyscf's own
    analytic Hessian (water/cc-pVDZ RHF)
17  <S1|H_SO|T1> = 0.000 cm-1 at both R*_S1 and R*_T1 (n->pi*/n->pi*, same
    configuration, vanishes by El-Sayed's rule); <S1|H_SO|T2> = 42.6 cm-1
    (n->pi*/pi->pi*, different configuration); the Herzberg-Teller scan over
    the 6 ground-state modes puts the largest |dV/dq| = 0.20 cm-1 at 1325
    cm-1, the out-of-plane wag -- the textbook promoting mode for this
    channel -- against 0.00-0.06 cm-1 for the rest (formaldehyde/cc-pVDZ)
18  HOMO +0.309 eV / LUMO -0.427 eV  (H2 slab, water | metal electrode)
19  adc2: S1 = 6.979 eV   T1 = 6.646 eV   E_ST = 0.333 eV
    adc3: S1 = 7.861 eV   T1 = 7.417 eV   E_ST = 0.444 eV
    (water/aug-cc-pVDZ, density-fitted)
```

## Large systems

`09_isdf_gw_space_time.py` is the production route once the dense `cderi` stops
fitting: the SCF and the GW share ONE separable (ISDF) factorization, which is
both the cheaper choice and the correct one — pairing factors from one fit with
a screened interaction from another stays self-consistent and silently moves the
spectrum. Pass `freq_block=` or `scratch_dir=` to `solve_qp_energy_space_time`
when even the frequency-axis W does not fit; the answer is unchanged.

## Periodic slabs

`SlabDielectricEnvironment` + `build_dfintegrals_screened`
(`src/SingleReference/Periodic/pbc_solvent_screening.py`) are the periodic
counterpart of the PCM continuum in `13`. A 3D crystal has no outside, so there
is no cavity; a slab does — the environment fills the vacuum. The closed cavity
is replaced by an **image-charge boundary condition** at each planar interface,
which is exactly rank 2 per in-plane momentum, so it enters the RI-V metric as a
rank-2 update of `J(q)` and `B(q)`. Everything downstream reads only
`PBCDFIntegrals.L`, so screened RPA / W^Q / BSE / Σ_c all follow.
`eps_bot=np.inf` is a metal electrode; `eps_top=solvent` is the electrolyte.

Slab-specific requirements: a non-negative Coulomb kernel (use the damped
kernel from `pbc_rpa_damping` — pyscf's `get_coulG` is negative at G=0 for
`dimension < 3`, and `low_dim_ft_type='inf_vacuum'` breaks pyscf's own
periodic SCF), and a cavity that encloses the density
(`leaked_density_fraction`). The single in-plane channel at q∥+G∥ = 0 is
dropped — a mixed-representation head cannot be regularized against a
G-diagonal `coulG`; see the module docstring.
