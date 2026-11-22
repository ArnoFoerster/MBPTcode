# Downfolding to an active space (HF and RPA)

`src.EffectiveHamiltonian.Downfolding` builds a static effective Hamiltonian
for an active space from a low-level method on the whole molecule: the orbitals
outside the active space enter as a frozen core (HF) and, for the RPA, as a
screening of the active-space Coulomb interaction. The scheme is that of the
RPA-downfolding paper (cRPA, cRPA_ph and mRPA screening of an active-space
Hamiltonian with a constant E_0), on canonical orbitals. The result goes to
any active-space solver in `src.EffectiveHamiltonian.solvers.solve`.

Available now:

- `method='hf'`: the HF limit, which is CASCI.
- `method='rpa'`, `mode='casida'` (dense Casida, every pole),
  `mode='imagfrequency'` (density fitted, imaginary frequency, N^4) or
  `mode='space-time'` (separable ISDF factors, imaginary time, N^3), with
  `screening` `'bare'`, `'crpa'` or `'crpa_ph'`, and on `mode='casida'`
  also `'mrpa'` (the numerical routes refuse it and point to casida); and
  optionally (every route) `sigma='srg'` or `'mode_a'`: the static
  environment self-energy Sigma^E that the paper drops (see below).
- E_0 always, term by term.
- Restricted closed-shell (RHF/RKS) and unrestricted (UHF/UKS) references,
  on every route. An unrestricted one gives an H_eff with different alpha
  and beta blocks; see [Unrestricted references](#unrestricted-references).

## Calling it

```python
from pyscf import gto, scf

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.solvers.solve import solve

mol = gto.M(atom='C 0 0.6695 0; C 0 -0.6695 0; H 0.9289 1.2321 0; '
                 'H -0.9289 1.2321 0; H 0.9289 -1.2321 0; H -0.9289 -1.2321 0',
            basis='cc-pvdz')   # ethylene; the snippet's pi space is on atoms 0, 1
mf = scf.RHF(mol).run()
active = ActiveSpace.from_population(mf, atoms=[0, 1], ao_type='p',
                                     n_occ=1, n_virt=1)

low = run_low_level(mf, method='rpa', mode='casida', df=True, numerics=None)
g_eff, space, energies = build_effective_hamiltonian(
    low, active, screening='crpa', sigma=None, spin_free=True)
roots = solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
              spin_free=space['spin_free'], nroots=3)
```

`run_low_level` solves the low-level method on the whole system once and does
not know the active space; `build_effective_hamiltonian` then makes the
effective Hamiltonian of one active space. One `low` serves any number of
active spaces and screenings, so a comparison loop pays for the RPA once.
It belongs to the mean field as it was at the run: when the same `mf`
object is converged again (another geometry in a scan), a build on the old
`low` is refused (`ValueError`), and `run_low_level` has to be called again.
`method='hf'` takes no options: `run_low_level(mf, method='hf')`, and
`build_effective_hamiltonian(low_hf, active)` is CASCI.

**The two limits.** A (0,0) active space,
`ActiveSpace.from_indices(norb, nocc, [], [])`, leaves everything in the
environment: `energies.e0` is then the total energy, E_HF for `method='hf'`
and E_HF + E_RPA,corr for `method='rpa'` (the paper's all-environment rows).
`solve` accepts it too: it returns `[g_eff[0]]` (E_act = 0) and warns with
`EmptyActiveSpaceWarning`, named flag `'empty_active_space'`. Every orbital
active, `ActiveSpace.from_indices(norb, nocc, range(nocc), range(nocc, norb))`,
is FCI with either method: the screening correction vanishes and the two RPA
correlation energies cancel exactly.

**The routes.** `'casida'` diagonalizes the full Casida problem once and
keeps every excitation energy and amplitude. `'imagfrequency'` diagonalizes
nothing: it keeps the full RPA energy of the existing imaginary-axis code,
its minimax frequency grid and the static polarizability P(0) of the whole
system in the fitting basis. Its screening correction is P - P^A (P^A the
active particle-hole pairs' part) and one Dyson step, which on the same
three-index object is the Casida route's deletion to rounding. E_c^A[v_eff]
is integrated on the full energy's grid (`numerics.quadrature.nfreq_rpa`), so
the E_c pair is one quadrature and cancels to rounding with every orbital
active, whatever the grid; the dense-Casida E_c^A is the
`'ec_active_casida'` diagnostic. Both routes agree on every term of H_eff
and E_0 to the quadrature error of E_c[v] (below 1e-11 Eh at the default
grid on water).

`'space-time'` is the same in imaginary time on separable (ISDF) factors
(X, D), built on the grid `numerics.isdf` names (there is no default grid:
pass e.g. `Numerics(isdf={'level': 'G1'})`). It keeps the factors, P(0) on
the static tau grid (`quadrature.ntau_w`) and the full RPA energy with its
time-frequency grid (`quadrature.ntau_rpa`, `nfreq_rpa`). The same
`nfreq_rpa` counts Gauss-Legendre points here (default 40) but minimax
points on imagfrequency (default 20, tables up to 34). P^A is built as P
was, from the active columns of X on the same grid, and E_c^A takes the
active pairs through the energy's own tau -> omega transform, so with every
orbital active both cancel to rounding on any grid. Against casida on the
mean field's RI factors the route differs by the ISDF fit (NH3/cc-pVDZ, G1
grid: v~ 5e-6 per element, the FCI energy of H_eff 1e-6 Eh, E_0 3e-5 Eh
through E_HF,ee[v~], which cancels against h_eff in the energy).

`'imagfrequency'` and `'space-time'` build their frequency grids on the
occupied-virtual transitions, so a reference without a virtual (or an
occupied) orbital, such as He in a one-function basis, is refused there
(`NotImplementedError`, pointing to casida). `'casida'` takes it: there is
no excitation, E_c = 0 and nothing is screened.

Options of the RPA:

| call | option | values |
|---|---|---|
| `run_low_level` | `mode` | `'casida'`, `'imagfrequency'`, `'space-time'` |
| | `df` | `True`: the screening from the mean field's density fit (an exact eigendecomposition of the MO integrals if it has none; ISDF factors on `'space-time'`); `False`: exact 4-centre integrals, `'casida'` only |
| | `numerics` | a `Numerics`, or `None` for the defaults (`'space-time'` needs one naming the ISDF grid) |
| `build_effective_hamiltonian` | `active_space` | one `ActiveSpace` for a restricted reference; the pair `(space_alpha, space_beta)` for UHF/UKS |
| | `spin_free` | `True` (default): spatial-orbital `g_eff`, for UHF/UKS the per-spin blocks `(h_alpha, h_beta)` and `(aa, ab, bb)`; `False`: the 2 n_act interleaved spin orbitals |
| | `screening` | `'bare'`, `'crpa'` (default), `'crpa_ph'`; `'mrpa'` on `'casida'` only |
| | `sigma` | `None` (default, the paper's scheme), `'srg'` or `'mode_a'` (Sigma^E, below) |
| | `diagnostics` | a tuple of names, default all; see below |
| | `numerics` | `None` (default): the Sigma^E settings of the run; a `Numerics` whose `downfolding` section (`srg_s`, `pole_margin`, `continuation_margin`) is used for this Hamiltonian instead, the rest equal to the run's |

`g_eff = {0: E_0, 1: h_eff, 2: v_eff}` and `space` follow the QDPT
`build_effective_hamiltonian` contract (`n_orbs`, `n_elec_active`, `act_idx`,
`act_occ_idx`, `act_vir_idx`, `spin_free`, `active_space`); for a UHF/UKS
reference `g_eff[1]` and `g_eff[2]` hold the per-spin blocks and the entries
of `space` are pairs (see
[Unrestricted references](#unrestricted-references)). Every backend of
`solve` adds `g_eff[0]`, so the roots it returns are **total energies**. The
`ActiveSpace` is the user's choice (`from_indices`, `from_counts`,
`from_irreps`, `from_population`); nothing here selects orbitals.

`energies` is a frozen record holding every term of E_0 on its own, `e0` for
their sum, and `diagnostics`:

| field | term | sign in E_0 |
|---|---|---|
| `e_nuc` | nuclear repulsion | + |
| `e_core` | one-electron energy of the environment (the inactive occupied orbitals) | + |
| `e_hartree`, `e_exchange` | Hartree and exchange energy of the environment | + |
| `e_hartree_dc`, `e_exchange_dc` | Hartree and exchange halves of E_HF,ee[v~]: the double-counting correction of the mean-field energy. Zero unless v~ has occupied-occupied elements, so nonzero for `crpa` only | + |
| `e_c_full` | correlation energy of the whole system, E_c[v] | + |
| `e_c_active` | correlation energy of the active space, E_c^A[v_eff]: the double-counting correction of the correlation energy | - |
| `e_exchange_term` | the exchange term that comes with Sigma^E (-2 `e_exchange_dc`); zero without `sigma` | + |
| `e_sigma_trace` | Tr[Sigma_bar gamma^A,0]; zero without `sigma` | - |

Three properties give the sums per kind, `eh = e_hartree + e_hartree_dc`,
`ex = e_exchange + e_exchange_dc + e_exchange_term` and
`ec = e_c_full - e_c_active`, so that
`e0 = e_nuc + e_core + eh + ex + ec - e_sigma_trace`. These are the `Enuc`,
`Ecore`, `Eh`, `Ex`, `Ec` columns of the paper's supporting data.

A restricted reference is one spin channel throughout: the general layer
builds the exact J and K of the environment once and transforms the active
integrals once, and the result goes to the restricted (spin-free) solver.
An unrestricted reference is two channels, alpha and beta, through the same
code; a restricted calculation computes nothing twice and gives the numbers
it gave before the second channel existed.

## What is built

H_eff = E_0 + sum h_eff,tu a†_t a_u + 1/2 sum (v + v~)_tu,vw a†_t a†_v a_w a_u,
with the active-space orbitals t, u, v, w:

- **v** is the bare Coulomb interaction on the active orbitals, always the
  exact 4-centre integral, never a density-fitted one.
- **v~** is the screening correction (zero for `hf` and `bare`). For `crpa` it
  is the static (omega = 0) screening of the *residual* polarizability, the
  one with the active particle-hole pairs removed, so that v + v~ is the
  partially screened W^r. `crpa_ph` keeps only the elements of v~ with a
  particle-hole pair in them, those that enter as (ia|jb). `mrpa`
  (Scott and Booth, PRL 132, 076401 (2024), Eq. 5; `mode='casida'` only)
  replaces the particle-hole elements by the static interaction whose RPA
  on the active particle-hole pairs alone has the full RPA's zeroth moment
  eta0 = (X+Y)(X+Y)^T on those pairs (and its first, A - B = Delta, by
  construction): (ia|jb)_eff = 1/4 [eta0_S^-1 Delta_A eta0_S^-1 - Delta_A],
  v~ = (ia|jb)_eff - (ia|jb), with eta0_S the active block of
  (X+Y)(X+Y)^T of the full singlet Casida solve, normalized so that
  (X+Y) Omega^-1 (X+Y)^T = (Delta + 4K)^-1, with no further factor: in spin
  orbitals a singlet mode puts (X+Y)/sqrt(2) on each spin and a triplet
  mode of the direct RPA has X+Y = 1 (A + B = A - B = Delta), so the
  same-spin block of eta0 is (eta0_S + 1)/2 and the opposite-spin block
  (eta0_S - 1)/2, and Eq. 5 gives 1/4 [eta0_S^-1 Delta eta0_S^-1 - Delta]
  in both blocks and nothing in the triplet channel. v~ is zero outside
  the particle-hole elements and zero with every orbital active. Only
  omega = 0 is implemented.
- **h_eff = f - t^DC.** f is the Fock operator of the inactive core, built
  from h^(0) and the bare v. t^DC removes what v~ would otherwise count twice:
  the Hartree and exchange of v~ with the active reference density. For
  `crpa_ph` this is the same general formula with v~ zero outside the
  particle-hole block; its exchange-like part is what survives, and it
  changes the virtual-virtual block of the one-body term. Nothing is
  special-cased for the ph screenings.

E_0 = E_nuc + E_HF,core + E_HF,ee + E_HF,ee[v~] + E_c[v] - E_c^A[v_eff]:

- E_HF,core and E_HF,ee are the HF energy of the inactive core alone; the
  couplings of core and active space are in f.
- E_HF,ee[v~] = 1/2 Tr[t^DC rho^A,0] is the HF energy of v~ in the active
  reference determinant, entering with a plus sign, so that the reference
  determinant of H_eff keeps the energy E_HF + E_c[v] - E_c^A.
- E_c[v] is the RPA correlation energy of the whole system, E_c^A[v_eff] that
  of the active space alone, both by the plasmon formula and **on the same v
  as the full RPA**. With every orbital active the pair cancels to rounding,
  and with none active E_0 = E_HF + E_c[v]. `bare` has v~ = 0, so it keeps
  the difference E_c[v] - E_c^A[v] (a mechanical embedding). With `hf` both
  E_c terms are zero.

Every formula has a spin-orbital and a restricted form (spin-summed, spatial
orbitals); the code is written per spin channel and contracts the two
identical channels of a restricted reference at the end. For the restricted
forms, with K_ia,jb = (ia|jb), I the inactive and A_o the active occupied
orbitals, `Delta` the orbital-energy differences:

| quantity | restricted |
|---|---|
| f_tu | h_tu + sum over I of [2 (tu\|ii) - (ti\|iu)] |
| t^DC_tu | sum over A_o of [2 v~_tu,kk - v~_tk,ku] |
| E_HF,ee[v~] | sum over k, l in A_o of [2 v~_kk,ll - v~_kl,lk] |
| E_RPA,corr | 1/2 sum Omega - 1/2 sum_ia (Delta_ia + 2 K_ia,ia); Omega from Delta^1/2 (Delta + 4K) Delta^1/2, singlet modes only |
| cRPA v~_tu,vw | -4 (tu\|ia) [(Delta + 4K)_rr^-1]_ia,jb (jb\|vw), "rr" the residual pairs |
| mRPA (ia\|jb)_eff | 1/4 [eta0_S^-1 Delta_A eta0_S^-1 - Delta_A]; spin orbitals: 1/2 [eta0_A^-1 Delta_A eta0_A^-1 - Delta_A] |

## Unrestricted references

A `scf.UHF` or `dft.UKS` mean field runs through the same two calls. The
mean-field object alone decides: RHF/RKS is the restricted path, UHF/UKS the
unrestricted one, also when its alpha and beta orbitals coincide (nothing is
collapsed). The alpha and beta orbitals, orbital energies and occupied counts
differ, each spin has its own active space, and H_eff has different alpha and
beta blocks.

```python
from pyscf import gto, scf

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import unrestricted_from_counts
from src.EffectiveHamiltonian.solvers.solve import solve

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1)   # OH
mf = scf.UHF(mol).run()
nocc = tuple(int((occ > 0).sum()) for occ in mf.mo_occ)            # (5, 4)
active = unrestricted_from_counts(mf.mo_energy, nocc, 4, (3, 2))

low = run_low_level(mf, method='rpa', mode='casida')
g_eff, space, energies = build_effective_hamiltonian(
    low, active, screening='crpa', sigma='srg')
out = {}
roots = solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
              spin_free=space['spin_free'], nroots=2, out=out,
              spin_overlap=space['spin_overlap'])
print(roots, out['spin_square'])
```

**The active space is a pair** `(space_alpha, space_beta)` of `ActiveSpace`
objects, each built from that spin's orbital count and occupied count, with
independent orbital indices and the same number of active orbitals. The
number of active electrons is `(N_alpha, N_beta)`, the active occupied counts
of the two spins. `unrestricted_from_counts(eps, nocc, n_act, nelec_active)`
builds the contiguous pair (per spin the highest occupied and the lowest
virtual orbitals, pyscf UCASCI's convention). Any other choice is two
`ActiveSpace.from_indices(norb, nocc_spin, occ, virt)`, made in the calling
script; no selection by character or by overlap between the spins is
provided. One `ActiveSpace` with an unrestricted reference, a pair with a
restricted one, two spaces of different size, and a space built with the
other spin's counts are refused. Both selections may be empty.

**What comes back.** With `spin_free=True` (the default)

    g_eff = {0: E_0, 1: (h_alpha, h_beta), 2: (aa, ab, bb)}

each two-body block in the physicist order of the restricted `g_eff[2]`,
`block[p,q,r,s] = <pq|rs>`, the opposite-spin one
`<p_alpha q_beta | r_alpha s_beta>`. With `spin_free=False` it is the same
2 n_act interleaved spin orbitals as for a restricted reference (alpha at
2p, beta at 2p+1), built from those blocks. In `space`, `n_elec_active` is
`(N_alpha, N_beta)`, `act_idx`, `act_occ_idx`, `act_vir_idx` and
`active_space` are pairs (alpha, beta), and `spin_overlap` holds the
alpha-beta overlaps of the orbitals that `solve` needs for <S^2>. Every term
of `energies` is the total over both spins. Diagnostics that belong to one
spin are pairs (the pole-free window, the pole distances, Z, the
continuation spread); a summary of the whole stays one number (the static
defect, the largest over the three blocks; the SRG damping, the largest over
both spins). Messages that name an orbital name its spin
("beta orbital 6").

**The solver.** `'pyscf_fci'` solves the per-spin blocks with
`fci.direct_uhf` at the given `(N_alpha, N_beta)`. In spin-polarized
orbitals H_eff conserves S_z only, so the roots are the lowest states of
that S_z of any spin, and are not spin eigenfunctions in general.
`solve(..., out=out, spin_overlap=space['spin_overlap'])` puts <S^2> of each
root in `out['spin_square']`: that of the whole molecule's state, the active
CI vector with the inactive core determinant, from the true alpha-beta
overlaps. (pyscf's own `spin_square` assumes alpha orbital p and beta
orbital p are one function: for the triplet root of a broken-symmetry H2
it gives 0 instead of 2.) `'openfermion'` takes the spin-orbital form with
N_alpha = N_beta only; the DMRG and coupled-cluster backends refuse a
spin-resolved H_eff and name `'pyscf_fci'`.

**The formulas** are those of the restricted case written over the
particle-hole pairs of both spins, stacked (the alpha pairs, then the beta
ones); the code is one implementation with the spin weight g, the number of
electrons an occupied orbital of one channel holds: 2 for the one channel
of a restricted reference, 1 for each of two.

| quantity | with g | restricted (g = 2) | unrestricted (g = 1) |
|---|---|---|---|
| RPA problem | A + B = Delta + 2g K | Delta + 4K, singlet | Delta + 2K on the stacked pairs, K_(ia s),(jb s') = (i_s a_s\|j_s' b_s') |
| W(0) - v, cRPA v~ | -2g V (Delta + 2gK)_rr^-1 V^T | -4 | -2, between each spin's own active orbitals: three blocks |
| residues of W | sqrt(g) sum (tr\|ia)(X+Y) | sqrt(2) | 1, the sum over both spins' pairs |
| chi0(0) | -2g sum B_ia B_ia^T / Delta_ia | -4 | -2 per spin, summed |
| mRPA (ia\|jb)_eff | 1/(2g) [eta0_A^-1 Delta_A eta0_A^-1 - Delta_A] | 1/4, singlet eta0 | 1/2, eta0 of both spins |
| active RPA | Delta_A + 2g K^A | Delta_A + 4 K^A | Delta_A + 2 K^A on the active pairs of both spins |

W is one spin-independent operator, so the three blocks of v~ are the same
W^r - v evaluated between alpha and alpha, alpha and beta, beta and beta
orbitals. For a spin-polarized reference the same-spin and opposite-spin
blocks of the mRPA correction differ; for a closed shell they coincide.
v~ for `crpa_ph` and `mrpa`, E_c^A and the correction of the static defect
vanish only when neither spin has an active particle-hole pair; a spin
without one (a triplet with N^A = (2, 0), say) is fine.

**Sigma^E** is built per spin, with that spin's residues (or, on the
numerical routes, that spin's factors and its own mid-gap), its orbital
energies and the excitation energies of the two-spin RPA, which both spins
share; one `sigma` and one `srg_s` serve both. The pole-free window is per
spin. Its Omega_min is the lowest excitation of the two-spin problem, which
includes the spin-density modes: for a closed shell these carry no weight in
W, so the window of a UHF run on a closed shell is slightly narrower than
that of the RHF run (water/6-31G: 0.705 against 0.728 Ha). That is on the
safe side: the near-pole warning can come too often, never too rarely.

**The routes.** `'casida'` takes every unrestricted reference.
`'imagfrequency'` and `'space-time'` need occupied and virtual orbitals in
both spins and refuse a reference without them (triplet H2 has no occupied
beta orbital), pointing to casida. On `'space-time'` the ISDF factors hold
one collocation per spin on a shared Coulomb factor; the `'rows'`
realization of the fit builds one collocation and is refused for an
unrestricted reference (`numerics.isdf.fit='replicated'`, the default,
works). On the numerical routes Omega_min is the lowest root of the
two-spin problem on the stacked pairs, found as for a restricted reference
(below). `'mrpa'` is casida only, as for a restricted reference.

**Not supported.** An open shell run restricted (ROHF/ROKS) is refused: its
orbitals are not canonical for the alpha and beta Fock operators; run it as
UHF/UKS. A UHF solution whose occupied orbitals are not the lowest of their
spin (a symmetry-constrained or MOM solution) is refused, naming the spin,
and so is one with fractional occupations (`frac_occ`, a smearing): it is
not one determinant, and counting each fractionally occupied orbital as
occupied would describe a system with another number of electrons.

**Spin contamination and instability are reported, not refused.**
`run_low_level` runs the reference diagnostics of the GW driver on a UHF/UKS
determinant: <S^2> against S(S+1), and pyscf's internal stability analysis.
A determinant more than 0.1 above S(S+1) (broken-symmetry H2: 0.98 against
0), or one pyscf finds unstable, is taken with a `RuntimeWarning` and the
named flag `'uhf_spin_contaminated'` or `'uhf_reference_unstable'` in
`low.numerical_flags` and in `energies.numerical_flags` of every Hamiltonian
built on it; the numbers are in `mf.reference_diagnostics`. They are
computed once per converged mean field, so the warning comes with the first
run on it and the flags with every run.

**A symmetry-broken reference stays visible in H_eff.** H_eff is built on
the UHF determinant, and its states inherit what that determinant breaks.
For the OH radical, whose 2-Pi ground state is doubly degenerate, the two
lowest states of the (5,4) H_eff are the two components, and they are split
by 0.97 eV with `method='hf'`, 0.90 eV with `crpa_ph` and `mrpa`, 3.6 eV
with `crpa`, and 0.02 eV with `crpa` and `sigma='srg'` (cc-pVDZ; the
example below prints them).

## Where the code differs from the printed paper

The code follows the self-consistent convention, which is the one the paper's
own appendix and its numerical data use:

| item | printed | code |
|---|---|---|
| A, B (Eq. 2.8) | A = Delta - v, B = -v | **A = Delta + v, B = +v** (as Appendix B and Eq. 2.18 need) |
| E_RPA,corr (Eq. 2.22) | mixed forms of the sums | **1/2 Tr[Omega - A]** |
| E_HF,ee[v_eff] | not written | **1/2 Tr[t^DC rho^A,0]** (matches the SI's Hartree and exchange terms of H2 cRPA) |
| E_HF,core, E_HF,ee (Eq. 2.6) | environment | **environment only**, the active-environment terms are in f |
| t^DC for cRPA_ph, mRPA | exchange-like only | **no special case**; the Hartree part vanishes because v~ has no (kk\|ll) element |

**Deletion is the paper's zeroing.** The paper makes the residual response by
zeroing the active-active block of A and B (Eq. 2.14). The code instead
deletes the active pairs (occupied in the active space, virtual in the active
space) from the Casida matrix and solves the smaller problem once. The two are
the same: zeroing A + B on that block alone is wrong (it leaves a block that
screens), but zeroing both A + B and A - B makes the Casida matrix block
diagonal, with zero modes of vanishing amplitude on the active block and
exactly the deleted problem on the rest. The two agree to 4e-15 in the tests.
Cross pairs (active occupied to environment virtual, environment occupied to
active virtual) and environment-environment pairs stay in the residual
response.

## The environment self-energy Sigma^E (`sigma=`)

The paper keeps only the screened interaction and drops the self-energy the
environment gives the active orbitals. With the same assumption as the paper
(delta v_eff / delta G = 0) the correlation self-energy of the embedded
active space is, in general,

    Sigma_emb = iG^E (W - v) + iG^A v~ + iG^A (W - W^A[v_eff]),

W the full screened interaction and W^A[v_eff] the active-space RPA screened
interaction built from v_eff. `sigma='srg'` or `'mode_a'` keeps the first two
terms and drops the third, for every screening:

| screening | W^A[v_eff] against W | the dropped term iG^A (W - W^A[v_eff]) |
|---|---|---|
| `crpa` | equal at omega = 0, differ at omega != 0 (v_eff is static) | the paper's static approximation |
| `crpa_ph` | the same active response, bare outer legs on non-ph pairs | nonzero, also at omega = 0 |
| `mrpa` | the same zeroth moment on the active ph pairs, not the same W(0) | nonzero, also at omega = 0 |
| `bare` | active-space screening only | the environment screening `bare` leaves out by design |

So the two-term split is exact only for `crpa`, and only in the static limit;
the `'static_defect'` diagnostic measures W(0) - W^A[v_eff](0) for the others.

The two kept terms change the one-body term and E_0 together, in one code
path (`dc='hartree'` in `DownfoldingTerms`):

- **One-body.** iG^A v~ is the exchange of v~ with the active reference
  density, so it cancels the exchange half of t^DC: only the Hartree double
  counting is left, h_eff = f - J~[gamma^A,0] + Sigma_bar^E (as in QDET).
  For `crpa_ph` and `mrpa` v~ has no (tu|kk) element, the Hartree part
  vanishes and h_eff = f + Sigma_bar^E.
- **E_0** gains the exchange energy term and -Tr[Sigma_bar^E gamma^A,0], at
  the same static Sigma_bar^E the solver sees (not in the paper). Together
  they undo what the one-body change does to the reference determinant,
  which keeps the energy E_HF + E_c[v] - E_c^A. For `crpa_ph` and `mrpa`
  the exchange term is zero.

| quantity with Sigma^E | restricted |
|---|---|
| t^DC_tu | sum over k in A_o of 2 v~_tu,kk (Hartree only) |
| exchange energy term | + 2 K~, K~ = sum over t, v in A_o of v~_tv,vt |
| trace term | - 2 sum over t in A_o of Sigma_bar_tt |

**The static reduction.** iG^E(W - v), with the internal orbital r over the
environment (inactive and environment virtual orbitals), is a pole sum
S_tu(omega) of the residues w of the full W (Casida route: from the RPA's
X + Y) with denominators D_tr = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n.
Every energy (eps_t, eps_r, Omega_n) is the mean field's. It is made static
in one of two ways (Marie and Loos, JCTC 19, 3943 (2023)):

- `'srg'` (recommended): Eq. 44 at s = `numerics.downfolding.srg_s`
  (default 500 Ha^-2),
  Sigma_bar_tu = sum_rn (D_tr + D_ur)/(D_tr^2 + D_ur^2) w_tr w_ur
  [1 - exp(-(D_tr^2 + D_ur^2) s)].
  Finite everywhere: a term at an exact pole is 0, its limit. `srg_s=None`
  is s = infinity, Eq. 42.
- `'mode_a'`: Eq. 12 at eta = 0, Sigma_bar_tu = 1/2 [S_tu(eps_t) +
  S_tu(eps_u)], the form of the user's PySCF prototype. Undamped, so it
  diverges at the poles.

At s = infinity the diagonal of `'srg'` is that of `'mode_a'`, and since
gamma^A,0 is diagonal every E_0 term is the same for the two; only the
off-diagonal h_eff differs.

**The pole check** (always on with `sigma`). Every pole of S lies outside the
window (max over inactive i of eps_i - Omega_min, min over environment
virtuals a of eps_a + Omega_min). `'mode_a'` warns
(`SigmaNearPoleWarning`, named flag `'sigma_near_pole'`), with the orbitals
named, when an active orbital energy is within `numerics.downfolding.
pole_margin` / sqrt(2 s) of an edge (default 3 widths, 0.095 Ha at s = 500)
or outside the window. Sigma_bar is computed and H_eff built all the same;
the elements on those orbitals are then dominated by the nearest pole.
`'srg'` with a finite `srg_s` does not warn of the poles on the Casida
route; near a pole its damping matters, which the `'sigma_poles'` diagnostic
reports. At `srg_s=None` it is as undamped as `'mode_a'` and warns the same
way.

Nothing is refused near a pole. Only a Sigma_bar that is not finite, an
active orbital energy exactly on a pole with `'mode_a'`, is
(`FloatingPointError`).

`srg_s`, `pole_margin` and `continuation_margin` are read from `low.numerics`,
the `Numerics` given to `run_low_level`. They act after the RPA, so a build
can take other values on the same low-level result:
`build_effective_hamiltonian(low, active, sigma='srg', numerics=other)` uses
the `downfolding` section of `other` for that Hamiltonian and leaves `low` as
it is; a scan over `srg_s` pays for the RPA once. Every other section of
`other` has to equal the run's, which used it (`NumericsMismatch`, a
`ValueError`, otherwise); `low.numerics.derive('why', downfolding={...})`
makes such an object.

**Sigma^E on `mode='imagfrequency'`.** There are no residues on this route.
S^E(mu + i omega), mu the whole system's mid-gap, comes from the GW
density's Lorentzian fit of W - 1 on the minimax W grid
(`numerics.imagfreq.nfreq` points; W built on the first Sigma^E request and
kept), with the internal orbital restricted to the environment. It is
sampled on the `numerics.quadrature.npade` Pade nodes (16 Gauss-Legendre
points of scale `w0_gl`, the GW Pade settings), continued by one Pade fit
per element, and each element is evaluated at its own point:

- `'srg'` is Eq. 42, Sigma_bar_tu = Re S_tu(z_tu) with z_tu = (eps_t +
  eps_u)/2 + i (eps_u - eps_t)/2, because (a + b)/(a^2 + b^2) = Re[(1 +
  i)/(a + i b)]. That is **s = infinity**: `srg_s` enters only the pole
  check here, and `'sigma_poles'` records `srg_s` None.
- `'mode_a'` is 1/2 Re[S_tu(eps_t) + S_tu(eps_u)] on the real axis.

Without damping, both forms warn near the poles on this route
(`SigmaNearPoleWarning`), and with a wider margin than on casida: within the
larger of pole_margin / sqrt(2 s) (s = `srg_s`, or 500 when it is None) and
`numerics.downfolding.continuation_margin` (default 0.26 Ha) of the window's
edge, or outside the window. The message names the orbitals and points to
`mode='casida'`, where `'srg'` is finite everywhere. Sigma_bar is continued
and H_eff built all the same, but **inside the margin the continued value is
not reliable**: on water its elements on the named orbitals were off by
1e-4 Eh at 0.16 Ha from the edge, 3.5e-4 at 0.11 Ha and 7.0e-4 at 0.09 Ha,
and no case closer than that or outside the window was measured. The
`'sigma_continuation'` spread does not show this error (3.6e-5 at 0.09 Ha),
so the pole warning is the only sign of it.

Omega_min of the window is the lowest root of the same Casida problem on
the route's own v, for one spin channel or two, computed once and kept. It
is found without building that problem. In the direct RPA A - B is the
diagonal delta of orbital-energy differences, so the Omega^2 are the
eigenvalues of M = delta^1/2 (delta + 2g B^T B) delta^1/2, B the
occupied-virtual fitting factors. A Lanczos iteration on M^-1, which the
Woodbury identity gives from the P(0) the route already has, finds the
lowest one in some tens of applications of B (through the collocation on
space-time, without the three-index object); Omega_min^2 is then the
Rayleigh quotient of M itself, so the quadrature error of space-time's
P(0) does not reach it. It agrees with the casida route's lowest root to
4e-14 Ha or better on water, NH3, N2, benzene, OH and CH2.

Omega_min changes no number of H_eff, only where the pole warning starts,
so an iteration that does not finish does not stop the build. Omega_min is
then replaced by its exact lower bound, the smallest orbital-energy gap,
with a `SigmaNearPoleWarning` and the named flag
`'sigma_pole_window_bound'`. The window is then narrower than the true one:
the near-pole warning can come for an orbital that is not near a pole,
never the other way.

The measured accuracy: the exact Casida S^E fed through the continuation
alone misses the exact Sigma_bar by at most 1.8e-7 Eh per element on
benzene (6,6) cc-pVDZ (2e-9 on water), against a target of 1e-4; the whole
route agrees with casida to 1.6e-7 Eh on benzene and 1e-7 on water, and E_0
to 2.5e-10. Towards the window's edge the continuation amplifies the small
error of the sampled S^E (2e-8): on a water space with a high virtual active
the error reaches 1e-4 at 0.16 Ha from the edge and 7e-4 at 0.09 Ha. Every
case measured beyond 0.17 Ha met 1e-4 (a few much closer did too), and
`continuation_margin` is 1.5 times that; the scan is in the comment of
`DOWNFOLDING_CONTINUATION_MARGIN` in `src/Base/numerics/constants.py`.

**Sigma^E on `mode='space-time'`** is sampled, continued, warned about and
diagnosed exactly as on imagfrequency, on the route's own objects: W on the
same minimax frequency grid (`numerics.imagfreq.nfreq`), from chi0 of the
ISDF factors (built on the first Sigma^E request and kept), its Lorentzian
fit convolved exactly with the environment's G, and the active-environment
fitting factors B_Q,tr = sum_P D_PQ X_Pt X_Pr. Only the active block is
needed, so the cost stays cubic, times the number of active orbitals.

Sampling S^E through the imaginary-time quadrature of the space-time GW
self-energy does not work here: it leaves about 1e-7 of noise on the
continuation's nodes, which the continuation amplifies to 1e-4 and more for
an active orbital far from the gap (2.2e-4 on an NH3 space with virtuals
1.2 Ha above the mid-gap). The fit's samples are an exact rational function
and continue cleanly. Against the Casida problem on the same ISDF factors,
which isolates the continuation, this was checked on 5 cases (three active
spaces of benzene and two of NH3, cc-pVDZ): 2.8e-8 on the nodes (the NH3
cases) and at most 9.3e-7 Eh per element of Sigma_bar^E. The scan over 160
active spaces of water, NH3 and benzene compares with casida on the mean
field's RI factors, where the difference also holds the ISDF fit of the
integrals: beyond the 0.26 Ha margin up to 4.2e-5 Eh on water, 8.2e-6 on
NH3 and 8.5e-5 on benzene (G1 grid). The FCI energy of H_eff with
`sigma='srg'` agrees to 1.2e-6 Eh on NH3.

## Kohn-Sham references

A restricted or unrestricted KS mean field (GGA or hybrid) runs, with the KS
orbitals and energies. f and t^DC are still built from h^(0) and the bare v, never from the
KS Fock matrix, and E_HF,core and E_HF,ee are the HF energy expression on the
KS orbitals, without any XC energy. The RPA uses the KS orbital energies.
This is what the paper's SI data for H2/PBE show. The paper's Appendix A
guarantees no double counting only for an HF Green's function; for a KS
reference E_0 is a well-defined working choice, not a derived one.

## Checks that are always on

- **Degenerate boundary.** An active orbital and an environment orbital of
  the same kind (both occupied or both virtual) closer than
  `QP_DEGENERACY_TOL` (1e-4 Eh) are refused, with the orbital numbers and
  energies on both sides: include or exclude the whole set (`from_irreps`,
  `from_population`). Arbitrary rotations inside a degenerate set would
  otherwise change single elements of H_eff.
- **Mean field.** Gas phase, T = 0 only; restricted closed-shell or
  unrestricted with the lowest orbitals of each spin occupied by one
  electron each. Open-shell ROHF, solvent models and, for UHF/UKS,
  fractional occupations (a smearing, pyscf's `frac_occ`) are refused.
- **Serial only.** More than one MPI rank is refused (every rank would repeat
  the whole calculation).
- **Option validation.** An unknown or not-yet-available mode, screening,
  sigma or diagnostic is refused with the valid list. Their names are taken
  in any case and with `-` or `_` (`'SRG'`, `'mode-a'`, `'CRPA-PH'`); for
  `sigma` only `None` itself means no Sigma^E (the string `'None'` is
  refused).
- **Sigma^E poles.** `sigma='mode_a'` warns near a pole of the
  environment self-energy (`SigmaNearPoleWarning`, named flag
  `'sigma_near_pole'`); on `mode='imagfrequency'` and `'space-time'` both
  `'srg'` and `'mode_a'` do, within the wider continuation margin (see
  above). H_eff is built all the same and the flag is in
  `energies.numerical_flags`; only a Sigma_bar that is not finite is
  refused.
- **Sigma^E warnings.** With `sigma` set, two checks of Sigma_bar always
  run, whatever `diagnostics` asks for, because Sigma_bar goes into H_eff
  and E_0 either way: a `SigmaDampingWarning` (casida, `'srg'`) when the
  SRG damping moves an element by more than 1e-6 Ha, and a
  `SigmaContinuationWarning` (imagfrequency, space-time) when two Pade fits
  disagree by more than 1e-4 Ha. Both are named flags
  (`docs/warnings.md`). The `diagnostics` keyword only decides whether the
  numbers behind them are recorded.
- **Named flags in the result.** A warning that is a named flag also leaves
  a `{'flag': name, 'message': text}` entry in the result, so a script can
  test for it without catching warnings: `low.numerical_flags` holds the
  flags of `run_low_level`, and `energies.numerical_flags` those plus the
  ones raised by `build_effective_hamiltonian` for this active space. Both
  are empty lists for a clean calculation.

Diagnostics are informational and do not change a number. `diagnostics=` takes
a tuple of names (default: all; `None` is the same as `()`); a name that does
not apply to the call is skipped, so one tuple works across a loop. A
diagnostic that applies but is exactly zero (an empty active space, no
environment, no particle-hole pair) is recorded as 0.0; one that does not
apply is absent. Results are in
`energies.diagnostics`, keyed by name. On the casida route two apply; on
imagfrequency and space-time all four:

- `'static_defect'`: the largest element of W(0) - W^A[v_eff](0) on the
  active orbitals. It is zero for `crpa`, by construction, and measures how
  far `bare`, `crpa_ph` and `mrpa` are from it.
- `'sigma_poles'` (with `sigma` only): a dict with `'window'`, the pole-free
  window (lo, hi); `'pole_distance'`, per active orbital the distance of
  its energy to the nearer edge (negative outside); `'z'`, per active
  orbital Z_t = 1/(1 - S'_tt(eps_t)) = 1/(1 + sum w_tr^2 / D_tr^2), 1 far
  from the poles and small near one; `'srg_s'`, the s applied (None for
  s = infinity and for `'mode_a'`); and `'srg_damping'`, the largest
  |Sigma_bar(s) - Sigma_bar(infinity)| (0 when no finite s is applied).
  When the damping exceeds `DOWNFOLDING_SRG_DAMPING_WARN` (1e-6 Ha) a
  `SigmaDampingWarning` names the active orbitals concerned: their
  Sigma_bar depends on `srg_s`. On imagfrequency and space-time `'z'`
  comes from the continued fraction, `'srg_s'` is None and `'srg_damping'` 0.0 (s =
  infinity, no damping applied).
- `'ec_active_casida'` (imagfrequency, space-time): `{'casida': the
  dense-Casida E_c^A on the same v_eff, 'difference': the quadrature value
  minus it}`.
- `'sigma_continuation'` (imagfrequency, space-time, with `sigma`): per
  element of Sigma_bar, the difference between two Pade fits on the even-
  and the odd-indexed nodes at the element's evaluation point. It is a
  lower-bound-type indicator, not an error bound: near the window's edge
  it falls short of the true error by ten to twenty times (3.6e-5 against
  7.0e-4 at 0.09 Ha from the edge, a space the pole warning names), so a
  small spread does not certify Sigma_bar there; far from
  the poles it can exceed the error (1.2e-5 against 1.6e-7 on benzene).
  Above `DOWNFOLDING_SIGMA_CONTINUATION_WARN` (1e-4 Ha) a
  `SigmaContinuationWarning` names the elements: use more nodes or
  `mode='casida'`.

## What is checked against what

- A brute-force oracle that builds H_eff from the definitions with dense
  matrices, on the same integrals (agreement 1e-8 Eh).
- The paper's SI numbers for H2 (2 electrons in 2 orbitals) and N2 (6 in 6),
  with spherical basis functions and exact integrals. The energy terms agree
  to at most 1.6e-5 Eh for H2 and 6.8e-4 Eh for N2 (the N2 numbers are the
  absolute HF terms; the screening-dependent differences agree far better),
  for every screening including `mrpa`. The SI comes from a different code
  and basis representation. Benzene, whose geometry the SI does not give,
  on the QUEST geometry instead: the Frobenius norms of v~ of the (6,6) pi
  space agree to 1.5e-3 (crpa) and 2e-4 (crpa_ph, mrpa) relative, and the
  lowest environment excitations Omega^R of the (10,10) space lie 1 to 3
  mEh above the SI's, the size a 0.5% change of the geometry makes.
- mRPA: the restricted form equals Eq. 5 written in spin orbitals (both
  spins, every singlet and triplet mode) to 2e-13, whose same-spin and
  opposite-spin blocks are equal; the RPA of the active pairs with the
  mRPA v_eff has the full RPA's projected zeroth moment; mRPA and crpa_ph
  norms differ by 8% on benzene (6,6), as in the SI.
- The user's PySCF prototype, run on its own orbitals: v_eff, f, t^DC, h_eff,
  the excitation energies and the CAS energies of the ethylene (2,2) and
  benzene (6,6) pi spaces for `bare`, `crpa` and `crpa_ph`, to machine
  precision, and its Sigma_bar (`'mode_a'`). The orbitals are pinned because
  the degenerate pi pairs of benzene rotate arbitrarily from one SCF to the
  next.
- Sigma^E: `'srg'` with every orbital as the environment is the active block
  of the qsGW SRG kernel; both forms equal the oracle's pole sums; the
  reference determinant keeps its energy when the switch is turned on (with
  the real Sigma_bar and with Sigma_bar forced to 0); every orbital active
  still gives FCI; along a scan through a pole crossing `'srg'` stays smooth.

- Unrestricted references (no published numbers exist): the same
  determinant handed over as RHF and as a UHF object gives the same H_eff
  and energy terms on every route (1e-13 on casida); with every orbital of
  both spins active the screening vanishes, the E_c pair cancels and H_eff
  gives the FCI energy of the molecule; with none active E_0 is E_UHF + E_c;
  the reference determinant of H_eff has the energy
  E_UHF + E_c - E_c^A for every screening, with and without Sigma^E.
  Independently of the code: the oracle written in spin orbitals without
  any spin adaptation, for E_c, v~ (cRPA, mRPA), the residues and both
  Sigma^E forms, on spaces with different orbitals per spin,
  N_alpha != N_beta and unequal cores (agreement 2e-13); pyscf's UCASCI for
  `method='hf'` and its URPA for E_c on the same density fit; mRPA
  reproduces the full RPA's zeroth moment on the active pairs. The
  numerical routes against casida on the same factors: the mean field's own
  density fit on imagfrequency, the route's own ISDF factors on space-time
  (the FCI energy of H_eff to 3e-8 Eh, Sigma_bar to 1e-6 Eh). Molecules:
  the OH radical, triplet CH2, broken-symmetry H2, and OH with PBE0.

Use spherical basis functions (`cart=False`, the PySCF default) when comparing
with the paper: Cartesian ones are far further from its numbers.

## Not yet available

- Natural orbitals (RPA or GW), semicanonicalization and the Delta
  hybridization; the active space is the user's.
- omega other than 0, and other ways of making W^r static.
- Self-consistent energies in Sigma^E; the term iG^A(W - W^A[v_eff]) for
  `bare`, `crpa_ph` and `mrpa`.
- For unrestricted references: ROHF/ROKS; spin-adapted states (H_eff
  conserves S_z only); the DMRG and coupled-cluster backends on a
  spin-resolved H_eff; selecting the two active spaces by character or by
  alpha-beta overlap.
- Further low-level methods (MP2, TDHF/BSE, CC polarizability), a MACIS
  backend and MPI-parallel downfolding.

Examples: [`examples/22_rpa_downfolding.py`](../../examples/22_rpa_downfolding.py)
(ethylene and benzene pi spaces) and
[`examples/23_unrestricted_downfolding.py`](../../examples/23_unrestricted_downfolding.py)
(the OH radical on a UHF reference, all three routes, <S^2>).
