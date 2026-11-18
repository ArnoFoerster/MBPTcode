# Downfolding to an active space (HF and RPA)

`src.MultiReference.Downfolding` builds a static effective Hamiltonian for an
active space from a low-level method on the whole molecule: the orbitals
outside the active space enter as a frozen core (HF) and, for the RPA, as a
screening of the active-space Coulomb interaction. The scheme is that of the
RPA-downfolding paper (cRPA, cRPA_ph and mRPA screening of an active-space
Hamiltonian with a constant E_0), on canonical orbitals. The result goes to
any active-space solver in `src.Solvers.active_space`.

Available now (the Casida route, Stage 1):

- `method='hf'`: the HF limit, which is CASCI.
- `method='rpa'`, `mode='casida'`, with `screening` `'bare'`, `'crpa'` or
  `'crpa_ph'`.
- E_0 always, term by term.

Not yet available, and refused when asked for: `sigma='srg'` and
`sigma='mode_a'` (the static self-energy Sigma^E that the paper drops),
`mode='imagfrequency'` and `mode='space-time'` (the low-scaling routes),
`screening='mrpa'`.

## Calling it

```python
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.SingleReference.common.active_space import ActiveSpace
from src.Solvers.active_space import solve

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
`method='hf'` takes no options: `run_low_level(mf, method='hf')`, and
`build_effective_hamiltonian(low_hf, active)` is CASCI.

Options of the RPA:

| call | option | values |
|---|---|---|
| `run_low_level` | `mode` | `'casida'` (the others: not yet available) |
| | `df` | `True`: Casida matrices from the mean field's density fit (an exact eigendecomposition of the MO integrals if it has none); `False`: exact 4-centre integrals |
| | `numerics` | a `Numerics`, or `None` for the defaults |
| `build_effective_hamiltonian` | `spin_free` | `True` (default): spatial-orbital `g_eff`; `False`: the 2 n_act interleaved spin orbitals |
| | `screening` | `'bare'`, `'crpa'` (default), `'crpa_ph'` |
| | `sigma` | `None` (the paper's scheme; `'srg'`, `'mode_a'` not yet available) |
| | `diagnostics` | a tuple of names, default all; see below |

`g_eff = {0: E_0, 1: h_eff, 2: v_eff}` and `space` follow the QDPT
`build_effective_hamiltonian` contract (`n_orbs`, `n_elec_active`, `act_idx`,
`act_occ_idx`, `act_vir_idx`, `spin_free`, `active_space`). Every backend of
`solve` adds `g_eff[0]`, so the roots it returns are **total energies**. The
`ActiveSpace` is the user's choice (`from_indices`, `from_counts`,
`from_irreps`, `from_population`); nothing here selects orbitals.

`energies` is a frozen record holding every term of E_0 separately (`e_nuc`,
`e_hf_core`, `e_hf_ee`, `e_hf_ee_vtilde`, `e_c_full`, `e_c_active`, and the
Sigma^E terms `e_exchange_term`, `e_sigma_trace`, zero for now), `e0` for their
sum, and `diagnostics`.

## What is built

H_eff = E_0 + sum h_eff,tu a†_t a_u + 1/2 sum (v + v~)_tu,vw a†_t a†_v a_w a_u,
with the active-space orbitals t, u, v, w:

- **v** is the bare Coulomb interaction on the active orbitals, always the
  exact 4-centre integral, never a density-fitted one.
- **v~** is the screening correction (zero for `hf` and `bare`). For `crpa` it
  is the static (omega = 0) screening of the *residual* polarizability, the
  one with the active particle-hole pairs removed, so that v + v~ is the
  partially screened W^r. `crpa_ph` keeps only the elements of v~ with a
  particle-hole pair in them, those that enter as (ia|jb). Only omega = 0 is
  implemented.
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

## Where the code differs from the printed paper

The code follows the self-consistent convention, which is the one the paper's
own appendix and its numerical data use:

| item | printed | code |
|---|---|---|
| A, B (Eq. 2.8) | A = Delta - v, B = -v | **A = Delta + v, B = +v** (as Appendix B and Eq. 2.18 need) |
| E_RPA,corr (Eq. 2.22) | mixed forms of the sums | **1/2 Tr[Omega - A]** |
| E_HF,ee[v_eff] | not written | **1/2 Tr[t^DC rho^A,0]** (matches the SI's Hartree and exchange terms of H2 cRPA) |
| E_HF,core, E_HF,ee (Eq. 2.6) | environment | **environment only**, the active-environment terms are in f |
| t^DC for cRPA_ph | exchange-like only | **no special case**; the Hartree part vanishes because v~ has no (kk\|ll) element |

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

## Kohn-Sham references

A restricted KS mean field (GGA or hybrid) runs, with the KS orbitals and
energies. f and t^DC are still built from h^(0) and the bare v, never from the
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
- **Mean field.** Restricted closed-shell, gas phase, T = 0 only. UHF,
  open-shell ROHF and solvent models are refused.
- **Serial only.** More than one MPI rank is refused (every rank would repeat
  the whole calculation).
- **Option validation.** An unknown or not-yet-available mode, screening,
  sigma or diagnostic is refused with the valid list.

Diagnostics are informational and do not change a number. `diagnostics=` takes
a tuple of names; a name that does not apply to the call is skipped, so one
tuple works across a loop. On the casida route one applies: `'static_defect'`,
the norm of W(0) - W^A[v_eff](0) on the active pairs, found in
`energies.diagnostics['static_defect']`; it is zero for `crpa`, by
construction, and measures how far `bare` and `crpa_ph` are from it.

## What is checked against what

- A brute-force oracle that builds H_eff from the definitions with dense
  matrices, on the same integrals (agreement 1e-8 Eh).
- The paper's SI numbers for H2 (2 electrons in 2 orbitals) and N2 (6 in 6),
  with spherical basis functions and exact integrals. The energy terms agree
  to at most 1.6e-5 Eh for H2 and 6.8e-4 Eh for N2 (the N2 numbers are the
  absolute HF terms; the screening-dependent differences agree far better).
  The SI comes from a different code and basis representation.
- The user's PySCF prototype, run on its own orbitals: v_eff, f, t^DC, h_eff,
  the excitation energies and the CAS energies of the ethylene (2,2) and
  benzene (6,6) pi spaces for `bare`, `crpa` and `crpa_ph`, to machine
  precision. The orbitals are pinned because the degenerate pi pairs of
  benzene rotate arbitrarily from one SCF to the next.

Use spherical basis functions (`cart=False`, the PySCF default) when comparing
with the paper: Cartesian ones are far further from its numbers.

## Not yet available

- The static self-energy Sigma^E (`sigma='srg'`, `'mode_a'`), the term the
  paper drops, with its extra E_0 terms; and `mode='imagfrequency'`,
  `'space-time'` and `screening='mrpa'`.
- Natural orbitals (RPA or GW), semicanonicalization and the Delta
  hybridization; the active space is the user's.
- omega other than 0, and other ways of making W^r static.
- Self-consistent energies in Sigma^E; the term iG^A(W - W^A[v_eff]) for
  `bare`, `crpa_ph` and `mrpa`.
- Unrestricted references (the internals are per spin channel, the
  interfaces are not yet).
- Further low-level methods (MP2, TDHF/BSE, CC polarizability), a MACIS
  backend and MPI-parallel downfolding.

Example: [`examples/22_rpa_downfolding.py`](../../examples/22_rpa_downfolding.py)
(ethylene and benzene pi spaces).
