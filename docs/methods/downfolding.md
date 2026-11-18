# Downfolding to an active space (HF and RPA)

`src.MultiReference.Downfolding` builds a static effective Hamiltonian for an
active space from a low-level method on the whole molecule: the orbitals
outside the active space enter as a frozen core (HF) and, for the RPA, as a
screening of the active-space Coulomb interaction. The scheme is that of the
RPA-downfolding paper (cRPA, cRPA_ph and mRPA screening of an active-space
Hamiltonian with a constant E_0), on canonical orbitals. The result goes to
any active-space solver in `src.Solvers.active_space`.

Available now:

- `method='hf'`: the HF limit, which is CASCI.
- `method='rpa'`, `mode='casida'` (dense Casida, every pole) or
  `mode='imagfrequency'` (density fitted, imaginary frequency, N^4), with
  `screening` `'bare'`, `'crpa'` or `'crpa_ph'`, and optionally
  `sigma='srg'` or `'mode_a'`: the static environment self-energy Sigma^E
  that the paper drops (see below).
- E_0 always, term by term.

Not yet available, and refused when asked for: `mode='space-time'` and
`screening='mrpa'` (Casida only once built).

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

**The two routes.** `'casida'` diagonalizes the full Casida problem once and
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

Options of the RPA:

| call | option | values |
|---|---|---|
| `run_low_level` | `mode` | `'casida'`, `'imagfrequency'` (`'space-time'`: not yet available) |
| | `df` | `True`: the screening from the mean field's density fit (an exact eigendecomposition of the MO integrals if it has none); `False`: exact 4-centre integrals, `'casida'` only |
| | `numerics` | a `Numerics`, or `None` for the defaults |
| `build_effective_hamiltonian` | `spin_free` | `True` (default): spatial-orbital `g_eff`; `False`: the 2 n_act interleaved spin orbitals |
| | `screening` | `'bare'`, `'crpa'` (default), `'crpa_ph'` |
| | `sigma` | `None` (default, the paper's scheme), `'srg'` or `'mode_a'` (Sigma^E, below) |
| | `diagnostics` | a tuple of names, default all; see below |

`g_eff = {0: E_0, 1: h_eff, 2: v_eff}` and `space` follow the QDPT
`build_effective_hamiltonian` contract (`n_orbs`, `n_elec_active`, `act_idx`,
`act_occ_idx`, `act_vir_idx`, `spin_free`, `active_space`). Every backend of
`solve` adds `g_eff[0]`, so the roots it returns are **total energies**. The
`ActiveSpace` is the user's choice (`from_indices`, `from_counts`,
`from_irreps`, `from_population`); nothing here selects orbitals.

`energies` is a frozen record holding every term of E_0 separately (`e_nuc`,
`e_hf_core`, `e_hf_ee`, `e_hf_ee_vtilde`, `e_c_full`, `e_c_active`, and the
Sigma^E terms `e_exchange_term`, `e_sigma_trace`, zero without `sigma`), `e0`
for their sum, and `diagnostics`.

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
| `bare` | active-space screening only | the environment screening `bare` leaves out by design |

So the two-term split is exact only for `crpa`, and only in the static limit;
the `'static_defect'` diagnostic measures W(0) - W^A[v_eff](0) for the others.

The two kept terms change the one-body term and E_0 together, in one code
path (`dc='hartree'` in `DownfoldingTerms`):

- **One-body.** iG^A v~ is the exchange of v~ with the active reference
  density, so it cancels the exchange half of t^DC: only the Hartree double
  counting is left, h_eff = f - J~[gamma^A,0] + Sigma_bar^E (as in QDET).
  For `crpa_ph` v~ has no (tu|kk) element, the Hartree part vanishes and
  h_eff = f + Sigma_bar^E.
- **E_0** gains the exchange energy term and -Tr[Sigma_bar^E gamma^A,0], at
  the same static Sigma_bar^E the solver sees (not in the paper). Together
  they undo what the one-body change does to the reference determinant,
  which keeps the energy E_HF + E_c[v] - E_c^A. For `crpa_ph` the exchange
  term is zero.

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
virtuals a of eps_a + Omega_min). `'mode_a'` is refused, with the orbitals
named, when an active orbital energy is within `numerics.downfolding.
pole_margin` / sqrt(2 s) of an edge (default 3 widths, 0.095 Ha at s = 500)
or outside the window. `'srg'` is never refused on the Casida route; near a
pole its damping matters, which the `'sigma_poles'` diagnostic reports.

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

Without damping, both forms are refused near the poles on this route, and
with a wider margin than on casida: within the larger of pole_margin /
sqrt(2 s) (s = `srg_s`, or 500 when it is None) and
`numerics.downfolding.continuation_margin` (default 0.26 Ha) of the window's
edge, or outside the window. The message names the orbitals and points to
`mode='casida'`, where `'srg'` is finite everywhere. Omega_min of the window
is the lowest root of the same Casida problem (dense while it fits in
memory, the Davidson above that), computed once and kept.

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
- **Sigma^E poles.** `sigma='mode_a'` is refused near a pole of the
  environment self-energy; on `mode='imagfrequency'` both `'srg'` and
  `'mode_a'` are, within the wider continuation margin (see above).

Diagnostics are informational and do not change a number. `diagnostics=` takes
a tuple of names (default: all); a name that does not apply to the call is
skipped, so one tuple works across a loop. Results are in
`energies.diagnostics`, keyed by name. On the casida route two apply; on
imagfrequency all four:

- `'static_defect'`: the largest element of W(0) - W^A[v_eff](0) on the
  active orbitals. It is zero for `crpa`, by construction, and measures how
  far `bare` and `crpa_ph` are from it.
- `'sigma_poles'` (with `sigma` only): a dict with `'window'`, the pole-free
  window (lo, hi); `'pole_distance'`, per active orbital the distance of
  its energy to the nearer edge (negative outside); `'z'`, per active
  orbital Z_t = 1/(1 - S'_tt(eps_t)) = 1/(1 + sum w_tr^2 / D_tr^2), 1 far
  from the poles and small near one; `'srg_s'`, the s applied (None for
  s = infinity and for `'mode_a'`); and `'srg_damping'`, the largest
  |Sigma_bar(s) - Sigma_bar(infinity)| (0 when no finite s is applied).
  When the damping exceeds `DOWNFOLDING_SRG_DAMPING_WARN` (1e-6 Ha) a
  `SigmaDampingWarning` names the active orbitals concerned: their
  Sigma_bar depends on `srg_s`. On imagfrequency `'z'` comes from the
  continued fraction, `'srg_s'` is None and `'srg_damping'` 0.0 (s =
  infinity, no damping applied).
- `'ec_active_casida'` (imagfrequency): `{'casida': the dense-Casida
  E_c^A on the same v_eff, 'difference': the quadrature value minus it}`.
- `'sigma_continuation'` (imagfrequency, with `sigma`): per element of
  Sigma_bar, the difference between two Pade fits on the even- and the
  odd-indexed nodes at the element's evaluation point. It is a
  lower-bound-type indicator, not an error bound: near the window's edge
  it falls short of the true error by ten to twenty times (3.6e-5 against
  7.0e-4 at 0.09 Ha from the edge, a space the continuation margin
  refuses), so a small spread does not certify Sigma_bar there; far from
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
  absolute HF terms; the screening-dependent differences agree far better).
  The SI comes from a different code and basis representation.
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

Use spherical basis functions (`cart=False`, the PySCF default) when comparing
with the paper: Cartesian ones are far further from its numbers.

## Not yet available

- `mode='space-time'` and `screening='mrpa'`.
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
