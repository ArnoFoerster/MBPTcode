# Density matrices

**Density matrices** — MP2, MP3 and MP4 correlated 1-RDMs, restricted and
unrestricted, dense and DF, with Laplace-fused amplitude routes. The MP2 and
MP3 densities are also orbital-relaxed, on RHF and UHF references alike. GW
and CCSD/CCSDT density matrices too. Generated with pdaggerq.

## GW density matrix

`compute_gw_density_matrix(mf, ..., mode=, relaxation=, numerics=)` returns
the total AO density of a restricted closed-shell reference: the mean-field
density plus the GW correction. The correction has a dynamic part, the
frequency integral of G Sigma G, and a static part, the exchange minus
exchange-correlation matrix element between occupied and virtual orbitals.
The static part is zero on HF and is always evaluated in closed form.

`mode` chooses how the dynamic part is obtained; all three agree up to
quadrature and fitting error, on HF and on a hybrid reference:

| mode | how Sigma is built | cost |
|---|---|---|
| `'casida'` (default) | analytic, from the explicit Casida problem | O(N⁶) |
| `'imagfrequency'` | imaginary-frequency quadrature of W, then G Sigma G on an outer grid | O(N⁴) |
| `'space-time'` | Sigma from the imaginary-time route on an ISDF factorization | O(N³) |

`relaxation` chooses what happens to the orbitals afterwards:

| `relaxation` | density |
|---|---|
| `'none'` | the unrelaxed GW density |
| `'dyson'` | the density of the linearized Dyson equation: the unrelaxed correction is relaxed with the Hartree-Fock kernel (bare Hartree plus full exact exchange, no f_xc, no scaling of the exchange) on the reference's orbitals and energies |
| `'zvector'` (default) | the relaxed RPA density, relaxed with the reference's own kernel |

On an HF reference the two kernels are the same and `'dyson'` and
`'zvector'` coincide. On a hybrid they differ. The Dyson density has been
benchmarked accurate at high exact-exchange fractions; the Z-vector density
is the derivative of the RPA energy and stays well behaved at low exact
exchange.

```python
from pyscf import gto, scf, dft
from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

numerics = Numerics()
dm = compute_gw_density_matrix(mf, mode='imagfrequency', relaxation='dyson',
                               numerics=numerics)
```

There is no Dyson loop. The static self-energy is linear in the density
change, so the iterated equation is one linear system for the relaxation
increment, solved by the same Krylov solver as the Z-vector equation. The
fixed-point iteration of the original formulation reaches the same answer
and is not offered.

**Integral backend.** The relaxation always uses the mean field's own J/K
build: exact, density-fitted with the same auxiliary basis, or ISDF. It is
never switched silently. To keep a space-time run low-scaling end to end,
build `mf` with the ISDF J/K. `auxbasis` and `screen_r_cut` belong to the
space-time route and are refused with the other modes; `screen_r_cut` is what
realizes the O(N³) scaling and is not forced on.

**Numerics.** `numerics` is required for both numerical modes and for any
`relaxation` other than `'none'`, so the default call (`'casida'` with
`'zvector'`) needs it too: only `mode='casida', relaxation='none'` runs without
one. The inner
integrals use the existing sections: `imagfreq` for W, and `isdf`,
`quadrature` and `layout` for the space-time route (the ISDF grid is named
there, as for `solve_qp_energy_space_time`). The outer frequency integral of
G Sigma G reads the `gwdensity` section: `nfreq` (default 34), `grid`
(`'minimax'`, the default, available only at tabulated sizes up to 34, or
`'gauss_legendre'`, at any size) and `w0_gl` (the Gauss-Legendre scale, by
default scaled to the gap). The minimax grid is fitted to the particle-hole
window, so it holds where the window is wide (deep cores, large basis sets,
small gaps); the mapped Gauss-Legendre grid has one scale and degrades there
(about 1e-5 at 48 points on H2S and HBr, against 1e-8 to 1e-7 for minimax
34). For a window beyond the widest tested (span/gap about 1500), switch to
`'gauss_legendre'` with many points. In `'imagfrequency'` the screened interaction on its minimax grid is
fitted by a sum of Lorentzians and the integral over G is done exactly; the
fit settings are fixed.

Accuracy at the default grids: `'imagfrequency'` against `'casida'` agrees
to about 4e-10 on water and NH3 in cc-pVDZ, HF and PBE0 (largest element of
the density correction), and to 1.3e-7 or better on every system of the
outer-grid study (up to HBr). `'space-time'` against `'casida'` agrees to
about 5e-6 on NH3 in cc-pVDZ (5e-5 with `'dyson'` on PBE0, whose solve
magnifies the error), set by the ISDF grid.

**Scope.** The numerical modes take RPA screening (`polarizability='RPA'`),
density fitting (`df=True`, the default), a closed-shell restricted reference,
the gas phase and zero temperature; anything else raises `NotImplementedError`.
A missing `numerics` where it is required (see above) raises `TypeError`. UHF/UKS references are refused in all modes.

**Changes from earlier versions.**
- The GW density lives in `src.SingleReference.GW.density_matrix`, not in the
  density-matrix package.
- `relax` and `relax_kernel` are replaced by `relaxation`; `relax=False` is
  `'none'`, `relax=True` is `'zvector'`, and `relax_kernel='tdhf'` is
  `'dyson'`. There is no compatibility shim.
- On a Kohn-Sham or generalized-KS reference the static occupied-virtual term
  used to be counted twice. It is now counted once, so results on such
  references change with the default `'zvector'`; HF is unaffected.
- `'dyson'` now relaxes on the mean field's own J/K backend, where before it
  used exact four-center integrals on a density-fitted or ISDF mean field.

**Open items and limitations.**
- `'dyson'` on a solvated mean field relaxes with the bare HF kernel and
  leaves out the solvent response; `'zvector'` keeps it. The numerical modes
  do not take an environment at all.
- Unrestricted references are not supported.
- The iterated-Dyson density at third order is not available (the third-order
  MP density already contains static terms that relaxing it would count
  twice), and orders above three are not consistent within the linearized
  equation. The MP2 density has no `relaxation` keyword yet, and the MP2/MPn
  `relax_kernel='tdhf'` path still builds its kernel with exact integrals
  rather than the mean field's backend; MP3 and unrestricted MP2 stay on
  their present keywords.
- Prebuilt ISDF factors cannot be passed to the density call.
- Whether the Z-vector right-hand side should use the reference's kernel in
  its occupied-occupied and virtual-virtual coupling is unchecked against the
  literature; `'zvector'` behaves as before.
- The conditioning of the Lorentzian fit for small-gap systems has not been
  tested.
- With `'dyson'` on a PBE0 reference and TDHF or BSE screening, NH3 has one
  negative natural occupation: the HF kernel on PBE0 orbitals has a soft mode
  (0.034 Ha) that this right-hand side projects onto strongly.
