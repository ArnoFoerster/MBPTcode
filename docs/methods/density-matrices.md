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

**Numerics.** The two numerical modes require `numerics`. The inner
integrals use the existing sections: `imagfreq` for W, and `isdf`,
`quadrature` and `layout` for the space-time route (the ISDF grid is named
there, as for `solve_qp_energy_space_time`). The outer frequency integral of
G Sigma G reads the `gwdensity` section: `nfreq` (default 48), `grid`
(`'gauss_legendre'`, the default, or `'minimax'`, available only at
tabulated sizes) and `w0` (the Gauss-Legendre scale, by default scaled to
the gap). In `'imagfrequency'` the screened interaction on its minimax grid is
fitted by a sum of Lorentzians and the integral over G is done exactly; the
fit settings are fixed.

Accuracy for water in cc-pVDZ: `'imagfrequency'` against `'casida'` agrees to
about 2e-9 on HF and 7.5e-7 on PBE0 (largest element of the density
correction) at the default outer grid; `'space-time'` against
`'imagfrequency'` agrees to 1.4e-5 on the finest ISDF grid tested.

**Scope.** The numerical modes take RPA screening (`polarizability='RPA'`),
`df=True`, a closed-shell restricted reference, the gas phase and zero
temperature; anything else raises `NotImplementedError` (or `TypeError` for a
missing `numerics`). UHF/UKS references are refused in all modes.

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
- With the outer minimax grid, 24 points reached about 5e-10 against 48
  Gauss-Legendre points on the test molecule. This is an observation only; the
  default stays Gauss-Legendre with 48 points.
