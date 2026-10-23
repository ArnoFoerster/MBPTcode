# torch.autograd oracle for the solvated ISDF / contour-deformation BSE@GW gradient

The production chain (`src/gradients/gwbse/excited_state.py` and what it calls:
`qp_space_time`, `contour_deformation`, `space_time_adjoint`, `bse_isdf`,
`reaction_field_adjoint`) is a hand-written reverse pass from the factors
(eps, X_mo, D_dressed, D_bare) to a BSE@GW excitation energy or a
quasiparticle energy. `torch_chain.py` evaluates the SAME function in torch,
from the same frozen discrete choices (quasiparticle window, residue sets,
Newton seeds, contour-deformation quadrature, static axis), so that
torch.autograd's adjoints can be compared with the production ones entry by
entry. Nothing below the factors is ported: dX/dR, dD/dR and the orbital
response are pyscf-integral chains gated piecewise by finite differences, and
the swap-in step below runs the torch adjoints through that same assembly.

## Run

    python -m src.autograd_oracle.gate_factors                    # cc-pVDZ water in PCM(water), singlet S1
    python -m src.autograd_oracle.gate_factors --gas              # gas-phase control
    python -m src.autograd_oracle.gate_factors --target qp        # HOMO quasiparticle
    python -m src.autograd_oracle.gate_factors --target qp --qp-offset -4   # O 1s: residues above the gap
    python -m src.autograd_oracle.gate_factors --tda --spin triplet
    python -m src.autograd_oracle.gate_factors --root-mode ift    # IFT instead of unrolled Newton
    python -m src.autograd_oracle.gate_factors --fd 0,2 --h 1e-3  # + a four-point FD of one component
    python -m src.autograd_oracle.gate_factors --basis sto-3g --auxbasis cc-pvdz-ri

`TORCH_CHAIN_TRACE=1` stamps every stage of the torch forward.

## What it prints

1. forward: Omega (or eps^QP), the window's roots and Z, the Eq. (18) shift,
   the BSE diagonal, the first Casida roots -- production vs torch
2. adjoints: eps_bar, X_bar, D_bar, D_bar_bare -- max|d|, relative to the
   array's scale, worst entry above 1 % of the scale, 1 - cos
3. swap-in: the torch adjoints through production's `nuclear_gradient`
   against production's force, plus the assembly's own floor (production's
   adjoints re-fed bitwise, and with 1e-13 relative noise)
4. `--fd`: a four-point finite difference of the whole chain (SCF in PCM,
   cavity, factors, quasiparticles, Casida redone at every displacement)

## Two processes, one thread

`gate_factors.py` (pyscf, production) writes the factors, the frozen
conventions and the captured production adjoints to an `.npz`
(`frozen.py`); `torch_oracle.py` (numpy + torch only) reads it, prints the
comparison and writes its adjoints back. The split is forced by the platform:
on an x86_64 Python under Rosetta on an arm64 Mac, the
torch build

* segfaults in the backward pass with two intra-op threads (3/3) and never
  with one (3/3) -- `torch_oracle.py` pins itself to one thread;
* deadlocks the interpreter's normal exit when pyscf's ISDF integrals and
  torch have both run in one process -- both scripts end with `os._exit`.

Neither is a property of the chain. The torch side reads only the `.npz`, so
it runs under any interpreter with numpy and torch, including a native arm64
one.

## Results (2026-09-11, water bent to C1, cc-pVDZ / cc-pVDZ-RI, M = 444, naux = 84)

Every variant agrees with torch.autograd at round-off on all four adjoint
arrays: relative 1e-14 or better, worst single entry above 1 % of the scale
1e-13. Forward values, roots and Z agree to 1e-15. The swap-in force differs
from production's by 1e-9 .. 1.5e-8 Ha/Bohr on forces of 0.03 .. 0.11, which
is the assembly's own floor (1e-13 noise on the adjoints moves it by the same
amount); both sit within 5e-9 of a four-point finite difference whose
Richardson residual is 1.1e-8. Gated: singlet full BSE, singlet TDA, triplet
TDA, BSE at the mean field, HOMO / inner-valence / core quasiparticles, gas
phase and PCM(water), unrolled Newton and IFT root differentiation.

The nuclear layer below the factors is not autograd-covered; it is closed by
`--fd all` on the solvated singlet: all nine components of dOmega/dR and of the
total force d(E_0 + Omega)/dR agree with the four-point finite difference of
the whole chain to 1.07e-8 Ha/Bohr (1.1e-7 relative) at h = 1e-3 Bohr, where
the finite difference's own translational residual is 1.0e-9; the analytic
total force sums to 6.8e-15 over the atoms. 103 s for the 36 displaced chains.

## The nuclear layer too: autograd from the coordinates

No test runs this layer (about 2.5 minutes on a workstation); it is a manual gate.

`gate_nuclear.py` / `torch_nuclear_oracle.py` / `torch_nuclear.py` /
`torch_integrals.py` (`python -m src.autograd_oracle.gate_nuclear`). Omega(R) and E_0(R) as ONE torch graph from the nuclear
coordinates, differentiated by autograd, with nothing from libcint, pyscf's
PCM gradient or any hand-written adjoint:

* `torch_integrals.py`: McMurchie-Davidson in torch (Hermite expansion, Boys
  function by series/asymptotics, Hermite Coulomb recursion) -- overlap,
  kinetic, nuclear attraction, (P|Q), (mu nu|P), Gaussian surface charges,
  collocation. Pinned against pyscf to <= 3e-14 relative (`pin_integrals.py`);
  the only pyscf input is the shell table (`_libcint_ctr_coeff`, which
  carries the radial normalization that `bas_ctr_coeff` strips) and
  `gto.cart2sph`.
* `torch_nuclear.py`: pyscf's PCM cavity ported (rigid points, York-Karplus
  switching, S, D, K, R for IEF-PCM/C-PCM/COSMO/SS(V)PE, Q = sym(K^-1 R)) at
  two dielectric constants; DF-RHF Fock with the reaction field; the converged
  density as an implicit fixed point (`SCFFixedPoint`: dense dg/dD by autograd,
  one transposed solve, the cotangent pushed through g onto the integrals);
  one generalized eigensolve for (eps, C); the ISDF fit (`fit_M_stable`) and
  the metric roots in both gauges; then `torch_chain.forward`.

Every stage pins at round-off at the reference geometry (Fock 1e-14, cavity
K/R 1e-16, vtilde 2e-14, e_pcm 1e-13; the fit M at 1.3e-8, which is cond(G) ~
2e8 times the integral round-off). Water/cc-pVDZ in PCM(water), 40 s forward,
65 s backward, one thread:

    dOmega/dR   torch vs production 1.03e-8   torch vs 4-point FD 9.5e-10   production vs FD 1.07e-8
    d eps^QP/dR torch vs production 1.3e-9
    gas phase   torch vs production 7.2e-9
    dE_0/dR     torch vs pyscf DF+PCM gradient 2.5e-13   both vs FD 1.3e-10

The torch surface has translational residual 1e-15 by construction, so its
gradient sits INSIDE the production-vs-FD scatter: the 1e-8 differences are
production's iterative Lagrangian solve, as the adjoint-noise probe said.
