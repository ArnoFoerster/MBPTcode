# Periodic systems

**Periodic** — the k/q-point counterparts of the RPA, BSE and GW routes, in
`src/SingleReference/Periodic`. Built on a converged KRHF/KRKS mean field
with density fitting; restricted (closed-shell) references only.

Two factorizations:

- **GDF / RI-V** (`pbc_integrals`, `pbc_casida`, `pbc_rpa`, `pbc_self_energy`)
  — builds explicit complex three-center integrals per momentum transfer.
  The validated reference route: checked against pyscf's `krgw_ac` and by
  supercell folding. Memory grows as N_k², so it caps out at modest meshes.
- **ISDF / THC** (`pbc_isdf`, `pbc_isdf_rpa`, `pbc_isdf_gw`, after Yeh and
  Morales, JCTC 2024, 20, 3184) — separable in the k-index, O(N_k). The
  production route: `rpa_ecorr_thc` and `qp_energy_thc` are its entry points.

**Integrals** (`pbc_integrals`)
- `PBCDFIntegrals.from_scf` reads the k-point GDF 3-center integrals the same
  way `krgw_ac` does, then repacks them by momentum transfer `q`:
  `L[q][ki]` holds the factor for the pair `(ki, ki+q)`.
- Everything downstream reads only this `L` — the single chokepoint that lets
  a damped or screened kernel (see Slabs) swap in with no other change.

**RPA / Casida** (`pbc_casida`, `pbc_rpa`)
- `build_chi0_aux` gives the transfer-q polarizability Pi0^q, matched to
  `krgw_ac.get_rho_response` to machine precision — the momentum-bookkeeping
  oracle used throughout.
- `build_rpa_matrices` gives the complex-Hermitian Casida blocks A(q), B(q),
  following the time-reversal-adapted formulation of Sander, Maggio and
  Kresse (PRB 92, 045209).
- `pbc_rpa.ri_rpa_ecorr` gives the RPA correlation energy from an explicit
  `coulG_fn` kernel — the seam that Coulomb damping (Slabs) hooks into.

**Self-energy** (`pbc_self_energy`)
- `sigma_c_diag` / `sigma_vertex_diag` build the diagonal G0W0 self-energy
  (plain, or GWΓ∞/PSD1 vertex-corrected) from the RPA or BSE excitons at
  each q.
- `qp_energy_g0w0` linearizes it into a quasiparticle energy.
- A subtle normalization trap: the BZ q-average and the response's own
  1/nkpts are two separate factors, easy to conflate. Only
  `tests/test_pbc_sigma_folding.py`'s supercell-folding check catches it — a
  single-k test cannot.

**ISDF / THC** (`pbc_isdf`, `pbc_isdf_rpa`, `pbc_isdf_gw`)
- `build_isdf_kpts` picks interpolation points by pivoted Cholesky on the
  pair-density Gram matrix and fits interpolating vectors, giving a THC
  factorization separable in k.
- Needs a pseudized (`gth-*`) basis: an all-electron uniform grid would need
  to be impractically fine (Zhu, Yeh, Morales et al., JCTC 2026, 22, 2904).
- `pbc_isdf_rpa` / `pbc_isdf_gw` build the RPA polarizability and GW
  self-energy from it in imaginary time, with the k-sum done as an FFT.
- The self-energy needs a *wider* tau range than the polarizability alone,
  since Σ = -G·W̃ is a product of decay rates. `qp_energy_thc` resolves and
  checks this automatically — call it rather than the pieces directly.
- `build_isdf_kpts_symm` adds an IBZ reduction over the space group for large
  meshes (Yeh & Morales, Sec. 4); symmorphic groups only.

**Metals** (`pbc_occupations`, `pbc_isdf_gw`)
- `pbc_occupations` replaces the fixed occ/virt partition with an
  occupation-weighted response built from ordered pairs. Stays Hermitian for
  any occupation that is monotone in energy — verified, not assumed.
- `sigma_c_matsubara_thc` builds the self-energy on a fermionic Matsubara
  grid, via the same intermediate-representation sampling used in
  [Finite temperature](finite-temperature.md) (Shinaoka et al., PRB 96,
  035147; Li, Wallerberger, Chikano, Yeh, Gull, Shinaoka, PRB 101, 035144).
- QP energies are solved for their root, never linearized — linearization is
  unstable near a Fermi surface.

**Slabs** (`pbc_rpa_damping`, `pbc_smallq`, `pbc_wav`, `pbc_solvent_screening`)
- `pbc_rpa_damping` supplies an AUTO Fermi-Dirac damped Coulomb kernel
  (Förster et al., JCTC 2025, 21, 9347) that removes the Γ-point divergence
  pyscf's bare kernel has for `cell.dimension < 3`. The damping radius tracks
  the k-grid's Nyquist radius, so it vanishes as the mesh is refined;
  `check_low_dim_support` verifies the damping range still fits the vacuum.
- `pbc_smallq` gives the 2D small-q physics a metallic slab needs, instead of
  a 3D Drude head — Stern's constant 2D polarizability (PRL 18, 546 (1967)).
- `pbc_wav` averages the interaction over the mini-Brillouin-zone around each
  grid point rather than special-casing q=0 (Guandalini and Sesti), which
  also smooths out the damped kernel's ringing.
- `pbc_solvent_screening` embeds a dielectric environment around the slab
  (electrolyte, a metal electrode, or both) as a rank-two update to the RI-V
  metric — the periodic form of Duchemin, Jacquemin and Blase, J. Chem.
  Phys. 144, 164106 (2016). RPA, W, BSE and Σ_c all pick up the screening
  through the same `L`. See `examples/18_periodic_slab_gw.py`.

**Band structures** (`pbc_kpath`)
- Interpolates the quasiparticle correction QP(k) - eps(k), which is smooth
  even where the bands themselves are not, then adds it to a mean-field band
  structure evaluated directly on the path.
- Uses plain Fourier interpolation over the k-mesh, not Shankland-Koelling-
  Wood: SKW's star functions need the full space group, and the mesh
  generally has fewer symmetries than the lattice.

**Key references**
- Sander, Maggio and Kresse, Phys. Rev. B 92, 045209 — complex-Hermitian
  Casida equations for extended systems.
- Yeh and Morales, J. Chem. Theory Comput. 2024, 20, 3184 — k-point ISDF/THC
  for RPA and GW, including the symmetry-adapted (IBZ) reduction (Sec. 4).
- Zhu, Yeh, Morales et al., J. Chem. Theory Comput. 2026, 22, 2904 — grid
  admissibility for periodic ISDF on pseudized vs. all-electron bases.
- Förster et al., J. Chem. Theory Comput. 2025, 21, 9347 — AUTO Fermi-Dirac
  Coulomb damping for periodic RI-RPA/RI-V.
- Stern, Phys. Rev. Lett. 18, 546 (1967) — the 2D electron gas static
  polarizability behind the slab small-q head.
- Guandalini and Sesti — mini-Brillouin-zone averaging of the screened
  interaction near q = 0.
- Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016) — the
  v → v + ṽ dielectric-embedding scheme, here specialized to a slab cavity.
- Shinaoka, Otsuki, Ohzeki and Yoshimi, Phys. Rev. B 96, 035147 (2017); Li,
  Wallerberger, Chikano, Yeh, Gull and Shinaoka, Phys. Rev. B 101, 035144
  (2020) — the intermediate-representation basis and sparse Matsubara
  sampling used for finite-temperature (metallic) GW.
