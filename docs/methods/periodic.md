# Periodic systems

**Periodic** — the k/q-point counterparts of the RPA, BSE and GW routes, in
`src/SingleReference/Periodic`, on two factorizations. The GDF / RI-V route
(`pbc_integrals`, `pbc_casida`, `pbc_rpa`, `pbc_self_energy`) builds complex
three-center integrals per momentum transfer and gives the RPA correlation
energy, the screened interaction W^q, the full BSE per q and the G0W0
self-energy with its GWΓ∞/PSD1 vertex variants; it is checked against pyscf's
`krgw_ac` and by supercell folding, and its memory grows as N_k². The ISDF /
THC route (`pbc_isdf`, `pbc_isdf_rpa`, `pbc_isdf_gw`, after Yeh and Morales,
JCTC 2024, 20, 3184) is separable in the k-index, O(N_k), and is the
production one: `rpa_ecorr_thc` and `qp_energy_thc` are its entry points.
Metals get an occupation-weighted response with Fermi smearing
(`pbc_occupations`) and a fermionic-IR Matsubara self-energy. Slabs get the
Coulomb-damped kernel of Förster et al., JCTC 2025, 21, 9347
(`pbc_rpa_damping`), the 2D small-q head and mini-Brillouin-zone averaging
(`pbc_smallq`, `pbc_wav`), and planar-interface dielectric screening — an
electrolyte above, a metal electrode below (`pbc_solvent_screening`, see
`examples/18_periodic_slab_gw.py`). Restricted (closed-shell) references
only.
