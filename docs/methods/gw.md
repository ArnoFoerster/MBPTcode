# GW and linear response

**Screening** — static RPA screened Coulomb interaction W, and the screened
C^(1) block of the screened multichannel Dyson equation, following
Romaniello and Berger, [arXiv:2603.27329](https://arxiv.org/abs/2603.27329).

**GW / linear response** — G0W0 and eigenvalue-self-consistent GW on the
real and imaginary axes, Casida/RPA/BSE, RPA correlation energies.

Three routes reach the same quasiparticle energy and differ only in cost:

| function | how W and Sigma are built | cost |
|---|---|---|
| `calc_qp_energy` | Casida problem solved explicitly | O(N⁶) |
| `solve_qp_energy_imaginary_axis` | quadrature on an imaginary-frequency grid | O(N⁴) |
| `solve_qp_energy_space_time` | pointwise product in imaginary time, on a separable (ISDF) factorization of the ERIs | O(N³) |

All three take an unrestricted (UHF/UKS) reference, with `spin_channel=`
naming the channel: one W from both spins' polarizabilities, the self-energy,
static exchange and continuum shift of the channel asked for. The space-time
route's blocked (`freq_block`, `scratch_dir`) and sliced paths stay
restricted; ROHF/ROKS and fractional occupations are refused.

A correlated density matrix is passed to any of them as
`dm_correction=`; the old `dm_ccsd=` alias is not accepted and raises
`TypeError`.

The imaginary-axis routes reach the real axis by one of four continuations,
`calc_qp_energy(..., continuation=...)`: `'pade'` (Thiele-Pade of
Sigma_c(i.omega), the default, no analytic gradient); `'cd'` (contour
deformation — the omega' contour rotated onto the imaginary axis, the poles of
G it sweeps over collected as residues, no continuation at all); `'sop'` (W
modelled by M poles fit on the imaginary axis, so Sigma_c is closed-form and
never evaluated off it, valence states only); and `'spectral'` (the Lehmann
sum at omega + i.eta, `mode='casida'` only). `MODE_CONTINUATIONS` is the
validity table pairing modes with the continuations they accept; a keyword
another continuation reads raises `TypeError` naming both.

A dense route (`GW.quasi_boson`, `LinearResponse.quasi_boson_bse`) builds the
same dRPA/BSE amplitudes as an explicit auxiliary-boson diagonalization rather
than a Davidson iteration — bitwise the physics of `casida(eta=0)`, at O(N^6),
and what the dense analytic gradients ([Gradients](gradients.md))
differentiate.

**evGW** — `calc_qp_energy(..., self_consistency='evGW')` drives any of those
three routes to a fixed point: the quasiparticle energies are reinjected into
G and P₀ until the spectrum stops moving, with every eigenvalue updated, DIIS
acceleration and convergence decided on the HOMO and LUMO. The equation stays
anchored on the mean field while the screening follows the iterate — anchored
on the iterate instead, each cycle adds its own correction a second time and
the gap runs away without ever converging. `evgw_eigenvalues` returns the whole
converged spectrum and a record of how it got there. See `examples/12_evgw.py`.
`self_consistency='evGW0'` reinjects them into G alone: the mean field's P₀ and
W stay and only the poles of Σ_c move, on the Casida route.

**qsGW** — `self_consistency='qsGW'` (or `'qsGW0'`, W kept at the mean field)
runs the quasiparticle-self-consistent loop on the Casida route: a static
Hermitian self-energy replaces v_xc, h + J + K + Σ̃ is diagonalized, and the
orbitals and eigenvalues are reinjected until the density and the frontier
eigenvalues stop moving. Σ̃ is the SRG-regularized form of Marie and Loos
([J. Chem. Theory Comput. 19, 3943 (2023)](https://doi.org/10.1021/acs.jctc.3c00281))
at flow s = 100 Ha⁻², since Kotani's mode A on the pole sum has no fixed
point. `qsgw_eigenvalues` returns the spectrum, the orbitals and, for a BSE on
top, the DF factors and static W of the result. Restricted or unrestricted
(per-spin Fock and Σ̃ around one W), in the gas phase or a continuum: there
the SCF's PCM potential stays in the Fock matrix every cycle and the solute's
response to its own added charge enters as a static operator from the
screened reaction field ΔW, whose diagonal is Duchemin et al.'s Eq. (18).
