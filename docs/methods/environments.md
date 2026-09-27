# Environments: solvent, dispersion, polarizable embedding

**Solvent** — polarizable-continuum screening in the style of Duchemin,
Jacquemin and Blase, [J. Chem. Phys. 144, 164106 (2016)](https://doi.org/10.1063/1.4946778).
Non-equilibrium solvation is two dielectric constants and needs both: the
ground state relaxes inside PCM(ε_static), since the solvent nuclei have had
time to reorient around it, and only the *response* is optical. One
`SolventScreening` carries both — `env.mean_field(mol, factory)` applies the
first, `attach_environment(mf, env)` the second, which substitutes v → v + ṽ
at the integral chokepoints.

Inside GW the reaction field is then Duchemin, Guido, Jacquemin and Blase,
Chem. Sci. 9, 4430 (2018), Eq. (18): the
self-polarization of the orbital carrying the added charge in the *screened*
reaction field, on every route — the contour-deformation, Laplace and
sum-over-poles continuations of the space-time route included — with Σ itself
screened by the bare interaction; screening Σ dynamically as well counts the
same polarization twice. The
static COHSEX operator remains the fallback for the routes that never form W
(ADC); the two differ by 0.39 eV of quasiparticle gap on water in water. An
unrestricted reference takes Eq. (18) too, per spin orbital against the one W
of both spins.

A quasiparticle level in a continuum is vertical: the added charge polarizes
the optical response only. `calc_qp_energy(..., equilibrium=True)` adds the
solvent's relaxation around the charged state on every route,
Eq18_p(ε_s) − Eq18_p(ε∞) on one cavity, so E^(N∓1) = E_0 ∓ ε_p is the ion in
equilibrium with its solvent; `ChargedExcitation(..., equilibrium=True)` does
the same on the charged surfaces, with its analytic gradient, and
`vibronic.reorganization_four_point` turns the vertical and equilibrium
surfaces into the outer-sphere λ_s beside the four-point inner one.

An explicit polarizable first shell inside the continuum is
`ContinuumWithSites(SolventScreening(mol, ..., cavity_atoms=shell),
PolarizableSites(...))`: the cavity encloses the solute and the shell for the
ground state's PCM and the optical response alike, and the surface charges
and the induced dipoles polarize each other in one linear problem, folded into
one kernel and so one Eq. (18) shift. Energies only; its force is refused.
See `examples/13_solvated_gw_bse.py`.

Both halves of the reaction field's nuclear derivative are analytic —
`aux_kernel_adjoint` for the dressed metric, `static_self_energy_adjoint` for
the static term — on the bilinear cavity derivative of `pcm_derivatives.py`;
a solvated excitation gradient reproduces finite differences at the same
8e-8-relative floor as the gas-phase chain.

**Empirical dispersion** — a semi-classical -C6/R^6 correction (D3/D4, via
pyscf's own `xc='pbe0-d4'`-style functional names) for a hybrid, which has no
long-range correlation of its own. It is a function of the nuclear
coordinates alone, so it moves the surface without moving the spectrum: every
orbital, and therefore every excitation energy, is bitwise unchanged by it.
It must never sit under a direct-RPA ground state, whose correlation energy
already contains dispersion, nor be added twice to a functional (r2SCAN-3c,
wB97X-V, ...) that already carries a correction of its own — both are refused
rather than silently double-counted.

**Polarizable embedding** — classical QM/MMPol, in the style of Li, D'Avino,
Duchemin, Beljonne and Blase,
[Phys. Rev. B 97, 035108 (2018)](https://doi.org/10.1103/PhysRevB.97.035108).
A site's induced dipole couples to every other site's,
`mu = (alpha^-1 - T)^-1 E`, giving the classical response matrix B that
dresses the interaction exactly as the PCM continuum's v -> v + ṽ does, field
(dipole) response in place of charge (surface) response. `PolarizableSites`
builds B from a hand-rolled, isotropic, uniformly Thole-damped coupling;
`cppe_interface.py` sources it instead from a real force-field potential file
(PyFraME, DALTON's PE library) through [cppe](https://github.com/maxscheurer/cppe),
with per-atom anisotropic tensors and 1-2/1-3 exclusions. `composite_environment.py`
carries permanent charges and polarizable sites as ONE environment with
neither channel leaking into the other's: permanent multipoles reach the
mean field and contribute nothing to the reaction-field kernel, sites screen
and leave the ground state alone — the split the paper's Sec. II A and II D
draw.
