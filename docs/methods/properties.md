# Potential-energy surfaces and properties

**Potential-energy surfaces** — `src/properties/` computes from a surface
rather than building one. The surfaces come from the gradient layer
([Gradients](gradients.md)): `potential_energy_surface(mol, scf_factory, *,
ground_state, excitation=None, environment=None, ...)`
(`src.gradients.gwbse.surface_factory`) dispatches on the declared physics
(`src.Base.declaration`: `GroundState`, `Excitation`, `ChargedExcitation`,
`QPStates`) to the gradient chain that realizes it and records the realization.
Its numerical choices come in as one object, `numerics=Numerics(...)`
(`src.Base.numerics.numerics`), handed whole to that chain; a choice the chain
does not read is refused, not dropped. `compare_surfaces` refuses to difference
two surfaces that do not share a ground-state functional and environment.
`calc_vertical_excitation`, `calc_adiabatic_excitation` and
`calc_adiabatic_gap` build both surfaces of a difference from one
`SurfaceSpec`, so the functional under an excited state and under its ground
state cannot silently be two different functionals. `optimize`/`relax` walk
Cartesian RFO/BFGS with the rigid-body directions projected out, or
[geomeTRIC](https://github.com/leeping/geomeTRIC) when it is installed. A
surface keeps a few choices fixed while it is walked (which orbitals get a full
quasiparticle solve, for one); after the walk converges they can be chosen
again at the minimum and the walk repeated until the energy stops moving
(`refreeze=1`): passes continue until a rebuild moves the energy by less than 3
meV (`REFREEZE_TOL_MEV`), at most four. A walk that still drifts, or does not
converge, keeps its last geometry and is warned about and flagged
(`refreeze_not_converged`; [Warnings, flags and refusals](../warnings.md)). A
property takes any surface that provides the ingredients it reads
([Gradients](gradients.md)). An excited-state walk starts from a tiny fixed
random displacement of the input geometry, so a symmetric start can still reach
a minimum of lower symmetry. `vibronic` gives normal modes and Huang-Rhys
factors by two independent routes; `conformers` the Boltzmann-weighted average
over torsional minima a soft emitter has (chiral minima counted twice), with
the ring puckers and nitrogen inversions of `ring_conformers` as extra starts,
free-energy populations from per-conformer Hessians in `conformer_weights`,
relaxed torsion scans, barriers and the following of one state through a large
twist in `torsion_scan` (an increment whose point is lost or does not converge is
halved, then the scan stops with the reason), and `excited_conformers` for a twisted-ICT emitter
whose emitting conformer is not its absorbing one; `spin_orbit` and `nonadiabatic` the
two couplings a Marcus-Levich-Jortner or golden-rule rate needs; `characters`
labels a BSE root by its charge-transfer weight; `hessian` the nuclear Hessian
by central differences of the analytic gradient, for a route pyscf's own
analytic Hessian cannot serve (on an interpolated mean field it asks for a fine
enough interpolation grid and refuses a coarse one, whose surface is too rough
for a finite difference); `mode_hessian` the force constants along a few chosen
normal modes only, which gives the frequency of one local vibration on an
excited state for a handful of gradients. See
`examples/15_excited_state_geometry_optimization.py` for `calc_adiabatic_gap`
relaxing S1 and T1 each to its own minimum, and
`examples/17_spin_orbit_coupling.py` for `spin_orbit` read at the two minima
`calc_adiabatic_gap` already relaxes (El-Sayed's rule, computed rather than
assumed) plus the Herzberg-Teller dV/dq scan over `vibronic`'s ground-state
modes that `spin_vibronic_rate` needs for an El-Sayed-forbidden pair, where the
Condon term alone is not the whole story.

**The spin-orbit operator on large molecules** — the two-electron part of the
spin-orbit mean-field operator costs a pass over all four-index integrals,
which grows with the fourth power of the basis and dominates the properties of
an emitter-size molecule. The default of a photophysics run,
`soc_two_electron='ri-somf'`, builds the same operator with the density side
fitted in an auxiliary basis: three-index integrals and matrix products in
place of the four-index pass, divided over the ranks of a parallel run. The
fitting basis is the JK-fitting set of the orbital basis (aug-cc-pvdz-jkfit
for aug-cc-pVDZ) unless `soc_auxbasis` names another; an orbital basis without
such a set is refused, and the run must name one or declare the exact pass,
`soc_two_electron='somf'`. The largest change of any coupling against the
exact pass, with the JK-fitting set of the orbital basis:

| molecule, basis | largest coupling | largest change |
|---|---|---|
| formaldehyde, aug-cc-pVDZ | 57 cm^-1 | 0.0006 cm^-1 |
| thiophene, aug-cc-pVDZ | 32 cm^-1 | 0.0011 cm^-1 |
| 2CzPN, cc-pVDZ | 3.0 cm^-1 | 0.0006 cm^-1 |

The change falls further with a larger fitting basis. On 2CzPN in cc-pVDZ
(594 basis functions) the fitted term takes one twentieth of the exact one's
time. A run that wants the exact pass declares it in its numerics,
`Numerics(properties={'soc_two_electron': 'somf'})`. A stored run written
before the setting existed holds exact-pass couplings and resumes as it is.

**Spin-vibronic coupling** — intersystem crossing between a singlet and a
triplet of the same orbital character (both n -> pi*, or both pi -> pi*) is
nearly forbidden by El-Sayed's rule: their spin-orbit coupling is close to
zero at a fixed geometry. A vibration can still switch it on, by mixing a
nearby second triplet (or singlet) of a different character into the pair.
`spin_vibronic_coupling` gives how fast the coupling between S1 and T1
changes along each nuclear motion, through those higher states, from one
excited-state calculation and without displaced geometries. This matters when
the direct coupling is small and T2 lies close to T1: forward intersystem
crossing of El-Sayed-forbidden pairs, and reverse intersystem crossing in
TADF emitters, where T2 a few k_B T above T1 often carries the process. The
derivative, projected on normal modes, is the Herzberg-Teller input of
`spin_vibronic_rate`, and `photoluminescence` takes the T2 rates
and the T1-T2 gap to weight a thermally populated T2.

```python
import numpy as np
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold
from src.properties.spin_orbit import declared_operator, soc_operator_mo
from src.properties.spin_vibronic import spin_vibronic_coupling


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    return mf


# twisted formaldehyde with a stretched C=O: T2 lies close above T1
mol = gto.M(atom='C 0 0 0; O 0.05 -0.03 1.45; H 0.12 0.943 -0.588; '
                 'H -0.20 -0.943 -0.588', basis='cc-pvdz')
numerics = Numerics(isdf={'level': 'G1'},
                    bse={'solver': 'davidson', 'adjoint': 'grid', 'nroots': 3})
chain = ExcitedStateChain(mol, rhf, mf=rhf(mol), numerics=numerics)
S1, S2, T1, T2 = ('singlet', 0), ('singlet', 1), ('triplet', 0), ('triplet', 1)
ev = StateManifold(chain, states=(S1, S2, T1, T2)).evaluate(
    couplings=((S2, S1), (T2, T1)))
h_mo = soc_operator_mo(ev.mf, ev.mol, **declared_operator(numerics))
out = spin_vibronic_coupling(ev, h_mo, S1, T1,
                             singlet_paths=(S2,), triplet_paths=(T2,))
print('|<S1|H_SO|T1>| =', out['v0'], 'Hartree')
print('|dV/dR| =', np.linalg.norm(out['dv_cart']), 'Hartree/Bohr')
```

`out['dv_cart']` holds the derivative for every atom and direction; pass
`modes`, `masses` and `omega` from `vibronic.normal_modes` at a minimum to get
it per normal mode as well.

**Vibronic band shapes** — `band_shape(s_k, omega_k, e00, temperature,
energies, *, gaussian_fwhm, lorentzian_fwhm)` gives the normalized absorption
(E x FC) and emission (E^3 x FC) spectra of the displaced-oscillator model from
`vibronic`'s Huang-Rhys factors, the 0-0 energy and the temperature, with their
peaks, FWHMs, the Stokes shift, and the emission per unit wavelength with its
FWHM in nm. Both broadenings are required arguments: a computed width is only
comparable with a measured one when the broadening added to it is stated. The
Franck-Condon density is `franck_condon.fc_weighted_dos`, the generating
function the golden-rule rates use, which takes a homogeneous Lorentzian
(integrated on the real time axis, with an Euler-Maclaurin correction at the
kink of e^{-gamma |t|}) and returns an exact zero where the saddle-point
exponent has underflowed, the far side of a cold band. `band_grid` builds an
energy grid that covers and resolves both bands, and `gaussian_limit_fwhm` the
width of the second cumulant alone. See `examples/20_vibronic_band_shape.py`
for formaldehyde S1 end to end, both Huang-Rhys routes.

`vibronic_band(model, e00, temperature, energies, *, gaussian_fwhm,
lorentzian_fwhm)` is the same band for any model on the `VibronicModel`
protocol (a density per direction and its first two cumulants);
`band_shape` is it on `DisplacedOscillators`. `properties.duschinsky` gives
each state its own modes and frequencies: `duschinsky_model(ground, excited,
coordinates)` takes two Hessians at their own minima (anything with `.mol` and `.hessian`;
`read_orca_hessian` reads an ORCA `.hess` file as data) and returns J, K and
both frequency sets, and `DuschinskyOscillators(model)` feeds
`vibronic_band`. The density is the Fourier transform of the thermal
correlation function of the two harmonic surfaces, with the zero-point shift
of the 0-0 line included. Formaldehyde's S0 -> T1 band at HF/6-31G, from the
repository root (about a minute on a single machine):

```python
import numpy as np
from pyscf import gto, scf

from src.gradients.mean_field_surface import MeanFieldSurface
from src.properties.band_shape import vibronic_band
from src.properties.duschinsky import (CartesianHessian, DuschinskyOscillators,
                                       duschinsky_model)
from src.properties.vibronic import relax_state

# formaldehyde S0 and T1, HF/STO-3G: each relaxed and its Hessian taken at its
# own minimum (T1 as the high-spin UHF state, slightly pyramidal at the start)
CH2O = 'C 0 0 -0.53; O 0 0 0.67; H 0 0.94 -1.12; H 0 -0.94 -1.12'
CH2O_T1 = 'C 0 0.02 -0.53; O 0 0 0.67; H 0.25 0.94 -1.12; H 0.25 -0.94 -1.12'


def relaxed(atom, spin):
    mol = gto.M(atom=atom, basis='6-31g', spin=spin, verbose=0)
    factory = lambda m: (scf.RHF(m) if spin == 0 else scf.UHF(m)).run(conv_tol=1e-12)
    rec = relax_state(MeanFieldSurface(mol, factory), mol, engine='cartesian',
                      verbose=False)
    return rec['mol'], rec['mf'], rec['e_total']


mol_s0, mf_s0, e_s0 = relaxed(CH2O, 0)
mol_t1, mf_t1, e_t1 = relaxed(CH2O_T1, 2)
model = duschinsky_model(CartesianHessian(mol_s0, mf_s0.Hessian().kernel(), 'S0'),
                         CartesianHessian(mol_t1, mf_t1.Hessian().kernel(), 'T1'),
                         coordinates='curvilinear')
fc = DuschinskyOscillators(model)
e_adiabatic = e_t1 - e_s0                                 # Hartree
# HF puts this 0-0 line low enough that the default grid would reach below
# zero photon energy, so the photon energies are given
e = np.linspace(0.005, 0.2, 4000)
band = vibronic_band(fc, e_adiabatic, 300.0, e, gaussian_fwhm=0.002,
                     lorentzian_fwhm=0.0)
print(band['fwhm_emission'], band['peak_emission'])
```

**Herzberg-Teller intensity** — `properties.herzberg_teller.TransitionDipole`
holds mu0 and dmu/dq at one state's minimum on that state's normal
coordinates; `DuschinskyOscillators(model, dipole)` then carries the Condon,
Herzberg-Teller and cross terms in the same correlation function, and its
density has <|mu|^2> as its area. dmu/dq comes from an analytic Cartesian
derivative (`dipole_from_cartesian`) or from central differences of any
routine that returns the transition dipole at a geometry
(`dipole_derivative`, which warns when the two displaced dipoles look like a
phase flip). For an excited state of a Casida-type solve (BSE@GW) the routine
is `PhaseTrackedDipole(solve_at, mol, root)`: each displaced solve is aligned
to the one at `mol` by the transported overlap of the eigenvectors, a group of
near-degenerate roots as a subspace, and a displaced solve that no longer holds
the roots is refused. On formaldehyde BSE@G0W0@PBE0/cc-pVDZ the forbidden S1
borrows along the out-of-plane wag and the b2 modes, and its dmu/dq does not
move under random sign flips of the displaced eigenvectors. A displaced-oscillator model with Herzberg-Teller terms is
`DuschinskyModel.from_displacement(omega, dq)` with the signed displacements.
`photophysics.rates.vibronic_radiative_rate` integrates E^3 over the emission
band, so a weakly allowed emitter's rate includes what it borrows.

**Rates** — `src/photophysics/rates.py` holds the golden-rule,
Marcus-Levich-Jortner and spin-vibronic rates, `photoluminescence`, and
`rate_table`, which assembles every rate between S0, S1, T1 (and T2) from the
minima a run relaxed. The Huang-Rhys factors of the rate tables are taken in
valence internal coordinates (`properties.curvilinear`), so a rotated methyl
or amino group does not show up as a fake displacement along the C-H
stretches.

**ISC and RISC of any method** — `src/photophysics/rate_chain.py` computes
intersystem crossing S1 -> T1 and its reverse from one method's S1 and T1
minima (`minimum.StateMinimum`, filled alike by GW-BSE, ADC(2) or any other
method) and two Hessians at their own minima, which may come from another
method (an ORCA TD-DFT `.hess` read with `read_orca_hessian`). The model is
declared field by field in `RateModel`: Duschinsky rotation, frequency change
and zero-point difference; the displacement from the method's own minima
(curvilinear or Cartesian) or from its gradient at the other minimum; the
solvent reorganization (non-equilibrium minus equilibrium energy of the
accepting state) as the classical broadening; the Condon coupling plus its
nuclear derivative; the Boltzmann-populated singlets and triplets with every
reachable pair, so RISC through T2 is included. A second-order path through a
nearly degenerate state (T2 a few meV above T1) is outside first-order
mixing: its coupling-to-gap ratio r, read off the path's own derivative
(`spin_vibronic.path_ratio`), switches that direction to the diabatic
two-state model above 0.1, where the path state's exact thermal weight
(`spin_vibronic.mixed_weight`, at most 1/2) replaces r^2; the record says
which model each rate used, and `mixing` is a switch of its own. The
Duschinsky model is checked before it is used (`duschinsky.duschinsky_guard`,
recorded as `duschinsky_guard`): a soft mode of one state that puts more than
1% of its weight on the other state's stretches, a sum-rule reorganization
more than a factor 2 from the accepting state's energy difference between the
minima, or a band whose mean vibrational energy is more than twice that
reorganization, refuses the Duschinsky rungs. Between two minima a large
torsion apart (4-(dimethylamino)benzonitrile S1 and T1) the Cartesian mode
sets mix the torsions into the stretches and the band's first moment comes out
at tens of eV. The production model therefore measures J through the valence
internals (`RateModel.duschinsky_coordinates = 'curvilinear'`,
`curvilinear.curvilinear_duschinsky`, after Reimers, J. Chem. Phys. 115, 9103
(2001)): a torsion of one state maps onto the torsions of the other, and J is
used as it is, not orthogonalized (each state keeps its own kinetic metric;
the correlation function carries the Jacobian). On the same molecule the soft
modes then put less than 0.5% of their weight on the stretches and the ISC
band sits 0.76 eV above its 0-0 line at 300 K against a reorganization of
0.54 eV; with every mode quantum, RISC is refused at 300 K, because a T1 mode at
10 cm^-1 becomes an S1 mode at 80 cm^-1 and its harmonic thermal energy alone
is about 1 eV. Such a mode is neither quantum nor harmonic at room
temperature, so the production model takes the T1 modes below 100 cm^-1
(`DUSCHINSKY_CLASSICAL_BELOW_CM`, about kT/2 at 300 K) and their image in S1
out of the correlation function as one classical, frequency-unchanged block
(`duschinsky.split_soft_modes`, `RateModel.classical_below_cm`): its
reorganization joins the classical broadening, as the low-frequency modes of
Ulstrup and Jortner (J. Chem. Phys. 63, 4358 (1975)). DMABN RISC then sits
0.53 eV above its 0-0 line against a reorganization of 0.45 eV and is
accepted; a cutoff of 200 cm^-1 moves its rate by 5%, one of 50 cm^-1 by a
factor 3. With no mode below the cutoff nothing changes, bit for bit; every
mode quantum is a rung of its own (`quantum_soft_modes`). The Cartesian J is a rung of its own (`rectilinear_duschinsky`), and so is the
displaced-oscillator model. `ablation_ladder` removes
one ingredient at a time down to a single Marcus-Levich-Jortner mode with an
assumed reorganization energy, and every rung records its model. DMABN's S1
and T1 from the test data, from the repository root (a few minutes):

```python
from dataclasses import replace

import numpy as np
from pyscf import gto

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import HARTREE_TO_CM, HARTREE_TO_EV
from src.photophysics.minimum import StateMinimum
from src.photophysics.rate_chain import (RateInputs, RateModel,
                                         ablation_ladder, rate_chain)
from src.photophysics.rates import ManifoldGaps
from src.properties.duschinsky import (CartesianHessian, displacement_from_minima,
                                       mode_sets)

# DMABN S1 and T1 (TDA-wB97X-D3), each at its own minimum, from the test data
d = np.load('tests/data/dmabn_s1_t1_hessians.npz')
mol = gto.M(atom=[(str(e), tuple(r)) for e, r in zip(d['symbols'], d['coords_s1'])],
            unit='Bohr', basis='sto-3g', verbose=0)
hessians = {'S1': CartesianHessian(mol, d['hessian_s1'], 'S1'),
            'T1': CartesianHessian(molecule_at(mol, d['coords_t1']),
                                   d['hessian_t1'], 'T1')}
e_s1, e_t1 = float(d['energy_s1']), float(d['energy_t1'])
socme = np.array([[0.0], [1.0]]) / HARTREE_TO_CM          # <S0|, <S1| x |T1>, 1 cm^-1


def minimum(state, coords, energy):
    """A state's minimum with both states' energies there."""
    return StateMinimum(
        method='TDA-wB97X-D3', where=state, coords_bohr=coords, e0=e_t1 - 3.0 / HARTREE_TO_EV,
        energy=energy, omega={k: v - e_t1 + 3.0 / HARTREE_TO_EV for k, v in energy.items()},
        emitter='S1', reservoir='T1', socme=socme, soc_dv_cart=None, soc_dv_paths=None,
        dipole=None, f_osc=None, d_ground=None, solvation='gas')


# each state's energy at the other minimum: a real run computes it; here the
# curvilinear harmonic estimate on that state's own modes stands in for it
lam = {}
for state, other in (('S1', 'T1'), ('T1', 'S1')):
    sg, se, _, _ = mode_sets(hessians[other], hessians[state], 'rectilinear')
    k = displacement_from_minima(sg, se, hessians[other].mol, hessians[state].mol,
                                 'curvilinear')
    lam[state] = 0.5 * float(np.sum((se.omega * k) ** 2))
inputs = RateInputs(s1=minimum('S1', d['coords_s1'], {'S1': e_s1, 'T1': e_t1 + lam['T1']}),
                    t1=minimum('T1', d['coords_t1'], {'S1': e_s1 + lam['S1'], 'T1': e_t1}),
                    hessians=hessians, gradients={}, solvent={})
full = replace(RateModel.full('curvilinear', 0.0), coupling='condon', triplets='T1')
gaps = ManifoldGaps('stacked', None)     # the declared gap rule, no default
table = rate_chain(inputs, full, 300.0, gaps)
print(table['ISC']['k_eff'])
for rung in ablation_ladder(inputs, full, 300.0, gaps, shared=inputs):
    print(rung['step'], rung.get('refused') or rung['table']['ISC']['k_eff'])
```

**Photophysics from the command line** — `src/photophysics` runs the TADF
rates of an emitter in stages: the S0 minimum, its modes, the vertical
spectrum, the S1 and T1 minima, then the rates. Each stage writes a record
that a relaunch reuses. Every procedural choice has to be given, so the
command line refuses a run until each one is set. The smallest run is
formaldehyde in the gas phase on a single machine, from the repository root:

```bash
python -m src.photophysics --system formaldehyde \
    --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \
    --basis cc-pvdz --environment gas --grid-accuracy G1 --scf-grid GH \
    --temperature 300 --lambda-env 0.05 \
    --nudge 0.02 --no-far-start --refreeze 1 --saddle-check 0 --df-check 0 \
    --hessian analytic-df --relaxed-solvation equilibrium --span full \
    --manifold-gaps stacked --out tadf
```

The records land in `tadf/formaldehyde/gas/`. `--rate-model uisc` runs the
UISC protocol of Samanta et al. instead, and `--distribute` under `srun`
spreads the stages over MPI ranks.

In a solvent the run names how its states are solvated, one way for every
stage: `--solvent-scheme scheme-a` (the solvated BSE of Duchemin et al.),
`scheme-b-eq` or `scheme-b-noneq`. Under both scheme-B names every state,
the ground state included, is made self-consistent with its own relaxed
density in the continuum (the same states as `solvated` below), the walks
relax the solvated free energy of each state, absorption and emission are
non-equilibrium energies, and the couplings and oscillator strengths at each
geometry are taken in one reaction field, held fixed. `scheme-b-eq` gives each
excited minimum its energy with the solvent relaxed around it (adiabatic
gaps in equilibrium); `scheme-b-noneq` with the solvent left as the ground
state had it. The S0 Hessian can come from another program (`--hessian
external --hessian-file s0.hess`, an ORCA .hess file computed at the geometry
in `s0.xyz`):

```bash
python -m src.photophysics --system formaldehyde \
    --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \
    --basis cc-pvdz --environment toluene --solvent-scheme scheme-b-eq \
    --temperature 300 --lambda-env 0.05 --nudge 0.02 --no-far-start \
    --refreeze 1 --saddle-check 0 --df-check 0 --hessian analytic-df \
    --relaxed-solvation equilibrium --span full --manifold-gaps stacked \
    --tda --qp-states frontier --scf df --grid-accuracy G1 --scf-grad-tol 1e-9 \
    --out tadf
```

A basis can differ by element or by atom: `--basis` names the set of every
atom, `--basis-for EL=NAME` another set for one element (aug-cc-pVDZ on the
heavy atoms and cc-pVDZ on hydrogen), and `--basis-for atoms=I-J,K:NAME`
another set for atoms counted from 1 as in the xyz file (aug-cc-pVDZ on an
emitter's atoms and cc-pVDZ on the host molecules around it). Both repeat,
and an atom range wins over its element. Where one element ends up on two
sets, the atoms on the second are labelled `C@1`, `H@1`, ... (pyscf reads a
labelled atom's basis under its label). Each atom's auxiliary sets follow
its own basis (its -jkfit set for a density-fitted mean field, its -ri set
for the GW and BSE factors), and so does its interpolation grid. An accuracy
level such as G3 is used where every named basis gives it the same grid
counts, which aug-cc-pVDZ and cc-pVDZ do at G1, G2 and G3. Every record
carries the whole map, so a run is never continued under another one.
Formaldehyde in the gas phase:

```bash
python -m src.photophysics --system formaldehyde \
    --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \
    --basis aug-cc-pvdz --basis-for H=cc-pvdz --environment gas \
    --scf df --grid-accuracy G1 --temperature 300 --lambda-env 0.05 \
    --nudge 0.02 --no-far-start --refreeze 1 --saddle-check 0 --df-check 0 \
    --hessian analytic-df --relaxed-solvation equilibrium --span full \
    --manifold-gaps stacked --out tadf
```

`--basis-for atoms=1-2:aug-cc-pvdz` in place of `H=cc-pvdz` puts
aug-cc-pVDZ on C and O only. `solvated` below takes `--basis-for` the same
way; there `--auxbasis` and `--jk-auxbasis` are left out under a map, which
pairs them atom by atom.

`python -m src.photophysics solvated` gives the absorption, emission and
adiabatic S1 and T1 energies of one molecule in the gas phase and in a
continuum, with ADC(2) (the default) or GW-BSE (`--method gwbse`), as one
JSON record that a rerun continues:

```bash
python -m src.photophysics solvated --xyz formaldehyde.xyz --basis cc-pvdz \
    --auxbasis cc-pvdz-ri --jk-auxbasis cc-pvdz-jkfit --environment toluene \
    --saddle-check 1 --out formaldehyde_solvated.json
```
