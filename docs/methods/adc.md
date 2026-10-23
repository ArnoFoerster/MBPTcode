# ADC

**ADC** — the algebraic-diagrammatic construction of Schirmer, Cederbaum and
Walter, [Phys. Rev. A 28, 1237 (1983)](https://doi.org/10.1103/PhysRevA.28.1237),
for ionization potentials and electron affinities (Dyson IP/EA-ADC) and for
neutral excitation energies (EE-ADC).

## Ionization potentials and electron affinities

`ADCSolver(mf, level=...)` picks the branch from the reference: spin-adapted
for RHF, spin-orbital for UHF.

| branch | hand-written levels | generated levels (`pdaggerq=True`) |
|---|---|---|
| RHF, spin-adapted (CSF basis) | ADC(2)-X, ADC(3), Faddeev-ADC(3) | ADC(2), ADC(3), ADC(4), ADC(4) without triples (`adc4t2`), ADC(5) |
| UHF / spin-orbital | ADC(3), Faddeev-ADC(3) (dense reference) | ADC(2), ADC(2)-X, ADC(3), ADC(4) |

The hand-written levels have dense and matrix-free (Davidson) routes, density
fitting, Epstein-Nesbet denominator dressing, and static self-energy
corrections; see `examples/` 01 to 08.

The generated levels run on modules the code generators write on your own
machine (see [Installation](../installation.md)). From ADC(4) on, the
hierarchy can be built in more than one way, and the results differ, so
`construction=` is required there and a call without it is refused:

| `construction=` | what it builds |
|---|---|
| `'twosector'` | the intermediate-state supermatrix of two excitation classes; incomplete from ADC(4) on |
| `'union'` | the complete form, on the union of the excitation spaces |
| `'diagrammatic'` | the complete form, assembled diagram by diagram |

Below ADC(4) the three coincide, and leaving `construction` out gives the
two-sector operator.

## Excitation energies

`solve_ee_adc(mf, level=..., nroots=..., spin=...)` (from
`src.SingleReference.ADC.eeADC.solve`) gives the lowest neutral excitations
at ADC(2) or ADC(3), singlets and triplets separately, dense or matrix-free,
with density fitting (`df=True`):

```python
from pyscf import gto, scf

from src.SingleReference.ADC.eeADC.solve import solve_ee_adc

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri').run()

result = solve_ee_adc(mf, level='adc2', df=True, nroots=3, spin='singlet')
e, Z = result
print(e * 27.2114, 'eV')            # the three lowest singlets
print(result.converged)             # one flag per root
```

For large molecules, `fold=True` solves ADC(2) on the singles space only,
with the doubles folded in at each root's own frequency, on density-fitted
tiles that bound the memory (`tile_gb`) and split over MPI ranks.
`spin_scaling=` applies spin-component scaling (SCS, SOS or any pair of
factors) to the doubles. `environment=` places the molecule in a solvent
continuum, and on the folded route `state_specific='vertical'` or
`'equilibrium'` with `ss_density='relaxed'` adds the state-specific response
of the solvent to the orbital-relaxed density of each root. See
`examples/19_ee_adc.py` and `examples/21_ee_fold.py`.

ADC(2) has analytic nuclear forces on the folded, density-fitted route; they
drive the gas-phase and solvated ADC(2) geometry optimizations of `python -m
src.photophysics solvated` ([Gradients](gradients.md),
[Environments](environments.md)).

**Vertical excitation energies from the command line** — the ADC(2)
singlets and triplets of one molecule, in the gas phase or in a continuum,
on the folded, density-fitted route, written as one JSON record. Run it from
the repository root (or with the root on `PYTHONPATH`):

```bash
python -m src.SingleReference.ADC.eeADC --xyz water.xyz --basis cc-pvdz \
    --auxbasis cc-pvdz-ri --jk-auxbasis cc-pvdz-jkfit --nroots 3 \
    --out-dir water_adc2
```

`water_adc2/record.json` holds the SCF, each spin channel's energies and
convergence flags, and the singlet-triplet gap. A rerun reuses the SCF density
(`scf.npz`) and every finished channel (`channel_<spin>.json`). `--environment
toluene` puts the molecule in a continuum, and `--estimate-only` prints the
memory a run would need. `--help` lists every option.
