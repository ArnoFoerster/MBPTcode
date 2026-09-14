# Basis sets

`cp2k_basis.py` makes the aug-MOLOPT basis families of Pasquier, Graml and Wilhelm,
[J. Chem. Theory Comput. 22, 540 (2026)](https://doi.org/10.1021/acs.jctc.5c01386),
available to PySCF and to every route in this tree. They are all-electron Gaussian
bases built for TDDFT, GW and BSE, each with auxiliary (RI) sets in several tiers.
CP2K ships them as `data/BASIS_AUG_MOLOPT` and `data/BASIS_RI_AUG_MOLOPT`; MBPTcode
copies neither file, reads them at run time and returns PySCF basis dicts.

## Usage

Run from the repository root, like every script here:

```python
from pyscf import dft, gto
from src.Base.basis.cp2k_basis import load_basis, load_ri_basis

elements = ['H', 'C', 'O']
mol = gto.M(atom='H 0 0.934 -0.588; H 0 -0.934 -0.588; C 0 0 0; O 0 0 1.221',
            basis=load_basis('aug-SZV-MOLOPT-ae', elements))
aux = load_ri_basis('aug-SZV-MOLOPT-ae', elements, max_error=1e-4)
mf = dft.RKS(mol, xc='PBE').density_fit(auxbasis=aux).run()
```

Both dicts hold exactly the elements passed, so list every element of the molecule.

Pass `aux` explicitly wherever a route takes an auxiliary basis, for example
`solve_bse_isdf(mf, mol, nocc, auxbasis=aux)`. The defaults in this tree build
`str(mol.basis) + '-ri'`, which means nothing for a dict. For the same reason no
row of the shipped ISDF radii (`src/Base/data/optimized_radii.json`, keyed on basis
names) matches these sets: `isdf_grid` falls back to `optimize_atomic_radii`, which
caches its result under `src/Base/data/radii_cache/`.

`examples/12_cp2k_aug_molopt.py` is the worked case, G0W0 and a dense BSE on
formaldehyde. `tests/test_cp2k_basis.py` builds every orbital block and RI tier and
checks each against the function count its header implies.

## Where the files come from

`data_file(kind)` looks in this order:

| source | used when |
|---|---|
| `$MBPT_CP2K_DATA/` | set; a CP2K `data/` directory, read as it is |
| `$MBPT_CP2K_CACHE/<commit>/`, default `~/.cache/mbptcode/cp2k/<commit>/` | the file is there |
| a download from `github.com/cp2k/cp2k` at the pinned commit, into that cache | otherwise |

Files at the pinned commit are checked against their sha256, and a mismatch raises.
On first use of each file the module prints a notice naming the source, CP2K's
license and the paper we would kindly ask you to cite. `provenance(kind)` returns
the path, digest and commit for a run record.

On a node without network, run `python -m src.Base.basis.cp2k_basis fetch` once
where the network is reachable and share the cache, or point `MBPT_CP2K_DATA` at a
CP2K checkout. `data_file(kind, download=False)` raises instead of downloading.

## Sets and elements

| orbital set | elements |
|---|---|
| `aug-SZV-MOLOPT-ae`, `aug-SZV-MOLOPT-ae-mini`, `aug-DZVP-MOLOPT-ae`, `aug-TZVP-MOLOPT-ae` | H to Cl |
| `aug-SZV-MOLOPT-ae-SR` | H, C, N, O, Al, Si |

RI tiers exist for H to Cl. `python -m src.Base.basis.cp2k_basis scout C O` lists,
per element, every orbital set with its function count and every RI tier with its
size and Delta-I; `available(element)` returns the same as a dict.

## Choosing the RI tier

The tier is your choice, and it is worth making deliberately. Without one you get
the tightest tier of every element: converged, but the auxiliary set can be twice
the size it needs to be, and every density-fitted and ISDF step pays for that.

A tier's name carries its Delta-I, the atomic RI-MP2 error of eq 29 in the paper.

| call | tier per element |
|---|---|
| `load_ri_basis(name, elements)` | the tightest |
| `max_error=1e-4` | the smallest with Delta-I <= 1e-4, the paper's recommendation |
| `min_lmax=L` | also requires auxiliary l_max >= L; if no tier qualifies, the tightest, with a warning |

`pick_ri_tier` takes the same arguments for one element and returns the tier's
name, size, Delta-I and pattern.

Delta-I is an MP2 criterion: it measures the (ia|jb) integrals. The BSE direct
term and the GW self-energy contract (ij|ab), and a product of two orbital
functions needs auxiliary functions up to twice the orbital l_max. So a tier
within 1e-4 can still miss the angular momenta these routes need. Measured on
formaldehyde in `aug-SZV-MOLOPT-ae`, density fitting against the exact four-center
tensor:

| RI tiers | auxiliary functions | five lowest BSE singlets | HOMO, LUMO |
|---|---|---|---|
| Delta-I 1e-4 (no d on H, no g on O) | 116 | up to 27 meV | up to 6.2 meV |
| tightest | 226 | up to 1.6 meV | under 1 meV |

Check the tier for your own system the same way before a production run: one
calculation against the exact tensor, or against the tightest tier where the
exact tensor does not fit, on the quantity you report.

## License

The basis data remain CP2K's, distributed under GPL-2.0-or-later; MBPTcode
redistributes none of it. If you use these basis sets, we would kindly ask you to
cite Pasquier, Graml and Wilhelm,
[J. Chem. Theory Comput. 22, 540 (2026)](https://doi.org/10.1021/acs.jctc.5c01386).
