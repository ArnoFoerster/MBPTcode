# Basis sets

`cp2k_basis.py` makes the aug-MOLOPT basis families of Pasquier, Graml and Wilhelm,
[J. Chem. Theory Comput. 22, 540 (2026)](https://doi.org/10.1021/acs.jctc.5c01386),
available to PySCF and to every route in this tree. They are all-electron Gaussian
bases built for TDDFT, GW and BSE, each with auxiliary (RI) sets in several tiers.
CP2K ships them as `data/BASIS_AUG_MOLOPT` and `data/BASIS_RI_AUG_MOLOPT`; MBPTcode
copies neither file, reads them at run time and hands them to PySCF.

## Usage

Run from the repository root, like every script here:

```python
from pyscf import dft, gto
from src.Base.basis.cp2k_basis import register

basis, aux = register('aug-SZV-MOLOPT-ae', max_error=1e-4)
mol = gto.M(atom='H 0 0.934 -0.588; H 0 -0.934 -0.588; C 0 0 0; O 0 0 1.221',
            basis=basis)
mf = dft.RKS(mol, xc='PBE').density_fit(auxbasis=aux).run()
```

`register` makes the orbital set and its RI tiers PySCF basis names, here
`aug-SZV-MOLOPT-ae` and `aug-SZV-MOLOPT-ae-ri-7.3e-05`, for every element that has
them. Every route in this tree then treats them like any named basis. PySCF keeps
the names in the running process only, so call `register` once at the top of each
script. It writes the sets into the cache (below) and returns the names.

One RI name means one set of tiers, in every process:

| call | RI name | tier per element |
|---|---|---|
| `register(name)`, or `max_error=0` | `<name>-ri` | the tightest |
| `register(name, max_error=1e-4)` | `<name>-ri-7.3e-05` | the smallest with Delta-I <= 1e-4 |

The number in the name is the smallest threshold that picks the same tiers, so
every threshold that gives the same set gives the same name, and passing it back
reproduces the set. A threshold above every tier's Delta-I, 1 or `inf`, gives the
smallest tiers. A threshold call registers only its own RI name; the defaults of
this tree form `str(mol.basis) + '-ri'`, which `register(name)` registers, so
after a threshold call pass `aux` to every route, or it raises.

`load_basis` and `load_ri_basis` return the same sets as dicts, and only
`load_ri_basis` takes `min_lmax` (below). A dict has no name, so pass it
explicitly wherever a route takes an auxiliary basis; the ISDF routes form
`str(mol.basis) + '-ri'` in one place no argument reaches, so they need a
registered name.

Names are only registered for the basis data of the pinned CP2K commit; a file
that differs in comments or spacing passes, one whose data differ is refused with
the elements named. The guard is per element and file, so a changed block of any
set refuses every registration. The ISDF radii cache is keyed on the name, and
other data under the same name would reuse a grid optimized for a different set.

`examples/14_cp2k_aug_molopt.py` is the worked case, G0W0 and a dense BSE on
formaldehyde. `tests/test_cp2k_basis.py` builds every orbital block and RI tier and
checks each against the function count its header implies.

## Where the files come from

`data_file(kind)` looks in this order:

| source | used when |
|---|---|
| `$MBPT_CP2K_DATA/` | set; a CP2K `data/` directory, read as it is |
| `$MBPT_CP2K_CACHE/<commit>/`, default `~/.cache/mbptcode/cp2k/<commit>/` | the file is there |
| a download from `github.com/cp2k/cp2k` at the pinned commit, into that cache | otherwise |

`register` writes its name files to `pyscf/` in the cache directory of the pinned
commit, whichever source the CP2K files came from.

`cp2k_pin.json` beside the module holds the pinned commit, the sha256 of both
files and, per element, the digest of its blocks with comments and spacing
dropped (`content_digests`). Files at the pinned commit are checked against
their sha256, and a mismatch raises.
On first use of each file the module prints a notice naming the source, CP2K's
license and the paper we would kindly ask you to cite. `provenance(kind)` returns
the path, the digest, the commit when the file is the pinned one and, as
`blocks`, the commit when the basis data are, for a run record.

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

`aug-SZV-MOLOPT-ae-mini` has no RI tiers for H and He in CP2K's file: H shares
its orbital block with `aug-SZV-MOLOPT-ae`, whose tiers are named after that set
only, so `aug-SZV-MOLOPT-ae-mini-ri` raises for any molecule with H. A fallback
used in production, the nanographene calculations of
[lsGW_Nanographenes](https://github.com/MGraml/lsGW_Nanographenes), is the H tier
of `aug-SZV-MOLOPT-ae-SR`. Build the dict and pass it explicitly:

```python
aux = load_ri_basis('aug-SZV-MOLOPT-ae-mini', ['C', 'N'], max_error=1e-4)
aux['H'] = load_ri_basis('aug-SZV-MOLOPT-ae-SR', ['H'], max_error=1e-4)['H']
```

The sets stay well conditioned despite their diffuse functions. On a
hydrogen-terminated graphene flake of 184 atoms, the overlap condition number is
2.2e5 in `aug-SZV-MOLOPT-ae` and 2.8e6 in `aug-DZVP-MOLOPT-ae`, against 3.0e14 in
aug-cc-pVDZ.

## Choosing the RI tier

The tier is your choice, and it is worth making deliberately. Without one you get
the tightest tier of every element: converged, but the auxiliary set can be twice
the size it needs to be, and every density-fitted and ISDF step pays for that.

A tier's name carries its Delta-I, the atomic RI-MP2 error of eq 29 in the paper.

| argument | tier per element |
|---|---|
| none, or `max_error=0` | the tightest |
| `max_error=1e-4` | the smallest with Delta-I <= 1e-4, the paper's recommendation |
| `min_lmax=L`, `load_ri_basis` only | also requires auxiliary l_max >= L; if no tier qualifies, the tightest, with a warning |

An element without a tier within the threshold gets its tightest one, and one
warning names the elements and the Delta-I each of them gets; a threshold that
picks every element's tightest tier gets one line saying so. `pick_ri_tier` takes
the same arguments for one element and returns the tier's name, size, Delta-I and
pattern.

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
