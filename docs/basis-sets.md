# Basis sets from CP2K

CP2K's aug-MOLOPT families, all-electron bases with tiered RI sets built for GW
and BSE, are read from CP2K at run time and registered as PySCF basis names:

```python
from src.Base.basis.cp2k_basis import register

basis, aux = register('aug-SZV-MOLOPT-ae', max_error=1e-4)
mol = gto.M(atom='O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240', basis=basis)
```

Without `max_error` you get the tightest RI tiers: converged, but possibly far
larger than needed. Choose the tier deliberately; the RI name then carries the
largest Delta-I among the tiers it holds, so one name means one set. Data
sources, offline use and the trade-off are in `src/Base/basis/README.md`.
