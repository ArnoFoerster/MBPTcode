"""Build a PBCDFIntegrals with an explicit (optionally damped) Coulomb kernel,
so the entire periodic pipeline (Pi0, W^q, chi_a, chi_b, plain + vertex
self-energy, hence GWGammaInf/PSD1) inherits the kernel choice with no other
change.

The kernel enters ONLY through coulG_fn(cell, q, Gv) -> v(q+G) (bare pyscf
get_coulG by default). Passing the AUTO Fermi-Dirac damped kernel from
pbc_rpa_damping.make_coulG_damped gives the Coulomb-damped scheme of Forster
et al., JCTC 2025, 21, 9347 (Option A: the damping modifies the DF metric
itself, not just the q=0 self-energy term).

L^q_{p,r}(ki) = [J(q)^{-1/2} B(q)]_{p,r}(ki), FULL MO pairs (the response, W,
the amplitudes and the self-energy need oo/ov/vv, not just ov), with
    B_{P,pr}(ki) = sum_G chi_P(q+G)^* v(q+G) <p_ki|e^{-i(q+G)r}|r_{ki+q}>,
    J_{PQ}(q)    = sum_G chi_P(q+G)^* v(q+G) chi_Q(q+G).
Same factorization convention as PBCDFIntegrals.from_scf
((ij|kl)=einsum('Pij,Pkl',L,L), no conjugation) -- verified by ERI
reconstruction vs pyscf ao2mo and, more stringently, by the bare-kernel RPA
correlation energy matching pbc_rpa.ri_rpa_ecorr to ~1e-14
(tests/test_pbc_rpa_damping.py). The G-space RI-V build is
pbc_rpa.gspace_ri_factors.
"""
import numpy as np

from src.SingleReference.Periodic.pbc_rpa import (gspace_ri_factors,
                                                  kernel_cell, make_auxcell)
from src.SingleReference.Periodic.pbc_integrals import (
    PBCDFIntegrals, get_momentum_transfer_map)


def build_dfintegrals_coulG(mf, coulG_fn=None, auxbasis='weigend', mesh=None,
                            check_support=True):
    """PBCDFIntegrals with 3-center MO factors built from an explicit Coulomb
    kernel coulG_fn(cell, q, Gv) -> v(q+G). Default: bare pyscf get_coulG.
    Pass a damped kernel (pbc_rpa_damping.make_coulG_damped) for AUTO damping."""
    kpts = np.asarray(mf.kpts)
    cell, coulG_fn = kernel_cell(mf.cell, kpts, coulG_fn, mesh, check_support)
    mo = np.asarray(mf.mo_coeff)
    aux = make_auxcell(cell, auxbasis)
    kconserv_pair = get_momentum_transfer_map(cell, kpts)

    L = [np.ascontiguousarray(Lhalf.transpose(1, 0, 2, 3))
         for _, Lhalf in gspace_ri_factors(cell, aux, mo, kpts, coulG_fn)]
    return PBCDFIntegrals(L, kconserv_pair, mf.mo_energy, mf.mo_occ, kpts,
                          cell=cell, mf=mf)
