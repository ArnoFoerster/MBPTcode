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
(tests/test_pbc_rpa_damping.py). The G-space RI-V machinery (ft_aopair, J, B)
is reused from pbc_rpa.
"""
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao

from src.SingleReference.Periodic.pbc_rpa import (make_auxcell, _bz_index,
                                                  coulomb_metric_inv_sqrt)
from src.SingleReference.Periodic.pbc_integrals import (
    PBCDFIntegrals, get_momentum_transfer_map)
from src.SingleReference.Periodic.pbc_rpa_damping import assert_damping_fits

einsum = lib.einsum


def build_dfintegrals_coulG(mf, coulG_fn=None, auxbasis='weigend', mesh=None,
                            check_support=True):
    """PBCDFIntegrals with 3-center MO factors built from an explicit Coulomb
    kernel coulG_fn(cell, q, Gv) -> v(q+G). Default: bare pyscf get_coulG.
    Pass a damped kernel (pbc_rpa_damping.make_coulG_damped) for AUTO damping."""
    cell = mf.cell
    if mesh is not None:
        cell = cell.copy()
        cell.mesh = mesh
        cell.build(False, False)
    if coulG_fn is None:
        coulG_fn = lambda c, q, Gv: tools.get_coulG(c, k=q, mesh=c.mesh, Gv=Gv)

    kpts = np.asarray(mf.kpts)
    # A damped kernel's real-space support must still fit the cell along
    # every unsampled direction at THIS k-mesh -- r0 grows with the grid while
    # the vacuum does not, so a slab that was fine on a coarse mesh can silently
    # stop being fine on a finer one. No-op for any non-damped kernel.
    if check_support:
        assert_damping_fits(cell, kpts, coulG_fn)
    nkpts = len(kpts)
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[-1]
    aux = make_auxcell(cell, auxbasis)
    naux = aux.nao_nr()
    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    b = cell.reciprocal_vectors()
    Gv, _, kws = cell.get_Gv_weights(cell.mesh)
    kws = np.asarray(kws)
    nG = len(Gv)
    kconserv_pair = get_momentum_transfer_map(cell, kpts)

    L = []
    for iq in range(nkpts):
        qs = kscaled[iq]
        J = np.zeros((naux, naux), dtype=np.complex128)
        Bf = np.zeros((naux, nkpts, nmo, nmo), dtype=np.complex128)
        for ki in range(nkpts):
            kj, G0 = _bz_index(kscaled, kscaled[ki] + qs)
            qtrue = (kscaled[kj] + G0 - kscaled[ki]).dot(b)
            vG = coulG_fn(cell, qtrue, Gv) * kws
            auxG = ft_ao.ft_ao(aux, Gv, kpt=qtrue)
            pqG = ft_ao.ft_aopair(cell, Gv, aosym='s1',
                                  kpti_kptj=(kpts[ki], kscaled[kj].dot(b)),
                                  q=qtrue).reshape(nG, nmo, nmo)
            rho = einsum('gpq,pP,qR->gPR', pqG, mo[ki].conj(), mo[kj])
            wA = auxG.conj() * vG[:, None]
            J += einsum('gP,gQ->PQ', wA, auxG)
            Bf[:, ki] += einsum('gP,gpr->Ppr', wA, rho)
        J /= nkpts
        Jm12 = coulomb_metric_inv_sqrt(J)
        Lhalf = einsum('PQ,Qkpr->Pkpr', Jm12, Bf)         # (naux, nkpts, nmo, nmo)
        L.append(np.ascontiguousarray(Lhalf.transpose(1, 0, 2, 3)))

    return PBCDFIntegrals(L, kconserv_pair, mf.mo_energy, mf.mo_occ, kpts,
                          cell=cell, mf=mf)
