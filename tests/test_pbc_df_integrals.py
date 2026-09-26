"""Validation of the k-point DF 3-center MO integrals (PBCDFIntegrals).

Primary oracle (gauge-invariant): the DF chemist ERI reconstructed from the
stored L^q tensors must reproduce pyscf.pbc.df.GDF.ao2mo for every
momentum-conserving MO quartet. This validates both the MO-transform
convention and the momentum-transfer bookkeeping in one shot -- any error in
get_momentum_transfer_map or the r_e2/_conc_mos transform shows up as a
nonzero ERI residual.

Secondary checks: naux is constant within each momentum transfer q (the
property that lets L^q be a dense array), and the 3-center Hermiticity
identity L^{ki,kj}.conj() = L^{kj,ki} transposed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import (
    PBCDFIntegrals, get_momentum_transfer_map)

if __name__ == '__main__':
    all_ok = True

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0.9 0.9 0.9'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 2.5
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts([1, 1, 3])   # asymmetric mesh: distinguishes conjugation conventions
    nk = len(kpts)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    mydf = mf.with_df
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[-1]

    dfints = PBCDFIntegrals.from_scf(cell, mf)

    # --- Secondary: naux constant within each q ---
    ok = all(dfints.L[q].shape[1] == dfints.L[q][0].shape[0] for q in range(nk))
    kscaled = cell.get_scaled_kpts(kpts)
    print(f"L^q is a dense (nk,naux_q,nmo,nmo) array for every q: {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # --- Secondary: momentum-transfer map self-consistency ---
    kcp = get_momentum_transfer_map(cell, kpts)
    ks = kscaled - kscaled[0]
    map_ok = True
    for q in range(nk):
        for ki in range(nk):
            kj = kcp[q, ki]
            d = ks[kj] - (ks[ki] + ks[q])
            if np.linalg.norm(np.round(d) - d) > 1e-8:
                map_ok = False
    print(f"momentum-transfer map kj = ki + q holds for all (q,ki): {'OK' if map_ok else 'FAIL'}")
    all_ok &= map_ok

    # --- Secondary: 3-center Hermiticity L^{ki,kj}.conj()_{ij} = L^{kj,ki}_{ji} ---
    herm_err = 0.0
    for q in range(nk):
        for ki in range(nk):
            kj = kcp[q, ki]
            qrev = dfints._pair_to_q(kj, ki)
            Lij = dfints.L[q][ki]
            Lji = dfints.L[qrev][kj]
            herm_err = max(herm_err, np.abs(Lij.conj() - Lji.transpose(0, 2, 1)).max())
    ok = herm_err < 1e-9
    print(f"3-center Hermiticity L^(ki,kj)* = L^(kj,ki)^T: maxerr={herm_err:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # --- Primary: ERI reconstruction vs pyscf.pbc GDF.ao2mo over all conserving quartets ---
    max_eri_err = 0.0
    n_quartets = 0
    for k1 in range(nk):
        for k2 in range(nk):
            for k3 in range(nk):
                # k4 fixed by conservation k1 - k2 + k3 - k4 = 0
                target = ks[k1] - ks[k2] + ks[k3]
                diff = ks - target
                m = np.where(np.linalg.norm(np.round(diff) - diff, axis=1) < 1e-8)[0]
                k4 = int(m[0])
                ref = mydf.ao2mo([mo[k1], mo[k2], mo[k3], mo[k4]],
                                 [kpts[k1], kpts[k2], kpts[k3], kpts[k4]],
                                 compact=False).reshape(nmo, nmo, nmo, nmo)
                got = dfints.reconstruct_eri(k1, k2, k3, k4)
                max_eri_err = max(max_eri_err, np.abs(got - ref).max())
                n_quartets += 1
    ok = max_eri_err < 1e-9
    print(f"DF ERI reconstruction vs GDF.ao2mo over {n_quartets} conserving quartets: "
          f"maxerr={max_eri_err:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
