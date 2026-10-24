"""k-point density-fitted 3-center MO integrals -- the periodic analog of
Base/pyscf/pyscf_interface.get_density_fitting_coefficients + DFIntegrals.

Builds the complex 3-center MO tensors organized by momentum transfer q, plus
the k-point occ/virt bookkeeping and momentum-conservation maps that every
other periodic module needs.

Convention (verified against pyscf.pbc.df.GDF.ao2mo, see
tests/test_pbc_df_integrals.py):

  L^{ki,kj}_{L,ij} is built from mydf.sr_loop([kpts[ki], kpts[kj]]) followed
  by the same _ao2mo.r_e2 + _conc_mos MO transform pyscf.pbc.gw.krgw_ac uses.
  The density-fitted chemist ERI then factorizes with NO conjugation:

      (i_{ki} j_{kj} | k_{kk} l_{kl}) = sum_L L^{ki,kj}_{L,ij} L^{kk,kl}_{L,kl}

  whenever momentum is conserved: ki - kj + kk - kl = 0 (mod reciprocal
  lattice). Hermiticity of the 3-center object gives
  L^{ki,kj}_{L,ij}.conj() = L^{kj,ki}_{L,ji}, which is why physics
  contractions (polarizability, W) conjugate the bra-side factor -- see
  krgw_ac.get_rho_response for that pattern; it is consistent with the
  no-conjugation ERI factorization above.

Organized by momentum transfer: for a fixed transfer index q, every k-pair
(ki, ki+q) shares the SAME auxiliary momentum (kj - ki = kpts[q]), hence the
same naux, so L for a given q is a dense array (nkpts, naux_q, nmo, nmo).
"""
import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc.df.df import _load3c
from pyscf.pbc.lib.kpts_helper import is_zero


def kpoint_matches(kscaled, target, tol=1e-8):
    """Indices of the scaled k-points equal to `target` up to a reciprocal
    lattice vector (fractional coordinates, `tol` on the distance)."""
    diff = kscaled - target
    return np.where(np.linalg.norm(np.round(diff) - diff, axis=1) < tol)[0]


def bz_index(kscaled, kvec):
    """(idx, G0scaled) with kvec == kscaled[idx] + G0scaled, G0scaled integer."""
    idx = kpoint_matches(kscaled, kvec)
    if len(idx) != 1:
        raise RuntimeError(f"no unique BZ match for {kvec} ({len(idx)} found)")
    i = idx[0]
    return i, np.round(kvec - kscaled[i])


def get_momentum_transfer_map(cell, kpts, tol=1e-8):
    """kconserv_pair[q, ki] = kj such that kpts[kj] = kpts[ki] + kpts[q] (mod G).

    The set of momentum transfers equals the k-mesh itself for a
    Gamma-centered Monkhorst-Pack grid (the standard case here), so the
    transfer index q ranges over the same 0..nkpts-1 as the k-points.
    """
    kscaled = cell.get_scaled_kpts(kpts)
    kscaled = kscaled - kscaled[0]          # Gamma-centered reference
    nk = len(kpts)
    kconserv_pair = -np.ones((nk, nk), dtype=int)
    for q in range(nk):
        for ki in range(nk):
            match = kpoint_matches(kscaled, kscaled[ki] + kscaled[q], tol)
            if len(match) != 1:
                raise RuntimeError(
                    f"momentum transfer map: q={q}, ki={ki} matched {len(match)} "
                    f"k-points (expected 1) -- is the mesh Gamma-centered?")
            kconserv_pair[q, ki] = match[0]
    return kconserv_pair


def gdf_pair_factors(mydf, kpts, pairs):
    """
    Yield (ki, kj, B^{ki,kj}) with B of shape (naux, nao, nao), complex: the
    GDF factors of every k-pair in `pairs`, exactly as `mydf.sr_loop` gives
    them, from one open cderi file. sr_loop opens and searches the file
    again for every pair; a rank reading thousands of pairs from a file of
    hundreds of GB on shared storage spends its time there, not reading.
    Pairs are best given grouped by transfer: the builder writes each
    transfer and its conjugate together. Three-dimensional cells only: a
    file with negative-metric blocks is refused.
    """
    nao = mydf.cell.nao_nr()
    with _load3c(mydf._cderi, mydf._dataname) as getitem:
        if mydf._dataname + '-' in getitem.__self__.feri:
            raise NotImplementedError('gdf_pair_factors: negative-metric GDF blocks '
                                      '(low-dimensional cell); three-dimensional only.')
        for ki, kj in pairs:
            kptij = np.asarray([kpts[ki], kpts[kj]])
            j3c = getitem(*kptij)
            data = np.asarray(j3c[0:j3c.shape[0]])
            # the post-processing of sr_loop(compact=False), on the whole block
            if is_zero(kptij):
                real = np.asarray(data.real)
                if is_zero(kptij[0] - kptij[1]) and real.shape[1] != nao ** 2:
                    real = lib.unpack_tril(real).reshape(-1, nao ** 2)
                imag = np.zeros_like(real)
            else:
                real = np.asarray(data.real, order='C')
                imag = np.asarray(data.imag, order='C')
                if is_zero(kptij[0] - kptij[1]) and real.shape[1] != nao ** 2:
                    real = lib.unpack_tril(real).reshape(-1, nao ** 2)
                    imag = lib.unpack_tril(imag, lib.ANTIHERMI).reshape(-1, nao ** 2)
            yield ki, kj, (real + 1j * imag).reshape(-1, nao, nao)


class PBCDFIntegrals:
    """k-point DF 3-center MO tensors L^q, container parallel to the molecular
    DFIntegrals (Base/pyscf/pyscf_interface.py).

    Attributes
    ----------
    L : list of ndarray
        L[q] has shape (nkpts, naux_q, nmo, nmo), complex. L[q][ki] is the
        3-center MO factor for the k-pair (ki, kj = kconserv_pair[q, ki]).
    kconserv_pair : ndarray (nkpts, nkpts), int
        kconserv_pair[q, ki] = kj with kpts[kj] = kpts[ki] + kpts[q].
    mo_energy : ndarray (nkpts, nmo)
    mo_occ : ndarray (nkpts, nmo)
    nocc : ndarray (nkpts,) int
        Occupied count per k-point.
    kpts : ndarray (nkpts, 3)
    nkpts, nmo : int
    """

    def __init__(self, L, kconserv_pair, mo_energy, mo_occ, kpts, cell=None, mf=None):
        self.L = L
        self.kconserv_pair = kconserv_pair
        self.mo_energy = np.asarray(mo_energy)
        self.mo_occ = np.asarray(mo_occ)
        self.kpts = np.asarray(kpts)
        self.nkpts = len(kpts)
        self.nmo = self.mo_energy.shape[1]
        self.nocc = (self.mo_occ > 1e-8).sum(axis=1).astype(int)
        self.cell = cell
        self.mf = mf

    @classmethod
    def from_scf(cls, cell, mf, tol=1e-8, factors=None):
        """Build all L^q from a converged KRHF/KRKS mean field with GDF.

        Mirrors DFIntegrals.from_scf: pulls the DF object off mf, transforms
        the AO 3-center integrals to the MO basis per k-pair, and packs them
        by momentum transfer. `factors` is a source of the AO factors
        (`pbc_gdf_build`: FileGDF, or DistributedGDF to build them over the
        ranks without a cderi file); by default each pair is read with
        mf.with_df.sr_loop, which also takes the negative-metric blocks of a
        low-dimensional cell.
        """
        mydf = getattr(mf, 'with_df', None)
        if mydf is None:
            raise ValueError("mf has no with_df -- periodic path is DF-only "
                             "(density_fit() the KRHF/KRKS object).")
        kpts = np.asarray(mf.kpts)
        nkpts = len(kpts)
        mo_coeff = np.asarray(mf.mo_coeff)          # (nkpts, nao, nmo)
        mo_energy = np.asarray(mf.mo_energy)        # (nkpts, nmo)
        mo_occ = np.asarray(mf.mo_occ)              # (nkpts, nmo)
        nao = cell.nao_nr()
        nmo = mo_coeff.shape[-1]

        kconserv_pair = get_momentum_transfer_map(cell, kpts, tol=tol)
        if factors is not None:
            B = factors(kpts, kconserv_pair, np.arange(nkpts))
            L = [np.stack([cls._mo_transform(B[q][ki].reshape(-1, nao * nao),
                                             mo_coeff, ki, kconserv_pair[q, ki], nmo)
                           for ki in range(nkpts)]) for q in range(nkpts)]
            return cls(L, kconserv_pair, mo_energy, mo_occ, kpts, cell=cell, mf=mf)

        L = []
        for q in range(nkpts):
            Lq = None
            for ki in range(nkpts):
                kj = kconserv_pair[q, ki]
                Lij = cls._build_Lij_mo(mydf, kpts, mo_coeff, ki, kj, nao, nmo)
                if Lq is None:
                    naux = Lij.shape[0]
                    Lq = np.empty((nkpts, naux, nmo, nmo), dtype=np.complex128)
                elif Lij.shape[0] != Lq.shape[1]:
                    raise RuntimeError(
                        f"naux mismatch within momentum transfer q={q}: "
                        f"ki={ki} gave naux={Lij.shape[0]}, expected {Lq.shape[1]}")
                Lq[ki] = Lij
            L.append(Lq)

        return cls(L, kconserv_pair, mo_energy, mo_occ, kpts, cell=cell, mf=mf)

    @staticmethod
    def _build_Lij_mo(mydf, kpts, mo_coeff, ki, kj, nao, nmo):
        """MO 3-center factor L^{ki,kj}_{L,ij}, shape (naux, nmo, nmo), complex.

        Same construction as krgw_ac.get_sigma_diag: read the AO short-range
        3-center integrals for the k-pair, then r_e2 MO-transform them.
        """
        Lpq = []
        for LpqR, LpqI, sign in mydf.sr_loop([kpts[ki], kpts[kj]], compact=False):
            Lpq.append(LpqR + LpqI * 1.0j)
        return PBCDFIntegrals._mo_transform(np.vstack(Lpq).reshape(-1, nao * nao),
                                            mo_coeff, ki, kj, nmo)

    @staticmethod
    def _mo_transform(Lpq, mo_coeff, ki, kj, nmo):
        """(naux, nao^2) AO factor of the pair (ki, kj) -> (naux, nmo, nmo) MO."""
        moij, ijslice = _conc_mos(mo_coeff[ki], mo_coeff[kj])[2:]
        Lij = _ao2mo.r_e2(Lpq, moij, ijslice, [], None)
        return Lij.reshape(-1, nmo, nmo)

    # ---- bookkeeping helpers ----

    def occ_slice(self, k):
        return slice(0, self.nocc[k])

    def virt_slice(self, k):
        return slice(self.nocc[k], self.nmo)

    def L_ov(self, q, ki):
        """Occ-virt block L^{q}_{i in ki, a in ki+q}, shape (naux, nocc_ki, nvirt_kj)."""
        kj = self.kconserv_pair[q, ki]
        return self.L[q][ki][:, self.occ_slice(ki), self.virt_slice(kj)]

    def Lblock(self, k1, k2):
        """Full MO 3-center factor L^{k1,k2}_{L,pq}, shape (naux, nmo, nmo).

        Works for any k-pair; picks the stored momentum-transfer slice
        (q = k2 - k1) automatically. Downstream A/B assembly slices this by
        occ/virt to get ov/oo/vv blocks at arbitrary momentum transfer.
        """
        q = self.pair_to_q(k1, k2)
        return self.L[q][k1]

    def eri_block(self, k1, k2, k3, k4, s1, s2, s3, s4):
        """DF chemist MO integral (p_k1 q_k2 | r_k3 s_k4) for orbital ranges
        s1..s4 (slices), from the stored L^q. Momentum must conserve:
        k1 - k2 + k3 - k4 = 0. Returns a 4-index block, no conjugation
        (see the module-level convention note)."""
        L12 = self.Lblock(k1, k2)[:, s1, s2]
        L34 = self.Lblock(k3, k4)[:, s3, s4]
        return np.einsum('Lpq,Lrs->pqrs', L12, L34)

    # ---- validation ----

    def reconstruct_eri(self, k1, k2, k3, k4):
        """DF chemist ERI (i_{k1} j_{k2} | k_{k3} l_{k4}) from the stored L^q.

        Materializes the nmo^4 tensor -- validation/plumbing only, exactly like
        DFIntegrals.reconstruct_g. Requires momentum conservation
        k1 - k2 + k3 - k4 = 0; picks the q-slices that realize the two pairs.
        """
        q12 = self.pair_to_q(k1, k2)
        q34 = self.pair_to_q(k3, k4)
        L12 = self.L[q12][k1]
        L34 = self.L[q34][k3]
        return np.einsum('Lij,Lkl->ijkl', L12, L34)

    def pair_to_q(self, ki, kj):
        """Transfer index q with kconserv_pair[q, ki] == kj."""
        matches = np.where(self.kconserv_pair[:, ki] == kj)[0]
        if len(matches) != 1:
            raise RuntimeError(f"no unique momentum transfer for pair ({ki},{kj})")
        return matches[0]

    def kpoint_minus_q(self, q, kp):
        """Index km with kpts[km] = kpts[kp] - kpts[q] (mod G), i.e.
        kconserv_pair[q, km] == kp: the transfers ARE k-points, so km is
        recovered by inverting kconserv_pair along its other axis."""
        matches = np.where(self.kconserv_pair[q] == kp)[0]
        if len(matches) != 1:
            raise RuntimeError(f"no unique k - q for kp={kp}, q={q}")
        return int(matches[0])
