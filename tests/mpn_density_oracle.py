"""The hand-written MP2/MP3 1-RDM corrections (interleaved spin-orbital
MP2/MP3DensityMatrixSolver, spin-blocked MP2/MP3DensityMatrixSolverUnrestricted)
and the spin-orbital MP2 block entry compute_mp2_density_matrix: the
independent oracles the generated MPn density drivers
(src/SingleReference/DensityMatrix/mpn_density_driver*.py), which production
runs, are checked against. Test-only."""
import numpy as np
from pyscf import scf

from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_eri, get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.utils.grids.grids import minimax_time_grid


class MP2DensityMatrixSolver:
    """
    Builds the unrelaxed MP2 1-particle density-matrix correction Delta_gamma^(2).

    Restricted spin only.

    Works in the interleaved spin-orbital basis with antisymmetrized <pq||rs>
    integrals, matching AmplitudeGenerator/get_antisymmetrized_spin_eri's
    convention. The ov block comes from the second-order singles amplitude
    t_i^{a(2)} (no CPHF/Z-vector solve), analogous to
    GWDensityMatrixSolver.compute_unrelaxed_blocks.
    """

    def __init__(self, eps_spin, g_anti_spin, nocc_spin, ncore_spin=0):
        self.eps = eps_spin
        self.g = g_anti_spin
        self.nocc = nocc_spin
        self.ncore = ncore_spin
        self.norb = len(eps_spin)

    def compute_t2(self):
        """t2[a,b,i,j] = t_ij^{ab(1)} = <ab||ij> / (ei+ej-ea-eb)."""
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        ei, ea = self.eps[occ], self.eps[virt]
        g_abij = self.g[virt, virt, occ, occ]
        denom = (ei[None, None, :, None] + ei[None, None, None, :]
                 - ea[:, None, None, None] - ea[None, :, None, None])
        return g_abij / denom

    def compute_blocks(self):
        """
        Spin-orbital blocks (dgamma_oo, dgamma_ov, dgamma_vv)
        of the unrelaxed MP2 density correction.

        When ncore_spin > 0, the returned oo/ov blocks are over active occupied
        orbitals only (dimensions nocc_active × nocc_active and nocc_active × nvirt).
        """
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        t2 = self.compute_t2()
        dgamma_oo = -0.5 * np.einsum('abik,abij->kj', t2, t2)
        dgamma_vv = 0.5 * np.einsum('abij,acij->bc', t2, t2)

        g_ajbc = g[virt, occ, virt, virt]
        g_jkib = g[occ, occ, occ, virt]
        term1 = 0.5 * np.einsum('ajbc,bcij->ia', g_ajbc, t2)
        term2 = 0.5 * np.einsum('jkib,abjk->ia', g_jkib, t2)
        dgamma_ov = (term1 - term2) / (ei[:, None] - ea[None, :])

        return dgamma_oo, dgamma_ov, dgamma_vv

    def compute_blocks_spatial(self):
        """
        Spatial-MO blocks, folding the interleaved spin-orbital blocks.
        restricted reference: alpha block == beta block, so spatial = 2*alpha-alpha).

        When ncore_spin > 0, the active-occ blocks are embedded into full
        (nocc_spatial, nocc_spatial) arrays with zeros in core rows/columns.
        """
        dgamma_oo_act, dgamma_ov_act, dgamma_vv_act = self.compute_blocks()
        oo_sp = dgamma_oo_act[0::2, 0::2] * 2.0
        ov_sp = dgamma_ov_act[0::2, 0::2] * 2.0
        vv_sp = dgamma_vv_act[0::2, 0::2] * 2.0

        ncore_spatial = self.ncore // 2
        if ncore_spatial == 0:
            return oo_sp, ov_sp, vv_sp

        # Embed active-occ blocks into full spatial-MO occupied space
        nocc_spatial = self.nocc // 2
        full_oo = np.zeros((nocc_spatial, nocc_spatial))
        full_oo[ncore_spatial:, ncore_spatial:] = oo_sp
        full_ov = np.zeros((nocc_spatial, vv_sp.shape[0]))
        full_ov[ncore_spatial:, :] = ov_sp
        return full_oo, full_ov, vv_sp


class MP3DensityMatrixSolver(MP2DensityMatrixSolver):
    """Builds the unrelaxed MP3 (third-order) 1-particle density-matrix correction Delta_gamma^(3)

    restricted spin only.
    """

    def compute_t1_second_order(self, t2_1=None):
        """
        t1_2[i,a] = t_i^{a(2)} -- same expression as MP2DensityMatrixSolver.compute_blocks's ov block.
        """
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term1 = 0.5 * np.einsum('ajbc,bcij->ia', g[virt, occ, virt, virt], t2_1, optimize=True)
        term2 = 0.5 * np.einsum('jkib,abjk->ia', g[occ, occ, occ, virt], t2_1, optimize=True)
        return (term1 - term2) / (ei[:, None] - ea[None, :])

    def compute_t2_second_order(self, t2_1=None):
        """
        t2_2[a,b,i,j] = t_ij^{ab(2)}.
        """
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term2 = 0.5 * np.einsum('klij,abkl->abij', g[occ, occ, occ, occ], t2_1, optimize=True)

        raw3 = np.einsum('kacj,cbik->abij', g[occ, virt, virt, occ], t2_1, optimize=True)
        term3 = (raw3 - raw3.transpose(1, 0, 2, 3)
                 - raw3.transpose(0, 1, 3, 2) + raw3.transpose(1, 0, 3, 2))

        term4 = 0.5 * np.einsum('abcd,cdij->abij', g[virt, virt, virt, virt], t2_1, optimize=True)

        denom = (ei[None, None, :, None] + ei[None, None, None, :]
                 - ea[:, None, None, None] - ea[None, :, None, None])
        return (term2 + term3 + term4) / denom

    def compute_t3_second_order(self, t2_1=None):
        """t3_2[a,b,c,i,j,k] = t_ijk^{abc(2)}."""
        if t2_1 is None:
            t2_1 = self.compute_t2()
        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        raw1 = np.einsum('lajk,bcil->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T1 = -(raw1 - raw1.transpose(1, 0, 2, 3, 4, 5)
               - raw1.transpose(0, 1, 2, 4, 3, 5) + raw1.transpose(1, 0, 2, 4, 3, 5))

        raw2 = np.einsum('abdk,dcij->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T2 = -(raw2 - raw2.transpose(0, 2, 1, 3, 4, 5)
               - raw2.transpose(0, 1, 2, 3, 5, 4) + raw2.transpose(0, 2, 1, 3, 5, 4))

        raw3 = np.einsum('laij,bckl->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T3 = -(raw3 - raw3.transpose(1, 0, 2, 3, 4, 5))

        raw4 = np.einsum('lcjk,abil->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T4 = -(raw4 - raw4.transpose(0, 1, 2, 4, 3, 5))

        raw5 = np.einsum('abdi,dcjk->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T5 = -(raw5 - raw5.transpose(0, 2, 1, 3, 4, 5))

        raw6 = np.einsum('bcdk,daij->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        T6 = -(raw6 - raw6.transpose(0, 1, 2, 3, 5, 4))

        T7 = -np.einsum('lcij,abkl->abcijk', g[occ, virt, occ, occ], t2_1, optimize=True)
        T8 = -np.einsum('bcdi,dajk->abcijk', g[virt, virt, virt, occ], t2_1, optimize=True)
        numerator = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8

        denom = (ei[None, None, None, :, None, None] + ei[None, None, None, None, :, None]
                 + ei[None, None, None, None, None, :] - ea[:, None, None, None, None, None]
                 - ea[None, :, None, None, None, None] - ea[None, None, :, None, None, None])
        return numerator / denom

    def compute_t1_third_order(self, t1_2=None, t2_2=None, t3_2=None):
        """
        t1_3[i,a] = t_i^{a(3)}.
        """
        if t1_2 is None:
            t1_2 = self.compute_t1_second_order()
        if t2_2 is None:
            t2_2 = self.compute_t2_second_order()
        if t3_2 is None:
            t3_2 = self.compute_t3_second_order()

        occ, virt = slice(self.ncore, self.nocc), slice(self.nocc, self.norb)
        eps, g = self.eps, self.g
        ei, ea = eps[occ], eps[virt]

        term2 = np.einsum('jabi,jb->ia', g[occ, virt, virt, occ], t1_2, optimize=True)
        term3 = -0.5 * np.einsum('kjbi,bakj->ia', g[occ, occ, virt, occ], t2_2, optimize=True)
        term4 = -0.5 * np.einsum('jabc,bcij->ia', g[occ, virt, virt, virt], t2_2, optimize=True)

        # t3_2.transpose(3,5,4,1,2,0)[i,k,j,b,c,a] == t3_2[a,b,c,i,j,k] (a straight axis
        # relabeling, reordering t3_2's (a,b,c,i,j,k) axes to (i,k,j,b,c,a)).
        t3_2_reordered = t3_2.transpose(3, 5, 4, 1, 2, 0)
        term5 = -0.25 * np.einsum('kjbc,ikjbca->ia', g[occ, occ, virt, virt], t3_2_reordered, optimize=True)

        numerator = term2 + term3 + term4 + term5
        return numerator / (ei[:, None] - ea[None, :])

    def compute_gamma3_blocks(self):
        """Spin-orbital blocks (dgamma_oo, dgamma_ov, dgamma_vv) of Delta_gamma^(3)."""
        t2_1 = self.compute_t2()
        t1_2 = self.compute_t1_second_order(t2_1)
        t2_2 = self.compute_t2_second_order(t2_1)
        t3_2 = self.compute_t3_second_order(t2_1)
        t1_3 = self.compute_t1_third_order(t1_2, t2_2, t3_2)

        gamma3_vv_raw = np.einsum('acij,bcij->ab', t2_1, t2_2, optimize=True)
        dgamma_vv = 0.5 * (gamma3_vv_raw + gamma3_vv_raw.T)

        gamma3_oo_raw = np.einsum('abik,abjk->ij', t2_1, t2_2, optimize=True)
        dgamma_oo = -0.5 * (gamma3_oo_raw + gamma3_oo_raw.T)

        term_b = np.einsum('jc,acij->ia', t1_2, t2_1, optimize=True)
        term_c = 0.25 * np.einsum('bcjk,abcijk->ia', t2_1, t3_2, optimize=True)
        dgamma_ov = t1_3 + term_b + term_c

        return dgamma_oo, dgamma_ov, dgamma_vv

    def compute_gamma3_blocks_spatial(self):
        """Spatial-MO blocks, folding the interleaved spin-orbital blocks.

        restricted reference: alpha block == beta block, so spatial = 2*alpha-alpha.

        When ncore_spin > 0, the active-occ blocks are embedded into full
        (nocc_spatial, nocc_spatial) arrays with zeros in core rows/columns.
        """
        dgamma_oo_act, dgamma_ov_act, dgamma_vv_act = self.compute_gamma3_blocks()
        oo_sp = dgamma_oo_act[0::2, 0::2] * 2.0
        ov_sp = dgamma_ov_act[0::2, 0::2] * 2.0
        vv_sp = dgamma_vv_act[0::2, 0::2] * 2.0

        ncore_spatial = self.ncore // 2
        if ncore_spatial == 0:
            return oo_sp, ov_sp, vv_sp

        # Embed active-occ blocks into full spatial-MO occupied space
        nocc_spatial = self.nocc // 2
        full_oo = np.zeros((nocc_spatial, nocc_spatial))
        full_oo[ncore_spatial:, ncore_spatial:] = oo_sp
        full_ov = np.zeros((nocc_spatial, vv_sp.shape[0]))
        full_ov[ncore_spatial:, :] = ov_sp
        return full_oo, full_ov, vv_sp


class MP2DensityMatrixSolverUnrestricted:
    """Spin-case-resolved (alpha/beta) unrelaxed MP2 1-particle-density-matrix correction.

    Genuinely UHF-general: works for any (eps_a, eps_b, g_aaaa, g_bbbb,
    g_abab, nocc_a, nocc_b), RHF being the special case eps_a=eps_b with all
    three integral blocks built from the same restricted integrals (see
    get_antisymmetrized_spin_block_eri). Reproduces MP2DensityMatrixSolver's
    alpha spin-orbital sub-block exactly for a restricted reference

    Every oo/vv/ov formula below is the direct spin decomposition of
    MP2DensityMatrixSolver.compute_blocks's spin-orbital sum: the same-spin
    (aaaa/bbbb) piece carries over unchanged, and the opposite-spin sum
    contributes via g_abab/t2_1_abab with a factor of 2
    """

    def __init__(self, eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc_a, nocc_b, ncore_a=0, ncore_b=0):
        self.eps_a, self.eps_b = eps_a, eps_b
        self.g_aaaa, self.g_bbbb, self.g_abab = g_aaaa, g_bbbb, g_abab
        self.nocc_a, self.nocc_b = nocc_a, nocc_b
        self.ncore_a, self.ncore_b = ncore_a, ncore_b
        self.norb_a, self.norb_b = len(eps_a), len(eps_b)

    def _slices(self):
        oa = slice(self.ncore_a, self.nocc_a)
        va = slice(self.nocc_a, self.norb_a)
        ob = slice(self.ncore_b, self.nocc_b)
        vb = slice(self.nocc_b, self.norb_b)
        return oa, va, ob, vb

    def compute_t2_1(self):
        """
        t2_1_aaaa/bbbb/abab[a,b,i,j] = t_ij^{ab(1)} = <ab||ij> / (ei+ej-ea-eb), per spin sector.

        abab convention throughout this class: axis order (virt_alpha,
        virt_beta, occ_alpha, occ_beta) -- matches g_abab's own convention.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        den_aa = (ei_a[None, None, :, None] + ei_a[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_a[None, :, None, None])
        t2_1_aaaa = self.g_aaaa[va, va, oa, oa] / den_aa

        den_bb = (ei_b[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_b[:, None, None, None] - ea_b[None, :, None, None])
        t2_1_bbbb = self.g_bbbb[vb, vb, ob, ob] / den_bb

        den_ab = (ei_a[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_b[None, :, None, None])
        t2_1_abab = self.g_abab[va, vb, oa, ob] / den_ab
        return t2_1_aaaa, t2_1_bbbb, t2_1_abab

    def compute_t1_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        """
        t1_2_aa/bb[i,a] = t_i^{a(2)} -- ov block of Delta_gamma^(2) (== MP2's singles response).

        Derived symbolically: pq.add_commutator(1.0,['v'],['t2']) projected
        onto singles (e1), spin_labels={'i':'a','a':'a'} / {'i':'b','a':'b'}
        A single commutator [V,T2] is automatically fully connected, so no filtering
        of disconnected terms needed.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_aa_ovvv = self.g_aaaa[oa, va, va, va]
        g_ab_ooov = self.g_abab[oa, ob, oa, vb]
        g_ab_vovv = self.g_abab[va, ob, va, vb]

        t1_2_aa = (-0.5 * np.einsum('jabc,bcij->ia', g_aa_ovvv, t2_1_aaaa, optimize=True)
                   - 0.5 * np.einsum('kjib,abkj->ia', g_ab_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jkib,abjk->ia', g_ab_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_1_aaaa, optimize=True)
                   + 0.5 * np.einsum('ajbc,bcij->ia', g_ab_vovv, t2_1_abab, optimize=True)
                   + 0.5 * np.einsum('ajcb,cbij->ia', g_ab_vovv, t2_1_abab, optimize=True))
        t1_2_aa = t1_2_aa / (ei_a[:, None] - ea_a[None, :])

        g_bb_ovvv = self.g_bbbb[ob, vb, vb, vb]
        g_ba_ooov = self.g_abab[oa, ob, va, ob]
        g_ba_ovvv = self.g_abab[oa, vb, va, vb]

        t1_2_bb = (-0.5 * np.einsum('kjbi,bakj->ia', g_ba_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jkbi,bajk->ia', g_ba_ooov, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_1_bbbb, optimize=True)
                   + 0.5 * np.einsum('jabc,bcji->ia', g_ba_ovvv, t2_1_abab, optimize=True)
                   + 0.5 * np.einsum('jacb,cbji->ia', g_ba_ovvv, t2_1_abab, optimize=True)
                   - 0.5 * np.einsum('jabc,bcij->ia', g_bb_ovvv, t2_1_bbbb, optimize=True))
        t1_2_bb = t1_2_bb / (ei_b[:, None] - ea_b[None, :])
        return t1_2_aa, t1_2_bb

    def compute_blocks(self):
        """
        Spin-resolved (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) blocks of Delta_gamma^(2), active-space sized.
        """
        t2_1_aaaa, t2_1_bbbb, t2_1_abab = self.compute_t2_1()

        oo_a = -0.5 * (np.einsum('abik,abij->kj', t2_1_aaaa, t2_1_aaaa, optimize=True)
                       + 2.0 * np.einsum('abki,abji->kj', t2_1_abab, t2_1_abab, optimize=True))
        oo_b = -0.5 * (np.einsum('abik,abij->kj', t2_1_bbbb, t2_1_bbbb, optimize=True)
                       + 2.0 * np.einsum('abik,abij->kj', t2_1_abab, t2_1_abab, optimize=True))

        vv_a = 0.5 * (np.einsum('abij,acij->bc', t2_1_aaaa, t2_1_aaaa, optimize=True)
                      + 2.0 * np.einsum('baij,caij->bc', t2_1_abab, t2_1_abab, optimize=True))
        vv_b = 0.5 * (np.einsum('abij,acij->bc', t2_1_bbbb, t2_1_bbbb, optimize=True)
                      + 2.0 * np.einsum('abij,acij->bc', t2_1_abab, t2_1_abab, optimize=True))

        ov_a, ov_b = self.compute_t1_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        return oo_a, oo_b, ov_a, ov_b, vv_a, vv_b


class MP3DensityMatrixSolverUnrestricted(MP2DensityMatrixSolverUnrestricted):
    """
    Spin-case-resolved (alpha/beta) unrelaxed MP3 1-particle-density-matrix correction.
    Same concept as for MP2DensityMatrixSolverUnrestricted
    """

    def compute_t2_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        def antisym4(raw):
            return (raw - raw.transpose(0, 1, 3, 2)
                    - raw.transpose(1, 0, 2, 3) + raw.transpose(1, 0, 3, 2))

        t2_2_aaaa = 0.5 * np.einsum('lkij,ablk->abij', self.g_aaaa[oa, oa, oa, oa], t2_1_aaaa, optimize=True)
        t2_2_aaaa += antisym4(np.einsum('kacj,cbik->abij', self.g_aaaa[oa, va, va, oa], t2_1_aaaa, optimize=True))
        t2_2_aaaa += antisym4(-np.einsum('akjc,bcik->abij', self.g_abab[va, ob, oa, vb], t2_1_abab, optimize=True))
        t2_2_aaaa += 0.5 * np.einsum('abcd,cdij->abij', self.g_aaaa[va, va, va, va], t2_1_aaaa, optimize=True)
        den_aa = (ei_a[None, None, :, None] + ei_a[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_a[None, :, None, None])
        t2_2_aaaa = t2_2_aaaa / den_aa

        t2_2_bbbb = 0.5 * np.einsum('lkij,ablk->abij', self.g_bbbb[ob, ob, ob, ob], t2_1_bbbb, optimize=True)
        t2_2_bbbb += antisym4(-np.einsum('kacj,cbki->abij', self.g_abab[oa, vb, va, ob], t2_1_abab, optimize=True))
        t2_2_bbbb += antisym4(np.einsum('kacj,cbik->abij', self.g_bbbb[ob, vb, vb, ob], t2_1_bbbb, optimize=True))
        t2_2_bbbb += 0.5 * np.einsum('abcd,cdij->abij', self.g_bbbb[vb, vb, vb, vb], t2_1_bbbb, optimize=True)
        den_bb = (ei_b[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_b[:, None, None, None] - ea_b[None, :, None, None])
        t2_2_bbbb = t2_2_bbbb / den_bb

        t2_2_abab = 0.5 * np.einsum('lkij,ablk->abij', self.g_abab[oa, ob, oa, ob], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('klij,abkl->abij', self.g_abab[oa, ob, oa, ob], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('akcj,cbik->abij', self.g_abab[va, ob, va, ob], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('kbcj,caik->abij', self.g_abab[oa, vb, va, ob], t2_1_aaaa, optimize=True)
        t2_2_abab += np.einsum('kbcj,acik->abij', self.g_bbbb[ob, vb, vb, ob], t2_1_abab, optimize=True)
        t2_2_abab += np.einsum('kaci,cbkj->abij', self.g_aaaa[oa, va, va, oa], t2_1_abab, optimize=True)
        t2_2_abab += -np.einsum('akic,cbjk->abij', self.g_abab[va, ob, oa, vb], t2_1_bbbb, optimize=True)
        t2_2_abab += -np.einsum('kbic,ackj->abij', self.g_abab[oa, vb, oa, vb], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('abcd,cdij->abij', self.g_abab[va, vb, va, vb], t2_1_abab, optimize=True)
        t2_2_abab += 0.5 * np.einsum('abdc,dcij->abij', self.g_abab[va, vb, va, vb], t2_1_abab, optimize=True)
        den_ab = (ei_a[None, None, :, None] + ei_b[None, None, None, :]
                  - ea_a[:, None, None, None] - ea_b[None, :, None, None])
        t2_2_abab = t2_2_abab / den_ab
        return t2_2_aaaa, t2_2_bbbb, t2_2_abab

    def _is_restricted(self):
        """
        True when this is effectively an RHF calculation
        """
        return self.g_aaaa is self.g_bbbb

    def compute_t3_2(self, t2_1_aaaa, t2_1_bbbb, t2_1_abab):
        """
        triples amplitude of second-order wave function. Materialized 6-index intermediates,
        so only really reelevant for benchmarking, not useful for production runs.
        For production runs, we have the Laplace transform version of this routine that never
        materializes the tensor but contracts it directly with other intermediates to avoid
        6-index tensor entirely.
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        def denom6(e_i, e_j, e_k, e_a, e_b, e_c):
            return (e_i[None, None, None, :, None, None] + e_j[None, None, None, None, :, None]
                    + e_k[None, None, None, None, None, :] - e_a[:, None, None, None, None, None]
                    - e_b[None, :, None, None, None, None] - e_c[None, None, :, None, None, None])

        def p_ij_ab(raw):
            return (raw - raw.transpose(0, 1, 2, 4, 3, 5)
                    - raw.transpose(1, 0, 2, 3, 4, 5) + raw.transpose(1, 0, 2, 4, 3, 5))

        def p_ab(raw):
            return raw - raw.transpose(1, 0, 2, 3, 4, 5)

        def p_ij(raw):
            return raw - raw.transpose(0, 1, 2, 4, 3, 5)

        def p_jk_bc(raw):
            return (raw - raw.transpose(0, 1, 2, 3, 5, 4)
                    - raw.transpose(0, 2, 1, 3, 4, 5) + raw.transpose(0, 2, 1, 3, 5, 4))

        def p_bc(raw):
            return raw - raw.transpose(0, 2, 1, 3, 4, 5)

        def p_jk(raw):
            return raw - raw.transpose(0, 1, 2, 3, 5, 4)

        # aaa
        t3_2_aaa = p_ij_ab(-np.einsum('lajk,bcil->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_ab(-np.einsum('laij,bckl->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += -np.einsum('lcij,abkl->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_aaaa, optimize=True)
        t3_2_aaa += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_bc(-np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += p_jk(-np.einsum('bcdk,daij->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True))
        t3_2_aaa += -np.einsum('bcdi,dajk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_aaaa, optimize=True)
        t3_2_aaa = t3_2_aaa / denom6(ei_a, ei_a, ei_a, ea_a, ea_a, ea_a)

        # RHF fast-path: on a restricted reference g_aaaa is g_bbbb and
        # t2_1_aaaa == t2_1_bbbb, so t3_2_bbb is identical to t3_2_aaa and
        # t3_2_abb[a,b,c,i,j,k] == t3_2_aab[b,c,a,j,k,i] (swap alpha<->beta
        # labels).  Skip building them explicitly -- just alias/transpose.
        if self._is_restricted():
            t3_2_bbb = t3_2_aaa  # identical arrays for RHF

            # aab (2 alpha virt/occ: a,b,i,j; 1 beta virt/occ: c,k)
            t3_2_aab = p_ij_ab(np.einsum('aljk,bcil->abcijk', self.g_abab[va, ob, oa, ob], t2_1_abab, optimize=True))
            t3_2_aab += p_ab(np.einsum('laij,bclk->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_abab, optimize=True))
            t3_2_aab += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_aaaa, optimize=True))
            t3_2_aab += np.einsum('acdk,dbij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
            t3_2_aab += np.einsum('abdj,dcik->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('acjd,bdik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
            t3_2_aab += np.einsum('acid,bdjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('bcdk,daij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
            t3_2_aab += np.einsum('bcjd,adik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab += -np.einsum('bcid,adjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
            t3_2_aab = t3_2_aab / denom6(ei_a, ei_a, ei_b, ea_a, ea_a, ea_b)

            # abb is a relabeling of aab under the full alpha<->beta spin flip
            # (valid since g_aaaa is g_bbbb and eps_a == eps_b here): the two
            # "pair" slots (a,b / i,j) of aab become abb's two beta slots
            # (b,c / j,k), and aab's single "unlike" slot (c / k) becomes
            # abb's single alpha slot (a / i) -- i.e.
            # t3_2_abb[a,b,c,i,j,k] = -t3_2_aab[b,c,a,k,j,i]  (axes: 2,0,1,5,4,3).
            # The extra minus sign comes from the antisymmetric-permutation
            # convention baked into t3_2_aab's own p_ij_ab/p_ab/... helpers,
            # which are not symmetric under this particular relabeling;
            t3_2_abb = -t3_2_aab.transpose(2, 0, 1, 5, 4, 3)

            return t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb

        # Full UHF path: compute all four sectors independently.
        # bbb (mirror of aaa with beta integrals/amplitudes)
        t3_2_bbb = p_ij_ab(-np.einsum('lajk,bcil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_ab(-np.einsum('laij,bckl->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += -np.einsum('lcij,abkl->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_bbbb, optimize=True)
        t3_2_bbb += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_bc(-np.einsum('abdi,dcjk->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += p_jk(-np.einsum('bcdk,daij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True))
        t3_2_bbb += -np.einsum('bcdi,dajk->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_bbbb, optimize=True)
        t3_2_bbb = t3_2_bbb / denom6(ei_b, ei_b, ei_b, ea_b, ea_b, ea_b)

        # aab (2 alpha virt/occ: a,b,i,j; 1 beta virt/occ: c,k)
        t3_2_aab = p_ij_ab(np.einsum('aljk,bcil->abcijk', self.g_abab[va, ob, oa, ob], t2_1_abab, optimize=True))
        t3_2_aab += p_ab(np.einsum('laij,bclk->abcijk', self.g_aaaa[oa, va, oa, oa], t2_1_abab, optimize=True))
        t3_2_aab += p_ij(-np.einsum('lcjk,abil->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_aaaa, optimize=True))
        t3_2_aab += np.einsum('acdk,dbij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
        t3_2_aab += np.einsum('abdj,dcik->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('acjd,bdik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('abdi,dcjk->abcijk', self.g_aaaa[va, va, va, oa], t2_1_abab, optimize=True)
        t3_2_aab += np.einsum('acid,bdjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('bcdk,daij->abcijk', self.g_abab[va, vb, va, ob], t2_1_aaaa, optimize=True)
        t3_2_aab += np.einsum('bcjd,adik->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab += -np.einsum('bcid,adjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_abab, optimize=True)
        t3_2_aab = t3_2_aab / denom6(ei_a, ei_a, ei_b, ea_a, ea_a, ea_b)

        # abb (1 alpha virt/occ: a,i; 2 beta virt/occ: b,c,j,k)
        t3_2_abb = np.einsum('lbjk,acil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('alik,bcjl->abcijk', self.g_abab[va, ob, oa, ob], t2_1_bbbb, optimize=True)
        t3_2_abb += np.einsum('lbik,aclj->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += np.einsum('alij,bckl->abcijk', self.g_abab[va, ob, oa, ob], t2_1_bbbb, optimize=True)
        t3_2_abb += -np.einsum('lbij,aclk->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('lcjk,abil->abcijk', self.g_bbbb[ob, vb, ob, ob], t2_1_abab, optimize=True)
        t3_2_abb += -np.einsum('lcik,ablj->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += np.einsum('lcij,ablk->abcijk', self.g_abab[oa, vb, oa, ob], t2_1_abab, optimize=True)
        t3_2_abb += p_jk_bc(-np.einsum('abdk,dcij->abcijk', self.g_abab[va, vb, va, ob], t2_1_abab, optimize=True))
        t3_2_abb += p_bc(np.einsum('abid,dcjk->abcijk', self.g_abab[va, vb, oa, vb], t2_1_bbbb, optimize=True))
        t3_2_abb += p_jk(np.einsum('bcdk,adij->abcijk', self.g_bbbb[vb, vb, vb, ob], t2_1_abab, optimize=True))
        t3_2_abb = t3_2_abb / denom6(ei_a, ei_b, ei_b, ea_a, ea_b, ea_b)

        return t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb

    def compute_t3_double_slice_aaa(self, k, j, t2_1_aaaa):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]

        g = self.g_aaaa

        ei_k = ei_a[k]
        ei_j = ei_a[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_a[None, :, None, None]
                    - ea_a[None, None, :, None])

        raw1_kj = np.einsum('la,bcil->abci', g[oa, va, j, k], t2_1_aaaa, optimize=True)
        raw1_alt_kj = np.einsum('lai,bcl->abci', g[oa, va, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T1_kj = -(raw1_kj - raw1_kj.transpose(1, 0, 2, 3)
                  - raw1_alt_kj + raw1_alt_kj.transpose(1, 0, 2, 3))

        raw2_kj = np.einsum('abd,dci->abci', g[va, va, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        raw2_alt_kj = np.einsum('abd,dci->abci', g[va, va, va, j], t2_1_aaaa[:, :, :, k], optimize=True)
        T2_kj = -(raw2_kj - raw2_kj.transpose(0, 2, 1, 3)
                  - raw2_alt_kj + raw2_alt_kj.transpose(0, 2, 1, 3))

        raw3_kj = np.einsum('lai,bcl->abci', g[oa, va, oa, j], t2_1_aaaa[:, :, k, :], optimize=True)
        T3_kj = -(raw3_kj - raw3_kj.transpose(1, 0, 2, 3))

        raw4_kj = np.einsum('lc,abil->abci', g[oa, va, j, k], t2_1_aaaa, optimize=True)
        raw4_alt_kj = np.einsum('lci,abl->abci', g[oa, va, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T4_kj = -(raw4_kj - raw4_alt_kj)

        raw5_kj = np.einsum('abdi,dc->abci', g[va, va, va, oa], t2_1_aaaa[:, :, j, k], optimize=True)
        T5_kj = -(raw5_kj - raw5_kj.transpose(0, 2, 1, 3))

        raw6_kj = np.einsum('bcd,dai->abci', g[va, va, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        raw6_alt_kj = np.einsum('bcd,dai->abci', g[va, va, va, j], t2_1_aaaa[:, :, :, k], optimize=True)
        T6_kj = -(raw6_kj - raw6_alt_kj)

        T7_kj = -np.einsum('lci,abl->abci', g[oa, va, oa, j], t2_1_aaaa[:, :, k, :], optimize=True)
        T8_kj = -np.einsum('bcdi,da->abci', g[va, va, va, oa], t2_1_aaaa[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_bbb(self, k, j, t2_1_bbbb):
        oa, va, ob, vb = self._slices()
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g = self.g_bbbb

        ei_k = ei_b[k]
        ei_j = ei_b[j]
        denom_kj = (ei_b[None, None, None, :] + ei_j + ei_k
                    - ea_b[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        raw1_kj = np.einsum('la,bcil->abci', g[ob, vb, j, k], t2_1_bbbb, optimize=True)
        raw1_alt_kj = np.einsum('lai,bcl->abci', g[ob, vb, ob, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T1_kj = -(raw1_kj - raw1_kj.transpose(1, 0, 2, 3)
                  - raw1_alt_kj + raw1_alt_kj.transpose(1, 0, 2, 3))

        raw2_kj = np.einsum('abd,dci->abci', g[vb, vb, vb, k], t2_1_bbbb[:, :, :, j], optimize=True)
        raw2_alt_kj = np.einsum('abd,dci->abci', g[vb, vb, vb, j], t2_1_bbbb[:, :, :, k], optimize=True)
        T2_kj = -(raw2_kj - raw2_kj.transpose(0, 2, 1, 3)
                  - raw2_alt_kj + raw2_alt_kj.transpose(0, 2, 1, 3))

        raw3_kj = np.einsum('lai,bcl->abci', g[ob, vb, ob, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T3_kj = -(raw3_kj - raw3_kj.transpose(1, 0, 2, 3))

        raw4_kj = np.einsum('lc,abil->abci', g[ob, vb, j, k], t2_1_bbbb, optimize=True)
        raw4_alt_kj = np.einsum('lci,abl->abci', g[ob, vb, ob, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T4_kj = -(raw4_kj - raw4_alt_kj)

        raw5_kj = np.einsum('abdi,dc->abci', g[vb, vb, vb, ob], t2_1_bbbb[:, :, j, k], optimize=True)
        T5_kj = -(raw5_kj - raw5_kj.transpose(0, 2, 1, 3))

        raw6_kj = np.einsum('bcd,dai->abci', g[vb, vb, vb, k], t2_1_bbbb[:, :, :, j], optimize=True)
        raw6_alt_kj = np.einsum('bcd,dai->abci', g[vb, vb, vb, j], t2_1_bbbb[:, :, :, k], optimize=True)
        T6_kj = -(raw6_kj - raw6_alt_kj)

        T7_kj = -np.einsum('lci,abl->abci', g[ob, vb, ob, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T8_kj = -np.einsum('bcdi,da->abci', g[vb, vb, vb, ob], t2_1_bbbb[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_aab(self, k, j, t2_1_aaaa, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_aaaa = self.g_aaaa
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_j = ei_a[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_a[None, :, None, None]
                    - ea_b[None, None, :, None])

        raw1_kj = np.einsum('al,bcil->abci', g_abab[va, ob, j, k], t2_1_abab, optimize=True)
        raw1_alt_kj = np.einsum('ali,bcl->abci', g_abab[va, ob, oa, k], t2_1_abab[:, :, j, :], optimize=True)
        T1_kj = raw1_kj - raw1_alt_kj - raw1_kj.transpose(1, 0, 2, 3) + raw1_alt_kj.transpose(1, 0, 2, 3)

        raw2_kj = np.einsum('lai,bcl->abci', g_aaaa[oa, va, oa, j], t2_1_abab[:, :, :, k], optimize=True)
        T2_kj = raw2_kj - raw2_kj.transpose(1, 0, 2, 3)

        raw3_kj = -np.einsum('lc,abil->abci', g_abab[oa, vb, j, k], t2_1_aaaa, optimize=True)
        raw3_alt_kj = -np.einsum('lci,abl->abci', g_abab[oa, vb, oa, k], t2_1_aaaa[:, :, j, :], optimize=True)
        T3_kj = raw3_kj - raw3_alt_kj

        T4_kj = np.einsum('acd,dbi->abci', g_abab[va, vb, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        T5_kj = np.einsum('abd,dci->abci', g_aaaa[va, va, va, j], t2_1_abab[:, :, :, k], optimize=True)
        T6_kj = -np.einsum('acd,bdi->abci', g_abab[va, vb, j, vb], t2_1_abab[:, :, :, k], optimize=True)
        T7_kj = -np.einsum('abdi,dc->abci', g_aaaa[va, va, va, oa], t2_1_abab[:, :, j, k], optimize=True)
        T8_kj = np.einsum('acid,bd->abci', g_abab[va, vb, oa, vb], t2_1_abab[:, :, j, k], optimize=True)
        T9_kj = -np.einsum('bcd,dai->abci', g_abab[va, vb, va, k], t2_1_aaaa[:, :, :, j], optimize=True)
        T10_kj = np.einsum('bcd,adi->abci', g_abab[va, vb, j, vb], t2_1_abab[:, :, :, k], optimize=True)
        T11_kj = -np.einsum('bcid,ad->abci', g_abab[va, vb, oa, vb], t2_1_abab[:, :, j, k], optimize=True)

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj + T9_kj + T10_kj + T11_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_abb(self, k, j, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_bbbb = self.g_bbbb
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_j = ei_b[j]
        denom_kj = (ei_a[None, None, None, :] + ei_j + ei_k
                    - ea_a[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        T1_kj = np.einsum('lb,acil->abci', g_bbbb[ob, vb, j, k], t2_1_abab, optimize=True)
        T2_kj = -np.einsum('ali,bcl->abci', g_abab[va, ob, oa, k], t2_1_bbbb[:, :, j, :], optimize=True)
        T3_kj = np.einsum('lbi,acl->abci', g_abab[oa, vb, oa, k], t2_1_abab[:, :, :, j], optimize=True)
        T4_kj = np.einsum('ali,bcl->abci', g_abab[va, ob, oa, j], t2_1_bbbb[:, :, k, :], optimize=True)
        T5_kj = -np.einsum('lbi,acl->abci', g_abab[oa, vb, oa, j], t2_1_abab[:, :, :, k], optimize=True)
        T6_kj = -np.einsum('lc,abil->abci', g_bbbb[ob, vb, j, k], t2_1_abab, optimize=True)
        T7_kj = -np.einsum('lci,abl->abci', g_abab[oa, vb, oa, k], t2_1_abab[:, :, :, j], optimize=True)
        T8_kj = np.einsum('lci,abl->abci', g_abab[oa, vb, oa, j], t2_1_abab[:, :, :, k], optimize=True)

        raw9_kj = -np.einsum('abd,dci->abci', g_abab[va, vb, va, k], t2_1_abab[:, :, :, j], optimize=True)
        raw9_alt_kj = -np.einsum('abd,dci->abci', g_abab[va, vb, va, j], t2_1_abab[:, :, :, k], optimize=True)
        T9_kj = raw9_kj - raw9_alt_kj - raw9_kj.transpose(0, 2, 1, 3) + raw9_alt_kj.transpose(0, 2, 1, 3)

        raw10_kj = np.einsum('abid,dc->abci', g_abab[va, vb, oa, vb], t2_1_bbbb[:, :, j, k], optimize=True)
        T10_kj = raw10_kj - raw10_kj.transpose(0, 2, 1, 3)

        raw11_kj = np.einsum('bcd,adi->abci', g_bbbb[vb, vb, vb, k], t2_1_abab[:, :, :, j], optimize=True)
        raw11_alt_kj = np.einsum('bcd,adi->abci', g_bbbb[vb, vb, vb, j], t2_1_abab[:, :, :, k], optimize=True)
        T11_kj = raw11_kj - raw11_alt_kj

        numerator_kj = T1_kj + T2_kj + T3_kj + T4_kj + T5_kj + T6_kj + T7_kj + T8_kj + T9_kj + T10_kj + T11_kj
        return numerator_kj / denom_kj

    def compute_t3_double_slice_abb_alpha(self, k, i, t2_1_bbbb, t2_1_abab):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        g_bbbb = self.g_bbbb
        g_abab = self.g_abab

        ei_k = ei_b[k]
        ei_i = ei_a[i]
        denom_ki = (ei_i + ei_b[None, None, None, :] + ei_k
                    - ea_a[:, None, None, None] - ea_b[None, :, None, None]
                    - ea_b[None, None, :, None])

        T1_ki = np.einsum('lbj,acl->abcj', g_bbbb[ob, vb, ob, k], t2_1_abab[:, :, i, :], optimize=True)
        T2_ki = -np.einsum('al,bcjl->abcj', g_abab[va, ob, i, k], t2_1_bbbb, optimize=True)
        T3_ki = np.einsum('lb,aclj->abcj', g_abab[oa, vb, i, k], t2_1_abab, optimize=True)
        T4_ki = np.einsum('alj,bcl->abcj', g_abab[va, ob, i, ob], t2_1_bbbb[:, :, k, :], optimize=True)
        T5_ki = -np.einsum('lbj,acl->abcj', g_abab[oa, vb, i, ob], t2_1_abab[:, :, :, k], optimize=True)
        T6_ki = -np.einsum('lcj,abl->abcj', g_bbbb[ob, vb, ob, k], t2_1_abab[:, :, i, :], optimize=True)
        T7_ki = -np.einsum('lc,ablj->abcj', g_abab[oa, vb, i, k], t2_1_abab, optimize=True)
        T8_ki = np.einsum('lcj,abl->abcj', g_abab[oa, vb, i, ob], t2_1_abab[:, :, :, k], optimize=True)

        raw9_ki = -np.einsum('abd,dcj->abcj', g_abab[va, vb, va, k], t2_1_abab[:, :, i, :], optimize=True)
        raw9_alt_ki = -np.einsum('abdj,dc->abcj', g_abab[va, vb, va, ob], t2_1_abab[:, :, i, k], optimize=True)
        T9_ki = raw9_ki - raw9_alt_ki - raw9_ki.transpose(0, 2, 1, 3) + raw9_alt_ki.transpose(0, 2, 1, 3)

        raw10_ki = np.einsum('abd,dcj->abcj', g_abab[va, vb, i, vb], t2_1_bbbb[:, :, :, k], optimize=True)
        T10_ki = raw10_ki - raw10_ki.transpose(0, 2, 1, 3)

        raw11_ki = np.einsum('bcd,adj->abcj', g_bbbb[vb, vb, vb, k], t2_1_abab[:, :, i, :], optimize=True)
        raw11_alt_ki = np.einsum('bcdj,ad->abcj', g_bbbb[vb, vb, vb, ob], t2_1_abab[:, :, i, k], optimize=True)
        T11_ki = raw11_ki - raw11_alt_ki

        numerator_ki = T1_ki + T2_ki + T3_ki + T4_ki + T5_ki + T6_ki + T7_ki + T8_ki + T9_ki + T10_ki + T11_ki
        return numerator_ki / denom_ki

    def compute_t1_3(self, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_bbbb, t2_2_abab,
                      t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb):
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]

        t1_3_aa = np.einsum('jabi,jb->ia', self.g_aaaa[oa, va, va, oa], t1_2_aa, optimize=True)
        t1_3_aa += np.einsum('ajib,jb->ia', self.g_abab[va, ob, oa, vb], t1_2_bb, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_2_aaaa, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjib,abkj->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jkib,abjk->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jabc,bcij->ia', self.g_aaaa[oa, va, va, va], t2_2_aaaa, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajbc,bcij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajcb,cbij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)

        t1_3_aa += 0.25 * np.einsum('kjbc,bcaikj->ia', self.g_aaaa[oa, oa, va, va], t3_2_aaa, optimize=True)
        t1_3_aa += -0.25 * np.einsum('kjbc,bacikj->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += -0.25 * np.einsum('jkbc,bacijk->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += 0.25 * np.einsum('kjcb,acbikj->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += 0.25 * np.einsum('jkcb,acbijk->ia', self.g_abab[oa, ob, va, vb], t3_2_aab, optimize=True)
        t1_3_aa += -0.25 * np.einsum('kjbc,acbikj->ia', self.g_bbbb[ob, ob, vb, vb], t3_2_abb, optimize=True)
        t1_3_aa = t1_3_aa / (ei_a[:, None] - ea_a[None, :])

        t1_3_bb = np.einsum('jabi,jb->ia', self.g_abab[oa, vb, va, ob], t1_2_aa, optimize=True)
        t1_3_bb += np.einsum('jabi,jb->ia', self.g_bbbb[ob, vb, vb, ob], t1_2_bb, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jkbi,bajk->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_2_bbbb, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jabc,bcji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jacb,cbji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jabc,bcij->ia', self.g_bbbb[ob, vb, vb, vb], t2_2_bbbb, optimize=True)

        t1_3_bb += -0.25 * np.einsum('kjbc,bcajki->ia', self.g_aaaa[oa, oa, va, va], t3_2_aab, optimize=True)
        t1_3_bb += -0.25 * np.einsum('kjbc,bcakij->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('jkbc,bcajki->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += -0.25 * np.einsum('kjcb,cbakij->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('jkcb,cbajki->ia', self.g_abab[oa, ob, va, vb], t3_2_abb, optimize=True)
        t1_3_bb += 0.25 * np.einsum('kjbc,bcaikj->ia', self.g_bbbb[ob, ob, vb, vb], t3_2_bbb, optimize=True)
        t1_3_bb = t1_3_bb / (ei_b[:, None] - ea_b[None, :])
        return t1_3_aa, t1_3_bb

    def _laplace_aaa_contribution(self, t2_1_aaaa, ntau):
        """
        Laplace-quadrature replacement for compute_gamma3_blocks's "Loop 1"
        (pure-alpha aaa sector)
        """
        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        g = self.g_aaaa
        no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
        g_aaaa_oooo = g[oa, oa, va, va]
        g_vvvo = g[va, va, va, oa]
        g_ovoo = g[oa, va, oa, oa]

        # e_min/e_max only need to bracket the true (positive) three-particle
        # gap ea+eb+ec-ei-ej-ek within the right order of magnitude for the
        # minimax table lookup (it's a ratio-based lookup, not a tight bound)
        gap_min = max(ea_a.min() - ei_a.max(), 1e-3)
        gap_max = ea_a.max() - ei_a.min()
        e_min, e_max = 3.0 * gap_min, 3.0 * gap_max
        tau, sigma = minimax_time_grid(ntau, e_min, e_max)

        # (k,j,b,c) -- feeds t1_3_aa
        outer1 = g_aaaa_oooo

        # (b,c,j,k) -- feeds term_c_a
        outer2 = t2_1_aaaa
        t1_3_aa_contrib = np.zeros((no_a, nv_a))
        term_c_a_contrib = np.zeros((no_a, nv_a))

        for tk in range(ntau):
            t = tau[tk]

            # 1/D = -sum_tau sigma_tau * exp(D*tau)
            w = -sigma[tk]

            Oe = np.exp(ei_a * t)
            Ve = np.exp(-ea_a * t)
            N_outer1 = np.zeros((no_a, nv_a))
            N_outer2 = np.zeros((no_a, nv_a))

            # Term 1: p_ij_ab(-einsum('lajk,bcil->abcijk', g_ovoo, t2))
            G1d = g_ovoo * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
            T1d = t2_1_aaaa * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, :, None]
            p1 = -np.einsum('kjbc,lajk,bcil->ia', outer1, G1d, T1d, optimize=True)
            p2 = -np.einsum('kjbc,laik,bcjl->ia', outer1, G1d, T1d, optimize=True)
            p3 = -np.einsum('kjbc,lbjk,acil->ia', outer1, G1d, T1d, optimize=True)
            p4 = -np.einsum('kjbc,lbik,acjl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += p1 - p2 - p3 + p4
            q1 = -np.einsum('bcjk,lajk,bcil->ia', outer2, G1d, T1d, optimize=True)
            q2 = -np.einsum('bcjk,laik,bcjl->ia', outer2, G1d, T1d, optimize=True)
            q3 = -np.einsum('bcjk,lbjk,acil->ia', outer2, G1d, T1d, optimize=True)
            q4 = -np.einsum('bcjk,lbik,acjl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += q1 - q2 - q3 + q4

            # Term 2: p_ab(-einsum('laij,bckl->abcijk', g_ovoo, t2)) -- same dressed legs as term 1
            r1 = -np.einsum('kjbc,laij,bckl->ia', outer1, G1d, T1d, optimize=True)
            r2 = -np.einsum('kjbc,lbij,ackl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += r1 - r2
            s1 = -np.einsum('bcjk,laij,bckl->ia', outer2, G1d, T1d, optimize=True)
            s2 = -np.einsum('bcjk,lbij,ackl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += s1 - s2

            # Term 3: p_ij(-einsum('lcjk,abil->abcijk', g_ovoo, t2)) -- same dressed legs
            u1 = -np.einsum('kjbc,lcjk,abil->ia', outer1, G1d, T1d, optimize=True)
            u2 = -np.einsum('kjbc,lcik,abjl->ia', outer1, G1d, T1d, optimize=True)
            N_outer1 += u1 - u2
            v1 = -np.einsum('bcjk,lcjk,abil->ia', outer2, G1d, T1d, optimize=True)
            v2 = -np.einsum('bcjk,lcik,abjl->ia', outer2, G1d, T1d, optimize=True)
            N_outer2 += v1 - v2

            # Term 4 (bare): -einsum('lcij,abkl->abcijk', g_ovoo, t2) -- same dressed legs
            N_outer1 += -np.einsum('kjbc,lcij,abkl->ia', outer1, G1d, T1d, optimize=True)
            N_outer2 += -np.einsum('bcjk,lcij,abkl->ia', outer2, G1d, T1d, optimize=True)

            # Term 5: p_jk_bc(-einsum('abdk,dcij->abcijk', g_vvvo, t2)) -- antisymmetry shortcut x4
            G5d = g_vvvo * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, None, :]
            T5d = t2_1_aaaa * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
            N_outer1 += 4.0 * (-np.einsum('kjbc,abdk,dcij->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 4.0 * (-np.einsum('bcjk,abdk,dcij->ia', outer2, G5d, T5d, optimize=True))

            # Term 6: p_bc(-einsum('abdi,dcjk->abcijk', g_vvvo, t2)) -- shortcut x2, same dressed legs
            N_outer1 += 2.0 * (-np.einsum('kjbc,abdi,dcjk->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 2.0 * (-np.einsum('bcjk,abdi,dcjk->ia', outer2, G5d, T5d, optimize=True))

            # Term 7: p_jk(-einsum('bcdk,daij->abcijk', g_vvvo, t2)) -- shortcut x2, same dressed legs
            N_outer1 += 2.0 * (-np.einsum('kjbc,bcdk,daij->ia', outer1, G5d, T5d, optimize=True))
            N_outer2 += 2.0 * (-np.einsum('bcjk,bcdk,daij->ia', outer2, G5d, T5d, optimize=True))

            # Term 8 (bare): -einsum('bcdi,dajk->abcijk', g_vvvo, t2) -- same dressed legs
            N_outer1 += -np.einsum('kjbc,bcdi,dajk->ia', outer1, G5d, T5d, optimize=True)
            N_outer2 += -np.einsum('bcjk,bcdi,dajk->ia', outer2, G5d, T5d, optimize=True)

            t1_3_aa_contrib += w * (-0.25) * N_outer1
            term_c_a_contrib += w * (0.25) * N_outer2

        return t1_3_aa_contrib, term_c_a_contrib

    def compute_gamma3_blocks(self, laplace_ntau=6):
        """Spin-resolved (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) blocks of Delta_gamma^(3), active-space sized.

        Memory-efficient: the t3_2 tensor (V^3 x O^3) is never fully built.
        We compute slices on the fly to keep memory complexity to O(V^3 O^2).
        """
        if self.ncore_a or self.ncore_b:
            wa, wb = slice(self.ncore_a, None), slice(self.ncore_b, None)
            g_aaaa_w = self.g_aaaa[wa, wa, wa, wa]
            # preserve the g_aaaa-is-g_bbbb identity _is_restricted() keys on
            g_bbbb_w = g_aaaa_w if self.g_aaaa is self.g_bbbb else self.g_bbbb[wb, wb, wb, wb]
            sub = MP3DensityMatrixSolverUnrestricted(
                self.eps_a[wa], self.eps_b[wb], g_aaaa_w, g_bbbb_w,
                self.g_abab[wa, wb, wa, wb],
                self.nocc_a - self.ncore_a, self.nocc_b - self.ncore_b)
            return sub.compute_gamma3_blocks(laplace_ntau=laplace_ntau)

        oa, va, ob, vb = self._slices()
        ei_a, ea_a = self.eps_a[oa], self.eps_a[va]
        ei_b, ea_b = self.eps_b[ob], self.eps_b[vb]
        no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
        no_b, nv_b = ei_b.shape[0], ea_b.shape[0]

        t2_1_aaaa, t2_1_bbbb, t2_1_abab = self.compute_t2_1()
        t1_2_aa, t1_2_bb = self.compute_t1_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        t2_2_aaaa, t2_2_bbbb, t2_2_abab = self.compute_t2_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)

        # 1. oo/vv blocks (t2_1 x t2_2 only -- no t3_2 involvement)
        vv_a_raw = (np.einsum('abij,acij->bc', t2_1_aaaa, t2_2_aaaa, optimize=True)
                    + 2.0 * np.einsum('baij,caij->bc', t2_1_abab, t2_2_abab, optimize=True))
        vv_a = 0.5 * (vv_a_raw + vv_a_raw.T)
        vv_b_raw = (np.einsum('abij,acij->bc', t2_1_bbbb, t2_2_bbbb, optimize=True)
                    + 2.0 * np.einsum('abij,acij->bc', t2_1_abab, t2_2_abab, optimize=True))
        vv_b = 0.5 * (vv_b_raw + vv_b_raw.T)

        oo_a_raw = (np.einsum('abik,abij->kj', t2_1_aaaa, t2_2_aaaa, optimize=True)
                    + 2.0 * np.einsum('abki,abji->kj', t2_1_abab, t2_2_abab, optimize=True))
        oo_a = -0.5 * (oo_a_raw + oo_a_raw.T)
        oo_b_raw = (np.einsum('abik,abij->kj', t2_1_bbbb, t2_2_bbbb, optimize=True)
                    + 2.0 * np.einsum('abik,abij->kj', t2_1_abab, t2_2_abab, optimize=True))
        oo_b = -0.5 * (oo_b_raw + oo_b_raw.T)

        # 2. t1_3 terms that do not depend on t3_2
        t1_3_aa = np.einsum('jabi,jb->ia', self.g_aaaa[oa, va, va, oa], t1_2_aa, optimize=True)
        t1_3_aa += np.einsum('ajib,jb->ia', self.g_abab[va, ob, oa, vb], t1_2_bb, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjbi,bakj->ia', self.g_aaaa[oa, oa, va, oa], t2_2_aaaa, optimize=True)
        t1_3_aa += -0.5 * np.einsum('kjib,abkj->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jkib,abjk->ia', self.g_abab[oa, ob, oa, vb], t2_2_abab, optimize=True)
        t1_3_aa += -0.5 * np.einsum('jabc,bcij->ia', self.g_aaaa[oa, va, va, va], t2_2_aaaa, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajbc,bcij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)
        t1_3_aa += 0.5 * np.einsum('ajcb,cbij->ia', self.g_abab[va, ob, va, vb], t2_2_abab, optimize=True)

        t1_3_bb = np.einsum('jabi,jb->ia', self.g_abab[oa, vb, va, ob], t1_2_aa, optimize=True)
        t1_3_bb += np.einsum('jabi,jb->ia', self.g_bbbb[ob, vb, vb, ob], t1_2_bb, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jkbi,bajk->ia', self.g_abab[oa, ob, va, ob], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('kjbi,bakj->ia', self.g_bbbb[ob, ob, vb, ob], t2_2_bbbb, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jabc,bcji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += 0.5 * np.einsum('jacb,cbji->ia', self.g_abab[oa, vb, va, vb], t2_2_abab, optimize=True)
        t1_3_bb += -0.5 * np.einsum('jabc,bcij->ia', self.g_bbbb[ob, vb, vb, vb], t2_2_bbbb, optimize=True)

        # 3. Contraction accumulators for t3_2-dependent terms (Double-loop batched)
        g_aaaa_oooo = self.g_aaaa[oa, oa, va, va]
        g_abab_oooo = self.g_abab[oa, ob, va, vb]
        g_bbbb_oooo = self.g_bbbb[ob, ob, vb, vb]

        term_c_a = np.zeros((no_a, nv_a))
        term_c_b = np.zeros((no_b, nv_b))

        restricted = self._is_restricted()

        # Loop 1: k_alpha, j_alpha (for aaa terms). laplace_ntau (default 6)
        # replaces this O(no_a^2) double loop with the Laplace quadrature from
        # _laplace_aaa_contribution; when restricted, the bbb-diagonal piece
        # of Loop 2a below is *also* skipped and filled in from this same
        # result (bbb is just a relabeled copy of aaa in that case).
        if laplace_ntau is not None:
            t1_3_aa_lap, term_c_a_lap = self._laplace_aaa_contribution(t2_1_aaaa, laplace_ntau)
            t1_3_aa += t1_3_aa_lap
            term_c_a += term_c_a_lap
            if restricted:
                t1_3_bb += t1_3_aa_lap
                term_c_b += term_c_a_lap
        else:
            for k in range(no_a):
                for j in range(no_a):
                    t3_aaa_kj = self.compute_t3_double_slice_aaa(k, j, t2_1_aaaa)
                    t1_3_aa += -0.25 * np.einsum('bc,abci->ia', g_aaaa_oooo[k, j], t3_aaa_kj, optimize=True)
                    term_c_a += 0.25 * np.einsum('bc,abci->ia', t2_1_aaaa[:, :, j, k], t3_aaa_kj, optimize=True)

        # Loop 2: k_beta (for bbb, aab, abb terms)
        for k in range(no_b):
            # 2a: j_beta (for bbb, abb terms). The bbb-diagonal accumulation
            # (t3_bbb_kj and its two accumulator lines) is skipped here when
            # restricted and laplace_ntau is set, since it was already added
            # above from the aaa Laplace result -- t3_abb_kj (cross-spin) is
            # not yet Laplace-accelerated and is still computed every time.
            skip_bbb_diagonal = restricted and laplace_ntau is not None
            for j in range(no_b):
                t3_abb_kj = self.compute_t3_double_slice_abb(k, j, t2_1_bbbb, t2_1_abab)
                t1_3_bb[j, :] += -0.25 * np.einsum('lbc,bcal->a', g_abab_oooo[:, k, :, :], t3_abb_kj, optimize=True)
                t1_3_bb[j, :] += -0.25 * np.einsum('lcb,cbal->a', g_abab_oooo[:, k, :, :], t3_abb_kj, optimize=True)
                term_c_a += 0.25 * np.einsum('bc,abci->ia', t2_1_bbbb[:, :, j, k], t3_abb_kj, optimize=True)

                if not skip_bbb_diagonal:
                    if restricted:
                        t3_bbb_kj = self.compute_t3_double_slice_aaa(k, j, t2_1_aaaa)
                    else:
                        t3_bbb_kj = self.compute_t3_double_slice_bbb(k, j, t2_1_bbbb)
                    t1_3_bb += -0.25 * np.einsum('bc,abci->ia', g_bbbb_oooo[k, j], t3_bbb_kj, optimize=True)
                    term_c_b += 0.25 * np.einsum('bc,abci->ia', t2_1_bbbb[:, :, j, k], t3_bbb_kj, optimize=True)

            # 2b: j_alpha (for aab, abb terms)
            for j in range(no_a):
                t3_aab_kj = self.compute_t3_double_slice_aab(k, j, t2_1_aaaa, t2_1_abab)
                t3_abb_ki = self.compute_t3_double_slice_abb_alpha(k, j, t2_1_bbbb, t2_1_abab)

                t1_3_aa += -0.25 * np.einsum('bc,baci->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += -0.25 * np.einsum('bc,baci->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += 0.25 * np.einsum('cb,acbi->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa += 0.25 * np.einsum('cb,acbi->ia', g_abab_oooo[j, k], t3_aab_kj, optimize=True)
                t1_3_aa[j, :] += 0.25 * np.einsum('lbc,acbl->a', g_bbbb_oooo[k], t3_abb_ki, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lbc,bcal->a', g_aaaa_oooo[:, j, :, :], t3_aab_kj, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lbc,bcal->a', g_abab_oooo[j], t3_abb_ki, optimize=True)
                t1_3_bb[k, :] += 0.25 * np.einsum('lcb,cbal->a', g_abab_oooo[j], t3_abb_ki, optimize=True)

                term_c_a += np.einsum('bc,abci->ia', t2_1_abab[:, :, j, k], t3_aab_kj, optimize=True)
                term_c_b[k, :] += 0.25 * np.einsum('bcl,bcal->a', t2_1_aaaa[:, :, :, j], t3_aab_kj, optimize=True)
                term_c_b += np.einsum('bc,baci->ia', t2_1_abab[:, :, j, k], t3_abb_ki, optimize=True)

        t1_3_aa = t1_3_aa / (ei_a[:, None] - ea_a[None, :])
        t1_3_bb = t1_3_bb / (ei_b[:, None] - ea_b[None, :])

        # 4. Final ov construction
        term_b_a = (np.einsum('jc,acij->ia', t1_2_aa, t2_1_aaaa, optimize=True)
                    + np.einsum('jc,acij->ia', t1_2_bb, t2_1_abab, optimize=True))
        ov_a = t1_3_aa + term_b_a + term_c_a

        term_b_b = (np.einsum('jc,acij->ia', t1_2_bb, t2_1_bbbb, optimize=True)
                    + np.einsum('jc,acij->ia', t1_2_aa, t2_1_abab.transpose(1, 0, 3, 2), optimize=True))
        ov_b = t1_3_bb + term_b_b + term_c_b

        return oo_a, oo_b, ov_a, ov_b, vv_a, vv_b


def compute_mp2_density_matrix(mf, mol=None, nocc=None):
    """
    Build the spin-orbital blocks (oo, ov, vv) of the unrelaxed
    MP2 1-particle density-matrix correction, restricted spin only.

    Spin-orbital basis is interleaved alpha/beta (even=alpha, odd=beta);
    each block is to be added to the corresponding block of the reference
    spin-orbital density (2*delta_ij for occ-occ, 0 elsewhere).
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("compute_mp2_density_matrix is currently restricted-spin only.")

    mol = mol if mol is not None else mf.mol
    nocc = nocc if nocc is not None else mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

    solver = MP2DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
    return solver.compute_blocks()
