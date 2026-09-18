import numpy as np
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.base import get_occ_virt_indices
from src.Base.constants import (CASIDA_NORM_TOL,
                                DEFAULT_BROADENING_ETA)
from src.SingleReference.LinearResponse import imaginary_frequency


def gram_product(C, block_elems=2**27):
    """V = C^T C for C of shape (naux, n), assembled in row blocks of at most
    block_elems values.

    Written as ``C.T @ C`` the product goes to the BLAS symmetric rank-k kernel,
    where the OpenBLAS bundled with the numpy wheel segfaults once n^2 reaches a
    few 10^9 values. A row block V[i0:i1] = C[:, i0:i1]^T C is a plain GEMM on
    distinct buffers, which holds at those sizes, so every block takes that
    path, straight into its slice of V.
    """
    n = C.shape[1]
    V = np.empty((n, n), dtype=np.result_type(C, C))
    rows = max(1, block_elems // max(n, 1))
    for i0 in range(0, n, rows):
        i1 = min(i0 + rows, n)
        np.matmul(C[:, i0:i1].T, C, out=V[i0:i1])
    return V


class LinearResponseSolver:
    """RPA and BSE linear response solver, restricted (RHF/singlet/triplet) or unrestricted (UHF), DF or full 4-center ERIs."""
    def __init__(self, eps, coeff_df=None, eri_chemist=None, spin_mode='restricted', eta=DEFAULT_BROADENING_ETA,
                 coeff_ov=None):
        """eps/coeff_df/eri_chemist: single array if restricted, (alpha, beta[, ab]) tuple if unrestricted.

        coeff_ov: the (naux, nocc*nvirt) occupied-virtual block on its own, for
        callers that never need the full (naux, norb, norb) coeff_df -- see
        `get_df_coefficients_ov`. When given, the RPA screening uses it directly
        and coeff_df may be left None.
        """
        self.spin_mode = spin_mode.lower()
        self.eta = eta
        self.df_coeff = coeff_df
        self.coeff_ov = coeff_ov
        self.eri_chemist = eri_chemist
        
        if self.spin_mode == 'unrestricted':
            self.eps_a, self.eps_b = eps
            if coeff_df is not None:
                if len(coeff_df) == 3:
                    self.coeff_a, self.coeff_b, self.df_ab = coeff_df
                else:
                    self.coeff_a, self.coeff_b = coeff_df
                    self.df_ab = None
            else:
                self.coeff_a, self.coeff_b, self.df_ab = None, None, None
            self.eri_a, self.eri_b, self.eri_ab = eri_chemist if eri_chemist is not None else (None, None, None)
            self.norb_a = len(self.eps_a)
            self.norb_b = len(self.eps_b)
            if self.coeff_a is not None:
                self.naux = self.coeff_a.shape[0]
        else:
            self.eps = eps
            self.norb = len(self.eps)
            if self.df_coeff is not None:
                self.naux = self.df_coeff.shape[0]
            elif self.coeff_ov is not None:
                self.naux = self.coeff_ov.shape[0]

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)

    def construct_4d_w_rpa(self, nocc, spin_channel='alpha'):
        """
        Full-ERI (non-DF) counterpart to solve_rpa_screening: builds the bare
        4D W_rpa = V - V.(X+Y)(X+Y)^T.V/omega tensor from an RPA Casida solve,
        for use as the vertex-correction screened interaction when df=False.
        """
        if self.spin_mode == 'unrestricted':
            eps = (self.eps_a, self.eps_b)
            eri = (self.eri_a, self.eri_b, self.eri_ab)

            omega_rpa, X_rpa, Y_rpa = CasidaSolver(
                *self.build_casida_matrices(nocc, lBSE=False)).solve()

            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(eps[0], nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(eps[1], nocc_b)
            n_pair_a = len(occ_a) * len(virt_a)
            n_pair_b = len(occ_b) * len(virt_b)

            XpY_a = (X_rpa[:n_pair_a] + Y_rpa[:n_pair_a])
            XpY_b = (X_rpa[n_pair_a:] + Y_rpa[n_pair_a:])
            del X_rpa, Y_rpa

            V_aa_matrix = eri[0][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
            V_ba_matrix = eri[2].transpose(2, 3, 0, 1)[np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
            M_a = V_aa_matrix.T @ XpY_a + V_ba_matrix.T @ XpY_b
            screened_a = 2.0 * (M_a / omega_rpa[None, :]) @ M_a.T
            del M_a
            W_rpa_a = screened_a.reshape(eri[0].shape)
            np.subtract(eri[0], W_rpa_a, out=W_rpa_a)

            V_ab_matrix = eri[2][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
            V_bb_matrix = eri[1][np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
            M_b = V_ab_matrix.T @ XpY_a + V_bb_matrix.T @ XpY_b
            screened_b = 2.0 * (M_b / omega_rpa[None, :]) @ M_b.T
            del M_b
            W_rpa_b = screened_b.reshape(eri[1].shape)
            np.subtract(eri[1], W_rpa_b, out=W_rpa_b)

            eri_w_singlet = W_rpa_a if spin_channel == 'alpha' else W_rpa_b
            eri_w_triplet = eri_w_singlet
        else:
            eps = self.eps
            eri = self.eri_chemist

            omega_rpa, X_rpa, Y_rpa = CasidaSolver(
                *self.build_casida_matrices(nocc, lBSE=False)).solve()
            rpa_factor = 4.0
            occ, virt = self._get_occ_virt_indices(eps, nocc)
            n_pair = len(occ) * len(virt)
            XpY = (X_rpa + Y_rpa).reshape(n_pair, -1)
            del X_rpa, Y_rpa
            V_matrix = eri[np.ix_(occ, virt)].reshape(n_pair, -1)
            V_exciton = V_matrix.T @ XpY
            del V_matrix, XpY
            screened = rpa_factor * (V_exciton / omega_rpa[None, :]) @ V_exciton.T
            del V_exciton
            W_rpa = screened.reshape(eri.shape)
            np.subtract(eri, W_rpa, out=W_rpa)
            eri_w_singlet = W_rpa
            eri_w_triplet = W_rpa

        return eri_w_singlet, eri_w_triplet

    def static_screening_aux(self, nocc):
        """Static (omega=0) RPA inverse-dielectric metric W_aux = (1 - chi0)^-1 in the DF
        auxiliary basis: the full screened Coulomb is W_pqrs = B_pq . W_aux . B_rs with
        B = coeff_df. The omega=0 point of solve_rpa_screening on the imaginary axis."""
        return self.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    def _get_f_rpa(self, d, w, is_imaginary):
        return imaginary_frequency._f_rpa(self, d, w, is_imaginary)

    def build_casida_matrices(self, nocc, lBSE=False, W_aux=None, triplet=False,
                              eps_screen=None):
        """
        Builds the Casida matrices A and B.
        Dispatches to _build_block_df or _build_block_full.

        eps_screen: evaluate the BSE screening at these orbital energies instead
        of the ones on the diagonal. BSE@G0W0 needs exactly that split -- the
        diagonal carries the quasiparticle energies while W is screened at the
        mean-field ones. A three-index solver takes its whole kernel from W_aux
        and so gets the split by construction; the 4-center builder rebuilds the
        direct term itself and has to be told. Full ERIs, restricted only.
        """
        if eps_screen is not None:
            if self.spin_mode == 'unrestricted':
                raise NotImplementedError(
                    'eps_screen is restricted-only; the unrestricted 4-center '
                    'block screens at its own eps_a/eps_b.')
            if self.df_coeff is not None or self.coeff_ov is not None:
                raise ValueError(
                    'eps_screen does not apply to a three-index solver: its '
                    'kernel is screened entirely by W_aux, so build W_aux at '
                    'those energies instead.')
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            
            nocc_a_val, nvirt_a_val = len(occ_a), len(virt_a)
            nocc_b_val, nvirt_b_val = len(occ_b), len(virt_b)
            n_pair_a = nocc_a_val * nvirt_a_val
            n_pair_b = nocc_b_val * nvirt_b_val
            n_pair = n_pair_a + n_pair_b
            
            A = np.zeros((n_pair, n_pair))
            B = np.zeros((n_pair, n_pair))
            
            if self.coeff_a is not None:
                A_aa, B_aa = self._build_block_df(
                    self.eps_a, occ_a, virt_a, lBSE, W_aux, factor=1.0, 
                    coeff_all=self.coeff_a, spin_channel='a'
                )
                A_bb, B_bb = self._build_block_df(
                    self.eps_b, occ_b, virt_b, lBSE, W_aux, factor=1.0, 
                    coeff_all=self.coeff_b, spin_channel='b'
                )
                C_ov_a = self.coeff_a[:, occ_a[:, None], virt_a].reshape(self.naux, -1)
                C_ov_b = self.coeff_b[:, occ_b[:, None], virt_b].reshape(self.naux, -1)
                V_ab = C_ov_a.T @ C_ov_b
            else:
                A_aa, B_aa = self._build_block_full(
                    self.eps_a, occ_a, virt_a, lBSE, W_aux, factor=1.0, 
                    eri_all=self.eri_a, spin_channel='a', nocc=nocc
                )
                A_bb, B_bb = self._build_block_full(
                    self.eps_b, occ_b, virt_b, lBSE, W_aux, factor=1.0, 
                    eri_all=self.eri_b, spin_channel='b', nocc=nocc
                )
                V_ab = self.eri_ab[np.ix_(occ_a, virt_a, occ_b, virt_b)].reshape(n_pair_a, n_pair_b)
                
            A[:n_pair_a, :n_pair_a] = A_aa
            B[:n_pair_a, :n_pair_a] = B_aa
            A[n_pair_a:, n_pair_a:] = A_bb
            B[n_pair_a:, n_pair_a:] = B_bb
            
            A[:n_pair_a, n_pair_a:] = V_ab
            A[n_pair_a:, :n_pair_a] = V_ab.T
            B[:n_pair_a, n_pair_a:] = V_ab
            B[n_pair_a:, :n_pair_a] = V_ab.T
            
            return A, B
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            factor = 0.0 if triplet else 2.0
            if self.df_coeff is not None:
                A, B = self._build_block_df(
                    self.eps, occ, virt, lBSE, W_aux, factor=factor, 
                    coeff_all=self.df_coeff, spin_channel='restricted'
                )
            else:
                A, B = self._build_block_full(
                    self.eps, occ, virt, lBSE, W_aux, factor=factor, 
                    eri_all=self.eri_chemist, spin_channel='restricted', nocc=nocc,
                    eps_screen=eps_screen
                )
            return A, B

    def build_spin_flip_casida_matrices(self, nocc, lBSE=False, W_aux=None, channel='ab'):
        """
        Builds the spin-flip Casida matrices A and B for unrestricted calculations.
        channel='ab': excitation from occ_a to virt_b.
        channel='ba': excitation from occ_b to virt_a.
        Note that B is zero for spin-flip excitations, and A only contains the exchange term.
        """
        assert self.spin_mode == 'unrestricted', "Spin-flip Casida is only defined for unrestricted spin mode."
        nocc_a, nocc_b = nocc
        
        if channel == 'ab':
            occ = np.arange(nocc_a)
            virt = np.arange(nocc_b, self.norb_b)
            eps_occ = self.eps_a
            eps_virt = self.eps_b
            coeff_occ = self.coeff_a
            coeff_virt = self.coeff_b
            eri_channel = self.eri_ab
        else:
            occ = np.arange(nocc_b)
            virt = np.arange(nocc_a, self.norb_a)
            eps_occ = self.eps_b
            eps_virt = self.eps_a
            coeff_occ = self.coeff_b
            coeff_virt = self.coeff_a
            # For ba channel, we use transposition on the full ERI blocks
            eri_channel = self.eri_ab
            
        nocc_val = len(occ)
        nvirt_val = len(virt)
        n_pair = nocc_val * nvirt_val
        
        diag_d = (eps_virt[virt][None, :] - eps_occ[occ][:, None]).ravel()

        B = np.zeros((n_pair, n_pair))
        
        if coeff_occ is not None:
            C_oo_flat = coeff_occ[:, occ[:, None], occ].reshape(self.naux, -1)
            C_vv_flat = coeff_virt[:, virt[:, None], virt].reshape(self.naux, -1)
            
            if not lBSE or W_aux is None:
                V_exchange = (C_oo_flat.T @ C_vv_flat).reshape(nocc_val, nocc_val, nvirt_val, nvirt_val).transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
                A = np.diag(diag_d) - V_exchange
            else:
                tmp = W_aux @ C_vv_flat
                res = C_oo_flat.T @ tmp
                W_exchange = res.reshape(nocc_val, nocc_val, nvirt_val, nvirt_val).transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
                A = np.diag(diag_d) - W_exchange
                
            # Compute B for density fitting
            if self.df_ab is not None:
                if channel == 'ab':
                    # df_ab has indices (p, beta, alpha). So we need (p, virt_b, occ_a) -> (p, occ_a, virt_b)
                    C_ov_sf = self.df_ab[:, virt, :][:, :, occ].transpose(0, 2, 1)
                else:
                    # df_ab has indices (p, beta, alpha). So we need (p, occ_b, virt_a)
                    C_ov_sf = self.df_ab[:, occ, :][:, :, virt]
                
                # optimize= is NOT cosmetic here. Three operands without it run
                # numpy's naive kernel at naux^2 n_ov^2 instead of contracting
                # W_aux into one factor first, naux n_ov^2 + naux^2 n_ov -- 768x
                # at naux 300 / n_ov 600, and the ratio grows linearly in naux.
                if not lBSE or W_aux is None:
                    B = - np.einsum('pib, pja -> iajb', C_ov_sf, C_ov_sf,
                                    optimize=True).reshape(n_pair, n_pair)
                else:
                    B = - np.einsum('pib, pq, qja -> iajb', C_ov_sf, W_aux,
                                    C_ov_sf, optimize=True).reshape(n_pair, n_pair)
        else:
            if channel == 'ab':
                V_exch_raw = eri_channel[np.ix_(occ, occ, virt, virt)]
                V_b_raw = eri_channel[np.ix_(occ, virt, occ, virt)]
            else:
                V_exch_raw = eri_channel[np.ix_(virt, virt, occ, occ)].transpose(2, 3, 0, 1)
                V_b_raw = eri_channel[np.ix_(virt, occ, virt, occ)]
                
            V_exchange = V_exch_raw.reshape(nocc_val, nocc_val, nvirt_val, nvirt_val).transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
            
            if not lBSE or W_aux is None:
                A = np.diag(diag_d) - V_exchange
                if channel == 'ab':
                    B = -V_b_raw.transpose(0, 3, 2, 1).reshape(n_pair, n_pair)
                else:
                    B = -V_b_raw.transpose(1, 2, 3, 0).reshape(n_pair, n_pair)
            else:
                # W_aux is the 4D tensor (either self.eri_ab or a screened version of it)
                if channel == 'ab':
                    W_exch_raw = W_aux[np.ix_(occ, occ, virt, virt)]
                    W_b_raw = W_aux[np.ix_(occ, virt, occ, virt)]
                else:
                    W_exch_raw = W_aux[np.ix_(virt, virt, occ, occ)].transpose(2, 3, 0, 1)
                    W_b_raw = W_aux[np.ix_(virt, occ, virt, occ)]
                W_exchange = W_exch_raw.reshape(nocc_val, nocc_val, nvirt_val, nvirt_val).transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
                A = np.diag(diag_d) - W_exchange
                if channel == 'ab':
                    B = -W_b_raw.transpose(0, 3, 2, 1).reshape(n_pair, n_pair)
                else:
                    B = -W_b_raw.transpose(1, 2, 3, 0).reshape(n_pair, n_pair)
                
        return A, B


    def _build_block_df(self, eps, occ, virt, lBSE, W_aux, factor, coeff_all, spin_channel):
        nocc = len(occ)
        nvirt = len(virt)
        n_pair = nocc * nvirt
        naux = coeff_all.shape[0]

        diag_d = (eps[virt][None, :] - eps[occ][:, None]).ravel()

        # Optimize indexing with 3D advanced indexing
        C_ov_flat = coeff_all[:, occ[:, None], virt].reshape(naux, n_pair)
        V_iajb = gram_product(C_ov_flat)

        if not lBSE:
            # A = diag(d) + factor V and B = factor V, assembled in V's
            # buffer and one copy.
            V_iajb *= factor
            B = V_iajb
            A = B.copy()
            A.flat[::n_pair + 1] += diag_d
            return A, B

        if W_aux is not None:
            C_oo_flat = coeff_all[:, occ[:, None], occ].reshape(naux, nocc * nocc)
            C_vv_flat = coeff_all[:, virt[:, None], virt].reshape(naux, nvirt * nvirt)
            tmp_dir = W_aux @ C_vv_flat
            res_dir = C_oo_flat.T @ tmp_dir
            del tmp_dir, C_oo_flat, C_vv_flat
            W_direct_att = (
                res_dir.reshape(nocc, nocc, nvirt, nvirt)
                .transpose(0, 2, 1, 3)
                .reshape(n_pair, n_pair)
            )
            del res_dir
            tmp_swap = W_aux @ C_ov_flat
            res_swap = C_ov_flat.T @ tmp_swap
            del tmp_swap
            W_swap_att = (
                res_swap.reshape(nocc, nvirt, nocc, nvirt)
                .transpose(0, 3, 2, 1)
                .reshape(n_pair, n_pair)
            )
            del res_swap
        else:
            C_oo_flat = coeff_all[:, occ[:, None], occ].reshape(naux, nocc * nocc)
            C_vv_flat = coeff_all[:, virt[:, None], virt].reshape(naux, nvirt * nvirt)
            W_direct_att = (
                (C_oo_flat.T @ C_vv_flat)
                .reshape(nocc, nocc, nvirt, nvirt)
                .transpose(0, 2, 1, 3)
                .reshape(n_pair, n_pair)
            )
            del C_oo_flat, C_vv_flat
            W_swap_att = (
                V_iajb.reshape(nocc, nvirt, nocc, nvirt)
                .transpose(0, 3, 2, 1)
                .reshape(n_pair, n_pair)
            )
            if np.may_share_memory(W_swap_att, V_iajb):
                # nocc == 1 or nvirt == 1: the swap is a view of V_iajb, which is
                # scaled in place below.
                W_swap_att = W_swap_att.copy()

        return self._assemble_in_place(diag_d, V_iajb, factor, W_direct_att, W_swap_att,
                                       swap_shares_caller=False)

    @staticmethod
    def _assemble_in_place(diag_d, V_iajb, factor, W_direct_att, W_swap_att,
                            swap_shares_caller):
        """Assemble the Casida A, B blocks in place, reusing V/W buffers.

        Computes ``A = diag(diag_d) + factor * V_iajb - W_direct_att`` and
        ``B = factor * V_iajb - W_swap_att``, writing each result into an
        existing buffer instead of allocating a fresh ``(n_pair, n_pair)``
        array. `V_iajb` is scaled by `factor` in place; the diagonal of A
        keeps the evaluation order ``(diag_d + factor * V_iajb_ii) -
        W_direct_att_ii`` to match the off-diagonal rounding.

        Parameters
        ----------
        diag_d : ndarray, shape (n_pair,)
            Orbital-energy gaps eps_virt - eps_occ, one per (i, a) pair.
        V_iajb : ndarray, shape (n_pair, n_pair)
            Bare Coulomb block. Scaled by `factor` in place.
        factor : float
            Spin factor multiplying `V_iajb` (2.0 singlet, 0.0 triplet,
            1.0 unrestricted).
        W_direct_att : ndarray, shape (n_pair, n_pair)
            Direct screened-exchange block. Overwritten in place and
            reused as the buffer `A` is returned in.
        W_swap_att : ndarray, shape (n_pair, n_pair)
            Swap screened-exchange block. Overwritten in place and reused
            as the buffer `B` is returned in, unless `swap_shares_caller`
            is True.
        swap_shares_caller : bool
            True when `W_swap_att` is a view into an array the caller
            still owns (for example `W_aux`), so it must not be written
            to; `B` is then computed out of place instead.

        Returns
        -------
        A : ndarray, shape (n_pair, n_pair)
            Alias of `W_direct_att`, written in place.
        B : ndarray, shape (n_pair, n_pair)
            Alias of `W_swap_att` written in place, or a fresh array when
            `swap_shares_caller` is True.
        """
        n_pair = V_iajb.shape[0]
        W_dir_ii = W_direct_att.flat[::n_pair + 1].copy()
        V_iajb *= factor
        A_ii = (diag_d + V_iajb.flat[::n_pair + 1]) - W_dir_ii
        np.subtract(V_iajb, W_direct_att, out=W_direct_att)
        A = W_direct_att
        A.flat[::n_pair + 1] = A_ii
        if swap_shares_caller:
            B = V_iajb - W_swap_att
        else:
            np.subtract(V_iajb, W_swap_att, out=W_swap_att)
            B = W_swap_att
        return A, B

    def _build_block_full(self, eps, occ, virt, lBSE, W_aux, factor, eri_all, spin_channel, nocc=None,
                          eps_screen=None):
        nocc_val = len(occ)
        nvirt_val = len(virt)
        n_pair = nocc_val * nvirt_val
        
        diag_d = (eps[virt][None, :] - eps[occ][:, None]).ravel()

        # Optimize indexing with np.ix_
        V_iajb_4d = eri_all[np.ix_(occ, virt, occ, virt)]
        V_iajb = V_iajb_4d.reshape(n_pair, n_pair)

        if not lBSE:
            V_iajb *= factor
            B = V_iajb
            A = B.copy()
            A.flat[::n_pair + 1] += diag_d
            return A, B
        else:
            V_exch_raw = eri_all[np.ix_(occ, occ, virt, virt)]
            V_exchange = (
                V_exch_raw.reshape(nocc_val, nocc_val, nvirt_val, nvirt_val)
                .transpose(0, 2, 1, 3)
                .reshape(n_pair, n_pair)
            )
            del V_exch_raw
            if W_aux is not None:
                if self.spin_mode == 'unrestricted':
                    nocc_a, nocc_b = nocc
                    occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
                    occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
                    
                    n_pair_a = len(occ_a) * len(virt_a)
                    n_pair_b = len(occ_b) * len(virt_b)
                    nvirt_a = len(virt_a)
                    nvirt_b = len(virt_b)
                    n_pair_tot = n_pair_a + n_pair_b
                    
                    d_a = (self.eps_a[virt_a][None, :] - self.eps_a[occ_a][:, None]).ravel()
                    d_b = (self.eps_b[virt_b][None, :] - self.eps_b[occ_b][:, None]).ravel()
                    
                    V_aa = self.eri_a[np.ix_(occ_a, virt_a, occ_a, virt_a)].reshape(n_pair_a, n_pair_a)
                    V_bb = self.eri_b[np.ix_(occ_b, virt_b, occ_b, virt_b)].reshape(n_pair_b, n_pair_b)
                    V_ab = self.eri_ab[np.ix_(occ_a, virt_a, occ_b, virt_b)].reshape(n_pair_a, n_pair_b)
                    
                    V_trans = np.block([[V_aa, V_ab], [V_ab.T, V_bb]])
                    
                    f_a = self._get_f_rpa(d_a, 0.0, is_imaginary=False)
                    f_b = self._get_f_rpa(d_b, 0.0, is_imaginary=False)
                    chi0_trans = np.diag(np.concatenate([f_a, f_b]))
                    
                    chi_trans = chi0_trans @ np.linalg.inv(np.eye(n_pair_tot) - V_trans @ chi0_trans)
                    
                    if spin_channel == 'a':
                        V_aa_ijkc = self.eri_a[np.ix_(occ_a, occ_a, occ_a, virt_a)].reshape(nocc_a*nocc_a, n_pair_a)
                        V_ab_ijkc = self.eri_ab[np.ix_(occ_a, occ_a, occ_b, virt_b)].reshape(nocc_a*nocc_a, n_pair_b)
                        V_ijkc_trans = np.block([V_aa_ijkc, V_ab_ijkc])
                        
                        V_aa_abld = self.eri_a[np.ix_(virt_a, virt_a, occ_a, virt_a)].reshape(nvirt_a*nvirt_a, n_pair_a)
                        V_ab_abld = self.eri_ab[np.ix_(virt_a, virt_a, occ_b, virt_b)].reshape(nvirt_a*nvirt_a, n_pair_b)
                        V_abld_trans = np.block([V_aa_abld, V_ab_abld])
                    else:
                        V_ba_ijkc = self.eri_ab[np.ix_(occ_a, virt_a, occ_b, occ_b)].reshape(n_pair_a, nocc_b*nocc_b).T
                        V_bb_ijkc = self.eri_b[np.ix_(occ_b, occ_b, occ_b, virt_b)].reshape(nocc_b*nocc_b, n_pair_b)
                        V_ijkc_trans = np.block([V_ba_ijkc, V_bb_ijkc])
                        
                        V_ba_abld = self.eri_ab[np.ix_(occ_a, virt_a, virt_b, virt_b)].reshape(n_pair_a, nvirt_b*nvirt_b).T
                        V_bb_abld = self.eri_b[np.ix_(virt_b, virt_b, occ_b, virt_b)].reshape(nvirt_b*nvirt_b, n_pair_b)
                        V_abld_trans = np.block([V_ba_abld, V_bb_abld])
                        
                    tmp = V_ijkc_trans @ chi_trans
                    del chi0_trans, chi_trans
                    W_minus_V_direct_raw = tmp @ V_abld_trans.T
                    W_direct_att = V_exchange
                    W_direct_att += (
                        W_minus_V_direct_raw.reshape(
                            nocc_val, nocc_val, nvirt_val, nvirt_val)
                        .transpose(0, 2, 1, 3)
                        .reshape(n_pair, n_pair)
                    )
                    del W_minus_V_direct_raw, tmp
                    
                    if spin_channel == 'a':
                        W_swap_att_raw = W_aux[:n_pair, :n_pair]
                    else:
                        W_swap_att_raw = W_aux[n_pair:, n_pair:]
                else:
                    # diag_d sets the transition energies of the A matrix;
                    # the polarizability that screens the kernel need not use
                    # the same ones, so it gets its own gaps. Defaults to eps,
                    # which is the pre-existing behaviour.
                    eps_scr = eps if eps_screen is None else eps_screen
                    diag_scr = (eps_scr[virt][None, :]
                                - eps_scr[occ][:, None]).ravel()
                    rpa_factor = 2.0
                    f = self._get_f_rpa(diag_scr, 0.0, is_imaginary=False)
                    chi0 = np.diag(rpa_factor * f)
                    chi = chi0 @ np.linalg.inv(np.eye(n_pair) - V_iajb @ chi0)
                    
                    V_ijkc = eri_all[np.ix_(occ, occ, occ, virt)]
                    V_abld = eri_all[np.ix_(virt, virt, occ, virt)]
                    
                    tmp = V_ijkc.reshape(nocc_val*nocc_val, n_pair) @ chi
                    del chi0, chi
                    W_minus_V_direct_raw = tmp @ V_abld.reshape(nvirt_val*nvirt_val, n_pair).T
                    W_direct_att = V_exchange
                    W_direct_att += (
                        W_minus_V_direct_raw.reshape(
                            nocc_val, nocc_val, nvirt_val, nvirt_val)
                        .transpose(0, 2, 1, 3)
                        .reshape(n_pair, n_pair)
                    )
                    del W_minus_V_direct_raw, tmp
                    W_swap_att_raw = W_aux
                
                W_swap_att = W_swap_att_raw.reshape(nocc_val, nvirt_val, nocc_val, nvirt_val).transpose(0, 3, 2, 1).reshape(n_pair, n_pair)
                swap_shares_caller = np.may_share_memory(W_swap_att, W_aux)
            else:
                W_direct_att = V_exchange
                W_swap_att = V_iajb_4d.transpose(0, 3, 2, 1).reshape(n_pair, n_pair)
                if np.may_share_memory(W_swap_att, V_iajb):
                    W_swap_att = W_swap_att.copy()
                swap_shares_caller = False

            return self._assemble_in_place(
                diag_d, V_iajb, factor, W_direct_att, W_swap_att,
                swap_shares_caller)

    def solve_rpa_screening(self, omega_grid, nocc, is_imaginary=False):
        """W(omega) by direct particle-hole summation -- route 2 of three.

        Implementation lives in `imaginary_frequency.py`, which documents how
        this route relates to the Casida (route 1) and imaginary-time (route 3)
        constructions of the same object.
        """
        return imaginary_frequency.solve_rpa_screening(self, omega_grid, nocc, is_imaginary)

    def solve_rpa_screening_df(self, omega_grid, nocc, is_imaginary=False):
        return imaginary_frequency.solve_rpa_screening_df(self, omega_grid, nocc, is_imaginary)

    def solve_rpa_screening_full(self, omega_grid, nocc, is_imaginary=False):
        return imaginary_frequency.solve_rpa_screening_full(self, omega_grid, nocc, is_imaginary)

    def solve_rpa_spectral(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W (dispatches to DF or full ERI version)."""
        if (self.spin_mode == 'unrestricted' and self.coeff_a is not None) or (self.spin_mode != 'unrestricted' and (self.df_coeff is not None or self.coeff_ov is not None)):
            return self.solve_rpa_spectral_df(omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary)
        else:
            return self.solve_rpa_spectral_full(omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary)

    def solve_rpa_spectral_df(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W using density fitting."""
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            
            C_ov_a = self.coeff_a[:, occ_a[:, None], virt_a].reshape(self.naux, -1)
            C_ov_b = self.coeff_b[:, occ_b[:, None], virt_b].reshape(self.naux, -1)
            XplusY_proj = np.block([[C_ov_a, C_ov_b]]) @ X_plus_Y
            
            prefactor = 1.0
            W_grid = []
            for w in omega_grid:
                if is_imaginary:
                    denom = -2.0 * prefactor * eigenvalues_casida / (eigenvalues_casida**2 + w**2)
                else:
                    denom = prefactor * np.real(1.0 / (w - eigenvalues_casida + 1j * self.eta) - 1.0 / (w + eigenvalues_casida + 1j * self.eta))
                W_w = np.eye(self.naux) + (XplusY_proj * denom) @ XplusY_proj.T
                W_grid.append(W_w)
            return np.array(W_grid)
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            C_ov = self.df_coeff[:, occ[:, None], virt].reshape(self.naux, -1)
            XplusY_proj = C_ov @ X_plus_Y
            
            prefactor = 2.0
            W_grid = []
            for w in omega_grid:
                if is_imaginary:
                    denom = -2.0 * prefactor * eigenvalues_casida / (eigenvalues_casida**2 + w**2)
                else:
                    denom = prefactor * np.real(1.0 / (w - eigenvalues_casida + 1j * self.eta) - 1.0 / (w + eigenvalues_casida + 1j * self.eta))
                W_w = np.eye(self.naux) + (XplusY_proj * denom) @ XplusY_proj.T
                W_grid.append(W_w)
            return np.array(W_grid)

    def solve_rpa_spectral_full(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W using full ERIs."""
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            
            nocc_a_val, nvirt_a_val = len(occ_a), len(virt_a)
            nocc_b_val, nvirt_b_val = len(occ_b), len(virt_b)
            n_pair_a = nocc_a_val * nvirt_a_val
            n_pair_b = nocc_b_val * nvirt_b_val
            
            # Optimized indexing with np.ix_
            V_aa = self.eri_a[np.ix_(occ_a, virt_a, occ_a, virt_a)].reshape(n_pair_a, n_pair_a)
            V_bb = self.eri_b[np.ix_(occ_b, virt_b, occ_b, virt_b)].reshape(n_pair_b, n_pair_b)
            V_ab = self.eri_ab[np.ix_(occ_a, virt_a, occ_b, virt_b)].reshape(n_pair_a, n_pair_b)
            V_trans = np.block([[V_aa, V_ab], [V_ab.T, V_bb]])
            
            prefactor = 1.0
            W_grid = []
            for w in omega_grid:
                if is_imaginary:
                    denom = -2.0 * prefactor * eigenvalues_casida / (eigenvalues_casida**2 + w**2)
                else:
                    denom = prefactor * np.real(1.0 / (w - eigenvalues_casida + 1j * self.eta) - 1.0 / (w + eigenvalues_casida + 1j * self.eta))
                W_w = V_trans + V_trans @ (X_plus_Y * denom) @ X_plus_Y.T @ V_trans
                W_grid.append(W_w)
            return np.array(W_grid)
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            n_pair = len(occ) * len(virt)

            # Optimized indexing with np.ix_
            V_trans = self.eri_chemist[np.ix_(occ, virt, occ, virt)].reshape(n_pair, n_pair)

            prefactor = 2.0
            W_grid = []
            for w in omega_grid:
                if is_imaginary:
                    denom = -2.0 * prefactor * eigenvalues_casida / (eigenvalues_casida**2 + w**2)
                else:
                    denom = prefactor * np.real(1.0 / (w - eigenvalues_casida + 1j * self.eta) - 1.0 / (w + eigenvalues_casida + 1j * self.eta))
                W_w = V_trans + V_trans @ (X_plus_Y * denom) @ X_plus_Y.T @ V_trans
                W_grid.append(W_w)
            return np.array(W_grid)


def static_screened_coulomb_chemist(eps, eri_chemist, nocc, coeff_df=None):
    """Static (omega=0) RPA screened Coulomb W as a spatial-MO chemist (pq|rs) 4-index
    tensor, same layout as `eri_chemist`. Restricted/RHF.

    W = v - v.(X+Y)(1/Omega)(X+Y)^T.v summed over RPA excitations (construct_4d_w_rpa).
    coeff_df: optional DF factor (naux, norb, norb); when given the RPA Casida solve uses
    the DF path (the returned W is still a dense 4-index tensor)."""
    lr = LinearResponseSolver(eps, coeff_df=coeff_df, eri_chemist=eri_chemist,
                              spin_mode='restricted')
    W_singlet, _W_triplet = lr.construct_4d_w_rpa(nocc)
    return np.asarray(W_singlet)


def static_screened_coulomb_aux(eps, coeff_df, nocc):
    """Static (omega=0) RPA inverse-dielectric metric W_aux (naux, naux) from a DF factor
    -- the memory-lean counterpart to static_screened_coulomb_chemist, never forming the
    dense norb^4 W. Restricted/RHF. See LinearResponseSolver.static_screening_aux."""
    lr = LinearResponseSolver(eps, coeff_df=coeff_df, spin_mode='restricted')
    return np.asarray(lr.static_screening_aux(nocc))


def static_screened_coulomb_chemist_uhf(eps_a, eps_b, eri_a, eri_b, eri_ab, nocc_a, nocc_b):
    """UHF counterpart of static_screened_coulomb_chemist: (W_a, W_b), the
    static RPA-screened same-spin chemist tensors (aa|W|aa)/(bb|W|bb) -- same
    layout as eri_a/eri_b. Two calls to construct_4d_w_rpa (one per
    spin_channel), each rerunning the combined-spin Casida solve -- fine for
    the dense (small-system) route this feeds; use
    static_screened_coulomb_aux_uhf (one shared W_aux, no redundant Casida
    solve) on the DF path instead."""
    lr = LinearResponseSolver((eps_a, eps_b), eri_chemist=(eri_a, eri_b, eri_ab),
                              spin_mode='unrestricted')
    W_a, _ = lr.construct_4d_w_rpa((nocc_a, nocc_b), spin_channel='alpha')
    W_b, _ = lr.construct_4d_w_rpa((nocc_a, nocc_b), spin_channel='beta')
    return np.asarray(W_a), np.asarray(W_b)


def static_screened_coulomb_aux_uhf(eps_a, eps_b, coeff_a, coeff_b, nocc_a, nocc_b):
    """UHF counterpart of static_screened_coulomb_aux: static (omega=0) RPA
    inverse-dielectric metric W_aux (naux, naux) in the DF auxiliary basis.

    ONE shared W_aux serves both spin channels: chi0 = chi0_alpha + chi0_beta
    (solve_rpa_screening_df's unrestricted branch sums both spins' particle-hole
    bubbles into a single (naux,naux) polarizability before inverting) -- the
    RPA screening felt by a test charge is a property of the TOTAL density
    response, not spin-resolved. coeff_a/coeff_b: DF factors (naux, norb_a,
    norb_a)/(naux, norb_b, norb_b) in the SAME canonical (energy-ordered)
    basis as eps_a/eps_b."""
    lr = LinearResponseSolver((eps_a, eps_b), coeff_df=(coeff_a, coeff_b),
                              spin_mode='unrestricted')
    return np.asarray(lr.static_screening_aux((nocc_a, nocc_b)))


def check_normalization(x, y=None, tol=CASIDA_NORM_TOL):
    """Refuse Casida vectors that are not in this repo's <X|X> - <Y|Y> = 1.

    Returns (X, Y) as (n_ov, nroots) float arrays with Y materialized as zeros
    for a Tamm-Dancoff root. Rescaling silently instead would hide a factor of
    two in every oscillator strength, which is the single most likely way to
    get this wrong.
    """
    x = np.atleast_2d(np.asarray(x, float).T).T if np.ndim(x) == 1 else np.asarray(x, float)
    y = np.zeros_like(x) if y is None else np.asarray(y, float)
    if y.shape != x.shape:
        raise ValueError(f'X {x.shape} and Y {y.shape} disagree')
    norms = (x ** 2).sum(axis=0) - (y ** 2).sum(axis=0)
    bad = np.abs(norms - 1.0) > tol
    if bad.any():
        worst = norms[bad][np.argmax(np.abs(norms[bad] - 1.0))]
        hint = (" -- that is pySCF's convention; convert with from_pyscf()"
                if abs(worst - 0.5) < 0.05 else '')
        raise ValueError(f'root {int(np.flatnonzero(bad)[0])} has '
                         f'<X|X> - <Y|Y> = {worst:.6f}, not 1 to {tol:g}{hint}')
    return x, y


def from_pyscf(td):
    """(omega, X, Y) from a pySCF TDA/TDDFT object, in THIS repo's normalization.

    pySCF normalizes to <X|X> - <Y|Y> = 1/2, so every vector is scaled by
    sqrt(2). Used to cross-check the module against an independent solver; it
    is not part of any production path.
    """
    nocc = int(np.count_nonzero(td._scf.mo_occ > 0))
    nvir = np.asarray(td._scf.mo_coeff).shape[1] - nocc
    x = np.zeros((nocc * nvir, len(td.e)))
    y = np.zeros_like(x)
    for k, xy in enumerate(td.xy):
        x[:, k] = np.asarray(xy[0]).ravel() * np.sqrt(2.0)
        y[:, k] = np.asarray(xy[1]).ravel() * np.sqrt(2.0) \
            if np.ndim(xy[1]) else 0.0
    return np.asarray(td.e, float), x, y
