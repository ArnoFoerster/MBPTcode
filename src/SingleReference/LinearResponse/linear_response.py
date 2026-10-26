import warnings

import numpy as np
from src.Base.orbital_spaces import ov_energies, ov_factor
from src.Base.orbital_spaces import get_occ_virt_indices
from src.Base.numerics.constants import (BSE2_GPU_BLOCK_GB, CASIDA_NORM_TOL,
                                         DEFAULT_BROADENING_ETA)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.pyscf.pyscf_interface import occupied_count, spin_index
from src.SingleReference.LinearResponse import imaginary_frequency

try:
    import cupy as cp
    from cupy import cublas      # a submodule `import cupy` leaves unbound
except ImportError:              # optional: only layout.device='gpu' needs it
    cp = cublas = None

DEVICES = ('cpu', 'gpu')


def gram_product(C, block_elems=2**27):
    """V = C^T C for C of shape (naux, n), assembled in row blocks of at most
    block_elems values.

    Written as ``C.T @ C`` the product goes to the BLAS symmetric rank-k kernel,
    where the OpenBLAS bundled with the numpy wheel segfaults once n^2 reaches a
    few 10^9 values. A row block V[i0:i1] = C[:, i0:i1]^T C is a plain GEMM on
    distinct buffers, which holds at those sizes, straight into its slice of V.
    One block covering every row, n^2 <= block_elems, is ``C.T @ C`` itself and
    takes the rank-k kernel again: harmless at the default, far below the
    fault, which is why block_elems stays well under 10^9. With several blocks
    V is symmetric to round-off only; its consumers read one triangle.
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
        and coeff_df may be left None. Unrestricted: the (alpha, beta) pair.
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
            elif self.coeff_ov is not None:
                self.naux = self.coeff_ov[0].shape[0]
        else:
            self.eps = eps
            self.norb = len(self.eps)
            if self.df_coeff is not None:
                self.naux = self.df_coeff.shape[0]
            elif self.coeff_ov is not None:
                self.naux = self.coeff_ov.shape[0]

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)

    def ov_coulomb_unrestricted(self, occ_a, virt_a, occ_b, virt_b):
        """(ia|jb) from the full ERIs on the stacked alpha + beta pair index:
        [[V_aa, V_ab], [V_ab^T, V_bb]], the Coulomb matrix every unrestricted
        full-ERI screening builds."""
        n_pair_a = len(occ_a) * len(virt_a)
        n_pair_b = len(occ_b) * len(virt_b)
        V_aa = self.eri_a[np.ix_(occ_a, virt_a, occ_a, virt_a)].reshape(n_pair_a, n_pair_a)
        V_bb = self.eri_b[np.ix_(occ_b, virt_b, occ_b, virt_b)].reshape(n_pair_b, n_pair_b)
        V_ab = self.eri_ab[np.ix_(occ_a, virt_a, occ_b, virt_b)].reshape(n_pair_a, n_pair_b)
        return np.block([[V_aa, V_ab], [V_ab.T, V_bb]])

    def ov_factors_unrestricted(self, occ_a, virt_a, occ_b, virt_b):
        """(C_ov_a, C_ov_b): its DF twin, the occupied-virtual blocks of the
        two spin factors (Base.orbital_spaces.ov_factor)."""
        return (ov_factor(self.coeff_a, occ_a, virt_a),
                ov_factor(self.coeff_b, occ_b, virt_b))

    def construct_4d_w_rpa(self, nocc, spin_channel='alpha'):
        """
        Full-ERI (non-DF) counterpart to solve_rpa_screening: the static 4D
        W_rpa = v - 2g v (A + B)^-1 v, the
        vertex-correction screened interaction when df=False.
        """
        if self.spin_mode == 'unrestricted':
            eps = (self.eps_a, self.eps_b)
            eri = (self.eri_a, self.eri_b, self.eri_ab)

            A_rpa, B_rpa = self.build_casida_matrices(nocc, lBSE=False)
            R = np.linalg.inv(A_rpa + B_rpa)

            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(eps[0], nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(eps[1], nocc_b)
            n_pair_a = len(occ_a) * len(virt_a)
            n_pair_b = len(occ_b) * len(virt_b)

            V_aa_matrix = eri[0][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
            V_ba_matrix = eri[2].transpose(2, 3, 0, 1)[np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
            V_a = np.vstack([V_aa_matrix, V_ba_matrix])
            screened_a = 2.0 * V_a.T @ R @ V_a
            W_rpa_a = eri[0] - screened_a.reshape(eri[0].shape)

            V_ab_matrix = eri[2][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
            V_bb_matrix = eri[1][np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
            V_b = np.vstack([V_ab_matrix, V_bb_matrix])
            screened_b = 2.0 * V_b.T @ R @ V_b
            W_rpa_b = eri[1] - screened_b.reshape(eri[1].shape)

            eri_w_singlet = (W_rpa_a, W_rpa_b)[spin_index(spin_channel)]
            # The spin-flip vertex reads the (aa|bb) block of W.
            eri_w_triplet = eri[2] - (2.0 * V_a.T @ R @ V_b).reshape(eri[2].shape)
        else:
            eps = self.eps
            eri = self.eri_chemist

            A_rpa, B_rpa = self.build_casida_matrices(nocc, lBSE=False)
            occ, virt = self._get_occ_virt_indices(eps, nocc)
            n_pair = len(occ) * len(virt)
            V_matrix = eri[np.ix_(occ, virt)].reshape(n_pair, -1)
            screened = 4.0 * V_matrix.T @ np.linalg.inv(A_rpa + B_rpa) @ V_matrix
            W_rpa = eri - screened.reshape(eri.shape)
            eri_w_singlet = W_rpa
            eri_w_triplet = W_rpa

        return eri_w_singlet, eri_w_triplet

    def static_screening_aux(self, nocc):
        """Static (omega=0) RPA inverse-dielectric metric W_aux = (1 - chi0)^-1 in the DF
        auxiliary basis: the full screened Coulomb is W_pqrs = B_pq . W_aux . B_rs with
        B = coeff_df. The omega=0 point of solve_rpa_screening on the imaginary axis."""
        return self.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

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
                C_ov_a, C_ov_b = self.ov_factors_unrestricted(occ_a, virt_a,
                                                              occ_b, virt_b)
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

    def build_spin_flip_casida_matrices(self, nocc, lBSE=False, W_aux=None, channel=None):
        """Spin-flip Casida matrices (A, B) of an unrestricted reference.

        The pair space is the ab pairs (occ_a -> virt_b) followed by the ba pairs
        (occ_b -> virt_a). A is block diagonal, orbital energy differences minus
        the exchange-type (i j|a b); B couples an ab excitation with a ba
        de-excitation, B[(i a), (j b)] = -(i b|j a), and nothing else, so the
        solution holds the M_s = -1 excitations (X on ab, Y on ba) and the
        M_s = +1 ones (X on ba, Y on ab). Under BSE (lBSE with W_aux) the
        exchange-type integrals are screened: W_aux is the DF metric, or the
        four-index (aa|bb) W without DF.

        channel='ab' or 'ba' returns that diagonal block with B as one square
        matrix, X and Y on one grid; the two pair grids coincide only for
        nocc_a == nocc_b, so it is refused otherwise.
        """
        assert self.spin_mode == 'unrestricted', "Spin-flip Casida is only defined for unrestricted spin mode."
        nocc_a, nocc_b = nocc
        occ_a, virt_a = np.arange(nocc_a), np.arange(nocc_a, self.norb_a)
        occ_b, virt_b = np.arange(nocc_b), np.arange(nocc_b, self.norb_b)
        screened = lBSE and W_aux is not None

        def integral(ia, ib, ja, jb):
            """(ia ib|ja jb): first pair of spin alpha, second of spin beta, W under BSE."""
            if self.coeff_a is not None:
                left = self.coeff_a[:, ia[:, None], ib].reshape(self.naux, -1)
                right = self.coeff_b[:, ja[:, None], jb].reshape(self.naux, -1)
                if screened:
                    right = W_aux @ right
                return (left.T @ right).reshape(len(ia), len(ib), len(ja), len(jb))
            eri = W_aux if screened else self.eri_ab
            return eri[np.ix_(ia, ib, ja, jb)]

        n_ab = nocc_a * len(virt_b)
        n_ba = nocc_b * len(virt_a)
        d_ab = (self.eps_b[virt_b][None, :] - self.eps_a[occ_a][:, None]).ravel()
        d_ba = (self.eps_a[virt_a][None, :] - self.eps_b[occ_b][:, None]).ravel()
        # (i j|a b) with i, j alpha occupied and a, b beta virtual, and its mirror
        A_ab = np.diag(d_ab) - integral(occ_a, occ_a, virt_b, virt_b).transpose(0, 2, 1, 3).reshape(n_ab, n_ab)
        A_ba = np.diag(d_ba) - integral(virt_a, virt_a, occ_b, occ_b).transpose(2, 0, 3, 1).reshape(n_ba, n_ba)
        # B[(i_a a_b), (j_b b_a)] = -(i_a b_a|j_b a_b)
        B_x = -integral(occ_a, virt_a, occ_b, virt_b).transpose(0, 3, 2, 1).reshape(n_ab, n_ba)

        if channel is not None:
            if nocc_a != nocc_b:
                raise ValueError("a single spin-flip channel puts X and Y on one pair grid, "
                                 "which holds only for nocc_a == nocc_b; build both "
                                 "channels together (channel=None)")
            return (A_ab, B_x) if channel == 'ab' else (A_ba, B_x.T)
        A = np.zeros((n_ab + n_ba, n_ab + n_ba))
        B = np.zeros_like(A)
        A[:n_ab, :n_ab] = A_ab
        A[n_ab:, n_ab:] = A_ba
        B[:n_ab, n_ab:] = B_x
        B[n_ab:, :n_ab] = B_x.T
        return A, B

    def _build_block_df(self, eps, occ, virt, lBSE, W_aux, factor, coeff_all, spin_channel):
        nocc = len(occ)
        nvirt = len(virt)
        n_pair = nocc * nvirt
        naux = coeff_all.shape[0]

        diag_d = ov_energies(eps, len(occ))

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
        
        diag_d = ov_energies(eps, len(occ))

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
                    
                    d_a = ov_energies(self.eps_a, nocc_a)
                    d_b = ov_energies(self.eps_b, nocc_b)
                    
                    V_trans = self.ov_coulomb_unrestricted(occ_a, virt_a, occ_b, virt_b)
                    
                    chi_trans = -2.0 * np.linalg.inv(
                        np.diag(np.concatenate([d_a, d_b])) + 2.0 * V_trans)
                    
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
                    del chi_trans
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
                    # the same ones, so it gets its own gaps. Defaults to eps.
                    eps_scr = eps if eps_screen is None else eps_screen
                    diag_scr = (eps_scr[virt][None, :]
                                - eps_scr[occ][:, None]).ravel()
                    chi = -4.0 * np.linalg.inv(np.diag(diag_scr)
                                                       + 4.0 * V_iajb)
                    
                    V_ijkc = eri_all[np.ix_(occ, occ, occ, virt)]
                    V_abld = eri_all[np.ix_(virt, virt, occ, virt)]
                    
                    tmp = V_ijkc.reshape(nocc_val*nocc_val, n_pair) @ chi
                    del chi
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

    def solve_rpa_spectral(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W (dispatches to DF or full ERI version)."""
        if (self.spin_mode == 'unrestricted' and self.coeff_a is not None) or (self.spin_mode != 'unrestricted' and (self.df_coeff is not None or self.coeff_ov is not None)):
            return self.solve_rpa_spectral_df(omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary)
        else:
            return self.solve_rpa_spectral_full(omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary)

    def _spectral_grid(self, omega_grid, eigenvalues_casida, prefactor,
                       is_imaginary, build):
        """[build(denom(w)) for w in omega_grid]: the spectral W's frequency
        loop, denom(w) = prefactor sum over the Casida poles +-Omega of
        1/(w -+ Omega), broadened by eta on the real axis."""
        W_grid = []
        for w in omega_grid:
            if is_imaginary:
                denom = -2.0 * prefactor * eigenvalues_casida / (eigenvalues_casida**2 + w**2)
            else:
                denom = prefactor * np.real(1.0 / (w - eigenvalues_casida + 1j * self.eta) - 1.0 / (w + eigenvalues_casida + 1j * self.eta))
            W_grid.append(build(denom))
        return np.array(W_grid)

    def solve_rpa_spectral_df(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W using density fitting."""
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            C_ov_a, C_ov_b = self.ov_factors_unrestricted(occ_a, virt_a, occ_b, virt_b)
            XplusY_proj = np.block([[C_ov_a, C_ov_b]]) @ X_plus_Y
            prefactor = 1.0
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            C_ov = self.df_coeff[:, occ[:, None], virt].reshape(self.naux, -1)
            XplusY_proj = C_ov @ X_plus_Y
            prefactor = 2.0
        return self._spectral_grid(
            omega_grid, eigenvalues_casida, prefactor, is_imaginary,
            lambda denom: np.eye(self.naux) + (XplusY_proj * denom) @ XplusY_proj.T)

    def solve_rpa_spectral_full(self, omega_grid, nocc, eigenvalues_casida, X_plus_Y, is_imaginary=False):
        """Spectral representation of screened potential W using full ERIs."""
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            V_trans = self.ov_coulomb_unrestricted(occ_a, virt_a, occ_b, virt_b)
            prefactor = 1.0
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            n_pair = len(occ) * len(virt)
            V_trans = self.eri_chemist[np.ix_(occ, virt, occ, virt)].reshape(n_pair, n_pair)
            prefactor = 2.0
        return self._spectral_grid(
            omega_grid, eigenvalues_casida, prefactor, is_imaginary,
            lambda denom: V_trans + V_trans @ (X_plus_Y * denom) @ X_plus_Y.T @ V_trans)


def static_screened_coulomb_chemist(eps, eri_chemist, nocc, coeff_df=None):
    """Static (omega=0) RPA screened Coulomb W as a spatial-MO chemist (pq|rs) 4-index
    tensor, same layout as `eri_chemist`. Restricted/RHF.

    W = v - 4 v (A + B)^-1 v (construct_4d_w_rpa).
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


def static_second_order_kernel_df(eps, coeff_df, W_aux, nocc, eta=0.0, blksize=8,
                                  channel='singlet', kernel='printed', numerics=None):
    """Static second-order GW kernel Θ^GW (BSE2@GW) as Casida A and B blocks.

    The two blocks to add to the BSE@GW Casida matrices of
    `LinearResponseSolver.build_casida_matrices(lBSE=True, W_aux=W_aux)` for one
    spin manifold, chosen by `channel`; the reference adds the singlet blocks to
    the singlet only. Restricted/RHF, DF factors only, never a norb^4 tensor.

    Parameters
    ----------
    eps : ndarray, shape (norb,)
        Orbital energies ε_p of the four denominators: the GW ones in the
        reference, the mean-field ones W was screened at in QuAcK's G0W0.
    coeff_df : ndarray, shape (naux, norb, norb), index order (P, p, q)
        DF factor B_P,pq with sum_P B_P,pr B_P,qs = (pr|qs).
    W_aux : ndarray, shape (naux, naux)
        Static screened Coulomb metric from `static_screened_coulomb_aux`, so
        W_pq,rs = sum_PQ B_P,pq W_PQ B_Q,rs is the static W in chemist order,
        bare term included.
    nocc : int
    eta : float
        Regularisation of every denominator, 1/x -> x / (x^2 + eta^2).
    blksize : int
        Virtual orbitals c per block of the particle-particle ladder and of the
        crossed class, whose (a, c, b, d) and (a, k, b, c) intermediates are held
        for one block only. On the GPU also the virtuals a per block of the
        hole-hole ladder of Θ^B, and every block capped by BSE2_GPU_BLOCK_GB.
    channel : {'singlet', 'triplet'}
        Spin adaptation of the M_S = 0 block: the six terms of the Notes for the
        singlet, their particle-hole terms alone for the triplet, which the
        crossed class of kernel='eq71' leaves empty.
    kernel : {'printed', 'eq71'}
        The class the particle-hole term carries: 'printed' the loop class of
        the reference's eq 72 as printed, 'eq71' the crossed class its eq 71
        holds (the Notes).
    numerics : Numerics or None
        The run's settings (None: the defaults); its layout.device 'gpu' forms
        the blocks on this process's GPU (CuPy, FP64) from the host's Cholesky
        factor of W_aux, `_second_order_kernel_gpu`, and returns them as host
        arrays, equal to the CPU path's to rounding. A device whose free memory
        cannot hold two (n_pair, n_pair) arrays and the DF factors raises
        ValueError before any upload.

    Returns
    -------
    theta_A : ndarray, shape (n_pair, n_pair), index order (ia, jb)
    theta_B : ndarray, shape (n_pair, n_pair), index order (ia, jb)

    Notes
    -----
    With d_kc = ε_c - ε_k, s_kl = ε_k + ε_l and s_cd = ε_c + ε_d,

        Θ^A_ia,jb = 4 sum_kc W_ij,kc W_ab,kc / d_kc
                  + 2 sum_kl W_ak,jl W_ki,lb / s_kl
                  - 2 sum_cd W_ac,jd W_ci,db / s_cd
        Θ^B_ia,jb = 4 sum_kc W_ib,kc W_aj,kc / d_kc
                  + 2 sum_kl W_ak,bl W_ki,lj / s_kl
                  - 2 sum_cd W_ac,bd W_ci,dj / s_cd

    is eq 72 of the reference in spatial orbitals, its two particle-hole terms
    merged (they coincide for real orbitals) and with the overall factor 2 of
    QuAcK's RGW_phBSE2_static_kernel_A/B. W factorises as
    W_pq,rs = sum_P D_P,pq D_P,rs with D_P,pq = sum_Q L_QP B_Q,pq and
    W_aux = L L^T, so the particle-hole terms are the direct and swap blocks
    of the BSE with the metric Π_PQ = sum_kc D_P,kc D_Q,kc / d_kc in place of
    W_aux, and the ladders are DF contractions blocked over c.

    In spin orbitals the particle-hole terms tie the spin of pair ia to that of
    pair jb (W_ij,kc needs spin(i) = spin(j)) and the ladders do not (W_ak,jl
    fixes spin(k) = spin(a) = spin(i) and spin(l) = spin(j) = spin(b)). In the
    M_S = 0 block, S = Θ_αα,αα + Θ_αα,ββ and T = Θ_αα,αα − Θ_αα,ββ, so the ladders
    enter the singlet twice and the triplet not at all, and the particle-hole
    terms, with their internal spin loop, enter both alike:

        Θ^S = the six terms above,    Θ^T = their particle-hole terms alone.

    The reference's text calls the triplet unaffected (its Sec. III D); its
    eq 72 in spin orbitals gives the triplet the particle-hole terms.

    Eq 71 of the reference, Θ = i G G (W W + W W) untruncated, holds the crossed
    class of the second-order block in place of the loop class the printed eq 72
    carries. Its static form by the same recipe (drop ω and every external orbital
    energy, keep the internal pair's) is, with kernel='eq71',

        Θ^A_ia,jb = 2 sum_kc (W_ik,jc W_ka,bc + W_jk,ic W_kb,ac) / d_kc + ladders
        Θ^B_ia,jb = 2 sum_kc (W_ik,bc W_ka,jc + W_jk,ac W_kb,ic) / d_kc + ladders

    the ladders those above and the B block by the letter swap j ↔ b that takes
    the printed A blocks to their B blocks, an assumption the reference does not
    settle. W_ik,jc ties spin(k) to
    spin(i) and spin(c) to spin(j), so the crossed class enters the singlet twice
    (the factor 2) and the triplet not at all: Θ^T = 0 with kernel='eq71'.

    References
    ----------
    E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 71-73.
    """
    if channel not in ('singlet', 'triplet'):
        raise ValueError(f"channel must be 'singlet' or 'triplet', got {channel!r}")
    if kernel not in ('printed', 'eq71'):
        raise ValueError(f"kernel must be 'printed' or 'eq71', got {kernel!r}")
    device = numerics_or_default(numerics).layout.device
    if device not in DEVICES:
        raise ValueError(f'numerics.layout.device={device!r}; expected one of '
                         f'{DEVICES}')
    occ, virt = get_occ_virt_indices(eps, nocc)
    no, nv = len(occ), len(virt)
    n_pair = no * nv
    naux = coeff_df.shape[0]

    def reg(x):
        return x / (x * x + eta * eta)

    if eps[virt].min() <= 0.0 or eps[occ].max() >= 0.0:
        warnings.warn('a ladder denominator of the second-order kernel changes sign '
                      f'(lowest virtual {eps[virt].min():.4f} Ha, highest occupied '
                      f'{eps[occ].max():.4f} Ha): eq 72 measures energies from the '
                      'chemical potential, so shift eps or set eta', stacklevel=2)
    try:
        L = np.linalg.cholesky(0.5 * (W_aux + W_aux.T))
    except np.linalg.LinAlgError:
        raise ValueError('W_aux is not positive definite: the static RPA '
                         'screening of an unstable reference.')
    if device == 'gpu':
        # the same Θ^A_ia,jb, Θ^B_ia,jb, with D_P,pq = sum_Q L_QP B_Q,pq on the device
        return _second_order_kernel_gpu(eps, coeff_df, L, occ, virt, reg, blksize,
                                        channel, kernel)
    # D_P,pq = sum_Q L_QP B_Q,pq on each orbital block
    D_oo = L.T @ coeff_df[:, occ[:, None], occ].reshape(naux, no * no)
    D_ov = L.T @ coeff_df[:, occ[:, None], virt].reshape(naux, n_pair)
    D_vv = L.T @ coeff_df[:, virt[:, None], virt].reshape(naux, nv * nv)
    D_vo = D_ov.reshape(naux, no, nv).transpose(0, 2, 1).reshape(naux, nv * no)
    del L
    if kernel == 'eq71' and channel == 'triplet':
        return np.zeros((n_pair, n_pair)), np.zeros((n_pair, n_pair))

    d_kc = (eps[virt][None, :] - eps[occ][:, None]).ravel()
    if kernel == 'printed':
        # Particle-hole terms: Pi_PQ = sum_kc D_P,kc D_Q,kc / d_kc
        Pi = (D_ov * reg(d_kc)) @ D_ov.T
        # Θ^A_ia,jb = 4 sum_PQ D_P,ij Pi_PQ D_Q,ab
        theta_A = 4.0 * (D_oo.T @ Pi @ D_vv).reshape(no, no, nv, nv) \
            .transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
        # Θ^B_ia,jb = 4 sum_PQ D_P,ib Pi_PQ D_Q,ja
        theta_B = 4.0 * (D_ov.T @ Pi @ D_ov).reshape(no, nv, no, nv) \
            .transpose(0, 3, 2, 1).reshape(n_pair, n_pair)
        del Pi
        if channel == 'triplet':
            return theta_A, theta_B
    else:
        # Crossed class, blocked over c; the (ia <-> jb) partner of each sum is
        # its transpose
        r_kc = reg(d_kc).reshape(no, nv)
        D_ov3 = D_ov.reshape(naux, no, nv)
        D_vv3 = D_vv.reshape(naux, nv, nv)
        M = np.zeros((n_pair, n_pair))
        N = np.zeros((n_pair, n_pair))
        for c0 in range(0, nv, blksize):
            cs = slice(c0, min(c0 + blksize, nv))
            nc = cs.stop - cs.start
            r = r_kc[:, cs]
            D_oc = D_ov3[:, :, cs].reshape(naux, no * nc)
            D_vc = D_vv3[:, :, cs].reshape(naux, nv * nc)
            # X_ik,jc = W_ik,jc = sum_P D_P,ik D_P,jc
            # Y_ak,bc = W_ak,bc = sum_P D_P,ak D_P,bc
            X = (D_oo.T @ D_oc).reshape(no, no, no, nc)
            Y = (D_vo.T @ D_vc).reshape(nv, no, nv, nc)
            # M_ia,jb += sum_kc X_ik,jc Y_ak,bc / d_kc
            U = X.transpose(0, 2, 1, 3).reshape(no * no, no * nc)
            V = (Y.transpose(1, 3, 0, 2) * r[:, :, None, None]) \
                .reshape(no * nc, nv * nv)
            M += (U @ V).reshape(no, no, nv, nv) \
                .transpose(0, 2, 1, 3).reshape(n_pair, n_pair)
            # X2_ik,bc = W_ik,bc = sum_P D_P,ik D_P,bc
            # Y2_ak,jc = W_ak,jc = sum_P D_P,ak D_P,jc
            X2 = (D_oo.T @ D_vc).reshape(no, no, nv, nc)
            Y2 = (D_vo.T @ D_oc).reshape(nv, no, no, nc)
            # N_ia,jb += sum_kc X2_ik,bc Y2_ak,jc / d_kc
            U = X2.transpose(0, 2, 1, 3).reshape(no * nv, no * nc)
            V = (Y2.transpose(1, 3, 0, 2) * r[:, :, None, None]) \
                .reshape(no * nc, nv * no)
            N += (U @ V).reshape(no, nv, nv, no) \
                .transpose(0, 2, 3, 1).reshape(n_pair, n_pair)
            del X, Y, X2, Y2, U, V
        # Θ^A_ia,jb = 2 (M_ia,jb + M_jb,ia),  Θ^B_ia,jb = 2 (N_ia,jb + N_jb,ia)
        theta_A = 2.0 * (M + M.T)
        theta_B = 2.0 * (N + N.T)
        del M, N

    # Hole-hole ladder, s_kl = ε_k + ε_l
    s_kl = reg(eps[occ][:, None] + eps[occ][None, :])
    # T_ak,jl = W_ak,jl = sum_P D_P,ak D_P,jl
    T = (D_vo.T @ D_oo).reshape(nv, no, no, no)
    # Θ^A_ia,jb += 2 sum_kl T_ak,jl T_bl,ki / s_kl   (W_ki,lb = T_bl,ki)
    U = T.transpose(0, 2, 1, 3).reshape(nv * no, no * no)
    V = (T.transpose(2, 1, 0, 3) * s_kl[:, :, None, None]).reshape(no * no, nv * no)
    theta_A += 2.0 * (U @ V).reshape(nv, no, nv, no) \
        .transpose(3, 0, 1, 2).reshape(n_pair, n_pair)
    del T, U, V
    # T2_ak,bl = W_ak,bl = sum_P D_P,ak D_P,bl
    # T3_ki,lj = W_ki,lj = sum_P D_P,ki D_P,lj
    T2 = (D_vo.T @ D_vo).reshape(nv, no, nv, no)
    T3 = (D_oo.T @ D_oo).reshape(no, no, no, no)
    # Θ^B_ia,jb += 2 sum_kl T2_ak,bl T3_ki,lj / s_kl
    U = T2.transpose(0, 2, 1, 3).reshape(nv * nv, no * no)
    V = (T3.transpose(0, 2, 1, 3) * s_kl[:, :, None, None]).reshape(no * no, no * no)
    theta_B += 2.0 * (U @ V).reshape(nv, nv, no, no) \
        .transpose(2, 0, 3, 1).reshape(n_pair, n_pair)
    del T2, T3, U, V

    # Particle-particle ladder, s_cd = ε_c + ε_d, blocked over c
    eps_v = eps[virt]
    D_vv3 = D_vv.reshape(naux, nv, nv)
    D_vo3 = D_vo.reshape(naux, nv, no)
    for c0 in range(0, nv, blksize):
        cs = slice(c0, min(c0 + blksize, nv))
        nc = cs.stop - cs.start
        s_cd = reg(eps_v[cs][:, None] + eps_v[None, :])
        D_ac = D_vv3[:, :, cs].reshape(naux, nv * nc)
        D_ci = D_vo3[:, cs, :].reshape(naux, nc * no)
        # X_ac,jd = W_ac,jd = sum_P D_P,ac D_P,jd
        # Y_ci,db = W_ci,db = sum_P D_P,ci D_P,db
        X = (D_ac.T @ D_ov).reshape(nv, nc, no, nv)
        Y = (D_ci.T @ D_vv).reshape(nc, no, nv, nv)
        # Θ^A_ia,jb -= 2 sum_cd X_ac,jd Y_ci,db / s_cd
        U = X.transpose(0, 2, 1, 3).reshape(nv * no, nc * nv)
        V = (Y.transpose(0, 2, 1, 3) * s_cd[:, :, None, None]).reshape(nc * nv, no * nv)
        theta_A -= 2.0 * (U @ V).reshape(nv, no, no, nv) \
            .transpose(2, 0, 1, 3).reshape(n_pair, n_pair)
        del X, Y, U, V
        # X2_ac,bd = W_ac,bd = sum_P D_P,ac D_P,bd
        # Y2_ci,dj = W_ci,dj = sum_P D_P,ci D_P,dj
        X2 = (D_ac.T @ D_vv).reshape(nv, nc, nv, nv)
        Y2 = (D_ci.T @ D_vo).reshape(nc, no, nv, no)
        # Θ^B_ia,jb -= 2 sum_cd X2_ac,bd Y2_ci,dj / s_cd
        U = X2.transpose(0, 2, 1, 3).reshape(nv * nv, nc * nv)
        V = (Y2.transpose(0, 2, 1, 3) * s_cd[:, :, None, None]).reshape(nc * nv, -1)
        theta_B -= 2.0 * (U @ V).reshape(nv, nv, no, no) \
            .transpose(2, 0, 3, 1).reshape(n_pair, n_pair)
        del X2, Y2, U, V
    return theta_A, theta_B


def device_memory_check(need, free, label):
    """Refuse a device allocation of `need` bytes that `free` bytes cannot hold.

    Parameters
    ----------
    need, free : int
        The bytes the caller is about to allocate, and the bytes the device has free.
    label : str
        What `need` holds, named first in the error.

    Raises
    ------
    ValueError
        When `need` exceeds `free`, naming both in GiB.
    """
    if need > free:
        raise ValueError(f'{label} need {need / 1024 ** 3:.3g} GiB, the device has '
                         f'{free / 1024 ** 3:.3g} GiB free')


def _second_order_kernel_gpu(eps, coeff_df, L, occ, virt, reg, blksize, channel,
                             kernel):
    """Θ^A, Θ^B of static_second_order_kernel_df on this process's GPU (CuPy, FP64).

    Parameters
    ----------
    eps : ndarray, shape (norb,)
    coeff_df : ndarray, shape (naux, norb, norb), index order (P, p, q)
    L : ndarray, shape (naux, naux)
        The host's lower Cholesky factor, W_aux_PQ = sum_R L_PR L_QR.
    occ, virt : ndarray of int, shapes (no,) and (nv,)
        Occupied and virtual orbital indices.
    reg : callable
        The caller's regularised reciprocal, 1/x -> x / (x^2 + eta^2).
    blksize, channel, kernel
        As static_second_order_kernel_df.

    Returns
    -------
    theta_A, theta_B : ndarray, shape (n_pair, n_pair), index order (ia, jb)
        Host arrays.

    Notes
    -----
    The terms of the CPU path's Notes in its GEMMs, with d_kc = ε_c - ε_k,
    s_kl = ε_k + ε_l, s_cd = ε_c + ε_d and every reciprocal taken by `reg`. Θ^A
    is built first, then Θ^B: the ladder part of each accumulates in place
    (GEMMs with beta = 1) in the GEMM's own index order, G_aj,ib for Θ^A_ia,jb
    and G_ab,ij for Θ^B_ia,jb, and is permuted into (ia, jb) once; then the
    particle-hole or crossed term is added and the block copied to the host.
    Each holds two pair-sized arrays on the device where the CPU path's
    per-block permutations hold four; W_ak,bl of the hole-hole ladder of Θ^B,
    pair-sized too, is formed over blocks of a. The sums run in another order,
    so the two paths agree to rounding, not to the bit.
    """
    if cp is None:
        raise ImportError("layout.device='gpu' needs CuPy, which is not "
                          'importable')
    no, nv = len(occ), len(virt)
    n_pair = no * nv
    if kernel == 'eq71' and channel == 'triplet':
        return np.zeros((n_pair, n_pair)), np.zeros((n_pair, n_pair))
    naux = coeff_df.shape[0]
    printed, singlet = kernel == 'printed', channel == 'singlet'
    # the two pair-sized arrays and D_P,pq over the oo, ov, vo and vv blocks
    # against the device's free memory, the bytes the pool caches but holds no
    # array in counted as free; the blocks' intermediates, capped by
    # BSE2_GPU_BLOCK_GB, come on top
    need = 2 * 8 * n_pair ** 2 + 8 * naux * (no + nv) ** 2
    free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
    device_memory_check(need, free, 'the pair-sized arrays and the DF factors')

    def blocks(per_virtual):
        # slices over the virtuals, blksize wide, capped where per_virtual
        # values for each reach BSE2_GPU_BLOCK_GB
        cap = BSE2_GPU_BLOCK_GB * 1024 ** 3 // (8 * per_virtual)
        nb = int(max(1, min(blksize, cap)))
        return [slice(p0, min(p0 + nb, nv)) for p0 in range(0, nv, nb)]

    def gemm(a, b, out, alpha):
        # out_pq += alpha sum_r a_pr b_rq, in place
        cublas.gemm('N', 'N', a, b, out=out, alpha=alpha, beta=1.0)

    Lg = cp.asarray(L)

    def factor(rows, cols):
        # D_P,pq = sum_Q L_QP B_Q,pq
        return Lg.T @ cp.asarray(coeff_df[:, rows[:, None], cols].reshape(naux, -1))

    D_oo, D_ov, D_vv = factor(occ, occ), factor(occ, virt), factor(virt, virt)
    del Lg
    D_vo = D_ov.reshape(naux, no, nv).transpose(0, 2, 1).reshape(naux, nv * no)
    D_ov3, D_vv3 = D_ov.reshape(naux, no, nv), D_vv.reshape(naux, nv, nv)
    D_vo3 = D_vo.reshape(naux, nv, no)
    eps_o, eps_v = eps[occ], eps[virt]
    r_kc = reg(eps_v[None, :] - eps_o[:, None])
    s_kl = cp.asarray(reg(eps_o[:, None] + eps_o[None, :]))
    s_cd = reg(eps_v[:, None] + eps_v[None, :])
    if printed:
        # Pi_PQ = sum_kc D_P,kc D_Q,kc / d_kc
        Pi = (D_ov * cp.asarray(r_kc.ravel())) @ D_ov.T

    # Θ^A
    if singlet:
        # G_aj,ib = 2 sum_kl T_ak,jl T_bl,ki / s_kl - 2 sum_cd X_ac,jd Y_ci,db / s_cd
        G = cp.zeros((nv * no, no * nv))
        T = (D_vo.T @ D_oo).reshape(nv, no, no, no)
        # U = T_ak,jl as (a j, k l), V = T_bl,ki / s_kl as (k l, i b)
        U = T.transpose(0, 2, 1, 3).reshape(nv * no, no * no)
        V = (T.transpose(2, 1, 3, 0) * s_kl[:, :, None, None]).reshape(no * no, -1)
        del T
        gemm(U, V, G, 2.0)
        del U, V
        for cs in blocks(4 * nv * no * nv + naux * (nv + no)):
            nc = cs.stop - cs.start
            D_ac = D_vv3[:, :, cs].reshape(naux, nv * nc)
            D_ci = D_vo3[:, cs, :].reshape(naux, nc * no)
            # U = X_ac,jd = sum_P D_P,ac D_P,jd as (a j, c d)
            U = (D_ac.T @ D_ov).reshape(nv, nc, no, nv).transpose(0, 2, 1, 3) \
                .reshape(nv * no, nc * nv)
            # V = Y_ci,db / s_cd = sum_P D_P,ci D_P,db / s_cd as (c d, i b)
            s = cp.asarray(s_cd[cs])
            V = ((D_ci.T @ D_vv).reshape(nc, no, nv, nv).transpose(0, 2, 1, 3)
                 * s[:, :, None, None]).reshape(nc * nv, no * nv)
            gemm(U, V, G, -2.0)
            del D_ac, D_ci, U, V
        # Θ^A_ia,jb = G_aj,ib
        F = G.reshape(nv, no, no, nv).transpose(2, 0, 1, 3).reshape(n_pair, n_pair)
        del G
    else:
        F = cp.zeros((n_pair, n_pair))
    F4 = F.reshape(no, nv, no, nv)
    if printed:
        # P_ij,ab = 4 sum_PQ D_P,ij Pi_PQ D_Q,ab;  Θ^A_ia,jb += P_ij,ab
        P = (4.0 * (D_oo.T @ Pi)) @ D_vv
        F4 += P.reshape(no, no, nv, nv).transpose(0, 2, 1, 3)
        del P
    else:
        # M_ij,ab = 2 sum_kc X_ik,jc Y_ak,bc / d_kc;  Θ^A_ia,jb += M_ij,ab + M_ji,ba
        M = cp.zeros((no * no, nv * nv))
        for cs in blocks(2 * no ** 3 + 3 * nv * no * nv + naux * (no + nv)):
            nc = cs.stop - cs.start
            r = cp.asarray(r_kc[:, cs])
            D_oc = D_ov3[:, :, cs].reshape(naux, no * nc)
            D_vc = D_vv3[:, :, cs].reshape(naux, nv * nc)
            # U = X_ik,jc = sum_P D_P,ik D_P,jc as (i j, k c)
            U = (D_oo.T @ D_oc).reshape(no, no, no, nc).transpose(0, 2, 1, 3) \
                .reshape(no * no, no * nc)
            # V = Y_ak,bc / d_kc = sum_P D_P,ak D_P,bc / d_kc as (k c, a b)
            V = ((D_vo.T @ D_vc).reshape(nv, no, nv, nc).transpose(1, 3, 0, 2)
                 * r[:, :, None, None]).reshape(no * nc, nv * nv)
            gemm(U, V, M, 2.0)
            del D_oc, D_vc, U, V
        M4 = M.reshape(no, no, nv, nv)
        F4 += M4.transpose(0, 2, 1, 3)
        F4 += M4.transpose(1, 3, 0, 2)
        del M, M4
    theta_A = cp.asnumpy(F)
    del F, F4

    # Θ^B
    if singlet:
        # G_ab,ij = 2 sum_kl T2_ak,bl T3_ki,lj / s_kl
        #         - 2 sum_cd X2_ac,bd Y2_ci,dj / s_cd
        G = cp.zeros((nv * nv, no * no))
        # V = T3_ki,lj / s_kl = sum_P D_P,ki D_P,lj / s_kl as (k l, i j)
        V = ((D_oo.T @ D_oo).reshape(no, no, no, no).transpose(0, 2, 1, 3)
             * s_kl[:, :, None, None]).reshape(no * no, no * no)
        for a_blk in blocks(3 * no * nv * no + naux * no):
            a0, a1 = a_blk.start, a_blk.stop
            # U = T2_ak,bl = sum_P D_P,ak D_P,bl as (a b, k l), a in the block
            U = (D_vo[:, a0 * no:a1 * no].T @ D_vo).reshape(a1 - a0, no, nv, no) \
                .transpose(0, 2, 1, 3).reshape((a1 - a0) * nv, no * no)
            gemm(U, V, G[a0 * nv:a1 * nv], 2.0)
            del U
        del V
        for cs in blocks(2 * nv ** 3 + 3 * no * no * nv + naux * (nv + no)):
            nc = cs.stop - cs.start
            D_ac = D_vv3[:, :, cs].reshape(naux, nv * nc)
            D_ci = D_vo3[:, cs, :].reshape(naux, nc * no)
            # U = X2_ac,bd = sum_P D_P,ac D_P,bd as (a b, c d)
            U = (D_ac.T @ D_vv).reshape(nv, nc, nv, nv).transpose(0, 2, 1, 3) \
                .reshape(nv * nv, nc * nv)
            # V = Y2_ci,dj / s_cd = sum_P D_P,ci D_P,dj / s_cd as (c d, i j)
            s = cp.asarray(s_cd[cs])
            V = ((D_ci.T @ D_vo).reshape(nc, no, nv, no).transpose(0, 2, 1, 3)
                 * s[:, :, None, None]).reshape(nc * nv, no * no)
            gemm(U, V, G, -2.0)
            del D_ac, D_ci, U, V
        # Θ^B_ia,jb = G_ab,ij
        F = G.reshape(nv, nv, no, no).transpose(2, 0, 3, 1).reshape(n_pair, n_pair)
        del G
    else:
        F = cp.zeros((n_pair, n_pair))
    F4 = F.reshape(no, nv, no, nv)
    if printed:
        # P_ib,ja = 4 sum_PQ D_P,ib Pi_PQ D_Q,ja;  Θ^B_ia,jb += P_ib,ja
        P = (4.0 * (D_ov.T @ Pi)) @ D_ov
        F4 += P.reshape(no, nv, no, nv).transpose(0, 3, 2, 1)
        del P
    else:
        # N_ib,aj = 2 sum_kc X2_ik,bc Y2_ak,jc / d_kc;  Θ^B_ia,jb += N_ib,aj + N_ja,bi
        N = cp.zeros((no * nv, nv * no))
        for cs in blocks(5 * no * no * nv + naux * (no + nv)):
            nc = cs.stop - cs.start
            r = cp.asarray(r_kc[:, cs])
            D_oc = D_ov3[:, :, cs].reshape(naux, no * nc)
            D_vc = D_vv3[:, :, cs].reshape(naux, nv * nc)
            # U = X2_ik,bc = sum_P D_P,ik D_P,bc as (i b, k c)
            U = (D_oo.T @ D_vc).reshape(no, no, nv, nc).transpose(0, 2, 1, 3) \
                .reshape(no * nv, no * nc)
            # V = Y2_ak,jc / d_kc = sum_P D_P,ak D_P,jc / d_kc as (k c, a j)
            V = ((D_vo.T @ D_oc).reshape(nv, no, no, nc).transpose(1, 3, 0, 2)
                 * r[:, :, None, None]).reshape(no * nc, nv * no)
            gemm(U, V, N, 2.0)
            del D_oc, D_vc, U, V
        N4 = N.reshape(no, nv, nv, no)
        F4 += N4.transpose(0, 2, 3, 1)
        F4 += N4.transpose(3, 1, 0, 2)
        del N, N4
    theta_B = cp.asnumpy(F)
    return theta_A, theta_B


def static_screened_coulomb_chemist_uhf(eps_a, eps_b, eri_a, eri_b, eri_ab, nocc_a, nocc_b):
    """UHF counterpart of static_screened_coulomb_chemist: (W_a, W_b), the
    static RPA-screened same-spin chemist tensors (aa|W|aa)/(bb|W|bb) -- same
    layout as eri_a/eri_b. Two calls to construct_4d_w_rpa (one per
    spin_channel), each rebuilding the combined-spin (A + B)^-1 -- fine for
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
    nocc = occupied_count(td._scf)
    nvir = np.asarray(td._scf.mo_coeff).shape[1] - nocc
    x = np.zeros((nocc * nvir, len(td.e)))
    y = np.zeros_like(x)
    for k, xy in enumerate(td.xy):
        x[:, k] = np.asarray(xy[0]).ravel() * np.sqrt(2.0)
        y[:, k] = np.asarray(xy[1]).ravel() * np.sqrt(2.0) \
            if np.ndim(xy[1]) else 0.0
    return np.asarray(td.e, float), x, y
