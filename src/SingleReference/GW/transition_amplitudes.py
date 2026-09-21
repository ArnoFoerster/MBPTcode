import numpy as np

class AmplitudeGenerator:
    """
    Generates transition amplitudes (GW chi_a and vertex chiXYb, PSD amplitudes)
    for single-reference many-body perturbation theory.
    """

    def _blocked_contract(self, term1_flat, integs1_flat, term2_flat=None, integs2_flat=None):
        """Chunked term1_flat[:, s:e].T @ integs1_flat (+ term2_flat[:, s:e].T @ integs2_flat), over self.block_size."""
        nexciton = term1_flat.shape[1]
        out_dim = integs1_flat.shape[1]
        result = np.zeros((nexciton, out_dim))
        for start in range(0, nexciton, self.block_size):
            end = min(start + self.block_size, nexciton)
            chunk = term1_flat[:, start:end].T @ integs1_flat
            if term2_flat is not None:
                chunk = chunk + term2_flat[:, start:end].T @ integs2_flat
            result[start:end, :] = chunk
        return result

    def _amp_cache(self, key, build):
        """Memoizes a p_state-independent intermediate (e.g. the DF transition density,
        or W-dressed occ/virt blocks) across repeated calls with the same X/Y/eri_w --
        the pattern hit when a caller loops get_chi_a/get_chi_b_vertex over many
        p_state values for the same Casida solve (see qp_energy.py's `for p_state in
        states`). Cache lives on `self`, so it is scoped to one SelfEnergySolver
        instance/one calc_qp_energy call and never leaks across molecules."""
        cache = self.__dict__.setdefault('_transition_amp_cache', {})
        if key not in cache:
            cache[key] = build()
        return cache[key]

    @staticmethod
    def _identity_key(arr):
        """id() is only safe to key on while `arr` stays alive (true here: X/Y/eri_w
        are held by the caller for the whole p_state loop) -- shape is included as a
        cheap extra guard against an unrelated same-address object."""
        if arr is None:
            return ('none',)
        return (id(arr), arr.shape)

    def preset_transition_density(self, nocc, X, Y, rho):
        """Store `rho`, shape (naux, nexciton), as the DF transition density of
        this X/Y, so get_chi_a contracts it with this solver's own DF factors.

        The qsGW0 amplitudes: the Casida problem was solved in the mean field's
        orbitals, and rho_PS = sum_ia B_P,ia (X + Y)_ia,S there is an
        auxiliary-space object. The amplitudes of the rotated orbitals are
        rho^T B' with the rotated factors this solver holds, provided B' shares
        B's auxiliary basis: a with_df's, or B rotated, never factors
        re-decomposed from the rotated ERI. Restricted only;
        keyed exactly as _rho_a_df keys its own build, so X and Y must stay
        alive while the solver is used.
        """
        if self.spin_mode != 'restricted':
            raise NotImplementedError(
                'preset_transition_density is restricted-spin only')
        key = ('rho_a_r', self._identity_key(X), self._identity_key(Y), nocc)
        cache = self.__dict__.setdefault('_transition_amp_cache', {})
        cache[key] = np.asarray(rho, float)

    def get_chi_a(self, nocc, X, Y, spin_channel='alpha', p_state=None):
        """Computes the GW transition amplitude (dispatches to DF or full ERI version)."""
        if (self.spin_mode == 'unrestricted' and self.df_a is not None) or (self.spin_mode != 'unrestricted' and self.df_coeff is not None):
            return self.get_chi_a_df(nocc, X, Y, spin_channel, p_state)
        else:
            return self.get_chi_a_full(nocc, X, Y, spin_channel, p_state)

    def get_chi_b_vertex(self, nocc, X, Y, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the vertex correction transition amplitude chiXYb (dispatches to DF or full ERI version)."""
        if (self.spin_mode == 'unrestricted' and self.df_a is not None) or (self.spin_mode != 'unrestricted' and self.df_coeff is not None):
            return self.get_chi_b_vertex_df(nocc, X, Y, spin_channel, eri_w, p_state)
        else:
            return self.get_chi_b_vertex_full(nocc, X, Y, spin_channel, eri_w, p_state)

    def get_chi_b_psd(self, nocc, eigenvalues_casida, chiXYa, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the PSD transition amplitude chiXYb (dispatches to DF or full ERI version)."""
        if (self.spin_mode == 'unrestricted' and self.df_a is not None) or (self.spin_mode != 'unrestricted' and self.df_coeff is not None):
            return self.get_chi_b_psd_df(nocc, eigenvalues_casida, chiXYa, spin_channel, eri_w, p_state)
        else:
            return self.get_chi_b_psd_full(nocc, eigenvalues_casida, chiXYa, spin_channel, eri_w, p_state)

    def _rho_a_df(self, nocc, X, Y):
        """(naux, nexciton) DF transition density C_ov^T.(X+Y) -- independent of
        spin_channel and p_state, so cached and shared across every get_chi_a_df
        call for this X/Y (see _amp_cache)."""
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            key = ('rho_a_u', self._identity_key(X), self._identity_key(Y), nocc_a, nocc_b)

            def build():
                XpY = X + Y
                occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
                occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
                n_pair_a = len(occ_a) * len(virt_a)
                XpY_a = XpY[:n_pair_a, :]
                XpY_b = XpY[n_pair_a:, :]
                C_ov_a = self.df_a[:, occ_a[:, None], virt_a].reshape(self.naux, -1)
                C_ov_b = self.df_b[:, occ_b[:, None], virt_b].reshape(self.naux, -1)
                return C_ov_a @ XpY_a + C_ov_b @ XpY_b
            return self._amp_cache(key, build)
        else:
            key = ('rho_a_r', self._identity_key(X), self._identity_key(Y), nocc)

            def build():
                occ, virt = self._get_occ_virt_indices(self.eps, nocc)
                n_pair = len(occ) * len(virt)
                UpY_flat = (X + Y).reshape(len(occ), len(virt), -1).reshape(n_pair, -1)
                C_ov_flat = self.df_coeff[:, occ[:, None], virt].reshape(self.naux, -1)
                return C_ov_flat @ UpY_flat
            return self._amp_cache(key, build)

    def get_chi_a_df(self, nocc, X, Y, spin_channel='alpha', p_state=None):
        """Computes the GW transition amplitude using density fitting."""
        rho = self._rho_a_df(nocc, X, Y)
        nexciton = rho.shape[1]
        if self.spin_mode == 'unrestricted':
            coeff_target = self.df_a if spin_channel == 'alpha' else self.df_b
        else:
            coeff_target = self.df_coeff

        if p_state is not None:
            # coeff_target[:, :, p_state] fixes the last axis of a C-contiguous
            # 3D array, producing a non-contiguous view; matmul-ing against that
            # on every loop iteration is 5-10x slower than against a contiguous
            # copy (and barely benefits from extra BLAS threads) -- copy once here.
            coeff_p = np.ascontiguousarray(coeff_target[:, :, p_state])
            chi = np.zeros((nexciton, coeff_target.shape[2]))
            for start in range(0, nexciton, self.block_size):
                end = min(start + self.block_size, nexciton)
                chi[start:end, :] = rho[:, start:end].T @ coeff_p
        else:
            chi = (rho.T @ coeff_target.reshape(self.naux, -1)).reshape(nexciton, coeff_target.shape[1], coeff_target.shape[2])
        return chi

    def get_chi_a_full(self, nocc, X, Y, spin_channel='alpha', p_state=None):
        """Computes the GW transition amplitude using full ERIs."""
        XpY = X + Y
        if self.spin_mode == 'unrestricted':
            nocc_a, nocc_b = nocc
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            occ_b, virt_b = self._get_occ_virt_indices(self.eps_b, nocc_b)
            n_pair_a = len(occ_a) * len(virt_a)
            n_pair_b = len(occ_b) * len(virt_b)
            
            XpY_a = XpY[:n_pair_a, :]
            XpY_b = XpY[n_pair_a:, :]
            
            if spin_channel == 'alpha':
                norb_target = self.norb_a
                if p_state is not None:
                    eri_a_slice = self.eri_a[:, :, p_state, :][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
                    eri_ab_slice = self.eri_ab[p_state, :, :, :][:, occ_b[:, None], virt_b].reshape(norb_target, n_pair_b).T
                    
                    chi = self._blocked_contract(XpY_a, eri_a_slice, XpY_b, eri_ab_slice)
                else:
                    eri_a_flat = self.eri_a[np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
                    eri_ab_flat = self.eri_ab[:, :, occ_b[:, None], virt_b].reshape(norb_target*norb_target, n_pair_b).T
                    chi = (XpY_a.T @ eri_a_flat + XpY_b.T @ eri_ab_flat).reshape(-1, norb_target, norb_target)
                return chi
            else:
                norb_target = self.norb_b
                if p_state is not None:
                    eri_ab_slice = self.eri_ab[:, :, p_state, :][np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
                    eri_b_slice = self.eri_b[:, :, p_state, :][np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
                    
                    chi = self._blocked_contract(XpY_a, eri_ab_slice, XpY_b, eri_b_slice)
                else:
                    eri_ab_flat = self.eri_ab[np.ix_(occ_a, virt_a)].reshape(n_pair_a, -1)
                    eri_b_flat = self.eri_b[np.ix_(occ_b, virt_b)].reshape(n_pair_b, -1)
                    chi = (XpY_a.T @ eri_ab_flat + XpY_b.T @ eri_b_flat).reshape(-1, norb_target, norb_target)
                return chi
        else:
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            n_pair = len(occ) * len(virt)
            UpY_3d = XpY.reshape(len(occ), len(virt), -1)
            UpY_flat = UpY_3d.reshape(n_pair, -1)
            
            if p_state is not None:
                eri_slice = self.eri_chemist[:, :, :, p_state][np.ix_(occ, virt)].reshape(n_pair, -1)
                nexciton = UpY_flat.shape[1]
                chi = np.zeros((nexciton, self.eri_chemist.shape[1]))
                for start in range(0, nexciton, self.block_size):
                    end = min(start + self.block_size, nexciton)
                    chi[start:end, :] = UpY_flat[:, start:end].T @ eri_slice
            else:
                eri_flat = self.eri_chemist[np.ix_(occ, virt)].reshape(n_pair, -1)
                chi = (UpY_flat.T @ eri_flat).reshape(UpY_flat.shape[1], self.eri_chemist.shape[1], self.eri_chemist.shape[2])
            return chi

    def get_chi_b_vertex_df(self, nocc, X, Y, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the vertex correction transition amplitude chiXYb using density fitting."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            coeff = self.df_a if spin_channel == 'alpha' else self.df_b
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
            occ, virt = self._get_occ_virt_indices(eps, nocc_spin)
            
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            n_pair_a = len(occ_a) * len(virt_a)
            
            if spin_channel == 'alpha':
                X_spin = X[:n_pair_a, :]
                Y_spin = Y[:n_pair_a, :]
            else:
                X_spin = X[n_pair_a:, :]
                Y_spin = Y[n_pair_a:, :]
                
            X_3d = X_spin.reshape(len(occ), len(virt), -1)
            Y_3d = Y_spin.reshape(len(occ), len(virt), -1)
        else:
            eps = self.eps
            coeff = self.df_coeff
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            X_3d = X.reshape(len(occ), len(virt), -1)
            Y_3d = Y.reshape(len(occ), len(virt), -1)
            
        norb = len(eps)
        nexciton = X_3d.shape[2]
        X_flat = X_3d.reshape(-1, nexciton)
        Y_flat = Y_3d.reshape(-1, nexciton)
        
        W = np.eye(self.naux) if eri_w is None else np.asarray(eri_w)
        w_key = self._identity_key(eri_w) if eri_w is not None else ('eye', self.naux)

        if p_state is not None:
            # Sliced occupied block
            C_vo = coeff[:, virt[:, None], occ]
            C_op_slice = coeff[:, occ, p_state]
            C_vp_slice = coeff[:, virt, p_state]
            C_oo = coeff[:, occ[:, None], occ]

            tmp_1 = W @ C_op_slice
            C_vo_flat = C_vo.reshape(self.naux, -1)
            res_1 = C_vo_flat.T @ tmp_1
            integs1_iak = res_1.reshape(len(virt), len(occ), len(occ)).transpose(2, 0, 1)

            # tmp_2/tmp_1_v below don't depend on p_state at all (only on nocc,
            # spin_channel, eri_w) -- cached so a caller looping get_chi_b_vertex_df
            # over many p_state values (qp_energy.py's `for p_state in states`) pays
            # for these naux x naux-scale contractions once, not once per state.
            tmp_2 = self._amp_cache(('chi_b_tmp2', nocc, spin_channel, w_key),
                                     lambda: W @ C_oo.reshape(self.naux, -1))
            res_2 = C_vp_slice.T @ tmp_2
            integs2_iak = res_2.reshape(len(virt), len(occ), len(occ)).transpose(1, 0, 2)

            integs1_flat_o = integs1_iak.reshape(len(occ) * len(virt), len(occ))
            integs2_flat_o = integs2_iak.reshape(len(occ) * len(virt), len(occ))

            chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o)

            # Sliced virtual block
            C_oc = coeff[:, occ[:, None], virt]
            C_vv = coeff[:, virt[:, None], virt]

            tmp_1_v = self._amp_cache(('chi_b_tmp1v', nocc, spin_channel, w_key),
                                       lambda: W @ C_oc.reshape(self.naux, -1))
            res_1_v = C_vp_slice.T @ tmp_1_v
            integs1_iac = res_1_v.reshape(len(virt), len(occ), len(virt)).transpose(1, 0, 2)

            tmp_2_v = tmp_1  # same expression (W @ C_op_slice) as tmp_1 above
            C_vv_flat = C_vv.reshape(self.naux, -1)
            res_2_v = C_vv_flat.T @ tmp_2_v
            integs2_iac = res_2_v.reshape(len(virt), len(virt), len(occ)).transpose(2, 0, 1)

            integs1_flat_v = integs1_iac.reshape(len(occ) * len(virt), len(virt))
            integs2_flat_v = integs2_iac.reshape(len(occ) * len(virt), len(virt))

            chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v)

            chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
            if self.spin_mode == 'unrestricted':
                return 2.0 * chiXYb
            return chiXYb
        else:
            # Full occupied block
            C_vo = coeff[:, virt[:, None], occ]
            C_op = coeff[:, occ, :]
            C_vp = coeff[:, virt, :]
            C_oo = coeff[:, occ[:, None], occ]
            
            tmp_1 = W @ C_op.reshape(self.naux, -1)
            C_vo_flat = C_vo.reshape(self.naux, -1)
            res_1 = C_vo_flat.T @ tmp_1
            integs1_iakp = res_1.reshape(len(virt), len(occ), len(occ), norb).transpose(2, 0, 1, 3)
            
            tmp_2 = W @ C_oo.reshape(self.naux, -1)
            C_vp_flat = C_vp.reshape(self.naux, -1)
            res_2 = C_vp_flat.T @ tmp_2
            integs2_iakp = res_2.reshape(len(virt), norb, len(occ), len(occ)).transpose(2, 0, 3, 1)
            
            integs1_flat_o = integs1_iakp.reshape(len(occ) * len(virt), len(occ) * norb)
            integs2_flat_o = integs2_iakp.reshape(len(occ) * len(virt), len(occ) * norb)
            
            chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o).reshape(nexciton, len(occ), norb)

            # Full virtual block
            C_oc = coeff[:, occ[:, None], virt]
            C_vv = coeff[:, virt[:, None], virt]

            tmp_1_v = W @ C_oc.reshape(self.naux, -1)
            C_vp_flat = C_vp.reshape(self.naux, -1)
            res_1_v = C_vp_flat.T @ tmp_1_v
            integs1_iacp = res_1_v.reshape(len(virt), norb, len(occ), len(virt)).transpose(2, 0, 3, 1)

            tmp_2_v = tmp_1  # same expression (W @ C_op.reshape(naux, -1)) as tmp_1 above
            C_vv_flat = C_vv.reshape(self.naux, -1)
            res_2_v = C_vv_flat.T @ tmp_2_v
            integs2_iacp = res_2_v.reshape(len(virt), len(virt), len(occ), norb).transpose(2, 0, 1, 3)

            integs1_flat_v = integs1_iacp.reshape(len(occ) * len(virt), len(virt) * norb)
            integs2_flat_v = integs2_iacp.reshape(len(occ) * len(virt), len(virt) * norb)

            chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v).reshape(nexciton, len(virt), norb)

            chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
            if self.spin_mode == 'unrestricted':
                return 2.0 * chiXYb
            return chiXYb

    def get_chi_b_vertex_full(self, nocc, X, Y, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the vertex correction transition amplitude chiXYb using full ERIs."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            eri = self.eri_a if spin_channel == 'alpha' else self.eri_b
            if eri_w is None:
                eri_w = eri
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
            occ, virt = self._get_occ_virt_indices(eps, nocc_spin)
            
            occ_a, virt_a = self._get_occ_virt_indices(self.eps_a, nocc_a)
            n_pair_a = len(occ_a) * len(virt_a)
            
            if spin_channel == 'alpha':
                X_spin = X[:n_pair_a, :]
                Y_spin = Y[:n_pair_a, :]
            else:
                X_spin = X[n_pair_a:, :]
                Y_spin = Y[n_pair_a:, :]
                
            X_3d = X_spin.reshape(len(occ), len(virt), -1)
            Y_3d = Y_spin.reshape(len(occ), len(virt), -1)
        else:
            eps = self.eps
            eri = self.eri_chemist
            if eri_w is None:
                eri_w = eri
            occ, virt = self._get_occ_virt_indices(self.eps, nocc)
            X_3d = X.reshape(len(occ), len(virt), -1)
            Y_3d = Y.reshape(len(occ), len(virt), -1)
            
        norb = len(eps)
        nexciton = X_3d.shape[2]
        X_flat = X_3d.reshape(-1, nexciton)
        Y_flat = Y_3d.reshape(-1, nexciton)
        
        if p_state is not None:
            # Sliced occupied block
            integs1_iak = eri_w[:, :, :, p_state][np.ix_(virt, occ, occ)].transpose(2, 0, 1)
            integs2_iak = eri_w[:, p_state, :, :][np.ix_(virt, occ, occ)].transpose(1, 0, 2)
            
            integs1_flat_o = integs1_iak.reshape(len(occ) * len(virt), len(occ))
            integs2_flat_o = integs2_iak.reshape(len(occ) * len(virt), len(occ))
            
            chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o)

            # Sliced virtual block
            integs1_iac = eri_w[:, p_state, :, :][np.ix_(virt, occ, virt)].transpose(1, 0, 2)
            integs2_iac = eri_w[:, :, :, p_state][np.ix_(virt, virt, occ)].transpose(2, 0, 1)

            integs1_flat_v = integs1_iac.reshape(len(occ) * len(virt), len(virt))
            integs2_flat_v = integs2_iac.reshape(len(occ) * len(virt), len(virt))

            chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v)

            chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
            if self.spin_mode == 'unrestricted':
                return 2.0 * chiXYb
            return chiXYb
        else:
            # Full occupied block
            integs1_iakp = eri_w[np.ix_(virt, occ, occ)].transpose(2, 0, 1, 3)
            integs2_iakp = eri_w[np.ix_(virt, np.arange(norb), occ, occ)].transpose(2, 0, 3, 1)

            integs1_flat_o = integs1_iakp.reshape(len(occ) * len(virt), len(occ) * norb)
            integs2_flat_o = integs2_iakp.reshape(len(occ) * len(virt), len(occ) * norb)

            chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o).reshape(nexciton, len(occ), norb)

            # Full virtual block
            integs1_iacp = eri_w[np.ix_(virt, np.arange(norb), occ, virt)].transpose(2, 0, 3, 1)
            integs2_iacp = eri_w[np.ix_(virt, virt, occ)].transpose(2, 0, 1, 3)

            integs1_flat_v = integs1_iacp.reshape(len(occ) * len(virt), len(virt) * norb)
            integs2_flat_v = integs2_iacp.reshape(len(occ) * len(virt), len(virt) * norb)

            chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v).reshape(nexciton, len(virt), norb)

            chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
            if self.spin_mode == 'unrestricted':
                return 2.0 * chiXYb
            return chiXYb

    def get_chi_b_psd_df(self, nocc, eigenvalues_casida, chiXYa, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the PSD transition amplitude chiXYb using density fitting."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            coeff = self.df_a if spin_channel == 'alpha' else self.df_b
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
        else:
            eps = self.eps
            coeff = self.df_coeff
            nocc_spin = nocc
            
        occ, virt = self._get_occ_virt_indices(eps, nocc_spin)
        norb = len(eps)
        nexciton = len(eigenvalues_casida)
        
        chiXYa_ov = chiXYa[:, occ, :][:, :, virt]
        
        # Vectorized denominators
        d_ia = eps[virt] - eps[occ][:, None]
        denom_minus = 1.0 / (d_ia[None, :, :] - eigenvalues_casida[:, None, None])
        denom_plus = 1.0 / (d_ia[None, :, :] + eigenvalues_casida[:, None, None])
        
        chiXY_minus = chiXYa_ov * denom_minus
        chiXY_plus = chiXYa_ov * denom_plus
        
        eri_pajk = np.einsum('Ppa, Pjk -> pajk', coeff[:, :, virt], coeff[:, occ[:, None], occ])
        wt_pk = np.einsum('pajk, Sja -> pkS', eri_pajk, chiXY_plus)
        
        eri_pjak = np.einsum('Ppj, Pak -> pjak', coeff[:, :, occ], coeff[:, virt[:, None], occ])
        wt_pk += np.einsum('pjak, Sja -> pkS', eri_pjak, chiXY_minus)
        
        eri_pajc = np.einsum('Ppa, Pjc -> pajc', coeff[:, :, virt], coeff[:, occ[:, None], virt])
        wt_pc = np.einsum('pajc, Sja -> pcS', eri_pajc, chiXY_minus)
        
        eri_pjac = np.einsum('Ppj, Pac -> pjac', coeff[:, :, occ], coeff[:, virt[:, None], virt])
        wt_pc += np.einsum('pjac, Sja -> pcS', eri_pjac, chiXY_plus)
        
        chiXYb = np.zeros((nexciton, norb, norb))
        for S in range(nexciton):
            chiXYb[S, :len(occ), :] = wt_pk[:, :, S].T
            chiXYb[S, len(occ):, :] = wt_pc[:, :, S].T
            
        return chiXYb

    def get_chi_b_psd_full(self, nocc, eigenvalues_casida, chiXYa, spin_channel='alpha', eri_w=None, p_state=None):
        """Computes the PSD transition amplitude chiXYb using full ERIs."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            eri = self.eri_a if spin_channel == 'alpha' else self.eri_b
            if eri_w is None:
                eri_w = eri
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
        else:
            eps = self.eps
            eri = self.eri_chemist
            if eri_w is None:
                eri_w = eri
            nocc_spin = nocc
            
        occ, virt = self._get_occ_virt_indices(eps, nocc_spin)
        norb = len(eps)
        nexciton = len(eigenvalues_casida)
        
        chiXYa_ov = chiXYa[:, occ, :][:, :, virt]
        
        # Vectorized denominators
        d_ia = eps[virt] - eps[occ][:, None]
        denom_minus = 1.0 / (d_ia[None, :, :] - eigenvalues_casida[:, None, None])
        denom_plus = 1.0 / (d_ia[None, :, :] + eigenvalues_casida[:, None, None])
        
        chiXY_minus = chiXYa_ov * denom_minus
        chiXY_plus = chiXYa_ov * denom_plus
        
        eri_pajk = eri_w[np.ix_(np.arange(norb), virt, occ, occ)]
        wt_pk = np.einsum('pajk, Sja -> pkS', eri_pajk, chiXY_plus)
        
        eri_pjak = eri_w[np.ix_(np.arange(norb), occ, virt, occ)]
        wt_pk += np.einsum('pjak, Sja -> pkS', eri_pjak, chiXY_minus)
        
        eri_pajc = eri_w[np.ix_(np.arange(norb), virt, occ, virt)]
        wt_pc = np.einsum('pajc, Sja -> pcS', eri_pajc, chiXY_minus)
        
        eri_pjac = eri_w[np.ix_(np.arange(norb), occ, virt, virt)]
        wt_pc += np.einsum('pjac, Sja -> pcS', eri_pjac, chiXY_plus)
        
        chiXYb = np.zeros((nexciton, norb, norb))
        for S in range(nexciton):
            chiXYb[S, :len(occ), :] = wt_pk[:, :, S].T
            chiXYb[S, len(occ):, :] = wt_pc[:, :, S].T
            
        return chiXYb

    def get_chi_b_vertex_sf_df(self, nocc, X, Y, spin_channel='alpha', channel='ba', eri_w=None, p_state=None):
        """Computes the spin-flip vertex correction transition amplitude chiXYb using density fitting."""
        nocc_a, nocc_b = nocc
        
        # Helper to get the correct density fitting block
        def get_df_block(spin1, spin2, idx1, idx2):
            if spin1 == 'a' and spin2 == 'a':
                return self.df_a[:, idx1[:, None], idx2]
            elif spin1 == 'b' and spin2 == 'b':
                return self.df_b[:, idx1[:, None], idx2]
            elif spin1 == 'b' and spin2 == 'a':
                return self.df_ab[:, idx1[:, None], idx2]
            else: # 'a' and 'b'
                return self.df_ab[:, idx2[:, None], idx1].transpose(0, 2, 1)
                
        # Determine excitation spins
        s_i = 'b' if channel == 'ba' else 'a'
        s_a = 'a' if channel == 'ba' else 'b'
        s_p = 'a' if spin_channel == 'alpha' else 'b'
        
        if channel == 'ba':
            occ = np.arange(nocc_b)
            virt = np.arange(nocc_a, self.norb_a)
            virt_target = np.arange(nocc_b, self.norb_b)
        else:
            occ = np.arange(nocc_a)
            virt = np.arange(nocc_b, self.norb_b)
            virt_target = np.arange(nocc_a, self.norb_a)
            
        nocc_val = len(occ)
        nvirt_val = len(virt)
        nvirt_target = len(virt_target)
        nexciton = X.shape[1]
        
        X_spin = X
        Y_spin = Y
        X_3d = X_spin.reshape(nocc_val, nvirt_val, -1)
        Y_3d = Y_spin.reshape(nocc_val, nvirt_val, -1)
        X_flat = X_3d.reshape(-1, nexciton)
        Y_flat = Y_3d.reshape(-1, nexciton)
        
        W = np.eye(self.naux) if eri_w is None else np.asarray(eri_w)
        w_key = self._identity_key(eri_w) if eri_w is not None else ('eye', self.naux)

        # Set up density fitting coefficients
        C_vo = get_df_block(s_a, s_i, virt, occ)
        C_oo = get_df_block(s_i, s_i, occ, occ)
        C_ko = get_df_block(s_i, s_i, virt_target, occ)
        C_vp_slice = get_df_block(s_a, s_p, virt, np.array([p_state]))[:, :, 0]
        C_op_slice = get_df_block(s_i, s_p, occ, np.array([p_state]))[:, :, 0]
        C_op_slice_v = get_df_block(s_i, s_p, virt_target, np.array([p_state]))[:, :, 0]

        C_vo_flat = C_vo.reshape(self.naux, -1)
        C_oo_flat = C_oo.reshape(self.naux, -1)
        C_ko_flat = C_ko.reshape(self.naux, -1)

        # Sliced occupied block (intermediate is occupied)
        tmp_1 = W @ C_op_slice
        res_1 = C_vo_flat.T @ tmp_1
        integs1_iak = res_1.reshape(nvirt_val, nocc_val, nocc_val).transpose(2, 0, 1)

        # tmp_2/tmp_2_v depend only on (nocc, channel, eri_w) -- independent of
        # spin_channel and p_state -- so cached across the p_state loop, same
        # rationale as get_chi_b_vertex_df.
        tmp_2 = self._amp_cache(('chi_b_sf_tmp2', nocc, channel, w_key),
                                 lambda: W @ C_oo_flat)
        res_2 = C_vp_slice.T @ tmp_2
        integs2_iak = res_2.reshape(nvirt_val, nocc_val, nocc_val).transpose(1, 0, 2)

        integs1_flat_o = integs1_iak.reshape(nocc_val * nvirt_val, nocc_val)
        integs2_flat_o = integs2_iak.reshape(nocc_val * nvirt_val, nocc_val)

        chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o)

        # Sliced virtual block (intermediate is virtual)
        tmp_1_v = W @ C_op_slice_v
        res_1_v = C_vo_flat.T @ tmp_1_v
        integs1_iac = res_1_v.reshape(nvirt_val, nocc_val, nvirt_target).transpose(1, 0, 2)

        tmp_2_v = self._amp_cache(('chi_b_sf_tmp2v', nocc, channel, w_key),
                                   lambda: W @ C_ko_flat)
        res_2_v = C_vp_slice.T @ tmp_2_v
        integs2_iac = res_2_v.reshape(nvirt_val, nvirt_target, nocc_val).transpose(2, 0, 1)

        integs1_flat_v = integs1_iac.reshape(nocc_val * nvirt_val, nvirt_target)
        integs2_flat_v = integs2_iac.reshape(nocc_val * nvirt_val, nvirt_target)

        chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v)

        chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
        return chiXYb

    def get_chi_b_vertex_sf_full(self, nocc, X, Y, spin_channel='alpha', channel='ba', eri_w=None, p_state=None):
        """Computes the spin-flip vertex correction transition amplitude chiXYb using full ERIs."""
        nocc_a, nocc_b = nocc
        
        if channel == 'ba':
            occ = np.arange(nocc_b)
            virt = np.arange(nocc_a, self.norb_a)
        else:
            occ = np.arange(nocc_a)
            virt = np.arange(nocc_b, self.norb_b)
            
        nocc_val = len(occ)
        nvirt_val = len(virt)
        nexciton = X.shape[1]
        
        X_spin = X
        Y_spin = Y
        X_3d = X_spin.reshape(nocc_val, nvirt_val, -1)
        Y_3d = Y_spin.reshape(nocc_val, nvirt_val, -1)
        X_flat = X_3d.reshape(-1, nexciton)
        Y_flat = Y_3d.reshape(-1, nexciton)
        
        if eri_w is None:
            eri_w = self.eri_ab
            
        if spin_channel == 'alpha':
            norb_target = len(self.eps_b)
            if channel == 'ba':
                integs1 = eri_w[np.ix_(virt, occ, np.array([p_state]), np.arange(norb_target))].transpose(1, 0, 3, 2)[:, :, :, 0]
                integs2 = eri_w[np.ix_(virt, np.arange(norb_target), np.array([p_state]), occ)].transpose(3, 0, 1, 2)[:, :, :, 0]
            else:
                integs1 = eri_w[np.ix_(np.array([p_state]), np.arange(norb_target), occ, virt)].transpose(2, 3, 1, 0)[:, :, :, 0]
                integs2 = eri_w[np.ix_(np.array([p_state]), virt, occ, np.arange(norb_target))].transpose(2, 1, 3, 0)[:, :, :, 0]
        else:
            norb_target = len(self.eps_a)
            if channel == 'ba':
                integs1 = eri_w[np.ix_(np.arange(norb_target), np.array([p_state]), virt, occ)].transpose(3, 2, 0, 1)[:, :, :, 0]
                integs2 = eri_w[np.ix_(np.arange(norb_target), np.array([p_state]), occ, virt)].transpose(2, 3, 0, 1)[:, :, :, 0]
            else:
                integs1 = eri_w[np.ix_(np.arange(norb_target), np.array([p_state]), occ, virt)].transpose(2, 3, 0, 1)[:, :, :, 0]
                integs2 = eri_w[np.ix_(occ, np.array([p_state]), np.arange(norb_target), virt)].transpose(0, 3, 2, 1)[:, :, :, 0]
                
        integs1_flat_o = integs1[:, :, :nocc_val].reshape(nocc_val * nvirt_val, nocc_val)
        integs2_flat_o = integs2[:, :, :nocc_val].reshape(nocc_val * nvirt_val, nocc_val)
        
        chi_occ = self._blocked_contract(X_flat, integs1_flat_o, Y_flat, integs2_flat_o)

        integs1_flat_v = integs1[:, :, nocc_val:].reshape(nocc_val * nvirt_val, nvirt_val)
        integs2_flat_v = integs2[:, :, nocc_val:].reshape(nocc_val * nvirt_val, nvirt_val)

        chi_virt = self._blocked_contract(X_flat, integs1_flat_v, Y_flat, integs2_flat_v)

        chiXYb = np.concatenate([chi_occ, chi_virt], axis=1)
        return chiXYb
