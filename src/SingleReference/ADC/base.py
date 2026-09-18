"""Dyson IP/EA-ADC solver classes: the ADCSolver dispatcher and the two
branch classes (state + option validation + routing). All physics lives in
the route modules (adc_r_* / adc_u_*) and the solver machinery in solve.

    solver = ADCSolver(mf, level='adc3', df=True, ...)
    e, Z = solver.solve(static_correction=sc)
"""
import math
import numpy as np
from itertools import combinations

from pyscf import scf as _scf

from src.Base.pyscf_interface import (GProxy, get_antisymmetrized_spin_eri,
                                      get_antisymmetrized_spin_block_eri,
                                      get_df_spin_orbital_factor,
                                      get_orbital_energies,
                                      get_two_electron_integrals_chemist,
                                      get_uhf_spin_orbital_arrays_blockstacked,
                                      get_uhf_spin_orbital_df_factor_blockstacked,
                                      DFIntegrals)
from src.SingleReference.EpsteinNesbet import validate_en_dress
from src.SingleReference.ADC import adc_r_utils
from src.SingleReference.ADC import adc_r_dense_full, adc_r_dense_df
from src.SingleReference.ADC import adc_r_sigma_full, adc_r_sigma_df
from src.SingleReference.ADC import adc_u_utils
from src.SingleReference.ADC import adc_u_dense_full, adc_u_dense_df
from src.SingleReference.ADC import adc_u_sigma_full, adc_u_sigma_df
from src.SingleReference.ADC import adc_r_driver, adc_u_driver
from src.SingleReference.ADC import spin_adapt  # module import: it imports this file
from src.SingleReference.ADC.solve import (diag_dense, lanczos_spectral,
                                           downfolded_seed_vectors,
                                           select_resonant_configs,
                                           satellite_seed_vectors,
                                           dominant_satellite_seed)


class ADCSolver:
    """Unified front end: dispatches on the reference (RHF -> restricted /
    spin-adapted, UHF -> spin-orbital; spin='spinorbital' forces the
    spin-orbital branch for an RHF reference).

        solver = ADCSolver(mf, level=..., df=..., matrix_free=...,
                           en_dress=..., screening=...)
        e, Z = solver.solve(static_correction=...)

    Returns an ADCSolverRestricted or ADCSolverUnrestricted instance."""

    def __new__(cls, mf, mol=None, spin='auto', **opts):
        is_uhf = isinstance(mf, _scf.uhf.UHF)
        if spin == 'auto':
            branch = ADCSolverUnrestricted if is_uhf else ADCSolverRestricted
        elif spin == 'spinorbital':
            branch = ADCSolverUnrestricted
        elif spin == 'restricted':
            if is_uhf:
                raise ValueError("spin='restricted' needs an RHF reference")
            branch = ADCSolverRestricted
        else:
            raise ValueError(f"spin={spin!r}; expected 'auto', 'restricted', "
                             "or 'spinorbital'")
        return branch(mf, mol=mol, **opts)


class ADCSolverUnrestricted:
    """Spin-orbital (determinant-basis) Dyson IP/EA-ADC(3) solver for an RHF
    or UHF (blockstacked) reference.

    State and routing only -- the physics lives in the route modules:
    adc_u_dense_full/adc_u_dense_df (supermatrix), adc_u_sigma_full/
    adc_u_sigma_df (matrix-free operator; the g-free B_spin route is the
    production DF path). EN dressing goes through dressed_t2_amplitudes +
    the t2_ijcd hook (u1 shift threaded automatically); the u2_denom_dress
    ATTRIBUTE stays refused (see its setter)."""

    LEVELS = ('adc3',)

    def __init__(self, mf=None, mol=None, level='adc3', df=False,
                 matrix_free=True, en_dress=None, screening=None, nocc=None):
        """mf=None is internal -- from_arrays supplies the arrays instead."""
        # ---- flat guard block: every inapplicable option RAISES, never ignored
        if level not in self.LEVELS:
            raise ValueError(f"level={level!r}; the spin-orbital solver is "
                             f"ADC(3) only (expected one of {self.LEVELS}). "
                             "ADC(2)-X is restricted-branch only.")
        if screening is not None:
            raise ValueError("screening is restricted-branch only")
        en_dress = validate_en_dress(en_dress)
        if en_dress is not None:
            if en_dress.get('spin_adapted', False):
                raise ValueError("en_dress['spin_adapted']=True is a "
                                 "restricted/CSF concept; the spin-orbital "
                                 "branch is determinant-wise by definition")
            if 'shift' in en_dress:
                raise ValueError("en_dress['shift'] is the spin-adapted "
                                 "weighting; determinant-wise EN has none")
            if not matrix_free:
                raise ValueError("dense EN is not implemented on the "
                                 "spin-orbital branch; use matrix_free=True "
                                 "(dressed-t2 route)")
        self.level = level
        self.df = df
        self.matrix_free = matrix_free
        self.en_dress = en_dress
        self.nocc = nocc
        self.last_result = {}

        if mf is None:
            return                      # from_arrays fills the state next
        # ---- mean-field plumbing ----
        mol = mol if mol is not None else mf.mol
        if self.df and getattr(mf, 'with_df', None) is None:
            raise ValueError(
                "df=True needs a density-fitted mean field (mf.with_df)")
        if isinstance(mf, _scf.uhf.UHF):
            eps_spin, g_anti, nocc_spin = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
            B_spin = (get_uhf_spin_orbital_df_factor_blockstacked(mol, mf)
                      if self.df else None)
        else:
            nocc_sp = mol.nelectron // 2
            eps = get_orbital_energies(mf, representation='spatial')
            eps_spin = np.repeat(eps, 2)
            eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
            g_anti = get_antisymmetrized_spin_eri(eri)
            nocc_spin = 2 * nocc_sp
            B_spin = (get_df_spin_orbital_factor(DFIntegrals.from_scf(mol, mf).B_aa)
                      if self.df else None)
        if self.df and self.matrix_free:
            g_anti = None                     # g-free production DF route
        if self.nocc is None:
            self.nocc = nocc_spin
        self._init_state(eps_spin, g_anti, B_spin)

    @classmethod
    def from_arrays(cls, eps_spin, g_anti_spin=None, B_spin=None, nocc=None,
                    **opts):
        """Construct from raw spin-orbital arrays; df inferred from B_spin."""
        opts.setdefault('df', B_spin is not None)
        opts.setdefault('nocc', nocc)
        self = cls(None, **opts)
        self._init_state(np.asarray(eps_spin), g_anti_spin, B_spin)
        return self

    def _init_state(self, eps_spin, g_anti_spin, B_spin):
        if g_anti_spin is None and B_spin is None:
            raise ValueError("ADCSolverUnrestricted: g_anti_spin=None requires "
                             "B_spin (g-free mode needs the DF factor)")
        self.eps = np.asarray(eps_spin)
        self.g = g_anti_spin
        self.B_spin = B_spin
        self.norb = len(self.eps)
        self._cache = {}

    def _require_dense_g(self, what):
        """Raise a clear error when a dense-only code path is entered on a
        g-free (B_spin-only) solver instead of failing with an opaque
        'NoneType is not subscriptable' deep inside an einsum."""
        if self.g is None:
            raise NotImplementedError(
                f"{what} requires the dense g_anti_spin tensor, but this solver "
                f"was built g-free (g_anti_spin=None, B_spin only). Use the "
                f"matrix-free ADC(3) path, or construct the solver with dense g.")

    # u2_denom_dress ATTRIBUTE guard: the merged hand-written U^(2) form
    # gives wrong couplings under a naive denominator swap. EN dressing
    # instead goes through dressed_t2_amplitudes + the t2_ijcd hook, whose
    # u1_dressing_shift matches the generated route exactly.
    _u2_denom_dress = None

    @property
    def u2_denom_dress(self):
        return self._u2_denom_dress

    @u2_denom_dress.setter
    def u2_denom_dress(self, value):
        if value:
            raise NotImplementedError(
                "the u2_denom_dress attribute is not supported by the spin-orbital "
                "solver: its U^(2) blocks are a merged form that cannot carry a "
                "naively re-divided amplitude. Use dressed_t2_amplitudes(...) with "
                "the t2_ijcd hook (exact, u1 shift threaded automatically), or "
                "the ADCSolverRestricted route.")
        self._u2_denom_dress = value

    def _g_proxy(self):
        """GProxy wrapping self.g/self.B_spin -- see GProxy's docstring."""
        return GProxy(self.B_spin, self.g)

    def _cached(self, name, nocc, compute_fn):
        """Memoize an nocc-only-dependent computation, keyed by (name, nocc)."""
        key = (name, nocc)
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def dimensions(self, nocc):
        """Segment sizes for the given number of occupied spin-orbitals."""
        norb = self.norb
        nvirt = norb - nocc
        n2h1p = nocc * (nocc - 1) // 2 * nvirt
        n2p1h = nvirt * (nvirt - 1) // 2 * nocc
        return {'norb': norb, 'nocc': nocc, 'nvirt': nvirt,
                'n2h1p': n2h1p, 'n2p1h': n2p1h, 'nH': norb + n2h1p + n2p1h}

    def dimensions_adc4(self, nocc):
        """dimensions() plus the 3h2p (extends 2h1p) / 3p2h (extends 2p1h) segment
        sizes that define the ADC(4) configuration space:
        3h2p is (i<j<k occ, a<b virt), 3p2h is (i<j occ, a<b<c virt) -- the Table I
        higher excitation classes for the "2h-1p"/M^II and "2p-1h"/M^I sectors,
        respectively, of Schirmer, Cederbaum & Walter, PRA 28, 1237 (1983).
        """
        d = self.dimensions(nocc)
        nocc_, nvirt_ = d['nocc'], d['nvirt']
        n3h2p = math.comb(nocc_, 3) * math.comb(nvirt_, 2)
        n3p2h = math.comb(nocc_, 2) * math.comb(nvirt_, 3)
        d = dict(d)
        d['n3h2p'] = n3h2p
        d['n3p2h'] = n3p2h
        d['nH_adc4'] = d['nH'] + n3h2p + n3p2h
        return d

    def _configs_3h2p(self, nocc):
        """3h2p configuration index arrays (i<j<k occ, a<b virt), flattened in
        occ-triple-outer / virt-pair-inner order (mirrors the 2h1p (i<j,a)
        ordering in build_supermatrix). Returns (i, j, k, a, b), each 1D of
        length n3h2p = C(nocc,3)*C(nvirt,2).
        """
        norb = self.norb
        occ = np.arange(nocc)
        virt = np.arange(nocc, norb)
        triples = np.array(list(combinations(occ, 3)), dtype=int).reshape(-1, 3)
        pairs = np.array(list(combinations(virt, 2)), dtype=int).reshape(-1, 2)
        ntrip, npair = len(triples), len(pairs)
        i = np.repeat(triples[:, 0], npair)
        j = np.repeat(triples[:, 1], npair)
        k = np.repeat(triples[:, 2], npair)
        a = np.tile(pairs[:, 0], ntrip)
        b = np.tile(pairs[:, 1], ntrip)
        return i, j, k, a, b

    def _configs_3p2h(self, nocc):
        """3p2h configuration index arrays (i<j occ, a<b<c virt), flattened in
        occ-pair-outer / virt-triple-inner order (mirrors the 2p1h (i,a<b)
        ordering in build_supermatrix). Returns (i, j, a, b, c), each 1D of
        length n3p2h = C(nocc,2)*C(nvirt,3).
        """
        norb = self.norb
        occ = np.arange(nocc)
        virt = np.arange(nocc, norb)
        pairs = np.array(list(combinations(occ, 2)), dtype=int).reshape(-1, 2)
        triples = np.array(list(combinations(virt, 3)), dtype=int).reshape(-1, 3)
        npair, ntrip = len(pairs), len(triples)
        i = np.repeat(pairs[:, 0], ntrip)
        j = np.repeat(pairs[:, 1], ntrip)
        a = np.tile(triples[:, 0], npair)
        b = np.tile(triples[:, 1], npair)
        c = np.tile(triples[:, 2], npair)
        return i, j, a, b, c


    def _configs_2h1p(self, nocc):
        """2h1p configuration index arrays (i<j occ, a virt), in the same
        outer-pair/inner-virt order used inline in build_supermatrix."""
        norb = self.norb
        occ = np.arange(nocc)
        virt = np.arange(nocc, norb)
        nvirt = len(virt)
        iu, ju = np.triu_indices(nocc, k=1)
        npair_o = len(iu)
        i = np.repeat(occ[iu], nvirt)
        j = np.repeat(occ[ju], nvirt)
        a = np.tile(virt, npair_o)
        return i, j, a

    def _configs_2p1h(self, nocc):
        """2p1h configuration index arrays (i occ, a<b virt), in the same
        outer-occ/inner-pair order used inline in build_supermatrix."""
        norb = self.norb
        occ = np.arange(nocc)
        virt = np.arange(nocc, norb)
        au, bu = np.triu_indices(len(virt), k=1)
        npair_v = len(au)
        i = np.repeat(occ, npair_v)
        a = np.tile(virt[au], nocc)
        b = np.tile(virt[bu], nocc)
        return i, a, b

    # ---- physics delegates (route on B_spin / g) ----

    def _sigma_mod(self):
        return adc_u_sigma_df if self.B_spin is not None else adc_u_sigma_full

    def _u2_denominators(self, nocc):
        return adc_u_utils.u2_denominators(self.eps, nocc)

    def u1_dressing_shift(self, nocc, t2_ijcd):
        return adc_u_utils.u1_dressing_shift(self, nocc, t2_ijcd)

    def dressed_t2_amplitudes(self, nocc, u2_denom_dress):
        return adc_u_utils.dressed_t2_amplitudes(self, nocc, u2_denom_dress)

    def _build_matrix_free_ingredients(self, nocc):
        return self._cached(
            '_build_matrix_free_ingredients', nocc,
            lambda: adc_u_utils.build_matrix_free_ingredients(self, nocc))

    def _C_2h1p_block(self, nocc, ket_idx, bra_idx=None):
        return adc_u_dense_full.C_2h1p_block(self, nocc, ket_idx, bra_idx)

    def _C_2p1h_block(self, nocc, ket_idx, bra_idx=None):
        return adc_u_dense_full.C_2p1h_block(self, nocc, ket_idx, bra_idx)

    def apply_U_2h1p(self, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None):
        return self._sigma_mod().apply_U_2h1p(self, nocc, z_p, Vfull,
                                              t2_ijcd, u1_shift)

    def apply_U_2p1h(self, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None):
        return self._sigma_mod().apply_U_2p1h(self, nocc, z_p, Vfull,
                                              t2_ijcd, u1_shift)

    def apply_C_2h1p(self, nocc, V):
        return self._sigma_mod().apply_C_2h1p(self, nocc, V)

    def apply_C_2p1h(self, nocc, V):
        return self._sigma_mod().apply_C_2p1h(self, nocc, V)

    def build_supermatrix(self, nocc, static_correction=None):
        """(nH, nH) supermatrix; dense g route, or B_spin reconstruction
        when the solver is g-free."""
        if self.g is None:
            return adc_u_dense_df.build_supermatrix(self, nocc, static_correction)
        return adc_u_dense_full.build_supermatrix(self, nocc, static_correction)

    def build_matrix_free_operator(self, nocc, static_correction=None,
                                   t2_ijcd=None):
        """(aop, diag, dims); g-free B_spin route iff B_spin is set.
        t2_ijcd: optional EN-dressed T2^(1) (u1 shift threaded automatically)."""
        return self._sigma_mod().build_operator(self, nocc, static_correction,
                                                t2_ijcd=t2_ijcd)

    def solve_dense(self, nocc, static_correction=None, threshold=5000):
        """Dense diagonalization (mid-level); (eGF, Z, Reigv) sorted ascending."""
        H = self.build_supermatrix(nocc, static_correction=static_correction)
        return diag_dense(H, self.norb, threshold=threshold)

    def _build_seed_operator(self, nocc, static_correction=None):
        """(aop, diag, nH) for THIS solver's operator -- shared by
        downfolded_seeds() and satellite_seeds() so the matrix-free/EN-dress
        wiring lives in exactly one place."""
        t2 = None
        if self.en_dress is not None:
            t2 = adc_u_utils.dressed_t2_amplitudes(self, nocc, self.en_dress)
        mod = adc_u_sigma_df if self.B_spin is not None else adc_u_sigma_full
        aop, diag, dims = mod.build_operator(self, nocc, static_correction,
                                             t2_ijcd=t2)
        return aop, diag, dims['nH']

    def downfolded_seeds(self, static_correction=None, orbital_window=None,
                         omega0=None, eta=0.01):
        """Mixed-orbital Davidson seeds from a one-shot on-shell downfold of
        THIS solver's operator at omega0 (default eps_HOMO): the full
        (non-diagonal) orbital-window H_eff(omega0) is diagonalized and its
        eigenvectors returned as ref_vec candidates for solve() -- they carry
        the pi/pi*-type mixing that no single Koopmans unit vector can, so
        root-following can reach strongly orbital-mixed lowest-IP states
        (see solve.downfolded_seed_vectors). orbital_window defaults to all
        occupied spin orbitals. Returns (e_eff, seeds) ascending in e_eff;
        seeds[:, k] is the length-nH seed for e_eff[k]."""
        if self.nocc is None:
            raise ValueError("downfolded_seeds() needs nocc (constructed "
                             "without it -- from_arrays(nocc=...))")
        if not self.matrix_free:
            raise ValueError("downfolded_seeds() is matrix-free only; the "
                             "dense route already enumerates every state via "
                             "solve_dense()")
        nocc = self.nocc
        if orbital_window is None:
            orbital_window = np.arange(nocc)
        if omega0 is None:
            omega0 = float(np.max(self.eps[:nocc]))
        aop, diag, nH = self._build_seed_operator(nocc, static_correction)
        return downfolded_seed_vectors(aop, diag, nH, self.norb,
                                       orbital_window, omega0, eta=eta)

    def satellite_seeds(self, omega0, static_correction=None,
                        orbital_window=None, tol=0.05, max_candidates=20,
                        eta=0.01):
        """Satellite-targeting Davidson seed at energy omega0 (Hartree):
        resonant 2h1p/2p1h configurations within `tol` of omega0 are kept
        EXPLICIT (not folded away, unlike downfolded_seeds) so the resulting
        eigenvectors can carry real weight on a 2h1p+ row -- see
        solve.satellite_seed_vectors for why downfolded_seeds structurally
        cannot seed a satellite. orbital_window defaults to all occupied
        spin orbitals.

        Returns (e_guess, seed_vec, e_eff, seeds, full_idx): e_guess/seed_vec
        is the single dominant_satellite_seed pick (feed straight into
        solve(ref_vec=seed_vec)); e_eff/seeds/full_idx is the full small-space
        diagonalization (satellite_seed_vectors) in case you want to inspect
        or pick a different root by hand."""
        if self.nocc is None:
            raise ValueError("satellite_seeds() needs nocc (constructed "
                             "without it -- from_arrays(nocc=...))")
        if not self.matrix_free:
            raise ValueError("satellite_seeds() is matrix-free only; the "
                             "dense route already enumerates every state via "
                             "solve_dense()")
        nocc = self.nocc
        if orbital_window is None:
            orbital_window = np.arange(nocc)
        aop, diag, nH = self._build_seed_operator(nocc, static_correction)
        sat_idx = select_resonant_configs(diag, self.norb, omega0, tol=tol,
                                          max_candidates=max_candidates)
        if sat_idx.size == 0:
            raise ValueError(f"satellite_seeds(): no 2h1p/2p1h configuration "
                             f"within tol={tol} Ha of omega0={omega0} -- "
                             f"widen tol or check omega0's units/reference")
        e_eff, seeds, full_idx = satellite_seed_vectors(
            aop, diag, nH, self.norb, orbital_window, sat_idx, omega0, eta=eta)
        e_guess, seed_vec = dominant_satellite_seed(
            e_eff, seeds, full_idx, len(orbital_window))
        return e_guess, seed_vec, e_eff, seeds, full_idx

    # ---- unified entry point ----

    def solve(self, static_correction=None, nroots=1, homo_index=None,
              ref_vec=None, conv_tol=1e-6, tol=1e-8, threshold=5000, verbose=0,
              method='davidson', omega_range=None):
        """(e, Z) for the configured route; details on self.last_result.
        static_correction is spin-orbital sized ((nso, nso)).

        method='lanczos': matrix-free Lanczos/continued-fraction spectral
        solve instead of Davidson root-following -- needs matrix_free=True
        and omega_range=(omega_lo, omega_hi) (Hartree, same sign convention
        as diag/eigenvalues elsewhere: negative for IP removal energies).
        ref_vec (or homo_index) is the PHYSICAL starting channel here (e.g.
        a unit vector on one orbital = that orbital's full removal
        spectrum), not a root-following seed. Returns the peak positions/
        weights found in omega_range as (e, Z); the full spectral function
        is on self.last_result (see solve.lanczos_spectral)."""
        if self.nocc is None:
            raise ValueError("solve() needs nocc (constructed without it -- "
                             "from_arrays(nocc=...), or use the mid-level API)")
        if method == 'lanczos':
            return self._solve_lanczos(static_correction, homo_index, ref_vec,
                                       omega_range)
        if method != 'davidson':
            raise ValueError(f"method={method!r}; expected 'davidson' or 'lanczos'")
        return adc_u_driver.solve(self, static_correction, nroots, homo_index,
                                  ref_vec, conv_tol, threshold, verbose)

    def _solve_lanczos(self, sc, homo_index, ref_vec, omega_range):
        if not self.matrix_free:
            raise ValueError("method='lanczos' needs matrix_free=True")
        if omega_range is None:
            raise ValueError("method='lanczos' needs omega_range=(lo, hi) "
                             "(Hartree, same sign convention as diag)")
        nocc = self.nocc
        homo = homo_index if homo_index is not None else nocc - 1
        t2 = None
        if self.en_dress is not None:
            t2 = adc_u_utils.dressed_t2_amplitudes(self, nocc, self.en_dress)
        mod = adc_u_sigma_df if self.B_spin is not None else adc_u_sigma_full
        op, diag, dims = mod.build_operator(self, nocc, sc, t2_ijcd=t2)
        n = dims['nH']
        v0 = np.asarray(ref_vec, float) if ref_vec is not None else np.eye(n)[homo]
        out = lanczos_spectral(op, diag, v0, omega_range)
        self.last_result = out
        return out['peak_omega'], out['peak_weight']


class ADCSolverRestricted:
    """Spin-adapted (RHF, closed-shell) Dyson IP/EA-ADC in the CSF basis
    {orbital p} + {2h1p Type I/II/III} + {2p1h Type I'/II'/III'}.

    State and routing only -- the physics lives in the route modules
    adc_r_dense_full/adc_r_dense_df (supermatrix) and adc_r_sigma_full/
    adc_r_sigma_df (matrix-free). Construct from a mean field
    (ADCSolverRestricted(mf, ...)) or raw arrays (from_arrays); solve()
    returns (e, Z), details on last_result.
    """

    LEVELS = ('adc2x', 'adc3')

    def __init__(self, mf=None, mol=None, level='adc3', df=False,
                 matrix_free=True, en_dress=None, screening=None, nocc=None):
        """mf=None is internal -- from_arrays supplies the arrays instead."""
        # ---- flat guard block: every inapplicable option RAISES, never ignored
        if level not in self.LEVELS:
            raise ValueError(f"level={level!r}; expected one of {self.LEVELS}")
        en_dress = validate_en_dress(en_dress)
        W_chemist = W_aux = None
        screen_coupling = False
        if screening is not None:
            unknown = set(screening) - {'W_chemist', 'W_aux', 'coupling'}
            if unknown:
                raise ValueError(f"screening: unknown key(s) {sorted(unknown)}")
            W_chemist = screening.get('W_chemist')
            W_aux = screening.get('W_aux')
            screen_coupling = bool(screening.get('coupling', False))
            if W_chemist is None and W_aux is None:
                raise ValueError("screening needs 'W_chemist' (dense) or "
                                 "'W_aux' (DF metric)")
            if W_aux is not None and level != 'adc2x':
                raise NotImplementedError("W_aux screening is adc2x-only")
            if W_aux is not None and not matrix_free:
                raise ValueError("W_aux is the matrix-free (screened DF factor) "
                                 "representation; use W_chemist for dense builds")
            if W_aux is not None and not df:
                raise ValueError("W_aux screening needs df=True (B_aa)")
            if W_chemist is not None and df and matrix_free:
                raise ValueError("dense W_chemist screening is a dense-integral "
                                 "route; with df=True use W_aux, or matrix_free=False")
            if screen_coupling and W_chemist is None:
                raise NotImplementedError("screen_coupling (PSD1 U^(1) screening) "
                                          "is implemented on the W_chemist path only")
        self.level = level
        self.df = df
        self.matrix_free = matrix_free
        self.en_dress = en_dress
        self.u2_denom_dress = en_dress          # name the route modules read
        self.nocc = nocc
        self.W_chemist = W_chemist
        self.W_aux = W_aux
        self.screen_coupling = screen_coupling
        self._is_adc2x = (level == 'adc2x')
        self.last_result = {}

        if mf is None:
            return                      # from_arrays fills the state next
        # ---- mean-field plumbing ----
        if isinstance(mf, _scf.uhf.UHF):
            raise ValueError("UHF reference: use ADCSolver(mf, ...) (dispatches "
                             "to the spin-orbital branch) or ADCSolverUnrestricted.")
        mol = mol if mol is not None else mf.mol
        if self.df and getattr(mf, 'with_df', None) is None:
            raise ValueError(
                "df=True needs a density-fitted mean field (run "
                "scf.RHF(mol).density_fit(...).run(), or attach mf.with_df); "
                "a plain mf would silently fall back to the naux=norb^2 "
                "exact factorization")
        nocc = self.nocc if self.nocc is not None else mol.nelectron // 2
        eps = get_orbital_energies(mf, representation='spatial')
        B_aa = DFIntegrals.from_scf(mol, mf).B_aa if self.df else None
        # the DF routes never need dense integrals
        eri = None
        if not self.df:
            eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        self._init_state(eps, eri, B_aa, nocc, mf=mf, mol=mol)

    @classmethod
    def from_arrays(cls, eps, eri_chemist=None, B_aa=None, nocc=None,
                    W_chemist=None, W_aux=None, screen_coupling=False,
                    share_caches_with=None, **opts):
        """Construct from raw arrays (tests / synthetic Hamiltonians); df is
        inferred from B_aa. Without nocc only the explicit-nocc mid-level API
        (build_supermatrix/build_matrix_free_operator/solve_dense) is usable."""
        opts.setdefault('df', B_aa is not None)
        opts.setdefault('nocc', nocc)
        if W_chemist is not None or W_aux is not None:
            screening = {}
            if W_chemist is not None:
                screening['W_chemist'] = W_chemist
            if W_aux is not None:
                screening['W_aux'] = W_aux
            if screen_coupling:
                screening['coupling'] = True
            opts.setdefault('screening', screening)
        self = cls(None, **opts)
        if self.df and B_aa is None:
            raise ValueError("df=True needs B_aa")
        self._init_state(np.asarray(eps), eri_chemist, B_aa, self.nocc,
                         share_caches_with=share_caches_with)
        return self

    def _init_state(self, eps, eri_chemist, B_aa, nocc, mf=None, mol=None,
                    share_caches_with=None):
        self.eps = np.asarray(eps)
        self.eri = eri_chemist
        self.B_aa = B_aa
        self.norb = self.eps.shape[0]
        self.nocc = nocc
        self._mf = mf
        self._mol = mol
        self._spin_solver = None
        self._csf_adapter = None
        if share_caches_with is not None:
            if (share_caches_with.norb != self.norb
                    or not np.allclose(share_caches_with.eps, self.eps)):
                raise ValueError("share_caches_with: solvers must be built from the "
                                  "same orbital energies/integrals")
            self._spin_solver = share_caches_with._get_spin_orbital_solver()
            self._csf_adapter = share_caches_with._get_csf_adapter()

    def _get_spin_orbital_solver(self):
        """Lazily build & cache the spin-orbital solver backing the generated
        ADC(4)+ path -- eps/g expand from the RHF spatial-MO quantities by
        the standard interleaved convention (np.repeat(eps, 2), no exchange
        term across spin blocks)."""
        if self._spin_solver is None:
            eps_spin = np.repeat(self.eps, 2)
            g_anti_spin = get_antisymmetrized_spin_eri(self.eri)
            B_spin = get_df_spin_orbital_factor(self.B_aa) if self.B_aa is not None else None
            self._spin_solver = ADCSolverUnrestricted.from_arrays(
                eps_spin, g_anti_spin, B_spin=B_spin, nocc=2 * self.nocc)
        return self._spin_solver

    def _get_csf_adapter(self):
        """Lazily build & cache the spin-adapted (doublet-CSF-basis) front-end
        for the wrapped spin-orbital solver -- see ADCSolverCSF in
        spin_adapt.py. Genuinely ~4x smaller per configuration dimension and
        ~16x smaller coupling matrices than the raw spin-orbital solver, with
        every block obtained by the numerically-validated isometry T rather
        than a re-derived CSF formula."""
        if getattr(self, '_csf_adapter', None) is None:
            self._csf_adapter = spin_adapt.ADCSolverCSF(
                self._get_spin_orbital_solver())
        return self._csf_adapter

    @staticmethod
    def _embed_static_correction_spin(static_correction):
        """Expand a spatial-MO static_correction (norb,norb) into its
        interleaved spin-orbital block-diagonal form (2*norb,2*norb): the
        same value at both spin-diagonal blocks, zero for spin-off-diagonal
        blocks, appropriate for an RHF-reference spin-independent matrix."""
        if static_correction is None:
            return None
        norb = static_correction.shape[0]
        sc_spin = np.zeros((2 * norb, 2 * norb))
        sc_spin[0::2, 0::2] = static_correction
        sc_spin[1::2, 1::2] = static_correction
        return sc_spin

    def dimensions(self, nocc):
        return adc_r_utils.dimensions(self.norb, nocc)

    def build_supermatrix(self, nocc, static_correction=None):
        """(nH, nH) supermatrix at self.level; DF route iff B_aa is set."""
        mod = adc_r_dense_df if self.B_aa is not None else adc_r_dense_full
        return mod.build_supermatrix(self, nocc, static_correction)

    def build_matrix_free_operator(self, nocc, static_correction=None):
        """(aop, diag, dims) sigma-vector operator; DF route iff B_aa is set."""
        mod = adc_r_sigma_df if self.B_aa is not None else adc_r_sigma_full
        return mod.build_operator(self, nocc, static_correction)

    def solve_dense(self, nocc, static_correction=None, threshold=5000):
        """Dense diagonalization (mid-level); (eGF, Z, Reigv) sorted ascending."""
        H = self.build_supermatrix(nocc, static_correction=static_correction)
        return diag_dense(H, self.norb, threshold=threshold)

    # ---- unified entry point ----

    def solve(self, static_correction=None, nroots=1, homo_index=None,
              ref_vec=None, conv_tol=1e-6, tol=1e-8, threshold=5000, verbose=0,
              method='davidson', omega_range=None):
        """(e, Z) for the configured route; eigenvectors on self.last_result.
        Dense routes return all poles; matrix-free routes the nroots
        root-followed ones (Koopmans guess at homo_index, default the HOMO).

        method='lanczos': matrix-free Lanczos/continued-fraction spectral
        solve instead of Davidson root-following -- needs matrix_free=True and
        omega_range=(omega_lo, omega_hi) (Hartree, same sign convention as
        diag/eigenvalues elsewhere: negative for IP removal energies).
        ref_vec (or homo_index) is the PHYSICAL starting channel here (e.g. a
        unit vector on one orbital = that orbital's full removal spectrum),
        not a root-following seed. Returns the peak positions/weights found in
        omega_range as (e, Z); the full spectral function is on
        self.last_result (see solve.lanczos_spectral)."""
        if self.nocc is None:
            raise ValueError("solve() needs nocc (constructed without it -- "
                             "from_arrays(nocc=...), or use the mid-level API)")
        if method == 'lanczos':
            return self._solve_lanczos(static_correction, homo_index, ref_vec,
                                       omega_range)
        if method != 'davidson':
            raise ValueError(f"method={method!r}; expected 'davidson' or 'lanczos'")
        return adc_r_driver.solve(self, static_correction, nroots, homo_index,
                                  ref_vec, conv_tol, threshold, verbose)

    def _solve_lanczos(self, sc, homo_index, ref_vec, omega_range):
        if not self.matrix_free:
            raise ValueError("method='lanczos' needs matrix_free=True")
        if omega_range is None:
            raise ValueError("method='lanczos' needs omega_range=(lo, hi) "
                             "(Hartree, same sign convention as diag)")
        nocc = self.nocc
        homo = homo_index if homo_index is not None else nocc - 1
        op, diag, dims = self.build_matrix_free_operator(
            nocc, static_correction=sc)
        n = dims['nH']
        v0 = np.asarray(ref_vec, float) if ref_vec is not None else None
        if v0 is None:
            v0 = np.zeros(n)
            v0[homo] = 1.0
        out = lanczos_spectral(op, diag, v0, omega_range)
        self.last_result = out
        return out['peak_omega'], out['peak_weight']

