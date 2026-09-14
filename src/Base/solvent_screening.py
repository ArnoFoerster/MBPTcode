"""Solvent-screened Coulomb kernel v -> v + vtilde, vtilde = v chi v.

Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016),
doi:10.1063/1.4946778, embed a quantum solute (region 1) in a polarizable
continuum (region 2).  Because the two regions have no overlapping orbitals,
P_0 is block diagonal and the Dyson equation for W folds exactly onto
region 1 (their Eqs. (11)-(13)):

    W_11    = v_11^tilde + v_11^tilde P_0,11 W_11
    v^tilde = v_11 + v_12 chi_22 v_21                              (Eq. 12)
    chi_22  = P_0,22 + P_0,22 v_22 chi_22                          (Eq. 13)

chi_22 is the *reducible* (interacting) polarizability of the solvent alone.
So the whole embedding is one substitution: build every self-energy exactly as
in the gas phase, but with

    v  ->  v + vtilde ,     vtilde = v chi v ,

the second term being the reaction potential (Eq. 15) felt inside the cavity.
Nothing in the diagrammatics changes, which is why attaching a screening
object to an mf covers *every* self-energy in this code at once -- both
chokepoints that hand out two-electron integrals (dense
get_two_electron_integrals_chemist and DF get_density_fitting_coefficients)
consult it.

Optical, not static
-------------------
chi must be the solvent response at *optical* frequencies: an added electron
or hole is a fast excitation, so only the solvent's electronic degrees of
freedom follow it (eps_infinity = n^2 ~ 1.78 for water), not its nuclear
reorientation (eps ~ 78.36).  `eps` here therefore defaults to n^2 looked up
from pyscf's SMD solvent table, and a value large enough to be a static
constant is rejected unless `allow_static_eps=True`.

The GROUND STATE is the other half of the same recipe, and it is not optional
for either kind of excitation: it relaxes inside PCM(eps_static), because the
solvent nuclei have had time to reorient around a state that is already there.
`mean_field` applies it, so an environment attached to a calculation carries
both constants and a caller cannot take one without the other.  Duchemin,
Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) split a solvatochromic
shift into a ground-state part, obtained by freezing the polarization
(eps_infinity -> 1) while keeping eps_static, and a response part; for a LOCAL
excitation the ground-state part dominates, +0.232 eV of the +0.252 eV that
water puts on acrolein's n->pi*.  It is small for a charged excitation instead
-- water moves its own HOMO by 2.4 meV -- which is why an IP is insensitive to
it and a neutral excitation is not.

Discretization
--------------
vtilde is evaluated with pyscf's PCM apparent-surface-charge machinery: a
source charge distribution produces the potential v_k on cavity surface point
k, the surface responds with charges q = K^-1 R v, and the reaction potential
that those charges generate back inside the cavity is the second term of
Eq. (12).  With the symmetrized response Q = (K^-1 R + (K^-1 R)^T)/2 that
pyscf itself uses,

    vtilde(rho_1, rho_2) = sum_kk' v_k[rho_1] Q_kk' v_k'[rho_2] ,

which is negative semi-definite (screening lowers the interaction) and only
needs the one-electron grid potentials v_k, never a four-index object.
"""
import numpy as np
import scipy.linalg

from pyscf import df as pyscf_df
from pyscf import gto
from pyscf import solvent as pyscf_solvent
from pyscf.solvent import pcm as pyscf_pcm
from pyscf.solvent.smd import solvent_db

from src.Base.constants import ISDF_TILE_GB
from src.Base.environment import attach_environment, environment_of
from src.Base.separable_ri import auxmol_key

# n^2 below this is not a plausible optical dielectric constant for a
# condensed phase; above it the caller almost certainly passed a static eps by
# mistake (water: 1.78 optical vs 78.36 static).
_MAX_PLAUSIBLE_OPTICAL_EPS = 10.0
# I + M (the screened Coulomb metric in the whitened auxiliary basis) must stay
# positive definite: v + vtilde is still a positive kernel, so an eigenvalue at
# or below this floor means the PCM discretization has over-screened.
_MIN_SCREENED_METRIC_EIGENVALUE = 1e-6


def solvent_dielectrics(name):
    """(eps_optical, eps_static) for a named solvent, from pyscf's SMD table.

    eps_optical = n^2 with n the refractive index -- the constant that screens
    a fast (photoemission/optical) excitation, and the one this module wants.
    eps_static additionally contains the nuclear reorientation of the solvent
    and belongs in the *ground-state* SCF, not here.
    """
    key = name.lower()
    if key not in solvent_db:
        raise KeyError(
            f"unknown solvent {name!r}; pyscf's SMD table has "
            f"{len(solvent_db)} entries, e.g. 'water', 'acetonitrile', "
            f"'toluene' -- or pass eps= explicitly")
    descriptors = solvent_db[key]
    return descriptors[0] ** 2, descriptors[5]


def resolve_optical_eps(eps=None, solvent=None, allow_static_eps=False):
    """(eps_optical, eps_static_or_None) from either an explicit eps or a
    solvent name, with the optical-vs-static guard.

    Split out from SolventScreening so every caller rejects a static constant
    the same way.
    """
    if (eps is None) == (solvent is None):
        raise ValueError("pass exactly one of eps= or solvent=")
    if solvent is not None:
        eps, eps_static = solvent_dielectrics(solvent)
    else:
        eps_static = None
    if eps < 1.0:
        raise ValueError(f"eps = {eps} < 1 is not a dielectric constant")
    if eps > _MAX_PLAUSIBLE_OPTICAL_EPS and not allow_static_eps:
        raise ValueError(
            f"eps = {eps} looks like a *static* dielectric constant. The "
            f"screening of an added electron or hole is optical: use "
            f"eps = n^2 (water 1.78, not 78.36) -- solvent_dielectrics() "
            f"returns both. Pass allow_static_eps=True to force it (that "
            f"is the equilibrium limit, correct only for a process slow "
            f"enough for the solvent nuclei to relax).")
    return eps, eps_static


class SolventScreening:
    """The reaction-field kernel vtilde = v chi v of a PCM continuum.

    Hands out vtilde in whichever representation a consumer needs:
      * `kernel_mo`      -- dense (pq|rs) chemist-order correction, for the
                            four-index route;
      * `whitened_transform` -- the (naux, naux) matrix T with
                            B -> T B turning a Coulomb-metric RI factor into
                            one that reproduces (pq|v + vtilde|rs), for the DF
                            route;
      * `aux_kernel`     -- vtilde_PQ itself, which dresses the metric of a
                            separable factorization.

    All come from the same surface response, so the routes agree to the RI
    error of the underlying B (tests/test_solvent_screening.py pins this). The
    class satisfies `src.Base.environment.Environment`: `for_geometry` rebuilds
    the cavity around displaced atoms and `static_self_energy` is the static
    COHSEX operator of `cohsex_correction`.
    """

    def __init__(self, mol, eps=None, solvent=None, method='IEF-PCM',
                 lebedev_order=29, vdw_scale=1.2, r_probe=0.0,
                 radii_table=None, allow_static_eps=False, eps_static=None):
        """eps: optical dielectric constant. Give this or `solvent` (a name in
        pyscf's SMD table, whose refractive index sets eps = n^2), not both.
        method/lebedev_order/vdw_scale/r_probe/radii_table are handed straight
        to pyscf's PCM and carry its meanings and defaults.

        eps_static: the constant the GROUND STATE relaxes in, which `mean_field`
        puts the SCF inside. A solvent name supplies it; an explicit optical eps
        needs it explicitly, and without one the ground state stays bare -- the
        frozen-polarization limit rather than a solvated calculation.

        THE STATIC ONE-BODY TERM IS NOT OPTIONAL. A continuum acts through two
        channels -- the reaction field of the transition density, which reaches
        a response kernel through the dressed metric, and the reaction field of
        the density DIFFERENCE, which shifts the one-body energies -- and
        neither is meaningful alone: the first misses charge transfer entirely,
        the second misses a bright local excitation. So there is no switch.
        """
        if hasattr(mol, 'lattice_vectors'):
            raise NotImplementedError(
                "PCM screening is molecular only -- a periodic Cell has no "
                "cavity to carve.")
        eps, named_static = resolve_optical_eps(eps, solvent, allow_static_eps)
        if eps_static is None:
            eps_static = named_static
        elif eps_static < eps:
            raise ValueError(
                f"eps_static = {eps_static} below the optical eps = {eps}: the "
                f"static constant contains the solvent's nuclear reorientation "
                f"on top of its electronic response and cannot be the smaller.")

        self.mol = mol
        self.eps = eps
        self.eps_static = eps_static
        self.solvent = solvent
        self.method = method
        self._cavity = dict(lebedev_order=lebedev_order, vdw_scale=vdw_scale,
                            r_probe=r_probe, radii_table=radii_table)

        self._pcm = pyscf_pcm.PCM(mol)
        self._pcm.eps = eps
        self._pcm.method = method
        self._pcm.lebedev_order = lebedev_order
        self._pcm.vdw_scale = vdw_scale
        self._pcm.r_probe = r_probe
        self._pcm.radii_table = radii_table
        self._pcm.verbose = 0
        self._pcm.build()

        self._response = None
        self._v_ao = None
        self._aux_cache = {}
        self._transform_cache = {}

    # ---- surface response -------------------------------------------------

    @property
    def ngrids(self):
        return self._pcm.surface['grid_coords'].shape[0]

    def response_matrix(self):
        """Symmetrized apparent-surface-charge response Q: q = Q v.

        pyscf solves q = K^-1 R v and symmetrizes as (q + R^T K^-T v)/2; the
        same symmetrization here makes vtilde exactly symmetric under
        (pq) <-> (rs), which the four-index and DF consumers both require.
        Negative semi-definite: a positive test charge induces negative
        surface charge, so the reaction potential screens.
        """
        if self._response is None:
            K = self._pcm._intermediates['K']
            R = self._pcm._intermediates['R']
            KiR = np.linalg.solve(K, R)
            self._response = 0.5 * (KiR + KiR.T)
        return self._response

    def _fakemol(self, grids=None):
        """Gaussian-smeared surface point charges -- the same regularized
        charges pyscf's PCM uses in _get_v/_get_vmat, so our grid potentials
        and its K/R matrices refer to one and the same discretization.

        `grids` restricts it to a slice of the surface: each grid point is one
        s-function, so slicing the points slices the third index of every
        potential below exactly.
        """
        surface = self._pcm.surface
        crd, expnt = surface['grid_coords'], surface['charge_exp']
        if grids is not None:
            crd, expnt = crd[grids], expnt[grids]
        return gto.fakemol_for_charges(crd, expnt=expnt ** 2)

    # ---- grid potentials of the source distributions ----------------------

    def grid_blocks(self, nbasis, live=3, budget_gb=ISDF_TILE_GB):
        """Slices of the surface grid whose (nbasis, nbasis, nk) potentials fit.

        The grid potential is the one object in this module that scales as
        nao^2 ngrids -- ~10 GB at 100 atoms in a double-zeta basis -- and the
        COHSEX correction wants three of them at once. `live` is how many are
        held per block.
        """
        per = 8.0 * nbasis * nbasis * max(live, 1)
        nk = max(1, int(budget_gb * 1024 ** 3 / per))
        return [(k0, min(k0 + nk, self.ngrids))
                for k0 in range(0, self.ngrids, nk)]

    def ao_grid_potential(self, mol, grids=None):
        """v_k[phi_mu phi_nu], shape (nao, nao, ngrids).

        The WHOLE grid is cached: this is the one genuinely expensive integral
        in the module. `grids` asks for one slice of it instead, which is what
        a streamed consumer wants -- a slice of the cache when the whole thing
        already exists, and otherwise the integrals of that slice alone, so the
        dense array is never built for a caller that does not need it.
        """
        if grids is not None:
            if self._v_ao is not None:
                return self._v_ao[:, :, grids]
            self._check_mol(mol)
            return pyscf_df.incore.aux_e2(mol, self._fakemol(grids),
                                          intor='int3c2e', aosym='s1')
        if self._v_ao is None:
            self._check_mol(mol)
            self._v_ao = pyscf_df.incore.aux_e2(
                mol, self._fakemol(), intor='int3c2e', aosym='s1')
        return self._v_ao

    def mo_grid_potential(self, mol, mo_coeff, budget_gb=ISDF_TILE_GB):
        """v_k[phi_p phi_q], shape (nmo, nmo, ngrids).

        Transformed grid block by grid block, so the AO potential -- the same
        size, and a second copy of it -- never has to exist beside the answer.
        """
        mo_coeff = np.asarray(mo_coeff, float)
        nmo = mo_coeff.shape[1]
        out = np.empty((nmo, nmo, self.ngrids))
        for k0, k1 in self.grid_blocks(max(nmo, mo_coeff.shape[0]),
                                       budget_gb=budget_gb):
            v_ao = self.ao_grid_potential(mol, grids=slice(k0, k1))
            half = np.tensordot(mo_coeff, v_ao, axes=(0, 0))    # (p, nu, k)
            out[:, :, k0:k1] = np.tensordot(mo_coeff, half,
                                            axes=(0, 1)).transpose(1, 0, 2)
        return out

    def aux_grid_potential(self, auxmol):
        """v_k[chi_P] for auxiliary basis functions, shape (naux, ngrids) --
        the paper's ingredient for its Eq. (16) vtilde_{beta beta'}."""
        return gto.mole.intor_cross('int2c2e', auxmol, self._fakemol())

    # ---- vtilde in the two representations consumers need -----------------

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """vtilde as a dense chemist-order (pq|rs) correction.

        Rows (pq) are built from `mo_bra`, columns (rs) from `mo_ket` (default:
        the same), so the UHF aa / ab / bb blocks each get their own call.
        """
        v_bra = self.mo_grid_potential(mol, mo_bra)
        v_ket = (v_bra if mo_ket is None or mo_ket is mo_bra
                 else self.mo_grid_potential(mol, mo_ket))
        n_bra, n_ket, ng = v_bra.shape[0], v_ket.shape[0], self.ngrids
        screened = v_bra.reshape(-1, ng) @ self.response_matrix()
        return (screened @ v_ket.reshape(-1, ng).T).reshape(
            n_bra, n_bra, n_ket, n_ket)

    def kernel_ao(self, mol):
        """vtilde as a dense chemist-order (mu nu|lambda sigma) correction."""
        v_ao = self.ao_grid_potential(mol)
        nao, ng = v_ao.shape[0], self.ngrids
        screened = v_ao.reshape(-1, ng) @ self.response_matrix()
        return (screened @ v_ao.reshape(-1, ng).T).reshape(nao, nao, nao, nao)

    def cohsex_correction(self, mol, mo_coeff, nocc,
                          budget_gb=ISDF_TILE_GB):
        """Static COHSEX self-energy of the reaction field, (nmo, nmo) in the MO
        basis of `mo_coeff` -- the *first order in vtilde* term that the
        v -> v + vtilde substitution cannot generate by itself.

        Why it is needed.  Every correlated method here is built on a mean
        field whose exchange operator is the BARE Sigma_x = -sum_occ v: the ADC
        secular matrix starts at Sigma^(2), and the GW quasiparticle equation
        adds only Sigma_c on top of eps_HF.  Substituting v -> v + vtilde
        therefore reaches those methods only at order vtilde*v and beyond --
        the leading reaction-field term drops out, and with it essentially all
        of the polarization energy.  What is missing is exactly the paper's
        static COHSEX operator (its Eqs. (20)-(22)) evaluated with vtilde in
        place of (W - v):

            Sigma^SEX_pq = - sum_i^occ  (p i|vtilde|i q)
            Sigma^COH_pq = 1/2 sum_n^all (p n|vtilde|n q)     [sum_n |n><n| ~ delta]

        so that, writing the two together,

            Sigma^solv = 1/2 ( sum_a^virt - sum_i^occ ) (p .|vtilde|. q) .

        The sign structure is the classical image-charge result: with vtilde
        locally constant at -lambda the diagonal is +lambda/2 for an occupied
        level and -lambda/2 for a virtual one, i.e. the Born stabilization
        -q^2/2a (1 - 1/eps) of both the cation and the anion.  The IP drops and
        the EA rises by the same amount; the gap closes.

        Add it to the static (one-body, frequency-independent) part of whatever
        self-energy is being solved -- Sigma(infinity) for ADC,
        `xc_correction` for the GW quasiparticle equation.  It is first order
        in vtilde and zeroth order in v, so it does not double count anything
        the substituted interaction produces (those terms all carry at least
        one bare v).

        nocc: occupied orbital count *in this spin channel* (for a closed-shell
        restricted reference, the doubly-occupied count -- exchange is
        same-spin, so there is no factor of two).

        The response matrix couples every grid point to every other, so the MO
        potential is held whole; what is streamed is the potential it is built
        from and the responded copy, which is the difference between three
        (nmo, nmo, ngrids) arrays at once and one plus a block.
        """
        v_mo = self.mo_grid_potential(mol, mo_coeff, budget_gb=budget_gb)
        nmo = v_mo.shape[0]
        if not 0 <= nocc <= nmo:
            raise ValueError(f"nocc={nocc} outside [0, {nmo}]")
        Q = self.response_matrix()
        virt = np.zeros((nmo, nmo))
        occ = np.zeros((nmo, nmo))
        for k0, k1 in self.grid_blocks(nmo, live=1, budget_gb=budget_gb):
            responded = np.tensordot(v_mo, Q[k0:k1], axes=(2, 1))
            virt += np.einsum('pak,aqk->pq', responded[:, nocc:],
                              v_mo[nocc:, :, k0:k1], optimize=True)
            occ += np.einsum('pik,iqk->pq', responded[:, :nocc],
                             v_mo[:nocc, :, k0:k1], optimize=True)
        return 0.5 * (virt - occ)

    def aux_kernel(self, auxmol):
        """vtilde_{PQ} between auxiliary functions -- the paper's Eq. (16)."""
        key = auxmol_key(auxmol)
        if key not in self._aux_cache:
            v_aux = self.aux_grid_potential(auxmol)
            self._aux_cache[key] = v_aux @ self.response_matrix() @ v_aux.T
        return self._aux_cache[key]

    def whitened_transform(self, mol, mf):
        """T with (B -> T B) turning a Coulomb-fitted RI factor into a
        (v + vtilde)-fitted one.

        With E_P,pq = (P|pq), J_PQ = (P|Q) and the RI-V ansatz that every pair
        density is replaced by its Coulomb-metric fit c = J^-1 E, substituting
        v -> v + vtilde changes only the auxiliary-basis metric:

            (pq|v + vtilde|rs) = c_pq^T (J + vtilde_aux) c_rs .

        pyscf's cderi is B = L^-1 E with L the lower Cholesky factor of J, so
        c = L^-T B and the whole substitution collapses to a single naux x naux
        congruence of the *whitened* Coulomb metric (which is the identity in
        the gas phase -- hence T = I when vtilde = 0):

            (pq|v + vtilde|rs) = B_pq^T (I + M) B_rs ,  M = L^-1 vtilde_aux L^-T
            T = (I + M)^(1/2) .

        Every DF consumer downstream keeps working unchanged: it still sees a
        plain three-index factor whose square is the interaction.
        """
        auxmol = auxmol_of(mf, mol)
        key = auxmol_key(auxmol)
        if key in self._transform_cache:
            return self._transform_cache[key]

        j2c = auxmol.intor('int2c2e')
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
        except scipy.linalg.LinAlgError as err:
            raise RuntimeError(
                f"the auxiliary Coulomb metric of {auxmol.basis} is not "
                f"positive definite, so pyscf's cderi did not come from a "
                f"Cholesky factorization and the whitened transform below "
                f"would be built against the wrong factor. Use a "
                f"better-conditioned auxiliary basis, or run without DF.") from err
        _verify_cderi_is_cholesky(mol, mf, auxmol, low)

        v_tilde = self.aux_kernel(auxmol)
        M = scipy.linalg.solve_triangular(low, v_tilde, lower=True)
        M = scipy.linalg.solve_triangular(low, M.T, lower=True).T
        M = 0.5 * (M + M.T)

        w, U = np.linalg.eigh(np.eye(M.shape[0]) + M)
        if w.min() < _MIN_SCREENED_METRIC_EIGENVALUE:
            raise RuntimeError(
                f"the screened Coulomb metric I + M has a smallest eigenvalue "
                f"of {w.min():.3e}, i.e. the PCM reaction field over-screens "
                f"the bare interaction to the point of making it indefinite. "
                f"v + vtilde must stay a positive kernel. Check eps={self.eps} "
                f"and the cavity (lebedev_order/vdw_scale).")
        transform = (U * np.sqrt(w)) @ U.T
        self._transform_cache[key] = transform
        return transform

    # ---- the Environment contract -----------------------------------------

    screens = True

    def for_geometry(self, mol):
        """The same continuum around the atoms of `mol`.

        The cavity moves with the atoms, so a displaced geometry needs its own
        surface; `_check_mol` compares only natm and nao and would let the
        reference cavity pass silently. The same molecule object gets itself.
        """
        if mol is self.mol:
            return self
        dielectric = (dict(solvent=self.solvent) if self.solvent is not None
                      else dict(eps=self.eps, allow_static_eps=True,
                                eps_static=self.eps_static))
        return SolventScreening(mol, method=self.method,
                                **dielectric, **self._cavity)

    def mean_field(self, mol, scf_factory):
        """The factory's mean field inside the ground-state reaction field.

        Non-equilibrium solvation is two dielectric constants, not one. The
        ground state relaxes in a continuum at eps_static, because the solvent
        nuclei have had time to reorient around it; only the RESPONSE to a fast
        excitation is optical, and that is the vtilde this class hands out
        everywhere else. Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9,
        4430 (2018) split a solvatochromic shift into those two, and for a
        local excitation the ground-state half is the larger one: of the
        +0.252 eV water puts on acrolein's n->pi*, their BSE assigns +0.232 to
        it. Leaving it out is their frozen-polarization Omega_0, not Omega.

        A factory that already converged its own PCM at eps_static is returned
        as it is, so the reaction field is applied once; one at any other
        constant relaxed a DIFFERENT ground state and raises. Otherwise the mean
        field is wrapped and re-converged from its own density.
        """
        mf = scf_factory(mol)
        if hasattr(mf, 'with_solvent'):
            if (self.eps_static is not None
                    and abs(mf.with_solvent.eps - self.eps_static) > 1e-8):
                raise ValueError(
                    f'the factory converged its SCF inside PCM(eps = '
                    f'{mf.with_solvent.eps}) but this environment relaxes the '
                    f'ground state at eps_static = {self.eps_static}: those '
                    f'are two different ground states, and keeping the '
                    f'factory one would put the optical response at eps = '
                    f'{self.eps} on top of the wrong reaction field.')
            return mf
        if self.eps_static is None:
            return mf
        wrapped = pyscf_solvent.PCM(mf)
        wrapped.with_solvent.method = self.method
        wrapped.with_solvent.eps = self.eps_static
        wrapped.with_solvent.lebedev_order = self._cavity['lebedev_order']
        wrapped.with_solvent.vdw_scale = self._cavity['vdw_scale']
        wrapped.with_solvent.r_probe = self._cavity['r_probe']
        wrapped.with_solvent.radii_table = self._cavity['radii_table']
        wrapped.kernel(dm0=mf.make_rdm1())
        if not wrapped.converged:
            raise RuntimeError(
                f'the SCF did not re-converge inside PCM(eps = '
                f'{self.eps_static}); no ground state follows from it')
        return wrapped

    def static_self_energy(self, mf, mol=None):
        """Sigma^solv of `cohsex_correction` in the MO basis of `mf`.

        RHF: one (nmo, nmo) array. UHF: an (alpha, beta) pair, each from its
        own coefficients and occupation. Never None for a continuum: the term
        is not optional (see __init__).
        """
        mol = mol if mol is not None else mf.mol
        mo_coeff = mf.mo_coeff
        if isinstance(mo_coeff, (tuple, list)) or np.asarray(mo_coeff).ndim == 3:
            mo_a, mo_b = mo_coeff
            nocc_a, nocc_b = mf.nelec
            return (self.cohsex_correction(mol, mo_a, nocc_a),
                    self.cohsex_correction(mol, mo_b, nocc_b))
        return self.cohsex_correction(mol, mo_coeff, mol.nelectron // 2)

    # ---- housekeeping -----------------------------------------------------

    def _check_mol(self, mol):
        if mol.nao != self.mol.nao or mol.natm != self.mol.natm:
            raise ValueError(
                f"screening was built for a molecule with {self.mol.natm} "
                f"atoms / {self.mol.nao} AOs but is being asked for integrals "
                f"over one with {mol.natm} / {mol.nao}. Re-attach the "
                f"screening to the mean field you are actually running.")

    def __repr__(self):
        label = f"solvent={self.solvent!r}, " if self.solvent else ""
        return (f"SolventScreening({label}eps={self.eps:.4f} (optical), "
                f"method={self.method!r}, ngrids={self.ngrids})")


def auxmol_of(mf, mol=None):
    """The auxiliary Mole behind mf.with_df (built on demand if pyscf has not
    materialized it yet)."""
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError("mf carries no with_df -- there is no auxiliary basis")
    if getattr(with_df, 'auxmol', None) is not None:
        return with_df.auxmol
    return pyscf_df.addons.make_auxmol(mol if mol is not None else mf.mol,
                                       with_df.auxbasis)


def _verify_cderi_is_cholesky(mol, mf, auxmol, low):
    """Assert mf's cderi really is L^-1 (P|mu nu) for the L we just built.

    whitened_transform's algebra is exact for that convention and *silently
    wrong* for any other choice of fitting factor X with X^T X = J^-1 (they
    differ by an orthogonal rotation of the auxiliary index, which leaves the
    gas-phase ERIs invariant and so cannot be caught by any ERI check). One
    AO pair is enough to pin the rotation down, so this costs a single
    int3c2e shell block.
    """
    cderi_head = next(iter(mf.with_df.loop(blksize=auxmol.nao)))
    naux = auxmol.nao
    if cderi_head.shape[0] > naux:
        raise RuntimeError(
            f"cderi has more auxiliary vectors ({cderi_head.shape[0]}) than "
            f"the auxiliary basis has functions ({naux})")
    got = np.zeros(naux)
    offset = 0
    for block in mf.with_df.loop(blksize=naux):
        # _cderi is packed s2ij over AO pairs; column 0 is the (0,0) pair.
        got[offset:offset + block.shape[0]] = block[:, 0]
        offset += block.shape[0]
    if offset != naux:
        raise RuntimeError(
            f"cderi has {offset} auxiliary vectors but the auxiliary basis has "
            f"{naux}: pyscf dropped linearly dependent auxiliary functions, so "
            f"its fitting factor is not the Cholesky factor of int2c2e that "
            f"whitened_transform inverts against. Use a better-conditioned "
            f"auxiliary basis, or run the dense (df=False) route.")

    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                                 shls_slice=(0, 1, 0, 1, 0, auxmol.nbas))
    expected = scipy.linalg.solve_triangular(low, e3c[0, 0], lower=True)
    scale = max(np.abs(expected).max(), 1.0)
    if not np.allclose(got, expected, rtol=0, atol=1e-8 * scale):
        raise RuntimeError(
            f"mf.with_df's fitting factor is not L^-1 (P|mu nu) for the "
            f"Cholesky L of int2c2e (max deviation "
            f"{np.abs(got - expected).max():.3e}). whitened_transform's "
            f"algebra assumes pyscf's default DF convention; this mean field "
            f"uses a different one, so screening it through the DF route "
            f"would be silently wrong. Run the dense (df=False) route.")


# ---- attachment: one object on mf, consulted by both integral chokepoints --

def attach_solvent_screening(mf, eps=None, solvent=None, method='IEF-PCM',
                             mol=None, **kwargs):
    """Make every post-SCF two-electron integral this code hands out use
    v + vtilde instead of v, and return `mf`.

    This is a *post-SCF* substitution: it does not touch mf's orbitals, orbital
    energies or Fock matrix. Ground-state polarization belongs in the SCF via
    pyscf's own `solvent.PCM(mf)` at the static eps; this adds the optical
    response of the solvent to the correlated part, which is exactly the
    non-equilibrium split of the paper (its Section II D).

    Keyword arguments go to SolventScreening.
    """
    screening = SolventScreening(mol if mol is not None else mf.mol,
                                 eps=eps, solvent=solvent, method=method,
                                 **kwargs)
    return attach_environment(mf, screening)


def detach_solvent_screening(mf):
    """Drop the screening, returning mf to gas-phase integrals."""
    mf.with_screening = None
    return mf


def solvent_static_selfenergy(mf, mol=None):
    """The static one-body term of the environment attached to `mf`, in that
    mean field's own MO basis, or None in the gas phase (so callers add it
    unconditionally, the way build_ks_static_correction is added)."""
    return environment_of(mf).static_self_energy(mf, mol)

