"""The frozen ISDF factorization a cubic gradient chain differentiates, and the
three branches that carry adjoints on its factors to the nuclei.

Every target on the space-time route -- a quasiparticle energy, a BSE
excitation, the dRPA correlation energy -- is a function of the orbital
energies eps, the collocation X_mo = X_ao C and the auxiliary factor D, and its
reverse pass ends with adjoints (eps_bar, X_bar, D_bar). From there on nothing
depends on the target:

    eps_bar, and X_bar through C          -> the orbital-response Lagrangian
    X_bar through the interpolation points -> the collocation adjoint
    D_bar                                  -> the fit adjoint

Neither does what has to be decided ONCE, at the reference geometry, for the
surface to be smooth at all: the grid radii, the pair layout and the frames.
Re-deciding any of them per geometry puts a step into the energy that no
gradient can follow.

The surroundings enter through ONE object, the chain's `environment`
(src.Base.environment): it builds the mean field at each geometry, dresses the
auxiliary gauge, supplies the static one-body term production adds, and owns
the adjoints of both. A chain therefore cannot see half of a reaction field.
"""
import time
import warnings
import collections
import weakref
from contextlib import contextmanager

import numpy as np
from pyscf import df as pyscf_df

from src.Base.isdf_jk import ISDFJK, mean_field_skeleton_force
from src.Base.constants import (ENVIRONMENT_CACHE_SIZE, ISDF_FIT_ERROR_FAILED,
                                SCF_GRAD_TOL, THREE_CENTER_BLOCK_BYTES)
from src.Base.environment import dresses_interaction, resolve_environment
from src.Base.separable_ri import (atomic_frames, aux_metric_sqrt, fit_M_stable,
                                   optimize_atomic_radii, resolve_isdf_grid,
                                   subshells, test_set_D, test_set_layout)
from src.SingleReference.GW.space_time import DEFAULT_COUNTS
from src.SingleReference.LinearResponse.rpa_energy import reference_energy
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            continued_frames,
                                            dfactor_adjoint_gauges,
                                            GaugeAdjoint,
                                            shell_blocks,
                                            eps_chain_gradient,
                                            exx_double_counting_Y,
                                            exx_double_counting_skeleton,
                                            point_layout)


def converged_factory(scf_factory):
    """`scf_factory` completed with `mf.kernel()` when it returns a mean field
    that has been built but not run, so a factory may hand back the cheaper of
    the two and let the chain finish the SCF."""
    def build(mol):
        mf = scf_factory(mol)
        if getattr(mf, 'mo_coeff', None) is None:
            mf.kernel()
        return mf

    return build


def radii_tag(radii):
    """Hashable summary of an explicit radii set, or None when the atomic
    optimizer decides them: two factorizations differing only in their radii are
    different functionals and must not be allowed to share."""
    if radii is None:
        return None
    return tuple((el, tuple(np.round(np.concatenate(
        [np.atleast_1d(radii[el][s]) for s in sorted(radii[el])]), 12)))
        for el in sorted(radii))


class FrozenFactorization:
    """The ISDF conventions fixed at ONE reference geometry, so that several
    chains are ONE functional rather than two that happen to agree.

    Radii, interpolation points, frames and pair layout are discrete choices.
    Re-deciding them per geometry puts a step in the energy; deciding them
    twice, once per chain, is subtler -- `separable_factors` is deterministic
    given the settings, so two chains handed matching settings produce bitwise
    identical factors and merely pay twice. They diverge silently the moment
    the settings stop matching: different `counts` or `n_start`, or
    `frames='continued'`, which carries orientation history from wherever each
    chain started.

    Holding them in one object makes the sharing structural rather than a
    property of the caller passing the same keywords, and a composed surface
    then costs ONE fit per geometry instead of one per chain.

    It owns exactly what `separable_factors` consumes and nothing else. The
    QUADRATURES ARE NOT HERE: E_c^dRPA integrates chi0 over frequency and the
    self-energy over imaginary time, different integrands with no reason to
    share a grid, so each chain keeps its own. Nor is the quasiparticle window,
    which belongs to whichever chain has one.

    The distinction that keeps this boundary from drifting: `refreeze` carries
    a quadrature so that it does not CHANGE BETWEEN GEOMETRIES, which is not
    the same requirement as being SHARED BETWEEN CHAINS. Only the second is
    this object's business.

    `grid_accuracy` is the named way to fix `counts` and `n_start` together --
    a validated accuracy level or four explicit shell counts, resolved against
    the shipped radii table by `resolve_isdf_grid`, which refuses a grid nobody
    optimized rather than re-optimizing one per geometry.
    """

    def __init__(self, mol, basis=None, auxbasis=None, counts=None, n_start=1,
                 frames='frozen', radii=None, grid_accuracy=None):
        self.basis = basis or str(mol.basis)
        self.auxbasis = auxbasis or (self.basis + '-ri')
        elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
        # The named way in. It is resolved HERE and stored as plain counts, so
        # the whole walk -- `rebuilt_at`, `settings`, `require_match` -- keeps
        # comparing one kind of object and a level cannot mean two grids.
        if grid_accuracy is not None:
            counts, n_start = resolve_isdf_grid(grid_accuracy, self.basis,
                                                elements, auxbasis=self.auxbasis)
        self.counts = counts or DEFAULT_COUNTS
        self.n_start = n_start
        self.frames_mode = frames
        self.with_frames = frames == 'continued'
        # Explicit radii -- `separable_ri.tailor_grid`'s, say -- replace the
        # per-element atomic optimization. They are frozen here like every other
        # discrete choice, so a grid tailored at the reference geometry is the
        # one the whole walk uses.
        self.radii_tag = radii_tag(radii)
        if radii is not None and not set(elements) <= set(radii):
            raise ValueError(f'radii given for {sorted(radii)} but the molecule '
                             f'holds {elements}')
        # The fit error is kept, not discarded: it is the only thing that says
        # whether this grid resolves this basis, and the default count is sized
        # for double zeta. A caller who does not pass `counts` gets a FAILED fit
        # for most elements at any larger basis, and nothing else would say so.
        self.fit_errors = {}
        if radii is not None:
            self.radii = radii
        else:
            found, errors = {}, {}
            for el in elements:
                found[el], errors[el] = optimize_atomic_radii(
                    el, self.basis, self.auxbasis, counts=self.counts,
                    n_start=n_start)
            self.radii, self.fit_errors = found, errors
            failed = {el: e for el, e in self.fit_errors.items()
                      if e >= ISDF_FIT_ERROR_FAILED}
            if failed:
                npts = sum(len(subshells()[n]) * len(r)
                           for n, r in next(iter(self.radii.values())).items())
                warnings.warn(
                    f'ISDF atomic grid FAILED its fit for '
                    f'{", ".join(f"{el} {e:.2f}" for el, e in sorted(failed.items()))} '
                    f'at {self.basis}/{npts} points per atom: a fit error near 1 '
                    f'is a failed factorization, not a coarse grid, and every '
                    f'quantity built on it is unreliable. The lever is the point '
                    f'COUNT -- pass counts= with a larger grid; the default is '
                    f'sized for double zeta.', RuntimeWarning, stacklevel=2)
        # The placed points, the frames and the pair layout follow from the
        # radii and the geometry.
        self.pts_local, self.owner = point_layout(mol, self.radii)
        self.frames = atomic_frames(mol)[0]
        self.M = len(self.owner)
        self.naux = self.auxmol(mol).nao_nr()
        crd = self.coords(mol)
        self.layout = test_set_layout(mol, crd)
        # Weak on the Mole: the value is arrays and pins nothing, so an entry
        # dies with its geometry. A weak key is wrong wherever the value
        # references the key -- `_environment_cache` holds `env.mol is mol`.
        self._fit_cache = weakref.WeakKeyDictionary()

    def rebuilt_at(self, mol):
        """The same conventions re-derived at `mol`.

        Everything this object OWNS is a choice and travels verbatim, radii
        included: an explicit set IS the choice and has nothing to re-derive,
        where atomic radii are element-only and rebuild identically. Only what
        depends on the geometry -- the placed points, the frames, the pair
        layout -- is derived again. Callers rebuild through here so that a
        setting added above cannot be silently dropped by one of them.
        """
        return type(self)(mol, basis=self.basis, auxbasis=self.auxbasis,
                          counts=self.counts, n_start=self.n_start,
                          frames=self.frames_mode,
                          radii=self.radii if self.radii_tag is not None else None)

    def auxmol(self, mol):
        """The auxiliary molecule of the frozen auxiliary basis at `mol`."""
        return pyscf_df.addons.make_auxmol(mol, auxbasis=self.auxbasis)

    def coords(self, mol):
        """Interpolation points r_g = p_g F_i + R_i on frozen or continued frames."""
        fr = continued_frames(mol, self.frames) if self.with_frames \
            else self.frames
        return np.vstack([self.pts_local[ia] @ fr[ia] + mol.atom_coord(ia)
                          for ia in range(mol.natm)])

    def shareable_factors(self, mol, auxmol, crd):
        """(X_ao, Mfit, V): the fit at `mol` before the auxiliary gauge.

        The split is drawn before the gauge because `aux_metric_sqrt` dresses
        it with the chain's own environment, so two chains sharing a layout and
        carrying different continua must not share a D. Everything above it is
        environment-independent and cached here, which is most of the
        per-geometry cost, so several chains of one composed surface pay it once.
        """
        hit = self._fit_cache.get(mol)
        if hit is None:
            hit = self._fit(mol, auxmol, crd)
            self._fit_cache[mol] = hit
        return hit

    def _fit(self, mol, auxmol, crd):
        """The least-squares fit itself: (X_ao, M, V) at `mol` on the frozen layout."""
        mu_i, nu_i, wc_l = self.layout
        naux = auxmol.nao_nr()
        Dt = test_set_D(mol, auxmol, crd, self.layout)
        V = auxmol.intor('int2c2e', aosym='s1')
        e3 = test_set_three_center(mol, auxmol, mu_i, nu_i)
        F = np.hstack([np.linalg.solve(V, e3.T) * wc_l[None, :],
                       np.eye(naux)])
        return (mol.eval_gto('GTOval_sph', crd), fit_M_stable(Dt, F), V)

    def settings(self):
        """What two chains must agree on to be allowed to share one of these."""
        return (self.basis, self.auxbasis, tuple(sorted(self.counts.items()))
                if isinstance(self.counts, dict) else tuple(np.ravel(self.counts)),
                self.n_start, self.frames_mode, self.radii_tag)

    def require_match(self, basis, auxbasis, counts, n_start, frames, radii=None):
        """Refuse a chain whose own settings contradict this factorization.

        Silently resolving to one side is the failure the object exists to
        prevent: the caller asked for a factorization it is not getting, and
        every number downstream would be of a functional nobody requested.
        """
        other = FrozenFactorization.__new__(FrozenFactorization)
        other.basis = basis or self.basis
        other.auxbasis = auxbasis or (other.basis + '-ri')
        other.counts = counts or DEFAULT_COUNTS
        other.n_start, other.frames_mode = n_start, frames
        # a chain with no opinion on radii accepts the factorization's
        other.radii_tag = radii_tag(radii) if radii is not None else self.radii_tag
        if other.settings() != self.settings():
            raise ValueError(
                f'this chain asks for {other.settings()} but the shared '
                f'factorization was built for {self.settings()}; build the '
                f'factorization with the settings you want and pass it to '
                f'both chains, or pass none and let each build its own')


class FactorChain:
    """Reference-frozen radii, layout and frames; the factors at any geometry;
    adjoints on (eps, X_mo, D) carried to the nuclei.

    `scf_factory(mol) -> mf` supplies the mean field at a displaced geometry;
    it must converge the orbital gradient to ~1e-11, since the Lagrangian
    assumes the occupied-virtual Fock block vanishes.
    """

    def __init__(self, mol, scf_factory, basis=None, auxbasis=None, counts=None,
                 n_start=1, frames='frozen', mf=None, environment=None,
                 factorization=None, radii=None, grid_accuracy=None):
        self.mol0, self.scf_factory = mol, scf_factory
        # Resolved before the branch, so that building a factorization and
        # matching a shared one are handed the same counts and recipe.
        if grid_accuracy is not None:
            basis = basis or str(mol.basis)
            counts, n_start = resolve_isdf_grid(
                grid_accuracy, basis,
                {mol.atom_pure_symbol(i) for i in range(mol.natm)},
                auxbasis=auxbasis)
        if factorization is None:
            factorization = FrozenFactorization(mol, basis=basis,
                                                auxbasis=auxbasis, counts=counts,
                                                n_start=n_start, frames=frames,
                                                radii=radii)
        else:
            factorization.require_match(basis, auxbasis, counts, n_start, frames,
                                        radii)
        self.factorization = factorization
        self.basis, self.auxbasis = factorization.basis, factorization.auxbasis
        self.counts, self.n_start = factorization.counts, factorization.n_start
        self.frames_mode = factorization.frames_mode
        self.with_frames = factorization.with_frames
        self.nocc = mol.nelectron // 2
        # the one given, else the one attached to the reference mean field,
        # else the gas phase; a given mean field is completed by it (charges)
        self.environment = resolve_environment(environment, mf)
        self._environment_cache = collections.OrderedDict()
        # a dict to accumulate phase timings into, or None for no timing
        self.timer = None
        with self.phase('t_scf'):
            self.mf0 = self.environment.mean_field(
                mol, converged_factory(
                    scf_factory if mf is None else (lambda _mol: mf)))
        self.scf_residual = check_scf_quality(self.mf0, self.nocc)

        # the SAME objects, not copies: two chains on one factorization share
        # the arrays, which is what makes their factors bitwise identical
        self.radii = factorization.radii
        self.pts_local, self.owner = factorization.pts_local, factorization.owner
        self.frames = factorization.frames
        self.M, self.naux = factorization.M, factorization.naux
        self.layout = factorization.layout

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the reference pair, or a fresh SCF at another geometry,
        built in the chain's environment.
        """
        mol = self.mol0 if mol is None else mol
        if mf is None:
            if mol is self.mol0:
                mf = self.mf0
            else:
                with self.phase('t_scf'):
                    mf = self.environment.mean_field(
                        mol, converged_factory(self.scf_factory))
        return mol, mf

    @contextmanager
    def phase(self, key):
        """Time one stage into `self.timer[key]` (accumulating), or do nothing.

        The forward and reverse halves of a gradient are only comparable when
        timed apart, so every stage of a chain runs inside one of these; a
        runner that wants the split sets `timer` to a dict, everyone else pays
        one attribute read.
        """
        if self.timer is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timer[key] = self.timer.get(key, 0.0) + time.perf_counter() - t0

    def require_differentiable_environment(self):
        """Refuse a force this environment cannot complete, before paying for it.

        The refusal itself comes from the environment's own adjoints, which
        raise; asking here means a production-sized reverse pass is not run
        first only to raise at its last branch.
        """
        if not self.environment.differentiable:
            raise NotImplementedError(
                f'{self.environment!r} has no nuclear derivative, so this chain '
                'reports energies and refuses forces: returning the gas-phase '
                'adjoints would give a force that is wrong by the whole '
                'environment response and looks perfectly reasonable. Finite '
                'differences of the energy are exact (each displaced geometry '
                'rebuilds its own environment).')

    @staticmethod
    def is_kohn_sham(mf):
        """True when this mean field's e_tot is a KS energy, not E_HF."""
        return getattr(mf, 'xc', None) is not None

    def reference_energy(self, mol, mf):
        """E_0^HF of the plasmon formula: production's E_HF AT this density.

        `mf.e_tot` on a Hartree-Fock mean field, and on a Kohn-Sham one the
        Hartree-Fock energy of ITS density, which is what the plasmon formula
        wants; the physics is in the production routine. TAKE IT WITH ITS
        GRADIENT: `kohn_sham_gradient_correction` below carries the two terms
        that move the force the same way, and a chain that adopts one and not
        the other is stationary for neither functional.
        """
        return reference_energy(mf, mol)

    def kohn_sham_gradient_correction(self, mol, mf):
        """(y_extra, g_extra): what `reference_energy` costs on the gradient side.

        `reference_energy` moves the surface from E_KS to E_HF at the same
        density. These are the two terms that move its DERIVATIVE the same way
        -- the EXX double-counting orbital response, which enters the Lagrangian
        before the multiplier solve because it shares Lambda, and its skeleton,
        which enters the orbital branch. They are the `y_extra` and `g_extra`
        hooks of `nuclear_gradient`, and those hooks are SHARED: sum into
        whatever a chain already passes rather than replacing it.

        Both are identically zero on a Hartree-Fock mean field, so callers add
        them unconditionally and that path stays bitwise unchanged.

        USE `mean_field_gradient` FOR THE OTHER HALF. `g_extra` is differenced
        against pyscf's own mean-field force, so that force must carry the grid
        response or the cancellation tears.

        A chain with no EXX skeleton of its own does not need these at all: it
        should REFUSE a Kohn-Sham reference outright, which makes the
        inconsistent state unreachable instead of merely documented.
        """
        return (exx_double_counting_Y(mf, self.nocc),
                exx_double_counting_skeleton(mf, mol))

    def mean_field_gradient(self, mf):
        """pyscf's own force for THIS mean field, with the grid response paired.

        The half of `kohn_sham_gradient_correction` that lives outside the
        Lagrangian. `grid_response=True` on a KS gradient is not an accuracy
        knob here: it is the partner of the double-counting skeleton, and
        setting it on one and not the other tears a cancellation the two are
        supposed to complete.
        """
        if isinstance(getattr(mf, 'with_df', None), ISDFJK):
            # `isdf_mean_field_gradient` sets grid_response on the reference it
            # differentiates, so the quadrature's own motion is carried there.
            return mean_field_skeleton_force(mf)
        g0 = mf.Gradients()
        if self.is_kohn_sham(mf):
            g0.grid_response = True
        return np.asarray(g0.kernel())

    def environment_at(self, mol):
        """The environment around THIS geometry, built once per geometry.

        A cavity moves with the atoms, so each displaced geometry gets its own,
        and a finite difference of the energy is the derivative of the real
        surface only because the two displaced points do not share one.

        THE KEY IS THE EXACT BYTES of the atomic charges and coordinates, and
        the nuclear count. Charges belong in it because a cavity follows the
        element radii, so coordinates alone do not determine one. The bytes are
        unrounded because rounding is what lets two finite-difference
        displacements collide and share a reaction field, which is a wrong force
        that no energy or gradient gate would report.

        Content rather than `id(mol)` because identity is safe only by
        accident of what the environments happen to return.
        """
        key = (mol.atom_charges().tobytes(), mol.atom_coords().tobytes(),
               mol.natm)
        hit = self._environment_cache.get(key)
        if hit is not None:
            self._environment_cache.move_to_end(key)
            return hit
        hit = self.environment.for_geometry(mol)
        self._environment_cache[key] = hit
        while len(self._environment_cache) > ENVIRONMENT_CACHE_SIZE:
            self._environment_cache.popitem(last=False)
        return hit

    def auxmol(self, mol):
        """The auxiliary molecule of the frozen auxiliary basis at `mol`."""
        return self.factorization.auxmol(mol)

    def coords(self, mol):
        """Interpolation points r_g = p_g F_i + R_i on frozen or continued frames."""
        return self.factorization.coords(mol)

    def factors(self, mol, auxmol, crd):
        """(X_ao, D) of the separable RI on the frozen layout, D = M^T V^(1/2).

        The least-squares fit target keeps the BARE metric; only the gauge
        V^(1/2) is dressed by the environment at this geometry, as in
        `space_time.separable_factors`.
        """
        X_ao, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return (X_ao,
                Mfit.T @ aux_metric_sqrt(auxmol, self.environment_at(mol), V=V))

    def bare_factor(self, mol, auxmol, crd):
        """D in the BARE gauge, or None when nothing screens.

        The self-energy screens with the bare interaction and takes the
        continuum as Duchemin et al.'s static Eq. (18) shift, while the BSE
        kernel keeps the dressed one, so a solvated chain carries both factors.
        They share the fit M and differ only in the metric root, which is one
        naux eigendecomposition.
        """
        if not dresses_interaction(self.environment_at(mol), auxmol):
            return None
        _, Mfit, V = self.factorization.shareable_factors(mol, auxmol, crd)
        return Mfit.T @ aux_metric_sqrt(auxmol, None, V=V)

    def factors_at(self, mol, mf):
        """(X_mo, D, eps, auxmol, coords, X_ao) at `mol`.

        The factorization every chain starts from, under the one timing phase,
        so the reference-frozen radii, layout and frames are consulted in
        exactly one place rather than once per target.
        """
        auxmol = self.auxmol(mol)
        crd = self.coords(mol)
        with self.phase('t_factors'):
            x_ao, d = self.factors(mol, auxmol, crd)
        return (x_ao @ mf.mo_coeff, d, np.asarray(mf.mo_energy, float),
                auxmol, crd, x_ao)

    def nuclear_gradient(self, mol, mf, auxmol, crd, x_mo, eps_bar, x_bar,
                         d_bar, y_extra=None, g_extra=None, d_bar_bare=None,
                         root_bar=None, kernel_bar=None):
        """(natm, 3) gradient from adjoints on (eps, X_mo, D), with diagnostics.

        X_mo = X_ao C depends on the geometry twice: through C, which enters the
        Lagrangian as the orbital-rotation gradient X_mo^T X_bar, and through the
        collocation X_ao, whose adjoint is X_bar C^T.

        eps_bar: the adjoint on the orbital energies, or a full symmetric MO
            Fock partial dE/dF_pq when the target's one-body term contracts F
            off the diagonal (an active-space model does).
        y_extra: orbital-rotation gradient of a term that is not a function of
            the factors -- the Kohn-Sham static correction, say. It joins Y
            BEFORE the multiplier solve, because it shares Lambda.
        g_extra: that term's own skeleton, added to the orbital branch.
        d_bar_bare: the adjoint on the BARE-gauge factor, when the target
            screens with both (`bare_factor`). It rides the same fit with no
            environment, so the cavity's motion reaches the force through
            the DRESSED factor only -- which is where vtilde actually sits.
        root_bar, kernel_bar: adjoints on the DRESSED gauge's metric root
            (V + vtilde)^(1/2) and on vtilde alone, from a term that reads the
            screened metric WITHOUT going through D. The fold's
            N = R^-1 vtilde R^-1 is that term. They join the dressed gauge so
            that one Frechet solve and one cavity derivative serve both it and
            D; differentiating them apart is the same number and twice the
            reaction-field work.
        """
        y = x_mo.T @ x_bar
        if y_extra is not None:
            y = y + y_extra
        with self.phase('t_orbital'):
            g_orb, diags = eps_chain_gradient(mf, eps_bar, self.nocc, Y_extra=y)
        if g_extra is not None:
            g_orb = g_orb + g_extra
        with self.phase('t_collocation'):
            g_coll = collocation_adjoint(mol, crd, x_bar @ mf.mo_coeff.T,
                                         self.pts_local, self.owner,
                                         frames=self.frames,
                                         with_frames=self.with_frames)
        with self.phase('t_fit'):
            # Both gauges of one fit in one pass: the dressed factor the kernel
            # uses and the bare one the self-energy screens with share the test
            # set, the three-centre integrals and M, and differ only in the
            # metric root.
            gauges = [GaugeAdjoint(d_bar, self.environment_at(mol),
                                   root_bar=root_bar, kernel_bar=kernel_bar)]
            if d_bar_bare is not None:
                gauges.append(GaugeAdjoint(d_bar_bare, None))
            g_fit = dfactor_adjoint_gauges(mol, auxmol, crd, gauges,
                                           self.layout, self.pts_local,
                                           self.owner, frames=self.frames,
                                           with_frames=self.with_frames)
        grad = g_orb + g_coll + g_fit
        # an exact gradient sums to zero over the atoms: a free check on every branch
        diags = dict(diags,
                     translation_residual=float(np.abs(grad.sum(axis=0)).max()),
                     branch_orbital=float(np.abs(g_orb).max()),
                     branch_collocation=float(np.abs(g_coll).max()),
                     branch_fit=float(np.abs(g_fit).max()))
        return grad, diags


def test_set_three_center(mol, auxmol, mu_i, nu_i,
                          max_bytes=THREE_CENTER_BLOCK_BYTES):
    """(mu_i nu_i|P) on the test-set pairs alone, blocked over the mu shells.

    The fit reads one row per test-set pair, of order a percent of the dense
    (nao, nao, naux) three-centre tensor. Building that tensor in order to index
    it costs nao^2 naux doubles, so it, not the flop count, is what ends the
    chain at large size. Blocking the first index holds one (block, nao, naux)
    slab at a time and returns the same rows to the bit.
    """
    nao, naux = mol.nao, auxmol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    mu_i, nu_i = np.asarray(mu_i), np.asarray(nu_i)
    out = np.empty((mu_i.size, naux))
    for sh0, sh1 in shell_blocks(mol, int(nao) * int(naux) * 8, max_bytes):
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        take = np.flatnonzero((mu_i >= a0) & (mu_i < a1))
        if not take.size:
            continue
        blk = pyscf_df.incore.aux_e2(
            mol, auxmol, intor='int3c2e', aosym='s1',
            shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas)
        ).reshape(a1 - a0, nao, naux)
        out[take] = blk[mu_i[take] - a0, nu_i[take], :]
        del blk
    return out


def check_scf_quality(mf, nocc, tol=SCF_GRAD_TOL, raise_on_fail=False):
    """max |F_ia| in the MO basis -- the quantity `conv_tol_grad` controls.

    The Lagrangian assumes this block vanishes, what is left is amplified in
    the force, and symmetry hides it: a symmetric molecule looks converged and
    is wrong in the fourth digit.
    """
    c = mf.mo_coeff
    resid = float(np.abs((c.T @ mf.get_fock() @ c)[:nocc, nocc:]).max())
    if resid > tol:
        msg = (f'the mean field carries |F_ia| = {resid:.2e}, above {tol:.0e}; '
               'the gradient Lagrangian assumes it vanishes. Set '
               'conv_tol_grad=1e-11.')
        if raise_on_fail:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
    return resid
