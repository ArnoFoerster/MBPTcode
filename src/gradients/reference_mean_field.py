"""The reference mean field of a factor chain, and the one SCF per geometry
every evaluation of the chain stands on.

The gradient Lagrangian assumes the occupied-virtual Fock block vanishes, so
every mean field the chain reads is measured (`check_scf_quality`) and a
force on one above `scf_force_tol` warns and is flagged (`scf_flags`). A mean field returned
built but not run is converged over the ranks (`converged_factory`).
"""
import gc
import time
import warnings

import numpy as np

from src.Base.numerics.constants import (EVALUATION_CACHE_SIZE, SCF_GRAD_TOL,
                                         SCF_OPTIMIZATION_GRAD_TOL)
from src.Base.factorization.distributed_df import distributed_fock, distributed_mean_field
from src.Base.factorization.distributed_isdf_jk import scf_pair_layout
from src.Base.factorization.isdf_jk import (frozen_frames, frozen_pair_layout, frozen_span,
                                            reference_frames)
from src.Base.utils.parallel.mpi_grid import lockstep_mean_field
from src.gradients.evaluation import GeometryCache


class ReferenceMeanField:
    """The reference mean field `mf0` and the mean field at every geometry
    the chain is evaluated at, each converged once in the chain's
    environment and measured once (`mean_field`, `residual_of`). The class
    using it sets `mol0`, `scf_factory`, `environment`, `nocc` and `phase`
    and calls `_init_reference` from its constructor."""

    def _init_reference(self, mol, scf_factory, mf):
        """Converge the reference mean field once, lock it to rank 0's, and
        freeze what every other geometry's SCF is solved in: its span, its
        atomic frames and (on the first displaced geometry) its pair layout."""
        t0 = time.perf_counter()
        with self.phase('t_scf'):
            self.mf0 = self.environment.mean_field(
                mol, converged_factory(
                    scf_factory if mf is None else (lambda _mol: mf)))
        # One reference on every rank from the start: what the chain reads off
        # mf0 directly must be rank 0's, and a given `mf=` may have been
        # converged on each rank alone.
        lockstep_mean_field(self.mf0)
        # the reference's span, which every other geometry's SCF is solved in
        # (`frozen_span`); None for a mean field in the full basis
        self.span_basis = getattr(self.mf0, 'span_basis', None)
        # the atomic frames the reference's ISDF-K points sit on, which every
        # other geometry's SCF places its points on turned with the molecule
        # (`isdf_jk.frozen_frames`): a covariant frame near a degenerate
        # neighbour tensor turns at 1/gap and dimples the surface between
        # refreezes (formaldehyde's carbon, 18.9 rad/Bohr). A refreeze takes
        # its own at the new reference; None without an ISDF-K fit.
        self.scf_frames = reference_frames(self.mf0)
        self.construction_seconds['mean_field'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.scf_residual = check_scf_quality(self.mf0, self.nocc)
        self.construction_seconds['scf_quality'] = time.perf_counter() - t0
        # ONE SCF PER GEOMETRY: the mean fields of the last geometries
        # evaluated, the reference's among them, and each one's |F_ia|
        self._mean_fields = GeometryCache()
        self._mean_fields.put(mol, self.mf0)
        self._scf_residuals = [(self.mf0, self.scf_residual)]
        # numerical flags of the mean fields read (`flag_scf_quality`): a
        # warned-about |F_ia| above `scf_force_tol` or an SCF that stopped
        # short, each {'where', 'residual', 'tol', 'coords'}
        self.scf_flags = []
        # the reference mean field's ISDF-K pair layout, read on the first
        # displaced geometry and shared by every shallow copy (`scf_layout`)
        self._scf_layout = {}

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the reference pair, or the SCF at another geometry,
        built in the chain's environment ONCE: a geometry asked again within
        the last EVALUATION_CACHE_SIZE gets the mean field it got before
        (`GeometryCache`, keyed by the nuclei), so the walk, its record and
        the ground state there stand on one SCF.

        The fresh one is converged over the ranks wherever the factory leaves
        it to them (`converged_factory`): a displaced geometry is a whole SCF,
        and it is the one stage of a walk that every rank would otherwise
        compute alone. Its |F_ia| is measured (`check_scf_quality`); one
        that stopped short (not converged, and |F_ia| above `scf_force_tol`)
        is warned about and flagged (`flag_scf_quality`, `scf_flags`), and so
        is any above `scf_force_tol` a force or coupling reads
        (`require_scf_quality`).
        A given `mf` of other nuclei than `mol` is refused: the kernels read
        `mol` and the skeletons `mf.mol`, so the force would mix two
        geometries.
        """
        mol = self.mol0 if mol is None else mol
        if mf is not None and not np.array_equal(
                np.asarray(mf.mol.atom_coords()), np.asarray(mol.atom_coords())):
            raise ValueError(
                'the mean field handed in belongs to another geometry than '
                'the molecule it is evaluated at')
        fresh = False
        if mf is None:
            mf = self.mf0 if mol is self.mol0 else self._mean_fields.get(mol)
            fresh = mf is None
            if fresh:
                # an abandoned mean field is a reference cycle (see
                # `response_kernel`), and numpy-sized garbage never wakes the
                # cyclic collector
                gc.collect()
                with self.phase('t_scf'), \
                        frozen_pair_layout(self.scf_layout()), \
                        frozen_span(self.span_basis), \
                        frozen_frames(self.scf_frames):
                    mf = self.environment.mean_field(
                        mol, converged_factory(self.scf_factory))
                self._mean_fields.put(mol, mf)
            if mf.mo_coeff.shape[-1] != self.mf0.mo_coeff.shape[-1]:
                # a span of another dimension is another surface
                raise RuntimeError(
                    f'the mean field at this geometry carries '
                    f'{mf.mo_coeff.shape[-1]} orbitals and the reference '
                    f'{self.mf0.mo_coeff.shape[-1]}: its basis was cut to '
                    f'another span (`span_confined` takes one count for '
                    f'every geometry)')
        # Rank 0's orbitals on every rank: the chain forms X_mo = X_ao C and
        # contracts the kernels' adjoints with this mean field's coefficients,
        # and another rank's SCF can differ from rank 0's by the phases and
        # degenerate rotations of its orbitals. A mean field the ranks
        # converged together arrives locked and this rewrites the same bits;
        # one each rank converged alone does not.
        lockstep_mean_field(mf)
        if fresh and not getattr(mf, 'converged', True):
            # an SCF this chain ran that stopped short -- not converged, and
            # its |F_ia| above what it was asked for -- is a numerical event:
            # warned about and flagged, and the run carries on.
            # One flagged unconverged on its energy alone (a
            # Kohn-Sham quadrature's noise against a tight conv_tol) with its
            # orbitals converged is neither.
            self.flag_scf_quality(mf, 'the SCF at this geometry stopped short')
        elif fresh:
            self.residual_of(mf)
        return mol, mf

    def residual_of(self, mf):
        """max |F_ia| of `mf` (`check_scf_quality`), measured once per mean
        field."""
        for held, resid in self._scf_residuals:
            if held is mf:
                return resid
        resid = check_scf_quality(mf, self.nocc)
        self._scf_residuals = ([(mf, resid)] + self._scf_residuals
                               )[:EVALUATION_CACHE_SIZE + 1]
        return resid

    def flag_scf_quality(self, mf, where):
        """Warn about, and flag in `scf_flags`, a mean field whose |F_ia|
        exceeds `scf_force_tol`, once per entry `where` and mean field; the
        run carries on (refuse inputs, warn
        numerics). Returns the residual."""
        resid, tol = self.residual_of(mf), scf_force_tol(mf)
        if resid > tol:
            coords = np.array(mf.mol.atom_coords(), float)
            # a chain rebuilt from a record, or a copy, may not have the list
            flags = self.__dict__.setdefault('scf_flags', [])
            if any(f['where'] == where and f['residual'] == float(resid)
                   and np.array_equal(f['coords'], coords) for f in flags):
                return resid
            flags.append({'where': where, 'residual': float(resid),
                          'tol': float(tol), 'coords': coords})
            warnings.warn(
                f'{where}: the mean field carries |F_ia| = {resid:.2e}, above '
                f'the {tol:.0e} a force on it may carry (`scf_force_tol`); '
                f'the gradient Lagrangian assumes the block vanishes',
                RuntimeWarning, stacklevel=3)
        return resid

    def require_scf_quality(self, mf, where='force'):
        """The flags (`scf_flags`) a force or coupling `where` on `mf`
        carries: one when its |F_ia| exceeds `scf_force_tol`, warned about
        (`flag_scf_quality`), none otherwise. The Lagrangian assumes the
        block vanishes, so what is left rides, amplified, in the force; a
        numerical event, flagged, and the run carries on
        (refuse inputs, warn numerics). A mean field that was
        never run is an input and is refused."""
        if getattr(mf, 'mo_coeff', None) is None:
            raise ValueError(
                f'{where}: the mean field handed in was never run (no '
                f'mo_coeff)')
        self.flag_scf_quality(mf, where)
        coords = np.array(mf.mol.atom_coords(), float)
        return [dict(f) for f in getattr(self, 'scf_flags', [])
                if f['where'] == where and np.array_equal(f['coords'], coords)
                and f['residual'] == float(self.residual_of(mf))]

    def scf_layout(self):
        """The AO-pair columns the reference mean field's ISDF-K fit kept,
        which the mean field at every other geometry of this chain fits on,
        so that E_0 and its force stay one function along a walk; None for a
        mean field without an ISDF-K fit. A `refreeze` takes its own at the
        new reference, as it does every other convention."""
        if 'layout' not in self._scf_layout:
            self._scf_layout['layout'] = scf_pair_layout(self.mf0)
        return self._scf_layout['layout']


def converged_factory(scf_factory):
    """`scf_factory` completed by the chain when it returns a mean field that
    has been built but not run.

    The completion is `distributed_mean_field`: inside a `distributed`
    context every rank runs pyscf's SCF driver against the reduced J/K and
    quadrature, each contributing its block of the auxiliary index and of the
    xc grid, and rank 0's spectrum and orbitals end on all of them. Outside
    one it is `mf.kernel()`. Inside one the mean field must use pyscf's own DF
    (the split is over the rows of `cderi`); `distributed_mean_field` refuses
    any other.
    """
    def build(mol):
        mf = scf_factory(mol)
        if getattr(mf, 'mo_coeff', None) is None:
            distributed_mean_field(mf)
        return mf

    return build


def scf_force_tol(mf):
    """The |F_ia| a force on `mf` may carry: what its SCF was converged to
    (pyscf's conv_tol_grad bounds twice the largest |F_ia|), never tighter
    than SCF_GRAD_TOL -- the exchange-correlation quadrature alone leaves
    1e-11 -- and never looser than SCF_OPTIMIZATION_GRAD_TOL, the loosest SCF
    a force is taken on."""
    declared = getattr(mf, 'conv_tol_grad', None)
    declared = (SCF_OPTIMIZATION_GRAD_TOL if declared is None
                else min(float(declared), SCF_OPTIMIZATION_GRAD_TOL))
    return max(declared, SCF_GRAD_TOL)


def check_scf_quality(mf, nocc):
    """max |F_ia| in the MO basis -- the quantity `conv_tol_grad` controls.

    A measurement, not a verdict. The Lagrangian assumes this block vanishes,
    what is left is amplified in the force, and symmetry hides it: a symmetric
    molecule looks converged and is wrong in the fourth digit. So every force,
    coupling and relaxed density warns about, and flags, a mean field above
    `scf_force_tol` (`require_scf_quality`, `scf_flags`; converge to
    SCF_DIFFERENTIABLE_GRAD_TOL),
    while an energy, second order in F_ia, reads it as it is. Under ranks the
    Fock matrix is the distributed SCF's own (`distributed_fock`).
    """
    c = mf.mo_coeff
    with distributed_fock(mf, build=False):
        fock = mf.get_fock()
    return float(np.abs((c.T @ fock @ c)[:nocc, nocc:]).max())
