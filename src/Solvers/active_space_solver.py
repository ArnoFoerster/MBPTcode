"""One interface behind every active-space many-body solver.

An active-space Hamiltonian is a one-body matrix h1 over spatial orbitals and a
two-body tensor (pq|rs) in chemist notation with the eightfold permutation
symmetry. The core energy is not part of it, so every energy here is the
in-space eigenvalue at ecore = 0 and the caller adds its own constant.

`roots` returns the lowest states of one spin sector; `rdms` returns the one-
and two-particle density matrices of a state in pyscf's make_rdm12 convention,

    dm1[p, q] = <q^+ p>,   dm2[p, q, r, s] = <p^+ r^+ s q>,
    E = sum_pq h1_pq dm1_qp + (1/2) sum_pqrs (pq|rs) dm2_pqrs,

which is the pair a Hellmann-Feynman derivative of the active space contracts
with. Three engines satisfy the interface: pyscf's FCI, the determinant basis
of exact_diagonalization.py (the only one that carries a three-body vertex),
and DMRG through block2, whose cost is polynomial in the number of orbitals at
fixed bond dimension and so reaches active spaces the two exact engines cannot.
"""
import os
import tempfile

import numpy as np
from pyscf import fci

from src.Base.constants import FCI_CONV_TOL
from src.Solvers.exact_diagonalization import (build_explicit_hamiltonian,
                                               spatial_to_spin_orbitals)


class ActiveSpaceSolver:
    """Lowest states and density matrices of an active-space Hamiltonian.

    max_roots is the largest number of states the engine can produce, None for
    no limit; a ground-state-only method (coupled cluster) would set it to 1.
    """

    max_roots = None

    def roots(self, h1, eri, norb, nelec, nroots=1, spin=0):
        """(energies, vectors): the lowest nroots states of the spin sector, ecore = 0.

        energies ascend and carry no core energy; vectors holds one opaque entry
        per root, readable only by the same solver's `rdms`. nroots=None asks
        for the whole spectrum, which only a dense engine can give. spin is 2S
        (0 singlets, 2 triplets) or None for no spin adaptation at fixed Sz.
        """
        raise NotImplementedError

    def rdms(self, vector, norb, nelec, spin=0):
        """(dm1, dm2) of one root, pyscf's make_rdm12 convention."""
        raise NotImplementedError

    def _check_roots(self, nroots):
        """Refuse a request the engine cannot fill."""
        if self.max_roots is not None and (nroots is None or nroots > self.max_roots):
            raise ValueError(f'{type(self).__name__} produces at most '
                             f'{self.max_roots} root(s), asked for {nroots}')
        return nroots


class PyscfFCISolver(ActiveSpaceSolver):
    """pyscf FCI: direct_spin0 in the spin-symmetric sector, direct_spin1 elsewhere.

    direct_spin0 imposes the alpha-beta symmetry of the CI vector, so its roots
    are the even-S states and a singlet request gets singlets; direct_spin1
    fixes Sz alone and returns every state of that projection, which is what
    spin=None asks for.
    """

    max_roots = None

    def __init__(self, conv_tol=FCI_CONV_TOL):
        self.conv_tol = conv_tol

    def _engine(self, nelec, spin):
        """The pyscf FCI module whose spin symmetry matches the request."""
        module = (fci.direct_spin0 if spin == 0 and nelec[0] == nelec[1]
                  else fci.direct_spin1)
        engine = module.FCI()
        engine.conv_tol = self.conv_tol
        engine.verbose = 0
        return engine

    def roots(self, h1, eri, norb, nelec, nroots=1, spin=0):
        if nroots is None:
            raise ValueError('a Davidson solve needs a root count; the whole '
                             'spectrum comes from ExactDiagonalizationSolver')
        nelec = spin_sector(nelec, spin)
        energies, vectors = self._engine(nelec, spin).kernel(
            h1, eri, norb, nelec, ecore=0.0, nroots=self._check_roots(nroots))
        energies = np.atleast_1d(energies)
        return energies, [vectors] if np.ndim(vectors) == 2 else list(vectors)

    def rdms(self, vector, norb, nelec, spin=0):
        return fci.direct_spin1.make_rdm12(vector, norb, spin_sector(nelec, spin))


class ExactDiagonalizationSolver(ActiveSpaceSolver):
    """Dense diagonalization in the Sz = 0 determinant basis (openfermion).

    Every state of the projection comes out, singlets and Ms = 0 triplets alike,
    so spin only fixes the sector and must leave n_alpha = n_beta. The cost is
    the full binomial dimension and no density matrices are formed.
    """

    max_roots = None

    def __init__(self, symmetrize=True):
        self.symmetrize = symmetrize

    def spectrum(self, g_eff, n_spin_orbs, n_elec):
        """(energies, vectors) of a spin-orbital g_eff at Sz = 0, ascending.

        g_eff maps body rank to its tensor, so a three-body vertex is carried
        exactly. symmetrize=False keeps a non-Hermitian effective Hamiltonian as
        it is and the eigenvalues come out complex.
        """
        matrix, _ = build_explicit_hamiltonian(g_eff, n_spin_orbs, n_elec)
        if self.symmetrize:
            energies, vectors = np.linalg.eigh((0.5 * (matrix + matrix.T)).toarray())
        else:
            energies, vectors = np.linalg.eig(matrix.toarray())
        order = np.argsort(energies.real)
        return energies[order], vectors[:, order]

    def roots(self, h1, eri, norb, nelec, nroots=1, spin=0):
        n_alpha, n_beta = spin_sector(nelec, spin)
        if n_alpha != n_beta:
            raise NotImplementedError(
                f'the determinant basis is built at Sz = 0, asked for '
                f'({n_alpha}, {n_beta})')
        self._check_roots(nroots)
        g_eff = spatial_to_spin_orbitals({1: h1, 2: eri.transpose(0, 2, 1, 3)}, norb)
        energies, vectors = self.spectrum(g_eff, 2 * norb, n_alpha + n_beta)
        n = len(energies) if nroots is None else min(nroots, len(energies))
        return energies[:n].real, [vectors[:, k] for k in range(n)]

    def rdms(self, vector, norb, nelec, spin=0):
        raise NotImplementedError(
            'the determinant basis carries no density matrices; use '
            'PyscfFCISolver or Block2DMRGSolver where dm1/dm2 are needed')


class Block2DMRGSolver(ActiveSpaceSolver):
    """DMRG through block2, spin-adapted (SU2).

    The bond dimension is the only approximation: the matrix product state is
    exact once it exceeds the largest Schmidt rank of the space, and the sweep
    cost grows polynomially with the number of orbitals rather than as the
    determinant count. spin is the SU2 target 2S, so singlets and triplets are
    separate solves; nroots > 1 optimizes one state-averaged MPS and the density
    matrices of a root come from splitting it out.

    block2 keeps a single global scratch frame, so only one instance may be in
    flight at a time and `rdms` reads the MPS the last `roots` call left on disk.
    n_threads=None takes the OpenMP environment's count.
    """

    max_roots = None

    def __init__(self, bond_dim=500, n_sweeps=20, tol=FCI_CONV_TOL, noise=1e-4,
                 davidson_tol=1e-12, scratch=None, n_threads=None):
        self.bond_dim, self.n_sweeps, self.tol = bond_dim, n_sweeps, tol
        self.noise, self.davidson_tol = noise, davidson_tol
        self.scratch, self.n_threads = scratch, n_threads
        self._driver = None

    def driver(self):
        """The block2 driver, made once so the global scratch frame stays single."""
        if self._driver is None:
            # block2 is an optional dependency; only this adapter needs it
            try:
                from pyblock2.driver.core import DMRGDriver, SymmetryTypes
            except ImportError as exc:
                raise ImportError(
                    'Block2DMRGSolver needs a loadable block2 (pip install '
                    'block2; its wheel also needs the OpenMP runtime it was '
                    'linked against)') from exc
            if self.scratch is None:
                self.scratch = tempfile.mkdtemp(prefix='block2-')
            os.makedirs(self.scratch, exist_ok=True)
            self._driver = DMRGDriver(scratch=self.scratch,
                                      symm_type=SymmetryTypes.SU2,
                                      n_threads=self.n_threads)
        return self._driver

    def _schedule(self):
        """(bond_dims, noises, thrds) per sweep: the noise is switched off halfway."""
        half = self.n_sweeps // 2
        return ([self.bond_dim] * self.n_sweeps,
                [self.noise] * half + [0.0] * (self.n_sweeps - half),
                [self.davidson_tol] * self.n_sweeps)

    def roots(self, h1, eri, norb, nelec, nroots=1, spin=0):
        if nroots is None:
            raise ValueError('DMRG optimizes a fixed number of roots; give nroots')
        if spin is None:
            raise ValueError('an SU2 target is a total spin; spin=None (fixed Sz, '
                             'mixed S) has no DMRG sector here')
        n_alpha, n_beta = spin_sector(nelec, spin)
        driver = self.driver()
        driver.initialize_system(n_sites=norb, n_elec=n_alpha + n_beta,
                                 spin=n_alpha - n_beta, orb_sym=None)
        mpo = driver.get_qc_mpo(h1e=np.ascontiguousarray(h1),
                                g2e=np.ascontiguousarray(eri), ecore=0.0, iprint=0)
        n = max(1, self._check_roots(nroots))
        ket = driver.get_random_mps(tag='KET', bond_dim=self.bond_dim, nroots=n)
        bond_dims, noises, thrds = self._schedule()
        energies = np.atleast_1d(driver.dmrg(mpo, ket, n_sweeps=self.n_sweeps,
                                             tol=self.tol, bond_dims=bond_dims,
                                             noises=noises, thrds=thrds, iprint=0))
        order = np.argsort(energies[:n])
        # a single root is a plain MPS; a state-averaged one has to be split first
        return energies[order], [(driver, ket, int(k) if n > 1 else None)
                                 for k in order]

    def rdms(self, vector, norb, nelec, spin=0):
        driver, ket, iroot = vector
        if iroot is not None:
            ket = driver.split_mps(ket, iroot, f'KET-{iroot}')
        # block2's SU2 npdm is <p^+ q> and <p^+ q^+ s r>; pyscf orders both differently
        dm1 = np.asarray(driver.get_1pdm(ket)).T
        dm2 = np.asarray(driver.get_2pdm(ket)).transpose(0, 3, 1, 2)
        return dm1, dm2


def spin_sector(nelec, spin=0):
    """(n_alpha, n_beta) carrying 2S = spin unpaired electrons.

    nelec is a total electron count or an (n_alpha, n_beta) pair; spin moves
    electrons from beta to alpha until n_alpha - n_beta = spin, so the triplet
    of a closed-shell count is (n/2 + 1, n/2 - 1). spin=None leaves the pair
    untouched -- the unadapted solve at whatever Sz it already has.
    """
    if np.isscalar(nelec):
        n = int(nelec)
        n_alpha = (n + (0 if spin is None else int(spin))) // 2
        return n_alpha, n - n_alpha
    n_alpha, n_beta = int(nelec[0]), int(nelec[1])
    if spin is None:
        return n_alpha, n_beta
    shift = (int(spin) - (n_alpha - n_beta)) // 2
    return n_alpha + shift, n_beta - shift
