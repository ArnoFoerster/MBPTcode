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
with. Four engines satisfy the interface: pyscf's FCI, the determinant basis
of exact_diagonalization.py (the only one that carries a three-body vertex),
DMRG through block2, whose cost is polynomial in the number of orbitals at
fixed bond dimension and so reaches active spaces the two exact engines cannot,
and high-order coupled cluster (CCSDT, CCSDTQ) on a Hartree-Fock reference in
the active space, a ground-state energy per spin sector at a cost polynomial in
the number of orbitals and electrons.
"""
import os
import tempfile
import types

import numpy as np
from pyscf import fci, gto, scf

from src.Base.numerical_flags import collecting, flag
from src.Base.numerics.constants import (ACTIVE_SPACE_VERTEX_TOL,
                                         CC_ACTIVE_ENERGY_TOL,
                                         CC_ACTIVE_MAX_CYCLE,
                                         CC_ACTIVE_RESIDUAL_TOL,
                                         CC_ACTIVE_SCF_MAX_CYCLE,
                                         CC_ACTIVE_SCF_TOL, DMRG_BOND_DIM,
                                         DMRG_DAVIDSON_TOL, DMRG_N_SWEEPS,
                                         DMRG_NOISE, FCI_CONV_TOL)
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

    def __init__(self, bond_dim=DMRG_BOND_DIM, n_sweeps=DMRG_N_SWEEPS,
                 tol=FCI_CONV_TOL, noise=DMRG_NOISE,
                 davidson_tol=DMRG_DAVIDSON_TOL, scratch=None, n_threads=None):
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


class ActiveSpaceCCNotConverged(RuntimeWarning):
    """The active-space Hartree-Fock reference or the coupled-cluster amplitudes
    ran out of cycles; the last iterate's energy is returned."""


class CoupledClusterSolver(ActiveSpaceSolver):
    """High-order coupled cluster on a Hartree-Fock reference in the active space.

    order 3 is CCSDT, order 4 CCSDTQ, both pyscf's implementations by Jin,
    Hillenbrand, Berkelbach and Zhai (J. Chem. Phys. 165, 144109 (2026)): the
    closed-shell solver is spin-free (spin-adapted, RHF reference) and keeps the
    highest amplitudes only for i <= j <= k (<= l). An open-shell sector (spin
    != 0, or n_alpha != n_beta) is solved by spin-unrestricted CCSDT on a UHF
    reference of the high-spin determinant; it is not spin-adapted, and there is
    no open-shell CCSDTQ.

    The reference is the Hartree-Fock solution of (h1, eri) inside the active
    space, started from the aufbau occupation of the orbitals the Hamiltonian is
    written in, so the canonical orbitals of a downfolding are its first guess.
    The engine gives energies only: one root per spin sector (the lowest state
    of that symmetry, so a triplet T1 at spin=2 but no S1), and no density
    matrices. It takes any real Hermitian one- and two-body operator: the
    integrals of the reference orbitals are transformed from the full (pq|rs),
    so an effective vertex without the p <-> q symmetry of real-orbital
    integrals (a downfolded one) is solved as it is, never averaged. Threads are
    pyscf's (the OpenMP environment); the contractions run on pytblis when it is
    importable, else on pyscf's einsum.

    `last_result` holds the reference and correlation energies of the last solve,
    its convergence and the named flags it raised.
    """

    max_roots = 1

    #: excitation order -> method name
    ORDERS = {3: 'CCSDT', 4: 'CCSDTQ'}

    def __init__(self, order=3):
        if order not in self.ORDERS:
            raise ValueError(f'CoupledClusterSolver has orders {sorted(self.ORDERS)} '
                             f'(CCSDT, CCSDTQ), asked for {order!r}')
        self.order = order
        self.last_result = None

    def roots(self, h1, eri, norb, nelec, nroots=1, spin=0):
        if nroots is None:
            raise ValueError('coupled cluster gives the lowest state of a spin '
                             'sector, not a spectrum; give nroots=1')
        self._check_roots(nroots)
        h1, eri = np.asarray(h1, dtype=float), np.asarray(eri, dtype=float)
        _require_hermitian(h1, eri, norb)
        n_alpha, n_beta = spin_sector(nelec, spin)
        if n_alpha != n_beta and self.order != 3:
            raise NotImplementedError(
                f'{self.ORDERS[self.order]} is closed-shell only; the open-shell '
                f'sector ({n_alpha}, {n_beta}) has spin-unrestricted CCSDT (order 3)')
        if min(n_alpha, n_beta) < 0 or max(n_alpha, n_beta) > norb:
            raise ValueError(f'({n_alpha}, {n_beta}) electrons do not fit in '
                             f'{norb} orbitals')
        flags = []
        with collecting(flags):
            reference = _active_space_reference(h1, eri, norb, n_alpha, n_beta)
            if not reference.converged:
                flag('active_space_reference_not_converged',
                     f'the active-space Hartree-Fock reference did not converge in '
                     f'{CC_ACTIVE_SCF_MAX_CYCLE} cycles; coupled cluster runs on its '
                     f'last iterate', ActiveSpaceCCNotConverged)
            mycc = self._engine(reference, eri, n_alpha == n_beta)
            mycc.kernel()
            if not mycc.converged:
                flag('active_space_cc_not_converged',
                     f'{type(mycc).__name__} in the active space did not converge '
                     f'in {CC_ACTIVE_MAX_CYCLE} cycles; its last energy is returned',
                     ActiveSpaceCCNotConverged)
        energy = float(mycc.e_tot)
        self.last_result = {
            'method': type(mycc).__name__, 'energy': energy,
            'e_reference': float(reference.e_tot), 'e_corr': float(mycc.e_corr),
            'reference_converged': bool(reference.converged),
            'converged': bool(mycc.converged), 'cycles': int(mycc.cycles),
            'numerical_flags': flags}
        return np.array([energy]), [dict(self.last_result)]

    def _engine(self, reference, eri, closed_shell):
        """The pyscf CC object of this order and sector on the full tensor eri,
        its settings set."""
        from pyscf.cc import rccsdt, rccsdtq, uccsdt
        if closed_shell:
            mycc = (rccsdt.RCCSDT if self.order == 3 else rccsdtq.RCCSDTQ)(reference)
            mycc.ao2mo = types.MethodType(_restricted_eris(eri), mycc)
        else:
            mycc = uccsdt.UCCSDT(reference)
            mycc.ao2mo = types.MethodType(_unrestricted_eris(eri), mycc)
        mycc.verbose = 0
        mycc.conv_tol = CC_ACTIVE_ENERGY_TOL
        mycc.conv_tol_normt = CC_ACTIVE_RESIDUAL_TOL
        mycc.max_cycle = CC_ACTIVE_MAX_CYCLE
        mycc.set_einsum_backend(_einsum_backend())
        return mycc

    def rdms(self, vector, norb, nelec, spin=0):
        raise NotImplementedError(
            'the coupled-cluster engine gives energies only (no Lambda equations); '
            'use PyscfFCISolver or Block2DMRGSolver where dm1/dm2 are needed')


def _einsum_backend():
    """'pytblis' when it is importable, else pyscf's own einsum."""
    try:
        import pytblis  # noqa: F401
    except ImportError:
        return 'pyscf'
    return 'pytblis'


def _require_hermitian(h1, eri, norb):
    """Refuse a Hamiltonian that is not a real Hermitian operator in the form
    the reference and the amplitudes read: h1 symmetric, (pq|rs) = (qp|sr), and
    (pq|rs) = (rs|pq), which only relabels the same operator (the dispatcher
    writes it so). Neither is enforced by averaging."""
    if h1.shape != (norb, norb) or eri.shape != (norb,) * 4:
        raise ValueError(f'h1 {h1.shape} and eri {eri.shape} do not match '
                         f'norb={norb}')
    scale = max(1.0, float(np.max(np.abs(eri))))
    residual = max(float(np.max(np.abs(h1 - h1.T))),
                   float(np.max(np.abs(eri - eri.transpose(1, 0, 3, 2)))),
                   float(np.max(np.abs(eri - eri.transpose(2, 3, 0, 1)))))
    if residual > ACTIVE_SPACE_VERTEX_TOL * scale:
        raise ValueError(
            f'CoupledClusterSolver needs a real Hermitian Hamiltonian with '
            f'(pq|rs) = (rs|pq), and this one breaks it by {residual:.2e}; it is '
            f'not symmetrized here (active_space.solve takes the Hermitian part '
            f'on request).')


def _mo_tensor(eri, c1, c2, c3, c4):
    """(ij|kl) of the full chemist tensor in the orbitals c1..c4, no symmetry assumed."""
    return np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, c1, c2, c3, c4, optimize=True)


def _restricted_eris(eri):
    """The `ao2mo` of a closed-shell CC object, reading the full tensor eri
    instead of pyscf's packed integrals; <pq|rs> = (pr|qs) as the engine keeps it."""
    def ao2mo(mycc, mo_coeff=None):
        from pyscf.cc import rccsdt
        eris = rccsdt._PhysicistsERIs()
        eris._common_init_(mycc, mo_coeff)
        c = eris.mo_coeff
        eris.pppp = np.ascontiguousarray(_mo_tensor(eri, c, c, c, c).transpose(0, 2, 1, 3))
        return eris
    return ao2mo


def _unrestricted_eris(eri):
    """The `ao2mo` of a spin-unrestricted CC object on the full tensor eri: the
    alpha-alpha, beta-beta and alpha-beta blocks the engine keeps."""
    def ao2mo(mycc, mo_coeff=None):
        from pyscf.cc import uccsdt
        eris = uccsdt._PhysicistsERIs()
        eris._common_init_(mycc, mo_coeff)
        a, b = eris.mo_coeff
        eris.pppp = np.ascontiguousarray(_mo_tensor(eri, a, a, a, a).transpose(0, 2, 1, 3))
        eris.PPPP = np.ascontiguousarray(_mo_tensor(eri, b, b, b, b).transpose(0, 2, 1, 3))
        eris.pPpP = np.ascontiguousarray(_mo_tensor(eri, a, a, b, b).transpose(0, 2, 1, 3))
        return eris
    return ao2mo


def _active_space_reference(h1, eri, norb, n_alpha, n_beta):
    """The converged (or last) Hartree-Fock solution of (h1, eri) in the active
    space: RHF for n_alpha == n_beta, else UHF, from the aufbau occupation of
    the orbitals the Hamiltonian is written in. eri is kept as the full tensor,
    so J and K are its own; the energy is in-space, ecore = 0."""
    mol = gto.M(verbose=0)
    mol.nelectron = n_alpha + n_beta
    mol.spin = n_alpha - n_beta
    mol.incore_anyway = True
    mf = scf.RHF(mol) if n_alpha == n_beta else scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = eri
    mf.conv_tol = CC_ACTIVE_SCF_TOL
    mf.max_cycle = CC_ACTIVE_SCF_MAX_CYCLE
    occ_a = np.diag((np.arange(norb) < n_alpha).astype(float))
    occ_b = np.diag((np.arange(norb) < n_beta).astype(float))
    mf.kernel(occ_a + occ_b if n_alpha == n_beta else np.array([occ_a, occ_b]))
    return mf


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
