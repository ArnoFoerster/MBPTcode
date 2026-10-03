"""Fragment-localized orbitals: the orbital basis the fragment-partitioned BSE
is written in.

A supersystem is split into fragments (lists of atom indices), and its
occupied and its virtual canonical orbitals are rotated, separately, into
orbitals that each belong to one fragment. An electron-hole configuration
(i, a) is then LOCAL when i and a sit on the same fragment and CHARGE
TRANSFER when they do not, and the BSE matrix in this basis splits into the
blocks `src.properties.fragment_bse` partitions.

THE FUNCTIONAL IS PIPEK-MEZEY OVER FRAGMENT POPULATIONS. Each orbital's
population on a fragment is the sum of its populations on that fragment's
atoms, and the localization maximizes sum_i sum_K q_Ki^2 (exponent 2) within
the occupied and within the virtual space (Pipek and Mezey, J. Chem. Phys.
90, 4916 (1989)), run through pyscf's `lo.pipek.PipekMezey` with the
population tensor replaced. Two population
schemes are offered:

* 'lowdin' (default): q_Ki = sum_{mu in K} [S^1/2 C]_{mu i}^2. Defined for
  both spaces, and its nuclear derivative is closed form through that of
  S^1/2, which is why the analytic diabatic gradient is written for it.
* 'iao': populations in Loewdin-orthogonalized intrinsic atomic orbitals
  (Knizia, J. Chem. Theory Comput. 9, 4834 (2013)) of the free-atom minimal
  basis. Defined for the occupied space only; a cross-check of how much the
  diabats depend on the population scheme, not a route with a gradient.

WHAT IS LEFT OUT ON PURPOSE. The populations are referred to the
supersystem's own basis (Loewdin) or to a free-atom minimal basis (IAO), never
to orbitals of the isolated fragments: that would tie every diabatic quantity
to separate fragment SCFs and make their response part of every gradient. Nor
are the local orbitals recanonicalized inside each fragment, because nothing
downstream needs it: every diabatic quantity depends on the fragment
SUBSPACES only (the span of the orbitals assigned to each fragment), not on
how the orbitals are rotated inside one fragment. That is also the invariance
the tests check.

THE ROTATION IS KEPT. `u_occ` and `u_vir` are the canonical-to-local
rotations, C_loc = C_can U, because the BSE is assembled and differentiated
in the canonical basis and only rotated into this one.
"""
from dataclasses import dataclass, field

import numpy as np
from pyscf import gto, lo
from pyscf.lo import iao as pyscf_iao
from pyscf.lo import orth
from scipy.sparse.linalg import LinearOperator, minres

from src.Base.constants import (FRAGMENT_PM_CONV_TOL, FRAGMENT_PM_CONV_TOL_GRAD,
                                FRAGMENT_PM_POLISH_MAX,
                                LOCALIZED_ASSIGNMENT_FLOOR)

POPULATION_SCHEMES = ('lowdin', 'iao')


def _sqrt_overlap(s):
    w, v = np.linalg.eigh(s)
    return (v * np.sqrt(w)) @ v.T


def fragment_ao_indices(mol, fragments):
    """One array of AO indices per fragment, from atom-index lists."""
    slices = mol.aoslice_by_atom()
    out = []
    for atoms in fragments:
        idx = [np.arange(slices[a][2], slices[a][3]) for a in atoms]
        out.append(np.concatenate(idx) if idx else np.zeros(0, int))
    return out


def check_partition(mol, fragments):
    """Atom-index lists, checked to partition the atoms exactly once."""
    fragments = [list(map(int, f)) for f in fragments]
    flat = sorted(a for f in fragments for a in f)
    if flat != list(range(mol.natm)):
        raise ValueError(f'fragments must partition the {mol.natm} atoms, each '
                         f'exactly once; got {fragments}')
    if len(fragments) < 2:
        raise ValueError('a fragment partition needs at least two fragments')
    return fragments


def population_tensor(mol, fragments, scheme='lowdin', orbocc=None, s=None):
    """pops(C) -> (nfrag, n, n): the fragment population of every orbital pair.

    Its diagonal [K, i, i] is the population q_Ki of orbital i on fragment K,
    which is what the Pipek-Mezey functional reads; the off-diagonal is what
    its gradient and Hessian read. For a normalized orbital the populations
    over a partition sum to one.
    """
    s = mol.intor_symmetric('int1e_ovlp') if s is None else s
    if scheme == 'lowdin':
        frame = _sqrt_overlap(s)                    # rows: orthogonal AOs
        rows = fragment_ao_indices(mol, fragments)
    elif scheme == 'iao':
        if orbocc is None:
            raise ValueError("scheme='iao' needs the occupied orbitals")
        iaos = orth.vec_lowdin(pyscf_iao.iao(mol, orbocc), s)
        frame = iaos.T @ s                          # rows: orthogonal IAOs
        ref = pyscf_iao.reference_mol(mol)
        offsets = ref.offset_nr_by_atom()
        rows = [np.concatenate([np.arange(offsets[a][2], offsets[a][3])
                                for a in atoms]) for atoms in fragments]
    else:
        raise ValueError(f'scheme must be one of {POPULATION_SCHEMES}, got '
                         f'{scheme!r}')

    def pops(mol_, mo_coeff, method=None):
        c = frame @ np.asarray(mo_coeff, float)
        return np.stack([c[r].T @ c[r] for r in rows])
    return pops


def _localize(mol, orbitals, pops, start=None, conv_tol=FRAGMENT_PM_CONV_TOL):
    """(rotation U with orbitals @ U localized, |gradient|) by fragment PM.

    Without a start the rotation begins from pyscf's 'atomic' guess (the
    orthogonalized AOs each orbital overlaps most), not from the canonical
    orbitals: in a near-symmetric supersystem the canonical orbitals are the
    in- and out-of-phase combinations of fragment orbitals, a stationary
    point of the fragment functional that a gradient method does not leave.
    """
    n = orbitals.shape[1]
    if n <= 1:
        return np.eye(n), 0.0
    s = mol.intor_symmetric('int1e_ovlp')
    pm = lo.pipek.PipekMezey(mol, orbitals if start is None
                             else orbitals @ start)
    pm.atomic_pops = pops
    pm.exponent = 2
    pm.conv_tol = conv_tol
    pm.conv_tol_grad = FRAGMENT_PM_CONV_TOL_GRAD
    pm.max_cycle = 500
    pm.verbose = 0
    if start is None:
        pm.init_guess = 'atomic'
        c_loc = pm.kernel()
    else:
        c_loc = pm.kernel(orbitals @ start)
    c_loc = _newton_polish(pm, c_loc)
    u = orbitals.T @ s @ c_loc
    pm.mo_coeff = c_loc
    grad = float(np.linalg.norm(pm.get_grad(np.eye(n))))
    return u, grad


def _newton_polish(pm, c_loc, tol=FRAGMENT_PM_CONV_TOL_GRAD, maxiter=8):
    """Newton steps on the PM functional from a converged `c_loc`.

    pyscf's second-order solver stops on a joint criterion and in practice
    leaves a gradient of 1e-8..1e-7 on this functional; the orbitals are
    differentiated afterwards, so a few full Newton steps with its own analytic
    gradient and Hessian-vector product (`gen_g_hop`) finish the job. The
    Hessian is symmetric and nearly singular inside each fragment, so each
    step is a MINRES solve (densely by least squares below
    FRAGMENT_PM_POLISH_MAX rotation parameters).
    """
    n = c_loc.shape[1]
    npar = n * (n - 1) // 2
    if npar == 0:
        return c_loc
    for _ in range(maxiter):
        pm.mo_coeff = c_loc
        g, h_op, _ = pm.gen_g_hop(np.eye(n))
        if np.linalg.norm(g) < tol:
            break
        if npar <= FRAGMENT_PM_POLISH_MAX:
            h = np.array([h_op(e) for e in np.eye(npar)]).T
            x = np.linalg.lstsq(0.5 * (h + h.T), -g, rcond=1e-12)[0]
        else:
            op = LinearOperator((npar, npar), matvec=h_op, dtype=float)
            x = minres(op, -g, rtol=1e-6, maxiter=500)[0]
        c_loc = c_loc @ pm.extract_rotation(x)
    return c_loc


def _stationarity(proj, labels):
    """(between, within): the largest Pipek-Mezey stationarity residual over
    rotations between two fragments and inside one.

    The fragment functional is nearly flat under rotations inside one
    fragment, so a second-order solver stops with a residual there that
    nothing downstream sees: every diabatic quantity depends on the fragment
    subspaces only. The residual that matters is the one between fragments.
    """
    q = np.einsum('kii->ki', proj)
    r = np.einsum('ki,kij->ij', q, proj)
    s = r - r.T
    same = labels[:, None] == labels[None, :]
    return (float(np.abs(np.where(same, 0.0, s)).max(initial=0.0)),
            float(np.abs(np.where(same, s, 0.0)).max(initial=0.0)))


def _transport(mol, orbitals, ref_mol, ref_local):
    """Rotation of `orbitals` closest to the reference local orbitals.

    The reference sits at another geometry; its orbitals are carried over by
    the AO cross overlap and the nearest unitary (polar factor) is the start
    of the localization, so a scan stays on one local maximum.
    """
    m = orbitals.T @ gto.mole.intor_cross('int1e_ovlp', mol, ref_mol) @ ref_local
    u, _, vt = np.linalg.svd(m)
    return u @ vt


@dataclass
class FragmentOrbitals:
    """Fragment-localized occupied and virtual orbitals of one mean field.

    `u_occ` (nocc, nocc) and `u_vir` (nvir, nvir) rotate the canonical
    orbitals into the local ones, fragment-contiguous; `occ_labels` and
    `vir_labels` give each local orbital's fragment, `occ_weights` and
    `vir_weights` its population there.
    """
    mol: object
    fragments: list
    scheme: str
    mo_coeff: np.ndarray
    nocc: int
    u_occ: np.ndarray
    u_vir: np.ndarray
    occ_labels: np.ndarray
    vir_labels: np.ndarray
    occ_weights: np.ndarray
    vir_weights: np.ndarray
    pm_gradient: dict = field(default_factory=dict)

    @property
    def nfrag(self):
        return len(self.fragments)

    @property
    def nvir(self):
        return self.u_vir.shape[0]

    @property
    def c_occ(self):
        return self.mo_coeff[:, :self.nocc] @ self.u_occ

    @property
    def c_vir(self):
        return self.mo_coeff[:, self.nocc:] @ self.u_vir

    def counts(self):
        """(occupied, virtual) orbital counts per fragment."""
        k = np.arange(self.nfrag)
        return (np.array([(self.occ_labels == f).sum() for f in k]),
                np.array([(self.vir_labels == f).sum() for f in k]))

    def transition_rotation(self):
        """T with x_canonical = T @ x_local over pairs ia = i*nvir + a."""
        return np.kron(self.u_occ, self.u_vir)

    def to_canonical(self, x_local):
        """(n_ov, k) local amplitudes -> canonical, without forming T."""
        x = np.asarray(x_local, float).reshape(self.nocc, self.nvir, -1)
        out = np.einsum('ij,jbk,ab->iak', self.u_occ, x, self.u_vir,
                        optimize=True)
        return out.reshape(self.nocc * self.nvir, -1)

    def to_local(self, x_canonical):
        """(n_ov, k) canonical amplitudes -> local, without forming T."""
        x = np.asarray(x_canonical, float).reshape(self.nocc, self.nvir, -1)
        out = np.einsum('ji,jbk,ba->iak', self.u_occ, x, self.u_vir,
                        optimize=True)
        return out.reshape(self.nocc * self.nvir, -1)

    def pair_labels(self):
        """(hole fragment, electron fragment) of every local pair ia."""
        return (np.repeat(self.occ_labels, self.nvir),
                np.tile(self.vir_labels, self.nocc))

    @classmethod
    def from_mf(cls, mf, fragments, scheme='lowdin', reference=None,
                conv_tol=FRAGMENT_PM_CONV_TOL,
                floor=LOCALIZED_ASSIGNMENT_FLOOR):
        """Localize `mf`'s orbitals onto `fragments` (atom-index lists).

        `reference`, a FragmentOrbitals at another geometry, starts the
        localization from the reference's local orbitals carried over to this
        geometry and orders the result like the reference, so that local
        orbital k means the same orbital along a scan.

        Refuses an orbital whose largest fragment population is below `floor`:
        it belongs to no fragment, and the local/charge-transfer split that
        everything downstream rests on is then not defined.
        """
        mol = mf.mol
        fragments = check_partition(mol, fragments)
        mo = np.asarray(mf.mo_coeff, float)
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        s = mol.intor_symmetric('int1e_ovlp')
        c_o, c_v = mo[:, :nocc], mo[:, nocc:]
        if scheme == 'iao':
            pops_o = population_tensor(mol, fragments, 'iao', orbocc=c_o, s=s)
            pops_v = population_tensor(mol, fragments, 'lowdin', s=s)
        else:
            pops_o = population_tensor(mol, fragments, scheme, s=s)
            pops_v = pops_o

        out = {}
        for tag, c, pops in (('occ', c_o, pops_o), ('vir', c_v, pops_v)):
            start = None
            if reference is not None:
                ref_c = reference.c_occ if tag == 'occ' else reference.c_vir
                start = _transport(mol, c, reference.mol, ref_c)
            u, _ = _localize(mol, c, pops, start=start, conv_tol=conv_tol)
            proj = pops(mol, c @ u)
            q = np.einsum('kii->ik', proj)
            labels, weights = q.argmax(axis=1), q.max(axis=1)
            grad = _stationarity(proj, labels)
            if weights.size and weights.min() < floor:
                worst = int(weights.argmin())
                raise ValueError(
                    f'{tag} local orbital {worst} has at most '
                    f'{weights[worst]:.3f} of its population on one fragment '
                    f'(floor {floor}); it belongs to no fragment, so the '
                    f'local / charge-transfer split is not defined. Use '
                    f'larger fragments or a smaller basis.')
            cl = c @ u
            if reference is not None:
                # local orbital k means the reference's local orbital k
                ref_c = reference.c_occ if tag == 'occ' else reference.c_vir
                ref_l = (reference.occ_labels if tag == 'occ'
                         else reference.vir_labels)
                order, signs = _match(mol, cl, reference.mol, ref_c)
                if not np.array_equal(labels[order], ref_l):
                    raise ValueError(f'{tag} local orbitals changed fragment '
                                     f'relative to the reference geometry')
            else:
                # fragment-contiguous, by orbital energy inside a fragment;
                # sign: the largest AO coefficient is positive
                energies = np.einsum('pi,pq,qi->i', cl, mf.get_fock(), cl)
                order = np.lexsort((energies, labels))
                big = cl[np.abs(cl).argmax(axis=0), np.arange(cl.shape[1])]
                signs = np.where(big < 0, -1.0, 1.0)[order]
            u = u[:, order] * signs[None, :]
            out[tag] = (u, labels[order], weights[order], grad)

        return cls(mol=mol, fragments=fragments, scheme=scheme, mo_coeff=mo,
                   nocc=nocc, u_occ=out['occ'][0], u_vir=out['vir'][0],
                   occ_labels=out['occ'][1], vir_labels=out['vir'][1],
                   occ_weights=out['occ'][2], vir_weights=out['vir'][2],
                   pm_gradient={'occ': out['occ'][3], 'vir': out['vir'][3]})


def _match(mol, c, ref_mol, ref_c):
    """(order, signs): column order of `c` matching `ref_c`, and the signs that
    make each matched overlap positive (signs are indexed in the new order).

    An optimal assignment on |overlap|, not a per-column argmax: inside one
    fragment the functional is nearly flat, so local orbitals of one fragment
    can rotate among themselves between geometries and several may overlap
    one reference orbital best. Nothing downstream depends on the order
    inside a fragment -- the diabats depend on the fragment subspaces only --
    so the assignment needs only to be one-to-one and keep each orbital on
    its fragment, which the caller checks.
    """
    from scipy.optimize import linear_sum_assignment
    t = c.T @ gto.mole.intor_cross('int1e_ovlp', mol, ref_mol) @ ref_c
    rows, cols = linear_sum_assignment(-np.abs(t))
    order = np.empty(t.shape[1], int)
    order[cols] = rows
    signs = np.sign(t[order, np.arange(t.shape[1])])
    signs[signs == 0] = 1.0
    return order, signs
