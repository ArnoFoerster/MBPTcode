"""One description of an active space of canonical MOs: the rule that picked it, and every index view its consumers ask of it.

`ActiveSpace` stores spatial MO indices in the ORIGINAL canonical order and
nothing else. Its constructors are the selection rules -- an energy window,
counts around the Fermi level, irrep character, Loewdin population on a set of
atoms -- and its views are the conventions consumers expect: the spatial
(core | active | external) partition, its interleaved spin-orbital twin, and
the permuted mean field that makes a non-contiguous selection contiguous.

Why selection by character exists
----------------------------------
A CONTIGUOUS energy window -- the n_occ_active highest occupied and
n_virt_active lowest virtual canonical orbitals -- is only the right active
space when the orbitals wanted happen to be the ones ranked there, and along a
dissociation curve they are not:

  * N2 below ~0.95 A: the occupied ordering changes and the top three occupied
    orbitals are (pi_y, 2sigma_u, 3sigma_g) -- the window takes ONE of the
    degenerate 1pi_u pair and leaves its partner in the frozen core. Every
    denominator with a core hole and an active particle is then exactly zero
    (eps_i - eps_p = 0), which is fatal to any perturbative treatment built on
    the window.
  * Any augmented basis: the diffuse virtuals fall between the valence
    antibonding orbitals, so an energy window picks diffuse orbitals into the
    active space and pushes sigma_u* out of it.

Both are failures of the SELECTION RULE, not of what is built on the active
space afterward. The fix is to pick the active orbitals by symmetry/character
and then PERMUTE the MOs so the selection is contiguous again.

Why permuting is safe
----------------------
A perturbative or CI treatment built on a canonical reference assumes a
diagonal Fock matrix -- every denominator is a sum of orbital energies. A
permutation of canonical MOs is still canonical: it only relabels, so the Fock
matrix stays diagonal and eps stays exact. (ROTATING orbitals -- localizing,
natural orbitals, anything mixing them -- would break that assumption and is
NOT what this module does.)
"""
import numpy as np
from pyscf import symm

#: N2 valence CAS(6,6): 3sigma_g + the 1pi_u pair occupied, the 1pi_g* pair +
#: 3sigma_u* virtual. Stable in character all the way from 0.8 to 4.0 A, unlike
#: the energy ordering.
N2_VALENCE_66 = ({'A1g': 1, 'E1ux': 1, 'E1uy': 1},
                 {'E1gx': 1, 'E1gy': 1, 'A1u': 1})


class ReorderedMF:
    """Minimal mean-field shim with the MOs permuted to
    [core occupied | active occupied | active virtual | external virtual].

    Quacks enough like a pyscf SCF object (mo_energy, mo_coeff, mo_occ,
    get_hcore) for a downstream integral builder, and carries the original's
    environment (with_screening, with_solvent) so a builder that refuses one
    still sees it; `n_occ_spatial` and `n_act_spatial` are the window sizes
    for that permuted ordering."""

    def __init__(self, mf, caslst):
        mo_occ = np.asarray(mf.mo_occ)
        caslst = np.asarray(sorted(caslst), dtype=int)
        occupied = mo_occ > 0
        act_occ = caslst[occupied[caslst]]
        act_vir = caslst[~occupied[caslst]]
        rest = np.setdiff1d(np.arange(len(mo_occ)), caslst)
        core = rest[occupied[rest]]
        ext = rest[~occupied[rest]]
        order = np.concatenate([core, act_occ, act_vir, ext])

        self.order = order
        self.mo_energy = np.asarray(mf.mo_energy)[order]
        self.mo_coeff = np.asarray(mf.mo_coeff)[:, order]
        self.mo_occ = mo_occ[order]
        self.e_tot = mf.e_tot
        self.get_hcore = mf.get_hcore
        self.with_screening = getattr(mf, 'with_screening', None)
        self.with_solvent = getattr(mf, 'with_solvent', None)
        self.n_occ_spatial = len(core)
        self.n_act_spatial = len(caslst)
        self.n_act_occ = len(act_occ)
        self.n_act_vir = len(act_vir)

    def __repr__(self):
        return (f'<ReorderedMF core={self.n_occ_spatial} '
                f'active={self.n_act_occ}o+{self.n_act_vir}v '
                f'external={len(self.mo_occ) - self.n_occ_spatial - self.n_act_spatial}>')


class ActiveSpace:
    """Which canonical spatial MOs of a closed-shell reference are active, and every index view of that choice.

    occ_active / virt_active are spatial MO indices in the original canonical
    order, strictly increasing, occupied in [0, nocc) and virtual in
    [nocc, norb), so two selection rules that pick the same orbitals compare
    equal whatever route built them. No constructor here ever mixes orbitals:
    the reference stays canonical, the Fock matrix stays diagonal, and every
    perturbative denominator stays a sum of orbital energies.
    """

    def __init__(self, norb, nocc, occ_active, virt_active):
        self.norb, self.nocc = int(norb), int(nocc)
        self.occ_active = _index_array(occ_active, 0, self.nocc, 'occ_active')
        self.virt_active = _index_array(virt_active, self.nocc, self.norb,
                                        'virt_active')

    @classmethod
    def from_counts(cls, eps, nocc, n_occ_active, n_virt_active):
        """The n_occ_active highest occupied and n_virt_active lowest virtual orbitals: the contiguous window around the Fermi level."""
        norb = len(eps)
        if n_occ_active > nocc or n_virt_active > (norb - nocc):
            raise ValueError(f"the active window asks for {n_occ_active} occupied "
                             f"and {n_virt_active} virtual orbitals; the reference "
                             f"has {nocc} and {norb - nocc}.")
        return cls(norb, nocc, np.arange(nocc - n_occ_active, nocc),
                   np.arange(nocc, nocc + n_virt_active))

    @classmethod
    def from_window(cls, eps, nocc, window):
        """Every orbital inside [E_F - window/2, E_F + window/2], E_F the midpoint of the gap."""
        norb = len(eps)
        occ_full, virt_full = np.arange(nocc), np.arange(nocc, norb)
        e_fermi = 0.5 * (eps[nocc - 1] + eps[nocc])
        lo, hi = e_fermi - 0.5 * window, e_fermi + 0.5 * window
        occ_active = occ_full[eps[occ_full] >= lo]
        virt_active = virt_full[eps[virt_full] <= hi]
        if len(occ_active) == 0 or len(virt_active) == 0:
            raise ValueError(f"window={window} selects an empty occ or virt active set.")
        return cls(norb, nocc, occ_active, virt_active)

    @classmethod
    def from_indices(cls, norb, nocc, occ_active, virt_active):
        """The window spelt out: index arrays into the canonical orbital list."""
        return cls(norb, nocc, occ_active, virt_active)

    @classmethod
    def from_irreps(cls, mol, mf, occ_irreps, vir_irreps):
        """Active orbitals chosen by irrep. Needs mol built with symmetry=True.

        occ_irreps / vir_irreps: {irrep label -> how many}. Occupied picks the
        HIGHEST-energy orbitals of that irrep (they are the frontier ones),
        virtual picks the LOWEST.

        Raises if an irrep does not have enough orbitals of the requested
        occupancy -- silently returning a smaller active space would change the
        method halfway along a scan."""
        mo_occ = np.asarray(mf.mo_occ)
        labels = irrep_labels(mol, mf.mo_coeff)
        occupied = mo_occ > 0
        chosen = []
        for spec, want_occ, take_last in ((occ_irreps, True, True),
                                          (vir_irreps, False, False)):
            for irrep, count in spec.items():
                idx = np.flatnonzero((labels == irrep) & (occupied == want_occ))
                if len(idx) < count:
                    raise ValueError(
                        f"asked for {count} {'occupied' if want_occ else 'virtual'} "
                        f"orbital(s) of irrep {irrep!r}, found {len(idx)}")
                chosen.extend(idx[-count:] if take_last else idx[:count])
        act = np.array(sorted(chosen), dtype=int)
        if len(set(act.tolist())) != len(act):
            raise ValueError('an orbital was selected twice; check the irrep specs')
        nocc = closed_shell_nocc(mo_occ)
        return cls(len(mo_occ), nocc, act[act < nocc], act[act >= nocc])

    @classmethod
    def from_population(cls, mf, atoms=None, ao_type='p', threshold=0.7,
                        n_occ=None, n_virt=None):
        """Active orbitals chosen by Loewdin population on one AO shell of a set of atoms -- the pi selector.

        A canonical MO is a candidate when its population on the `ao_type` AOs
        of `atoms` (None = every atom) exceeds `threshold`; population measures
        whether the orbital is built on the chromophore, not whether an
        excitation uses it. n_occ / n_virt then keep the FRONTIER candidates --
        the highest-energy occupied, the lowest-energy virtual, as in
        from_irreps -- and without them every candidate enters the window.
        Ranking the candidates by population instead would take a high-lying
        member of the same pi manifold over the pi* LUMO: in ethylene/cc-pVDZ
        four MOs carry the carbons' out-of-plane p character, and the two with
        the LARGEST population are the third and fourth of them.

        This SELECTS canonical orbitals, it never rotates them, so the model's
        Fock operator stays diagonal.

        Raises when the threshold leaves an empty occupied or virtual candidate
        set, or when it leaves fewer candidates than n_occ / n_virt ask for."""
        mo_occ = np.asarray(mf.mo_occ)
        nocc, norb = closed_shell_nocc(mo_occ), len(mo_occ)
        pop = lowdin_populations(mf, atoms=atoms, ao_type=ao_type)
        picked = []
        for lo, hi, count, take_last, what in ((0, nocc, n_occ, True, 'occupied'),
                                               (nocc, norb, n_virt, False, 'virtual')):
            idx = lo + np.flatnonzero(pop[lo:hi] > threshold)
            if len(idx) == 0:
                raise ValueError(
                    f"no {what} orbital has {ao_type!r} population above "
                    f"{threshold} on atoms {atoms} (largest {pop[lo:hi].max():.3f})")
            if count is not None:
                if len(idx) < count:
                    raise ValueError(
                        f"asked for {count} {what} orbital(s), only {len(idx)} "
                        f"have {ao_type!r} population above {threshold}")
                idx = idx[-count:] if take_last else idx[:count]
            picked.append(idx)
        return cls(norb, nocc, picked[0], picked[1])

    @property
    def env_occ(self):
        """Occupied orbitals outside the window (the frozen core)."""
        return np.setdiff1d(np.arange(self.nocc), self.occ_active)

    @property
    def env_virt(self):
        """Virtual orbitals outside the window."""
        return np.setdiff1d(np.arange(self.nocc, self.norb), self.virt_active)

    @property
    def act(self):
        """Active orbitals, occupied first: the order the active-space Hamiltonian's rows are in."""
        return np.concatenate([self.occ_active, self.virt_active])

    @property
    def n_act_occ(self):
        return len(self.occ_active)

    @property
    def n_act(self):
        return len(self.occ_active) + len(self.virt_active)

    @property
    def nelec_active(self):
        """Electrons in the window: doubly occupied active orbitals only."""
        return 2 * self.n_act_occ

    @property
    def is_contiguous(self):
        """True when the window is the energy-ordered frontier one, i.e. what from_counts builds."""
        n_virt_act = self.n_act - self.n_act_occ
        return (np.array_equal(self.occ_active,
                               np.arange(self.nocc - self.n_act_occ, self.nocc))
                and np.array_equal(self.virt_active,
                                   np.arange(self.nocc, self.nocc + n_virt_act)))

    def as_tuple(self):
        """(occ_active, virt_active, occ_env, virt_env), the four spatial index arrays."""
        return self.occ_active, self.virt_active, self.env_occ, self.env_virt

    def permuted_mf(self, mf):
        """Mean-field shim with the MOs permuted so this window is contiguous (see the module docstring on why that is safe)."""
        return ReorderedMF(mf, self.act)

    def contiguous_counts(self):
        """(n_occ_spatial, n_act_spatial), the window sizes for permuted_mf(mf)'s orbital ordering."""
        return self.nocc - self.n_act_occ, self.n_act

    def spatial_partition(self):
        """(core, active, external) SPATIAL index arrays for permuted_mf(mf), not for the original MO order."""
        n_core, n_act = self.contiguous_counts()
        return (np.arange(0, n_core, dtype=int),
                np.arange(n_core, n_core + n_act, dtype=int),
                np.arange(n_core + n_act, self.norb, dtype=int))

    def spin_orbital(self):
        """(core, active, external) SPIN-orbital index arrays: contiguous blocks of the interleaved list, alpha at 2p and beta at 2p+1, for permuted_mf(mf)."""
        n_core, n_act = self.contiguous_counts()
        n_occ_spin, n_act_spin = 2 * n_core, 2 * n_act
        return (np.arange(0, n_occ_spin, dtype=int),
                np.arange(n_occ_spin, n_occ_spin + n_act_spin, dtype=int),
                np.arange(n_occ_spin + n_act_spin, 2 * self.norb, dtype=int))

    def __repr__(self):
        return (f'<ActiveSpace {self.n_act_occ}o+{self.n_act - self.n_act_occ}v '
                f'in {self.nocc}o+{self.norb - self.nocc}v, '
                f'{"contiguous" if self.is_contiguous else "character-selected"}>')


def _index_array(idx, lo, hi, name):
    """A strictly increasing int index array inside [lo, hi); raises otherwise."""
    out = np.asarray(idx, dtype=int).ravel()
    if len(out) and (out[0] < lo or out[-1] >= hi or np.any(np.diff(out) <= 0)):
        raise ValueError(f'{name} must be strictly increasing indices in '
                         f'[{lo}, {hi}); got {out}')
    return out


def closed_shell_nocc(mo_occ):
    """Doubly occupied orbital count; raises unless the occupations are the aufbau [2 ... 2 0 ... 0].

    Every view here indexes the occupied block as [0, nocc), so a hole in the
    occupation list would label a virtual orbital as core."""
    mo_occ = np.asarray(mo_occ)
    if mo_occ.ndim != 1:
        raise ValueError('ActiveSpace describes a restricted closed-shell '
                         'reference; mo_occ is not a single occupation vector')
    nocc = int(np.count_nonzero(mo_occ > 0))
    if np.any(mo_occ[nocc:] > 0):
        raise ValueError('the occupied orbitals are not the lowest ones; a '
                         'non-aufbau reference has no [0, nocc) occupied block')
    return nocc


def irrep_labels(mol, mo_coeff):
    """Irrep label per MO. Needs mol built with symmetry=True."""
    return np.asarray(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                          mo_coeff))


def lowdin_populations(mf, atoms=None, ao_type='p'):
    """Loewdin population of every MO on the `ao_type` AO shells of `atoms` (None = every atom).

    The population of MO p on AO mu is [S^1/2 C]_{mu p}^2: the coefficients are
    taken in the symmetrically orthogonalized AO basis, so unlike a Mulliken
    partition the contributions are positive and sum to one over all AOs."""
    mol = mf.mol
    wanted = None if atoms is None else {int(a) for a in np.atleast_1d(atoms)}
    keep = np.array([(wanted is None or lab[0] in wanted) and lab[2][-1] == ao_type
                     for lab in mol.ao_labels(fmt=None)], dtype=bool)
    if not keep.any():
        raise ValueError(f'the basis has no {ao_type!r} AO on atoms {atoms}')
    w, v = np.linalg.eigh(mf.get_ovlp())
    c_orth = (v * np.sqrt(w)) @ v.T @ np.asarray(mf.mo_coeff)
    return np.einsum('mp,mp->p', c_orth[keep], c_orth[keep])
