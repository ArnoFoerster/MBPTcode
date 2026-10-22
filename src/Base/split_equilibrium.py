"""Split equilibrium solvation: the excitation's state-specific reaction field
relaxed at eps_static, its transition-density reaction field kept optical.

THE PARTITION. The solvated BSE of Duchemin, Guido, Jacquemin and Blase,
Chem. Sci. 9, 4430 (2018), Eqs. (17)-(18), reaches the continuum in three
places:

    Eq. (18)   Delta eps_p = -/+ 1/2 <pp|W[v + vt] - W[v]|pp>     QP levels
    direct     -(ij| W[v + vt] |ab)                              e-h attraction
    exchange   2 kappa (ia| v + vt |jb)                          LR term

The first two are built from orbital DENSITIES: summed over the exciton they
are the reaction field of the excitation's density difference, -1/2 drho R drho
in the screened form, which is the state-specific (SS) solvation of the excited
state (Duchemin et al.'s SS-like part). The exchange term is built from the
TRANSITION density rho_ia, which oscillates at the excitation frequency, and
is the linear-response (LR) term of LR-PCM (Cammi and Mennucci, J. Chem. Phys.
110, 9877 (1999)).

Equilibrium is a statement about the SLOW polarization: once the solvent nuclei
have reoriented around the relaxed excited state, its reaction field to the
state's own density is the static one, R(eps_static). A transition density has
no time-averaged component for the nuclei to follow, so its reaction field
stays electronic, R(eps_inf), in equilibrium as out of it. That is the
partition behind the corrected-LR scheme (Caricato, Mennucci, Tomasi, Ingrosso,
Cammi, Corni and Scalmani, J. Chem. Phys. 124, 124520 (2006)), which lets the
state-specific term alone equilibrate, and behind its SS + LR combination
(Guido, Chrayteh, Scalmani, Mennucci and Jacquemin, J. Chem. Theory Comput.
17, 5155 (2021)). Equilibrium LR-PCM (Scalmani, Frisch, Mennucci,
Tomasi, Cammi and Barone, J. Chem. Phys. 124, 094107 (2006)) instead puts the
transition term at eps_static too -- the analytic-gradient convenience of
excited-state optimizations, not a physical statement -- and overstates a
bright local state's equilibrium shift by exactly that term.

SO, in this object:

    Eq. (18) and the direct W      W[v + vt(eps_static)]       relaxed
    exchange                       v + vt(eps_inf)             optical
    ground state                   PCM(eps_static)             unchanged

which is the default of `environment.equilibrium_environment`; the all-static
response stays available as `SolventScreening(equilibrium=True)` ('full'). A
triplet has no exchange term, so the two coincide there; for a charge-transfer
singlet the transition density is small and they nearly do.

HOW THE KERNEL CARRIES TWO DIELECTRICS. Both dressed factors share the fit M
and differ in the metric root, D_inf = M^T R_inf and D_s = M^T R_s with
R = (V + vt)^(1/2), so D_s = D_inf T exactly with T = R_inf^-1 R_s. The BSE
kernel keeps D_inf -- its exchange is then (ia|v + vt_inf|jb) -- and is handed
W' = T W_s T^T, W_s the static screening built on D_s, so that its direct term
B_inf^T W' B_inf = B_s^T W_s B_s is the static one. No kernel changes;
`ExcitedStateChain` builds D_s, W_s and T, and its force carries both gauges
with their roots' adjoints through T.

Only `ExcitedStateChain` assembles this. Every route that reads ONE
interaction -- the dense and DF integral hand-outs, the COHSEX operator, a
correlation energy's frequency fold -- is refused rather than handed the
optical half alone.
"""


class SplitEquilibrium:
    """A non-equilibrium continuum read as the split equilibrium response.

    Delegates everything else to the optical continuum it wraps: the ground
    state (`mean_field`, at eps_static), the cavity, and the optical dressing
    of the factor the exchange term uses (`aux_kernel` and its adjoint).
    `static_response()` is the full-equilibrium partner on the same cavity.
    """

    equilibrium = 'split'

    def __init__(self, optical):
        if getattr(optical, 'equilibrium', False):
            raise ValueError('wrap the NON-equilibrium continuum; the split '
                             'takes its static half from its partner')
        if getattr(optical, 'eps_static', None) is None:
            raise ValueError('split equilibrium needs eps_static: pass it or '
                             'a solvent name')
        self._hold(optical, optical.equilibrium_partner())

    def _hold(self, optical, static):
        """Every attribute a split holds: the optical continuum and its
        static half (None where a subclass has none)."""
        self.optical, self.static = optical, static

    def __getattr__(self, name):
        optical = self.__dict__.get('optical')
        if optical is None:
            raise AttributeError(name)
        return getattr(optical, name)

    @property
    def collapses(self):
        """eps_static == eps_inf: the split IS the non-equilibrium object."""
        return self.static.eps == self.optical.eps

    def static_response(self):
        """The continuum at eps_static the state-specific part relaxes in."""
        return self.static

    def equilibrium_partner(self):
        return self

    def non_equilibrium_partner(self):
        """The optical continuum, which a correlation energy reads in any mode."""
        return self.optical

    def for_geometry(self, mol):
        """The split around the atoms of `mol`."""
        if mol is self.optical.mol:
            return self
        return SplitEquilibrium(self.optical.for_geometry(mol))

    def _refused(self, what):
        raise ValueError(
            f'{what} reads one interaction, and the split equilibrium response '
            f'has two (exchange optical, direct and Eq. (18) static); only '
            f'ExcitedStateChain assembles it. Use the full equilibrium '
            f"object, equilibrium_environment(env, 'full'), for this route.")

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        self._refused('a four-index integral route')

    def kernel_ao(self, mol):
        self._refused('a four-index integral route')

    def whitened_transform(self, mol, mf):
        self._refused('a density-fitted integral route')

    def static_self_energy(self, mf, mol=None):
        self._refused('the static COHSEX operator')

    def static_self_energy_adjoint(self, mf, weights):
        self._refused('the static COHSEX operator')

    def dynamic_factor(self, omega):
        self._refused("a correlation energy's frequency fold")

    def __repr__(self):
        return (f'SplitEquilibrium({self.optical!r} in the exchange, '
                f'eps={self.static.eps:.4f} (static, equilibrium) in the '
                f'direct term and Eq. (18))')


#: The two ways an excitation reaches its state-specific solvation: Duchemin's
#: (Eq. (18) and the dressed direct W, `SplitEquilibrium`) and the classical
#: one (`ClassicalStateSpecific`).
SOLVENT_SCHEMES = ('duchemin', 'classical-ss')

#: Which density difference the classical state-specific energy reads.
SS_DENSITIES = ('relaxed', 'unrelaxed')


class ClassicalStateSpecific(SplitEquilibrium):
    """Scheme B: the BSE's state-specific-like part replaced by the classical
    state-specific energy of the excitation's density difference.

    The split partition (module docstring) keeps three terms; this object
    keeps the ground state at eps_static and the transition-density exchange
    at eps_inf, as `SplitEquilibrium` does, and puts Eq. (18) and the direct
    W on the BARE interaction: its static partner is no continuum at all, so
    D_s = M^T V^(1/2), T = R_inf^-1 V^(1/2), Delta W = 0 and no
    quasiparticle level moves. That is the partition of a wavefunction
    method with a perturbative state-specific term (PTE ground state, LR on
    the transition density, SS added after): omega_B is the excitation
    without its density's reaction field, and

        E_B = omega_B + E_SS[drho],   E_SS = 1/2 dV^T Q dV

    (`SolventScreening.state_specific_energy`), with drho the orbital-relaxed
    difference density (d omega_B / d h, its orbital response taken in the
    frozen ground-state field) or the unrelaxed amplitude density, and Q at
    eps_inf (vertical) or eps_static (equilibrium). E_SS is a single-point
    term: the force of a scheme B surface is omega_B's.

    ss_density: 'relaxed' (default) or 'unrelaxed'. ss_equilibrium: False
    (vertical, Q(eps_inf)) or True (Q(eps_static)).
    """

    scheme = 'classical-ss'

    def __init__(self, optical, ss_density='relaxed', ss_equilibrium=False):
        super().__init__(optical)
        if ss_density not in SS_DENSITIES:
            raise ValueError(f'ss_density={ss_density!r}: one of '
                             f'{SS_DENSITIES}')
        self._hold_ss(ss_density, bool(ss_equilibrium))

    def _hold_ss(self, ss_density, ss_equilibrium):
        """Every attribute of the state-specific energy (None where a
        subclass takes none)."""
        self.ss_density, self.ss_equilibrium = ss_density, ss_equilibrium

    @property
    def collapses(self):
        """Never: the bare partner differs from the optical continuum
        wherever the optical one screens at all."""
        return False

    def static_response(self):
        """The interaction Eq. (18) and the direct W read: the bare one."""
        return None

    def ss_continuum(self):
        """The continuum E_SS is taken in: optical (vertical) or static."""
        return self.static if self.ss_equilibrium else self.optical

    def for_geometry(self, mol):
        if mol is self.optical.mol:
            return self
        return ClassicalStateSpecific(self.optical.for_geometry(mol),
                                      ss_density=self.ss_density,
                                      ss_equilibrium=self.ss_equilibrium)

    def __repr__(self):
        mode = 'equilibrium' if self.ss_equilibrium else 'vertical'
        return (f'ClassicalStateSpecific({self.optical!r} in the exchange, '
                f'bare direct term and Eq. (18), E_SS of the '
                f'{self.ss_density} density, {mode})')


class LinearResponseOnly(ClassicalStateSpecific):
    """Scheme B's partition with no ground state of its own: what a GW-BSE
    state reads inside a self-consistent state-specific loop
    (`src.Base.solvation_loop`).

    The loop owns the slow and the fast reaction field of the state's whole
    relaxed density and hands the state a mean field converged in fixed
    surface charges; the BSE keeps only the fast answer to the transition
    density, its exchange dressed at eps_inf, with Eq. (18) and the direct W
    on the bare interaction as in `ClassicalStateSpecific`. Built on the
    medium's optical continuum (`reaction_field.ContinuumField
    .transition_environment()`, which has no eps_static), so it has no static
    partner and no E_SS, and its mean field is the one the loop converged in
    the charges it holds: `mean_field` hands the factory's back untouched
    (no continuum attached, nothing re-converged), as the gas phase does.
    """

    # not a `SOLVENT_SCHEMES` choice of `environment.solvent_scheme`: the
    # partition a state model reads inside a loop, built by the model itself
    scheme = 'loop'

    def __init__(self, optical):
        if getattr(optical, 'equilibrium', False):
            raise ValueError('wrap the NON-equilibrium (optical) continuum')
        # no static half and no E_SS: both are the loop's
        self._hold(optical, None)
        self._hold_ss(None, None)

    @property
    def vanishes(self):
        """eps_inf = 1: the continuum answers nothing, and the BSE is the gas
        phase's (the state model then drops it, so the gas limit holds in the
        bits rather than through a dressing by an exact zero)."""
        return self.optical.eps == 1.0

    def ss_continuum(self):
        raise ValueError('inside a solvation loop the state-specific field is '
                         "the loop's; there is no first-order E_SS")

    def mean_field(self, mol, scf_factory):
        mf = scf_factory(mol)
        if hasattr(mf, 'with_solvent'):
            raise ValueError('inside a solvation loop the mean field carries '
                             "the loop's charges, not a continuum of its own")
        return mf

    def for_geometry(self, mol):
        if mol is self.optical.mol:
            return self
        return LinearResponseOnly(self.optical.for_geometry(mol))

    def __repr__(self):
        return (f'LinearResponseOnly({self.optical!r} in the exchange, bare '
                f'direct term and Eq. (18); the state-specific field is a '
                f"solvation loop's)")
