"""ADC(2) behind the fragment partition and its gradient: the folded singles
operator at Omega_0 (`ADC2Operator`, a `SinglesOperator` of
`src.fragments.partition`), the `FragmentMethod` of a closed-shell DF-RHF
reference (`ADC2FragmentMethod`) and its `SinglesResponse` (`ADC2Response`),
for `src.gradients.fragment_diabatic`.

THE OPERATOR. The ADC(2) secular matrix has a diagonal doubles block, so
eliminating the doubles at a fixed energy gives the folded singles operator
of one spin channel (`ee_ri_fold.RIFoldADC2`),

    K(Omega) = M + C (Omega - D)^-1 C^T ,   dK/dOmega = -C (Omega - D)^-2 C^T ,

symmetric, with metric 1. The partition eliminates Q = (the other singles +
all doubles) at Omega_0 as the fold followed by its Feshbach step on
K(Omega_0); the fold's Omega dependence adds u^T (dK/dOmega) u to
dA_eff/dOmega (`partition` module docstring). Omega_0 must lie below the
lowest doubles energy D_min = 2 (e_LUMO - e_HOMO), where the fold has its
first pole; it is refused otherwise. Singlets and triplets are the two
channels (ADC(2) has no B block and no triplet instability). The orbitals are
Hartree-Fock; the zeroth-order diagonal of the working basis is e_a - e_i of
the Hartree-Fock energies.

THE REVERSE PASS (term 1 of the Lagrangian). The nuclear derivative of
s^T K(Omega_0) t at fixed Omega_0 equals that of the full ADC(2) matrix
contracted with (s, Y_s) and (t, Y_t), Y the doubles images: the derivative
of Phi(y) = y^T K(Omega_0) y (`ee_relaxed_density`, the functional whose
derivative is an ADC(2) root's force) polarized. Phi is quadratic in y, so
the weighted sum of pairs, sum_k w_k sym(s_k t_k^T), is diagonalized in the
span of its vectors, lambda_j z_j z_j^T, and each z_j contributes lambda_j
times Phi's pieces (`ee_gradient.adc2_pieces` with the weight lambda_j in
place of a root's singles weight): ONE correlation force
(`df_correlation_force.correlation_force`, its one Z-vector) takes the sum.
No MP2 or Hartree-Fock term enters: an element of A_eff is an excitation
quantity.

THE GAUGE (term 3). That correlation force takes the Fock derivative gamma in
a non-canonical basis and keeps no canonical condition inside the occupied or
the virtual block, so its orbitals move in the symmetric gauge:
`frame_force` is `orbital_coupling.symmetric_gauge_coupling`, and no
canonical rotation (nor its refusal at degenerate orbitals) enters.

WHAT IS NOT HERE. A frozen core (every occupied orbital is localized and
correlated); an environment (the fold's reaction field is not differentiated
here); open shells; the tiled force route (the dense pieces hold the
(ov|ov) block whole: small and medium molecules).
"""
import numpy as np

from src.Base.environment_seam import NoEnvironment
from src.Base.utils.parallel.mpi_grid import lockstep
from src.Base.utils.timing import stage
from src.SingleReference.ADC.eeADC import ee_ri_fold
from src.fragments.partition import split_xy
from src.gradients import df_correlation_force as dcf
from src.gradients.adc.ee_gradient import adc2_pieces
from src.gradients.orbital_coupling import symmetric_gauge_coupling
from src.gradients.reference_mean_field import ReferenceMeanField

#: The singles channels of the folded operator.
SPINS = ('singlet', 'triplet')
#: The operator's routes: 'fold' applies the folded operator matrix-free
#: (blocks of the partition up to FRAGMENT_MATRIX_FREE_DENSE_MAX rows formed
#: by its action), 'dense' forms K(Omega_0) whole (n_ov folded products).
ROUTES = ('auto', 'dense', 'fold')
#: Columns per folded product when K(Omega_0) is formed whole.
DENSE_BLOCK = 64
#: Eigenvalues of the pair matrix sum_k w_k sym(s_k t_k^T) below this
#: fraction of its largest carry no weight (`ADC2Response.pair_force`): the
#: pair vectors' span holds the rest exactly. A module constant, owned here
#: (the shared constants file is not this module's).
PAIR_RANK_CUT = 1e-14


class ADC2Operator:
    """K(Omega_0) of one spin channel of closed-shell ADC(2) on the canonical
    (i, a) pairs, Hartree-Fock orbitals, no frozen core (module docstring):
    the `SinglesOperator` of `src.fragments.partition`."""

    tda = True

    def __init__(self, fold, eps, omega, dense=None):
        self.fold = fold
        self.eps = np.asarray(eps, float)
        self.nocc = int(fold.no)
        self.nvir = int(fold.nv)
        self.omega = float(omega)
        self.dense = dense

    @property
    def spin(self):
        """The channel, 'singlet' or 'triplet' (SPINS)."""
        return self.fold.spin

    @property
    def n_ov(self):
        return self.nocc * self.nvir

    @property
    def dim(self):
        return self.n_ov

    @property
    def metric(self):
        return np.ones(self.n_ov)

    def _act(self, x, derivative):
        x = np.asarray(x, float)
        vec = x.ndim == 1
        # the fold's products are collective: every rank hands it rank 0's
        # vectors
        x = lockstep(np.ascontiguousarray(x.reshape(self.dim, -1)), check=True)
        if derivative:
            y = self.fold.fold_images(x, [self.omega], derivative=[True])[1][0]
        elif self.dense is not None:
            y = self.dense @ x
        else:
            y = self.fold.apply(x, np.full(x.shape[1], self.omega))[0]
        return y[:, 0] if vec else y

    def apply(self, x):
        """K(Omega_0) x for x (n_ov, k)."""
        return self._act(x, False)

    def apply_domega(self, x):
        """dK/dOmega x at Omega_0, x (n_ov, k)."""
        return self._act(x, True)

    @classmethod
    def from_mean_field(cls, mol, mf, spin, auxbasis, omega, route='auto'):
        """The operator of `spin` at Omega_0 = `omega` on the DF-RHF `mf`,
        the doubles fitted in `auxbasis`; route 'auto' (= 'fold'), 'dense'
        or 'fold' (ROUTES)."""
        if spin not in SPINS:
            raise ValueError(f'spin must be one of {SPINS}, got {spin!r}')
        if route not in ROUTES:
            raise ValueError(f'route must be one of {ROUTES}, got {route!r}')
        if omega is None:
            raise ValueError('the ADC(2) operator depends on Omega: give '
                             'omega0, the energy the doubles are folded at')
        f = ee_ri_fold.bare_factors(mol, mf, frozen=0, auxbasis=auxbasis)
        fold = ee_ri_fold.RIFoldADC2(f, spin, level='adc2')
        if not float(omega) < fold.d_min:
            raise ValueError(
                f'Omega_0 = {float(omega):.6f} Ha is not below the lowest '
                f'doubles energy {fold.d_min:.6f} Ha, where the folded '
                f'operator has its first pole')
        op = cls(fold, mf.mo_energy, omega)
        if route == 'dense':
            n = op.n_ov
            k = np.empty((n, n))
            for c0 in range(0, n, DENSE_BLOCK):
                c1 = min(c0 + DENSE_BLOCK, n)
                cols = np.zeros((n, c1 - c0))
                cols[np.arange(c0, c1), np.arange(c1 - c0)] = 1.0
                k[:, c0:c1] = op.apply(cols)
            op.dense = 0.5 * (k + k.T)
        return op


class ADC2FragmentMethod(ReferenceMeanField):
    """The `FragmentMethod` (`src.gradients.fragment_diabatic`) of ADC(2) on a
    closed-shell DF-RHF reference: `scf_factory(mol)` returns the mean field
    (density-fitted, as the correlation force's Fock skeleton is its JK
    fit's), `spin` the channel, `auxbasis` the fit of the doubles and of the
    correlation force (named: the two must be one fit). One SCF per geometry
    (`ReferenceMeanField`), in the gas phase."""

    def __init__(self, mol, scf_factory, spin='singlet', auxbasis=None,
                 mf=None):
        if spin not in SPINS:
            raise ValueError(f'spin must be one of {SPINS}, got {spin!r}')
        if auxbasis is None:
            raise ValueError('name the auxiliary basis of the doubles: the '
                             'fold and the correlation force must fit in one')
        self.mol0, self.scf_factory = mol, scf_factory
        self.spin, self.auxbasis = spin, auxbasis
        self.nocc = mol.nelectron // 2
        self.environment = NoEnvironment()
        self.timer = None
        self.construction_seconds = {}
        self._init_reference(mol, scf_factory, mf)

    def phase(self, key):
        """Time one stage into `self.timer[key]`, or nothing (None)."""
        return stage(self.timer, key)

    def require_differentiable_environment(self):
        """The gas phase: nothing to refuse."""

    def singles(self, mol, mf, route='auto', omega0=None):
        """(ADC2Operator at omega0, ADC2Response) at one geometry."""
        op = ADC2Operator.from_mean_field(mol, mf, self.spin, self.auxbasis,
                                          omega0, route)
        return op, ADC2Response(op, mol, mf, self.auxbasis)

    def ground_energy(self, mf):
        """E_HF + E_MP2 in the doubles' fit: the ground state ADC(2)
        excitation energies are measured from."""
        ri = dcf.RIFactors(mf.mol, mf, self.auxbasis)
        return mf.e_tot + dcf.mp2_pieces(ri.B, mf.mo_energy, 0, self.nocc)[0]

    def ground_gradient(self, mf):
        """dE/dR of E_HF + E_MP2 (`df_correlation_force.mp2_force`)."""
        return dcf.mp2_force(mf, mf.mol, 0, self.auxbasis)[0]

    def refreeze(self, mol, mf=None):
        return ADC2FragmentMethod(mol, self.scf_factory, self.spin,
                                  self.auxbasis, mf=mf)


class ADC2Response:
    """The `SinglesResponse` of one ADC2Operator (module docstring)."""

    def __init__(self, operator, mol, mf, auxbasis):
        self.operator, self.mol, self.mf = operator, mol, mf
        self.auxbasis = auxbasis
        # {name: detail} of the last pair_force's Z-vector events
        self.flags = {}

    def pair_terms(self, pairs):
        """[(lambda_j, z_j)] with sum_k w_k sym(s_k t_k^T) = sum_j lambda_j
        z_j z_j^T, z_j unit, from the span of the pair vectors."""
        cols, weights = [], []
        for v, same, wgt in pairs:
            s, t = split_xy(v, self.operator.n_ov)[0].T
            cols += [s, t]
            weights.append(wgt)
        basis, _ = np.linalg.qr(np.column_stack(cols))
        small = np.zeros((basis.shape[1],) * 2)
        for k, wgt in enumerate(weights):
            bs, bt = basis.T @ cols[2 * k], basis.T @ cols[2 * k + 1]
            small += 0.5 * wgt * (np.outer(bs, bt) + np.outer(bt, bs))
        lam, vec = np.linalg.eigh(small)
        keep = np.abs(lam) > PAIR_RANK_CUT * max(np.abs(lam).max(), 1e-300)
        return [(float(lam[j]), basis @ vec[:, j]) for j in np.flatnonzero(keep)]

    def pair_force(self, pairs):
        """sum_k w_k s_k^T dK(Omega_0)/dR t_k (natm, 3): one correlation
        force on the summed pieces (module docstring)."""
        op = self.operator
        C = np.asarray(self.mf.mo_coeff)
        nmo, nocc = C.shape[1], op.nocc
        gamma = Gam = None
        for lam, z in self.pair_terms(pairs):
            g, G, _, _ = adc2_pieces(op.fold, z, op.omega, lam, op.fold.spin, 0,
                                     nocc, nmo)
            gamma = g if gamma is None else gamma + g
            Gam = G if Gam is None else Gam + G
        if gamma is None:
            self.flags = {}
            return np.zeros((self.mol.natm, 3))
        ri = dcf.RIFactors(self.mol, self.mf, self.auxbasis)
        events = []
        force, _, _ = dcf.correlation_force(self.mol, self.mf, ri, gamma, Gam,
                                            0, flags=events)
        self.flags = {e['flag']: {'message': e['message']} for e in events}
        return force

    def frame_force(self, dc_o, dc_v):
        """The symmetric gauge: - sum D (.) A^sym,
        `symmetric_gauge_coupling`."""
        return symmetric_gauge_coupling(self.mol, self.mf, self.operator.nocc,
                                        dc_o.T, -dc_v)
