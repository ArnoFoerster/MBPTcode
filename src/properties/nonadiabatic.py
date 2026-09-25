"""Derivative couplings d_IJ = <Psi_I | d/dR Psi_J> between BSE states, and
between the ground state and one of them.

The second of the two electronic couplings `rates.py` is written around. The
spin-orbit one is a matrix element of a GIVEN operator over the Casida vectors
and costs nothing beyond them. This one differentiates the WAVEFUNCTION, so it
sees the geometry dependence of the orbitals and of the amplitudes -- the same
object an analytic gradient sees. A derivative coupling costs what a gradient
costs, not what a property costs, and that is the whole difficulty.

WHAT IS COMPUTED HERE is the finite-difference coupling from OVERLAPS of the
BSE eigenvectors at displaced geometries, the construction Plasser,
Ruckenbauer, Mai, Oppel, Marquetand and Gonzalez (J. Chem. Theory Comput. 12,
1207 (2016)) implement for arbitrary wavefunctions, in its leading-order form
for a single-excitation ansatz. It is the validation route and the reference an
analytic term has to reproduce, exactly as `FiniteDifferenceGradient` is for
forces, and it costs 6 natm + 1 BSE solves -- one finite-difference gradient,
with every state pair coming off the same solves.

THE TRANSPORTED RPA METRIC. With T_pq = <phi_p^A|phi_q^B> the molecular-orbital
overlap between the two geometries, evaluating <0|[O_I(A), O_J^+(B)]|0> for the
excitation operators O^+ = sum (X a_a^+ a_i - Y a_i^+ a_a) of each geometry
gives

    S_IJ = D sum_ijab ( X^I_ia X^J_jb - Y^I_ia Y^J_jb ) [(T^oo)^-1]_ji T^vv_ab
    S_0J = sqrt(2) D sum_jb (X^J - Y^J)_jb [(T^oo)^-1 T^ov]_jb
    S_00 = D ,     D = det(T^oo)^2 .

EVERY INDEX PAIR BELONGS TO ONE GEOMETRY. (T^oo)^-1 carries a ket-occupied and
a bra-occupied index, T^vv a bra-virtual and a ket-virtual one, and the
amplitudes are contracted only against indices of their own geometry -- which
is what the determinant algebra produces and is not optional. Replacing
[(T^oo)^-1 T^ov]_jb by T^ov_jb differs at O(h^2) and looks harmless, but it
contracts a BRA occupied index against a KET amplitude and loses the invariance
below: on a rigidly translated formaldehyde, where S_0J is zero by symmetry for
both low roots, a random occupied/virtual rotation of the displaced orbitals
moves the correct form by 9e-17 and that one by 4e-04.

THE MINUS SIGN IS FORCED, NOT CHOSEN. At A = B the orbital overlaps are unit
matrices and S_IJ collapses to sum (X^I X^J - Y^I Y^J), which is the Casida
metric: every solver here returns <X|X> - <Y|Y> = 1 and (X+Y)^T(X-Y) = 1, from
which X^I.X^J - Y^I.Y^J = delta_IJ follows identically. X alone, X+Y, or the
XX + YY combination that a SYMMETRIC operator takes all give S(A,A) != 1, and
a finite difference of that carries an O(1/h) term that is not a derivative of
anything. Himmelsbach and Holzer (J. Chem. Phys. 161, 244105 (2024)) reach the
same split from the BSE state-to-state density: their Eqs. (18)-(21) build the
symmetric block from R.R + L.L and the skew block from R.L + L.R with
R = X + Y, L = X - Y, and the second expands to 2(XX - YY). The derivative of
an overlap is skew-symmetric -- T = 1 + h K with K antisymmetric, because
<p|q> = delta_pq at every geometry -- so the minus combination is the one a
coupling takes, as their Eq. (37) says for the ground-state element.

MO PHASES AND DEGENERATE ROTATIONS CANCEL IDENTICALLY. A sign flip or an
orthogonal rotation U of the displaced occupied orbitals sends T^oo -> T^oo U
and X^J -> U^T X^J, because the BSE was solved in that very basis, and the
product above is unchanged. Only the overall sign of each Casida vector and the
energy ordering of the roots have to be fixed against the reference, which is
what `align_roots` does. This is the structural advantage of the overlap route:
the near-degenerate orbital noise that dominates a finite-difference GRADIENT
cannot enter a finite-difference coupling.

TRANSLATIONAL INVARIANCE IS NOT WHAT IT IS USUALLY SAID TO BE. Translating the
nuclei rigidly shifts the Born-Oppenheimer wavefunction, Psi(r; R+d) =
Psi(r-d; R), so

    sum_A d_IJ^A = - <Psi_I| sum_k grad_k |Psi_J> ,

the velocity-gauge transition moment, which is NOT zero. `translational_sum`
returns it from one-electron integrals at the reference geometry and is the
sharpest available check on the whole displacement machinery. The sum rule
sum_A d_IJ^A = 0 quoted in the dynamics literature belongs to the
ELECTRON-TRANSLATION-FACTOR-corrected coupling (Fatehi, Alguire, Shao and
Subotnik, J. Chem. Phys. 135, 234105 (2011)), a different object; the identity
above is the size of that correction. The length-gauge form
sum_A d_IJ^A = (E_J - E_I) <I|r|J> holds for exact eigenstates and NOT for
BSE@GW, whose diagonal is a quasiparticle spectrum rather than a one-body
operator, so the commutator behind the hypervirial relation is broken.

WHAT IS NOT COMPUTED HERE. Send and Furche (J. Chem. Phys. 132, 044107 (2010))
and Ou, Bellchambers, Furche and Subotnik (J. Chem. Phys. 142, 064114 (2015))
show the response-theory coupling to be the Hellmann-Feynman contraction
<I|dH/dR|J>/(E_J - E_I) plus orbital-relaxation and Pulay terms. Differencing
overlaps of the pseudo-wavefunction (Ou, Alguire and Subotnik, J. Phys. Chem. B
119, 7150 (2015)) captures all of them for the amplitudes and orbitals as they
actually move, and none of the terms that the pseudo-wavefunction ansatz itself
leaves out -- which is a choice of what "the wavefunction" is, not a small
neglected quantity, and whose known failure is the behaviour at a conical
intersection.
"""
import numpy as np
from pyscf import gto

from src.Base.constants import NUCLEAR_FD_STEP
from src.SingleReference.LinearResponse.linear_response import check_normalization
from src.properties.spin_orbit import EXCITED_SPIN_FACTOR, GROUND_SPIN_FACTOR
from src.properties.surface import driven_chain

#: Bohr displacement of one Cartesian component. The central difference has an
#: O(h^2) truncation error and an O(eps/h) noise floor from the Casida vectors'
#: own convergence, the same trade `FiniteDifferenceGradient` makes for an
#: energy; a mean field converged to 1e-13 and a dense Casida support 1e-3.
class DerivativeCouplings:
    """d_IJ over the ground state and a set of BSE roots, in 1/Bohr.

    `d[I, J]` is the (natm, 3) array <Psi_I | d/dR Psi_J>, with index 0 the
    ground state and index 1 + k the reference root `states[k]`. Indexing with
    a pair returns that array, so `nac[0, 1]` is the S0-S1 coupling.
    """

    def __init__(self, d, states, step, diagnostics):
        self.d, self.states = np.asarray(d, float), tuple(states)
        self.step, self.diagnostics = float(step), dict(diagnostics)

    def __getitem__(self, pair):
        i, j = pair
        return self.d[i, j]

    @property
    def antisymmetry(self):
        """max |d_IJ + d_JI|, which the stencil does NOT enforce.

        The reference-anchored difference is antisymmetric only because the
        states at the displaced geometries really are orthonormal, so this
        number measures the truncation error, the root assignment and the
        solver's own convergence at once.
        """
        return float(np.abs(self.d + self.d.transpose(1, 0, 2, 3)).max())

    def label(self):
        return (f'derivative couplings over the ground state and roots '
                f'{list(self.states)} [overlap finite difference, '
                f'h = {self.step} Bohr]')


def mo_overlap(mol_a, mo_a, mol_b, mo_b):
    """T_pq = <phi_p^A | phi_q^B>, the MO overlap between two geometries.

    The atomic-orbital cross overlap is exact for any separation; the basis
    functions of the two geometries sit on different centres and `intor_cross`
    integrates them against each other.
    """
    s_ao = gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)
    return np.asarray(mo_a).T @ s_ao @ np.asarray(mo_b)


def state_overlap(t, nocc, x_a, y_a, x_b, y_b):
    """(1 + n_A, 1 + n_B) overlaps <Psi_I(A)|Psi_J(B)>, ground state at 0.

    The transported RPA metric of the module docstring. Row and column 0 are
    the reference determinant, whose ground-to-excited block carries the
    sqrt(2) of the singlet spin adaptation -- the same factor
    `oscillator_strengths` and `spin_orbit.ground_state_element` carry for
    <X|X> - <Y|Y> = 1.

    Exact to O(h^2) in the displacement: the dropped pieces are the alpha-beta
    spin cross term of the determinant overlap and the T^vo (T^oo)^-1 T^ov
    screening of T^vv, both quadratic in the occupied-virtual block of T, which
    is itself O(h). That is the order of the central difference taken of it.
    """
    t = np.asarray(t, float)
    nvir = t.shape[0] - nocc
    x_a, y_a = check_normalization(x_a, y_a)
    x_b, y_b = check_normalization(x_b, y_b)
    n_a, n_b = x_a.shape[1], x_b.shape[1]
    xa = x_a.reshape(nocc, nvir, n_a)
    ya = y_a.reshape(nocc, nvir, n_a)
    xb = x_b.reshape(nocc, nvir, n_b)
    yb = y_b.reshape(nocc, nvir, n_b)
    t_oo, t_ov = t[:nocc, :nocc], t[:nocc, nocc:]
    t_vo, t_vv = t[nocc:, :nocc], t[nocc:, nocc:]
    # (T^oo)^-1 runs ket-occupied by bra-occupied; the closed-shell reference
    # overlap is det(T^oo) once per spin string.
    inv_oo = np.linalg.inv(t_oo)
    det2 = np.linalg.det(t_oo) ** 2

    out = np.zeros((n_a + 1, n_b + 1))
    out[0, 0] = det2
    out[0, 1:] = GROUND_SPIN_FACTOR * det2 * np.einsum(
        'jbJ,jb->J', xb - yb, inv_oo @ t_ov, optimize=True)
    out[1:, 0] = GROUND_SPIN_FACTOR * det2 * np.einsum(
        'iaI,ai->I', xa - ya, t_vo @ inv_oo, optimize=True)
    for u, v in ((xa, xb), (ya, yb)):
        moved = np.einsum('iaI,ji,ab->jbI', u, inv_oo, t_vv, optimize=True)
        block = det2 * np.einsum('jbI,jbJ->IJ', moved, v, optimize=True)
        out[1:, 1:] += EXCITED_SPIN_FACTOR * (block if u is xa else -block)
    return out


def align_roots(s, nstates):
    """(order, sign, weight) matching displaced roots to the first `nstates`.

    The displaced solve returns its roots in its own energy order and with an
    arbitrary overall sign, and both are fixed against the reference by the
    overlap matrix the coupling is built from: each reference root takes the
    unused displaced root of largest |S|, then the sign that makes S positive.

    `weight` is the smallest |S| any assignment had to accept. A value far
    below one is a root the reference cannot identify -- a near-degenerate pair
    or a genuine crossing inside the displacement -- and the coupling built
    from it is a number about two states that are not the two asked for.
    """
    s = np.asarray(s, float)
    block = np.abs(s[1:, 1:])
    order, sign = np.zeros(nstates, int), np.ones(nstates)
    # Largest element first, so an unambiguous root is never left with whatever
    # column the scan order hands an ambiguous one.
    rows, columns, weight = set(range(nstates)), set(range(block.shape[1])), 1.0
    while rows:
        i, j = max(((i, j) for i in rows for j in columns),
                   key=lambda p: block[p])
        order[i], sign[i] = j, (1.0 if s[1 + i, 1 + j] >= 0.0 else -1.0)
        weight = min(weight, block[i, j])
        rows.discard(i)
        columns.discard(j)
    return order, sign, float(weight)


def follow_state(s, state):
    """(index, weight, margin) of the displaced root that IS `state`.

    `align_roots` assigns a whole manifold; this follows ONE state, which is
    what a geometry optimization needs. The displaced root of largest |S| with
    the reference state is the same state, whatever energy order it came back
    in.

    weight: |S| with the chosen root. Near 1 the identification is certain;
            far below it the reference state has no counterpart in the
            displaced manifold -- it left the solved window, or `nroots` is
            too small to contain it.
    margin: |S| of the best minus the runner-up. SMALL MARGIN IS THE DANGEROUS
            CASE, and it is not the same as small weight: two roots that have
            mixed share the reference character between them, so both overlaps
            are moderate and neither is the state. Following the larger one
            then picks a state by a coin toss the size of the margin.
    """
    row = np.abs(np.asarray(s, float)[1 + state, 1:])
    if row.size == 0:
        raise ValueError('no displaced roots to follow into')
    order = np.argsort(row)[::-1]
    best = int(order[0])
    runner = float(row[order[1]]) if row.size > 1 else 0.0
    return best, float(row[best]), float(row[best] - runner)


def aligned_overlap(s, nstates):
    """`s` with its displaced columns permuted and signed onto the reference.

    Linear in the displaced vectors, so signing the columns of the overlap is
    the same as signing the vectors and rebuilding it.
    """
    order, sign, weight = align_roots(s, nstates)
    out = np.zeros((nstates + 1, nstates + 1))
    out[:, 0] = s[:nstates + 1, 0]
    out[:, 1:] = s[:nstates + 1, 1 + order] * sign[None, :]
    return out, order, weight


def nabla_mo(mol, mo):
    """<phi_p| d/dr |phi_q> in the MO basis, (3, nmo, nmo) and antisymmetric.

    `int1e_ipovlp` is <grad chi_mu|chi_nu>; integrating by parts turns it into
    the operator wanted here with a sign. The operator is antisymmetric because
    the orbitals are orthonormal at every geometry, which is exactly why a
    derivative coupling takes the X - Y combination of a skew-symmetric
    operator rather than the X + Y one of a symmetric operator.
    """
    nabla_ao = -np.asarray(mol.intor('int1e_ipovlp'), float)
    mo = np.asarray(mo, float)
    return np.einsum('xmn,mp,nq->xpq', nabla_ao, mo, mo, optimize=True)


def translational_sum(mol, mo, nocc, x, y, states=None):
    """(1 + n, 1 + n, 3) value of sum_A d_IJ^A, from one-electron integrals.

    The rigid nuclear translation of the module docstring: the orbitals ride
    with their centres, so d T_pq / d(translation) = -<phi_p|grad|phi_q> and
    the sum over atoms of the coupling is minus the velocity-gauge transition
    moment. NOT zero, and not the ETF-corrected sum rule.
    """
    nab = nabla_mo(mol, mo)
    x, y = check_normalization(x, y)
    states = range(x.shape[1]) if states is None else states
    states = tuple(int(n) for n in states)
    nvir = nab.shape[1] - nocc
    xs = x.reshape(nocc, nvir, -1)[:, :, states]
    ys = y.reshape(nocc, nvir, -1)[:, :, states]
    n_oo, n_vv = nab[:, :nocc, :nocc], nab[:, nocc:, nocc:]
    n_ov = nab[:, :nocc, nocc:]

    n = len(states)
    out = np.zeros((n + 1, n + 1, 3))
    ground = -GROUND_SPIN_FACTOR * np.einsum('iaJ,xia->Jx', xs - ys, n_ov,
                                             optimize=True)
    out[0, 1:] = ground
    out[1:, 0] = -ground
    for u, v, s in ((xs, xs, 1.0), (ys, ys, -1.0)):
        c = EXCITED_SPIN_FACTOR * s
        out[1:, 1:] -= c * np.einsum('iaI,jaJ,xij->IJx', u, v, n_oo,
                                     optimize=True)
        out[1:, 1:] -= c * np.einsum('iaI,ibJ,xab->IJx', u, v, n_vv,
                                     optimize=True)
    return out


def spectrum_solve(source):
    """`solve(mol) -> (mf, omega, X, Y)` for a chain, composed surface or manifold.

    Reaching for `_forward` is the deliberate cost of leaving the BSE and GW
    code untouched, the same trade `spin_orbit.chain_manifolds` makes: the
    forward pass returns the whole Casida spectrum and its eigenvectors, and
    nothing in it depends on which root is selected.

    The mean field comes from the object's own accessor, never from
    `scf_factory`, so a surface standing in an environment is displaced inside
    it. The frozen conventions are NOT rebuilt at a displaced geometry -- the
    factorization, the quasiparticle window and the frames stay the reference
    ones -- for the same reason a finite-difference gradient does not refreeze:
    a convention that moves with the displacement is a discontinuity in the
    quantity being differenced.
    """
    owner = getattr(source, 'chain', source)
    chain = driven_chain(owner)
    holder = owner if hasattr(owner, 'mean_field') else chain

    def solve(mol):
        mol, mf = holder.mean_field(mol)
        omega, pieces = chain._forward(mol, mf)
        return mf, omega, pieces[10], pieces[11]

    return solve


def displaced(mol, coords, atom, axis, step):
    """`mol` with one Cartesian component moved by `step` Bohr."""
    shift = np.zeros((mol.natm, 3))
    shift[atom, axis] = step
    out = mol.copy()
    out.set_geom_(coords + shift, unit='Bohr')
    out.build(False, False)
    return out


def derivative_couplings(source, mol=None, states=(0, 1), step=NUCLEAR_FD_STEP,
                         map_fn=None):
    """`DerivativeCouplings` over the ground state and the BSE roots `states`.

    `source` is an `ExcitedStateChain`, a composed surface or a `StateManifold`
    -- anything `spectrum_solve` can drive -- or a callable
    `solve(mol) -> (mf, omega, X, Y)` for a route that is not one of those.

    The bra sits at the reference geometry for every displacement, so
    S(R0, R0) is the identity exactly and the central difference

        d_IJ = [ S_IJ(R0, R0 + h) - S_IJ(R0, R0 - h) ] / 2h

    is a derivative of the identity's neighbourhood. The two-sided form
    [<I(-h)|J(+h)> - <I(+h)|J(-h)>]/4h of Hammes-Schiffer and Tully
    (J. Chem. Phys. 101, 4657 (1994)) is antisymmetric in I and J BY
    CONSTRUCTION and so cannot be used to test itself; this one is
    antisymmetric only because the displaced states really are orthonormal,
    which makes d_IJ + d_JI a measurement rather than a tautology.

    Costs 6 natm + 1 solves, and every state pair comes off the same ones.
    """
    solve = source if callable(source) else spectrum_solve(source)
    states = tuple(sorted({int(n) for n in states}))
    mf0, omega0, x0, y0 = solve(mol)
    mol = mf0.mol if mol is None else mol
    nocc = int(np.count_nonzero(np.asarray(mf0.mo_occ) > 0))
    if states[-1] >= x0.shape[1]:
        raise ValueError(f'root {states[-1]} was asked for but the solve '
                         f'returned {x0.shape[1]} roots')
    keep = list(states)
    x_ref, y_ref = x0[:, keep], y0[:, keep]
    nstates = len(keep)

    crd = np.asarray(mol.atom_coords())
    d = np.zeros((nstates + 1, nstates + 1, mol.natm, 3))
    d_omega = np.zeros((nstates, mol.natm, 3))
    weight, swaps = 1.0, 0

    def displaced_side(job):
        """One displaced solve, aligned against the FIXED reference: nothing
        carries from one displacement to the next, so the 6 natm of them are
        independent and `map_fn` may spread them over a process pool."""
        atom, axis, sign = job
        mfd, om, xd, yd = solve(displaced(mol, crd, atom, axis, sign * step))
        t = mo_overlap(mol, mf0.mo_coeff, mfd.mol, mfd.mo_coeff)
        s = state_overlap(t, nocc, x_ref, y_ref, xd, yd)
        s, order, w = aligned_overlap(s, nstates)
        # The TRACKED root's energy, not the k-th in energy order: a
        # swapped pair differenced by index is a gradient of neither.
        return s, np.asarray(om, float)[order], w, order

    jobs = [(atom, axis, sign) for atom in range(mol.natm) for axis in range(3)
            for sign in (-1.0, 1.0)]
    runner = map if map_fn is None else map_fn
    side = dict(zip(jobs, runner(displaced_side, jobs)))
    for atom in range(mol.natm):
        for axis in range(3):
            sides, energies = [], []
            for sign in (-1.0, 1.0):
                s, om_tracked, w, order = side[(atom, axis, sign)]
                weight = min(weight, w)
                swaps += int(np.any(order != np.arange(nstates)))
                sides.append(s)
                energies.append(om_tracked)
            d[:, :, atom, axis] = (sides[1] - sides[0]) / (2.0 * step)
            d_omega[:, atom, axis] = (energies[1] - energies[0]) / (2.0 * step)

    t_ref = mo_overlap(mol, mf0.mo_coeff, mol, mf0.mo_coeff)
    s_ref = state_overlap(t_ref, nocc, x_ref, y_ref, x_ref, y_ref)
    diagnostics = {
        'omega': np.asarray(omega0, float)[keep],
        # dOmega/dR off the SAME displaced solves: the analytic excitation
        # gradient is gated against this, so one number certifies the
        # displacement, the mean field and the root tracking the coupling used.
        'omega_gradient': d_omega,
        'identity_residual': float(np.abs(s_ref - np.eye(nstates + 1)).max()),
        'assignment_weight': weight,
        'root_swaps': swaps,
        'fd_step': step,
        'nocc': nocc}
    return DerivativeCouplings(d, states, step, diagnostics)
