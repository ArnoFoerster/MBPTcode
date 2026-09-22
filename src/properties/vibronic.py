"""Vibronic quantities from a potential-energy surface: normal modes,
Huang-Rhys factors, adiabatic gaps and the Marcus reorganization energy.

Everything here expands BOTH surfaces in the GROUND state's normal modes, which
is the displaced-oscillator model the effective-mode treatments in the TADF
literature use. Two independent routes to the Huang-Rhys spectrum are kept on
purpose: one needs a single excited-state gradient at the Franck-Condon point,
the other a full excited-state optimization, and the extent to which they agree
IS the measurement of how good the harmonic, unchanged-curvature approximation
is for that molecule.

The surfaces themselves are only ever touched through `PotentialEnergySurface`,
so the same routines serve a BSE@GW excitation, a dRPA ground state or a
downfolded active-space root.
"""
import numpy as np

from src.Base.constants import AMU_TO_ME, HARTREE_TO_CM, HARTREE_TO_EV
from src.Base.declaration import PhysicsMismatch
from src.Base.isdf_jk import refuse_isdf_jk_gradient
from src.properties.optimize import relax, translation_rotation_basis
from src.properties.surface import surface_mean_field


def normal_modes(mf, mol=None, hess=None, project=True, isotope_avg=True):
    """Ground-state harmonic analysis: (omega, modes, masses, hess).

    `omega` in Hartree (atomic frequency units), `modes` are the columns of the
    MASS-WEIGHTED eigenvectors, (3N, nmode), orthonormal. Imaginary frequencies
    come back NEGATIVE, which is how a saddle point announces itself; nothing
    downstream is meaningful at one.

    The Hessian is pyscf's analytic ground-state one, which is what the
    Huang-Rhys factors are defined against: the standard displaced-oscillator
    model expands BOTH surfaces in the GROUND state's modes.

    isotope_avg: use isotope-AVERAGED atomic masses (default), matching
        `pyscf.hessian.thermo.harmonic_analysis` and every other package's
        default. The alternative is the most abundant isotope, which is lighter
        and raises every frequency -- by up to 11.8 cm^-1 on formaldehyde/
        cc-pVDZ, which is small but entirely systematic and would quietly
        offset any comparison. Pass False only for a specific isotopologue, and
        then use the same masses for the displacement projection.
    """
    mol = mf.mol if mol is None else mol
    if hess is None:
        # pyscf's Hessian differentiates the FITTED interaction TWICE, and on
        # an ISDF mean field it knows nothing of the interpolation -- the same
        # defect the gradient guard exists for, one derivative further. The
        # force constants would be those of a different functional, and the
        # softest modes, which carry the largest Huang-Rhys factors, are the
        # ones that suffer most.
        refuse_isdf_jk_gradient(mf, 'the analytic Hessian')
        hess = mf.Hessian().kernel()
    n = mol.natm
    h = np.asarray(hess).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    h = 0.5 * (h + h.T)
    masses = np.asarray(mol.atom_mass_list(isotope_avg=isotope_avg)) * AMU_TO_ME
    w = np.repeat(masses, 3) ** -0.5
    hm = h * w[:, None] * w[None, :]
    if project:
        tr = translation_rotation_basis(mol.atom_coords(), masses)
        # mass-weight the rigid-body basis, then re-orthonormalize
        tr = tr / w[None, :]
        q, _ = np.linalg.qr(tr.T)
        p = np.eye(3 * n) - q @ q.T
        hm = p @ hm @ p
    ev, vec = np.linalg.eigh(hm)
    keep = np.argsort(-np.abs(ev))[:3 * n - (len(tr) if project else 0)]
    keep = np.sort(keep)
    omega = np.sign(ev[keep]) * np.sqrt(np.abs(ev[keep]))
    order = np.argsort(omega)
    return omega[order], vec[:, keep][:, order], masses, h


def huang_rhys_from_gradient(grad, omega, modes, masses):
    """S_k from the excited-state gradient at the GROUND-state minimum.

    The linear-coupling (vertical) model: expand the excited surface to first
    order at the Franck-Condon point and give it the ground state's curvature.
    In mass-weighted normal coordinates with the gradient projection g_k,

        Delta q_k = -g_k / omega_k^2,   S_k = g_k^2 / (2 omega_k^3)

    ONE gradient and one Hessian, no excited-state optimization -- which is the
    whole reason to have it: it gives spectra and reorganization energies for
    systems too large to relax, and it is the version the effective-mode
    treatments in the TADF literature use.
    """
    w = np.repeat(np.asarray(masses), 3) ** -0.5
    g_mw = np.asarray(grad).ravel() * w              # d/dq = M^-1/2 d/dx
    gk = modes.T @ g_mw
    ok = omega > 0
    s = np.zeros_like(gk)
    s[ok] = gk[ok] ** 2 / (2.0 * omega[ok] ** 3)
    return s, gk


def project_coupling(d_cart, modes, masses):
    """d_k, a Cartesian derivative coupling on the MASS-WEIGHTED normal modes.

    The same projection `huang_rhys_from_gradient` makes of a gradient, because
    a derivative coupling is the same kind of object -- one d/dR per Cartesian
    component -- and d/dq = M^-1/2 d/dx.

    The result is in mass-weighted units, 1/(Bohr sqrt(m_e)); multiplying by
    sqrt(omega_k) makes it the Hartree coupling of a one-quantum channel, which
    is what `rates.internal_conversion_rate` does.
    """
    w = np.repeat(np.asarray(masses), 3) ** -0.5
    return modes.T @ (np.asarray(d_cart, float).ravel() * w)


def displace_along_mode(mol0, modes, masses, omega, k, dq):
    """`mol0` displaced by a dimensionless step `dq` along mode `k`.

    The inverse of the projection `huang_rhys_from_displacement` reads a
    geometry with: x = M^-1/2 L_k Q_k, Q_k = q_k / sqrt(omega_k), so that
    `dq` is in the same dimensionless normal coordinate (<1|q|0> = 1/sqrt 2)
    a Franck-Condon or Herzberg-Teller expansion is written in.
    """
    w = np.repeat(np.asarray(masses), 3) ** -0.5
    dx = (modes[:, k] * w).reshape(-1, 3) * (dq / np.sqrt(omega[k]))
    m = mol0.copy()
    m.set_geom_(mol0.atom_coords() + dx, unit='Bohr')
    m.build(False, False)
    return m


def huang_rhys_from_displacement(mol_gs, mol_ex, omega, modes, masses):
    """S_k from two optimized geometries, projected on the ground-state modes.

    The adiabatic route: S_k = omega_k (Delta q_k)^2 / 2 with Delta q the
    mass-weighted displacement between minima. It costs an excited-state
    optimization that `huang_rhys_from_gradient` does not, and in exchange it
    is not limited to first order in the displacement -- so the two AGREEING is
    evidence that the linear model holds for this state, and their disagreeing
    is a measurement of the anharmonicity/curvature change rather than a bug.

    The two geometries must be in the same orientation; the caller is
    responsible for that (`align_to` does it).
    """
    dx = (np.asarray(mol_ex.atom_coords()) - np.asarray(mol_gs.atom_coords())).ravel()
    dq = dx * np.repeat(np.asarray(masses), 3) ** 0.5   # q = M^1/2 x
    dqk = modes.T @ dq
    ok = omega > 0
    s = np.zeros_like(dqk)
    s[ok] = omega[ok] * dqk[ok] ** 2 / 2.0
    return s, dqk


def align_to(mol_ref, mol_move, return_rotation=False):
    """`mol_move` rigidly superposed on `mol_ref` (mass-weighted Kabsch).

    A displacement projected on normal modes is meaningless until this is done:
    an optimizer is free to translate and rotate, those directions were
    projected out of the modes, and what leaks through shows up as spurious
    displacement along the softest modes -- exactly the ones with the largest
    Huang-Rhys factors.

    return_rotation also gives the (3, 3) matrix, which anything ELSE computed
    at the moved geometry needs: a gradient or a derivative coupling is a
    vector per atom in ITS OWN frame, and projecting it on the reference's
    modes without the same rotation silently mixes the components.
    """
    m = np.asarray(mol_ref.atom_mass_list(isotope_avg=True))
    a = np.asarray(mol_ref.atom_coords())
    b = np.asarray(mol_move.atom_coords())
    ca = (m[:, None] * a).sum(0) / m.sum()
    cb = (m[:, None] * b).sum(0) / m.sum()
    a0, b0 = a - ca, b - cb
    u, _, vt = np.linalg.svd((m[:, None] * b0).T @ a0)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    out = mol_move.copy()
    out.set_geom_(b0 @ rot + ca, unit='Bohr')
    out.build(False, False)
    return (out, rot) if return_rotation else out


def reorganization_from_huang_rhys(s, omega):
    """lambda = sum_k S_k hbar omega_k, in Hartree.

    The sum rule that ties the two halves together. Computed independently, a
    Huang-Rhys spectrum and a relaxation energy must satisfy it, so it is a
    gate and not a definition -- `vibronic_analysis` checks it.
    """
    ok = omega > 0
    return float((np.asarray(s)[ok] * omega[ok]).sum())


def relax_state(surface, mol=None, engine='auto', **kw):
    """Relax one state and return everything the vibronic analysis needs.

    The record is built on the surface's OWN mean field: the optimizer stepped
    on that one, and `scf_factory`'s carries no environment, so a solvated
    record would report the bare energy of a geometry relaxed in the continuum.
    """
    mol_opt, info = relax(surface, mol, engine=engine, **kw)
    mf = surface_mean_field(surface, mol_opt)
    record = {'mol': mol_opt, 'mf': mf,
              'e_total': surface.total_energy(mol_opt, mf),
              'e_scf': float(mf.e_tot), 'info': info}
    # An excited surface carries its excitation energy alongside the total; a
    # ground-state one has none, and reporting a zero there would read as a
    # degeneracy.
    if hasattr(surface, 'excitation'):
        record['omega'] = surface.excitation(mol_opt, mf)
    # WHAT WAS RELAXED, carried with the number. A surface built through
    # `potential_energy_surface` knows which functional its E_0 is and which
    # realization computed it; one built by hand does not, and None is how the
    # record says so rather than implying a default nobody declared.
    record['physics'] = getattr(surface, 'physics', None)
    record['realization'] = getattr(surface, 'realization', None)
    return record


def energy_at(surface, mol):
    """A state's TOTAL energy at someone else's geometry -- the off-diagonal
    entries of a four-point reorganization scheme.

    On the surface's own mean field, since a Marcus lambda is a difference of
    two of these and an environment that entered one and not the other would
    show up as reorganization.
    """
    return surface.total_energy(mol, surface_mean_field(surface, mol))


def adiabatic_gap(state_a, state_b):
    """E_a(R_a) - E_b(R_b) in Hartree: the ADIABATIC gap between two relaxed
    states, each at its own minimum.

    For Delta-E_ST pass (singlet, triplet). This is the quantity a TADF rate
    depends on exponentially, and it is NOT the vertical gap: the two states
    relax by different amounts, so the ground-state energy does not cancel and
    both surfaces have to be carried in full (which is why a surface reports
    E_0 + Omega rather than Omega).

    REFUSES two records whose declared physics cannot be differenced: E_0 is
    carried in full by both, so a mean-field E_0 on one side and
    E_HF + E_c^dRPA on the other is 6.3 eV of correlation energy on water that
    only one side has. Two records that carry no declaration -- a hand-built
    surface, a toy -- are differenced as they always were: this module records
    and refuses nothing it cannot check.
    """
    physics_a, physics_b = state_a.get('physics'), state_b.get('physics')
    if (physics_a is not None and physics_b is not None
            and not physics_a.comparable_with(physics_b)):
        raise PhysicsMismatch(physics_a, physics_b)
    return state_a['e_total'] - state_b['e_total']


def reorganization(surface_target, state_from, state_target):
    """Marcus lambda for a transition INTO `surface_target`'s state, in Hartree.

        lambda = E_target(R_from) - E_target(R_target)

    the energy the accepting state sheds relaxing from the donor's geometry to
    its own. `state_from` and `state_target` are `relax_state` records; the
    geometries must already be relaxed, and the target surface must be the one
    that produced `state_target`.

    The four-point total, used when both surfaces reorganize, is this plus its
    mirror with donor and acceptor exchanged; the literature is not consistent
    about which of the two, or their average, is called lambda, so this returns
    the ONE-WAY quantity and leaves the convention to the caller.
    """
    return energy_at(surface_target, state_from['mol']) - state_target['e_total']


def vibronic_analysis(surface, state, mol_gs, mf_gs, hess=None, top=8,
                      verbose=True):
    """Huang-Rhys factors by BOTH routes, plus the sum rule that ties them.

    Returns a dict with `s_gradient` (linear coupling at the ground-state
    minimum), `s_displacement` (from the relaxed excited geometry),
    `lambda_gradient` / `lambda_displacement` (their sum rules) and
    `lambda_relaxation` (the energy actually released relaxing the excited
    state, computed with no reference to modes at all).

    THE GATE IS `lambda_relaxation` AGAINST `lambda_gradient`. Both are energies
    in Hartree and they are computed by completely different routes -- one is a
    difference of two total energies, the other a sum over sum_k S_k hbar
    omega_k built from a gradient and a Hessian. They agree only to the extent
    that the excited surface is harmonic with the ground state's curvature,
    which is exactly the approximation the effective-mode TADF treatments make.
    Their ratio is therefore not a bug report; it is the measurement of how good
    that approximation is for this molecule.
    """
    omega, modes, masses, hess = normal_modes(mf_gs, mol_gs, hess=hess)
    g_fc, _, _ = surface.total_gradient(mol_gs, mf_gs)
    s_grad, gk = huang_rhys_from_gradient(g_fc, omega, modes, masses)
    aligned = align_to(mol_gs, state['mol'])
    s_disp, dqk = huang_rhys_from_displacement(mol_gs, aligned, omega, modes,
                                               masses)
    lam_g = reorganization_from_huang_rhys(s_grad, omega)
    lam_d = reorganization_from_huang_rhys(s_disp, omega)
    e_vert = surface.total_energy(mol_gs, mf_gs)
    lam_relax = e_vert - state['e_total']

    out = {'omega_cm': omega * HARTREE_TO_CM, 's_gradient': s_grad,
           's_displacement': s_disp, 'gk': gk, 'dqk': dqk,
           'lambda_gradient': lam_g, 'lambda_displacement': lam_d,
           'lambda_relaxation': float(lam_relax),
           'effective_mode_cm': float((s_grad * omega).sum()
                                      / max(s_grad.sum(), 1e-30)
                                      * HARTREE_TO_CM),
           'total_S': float(s_grad.sum())}
    if verbose:
        print(f'  lambda: relaxation {lam_relax * HARTREE_TO_EV:.4f} eV | '
              f'sum-rule(gradient) {lam_g * HARTREE_TO_EV:.4f} eV | '
              f'sum-rule(displacement) {lam_d * HARTREE_TO_EV:.4f} eV')
        print(f'  total S = {out["total_S"]:.3f}, effective mode = '
              f'{out["effective_mode_cm"]:.0f} cm-1')
        idx = np.argsort(-s_grad)[:top]
        print(f'  {"mode/cm-1":>10s} {"S (grad)":>10s} {"S (disp)":>10s}')
        for i in idx:
            print(f'  {omega[i] * HARTREE_TO_CM:10.1f} {s_grad[i]:10.4f} '
                  f'{s_disp[i]:10.4f}')
    return out
