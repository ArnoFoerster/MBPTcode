"""The frozen conventions of one production chain, and the file forms the two
processes exchange.

numpy only: the production side (pyscf) writes it, the torch side reads it, and
the two never share a process -- torch's x86 Intel OpenMP runtime and the rest
of the stack deadlock at interpreter exit when they do.
"""
import numpy as np


class FrozenConventions:
    """Every discrete choice the surface is differentiated on, read off a
    production chain after its first solve."""

    def __init__(self, nocc, mu, states, xc_const, roots, residues, gw_tau,
                 gw_cosft, nu, wt, w_tau, w_cosft, spin_kappa, bse_tda, state,
                 outside_window, scissor, z_prod=None):
        self.nocc, self.mu = int(nocc), float(mu)
        self.states = [int(p) for p in states]
        self.xc_const = np.asarray(xc_const, float)          # <Sigma_x - v_xc>_pp, (norb,)
        self.roots = {int(p): float(w) for p, w in roots.items()}
        self.residues = {int(p): [(int(q), float(w)) for q, w in r]
                         for p, r in residues.items()}
        self.gw_tau, self.gw_cosft = np.asarray(gw_tau, float), np.asarray(gw_cosft, float)
        self.nu, self.wt = np.asarray(nu, float), np.asarray(wt, float)
        self.w_tau, self.w_cosft = np.asarray(w_tau, float), np.asarray(w_cosft, float)
        self.kappa = float(spin_kappa)
        self.bse_tda, self.state = bool(bse_tda), int(state)
        self.outside_window = np.asarray(outside_window, bool)
        # the frozen scissor each orbital outside the window carries, (norb,)
        self.scissor = np.asarray(scissor, float)
        self.z_prod = {} if z_prod is None else {int(p): float(z) for p, z in z_prod.items()}


_KEYS = {'x_mo', 'd', 'd_bare', 'eps', 'nocc', 'mu', 'states', 'xc_const',
         'root_orbs', 'root_vals', 'res_orbs', 'res_len', 'res_flat', 'gw_tau',
         'gw_cosft', 'nu', 'wt', 'w_tau', 'w_cosft', 'kappa', 'bse_tda', 'state',
         'outside_window', 'scissor', 'z_orbs', 'z_vals'}


def frozen_conventions_of(chain, mol, mf, target, qp_orb):
    """(`FrozenConventions`, `SharedForward`, `CasidaSolution`) of a production
    chain at (mol, mf), read AFTER its first solve through its public stages
    (`forward`, `xc_correction`, `qp_set_solve`, `outside_window`): the set,
    with `qp_orb` added for a quasiparticle target, its roots, pole strengths
    and residues, the static terms, the grids and the frozen scissor.

    Production imports are local: this module is the torch side's too, which
    loads numpy only. An orbital that took a residue route other than the
    explicit backend is refused (`SystemExit`): the oracle ports that one.
    """
    from src.Base.numerics.constants import KAPPA
    from src.SingleReference.GW.qp_states import frozen_scissor
    from src.SingleReference.GW.real_screening import residue_set
    shared, solution = chain.forward(mol, mf)
    nocc, norb = chain.nocc, len(mf.mo_energy)
    eps, shift = shared.eps, shared.shift
    d_sigma = shared.d if shared.d_bare is None else shared.d_bare
    states = [int(p) for p in chain.resolved.qp_set]
    if target == 'qp' and qp_orb not in states:
        states = sorted(set(states) | {qp_orb})
    xc_set = chain.xc_correction(mf, states, shift)
    (ws, _, _, _), ro = chain.quasiparticles.qp_set_solve(shared.x_mo, d_sigma, eps, shared.mu,
                                           xc_set, np.zeros(len(states)),
                                           states=states)
    roots = {int(p): float(w) for p, w in zip(states, ws)}
    z_prod = {int(p): float(z) for p, z in zip(states, ro['z'])}
    residues = {p: residue_set(eps, nocc, roots[p]) for p in roots}
    for p in roots:
        route = ro['routes'].get(p)
        if route not in ('explicit', 'none') and residues[p]:
            raise SystemExit(f'orbital {p} took the {route!r} residue route; the '
                             f'oracle ports the explicit backend only')
    shift_vec = np.zeros(norb) if shift is None else np.asarray(shift, float)
    xc_all = np.asarray(chain.xc_correction(mf, np.arange(norb), shift_vec),
                        float)
    conv = chain.conventions
    fz = FrozenConventions(
        nocc=nocc, mu=shared.mu, states=states, xc_const=xc_all, roots=roots,
        residues=residues, gw_tau=conv.gw_grid.tau_points,
        gw_cosft=conv.gw_grid.cosft_wt, nu=conv.nu, wt=conv.wt,
        w_tau=chain.w_grid.tau_points, w_cosft=chain.w_grid.cosft_wt,
        spin_kappa=KAPPA[chain.spin], bse_tda=chain.bse_tda, state=chain.state,
        outside_window=chain.quasiparticles.outside_window(norb),
        scissor=[frozen_scissor(chain.resolved.outside_shift, p) or 0.0
                 for p in range(norb)], z_prod=z_prod)
    return fz, shared, solution

def dump(path, x_mo, d, d_bare, eps, fz, **extra):
    """The factors, the conventions, and whatever else the caller wants carried
    (production adjoints, say). None values travel as empty arrays."""
    res_p = sorted(fz.residues)
    np.savez(path, x_mo=x_mo, d=d, d_bare=(np.zeros(0) if d_bare is None else d_bare),
             eps=eps, nocc=fz.nocc, mu=fz.mu, states=np.asarray(fz.states, int),
             xc_const=fz.xc_const,
             root_orbs=np.asarray(sorted(fz.roots), int),
             root_vals=np.asarray([fz.roots[p] for p in sorted(fz.roots)], float),
             res_orbs=np.asarray(res_p, int),
             res_len=np.asarray([len(fz.residues[p]) for p in res_p], int),
             res_flat=np.asarray([[q, w] for p in res_p for q, w in fz.residues[p]],
                                 float).reshape(-1, 2),
             gw_tau=fz.gw_tau, gw_cosft=fz.gw_cosft, nu=fz.nu, wt=fz.wt,
             w_tau=fz.w_tau, w_cosft=fz.w_cosft, kappa=fz.kappa,
             bse_tda=fz.bse_tda, state=fz.state, outside_window=fz.outside_window,
             scissor=fz.scissor,
             z_orbs=np.asarray(sorted(fz.z_prod), int),
             z_vals=np.asarray([fz.z_prod[p] for p in sorted(fz.z_prod)], float),
             **{k: (np.zeros(0) if v is None else np.asarray(v)) for k, v in extra.items()})


def load(path):
    """(x_mo, d, d_bare or None, eps, FrozenConventions, extras)."""
    z = np.load(path, allow_pickle=False)
    res, k = {}, 0
    for p, n in zip(z['res_orbs'], z['res_len']):
        res[int(p)] = [(int(q), float(w)) for q, w in z['res_flat'][k:k + n]]
        k += n
    fz = FrozenConventions(
        nocc=int(z['nocc']), mu=float(z['mu']), states=z['states'],
        xc_const=z['xc_const'],
        roots=dict(zip(z['root_orbs'].tolist(), z['root_vals'].tolist())),
        residues=res, gw_tau=z['gw_tau'], gw_cosft=z['gw_cosft'], nu=z['nu'],
        wt=z['wt'], w_tau=z['w_tau'], w_cosft=z['w_cosft'],
        spin_kappa=float(z['kappa']), bse_tda=bool(z['bse_tda']),
        state=int(z['state']), outside_window=z['outside_window'],
        scissor=z['scissor'],
        z_prod=dict(zip(z['z_orbs'].tolist(), z['z_vals'].tolist())))
    extra = {k: (None if z[k].size == 0 else z[k]) for k in z.files if k not in _KEYS}
    d_bare = None if z['d_bare'].size == 0 else z['d_bare']
    return z['x_mo'], z['d'], d_bare, z['eps'], fz, extra


def shell_table_arrays(atom_of, ls, exps, coeffs, c2s, prefix):
    """One basis as npz arrays: per shell the atom, l, exponents, contraction
    coefficients (nprim, nctr) and the Cartesian->spherical matrix."""
    out = {f'{prefix}_atom_of': np.asarray(atom_of, int), f'{prefix}_ls': np.asarray(ls, int),
           f'{prefix}_nsh': len(ls)}
    for i in range(len(ls)):
        out[f'{prefix}_exp{i}'] = np.asarray(exps[i], float)
        out[f'{prefix}_coef{i}'] = np.asarray(coeffs[i], float)
        out[f'{prefix}_c2s{i}'] = np.asarray(c2s[i], float)
    return out


def shell_table_from_npz(z, prefix):
    """(atom_of, ls, exps, coeffs, c2s) back from `shell_table_arrays`."""
    n = int(z[f'{prefix}_nsh'])
    return (z[f'{prefix}_atom_of'], z[f'{prefix}_ls'],
            [z[f'{prefix}_exp{i}'] for i in range(n)],
            [z[f'{prefix}_coef{i}'] for i in range(n)],
            [z[f'{prefix}_c2s{i}'] for i in range(n)])
