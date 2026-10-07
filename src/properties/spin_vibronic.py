"""Second-order spin-vibronic coupling: the nuclear derivative of <S|H_SO|T>
from the analytic derivative couplings, with no displaced solve.

THE ELEMENT AND ITS DERIVATIVE. With V^eta(Q) = <S1(Q)|H_SO^eta|T1(Q)> the
spin-factored real three-vector of `spin_orbit` (|V|^2 = sum_eta V_eta^2), real
states and the resolution of identity inserted on each side,

    dV^eta/dQ_k = sum_{n != 1} d^T_{n1,k} V^eta(S1, T_n)            (1)
                + sum_{m != 1} d^S_{m1,k} V^eta(S_m, T1)
                + D^eta_k

with d_{n1,k} = <n|d/dQ_k 1> = <n|dH/dQ_k|1> / (E_1 - E_n). The first sum is
the second-order spin-vibronic term of Penfold, Gindensperger, Daniel and
Marian, Chem. Rev. 118, 6975 (2018),

    V_SO(S1, T_n) V_vib(T_n, T1) / (E_T1 - E_Tn),

spin-orbit coupling into a higher triplet that the promoting mode mixes back
into T1 (Etherington, Gibson, Monkman and Penfold, Nat. Commun. 7, 13680
(2016); Gibson and Penfold, Phys. Chem. Chem. Phys. 19, 8428 (2017)). The
second sum is the same path through the excited singlets. D holds the rest:
the operator's own geometry dependence and the occupied-virtual orbital
rotation, which reaches S1 x T1 through the ground state and through the
doubly excited configurations.

THE GROUND STATE IS NOT A PATH ON ITS OWN. <S0|d/dQ S1> <S0|H_SO|T1> is the S0
half of that rotation; its doubles half, which a singles manifold cannot hold,
cancels most of it. On twisted formaldehyde (BSE@G0W0/cc-pVDZ) the S0 term
alone is ten times the element's finite difference, so it stays in D.

TRUNCATION. The usual form keeps T2 and S2. That is the leading term when T2
is close to T1, because d_21 grows as 1/(E_T2 - E_T1) while the rest is
bounded: on formaldehyde with T2 85 meV above T1 the T2 + S2 truncation
matches the element's finite difference to 1.3%, at 0.53 eV to 22% with an
absolute residual of the same size. A pair S_n/T_n of one orbital character
enters with two large opposite paths, so truncate both manifolds at matching
states.

THE RATE. With dimensionless q_k = sqrt(omega_k) Q_k, `rates.spin_vibronic_rate`
takes |dV/dq_k| = |dV/dQ_k| / sqrt(omega_k) as the Herzberg-Teller coupling,
added incoherently to the Condon channel.

WHY NOT THE NORM OVER A BLOCK. Differencing sum_{i,j in block} |V(S_i, T_j)|^2
over displaced geometries is invariant under any rotation among the block's
triplets, and the T1-T2 mixing a promoting mode causes is such a rotation: the
block norm cancels the term that carries reverse intersystem crossing in a
multiresonance emitter. A single element differenced with root following keeps
it; (1) is that element's derivative.

SIGNS. Every d_{n1} V(S1, T_n) product is invariant under the sign of T_n, and
the whole sum flips with T1 as V(S1, T1) does, so (1) is consistent provided
the couplings and the elements come from the same Casida vectors: one
`StateManifold.evaluate`, never two solves.

One evaluation serves every mode: the shared forward pass, one reverse pass
and one Z-vector per coupling pair (T1-T2, S1-S2).
"""
import numpy as np

from src.gradients.derivative_coupling import (configuration_coupling,
                                               state_to_state_density)
from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR,
                                       GROUND_SPIN_FACTOR,
                                       ground_state_element,
                                       interstate_element)
from src.properties.vibronic import project_coupling
from src.SingleReference.LinearResponse.linear_response import (
    check_normalization)


def soc_vector(h_mo, nocc, singlet, triplet):
    """(3,) spin-factored real <S|H_SO^eta|T>, Hartree; |.| is `spin_orbit.coupling`.

    singlet: (X, Y) of one singlet root, or None for the ground state;
    triplet: (X, Y) of one triplet root.
    """
    xt, yt = (np.asarray(a, float).ravel() for a in triplet)
    if singlet is None:
        return GROUND_SPIN_FACTOR * ground_state_element(h_mo, nocc, xt, yt)
    xs, ys = (np.asarray(a, float).ravel() for a in singlet)
    return EXCITED_SPIN_FACTOR * interstate_element(h_mo, nocc, xs, xt, ys, yt)


def second_order_derivative(v_singlet_paths, v_triplet_paths, d_singlet,
                            d_triplet, direct=None):
    """(natm, 3, 3) dV^eta/dR_{A,x} of one S-T element by Eq. (1), Hartree/Bohr.

    v_triplet_paths: {n: (3,) V(S, T_n)}; d_triplet: {n: (natm, 3)
        d_{n,T} = <T_n|d/dR T>}, the intermediate triplets.
    v_singlet_paths: {m: (3,) V(S_m, T)}; d_singlet: {m: (natm, 3)
        d_{m,S} = <S_m|d/dR S>}, the intermediate excited singlets.
    direct: optional (natm, 3, 3) remainder D of Eq. (1).

    Last axis eta, the spin-orbit component.
    """
    if set(v_triplet_paths) != set(d_triplet):
        raise ValueError(f'triplet paths {sorted(v_triplet_paths)} and '
                         f'couplings {sorted(d_triplet)} differ')
    if set(v_singlet_paths) != set(d_singlet):
        raise ValueError(f'singlet paths {sorted(v_singlet_paths)} and '
                         f'couplings {sorted(d_singlet)} differ')
    terms = [np.asarray(d_triplet[n], float)[..., None]
             * np.asarray(v_triplet_paths[n], float) for n in d_triplet]
    terms += [np.asarray(d_singlet[m], float)[..., None]
              * np.asarray(v_singlet_paths[m], float) for m in d_singlet]
    if direct is not None:
        terms.append(np.asarray(direct, float))
    if not terms:
        raise ValueError('no path and no direct term: nothing to differentiate')
    return np.sum(terms, axis=0)


def mode_derivatives(dv_cart, modes, masses, omega, rotation=None):
    """((nmode, 3) dV^eta/dq_k, (nmode,) |dV/dq_k|) on dimensionless modes.

    dq_k = sqrt(omega_k) dQ_k with Q_k mass-weighted, so the Cartesian
    derivative is projected like a gradient (`vibronic.project_coupling`) and
    divided by sqrt(omega_k); |dV/dq_k| is what `rates.spin_vibronic_rate`
    takes. Imaginary and zero modes come back as zero.

    rotation: the (3, 3) Kabsch matrix of `vibronic.align_to(...,
    return_rotation=True)` when the couplings were computed in another frame
    than the modes'; a per-atom vector v goes to v @ rotation.
    """
    dv = np.asarray(dv_cart, float)
    if rotation is not None:
        dv = np.einsum('axe,xy->aye', dv, np.asarray(rotation, float))
    w = np.asarray(omega, float)
    out = np.zeros((len(w), dv.shape[-1]))
    real = w > 0
    for eta in range(dv.shape[-1]):
        proj = project_coupling(dv[..., eta], modes, masses)
        out[real, eta] = proj[real] / np.sqrt(w[real])
    return out, np.linalg.norm(out, axis=1)


def _root(ev, key):
    """(spin, X, Y, omega) of state `key` of a `StateEvaluation`."""
    spin = ev.target[key][0]
    om, x, y = ev.spectrum[spin]
    r = ev.root[key]
    return spin, x[:, r], y[:, r], float(om[r])


def evaluated_coupling(ev, m, n):
    """(natm, 3) d_mn = <m|d/dR n> between two roots of one `StateEvaluation`.

    The amplitude term is the interstate numerator `evaluate(couplings=...)`
    computed, over Omega_n - Omega_m; the configuration term is one Z-vector
    (`derivative_coupling.configuration_coupling`). The numerator is
    symmetric, so either ordering of the pair in `ev.interstate` serves.
    """
    pair = (m, n) if (m, n) in ev.interstate else (n, m)
    if pair not in ev.interstate:
        raise ValueError(f'no interstate numerator for {(m, n)!r}; pass '
                         f'couplings=[{(m, n)!r}] to evaluate')
    num = ev.interstate[pair][0]
    spin_m, xm, ym, om_m = _root(ev, m)
    spin_n, xn, yn, om_n = _root(ev, n)
    if spin_m != spin_n:
        raise ValueError(f'{m!r} and {n!r} are of different spin')
    nocc = int(np.count_nonzero(np.asarray(ev.mf.mo_occ) > 0))
    goo, gvv = state_to_state_density(nocc, xm, ym, xn, yn)
    return (np.asarray(num) / (om_n - om_m)
            + configuration_coupling(ev.mol, ev.mf, nocc, goo, gvv))


def spin_vibronic_coupling(ev, h_mo, singlet, triplet, singlet_paths=(),
                           triplet_paths=(), modes=None, masses=None,
                           omega=None, rotation=None, direct=None):
    """Condon element and its second-order derivative off ONE evaluation.

        man = StateManifold(chain, states=(S1, S2, T1, T2))
        ev = man.evaluate(couplings=((S2, S1), (T2, T1)))
        h_mo = spin_orbit.soc_operator_mo(ev.mf, ev.mol)
        out = spin_vibronic_coupling(ev, h_mo, S1, T1, (S2,), (T2,),
                                     modes=modes, masses=masses, omega=omega)
        rates.spin_vibronic_rate(dE, out['v0'], out['dv_dq'], omega, rho, T)

    with S1 = ('singlet', 0) etc. `singlet_paths`/`triplet_paths` are the
    intermediate states of Eq. (1) as keys of `ev`; each needs its pair with
    `singlet`/`triplet` in `ev.interstate`. `direct` is the optional remainder
    D, (natm, 3, 3), in the frame of `ev`.

    Returns a dict: 'v0' |V(S, T)| and 'v0_vector', 'dv_cart' (natm, 3, 3),
    'paths' {label: (natm, 3, 3)} per intermediate state, 'couplings'
    {label: (natm, 3)} the d used, 'soc' {label: (3,)} the elements used, and
    with modes, masses and omega given, 'dv_dq' (nmode,) and 'dv_dq_vector'
    (nmode, 3), plus 'dv_dq_paths' {label: (nmode,)}.
    """
    nocc = int(np.count_nonzero(np.asarray(ev.mf.mo_occ) > 0))
    for key, spin in ((singlet, 'singlet'), (triplet, 'triplet')):
        if ev.target[key][0] != spin:
            raise ValueError(f'{key!r} is not a {spin}')
    _, xs, ys, _ = _root(ev, singlet)
    _, xt, yt, _ = _root(ev, triplet)
    check_normalization(np.c_[xs], np.c_[ys])
    check_normalization(np.c_[xt], np.c_[yt])
    v0 = soc_vector(h_mo, nocc, (xs, ys), (xt, yt))
    v_s, d_s, v_t, d_t, labels = {}, {}, {}, {}, {}
    for key in triplet_paths:
        _, xn, yn, _ = _root(ev, key)
        v_t[key] = soc_vector(h_mo, nocc, (xs, ys), (xn, yn))
        d_t[key] = evaluated_coupling(ev, key, triplet)
        labels[key] = f'{key[0]}{key[1] + 1}'
    for key in singlet_paths:
        _, xm, ym, _ = _root(ev, key)
        v_s[key] = soc_vector(h_mo, nocc, (xm, ym), (xt, yt))
        d_s[key] = evaluated_coupling(ev, key, singlet)
        labels[key] = f'{key[0]}{key[1] + 1}'
    paths = {}
    for key in v_t:
        paths[labels[key]] = second_order_derivative({}, {key: v_t[key]}, {},
                                                     {key: d_t[key]})
    for key in v_s:
        paths[labels[key]] = second_order_derivative({key: v_s[key]}, {},
                                                     {key: d_s[key]}, {})
    dv = second_order_derivative(v_s, v_t, d_s, d_t, direct)
    out = {'v0': float(np.linalg.norm(v0)), 'v0_vector': v0, 'dv_cart': dv,
           'paths': paths,
           'couplings': {labels[k]: v for k, v in {**d_s, **d_t}.items()},
           'soc': {labels[k]: v for k, v in {**v_s, **v_t}.items()}}
    if modes is not None:
        out['dv_dq_vector'], out['dv_dq'] = mode_derivatives(
            dv, modes, masses, omega, rotation)
        out['dv_dq_paths'] = {k: mode_derivatives(p, modes, masses, omega,
                                                  rotation)[1]
                              for k, p in paths.items()}
    return out
