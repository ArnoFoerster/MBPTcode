"""THC-RPA correlation energy with k-point sampling.

Consumes the factorization from `pbc_isdf.build_isdf_kpts` and produces the
same number as `pbc_rpa.ri_rpa_ecorr`, by the same formula, with the RI-V
auxiliary basis replaced by the THC one:

    E_c = (1/Nq) sum_q (1/2pi) int dw [ logdet(1 - V^q Pi^q(iw)) + Tr(V^q Pi^q(iw)) ]

WHERE THE CONJUGATIONS GO, since this is the step that silently produces a
plausible wrong number. Writing B^{q,k}_{mu,ia} = X^{k*}_{i mu} X^{k+q}_{a mu}
for the THC factor of an occupied-virtual pair density (i occupied at k, a
virtual at k+q -- pbc_rpa's convention, kept deliberately), the ring
interaction is

    v_{ia,jb} = (ia|bj) = sum_{mu nu} B_{mu,ia} V^q_{mu nu} conj(B_{nu,jb})

-- note the SECOND pair is (bj), not (jb): both pairs in a ring carry transfer
+q, but an ERI needs its two pair densities to carry OPPOSITE momenta, and
(bj) is what reconciles those. Then Tr ln(1 - chi0 v) = Tr ln(1 - V^q Pi^q)
with

    Pi^q_{nu mu}(iw) = (2/Nk) sum_k sum_{ia} conj(B_{nu,ia}) chi_ia(w) B_{mu,ia}

so the conjugate sits on the FIRST index. Pi is Hermitian either way, which is
exactly why getting it backwards would not announce itself.

Three routes to the same Pi, each validating the next:

  * `polarizability_q_frequency` -- the definition, O(Nk N_mu^2 n_o n_v) per
    (q, w). No factorization exploited, no time grid, nothing to converge. It
    exists to be right, not to be fast, and it is the reference the other two
    are held to.
  * `polarizability_tau` -- the cubic route. In imaginary time the ov sum
    factorizes into an occupied and a virtual Green's function at the
    interpolation points and Pi is their elementwise product,
    Pi^q(tau) = sum_k Go^k(tau) * Gv^{k+q}(tau).
  * `polarizability_tau_all_q_fft` -- the same, with the k-sum done as an FFT
    over the mesh, O(Nk ln Nk) instead of O(Nk^2) for the whole q-set. This is
    the only genuinely new algorithmic step on the consumer side.

GAPPED SYSTEMS ONLY. Every route here splits orbitals with a single integer
`nocc` applied at every k-point. `require_integer_occupation` refuses anything
else rather than returning a plausible wrong number -- see its docstring for
why no pointwise test can catch the metallic case.
"""
import hashlib
import json
import os

import numpy as np

from src.Base.separable_ri import write_json_atomic
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.matsubara import beta_from_mf, thermal_e_min
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.Periodic.pbc_isdf import (isdf_prepare, isdf_vq,
                                                   resolve_grid,
                                                   resolve_npoints)
from src.SingleReference.Periodic.pbc_occupations import is_integer_occupation


def require_integer_occupation(mo_occ, who='THC-RPA'):
    """Refuse a partially occupied or ragged k-point occupation.

    EVERYTHING in this module splits orbitals with ONE integer `nocc` applied
    at every k-point -- `X[k][:, :nocc]` / `X[k][:, nocc:]` in the pair-density
    factor, and the same split in `green_functions_tau`. For a gapped
    insulator that is exact. For a metal it is simply wrong, and, worse, wrong
    without complaining: measured per-k occupied counts are [8, 9, 11] for a
    Li monolayer (gth-dzvp, 3x3, sigma = 0.02) and
    [3, 4, 4, 6, 4, 6, 6, 8] for bulk BCC Li at 2x2x2, so a single count taken
    from k-point 0 is wrong at every k but one.

    The correct object weights each transition by (f_i - f_a) over a RAGGED,
    per-k pair list rather than slicing a rectangle -- `pbc_occupations`
    has that machinery, and folding sqrt(f_i - f_a) into the pair factor
    leaves Pi's formula, its conjugation convention and the k -> R FFT
    identity untouched. That is the occupation-weighted section further down
    (`polarizability_q_frequency_occ` and its tau counterpart); the integer
    routes here do not do it, and this guard makes calling them on a metal
    loud instead of silent.

    Why a guard and not a warning: the failure is a CONTINUITY failure across
    a parameter, not a pointwise one. Every pointwise check -- reduction to
    the integer path, a limit at fixed geometry, a consistency relation --
    passes while the answer is wrong, because at ANY single geometry the
    integer split is self-consistent. It shows up only in a physical curve
    (an equation of state jumping when a per-k count steps). Nothing in this
    module's test suite would catch it.
    """
    f = np.asarray(mo_occ, dtype=float)
    if not is_integer_occupation(f):
        raise NotImplementedError(
            f"{who}: fractional occupations detected (a smeared / metallic "
            f"mean field). This module slices occ/virt with a single integer "
            f"nocc at every k-point, which is wrong for a metal and silently "
            f"so. Use the occupation-weighted response "
            f"(src/SingleReference/Periodic/pbc_occupations.py) instead, or "
            f"pass an unsmeared gapped mean field.")
    counts = (f > 1e-8).sum(axis=1)
    if len(np.unique(counts)) > 1:
        raise NotImplementedError(
            f"{who}: the occupied count varies across k-points "
            f"({counts.tolist()}). Occupations are integer, but the band "
            f"filling is not k-independent, so one nocc cannot describe every "
            f"k-point. Same remedy as above.")
    return int(counts[0])


def _energies(mo_energy, nocc):
    e = np.asarray(mo_energy)
    return e[:, :nocc], e[:, nocc:]


def chemical_potential(mo_energy, nocc):
    """Midgap, used only to keep the imaginary-time exponentials bounded."""
    e_o, e_v = _energies(mo_energy, nocc)
    return 0.5 * (e_o.max() + e_v.min())


def polarizability_q_frequency(X, mo_energy, nocc, kplus, q, freqs):
    """Pi^q(i.omega) by the definition, shape (nfreq, npts, npts).

    kplus[q, k] = index of k + q (pbc_integrals.get_momentum_transfer_map).
    """
    nkpts = len(X)
    npts = X[0].shape[0]
    e = np.asarray(mo_energy)
    Pi = np.zeros((len(freqs), npts, npts), dtype=np.complex128)
    for k in range(nkpts):
        kq = kplus[q, k]
        B = np.einsum('mi,ma->mia', X[k][:, :nocc].conj(), X[kq][:, nocc:],
                      optimize=True).reshape(npts, -1)
        eia = (e[k][:nocc, None] - e[kq][None, nocc:]).ravel()      # < 0
        for w, omega in enumerate(freqs):
            chi = 2.0 * eia / (omega ** 2 + eia ** 2)
            Pi[w] += (2.0 / nkpts) * (B.conj() * chi) @ B.T
    return Pi


def green_functions_tau(X, mo_energy, nocc, tau, mu):
    """(Go, Gv) at one imaginary time, each (nkpts, npts, npts).

        Go^k_{nu mu} = sum_{i occ}  X^k_{i nu}  X^{k*}_{i mu} e^{+(eps_i - mu) tau}
        Gv^k_{nu mu} = sum_{a virt} X^{k*}_{a nu} X^k_{a mu}  e^{-(eps_a - mu) tau}

    Both exponents are negative for tau > 0, which is the only reason mu is
    here.
    """
    e = np.asarray(mo_energy)
    Go, Gv = [], []
    for k in range(len(X)):
        Xo, Xv = X[k][:, :nocc], X[k][:, nocc:]
        wo = np.exp((e[k][:nocc] - mu) * tau)
        wv = np.exp(-(e[k][nocc:] - mu) * tau)
        Go.append((Xo * wo) @ Xo.conj().T)
        Gv.append((Xv.conj() * wv) @ Xv.T)
    return np.asarray(Go), np.asarray(Gv)


def polarizability_tau(Go, Gv, kplus, q):
    """Pi^q(i.tau) = sum_k Go^k * Gv^{k+q}, elementwise. Direct O(Nk) k-sum."""
    return sum(Go[k] * Gv[kplus[q, k]] for k in range(len(Go)))


def polarizability_tau_all_q_fft(Go, Gv, kmesh):
    """All q at once via the lattice transform, shape (nkpts, npts, npts).

    The k-sum sum_k Go^k * Gv^{k+q} is a correlation over the mesh, so

        sum_k Go_k * Gv_{k+q} = (1/Nk) FFT_m->q [ Go_{-m} * Gv_m ]

    with Go_m the transform of Go_k. Requires the k-points in
    `cell.make_kpts(kmesh)` order, i.e. C-order over the mesh triple.
    """
    kmesh = tuple(int(n) for n in kmesh)
    nk = int(np.prod(kmesh))
    if len(Go) != nk:
        raise ValueError(f"{len(Go)} k-points for kmesh {kmesh} (expected {nk}) "
                         "-- the FFT route needs the full regular mesh in "
                         "make_kpts order.")
    shape = kmesh + Go.shape[1:]
    axes = (0, 1, 2)
    Go_m = np.fft.ifftn(Go.reshape(shape), axes=axes) * nk
    Gv_m = np.fft.ifftn(Gv.reshape(shape), axes=axes) * nk
    neg = np.ix_(*[(-np.arange(n)) % n for n in kmesh])
    h = Go_m[neg] * Gv_m
    return (np.fft.fftn(h, axes=axes) / nk).reshape(nk, *Go.shape[1:])


def polarizability_all_q_imaginary_time(X, mo_energy, nocc, grid, kmesh,
                                        mu=None, kplus=None, use_fft=True):
    """Pi^q(i.omega) for every q, shape (nkpts, nfreq, npts, npts).

    The tau -> omega map is the grid's cosine transform, and the leading -2
    is the sign of chi_ia = -2 Delta/(Delta^2 + w^2) against the transform of
    e^{-Delta tau} = +2 Delta/(Delta^2 + w^2) -- the same convention as
    LinearResponse/space_time.py, kept identical on purpose.
    """
    mu = chemical_potential(mo_energy, nocc) if mu is None else mu
    nk, npts = len(X), X[0].shape[0]
    Pi = np.zeros((nk, grid.nfreq, npts, npts), dtype=np.complex128)
    for t, tau in enumerate(grid.tau_points):
        Go, Gv = green_functions_tau(X, mo_energy, nocc, tau, mu)
        if use_fft:
            Pi_tau = polarizability_tau_all_q_fft(Go, Gv, kmesh)
        else:
            Pi_tau = np.asarray([polarizability_tau(Go, Gv, kplus, q)
                                 for q in range(nk)])
        Pi += (-2.0 / nk) * grid.cosft_wt[None, :, t, None, None] * Pi_tau[:, None]
    return Pi


def rpa_ecorr_from_pi(Pi_all_q, V, omega_weights):
    """E_c = (1/Nq) sum_q (1/2pi) int dw [logdet(1 - V^q Pi^q) + Tr(V^q Pi^q)].

    Pi_all_q[q] is (nfreq, npts, npts). Returns (E_c, per-q contributions).
    """
    nq = len(V)
    per_q = np.array([ecorr_q_from_pi(Pi_all_q[q], V[q], omega_weights, q=q)
                      for q in range(nq)])
    return per_q.sum() / nq, per_q


def ecorr_q_from_pi(Pi_block, Vq, weights_block, q=0):
    """The (q, omega-block) contribution to E_c, unnormalized by 1/Nq.

    Pulled out of `rpa_ecorr_from_pi` so the streaming route can call it on a
    BLOCK of frequencies and then discard them; the all-q function is a thin
    loop over this.
    """
    eye = np.eye(Vq.shape[0])
    acc = 0.0
    for w, Pi in enumerate(Pi_block):
        M = Vq @ Pi
        sign, logdet = np.linalg.slogdet(eye - M)
        if sign.real <= 0:
            raise RuntimeError(
                f"det(1 - V Pi) is non-positive at q={q}, w={w} "
                f"(sign {sign}) -- the RPA dielectric has been driven "
                f"singular, which on a gapped system means a convention "
                f"error rather than physics.")
        acc += weights_block[w] * (logdet.real + np.trace(M).real)
    return acc / (2.0 * np.pi)


def _checkpoint_path(checkpoint_dir, q):
    return os.path.join(checkpoint_dir, f'ec_q{int(q):04d}.json')


def _read_checkpoint(checkpoint_dir, q):
    """The per-q record, or None if absent. A CORRUPT file is not 'absent'."""
    path = _checkpoint_path(checkpoint_dir, q)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            rec = json.load(fh)
        return {'ec_q': float(rec['ec_q']), 'signature': str(rec['signature'])}
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError(
            f"checkpoint {path} exists but could not be read ({exc}). Treating "
            f"it as missing would silently recompute one transfer with no "
            f"record of why; delete it deliberately instead.") from exc


def _write_checkpoint(checkpoint_dir, q, ec_q, signature):
    """Write atomically: a half-written file read by a merge job is a wrong
    energy, not a crash, and an array job is exactly where that happens.

    A job array runs one q per task, one task per machine, sharing this
    directory, so the temporary name must be unique across machines, not just
    processes -- see `write_json_atomic`."""
    path = _checkpoint_path(checkpoint_dir, q)
    write_json_atomic(path, {'q': int(q), 'ec_q': float(ec_q), 'signature': signature})


def _run_signature(npoints, freqs, mesh, nkpts, mo_energy, mo_occ, coulG_fn):
    """A fingerprint of everything a per-q E_c depends on.

    Checkpoints exist to let a long run resume or a job array merge, and the
    way that goes wrong is a stale file being silently reused after a parameter
    changed -- the answer then looks converged and is a mixture of two
    calculations. So each file carries this, and a mismatch RAISES.

    Covers the rank, the frequency grid (which encodes nw and the transition
    window, hence the mean field's spectrum), the real-space mesh, the k-point
    count, the orbital energies and occupations, and the kernel's identity.
    """
    h = hashlib.sha256()
    for a in (np.asarray(npoints), np.asarray(freqs), np.asarray(mesh),
              np.asarray(nkpts), np.asarray(mo_energy), np.asarray(mo_occ)):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    tag = getattr(coulG_fn, 'damping', None) or getattr(coulG_fn, 'head_2d', None)
    h.update(repr((getattr(coulG_fn, '__name__', None), tag)).encode())
    return h.hexdigest()[:32]


def rpa_ecorr_thc_streaming(cell, mf, npoints=None, alpha=None, nw=32,
                            freq_block=None, q_list=None, mesh=None,
                            ke_cutoff=None, coulG_fn=None, threshold=None,
                            checkpoint_dir=None, return_per_q=False,
                            beta=None):
    """dRPA correlation energy with the q-OUTER, omega-blocked loop.

    Same number as `rpa_ecorr_thc(route='frequency')` -- pinned bit-for-bit in
    the tests -- with the memory discipline a quadruple-zeta slab requires.
    The all-q route materializes two objects this one never forms:

        V for every q            nq x M^2
        Pi for every (q, omega)  nq x nw x M^2

    and the second is why this exists. Note that omega-blocking is needed even
    PER TRANSFER: Pi^q(i.omega) over a 32-point grid is 32 x M^2, so a q-outer
    loop alone does not fit either.

    The structure, which is also the parallel decomposition: each q is
    independent and contributes one scalar, so `q_list` lets a rank own a
    subset and the reduction is a sum. Nothing couples the transfers until
    that sum.

    `freq_block=None` uses one block, i.e. the same footprint per q as the
    all-q route divided by Nq. Any block size gives the same energy; the
    molecular counterpart (`GW/space_time.py`) measures its equivalent exact
    to 6e-15.
    """
    from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map

    kpts = np.asarray(mf.kpts)
    mo = [np.asarray(c) for c in mf.mo_coeff]
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)
    kplus = get_momentum_transfer_map(cell, kpts)
    kw = {} if threshold is None else {'threshold': threshold}

    # Defaulted from the mean field, so a smeared SCF gets a grid that matches
    # its own temperature without the caller restating it -- the same defaulting
    # `ri_rpa_ecorr` does. An unsmeared mf leaves this None and keeps its T = 0
    # grid; beta=0 forces the old unfloored grid.
    if beta is None:
        beta = beta_from_mf(mf)
    freqs, wts, _, _ = frequency_grid_occ(mo_energy, mo_occ, kplus, nw,
                                          beta=beta, threshold=threshold)
    nq = len(kpts)
    q_list = list(range(nq)) if q_list is None else list(q_list)
    blk = nw if freq_block is None else int(freq_block)
    if blk <= 0:
        raise ValueError(f"freq_block must be positive, got {freq_block}")

    # Resolve the rank and the grid BEFORE anything expensive: the checkpoint
    # signature has to be the same whether or not the factorization gets built
    # this call, and both resolvers are cheap.
    npoints = resolve_npoints(npoints, alpha, np.shape(mo[0])[1])
    _, mesh = resolve_grid(cell, mesh, ke_cutoff)

    per_q, todo = {}, list(q_list)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        sig = _run_signature(npoints, freqs, mesh, len(kpts), mo_energy,
                             mo_occ, coulG_fn)
        todo = []
        for q in q_list:
            rec = _read_checkpoint(checkpoint_dir, q)
            if rec is None:
                todo.append(q)
            elif rec['signature'] != sig:
                raise ValueError(
                    f"checkpoint for q={q} in {checkpoint_dir} was written by a "
                    f"DIFFERENT calculation (signature {rec['signature']} "
                    f"against {sig}). Reusing it would silently mix two runs. "
                    f"Point checkpoint_dir somewhere else, or delete it.")
            else:
                per_q[q] = rec['ec_q']

    if todo:                       # only build the factorization if needed
        ctx = isdf_prepare(cell, mo, kpts, npoints=npoints, mesh=mesh,
                           coulG_fn=coulG_fn)
        X = ctx['X']
        for q in todo:
            Vq = isdf_vq(ctx, q, coulG_fn=coulG_fn)      # M^2, kept for this q
            acc = 0.0
            for lo in range(0, nw, blk):                 # omega block
                hi = min(lo + blk, nw)
                Pi = polarizability_q_frequency_occ(X, mo_energy, mo_occ,
                                                    kplus, q, freqs[lo:hi],
                                                    **kw)
                acc += ecorr_q_from_pi(Pi, Vq, wts[lo:hi], q=q)
                del Pi                                   # and DISCARD
            per_q[q] = acc
            del Vq
            if checkpoint_dir is not None:
                _write_checkpoint(checkpoint_dir, q, acc, sig)

    e_c = sum(per_q.values()) / nq
    if return_per_q:
        return e_c, per_q
    return e_c


def rpa_ecorr_thc(cell, mf, npoints=None, nw=32, kmesh=None, mesh=None,
                  coulG_fn=None, route='frequency', ntau=None, beta=None,
                  threshold=None, ke_cutoff=None, alpha=None):
    """dRPA correlation energy from a converged KRHF/KRKS, through THC.

    ONE PATH, NOT A DISPATCH. The frequency route always uses the
    occupation-weighted response, including for a gapped system, because a gap
    is exactly its T -> 0 limit -- verified to 4e-17 in E_c and 5e-16 in Pi on
    diamond. Keeping an integer fast path and choosing between them is how the
    GDF route once went wrong: beta was threaded into the grid, the integer
    chi0 was still being called, every unit test passed, and it surfaced only
    as a discontinuous equation of state. There is nothing here to choose wrongly.

    route='frequency' -- quartic, works for metals and gaps alike.
    route='time'      -- cubic. Needs `beta` for a metal (a bosonic IR grid);
                         without it, falls back to the T = 0 minimax route,
                         which is integer-occupation only and guarded.

    `kmesh` is DERIVED from mf.kpts, not taken on trust: the time route feeds it
    to a lattice FFT, so a mesh that disagrees with the k-points it is indexing
    would reorder the transform silently. It stays in the signature only so an
    explicit value can be CHECKED; passing a wrong one raises.
    """
    from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
    from src.SingleReference.Periodic.pbc_isdf import build_isdf_kpts
    from src.SingleReference.Periodic.pbc_rpa_damping import kmesh_from_kpts

    kpts = np.asarray(mf.kpts)
    mo = [np.asarray(c) for c in mf.mo_coeff]
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)
    kplus = get_momentum_transfer_map(cell, kpts)
    kw = {} if threshold is None else {'threshold': threshold}

    derived = kmesh_from_kpts(cell, kpts)
    if kmesh is not None and tuple(int(n) for n in kmesh) != derived:
        raise ValueError(
            f"kmesh={tuple(kmesh)} does not match the {derived} mesh behind "
            f"mf.kpts. The time route FFTs over this mesh, so a mismatch would "
            f"reorder the transform rather than fail.")
    kmesh = derived

    X, V, info = build_isdf_kpts(cell, mo, kpts, npoints, mesh=mesh,
                                 coulG_fn=coulG_fn, ke_cutoff=ke_cutoff,
                                 alpha=alpha)

    if route == 'frequency':
        # Same defaulting as the streaming driver and as ri_rpa_ecorr: a T = 0
        # grid regardless of smearing is undefined for a metal. beta=0 asks
        # for the T = 0 grid.
        if beta is None:
            beta = beta_from_mf(mf)
        freqs, wts, e_min, e_max = frequency_grid_occ(
            mo_energy, mo_occ, kplus, nw, beta=beta, threshold=threshold)
        Pi = [polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus, q,
                                             freqs, **kw)
              for q in range(len(kpts))]
    elif route == 'time':
        if beta is not None:
            e_min, e_max = transition_window_occ(mo_energy, mo_occ, kplus, **kw)
            grid = TimeFrequencyGrid.ir(beta, 1.5 * e_max, statistics='boson')
            Pi = polarizability_all_q_imaginary_time_occ(
                X, mo_energy, mo_occ, grid, kmesh, kplus=kplus)
        else:
            nocc = require_integer_occupation(mo_occ, who="rpa_ecorr_thc(route='time')")
            e_o, e_v = _energies(mo_energy, nocc)
            grid = TimeFrequencyGrid.minimax(ntau or 18, e_v.min() - e_o.max(),
                                             e_v.max() - e_o.min())
            Pi = polarizability_all_q_imaginary_time(X, mo_energy, nocc, grid,
                                                    kmesh, kplus=kplus)
        freqs, wts = grid.omega_points, grid.omega_weights
    else:
        raise ValueError(f"route={route!r}; choose 'frequency' or 'time'.")

    e_c, per_q = rpa_ecorr_from_pi(Pi, V, wts)
    return e_c, {'per_q': per_q, 'npoints': len(info['points']),
                 'freqs': freqs, 'info': info}


# ---------------------------------------------------------------------------
# Occupation-weighted response (metals)
# ---------------------------------------------------------------------------
#
# The integer routes above split orbitals with one nocc at every k-point. This
# section replaces that with the ragged, occupation-weighted transition space
# of `pbc_occupations`, which is correct for a metal and reduces to the
# integer answer for a gap.
#
# THE WEIGHT IS NOT THE SAME OBJECT IN THE TWO DOMAINS, and conflating them is
# a 1e12 error, not a factor of two. Measured on a 4-level Fermi-Dirac model
# against the Lehmann form:
#
#   * IN FREQUENCY the weight is (f_m - f_n). It is NOT the product
#     f_m (2 - f_n)/2 -- at f_m = f_n = 1 those are 0 and 0.5.
#   * IN IMAGINARY TIME the weight IS the single product f_m (2 - f_n)/2 over
#     ALL orbital pairs, and (f_m - f_n) is then generated by the beta-mirror
#     tau -> beta - tau. Adding a second Hadamard product to "correct" the
#     weight double-counts the mirror and blows up by 1e12, because the
#     reversed pairs it introduces carry the growing exponentials.
#
# The mirror is already in the molecular code and says exactly this:
# LinearResponse/space_time.polarizability_imaginary_time(..., beta=...),
# "the mirror contributes EXACTLY as much as the direct term".
#
# Consequence for the tau route on a metal, which is stronger than the range
# trap of `pbc_isdf_gw`: a
# T = 0 half-line grid is wrong TWICE -- the summand f_m e^{(eps_m - mu) tau}
# is unbounded past tau = beta (1.0 at beta/2, 2.0 at beta, 1.0e22 at
# 1.5 beta), AND without the mirror the weight is not (f_m - f_n) at all.
# thermal_e_min fixes the grid's SCALE; only the beta-periodic structure fixes
# its EXTENT and its WEIGHT.
#
# THE FACTOR OF TWO. The integer routes carry (2/N_k). That 2 is NOT a spin
# factor -- it is the occupation difference itself, f_m - f_n = 2 - 0 for a
# restricted gapped calculation. Folding sqrt(f_m - f_n) into the pair factor
# therefore moves it out of the prefactor, which becomes (1/N_k). Derived from
# the integer limit here; the GDF route reaches the same displacement
# independently by matching spectral W against (1 - Pi0)^-1.

from src.SingleReference.Periodic.pbc_occupations import (
    DEFAULT_OCCUPATION_THRESHOLD, transition_pairs)


def pair_factor_q(X, mo_energy, mo_occ, kplus, q, k,
                  threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """(B, gap) for one (q, k): B[mu, t] = sqrt(f_m - f_n) X^{k*}_{m mu} X^{k+q}_{n mu}.

    Folding sqrt(f_m - f_n) into the pair factor is what keeps the weighting
    invisible downstream -- Pi's formula, its conjugation convention and the
    k -> R FFT identity are all unchanged. Same move as
    `pbc_occupations.stacked_Lw(..., sqrt_weight=True)`.

    The pair list is ragged and ORDERED so that f_m > f_n, which makes the
    weight positive by construction and (given monotonicity of f in e, which
    Fermi-Dirac at one chemical potential gives identically) makes gap > 0.
    """
    kq = kplus[q, k]
    m_idx, n_idx, w, gap = transition_pairs(mo_energy, mo_occ, k, kq,
                                            threshold=threshold)
    if not len(m_idx):
        return None, None
    B = (X[k][:, m_idx].conj() * X[kq][:, n_idx]) * np.sqrt(w)[None, :]
    return B, gap


def polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus, q, freqs,
                                   threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Pi^q(i.omega), occupation-weighted. Correct for a metal AND for a gap.

    Reduces to `polarizability_q_frequency` exactly when the occupations are
    integer and k-independent -- that is the T -> 0 limit, and it is asserted
    rather than assumed (tests/test_pbc_isdf_rpa.py).
    """
    nkpts, npts = len(X), X[0].shape[0]
    Pi = np.zeros((len(freqs), npts, npts), dtype=np.complex128)
    for k in range(nkpts):
        B, gap = pair_factor_q(X, mo_energy, mo_occ, kplus, q, k, threshold)
        if B is None:
            continue
        for w, omega in enumerate(freqs):
            chi = -2.0 * gap / (omega ** 2 + gap ** 2)
            Pi[w] += (1.0 / nkpts) * (B.conj() * chi) @ B.T
    return Pi


def frequency_grid_occ(mo_energy, mo_occ, kplus, nw, beta=None, threshold=None):
    """Gauss-Legendre frequency grid scaled by the occupation-weighted window.

    The scale is floored at the first Matsubara frequency, pi/beta, whenever
    `beta` is given. Unfloored, the grid is scaled by the smallest particle-hole
    transition, which in a metal is set by the level spacing at mu rather than
    by the system: it collapses as the k-mesh refines, so the quadrature comes
    to span a small fraction of the transition spectrum while still returning a
    plausible energy. `thermal_e_min` returns max(gap, pi/beta), leaving a
    gapped system's grid bit-identical.

    beta=0 selects the unfloored grid, since `thermal_e_min` rejects it.
    """
    kw = {} if threshold is None else {'threshold': threshold}
    e_min, e_max = transition_window_occ(mo_energy, mo_occ, kplus, **kw)
    if beta is not None and beta > 0:
        e_min = thermal_e_min(beta, e_min)
    freqs, wts = gauss_legendre_grid(nw, w0=0.5 * e_min)
    return freqs, wts, e_min, e_max


def transition_window_occ(mo_energy, mo_occ, kplus,
                          threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """(e_min, e_max) over the WEIGHTED transition set, for grid construction.

    Computed here from `transition_pairs` over the THC route's momentum map
    `kplus`; `pbc_occupations.transition_window` is the GDF route's
    counterpart, over all k-pairs.
    """
    gaps = []
    for q in range(len(kplus)):
        for k in range(len(kplus)):
            _, _, _, g = transition_pairs(mo_energy, mo_occ, k, kplus[q, k],
                                          threshold=threshold)
            if len(g):
                gaps.append(g)
    if not gaps:
        raise ValueError("no transitions above the occupation threshold")
    g = np.concatenate(gaps)
    return float(g.min()), float(g.max())


def green_functions_tau_occ(X, mo_energy, mo_occ, tau, mu):
    """(Go, Gv) with occupation weights, summed over ALL orbitals.

        Go^k_{nu mu}(tau) = sum_m      f_m  X^k_{m nu}  X^{k*}_{m mu} e^{+(eps_m - mu) tau}
        Gv^k_{nu mu}(tau) = sum_n (2 - f_n) X^{k*}_{n nu} X^k_{n mu}  e^{-(eps_n - mu) tau}

    A SINGLE product of these is the correct imaginary-time object -- see the
    section note. Both sums run over every orbital, not over a slice, which is
    the whole difference from `green_functions_tau` and also why the result is
    only bounded on [0, beta]: for a state well above mu, f_m ~ e^{-beta(eps_m-mu)}
    so the summand goes as e^{(eps_m - mu)(tau - beta)}.

    FORMED IN LOG SPACE, which is not optional. The product f_m e^{(eps_m-mu)tau}
    is bounded for tau <= beta only because the two factors cancel: f_m ~
    e^{-beta(eps_m-mu)} underflows exactly as the exponential overflows.
    Computing them separately and multiplying gives 0 * inf = NaN at any beta
    large enough to matter (measured: NaN at beta = 100 on diamond). In logs,
    log f_m + (eps_m - mu) tau ~ (eps_m - mu)(tau - beta), which is simply
    small. No Fermi-Dirac form is assumed -- only that mo_occ is the occupation.
    """
    e = np.asarray(mo_energy)
    f = np.asarray(mo_occ, dtype=float)
    Go, Gv = [], []
    with np.errstate(divide='ignore', invalid='ignore'):
        logf, logh = np.log(f), np.log(2.0 - f)
    for k in range(len(X)):
        Xk, x = X[k], e[k] - mu
        eo = np.where(f[k] > 0, np.exp(logf[k] + x * tau), 0.0)
        ev = np.where(f[k] < 2.0, np.exp(logh[k] - x * tau), 0.0)
        Go.append((Xk * eo) @ Xk.conj().T)
        Gv.append((Xk.conj() * ev) @ Xk.T)
    return np.asarray(Go), np.asarray(Gv)


def polarizability_all_q_imaginary_time_occ(X, mo_energy, mo_occ, grid, kmesh,
                                            mu=None, kplus=None, use_fft=True,
                                            symmetrize=False):
    """Pi^q(i.omega) for every q, occupation-weighted, via imaginary time.

    REQUIRES a bosonic IR grid, i.e. one whose tau axis IS [0, beta]. Three
    independent reasons, and the third is the one that surprises people:

      1. BOUNDEDNESS -- the summand diverges past tau = beta (1.0 at beta/2,
         2.0 at beta, 1.0e22 at 1.5 beta).
      2. THE WEIGHT -- integrating over the FINITE interval supplies
         int_0^beta e^{i nu tau} e^{-Delta tau} dtau = (e^{-Delta beta} - 1)/(i nu - Delta),
         whose (1 - e^{-beta Delta}) factor is exactly what turns the single
         product f_m (2 - f_n)/2 into f_m - f_n. On a half-line you get the
         wrong weight -- by exactly a factor of two.
      3. They are INDEPENDENT: (2) bites even where the integrand is bounded.

    So no explicit tau -> beta - tau mirror appears here. The mirror in
    `LinearResponse/space_time.polarizability_imaginary_time(..., beta=...)` and
    this finite-interval factor are the same object seen from two sides -- that
    code needs it because its minimax transform is a HALF-LINE integral, and
    doing both would double count.

    Prefactor -1/Nk, PINNED BY MEASUREMENT and then explained, in that order.
    The integral above gives -1/(2 Nk); the measured deviation from the
    frequency route was then exactly 0.500 at beta = 100, 300 and 1000, on
    every q and every matrix element. The missing 2 is the even-sector
    projection: `cosft_wt` represents the component of its argument that is
    EVEN about tau = beta/2, and the even part of S is (S(tau) + S(beta-tau))/2
    -- half of the symmetrized object the bosonic transform acts on. Passing
    the explicitly symmetrized S with -1/(2 Nk) gives the identical answer,
    which is the check that this is a projection and not a fudge
    (`symmetrize=True` below).
    """
    mu = chemical_potential_occ(mo_energy, mo_occ) if mu is None else mu
    if grid.method != 'IR' or grid.meta.get('statistics') != 'boson':
        raise ValueError(
            f"occupation-weighted imaginary time needs a bosonic IR grid; got "
            f"method={grid.method!r}, statistics={grid.meta.get('statistics')!r}. "
            f"A T = 0 half-line grid (minimax) is wrong here for three separate "
            f"reasons -- see this function's docstring. Build it with "
            f"TimeFrequencyGrid.ir(beta, omega_max, statistics='boson').")

    beta = grid.meta['beta']
    nk, npts = len(X), X[0].shape[0]
    Pi = np.zeros((nk, grid.nfreq, npts, npts), dtype=np.complex128)
    taus = [(tau, beta - tau) if symmetrize else (tau,)
            for tau in grid.tau_points]
    scale = -0.5 / nk if symmetrize else -1.0 / nk
    for t, ts in enumerate(taus):
        P = None
        for tau in ts:
            Go, Gv = green_functions_tau_occ(X, mo_energy, mo_occ, tau, mu)
            Pt = (polarizability_tau_all_q_fft(Go, Gv, kmesh) if use_fft else
                  np.asarray([polarizability_tau(Go, Gv, kplus, q)
                              for q in range(nk)]))
            P = Pt if P is None else P + Pt
        Pi += scale * grid.cosft_wt[None, :, t, None, None] * P[:, None]
    return Pi


def chemical_potential_occ(mo_energy, mo_occ, threshold=0.5):
    """Midpoint of the partially occupied band, for the tau exponentials only.

    Uses the occupations rather than an integer split, so it is defined for a
    metal. Falls back to the midgap when the occupations are integer.
    """
    e = np.asarray(mo_energy)
    f = np.asarray(mo_occ, dtype=float)
    partial = (f > threshold) & (f < 2.0 - threshold)
    if partial.any():
        return float(e[partial].mean())
    occ, virt = f > 1.0, f <= 1.0
    return 0.5 * (float(e[occ].max()) + float(e[virt].min()))
