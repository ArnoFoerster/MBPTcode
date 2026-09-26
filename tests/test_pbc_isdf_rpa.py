"""THC-RPA with k-point sampling.

The validation here is deliberately a ladder of three, because only the last
rung needs an external oracle and the first two are exact:

  1. the k -> R FFT against the direct k-sum. Same quantity, no grid, no
     approximation -- so this must hold at machine precision. It is the only
     genuinely new algorithmic step on the consumer side, and the one place a
     momentum-mesh indexing error could hide;
  2. the imaginary-time route against Pi built from its definition. Isolates
     the tau grid and the cosine transform from everything else;
  3. E_c against `pbc_rpa.ri_rpa_ecorr`, which is the physics.

All three run here. Rung 3 compares against a stored RI-V value rather than
recomputing it, because that reference costs hours on this mesh while the
whole THC route costs seconds -- see RI_V_ECORR below.
"""
import os
import sys
import threading
import time

import numpy as np
import pytest
from pyscf.pbc import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.mpi_grid import run_simulated
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.Periodic import pbc_isdf_rpa as thcrpa
from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
from src.SingleReference.Periodic.pbc_isdf import build_isdf_kpts

KMESH = [2, 2, 1]


def _diamond():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.build()
    return cell


@pytest.fixture(scope='module')
def thc():
    cell = _diamond()
    kpts = cell.make_kpts(KMESH)
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.kernel()
    assert mf.converged
    mo = [np.asarray(c) for c in mf.mo_coeff]
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)
    nocc = int((np.asarray(mf.mo_occ)[0] > 1e-8).sum())
    nmo = mo[0].shape[1]
    kplus = get_momentum_transfer_map(cell, kpts)
    X, V, info = build_isdf_kpts(cell, mo, kpts, 8 * nmo)
    return cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V


def test_kspace_fft_matches_direct_ksum(thc):
    """Rung 1. The correlation sum_k Go^k * Gv^{k+q} done as an FFT over the
    mesh is the SAME number, not an approximation of it, so anything above
    machine precision is a mesh-indexing bug.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    mu = thcrpa.chemical_potential(mo_energy, nocc)
    worst = 0.0
    for tau in (0.05, 0.5, 2.0):
        Go, Gv = thcrpa.green_functions_tau(X, mo_energy, nocc, tau, mu)
        fft = thcrpa.polarizability_tau_all_q_fft(Go, Gv, KMESH)
        for q in range(len(kpts)):
            direct = thcrpa.polarizability_tau(Go, Gv, kplus, q)
            worst = max(worst, abs(fft[q] - direct).max() / abs(direct).max())
    assert worst < 1e-12, f"FFT vs direct k-sum: {worst:.3e}"


def test_fft_route_rejects_a_mismatched_mesh(thc):
    """The FFT route silently gives nonsense if handed k-points that are not
    the full regular mesh in make_kpts order, so it refuses instead."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    mu = thcrpa.chemical_potential(mo_energy, nocc)
    Go, Gv = thcrpa.green_functions_tau(X, mo_energy, nocc, 0.5, mu)
    with pytest.raises(ValueError, match='regular mesh'):
        thcrpa.polarizability_tau_all_q_fft(Go, Gv, [2, 2, 2])


def test_imaginary_time_route_matches_the_definition(thc):
    """Rung 2. Pi from Go * Gv plus a cosine transform, against Pi built term
    by term in frequency. Isolates the tau grid from the factorization."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    e_min = mo_energy[:, nocc:].min() - mo_energy[:, :nocc].max()
    e_max = mo_energy[:, nocc:].max() - mo_energy[:, :nocc].min()
    grid = TimeFrequencyGrid.minimax(18, e_min, e_max)
    Pi_t = thcrpa.polarizability_all_q_imaginary_time(X, mo_energy, nocc, grid,
                                                      KMESH)
    worst = 0.0
    for q in range(len(kpts)):
        Pi_f = thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q,
                                                 grid.omega_points)
        worst = max(worst, abs(Pi_t[q] - Pi_f).max() / abs(Pi_f).max())
    assert worst < 1e-7, f"tau route vs definition: {worst:.3e}"


def test_polarizability_is_hermitian(thc):
    """Pi is Hermitian in the interpolation index -- which is exactly why a
    misplaced conjugate in its construction would NOT show up here, and why
    the E_c comparison is the real check on that."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    freqs = np.array([0.1, 1.0])
    for q in range(len(kpts)):
        Pi = thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q, freqs)
        for P in Pi:
            assert abs(P - P.conj().T).max() < 1e-10 * abs(P).max()


def test_polarizability_is_negative_semidefinite(thc):
    """chi0 on the imaginary axis has non-positive eigenvalues. A sign error
    anywhere in the transform chain flips this, and unlike Hermiticity it is
    not automatic."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    for q in range(len(kpts)):
        Pi = thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q,
                                               np.array([0.5]))
        w = np.linalg.eigvalsh(Pi[0])
        assert w.max() < 1e-10 * max(1.0, abs(w).max())


#: `pbc_rpa.ri_rpa_ecorr(mf, nw=32)` on the KMESH fixture above. Pinned
#: rather than recomputed: the RI-V reference rebuilds ft_aopair over the full
#: 47^3 G-grid for every k-pair, orders of magnitude slower than the whole THC
#: route including the factorization.
RI_V_ECORR = -0.11677862


def test_ecorr_matches_ri_v(thc):
    """Rung 3, the physics. THC-RPA against RI-V dRPA on the same mean field.

    Held to 1e-4 Ha. Note this is ~25x TIGHTER than the ~2.6e-3 GDF-vs-FFTDF
    ERI discrepancy of test_pbc_isdf_gamma.py would suggest, and deliberately so: that bar
    bounds comparisons of ERIs, not of energies. E_c contracts the ERI error
    against a smooth response function and the bulk of it cancels, so the
    measured agreement at alpha >= 8 is 1e-5 Ha or better. Setting the bar at
    2e-3 here would have made this test unable to fail.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    from src.SingleReference.Periodic.pbc_rpa import build_freq_grid
    freqs, wts = build_freq_grid(mo_energy, nocc, 32)
    Pi = [thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q, freqs)
          for q in range(len(kpts))]
    e_c, per_q = thcrpa.rpa_ecorr_from_pi(Pi, V, wts)
    assert abs(e_c - RI_V_ECORR) < 1e-4, f"E_c(THC) {e_c:.8f} vs {RI_V_ECORR}"


def test_cubic_time_route_matches_quartic_frequency_route():
    """Rung 3b. The CUBIC route reaches the same E_c as the quartic one.

    The imaginary-time route is the whole point of the space-time formulation
    -- O(N_mu^2) per tau against O(N_mu^2 n_occ n_virt) per omega -- and rung
    2 compares Pi alone while rung 3 above runs the frequency route, so without
    this the cubic path could be wrong in any way that survives a Pi comparison
    and no test would say so. It also pins that `route='time'` derives its
    mesh from mf.kpts.

    Driven through the public entry point on purpose, so the driver's own grid
    and mesh plumbing is under test and not just the kernels.
    """
    cell = _diamond()
    kpts = cell.make_kpts(KMESH)
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.kernel()
    assert mf.converged
    npts = 12 * np.asarray(mf.mo_coeff).shape[-1]

    e_freq, _ = thcrpa.rpa_ecorr_thc(cell, mf, npts, nw=32, route='frequency')
    e_time, _ = thcrpa.rpa_ecorr_thc(cell, mf, npts, route='time', ntau=18)

    # The two routes are the same integral on two different quadratures, so
    # they agree far better than either agrees with RI-V: measured 2.4e-11.
    assert abs(e_freq - e_time) < 1e-8, (e_freq, e_time)
    assert abs(e_time - RI_V_ECORR) < 1e-4, f"E_c(time) {e_time:.8f}"


def test_time_route_refuses_a_mismatched_kmesh():
    """A kmesh that disagrees with mf.kpts must raise, not silently reorder.

    The time route FFTs over the mesh, so a wrong-but-plausible mesh permutes
    the transform instead of failing -- the same class of silent error as the
    correlation/convolution swap in test_pbc_isdf_kindex.py.
    """
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts(KMESH)).density_fit()
    mf.kernel()
    npts = 4 * np.asarray(mf.mo_coeff).shape[-1]
    with pytest.raises(ValueError, match='does not match'):
        thcrpa.rpa_ecorr_thc(cell, mf, npts, kmesh=[2, 2, 2], route='time',
                             ntau=18)


def test_streaming_route_is_bit_identical_to_the_all_q_route():
    """The q-outer, omega-blocked loop must be the SAME number.

    It exists for memory, not accuracy, so any deviation is a defect rather
    than a trade-off. Measured: exactly 0 at one block, and 1.4e-17 (one ULP,
    from summation order) at every smaller block size.

    What it avoids: V for every q and Pi for every (q, omega) -- but note the
    binding term is neither of those: it is the (N_mu x N_r) zeta
    intermediates.
    """
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts(KMESH)).density_fit()
    mf.kernel()
    assert mf.converged
    ref, _ = thcrpa.rpa_ecorr_thc(cell, mf, alpha=8, nw=32)
    for blk in (None, 8, 1):
        e = thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=32,
                                           freq_block=blk)
        assert abs(e - ref) < 1e-15, (blk, e, ref)


def test_q_subsets_reduce_to_the_whole():
    """Each transfer contributes one scalar and nothing couples them.

    This is the parallel decomposition: a rank owns a `q_list`, and the
    reduction is a plain sum. Asserted here so the distributed route inherits
    a checked property rather than an assumed one.
    """
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts(KMESH)).density_fit()
    mf.kernel()
    ref, _ = thcrpa.rpa_ecorr_thc(cell, mf, alpha=8, nw=16)
    nq = len(mf.kpts)
    acc = {}
    for qs in ([0, 1], [2, 3]):
        _, per_q = thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=16,
                                                  q_list=qs, return_per_q=True)
        acc.update(per_q)
    assert sorted(acc) == list(range(nq))
    assert abs(sum(acc.values()) / nq - ref) < 1e-15


def test_checkpoints_shard_merge_and_resume(tmp_path):
    """The job-array pattern: shards write, a merge reads, exactly.

    Each transfer's E_c is one scalar, so an array task can own a `q_list`,
    write per-q files, and a merge job sum them. Verified here in one process
    because the mechanism, not the scheduler, is what can be wrong.

    Resume must also SKIP the factorization when everything is cached -- the
    build is the expensive half, so a resume that rebuilds it has not resumed.
    """
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts(KMESH)).density_fit()
    mf.kernel()
    ck = str(tmp_path / 'ck')
    ref, _ = thcrpa.rpa_ecorr_thc(cell, mf, alpha=8, nw=16)

    for shard in ([0, 1], [2, 3]):
        thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=16, q_list=shard,
                                       checkpoint_dir=ck)
    assert sorted(os.listdir(ck)) == [f'ec_q{q:04d}.json' for q in range(4)]

    merged = thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=16,
                                            checkpoint_dir=ck)
    assert abs(merged - ref) < 1e-15, (merged, ref)

    t0 = time.time()
    again = thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=16,
                                           checkpoint_dir=ck)
    assert abs(again - ref) < 1e-15
    assert time.time() - t0 < 1.0, 'a fully cached resume rebuilt something'


def test_a_stale_checkpoint_is_refused(tmp_path):
    """Reusing a checkpoint from other parameters would MIX two calculations.

    The result would look converged and be a blend, which no downstream check
    could detect -- so the signature covers the rank, the frequency grid, the
    mesh, the k-count, and the mean field's spectrum, and a mismatch raises.
    """
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts(KMESH)).density_fit()
    mf.kernel()
    ck = str(tmp_path / 'ck')
    thcrpa.rpa_ecorr_thc_streaming(cell, mf, alpha=8, nw=16, checkpoint_dir=ck)
    for changed in ({'alpha': 12, 'nw': 16}, {'alpha': 8, 'nw': 24}):
        with pytest.raises(ValueError, match='DIFFERENT calculation'):
            thcrpa.rpa_ecorr_thc_streaming(cell, mf, checkpoint_dir=ck,
                                           **changed)


def test_checkpoint_write_survives_a_shared_path(tmp_path):
    """The per-q checkpoint is a `write_json_atomic` call site of its own, and
    a job array runs one q per task, one task per machine, onto one shared
    checkpoint directory -- so it is exposed to a temporary-name collision
    between writers.

    A large signature forces `json.dump` into several writes, so a temporary
    name shared between writers would have something to tear.
    """
    checkpoint_dir = str(tmp_path)
    signatures = ['a' * 200_000 + str(r) for r in range(2)]
    barrier = threading.Barrier(2)

    def one_rank(comm):
        r = comm.Get_rank()
        seen = []
        for _ in range(20):
            barrier.wait(timeout=60)
            thcrpa._write_checkpoint(checkpoint_dir, 0, float(r), signatures[r])
            seen.append(thcrpa._read_checkpoint(checkpoint_dir, 0))
        return seen

    for seen in run_simulated(one_rank, 2):
        assert len(seen) == 20
        for got in seen:
            assert got['signature'] in signatures  # whole, and one writer's own
    got = thcrpa._read_checkpoint(checkpoint_dir, 0)
    assert got['signature'] in signatures
    # a private name is only unique while it exists, so it has to be cleaned up
    assert not [p for p in os.listdir(tmp_path) if p.endswith('.tmp')]


def test_ecorr_improves_with_rank(thc):
    """alpha = 4 must be visibly worse than the alpha = 8 fixture.

    Guards against the E_c agreement being insensitive to the factorization
    -- if a bug made V or Pi independent of the rank, the rung-3 test alone
    would still pass.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    from src.SingleReference.Periodic.pbc_rpa import build_freq_grid
    freqs, wts = build_freq_grid(mo_energy, nocc, 32)
    X4, V4, _ = build_isdf_kpts(cell, mo, kpts, 4 * mo[0].shape[1])
    Pi4 = [thcrpa.polarizability_q_frequency(X4, mo_energy, nocc, kplus, q, freqs)
           for q in range(len(kpts))]
    e4, _ = thcrpa.rpa_ecorr_from_pi(Pi4, V4, wts)
    assert abs(e4 - RI_V_ECORR) > 1e-3


def test_metallic_occupations_are_refused():
    """The metallic case must fail LOUDLY, because it cannot fail visibly.

    Every route in pbc_isdf_rpa slices occ/virt with one integer nocc at every
    k-point. That is exact for a gap and wrong for a metal, and the wrongness
    is a continuity failure across a parameter -- an equation of state jumping
    when a per-k occupied count steps -- not a pointwise one. No test in this
    file, including all of the ones above, would notice it, because at any
    single fixed geometry the integer split is perfectly self-consistent.
    Hence a guard rather than a warning.

    Both real failure modes are covered: genuinely fractional occupations
    (Fermi smearing), and integer occupations whose COUNT differs per k-point
    -- measured [3, 4, 4, 6, 4, 6, 6, 8] for bulk BCC Li at 2x2x2.

    NOTE this guard protects only the T = 0 imaginary-time route. The
    frequency route and the beta/IR time route are occupation-weighted and
    handle metals directly.
    """
    from src.SingleReference.Periodic.pbc_isdf_rpa import require_integer_occupation

    gapped = np.array([[2., 2., 0., 0.], [2., 2., 0., 0.]])
    assert require_integer_occupation(gapped) == 2

    smeared = np.array([[2., 1.3, 0.7, 0.], [2., 1.9, 0.1, 0.]])
    with pytest.raises(NotImplementedError, match='fractional occupations'):
        require_integer_occupation(smeared)

    ragged = np.array([[2., 2., 0., 0.], [2., 2., 2., 0.]])
    with pytest.raises(NotImplementedError, match='varies across k-points'):
        require_integer_occupation(ragged)


# --- occupation-weighted response ------------------------------------------

def test_weighted_response_reduces_to_the_integer_one_exactly(thc):
    """THE GATE. A gapped system is the T -> 0 limit of the weighted
    expression, so the two must agree to machine precision -- not to a
    tolerance.

    This is what fixes the CONSTANTS, and it is precisely what a continuity
    test cannot see: folding sqrt(f_m - f_n) into the pair factor moves a
    factor of 2 out of the prefactor (that 2 was never a spin factor, it is
    f_m - f_n = 2 - 0 for a restricted gapped calculation), and only an exact
    reduction pins that.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    from src.SingleReference.Periodic.pbc_rpa import build_freq_grid
    freqs, wts = build_freq_grid(mo_energy, nocc, 32)

    worst = 0.0
    for q in range(len(kpts)):
        Pi_i = thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q, freqs)
        Pi_w = thcrpa.polarizability_q_frequency_occ(X, mo_energy, mo_occ,
                                                     kplus, q, freqs)
        worst = max(worst, abs(Pi_i - Pi_w).max() / abs(Pi_i).max())
    assert worst < 1e-13, f"weighted vs integer Pi: {worst:.3e}"

    Pi_i = [thcrpa.polarizability_q_frequency(X, mo_energy, nocc, kplus, q, freqs)
            for q in range(len(kpts))]
    Pi_w = [thcrpa.polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus, q, freqs)
            for q in range(len(kpts))]
    e_i, _ = thcrpa.rpa_ecorr_from_pi(Pi_i, V, wts)
    e_w, _ = thcrpa.rpa_ecorr_from_pi(Pi_w, V, wts)
    assert abs(e_w - e_i) < 1e-12, f"E_c weighted {e_w} vs integer {e_i}"
    assert abs(e_w - RI_V_ECORR) < 1e-4


def test_weighted_tau_route_matches_weighted_frequency(thc):
    """The cubic weighted route against the quartic weighted one, on a bosonic
    IR grid. beta = 100 is the physically relevant scale (a metal at Fermi
    sigma = 0.01 Ha IS beta = 100); accuracy degrades at larger beta as the
    T -> 0 limit stresses the IR representation, which is the right trade.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    _, e_max = thcrpa.transition_window_occ(mo_energy, mo_occ, kplus)
    grid = TimeFrequencyGrid.ir(100.0, 1.5 * e_max, statistics='boson')
    Pi_t = thcrpa.polarizability_all_q_imaginary_time_occ(X, mo_energy, mo_occ,
                                                          grid, KMESH)
    worst = 0.0
    for q in range(len(kpts)):
        Pi_f = thcrpa.polarizability_q_frequency_occ(X, mo_energy, mo_occ,
                                                     kplus, q, grid.omega_points)
        worst = max(worst, abs(Pi_t[q] - Pi_f).max() / abs(Pi_f).max())
    assert worst < 1e-5, f"weighted tau vs weighted frequency: {worst:.3e}"


def test_symmetrizing_tau_is_the_same_object(thc):
    """cosft_wt projects onto the component even about tau = beta/2, and the
    even part of S is half of S(tau) + S(beta - tau). Feeding the explicitly
    symmetrized S with half the prefactor must therefore give the same answer.

    This is the check that the factor of 2 in the prefactor is a projection
    and not a fudge -- the analytic route gives -1/(2 Nk) and the measured
    deviation from the frequency route was exactly 0.500 at every beta.
    """
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    _, e_max = thcrpa.transition_window_occ(mo_energy, mo_occ, kplus)
    grid = TimeFrequencyGrid.ir(100.0, 1.5 * e_max, statistics='boson')
    A = thcrpa.polarizability_all_q_imaginary_time_occ(X, mo_energy, mo_occ,
                                                       grid, KMESH)
    S = thcrpa.polarizability_all_q_imaginary_time_occ(X, mo_energy, mo_occ,
                                                       grid, KMESH,
                                                       symmetrize=True)
    assert abs(S - A).max() / abs(A).max() < 1e-5


def test_weighted_tau_refuses_a_t0_grid(thc):
    """A T = 0 half-line grid is wrong here for three independent reasons --
    unbounded summand past tau = beta, wrong weight without the finite-interval
    (1 - e^{-beta Delta}) factor, and the two fail separately. So it is
    refused rather than silently used."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    e_min, e_max = thcrpa.transition_window_occ(mo_energy, mo_occ, kplus)
    minimax = TimeFrequencyGrid.minimax(18, e_min, e_max)
    with pytest.raises(ValueError, match='bosonic IR grid'):
        thcrpa.polarizability_all_q_imaginary_time_occ(X, mo_energy, mo_occ,
                                                       minimax, KMESH)


def test_green_functions_do_not_overflow_at_large_beta(thc):
    """f_m e^{(eps_m - mu) tau} is bounded only because the factors cancel.
    Formed naively it is 0 * inf = NaN at any beta large enough to matter
    (measured: NaN at beta = 100 on this system), so it is formed in logs."""
    cell, kpts, mo, mo_energy, mo_occ, nocc, kplus, X, V = thc
    mu = thcrpa.chemical_potential_occ(mo_energy, mo_occ)
    for tau in (1.0, 100.0, 1000.0):
        Go, Gv = thcrpa.green_functions_tau_occ(X, mo_energy, mo_occ, tau, mu)
        assert np.isfinite(Go).all(), f"Go not finite at tau={tau}"
        assert np.isfinite(Gv).all(), f"Gv not finite at tau={tau}"

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
