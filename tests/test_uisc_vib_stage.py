"""The vib stage of the UISC model: finite differences of V^2 along the
promoting modes at each crossing's initial minimum.

The claims kept from the UISC protocol of Samanta, Kim, Coropceanu and
Bredas, J. Am. Chem. Soc. 139, 4042 (2017), as the staged run carries them:

- |V| is a norm and passes through zero for a Condon-forbidden pair, so a
  central difference of the modulus vanishes; V^2 is differenced instead,
  which is exact for V linear in q, Condon term included;
- the promoting modes are picked by Huang-Rhys weight of the S1 <-> T1
  displacement on the S0 modes, not by |dV/dq|, and S0 modes with an
  imaginary one are refused;
- a displaced root is identified by the overlap of its Casida vector with the
  centre's, never by its index or the sign of its eigenvector;
- the stage's record is a function of the declaration: a stored one is taken
  as it is, and a store holding it refuses another declaration.

No SCF and no GW-BSE: the evaluation at a geometry is replaced by an analytic
toy whose spin-orbit table is linear in the dimensionless normal coordinates
(roots returned permuted and sign-flipped, as a solver may), on formaldehyde's
six internal modes.
"""
import numpy as np
import pytest

from src.photophysics import cli, uisc
from src.photophysics.stages import StageRefused, vib as vib_stage
from src.photophysics.store import StoreRefused
from src.properties.curvilinear import huang_rhys_curvilinear
from src.properties.vibronic import align_to
from tests.photophysics_cases import (FORMALDEHYDE_TOY, internal_modes_of,
                                      run_in)

OMEGA = np.array([0.006, 0.0075, 0.002, 0.001, 0.004, 0.012])
#: the S1 -> T1 displacement along each mode (mass-weighted units); mode 3,
#: the softest, is not displaced
DELTA = np.array([2.0, 5.0, 1.0, 0.0, 3.0, 0.5])
N_ROOTS, N_OV = 3, 6
#: how the toy solver returns its roots at a displaced geometry
PERM = {'singlet': np.array([2, 0, 1]), 'triplet': np.array([1, 2, 0])}
SIGN = {'singlet': np.array([-1.0, 1.0, -1.0]),
        'triplet': np.array([1.0, -1.0, -1.0])}


def declaration(*extra, rate_model='uisc'):
    args = cli.parser().parse_args(
        ['--atom', FORMALDEHYDE_TOY, '--system', 'toy', '--basis', 'cc-pvdz',
         '--environment', 'gas', '--temperature', '300', '--lambda-env',
         '0.05', '--nudge', '0.01', '--no-far-start', '--refreeze', '1',
         '--saddle-check', '0', '--df-check', '0', '--hessian',
         'analytic-df', '--relaxed-solvation', 'equilibrium', '--span',
         'full', '--manifold-gaps', 'stacked', '--scf', 'df',
         '--qp-states', 'frontier', '--rate-model', rate_model,
         '--uisc-modes', '2', '--uisc-blocks', '2', '2', '--out', 'x',
         *extra])
    return cli.declaration_of(args)


@pytest.fixture
def staged(tmp_path):
    """(run, modes, masses, g): a run on a store holding the hess, vert, s1
    and t1 records of the toy, and the toy's derivative tensors g[crossing]
    of shape (6, 3, 3, 3), the first index the mode."""
    d = declaration()
    run = run_in(tmp_path, d, 1000)
    modes, masses = internal_modes_of(run.mol_in)
    w = np.repeat(masses, 3) ** -0.5
    x0 = run.mol_in.atom_coords()
    a_s = np.array([1.0, -2.0, 0.5, 0.3, 1.5, -1.0])

    def at(a):
        return x0 + (w * (modes @ a)).reshape(-1, 3)

    run.write('hess', {'omega': OMEGA, 'modes': modes, 'masses': masses,
                       'route': 'analytic-df'})
    run.write('vert', {'props': {'coords_bohr': x0}})
    run.write('s1', {'props': {'coords_bohr': at(a_s)}, 'tag': 'isc'})
    run.write('t1', {'props': {'coords_bohr': at(a_s - DELTA)},
                     'tag': 'risc'})
    rng = np.random.default_rng(11)
    g = {c: 1e-6 * rng.standard_normal((6, 3, 3, 3))
         for c in ('isc', 'risc')}
    t0 = {c: 1e-6 * rng.standard_normal((3, 3, 3)) for c in ('isc', 'risc')}
    return run, modes, masses, g, t0


def toy_electronics(monkeypatch, run, modes, masses, g, t0):
    """Replace the chain and the evaluation of `uisc` by the analytic toy;
    returns the list of geometries the toy was asked for."""
    calls = []
    sq = np.sqrt(np.repeat(masses, 3))
    ref = np.eye(N_OV)[:, :N_ROOTS]

    def chain_at(run_, centre, record):
        return centre.atom_coords().copy(), record['tag']

    def spectrum(spin, flipped):
        perm = PERM[spin] if flipped else np.arange(N_ROOTS)
        sign = SIGN[spin] if flipped else np.ones(N_ROOTS)
        x = ref[:, perm] * sign
        # triplets carry a Y block, singlets are Tamm-Dancoff
        y = (0.1 * np.roll(ref, 3, axis=0)[:, perm] * sign
             if spin == 'triplet' else None)
        return np.arange(N_ROOTS, dtype=float), x, y

    def solve_at(run_, chain_, mol):
        centre, tag = chain_
        calls.append((tag, mol.atom_coords().copy()))
        dx = (mol.atom_coords() - centre).ravel()
        # dimensionless coordinate of mode k: sqrt(omega_k) L_k^T M^1/2 dx
        q = np.sqrt(np.abs(OMEGA)) * (modes.T @ (sq * dx))
        true = t0[tag] + np.einsum('k,kijx->ijx', q, g[tag])
        flipped = bool(np.abs(dx).max() > 0)
        ps, pt = ((PERM['singlet'], PERM['triplet']) if flipped
                  else (np.arange(N_ROOTS),) * 2)
        ss, st = ((SIGN['singlet'], SIGN['triplet']) if flipped
                  else (np.ones(N_ROOTS),) * 2)
        out = (true[np.ix_(ps, pt)] * ss[:, None, None] * st[None, :, None])
        return (spectrum('singlet', flipped), spectrum('triplet', flipped), out,
                (mol, None, None))

    def overlap(key, here, nocc, x_ref, y_ref, x, y):
        # the toy's orbitals do not move: the transported overlap is the
        # normalized dot product of the two Casida vectors
        y_ref = np.zeros_like(x_ref) if y_ref is None else y_ref
        y = np.zeros_like(x) if y is None else y
        ov = x_ref.T @ x + y_ref.T @ y
        nr = np.sqrt((x_ref ** 2 + y_ref ** 2).sum(axis=0))
        nd = np.sqrt((x ** 2 + y ** 2).sum(axis=0))
        s = np.eye(ov.shape[0] + 1, ov.shape[1] + 1)
        s[1:, 1:] = ov / np.outer(nr, nd)
        return s

    monkeypatch.setattr(uisc, 'chain_at', chain_at)
    monkeypatch.setattr(uisc, 'solve_at', solve_at)
    monkeypatch.setattr(uisc, 'transported_overlap', overlap)
    return calls


def block(t, rows=2, cols=2):
    return float(np.sqrt((np.asarray(t)[:rows, :cols] ** 2).sum()))


def test_difference_of_v_squared_is_exact_for_v_linear_in_q(staged,
                                                            monkeypatch):
    run, modes, masses, g, t0 = staged
    calls = toy_electronics(monkeypatch, run, modes, masses, g, t0)
    record = uisc.vib(run)
    picked = [int(k) for k in record['modes']]
    assert len(picked) == 2
    assert len(calls) == 2 * (1 + 2 * len(picked))
    for crossing in ('isc', 'risc'):
        assert record[f'v0_{crossing}'] == pytest.approx(
            block(t0[crossing]), rel=1e-12)
        for i, k in enumerate(picked):
            # |dV/dq_k| of the block: exact, with a Condon term present
            assert record[f'dv_dq_{crossing}'][i] == pytest.approx(
                block(g[crossing][k]), rel=1e-8)
        # the permuted, sign-flipped roots were followed: overlap 1
        assert np.allclose(record[f'overlap_{crossing}'], 1.0)
    # the forbidden pair: no Condon term, the modulus differenced is zero and
    # V^2 is not
    t0_zero = {c: np.zeros((3, 3, 3)) for c in t0}
    calls.clear()
    run.store.path('vib').unlink()
    toy_electronics(monkeypatch, run, modes, masses, g, t0_zero)
    forbidden = uisc.vib(run)
    assert forbidden['v0_isc'] == 0.0
    assert forbidden['dv_dq_isc'][0] == pytest.approx(
        block(g['isc'][int(forbidden['modes'][0])]), rel=1e-8)


def test_modes_are_picked_by_huang_rhys_weight(staged, monkeypatch):
    run, modes, masses, g, t0 = staged
    toy_electronics(monkeypatch, run, modes, masses, g, t0)
    record = uisc.vib(run)
    # the weights from the two minima in R0's frame, independently: the
    # displacement in valence internals linearized at R0
    mol0 = run.molecule_at(run.read('vert')['props']['coords_bohr'])
    s1 = align_to(mol0, run.molecule_at(run.read('s1')['props']['coords_bohr']))
    t1 = align_to(mol0, run.molecule_at(run.read('t1')['props']['coords_bohr']))
    s_k = huang_rhys_curvilinear(mol0, s1, t1, OMEGA, modes, masses)[0]
    want = [int(k) for k in np.argsort(-s_k)[:2]]
    assert [int(k) for k in record['modes']] == want == [1, 4]
    assert np.array_equal(record['omega'], OMEGA[want])
    assert np.allclose(record['huang_rhys'], s_k[want])
    for key in uisc.VIB_KEYS:
        assert key in record


def test_an_imaginary_mode_is_refused(staged, monkeypatch):
    run, modes, masses, g, t0 = staged
    toy_electronics(monkeypatch, run, modes, masses, g, t0)
    hess = run.read('hess')
    run.write('hess', dict(hess, omega=np.where(np.arange(6) == 3, -0.001,
                                                OMEGA)))
    with pytest.raises(ValueError, match='imaginary'):
        uisc.vib(run)


def test_a_stored_record_is_reused_and_another_declaration_is_refused(
        staged, monkeypatch, tmp_path):
    run, modes, masses, g, t0 = staged
    toy_electronics(monkeypatch, run, modes, masses, g, t0)
    first = uisc.vib(run)

    def refuse(*args):
        raise AssertionError('the stored vib record is solved again')

    monkeypatch.setattr(uisc, 'solve_at', refuse)
    again = vib_stage(run)
    for key in uisc.VIB_KEYS:
        assert np.array_equal(again[key], first[key])
    # another declaration on the same store: the record is another surface's
    other = declaration('--uisc-modes', '3')
    assert other.identity() != run.declaration.identity()
    run2 = run_in(tmp_path, other, 1000)
    with pytest.raises(StoreRefused):
        vib_stage(run2)


def test_vib_is_a_stage_of_the_uisc_model_only(tmp_path):
    d = declaration(rate_model='tadf')
    run = run_in(tmp_path, d, 1000)
    with pytest.raises(StageRefused):
        vib_stage(run)
