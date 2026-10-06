"""The xc skeleton on the Becke grid that moves with the atoms
(`skeleton_tiles.xc_grid_skeleton`, reached as `isdf_derivatives.xc_skeleton`).

T(R) = Tr[g v_xc^DFT[D](R)] with the AO matrices g and D held is what every
Kohn-Sham force's xc skeleton differentiates (the relaxed-density Fock term
and the Sigma_x - v_xc correction of an excitation chain), and the energy
evaluates it on a grid rebuilt at every geometry: its points ride with their
atoms and its Becke weights move with all of them. On water, ethylene and
formaldehyde/cc-pVDZ (the last two with a seeded 0.05 Bohr distortion), PBE0
and LRC-wPBEh, density-fitted and ISDF-K references, at the default grid:

  (a) the skeleton against a 4-point difference of T on the moving grid, to
      `ISDF_GRADIENT_FLOOR` (6e-10 to 2.9e-9 with a random g of O(1)
      entries, the difference's own floor); it sums to zero over atoms to
      1e-13;
  (b) pyscf's fixed-grid derivative (`hessian.rks._get_vxc_deriv1`) misses
      the same difference by 2.5e-5 to 5e-4 while it meets a difference on
      the held grid, so the gate sees the grid's motion;
  (c) planted, the weights' response dropped fails (a);
  (d) the Becke weights' response is pyscf's `grids_response_cc`, contracted,
      to 1e-13 relative;
  (e) a meta-GGA (TPSS) and an LDA follow their own differences;
  (f) over 2, 3 and 8 simulated ranks every rank holds the one-rank
      skeleton bitwise (the tiles' addends are gathered verbatim and summed
      in tile order); planted, a rank dropping one of its tiles moves it;
  (g) a VV10 functional is refused, since its kernel is not differentiated.
"""
import pathlib
import sys

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.grad import rks as grad_rks
from pyscf.hessian import rks as hessian_rks

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.Base import skeleton_tiles                                  # noqa: E402
from src.Base.constants import ISDF_GRADIENT_FLOOR                   # noqa: E402
from src.Base.isdf_jk import isdf_jk                                 # noqa: E402
from src.Base.utils.mpi_grid import current_comm, run_simulated      # noqa: E402
from src.gradients.isdf_derivatives import xc_skeleton               # noqa: E402

#: The test molecules' geometries, Angstrom.
SYSTEMS = {
    'water': 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692',
    'formaldehyde': ('C 0.0 0.0 -0.5296; O 0.0 0.0 0.6742; '
                     'H 0.0 0.9429 -1.1123; H 0.0 -0.9429 -1.1123'),
    'ethylene': ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
                 'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; '
                 'H 0.0 -0.923 -1.238'),
}

#: The seeded distortion of ethylene and formaldehyde, Bohr.
DISTORTION = 0.05
DISTORTION_SEED = 7
#: The 4-point difference's step, Bohr.
STEP = 1e-4
MOLECULES = ('water', 'ethylene', 'formaldehyde')
XCS = ('pbe0', 'lrc-wpbeh')
REFERENCES = ('df', 'isdf-k')
#: What the fixed-grid derivative misses the moving grid by at least.
GRID_MOTION = 1e-6
#: The skeleton's own translation residual.
TRANSLATION_BAR = 1e-12
SIZES = (2, 3, 8)
#: {(molecule, xc, reference): the converged mean field}
MEAN_FIELDS = {}


def molecule(name):
    mol = gto.M(atom=SYSTEMS[name], basis='cc-pvdz', verbose=0)
    if name == 'water':
        return mol
    rng = np.random.default_rng(DISTORTION_SEED)
    shift = rng.uniform(-DISTORTION, DISTORTION, (mol.natm, 3))
    return mol.set_geom_(mol.atom_coords() + shift, unit='Bohr', inplace=False)


def mean_field(name, xc, reference='df'):
    key = (name, xc, reference)
    if key not in MEAN_FIELDS:
        mol = molecule(name)
        mf = (isdf_jk(dft.RKS(mol, xc=xc), auxbasis='cc-pvdz-ri')
              if reference == 'isdf-k' else dft.RKS(mol, xc=xc).density_fit())
        mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-8
        mf.kernel()
        assert mf.converged
        MEAN_FIELDS[key] = mf
    return MEAN_FIELDS[key]


def partial(mf, seed=3):
    """A random symmetric MO partial with O(1) entries: every AO pair on."""
    n = mf.mo_coeff.shape[1]
    r = np.random.default_rng(seed).standard_normal((n, n))
    return r + r.T


def at(mol, atom, axis, step):
    crd = mol.atom_coords().copy()
    crd[atom, axis] += step
    return mol.set_geom_(crd, unit='Bohr', inplace=False)


def grid_of(mf, mol, moving):
    """The mean field's grid rebuilt on `mol`, or its points and weights held."""
    ref = mf.grids
    grids = dft.gen_grid.Grids(mol)
    if moving:
        grids.level, grids.prune = ref.level, ref.prune
        grids.atom_grid, grids.radi_method = ref.atom_grid, ref.radi_method
        return grids.build()
    grids.coords, grids.weights = ref.coords, ref.weights
    return grids


def trace_difference(mf, g_ao, moving):
    """4-point difference of Tr[g_ao v_xc^DFT[D]] over every atom and axis."""
    D = mf.make_rdm1()
    mol = mf.mol
    out = np.zeros((mol.natm, 3))
    for atom in range(mol.natm):
        for axis in range(3):
            t = {}
            for k in (2, 1, -1, -2):
                m = at(mol, atom, axis, k * STEP)
                v = mf._numint.nr_rks(m, grid_of(mf, m, moving), mf.xc, D)[2]
                t[k] = float(np.einsum('ij,ij->', g_ao, v))
            out[atom, axis] = (8.0 * (t[1] - t[-1])
                               - (t[2] - t[-2])) / (12.0 * STEP)
    return out


def g_ao_of(mf, gamma):
    C = mf.mo_coeff
    return C @ (0.5 * (gamma + gamma.T)) @ C.T


CASES = [(n, x, r) for n in MOLECULES for x in XCS for r in REFERENCES]


# ------------------------------------ (a), (b) the moving grid, and it matters
@pytest.mark.parametrize('case', CASES, ids=['-'.join(c) for c in CASES])
def test_the_skeleton_follows_the_moving_grid(case):
    mf = mean_field(*case)
    gamma = partial(mf)
    g_ao = g_ao_of(mf, gamma)
    got = xc_skeleton(mf, gamma)
    moving = trace_difference(mf, g_ao, moving=True)
    miss = float(np.abs(got - moving).max())
    translation = float(np.abs(got.sum(axis=0)).max())
    fixed = np.einsum('axij,ij->ax', np.asarray(hessian_rks._get_vxc_deriv1(
        mf.Hessian(), mf.mo_coeff, mf.mo_occ, mf.max_memory)), g_ao)
    fixed_miss = float(np.abs(fixed - moving).max())
    print(f'\n{case}: skeleton - FD {miss:.2e}, translation '
          f'{translation:.1e}; fixed-grid derivative - FD {fixed_miss:.2e}')
    assert miss < ISDF_GRADIENT_FLOOR, miss
    assert translation < TRANSLATION_BAR, translation
    assert fixed_miss > GRID_MOTION, fixed_miss


def test_the_fixed_grid_derivative_meets_the_held_grid():
    """(b)'s other half: the fixed-grid derivative meets a difference on the
    held grid."""
    mf = mean_field('formaldehyde', 'lrc-wpbeh')
    g_ao = g_ao_of(mf, partial(mf))
    fixed = np.einsum('axij,ij->ax', np.asarray(hessian_rks._get_vxc_deriv1(
        mf.Hessian(), mf.mo_coeff, mf.mo_occ, mf.max_memory)), g_ao)
    held = trace_difference(mf, g_ao, moving=False)
    assert np.abs(fixed - held).max() < ISDF_GRADIENT_FLOOR


# ------------------------------------------------ (c) the gate can fail
def test_without_the_weights_response_the_gate_fails(monkeypatch):
    mf = mean_field('water', 'lrc-wpbeh')
    gamma = partial(mf)
    moving = trace_difference(mf, g_ao_of(mf, gamma), moving=True)
    monkeypatch.setattr(skeleton_tiles, 'becke_weight_response',
                        lambda mol, *a: np.zeros((mol.natm, 3)))
    miss = float(np.abs(xc_skeleton(mf, gamma) - moving).max())
    print(f'\nweights held: skeleton - FD {miss:.2e}')
    assert miss > GRID_MOTION, miss


# --------------------------------------- (d) the Becke weights are pyscf's
@pytest.mark.parametrize('name', ('water', 'formaldehyde'))
def test_the_weights_response_is_pyscfs(name):
    mol = molecule(name)
    grids = dft.gen_grid.Grids(mol)
    grids.alignment = 0
    grids.build(sort_grids=False)
    phi = np.random.default_rng(5).standard_normal(grids.weights.size)
    got = skeleton_tiles.becke_weight_response(
        mol, grids.coords, grids.atm_idx, grids.weights * phi,
        skeleton_tiles.becke_adjustment(mol, grids))
    ref = np.zeros((mol.natm, 3))
    p0 = 0
    for coords, w0, w1 in grad_rks.grids_response_cc(grids):
        p1 = p0 + len(w0)
        assert np.array_equal(coords, grids.coords[p0:p1])
        ref += np.einsum('axk,k->ax', w1, phi[p0:p1])
        p0 = p1
    assert np.abs(got - ref).max() < 1e-13 * np.abs(ref).max()


# ------------------------------------------ (e) a meta-GGA and an LDA
@pytest.mark.parametrize('xc', ('tpss', 'lda'))
def test_other_rungs_follow_the_moving_grid(xc):
    mf = mean_field('water', xc)
    gamma = partial(mf)
    moving = trace_difference(mf, g_ao_of(mf, gamma), moving=True)
    got = xc_skeleton(mf, gamma)
    assert np.abs(got - moving).max() < ISDF_GRADIENT_FLOOR
    assert np.abs(got.sum(axis=0)).max() < TRANSLATION_BAR


# ------------------------------------------------- (f) over the ranks
def over_ranks(size, plant=False):
    mf = mean_field('formaldehyde', 'pbe0')
    gamma = partial(mf)
    original = skeleton_tiles._xc_tile_addend

    def dropped(mol, *args):
        out = original(mol, *args)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1 and not dropped.done:
            dropped.done = True
            return 0.0 * out
        return out

    dropped.done = False
    if plant:
        skeleton_tiles._xc_tile_addend = dropped
    try:
        return run_simulated(lambda comm: xc_skeleton(mf, gamma), size)
    finally:
        skeleton_tiles._xc_tile_addend = original


@pytest.mark.parametrize('size', SIZES)
def test_every_rank_holds_the_one_rank_skeleton(size):
    mf = mean_field('formaldehyde', 'pbe0')
    one = xc_skeleton(mf, partial(mf))
    for r, got in enumerate(over_ranks(size)):
        assert np.array_equal(got, one), (size, r)


def test_a_dropped_tile_moves_it():
    mf = mean_field('formaldehyde', 'pbe0')
    one = xc_skeleton(mf, partial(mf))
    moved = [float(np.abs(g - one).max()) for g in over_ranks(3, plant=True)]
    assert min(moved) > ISDF_GRADIENT_FLOOR, moved


# ------------------------------------------------------ (g) refused
def test_a_vv10_functional_is_refused():
    mf = dft.RKS(molecule('water'), xc='wb97m_v')
    with pytest.raises(NotImplementedError, match='VV10'):
        xc_skeleton(mf, np.eye(mf.mol.nao_nr()))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
