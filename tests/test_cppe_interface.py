"""`polarizable_sites_from_potfile`: a `PolarizableSites` built by cppe.

Undamped, cppe's own dipole-dipole coupling and wicks's hand-rolled
`dipole_interaction_matrix` agree to machine precision (checked directly
against the classes here), so that is the case that pins the interface
against the environment it wraps rather than against a second, independent
implementation of the same physics. Damped, the two conventions are not
bit-identical -- a different but equally standard choice of Thole formula,
not a bug -- so no test below asks them to agree there.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df, gto

cppe = pytest.importorskip('cppe', reason='pip install cppe')

from src.Base.environment import Environment
from src.Base.cppe_interface import polarizable_sites_from_potfile
from src.Base.polarizable_sites import PolarizableSites

ATOM = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Bohr: outside the QM density, where two isotropic sites are built both by
#: hand and through a potential file, undamped, for the exact-match test.
SITES_BOHR = np.array([[0.0, 0.0, 9.0], [5.0, 2.0, -6.0]])
ALPHAS = np.array([5.0, 3.0])
#: Bohr, 3.30 from the nearest nucleus: inside MIN_SITE_TO_QM_DISTANCE.
CLOSE_SITE_BOHR = np.array([[0.0, 0.0, 3.0]])


def molecules():
    mol = gto.M(atom=ATOM, basis='sto-3g', verbose=0)
    return mol, df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')


def _potfile(path, coords, alphas, exclusions=None):
    """A minimal potential file: `alphas` isotropic, zero permanent charge.

    `exclusions[k]` is the (possibly empty) list of 1-based partner indices
    excluded from site k's coupling; omitted entirely for no exclusions at
    all, which is the shape every hand-built `PolarizableSites` site list has.
    """
    n = len(coords)
    lines = ['! synthetic potential file for the cppe interface tests',
             '@COORDINATES', str(n), 'AU']
    for k, (x, y, z) in enumerate(coords, start=1):
        lines.append(f'X {x:.10f} {y:.10f} {z:.10f} {k}')
    lines += ['@MULTIPOLES', 'ORDER 0', str(n)]
    lines += [f'{k} 0.0' for k in range(1, n + 1)]
    lines += ['@POLARIZABILITIES', 'ORDER 1 1', str(n)]
    lines += [f'{k} {a:.10f} 0.0 0.0 {a:.10f} 0.0 {a:.10f}'
              for k, a in enumerate(alphas, start=1)]
    if exclusions is None:
        exclusions = [[] for _ in range(n)]
    width = max((len(e) for e in exclusions), default=0)
    lines.append(f'EXCLISTS\n{n} {width}')
    lines += [f'{k} ' + ' '.join(str(e) for e in ex)
             for k, ex in enumerate(exclusions, start=1)]
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.fixture
def undamped_env(tmp_path):
    potfile = _potfile(tmp_path / 'sites.pot', SITES_BOHR, ALPHAS)
    return polarizable_sites_from_potfile(potfile, damp_induced=False)


def test_it_satisfies_the_environment_protocol(undamped_env):
    assert isinstance(undamped_env, Environment)
    for name in ('for_geometry', 'mean_field', 'aux_kernel',
                 'static_self_energy', 'aux_kernel_adjoint',
                 'static_self_energy_adjoint'):
        assert callable(getattr(undamped_env, name)), name
    assert undamped_env.differentiable is True


def test_undamped_matches_the_hand_built_environment_exactly(undamped_env):
    """No exclusions, no damping: cppe's B and wicks's own are the same matrix,
    so the folded kernel is too -- this is what pins the interface's field
    integrals, coordinate units, and B convention against the object it wraps."""
    _, auxmol = molecules()
    hand_built = PolarizableSites(SITES_BOHR, ALPHAS, unit='Bohr')
    assert np.abs(undamped_env.B - hand_built.B).max() < 1e-12
    assert np.abs(undamped_env.aux_kernel(auxmol)
                 - hand_built.aux_kernel(auxmol)).max() < 1e-10


def test_exclusions_zero_the_pairwise_coupling(tmp_path):
    """Two mutually excluded sites respond as if alone: B is block-diagonal,
    each block exactly alpha (cppe's own exclusion list, not a distance cut)."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    potfile = _potfile(tmp_path / 'excluded.pot', coords, np.array([5.0, 5.0]),
                       exclusions=[[2], [1]])
    env = polarizable_sites_from_potfile(potfile, damp_induced=True)
    assert np.abs(env.B - 5.0 * np.eye(6)).max() < 1e-10


def test_a_site_the_density_reaches_is_refused(tmp_path):
    mol, _ = molecules()
    potfile = _potfile(tmp_path / 'close.pot', CLOSE_SITE_BOHR, np.array([5.0]))
    with pytest.raises(ValueError, match='non-overlap|Bohr from QM centre'):
        polarizable_sites_from_potfile(potfile, mol=mol)


def test_a_potfile_with_no_polarizable_site_is_refused(tmp_path):
    """A pure point-charge potential file: `PointCharges` is the environment
    for that, not this constructor, and it says so rather than building an
    empty B."""
    lines = ['@COORDINATES', '1', 'AU', 'X 0.0 0.0 0.0 1',
             '@MULTIPOLES', 'ORDER 0', '1', '1 -1.0']
    potfile = tmp_path / 'charge_only.pot'
    potfile.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='no polarizable site'):
        polarizable_sites_from_potfile(potfile)


def test_anisotropic_polarizability_gives_a_positive_definite_response(tmp_path):
    """cppe accepts a genuinely anisotropic tensor -- the one thing
    `PolarizableSites`'s own isotropic-alpha constructor cannot build -- and
    the coupled response of one such site alone is still SPD."""
    n = 1
    lines = ['@COORDINATES', str(n), 'AU', 'X 0.0 0.0 0.0 1',
             '@MULTIPOLES', 'ORDER 0', str(n), '1 0.0',
             '@POLARIZABILITIES', 'ORDER 1 1', str(n),
             '1 6.0 1.0 0.5 4.0 -0.5 3.0',
             f'EXCLISTS\n{n} 0', '1']
    potfile = tmp_path / 'aniso.pot'
    potfile.write_text('\n'.join(lines) + '\n')
    env = polarizable_sites_from_potfile(potfile)
    assert np.linalg.eigvalsh(env.B).min() > 0.0
    assert not np.allclose(env.B, env.B[0, 0] * np.eye(3))
