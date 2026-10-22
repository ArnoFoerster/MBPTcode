"""The excited-state chain reads every setting it holds a choice for from ONE
`Numerics`.

The kernels read flat names (`nroots`, `residue_route`, ...); each is a
view of the chain's `Numerics` (`chain.settings`), the object it is
handed, replaced on a write, so the object the products will be stamped
with and the values the kernels used cannot differ.

H2/cc-pVDZ Hartree-Fock on the G1 grid; nothing is solved.
"""
import pytest
from pyscf import gto

from src.Base.numerics import Numerics
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.excited_state import ExcitedStateChain
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
#: flat name -> its choice in `Numerics`
READS = {'ntau_gw': 'quadrature.ntau_gw',
         'e_min_below_gap': 'quadrature.e_min_below_gap',
         'n_poles': 'qp.n_poles', 'sop_stride': 'qp.sop_stride',
         'nroots': 'bse.nroots', 'bse_conv_tol': 'bse.conv_tol',
         'probe': 'bse.probe', 'degeneracy_tol': 'qp.degeneracy_tol',
         'tile_gb': 'layout.tile_gb', 'qp_window': 'qp.window',
         'residue_route': 'qp.residue_route', 'scissor': 'qp.scissor',
         'outside_scissor': 'qp.outside_scissor',
         'qp_hamiltonian': 'qp.hamiltonian', 'solver': 'bse.solver',
         'bse_adjoint': 'bse.adjoint', 'track': 'bse.track'}


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture
def chain(h2):
    counts = resolve_isdf_grid('G1', BASIS, ['H'])[0]
    return ExcitedStateChain(
        h2, rhf_df_differentiable_checked, numerics=Numerics(isdf={'counts': counts},
                                   qp={'residue_route': 'sop'},
                                   quadrature={'ntau_w': 14},
                                   bse={'track': 'overlap', 'nroots': 6}))


def test_the_chain_holds_its_numerics(chain):
    n = chain.settings
    assert isinstance(n, Numerics)
    assert n.bse.nroots == 6 and n.quadrature.ntau_w == 14
    assert n.qp.residue_route == 'sop' and n.bse.track == 'overlap'


@pytest.mark.parametrize('name', sorted(READS))
def test_every_flat_setting_reads_the_numerics(chain, name):
    """Replace the held object by one whose choice differs: the flat name
    follows the object, not a copy."""
    section, key = READS[name].split('.')
    n = chain.settings
    other = {'qp.window': 3, 'qp.residue_route': 'laplace',
             'qp.scissor': 'calibrate', 'qp.outside_scissor': 'curve',
             'qp.hamiltonian': True, 'bse.solver': 'dense',
             'bse.adjoint': 'grid', 'bse.track': None, 'bse.probe': 'sign',
             'layout.tile_gb': 1.5, 'quadrature.e_min_below_gap': 0.5,
             'qp.sop_stride': 4}.get(READS[name], 'unset')
    if other == 'unset':
        other = getattr(getattr(n, section), key) * 2
    derived = n.derive('test', **{section: {key: other}})
    chain.__dict__['settings'] = derived
    assert getattr(chain, name) == other


def test_a_write_rebuilds_the_numerics(chain):
    before = chain.settings.identity()
    chain.nroots = 9
    assert chain.settings.bse.nroots == 9
    assert chain.settings.identity() != before
    chain.nroots = 6
    assert chain.settings.identity() == before


def test_a_refreeze_keeps_the_numerics(h2, chain):
    """Only the geometry changes: the numerics is the object that was asked
    for, and the resolved quadrature sizes live in the conventions."""
    fresh = chain.refreeze(h2)
    assert fresh.settings.values() == chain.settings.values()
    assert fresh.settings.identity() == chain.settings.identity()
    assert fresh.conventions.nfreq_cd >= chain.conventions.nfreq_cd
