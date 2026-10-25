"""An explicit eps_static beside a solvent name survives every rebuild.

WHAT BREAKS IF THIS FAILS. A `SolventScreening._rebuilt` that handed a named
solvent on as its name alone would replace an eps_static given beside it by
the table's at every displaced geometry (`for_geometry`) and in both partners.
The reference geometry would then stand in one continuum and every other
geometry in another: a geometry-dependent step in the surface, in solvent
only. A lost eps_static reads the table's 78.36 for water in every check below.
"""
import pytest
from pyscf import gto

from src.environments.solvent_screening import SolventScreening, solvent_dielectrics

#: A static constant that is not water's table value.
EPS_STATIC = 50.0


def test_eps_static_travels_with_a_solvent_name():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='sto-3g', verbose=0)
    optical, table = solvent_dielectrics('water')
    assert table != pytest.approx(EPS_STATIC)
    env = SolventScreening(mol, solvent='water', eps_static=EPS_STATIC)
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() + 0.01, unit='Bohr')
    displaced = env.for_geometry(moved)
    assert displaced.eps_static == EPS_STATIC
    assert displaced.eps == pytest.approx(optical)
    partner = env.equilibrium_partner()
    assert partner.eps_static == EPS_STATIC and partner.eps == EPS_STATIC
    back = partner.non_equilibrium_partner()
    assert back.eps_static == EPS_STATIC
    assert partner.for_geometry(moved).eps == EPS_STATIC
