"""Water geometries the tests share, Angstrom, as pyscf `atom` strings.

Each is one geometry many test files use digit for digit, so a test that
pins a number on one keeps it by naming the geometry here. A test that needs
its own digits (a pin recorded on another geometry, a deliberately distorted
or displaced molecule) keeps its own string.
"""

#: pyscf's example water, C2v.
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: The same to three decimals, C2v.
WATER_ROUNDED = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Three decimals with the hydrogens 1 mA lower, C2v.
WATER_ROUNDED_LOW = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
#: Oxygen at the origin, C2v.
WATER_AT_ORIGIN = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
#: Oxygen at the origin, two decimals, C2v.
WATER_AT_ORIGIN_COARSE = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'
#: Both OH bonds 0.96 A at a right angle.
WATER_RIGHT_ANGLE = 'O 0 0 0; H 0 0 0.96; H 0.96 0 0'
#: One OH along z and the other in the xz plane at 104 degrees: Cs.
WATER_CS = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: No symmetry: oxygen and one hydrogen off the plane.
WATER_C1 = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
#: No symmetry: pyscf's water with both hydrogens displaced.
WATER_DISTORTED = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
