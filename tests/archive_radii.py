"""The shipped ISDF radii table, for tests that run an archived tree.

A test that extracts an archived commit with `git archive` and compares its
numbers with the current code must hand the archive the current
`optimized_radii.json`: the archived code reads the table it ships, and the
table differs between commits, so left alone the comparison is of two grids
and not of two codes.
"""
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
RADII = pathlib.Path('src') / 'Base' / 'factorization' / 'data' / 'optimized_radii.json'
#: Where an extracted tree keeps its table: as above in current trees,
#: under src/Base/data.
ARCHIVED_RADII = (RADII, pathlib.Path('src') / 'Base' / 'data' / 'optimized_radii.json')


def install_current_radii(tree):
    """Overwrite the radii table of the extracted tree `tree` with this
    checkout's."""
    target = next((pathlib.Path(tree) / p for p in ARCHIVED_RADII
                   if (pathlib.Path(tree) / p).parent.is_dir()), None)
    assert target is not None, f'{tree} is not an extracted src tree'
    shutil.copy(REPO / RADII, target)
