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
RADII = pathlib.Path('src') / 'Base' / 'data' / 'optimized_radii.json'


def install_current_radii(tree):
    """Overwrite the radii table of the extracted tree `tree` with this
    checkout's."""
    target = pathlib.Path(tree) / RADII
    assert target.parent.is_dir(), f'{tree} is not an extracted src tree'
    shutil.copy(REPO / RADII, target)
