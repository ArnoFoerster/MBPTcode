"""Every test that runs an archived tree hands it today's radii table.

`git archive` of an old commit carries the radii table that commit shipped;
an older table names other grids, so an archive-based comparison
that does not copy today's file in compares two grids. A test file that
extracts an archive must call `install_current_radii` on every tree it
extracts, or be listed in EXEMPT with the reason.
"""
import pathlib
import re

from tests.archive_radii import install_current_radii

TESTS = pathlib.Path(__file__).resolve().parent
#: file -> why the archive needs no current table
EXEMPT = {}


def _extracting_files():
    for path in sorted(TESTS.glob('test_*.py')):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        if "'archive', '--format=tar'" in text:
            yield path, text


def test_every_archive_extraction_installs_the_current_radii():
    bare = []
    for path, text in _extracting_files():
        extracted = len(re.findall(r'\.extractall\(', text))
        installed = len(re.findall(r'^\s+install_current_radii\(', text, re.M))
        if installed < extracted and path.name not in EXEMPT:
            bare.append(f'{path.name}: {extracted} extractions, '
                        f'{installed} installs')
    assert not bare, bare
