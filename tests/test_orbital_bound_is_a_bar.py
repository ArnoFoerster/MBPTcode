"""ISDF_ORBITAL_RESIDUAL_TOL is a bar the code judges by, not a choice.

Separable-RI factors that miss an orbital density by more than the bar are
refused (`refuse_unrepresented_factors`). A factor builder or chain taking
`orbital_tol` as a keyword, None skipping the measurement, or a record or spec
carrying it would let a run loosen the bar by name. No function in src takes
the bound, no record or spec holds it, and changing it
takes a commit. A tool or a test that measures factors the bar refuses
builds them with `space_time.unrefused_separable_factors`.
"""
import ast
import pathlib

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.surface_factory import GRID_READS

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_no_src_signature_takes_the_bound():
    takers = []
    for path in sorted((REPO / 'src').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                names = {a.arg for a in args.args + args.kwonlyargs
                         + args.posonlyargs}
                if 'orbital_tol' in names:
                    takers.append(f'{path.relative_to(REPO)}:{node.name}')
    assert not takers, takers


def test_no_record_or_spec_holds_the_bound():
    assert not hasattr(Numerics(), 'orbital_tol')
    assert not any(part.endswith('orbital_tol') for part in GRID_READS)
