"""Every test file runs as a script and under pytest (tests/README.md); this
file holds pytest to its half of that.

A script-style check named test_* is the failure this guards against: one that
takes the molecule or mean field its own __main__ builds errors at collection
("fixture 'mf' not found"), and one that returns its verdict instead of
asserting it passes under pytest whatever the verdict is. Such checks are
named check_*, the __main__ body is run(), and one argument-free test_* asserts
run().

Checked by parsing, without importing any test module: every top-level test_*
function, and every test_* method of a Test* class, takes only fixtures pytest
can supply (its built-ins, fixtures defined in the module or in conftest.py,
parametrized names, imported names) and returns nothing.

Run: python tests/test_conventions.py, or under pytest.
"""
import ast
import glob
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))

BUILTIN_FIXTURES = {'cache', 'capfd', 'capfdbinary', 'caplog', 'capsys',
                    'capsysbinary', 'doctest_namespace', 'monkeypatch',
                    'pytestconfig', 'record_property', 'record_testsuite_property',
                    'record_xml_attribute', 'recwarn', 'request', 'subtests',
                    'tmp_path', 'tmp_path_factory', 'tmpdir', 'tmpdir_factory'}


def _is_fixture(node):
    """A function decorated with pytest.fixture, bare or called."""
    return any('fixture' in ast.unparse(d) for d in node.decorator_list)


def _parametrized_names(decorators, aliases):
    """Argument names a parametrize decorator (direct, or a module-level alias
    of one) supplies."""
    names = set()
    for d in decorators:
        call = d if isinstance(d, ast.Call) else None
        if isinstance(d, ast.Name) and d.id in aliases:
            call = aliases[d.id]
        if call is None or 'parametrize' not in ast.unparse(call.func):
            continue
        if call.args and isinstance(call.args[0], ast.Constant):
            names |= {s.strip() for s in str(call.args[0].value).split(',')}
    return names


def _returns_a_value(fn):
    """True if fn itself (not a function nested in it) returns something."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                             ast.ClassDef)):
            continue
        if (isinstance(node, ast.Return) and node.value is not None
                and not (isinstance(node.value, ast.Constant)
                         and node.value.value is None)):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _conftest_fixtures():
    names = set()
    for path in glob.glob(os.path.join(TESTS, '**', 'conftest.py'), recursive=True):
        tree = ast.parse(open(path).read())
        names |= {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and _is_fixture(n)}
    return names


def violations(path, shared_fixtures):
    """(name, problem) for every test the file defines that pytest cannot run
    as written."""
    tree = ast.parse(open(path).read())
    fixtures = {n.name for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and _is_fixture(n)}
    imported = {a.asname or a.name.split('.')[0] for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    aliases = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and 'parametrize' in ast.unparse(n.value.func)
               for t in n.targets if isinstance(t, ast.Name)}
    known = BUILTIN_FIXTURES | shared_fixtures | fixtures | imported

    tests = [(n, None) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith('test')]
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name.startswith('Test'):
            tests += [(n, cls) for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name.startswith('test')]
    found = []
    for fn, cls in tests:
        label = fn.name if cls is None else f'{cls.name}.{fn.name}'
        args = [a.arg for a in fn.args.args]
        if cls is not None and args and args[0] in ('self', 'cls'):
            args = args[1:]
        supplied = known | _parametrized_names(fn.decorator_list, aliases)
        if cls is not None:
            supplied |= _parametrized_names(cls.decorator_list, aliases)
        missing = [a for a in args if a not in supplied]
        if missing:
            found.append((label, f'takes {", ".join(missing)}, which no fixture supplies'))
        if _returns_a_value(fn):
            found.append((label, 'returns a value instead of asserting'))
    return found


def run():
    shared = _conftest_fixtures()
    files = sorted(glob.glob(os.path.join(TESTS, '**', 'test_*.py'), recursive=True))
    bad = {path: v for path in files if (v := violations(path, shared))}
    for path, found in bad.items():
        for name, problem in found:
            print(f'  {os.path.relpath(path, TESTS)}::{name} {problem}')
    print(f'{len(files)} test files, {sum(map(len, bad.values()))} pytest-incompatible '
          f'test_* functions in {len(bad)} files: {"FAIL" if bad else "OK"}')
    return not bad


def test_every_test_function_runs_under_pytest():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
