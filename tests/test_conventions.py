"""Test files run under pytest or as scripts (tests/README.md); this file holds
the test_* functions to what pytest needs.

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

Every tests/test_*.py is in exactly one CI tier list (tools/ci/fast.txt,
slow.txt, heavy.txt, triage.txt), and every line of those lists names a file
that exists: a file in no list never runs in CI.

Run: python tests/test_conventions.py, or under pytest.
"""
import ast
import glob
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
CI = os.path.join(os.path.dirname(TESTS), 'tools', 'ci')
CI_TIERS = ('fast', 'slow', 'heavy', 'triage')

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


def tier_placement():
    """{test file name: [tiers whose list names it]} over every tests/test_*.py
    and every file a tier list names."""
    placed = {os.path.basename(p): []
              for p in glob.glob(os.path.join(TESTS, 'test_*.py'))}
    for tier in CI_TIERS:
        with open(os.path.join(CI, tier + '.txt')) as f:
            for line in f:
                line = line.split('#')[0].strip()
                if line:
                    placed.setdefault(line.split()[1], []).append(tier)
    return placed


def tier_problems():
    """(file, problem) for a test file in no tier or in several, and for a tier
    line naming a file that does not exist."""
    problems = []
    for name, tiers in sorted(tier_placement().items()):
        if not os.path.exists(os.path.join(TESTS, name)):
            problems.append((name, f'listed in {", ".join(tiers)} but does not exist'))
        elif len(tiers) != 1:
            problems.append((name, f'in {len(tiers)} tier lists {tiers}, not exactly one'))
    return problems


def pytest_violations():
    """(number of test files, {path: violations}) over every tests/**/test_*.py."""
    shared = _conftest_fixtures()
    files = sorted(glob.glob(os.path.join(TESTS, '**', 'test_*.py'), recursive=True))
    return len(files), {path: v for path in files if (v := violations(path, shared))}


def run():
    problems = tier_problems()
    for name, problem in problems:
        print(f'  {name}: {problem}')
    print(f'CI tier lists: {len(problems)} problems: {"FAIL" if problems else "OK"}')
    nfiles, bad = pytest_violations()
    for path, found in bad.items():
        for name, problem in found:
            print(f'  {os.path.relpath(path, TESTS)}::{name} {problem}')
    print(f'{nfiles} test files, {sum(map(len, bad.values()))} pytest-incompatible '
          f'test_* functions in {len(bad)} files: {"FAIL" if bad else "OK"}')
    return not bad and not problems


def test_every_test_function_runs_under_pytest():
    assert pytest_violations()[1] == {}


def test_every_test_file_is_in_exactly_one_tier():
    assert tier_problems() == []


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
