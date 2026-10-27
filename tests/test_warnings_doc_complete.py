"""docs/warnings.md names every numerical flag and every warning class in src.

A flag is a name a solver raises when it stops short; a reader looks it up in
docs/warnings.md. The names are read from the source, not from a list, so a
new flag or warning class without a row fails here:

- the string argument (first or second) of a call to a flag function
  (`numerical_flags.flag`, `flag_numerics`, `ee_follow.flag`, a module's
  own `_flag`);
- the value of a 'flag' key in a dict literal ({'flag': name, ...});
- the string key stored into a mapping whose name contains 'flag'
  (flags['name'] = ...);
- every class in src that derives from a ...Warning class.

A name counts as documented when it appears in backticks in the page, alone or
quoted ('name').
"""
import ast
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NAME = re.compile(r'[a-z][a-z0-9]*(_[a-z0-9]+)+$')
FLAG_CALL = re.compile(r'_?flag(_numerics)?$')


def _name_of(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return ''


def _string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def raised_names():
    """{name: sorted source files} of the flags and warning classes in src."""
    found = {}

    def add(name, where):
        found.setdefault(name, set()).add(where)

    for d, _, files in os.walk(os.path.join(REPO, 'src')):
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(d, f)
            where = os.path.relpath(path, REPO)
            with open(path) as fh:
                text = fh.read()
            if 'flag' not in text and 'Warning' not in text:
                continue                  # the generated modules, mostly
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and FLAG_CALL.match(
                        _name_of(node.func)):
                    for arg in node.args[:2]:
                        s = _string(arg)
                        if s is not None and NAME.match(s):
                            add(s, where)
                elif isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values):
                        if _string(k) == 'flag' and _string(v) is not None:
                            add(_string(v), where)
                elif isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    for t in targets:
                        if (isinstance(t, ast.Subscript)
                                and 'flag' in _name_of(t.value).lower()):
                            s = _string(t.slice)
                            if s is not None and NAME.match(s):
                                add(s, where)
                elif isinstance(node, ast.ClassDef):
                    if any(_name_of(b).endswith('Warning') for b in node.bases):
                        add(node.name, where)
    return {k: sorted(v) for k, v in found.items()}


def documented():
    with open(os.path.join(REPO, 'docs', 'warnings.md')) as f:
        text = f.read()
    return {m.strip("'") for m in re.findall(r'`([^`]+)`', text)}


def test_scanner_sees_each_kind_of_raise():
    names = raised_names()
    for name in ('z_outside', 'orbital_response_not_converged',
                 'davidson_not_converged', 'CphfNotConverged',
                 'adc_roots_not_converged'):
        assert name in names, name


def test_every_flag_and_warning_class_has_a_row():
    names, docs = raised_names(), documented()
    missing = {k: v for k, v in names.items() if k not in docs}
    assert not missing, (
        'docs/warnings.md does not name these flags or warning classes '
        f'(name: files raising it): {missing}')
