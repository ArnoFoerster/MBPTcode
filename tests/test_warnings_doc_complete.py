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


# warnings.warn calls in src that are not numerical shortfalls: the number a
# caller gets is the same (a route or pool choice, a provenance stamp), so
# there is no flag to record.
NOT_NUMERICAL = {
    'src/Base/numerical_flags.py': 1,                    # `flag` itself
    'src/Base/utils/linear_algebra/diagonalization.py': 3,   # GPU / ELPA fall back to the host solve
    'src/Base/utils/parallel/threads.py': 1,             # pool capped at the CPUs the thread may use
    'src/Base/utils/record_io.py': 1,                    # the commit is stamped 'unknown' in the record
    # per element, silenced under `register`, which flags 'ri_tier_short';
    # and max_error picking the tightest tiers, the max_error=None set
    'src/Base/basis/cp2k_basis.py': 2,
}

# plain numerical warnings still to become named flags, by file: a count that
# may only go down (a file that drops to zero leaves the list)
PLAIN_WARNING_DEBT = {
    'src/SingleReference/ADC/eeADC/ee_dbse2_ri_fold.py': 1,
    'src/SingleReference/ADC/eeADC/ee_fold.py': 2,
    'src/SingleReference/ADC/eeADC/vertical.py': 1,
    'src/SingleReference/ADC/sigma_driver.py': 2,
    'src/gradients/gwbse/adaptive_selection.py': 1,
    'src/gradients/gwbse/excited_state.py': 1,
    'src/gradients/gwbse/qp_set.py': 5,
    'src/gradients/gwbse/state_following.py': 2,
    'src/gradients/isdf_derivatives.py': 1,
    'src/gradients/reference_mean_field.py': 1,
    'src/gradients/rpa/factor_chain.py': 1,
    'src/gradients/state_identity.py': 2,
    'src/gradients/state_models.py': 1,
    'src/photophysics/chain.py': 1,
    'src/photophysics/rate_chain.py': 1,
    'src/photophysics/stages.py': 2,
    'src/photophysics/state_specific.py': 1,
    'src/properties/characters.py': 1,
    'src/properties/duschinsky.py': 1,
    'src/properties/excitations.py': 3,
    'src/properties/franck_condon.py': 2,
    'src/properties/herzberg_teller.py': 1,
    'src/properties/hessian.py': 1,
    'src/properties/mode_hessian.py': 1,
    'src/properties/optimize.py': 2,
}


def plain_warnings():
    """{source file: number of warnings.warn calls} in src."""
    found = {}
    for d, _, files in os.walk(os.path.join(REPO, 'src')):
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(d, f)
            with open(path) as fh:
                text = fh.read()
            if 'warnings' not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'warn'
                        and _name_of(node.func.value) == 'warnings'):
                    where = os.path.relpath(path, REPO)
                    found[where] = found.get(where, 0) + 1
    return found


def test_every_numerical_warning_is_a_named_flag():
    """A numerical shortfall warns through `numerical_flags.flag`, so its
    name reaches the record; a plain warnings.warn is either not numerical
    (NOT_NUMERICAL) or a debt that only shrinks (PLAIN_WARNING_DEBT)."""
    allowed = dict(PLAIN_WARNING_DEBT)
    for k, v in NOT_NUMERICAL.items():
        allowed[k] = allowed.get(k, 0) + v
    found = plain_warnings()
    over = {k: (v, allowed.get(k, 0)) for k, v in found.items()
            if v > allowed.get(k, 0)}
    assert not over, (
        'plain warnings.warn calls beyond the lists (file: (found, allowed)); '
        f'raise a named flag with numerical_flags.flag instead: {over}')
    stale = {k: (found.get(k, 0), v) for k, v in allowed.items()
             if found.get(k, 0) < v}
    assert not stale, (
        'lower these counts in PLAIN_WARNING_DEBT / NOT_NUMERICAL, the '
        f'warnings are gone (file: (found, listed)): {stale}')
