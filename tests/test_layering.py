"""The physics layer carries no communicator: MPI is reached only through the
realization kernels.

Chains, surfaces and the optimizer in `src/gradients/` and `src/properties/`
are serial code that runs replicated on every rank inside
`with distributed(comm):`; a kernel finds the communicator through
`current_comm()`. Three rules, read off the source by AST so that no import
and no MPI is needed to check them, each over every module of both packages:

  * no function, method or lambda has a parameter named `comm` -- a physics
    object that takes one is a second way to reach the ranks, and the two
    disagree the moment a caller passes one comm and the context holds
    another;
  * no module imports a serve loop (`serve_*`), `broadcast` or a
    `replicate*` -- the root-driven protocol and the rank-0-decides-then-sends
    pattern these names implement are what the context takes the place of;
  * the only `mpi_grid` names a module uses are `lockstep`,
    `lockstep_mean_field` and `current_comm` -- lock what the layer decided
    itself, ask for the rank where a side effect needs it, nothing more.

The layer rules hold over every module of src/ and tools/
(docs/layout.md):

  * an import only points down the layers (`LAYERS`): physics -> properties
    -> gradients (method subpackages above the core) -> electronic structure
    -> environments -> foundation, never back; the generators are imported by
    no runtime module; a fresh interpreter that imports one layer holds
    nothing above it;
  * the method subpackages of the gradient layer never import each other
    (`SIBLINGS`), and nothing outside Periodic imports it (`ISOLATED`);
  * no private name of another module is read (`PRIVATE_EXCEPTIONS` names the
    few that are allowed for now).

The scanners are checked against planted sources below, so each rule is shown
to fail on the pattern it forbids and to pass on the one it allows.
"""
import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The physics layer: everything under these carries no communicator.
PACKAGES = ('src/gradients', 'src/properties')

#: Realization kernels that live under src/gradients: the reverse-mode sweeps
#: over imaginary time and frequency (`qp_set_gradient`, the space-time
#: adjoint, the reaction-field screening). They partition and reduce, so they
#: take `comm` and use the primitives. Their rule is the kernel rule: `comm`
#: defaults to None, the context decides, and no serve loop or broadcast
#: appears. Everything else under PACKAGES is the physics layer.
KERNEL_MODULES = frozenset({'src/gradients/gwbse/qp_space_time.py',
                            'src/gradients/rpa/space_time_adjoint.py',
                            'src/gradients/gwbse/reaction_field_adjoint.py'})

#: Distributed kernels and state models that live under src/gradients:
#: the tiled
#: skeleton and correlation forces and the state models of the solvation
#: loop, moved in from Base and SingleReference. They split their work over
#: the ranks themselves, so they may take `comm` and use any primitive of
#: mpi_grid, broadcast included; a serve loop or a replicate* stays
#: forbidden. The list is closed: a new module under PACKAGES that takes
#: `comm` or broadcasts fails until it is named here.
DISTRIBUTED_MODULES = frozenset({'src/gradients/skeleton_tiles.py',
                                 'src/gradients/df_correlation_force_tiled.py',
                                 'src/gradients/state_models.py',
                                 'src/gradients/adc/ee_gradient.py',
                                 'src/gradients/adc/ee_state_model.py'})

#: The module the distribution primitives live in.
MPI_GRID = 'src.Base.utils.parallel.mpi_grid'

#: What the physics layer may take from it.
ALLOWED_MPI_GRID = frozenset({'lockstep', 'lockstep_mean_field',
                              'current_comm'})

MODULES = sorted(path for package in PACKAGES
                 for path in (REPO / package).rglob('*.py'))


def module_id(path):
    """A module's path relative to the repository, for a test id."""
    return str(path.relative_to(REPO))


def comm_parameters_without_none_default(tree):
    """'name:line' of every `comm` parameter whose default is not None.

    The kernel rule: a kernel may take `comm`, but only as `comm=None`, so
    that a caller inside a region never has to know one exists.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda)):
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        pad = [None] * (len(positional) - len(args.defaults))
        pairs = list(zip(positional, pad + list(args.defaults)))
        pairs += list(zip(args.kwonlyargs, args.kw_defaults))
        for arg, default in pairs:
            if arg.arg != 'comm':
                continue
            if isinstance(default, ast.Constant) and default.value is None:
                continue
            found.append(f'{getattr(node, "name", "<lambda>")}:{node.lineno}')
    return found


def comm_parameters(tree):
    """'name:line' of every function, method or lambda taking a `comm`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda)):
            continue
        args = node.args
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        if 'comm' in names:
            found.append(f'{getattr(node, "name", "<lambda>")}:{node.lineno}')
    return found


def forbidden_imports(tree):
    """'module.name:line' of every serve loop, broadcast or replicate imported."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [(node.module or '', a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            names = [tuple(a.name.rsplit('.', 1)) if '.' in a.name
                     else ('', a.name) for a in node.names]
        else:
            continue
        for module, name in names:
            if (name.startswith('serve_') or name == 'broadcast'
                    or 'replicate' in name):
                found.append(f'{module}.{name}:{node.lineno}')
    return found


def dotted(node):
    """'a.b.c' for a Name/Attribute chain, None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return '.'.join([node.id] + parts[::-1])


def mpi_grid_names(tree):
    """{name: line} of every `mpi_grid` name used: imported from it, or read
    as an attribute of the module under whatever name it was bound to."""
    used, aliases = {}, set()
    parent, leaf = MPI_GRID.rsplit('.', 1)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == MPI_GRID:
            for a in node.names:
                used.setdefault(a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == parent:
            aliases.update(a.asname or a.name for a in node.names
                           if a.name == leaf)
        elif isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names
                           if a.name == MPI_GRID)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and dotted(node.value) in aliases:
            used.setdefault(node.attr, node.lineno)
    return used


def parsed(path):
    return ast.parse(path.read_text(), filename=str(path))


def test_both_packages_are_scanned():
    """An empty or mistyped package list would pass every rule below."""
    for package in PACKAGES:
        assert any(module_id(p).startswith(package) for p in MODULES), package
    assert len(MODULES) > 20


def test_the_kernel_whitelist_names_existing_modules():
    """A renamed kernel would otherwise silently fall under the physics rule."""
    for name in KERNEL_MODULES:
        assert (REPO / name).is_file(), name


def test_the_distributed_list_names_existing_modules():
    """A renamed distributed module would otherwise fail every MPI rule, and a
    module named twice would be under two rules."""
    for name in DISTRIBUTED_MODULES:
        assert (REPO / name).is_file(), name
    assert not DISTRIBUTED_MODULES & KERNEL_MODULES


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_no_function_takes_a_comm(path):
    """A chain, surface or optimizer reads the ranks from the context only;
    a realization kernel may take `comm`, but only defaulting to None; a
    distributed module (`DISTRIBUTED_MODULES`) may take it."""
    if module_id(path) in DISTRIBUTED_MODULES:
        return
    if module_id(path) in KERNEL_MODULES:
        found = comm_parameters_without_none_default(parsed(path))
        assert not found, (f'{module_id(path)} is a kernel and takes comm '
                           f'without a None default in {found}')
        return
    found = comm_parameters(parsed(path))
    assert not found, f'{module_id(path)} takes comm= in {found}'


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_no_serve_loop_broadcast_or_replicate_is_imported(path):
    """The root-driven protocol and its broadcasts stay deleted; a distributed
    module may broadcast, and no module imports a serve loop or a replicate."""
    found = forbidden_imports(parsed(path))
    if module_id(path) in DISTRIBUTED_MODULES:
        found = [f for f in found if f.split(':')[0].rsplit('.', 1)[1] != 'broadcast']
    assert not found, f'{module_id(path)} imports {found}'


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_only_the_lockstep_names_come_from_mpi_grid(path):
    """lockstep, lockstep_mean_field and current_comm, and nothing else."""
    if module_id(path) in KERNEL_MODULES:
        pytest.skip('a realization kernel partitions and reduces by design')
    if module_id(path) in DISTRIBUTED_MODULES:
        pytest.skip('a distributed module splits its work over the ranks')
    extra = {name: line for name, line in mpi_grid_names(parsed(path)).items()
             if name not in ALLOWED_MPI_GRID}
    assert not extra, (f'{module_id(path)} uses {sorted(extra)} from '
                       f'{MPI_GRID} (lines {sorted(extra.values())})')


# ------------------------------------------------ the scanners can fail
PLANTED = '''
import src.Base.utils.parallel.mpi_grid as grid
from src.Base.utils.parallel import mpi_grid
from src.Base.utils.parallel.mpi_grid import broadcast, lockstep
from src.SingleReference.GW.space_time import replicate_mean_field
from src.Base.factorization.distributed_df import serve_requests


class Chain:
    def __init__(self, mol, comm=None):
        self.mol = mol

    def run(self, *, comm):
        return grid.reduce_sum(lockstep(1), comm)


def walk(x, **kw):
    return mpi_grid.partition(x, 0, 1), (lambda comm: comm)
'''

ALLOWED = '''
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep, lockstep_mean_field


def evaluate(surface, mol, communicate=False):
    comm = current_comm()
    return lockstep(surface.total_gradient(mol)), comm
'''


def test_the_kernel_rule_scan_finds_the_undefaulted_parameters():
    found = comm_parameters_without_none_default(ast.parse(PLANTED))
    names = sorted(f.split(':')[0] for f in found)
    assert names == ['<lambda>', 'run'], found
    assert comm_parameters_without_none_default(ast.parse(ALLOWED)) == []


def test_the_comm_scan_finds_every_planted_parameter():
    assert comm_parameters(ast.parse(PLANTED)) == [
        '__init__:10', 'run:13', '<lambda>:18']
    assert comm_parameters(ast.parse(ALLOWED)) == []


def test_the_import_scan_finds_every_planted_name():
    assert forbidden_imports(ast.parse(PLANTED)) == [
        'src.Base.utils.parallel.mpi_grid.broadcast:4',
        'src.SingleReference.GW.space_time.replicate_mean_field:5',
        'src.Base.factorization.distributed_df.serve_requests:6']
    assert forbidden_imports(ast.parse(ALLOWED)) == []


def test_the_mpi_grid_scan_follows_every_binding():
    used = mpi_grid_names(ast.parse(PLANTED))
    assert set(used) - ALLOWED_MPI_GRID == {'broadcast', 'reduce_sum',
                                            'partition'}
    assert set(mpi_grid_names(ast.parse(ALLOWED))) <= ALLOWED_MPI_GRID


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))


# --------------------------------------------------------------- the layers
#: The layering rules, enforced on
#: every module of src/ and tools/ too: an import may only point to an
#: equal or lower rank. The longest prefix decides a module's rank, so the
#: method subpackages of the gradient layer sit above its core.
LAYERS = {
    'src.Base': 0, 'src.Solvers': 0,                    # foundation
    'src.environments': 1,                              # environments: Base only
    'src.SingleReference': 2, 'src.MultiReference': 2,  # electronic structure
    'src.fragments': 2,
    'src.gradients': 3.0,                               # the method-neutral core
    'src.gradients.rpa': 3.1,                           # the RPA screening GW-BSE is built on
    'src.gradients.gwbse': 3.2, 'src.gradients.adc': 3.2,
    'src.properties': 4, 'src.autograd_oracle': 4,
    'src.photophysics': 5, 'tools': 5,                  # physics
    'src.CodeGeneration': 9,                            # generators: no runtime module imports them
}
#: Packages that never import each other, at any rank: the methods behind the
#: one surface protocol are interchangeable, not stacked. The one order among
#: them follows the physics: GW-BSE is built on RPA
#: screening, so gwbse may import rpa (its rank is higher) and rpa never gwbse;
#: ADC is built on neither, so adc and rpa are kept apart like adc and gwbse.
SIBLINGS = (('src.gradients.gwbse', 'src.gradients.adc'),
            ('src.gradients.rpa', 'src.gradients.adc'))
#: Separate side by side: imported by no
#: module outside it.
ISOLATED = ('src.SingleReference.Periodic',)
#: Private names read across modules that stay for now:
#: ee_spin_blocks keeps `_ew`, `_ksum`, `_slab_plan`, `_stream` and ee_driver
#: `_channel_basis` as aliases for ee_fold and ee_gw_pieces.
PRIVATE_EXCEPTIONS = {
    'src/SingleReference/ADC/eeADC/ee_fold.py': {
        'src.SingleReference.ADC.eeADC.ee_driver._channel_basis'},
    'src/SingleReference/ADC/eeADC/ee_gw_pieces.py': {
        'src.SingleReference.ADC.eeADC.ee_spin_blocks._ew',
        'src.SingleReference.ADC.eeADC.ee_spin_blocks._ksum',
        'src.SingleReference.ADC.eeADC.ee_spin_blocks._slab_plan',
        'src.SingleReference.ADC.eeADC.ee_spin_blocks._stream'},
}
LAYERED_MODULES = sorted(path for package in ('src', 'tools')
                         for path in (REPO / package).rglob('*.py'))


def package_of(name, packages):
    """The longest of `packages` that `name` is in or under, None if none."""
    best = None
    for package in packages:
        if name == package or name.startswith(package + '.'):
            if best is None or len(package) > len(best):
                best = package
    return best


def layer_of(name):
    """The rank of the package module `name` belongs to, None outside
    `LAYERS` (pySCF, numpy, the standard library)."""
    package = package_of(name, LAYERS)
    return None if package is None else LAYERS[package]


def imported(tree):
    """[(line, module)] of every absolute import in `tree`, at any depth (a
    call-time import is still an import); `from a import b` names a.b too, so
    `from src import properties` is caught."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out += [(node.lineno, node.module)]
            out += [(node.lineno, f'{node.module}.{a.name}') for a in node.names]
    return out


def upward_edges(tree, own_rank):
    """[(line, module)] of every import of a layer above `own_rank`."""
    return [(line, name) for line, name in imported(tree)
            if (layer_of(name) or 0) > own_rank]


def sideways_edges(tree, own):
    """[(line, module)] of every import of a sibling of module `own`, or of an
    isolated package `own` is not in."""
    found = []
    for line, name in imported(tree):
        for group in SIBLINGS:
            mine, theirs = package_of(own, group), package_of(name, group)
            if mine and theirs and mine != theirs:
                found.append((line, name))
        walled = package_of(name, ISOLATED)
        if walled and package_of(own, ISOLATED) != walled:
            found.append((line, name))
    return found


def private_names(tree):
    """['module.name'] of every private name of another src or tools module
    this tree reads: imported by name, or read as an attribute of a module it
    imported."""
    found, modules = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and \
                layer_of(node.module) is not None:
            for a in node.names:
                if a.name.startswith('_') and not a.name.startswith('__'):
                    found.append(f'{node.module}.{a.name}')
                else:
                    modules[a.asname or a.name] = f'{node.module}.{a.name}'
        elif isinstance(node, ast.Import):
            modules.update((a.asname or a.name, a.name) for a in node.names
                           if layer_of(a.name) is not None and a.asname)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in modules and node.attr.startswith('_')
                and not node.attr.startswith('__')):
            found.append(f'{modules[node.value.id]}.{node.attr}')
    return found


def dotted_module(path):
    """'src.a.b' of the module at `path` (a package's __init__ is the package)."""
    name = module_id(path)[:-3].replace('/', '.')
    return name[:-len('.__init__')] if name.endswith('.__init__') else name


@pytest.mark.parametrize('planted, rank, bad', [
    ('from src.properties.optimize import relax\n', 3, True),
    ('def f():\n    from src.photophysics import uisc\n', 4, True),
    ('import tools.ci.run_tier\n', 3, True),
    ('from src import properties\n', 3, True),
    ('from src.gradients.state_identity import follow_state\n', 4, False),
    ('from src.Base.numerics.constants import KAPPA\n', 3, False),
    ('from src.gradients.gwbse.excited_state import ExcitedStateChain\n', 3.0, True),
    ('from src.gradients.rpa.ground_state import RPAGroundStateChain\n', 3.2, False),
    ('from src.gradients.gwbse.excited_state import ExcitedStateChain\n', 3.1, True),
    ('from src.environments.reaction_field import x\n', 0, True),
    ('from src.SingleReference.GW import qp_solve\n', 1, True),
    ('from src.CodeGeneration import generate_adcn\n', 5, True),
])
def test_the_layer_scanner_sees_an_upward_import(planted, rank, bad):
    assert bool(upward_edges(ast.parse(planted), rank)) is bad


@pytest.mark.parametrize('planted, own, bad', [
    ('from src.gradients.adc import ee_gradient\n', 'src.gradients.gwbse.excited_state', True),
    ('from src.gradients.gwbse.conventions import x\n', 'src.gradients.adc.ee_state_model', True),
    ('from src.gradients.gwbse.conventions import x\n', 'src.gradients.gwbse.qp_set', False),
    ('from src.gradients.rpa.factor_chain import FactorChain\n', 'src.gradients.adc.ee_gradient', True),
    ('from src.gradients.adc import ee_state_model\n', 'src.gradients.rpa.ground_state', True),
    ('from src.gradients.rpa.factor_chain import FactorChain\n', 'src.gradients.gwbse.excited_state', False),
    ('from src.SingleReference.Periodic import pbc_rpa\n', 'src.SingleReference.GW.qp_solve', True),
    ('from src.SingleReference.Periodic import pbc_rpa\n', 'src.SingleReference.Periodic.pbc_wav', False),
])
def test_the_sideways_scanner_sees_a_sibling_or_a_walled_package(planted, own, bad):
    assert bool(sideways_edges(ast.parse(planted), own)) is bad


@pytest.mark.parametrize('planted, bad', [
    ('from src.Base.factorization.isdf_jk import _base_functional\n', True),
    ('from src.Base.factorization import isdf_jk\nisdf_jk._base_functional\n', True),
    ('from src.Base.factorization.isdf_jk import base_functional\n', False),
    ('from numpy import _NoValue\n', False),
])
def test_the_private_name_scanner_sees_a_private_name(planted, bad):
    assert bool(private_names(ast.parse(planted))) is bad


def test_every_module_has_a_layer():
    """A module outside every package of LAYERS would be under no rule."""
    lost = [module_id(p) for p in LAYERED_MODULES
            if layer_of(dotted_module(p)) is None]
    assert not lost, lost


@pytest.mark.parametrize('path', LAYERED_MODULES, ids=module_id)
def test_no_import_points_up(path):
    """physics -> properties -> gradients -> electronic structure ->
    environments -> foundation, never back."""
    own = layer_of(dotted_module(path))
    assert not upward_edges(parsed(path), own), module_id(path)


@pytest.mark.parametrize('path', LAYERED_MODULES, ids=module_id)
def test_no_import_points_sideways(path):
    """The method subpackages of the gradient layer never import each other,
    and nothing outside Periodic imports it."""
    found = sideways_edges(parsed(path), dotted_module(path))
    assert not found, (module_id(path), found)


@pytest.mark.parametrize('path', LAYERED_MODULES, ids=module_id)
def test_no_private_name_crosses_a_module(path):
    """A name with a leading underscore is its module's own; reading it from
    another module is an interface nobody declared."""
    found = set(private_names(parsed(path)))
    found -= PRIVATE_EXCEPTIONS.get(module_id(path), set())
    assert not found, (module_id(path), sorted(found))


def test_the_private_exceptions_are_still_read():
    """An exception nobody needs any more is removed, not kept."""
    for path, names in PRIVATE_EXCEPTIONS.items():
        assert set(private_names(parsed(REPO / path))) >= names, path


#: The packages of each rank a fresh interpreter imports whole; the
#: generators (pdaggerq) and the torch oracle are left out, and a module whose
#: optional library is missing here (pyelpa) is skipped.
FRESH_RANKS = sorted({rank for rank in LAYERS.values() if rank < 9})


@pytest.mark.parametrize('rank', FRESH_RANKS)
def test_importing_a_layer_loads_nothing_above_it(rank):
    """A fresh interpreter that imports every module of one rank holds no
    module of a higher rank, and none of a sibling of what it imported."""
    mine = [p for p, r in LAYERS.items() if r == rank and p != 'src.autograd_oracle']
    modules = sorted(
        m for m in (dotted_module(p) for p in LAYERED_MODULES)
        if package_of(m, LAYERS) in mine and not m.endswith('__main__'))
    above = tuple(p for p, r in LAYERS.items() if r > rank)
    code = ('import importlib, sys\n'
            f'for m in {modules!r}:\n'
            '    try:\n'
            '        importlib.import_module(m)\n'
            '    except ImportError as exc:\n'
            "        if (exc.name or '').startswith('src'):\n"
            '            raise\n'
            f'print(sorted(m for m in sys.modules if m.startswith({above!r})))\n')
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == '[]', out.stdout
