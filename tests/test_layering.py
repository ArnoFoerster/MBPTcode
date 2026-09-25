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
    pattern these names implement are what the context replaced;
  * the only `mpi_grid` names a module uses are `lockstep`,
    `lockstep_mean_field` and `current_comm` -- lock what the layer decided
    itself, ask for the rank where a side effect needs it, nothing more.

The scanners are checked against planted sources below, so each rule is shown
to fail on the pattern it forbids and to pass on the one it allows.
"""
import ast
import pathlib
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
KERNEL_MODULES = frozenset({'src/gradients/qp_space_time.py',
                            'src/gradients/space_time_adjoint.py',
                            'src/gradients/reaction_field_adjoint.py'})

#: The module the distribution primitives live in.
MPI_GRID = 'src.Base.utils.mpi_grid'

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


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_no_function_takes_a_comm(path):
    """A chain, surface or optimizer reads the ranks from the context only;
    a realization kernel may take `comm`, but only defaulting to None."""
    if module_id(path) in KERNEL_MODULES:
        found = comm_parameters_without_none_default(parsed(path))
        assert not found, (f'{module_id(path)} is a kernel and takes comm '
                           f'without a None default in {found}')
        return
    found = comm_parameters(parsed(path))
    assert not found, f'{module_id(path)} takes comm= in {found}'


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_no_serve_loop_broadcast_or_replicate_is_imported(path):
    """The root-driven protocol and its broadcasts stay deleted."""
    found = forbidden_imports(parsed(path))
    assert not found, f'{module_id(path)} imports {found}'


@pytest.mark.parametrize('path', MODULES, ids=module_id)
def test_only_the_lockstep_names_come_from_mpi_grid(path):
    """lockstep, lockstep_mean_field and current_comm, and nothing else."""
    if module_id(path) in KERNEL_MODULES:
        pytest.skip('a realization kernel partitions and reduces by design')
    extra = {name: line for name, line in mpi_grid_names(parsed(path)).items()
             if name not in ALLOWED_MPI_GRID}
    assert not extra, (f'{module_id(path)} uses {sorted(extra)} from '
                       f'{MPI_GRID} (lines {sorted(extra.values())})')


# ------------------------------------------------ the scanners can fail
PLANTED = '''
import src.Base.utils.mpi_grid as grid
from src.Base.utils import mpi_grid
from src.Base.utils.mpi_grid import broadcast, lockstep
from src.SingleReference.GW.space_time import replicate_mean_field
from src.Base.distributed_df import serve_requests


class Chain:
    def __init__(self, mol, comm=None):
        self.mol = mol

    def run(self, *, comm):
        return grid.reduce_sum(lockstep(1), comm)


def walk(x, **kw):
    return mpi_grid.partition(x, 0, 1), (lambda comm: comm)
'''

ALLOWED = '''
from src.Base.utils.mpi_grid import current_comm, lockstep, lockstep_mean_field


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
        'src.Base.utils.mpi_grid.broadcast:4',
        'src.SingleReference.GW.space_time.replicate_mean_field:5',
        'src.Base.distributed_df.serve_requests:6']
    assert forbidden_imports(ast.parse(ALLOWED)) == []


def test_the_mpi_grid_scan_follows_every_binding():
    used = mpi_grid_names(ast.parse(PLANTED))
    assert set(used) - ALLOWED_MPI_GRID == {'broadcast', 'reduce_sum',
                                            'partition'}
    assert set(mpi_grid_names(ast.parse(ALLOWED))) <= ALLOWED_MPI_GRID


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
