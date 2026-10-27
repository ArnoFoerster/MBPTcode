"""Skip a comparison against an archived commit when the tree has no history.

Several tests compare the current code with the code of a pinned commit,
unpacked with `git archive`. A tree that is not a checkout (a source archive,
a copied job directory) has no `.git`, or one without that commit; the
comparison cannot be made there and is skipped with that reason, while the
other tests of the file run.
"""
import subprocess
import tarfile

import pytest


def require_commit(repo, commit):
    """Skip the calling test unless `repo` is a git repository holding `commit`."""
    held = subprocess.run(['git', '-C', str(repo), 'cat-file', '-e',
                           f'{commit}^{{commit}}'],
                          capture_output=True, text=True)
    if held.returncode != 0:
        pytest.skip(f'{repo} is not a git repository holding {commit} (a '
                    f'copy or archive of the tree): the baseline code cannot '
                    f'be extracted; run in a checkout or a detached worktree')


def unpack_commit(repo, commit, tar, out):
    """Unpack the tree of `commit` of `repo` into `out`, via the file `tar`."""
    done = subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar',
                           '-o', str(tar), commit],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
