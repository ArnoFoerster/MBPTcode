"""Helpers shared by the run tools: the frozen core and the code identity of a record."""
import os
import subprocess


def frozen_core(mol):
    """Doubly occupied core orbitals: 1s past He, 1s2s2p past Ne, 1s..3d past Ar
    (the last only for Ga onwards; K-Zn keep their 3s3p frozen too)."""
    n = 0
    for a in range(mol.natm):
        z = mol.atom_charge(a)
        n += (z > 2) + 4 * (z > 10) + 4 * (z > 18) + 5 * (z > 30)
    return int(n)


def code_id(root):
    """The commit a record was made with: SNAPSHOT.txt's first word in an exported
    tree, else the checkout's HEAD, else None."""
    snap = os.path.join(root, 'SNAPSHOT.txt')
    if os.path.isfile(snap):
        with open(snap) as fh:
            return fh.read().split()[0]
    try:
        return subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:                                   # noqa: BLE001
        return None
