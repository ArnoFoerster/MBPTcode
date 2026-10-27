import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.orbital_spaces import get_occ_virt_indices
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.DensityMatrix.density_matrix import GWDensityMatrixSolver

if __name__ == '__main__':
    all_ok = True

    eps = np.array([-2.0, -1.0, -0.5, 0.3, 0.8, 1.2])
    nocc = 3
    occ, virt = get_occ_virt_indices(eps, nocc)
    ok = np.array_equal(occ, [0, 1, 2]) and np.array_equal(virt, [3, 4, 5])
    all_ok &= ok
    print(f"get_occ_virt_indices basic split: {'OK' if ok else 'FAIL'}")

    # The three classes that delegate _get_occ_virt_indices to this function
    # must all agree with each other and with the standalone function.
    lr = LinearResponseSolver(eps, spin_mode='restricted')
    se = SelfEnergySolver(eps, spin_mode='restricted')
    gw = GWDensityMatrixSolver(eps)
    for name, obj in [('LinearResponseSolver', lr), ('SelfEnergySolver', se), ('GWDensityMatrixSolver', gw)]:
        o, v = obj._get_occ_virt_indices(eps, nocc)
        ok = np.array_equal(o, occ) and np.array_equal(v, virt)
        all_ok &= ok
        print(f"{name}._get_occ_virt_indices delegates correctly: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
