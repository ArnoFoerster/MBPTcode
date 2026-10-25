import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.grids.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid, minimax_supported_sizes

if __name__ == '__main__':
    all_ok = True

    # int_0^infty domega / (1+omega^2) = pi/2
    for nfreq in [10, 20, 40]:
        pts, wts = gauss_legendre_grid(nfreq, w0=0.5)
        integral = np.sum(wts / (1.0 + pts**2))
        ok = abs(integral - np.pi / 2) < 1e-4
        all_ok &= ok
        print(f"gauss_legendre_grid nfreq={nfreq}: int 1/(1+w^2) = {integral:.8f} (pi/2={np.pi/2:.8f}) {'OK' if ok else 'FAIL'}")

    eps = np.array([-1.0, -0.5, 0.2, 0.8, 1.5])
    nocc = 2
    w0 = gap_scaled_w0(eps, nocc, scale=0.5)
    ok = abs(w0 - 0.5 * (eps[2] - eps[1])) < 1e-12
    all_ok &= ok
    print(f"gap_scaled_w0 arithmetic: {'OK' if ok else 'FAIL'}")

    sizes = minimax_supported_sizes()
    ok = sizes == sorted(sizes) and 20 in sizes and 6 in sizes
    all_ok &= ok
    print(f"minimax_supported_sizes(): {sizes} {'OK' if ok else 'FAIL'}")

    try:
        minimax_frequency_grid(7, e_min=0.3, e_max=3.0)
        print("minimax_frequency_grid(nfreq=7) [unsupported size] did NOT raise: FAIL")
        all_ok = False
    except ValueError:
        print("minimax_frequency_grid(nfreq=7) [unsupported size] raises ValueError: OK")

    # Minimax grids are optimized so that sum_k w_k * (2/pi) * x/(x^2+omega_k^2) ~= 1
    # for x in [e_min, e_max] (the standard int_0^infty (2/pi) x/(x^2+w^2) dw = 1 identity
    # restricted to the range the grid targets) -- the actual design criterion, not a
    # generic quadrature check.
    e_min, e_max = 0.3, 3.0
    for nfreq in [10, 20, 30]:
        pts, wts = minimax_frequency_grid(nfreq, e_min, e_max)
        ok_shape = pts.shape == (nfreq,) and wts.shape == (nfreq,)
        x_mid = np.sqrt(e_min * e_max)
        integral = np.sum(wts * (2.0 / np.pi) * x_mid / (x_mid**2 + pts**2))
        ok = ok_shape and abs(integral - 1.0) < 0.05
        all_ok &= ok
        print(f"minimax_frequency_grid nfreq={nfreq}: int-check = {integral:.6f} (target 1.0) {'OK' if ok else 'FAIL'}")

    # Rescaling sanity: doubling both e_min and e_max should double the points
    # and weights (the tabulated data is dimensionless, physical grid = table * e_min).
    pts1, wts1 = minimax_frequency_grid(20, 0.3, 3.0)
    pts2, wts2 = minimax_frequency_grid(20, 0.6, 6.0)
    ok = np.allclose(pts2, 2.0 * pts1) and np.allclose(wts2, 2.0 * wts1)
    all_ok &= ok
    print(f"minimax_frequency_grid linear rescaling with e_min: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
