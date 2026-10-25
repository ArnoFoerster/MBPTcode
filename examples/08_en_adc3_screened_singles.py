"""EN-ADC(3) with the RPA-screened singles shift: singles='screened'.

The singles denominator shift is normally the bare CIS/BSE diagonal
<ia||ia> = J - K (Jiang & Engel Eq. 21). singles='screened' replaces the
e-h DIRECT term J by its static (omega=0) full-RPA-screened counterpart
J_W = (aa|W|ii), giving J_W - K -- the same channel split as the BSE
kernel: direct term screened, exchange/ring term (K) left bare.

Restricted only (RHF); raises on the UHF branch. Compares against bare
singles (True) and doubles-only (False) so the size of the effect is
visible on one molecule."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.ADC import ADCSolver, build_static_correction


BASE = {'hh': True, 'pp': True, 'spin_adapted': True, 'shift': 'sum'}

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

for singles, tag in ((False, 'doubles only'), (True, 'bare J-K'),
                     ('screened', 'screened J_W-K')):
    dress = dict(BASE, singles=singles)
    solver = ADCSolver(mf, level='adc3', en_dress=dress)
    static = build_static_correction(mf, kind='mp2_relaxed', en_dress=dress)
    eGF, Z = solver.solve(static_correction=static)
    print(f'EN-ADC(3) singles={tag:16s} IP = {-eGF[0] * HARTREE_TO_EV:.3f} eV   '
          f'Z = {Z[0]:.3f}')
