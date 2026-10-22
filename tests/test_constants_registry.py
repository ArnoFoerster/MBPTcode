import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import get_method_info, METHOD_REGISTRY

if __name__ == '__main__':
    all_ok = True

    for name in METHOD_REGISTRY:
        info = get_method_info(name)
        assert set(info) == {'vertex_mode', 'force_rpa_casida', 'needs_vertex', 'needs_triplet'}
    print(f"all {len(METHOD_REGISTRY)} registered methods return well-formed info: OK")

    lower = get_method_info('gwgammainf')
    upper = get_method_info('GWGammaInf')
    ok = lower == upper
    all_ok &= ok
    print(f"case-insensitive lookup (gwgammainf == GWGammaInf): {'OK' if ok else 'FAIL'}")

    for bad in ['PSD3', 'not_a_method', '']:
        try:
            get_method_info(bad)
            print(f"get_method_info('{bad}') did NOT raise: FAIL")
            all_ok = False
        except ValueError:
            print(f"get_method_info('{bad}') raises ValueError: OK")

    for name in ['GW', 'GW@RPA']:
        ok = get_method_info(name)['force_rpa_casida'] is True
        all_ok &= ok
        print(f"{name}.force_rpa_casida == True: {'OK' if ok else 'FAIL'}")
    for name in ['GW@BSE', 'GWGammaInf', 'PSD1']:
        ok = get_method_info(name)['force_rpa_casida'] is False
        all_ok &= ok
        print(f"{name}.force_rpa_casida == False: {'OK' if ok else 'FAIL'}")

    for name in ['GW', 'GW@RPA', 'GW@BSE']:
        ok = get_method_info(name)['needs_vertex'] is False
        all_ok &= ok
        print(f"{name}.needs_vertex == False: {'OK' if ok else 'FAIL'}")
    for name in ['GWGammaInf', 'PSD1', 'PSD2']:
        ok = get_method_info(name)['needs_vertex'] is True
        all_ok &= ok
        print(f"{name}.needs_vertex == True: {'OK' if ok else 'FAIL'}")

    for name in ['PSD2', 'PSD4', 'PSD7', 'PSD9']:
        ok = get_method_info(name)['needs_triplet'] is True
        all_ok &= ok
        print(f"{name}.needs_triplet == True: {'OK' if ok else 'FAIL'}")
    for name in ['GW', 'PSD1', 'PSD5', 'PSD6', 'PSD8']:
        ok = get_method_info(name)['needs_triplet'] is False
        all_ok &= ok
        print(f"{name}.needs_triplet == False: {'OK' if ok else 'FAIL'}")

    # End-to-end: calc_qp_energy must fail loudly for an unimplemented method
    # (no silent fallback to plain GW).
    from pyscf import gto, scf, df
    from src.SingleReference.GW.qp_energy import calc_qp_energy

    mol = gto.M(atom='Ne 0 0 0', basis='sto-3g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    try:
        calc_qp_energy(mf, selfenergy='PSD3', polarizability='BSE')
        print("calc_qp_energy(selfenergy='PSD3') did NOT raise: FAIL")
        all_ok = False
    except ValueError:
        print("calc_qp_energy(selfenergy='PSD3') raises ValueError: OK")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
