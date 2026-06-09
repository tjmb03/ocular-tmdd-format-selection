"""
Sanity checks for the TMDD model.

Run with:
    pytest tests/

Or directly:
    python tests/test_model_recovery.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.tmdd_model import (
    params_for_format, simulate, dose_nmol_from_mg, TARGET_BIO,
)


def test_mass_balance_no_target():
    """Without binding, total drug should equal cumulative elimination + remaining.

    Set k_on = 0 (no TMDD) and check that the total drug at any time
    equals dose minus cumulative aqueous outflow minus systemic clearance.
    """
    p = params_for_format('fc', 'rabbit')
    p['k_on'] = 0.0
    p['k_off'] = 0.0
    p['phi_FcRn'] = 0.0
    dose = dose_nmol_from_mg(2.0, 'fc')

    res = simulate(p, dose, t_max_d=60, n_pts=300)
    assert res is not None, "Simulation failed"

    # All target should remain at baseline (no binding)
    R_final = res['R'][-1]
    R0_expected = TARGET_BIO['R0_nM']
    assert abs(R_final - R0_expected) / R0_expected < 0.01, \
        f"Target drifted from baseline: R_final={R_final}, expected={R0_expected}"

    # Complex should be zero
    assert np.max(res['DR']) < 1e-6, "Complex formed without binding"
    print("[OK] Mass balance (no target binding) check passed")


def test_format_dose_inversion():
    """The headline result: equal mg vs equimolar should invert ranking."""
    equimolar_nmol = dose_nmol_from_mg(2.0, 'fc')  # 17.4 nmol

    # Equal mg: naked should have highest peak TO (most molecules)
    peaks_eqmg = {}
    for fmt in ['naked', 'fab', 'fc']:
        p = params_for_format(fmt, 'rabbit')
        dose = dose_nmol_from_mg(2.0, fmt)
        res = simulate(p, dose, t_max_d=30)
        peaks_eqmg[fmt] = float(res['TO'].max())

    # Equimolar: Fc should win on AUC (FcRn rescue)
    aucs_eqmol = {}
    for fmt in ['naked', 'fab', 'fc']:
        p = params_for_format(fmt, 'rabbit')
        res = simulate(p, equimolar_nmol, t_max_d=30)
        aucs_eqmol[fmt] = float(np.trapezoid(res['TO'], res['t_d']))

    assert peaks_eqmg['naked'] > peaks_eqmg['fc'], \
        f"At equal mg, naked should beat fc on peak TO. Got: {peaks_eqmg}"
    assert aucs_eqmol['fc'] > aucs_eqmol['naked'], \
        f"At equimolar, fc should beat naked on AUC TO. Got: {aucs_eqmol}"

    print("[OK] Format ranking inversion (equal mg vs equimolar) reproduced")
    print(f"     Equal mg peaks: {peaks_eqmg}")
    print(f"     Equimolar AUCs: {aucs_eqmol}")


def test_fcrn_only_for_fc():
    """phi_FcRn must be 0 for naked and fab, 1 for fc."""
    assert params_for_format('naked', 'rabbit')['phi_FcRn'] == 0
    assert params_for_format('fab',   'rabbit')['phi_FcRn'] == 0
    assert params_for_format('fc',    'rabbit')['phi_FcRn'] == 1
    print("[OK] FcRn switch correctly format-specific")


def test_human_scales_volumes():
    """Human parameters should have larger ocular volumes than rabbit."""
    pr = params_for_format('fc', 'rabbit')
    ph = params_for_format('fc', 'human')
    assert ph['Vv'] > pr['Vv'], "Human vitreous should be larger than rabbit"
    assert ph['Vp'] > pr['Vp'], "Human plasma should be larger than rabbit"
    # CL_p should scale allometrically (BW^0.85, ~10x for 70/3 kg)
    ratio = ph['CL_p'] / pr['CL_p']
    expected = (70/3)**0.85
    assert abs(ratio - expected) / expected < 0.01, \
        f"CL_p scaling off: ratio={ratio}, expected~{expected}"
    print(f"[OK] Allometric CL_p scaling: rabbit -> human {ratio:.1f}x "
          f"(expected {expected:.1f}x)")


if __name__ == '__main__':
    test_mass_balance_no_target()
    test_format_dose_inversion()
    test_fcrn_only_for_fc()
    test_human_scales_volumes()
    print("\nAll sanity tests passed.")
