"""
Generate synthetic IVT dataset used by identifiability analyses (Phase B, C).
Run once to produce data/synthetic_ivt_study.csv before running 09 and 10.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from model.tmdd_model import (
    rhs, params_for_format, dose_nmol_from_mg, TARGET_BIO,
)

TRUE_PARAMS = params_for_format('fc', 'human')
DOSE_NMOL = dose_nmol_from_mg(5.0, 'fc')
NOISE_CV = 0.15

sample_days = np.array([0.5, 1, 2, 5, 14, 30, 60])
sample_h = sample_days * 24
y0 = [DOSE_NMOL, 0, 0, TARGET_BIO['R0_nM'] * TRUE_PARAMS['Vr'], 0, 0]
sol = solve_ivp(rhs, (0, sample_h[-1]), y0, args=(TRUE_PARAMS,),
                t_eval=sample_h, method='LSODA', rtol=1e-9, atol=1e-12)

Cv_true = sol.y[0] / TRUE_PARAMS['Vv']
R_true = sol.y[3] / TRUE_PARAMS['Vr']
DR_true = sol.y[4] / TRUE_PARAMS['Vr']
TO_true = DR_true / (R_true + DR_true + 1e-30)

rng = np.random.default_rng(42)
Cv_obs = Cv_true * np.exp(rng.normal(0, NOISE_CV, size=len(Cv_true)))
TO_obs = np.clip(TO_true * np.exp(rng.normal(0, NOISE_CV, size=len(TO_true))),
                 1e-6, 0.999)

df = pd.DataFrame({
    't_day': sample_days, 't_h': sample_h,
    'Cv_true_nM': Cv_true, 'TO_true': TO_true,
    'Cv_obs_nM': Cv_obs, 'TO_obs': TO_obs,
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_ivt_study.csv', index=False)
print("Saved data/synthetic_ivt_study.csv")
print(df.to_string(index=False))
