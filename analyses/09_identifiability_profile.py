"""
Phase B: Practical Identifiability via Profile Likelihood
==========================================================
Raue et al. 2009 framework. Generates synthetic preclinical data at the
truth parameter values, finds the MLE, then for each profiled parameter
fixes it at a grid of values and re-optimizes the rest. The 95% CI is the
range over which Delta NLL stays below the chi-squared 1-DOF threshold (1.92).

Profiles 4 mechanistically key parameters (k_va, k_off, f_rec, k_int) by
default - extend PARAMS_TO_PROFILE to all 12 for the full analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from model.tmdd_model import (
    rhs, params_for_format, dose_nmol_from_mg, TARGET_BIO,
)
from scipy.integrate import solve_ivp

# ==========================================================================
# Generate (or load) synthetic data
# ==========================================================================
SYNTHETIC_PATH = 'data/synthetic_ivt_study.csv'
TRUE_PARAMS = params_for_format('fc', 'human')
PARAM_NAMES = ['k_va', 'k_av', 'k_ao', 'k_vr', 'k_rv',
               'k_on', 'k_off', 'k_int', 'k_syn', 'k_deg',
               'f_rec', 'CL_p']
LOG_TRUE = np.log(np.array([TRUE_PARAMS[k] for k in PARAM_NAMES]))
DOSE_NMOL = dose_nmol_from_mg(5.0, 'fc')
NOISE_CV = 0.15

if not os.path.exists(SYNTHETIC_PATH):
    print("Generating synthetic data...")
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

    df_data = pd.DataFrame({
        't_day': sample_days, 't_h': sample_h,
        'Cv_true_nM': Cv_true, 'TO_true': TO_true,
        'Cv_obs_nM': Cv_obs, 'TO_obs': TO_obs,
    })
    os.makedirs('data', exist_ok=True)
    df_data.to_csv(SYNTHETIC_PATH, index=False)
else:
    df_data = pd.read_csv(SYNTHETIC_PATH)
    print(f"Loaded existing synthetic data from {SYNTHETIC_PATH}")

t_obs_h = df_data['t_h'].values
Cv_obs = df_data['Cv_obs_nM'].values
TO_obs = df_data['TO_obs'].values


# ==========================================================================
# Likelihood
# ==========================================================================
def simulate_at_obs(p_array):
    p = dict(zip(PARAM_NAMES, p_array))
    p.update(dict(Vv=TRUE_PARAMS['Vv'], Va=TRUE_PARAMS['Va'],
                  Vr=TRUE_PARAMS['Vr'], Vp=TRUE_PARAMS['Vp'],
                  phi_FcRn=1.0))
    R0 = p['k_syn'] / max(p['k_deg'], 1e-10)
    y0 = [DOSE_NMOL, 0, 0, R0 * p['Vr'], 0, 0]
    try:
        sol = solve_ivp(rhs, (0, t_obs_h[-1]), y0, args=(p,),
                        t_eval=t_obs_h, method='LSODA', rtol=1e-7, atol=1e-10)
        if not sol.success:
            return None, None
        Cv = sol.y[0] / p['Vv']
        R = sol.y[3] / p['Vr']
        DR = sol.y[4] / p['Vr']
        return Cv, DR / (R + DR + 1e-30)
    except Exception:
        return None, None


def nll(log_params):
    p_array = np.exp(log_params)
    if np.any(p_array < 1e-12) or np.any(p_array > 1e8):
        return 1e10
    Cv, TO = simulate_at_obs(p_array)
    if Cv is None:
        return 1e10
    Cv = np.clip(Cv, 1e-12, None)
    TO_p = np.clip(TO, 1e-12, 1 - 1e-12)
    TO_o = np.clip(TO_obs, 1e-12, 1 - 1e-12)
    r_Cv = (np.log(Cv) - np.log(Cv_obs)) / NOISE_CV
    r_TO = (np.log(TO_p) - np.log(TO_o)) / NOISE_CV
    v = 0.5 * np.sum(r_Cv**2 + r_TO**2)
    return v if np.isfinite(v) else 1e10


# ==========================================================================
# Optimization
# ==========================================================================
N_PARAMS = len(PARAM_NAMES)
LOG_BOUNDS = [(-12, 5)] * N_PARAMS
DELTA_CHI2_95 = 3.84
N_GRID = 8          # points per side; finer grid for accurate CI bounds
LOG_RADIUS = 1.5
PARAMS_TO_PROFILE = ['k_va', 'k_off', 'f_rec', 'k_int']


def fit(start_log, fixed_idx=None, fixed_log_val=None):
    if fixed_idx is None:
        res = minimize(nll, start_log, method='L-BFGS-B', bounds=LOG_BOUNDS,
                       options=dict(maxiter=100, ftol=1e-4))
        return res.x, res.fun
    free_idx = [i for i in range(N_PARAMS) if i != fixed_idx]
    x0 = start_log[free_idx]
    bounds = [LOG_BOUNDS[i] for i in free_idx]
    def fn(log_free):
        full = np.zeros(N_PARAMS)
        for k, i in enumerate(free_idx):
            full[i] = log_free[k]
        full[fixed_idx] = fixed_log_val
        return nll(full)
    res = minimize(fn, x0, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=80, ftol=1e-4))
    return res.x, res.fun


print("="*72)
print("Phase B: Profile Likelihood")
print("="*72)
t0 = time.time()
mle_log, NLL_min = fit(LOG_TRUE)
print(f"MLE found in {time.time()-t0:.1f}s, NLL_min = {NLL_min:.4f}")

all_profiles = {}
print(f"\nProfiling {len(PARAMS_TO_PROFILE)} parameters...")
for name in PARAMS_TO_PROFILE:
    i = PARAM_NAMES.index(name)
    t0 = time.time()
    center = mle_log[i]
    grid = np.concatenate([
        np.linspace(center - LOG_RADIUS, center, N_GRID + 1)[:-1],
        [center],
        np.linspace(center, center + LOG_RADIUS, N_GRID + 1)[1:],
    ])
    profile = []
    for lg in grid:
        _, v = fit(mle_log, fixed_idx=i, fixed_log_val=lg)
        profile.append((lg, v))
    all_profiles[name] = np.array(profile)
    in_ci = (all_profiles[name][:, 1] - NLL_min) < DELTA_CHI2_95 / 2
    print(f"  {name:>8s}: {time.time()-t0:.0f}s, {in_ci.sum()}/{len(in_ci)} in CI")


# ==========================================================================
# Summary
# ==========================================================================
print("\n95% CI summary:")
print(f"{'Param':<10} {'Truth':>10} {'CI lower':>12} {'CI upper':>12} {'Status':<22}")
rows = []
for name in PARAMS_TO_PROFILE:
    i = PARAM_NAMES.index(name)
    prof = all_profiles[name]
    delta = prof[:, 1] - NLL_min
    in_ci = delta < DELTA_CHI2_95 / 2

    if not in_ci.any():
        ci_l = ci_u = np.nan
        status = "no CI"
    else:
        bl = prof[in_ci, 0]
        ci_l_log, ci_u_log = bl.min(), bl.max()
        edge_tol = 0.1
        lower_open = (ci_l_log - prof[:, 0].min()) < edge_tol
        upper_open = (prof[:, 0].max() - ci_u_log) < edge_tol
        ci_l = np.exp(ci_l_log) if not lower_open else -np.inf
        ci_u = np.exp(ci_u_log) if not upper_open else np.inf
        width = (ci_u_log - ci_l_log) / np.log(10) \
                if np.isfinite(ci_l) and np.isfinite(ci_u) else np.nan
        if lower_open and upper_open:
            status = "non-identifiable"
        elif lower_open or upper_open:
            status = "one-sided open"
        elif width < 0.5:
            status = "well-identified"
        else:
            status = "weakly identified"

    print(f"{name:<10} {TRUE_PARAMS[name]:>10.4g} "
          f"{ci_l:>12.4g} {ci_u:>12.4g} {status:<22}")
    rows.append(dict(parameter=name, truth=TRUE_PARAMS[name],
                     CI_lower=ci_l, CI_upper=ci_u, status=status))

pd.DataFrame(rows).to_csv('results/tables/profile_likelihood_summary.csv', index=False)

with open('results/tables/profile_likelihood_curves.json', 'w') as f:
    json.dump({
        'NLL_min': NLL_min, 'MLE_log': mle_log.tolist(),
        'profiles': {k: v.tolist() for k, v in all_profiles.items()},
    }, f, indent=2)


# Plot
fig, axes = plt.subplots(1, len(PARAMS_TO_PROFILE),
                          figsize=(4 * len(PARAMS_TO_PROFILE), 4.2))
if len(PARAMS_TO_PROFILE) == 1:
    axes = [axes]

for ax, name in zip(axes, PARAMS_TO_PROFILE):
    prof = all_profiles[name]
    pv = np.exp(prof[:, 0])
    delta = prof[:, 1] - NLL_min
    ax.plot(pv, delta, 'o-', color='#1D9E75', linewidth=1.6, markersize=6,
            label='Profile')
    ax.axhline(DELTA_CHI2_95 / 2, color='black', linestyle='--', alpha=0.6,
               label='95% CI threshold')
    ax.axvline(TRUE_PARAMS[name], color='#D85A30', linestyle=':',
               linewidth=1.4, label='Truth')
    in_ci = delta < DELTA_CHI2_95 / 2
    if in_ci.any():
        ax.axvspan(pv[in_ci].min(), pv[in_ci].max(), alpha=0.1, color='#1D9E75')
    ax.set_xscale('log')
    ax.set_xlabel(f'{name} (log scale)')
    ax.set_ylabel(r'$\Delta$NLL')
    ax.set_title(f'Profile: {name}')
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.legend(fontsize=8)

fig.suptitle('Profile likelihood: practical identifiability of key parameters',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('results/plots/profile_likelihood.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved results/plots/profile_likelihood.png")
